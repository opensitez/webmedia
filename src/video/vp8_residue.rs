//! VP8 macroblock token decoding and inverse-transform reconstruction.

use super::backend::MediaDecodeError;
use super::vp8::{BoolDecoder, KeyFrameLayout, KeyMacroblockMode};
use super::vp8_inter::{InterFrameLayout, InterMacroblock};
use super::vp8_transform::{inverse_dct, inverse_dct_dc, inverse_walsh};

pub(super) struct ResidualMacroblock {
    pub(super) has_coefficients: bool,
    pub(super) y: [[i32; 16]; 16],
    pub(super) u: [[i32; 16]; 4],
    pub(super) v: [[i32; 16]; 4],
}

impl Default for ResidualMacroblock {
    fn default() -> Self {
        Self {
            has_coefficients: false,
            y: [[0; 16]; 16],
            u: [[0; 16]; 4],
            v: [[0; 16]; 4],
        }
    }
}

pub(super) struct ResidueDecoder<'a> {
    partitions: Vec<Option<BoolDecoder<'a>>>,
    quantizers: [[i32; 6]; 4],
    above_y2: Vec<bool>,
    above_y: Vec<bool>,
    above_u: Vec<bool>,
    above_v: Vec<bool>,
    left_y2: bool,
    left_y: [bool; 4],
    left_u: [bool; 2],
    left_v: [bool; 2],
}

pub(super) trait ResidueLayout<'a> {
    fn token_partitions(&self) -> &[&'a [u8]];
    fn macroblocks_wide(&self) -> usize;
    fn dequant_factors(&self, segment: u8) -> [i32; 6];
    fn decode_coeff_block(
        &self,
        partition: &mut BoolDecoder<'_>,
        plane: usize,
        neighbor_context: usize,
        first_coefficient: usize,
    ) -> Result<([i32; 16], bool), MediaDecodeError>;
}

impl<'a> ResidueLayout<'a> for KeyFrameLayout<'a> {
    fn token_partitions(&self) -> &[&'a [u8]] { &self.token_partitions }
    fn macroblocks_wide(&self) -> usize { self.macroblocks_wide() }
    fn dequant_factors(&self, segment: u8) -> [i32; 6] { self.dequant_factors(segment) }
    fn decode_coeff_block(&self, partition: &mut BoolDecoder<'_>, plane: usize, context: usize, first: usize) -> Result<([i32; 16], bool), MediaDecodeError> {
        self.decode_coeff_block(partition, plane, context, first)
    }
}

impl<'a> ResidueLayout<'a> for InterFrameLayout<'a> {
    fn token_partitions(&self) -> &[&'a [u8]] { &self.token_partitions }
    fn macroblocks_wide(&self) -> usize { self.mb_width }
    fn dequant_factors(&self, segment: u8) -> [i32; 6] { self.dequant_factors(segment) }
    fn decode_coeff_block(&self, partition: &mut BoolDecoder<'_>, plane: usize, context: usize, first: usize) -> Result<([i32; 16], bool), MediaDecodeError> {
        self.coeff_probs.decode_block(partition, plane, context, first)
    }
}

pub(super) trait ResidueMode {
    fn segment(&self) -> u8;
    fn skip_coefficients(&self) -> bool;
    fn has_y2(&self) -> bool;
}

impl ResidueMode for KeyMacroblockMode {
    fn segment(&self) -> u8 { self.segment }
    fn skip_coefficients(&self) -> bool { self.skip_coefficients }
    fn has_y2(&self) -> bool { self.luma != 4 }
}

impl ResidueMode for InterMacroblock {
    fn segment(&self) -> u8 { self.segment }
    fn skip_coefficients(&self) -> bool { self.skip_coefficients }
    fn has_y2(&self) -> bool { self.mode != 4 && self.mode != 9 }
}

impl<'a> ResidueDecoder<'a> {
    pub(super) fn new(layout: &impl ResidueLayout<'a>) -> Result<Self, MediaDecodeError> {
        let partitions = layout
            .token_partitions()
            .iter()
            .map(|bytes| {
                if bytes.len() >= 2 {
                    BoolDecoder::new(bytes).map(Some)
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let width = layout.macroblocks_wide();
        Ok(Self {
            partitions,
            quantizers: std::array::from_fn(|segment| layout.dequant_factors(segment as u8)),
            above_y2: vec![false; width],
            above_y: vec![false; width * 4],
            above_u: vec![false; width * 2],
            above_v: vec![false; width * 2],
            left_y2: false,
            left_y: [false; 4],
            left_u: [false; 2],
            left_v: [false; 2],
        })
    }

    pub(super) fn decode(
        &mut self,
        layout: &impl ResidueLayout<'a>,
        mode: &impl ResidueMode,
        x: usize,
        row: usize,
    ) -> Result<ResidualMacroblock, MediaDecodeError> {
        if x == 0 {
            self.left_y2 = false;
            self.left_y.fill(false);
            self.left_u.fill(false);
            self.left_v.fill(false);
        }
        let has_y2 = mode.has_y2();
        let mut result = ResidualMacroblock::default();
        if mode.skip_coefficients() {
            if has_y2 {
                self.above_y2[x] = false;
                self.left_y2 = false;
            }
            self.above_y[x * 4..x * 4 + 4].fill(false);
            self.above_u[x * 2..x * 2 + 2].fill(false);
            self.above_v[x * 2..x * 2 + 2].fill(false);
            self.left_y.fill(false);
            self.left_u.fill(false);
            self.left_v.fill(false);
            return Ok(result);
        }
        let partition_index = row % self.partitions.len();
        let partition = self.partitions[partition_index]
            .as_mut()
            .ok_or_else(|| MediaDecodeError::InvalidData("empty VP8 token partition".into()))?;
        let quant = self.quantizers[mode.segment() as usize];
        let mut y2_dc = [0; 16];
        if has_y2 {
            let context = usize::from(self.above_y2[x]) + usize::from(self.left_y2);
            let (mut block, present) = layout.decode_coeff_block(partition, 1, context, 0)?;
            result.has_coefficients |= present;
            self.above_y2[x] = present;
            self.left_y2 = present;
            if present {
                block[0] *= quant[2];
                for value in &mut block[1..] {
                    *value *= quant[3];
                }
                y2_dc = inverse_walsh(&block);
            }
        }
        for block_y in 0..4 {
            for block_x in 0..4 {
                let index = block_y * 4 + block_x;
                let above = &mut self.above_y[x * 4 + block_x];
                let left = &mut self.left_y[block_y];
                let context = usize::from(*above) + usize::from(*left);
                let (mut block, present) = layout.decode_coeff_block(
                    partition,
                    if has_y2 { 0 } else { 3 },
                    context,
                    usize::from(has_y2),
                )?;
                *above = present;
                *left = present;
                result.has_coefficients |= present;
                if !present {
                    if has_y2 && y2_dc[index] != 0 {
                        result.y[index] = inverse_dct_dc(y2_dc[index]);
                    }
                    continue;
                }
                if has_y2 {
                    block[0] = y2_dc[index];
                } else {
                    block[0] *= quant[0];
                }
                for value in &mut block[1..] {
                    *value *= quant[1];
                }
                result.y[index] = inverse_dct(&block);
            }
        }
        for (above, left, output) in [
            (&mut self.above_u, &mut self.left_u, &mut result.u),
            (&mut self.above_v, &mut self.left_v, &mut result.v),
        ] {
            for block_y in 0..2 {
                for block_x in 0..2 {
                    let index = block_y * 2 + block_x;
                    let above = &mut above[x * 2 + block_x];
                    let left = &mut left[block_y];
                    let context = usize::from(*above) + usize::from(*left);
                    let (mut block, present) =
                        layout.decode_coeff_block(partition, 2, context, 0)?;
                    *above = present;
                    *left = present;
                    result.has_coefficients |= present;
                    if !present { continue; }
                    block[0] *= quant[4];
                    for value in &mut block[1..] {
                        *value *= quant[5];
                    }
                    output[index] = inverse_dct(&block);
                }
            }
        }
        Ok(result)
    }
}
