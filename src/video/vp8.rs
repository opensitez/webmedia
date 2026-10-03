//! VP8 uncompressed frame header (RFC 6386, section 9.1).

use super::backend::MediaDecodeError;
use super::vp8_coeff::CoeffProbs;
use super::vp8_probs::KEYFRAME_BMODE_PROBS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    pub key_frame: bool,
    pub version: u8,
    pub show_frame: bool,
    pub first_partition_size: usize,
    pub width: Option<u16>,
    pub height: Option<u16>,
}

impl FrameHeader {
    pub fn parse(data: &[u8]) -> Result<Self, MediaDecodeError> {
        if data.len() < 3 {
            return Err(MediaDecodeError::InvalidData(
                "truncated VP8 frame tag".into(),
            ));
        }
        let tag = u32::from(data[0]) | (u32::from(data[1]) << 8) | (u32::from(data[2]) << 16);
        let key_frame = tag & 1 == 0;
        let header_len = if key_frame { 10 } else { 3 };
        if data.len() < header_len {
            return Err(MediaDecodeError::InvalidData(
                "truncated VP8 keyframe header".into(),
            ));
        }
        let first_partition_size = (tag >> 5) as usize;
        if first_partition_size > data.len() - header_len {
            return Err(MediaDecodeError::InvalidData(
                "VP8 first partition exceeds frame".into(),
            ));
        }
        let (width, height) = if key_frame {
            if data[3..6] != [0x9d, 0x01, 0x2a] {
                return Err(MediaDecodeError::InvalidData(
                    "invalid VP8 keyframe marker".into(),
                ));
            }
            let width = u16::from_le_bytes([data[6], data[7]]) & 0x3fff;
            let height = u16::from_le_bytes([data[8], data[9]]) & 0x3fff;
            if width == 0 || height == 0 {
                return Err(MediaDecodeError::InvalidData(
                    "zero-sized VP8 keyframe".into(),
                ));
            }
            (Some(width), Some(height))
        } else {
            (None, None)
        };
        Ok(Self {
            key_frame,
            version: ((tag >> 1) & 7) as u8,
            show_frame: tag & 0x10 != 0,
            first_partition_size,
            width,
            height,
        })
    }

    pub fn control_partition<'a>(&self, frame: &'a [u8]) -> &'a [u8] {
        let start = if self.key_frame { 10 } else { 3 };
        &frame[start..start + self.first_partition_size]
    }
}

/// Probability decoder for VP8 control and token partitions.
#[derive(Debug)]
pub struct BoolDecoder<'a> {
    data: &'a [u8],
    next_bit: usize,
    value: u32,
    range: u32,
}

#[derive(Debug)]
pub struct KeyFrameLayout<'a> {
    pub simple_filter: bool,
    pub loop_filter_level: u8,
    pub sharpness_level: u8,
    pub(super) refresh_entropy_probs: bool,
    segment_filter_levels: [i16; 4],
    filter_adjustments: bool,
    reference_filter_deltas: [i16; 4],
    mode_filter_deltas: [i16; 4],
    pub quantizer: u8,
    quant_deltas: [i8; 5],
    segment_quantizers: [i16; 4],
    segment_absolute: bool,
    pub token_partitions: Vec<&'a [u8]>,
    coeff_probs: CoeffProbs,
    control: BoolDecoder<'a>,
    segment_map_update: bool,
    segment_probs: [u8; 3],
    mb_no_skip_coeff: bool,
    prob_skip_false: u8,
    macroblocks_wide: usize,
    macroblocks_high: usize,
    next_macroblock: usize,
    above_b_modes: Vec<u8>,
    left_b_modes: [u8; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyMacroblockMode {
    pub segment: u8,
    pub skip_coefficients: bool,
    pub luma: u8,
    pub subblocks: [u8; 16],
    pub chroma: u8,
}

impl<'a> KeyFrameLayout<'a> {
    pub fn parse(frame: &'a [u8]) -> Result<Self, MediaDecodeError> {
        let header = FrameHeader::parse(frame)?;
        if !header.key_frame {
            return Err(MediaDecodeError::Unsupported);
        }
        let mut control = BoolDecoder::new(header.control_partition(frame))?;
        if control.read_bit()? {
            return Err(MediaDecodeError::Unsupported); // reserved color space
        }
        let _clamping_type = control.read_bit()?;
        let mut segment_probs = [255; 3];
        let mut segment_map_update = false;
        let mut segment_quantizers = [0; 4];
        let mut segment_filter_levels = [0; 4];
        let mut segment_absolute = false;
        if control.read_bit()? {
            let update_map = control.read_bit()?;
            segment_map_update = update_map;
            let update_features = control.read_bit()?;
            if update_features {
                segment_absolute = control.read_bit()?;
                for bits in [7, 6] {
                    for segment in 0..4 {
                        if control.read_bit()? {
                            let magnitude = control.read_literal(bits)? as i16;
                            let negative = control.read_bit()?;
                            if bits == 7 {
                                segment_quantizers[segment] =
                                    if negative { -magnitude } else { magnitude };
                            } else {
                                segment_filter_levels[segment] =
                                    if negative { -magnitude } else { magnitude };
                            }
                        }
                    }
                }
            }
            if update_map {
                for probability in &mut segment_probs {
                    if control.read_bit()? {
                        *probability = control.read_literal(8)? as u8;
                    }
                }
            }
        }
        let simple_filter = control.read_bit()?;
        let loop_filter_level = control.read_literal(6)? as u8;
        let sharpness_level = control.read_literal(3)? as u8;
        let filter_adjustments = control.read_bit()?;
        let mut reference_filter_deltas = [0; 4];
        let mut mode_filter_deltas = [0; 4];
        if filter_adjustments && control.read_bit()? {
            for deltas in [&mut reference_filter_deltas, &mut mode_filter_deltas] {
                for delta in deltas {
                    if control.read_bit()? {
                        let magnitude = control.read_literal(6)? as i16;
                        *delta = if control.read_bit()? {
                            -magnitude
                        } else {
                            magnitude
                        };
                    }
                }
            }
        }
        let count = 1usize << control.read_literal(2)?;
        let quantizer = control.read_literal(7)? as u8;
        let mut quant_deltas = [0; 5];
        for delta in &mut quant_deltas {
            if control.read_bit()? {
                let magnitude = control.read_literal(4)? as i8;
                *delta = if control.read_bit()? {
                    -magnitude
                } else {
                    magnitude
                };
            }
        }
        let refresh_entropy_probs = control.read_bit()?;
        let mut coeff_probs = CoeffProbs::default();
        coeff_probs.update(&mut control)?;
        let mb_no_skip_coeff = control.read_bit()?;
        let prob_skip_false = if mb_no_skip_coeff {
            control.read_literal(8)? as u8
        } else {
            0
        };
        let mut pos = 10 + header.first_partition_size;
        let table_len = (count - 1) * 3;
        if frame.len() < pos + table_len {
            return Err(MediaDecodeError::InvalidData(
                "truncated VP8 partition sizes".into(),
            ));
        }
        let table = &frame[pos..pos + table_len];
        pos += table_len;
        let mut token_partitions = Vec::with_capacity(count);
        for entry in table.chunks_exact(3) {
            let length =
                usize::from(entry[0]) | usize::from(entry[1]) << 8 | usize::from(entry[2]) << 16;
            let end = pos
                .checked_add(length)
                .filter(|end| *end <= frame.len())
                .ok_or_else(|| {
                    MediaDecodeError::InvalidData("VP8 token partition exceeds frame".into())
                })?;
            token_partitions.push(&frame[pos..end]);
            pos = end;
        }
        token_partitions.push(&frame[pos..]);
        let macroblocks_wide = usize::from(header.width.unwrap()).div_ceil(16);
        let macroblocks_high = usize::from(header.height.unwrap()).div_ceil(16);
        Ok(Self {
            simple_filter,
            loop_filter_level,
            sharpness_level,
            refresh_entropy_probs,
            segment_filter_levels,
            filter_adjustments,
            reference_filter_deltas,
            mode_filter_deltas,
            quantizer,
            quant_deltas,
            segment_quantizers,
            segment_absolute,
            token_partitions,
            coeff_probs,
            control,
            segment_map_update,
            segment_probs,
            mb_no_skip_coeff,
            prob_skip_false,
            macroblocks_wide,
            macroblocks_high,
            next_macroblock: 0,
            above_b_modes: vec![0; macroblocks_wide * 4],
            left_b_modes: [0; 4],
        })
    }

    pub fn next_macroblock_mode(&mut self) -> Result<Option<KeyMacroblockMode>, MediaDecodeError> {
        if self.next_macroblock == self.macroblocks_wide * self.macroblocks_high {
            return Ok(None);
        }
        let x = self.next_macroblock % self.macroblocks_wide;
        if x == 0 {
            self.left_b_modes = [0; 4];
        }
        let segment = if self.segment_map_update {
            if self.control.read(self.segment_probs[0])? {
                if self.control.read(self.segment_probs[2])? {
                    3
                } else {
                    2
                }
            } else if self.control.read(self.segment_probs[1])? {
                1
            } else {
                0
            }
        } else {
            0
        };
        let skip_coefficients = self.mb_no_skip_coeff && self.control.read(self.prob_skip_false)?;
        let luma = if !self.control.read(145)? {
            4
        } else if !self.control.read(156)? {
            if self.control.read(163)? {
                1
            } else {
                0
            }
        } else if self.control.read(128)? {
            3
        } else {
            2
        };
        let mut subblocks = [0; 16];
        if luma == 4 {
            for row in 0..4 {
                for col in 0..4 {
                    let above = self.above_b_modes[x * 4 + col];
                    let left = self.left_b_modes[row];
                    let probs: &[u8; 9] =
                        (&KEYFRAME_BMODE_PROBS[((usize::from(above) * 10 + usize::from(left)) * 9)
                            ..((usize::from(above) * 10 + usize::from(left)) * 9 + 9)])
                            .try_into()
                            .unwrap();
                    let mode = read_bmode(&mut self.control, probs)?;
                    subblocks[row * 4 + col] = mode;
                    self.above_b_modes[x * 4 + col] = mode;
                    self.left_b_modes[row] = mode;
                }
            }
        } else {
            let mode = [0, 2, 3, 1][luma as usize];
            subblocks.fill(mode);
            self.above_b_modes[x * 4..x * 4 + 4].fill(mode);
            self.left_b_modes.fill(mode);
        }
        let chroma = if !self.control.read(142)? {
            0
        } else if !self.control.read(114)? {
            1
        } else if !self.control.read(183)? {
            2
        } else {
            3
        };
        self.next_macroblock += 1;
        Ok(Some(KeyMacroblockMode {
            segment,
            skip_coefficients,
            luma,
            subblocks,
            chroma,
        }))
    }

    pub fn decode_coeff_block(
        &self,
        partition: &mut BoolDecoder<'_>,
        plane: usize,
        neighbor_context: usize,
        first_coefficient: usize,
    ) -> Result<([i32; 16], bool), MediaDecodeError> {
        self.coeff_probs
            .decode_block(partition, plane, neighbor_context, first_coefficient)
    }

    pub fn control_decoder(&mut self) -> &mut BoolDecoder<'a> {
        &mut self.control
    }

    pub fn dequant_factors(&self, segment: u8) -> [i32; 6] {
        let base = if self.segment_absolute {
            self.segment_quantizers[segment as usize]
        } else {
            i16::from(self.quantizer) + self.segment_quantizers[segment as usize]
        };
        super::vp8_quant::factors(base, self.quant_deltas)
    }

    pub fn macroblocks_wide(&self) -> usize {
        self.macroblocks_wide
    }

    pub(super) fn macroblocks_high(&self) -> usize {
        self.macroblocks_high
    }

    pub(super) fn coeff_probabilities(&self) -> CoeffProbs {
        self.coeff_probs.clone()
    }

    pub(super) fn segment_features(&self) -> ([i16; 4], [i16; 4], bool) {
        (self.segment_quantizers, self.segment_filter_levels, self.segment_absolute)
    }

    pub(super) fn filter_deltas(&self) -> ([i16; 4], [i16; 4]) {
        (self.reference_filter_deltas, self.mode_filter_deltas)
    }

    pub fn filter_level(&self, segment: u8, luma_mode: u8) -> u8 {
        if self.loop_filter_level == 0 { return 0; }
        let base = if self.segment_absolute {
            self.segment_filter_levels[segment as usize]
        } else {
            i16::from(self.loop_filter_level) + self.segment_filter_levels[segment as usize]
        }.clamp(0, 63);
        let level = if self.filter_adjustments {
            base + self.reference_filter_deltas[0]
                + if luma_mode == 4 { self.mode_filter_deltas[0] } else { 0 }
        } else {
            base
        };
        level.clamp(0, 63) as u8
    }
}

pub(super) fn read_bmode(decoder: &mut BoolDecoder<'_>, p: &[u8; 9]) -> Result<u8, MediaDecodeError> {
    if !decoder.read(p[0])? {
        return Ok(0);
    }
    if !decoder.read(p[1])? {
        return Ok(1);
    }
    if !decoder.read(p[2])? {
        return Ok(2);
    }
    if !decoder.read(p[3])? {
        if !decoder.read(p[4])? {
            return Ok(3);
        }
        return Ok(if decoder.read(p[5])? { 6 } else { 5 });
    }
    if !decoder.read(p[6])? {
        return Ok(4);
    }
    if !decoder.read(p[7])? {
        return Ok(7);
    }
    Ok(if decoder.read(p[8])? { 9 } else { 8 })
}

impl<'a> BoolDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self, MediaDecodeError> {
        if data.len() < 2 {
            return Err(MediaDecodeError::InvalidData(
                "truncated VP8 partition".into(),
            ));
        }
        Ok(Self {
            data,
            next_bit: 16,
            value: u32::from(u16::from_be_bytes([data[0], data[1]])),
            range: 255,
        })
    }

    #[inline]
    pub fn read(&mut self, probability: u8) -> Result<bool, MediaDecodeError> {
        let split = 1 + ((self.range - 1) * u32::from(probability) >> 8);
        let threshold = split << 8;
        let bit = self.value >= threshold;
        if bit {
            self.value -= threshold;
            self.range -= split;
        } else {
            self.range = split;
        }
        let shift = self.range.leading_zeros() - 24;
        if shift != 0 {
            let byte = self.next_bit / 8;
            if self.next_bit + shift as usize > self.data.len() * 8 + 16 {
                return Err(MediaDecodeError::InvalidData(
                    "truncated VP8 arithmetic code".into(),
                ));
            }
            let window = u32::from(self.data.get(byte).copied().unwrap_or(0)) << 8
                | u32::from(self.data.get(byte + 1).copied().unwrap_or(0));
            let next = (window >> (16 - self.next_bit % 8 - shift as usize))
                & ((1 << shift) - 1);
            self.range <<= shift;
            self.value = (self.value << shift) | next;
            self.next_bit += shift as usize;
        }
        Ok(bit)
    }

    pub fn read_bit(&mut self) -> Result<bool, MediaDecodeError> {
        self.read(128)
    }

    pub fn read_literal(&mut self, bits: usize) -> Result<u32, MediaDecodeError> {
        let mut value = 0;
        for _ in 0..bits {
            value = (value << 1) | u32::from(self.read_bit()?);
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bulk_arithmetic_refill_matches_bitwise_reader() {
        for length in [2, 3, 4, 7, 8, 9, 17, 31, 32, 33, 128, 257, 1024] {
            let mut seed = 31u32;
            let data: Vec<u8> = (0..length).map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 24) as u8
            }).collect();
            let mut bulk = BoolDecoder::new(&data).unwrap();
            let mut scalar = BoolDecoder::new(&data).unwrap();
            for i in 0..10000 {
                let probability = (i * 73) as u8;
                let split = 1 + ((scalar.range - 1) * u32::from(probability) >> 8);
                let threshold = split << 8;
                let expected = scalar.value >= threshold;
                if expected {
                    scalar.value -= threshold;
                    scalar.range -= split;
                } else {
                    scalar.range = split;
                }
                let mut truncated = false;
                while scalar.range < 128 {
                    scalar.range <<= 1;
                    if scalar.next_bit >= data.len() * 8 + 16 {
                        truncated = true;
                        break;
                    }
                    let next = data.get(scalar.next_bit / 8)
                        .map_or(0, |byte| (byte >> (7 - scalar.next_bit % 8)) & 1);
                    scalar.value = (scalar.value << 1) | u32::from(next);
                    scalar.next_bit += 1;
                }
                let result = bulk.read(probability);
                if truncated {
                    assert!(result.is_err());
                    break;
                }
                assert_eq!(result.unwrap(), expected);
                assert_eq!((bulk.range, bulk.value, bulk.next_bit),
                    (scalar.range, scalar.value, scalar.next_bit));
            }
        }
    }

    #[test]
    fn parses_keyframe_dimensions_and_rejects_bad_marker() {
        let frame = [0x30, 0, 0, 0x9d, 1, 0x2a, 0x80, 0x02, 0x68, 0x01, 0];
        let header = FrameHeader::parse(&frame).unwrap();
        assert_eq!((header.width, header.height), (Some(640), Some(360)));
        assert!(header.key_frame && header.show_frame);
        let mut bad = frame;
        bad[3] = 0;
        assert!(FrameHeader::parse(&bad).is_err());
    }
}
