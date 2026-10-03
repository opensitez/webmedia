//! VP9 interframe probability tables and compressed-header updates (spec sections 6.3 and 10.5).

use super::backend::MediaDecodeError;
use super::vp8::BoolDecoder;
use super::vp9::InterframeHeader;
use super::vp9_compressed::{inverse_remap_probability, read_probability_update};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InterframeProbabilities {
    pub tx_8x8: [[u8; 1]; 2],
    pub tx_16x16: [[u8; 2]; 2],
    pub tx_32x32: [[u8; 3]; 2],
    pub inter_mode: [[u8; 3]; 7],
    pub interp_filter: [[u8; 2]; 4],
    pub is_inter: [u8; 4],
    pub comp_mode: [u8; 5],
    pub single_ref: [[u8; 2]; 5],
    pub comp_ref: [u8; 5],
    pub y_mode: [[u8; 9]; 4],
    pub uv_mode: [[u8; 9]; 10],
    pub partition: [[u8; 3]; 16],
    pub mv_joint: [u8; 3],
    pub mv_sign: [u8; 2],
    pub mv_class: [[u8; 10]; 2],
    pub mv_class0_bit: [u8; 2],
    pub mv_bits: [[u8; 10]; 2],
    pub mv_class0_fr: [[[u8; 3]; 2]; 2],
    pub mv_fr: [[u8; 3]; 2],
    pub mv_class0_hp: [u8; 2],
    pub mv_hp: [u8; 2],
}

impl Default for InterframeProbabilities {
    fn default() -> Self {
        Self {
            tx_8x8: [[100], [66]],
            tx_16x16: [[20, 152], [15, 101]],
            tx_32x32: [[3, 136, 37], [5, 52, 13]],
            inter_mode: [
                [2, 173, 34], [7, 145, 85], [7, 166, 63], [7, 94, 66],
                [8, 64, 46], [17, 81, 31], [25, 29, 30],
            ],
            interp_filter: [[235, 162], [36, 255], [34, 3], [149, 144]],
            is_inter: [9, 102, 187, 225],
            comp_mode: [239, 183, 119, 96, 41],
            single_ref: [[33, 16], [77, 74], [142, 142], [172, 170], [238, 247]],
            comp_ref: [50, 126, 123, 221, 226],
            y_mode: [
                [65, 32, 18, 144, 162, 194, 41, 51, 98],
                [132, 68, 18, 165, 217, 196, 45, 40, 78],
                [173, 80, 19, 176, 240, 193, 64, 35, 46],
                [221, 135, 38, 194, 248, 121, 96, 85, 29],
            ],
            uv_mode: super::vp9_tile::INTERFRAME_UV_MODE_PROBS,
            partition: [
                [199, 122, 141], [147, 63, 159], [148, 133, 118], [121, 104, 114],
                [174, 73, 87], [92, 41, 83], [82, 99, 50], [53, 39, 39],
                [177, 58, 59], [68, 26, 63], [52, 79, 25], [17, 14, 12],
                [222, 34, 30], [72, 16, 44], [58, 32, 12], [10, 7, 6],
            ],
            mv_joint: [32, 64, 96],
            mv_sign: [128; 2],
            mv_class: [
                [224, 144, 192, 168, 192, 176, 192, 198, 198, 245],
                [216, 128, 176, 160, 176, 176, 192, 198, 198, 208],
            ],
            mv_class0_bit: [216, 208],
            mv_bits: [[136, 140, 148, 160, 176, 192, 224, 234, 234, 240]; 2],
            mv_class0_fr: [
                [[128, 128, 64], [96, 112, 64]],
                [[128, 128, 64], [96, 112, 64]],
            ],
            mv_fr: [[64, 96, 64]; 2],
            mv_class0_hp: [160; 2],
            mv_hp: [128; 2],
        }
    }
}

impl InterframeProbabilities {
    pub fn update_tx(
        &mut self, bits: &mut BoolDecoder<'_>, updates: &mut usize,
    ) -> Result<(), MediaDecodeError> {
        for row in &mut self.tx_8x8 {
            update_slice(bits, row, updates)?;
        }
        for row in &mut self.tx_16x16 {
            update_slice(bits, row, updates)?;
        }
        for row in &mut self.tx_32x32 {
            update_slice(bits, row, updates)?;
        }
        Ok(())
    }

    pub fn update_noncoef(
        &mut self,
        bits: &mut BoolDecoder<'_>,
        frame: &InterframeHeader,
        updates: &mut usize,
    ) -> Result<u8, MediaDecodeError> {
        for row in &mut self.inter_mode {
            update_slice(bits, row, updates)?;
        }
        if frame.interpolation_filter == 4 {
            for row in &mut self.interp_filter {
                update_slice(bits, row, updates)?;
            }
        }
        update_slice(bits, &mut self.is_inter, updates)?;
        let compound_allowed = frame.reference_sign_bias[1..]
            .iter()
            .any(|bias| *bias != frame.reference_sign_bias[0]);
        let reference_mode = if compound_allowed && bits.read_bit()? {
            if bits.read_bit()? { 2 } else { 1 }
        } else {
            0
        };
        if reference_mode == 2 {
            update_slice(bits, &mut self.comp_mode, updates)?;
        }
        if reference_mode != 1 {
            for row in &mut self.single_ref {
                update_slice(bits, row, updates)?;
            }
        }
        if reference_mode != 0 {
            update_slice(bits, &mut self.comp_ref, updates)?;
        }
        for row in &mut self.y_mode {
            update_slice(bits, row, updates)?;
        }
        for row in &mut self.partition {
            update_slice(bits, row, updates)?;
        }
        update_mv_slice(bits, &mut self.mv_joint, updates)?;
        for axis in 0..2 {
            update_mv_slice(bits, &mut self.mv_sign[axis..axis + 1], updates)?;
            update_mv_slice(bits, &mut self.mv_class[axis], updates)?;
            update_mv_slice(bits, &mut self.mv_class0_bit[axis..axis + 1], updates)?;
            update_mv_slice(bits, &mut self.mv_bits[axis], updates)?;
        }
        for axis in 0..2 {
            for row in &mut self.mv_class0_fr[axis] {
                update_mv_slice(bits, row, updates)?;
            }
            update_mv_slice(bits, &mut self.mv_fr[axis], updates)?;
        }
        if frame.allow_high_precision_mv {
            for axis in 0..2 {
                update_mv_slice(bits, &mut self.mv_class0_hp[axis..axis + 1], updates)?;
                update_mv_slice(bits, &mut self.mv_hp[axis..axis + 1], updates)?;
            }
        }
        Ok(reference_mode)
    }
}

fn update_slice(
    bits: &mut BoolDecoder<'_>, probabilities: &mut [u8], updates: &mut usize,
) -> Result<(), MediaDecodeError> {
    for probability in probabilities {
        if let Some(delta) = read_probability_update(bits)? {
            *probability = inverse_remap_probability(delta, *probability);
            *updates += 1;
        }
    }
    Ok(())
}

fn update_mv_slice(
    bits: &mut BoolDecoder<'_>, probabilities: &mut [u8], updates: &mut usize,
) -> Result<(), MediaDecodeError> {
    for probability in probabilities {
        if bits.read(252)? {
            *probability = ((bits.read_literal(7)? as u8) << 1) | 1;
            *updates += 1;
        }
    }
    Ok(())
}
