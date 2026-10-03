//! Backward VP9 probability updates from decoded tile syntax.

use super::vp9_compressed::CompressedHeader;

#[derive(Clone)]
pub(super) struct CoefficientCounts {
    token: [[[[[[u32; 3]; 6]; 6]; 2]; 2]; 4],
    more: [[[[[[u32; 2]; 6]; 6]; 2]; 2]; 4],
}

impl Default for CoefficientCounts {
    fn default() -> Self {
        Self {
            token: [[[[[[0; 3]; 6]; 6]; 2]; 2]; 4],
            more: [[[[[[0; 2]; 6]; 6]; 2]; 2]; 4],
        }
    }
}

impl CoefficientCounts {
    pub(super) fn record_more(
        &mut self, tx: usize, plane: usize, reference: usize, band: usize,
        context: usize, more: bool,
    ) {
        self.more[tx][plane][reference][band][context][usize::from(more)] += 1;
    }

    pub(super) fn record_token(
        &mut self, tx: usize, plane: usize, reference: usize, band: usize,
        context: usize, token: u8,
    ) {
        self.token[tx][plane][reference][band][context][usize::from(token.min(2))] += 1;
    }

    pub(super) fn adapt(&self, probabilities: &mut CompressedHeader, factor: u32) {
        for tx in 0..4 {
            for plane in 0..2 {
                for reference in 0..2 {
                    for band in 0..6 {
                        for context in 0..if band == 0 { 3 } else { 6 } {
                            let counts = self.token[tx][plane][reference][band][context];
                            let more = self.more[tx][plane][reference][band][context];
                            let probs = &mut probabilities.coef_probs[tx][plane][reference][band][context];
                            probs[1] = merge_prob(probs[1], counts[0], counts[1] + counts[2], 24, factor);
                            probs[2] = merge_prob(probs[2], counts[1], counts[2], 24, factor);
                            probs[0] = merge_prob(probs[0], more[0], more[1], 24, factor);
                        }
                    }
                }
            }
        }
    }
}

pub(super) fn merge_prob(
    previous: u8, zero: u32, one: u32, saturation: u32, maximum_factor: u32,
) -> u8 {
    let total = zero + one;
    let observed = if total == 0 {
        128
    } else {
        ((zero * 256 + total / 2) / total).clamp(1, 255)
    };
    let factor = maximum_factor * total.min(saturation) / saturation;
    ((u32::from(previous) * (256 - factor) + observed * factor + 128) >> 8) as u8
}

#[derive(Clone, Default)]
pub(super) struct NonCoefficientCounts {
    pub skip: [[u32; 2]; 3],
    pub is_inter: [[u32; 2]; 4],
    pub comp_mode: [[u32; 2]; 5],
    pub single_ref: [[[u32; 2]; 2]; 5],
    pub comp_ref: [[u32; 2]; 5],
    pub partition: [[u32; 4]; 16],
    pub inter_mode: [[u32; 4]; 7],
    pub interp_filter: [[u32; 3]; 4],
    pub y_mode: [[u32; 10]; 4],
    pub uv_mode: [[u32; 10]; 10],
    pub tx_size: [[[u32; 4]; 2]; 4],
    pub mv_joint: [u32; 4],
    pub mv_sign: [[u32; 2]; 2],
    pub mv_class: [[u32; 11]; 2],
    pub mv_class0_bit: [[u32; 2]; 2],
    pub mv_bits: [[[u32; 2]; 10]; 2],
    pub mv_class0_fr: [[[u32; 4]; 2]; 2],
    pub mv_fr: [[u32; 4]; 2],
    pub mv_class0_hp: [[u32; 2]; 2],
    pub mv_hp: [[u32; 2]; 2],
}

impl NonCoefficientCounts {
    pub fn adapt(&self, probabilities: &mut CompressedHeader, switchable_filter: bool, high_precision: bool) {
        const PARTITION: [i8; 6] = [0, 2, -1, 4, -2, -3];
        const INTER_MODE: [i8; 6] = [-2, 2, 0, 4, -1, -3];
        const INTERP: [i8; 4] = [0, 2, -1, -2];
        const INTRA: [i8; 18] = [0, 2, -9, 4, -1, 6, 8, 12, -2, 10, -4, -5, -3, 14, -8, 16, -6, -7];
        const MV_CLASS: [i8; 20] = [0, 2, -1, 4, 6, 8, -2, -3, 10, 12, -4, -5, -6, 14, 16, 18, -7, -8, -9, -10];
        let inter = &mut probabilities.inter_probs;
        for context in 0..3 {
            adapt_binary(&mut probabilities.skip_probs[context], self.skip[context]);
        }
        for context in 0..4 {
            adapt_binary(&mut inter.is_inter[context], self.is_inter[context]);
            adapt_tree(&PARTITION, &mut inter.partition[context], &self.partition[context]);
            adapt_tree(&INTRA, &mut inter.y_mode[context], &self.y_mode[context]);
            if switchable_filter {
                adapt_tree(&INTERP, &mut inter.interp_filter[context], &self.interp_filter[context]);
            }
        }
        for context in 4..16 {
            adapt_tree(&PARTITION, &mut inter.partition[context], &self.partition[context]);
        }
        for mode in 0..10 {
            adapt_tree(&INTRA, &mut inter.uv_mode[mode], &self.uv_mode[mode]);
        }
        for context in 0..5 {
            adapt_binary(&mut inter.comp_mode[context], self.comp_mode[context]);
            adapt_binary(&mut inter.comp_ref[context], self.comp_ref[context]);
            for node in 0..2 {
                adapt_binary(&mut inter.single_ref[context][node], self.single_ref[context][node]);
            }
        }
        for context in 0..7 {
            adapt_tree(&INTER_MODE, &mut inter.inter_mode[context], &self.inter_mode[context]);
        }
        if probabilities.tx_mode == 4 {
            for context in 0..2 {
                adapt_tree(&[0, -1], &mut inter.tx_8x8[context], &self.tx_size[1][context][..2]);
                adapt_tree(&[0, 2, -1, -2], &mut inter.tx_16x16[context], &self.tx_size[2][context][..3]);
                adapt_tree(&PARTITION, &mut inter.tx_32x32[context], &self.tx_size[3][context]);
            }
        }
        adapt_tree(&PARTITION, &mut inter.mv_joint, &self.mv_joint);
        for axis in 0..2 {
            adapt_binary(&mut inter.mv_sign[axis], self.mv_sign[axis]);
            adapt_tree(&MV_CLASS, &mut inter.mv_class[axis], &self.mv_class[axis]);
            adapt_binary(&mut inter.mv_class0_bit[axis], self.mv_class0_bit[axis]);
            for index in 0..10 {
                adapt_binary(&mut inter.mv_bits[axis][index], self.mv_bits[axis][index]);
            }
            for index in 0..2 {
                adapt_tree(&PARTITION, &mut inter.mv_class0_fr[axis][index], &self.mv_class0_fr[axis][index]);
            }
            adapt_tree(&PARTITION, &mut inter.mv_fr[axis], &self.mv_fr[axis]);
            if high_precision {
                adapt_binary(&mut inter.mv_class0_hp[axis], self.mv_class0_hp[axis]);
                adapt_binary(&mut inter.mv_hp[axis], self.mv_hp[axis]);
            }
        }
    }
}

fn adapt_binary(probability: &mut u8, counts: [u32; 2]) {
    *probability = merge_prob(*probability, counts[0], counts[1], 20, 128);
}

fn adapt_tree(tree: &[i8], probabilities: &mut [u8], counts: &[u32]) {
    fn visit(tree: &[i8], node: usize, probabilities: &mut [u8], counts: &[u32]) -> u32 {
        let branch_count = |branch: i8, probabilities: &mut [u8]| {
            if branch <= 0 { counts[(-branch) as usize] }
            else { visit(tree, branch as usize, probabilities, counts) }
        };
        let zero = branch_count(tree[node], probabilities);
        let one = branch_count(tree[node + 1], probabilities);
        probabilities[node / 2] = merge_prob(probabilities[node / 2], zero, one, 20, 128);
        zero + one
    }
    visit(tree, 0, probabilities, counts);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probability_merge_respects_empty_and_saturated_counts() {
        assert_eq!(merge_prob(77, 0, 0, 24, 112), 77);
        assert_eq!(merge_prob(128, 24, 0, 24, 128), 192);
        assert_eq!(merge_prob(128, 0, 24, 24, 128), 65);
    }
}
