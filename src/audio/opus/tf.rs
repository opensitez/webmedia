//! CELT TF-choice decoding and exact adjustment tables from RFC 6716
//! section 4.3.4.5, Tables 60, 61, 62, and 63. No reference-only table needed.

use super::{celt::Layout, invalid, range::RangeDecoder};
use crate::video::backend::MediaDecodeError;

// [transient][tf_select][LM][choice].
const ADJUSTMENTS: [[[[i8; 2]; 4]; 2]; 2] = [
    [
        [[0, -1], [0, -1], [0, -2], [0, -2]],
        [[0, -1], [0, -2], [0, -3], [0, -3]],
    ],
    [
        [[0, -1], [1, 0], [2, 0], [3, 0]],
        [[0, -1], [1, -1], [1, -1], [1, -1]],
    ],
];

/// Inverse band transform in RFC 6716 section 4.3.4.5, final paragraph.
/// Both input and output use frequency-major, time-block-minor storage.
/// This implements the specified normalized Hadamard math; packet-level
/// interoperability still depends on the missing allocation/split decisions.
#[derive(Clone, Copy, Debug)]
pub struct BandTransform {
    width: usize,
    original_blocks: usize,
    coded_blocks: usize,
    adjustment: i8,
}

impl BandTransform {
    pub fn new(
        layout: Layout,
        band: usize,
        transient: bool,
        adjustment: i8,
    ) -> Result<Self, MediaDecodeError> {
        let original_blocks = layout.blocks(transient)?;
        let width = layout.band(band)?.len();
        let lm = (layout.samples() / 120).trailing_zeros() as usize;
        if !ADJUSTMENTS[usize::from(transient)]
            .iter()
            .any(|table| table[lm].contains(&adjustment))
        {
            return Err(invalid("invalid CELT band TF adjustment"));
        }
        let factor = 1usize << adjustment.unsigned_abs();
        let coded_blocks = if adjustment >= 0 {
            if original_blocks % factor != 0 {
                return Err(invalid("CELT TF exceeds available time blocks"));
            }
            original_blocks / factor
        } else {
            original_blocks * factor
        };
        if width % coded_blocks != 0 {
            // The prose does not define narrow-band omission rules. Do not
            // invent a partial transform when a full level does not fit.
            return Err(MediaDecodeError::Unsupported);
        }
        Ok(Self {
            width,
            original_blocks,
            coded_blocks,
            adjustment,
        })
    }

    /// Spreading precedes the inverse TF transform and uses coded time blocks.
    pub fn coded_blocks(self) -> usize {
        self.coded_blocks
    }

    pub fn inverse(self, vector: &mut [f64]) -> Result<(), MediaDecodeError> {
        if vector.len() != self.width || vector.iter().any(|value| !value.is_finite()) {
            return Err(invalid("invalid CELT inverse TF vector"));
        }
        if self.adjustment == 0 {
            return Ok(());
        }
        let factor = 1usize << self.adjustment.unsigned_abs();
        let gain = (factor as f64).sqrt().recip();
        let mut output = vec![0.0; self.width];
        if self.adjustment > 0 {
            // Undo frequency refinement across the original short MDCTs.
            for frequency in 0..self.width / self.original_blocks {
                for block in 0..self.coded_blocks {
                    let mut group = [0.0; 8];
                    for (k, value) in group[..factor].iter_mut().enumerate() {
                        *value = vector[(frequency * factor + k) * self.coded_blocks + block];
                    }
                    hadamard(&mut group[..factor]);
                    for (k, &value) in group[..factor].iter().enumerate() {
                        output[frequency * self.original_blocks + block * factor + k] =
                            value * gain;
                    }
                }
            }
        } else {
            // Time-refined input is in sequency order, not Sylvester order.
            // H_sequency = P*H, so the inverse is H*P^-1, not P*H.
            for frequency in 0..self.width / self.coded_blocks {
                for block in 0..self.original_blocks {
                    let mut group = [0.0; 8];
                    for time in 0..factor {
                        let natural = sequency_index(time, factor.trailing_zeros());
                        group[natural] =
                            vector[frequency * self.coded_blocks + block * factor + time];
                    }
                    hadamard(&mut group[..factor]);
                    for (k, &value) in group[..factor].iter().enumerate() {
                        output[(frequency * factor + k) * self.original_blocks + block] =
                            value * gain;
                    }
                }
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(invalid("CELT inverse TF overflow"));
        }
        vector.copy_from_slice(&output);
        Ok(())
    }
}

fn hadamard(vector: &mut [f64]) {
    let mut half = 1;
    while half < vector.len() {
        for group in vector.chunks_exact_mut(2 * half) {
            for index in 0..half {
                let (a, b) = (group[index], group[index + half]);
                group[index] = a + b;
                group[index + half] = a - b;
            }
        }
        half *= 2;
    }
}

fn sequency_index(index: usize, bits: u32) -> usize {
    let gray = index ^ (index >> 1);
    gray.reverse_bits() >> (usize::BITS - bits)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TfResolution {
    pub selected: bool,
    pub choices: Vec<bool>,
    pub adjustments: Vec<i8>,
}

impl TfResolution {
    /// Invoke after coarse energy, before spread and allocation. A conservative
    /// whole-sequence budget guard intentionally rejects small-budget cases
    /// until their precise omission/reservation rules are independently known.
    /// Failure does not consume entropy.
    pub fn decode(
        decoder: &mut RangeDecoder<'_>,
        layout: Layout,
        transient: bool,
    ) -> Result<Self, MediaDecodeError> {
        layout.blocks(transient)?;
        let lm = (layout.samples() / 120).trailing_zeros() as usize;
        let bands = layout.bands().len();
        let first_logp = if transient { 2 } else { 4 };
        let later_logp = if transient { 4 } else { 5 };
        let selection_possible = ADJUSTMENTS[usize::from(transient)][0][lm]
            != ADJUSTMENTS[usize::from(transient)][1][lm];
        let maximum_bits = first_logp + (bands - 1) * later_logp + usize::from(selection_possible);
        if decoder.tell().saturating_add(maximum_bits as u64) > decoder.frame_bytes() as u64 * 8 {
            return Err(MediaDecodeError::Unsupported);
        }
        let mut cursor = decoder.clone();
        let mut choices = Vec::with_capacity(bands);
        let mut choice = false;
        for index in 0..bands {
            choice ^= cursor.bit(if index == 0 {
                first_logp as u8
            } else {
                later_logp as u8
            })?;
            choices.push(choice);
        }
        let selection_matters = choices.iter().any(|&choice| {
            ADJUSTMENTS[usize::from(transient)][0][lm][usize::from(choice)]
                != ADJUSTMENTS[usize::from(transient)][1][lm][usize::from(choice)]
        });
        let selected = selection_matters && cursor.bit(1)?;
        let adjustments = choices
            .iter()
            .map(|&choice| {
                ADJUSTMENTS[usize::from(transient)][usize::from(selected)][lm][usize::from(choice)]
            })
            .collect();
        *decoder = cursor;
        Ok(Self {
            selected,
            choices,
            adjustments,
        })
    }

    pub fn adjustment(
        lm: u8,
        transient: bool,
        selected: bool,
        choice: bool,
    ) -> Result<i8, MediaDecodeError> {
        if lm > 3 {
            return Err(invalid("invalid CELT TF frame duration"));
        }
        Ok(
            ADJUSTMENTS[usize::from(transient)][usize::from(selected)][usize::from(lm)]
                [usize::from(choice)],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequency_rows_have_exactly_the_requested_number_of_sign_changes() {
        for bits in 1..=3 {
            let size = 1 << bits;
            for row in 0..size {
                let natural = sequency_index(row, bits);
                let signs: Vec<_> = (0..size)
                    .map(|column| (natural & column).count_ones() & 1)
                    .collect();
                assert_eq!(
                    signs.windows(2).filter(|pair| pair[0] != pair[1]).count(),
                    row
                );
            }
        }
    }

    #[test]
    fn inverse_tf_matches_direct_hadamard_matrix_and_preserves_energy() {
        let layout = Layout::from_configuration(31).unwrap().unwrap();
        for (transient, adjustment) in [(true, 1), (true, 3), (true, -1), (false, -2), (false, -3)]
        {
            let transform = BandTransform::new(layout, 20, transient, adjustment).unwrap();
            let input: Vec<_> = (0..transform.width)
                .map(|i| (i % 17) as f64 - 8.0)
                .collect();
            let mut actual = input.clone();
            transform.inverse(&mut actual).unwrap();
            let factor = 1usize << adjustment.unsigned_abs();
            let mut expected = vec![0.0; actual.len()];
            let scale = (factor as f64).sqrt().recip();
            if adjustment > 0 {
                for frequency in 0..actual.len() / transform.original_blocks {
                    for block in 0..transform.coded_blocks {
                        for time in 0..factor {
                            let value = (0..factor)
                                .map(|column| {
                                    let sign = if (time & column).count_ones() & 1 == 0 {
                                        1.0
                                    } else {
                                        -1.0
                                    };
                                    sign * input[(frequency * factor + column)
                                        * transform.coded_blocks
                                        + block]
                                })
                                .sum::<f64>();
                            expected
                                [frequency * transform.original_blocks + block * factor + time] =
                                scale * value;
                        }
                    }
                }
            } else {
                // Derive the sequency permutation by sorting Sylvester rows
                // by sign changes, independent of the Gray-code implementation.
                let mut rows: Vec<_> = (0..factor).collect();
                rows.sort_by_key(|&row| {
                    (0..factor - 1)
                        .filter(|&i| (row & i).count_ones() & 1 != (row & (i + 1)).count_ones() & 1)
                        .count()
                });
                for frequency in 0..actual.len() / transform.coded_blocks {
                    for block in 0..transform.original_blocks {
                        for bin in 0..factor {
                            let value = rows
                                .iter()
                                .enumerate()
                                .map(|(time, &row)| {
                                    let sign = if (row & bin).count_ones() & 1 == 0 {
                                        1.0
                                    } else {
                                        -1.0
                                    };
                                    sign * input
                                        [frequency * transform.coded_blocks + block * factor + time]
                                })
                                .sum::<f64>();
                            expected
                                [(frequency * factor + bin) * transform.original_blocks + block] =
                                scale * value;
                        }
                    }
                }
            }
            for (&actual, &expected) in actual.iter().zip(&expected) {
                assert!((actual - expected).abs() < 1e-12);
            }
            let before = input.iter().map(|v| v * v).sum::<f64>();
            let after = actual.iter().map(|v| v * v).sum::<f64>();
            assert!((before - after).abs() < 1e-9);
        }
    }

    #[test]
    fn inverse_tf_rejects_invalid_shapes_without_modifying_output() {
        let layout = Layout::from_configuration(31).unwrap().unwrap();
        assert!(BandTransform::new(layout, 0, false, 1).is_err());
        assert!(BandTransform::new(layout, 0, true, -1).is_err());
        assert!(BandTransform::new(layout, 0, true, i8::MIN).is_err());
        let transform = BandTransform::new(layout, 0, true, 1).unwrap();
        let mut wrong_size = [1.0; 7];
        assert!(transform.inverse(&mut wrong_size).is_err());
        assert_eq!(wrong_size, [1.0; 7]);
        let mut overflow = [f64::MAX; 8];
        assert!(transform.inverse(&mut overflow).is_err());
        assert_eq!(overflow, [f64::MAX; 8]);
    }

    #[test]
    fn recovered_tables_have_all_normative_adjustments() {
        for lm in 0..=3 {
            assert_eq!(
                TfResolution::adjustment(lm, false, false, false).unwrap(),
                0
            );
            assert_eq!(TfResolution::adjustment(lm, false, true, false).unwrap(), 0);
            assert_eq!(
                TfResolution::adjustment(lm, false, false, true).unwrap(),
                [-1, -1, -2, -2][lm as usize]
            );
            assert_eq!(
                TfResolution::adjustment(lm, false, true, true).unwrap(),
                [-1, -2, -3, -3][lm as usize]
            );
            assert_eq!(
                TfResolution::adjustment(lm, true, false, false).unwrap(),
                [0, 1, 2, 3][lm as usize]
            );
            assert_eq!(
                TfResolution::adjustment(lm, true, true, false).unwrap(),
                [0, 1, 1, 1][lm as usize]
            );
            assert_eq!(
                TfResolution::adjustment(lm, true, false, true).unwrap(),
                [-1, 0, 0, 0][lm as usize]
            );
            assert_eq!(TfResolution::adjustment(lm, true, true, true).unwrap(), -1);
        }
    }

    #[test]
    fn tf_flags_follow_relative_choice_and_conditional_select() {
        for config in 16..=31 {
            let layout = Layout::from_configuration(config).unwrap().unwrap();
            for transient in [false, true] {
                if layout.blocks(transient).is_err() {
                    continue;
                }
                for seed in 0..=255u32 {
                    let bytes: Vec<_> = (0..128).map(|i| (seed * 73 + i * 137) as u8).collect();
                    let mut actual = RangeDecoder::new(&bytes);
                    let mut expected = actual.clone();
                    let decoded = TfResolution::decode(&mut actual, layout, transient).unwrap();
                    let mut choice = false;
                    let mut matters = false;
                    let lm = (layout.samples() / 120).trailing_zeros() as u8;
                    for (index, &decoded_choice) in decoded.choices.iter().enumerate() {
                        let logp = if index == 0 {
                            if transient { 2 } else { 4 }
                        } else {
                            if transient { 4 } else { 5 }
                        };
                        choice ^= expected.symbol(&[((1u32 << logp) - 1) as u16, 1]).unwrap() != 0;
                        assert_eq!(decoded_choice, choice);
                        matters |= TfResolution::adjustment(lm, transient, false, choice).unwrap()
                            != TfResolution::adjustment(lm, transient, true, choice).unwrap();
                    }
                    let selected = matters && expected.symbol(&[1, 1]).unwrap() != 0;
                    assert_eq!(decoded.selected, selected);
                    assert_eq!(actual.tell_fractional(), expected.tell_fractional());
                }
            }
        }
        let mut truncated = RangeDecoder::new(&[0]);
        assert!(matches!(
            TfResolution::decode(
                &mut truncated,
                Layout::from_configuration(31).unwrap().unwrap(),
                false
            ),
            Err(MediaDecodeError::Unsupported)
        ));
        assert_eq!(truncated.tell(), 1);
    }
}
