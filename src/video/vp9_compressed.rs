//! VP9 arithmetic-coded frame header, following section 6.3 of the bitstream spec.

use super::backend::MediaDecodeError;
use super::vp8::BoolDecoder;
use super::vp9::{InterframeHeader, KeyframeLayout};
use super::vp9_coef_probs::DEFAULT_COEF_PROBS;
use super::vp9_inter_probs::InterframeProbabilities;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressedHeader {
    pub tx_mode: u8,
    pub skip_probs: [u8; 3],
    pub coef_probs: [[[[[[u8; 3]; 6]; 6]; 2]; 2]; 4],
    pub inter_probs: InterframeProbabilities,
    pub reference_mode: u8,
    pub probability_updates: usize,
}

impl Default for CompressedHeader {
    fn default() -> Self {
        Self {
            tx_mode: 0,
            skip_probs: [192, 128, 64],
            coef_probs: DEFAULT_COEF_PROBS,
            inter_probs: InterframeProbabilities::default(),
            reference_mode: 0,
            probability_updates: 0,
        }
    }
}

impl CompressedHeader {
    pub fn parse_keyframe(layout: &KeyframeLayout<'_>) -> Result<Self, MediaDecodeError> {
        Self::parse_frame(layout, None, None)
    }

    pub fn parse_interframe(
        layout: &KeyframeLayout<'_>,
        frame: &InterframeHeader,
        previous: &Self,
    ) -> Result<Self, MediaDecodeError> {
        Self::parse_frame(layout, Some(frame), Some(previous))
    }

    fn parse_frame(
        layout: &KeyframeLayout<'_>,
        frame: Option<&InterframeHeader>,
        previous: Option<&Self>,
    ) -> Result<Self, MediaDecodeError> {
        let mut bits = BoolDecoder::new(layout.compressed_header)?;
        if bits.read_bit()? {
            return Err(invalid("invalid VP9 arithmetic marker"));
        }
        let tx_mode = if layout.lossless {
            0
        } else {
            let mode = bits.read_literal(2)? as u8;
            if mode == 3 && bits.read_bit()? { 4 } else { mode }
        };
        let mut updates = 0;
        let mut inter_probs = previous.map_or_else(InterframeProbabilities::default, |value| value.inter_probs.clone());
        if tx_mode == 4 {
            inter_probs.update_tx(&mut bits, &mut updates)?;
        }
        let max_tx_size = if tx_mode == 4 { 3 } else { tx_mode as usize };
        let mut coef_probs = previous.map_or(DEFAULT_COEF_PROBS, |value| value.coef_probs);
        for tx_size in 0..=max_tx_size {
            if !bits.read_bit()? {
                continue;
            }
            for block_type in 0..2 {
                for reference_type in 0..2 {
                    for band in 0..6 {
                        let contexts = if band == 0 { 3 } else { 6 };
                        for context in 0..contexts {
                            for node in 0..3 {
                                if let Some(delta) = read_probability_update(&mut bits)? {
                                    let probability = &mut coef_probs[tx_size][block_type]
                                        [reference_type][band][context][node];
                                    *probability = inverse_remap_probability(delta, *probability);
                                    updates += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut skip_probs = previous.map_or([192, 128, 64], |value| value.skip_probs);
        for probability in &mut skip_probs {
            if let Some(delta) = read_probability_update(&mut bits)? {
                *probability = inverse_remap_probability(delta, *probability);
                updates += 1;
            }
        }
        let reference_mode = if let Some(frame) = frame.filter(|frame| !frame.intra_only) {
            inter_probs.update_noncoef(&mut bits, frame, &mut updates)?
        } else {
            0
        };
        Ok(Self {
            tx_mode, skip_probs, coef_probs, inter_probs, reference_mode,
            probability_updates: updates,
        })
    }
}

pub(super) fn read_probability_update(bits: &mut BoolDecoder<'_>) -> Result<Option<u32>, MediaDecodeError> {
    if !bits.read(252)? {
        return Ok(None);
    }
    Ok(Some(read_term_subexp(bits)?))
}

pub(super) fn inverse_remap_probability(delta: u32, probability: u8) -> u8 {
    let value = inverse_map(delta);
    let m = u32::from(probability) - 1;
    let updated = if (m << 1) <= 255 {
        1 + inverse_recenter(value, m)
    } else {
        255 - inverse_recenter(value, 254 - m)
    };
    updated as u8
}

fn inverse_map(delta: u32) -> u32 {
    if delta < 20 {
        return 7 + 13 * delta;
    }
    let mut index = 20;
    for value in 1..=253 {
        if value >= 7 && (value - 7) % 13 == 0 && (value - 7) / 13 < 20 {
            continue;
        }
        if index == delta {
            return value;
        }
        index += 1;
    }
    253
}

fn inverse_recenter(value: u32, midpoint: u32) -> u32 {
    if value > 2 * midpoint {
        value
    } else if value & 1 != 0 {
        midpoint - ((value + 1) >> 1)
    } else {
        midpoint + (value >> 1)
    }
}

fn read_term_subexp(bits: &mut BoolDecoder<'_>) -> Result<u32, MediaDecodeError> {
    if !bits.read_bit()? {
        return bits.read_literal(4);
    }
    if !bits.read_bit()? {
        return Ok(bits.read_literal(4)? + 16);
    }
    if !bits.read_bit()? {
        return Ok(bits.read_literal(5)? + 32);
    }
    let value = bits.read_literal(7)?;
    if value < 65 {
        Ok(value + 64)
    } else {
        Ok((value << 1) - 1 + u32::from(bits.read_bit()?))
    }
}

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_map_covers_specification_table() {
        assert_eq!((0..20).map(inverse_map).collect::<Vec<_>>(),
            (0..20).map(|index| 7 + 13 * index).collect::<Vec<_>>());
        assert_eq!(inverse_map(20), 1);
        assert_eq!(inverse_map(254), 253);
        for probability in 1..=255 {
            for delta in 0..=254 {
                assert_ne!(inverse_remap_probability(delta, probability), 0);
            }
        }
    }
}
