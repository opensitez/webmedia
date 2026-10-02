//! VP8 quantized coefficient tokens and frame-local probability updates.

use super::backend::MediaDecodeError;
use super::vp8::BoolDecoder;
use super::vp8_probs::{COEFF_UPDATE_PROBS, DEFAULT_COEFF_PROBS};

const BANDS: [usize; 16] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7];
const ZIGZAG: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
const CATEGORY_BASE: [i32; 6] = [5, 7, 11, 19, 35, 67];
const CATEGORY_PROBS: [&[u8]; 6] = [
    &[159],
    &[165, 145],
    &[173, 148, 140],
    &[176, 155, 140, 135],
    &[180, 157, 141, 134, 130],
    &[254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129],
];

#[derive(Clone, Debug)]
pub(super) struct CoeffProbs {
    values: [u8; 1056],
}

impl Default for CoeffProbs {
    fn default() -> Self {
        Self {
            values: DEFAULT_COEFF_PROBS,
        }
    }
}

impl CoeffProbs {
    pub(super) fn update(&mut self, decoder: &mut BoolDecoder<'_>) -> Result<(), MediaDecodeError> {
        for (index, value) in self.values.iter_mut().enumerate() {
            if decoder.read(COEFF_UPDATE_PROBS[index])? {
                *value = decoder.read_literal(8)? as u8;
            }
        }
        Ok(())
    }

    pub(super) fn decode_block(
        &self,
        decoder: &mut BoolDecoder<'_>,
        plane: usize,
        mut context: usize,
        first_coefficient: usize,
    ) -> Result<([i32; 16], bool), MediaDecodeError> {
        if plane >= 4 || context >= 3 || first_coefficient > 1 {
            return Err(MediaDecodeError::InvalidData(
                "invalid VP8 coefficient context".into(),
            ));
        }
        let mut values = [0; 16];
        let mut nonzero = false;
        let mut previous_zero = false;
        for index in first_coefficient..16 {
            let base = (((plane * 8 + BANDS[index]) * 3 + context) * 11) as usize;
            let probs: &[u8; 11] = (&self.values[base..base + 11]).try_into().unwrap();
            let token = decode_token(decoder, probs, previous_zero)?;
            if token == 11 {
                break;
            }
            let magnitude = if token <= 4 {
                token as i32
            } else {
                let category = token - 5;
                let mut extra = 0;
                for &probability in CATEGORY_PROBS[category].iter() {
                    extra = (extra << 1) | i32::from(decoder.read(probability)?);
                }
                CATEGORY_BASE[category] + extra
            };
            let signed = if magnitude != 0 && decoder.read_bit()? {
                -magnitude
            } else {
                magnitude
            };
            values[ZIGZAG[index]] = signed;
            nonzero |= magnitude != 0;
            context = match magnitude {
                0 => 0,
                1 => 1,
                _ => 2,
            };
            previous_zero = magnitude == 0;
        }
        Ok((values, nonzero))
    }
}

fn decode_token(
    decoder: &mut BoolDecoder<'_>,
    probs: &[u8; 11],
    skip_eob: bool,
) -> Result<usize, MediaDecodeError> {
    if !skip_eob && !decoder.read(probs[0])? {
        return Ok(11);
    }
    if !decoder.read(probs[1])? {
        return Ok(0);
    }
    if !decoder.read(probs[2])? {
        return Ok(1);
    }
    if !decoder.read(probs[3])? {
        if !decoder.read(probs[4])? {
            return Ok(2);
        }
        return Ok(if decoder.read(probs[5])? { 4 } else { 3 });
    }
    if !decoder.read(probs[6])? {
        return Ok(if decoder.read(probs[7])? { 6 } else { 5 });
    }
    if !decoder.read(probs[8])? {
        return Ok(if decoder.read(probs[9])? { 8 } else { 7 });
    }
    Ok(if decoder.read(probs[10])? { 10 } else { 9 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probability_tables_have_expected_dimensions() {
        assert_eq!(DEFAULT_COEFF_PROBS.len(), 4 * 8 * 3 * 11);
        assert_eq!(COEFF_UPDATE_PROBS.len(), DEFAULT_COEFF_PROBS.len());
        assert_eq!(&DEFAULT_COEFF_PROBS[..11], &[128; 11]);
        assert_eq!(&COEFF_UPDATE_PROBS[..11], &[255; 11]);
        assert_eq!(BANDS[15], 7);
        assert_eq!(
            ZIGZAG
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            16
        );
    }
}
