//! Vorbis floor-zero LSP packet decoding and Bark-scale envelope (section 6.2).

use super::entropy::{Codebook, PacketBits};
use super::setup::FloorZero;
use crate::video::backend::MediaDecodeError;
use std::f64::consts::PI;

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.into())
}

#[derive(Debug, Default)]
pub struct Lsp {
    pub amplitude: u64,
    pub coefficients: Vec<f64>,
}

fn amplitude(bits: &mut PacketBits<'_>, width: u8) -> Option<u64> {
    let low_width = width.min(32);
    let low = u64::from(bits.read(low_width)?);
    if width <= 32 {
        Some(low)
    } else {
        Some(low | (u64::from(bits.read(width - 32)?) << 32))
    }
}

fn bark(frequency: f64) -> f64 {
    13.1 * (0.00074 * frequency).atan()
        + 2.24 * (0.0000000185 * frequency * frequency).atan()
        + 0.0001 * frequency
}

impl FloorZero {
    pub fn decode_lsp(
        &self,
        bits: &mut PacketBits<'_>,
        books: &[Codebook],
    ) -> Result<Option<Lsp>, MediaDecodeError> {
        let mut lsp = Lsp::default();
        let mut vector = Vec::new();
        Ok(self
            .decode_lsp_into(bits, books, &mut lsp, &mut vector)?
            .then_some(lsp))
    }

    pub(super) fn decode_lsp_into(
        &self,
        bits: &mut PacketBits<'_>,
        books: &[Codebook],
        lsp: &mut Lsp,
        vector: &mut Vec<f64>,
    ) -> Result<bool, MediaDecodeError> {
        if !(1..=255).contains(&self.order)
            || self.amplitude_bits > 63
            || self.books.is_empty()
            || self.books.len() > 16
        {
            return Err(invalid("invalid Vorbis floor-zero packet configuration"));
        }
        lsp.amplitude = 0;
        lsp.coefficients.clear();
        let Some(amplitude) = amplitude(bits, self.amplitude_bits) else {
            return Ok(false);
        };
        if amplitude == 0 {
            return Ok(false);
        }
        let width = (usize::BITS - self.books.len().leading_zeros()) as u8;
        let Some(selection) = bits.read(width) else {
            return Ok(false);
        };
        let index = self
            .books
            .get(selection as usize)
            .ok_or_else(|| invalid("reserved Vorbis floor-zero book selection"))?;
        let book = books
            .get(*index)
            .ok_or_else(|| invalid("invalid Vorbis floor-zero codebook"))?;
        if book.dimensions == 0 || book.dimensions > 65535 || !book.has_lookup() {
            return Err(invalid("invalid Vorbis floor-zero vector book"));
        }
        lsp.coefficients.reserve(self.order);
        let coefficients = &mut lsp.coefficients;
        vector.resize(book.dimensions, 0.0);
        let mut last = 0.0;
        while coefficients.len() < self.order {
            let Some(entry) = book.huffman.decode(bits) else {
                return Ok(false);
            };
            book.vector(entry, vector)?;
            let count = (self.order - coefficients.len()).min(vector.len());
            coefficients.extend(vector.iter().take(count).map(|value| value + last));
            last += vector[vector.len() - 1];
        }
        lsp.amplitude = amplitude;
        Ok(true)
    }

    pub fn render_curve(&self, lsp: &Lsp, output: &mut [f64]) -> Result<(), MediaDecodeError> {
        self.render_curve_with_workspace(lsp, output, &mut Vec::new())
    }

    pub(super) fn render_curve_with_workspace(
        &self,
        lsp: &Lsp,
        output: &mut [f64],
        cosines: &mut Vec<f64>,
    ) -> Result<(), MediaDecodeError> {
        if !(1..=255).contains(&self.order)
            || !(1..=65535).contains(&self.rate)
            || !(1..=65535).contains(&self.bark_map_size)
            || self.amplitude_bits > 63
            || self.amplitude_offset > 255
            || output.len() > 4096
            || lsp.coefficients.len() != self.order
            || lsp.coefficients.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("invalid Vorbis floor-zero curve configuration"));
        }
        let maximum = (1u64 << self.amplitude_bits) - 1;
        if lsp.amplitude > maximum {
            return Err(invalid("invalid Vorbis floor-zero amplitude"));
        }
        if lsp.amplitude == 0 || output.is_empty() {
            output.fill(0.0);
            return Ok(());
        }
        let amplitude = lsp.amplitude as f64 * self.amplitude_offset as f64 / maximum as f64;
        cosines.clear();
        cosines.extend(lsp.coefficients.iter().map(|coefficient| coefficient.cos()));
        let bark_scale = self.bark_map_size as f64 / bark(self.rate as f64 / 2.0);
        let frequency_step = self.rate as f64 / (2.0 * output.len() as f64);
        let mut previous_map = None;
        let mut previous_value = 0.0;
        for (index, value) in output.iter_mut().enumerate() {
            let map = ((bark(index as f64 * frequency_step) * bark_scale).floor() as usize)
                .min(self.bark_map_size - 1);
            if previous_map == Some(map) {
                *value = previous_value;
                continue;
            }
            let cosine = (PI * map as f64 / self.bark_map_size as f64).cos();
            let mut even = 1.0;
            let mut odd = 1.0;
            for (index, &coefficient) in cosines.iter().enumerate() {
                let factor = 4.0 * (coefficient - cosine).powi(2);
                if index & 1 == 0 {
                    even *= factor;
                } else {
                    odd *= factor;
                }
            }
            let (p, q) = if self.order & 1 != 0 {
                ((1.0 - cosine * cosine) * odd, 0.25 * even)
            } else {
                ((1.0 - cosine) * 0.5 * odd, (1.0 + cosine) * 0.5 * even)
            };
            previous_value =
                (0.11512925 * (amplitude / (p + q).sqrt() - self.amplitude_offset as f64)).exp();
            if !previous_value.is_finite() {
                return Err(invalid("nonfinite Vorbis floor-zero envelope"));
            }
            *value = previous_value;
            previous_map = Some(map);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn floor(order: usize) -> FloorZero {
        FloorZero {
            order,
            rate: 48000,
            bark_map_size: 32,
            amplitude_bits: 6,
            amplitude_offset: 60,
            books: vec![0],
        }
    }

    fn pack(fields: &[(u32, u8)]) -> Vec<u8> {
        let mut bytes = vec![
            0;
            fields
                .iter()
                .map(|&(_, width)| usize::from(width))
                .sum::<usize>()
                .div_ceil(8)
        ];
        let mut position = 0;
        for &(value, width) in fields {
            for bit in 0..width {
                bytes[position / 8] |= ((value >> bit) as u8 & 1) << (position & 7);
                position += 1;
            }
        }
        bytes
    }

    #[test]
    fn vector_carry_and_overfilled_last_vector_are_handled() {
        let bytes = pack(&[
            (0x564342, 24),
            (2, 16),
            (1, 24),
            (0, 1),
            (0, 1),
            (0, 5),
            (2, 4),
            (0, 32),
            ((787 << 21) | 1, 32),
            (1, 4),
            (0, 1),
            (1, 2),
            (2, 2),
        ]);
        let book = Codebook::parse(&mut PacketBits::new(&bytes), &mut 1, &mut 2).unwrap();
        let bytes = pack(&[(31, 6), (0, 1), (0, 1), (0, 1)]);
        let mut bits = PacketBits::new(&bytes);
        let lsp = floor(3).decode_lsp(&mut bits, &[book]).unwrap().unwrap();
        assert_eq!(lsp.amplitude, 31);
        assert_eq!(lsp.coefficients, [0.5, 1.0, 1.5]);
        assert_eq!(bits.position(), 9);
        assert!(
            floor(3)
                .decode_lsp(&mut PacketBits::new(&[]), &[])
                .unwrap()
                .is_none()
        );
        assert!(
            floor(3)
                .decode_lsp(&mut PacketBits::new(&[0]), &[])
                .unwrap()
                .is_none()
        );
        assert!(
            floor(3)
                .decode_lsp(&mut PacketBits::new(&[255]), &[])
                .is_err()
        );
    }

    #[test]
    fn amplitude_supports_all_six_bit_widths() {
        for width in 0..=63 {
            let bytes = [255; 8];
            let mut bits = PacketBits::new(&bytes);
            assert_eq!(amplitude(&mut bits, width).unwrap(), (1u64 << width) - 1);
            assert_eq!(bits.position(), usize::from(width));
        }
    }

    #[test]
    fn lsp_workspace_reuses_storage_and_recovers_after_partial_decode() {
        let encoded_book = pack(&[
            (0x564342, 24),
            (2, 16),
            (1, 24),
            (0, 1),
            (0, 1),
            (0, 5),
            (2, 4),
            (0, 32),
            ((787 << 21) | 1, 32),
            (1, 4),
            (0, 1),
            (1, 2),
            (2, 2),
        ]);
        let books =
            vec![Codebook::parse(&mut PacketBits::new(&encoded_book), &mut 1, &mut 2).unwrap()];
        let complete = pack(&[(31, 6), (0, 1), (0, 1), (0, 1)]);
        let floor = floor(3);
        let mut lsp = Lsp::default();
        let mut vector = Vec::new();
        assert!(
            floor
                .decode_lsp_into(
                    &mut PacketBits::new(&complete),
                    &books,
                    &mut lsp,
                    &mut vector
                )
                .unwrap()
        );
        let coefficients_pointer = lsp.coefficients.as_ptr();
        let vector_pointer = vector.as_ptr();
        for _ in 0..4 {
            let mut bits = PacketBits::new(&complete[..1]);
            assert!(
                !floor
                    .decode_lsp_into(&mut bits, &books, &mut lsp, &mut vector)
                    .unwrap()
            );
            assert!(bits.ended());
            assert_eq!(lsp.amplitude, 0);
            assert_eq!(lsp.coefficients, [0.5, 1.0]);
            assert!(
                floor
                    .decode_lsp_into(&mut PacketBits::new(&[255]), &books, &mut lsp, &mut vector)
                    .is_err()
            );
            assert!(
                !floor
                    .decode_lsp_into(&mut PacketBits::new(&[0]), &books, &mut lsp, &mut vector)
                    .unwrap()
            );
            assert!(lsp.coefficients.is_empty());
            assert!(
                floor
                    .decode_lsp_into(
                        &mut PacketBits::new(&complete),
                        &books,
                        &mut lsp,
                        &mut vector
                    )
                    .unwrap()
            );
            assert_eq!(lsp.amplitude, 31);
            assert_eq!(lsp.coefficients, [0.5, 1.0, 1.5]);
            assert_eq!(lsp.coefficients.as_ptr(), coefficients_pointer);
            assert_eq!(vector.as_ptr(), vector_pointer);
        }
    }

    #[test]
    fn curve_workspace_matches_owned_curve_and_recovers_after_nonfinite_envelope() {
        let mut cosines = Vec::new();
        for order in [3, 1, 2, 3] {
            let floor = floor(order);
            let mut lsp = Lsp {
                amplitude: 31,
                coefficients: (1..=order).map(|index| 0.8 * index as f64).collect(),
            };
            let mut output = [0.0; 64];
            let mut expected = [0.0; 64];
            floor.render_curve(&lsp, &mut expected).unwrap();
            floor
                .render_curve_with_workspace(&lsp, &mut output, &mut cosines)
                .unwrap();
            assert_eq!(output, expected);
            let pointer = cosines.as_ptr();
            lsp.coefficients[0] = 0.0;
            // At DC the even-order envelope has a zero denominator.
            if order == 2 {
                assert!(
                    floor
                        .render_curve_with_workspace(&lsp, &mut output, &mut cosines)
                        .is_err()
                );
            }
            lsp.coefficients[0] = 0.8;
            floor
                .render_curve_with_workspace(&lsp, &mut output, &mut cosines)
                .unwrap();
            assert_eq!(output, expected);
            assert_eq!(cosines.as_ptr(), pointer);
        }
    }

    #[test]
    fn order_one_matches_closed_form_lsp_response() {
        let floor = floor(1);
        let lsp = Lsp {
            amplitude: 31,
            coefficients: vec![0.8],
        };
        let mut output = [0.0; 64];
        floor.render_curve(&lsp, &mut output).unwrap();
        for (index, &value) in output.iter().enumerate() {
            let map = ((bark(24000.0 * index as f64 / 64.0) / bark(24000.0) * 32.0).floor()
                as usize)
                .min(31);
            let cosine = (PI * map as f64 / 32.0).cos();
            let denominator = (1.0 - cosine * cosine + (0.8f64.cos() - cosine).powi(2)).sqrt();
            let expected = (0.11512925 * (31.0 * 60.0 / 63.0 / denominator - 60.0)).exp();
            assert!((value - expected).abs() <= expected.abs() * 1e-12);
        }
    }

    #[test]
    fn even_order_has_the_correct_dc_response_and_rejects_nonfinite_input() {
        let floor = floor(2);
        let mut lsp = Lsp {
            amplitude: 31,
            coefficients: vec![0.8, 1.6],
        };
        let mut output = [0.0; 64];
        floor.render_curve(&lsp, &mut output).unwrap();
        let expected =
            (0.11512925 * (31.0 * 60.0 / 63.0 / (2.0 * (0.8f64.cos() - 1.0).abs()) - 60.0)).exp();
        assert_eq!(output[0], expected);
        lsp.coefficients[0] = f64::NAN;
        assert!(floor.render_curve(&lsp, &mut output).is_err());
        lsp.coefficients[0] = 0.8;
        lsp.amplitude = 64;
        assert!(floor.render_curve(&lsp, &mut output).is_err());
        lsp.amplitude = 0;
        floor.render_curve(&lsp, &mut output).unwrap();
        assert_eq!(output, [0.0; 64]);
    }
}
