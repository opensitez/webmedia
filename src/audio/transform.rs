//! Inverse MDCT via a zero-padded Fourier transform, without codec dependencies.

use crate::video::backend::MediaDecodeError;
use std::f64::consts::TAU;

pub struct MdctPlan {
    size: usize,
    roots: Vec<(f64, f64)>,
    permutation: Vec<usize>,
    scratch: Vec<(f64, f64)>,
    mixed: Option<MixedRadixPlan>,
}

impl MdctPlan {
    pub fn new(size: usize) -> Result<Self, MediaDecodeError> {
        if !(64..=8192).contains(&size)
            || (!size.is_power_of_two() && ![240, 480, 960, 1920].contains(&size))
        {
            return Err(MediaDecodeError::InvalidData(
                "invalid audio MDCT size".into(),
            ));
        }
        let length = size * 4;
        if !size.is_power_of_two() {
            return Ok(Self {
                size,
                roots: Vec::new(),
                permutation: Vec::new(),
                scratch: vec![(0.0, 0.0); length],
                mixed: Some(MixedRadixPlan::new(length)),
            });
        }
        let roots = (0..length / 2)
            .map(|index| {
                let (sin, cos) = (-TAU * index as f64 / length as f64).sin_cos();
                (cos, sin)
            })
            .collect();
        let shift = usize::BITS - length.trailing_zeros();
        let permutation = (0..length)
            .map(|index| index.reverse_bits() >> shift)
            .collect();
        Ok(Self {
            size,
            roots,
            permutation,
            scratch: vec![(0.0, 0.0); length],
            mixed: None,
        })
    }

    pub fn inverse(
        &mut self,
        spectrum: &[f64],
        output: &mut [f64],
    ) -> Result<(), MediaDecodeError> {
        if spectrum.len() != self.size / 2
            || output.len() != self.size
            || spectrum.iter().any(|value| !value.is_finite())
        {
            return Err(MediaDecodeError::InvalidData(
                "invalid audio MDCT input".into(),
            ));
        }
        self.scratch.fill((0.0, 0.0));
        if let Some(mixed) = &mut self.mixed {
            for (index, &value) in spectrum.iter().enumerate() {
                self.scratch[2 * index + 1].0 = value;
            }
            let transformed = mixed.forward(&self.scratch);
            for (index, value) in output.iter_mut().enumerate() {
                *value = transformed[2 * index + 1 + self.size / 2].0;
            }
            return Ok(());
        }
        for (index, &value) in spectrum.iter().enumerate() {
            self.scratch[self.permutation[2 * index + 1]].0 = value;
        }
        let length = self.scratch.len();
        let mut width = 2;
        while width <= length {
            let half = width / 2;
            let root_step = length / width;
            for start in (0..length).step_by(width) {
                for offset in 0..half {
                    let (real, imag) = self.scratch[start + half + offset];
                    let (cos, sin) = self.roots[offset * root_step];
                    let rotated = (real * cos - imag * sin, real * sin + imag * cos);
                    let (real, imag) = self.scratch[start + offset];
                    self.scratch[start + offset] = (real + rotated.0, imag + rotated.1);
                    self.scratch[start + half + offset] = (real - rotated.0, imag - rotated.1);
                }
            }
            width *= 2;
        }
        for (index, value) in output.iter_mut().enumerate() {
            *value = self.scratch[2 * index + 1 + self.size / 2].0;
        }
        Ok(())
    }
}

struct FourierStage {
    length: usize,
    radix: usize,
    roots: Vec<(f64, f64)>,
}

struct MixedRadixPlan {
    stages: Vec<FourierStage>,
    output: Vec<(f64, f64)>,
    scratch: Vec<(f64, f64)>,
}

impl MixedRadixPlan {
    fn new(length: usize) -> Self {
        let mut stages = Vec::new();
        let mut remaining = length;
        while remaining > 1 {
            let radix = if remaining % 2 == 0 {
                2
            } else if remaining % 3 == 0 {
                3
            } else {
                5
            };
            debug_assert_eq!(remaining % radix, 0);
            let roots = (0..remaining)
                .map(|index| {
                    let (sine, cosine) = (-TAU * index as f64 / remaining as f64).sin_cos();
                    (cosine, sine)
                })
                .collect();
            stages.push(FourierStage {
                length: remaining,
                radix,
                roots,
            });
            remaining /= radix;
        }
        Self {
            stages,
            output: vec![(0.0, 0.0); length],
            scratch: vec![(0.0, 0.0); length],
        }
    }

    fn forward(&mut self, input: &[(f64, f64)]) -> &[(f64, f64)] {
        fourier(
            &self.stages,
            input,
            0,
            1,
            &mut self.output,
            &mut self.scratch,
        );
        &self.output
    }
}

fn fourier(
    stages: &[FourierStage],
    input: &[(f64, f64)],
    start: usize,
    stride: usize,
    output: &mut [(f64, f64)],
    scratch: &mut [(f64, f64)],
) {
    let Some((stage, next)) = stages.split_first() else {
        output[0] = input[start];
        return;
    };
    let width = stage.length / stage.radix;
    for (branch, (output, scratch)) in output
        .chunks_exact_mut(width)
        .zip(scratch.chunks_exact_mut(width))
        .enumerate()
    {
        fourier(
            next,
            input,
            start + branch * stride,
            stride * stage.radix,
            output,
            scratch,
        );
    }
    // Child transforms are contiguous; combine into reusable scratch before
    // replacing them, so no packet-time allocation or trigonometry is needed.
    for (index, value) in scratch.iter_mut().enumerate() {
        let bin = index % width;
        let mut sum = output[bin];
        for branch in 1..stage.radix {
            let (real, imaginary) = output[branch * width + bin];
            let (cosine, sine) = stage.roots[branch * index % stage.length];
            sum.0 += real * cosine - imaginary * sine;
            sum.1 += real * sine + imaginary * cosine;
        }
        *value = sum;
    }
    output.copy_from_slice(scratch);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourier_mdct_matches_independent_cosine_sum() {
        for size in [64, 128, 256, 240, 480, 960, 1920] {
            let spectrum: Vec<_> = (0..size / 2)
                .map(|i| ((i * 137 + 71) % 100) as f64 / 100.0 - 0.5)
                .collect();
            let mut plan = MdctPlan::new(size).unwrap();
            let mut output = vec![0.0; size];
            for _ in 0..2 {
                plan.inverse(&spectrum, &mut output).unwrap();
                for (time, &value) in output.iter().enumerate() {
                    let expected: f64 = spectrum
                        .iter()
                        .enumerate()
                        .map(|(frequency, &value)| {
                            value
                                * (TAU / size as f64
                                    * (time as f64 + 0.5 + size as f64 / 4.0)
                                    * (frequency as f64 + 0.5))
                                    .cos()
                        })
                        .sum();
                    assert!(
                        (expected - value).abs() < 1e-10,
                        "{size}/{time}: {expected} != {value}"
                    );
                }
            }
        }
    }

    #[test]
    fn zero_input_and_invalid_shapes_are_bounded() {
        let mut plan = MdctPlan::new(64).unwrap();
        let mut output = [1.0; 64];
        plan.inverse(&[0.0; 32], &mut output).unwrap();
        assert_eq!(output, [0.0; 64]);
        assert!(plan.inverse(&[0.0; 31], &mut output).is_err());
        assert!(plan.inverse(&[f64::NAN; 32], &mut output).is_err());
        for size in [0, 32, 65, 16384, usize::MAX] {
            assert!(MdctPlan::new(size).is_err());
        }
    }

    #[test]
    fn mixed_radix_fourier_matches_complex_definition() {
        for length in [6, 10, 15, 30, 60, 120] {
            let input: Vec<_> = (0..length)
                .map(|index| ((index * 37 % 11) as f64, (index * 13 % 7) as f64))
                .collect();
            let mut plan = MixedRadixPlan::new(length);
            let output = plan.forward(&input);
            for (bin, &actual) in output.iter().enumerate() {
                let expected =
                    input
                        .iter()
                        .enumerate()
                        .fold((0.0, 0.0), |sum, (time, &(real, imaginary))| {
                            let (sine, cosine) =
                                (-TAU * (time * bin) as f64 / length as f64).sin_cos();
                            (
                                sum.0 + real * cosine - imaginary * sine,
                                sum.1 + real * sine + imaginary * cosine,
                            )
                        });
                assert!((actual.0 - expected.0).abs() < 1e-10);
                assert!((actual.1 - expected.1).abs() < 1e-10);
            }
            let zeros = vec![(0.0, 0.0); length];
            assert!(
                plan.forward(&zeros)
                    .iter()
                    .all(|&value| value == (0.0, 0.0))
            );
        }
    }
}
