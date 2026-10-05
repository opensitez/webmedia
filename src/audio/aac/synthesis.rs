//! AAC filterbank, ISO/IEC 13818-7:2004 section 15.3.

use super::invalid;
use crate::audio::transform::MdctPlan;
use crate::video::backend::MediaDecodeError;
use std::f64::consts::PI;

pub struct AacSynthesis {
    long: MdctPlan,
    short: MdctPlan,
    long_windows: [Vec<f64>; 2],
    short_windows: [Vec<f64>; 2],
    overlap: Vec<f64>,
    transformed: Vec<f64>,
    short_transformed: Vec<f64>,
    windowed: Vec<f64>,
    previous_shape: Option<usize>,
}

impl AacSynthesis {
    pub fn new() -> Result<Self, MediaDecodeError> {
        Self::with_frame_samples(1024)
    }

    pub fn with_frame_samples(samples: usize) -> Result<Self, MediaDecodeError> {
        if !matches!(samples, 1024 | 960) {
            return Err(invalid("invalid AAC filterbank length"));
        }
        let short = samples / 8;
        Ok(Self {
            long: MdctPlan::new(samples * 2)?,
            short: MdctPlan::new(short * 2)?,
            long_windows: windows(samples, 4.0),
            short_windows: windows(short, 6.0),
            overlap: vec![0.0; samples],
            transformed: vec![0.0; samples * 2],
            short_transformed: vec![0.0; short * 2],
            windowed: vec![0.0; samples * 2],
            previous_shape: None,
        })
    }

    pub fn reset(&mut self) {
        self.overlap.fill(0.0);
        self.previous_shape = None;
    }

    /// Spectrum is window-major for an eight-short sequence. Output is one channel.
    pub fn synthesize(
        &mut self,
        spectrum: &[f64],
        sequence: u8,
        shape: u8,
        output: &mut [f64],
    ) -> Result<(), MediaDecodeError> {
        let samples = self.overlap.len();
        let short = samples / 8;
        let transition = (samples - short) / 2;
        if spectrum.len() != samples
            || output.len() != samples
            || sequence > 3
            || shape > 1
            || spectrum.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("invalid AAC synthesis input"));
        }
        let shape = usize::from(shape);
        let previous = self.previous_shape.unwrap_or(shape);
        self.windowed.fill(0.0);
        if sequence == 2 {
            for window in 0..8 {
                self.short.inverse(
                    &spectrum[window * short..(window + 1) * short],
                    &mut self.short_transformed,
                )?;
                let left = &self.short_windows[if window == 0 { previous } else { shape }];
                let right = &self.short_windows[shape];
                let start = transition + window * short;
                for i in 0..short {
                    self.windowed[start + i] += self.short_transformed[i] * left[i] / short as f64;
                    self.windowed[start + short + i] +=
                        self.short_transformed[short + i] * right[short - 1 - i] / short as f64;
                }
            }
        } else {
            self.long.inverse(spectrum, &mut self.transformed)?;
            for i in 0..samples * 2 {
                let weight = match sequence {
                    0 if i < samples => self.long_windows[previous][i],
                    0 => self.long_windows[shape][samples * 2 - 1 - i],
                    1 if i < samples => self.long_windows[previous][i],
                    1 if i < samples + transition => 1.0,
                    1 if i < samples + transition + short => {
                        self.short_windows[shape][samples + transition + short - 1 - i]
                    }
                    1 => 0.0,
                    3 if i < transition => 0.0,
                    3 if i < transition + short => self.short_windows[previous][i - transition],
                    3 if i < samples => 1.0,
                    3 => self.long_windows[shape][samples * 2 - 1 - i],
                    _ => unreachable!("validated sequence"),
                };
                self.windowed[i] = self.transformed[i] * weight / samples as f64;
            }
        }
        if self
            .windowed
            .iter()
            .zip(self.overlap.iter().chain(std::iter::repeat(&0.0)))
            .any(|(value, overlap)| !(value + overlap).is_finite())
        {
            return Err(invalid("AAC synthesis overflow"));
        }
        for (i, value) in output.iter_mut().enumerate() {
            *value = self.overlap[i] + self.windowed[i];
        }
        self.overlap.copy_from_slice(&self.windowed[samples..]);
        self.previous_shape = Some(shape);
        Ok(())
    }
}

fn bessel_zero(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for n in 1..=128 {
        term *= (x * x * 0.25) / f64::from(n * n);
        sum += term;
        if term <= sum * f64::EPSILON {
            break;
        }
    }
    sum
}

fn windows(half: usize, alpha: f64) -> [Vec<f64>; 2] {
    let sine = (0..half)
        .map(|n| (PI * (n as f64 + 0.5) / (2 * half) as f64).sin())
        .collect();
    // The common I0(pi*alpha) denominator cancels in the cumulative ratio.
    let kernel: Vec<_> = (0..=half)
        .map(|n| {
            let coordinate = 2.0 * n as f64 / half as f64 - 1.0;
            bessel_zero(PI * alpha * (1.0 - coordinate * coordinate).max(0.0).sqrt())
        })
        .collect();
    let total: f64 = kernel.iter().sum();
    let mut cumulative = 0.0;
    let kbd = kernel[..half]
        .iter()
        .map(|value| {
            cumulative += value;
            (cumulative / total).sqrt()
        })
        .collect();
    [sine, kbd]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_obey_overlap_power_identity() {
        for (half, alpha) in [(1024, 4.0), (128, 6.0), (960, 4.0), (120, 6.0)] {
            for window in windows(half, alpha) {
                for n in 0..half {
                    assert!((window[n].powi(2) + window[half - 1 - n].powi(2) - 1.0).abs() < 1e-13);
                }
            }
        }
    }

    #[test]
    fn dc_long_blocks_match_direct_cosine_and_previous_shape() {
        let mut synthesis = AacSynthesis::new().unwrap();
        let mut spectrum = vec![0.0; 1024];
        spectrum[0] = 1024.0;
        let mut output = vec![0.0; 1024];
        synthesis.synthesize(&spectrum, 0, 0, &mut output).unwrap();
        for i in 0..1024 {
            let expected =
                (PI / 1024.0 * (i as f64 + 512.5) * 0.5).cos() * synthesis.long_windows[0][i];
            assert!((output[i] - expected).abs() < 1e-12);
        }
        let overlap = synthesis.overlap.clone();
        synthesis.synthesize(&spectrum, 0, 1, &mut output).unwrap();
        for i in 0..1024 {
            let expected = overlap[i]
                + (PI / 1024.0 * (i as f64 + 512.5) * 0.5).cos() * synthesis.long_windows[0][i];
            assert!((output[i] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn short_placement_and_transition_extents_match_standard() {
        let mut synthesis = AacSynthesis::new().unwrap();
        let mut spectrum = vec![0.0; 1024];
        spectrum[0] = 128.0;
        let mut output = vec![0.0; 1024];
        synthesis.synthesize(&spectrum, 2, 0, &mut output).unwrap();
        assert!(output[..448].iter().all(|&value| value == 0.0));
        for i in 0..256 {
            let expected = (PI / 128.0 * (i as f64 + 64.5) * 0.5).cos()
                * synthesis.short_windows[0][if i < 128 { i } else { 255 - i }];
            assert!((output[448 + i] - expected).abs() < 1e-12);
        }
        synthesis.reset();
        synthesis.synthesize(&spectrum, 1, 1, &mut output).unwrap();
        assert!(synthesis.overlap[576..].iter().all(|&value| value == 0.0));
        synthesis.reset();
        synthesis.synthesize(&spectrum, 3, 1, &mut output).unwrap();
        assert!(output[..448].iter().all(|&value| value == 0.0));
    }

    #[test]
    fn failed_input_preserves_overlap_and_output() {
        let mut synthesis = AacSynthesis::new().unwrap();
        let spectrum = vec![1.0; 1024];
        let mut output = vec![0.0; 1024];
        synthesis.synthesize(&spectrum, 0, 1, &mut output).unwrap();
        let overlap = synthesis.overlap.clone();
        output.fill(5.0);
        assert!(synthesis.synthesize(&spectrum, 4, 0, &mut output).is_err());
        assert_eq!(synthesis.overlap, overlap);
        assert!(output.iter().all(|&value| value == 5.0));
        assert_eq!(synthesis.previous_shape, Some(1));
    }

    #[test]
    fn mixed_radix_960_filterbank_matches_cosine_for_every_sequence() {
        for sequence in 0..=3 {
            let mut synthesis = AacSynthesis::with_frame_samples(960).unwrap();
            let mut spectrum = vec![0.0; 960];
            let mut output = vec![0.0; 960];
            synthesis.synthesize(&spectrum, 0, 0, &mut output).unwrap();
            let mut expected = vec![0.0; 1920];
            if sequence == 2 {
                for window in 0..8 {
                    for (bin, value) in [(0, 120.0), (37, -43.0), (119, 31.0)] {
                        spectrum[window * 120 + bin] = value;
                        for i in 0..240 {
                            let weight = if i < 120 {
                                synthesis.short_windows[usize::from(window != 0)][i]
                            } else {
                                synthesis.short_windows[1][239 - i]
                            };
                            expected[420 + window * 120 + i] += value / 120.0
                                * (PI / 120.0 * (i as f64 + 60.5) * (bin as f64 + 0.5)).cos()
                                * weight;
                        }
                    }
                }
            } else {
                for (bin, value) in [(0, 960.0), (417, -203.0), (959, 59.0)] {
                    spectrum[bin] = value;
                    for (i, expected) in expected.iter_mut().enumerate() {
                        let weight = match sequence {
                            0 if i < 960 => synthesis.long_windows[0][i],
                            0 => synthesis.long_windows[1][1919 - i],
                            1 if i < 960 => synthesis.long_windows[0][i],
                            1 if i < 1380 => 1.0,
                            1 if i < 1500 => synthesis.short_windows[1][1499 - i],
                            1 => 0.0,
                            3 if i < 420 => 0.0,
                            3 if i < 540 => synthesis.short_windows[0][i - 420],
                            3 if i < 960 => 1.0,
                            3 => synthesis.long_windows[1][1919 - i],
                            _ => unreachable!(),
                        };
                        *expected += value / 960.0
                            * (PI / 960.0 * (i as f64 + 480.5) * (bin as f64 + 0.5)).cos()
                            * weight;
                    }
                }
            }
            synthesis
                .synthesize(&spectrum, sequence, 1, &mut output)
                .unwrap();
            for (actual, expected) in output.iter().chain(&synthesis.overlap).zip(expected) {
                assert!(
                    (actual - expected).abs() < 2e-12,
                    "sequence {sequence}: {actual} != {expected}"
                );
            }
        }
        assert!(AacSynthesis::with_frame_samples(480).is_err());
        let mut synthesis = AacSynthesis::with_frame_samples(960).unwrap();
        assert!(
            synthesis
                .synthesize(&[0.0; 1024], 0, 0, &mut [0.0; 960])
                .is_err()
        );
    }
}
