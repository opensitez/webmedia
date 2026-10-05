//! AAC Main backward-adaptive prediction, ISO/IEC 13818-7:2004 section 13.

use super::{frame::IcsInfo, invalid};
use crate::video::backend::MediaDecodeError;
use std::sync::OnceLock;

pub(crate) const MAX_BANDS: [usize; 12] = [33, 33, 38, 40, 40, 40, 41, 41, 37, 37, 37, 34];
const ATTENUATION: f32 = 0.953125;
const ADAPTATION: f32 = 0.90625;

#[derive(Clone, Copy, Debug, PartialEq)]
struct State {
    delay: [f32; 2],
    correlation: [f32; 2],
    variance: [f32; 2],
}

impl Default for State {
    fn default() -> Self {
        Self {
            delay: [0.0; 2],
            correlation: [0.0; 2],
            variance: [1.0; 2],
        }
    }
}

fn truncate(value: f32) -> f32 {
    f32::from_bits(value.to_bits() & 0xffff0000)
}

fn round_infinity(value: f32) -> f32 {
    f32::from_bits(value.to_bits().wrapping_add(0x8000) & 0xffff0000)
}

fn round_even(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
}

fn inverse_variance(value: f32) -> f32 {
    static TABLES: OnceLock<([f32; 128], [f32; 256])> = OnceLock::new();
    let bits = value.to_bits();
    let exponent = (bits >> 23) & 255;
    // Section 13.3.2.4's exponent table disables coefficients at small variance.
    if exponent <= 127 {
        return 0.0;
    }
    let (mantissas, exponents) = TABLES.get_or_init(|| {
        (
            std::array::from_fn(|i| {
                round_even(ATTENUATION / f32::from_bits(0x3f800000 | ((i as u32) << 16)))
            }),
            std::array::from_fn(|i| {
                if i > 127 {
                    1.0 / f32::from_bits((i as u32) << 23)
                } else {
                    0.0
                }
            }),
        )
    });
    mantissas[((bits >> 16) & 127) as usize] * exponents[exponent as usize]
}

impl State {
    fn step(&mut self, input: f32, enabled: bool) -> Result<f32, MediaDecodeError> {
        let first = self.correlation[0] * inverse_variance(self.variance[0]);
        let second = self.correlation[1] * inverse_variance(self.variance[1]);
        let estimate = round_infinity(first * self.delay[0] + second * self.delay[1]);
        let value = if enabled { input + estimate } else { input };
        let error = value - first * self.delay[0];
        let next = Self {
            delay: [
                truncate(ATTENUATION * value),
                truncate(ATTENUATION * (self.delay[0] - first * value)),
            ],
            correlation: [
                truncate(ADAPTATION * self.correlation[0] + self.delay[0] * value),
                truncate(ADAPTATION * self.correlation[1] + self.delay[1] * error),
            ],
            variance: [
                truncate(
                    ADAPTATION * self.variance[0]
                        + 0.5 * (self.delay[0] * self.delay[0] + value * value),
                ),
                truncate(
                    ADAPTATION * self.variance[1]
                        + 0.5 * (self.delay[1] * self.delay[1] + error * error),
                ),
            ],
        };
        if !value.is_finite()
            || next
                .delay
                .iter()
                .chain(&next.correlation)
                .chain(&next.variance)
                .any(|value| !value.is_finite())
        {
            return Err(invalid("AAC predictor overflow"));
        }
        *self = next;
        Ok(value)
    }
}

pub(crate) struct Predictor {
    states: Vec<State>,
    pending: Vec<State>,
    output: Vec<f32>,
}

impl Predictor {
    pub(crate) fn new(samples: usize) -> Self {
        Self {
            states: vec![State::default(); samples],
            pending: vec![State::default(); samples],
            output: vec![0.0; samples],
        }
    }

    pub(crate) fn reset(&mut self) {
        self.states.fill(State::default());
    }

    pub(crate) fn apply(
        &mut self,
        spectrum: &mut [f64],
        info: &IcsInfo,
        books: &[u8],
        frequency_index: u8,
    ) -> Result<(), MediaDecodeError> {
        if spectrum.len() != self.states.len()
            || spectrum
                .iter()
                .any(|value| !value.is_finite() || value.abs() > f64::from(f32::MAX))
        {
            return Err(invalid("invalid AAC predictor spectrum"));
        }
        if info.short() {
            self.reset();
            return Ok(());
        }
        let bands = *MAX_BANDS
            .get(usize::from(frequency_index))
            .ok_or(MediaDecodeError::Unsupported)?;
        self.pending.copy_from_slice(&self.states);
        for band in 0..bands {
            let start = usize::from(info.offsets[band]);
            let end = usize::from(info.offsets[band + 1]);
            let book = books.get(band).copied().unwrap_or(0);
            let enabled = info.prediction_used.get(band) == Some(&true) && !matches!(book, 13..=15);
            for bin in start..end {
                if book == 13 {
                    self.pending[bin] = State::default();
                    self.output[bin] = spectrum[bin] as f32;
                } else {
                    self.output[bin] = self.pending[bin].step(spectrum[bin] as f32, enabled)?;
                }
            }
        }
        if let Some(group) = info.predictor_reset {
            for bin in (usize::from(group) - 1..self.pending.len()).step_by(30) {
                self.pending[bin] = State::default();
            }
        }
        self.states.copy_from_slice(&self.pending);
        for bin in 0..usize::from(info.offsets[bands]) {
            spectrum[bin] = f64::from(self.output[bin]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_states_are_truncated_and_rounding_obeys_ties() {
        for bits in [0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000, 0x00008000] {
            let value = f32::from_bits(bits);
            assert_eq!(truncate(value).to_bits(), bits & 0xffff0000);
            assert_eq!(
                round_infinity(value).to_bits(),
                (bits + 0x8000) & 0xffff0000
            );
            assert_eq!(
                round_even(value).to_bits(),
                (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
            );
        }
        let mut state = State::default();
        for sample in [16384.0, -7103.0, 9841.0, 0.0, 29103.0] {
            state.step(sample, true).unwrap();
            for value in state
                .delay
                .iter()
                .chain(&state.correlation)
                .chain(&state.variance)
            {
                assert_eq!(value.to_bits() & 0xffff, 0);
            }
        }
    }

    #[test]
    fn adaptation_runs_even_when_prediction_is_disabled() {
        let mut state = State::default();
        assert_eq!(state.step(8192.0, false).unwrap(), 8192.0);
        assert_eq!(state.delay, [7808.0, 0.0]);
        assert_eq!(state.correlation, [0.0, 0.0]);
        assert_eq!(state.variance, [33554432.0; 2]);
        assert_eq!(state.step(8192.0, false).unwrap(), 8192.0);
        assert_eq!(state.delay, [7808.0, 7424.0]);
        assert_eq!(state.correlation, [63963136.0, 0.0]);
        assert_eq!(state.variance, [94371840.0, 63963136.0]);
        assert!(state.step(0.0, true).unwrap() > 0.0);
        state = State::default();
        assert_eq!(state.step(0.0, true).unwrap(), 0.0);
    }

    #[test]
    fn disabled_intensity_predictor_keeps_the_reconstructed_sample_in_history() {
        let mut state = State::default();
        state.step(8192.0, true).unwrap();
        state.step(8192.0, true).unwrap();
        assert_eq!(state.step(0.125, false).unwrap(), 0.125);
        assert_eq!(state.delay[0], 0.119140625);
        assert_ne!(state.correlation[0], 0.0);
        assert_ne!(state.variance[0], 1.0);
    }

    #[test]
    fn reciprocal_tables_cover_subnormals_without_exponent_wrap() {
        for exponent in 128..=254 {
            for mantissa in 0..128 {
                let value = f32::from_bits((exponent << 23) | (mantissa << 16));
                let result = inverse_variance(value);
                assert!(result.is_finite() && result > 0.0);
                let expected =
                    round_even(ATTENUATION / f32::from_bits(0x3f800000 | (mantissa << 16)))
                        * (1.0 / f32::from_bits(exponent << 23));
                assert_eq!(result.to_bits(), expected.to_bits());
            }
        }
        assert_eq!(inverse_variance(0.0), 0.0);
        assert_eq!(inverse_variance(1.5), 0.0);
    }

    #[test]
    fn reset_groups_short_windows_and_noise_follow_tool_rules() {
        let info = IcsInfo {
            sequence: 0,
            shape: 0,
            max_sfb: 1,
            groups: vec![1],
            offsets: super::super::bands::bands(3, false).unwrap(),
            prediction_used: vec![true],
            predictor_reset: None,
        };
        let mut predictor = Predictor::new(1024);
        let mut spectrum = vec![0.0; 1024];
        for _ in 0..3 {
            spectrum[0] = 8192.0;
            predictor.apply(&mut spectrum, &info, &[1], 3).unwrap();
        }
        let mut reset = info.clone();
        reset.predictor_reset = Some(1);
        predictor.apply(&mut spectrum, &reset, &[1], 3).unwrap();
        assert_eq!(predictor.states[0], State::default());
        assert_ne!(predictor.states[1], State::default());
        let mut short = info.clone();
        short.sequence = 2;
        predictor.apply(&mut spectrum, &short, &[1], 3).unwrap();
        assert!(
            predictor
                .states
                .iter()
                .all(|state| *state == State::default())
        );
        spectrum[0] = 8192.0;
        predictor.apply(&mut spectrum, &info, &[1], 3).unwrap();
        spectrum[0] = 0.125;
        predictor.apply(&mut spectrum, &info, &[13], 3).unwrap();
        assert_eq!(spectrum[0], 0.125);
        assert_eq!(predictor.states[0], State::default());
        predictor.apply(&mut spectrum, &info, &[15], 3).unwrap();
        assert_eq!(spectrum[0], 0.125);
        assert_ne!(predictor.states[0], State::default());
        let before = predictor.states.clone();
        spectrum[2] = f64::MAX;
        let original = spectrum.clone();
        assert!(predictor.apply(&mut spectrum, &info, &[1], 3).is_err());
        assert_eq!(predictor.states, before);
        assert_eq!(spectrum, original);
    }

    #[test]
    #[ignore = "diagnostic only: set WEBMEDIA_AAC_MAIN_STEREO_REFERENCE_PCM"]
    fn diagnose_intensity_transition_history_without_changing_decoder() {
        use super::super::AacSynthesis;

        let bytes = std::fs::read(std::env::var("WEBMEDIA_AAC_MAIN_STEREO_REFERENCE_PCM").unwrap())
            .unwrap();
        let reference: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(reference.len(), 204800);
        for history in [
            "normative",
            "zero",
            "frozen",
            "reset",
            "residual",
            "zero-enabled",
        ] {
            let mut left = State::default();
            let mut right = State::default();
            let mut synthesis = AacSynthesis::new().unwrap();
            let mut spectrum = vec![0.0; 1024];
            let mut output = vec![0.0; 1024];
            let mut maximum = 0.0f32;
            for number in 0..100 {
                let intensity = (30..60).contains(&number);
                let sign = if number % 2 == 0 { 1.0 } else { -1.0 };
                let residual = sign * if intensity { 8192.0 } else { 4096.0 };
                let reconstructed = left.step(residual, true).unwrap();
                let value = if intensity {
                    let scaled = (-f64::from(reconstructed) * 2f64.powf(-0.25)) as f32;
                    match history {
                        "normative" => {
                            right.step(scaled, false).unwrap();
                        }
                        "zero" => {
                            right.step(0.0, false).unwrap();
                        }
                        "frozen" => {}
                        "reset" => {
                            right = State::default();
                        }
                        "residual" => {
                            right
                                .step((-f64::from(residual) * 2f64.powf(-0.25)) as f32, false)
                                .unwrap();
                        }
                        "zero-enabled" => {
                            right.step(0.0, true).unwrap();
                        }
                        _ => unreachable!(),
                    }
                    scaled
                } else {
                    right.step(sign * 12288.0, true).unwrap()
                };
                spectrum[0] = f64::from(value);
                synthesis.synthesize(&spectrum, 0, 0, &mut output).unwrap();
                for (bin, value) in output.iter().enumerate() {
                    maximum = maximum.max(
                        ((value / 32768.0) as f32 - reference[number * 2048 + bin * 2 + 1]).abs(),
                    );
                }
            }
            eprintln!("Main intensity history {history}: maximum PCM error={maximum}");
        }
    }
}
