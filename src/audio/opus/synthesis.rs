//! Stateful CELT spectral synthesis from RFC 6716 sections 4.3.6 and 4.3.7.
//!
//! The window and symmetric pitch-filter equation are independently specified
//! in Valin et al., AES 135 (2013), sections 3 and 5.1:
//! https://jmvalin.ca/papers/aes135_opus_celt.pdf
//! This is a spectral-to-time path, not a compressed-packet decoder. Absolute
//! band amplitudes must include the upstream mean-energy offsets. Packet-level
//! postfilter transition timing is exercised by the whole-track configuration-31
//! Partyfire and Spacewalk PCM oracles in the packet decoder tests.

use super::{
    celt::{Deemphasis, Layout},
    energy::FineEnergy,
    invalid,
    range::RangeDecoder,
};
use crate::{audio::transform::MdctPlan, video::backend::MediaDecodeError};
use std::f64::consts::FRAC_PI_2;

const OVERLAP: usize = 120;
const PITCH_HISTORY: usize = 1024;

/// Exact transmitted pitch parameters. The absent filter is represented by
/// None, rather than a guessed pitch or a zero-gain transmitted value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PitchParameters {
    period: u16,
    gain_index: u8,
    tapset: u8,
}

impl PitchParameters {
    pub fn new(period: u16, gain_index: u8, tapset: u8) -> Result<Self, MediaDecodeError> {
        if !(15..=1022).contains(&period) || gain_index > 7 || tapset > 2 {
            return Err(invalid("invalid CELT postfilter parameters"));
        }
        Ok(Self {
            period,
            gain_index,
            tapset,
        })
    }

    /// Call at the postfilter field, after the silence flag, only when upstream
    /// frame syntax permits this field. No packet budget or Hybrid gate guessed.
    pub fn decode(decoder: &mut RangeDecoder<'_>) -> Result<Option<Self>, MediaDecodeError> {
        let mut cursor = decoder.clone();
        let parameters = if cursor.bit(1)? {
            // Table 56 specifies six values: 0..5. Section 4.3.7.1's
            // "0 and 6" is inconsistent with its own maximum period 1022;
            // octave 6 would permit 2046. Use the table and bounded formula.
            let octave = cursor.uniform(6)? as u8;
            let fine = cursor.raw_bits(4 + octave)?;
            let period = (16u32 << octave) + fine - 1;
            let gain = cursor.raw_bits(3)? as u8;
            let tapset = cursor.symbol(&[2, 1, 1])? as u8;
            Some(Self::new(period as u16, gain, tapset)?)
        } else {
            None
        };
        *decoder = cursor;
        Ok(parameters)
    }

    fn coefficients(self) -> [f64; 3] {
        // Decimal coefficients as printed in RFC 6716 section 4.3.7.1.
        let taps = [
            [0.3066406250, 0.2170410156, 0.1296386719],
            [0.4638671875, 0.2680664062, 0.0],
            [0.7998046875, 0.1000976562, 0.0],
        ][usize::from(self.tapset)];
        let gain = 3.0 * (f64::from(self.gain_index) + 1.0) / 32.0;
        taps.map(|tap| gain * tap)
    }
}

fn rising_window() -> [f64; OVERLAP] {
    std::array::from_fn(|n| {
        let inner = (FRAC_PI_2 * (n as f64 + 0.5) / OVERLAP as f64).sin();
        (FRAC_PI_2 * inner * inner).sin()
    })
}

/// Fully reconstructed normalized channel bands, in active-band order. Short
/// transforms are interleaved bin-major/block-minor. log_amplitudes is band-
/// major/channel-minor, in absolute base-2 amplitude units, not log power.
/// Stereo uncoupling, TF transforms, spreading, folding and anti-collapse must
/// already have been performed. Hybrid frames are rejected, not zero-filled.
pub struct SpectralFrame<'a> {
    pub layout: Layout,
    pub transient: bool,
    pub normalized: &'a [&'a [f64]],
    pub log_amplitudes: &'a [f64],
}

#[derive(Clone)]
struct PitchFilter {
    history: Vec<[f64; PITCH_HISTORY]>,
    position: usize,
    previous: Option<PitchParameters>,
}

impl PitchFilter {
    fn new(channels: usize) -> Self {
        Self {
            history: vec![[0.0; PITCH_HISTORY]; channels],
            position: 0,
            previous: None,
        }
    }

    fn response(&self, channel: usize, parameters: Option<PitchParameters>) -> f64 {
        let Some(parameters) = parameters else {
            return 0.0;
        };
        let [g0, g1, g2] = parameters.coefficients();
        let period = usize::from(parameters.period);
        let delayed = |delay: usize| {
            self.history[channel][(self.position + PITCH_HISTORY - delay) % PITCH_HISTORY]
        };
        g0 * delayed(period)
            + g1 * (delayed(period - 1) + delayed(period + 1))
            + g2 * (delayed(period - 2) + delayed(period + 2))
    }

    fn process(
        &mut self,
        samples: &mut [f64],
        current: Option<PitchParameters>,
        window: &[f64; OVERLAP],
    ) -> Result<(), MediaDecodeError> {
        let channels = self.history.len();
        let transition_start = if samples.len() / channels > OVERLAP {
            OVERLAP
        } else {
            0
        };
        for (time, frame) in samples.chunks_exact_mut(channels).enumerate() {
            let weight = if time < transition_start {
                0.0
            } else if time < transition_start + OVERLAP {
                window[time - transition_start] * window[time - transition_start]
            } else {
                1.0
            };
            for (channel, value) in frame.iter_mut().enumerate() {
                *value += (1.0 - weight) * self.response(channel, self.previous)
                    + weight * self.response(channel, current);
                if !value.is_finite() {
                    return Err(invalid("nonfinite CELT postfilter output"));
                }
                self.history[channel][self.position] = *value;
            }
            self.position = (self.position + 1) % PITCH_HISTORY;
        }
        self.previous = current;
        Ok(())
    }
}

/// Cached IMDCT plans and independent channel overlap/filter histories. Returns
/// interleaved f64 time samples in the supplied spectrum's amplitude scale.
/// No clipping, output gain, pre-skip, resampling or fabricated packet output.
pub struct CeltSynthesis {
    channels: usize,
    plans: Vec<MdctPlan>,
    window: [f64; OVERLAP],
    overlap: Vec<[f64; OVERLAP]>,
    pitch: PitchFilter,
    deemphasis: Deemphasis,
    spectrum: Vec<f64>,
    block_spectrum: Vec<f64>,
    inverse: Vec<f64>,
}

impl CeltSynthesis {
    pub fn new(channels: usize) -> Result<Self, MediaDecodeError> {
        if !(1..=2).contains(&channels) {
            return Err(invalid("invalid CELT synthesis channel count"));
        }
        let plans = [240, 480, 960, 1920]
            .into_iter()
            .map(MdctPlan::new)
            .collect::<Result<_, _>>()?;
        Ok(Self {
            channels,
            plans,
            window: rising_window(),
            overlap: vec![[0.0; OVERLAP]; channels],
            pitch: PitchFilter::new(channels),
            deemphasis: Deemphasis::new(channels)?,
            spectrum: vec![0.0; 960],
            block_spectrum: vec![0.0; 960],
            inverse: vec![0.0; 1920],
        })
    }

    pub fn reset(&mut self) {
        self.overlap.fill([0.0; OVERLAP]);
        self.pitch = PitchFilter::new(self.channels);
        self.deemphasis.reset();
    }

    /// Connect the coarse/fine/final energy stages to reconstructed shape
    /// synthesis. Exact mean-energy offsets are mandatory, one per active band;
    /// there is no fallback table. Finalization must have completed first.
    pub fn synthesize_refined(
        &mut self,
        transient: bool,
        normalized: &[&[f64]],
        energy: &FineEnergy,
        mean_log_amplitudes: &[f64],
        pitch: Option<PitchParameters>,
    ) -> Result<Vec<f64>, MediaDecodeError> {
        if energy.channels() != self.channels
            || !energy.is_finalized()
            || mean_log_amplitudes.len() != energy.layout().bands().len()
            || mean_log_amplitudes.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("invalid CELT finalized energy envelope"));
        }
        let mut log_amplitudes = [0.0; 42];
        for (index, energy) in energy.log_energies().iter().enumerate() {
            // RFC 8251 section 8: cap after restoring the band mean,
            // before converting the absolute log amplitude to linear.
            log_amplitudes[index] = (energy + mean_log_amplitudes[index / self.channels]).min(32.0);
        }
        self.synthesize(
            SpectralFrame {
                layout: energy.layout(),
                transient,
                normalized,
                log_amplitudes: &log_amplitudes[..energy.log_energies().len()],
            },
            pitch,
        )
    }

    /// All stream histories commit together only after the entire frame passes
    /// validation. Scratch state in the cached plans is not stream history.
    pub fn synthesize(
        &mut self,
        frame: SpectralFrame<'_>,
        pitch: Option<PitchParameters>,
    ) -> Result<Vec<f64>, MediaDecodeError> {
        if frame.layout.bands().start != 0 {
            return Err(MediaDecodeError::Unsupported);
        }
        let bands = frame.layout.bands();
        let coded_bins = frame.layout.band(bands.end - 1)?.end;
        let samples = frame.layout.samples();
        let blocks = frame.layout.blocks(frame.transient)?;
        let block_samples = samples / blocks;
        if frame.normalized.len() != self.channels
            || frame.normalized.iter().any(|channel| {
                channel.len() != coded_bins || channel.iter().any(|value| !value.is_finite())
            })
            || frame.log_amplitudes.len() != bands.len() * self.channels
            || frame.log_amplitudes.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("invalid CELT reconstructed spectrum"));
        }
        let mut overlap_storage = [[0.0; OVERLAP]; 2];
        let overlap = &mut overlap_storage[..self.channels];
        overlap.copy_from_slice(&self.overlap);
        let mut output = vec![0.0; samples * self.channels];
        let spectrum = &mut self.spectrum[..samples];
        let block_spectrum = &mut self.block_spectrum[..block_samples];
        let inverse = &mut self.inverse[..block_samples * 2];
        let plan_index = (block_samples / 120).trailing_zeros() as usize;
        let padding = (block_samples - OVERLAP) / 2;
        for channel in 0..self.channels {
            spectrum.fill(0.0);
            for (index, band) in bands.clone().enumerate() {
                let bins = frame.layout.band(band)?;
                let amplitude = frame.log_amplitudes[index * self.channels + channel].exp2();
                if !amplitude.is_finite() || amplitude == 0.0 {
                    return Err(invalid("CELT band amplitude outside numeric range"));
                }
                super::simd::scale_copy(
                    &mut spectrum[bins.clone()],
                    &frame.normalized[channel][bins],
                    amplitude,
                );
            }
            for block in 0..blocks {
                for bin in 0..block_samples {
                    block_spectrum[bin] = spectrum[bin * blocks + block];
                }
                self.plans[plan_index].inverse(block_spectrum, inverse)?;
                // Compact low-overlap time coordinates discard the leading and
                // trailing zero padding. Remaining support is N+120 samples.
                for time in 0..block_samples {
                    let weight = if time < OVERLAP {
                        self.window[time]
                    } else {
                        1.0
                    };
                    let old = if time < OVERLAP {
                        overlap[channel][time]
                    } else {
                        0.0
                    };
                    let value = 0.5 * inverse[padding + time] * weight + old;
                    if !value.is_finite() {
                        return Err(invalid("nonfinite CELT overlap output"));
                    }
                    output[(block * block_samples + time) * self.channels + channel] = value;
                }
                for time in 0..OVERLAP {
                    let value = 0.5
                        * inverse[padding + block_samples + time]
                        * self.window[OVERLAP - 1 - time];
                    if !value.is_finite() {
                        return Err(invalid("nonfinite CELT overlap history"));
                    }
                    overlap[channel][time] = value;
                }
            }
        }
        let mut filter = self.pitch.clone();
        let mut deemphasis = self.deemphasis.clone();
        filter.process(&mut output, pitch, &self.window)?;
        deemphasis.process(&mut output)?;
        self.overlap.copy_from_slice(overlap);
        self.pitch = filter;
        self.deemphasis = deemphasis;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn scratch_is_retained_across_long_and_short_transforms() {
        let mut synthesis = CeltSynthesis::new(2).unwrap();
        let pointers = (
            synthesis.spectrum.as_ptr(),
            synthesis.block_spectrum.as_ptr(),
            synthesis.inverse.as_ptr(),
        );
        for (configuration, transient) in [
            (31, false),
            (31, true),
            (28, false),
            (30, true),
            (31, false),
        ] {
            let layout = Layout::from_configuration(configuration).unwrap().unwrap();
            let bins = layout.band(20).unwrap().end;
            let zero = vec![0.0; bins];
            let output = synthesis
                .synthesize(
                    SpectralFrame {
                        layout,
                        transient,
                        normalized: &[&zero, &zero],
                        log_amplitudes: &[-28.0; 42],
                    },
                    None,
                )
                .unwrap();
            assert_eq!(output.len(), layout.samples() * 2);
            assert!(output.iter().all(|&sample| sample == 0.0));
            assert_eq!(
                pointers,
                (
                    synthesis.spectrum.as_ptr(),
                    synthesis.block_spectrum.as_ptr(),
                    synthesis.inverse.as_ptr(),
                )
            );
        }
    }

    #[test]
    fn postfilter_fields_match_independent_symbol_and_raw_bit_reads() {
        let mut seen = [[false; 3]; 6];
        for seed in 0..4096u32 {
            let bytes: Vec<_> = (0..16)
                .map(|i| (seed.wrapping_mul(73).wrapping_add(i * 137)) as u8)
                .collect();
            let mut cursor = RangeDecoder::new(&bytes);
            let mut expected = cursor.clone();
            let enabled = expected.symbol(&[1, 1]).unwrap() != 0;
            let parameters = PitchParameters::decode(&mut cursor).unwrap();
            if enabled {
                let octave = expected.symbol(&[1; 6]).unwrap();
                let fine = expected.raw_bits(4 + octave as u8).unwrap();
                let gain = expected.raw_bits(3).unwrap();
                let tapset = expected.inverse_cdf(&[2, 1, 0], 2).unwrap();
                assert_eq!(
                    parameters,
                    Some(
                        PitchParameters::new(
                            ((16 << octave) + fine - 1) as u16,
                            gain as u8,
                            tapset as u8
                        )
                        .unwrap()
                    )
                );
                seen[octave][tapset] = true;
            } else {
                assert_eq!(parameters, None);
            }
            assert_eq!(cursor.tell_fractional(), expected.tell_fractional());
        }
        assert!(seen.iter().flatten().all(|&value| value));
        assert!(PitchParameters::new(14, 0, 0).is_err());
        assert!(PitchParameters::new(1023, 0, 0).is_err());
        assert!(PitchParameters::new(15, 8, 0).is_err());
        assert!(PitchParameters::new(15, 0, 3).is_err());
        let mut truncated = RangeDecoder::new(&[255]);
        assert!(PitchParameters::decode(&mut truncated).is_err());
        assert_eq!(truncated.tell(), 1);
        assert_eq!(truncated.raw_bits(8).unwrap(), 255);
    }

    #[test]
    fn finalized_energy_connects_to_spectral_synthesis_with_explicit_means() {
        let layout = Layout::from_configuration(31).unwrap().unwrap();
        let mut cursor = RangeDecoder::new(&[0x96; 16]);
        let mut energy = FineEnergy::decode(&mut cursor, layout, 2, &[0.0; 42], &[1; 21]).unwrap();
        let left = vec![0.1; 800];
        let right = vec![-0.2; 800];
        let channels = [&left[..], &right[..]];
        let mut synth = CeltSynthesis::new(2).unwrap();
        assert!(
            synth
                .synthesize_refined(false, &channels, &energy, &[2.0; 21], None)
                .is_err()
        );
        energy
            .finalize(&mut cursor, &[0; 21], &[true; 21], 42)
            .unwrap();
        assert!(
            synth
                .synthesize_refined(false, &channels, &energy, &[], None)
                .is_err()
        );
        let combined: Vec<_> = energy
            .log_energies()
            .iter()
            .map(|energy| energy + 2.0)
            .collect();
        let expected = CeltSynthesis::new(2)
            .unwrap()
            .synthesize(
                SpectralFrame {
                    layout,
                    transient: false,
                    normalized: &channels,
                    log_amplitudes: &combined,
                },
                None,
            )
            .unwrap();
        assert_eq!(
            synth
                .synthesize_refined(false, &channels, &energy, &[2.0; 21], None)
                .unwrap(),
            expected
        );
    }

    #[test]
    fn finalized_energy_cap_follows_mean_restoration_per_channel() {
        let layout = Layout::from_configuration(31).unwrap().unwrap();
        let left = vec![0.1; 800];
        let right = vec![-0.2; 800];
        let channels = [&left[..], &right[..]];
        for (coarse_pair, mean, expected_pair) in [
            ([31.0, 28.0], 2.0, [32.0, 30.0]),
            ([34.0, 31.0], -3.0, [31.0, 28.0]),
            ([1000.0, 32.0], 0.0, [32.0, 32.0]),
            ([f64::MAX, 1.0], f64::MAX, [32.0, 32.0]),
        ] {
            let coarse: Vec<_> = (0..21).flat_map(|_| coarse_pair).collect();
            let mut cursor = RangeDecoder::new(&[0; 1]);
            let mut energy = FineEnergy::decode(&mut cursor, layout, 2, &coarse, &[0; 21]).unwrap();
            energy
                .finalize(&mut cursor, &[0; 21], &[false; 21], 0)
                .unwrap();
            let expected_logs: Vec<_> = (0..21).flat_map(|_| expected_pair).collect();
            let expected = CeltSynthesis::new(2)
                .unwrap()
                .synthesize(
                    SpectralFrame {
                        layout,
                        transient: false,
                        normalized: &channels,
                        log_amplitudes: &expected_logs,
                    },
                    None,
                )
                .unwrap();
            let actual = CeltSynthesis::new(2)
                .unwrap()
                .synthesize_refined(false, &channels, &energy, &[mean; 21], None)
                .unwrap();
            assert!(actual.iter().all(|value| value.is_finite()));
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn short_blocks_equal_separately_submitted_two_point_five_ms_frames() {
        let long_layout = Layout::from_configuration(31).unwrap().unwrap();
        let short_layout = Layout::from_configuration(28).unwrap().unwrap();
        let input: Vec<_> = (0..800)
            .map(|i| ((i * 29 % 53) as f64 - 26.0) / 100.0)
            .collect();
        let mut packed = CeltSynthesis::new(1).unwrap();
        let mut separate = CeltSynthesis::new(1).unwrap();
        let channels = [&input[..]];
        let actual = packed
            .synthesize(
                SpectralFrame {
                    layout: long_layout,
                    transient: true,
                    normalized: &channels,
                    log_amplitudes: &[0.0; 21],
                },
                None,
            )
            .unwrap();
        let mut expected = Vec::new();
        for block in 0..8 {
            let block_input: Vec<_> = (0..100).map(|bin| input[bin * 8 + block]).collect();
            let block_channels = [&block_input[..]];
            expected.extend(
                separate
                    .synthesize(
                        SpectralFrame {
                            layout: short_layout,
                            transient: false,
                            normalized: &block_channels,
                            log_amplitudes: &[0.0; 21],
                        },
                        None,
                    )
                    .unwrap(),
            );
        }
        assert_eq!(actual, expected);
        // Histories must agree as well when switching back to a long transform.
        let next = |s: &mut CeltSynthesis| {
            s.synthesize(
                SpectralFrame {
                    layout: long_layout,
                    transient: false,
                    normalized: &channels,
                    log_amplitudes: &[0.0; 21],
                },
                None,
            )
            .unwrap()
        };
        assert_eq!(next(&mut packed), next(&mut separate));
    }

    fn cosine_inverse(input: &[f64]) -> Vec<f64> {
        let n = input.len();
        (0..2 * n)
            .map(|time| {
                0.5 * input
                    .iter()
                    .enumerate()
                    .map(|(bin, value)| {
                        value
                            * (PI / n as f64
                                * (time as f64 + 0.5 + n as f64 / 2.0)
                                * (bin as f64 + 0.5))
                                .cos()
                    })
                    .sum::<f64>()
            })
            .collect()
    }

    #[test]
    fn low_overlap_window_is_power_complementary() {
        let rise = rising_window();
        for n in [120, 240, 480, 960] {
            let padding = (n - OVERLAP) / 2;
            let mut window = vec![0.0; 2 * n];
            window[padding..padding + OVERLAP].copy_from_slice(&rise);
            window[padding + OVERLAP..padding + n].fill(1.0);
            for i in 0..OVERLAP {
                window[padding + n + i] = rise[OVERLAP - 1 - i];
            }
            for i in 0..n {
                assert!(
                    (window[i] * window[i] + window[n + i] * window[n + i] - 1.0).abs() < 1e-14
                );
            }
        }
    }

    #[test]
    fn long_and_short_pipeline_matches_direct_cosine_overlap_and_filter_math() {
        for config in 28..=31 {
            let layout = Layout::from_configuration(config).unwrap().unwrap();
            for transient in [false, true] {
                if layout.blocks(transient).is_err() {
                    continue;
                }
                let n = layout.samples();
                let coded = layout.band(20).unwrap().end;
                let left: Vec<_> = (0..coded)
                    .map(|i| ((i * 13 % 37) as f64 - 18.0) / 100.0)
                    .collect();
                let right = vec![0.0; coded];
                let input = [&left[..], &right[..]];
                let blocks = layout.blocks(transient).unwrap();
                let block_n = n / blocks;
                let padding = (block_n - OVERLAP) / 2;
                let window = rising_window();
                let mut tail = [0.0; OVERLAP];
                let mut history = 0.0;
                let mut synth = CeltSynthesis::new(2).unwrap();
                for _ in 0..3 {
                    let output = synth
                        .synthesize(
                            SpectralFrame {
                                layout,
                                transient,
                                normalized: &input,
                                log_amplitudes: &[0.0; 42],
                            },
                            None,
                        )
                        .unwrap();
                    for block in 0..blocks {
                        let coefficients: Vec<_> = (0..block_n)
                            .map(|i| left.get(i * blocks + block).copied().unwrap_or(0.0))
                            .collect();
                        let inverse = cosine_inverse(&coefficients);
                        for time in 0..block_n {
                            let weight = if time < OVERLAP { window[time] } else { 1.0 };
                            let old = if time < OVERLAP { tail[time] } else { 0.0 };
                            history = inverse[padding + time] * weight
                                + old
                                + (27853.0 / 32768.0) * history;
                            assert!((output[2 * (block * block_n + time)] - history).abs() < 1e-9);
                            assert_eq!(output[2 * (block * block_n + time) + 1], 0.0);
                        }
                        for time in 0..OVERLAP {
                            tail[time] =
                                inverse[padding + block_n + time] * window[OVERLAP - 1 - time];
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn pitch_filter_inverts_independent_crossfaded_prefilter() {
        let window = rising_window();
        for period in [15, 31, 120, 1022] {
            for tapset in 0..=2 {
                let parameters = PitchParameters::new(period, 7, tapset).unwrap();
                let [g0, g1, g2] = parameters.coefficients();
                let signal: Vec<_> = (0..2880)
                    .map(|i| ((i as f64 * 0.17).sin() + 0.3 * (i as f64 * 0.37).cos(), 0.0))
                    .collect();
                let mut filter = PitchFilter::new(2);
                for (frame_index, enabled) in [true, false, true].into_iter().enumerate() {
                    let mut filtered = vec![0.0; 1920];
                    for time in 0..960 {
                        let absolute = frame_index * 960 + time;
                        let delayed =
                            |delay: usize| absolute.checked_sub(delay).map_or(0.0, |i| signal[i].0);
                        let prediction = g0 * delayed(period as usize)
                            + g1 * (delayed(period as usize - 1) + delayed(period as usize + 1))
                            + g2 * (delayed(period as usize - 2) + delayed(period as usize + 2));
                        let weight = if time < OVERLAP {
                            0.0
                        } else if time < 2 * OVERLAP {
                            window[time - OVERLAP].powi(2)
                        } else {
                            1.0
                        };
                        let previous = frame_index == 1;
                        let mix = if enabled { weight } else { 0.0 }
                            + if previous { 1.0 - weight } else { 0.0 };
                        filtered[2 * time] = signal[absolute].0 - mix * prediction;
                    }
                    filter
                        .process(&mut filtered, enabled.then_some(parameters), &window)
                        .unwrap();
                    for time in 0..960 {
                        assert!(
                            (filtered[2 * time] - signal[frame_index * 960 + time].0).abs() < 1e-12
                        );
                        assert_eq!(filtered[2 * time + 1], 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_frame_does_not_commit_histories_and_reset_restores_startup() {
        let layout = Layout::from_configuration(31).unwrap().unwrap();
        let input = [vec![1.0; 800]];
        let channels = [&input[0][..]];
        let mut synthesis = CeltSynthesis::new(1).unwrap();
        let mut fresh = CeltSynthesis::new(1).unwrap();
        assert!(
            synthesis
                .synthesize(
                    SpectralFrame {
                        layout,
                        transient: false,
                        normalized: &channels,
                        log_amplitudes: &[2000.0; 21]
                    },
                    None
                )
                .is_err()
        );
        let run = |s: &mut CeltSynthesis| {
            s.synthesize(
                SpectralFrame {
                    layout,
                    transient: false,
                    normalized: &channels,
                    log_amplitudes: &[0.0; 21],
                },
                None,
            )
            .unwrap()
        };
        assert_eq!(run(&mut synthesis), run(&mut fresh));
        synthesis.reset();
        fresh.reset();
        assert_eq!(run(&mut synthesis), run(&mut fresh));
        let hybrid = Layout::from_configuration(15).unwrap().unwrap();
        assert!(matches!(
            synthesis.synthesize(
                SpectralFrame {
                    layout: hybrid,
                    transient: false,
                    normalized: &channels,
                    log_amplitudes: &[0.0; 21]
                },
                None
            ),
            Err(MediaDecodeError::Unsupported)
        ));
    }
}
