//! CELT coarse reconstruction and fine energy stages, RFC 6716 section 4.3.2.
//!
//! Coarse models/prediction parameters and exact fine allocation must be
//! supplied by earlier stages. This module cannot decode a CELT frame by itself.
//!
//! Fixed 20-ms packet models and band means are established from RFC 6716's
//! normative Appendix A. The independent predictor algebra follows section
//! 4.3.2.1; integer Laplace decoding and budget gates follow the normative
//! specification. Other durations still require their own packet entry path.
//! https://www.rfc-editor.org/rfc/rfc6716.html#section-4.3.2.1
//! https://jmvalin.ca/papers/aes135_opus_celt.pdf

use super::{celt::Layout, invalid, range::RangeDecoder};
use crate::video::backend::MediaDecodeError;

// RFC 6716 Appendix A, quant_bands.c: e_prob_model[3][inter/intra].
const COARSE_20MS: [[u8; 42]; 2] = [
    [
        42, 121, 96, 66, 108, 43, 111, 40, 117, 44, 123, 32, 120, 36, 119, 33, 127, 33, 134, 34,
        139, 21, 147, 23, 152, 20, 158, 25, 154, 26, 166, 21, 173, 16, 184, 13, 184, 10, 150, 13,
        139, 15,
    ],
    [
        22, 178, 63, 114, 74, 82, 84, 83, 92, 82, 103, 62, 96, 72, 96, 67, 101, 73, 107, 72, 113,
        55, 118, 52, 125, 52, 118, 52, 117, 55, 135, 49, 137, 39, 157, 32, 145, 29, 97, 33, 77, 40,
    ],
];

/// Absolute log-amplitude means: RFC 6716 Appendix A, float eMeans initializer.
pub const BAND_MEANS: [f64; 21] = [
    6.4375, 6.25, 5.75, 5.3125, 5.0625, 4.8125, 4.5, 4.375, 4.875, 4.6875, 4.5625, 4.4375, 4.875,
    4.625, 4.3125, 4.5, 4.375, 4.625, 4.75, 4.4375, 3.75,
];

/// Fixed 20-ms CELT coarse syntax, including the normative budget fallbacks.
/// Unlike decode_coarse, this consumes an actual packet without supplied PDFs.
pub fn decode_coarse_20ms(
    decoder: &mut RangeDecoder<'_>,
    layout: Layout,
    channels: usize,
    intra: bool,
    previous: &[f64],
) -> Result<Vec<f64>, MediaDecodeError> {
    if layout.samples() != 960 || layout.bands().start != 0 {
        return Err(MediaDecodeError::Unsupported);
    }
    if !(1..=2).contains(&channels)
        || previous.len() != 21 * channels
        || previous.iter().any(|value| !value.is_finite())
    {
        return Err(invalid("invalid CELT packet energy history"));
    }
    let mut cursor = decoder.clone();
    let budget = cursor.frame_bytes() as u64 * 8;
    let mut errors = Vec::with_capacity(previous.len());
    let probabilities = &COARSE_20MS[usize::from(intra)];
    for pair in probabilities.chunks_exact(2) {
        for _ in 0..channels {
            let available = budget.saturating_sub(cursor.tell());
            let error = match available {
                15.. => cursor.laplace(u16::from(pair[0]) * 128, u16::from(pair[1]) * 64)?,
                2..=14 => [0, -1, 1][cursor.inverse_cdf(&[2, 1, 0], 2)?],
                1 => -i16::from(cursor.bit(1)?),
                _ => -1,
            };
            errors.push(error);
        }
    }
    // The float normative path has no output floor; -28 is fixed-point only.
    let prediction = CoarsePrediction {
        alpha: if intra { 0.0 } else { 0.5 },
        beta: if intra { 4915.0 } else { 6554.0 } / 32768.0,
        history_floor: -9.0,
        energy_floor: -f64::MAX,
    };
    let energies = reconstruct_coarse(layout, channels, previous, &errors, prediction)?;
    *decoder = cursor;
    Ok(energies)
}

/// Exact mode-dependent prediction parameters. Section 4.3.2.1 specifies the
/// filter, but does not tabulate the inter-frame coefficients or clamp limits.
/// Callers must supply independently established values, not estimates.
#[derive(Clone, Copy, Debug)]
pub struct CoarsePrediction {
    pub alpha: f64,
    pub beta: f64,
    pub history_floor: f64,
    pub energy_floor: f64,
}

impl CoarsePrediction {
    /// Intra disables temporal prediction; beta is given in section 4.3.2.1.
    pub fn intra(history_floor: f64, energy_floor: f64) -> Self {
        Self {
            alpha: 0.0,
            beta: 4915.0 / 32768.0,
            history_floor,
            energy_floor,
        }
    }

    fn validate(self) -> Result<(), MediaDecodeError> {
        if !self.alpha.is_finite()
            || !self.beta.is_finite()
            || !(0.0..1.0).contains(&self.alpha)
            || !(0.0..1.0).contains(&self.beta)
            || !self.history_floor.is_finite()
            || !self.energy_floor.is_finite()
        {
            return Err(invalid("invalid CELT coarse prediction parameters"));
        }
        Ok(())
    }
}

/// Integer intervals for one band's signed prediction errors, in wire order.
/// This accepts exact expanded probability models, not a fitted Laplace PDF.
/// The normative model parameters are not supplied by RFC 6716's prose.
pub struct CoarseModel<'a> {
    pub frequencies: &'a [u16],
    pub residuals: &'a [i16],
}

/// Invert A(z_l,z_b)=(1-alpha*z_l^-1)(1-z_b^-1)/(1-beta*z_b^-1).
/// Inputs and output are band-major/channel-minor log amplitudes without means.
/// `previous` must be the preceding frame's FINAL refined energies, not coarse
/// energies. Frequency prediction uses only this frame's integer residuals.
pub fn reconstruct_coarse(
    layout: Layout,
    channels: usize,
    previous: &[f64],
    residuals: &[i16],
    prediction: CoarsePrediction,
) -> Result<Vec<f64>, MediaDecodeError> {
    prediction.validate()?;
    if !(1..=2).contains(&channels) {
        return Err(invalid("invalid CELT coarse channel count"));
    }
    let count = layout.bands().len() * channels;
    if previous.len() != count
        || residuals.len() != count
        || previous.iter().any(|value| !value.is_finite())
    {
        return Err(invalid("invalid CELT coarse energy input"));
    }
    let mut frequency = [0.0; 2];
    let mut energies = Vec::with_capacity(count);
    for (index, (&old, &residual)) in previous.iter().zip(residuals).enumerate() {
        let channel = index % channels;
        let error = f64::from(residual);
        let predicted = prediction.alpha * old.max(prediction.history_floor);
        let energy = predicted + frequency[channel] + error;
        if !energy.is_finite() {
            return Err(invalid("CELT coarse energy overflow"));
        }
        energies.push(energy.max(prediction.energy_floor));
        // The numerator (1-beta*z_b^-1) leaves (1-beta) of each previous
        // residual in the running frequency predictor. Fine bits never enter it.
        frequency[channel] += (1.0 - prediction.beta) * error;
    }
    Ok(energies)
}

/// Consume supplied exact coarse models immediately after the intra flag.
/// Small-budget fallback models must be selected by the caller; no default
/// model or mode-dependent coefficient is guessed here. Errors are transactional.
pub fn decode_coarse(
    decoder: &mut RangeDecoder<'_>,
    layout: Layout,
    channels: usize,
    previous: &[f64],
    prediction: CoarsePrediction,
    models: &[CoarseModel<'_>],
) -> Result<Vec<f64>, MediaDecodeError> {
    prediction.validate()?;
    if !(1..=2).contains(&channels)
        || previous.len() != layout.bands().len() * channels
        || previous.iter().any(|value| !value.is_finite())
        || models.len() != layout.bands().len()
        || models.iter().any(|model| {
            model.frequencies.is_empty() || model.frequencies.len() != model.residuals.len()
        })
    {
        return Err(invalid("invalid CELT coarse probability models"));
    }
    let mut cursor = decoder.clone();
    let mut residuals = Vec::with_capacity(previous.len());
    for model in models {
        for _ in 0..channels {
            let symbol = cursor.symbol(model.frequencies)?;
            if cursor.tell() > cursor.frame_bytes() as u64 * 8 {
                return Err(invalid("CELT coarse energy exceeds frame budget"));
            }
            residuals.push(model.residuals[symbol]);
        }
    }
    let energies = reconstruct_coarse(layout, channels, previous, &residuals, prediction)?;
    *decoder = cursor;
    Ok(energies)
}

/// Energies are in base-2 log units, band-major then channel-major, restricted
/// to the active bands. These are not PCM amplitudes or denormalized spectra.
#[derive(Clone, Debug)]
pub struct FineEnergy {
    layout: Layout,
    channels: usize,
    log_energies: Vec<f64>,
    fine_bits: Vec<u8>,
    finalized: bool,
}

impl FineEnergy {
    /// Consume each band's allocated raw bits for each channel in coding order.
    /// Errors leave both the entropy cursor and caller's coarse values intact.
    pub fn decode(
        decoder: &mut RangeDecoder<'_>,
        layout: Layout,
        channels: usize,
        coarse: &[f64],
        fine_bits: &[u8],
    ) -> Result<Self, MediaDecodeError> {
        let bands = layout.bands().len();
        if !(1..=2).contains(&channels)
            || coarse.len() != bands * channels
            || fine_bits.len() != bands
            || coarse.iter().any(|value| !value.is_finite())
            || fine_bits.iter().any(|&bits| bits > 32)
        {
            return Err(invalid("invalid CELT fine energy input"));
        }
        let mut cursor = decoder.clone();
        let mut log_energies = coarse.to_vec();
        for (&bits, band) in fine_bits
            .iter()
            .zip(log_energies.chunks_exact_mut(channels))
        {
            if bits == 0 {
                continue;
            }
            let steps = (1u64 << bits) as f64;
            for energy in band {
                let symbol = cursor.raw_bits(bits)?;
                *energy += (f64::from(symbol) + 0.5) / steps - 0.5;
            }
        }
        *decoder = cursor;
        Ok(Self {
            layout,
            channels,
            log_energies,
            fine_bits: fine_bits.to_vec(),
            finalized: false,
        })
    }

    pub fn log_energies(&self) -> &[f64] {
        &self.log_energies
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Called after shape and all other flags. `bits_left` is the caller's
    /// remaining frame budget; priority 0 bands precede priority 1 bands.
    /// The caller must mark bands ineligible when allocation limits forbid
    /// another refinement. Returns the actual number of raw bits consumed.
    pub fn finalize(
        &mut self,
        decoder: &mut RangeDecoder<'_>,
        priorities: &[u8],
        eligible: &[bool],
        bits_left: usize,
    ) -> Result<usize, MediaDecodeError> {
        if self.finalized
            || priorities.len() != self.fine_bits.len()
            || eligible.len() != self.fine_bits.len()
            || priorities.iter().any(|&priority| priority > 1)
        {
            return Err(invalid("invalid CELT final energy allocation"));
        }
        let mut cursor = decoder.clone();
        let mut energies = self.log_energies.clone();
        let mut consumed = 0;
        for priority in 0..=1 {
            for (band, &band_priority) in priorities.iter().enumerate() {
                if band_priority != priority
                    || !eligible[band]
                    || bits_left - consumed < self.channels
                {
                    continue;
                }
                let step = 2.0f64.powi(-i32::from(self.fine_bits[band]) - 1);
                for channel in 0..self.channels {
                    let bit = cursor.raw_bits(1)?;
                    energies[band * self.channels + channel] += (f64::from(bit) - 0.5) * step;
                }
                consumed += self.channels;
            }
        }
        self.log_energies = energies;
        self.finalized = true;
        *decoder = cursor;
        Ok(consumed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coarse_reconstruction_inverts_two_dimensional_filter_per_channel() {
        let layout = Layout::from_configuration(31).unwrap().unwrap();
        let previous: Vec<_> = (0..42).map(|i| (i % 9) as f64 / 4.0).collect();
        let residuals: Vec<_> = (0..42).map(|i| (i % 7) as i16 - 3).collect();
        let prediction = CoarsePrediction {
            alpha: 0.5,
            beta: 0.25,
            history_floor: -100.0,
            energy_floor: -100.0,
        };
        let energies = reconstruct_coarse(layout, 2, &previous, &residuals, prediction).unwrap();
        // Apply the forward filter, independently of the running-sum inverse.
        for i in 0..42 {
            let mut difference = energies[i] - prediction.alpha * previous[i];
            if i >= 2 {
                difference -= energies[i - 2] - prediction.alpha * previous[i - 2];
                difference += prediction.beta * f64::from(residuals[i - 2]);
            }
            assert_eq!(difference, f64::from(residuals[i]));
        }
    }

    #[test]
    fn intra_discards_previous_frame_and_clamps_do_not_feed_frequency_history() {
        let layout = Layout::from_configuration(31).unwrap().unwrap();
        let mut residuals = [0i16; 42];
        residuals[..4].copy_from_slice(&[-20, 2, 10, -1]);
        let prediction = CoarsePrediction::intra(-9.0, -28.0);
        let first = reconstruct_coarse(layout, 2, &[-99.0; 42], &residuals, prediction).unwrap();
        let second = reconstruct_coarse(layout, 2, &[99.0; 42], &residuals, prediction).unwrap();
        assert_eq!(first, second);
        let gain = 1.0 - 4915.0 / 32768.0;
        assert_eq!(first[0], -20.0);
        assert_eq!(first[1], 2.0);
        assert_eq!(first[2], 10.0 - 20.0 * gain);
        assert_eq!(first[3], -1.0 + 2.0 * gain);
        let mut clamped = prediction;
        clamped.energy_floor = -5.0;
        let third = reconstruct_coarse(layout, 2, &[0.0; 42], &residuals, clamped).unwrap();
        assert_eq!(third[0], -5.0);
        assert_eq!(third[4], (-10.0 * gain).max(-5.0));
    }

    #[test]
    fn coarse_models_consume_band_major_symbols_and_connect_to_fine_energy() {
        let layout = Layout::from_configuration(31).unwrap().unwrap();
        let bytes = [0x96; 64];
        let frequencies = [3, 5, 2];
        let values = [0, -1, 1];
        let models: Vec<_> = (0..21)
            .map(|_| CoarseModel {
                frequencies: &frequencies,
                residuals: &values,
            })
            .collect();
        let mut actual = RangeDecoder::new(&bytes);
        let mut expected = actual.clone();
        let mut residuals = Vec::new();
        for _ in 0..42 {
            residuals.push(values[expected.symbol(&frequencies).unwrap()]);
        }
        let prediction = CoarsePrediction::intra(-100.0, -100.0);
        let coarse =
            decode_coarse(&mut actual, layout, 2, &[0.0; 42], prediction, &models).unwrap();
        assert_eq!(actual.tell_fractional(), expected.tell_fractional());
        assert_eq!(
            coarse,
            reconstruct_coarse(layout, 2, &[0.0; 42], &residuals, prediction).unwrap()
        );
        let fine = FineEnergy::decode(&mut actual, layout, 2, &coarse, &[1; 21]).unwrap();
        for (index, &energy) in fine.log_energies().iter().enumerate() {
            let midpoint = f64::from(expected.raw_bits(1).unwrap()) * 0.5 - 0.25;
            assert_eq!(energy, coarse[index] + midpoint);
        }
        assert_eq!(actual.tell_fractional(), expected.tell_fractional());
    }

    #[test]
    fn coarse_failure_preserves_entropy_and_validates_prediction() {
        let layout = Layout::from_configuration(31).unwrap().unwrap();
        let models: Vec<_> = (0..21)
            .map(|_| CoarseModel {
                frequencies: &[1, 1],
                residuals: &[0, 1],
            })
            .collect();
        let mut decoder = RangeDecoder::new(&[0]);
        let original = decoder.tell_fractional();
        let prediction = CoarsePrediction::intra(-9.0, -28.0);
        assert!(decode_coarse(&mut decoder, layout, 2, &[0.0; 42], prediction, &models).is_err());
        assert_eq!(decoder.tell_fractional(), original);
        let mut bad = prediction;
        bad.alpha = f64::NAN;
        assert!(reconstruct_coarse(layout, 2, &[0.0; 42], &[0; 42], bad).is_err());
        let mut history = [0.0; 42];
        history[0] = f64::INFINITY;
        assert!(reconstruct_coarse(layout, 2, &history, &[0; 42], prediction).is_err());
    }

    #[test]
    fn fine_energy_matches_midpoints_exhaustively() {
        let layout = Layout::from_configuration(31).unwrap().unwrap();
        for bits in 1..=8 {
            for symbol in 0..1u32 << bits {
                let bytes = symbol.to_le_bytes();
                // Raw fields run from the last byte towards the first.
                let bytes = [bytes[3], bytes[2], bytes[1], bytes[0]];
                let mut decoder = RangeDecoder::new(&bytes);
                let mut allocation = [0; 21];
                allocation[0] = bits;
                let energies =
                    FineEnergy::decode(&mut decoder, layout, 1, &[2.0; 21], &allocation).unwrap();
                let lower = 1.5 + f64::from(symbol) / f64::from(1u32 << bits);
                let upper = 1.5 + f64::from(symbol + 1) / f64::from(1u32 << bits);
                assert_eq!(energies.log_energies()[0], (lower + upper) / 2.0);
                assert_eq!(&energies.log_energies()[1..], &[2.0; 20]);
                assert_eq!(decoder.tell(), 1 + u64::from(bits));
            }
        }
    }

    #[test]
    fn final_energy_respects_priority_channel_order_and_whole_band_budget() {
        let layout = Layout::from_configuration(31).unwrap().unwrap();
        let mut decoder = RangeDecoder::new(&[0b0000_1001]);
        let mut energy = FineEnergy::decode(&mut decoder, layout, 2, &[0.0; 42], &[0; 21]).unwrap();
        let mut priority = [1; 21];
        priority[1] = 0;
        let mut eligible = [false; 21];
        eligible[0] = true;
        eligible[1] = true;
        assert_eq!(
            energy
                .finalize(&mut decoder, &priority, &eligible, 5)
                .unwrap(),
            4
        );
        assert_eq!(&energy.log_energies()[..4], &[-0.25, 0.25, 0.25, -0.25]);
        assert_eq!(&energy.log_energies()[4..], &[0.0; 38]);
        assert!(
            energy
                .finalize(&mut decoder, &priority, &eligible, 5)
                .is_err()
        );
    }

    #[test]
    fn refinement_combines_into_a_single_extra_midpoint_bit() {
        let layout = Layout::from_configuration(31).unwrap().unwrap();
        for bits in 0..=6 {
            for symbol in 0..1u32 << bits {
                for extra in 0..=1 {
                    let packed = symbol | extra << bits;
                    let bytes = [packed as u8];
                    let mut decoder = RangeDecoder::new(&bytes);
                    let mut allocation = [0; 21];
                    allocation[0] = bits;
                    let mut energy =
                        FineEnergy::decode(&mut decoder, layout, 1, &[0.0; 21], &allocation)
                            .unwrap();
                    let mut eligible = [false; 21];
                    eligible[0] = true;
                    energy
                        .finalize(&mut decoder, &[0; 21], &eligible, 1)
                        .unwrap();
                    let expected =
                        (f64::from(2 * symbol + extra) + 0.5) / f64::from(1u32 << (bits + 1)) - 0.5;
                    assert_eq!(energy.log_energies()[0], expected);
                }
            }
        }
    }

    #[test]
    fn truncation_is_transactional_in_both_stages() {
        let layout = Layout::from_configuration(31).unwrap().unwrap();
        let mut decoder = RangeDecoder::new(&[255]);
        let mut allocation = [0; 21];
        allocation[0] = 5;
        assert!(FineEnergy::decode(&mut decoder, layout, 2, &[0.0; 42], &allocation).is_err());
        assert_eq!(decoder.tell(), 1);
        let mut energy = FineEnergy::decode(&mut decoder, layout, 2, &[0.0; 42], &[0; 21]).unwrap();
        assert!(
            energy
                .finalize(&mut decoder, &[0; 21], &[true; 21], 42)
                .is_err()
        );
        assert_eq!(energy.log_energies(), &[0.0; 42]);
        assert!(!energy.finalized);
        assert_eq!(decoder.raw_bits(8).unwrap(), 255);
    }
}
