//! Connected fine-energy -> unsplit residual -> final-energy -> synthesis path.
//! RFC 6716 sections 4.3.2.2, 4.3.4.2/3 and 4.3.6/7.
//! This is not an allocation substitute: exact pulses/fine bits must be supplied
//! by the still-incomplete upstream allocation. Zero-pulse folding, splitting,
//! coupled stereo and enabled anti-collapse remain explicit
//! Unsupported cases, never zero-filled spectral bands.

use super::{
    celt::Layout,
    energy::FineEnergy,
    invalid,
    pvq::{Codebook, spread},
    range::RangeDecoder,
    synthesis::{CeltSynthesis, PitchParameters},
    tf::BandTransform,
};
use crate::video::backend::MediaDecodeError;

/// Reconstruct normalized channels from decoded normalized mid/side shapes.
/// Valin et al., AES 135 (2013), section 4.5.1, equations (9)/(10), and the
/// renormalization paragraph immediately below them:
/// https://jmvalin.ca/papers/aes135_opus_celt.pdf
/// This is residual reconstruction only, NOT an angle entropy decoder: the
/// angle's exact quantization and allocation rules remain unresolved.
pub fn uncouple_mid_side(
    mid: &mut [f64],
    side: &mut [f64],
    angle: f64,
) -> Result<(), MediaDecodeError> {
    if mid.is_empty()
        || mid.len() > 176
        || mid.len() != side.len()
        || !angle.is_finite()
        || !(0.0..=std::f64::consts::FRAC_PI_2).contains(&angle)
        || mid
            .iter()
            .chain(side.iter())
            .any(|value| !value.is_finite())
    {
        return Err(invalid("invalid CELT mid-side reconstruction"));
    }
    let (sine, cosine) = if angle == 0.0 {
        (0.0, 1.0)
    } else if angle == std::f64::consts::FRAC_PI_2 {
        (1.0, 0.0)
    } else {
        angle.sin_cos()
    };
    let mut left = Vec::with_capacity(mid.len());
    let mut right = Vec::with_capacity(mid.len());
    for (&m, &s) in mid.iter().zip(side.iter()) {
        left.push(m * cosine + s * sine);
        right.push(m * cosine - s * sine);
    }
    // Quantized shapes need not be orthogonal. Normalize each channel, not
    // their combined energy, so separately transmitted L/R energies survive.
    for channel in [&mut left, &mut right] {
        let norm = channel
            .iter()
            .fold(0.0f64, |norm, &value| norm.hypot(value));
        if !norm.is_finite() || norm == 0.0 {
            return Err(invalid("degenerate CELT mid-side reconstruction"));
        }
        for value in channel.iter_mut() {
            *value /= norm;
        }
    }
    mid.copy_from_slice(&left);
    side.copy_from_slice(&right);
    Ok(())
}

/// Complete upstream decisions for the restricted unsplit,
/// independent-channel path. Arrays follow active bands, with pulses ordered
/// band-major/channel-minor. Exact mean offsets are mandatory.
pub struct UnsplitPlan<'a> {
    pub layout: Layout,
    pub transient: bool,
    pub coupled_stereo: bool,
    pub tf_adjustments: &'a [i8],
    pub coarse_log_amplitudes: &'a [f64],
    pub mean_log_amplitudes: &'a [f64],
    pub fine_bits: &'a [u8],
    pub pulses: &'a [u16],
    pub spread: u8,
    pub final_priorities: &'a [u8],
    pub final_eligible: &'a [bool],
    pub final_bits: usize,
    pub anti_collapse_reserved: bool,
    pub pitch: Option<PitchParameters>,
}

pub struct ResidualDecoder {
    channels: usize,
    synthesis: CeltSynthesis,
}

impl ResidualDecoder {
    pub fn new(channels: usize) -> Result<Self, MediaDecodeError> {
        Ok(Self {
            channels,
            synthesis: CeltSynthesis::new(channels)?,
        })
    }

    pub fn reset(&mut self) {
        self.synthesis.reset();
    }

    /// `entropy` must be at fine energy: coarse, TF, spread and allocation have
    /// already been decoded. Returns time samples in the exact supplied gains'
    /// amplitude scale, with no container trim or gain. Entropy and synthesis
    /// histories commit only when the connected path successfully completes.
    pub fn decode(
        &mut self,
        entropy: &mut RangeDecoder<'_>,
        plan: UnsplitPlan<'_>,
    ) -> Result<Vec<f64>, MediaDecodeError> {
        let bands = plan.layout.bands();
        let count = bands.len();
        if plan.coupled_stereo || plan.layout.bands().start != 0 {
            return Err(MediaDecodeError::Unsupported);
        }
        plan.layout.blocks(plan.transient)?;
        if plan.anti_collapse_reserved && (!plan.transient || plan.layout.samples() < 480) {
            return Err(invalid("invalid CELT anti-collapse reservation"));
        }
        if plan.tf_adjustments.len() != count
            || plan.mean_log_amplitudes.len() != count
            || plan.pulses.len() != count * self.channels
            || plan.fine_bits.len() != count
            || plan.final_priorities.len() != count
            || plan.final_eligible.len() != count
            || plan.final_priorities.iter().any(|&p| p > 1)
            || plan.spread > 3
            || plan.mean_log_amplitudes.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("invalid CELT unsplit residual plan"));
        }
        if plan.pulses.contains(&0) {
            return Err(MediaDecodeError::Unsupported);
        }
        let transforms = bands
            .clone()
            .enumerate()
            .map(|(index, band)| {
                BandTransform::new(
                    plan.layout,
                    band,
                    plan.transient,
                    plan.tf_adjustments[index],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut books = Vec::with_capacity(plan.pulses.len());
        for (band_index, band) in bands.clone().enumerate() {
            let width = plan.layout.band(band)?.len();
            for channel in 0..self.channels {
                books.push(Codebook::new(
                    width,
                    usize::from(plan.pulses[band_index * self.channels + channel]),
                )?);
            }
        }
        let mut cursor = entropy.clone();
        let mut energy = FineEnergy::decode(
            &mut cursor,
            plan.layout,
            self.channels,
            plan.coarse_log_amplitudes,
            plan.fine_bits,
        )?;
        let coded = plan.layout.band(bands.end - 1)?.end;
        let mut normalized = vec![vec![0.0; coded]; self.channels];
        let capacity = cursor.frame_bytes() as u64 * 8;
        for (band_index, band) in bands.clone().enumerate() {
            let bins = plan.layout.band(band)?;
            for channel in 0..self.channels {
                let index = band_index * self.channels + channel;
                let shape = &mut normalized[channel][bins.clone()];
                books[index].decode_shape(&mut cursor, shape)?;
                if cursor.tell() > capacity {
                    return Err(invalid("CELT residual exceeds frame budget"));
                }
                spread(
                    shape,
                    u32::from(plan.pulses[index]),
                    transforms[band_index].coded_blocks(),
                    plan.spread,
                )?;
                transforms[band_index].inverse(shape)?;
            }
        }
        if plan.anti_collapse_reserved && cursor.bit(1)? {
            return Err(MediaDecodeError::Unsupported);
        }
        let remaining = capacity.saturating_sub(cursor.tell());
        if plan.final_bits as u64 > remaining {
            return Err(invalid("CELT final energy exceeds frame budget"));
        }
        energy.finalize(
            &mut cursor,
            plan.final_priorities,
            plan.final_eligible,
            plan.final_bits,
        )?;
        let channels: Vec<_> = normalized.iter().map(Vec::as_slice).collect();
        let pcm = self.synthesis.synthesize_refined(
            plan.transient,
            &channels,
            &energy,
            plan.mean_log_amplitudes,
            plan.pitch,
        )?;
        *entropy = cursor;
        Ok(pcm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mid_side_equations_recover_independent_normalized_channels() {
        for width in [2, 8, 16, 176] {
            let mut left: Vec<_> = (0..width).map(|i| (i as f64 + 0.5).sin()).collect();
            let mut right: Vec<_> = (0..width).map(|i| (i as f64 * 1.7 + 0.2).cos()).collect();
            for channel in [&mut left, &mut right] {
                let norm = channel.iter().map(|v| v * v).sum::<f64>().sqrt();
                for value in channel.iter_mut() {
                    *value /= norm;
                }
            }
            let mut mid: Vec<_> = left
                .iter()
                .zip(&right)
                .map(|(l, r)| (l + r) / 2.0)
                .collect();
            let mut side: Vec<_> = left
                .iter()
                .zip(&right)
                .map(|(l, r)| (l - r) / 2.0)
                .collect();
            let mid_norm = mid.iter().map(|v| v * v).sum::<f64>().sqrt();
            let side_norm = side.iter().map(|v| v * v).sum::<f64>().sqrt();
            for value in &mut mid {
                *value /= mid_norm;
            }
            for value in &mut side {
                *value /= side_norm;
            }
            uncouple_mid_side(&mut mid, &mut side, side_norm.atan2(mid_norm)).unwrap();
            for (&actual, &expected) in mid.iter().zip(&left).chain(side.iter().zip(&right)) {
                assert!((actual - expected).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn quantized_nonorthogonal_shapes_are_renormalized_per_channel() {
        let mut mid = [1.0, 0.0];
        let mut side = [0.6, 0.8];
        let angle = std::f64::consts::PI / 6.0;
        let expected_left = [angle.cos() + 0.6 * angle.sin(), 0.8 * angle.sin()];
        let expected_right = [angle.cos() - 0.6 * angle.sin(), -0.8 * angle.sin()];
        uncouple_mid_side(&mut mid, &mut side, angle).unwrap();
        for (actual, expected) in [(mid, expected_left), (side, expected_right)] {
            let norm = expected.iter().map(|v| v * v).sum::<f64>().sqrt();
            for (a, e) in actual.into_iter().zip(expected) {
                assert!((a - e / norm).abs() < 1e-15);
            }
            assert!((actual.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn mid_side_endpoints_and_failure_do_not_fabricate_channels() {
        for (angle, mid, side, expected_left, expected_right) in [
            (0.0, [0.6, 0.8], [0.0, 0.0], [0.6, 0.8], [0.6, 0.8]),
            (
                std::f64::consts::FRAC_PI_2,
                [0.0, 0.0],
                [0.6, 0.8],
                [0.6, 0.8],
                [-0.6, -0.8],
            ),
        ] {
            let (mut mid, mut side) = (mid, side);
            uncouple_mid_side(&mut mid, &mut side, angle).unwrap();
            assert_eq!(mid, expected_left);
            assert_eq!(side, expected_right);
        }
        let (mut mid, mut side) = ([0.0; 2], [0.0; 2]);
        assert!(uncouple_mid_side(&mut mid, &mut side, 0.0).is_err());
        assert_eq!((mid, side), ([0.0; 2], [0.0; 2]));
        let (mut mid, mut side) = ([1.0, 0.0], [0.0, 1.0]);
        assert!(uncouple_mid_side(&mut mid, &mut side, f64::NAN).is_err());
        assert_eq!((mid, side), ([1.0, 0.0], [0.0, 1.0]));
    }

    #[test]
    fn connected_residual_path_matches_independent_stage_composition() {
        let layout = Layout::from_configuration(19).unwrap().unwrap();
        let count = layout.bands().len();
        let coarse = vec![0.0; count];
        let means = vec![2.0; count];
        let fine = vec![2; count];
        let pulses = vec![1; count];
        let priorities = vec![0; count];
        let eligible = vec![true; count];
        let tf = vec![0; count];
        let bytes = [0x96; 128];
        let mut actual = RangeDecoder::new(&bytes);
        let mut expected = actual.clone();
        let mut decoder = ResidualDecoder::new(1).unwrap();
        let output = decoder
            .decode(
                &mut actual,
                UnsplitPlan {
                    layout,
                    transient: false,
                    coupled_stereo: false,
                    tf_adjustments: &tf,
                    coarse_log_amplitudes: &coarse,
                    mean_log_amplitudes: &means,
                    fine_bits: &fine,
                    pulses: &pulses,
                    spread: 2,
                    final_priorities: &priorities,
                    final_eligible: &eligible,
                    final_bits: count,
                    anti_collapse_reserved: false,
                    pitch: None,
                },
            )
            .unwrap();
        let mut energy = FineEnergy::decode(&mut expected, layout, 1, &coarse, &fine).unwrap();
        let coded = layout.band(count - 1).unwrap().end;
        let mut shape = vec![0.0; coded];
        for band in layout.bands() {
            let bins = layout.band(band).unwrap();
            let width = bins.len();
            // K=1 has 2*N entries, so derive its vector directly from index
            // intervals rather than calling decode_shape in the expected path.
            let index = expected.uniform((2 * width) as u32).unwrap() as usize;
            let mut vector = vec![0.0; width];
            if index < width {
                vector[index] = 1.0;
            } else {
                vector[2 * width - 1 - index] = -1.0;
            }
            spread(&mut vector, 1, 1, 2).unwrap();
            shape[bins].copy_from_slice(&vector);
        }
        energy
            .finalize(&mut expected, &priorities, &eligible, count)
            .unwrap();
        let expected = CeltSynthesis::new(1)
            .unwrap()
            .synthesize_refined(false, &[&shape], &energy, &means, None)
            .unwrap();
        assert_eq!(output, expected);
        assert!((output.iter().map(|v| v * v).sum::<f64>()) > 0.0);
    }

    #[test]
    fn configuration_31_stereo_residual_synthesizes_nonzero_tf_shapes() {
        let layout = Layout::from_configuration(31).unwrap().unwrap();
        let bytes = [0x96; 512];
        let mut tf = [1; 21];
        tf[1] = 3;
        tf[20] = -1;
        let decode = |adjustments: &[i8]| {
            let mut entropy = RangeDecoder::new(&bytes);
            let pcm = ResidualDecoder::new(2)
                .unwrap()
                .decode(
                    &mut entropy,
                    UnsplitPlan {
                        layout,
                        transient: true,
                        coupled_stereo: false,
                        tf_adjustments: adjustments,
                        coarse_log_amplitudes: &[0.0; 42],
                        mean_log_amplitudes: &[0.0; 21],
                        fine_bits: &[0; 21],
                        pulses: &[1; 42],
                        spread: 2,
                        final_priorities: &[0; 21],
                        final_eligible: &[false; 21],
                        final_bits: 0,
                        anti_collapse_reserved: false,
                        pitch: None,
                    },
                )
                .unwrap();
            (pcm, entropy.tell_fractional())
        };
        let (changed, changed_tell) = decode(&tf);
        let (unchanged, unchanged_tell) = decode(&[0; 21]);
        assert_eq!(changed.len(), 1920);
        assert!(changed.iter().all(|v| v.is_finite()));
        assert!(changed.iter().any(|&v| v != 0.0));
        assert_ne!(changed, unchanged);
        assert_eq!(changed_tell, unchanged_tell);
    }

    #[test]
    fn folding_is_unsupported_not_silence_and_leaves_entropy_untouched() {
        let layout = Layout::from_configuration(31).unwrap().unwrap();
        let mut entropy = RangeDecoder::new(&[0; 128]);
        let mut decoder = ResidualDecoder::new(2).unwrap();
        assert!(matches!(
            decoder.decode(
                &mut entropy,
                UnsplitPlan {
                    layout,
                    transient: false,
                    coupled_stereo: false,
                    tf_adjustments: &[0; 21],
                    coarse_log_amplitudes: &[0.0; 42],
                    mean_log_amplitudes: &[0.0; 21],
                    fine_bits: &[0; 21],
                    pulses: &[0; 42],
                    spread: 0,
                    final_priorities: &[0; 21],
                    final_eligible: &[true; 21],
                    final_bits: 0,
                    anti_collapse_reserved: false,
                    pitch: None,
                }
            ),
            Err(MediaDecodeError::Unsupported)
        ));
        assert_eq!(entropy.tell(), 1);
    }

    #[test]
    fn transient_shape_path_decodes_short_blocks_when_anti_collapse_is_off() {
        let layout = Layout::from_configuration(31).unwrap().unwrap();
        let coarse = [0.0; 21];
        let mut success = 0;
        let mut rejected = 0;
        for seed in 0..64u32 {
            let bytes: Vec<_> = (0..256).map(|i| (seed * 73 + i * 137) as u8).collect();
            let mut entropy = RangeDecoder::new(&bytes);
            let mut manual = entropy.clone();
            for band in layout.bands() {
                let width = layout.band(band).unwrap().len();
                manual.uniform((2 * width) as u32).unwrap();
            }
            let anti = manual.bit(1).unwrap();
            let result = ResidualDecoder::new(1).unwrap().decode(
                &mut entropy,
                UnsplitPlan {
                    layout,
                    transient: true,
                    coupled_stereo: false,
                    tf_adjustments: &[0; 21],
                    coarse_log_amplitudes: &coarse,
                    mean_log_amplitudes: &[0.0; 21],
                    fine_bits: &[0; 21],
                    pulses: &[1; 21],
                    spread: 2,
                    final_priorities: &[0; 21],
                    final_eligible: &[false; 21],
                    final_bits: 0,
                    anti_collapse_reserved: true,
                    pitch: None,
                },
            );
            if anti {
                assert!(matches!(result, Err(MediaDecodeError::Unsupported)));
                assert_eq!(entropy.tell(), 1);
                rejected += 1;
            } else {
                let pcm = result.unwrap();
                assert_eq!(pcm.len(), 960);
                assert!(pcm.iter().any(|&sample| sample != 0.0));
                assert_eq!(entropy.tell_fractional(), manual.tell_fractional());
                success += 1;
            }
        }
        assert!(success > 0 && rejected > 0);
    }
}
