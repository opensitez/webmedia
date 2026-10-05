//! AAC spectral reconstruction, from the normative specifications only.
//!
//! ISO/IEC 13818-7:2004: 10.3, 11.3.3, 12.1.3, 12.2.3, and 14.3
//! (physical PDF pages 87-93 and 107-109). MPEG-4 clarifications:
//! ISO/IEC 14496-3:2001, 4.6.8 and 4.6.9 (physical pages 577-586).
//! Inputs are already decoded and deinterleaved; no frame parser is required.

use super::invalid;
use crate::video::backend::MediaDecodeError;
use std::f64::consts::FRAC_PI_2;
use std::ops::Range;
use std::sync::OnceLock;

const MAX_SPECTRUM: usize = 1024;
const MAX_TNS_ORDER: usize = 20;

struct QuantizationTables {
    magnitudes: Box<[f64]>,
    gains: [f64; 256],
}

fn quantization_tables() -> &'static QuantizationTables {
    static TABLES: OnceLock<QuantizationTables> = OnceLock::new();
    TABLES.get_or_init(|| QuantizationTables {
        magnitudes: (0..=8191)
            .map(|q| {
                let q = f64::from(q);
                q * q.cbrt()
            })
            .collect(),
        gains: std::array::from_fn(|sf| 2.0_f64.powf((sf as f64 - 100.0) * 0.25)),
    })
}

fn validate_bands(length: usize, bands: &[Range<usize>]) -> Result<(), MediaDecodeError> {
    if length == 0 || length > MAX_SPECTRUM || bands.len() > length {
        return Err(invalid("invalid AAC spectral extent"));
    }
    let mut covered = [false; MAX_SPECTRUM];
    for band in bands {
        if band.start >= band.end || band.end > length {
            return Err(invalid("invalid AAC spectral band"));
        }
        for entry in &mut covered[band.clone()] {
            if *entry {
                return Err(invalid("overlapping AAC spectral bands"));
            }
            *entry = true;
        }
    }
    Ok(())
}

/// Bands may be unordered but must not overlap. Uncovered bins become zero.
/// Noise/intensity/zero-codebook bands should be omitted by the caller.
pub(crate) fn inverse_quantize(
    quantized: &[i32],
    output: &mut [f64],
    bands: &[Range<usize>],
    scalefactors: &[i16],
) -> Result<(), MediaDecodeError> {
    inverse_quantize_impl::<true>(quantized, output, bands, scalefactors)
}

#[cfg(test)]
pub(super) fn inverse_quantize_reference(
    quantized: &[i32],
    output: &mut [f64],
    bands: &[Range<usize>],
    scalefactors: &[i16],
) -> Result<(), MediaDecodeError> {
    inverse_quantize_impl::<false>(quantized, output, bands, scalefactors)
}

fn inverse_quantize_impl<const CACHED: bool>(
    quantized: &[i32],
    output: &mut [f64],
    bands: &[Range<usize>],
    scalefactors: &[i16],
) -> Result<(), MediaDecodeError> {
    validate_bands(quantized.len(), bands)?;
    if output.len() != quantized.len()
        || scalefactors.len() != bands.len()
        || scalefactors.iter().any(|&sf| !(0..=255).contains(&sf))
        || quantized.iter().any(|&q| !(-8191..=8191).contains(&q))
    {
        return Err(invalid("invalid AAC inverse quantization input"));
    }
    let tables = CACHED.then(quantization_tables);
    output.fill(0.0);
    for (band, &sf) in bands.iter().zip(scalefactors) {
        let gain = if let Some(tables) = tables {
            tables.gains[sf as usize]
        } else {
            2.0_f64.powf((f64::from(sf) - 100.0) * 0.25)
        };
        for index in band.clone() {
            let q = quantized[index];
            let value = if let Some(tables) = tables {
                let magnitude = tables.magnitudes[q.unsigned_abs() as usize];
                if q < 0 {
                    -magnitude
                } else {
                    magnitude
                }
            } else {
                let q = f64::from(q);
                q * q.abs().cbrt()
            };
            output[index] = value * gain;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StereoMode {
    Independent,
    MidSide,
    Intensity {
        position: i16,
        out_of_phase: bool,
        /// True only for ms_mask_present == 1 && ms_used, in non-scalable AAC.
        invert: bool,
    },
}

fn intensity_gain(mode: StereoMode) -> Result<Option<f64>, MediaDecodeError> {
    if let StereoMode::Intensity {
        position,
        out_of_phase,
        invert,
    } = mode
    {
        let magnitude = 2.0_f64.powf(-f64::from(position) * 0.25);
        if !magnitude.is_finite() || magnitude == 0.0 {
            return Err(invalid("AAC intensity gain exceeds numeric range"));
        }
        Ok(Some(if out_of_phase ^ invert {
            -magnitude
        } else {
            magnitude
        }))
    } else {
        Ok(None)
    }
}

/// Use Independent for PNS bands: their MS flag controls noise correlation,
/// not a mid/side matrix. Intensity replaces only the right channel.
pub(crate) fn apply_stereo(
    left: &mut [f64],
    right: &mut [f64],
    bands: &[Range<usize>],
    modes: &[StereoMode],
) -> Result<(), MediaDecodeError> {
    validate_bands(left.len(), bands)?;
    if left.len() != right.len()
        || modes.len() != bands.len()
        || left.iter().chain(right.iter()).any(|v| !v.is_finite())
    {
        return Err(invalid("invalid AAC stereo input"));
    }
    // Validate every result before modifying either channel.
    for (band, &mode) in bands.iter().zip(modes) {
        let gain = intensity_gain(mode)?;
        for index in band.clone() {
            let valid = match mode {
                StereoMode::Independent => true,
                StereoMode::MidSide => {
                    (left[index] + right[index]).is_finite()
                        && (left[index] - right[index]).is_finite()
                }
                StereoMode::Intensity { .. } => (left[index] * gain.unwrap()).is_finite(),
            };
            if !valid {
                return Err(invalid("AAC stereo arithmetic overflow"));
            }
        }
    }
    for (band, &mode) in bands.iter().zip(modes) {
        let gain = intensity_gain(mode)?;
        for index in band.clone() {
            match mode {
                StereoMode::Independent => {}
                StereoMode::MidSide => {
                    let mid = left[index];
                    let side = right[index];
                    left[index] = mid + side;
                    right[index] = mid - side;
                }
                StereoMode::Intensity { .. } => right[index] = left[index] * gain.unwrap(),
            }
        }
    }
    Ok(())
}

fn validate_tns_codes(
    resolution: u8,
    compressed: bool,
    codes: &[u8],
) -> Result<u8, MediaDecodeError> {
    if !(3..=4).contains(&resolution) || codes.len() > 31 {
        return Err(invalid("invalid AAC TNS coefficient dimensions"));
    }
    let transmitted_bits = resolution - u8::from(compressed);
    if codes.iter().any(|&code| code >= (1 << transmitted_bits)) {
        return Err(invalid("invalid AAC TNS coefficient code"));
    }
    Ok(transmitted_bits)
}

/// Resolution is 3 or 4, not the one-bit coef_res token. Output includes a[0]=1.
/// The caller clips the decoded order to its profile limit before using this API.
pub(crate) fn decode_tns_coefficients(
    resolution: u8,
    compressed: bool,
    codes: &[u8],
    lpc: &mut [f64],
) -> Result<(), MediaDecodeError> {
    let bits = validate_tns_codes(resolution, compressed, codes)?;
    if codes.len() > MAX_TNS_ORDER || lpc.len() != codes.len() + 1 {
        return Err(invalid("invalid AAC TNS LPC dimensions"));
    }
    let mut coefficients = [0.0; MAX_TNS_ORDER + 1];
    coefficients[0] = 1.0;
    let midpoint = f64::from(1_u8 << (resolution - 1));
    for (index, &code) in codes.iter().enumerate() {
        let signed = if code & (1 << (bits - 1)) == 0 {
            i32::from(code)
        } else {
            i32::from(code) - (1 << bits)
        };
        let denominator = midpoint + if signed < 0 { 0.5 } else { -0.5 };
        let reflection = (f64::from(signed) * FRAC_PI_2 / denominator).sin();
        let order = index + 1;
        let previous = coefficients;
        for i in 1..order {
            coefficients[i] = previous[i] + reflection * previous[order - i];
        }
        coefficients[order] = reflection;
    }
    lpc.copy_from_slice(&coefficients[..lpc.len()]);
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TnsFilter<'a> {
    pub length_bands: usize,
    pub reverse: bool,
    pub resolution: u8,
    pub compressed: bool,
    pub coefficients: &'a [u8],
}

/// Apply one window's filters from high to low bands. LC limits are 12 for a
/// long window and 7 for a short window (14496-3:2001 table 4.102).
/// max_bands is the sampling-rate/object-type limit from table 4.103.
pub(crate) fn apply_tns(
    spectrum: &mut [f64],
    swb_offsets: &[usize],
    max_sfb: usize,
    max_bands: usize,
    max_order: usize,
    filters: &[TnsFilter<'_>],
) -> Result<(), MediaDecodeError> {
    apply_tns_impl::<false>(
        spectrum,
        swb_offsets,
        max_sfb,
        max_bands,
        max_order,
        filters,
    )
}

fn apply_tns_impl<const SKIP_NOOP: bool>(
    spectrum: &mut [f64],
    swb_offsets: &[usize],
    max_sfb: usize,
    max_bands: usize,
    max_order: usize,
    filters: &[TnsFilter<'_>],
) -> Result<(), MediaDecodeError> {
    if spectrum.is_empty()
        || spectrum.len() > MAX_SPECTRUM
        || spectrum.iter().any(|v| !v.is_finite())
        || swb_offsets.len() < 2
        || swb_offsets.len() > spectrum.len() + 1
        || swb_offsets[0] != 0
        || swb_offsets.last() != Some(&spectrum.len())
        || swb_offsets.windows(2).any(|pair| pair[0] >= pair[1])
        || max_sfb >= swb_offsets.len()
        || max_bands > MAX_SPECTRUM
        || max_order > MAX_TNS_ORDER
        || filters.len() > 3
    {
        return Err(invalid("invalid AAC TNS window input"));
    }
    for filter in filters {
        if filter.length_bands > 63 {
            return Err(invalid("invalid AAC TNS band length"));
        }
        validate_tns_codes(filter.resolution, filter.compressed, filter.coefficients)?;
    }
    if SKIP_NOOP && (max_order == 0 || filters.iter().all(|filter| filter.coefficients.is_empty()))
    {
        return Ok(());
    }
    let mut reconstructed = [0.0; MAX_SPECTRUM];
    reconstructed[..spectrum.len()].copy_from_slice(spectrum);
    let mut bottom = swb_offsets.len() - 1;
    for filter in filters {
        let top = bottom;
        bottom = top.saturating_sub(filter.length_bands);
        let order = filter.coefficients.len().min(max_order);
        if order == 0 {
            continue;
        }
        let mut lpc = [0.0; MAX_TNS_ORDER + 1];
        decode_tns_coefficients(
            filter.resolution,
            filter.compressed,
            &filter.coefficients[..order],
            &mut lpc[..order + 1],
        )?;
        let start = swb_offsets[bottom.min(max_bands).min(max_sfb)];
        let end = swb_offsets[top.min(max_bands).min(max_sfb)];
        for step in 0..end - start {
            let index = if filter.reverse {
                end - 1 - step
            } else {
                start + step
            };
            let mut value = reconstructed[index];
            for lag in 1..=order.min(step) {
                let previous = if filter.reverse {
                    index + lag
                } else {
                    index - lag
                };
                value -= lpc[lag] * reconstructed[previous];
            }
            if !value.is_finite() {
                return Err(invalid("AAC TNS arithmetic overflow"));
            }
            reconstructed[index] = value;
        }
    }
    spectrum.copy_from_slice(&reconstructed[..spectrum.len()]);
    Ok(())
}

#[cfg(test)]
pub(super) fn apply_tns_candidate(
    spectrum: &mut [f64],
    swb_offsets: &[usize],
    max_sfb: usize,
    max_bands: usize,
    max_order: usize,
    filters: &[TnsFilter<'_>],
) -> Result<(), MediaDecodeError> {
    apply_tns_impl::<true>(
        spectrum,
        swb_offsets,
        max_sfb,
        max_bands,
        max_order,
        filters,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 4e-14 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    #[test]
    fn cached_quantizer_matches_reference_bits_for_all_values_and_gains() {
        let tables = quantization_tables();
        assert_eq!(tables.magnitudes.len(), 8192);
        for sf in 0..=255 {
            let gain = 2.0_f64.powf((sf as f64 - 100.0) * 0.25);
            assert_eq!(tables.gains[sf].to_bits(), gain.to_bits());
            for q in -8191_i32..=8191 {
                let value = f64::from(q);
                let reference = value * value.abs().cbrt() * gain;
                let magnitude = tables.magnitudes[q.unsigned_abs() as usize];
                let cached = (if q < 0 { -magnitude } else { magnitude }) * tables.gains[sf];
                assert_eq!(cached.to_bits(), reference.to_bits(), "q={q}, sf={sf}");
            }
        }
        let quantized: Vec<_> = (0..1024).map(|i| (i * 17) % 16383 - 8191).collect();
        let mut cached = vec![0.0; 1024];
        let mut reference = cached.clone();
        let bands = [512..1024, 0..256];
        inverse_quantize(&quantized, &mut cached, &bands, &[0, 255]).unwrap();
        inverse_quantize_reference(&quantized, &mut reference, &bands, &[0, 255]).unwrap();
        assert!(cached
            .iter()
            .zip(reference)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn inverse_quantization_matches_integer_cube_oracles_and_all_scalefactors() {
        let quantized = [-64, -27, -8, -1, 0, 1, 8, 27, 64];
        let mut output = [0.0; 9];
        inverse_quantize(&quantized, &mut output, &[0..9], &[100]).unwrap();
        for (actual, expected) in output
            .into_iter()
            .zip([-256.0, -81.0, -16.0, -1.0, 0.0, 1.0, 16.0, 81.0, 256.0])
        {
            close(actual, expected);
        }
        for sf in 0..=255 {
            let mut output = [0.0];
            inverse_quantize(&[1], &mut output, &[0..1], &[sf]).unwrap();
            close(
                output[0],
                ((f64::from(sf) - 100.0) * std::f64::consts::LN_2 / 4.0).exp(),
            );
        }
    }

    #[test]
    fn entire_normative_quantizer_range_matches_power_formula() {
        for q in -8191_i32..=8191 {
            let mut output = [0.0];
            inverse_quantize(&[q], &mut output, &[0..1], &[100]).unwrap();
            let magnitude = f64::from(q).abs().powf(4.0 / 3.0);
            close(output[0], magnitude * f64::from(q.signum()));
        }
    }

    #[test]
    fn repeated_short_window_bands_allow_unordered_ranges_and_zero_unused_bins() {
        let mut output = [99.0; 256];
        inverse_quantize(
            &[8; 256],
            &mut output,
            &[128..132, 0..4, 132..136, 4..8],
            &[104, 100, 96, 108],
        )
        .unwrap();
        assert_eq!(&output[128..132], &[32.0; 4]);
        assert_eq!(&output[..4], &[16.0; 4]);
        assert_eq!(&output[132..136], &[8.0; 4]);
        assert_eq!(&output[4..8], &[64.0; 4]);
        assert!(output[8..128]
            .iter()
            .chain(&output[136..])
            .all(|&v| v == 0.0));
    }

    #[test]
    fn inverse_quantization_rejects_invalid_input_without_writing_output() {
        for (quantized, bands, sf) in [
            (vec![8192], vec![0..1], vec![100]),
            (vec![i32::MIN], vec![0..1], vec![100]),
            (vec![1, 2], vec![0..2, 1..2], vec![100, 100]),
            (vec![1], vec![0..2], vec![100]),
            (vec![1], vec![0..0], vec![100]),
            (vec![1], vec![0..1], vec![-1]),
            (vec![1], vec![0..1], vec![256]),
            (vec![1], vec![0..1], vec![]),
        ] {
            let mut output = vec![7.0; quantized.len()];
            assert!(inverse_quantize(&quantized, &mut output, &bands, &sf).is_err());
            assert!(output.iter().all(|&v| v == 7.0));
        }
        assert!(inverse_quantize(&[], &mut [], &[], &[]).is_err());
        assert!(inverse_quantize(&[0; 1025], &mut [0.0; 1025], &[], &[]).is_err());
    }

    #[test]
    fn mid_side_matrix_has_no_extra_normalization() {
        let mut left = [2.0, 6.0, 11.0];
        let mut right = [1.0, -2.0, 13.0];
        apply_stereo(
            &mut left,
            &mut right,
            &[0..2, 2..3],
            &[StereoMode::MidSide, StereoMode::Independent],
        )
        .unwrap();
        assert_eq!(left, [3.0, 4.0, 11.0]);
        assert_eq!(right, [1.0, 8.0, 13.0]);
    }

    #[test]
    fn intensity_phase_and_position_match_stated_gain() {
        for out_of_phase in [false, true] {
            for invert in [false, true] {
                for (position, scale) in [(-4, 2.0), (0, 1.0), (4, 0.5)] {
                    let mut left = [2.0, -4.0];
                    let mut right = [99.0; 2];
                    apply_stereo(
                        &mut left,
                        &mut right,
                        &[0..2],
                        &[StereoMode::Intensity {
                            position,
                            out_of_phase,
                            invert,
                        }],
                    )
                    .unwrap();
                    let sign = if out_of_phase ^ invert { -1.0 } else { 1.0 };
                    assert_eq!(left, [2.0, -4.0]);
                    assert_eq!(right, [sign * scale * 2.0, sign * scale * -4.0]);
                }
            }
        }
    }

    #[test]
    fn stereo_failure_is_transactional_even_after_valid_earlier_bands() {
        let mut left = [1.0, f64::MAX];
        let mut right = [2.0, f64::MAX];
        let before = (left, right);
        assert!(apply_stereo(
            &mut left,
            &mut right,
            &[0..1, 1..2],
            &[StereoMode::MidSide; 2]
        )
        .is_err());
        assert_eq!((left, right), before);
        assert!(apply_stereo(&mut left, &mut right, &[0..1], &[]).is_err());
        assert!(apply_stereo(
            &mut left,
            &mut right,
            &[0..1],
            &[StereoMode::Intensity {
                position: i16::MIN,
                out_of_phase: false,
                invert: false
            }]
        )
        .is_err());
        assert_eq!((left, right), before);
    }

    #[test]
    fn tns_sign_extension_and_compression_use_original_resolution() {
        for (resolution, plain, compressed) in [(3, 7, 3), (4, 15, 7)] {
            let mut a = [0.0; 2];
            let mut b = [0.0; 2];
            decode_tns_coefficients(resolution, false, &[plain], &mut a).unwrap();
            decode_tns_coefficients(resolution, true, &[compressed], &mut b).unwrap();
            assert_eq!(a, b);
            close(a[0], 1.0);
            close(
                a[1],
                -(std::f64::consts::PI / ((1_u32 << resolution) + 1) as f64).sin(),
            );
        }
        let mut a = [0.0; 2];
        decode_tns_coefficients(3, false, &[1], &mut a).unwrap();
        close(a[1], (std::f64::consts::PI / 7.0).sin());
    }

    #[test]
    fn tns_lpc_recursion_matches_closed_form_three_reflection_polynomial() {
        let mut a = [0.0; 4];
        decode_tns_coefficients(3, false, &[1, 2, 3], &mut a).unwrap();
        let k1 = (std::f64::consts::PI / 7.0).sin();
        let k2 = (2.0 * std::f64::consts::PI / 7.0).sin();
        let k3 = (3.0 * std::f64::consts::PI / 7.0).sin();
        close(a[0], 1.0);
        close(a[1], k1 * (1.0 + k2) + k3 * k2);
        close(a[2], k2 + k3 * k1 * (1.0 + k2));
        close(a[3], k3);
    }

    fn filter<'a>(length_bands: usize, reverse: bool, codes: &'a [u8]) -> TnsFilter<'a> {
        TnsFilter {
            length_bands,
            reverse,
            resolution: 3,
            compressed: false,
            coefficients: codes,
        }
    }

    #[test]
    fn tns_first_order_impulse_and_reverse_have_analytic_response() {
        let k = (std::f64::consts::PI / 7.0).sin();
        for reverse in [false, true] {
            let mut spectrum = [0.0; 16];
            spectrum[if reverse { 15 } else { 0 }] = 1.0;
            apply_tns(
                &mut spectrum,
                &[0, 16],
                1,
                1,
                12,
                &[filter(1, reverse, &[1])],
            )
            .unwrap();
            for n in 0..16 {
                close(
                    spectrum[if reverse { 15 - n } else { n }],
                    (-k).powi(n as i32),
                );
            }
        }
    }

    #[test]
    fn tns_band_caps_order_clipping_and_filter_state_reset() {
        let mut spectrum = [2.0, 0.0, 1.0, 0.0, 17.0, 19.0];
        let k = (std::f64::consts::PI / 7.0).sin();
        apply_tns(
            &mut spectrum,
            &[0, 2, 4, 6],
            2,
            2,
            1,
            &[filter(2, false, &[1, 2, 3]), filter(63, false, &[1])],
        )
        .unwrap();
        close(spectrum[0], 2.0);
        close(spectrum[1], -2.0 * k);
        close(spectrum[2], 1.0);
        close(spectrum[3], -k);
        assert_eq!(&spectrum[4..], &[17.0, 19.0]);
        let before = spectrum;
        apply_tns(
            &mut spectrum,
            &[0, 2, 4, 6],
            0,
            2,
            7,
            &[filter(63, true, &[1])],
        )
        .unwrap();
        assert_eq!(spectrum, before);
    }

    #[test]
    fn tns_noop_preserves_signed_zero_and_all_finite_bits() {
        let empty_orders = [filter(63, true, &[]), filter(0, false, &[])];
        let clipped_orders = [filter(1, false, &[0; 31]), filter(0, true, &[1])];
        let cases: [(usize, &[TnsFilter<'_>]); 5] = [
            (0, &[]),
            (20, &[]),
            (20, &empty_orders),
            (0, &empty_orders),
            (0, &clipped_orders),
        ];
        let values = [
            -0.0,
            0.0,
            f64::MAX,
            -f64::MAX,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            f64::from_bits(1),
            -f64::from_bits(1),
            1.25,
        ];
        for length in [1, 4, 120, 128, 960, 1024] {
            let original: Vec<_> = (0..length)
                .map(|index| values[index % values.len()])
                .collect();
            let original_bits: Vec<_> = original.iter().map(|value| value.to_bits()).collect();
            for (order, filters) in cases {
                let mut fast = original.clone();
                let mut reference = original.clone();
                apply_tns_candidate(&mut fast, &[0, length], 1, 1, order, filters).unwrap();
                apply_tns_impl::<false>(&mut reference, &[0, length], 1, 1, order, filters)
                    .unwrap();
                assert_eq!(
                    fast.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                    original_bits
                );
                assert_eq!(
                    reference
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    original_bits
                );
            }
        }
    }

    #[test]
    fn tns_noop_retains_every_validation_and_reference_error() {
        let check = |input: &[f64],
                     offsets: &[usize],
                     max_sfb,
                     max_bands,
                     max_order,
                     filters: &[TnsFilter<'_>]| {
            let original: Vec<_> = input.iter().map(|value| value.to_bits()).collect();
            let mut fast = input.to_vec();
            let mut reference = input.to_vec();
            let fast_result = apply_tns_candidate(&mut fast, offsets, max_sfb, max_bands, max_order, filters);
            let reference_result = apply_tns_impl::<false>(
                &mut reference,
                offsets,
                max_sfb,
                max_bands,
                max_order,
                filters,
            );
            assert!(fast_result.is_err());
            assert_eq!(format!("{fast_result:?}"), format!("{reference_result:?}"));
            assert_eq!(
                fast.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                original
            );
            assert_eq!(
                reference
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                original
            );
        };
        let input = [-0.0, 0.0, 1.0, -1.0];
        for offsets in [
            &[][..],
            &[0][..],
            &[1, 4][..],
            &[0, 3][..],
            &[0, 2, 2, 4][..],
            &[0, 1, 2, 3, 4, 5][..],
        ] {
            check(&input, offsets, 1, 1, 0, &[]);
        }
        check(&[], &[0, 0], 0, 0, 0, &[]);
        check(&[0.0; 1025], &[0, 1025], 1, 1, 0, &[]);
        check(&[f64::from_bits(0x7ff8000000001234)], &[0, 1], 1, 1, 0, &[]);
        check(&[f64::INFINITY], &[0, 1], 1, 1, 0, &[]);
        check(&input, &[0, 4], 2, 1, 0, &[]);
        check(&input, &[0, 4], 1, 1025, 0, &[]);
        check(&input, &[0, 4], 1, 1, 21, &[]);
        check(&input, &[0, 4], 1, 1, 0, &[filter(1, false, &[]); 4]);
        check(&input, &[0, 4], 1, 1, 0, &[filter(64, false, &[])]);
        check(&input, &[0, 4], 1, 1, 0, &[filter(1, false, &[0; 32])]);
        for (resolution, compressed, codes) in [
            (2, false, &[][..]),
            (5, false, &[][..]),
            (3, true, &[4][..]),
            (4, false, &[16][..]),
        ] {
            let filters = [
                filter(1, false, &[]),
                TnsFilter {
                    length_bands: 1,
                    reverse: false,
                    resolution,
                    compressed,
                    coefficients: codes,
                },
            ];
            check(&input, &[0, 4], 1, 1, 0, &filters);
        }
    }

    #[test]
    fn tns_active_filters_remain_bit_exact_against_reference() {
        for reverse in [false, true] {
            for max_order in [1, 7, 12, 20] {
                let mut fast: Vec<_> = (0..128)
                    .map(|i| ((i * 137 + 19) % 97) as f64 - 48.0)
                    .collect();
                let mut reference = fast.clone();
                let filters = [filter(2, reverse, &[]), filter(3, reverse, &[1, 7, 2, 0])];
                apply_tns_candidate(
                    &mut fast,
                    &[0, 16, 32, 64, 96, 128],
                    5,
                    5,
                    max_order,
                    &filters,
                )
                .unwrap();
                apply_tns_impl::<false>(
                    &mut reference,
                    &[0, 16, 32, 64, 96, 128],
                    5,
                    5,
                    max_order,
                    &filters,
                )
                .unwrap();
                assert!(fast
                    .iter()
                    .zip(reference)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
    }

    #[test]
    fn malformed_tns_bounds_and_overflow_do_not_modify_input() {
        let mut lpc = [7.0; 2];
        for (resolution, compressed, code) in
            [(2, false, 0), (5, false, 0), (3, true, 4), (4, false, 16)]
        {
            assert!(decode_tns_coefficients(resolution, compressed, &[code], &mut lpc).is_err());
            assert_eq!(lpc, [7.0; 2]);
        }
        assert!(decode_tns_coefficients(3, false, &[0; 21], &mut [0.0; 22]).is_err());
        assert!(decode_tns_coefficients(3, false, &[0], &mut []).is_err());
        for offsets in [&[1, 4][..], &[0, 2, 2, 4][..], &[0, 5][..], &[][..]] {
            let mut spectrum = [1.0; 4];
            assert!(apply_tns(&mut spectrum, offsets, 1, 1, 12, &[]).is_err());
            assert_eq!(spectrum, [1.0; 4]);
        }
        let mut spectrum = [f64::MAX; 4];
        let before = spectrum;
        assert!(apply_tns(&mut spectrum, &[0, 4], 1, 1, 12, &[filter(1, false, &[7])]).is_err());
        assert_eq!(spectrum, before);
        assert!(apply_tns(&mut spectrum, &[0, 4], 1, 1, 12, &[filter(64, false, &[1])]).is_err());
        assert_eq!(spectrum, before);
    }
}
