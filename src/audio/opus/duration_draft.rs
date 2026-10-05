//! Test-only fullband CELT duration preparation; not a packet decoder.
//!
//! Algorithms follow RFC 6716 prose sections 4.3.2.1 and 4.3.3. Numeric
//! initializers alone were extracted from its cached normative Appendix A:
//! quant_bands.c (e_prob_model, pred_coef, beta_coef), static_modes_float.h
//! (cache_caps50). Archive SHA1, matching the RFC's published digest:
//! 86a927223e73d2476646a1b933fcd3fffb6ecc8c. No Appendix functions are used.

use super::{allocation, celt::Layout, energy, invalid, range::RangeDecoder};
use crate::video::backend::MediaDecodeError;

use super::{IdentificationHeader, Packet, celt, pvq, range, simd, synthesis, tf};

#[path = "duration_draft/allocation.rs"]
mod draft_allocation;
#[path = "duration_draft/frame.rs"]
mod frame;
#[path = "duration_draft/music.rs"]
mod music;

#[path = "duration_draft/fixtures.rs"]
mod fixtures;

const PRED_Q15: [u16; 4] = [29440, 26112, 21248, 16384];
const BETA_Q15: [u16; 4] = [30147, 22282, 12124, 6554];
const MODELS: [[[u8; 42]; 2]; 4] = [
    [
        [
            72, 127, 65, 129, 66, 128, 65, 128, 64, 128, 62, 128, 64, 128, 64, 128, 92, 78, 92, 79,
            92, 78, 90, 79, 116, 41, 115, 40, 114, 40, 132, 26, 132, 26, 145, 17, 161, 12, 176, 10,
            177, 11,
        ],
        [
            24, 179, 48, 138, 54, 135, 54, 132, 53, 134, 56, 133, 55, 132, 55, 132, 61, 114, 70,
            96, 74, 88, 75, 88, 87, 74, 89, 66, 91, 67, 100, 59, 108, 50, 120, 40, 122, 37, 97, 43,
            78, 50,
        ],
    ],
    [
        [
            83, 78, 84, 81, 88, 75, 86, 74, 87, 71, 90, 73, 93, 74, 93, 74, 109, 40, 114, 36, 117,
            34, 117, 34, 143, 17, 145, 18, 146, 19, 162, 12, 165, 10, 178, 7, 189, 6, 190, 8, 177,
            9,
        ],
        [
            23, 178, 54, 115, 63, 102, 66, 98, 69, 99, 74, 89, 71, 91, 73, 91, 78, 89, 86, 80, 92,
            66, 93, 64, 102, 59, 103, 60, 104, 60, 117, 52, 123, 44, 138, 35, 133, 31, 97, 38, 77,
            45,
        ],
    ],
    [
        [
            61, 90, 93, 60, 105, 42, 107, 41, 110, 45, 116, 38, 113, 38, 112, 38, 124, 26, 132, 27,
            136, 19, 140, 20, 155, 14, 159, 16, 158, 18, 170, 13, 177, 10, 187, 8, 192, 6, 175, 9,
            159, 10,
        ],
        [
            21, 178, 59, 110, 71, 86, 75, 85, 84, 83, 91, 66, 88, 73, 87, 72, 92, 75, 98, 72, 105,
            58, 107, 54, 115, 52, 114, 55, 112, 56, 129, 51, 132, 40, 150, 33, 140, 29, 98, 35, 77,
            42,
        ],
    ],
    [
        [
            42, 121, 96, 66, 108, 43, 111, 40, 117, 44, 123, 32, 120, 36, 119, 33, 127, 33, 134,
            34, 139, 21, 147, 23, 152, 20, 158, 25, 154, 26, 166, 21, 173, 16, 184, 13, 184, 10,
            150, 13, 139, 15,
        ],
        [
            22, 178, 63, 114, 74, 82, 84, 83, 92, 82, 103, 62, 96, 72, 96, 67, 101, 73, 107, 72,
            113, 55, 118, 52, 125, 52, 118, 52, 117, 55, 135, 49, 137, 39, 157, 32, 145, 29, 97,
            33, 77, 40,
        ],
    ],
];

const CAPS: [[u8; 21]; 8] = [
    [
        224, 224, 224, 224, 224, 224, 224, 224, 160, 160, 160, 160, 185, 185, 185, 178, 178, 168,
        134, 61, 37,
    ],
    [
        224, 224, 224, 224, 224, 224, 224, 224, 240, 240, 240, 240, 207, 207, 207, 198, 198, 183,
        144, 66, 40,
    ],
    [
        160, 160, 160, 160, 160, 160, 160, 160, 185, 185, 185, 185, 193, 193, 193, 183, 183, 172,
        138, 64, 38,
    ],
    [
        240, 240, 240, 240, 240, 240, 240, 240, 207, 207, 207, 207, 204, 204, 204, 193, 193, 180,
        143, 66, 40,
    ],
    [
        185, 185, 185, 185, 185, 185, 185, 185, 193, 193, 193, 193, 193, 193, 193, 183, 183, 172,
        138, 65, 39,
    ],
    [
        207, 207, 207, 207, 207, 207, 207, 207, 204, 204, 204, 204, 201, 201, 201, 188, 188, 176,
        141, 66, 40,
    ],
    [
        193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 194, 194, 194, 184, 184, 173,
        139, 65, 39,
    ],
    [
        204, 204, 204, 204, 204, 204, 204, 204, 201, 201, 201, 201, 198, 198, 198, 187, 187, 175,
        140, 66, 40,
    ],
];

fn mode(layout: Layout) -> Result<usize, MediaDecodeError> {
    if layout.bands() != (0..21) {
        return Err(MediaDecodeError::Unsupported);
    }
    Ok((layout.samples() / 120).trailing_zeros() as usize)
}

fn coarse(
    decoder: &mut RangeDecoder<'_>,
    layout: Layout,
    channels: usize,
    intra: bool,
    previous: &[f64],
) -> Result<Vec<f64>, MediaDecodeError> {
    let lm = mode(layout)?;
    if !(1..=2).contains(&channels)
        || previous.len() != 21 * channels
        || previous.iter().any(|value| !value.is_finite())
    {
        return Err(invalid("invalid draft CELT energy history"));
    }
    let mut cursor = decoder.clone();
    let budget = cursor.frame_bytes() as u64 * 8;
    let mut residuals = Vec::with_capacity(previous.len());
    for pair in MODELS[lm][usize::from(intra)].chunks_exact(2) {
        for _ in 0..channels {
            residuals.push(match budget.saturating_sub(cursor.tell()) {
                15.. => cursor.laplace(u16::from(pair[0]) * 128, u16::from(pair[1]) * 64)?,
                2..=14 => [0, -1, 1][cursor.inverse_cdf(&[2, 1, 0], 2)?],
                1 => -i16::from(cursor.bit(1)?),
                _ => -1,
            });
        }
    }
    let prediction = energy::CoarsePrediction {
        alpha: if intra {
            0.0
        } else {
            f64::from(PRED_Q15[lm]) / 32768.0
        },
        beta: f64::from(if intra { 4915 } else { BETA_Q15[lm] }) / 32768.0,
        history_floor: -9.0,
        energy_floor: -f64::MAX,
    };
    let result = energy::reconstruct_coarse(layout, channels, previous, &residuals, prediction)?;
    *decoder = cursor;
    Ok(result)
}

#[test]
fn draft_20ms_energy_is_bit_identical_to_existing_stage() {
    let layout = Layout::from_configuration(31).unwrap().unwrap();
    for channels in 1..=2 {
        let previous: Vec<_> = (0..21 * channels).map(|i| i as f64 / 8.0 - 12.0).collect();
        for intra in [false, true] {
            for length in [0, 1, 2, 8, 32, 128] {
                for fill in [0, 37, 150, 255] {
                    let bytes = vec![fill; length];
                    let mut old = RangeDecoder::new(&bytes);
                    let mut draft = old.clone();
                    let expected =
                        energy::decode_coarse_20ms(&mut old, layout, channels, intra, &previous)
                            .unwrap();
                    let actual = coarse(&mut draft, layout, channels, intra, &previous).unwrap();
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    assert_eq!(draft.tell_fractional(), old.tell_fractional());
                    assert_eq!(
                        draft.raw_bits(4).map_err(|error| format!("{error:?}")),
                        old.raw_bits(4).map_err(|error| format!("{error:?}"))
                    );
                    assert_eq!(draft.bit(3).unwrap(), old.bit(3).unwrap());
                }
            }
        }
    }
}

#[test]
fn draft_all_durations_have_finite_energy_and_exact_cap_conversion() {
    for configuration in 28..=31 {
        let layout = Layout::from_configuration(configuration).unwrap().unwrap();
        let lm = mode(layout).unwrap();
        assert_eq!(layout.samples(), 120 << lm);
        for channels in 1..=2 {
            let row = &CAPS[2 * lm + channels - 1];
            let caps = allocation::caps_from_cache(layout, channels, row).unwrap();
            for (band, cap) in caps.iter().enumerate() {
                assert_eq!(
                    usize::from(*cap),
                    (usize::from(row[band]) + 64) * channels * layout.band(band).unwrap().len() / 4
                );
            }
            for intra in [false, true] {
                let bytes = [150; 64];
                let mut decoder = RangeDecoder::new(&bytes);
                let previous = vec![0.0; 21 * channels];
                let energy = coarse(&mut decoder, layout, channels, intra, &previous).unwrap();
                assert_eq!(energy.len(), previous.len());
                assert!(energy.iter().all(|v| v.is_finite()));
            }
        }
    }
}

#[test]
fn draft_rejects_invalid_history_without_advancing_entropy() {
    for configuration in 28..=31 {
        let layout = Layout::from_configuration(configuration).unwrap().unwrap();
        for channels in [0, 1, 2, 3] {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut decoder = RangeDecoder::new(&[150; 64]);
                let before = decoder.tell_fractional();
                assert!(
                    coarse(
                        &mut decoder,
                        layout,
                        channels,
                        false,
                        &vec![value; 21 * channels]
                    )
                    .is_err()
                );
                assert_eq!(decoder.tell_fractional(), before);
            }
        }
    }
}

#[test]
fn draft_does_not_enable_production_durations_or_change_failure_state() {
    for channels in 1..=2 {
        let header = super::IdentificationHeader {
            channels,
            pre_skip: 0,
            input_sample_rate: 48000,
            output_gain_q8: 0,
            mapping_family: 0,
            streams: 1,
            coupled_streams: channels - 1,
            channel_mapping: (0..channels).collect(),
        };
        let stereo = u8::from(channels == 2) << 2;
        let silence = [(31 << 3) | stereo, 0xff, 0xfe];
        for configuration in 28..=30 {
            let mut decoder = super::music::MusicPacketDecoder::new(&header).unwrap();
            let packet = [(configuration << 3) | stereo, 0xff, 0xfe];
            assert!(matches!(
                decoder.decode_packet(&packet),
                Err(MediaDecodeError::Unsupported)
            ));
            assert!(matches!(
                decoder.decode_packet(&silence),
                Err(MediaDecodeError::Unsupported)
            ));
            let mut fresh = super::music::MusicPacketDecoder::new(&header).unwrap();
            let pcm = fresh.decode_packet(&silence).unwrap();
            assert_eq!(pcm.samples.len(), 960 * usize::from(channels));
            assert!(
                pcm.samples
                    .iter()
                    .all(|sample| sample.is_finite() && *sample == 0.0)
            );
        }
    }
}

#[test]
fn draft_20ms_allocation_and_packet_pcm_preserve_existing_behavior() {
    let layout = Layout::from_configuration(31).unwrap().unwrap();
    for channels in 1..=2 {
        for transient in [false, true] {
            for fill in [0, 37, 150, 255] {
                let bytes = [fill; 128];
                let mut old = RangeDecoder::new(&bytes);
                let mut draft = old.clone();
                let expected = allocation::AllocationPreparation::decode_fixed(
                    &mut old, layout, channels, transient,
                )
                .unwrap();
                let actual = draft_allocation::AllocationPreparation::decode_fixed(
                    &mut draft, layout, channels, transient,
                )
                .unwrap();
                assert_eq!(actual.caps, expected.caps);
                assert_eq!(actual.boosts_eighths, expected.boosts_eighths);
                assert_eq!(actual.thresholds_eighths, expected.thresholds_eighths);
                assert_eq!(actual.trim_offsets_eighths, expected.trim_offsets_eighths);
                assert_eq!(actual.remaining_eighths, expected.remaining_eighths);
                assert_eq!(
                    actual.anti_collapse_reserved_eighths,
                    expected.anti_collapse_reserved_eighths
                );
                let expected = expected.finish(&mut old, layout, channels).unwrap();
                let actual = actual.finish(&mut draft, layout, channels).unwrap();
                assert_eq!(actual.shape_eighths, expected.shape_eighths);
                assert_eq!(actual.fine_bits, expected.fine_bits);
                assert_eq!(actual.final_priorities, expected.final_priorities);
                assert_eq!(actual.coded_bands, expected.coded_bands);
                assert_eq!(actual.intensity, expected.intensity);
                assert_eq!(actual.dual_stereo, expected.dual_stereo);
                assert_eq!(actual.balance_eighths, expected.balance_eighths);
                assert_eq!(old.tell_fractional(), draft.tell_fractional());
            }
        }
        let header = fixtures::header(channels as u8, 0);
        for first in [0, 37, 150, 255] {
            let mut packet = [150; 129];
            packet[0] = (31 << 3) | (u8::from(channels == 2) << 2);
            packet[1] = first;
            let mut old = super::music::MusicPacketDecoder::new(&header).unwrap();
            let mut draft = music::MusicPacketDecoder::new(&header).unwrap();
            for _ in 0..3 {
                let expected = old
                    .decode_packet(&packet)
                    .map(|pcm| pcm.samples.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
                    .map_err(|error| format!("{error:?}"));
                let actual = draft
                    .decode_packet(&packet)
                    .map(|pcm| pcm.samples.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
                    .map_err(|error| format!("{error:?}"));
                assert_eq!(actual, expected);
                assert_eq!(draft.uniform_errors(), old.uniform_errors());
            }
        }
    }
}

#[test]
fn draft_failure_reset_and_silence_are_bounded_for_all_durations() {
    for configuration in 28..=31 {
        for channels in 1..=2 {
            let header = fixtures::header(channels, 0);
            let mut draft = music::MusicPacketDecoder::new(&header).unwrap();
            let silence = [
                (configuration << 3) | (u8::from(channels == 2) << 2),
                0xff,
                0xfe,
            ];
            for _ in 0..3 {
                assert!(draft.decode_packet(&[]).is_err());
                assert!(matches!(
                    draft.decode_packet(&silence),
                    Err(MediaDecodeError::Unsupported)
                ));
                draft.reset();
                let actual = draft.decode_packet(&silence).unwrap();
                let mut fresh = music::MusicPacketDecoder::new(&header).unwrap();
                let expected = fresh.decode_packet(&silence).unwrap();
                assert_eq!(
                    actual.samples.len(),
                    (120 << (configuration & 3)) * usize::from(channels)
                );
                assert!(actual.samples.iter().all(|v| v.is_finite() && *v == 0.0));
                assert_eq!(
                    actual
                        .samples
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .samples
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn draft_two_bin_stereo_allocation_uses_four_not_five_degrees() {
    use draft_allocation::{AllocationPreparation, shape_degrees};
    assert_eq!(shape_degrees(1, 2, false, true), 2);
    assert_eq!(shape_degrees(2, 2, false, true), 4);
    assert_eq!(shape_degrees(2, 4, false, true), 9);
    assert_eq!(shape_degrees(2, 4, true, true), 8);
    assert_eq!(shape_degrees(2, 4, false, false), 8);
    let layout = Layout::from_configuration(29).unwrap().unwrap();
    let preparation = AllocationPreparation {
        caps: vec![32767; 21],
        boosts_eighths: (0..21).map(|band| u32::from(band == 20)).collect(),
        total_boost_eighths: 1,
        trim: 5,
        thresholds_eighths: vec![0; 21],
        trim_offsets_eighths: vec![0; 21],
        remaining_eighths: 2400,
        anti_collapse_reserved_eighths: 0,
        skip_reserved_eighths: 8,
        intensity_reserved_eighths: 36,
        dual_stereo_reserved_eighths: 8,
    };
    let mut covered = 0;
    for fill in 0..=254 {
        let bytes = [fill; 128];
        let mut decoder = RangeDecoder::new(&bytes);
        let mut expected_entropy = decoder.clone();
        let intensity = expected_entropy.uniform(22).unwrap() as usize;
        let dual = intensity > 0 && expected_entropy.bit(1).unwrap();
        let allocation = preparation.finish(&mut decoder, layout, 2).unwrap();
        assert_eq!(allocation.coded_bands, 21);
        assert_eq!(allocation.intensity, intensity);
        assert_eq!(allocation.dual_stereo, dual);
        assert_eq!(
            decoder.tell_fractional(),
            expected_entropy.tell_fractional()
        );
        assert_eq!(
            decoder.raw_bits(3).unwrap(),
            expected_entropy.raw_bits(3).unwrap()
        );
        if intensity == 0 || dual {
            continue;
        }
        for band in 0..intensity.min(8) {
            // N=2, C=2, log2(N) in eighths=8. Recover the pre-fine
            // budget and independently apply the existing scalar fine rule.
            let bits = allocation.shape_eighths[band] + 16 * i32::from(allocation.fine_bits[band]);
            let mut offset = 16 - 4 * 21;
            if bits + offset < 4 * 16 {
                offset += 32 >> 2;
            } else if bits + offset < 4 * 24 {
                offset += 32 >> 3;
            }
            let fine = ((bits + offset + 16) / 32).max(0).min(bits / 16).min(8);
            assert_eq!(i32::from(allocation.fine_bits[band]), fine);
            assert_eq!(
                allocation.final_priorities[band],
                u8::from(fine * 32 >= bits + offset)
            );
            covered += 1;
        }
    }
    assert!(covered > 100, "coupled N=2 allocation must be exercised");
}
