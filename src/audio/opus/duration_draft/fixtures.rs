//! Bounded, opt-in encoded fixtures. Only FFmpeg/ffprobe binaries are oracles.
use super::{IdentificationHeader, Packet, music};
use std::process::Command;

pub(super) fn header(channels: u8, pre_skip: u16) -> IdentificationHeader {
    IdentificationHeader {
        channels,
        pre_skip,
        input_sample_rate: 48000,
        output_gain_q8: 0,
        mapping_family: 0,
        streams: 1,
        coupled_streams: channels - 1,
        channel_mapping: (0..channels).collect(),
    }
}

fn binary(program: &str, arguments: &[&str]) -> Vec<u8> {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .expect("FFmpeg binary required");
    assert!(
        output.status.success(),
        "{program}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

#[test]
#[ignore = "bounded FFmpeg encoded duration PCM gates; requires coordinated correctness slot"]
fn encoded_fullband_durations_match_unchanged_pcm_acceptance() {
    let mut failures = Vec::new();
    for (duration, configuration) in [("2.5", 28), ("5", 29), ("10", 30), ("20", 31)] {
        for channels in 1..=2 {
            let path = format!(
                "/private/tmp/rune-opus-duration-{}-{duration}-{channels}.ogg",
                std::process::id()
            );
            let mono = "0.13*sin(2*PI*431*t)+0.07*sin(2*PI*(2100*t+730*t*t))+if(lt(mod(t,0.037),0.001),0.09,0)";
            let stereo = format!("{mono}|0.11*sin(2*PI*719*t)+0.05*sin(2*PI*11913*t)");
            let source = format!(
                "aevalsrc={}:s=48000:d=0.2",
                if channels == 1 { mono } else { &stereo }.replace(',', "\\,")
            );
            binary(
                "ffmpeg",
                &[
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    &source,
                    "-c:a",
                    "libopus",
                    "-application",
                    "lowdelay",
                    "-frame_duration",
                    duration,
                    "-b:a",
                    if channels == 1 { "128k" } else { "256k" },
                    "-vbr",
                    "off",
                    "-cutoff",
                    "20000",
                    &path,
                ],
            );
            let metadata = String::from_utf8(binary(
                "ffprobe",
                &[
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_packets",
                    "-show_entries",
                    "packet=size:packet_side_data=skip_samples,discard_padding",
                    "-of",
                    "default=noprint_wrappers=1",
                    &path,
                ],
            ))
            .unwrap();
            let values = |key: &str| -> Vec<usize> {
                metadata
                    .lines()
                    .filter_map(|line| line.strip_prefix(key))
                    .map(|v| v.parse().unwrap())
                    .collect()
            };
            let sizes = values("size=");
            let pre_skip: usize = values("skip_samples=").iter().sum();
            let discard: usize = values("discard_padding=").iter().sum();
            let bytes = binary(
                "ffmpeg",
                &[
                    "-v", "error", "-i", &path, "-map", "0:a:0", "-c:a", "copy", "-f", "data",
                    "pipe:1",
                ],
            );
            let oracle = binary(
                "ffmpeg",
                &[
                    "-v",
                    "error",
                    "-i",
                    &path,
                    "-map",
                    "0:a:0",
                    "-f",
                    "f32le",
                    "-acodec",
                    "pcm_f32le",
                    "pipe:1",
                ],
            );
            assert_eq!(sizes.iter().sum::<usize>(), bytes.len());
            assert_eq!(oracle.len(), 9600 * channels as usize * 4);
            let header = header(channels, pre_skip.try_into().unwrap());
            let mut decoder = music::MusicPacketDecoder::new(&header).unwrap();
            let mut decoded = Vec::new();
            let mut offset = 0;
            let mut first_error = None;
            let mut first_entropy_failure = false;
            for (index, &size) in sizes.iter().enumerate() {
                let packet_bytes = &bytes[offset..offset + size];
                let packet = Packet::parse(packet_bytes).unwrap();
                assert_eq!(packet.configuration, configuration);
                assert_eq!(packet.stereo, channels == 2);
                assert_eq!(packet.frames.len(), 1);
                match decoder.decode_packet(packet_bytes) {
                    Ok(pcm) => {
                        if !first_entropy_failure {
                            if let Some(trace) = decoder
                                .band_trace
                                .iter()
                                .find(|trace| trace.errors_after > trace.errors_before)
                            {
                                eprintln!(
                                    "FIRST ENTROPY ERROR duration={duration} channels={channels} packet={index} stage_positions={:?} band={} tf={} fine={} budget={} tell_before={} tell_after={} errors_before={} errors_after={}",
                                    decoder.stage_positions,
                                    trace.band,
                                    trace.tf,
                                    trace.fine_bits,
                                    trace.budget,
                                    trace.tell_before,
                                    trace.tell_after,
                                    trace.errors_before,
                                    trace.errors_after
                                );
                                first_entropy_failure = true;
                            }
                        }
                        assert!(pcm.samples.iter().all(|v| v.is_finite()));
                        assert_eq!(
                            pcm.samples.len(),
                            usize::from(packet.frame_samples_48khz) * channels as usize
                        );
                        decoded.extend(pcm.samples);
                    }
                    Err(error) => {
                        first_error = Some(format!("packet={index} bytes={size} error={error:?}"));
                        break;
                    }
                }
                offset += size;
            }
            if let Some(error) = first_error {
                let failure = format!("duration={duration} channels={channels} {error}");
                eprintln!("DRAFT REJECTED {failure}");
                failures.push(failure);
            } else {
                let frames = oracle.len() / (4 * channels as usize);
                assert_eq!(
                    decoded.len() / channels as usize,
                    pre_skip + frames + discard
                );
                let actual =
                    &decoded[pre_skip * channels as usize..(pre_skip + frames) * channels as usize];
                let mut cross = 0.0;
                let mut actual_energy = 0.0;
                let mut reference_energy = 0.0;
                let mut squared_error = 0.0;
                let mut peak = 0.0f64;
                for (&sample, bytes) in actual.iter().zip(oracle.chunks_exact(4)) {
                    let reference = f64::from(f32::from_le_bytes(bytes.try_into().unwrap()));
                    let sample = f64::from(sample);
                    cross += sample * reference;
                    actual_energy += sample * sample;
                    reference_energy += reference * reference;
                    squared_error += (sample - reference).powi(2);
                    peak = peak.max((sample - reference).abs());
                }
                let correlation = cross / (actual_energy * reference_energy).sqrt();
                let rmse = (squared_error / reference_energy).sqrt();
                eprintln!(
                    "DRAFT duration={duration} channels={channels} packets={} frames={frames} pre_skip={pre_skip} discard={discard} correlation={correlation} relative_rmse={rmse} peak={peak} entropy_errors={}",
                    sizes.len(),
                    decoder.uniform_errors()
                );
                if !(correlation > 0.999 && rmse < 0.02 && decoder.uniform_errors() == 0) {
                    failures.push(format!(
                        "duration={duration} channels={channels} PCM acceptance failed"
                    ));
                }
                decoder.reset();
                let mut replay = decoder;
                let mut offset = 0;
                let mut sample_offset = 0;
                for &size in &sizes {
                    let pcm = replay.decode_packet(&bytes[offset..offset + size]).unwrap();
                    assert!(
                        pcm.samples
                            .iter()
                            .zip(&decoded[sample_offset..])
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                    sample_offset += pcm.samples.len();
                    offset += size;
                }
                assert_eq!(sample_offset, decoded.len());
                let first_packet = &bytes[..sizes[0]];
                let mut fresh = music::MusicPacketDecoder::new(&header).unwrap();
                let first_pcm = fresh.decode_packet(first_packet).unwrap();
                for cut in 0..first_packet.len() {
                    replay.reset();
                    let truncated = replay.decode_packet(&first_packet[..cut]);
                    match truncated {
                        Ok(pcm) => {
                            assert_eq!(pcm.samples.len(), first_pcm.samples.len());
                            assert!(pcm.samples.iter().all(|v| v.is_finite()));
                        }
                        Err(_) => assert!(replay.decode_packet(first_packet).is_err()),
                    }
                    replay.reset();
                    let after = replay.decode_packet(first_packet).unwrap();
                    assert!(
                        after
                            .samples
                            .iter()
                            .zip(&first_pcm.samples)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                }
            }
            std::fs::remove_file(&path).unwrap();
        }
    }
    assert!(
        failures.is_empty(),
        "duration draft is NOT ready: {failures:?}"
    );
}
