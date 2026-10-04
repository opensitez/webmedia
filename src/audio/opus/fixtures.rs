//! Binary-only actual Partyfire fixture checks. This test does not pretend that
//! successful frame-prefix parsing is a successful compressed-audio decode.

use super::{IdentificationHeader, Packet, frame::FrameStart, silence::SilencePacketDecoder};
use crate::video::backend::MediaDecodeError;
use std::{collections::BTreeMap, process::Command};

fn binary(name: &str, args: &[&str]) -> Vec<u8> {
    let output = Command::new(name)
        .args(args)
        .output()
        .expect("FFmpeg/ffprobe binary required");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

#[test]
#[ignore = "requires actual Spacewalk fixture and FFmpeg/ffprobe binaries"]
fn spacewalk_whole_track_music_oracle() {
    let path = std::env::var("OPUS_SPACEWALK_WEBM")
        .expect("set OPUS_SPACEWALK_WEBM to spacewalk_av1.webm");
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
            .map(|value| value.parse().unwrap())
            .collect()
    };
    let sizes = values("size=");
    let pre_skip: usize = values("skip_samples=").iter().sum();
    let discard: usize = values("discard_padding=").iter().sum();
    let bytes = binary(
        "ffmpeg",
        &[
            "-v", "error", "-i", &path, "-map", "0:a:0", "-c:a", "copy", "-f", "data", "pipe:1",
        ],
    );
    assert_eq!(sizes.iter().sum::<usize>(), bytes.len());
    let header =
        IdentificationHeader::parse(b"OpusHead\x01\x02\x38\x01\x80\xbb\x00\x00\x00\x00\x00")
            .unwrap();
    assert_eq!(pre_skip, usize::from(header.pre_skip));
    let mut modes = BTreeMap::new();
    let mut offset = 0;
    for &size in &sizes {
        let packet = Packet::parse(&bytes[offset..offset + size]).unwrap();
        *modes
            .entry((
                packet.configuration,
                packet.stereo,
                packet.frame_samples_48khz,
                packet.frames.len(),
            ))
            .or_insert(0usize) += 1;
        offset += size;
    }
    eprintln!("Spacewalk packet configurations: {modes:?}");
    let mut decoder = super::music::MusicPacketDecoder::new(&header).unwrap();
    let mut decoded = Vec::new();
    offset = 0;
    let began = std::time::Instant::now();
    for (index, &size) in sizes.iter().enumerate() {
        let packet = &bytes[offset..offset + size];
        let pcm = decoder.decode_packet(packet).unwrap_or_else(|error| {
            panic!(
                "Spacewalk first failing packet={index} size={size} TOC={} error={error:?}",
                packet[0]
            )
        });
        assert_eq!(pcm.channels, 2);
        assert_eq!(pcm.sample_rate, 48000);
        assert!(
            pcm.samples.iter().all(|sample| sample.is_finite()),
            "packet {index}"
        );
        decoded.extend(pcm.samples);
        offset += size;
    }
    let decode_seconds = began.elapsed().as_secs_f64();
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
    assert_eq!(oracle.len() % 8, 0);
    let reference_frames = oracle.len() / 8;
    assert_eq!(decoded.len() / 2, pre_skip + reference_frames + discard);
    let actual = &decoded[pre_skip * 2..(pre_skip + reference_frames) * 2];
    let mut cross = 0.0;
    let mut actual_energy = 0.0;
    let mut reference_energy = 0.0;
    let mut error = 0.0;
    let mut peak_error = 0.0f64;
    for (&value, bytes) in actual.iter().zip(oracle.chunks_exact(4)) {
        let reference = f64::from(f32::from_le_bytes(bytes.try_into().unwrap()));
        let value = f64::from(value);
        cross += value * reference;
        actual_energy += value * value;
        reference_energy += reference * reference;
        error += (value - reference).powi(2);
        peak_error = peak_error.max((value - reference).abs());
    }
    let correlation = cross / (actual_energy * reference_energy).sqrt();
    let relative_rmse = (error / reference_energy).sqrt();
    eprintln!(
        "Spacewalk WHOLE CLIP: packets={} trimmed_frames={reference_frames} pre_skip={pre_skip} discard={discard} correlation={correlation} relative_rmse={relative_rmse} peak_error={peak_error} entropy_errors={} decode_seconds={decode_seconds}",
        sizes.len(),
        decoder.uniform_errors()
    );
    assert!(
        correlation > 0.999 && relative_rmse < 0.02 && decoder.uniform_errors() == 0,
        "Spacewalk whole music PCM acceptance failed"
    );
}

/// Run with OPUS_AUDIO_ORACLE_WEBM pointing at partyfire_vp8.webm. FFmpeg is
/// used only for packet extraction and the non-silent PCM oracle, not decoding
/// support inside the implementation.
#[test]
#[ignore = "requires actual Partyfire fixture and FFmpeg/ffprobe binaries"]
fn partyfire_packet_modes_prefixes_and_non_silent_oracle() {
    let path = std::env::var("OPUS_AUDIO_ORACLE_WEBM")
        .expect("set OPUS_AUDIO_ORACLE_WEBM to partyfire_vp8.webm");
    let sizes = binary(
        "ffprobe",
        &[
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_packets",
            "-show_entries",
            "packet=size",
            "-of",
            "default=noprint_wrappers=1",
            &path,
        ],
    );
    let sizes: Vec<usize> = String::from_utf8(sizes)
        .unwrap()
        .lines()
        .filter_map(|line| line.strip_prefix("size="))
        .map(|size| size.parse().unwrap())
        .collect();
    let bytes = binary(
        "ffmpeg",
        &[
            "-v", "error", "-i", &path, "-map", "0:a:0", "-c:a", "copy", "-f", "data", "pipe:1",
        ],
    );
    assert_eq!(sizes.iter().sum::<usize>(), bytes.len());
    let mut modes = BTreeMap::new();
    let mut prefixes = BTreeMap::new();
    let mut canonical = 0;
    let mut offset = 0;
    for size in &sizes {
        let data = &bytes[offset..offset + size];
        offset += size;
        let packet = Packet::parse(data).unwrap();
        *modes
            .entry((
                packet.configuration,
                packet.stereo,
                packet.frame_samples_48khz,
                packet.frames.len(),
            ))
            .or_insert(0usize) += 1;
        if data == [0xfc, 0xff, 0xfe] {
            canonical += 1;
        } else {
            let mut start = FrameStart::parse(&packet, 0).unwrap();
            *prefixes
                .entry((start.transient, start.intra, start.pitch.is_some()))
                .or_insert(0usize) += 1;
            // Independent history input exercises real packet entropy without
            // pretending coarse-only energies are valid stream history.
            let shapes = start.start_shapes(&[-28.0; 42]).unwrap();
            assert!(
                shapes
                    .energy
                    .log_energies()
                    .iter()
                    .all(|value| value.is_finite())
            );
            assert!((1..=21).contains(&shapes.allocation.coded_bands));
            assert!(
                shapes
                    .allocation
                    .shape_eighths
                    .iter()
                    .all(|&bits| bits >= 0)
            );
            assert!(start.entropy.tell() <= start.entropy.frame_bytes() as u64 * 8);
        }
    }
    assert_eq!(sizes.len(), 7718);
    assert_eq!(modes.get(&(31, true, 960, 1)), Some(&7718));
    assert_eq!(canonical, 732);
    assert_eq!(prefixes.values().sum::<usize>(), 6986);
    let first = Packet::parse(&bytes[..sizes[0]]).unwrap();
    let start = FrameStart::parse(&first, 0).unwrap();
    assert!(start.transient);
    assert!(!start.intra);
    assert!(start.pitch.is_none());
    assert_eq!(start.entropy.tell_fractional(), 42);
    // The public narrow decoder must not turn the real first music packet into
    // the canonical-silence PCM path, even though both use configuration 31.
    let header =
        IdentificationHeader::parse(b"OpusHead\x01\x02\x00\x00\x80\xbb\x00\x00\x00\x00\x00")
            .unwrap();
    let mut narrow = SilencePacketDecoder::new(&header).unwrap();
    assert!(matches!(
        narrow.decode_packet(&bytes[..sizes[0]]),
        Err(MediaDecodeError::Unsupported)
    ));
    assert!(matches!(
        narrow.decode_packet(&[0xfc, 0xff, 0xfe]),
        Err(MediaDecodeError::Unsupported)
    ));
    let pcm = binary(
        "ffmpeg",
        &[
            "-v",
            "error",
            "-i",
            &path,
            "-map",
            "0:a:0",
            "-t",
            "0.1",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
    );
    let samples: Vec<_> = pcm
        .chunks_exact(4)
        .map(|sample| f32::from_le_bytes(sample.try_into().unwrap()))
        .collect();
    assert_eq!(samples.len(), 9600);
    assert!(samples.iter().all(|sample| sample.is_finite()));
    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0f32, f32::max);
    let rms = (samples
        .iter()
        .map(|&sample| f64::from(sample).powi(2))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt();
    assert!(peak > 0.0 && rms > 0.0);
    let mut music = super::music::MusicPacketDecoder::new(&header).unwrap();
    let began = std::time::Instant::now();
    let mut decoded = Vec::new();
    let mut offset = 0;
    for (index, &size) in sizes.iter().enumerate() {
        if index == 0 {
            let packet = Packet::parse(&bytes[..size]).unwrap();
            let mut frame = FrameStart::parse(&packet, 0).unwrap();
            let start = frame.start_shapes(&[0.0; 42]).unwrap();
            eprintln!(
                "first music allocation: tell={} spread={} tf={:?} coded={} intensity={} dual={} fine={:?} shape={:?}",
                frame.entropy.tell_fractional(),
                start.spread,
                start.tf.adjustments,
                start.allocation.coded_bands,
                start.allocation.intensity,
                start.allocation.dual_stereo,
                start.allocation.fine_bits,
                start.allocation.shape_eighths
            );
        }
        let output = music
            .decode_packet(&bytes[offset..offset + size])
            .unwrap_or_else(|error| panic!("Partyfire music packet {index}: {error:?}"));
        assert_eq!(output.samples.len(), 1920);
        assert!(output.samples.iter().all(|value| value.is_finite()));
        if index < 6 {
            let p = Packet::parse(&bytes[offset..offset + size]).unwrap();
            let prefix = FrameStart::parse(&p, 0).unwrap();
            eprintln!(
                "packet {index} transient={} intra={} pitch={:?} errors={}",
                prefix.transient,
                prefix.intra,
                prefix.pitch,
                music.uniform_errors()
            );
        }
        decoded.extend(output.samples);
        offset += size;
    }
    let actual = &decoded[312 * 2..312 * 2 + samples.len()];
    eprintln!(
        "Partyfire uniform recovery errors: {}",
        music.uniform_errors()
    );
    let mut best = (0.0, -999, false);
    let reference_energy: f64 = samples.iter().map(|&x| f64::from(x).powi(2)).sum();
    for swapped in [false, true] {
        for lag in -312..=312 {
            let begin = (312 + lag) as usize * 2;
            let mut cross = 0.0;
            let mut energy = 0.0;
            for (index, &reference) in samples.iter().enumerate() {
                let channel = if swapped { index ^ 1 } else { index };
                let value = f64::from(decoded[begin + channel]);
                cross += value * f64::from(reference);
                energy += value * value;
            }
            let correlation = cross / (energy * reference_energy).sqrt();
            if correlation > best.0 {
                best = (correlation, lag, swapped);
            }
        }
    }
    eprintln!("Partyfire best diagnostic alignment: {best:?}");
    let cross: f64 = actual
        .iter()
        .zip(&samples)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum();
    let actual_energy: f64 = actual.iter().map(|&a| f64::from(a).powi(2)).sum();
    let reference_energy: f64 = samples.iter().map(|&b| f64::from(b).powi(2)).sum();
    let error: f64 = actual
        .iter()
        .zip(&samples)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum();
    let correlation = cross / (actual_energy * reference_energy).sqrt();
    for packet in 0..6 {
        let from = (packet * 960usize).saturating_sub(312) * 2;
        let to = ((packet + 1) * 960usize).saturating_sub(312).min(4800) * 2;
        if from >= to {
            continue;
        }
        let mut x = 0.0;
        let mut a = 0.0;
        let mut b = 0.0;
        for (&left, &right) in actual[from..to].iter().zip(&samples[from..to]) {
            x += f64::from(left) * f64::from(right);
            a += f64::from(left).powi(2);
            b += f64::from(right).powi(2);
        }
        eprintln!(
            "packet {packet} waveform correlation={}",
            x / (a * b).sqrt()
        );
    }
    let relative_rmse = (error / reference_energy).sqrt();
    eprintln!(
        "Partyfire first 6 decoded music packets: correlation={} relative_rmse={} gain={} rms={}",
        correlation,
        relative_rmse,
        cross / reference_energy,
        (actual_energy / actual.len() as f64).sqrt()
    );
    eprintln!(
        "Partyfire: packets={} modes={modes:?} canonical={canonical} noncanonical=6986 first_coarse_tell_eighths=42 prefixes={prefixes:?}",
        sizes.len()
    );
    eprintln!("Partyfire PCM binary oracle: frames=4800 channels=2 peak={peak} rms={rms}");
    assert!(
        correlation > 0.999 && relative_rmse < 0.02 && music.uniform_errors() == 0,
        "Partyfire music PCM acceptance failed: correlation={correlation} relative_rmse={relative_rmse} soft_errors={}",
        music.uniform_errors()
    );
    let oracle_path = std::env::var("OPUS_AUDIO_ORACLE_PCM")
        .unwrap_or_else(|_| "/private/tmp/opus-partyfire-reference.f32".into());
    let oracle = std::fs::read(&oracle_path).expect("full Partyfire FFmpeg oracle required");
    assert_eq!(oracle.len() % 8, 0);
    let reference_frames = oracle.len() / 8;
    assert_eq!(reference_frames, 7_408_609);
    assert_eq!(decoded.len() / 2, 7718 * 960);
    assert_eq!(decoded.len() / 2 - 312 - reference_frames, 359);
    let actual = &decoded[624..624 + reference_frames * 2];
    let mut cross = 0.0;
    let mut actual_energy = 0.0;
    let mut reference_energy = 0.0;
    let mut error = 0.0;
    let mut peak_error = 0.0f64;
    let mut packet_errors = vec![0.0f64; sizes.len()];
    for (index, (&value, bytes)) in actual.iter().zip(oracle.chunks_exact(4)).enumerate() {
        let reference = f64::from(f32::from_le_bytes(bytes.try_into().unwrap()));
        let value = f64::from(value);
        cross += value * reference;
        actual_energy += value * value;
        reference_energy += reference * reference;
        let difference = value - reference;
        error += difference * difference;
        packet_errors[(index / 2 + 312) / 960] += difference * difference;
        peak_error = peak_error.max(difference.abs());
    }
    let correlation = cross / (actual_energy * reference_energy).sqrt();
    let relative_rmse = (error / reference_energy).sqrt();
    let mut worst: Vec<_> = packet_errors.into_iter().enumerate().collect();
    worst.sort_by(|a, b| b.1.total_cmp(&a.1));
    for &(index, error) in worst.iter().take(24) {
        let offset: usize = sizes[..index].iter().sum();
        let data = &bytes[offset..offset + sizes[index]];
        let prefix = FrameStart::parse(&Packet::parse(data).unwrap(), 0);
        eprintln!(
            "worst packet {index}: error={error} size={} prefix={:?}",
            sizes[index],
            prefix.map(|start| (start.transient, start.intra, start.pitch.is_some()))
        );
    }
    eprintln!(
        "Partyfire WHOLE CLIP: packets={} raw_frames={} trimmed_frames={reference_frames} pre_skip=312 discard=359 correlation={correlation} relative_rmse={relative_rmse} peak_error={peak_error} entropy_errors={} decode_seconds={}",
        sizes.len(),
        decoded.len() / 2,
        music.uniform_errors(),
        began.elapsed().as_secs_f64()
    );
    assert!(
        correlation > 0.999 && relative_rmse < 0.02 && music.uniform_errors() == 0,
        "whole Partyfire music PCM acceptance failed"
    );
}
