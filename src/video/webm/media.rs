//! One shared WebM demuxer feeding independent audio and video codec states.

use super::{CodecDecoder, PendingPacket, WebmPacket, WebmStream, decode_video_packet};
use crate::audio::{
    opus, vorbis,
    webm::{WebmAudioBlock, WebmAudioCodec, WebmAudioTrack},
};
use crate::video::backend::{MediaDecodeError, MediaMetadata, MediaSample, StreamingMediaDecoder};
use std::collections::VecDeque;

enum Pending {
    Video(PendingPacket),
    Audio(PendingAudio),
}

struct PendingAudio {
    block: WebmAudioBlock,
    next_packet: usize,
    plan: Option<Vec<AudioPacketPlan>>,
}

struct AudioPacketPlan {
    start_frames: usize,
    output_frames: usize,
    skip_front: usize,
    skip_end: usize,
}

enum AudioState {
    New,
    Vorbis {
        decoder: vorbis::decoder::Decoder,
        delay_ns: u64,
        sample_rate: u32,
    },
    Opus {
        decoder: opus::music::MusicPacketDecoder,
        delay_ns: u64,
    },
    Failed,
}

pub struct WebmMediaDecoder {
    stream: WebmStream,
    video: Option<CodecDecoder>,
    audio: AudioState,
    pending: VecDeque<Pending>,
    audio_plan: Vec<AudioPacketPlan>,
}

impl Default for WebmMediaDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl WebmMediaDecoder {
    /// Independent codec state for packets from one shared demuxer. Construct
    /// one per decode worker; the container bytes are fetched and parsed once.
    pub fn from_tracks(stream: &WebmStream) -> Self {
        let mut decoder = Self::new();
        decoder.stream.video_codec = stream.video_codec;
        decoder.stream.video_codec_private = stream.video_codec_private.clone();
        decoder.stream.audio_track = stream.audio_track.clone();
        decoder
    }

    pub fn push_packet(&mut self, packet: WebmPacket) -> Result<Vec<MediaSample>, MediaDecodeError> {
        self.pending.push_back(match packet {
            WebmPacket::Video(packet) => Pending::Video(PendingPacket { packet, next_frame: 0 }),
            WebmPacket::Audio(block) => Pending::Audio(PendingAudio { block, next_packet: 0, plan: None }),
        });
        self.drain_packet()
    }

    pub fn drain_packet(&mut self) -> Result<Vec<MediaSample>, MediaDecodeError> {
        self.decode_pending()
    }
    pub fn new() -> Self {
        Self {
            stream: WebmStream::new(),
            video: None,
            audio: AudioState::New,
            pending: VecDeque::new(),
            audio_plan: Vec::new(),
        }
    }

    fn initialize_audio(track: &WebmAudioTrack) -> Result<AudioState, MediaDecodeError> {
        match track.codec {
            WebmAudioCodec::Vorbis => {
                let headers = vorbis::Headers::from_webm(&track.codec_private)?;
                if u16::from(headers.identification.channels) != track.channels
                    || f64::from(headers.identification.sample_rate) != track.sampling_frequency
                {
                    return Err(MediaDecodeError::InvalidData(
                        "Vorbis headers disagree with WebM audio track".into(),
                    ));
                }
                Ok(AudioState::Vorbis {
                    decoder: vorbis::decoder::Decoder::new(&headers, 1 << 20, 1 << 20)?,
                    delay_ns: track.codec_delay_ns,
                    sample_rate: headers.identification.sample_rate,
                })
            }
            WebmAudioCodec::Opus => {
                let header = opus::IdentificationHeader::parse(&track.codec_private)?;
                if u16::from(header.channels) != track.channels {
                    return Err(MediaDecodeError::InvalidData(
                        "Opus headers disagree with WebM audio track".into(),
                    ));
                }
                Ok(AudioState::Opus {
                    decoder: opus::music::MusicPacketDecoder::new(&header)?,
                    delay_ns: track.codec_delay_ns,
                })
            }
        }
    }

    fn decode_audio(
        &mut self,
        pending: &mut PendingAudio,
    ) -> Result<(Option<MediaSample>, bool), MediaDecodeError> {
        if matches!(self.audio, AudioState::New) {
            self.audio = Self::initialize_audio(self.stream.audio_track().ok_or_else(|| {
                MediaDecodeError::InvalidData("missing WebM audio track".into())
            })?)?;
        }
        let (delay_ns, sample_rate) = match &self.audio {
            AudioState::Vorbis { delay_ns, sample_rate, .. } => (*delay_ns, *sample_rate),
            AudioState::Opus { delay_ns, .. } => (*delay_ns, 48_000),
            _ => return Ok((None, true)),
        };
        if pending.plan.is_none() {
            let mut previous = match &self.audio {
                AudioState::Vorbis { decoder, .. } => decoder.previous_window_frames(),
                _ => None,
            };
            let mut elapsed = 0;
            let mut total_output = 0;
            let mut plan = std::mem::take(&mut self.audio_plan);
            plan.clear();
            plan.reserve(pending.block.packets.len());
            for packet in &pending.block.packets {
                let (duration, output) = match &self.audio {
                    AudioState::Vorbis { decoder, .. } => {
                        let window = decoder.packet_window_frames(packet)?;
                        let duration = window.map_or(0, |window| {
                            previous.map_or(window / 2, |previous| (previous + window) / 4)
                        });
                        let output = if previous.is_some() { duration } else { 0 };
                        if window.is_some() { previous = window; }
                        (duration, output)
                    }
                    AudioState::Opus { .. } => {
                        let packet = opus::Packet::parse(packet)?;
                        let duration = packet.samples_48khz();
                        (duration, duration)
                    }
                    _ => unreachable!(),
                };
                plan.push(AudioPacketPlan {
                    start_frames: elapsed,
                    output_frames: output,
                    skip_front: 0,
                    skip_end: 0,
                });
                elapsed += duration;
                total_output += output;
            }
            let padding = pending.block.discard_padding_ns.unwrap_or(0);
            let mut remaining = crate::audio::webm::discard_padding_frames(padding, sample_rate)?;
            if remaining > total_output {
                return Err(MediaDecodeError::InvalidData(
                    "WebM discard padding exceeds decoded block".into(),
                ));
            }
            if padding < 0 {
                for packet in &mut plan {
                    packet.skip_front = remaining.min(packet.output_frames);
                    remaining -= packet.skip_front;
                }
            } else {
                for packet in plan.iter_mut().rev() {
                    packet.skip_end = remaining.min(packet.output_frames);
                    remaining -= packet.skip_end;
                }
            }
            pending.plan = Some(plan);
        }
        let Some(packet) = pending.block.packets.get(pending.next_packet) else {
            return Ok((None, true));
        };
        let plan = &pending.plan.as_ref().unwrap()[pending.next_packet];
        let samples = match &mut self.audio {
            AudioState::Vorbis { decoder, .. } => decoder.decode(packet)?,
            AudioState::Opus { decoder, .. } => Some(decoder.decode_packet(packet)?),
            _ => unreachable!(),
        };
        pending.next_packet += 1;
        let done = pending.next_packet == pending.block.packets.len();
        let Some(mut samples) = samples else {
            if plan.output_frames != 0 {
                return Err(MediaDecodeError::InvalidData(
                    "audio packet produced no expected PCM".into(),
                ));
            }
            return Ok((None, done));
        };
        let channels = usize::from(samples.channels);
        if samples.samples.len() != plan.output_frames * channels {
            return Err(MediaDecodeError::InvalidData(
                "audio packet duration disagrees with PCM".into(),
            ));
        }
        let first = plan.skip_front * channels;
        let last = samples.samples.len() - plan.skip_end * channels;
        samples.samples.copy_within(first..last, 0);
        samples.samples.truncate(last - first);
        if samples.samples.is_empty() {
            return Ok((None, done));
        }
        let timestamp = i128::from(pending.block.timestamp_ns) - i128::from(delay_ns)
            + (plan.start_frames + plan.skip_front) as i128 * 1_000_000_000
                / i128::from(samples.sample_rate);
        let timestamp_ns = i64::try_from(timestamp).map_err(|_| {
            MediaDecodeError::InvalidData("WebM audio presentation timestamp overflow".into())
        })?;
        Ok((
            Some(MediaSample::Audio {
                timestamp_ns,
                samples,
            }),
            done,
        ))
    }

    fn recycle_audio_plan(&mut self, pending: &mut PendingAudio) {
        if let Some(mut plan) = pending.plan.take() {
            plan.clear();
            self.audio_plan = plan;
        }
    }
}

impl StreamingMediaDecoder for WebmMediaDecoder {
    fn push_media(&mut self, bytes: &[u8]) -> Result<Vec<MediaSample>, MediaDecodeError> {
        self.pending
            .extend(
                self.stream
                    .push_packets(bytes)?
                    .into_iter()
                    .map(|packet| match packet {
                        WebmPacket::Video(packet) => Pending::Video(PendingPacket {
                            packet,
                            next_frame: 0,
                        }),
                        WebmPacket::Audio(block) => Pending::Audio(PendingAudio {
                            block,
                            next_packet: 0,
                            plan: None,
                        }),
                    }),
            );
        self.decode_pending()
    }

    fn metadata(&self) -> Option<MediaMetadata> {
        self.stream.metadata()
    }

    fn finish(&self) -> Result<(), MediaDecodeError> {
        self.stream.finish()?;
        if !self.pending.is_empty() {
            return Err(MediaDecodeError::InvalidData("undrained WebM media samples".into()));
        }
        Ok(())
    }

    fn has_buffered_samples(&self) -> bool { !self.pending.is_empty() }
}

impl WebmMediaDecoder {
    fn decode_pending(&mut self) -> Result<Vec<MediaSample>, MediaDecodeError> {
        let Some(mut pending) = self.pending.pop_front() else {
            return Ok(Vec::new());
        };
        // One coded audio packet or video frame per call, including hidden reference updates,
        // keeps the same bounded work contract as the video-only decoder.
        match &mut pending {
            Pending::Video(packet) => {
                let (frame, done) = decode_video_packet(
                    &mut self.video,
                    self.stream.video_codec(),
                    &self.stream.video_codec_private,
                    packet,
                )?;
                if !done {
                    self.pending.push_front(pending);
                }
                Ok(frame.into_iter().map(MediaSample::Video).collect())
            }
            Pending::Audio(block) => match self.decode_audio(block) {
                Ok((sample, done)) => {
                    if done {
                        self.recycle_audio_plan(block);
                    } else {
                        self.pending.push_front(pending);
                    }
                    Ok(sample.into_iter().collect())
                }
                Err(error) => {
                    self.recycle_audio_plan(block);
                    self.audio = AudioState::Failed;
                    Ok(vec![MediaSample::AudioError(error)])
                }
            },
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "explicit VP8/Opus WebM fixture; independent packet codec equivalence"]
    fn independent_track_decoders_match_shared_media_decoder() {
        let path = std::env::var("WEBMEDIA_PACKET_SPLIT_WEBM").expect("VP8/Opus fixture");
        let mut reader = std::fs::File::open(path).unwrap();
        let mut stream = WebmStream::new();
        let mut shared = WebmMediaDecoder::new();
        let mut audio = None;
        let mut video = None;
        let mut video_count = 0;
        let mut buffer = [0u8; 16 * 1024];
        while video_count < 16 {
            let count = std::io::Read::read(&mut reader, &mut buffer).unwrap();
            assert!(count > 0);
            let mut expected = shared.push_media(&buffer[..count]).unwrap();
            while shared.has_buffered_samples() { expected.extend(shared.push_media(&[]).unwrap()); }
            let mut actual = Vec::new();
            for packet in stream.push_packets(&buffer[..count]).unwrap() {
                let decoder = match &packet {
                    WebmPacket::Video(_) => video.get_or_insert_with(|| WebmMediaDecoder::from_tracks(&stream)),
                    WebmPacket::Audio(_) => audio.get_or_insert_with(|| WebmMediaDecoder::from_tracks(&stream)),
                };
                actual.extend(decoder.push_packet(packet).unwrap());
                while decoder.has_buffered_samples() { actual.extend(decoder.drain_packet().unwrap()); }
            }
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.into_iter().zip(expected) {
                match (actual, expected) {
                    (MediaSample::Video(actual), MediaSample::Video(expected)) => {
                        video_count += 1;
                        assert_eq!((actual.width, actual.height, actual.timestamp), (expected.width, expected.height, expected.timestamp));
                        assert_eq!(actual.rgba, expected.rgba);
                    }
                    (MediaSample::Audio { timestamp_ns: actual_time, samples: actual },
                     MediaSample::Audio { timestamp_ns: expected_time, samples: expected }) => {
                        assert_eq!(actual_time, expected_time);
                        assert_eq!((actual.channels, actual.sample_rate), (expected.channels, expected.sample_rate));
                        assert_eq!(actual.samples, expected.samples);
                    }
                    _ => panic!("independent track decode changed sample type or failed"),
                }
            }
        }
    }

    #[test]
    #[ignore = "explicit full Opus WebM fixture and FFmpeg f32 reference"]
    fn public_opus_webm_music_matches_whole_reference() {
        let path = std::env::var("WEBMEDIA_OPUS_WEBM").expect("Opus WebM fixture");
        let reference = std::fs::read(std::env::var("WEBMEDIA_OPUS_REFERENCE_F32")
            .expect("stereo 48 kHz float reference")).unwrap();
        let mut reader = std::fs::File::open(path).unwrap();
        let mut decoder = WebmMediaDecoder::new();
        let mut pcm = Vec::new();
        let mut buffer = [0u8; 16 * 1024];
        loop {
            let count = std::io::Read::read(&mut reader, &mut buffer).unwrap();
            if count == 0 { break; }
            let mut input = &buffer[..count];
            loop {
                for sample in decoder.push_media(input).unwrap() {
                    match sample {
                        MediaSample::Audio { timestamp_ns, samples } => {
                            assert_eq!((samples.sample_rate, samples.channels), (48_000, 2));
                            assert!(samples.samples.iter().all(|value| value.is_finite()));
                            let skip = if timestamp_ns < 0 {
                                (timestamp_ns.unsigned_abs() as u128 * 48_000).div_ceil(1_000_000_000) as usize
                            } else { 0 };
                            pcm.extend_from_slice(&samples.samples[(skip * 2).min(samples.samples.len())..]);
                        }
                        MediaSample::AudioError(error) => panic!("public Opus decode failed: {error:?}"),
                        MediaSample::Video(_) => {}
                    }
                }
                if !decoder.has_buffered_samples() { break; }
                input = &[];
            }
        }
        decoder.finish().unwrap();
        assert_eq!(pcm.len() * 4, reference.len(), "codec delay and discard padding must preserve exact duration");
        let mut error = 0.0f64;
        let mut power = 0.0f64;
        for (sample, bytes) in pcm.iter().zip(reference.chunks_exact(4)) {
            let expected = f64::from(f32::from_le_bytes(bytes.try_into().unwrap()));
            error += (f64::from(*sample) - expected).powi(2);
            power += expected * expected;
        }
        let relative_rms = (error / power).sqrt();
        eprintln!("public Opus WebM: {} stereo frames, relative RMS error {relative_rms}", pcm.len() / 2);
        assert!(power > 1.0, "reference must contain real music");
        assert!(relative_rms < 0.001, "whole soundtrack differs: {relative_rms}");
    }

    #[test]
    fn streaming_vorbis_pcm_from_supplied_clip() {
        let Some(path) = std::env::var_os("WEBMEDIA_AUDIO_SAMPLE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let reference = std::env::var_os("WEBMEDIA_AUDIO_REFERENCE_F32")
            .map(|path| std::fs::read(path).unwrap());
        let expected_pts: Option<Vec<i64>> =
            std::env::var_os("WEBMEDIA_AUDIO_REFERENCE_PTS").map(|path| {
                std::fs::read_to_string(path)
                    .unwrap()
                    .lines()
                    .filter(|line| !line.trim().is_empty())
                    .map(|line| (line.trim().parse::<f64>().unwrap() * 1e9).round() as i64)
                    .collect()
            });
        let mut baseline = None;
        for chunk_size in [1, 17, 4096, bytes.len()] {
            let mut decoder = WebmMediaDecoder::new();
            let mut pcm = Vec::new();
            let mut pts = Vec::new();
            for chunk in bytes.chunks(chunk_size) {
                let mut input = chunk;
                loop {
                    for sample in decoder.push_media(input).unwrap() {
                        match sample {
                            MediaSample::Audio {
                                timestamp_ns,
                                samples,
                            } => {
                                assert!(samples.samples.iter().all(|value| value.is_finite()));
                                assert_eq!(
                                    samples.channels,
                                    decoder.metadata().unwrap().channels.unwrap()
                                );
                                assert_eq!(
                                    samples.sample_rate,
                                    decoder.metadata().unwrap().sample_rate.unwrap()
                                );
                                pts.push(timestamp_ns);
                                pcm.extend(samples.samples);
                            }
                            MediaSample::Video(_) => {}
                            MediaSample::AudioError(error) => {
                                panic!("audio track failed: {error:?}")
                            }
                        }
                    }
                    if !decoder.has_buffered_samples() {
                        break;
                    }
                    input = &[];
                }
            }
            decoder.finish().unwrap();
            assert!(!pcm.is_empty());
            if let Some(reference) = &reference {
                assert_eq!(reference.len(), pcm.len() * 4);
                let maximum = reference
                    .chunks_exact(4)
                    .zip(&pcm)
                    .map(|(bytes, sample)| {
                        (f32::from_le_bytes(bytes.try_into().unwrap()) - sample).abs()
                    })
                    .fold(0.0f32, f32::max);
                assert!(maximum < 1e-5, "PCM maximum error {maximum}");
            }
            if let Some(expected) = &expected_pts {
                assert_eq!(pts.len(), expected.len());
                // ffprobe reports this fixture's delay in container ticks;
                // our API preserves the sub-tick CodecDelay in nanoseconds.
                let remainder = decoder.stream.audio_track().unwrap().codec_delay_ns
                    % decoder.stream.time_code_scale;
                for (&actual, &expected) in pts.iter().zip(expected) {
                    assert!(
                        (i128::from(actual) + i128::from(remainder) - i128::from(expected)).abs()
                            <= 1000,
                        "timestamp {actual} != {expected}"
                    );
                }
            }
            let current = (pcm, pts);
            if let Some(baseline) = &baseline {
                assert_eq!(&current, baseline);
            } else {
                eprintln!(
                    "streaming Vorbis PCM samples={} blocks={} first_ns={:?} last_ns={:?} codec_delay_ns={}",
                    current.0.len(),
                    current.1.len(),
                    current.1.first(),
                    current.1.last(),
                    decoder.stream.audio_track().unwrap().codec_delay_ns
                );
                baseline = Some(current);
            }
        }
    }
}
