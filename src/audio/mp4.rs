//! Incremental AAC access-unit extraction. PCM synthesis is codec-owned.

use crate::audio::aac::{AacConfig, AacDecoder};
use crate::video::backend::{AudioSamples, MediaDecodeError};
use crate::video::mp4::{Mp4AudioIndex, Mp4Error};

const MAX_BUFFER_BYTES: usize = 64 * 1024 * 1024;
const MAX_PACKETS_PER_PUSH: usize = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AacPacket {
    pub presentation_time: i64,
    pub timescale: u32,
    pub data: Vec<u8>,
}

#[derive(Default)]
pub struct Mp4AudioStream {
    bytes: Vec<u8>,
    base_offset: u64,
    index: Option<Mp4AudioIndex>,
    indexed: bool,
    next_sample: usize,
    start_time: f64,
}

impl Mp4AudioStream {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn index(&self) -> Option<&Mp4AudioIndex> {
        self.index.as_ref()
    }

    pub fn with_start_time(seconds: f64) -> Result<Self, Mp4Error> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(Mp4Error::Invalid("invalid audio seek time"));
        }
        Ok(Self {
            start_time: seconds,
            ..Self::default()
        })
    }

    /// Empty pushes drain buffered packets without reading more input.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<AacPacket>, Mp4Error> {
        if self.indexed && self.index.is_none() {
            return Ok(Vec::new());
        }
        if bytes.len() > MAX_BUFFER_BYTES.saturating_sub(self.bytes.len()) {
            return Err(Mp4Error::TooLarge);
        }
        self.bytes.extend_from_slice(bytes);
        if !self.indexed {
            match Mp4AudioIndex::parse_prefix(&self.bytes) {
                Ok(index) => {
                    if self.start_time > 0.0
                        && let Some(index) = &index
                    {
                        let first = index.samples.partition_point(|sample| {
                            sample.presentation_time as f64 / f64::from(index.timescale)
                                < self.start_time
                        });
                        self.next_sample = first.saturating_sub(2);
                    }
                    self.index = index;
                    self.indexed = true;
                }
                Err(Mp4Error::Incomplete) => return Ok(Vec::new()),
                Err(error) => return Err(error),
            }
        }
        let Some(index) = &self.index else {
            self.bytes.clear();
            return Ok(Vec::new());
        };
        let mut packets = Vec::new();
        while packets.len() < MAX_PACKETS_PER_PUSH {
            let Some(sample) = index.samples.get(self.next_sample) else {
                break;
            };
            let start = sample
                .offset
                .checked_sub(self.base_offset)
                .ok_or(Mp4Error::Invalid("audio sample precedes stream buffer"))?;
            let start = usize::try_from(start).map_err(|_| Mp4Error::TooLarge)?;
            let end = start
                .checked_add(sample.size as usize)
                .ok_or(Mp4Error::TooLarge)?;
            let Some(data) = self.bytes.get(start..end) else {
                break;
            };
            packets.push(AacPacket {
                presentation_time: sample.presentation_time,
                timescale: index.timescale,
                data: data.to_vec(),
            });
            self.next_sample += 1;
        }
        // Keep a small overlap buffer, not the complete downloaded file.
        let retain_offset = index
            .samples
            .get(self.next_sample)
            .map(|sample| sample.offset)
            .unwrap_or(self.base_offset + self.bytes.len() as u64);
        let discard = retain_offset
            .saturating_sub(self.base_offset)
            .min(self.bytes.len() as u64) as usize;
        if discard >= 256 * 1024 || self.next_sample == index.samples.len() {
            self.bytes.drain(..discard);
            self.base_offset += discard as u64;
        }
        Ok(packets)
    }

    /// Call after draining with empty pushes until no more packets are returned.
    pub fn finish(&self) -> Result<(), Mp4Error> {
        if !self.indexed
            || self
                .index
                .as_ref()
                .is_some_and(|index| self.next_sample != index.samples.len())
        {
            return Err(Mp4Error::Incomplete);
        }
        Ok(())
    }
}

pub struct AacAudioPacket {
    pub timestamp: f64,
    pub samples: AudioSamples,
}

#[derive(Default)]
pub struct Mp4AacStream {
    packets: Mp4AudioStream,
    decoder: Option<Mp4AacDecoder>,
}

/// Stateful AAC synthesis for access units supplied by a shared MP4 demuxer.
pub struct Mp4AacDecoder {
    decoder: AacDecoder,
    timescale: u32,
    duration_ticks: u64,
}

impl Mp4AacDecoder {
    pub fn new(index: &Mp4AudioIndex) -> Result<Self, MediaDecodeError> {
        if index.timescale == 0 {
            return Err(MediaDecodeError::InvalidData("zero audio timescale".into()));
        }
        Ok(Self {
            decoder: AacDecoder::new(AacConfig::parse(&index.audio_specific_config)?)?,
            timescale: index.timescale,
            duration_ticks: index.duration_ticks,
        })
    }

    pub fn decode(
        &mut self,
        presentation_time: i64,
        data: &[u8],
    ) -> Result<Option<AacAudioPacket>, MediaDecodeError> {
        // Priming must update synthesis overlap even when none of its PCM is presented.
        let mut samples = self.decoder.decode(data)?;
        let channels = usize::from(samples.channels);
        let rate = i128::from(samples.sample_rate);
        let scale = i128::from(self.timescale);
        let start = if presentation_time < 0 {
            usize::try_from((-i128::from(presentation_time) * rate + scale - 1) / scale)
                .unwrap_or(usize::MAX)
        } else {
            0
        };
        // Fragment initialization can advertise zero before any media is known.
        // That is not a zero-length presentation; priming still applies below.
        let end = presentation_end(self.duration_ticks, presentation_time, rate, scale,
            samples.samples.len() / channels);
        if start >= end {
            return Ok(None);
        }
        samples.samples.truncate(end * channels);
        if start != 0 {
            samples.samples.drain(..start * channels);
        }
        let timestamp = presentation_time as f64 / f64::from(self.timescale)
            + start as f64 / f64::from(samples.sample_rate);
        Ok(Some(AacAudioPacket { timestamp, samples }))
    }
}

fn presentation_end(duration: u64, time: i64, rate: i128, scale: i128, frames: usize) -> usize {
    if duration == 0 { return frames; }
    let ticks = i128::from(duration) - i128::from(time);
    usize::try_from((ticks * rate / scale).max(0)).unwrap_or(usize::MAX).min(frames)
}

impl Mp4AacStream {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_start_time(seconds: f64) -> Result<Self, Mp4Error> {
        Ok(Self {
            packets: Mp4AudioStream::with_start_time(seconds)?,
            decoder: None,
        })
    }

    pub fn index(&self) -> Option<&Mp4AudioIndex> {
        self.packets.index()
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<AacAudioPacket>, MediaDecodeError> {
        let packets = self.packets.push(bytes).map_err(mp4_error)?;
        let Some(index) = self.packets.index() else {
            return Ok(Vec::new());
        };
        if self.decoder.is_none() {
            self.decoder = Some(Mp4AacDecoder::new(index)?);
        }
        let mut output = Vec::new();
        for packet in packets {
            if let Some(decoded) = self
                .decoder
                .as_mut()
                .unwrap()
                .decode(packet.presentation_time, &packet.data)?
            {
                output.push(decoded);
            }
        }
        Ok(output)
    }

    pub fn finish(&self) -> Result<(), MediaDecodeError> {
        self.packets.finish().map_err(mp4_error)
    }
}

fn mp4_error(error: Mp4Error) -> MediaDecodeError {
    match error {
        Mp4Error::Unsupported(_) => MediaDecodeError::Unsupported,
        other => MediaDecodeError::InvalidData(format!("MP4 audio: {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn unknown_fragment_duration_preserves_pcm_end_and_known_duration_trims() {
        assert_eq!(super::presentation_end(0, 4096, 22050, 22050, 1024), 1024);
        assert_eq!(super::presentation_end(0, -512, 22050, 22050, 1024), 1024);
        assert_eq!(super::presentation_end(4608, 4096, 22050, 22050, 1024), 512);
        assert_eq!(super::presentation_end(4096, 4096, 22050, 22050, 1024), 0);
    }
    use super::*;

    #[test]
    fn truncated_movie_is_not_successful_audio() {
        let mut stream = Mp4AudioStream::new();
        assert!(
            stream
                .push(&[0, 0, 0, 8, b'f', b't', b'y', b'p'])
                .unwrap()
                .is_empty()
        );
        assert_eq!(stream.finish(), Err(Mp4Error::Incomplete));
    }

    #[test]
    #[ignore = "set WEBMEDIA_AAC_MP4 to an AAC MP4 fixture"]
    fn streamed_packets_match_file_samples() {
        let bytes = std::fs::read(std::env::var("WEBMEDIA_AAC_MP4").unwrap()).unwrap();
        let index = Mp4AudioIndex::parse_prefix(&bytes).unwrap().unwrap();
        let mut stream = Mp4AudioStream::new();
        let mut packets = Vec::new();
        for chunk in bytes.chunks(8191) {
            packets.extend(stream.push(chunk).unwrap());
        }
        loop {
            let next = stream.push(&[]).unwrap();
            if next.is_empty() {
                break;
            }
            packets.extend(next);
        }
        stream.finish().unwrap();
        assert_eq!(packets.len(), index.samples.len());
        for (packet, sample) in packets.iter().zip(&index.samples) {
            assert_eq!(packet.presentation_time, sample.presentation_time);
            assert_eq!(packet.timescale, index.timescale);
            assert_eq!(
                packet.data,
                bytes[sample.offset as usize..sample.offset as usize + sample.size as usize]
            );
        }
        assert!(stream.bytes.len() < 256 * 1024);
    }

    #[test]
    #[ignore = "set WEBMEDIA_AAC_MP4 to goodtimes_h264.mp4"]
    fn streamed_pcm_applies_priming_padding_and_seek_preroll() {
        let bytes = std::fs::read(std::env::var("WEBMEDIA_AAC_MP4").unwrap()).unwrap();
        let mut stream = Mp4AacStream::new();
        let mut frames = 0;
        let mut first = None;
        let mut last = 0.0;
        let mut consume = |packets: Vec<AacAudioPacket>| {
            for packet in packets {
                first.get_or_insert(packet.timestamp);
                assert!((packet.timestamp - last).abs() < 1e-9);
                frames += packet.samples.samples.len() / usize::from(packet.samples.channels);
                last = packet.timestamp
                    + packet.samples.samples.len() as f64
                        / f64::from(packet.samples.channels)
                        / f64::from(packet.samples.sample_rate);
            }
        };
        for chunk in bytes.chunks(65521) {
            consume(stream.push(chunk).unwrap());
        }
        loop {
            let packets = stream.push(&[]).unwrap();
            if packets.is_empty() {
                break;
            }
            consume(packets);
        }
        stream.finish().unwrap();
        assert_eq!(first.unwrap(), 0.0);
        assert_eq!(frames, 6_325_704);
        assert!((last - 143.44).abs() < 1e-9);
        let mut stream = Mp4AacStream::with_start_time(60.0).unwrap();
        let mut packets = Vec::new();
        for chunk in bytes.chunks(65521) {
            packets = stream.push(chunk).unwrap();
            if !packets.is_empty() {
                break;
            }
        }
        assert!(!packets.is_empty());
        assert!(packets[0].timestamp > 59.94 && packets[0].timestamp < 60.0);
        assert!(packets[0].samples.samples.iter().any(|s| s.abs() > 1e-6));
    }
}
