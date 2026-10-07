//! Incremental little-endian RIFF/WAVE PCM decoding, independently implemented
//! from Microsoft's RIFF and WAVEFORMATEX specifications:
//! https://learn.microsoft.com/en-us/windows/win32/xaudio2/resource-interchange-file-format--riff-
//! https://learn.microsoft.com/en-us/windows/win32/api/mmreg/ns-mmreg-waveformatex
//! https://learn.microsoft.com/en-us/windows/win32/multimedia/devices-and-data-types
//!
//! Supports mono/stereo integer PCM 8/16/24/32 and IEEE float32. Compressed,
//! extensible, RF64, RIFX, multiple data chunks and float64 are not supported.
//! Nonfinite floats are invalid; finite floats saturate to the PCM output range.

use crate::video::backend::{
    AudioSamples, MediaDecodeError, MediaMetadata, MediaSample, StreamingMediaDecoder,
};

const MAX_BUFFER: usize = 1024 * 1024;
const MAX_FRAMES: usize = 4096;
const MAX_PACKETS: usize = 32;
const MAX_STEPS: usize = 128;
const MAX_FORMAT: u32 = 65536;

#[derive(Clone, Copy)]
struct Format {
    tag: u16,
    channels: u16,
    rate: u32,
    bits: u16,
    align: u16,
}

#[derive(Clone, Copy, Default)]
enum State {
    #[default]
    Riff,
    Header,
    Format(u32),
    Skip {
        left: u32,
        pad: bool,
    },
    Data {
        left: u32,
        pad: bool,
    },
    Pad,
    Done,
}

#[derive(Default)]
pub struct WavStream {
    bytes: Vec<u8>,
    state: State,
    offset: u64,
    end: u64,
    format: Option<Format>,
    data_size: Option<u32>,
    frames: u64,
    drain_pending: bool,
    failure: Option<MediaDecodeError>,
}

impl WavStream {
    pub fn new() -> Self {
        Self::default()
    }

    /// Supply input in bounded pieces. Empty pushes drain a completed output
    /// batch before more input is supplied. An error permanently fails the stream.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<MediaSample>, MediaDecodeError> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        let result = self.push_inner(bytes);
        if let Err(error) = &result {
            self.failure = Some(error.clone());
            self.bytes.clear();
        }
        result
    }

    fn push_inner(&mut self, bytes: &[u8]) -> Result<Vec<MediaSample>, MediaDecodeError> {
        if bytes.len() > MAX_BUFFER.saturating_sub(self.bytes.len()) {
            return Err(invalid("WAVE input buffer limit exceeded"));
        }
        self.bytes.extend_from_slice(bytes);
        self.drain_pending = false;
        let mut used = 0usize;
        let mut output = Vec::new();
        let mut frame_budget = MAX_FRAMES;
        for step in 0..MAX_STEPS {
            let available = &self.bytes[used..];
            let position = self.offset + used as u64;
            match self.state {
                State::Riff => {
                    if available.len() < 12 {
                        break;
                    }
                    if &available[..4] != b"RIFF" {
                        return Err(MediaDecodeError::Unsupported);
                    }
                    if &available[8..12] != b"WAVE" {
                        return Err(MediaDecodeError::Unsupported);
                    }
                    self.end = u64::from(u32le(&available[4..8])) + 8;
                    if self.end < 12 {
                        return Err(invalid("invalid RIFF extent"));
                    }
                    used += 12;
                    self.state = State::Header;
                }
                State::Header => {
                    if position == self.end {
                        if self.data_size.is_none() {
                            return Err(invalid("missing WAVE data chunk"));
                        }
                        self.state = State::Done;
                        continue;
                    }
                    if self.end.saturating_sub(position) < 8 {
                        return Err(invalid("truncated RIFF chunk header"));
                    }
                    if available.len() < 8 {
                        break;
                    }
                    let size = u32le(&available[4..8]);
                    let pad = size & 1 != 0;
                    if position + 8 + u64::from(size) + u64::from(pad) > self.end {
                        return Err(invalid("WAVE chunk exceeds RIFF extent"));
                    }
                    self.state = match &available[..4] {
                        b"fmt " => {
                            if self.format.is_some() {
                                return Err(invalid("duplicate WAVE format"));
                            }
                            if !(16..=MAX_FORMAT).contains(&size) {
                                return Err(invalid("invalid WAVE format size"));
                            }
                            State::Format(size)
                        }
                        b"data" => {
                            let format = self
                                .format
                                .ok_or_else(|| invalid("WAVE data precedes format"))?;
                            if self.data_size.is_some() {
                                return Err(MediaDecodeError::Unsupported);
                            }
                            if size % u32::from(format.align) != 0 {
                                return Err(invalid("partial PCM frame in WAVE data"));
                            }
                            self.data_size = Some(size);
                            State::Data { left: size, pad }
                        }
                        _ => State::Skip { left: size, pad },
                    };
                    used += 8;
                }
                State::Format(size) => {
                    if available.len() < size as usize {
                        break;
                    }
                    self.format = Some(parse_format(&available[..size as usize])?);
                    used += size as usize;
                    self.state = if size & 1 != 0 {
                        State::Pad
                    } else {
                        State::Header
                    };
                }
                State::Skip { left, pad } => {
                    let count = available.len().min(left as usize);
                    used += count;
                    if count == left as usize {
                        self.state = if pad { State::Pad } else { State::Header };
                    } else {
                        self.state = State::Skip {
                            left: left - count as u32,
                            pad,
                        };
                        break;
                    }
                }
                State::Data { left, pad } => {
                    if left == 0 {
                        self.state = if pad { State::Pad } else { State::Header };
                        continue;
                    }
                    let format = self.format.unwrap();
                    let align = usize::from(format.align);
                    let frames = (available.len().min(left as usize) / align).min(frame_budget);
                    if frames == 0 {
                        break;
                    }
                    let count = frames * align;
                    let timestamp_ns = ((u128::from(self.frames) * 1_000_000_000)
                        / u128::from(format.rate)) as i64;
                    let samples = decode(&available[..count], format)?;
                    output.push(MediaSample::Audio {
                        timestamp_ns,
                        samples,
                    });
                    self.frames += frames as u64;
                    frame_budget -= frames;
                    used += count;
                    self.state = State::Data {
                        left: left - count as u32,
                        pad,
                    };
                    if frame_budget == 0 || output.len() == MAX_PACKETS {
                        self.drain_pending = used < self.bytes.len();
                        break;
                    }
                }
                State::Pad => {
                    if available.is_empty() {
                        break;
                    }
                    used += 1;
                    self.state = State::Header;
                }
                State::Done => {
                    if !available.is_empty() {
                        return Err(invalid("bytes after RIFF extent"));
                    }
                    break;
                }
            }
            if step + 1 == MAX_STEPS {
                self.drain_pending = used < self.bytes.len();
            }
        }
        self.offset += used as u64;
        self.bytes.drain(..used);
        Ok(output)
    }

    pub fn metadata(&self) -> Option<MediaMetadata> {
        if self.failure.is_some() {
            return None;
        }
        let format = self.format?;
        Some(MediaMetadata {
            presentation_size: None,
            width: None,
            height: None,
            duration: self
                .data_size
                .map(|size| size as f32 / f32::from(format.align) / format.rate as f32),
            sample_rate: Some(format.rate),
            channels: Some(format.channels),
        })
    }

    pub fn finish(&self) -> Result<(), MediaDecodeError> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        let at_end = matches!(
            self.state,
            State::Done
                | State::Header
                | State::Data {
                    left: 0,
                    pad: false
                }
                | State::Skip {
                    left: 0,
                    pad: false
                }
        );
        if !at_end || self.offset != self.end || !self.bytes.is_empty() || self.data_size.is_none()
        {
            return Err(invalid(
                "incomplete WAVE stream; drain buffered output before finish",
            ));
        }
        Ok(())
    }

    pub fn has_buffered_samples(&self) -> bool {
        self.failure.is_none()
            && (self.drain_pending
                || match self.state {
                    State::Data { left, .. } => self.format.is_some_and(|format| {
                        left >= u32::from(format.align)
                            && self.bytes.len() >= usize::from(format.align)
                    }),
                    _ => false,
                })
    }
}

impl StreamingMediaDecoder for WavStream {
    fn push_media(&mut self, bytes: &[u8]) -> Result<Vec<MediaSample>, MediaDecodeError> {
        self.push(bytes)
    }
    fn metadata(&self) -> Option<MediaMetadata> {
        WavStream::metadata(self)
    }
    fn finish(&self) -> Result<(), MediaDecodeError> {
        WavStream::finish(self)
    }
    fn has_buffered_samples(&self) -> bool {
        WavStream::has_buffered_samples(self)
    }
}

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.into())
}
fn u16le(bytes: &[u8]) -> u16 {
    u16::from_le_bytes(bytes[..2].try_into().unwrap())
}
fn u32le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes[..4].try_into().unwrap())
}

fn parse_format(bytes: &[u8]) -> Result<Format, MediaDecodeError> {
    let tag = u16le(bytes);
    let channels = u16le(&bytes[2..]);
    let rate = u32le(&bytes[4..]);
    let byte_rate = u32le(&bytes[8..]);
    let align = u16le(&bytes[12..]);
    let bits = u16le(&bytes[14..]);
    if !matches!((tag, bits), (1, 8 | 16 | 24 | 32) | (3, 32))
        || !matches!(channels, 1 | 2)
        || rate > 384000
    {
        return Err(MediaDecodeError::Unsupported);
    }
    if rate == 0
        || align != channels * (bits / 8)
        || rate.checked_mul(u32::from(align)) != Some(byte_rate)
    {
        return Err(invalid("inconsistent WAVE PCM format"));
    }
    if bytes.len() == 17 {
        return Err(invalid("truncated WAVE format extension"));
    }
    if tag == 3 && bytes.len() >= 18 && u16le(&bytes[16..]) != 0 {
        return Err(MediaDecodeError::Unsupported);
    }
    Ok(Format {
        tag,
        channels,
        rate,
        bits,
        align,
    })
}

fn decode(bytes: &[u8], format: Format) -> Result<AudioSamples, MediaDecodeError> {
    let width = usize::from(format.bits / 8);
    let mut samples = Vec::with_capacity(bytes.len() / width);
    for sample in bytes.chunks_exact(width) {
        let value = match (format.tag, format.bits) {
            (1, 8) => (f32::from(sample[0]) - 128.0) / 128.0,
            (1, 16) => f32::from(i16::from_le_bytes(sample.try_into().unwrap())) / 32768.0,
            (1, 24) => {
                let signed = i32::from_le_bytes([0, sample[0], sample[1], sample[2]]) >> 8;
                signed as f32 / 8388608.0
            }
            (1, 32) => i32::from_le_bytes(sample.try_into().unwrap()) as f32 / 2147483648.0,
            (3, 32) => {
                let value = f32::from_le_bytes(sample.try_into().unwrap());
                if !value.is_finite() {
                    return Err(invalid("nonfinite WAVE float sample"));
                }
                value.clamp(-1.0, 1.0)
            }
            _ => return Err(MediaDecodeError::Unsupported),
        };
        samples.push(value);
    }
    Ok(AudioSamples {
        sample_rate: format.rate,
        channels: format.channels,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(id: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut bytes = id.to_vec();
        bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(data);
        if data.len() & 1 != 0 {
            bytes.push(0);
        }
        bytes
    }

    fn wave(tag: u16, bits: u16, channels: u16, samples: &[u8], junk: bool) -> Vec<u8> {
        let rate = 8000u32;
        let align = channels * (bits / 8);
        let mut fmt = Vec::new();
        for field in [tag, channels] {
            fmt.extend_from_slice(&field.to_le_bytes());
        }
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&(rate * u32::from(align)).to_le_bytes());
        fmt.extend_from_slice(&align.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        let mut body = b"WAVE".to_vec();
        if junk {
            body.extend(chunk(b"JUNK", &[1, 2, 3]));
        }
        body.extend(chunk(b"fmt ", &fmt));
        body.extend(chunk(b"data", samples));
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(body.len() as u32).to_le_bytes());
        bytes.extend(body);
        bytes
    }

    fn collect(decoder: &mut WavStream, packets: Vec<MediaSample>, pcm: &mut Vec<f32>) {
        for packet in packets {
            let MediaSample::Audio { samples, .. } = packet else {
                panic!("non-audio sample")
            };
            pcm.extend(samples.samples);
        }
        while decoder.has_buffered_samples() {
            let packets = decoder.push(&[]).unwrap();
            for packet in packets {
                let MediaSample::Audio { samples, .. } = packet else {
                    panic!("non-audio sample")
                };
                pcm.extend(samples.samples);
            }
        }
    }

    #[test]
    fn arbitrary_fragmentation_padding_and_unsigned_pcm() {
        let bytes = wave(1, 8, 1, &[0, 128, 255], true);
        for split in 1..=bytes.len() {
            let mut decoder = WavStream::new();
            let mut pcm = Vec::new();
            for input in bytes.chunks(split) {
                let packets = decoder.push(input).unwrap();
                collect(&mut decoder, packets, &mut pcm);
            }
            decoder.finish().unwrap();
            assert_eq!(pcm, [-1.0, 0.0, 127.0 / 128.0]);
            let metadata = decoder.metadata().unwrap();
            assert_eq!(metadata.width, None);
            assert_eq!(metadata.height, None);
            assert_eq!(metadata.sample_rate, Some(8000));
            assert_eq!(metadata.channels, Some(1));
        }
    }

    #[test]
    fn integer_depths_and_stereo_order() {
        for (bits, data) in [
            (16, vec![0, 128, 0, 64, 0, 0, 0, 192]),
            (24, vec![0, 0, 128, 0, 0, 64, 0, 0, 0, 0, 0, 192]),
            (
                32,
                vec![0, 0, 0, 128, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 192],
            ),
        ] {
            let mut decoder = WavStream::new();
            let packets = decoder.push(&wave(1, bits, 2, &data, false)).unwrap();
            let mut pcm = Vec::new();
            collect(&mut decoder, packets, &mut pcm);
            assert_eq!(pcm, [-1.0, 0.5, 0.0, -0.5]);
            decoder.finish().unwrap();
        }
    }

    #[test]
    fn ieee_float_is_finite_and_saturates() {
        let data: Vec<u8> = [-0.75f32, 0.0, 0.25, 1.25]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let mut decoder = WavStream::new();
        let mut pcm = Vec::new();
        let packets = decoder.push(&wave(3, 32, 2, &data, false)).unwrap();
        collect(&mut decoder, packets, &mut pcm);
        assert_eq!(pcm, [-0.75, 0.0, 0.25, 1.0]);
        decoder.finish().unwrap();
        assert!(
            WavStream::new()
                .push(&wave(3, 32, 1, &f32::NAN.to_le_bytes(), false))
                .is_err()
        );
        for value in [f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                WavStream::new()
                    .push(&wave(3, 32, 1, &value.to_le_bytes(), false))
                    .is_err()
            );
        }
    }

    #[test]
    fn bounded_output_timestamp_and_drain() {
        let bytes = wave(1, 16, 2, &vec![0; (MAX_FRAMES + 17) * 4], false);
        let mut decoder = WavStream::new();
        let first = decoder.push(&bytes).unwrap();
        assert_eq!(first.len(), 1);
        let MediaSample::Audio {
            timestamp_ns,
            samples,
        } = &first[0]
        else {
            panic!()
        };
        assert_eq!(*timestamp_ns, 0);
        assert_eq!(samples.samples.len(), MAX_FRAMES * 2);
        assert!(decoder.has_buffered_samples());
        assert!(decoder.finish().is_err());
        let second = decoder.push(&[]).unwrap();
        let MediaSample::Audio {
            timestamp_ns,
            samples,
        } = &second[0]
        else {
            panic!()
        };
        assert_eq!(*timestamp_ns, MAX_FRAMES as i64 * 1_000_000_000 / 8000);
        assert_eq!(samples.samples.len(), 34);
        assert!(!decoder.has_buffered_samples());
        decoder.finish().unwrap();
    }

    #[test]
    fn truncation_bounds_and_unsupported_formats() {
        let bytes = wave(1, 16, 1, &[0, 0, 1, 0], false);
        for end in 0..bytes.len() {
            let mut decoder = WavStream::new();
            if decoder.push(&bytes[..end]).is_ok() {
                assert!(decoder.finish().is_err());
            }
        }
        for tag in [2, 6, 7, 0xfffe] {
            assert_eq!(
                WavStream::new().push(&wave(tag, 16, 1, &[0, 0], false)),
                Err(MediaDecodeError::Unsupported)
            );
        }
        let mut invalid_align = bytes.clone();
        invalid_align[32..34].copy_from_slice(&3u16.to_le_bytes());
        assert!(WavStream::new().push(&invalid_align).is_err());
        let mut oversized_chunk = bytes.clone();
        oversized_chunk[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(WavStream::new().push(&oversized_chunk).is_err());
        assert!(WavStream::new().push(&vec![0; MAX_BUFFER + 1]).is_err());
    }

    #[test]
    fn metadata_precedes_samples_and_partial_frame_is_incomplete() {
        let bytes = wave(1, 16, 2, &[0; 8], false);
        let mut decoder = WavStream::new();
        assert!(decoder.push_media(&bytes[..44]).unwrap().is_empty());
        let metadata = decoder.metadata().unwrap();
        assert_eq!(metadata.sample_rate, Some(8000));
        assert_eq!(metadata.channels, Some(2));
        assert_eq!(metadata.duration, Some(2.0 / 8000.0));
        assert!(decoder.push(&bytes[44..47]).unwrap().is_empty());
        assert!(!decoder.has_buffered_samples());
        assert!(decoder.finish().is_err());
        assert_eq!(decoder.push(&bytes[47..]).unwrap().len(), 1);
        decoder.finish().unwrap();
        assert!(decoder.push(&[]).unwrap().is_empty());
    }

    #[test]
    fn truncated_format_missing_padding_and_partial_declared_block_fail() {
        let odd = wave(1, 8, 1, &[128], false);
        let mut decoder = WavStream::new();
        decoder.push(&odd[..odd.len() - 1]).unwrap();
        assert!(decoder.finish().is_err());
        decoder.push(&odd[odd.len() - 1..]).unwrap();
        decoder.finish().unwrap();

        let bytes = wave(1, 16, 1, &[0, 0], false);
        let mut decoder = WavStream::new();
        decoder.push(&bytes[..35]).unwrap();
        assert!(decoder.metadata().is_none());
        assert!(decoder.finish().is_err());
        assert!(WavStream::new().push(&wave(1, 16, 1, &[0], false)).is_err());
    }

    #[test]
    fn unknown_chunks_do_not_accumulate_the_complete_file() {
        let padding = vec![0; MAX_BUFFER + 17];
        let mut bytes = wave(1, 8, 1, &[128], false);
        let unknown = chunk(b"JUNK", &padding);
        bytes.splice(12..12, unknown);
        let extent = (bytes.len() - 8) as u32;
        bytes[4..8].copy_from_slice(&extent.to_le_bytes());
        let mut decoder = WavStream::new();
        let mut pcm = Vec::new();
        for input in bytes.chunks(16384) {
            let packets = decoder.push(input).unwrap();
            collect(&mut decoder, packets, &mut pcm);
            assert!(decoder.bytes.len() < 16384);
        }
        assert_eq!(pcm, [0.0]);
        decoder.finish().unwrap();
    }
}
