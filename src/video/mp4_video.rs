//! Codec selection and incremental sample playback for classic MP4.

use super::backend::{MediaDecodeError, MediaMetadata, StreamingVideoDecoder, VideoFrame};
use super::mp4::{Mp4Error, Mp4Index, Mp4VideoCodec, Mp4VideoIndex};
use super::mp4_avc::{Mp4AvcPackets, Mp4AvcStream};
use super::vp8_decoder::Vp8Decoder;
use super::vp9::split_superframe;
use super::vp9_decoder::Vp9Decoder;
use std::sync::Arc;

const MAX_BUFFER_BYTES: usize = 64 * 1024 * 1024;
const MAX_FRAMES_PER_PUSH: usize = 4;

pub struct Mp4VideoPackets {
    decoder: PacketDecoder,
}

enum PacketDecoder {
    Avc(Mp4AvcPackets),
    VpX(Mp4VpXStream),
}

impl Mp4VideoPackets {
    pub fn new(index: Mp4VideoIndex, start: usize) -> Result<Self, MediaDecodeError> {
        if index.timescale == 0
            || (start != 0 && !index.samples.get(start).is_some_and(|s| s.keyframe))
        {
            return Err(MediaDecodeError::InvalidData(
                "invalid MP4 video restart".into(),
            ));
        }
        let decoder = match index.codec {
            Mp4VideoCodec::Avc(config) => PacketDecoder::Avc(Mp4AvcPackets::with_start_sample(
                Mp4Index {
                    config,
                    timescale: index.timescale,
                    duration_ticks: index.duration_ticks,
                    samples: index.samples,
                },
                start,
            )?),
            _ => {
                let mut stream = Mp4VpXStream::new(index);
                stream.next_sample = start;
                PacketDecoder::VpX(stream)
            }
        };
        Ok(Self { decoder })
    }

    pub fn push(
        &mut self,
        number: usize,
        data: &[u8],
    ) -> Result<Vec<VideoFrame>, MediaDecodeError> {
        match &mut self.decoder {
            PacketDecoder::Avc(decoder) => decoder.push(number, data),
            PacketDecoder::VpX(stream) => {
                let sample = stream.index.samples.get(number).ok_or_else(|| {
                    MediaDecodeError::InvalidData("MP4 sample out of bounds".into())
                })?;
                if number != stream.next_sample
                    || data.len() != sample.size as usize
                    || data.len() > MAX_BUFFER_BYTES
                {
                    return Err(MediaDecodeError::InvalidData(
                        "invalid MP4 access unit".into(),
                    ));
                }
                stream.base_offset = sample.offset;
                stream.bytes.clear();
                stream.bytes.extend_from_slice(data);
                stream.push(&[])
            }
        }
    }
}

#[derive(Default)]
pub struct Mp4VideoDecoder {
    probe: Vec<u8>,
    decoder: Option<Box<dyn StreamingVideoDecoder>>,
}

impl Mp4VideoDecoder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl StreamingVideoDecoder for Mp4VideoDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<VideoFrame>, MediaDecodeError> {
        if let Some(decoder) = &mut self.decoder {
            return decoder.push(bytes);
        }
        if bytes.len() > MAX_BUFFER_BYTES.saturating_sub(self.probe.len()) {
            return Err(MediaDecodeError::Unsupported);
        }
        self.probe.extend_from_slice(bytes);
        let index = match Mp4VideoIndex::parse_prefix(&self.probe) {
            Ok(index) => index,
            Err(Mp4Error::Incomplete) => return Ok(Vec::new()),
            Err(error) => return Err(mp4_error(error)),
        };
        let mut decoder: Box<dyn StreamingVideoDecoder> = match index.codec {
            Mp4VideoCodec::Avc(_) => Box::new(Mp4AvcStream::new()),
            Mp4VideoCodec::Vp8 | Mp4VideoCodec::Vp9 => Box::new(Mp4VpXStream::new(index)),
        };
        let frames = decoder.push(&self.probe)?;
        self.probe.clear();
        self.decoder = Some(decoder);
        Ok(frames)
    }

    fn metadata(&self) -> Option<MediaMetadata> {
        self.decoder.as_ref()?.metadata()
    }

    fn finish(&self) -> Result<(), MediaDecodeError> {
        self.decoder
            .as_ref()
            .ok_or(MediaDecodeError::Unsupported)?
            .finish()
    }

    fn has_buffered_samples(&self) -> bool {
        self.decoder
            .as_ref()
            .is_some_and(|decoder| decoder.has_buffered_samples())
    }
}

enum VpXDecoder {
    Vp8(Vp8Decoder),
    Vp9(Vp9Decoder),
}

struct Mp4VpXStream {
    bytes: Vec<u8>,
    base_offset: u64,
    index: Mp4VideoIndex,
    next_sample: usize,
    decoder: VpXDecoder,
}

impl Mp4VpXStream {
    fn new(index: Mp4VideoIndex) -> Self {
        let decoder = match &index.codec {
            Mp4VideoCodec::Vp8 => VpXDecoder::Vp8(Vp8Decoder::new()),
            Mp4VideoCodec::Vp9 => VpXDecoder::Vp9(Vp9Decoder::new()),
            Mp4VideoCodec::Avc(_) => unreachable!("AVC uses its own sample stream"),
        };
        Self {
            bytes: Vec::new(),
            base_offset: 0,
            index,
            next_sample: 0,
            decoder,
        }
    }

    fn sample_available(&self) -> bool {
        self.index
            .samples
            .get(self.next_sample)
            .and_then(|sample| sample.offset.checked_add(u64::from(sample.size)))
            .is_some_and(|end| end <= self.base_offset + self.bytes.len() as u64)
    }
}

impl StreamingVideoDecoder for Mp4VpXStream {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<VideoFrame>, MediaDecodeError> {
        if bytes.len() > MAX_BUFFER_BYTES.saturating_sub(self.bytes.len()) {
            return Err(MediaDecodeError::Unsupported);
        }
        self.bytes.extend_from_slice(bytes);
        let mut frames = Vec::new();
        while frames.len() < MAX_FRAMES_PER_PUSH && self.sample_available() {
            let sample = &self.index.samples[self.next_sample];
            let start = usize::try_from(
                sample
                    .offset
                    .checked_sub(self.base_offset)
                    .ok_or(MediaDecodeError::Unsupported)?,
            )
            .map_err(|_| MediaDecodeError::Unsupported)?;
            let end = start
                .checked_add(sample.size as usize)
                .ok_or(MediaDecodeError::Unsupported)?;
            let data = self
                .bytes
                .get(start..end)
                .ok_or(MediaDecodeError::Unsupported)?;
            let timestamp = sample.presentation_time as f32 / self.index.timescale as f32;
            match &mut self.decoder {
                VpXDecoder::Vp8(decoder) => {
                    let frame = decoder.decode(data)?;
                    frames.push(VideoFrame {
                        width: frame.width as u32,
                        height: frame.height as u32,
                        rgba: Arc::new(frame.rgba()),
                        timestamp,
                    });
                }
                VpXDecoder::Vp9(decoder) => {
                    for frame_data in split_superframe(data)? {
                        if let Some(frame) = decoder.decode(frame_data)? {
                            frames.push(VideoFrame {
                                width: frame.width as u32,
                                height: frame.height as u32,
                                rgba: Arc::new(frame.rgba()),
                                timestamp,
                            });
                        }
                    }
                }
            }
            self.next_sample += 1;
        }
        let keep_from = self
            .index
            .samples
            .get(self.next_sample)
            .map(|sample| sample.offset)
            .unwrap_or(self.base_offset + self.bytes.len() as u64);
        let discard = keep_from
            .saturating_sub(self.base_offset)
            .min(self.bytes.len() as u64) as usize;
        self.bytes.drain(..discard);
        self.base_offset += discard as u64;
        Ok(frames)
    }

    fn metadata(&self) -> Option<MediaMetadata> {
        Some(MediaMetadata {
            duration: Some(self.index.duration_ticks as f32 / self.index.timescale as f32),
            width: Some(self.index.width),
            height: Some(self.index.height),
            sample_rate: None,
            channels: None,
        })
    }

    fn finish(&self) -> Result<(), MediaDecodeError> {
        if self.next_sample == self.index.samples.len() {
            Ok(())
        } else {
            Err(MediaDecodeError::InvalidData(
                "truncated MP4 video stream".into(),
            ))
        }
    }

    fn has_buffered_samples(&self) -> bool {
        self.sample_available()
    }
}

fn mp4_error(error: Mp4Error) -> MediaDecodeError {
    match error {
        Mp4Error::Unsupported(_) => MediaDecodeError::Unsupported,
        other => MediaDecodeError::InvalidData(format!("MP4: {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::super::mp4::Sample;
    use super::*;

    #[test]
    fn routes_vp8_mp4_samples_through_shared_decoder() {
        let ivf = include_bytes!("../../tests/fixtures/vp8-keyframe.ivf");
        assert_eq!(&ivf[..4], b"DKIF");
        let size = u32::from_le_bytes(ivf[32..36].try_into().unwrap()) as usize;
        let packet = &ivf[44..44 + size];
        let index = Mp4VideoIndex {
            timescale: 1,
            duration_ticks: 1,
            codec: Mp4VideoCodec::Vp8,
            width: 32,
            height: 32,
            samples: vec![Sample {
                offset: 0,
                size: size as u32,
                decode_time: 0,
                presentation_time: 0,
                keyframe: true,
            }],
        };
        let mut packets = Mp4VideoPackets::new(index.clone(), 0).unwrap();
        assert!(packets.push(1, packet).is_err());
        let packet_frames = packets.push(0, packet).unwrap();
        let mut stream = Mp4VpXStream::new(index);
        assert!(stream.push(&packet[..size / 2]).unwrap().is_empty());
        let frames = stream.push(&packet[size / 2..]).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].width, frames[0].height), (32, 32));
        assert_eq!(frames[0].rgba.len(), 32 * 32 * 4);
        assert_eq!(packet_frames[0], frames[0]);
        stream.finish().unwrap();
    }

    #[test]
    fn routes_vp9_mp4_samples_through_shared_decoder() {
        let bytes = std::env::var("WEBMEDIA_VP9_MP4_SAMPLE")
            .map(|path| std::fs::read(path).unwrap())
            .unwrap_or_else(|_| include_bytes!("../../tests/fixtures/vp9-keyframe.mp4").to_vec());
        let index = Mp4VideoIndex::parse_prefix(&bytes).unwrap();
        assert_eq!(index.codec, Mp4VideoCodec::Vp9);
        let expected = index.samples.len();
        let mut packets = Mp4VideoPackets::new(index.clone(), 0).unwrap();
        let mut packet_frames = Vec::new();
        for (number, sample) in index.samples.iter().enumerate() {
            packet_frames.extend(
                packets
                    .push(
                        number,
                        &bytes
                            [sample.offset as usize..sample.offset as usize + sample.size as usize],
                    )
                    .unwrap(),
            );
        }
        let mut decoder = Mp4VideoDecoder::new();
        let mut frames = Vec::new();
        for chunk in bytes.chunks(4096) {
            frames.extend(decoder.push(chunk).unwrap());
            while decoder.has_buffered_samples() {
                frames.extend(decoder.push(&[]).unwrap());
            }
        }
        decoder.finish().unwrap();
        assert_eq!(frames.len(), expected);
        assert_eq!(packet_frames, frames);
        assert_eq!(decoder.metadata().unwrap().width, Some(frames[0].width));
        assert_eq!(decoder.metadata().unwrap().height, Some(frames[0].height));
        assert!(
            frames
                .windows(2)
                .all(|pair| pair[0].timestamp <= pair[1].timestamp)
        );
        assert!(
            frames
                .iter()
                .all(|frame| frame.rgba.len() == frame.width as usize * frame.height as usize * 4)
        );
    }

    #[test]
    fn recognizes_vp08_mp4_sample_entry() {
        let mut bytes = include_bytes!("../../tests/fixtures/vp9-keyframe.mp4").to_vec();
        let entry = bytes.windows(4).position(|bytes| bytes == b"vp09").unwrap();
        bytes[entry..entry + 4].copy_from_slice(b"vp08");
        let index = Mp4VideoIndex::parse_prefix(&bytes).unwrap();
        assert_eq!(index.codec, Mp4VideoCodec::Vp8);
    }

    #[test]
    fn routes_avc_mp4_to_existing_decoder() {
        let Ok(path) = std::env::var("WEBMEDIA_AVC_MP4_SAMPLE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let index = Mp4VideoIndex::parse_prefix(&bytes).unwrap();
        assert!(matches!(index.codec, Mp4VideoCodec::Avc(_)));
        let mut decoder = Mp4VideoDecoder::new();
        let mut frames = Vec::new();
        for chunk in bytes.chunks(16 * 1024) {
            frames.extend(decoder.push(chunk).unwrap());
            while decoder.has_buffered_samples() {
                frames.extend(decoder.push(&[]).unwrap());
            }
            if !frames.is_empty() {
                break;
            }
        }
        assert!(!frames.is_empty());
        assert_eq!(decoder.metadata().unwrap().width, Some(frames[0].width));
    }
}
