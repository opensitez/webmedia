//! Codec selection and incremental sample playback for classic MP4.

use super::backend::{MediaDecodeError, MediaMetadata, StreamingVideoDecoder, VideoFrame};
use super::mp4::{Mp4Error, Mp4Index, Mp4VideoCodec, Mp4VideoIndex, Sample};
use super::mp4_avc::{Mp4AvcPackets, Mp4AvcStream};
use super::vp8_decoder::Vp8Decoder;
use super::vp9::split_superframe;
use super::vp9_decoder::Vp9Decoder;
use std::sync::Arc;

const MAX_BUFFER_BYTES: usize = 64 * 1024 * 1024;
const MAX_FRAMES_PER_PUSH: usize = 4;

pub struct Mp4VideoPackets {
    decoder: PacketDecoder,
    fragmented: bool,
    finished: bool,
    sample_base: usize,
    requires_keyframe: bool,
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
        Ok(Self { decoder, fragmented: false, finished: false, sample_base: 0, requires_keyframe: false })
    }

    pub fn new_fragmented(mut index: Mp4VideoIndex, start: usize) -> Result<Self, MediaDecodeError> {
        if index.timescale == 0 { return Err(MediaDecodeError::InvalidData("zero video timescale".into())); }
        index.samples.clear();
        let decoder = match index.codec {
            Mp4VideoCodec::Avc(config) => PacketDecoder::Avc(Mp4AvcPackets::new_fragmented(
                Mp4Index { config, timescale: index.timescale, duration_ticks: index.duration_ticks,
                    samples: index.samples }, start)?),
            _ => {
                PacketDecoder::VpX(Mp4VpXStream::new(index))
            }
        };
        Ok(Self { decoder, fragmented: true, finished: false,
            sample_base: start, requires_keyframe: true })
    }

    pub fn update_samples(&mut self, samples: &[Sample], duration_ticks: u64) -> Result<(), MediaDecodeError> {
        if self.finished { return Err(MediaDecodeError::InvalidData("samples after EOF".into())); }
        match &mut self.decoder {
            PacketDecoder::Avc(decoder) => decoder.update_samples(samples, duration_ticks),
            PacketDecoder::VpX(stream) => {
                let supplied = samples.get(self.sample_base..)
                    .ok_or_else(|| MediaDecodeError::InvalidData("missing restart prefix".into()))?;
                if supplied.len() > 1_000_000 || !supplied.starts_with(&stream.index.samples) {
                    return Err(MediaDecodeError::InvalidData("MP4 sample table is not append-only".into()));
                }
                stream.index.samples.extend_from_slice(&supplied[stream.index.samples.len()..]);
                stream.index.duration_ticks = duration_ticks;
                Ok(())
            }
        }
    }

    pub fn push_sample(&mut self, number: usize, sample: Sample, data: &[u8]) -> Result<Vec<VideoFrame>, MediaDecodeError> {
        if self.finished { return Err(MediaDecodeError::InvalidData("packet after EOF".into())); }
        match &mut self.decoder {
            PacketDecoder::Avc(decoder) => decoder.push_sample(number, sample, data),
            PacketDecoder::VpX(stream) => {
                let local = number.checked_sub(self.sample_base)
                    .ok_or_else(|| MediaDecodeError::InvalidData("sample precedes restart".into()))?;
                if local != stream.next_sample || data.len() != sample.size as usize
                    || data.len() > MAX_BUFFER_BYTES || (self.requires_keyframe && !sample.keyframe)
                    || sample.offset.checked_add(u64::from(sample.size)).is_none() {
                    return Err(MediaDecodeError::InvalidData("invalid fragmented video packet".into()));
                }
                if let Some(known) = stream.index.samples.get(local) {
                    if known != &sample { return Err(MediaDecodeError::InvalidData("conflicting sample metadata".into())); }
                } else if self.fragmented && local == stream.index.samples.len() && local < 1_000_000 {
                    if stream.index.samples.last().is_some_and(|last| sample.decode_time < last.decode_time) {
                        return Err(MediaDecodeError::InvalidData("nonmonotonic decode time".into()));
                    }
                    stream.index.samples.push(sample);
                } else { return Err(MediaDecodeError::InvalidData("missing or excessive sample metadata".into())); }
                stream.base_offset = stream.index.samples[local].offset;
                stream.bytes.clear();
                stream.bytes.extend_from_slice(data);
                let frames = stream.push(&[])?;
                self.requires_keyframe = false;
                Ok(frames)
            }
        }
    }

    /// Repeat until empty at definitive EOF; AVC may retain a reorder tail.
    pub fn finish_input(&mut self) -> Result<Vec<VideoFrame>, MediaDecodeError> {
        let frames = match &mut self.decoder {
            PacketDecoder::Avc(decoder) => decoder.finish_input()?,
            PacketDecoder::VpX(stream) => { stream.finish()?; Vec::new() }
        };
        self.finished = true;
        Ok(frames)
    }

    pub fn push(
        &mut self,
        number: usize,
        data: &[u8],
    ) -> Result<Vec<VideoFrame>, MediaDecodeError> {
        if self.finished { return Err(MediaDecodeError::InvalidData("packet after EOF".into())); }
        match &mut self.decoder {
            PacketDecoder::Avc(decoder) => decoder.push(number, data),
            PacketDecoder::VpX(stream) => {
                let local = number.checked_sub(self.sample_base)
                    .ok_or_else(|| MediaDecodeError::InvalidData("sample precedes restart".into()))?;
                let sample = stream.index.samples.get(local).ok_or_else(|| {
                    MediaDecodeError::InvalidData("MP4 sample out of bounds".into())
                })?;
                if local != stream.next_sample
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
        let mut decoded_samples = 0;
        while decoded_samples < MAX_FRAMES_PER_PUSH
            && frames.len() < MAX_FRAMES_PER_PUSH && self.sample_available()
        {
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
                    let show_frame = super::vp8::FrameHeader::parse(data)?.show_frame;
                    let frame = decoder.decode(data)?;
                    if show_frame {
                        frames.push(VideoFrame {
                            presentation_size: decoder.presentation_size(),
                            width: frame.width as u32,
                            height: frame.height as u32,
                            rgba: Arc::new(frame.rgba()),
                            timestamp,
                        });
                    }
                }
                VpXDecoder::Vp9(decoder) => {
                    for frame_data in split_superframe(data)? {
                        if let Some(frame) = decoder.decode(frame_data)? {
                            frames.push(VideoFrame {
                                presentation_size: None,
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
            decoded_samples += 1;
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
        let (width, height) = match &self.decoder {
            VpXDecoder::Vp8(decoder) => decoder.coded_size(),
            VpXDecoder::Vp9(_) => None,
        }.unwrap_or((self.index.width, self.index.height));
        Some(MediaMetadata {
            presentation_size: match &self.decoder {
                VpXDecoder::Vp8(decoder) => decoder.presentation_size(),
                VpXDecoder::Vp9(_) => None,
            },
            duration: Some(self.index.duration_ticks as f32 / self.index.timescale as f32),
            width: Some(width),
            height: Some(height),
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

    fn vp8_test_container(index: &Mp4VideoIndex, payload: &[u8], trailing_moov: bool) -> Vec<u8> {
        fn atom(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
            let mut bytes = ((data.len() + 8) as u32).to_be_bytes().to_vec();
            bytes.extend_from_slice(kind);
            bytes.extend_from_slice(data);
            bytes
        }
        fn table(entries: &[u32], fields: usize) -> Vec<u8> {
            let mut bytes = vec![0; 4];
            bytes.extend(((entries.len() / fields) as u32).to_be_bytes());
            for entry in entries { bytes.extend(entry.to_be_bytes()); }
            bytes
        }
        let mut entry = vec![0; 78];
        entry[6..8].copy_from_slice(&1u16.to_be_bytes());
        entry[24..26].copy_from_slice(&(index.width as u16).to_be_bytes());
        entry[26..28].copy_from_slice(&(index.height as u16).to_be_bytes());
        let mut descriptions = table(&[], 1);
        descriptions[4..8].copy_from_slice(&1u32.to_be_bytes());
        descriptions.extend(atom(b"vp08", &entry));
        let mut sizes = vec![0; 8];
        sizes.extend((index.samples.len() as u32).to_be_bytes());
        for sample in &index.samples { sizes.extend(sample.size.to_be_bytes()); }
        let sync: Vec<_> = index.samples.iter().enumerate()
            .filter(|(_, sample)| sample.keyframe).map(|(n, _)| n as u32 + 1).collect();
        let make_moov = |offset| {
            let mut sample_table = Vec::new();
            for (kind, data) in [
                (b"stsd", descriptions.clone()), (b"stsz", sizes.clone()),
                (b"stsc", table(&[1, index.samples.len() as u32, 1], 3)),
                (b"stts", table(&[index.samples.len() as u32, 1], 2)),
                (b"stss", table(&sync, 1)), (b"stco", table(&[offset], 1)),
            ] { sample_table.extend(atom(kind, &data)); }
            let mut header = vec![0; 12];
            header.extend(index.timescale.to_be_bytes());
            header.extend((index.duration_ticks as u32).to_be_bytes());
            let mut handler = vec![0; 8];
            handler.extend(b"vide");
            let mut media = atom(b"mdhd", &header);
            media.extend(atom(b"hdlr", &handler));
            media.extend(atom(b"minf", &atom(b"stbl", &sample_table)));
            atom(b"moov", &atom(b"trak", &atom(b"mdia", &media)))
        };
        let mut bytes = atom(b"ftyp", b"isom\0\0\0\0isom");
        let offset = bytes.len() + 8 + if trailing_moov { 0 } else { make_moov(0).len() };
        let moov = make_moov(offset as u32);
        if !trailing_moov { bytes.extend_from_slice(&moov); }
        bytes.extend(atom(b"mdat", payload));
        if trailing_moov { bytes.extend(moov); }
        bytes
    }

    #[test]
    fn vp8_presentation_dimensions_propagate_through_webm_and_mp4() {
        use super::super::vp8::FrameHeader;
        use super::super::webm::{WebmVp8Stream, WebmVideoDecoder};
        let mut bytes = include_bytes!("../../tests/fixtures/vp8-motion.webm").to_vec();
        let mut demux = WebmVp8Stream::new();
        let packets = demux.push(&bytes).unwrap();
        for packet in &packets {
            if FrameHeader::parse(&packet.data).unwrap().key_frame {
                let offset = bytes.windows(packet.data.len()).position(|window| window == packet.data).unwrap();
                bytes[offset + 7] = (bytes[offset + 7] & 0x3f) | (1 << 6);
                bytes[offset + 9] = (bytes[offset + 9] & 0x3f) | (3 << 6);
            }
        }
        let mut demux = WebmVp8Stream::new();
        let packets = demux.push(&bytes).unwrap();
        let header = FrameHeader::parse(&packets[0].data).unwrap();
        let presentation = header.display_size();
        let mut payload = Vec::new();
        let mut samples = Vec::new();
        for (number, packet) in packets.iter().enumerate() {
            samples.push(Sample {
                offset: payload.len() as u64, size: packet.data.len() as u32,
                decode_time: number as u64, presentation_time: number as i64,
                keyframe: FrameHeader::parse(&packet.data).unwrap().key_frame,
            });
            payload.extend_from_slice(&packet.data);
        }
        let index = Mp4VideoIndex {
            timescale: 30, duration_ticks: samples.len() as u64, codec: Mp4VideoCodec::Vp8,
            width: header.width.unwrap() as u32, height: header.height.unwrap() as u32, samples,
        };
        let mp4 = vp8_test_container(&index, &payload, false);
        let mut decoders: Vec<(Box<dyn StreamingVideoDecoder>, &[u8])> = vec![
            (Box::new(WebmVideoDecoder::new()), &bytes),
            (Box::new(Mp4VideoDecoder::new()), &mp4),
        ];
        for (decoder, input) in &mut decoders {
            let mut count = 0;
            for chunk in input.chunks(4096) {
                let mut frames = decoder.push(chunk).unwrap();
                while decoder.has_buffered_samples() { frames.extend(decoder.push(&[]).unwrap()); }
                for frame in frames {
                    count += 1;
                    assert_eq!(frame.presentation_size, presentation);
                    assert_eq!((frame.width, frame.height), (index.width, index.height));
                    assert_eq!(frame.rgba.len(), index.width as usize * index.height as usize * 4);
                }
            }
            decoder.finish().unwrap();
            assert_eq!(count, 60);
            assert_eq!(decoder.metadata().unwrap().presentation_size, presentation);
        }
        use super::super::backend::{MediaSample, StreamingMediaDecoder};
        let mut media = super::super::webm::WebmMediaDecoder::new();
        let mut count = 0;
        for chunk in bytes.chunks(4096) {
            let mut samples = media.push_media(chunk).unwrap();
            while media.has_buffered_samples() { samples.extend(media.push_media(&[]).unwrap()); }
            for sample in samples {
                if let MediaSample::Video(frame) = sample {
                    count += 1;
                    assert_eq!(frame.presentation_size, presentation);
                    assert_eq!((frame.width, frame.height), (index.width, index.height));
                }
            }
        }
        media.finish().unwrap();
        assert_eq!(count, 60);
        assert_eq!(media.metadata().unwrap().presentation_size, presentation);
    }

    #[test]
    fn fragmented_vp8_future_epoch_metadata_and_eof_match_classic_pixels() {
        let ivf = include_bytes!("../../tests/fixtures/vp8-keyframe.ivf");
        let size = u32::from_le_bytes(ivf[32..36].try_into().unwrap()) as usize;
        let data = &ivf[44..44 + size];
        let sample = Sample { offset: 0, size: size as u32,
            decode_time: 0, presentation_time: 0, keyframe: true };
        let index = Mp4VideoIndex { timescale: 1, duration_ticks: 1,
            codec: Mp4VideoCodec::Vp8, width: 32, height: 32, samples: vec![sample.clone()] };
        let mut classic = Mp4VideoPackets::new(index.clone(), 0).unwrap();
        let expected = classic.push(0, data).unwrap();
        assert_eq!(expected.len(), 1);
        assert!(classic.finish_input().unwrap().is_empty());

        let mut fragmented = Mp4VideoPackets::new_fragmented(index.clone(), 90_000).unwrap();
        let mut not_key = sample.clone();
        not_key.keyframe = false;
        assert!(fragmented.push_sample(90_000, not_key, data).is_err());
        assert!(fragmented.push_sample(89_999, sample.clone(), data).is_err());
        assert_eq!(fragmented.push_sample(90_000, sample.clone(), data).unwrap(), expected);
        assert!(fragmented.finish_input().unwrap().is_empty());
        assert!(fragmented.finish_input().unwrap().is_empty());
        assert!(fragmented.push_sample(90_001, sample.clone(), data).is_err());

        let mut known = Mp4VideoPackets::new_fragmented(index, 0).unwrap();
        known.update_samples(std::slice::from_ref(&sample), 0).unwrap();
        let mut conflicting = sample.clone();
        conflicting.presentation_time = 1;
        assert!(known.push_sample(0, conflicting, data).is_err());
        assert!(known.finish_input().is_err());
        assert_eq!(known.push_sample(0, sample, data).unwrap(), expected);
        assert!(known.finish_input().unwrap().is_empty());
    }

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
    fn vp8_mp4_hidden_samples_update_references_without_presenting() {
        use super::super::vp8::FrameHeader;
        use super::super::webm::WebmVp8Stream;
        let mut demux = WebmVp8Stream::new();
        let input = demux.push(include_bytes!("../../tests/fixtures/vp8-altref.webm")).unwrap();
        demux.finish().unwrap();
        let mut bytes = Vec::new();
        let mut samples = Vec::new();
        let mut expected = Vec::new();
        let mut reference = Vp8Decoder::new();
        let mut hidden = 0;
        for (number, packet) in input.iter().enumerate() {
            let header = FrameHeader::parse(&packet.data).unwrap();
            samples.push(Sample {
                offset: bytes.len() as u64,
                size: packet.data.len() as u32,
                decode_time: number as u64,
                presentation_time: number as i64,
                keyframe: header.key_frame,
            });
            bytes.extend_from_slice(&packet.data);
            let frame = reference.decode(&packet.data).unwrap();
            if header.show_frame {
                expected.push(VideoFrame {
                    presentation_size: None,
                    width: frame.width as u32, height: frame.height as u32,
                    rgba: Arc::new(frame.rgba()), timestamp: number as f32 / 30.0,
                });
            } else {
                hidden += 1;
            }
        }
        assert_eq!(hidden, 6);
        assert_eq!(expected.len(), 120);
        let index = Mp4VideoIndex {
            timescale: 30, duration_ticks: samples.len() as u64,
            codec: Mp4VideoCodec::Vp8,
            width: expected[0].width, height: expected[0].height, samples,
        };
        let mut packets = Mp4VideoPackets::new(index.clone(), 0).unwrap();
        let mut actual = Vec::new();
        for (number, packet) in input.iter().enumerate() {
            let frames = packets.push(number, &packet.data).unwrap();
            if !FrameHeader::parse(&packet.data).unwrap().show_frame {
                assert!(frames.is_empty(), "hidden sample {number} was presented");
            }
            actual.extend(frames);
        }
        assert_eq!(actual, expected);
        for chunk_size in [1, 4096, bytes.len()] {
            let mut stream = Mp4VpXStream::new(index.clone());
            let mut actual = Vec::new();
            for chunk in bytes.chunks(chunk_size) {
                let before = stream.next_sample;
                actual.extend(stream.push(chunk).unwrap());
                assert!(stream.next_sample - before <= MAX_FRAMES_PER_PUSH);
                while stream.has_buffered_samples() {
                    let before = stream.next_sample;
                    actual.extend(stream.push(&[]).unwrap());
                    assert!(stream.next_sample - before <= MAX_FRAMES_PER_PUSH);
                }
            }
            stream.finish().unwrap();
            assert_eq!(actual, expected, "chunk size {chunk_size}");
        }
        for trailing_moov in [false, true] {
            let container = vp8_test_container(&index, &bytes, trailing_moov);
            let parsed = Mp4VideoIndex::parse_prefix(&container).unwrap();
            assert_eq!(parsed.codec, Mp4VideoCodec::Vp8);
            assert_eq!(parsed.samples.len(), input.len());
            for (sample, packet) in parsed.samples.iter().zip(&input) {
                assert_eq!(&container[sample.offset as usize..sample.offset as usize + sample.size as usize],
                    packet.data.as_slice());
            }
            for chunk_size in [1, 4096, container.len()] {
                let mut decoder = Mp4VideoDecoder::new();
                let mut actual = Vec::new();
                for chunk in container.chunks(chunk_size) {
                    actual.extend(decoder.push(chunk).unwrap());
                    while decoder.has_buffered_samples() {
                        actual.extend(decoder.push(&[]).unwrap());
                    }
                }
                decoder.finish().unwrap();
                assert_eq!(actual, expected, "trailing moov={trailing_moov} chunk size={chunk_size}");
                let metadata = decoder.metadata().unwrap();
                assert_eq!(metadata.width, Some(index.width));
                assert_eq!(metadata.height, Some(index.height));
            }
        }
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
