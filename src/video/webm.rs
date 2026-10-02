//! Incremental WebM demuxing for VP8 video tracks.

use super::backend::{MediaDecodeError, MediaMetadata, StreamingVideoDecoder, VideoFrame};
use super::vp8::FrameHeader;
use super::vp8_decoder::Vp8Decoder;

const EBML: u32 = 0x1a45dfa3;
const SEGMENT: u32 = 0x18538067;
const INFO: u32 = 0x1549a966;
const TRACKS: u32 = 0x1654ae6b;
const TRACK_ENTRY: u32 = 0xae;
const VIDEO: u32 = 0xe0;
const CLUSTER: u32 = 0x1f43b675;
const BLOCK_GROUP: u32 = 0xa0;
const SIMPLE_BLOCK: u32 = 0xa3;
const BLOCK: u32 = 0xa1;
const DOC_TYPE: u32 = 0x4282;
const TIME_CODE_SCALE: u32 = 0x2ad7b1;
const DURATION: u32 = 0x4489;
const TRACK_NUMBER: u32 = 0xd7;
const TRACK_TYPE: u32 = 0x83;
const CODEC_ID: u32 = 0x86;
const PIXEL_WIDTH: u32 = 0xb0;
const PIXEL_HEIGHT: u32 = 0xba;
const TIMESTAMP: u32 = 0xe7;

#[derive(Clone, Debug, PartialEq)]
pub struct VideoPacket {
    pub timestamp: f32,
    pub key_frame: bool,
    pub data: Vec<u8>,
}

#[derive(Default)]
struct Track {
    number: u64,
    kind: u64,
    codec: Vec<u8>,
    width: Option<u32>,
    height: Option<u32>,
}

struct Parent {
    id: u32,
    end: Option<u64>,
}

pub struct WebmVp8Stream {
    pending: Vec<u8>,
    offset: u64,
    skip: u64,
    parents: Vec<Parent>,
    doc_type: Option<Vec<u8>>,
    track: Option<Track>,
    video_track: Option<u64>,
    width: Option<u32>,
    height: Option<u32>,
    time_code_scale: u64,
    duration_ticks: Option<f64>,
    cluster_timestamp: u64,
}

pub struct WebmVp8Decoder {
    stream: WebmVp8Stream,
    decoder: Vp8Decoder,
}

impl Default for WebmVp8Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl WebmVp8Decoder {
    pub fn new() -> Self {
        Self { stream: WebmVp8Stream::new(), decoder: Vp8Decoder::new() }
    }
}

impl StreamingVideoDecoder for WebmVp8Decoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<VideoFrame>, MediaDecodeError> {
        let packets = self.stream.push(bytes)?;
        let mut frames = Vec::with_capacity(packets.len());
        for packet in packets {
            let header = FrameHeader::parse(&packet.data)?;
            let decoded = self.decoder.decode(&packet.data)?;
            if header.show_frame {
                frames.push(VideoFrame {
                    width: decoded.width as u32,
                    height: decoded.height as u32,
                    rgba: std::sync::Arc::new(decoded.rgba()),
                    timestamp: packet.timestamp,
                });
            }
        }
        Ok(frames)
    }

    fn metadata(&self) -> Option<MediaMetadata> {
        self.stream.metadata()
    }

    fn finish(&self) -> Result<(), MediaDecodeError> {
        self.stream.finish()
    }
}

impl Default for WebmVp8Stream {
    fn default() -> Self {
        Self::new()
    }
}

impl WebmVp8Stream {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            offset: 0,
            skip: 0,
            parents: Vec::new(),
            doc_type: None,
            track: None,
            video_track: None,
            width: None,
            height: None,
            time_code_scale: 1_000_000,
            duration_ticks: None,
            cluster_timestamp: 0,
        }
    }

    pub fn metadata(&self) -> Option<MediaMetadata> {
        self.video_track?;
        Some(MediaMetadata {
            duration: self
                .duration_ticks
                .map(|ticks| (ticks * self.time_code_scale as f64 / 1e9) as f32),
            width: self.width,
            height: self.height,
            sample_rate: None,
            channels: None,
        })
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<VideoPacket>, MediaDecodeError> {
        self.pending.extend_from_slice(bytes);
        let mut packets = Vec::new();
        let mut cursor = 0usize;
        loop {
            if self.skip != 0 {
                let count = (self.skip as usize).min(self.pending.len() - cursor);
                cursor += count;
                self.skip -= count as u64;
                if self.skip != 0 || cursor == self.pending.len() {
                    break;
                }
            }
            let position = self.offset + cursor as u64;
            while self
                .parents
                .last()
                .and_then(|p| p.end)
                .is_some_and(|end| position >= end)
            {
                self.close_parent();
            }
            let Some((id, id_len)) = element_id(&self.pending[cursor..])? else {
                break;
            };
            let Some((size, size_len)) = element_size(&self.pending[cursor + id_len..])? else {
                break;
            };
            if id == CLUSTER
                && self
                    .parents
                    .last()
                    .is_some_and(|p| p.id == CLUSTER && p.end.is_none())
            {
                self.close_parent();
            }
            if self
                .parents
                .last()
                .is_some_and(|p| p.id == CLUSTER && p.end.is_none())
                && matches!(id, INFO | TRACKS | 0x1c53bb6b | 0x114d9b74)
            {
                self.close_parent();
            }
            let header_len = id_len + size_len;
            let input = &self.pending[cursor..];
            let data_start = position + header_len as u64;
            let end = size.and_then(|len| data_start.checked_add(len));
            if size.is_some() && end.is_none() {
                return Err(invalid("WebM element length overflow"));
            }
            if let (Some(parent_end), Some(end)) = (self.parents.last().and_then(|p| p.end), end) {
                if end > parent_end {
                    return Err(invalid("WebM element exceeds its parent"));
                }
            }
            if is_container(id) {
                if self.parents.len() >= 16 {
                    return Err(invalid("WebM nesting too deep"));
                }
                if id == TRACK_ENTRY {
                    self.track = Some(Track::default());
                }
                if id == CLUSTER {
                    self.cluster_timestamp = 0;
                }
                self.parents.push(Parent { id, end });
                cursor += header_len;
                continue;
            }
            let Some(size) = size else {
                return Err(invalid("unknown-length WebM leaf"));
            };
            let wanted = matches!(
                id,
                DOC_TYPE
                    | TIME_CODE_SCALE
                    | DURATION
                    | TRACK_NUMBER
                    | TRACK_TYPE
                    | CODEC_ID
                    | PIXEL_WIDTH
                    | PIXEL_HEIGHT
                    | TIMESTAMP
                    | SIMPLE_BLOCK
                    | BLOCK
            );
            if !wanted {
                cursor += header_len;
                self.skip = size;
                continue;
            }
            if size > 16 * 1024 * 1024 {
                return Err(invalid("WebM field exceeds 16 MiB"));
            }
            let len = size as usize;
            if input.len() - header_len < len {
                break;
            }
            let payload = &input[header_len..header_len + len];
            match id {
                DOC_TYPE => {
                    if payload != b"webm" {
                        return Err(MediaDecodeError::Unsupported);
                    }
                    self.doc_type = Some(payload.to_vec());
                }
                TIME_CODE_SCALE => self.time_code_scale = unsigned(payload)?,
                DURATION => {
                    self.duration_ticks = Some(match payload.len() {
                        4 => f32::from_be_bytes(payload.try_into().unwrap()) as f64,
                        8 => f64::from_be_bytes(payload.try_into().unwrap()),
                        _ => return Err(invalid("invalid WebM duration")),
                    });
                }
                TRACK_NUMBER => {
                    if let Some(track) = &mut self.track {
                        track.number = unsigned(payload)?;
                    }
                }
                TRACK_TYPE => {
                    if let Some(track) = &mut self.track {
                        track.kind = unsigned(payload)?;
                    }
                }
                CODEC_ID => {
                    if let Some(track) = &mut self.track {
                        track.codec = payload.to_vec();
                    }
                }
                PIXEL_WIDTH => {
                    if let Some(track) = &mut self.track {
                        track.width = Some(
                            unsigned(payload)?
                                .try_into()
                                .map_err(|_| invalid("WebM width overflow"))?,
                        );
                    }
                }
                PIXEL_HEIGHT => {
                    if let Some(track) = &mut self.track {
                        track.height = Some(
                            unsigned(payload)?
                                .try_into()
                                .map_err(|_| invalid("WebM height overflow"))?,
                        );
                    }
                }
                TIMESTAMP => self.cluster_timestamp = unsigned(payload)?,
                SIMPLE_BLOCK | BLOCK => {
                    if self.doc_type.is_none() {
                        return Err(invalid("missing WebM DocType"));
                    }
                    if let Some(packet) = self.block(payload, id == SIMPLE_BLOCK)? {
                        packets.push(packet);
                    }
                }
                _ => {}
            }
            cursor += header_len + len;
        }
        self.pending.drain(..cursor);
        self.offset += cursor as u64;
        Ok(packets)
    }

    pub fn finish(&self) -> Result<(), MediaDecodeError> {
        if self.skip != 0 || !self.pending.is_empty() {
            return Err(invalid("truncated WebM element"));
        }
        if self
            .parents
            .iter()
            .filter_map(|parent| parent.end)
            .any(|end| end > self.offset)
        {
            return Err(invalid("truncated WebM container"));
        }
        if self.doc_type.is_none() || self.video_track.is_none() {
            return Err(MediaDecodeError::Unsupported);
        }
        Ok(())
    }

    fn close_parent(&mut self) {
        if self.parents.pop().is_some_and(|p| p.id == TRACK_ENTRY) {
            if let Some(track) = self.track.take() {
                if track.kind == 1 && track.codec == b"V_VP8" && self.video_track.is_none() {
                    self.video_track = Some(track.number);
                    self.width = track.width;
                    self.height = track.height;
                }
            }
        }
    }

    fn block(&self, payload: &[u8], simple: bool) -> Result<Option<VideoPacket>, MediaDecodeError> {
        let Some((track, len)) = element_size(payload)? else {
            return Err(invalid("truncated WebM block track"));
        };
        let Some(track) = track else {
            return Err(invalid("unknown WebM block track"));
        };
        if payload.len() < len + 3 {
            return Err(invalid("truncated WebM block"));
        }
        if Some(track) != self.video_track {
            return Ok(None);
        }
        let flags = payload[len + 2];
        if flags & 0x06 != 0 {
            return Err(MediaDecodeError::Unsupported);
        }
        let relative = i16::from_be_bytes([payload[len], payload[len + 1]]) as i64;
        let ticks = self.cluster_timestamp as i64 + relative;
        if ticks < 0 {
            return Err(invalid("negative WebM timestamp"));
        }
        Ok(Some(VideoPacket {
            timestamp: (ticks as f64 * self.time_code_scale as f64 / 1e9) as f32,
            key_frame: if simple {
                flags & 0x80 != 0
            } else {
                payload.get(len + 3).is_some_and(|tag| tag & 1 == 0)
            },
            data: payload[len + 3..].to_vec(),
        }))
    }
}

fn is_container(id: u32) -> bool {
    matches!(
        id,
        EBML | SEGMENT | INFO | TRACKS | TRACK_ENTRY | VIDEO | CLUSTER | BLOCK_GROUP
    )
}

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.into())
}

fn unsigned(bytes: &[u8]) -> Result<u64, MediaDecodeError> {
    if bytes.len() > 8 {
        return Err(invalid("oversized WebM integer"));
    }
    Ok(bytes
        .iter()
        .fold(0, |value, byte| (value << 8) | u64::from(*byte)))
}

fn vint_length(first: u8) -> Result<usize, MediaDecodeError> {
    if first == 0 {
        return Err(invalid("invalid EBML variable-length integer"));
    }
    Ok(first.leading_zeros() as usize + 1)
}

fn element_id(bytes: &[u8]) -> Result<Option<(u32, usize)>, MediaDecodeError> {
    let Some(&first) = bytes.first() else {
        return Ok(None);
    };
    let len = vint_length(first)?;
    if len > 4 {
        return Err(invalid("oversized EBML element ID"));
    }
    if bytes.len() < len {
        return Ok(None);
    }
    let id = bytes[..len]
        .iter()
        .fold(0u32, |id, byte| (id << 8) | u32::from(*byte));
    Ok(Some((id, len)))
}

fn element_size(bytes: &[u8]) -> Result<Option<(Option<u64>, usize)>, MediaDecodeError> {
    let Some(&first) = bytes.first() else {
        return Ok(None);
    };
    let len = vint_length(first)?;
    if bytes.len() < len {
        return Ok(None);
    }
    let marker = 0x80 >> (len - 1);
    let mut value = u64::from(first & !marker);
    for &byte in &bytes[1..len] {
        value = (value << 8) | u64::from(byte);
    }
    let unknown = (1u64 << (7 * len)) - 1;
    Ok(Some(((value != unknown).then_some(value), len)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_decoder_emits_frames_from_supplied_webm() {
        let Ok(path) = std::env::var("WEBMEDIA_WEBM_SAMPLE") else { return };
        let bytes = std::fs::read(path).unwrap();
        let mut decoder = WebmVp8Decoder::new();
        let mut frames = Vec::new();
        for chunk in bytes.chunks(1024) {
            frames.extend(decoder.push(chunk).unwrap());
            if frames.len() >= 4 { break; }
        }
        let metadata = decoder.metadata().unwrap();
        assert!(frames.len() >= 4);
        assert!(frames.windows(2).all(|pair| pair[0].timestamp <= pair[1].timestamp));
        assert!(frames.iter().all(|frame| {
            Some(frame.width) == metadata.width
                && Some(frame.height) == metadata.height
                && frame.rgba.len() == frame.width as usize * frame.height as usize * 4
        }));
    }

    #[test]
    fn streams_entire_supplied_webm() {
        let Ok(path) = std::env::var("WEBMEDIA_WEBM_SAMPLE") else { return };
        let bytes = std::fs::read(path).unwrap();
        let mut decoder = WebmVp8Decoder::new();
        let mut count = 0usize;
        let start = std::time::Instant::now();
        for chunk in bytes.chunks(16384) {
            count += decoder.push(chunk).unwrap().len();
        }
        decoder.finish().unwrap();
        assert!(count > 1);
        if std::env::var_os("WEBMEDIA_VP8_REPORT").is_some() {
            eprintln!("streamed {count} VP8 frames in {:.3}s", start.elapsed().as_secs_f64());
        }
    }

    fn element(id: &[u8], value: &[u8]) -> Vec<u8> {
        assert!(value.len() < 127);
        let mut bytes = id.to_vec();
        bytes.push(0x80 | value.len() as u8);
        bytes.extend_from_slice(value);
        bytes
    }

    fn sample() -> Vec<u8> {
        let ebml = element(&[0x1a, 0x45, 0xdf, 0xa3], &element(&[0x42, 0x82], b"webm"));
        let mut track = element(&[0xd7], &[1]);
        track.extend(element(&[0x83], &[1]));
        track.extend(element(&[0x86], b"V_VP8"));
        let mut video = element(&[0xb0], &[0x20]);
        video.extend(element(&[0xba], &[0x18]));
        track.extend(element(&[0xe0], &video));
        let tracks = element(&[0x16, 0x54, 0xae, 0x6b], &element(&[0xae], &track));
        let mut cluster = element(&[0xe7], &[10]);
        cluster.extend(element(
            &[0xa3],
            &[0x81, 0, 2, 0x80, 0x30, 0, 0, 0x9d, 1, 0x2a, 32, 0, 24, 0, 0],
        ));
        let mut segment = tracks;
        segment.extend(element(&[0x1f, 0x43, 0xb6, 0x75], &cluster));
        let mut bytes = ebml;
        bytes.extend(element(&[0x18, 0x53, 0x80, 0x67], &segment));
        bytes
    }

    #[test]
    fn streams_vp8_packet_across_every_chunk_boundary() {
        let sample = sample();
        for chunk_size in 1..=sample.len() {
            let mut stream = WebmVp8Stream::new();
            let packets: Vec<_> = sample
                .chunks(chunk_size)
                .flat_map(|chunk| stream.push(chunk).unwrap())
                .collect();
            stream.finish().unwrap();
            assert_eq!(stream.metadata().unwrap().width, Some(32));
            assert_eq!(stream.metadata().unwrap().height, Some(24));
            assert_eq!(packets.len(), 1);
            assert!((packets[0].timestamp - 0.012).abs() < 0.000001);
            assert!(packets[0].key_frame);
            assert_eq!(
                super::super::vp8::FrameHeader::parse(&packets[0].data)
                    .unwrap()
                    .width,
                Some(32)
            );
        }
    }

    #[test]
    fn rejects_wrong_document_type_and_lacing() {
        let mut bytes = sample();
        let at = bytes.windows(4).position(|w| w == b"webm").unwrap();
        bytes[at..at + 4].copy_from_slice(b"mkv!");
        assert_eq!(
            WebmVp8Stream::new().push(&bytes),
            Err(MediaDecodeError::Unsupported)
        );

        let mut bytes = sample();
        let at = bytes
            .windows(4)
            .position(|w| w == [0x81, 0, 2, 0x80])
            .unwrap();
        bytes[at + 3] = 0x82;
        assert_eq!(
            WebmVp8Stream::new().push(&bytes),
            Err(MediaDecodeError::Unsupported)
        );
    }

    #[test]
    fn demuxes_real_webm_fixture_when_supplied() {
        let Ok(path) = std::env::var("WEBMEDIA_WEBM_SAMPLE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let mut stream = WebmVp8Stream::new();
        let mut packets = Vec::new();
        for chunk in bytes.chunks(23) {
            packets.extend(stream.push(chunk).unwrap());
        }
        stream.finish().unwrap();
        assert!(!packets.is_empty());
        assert!(packets[0].key_frame);
        let header = super::super::vp8::FrameHeader::parse(&packets[0].data).unwrap();
        assert_eq!(
            stream.metadata().unwrap().width,
            header.width.map(u32::from)
        );
        assert_eq!(
            stream.metadata().unwrap().height,
            header.height.map(u32::from)
        );
        let mut control =
            super::super::vp8::BoolDecoder::new(header.control_partition(&packets[0].data))
                .unwrap();
        assert!(!control.read_bit().unwrap()); // YUV color space
        assert!(!control.read_bit().unwrap()); // no pixel clamping
        let mut layout = super::super::vp8::KeyFrameLayout::parse(&packets[0].data).unwrap();
        assert!(!layout.token_partitions.is_empty());
        assert!(
            layout
                .token_partitions
                .iter()
                .all(|partition| !partition.is_empty())
        );
        let mut modes = Vec::new();
        while let Some(mode) = layout.next_macroblock_mode().unwrap() {
            assert!(mode.luma <= 4 && mode.chroma <= 3);
            assert!(mode.subblocks.iter().all(|subblock| *subblock <= 9));
            modes.push(mode);
        }
        let meta = stream.metadata().unwrap();
        let expected = meta.width.unwrap().div_ceil(16) * meta.height.unwrap().div_ceil(16);
        assert_eq!(modes.len(), expected as usize);
        let mut residue = super::super::vp8_residue::ResidueDecoder::new(&layout).unwrap();
        let mut nonzero_pixels = 0;
        for (index, mode) in modes.iter().enumerate() {
            let block = residue
                .decode(
                    &layout,
                    mode,
                    index % meta.width.unwrap().div_ceil(16) as usize,
                    index / meta.width.unwrap().div_ceil(16) as usize,
                )
                .unwrap_or_else(|error| panic!("VP8 residue macroblock {index}: {error:?}"));
            nonzero_pixels += block
                .y
                .iter()
                .flatten()
                .filter(|value| **value != 0)
                .count();
        }
        assert!(nonzero_pixels > 0);
        if let Ok(path) = std::env::var("WEBMEDIA_VP8_REFERENCE") {
            let reference = std::fs::read(path).unwrap();
            let decoded = super::super::vp8_keyframe::decode_keyframe(&packets[0].data).unwrap();
            let len = decoded.width * decoded.height;
            assert!(reference.len() >= len);
            let error: usize = (0..decoded.height)
                .flat_map(|row| (0..decoded.width).map(move |col| (row, col)))
                .map(|(row, col)| {
                    decoded.y.pixels[row * decoded.y.width + col]
                        .abs_diff(reference[row * decoded.width + col]) as usize
                })
                .sum();
            assert!(
                error < len * 2,
                "VP8 keyframe luma MAE = {:.3}",
                error as f64 / len as f64
            );
            let chroma_width = decoded.width.div_ceil(2);
            let chroma_height = decoded.height.div_ceil(2);
            let chroma_len = chroma_width * chroma_height;
            assert!(reference.len() >= len + 2 * chroma_len);
            for (plane, offset) in [(&decoded.u, len), (&decoded.v, len + chroma_len)] {
                let chroma_error: usize = (0..chroma_height)
                    .flat_map(|row| (0..chroma_width).map(move |col| (row, col)))
                    .map(|(row, col)| {
                        plane.pixels[row * plane.width + col]
                            .abs_diff(reference[offset + row * chroma_width + col])
                            as usize
                    })
                    .sum();
                assert!(
                    chroma_error < chroma_len * 2,
                    "VP8 keyframe chroma MAE = {:.3}",
                    chroma_error as f64 / chroma_len as f64
                );
            }
            assert_eq!(decoded.rgba().len(), len * 4);
        }
        assert!(
            packets
                .windows(2)
                .all(|pair| pair[0].timestamp <= pair[1].timestamp)
        );
    }
}
