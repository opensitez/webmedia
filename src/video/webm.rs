//! Incremental WebM demuxing for VP8 and VP9 video tracks.

use super::backend::{MediaDecodeError, MediaMetadata, StreamingVideoDecoder, VideoFrame};
use super::vp8::FrameHeader;
use super::vp8_decoder::Vp8Decoder;
use super::vp9::split_superframe;
use super::vp9_decoder::Vp9Decoder;

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

pub struct WebmVideoStream {
    pending: Vec<u8>,
    offset: u64,
    skip: u64,
    parents: Vec<Parent>,
    doc_type: Option<Vec<u8>>,
    track: Option<Track>,
    video_track: Option<u64>,
    requested_codec: Option<WebmVideoCodec>,
    video_codec: Option<WebmVideoCodec>,
    width: Option<u32>,
    height: Option<u32>,
    time_code_scale: u64,
    duration_ticks: Option<f64>,
    cluster_timestamp: u64,
}

pub struct WebmVp8Decoder {
    stream: WebmVideoStream,
    decoder: Vp8Decoder,
}

pub struct WebmVideoDecoder {
    stream: WebmVideoStream,
    codec: Option<CodecDecoder>,
}

enum CodecDecoder {
    Vp8(Vp8Decoder),
    Vp9(Vp9Decoder),
}

pub type WebmVp8Stream = WebmVideoStream;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WebmVideoCodec {
    Vp8,
    Vp9,
}

impl WebmVideoCodec {
    fn id(self) -> &'static [u8] {
        match self {
            Self::Vp8 => b"V_VP8",
            Self::Vp9 => b"V_VP9",
        }
    }
}

impl Default for WebmVp8Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for WebmVideoDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl WebmVideoDecoder {
    pub fn new() -> Self {
        Self { stream: WebmVideoStream::new(), codec: None }
    }
}

impl StreamingVideoDecoder for WebmVideoDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<VideoFrame>, MediaDecodeError> {
        let packets = self.stream.push(bytes)?;
        let mut frames = Vec::new();
        for packet in packets {
            if self.codec.is_none() {
                self.codec = Some(match self.stream.video_codec().ok_or(MediaDecodeError::Unsupported)? {
                    WebmVideoCodec::Vp8 => CodecDecoder::Vp8(Vp8Decoder::new()),
                    WebmVideoCodec::Vp9 => CodecDecoder::Vp9(Vp9Decoder::new()),
                });
            }
            let codec = self.codec.as_mut().ok_or(MediaDecodeError::Unsupported)?;
            match codec {
                CodecDecoder::Vp8(decoder) => {
                    let header = FrameHeader::parse(&packet.data)?;
                    let decoded = decoder.decode(&packet.data)?;
                    if header.show_frame {
                        frames.push(VideoFrame {
                            width: decoded.width as u32,
                            height: decoded.height as u32,
                            rgba: std::sync::Arc::new(decoded.rgba()),
                            timestamp: packet.timestamp,
                        });
                    }
                }
                CodecDecoder::Vp9(decoder) => {
                    for data in split_superframe(&packet.data)? {
                        if let Some(decoded) = decoder.decode(data)? {
                            frames.push(VideoFrame {
                                width: decoded.width as u32,
                                height: decoded.height as u32,
                                rgba: std::sync::Arc::new(decoded.rgba()),
                                timestamp: packet.timestamp,
                            });
                        }
                    }
                }
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

impl WebmVp8Decoder {
    pub fn new() -> Self {
        Self { stream: WebmVideoStream::for_codec(WebmVideoCodec::Vp8), decoder: Vp8Decoder::new() }
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

impl Default for WebmVideoStream {
    fn default() -> Self {
        Self::new()
    }
}

impl WebmVideoStream {
    pub fn new() -> Self {
        Self::with_codec(None)
    }

    pub fn for_codec(video_codec: WebmVideoCodec) -> Self {
        Self::with_codec(Some(video_codec))
    }

    fn with_codec(requested_codec: Option<WebmVideoCodec>) -> Self {
        Self {
            pending: Vec::new(),
            offset: 0,
            skip: 0,
            parents: Vec::new(),
            doc_type: None,
            track: None,
            video_track: None,
            requested_codec,
            video_codec: None,
            width: None,
            height: None,
            time_code_scale: 1_000_000,
            duration_ticks: None,
            cluster_timestamp: 0,
        }
    }

    pub fn video_codec(&self) -> Option<WebmVideoCodec> {
        self.video_codec
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
                let codec = [WebmVideoCodec::Vp8, WebmVideoCodec::Vp9]
                    .into_iter().find(|codec| track.codec == codec.id());
                if track.kind == 1 && self.video_track.is_none()
                    && codec.is_some_and(|codec| self.requested_codec.is_none_or(|requested| requested == codec)) {
                    self.video_track = Some(track.number);
                    self.video_codec = codec;
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
    fn reconstructs_first_vp9_transform_against_reference() {
        let (Ok(sample), Ok(reference)) = (
            std::env::var("WEBMEDIA_VP9_SAMPLE"),
            std::env::var("WEBMEDIA_VP9_REFERENCE"),
        ) else { return };
        let bytes = std::fs::read(sample).unwrap();
        let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
        let packet = stream.push(&bytes).unwrap().into_iter().next().unwrap();
        let frames = super::super::vp9::split_superframe(&packet.data).unwrap();
        let layout = super::super::vp9::KeyframeLayout::parse(frames[0]).unwrap();
        let compressed = super::super::vp9_compressed::CompressedHeader::parse_keyframe(&layout).unwrap();
        let tiles = layout.tile_partitions().unwrap();
        let (blocks, next_partition) = super::super::vp9_tile::first_tile_row_prefix(
            &layout, &compressed, tiles[0], 7,
        ).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(next_partition, Some(super::super::vp9_tile::Partition::Split));
        let leaf = super::super::vp9_tile::first_split_top_left_16x16(
            &layout, &compressed, tiles[0], 7,
        ).unwrap().unwrap();
        let generic = super::super::vp9_tile::decode_keyframe_tile_prefix(
            &layout, &compressed, tiles[0], 0, 4,
        ).unwrap();
        assert_eq!(generic.len(), 4);
        for (index, expected) in blocks.iter().enumerate() {
            assert_eq!((generic[index].x, generic[index].y), (index * 64, 0));
            assert_eq!(&generic[index].block, expected);
        }
        assert_eq!((generic[3].x, generic[3].y), (192, 0));
        assert_eq!(generic[3].block, leaf);
        let first_eight = super::super::vp9_tile::decode_keyframe_tile_prefix(
            &layout, &compressed, tiles[0], 0, 5,
        ).unwrap();
        assert_eq!((first_eight[4].x, first_eight[4].y, first_eight[4].block.y_mode), (208, 0, 0));
        let mixed_blocks = super::super::vp9_tile::decode_keyframe_tile_prefix(
            &layout, &compressed, tiles[0], 0, 54,
        ).unwrap();
        assert_eq!(mixed_blocks.len(), 54);
        if std::env::var_os("WEBMEDIA_VP9_PROBE").is_some() {
            eprintln!("VP9 first 54 blocks: {:?}", mixed_blocks.iter()
                .map(|entry| (entry.x, entry.y, entry.width, entry.height, entry.block.y_mode))
                .collect::<Vec<_>>());
        }
        if std::env::var_os("WEBMEDIA_VP9_PROBE").is_some() {
            let decode = |limit| super::super::vp9_tile::decode_keyframe_tile_prefix(
                &layout, &compressed, tiles[0], 0, limit,
            );
            let mut good = 4;
            let mut bad = 8;
            while decode(bad).is_ok() && bad < 16384 {
                good = bad;
                bad *= 2;
            }
            while good + 1 < bad {
                let middle = good + (bad - good) / 2;
                if decode(middle).is_ok() { good = middle; } else { bad = middle; }
            }
            let decoded = decode(good).unwrap();
            eprintln!("VP9 tile prefix: {good} blocks decoded; last={:?}; next limit {bad}: {:?}",
                decoded.last().map(|entry| (entry.x, entry.y, entry.width, entry.height, entry.block.y_mode)),
                decode(bad).err());
        }
        let mut plane = super::super::vp8_predict::Plane::new(216, 64);
        for (block_index, block) in blocks.iter().enumerate() {
            let q = match layout.segment_alt_q[block.segment_id as usize] {
                Some(alt) if layout.segmentation_abs_or_delta_update => i32::from(alt),
                Some(alt) => i32::from(layout.base_q_idx) + i32::from(alt),
                None => i32::from(layout.base_q_idx),
            };
            for (index, coefficients) in block.luma_coefficients.as_ref().unwrap().iter().enumerate() {
                let x = block_index * 64 + index % 2 * 32;
                let y = index / 2 * 32;
                super::super::vp9_transform::reconstruct_32x32_intra(
                    &mut plane, x, y, block.y_mode, x != 0, y != 0, false, 216, 64,
                    coefficients, layout.header.bit_depth.unwrap(),
                    q + i32::from(layout.delta_q_y_dc), q,
                ).unwrap();
            }
        }
        let leaf_q = match layout.segment_alt_q[leaf.segment_id as usize] {
            Some(alt) if layout.segmentation_abs_or_delta_update => i32::from(alt),
            Some(alt) => i32::from(layout.base_q_idx) + i32::from(alt),
            None => i32::from(layout.base_q_idx),
        };
        super::super::vp9_transform::reconstruct_16x16_intra(
            &mut plane, 192, 0, leaf.y_mode, true, false, 216, 64,
            &leaf.luma_coefficients.as_ref().unwrap()[0], layout.header.bit_depth.unwrap(),
            leaf_q + i32::from(layout.delta_q_y_dc), leaf_q,
        ).unwrap();
        let eight = &first_eight[4].block;
        let eight_q = match layout.segment_alt_q[eight.segment_id as usize] {
            Some(alt) if layout.segmentation_abs_or_delta_update => i32::from(alt),
            Some(alt) => i32::from(layout.base_q_idx) + i32::from(alt),
            None => i32::from(layout.base_q_idx),
        };
        super::super::vp9_transform::reconstruct_8x8_intra(
            &mut plane, 208, 0, eight.y_mode, true, false, 216, 64,
            &eight.luma_coefficients.as_ref().unwrap()[0], layout.header.bit_depth.unwrap(),
            eight_q + i32::from(layout.delta_q_y_dc), eight_q,
        ).unwrap();
        let reference = std::fs::read(reference).unwrap();
        let width = layout.header.width.unwrap() as usize;
        if std::env::var_os("WEBMEDIA_VP9_RECON_PROBE").is_some() {
            let height = layout.header.height.unwrap() as usize;
            let decoded = super::super::vp9_tile::decode_keyframe_tile_prefix(
                &layout, &compressed, tiles[0], 0, 54,
            ).unwrap();
            let mut reconstructed = super::super::vp8_predict::Plane::new(width, height);
            let mut suspicious = 0;
            for (index, entry) in decoded.iter().enumerate() {
                let block = &entry.block;
                let transform_size = 4usize << block.tx_size;
                let transforms_wide = entry.width.max(8) / transform_size;
                let q = match layout.segment_alt_q[block.segment_id as usize] {
                    Some(alt) if layout.segmentation_abs_or_delta_update => i32::from(alt),
                    Some(alt) => i32::from(layout.base_q_idx) + i32::from(alt),
                    None => i32::from(layout.base_q_idx),
                };
                for (transform_index, coefficients) in block.luma_coefficients.as_ref().unwrap().iter().enumerate() {
                    let block_x = transform_index % transforms_wide;
                    let block_y = transform_index / transforms_wide;
                    let x = entry.x + block_x * transform_size;
                    let y = entry.y + block_y * transform_size;
                    let mode = block.sub_modes.map_or(block.y_mode, |modes| modes[block_y * 2 + block_x]);
                    super::super::vp9_transform::reconstruct_intra(
                        &mut reconstructed, x, y, transform_size, mode, false,
                        x != 0, y != 0, block_x + 1 < transforms_wide, width, height,
                        coefficients, 8, q + i32::from(layout.delta_q_y_dc), q,
                        layout.lossless,
                    ).unwrap_or_else(|error| panic!("VP9 block {index} transform {transform_index} at ({x},{y}) size={transform_size} coeffs={} mode={mode}: {error:?}", coefficients.len()));
                }
                let mut large = 0;
                let mut total = 0;
                let mut max_diff = 0;
                for y in entry.y + 1..entry.y + entry.height.max(8) - 1 {
                    for x in entry.x + 1..entry.x + entry.width.max(8) - 1 {
                        let p = y * width + x;
                        let difference = reconstructed.pixels[p].abs_diff(reference[p]);
                        large += usize::from(difference > 4);
                        max_diff = max_diff.max(difference);
                        total += 1;
                    }
                }
                if large * 4 > total {
                    suspicious += 1;
                    if suspicious <= 20 {
                        eprintln!("VP9 contiguous block {index} at ({}, {}) {}x{} y_mode={} sub={:?} tx={} large={large}/{total} max={max_diff}",
                            entry.x, entry.y, entry.width, entry.height, block.y_mode, block.sub_modes, block.tx_size);
                    }
                }
            }
            eprintln!("VP9 contiguous reconstruction: {} suspicious of {} blocks", suspicious, decoded.len());
        }
        if std::env::var_os("WEBMEDIA_VP9_CHROMA_PROBE").is_some() {
            let height = layout.header.height.unwrap() as usize;
            let chroma_width = width / 2;
            let chroma_height = height / 2;
            let decoded = super::super::vp9_tile::decode_keyframe_tile_prefix(
                &layout, &compressed, tiles[0], 0, 96,
            ).unwrap();
            for plane_index in 0..2 {
                let offset = width * height + plane_index * chroma_width * chroma_height;
                let expected = &reference[offset..offset + chroma_width * chroma_height];
                let mut oracle = super::super::vp8_predict::Plane::new(chroma_width, chroma_height);
                oracle.pixels.copy_from_slice(expected);
                for (index, entry) in decoded.iter().enumerate() {
                    let x0 = entry.x / 2;
                    let y0 = entry.y / 2;
                    let block_width = entry.width.max(8) / 2;
                    let block_height = entry.height.max(8) / 2;
                    let tx_size = entry.block.tx_size.min((block_width.min(block_height) / 4).trailing_zeros() as u8);
                    let transform_size = 4usize << tx_size;
                    let transforms_wide = block_width / transform_size;
                    let coefficients = &entry.block.chroma_coefficients.as_ref().unwrap()[plane_index];
                    let q = match layout.segment_alt_q[entry.block.segment_id as usize] {
                        Some(alt) if layout.segmentation_abs_or_delta_update => i32::from(alt),
                        Some(alt) => i32::from(layout.base_q_idx) + i32::from(alt),
                        None => i32::from(layout.base_q_idx),
                    };
                    for (transform_index, transform) in coefficients.chunks_exact(transform_size * transform_size).enumerate() {
                        let x = x0 + transform_index % transforms_wide * transform_size;
                        let y = y0 + transform_index / transforms_wide * transform_size;
                        let result = match transform_size {
                            32 => super::super::vp9_transform::reconstruct_32x32_intra(
                                &mut oracle, x, y, entry.block.uv_mode, x != 0, y != 0, false,
                                chroma_width, chroma_height, transform, 8,
                                q + i32::from(layout.delta_q_uv_dc), q + i32::from(layout.delta_q_uv_ac)),
                            16 => super::super::vp9_transform::reconstruct_16x16_intra(
                                &mut oracle, x, y, entry.block.uv_mode, x != 0, y != 0,
                                chroma_width, chroma_height, transform, 8,
                                q + i32::from(layout.delta_q_uv_dc), q + i32::from(layout.delta_q_uv_ac)),
                            8 => super::super::vp9_transform::reconstruct_8x8_intra(
                                &mut oracle, x, y, entry.block.uv_mode, x != 0, y != 0,
                                chroma_width, chroma_height, transform, 8,
                                q + i32::from(layout.delta_q_uv_dc), q + i32::from(layout.delta_q_uv_ac)),
                            4 => super::super::vp9_transform::reconstruct_4x4_intra(
                                &mut oracle, x, y, entry.block.uv_mode, x != 0, y != 0,
                                chroma_width, chroma_height, transform, 8,
                                q + i32::from(layout.delta_q_uv_dc), q + i32::from(layout.delta_q_uv_ac)),
                            _ => unreachable!(),
                        };
                        result.unwrap();
                    }
                    let mut large = 0;
                    let mut total = 0;
                    for y in y0 + 1..y0 + block_height - 1 {
                        for x in x0 + 1..x0 + block_width - 1 {
                            let p = y * chroma_width + x;
                            large += usize::from(oracle.pixels[p].abs_diff(expected[p]) > 8);
                            total += 1;
                        }
                    }
                    if large * 4 > total {
                        eprintln!("VP9 chroma {plane_index} block {index} at ({}, {}) {}x{}: {large}/{total} pixels differ by >8, mode={}",
                            entry.x, entry.y, entry.width, entry.height, entry.block.uv_mode);
                    }
                    for y in y0..y0 + block_height {
                        let start = y * chroma_width + x0;
                        oracle.pixels[start..start + block_width]
                            .copy_from_slice(&expected[start..start + block_width]);
                    }
                }
            }
        }
        if std::env::var_os("WEBMEDIA_VP9_PIXEL_PROBE").is_some() {
            eprintln!("VP9 sample layout: segmentation={} update_map={} tile_grid={}x{} tile_bytes={:?}",
                layout.segmentation_enabled, layout.segmentation_update_map,
                1usize << layout.tile_cols_log2, 1usize << layout.tile_rows_log2,
                tiles.iter().map(|tile| tile.len()).collect::<Vec<_>>());
            let height = layout.header.height.unwrap() as usize;
            let decoded = super::super::vp9_tile::decode_keyframe_tile_prefix(
                &layout, &compressed, tiles[0], 0, 1200,
            ).unwrap();
            let mut oracle = super::super::vp8_predict::Plane::new(width, height);
            oracle.pixels.copy_from_slice(&reference[..width * height]);
            let mut suspicious = 0;
            let mut checked = 0;
            for (index, entry) in decoded.iter().enumerate() {
                let block = &entry.block;
                if block.y_mode != 0 || block.sub_modes.is_some() || block.tx_size == 0
                    || entry.x + entry.width > width || entry.y + entry.height > height {
                    continue;
                }
                let transform_size = 4usize << block.tx_size;
                let transforms_wide = entry.width / transform_size;
                let coefficients = block.luma_coefficients.as_ref().unwrap();
                if coefficients.len() != transforms_wide * (entry.height / transform_size) {
                    continue;
                }
                let q = match layout.segment_alt_q[block.segment_id as usize] {
                    Some(alt) if layout.segmentation_abs_or_delta_update => i32::from(alt),
                    Some(alt) => i32::from(layout.base_q_idx) + i32::from(alt),
                    None => i32::from(layout.base_q_idx),
                };
                for (transform_index, transform) in coefficients.iter().enumerate() {
                    let x = entry.x + transform_index % transforms_wide * transform_size;
                    let y = entry.y + transform_index / transforms_wide * transform_size;
                    let result = match transform_size {
                        32 => super::super::vp9_transform::reconstruct_32x32_intra(
                            &mut oracle, x, y, 0, x != 0, y != 0, false, width, height,
                            transform, 8, q + i32::from(layout.delta_q_y_dc), q),
                        16 => super::super::vp9_transform::reconstruct_16x16_intra(
                            &mut oracle, x, y, 0, x != 0, y != 0, width, height,
                            transform, 8, q + i32::from(layout.delta_q_y_dc), q),
                        8 => super::super::vp9_transform::reconstruct_8x8_intra(
                            &mut oracle, x, y, 0, x != 0, y != 0, width, height,
                            transform, 8, q + i32::from(layout.delta_q_y_dc), q),
                        _ => continue,
                    };
                    result.unwrap();
                }
                let mut large = 0;
                let mut total = 0;
                for y in entry.y + 2..entry.y + entry.height - 2 {
                    for x in entry.x + 2..entry.x + entry.width - 2 {
                        let position = y * width + x;
                        let difference = oracle.pixels[position].abs_diff(reference[position]);
                        large += usize::from(difference > 4);
                        total += 1;
                    }
                }
                checked += 1;
                if large * 10 > total {
                    suspicious += 1;
                    if suspicious <= 12 {
                        eprintln!("VP9 pixel probe block {index} at ({}, {}) {}x{}: {large}/{total} pixels differ by >4",
                            entry.x, entry.y, entry.width, entry.height);
                        if index == 26 || index == 87 {
                            eprintln!("  q={q} tx={} skip={} first={:?} eob={} coeffs={:?}",
                                block.tx_size, block.skip, block.first_luma_coefficient,
                                block.first_luma_eob,
                                coefficients.iter().map(|values| values.iter().filter(|&&v| v != 0).count()).collect::<Vec<_>>());
                            for sample_y in [4, 8, 12, 16, 20, 24] {
                                let y = entry.y + sample_y.min(entry.height - 1);
                                let row = [4, 8, 12, 16, 20, 24].map(|sample_x| {
                                    let x = entry.x + sample_x.min(entry.width - 1);
                                    let p = y * width + x;
                                    (oracle.pixels[p], reference[p])
                                });
                                eprintln!("  y={sample_y} {row:?}");
                            }
                        }
                    }
                }
                for y in entry.y..entry.y + entry.height {
                    let start = y * width + entry.x;
                    oracle.pixels[start..start + entry.width]
                        .copy_from_slice(&reference[start..start + entry.width]);
                }
            }
            eprintln!("VP9 pixel probe checked {checked} DC blocks, {suspicious} suspicious");
        }
        let mut matches = 0;
        let mut mismatches = Vec::new();
        for y in 2..62 {
            for x in 2..190 {
                let actual = plane.pixels[y * 216 + x];
                let expected = reference[y * width + x];
                if actual == expected {
                    matches += 1;
                } else if mismatches.len() < 8 {
                    mismatches.push((x, y, actual, expected));
                }
            }
        }
        eprintln!("VP9 tile prefix luma: {matches}/11280 matching interior pixels; mismatches={mismatches:?}");
        assert!(matches > 10000, "tile prefix differs from reference frame");
        let leaf_y_matches = (1..15).flat_map(|y| (193..207).map(move |x| (x, y)))
            .filter(|&(x, y)| plane.pixels[y * 216 + x] == reference[y * width + x]).count();
        eprintln!("VP9 split leaf luma: {leaf_y_matches}/196 matching interior pixels");
        assert!(leaf_y_matches > 160, "split leaf luma differs from reference frame");
        let eight_matches = (1..7).flat_map(|y| (209..215).map(move |x| (x, y)))
            .filter(|&(x, y)| plane.pixels[y * 216 + x] == reference[y * width + x]).count();
        eprintln!("VP9 first 8x8 luma: {eight_matches}/36 matching interior pixels");
        assert!(eight_matches > 25, "first 8x8 block differs from reference frame");
        let chroma_width = width / 2;
        let chroma_plane_size = chroma_width * (layout.header.height.unwrap() as usize / 2);
        for plane_index in 0..2 {
            let mut chroma = super::super::vp8_predict::Plane::new(104, 32);
            for (block_index, block) in blocks.iter().enumerate() {
                let q = match layout.segment_alt_q[block.segment_id as usize] {
                    Some(alt) if layout.segmentation_abs_or_delta_update => i32::from(alt),
                    Some(alt) => i32::from(layout.base_q_idx) + i32::from(alt),
                    None => i32::from(layout.base_q_idx),
                };
                super::super::vp9_transform::reconstruct_32x32_intra(
                    &mut chroma, block_index * 32, 0, block.uv_mode,
                    block_index != 0, false, false, 104, 32,
                    &block.chroma_coefficients.as_ref().unwrap()[plane_index],
                    layout.header.bit_depth.unwrap(),
                    q + i32::from(layout.delta_q_uv_dc),
                    q + i32::from(layout.delta_q_uv_ac),
                ).unwrap();
            }
            super::super::vp9_transform::reconstruct_8x8_intra(
                &mut chroma, 96, 0, leaf.uv_mode, true, false, 104, 32,
                &leaf.chroma_coefficients.as_ref().unwrap()[plane_index],
                layout.header.bit_depth.unwrap(),
                leaf_q + i32::from(layout.delta_q_uv_dc),
                leaf_q + i32::from(layout.delta_q_uv_ac),
            ).unwrap();
            let offset = width * layout.header.height.unwrap() as usize + plane_index * chroma_plane_size;
            let mut chroma_matches = 0;
            let mut chroma_mismatches = Vec::new();
            for y in 2..30 {
                for x in 2..94 {
                    let actual = chroma.pixels[y * 104 + x];
                    let expected = reference[offset + y * chroma_width + x];
                    if actual == expected {
                        chroma_matches += 1;
                    } else if chroma_mismatches.len() < 8 {
                        chroma_mismatches.push((x, y, actual, expected));
                    }
                }
            }
            eprintln!("VP9 tile prefix chroma {plane_index}: {chroma_matches}/2576 matching interior pixels; mismatches={chroma_mismatches:?}");
            assert!(chroma_matches > 2300, "chroma plane differs from reference frame");
            let leaf_chroma_matches = (1..7).flat_map(|y| (97..103).map(move |x| (x, y)))
                .filter(|&(x, y)| chroma.pixels[y * 104 + x] == reference[offset + y * chroma_width + x]).count();
            eprintln!("VP9 split leaf chroma {plane_index}: {leaf_chroma_matches}/36 matching interior pixels");
            assert!(leaf_chroma_matches > 25, "split leaf chroma differs from reference frame");
        }
    }

    #[test]
    fn demuxes_vp9_track_and_parses_real_keyframe() {
        let Ok(path) = std::env::var("WEBMEDIA_VP9_SAMPLE") else { return };
        let mut source = std::fs::File::open(path).unwrap();
        let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
        let mut buf = [0u8; 16 * 1024];
        let packet = loop {
            let count = std::io::Read::read(&mut source, &mut buf).unwrap();
            assert!(count != 0, "VP9 sample contains no video packets");
            if let Some(packet) = stream.push(&buf[..count]).unwrap().into_iter().next() {
                break packet;
            }
        };
        let frames = super::super::vp9::split_superframe(&packet.data).unwrap();
        let header = super::super::vp9::FrameHeader::parse(frames[0]).unwrap();
        let layout = super::super::vp9::KeyframeLayout::parse(frames[0]).unwrap();
        let metadata = stream.metadata().unwrap();
        assert!(packet.key_frame && header.key_frame);
        assert_eq!(header.width, metadata.width);
        assert_eq!(header.height, metadata.height);
        assert!(!layout.compressed_header.is_empty());
        assert!(!layout.tiles.is_empty());
        let tiles = layout.tile_partitions().unwrap();
        assert!(!tiles.is_empty());
        if let Ok(expected) = std::env::var("WEBMEDIA_VP9_EXPECT_TILES") {
            assert_eq!(tiles.len(), expected.parse::<usize>().unwrap());
        }
        let mut arithmetic = super::super::vp8::BoolDecoder::new(layout.compressed_header).unwrap();
        assert!(!arithmetic.read_bit().unwrap(), "invalid VP9 arithmetic marker");
        let compressed = super::super::vp9_compressed::CompressedHeader::parse_keyframe(&layout).unwrap();
        assert!(compressed.tx_mode <= 4);
        for tile in &tiles {
            let mut arithmetic = super::super::vp8::BoolDecoder::new(tile).unwrap();
            assert!(!arithmetic.read_bit().unwrap(), "invalid VP9 tile arithmetic marker");
        }
        let first_partition = super::super::vp9_tile::first_partition(tiles[0]).unwrap();
        let first_block = super::super::vp9_tile::first_block(&layout, &compressed, tiles[0]).unwrap();
        let (row_prefix, next_partition) = super::super::vp9_tile::first_tile_row_prefix(
            &layout, &compressed, tiles[0], 7,
        ).unwrap();
        let split_child = super::super::vp9_tile::first_split_top_left_path(
            &layout, &compressed, tiles[0], 7,
        ).unwrap();
        let split_leaf = super::super::vp9_tile::first_split_top_left_16x16(
            &layout, &compressed, tiles[0], 7,
        ).unwrap();
        if let Some(ref leaf) = split_leaf {
            assert_eq!(leaf.block_size, 16);
            assert_eq!(leaf.luma_coefficients.as_ref().unwrap()[0].len(), 256);
            assert!(leaf.chroma_coefficients.as_ref().unwrap().iter().all(|plane| plane.len() == 64));
        }
        assert_eq!(row_prefix.first().map(|block| block.first_luma_coefficient),
            first_block.as_ref().map(|block| block.first_luma_coefficient));
        if let Some(ref block) = first_block {
            assert_eq!(block.partition, first_partition);
            assert!(block.segment_id < 8 && block.tx_size < 4);
            assert!(block.y_mode < 10 && block.uv_mode < 10);
            assert!(block.first_luma_token.is_none_or(|token| token <= 10));
            assert_eq!(block.first_luma_token.is_some(), block.first_luma_coefficient.is_some());
            if let Some(coefficients) = &block.luma_coefficients {
                assert_eq!(coefficients.len(), 4);
                assert!(coefficients.iter().all(|transform| transform.len() == 1024));
                assert_eq!(coefficients[0][0], block.first_luma_coefficient.unwrap_or(0));
                assert!(block.first_luma_eob <= 1024);
            }
            if let Some(planes) = &block.chroma_coefficients {
                assert!(planes.iter().all(|coefficients| coefficients.len() == 1024));
            }
        }
        if std::env::var_os("WEBMEDIA_VP9_REPORT").is_some() {
            eprintln!("VP9 keyframe: {}x{}, q={}, lossless={}, tiles={}, tx_mode={}, prob_updates={}, skip_probs={:?}, first_partition={first_partition:?}, row_prefix={}, next_partition={next_partition:?}, split_child={split_child:?}, split_leaf={:?}, first_block={:?}",
                header.width.unwrap(), header.height.unwrap(), layout.base_q_idx, layout.lossless,
                tiles.len(), compressed.tx_mode, compressed.probability_updates, compressed.skip_probs, row_prefix.len(),
                split_leaf.as_ref().map(|leaf| (leaf.y_mode, leaf.uv_mode, leaf.first_luma_coefficient)),
                first_block.as_ref().map(|block| (block.segment_id, block.skip, block.tx_size,
                    block.y_mode, block.uv_mode, block.first_luma_token,
                    block.first_luma_coefficient, block.first_luma_eob)));
        }
    }

    #[test]
    fn scans_every_vp9_packet_in_supplied_webm() {
        let Ok(path) = std::env::var("WEBMEDIA_VP9_SAMPLE") else { return };
        let mut source = std::fs::File::open(path).unwrap();
        let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
        let mut buf = [0u8; 16 * 1024];
        let mut packets = 0usize;
        let mut frames = 0usize;
        loop {
            let count = std::io::Read::read(&mut source, &mut buf).unwrap();
            if count == 0 {
                break;
            }
            for packet in stream.push(&buf[..count]).unwrap() {
                packets += 1;
                for frame in super::super::vp9::split_superframe(&packet.data).unwrap() {
                    super::super::vp9::FrameHeader::parse(frame).unwrap();
                    frames += 1;
                }
            }
        }
        stream.finish().unwrap();
        assert!(packets > 1 && frames >= packets);
    }

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
    fn auto_decoder_plays_vp8_and_vp9_webm() {
        for variable in ["WEBMEDIA_WEBM_SAMPLE", "WEBMEDIA_VP9_SAMPLE"] {
            let Ok(path) = std::env::var(variable) else { continue };
            let mut source = std::fs::File::open(path).unwrap();
            let mut decoder = WebmVideoDecoder::new();
            let mut buffer = [0u8; 16 * 1024];
            let mut frames = Vec::new();
            while frames.len() < 3 {
                let count = std::io::Read::read(&mut source, &mut buffer).unwrap();
                assert!(count != 0, "{variable} ended before three frames");
                frames.extend(decoder.push(&buffer[..count]).unwrap());
            }
            assert!(frames.len() >= 3);
            assert!(frames.windows(2).all(|pair| pair[0].timestamp <= pair[1].timestamp));
            assert!(frames.iter().all(|frame| frame.rgba.len()
                == frame.width as usize * frame.height as usize * 4));
            assert_eq!(decoder.metadata().unwrap().width, Some(frames[0].width));
        }
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
    fn selects_vp8_or_vp9_from_webm_track() {
        for codec in [WebmVideoCodec::Vp8, WebmVideoCodec::Vp9] {
            let mut bytes = sample();
            let offset = bytes.windows(5).position(|window| window == b"V_VP8").unwrap();
            bytes[offset..offset + 5].copy_from_slice(codec.id());
            let mut stream = WebmVideoStream::new();
            assert_eq!(stream.video_codec(), None);
            assert_eq!(stream.push(&bytes).unwrap().len(), 1);
            assert_eq!(stream.video_codec(), Some(codec));
            stream.finish().unwrap();
        }
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
