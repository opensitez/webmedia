//! Incremental WebM demuxing shared by video and audio tracks.

use super::backend::{MediaDecodeError, MediaMetadata, StreamingVideoDecoder, VideoFrame};
use super::vp8::FrameHeader;
use super::vp8_decoder::Vp8Decoder;
use super::vp9::split_superframe;
use super::vp9_decoder::Vp9Decoder;
use crate::audio::webm::{WebmAudioBlock, WebmAudioCodec, WebmAudioTrack, split_lace};

mod media;
mod av1;
pub use media::WebmMediaDecoder;

const EBML: u32 = 0x1a45dfa3;
const SEGMENT: u32 = 0x18538067;
const INFO: u32 = 0x1549a966;
const TRACKS: u32 = 0x1654ae6b;
const TRACK_ENTRY: u32 = 0xae;
const VIDEO: u32 = 0xe0;
const AUDIO: u32 = 0xe1;
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
const CODEC_PRIVATE: u32 = 0x63a2;
const SAMPLING_FREQUENCY: u32 = 0xb5;
const CHANNELS: u32 = 0x9f;
const CODEC_DELAY: u32 = 0x56aa;
const SEEK_PRE_ROLL: u32 = 0x56bb;
const DEFAULT_DURATION: u32 = 0x23e383;
const DISCARD_PADDING: u32 = 0x75a2;
const PIXEL_WIDTH: u32 = 0xb0;
const PIXEL_HEIGHT: u32 = 0xba;
const TIMESTAMP: u32 = 0xe7;

#[derive(Clone, Debug, PartialEq)]
pub struct VideoPacket {
    pub timestamp: f32,
    pub key_frame: bool,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum WebmPacket {
    Video(VideoPacket),
    Audio(WebmAudioBlock),
}

#[derive(Default)]
struct Track {
    number: u64,
    kind: u64,
    codec: Vec<u8>,
    width: Option<u32>,
    height: Option<u32>,
    codec_private: Vec<u8>,
    sampling_frequency: Option<f64>,
    channels: Option<u16>,
    codec_delay_ns: u64,
    seek_pre_roll_ns: u64,
    default_duration_ns: Option<u64>,
}

struct Parent {
    id: u32,
    end: Option<u64>,
}

pub struct WebmStream {
    pending: Vec<u8>,
    offset: u64,
    skip: u64,
    parents: Vec<Parent>,
    doc_type: Option<Vec<u8>>,
    track: Option<Track>,
    video_track: Option<u64>,
    requested_codec: Option<WebmVideoCodec>,
    video_codec: Option<WebmVideoCodec>,
    video_codec_private: Vec<u8>,
    width: Option<u32>,
    height: Option<u32>,
    time_code_scale: u64,
    duration_ticks: Option<f64>,
    cluster_timestamp: u64,
    audio_enabled: bool,
    audio_track: Option<WebmAudioTrack>,
    group_audio: Option<WebmAudioBlock>,
    group_discard_padding: Option<i64>,
}

/// VP8-only streaming decoder. Each push decodes at most one coded frame;
/// drain buffered samples with empty pushes before supplying more input.
pub struct WebmVp8Decoder {
    decoder: WebmVideoDecoder,
}

pub struct WebmVideoDecoder {
    stream: WebmVideoStream,
    codec: Option<CodecDecoder>,
    pending: std::collections::VecDeque<PendingPacket>,
}

struct PendingPacket {
    packet: VideoPacket,
    next_frame: usize,
}

enum CodecDecoder {
    Vp8(Vp8Decoder),
    Vp9(Vp9Decoder),
    Av1(av1::Av1PacketDecoder),
}

pub type WebmVideoStream = WebmStream;
pub type WebmVp8Stream = WebmStream;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WebmVideoCodec {
    Vp8,
    Vp9,
    Av1,
}

impl WebmVideoCodec {
    fn id(self) -> &'static [u8] {
        match self {
            Self::Vp8 => b"V_VP8",
            Self::Vp9 => b"V_VP9",
            Self::Av1 => b"V_AV1",
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
        Self {
            stream: WebmVideoStream::new(),
            codec: None,
            pending: Default::default(),
        }
    }
}

fn decode_video_packet(
    codec: &mut Option<CodecDecoder>,
    video_codec: Option<WebmVideoCodec>,
    codec_private: &[u8],
    pending: &mut PendingPacket,
) -> Result<(Option<VideoFrame>, bool), MediaDecodeError> {
    let packet = &pending.packet;
    if codec.is_none() {
        *codec = Some(match video_codec.ok_or(MediaDecodeError::Unsupported)? {
            WebmVideoCodec::Vp8 => CodecDecoder::Vp8(Vp8Decoder::new()),
            WebmVideoCodec::Vp9 => CodecDecoder::Vp9(Vp9Decoder::new()),
            WebmVideoCodec::Av1 => CodecDecoder::Av1(av1::Av1PacketDecoder::new(codec_private)?),
        });
    }
    let (frame, done) = match codec.as_mut().ok_or(MediaDecodeError::Unsupported)? {
        CodecDecoder::Vp8(decoder) => {
            let header = FrameHeader::parse(&packet.data)?;
            let decoded = decoder.decode(&packet.data)?;
            (header.show_frame.then(|| VideoFrame {
                width: decoded.width as u32,
                height: decoded.height as u32,
                rgba: std::sync::Arc::new(decoded.rgba()),
                timestamp: packet.timestamp,
            }), true)
        }
        CodecDecoder::Vp9(decoder) => {
            let coded = split_superframe(&packet.data)?;
            let decoded = decoder.decode(coded[pending.next_frame])?;
            pending.next_frame += 1;
            (decoded.map(|decoded| VideoFrame {
                width: decoded.width as u32,
                height: decoded.height as u32,
                rgba: std::sync::Arc::new(decoded.rgba()),
                timestamp: packet.timestamp,
            }), pending.next_frame == coded.len())
        }
        CodecDecoder::Av1(decoder) => {
            if pending.next_frame == 0 {
                decoder.begin_packet(&packet.data)?;
                pending.next_frame = 1;
            }
            decoder.next_frame(packet.timestamp)?
        }
    };
    Ok((frame, done))
}

impl StreamingVideoDecoder for WebmVideoDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<VideoFrame>, MediaDecodeError> {
        self.pending.extend(
            self.stream
                .push(bytes)?
                .into_iter()
                .map(|packet| PendingPacket {
                    packet,
                    next_frame: 0,
                }),
        );
        let mut frames = Vec::new();
        // Publish after one coded frame, including hidden reference updates. The
        // streaming caller drains buffered samples before reading more input.
        if let Some(pending) = self.pending.front_mut() {
            let (frame, done) = decode_video_packet(&mut self.codec, self.stream.video_codec(), &self.stream.video_codec_private, pending)?;
            frames.extend(frame);
            if done { self.pending.pop_front(); }
        }
        Ok(frames)
    }

    fn metadata(&self) -> Option<MediaMetadata> {
        self.stream.metadata()
    }

    fn finish(&self) -> Result<(), MediaDecodeError> {
        self.stream.finish()?;
        if self.pending.is_empty() {
            Ok(())
        } else {
            Err(MediaDecodeError::InvalidData(
                "undrained WebM video frames".into(),
            ))
        }
    }

    fn has_buffered_samples(&self) -> bool {
        !self.pending.is_empty()
    }
}

impl WebmVp8Decoder {
    pub fn new() -> Self {
        Self {
            decoder: WebmVideoDecoder {
                stream: WebmVideoStream::for_codec(WebmVideoCodec::Vp8),
                codec: None,
                pending: Default::default(),
            },
        }
    }
}

impl StreamingVideoDecoder for WebmVp8Decoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<VideoFrame>, MediaDecodeError> {
        self.decoder.push(bytes)
    }

    fn metadata(&self) -> Option<MediaMetadata> {
        self.decoder.metadata()
    }

    fn finish(&self) -> Result<(), MediaDecodeError> {
        self.decoder.finish()
    }

    fn has_buffered_samples(&self) -> bool {
        self.decoder.has_buffered_samples()
    }
}

impl Default for WebmStream {
    fn default() -> Self {
        Self::new()
    }
}

impl WebmStream {
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
            video_codec_private: Vec::new(),
            width: None,
            height: None,
            time_code_scale: 1_000_000,
            duration_ticks: None,
            cluster_timestamp: 0,
            audio_enabled: false,
            audio_track: None,
            group_audio: None,
            group_discard_padding: None,
        }
    }

    pub fn video_codec(&self) -> Option<WebmVideoCodec> {
        self.video_codec
    }

    pub fn audio_track(&self) -> Option<&WebmAudioTrack> {
        self.audio_track.as_ref()
    }

    pub fn metadata(&self) -> Option<MediaMetadata> {
        if self.video_track.is_none() && self.audio_track.is_none() {
            return None;
        }
        let sample_rate = self.audio_track.as_ref().and_then(|track| {
            if track.codec == WebmAudioCodec::Opus {
                Some(48000)
            } else if track.sampling_frequency.fract() == 0.0 {
                Some(track.sampling_frequency as u32)
            } else {
                None
            }
        });
        Some(MediaMetadata {
            duration: self
                .duration_ticks
                .map(|ticks| (ticks * self.time_code_scale as f64 / 1e9) as f32),
            width: self.width,
            height: self.height,
            sample_rate,
            channels: self.audio_track.as_ref().map(|track| track.channels),
        })
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<VideoPacket>, MediaDecodeError> {
        let mut packets = Vec::new();
        self.push_into(bytes, |packet| {
            if let WebmPacket::Video(packet) = packet {
                packets.push(packet);
            }
        })?;
        Ok(packets)
    }

    /// Enable audio from the first input chunk and emit both track types.
    /// The video-only `push` API does not copy audio packet payloads.
    pub fn push_packets(&mut self, bytes: &[u8]) -> Result<Vec<WebmPacket>, MediaDecodeError> {
        if !self.audio_enabled && (self.offset != 0 || !self.pending.is_empty()) {
            return Err(invalid("audio must be enabled before the first WebM input"));
        }
        self.audio_enabled = true;
        let mut packets = Vec::new();
        self.push_into(bytes, |packet| packets.push(packet))?;
        Ok(packets)
    }

    fn push_into(
        &mut self,
        bytes: &[u8],
        mut emit: impl FnMut(WebmPacket),
    ) -> Result<(), MediaDecodeError> {
        self.pending.extend_from_slice(bytes);
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
                if let Some(packet) = self.close_parent() {
                    emit(packet);
                }
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
                if let Some(packet) = self.close_parent() {
                    emit(packet);
                }
            }
            if self
                .parents
                .last()
                .is_some_and(|p| p.id == CLUSTER && p.end.is_none())
                && matches!(id, INFO | TRACKS | 0x1c53bb6b | 0x114d9b74)
            {
                if let Some(packet) = self.close_parent() {
                    emit(packet);
                }
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
            if is_container(id) && (id != AUDIO || self.audio_enabled) {
                if self.parents.len() >= 16 {
                    return Err(invalid("WebM nesting too deep"));
                }
                if id == TRACK_ENTRY {
                    self.track = Some(Track::default());
                }
                if id == CLUSTER {
                    self.cluster_timestamp = 0;
                }
                if id == BLOCK_GROUP {
                    if size.is_none() || self.parents.iter().any(|parent| parent.id == BLOCK_GROUP)
                    {
                        return Err(invalid("invalid WebM BlockGroup nesting or size"));
                    }
                    self.group_audio = None;
                    self.group_discard_padding = None;
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
                    | CODEC_PRIVATE
            ) || (self.audio_enabled
                && matches!(
                    id,
                    SAMPLING_FREQUENCY
                        | CHANNELS
                        | CODEC_DELAY
                        | SEEK_PRE_ROLL
                        | DEFAULT_DURATION
                        | DISCARD_PADDING
                ));
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
                TIME_CODE_SCALE => {
                    let scale = unsigned(payload)?;
                    if scale == 0 { return Err(invalid("zero WebM timestamp scale")); }
                    self.time_code_scale = scale;
                },
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
                CODEC_PRIVATE => {
                    if let Some(track) = &mut self.track {
                        track.codec_private = payload.to_vec();
                    }
                }
                SAMPLING_FREQUENCY => {
                    let rate = match payload.len() {
                        4 => f64::from(f32::from_be_bytes(payload.try_into().unwrap())),
                        8 => f64::from_be_bytes(payload.try_into().unwrap()),
                        _ => return Err(invalid("invalid audio sampling frequency")),
                    };
                    if !rate.is_finite() || rate <= 0.0 || rate > f64::from(u32::MAX) {
                        return Err(invalid("invalid audio sampling frequency"));
                    }
                    if let Some(track) = &mut self.track {
                        track.sampling_frequency = Some(rate);
                    }
                }
                CHANNELS => {
                    let channels = u16::try_from(unsigned(payload)?)
                        .ok()
                        .filter(|value| *value != 0)
                        .ok_or_else(|| invalid("invalid audio channel count"))?;
                    if let Some(track) = &mut self.track {
                        track.channels = Some(channels);
                    }
                }
                CODEC_DELAY | SEEK_PRE_ROLL | DEFAULT_DURATION => {
                    let value = unsigned(payload)?;
                    if let Some(track) = &mut self.track {
                        match id {
                            CODEC_DELAY => track.codec_delay_ns = value,
                            SEEK_PRE_ROLL => track.seek_pre_roll_ns = value,
                            _ => track.default_duration_ns = (value != 0).then_some(value),
                        }
                    }
                }
                DISCARD_PADDING => {
                    if !self.parents.iter().any(|parent| parent.id == BLOCK_GROUP) {
                        return Err(invalid("DiscardPadding outside BlockGroup"));
                    }
                    self.group_discard_padding = Some(signed(payload)?);
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
                        emit(WebmPacket::Video(packet));
                    } else if self.audio_enabled
                        && let Some(packet) = self.audio_block(payload)?
                    {
                        if id == BLOCK {
                            if !self.parents.iter().any(|parent| parent.id == BLOCK_GROUP)
                                || self.group_audio.replace(packet).is_some()
                            {
                                return Err(invalid("invalid audio BlockGroup"));
                            }
                        } else {
                            emit(WebmPacket::Audio(packet));
                        }
                    }
                }
                _ => {}
            }
            cursor += header_len + len;
        }
        self.pending.drain(..cursor);
        self.offset += cursor as u64;
        Ok(())
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
        if self.doc_type.is_none() || (self.video_track.is_none() && self.audio_track.is_none()) {
            return Err(MediaDecodeError::Unsupported);
        }
        Ok(())
    }

    fn close_parent(&mut self) -> Option<WebmPacket> {
        let parent = self.parents.pop()?;
        if parent.id == BLOCK_GROUP {
            let mut packet = self.group_audio.take()?;
            packet.discard_padding_ns = self.group_discard_padding.take();
            return Some(WebmPacket::Audio(packet));
        }
        if parent.id == TRACK_ENTRY {
            if let Some(track) = self.track.take() {
                let codec = [WebmVideoCodec::Vp8, WebmVideoCodec::Vp9, WebmVideoCodec::Av1]
                    .into_iter()
                    .find(|codec| track.codec == codec.id());
                if track.kind == 1
                    && self.video_track.is_none()
                    && codec.is_some_and(|codec| {
                        self.requested_codec
                            .is_none_or(|requested| requested == codec)
                    })
                {
                    self.video_track = Some(track.number);
                    self.video_codec = codec;
                    self.video_codec_private = track.codec_private.clone();
                    self.width = track.width;
                    self.height = track.height;
                }
                if self.audio_enabled
                    && track.kind == 2
                    && self.audio_track.is_none()
                    && let Some(codec) = WebmAudioCodec::from_id(&track.codec)
                {
                    self.audio_track = Some(WebmAudioTrack {
                        number: track.number,
                        codec,
                        sampling_frequency: track.sampling_frequency.unwrap_or(8000.0),
                        channels: track.channels.unwrap_or(1),
                        codec_private: track.codec_private,
                        codec_delay_ns: track.codec_delay_ns,
                        seek_pre_roll_ns: track.seek_pre_roll_ns,
                        default_duration_ns: track.default_duration_ns,
                    });
                }
            }
        }
        None
    }

    fn audio_block(&self, payload: &[u8]) -> Result<Option<WebmAudioBlock>, MediaDecodeError> {
        let Some(audio) = &self.audio_track else {
            return Ok(None);
        };
        let Some((Some(track), length)) = element_size(payload)? else {
            return Err(invalid("invalid audio block track"));
        };
        if track != audio.number {
            return Ok(None);
        }
        let header = payload
            .get(length..length + 3)
            .ok_or_else(|| invalid("truncated audio block"))?;
        let relative = i16::from_be_bytes([header[0], header[1]]);
        let ticks = i128::from(self.cluster_timestamp) + i128::from(relative);
        let timestamp_ns = ticks
            .checked_mul(i128::from(self.time_code_scale))
            .and_then(|timestamp| i64::try_from(timestamp).ok())
            .ok_or_else(|| invalid("audio timestamp overflow"))?;
        Ok(Some(WebmAudioBlock {
            timestamp_ns,
            packets: split_lace(&payload[length + 3..], header[2])?,
            discard_padding_ns: None,
        }))
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
        EBML | SEGMENT | INFO | TRACKS | TRACK_ENTRY | VIDEO | AUDIO | CLUSTER | BLOCK_GROUP
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

fn signed(bytes: &[u8]) -> Result<i64, MediaDecodeError> {
    let first = *bytes
        .first()
        .ok_or_else(|| invalid("empty WebM signed integer"))?;
    if bytes.len() > 8 {
        return Err(invalid("oversized WebM signed integer"));
    }
    let mut result = [if first & 0x80 != 0 { 0xff } else { 0 }; 8];
    result[8 - bytes.len()..].copy_from_slice(bytes);
    Ok(i64::from_be_bytes(result))
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
    use super::super::backend::{MediaSample, StreamingMediaDecoder};

    fn collect_media(bytes: &[u8], chunk_size: usize) -> Vec<MediaSample> {
        let mut decoder = WebmMediaDecoder::new();
        let mut samples = Vec::new();
        for chunk in bytes.chunks(chunk_size) {
            let batch = decoder.push_media(chunk).unwrap();
            assert!(batch.len() <= 1);
            samples.extend(batch);
            while decoder.has_buffered_samples() {
                assert!(decoder.finish().is_err());
                let batch = decoder.push_media(&[]).unwrap();
                assert!(batch.len() <= 1);
                samples.extend(batch);
            }
        }
        decoder.finish().unwrap();
        samples
    }

    #[test]
    fn timestamp_scale_must_be_positive() {
        let mut bytes = element(&[0x1a, 0x45, 0xdf, 0xa3], &element(&[0x42, 0x82], b"webm"));
        bytes.extend(element(&[0x18, 0x53, 0x80, 0x67],
            &element(&[0x15, 0x49, 0xa9, 0x66], &element(&[0x2a, 0xd7, 0xb1], &[0]))));
        assert!(WebmStream::new().push(&bytes).is_err());
        assert!(WebmMediaDecoder::new().push_media(&bytes).is_err());
    }

    #[test]
    fn media_delivery_preserves_vp9_superframes_and_reference_updates() {
        let bytes = include_bytes!("../../tests/fixtures/vp9-serial-static.webm");
        let mut decoder = WebmVideoDecoder::new();
        let mut expected = decoder.push(bytes).unwrap();
        while decoder.has_buffered_samples() { expected.extend(decoder.push(&[]).unwrap()); }
        decoder.finish().unwrap();
        assert!(!expected.is_empty());
        for chunk in [1, 7, 257, bytes.len()] {
            let actual = collect_media(bytes, chunk);
            assert_eq!(actual, expected.iter().cloned().map(MediaSample::Video).collect::<Vec<_>>());
        }
    }

    #[test]
    fn failed_audio_track_does_not_halt_video_or_repeat_errors() {
        let frame = &include_bytes!("../../tests/fixtures/vp8-keyframe.ivf")[44..];
        let header = FrameHeader::parse(frame).unwrap();
        for codec in [WebmAudioCodec::Opus, WebmAudioCodec::Vorbis] {
            let mut audio_track = element(&[0xd7], &[2]);
            audio_track.extend(element(&[0x83], &[2]));
            audio_track.extend(element(&[0x86], if codec == WebmAudioCodec::Opus { b"A_OPUS" } else { b"A_VORBIS" }));
            // Valid family-zero Opus is still unsupported for PCM synthesis;
            // corrupt Vorbis headers exercise independent audio failure.
            let private = if codec == WebmAudioCodec::Opus {
                b"OpusHead\x01\x02\x00\x00\x80\xbb\x00\x00\x00\x00\x00".as_slice()
            } else { b"invalid Vorbis headers" };
            audio_track.extend(element(&[0x63, 0xa2], private));
            let mut audio = element(&[0xb5], &48000f64.to_be_bytes());
            audio.extend(element(&[0x9f], &[2]));
            audio_track.extend(element(&[0xe1], &audio));
            let mut video_track = element(&[0xd7], &[1]);
            video_track.extend(element(&[0x83], &[1]));
            video_track.extend(element(&[0x86], b"V_VP8"));
            let mut video = element(&[0xb0], &header.width.unwrap().to_be_bytes());
            video.extend(element(&[0xba], &header.height.unwrap().to_be_bytes()));
            video_track.extend(element(&[0xe0], &video));
            let mut tracks = element(&[0xae], &audio_track);
            tracks.extend(element(&[0xae], &video_track));
            let mut segment = element(&[0x16, 0x54, 0xae, 0x6b], &tracks);
            let mut cluster = element(&[0xe7], &[0]);
            for _ in 0..2 {
                cluster.extend(element(&[0xa3], &[0x82, 0, 0, 0x80, 0xf8, 0]));
                let mut block = vec![0x81, 0, 0, 0x80];
                block.extend(frame);
                cluster.extend(element(&[0xa3], &block));
            }
            segment.extend(element(&[0x1f, 0x43, 0xb6, 0x75], &cluster));
            let mut bytes = element(&[0x1a, 0x45, 0xdf, 0xa3], &element(&[0x42, 0x82], b"webm"));
            bytes.extend(element(&[0x18, 0x53, 0x80, 0x67], &segment));
            for chunk_size in [1, 13, bytes.len()] {
                let samples = collect_media(&bytes, chunk_size);
                assert_eq!(samples.len(), 3);
                assert!(matches!(&samples[0], MediaSample::AudioError(_)));
                if codec == WebmAudioCodec::Opus { assert_eq!(samples[0], MediaSample::AudioError(MediaDecodeError::Unsupported)); }
                let expected = Vp8Decoder::new().decode(frame).unwrap().rgba();
                for sample in &samples[1..] {
                    let MediaSample::Video(frame) = sample else { panic!("video did not survive audio failure") };
                    assert_eq!(*frame.rgba, expected);
                }
            }
        }
    }

    #[test]
    fn av1_configuration_and_in_band_sequence_share_pixel_decoder() {
        // FFmpeg/SVT-AV1 binary fixture: a 64x64 neutral, skipped-DC frame.
        let coded = [
            0x12, 0x00, 0x0a, 0x0b, 0x02, 0x00, 0x00, 0x05, 0x15, 0x7f, 0xfc, 0x4a,
            0xf9, 0x00, 0x40, 0x32, 0x0c, 0x10, 0x00, 0xac, 0x02, 0x05, 0x14,
            0x20, 0x81, 0x00, 0x00, 0x98, 0x80,
        ];
        let obus = super::super::av1::ObuStream::new().push(&coded).unwrap();
        let sequence_obu = obus.iter().find(|obu| obu.kind == 1).unwrap();
        let sequence = super::super::av1::syntax::SequenceHeader::parse(&sequence_obu.payload).unwrap();
        let operating = &sequence.operating_points[0];
        let configuration = [0x81, (sequence.profile << 5) | operating.level,
            (u8::from(operating.tier) << 7) | (u8::from(sequence.monochrome) << 4)
                | (u8::from(sequence.subsampling_x) << 3) | (u8::from(sequence.subsampling_y) << 2)
                | sequence.chroma_sample_position, 0];
        for config_sequence in [false, true] {
            let mut private = configuration.to_vec();
            let mut packet = Vec::new();
            for obu in &obus {
                let target = if config_sequence && obu.kind == 1 { &mut private } else { &mut packet };
                target.extend([(obu.kind << 3) | 2, obu.payload.len() as u8]);
                target.extend(&obu.payload);
            }
            let mut track = element(&[0xd7], &[1]);
            track.extend(element(&[0x83], &[1]));
            track.extend(element(&[0x86], b"V_AV1"));
            track.extend(element(&[0x63, 0xa2], &private));
            let mut video = element(&[0xb0], &[64]);
            video.extend(element(&[0xba], &[64]));
            track.extend(element(&[0xe0], &video));
            let mut segment = element(&[0x16, 0x54, 0xae, 0x6b], &element(&[0xae], &track));
            let mut block = vec![0x81, 0, 12, 0x80];
            block.extend(packet);
            let mut cluster = element(&[0xe7], &[0]);
            cluster.extend(element(&[0xa3], &block));
            segment.extend(element(&[0x1f, 0x43, 0xb6, 0x75], &cluster));
            let mut bytes = element(&[0x1a, 0x45, 0xdf, 0xa3], &element(&[0x42, 0x82], b"webm"));
            bytes.extend(element(&[0x18, 0x53, 0x80, 0x67], &segment));
            for chunk_size in [1, 13, bytes.len()] {
                let actual = collect_media(&bytes, chunk_size);
                let [MediaSample::Video(frame)] = actual.as_slice() else { panic!("missing AV1 pixels"); };
                assert_eq!((frame.width, frame.height), (64, 64));
                assert!((frame.timestamp - 0.012).abs() < 1e-6);
                assert_eq!(frame.rgba.len(), 64 * 64 * 4);
                assert!(frame.rgba.chunks_exact(4).all(|pixel| pixel == [130, 130, 130, 255]));
                let mut legacy = WebmVideoDecoder::new();
                let mut expected = Vec::new();
                for chunk in bytes.chunks(chunk_size) {
                    expected.extend(legacy.push(chunk).unwrap());
                    while legacy.has_buffered_samples() { expected.extend(legacy.push(&[]).unwrap()); }
                }
                legacy.finish().unwrap();
                assert_eq!(expected, vec![frame.clone()]);
            }
            let mut bad = private.clone();
            bad[2] ^= 0x40;
            let mut decoder = av1::Av1PacketDecoder::new(&bad);
            if !config_sequence {
                let decoder = decoder.as_mut().unwrap();
                decoder.begin_packet(&coded).unwrap();
                assert!(matches!(decoder.next_frame(0.0), Err(MediaDecodeError::InvalidData(_))));
            } else {
                assert!(matches!(decoder, Err(MediaDecodeError::InvalidData(_))));
            }
        }
    }

    #[test]
    fn demuxes_supplied_av1_clip_through_shared_container() {
        let Some(path) = std::env::var_os("WEBMEDIA_AV1_SAMPLE") else { return; };
        let bytes = std::fs::read(path).unwrap();
        let mut stream = WebmStream::for_codec(WebmVideoCodec::Av1);
        let mut count = 0;
        let mut first = None;
        for chunk in bytes.chunks(16 * 1024) {
            for packet in stream.push(chunk).unwrap() {
                if first.is_none() { first = Some(packet.clone()); }
                count += 1;
            }
        }
        stream.finish().unwrap();
        assert_eq!(stream.video_codec(), Some(WebmVideoCodec::Av1));
        assert!(count > 0);
        if let Ok(expected) = std::env::var("WEBMEDIA_AV1_EXPECTED_PACKETS") {
            assert_eq!(count, expected.parse::<usize>().unwrap());
        }
        let first = first.unwrap();
        let obus = super::super::av1::ObuStream::new().push(&first.data).unwrap();
        let sequence = obus.iter().find(|obu| obu.kind == 1).unwrap();
        let sequence = super::super::av1::syntax::SequenceHeader::parse(&sequence.payload).unwrap();
        assert_eq!(stream.metadata().unwrap().width, Some(sequence.max_width));
        assert_eq!(stream.metadata().unwrap().height, Some(sequence.max_height));
        let mut codec = None;
        let mut packet = PendingPacket { packet: first, next_frame: 0 };
        let (frame, _) = decode_video_packet(&mut codec, stream.video_codec(), &stream.video_codec_private, &mut packet)
            .expect("supplied AV1 clip must reconstruct its first frame, not only route the packet");
        let frame = frame.expect("supplied AV1 clip must present its first frame");
        assert_eq!((frame.width, frame.height), (sequence.max_width, sequence.max_height));
        assert_eq!(frame.rgba.len(), frame.width as usize * frame.height as usize * 4);
        assert!(frame.rgba.chunks_exact(4).all(|pixel| pixel[3] == 255));
        eprintln!("AV1 packets={count} dimensions={}x{}", sequence.max_width, sequence.max_height);
    }

    #[test]
    fn bounded_vp9_delivery_preserves_superframes_and_reference_order() {
        for bytes in [
            include_bytes!("../../tests/fixtures/vp9-altref.webm").as_slice(),
            include_bytes!("../../tests/fixtures/vp9-serial-motion.webm").as_slice(),
            include_bytes!("../../tests/fixtures/vp9-lossless.webm").as_slice(),
        ] {
            let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
            let mut direct = Vp9Decoder::new();
            let mut expected = Vec::new();
            for packet in stream.push(bytes).unwrap() {
                for data in split_superframe(&packet.data).unwrap() {
                    if let Some(decoded) = direct.decode(data).unwrap() {
                        expected.push(VideoFrame {
                            width: decoded.width as u32,
                            height: decoded.height as u32,
                            rgba: std::sync::Arc::new(decoded.rgba()),
                            timestamp: packet.timestamp,
                        });
                    }
                }
            }
            assert!(!expected.is_empty());
            for chunk_size in [1, 257, 16384, bytes.len()] {
                let mut decoder = WebmVideoDecoder::new();
                let mut actual = Vec::new();
                for chunk in bytes.chunks(chunk_size) {
                    let batch = decoder.push(chunk).unwrap();
                    assert!(batch.len() <= 1);
                    actual.extend(batch);
                    while decoder.has_buffered_samples() {
                        assert!(decoder.finish().is_err());
                        let batch = decoder.push(&[]).unwrap();
                        assert!(batch.len() <= 1);
                        actual.extend(batch);
                    }
                }
                decoder.finish().unwrap();
                assert!(decoder.push(&[]).unwrap().is_empty());
                assert_eq!(actual, expected, "chunk_size={chunk_size}");
            }
        }
    }

    #[test]
    fn vp8_specialized_decoder_keeps_codec_selection_restricted() {
        let bytes = include_bytes!("../../tests/fixtures/vp9-lossless.webm");
        let mut decoder = WebmVp8Decoder::new();
        assert!(decoder.push(bytes).unwrap().is_empty());
        assert!(!decoder.has_buffered_samples());
        assert!(decoder.metadata().is_none());
        assert!(matches!(decoder.finish(), Err(MediaDecodeError::Unsupported)));
    }

    #[test]
    fn vp8_specialized_decoder_bounds_large_pushes_and_requires_drain() {
        let bytes = include_bytes!("../../tests/fixtures/vp8-motion.webm");
        let mut stream = WebmVp8Stream::new();
        let expected = stream.push(bytes).unwrap().into_iter()
            .filter(|packet| FrameHeader::parse(&packet.data).unwrap().show_frame).count();
        assert!(expected > 1);
        let mut decoder = WebmVp8Decoder::new();
        let first = decoder.push(bytes).unwrap();
        assert_eq!(first.len(), 1);
        assert!(decoder.has_buffered_samples());
        assert!(decoder.finish().is_err());
        let mut count = first.len();
        let mut last_timestamp = first[0].timestamp;
        while decoder.has_buffered_samples() {
            let batch = decoder.push(&[]).unwrap();
            assert!(batch.len() <= 1);
            for frame in batch {
                assert!(frame.timestamp >= last_timestamp);
                assert_eq!(frame.rgba.len(), frame.width as usize * frame.height as usize * 4);
                last_timestamp = frame.timestamp;
                count += 1;
            }
        }
        assert_eq!(count, expected);
        decoder.finish().unwrap();
        assert!(decoder.push(&[]).unwrap().is_empty());
    }

    #[test]
    fn vp8_hidden_reference_stream_is_independent_of_input_chunk_boundaries() {
        let bytes = include_bytes!("../../tests/fixtures/vp8-altref.webm");
        let mut complete = WebmVp8Decoder::new();
        let mut expected = complete.push(bytes).unwrap();
        while complete.has_buffered_samples() {
            expected.extend(complete.push(&[]).unwrap());
        }
        complete.finish().unwrap();
        assert_eq!(expected.len(), 120);
        let metadata = complete.metadata().unwrap();
        let mut stream = WebmVp8Stream::new();
        let timestamps: Vec<_> = stream
            .push(bytes)
            .unwrap()
            .into_iter()
            .filter(|packet| FrameHeader::parse(&packet.data).unwrap().show_frame)
            .map(|packet| packet.timestamp)
            .collect();
        assert_eq!(
            expected
                .iter()
                .map(|frame| frame.timestamp)
                .collect::<Vec<_>>(),
            timestamps
        );
        for specialized in [false, true] {
            for chunk_size in [1, 17, 257, 4096] {
                let mut decoder: Box<dyn StreamingVideoDecoder> = if specialized {
                    Box::new(WebmVp8Decoder::new())
                } else {
                    Box::new(WebmVideoDecoder::new())
                };
                assert!(decoder.metadata().is_none());
                let mut actual = Vec::new();
                let mut emitted_before_end = false;
                for (index, chunk) in bytes.chunks(chunk_size).enumerate() {
                    let frames = decoder.push(chunk).unwrap();
                    assert!(frames.len() <= 1);
                    if !frames.is_empty() && (index + 1) * chunk_size < bytes.len() {
                        emitted_before_end = true;
                    }
                    actual.extend(frames);
                    while decoder.has_buffered_samples() {
                        let frames = decoder.push(&[]).unwrap();
                        assert!(frames.len() <= 1);
                        actual.extend(frames);
                    }
                }
                assert!(emitted_before_end);
                assert!(decoder.push(&[]).unwrap().is_empty());
                decoder.finish().unwrap();
                assert_eq!(decoder.metadata(), Some(metadata.clone()));
                assert_eq!(
                    actual, expected,
                    "specialized={specialized}, chunk_size={chunk_size}"
                );
            }
        }
    }

    #[test]
    fn reconstructs_first_vp9_transform_against_reference() {
        let (Ok(sample), Ok(reference)) = (
            std::env::var("WEBMEDIA_VP9_SAMPLE"),
            std::env::var("WEBMEDIA_VP9_REFERENCE"),
        ) else {
            return;
        };
        let bytes = std::fs::read(sample).unwrap();
        let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
        let packet = stream.push(&bytes).unwrap().into_iter().next().unwrap();
        let frames = super::super::vp9::split_superframe(&packet.data).unwrap();
        let layout = super::super::vp9::KeyframeLayout::parse(frames[0]).unwrap();
        let compressed =
            super::super::vp9_compressed::CompressedHeader::parse_keyframe(&layout).unwrap();
        let tiles = layout.tile_partitions().unwrap();
        let (blocks, next_partition) =
            super::super::vp9_tile::first_tile_row_prefix(&layout, &compressed, tiles[0], 7)
                .unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(
            next_partition,
            Some(super::super::vp9_tile::Partition::Split)
        );
        let leaf =
            super::super::vp9_tile::first_split_top_left_16x16(&layout, &compressed, tiles[0], 7)
                .unwrap()
                .unwrap();
        let generic = super::super::vp9_tile::decode_keyframe_tile_prefix(
            &layout,
            &compressed,
            tiles[0],
            0,
            4,
        )
        .unwrap();
        assert_eq!(generic.len(), 4);
        for (index, expected) in blocks.iter().enumerate() {
            assert_eq!((generic[index].x, generic[index].y), (index * 64, 0));
            assert_eq!(&generic[index].block, expected);
        }
        assert_eq!((generic[3].x, generic[3].y), (192, 0));
        assert_eq!(generic[3].block, leaf);
        let first_eight = super::super::vp9_tile::decode_keyframe_tile_prefix(
            &layout,
            &compressed,
            tiles[0],
            0,
            5,
        )
        .unwrap();
        assert_eq!(
            (
                first_eight[4].x,
                first_eight[4].y,
                first_eight[4].block.y_mode
            ),
            (208, 0, 0)
        );
        let mixed_blocks = super::super::vp9_tile::decode_keyframe_tile_prefix(
            &layout,
            &compressed,
            tiles[0],
            0,
            54,
        )
        .unwrap();
        assert_eq!(mixed_blocks.len(), 54);
        if std::env::var_os("WEBMEDIA_VP9_PROBE").is_some() {
            eprintln!(
                "VP9 first 54 blocks: {:?}",
                mixed_blocks
                    .iter()
                    .map(|entry| (
                        entry.x,
                        entry.y,
                        entry.width,
                        entry.height,
                        entry.block.y_mode
                    ))
                    .collect::<Vec<_>>()
            );
        }
        if std::env::var_os("WEBMEDIA_VP9_PROBE").is_some() {
            let decode = |limit| {
                super::super::vp9_tile::decode_keyframe_tile_prefix(
                    &layout,
                    &compressed,
                    tiles[0],
                    0,
                    limit,
                )
            };
            let mut good = 4;
            let mut bad = 8;
            while decode(bad).is_ok() && bad < 16384 {
                good = bad;
                bad *= 2;
            }
            while good + 1 < bad {
                let middle = good + (bad - good) / 2;
                if decode(middle).is_ok() {
                    good = middle;
                } else {
                    bad = middle;
                }
            }
            let decoded = decode(good).unwrap();
            eprintln!(
                "VP9 tile prefix: {good} blocks decoded; last={:?}; next limit {bad}: {:?}",
                decoded.last().map(|entry| (
                    entry.x,
                    entry.y,
                    entry.width,
                    entry.height,
                    entry.block.y_mode
                )),
                decode(bad).err()
            );
        }
        let mut plane = super::super::vp8_predict::Plane::new(216, 64);
        for (block_index, block) in blocks.iter().enumerate() {
            let q = match layout.segment_alt_q[block.segment_id as usize] {
                Some(alt) if layout.segmentation_abs_or_delta_update => i32::from(alt),
                Some(alt) => i32::from(layout.base_q_idx) + i32::from(alt),
                None => i32::from(layout.base_q_idx),
            };
            for (index, coefficients) in
                block.luma_coefficients.as_ref().unwrap().iter().enumerate()
            {
                let x = block_index * 64 + index % 2 * 32;
                let y = index / 2 * 32;
                super::super::vp9_transform::reconstruct_32x32_intra(
                    &mut plane,
                    x,
                    y,
                    block.y_mode,
                    x != 0,
                    y != 0,
                    false,
                    216,
                    64,
                    coefficients,
                    layout.header.bit_depth.unwrap(),
                    q + i32::from(layout.delta_q_y_dc),
                    q,
                )
                .unwrap();
            }
        }
        let leaf_q = match layout.segment_alt_q[leaf.segment_id as usize] {
            Some(alt) if layout.segmentation_abs_or_delta_update => i32::from(alt),
            Some(alt) => i32::from(layout.base_q_idx) + i32::from(alt),
            None => i32::from(layout.base_q_idx),
        };
        super::super::vp9_transform::reconstruct_16x16_intra(
            &mut plane,
            192,
            0,
            leaf.y_mode,
            true,
            false,
            216,
            64,
            &leaf.luma_coefficients.as_ref().unwrap()[0],
            layout.header.bit_depth.unwrap(),
            leaf_q + i32::from(layout.delta_q_y_dc),
            leaf_q,
        )
        .unwrap();
        let eight = &first_eight[4].block;
        let eight_q = match layout.segment_alt_q[eight.segment_id as usize] {
            Some(alt) if layout.segmentation_abs_or_delta_update => i32::from(alt),
            Some(alt) => i32::from(layout.base_q_idx) + i32::from(alt),
            None => i32::from(layout.base_q_idx),
        };
        super::super::vp9_transform::reconstruct_8x8_intra(
            &mut plane,
            208,
            0,
            eight.y_mode,
            true,
            false,
            216,
            64,
            &eight.luma_coefficients.as_ref().unwrap()[0],
            layout.header.bit_depth.unwrap(),
            eight_q + i32::from(layout.delta_q_y_dc),
            eight_q,
        )
        .unwrap();
        let reference = std::fs::read(reference).unwrap();
        let width = layout.header.width.unwrap() as usize;
        if std::env::var_os("WEBMEDIA_VP9_RECON_PROBE").is_some() {
            let height = layout.header.height.unwrap() as usize;
            let decoded = super::super::vp9_tile::decode_keyframe_tile_prefix(
                &layout,
                &compressed,
                tiles[0],
                0,
                54,
            )
            .unwrap();
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
                for (transform_index, coefficients) in
                    block.luma_coefficients.as_ref().unwrap().iter().enumerate()
                {
                    let block_x = transform_index % transforms_wide;
                    let block_y = transform_index / transforms_wide;
                    let x = entry.x + block_x * transform_size;
                    let y = entry.y + block_y * transform_size;
                    let mode = block
                        .sub_modes
                        .map_or(block.y_mode, |modes| modes[block_y * 2 + block_x]);
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
                        eprintln!(
                            "VP9 contiguous block {index} at ({}, {}) {}x{} y_mode={} sub={:?} tx={} large={large}/{total} max={max_diff}",
                            entry.x,
                            entry.y,
                            entry.width,
                            entry.height,
                            block.y_mode,
                            block.sub_modes,
                            block.tx_size
                        );
                    }
                }
            }
            eprintln!(
                "VP9 contiguous reconstruction: {} suspicious of {} blocks",
                suspicious,
                decoded.len()
            );
        }
        if std::env::var_os("WEBMEDIA_VP9_CHROMA_PROBE").is_some() {
            let height = layout.header.height.unwrap() as usize;
            let chroma_width = width / 2;
            let chroma_height = height / 2;
            let decoded = super::super::vp9_tile::decode_keyframe_tile_prefix(
                &layout,
                &compressed,
                tiles[0],
                0,
                96,
            )
            .unwrap();
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
                    let tx_size = entry
                        .block
                        .tx_size
                        .min((block_width.min(block_height) / 4).trailing_zeros() as u8);
                    let transform_size = 4usize << tx_size;
                    let transforms_wide = block_width / transform_size;
                    let coefficients =
                        &entry.block.chroma_coefficients.as_ref().unwrap()[plane_index];
                    let q = match layout.segment_alt_q[entry.block.segment_id as usize] {
                        Some(alt) if layout.segmentation_abs_or_delta_update => i32::from(alt),
                        Some(alt) => i32::from(layout.base_q_idx) + i32::from(alt),
                        None => i32::from(layout.base_q_idx),
                    };
                    for (transform_index, transform) in coefficients
                        .chunks_exact(transform_size * transform_size)
                        .enumerate()
                    {
                        let x = x0 + transform_index % transforms_wide * transform_size;
                        let y = y0 + transform_index / transforms_wide * transform_size;
                        let result = match transform_size {
                            32 => super::super::vp9_transform::reconstruct_32x32_intra(
                                &mut oracle,
                                x,
                                y,
                                entry.block.uv_mode,
                                x != 0,
                                y != 0,
                                false,
                                chroma_width,
                                chroma_height,
                                transform,
                                8,
                                q + i32::from(layout.delta_q_uv_dc),
                                q + i32::from(layout.delta_q_uv_ac),
                            ),
                            16 => super::super::vp9_transform::reconstruct_16x16_intra(
                                &mut oracle,
                                x,
                                y,
                                entry.block.uv_mode,
                                x != 0,
                                y != 0,
                                chroma_width,
                                chroma_height,
                                transform,
                                8,
                                q + i32::from(layout.delta_q_uv_dc),
                                q + i32::from(layout.delta_q_uv_ac),
                            ),
                            8 => super::super::vp9_transform::reconstruct_8x8_intra(
                                &mut oracle,
                                x,
                                y,
                                entry.block.uv_mode,
                                x != 0,
                                y != 0,
                                chroma_width,
                                chroma_height,
                                transform,
                                8,
                                q + i32::from(layout.delta_q_uv_dc),
                                q + i32::from(layout.delta_q_uv_ac),
                            ),
                            4 => super::super::vp9_transform::reconstruct_4x4_intra(
                                &mut oracle,
                                x,
                                y,
                                entry.block.uv_mode,
                                x != 0,
                                y != 0,
                                chroma_width,
                                chroma_height,
                                transform,
                                8,
                                q + i32::from(layout.delta_q_uv_dc),
                                q + i32::from(layout.delta_q_uv_ac),
                            ),
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
                        eprintln!(
                            "VP9 chroma {plane_index} block {index} at ({}, {}) {}x{}: {large}/{total} pixels differ by >8, mode={}",
                            entry.x, entry.y, entry.width, entry.height, entry.block.uv_mode
                        );
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
            eprintln!(
                "VP9 sample layout: segmentation={} update_map={} tile_grid={}x{} tile_bytes={:?}",
                layout.segmentation_enabled,
                layout.segmentation_update_map,
                1usize << layout.tile_cols_log2,
                1usize << layout.tile_rows_log2,
                tiles.iter().map(|tile| tile.len()).collect::<Vec<_>>()
            );
            let height = layout.header.height.unwrap() as usize;
            let decoded = super::super::vp9_tile::decode_keyframe_tile_prefix(
                &layout,
                &compressed,
                tiles[0],
                0,
                1200,
            )
            .unwrap();
            let mut oracle = super::super::vp8_predict::Plane::new(width, height);
            oracle.pixels.copy_from_slice(&reference[..width * height]);
            let mut suspicious = 0;
            let mut checked = 0;
            for (index, entry) in decoded.iter().enumerate() {
                let block = &entry.block;
                if block.y_mode != 0
                    || block.sub_modes.is_some()
                    || block.tx_size == 0
                    || entry.x + entry.width > width
                    || entry.y + entry.height > height
                {
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
                            &mut oracle,
                            x,
                            y,
                            0,
                            x != 0,
                            y != 0,
                            false,
                            width,
                            height,
                            transform,
                            8,
                            q + i32::from(layout.delta_q_y_dc),
                            q,
                        ),
                        16 => super::super::vp9_transform::reconstruct_16x16_intra(
                            &mut oracle,
                            x,
                            y,
                            0,
                            x != 0,
                            y != 0,
                            width,
                            height,
                            transform,
                            8,
                            q + i32::from(layout.delta_q_y_dc),
                            q,
                        ),
                        8 => super::super::vp9_transform::reconstruct_8x8_intra(
                            &mut oracle,
                            x,
                            y,
                            0,
                            x != 0,
                            y != 0,
                            width,
                            height,
                            transform,
                            8,
                            q + i32::from(layout.delta_q_y_dc),
                            q,
                        ),
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
                        eprintln!(
                            "VP9 pixel probe block {index} at ({}, {}) {}x{}: {large}/{total} pixels differ by >4",
                            entry.x, entry.y, entry.width, entry.height
                        );
                        if index == 26 || index == 87 {
                            eprintln!(
                                "  q={q} tx={} skip={} first={:?} eob={} coeffs={:?}",
                                block.tx_size,
                                block.skip,
                                block.first_luma_coefficient,
                                block.first_luma_eob,
                                coefficients
                                    .iter()
                                    .map(|values| values.iter().filter(|&&v| v != 0).count())
                                    .collect::<Vec<_>>()
                            );
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
        eprintln!(
            "VP9 tile prefix luma: {matches}/11280 matching interior pixels; mismatches={mismatches:?}"
        );
        assert!(matches > 10000, "tile prefix differs from reference frame");
        let leaf_y_matches = (1..15)
            .flat_map(|y| (193..207).map(move |x| (x, y)))
            .filter(|&(x, y)| plane.pixels[y * 216 + x] == reference[y * width + x])
            .count();
        eprintln!("VP9 split leaf luma: {leaf_y_matches}/196 matching interior pixels");
        assert!(
            leaf_y_matches > 160,
            "split leaf luma differs from reference frame"
        );
        let eight_matches = (1..7)
            .flat_map(|y| (209..215).map(move |x| (x, y)))
            .filter(|&(x, y)| plane.pixels[y * 216 + x] == reference[y * width + x])
            .count();
        eprintln!("VP9 first 8x8 luma: {eight_matches}/36 matching interior pixels");
        assert!(
            eight_matches > 25,
            "first 8x8 block differs from reference frame"
        );
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
                    &mut chroma,
                    block_index * 32,
                    0,
                    block.uv_mode,
                    block_index != 0,
                    false,
                    false,
                    104,
                    32,
                    &block.chroma_coefficients.as_ref().unwrap()[plane_index],
                    layout.header.bit_depth.unwrap(),
                    q + i32::from(layout.delta_q_uv_dc),
                    q + i32::from(layout.delta_q_uv_ac),
                )
                .unwrap();
            }
            super::super::vp9_transform::reconstruct_8x8_intra(
                &mut chroma,
                96,
                0,
                leaf.uv_mode,
                true,
                false,
                104,
                32,
                &leaf.chroma_coefficients.as_ref().unwrap()[plane_index],
                layout.header.bit_depth.unwrap(),
                leaf_q + i32::from(layout.delta_q_uv_dc),
                leaf_q + i32::from(layout.delta_q_uv_ac),
            )
            .unwrap();
            let offset =
                width * layout.header.height.unwrap() as usize + plane_index * chroma_plane_size;
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
            eprintln!(
                "VP9 tile prefix chroma {plane_index}: {chroma_matches}/2576 matching interior pixels; mismatches={chroma_mismatches:?}"
            );
            assert!(
                chroma_matches > 2300,
                "chroma plane differs from reference frame"
            );
            let leaf_chroma_matches = (1..7)
                .flat_map(|y| (97..103).map(move |x| (x, y)))
                .filter(|&(x, y)| {
                    chroma.pixels[y * 104 + x] == reference[offset + y * chroma_width + x]
                })
                .count();
            eprintln!(
                "VP9 split leaf chroma {plane_index}: {leaf_chroma_matches}/36 matching interior pixels"
            );
            assert!(
                leaf_chroma_matches > 25,
                "split leaf chroma differs from reference frame"
            );
        }
    }

    #[test]
    fn demuxes_vp9_track_and_parses_real_keyframe() {
        let Ok(path) = std::env::var("WEBMEDIA_VP9_SAMPLE") else {
            return;
        };
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
        assert!(
            !arithmetic.read_bit().unwrap(),
            "invalid VP9 arithmetic marker"
        );
        let compressed =
            super::super::vp9_compressed::CompressedHeader::parse_keyframe(&layout).unwrap();
        assert!(compressed.tx_mode <= 4);
        for tile in &tiles {
            let mut arithmetic = super::super::vp8::BoolDecoder::new(tile).unwrap();
            assert!(
                !arithmetic.read_bit().unwrap(),
                "invalid VP9 tile arithmetic marker"
            );
        }
        let first_partition = super::super::vp9_tile::first_partition(tiles[0]).unwrap();
        let first_block =
            super::super::vp9_tile::first_block(&layout, &compressed, tiles[0]).unwrap();
        let (row_prefix, next_partition) =
            super::super::vp9_tile::first_tile_row_prefix(&layout, &compressed, tiles[0], 7)
                .unwrap();
        let split_child =
            super::super::vp9_tile::first_split_top_left_path(&layout, &compressed, tiles[0], 7)
                .unwrap();
        let split_leaf =
            super::super::vp9_tile::first_split_top_left_16x16(&layout, &compressed, tiles[0], 7)
                .unwrap();
        if let Some(ref leaf) = split_leaf {
            assert_eq!(leaf.block_size, 16);
            assert_eq!(leaf.luma_coefficients.as_ref().unwrap()[0].len(), 256);
            assert!(
                leaf.chroma_coefficients
                    .as_ref()
                    .unwrap()
                    .iter()
                    .all(|plane| plane.len() == 64)
            );
        }
        assert_eq!(
            row_prefix.first().map(|block| block.first_luma_coefficient),
            first_block
                .as_ref()
                .map(|block| block.first_luma_coefficient)
        );
        if let Some(ref block) = first_block {
            assert_eq!(block.partition, first_partition);
            assert!(block.segment_id < 8 && block.tx_size < 4);
            assert!(block.y_mode < 10 && block.uv_mode < 10);
            assert!(block.first_luma_token.is_none_or(|token| token <= 10));
            assert_eq!(
                block.first_luma_token.is_some(),
                block.first_luma_coefficient.is_some()
            );
            if let Some(coefficients) = &block.luma_coefficients {
                assert_eq!(coefficients.len(), 4);
                assert!(coefficients.iter().all(|transform| transform.len() == 1024));
                assert_eq!(
                    coefficients[0][0],
                    block.first_luma_coefficient.unwrap_or(0)
                );
                assert!(block.first_luma_eob <= 1024);
            }
            if let Some(planes) = &block.chroma_coefficients {
                assert!(planes.iter().all(|coefficients| coefficients.len() == 1024));
            }
        }
        if std::env::var_os("WEBMEDIA_VP9_REPORT").is_some() {
            eprintln!(
                "VP9 keyframe: {}x{}, q={}, lossless={}, tiles={}, tx_mode={}, prob_updates={}, skip_probs={:?}, first_partition={first_partition:?}, row_prefix={}, next_partition={next_partition:?}, split_child={split_child:?}, split_leaf={:?}, first_block={:?}",
                header.width.unwrap(),
                header.height.unwrap(),
                layout.base_q_idx,
                layout.lossless,
                tiles.len(),
                compressed.tx_mode,
                compressed.probability_updates,
                compressed.skip_probs,
                row_prefix.len(),
                split_leaf.as_ref().map(|leaf| (
                    leaf.y_mode,
                    leaf.uv_mode,
                    leaf.first_luma_coefficient
                )),
                first_block.as_ref().map(|block| (
                    block.segment_id,
                    block.skip,
                    block.tx_size,
                    block.y_mode,
                    block.uv_mode,
                    block.first_luma_token,
                    block.first_luma_coefficient,
                    block.first_luma_eob
                ))
            );
        }
    }

    #[test]
    fn scans_every_vp9_packet_in_supplied_webm() {
        let Ok(path) = std::env::var("WEBMEDIA_VP9_SAMPLE") else {
            return;
        };
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
        let Ok(path) = std::env::var("WEBMEDIA_WEBM_SAMPLE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let mut decoder = WebmVp8Decoder::new();
        let mut frames = Vec::new();
        for chunk in bytes.chunks(1024) {
            frames.extend(decoder.push(chunk).unwrap());
            while frames.len() < 4 && decoder.has_buffered_samples() {
                frames.extend(decoder.push(&[]).unwrap());
            }
            if frames.len() >= 4 {
                break;
            }
        }
        let metadata = decoder.metadata().unwrap();
        assert!(frames.len() >= 4);
        assert!(
            frames
                .windows(2)
                .all(|pair| pair[0].timestamp <= pair[1].timestamp)
        );
        assert!(frames.iter().all(|frame| {
            Some(frame.width) == metadata.width
                && Some(frame.height) == metadata.height
                && frame.rgba.len() == frame.width as usize * frame.height as usize * 4
        }));
    }

    #[test]
    fn auto_decoder_plays_vp8_and_vp9_webm() {
        for variable in ["WEBMEDIA_WEBM_SAMPLE", "WEBMEDIA_VP9_SAMPLE"] {
            let Ok(path) = std::env::var(variable) else {
                continue;
            };
            let mut source = std::fs::File::open(path).unwrap();
            let mut decoder = WebmVideoDecoder::new();
            let mut buffer = [0u8; 16 * 1024];
            let mut frames = Vec::new();
            while frames.len() < 3 {
                let count = std::io::Read::read(&mut source, &mut buffer).unwrap();
                assert!(count != 0, "{variable} ended before three frames");
                frames.extend(decoder.push(&buffer[..count]).unwrap());
                while decoder.has_buffered_samples() {
                    frames.extend(decoder.push(&[]).unwrap());
                }
            }
            assert!(frames.len() >= 3);
            assert!(
                frames
                    .windows(2)
                    .all(|pair| pair[0].timestamp <= pair[1].timestamp)
            );
            assert!(
                frames
                    .iter()
                    .all(|frame| frame.rgba.len()
                        == frame.width as usize * frame.height as usize * 4)
            );
            assert_eq!(decoder.metadata().unwrap().width, Some(frames[0].width));
        }
    }

    #[test]
    #[ignore = "manual streamed frame-delivery timing"]
    fn benchmark_streaming_delivery() {
        use std::time::Instant;
        let path = std::env::var("WEBMEDIA_VP9_SAMPLE")
            .or_else(|_| std::env::var("WEBMEDIA_WEBM_SAMPLE"))
            .unwrap();
        let mut source = std::fs::File::open(path).unwrap();
        let mut decoder = WebmVideoDecoder::new();
        let mut buffer = [0u8; 16 * 1024];
        let started = Instant::now();
        let mut frames = 0usize;
        let mut calls = 0usize;
        let mut max_batch = 0usize;
        let mut max_call = std::time::Duration::ZERO;
        while frames < 120 {
            let count = if decoder.has_buffered_samples() {
                0
            } else {
                std::io::Read::read(&mut source, &mut buffer).unwrap()
            };
            assert!(
                count != 0 || decoder.has_buffered_samples(),
                "clip ended before 120 frames"
            );
            let before = Instant::now();
            let batch = decoder.push(&buffer[..count]).unwrap();
            let elapsed = before.elapsed();
            calls += 1;
            max_batch = max_batch.max(batch.len());
            max_call = max_call.max(elapsed);
            if std::env::var_os("WEBMEDIA_WEBM_TRACE_DELIVERY").is_some() {
                for frame in &batch {
                    eprintln!(
                        "delivery frame={} timestamp={:.3} decode_us={}",
                        frames,
                        frame.timestamp,
                        elapsed.as_micros()
                    );
                }
            }
            if frames == 0 && !batch.is_empty() {
                eprintln!(
                    "first delivery {:?}: {} frames",
                    started.elapsed(),
                    batch.len()
                );
            }
            frames += batch.len();
        }
        eprintln!(
            "streamed frames={frames} calls={calls} max_batch={max_batch} max_call={max_call:?} total={:?}",
            started.elapsed()
        );
    }

    #[test]
    fn streams_entire_supplied_webm() {
        let Ok(path) = std::env::var("WEBMEDIA_WEBM_SAMPLE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let mut decoder = WebmVp8Decoder::new();
        let mut stream = WebmVp8Stream::new();
        let mut expected = 0usize;
        let mut count = 0usize;
        let mut last_timestamp = None;
        let start = std::time::Instant::now();
        for chunk in bytes.chunks(16384) {
            for packet in stream.push(chunk).unwrap() {
                expected += usize::from(FrameHeader::parse(&packet.data).unwrap().show_frame);
            }
            let mut batch = decoder.push(chunk).unwrap();
            loop {
                assert!(batch.len() <= 1);
                for frame in batch {
                    assert!(last_timestamp.is_none_or(|previous| frame.timestamp >= previous));
                    assert_eq!(
                        frame.rgba.len(),
                        frame.width as usize * frame.height as usize * 4
                    );
                    last_timestamp = Some(frame.timestamp);
                    count += 1;
                }
                if !decoder.has_buffered_samples() { break; }
                batch = decoder.push(&[]).unwrap();
            }
        }
        stream.finish().unwrap();
        decoder.finish().unwrap();
        assert!(count > 1);
        assert_eq!(
            count, expected,
            "every visible VP8 packet must produce a frame"
        );
        if std::env::var_os("WEBMEDIA_VP8_REPORT").is_some() {
            eprintln!(
                "streamed {count} VP8 frames in {:.3}s",
                start.elapsed().as_secs_f64()
            );
        }
    }

    #[test]
    #[ignore = "requires WEBMEDIA_WEBM_SAMPLE and the ffmpeg binary pixel oracle"]
    fn supplied_vp8_stream_matches_binary_pixel_oracle() {
        use std::io::Read;
        use std::process::{Command, Stdio};

        struct Oracle(std::process::Child);
        impl Drop for Oracle {
            fn drop(&mut self) {
                if self.0.try_wait().ok().flatten().is_none() {
                    let _ = self.0.kill();
                }
                let _ = self.0.wait();
            }
        }

        let path = std::env::var("WEBMEDIA_WEBM_SAMPLE")
            .expect("set WEBMEDIA_WEBM_SAMPLE to a VP8 WebM fixture");
        let bytes = std::fs::read(&path).unwrap();
        let mut oracle = Oracle(Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin", "-threads", "1", "-i", &path,
                "-map", "0:v:0", "-fps_mode", "passthrough", "-pix_fmt", "yuv420p",
                "-f", "rawvideo", "pipe:1"])
            .stdout(Stdio::piped())
            .spawn().expect("start ffmpeg binary pixel oracle"));
        let mut output = oracle.0.stdout.take().unwrap();
        let mut stream = WebmVp8Stream::new();
        let mut decoder = super::super::vp8_decoder::Vp8Decoder::new();
        let mut row = Vec::new();
        let mut visible = 0usize;
        let mut samples = 0usize;
        for chunk in bytes.chunks(16384) {
            for packet in stream.push(chunk).unwrap() {
                let header = FrameHeader::parse(&packet.data).unwrap();
                let frame = decoder.decode(&packet.data).unwrap();
                if !header.show_frame { continue; }
                for (name, plane, width, height) in [
                    ("Y", &frame.y, frame.width, frame.height),
                    ("U", &frame.u, frame.width.div_ceil(2), frame.height.div_ceil(2)),
                    ("V", &frame.v, frame.width.div_ceil(2), frame.height.div_ceil(2)),
                ] {
                    row.resize(width, 0);
                    for y in 0..height {
                        output.read_exact(&mut row).expect("oracle ended before decoded frame");
                        let actual = &plane.pixels[y * plane.width..y * plane.width + width];
                        if let Some(x) = actual.iter().zip(&row).position(|(a, b)| a != b) {
                            panic!("VP8 frame {visible} plane {name} ({x},{y}): decoded {} oracle {}",
                                actual[x], row[x]);
                        }
                        samples += width;
                    }
                }
                visible += 1;
            }
        }
        stream.finish().unwrap();
        assert!(visible > 1);
        assert_eq!(output.read(&mut [0]).unwrap(), 0, "oracle has extra visible frames");
        assert!(oracle.0.wait().unwrap().success(), "ffmpeg binary oracle failed");
        eprintln!("VP8 binary oracle: {visible} visible frames, {samples} exact YUV samples");
    }

    fn element(id: &[u8], value: &[u8]) -> Vec<u8> {
        assert!(value.len() < 16383);
        let mut bytes = id.to_vec();
        if value.len() < 127 {
            bytes.push(0x80 | value.len() as u8);
        } else {
            bytes.extend([0x40 | (value.len() >> 8) as u8, value.len() as u8]);
        }
        bytes.extend_from_slice(value);
        bytes
    }

    #[test]
    fn synthesizes_spec_built_floor_zero_stream() {
        struct Bits(Vec<u8>, usize);
        impl Bits {
            fn put(&mut self, value: u32, width: usize) {
                for bit in 0..width {
                    if self.1 % 8 == 0 {
                        self.0.push(0);
                    }
                    let index = self.1 / 8;
                    self.0[index] |= (((value >> bit) & 1) as u8) << (self.1 % 8);
                    self.1 += 1;
                }
            }
            fn book(
                &mut self,
                dimensions: u32,
                vector: bool,
                minimum: u32,
                delta: u32,
                values: &[u32],
                width: usize,
            ) {
                self.put(0x564342, 24);
                self.put(dimensions, 16);
                self.put(if vector { 4 } else { 1 }, 24);
                self.put(0, 1); // Unordered, dense codewords.
                self.put(0, 1);
                for _ in 0..if vector { 4 } else { 1 } {
                    self.put(if vector { 1 } else { 0 }, 5);
                }
                self.put(if vector { 1 } else { 0 }, 4);
                if vector {
                    self.put(minimum, 32);
                    self.put(delta, 32);
                    self.put((width - 1) as u32, 4);
                    self.put(0, 1);
                    for &value in values {
                        self.put(value, width);
                    }
                }
            }
        }
        for order in [1, 2, 3] {
            let mut identification = b"\x01vorbis".to_vec();
            identification.extend(0u32.to_le_bytes());
            identification.push(1);
            identification.extend(48000u32.to_le_bytes());
            identification.extend([0; 12]);
            identification.extend([0x66, 1]);
            let mut comment = b"\x03vorbis".to_vec();
            comment.extend([0; 8]);
            comment.push(1);
            let mut bits = Bits(Vec::new(), 0);
            bits.put(2, 8);
            bits.book(2, true, 0, (787 << 21) | 1, &[1, 3], 2);
            bits.book(1, false, 0, 0, &[], 0);
            bits.book(
                2,
                true,
                0x80000000 | (788 << 21) | 1,
                (789 << 21) | 1,
                &[0, 1],
                1,
            );
            bits.put(0, 6);
            bits.put(0, 16); // Time transform.
            bits.put(0, 6);
            bits.put(0, 16); // Floor zero.
            bits.put(order, 8);
            bits.put(48000, 16);
            bits.put(32, 16);
            bits.put(6, 6);
            bits.put(60, 8);
            bits.put(0, 4);
            bits.put(0, 8);
            bits.put(0, 6);
            bits.put(1, 16); // Residue one.
            bits.put(0, 24);
            bits.put(32, 24);
            bits.put(7, 24);
            bits.put(0, 6);
            bits.put(1, 8);
            bits.put(1, 3);
            bits.put(0, 1);
            bits.put(2, 8);
            bits.put(0, 6);
            bits.put(0, 16); // Mapping zero, one submap.
            bits.put(0, 1);
            bits.put(0, 1);
            bits.put(0, 2);
            bits.put(0, 8);
            bits.put(0, 8);
            bits.put(0, 8);
            bits.put(0, 6);
            bits.put(0, 1); // One short mode.
            bits.put(0, 16);
            bits.put(0, 16);
            bits.put(0, 8);
            bits.put(1, 1);
            let mut setup = b"\x05vorbis".to_vec();
            setup.extend(bits.0);
            let mut private = vec![2, identification.len() as u8, comment.len() as u8];
            private.extend(identification);
            private.extend(comment);
            private.extend(setup);
            let mut packet = Bits(Vec::new(), 0);
            packet.put(0, 1);
            packet.put(2, 6);
            packet.put(0, 1);
            packet.put(1, 2);
            if order == 3 {
                packet.put(1, 2);
            }
            for _ in 0..4 {
                packet.put(0, 1);
                for _ in 0..4 {
                    packet.put(2, 2);
                }
            }
            let mut track = element(&[0xd7], &[2]);
            track.extend(element(&[0x83], &[2]));
            track.extend(element(&[0x86], b"A_VORBIS"));
            track.extend(element(&[0x63, 0xa2], &private));
            let mut audio = element(&[0xb5], &48000f64.to_be_bytes());
            audio.extend(element(&[0x9f], &[1]));
            track.extend(element(&[0xe1], &audio));
            let mut segment = element(
                &[0x15, 0x49, 0xa9, 0x66],
                &element(&[0x2a, 0xd7, 0xb1], &[3, 0xe8]),
            );
            segment.extend(element(
                &[0x16, 0x54, 0xae, 0x6b],
                &element(&[0xae], &track),
            ));
            let mut cluster = element(&[0xe7], &[0]);
            for timestamp in [0i16, 667, 1333, 2000] {
                let mut block = vec![0x82];
                block.extend(timestamp.to_be_bytes());
                block.push(0x80);
                block.extend(&packet.0);
                cluster.extend(element(&[0xa3], &block));
            }
            segment.extend(element(&[0x1f, 0x43, 0xb6, 0x75], &cluster));
            let mut bytes = element(&[0x1a, 0x45, 0xdf, 0xa3], &element(&[0x42, 0x82], b"webm"));
            bytes.extend(element(&[0x18, 0x53, 0x80, 0x67], &segment));
            if let Some(path) = std::env::var_os("WEBMEDIA_VORBIS_FLOOR_ZERO_FIXTURE") {
                let path = std::path::PathBuf::from(path);
                let path = if order == 2 {
                    path
                } else {
                    path.with_extension(format!("order{order}.webm"))
                };
                std::fs::write(path, &bytes).unwrap();
            }
            let mut stream = WebmStream::new();
            let packets = stream.push_packets(&bytes).unwrap();
            stream.finish().unwrap();
            let headers = crate::audio::vorbis::Headers::from_webm(&private).unwrap();
            let mut decoder =
                crate::audio::vorbis::decoder::Decoder::new(&headers, 1024, 1024).unwrap();
            let mut pcm = Vec::new();
            for packet in packets {
                if let WebmPacket::Audio(block) = packet {
                    if let Some(audio) = decoder.decode_block(&block).unwrap() {
                        pcm.extend(audio.samples);
                    }
                }
            }
            assert_eq!(pcm.len(), 96);
            assert!(pcm.iter().all(|value| value.is_finite()));
            assert!(pcm.iter().any(|value| value.abs() > 1e-8));
            for chunk in [1, 7, 64, bytes.len()] {
                let actual = collect_media(&bytes, chunk);
                assert_eq!(actual.len(), 3);
                let mut samples = Vec::new();
                for (index, sample) in actual.into_iter().enumerate() {
                    let MediaSample::Audio { timestamp_ns, samples: audio } = sample else { panic!("missing Vorbis PCM") };
                    assert_eq!(timestamp_ns, [667_000, 1_333_000, 2_000_000][index]);
                    assert_eq!(audio.sample_rate, 48000);
                    assert_eq!(audio.channels, 1);
                    samples.extend(audio.samples);
                }
                assert_eq!(samples, pcm);
            }
            let video_packet = &include_bytes!("../../tests/fixtures/vp8-keyframe.ivf")[44..];
            let video_header = FrameHeader::parse(video_packet).unwrap();
            let mut video_track = element(&[0xd7], &[1]);
            video_track.extend(element(&[0x83], &[1]));
            video_track.extend(element(&[0x86], b"V_VP8"));
            let mut video = element(&[0xb0], &video_header.width.unwrap().to_be_bytes());
            video.extend(element(&[0xba], &video_header.height.unwrap().to_be_bytes()));
            video_track.extend(element(&[0xe0], &video));
            let mut tracks = element(&[0xae], &track);
            tracks.extend(element(&[0xae], &video_track));
            let mut mixed_cluster = element(&[0xe7], &[0]);
            for timestamp in [0i16, 667, 1333, 2000] {
                let mut audio_block = vec![0x82];
                audio_block.extend(timestamp.to_be_bytes());
                audio_block.push(0x80);
                audio_block.extend(&packet.0);
                mixed_cluster.extend(element(&[0xa3], &audio_block));
                let mut video_block = vec![0x81];
                video_block.extend(timestamp.to_be_bytes());
                video_block.push(0x80);
                video_block.extend(video_packet);
                mixed_cluster.extend(element(&[0xa3], &video_block));
            }
            let mut mixed_segment = element(&[0x15, 0x49, 0xa9, 0x66],
                &element(&[0x2a, 0xd7, 0xb1], &[3, 0xe8]));
            mixed_segment.extend(element(&[0x16, 0x54, 0xae, 0x6b], &tracks));
            mixed_segment.extend(element(&[0x1f, 0x43, 0xb6, 0x75], &mixed_cluster));
            let mut mixed_bytes = element(&[0x1a, 0x45, 0xdf, 0xa3],
                &element(&[0x42, 0x82], b"webm"));
            mixed_bytes.extend(element(&[0x18, 0x53, 0x80, 0x67], &mixed_segment));
            let video_pixels = Vp8Decoder::new().decode(video_packet).unwrap().rgba();
            for chunk_size in [1, 13, mixed_bytes.len()] {
                let mixed = collect_media(&mixed_bytes, chunk_size);
                assert_eq!(mixed.len(), 7);
                let mut audio_samples = Vec::new();
                for (index, sample) in mixed.iter().enumerate() {
                    match sample {
                        MediaSample::Video(frame) => {
                            assert_eq!(index % 2, 0);
                            assert_eq!(*frame.rgba, video_pixels);
                        }
                        MediaSample::Audio { samples, .. } => {
                            assert_eq!(index % 2, 1);
                            audio_samples.extend_from_slice(&samples.samples);
                        }
                        MediaSample::AudioError(error) => panic!("mixed audio failed: {error:?}"),
                    }
                }
                assert_eq!(audio_samples, pcm);
            }
            for padding in [0i64, 41667, -41667, i64::MAX, i64::MIN] {
                let mut laced = vec![0x82, 0, 0, 0x82, 3];
                laced.extend([packet.0.len() as u8; 3]);
                for _ in 0..4 { laced.extend(&packet.0); }
                let mut group = element(&[0xa1], &laced);
                group.extend(element(&[0x75, 0xa2], &padding.to_be_bytes()));
                let mut laced_cluster = element(&[0xe7], &[0]);
                laced_cluster.extend(element(&[0xa0], &group));
                let mut delayed_track = track.clone();
                delayed_track.extend(element(&[0x56, 0xaa], &500_000u32.to_be_bytes()));
                let mut segment = element(&[0x16, 0x54, 0xae, 0x6b], &element(&[0xae], &delayed_track));
                segment.extend(element(&[0x1f, 0x43, 0xb6, 0x75], &laced_cluster));
                let mut bytes = element(&[0x1a, 0x45, 0xdf, 0xa3], &element(&[0x42, 0x82], b"webm"));
                bytes.extend(element(&[0x18, 0x53, 0x80, 0x67], &segment));
                for chunk in [1, 7, bytes.len()] {
                    let actual = collect_media(&bytes, chunk);
                    if padding.unsigned_abs() > 41667 {
                        assert!(matches!(actual.as_slice(), [MediaSample::AudioError(MediaDecodeError::InvalidData(_))]));
                        continue;
                    }
                    assert_eq!(actual.len(), 3);
                    let mut samples = Vec::new();
                    for (index, sample) in actual.iter().enumerate() {
                        let MediaSample::Audio { timestamp_ns, samples: audio } = sample else { panic!("missing laced PCM") };
                        assert_eq!(*timestamp_ns, match index {
                            0 => if padding < 0 { 208_333 } else { 166_666 },
                            1 => 833_333,
                            _ => 1_500_000,
                        });
                        assert!(audio.samples.len() <= 32);
                        samples.extend_from_slice(&audio.samples);
                    }
                    let expected = if padding < 0 { &pcm[2..] } else if padding > 0 { &pcm[..94] } else { &pcm };
                    assert_eq!(samples, expected);
                }
            }
        }
    }

    fn audio_sample(codec: WebmAudioCodec, grouped: bool, padding_first: bool) -> Vec<u8> {
        let mut track = element(&[0xd7], &[2]);
        track.extend(element(&[0x83], &[2]));
        track.extend(element(
            &[0x86],
            match codec {
                WebmAudioCodec::Opus => b"A_OPUS" as &[u8],
                WebmAudioCodec::Vorbis => b"A_VORBIS",
            },
        ));
        track.extend(element(&[0x63, 0xa2], b"codec configuration"));
        track.extend(element(&[0x56, 0xaa], &[0x63, 0x2e, 0xa0]));
        track.extend(element(&[0x56, 0xbb], &[0x04, 0xc4, 0xb4, 0x00]));
        let mut audio = element(&[0xb5], &48000f64.to_be_bytes());
        audio.extend(element(&[0x9f], &[2]));
        track.extend(element(&[0xe1], &audio));
        let tracks = element(&[0x16, 0x54, 0xae, 0x6b], &element(&[0xae], &track));
        let mut cluster = element(&[0xe7], &[0]);
        // Audio may begin before timestamp zero; CodecDelay and pre-skip
        // are applied by synthesis rather than discarding compressed input.
        let block = [0x82, 0xff, 0xff, 0x80, 10, 20, 30];
        if grouped {
            let padding = element(&[0x75, 0xa2], &(-1000i64).to_be_bytes());
            let mut group = Vec::new();
            if padding_first {
                group.extend(&padding);
            }
            group.extend(element(&[0xa1], &block));
            if !padding_first {
                group.extend(&padding);
            }
            cluster.extend(element(&[0xa0], &group));
        } else {
            cluster.extend(element(&[0xa3], &block));
        }
        let mut segment = tracks;
        segment.extend(element(&[0x1f, 0x43, 0xb6, 0x75], &cluster));
        let mut bytes = element(&[0x1a, 0x45, 0xdf, 0xa3], &element(&[0x42, 0x82], b"webm"));
        bytes.extend(element(&[0x18, 0x53, 0x80, 0x67], &segment));
        bytes
    }

    #[test]
    fn streams_both_audio_codecs_across_every_chunk_boundary() {
        for codec in [WebmAudioCodec::Opus, WebmAudioCodec::Vorbis] {
            for (grouped, padding_first) in [(false, false), (true, false), (true, true)] {
                let bytes = audio_sample(codec, grouped, padding_first);
                for size in 1..=bytes.len() {
                    let mut stream = WebmStream::new();
                    let mut packets = Vec::new();
                    for chunk in bytes.chunks(size) {
                        packets.extend(stream.push_packets(chunk).unwrap());
                    }
                    stream.finish().unwrap();
                    let track = stream.audio_track().unwrap();
                    assert_eq!(track.codec, codec);
                    assert_eq!(track.number, 2);
                    assert_eq!(track.sampling_frequency, 48000.0);
                    assert_eq!(track.channels, 2);
                    assert_eq!(track.codec_private, b"codec configuration");
                    assert_eq!(track.codec_delay_ns, 6_500_000);
                    assert_eq!(track.seek_pre_roll_ns, 80_000_000);
                    assert_eq!(
                        packets,
                        vec![WebmPacket::Audio(WebmAudioBlock {
                            timestamp_ns: -1_000_000,
                            packets: vec![vec![10, 20, 30]],
                            discard_padding_ns: grouped.then_some(-1000),
                        })]
                    );
                }
            }
        }
    }

    #[test]
    fn video_only_demux_does_not_select_or_copy_audio() {
        for codec in [WebmAudioCodec::Opus, WebmAudioCodec::Vorbis] {
            let mut stream = WebmVideoStream::new();
            assert!(
                stream
                    .push(&audio_sample(codec, true, false))
                    .unwrap()
                    .is_empty()
            );
            assert!(stream.audio_track().is_none());
            assert_eq!(stream.finish(), Err(MediaDecodeError::Unsupported));
            assert!(stream.push_packets(&[]).is_err());
        }
    }

    #[test]
    fn rejects_invalid_audio_metadata_and_truncated_groups() {
        let mut bytes = audio_sample(WebmAudioCodec::Opus, false, false);
        let offset = bytes
            .windows(8)
            .position(|window| window == 48000f64.to_be_bytes())
            .unwrap();
        for rate in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            bytes[offset..offset + 8].copy_from_slice(&rate.to_be_bytes());
            assert!(WebmStream::new().push_packets(&bytes).is_err());
        }
        let bytes = audio_sample(WebmAudioCodec::Vorbis, true, false);
        let mut stream = WebmStream::new();
        assert!(
            stream
                .push_packets(&bytes[..bytes.len() - 1])
                .unwrap()
                .is_empty()
        );
        assert!(stream.finish().is_err());
        let mut stream = WebmStream::new();
        stream
            .push_packets(&audio_sample(WebmAudioCodec::Opus, false, false))
            .unwrap();
        stream.time_code_scale = u64::MAX;
        stream.cluster_timestamp = u64::MAX;
        assert!(stream.audio_block(&[0x82, 0, 1, 0, 0]).is_err());
    }

    #[test]
    fn demuxes_supplied_audio_clip_without_dropping_packets() {
        let Ok(path) = std::env::var("WEBMEDIA_AUDIO_SAMPLE") else {
            return;
        };
        let mut source = std::fs::File::open(path).unwrap();
        let mut stream = WebmStream::new();
        let mut bytes = [0u8; 4093];
        let mut blocks = 0usize;
        let mut packets = 0usize;
        let mut total_bytes = 0usize;
        let mut video_packets = 0usize;
        let mut opus_samples = 0usize;
        let mut opus_configurations = std::collections::BTreeMap::<u8, usize>::new();
        let mut vorbis_setup = None;
        let mut vorbis_decoder = None;
        let mut vorbis_pcm = Vec::new();
        let mut vorbis_modes = std::collections::BTreeMap::<usize, usize>::new();
        let mut vorbis_nonzero_floors = 0usize;
        let mut first_timestamp = None;
        let mut last_timestamp = None;
        loop {
            let count = std::io::Read::read(&mut source, &mut bytes).unwrap();
            if count == 0 {
                break;
            }
            for packet in stream.push_packets(&bytes[..count]).unwrap() {
                if let WebmPacket::Audio(block) = packet {
                    assert!(last_timestamp.is_none_or(|last| block.timestamp_ns >= last));
                    first_timestamp.get_or_insert(block.timestamp_ns);
                    last_timestamp = Some(block.timestamp_ns);
                    blocks += 1;
                    packets += block.packets.len();
                    total_bytes += block.packets.iter().map(Vec::len).sum::<usize>();
                    if stream.audio_track().unwrap().codec == WebmAudioCodec::Opus {
                        for bytes in &block.packets {
                            let packet = crate::audio::opus::Packet::parse(bytes).unwrap();
                            if let Some(layout) = packet.celt_layout().unwrap() {
                                assert_eq!(layout.samples(), usize::from(packet.frame_samples_48khz));
                                for band in layout.bands() {
                                    assert!(layout.band(band).unwrap().end <= layout.samples());
                                }
                            }
                            opus_samples += packet.samples_48khz();
                            *opus_configurations.entry(packet.configuration).or_default() += 1;
                        }
                    } else {
                        if vorbis_setup.is_none() {
                            let headers = crate::audio::vorbis::Headers::from_webm(
                                &stream.audio_track().unwrap().codec_private,
                            )
                            .unwrap();
                            vorbis_setup = Some(headers.decode_setup(1 << 20, 1 << 20).unwrap());
                            vorbis_decoder = Some(
                                crate::audio::vorbis::decoder::Decoder::new(
                                    &headers,
                                    1 << 20,
                                    1 << 20,
                                )
                                .unwrap(),
                            );
                        }
                        let setup = vorbis_setup.as_ref().unwrap();
                        if let Some(audio) = vorbis_decoder
                            .as_mut()
                            .unwrap()
                            .decode_block(&block)
                            .unwrap()
                        {
                            assert!(audio.samples.iter().all(|sample| sample.is_finite()));
                            vorbis_pcm.extend(audio.samples);
                        }
                        for bytes in &block.packets {
                            let mut bits = crate::audio::vorbis::entropy::PacketBits::new(bytes);
                            let header = setup.packet_header(&mut bits).unwrap().unwrap();
                            *vorbis_modes.entry(header.mode).or_default() += 1;
                            let mapping = &setup.mappings[setup.modes[header.mode].mapping];
                            for &mux in &mapping.mux {
                                if let crate::audio::vorbis::setup::Floor::One(floor) =
                                    &setup.floors[mapping.submaps[mux].floor]
                                {
                                    if floor
                                        .decode_amplitudes(&mut bits, &setup.codebooks)
                                        .unwrap()
                                        .is_some()
                                    {
                                        vorbis_nonzero_floors += 1;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    video_packets += 1;
                }
            }
        }
        stream.finish().unwrap();
        assert!(packets > 0);
        let track = stream.audio_track().unwrap();
        match track.codec {
            WebmAudioCodec::Opus => {
                let header =
                    crate::audio::opus::IdentificationHeader::parse(&track.codec_private).unwrap();
                assert_eq!(u16::from(header.channels), track.channels);
                assert_eq!(
                    track.codec_delay_ns,
                    u64::from(header.pre_skip) * 1_000_000_000 / 48000
                );
            }
            WebmAudioCodec::Vorbis => {
                let headers =
                    crate::audio::vorbis::Headers::from_webm(&track.codec_private).unwrap();
                assert_eq!(u16::from(headers.identification.channels), track.channels);
                assert_eq!(
                    f64::from(headers.identification.sample_rate),
                    track.sampling_frequency
                );
                let (books, offset) = headers.codebooks(1 << 20, 1 << 20).unwrap();
                assert!(!books.is_empty());
                eprintln!(
                    "Vorbis codebooks={} setup_prefix_bits={offset}",
                    books.len()
                );
            }
        }
        if let Ok(expected) = std::env::var("WEBMEDIA_AUDIO_EXPECTED_PACKETS") {
            assert_eq!(packets, expected.parse::<usize>().unwrap());
        }
        if let Ok(expected) = std::env::var("WEBMEDIA_VIDEO_EXPECTED_PACKETS") {
            assert_eq!(video_packets, expected.parse::<usize>().unwrap());
        }
        eprintln!(
            "audio codec={:?} rate={} channels={} blocks={blocks} packets={packets} bytes={total_bytes} first={first_timestamp:?} last={last_timestamp:?}",
            track.codec, track.sampling_frequency, track.channels
        );
        if opus_samples != 0 {
            eprintln!("Opus samples={opus_samples} configurations={opus_configurations:?}");
        }
        if let Some(setup) = vorbis_setup {
            eprintln!(
                "Vorbis floors={} residues={} mappings={} modes={vorbis_modes:?} nonzero_channel_floors={vorbis_nonzero_floors}",
                setup.floors.len(),
                setup.residues.len(),
                setup.mappings.len()
            );
            eprintln!("Vorbis interleaved_pcm_samples={}", vorbis_pcm.len());
            assert!(!vorbis_pcm.is_empty());
            if let Ok(path) = std::env::var("WEBMEDIA_AUDIO_REFERENCE_F32") {
                let reference = std::fs::read(path).unwrap();
                assert_eq!(reference.len(), vorbis_pcm.len() * 4);
                let mut maximum = 0.0f32;
                let mut square_error = 0.0f64;
                for (bytes, &sample) in reference.chunks_exact(4).zip(&vorbis_pcm) {
                    let expected = f32::from_le_bytes(bytes.try_into().unwrap());
                    let error = (expected - sample).abs();
                    maximum = maximum.max(error);
                    square_error += f64::from(error).powi(2);
                }
                eprintln!(
                    "Vorbis PCM reference max_error={maximum} rms_error={}",
                    (square_error / vorbis_pcm.len() as f64).sqrt()
                );
                assert!(maximum < 1e-5);
            }
        }
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
            let offset = bytes
                .windows(5)
                .position(|window| window == b"V_VP8")
                .unwrap();
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
