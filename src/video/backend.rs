//! Media backend boundary.
//!
//! The DOM/control runtime is independent from codecs. Decoder implementations
//! feed metadata, audio samples, and video frames through this narrow surface.

#[derive(Clone, Debug, PartialEq)]
pub struct MediaMetadata {
    pub duration: Option<f32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AudioSamples {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VideoFrame {
    pub width: u32,
    pub height: u32,
    pub rgba: std::sync::Arc<Vec<u8>>,
    pub timestamp: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DecodedMedia {
    Audio {
        metadata: MediaMetadata,
        samples: AudioSamples,
    },
    Video {
        metadata: MediaMetadata,
        first_frame: Option<VideoFrame>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MediaDecodeError {
    Unsupported,
    InvalidData(String),
    DecodeFailed(String),
}

pub trait MediaDecoder {
    fn decode(&self, bytes: &[u8], mime: Option<&str>) -> Result<DecodedMedia, MediaDecodeError>;
}

pub trait StreamingVideoDecoder {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<VideoFrame>, MediaDecodeError>;
    fn metadata(&self) -> Option<MediaMetadata>;
    fn finish(&self) -> Result<(), MediaDecodeError>;

    fn has_buffered_samples(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MediaSample {
    Video(VideoFrame),
    Audio {
        timestamp_ns: i64,
        samples: AudioSamples,
    },
    /// A failed audio track must not stop an independently decodable video track.
    AudioError(MediaDecodeError),
}

pub trait StreamingMediaDecoder {
    fn push_media(&mut self, bytes: &[u8]) -> Result<Vec<MediaSample>, MediaDecodeError>;
    fn metadata(&self) -> Option<MediaMetadata>;
    fn finish(&self) -> Result<(), MediaDecodeError>;
    fn has_buffered_samples(&self) -> bool;
}

pub struct NullMediaDecoder;

impl MediaDecoder for NullMediaDecoder {
    fn decode(&self, _bytes: &[u8], _mime: Option<&str>) -> Result<DecodedMedia, MediaDecodeError> {
        Err(MediaDecodeError::Unsupported)
    }
}
