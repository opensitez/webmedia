//! Media backend boundary.
//!
//! The DOM/control runtime is independent from codecs. Decoder implementations
//! feed metadata, audio samples, and video frames through this narrow surface.

#[derive(Clone, Debug, PartialEq)]
pub struct MediaMetadata {
    /// Display extent, separate from coded pixel dimensions.
    pub presentation_size: Option<(f64, f64)>,
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
    /// Display extent, separate from the tightly packed RGBA buffer dimensions.
    pub presentation_size: Option<(f64, f64)>,
    pub width: u32,
    pub height: u32,
    pub rgba: std::sync::Arc<Vec<u8>>,
    pub timestamp: f32,
}

fn presentation_dimensions(size: Option<(f64, f64)>) -> Option<(u32, u32)> {
    let (width, height) = size?;
    if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0
        || width > f64::from(u32::MAX) || height > f64::from(u32::MAX)
    {
        return None;
    }
    Some((width.ceil() as u32, height.ceil() as u32))
}

impl VideoFrame {
    pub fn display_dimensions(&self) -> (u32, u32) {
        presentation_dimensions(self.presentation_size).unwrap_or((self.width, self.height))
    }

    #[cfg(feature = "acceleration")]
    pub fn surface(&self) -> Result<crate::RgbaSurface, crate::InvalidRgbaSurface> {
        crate::RgbaSurface::new(self.rgba.clone(), self.width, self.height)
            .map(|surface| surface.with_presentation_size(presentation_dimensions(self.presentation_size)))
    }
}

impl MediaMetadata {
    pub fn display_dimensions(&self) -> Option<(u32, u32)> {
        presentation_dimensions(self.presentation_size).or_else(||
            self.width.zip(self.height).filter(|&(width, height)| width > 0 && height > 0))
    }
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

#[cfg(test)]
mod presentation_tests {
    use super::*;

    #[cfg(feature = "acceleration")]
    #[test]
    fn video_surface_shares_frame_pixels_and_preserves_display_extent() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<crate::RgbaSurface>();
        let frame = VideoFrame { presentation_size: Some((5.0 / 3.0, 2.5)),
            width: 1, height: 2, rgba: std::sync::Arc::new(vec![0; 8]), timestamp: 1. }; 
        let surface = frame.surface().unwrap();
        assert!(std::sync::Arc::ptr_eq(surface.pixels(), &frame.rgba));
        assert_eq!(surface.dimensions(), (1, 2));
        assert_eq!(surface.display_dimensions(), (2, 3));
        let mut invalid = frame;
        invalid.width = u32::MAX;
        assert!(invalid.surface().is_err());
    }

    #[test]
    fn presentation_extent_rounds_without_changing_coded_size_and_rejects_invalid_values() {
        let mut frame = VideoFrame {
            presentation_size: Some((5.0 / 3.0, 2.5)), width: 1, height: 2,
            rgba: std::sync::Arc::new(vec![0; 8]), timestamp: 0.0,
        };
        assert_eq!(frame.display_dimensions(), (2, 3));
        assert_eq!((frame.width, frame.height, frame.rgba.len()), (1, 2, 8));
        for size in [(f64::NAN, 2.0), (1.0, f64::INFINITY), (0.0, 1.0),
            (-1.0, 1.0), (f64::from(u32::MAX) + 1.0, 1.0)]
        {
            frame.presentation_size = Some(size);
            assert_eq!(frame.display_dimensions(), (1, 2));
        }
        let metadata = MediaMetadata {
            presentation_size: Some((5.0 / 3.0, 2.5)), width: Some(1), height: Some(2),
            duration: None, sample_rate: None, channels: None,
        };
        assert_eq!(metadata.display_dimensions(), Some((2, 3)));
    }
}

impl MediaDecoder for NullMediaDecoder {
    fn decode(&self, _bytes: &[u8], _mime: Option<&str>) -> Result<DecodedMedia, MediaDecodeError> {
        Err(MediaDecodeError::Unsupported)
    }
}
