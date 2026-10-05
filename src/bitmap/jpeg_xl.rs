//! Stateful JPEG XL decoding using jxl-rs. Animated images yield their first frame.

use super::{RasterImage, premultiply_rgba};
use jxl::api::{JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat, ProcessingResult};
use std::io::Read;
use std::time::{Duration, Instant};

pub fn is_jpeg_xl(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xff, 0x0a]) || bytes.starts_with(b"\0\0\0\x0cJXL \r\n\x87\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    const SAMPLE: &[u8] = include_bytes!("../../tests/fixtures/jxl-alpha.jxl");

    struct Chunks<'a> {
        bytes: &'a [u8],
        chunk: usize,
        read: &'a Cell<usize>,
    }

    impl Read for Chunks<'_> {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            let count = out.len().min(self.chunk).min(self.bytes.len());
            out[..count].copy_from_slice(&self.bytes[..count]);
            self.bytes = &self.bytes[count..];
            self.read.set(self.read.get() + count);
            Ok(count)
        }
    }

    #[test]
    fn streaming_matches_complete_decode_and_reports_dimensions_early() {
        let complete = super::super::decode_raster(SAMPLE).unwrap();
        let original =
            super::super::decode_raster(include_bytes!("../../tests/fixtures/jxl-alpha.png"))
                .unwrap();
        assert_eq!(
            (complete.width, complete.height),
            (original.width, original.height)
        );
        for (index, (&actual, &expected)) in complete.rgba.iter().zip(&original.rgba).enumerate() {
            assert_eq!(actual, expected, "sample {index}");
        }
        for chunk in [1, 7, 31, 4096] {
            let read = Cell::new(0);
            let mut dimensions = None;
            let decoded = decode_stream(
                Chunks {
                    bytes: SAMPLE,
                    chunk,
                    read: &read,
                },
                |width, height| {
                    assert_eq!((width, height), (complete.width, complete.height));
                    dimensions = Some(read.get());
                },
                |preview| {
                    assert_eq!(preview.rgba.len(), complete.rgba.len());
                    assert!(read.get() < SAMPLE.len());
                },
            )
            .unwrap();
            assert_eq!(decoded, complete);
            if chunk == 1 {
                assert!(dimensions.unwrap() < SAMPLE.len());
            }
        }
        assert!(complete.rgba.chunks_exact(4).any(|pixel| pixel[3] < 255));
        assert!(
            complete
                .rgba
                .chunks_exact(4)
                .all(|pixel| pixel[..3].iter().all(|&c| c <= pixel[3]))
        );
    }

    #[test]
    fn truncated_stream_is_an_error() {
        for len in [0, 1, 2, SAMPLE.len() / 2] {
            assert!(decode_stream(&SAMPLE[..len], |_, _| {}, |_| {}).is_err());
        }
    }

    #[test]
    fn progressive_container_publishes_pixels_before_download_finishes() {
        let bytes = include_bytes!("../../tests/fixtures/jxl-progressive.jxl");
        let read = Cell::new(0);
        let mut previews = 0;
        let decoded = decode_stream(
            Chunks {
                bytes,
                chunk: 31,
                read: &read,
            },
            |_, _| assert!(read.get() < bytes.len()),
            |_| {
                assert!(read.get() < bytes.len());
                previews += 1;
            },
        )
        .unwrap();
        assert!(previews > 0);
        assert_eq!(decoded, super::super::decode_raster(bytes).unwrap());
    }
}

struct Input<R> {
    reader: R,
    bytes: Vec<u8>,
    consumed: usize,
}

impl<R: Read> Input<R> {
    fn refill(&mut self) -> Result<(), String> {
        self.bytes.drain(..self.consumed);
        self.consumed = 0;
        let mut chunk = [0; 16 * 1024];
        let count = loop {
            match self.reader.read(&mut chunk) {
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                result => break result.map_err(|error| error.to_string())?,
            }
        };
        if count == 0 {
            return Err("truncated JPEG XL image".into());
        }
        self.bytes.extend_from_slice(&chunk[..count]);
        Ok(())
    }
}

/// Decode from a streaming reader without seeking or waiting for end-of-file.
/// Dimensions are reported once; previews are emitted only when pixels change.
pub fn decode_stream(
    reader: impl Read,
    mut on_dimensions: impl FnMut(u32, u32),
    mut on_preview: impl FnMut(RasterImage),
) -> Result<RasterImage, String> {
    let mut options = JxlDecoderOptions::default();
    options.sample_limit = Some(64 * 1024 * 1024);
    let mut input = Input {
        reader,
        bytes: Vec::with_capacity(16 * 1024),
        consumed: 0,
    };
    let mut decoder = JxlDecoder::new(options);
    let mut decoder = loop {
        let mut bytes = &input.bytes[input.consumed..];
        let result = decoder
            .process(&mut bytes, None)
            .map_err(|error| error.to_string())?;
        input.consumed = input.bytes.len() - bytes.len();
        match result {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                decoder = fallback;
                input.refill()?;
            }
        }
    };
    let (width, height) = decoder.basic_info().size;
    let len = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(4))
        .filter(|&n| n <= 256 * 1024 * 1024)
        .ok_or("JPEG XL output exceeds allocation limit")?;
    let width = u32::try_from(width).map_err(|_| "JPEG XL width overflow")?;
    let height = u32::try_from(height).map_err(|_| "JPEG XL height overflow")?;
    on_dimensions(width, height);
    decoder
        .set_pixel_format(JxlPixelFormat::rgba8(
            decoder.basic_info().extra_channels.len(),
        ))
        .map_err(|error| error.to_string())?;
    let mut decoder = loop {
        let mut bytes = &input.bytes[input.consumed..];
        let result = decoder
            .process(&mut bytes, None)
            .map_err(|error| error.to_string())?;
        input.consumed = input.bytes.len() - bytes.len();
        match result {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                decoder = fallback;
                input.refill()?;
            }
        }
    };
    let mut image = RasterImage {
        width,
        height,
        rgba: vec![0; len],
    };
    let mut last_preview: Option<Instant> = None;
    loop {
        let mut bytes = &input.bytes[input.consumed..];
        let mut buffers = [JxlOutputBuffer::new(
            &mut image.rgba,
            height as usize,
            width as usize * 4,
        )];
        let result = decoder
            .process(&mut bytes, &mut buffers, None)
            .map_err(|error| error.to_string())?;
        input.consumed = input.bytes.len() - bytes.len();
        match result {
            ProcessingResult::Complete { .. } => {
                premultiply_rgba(&mut image.rgba);
                return Ok(image);
            }
            ProcessingResult::NeedsMoreInput { mut fallback, .. } => {
                let mut buffers = [JxlOutputBuffer::new(
                    &mut image.rgba,
                    height as usize,
                    width as usize * 4,
                )];
                if last_preview.is_none_or(|last| last.elapsed() >= Duration::from_millis(32))
                    && fallback
                        .flush_pixels(&mut buffers, None)
                        .map_err(|error| error.to_string())?
                {
                    let mut preview = image.clone();
                    premultiply_rgba(&mut preview.rgba);
                    on_preview(preview);
                    last_preview = Some(Instant::now());
                }
                decoder = fallback;
                input.refill()?;
            }
        }
    }
}
