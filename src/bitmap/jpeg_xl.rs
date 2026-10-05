//! Stateful JPEG XL decoding using jxl-rs. Animated images yield their first frame.

use super::RasterImage;
use jxl::api::{JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat, ProcessingResult};
use std::io::Read;

pub fn is_jpeg_xl(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xff, 0x0a])
        || bytes.starts_with(b"\0\0\0\x0cJXL \r\n\x87\n")
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
    options.premultiply_output = true;
    let mut input = Input { reader, bytes: Vec::new(), consumed: 0 };
    let mut decoder = JxlDecoder::new(options);
    let mut decoder = loop {
        let mut bytes = &input.bytes[input.consumed..];
        let result = decoder.process(&mut bytes, None).map_err(|error| error.to_string())?;
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
    let len = width.checked_mul(height).and_then(|n| n.checked_mul(4))
        .filter(|&n| n <= 256 * 1024 * 1024)
        .ok_or("JPEG XL output exceeds allocation limit")?;
    let width = u32::try_from(width).map_err(|_| "JPEG XL width overflow")?;
    let height = u32::try_from(height).map_err(|_| "JPEG XL height overflow")?;
    on_dimensions(width, height);
    decoder.set_pixel_format(JxlPixelFormat::rgba8(decoder.basic_info().extra_channels.len()))
        .map_err(|error| error.to_string())?;
    let mut decoder = loop {
        let mut bytes = &input.bytes[input.consumed..];
        let result = decoder.process(&mut bytes, None).map_err(|error| error.to_string())?;
        input.consumed = input.bytes.len() - bytes.len();
        match result {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                decoder = fallback;
                input.refill()?;
            }
        }
    };
    let mut image = RasterImage { width, height, rgba: vec![0; len] };
    loop {
        let mut bytes = &input.bytes[input.consumed..];
        let mut buffers = [JxlOutputBuffer::new(&mut image.rgba, height as usize, width as usize * 4)];
        let result = decoder.process(&mut bytes, &mut buffers, None)
            .map_err(|error| error.to_string())?;
        input.consumed = input.bytes.len() - bytes.len();
        match result {
            ProcessingResult::Complete { .. } => return Ok(image),
            ProcessingResult::NeedsMoreInput { mut fallback, .. } => {
                let mut buffers = [JxlOutputBuffer::new(&mut image.rgba, height as usize, width as usize * 4)];
                if fallback.flush_pixels(&mut buffers, None).map_err(|error| error.to_string())? {
                    on_preview(image.clone());
                }
                decoder = fallback;
                input.refill()?;
            }
        }
    }
}
