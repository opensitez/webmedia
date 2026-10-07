//! Incremental decoder for uncompressed YUV4MPEG2 video streams.

use super::backend::{MediaDecodeError, MediaMetadata, StreamingVideoDecoder, VideoFrame};

const MAX_PIXELS: u64 = 8 * 1024 * 1024;
const MAX_LINE: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Chroma {
    C420,
    C422,
    C444,
    Mono,
}

#[derive(Debug, Clone, Copy)]
struct Header {
    width: usize,
    height: usize,
    fps_num: u32,
    fps_den: u32,
    chroma: Chroma,
    frame_bytes: usize,
}

#[derive(Default)]
pub struct Y4mStream {
    pending: Vec<u8>,
    header: Option<Header>,
    frame_index: u64,
    in_frame: bool,
}

impl Y4mStream {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn metadata(&self) -> Option<MediaMetadata> {
        let header = self.header?;
        Some(MediaMetadata {
            presentation_size: None,
            duration: None,
            width: Some(header.width as u32),
            height: Some(header.height as u32),
            sample_rate: None,
            channels: None,
        })
    }

    pub fn push(&mut self, input: &[u8]) -> Result<Vec<VideoFrame>, String> {
        let mut frames = Vec::new();
        for chunk in input.chunks(16 * 1024) {
            self.pending.extend_from_slice(chunk);
            self.consume(&mut frames)?;
        }
        Ok(frames)
    }

    pub fn finish(&self) -> Result<(), String> {
        if self.header.is_some() && !self.in_frame && self.pending.is_empty() {
            Ok(())
        } else {
            Err("incomplete Y4M stream".to_string())
        }
    }

    fn consume(&mut self, frames: &mut Vec<VideoFrame>) -> Result<(), String> {
        loop {
            if self.header.is_none() {
                let Some(line) = self.take_line()? else {
                    return Ok(());
                };
                self.header = Some(parse_header(&line)?);
            }
            if !self.in_frame {
                let Some(line) = self.take_line()? else {
                    return Ok(());
                };
                if !line.starts_with(b"FRAME") || line.len() > 5 && line[5] != b' ' {
                    return Err("invalid Y4M frame marker".to_string());
                }
                self.in_frame = true;
            }
            let header = self.header.unwrap();
            if self.pending.len() < header.frame_bytes {
                return Ok(());
            }
            let rgba = convert_frame(&self.pending[..header.frame_bytes], header);
            self.pending.drain(..header.frame_bytes);
            let timestamp = self.frame_index as f64 * header.fps_den as f64 / header.fps_num as f64;
            frames.push(VideoFrame {
                presentation_size: None,
                width: header.width as u32,
                height: header.height as u32,
                rgba: std::sync::Arc::new(rgba),
                timestamp: timestamp as f32,
            });
            self.frame_index += 1;
            self.in_frame = false;
        }
    }

    fn take_line(&mut self) -> Result<Option<Vec<u8>>, String> {
        if let Some(end) = self.pending.iter().position(|byte| *byte == b'\n') {
            if end > MAX_LINE {
                return Err("Y4M header too long".to_string());
            }
            let line = self.pending[..end].to_vec();
            self.pending.drain(..=end);
            Ok(Some(line))
        } else if self.pending.len() > MAX_LINE {
            Err("Y4M header too long".to_string())
        } else {
            Ok(None)
        }
    }
}

impl StreamingVideoDecoder for Y4mStream {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<VideoFrame>, MediaDecodeError> {
        Y4mStream::push(self, bytes).map_err(MediaDecodeError::InvalidData)
    }

    fn metadata(&self) -> Option<MediaMetadata> {
        Y4mStream::metadata(self)
    }

    fn finish(&self) -> Result<(), MediaDecodeError> {
        Y4mStream::finish(self).map_err(MediaDecodeError::InvalidData)
    }
}

fn parse_header(line: &[u8]) -> Result<Header, String> {
    let text = std::str::from_utf8(line).map_err(|_| "invalid Y4M header")?;
    let Some(fields) = text.strip_prefix("YUV4MPEG2 ") else {
        return Err("invalid Y4M signature".to_string());
    };
    let (mut width, mut height, mut fps) = (None, None, None);
    let mut chroma = Chroma::C420;
    for field in fields.split_ascii_whitespace() {
        if let Some(value) = field.strip_prefix('W') {
            width = value.parse::<usize>().ok();
        } else if let Some(value) = field.strip_prefix('H') {
            height = value.parse::<usize>().ok();
        } else if let Some(value) = field.strip_prefix('F') {
            let (num, den) = value.split_once(':').ok_or("invalid Y4M frame rate")?;
            fps = Some((
                num.parse::<u32>().map_err(|_| "invalid Y4M frame rate")?,
                den.parse::<u32>().map_err(|_| "invalid Y4M frame rate")?,
            ));
        } else if let Some(value) = field.strip_prefix('C') {
            chroma = match value {
                "420" | "420jpeg" | "420mpeg2" | "420paldv" => Chroma::C420,
                "422" => Chroma::C422,
                "444" => Chroma::C444,
                "mono" => Chroma::Mono,
                _ => return Err("unsupported Y4M chroma format".to_string()),
            };
        }
    }
    let (width, height) = (
        width.ok_or("missing Y4M width")?,
        height.ok_or("missing Y4M height")?,
    );
    let (fps_num, fps_den) = fps.ok_or("missing Y4M frame rate")?;
    if width == 0
        || height == 0
        || fps_num == 0
        || fps_den == 0
        || (width as u64)
            .checked_mul(height as u64)
            .is_none_or(|pixels| pixels > MAX_PIXELS)
    {
        return Err("invalid Y4M dimensions or frame rate".to_string());
    }
    let y_size = width * height;
    let chroma_width = match chroma {
        Chroma::C444 => width,
        _ => width.div_ceil(2),
    };
    let chroma_height = match chroma {
        Chroma::C420 => height.div_ceil(2),
        _ => height,
    };
    let frame_bytes = if chroma == Chroma::Mono {
        y_size
    } else {
        y_size + 2 * chroma_width * chroma_height
    };
    Ok(Header {
        width,
        height,
        fps_num,
        fps_den,
        chroma,
        frame_bytes,
    })
}

fn convert_frame(data: &[u8], header: Header) -> Vec<u8> {
    let (width, height) = (header.width, header.height);
    let y_size = width * height;
    let chroma_width = if header.chroma == Chroma::C444 {
        width
    } else {
        width.div_ceil(2)
    };
    let chroma_height = if header.chroma == Chroma::C420 {
        height.div_ceil(2)
    } else {
        height
    };
    let chroma_size = chroma_width * chroma_height;
    #[cfg(feature = "acceleration")]
    if header.chroma == Chroma::C420 {
        use accelerate::video::{LumaPolicy, Plane, YuvMatrix, yuv420_to_rgba};
        return yuv420_to_rgba(
            width, height,
            Plane { data: &data[..y_size], stride: width },
            Plane { data: &data[y_size..y_size + chroma_size], stride: chroma_width },
            Plane { data: &data[y_size + chroma_size..y_size + 2 * chroma_size], stride: chroma_width },
            YuvMatrix::Bt601, LumaPolicy::Clamped,
        ).expect("validated Y4M planes must cover the visible frame");
    }
    let mut rgba = vec![0; y_size * 4];
    for y in 0..height {
        for x in 0..width {
            let uv_x = if header.chroma == Chroma::C444 {
                x
            } else {
                x / 2
            };
            let uv_y = if header.chroma == Chroma::C420 {
                y / 2
            } else {
                y
            };
            let uv_index = uv_y * chroma_width + uv_x;
            let luma = i32::from(data[y * width + x]) - 16;
            let (u, v) = if header.chroma == Chroma::Mono {
                (0, 0)
            } else {
                (
                    i32::from(data[y_size + uv_index]) - 128,
                    i32::from(data[y_size + chroma_size + uv_index]) - 128,
                )
            };
            let c = luma.max(0) * 298;
            let pixel = &mut rgba[(y * width + x) * 4..][..4];
            pixel[0] = ((c + 409 * v + 128) >> 8).clamp(0, 255) as u8;
            pixel[1] = ((c - 100 * u - 208 * v + 128) >> 8).clamp(0, 255) as u8;
            pixel[2] = ((c + 516 * u + 128) >> 8).clamp(0, 255) as u8;
            pixel[3] = 255;
        }
    }
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_arrive_independently_of_network_chunking() {
        let mut bytes = b"YUV4MPEG2 W2 H2 F25:1 C420jpeg\nFRAME\n".to_vec();
        bytes.extend_from_slice(&[16, 16, 16, 16, 128, 128]);
        bytes.extend_from_slice(b"FRAME\n");
        bytes.extend_from_slice(&[235, 235, 235, 235, 128, 128]);
        for chunk_size in [1, 3, 17, bytes.len()] {
            let mut stream = Y4mStream::new();
            let mut frames = Vec::new();
            for chunk in bytes.chunks(chunk_size) {
                frames.extend(stream.push(chunk).unwrap());
            }
            assert_eq!(stream.metadata().unwrap().width, Some(2));
            assert_eq!(frames.len(), 2);
            assert_eq!(&frames[0].rgba[..4], &[0, 0, 0, 255]);
            assert_eq!(&frames[1].rgba[..4], &[255, 255, 255, 255]);
            assert_eq!(frames[1].timestamp, 0.04);
            stream.finish().unwrap();
        }
    }

    #[test]
    fn rejects_truncated_and_oversized_video() {
        let mut stream = Y4mStream::new();
        assert!(
            stream
                .push(b"YUV4MPEG2 W2 H2 F25:1 C420\nFRAME\n\x10")
                .unwrap()
                .is_empty()
        );
        assert!(stream.finish().is_err());
        assert!(
            Y4mStream::new()
                .push(b"YUV4MPEG2 W99999 H99999 F25:1\n")
                .is_err()
        );
        assert!(Y4mStream::new().push(b"YUV4MPEG2 W2 H2 F0:1\n").is_err());
    }

    #[test]
    fn production_yuv420_matches_clamped_reference_for_odd_rows_and_tails() {
        for width in [1_usize, 7, 8, 15, 16, 17, 33] {
            for height in [1_usize, 3, 6] {
                let y_size = width * height;
                let chroma_width = width.div_ceil(2);
                let chroma_size = chroma_width * height.div_ceil(2);
                let data: Vec<_> = (0..y_size + 2 * chroma_size).map(|i| (i * 53 + 7) as u8).collect();
                let header = Header { width, height, fps_num: 30, fps_den: 1, chroma: Chroma::C420, frame_bytes: data.len() };
                let actual = convert_frame(&data, header);
                for y in 0..height {
                    for x in 0..width {
                        let c = (i32::from(data[y * width + x]) - 16).max(0) * 298;
                        let uv = y / 2 * chroma_width + x / 2;
                        let u = i32::from(data[y_size + uv]) - 128;
                        let v = i32::from(data[y_size + chroma_size + uv]) - 128;
                        assert_eq!(&actual[(y * width + x) * 4..][..4], &[
                            ((c + 409 * v + 128) >> 8).clamp(0, 255) as u8,
                            ((c - 100 * u - 208 * v + 128) >> 8).clamp(0, 255) as u8,
                            ((c + 516 * u + 128) >> 8).clamp(0, 255) as u8, 255,
                        ], "{width}x{height} at {x},{y}");
                    }
                }
            }
        }
    }

    #[test]
    fn odd_dimensions_and_other_chroma_formats_are_bounded() {
        for (chroma, planes) in [
            ("420", vec![16, 16, 16, 128, 128, 128, 128]),
            ("422", vec![16, 16, 16, 128, 128, 128, 128]),
            ("444", vec![16, 16, 16, 128, 128, 128, 128, 128, 128]),
            ("mono", vec![16, 16, 16]),
        ] {
            let mut stream = Y4mStream::new();
            let mut bytes = format!("YUV4MPEG2 W3 H1 F30:1 C{chroma}\nFRAME\n").into_bytes();
            bytes.extend_from_slice(&planes);
            let frames = StreamingVideoDecoder::push(&mut stream, &bytes).unwrap();
            assert_eq!(frames.len(), 1, "{chroma}");
            assert_eq!(frames[0].rgba.len(), 3 * 4);
            assert_eq!(&frames[0].rgba[..4], &[0, 0, 0, 255]);
            StreamingVideoDecoder::finish(&stream).unwrap();
        }
    }
}
