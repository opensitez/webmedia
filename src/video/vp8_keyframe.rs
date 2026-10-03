//! VP8 keyframe reconstruction before the in-loop deblocking pass.

use super::backend::MediaDecodeError;
use super::vp8::{FrameHeader, KeyFrameLayout};
use super::vp8_filter::{filter_frame, FilterMacroblock};
use super::vp8_predict::Plane;
use super::vp8_residue::ResidueDecoder;
use super::vp8_residue::ResidualMacroblock;

#[derive(Clone)]
pub(super) struct YuvKeyFrame {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) y: Plane,
    pub(super) u: Plane,
    pub(super) v: Plane,
}

pub(super) fn decode_keyframe(data: &[u8]) -> Result<YuvKeyFrame, MediaDecodeError> {
    let header = FrameHeader::parse(data)?;
    if !header.key_frame {
        return Err(MediaDecodeError::Unsupported);
    }
    let width = usize::from(header.width.unwrap());
    let height = usize::from(header.height.unwrap());
    let mb_width = width.div_ceil(16);
    let mb_height = height.div_ceil(16);
    let mut frame = YuvKeyFrame::new(width, height);
    let mut layout = KeyFrameLayout::parse(data)?;
    let mut residue = ResidueDecoder::new(&layout)?;
    let mut filter_settings = Vec::with_capacity(mb_width * mb_height);
    for mb_y in 0..mb_height {
        for mb_x in 0..mb_width {
            let mode = layout
                .next_macroblock_mode()?
                .ok_or_else(|| MediaDecodeError::InvalidData("missing VP8 macroblock".into()))?;
            let blocks = residue.decode(&layout, &mode, mb_x, mb_y)?;
            filter_settings.push(FilterMacroblock {
                level: layout.filter_level(mode.segment, mode.luma),
                skip_inner: mode.luma != 4 && mode.skip_coefficients,
            });
            reconstruct_intra(&mut frame, mb_x, mb_y, mode.luma, mode.chroma, &mode.subblocks, &blocks);
        }
    }
    filter_frame(
        &mut frame.y,
        &mut frame.u,
        &mut frame.v,
        mb_width,
        mb_height,
        &filter_settings,
        layout.sharpness_level,
        layout.simple_filter,
        true,
    );
    Ok(frame)
}

pub(super) fn reconstruct_intra(
    frame: &mut YuvKeyFrame,
    mb_x: usize,
    mb_y: usize,
    luma: u8,
    chroma: u8,
    subblocks: &[u8; 16],
    blocks: &ResidualMacroblock,
) {
    let x = mb_x * 16;
    let y = mb_y * 16;
    if luma == 4 {
        for block_y in 0..4 {
            for block_x in 0..4 {
                let sx = x + block_x * 4;
                let sy = y + block_y * 4;
                frame.y.predict_small(sx, sy, subblocks[block_y * 4 + block_x], x, y);
                frame.y.add_residual(sx, sy, &blocks.y[block_y * 4 + block_x]);
            }
        }
    } else {
        frame.y.predict_large(x, y, 16, luma);
        for block_y in 0..4 {
            for block_x in 0..4 {
                frame.y.add_residual(x + block_x * 4, y + block_y * 4, &blocks.y[block_y * 4 + block_x]);
            }
        }
    }
    for (plane, residuals) in [(&mut frame.u, &blocks.u), (&mut frame.v, &blocks.v)] {
        plane.predict_large(mb_x * 8, mb_y * 8, 8, chroma);
        for block_y in 0..2 {
            for block_x in 0..2 {
                plane.add_residual(
                    mb_x * 8 + block_x * 4, mb_y * 8 + block_y * 4,
                    &residuals[block_y * 2 + block_x],
                );
            }
        }
    }
}

impl YuvKeyFrame {
    pub(super) fn new(width: usize, height: usize) -> Self {
        let mb_width = width.div_ceil(16);
        Self {
            width,
            height,
            y: Plane::new(mb_width * 16, height.div_ceil(16) * 16),
            u: Plane::new(mb_width * 8, height.div_ceil(16) * 8),
            v: Plane::new(mb_width * 8, height.div_ceil(16) * 8),
        }
    }

    pub(super) fn rgba(&self) -> Vec<u8> {
        let mut rgba = vec![0; self.width * self.height * 4];
        for y in 0..self.height {
            let luma = &self.y.pixels[y * self.y.width..][..self.width];
            let u = &self.u.pixels[(y / 2) * self.u.width..][..self.width.div_ceil(2)];
            let v = &self.v.pixels[(y / 2) * self.v.width..][..self.width.div_ceil(2)];
            let target = &mut rgba[y * self.width * 4..(y + 1) * self.width * 4];
            for (((pixels, samples), &cb), &cr) in target.chunks_mut(8)
                .zip(luma.chunks(2)).zip(u).zip(v)
            {
                let cb = i32::from(cb) - 128;
                let cr = i32::from(cr) - 128;
                let red = 409 * cr + 128;
                let green = -100 * cb - 208 * cr + 128;
                let blue = 516 * cb + 128;
                for (pixel, &sample) in pixels.chunks_exact_mut(4).zip(samples) {
                    let luma = 298 * (i32::from(sample) - 16);
                    pixel[0] = ((luma + red) >> 8).clamp(0, 255) as u8;
                    pixel[1] = ((luma + green) >> 8).clamp(0, 255) as u8;
                    pixel[2] = ((luma + blue) >> 8).clamp(0, 255) as u8;
                    pixel[3] = 255;
                }
            }
        }
        rgba
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_rgba(frame: &YuvKeyFrame) -> Vec<u8> {
        let mut rgba = vec![0; frame.width * frame.height * 4];
        for y in 0..frame.height {
            for x in 0..frame.width {
                let luma = i32::from(frame.y.pixels[y * frame.y.width + x]) - 16;
                let cb = i32::from(frame.u.pixels[(y / 2) * frame.u.width + x / 2]) - 128;
                let cr = i32::from(frame.v.pixels[(y / 2) * frame.v.width + x / 2]) - 128;
                let at = (y * frame.width + x) * 4;
                rgba[at] = ((298 * luma + 409 * cr + 128) >> 8).clamp(0, 255) as u8;
                rgba[at + 1] = ((298 * luma - 100 * cb - 208 * cr + 128) >> 8).clamp(0, 255) as u8;
                rgba[at + 2] = ((298 * luma + 516 * cb + 128) >> 8).clamp(0, 255) as u8;
                rgba[at + 3] = 255;
            }
        }
        rgba
    }

    fn patterned_frame(width: usize, height: usize) -> YuvKeyFrame {
        let mut frame = YuvKeyFrame::new(width, height);
        for (plane_index, plane) in [&mut frame.y, &mut frame.u, &mut frame.v].into_iter().enumerate() {
            for (index, pixel) in plane.pixels.iter_mut().enumerate() {
                *pixel = index.wrapping_mul(73 + plane_index * 18) as u8;
            }
        }
        frame
    }

    #[test]
    fn paired_rgba_matches_scalar_for_odd_sizes_and_padded_strides() {
        for (width, height) in [(1, 1), (7, 9), (16, 16), (33, 17), (192, 108)] {
            let frame = patterned_frame(width, height);
            assert_eq!(frame.rgba(), scalar_rgba(&frame));
        }
    }

    #[test]
    #[ignore = "manual release-mode pixel conversion benchmark"]
    fn benchmark_paired_rgba_conversion() {
        let frame = patterned_frame(1920, 1080);
        for _ in 0..3 {
            let start = std::time::Instant::now();
            for _ in 0..30 { std::hint::black_box(scalar_rgba(std::hint::black_box(&frame))); }
            let scalar = start.elapsed();
            let start = std::time::Instant::now();
            for _ in 0..30 { std::hint::black_box(std::hint::black_box(&frame).rgba()); }
            eprintln!("VP8/VP9 RGBA: scalar={scalar:?} paired={:?}", start.elapsed());
        }
    }
}
