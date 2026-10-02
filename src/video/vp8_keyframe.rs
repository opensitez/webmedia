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
            for x in 0..self.width {
                let luma = i32::from(self.y.pixels[y * self.y.width + x]) - 16;
                let cb = i32::from(self.u.pixels[(y / 2) * self.u.width + x / 2]) - 128;
                let cr = i32::from(self.v.pixels[(y / 2) * self.v.width + x / 2]) - 128;
                let at = (y * self.width + x) * 4;
                rgba[at] = ((298 * luma + 409 * cr + 128) >> 8).clamp(0, 255) as u8;
                rgba[at + 1] = ((298 * luma - 100 * cb - 208 * cr + 128) >> 8).clamp(0, 255) as u8;
                rgba[at + 2] = ((298 * luma + 516 * cb + 128) >> 8).clamp(0, 255) as u8;
                rgba[at + 3] = 255;
            }
        }
        rgba
    }
}
