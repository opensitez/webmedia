//! VP8 keyframe reconstruction before the in-loop deblocking pass.

use super::backend::MediaDecodeError;
use super::vp8::{FrameHeader, KeyFrameLayout};
use super::vp8_filter::{filter_frame, FilterMacroblock};
use super::vp8_inter::InterState;
use super::vp8_predict::Plane;
use super::vp8_residue::ResidueDecoder;
use super::vp8_residue::ResidualMacroblock;
use std::sync::Arc;

#[derive(Clone)]
pub(super) struct YuvKeyFrame {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) y: Plane,
    pub(super) u: Plane,
    pub(super) v: Plane,
}

#[cfg(test)]
pub(super) fn decode_keyframe(data: &[u8]) -> Result<YuvKeyFrame, MediaDecodeError> {
    decode_keyframe_with_state(data).map(|(frame, _)| frame)
}

pub(super) fn decode_keyframe_with_state(data: &[u8])
    -> Result<(YuvKeyFrame, InterState), MediaDecodeError>
{
    let header = FrameHeader::parse(data)?;
    if !header.key_frame || header.version > 3 {
        return Err(MediaDecodeError::Unsupported);
    }
    let width = usize::from(header.width.unwrap());
    let height = usize::from(header.height.unwrap());
    let mb_width = width.div_ceil(16);
    let mb_height = height.div_ceil(16);
    let mut layout = KeyFrameLayout::parse(data)?;
    let mut frame = YuvKeyFrame::new(width, height);
    #[cfg(test)]
    if std::env::var_os("WEBMEDIA_VP8_REPORT").is_some() {
        eprintln!("VP8 key filter: base={} deltas={:?} levels={:?}", layout.loop_filter_level,
            layout.filter_deltas(), (0..5).map(|mode| layout.filter_level(0, mode)).collect::<Vec<_>>());
    }
    let mut state = InterState::after_keyframe(&layout);
    let segment_map = Arc::make_mut(&mut state.segment_map);
    let mut residue = ResidueDecoder::new(&layout)?;
    let mut filter_settings = Vec::with_capacity(mb_width * mb_height);
    for mb_y in 0..mb_height {
        for mb_x in 0..mb_width {
            let mode = layout
                .next_macroblock_mode()?
                .ok_or_else(|| MediaDecodeError::InvalidData("missing VP8 macroblock".into()))?;
            let blocks = residue.decode(&layout, &mode, mb_x, mb_y)?;
            segment_map[mb_y * mb_width + mb_x] = mode.segment;
            filter_settings.push(FilterMacroblock {
                level: layout.filter_level(mode.segment, mode.luma),
                skip_inner: mode.luma != 4 && !blocks.has_coefficients,
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
    Ok((frame, state))
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
                if blocks.has_coefficients {
                    frame.y.add_residual(sx, sy, &blocks.y[block_y * 4 + block_x]);
                }
            }
        }
    } else {
        frame.y.predict_large(x, y, 16, luma);
        if blocks.has_coefficients {
            for block_y in 0..4 {
                for block_x in 0..4 {
                    frame.y.add_residual(x + block_x * 4, y + block_y * 4, &blocks.y[block_y * 4 + block_x]);
                }
            }
        }
    }
    for (plane, residuals) in [(&mut frame.u, &blocks.u), (&mut frame.v, &blocks.v)] {
        plane.predict_large(mb_x * 8, mb_y * 8, 8, chroma);
        if !blocks.has_coefficients { continue; }
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
            #[cfg(target_arch = "aarch64")]
            let processed = unsafe { rgba_row_neon(luma, u, v, target) };
            #[cfg(not(target_arch = "aarch64"))]
            let processed = 0;
            for (((pixels, samples), &cb), &cr) in target[processed * 4..].chunks_mut(8)
                .zip(luma[processed..].chunks(2)).zip(&u[processed / 2..]).zip(&v[processed / 2..])
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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rgba_row_neon(luma: &[u8], cb: &[u8], cr: &[u8], target: &mut [u8]) -> usize {
    use std::arch::aarch64::*;
    let convert_eight = |luma, cb, cr| {
        let luma = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(luma)), vdupq_n_s16(16));
        let cb = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cb)), vdupq_n_s16(128));
        let cr = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cr)), vdupq_n_s16(128));
        let convert_four = |luma, cb, cr| {
            let base = vmull_n_s16(luma, 298);
            let red = vmlal_n_s16(base, cr, 409);
            let green = vsubq_s32(vsubq_s32(base, vmull_n_s16(cb, 100)), vmull_n_s16(cr, 208));
            let blue = vmlal_n_s16(base, cb, 516);
            let rounded = |value| vqmovun_s32(vshrq_n_s32::<8>(vaddq_s32(value, vdupq_n_s32(128))));
            (rounded(red), rounded(green), rounded(blue))
        };
        let lo = convert_four(vget_low_s16(luma), vget_low_s16(cb), vget_low_s16(cr));
        let hi = convert_four(vget_high_s16(luma), vget_high_s16(cb), vget_high_s16(cr));
        uint8x8x4_t(vqmovn_u16(vcombine_u16(lo.0, hi.0)),
            vqmovn_u16(vcombine_u16(lo.1, hi.1)),
            vqmovn_u16(vcombine_u16(lo.2, hi.2)), vdup_n_u8(255))
    };
    let processed = luma.len() / 16 * 16;
    for x in (0..processed).step_by(16) {
        let y = unsafe { vld1q_u8(luma[x..x + 16].as_ptr()) };
        let u = unsafe { vld1_u8(cb[x / 2..x / 2 + 8].as_ptr()) };
        let v = unsafe { vld1_u8(cr[x / 2..x / 2 + 8].as_ptr()) };
        let lo = convert_eight(vget_low_u8(y), vzip1_u8(u, u), vzip1_u8(v, v));
        let hi = convert_eight(vget_high_u8(y), vzip2_u8(u, u), vzip2_u8(v, v));
        unsafe {
            vst4_u8(target[x * 4..x * 4 + 32].as_mut_ptr(), lo);
            vst4_u8(target[x * 4 + 32..x * 4 + 64].as_mut_ptr(), hi);
        }
    }
    processed
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
        for (width, height) in [(1, 1), (7, 9), (15, 3), (16, 16), (17, 5), (31, 9),
            (32, 7), (33, 17), (192, 108)] {
            let frame = patterned_frame(width, height);
            assert_eq!(frame.rgba(), scalar_rgba(&frame));
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn vector_rgba_rows_match_every_luma_chroma_pair_and_preserve_bounds() {
        for value in 0..65536u32 {
            let luma: [u8; 16] = std::array::from_fn(|lane| ((value + lane as u32 * 17) & 255) as u8);
            let u = [(value >> 8) as u8; 8];
            let v: [u8; 8] = std::array::from_fn(|lane| (value as u8).wrapping_add(lane as u8 * 31));
            let mut actual = [91u8; 72];
            unsafe { assert_eq!(rgba_row_neon(&luma, &u, &v, &mut actual[4..68]), 16); }
            assert!(actual[..4].iter().chain(&actual[68..]).all(|&sample| sample == 91));
            for lane in 0..16 {
                let y = 298 * (i32::from(luma[lane]) - 16);
                let cb = i32::from(u[lane / 2]) - 128;
                let cr = i32::from(v[lane / 2]) - 128;
                assert_eq!(&actual[4 + lane * 4..8 + lane * 4], &[
                    ((y + 409 * cr + 128) >> 8).clamp(0, 255) as u8,
                    ((y - 100 * cb - 208 * cr + 128) >> 8).clamp(0, 255) as u8,
                    ((y + 516 * cb + 128) >> 8).clamp(0, 255) as u8, 255,
                ]);
            }
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
