//! VP8 keyframe reconstruction before the in-loop deblocking pass.

use super::backend::MediaDecodeError;
use super::vp8::{FrameHeader, KeyFrameLayout};
use super::vp8_filter::{filter_frame, FilterMacroblock};
use super::vp8_inter::InterState;
use super::vp8_predict::Plane;
use super::vp8_residue::ResidueDecoder;
use super::vp8_residue::ResidualMacroblock;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub(super) enum YuvMatrix {
    Bt601,
    Bt709,
}

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
        self.rgba_with_matrix(YuvMatrix::Bt601)
    }

    pub(super) fn rgba_with_matrix(&self, matrix: YuvMatrix) -> Vec<u8> {
        match matrix {
            YuvMatrix::Bt601 => self.rgba_row_pairs::<true>(),
            YuvMatrix::Bt709 => self.rgba_row_pairs_matrix::<true, true>(),
        }
    }

    fn rgba_row_pairs<const VECTOR: bool>(&self) -> Vec<u8> {
        self.rgba_row_pairs_matrix::<VECTOR, false>()
    }

    fn rgba_row_pairs_matrix<const VECTOR: bool, const BT709: bool>(&self) -> Vec<u8> {
        let mut rgba = vec![0; self.width * self.height * 4];
        if self.width == 0 { return rgba; }
        for (pair, output) in rgba.chunks_mut(self.width * 8).enumerate() {
            let y = pair * 2;
            let luma = &self.y.pixels[y * self.y.width..][..self.width];
            let next_luma = if y + 1 < self.height {
                &self.y.pixels[(y + 1) * self.y.width..][..self.width]
            } else { &[] };
            let u = &self.u.pixels[(y / 2) * self.u.width..][..self.width.div_ceil(2)];
            let v = &self.v.pixels[(y / 2) * self.v.width..][..self.width.div_ceil(2)];
            let (target, next_target) = output.split_at_mut(self.width * 4);
            #[cfg(target_arch = "aarch64")]
            let processed = if VECTOR {
                if next_luma.is_empty() {
                    unsafe { rgba_row_neon_matrix::<BT709>(luma, u, v, target) }
                } else {
                    unsafe { rgba_two_rows_neon_matrix::<BT709>(luma, next_luma, u, v, target, next_target) }
                }
            } else { 0 };
            #[cfg(not(target_arch = "aarch64"))]
            let processed = 0;
            rgba_two_rows_scalar::<BT709>(luma, next_luma, u, v, target, next_target, processed);
        }
        rgba
    }
}

fn rgba_two_rows_scalar<const BT709: bool>(first: &[u8], second: &[u8], cb: &[u8], cr: &[u8],
    first_target: &mut [u8], second_target: &mut [u8], start: usize)
{
    #[inline]
    fn write(samples: &[u8], target: &mut [u8], red: i32, green: i32, blue: i32) {
        for (pixel, &sample) in target.chunks_exact_mut(4).zip(samples) {
            let luma = 298 * (i32::from(sample) - 16);
            pixel[0] = ((luma + red) >> 8).clamp(0, 255) as u8;
            pixel[1] = ((luma + green) >> 8).clamp(0, 255) as u8;
            pixel[2] = ((luma + blue) >> 8).clamp(0, 255) as u8;
            pixel[3] = 255;
        }
    }
    for x in (start..first.len()).step_by(2) {
        let cb = i32::from(cb[x / 2]) - 128;
        let cr = i32::from(cr[x / 2]) - 128;
        let red = (if BT709 { 459 } else { 409 }) * cr + 128;
        let green = -(if BT709 { 55 } else { 100 }) * cb
            - (if BT709 { 136 } else { 208 }) * cr + 128;
        let blue = (if BT709 { 541 } else { 516 }) * cb + 128;
        let end = (x + 2).min(first.len());
        write(&first[x..end], &mut first_target[x * 4..end * 4], red, green, blue);
        if !second.is_empty() {
            write(&second[x..end], &mut second_target[x * 4..end * 4], red, green, blue);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[cfg(test)]
#[target_feature(enable = "neon")]
unsafe fn rgba_two_rows_neon(first: &[u8], second: &[u8], cb: &[u8], cr: &[u8],
    first_target: &mut [u8], second_target: &mut [u8]) -> usize
{
    unsafe { rgba_two_rows_neon_matrix::<false>(first, second, cb, cr, first_target, second_target) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rgba_two_rows_neon_matrix<const BT709: bool>(first: &[u8], second: &[u8], cb: &[u8], cr: &[u8],
    first_target: &mut [u8], second_target: &mut [u8]) -> usize
{
    use std::arch::aarch64::*;
    let chroma = |cb, cr| {
        let cb = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cb)), vdupq_n_s16(128));
        let cr = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cr)), vdupq_n_s16(128));
        let terms = |cb, cr| (
            vmull_n_s16(cr, if BT709 { 459 } else { 409 }),
            vsubq_s32(vnegq_s32(vmull_n_s16(cb, if BT709 { 55 } else { 100 })), vmull_n_s16(cr, if BT709 { 136 } else { 208 })),
            vmull_n_s16(cb, if BT709 { 541 } else { 516 }),
        );
        [terms(vget_low_s16(cb), vget_low_s16(cr)),
            terms(vget_high_s16(cb), vget_high_s16(cr))]
    };
    let convert = |luma, chroma: &[(int32x4_t, int32x4_t, int32x4_t); 2]| {
        let luma = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(luma)), vdupq_n_s16(16));
        let four = |luma, terms: &(int32x4_t, int32x4_t, int32x4_t)| {
            let base = vmlal_n_s16(vdupq_n_s32(128), luma, 298);
            let rounded = |value| vqmovun_s32(vshrq_n_s32::<8>(vaddq_s32(base, value)));
            (rounded(terms.0), rounded(terms.1), rounded(terms.2))
        };
        let lo = four(vget_low_s16(luma), &chroma[0]);
        let hi = four(vget_high_s16(luma), &chroma[1]);
        uint8x8x4_t(vqmovn_u16(vcombine_u16(lo.0, hi.0)),
            vqmovn_u16(vcombine_u16(lo.1, hi.1)),
            vqmovn_u16(vcombine_u16(lo.2, hi.2)), vdup_n_u8(255))
    };
    let processed = first.len() / 16 * 16;
    for x in (0..processed).step_by(16) {
        let a = unsafe { vld1q_u8(first[x..x + 16].as_ptr()) };
        let b = unsafe { vld1q_u8(second[x..x + 16].as_ptr()) };
        let u = unsafe { vld1_u8(cb[x / 2..x / 2 + 8].as_ptr()) };
        let v = unsafe { vld1_u8(cr[x / 2..x / 2 + 8].as_ptr()) };
        // Each chroma sample serves a 2x2 luma block; retain its wide products
        // for both output rows without changing rounding or saturation.
        let lo = chroma(vzip1_u8(u, u), vzip1_u8(v, v));
        unsafe {
            vst4_u8(first_target[x * 4..x * 4 + 32].as_mut_ptr(), convert(vget_low_u8(a), &lo));
            vst4_u8(second_target[x * 4..x * 4 + 32].as_mut_ptr(), convert(vget_low_u8(b), &lo));
        }
        let hi = chroma(vzip2_u8(u, u), vzip2_u8(v, v));
        unsafe {
            vst4_u8(first_target[x * 4 + 32..x * 4 + 64].as_mut_ptr(), convert(vget_high_u8(a), &hi));
            vst4_u8(second_target[x * 4 + 32..x * 4 + 64].as_mut_ptr(), convert(vget_high_u8(b), &hi));
        }
    }
    processed
}

#[cfg(target_arch = "aarch64")]
#[cfg(test)]
#[target_feature(enable = "neon")]
unsafe fn rgba_row_neon(luma: &[u8], cb: &[u8], cr: &[u8], target: &mut [u8]) -> usize {
    unsafe { rgba_row_neon_matrix::<false>(luma, cb, cr, target) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rgba_row_neon_matrix<const BT709: bool>(luma: &[u8], cb: &[u8], cr: &[u8], target: &mut [u8]) -> usize {
    use std::arch::aarch64::*;
    let convert_eight = |luma, cb, cr| {
        let luma = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(luma)), vdupq_n_s16(16));
        let cb = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cb)), vdupq_n_s16(128));
        let cr = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cr)), vdupq_n_s16(128));
        let convert_four = |luma, cb, cr| {
            let base = vmull_n_s16(luma, 298);
            let red = vmlal_n_s16(base, cr, if BT709 { 459 } else { 409 });
            let green = vsubq_s32(vsubq_s32(base, vmull_n_s16(cb, if BT709 { 55 } else { 100 })), vmull_n_s16(cr, if BT709 { 136 } else { 208 }));
            let blue = vmlal_n_s16(base, cb, if BT709 { 541 } else { 516 });
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

    #[test]
    fn bt709_vector_conversion_matches_scalar_with_odd_rows_and_tails() {
        for width in 1..=65 {
            for height in 1..=5 {
                let mut frame = YuvKeyFrame::new(width, height);
                for (plane, seed) in [(&mut frame.y, 17), (&mut frame.u, 43), (&mut frame.v, 71)] {
                    for (index, sample) in plane.pixels.iter_mut().enumerate() {
                        *sample = (index * seed + 13) as u8;
                    }
                }
                assert_eq!(frame.rgba_with_matrix(YuvMatrix::Bt709), frame.rgba_row_pairs_matrix::<false, true>());
            }
        }
    }

    #[test]
    fn bt709_conversion_matches_independent_primary_color_equations() {
        let mut frame = YuvKeyFrame::new(1, 1);
        let kr = 0.2126;
        let kb = 0.0722;
        let kg = 1.0 - kr - kb;
        for y in [0, 1, 16, 32, 80, 100, 235, 255] {
            for cb in [0, 16, 128, 240, 255] {
                for cr in [0, 16, 128, 240, 255] {
                    frame.y.pixels[0] = y;
                    frame.u.pixels[0] = cb;
                    frame.v.pixels[0] = cr;
                    let luma = (f64::from(y) - 16.0) * 255.0 / 219.0;
                    let u = (f64::from(cb) - 128.0) * 255.0 / 224.0;
                    let v = (f64::from(cr) - 128.0) * 255.0 / 224.0;
                    let expected = [
                        luma + 2.0 * (1.0 - kr) * v,
                        luma - 2.0 * kb * (1.0 - kb) / kg * u - 2.0 * kr * (1.0 - kr) / kg * v,
                        luma + 2.0 * (1.0 - kb) * u,
                    ];
                    let actual = frame.rgba_with_matrix(YuvMatrix::Bt709);
                    for (value, expected) in actual.iter().zip(expected) {
                        assert!((f64::from(*value) - expected.clamp(0.0, 255.0).round()).abs() <= 1.0);
                    }
                    assert_eq!(actual[3], 255);
                }
            }
        }
    }

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
            assert_eq!(frame.rgba_row_pairs::<false>(), scalar_rgba(&frame));
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
            let second = luma.map(|sample| sample.wrapping_add(113));
            let mut paired = [91u8; 72];
            let mut next = [91u8; 72];
            unsafe { assert_eq!(rgba_two_rows_neon(&luma, &second, &u, &v,
                &mut paired[4..68], &mut next[4..68]), 16); }
            assert_eq!(paired, actual);
            let mut expected_next = [91u8; 72];
            rgba_two_rows_scalar::<false>(&second, &[], &u, &v, &mut expected_next[4..68], &mut [], 0);
            assert_eq!(next, expected_next);
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
        fn single_rows<const VECTOR: bool>(frame: &YuvKeyFrame) -> Vec<u8> {
            let mut rgba = vec![0; frame.width * frame.height * 4];
            for (y, target) in rgba.chunks_mut(frame.width * 4).enumerate() {
                let luma = &frame.y.pixels[y * frame.y.width..][..frame.width];
                let u = &frame.u.pixels[(y / 2) * frame.u.width..][..frame.width.div_ceil(2)];
                let v = &frame.v.pixels[(y / 2) * frame.v.width..][..frame.width.div_ceil(2)];
                #[cfg(target_arch = "aarch64")]
                let processed = if VECTOR { unsafe { rgba_row_neon(luma, u, v, target) } } else { 0 };
                #[cfg(not(target_arch = "aarch64"))]
                let processed = 0;
                rgba_two_rows_scalar::<false>(luma, &[], u, v, target, &mut [], processed);
            }
            rgba
        }
        fn measure(frame: &YuvKeyFrame, convert: fn(&YuvKeyFrame) -> Vec<u8>) -> std::time::Duration {
            let start = std::time::Instant::now();
            for _ in 0..30 { std::hint::black_box(convert(std::hint::black_box(frame))); }
            start.elapsed()
        }
        let frame = patterned_frame(1920, 1080);
        for (label, single, paired) in [
            ("scalar", single_rows::<false> as fn(&YuvKeyFrame) -> Vec<u8>, YuvKeyFrame::rgba_row_pairs::<false> as fn(&YuvKeyFrame) -> Vec<u8>),
            ("vector", single_rows::<true> as fn(&YuvKeyFrame) -> Vec<u8>, YuvKeyFrame::rgba_row_pairs::<true> as fn(&YuvKeyFrame) -> Vec<u8>),
        ] {
            assert_eq!(single(&frame), paired(&frame));
            for trial in 0..4 {
                let (old, new) = if trial % 2 == 0 {
                    (measure(&frame, single), measure(&frame, paired))
                } else {
                    let new = measure(&frame, paired);
                    (measure(&frame, single), new)
                };
                eprintln!("VP8/VP9 RGBA {label}: single={old:?} paired={new:?}");
            }
        }
    }
}
