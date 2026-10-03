//! VP9 in-loop deblocking (spec section 8.8). Filtered planes become references.

use super::vp8_keyframe::YuvKeyFrame;
use super::vp8_predict::Plane;
use super::vp9::KeyframeLayout;
use super::vp9_tile::TileBlock;

#[derive(Clone, Copy, Default)]
struct Cell {
    width: u8,
    height: u8,
    tx_size: u8,
    segment: u8,
    reference: u8,
    mode: u8,
    skip: bool,
}

#[derive(Clone, Copy, Default)]
struct Strength {
    limit: u8,
    blimit: u8,
    threshold: u8,
}

pub(super) struct FilterGrid {
    cols: usize,
    rows: usize,
    cells: Vec<Cell>,
}

impl FilterGrid {
    pub(super) fn new(width: usize, height: usize) -> Self {
        let cols = width.div_ceil(8);
        let rows = height.div_ceil(8);
        Self { cols, rows, cells: vec![Cell::default(); cols * rows] }
    }

    pub(super) fn record(&mut self, entry: &TileBlock) {
        let block = &entry.block;
        let cell = Cell {
            width: entry.width as u8, height: entry.height as u8,
            tx_size: block.tx_size, segment: block.segment_id,
            reference: block.inter.map_or(0, |inter| inter.reference),
            mode: block.inter.map_or(0, |inter| inter.mode), skip: block.skip,
        };
        for row in entry.y / 8..((entry.y + usize::from(cell.height.max(8))) / 8).min(self.rows) {
            for col in entry.x / 8..((entry.x + usize::from(cell.width.max(8))) / 8).min(self.cols) {
                self.cells[row * self.cols + col] = cell;
            }
        }
    }

    pub(super) fn apply(&self, frame: &mut YuvKeyFrame, layout: &KeyframeLayout<'_>) {
        if layout.loop_filter_level == 0 { return; }
        let shift = layout.loop_filter_level >> 5;
        let sharpness = i32::from(layout.loop_filter_sharpness);
        let sharp_shift = if sharpness > 4 { 2 } else if sharpness > 0 { 1 } else { 0 };
        let strength_table: [[[Strength; 2]; 4]; 8] = std::array::from_fn(|segment| {
            std::array::from_fn(|reference| std::array::from_fn(|mode| {
                let mut level = match layout.segment_alt_l[segment] {
                    Some(value) if layout.segmentation_abs_or_delta_update => i32::from(value),
                    Some(value) => i32::from(layout.loop_filter_level) + i32::from(value),
                    None => i32::from(layout.loop_filter_level),
                };
                level = level.clamp(0, 63);
                if layout.loop_filter.delta_enabled {
                    level += i32::from(layout.loop_filter.reference_deltas[reference]) << shift;
                    if reference != 0 {
                        level += i32::from(layout.loop_filter.mode_deltas[mode]) << shift;
                    }
                }
                let level = level.clamp(0, 63);
                if level == 0 { return Strength::default(); }
                let limit = if sharpness > 0 {
                    (level >> sharp_shift).clamp(1, 9 - sharpness)
                } else { (level >> sharp_shift).max(1) };
                Strength { limit: limit as u8, blimit: (2 * (level + 2) + limit) as u8,
                    threshold: (level >> 4) as u8 }
            }))
        });
        let strengths: Vec<Strength> = self.cells.iter().map(|cell| {
            if cell.width == 0 { Strength::default() }
            else { strength_table[cell.segment as usize][cell.reference as usize][usize::from(cell.mode != 2)] }
        }).collect();
        if strengths.iter().all(|strength| strength.blimit == 0) { return; }
        for sb_row in (0..self.rows).step_by(8) {
            for sb_col in (0..self.cols).step_by(8) {
                for plane_index in 0..3 {
                    let chroma = plane_index != 0;
                    let plane = match plane_index {
                        0 => &mut frame.y, 1 => &mut frame.u, _ => &mut frame.v,
                    };
                    for horizontal in [false, true] {
                        let sub = usize::from(chroma);
                        for edge in 0..(16 >> sub) {
                            for span in (0..(64 >> sub)).step_by(8) {
                                let (base_x, base_y) = if horizontal {
                                    (sb_col * 8 + (span << sub), sb_row * 8 + edge * (4 << sub))
                                } else {
                                    (sb_col * 8 + edge * (4 << sub), sb_row * 8 + (span << sub))
                                };
                                if base_x >= self.cols * 8 || base_y >= self.rows * 8
                                    || (!horizontal && base_x == 0) || (horizontal && base_y == 0) {
                                    continue;
                                }
                                let mi_col = ((base_x / 8) >> sub) << sub;
                                let mi_row = ((base_y / 8) >> sub) << sub;
                                if mi_col >= self.cols || mi_row >= self.rows { continue; }
                                let index = mi_row * self.cols + mi_col;
                                let cell = self.cells[index];
                                let strength = strengths[index];
                                if cell.width == 0 || strength.blimit == 0 { continue; }
                                let dimension = usize::from((if horizontal { cell.height } else { cell.width })
                                    .max(if chroma { 16 } else { 4 }));
                                let block_edge = (if horizontal { base_y } else { base_x }) % dimension == 0;
                                let tx_size = if chroma {
                                    cell.tx_size.min((cell.width.max(8).min(cell.height.max(8)) / 8).trailing_zeros() as u8)
                                } else { cell.tx_size };
                                let transform_edge = edge % (1 << tx_size) == 0;
                                if !block_edge && !(transform_edge && (cell.reference == 0 || !cell.skip)) {
                                    continue;
                                }
                                let size = if tx_size == 0 && edge % 8 == 0 { 8 }
                                    else { (4usize << tx_size).min(16) };
                                #[cfg(target_arch = "aarch64")]
                                if horizontal && base_x + (7 << sub) < self.cols * 8 {
                                    let span_size = if chroma && size == 16
                                        && base_y / 8 == self.rows - 1 { 8 } else { size };
                                    if filter_horizontal_span(plane, base_x >> sub, base_y >> sub,
                                        span_size, strength.limit, strength.blimit, strength.threshold) {
                                        continue;
                                    }
                                }
                                for offset in 0..8 {
                                    let (x, y) = if horizontal {
                                        (base_x + (offset << sub), base_y)
                                    } else {
                                        (base_x, base_y + (offset << sub))
                                    };
                                    if x >= self.cols * 8 || y >= self.rows * 8 { break; }
                                    let size = if chroma && size == 16 && (
                                        (!horizontal && x / 8 == self.cols - 1)
                                        || (horizontal && y / 8 == self.rows - 1)) { 8 } else { size };
                                    filter_edge(plane, x >> sub, y >> sub, horizontal,
                                        size, i32::from(strength.limit), i32::from(strength.blimit),
                                        i32::from(strength.threshold));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn filter_horizontal_span(plane: &mut Plane, x: usize, y: usize,
    size: usize, limit: u8, blimit: u8, threshold: u8,
) -> bool {
    let taps = if size == 16 { 8 } else { 4 };
    if plane.width == 0 || x.checked_add(8).is_none_or(|end| end > plane.width)
        || y < taps || y.checked_add(taps).is_none_or(|end| end > plane.pixels.len() / plane.width) {
        return false;
    }
    // Every vector load is contained in a checked eight-column row span.
    unsafe { filter_horizontal_span_neon(plane, x, y, size, limit, blimit, threshold); }
    true
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn filter_horizontal_span_neon(plane: &mut Plane, x: usize, y: usize,
    size: usize, limit: u8, blimit: u8, threshold: u8,
) {
    use std::arch::aarch64::*;
    let zero = vdupq_n_u16(0);
    let mut p = [zero; 8];
    let mut q = [zero; 8];
    let pos = y * plane.width + x;
    for i in 0..4 {
        let start_p = pos - (i + 1) * plane.width;
        let start_q = pos + i * plane.width;
        p[i] = vmovl_u8(unsafe { vld1_u8(plane.pixels[start_p..start_p + 8].as_ptr()) });
        q[i] = vmovl_u8(unsafe { vld1_u8(plane.pixels[start_q..start_q + 8].as_ptr()) });
    }
    let mut adjacent = zero;
    for i in 1..4 {
        adjacent = vmaxq_u16(adjacent, vmaxq_u16(vabdq_u16(p[i], p[i - 1]),
            vabdq_u16(q[i], q[i - 1])));
    }
    let across = vaddq_u16(vshlq_n_u16::<1>(vabdq_u16(p[0], q[0])),
        vshrq_n_u16::<1>(vabdq_u16(p[1], q[1])));
    let mask = vandq_u16(vcleq_u16(adjacent, vdupq_n_u16(u16::from(limit))),
        vcleq_u16(across, vdupq_n_u16(u16::from(blimit))));
    if vmaxvq_u16(mask) == 0 { return; }
    let hev = vcgtq_u16(vmaxq_u16(vabdq_u16(p[1], p[0]), vabdq_u16(q[1], q[0])),
        vdupq_n_u16(u16::from(threshold)));
    let clamp = |value| vminq_s16(vmaxq_s16(value, vdupq_n_s16(-128)), vdupq_n_s16(127));
    let signed = |value| vreinterpretq_s16_u16(value);
    let clipped = |value| vmovl_u8(vqmovun_s16(value));
    let filter = vbslq_s16(hev, clamp(vsubq_s16(signed(p[1]), signed(q[1]))), vdupq_n_s16(0));
    let filter = clamp(vaddq_s16(filter, vmulq_n_s16(vsubq_s16(signed(q[0]), signed(p[0])), 3)));
    let f1 = vshrq_n_s16::<3>(clamp(vaddq_s16(filter, vdupq_n_s16(4))));
    let f2 = vshrq_n_s16::<3>(clamp(vaddq_s16(filter, vdupq_n_s16(3))));
    let half = vshrq_n_s16::<1>(vaddq_s16(f1, vdupq_n_s16(1)));
    let mut out_p = p;
    let mut out_q = q;
    out_p[0] = vbslq_u16(mask, clipped(vaddq_s16(signed(p[0]), f2)), p[0]);
    out_q[0] = vbslq_u16(mask, clipped(vsubq_s16(signed(q[0]), f1)), q[0]);
    let smooth = vbicq_u16(mask, hev);
    out_p[1] = vbslq_u16(smooth, clipped(vaddq_s16(signed(p[1]), half)), p[1]);
    out_q[1] = vbslq_u16(smooth, clipped(vsubq_s16(signed(q[1]), half)), q[1]);
    let mut stored = 2;
    if size >= 8 {
        let mut difference = zero;
        for i in 1..4 {
            difference = vmaxq_u16(difference, vmaxq_u16(vabdq_u16(p[i], p[0]), vabdq_u16(q[i], q[0])));
        }
        let flat = vandq_u16(mask, vcleq_u16(difference, vdupq_n_u16(1)));
        if vmaxvq_u16(flat) != 0 {
            let filtered = unsafe { flat_filter_neon::<3, 3>(&p, &q) };
            for i in 0..3 {
                out_p[i] = vbslq_u16(flat, filtered[2 - i], out_p[i]);
                out_q[i] = vbslq_u16(flat, filtered[3 + i], out_q[i]);
            }
            stored = 3;
            if size == 16 {
                difference = zero;
                for i in 4..8 {
                    let start_p = pos - (i + 1) * plane.width;
                    let start_q = pos + i * plane.width;
                    p[i] = vmovl_u8(unsafe { vld1_u8(plane.pixels[start_p..start_p + 8].as_ptr()) });
                    q[i] = vmovl_u8(unsafe { vld1_u8(plane.pixels[start_q..start_q + 8].as_ptr()) });
                    out_p[i] = p[i];
                    out_q[i] = q[i];
                    difference = vmaxq_u16(difference, vmaxq_u16(vabdq_u16(p[i], p[0]), vabdq_u16(q[i], q[0])));
                }
                let flat2 = vandq_u16(flat, vcleq_u16(difference, vdupq_n_u16(1)));
                if vmaxvq_u16(flat2) != 0 {
                    let filtered = unsafe { flat_filter_neon::<7, 4>(&p, &q) };
                    for i in 0..7 {
                        out_p[i] = vbslq_u16(flat2, filtered[6 - i], out_p[i]);
                        out_q[i] = vbslq_u16(flat2, filtered[7 + i], out_q[i]);
                    }
                    stored = 7;
                }
            }
        }
    }
    for i in 0..stored {
        let start_p = pos - (i + 1) * plane.width;
        let start_q = pos + i * plane.width;
        unsafe {
            vst1_u8(plane.pixels[start_p..start_p + 8].as_mut_ptr(), vmovn_u16(out_p[i]));
            vst1_u8(plane.pixels[start_q..start_q + 8].as_mut_ptr(), vmovn_u16(out_q[i]));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn flat_filter_neon<const N: usize, const SHIFT: i32>(
    p: &[std::arch::aarch64::uint16x8_t; 8], q: &[std::arch::aarch64::uint16x8_t; 8],
) -> [std::arch::aarch64::uint16x8_t; 14] {
    use std::arch::aarch64::*;
    let mut window = vaddq_u16(vmulq_n_u16(p[N], N as u16), q[0]);
    for &sample in &p[..N] { window = vaddq_u16(window, sample); }
    let rounding = vdupq_n_u16(1 << (SHIFT - 1));
    let mut filtered = [vdupq_n_u16(0); 14];
    for i in 0..N {
        filtered[i] = vshrq_n_u16::<SHIFT>(vaddq_u16(vaddq_u16(window, p[N - i - 1]), rounding));
        window = vaddq_u16(vsubq_u16(window, p[N]), q[i + 1]);
    }
    for i in 0..N {
        filtered[N + i] = vshrq_n_u16::<SHIFT>(vaddq_u16(vaddq_u16(window, q[i]), rounding));
        window = vaddq_u16(vsubq_u16(window, p[N - i - 1]), q[N]);
    }
    filtered
}

fn filter_edge(plane: &mut Plane, x: usize, y: usize, horizontal: bool,
    size: usize, limit: i32, blimit: i32, threshold: i32,
) {
    let step = if horizontal { plane.width } else { 1 };
    let pos = y * plane.width + x;
    let taps = if size == 16 { 8 } else { 4 };
    if x >= plane.width || pos < taps * step || pos + (taps - 1) * step >= plane.pixels.len()
        || (!horizontal && x < taps) || (!horizontal && x + taps > plane.width) {
        return;
    }
    let mut p = [0i32; 8];
    let mut q = [0i32; 8];
    for i in 0..4 {
        p[i] = i32::from(plane.pixels[pos - (i + 1) * step]);
        q[i] = i32::from(plane.pixels[pos + i * step]);
    }
    let difference = |a: i32, b: i32| (a - b).abs();
    if difference(p[3], p[2]) > limit || difference(p[2], p[1]) > limit
        || difference(p[1], p[0]) > limit || difference(q[1], q[0]) > limit
        || difference(q[2], q[1]) > limit || difference(q[3], q[2]) > limit
        || 2 * difference(p[0], q[0]) + difference(p[1], q[1]) / 2 > blimit {
        return;
    }
    let flat = size >= 8 && (1..4).all(|i| difference(p[i], p[0]) <= 1
        && difference(q[i], q[0]) <= 1);
    // Outer taps matter only after the mask and inner-flat tests have passed.
    let flat2 = if flat && size == 16 {
        for i in 4..8 {
            p[i] = i32::from(plane.pixels[pos - (i + 1) * step]);
            q[i] = i32::from(plane.pixels[pos + i * step]);
        }
        (4..8).all(|i| difference(p[i], p[0]) <= 1 && difference(q[i], q[0]) <= 1)
    } else { false };
    if flat {
        let log2_size = if flat2 { 4 } else { 3 };
        let n = (1usize << (log2_size - 1)) - 1;
        if p[0] == q[0] && p[1..=n].iter().chain(&q[1..=n]).all(|&sample| sample == p[0]) {
            return;
        }
        let filtered = flat_filter_samples(&p, &q, log2_size);
        for i in -(n as isize)..(n as isize) {
            let index = (pos as isize + i * step as isize) as usize;
            plane.pixels[index] = filtered[(i + n as isize) as usize];
        }
    } else {
        let hev = difference(p[1], p[0]) > threshold || difference(q[1], q[0]) > threshold;
        let clamp = |value: i32| value.clamp(-128, 127);
        let mut filter = if hev { clamp(p[1] - q[1]) } else { 0 };
        filter = clamp(filter + 3 * (q[0] - p[0]));
        let f1 = clamp(filter + 4) >> 3;
        let f2 = clamp(filter + 3) >> 3;
        plane.pixels[pos] = (clamp(q[0] - 128 - f1) + 128) as u8;
        plane.pixels[pos - step] = (clamp(p[0] - 128 + f2) + 128) as u8;
        if !hev {
            let half = (f1 + 1) >> 1;
            plane.pixels[pos + step] = (clamp(q[1] - 128 - half) + 128) as u8;
            plane.pixels[pos - 2 * step] = (clamp(p[1] - 128 + half) + 128) as u8;
        }
    }
}

fn flat_filter_samples(p: &[i32; 8], q: &[i32; 8], log2_size: u32) -> [u8; 14] {
    match log2_size {
        3 => flat_filter_halves::<3, 3>(p, q),
        4 => flat_filter_halves::<7, 4>(p, q),
        _ => unreachable!("VP9 flat filter size"),
    }
}

fn flat_filter_halves<const N: usize, const SHIFT: u32>(p: &[i32; 8], q: &[i32; 8]) -> [u8; 14] {
    // The negative half always removes p[N]; the positive half always adds q[N].
    let mut window = N as i32 * p[N] + p[..N].iter().sum::<i32>() + q[0];
    let rounding = 1 << (SHIFT - 1);
    let mut filtered = [0u8; 14];
    for index in 0..N {
        filtered[index] = ((window + p[N - index - 1] + rounding) >> SHIFT) as u8;
        window += q[index + 1] - p[N];
    }
    for index in 0..N {
        filtered[N + index] = ((window + q[index] + rounding) >> SHIFT) as u8;
        window += q[N] - p[N - index - 1];
    }
    filtered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn horizontal_vectors_match_scalar_edges() {
        let mut state = 73u32;
        for size in [4, 8, 16] {
            for trial in 0..2048 {
                let mut actual = Plane::new(19, 24);
                actual.pixels.fill(91);
                for lane in 0..8 {
                    for row in 0..16 {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        let random = (state >> 24) as u8;
                        actual.pixels[row * 19 + 3 + lane] = match (trial + lane) % 6 {
                            0 => random,
                            1 => 127 + (random & 1),
                            2 => if (4..12).contains(&row) { 127 + (random & 1) } else { random },
                            3 => 112 + random % 32,
                            4 => random % 5,
                            _ => 251 + random % 5,
                        };
                    }
                }
                let mut expected = actual.clone();
                let limit = [1, 3, 8, 32, 63][trial % 5];
                let blimit = [4, 16, 64, 193][trial % 4];
                let threshold = (trial % 4) as u8;
                for lane in 0..8 {
                    filter_edge(&mut expected, 3 + lane, 8, true, size,
                        i32::from(limit), i32::from(blimit), i32::from(threshold));
                }
                assert!(filter_horizontal_span(&mut actual, 3, 8, size, limit, blimit, threshold));
                assert_eq!(actual.pixels, expected.pixels, "size={size} trial={trial}");
            }
        }
        let mut plane = Plane::new(19, 24);
        plane.pixels.fill(127);
        let before = plane.pixels.clone();
        for (x, y) in [(12, 8), (3, 7), (3, 17), (usize::MAX, 8), (3, usize::MAX)] {
            assert!(!filter_horizontal_span(&mut plane, x, y, 16, 63, 193, 3));
            assert_eq!(plane.pixels, before);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore = "manual horizontal deblocking kernel timing"]
    fn benchmark_horizontal_filter() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut source = Plane::new(19, 24);
        for (index, sample) in source.pixels.iter_mut().enumerate() {
            *sample = 120 + ((index * 13 + index / 19) % 16) as u8;
        }
        let mut plane = source.clone();
        for trial in 0..5 {
            for vector in if trial % 2 == 0 { [false, true] } else { [true, false] } {
                let start = Instant::now();
                for iteration in 0..20_000 {
                    plane.pixels.copy_from_slice(black_box(&source.pixels));
                    let size = [4, 8, 16][iteration % 3];
                    if vector {
                        black_box(filter_horizontal_span(black_box(&mut plane), 3, 8, size, 32, 193, 3));
                    } else {
                        for lane in 0..8 { filter_edge(black_box(&mut plane), 3 + lane, 8, true, size, 32, 193, 3); }
                    }
                    black_box(&plane.pixels);
                }
                eprintln!("VP9 horizontal trial={trial} vector={vector} elapsed={:?}", start.elapsed());
            }
        }
    }

    fn clamped_rolling_filter(p: &[i32; 8], q: &[i32; 8], log2_size: u32) -> [u8; 14] {
        let n = (1isize << (log2_size - 1)) - 1;
        let sample = |i: isize| {
            let i = i.clamp(-(n + 1), n);
            if i < 0 { p[(-i - 1) as usize] } else { q[i as usize] }
        };
        let mut window: i32 = (-n..=n).map(|j| sample(-n + j)).sum();
        let mut filtered = [0u8; 14];
        for i in -n..n {
            filtered[(i + n) as usize] =
                ((window + sample(i) + (1 << (log2_size - 1))) >> log2_size) as u8;
            window += sample(i + n + 1) - sample(i - n);
        }
        filtered
    }

    #[test]
    #[ignore = "manual flat-edge kernel timing"]
    fn benchmark_flat_filter_halves() {
        use std::hint::black_box;
        use std::time::Instant;
        let p = [127, 128, 127, 128, 127, 128, 127, 128];
        let q = [128, 127, 128, 127, 128, 127, 128, 127];
        for trial in 0..5 {
            for specialized in if trial % 2 == 0 { [false, true] } else { [true, false] } {
                let start = Instant::now();
                for iteration in 0..50_000 {
                    let size = black_box(3 + iteration % 2);
                    let result = if specialized {
                        flat_filter_samples(black_box(&p), black_box(&q), size)
                    } else {
                        clamped_rolling_filter(black_box(&p), black_box(&q), size)
                    };
                    black_box(result);
                }
                eprintln!("VP9 flat trial={trial} specialized={specialized} elapsed={:?}", start.elapsed());
            }
        }
    }

    #[test]
    fn rolling_flat_filter_matches_specification_sum() {
        let mut state = 17u32;
        for _ in 0..4096 {
            let mut p = [0i32; 8];
            let mut q = [0i32; 8];
            for sample in p.iter_mut().chain(q.iter_mut()) {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *sample = (state >> 24) as i32;
            }
            for log2_size in [3, 4] {
                let n = (1isize << (log2_size - 1)) - 1;
                let sample = |i: isize| {
                    let i = i.clamp(-(n + 1), n);
                    if i < 0 { p[(-i - 1) as usize] } else { q[i as usize] }
                };
                let mut expected = [0u8; 14];
                for i in -n..n {
                    let total = sample(i) + (-n..=n).map(|j| sample(i + j)).sum::<i32>();
                    expected[(i + n) as usize] =
                        ((total + (1 << (log2_size - 1))) >> log2_size) as u8;
                }
                assert_eq!(flat_filter_samples(&p, &q, log2_size), expected);
                assert_eq!(clamped_rolling_filter(&p, &q, log2_size), expected);
            }
        }
    }
}
