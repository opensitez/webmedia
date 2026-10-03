//! VP9 in-loop deblocking (spec section 8.8). Filtered planes become references.

use super::vp8_keyframe::YuvKeyFrame;
use super::vp8_predict::Plane;
use super::vp9::KeyframeLayout;
use super::vp9_tile::TileBlock;

#[derive(Clone, Copy, Default)]
struct Cell {
    width: usize,
    height: usize,
    tx_size: u8,
    segment: u8,
    reference: u8,
    mode: u8,
    skip: bool,
}

#[derive(Clone, Copy, Default)]
struct Strength {
    limit: i32,
    blimit: i32,
    threshold: i32,
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
            width: entry.width, height: entry.height,
            tx_size: block.tx_size, segment: block.segment_id,
            reference: block.inter.map_or(0, |inter| inter.reference),
            mode: block.inter.map_or(0, |inter| inter.mode), skip: block.skip,
        };
        for row in entry.y / 8..((entry.y + cell.height.max(8)) / 8).min(self.rows) {
            for col in entry.x / 8..((entry.x + cell.width.max(8)) / 8).min(self.cols) {
                self.cells[row * self.cols + col] = cell;
            }
        }
    }

    pub(super) fn apply(&self, frame: &mut YuvKeyFrame, layout: &KeyframeLayout<'_>) {
        if layout.loop_filter_level == 0 { return; }
        let shift = layout.loop_filter_level >> 5;
        let sharpness = i32::from(layout.loop_filter_sharpness);
        let sharp_shift = if sharpness > 4 { 2 } else if sharpness > 0 { 1 } else { 0 };
        let strengths: Vec<Strength> = self.cells.iter().map(|cell| {
            if cell.width == 0 { return Strength::default(); }
            let mut level = match layout.segment_alt_l[cell.segment as usize] {
                Some(value) if layout.segmentation_abs_or_delta_update => i32::from(value),
                Some(value) => i32::from(layout.loop_filter_level) + i32::from(value),
                None => i32::from(layout.loop_filter_level),
            };
            level = level.clamp(0, 63);
            if layout.loop_filter.delta_enabled {
                level += i32::from(layout.loop_filter.reference_deltas[cell.reference as usize]) << shift;
                if cell.reference != 0 {
                    level += i32::from(layout.loop_filter.mode_deltas[usize::from(cell.mode != 2)]) << shift;
                }
            }
            let level = level.clamp(0, 63);
            if level == 0 { return Strength::default(); }
            let limit = if sharpness > 0 {
                (level >> sharp_shift).clamp(1, 9 - sharpness)
            } else { (level >> sharp_shift).max(1) };
            Strength { limit, blimit: 2 * (level + 2) + limit, threshold: level >> 4 }
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
                                let dimension = (if horizontal { cell.height } else { cell.width })
                                    .max(if chroma { 16 } else { 4 });
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
                                        size, strength.limit, strength.blimit, strength.threshold);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
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
    for i in 0..taps {
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
    let flat2 = size == 16 && (4..8).all(|i| difference(p[i], p[0]) <= 1
        && difference(q[i], q[0]) <= 1);
    if flat {
        let log2_size = if flat2 { 4 } else { 3 };
        let n = (1usize << (log2_size - 1)) - 1;
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
    let n = (1isize << (log2_size - 1)) - 1;
    let sample = |i: isize| {
        let i = i.clamp(-(n + 1), n);
        if i < 0 { p[(-i - 1) as usize] } else { q[i as usize] }
    };
    // Adjacent filter windows differ by one entering and one leaving sample.
    let mut window: i32 = (-n..=n).map(|j| sample(-n + j)).sum();
    let mut filtered = [0u8; 14];
    for i in -n..n {
        filtered[(i + n) as usize] =
            ((window + sample(i) + (1 << (log2_size - 1))) >> log2_size) as u8;
        window += sample(i + n + 1) - sample(i - n);
    }
    filtered
}

#[cfg(test)]
mod tests {
    use super::*;

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
            }
        }
    }
}
