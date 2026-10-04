//! Local affine estimation and warped prediction, AV1 7.10.2.13 and 7.11.3.5-8.

use super::decoder::DecodedPlane;
use super::inter::MotionCell;
use super::syntax::Error;
use super::warp_tables::{DIV_LUT, WARPED_FILTERS};

fn rounded(v: i64, bits: u32) -> i64 {
    if bits == 0 {
        v
    } else {
        v.signum() * ((v.abs() + (1 << (bits - 1))) >> bits)
    }
}

fn divisor(d: i64) -> (u32, i64) {
    let n = d.unsigned_abs().ilog2();
    let e = d.abs() - (1_i64 << n);
    let f = if n > 8 {
        (e + (1 << (n - 9))) >> (n - 8)
    } else {
        e << (8 - n)
    };
    (n + 14, d.signum() * i64::from(DIV_LUT[f as usize]))
}

pub(crate) fn samples(
    cells: &[Option<MotionCell>],
    dims: [usize; 2],
    origin: [usize; 2],
    size: [usize; 2],
    reference: i8,
    mv: [i32; 2],
) -> Vec<[i64; 4]> {
    let at = |r: isize, c: isize| {
        if r < 0 || c < 0 || r >= dims[0] as isize || c >= dims[1] as isize {
            None
        } else {
            cells[r as usize * dims[1] + c as usize]
        }
    };
    let mut out = Vec::new();
    let mut scanned = 0;
    let mut fallback = None;
    let mut add = |r: isize, c: isize| {
        if scanned >= 8 {
            return;
        }
        let Some(cell) = at(r, c) else {
            return;
        };
        if cell.refs != [reference, -1] {
            return;
        }
        let mid_y = (r as usize & !(cell.height - 1)) * 4 + cell.height * 2 - 1;
        let mid_x = (c as usize & !(cell.width - 1)) * 4 + cell.width * 2 - 1;
        let sample = [
            mid_y as i64 * 8,
            mid_x as i64 * 8,
            mid_y as i64 * 8 + i64::from(cell.mvs[0][0]),
            mid_x as i64 * 8 + i64::from(cell.mvs[0][1]),
        ];
        if scanned == 0 {
            fallback = Some(sample);
        }
        scanned += 1;
        let threshold = (size[0].max(size[1]) * 4).clamp(16, 112) as i32;
        if (cell.mvs[0][0] - mv[0]).abs() + (cell.mvs[0][1] - mv[1]).abs() <= threshold {
            out.push(sample);
        }
    };
    let r = origin[0] as isize;
    let c = origin[1] as isize;
    let mut top_left = true;
    let mut top_right = true;
    if let Some(cell) = at(r - 1, c) {
        if size[1] <= cell.width {
            let offset = -(origin[1] as isize & (cell.width as isize - 1));
            if offset < 0 {
                top_left = false;
            }
            if offset + cell.width as isize > size[1] as isize {
                top_right = false;
            }
            add(r - 1, c);
        } else {
            let mut i = 0;
            while i < size[1].min(dims[1] - origin[1]) {
                add(r - 1, c + i as isize);
                i += at(r - 1, c + i as isize).map_or(1, |cell| cell.width.min(size[1]));
            }
        }
    }
    if let Some(cell) = at(r, c - 1) {
        if size[0] <= cell.height {
            if origin[0] & (cell.height - 1) != 0 {
                top_left = false;
            }
            add(r, c - 1);
        } else {
            let mut i = 0;
            while i < size[0].min(dims[0] - origin[0]) {
                add(r + i as isize, c - 1);
                i += at(r + i as isize, c - 1).map_or(1, |cell| cell.height.min(size[0]));
            }
        }
    }
    if top_left {
        add(r - 1, c - 1);
    }
    if top_right && size[0].max(size[1]) <= 16 {
        add(r - 1, c + size[1] as isize);
    }
    if out.is_empty() {
        if let Some(first) = fallback {
            out.push(first);
        }
    }
    out
}

pub(crate) fn estimate(
    samples: &[[i64; 4]],
    origin: [usize; 2],
    size: [usize; 2],
    mv: [i32; 2],
) -> Option<[i32; 6]> {
    let y = (origin[0] * 4 + size[0] * 2 - 1) as i64;
    let x = (origin[1] * 4 + size[1] * 2 - 1) as i64;
    let mut a = [0_i64; 3];
    let mut b = [[0_i64; 2]; 2];
    let product = |a: i64, b: i64| ((a * b) >> 2) + a + b;
    for s in samples {
        let sy = s[0] - y * 8;
        let sx = s[1] - x * 8;
        let dy = s[2] - y * 8 - i64::from(mv[0]);
        let dx = s[3] - x * 8 - i64::from(mv[1]);
        if (sx - dx).abs() >= 256 || (sy - dy).abs() >= 256 {
            continue;
        }
        a[0] += product(sx, sx) + 8;
        a[1] += product(sx, sy) + 4;
        a[2] += product(sy, sy) + 8;
        b[0][0] += product(sx, dx) + 8;
        b[0][1] += product(sy, dx) + 4;
        b[1][0] += product(sx, dy) + 4;
        b[1][1] += product(sy, dy) + 8;
    }
    let det = a[0] * a[2] - a[1] * a[1];
    if det == 0 {
        return None;
    }
    let (shift, factor) = divisor(det);
    let (shift, factor) = if shift < 16 {
        (0, factor << (16 - shift))
    } else {
        (shift - 16, factor)
    };
    let terms = [
        a[2] * b[0][0] - a[1] * b[0][1],
        -a[1] * b[0][0] + a[0] * b[0][1],
        a[2] * b[1][0] - a[1] * b[1][1],
        -a[1] * b[1][0] + a[0] * b[1][1],
    ];
    let mut p = [0; 6];
    for i in 0..4 {
        let center = if i == 0 || i == 3 { 65536 } else { 0 };
        p[i + 2] = rounded(terms[i] * factor, shift).clamp(center - 8191, center + 8191) as i32;
    }
    p[0] = (i64::from(mv[1]) * 8192 - x * i64::from(p[2] - 65536) - y * i64::from(p[3]))
        .clamp(-(1 << 23), (1 << 23) - 1) as i32;
    p[1] = (i64::from(mv[0]) * 8192 - x * i64::from(p[4]) - y * i64::from(p[5] - 65536))
        .clamp(-(1 << 23), (1 << 23) - 1) as i32;
    shear(p).map(|_| p)
}

fn shear(p: [i32; 6]) -> Option<[i64; 4]> {
    if p[2] == 0 {
        return None;
    }
    let (shift, factor) = divisor(i64::from(p[2]));
    let mut values = [
        i64::from(p[2]) - 65536,
        i64::from(p[3]),
        rounded((i64::from(p[4]) << 16) * factor, shift),
        i64::from(p[5]) - rounded(i64::from(p[3]) * i64::from(p[4]) * factor, shift) - 65536,
    ];
    for v in &mut values {
        *v = rounded((*v).clamp(-32768, 32767), 6) << 6;
    }
    if 4 * values[0].abs() + 7 * values[1].abs() >= 65536
        || 4 * values[2].abs() + 4 * values[3].abs() >= 65536
    {
        None
    } else {
        Some(values)
    }
}

pub(crate) fn predict(
    reference: &DecodedPlane,
    origin: [usize; 2],
    size: [usize; 2],
    subsampling: [bool; 2],
    bit_depth: u8,
    p: [i32; 6],
    compound: bool,
) -> Result<(Vec<i32>, u32), Error> {
    let [alpha, beta, gamma, delta] = shear(p).ok_or(Error::Invalid("invalid warped shear"))?;
    let [h, w] = size;
    let [y, x] = origin;
    let [sy, sx] = subsampling.map(u32::from);
    let round0 = if bit_depth == 12 { 5 } else { 3 };
    let round1 = if compound { 7 } else { 14 - round0 };
    let mut out = vec![0; h * w];
    for by in (0..h).step_by(8) {
        for bx in (0..w).step_by(8) {
            let src_x = ((x + bx + 4) as i64) << sx;
            let src_y = ((y + by + 4) as i64) << sy;
            let x4 = (i64::from(p[2]) * src_x + i64::from(p[3]) * src_y + i64::from(p[0])) >> sx;
            let y4 = (i64::from(p[4]) * src_x + i64::from(p[5]) * src_y + i64::from(p[1])) >> sy;
            let ix = x4 >> 16;
            let iy = y4 >> 16;
            let fx = x4 & 65535;
            let fy = y4 & 65535;
            let mut intermediate = [[0_i64; 8]; 15];
            for row in -7_i64..8 {
                for col in -4_i64..4 {
                    let index = ((fx + alpha * col + beta * row + 512) >> 10) + 64;
                    let filter = WARPED_FILTERS
                        .get(index as usize)
                        .ok_or(Error::Invalid("warped horizontal phase"))?;
                    let ry = (iy + row).clamp(0, reference.height as i64 - 1) as usize;
                    let mut sum = 0;
                    for (t, &f) in filter.iter().enumerate() {
                        let rx =
                            (ix + col - 3 + t as i64).clamp(0, reference.width as i64 - 1) as usize;
                        sum +=
                            i64::from(f) * i64::from(reference.samples[ry * reference.stride + rx]);
                    }
                    intermediate[(row + 7) as usize][(col + 4) as usize] =
                        (sum + (1 << (round0 - 1))) >> round0;
                }
            }
            for row in 0..8.min(h - by) {
                for col in 0..8.min(w - bx) {
                    let index = ((fy + gamma * (col as i64 - 4) + delta * (row as i64 - 4) + 512)
                        >> 10)
                        + 64;
                    let filter = WARPED_FILTERS
                        .get(index as usize)
                        .ok_or(Error::Invalid("warped vertical phase"))?;
                    let mut sum = 0;
                    for (t, &f) in filter.iter().enumerate() {
                        sum += i64::from(f) * intermediate[row + t][col];
                    }
                    out[(by + row) * w + bx + col] = ((sum + (1 << (round1 - 1))) >> round1) as i32;
                }
            }
        }
    }
    Ok((out, 14 - round0 - round1))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normative_warp_tables_preserve_dc() {
        for filter in WARPED_FILTERS {
            assert_eq!(filter.iter().map(|&x| i32::from(x)).sum::<i32>(), 128);
        }
        for d in [1, 2, 255, 256, 65536, -65536] {
            let (shift, factor) = divisor(d);
            assert!((rounded(d * factor, shift) - 1).abs() <= 1);
        }
    }
    #[test]
    fn identity_warp_preserves_constant_and_integer_origin() {
        let plane = DecodedPlane {
            width: 16,
            height: 16,
            stride: 16,
            samples: vec![173; 256],
        };
        for depth in [8, 10, 12] {
            let (p, post) = predict(
                &plane,
                [0, 0],
                [13, 11],
                [false, false],
                depth,
                [0, 0, 65536, 0, 0, 65536],
                false,
            )
            .unwrap();
            assert_eq!(post, 0);
            assert!(p.iter().all(|&v| v == 173));
        }
    }

    #[test]
    fn affine_estimation_anchors_the_block_center_to_its_motion_vector() {
        let origin = [4, 6];
        let size = [4, 4];
        let mv = [16, -24];
        let samples = [[8, 20], [8, 36], [24, 20], [32, 16]].map(|[y, x]| {
            [
                y * 8,
                x * 8,
                y * 8 + i64::from(mv[0]),
                x * 8 + i64::from(mv[1]),
            ]
        });
        let p = estimate(&samples, origin, size, mv).unwrap();
        let y = (origin[0] * 4 + size[0] * 2 - 1) as i64;
        let x = (origin[1] * 4 + size[1] * 2 - 1) as i64;
        assert_eq!(
            i64::from(p[0]) + x * i64::from(p[2]) + y * i64::from(p[3]),
            (x << 16) + i64::from(mv[1]) * 8192
        );
        assert_eq!(
            i64::from(p[1]) + x * i64::from(p[4]) + y * i64::from(p[5]),
            (y << 16) + i64::from(mv[0]) * 8192
        );
        assert!(shear(p).is_some());
        assert_eq!(estimate(&[], origin, size, mv), None);
    }
}
