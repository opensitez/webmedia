//! Integer DCT butterflies and per-stage rounding from AV1 sections 7.13.2-3.

use super::coefficients::tx_index;
use super::syntax::Error;

const COS: [i64; 65] = [
    4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948, 3920, 3889, 3857, 3822,
    3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967,
    2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
    1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401, 301, 201, 101, 0,
];

fn cos(angle: i32) -> i64 {
    let angle = angle & 255;
    match angle {
        0..=64 => COS[angle as usize],
        65..=128 => -COS[(128 - angle) as usize],
        129..=192 => -COS[(angle - 128) as usize],
        _ => COS[(256 - angle) as usize],
    }
}
fn round(v: i64, n: u8) -> i64 {
    if n == 0 {
        v
    } else {
        (v + (1i64 << (n - 1))) >> n
    }
}
fn reverse(value: usize, n: u32) -> usize {
    value.reverse_bits() >> (usize::BITS - n)
}
fn rotation(t: &mut [i64], a: usize, b: usize, angle: i32, flip: bool, r: u8) -> Result<(), Error> {
    let x = round(t[a] * cos(angle) - t[b] * cos(angle - 64), 12);
    let y = round(t[a] * cos(angle - 64) + t[b] * cos(angle), 12);
    let bound = 1i64 << (r - 1);
    if x < -bound || x >= bound || y < -bound || y >= bound {
        return Err(Error::Invalid("DCT butterfly range"));
    }
    if flip {
        t[a] = y;
        t[b] = x;
    } else {
        t[a] = x;
        t[b] = y;
    }
    Ok(())
}
fn hadamard(t: &mut [i64], a: usize, b: usize, flip: bool, r: u8) {
    let (a, b) = if flip { (b, a) } else { (a, b) };
    let (x, y) = (t[a], t[b]);
    let bound = 1i64 << (r - 1);
    t[a] = (x + y).clamp(-bound, bound - 1);
    t[b] = (x - y).clamp(-bound, bound - 1);
}

fn inverse_1d(t: &mut [i64], r: u8) -> Result<(), Error> {
    let n = t.len().ilog2();
    let copy = t.to_vec();
    for i in 0..t.len() {
        t[i] = copy[reverse(i, n)];
    }
    if n == 6 {
        for i in 0..16 {
            rotation(t, 32 + i, 63 - i, 63 - 4 * reverse(i, 4) as i32, false, r)?;
        }
    }
    if n >= 5 {
        for i in 0..8 {
            rotation(
                t,
                16 + i,
                31 - i,
                6 + ((reverse(7 - i, 3) as i32) << 3),
                false,
                r,
            )?;
        }
    }
    if n == 6 {
        for i in 0..16 {
            hadamard(t, 32 + 2 * i, 33 + 2 * i, i & 1 != 0, r);
        }
    }
    if n >= 4 {
        for i in 0..4 {
            rotation(
                t,
                8 + i,
                15 - i,
                12 + ((reverse(3 - i, 2) as i32) << 4),
                false,
                r,
            )?;
        }
    }
    if n >= 5 {
        for i in 0..8 {
            hadamard(t, 16 + 2 * i, 17 + 2 * i, i & 1 != 0, r);
        }
    }
    if n == 6 {
        for i in 0..4 {
            for j in 0..2 {
                rotation(
                    t,
                    62 - 4 * i - j,
                    33 + 4 * i + j,
                    60 - 16 * reverse(i, 2) as i32 + 64 * j as i32,
                    true,
                    r,
                )?;
            }
        }
    }
    if n >= 3 {
        for i in 0..2 {
            rotation(t, 4 + i, 7 - i, 56 - 32 * i as i32, false, r)?;
        }
    }
    if n >= 4 {
        for i in 0..4 {
            hadamard(t, 8 + 2 * i, 9 + 2 * i, i & 1 != 0, r);
        }
    }
    if n >= 5 {
        for i in 0..2 {
            for j in 0..2 {
                rotation(
                    t,
                    30 - 4 * i - j,
                    17 + 4 * i + j,
                    24 + 64 * j as i32 + 32 * (1 - i as i32),
                    true,
                    r,
                )?;
            }
        }
    }
    if n == 6 {
        for i in 0..8 {
            for j in 0..2 {
                hadamard(t, 32 + 4 * i + j, 35 + 4 * i - j, i & 1 != 0, r);
            }
        }
    }
    for i in 0..2 {
        rotation(t, 2 * i, 2 * i + 1, 32 + 16 * i as i32, i == 0, r)?;
    }
    if n >= 3 {
        for i in 0..2 {
            hadamard(t, 4 + 2 * i, 5 + 2 * i, i != 0, r);
        }
    }
    if n >= 4 {
        for i in 0..2 {
            rotation(t, 14 - i, 9 + i, 48 + 64 * i as i32, true, r)?;
        }
    }
    if n >= 5 {
        for i in 0..4 {
            for j in 0..2 {
                hadamard(t, 16 + 4 * i + j, 19 + 4 * i - j, i & 1 != 0, r);
            }
        }
    }
    if n == 6 {
        for i in 0..2 {
            for j in 0..4 {
                rotation(
                    t,
                    61 - 8 * i - j,
                    34 + 8 * i + j,
                    56 - 32 * i as i32 + 64 * (j >> 1) as i32,
                    true,
                    r,
                )?;
            }
        }
    }
    for i in 0..2 {
        hadamard(t, i, 3 - i, false, r);
    }
    if n >= 3 {
        rotation(t, 6, 5, 32, true, r)?;
    }
    if n >= 4 {
        for i in 0..2 {
            for j in 0..2 {
                hadamard(t, 8 + 4 * i + j, 11 + 4 * i - j, i != 0, r);
            }
        }
    }
    if n >= 5 {
        for i in 0..4 {
            rotation(t, 29 - i, 18 + i, 48 + 64 * (i >> 1) as i32, true, r)?;
        }
    }
    if n == 6 {
        for i in 0..4 {
            for j in 0..4 {
                hadamard(t, 32 + 8 * i + j, 39 + 8 * i - j, i & 1 != 0, r);
            }
        }
    }
    if n >= 3 {
        for i in 0..4 {
            hadamard(t, i, 7 - i, false, r);
        }
    }
    if n >= 4 {
        for i in 0..2 {
            rotation(t, 13 - i, 10 + i, 32, true, r)?;
        }
    }
    if n >= 5 {
        for i in 0..2 {
            for j in 0..4 {
                hadamard(t, 16 + 8 * i + j, 23 + 8 * i - j, i != 0, r);
            }
        }
    }
    if n == 6 {
        for i in 0..8 {
            rotation(t, 59 - i, 36 + i, if i < 4 { 48 } else { 112 }, true, r)?;
        }
    }
    if n >= 4 {
        for i in 0..8 {
            hadamard(t, i, 15 - i, false, r);
        }
    }
    if n >= 5 {
        for i in 0..4 {
            rotation(t, 27 - i, 20 + i, 32, true, r)?;
        }
    }
    if n == 6 {
        for i in 0..8 {
            hadamard(t, 32 + i, 47 - i, false, r);
            hadamard(t, 48 + i, 63 - i, true, r);
        }
    }
    if n >= 5 {
        for i in 0..16 {
            hadamard(t, i, 31 - i, false, r);
        }
    }
    if n == 6 {
        for i in 0..8 {
            rotation(t, 55 - i, 40 + i, 32, true, r)?;
        }
        for i in 0..32 {
            hadamard(t, i, 63 - i, false, r);
        }
    }
    Ok(())
}

#[cfg(test)]
fn inverse_dct(w: usize, h: usize, coefficients: &[i32], bit_depth: u8) -> Result<Vec<i32>, Error> {
    inverse_transform(w, h, coefficients, bit_depth, 0)
}

fn adst(t: &mut [i64], r: u8) -> Result<(), Error> {
    let n = t.len();
    if n == 4 {
        let [a, b, c, d] = <[i64; 4]>::try_from(&*t).unwrap();
        let s0 = 1321 * a + 3803 * c + 2482 * d;
        let s1 = 2482 * a - 1321 * c - 3803 * d;
        let s2 = 3344 * (a - c + d);
        let s3 = 3344 * b;
        for (slot, v) in t.iter_mut().zip([s0 + s3, s1 + s3, s2, s0 + s1 - s3]) {
            *slot = round(v, 12);
        }
        let bound = 1i64 << (r - 1);
        if t.iter().any(|&v| v < -bound || v >= bound) {
            return Err(Error::Invalid("ADST4 range"));
        }
        return Ok(());
    }
    if n != 8 && n != 16 {
        return Err(Error::Unsupported("ADST dimensions"));
    }
    let copy = t.to_vec();
    for i in 0..n {
        t[i] = copy[if i & 1 != 0 { i - 1 } else { n - i - 1 }];
    }
    if n == 16 {
        for i in 0..8 {
            rotation(t, 2 * i, 2 * i + 1, 62 - 8 * i as i32, true, r)?;
        }
        for i in 0..8 {
            hadamard(t, i, 8 + i, false, r);
        }
        for i in 0..2 {
            rotation(t, 8 + 2 * i, 9 + 2 * i, 56 - 32 * i as i32, true, r)?;
            rotation(t, 13 + 2 * i, 12 + 2 * i, 8 + 32 * i as i32, true, r)?;
        }
        for i in 0..4 {
            for j in 0..2 {
                hadamard(t, 8 * j + i, 4 + 8 * j + i, false, r);
            }
        }
    } else {
        for i in 0..4 {
            rotation(t, 2 * i, 2 * i + 1, 60 - 16 * i as i32, true, r)?;
        }
        for i in 0..4 {
            hadamard(t, i, 4 + i, false, r);
        }
    }
    for j in 0..n / 8 {
        for i in 0..2 {
            rotation(
                t,
                8 * j + 4 + 3 * i,
                8 * j + 5 + i,
                48 - 32 * i as i32,
                true,
                r,
            )?;
        }
    }
    for j in 0..n / 4 {
        for i in 0..2 {
            hadamard(t, 4 * j + i, 2 + 4 * j + i, false, r);
        }
    }
    for i in 0..n / 4 {
        rotation(t, 2 + 4 * i, 3 + 4 * i, 32, true, r)?;
    }
    let copy = t.to_vec();
    for i in 0..n {
        let a = (i >> 3) & 1;
        let b = ((i >> 2) ^ (i >> 3)) & 1;
        let c = ((i >> 1) ^ (i >> 2)) & 1;
        let d = (i ^ (i >> 1)) & 1;
        let idx = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n.ilog2());
        t[i] = copy[idx] * if i & 1 != 0 { -1 } else { 1 };
    }
    Ok(())
}

fn axis(t: &mut [i64], r: u8, kind: usize) -> Result<(), Error> {
    match kind {
        0 => inverse_1d(t, r),
        1 => adst(t, r),
        2 => {
            let n = t.len();
            if ![4, 8, 16, 32].contains(&n) {
                return Err(Error::Unsupported("identity dimensions"));
            }
            for v in t {
                *v = match n {
                    4 => round(*v * 5793, 12),
                    8 => *v * 2,
                    16 => round(*v * 11586, 12),
                    _ => *v * 4,
                };
            }
            Ok(())
        }
        _ => Err(Error::Unsupported("transform axis")),
    }
}

pub(crate) fn inverse_transform(
    w: usize,
    h: usize,
    coefficients: &[i32],
    bit_depth: u8,
    tx_type: usize,
) -> Result<Vec<i32>, Error> {
    #[cfg(test)]
    let _measure = super::profile::measure(2);
    let tx = tx_index(w, h)?;
    let (row_kind, col_kind) = match tx_type {
        0 => (0, 0),
        1 => (0, 1),
        2 => (1, 0),
        3 | 6 | 7 | 8 => (1, 1),
        4 => (0, 1),
        5 => (1, 0),
        9 => (2, 2),
        10 => (2, 0),
        11 => (0, 2),
        12 | 14 => (2, 1),
        13 | 15 => (1, 2),
        _ => return Err(Error::Unsupported("inverse transform type")),
    };
    if ![8, 10, 12].contains(&bit_depth) || coefficients.len() != w * h {
        return Err(Error::Invalid("DCT input dimensions"));
    }
    let row_shift = [0, 1, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2][tx];
    let mut residual = vec![0i64; w * h];
    let clamp = 1i64 << ((bit_depth + 6).max(16) - 1);
    for y in 0..h {
        let mut row: Vec<i64> = (0..w).map(|x| i64::from(coefficients[y * w + x])).collect();
        if w.ilog2().abs_diff(h.ilog2()) == 1 {
            for v in &mut row {
                *v = round(*v * 2896, 12);
            }
        }
        axis(&mut row, bit_depth + 8, row_kind)?;
        for x in 0..w {
            residual[y * w + x] = round(row[x], row_shift).clamp(-clamp, clamp - 1);
        }
    }
    for x in 0..w {
        let mut col: Vec<i64> = (0..h).map(|y| residual[y * w + x]).collect();
        axis(&mut col, (bit_depth + 6).max(16), col_kind)?;
        for y in 0..h {
            residual[y * w + x] = round(col[y], 4);
        }
    }
    let flip_y = [4, 6, 8, 14].contains(&tx_type);
    let flip_x = [5, 6, 7, 15].contains(&tx_type);
    let mut output = vec![0; w * h];
    for y in 0..h {
        for x in 0..w {
            output[y * w + x] = residual
                [(if flip_y { h - y - 1 } else { y }) * w + if flip_x { w - x - 1 } else { x }]
                as i32;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adst_matches_independent_sine_matrix() {
        for size in [4, 8, 16] {
            for k in 0..size {
                let mut decoded = vec![0; size];
                decoded[k] = 1024;
                adst(&mut decoded, 24).unwrap();
                for x in 0..size {
                    let angle = if size == 4 {
                        std::f64::consts::PI * (2 * k + 1) as f64 * (x + 1) as f64 / 9.
                    } else {
                        std::f64::consts::PI * (2 * k + 1) as f64 * (2 * x + 1) as f64
                            / (4 * size) as f64
                    };
                    let scale = if size == 4 { (8f64 / 9.).sqrt() } else { 1. };
                    let expected = 1024. * angle.sin() * scale;
                    assert!(
                        (decoded[x] as f64 - expected).abs() < 3.,
                        "ADST{size} ({k},{x}): {} vs {expected}",
                        decoded[x]
                    );
                }
            }
        }
    }
    #[test]
    fn butterflies_match_independent_cosine_matrix() {
        let mut state = 137u32;
        for size in [4, 8, 16, 32, 64] {
            for _ in 0..32 {
                let input: Vec<i64> = (0..size)
                    .map(|_| {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        ((state >> 24) as i64) - 128
                    })
                    .collect();
                let mut decoded = input.clone();
                inverse_1d(&mut decoded, 24).unwrap();
                for x in 0..size {
                    let expected = input[0] as f64 / 2f64.sqrt()
                        + (1..size)
                            .map(|k| {
                                input[k] as f64
                                    * (std::f64::consts::PI * (2 * x + 1) as f64 * k as f64
                                        / (2 * size) as f64)
                                        .cos()
                            })
                            .sum::<f64>();
                    assert!(
                        (decoded[x] as f64 - expected).abs() <= 12.,
                        "size {size}, x {x}: {} vs {expected}",
                        decoded[x]
                    );
                }
            }
        }
    }
    #[test]
    fn full_dct_agrees_with_dc_specialization() {
        for &(w, h) in &super::super::coefficients::TX_DIMENSIONS {
            for dc in [-8192, -17, 17, 8192] {
                let mut coefficients = vec![0; w * h];
                coefficients[0] = dc;
                assert_eq!(
                    inverse_dct(w, h, &coefficients, 8).unwrap(),
                    super::super::reconstruction::inverse_dc(w, h, dc, 8, false).unwrap()
                );
            }
        }
    }
}
