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

const BIT_REVERSE: [[usize; 64]; 5] = {
    let mut table = [[0; 64]; 5];
    let mut size = 0;
    while size < 5 {
        let bits = size as u32 + 2;
        let mut index = 0usize;
        while index < 1 << bits {
            table[size][index] = index.reverse_bits() >> (usize::BITS - bits);
            index += 1;
        }
        size += 1;
    }
    table
};

fn inverse_1d(t: &mut [i64], r: u8) -> Result<(), Error> {
    match t.len() {
        4 => inverse_dct_axis::<4, false>(t, r),
        8 => inverse_dct_axis::<8, false>(t, r),
        16 => inverse_dct_axis::<16, false>(t, r),
        32 => inverse_dct_axis::<32, false>(t, r),
        64 => inverse_dct_axis::<64, false>(t, r),
        _ => Err(Error::Unsupported("DCT dimensions")),
    }
}

fn inverse_dct_axis<const SIZE: usize, const REFERENCE: bool>(
    t: &mut [i64],
    r: u8,
) -> Result<(), Error> {
    let size = if REFERENCE { t.len() } else { SIZE };
    let n = size.ilog2();
    let heap;
    let mut stack = [0; 64];
    let copy = if REFERENCE {
        heap = t.to_vec();
        heap.as_slice()
    } else {
        stack[..size].copy_from_slice(t);
        &stack[..size]
    };
    for i in 0..size {
        t[i] = copy[if REFERENCE {
            reverse(i, n)
        } else {
            BIT_REVERSE[n as usize - 2][i]
        }];
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
    adst_axis::<false>(t, r)
}

fn adst_axis<const REFERENCE: bool>(t: &mut [i64], r: u8) -> Result<(), Error> {
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
    let heap;
    let mut stack = [0; 16];
    let copy = if REFERENCE {
        heap = t.to_vec();
        heap.as_slice()
    } else {
        stack[..n].copy_from_slice(t);
        &stack[..n]
    };
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
    let heap;
    let mut stack = [0; 16];
    let copy = if REFERENCE {
        heap = t.to_vec();
        heap.as_slice()
    } else {
        stack[..n].copy_from_slice(t);
        &stack[..n]
    };
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

fn validate_axis(size: usize, kind: usize) -> Result<(), Error> {
    match kind {
        0 if [4, 8, 16, 32, 64].contains(&size) => Ok(()),
        0 => Err(Error::Unsupported("DCT dimensions")),
        1 if [4, 8, 16].contains(&size) => Ok(()),
        1 => Err(Error::Unsupported("ADST dimensions")),
        2 if [4, 8, 16, 32].contains(&size) => Ok(()),
        2 => Err(Error::Unsupported("identity dimensions")),
        _ => Err(Error::Unsupported("transform axis")),
    }
}

fn dct_constant(value: i64, r: u8) -> Result<i64, Error> {
    let value = round(value * COS[32], 12);
    let bound = 1i64 << (r - 1);
    if value < -bound || value >= bound {
        return Err(Error::Invalid("DCT butterfly range"));
    }
    Ok(value)
}

fn axis_sparse(t: &mut [i64], r: u8, kind: usize) -> Result<(), Error> {
    validate_axis(t.len(), kind)?;
    if kind == 0 && t[1..].iter().all(|&value| value == 0) {
        // The first nonzero rotation is angle 32. Its equal rounded outputs
        // propagate through the remaining butterflies without further sums.
        let value = dct_constant(t[0], r)?;
        t.fill(value);
        return Ok(());
    }
    if kind != 0 && t.iter().all(|&value| value == 0) {
        return Ok(());
    }
    axis(t, r, kind)
}

fn axis_dispatch<const SPARSE: bool>(t: &mut [i64], r: u8, kind: usize) -> Result<(), Error> {
    if SPARSE {
        axis_sparse(t, r, kind)
    } else {
        axis(t, r, kind)
    }
}

// Retained for allocating callers and independent oracle tests.
#[allow(dead_code)]
pub(crate) fn inverse_transform(
    w: usize,
    h: usize,
    coefficients: &[i32],
    bit_depth: u8,
    tx_type: usize,
) -> Result<Vec<i32>, Error> {
    transform_parameters(w, h, coefficients, bit_depth, tx_type)?;
    let mut output = vec![0; w * h];
    inverse_transform_into(w, h, coefficients, bit_depth, tx_type, &mut output)?;
    Ok(output)
}

/// Requires exactly `w * h` output samples; errors leave output unchanged.
pub(crate) fn inverse_transform_into(
    w: usize,
    h: usize,
    coefficients: &[i32],
    bit_depth: u8,
    tx_type: usize,
    output: &mut [i32],
) -> Result<(), Error> {
    #[cfg(test)]
    let _measure = super::profile::measure(2);
    #[cfg(test)]
    if sparse_profile::enabled() {
        return inverse_transform_fast_into::<false, false>(
            w,
            h,
            coefficients,
            bit_depth,
            tx_type,
            output,
        );
    }
    // Same-binary CPU-time ABBA favors NEON setup for small transforms only.
    // The larger butterflies dominate, and their scalar setup measured faster.
    if w <= 16 && h <= 16 {
        inverse_transform_fast_into::<true, true>(w, h, coefficients, bit_depth, tx_type, output)
    } else {
        inverse_transform_fast_into::<false, true>(w, h, coefficients, bit_depth, tx_type, output)
    }
}

fn transform_parameters(
    w: usize,
    h: usize,
    coefficients: &[i32],
    bit_depth: u8,
    tx_type: usize,
) -> Result<(usize, usize, u8), Error> {
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
    Ok((row_kind, col_kind, row_shift))
}

fn prepare_row_scalar(output: &mut [i64], input: &[i32], rectangular: bool) {
    for (output, &input) in output.iter_mut().zip(input) {
        *output = if rectangular {
            round(i64::from(input) * 2896, 12)
        } else {
            i64::from(input)
        };
    }
}

fn prepare_row<const NATIVE: bool>(output: &mut [i64], input: &[i32], rectangular: bool) {
    debug_assert_eq!(output.len(), input.len());
    #[cfg(target_arch = "aarch64")]
    if NATIVE {
        // AArch64 has baseline NEON; matching slices bound every vector access.
        unsafe { prepare_row_neon(output, input, rectangular) };
        return;
    }
    prepare_row_scalar(output, input, rectangular);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn prepare_row_neon(output: &mut [i64], input: &[i32], rectangular: bool) {
    use std::arch::aarch64::*;
    let mut index = 0;
    unsafe {
        while input.len() - index >= 4 {
            let values = vld1q_s32(input.as_ptr().add(index));
            let (low, high) = if rectangular {
                (
                    vrshrq_n_s64::<12>(vmull_n_s32(vget_low_s32(values), 2896)),
                    vrshrq_n_s64::<12>(vmull_n_s32(vget_high_s32(values), 2896)),
                )
            } else {
                (
                    vmovl_s32(vget_low_s32(values)),
                    vmovl_s32(vget_high_s32(values)),
                )
            };
            vst1q_s64(output.as_mut_ptr().add(index), low);
            vst1q_s64(output.as_mut_ptr().add(index + 2), high);
            index += 4;
        }
    }
    prepare_row_scalar(&mut output[index..], &input[index..], rectangular);
}

fn transpose_scalar(output: &mut [i64], input: &[i64], w: usize, h: usize) {
    for y in 0..h {
        for x in 0..w {
            output[x * h + y] = input[y * w + x];
        }
    }
}

fn transpose<const NATIVE: bool>(output: &mut [i64], input: &[i64], w: usize, h: usize) {
    debug_assert_eq!(input.len(), w * h);
    debug_assert_eq!(output.len(), input.len());
    #[cfg(target_arch = "aarch64")]
    if NATIVE {
        // Each 2x2 tile is entirely within both disjoint slice extents.
        unsafe { transpose_neon(output, input, w, h) };
        return;
    }
    transpose_scalar(output, input, w, h);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn transpose_neon(output: &mut [i64], input: &[i64], w: usize, h: usize) {
    use std::arch::aarch64::*;
    unsafe {
        for y in (0..h - h % 2).step_by(2) {
            for x in (0..w - w % 2).step_by(2) {
                let a = vld1q_s64(input.as_ptr().add(y * w + x));
                let b = vld1q_s64(input.as_ptr().add((y + 1) * w + x));
                vst1q_s64(output.as_mut_ptr().add(x * h + y), vtrn1q_s64(a, b));
                vst1q_s64(output.as_mut_ptr().add((x + 1) * h + y), vtrn2q_s64(a, b));
            }
            if w % 2 != 0 {
                output[(w - 1) * h + y] = input[y * w + w - 1];
                output[(w - 1) * h + y + 1] = input[(y + 1) * w + w - 1];
            }
        }
    }
    if h % 2 != 0 {
        for x in 0..w {
            output[x * h + h - 1] = input[(h - 1) * w + x];
        }
    }
}

fn initialized_scratch(storage: &mut [std::mem::MaybeUninit<i64>]) -> &mut [i64] {
    for value in storage.iter_mut() {
        value.write(0);
    }
    // Only the active prefix is initialized, avoiding a 32-KiB clear for 4x4.
    unsafe { std::slice::from_raw_parts_mut(storage.as_mut_ptr().cast(), storage.len()) }
}

#[cfg(test)]
fn inverse_transform_fast<const NATIVE: bool, const SPARSE: bool>(
    w: usize,
    h: usize,
    coefficients: &[i32],
    bit_depth: u8,
    tx_type: usize,
) -> Result<Vec<i32>, Error> {
    transform_parameters(w, h, coefficients, bit_depth, tx_type)?;
    let mut output = vec![0; w * h];
    inverse_transform_fast_into::<NATIVE, SPARSE>(
        w,
        h,
        coefficients,
        bit_depth,
        tx_type,
        &mut output,
    )?;
    Ok(output)
}

fn inverse_transform_fast_into<const NATIVE: bool, const SPARSE: bool>(
    w: usize,
    h: usize,
    coefficients: &[i32],
    bit_depth: u8,
    tx_type: usize,
    output: &mut [i32],
) -> Result<(), Error> {
    let (row_kind, col_kind, row_shift) =
        transform_parameters(w, h, coefficients, bit_depth, tx_type)?;
    if output.len() != w * h {
        return Err(Error::Invalid("inverse transform output dimensions"));
    }
    #[cfg(test)]
    sparse_profile::input((w, h, bit_depth, tx_type), coefficients);
    if SPARSE && tx_type == 0 && coefficients[1..].iter().all(|&value| value == 0) {
        let mut value = i64::from(coefficients[0]);
        if w.ilog2().abs_diff(h.ilog2()) == 1 {
            value = round(value * 2896, 12);
        }
        value = dct_constant(value, bit_depth + 8)?;
        let clamp = 1i64 << ((bit_depth + 6).max(16) - 1);
        value = round(value, row_shift).clamp(-clamp, clamp - 1);
        value = round(dct_constant(value, (bit_depth + 6).max(16))?, 4);
        output.fill(value as i32);
        return Ok(());
    }
    let mut residual_storage = [std::mem::MaybeUninit::uninit(); 4096];
    let mut columns_storage = [std::mem::MaybeUninit::uninit(); 4096];
    let residual = initialized_scratch(&mut residual_storage[..w * h]);
    let columns = initialized_scratch(&mut columns_storage[..w * h]);
    let mut row_storage = [0; 64];
    let row = &mut row_storage[..w];
    let rectangular = w.ilog2().abs_diff(h.ilog2()) == 1;
    let clamp = 1i64 << ((bit_depth + 6).max(16) - 1);
    for y in 0..h {
        let input = &coefficients[y * w..(y + 1) * w];
        if SPARSE && input.iter().all(|&value| value == 0) {
            validate_axis(w, row_kind)?;
            continue;
        }
        prepare_row::<NATIVE>(row, input, rectangular);
        axis_dispatch::<SPARSE>(row, bit_depth + 8, row_kind)?;
        for x in 0..w {
            residual[y * w + x] = round(row[x], row_shift).clamp(-clamp, clamp - 1);
        }
    }
    transpose::<NATIVE>(columns, residual, w, h);
    #[cfg(test)]
    sparse_profile::columns((w, h, bit_depth, tx_type), columns);
    for column in columns.chunks_exact_mut(h) {
        axis_dispatch::<SPARSE>(column, (bit_depth + 6).max(16), col_kind)?;
    }
    let flip_y = [4, 6, 8, 14].contains(&tx_type);
    let flip_x = [5, 6, 7, 15].contains(&tx_type);
    for y in 0..h {
        for x in 0..w {
            output[y * w + x] = round(
                columns
                    [(if flip_x { w - x - 1 } else { x }) * h + if flip_y { h - y - 1 } else { y }],
                4,
            ) as i32;
        }
    }
    Ok(())
}

#[cfg(test)]
mod sparse_profile {
    use std::{
        cell::{Cell, RefCell},
        collections::BTreeMap,
    };
    pub type Key = (usize, usize, u8, usize);
    #[derive(Default)]
    pub struct Counts {
        pub blocks: usize,
        pub dc_blocks: usize,
        pub nonzero: usize,
        pub rows: usize,
        pub zero_rows: usize,
        pub dc_rows: usize,
        pub columns: usize,
        pub zero_columns: usize,
        pub dc_columns: usize,
        pub samples: Vec<Vec<i32>>,
    }
    std::thread_local! {
        static ENABLED: Cell<bool> = const { Cell::new(false) };
        static COUNTS: RefCell<BTreeMap<Key, Counts>> = RefCell::new(BTreeMap::new());
    }
    pub fn reset(enabled: bool) {
        ENABLED.with(|flag| flag.set(enabled));
        COUNTS.with(|counts| counts.borrow_mut().clear());
    }
    pub fn enabled() -> bool {
        ENABLED.with(Cell::get)
    }
    pub fn take() -> BTreeMap<Key, Counts> {
        ENABLED.with(|flag| flag.set(false));
        COUNTS.with(|counts| std::mem::take(&mut *counts.borrow_mut()))
    }
    pub fn input(key: Key, coefficients: &[i32]) {
        if !ENABLED.with(Cell::get) {
            return;
        }
        COUNTS.with(|counts| {
            let mut counts = counts.borrow_mut();
            let entry = counts.entry(key).or_default();
            entry.blocks += 1;
            entry.dc_blocks += usize::from(
                coefficients[0] != 0 && coefficients[1..].iter().all(|&value| value == 0),
            );
            entry.nonzero += coefficients.iter().filter(|&&value| value != 0).count();
            for row in coefficients.chunks_exact(key.0) {
                entry.rows += 1;
                entry.zero_rows += usize::from(row.iter().all(|&value| value == 0));
                entry.dc_rows +=
                    usize::from(row[0] != 0 && row[1..].iter().all(|&value| value == 0));
            }
            // Bounded, deterministic reservoir per actual transform class.
            if entry.samples.len() < 16 {
                entry.samples.push(coefficients.to_vec());
            } else {
                let mut hash = entry.blocks as u64 ^ 0x9e3779b97f4a7c15;
                hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d049bb133111eb);
                let slot = (hash ^ (hash >> 31)) as usize % entry.blocks;
                if slot < 16 {
                    entry.samples[slot] = coefficients.to_vec();
                }
            }
        });
    }
    pub fn columns(key: Key, columns: &[i64]) {
        if !ENABLED.with(Cell::get) {
            return;
        }
        COUNTS.with(|counts| {
            let mut counts = counts.borrow_mut();
            let entry = counts.get_mut(&key).unwrap();
            for column in columns.chunks_exact(key.1) {
                entry.columns += 1;
                entry.zero_columns += usize::from(column.iter().all(|&value| value == 0));
                entry.dc_columns +=
                    usize::from(column[0] != 0 && column[1..].iter().all(|&value| value == 0));
            }
        });
    }
}

#[cfg(test)]
fn axis_reference(t: &mut [i64], r: u8, kind: usize) -> Result<(), Error> {
    match kind {
        0 => inverse_dct_axis::<0, true>(t, r),
        1 => adst_axis::<true>(t, r),
        _ => axis(t, r, kind),
    }
}

#[cfg(test)]
fn inverse_transform_reference(
    w: usize,
    h: usize,
    coefficients: &[i32],
    bit_depth: u8,
    tx_type: usize,
) -> Result<Vec<i32>, Error> {
    let (row_kind, col_kind, row_shift) =
        transform_parameters(w, h, coefficients, bit_depth, tx_type)?;
    let mut residual = vec![0i64; w * h];
    let clamp = 1i64 << ((bit_depth + 6).max(16) - 1);
    for y in 0..h {
        let mut row: Vec<i64> = (0..w).map(|x| i64::from(coefficients[y * w + x])).collect();
        if w.ilog2().abs_diff(h.ilog2()) == 1 {
            for v in &mut row {
                *v = round(*v * 2896, 12);
            }
        }
        axis_reference(&mut row, bit_depth + 8, row_kind)?;
        for x in 0..w {
            residual[y * w + x] = round(row[x], row_shift).clamp(-clamp, clamp - 1);
        }
    }
    for x in 0..w {
        let mut col: Vec<i64> = (0..h).map(|y| residual[y * w + x]).collect();
        axis_reference(&mut col, (bit_depth + 6).max(16), col_kind)?;
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

    fn cpu_time() -> u64 {
        #[cfg(target_os = "macos")]
        {
            unsafe extern "C" {
                fn clock_gettime_nsec_np(clock_id: i32) -> u64;
            }
            // The SDK defines CLOCK_THREAD_CPUTIME_ID as 16.
            unsafe { clock_gettime_nsec_np(16) }
        }
        #[cfg(not(target_os = "macos"))]
        {
            0
        }
    }

    fn collect_spacewalk_sparse_profile()
    -> std::collections::BTreeMap<sparse_profile::Key, sparse_profile::Counts> {
        let bytes = std::fs::read(std::env::var("AV1_STREAM_OBU").unwrap()).unwrap();
        let frames = std::env::var("AV1_SPARSE_FRAMES")
            .ok()
            .map_or(2, |value| value.parse::<usize>().unwrap());
        assert!(frames > 0);
        let mut decoder = super::super::Av1Decoder::new();
        let mut stream = super::super::ObuStream::new();
        let mut displayed = 0;
        sparse_profile::reset(true);
        for obu in stream.push(&bytes).unwrap() {
            if decoder.decode_obu(&obu).unwrap().is_some() {
                displayed += 1;
                if displayed == frames {
                    break;
                }
            }
        }
        assert_eq!(displayed, frames);
        sparse_profile::take()
    }

    #[test]
    #[ignore = "bounded real-stream sparse transform profiling; requires AV1_STREAM_OBU"]
    fn spacewalk_sparse_distribution() {
        let counts = collect_spacewalk_sparse_profile();
        let mut totals = [0usize; 10];
        for (&(w, h, depth, kind), values) in &counts {
            eprintln!(
                "SPARSE {w}x{h} depth={depth} type={kind} blocks={} dc_blocks={} nonzero={} rows={} zero_rows={} dc_rows={} columns={} zero_columns={} dc_columns={}",
                values.blocks,
                values.dc_blocks,
                values.nonzero,
                values.rows,
                values.zero_rows,
                values.dc_rows,
                values.columns,
                values.zero_columns,
                values.dc_columns
            );
            for (sum, value) in totals.iter_mut().zip([
                values.blocks,
                values.dc_blocks,
                values.nonzero,
                w * h * values.blocks,
                values.rows,
                values.zero_rows,
                values.dc_rows,
                values.columns,
                values.zero_columns,
                values.dc_columns,
            ]) {
                *sum += value;
            }
        }
        eprintln!(
            "SPARSE TOTAL blocks={} dc_blocks={} nonzero={} coefficients={} rows={} zero_rows={} dc_rows={} columns={} zero_columns={} dc_columns={}",
            totals[0],
            totals[1],
            totals[2],
            totals[3],
            totals[4],
            totals[5],
            totals[6],
            totals[7],
            totals[8],
            totals[9]
        );
        assert!(totals[0] > 0);
    }

    #[test]
    fn caller_output_matches_reference_and_preserves_errors() {
        let mut state = 71u32;
        let mut storage = [1234567; 4098];
        for depth in [8, 10, 12] {
            for &(w, h) in &super::super::coefficients::TX_DIMENSIONS {
                for kind in 0..16 {
                    for trial in 0..4 {
                        let mut input = vec![0; w * h];
                        for value in &mut input {
                            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                            if trial == 1 {
                                *value = (state >> 23) as i32 - 256;
                            }
                        }
                        if trial == 2 {
                            input[0] = 40000;
                        }
                        if trial == 3 {
                            input[w * h - 1] = i32::MAX;
                        }
                        storage.fill(1234567);
                        let expected = inverse_transform_reference(w, h, &input, depth, kind);
                        let actual = inverse_transform_into(
                            w,
                            h,
                            &input,
                            depth,
                            kind,
                            &mut storage[1..1 + w * h],
                        );
                        match expected {
                            Ok(expected) => {
                                assert_eq!(actual, Ok(()));
                                assert_eq!(&storage[1..1 + w * h], expected);
                            }
                            Err(error) => {
                                assert_eq!(actual, Err(error));
                                assert!(storage.iter().all(|&value| value == 1234567));
                            }
                        }
                        assert_eq!(storage[0], 1234567);
                        assert!(storage[1 + w * h..].iter().all(|&value| value == 1234567));
                    }
                }
            }
        }
        for (w, h, depth, kind, input_len, output_len) in [
            (4, 4, 8, 0, 16, 0),
            (4, 4, 8, 0, 16, 15),
            (4, 4, 8, 0, 16, 17),
            (4, 4, 8, 0, 15, 16),
            (4, 4, 9, 0, 16, 16),
            (4, 4, 8, 16, 16, 16),
            (3, 4, 8, 0, 12, 12),
            (usize::MAX, 4, 8, 0, 0, 0),
        ] {
            storage.fill(1234567);
            assert!(
                inverse_transform_into(
                    w,
                    h,
                    &vec![0; input_len],
                    depth,
                    kind,
                    &mut storage[..output_len]
                )
                .is_err()
            );
            assert!(storage.iter().all(|&value| value == 1234567));
        }
    }

    #[test]
    #[ignore = "real-input allocation/reuse ABBA; requires AV1_STREAM_OBU"]
    fn spacewalk_output_reuse_weighted_abba() {
        use std::{hint::black_box, time::Instant};
        let counts = collect_spacewalk_sparse_profile();
        let blocks: usize = counts.values().map(|values| values.blocks).sum();
        let mut weighted = [0.0; 2];
        let mut output = [0; 4096];
        for (&(w, h, depth, kind), values) in &counts {
            for input in &values.samples {
                let expected = inverse_transform_reference(w, h, input, depth, kind).unwrap();
                inverse_transform_into(w, h, input, depth, kind, &mut output[..w * h]).unwrap();
                assert_eq!(&output[..w * h], expected);
            }
            let repeats = (40000 / (w * h)).clamp(8, 500);
            let mut timings: [Vec<f64>; 2] = Default::default();
            for trial in 0..16 {
                let mode = [0, 1, 1, 0][trial % 4];
                let wall = Instant::now();
                let start = cpu_time();
                for _ in 0..repeats {
                    for input in &values.samples {
                        if mode == 0 {
                            black_box(
                                inverse_transform(w, h, black_box(input), depth, kind).unwrap(),
                            );
                        } else {
                            inverse_transform_into(
                                w,
                                h,
                                black_box(input),
                                depth,
                                kind,
                                black_box(&mut output[..w * h]),
                            )
                            .unwrap();
                            black_box(&output[..w * h]);
                        }
                    }
                }
                let elapsed = if cfg!(target_os = "macos") {
                    (cpu_time() - start) as f64
                } else {
                    wall.elapsed().as_nanos() as f64
                };
                timings[mode].push(elapsed / (repeats * values.samples.len()) as f64);
            }
            for (mode, mut times) in timings.into_iter().enumerate() {
                times.sort_by(f64::total_cmp);
                weighted[mode] += times[times.len() / 2] * values.blocks as f64;
            }
        }
        eprintln!(
            "OUTPUT REUSE WEIGHTED ABBA actual_blocks={blocks} allocating_ns={:.1} reused_ns={:.1} speedup={:.3}",
            weighted[0] / blocks as f64,
            weighted[1] / blocks as f64,
            weighted[0] / weighted[1]
        );
    }

    #[test]
    fn sparse_dc_preserves_rounding_clipping_and_error_boundaries() {
        for depth in [8, 10, 12] {
            let bound = 1i32 << (depth + 7);
            for &(w, h) in &super::super::coefficients::TX_DIMENSIONS {
                for kind in 0..16 {
                    for dc in [
                        i32::MIN,
                        -bound - 1,
                        -bound,
                        -40000,
                        -2049,
                        -1,
                        0,
                        1,
                        2048,
                        40000,
                        46340,
                        46341,
                        bound - 1,
                        bound,
                        bound + 1,
                        i32::MAX,
                    ] {
                        let mut input = vec![0; w * h];
                        input[0] = dc;
                        let expected = inverse_transform_reference(w, h, &input, depth, kind);
                        assert_eq!(
                            inverse_transform_fast::<false, true>(w, h, &input, depth, kind),
                            expected,
                            "DC {w}x{h} depth={depth} type={kind} value={dc}"
                        );
                        assert_eq!(
                            inverse_transform_fast::<true, true>(w, h, &input, depth, kind),
                            expected
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "bounded real-input weighted ABBA; requires AV1_STREAM_OBU"]
    fn spacewalk_sparse_weighted_abba() {
        use std::{hint::black_box, time::Instant};
        type Transform = fn(usize, usize, &[i32], u8, usize) -> Result<Vec<i32>, Error>;
        fn dense_selected(
            w: usize,
            h: usize,
            input: &[i32],
            depth: u8,
            kind: usize,
        ) -> Result<Vec<i32>, Error> {
            if w <= 16 && h <= 16 {
                inverse_transform_fast::<true, false>(w, h, input, depth, kind)
            } else {
                inverse_transform_fast::<false, false>(w, h, input, depth, kind)
            }
        }
        let paths: [Transform; 4] = [
            inverse_transform_reference,
            dense_selected,
            inverse_transform_fast::<false, true>,
            inverse_transform,
        ];
        let counts = collect_spacewalk_sparse_profile();
        let blocks: usize = counts.values().map(|counts| counts.blocks).sum();
        for &(a, b) in &[(1, 3), (0, 3), (2, 3)] {
            let mut weighted = [0.0; 2];
            for (&(w, h, depth, kind), counts) in &counts {
                for input in &counts.samples {
                    let expected = paths[0](w, h, input, depth, kind).unwrap();
                    for path in paths {
                        assert_eq!(path(w, h, input, depth, kind).unwrap(), expected);
                    }
                }
                let repeats = (40000 / (w * h)).clamp(8, 500);
                let mut timings: [Vec<f64>; 2] = Default::default();
                for trial in 0..12 {
                    let mode = [0, 1, 1, 0][trial % 4];
                    let path = paths[if mode == 0 { a } else { b }];
                    let wall = Instant::now();
                    let start = cpu_time();
                    for _ in 0..repeats {
                        for input in &counts.samples {
                            black_box(path(w, h, black_box(input), depth, kind).unwrap());
                        }
                    }
                    let elapsed = if cfg!(target_os = "macos") {
                        (cpu_time() - start) as f64
                    } else {
                        wall.elapsed().as_nanos() as f64
                    };
                    timings[mode].push(elapsed / (repeats * counts.samples.len()) as f64);
                }
                let times = timings.map(|mut values| {
                    values.sort_by(f64::total_cmp);
                    values[values.len() / 2]
                });
                for mode in 0..2 {
                    weighted[mode] += times[mode] * counts.blocks as f64;
                }
                if a == 1 && counts.blocks >= 100 {
                    eprintln!(
                        "REAL SPARSE ABBA {w}x{h} depth={depth} type={kind} blocks={} samples={} dense_ns={:.1} sparse_ns={:.1} speedup={:.3}",
                        counts.blocks,
                        counts.samples.len(),
                        times[0],
                        times[1],
                        times[0] / times[1]
                    );
                }
            }
            eprintln!(
                "REAL SPARSE WEIGHTED ABBA paths={a}/{b} actual_blocks={blocks} reservoir_limit=16 thread_cpu={} first_ns={:.1} second_ns={:.1} speedup={:.3}",
                cfg!(target_os = "macos"),
                weighted[0] / blocks as f64,
                weighted[1] / blocks as f64,
                weighted[0] / weighted[1]
            );
        }
    }

    #[test]
    fn stack_scalar_and_neon_match_retained_transform() {
        let mut state = 29u32;
        for bit_depth in [8, 10, 12] {
            for &(w, h) in &super::super::coefficients::TX_DIMENSIONS {
                for tx_type in 0..16 {
                    for trial in 0..4 {
                        let mut coefficients = vec![0; w * h];
                        for value in &mut coefficients {
                            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                            *value = if trial == 0 {
                                0
                            } else {
                                (state >> 23) as i32 - 256
                            };
                        }
                        if trial == 2 {
                            coefficients[0] = 8192;
                        } else if trial == 3 {
                            coefficients[0] = i32::MAX;
                            coefficients[w * h - 1] = i32::MIN;
                        }
                        let expected =
                            inverse_transform_reference(w, h, &coefficients, bit_depth, tx_type);
                        for actual in [
                            inverse_transform_fast::<false, false>(
                                w,
                                h,
                                &coefficients,
                                bit_depth,
                                tx_type,
                            ),
                            inverse_transform_fast::<false, true>(
                                w,
                                h,
                                &coefficients,
                                bit_depth,
                                tx_type,
                            ),
                            inverse_transform_fast::<true, true>(
                                w,
                                h,
                                &coefficients,
                                bit_depth,
                                tx_type,
                            ),
                        ] {
                            assert_eq!(
                                actual, expected,
                                "{w}x{h} depth={bit_depth} type={tx_type} trial={trial}"
                            );
                        }
                    }
                }
            }
        }
        for (w, h, depth, kind, count) in [
            (3, 4, 8, 0, 12),
            (4, 4, 9, 0, 16),
            (4, 4, 8, 16, 16),
            (4, 4, 8, 0, 15),
        ] {
            let input = vec![0; count];
            assert_eq!(
                inverse_transform_fast::<true, true>(w, h, &input, depth, kind),
                inverse_transform_reference(w, h, &input, depth, kind)
            );
        }
    }

    #[test]
    fn neon_setup_matches_scalar_at_cropped_tails() {
        let source: Vec<_> = (0..70)
            .map(|i| match i % 5 {
                0 => i32::MAX,
                1 => i32::MIN,
                _ => i * 117 - 403,
            })
            .collect();
        for count in 0..=65 {
            for rectangular in [false, true] {
                let mut expected = vec![123i64; count + 6];
                let mut actual = expected.clone();
                prepare_row::<false>(
                    &mut expected[3..3 + count],
                    &source[1..1 + count],
                    rectangular,
                );
                prepare_row::<true>(
                    &mut actual[3..3 + count],
                    &source[1..1 + count],
                    rectangular,
                );
                assert_eq!(actual, expected);
            }
        }
        for (w, h) in [(0, 0), (1, 1), (3, 5), (5, 3), (31, 33), (64, 64), (65, 3)] {
            let input: Vec<_> = (0..w * h + 1).map(|i| (i as i64 - 2037) * 19997).collect();
            let mut expected = vec![123; w * h + 6];
            let mut actual = expected.clone();
            transpose::<false>(&mut expected[3..3 + w * h], &input[1..], w, h);
            transpose::<true>(&mut actual[3..3 + w * h], &input[1..], w, h);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    #[ignore = "manual same-binary ABBA transform benchmark"]
    fn retained_scalar_neon_abba_benchmark() {
        use std::{hint::black_box, time::Instant};
        type Transform = fn(usize, usize, &[i32], u8, usize) -> Result<Vec<i32>, Error>;
        let paths: [Transform; 3] = [
            inverse_transform_reference,
            inverse_transform_fast::<false, false>,
            inverse_transform_fast::<true, false>,
        ];
        let median = |values: &mut Vec<f64>| {
            values.sort_by(f64::total_cmp);
            values[values.len() / 2]
        };
        for (w, h, kind) in [
            (4, 4, 0),
            (8, 8, 0),
            (16, 16, 0),
            (32, 32, 0),
            (64, 64, 0),
            (16, 8, 0),
            (32, 16, 0),
            (4, 4, 3),
            (8, 8, 3),
            (16, 16, 3),
            (4, 4, 9),
            (8, 8, 9),
            (16, 16, 9),
            (32, 32, 9),
        ] {
            let input: Vec<i32> = (0..w * h).map(|i| (i as i32 * 73 % 257) - 128).collect();
            let expected = paths[0](w, h, &input, 10, kind).unwrap();
            for path in paths {
                assert_eq!(path(w, h, &input, 10, kind).unwrap(), expected);
                black_box(path(w, h, &input, 10, kind).unwrap());
            }
            let repeats = (500_000 / (w * h)).max(32);
            for (a, b) in [(0, 1), (1, 2)] {
                let mut timings: [Vec<f64>; 2] = Default::default();
                let mut wall_times: [Vec<f64>; 2] = Default::default();
                for trial in 0..12 {
                    let choice = [0, 1, 1, 0][trial % 4];
                    let path = paths[if choice == 0 { a } else { b }];
                    let began = Instant::now();
                    let cpu_start = cpu_time();
                    for _ in 0..repeats {
                        black_box(path(w, h, black_box(&input), 10, kind).unwrap());
                    }
                    let cpu_elapsed = cpu_time() - cpu_start;
                    let wall = began.elapsed().as_secs_f64() * 1e9 / repeats as f64;
                    timings[choice].push(if cfg!(target_os = "macos") {
                        cpu_elapsed as f64 / repeats as f64
                    } else {
                        wall
                    });
                    wall_times[choice].push(wall);
                }
                let first = median(&mut timings[0]);
                let second = median(&mut timings[1]);
                let first_wall = median(&mut wall_times[0]);
                let second_wall = median(&mut wall_times[1]);
                eprintln!(
                    "TRANSFORM ABBA {w}x{h} type={kind} paths={a}/{b} repeats={repeats} thread_cpu={} first_ns={first:.1} second_ns={second:.1} speedup={:.3} wall_ns={first_wall:.1}/{second_wall:.1}",
                    cfg!(target_os = "macos"),
                    first / second
                );
            }
        }
    }

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
