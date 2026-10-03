//! VP8's 4x4 inverse transforms, using the fixed-point rounding of RFC 6386.

const COS_PI_8_SQRT2_MINUS_1: i32 = 20091;
const SIN_PI_8_SQRT2: i32 = 35468;

pub(super) fn inverse_walsh(input: &[i32; 16]) -> [i32; 16] {
    if input[1..].iter().all(|&value| value == 0) {
        return [(input[0] + 3) >> 3; 16];
    }
    inverse_walsh_full(input)
}

fn inverse_walsh_full(input: &[i32; 16]) -> [i32; 16] {
    let mut columns = [0; 16];
    for x in 0..4 {
        let a = input[x] + input[12 + x];
        let b = input[4 + x] + input[8 + x];
        let c = input[4 + x] - input[8 + x];
        let d = input[x] - input[12 + x];
        columns[x] = a + b;
        columns[4 + x] = c + d;
        columns[8 + x] = a - b;
        columns[12 + x] = d - c;
    }
    let mut result = [0; 16];
    for y in 0..4 {
        let row = y * 4;
        let a = columns[row] + columns[row + 3];
        let b = columns[row + 1] + columns[row + 2];
        let c = columns[row + 1] - columns[row + 2];
        let d = columns[row] - columns[row + 3];
        result[row] = (a + b + 3) >> 3;
        result[row + 1] = (c + d + 3) >> 3;
        result[row + 2] = (a - b + 3) >> 3;
        result[row + 3] = (d - c + 3) >> 3;
    }
    result
}

#[cfg(any(not(target_arch = "aarch64"), test))]
fn odd_terms(first: i32, third: i32) -> (i32, i32) {
    let cosine = |value: i32| ((i64::from(value) * i64::from(COS_PI_8_SQRT2_MINUS_1)) >> 16) as i32;
    let sine = |value: i32| ((i64::from(value) * i64::from(SIN_PI_8_SQRT2)) >> 16) as i32;
    let high = first + cosine(first);
    let low = sine(third);
    let outer = high + low;
    let inner = sine(first) - third - cosine(third);
    (outer, inner)
}

pub(super) fn inverse_dct(input: &[i32; 16]) -> [i32; 16] {
    if input[1..].iter().all(|&value| value == 0) {
        return inverse_dct_dc(input[0]);
    }
    #[cfg(target_arch = "aarch64")]
    { unsafe { inverse_dct_neon(input) } }
    #[cfg(not(target_arch = "aarch64"))]
    { inverse_dct_full(input) }
}

pub(super) fn inverse_dct_dc(dc: i32) -> [i32; 16] {
    [(dc + 4) >> 3; 16]
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn inverse_dct_neon(input: &[i32; 16]) -> [i32; 16] {
    use std::arch::aarch64::*;
    let multiply = |value: int32x4_t, coefficient| {
        // Keep the scalar transform's signed 64-bit products before truncation.
        vcombine_s32(vshrn_n_s64::<16>(vmull_n_s32(vget_low_s32(value), coefficient)),
            vshrn_n_s64::<16>(vmull_high_n_s32(value, coefficient)))
    };
    let pass = |rows: [int32x4_t; 4]| {
        let even_sum = vaddq_s32(rows[0], rows[2]);
        let even_diff = vsubq_s32(rows[0], rows[2]);
        let outer = vaddq_s32(vaddq_s32(rows[1], multiply(rows[1], COS_PI_8_SQRT2_MINUS_1)),
            multiply(rows[3], SIN_PI_8_SQRT2));
        let inner = vsubq_s32(vsubq_s32(multiply(rows[1], SIN_PI_8_SQRT2), rows[3]),
            multiply(rows[3], COS_PI_8_SQRT2_MINUS_1));
        [vaddq_s32(even_sum, outer), vaddq_s32(even_diff, inner),
            vsubq_s32(even_diff, inner), vsubq_s32(even_sum, outer)]
    };
    let pass_narrow = |rows: [int32x4_t; 4]| {
        let first = vmovn_s32(rows[1]);
        let third = vmovn_s32(rows[3]);
        let cosine_first = vshrq_n_s32::<16>(vmull_n_s16(first, COS_PI_8_SQRT2_MINUS_1 as i16));
        let cosine_third = vshrq_n_s32::<16>(vmull_n_s16(third, COS_PI_8_SQRT2_MINUS_1 as i16));
        // 35468 = 65536 - 30068, so signed 16-bit multiplies preserve the exact floor.
        let sine_first = vaddq_s32(rows[1], vshrq_n_s32::<16>(vmull_n_s16(first,
            (SIN_PI_8_SQRT2 - 65536) as i16)));
        let sine_third = vaddq_s32(rows[3], vshrq_n_s32::<16>(vmull_n_s16(third,
            (SIN_PI_8_SQRT2 - 65536) as i16)));
        let even_sum = vaddq_s32(rows[0], rows[2]);
        let even_diff = vsubq_s32(rows[0], rows[2]);
        let outer = vaddq_s32(vaddq_s32(rows[1], cosine_first), sine_third);
        let inner = vsubq_s32(vsubq_s32(sine_first, rows[3]), cosine_third);
        [vaddq_s32(even_sum, outer), vaddq_s32(even_diff, inner),
            vsubq_s32(even_diff, inner), vsubq_s32(even_sum, outer)]
    };
    let transpose = |rows: [int32x4_t; 4]| {
        let first = vtrn1q_s32(rows[0], rows[1]);
        let second = vtrn2q_s32(rows[0], rows[1]);
        let third = vtrn1q_s32(rows[2], rows[3]);
        let fourth = vtrn2q_s32(rows[2], rows[3]);
        [vcombine_s32(vget_low_s32(first), vget_low_s32(third)),
            vcombine_s32(vget_low_s32(second), vget_low_s32(fourth)),
            vcombine_s32(vget_high_s32(first), vget_high_s32(third)),
            vcombine_s32(vget_high_s32(second), vget_high_s32(fourth))]
    };
    let mut rows = [vdupq_n_s32(0); 4];
    for (row, values) in rows.iter_mut().zip(input.chunks_exact(4)) {
        *row = unsafe { vld1q_s32(values.as_ptr()) };
    }
    let mut maximum = vdupq_n_u32(0);
    for &row in &rows { maximum = vmaxq_u32(maximum, vreinterpretq_u32_s32(vabsq_s32(row))); }
    // This bound keeps every first-pass result inside signed 16-bit range as well.
    let narrow = vmaxvq_u32(maximum) <= 8191;
    let columns = transpose(if narrow { pass_narrow(rows) } else { pass(rows) });
    let rows = transpose(if narrow { pass_narrow(columns) } else { pass(columns) });
    let mut result = [0; 16];
    for (row, output) in rows.into_iter().zip(result.chunks_exact_mut(4)) {
        let rounded = vshrq_n_s32::<3>(vaddq_s32(row, vdupq_n_s32(4)));
        unsafe { vst1q_s32(output.as_mut_ptr(), rounded); }
    }
    result
}

#[cfg(any(not(target_arch = "aarch64"), test))]
fn inverse_dct_full(input: &[i32; 16]) -> [i32; 16] {
    let mut columns = [0; 16];
    for x in 0..4 {
        let even_sum = input[x] + input[8 + x];
        let even_diff = input[x] - input[8 + x];
        let (outer, inner) = odd_terms(input[4 + x], input[12 + x]);
        columns[x] = even_sum + outer;
        columns[4 + x] = even_diff + inner;
        columns[8 + x] = even_diff - inner;
        columns[12 + x] = even_sum - outer;
    }
    let mut result = [0; 16];
    for y in 0..4 {
        let row = y * 4;
        let even_sum = columns[row] + columns[row + 2];
        let even_diff = columns[row] - columns[row + 2];
        let (outer, inner) = odd_terms(columns[row + 1], columns[row + 3]);
        result[row] = (even_sum + outer + 4) >> 3;
        result[row + 1] = (even_diff + inner + 4) >> 3;
        result[row + 2] = (even_diff - inner + 4) >> 3;
        result[row + 3] = (even_sum - outer + 4) >> 3;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accelerated_dct_matches_sparse_dense_and_wide_coefficients() {
        let mut state = 73u32;
        for mask in 0..65536u32 {
            let input = std::array::from_fn(|index| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                if mask & (1 << index) == 0 { 0 }
                else { ((state >> 16) as i32 - 32768) * 32 }
            });
            assert_eq!(inverse_dct(&input), inverse_dct_full(&input), "mask={mask}");
            let small = input.map(|value| value >> 7);
            assert_eq!(inverse_dct(&small), inverse_dct_full(&small), "small={mask}");
            let boundaries = std::array::from_fn(|index| {
                if mask & (1 << index) == 0 { -8191 } else { 8191 }
            });
            assert_eq!(inverse_dct(&boundaries), inverse_dct_full(&boundaries), "boundaries={mask}");
        }
    }

    #[test]
    #[ignore = "manual full inverse-DCT kernel timing"]
    fn benchmark_full_dct() {
        use std::hint::black_box;
        use std::time::Instant;
        for occupied in [2, 8, 16] {
            let mut input = [0; 16];
            for (index, value) in input[..occupied].iter_mut().enumerate() {
                *value = (index * 73 % 511) as i32 - 255;
            }
            for trial in 0..5 {
                for accelerated in if trial % 2 == 0 { [false, true] } else { [true, false] } {
                    let start = Instant::now();
                    for _ in 0..1_000_000 {
                        let output = if accelerated { inverse_dct(black_box(&input)) }
                            else { inverse_dct_full(black_box(&input)) };
                        black_box(output);
                    }
                    eprintln!("VP8 DCT occupied={occupied} trial={trial} accelerated={accelerated} elapsed={:?}", start.elapsed());
                }
            }
        }
    }

    #[test]
    fn dc_fast_paths_match_full_transforms_for_signed_values() {
        for dc in -32768..=32767 {
            let mut input = [0; 16];
            input[0] = dc;
            assert_eq!(inverse_dct_dc(dc), inverse_dct_full(&input));
            assert_eq!(inverse_dct(&input), inverse_dct_full(&input));
            assert_eq!(inverse_walsh(&input), inverse_walsh_full(&input));
        }
    }

    #[test]
    fn transform_products_use_wide_intermediates() {
        for first in [-1_000_000, -32768, 32767, 1_000_000] {
            for third in [-1_000_000, -32768, 32767, 1_000_000] {
                let cosine = |value| (i64::from(value) * 20091) >> 16;
                let sine = |value| (i64::from(value) * 35468) >> 16;
                let outer = i64::from(first) + cosine(first) + sine(third);
                let inner = sine(first) - i64::from(third) - cosine(third);
                assert_eq!(odd_terms(first, third), (outer as i32, inner as i32));
            }
        }
    }

    #[test]
    fn zero_and_dc_only_blocks() {
        assert_eq!(inverse_walsh(&[0; 16]), [0; 16]);
        assert_eq!(inverse_dct(&[0; 16]), [0; 16]);
        let mut dc = [0; 16];
        dc[0] = 80;
        assert_eq!(inverse_walsh(&dc), [10; 16]);
        assert_eq!(inverse_dct(&dc), [10; 16]);
    }

    #[test]
    fn alternating_coefficients_have_spatial_structure() {
        let mut coefficients = [0; 16];
        coefficients[1] = 64;
        let pixels = inverse_dct(&coefficients);
        assert!(pixels[..4].windows(2).any(|pair| pair[0] != pair[1]));
        assert!(pixels.chunks(4).all(|row| row == &pixels[..4]));
    }
}
