//! Seven-bit fixed-point row convolution shared by VP8 and VP9.

/// Copy an integer-pixel block with replication of the visible reference edges.
#[inline]
pub(super) fn copy_replicated_block(
    destination: &mut [u8], destination_stride: usize, destination_start: usize,
    reference: &[u8], reference_stride: usize, visible_width: usize, visible_height: usize,
    source_x: i32, source_y: i32, width: usize, height: usize,
) {
    debug_assert!(visible_width > 0 && visible_height > 0);
    debug_assert!(visible_width <= reference_stride);
    debug_assert!(visible_height <= reference.len() / reference_stride);
    debug_assert!(width > 0 && height > 0 && width <= 64 && height <= 64);
    debug_assert!(destination_start % destination_stride + width <= destination_stride);
    debug_assert!(destination_start + (height - 1) * destination_stride + width <= destination.len());
    let left = (-i64::from(source_x)).max(0) as usize;
    let left = left.min(width);
    let from_x = i64::from(source_x).clamp(0, visible_width as i64) as usize;
    let copied = (width - left).min(visible_width - from_x);
    let mut previous_row = None;
    for row in 0..height {
        let source_row = (i64::from(source_y) + row as i64)
            .clamp(0, visible_height as i64 - 1) as usize;
        let to = destination_start + row * destination_stride;
        if previous_row == Some(source_row) {
            let previous = to - destination_stride;
            destination.copy_within(previous..previous + width, to);
        } else {
            let from = source_row * reference_stride;
            let source = &reference[from..from + visible_width];
            let target = &mut destination[to..to + width];
            target[..left].fill(source[0]);
            target[left..left + copied].copy_from_slice(&source[from_x..from_x + copied]);
            target[left + copied..].fill(source[visible_width - 1]);
        }
        previous_row = Some(source_row);
    }
}

#[derive(Clone, Copy)]
pub(super) struct ConvolutionFilter<const N: usize> {
    pub(super) taps: [(usize, i32); N],
    pub(super) count: usize,
    pub(super) first_tap: usize,
    pub(super) last_tap: usize,
    narrow: bool,
}

impl<const N: usize> ConvolutionFilter<N> {
    pub(super) const fn new(coefficients: [i32; N]) -> Self {
        let mut filter = Self { taps: [(0, 0); N], count: 0, first_tap: 0, last_tap: 0, narrow: false };
        let mut sum = 0i64;
        let mut magnitude = 0i64;
        let mut tap = 0;
        while tap < N {
            let coefficient = coefficients[tap];
            sum += coefficient as i64;
            magnitude += (coefficient as i64).abs();
            if coefficient != 0 {
                if filter.count == 0 { filter.first_tap = tap; }
                filter.last_tap = tap;
                filter.taps[filter.count] = (tap, coefficient);
                filter.count += 1;
            }
            tap += 1;
        }
        filter.narrow = sum == 128 && magnitude <= 256;
        filter
    }
}

#[inline]
pub(super) fn convolve_prepared_row<const N: usize>(source: &[u8], origin: usize, stride: usize,
    filter: &ConvolutionFilter<N>, output: &mut [u8])
{
    let taps = &filter.taps[..filter.count];
    convolve_row_with_bound(source, origin, stride, taps, output, filter.narrow);
}

#[cfg(test)]
#[inline(always)]
pub(super) fn convolve_row(source: &[u8], origin: usize, stride: usize,
    taps: &[(usize, i32)], output: &mut [u8])
{
    convolve_row_with_bound(source, origin, stride, taps, output, false);
}

#[inline(always)]
fn convolve_row_with_bound(source: &[u8], origin: usize, stride: usize,
    taps: &[(usize, i32)], output: &mut [u8], _narrow: bool)
{
    if let [(tap, 128)] = taps {
        let start = origin + tap * stride;
        output.copy_from_slice(&source[start..start + output.len()]);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if output.len() == 4 {
        if _narrow {
            unsafe { convolve_four_centered_neon(source, origin, stride, taps, output); }
        } else {
            unsafe { convolve_four_neon(source, origin, stride, taps, output); }
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if !output.is_empty() && output.len().is_multiple_of(8) {
        // NEON is baseline on AArch64; loads and stores use checked byte slices.
        unsafe { convolve_row_neon(source, origin, stride, taps, output); }
        return;
    }
    convolve_row_scalar(source, origin, stride, taps, output);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn convolve_four_centered_neon(source: &[u8], origin: usize, stride: usize,
    taps: &[(usize, i32)], output: &mut [u8])
{
    use std::arch::aarch64::*;
    // Centering samples bounds every partial sum by 128 * sum(abs(taps)).
    // A unity-gain filter restores exactly 128 after the seven-bit rounding.
    let mut sum = vdupq_n_s16(0);
    for &(tap, coefficient) in taps {
        let start = origin + tap * stride;
        let packed = u32::from_le_bytes(source[start..start + 4].try_into().unwrap());
        let centered = vreinterpret_s8_u8(veor_u8(vcreate_u8(u64::from(packed)), vdup_n_u8(128)));
        sum = vmlaq_n_s16(sum, vmovl_s8(centered), coefficient as i16);
    }
    let result = vqmovun_s16(vaddq_s16(vrshrq_n_s16::<7>(sum), vdupq_n_s16(128)));
    let packed = vget_lane_u32::<0>(vreinterpret_u32_u8(result));
    output.copy_from_slice(&packed.to_le_bytes());
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn convolve_four_neon(source: &[u8], origin: usize, stride: usize,
    taps: &[(usize, i32)], output: &mut [u8])
{
    use std::arch::aarch64::*;
    let mut sum = vdupq_n_s32(0);
    for &(tap, coefficient) in taps {
        let start = origin + tap * stride;
        // Load exactly four bytes, including when the last tap ends at the plane boundary.
        let packed = u32::from_le_bytes(source[start..start + 4].try_into().unwrap());
        let samples = vreinterpret_s16_u16(vget_low_u16(vmovl_u8(vcreate_u8(u64::from(packed)))));
        sum = vmlal_n_s16(sum, samples, coefficient as i16);
    }
    let rounded = vqrshrn_n_s32::<7>(sum);
    let result = vqmovun_s16(vcombine_s16(rounded, vdup_n_s16(0)));
    let packed = vget_lane_u32::<0>(vreinterpret_u32_u8(result));
    output.copy_from_slice(&packed.to_le_bytes());
}

pub(super) fn convolve_row_scalar(source: &[u8], origin: usize, stride: usize,
    taps: &[(usize, i32)], output: &mut [u8])
{
    let mut sums = [0i32; 64];
    for &(tap, coefficient) in taps {
        let start = origin + tap * stride;
        for (sum, &sample) in sums[..output.len()].iter_mut()
            .zip(&source[start..start + output.len()])
        {
            *sum += i32::from(sample) * coefficient;
        }
    }
    for (sample, sum) in output.iter_mut().zip(sums) {
        *sample = ((sum + 64) >> 7).clamp(0, 255) as u8;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn convolve_row_neon(source: &[u8], origin: usize, stride: usize,
    taps: &[(usize, i32)], output: &mut [u8])
{
    use std::arch::aarch64::*;
    let mut column = 0;
    while column + 16 <= output.len() {
        let mut first = vdupq_n_s32(0);
        let mut second = vdupq_n_s32(0);
        let mut third = vdupq_n_s32(0);
        let mut fourth = vdupq_n_s32(0);
        for &(tap, coefficient) in taps {
            let start = origin + tap * stride + column;
            let samples = &source[start..start + 16];
            let samples = unsafe { vld1q_u8(samples.as_ptr()) };
            let low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(samples)));
            let high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(samples)));
            first = vmlal_n_s16(first, vget_low_s16(low), coefficient as i16);
            second = vmlal_n_s16(second, vget_high_s16(low), coefficient as i16);
            third = vmlal_n_s16(third, vget_low_s16(high), coefficient as i16);
            fourth = vmlal_n_s16(fourth, vget_high_s16(high), coefficient as i16);
        }
        let low = vqmovun_s16(vcombine_s16(vqrshrn_n_s32::<7>(first), vqrshrn_n_s32::<7>(second)));
        let high = vqmovun_s16(vcombine_s16(vqrshrn_n_s32::<7>(third), vqrshrn_n_s32::<7>(fourth)));
        let destination = &mut output[column..column + 16];
        unsafe { vst1q_u8(destination.as_mut_ptr(), vcombine_u8(low, high)); }
        column += 16;
    }
    for column in (column..output.len()).step_by(8) {
        let mut low = vdupq_n_s32(0);
        let mut high = vdupq_n_s32(0);
        for &(tap, coefficient) in taps {
            let start = origin + tap * stride + column;
            let samples = &source[start..start + 8];
            let samples = vreinterpretq_s16_u16(vmovl_u8(unsafe { vld1_u8(samples.as_ptr()) }));
            low = vmlal_n_s16(low, vget_low_s16(samples), coefficient as i16);
            high = vmlal_n_s16(high, vget_high_s16(samples), coefficient as i16);
        }
        let rounded = vcombine_s16(vqrshrn_n_s32::<7>(low), vqrshrn_n_s32::<7>(high));
        let result = vqmovun_s16(rounded);
        let destination = &mut output[column..column + 8];
        unsafe { vst1_u8(destination.as_mut_ptr(), result); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replicated_blocks_match_clamped_samples_and_preserve_guards() {
        let reference: Vec<u8> = (0..40 * 36).map(|i| ((i * 73) ^ (i >> 3)) as u8).collect();
        for (visible_width, visible_height) in [(1, 1), (17, 19), (37, 31)] {
            for (width, height) in [(1, 1), (4, 8), (8, 4), (16, 16), (64, 64)] {
                let stride = width + 7;
                for source_x in [i32::MIN, -65, -1, 0, 1, 36, 65, i32::MAX] {
                    for source_y in [i32::MIN, -65, -1, 0, 1, 30, 65, i32::MAX] {
                        let mut expected = vec![91; stride * (height + 4)];
                        let start = 2 * stride + 3;
                        for row in 0..height {
                            for col in 0..width {
                                let sx = (i64::from(source_x) + col as i64)
                                    .clamp(0, visible_width as i64 - 1) as usize;
                                let sy = (i64::from(source_y) + row as i64)
                                    .clamp(0, visible_height as i64 - 1) as usize;
                                expected[start + row * stride + col] = reference[sy * 40 + sx];
                            }
                        }
                        let mut actual = vec![91; expected.len()];
                        copy_replicated_block(&mut actual, stride, start, &reference, 40,
                            visible_width, visible_height, source_x, source_y, width, height);
                        assert_eq!(actual, expected,
                            "visible=({visible_width},{visible_height}) block=({width},{height}) origin=({source_x},{source_y})");
                    }
                }
            }
        }
    }

    #[test]
    fn centered_rows_match_wide_arithmetic_and_preserve_bounds() {
        for taps in [vec![(0, 128)], vec![(0, 77), (1, -16), (2, 77), (3, -10)],
            vec![(0, 192), (1, -64)], vec![(0, 193), (1, -65)],
            vec![(0, 63), (1, 64)]] {
            for width in [4, 8, 16, 24, 32, 64] {
                for stride in [1, 73] {
                    for seed in 0..256usize {
                        let length: usize = 1 + taps.last().unwrap().0 * stride + width;
                        let source: Vec<_> = (0..length).map(|index|
                            (index.wrapping_mul(73).wrapping_add(seed)) as u8).collect();
                        let mut expected = vec![91; width + 8];
                        let mut actual = expected.clone();
                        convolve_row_scalar(&source, 1, stride, &taps, &mut expected[4..4 + width]);
                        let mut coefficients = [0; 4];
                        for &(tap, coefficient) in &taps { coefficients[tap] = coefficient; }
                        convolve_prepared_row(&source, 1, stride, &ConvolutionFilter::new(coefficients), &mut actual[4..4 + width]);
                        assert_eq!(actual, expected, "width={width} stride={stride} seed={seed} taps={taps:?}");
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "manual paired row convolution timing"]
    fn benchmark_centered_rows() {
        let source: Vec<_> = (0..600usize).map(|index| index.wrapping_mul(73) as u8).collect();
        let filter = ConvolutionFilter::new([3, -16, 77, 77, -16, 3]);
        for width in [4, 8, 16, 32, 64] {
            let mut output = [0; 64];
            let mut measure = |narrow| {
                let start = std::time::Instant::now();
                for _ in 0..2_000_000 {
                    let source = std::hint::black_box(source.as_slice());
                    let filter = std::hint::black_box(&filter);
                    let taps = &filter.taps[..filter.count];
                    let output = std::hint::black_box(&mut output[..width]);
                    if narrow { convolve_prepared_row(source, 1, 73, filter, output); }
                    else { convolve_row(source, 1, 73, taps, output); }
                }
                start.elapsed()
            };
            for trial in 0..4 {
                let (old, new) = if trial % 2 == 0 { (measure(false), measure(true)) }
                    else { let new = measure(true); (measure(false), new) };
                eprintln!("convolution width={width}: wide={old:?} centered={new:?}");
            }
        }
    }
}
