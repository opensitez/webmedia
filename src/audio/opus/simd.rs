//! Independent coefficient scaling, with scalar and AArch64 NEON paths.
//! Reductions deliberately retain their original scalar summation order.

#[inline]
pub fn scale_copy_scalar(output: &mut [f64], input: &[f64], gain: f64) {
    assert_eq!(output.len(), input.len());
    for (output, input) in output.iter_mut().zip(input) {
        *output = *input * gain;
    }
}

#[inline]
pub fn scale_in_place_scalar(values: &mut [f64], gain: f64) {
    for value in values {
        *value *= gain;
    }
}

#[inline]
pub fn scale_copy(output: &mut [f64], input: &[f64], gain: f64) {
    assert_eq!(output.len(), input.len());
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is baseline AArch64; equal-length slices bound all accesses.
        unsafe { scale_neon(output.as_mut_ptr(), input.as_ptr(), input.len(), gain) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    scale_copy_scalar(output, input, gain);
}

#[inline]
pub fn scale_in_place(values: &mut [f64], gain: f64) {
    #[cfg(target_arch = "aarch64")]
    {
        // The kernel permits exact in-place aliasing and never crosses a tail.
        unsafe { scale_neon(values.as_mut_ptr(), values.as_ptr(), values.len(), gain) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    scale_in_place_scalar(values, gain);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_neon(output: *mut f64, input: *const f64, count: usize, gain: f64) {
    use std::arch::aarch64::{vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64};
    let multiplier = vdupq_n_f64(gain);
    let mut index = 0;
    // Each load/store covers two valid elements. Source and destination must
    // either be disjoint or identical; both safe wrappers enforce that.
    unsafe {
        while count - index >= 8 {
            let a = vld1q_f64(input.add(index));
            let b = vld1q_f64(input.add(index + 2));
            let c = vld1q_f64(input.add(index + 4));
            let d = vld1q_f64(input.add(index + 6));
            vst1q_f64(output.add(index), vmulq_f64(a, multiplier));
            vst1q_f64(output.add(index + 2), vmulq_f64(b, multiplier));
            vst1q_f64(output.add(index + 4), vmulq_f64(c, multiplier));
            vst1q_f64(output.add(index + 6), vmulq_f64(d, multiplier));
            index += 8;
        }
        while count - index >= 2 {
            vst1q_f64(
                output.add(index),
                vmulq_f64(vld1q_f64(input.add(index)), multiplier),
            );
            index += 2;
        }
        if index < count {
            *output.add(index) = *input.add(index) * gain;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_and_scalar_scaling_match_bits_and_preserve_guards() {
        let source: Vec<_> = (0..963)
            .map(|index| match index % 9 {
                0 => -0.0,
                1 => f64::from_bits(1),
                2 => f64::MIN_POSITIVE,
                3 => f64::MAX,
                _ => (index as f64 - 480.0) / 17.0,
            })
            .collect();
        for length in 0..=960 {
            for gain in [-3.0, -0.0, 0.125, 1.0, 32768.0] {
                let input = &source[1..1 + length];
                let mut expected = vec![123.0; length + 6];
                let mut actual = expected.clone();
                scale_copy_scalar(&mut expected[3..3 + length], input, gain);
                scale_copy(&mut actual[3..3 + length], input, gain);
                assert!(
                    expected
                        .iter()
                        .zip(&actual)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                let mut expected = input.to_vec();
                let mut actual = expected.clone();
                scale_in_place_scalar(&mut expected, gain);
                scale_in_place(&mut actual, gain);
                assert!(
                    expected
                        .iter()
                        .zip(&actual)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
        }
    }

    #[test]
    #[ignore = "manual scalar/native coefficient throughput measurement"]
    fn scalar_native_scaling_throughput() {
        use std::{hint::black_box, time::Instant};
        let input = vec![0.125; 800];
        let mut output = vec![0.0; input.len()];
        for native in [false, true, false, true] {
            let began = Instant::now();
            for _ in 0..200_000 {
                if native {
                    scale_copy(black_box(&mut output), black_box(&input), black_box(1.25));
                } else {
                    scale_copy_scalar(black_box(&mut output), black_box(&input), black_box(1.25));
                }
            }
            black_box(&output);
            eprintln!(
                "coefficient scaling native={native} samples=160000000 seconds={}",
                began.elapsed().as_secs_f64()
            );
        }
    }
}
