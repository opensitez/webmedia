//! Seven-bit fixed-point row convolution shared by VP8 and VP9.

pub(super) fn convolve_row(source: &[u8], origin: usize, stride: usize,
    taps: &[(usize, i32)], output: &mut [u8])
{
    if let [(tap, 128)] = taps {
        let start = origin + tap * stride;
        output.copy_from_slice(&source[start..start + output.len()]);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if output.len() == 4 {
        unsafe { convolve_four_neon(source, origin, stride, taps, output); }
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
