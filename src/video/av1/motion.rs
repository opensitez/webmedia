//! Motion-vector entropy and separable inter prediction, AV1 5.11.31-32, 7.11.3.2-4.

use super::decoder::DecodedPlane;
use super::entropy::SymbolDecoder;
use super::motion_tables::SUBPEL_FILTERS;
use super::syntax::Error;

#[derive(Clone)]
pub(crate) struct MvCdfs {
    joint: [u16; 5],
    sign: [[u16; 3]; 2],
    class: [[u16; 12]; 2],
    class0_bit: [[u16; 3]; 2],
    bit: [[[u16; 3]; 10]; 2],
    class0_fraction: [[[u16; 5]; 2]; 2],
    fraction: [[u16; 5]; 2],
    class0_hp: [[u16; 3]; 2],
    hp: [[u16; 3]; 2],
}

impl Default for MvCdfs {
    fn default() -> Self {
        Self {
            joint: [4096, 11264, 19328, 32768, 0],
            sign: [[16384, 32768, 0]; 2],
            class: [[
                28672, 30976, 31858, 32320, 32551, 32656, 32740, 32757, 32762, 32767, 32768, 0,
            ]; 2],
            class0_bit: [[27648, 32768, 0]; 2],
            bit: [[136, 140, 148, 160, 176, 192, 224, 234, 234, 240].map(|p| [p * 128, 32768, 0]);
                2],
            class0_fraction: [[
                [16384, 24576, 26624, 32768, 0],
                [12288, 21248, 24128, 32768, 0],
            ]; 2],
            fraction: [[8192, 17408, 21248, 32768, 0]; 2],
            class0_hp: [[20480, 32768, 0]; 2],
            hp: [[16384, 32768, 0]; 2],
        }
    }
}

impl MvCdfs {
    pub(crate) fn reset_counts(&mut self) {
        self.joint[4] = 0;
        for c in 0..2 {
            self.sign[c][2] = 0;
            self.class[c][11] = 0;
            self.class0_bit[c][2] = 0;
            for row in &mut self.bit[c] {
                row[2] = 0;
            }
            for row in &mut self.class0_fraction[c] {
                row[4] = 0;
            }
            self.fraction[c][4] = 0;
            self.class0_hp[c][2] = 0;
            self.hp[c][2] = 0;
        }
    }

    pub(crate) fn read(
        &mut self,
        decoder: &mut SymbolDecoder<'_>,
        prediction: [i32; 2],
        integer: bool,
        high_precision: bool,
    ) -> Result<[i32; 2], Error> {
        let joint = decoder.read_symbol(&mut self.joint)?;
        let mut result = prediction;
        for c in 0..2 {
            if !(joint == 3 || (c == 0 && joint == 2) || (c == 1 && joint == 1)) {
                continue;
            }
            let negative = decoder.read_symbol(&mut self.sign[c])? != 0;
            let class = decoder.read_symbol(&mut self.class[c])?;
            let d = if class == 0 {
                decoder.read_symbol(&mut self.class0_bit[c])?
            } else {
                let mut d = 0;
                for i in 0..class {
                    d |= decoder.read_symbol(&mut self.bit[c][i])? << i;
                }
                d
            };
            let fraction = if integer {
                3
            } else if class == 0 {
                decoder.read_symbol(&mut self.class0_fraction[c][d])?
            } else {
                decoder.read_symbol(&mut self.fraction[c])?
            };
            let hp = if !high_precision {
                1
            } else if class == 0 {
                decoder.read_symbol(&mut self.class0_hp[c])?
            } else {
                decoder.read_symbol(&mut self.hp[c])?
            };
            let magnitude = (if class == 0 { 0 } else { 2 << (class + 2) })
                + ((d << 3) | (fraction << 1) | hp)
                + 1;
            result[c] += if negative {
                -(magnitude as i32)
            } else {
                magnitude as i32
            };
            if !(-16384..=16383).contains(&result[c]) {
                return Err(Error::Invalid("motion vector range"));
            }
        }
        Ok(result)
    }
}

pub(crate) fn lower_precision(mv: [i32; 2], integer: bool, high_precision: bool) -> [i32; 2] {
    mv.map(|v| {
        if integer {
            let rounded = (v.abs() + 3) >> 3;
            v.signum() * (rounded << 3)
        } else if !high_precision {
            v - v % 2
        } else {
            v
        }
    })
}

fn round_signed(value: i64, bits: u32) -> i64 {
    value.signum() * ((value.abs() + (1 << (bits - 1))) >> bits)
}

#[derive(Clone, Copy)]
pub(crate) struct Sampling {
    pub start: [i64; 2],
    pub step: [i64; 2],
}

/// Centers are projected before choosing the 1/16-sample filter phase.
pub(crate) fn sampling(
    origin: [usize; 2],
    mv: [i32; 2],
    subsampling: [bool; 2],
    frame: [u32; 2],
    reference: [u32; 2],
) -> Result<Sampling, Error> {
    let mut result = Sampling {
        start: [0; 2],
        step: [0; 2],
    };
    for axis in 0..2 {
        if frame[axis] == 0
            || reference[axis] == 0
            || 2 * u64::from(frame[axis]) < u64::from(reference[axis])
            || u64::from(frame[axis]) > 16 * u64::from(reference[axis])
        {
            return Err(Error::Invalid("reference scaling dimensions"));
        }
        let scale = ((i64::from(reference[axis]) << 14) + i64::from(frame[axis] / 2))
            / i64::from(frame[axis]);
        let original = (origin[axis] as i64 * 16)
            + ((2 * i64::from(mv[axis])) >> u32::from(subsampling[axis]))
            + 8;
        result.start[axis] = round_signed(original * scale - (8 << 14), 8) + 32;
        result.step[axis] = round_signed(scale, 4);
    }
    Ok(result)
}

/// Returns prediction in the precision used for compound blending (4 fractional bits at 8-bit).
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn predict(
    plane: &DecodedPlane,
    width: usize,
    height: usize,
    bit_depth: u8,
    location: Sampling,
    filters: [u8; 2],
    compound: bool,
) -> Result<(Vec<i32>, u32), Error> {
    let mut output = Vec::new();
    let mut intermediate = PredictionScratch::default();
    let post = predict_into(
        plane,
        width,
        height,
        bit_depth,
        location,
        filters,
        compound,
        std::env::var_os("AV1_DISABLE_NEON").is_none(),
        &mut output,
        &mut intermediate,
    )?;
    Ok((output, post))
}

#[derive(Default)]
pub(crate) struct PredictionScratch {
    wide: Vec<i32>,
    #[cfg(test)]
    narrow: Vec<i16>,
}

pub(crate) fn predict_into(
    plane: &DecodedPlane,
    width: usize,
    height: usize,
    bit_depth: u8,
    location: Sampling,
    filters: [u8; 2],
    compound: bool,
    use_neon: bool,
    output: &mut Vec<i32>,
    scratch: &mut PredictionScratch,
) -> Result<u32, Error> {
    let intermediate = &mut scratch.wide;
    #[cfg(test)]
    if MOTION_REFERENCE.with(|flag| flag.get()) {
        let (samples, post) =
            predict_reference(plane, width, height, bit_depth, location, filters, compound)?;
        *output = samples;
        return Ok(post);
    }
    #[cfg(test)]
    let _measure = super::profile::measure(0);
    if ![8, 10, 12].contains(&bit_depth)
        || width == 0
        || height == 0
        || width > 128
        || height > 128
        || filters.iter().any(|&f| f > 3)
        || plane.width == 0
        || plane.height == 0
        || plane.stride < plane.width
        || plane
            .height
            .checked_mul(plane.stride)
            .is_none_or(|n| n > plane.samples.len())
        || location.step.iter().any(|&s| !(64..=2048).contains(&s))
    {
        return Err(Error::Invalid("inter prediction inputs"));
    }
    let round0 = if bit_depth == 12 { 5 } else { 3 };
    let round1 = if compound { 7 } else { 14 - round0 };
    let post = 14 - round0 - round1;
    if location.step == [1024; 2] && location.start.iter().all(|&p| ((p >> 6) & 15) == 0) {
        let base_y = location.start[0] >> 10;
        let base_x = location.start[1] >> 10;
        output.clear();
        output.reserve(width * height);
        for row in 0..height {
            let y = (base_y + row as i64).clamp(0, plane.height as i64 - 1) as usize;
            for col in 0..width {
                let x = (base_x + col as i64).clamp(0, plane.width as i64 - 1) as usize;
                output.push(i32::from(plane.samples[y * plane.stride + x]) << post);
            }
        }
        return Ok(post);
    }
    let ih = (((height - 1) as i64 * location.step[0] + 1023) >> 10) as usize + 8;
    let filter = |f: u8, n: usize| -> usize {
        if n > 4 || f == 3 {
            f as usize
        } else if f == 1 {
            5
        } else {
            4
        }
    };
    let xf = filter(filters[1], width);
    let yf = filter(filters[0], height);
    let mut coordinates = [[0usize; 8]; 128];
    let mut horizontal_weights = [[0i16; 8]; 128];
    for col in 0..width {
        let p = location.start[1] + location.step[1] * col as i64;
        horizontal_weights[col] = SUBPEL_FILTERS[xf][((p >> 6) & 15) as usize];
        for tap in 0..8 {
            coordinates[col][tap] =
                ((p >> 10) + tap as i64 - 3).clamp(0, plane.width as i64 - 1) as usize;
        }
    }
    intermediate.resize(ih * width, 0);
    for row in 0..ih {
        let y =
            ((location.start[0] >> 10) + row as i64 - 3).clamp(0, plane.height as i64 - 1) as usize;
        let samples = &plane.samples[y * plane.stride..y * plane.stride + plane.width];
        let target = &mut intermediate[row * width..(row + 1) * width];
        let mut col = 0;
        #[cfg(target_arch = "aarch64")]
        if use_neon
            && location.step[1] == 1024
            && (location.start[1] >> 10) >= 3
            && (location.start[1] >> 10) + width as i64 + 4 <= plane.width as i64
        {
            let base = (location.start[1] >> 10) as usize - 3;
            while col + 8 <= width {
                // Full rows and their seven-sample tap halo were checked above.
                unsafe {
                    horizontal_neon_wide(
                        samples.as_ptr().add(base + col),
                        target.as_mut_ptr().add(col),
                        &horizontal_weights[0],
                        round0,
                    );
                }
                col += 8;
            }
        }
        for col in col..width {
            let mut sum = 0;
            for tap in 0..8 {
                let weight = horizontal_weights[col][tap];
                if weight != 0 {
                    sum += i32::from(weight) * i32::from(samples[coordinates[col][tap]]);
                }
            }
            target[col] = (sum + (1 << (round0 - 1))) >> round0;
        }
    }
    output.resize(width * height, 0);
    for row in 0..height {
        let p = (location.start[0] & 1023) + location.step[0] * row as i64;
        let weights = &SUBPEL_FILTERS[yf][((p >> 6) & 15) as usize];
        let top = (p >> 10) as usize * width;
        let mut col = 0;
        #[cfg(target_arch = "aarch64")]
        if use_neon {
            while col + 4 <= width {
                unsafe {
                    vertical_neon_wide(
                        intermediate.as_ptr().add(top + col),
                        width,
                        output.as_mut_ptr().add(row * width + col),
                        weights,
                        round1,
                    );
                }
                col += 4;
            }
        }
        for col in col..width {
            let mut sum = 0;
            for (tap, &weight) in weights.iter().enumerate() {
                if weight != 0 {
                    sum += i32::from(weight) * intermediate[top + tap * width + col];
                }
            }
            output[row * width + col] = (sum + (1 << (round1 - 1))) >> round1;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = use_neon;
    Ok(post)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn horizontal_neon_wide(
    source: *const u16,
    target: *mut i32,
    weights: &[i16; 8],
    round: u32,
) {
    use std::arch::aarch64::*;
    unsafe {
        let mut lo = vdupq_n_s32(0);
        let mut hi = lo;
        for (tap, &weight) in weights.iter().enumerate() {
            if weight != 0 {
                let values = vreinterpretq_s16_u16(vld1q_u16(source.add(tap)));
                lo = vmlal_n_s16(lo, vget_low_s16(values), weight);
                hi = vmlal_n_s16(hi, vget_high_s16(values), weight);
            }
        }
        let rounding = vdupq_n_s32(1 << (round - 1));
        let shift = vdupq_n_s32(-(round as i32));
        vst1q_s32(target, vshlq_s32(vaddq_s32(lo, rounding), shift));
        vst1q_s32(target.add(4), vshlq_s32(vaddq_s32(hi, rounding), shift));
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vertical_neon_wide(
    source: *const i32,
    stride: usize,
    target: *mut i32,
    weights: &[i16; 8],
    round: u32,
) {
    use std::arch::aarch64::*;
    unsafe {
        let mut sum = vdupq_n_s32(0);
        for (tap, &weight) in weights.iter().enumerate() {
            if weight != 0 {
                sum = vmlaq_n_s32(sum, vld1q_s32(source.add(tap * stride)), i32::from(weight));
            }
        }
        let rounded = vaddq_s32(sum, vdupq_n_s32(1 << (round - 1)));
        vst1q_s32(target, vshlq_s32(rounded, vdupq_n_s32(-(round as i32))));
    }
}

#[cfg(test)]
fn predict_into_narrow(
    plane: &DecodedPlane,
    width: usize,
    height: usize,
    bit_depth: u8,
    location: Sampling,
    filters: [u8; 2],
    compound: bool,
    use_neon: bool,
    output: &mut Vec<i32>,
    scratch: &mut PredictionScratch,
) -> Result<u32, Error> {
    let intermediate = &mut scratch.narrow;
    #[cfg(test)]
    if MOTION_REFERENCE.with(|flag| flag.get()) {
        let (samples, post) =
            predict_reference(plane, width, height, bit_depth, location, filters, compound)?;
        *output = samples;
        return Ok(post);
    }
    #[cfg(test)]
    let _measure = super::profile::measure(0);
    if ![8, 10, 12].contains(&bit_depth)
        || width == 0
        || height == 0
        || width > 128
        || height > 128
        || filters.iter().any(|&f| f > 3)
        || plane.width == 0
        || plane.height == 0
        || plane.stride < plane.width
        || plane
            .height
            .checked_mul(plane.stride)
            .is_none_or(|n| n > plane.samples.len())
        || location.step.iter().any(|&s| !(64..=2048).contains(&s))
    {
        return Err(Error::Invalid("inter prediction inputs"));
    }
    let round0 = if bit_depth == 12 { 5 } else { 3 };
    let round1 = if compound { 7 } else { 14 - round0 };
    let post = 14 - round0 - round1;
    if location.step == [1024; 2] && location.start.iter().all(|&p| ((p >> 6) & 15) == 0) {
        let base_y = location.start[0] >> 10;
        let base_x = location.start[1] >> 10;
        output.clear();
        output.reserve(width * height);
        for row in 0..height {
            let y = (base_y + row as i64).clamp(0, plane.height as i64 - 1) as usize;
            for col in 0..width {
                let x = (base_x + col as i64).clamp(0, plane.width as i64 - 1) as usize;
                output.push(i32::from(plane.samples[y * plane.stride + x]) << post);
            }
        }
        return Ok(post);
    }
    let ih = (((height - 1) as i64 * location.step[0] + 1023) >> 10) as usize + 8;
    let filter = |f: u8, n: usize| -> usize {
        if n > 4 || f == 3 {
            f as usize
        } else if f == 1 {
            5
        } else {
            4
        }
    };
    let xf = filter(filters[1], width);
    let yf = filter(filters[0], height);
    let mut coordinates = [[0usize; 8]; 128];
    let mut horizontal_weights = [[0i16; 8]; 128];
    for col in 0..width {
        let p = location.start[1] + location.step[1] * col as i64;
        horizontal_weights[col] = SUBPEL_FILTERS[xf][((p >> 6) & 15) as usize];
        for tap in 0..8 {
            coordinates[col][tap] =
                ((p >> 10) + tap as i64 - 3).clamp(0, plane.width as i64 - 1) as usize;
        }
    }
    intermediate.resize(ih * width, 0);
    for row in 0..ih {
        let y =
            ((location.start[0] >> 10) + row as i64 - 3).clamp(0, plane.height as i64 - 1) as usize;
        let samples = &plane.samples[y * plane.stride..y * plane.stride + plane.width];
        let target = &mut intermediate[row * width..(row + 1) * width];
        let mut col = 0;
        #[cfg(target_arch = "aarch64")]
        if use_neon
            && location.step[1] == 1024
            && (location.start[1] >> 10) >= 3
            && (location.start[1] >> 10) + width as i64 + 4 <= plane.width as i64
        {
            let base = (location.start[1] >> 10) as usize - 3;
            while col + 8 <= width {
                // Full rows and their seven-sample tap halo were checked above.
                unsafe {
                    horizontal_neon(
                        samples.as_ptr().add(base + col),
                        target.as_mut_ptr().add(col),
                        &horizontal_weights[0],
                        round0,
                    );
                }
                col += 8;
            }
        }
        for col in col..width {
            let mut sum = 0;
            for tap in 0..8 {
                let weight = horizontal_weights[col][tap];
                if weight != 0 {
                    sum += i32::from(weight) * i32::from(samples[coordinates[col][tap]]);
                }
            }
            // Normative tap sums are at most +184/-56. Decoded 8/10/12-bit
            // samples and round0 constrain this intermediate to signed 16 bits.
            target[col] = ((sum + (1 << (round0 - 1))) >> round0) as i16;
        }
    }
    output.resize(width * height, 0);
    for row in 0..height {
        let p = (location.start[0] & 1023) + location.step[0] * row as i64;
        let weights = &SUBPEL_FILTERS[yf][((p >> 6) & 15) as usize];
        let top = (p >> 10) as usize * width;
        let mut col = 0;
        #[cfg(target_arch = "aarch64")]
        if use_neon {
            while col + 8 <= width {
                unsafe {
                    vertical_neon(
                        intermediate.as_ptr().add(top + col),
                        width,
                        output.as_mut_ptr().add(row * width + col),
                        weights,
                        round1,
                    );
                }
                col += 8;
            }
        }
        for col in col..width {
            let mut sum = 0;
            for (tap, &weight) in weights.iter().enumerate() {
                if weight != 0 {
                    sum += i32::from(weight) * i32::from(intermediate[top + tap * width + col]);
                }
            }
            output[row * width + col] = (sum + (1 << (round1 - 1))) >> round1;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = use_neon;
    Ok(post)
}

#[cfg(all(test, target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn horizontal_neon(source: *const u16, target: *mut i16, weights: &[i16; 8], round: u32) {
    use std::arch::aarch64::*;
    unsafe {
        let mut lo = vdupq_n_s32(0);
        let mut hi = lo;
        for (tap, &weight) in weights.iter().enumerate() {
            if weight != 0 {
                let values = vreinterpretq_s16_u16(vld1q_u16(source.add(tap)));
                lo = vmlal_n_s16(lo, vget_low_s16(values), weight);
                hi = vmlal_n_s16(hi, vget_high_s16(values), weight);
            }
        }
        let rounding = vdupq_n_s32(1 << (round - 1));
        let shift = vdupq_n_s32(-(round as i32));
        let lo = vmovn_s32(vshlq_s32(vaddq_s32(lo, rounding), shift));
        let hi = vmovn_s32(vshlq_s32(vaddq_s32(hi, rounding), shift));
        vst1q_s16(target, vcombine_s16(lo, hi));
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn vertical_neon(
    source: *const i16,
    stride: usize,
    target: *mut i32,
    weights: &[i16; 8],
    round: u32,
) {
    use std::arch::aarch64::*;
    unsafe {
        let mut lo = vdupq_n_s32(0);
        let mut hi = lo;
        for (tap, &weight) in weights.iter().enumerate() {
            if weight != 0 {
                let values = vld1q_s16(source.add(tap * stride));
                lo = vmlal_n_s16(lo, vget_low_s16(values), weight);
                hi = vmlal_n_s16(hi, vget_high_s16(values), weight);
            }
        }
        let rounding = vdupq_n_s32(1 << (round - 1));
        let shift = vdupq_n_s32(-(round as i32));
        vst1q_s32(target, vshlq_s32(vaddq_s32(lo, rounding), shift));
        vst1q_s32(target.add(4), vshlq_s32(vaddq_s32(hi, rounding), shift));
    }
}

#[cfg(test)]
std::thread_local! {
    static MOTION_REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
pub(crate) fn set_motion_reference(enabled: bool) {
    MOTION_REFERENCE.with(|flag| flag.set(enabled));
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn predict_reference(
    plane: &DecodedPlane,
    width: usize,
    height: usize,
    bit_depth: u8,
    location: Sampling,
    filters: [u8; 2],
    compound: bool,
) -> Result<(Vec<i32>, u32), Error> {
    #[cfg(test)]
    let _measure = super::profile::measure(0);
    if ![8, 10, 12].contains(&bit_depth)
        || width == 0
        || height == 0
        || width > 128
        || height > 128
        || filters.iter().any(|&f| f > 3)
        || plane.width == 0
        || plane.height == 0
        || plane.stride < plane.width
        || plane
            .height
            .checked_mul(plane.stride)
            .is_none_or(|n| n > plane.samples.len())
        || location.step.iter().any(|&s| !(64..=2048).contains(&s))
    {
        return Err(Error::Invalid("inter prediction inputs"));
    }
    let round0 = if bit_depth == 12 { 5 } else { 3 };
    let round1 = if compound { 7 } else { 14 - round0 };
    let post = 14 - round0 - round1;
    if location.step == [1024; 2] && location.start.iter().all(|&p| ((p >> 6) & 15) == 0) {
        let base_y = location.start[0] >> 10;
        let base_x = location.start[1] >> 10;
        let mut output = Vec::with_capacity(width * height);
        for row in 0..height {
            let y = (base_y + row as i64).clamp(0, plane.height as i64 - 1) as usize;
            for col in 0..width {
                let x = (base_x + col as i64).clamp(0, plane.width as i64 - 1) as usize;
                output.push(i32::from(plane.samples[y * plane.stride + x]) << post);
            }
        }
        return Ok((output, post));
    }
    let ih = (((height - 1) as i64 * location.step[0] + 1023) >> 10) as usize + 8;
    let filter = |f: u8, n: usize| -> usize {
        if n > 4 || f == 3 {
            f as usize
        } else if f == 1 {
            5
        } else {
            4
        }
    };
    let xf = filter(filters[1], width);
    let yf = filter(filters[0], height);
    let mut intermediate = vec![0i32; ih * width];
    for row in 0..ih {
        let y =
            ((location.start[0] >> 10) + row as i64 - 3).clamp(0, plane.height as i64 - 1) as usize;
        for col in 0..width {
            let p = location.start[1] + location.step[1] * col as i64;
            let weights = &SUBPEL_FILTERS[xf][((p >> 6) & 15) as usize];
            let mut sum = 0;
            for (tap, &weight) in weights.iter().enumerate() {
                let x = ((p >> 10) + tap as i64 - 3).clamp(0, plane.width as i64 - 1) as usize;
                sum += i32::from(weight) * i32::from(plane.samples[y * plane.stride + x]);
            }
            intermediate[row * width + col] = (sum + (1 << (round0 - 1))) >> round0;
        }
    }
    let mut output = vec![0; width * height];
    for row in 0..height {
        let p = (location.start[0] & 1023) + location.step[0] * row as i64;
        let weights = &SUBPEL_FILTERS[yf][((p >> 6) & 15) as usize];
        for col in 0..width {
            let mut sum = 0;
            for (tap, &weight) in weights.iter().enumerate() {
                sum += i32::from(weight) * intermediate[((p >> 10) as usize + tap) * width + col];
            }
            output[row * width + col] = (sum + (1 << (round1 - 1))) >> round1;
        }
    }
    Ok((output, post))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normative_horizontal_intermediates_fit_signed_sixteen_bits() {
        for bit_depth in [8, 10, 12] {
            let max = (1_i32 << bit_depth) - 1;
            let round = if bit_depth == 12 { 5 } else { 3 };
            for family in SUBPEL_FILTERS {
                for weights in family {
                    let positive: i32 = weights.iter().map(|&w| i32::from(w.max(0))).sum();
                    let negative: i32 = weights.iter().map(|&w| i32::from(w.min(0))).sum();
                    assert!(positive <= 184 && negative >= -56);
                    for extreme in [positive, negative] {
                        let value = (extreme * max + (1 << (round - 1))) >> round;
                        assert!(i16::try_from(value).is_ok());
                    }
                }
            }
        }
    }

    #[test]
    fn reusable_prediction_buffers_match_allocating_reference() {
        let mut output = Vec::with_capacity(128 * 128);
        let mut intermediate = PredictionScratch {
            narrow: Vec::with_capacity(40000),
            wide: Vec::with_capacity(40000),
        };
        let output_pointer = output.as_ptr();
        let intermediate_pointer = intermediate.wide.as_ptr();
        for bit_depth in [8, 10, 12] {
            let plane = DecodedPlane {
                width: 71,
                height: 67,
                stride: 79,
                samples: (0..79 * 67)
                    .map(|i| ((i * 173 + 29) & ((1 << bit_depth) - 1)) as u16)
                    .collect(),
            };
            for use_neon in [false, true] {
                for (width, height) in [(128, 128), (4, 4), (17, 11), (64, 32), (8, 8)] {
                    for compound in [false, true] {
                        for location in [
                            Sampling {
                                start: [8192, 8192],
                                step: [1024; 2],
                            },
                            Sampling {
                                start: [8704, 8384],
                                step: [1024; 2],
                            },
                            Sampling {
                                start: [-128, 65024],
                                step: [2048, 768],
                            },
                        ] {
                            output.fill(i32::MIN);
                            intermediate.wide.fill(i32::MAX);
                            let expected = predict_reference(
                                &plane,
                                width,
                                height,
                                bit_depth,
                                location,
                                [1, 2],
                                compound,
                            )
                            .unwrap();
                            let post = predict_into(
                                &plane,
                                width,
                                height,
                                bit_depth,
                                location,
                                [1, 2],
                                compound,
                                use_neon,
                                &mut output,
                                &mut intermediate,
                            )
                            .unwrap();
                            assert_eq!((&output, post), (&expected.0, expected.1));
                            assert_eq!(output.as_ptr(), output_pointer);
                            assert_eq!(intermediate.wide.as_ptr(), intermediate_pointer);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cached_scalar_and_neon_interpolation_match_original() {
        let mut state = 511u32;
        let mut scratch = PredictionScratch::default();
        let mut output = Vec::new();
        for bit_depth in [8, 10, 12] {
            let plane = DecodedPlane {
                width: 64,
                height: 64,
                stride: 64,
                samples: (0..64 * 64)
                    .map(|_| {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        ((state >> 16) & ((1 << bit_depth) - 1)) as u16
                    })
                    .collect(),
            };
            for compound in [false, true] {
                for (width, height) in [(4, 4), (8, 8), (17, 11), (32, 16)] {
                    for filters in [[0, 0], [0, 1], [1, 2], [2, 3], [3, 0], [3, 3]] {
                        for phase in 0..16 {
                            for origin in [-3, 8, 62] {
                                for step in [[1024; 2], [1025, 768], [2048, 64]] {
                                    let location = Sampling {
                                        start: [
                                            origin * 1024 + phase * 64,
                                            origin * 1024 + ((phase * 7) % 16) * 64,
                                        ],
                                        step,
                                    };
                                    let expected = predict_reference(
                                        &plane, width, height, bit_depth, location, filters,
                                        compound,
                                    )
                                    .unwrap();
                                    let actual = predict(
                                        &plane, width, height, bit_depth, location, filters,
                                        compound,
                                    )
                                    .unwrap();
                                    assert_eq!(
                                        actual, expected,
                                        "depth={bit_depth} compound={compound} size={width}x{height} filter={filters:?} phase={phase} origin={origin} step={step:?}"
                                    );
                                    for use_neon in [false, true] {
                                        let post = predict_into_narrow(
                                            &plane,
                                            width,
                                            height,
                                            bit_depth,
                                            location,
                                            filters,
                                            compound,
                                            use_neon,
                                            &mut output,
                                            &mut scratch,
                                        )
                                        .unwrap();
                                        assert_eq!((&output, post), (&expected.0, expected.1));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn integer_phase_copy_matches_two_pass_filtering_at_every_precision() {
        for family in SUBPEL_FILTERS {
            assert_eq!(family[0], [0, 0, 0, 128, 0, 0, 0, 0]);
        }
        for depth in [8, 10, 12] {
            let plane = DecodedPlane {
                width: 16,
                height: 16,
                stride: 16,
                samples: (0..256)
                    .map(|i| ((i * 73 + 71) & ((1 << depth) - 1)) as u16)
                    .collect(),
            };
            for size in [4, 8, 64] {
                for start in [-2048, 0, 7168] {
                    for filter in 0..4 {
                        for compound in [false, true] {
                            let fast = predict(
                                &plane,
                                size,
                                size,
                                depth,
                                Sampling {
                                    start: [start; 2],
                                    step: [1024; 2],
                                },
                                [filter; 2],
                                compound,
                            )
                            .unwrap();
                            // A one-unit scale increment stays in phase zero over this block,
                            // but forces the general two-pass implementation.
                            let general = predict(
                                &plane,
                                size,
                                size,
                                depth,
                                Sampling {
                                    start: [start; 2],
                                    step: [1025; 2],
                                },
                                [filter; 2],
                                compound,
                            )
                            .unwrap();
                            assert_eq!(fast, general);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn all_normative_filter_phases_preserve_dc() {
        for family in SUBPEL_FILTERS {
            for phase in family {
                assert_eq!(phase.iter().map(|&v| i32::from(v)).sum::<i32>(), 128);
            }
        }
        for depth in [8, 10, 12] {
            let value = (1 << depth) - 7;
            let plane = DecodedPlane {
                width: 8,
                height: 8,
                stride: 8,
                samples: vec![value; 64],
            };
            for f in 0..4 {
                for phase in 0..16 {
                    for compound in [false, true] {
                        let (samples, bits) = predict(
                            &plane,
                            4,
                            8,
                            depth,
                            Sampling {
                                start: [phase * 64, phase * 64],
                                step: [1024; 2],
                            },
                            [f; 2],
                            compound,
                        )
                        .unwrap();
                        assert!(samples.iter().all(|&v| v == i32::from(value) << bits));
                    }
                }
            }
        }
    }

    #[test]
    fn integer_motion_and_border_extension() {
        let plane = DecodedPlane {
            width: 8,
            height: 8,
            stride: 8,
            samples: (0..64).collect(),
        };
        let location = sampling([2, 2], [8, -8], [false; 2], [8; 2], [8; 2]).unwrap();
        let (samples, bits) = predict(&plane, 4, 4, 8, location, [0; 2], false).unwrap();
        assert_eq!(bits, 0);
        for row in 0..4 {
            for col in 0..4 {
                assert_eq!(
                    samples[row * 4 + col],
                    plane.samples[(row + 3) * 8 + col + 1] as i32
                );
            }
        }
        let location = sampling([0, 0], [-32, -32], [false; 2], [8; 2], [8; 2]).unwrap();
        assert_eq!(
            predict(&plane, 4, 4, 8, location, [2; 2], false).unwrap().0,
            vec![0; 16]
        );
    }

    #[test]
    fn motion_precision_signed_ties() {
        assert_eq!(lower_precision([4, -4], true, false), [0, 0]);
        assert_eq!(lower_precision([5, -5], true, false), [8, -8]);
        assert_eq!(lower_precision([7, -7], false, false), [6, -6]);
        assert_eq!(lower_precision([7, -7], false, true), [7, -7]);
    }
}
