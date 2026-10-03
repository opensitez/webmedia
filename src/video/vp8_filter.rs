//! VP8 in-loop deblocking (RFC 6386, section 15).

use super::vp8_predict::Plane;

#[derive(Clone, Copy)]
pub(super) struct FilterMacroblock {
    pub level: u8,
    pub skip_inner: bool,
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct FilterStrength {
    interior: i32,
    hev: i32,
    macroblock: i32,
    subblock: i32,
}

impl FilterStrength {
    const fn new(level: u8, sharpness: u8, keyframe: bool) -> Self {
        let level = level as i32;
        let mut interior = level;
        if sharpness != 0 {
            interior >>= if sharpness > 4 { 2 } else { 1 };
            let limit = 9 - sharpness as i32;
            if interior > limit { interior = limit; }
        }
        if interior < 1 { interior = 1; }
        let hev = if keyframe {
            if level >= 40 { 2 } else if level >= 15 { 1 } else { 0 }
        } else if level >= 40 { 3 } else if level >= 20 { 2 }
            else if level >= 15 { 1 } else { 0 };
        Self { interior, hev, macroblock: level * 2 + 4 + interior,
            subblock: level * 2 + interior }
    }
}

static FILTER_STRENGTHS: [[[FilterStrength; 64]; 8]; 2] = {
    let zero = FilterStrength { interior: 0, hev: 0, macroblock: 0, subblock: 0 };
    let mut table = [[[zero; 64]; 8]; 2];
    let mut frame = 0;
    while frame < 2 {
        let mut sharpness = 0;
        while sharpness < 8 {
            let mut level = 0;
            while level < 64 {
                table[frame][sharpness][level] = FilterStrength::new(level as u8,
                    sharpness as u8, frame != 0);
                level += 1;
            }
            sharpness += 1;
        }
        frame += 1;
    }
    table
};

pub(super) fn filter_frame(
    y: &mut Plane,
    u: &mut Plane,
    v: &mut Plane,
    mb_width: usize,
    mb_height: usize,
    settings: &[FilterMacroblock],
    sharpness: u8,
    simple: bool,
    keyframe: bool,
) {
    #[cfg(test)]
    if std::env::var_os("WEBMEDIA_VP8_SKIP_FILTER").is_some() { return; }
    let strengths = FILTER_STRENGTHS[usize::from(keyframe)].get(usize::from(sharpness));
    for mb_y in 0..mb_height {
        for mb_x in 0..mb_width {
            let setting = settings[mb_y * mb_width + mb_x];
            if setting.level == 0 {
                continue;
            }
            let strength = strengths.and_then(|levels| levels.get(usize::from(setting.level)))
                .copied().unwrap_or_else(|| FilterStrength::new(setting.level, sharpness, keyframe));
            let interior = strength.interior;
            let hev = strength.hev;
            for (plane, size) in [(&mut *y, 16), (&mut *u, 8), (&mut *v, 8)] {
                if simple && size == 8 {
                    continue;
                }
                let x = mb_x * size;
                let y = mb_y * size;
                let row_stride = plane.width;
                if mb_x != 0 {
                    filter_edge(
                        plane,
                        x,
                        y,
                        size,
                        1,
                        strength.macroblock,
                        interior,
                        hev,
                        simple,
                        true,
                    );
                }
                if !setting.skip_inner {
                    for offset in (4..size).step_by(4) {
                        filter_edge(
                            plane,
                            x + offset,
                            y,
                            size,
                            1,
                            strength.subblock,
                            interior,
                            hev,
                            simple,
                            false,
                        );
                    }
                }
                if mb_y != 0 {
                    filter_edge(
                        plane,
                        x,
                        y,
                        size,
                        row_stride,
                        strength.macroblock,
                        interior,
                        hev,
                        simple,
                        true,
                    );
                }
                if !setting.skip_inner {
                    for offset in (4..size).step_by(4) {
                        filter_edge(
                            plane,
                            x,
                            y + offset,
                            size,
                            row_stride,
                            strength.subblock,
                            interior,
                            hev,
                            simple,
                            false,
                        );
                    }
                }
            }
        }
    }
}

fn filter_edge(
    plane: &mut Plane,
    x: usize,
    y: usize,
    count: usize,
    step: usize,
    edge_limit: i32,
    interior_limit: i32,
    hev_limit: i32,
    simple: bool,
    macroblock_edge: bool,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if step == 1 && count != 0 && count.is_multiple_of(8) && x >= 4
            && x.checked_add(4).is_some_and(|end| end <= plane.width)
            && y.checked_add(count).is_some_and(|end| end <= plane.pixels.len() / plane.width)
        {
            unsafe { filter_vertical_neon(plane, x, y, count, edge_limit, interior_limit,
                hev_limit, simple, macroblock_edge); }
            return;
        }
        let taps = if simple { 2 } else { 4 };
        if step == plane.width && step > 1 && count != 0 && count.is_multiple_of(8)
            && x.checked_add(count).is_some_and(|end| end <= plane.width)
            && y >= taps && y.checked_add(taps).is_some_and(|end| end <= plane.pixels.len() / plane.width)
        {
            // Horizontal columns are independent; preserve edge ordering between calls.
            unsafe { filter_horizontal_neon(plane, x, y, count, edge_limit, interior_limit,
                hev_limit, simple, macroblock_edge); }
            return;
        }
    }
    filter_edge_scalar(plane, x, y, count, step, edge_limit, interior_limit, hev_limit,
        simple, macroblock_edge);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn filter_horizontal_neon(plane: &mut Plane, x: usize, y: usize, count: usize,
    edge_limit: i32, interior_limit: i32, hev_limit: i32, simple: bool, macroblock_edge: bool,
) {
    use std::arch::aarch64::*;
    for offset in (0..count).step_by(8) {
        let pos = y * plane.width + x + offset;
        let zero = vdupq_n_u16(0);
        let mut p = [zero; 4];
        let mut q = [zero; 4];
        for i in 0..2 {
            let before = pos - (i + 1) * plane.width;
            let after = pos + i * plane.width;
            p[i] = vmovl_u8(unsafe { vld1_u8(plane.pixels[before..before + 8].as_ptr()) });
            q[i] = vmovl_u8(unsafe { vld1_u8(plane.pixels[after..after + 8].as_ptr()) });
        }
        let result = unsafe { filter_eight_neon(p, q, edge_limit, interior_limit, hev_limit,
            simple, macroblock_edge, || {
                let mut outer_p = [zero; 2];
                let mut outer_q = [zero; 2];
                for i in 0..2 {
                    let before = pos - (i + 3) * plane.width;
                    let after = pos + (i + 2) * plane.width;
                    outer_p[i] = vmovl_u8(vld1_u8(plane.pixels[before..before + 8].as_ptr()));
                    outer_q[i] = vmovl_u8(vld1_u8(plane.pixels[after..after + 8].as_ptr()));
                }
                (outer_p, outer_q)
            }) };
        let Some((out_p, out_q, stored)) = result else { continue; };
        for i in 0..stored {
            let before = pos - (i + 1) * plane.width;
            let after = pos + i * plane.width;
            unsafe {
                vst1_u8(plane.pixels[before..before + 8].as_mut_ptr(), vmovn_u16(out_p[i]));
                vst1_u8(plane.pixels[after..after + 8].as_mut_ptr(), vmovn_u16(out_q[i]));
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn transpose_eight_neon(rows: [std::arch::aarch64::uint8x8_t; 8])
    -> [std::arch::aarch64::uint8x8_t; 8]
{
    use std::arch::aarch64::*;
    let mut pairs = [vdup_n_u8(0); 8];
    for i in 0..4 {
        pairs[i * 2] = vtrn1_u8(rows[i * 2], rows[i * 2 + 1]);
        pairs[i * 2 + 1] = vtrn2_u8(rows[i * 2], rows[i * 2 + 1]);
    }
    let mut quads = [vdup_n_u16(0); 8];
    for half in [0, 4] {
        for odd in 0..2 {
            let a = vreinterpret_u16_u8(pairs[half + odd]);
            let b = vreinterpret_u16_u8(pairs[half + 2 + odd]);
            quads[half + odd] = vtrn1_u16(a, b);
            quads[half + 2 + odd] = vtrn2_u16(a, b);
        }
    }
    let mut columns = [vdup_n_u8(0); 8];
    for i in 0..4 {
        let a = vreinterpret_u32_u16(quads[i]);
        let b = vreinterpret_u32_u16(quads[i + 4]);
        columns[i] = vreinterpret_u8_u32(vtrn1_u32(a, b));
        columns[i + 4] = vreinterpret_u8_u32(vtrn2_u32(a, b));
    }
    columns
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn filter_vertical_neon(plane: &mut Plane, x: usize, y: usize, count: usize,
    edge_limit: i32, interior_limit: i32, hev_limit: i32, simple: bool, macroblock_edge: bool,
) {
    use std::arch::aarch64::*;
    for offset in (0..count).step_by(8) {
        let pos = (y + offset) * plane.width + x - 4;
        let mut rows = [vdup_n_u8(0); 8];
        for (i, row) in rows.iter_mut().enumerate() {
            let start = pos + i * plane.width;
            *row = unsafe { vld1_u8(plane.pixels[start..start + 8].as_ptr()) };
        }
        // Transpose eight bounded row loads so each lane is an independent edge.
        let mut columns = unsafe { transpose_eight_neon(rows) };
        let zero = vdupq_n_u16(0);
        let p = [vmovl_u8(columns[3]), vmovl_u8(columns[2]), zero, zero];
        let q = [vmovl_u8(columns[4]), vmovl_u8(columns[5]), zero, zero];
        let result = unsafe { filter_eight_neon(p, q, edge_limit, interior_limit, hev_limit,
            simple, macroblock_edge, || {
                ([vmovl_u8(columns[1]), vmovl_u8(columns[0])],
                 [vmovl_u8(columns[6]), vmovl_u8(columns[7])])
            }) };
        let Some((out_p, out_q, stored)) = result else { continue; };
        for i in 0..stored {
            columns[3 - i] = vmovn_u16(out_p[i]);
            columns[4 + i] = vmovn_u16(out_q[i]);
        }
        let rows = unsafe { transpose_eight_neon(columns) };
        for (i, row) in rows.into_iter().enumerate() {
            let start = pos + i * plane.width;
            unsafe { vst1_u8(plane.pixels[start..start + 8].as_mut_ptr(), row); }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn filter_eight_neon(
    mut p: [std::arch::aarch64::uint16x8_t; 4],
    mut q: [std::arch::aarch64::uint16x8_t; 4],
    edge_limit: i32, interior_limit: i32, hev_limit: i32, simple: bool, macroblock_edge: bool,
    load_outer: impl FnOnce() -> ([std::arch::aarch64::uint16x8_t; 2], [std::arch::aarch64::uint16x8_t; 2]),
) -> Option<([std::arch::aarch64::uint16x8_t; 4], [std::arch::aarch64::uint16x8_t; 4], usize)> {
    use std::arch::aarch64::*;
    let clamp = |value| vminq_s16(vmaxq_s16(value, vdupq_n_s16(-128)), vdupq_n_s16(127));
    let signed = |value| vreinterpretq_s16_u16(value);
    let clipped = |value| vmovl_u8(vqmovun_s16(value));
    let across = vaddq_u16(vshlq_n_u16::<1>(vabdq_u16(p[0], q[0])),
        vshrq_n_u16::<1>(vabdq_u16(p[1], q[1])));
    let mut mask = vcleq_u16(across, vdupq_n_u16(edge_limit as u16));
    if vmaxvq_u16(mask) == 0 { return None; }
    let nearest = vmaxq_u16(vabdq_u16(p[1], p[0]), vabdq_u16(q[1], q[0]));
    let hev = if simple { vdupq_n_u16(u16::MAX) }
        else { vcgtq_u16(nearest, vdupq_n_u16(hev_limit as u16)) };
    if !simple {
        mask = vandq_u16(mask, vcleq_u16(nearest, vdupq_n_u16(interior_limit as u16)));
        if vmaxvq_u16(mask) == 0 { return None; }
        let mut adjacent = nearest;
        let (outer_p, outer_q) = load_outer();
        p[2..].copy_from_slice(&outer_p);
        q[2..].copy_from_slice(&outer_q);
        for i in 2..4 {
            adjacent = vmaxq_u16(adjacent, vmaxq_u16(vabdq_u16(p[i], p[i - 1]),
                vabdq_u16(q[i], q[i - 1])));
        }
        mask = vandq_u16(mask, vcleq_u16(adjacent, vdupq_n_u16(interior_limit as u16)));
        if vmaxvq_u16(mask) == 0 { return None; }
    }
    let extra = clamp(vsubq_s16(signed(p[1]), signed(q[1])));
    let delta = vmulq_n_s16(vsubq_s16(signed(q[0]), signed(p[0])), 3);
    let filter = clamp(vaddq_s16(vbslq_s16(hev, extra, vdupq_n_s16(0)), delta));
    let before = vshrq_n_s16::<3>(clamp(vaddq_s16(filter, vdupq_n_s16(3))));
    let after = vshrq_n_s16::<3>(clamp(vaddq_s16(filter, vdupq_n_s16(4))));
    let mut out_p = p;
    let mut out_q = q;
    out_p[0] = vbslq_u16(mask, clipped(vaddq_s16(signed(p[0]), before)), p[0]);
    out_q[0] = vbslq_u16(mask, clipped(vsubq_s16(signed(q[0]), after)), q[0]);
    let mut stored = 1;
    if !simple {
        let smooth = vbicq_u16(mask, hev);
        if macroblock_edge {
            let wide = clamp(vaddq_s16(extra, delta));
            for (i, multiplier) in [(0, 27), (1, 18), (2, 9)] {
                let adjustment = vshrq_n_s16::<7>(vaddq_s16(vmulq_n_s16(wide, multiplier), vdupq_n_s16(63)));
                out_p[i] = vbslq_u16(smooth, clipped(vaddq_s16(signed(p[i]), adjustment)), out_p[i]);
                out_q[i] = vbslq_u16(smooth, clipped(vsubq_s16(signed(q[i]), adjustment)), out_q[i]);
            }
            stored = 3;
        } else {
            let half = vshrq_n_s16::<1>(vaddq_s16(after, vdupq_n_s16(1)));
            out_p[1] = vbslq_u16(smooth, clipped(vaddq_s16(signed(p[1]), half)), p[1]);
            out_q[1] = vbslq_u16(smooth, clipped(vsubq_s16(signed(q[1]), half)), q[1]);
            stored = 2;
        }
    }
    Some((out_p, out_q, stored))
}

fn filter_edge_scalar(
    plane: &mut Plane,
    x: usize,
    y: usize,
    count: usize,
    step: usize,
    edge_limit: i32,
    interior_limit: i32,
    hev_limit: i32,
    simple: bool,
    macroblock_edge: bool,
) {
    for offset in 0..count {
        let base = y * plane.width
            + x
            + if step == 1 {
                offset * plane.width
            } else {
                offset
            };
        let mut p = [
            i32::from(plane.pixels[base - step]),
            i32::from(plane.pixels[base - 2 * step]),
            0,
            0,
        ];
        let mut q = [
            i32::from(plane.pixels[base]),
            i32::from(plane.pixels[base + step]),
            0,
            0,
        ];
        if 2 * (p[0] - q[0]).abs() + (p[1] - q[1]).abs() / 2 > edge_limit {
            continue;
        }
        if simple {
            common_adjust(plane, base, step, p, q, true);
            continue;
        }
        if (p[0] - p[1]).abs() > interior_limit || (q[0] - q[1]).abs() > interior_limit {
            continue;
        }
        p[2] = i32::from(plane.pixels[base - 3 * step]);
        p[3] = i32::from(plane.pixels[base - 4 * step]);
        q[2] = i32::from(plane.pixels[base + 2 * step]);
        q[3] = i32::from(plane.pixels[base + 3 * step]);
        if (p[1] - p[2]).abs() > interior_limit
            || (p[2] - p[3]).abs() > interior_limit
            || (q[1] - q[2]).abs() > interior_limit
            || (q[2] - q[3]).abs() > interior_limit
        {
            continue;
        }
        let high_variance = (p[1] - p[0]).abs() > hev_limit || (q[1] - q[0]).abs() > hev_limit;
        if high_variance {
            common_adjust(plane, base, step, p, q, true);
        } else if macroblock_edge {
            let w = signed_clamp(signed_clamp(p[1] - q[1]) + 3 * (q[0] - p[0]));
            for (distance, multiplier) in [(0, 27), (1, 18), (2, 9)] {
                let adjustment = signed_clamp((multiplier * w + 63) >> 7);
                write(
                    plane,
                    base - (distance + 1) * step,
                    p[distance] + adjustment,
                );
                write(plane, base + distance * step, q[distance] - adjustment);
            }
        } else {
            let adjustment = common_adjust(plane, base, step, p, q, false);
            let inner = (adjustment + 1) >> 1;
            write(plane, base - 2 * step, p[1] + inner);
            write(plane, base + step, q[1] - inner);
        }
    }
}

fn common_adjust(
    plane: &mut Plane,
    base: usize,
    step: usize,
    p: [i32; 4],
    q: [i32; 4],
    outer: bool,
) -> i32 {
    let extra = if outer { signed_clamp(p[1] - q[1]) } else { 0 };
    let a = signed_clamp(extra + 3 * (q[0] - p[0]));
    let before = signed_clamp(a + 3) >> 3;
    let after = signed_clamp(a + 4) >> 3;
    write(plane, base - step, p[0] + before);
    write(plane, base, q[0] - after);
    after
}

fn signed_clamp(value: i32) -> i32 {
    value.clamp(-128, 127)
}

fn write(plane: &mut Plane, index: usize, value: i32) {
    plane.pixels[index] = value.clamp(0, 255) as u8;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizontal_filter_matches_scalar_for_all_strengths() {
        let mut random = 19u32;
        for count in [8, 16] {
            for simple in [false, true] {
                for macroblock in [false, true] {
                    for trial in 0..2048 {
                        let mut actual = Plane::new(31, 16);
                        for (index, pixel) in actual.pixels.iter_mut().enumerate() {
                            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                            let byte = (random >> 24) as u8;
                            *pixel = match (trial + index % 31) % 5 {
                                0 => byte,
                                1 => 120 + byte % 16,
                                2 => byte % 4,
                                3 => 252 + byte % 4,
                                _ => 127,
                            };
                        }
                        let mut expected = actual.clone();
                        let strength = FilterStrength::new((trial % 64) as u8,
                            ((trial / 64) % 8) as u8, trial % 2 == 0);
                        let limit = if macroblock { strength.macroblock } else { strength.subblock };
                        filter_edge_scalar(&mut expected, 3, 8, count, 31, limit,
                            strength.interior, strength.hev, simple, macroblock);
                        filter_edge(&mut actual, 3, 8, count, 31, limit,
                            strength.interior, strength.hev, simple, macroblock);
                        assert_eq!(actual.pixels, expected.pixels,
                            "count={count} simple={simple} macroblock={macroblock} trial={trial}");
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "manual horizontal VP8 deblocking kernel timing"]
    fn benchmark_horizontal_filter() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut source = Plane::new(31, 16);
        for (index, sample) in source.pixels.iter_mut().enumerate() {
            *sample = 120 + ((index * 13 + index / 31) % 16) as u8;
        }
        let mut plane = source.clone();
        for trial in 0..5 {
            for accelerated in if trial % 2 == 0 { [false, true] } else { [true, false] } {
                let start = Instant::now();
                for iteration in 0..20_000 {
                    plane.pixels.copy_from_slice(black_box(&source.pixels));
                    if accelerated {
                        filter_edge(black_box(&mut plane), 3, 8, 16, 31, 100, 32, 3,
                            iteration % 3 == 0, iteration % 2 == 0);
                    } else {
                        filter_edge_scalar(black_box(&mut plane), 3, 8, 16, 31, 100, 32, 3,
                            iteration % 3 == 0, iteration % 2 == 0);
                    }
                    black_box(&plane.pixels);
                }
                eprintln!("VP8 horizontal trial={trial} accelerated={accelerated} elapsed={:?}", start.elapsed());
            }
        }
    }

    #[test]
    fn strength_matches_all_levels_and_sharpness_values() {
        for level in 0..64u8 {
            for sharpness in 0..8 {
                for keyframe in [false, true] {
                    let mut interior = i32::from(level);
                    if sharpness != 0 {
                        interior >>= if sharpness > 4 { 2 } else { 1 };
                        interior = interior.min(9 - i32::from(sharpness));
                    }
                    interior = interior.max(1);
                    let hev = match (keyframe, level) {
                        (true, 40..) => 2, (true, 15..) => 1, (true, _) => 0,
                        (false, 40..) => 3, (false, 20..) => 2,
                        (false, 15..) => 1, (false, _) => 0,
                    };
                    let expected =
                        FilterStrength { interior, hev,
                            macroblock: i32::from(level) * 2 + 4 + interior,
                            subblock: i32::from(level) * 2 + interior };
                    assert_eq!(FilterStrength::new(level, sharpness, keyframe), expected);
                    assert_eq!(FILTER_STRENGTHS[usize::from(keyframe)][usize::from(sharpness)]
                        [usize::from(level)], expected);
                }
            }
        }
    }

    #[test]
    fn simple_filter_requires_only_two_samples_on_each_side() {
        for horizontal in [false, true] {
            let (width, height, x, y, step) = if horizontal { (16, 4, 0, 2, 16) }
                else { (4, 16, 2, 0, 1) };
            let mut plane = Plane::new(width, height);
            for row in 0..height {
                for col in 0..width {
                    let side = if horizontal { row } else { col };
                    plane.pixels[row * width + col] = [80, 81, 84, 85][side];
                }
            }
            filter_edge(&mut plane, x, y, 16, step, 20, 1, 0, true, true);
            for row in 0..height {
                for col in 0..width {
                    let side = if horizontal { row } else { col };
                    assert_eq!(plane.pixels[row * width + col], [80, 81, 83, 85][side]);
                }
            }
        }
    }

    #[test]
    fn rejected_edges_do_not_read_unused_outer_samples() {
        let mut plane = Plane::new(4, 16);
        for row in 0..16 {
            plane.pixels[row * 4..row * 4 + 4].copy_from_slice(&[0, 0, 255, 255]);
        }
        let original = plane.pixels.clone();
        filter_edge(&mut plane, 2, 0, 16, 1, 40, 20, 0, false, true);
        assert_eq!(plane.pixels, original);
        for row in 0..16 {
            plane.pixels[row * 4..row * 4 + 4].copy_from_slice(&[0, 100, 100, 255]);
        }
        let original = plane.pixels.clone();
        filter_edge(&mut plane, 2, 0, 16, 1, 200, 20, 0, false, true);
        assert_eq!(plane.pixels, original);
    }

    #[test]
    fn matching_nearest_pairs_leave_all_edges_unchanged() {
        for nearest in 0..65536u32 {
            for horizontal in [false, true] {
                let mut plane = Plane::new(8, 8);
                for (index, pixel) in plane.pixels.iter_mut().enumerate() {
                    *pixel = (index * 73) as u8;
                }
                for offset in 0..8 {
                    let (base, step) = if horizontal { (4 * 8 + offset, 8) } else { (offset * 8 + 4, 1) };
                    plane.pixels[base - step] = nearest as u8;
                    plane.pixels[base] = nearest as u8;
                    plane.pixels[base - 2 * step] = (nearest >> 8) as u8;
                    plane.pixels[base + step] = (nearest >> 8) as u8;
                }
                let original = plane.pixels.clone();
                let (x, y, step) = if horizontal { (0, 4, 8) } else { (4, 0, 1) };
                for simple in [false, true] {
                    for macroblock in [false, true] {
                        filter_edge(&mut plane, x, y, 8, step, 193, 63, 3, simple, macroblock);
                        assert_eq!(plane.pixels, original);
                    }
                }
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn transpose_preserves_all_samples() {
        use std::arch::aarch64::*;
        let samples: [u8; 64] = std::array::from_fn(|index| index as u8);
        unsafe {
            let rows = std::array::from_fn(|row| vld1_u8(samples[row * 8..].as_ptr()));
            let columns = transpose_eight_neon(rows);
            for (col, vector) in columns.into_iter().enumerate() {
                let mut actual = [0u8; 8];
                vst1_u8(actual.as_mut_ptr(), vector);
                for row in 0..8 { assert_eq!(actual[row], samples[row * 8 + col]); }
            }
            let restored = transpose_eight_neon(columns);
            let mut actual = [0u8; 64];
            for (row, vector) in restored.into_iter().enumerate() {
                vst1_u8(actual[row * 8..].as_mut_ptr(), vector);
            }
            assert_eq!(actual, samples);
        }
    }

    #[test]
    fn vertical_batches_match_scalar_edges() {
        let mut random = 0x89123abdu32;
        for count in [8, 16] {
            for simple in [false, true] {
                for macroblock in [false, true] {
                    for trial in 0..2048 {
                        let width = if trial % 3 == 0 { 8 } else { 31 };
                        let x = if width == 8 { 4 } else { 7 };
                        let mut actual = Plane::new(width, count + 3);
                        for (index, pixel) in actual.pixels.iter_mut().enumerate() {
                            random ^= random << 13;
                            random ^= random >> 17;
                            random ^= random << 5;
                            let byte = random as u8;
                            *pixel = match (trial + index % 31) % 5 {
                                0 => byte,
                                1 => 120 + byte % 16,
                                2 => byte % 4,
                                3 => 252 + byte % 4,
                                _ => 127,
                            };
                        }
                        let mut expected = actual.clone();
                        let strength = FilterStrength::new((trial % 64) as u8,
                            ((trial / 64) % 8) as u8, trial % 2 == 0);
                        let limit = if macroblock { strength.macroblock } else { strength.subblock };
                        filter_edge_scalar(&mut expected, x, 3, count, 1, limit,
                            strength.interior, strength.hev, simple, macroblock);
                        filter_edge(&mut actual, x, 3, count, 1, limit,
                            strength.interior, strength.hev, simple, macroblock);
                        assert_eq!(actual.pixels, expected.pixels,
                            "width={width} count={count} simple={simple} macroblock={macroblock} trial={trial}");
                    }
                }
            }
        }
    }

    #[test]
    fn vertical_short_and_narrow_edges_use_scalar_fallback() {
        for count in [1, 7, 8, 9, 16, 17] {
            for simple in [false, true] {
                let width = if simple { 4 } else { 8 };
                let mut actual = Plane::new(width, count);
                for row in 0..count {
                    for col in 0..width {
                        actual.pixels[row * width + col] = 120 + ((row + col) % 8) as u8;
                    }
                }
                let mut expected = actual.clone();
                filter_edge_scalar(&mut expected, width / 2, 0, count, 1, 50, 16, 1, simple, true);
                filter_edge(&mut actual, width / 2, 0, count, 1, 50, 16, 1, simple, true);
                assert_eq!(actual.pixels, expected.pixels);
            }
        }
    }

    #[test]
    #[ignore = "manual vertical VP8 deblocking kernel timing"]
    fn benchmark_vertical_filter() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut source = Plane::new(31, 24);
        for (index, sample) in source.pixels.iter_mut().enumerate() {
            *sample = 120 + ((index * 13 + index / 31) % 16) as u8;
        }
        let mut plane = source.clone();
        for trial in 0..5 {
            for accelerated in if trial % 2 == 0 { [false, true] } else { [true, false] } {
                let start = Instant::now();
                for iteration in 0..200_000 {
                    plane.pixels.copy_from_slice(black_box(&source.pixels));
                    if accelerated {
                        filter_edge(black_box(&mut plane), 8, 3, 16, 1, 100, 32, 3,
                            iteration % 3 == 0, iteration % 2 == 0);
                    } else {
                        filter_edge_scalar(black_box(&mut plane), 8, 3, 16, 1, 100, 32, 3,
                            iteration % 3 == 0, iteration % 2 == 0);
                    }
                    black_box(&plane.pixels);
                }
                eprintln!("VP8 vertical trial={trial} accelerated={accelerated} elapsed={:?}", start.elapsed());
            }
        }
    }
}
