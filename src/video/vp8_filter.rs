//! VP8 in-loop deblocking (RFC 6386, section 15).

use super::vp8_predict::Plane;

#[derive(Clone, Copy)]
pub(super) struct FilterMacroblock {
    pub level: u8,
    pub skip_inner: bool,
}

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
    for mb_y in 0..mb_height {
        for mb_x in 0..mb_width {
            let setting = settings[mb_y * mb_width + mb_x];
            let level = i32::from(setting.level);
            if level == 0 {
                continue;
            }
            let mut interior = level;
            if sharpness != 0 {
                interior >>= if sharpness > 4 { 2 } else { 1 };
                interior = interior.min(9 - i32::from(sharpness));
            }
            interior = interior.max(1);
            let hev = if keyframe {
                if level >= 40 {
                    2
                } else if level >= 15 {
                    1
                } else {
                    0
                }
            } else if level >= 40 {
                3
            } else if level >= 20 {
                2
            } else if level >= 15 {
                1
            } else {
                0
            };
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
                        level * 2 + 4 + interior,
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
                            level * 2 + interior,
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
                        level * 2 + 4 + interior,
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
                            level * 2 + interior,
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
    for offset in 0..count {
        let base = y * plane.width
            + x
            + if step == 1 {
                offset * plane.width
            } else {
                offset
            };
        let p = [
            i32::from(plane.pixels[base - step]),
            i32::from(plane.pixels[base - 2 * step]),
            i32::from(plane.pixels[base - 3 * step]),
            i32::from(plane.pixels[base - 4 * step]),
        ];
        let q = [
            i32::from(plane.pixels[base]),
            i32::from(plane.pixels[base + step]),
            i32::from(plane.pixels[base + 2 * step]),
            i32::from(plane.pixels[base + 3 * step]),
        ];
        if 2 * (p[0] - q[0]).abs() + (p[1] - q[1]).abs() / 2 > edge_limit {
            continue;
        }
        if simple {
            common_adjust(plane, base, step, p, q, true);
            continue;
        }
        if (p[0] - p[1]).abs() > interior_limit
            || (p[1] - p[2]).abs() > interior_limit
            || (p[2] - p[3]).abs() > interior_limit
            || (q[0] - q[1]).abs() > interior_limit
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
