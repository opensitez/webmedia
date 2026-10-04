//! In-loop deblocking and CDEF, normative AV1 sections 7.14 and 7.15.
use super::{
    decoder::DecodedPlane,
    syntax::{Error, IntraFrameHeader, SequenceHeader},
};

pub(crate) fn deblock(
    planes: &mut [DecodedPlane],
    s: &SequenceHeader,
    h: &IntraFrameHeader,
    tx: &[Vec<(u8, u8)>],
    motion: &[Option<super::inter::MotionCell>],
    skips: &[bool],
    mi_cols: usize,
) -> Result<(), Error> {
    if h.segmentation_enabled || h.delta_lf.is_some() {
        return Err(Error::Unsupported("segmented or delta loop filter"));
    }
    for (plane, p) in planes.iter_mut().enumerate() {
        if plane > 0 && h.loop_filter_levels[plane + 1] == 0 {
            continue;
        }
        for pass in 0..2 {
            let index = if plane == 0 { pass } else { plane + 1 };
            let sx = usize::from(plane > 0 && s.subsampling_x);
            let sy = usize::from(plane > 0 && s.subsampling_y);
            let visible_w = (h.width as usize).div_ceil(1 << sx);
            let visible_h = (h.height as usize).div_ceil(1 << sy);
            let cols = p.width / 4;
            for row in 0..p.height / 4 {
                for col in 0..cols {
                    let (x, y) = (col * 4, row * 4);
                    if x >= visible_w
                        || y >= visible_h
                        || (pass == 0 && x == 0)
                        || (pass == 1 && y == 0)
                    {
                        continue;
                    }
                    let (w, ht) = tx[plane][row * cols + col];
                    let (pw, ph) = tx[plane][if pass == 0 {
                        row * cols + col - 1
                    } else {
                        (row - 1) * cols + col
                    }];
                    if (if pass == 0 {
                        x % usize::from(w)
                    } else {
                        y % usize::from(ht)
                    }) != 0
                    {
                        continue;
                    }
                    let mr = (row << sy) | sy;
                    let mc = (col << sx) | sx;
                    let current = mr * mi_cols + mc;
                    let previous = if pass == 0 {
                        current - (1 << sx)
                    } else {
                        current - mi_cols * (1 << sy)
                    };
                    let cell =
                        motion[current].ok_or(Error::Invalid("missing loop filter mode info"))?;
                    let block_width = ((cell.width * 4) >> sx).max(4);
                    let block_height = ((cell.height * 4) >> sy).max(4);
                    let block_edge = if pass == 0 {
                        x % block_width == 0
                    } else {
                        y % block_height == 0
                    };
                    if !block_edge && skips[current] && cell.refs[0] > 0 {
                        continue;
                    }
                    let strength = |at: usize| -> Result<i32, Error> {
                        let cell = motion[at]
                            .ok_or(Error::Invalid("missing adjacent loop filter mode info"))?;
                        let mut level = i32::from(h.loop_filter_levels[index]);
                        if h.loop_filter_delta_enabled {
                            let shift = level >> 5;
                            level +=
                                h.loop_filter_ref_deltas[cell.refs[0].max(0) as usize] << shift;
                            if cell.refs[0] > 0 {
                                let mode_type = usize::from(
                                    cell.mode >= 13
                                        && cell.mode != 15
                                        && cell.mode != super::inter::GLOBAL_GLOBAL,
                                );
                                level += h.loop_filter_mode_deltas[mode_type] << shift;
                            }
                            level = level.clamp(0, 63);
                        }
                        Ok(level)
                    };
                    let mut level = strength(current)?;
                    if level == 0 {
                        level = strength(previous)?;
                    }
                    if level == 0 {
                        continue;
                    }
                    let sharp = i32::from(h.loop_filter_sharpness);
                    let shift = if sharp > 4 { 2 } else { usize::from(sharp > 0) };
                    let limit = if sharp > 0 {
                        (level >> shift).clamp(1, 9 - sharp)
                    } else {
                        (level >> shift).max(1)
                    };
                    let scale = 1 << (s.bit_depth - 8);
                    let (limit, blimit, threshold) = (
                        limit * scale,
                        (2 * (level + 2) + limit) * scale,
                        (level >> 4) * scale,
                    );
                    let size = usize::from(if pass == 0 { w.min(pw) } else { ht.min(ph) })
                        .min(if plane == 0 { 16 } else { 8 });
                    for i in 0..4 {
                        let (xx, yy) = if pass == 0 { (x, y + i) } else { (x + i, y) };
                        if xx >= visible_w || yy >= visible_h {
                            continue;
                        }
                        filter_sample(
                            p,
                            xx,
                            yy,
                            pass,
                            plane,
                            size,
                            s.bit_depth,
                            limit,
                            blimit,
                            threshold,
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn filter_sample(
    p: &mut DecodedPlane,
    x: usize,
    y: usize,
    pass: usize,
    plane: usize,
    size: usize,
    depth: u8,
    limit: i32,
    blimit: i32,
    threshold: i32,
) {
    #[cfg(test)]
    if DEBLOCK_REFERENCE.with(|flag| flag.get()) {
        return filter_sample_reference(
            p, x, y, pass, plane, size, depth, limit, blimit, threshold,
        );
    }
    let n = if size == 4 {
        2
    } else if plane > 0 {
        3
    } else if size == 8 {
        4
    } else {
        7
    };
    let mut a = [0i32; 7];
    let mut b = [0i32; 7];
    for i in 0..n {
        let sample = |offset: isize| -> i32 {
            let xx = if pass == 0 {
                (x as isize + offset).clamp(0, p.width as isize - 1) as usize
            } else {
                x
            };
            let yy = if pass == 1 {
                (y as isize + offset).clamp(0, p.height as isize - 1) as usize
            } else {
                y
            };
            i32::from(p.samples[yy * p.stride + xx])
        };
        a[i] = sample(-(i as isize) - 1);
        b[i] = sample(i as isize);
    }
    if (a[0] - b[0]).abs() * 2 + (a[1] - b[1]).abs() / 2 > blimit
        || (1..n.min(4)).any(|i| (a[i] - a[i - 1]).abs() > limit || (b[i] - b[i - 1]).abs() > limit)
    {
        return;
    }
    let flat = size >= 8
        && (1..n.min(4)).all(|i| {
            (a[i] - a[0]).abs() <= 1 << (depth - 8) && (b[i] - b[0]).abs() <= 1 << (depth - 8)
        });
    let flat2 = size >= 16
        && (4..7).all(|i| {
            (a[i] - a[0]).abs() <= 1 << (depth - 8) && (b[i] - b[0]).abs() <= 1 << (depth - 8)
        });
    let mut changes = [(0isize, 0i32); 12];
    let mut count = 0;
    let mut push = |change| {
        changes[count] = change;
        count += 1;
    };
    if flat {
        let log = if size == 16 && flat2 { 4 } else { 3 };
        let n = if log == 4 {
            6
        } else if plane == 0 {
            3
        } else {
            2
        };
        let n2 = usize::from(log == 4 || plane > 0) as isize;
        let n = n as isize;
        let sample = |offset: isize| {
            let off = offset.clamp(-n - 1, n);
            if off < 0 {
                a[(-off - 1) as usize]
            } else {
                b[off as usize]
            }
        };
        // Each next output shifts the same box window by one sample; the
        // central one or three taps have a second unit of weight.
        let mut window: i32 = (-n..=n).map(|j| sample(-n + j)).sum();
        for i in -n..n {
            let extra: i32 = (-n2..=n2).map(|j| sample(i + j)).sum();
            push((i, (window + extra + (1 << (log - 1))) >> log));
            window += sample(i + n + 1) - sample(i - n);
        }
    } else {
        let hev = (a[1] - a[0]).abs() > threshold || (b[1] - b[0]).abs() > threshold;
        let half = 1 << (depth - 1);
        let clamp = |v: i32| v.clamp(-half, half - 1);
        let f = clamp((if hev { clamp(a[1] - b[1]) } else { 0 }) + 3 * (b[0] - a[0]));
        let f1 = clamp(f + 4) >> 3;
        let f2 = clamp(f + 3) >> 3;
        push((-1, clamp(a[0] - half + f2) + half));
        push((0, clamp(b[0] - half - f1) + half));
        if !hev {
            let f = (f1 + 1) >> 1;
            push((-2, clamp(a[1] - half + f) + half));
            push((1, clamp(b[1] - half - f) + half));
        }
    }
    for &(off, value) in &changes[..count] {
        let xx = if pass == 0 {
            x as isize + off
        } else {
            x as isize
        };
        let yy = if pass == 1 {
            y as isize + off
        } else {
            y as isize
        };
        if xx >= 0 && yy >= 0 && (xx as usize) < p.width && (yy as usize) < p.height {
            p.samples[yy as usize * p.stride + xx as usize] = value as u16;
        }
    }
}
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn filter_sample_reference(
    p: &mut DecodedPlane,
    x: usize,
    y: usize,
    pass: usize,
    plane: usize,
    size: usize,
    depth: u8,
    limit: i32,
    blimit: i32,
    threshold: i32,
) {
    let n = if size == 4 {
        2
    } else if plane > 0 {
        3
    } else if size == 8 {
        4
    } else {
        7
    };
    let mut a = [0i32; 7];
    let mut b = [0i32; 7];
    for i in 0..n {
        let sample = |offset: isize| -> i32 {
            let xx = if pass == 0 {
                (x as isize + offset).clamp(0, p.width as isize - 1) as usize
            } else {
                x
            };
            let yy = if pass == 1 {
                (y as isize + offset).clamp(0, p.height as isize - 1) as usize
            } else {
                y
            };
            i32::from(p.samples[yy * p.stride + xx])
        };
        a[i] = sample(-(i as isize) - 1);
        b[i] = sample(i as isize);
    }
    if (a[0] - b[0]).abs() * 2 + (a[1] - b[1]).abs() / 2 > blimit
        || (1..n.min(4)).any(|i| (a[i] - a[i - 1]).abs() > limit || (b[i] - b[i - 1]).abs() > limit)
    {
        return;
    }
    let flat = size >= 8
        && (1..n.min(4)).all(|i| {
            (a[i] - a[0]).abs() <= 1 << (depth - 8) && (b[i] - b[0]).abs() <= 1 << (depth - 8)
        });
    let flat2 = size >= 16
        && (4..7).all(|i| {
            (a[i] - a[0]).abs() <= 1 << (depth - 8) && (b[i] - b[0]).abs() <= 1 << (depth - 8)
        });
    let mut changes = Vec::new();
    if flat {
        let log = if size == 16 && flat2 { 4 } else { 3 };
        let n = if log == 4 {
            6
        } else if plane == 0 {
            3
        } else {
            2
        };
        let n2 = usize::from(log == 4 || plane > 0) as isize;
        for i in -(n as isize)..n as isize {
            let mut sum = 0;
            for j in -(n as isize)..=n as isize {
                let off = (i + j).clamp(-(n as isize) - 1, n as isize);
                let v = if off < 0 {
                    a[(-off - 1) as usize]
                } else {
                    b[off as usize]
                };
                sum += v * if j.abs() <= n2 { 2 } else { 1 };
            }
            changes.push((i, (sum + (1 << (log - 1))) >> log));
        }
    } else {
        let hev = (a[1] - a[0]).abs() > threshold || (b[1] - b[0]).abs() > threshold;
        let half = 1 << (depth - 1);
        let clamp = |v: i32| v.clamp(-half, half - 1);
        let f = clamp((if hev { clamp(a[1] - b[1]) } else { 0 }) + 3 * (b[0] - a[0]));
        let f1 = clamp(f + 4) >> 3;
        let f2 = clamp(f + 3) >> 3;
        changes.push((-1, clamp(a[0] - half + f2) + half));
        changes.push((0, clamp(b[0] - half - f1) + half));
        if !hev {
            let f = (f1 + 1) >> 1;
            changes.push((-2, clamp(a[1] - half + f) + half));
            changes.push((1, clamp(b[1] - half - f) + half));
        }
    }
    for (off, value) in changes {
        let xx = if pass == 0 {
            x as isize + off
        } else {
            x as isize
        };
        let yy = if pass == 1 {
            y as isize + off
        } else {
            y as isize
        };
        if xx >= 0 && yy >= 0 && (xx as usize) < p.width && (yy as usize) < p.height {
            p.samples[yy as usize * p.stride + xx as usize] = value as u16;
        }
    }
}

#[cfg(test)]
std::thread_local! {
    static DEBLOCK_REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
pub(crate) fn set_deblock_reference(enabled: bool) {
    DEBLOCK_REFERENCE.with(|flag| flag.set(enabled));
}

const DIRECTIONS: [[(i32, i32); 2]; 8] = [
    [(-1, 1), (-2, 2)],
    [(0, 1), (-1, 2)],
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2)],
    [(1, 1), (2, 2)],
    [(1, 0), (2, 1)],
    [(1, 0), (2, 0)],
    [(1, 0), (2, -1)],
];
fn direction(p: &DecodedPlane, x: usize, y: usize, depth: u8) -> (usize, i32) {
    let mut partial = [[0i32; 15]; 8];
    for i in 0..8 {
        for j in 0..8 {
            let v = (i32::from(
                p.samples[(y + i).min(p.height - 1) * p.stride + (x + j).min(p.width - 1)],
            ) >> (depth - 8))
                - 128;
            for (dir, index) in [
                i + j,
                i + j / 2,
                i,
                3 + i - j / 2,
                7 + i - j,
                3 - i / 2 + j,
                j,
                i / 2 + j,
            ]
            .into_iter()
            .enumerate()
            {
                partial[dir][index] += v;
            }
        }
    }
    let div = [0, 840, 420, 280, 210, 168, 140, 120, 105];
    let mut cost = [0i32; 8];
    for i in 0..8 {
        cost[2] += partial[2][i] * partial[2][i];
        cost[6] += partial[6][i] * partial[6][i];
    }
    cost[2] *= div[8];
    cost[6] *= div[8];
    for dir in [0, 4] {
        for i in 0..7 {
            cost[dir] += (partial[dir][i] * partial[dir][i]
                + partial[dir][14 - i] * partial[dir][14 - i])
                * div[i + 1];
        }
        cost[dir] += partial[dir][7] * partial[dir][7] * div[8];
    }
    for dir in [1, 3, 5, 7] {
        for i in 0..5 {
            cost[dir] += partial[dir][3 + i] * partial[dir][3 + i];
        }
        cost[dir] *= div[8];
        for i in 0..3 {
            cost[dir] += (partial[dir][i] * partial[dir][i]
                + partial[dir][10 - i] * partial[dir][10 - i])
                * div[2 * i + 2];
        }
    }
    let mut best = 0;
    for i in 1..8 {
        if cost[i] > cost[best] {
            best = i;
        }
    }
    (best, (cost[best] - cost[(best + 4) & 7]) >> 10)
}
#[cfg(test)]
fn constrain(diff: i32, threshold: i32, damping: i32) -> i32 {
    if threshold == 0 {
        return 0;
    }
    let shift = (damping - threshold.ilog2() as i32).max(0);
    diff.signum() * (threshold - (diff.abs() >> shift)).clamp(0, diff.abs())
}

pub(crate) fn cdef(
    planes: &mut [DecodedPlane],
    s: &SequenceHeader,
    h: &IntraFrameHeader,
    skips: &[bool],
    cols: usize,
    indices: &[i16],
) -> Result<(), Error> {
    if !s.enable_cdef || h.coded_lossless || h.allow_intrabc {
        return Ok(());
    }
    #[cfg(test)]
    if CDEF_REFERENCE.with(|flag| flag.get()) {
        return cdef_reference(planes, s, h, skips, cols, indices);
    }
    #[cfg(test)]
    if CDEF_SNAPSHOT_REFERENCE.with(|flag| flag.get()) {
        return cdef_snapshot_reference(planes, s, h, skips, cols, indices);
    }
    let rows = planes[0].height / 4;
    let mut blocks = Vec::new();
    // Direction search reads the complete reconstructed luma window. Do it
    // before writing any filtered samples, without a second full-frame clone.
    for r in (0..rows).step_by(2) {
        for c in (0..cols).step_by(2) {
            let index = indices[(r / 16) * cols.div_ceil(16) + c / 16];
            if index < 0
                || [
                    r * cols + c,
                    r * cols + (c + 1).min(cols - 1),
                    (r + 1).min(rows - 1) * cols + c,
                    (r + 1).min(rows - 1) * cols + (c + 1).min(cols - 1),
                ]
                .into_iter()
                .all(|i| skips[i])
            {
                continue;
            }
            let strengths = *h
                .cdef_strengths
                .get(index as usize)
                .ok_or(Error::Invalid("CDEF strength index"))?;
            if strengths == [0; 4] {
                continue;
            }
            let (dir, variance) = direction(&planes[0], c * 4, r * 4, s.bit_depth);
            blocks.push((r, c, strengths, dir, variance));
        }
    }
    if blocks.is_empty() {
        return Ok(());
    }
    let use_neon = std::env::var_os("AV1_DISABLE_NEON").is_none();
    let padded: Vec<_> = planes
        .iter()
        .enumerate()
        .map(|(plane, p)| {
            let sx = usize::from(plane > 0 && s.subsampling_x);
            let sy = usize::from(plane > 0 && s.subsampling_y);
            let width = (h.width as usize).div_ceil(1 << sx);
            let height = (h.height as usize).div_ceil(1 << sy);
            let stride = width + 16;
            // Two rows and wide horizontal halos also make partial SIMD rows safe.
            let mut samples = vec![i16::MAX; (height + 4) * stride];
            for y in 0..height {
                for x in 0..width {
                    samples[(y + 2) * stride + x + 4] = p.samples[y * p.stride + x] as i16;
                }
            }
            (samples, stride)
        })
        .collect();
    for (r, c, strengths, dir, variance) in blocks {
        for plane in 0..planes.len() {
            let shift = s.bit_depth - 8;
            let mut pri = i32::from(strengths[if plane == 0 { 0 } else { 2 }]) << shift;
            let sec = i32::from(strengths[if plane == 0 { 1 } else { 3 }]) << shift;
            let mut d = if pri == 0 { 0 } else { dir };
            if plane == 0 {
                let v = variance >> 6;
                let vstr = if v != 0 { v.ilog2().min(12) } else { 0 };
                pri = if variance != 0 {
                    (pri * (4 + vstr as i32) + 8) >> 4
                } else {
                    0
                };
            }
            let sx = usize::from(plane > 0 && s.subsampling_x);
            let sy = usize::from(plane > 0 && s.subsampling_y);
            if plane > 0 && pri != 0 {
                d = match (sx, sy) {
                    (0, 1) => [1, 2, 2, 2, 3, 4, 6, 0][dir],
                    (1, 0) => [7, 0, 2, 4, 5, 6, 6, 6][dir],
                    _ => dir,
                };
            }
            let damping = i32::from(h.cdef_damping) + i32::from(shift) - i32::from(plane > 0);
            let (x0, y0) = ((c * 4) >> sx, (r * 4) >> sy);
            let (w, ht) = (8 >> sx, 8 >> sy);
            let output_stride = planes[plane].stride;
            let max_w = (h.width as usize).div_ceil(1 << sx);
            let max_h = (h.height as usize).div_ceil(1 << sy);
            if pri == 0 && sec == 0 || x0 >= max_w || y0 >= max_h {
                continue;
            }
            let (source, stride) = &padded[plane];
            let primary = if (pri >> shift) & 1 == 0 {
                [4, 2]
            } else {
                [3, 3]
            };
            let mut taps = [CdefTap::default(); 12];
            let mut n = 0;
            for k in 0..2 {
                for sign in [-1, 1] {
                    for off in [0, -2, 2] {
                        let (dy, dx) = DIRECTIONS[((d as i32 + off) & 7) as usize][k];
                        let threshold = if off == 0 { pri } else { sec };
                        taps[n] = CdefTap {
                            offset: sign as isize * (dy as isize * *stride as isize + dx as isize),
                            threshold: threshold as u16,
                            shift: if threshold == 0 {
                                0
                            } else {
                                (damping - threshold.ilog2() as i32).max(0) as u32
                            },
                            weight: if off == 0 { primary[k] } else { [2, 1][k] },
                        };
                        n += 1;
                    }
                }
            }
            let count = w.min(max_w - x0);
            for y in y0..(y0 + ht).min(max_h) {
                let position = (y + 2) * *stride + x0 + 4;
                let output = &mut planes[plane].samples
                    [y * output_stride + x0..y * output_stride + x0 + count];
                cdef_row(source, position, output, &taps, use_neon);
            }
        }
    }
    Ok(())
}
#[cfg(test)]
fn cdef_snapshot_reference(
    planes: &mut [DecodedPlane],
    s: &SequenceHeader,
    h: &IntraFrameHeader,
    skips: &[bool],
    cols: usize,
    indices: &[i16],
) -> Result<(), Error> {
    if !s.enable_cdef || h.coded_lossless || h.allow_intrabc {
        return Ok(());
    }
    #[cfg(test)]
    if CDEF_REFERENCE.with(|flag| flag.get()) {
        return cdef_reference(planes, s, h, skips, cols, indices);
    }
    let input = planes.to_vec();
    let use_neon = std::env::var_os("AV1_DISABLE_NEON").is_none();
    let padded: Vec<_> = input
        .iter()
        .enumerate()
        .map(|(plane, p)| {
            let sx = usize::from(plane > 0 && s.subsampling_x);
            let sy = usize::from(plane > 0 && s.subsampling_y);
            let width = (h.width as usize).div_ceil(1 << sx);
            let height = (h.height as usize).div_ceil(1 << sy);
            let stride = width + 16;
            // Two rows and wide horizontal halos also make partial SIMD rows safe.
            let mut samples = vec![i16::MAX; (height + 4) * stride];
            for y in 0..height {
                for x in 0..width {
                    samples[(y + 2) * stride + x + 4] = p.samples[y * p.stride + x] as i16;
                }
            }
            (samples, stride)
        })
        .collect();
    let rows = input[0].height / 4;
    for r in (0..rows).step_by(2) {
        for c in (0..cols).step_by(2) {
            let index = indices[(r / 16) * cols.div_ceil(16) + c / 16];
            if index < 0
                || [
                    r * cols + c,
                    r * cols + (c + 1).min(cols - 1),
                    (r + 1).min(rows - 1) * cols + c,
                    (r + 1).min(rows - 1) * cols + (c + 1).min(cols - 1),
                ]
                .into_iter()
                .all(|i| skips[i])
            {
                continue;
            }
            let strengths = *h
                .cdef_strengths
                .get(index as usize)
                .ok_or(Error::Invalid("CDEF strength index"))?;
            let (dir, variance) = direction(&input[0], c * 4, r * 4, s.bit_depth);
            for plane in 0..planes.len() {
                let shift = s.bit_depth - 8;
                let mut pri = i32::from(strengths[if plane == 0 { 0 } else { 2 }]) << shift;
                let sec = i32::from(strengths[if plane == 0 { 1 } else { 3 }]) << shift;
                let mut d = if pri == 0 { 0 } else { dir };
                if plane == 0 {
                    let v = variance >> 6;
                    let vstr = if v != 0 { v.ilog2().min(12) } else { 0 };
                    pri = if variance != 0 {
                        (pri * (4 + vstr as i32) + 8) >> 4
                    } else {
                        0
                    };
                }
                let sx = usize::from(plane > 0 && s.subsampling_x);
                let sy = usize::from(plane > 0 && s.subsampling_y);
                if plane > 0 && pri != 0 {
                    d = match (sx, sy) {
                        (0, 1) => [1, 2, 2, 2, 3, 4, 6, 0][dir],
                        (1, 0) => [7, 0, 2, 4, 5, 6, 6, 6][dir],
                        _ => dir,
                    };
                }
                let damping = i32::from(h.cdef_damping) + i32::from(shift) - i32::from(plane > 0);
                let (x0, y0) = ((c * 4) >> sx, (r * 4) >> sy);
                let (w, ht) = (8 >> sx, 8 >> sy);
                let p = &input[plane];
                let max_w = (h.width as usize).div_ceil(1 << sx);
                let max_h = (h.height as usize).div_ceil(1 << sy);
                if pri == 0 && sec == 0 || x0 >= max_w || y0 >= max_h {
                    continue;
                }
                let (source, stride) = &padded[plane];
                let primary = if (pri >> shift) & 1 == 0 {
                    [4, 2]
                } else {
                    [3, 3]
                };
                let mut taps = [CdefTap::default(); 12];
                let mut n = 0;
                for k in 0..2 {
                    for sign in [-1, 1] {
                        for off in [0, -2, 2] {
                            let (dy, dx) = DIRECTIONS[((d as i32 + off) & 7) as usize][k];
                            let threshold = if off == 0 { pri } else { sec };
                            taps[n] = CdefTap {
                                offset: sign as isize
                                    * (dy as isize * *stride as isize + dx as isize),
                                threshold: threshold as u16,
                                shift: if threshold == 0 {
                                    0
                                } else {
                                    (damping - threshold.ilog2() as i32).max(0) as u32
                                },
                                weight: if off == 0 { primary[k] } else { [2, 1][k] },
                            };
                            n += 1;
                        }
                    }
                }
                let count = w.min(max_w - x0);
                for y in y0..(y0 + ht).min(max_h) {
                    let position = (y + 2) * *stride + x0 + 4;
                    let output =
                        &mut planes[plane].samples[y * p.stride + x0..y * p.stride + x0 + count];
                    cdef_row(source, position, output, &taps, use_neon);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
std::thread_local! {
    static CDEF_SNAPSHOT_REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
pub(crate) fn set_cdef_snapshot_reference(enabled: bool) {
    CDEF_SNAPSHOT_REFERENCE.with(|flag| flag.set(enabled));
}

#[derive(Clone, Copy, Default)]
struct CdefTap {
    offset: isize,
    threshold: u16,
    shift: u32,
    weight: i16,
}

fn cdef_row_scalar(source: &[i16], position: usize, output: &mut [u16], taps: &[CdefTap; 12]) {
    for (x, pixel) in output.iter_mut().enumerate() {
        let center = i32::from(source[position + x]);
        let (mut lo, mut hi, mut sum) = (center, center, 0);
        for tap in taps {
            let neighbor = source[(position + x).checked_add_signed(tap.offset).unwrap()];
            if neighbor == i16::MAX {
                continue;
            }
            let value = i32::from(neighbor);
            lo = lo.min(value);
            hi = hi.max(value);
            let diff = value - center;
            let magnitude = diff.abs();
            let limited = (i32::from(tap.threshold) - (magnitude >> tap.shift)).clamp(0, magnitude);
            sum += i32::from(tap.weight) * diff.signum() * limited;
        }
        *pixel = (center + ((8 + sum - i32::from(sum < 0)) >> 4)).clamp(lo, hi) as u16;
    }
}

fn cdef_row(
    source: &[i16],
    position: usize,
    output: &mut [u16],
    taps: &[CdefTap; 12],
    use_neon: bool,
) {
    #[cfg(target_arch = "aarch64")]
    {
        // The padded source has two vertical rows, four left and twelve right
        // samples. Every eight-lane neighbor load is inside that allocation.
        if use_neon {
            unsafe { cdef_row_neon(source.as_ptr().add(position), output, taps) };
            return;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = use_neon;
    cdef_row_scalar(source, position, output, taps);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn cdef_row_neon(source: *const i16, output: &mut [u16], taps: &[CdefTap; 12]) {
    use std::arch::aarch64::*;
    unsafe {
        let center = vld1q_s16(source);
        let zero = vdupq_n_s16(0);
        let mut lo = center;
        let mut hi = center;
        let mut sum = zero;
        for tap in taps {
            let raw = vld1q_s16(source.offset(tap.offset));
            let value = vbslq_s16(vceqq_s16(raw, vdupq_n_s16(i16::MAX)), center, raw);
            lo = vminq_s16(lo, value);
            hi = vmaxq_s16(hi, value);
            if tap.threshold == 0 {
                continue;
            }
            let diff = vsubq_s16(value, center);
            let magnitude = vreinterpretq_u16_s16(vabsq_s16(diff));
            let reduced = vshlq_u16(magnitude, vdupq_n_s16(-(tap.shift as i16)));
            let allowed = vqsubq_u16(vdupq_n_u16(tap.threshold), reduced);
            let limited = vreinterpretq_s16_u16(vminq_u16(magnitude, allowed));
            let signed = vbslq_s16(vcltq_s16(diff, zero), vnegq_s16(limited), limited);
            sum = vmlaq_n_s16(sum, signed, tap.weight);
        }
        let correction = vreinterpretq_s16_u16(vcltq_s16(sum, zero));
        let rounded = vshrq_n_s16::<4>(vaddq_s16(vaddq_s16(sum, vdupq_n_s16(8)), correction));
        let result =
            vreinterpretq_u16_s16(vminq_s16(hi, vmaxq_s16(lo, vaddq_s16(center, rounded))));
        match output.len() {
            8 => vst1q_u16(output.as_mut_ptr(), result),
            4 => vst1_u16(output.as_mut_ptr(), vget_low_u16(result)),
            _ => {
                let mut tail = [0u16; 8];
                vst1q_u16(tail.as_mut_ptr(), result);
                output.copy_from_slice(&tail[..output.len()]);
            }
        }
    }
}

#[cfg(test)]
std::thread_local! {
    static CDEF_REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
pub(crate) fn set_cdef_reference(enabled: bool) {
    CDEF_REFERENCE.with(|flag| flag.set(enabled));
}

#[cfg(test)]
fn cdef_reference(
    planes: &mut [DecodedPlane],
    s: &SequenceHeader,
    h: &IntraFrameHeader,
    skips: &[bool],
    cols: usize,
    indices: &[i16],
) -> Result<(), Error> {
    if !s.enable_cdef || h.coded_lossless || h.allow_intrabc {
        return Ok(());
    }
    let input = planes.to_vec();
    let rows = input[0].height / 4;
    for r in (0..rows).step_by(2) {
        for c in (0..cols).step_by(2) {
            let index = indices[(r / 16) * cols.div_ceil(16) + c / 16];
            if index < 0
                || [
                    r * cols + c,
                    r * cols + (c + 1).min(cols - 1),
                    (r + 1).min(rows - 1) * cols + c,
                    (r + 1).min(rows - 1) * cols + (c + 1).min(cols - 1),
                ]
                .into_iter()
                .all(|i| skips[i])
            {
                continue;
            }
            let strengths = *h
                .cdef_strengths
                .get(index as usize)
                .ok_or(Error::Invalid("CDEF strength index"))?;
            let (dir, variance) = direction(&input[0], c * 4, r * 4, s.bit_depth);
            for plane in 0..planes.len() {
                let shift = s.bit_depth - 8;
                let mut pri = i32::from(strengths[if plane == 0 { 0 } else { 2 }]) << shift;
                let sec = i32::from(strengths[if plane == 0 { 1 } else { 3 }]) << shift;
                let mut d = if pri == 0 { 0 } else { dir };
                if plane == 0 {
                    let v = variance >> 6;
                    let vstr = if v != 0 { v.ilog2().min(12) } else { 0 };
                    pri = if variance != 0 {
                        (pri * (4 + vstr as i32) + 8) >> 4
                    } else {
                        0
                    };
                }
                let sx = usize::from(plane > 0 && s.subsampling_x);
                let sy = usize::from(plane > 0 && s.subsampling_y);
                if plane > 0 && pri != 0 {
                    d = match (sx, sy) {
                        (0, 1) => [1, 2, 2, 2, 3, 4, 6, 0][dir],
                        (1, 0) => [7, 0, 2, 4, 5, 6, 6, 6][dir],
                        _ => dir,
                    };
                }
                let damping = i32::from(h.cdef_damping) + i32::from(shift) - i32::from(plane > 0);
                let (x0, y0) = ((c * 4) >> sx, (r * 4) >> sy);
                let (w, ht) = (8 >> sx, 8 >> sy);
                let p = &input[plane];
                let max_w = (h.width as usize).div_ceil(1 << sx);
                let max_h = (h.height as usize).div_ceil(1 << sy);
                for y in y0..(y0 + ht).min(max_h) {
                    for x in x0..(x0 + w).min(max_w) {
                        let center = i32::from(p.samples[y * p.stride + x]);
                        let (mut lo, mut hi, mut sum) = (center, center, 0);
                        for k in 0..2 {
                            for sign in [-1, 1] {
                                for off in [0, -2, 2] {
                                    let dd = ((d as i32 + off) & 7) as usize;
                                    let (dy, dx) = DIRECTIONS[dd][k];
                                    let xx = x as i32 + sign * dx;
                                    let yy = y as i32 + sign * dy;
                                    if xx < 0 || yy < 0 || xx >= max_w as i32 || yy >= max_h as i32
                                    {
                                        continue;
                                    }
                                    let v =
                                        i32::from(p.samples[yy as usize * p.stride + xx as usize]);
                                    lo = lo.min(v);
                                    hi = hi.max(v);
                                    let threshold = if off == 0 { pri } else { sec };
                                    let taps = if off == 0 {
                                        if (pri >> shift) & 1 == 0 {
                                            [4, 2]
                                        } else {
                                            [3, 3]
                                        }
                                    } else {
                                        [2, 1]
                                    };
                                    sum += taps[k] * constrain(v - center, threshold, damping);
                                }
                            }
                        }
                        planes[plane].samples[y * p.stride + x] =
                            (center + ((8 + sum - i32::from(sum < 0)) >> 4)).clamp(lo, hi) as u16;
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod cdef_tests {
    use super::*;

    #[test]
    fn stack_and_rolling_deblock_match_original_windows() {
        let mut state = 511u32;
        for depth in [8, 10, 12] {
            for pass in 0..2 {
                for plane in 0..3 {
                    for size in [4, 8, 16] {
                        for pattern in 0..64 {
                            let scale = 1 << (depth - 8);
                            let samples = (0..24 * 24)
                                .map(|_| {
                                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                                    if pattern < 32 {
                                        (128 * scale + ((state >> 24) % 3) as i32 - 1) as u16
                                    } else {
                                        ((state >> 16) & ((1 << depth) - 1)) as u16
                                    }
                                })
                                .collect();
                            let input = DecodedPlane {
                                width: 24,
                                height: 24,
                                stride: 24,
                                samples,
                            };
                            for (x, y) in [(1, 1), (4, 4), (12, 12), (22, 22)] {
                                let mut scalar = input.clone();
                                let mut optimized = input.clone();
                                filter_sample_reference(
                                    &mut scalar,
                                    x,
                                    y,
                                    pass,
                                    plane,
                                    size,
                                    depth,
                                    63 * scale,
                                    134 * scale,
                                    3 * scale,
                                );
                                filter_sample(
                                    &mut optimized,
                                    x,
                                    y,
                                    pass,
                                    plane,
                                    size,
                                    depth,
                                    63 * scale,
                                    134 * scale,
                                    3 * scale,
                                );
                                assert_eq!(
                                    optimized, scalar,
                                    "depth={depth} pass={pass} plane={plane} size={size} pattern={pattern} at={x},{y}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn padded_scalar_and_neon_match_original_at_cropped_edges() {
        let bytes = [
            0x12, 0x00, 0x0a, 0x0b, 0x02, 0x00, 0x00, 0x05, 0x15, 0x7f, 0xfc, 0x4a, 0xf9, 0x00,
            0x40, 0x32, 0x0c, 0x10, 0x00, 0xac, 0x02, 0x05, 0x14, 0x20, 0x81, 0x00, 0x00, 0x98,
            0x80,
        ];
        let mut stream = super::super::ObuStream::new();
        let obus = stream.push(&bytes).unwrap();
        let mut s =
            SequenceHeader::parse(&obus.iter().find(|o| o.kind == 1).unwrap().payload).unwrap();
        let frame = obus.iter().find(|o| o.kind == 6).unwrap();
        let mut h =
            IntraFrameHeader::parse(&frame.payload, &s, frame.temporal_id, frame.spatial_id)
                .unwrap();
        s.enable_cdef = true;
        h.coded_lossless = false;
        h.allow_intrabc = false;
        h.cdef_strengths = vec![[15, 4, 15, 4], [7, 2, 9, 1], [0, 2, 0, 0], [0, 0, 15, 0]];
        for depth in [8, 10, 12] {
            s.bit_depth = depth;
            for (sx, sy) in [(false, false), (false, true), (true, false), (true, true)] {
                s.subsampling_x = sx;
                s.subsampling_y = sy;
                for (width, height) in [(8usize, 8usize), (17, 19), (64, 64), (73, 65)] {
                    h.width = width as u32;
                    h.height = height as u32;
                    let cols = width.div_ceil(8) * 2;
                    let rows = height.div_ceil(8) * 2;
                    let skips: Vec<_> = (0..cols * rows).map(|i| i % 11 < 3).collect();
                    let indices: Vec<_> = (0..cols.div_ceil(16) * rows.div_ceil(16))
                        .map(|i| if i % 5 == 4 { -1 } else { (i % 4) as i16 })
                        .collect();
                    for pattern in 0..8 {
                        h.cdef_damping = 3 + pattern % 4;
                        let mut state = 73u32;
                        let input: Vec<_> = (0..3)
                            .map(|plane| {
                                let px = usize::from(plane > 0 && sx);
                                let py = usize::from(plane > 0 && sy);
                                let w = cols * 4 >> px;
                                let ht = rows * 4 >> py;
                                let samples = (0..w * ht)
                                    .map(|i| {
                                        state =
                                            state.wrapping_mul(1664525).wrapping_add(1013904223);
                                        let x = i % w;
                                        let y = i / w;
                                        let base = match pattern {
                                            0 => 128,
                                            1 => (x * 3 + y * 2) % 256,
                                            2 => (x + y * 8) % 256,
                                            3 => (x * 8 + y) % 256,
                                            4 => (x * 3 + 256 - y % 256) % 256,
                                            _ => (state >> 24) as usize,
                                        };
                                        ((base << (depth - 8))
                                            | ((state >> 16) as usize & ((1 << (depth - 8)) - 1)))
                                            as u16
                                    })
                                    .collect();
                                DecodedPlane {
                                    width: w,
                                    height: ht,
                                    stride: w,
                                    samples,
                                }
                            })
                            .collect();
                        let mut expected = input.clone();
                        cdef_reference(&mut expected, &s, &h, &skips, cols, &indices).unwrap();
                        let mut actual = input;
                        cdef(&mut actual, &s, &h, &skips, cols, &indices).unwrap();
                        assert_eq!(
                            actual, expected,
                            "depth={depth} sx={sx} sy={sy} size={width}x{height} pattern={pattern}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn row_scalar_and_neon_match_all_directions_and_strengths() {
        let stride = 32;
        let mut state = 511u32;
        for depth in [8, 10, 12] {
            for d in 0i32..8 {
                for pri in 0..16 {
                    for sec in [0, 1, 2, 4] {
                        let mut source = vec![i16::MAX; stride * 9];
                        for y in 2..7 {
                            for x in 4..20 {
                                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                                source[y * stride + x] =
                                    ((state >> 16) & ((1 << depth) - 1)) as i16;
                            }
                        }
                        let mut taps = [CdefTap::default(); 12];
                        let mut n = 0;
                        for k in 0..2 {
                            for sign in [-1, 1] {
                                for off in [0, -2, 2] {
                                    let (dy, dx) = DIRECTIONS[((d + off) & 7) as usize][k];
                                    let threshold: i32 =
                                        (if off == 0 { pri } else { sec }) << (depth - 8);
                                    taps[n] = CdefTap {
                                        offset: sign
                                            * (dy as isize * stride as isize + dx as isize),
                                        threshold: threshold as u16,
                                        shift: if threshold == 0 {
                                            0
                                        } else {
                                            (3 + depth - 8 - threshold.ilog2() as i32).max(0) as u32
                                        },
                                        weight: if off == 0 {
                                            if pri & 1 == 0 { [4, 2][k] } else { 3 }
                                        } else {
                                            [2, 1][k]
                                        },
                                    };
                                    n += 1;
                                }
                            }
                        }
                        for count in 1..=8 {
                            let position = 2 * stride + 4;
                            let mut scalar = vec![0; count];
                            let mut simd = vec![0; count];
                            cdef_row_scalar(&source, position, &mut scalar, &taps);
                            cdef_row(&source, position, &mut simd, &taps, true);
                            assert_eq!(
                                scalar, simd,
                                "depth={depth} dir={d} pri={pri} sec={sec} count={count}"
                            );
                        }
                    }
                }
            }
        }
    }
}
