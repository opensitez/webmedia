//! Extended-edge intra prediction, AV1 sections 7.11.2.3-12.
use super::{syntax::Error, tables};

fn strength(w: usize, h: usize, smooth: bool, d: i32) -> usize {
    let d = d.abs();
    let n = w + h;
    if smooth {
        match n {
            0..=8 => {
                if d >= 64 {
                    2
                } else {
                    usize::from(d >= 40)
                }
            }
            9..=16 => {
                if d >= 48 {
                    2
                } else {
                    usize::from(d >= 20)
                }
            }
            17..=24 => {
                if d >= 4 {
                    3
                } else {
                    0
                }
            }
            _ => 3,
        }
    } else {
        match n {
            0..=8 => usize::from(d >= 56),
            9..=16 => usize::from(d >= 40),
            17..=24 => {
                if d >= 32 {
                    3
                } else if d >= 16 {
                    2
                } else {
                    usize::from(d >= 8)
                }
            }
            25..=32 => {
                if d >= 32 {
                    3
                } else if d >= 4 {
                    2
                } else {
                    1
                }
            }
            _ => 3,
        }
    }
}
fn use_upsample(w: usize, h: usize, smooth: bool, d: i32) -> bool {
    let d = d.abs();
    d > 0 && d < 40 && w + h <= if smooth { 8 } else { 16 }
}
fn filter(edge: &mut [i32], num: usize, s: usize) {
    if s == 0 {
        return;
    }
    const K: [[i32; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
    let input: Vec<i32> = (0..num).map(|i| edge[i + 1]).collect();
    for i in 1..num {
        let sum: i32 = (0..5)
            .map(|j| {
                K[s - 1][j]
                    * input[(i as isize + j as isize - 2).clamp(0, num as isize - 1) as usize]
            })
            .sum();
        edge[i + 1] = (sum + 8) >> 4;
    }
}
fn upsample(edge: &mut [i32], num: usize, max: i32) {
    let mut dup = vec![edge[1]];
    dup.extend_from_slice(&edge[1..num + 2]);
    dup.push(edge[num + 1]);
    edge[0] = dup[0];
    for i in 0..num {
        edge[2 * i + 1] =
            ((-dup[i] + 9 * dup[i + 1] + 9 * dup[i + 2] - dup[i + 3] + 8) >> 4).clamp(0, max);
        edge[2 * i + 2] = dup[i + 2];
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn directional(
    w: usize,
    h: usize,
    depth: u8,
    mode: usize,
    delta: i32,
    above: Option<&[u16]>,
    left: Option<&[u16]>,
    corner: u16,
    edge_filter: bool,
    smooth: bool,
    remaining_w: usize,
    remaining_h: usize,
) -> Result<Vec<u16>, Error> {
    if !(1..=8).contains(&mode) || !(-3..=3).contains(&delta) {
        return Err(Error::Invalid("directional mode or angle"));
    }
    let angle = [0, 90, 180, 45, 135, 113, 157, 203, 67][mode] + 3 * delta;
    let midpoint = 1i32 << (depth - 1);
    let max = (1i32 << depth) - 1;
    let size = 2 * (w + h) + 4;
    let mut top = vec![
        i32::from(above.and_then(|a| a.first()).copied().unwrap_or_else(|| {
            left.and_then(|l| l.first())
                .copied()
                .unwrap_or((midpoint - 1) as u16)
        }));
        size
    ];
    let mut side = vec![
        i32::from(left.and_then(|l| l.first()).copied().unwrap_or_else(|| {
            above
                .and_then(|a| a.first())
                .copied()
                .unwrap_or((midpoint + 1) as u16)
        }));
        size
    ];
    top[0] = i32::from(corner);
    top[1] = i32::from(corner);
    side[0] = i32::from(corner);
    side[1] = i32::from(corner);
    for i in 0..w + h {
        if let Some(a) = above {
            top[i + 2] = i32::from(a[i.min(a.len() - 1)]);
        }
        if let Some(l) = left {
            side[i + 2] = i32::from(l[i.min(l.len() - 1)]);
        }
    }
    let (mut ua, mut ul) = (false, false);
    if edge_filter {
        if angle != 90 && angle != 180 {
            if angle > 90 && angle < 180 && w + h >= 24 {
                let c = (side[2] * 5 + top[1] * 6 + top[2] * 5 + 8) >> 4;
                top[1] = c;
                side[1] = c;
            }
            if above.is_some() {
                filter(
                    &mut top,
                    w.min(remaining_w) + if angle < 90 { h } else { 0 } + 1,
                    strength(w, h, smooth, angle - 90),
                );
            }
            if left.is_some() {
                filter(
                    &mut side,
                    h.min(remaining_h) + if angle > 180 { w } else { 0 } + 1,
                    strength(w, h, smooth, angle - 180),
                );
            }
        }
        ua = use_upsample(w, h, smooth, angle - 90);
        ul = use_upsample(w, h, smooth, angle - 180);
        if ua {
            upsample(&mut top, w + if angle < 90 { h } else { 0 }, max);
        }
        if ul {
            upsample(&mut side, h + if angle > 180 { w } else { 0 }, max);
        }
    }
    let dx = if angle < 90 {
        i32::from(tables::ANGLE_DERIVATIVE[angle as usize])
    } else if angle > 90 && angle < 180 {
        i32::from(tables::ANGLE_DERIVATIVE[(180 - angle) as usize])
    } else {
        0
    };
    let dy = if angle > 180 {
        i32::from(tables::ANGLE_DERIVATIVE[(270 - angle) as usize])
    } else if angle > 90 && angle < 180 {
        i32::from(tables::ANGLE_DERIVATIVE[(angle - 90) as usize])
    } else {
        0
    };
    let lerp = |edge: &[i32], base: i32, shift: i32| -> Result<u16, Error> {
        let i =
            usize::try_from(base + 2).map_err(|_| Error::Invalid("directional edge underflow"))?;
        let a = *edge
            .get(i)
            .ok_or(Error::Invalid("directional edge overflow"))?;
        let b = *edge
            .get(i + 1)
            .ok_or(Error::Invalid("directional edge overflow"))?;
        Ok(((a * (32 - shift) + b * shift + 16) >> 5) as u16)
    };
    let mut output = vec![0; w * h];
    for y in 0..h {
        for x in 0..w {
            let (x, y) = (x as i32, y as i32);
            let a = i32::from(ua);
            let l = i32::from(ul);
            let v = if angle < 90 {
                let idx = (y + 1) * dx;
                let base = (idx >> (6 - a)) + (x << a);
                let limit = ((w + h - 1) as i32) << a;
                if base < limit {
                    lerp(&top, base, ((idx << a) >> 1) & 31)?
                } else {
                    top[(limit + 2) as usize] as u16
                }
            } else if angle == 90 {
                top[x as usize + 2] as u16
            } else if angle < 180 {
                let idx = (x << 6) - (y + 1) * dx;
                let base = idx >> (6 - a);
                if base >= -(1 << a) {
                    lerp(&top, base, ((idx << a) >> 1) & 31)?
                } else {
                    let idx = (y << 6) - (x + 1) * dy;
                    lerp(&side, idx >> (6 - l), ((idx << l) >> 1) & 31)?
                }
            } else if angle == 180 {
                side[y as usize + 2] as u16
            } else {
                let idx = (x + 1) * dy;
                lerp(&side, (idx >> (6 - l)) + (y << l), ((idx << l) >> 1) & 31)?
            };
            output[y as usize * w + x as usize] = v;
        }
    }
    Ok(output)
}

fn weights(n: usize) -> Result<&'static [u16], Error> {
    Ok(match n {
        4 => &tables::SMOOTH4,
        8 => &tables::SMOOTH8,
        16 => &tables::SMOOTH16,
        32 => &tables::SMOOTH32,
        64 => &tables::SMOOTH64,
        _ => return Err(Error::Invalid("smooth dimensions")),
    })
}
pub(crate) fn smooth(
    w: usize,
    h: usize,
    mode: usize,
    top: &[u16],
    side: &[u16],
) -> Result<Vec<u16>, Error> {
    let wx = weights(w)?;
    let wy = weights(h)?;
    let mut out = vec![0; w * h];
    for y in 0..h {
        for x in 0..w {
            let v = i32::from(wy[y]) * i32::from(top[x])
                + (256 - i32::from(wy[y])) * i32::from(side[h - 1]);
            let u = i32::from(wx[x]) * i32::from(side[y])
                + (256 - i32::from(wx[x])) * i32::from(top[w - 1]);
            out[y * w + x] = match mode {
                9 => ((u + v + 256) >> 9) as u16,
                10 => ((v + 128) >> 8) as u16,
                11 => ((u + 128) >> 8) as u16,
                _ => return Err(Error::Invalid("smooth mode")),
            };
        }
    }
    Ok(out)
}

pub(crate) fn recursive(
    w: usize,
    h: usize,
    mode: usize,
    depth: u8,
    top: &[u16],
    side: &[u16],
    corner: u16,
) -> Result<Vec<u16>, Error> {
    if mode >= 5 || w > 32 || h > 32 {
        return Err(Error::Invalid("filter intra inputs"));
    }
    let mut out = vec![0u16; w * h];
    let max = (1i32 << depth) - 1;
    for i2 in 0..h / 2 {
        for j4 in 0..w / 4 {
            let mut p = [0i32; 7];
            for i in 0..7 {
                let x = j4 * 4;
                let y = i2 * 2;
                p[i] = i32::from(if i < 5 {
                    if i2 == 0 {
                        if x + i == 0 { corner } else { top[x + i - 1] }
                    } else if j4 == 0 && i == 0 {
                        side[y - 1]
                    } else {
                        out[(y - 1) * w + x + i - 1]
                    }
                } else if j4 == 0 {
                    side[y + i - 5]
                } else {
                    out[(y + i - 5) * w + x - 1]
                });
            }
            for i in 0..8 {
                let base = (mode * 8 + i) * 7;
                let sum: i32 = (0..7)
                    .map(|j| i32::from(tables::FILTER_TAPS[base + j]) * p[j])
                    .sum();
                let value = sum.signum() * ((sum.abs() + 8) >> 4);
                out[(i2 * 2 + i / 4) * w + j4 * 4 + i % 4] = value.clamp(0, max) as u16;
            }
        }
    }
    Ok(out)
}
