//! Progressive 8-bit 4:2:0 in-loop filtering from H.264 (03/2005), section 8.7.

use super::h264::AvcError;
use super::h264_high::MotionCell;
use super::h264_transform::chroma_qp;

const ALPHA: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20,
    22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226,
    255, 255,
];
const BETA: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];
const fn tc0_bs3(index: usize) -> i32 {
    match index {
        0..=16 => 0,
        17..=26 => 1,
        27..=30 => 2,
        31..=33 => 3,
        34..=36 => 4,
        37 => 5,
        38..=39 => 6,
        40 => 7,
        41 => 8,
        42 => 9,
        43 => 10,
        44 => 11,
        45 => 13,
        46 => 14,
        47 => 16,
        48 => 18,
        49 => 20,
        50 => 23,
        _ => 25,
    }
}
const TC0_BS1: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13,
];
const fn tc0_bs2(index: usize) -> i32 {
    match index {
        0..=20 => 0,
        21..=30 => 1,
        31..=34 => 2,
        35..=37 => 3,
        38..=39 => 4,
        40..=41 => 5,
        42 => 6,
        43 => 7,
        44..=45 => 8,
        46 => 10,
        47 => 11,
        48 => 12,
        49 => 13,
        50 => 15,
        _ => 17,
    }
}

#[derive(Clone, Copy)]
pub(super) struct DeblockMb {
    pub qp: i32,
    pub intra: bool,
    pub transform8x8: bool,
    pub coded_luma: u8,
    pub motion: [MotionCell; 4],
}

#[inline]
fn clip(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

fn filter_edge(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    vertical: bool,
    length: usize,
    strength: u8,
    qp: i32,
    alpha_offset: i32,
    beta_offset: i32,
    chroma: bool,
) {
    if strength == 0 {
        return;
    }
    let index_a = (qp + alpha_offset).clamp(0, 51) as usize;
    let index_b = (qp + beta_offset).clamp(0, 51) as usize;
    let alpha = ALPHA[index_a];
    let beta = BETA[index_b];
    if alpha == 0 || beta == 0 {
        return;
    }
    let across = if vertical { 1 } else { stride };
    let along = if vertical { stride } else { 1 };
    let origin = y * stride + x;
    for line in 0..length {
        let q = origin + line * along;
        let p0 = plane[q - across] as i32;
        let p1 = plane[q - 2 * across] as i32;
        let q0 = plane[q] as i32;
        let q1 = plane[q + across] as i32;
        if (p0 - q0).abs() >= alpha
            || (p1 - p0).abs() >= beta
            || (q1 - q0).abs() >= beta
        {
            continue;
        }
        let p = [
            p0,
            p1,
            plane[q - 3 * across] as i32,
        ];
        let r = [
            q0,
            q1,
            plane[q + 2 * across] as i32,
        ];
        let (mut p0, mut p1, mut p2) = (p[0], p[1], p[2]);
        let (mut q0, mut q1, mut q2) = (r[0], r[1], r[2]);
        if strength == 4 {
            let strong = (p[0] - r[0]).abs() < (alpha >> 2) + 2;
            if !chroma && strong && (p[2] - p[0]).abs() < beta {
                p0 = (p[2] + 2 * p[1] + 2 * p[0] + 2 * r[0] + r[1] + 4) >> 3;
                p1 = (p[2] + p[1] + p[0] + r[0] + 2) >> 2;
                let p3 = plane[q - 4 * across] as i32;
                p2 = (2 * p3 + 3 * p[2] + p[1] + p[0] + r[0] + 4) >> 3;
            } else {
                p0 = (2 * p[1] + p[0] + r[1] + 2) >> 2;
            }
            if !chroma && strong && (r[2] - r[0]).abs() < beta {
                q0 = (p[1] + 2 * p[0] + 2 * r[0] + 2 * r[1] + r[2] + 4) >> 3;
                q1 = (p[0] + r[0] + r[1] + r[2] + 2) >> 2;
                let q3 = plane[q + 3 * across] as i32;
                q2 = (2 * q3 + 3 * r[2] + r[1] + r[0] + p[0] + 4) >> 3;
            } else {
                q0 = (2 * r[1] + r[0] + p[1] + 2) >> 2;
            }
        } else {
            let tc0 = match strength {
                1 => TC0_BS1[index_a],
                2 => tc0_bs2(index_a),
                _ => tc0_bs3(index_a),
            };
            let ap = (p[2] - p[0]).abs() < beta;
            let aq = (r[2] - r[0]).abs() < beta;
            let tc = tc0
                + if chroma {
                    1
                } else {
                    i32::from(ap) + i32::from(aq)
                };
            let delta = ((((r[0] - p[0]) << 2) + (p[1] - r[1]) + 4) >> 3).clamp(-tc, tc);
            p0 += delta;
            q0 -= delta;
            if !chroma && ap {
                p1 += ((p[2] + ((p[0] + r[0] + 1) >> 1) - 2 * p[1]) >> 1).clamp(-tc0, tc0);
            }
            if !chroma && aq {
                q1 += ((r[2] + ((p[0] + r[0] + 1) >> 1) - 2 * r[1]) >> 1).clamp(-tc0, tc0);
            }
        }
        plane[q - across] = clip(p0);
        plane[q] = clip(q0);
        if !chroma {
            plane[q - 2 * across] = clip(p1);
            plane[q + across] = clip(q1);
            if strength == 4 {
                plane[q - 3 * across] = clip(p2);
                plane[q + 2 * across] = clip(q2);
            }
        }
    }
}

pub(super) fn filter_intra_picture(
    luma: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    qps: &[i32],
    transform8x8: &[bool],
    chroma_offsets: [i32; 2],
    alpha_offset: i32,
    beta_offset: i32,
) -> Result<(), AvcError> {
    let mb_width = width / 16;
    for mb in 0..qps.len() {
        let x = mb % mb_width;
        let y = mb / mb_width;
        let qp = qps[mb];
        for edge in 0..4 {
            if (edge != 0 || x != 0) && (edge == 0 || edge == 2 || !transform8x8[mb]) {
                let edge_qp = if edge == 0 {
                    (qp + qps[mb - 1] + 1) >> 1
                } else {
                    qp
                };
                filter_edge(
                    luma,
                    width,
                    x * 16 + edge * 4,
                    y * 16,
                    true,
                    16,
                    if edge == 0 { 4 } else { 3 },
                    edge_qp,
                    alpha_offset,
                    beta_offset,
                    false,
                );
            }
        }
        for (plane, offset) in [
            (cb as &mut [u8], chroma_offsets[0]),
            (cr as &mut [u8], chroma_offsets[1]),
        ] {
            let current = chroma_qp(qp, offset)?;
            if x != 0 {
                let previous = chroma_qp(qps[mb - 1], offset)?;
                filter_edge(
                    plane,
                    width / 2,
                    x * 8,
                    y * 8,
                    true,
                    8,
                    4,
                    (current + previous + 1) >> 1,
                    alpha_offset,
                    beta_offset,
                    true,
                );
            }
            filter_edge(
                plane,
                width / 2,
                x * 8 + 4,
                y * 8,
                true,
                8,
                3,
                current,
                alpha_offset,
                beta_offset,
                true,
            );
        }
        for edge in 0..4 {
            if (edge != 0 || y != 0) && (edge == 0 || edge == 2 || !transform8x8[mb]) {
                let edge_qp = if edge == 0 {
                    (qp + qps[mb - mb_width] + 1) >> 1
                } else {
                    qp
                };
                filter_edge(
                    luma,
                    width,
                    x * 16,
                    y * 16 + edge * 4,
                    false,
                    16,
                    if edge == 0 { 4 } else { 3 },
                    edge_qp,
                    alpha_offset,
                    beta_offset,
                    false,
                );
            }
        }
        for (plane, offset) in [
            (cb as &mut [u8], chroma_offsets[0]),
            (cr as &mut [u8], chroma_offsets[1]),
        ] {
            let current = chroma_qp(qp, offset)?;
            if y != 0 {
                let previous = chroma_qp(qps[mb - mb_width], offset)?;
                filter_edge(
                    plane,
                    width / 2,
                    x * 8,
                    y * 8,
                    false,
                    8,
                    4,
                    (current + previous + 1) >> 1,
                    alpha_offset,
                    beta_offset,
                    true,
                );
            }
            filter_edge(
                plane,
                width / 2,
                x * 8,
                y * 8 + 4,
                false,
                8,
                3,
                current,
                alpha_offset,
                beta_offset,
                true,
            );
        }
    }
    Ok(())
}

fn motion_differs(a: MotionCell, b: MotionCell) -> bool {
    [a.l0, a.l1]
        .into_iter()
        .zip([b.l0, b.l1])
        .any(|(left, right)| match (left, right) {
            (Some((left_ref, left_mv)), Some((right_ref, right_mv))) => {
                left_ref != right_ref
                    || (left_mv[0] - right_mv[0]).abs() >= 4
                    || (left_mv[1] - right_mv[1]).abs() >= 4
            }
            (None, None) => false,
            _ => true,
        })
}

fn boundary_strength(
    current: &DeblockMb,
    previous: &DeblockMb,
    current_region: usize,
    previous_region: usize,
    mb_edge: bool,
) -> u8 {
    if current.intra || previous.intra {
        return if mb_edge { 4 } else { 3 };
    }
    if current.coded_luma & (1 << current_region) != 0
        || previous.coded_luma & (1 << previous_region) != 0
    {
        return 2;
    }
    if motion_differs(
        current.motion[current_region],
        previous.motion[previous_region],
    ) {
        return 1;
    }
    0
}

pub(super) fn filter_inter_picture(
    luma: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    macroblocks: &[DeblockMb],
    chroma_offsets: [i32; 2],
    alpha_offset: i32,
    beta_offset: i32,
) -> Result<(), AvcError> {
    let mb_width = width / 16;
    for mb in 0..macroblocks.len() {
        let x = mb % mb_width;
        let y = mb / mb_width;
        let current = &macroblocks[mb];
        let mut vertical_chroma_strength = [[0u8; 2]; 2];
        for edge in 0..4 {
            if edge == 0 && x == 0 || edge % 2 != 0 && current.transform8x8 {
                continue;
            }
            let previous = if edge == 0 {
                &macroblocks[mb - 1]
            } else {
                current
            };
            let qp = if edge == 0 {
                (current.qp + previous.qp + 1) >> 1
            } else {
                current.qp
            };
            for segment in 0..4 {
                let q_region = (segment / 2) * 2 + edge / 2;
                let p_region = (segment / 2) * 2 + if edge == 0 { 1 } else { (edge - 1) / 2 };
                let strength = boundary_strength(current, previous, q_region, p_region, edge == 0);
                if edge == 0 || edge == 2 {
                    vertical_chroma_strength[edge / 2][segment / 2] = strength;
                }
                filter_edge(
                    luma, width, x * 16 + edge * 4, y * 16 + segment * 4,
                    true, 4, strength, qp, alpha_offset, beta_offset, false,
                );
            }
        }
        for (plane, offset) in [
            (cb as &mut [u8], chroma_offsets[0]),
            (cr as &mut [u8], chroma_offsets[1]),
        ] {
            for edge in 0..2 {
                if edge == 0 && x == 0 {
                    continue;
                }
                let previous = if edge == 0 { &macroblocks[mb - 1] } else { current };
                let qpc = chroma_qp(current.qp, offset)?;
                let ppc = chroma_qp(previous.qp, offset)?;
                let qp = (qpc + ppc + 1) >> 1;
                for segment in 0..2 {
                    let strength = vertical_chroma_strength[edge][segment];
                    filter_edge(
                        plane, width / 2, x * 8 + edge * 4, y * 8 + segment * 4,
                        true, 4, strength, qp, alpha_offset, beta_offset, true,
                    );
                }
            }
        }
        let mut horizontal_chroma_strength = [[0u8; 2]; 2];
        for edge in 0..4 {
            if edge == 0 && y == 0 || edge % 2 != 0 && current.transform8x8 {
                continue;
            }
            let previous = if edge == 0 {
                &macroblocks[mb - mb_width]
            } else {
                current
            };
            let qp = if edge == 0 {
                (current.qp + previous.qp + 1) >> 1
            } else {
                current.qp
            };
            for segment in 0..4 {
                let q_region = (edge / 2) * 2 + segment / 2;
                let p_region = (if edge == 0 { 1 } else { (edge - 1) / 2 }) * 2 + segment / 2;
                let strength = boundary_strength(current, previous, q_region, p_region, edge == 0);
                if edge == 0 || edge == 2 {
                    horizontal_chroma_strength[edge / 2][segment / 2] = strength;
                }
                filter_edge(
                    luma, width, x * 16 + segment * 4, y * 16 + edge * 4,
                    false, 4, strength, qp, alpha_offset, beta_offset, false,
                );
            }
        }
        for (plane, offset) in [
            (cb as &mut [u8], chroma_offsets[0]),
            (cr as &mut [u8], chroma_offsets[1]),
        ] {
            for edge in 0..2 {
                if edge == 0 && y == 0 {
                    continue;
                }
                let previous = if edge == 0 { &macroblocks[mb - mb_width] } else { current };
                let qpc = chroma_qp(current.qp, offset)?;
                let ppc = chroma_qp(previous.qp, offset)?;
                let qp = (qpc + ppc + 1) >> 1;
                for segment in 0..2 {
                    let strength = horizontal_chroma_strength[edge][segment];
                    filter_edge(
                        plane, width / 2, x * 8 + segment * 4, y * 8 + edge * 4,
                        false, 4, strength, qp, alpha_offset, beta_offset, true,
                    );
                }
            }
        }
    }
    Ok(())
}
