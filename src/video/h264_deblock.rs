//! Progressive 8-bit 4:2:0 in-loop filtering from H.264 (03/2005), section 8.7.

use super::h264::AvcError;
use super::h264_transform::chroma_qp;

const ALPHA: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15,
    17, 20, 22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144,
    162, 182, 203, 226, 255, 255,
];
const BETA: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7,
    7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18,
    18,
];
const TC0_BS3: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
    2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 23, 25,
];

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
        let p = [
            plane[q - across] as i32,
            plane[q - 2 * across] as i32,
            plane[q - 3 * across] as i32,
            plane[q - 4 * across] as i32,
        ];
        let r = [
            plane[q] as i32,
            plane[q + across] as i32,
            plane[q + 2 * across] as i32,
            plane[q + 3 * across] as i32,
        ];
        if (p[0] - r[0]).abs() >= alpha
            || (p[1] - p[0]).abs() >= beta
            || (r[1] - r[0]).abs() >= beta
        {
            continue;
        }
        let (mut p0, mut p1, mut p2) = (p[0], p[1], p[2]);
        let (mut q0, mut q1, mut q2) = (r[0], r[1], r[2]);
        if strength == 4 {
            let strong = (p[0] - r[0]).abs() < (alpha >> 2) + 2;
            if !chroma && strong && (p[2] - p[0]).abs() < beta {
                p0 = (p[2] + 2 * p[1] + 2 * p[0] + 2 * r[0] + r[1] + 4) >> 3;
                p1 = (p[2] + p[1] + p[0] + r[0] + 2) >> 2;
                p2 = (2 * p[3] + 3 * p[2] + p[1] + p[0] + r[0] + 4) >> 3;
            } else {
                p0 = (2 * p[1] + p[0] + r[1] + 2) >> 2;
            }
            if !chroma && strong && (r[2] - r[0]).abs() < beta {
                q0 = (p[1] + 2 * p[0] + 2 * r[0] + 2 * r[1] + r[2] + 4) >> 3;
                q1 = (p[0] + r[0] + r[1] + r[2] + 2) >> 2;
                q2 = (2 * r[3] + 3 * r[2] + r[1] + r[0] + p[0] + 4) >> 3;
            } else {
                q0 = (2 * r[1] + r[0] + p[1] + 2) >> 2;
            }
        } else {
            let tc0 = TC0_BS3[index_a];
            let ap = (p[2] - p[0]).abs() < beta;
            let aq = (r[2] - r[0]).abs() < beta;
            let tc = tc0 + if chroma { 1 } else { i32::from(ap) + i32::from(aq) };
            let delta = ((((r[0] - p[0]) << 2) + (p[1] - r[1]) + 4) >> 3).clamp(-tc, tc);
            p0 += delta;
            q0 -= delta;
            if !chroma && ap {
                p1 += ((p[2] + ((p[0] + r[0] + 1) >> 1) - 2 * p[1]) >> 1)
                    .clamp(-tc0, tc0);
            }
            if !chroma && aq {
                q1 += ((r[2] + ((p[0] + r[0] + 1) >> 1) - 2 * r[1]) >> 1)
                    .clamp(-tc0, tc0);
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
                let edge_qp = if edge == 0 { (qp + qps[mb - 1] + 1) >> 1 } else { qp };
                filter_edge(luma, width, x * 16 + edge * 4, y * 16, true, 16,
                    if edge == 0 { 4 } else { 3 }, edge_qp, alpha_offset, beta_offset, false);
            }
        }
        for (plane, offset) in [(cb as &mut [u8], chroma_offsets[0]), (cr as &mut [u8], chroma_offsets[1])] {
            let current = chroma_qp(qp, offset)?;
            if x != 0 {
                let previous = chroma_qp(qps[mb - 1], offset)?;
                filter_edge(plane, width / 2, x * 8, y * 8, true, 8, 4,
                    (current + previous + 1) >> 1, alpha_offset, beta_offset, true);
            }
            filter_edge(plane, width / 2, x * 8 + 4, y * 8, true, 8, 3,
                current, alpha_offset, beta_offset, true);
        }
        for edge in 0..4 {
            if (edge != 0 || y != 0) && (edge == 0 || edge == 2 || !transform8x8[mb]) {
                let edge_qp = if edge == 0 { (qp + qps[mb - mb_width] + 1) >> 1 } else { qp };
                filter_edge(luma, width, x * 16, y * 16 + edge * 4, false, 16,
                    if edge == 0 { 4 } else { 3 }, edge_qp, alpha_offset, beta_offset, false);
            }
        }
        for (plane, offset) in [(cb as &mut [u8], chroma_offsets[0]), (cr as &mut [u8], chroma_offsets[1])] {
            let current = chroma_qp(qp, offset)?;
            if y != 0 {
                let previous = chroma_qp(qps[mb - mb_width], offset)?;
                filter_edge(plane, width / 2, x * 8, y * 8, false, 8, 4,
                    (current + previous + 1) >> 1, alpha_offset, beta_offset, true);
            }
            filter_edge(plane, width / 2, x * 8, y * 8 + 4, false, 8, 3,
                current, alpha_offset, beta_offset, true);
        }
    }
    Ok(())
}
