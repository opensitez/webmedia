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
    // Nonzero coefficients in raster-ordered 4x4 luma blocks (8.7.2.1).
    pub coded_luma: u16,
    // Raster 4x4 motion cells, with reference indices normalized to picture identities.
    pub motion: [MotionCell; 16],
}

#[inline]
fn clip(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

#[cfg(test)]
#[inline]
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
    filter_edge_dispatch::<2>(
        plane,
        stride,
        x,
        y,
        vertical,
        length,
        strength,
        qp,
        alpha_offset,
        beta_offset,
        chroma,
    );
}

#[inline]
fn filter_edge_dispatch<const VECTOR: u8>(
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
    // Direction and plane type are fixed for the entire edge, not each pixel.
    match (vertical, chroma) {
        (true, false) => filter_edge_impl::<true, true, false, VECTOR>(
            plane,
            stride,
            x,
            y,
            length,
            strength,
            qp,
            alpha_offset,
            beta_offset,
            vertical,
            chroma,
        ),
        (false, false) => filter_edge_impl::<true, false, false, VECTOR>(
            plane,
            stride,
            x,
            y,
            length,
            strength,
            qp,
            alpha_offset,
            beta_offset,
            vertical,
            chroma,
        ),
        (true, true) => filter_edge_impl::<true, true, true, VECTOR>(
            plane,
            stride,
            x,
            y,
            length,
            strength,
            qp,
            alpha_offset,
            beta_offset,
            vertical,
            chroma,
        ),
        (false, true) => filter_edge_impl::<true, false, true, VECTOR>(
            plane,
            stride,
            x,
            y,
            length,
            strength,
            qp,
            alpha_offset,
            beta_offset,
            vertical,
            chroma,
        ),
    }
}

fn filter_edge_impl<
    const SPECIALIZE: bool,
    const VERTICAL: bool,
    const CHROMA: bool,
    const VECTOR: u8,
>(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    length: usize,
    strength: u8,
    qp: i32,
    alpha_offset: i32,
    beta_offset: i32,
    vertical: bool,
    chroma: bool,
) {
    let vertical = if SPECIALIZE { VERTICAL } else { vertical };
    let chroma = if SPECIALIZE { CHROMA } else { chroma };
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
    let first_scalar = 0;
    #[cfg(target_arch = "aarch64")]
    let first_scalar =
        if VECTOR != 0 && (VECTOR == 2 || !VERTICAL) && SPECIALIZE && strength < 4 && length >= 4 {
            let tc0 = match strength {
                1 => TC0_BS1[index_a],
                2 => tc0_bs2(index_a),
                _ => tc0_bs3(index_a),
            };
            let mut line = 0;
            while line + 4 <= length {
                unsafe {
                    filter_normal_four::<VERTICAL, CHROMA>(
                        plane,
                        origin + line * along,
                        across,
                        along,
                        alpha,
                        beta,
                        tc0,
                    );
                }
                line += 4;
            }
            line
        } else {
            first_scalar
        };
    for line in first_scalar..length {
        let q = origin + line * along;
        let p0 = plane[q - across] as i32;
        let p1 = plane[q - 2 * across] as i32;
        let q0 = plane[q] as i32;
        let q1 = plane[q + across] as i32;
        if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
            continue;
        }
        let p = [p0, p1, plane[q - 3 * across] as i32];
        let r = [q0, q1, plane[q + 2 * across] as i32];
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

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn filter_normal_four<const VERTICAL: bool, const CHROMA: bool>(
    plane: &mut [u8],
    q: usize,
    across: usize,
    along: usize,
    alpha: i32,
    beta: i32,
    tc0: i32,
) {
    use std::arch::aarch64::*;

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn load<const VERTICAL: bool>(plane: &[u8], at: usize, along: usize) -> int32x4_t {
        #[cfg(target_endian = "little")]
        if !VERTICAL {
            let word = u32::from_le_bytes(plane[at..at + 4].try_into().unwrap());
            let bytes = vmovl_u8(vcreate_u8(u64::from(word)));
            return vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(bytes)));
        }
        let values = [
            plane[at] as i32,
            plane[at + along] as i32,
            plane[at + 2 * along] as i32,
            plane[at + 3 * along] as i32,
        ];
        unsafe { vld1q_s32(values.as_ptr()) }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn store<const VERTICAL: bool>(
        plane: &mut [u8],
        at: usize,
        along: usize,
        value: int32x4_t,
    ) {
        let bytes = vqmovn_u16(vcombine_u16(vqmovun_s32(value), vdup_n_u16(0)));
        #[cfg(target_endian = "little")]
        if !VERTICAL {
            let word = vget_lane_u32::<0>(vreinterpret_u32_u8(bytes));
            plane[at..at + 4].copy_from_slice(&word.to_le_bytes());
            return;
        }
        plane[at] = vget_lane_u8::<0>(bytes);
        plane[at + along] = vget_lane_u8::<1>(bytes);
        plane[at + 2 * along] = vget_lane_u8::<2>(bytes);
        plane[at + 3 * along] = vget_lane_u8::<3>(bytes);
    }

    unsafe {
        let p0 = load::<VERTICAL>(plane, q - across, along);
        let q0 = load::<VERTICAL>(plane, q, along);
        let across_enabled = vcltq_s32(vabsq_s32(vsubq_s32(p0, q0)), vdupq_n_s32(alpha));
        if vmaxvq_u32(across_enabled) == 0 {
            return;
        }
        let p1 = load::<VERTICAL>(plane, q - 2 * across, along);
        let q1 = load::<VERTICAL>(plane, q + across, along);
        let beta = vdupq_n_s32(beta);
        let enabled = vandq_u32(
            across_enabled,
            vandq_u32(
                vcltq_s32(vabsq_s32(vsubq_s32(p1, p0)), beta),
                vcltq_s32(vabsq_s32(vsubq_s32(q1, q0)), beta),
            ),
        );
        if vmaxvq_u32(enabled) == 0 {
            return;
        }
        let zero = vdupq_n_s32(0);
        let tc0 = vdupq_n_s32(tc0);
        let (p2, q2, ap, aq) = if CHROMA {
            (zero, zero, vdupq_n_u32(0), vdupq_n_u32(0))
        } else {
            let p2 = load::<VERTICAL>(plane, q - 3 * across, along);
            let q2 = load::<VERTICAL>(plane, q + 2 * across, along);
            (
                p2,
                q2,
                vcltq_s32(vabsq_s32(vsubq_s32(p2, p0)), beta),
                vcltq_s32(vabsq_s32(vsubq_s32(q2, q0)), beta),
            )
        };
        let tc = if CHROMA {
            vaddq_s32(tc0, vdupq_n_s32(1))
        } else {
            vaddq_s32(
                tc0,
                vreinterpretq_s32_u32(vaddq_u32(vshrq_n_u32::<31>(ap), vshrq_n_u32::<31>(aq))),
            )
        };
        let delta = vshrq_n_s32::<3>(vaddq_s32(
            vaddq_s32(vshlq_n_s32::<2>(vsubq_s32(q0, p0)), vsubq_s32(p1, q1)),
            vdupq_n_s32(4),
        ));
        let delta = vmaxq_s32(vnegq_s32(tc), vminq_s32(tc, delta));
        store::<VERTICAL>(
            plane,
            q - across,
            along,
            vbslq_s32(enabled, vaddq_s32(p0, delta), p0),
        );
        store::<VERTICAL>(
            plane,
            q,
            along,
            vbslq_s32(enabled, vsubq_s32(q0, delta), q0),
        );
        if !CHROMA {
            // Secondary updates use the original p0/q0, never the modified samples.
            let mean = vshrq_n_s32::<1>(vaddq_s32(vaddq_s32(p0, q0), vdupq_n_s32(1)));
            let dp = vshrq_n_s32::<1>(vsubq_s32(vaddq_s32(p2, mean), vshlq_n_s32::<1>(p1)));
            let dq = vshrq_n_s32::<1>(vsubq_s32(vaddq_s32(q2, mean), vshlq_n_s32::<1>(q1)));
            let dp = vmaxq_s32(vnegq_s32(tc0), vminq_s32(tc0, dp));
            let dq = vmaxq_s32(vnegq_s32(tc0), vminq_s32(tc0, dq));
            store::<VERTICAL>(
                plane,
                q - 2 * across,
                along,
                vbslq_s32(vandq_u32(enabled, ap), vaddq_s32(p1, dp), p1),
            );
            store::<VERTICAL>(
                plane,
                q + across,
                along,
                vbslq_s32(vandq_u32(enabled, aq), vaddq_s32(q1, dq), q1),
            );
        }
    }
}

#[cfg(test)]
std::thread_local! {
    static VECTOR_DEBLOCK: std::cell::Cell<u8> = const { std::cell::Cell::new(2) };
}

#[cfg(test)]
pub(super) fn with_deblock_mode<T>(mode: u8, action: impl FnOnce() -> T) -> T {
    assert!(mode <= 2);
    struct Restore(u8);
    impl Drop for Restore {
        fn drop(&mut self) {
            VECTOR_DEBLOCK.with(|cell| cell.set(self.0));
        }
    }
    let _restore = Restore(VECTOR_DEBLOCK.with(|cell| cell.replace(mode)));
    action()
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
    #[cfg(test)]
    if VECTOR_DEBLOCK.with(|cell| cell.get()) != 2 {
        let mode = VECTOR_DEBLOCK.with(|cell| cell.get());
        let filter = if mode == 0 {
            filter_intra_picture_impl::<0>
        } else {
            filter_intra_picture_impl::<1>
        };
        return filter(
            luma,
            cb,
            cr,
            width,
            qps,
            transform8x8,
            chroma_offsets,
            alpha_offset,
            beta_offset,
        );
    }
    filter_intra_picture_impl::<2>(
        luma,
        cb,
        cr,
        width,
        qps,
        transform8x8,
        chroma_offsets,
        alpha_offset,
        beta_offset,
    )
}

fn filter_intra_picture_impl<const VECTOR: u8>(
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
                filter_edge_dispatch::<VECTOR>(
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
                filter_edge_dispatch::<VECTOR>(
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
            filter_edge_dispatch::<VECTOR>(
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
                filter_edge_dispatch::<VECTOR>(
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
                filter_edge_dispatch::<VECTOR>(
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
            filter_edge_dispatch::<VECTOR>(
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
    let count = |cell: MotionCell| usize::from(cell.l0.is_some()) + usize::from(cell.l1.is_some());
    if count(a) != count(b) {
        return true;
    }
    let matches = |a: (u8, [i32; 2]), b: (u8, [i32; 2])| {
        a.0 == b.0 && (a.1[0] - b.1[0]).abs() < 4 && (a.1[1] - b.1[1]).abs() < 4
    };
    match count(a) {
        0 => false,
        1 => !matches(a.l0.or(a.l1).unwrap(), b.l0.or(b.l1).unwrap()),
        _ => {
            let (a0, a1, b0, b1) = (a.l0.unwrap(), a.l1.unwrap(), b.l0.unwrap(), b.l1.unwrap());
            !((matches(a0, b0) && matches(a1, b1)) || (matches(a0, b1) && matches(a1, b0)))
        }
    }
}

fn boundary_strength(
    current: &DeblockMb,
    previous: &DeblockMb,
    current_cell: usize,
    previous_cell: usize,
    mb_edge: bool,
) -> u8 {
    if current.intra || previous.intra {
        return if mb_edge { 4 } else { 3 };
    }
    if current.coded_luma & (1 << current_cell) != 0
        || previous.coded_luma & (1 << previous_cell) != 0
    {
        return 2;
    }
    if motion_differs(current.motion[current_cell], previous.motion[previous_cell]) {
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
    #[cfg(test)]
    if VECTOR_DEBLOCK.with(|cell| cell.get()) != 2 {
        let mode = VECTOR_DEBLOCK.with(|cell| cell.get());
        let filter = if mode == 0 {
            filter_inter_picture_impl::<0>
        } else {
            filter_inter_picture_impl::<1>
        };
        return filter(
            luma,
            cb,
            cr,
            width,
            macroblocks,
            chroma_offsets,
            alpha_offset,
            beta_offset,
        );
    }
    filter_inter_picture_impl::<2>(
        luma,
        cb,
        cr,
        width,
        macroblocks,
        chroma_offsets,
        alpha_offset,
        beta_offset,
    )
}

fn filter_inter_picture_impl<const VECTOR: u8>(
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
        let mut vertical_chroma_strength = [[0u8; 4]; 2];
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
                let q_region = segment * 4 + edge;
                let p_region = segment * 4 + if edge == 0 { 3 } else { edge - 1 };
                let strength = boundary_strength(current, previous, q_region, p_region, edge == 0);
                if edge == 0 || edge == 2 {
                    vertical_chroma_strength[edge / 2][segment] = strength;
                }
                filter_edge_dispatch::<VECTOR>(
                    luma,
                    width,
                    x * 16 + edge * 4,
                    y * 16 + segment * 4,
                    true,
                    4,
                    strength,
                    qp,
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
            for edge in 0..2 {
                if edge == 0 && x == 0 {
                    continue;
                }
                let previous = if edge == 0 {
                    &macroblocks[mb - 1]
                } else {
                    current
                };
                let qpc = chroma_qp(current.qp, offset)?;
                let ppc = chroma_qp(previous.qp, offset)?;
                let qp = (qpc + ppc + 1) >> 1;
                for segment in 0..4 {
                    let strength = vertical_chroma_strength[edge][segment];
                    filter_edge_dispatch::<VECTOR>(
                        plane,
                        width / 2,
                        x * 8 + edge * 4,
                        y * 8 + segment * 2,
                        true,
                        2,
                        strength,
                        qp,
                        alpha_offset,
                        beta_offset,
                        true,
                    );
                }
            }
        }
        let mut horizontal_chroma_strength = [[0u8; 4]; 2];
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
                let q_region = edge * 4 + segment;
                let p_region = (if edge == 0 { 3 } else { edge - 1 }) * 4 + segment;
                let strength = boundary_strength(current, previous, q_region, p_region, edge == 0);
                if edge == 0 || edge == 2 {
                    horizontal_chroma_strength[edge / 2][segment] = strength;
                }
                filter_edge_dispatch::<VECTOR>(
                    luma,
                    width,
                    x * 16 + segment * 4,
                    y * 16 + edge * 4,
                    false,
                    4,
                    strength,
                    qp,
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
            for edge in 0..2 {
                if edge == 0 && y == 0 {
                    continue;
                }
                let previous = if edge == 0 {
                    &macroblocks[mb - mb_width]
                } else {
                    current
                };
                let qpc = chroma_qp(current.qp, offset)?;
                let ppc = chroma_qp(previous.qp, offset)?;
                let qp = (qpc + ppc + 1) >> 1;
                for segment in 0..4 {
                    let strength = horizontal_chroma_strength[edge][segment];
                    filter_edge_dispatch::<VECTOR>(
                        plane,
                        width / 2,
                        x * 8 + segment * 2,
                        y * 8 + edge * 4,
                        false,
                        2,
                        strength,
                        qp,
                        alpha_offset,
                        beta_offset,
                        true,
                    );
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_edges_match_scalar_at_branch_and_tail_boundaries() {
        for qp in 0..52 {
            for alpha_offset in [-12, 0, 12] {
                for beta_offset in [-12, 0, 12] {
                    let alpha = ALPHA[(qp + alpha_offset).clamp(0, 51) as usize];
                    let beta = BETA[(qp + beta_offset).clamp(0, 51) as usize];
                    for strength in 0..=4 {
                        for vertical in [false, true] {
                            for chroma in [false, true] {
                                let (across, along) = if vertical { (1, 16) } else { (16, 1) };
                                for length in [2, 4, 6, 8] {
                                    for case in 0..7 {
                                        let mut input = vec![128u8; 256];
                                        for lane in 0..length {
                                            let near = (lane % 4) as i32 - 2;
                                            let sign = if lane % 2 == 0 { -1 } else { 1 };
                                            let mut p = [128, 127, 128];
                                            let mut r = [129, 130, 129];
                                            match case {
                                                0 => {
                                                    p = [0; 3];
                                                    r = [(alpha + near).clamp(0, 255); 3];
                                                }
                                                1 => p[1] = p[0] + sign * (beta + near),
                                                2 => r[1] = r[0] + sign * (beta + near),
                                                3 => p[2] = p[0] + sign * (beta + near),
                                                4 => r[2] = r[0] + sign * (beta + near),
                                                5 => {
                                                    p = [0, 0, 0];
                                                    r = [0, 6, 0];
                                                }
                                                _ => {
                                                    p = [255, 255, 255];
                                                    r = [255, 249, 255];
                                                }
                                            }
                                            let q = 4 * 16 + 4 + lane * along;
                                            for tap in 0..3 {
                                                input[q - (tap + 1) * across] =
                                                    p[tap].clamp(0, 255) as u8;
                                                input[q + tap * across] =
                                                    r[tap].clamp(0, 255) as u8;
                                            }
                                        }
                                        let mut scalar = input.clone();
                                        let mut vector = input;
                                        filter_edge_dispatch::<0>(
                                            &mut scalar,
                                            16,
                                            4,
                                            4,
                                            vertical,
                                            length,
                                            strength,
                                            qp,
                                            alpha_offset,
                                            beta_offset,
                                            chroma,
                                        );
                                        filter_edge_dispatch::<2>(
                                            &mut vector,
                                            16,
                                            4,
                                            4,
                                            vertical,
                                            length,
                                            strength,
                                            qp,
                                            alpha_offset,
                                            beta_offset,
                                            chroma,
                                        );
                                        assert_eq!(
                                            scalar, vector,
                                            "QP={qp} offsets={alpha_offset}/{beta_offset} bS={strength} vertical={vertical} chroma={chroma} length={length} case={case}"
                                        );
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
    #[ignore = "explicit same-process scalar versus vector deblock benchmark"]
    fn benchmark_deblock_scalar_vs_vector() {
        let input: Vec<u8> = (0..256 * 256)
            .map(|i| (96 + ((i % 256) / 8 + (i / 256) / 8) % 48) as u8)
            .collect();
        for vertical in [false, true] {
            let mut expected = None;
            for vector in [false, true, true, false] {
                let mut plane = input.clone();
                let timer = std::time::Instant::now();
                for round in 0..200 {
                    for y in (8..248).step_by(8) {
                        for x in (8..248).step_by(8) {
                            let strength = 1 + (round % 3) as u8;
                            if vector {
                                filter_edge_dispatch::<2>(
                                    &mut plane, 256, x, y, vertical, 4, strength, 32, 0, 0, false,
                                );
                            } else {
                                filter_edge_dispatch::<0>(
                                    &mut plane, 256, x, y, vertical, 4, strength, 32, 0, 0, false,
                                );
                            }
                        }
                    }
                }
                eprintln!(
                    "720000 pixels vertical={vertical} vector={vector}: {:?}",
                    timer.elapsed()
                );
                if let Some(expected) = &expected {
                    assert_eq!(&plane, expected);
                } else {
                    expected = Some(plane);
                }
            }
        }
    }

    #[test]
    fn specialized_edges_match_runtime_edges() {
        for seed in 0..8 {
            let input: Vec<u8> = (0..256)
                .map(|i| ((i * (seed * 2 + 1) + seed * 31) & 255) as u8)
                .collect();
            for qp in 0..52 {
                for strength in 0..=4 {
                    for vertical in [false, true] {
                        for chroma in [false, true] {
                            let mut runtime = input.clone();
                            let mut specialized = input.clone();
                            filter_edge_impl::<false, false, false, 0>(
                                &mut runtime,
                                16,
                                8,
                                8,
                                4,
                                strength,
                                qp,
                                0,
                                0,
                                vertical,
                                chroma,
                            );
                            filter_edge(
                                &mut specialized,
                                16,
                                8,
                                8,
                                vertical,
                                4,
                                strength,
                                qp,
                                0,
                                0,
                                chroma,
                            );
                            assert_eq!(
                                runtime, specialized,
                                "seed={seed} QP={qp} bS={strength} vertical={vertical} chroma={chroma}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "explicit same-process deblock specialization benchmark"]
    fn benchmark_deblock_edge_specialization() {
        let input: Vec<u8> = (0..256 * 256)
            .map(|i| (96 + ((i % 256) / 8 + (i / 256) / 8) % 48) as u8)
            .collect();
        let mut expected = None;
        for specialized in [false, true, true, false] {
            let mut plane = input.clone();
            let timer = std::time::Instant::now();
            for round in 0..100 {
                for y in (8..248).step_by(8) {
                    for x in (8..248).step_by(8) {
                        for vertical in [false, true] {
                            let vertical = std::hint::black_box(vertical);
                            let chroma = std::hint::black_box(round & 1 != 0);
                            let strength = 1 + (round % 4) as u8;
                            if specialized {
                                filter_edge(
                                    &mut plane, 256, x, y, vertical, 4, strength, 32, 0, 0, chroma,
                                );
                            } else {
                                filter_edge_impl::<false, false, false, 0>(
                                    &mut plane, 256, x, y, 4, strength, 32, 0, 0, vertical, chroma,
                                );
                            }
                        }
                    }
                }
            }
            eprintln!(
                "720000 deblocked pixels specialized={specialized}: {:?}",
                timer.elapsed()
            );
            if let Some(expected) = &expected {
                assert_eq!(&plane, expected);
            } else {
                expected = Some(plane);
            }
        }
    }

    #[test]
    fn nonzero_strength_is_local_to_the_adjoining_four_by_four_blocks() {
        let mb = DeblockMb {
            qp: 26,
            intra: false,
            transform8x8: false,
            coded_luma: 1 << 1,
            motion: [MotionCell::default(); 16],
        };
        assert_eq!(boundary_strength(&mb, &mb, 1, 0, false), 2);
        assert_eq!(boundary_strength(&mb, &mb, 5, 4, false), 0);
        assert_eq!(boundary_strength(&mb, &mb, 4, 0, false), 0);
        assert_eq!(boundary_strength(&mb, &mb, 5, 1, false), 2);
    }

    #[test]
    fn strength_compares_reference_pictures_without_list_order() {
        let a = MotionCell {
            l0: Some((0, [8, 4])),
            l1: Some((1, [-4, 12])),
        };
        let swapped = MotionCell { l0: a.l1, l1: a.l0 };
        assert!(!motion_differs(a, swapped));
        let one = MotionCell { l0: a.l0, l1: None };
        assert!(!motion_differs(one, MotionCell { l0: None, l1: a.l0 }));
        assert!(motion_differs(
            one,
            MotionCell {
                l0: Some((1, [8, 4])),
                l1: None
            }
        ));
        assert!(motion_differs(
            one,
            MotionCell {
                l0: Some((0, [12, 4])),
                l1: None
            }
        ));
        assert!(motion_differs(a, one));
        let same_pic = MotionCell {
            l0: Some((0, [8, 4])),
            l1: Some((0, [-4, 12])),
        };
        assert!(!motion_differs(
            same_pic,
            MotionCell {
                l0: same_pic.l1,
                l1: same_pic.l0
            }
        ));
    }

    #[test]
    fn motion_comparison_matches_option_pair_reference() {
        let reference_differs = |a: MotionCell, b: MotionCell| {
            let matches = |a: (u8, [i32; 2]), b: (u8, [i32; 2])| {
                a.0 == b.0 && (a.1[0] - b.1[0]).abs() < 4 && (a.1[1] - b.1[1]).abs() < 4
            };
            match ((a.l0, a.l1), (b.l0, b.l1)) {
                ((None, None), (None, None)) => false,
                ((Some(a), None) | (None, Some(a)), (Some(b), None) | (None, Some(b))) => !matches(a, b),
                ((Some(a0), Some(a1)), (Some(b0), Some(b1))) => {
                    !((matches(a0, b0) && matches(a1, b1)) || (matches(a0, b1) && matches(a1, b0)))
                }
                _ => true,
            }
        };
        let mut cells = vec![MotionCell::default()];
        for reference in 0..3 {
            for x in [-8, -4, -3, 0, 3, 4, 8] {
                for y in [-4, 0, 4] {
                    let a = Some((reference, [x, y]));
                    cells.push(MotionCell { l0: a, l1: None });
                    cells.push(MotionCell { l0: None, l1: a });
                    cells.push(MotionCell { l0: a, l1: a });
                    cells.push(MotionCell { l0: a, l1: Some(((reference + 1) % 3, [-x, -y])) });
                }
            }
        }
        for &a in &cells {
            for &b in &cells {
                assert_eq!(motion_differs(a, b), reference_differs(a, b), "{a:?} vs {b:?}");
            }
        }
    }
}
