//! H.264 (2003/2005) intra 4x4 and 8x8 luma prediction and block construction.

use super::h264::AvcError;
use super::h264_transform::{
    inverse_4x4_frame_scan, inverse_4x4_residual, inverse_8x8_frame_scan, inverse_8x8_residual,
    inverse_chroma_dc, inverse_luma16x16_dc,
};

pub fn predict_intra16x16(
    mode: u8,
    top: Option<[u8; 16]>,
    left: Option<[u8; 16]>,
    top_left: Option<u8>,
) -> Result<[u8; 256], AvcError> {
    let dc = match (top, left) {
        (Some(top), Some(left)) => {
            (top.iter()
                .chain(left.iter())
                .map(|&v| u32::from(v))
                .sum::<u32>()
                + 16)
                >> 5
        }
        (Some(edge), None) | (None, Some(edge)) => {
            (edge.iter().map(|&v| u32::from(v)).sum::<u32>() + 8) >> 4
        }
        (None, None) => 128,
    } as u8;
    let plane = if mode == 3 {
        let top = top.ok_or(AvcError::InvalidData("unavailable Intra16x16 top"))?;
        let left = left.ok_or(AvcError::InvalidData("unavailable Intra16x16 left"))?;
        let corner = top_left.ok_or(AvcError::InvalidData("unavailable Intra16x16 corner"))?;
        let mut h = 0i32;
        let mut v = 0i32;
        for index in 0..8 {
            let previous_top = if index == 7 { corner } else { top[6 - index] };
            let previous_left = if index == 7 { corner } else { left[6 - index] };
            h += (index as i32 + 1) * (i32::from(top[8 + index]) - i32::from(previous_top));
            v += (index as i32 + 1) * (i32::from(left[8 + index]) - i32::from(previous_left));
        }
        Some((
            16 * (i32::from(top[15]) + i32::from(left[15])),
            (5 * h + 32) >> 6,
            (5 * v + 32) >> 6,
        ))
    } else {
        None
    };
    let mut predicted = [0; 256];
    for y in 0..16 {
        for x in 0..16 {
            predicted[y * 16 + x] = match mode {
                0 => top.ok_or(AvcError::InvalidData("unavailable Intra16x16 top"))?[x],
                1 => left.ok_or(AvcError::InvalidData("unavailable Intra16x16 left"))?[y],
                2 => dc,
                3 => {
                    let (a, b, c) = plane.unwrap();
                    ((a + b * (x as i32 - 7) + c * (y as i32 - 7) + 16) >> 5).clamp(0, 255) as u8
                }
                _ => return Err(AvcError::InvalidData("Intra16x16 mode out of range")),
            };
        }
    }
    Ok(predicted)
}

pub fn reconstruct_intra16x16_luma(
    mode: u8,
    dc_levels: &[i32; 16],
    ac_levels: &[[i32; 15]; 16],
    qp: i32,
    top: Option<[u8; 16]>,
    left: Option<[u8; 16]>,
    top_left: Option<u8>,
) -> Result<[u8; 256], AvcError> {
    let mut pixels = predict_intra16x16(mode, top, left, top_left)?;
    let dc = inverse_luma16x16_dc(dc_levels, qp)?;
    for block in 0..16 {
        let region = block / 4;
        let sub = block % 4;
        let x0 = (region % 2) * 8 + (sub % 2) * 4;
        let y0 = (region / 2) * 8 + (sub / 2) * 4;
        let mut scanned = [0; 16];
        scanned[1..].copy_from_slice(&ac_levels[block]);
        let mut coefficients = inverse_4x4_frame_scan(&scanned);
        coefficients[0] = dc[block];
        let residual = inverse_4x4_residual(&coefficients, qp, true)?;
        for y in 0..4 {
            for x in 0..4 {
                let index = (y0 + y) * 16 + x0 + x;
                pixels[index] =
                    (i32::from(pixels[index]) + residual[y * 4 + x]).clamp(0, 255) as u8;
            }
        }
    }
    Ok(pixels)
}

/// H.264 (03/2005) 8.3.2.2 reference filtering and Intra8x8 prediction.
pub fn predict_intra8x8(
    mode: u8,
    top: Option<[u8; 16]>,
    left: Option<[u8; 8]>,
    top_left: Option<u8>,
) -> Result<[u8; 64], AvcError> {
    let average = |a: u8, b: u8, c: u8| -> u8 {
        ((u32::from(a) + 2 * u32::from(b) + u32::from(c) + 2) >> 2) as u8
    };
    let filtered_top = top.map(|edge| {
        let mut filtered = [0; 16];
        filtered[0] = average(top_left.unwrap_or(edge[0]), edge[0], edge[1]);
        for x in 1..15 {
            filtered[x] = average(edge[x - 1], edge[x], edge[x + 1]);
        }
        filtered[15] = average(edge[14], edge[15], edge[15]);
        filtered
    });
    let filtered_left = left.map(|edge| {
        let mut filtered = [0; 8];
        filtered[0] = average(top_left.unwrap_or(edge[0]), edge[0], edge[1]);
        for y in 1..7 {
            filtered[y] = average(edge[y - 1], edge[y], edge[y + 1]);
        }
        filtered[7] = average(edge[6], edge[7], edge[7]);
        filtered
    });
    let filtered_corner = top_left.map(|corner| match (top, left) {
        (Some(top), Some(left)) => average(top[0], corner, left[0]),
        (Some(top), None) => average(corner, corner, top[0]),
        (None, Some(left)) => average(corner, corner, left[0]),
        (None, None) => corner,
    });
    let sample = |x: isize, y: isize| -> Result<u8, AvcError> {
        let value = if x == -1 && y == -1 {
            filtered_corner
        } else if y == -1 && (0..16).contains(&x) {
            filtered_top.map(|edge| edge[x as usize])
        } else if x == -1 && (0..8).contains(&y) {
            filtered_left.map(|edge| edge[y as usize])
        } else {
            None
        };
        value.ok_or(AvcError::InvalidData("unavailable intra 8x8 neighbour"))
    };
    let half = |a: u8, b: u8| -> u8 { ((u32::from(a) + u32::from(b) + 1) >> 1) as u8 };
    let dc = match (filtered_top, filtered_left) {
        (Some(top), Some(left)) => {
            (top[..8]
                .iter()
                .chain(left.iter())
                .map(|&v| u32::from(v))
                .sum::<u32>()
                + 8)
                >> 4
        }
        (Some(top), None) => (top[..8].iter().map(|&v| u32::from(v)).sum::<u32>() + 4) >> 3,
        (None, Some(left)) => (left.iter().map(|&v| u32::from(v)).sum::<u32>() + 4) >> 3,
        (None, None) => 128,
    } as u8;
    let mut predicted = [0; 64];
    for y in 0..8 {
        for x in 0..8 {
            predicted[y * 8 + x] = match mode {
                0 => sample(x as isize, -1)?,
                1 => sample(-1, y as isize)?,
                2 => dc,
                3 if x == 7 && y == 7 => {
                    ((u32::from(sample(14, -1)?) + 3 * u32::from(sample(15, -1)?) + 2) >> 2) as u8
                }
                3 => average(
                    sample((x + y) as isize, -1)?,
                    sample((x + y + 1) as isize, -1)?,
                    sample((x + y + 2) as isize, -1)?,
                ),
                4..=7 => {
                    let x = x as isize;
                    let y = y as isize;
                    let quarter = |a: (isize, isize),
                                   b: (isize, isize),
                                   c: (isize, isize)|
                     -> Result<u8, AvcError> {
                        Ok(average(
                            sample(a.0, a.1)?,
                            sample(b.0, b.1)?,
                            sample(c.0, c.1)?,
                        ))
                    };
                    match mode {
                        4 if x > y => quarter((x - y - 2, -1), (x - y - 1, -1), (x - y, -1))?,
                        4 if x < y => quarter((-1, y - x - 2), (-1, y - x - 1), (-1, y - x))?,
                        4 => quarter((0, -1), (-1, -1), (-1, 0))?,
                        5 if 2 * x - y >= 0 && (2 * x - y) % 2 == 0 => {
                            half(sample(x - y / 2 - 1, -1)?, sample(x - y / 2, -1)?)
                        }
                        5 if 2 * x - y > 0 => {
                            quarter((x - y / 2 - 2, -1), (x - y / 2 - 1, -1), (x - y / 2, -1))?
                        }
                        5 if 2 * x - y == -1 => quarter((-1, 0), (-1, -1), (0, -1))?,
                        5 => quarter(
                            (-1, y - 2 * x - 1),
                            (-1, y - 2 * x - 2),
                            (-1, y - 2 * x - 3),
                        )?,
                        6 if 2 * y - x >= 0 && (2 * y - x) % 2 == 0 => {
                            half(sample(-1, y - x / 2 - 1)?, sample(-1, y - x / 2)?)
                        }
                        6 if 2 * y - x > 0 => {
                            quarter((-1, y - x / 2 - 2), (-1, y - x / 2 - 1), (-1, y - x / 2))?
                        }
                        6 if 2 * y - x == -1 => quarter((-1, 0), (-1, -1), (0, -1))?,
                        6 => quarter(
                            (x - 2 * y - 1, -1),
                            (x - 2 * y - 2, -1),
                            (x - 2 * y - 3, -1),
                        )?,
                        7 if y % 2 == 0 => half(sample(x + y / 2, -1)?, sample(x + y / 2 + 1, -1)?),
                        7 => quarter((x + y / 2, -1), (x + y / 2 + 1, -1), (x + y / 2 + 2, -1))?,
                        _ => unreachable!(),
                    }
                }
                8 => {
                    let edge =
                        filtered_left.ok_or(AvcError::InvalidData("unavailable intra 8x8 left"))?;
                    let z = x + 2 * y;
                    if z <= 12 && z % 2 == 0 {
                        ((u32::from(edge[y + x / 2]) + u32::from(edge[y + x / 2 + 1]) + 1) >> 1)
                            as u8
                    } else if z <= 11 {
                        average(edge[y + x / 2], edge[y + x / 2 + 1], edge[y + x / 2 + 2])
                    } else if z == 13 {
                        ((u32::from(edge[6]) + 3 * u32::from(edge[7]) + 2) >> 2) as u8
                    } else {
                        edge[7]
                    }
                }
                _ => return Err(AvcError::Unsupported("intra 8x8 prediction mode")),
            };
        }
    }
    Ok(predicted)
}

pub fn reconstruct_intra8x8_luma(
    modes: &[u8; 4],
    levels: &[[i32; 64]; 4],
    qp: i32,
    weights: &[u8; 64],
    above: Option<[u8; 24]>,
    left: Option<[u8; 16]>,
    top_left: Option<u8>,
) -> Result<[u8; 256], AvcError> {
    let mut pixels = [0; 256];
    let mut constructed = [false; 256];
    for block in 0..4 {
        let x0 = (block % 2) * 8;
        let y0 = (block / 2) * 8;
        let top_sample = |x: usize| -> Option<u8> {
            if y0 == 0 {
                above.map(|edge| edge[x])
            } else if x < 16 && constructed[(y0 - 1) * 16 + x] {
                Some(pixels[(y0 - 1) * 16 + x])
            } else {
                None
            }
        };
        let mut top_edge = [0; 16];
        let mut top_available = true;
        for x in 0..8 {
            if let Some(value) = top_sample(x0 + x) {
                top_edge[x] = value;
            } else {
                top_available = false;
            }
        }
        let top_edge = top_available.then(|| {
            for x in 8..16 {
                top_edge[x] = top_sample(x0 + x).unwrap_or(top_edge[7]);
            }
            top_edge
        });
        let mut left_edge = [0; 8];
        let mut left_available = true;
        for y in 0..8 {
            let sample = if x0 == 0 {
                left.map(|edge| edge[y0 + y])
            } else if constructed[(y0 + y) * 16 + x0 - 1] {
                Some(pixels[(y0 + y) * 16 + x0 - 1])
            } else {
                None
            };
            if let Some(value) = sample {
                left_edge[y] = value;
            } else {
                left_available = false;
            }
        }
        let corner = if x0 == 0 && y0 == 0 {
            top_left
        } else if y0 == 0 {
            above.map(|edge| edge[x0 - 1])
        } else if x0 == 0 {
            left.map(|edge| edge[y0 - 1])
        } else if constructed[(y0 - 1) * 16 + x0 - 1] {
            Some(pixels[(y0 - 1) * 16 + x0 - 1])
        } else {
            None
        };
        let prediction = predict_intra8x8(
            modes[block],
            top_edge,
            left_available.then_some(left_edge),
            corner,
        )?;
        let coefficients = inverse_8x8_frame_scan(&levels[block]);
        let residual = inverse_8x8_residual(&coefficients, qp, weights)?;
        for y in 0..8 {
            for x in 0..8 {
                let index = (y0 + y) * 16 + x0 + x;
                pixels[index] =
                    (i32::from(prediction[y * 8 + x]) + residual[y * 8 + x]).clamp(0, 255) as u8;
                constructed[index] = true;
            }
        }
    }
    Ok(pixels)
}

pub fn predict_intra4x4(
    mode: u8,
    top: Option<[u8; 8]>,
    left: Option<[u8; 4]>,
    top_left: Option<u8>,
) -> Result<[u8; 16], AvcError> {
    let sample = |x: isize, y: isize| -> Result<i32, AvcError> {
        let value = if y == -1 && x == -1 {
            top_left
        } else if y == -1 && (0..8).contains(&x) {
            top.map(|edge| edge[x as usize])
        } else if x == -1 && (0..4).contains(&y) {
            left.map(|edge| edge[y as usize])
        } else {
            None
        };
        value
            .map(i32::from)
            .ok_or(AvcError::InvalidData("unavailable intra 4x4 neighbour"))
    };
    let dc = match (top, left) {
        (Some(top), Some(left)) => {
            (top[..4]
                .iter()
                .chain(left.iter())
                .map(|&v| u32::from(v))
                .sum::<u32>()
                + 4)
                >> 3
        }
        (Some(top), None) => (top[..4].iter().map(|&v| u32::from(v)).sum::<u32>() + 2) >> 2,
        (None, Some(left)) => (left.iter().map(|&v| u32::from(v)).sum::<u32>() + 2) >> 2,
        (None, None) => 128,
    } as i32;
    let mut predicted = [0u8; 16];
    for y in 0..4isize {
        for x in 0..4isize {
            let zvr = 2 * x - y;
            let zhd = 2 * y - x;
            let zhu = x + 2 * y;
            let value = match mode {
                0 => sample(x, -1)?,
                1 => sample(-1, y)?,
                2 => dc,
                3 if x == 3 && y == 3 => (sample(6, -1)? + 3 * sample(7, -1)? + 2) >> 2,
                3 => {
                    (sample(x + y, -1)? + 2 * sample(x + y + 1, -1)? + sample(x + y + 2, -1)? + 2)
                        >> 2
                }
                4 if x > y => {
                    (sample(x - y - 2, -1)? + 2 * sample(x - y - 1, -1)? + sample(x - y, -1)? + 2)
                        >> 2
                }
                4 if x < y => {
                    (sample(-1, y - x - 2)? + 2 * sample(-1, y - x - 1)? + sample(-1, y - x)? + 2)
                        >> 2
                }
                4 => (sample(0, -1)? + 2 * sample(-1, -1)? + sample(-1, 0)? + 2) >> 2,
                5 if zvr >= 0 && zvr % 2 == 0 => {
                    (sample(x - (y >> 1) - 1, -1)? + sample(x - (y >> 1), -1)? + 1) >> 1
                }
                5 if zvr > 0 => {
                    (sample(x - (y >> 1) - 2, -1)?
                        + 2 * sample(x - (y >> 1) - 1, -1)?
                        + sample(x - (y >> 1), -1)?
                        + 2)
                        >> 2
                }
                5 if zvr == -1 => (sample(-1, 0)? + 2 * sample(-1, -1)? + sample(0, -1)? + 2) >> 2,
                5 => (sample(-1, y - 1)? + 2 * sample(-1, y - 2)? + sample(-1, y - 3)? + 2) >> 2,
                6 if zhd >= 0 && zhd % 2 == 0 => {
                    (sample(-1, y - (x >> 1) - 1)? + sample(-1, y - (x >> 1))? + 1) >> 1
                }
                6 if zhd > 0 => {
                    (sample(-1, y - (x >> 1) - 2)?
                        + 2 * sample(-1, y - (x >> 1) - 1)?
                        + sample(-1, y - (x >> 1))?
                        + 2)
                        >> 2
                }
                6 if zhd == -1 => (sample(-1, 0)? + 2 * sample(-1, -1)? + sample(0, -1)? + 2) >> 2,
                6 => (sample(x - 1, -1)? + 2 * sample(x - 2, -1)? + sample(x - 3, -1)? + 2) >> 2,
                7 if y % 2 == 0 => {
                    (sample(x + (y >> 1), -1)? + sample(x + (y >> 1) + 1, -1)? + 1) >> 1
                }
                7 => {
                    (sample(x + (y >> 1), -1)?
                        + 2 * sample(x + (y >> 1) + 1, -1)?
                        + sample(x + (y >> 1) + 2, -1)?
                        + 2)
                        >> 2
                }
                8 if zhu == 0 || zhu == 2 || zhu == 4 => {
                    (sample(-1, y + (x >> 1))? + sample(-1, y + (x >> 1) + 1)? + 1) >> 1
                }
                8 if zhu == 1 || zhu == 3 => {
                    (sample(-1, y + (x >> 1))?
                        + 2 * sample(-1, y + (x >> 1) + 1)?
                        + sample(-1, y + (x >> 1) + 2)?
                        + 2)
                        >> 2
                }
                8 if zhu == 5 => (sample(-1, 2)? + 3 * sample(-1, 3)? + 2) >> 2,
                8 => sample(-1, 3)?,
                _ => return Err(AvcError::InvalidData("intra 4x4 mode out of range")),
            };
            predicted[y as usize * 4 + x as usize] = value as u8;
        }
    }
    Ok(predicted)
}

/// Reconstruct one luma macroblock in the normative 4x4 decoding order.
pub fn reconstruct_intra4x4_luma(
    modes: &[u8; 16],
    levels: &[[i32; 16]; 16],
    qp: i32,
    above: Option<[u8; 20]>,
    left: Option<[u8; 16]>,
    top_left: Option<u8>,
) -> Result<[u8; 256], AvcError> {
    let mut pixels = [0u8; 256];
    let mut constructed = [false; 256];
    for block in 0..16 {
        let region = block / 4;
        let sub = block % 4;
        let x0 = (region % 2) * 8 + (sub % 2) * 4;
        let y0 = (region / 2) * 8 + (sub / 2) * 4;
        let top_sample = |x: usize| -> Option<u8> {
            if y0 == 0 {
                above.map(|edge| edge[x])
            } else if x < 16 && constructed[(y0 - 1) * 16 + x] {
                Some(pixels[(y0 - 1) * 16 + x])
            } else {
                None
            }
        };
        let mut top_values = [0u8; 8];
        let mut top_available = true;
        for index in 0..4 {
            if let Some(value) = top_sample(x0 + index) {
                top_values[index] = value;
            } else {
                top_available = false;
            }
        }
        let top = top_available.then(|| {
            for index in 4..8 {
                top_values[index] = top_sample(x0 + index).unwrap_or(top_values[3]);
            }
            top_values
        });
        let mut side_values = [0u8; 4];
        let mut side_available = true;
        for index in 0..4 {
            let sample = if x0 == 0 {
                left.map(|edge| edge[y0 + index])
            } else if constructed[(y0 + index) * 16 + x0 - 1] {
                Some(pixels[(y0 + index) * 16 + x0 - 1])
            } else {
                None
            };
            if let Some(value) = sample {
                side_values[index] = value;
            } else {
                side_available = false;
            }
        }
        let left_values = side_available.then_some(side_values);
        let corner = if x0 == 0 && y0 == 0 {
            top_left
        } else if y0 == 0 {
            above.map(|edge| edge[x0 - 1])
        } else if x0 == 0 {
            left.map(|edge| edge[y0 - 1])
        } else if constructed[(y0 - 1) * 16 + x0 - 1] {
            Some(pixels[(y0 - 1) * 16 + x0 - 1])
        } else {
            None
        };
        let prediction = predict_intra4x4(modes[block], top, left_values, corner)?;
        let coefficients = inverse_4x4_frame_scan(&levels[block]);
        let residual = inverse_4x4_residual(&coefficients, qp, false)?;
        for y in 0..4 {
            for x in 0..4 {
                let index = (y0 + y) * 16 + x0 + x;
                pixels[index] =
                    (i32::from(prediction[y * 4 + x]) + residual[y * 4 + x]).clamp(0, 255) as u8;
                constructed[index] = true;
            }
        }
    }
    Ok(pixels)
}

/// Reconstruct 4:2:0 chroma for an intra DC prediction macroblock with no AC.
pub fn reconstruct_chroma_dc_only(
    dc_levels: &[i32; 4],
    qp: i32,
    above: Option<[u8; 8]>,
    left: Option<[u8; 8]>,
) -> Result<[u8; 64], AvcError> {
    reconstruct_chroma(0, dc_levels, &[[0; 15]; 4], qp, above, left, None)
}

pub fn reconstruct_chroma(
    mode: u8,
    dc_levels: &[i32; 4],
    ac_levels: &[[i32; 15]; 4],
    qp: i32,
    above: Option<[u8; 8]>,
    left: Option<[u8; 8]>,
    top_left: Option<u8>,
) -> Result<[u8; 64], AvcError> {
    let dc = inverse_chroma_dc(dc_levels, qp)?;
    let plane = if mode == 3 {
        let top = above.ok_or(AvcError::InvalidData("unavailable chroma top"))?;
        let side = left.ok_or(AvcError::InvalidData("unavailable chroma left"))?;
        let corner = top_left.ok_or(AvcError::InvalidData("unavailable chroma corner"))?;
        let mut h = 0i32;
        let mut v = 0i32;
        for index in 0..4 {
            let previous_top = if index == 3 { corner } else { top[2 - index] };
            let previous_left = if index == 3 { corner } else { side[2 - index] };
            h += (index as i32 + 1) * (i32::from(top[4 + index]) - i32::from(previous_top));
            v += (index as i32 + 1) * (i32::from(side[4 + index]) - i32::from(previous_left));
        }
        Some((
            16 * (i32::from(top[7]) + i32::from(side[7])),
            (34 * h + 32) >> 6,
            (34 * v + 32) >> 6,
        ))
    } else {
        None
    };
    let mut pixels = [0u8; 64];
    for quadrant in 0..4 {
        let qx = quadrant % 2;
        let qy = quadrant / 2;
        let top = above.map(|edge| {
            edge[qx * 4..qx * 4 + 4]
                .iter()
                .map(|&value| u32::from(value))
                .sum::<u32>()
        });
        let side = left.map(|edge| {
            edge[qy * 4..qy * 4 + 4]
                .iter()
                .map(|&value| u32::from(value))
                .sum::<u32>()
        });
        let (top, side) = match (qx, qy, top.is_some(), side.is_some()) {
            (1, 0, true, true) => (top, None),
            (0, 1, true, true) => (None, side),
            _ => (top, side),
        };
        let prediction = match (top, side) {
            (Some(top), Some(side)) => (top + side + 4) >> 3,
            (Some(sum), None) | (None, Some(sum)) => (sum + 2) >> 2,
            (None, None) => 128,
        } as i32;
        let mut scanned = [0; 16];
        scanned[1..].copy_from_slice(&ac_levels[quadrant]);
        let mut coefficients = inverse_4x4_frame_scan(&scanned);
        coefficients[0] = dc[quadrant];
        let residual = inverse_4x4_residual(&coefficients, qp, true)?;
        for y in 0..4 {
            for x in 0..4 {
                let px = qx * 4 + x;
                let py = qy * 4 + y;
                let predicted = match mode {
                    0 => prediction,
                    1 => {
                        i32::from(left.ok_or(AvcError::InvalidData("unavailable chroma left"))?[py])
                    }
                    2 => {
                        i32::from(above.ok_or(AvcError::InvalidData("unavailable chroma top"))?[px])
                    }
                    3 => {
                        let (a, b, c) = plane.unwrap();
                        ((a + b * (px as i32 - 3) + c * (py as i32 - 3) + 16) >> 5).clamp(0, 255)
                    }
                    _ => return Err(AvcError::InvalidData("chroma intra mode out of range")),
                };
                pixels[(qy * 4 + y) * 8 + qx * 4 + x] =
                    (predicted + residual[y * 4 + x]).clamp(0, 255) as u8;
            }
        }
    }
    Ok(pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_2003_intra4x4_modes_predict_from_edges() {
        let top = [10, 20, 30, 40, 50, 60, 70, 80];
        let left = [90, 100, 110, 120];
        for mode in 0..=8 {
            let prediction = predict_intra4x4(mode, Some(top), Some(left), Some(50)).unwrap();
            assert!(prediction.iter().any(|&value| value != 0));
        }
        assert_eq!(
            predict_intra4x4(0, Some(top), None, None).unwrap()[0..4],
            top[0..4]
        );
        assert_eq!(
            predict_intra4x4(1, None, Some(left), None).unwrap()[0..4],
            [90; 4]
        );
        assert_eq!(predict_intra4x4(2, None, None, None).unwrap(), [128; 16]);
        assert!(predict_intra4x4(3, None, None, None).is_err());
    }

    #[test]
    fn zero_residual_dc_macroblock_reconstructs_to_128() {
        let pixels =
            reconstruct_intra4x4_luma(&[2; 16], &[[0; 16]; 16], 26, None, None, None).unwrap();
        assert_eq!(pixels, [128; 256]);
    }

    #[test]
    fn intra8x8_prediction_filters_reference_edges() {
        let left = [10, 20, 30, 40, 50, 60, 70, 80];
        let horizontal = predict_intra8x8(1, None, Some(left), None).unwrap();
        assert_eq!(&horizontal[..8], &[13; 8]);
        let dc = predict_intra8x8(2, None, Some(left), None).unwrap();
        assert_eq!(dc, [45; 64]);
        let up = predict_intra8x8(8, None, Some(left), None).unwrap();
        assert_eq!(up[0], 17);
        assert_eq!(up[63], 78);
        assert!(predict_intra8x8(8, None, None, None).is_err());
        assert!(predict_intra8x8(3, None, Some(left), None).is_err());
    }

    #[test]
    fn all_2005_intra8x8_modes_predict_from_filtered_edges() {
        let top = std::array::from_fn(|x| 20 + x as u8 * 5);
        let left = std::array::from_fn(|y| 40 + y as u8 * 7);
        for mode in 0..=8 {
            let pixels = predict_intra8x8(mode, Some(top), Some(left), Some(30)).unwrap();
            assert_eq!(pixels.len(), 64);
            assert!(pixels.iter().all(|&pixel| pixel > 0));
        }
        assert!(predict_intra8x8(4, Some(top), Some(left), None).is_err());
        assert!(predict_intra8x8(9, Some(top), Some(left), Some(30)).is_err());
    }

    #[test]
    fn intra8x8_macroblock_uses_constructed_neighbours() {
        let pixels = reconstruct_intra8x8_luma(
            &[2, 2, 2, 8],
            &[[0; 64]; 4],
            26,
            &[16; 64],
            None,
            Some([64; 16]),
            None,
        )
        .unwrap();
        assert_eq!(pixels[0], 64);
        assert_eq!(pixels[15], 64);
        assert_eq!(pixels[15 * 16 + 15], 64);
    }

    #[test]
    fn intra16x16_horizontal_and_dc_residual_reconstruct() {
        let mut levels = [0; 16];
        levels[0] = 512;
        let pixels =
            reconstruct_intra16x16_luma(1, &levels, &[[0; 15]; 16], 0, None, Some([64; 16]), None)
                .unwrap();
        assert_eq!(pixels, [84; 256]);
        assert_eq!(predict_intra16x16(2, None, None, None).unwrap(), [128; 256]);
        assert!(predict_intra16x16(3, Some([64; 16]), Some([64; 16]), None).is_err());
    }

    #[test]
    fn intra16x16_ac_varies_pixels_within_one_block() {
        let mut ac = [[0; 15]; 16];
        ac[0][0] = 100;
        let pixels = reconstruct_intra16x16_luma(2, &[0; 16], &ac, 26, None, None, None).unwrap();
        assert_ne!(pixels[0], pixels[3]);
        assert_eq!(pixels[15], 128);
    }

    #[test]
    fn chroma_dc_only_macroblock_uses_four_predictions() {
        let pixels = reconstruct_chroma_dc_only(&[0; 4], 26, None, None).unwrap();
        assert_eq!(pixels, [128; 64]);
        let top = [10; 8];
        let side = [30; 8];
        let pixels = reconstruct_chroma_dc_only(&[0; 4], 26, Some(top), Some(side)).unwrap();
        assert_eq!(pixels[0], 20);
        assert_eq!(pixels[4], 10);
        assert_eq!(pixels[32], 30);
        assert_eq!(pixels[36], 20);
    }

    #[test]
    fn chroma_dc_missing_edge_uses_the_available_edge_in_every_quadrant() {
        let top = [20, 20, 20, 20, 60, 60, 60, 60];
        let left = [40, 40, 40, 40, 80, 80, 80, 80];
        let top_only = reconstruct_chroma_dc_only(&[0; 4], 26, Some(top), None).unwrap();
        let left_only = reconstruct_chroma_dc_only(&[0; 4], 26, None, Some(left)).unwrap();
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(top_only[y * 8 + x], if x < 4 { 20 } else { 60 });
                assert_eq!(left_only[y * 8 + x], if y < 4 { 40 } else { 80 });
            }
        }
    }

    #[test]
    fn chroma_ac_adds_spatial_variation_to_dc_prediction() {
        let mut ac = [[0; 15]; 4];
        ac[0][0] = 100;
        let pixels = reconstruct_chroma(0, &[0; 4], &ac, 26, None, None, None).unwrap();
        assert_ne!(pixels[0], pixels[3]);
        assert_eq!(pixels[7], 128);
    }

    #[test]
    fn chroma_vertical_horizontal_and_plane_follow_edges() {
        let top = [10, 20, 30, 40, 50, 60, 70, 80];
        let side = [90, 100, 110, 120, 130, 140, 150, 160];
        let ac = [[0; 15]; 4];
        let vertical = reconstruct_chroma(2, &[0; 4], &ac, 26, Some(top), None, None).unwrap();
        assert_eq!(&vertical[..8], &top);
        assert_eq!(&vertical[56..], &top);
        let horizontal = reconstruct_chroma(1, &[0; 4], &ac, 26, None, Some(side), None).unwrap();
        assert_eq!(&horizontal[..8], &[90; 8]);
        assert_eq!(&horizontal[56..], &[160; 8]);
        let plane = reconstruct_chroma(3, &[0; 4], &ac, 26, Some([20; 8]), Some([20; 8]), Some(20))
            .unwrap();
        assert_eq!(plane, [20; 64]);
    }
}
