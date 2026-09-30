//! Inverse coefficient scanning and integer transforms from H.264 (2003/2005) 8.5.

use super::h264::AvcError;

// Table 8-12, frame macroblocks. Entries are row-major positions.
const FRAME_ZIGZAG: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

// H.264 (03/2005) Table 8-14, frame 8x8 zig-zag scan.
const FRAME_ZIGZAG_8X8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

// H.264 (03/2005) equation 8-317, v[QP % 6][position class].
const NORM_ADJUST_8X8: [[i32; 6]; 6] = [
    [20, 18, 32, 19, 25, 24],
    [22, 19, 35, 21, 28, 26],
    [26, 23, 42, 24, 33, 31],
    [28, 25, 45, 26, 35, 33],
    [32, 28, 51, 30, 40, 38],
    [36, 32, 58, 34, 46, 43],
];

// Equation 8-253: LevelScale(qP % 6) for even/even, odd/odd, and mixed positions.
const LEVEL_SCALE: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

// Table 8-13. QP indices below 30 map to themselves.
const CHROMA_QP_30_TO_51: [i32; 22] = [
    29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39,
];

pub fn chroma_qp(luma_qp: i32, offset: i32) -> Result<i32, AvcError> {
    if !(0..=51).contains(&luma_qp) || !(-12..=12).contains(&offset) {
        return Err(AvcError::InvalidData("chroma QP input out of range"));
    }
    let index = (luma_qp + offset).clamp(0, 51);
    Ok(if index < 30 {
        index
    } else {
        CHROMA_QP_30_TO_51[(index - 30) as usize]
    })
}

/// Equations 8-257..8-259, returning pre-scaled DC for four chroma blocks.
pub fn inverse_chroma_dc(levels: &[i32; 4], qp: i32) -> Result<[i32; 4], AvcError> {
    if !(0..=51).contains(&qp) {
        return Err(AvcError::InvalidData("chroma DC QP out of range"));
    }
    let [a, b, c, d] = levels.map(i64::from);
    let transformed = [a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d];
    let scale = i64::from(LEVEL_SCALE[(qp % 6) as usize][0]);
    let mut dc = [0; 4];
    for (output, coefficient) in dc.iter_mut().zip(transformed) {
        let value = if qp >= 6 {
            (coefficient * scale) << (qp / 6 - 1)
        } else {
            (coefficient * scale) >> 1
        };
        *output = i32::try_from(value)
            .ok()
            .filter(|value| (-32768..=32767).contains(value))
            .ok_or(AvcError::InvalidData("scaled chroma DC out of range"))?;
    }
    Ok(dc)
}

pub fn inverse_4x4_frame_scan(levels: &[i32; 16]) -> [i32; 16] {
    let mut coefficients = [0; 16];
    for (index, &level) in levels.iter().enumerate() {
        coefficients[FRAME_ZIGZAG[index]] = level;
    }
    coefficients
}

fn inverse_hadamard_4(values: [i64; 4]) -> [i64; 4] {
    let a0 = values[0] + values[2];
    let a1 = values[0] - values[2];
    let a2 = values[1] - values[3];
    let a3 = values[1] + values[3];
    [a0 + a3, a1 + a2, a1 - a2, a0 - a3]
}

/// H.264 8.5.8. Returns scaled DC values in luma 4x4 block-scan order.
pub fn inverse_luma16x16_dc(levels: &[i32; 16], qp: i32) -> Result<[i32; 16], AvcError> {
    if !(0..=51).contains(&qp) {
        return Err(AvcError::InvalidData("Intra16x16 DC QP out of range"));
    }
    let coefficients = inverse_4x4_frame_scan(levels);
    let mut horizontal = [0i64; 16];
    for row in 0..4 {
        horizontal[row * 4..row * 4 + 4].copy_from_slice(&inverse_hadamard_4(std::array::from_fn(
            |column| i64::from(coefficients[row * 4 + column]),
        )));
    }
    let mut scaled = [0i32; 16];
    let factor = i64::from(LEVEL_SCALE[(qp % 6) as usize][0]);
    for column in 0..4 {
        let transformed =
            inverse_hadamard_4(std::array::from_fn(|row| horizontal[row * 4 + column]));
        for row in 0..4 {
            let product = transformed[row] * factor;
            let value = if qp >= 36 {
                product << (qp / 6 - 6)
            } else {
                (product + (1_i64 << (5 - qp / 6))) >> (6 - qp / 6)
            };
            scaled[row * 4 + column] = i32::try_from(value)
                .ok()
                .filter(|value| (-32768..=32767).contains(value))
                .ok_or(AvcError::InvalidData("scaled Intra16x16 DC out of range"))?;
        }
    }
    Ok(std::array::from_fn(|block| scaled[FRAME_ZIGZAG[block]]))
}

pub fn inverse_8x8_frame_scan(levels: &[i32; 64]) -> [i32; 64] {
    let mut coefficients = [0; 64];
    for (index, &level) in levels.iter().enumerate() {
        coefficients[FRAME_ZIGZAG_8X8[index]] = level;
    }
    coefficients
}

fn inverse_8x8_1d(d: [i64; 8]) -> [i64; 8] {
    let e0 = d[0] + d[4];
    let e1 = -d[3] + d[5] - d[7] - (d[7] >> 1);
    let e2 = d[0] - d[4];
    let e3 = d[1] + d[7] - d[3] - (d[3] >> 1);
    let e4 = (d[2] >> 1) - d[6];
    let e5 = -d[1] + d[7] + d[5] + (d[5] >> 1);
    let e6 = d[2] + (d[6] >> 1);
    let e7 = d[3] + d[5] + d[1] + (d[1] >> 1);
    let f0 = e0 + e6;
    let f1 = e1 + (e7 >> 2);
    let f2 = e2 + e4;
    let f3 = e3 + (e5 >> 2);
    let f4 = e2 - e4;
    let f5 = (e3 >> 2) - e5;
    let f6 = e0 - e6;
    let f7 = e7 - (e1 >> 2);
    [
        f0 + f7,
        f2 + f5,
        f4 + f3,
        f6 + f1,
        f6 - f1,
        f4 - f3,
        f2 - f5,
        f0 - f7,
    ]
}

/// H.264 (03/2005) 8.5.11. `weights` is the inverse-scanned scaling list;
/// absent SPS/PPS scaling lists use the flat matrix `[16; 64]`.
pub fn inverse_8x8_residual(
    coefficients: &[i32; 64],
    qp: i32,
    weights: &[u8; 64],
) -> Result<[i32; 64], AvcError> {
    if !(0..=51).contains(&qp) {
        return Err(AvcError::InvalidData("8x8 residual QP out of range"));
    }
    let mut scaled = [0i64; 64];
    for (index, &coefficient) in coefficients.iter().enumerate() {
        let row = index / 8;
        let column = index % 8;
        let class = match (row % 4, column % 4) {
            (0, 0) => 0,
            (2, 2) => 2,
            (0, 2) | (2, 0) => 4,
            (0, 1 | 3) | (1 | 3, 0) => 3,
            (1 | 3, 1 | 3) => 1,
            _ => 5,
        };
        let factor =
            i64::from(weights[index]) * i64::from(NORM_ADJUST_8X8[(qp % 6) as usize][class]);
        let value = i64::from(coefficient) * factor;
        scaled[index] = if qp >= 36 {
            value << (qp / 6 - 6)
        } else {
            (value + (1_i64 << (5 - qp / 6))) >> (6 - qp / 6)
        };
        if !(-32768..=32767).contains(&scaled[index]) {
            return Err(AvcError::InvalidData("scaled 8x8 coefficient out of range"));
        }
    }
    let mut horizontal = [0i64; 64];
    for row in 0..8 {
        let values = inverse_8x8_1d(scaled[row * 8..row * 8 + 8].try_into().unwrap());
        horizontal[row * 8..row * 8 + 8].copy_from_slice(&values);
    }
    let mut residual = [0i32; 64];
    for column in 0..8 {
        let values = inverse_8x8_1d(std::array::from_fn(|row| horizontal[row * 8 + column]));
        for row in 0..8 {
            residual[row * 8 + column] = i32::try_from((values[row] + 32) >> 6)
                .map_err(|_| AvcError::InvalidData("8x8 residual out of range"))?;
        }
    }
    Ok(residual)
}

fn inverse_1d(values: [i32; 4]) -> [i32; 4] {
    let e0 = values[0] + values[2];
    let e1 = values[0] - values[2];
    let e2 = (values[1] >> 1) - values[3];
    let e3 = values[1] + (values[3] >> 1);
    [e0 + e3, e1 + e2, e1 - e2, e0 - e3]
}

/// Equations 8-264..8-282. `dc_is_pre_scaled` is true for Intra16x16/chroma
/// blocks whose DC entry has already passed through the separate DC transform.
pub fn inverse_4x4_residual(
    coefficients: &[i32; 16],
    qp: i32,
    dc_is_pre_scaled: bool,
) -> Result<[i32; 16], AvcError> {
    if !(0..=51).contains(&qp) {
        return Err(AvcError::InvalidData("4x4 residual QP out of range"));
    }
    let mut scaled = [0i32; 16];
    for (index, &coefficient) in coefficients.iter().enumerate() {
        let row = index / 4;
        let column = index % 4;
        let group = if row % 2 == 0 && column % 2 == 0 {
            0
        } else if row % 2 == 1 && column % 2 == 1 {
            1
        } else {
            2
        };
        let value = if index == 0 && dc_is_pre_scaled {
            i64::from(coefficient)
        } else {
            (i64::from(coefficient) * i64::from(LEVEL_SCALE[(qp % 6) as usize][group])) << (qp / 6)
        };
        scaled[index] = i32::try_from(value)
            .ok()
            .filter(|value| (-32768..=32767).contains(value))
            .ok_or(AvcError::InvalidData("scaled 4x4 coefficient out of range"))?;
    }
    let mut horizontal = [0i32; 16];
    for row in 0..4 {
        horizontal[row * 4..row * 4 + 4].copy_from_slice(&inverse_1d(
            scaled[row * 4..row * 4 + 4].try_into().unwrap(),
        ));
    }
    let mut residual = [0i32; 16];
    for column in 0..4 {
        let values = inverse_1d([
            horizontal[column],
            horizontal[4 + column],
            horizontal[8 + column],
            horizontal[12 + column],
        ]);
        for row in 0..4 {
            residual[row * 4 + column] = (values[row] + 32) >> 6;
        }
    }
    Ok(residual)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_zigzag_matches_table_8_12() {
        let mut levels = [0; 16];
        for (index, level) in levels.iter_mut().enumerate() {
            *level = index as i32;
        }
        assert_eq!(
            inverse_4x4_frame_scan(&levels),
            [0, 1, 5, 6, 2, 4, 7, 12, 3, 8, 11, 13, 9, 10, 14, 15,]
        );
    }

    #[test]
    fn frame_8x8_zigzag_matches_2005_table_8_14() {
        let levels = std::array::from_fn(|index| index as i32);
        let coefficients = inverse_8x8_frame_scan(&levels);
        assert_eq!(&coefficients[..8], &[0, 1, 5, 6, 14, 15, 27, 28]);
        assert_eq!(&coefficients[56..], &[35, 36, 48, 49, 57, 58, 62, 63]);
        for (scan_index, &position) in FRAME_ZIGZAG_8X8.iter().enumerate() {
            assert_eq!(coefficients[position], scan_index as i32);
        }
    }

    #[test]
    fn eight_by_eight_dc_and_scaling_follow_2005_transform() {
        let mut coefficients = [0; 64];
        coefficients[0] = 64;
        let weights = [16; 64];
        assert_eq!(
            inverse_8x8_residual(&coefficients, 0, &weights).unwrap(),
            [5; 64]
        );
        assert_eq!(
            inverse_8x8_residual(&coefficients, 6, &weights).unwrap(),
            [10; 64]
        );
        assert_eq!(
            inverse_8x8_residual(&[0; 64], 26, &weights).unwrap(),
            [0; 64]
        );
        assert!(inverse_8x8_residual(&coefficients, 52, &weights).is_err());
        assert!(inverse_8x8_residual(&[1_000_000; 64], 51, &weights).is_err());
    }

    #[test]
    fn intra16x16_dc_hadamard_and_qp_scaling() {
        let mut levels = [0; 16];
        levels[0] = 512;
        assert_eq!(inverse_luma16x16_dc(&levels, 0).unwrap(), [80; 16]);
        assert_eq!(inverse_luma16x16_dc(&levels, 6).unwrap(), [160; 16]);
        assert!(inverse_luma16x16_dc(&levels, 52).is_err());
        assert!(inverse_luma16x16_dc(&[1_000_000; 16], 51).is_err());
    }

    #[test]
    fn dc_only_residual_is_uniform() {
        let mut coefficients = [0; 16];
        coefficients[0] = -47;
        assert_eq!(
            inverse_4x4_residual(&coefficients, 0, false).unwrap(),
            [-7; 16]
        );
        assert_eq!(
            inverse_4x4_residual(&coefficients, 6, false).unwrap(),
            [-15; 16]
        );
        assert_eq!(
            inverse_4x4_residual(&coefficients, 0, true).unwrap(),
            [-1; 16]
        );
    }

    #[test]
    fn coefficient_ranges_are_checked() {
        assert!(inverse_4x4_residual(&[0; 16], 52, false).is_err());
        assert!(inverse_4x4_residual(&[1_000_000; 16], 51, false).is_err());
    }

    #[test]
    fn chroma_qp_and_dc_follow_2003_tables() {
        assert_eq!(chroma_qp(30, 0).unwrap(), 29);
        assert_eq!(chroma_qp(51, 0).unwrap(), 39);
        assert_eq!(chroma_qp(0, -12).unwrap(), 0);
        assert_eq!(inverse_chroma_dc(&[-1, 0, 0, 0], 0).unwrap(), [-5; 4]);
        assert_eq!(inverse_chroma_dc(&[-1, 0, 0, 0], 6).unwrap(), [-10; 4]);
        assert!(inverse_chroma_dc(&[1_000_000; 4], 51).is_err());
    }
}
