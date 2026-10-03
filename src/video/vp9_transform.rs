//! VP9 integer inverse DCT, following specification section 8.7.

use super::backend::MediaDecodeError;
use super::vp8_predict::Plane;
use super::vp9_predict;
use super::vp9_quant::{AC_QLOOKUP, DC_QLOOKUP};

const COS64_LOOKUP: [i32; 33] = [
    16384, 16364, 16305, 16207, 16069, 15893, 15679, 15426,
    15137, 14811, 14449, 14053, 13623, 13160, 12665, 12140,
    11585, 11003, 10394, 9760, 9102, 8423, 7723, 7005,
    6270, 5520, 4756, 3981, 3196, 2404, 1606, 804, 0,
];

fn cos64(angle: i32) -> i32 {
    let angle = angle.rem_euclid(128) as usize;
    match angle {
        0..=32 => COS64_LOOKUP[angle],
        33..=64 => -COS64_LOOKUP[64 - angle],
        65..=96 => -COS64_LOOKUP[angle - 64],
        _ => COS64_LOOKUP[128 - angle],
    }
}

fn round2(value: i64, shift: u32) -> i32 {
    ((value + (1 << (shift - 1))) >> shift) as i32
}

fn butterfly(values: &mut [i32], a: usize, b: usize, angle: i32, flip: bool) {
    let x = i64::from(values[a]) * i64::from(cos64(angle))
        - i64::from(values[b]) * i64::from(cos64(angle - 32));
    let y = i64::from(values[a]) * i64::from(cos64(angle - 32))
        + i64::from(values[b]) * i64::from(cos64(angle));
    let x = round2(x, 14);
    let y = round2(y, 14);
    values[a] = if flip { y } else { x };
    values[b] = if flip { x } else { y };
}

fn hadamard(values: &mut [i32], a: usize, b: usize, flip: bool) {
    let x = values[a];
    let y = values[b];
    values[a] = if flip { y - x } else { x + y };
    values[b] = if flip { y + x } else { x - y };
}

fn bit_reverse(bits: u32, value: usize) -> usize {
    value.reverse_bits() >> (usize::BITS - bits)
}

fn idct_stage(values: &mut [i32], n: u32) {
    let n0 = 1usize << n;
    let n1 = n0 >> 1;
    let n2 = n0 >> 2;
    let n3 = n0 >> 3;
    if n == 2 {
        butterfly(values, 0, 1, 16, true);
    } else {
        idct_stage(&mut values[..n1], n - 1);
    }
    for i in 0..n2 {
        butterfly(values, n1 + i, n0 - 1 - i, 32 - bit_reverse(5, n1 + i) as i32, false);
    }
    if n >= 3 {
        for i in 0..n3 {
            for j in 0..2 {
                hadamard(values, n1 + 4 * i + 2 * j, n1 + 1 + 4 * i + 2 * j, j != 0);
            }
        }
    }
    if n == 5 {
        for i in 0..2 {
            for j in 0..2 {
                butterfly(values, n0 - n as usize + 3 - n2 * j - 4 * i,
                    n1 + n as usize - 4 + n2 * j + 4 * i, 28 - 16 * i as i32 + 56 * j as i32, true);
            }
        }
        for i in 0..2 {
            for j in 0..4 {
                hadamard(values, n1 + n3 * j + i, n1 + n2 - 5 + n3 * j - i, j & 1 != 0);
            }
        }
    }
    if n >= 4 {
        for i in 0..if n == 5 { 2 } else { 1 } {
            for j in 0..2 {
                butterfly(values, n0 - n as usize + 2 - i - n2 * j,
                    n1 + n as usize - 3 + i + n2 * j, 24 + 48 * j as i32, true);
            }
        }
        for i in 0..n2 / 2 {
            for j in 0..2 {
                hadamard(values, n1 + n2 * j + i, n1 + n2 - 1 + n2 * j - i, j & 1 != 0);
            }
        }
    }
    if n >= 3 {
        for i in 0..n3 {
            butterfly(values, n0 - n3 - 1 - i, n1 + n3 + i, 16, true);
        }
    }
    for i in 0..n1 {
        hadamard(values, i, n0 - 1 - i, false);
    }
}

fn idct(values: &mut [i32], n: u32) {
    let mut source = [0i32; 32];
    source[..values.len()].copy_from_slice(values);
    for (index, value) in values.iter_mut().enumerate() {
        *value = source[bit_reverse(n, index)];
    }
    idct_stage(values, n);
}

fn adst_input_permute(values: &mut [i32]) {
    let mut source = [0i32; 16];
    source[..values.len()].copy_from_slice(values);
    let half = values.len() / 2;
    for index in 0..half {
        values[index * 2] = source[values.len() - 1 - index * 2];
        values[index * 2 + 1] = source[index * 2];
    }
}

fn adst_output_permute(values: &mut [i32]) {
    let mut source = [0i32; 16];
    source[..values.len()].copy_from_slice(values);
    if values.len() == 8 {
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    values[4 * a + 2 * b + c] = source[4 * (c ^ b) + 2 * (b ^ a) + a];
                }
            }
        }
    } else {
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    for d in 0..2 {
                        values[8 * a + 4 * b + 2 * c + d] =
                            source[8 * (d ^ c) + 4 * (c ^ b) + 2 * (b ^ a) + a];
                    }
                }
            }
        }
    }
}

fn adst_butterfly(values: &[i32], scaled: &mut [i64; 16], a: usize, b: usize, angle: i32, flip: bool) {
    let x = i64::from(values[a]) * i64::from(cos64(angle))
        - i64::from(values[b]) * i64::from(cos64(angle - 32));
    let y = i64::from(values[a]) * i64::from(cos64(angle - 32))
        + i64::from(values[b]) * i64::from(cos64(angle));
    scaled[a] = if flip { y } else { x };
    scaled[b] = if flip { x } else { y };
}

fn adst_hadamard(values: &mut [i32], scaled: &[i64; 16], a: usize, b: usize) {
    values[a] = round2(scaled[a] + scaled[b], 14);
    values[b] = round2(scaled[a] - scaled[b], 14);
}

fn iadst(values: &mut [i32]) {
    if values.len() == 4 {
        let [a, b, c, d] = [values[0], values[1], values[2], values[3]];
        let s0 = 5283i64 * i64::from(a);
        let s1 = 9929i64 * i64::from(a);
        let s2 = 13377i64 * i64::from(b);
        let s3 = 15212i64 * i64::from(c);
        let s4 = 5283i64 * i64::from(c);
        let s5 = 9929i64 * i64::from(d);
        let s6 = 15212i64 * i64::from(d);
        let s7 = 13377i64 * i64::from(a - c + d);
        let x0 = s0 + s3 + s5;
        let x1 = s1 - s4 - s6;
        values[0] = round2(x0 + s2, 14);
        values[1] = round2(x1 + s2, 14);
        values[2] = round2(s7, 14);
        values[3] = round2(x0 + x1 - s2, 14);
        return;
    }
    adst_input_permute(values);
    let mut scaled = [0i64; 16];
    if values.len() == 8 {
        for i in 0..4 {
            adst_butterfly(values, &mut scaled, 2 * i, 2 * i + 1, 30 - 8 * i as i32, true);
        }
        for i in 0..4 {
            adst_hadamard(values, &scaled, i, 4 + i);
        }
        for i in 0..2 {
            adst_butterfly(values, &mut scaled, 4 + 3 * i, 5 + i, 24 - 16 * i as i32, true);
        }
        for i in 0..2 {
            adst_hadamard(values, &scaled, 4 + i, 6 + i);
        }
        for i in 0..2 {
            hadamard(values, i, 2 + i, false);
        }
        for i in 0..2 {
            butterfly(values, 2 + 4 * i, 3 + 4 * i, 16, true);
        }
        adst_output_permute(values);
        for i in 0..4 {
            values[1 + 2 * i] = -values[1 + 2 * i];
        }
        return;
    }
    for i in 0..8 {
        adst_butterfly(values, &mut scaled, 2 * i, 2 * i + 1, 31 - 4 * i as i32, true);
    }
    for i in 0..8 {
        adst_hadamard(values, &scaled, i, 8 + i);
    }
    for i in 0..4 {
        adst_butterfly(values, &mut scaled, 8 + 2 * i, 9 + 2 * i, 28 - 16 * i as i32, true);
    }
    for i in 0..4 {
        adst_hadamard(values, &scaled, 8 + i, 12 + i);
        hadamard(values, i, 4 + i, false);
    }
    for i in 0..2 {
        for j in 0..2 {
            adst_butterfly(values, &mut scaled, 4 + 8 * i + 3 * j, 5 + 8 * i + j,
                24 - 16 * j as i32, true);
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            adst_hadamard(values, &scaled, 4 + 8 * j + i, 6 + 8 * j + i);
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            hadamard(values, 8 * j + i, 2 + 8 * j + i, false);
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            butterfly(values, 2 + 4 * j + 8 * i, 3 + 4 * j + 8 * i,
                48 + 64 * (i ^ j) as i32, false);
        }
    }
    adst_output_permute(values);
    for i in 0..2 {
        for j in 0..2 {
            values[1 + 12 * j + 2 * i] = -values[1 + 12 * j + 2 * i];
        }
    }
}

fn inverse_transform_square(dequant: &mut [i32], coefficients: &[i32], size: usize,
    bit_depth: u8, dc_index: i32, ac_index: i32, tx_type: u8) {
    let table = usize::from((bit_depth.saturating_sub(8)) >> 1).min(2);
    let dc = DC_QLOOKUP[table][dc_index.clamp(0, 255) as usize] as i32;
    let ac = AC_QLOOKUP[table][ac_index.clamp(0, 255) as usize] as i32;
    let divisor = if size == 32 { 2 } else { 1 };
    for (output, &coefficient) in dequant.iter_mut().zip(coefficients.iter()) {
        *output = coefficient * ac / divisor;
    }
    dequant[0] = coefficients[0] * dc / divisor;
    if tx_type == 0 && dequant[1..].iter().all(|&value| value == 0) {
        dequant.fill(dc_residual(dequant[0], size));
        return;
    }
    inverse_transform_2d(dequant, size, tx_type);
}

fn dc_residual(dequantized: i32, size: usize) -> i32 {
    let horizontal = round2(i64::from(dequantized) * i64::from(cos64(16)), 14);
    let vertical = round2(i64::from(horizontal) * i64::from(cos64(16)), 14);
    round2(i64::from(vertical), (size.trailing_zeros() + 2).min(6))
}

fn inverse_transform_2d(dequant: &mut [i32], size: usize, tx_type: u8) {
    let n = size.trailing_zeros();
    for row in dequant.chunks_exact_mut(size) {
        if row.iter().all(|&value| value == 0) { continue; }
        if tx_type == 0 || tx_type == 1 { idct(row, n); } else { iadst(row); }
    }
    for column in 0..size {
        let mut values = [0i32; 32];
        for row in 0..size {
            values[row] = dequant[row * size + column];
        }
        if values[..size].iter().all(|&value| value == 0) { continue; }
        if tx_type == 0 || tx_type == 2 { idct(&mut values[..size], n); }
        else { iadst(&mut values[..size]); }
        for row in 0..size {
            dequant[row * size + column] = round2(i64::from(values[row]), (n + 2).min(6));
        }
    }
}

fn tx_type(mode: u8, chroma: bool, size: usize) -> u8 {
    if chroma || size == 32 {
        return 0;
    }
    match mode {
        1 | 5 | 8 => 1,
        2 | 6 | 7 => 2,
        4 | 9 => 3,
        _ => 0,
    }
}

fn inverse_wht(values: &mut [i32], shift: u32) {
    let mut a = values[0] >> shift;
    let mut c = values[1] >> shift;
    let mut d = values[2] >> shift;
    let mut b = values[3] >> shift;
    a += c;
    d -= b;
    let e = (a - d) >> 1;
    b = e - b;
    c = e - c;
    a -= b;
    d += c;
    values.copy_from_slice(&[a, b, c, d]);
}

// Sections 8.7.1.10 and 8.7.2: lossless dequantization is four, with no final rounding.
fn inverse_lossless(coefficients: &[i32]) -> [i32; 16] {
    let mut residual = [0; 16];
    for (output, &coefficient) in residual.iter_mut().zip(coefficients) {
        *output = coefficient * 4;
    }
    for row in residual.chunks_exact_mut(4) {
        inverse_wht(row, 2);
    }
    for column in 0..4 {
        let mut values = std::array::from_fn::<_, 4, _>(|row| residual[row * 4 + column]);
        inverse_wht(&mut values, 0);
        for row in 0..4 {
            residual[row * 4 + column] = values[row];
        }
    }
    residual
}

pub(super) fn reconstruct_intra(
    plane: &mut Plane,
    x: usize,
    y: usize,
    size: usize,
    mode: u8,
    chroma: bool,
    have_left: bool,
    have_above: bool,
    not_on_right: bool,
    visible_width: usize,
    visible_height: usize,
    coefficients: &[i32],
    bit_depth: u8,
    dc_index: i32,
    ac_index: i32,
    lossless: bool,
) -> Result<(), MediaDecodeError> {
    if bit_depth != 8 {
        return Err(MediaDecodeError::Unsupported);
    }
    if !matches!(size, 4 | 8 | 16 | 32) || coefficients.len() != size * size
        || (lossless && size != 4) {
        return Err(MediaDecodeError::InvalidData("invalid VP9 transform coefficient count".into()));
    }
    vp9_predict::predict(plane, x, y, size, mode, have_left, have_above, not_on_right,
        visible_width, visible_height)?;
    if coefficients.iter().all(|&value| value == 0) { return Ok(()); }
    let mut storage = [0; 1024];
    let residual = &mut storage[..size * size];
    if lossless {
        residual.copy_from_slice(&inverse_lossless(coefficients));
    } else {
        inverse_transform_square(residual, coefficients, size, bit_depth, dc_index, ac_index,
            tx_type(mode, chroma, size));
    }
    for row in 0..size {
        for column in 0..size {
            let index = (y + row) * plane.width + x + column;
            plane.pixels[index] = (i32::from(plane.pixels[index])
                + residual[row * size + column]).clamp(0, 255) as u8;
        }
    }
    Ok(())
}

pub(super) fn add_inter_residual(
    plane: &mut Plane, x: usize, y: usize, size: usize, coefficients: &[i32],
    dc_index: i32, ac_index: i32, lossless: bool,
) -> Result<(), MediaDecodeError> {
    if !matches!(size, 4 | 8 | 16 | 32) || coefficients.len() != size * size
        || x + size > plane.width || (y + size) * plane.width > plane.pixels.len()
        || (lossless && size != 4) {
        return Err(MediaDecodeError::InvalidData("invalid VP9 inter transform".into()));
    }
    if !lossless && coefficients[1..].iter().all(|&value| value == 0) {
        let dc = i32::from(DC_QLOOKUP[0][dc_index.clamp(0, 255) as usize]);
        let value = dc_residual(coefficients[0] * dc / if size == 32 { 2 } else { 1 }, size);
        if value != 0 {
            for row in 0..size {
                let start = (y + row) * plane.width + x;
                for pixel in &mut plane.pixels[start..start + size] {
                    *pixel = (i32::from(*pixel) + value).clamp(0, 255) as u8;
                }
            }
        }
        return Ok(());
    }
    let mut storage = [0; 1024];
    let residual = &mut storage[..size * size];
    if lossless {
        residual.copy_from_slice(&inverse_lossless(coefficients));
    } else {
        inverse_transform_square(residual, coefficients, size, 8, dc_index, ac_index, 0);
    }
    for row in 0..size {
        let start = (y + row) * plane.width + x;
        for (pixel, &value) in plane.pixels[start..start + size].iter_mut()
            .zip(&residual[row * size..(row + 1) * size])
        {
            *pixel = (i32::from(*pixel) + value).clamp(0, 255) as u8;
        }
    }
    Ok(())
}

pub(super) fn inverse_dct_32x32(coefficients: &[i32], bit_depth: u8, dc_index: i32, ac_index: i32) -> [i32; 1024] {
    let mut dequant = [0i32; 1024];
    inverse_transform_square(&mut dequant, coefficients, 32, bit_depth, dc_index, ac_index, 0);
    dequant
}

pub(super) fn inverse_dct_16x16(coefficients: &[i32], bit_depth: u8, dc_index: i32, ac_index: i32) -> [i32; 256] {
    let mut dequant = [0i32; 256];
    inverse_transform_square(&mut dequant, coefficients, 16, bit_depth, dc_index, ac_index, 0);
    dequant
}

pub(super) fn inverse_dct_8x8(coefficients: &[i32], bit_depth: u8, dc_index: i32, ac_index: i32) -> [i32; 64] {
    let mut dequant = [0i32; 64];
    inverse_transform_square(&mut dequant, coefficients, 8, bit_depth, dc_index, ac_index, 0);
    dequant
}

pub(super) fn inverse_dct_4x4(coefficients: &[i32], bit_depth: u8, dc_index: i32, ac_index: i32) -> [i32; 16] {
    let mut dequant = [0i32; 16];
    inverse_transform_square(&mut dequant, coefficients, 4, bit_depth, dc_index, ac_index, 0);
    dequant
}

pub(super) fn reconstruct_32x32_intra(
    plane: &mut Plane,
    x: usize,
    y: usize,
    mode: u8,
    have_left: bool,
    have_above: bool,
    not_on_right: bool,
    visible_width: usize,
    visible_height: usize,
    coefficients: &[i32],
    bit_depth: u8,
    dc_index: i32,
    ac_index: i32,
) -> Result<(), MediaDecodeError> {
    if bit_depth != 8 {
        return Err(MediaDecodeError::Unsupported);
    }
    if coefficients.len() != 1024 {
        return Err(MediaDecodeError::InvalidData("invalid VP9 transform coefficient count".into()));
    }
    vp9_predict::predict(
        plane, x, y, 32, mode, have_left, have_above, not_on_right,
        visible_width, visible_height,
    )?;
    let residual = inverse_dct_32x32(coefficients, bit_depth, dc_index, ac_index);
    for row in 0..32 {
        for column in 0..32 {
            let index = (y + row) * plane.width + x + column;
            plane.pixels[index] = (i32::from(plane.pixels[index])
                + residual[row * 32 + column]).clamp(0, 255) as u8;
        }
    }
    Ok(())
}

pub(super) fn reconstruct_16x16_intra(
    plane: &mut Plane,
    x: usize,
    y: usize,
    mode: u8,
    have_left: bool,
    have_above: bool,
    visible_width: usize,
    visible_height: usize,
    coefficients: &[i32],
    bit_depth: u8,
    dc_index: i32,
    ac_index: i32,
) -> Result<(), MediaDecodeError> {
    if bit_depth != 8 {
        return Err(MediaDecodeError::Unsupported);
    }
    if coefficients.len() != 256 {
        return Err(MediaDecodeError::InvalidData("invalid VP9 transform coefficient count".into()));
    }
    vp9_predict::predict(plane, x, y, 16, mode, have_left, have_above, false,
        visible_width, visible_height)?;
    let residual = inverse_dct_16x16(coefficients, bit_depth, dc_index, ac_index);
    for row in 0..16 {
        for column in 0..16 {
            let index = (y + row) * plane.width + x + column;
            plane.pixels[index] = (i32::from(plane.pixels[index])
                + residual[row * 16 + column]).clamp(0, 255) as u8;
        }
    }
    Ok(())
}

pub(super) fn reconstruct_8x8_intra(
    plane: &mut Plane,
    x: usize,
    y: usize,
    mode: u8,
    have_left: bool,
    have_above: bool,
    visible_width: usize,
    visible_height: usize,
    coefficients: &[i32],
    bit_depth: u8,
    dc_index: i32,
    ac_index: i32,
) -> Result<(), MediaDecodeError> {
    if bit_depth != 8 {
        return Err(MediaDecodeError::Unsupported);
    }
    if coefficients.len() != 64 {
        return Err(MediaDecodeError::InvalidData("invalid VP9 transform coefficient count".into()));
    }
    vp9_predict::predict(plane, x, y, 8, mode, have_left, have_above, false,
        visible_width, visible_height)?;
    let residual = inverse_dct_8x8(coefficients, bit_depth, dc_index, ac_index);
    for row in 0..8 {
        for column in 0..8 {
            let index = (y + row) * plane.width + x + column;
            plane.pixels[index] = (i32::from(plane.pixels[index])
                + residual[row * 8 + column]).clamp(0, 255) as u8;
        }
    }
    Ok(())
}

pub(super) fn reconstruct_4x4_intra(
    plane: &mut Plane,
    x: usize,
    y: usize,
    mode: u8,
    have_left: bool,
    have_above: bool,
    visible_width: usize,
    visible_height: usize,
    coefficients: &[i32],
    bit_depth: u8,
    dc_index: i32,
    ac_index: i32,
) -> Result<(), MediaDecodeError> {
    if bit_depth != 8 {
        return Err(MediaDecodeError::Unsupported);
    }
    if coefficients.len() != 16 {
        return Err(MediaDecodeError::InvalidData("invalid VP9 transform coefficient count".into()));
    }
    vp9_predict::predict(plane, x, y, 4, mode, have_left, have_above, false,
        visible_width, visible_height)?;
    let residual = inverse_dct_4x4(coefficients, bit_depth, dc_index, ac_index);
    for row in 0..4 {
        for column in 0..4 {
            let index = (y + row) * plane.width + x + column;
            plane.pixels[index] = (i32::from(plane.pixels[index])
                + residual[row * 4 + column]).clamp(0, 255) as u8;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_dc_addition_matches_full_transform_and_clipping() {
        for size in [4, 8, 16, 32] {
            for quantizer in [0, 46, 255] {
                for dc in [-1024, -17, 0, 1, 64, 1024] {
                    let mut coefficients = vec![0; size * size];
                    coefficients[0] = dc;
                    let mut residual = vec![0; size * size];
                    let scale = i32::from(DC_QLOOKUP[0][quantizer]);
                    residual[0] = dc * scale / if size == 32 { 2 } else { 1 };
                    inverse_transform_2d(&mut residual, size, 0);
                    let mut plane = Plane::new(size + 8, size + 8);
                    for (index, pixel) in plane.pixels.iter_mut().enumerate() {
                        *pixel = index.wrapping_mul(37) as u8;
                    }
                    let mut expected = plane.pixels.clone();
                    for row in 0..size {
                        for col in 0..size {
                            let index = (row + 4) * plane.width + col + 4;
                            expected[index] = (i32::from(expected[index]) + residual[row * size + col])
                                .clamp(0, 255) as u8;
                        }
                    }
                    add_inter_residual(&mut plane, 4, 4, size, &coefficients,
                        quantizer as i32, quantizer as i32, false).unwrap();
                    assert_eq!(plane.pixels, expected);
                }
            }
        }
    }

    #[test]
    fn zero_transform_stays_zero() {
        assert!(inverse_dct_32x32(&[0; 1024], 8, 46, 46).iter().all(|&value| value == 0));
        assert!(inverse_dct_16x16(&[0; 256], 8, 46, 46).iter().all(|&value| value == 0));
        assert!(inverse_dct_8x8(&[0; 64], 8, 46, 46).iter().all(|&value| value == 0));
        assert!(inverse_dct_4x4(&[0; 16], 8, 46, 46).iter().all(|&value| value == 0));
    }

    #[test]
    fn dc_only_transform_is_constant() {
        let mut coefficients = [0; 1024];
        coefficients[0] = 64;
        let output = inverse_dct_32x32(&coefficients, 8, 46, 46);
        assert!(output.iter().all(|&value| value == output[0]));
        assert!(output[0] > 0);
    }

    #[test]
    fn dc_shortcut_matches_full_transform_at_every_size_and_depth() {
        for size in [4usize, 8, 16, 32] {
            for bit_depth in [8u8, 10, 12] {
                for quantizer in [0i32, 46, 255] {
                    for coefficient in [-1024i32, -17, 0, 1, 64, 1024] {
                        let mut coefficients = vec![0; size * size];
                        coefficients[0] = coefficient;
                        let mut actual = vec![0; size * size];
                        inverse_transform_square(&mut actual, &coefficients, size,
                            bit_depth, quantizer, quantizer, 0);
                        let mut expected = vec![0; size * size];
                        expected[0] = coefficient * i32::from(
                            DC_QLOOKUP[usize::from((bit_depth - 8) >> 1)][quantizer as usize])
                            / if size == 32 { 2 } else { 1 };
                        inverse_transform_2d(&mut expected, size, 0);
                        assert_eq!(actual, expected, "size={size} depth={bit_depth} q={quantizer} dc={coefficient}");
                    }
                }
            }
        }
    }
}
