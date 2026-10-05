//! Bounded pixel reconstruction stages, AV1 sections 7.11.2, 7.12.3, 7.13.3.
//! These consume decoded block data, not compressed tile bytes.

use super::syntax::Error;

/// Apply section 7.11.5 CFL to a DC prediction. Luma carries three fractional bits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_cfl(
    prediction: &mut [u16],
    w: usize,
    h: usize,
    bit_depth: u8,
    luma: &[u16],
    stride: usize,
    max_w: usize,
    max_h: usize,
    x: usize,
    y: usize,
    sub_x: bool,
    sub_y: bool,
    alpha: i32,
) -> Result<(), Error> {
    let sx = usize::from(sub_x);
    let sy = usize::from(sub_y);
    if ![8, 10, 12].contains(&bit_depth)
        || prediction.len() != w * h
        || max_w < 1 << sx
        || max_h < 1 << sy
        || max_w > stride
        || max_h.checked_mul(stride).is_none_or(|n| n > luma.len())
        || !(4..=32).contains(&w)
        || !(4..=32).contains(&h)
        || !w.is_power_of_two()
        || !h.is_power_of_two()
        || alpha.abs() > 16
    {
        return Err(Error::Invalid("CFL inputs"));
    }
    let mut ac = vec![0i32; w * h];
    for row in 0..h {
        let ly = ((y + row) << sy).min(max_h - (1 << sy));
        for col in 0..w {
            let lx = ((x + col) << sx).min(max_w - (1 << sx));
            let mut sum = 0;
            for dy in 0..=sy {
                for dx in 0..=sx {
                    sum += i32::from(luma[(ly + dy) * stride + lx + dx]);
                }
            }
            ac[row * w + col] = sum << (3 - sx - sy);
        }
    }
    let sum: i32 = ac.iter().sum();
    let shift = w.ilog2() + h.ilog2();
    let average = (sum + (1 << (shift - 1))) >> shift;
    let max = (1i32 << bit_depth) - 1;
    for (sample, luma) in prediction.iter_mut().zip(ac) {
        let value = alpha * (luma - average);
        let scaled = value.signum() * ((value.abs() + 32) >> 6);
        *sample = (i32::from(*sample) + scaled).clamp(0, max) as u16;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraMode {
    Dc,
    Vertical,
    Horizontal,
    Paeth,
}

#[cfg(test)]
thread_local! {
    static LAZY_EDGES_REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn set_lazy_edges_reference(reference: bool) {
    LAZY_EDGES_REFERENCE.set(reference);
}

#[cfg(test)]
pub(crate) fn lazy_edges_reference() -> bool {
    LAZY_EDGES_REFERENCE.get()
}

#[inline(always)]
pub(crate) fn validate_intra_inputs(
    mode: IntraMode,
    width: usize,
    height: usize,
    bit_depth: u8,
    above: Option<&[u16]>,
    left: Option<&[u16]>,
    top_left: Option<u16>,
) -> Result<(), Error> {
    if ![8, 10, 12].contains(&bit_depth)
        || ![4, 8, 16, 32, 64].contains(&width)
        || ![4, 8, 16, 32, 64].contains(&height)
        || width > height * 4
        || height > width * 4
    {
        return Err(Error::Invalid("intra block dimensions or bit depth"));
    }
    let max = (1u16 << bit_depth) - 1;
    if above.is_some_and(|a| a.len() < width || a.iter().any(|&v| v > max))
        || left.is_some_and(|l| l.len() < height || l.iter().any(|&v| v > max))
        || top_left.is_some_and(|v| v > max)
    {
        return Err(Error::Invalid("intra boundary samples"));
    }
    if mode == IntraMode::Paeth && above.is_some() && left.is_some() && top_left.is_none() {
        return Err(Error::Invalid("missing available corner"));
    }
    Ok(())
}

/// Predict an AV1 transform block using already reconstructed boundary samples.
/// Missing boundaries are expanded per 7.11.2.1, not treated as zero samples.
pub fn predict_intra(
    mode: IntraMode,
    width: usize,
    height: usize,
    bit_depth: u8,
    above: Option<&[u16]>,
    left: Option<&[u16]>,
    top_left: Option<u16>,
) -> Result<Vec<u16>, Error> {
    #[cfg(test)]
    let _measure = super::profile::measure(3);
    validate_intra_inputs(mode, width, height, bit_depth, above, left, top_left)?;
    let midpoint = 1u16 << (bit_depth - 1);
    #[cfg(not(test))]
    let reference = false;
    #[cfg(test)]
    let reference = lazy_edges_reference();
    let top = (reference || matches!(mode, IntraMode::Vertical | IntraMode::Paeth)).then(|| {
        above.map_or_else(
            || std::borrow::Cow::Owned(vec![left.map_or(midpoint - 1, |l| l[0]); width]),
            |a| {
                if reference {
                    std::borrow::Cow::Owned(a[..width].to_vec())
                } else {
                    std::borrow::Cow::Borrowed(&a[..width])
                }
            },
        )
    });
    let side = (reference || matches!(mode, IntraMode::Horizontal | IntraMode::Paeth)).then(|| {
        left.map_or_else(
            || std::borrow::Cow::Owned(vec![above.map_or(midpoint + 1, |a| a[0]); height]),
            |l| {
                if reference {
                    std::borrow::Cow::Owned(l[..height].to_vec())
                } else {
                    std::borrow::Cow::Borrowed(&l[..height])
                }
            },
        )
    });
    let corner = match (above, left) {
        (Some(_), Some(_)) => top_left.unwrap_or(midpoint),
        (Some(a), None) => a[0],
        (None, Some(l)) => l[0],
        (None, None) => midpoint,
    };
    let mut sum = 0u32;
    let mut count = 0u32;
    if let Some(a) = above {
        sum += a[..width].iter().map(|&v| u32::from(v)).sum::<u32>();
        count += width as u32;
    }
    if let Some(l) = left {
        sum += l[..height].iter().map(|&v| u32::from(v)).sum::<u32>();
        count += height as u32;
    }
    let dc = if count == 0 {
        midpoint
    } else {
        ((sum + count / 2) / count) as u16
    };
    let mut pixels = vec![0; width * height];
    for y in 0..height {
        for x in 0..width {
            pixels[y * width + x] = match mode {
                IntraMode::Dc => dc,
                IntraMode::Vertical => top.as_ref().unwrap()[x],
                IntraMode::Horizontal => side.as_ref().unwrap()[y],
                IntraMode::Paeth => {
                    let top = top.as_ref().unwrap();
                    let side = side.as_ref().unwrap();
                    let base = i32::from(top[x]) + i32::from(side[y]) - i32::from(corner);
                    let dl = (base - i32::from(side[y])).abs();
                    let dt = (base - i32::from(top[x])).abs();
                    let dc = (base - i32::from(corner)).abs();
                    if dl <= dt && dl <= dc {
                        side[y]
                    } else if dt <= dc {
                        top[x]
                    } else {
                        corner
                    }
                }
            };
        }
    }
    Ok(pixels)
}

fn wht(t: [i64; 4], shift: u8) -> [i64; 4] {
    let (mut a, mut c, mut d, mut b) = (t[0] >> shift, t[1] >> shift, t[2] >> shift, t[3] >> shift);
    a += c;
    d -= b;
    let e = (a - d) >> 1;
    b = e - b;
    c = e - c;
    a -= b;
    d += c;
    [a, b, c, d]
}

/// Lossless 4x4 inverse transform. Input must already be dequantized.
pub fn inverse_lossless_4x4(coefficients: &[i32; 16], bit_depth: u8) -> Result<[i32; 16], Error> {
    if ![8, 10, 12].contains(&bit_depth) {
        return Err(Error::Invalid("transform bit depth"));
    }
    let bound = 1i32 << (7 + bit_depth);
    if coefficients.iter().any(|&v| v < -bound || v >= bound) {
        return Err(Error::Invalid("dequantized coefficient range"));
    }
    let mut rows = [[0i64; 4]; 4];
    let clamp = 1i64 << ((bit_depth + 6).max(16) - 1);
    for y in 0..4 {
        rows[y] = wht(
            std::array::from_fn(|x| i64::from(coefficients[y * 4 + x])),
            2,
        )
        .map(|v| v.clamp(-clamp, clamp - 1));
    }
    let mut output = [0; 16];
    for x in 0..4 {
        let col = wht(std::array::from_fn(|y| rows[y][x]), 0);
        for y in 0..4 {
            output[y * 4 + x] = col[y] as i32;
        }
    }
    Ok(output)
}

/// Add actual residual samples and Clip1 to the coded sample precision.
pub fn add_residual(prediction: &mut [u16], residual: &[i32], bit_depth: u8) -> Result<(), Error> {
    if ![8, 10, 12].contains(&bit_depth) || prediction.len() != residual.len() {
        return Err(Error::Invalid("residual dimensions or bit depth"));
    }
    let max = (1i64 << bit_depth) - 1;
    if prediction.iter().any(|&v| i64::from(v) > max) {
        return Err(Error::Invalid("prediction sample range"));
    }
    for (p, &r) in prediction.iter_mut().zip(residual) {
        *p = (i64::from(*p) + i64::from(r)).clamp(0, max) as u16;
    }
    Ok(())
}

/// Exact staged inverse transform for a single dequantized DC coefficient.
/// No AC coefficient is discarded: callers must have decoded EOB equal to one.
pub fn inverse_dc(
    w: usize,
    h: usize,
    dc: i32,
    bit_depth: u8,
    lossless: bool,
) -> Result<Vec<i32>, Error> {
    let index = super::coefficients::tx_index(w, h)?;
    if ![8, 10, 12].contains(&bit_depth) {
        return Err(Error::Invalid("transform bit depth"));
    }
    let bound = 1i32 << (7 + bit_depth);
    if dc < -bound || dc >= bound {
        return Err(Error::Invalid("dequantized coefficient range"));
    }
    if lossless {
        if w != 4 || h != 4 {
            return Err(Error::Invalid("lossless transform dimensions"));
        }
        let mut coefficients = [0; 16];
        coefficients[0] = dc;
        return Ok(inverse_lossless_4x4(&coefficients, bit_depth)?.to_vec());
    }
    fn round(value: i64, shift: u8) -> i64 {
        if shift == 0 {
            value
        } else {
            (value + (1i64 << (shift - 1))) >> shift
        }
    }
    let mut value = i64::from(dc);
    if w.ilog2().abs_diff(h.ilog2()) == 1 {
        value = round(value * 2896, 12);
    }
    value = round(value * 2896, 12);
    let row_shift = [0, 1, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2][index];
    value = round(value, row_shift);
    let clamp = 1i64 << ((bit_depth + 6).max(16) - 1);
    value = value.clamp(-clamp, clamp - 1);
    value = round(round(value * 2896, 12), 4);
    Ok(vec![value as i32; w * h])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lazy_edges_match_allocating_prediction_and_errors() {
        for depth in [8, 10, 12] {
            for (w, h) in super::super::coefficients::TX_DIMENSIONS {
                for mode in [
                    IntraMode::Dc,
                    IntraMode::Vertical,
                    IntraMode::Horizontal,
                    IntraMode::Paeth,
                ] {
                    for trial in 0..12 {
                        let mut above = vec![73; w + h];
                        let mut left = vec![39; w + h];
                        if trial == 8 {
                            above[w + h - 1] = u16::MAX;
                        }
                        if trial == 9 {
                            left[w + h - 1] = u16::MAX;
                        }
                        if trial == 10 {
                            above.truncate(w - 1);
                        }
                        if trial == 11 {
                            left.truncate(h - 1);
                        }
                        let a = (trial != 1 && trial != 3).then_some(above.as_slice());
                        let l = (trial != 2 && trial != 3).then_some(left.as_slice());
                        let corner = match trial {
                            4 => None,
                            5 => Some(u16::MAX),
                            _ => Some(55),
                        };
                        set_lazy_edges_reference(true);
                        let expected = predict_intra(mode, w, h, depth, a, l, corner);
                        set_lazy_edges_reference(false);
                        assert_eq!(predict_intra(mode, w, h, depth, a, l, corner), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn cfl_averages_complete_transforms_before_visible_cropping() {
        let mut luma = vec![0; 64];
        luma[32..].fill(64);
        let mut full = vec![128; 16];
        predict_cfl(&mut full, 4, 4, 8, &luma, 8, 8, 8, 0, 0, true, true, 8).unwrap();
        assert_eq!(&full[..8], &[96; 8]);
        assert_eq!(&full[8..], &[160; 8]);
        let mut prematurely_cropped = vec![128; 16];
        predict_cfl(
            &mut prematurely_cropped,
            4,
            4,
            8,
            &luma,
            8,
            8,
            4,
            0,
            0,
            true,
            true,
            8,
        )
        .unwrap();
        assert_eq!(prematurely_cropped, vec![128; 16]);
    }

    #[test]
    fn dc_edges_and_missing_edges() {
        for depth in [8, 10, 12] {
            assert_eq!(
                predict_intra(IntraMode::Dc, 4, 8, depth, None, None, None).unwrap(),
                vec![1 << (depth - 1); 32]
            );
        }
        let a = [10, 20, 30, 40];
        let l = [50; 8];
        assert_eq!(
            predict_intra(IntraMode::Dc, 4, 8, 8, Some(&a), Some(&l), None).unwrap(),
            vec![42; 32]
        );
        assert_eq!(
            predict_intra(IntraMode::Dc, 4, 8, 8, Some(&a), None, None).unwrap(),
            vec![25; 32]
        );
        assert_eq!(
            predict_intra(IntraMode::Vertical, 4, 8, 8, None, Some(&l), None).unwrap(),
            vec![50; 32]
        );
        let horizontal =
            predict_intra(IntraMode::Horizontal, 4, 4, 8, None, Some(&a), None).unwrap();
        assert_eq!(
            horizontal,
            [vec![10; 4], vec![20; 4], vec![30; 4], vec![40; 4]].concat()
        );
    }

    #[test]
    fn paeth_tie_uses_left_then_top() {
        let a = [100, 120, 80, 100];
        let l = [100; 4];
        assert_eq!(
            predict_intra(IntraMode::Paeth, 4, 4, 8, Some(&a), Some(&l), Some(100)).unwrap(),
            a.repeat(4)
        );
    }

    #[test]
    fn lossless_dc_and_zero_transform() {
        assert_eq!(inverse_lossless_4x4(&[0; 16], 8).unwrap(), [0; 16]);
        for dc in [-4096, -16, 0, 16, 4096] {
            let mut input = [0; 16];
            input[0] = dc;
            assert_eq!(inverse_lossless_4x4(&input, 8).unwrap(), [dc / 16; 16]);
        }
    }

    #[test]
    fn residual_clips_without_overflow() {
        let mut p = [128, 128, 128, 128];
        add_residual(&mut p, &[i32::MIN, -1, 1, i32::MAX], 8).unwrap();
        assert_eq!(p, [0, 127, 129, 255]);
    }

    #[test]
    fn lossless_transform_matches_independent_forward_lifting() {
        // Reverse the lifting equations from samples to coefficients. This
        // checks negative rounding and all AC positions, not just DC impulses.
        fn forward(t: [i32; 4]) -> [i32; 4] {
            let a = t[0] + t[1];
            let d = t[3] - t[2];
            let e = (a - d).div_euclid(2);
            let b = e - t[1];
            let c = e - t[2];
            [a - c, c, d + b, b]
        }
        let mut state = 173u32;
        for _ in 0..512 {
            let pixels: [i32; 16] = std::array::from_fn(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 24) as i32) - 128
            });
            let mut intermediate = [0; 16];
            for x in 0..4 {
                let column = forward(std::array::from_fn(|y| pixels[y * 4 + x]));
                for y in 0..4 {
                    intermediate[y * 4 + x] = column[y];
                }
            }
            let mut coefficients = [0; 16];
            for y in 0..4 {
                let row = forward(std::array::from_fn(|x| intermediate[y * 4 + x]));
                for x in 0..4 {
                    coefficients[y * 4 + x] = row[x] * 4;
                }
            }
            assert_eq!(inverse_lossless_4x4(&coefficients, 8).unwrap(), pixels);
        }
    }
}
