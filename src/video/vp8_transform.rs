//! VP8's 4x4 inverse transforms, using the fixed-point rounding of RFC 6386.

const COS_PI_8_SQRT2_MINUS_1: i32 = 20091;
const SIN_PI_8_SQRT2: i32 = 35468;

pub(super) fn inverse_walsh(input: &[i32; 16]) -> [i32; 16] {
    let mut columns = [0; 16];
    for x in 0..4 {
        let a = input[x] + input[12 + x];
        let b = input[4 + x] + input[8 + x];
        let c = input[4 + x] - input[8 + x];
        let d = input[x] - input[12 + x];
        columns[x] = a + b;
        columns[4 + x] = c + d;
        columns[8 + x] = a - b;
        columns[12 + x] = d - c;
    }
    let mut result = [0; 16];
    for y in 0..4 {
        let row = y * 4;
        let a = columns[row] + columns[row + 3];
        let b = columns[row + 1] + columns[row + 2];
        let c = columns[row + 1] - columns[row + 2];
        let d = columns[row] - columns[row + 3];
        result[row] = (a + b + 3) >> 3;
        result[row + 1] = (c + d + 3) >> 3;
        result[row + 2] = (a - b + 3) >> 3;
        result[row + 3] = (d - c + 3) >> 3;
    }
    result
}

fn odd_terms(first: i32, third: i32) -> (i32, i32) {
    let high = first + ((first * COS_PI_8_SQRT2_MINUS_1) >> 16);
    let low = (third * SIN_PI_8_SQRT2) >> 16;
    let outer = high + low;
    let inner = ((first * SIN_PI_8_SQRT2) >> 16) - third - ((third * COS_PI_8_SQRT2_MINUS_1) >> 16);
    (outer, inner)
}

pub(super) fn inverse_dct(input: &[i32; 16]) -> [i32; 16] {
    let mut columns = [0; 16];
    for x in 0..4 {
        let even_sum = input[x] + input[8 + x];
        let even_diff = input[x] - input[8 + x];
        let (outer, inner) = odd_terms(input[4 + x], input[12 + x]);
        columns[x] = even_sum + outer;
        columns[4 + x] = even_diff + inner;
        columns[8 + x] = even_diff - inner;
        columns[12 + x] = even_sum - outer;
    }
    let mut result = [0; 16];
    for y in 0..4 {
        let row = y * 4;
        let even_sum = columns[row] + columns[row + 2];
        let even_diff = columns[row] - columns[row + 2];
        let (outer, inner) = odd_terms(columns[row + 1], columns[row + 3]);
        result[row] = (even_sum + outer + 4) >> 3;
        result[row + 1] = (even_diff + inner + 4) >> 3;
        result[row + 2] = (even_diff - inner + 4) >> 3;
        result[row + 3] = (even_sum - outer + 4) >> 3;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_dc_only_blocks() {
        assert_eq!(inverse_walsh(&[0; 16]), [0; 16]);
        assert_eq!(inverse_dct(&[0; 16]), [0; 16]);
        let mut dc = [0; 16];
        dc[0] = 80;
        assert_eq!(inverse_walsh(&dc), [10; 16]);
        assert_eq!(inverse_dct(&dc), [10; 16]);
    }

    #[test]
    fn alternating_coefficients_have_spatial_structure() {
        let mut coefficients = [0; 16];
        coefficients[1] = 64;
        let pixels = inverse_dct(&coefficients);
        assert!(pixels[..4].windows(2).any(|pair| pair[0] != pair[1]));
        assert!(pixels.chunks(4).all(|row| row == &pixels[..4]));
    }
}
