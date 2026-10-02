//! VP8 fractional-pixel inter prediction (RFC 6386, section 18).

use super::vp8_inter::MotionVector;
use super::vp8_predict::Plane;

const BILINEAR: [[i32; 6]; 8] = [
    [0, 0, 128, 0, 0, 0],
    [0, 0, 112, 16, 0, 0],
    [0, 0, 96, 32, 0, 0],
    [0, 0, 80, 48, 0, 0],
    [0, 0, 64, 64, 0, 0],
    [0, 0, 48, 80, 0, 0],
    [0, 0, 32, 96, 0, 0],
    [0, 0, 16, 112, 0, 0],
];
const SIX_TAP: [[i32; 6]; 8] = [
    [0, 0, 128, 0, 0, 0],
    [0, -6, 123, 12, -1, 0],
    [2, -11, 108, 36, -8, 1],
    [0, -9, 93, 50, -6, 0],
    [3, -16, 77, 77, -16, 3],
    [0, -6, 50, 93, -9, 0],
    [1, -8, 36, 108, -11, 2],
    [0, -1, 12, 123, -6, 0],
];

pub(super) fn predict_block(
    destination: &mut Plane,
    reference: &Plane,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    mv: MotionVector,
    chroma: bool,
    bilinear: bool,
    visible_width: usize,
    visible_height: usize,
) {
    let filters = if bilinear { &BILINEAR } else { &SIX_TAP };
    let row_eighths = i32::from(mv.row) * if chroma { 1 } else { 2 };
    let col_eighths = i32::from(mv.col) * if chroma { 1 } else { 2 };
    let origin_x = x as i32 + col_eighths.div_euclid(8);
    let origin_y = y as i32 + row_eighths.div_euclid(8);
    let hfrac = col_eighths.rem_euclid(8) as usize;
    let vfrac = row_eighths.rem_euclid(8) as usize;
    if hfrac == 0
        && vfrac == 0
        && origin_x >= 0
        && origin_y >= 0
        && origin_x as usize + width <= visible_width
        && origin_y as usize + height <= visible_height
    {
        let source_x = origin_x as usize;
        let source_y = origin_y as usize;
        for row in 0..height {
            let source_start = (source_y + row) * reference.width + source_x;
            let destination_start = (y + row) * destination.width + x;
            destination.pixels[destination_start..destination_start + width]
                .copy_from_slice(&reference.pixels[source_start..source_start + width]);
        }
        return;
    }
    let hfilter = &filters[hfrac];
    let vfilter = &filters[vfrac];
    for row in 0..height {
        for col in 0..width {
            let px = origin_x + col as i32;
            let py = origin_y + row as i32;
            let sample = if vfrac == 0 {
                horizontal(reference, px, py, hfilter, visible_width, visible_height)
            } else {
                let mut value = 0;
                for tap in 0..6 {
                    let horizontal = horizontal(
                        reference,
                        px,
                        py + tap as i32 - 2,
                        hfilter,
                        visible_width,
                        visible_height,
                    );
                    value += i32::from(horizontal) * vfilter[tap];
                }
                ((value + 64) >> 7).clamp(0, 255) as u8
            };
            destination.pixels[(y + row) * destination.width + x + col] = sample;
        }
    }
}

fn horizontal(plane: &Plane, x: i32, y: i32, filter: &[i32; 6], width: usize, height: usize) -> u8 {
    let row = y.clamp(0, height as i32 - 1) as usize;
    let mut value = 0;
    for (tap, coefficient) in filter.iter().enumerate() {
        let col = (x + tap as i32 - 2).clamp(0, width as i32 - 1) as usize;
        value += i32::from(plane.pixels[row * plane.width + col]) * coefficient;
    }
    ((value + 64) >> 7).clamp(0, 255) as u8
}

pub(super) fn chroma_vector(luma: &[MotionVector; 16], row: usize, col: usize) -> MotionVector {
    let indices = [
        row * 8 + col * 2,
        row * 8 + col * 2 + 1,
        row * 8 + col * 2 + 4,
        row * 8 + col * 2 + 5,
    ];
    let average = |component: fn(MotionVector) -> i16| {
        let sum: i32 = indices
            .iter()
            .map(|&index| i32::from(component(luma[index])))
            .sum();
        if sum >= 0 {
            ((sum + 2) >> 2) as i16
        } else {
            -(((-sum + 2) >> 2) as i16)
        }
    };
    MotionVector {
        row: average(|mv| mv.row),
        col: average(|mv| mv.col),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_pixel_and_fractional_prediction() {
        let mut reference = Plane::new(16, 16);
        for row in 0..16 {
            for col in 0..16 {
                reference.pixels[row * 16 + col] = (col * 10 + row) as u8;
            }
        }
        let mut output = Plane::new(16, 16);
        predict_block(
            &mut output,
            &reference,
            4,
            4,
            4,
            4,
            MotionVector::default(),
            false,
            true,
            16,
            16,
        );
        assert_eq!(output.pixels[4 * 16 + 4], reference.pixels[4 * 16 + 4]);
        predict_block(
            &mut output,
            &reference,
            4,
            4,
            4,
            4,
            MotionVector { row: 0, col: 2 },
            false,
            true,
            16,
            16,
        );
        assert_eq!(output.pixels[4 * 16 + 4], 49);
    }

    #[test]
    fn chroma_vector_preserves_whole_macroblock_displacement() {
        let luma = [MotionVector { row: -5, col: 7 }; 16];
        assert_eq!(chroma_vector(&luma, 0, 0), MotionVector { row: -5, col: 7 });
        assert_eq!(chroma_vector(&luma, 1, 1), MotionVector { row: -5, col: 7 });
    }
}
