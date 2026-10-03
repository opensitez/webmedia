//! VP8 fractional-pixel inter prediction (RFC 6386, section 18).

use super::vp8_inter::MotionVector;
use super::vp8_predict::Plane;
use super::subpel::convolve_row;
#[cfg(test)]
use super::subpel::convolve_row_scalar;

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

struct ActiveFilter {
    taps: [(usize, i32); 6],
    count: usize,
    first_tap: usize,
    last_tap: usize,
}

const fn active_filters(filters: &[[i32; 6]; 8]) -> [ActiveFilter; 8] {
    let mut result = [const { ActiveFilter {
        taps: [(0, 0); 6], count: 0, first_tap: 0, last_tap: 0,
    } }; 8];
    let mut phase = 0;
    while phase < 8 {
        let mut tap = 0;
        while tap < 6 {
            if filters[phase][tap] != 0 {
                let count = result[phase].count;
                if count == 0 { result[phase].first_tap = tap; }
                result[phase].last_tap = tap;
                result[phase].taps[count] = (tap, filters[phase][tap]);
                result[phase].count += 1;
            }
            tap += 1;
        }
        phase += 1;
    }
    result
}

const ACTIVE_BILINEAR: [ActiveFilter; 8] = active_filters(&BILINEAR);
const ACTIVE_SIX_TAP: [ActiveFilter; 8] = active_filters(&SIX_TAP);

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
    let filters = if bilinear { &ACTIVE_BILINEAR } else { &ACTIVE_SIX_TAP };
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
    let active_h = &hfilter.taps[..hfilter.count];
    let active_v = &vfilter.taps[..vfilter.count];
    debug_assert!(width <= 16 && height <= 16);
    if hfrac == 0 && vfrac != 0 && origin_x >= 0 && origin_y >= 2
        && origin_x as usize + width <= visible_width
        && origin_y as usize + height + 3 <= visible_height
    {
        for row in 0..height {
            let source_start = (origin_y as usize + row - 2) * reference.width + origin_x as usize;
            let destination_start = (y + row) * destination.width + x;
            convolve_row(&reference.pixels, source_start, reference.width, active_v,
                &mut destination.pixels[destination_start..destination_start + width]);
        }
        return;
    }
    let mut intermediate = [0u8; 16 * 21];
    let first_row = if vfrac == 0 { 0 } else { vfilter.first_tap };
    let rows = height + if vfrac == 0 { 0 } else { vfilter.last_tap };
    let contiguous = origin_x >= 2 && origin_x as usize + width + 3 <= visible_width;
    let mut columns = [[0usize; 6]; 16];
    if !contiguous {
        for col in 0..width {
            for tap in 0..6 {
                columns[col][tap] = (origin_x + col as i32 + tap as i32 - 2)
                    .clamp(0, visible_width as i32 - 1) as usize;
            }
        }
    }
    // Intermediate rows outside the active vertical taps are never consumed.
    for row in first_row..rows {
        let py = origin_y + row as i32 - if vfrac == 0 { 0 } else { 2 };
        let source_y = py.clamp(0, visible_height as i32 - 1) as usize;
        let source = &reference.pixels[source_y * reference.width..][..visible_width];
        let target = if vfrac == 0 {
            let start = (y + row) * destination.width + x;
            &mut destination.pixels[start..start + width]
        } else {
            &mut intermediate[row * width..(row + 1) * width]
        };
        if contiguous {
            if hfrac == 0 {
                target.copy_from_slice(&source[origin_x as usize..origin_x as usize + width]);
            } else {
                convolve_row(source, origin_x as usize - 2, 1, active_h, target);
            }
        } else {
            for (col, sample) in target.iter_mut().enumerate() {
                *sample = if hfrac == 0 { source[columns[col][2]] } else {
                    let sum: i32 = active_h.iter()
                        .map(|&(tap, coefficient)| i32::from(source[columns[col][tap]]) * coefficient)
                        .sum();
                    ((sum + 64) >> 7).clamp(0, 255) as u8
                };
            }
        }
    }
    if vfrac == 0 {
        return;
    }
    for row in 0..height {
        let target_start = (y + row) * destination.width + x;
        convolve_row(&intermediate, row * width, width, active_v,
            &mut destination.pixels[target_start..target_start + width]);
    }
}

#[cfg(test)]
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
    fn cached_filter_taps_match_coefficients() {
        for (filters, active) in [(&BILINEAR, &ACTIVE_BILINEAR), (&SIX_TAP, &ACTIVE_SIX_TAP)] {
            for phase in 0..8 {
                let expected: Vec<_> = filters[phase].iter().enumerate()
                    .filter(|(_, coefficient)| **coefficient != 0)
                    .map(|(tap, &coefficient)| (tap, coefficient)).collect();
                assert_eq!(&active[phase].taps[..active[phase].count], expected);
                assert_eq!(active[phase].first_tap, expected.first().unwrap().0);
                assert_eq!(active[phase].last_tap, expected.last().unwrap().0);
            }
        }
    }

    #[test]
    fn vertical_prediction_respects_visible_edges_and_destination_bounds() {
        let mut reference = Plane::new(32, 32);
        for (index, pixel) in reference.pixels.iter_mut().enumerate() {
            *pixel = (index.wrapping_mul(73) ^ (index / 32)) as u8;
        }
        for size in [4, 8, 16] {
            for bilinear in [false, true] {
                for chroma in [false, true] {
                    for (x, y) in [(0, 0), (0, 4), (27 - size, 4), (3, 25 - size)] {
                        for row in -8..8 {
                            let mut actual = Plane::new(40, 40);
                            actual.pixels.fill(91);
                            let mut expected = actual.clone();
                            predict_block(&mut actual, &reference, x, y, size, size,
                                MotionVector { row, col: 0 }, chroma, bilinear, 27, 25);
                            let eighths = i32::from(row) * if chroma { 1 } else { 2 };
                            let filters = if bilinear { &BILINEAR } else { &SIX_TAP };
                            for dy in 0..size {
                                for dx in 0..size {
                                    let py = (y + dy) as i32 + eighths.div_euclid(8);
                                    let sum: i32 = filters[eighths.rem_euclid(8) as usize].iter().enumerate()
                                        .map(|(tap, &coefficient)| {
                                            let sy = (py + tap as i32 - 2).clamp(0, 24) as usize;
                                            i32::from(reference.pixels[sy * 32 + x + dx]) * coefficient
                                        }).sum();
                                    expected.pixels[(y + dy) * 40 + x + dx] =
                                        ((sum + 64) >> 7).clamp(0, 255) as u8;
                                }
                            }
                            assert_eq!(actual.pixels, expected.pixels,
                                "size={size} bilinear={bilinear} chroma={chroma} x={x} y={y} row={row}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn bulk_rows_match_portable_rounding_and_clipping() {
        for width in [4, 8, 16] {
            for stride in [1, 19] {
                for origin in [0, 1, 7] {
                    for pattern in 0..4 {
                        let source: Vec<_> = (0usize..160).map(|index| match pattern {
                            0 => 0, 1 => 255, 2 => if index % 2 == 0 { 0 } else { 255 },
                            _ => index.wrapping_mul(73) as u8,
                        }).collect();
                        for filter in BILINEAR.iter().chain(&SIX_TAP) {
                            let taps: Vec<_> = filter.iter().enumerate()
                                .filter(|(_, coefficient)| **coefficient != 0)
                                .map(|(tap, &coefficient)| (tap, coefficient)).collect();
                            let mut expected = vec![0; width];
                            let mut actual = vec![19; width + 8];
                            convolve_row_scalar(&source, origin, stride, &taps, &mut expected);
                            convolve_row(&source, origin, stride, &taps, &mut actual[4..4 + width]);
                            assert_eq!(&actual[4..4 + width], expected);
                            assert!(actual[..4].iter().chain(&actual[4 + width..]).all(|&pixel| pixel == 19));
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "manual release-mode separable prediction benchmark"]
    fn benchmark_separable_prediction() {
        let mut reference = Plane::new(32, 32);
        for (index, pixel) in reference.pixels.iter_mut().enumerate() {
            *pixel = index.wrapping_mul(73) as u8;
        }
        let mut output = Plane::new(16, 16);
        let mv = MotionVector { row: 5, col: 7 };
        for _ in 0..3 {
            let start = std::time::Instant::now();
            for _ in 0..5000 {
                let reference = std::hint::black_box(&reference);
                let output = std::hint::black_box(&mut output);
                for row in 0..16 {
                    for col in 0..16 {
                        let value: i32 = SIX_TAP[5].iter().enumerate().map(|(tap, &coefficient)| {
                            i32::from(horizontal(reference, col as i32, row as i32 + tap as i32 - 2,
                                &SIX_TAP[7], 32, 32)) * coefficient
                        }).sum();
                        output.pixels[row * 16 + col] = ((value + 64) >> 7).clamp(0, 255) as u8;
                    }
                }
                std::hint::black_box(&output.pixels);
            }
            let scalar = start.elapsed();
            let start = std::time::Instant::now();
            for _ in 0..5000 {
                predict_block(std::hint::black_box(&mut output), std::hint::black_box(&reference),
                    0, 0, 16, 16, std::hint::black_box(mv), true, false, 32, 32);
                std::hint::black_box(&output.pixels);
            }
            eprintln!("VP8 prediction: scalar={scalar:?} separable={:?}", start.elapsed());
        }
    }

    #[test]
    fn separable_prediction_matches_scalar_at_all_phases_and_edges() {
        let mut reference = Plane::new(32, 32);
        for (index, sample) in reference.pixels.iter_mut().enumerate() {
            *sample = (index.wrapping_mul(73) ^ (index >> 3)) as u8;
        }
        for size in [4, 8, 16] {
            for bilinear in [false, true] {
                for chroma in [false, true] {
                    for (start_x, start_y) in [(0, 0), (8, 8), (25, 25)] {
                        for row in -8..8 {
                            for col in -8..8 {
                                let mut actual = Plane::new(48, 48);
                                let mv = MotionVector { row, col };
                                predict_block(&mut actual, &reference, start_x, start_y, size, size,
                                    mv, chroma, bilinear, 32, 32);
                                let factor = if chroma { 1 } else { 2 };
                                let row = i32::from(row) * factor;
                                let col = i32::from(col) * factor;
                                let filters = if bilinear { &BILINEAR } else { &SIX_TAP };
                                let hfilter = &filters[col.rem_euclid(8) as usize];
                                let vfilter = &filters[row.rem_euclid(8) as usize];
                                for y in 0..size {
                                    for x in 0..size {
                                        let px = (start_x + x) as i32 + col.div_euclid(8);
                                        let py = (start_y + y) as i32 + row.div_euclid(8);
                                        let value: i32 = vfilter.iter().enumerate().map(|(tap, &coefficient)| {
                                            i32::from(horizontal(&reference, px, py + tap as i32 - 2,
                                                hfilter, 32, 32)) * coefficient
                                        }).sum();
                                        assert_eq!(actual.pixels[(start_y + y) * actual.width + start_x + x],
                                            ((value + 64) >> 7).clamp(0, 255) as u8);
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
