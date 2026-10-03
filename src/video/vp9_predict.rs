//! Eight-bit VP9 intra prediction (specification section 8.5.1).

use super::backend::MediaDecodeError;
use super::vp8_predict::Plane;

fn avg2(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b) + 1) >> 1) as u8
}

fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((u16::from(a) + 2 * u16::from(b) + u16::from(c) + 2) >> 2) as u8
}

pub(super) fn predict(
    plane: &mut Plane,
    x: usize,
    y: usize,
    size: usize,
    mode: u8,
    have_left: bool,
    have_above: bool,
    not_on_right: bool,
    visible_width: usize,
    visible_height: usize,
) -> Result<(), MediaDecodeError> {
    if plane.width == 0 || !matches!(size, 4 | 8 | 16 | 32) || mode > 9
        || visible_width == 0 || visible_height == 0
        || visible_width > plane.width || visible_height > plane.pixels.len() / plane.width
        || (have_left && x == 0) || (have_above && y == 0)
        || x.checked_add(size).is_none_or(|end| end > plane.width)
        || y.checked_add(size).is_none_or(|end| end > plane.pixels.len() / plane.width)
    {
        return Err(MediaDecodeError::InvalidData("invalid VP9 prediction block".into()));
    }
    let mut above_storage = [127u8; 65];
    let above = &mut above_storage[..size * 2 + 1];
    let mut left_storage = [129u8; 32];
    let left = &mut left_storage[..size];
    if have_above {
        for (i, sample) in above[1..=size].iter_mut().enumerate() {
            *sample = plane.pixels[(y - 1) * plane.width + (x + i).min(visible_width - 1)];
        }
        for i in size..size * 2 {
            above[i + 1] = if not_on_right && size == 4 {
                plane.pixels[(y - 1) * plane.width + (x + i).min(visible_width - 1)]
            } else {
                above[size]
            };
        }
        above[0] = if have_left {
            plane.pixels[(y - 1) * plane.width + (x - 1).min(visible_width - 1)]
        } else {
            129
        };
    }
    if have_left {
        for (i, sample) in left.iter_mut().enumerate() {
            *sample = plane.pixels[(y + i).min(visible_height - 1) * plane.width + x - 1];
        }
    }
    let a = |i: isize| above[(i + 1) as usize];
    let mut out_storage = [0u8; 1024];
    let out = &mut out_storage[..size * size];
    let at = |row: usize, col: usize| row * size + col;
    match mode {
        0 => {
            let (sum, count) = match (have_left, have_above) {
                (true, true) => (
                    left.iter().chain(&above[1..=size]).map(|&v| u32::from(v)).sum::<u32>(),
                    size * 2,
                ),
                (true, false) => (left.iter().map(|&v| u32::from(v)).sum(), size),
                (false, true) => (above[1..=size].iter().map(|&v| u32::from(v)).sum(), size),
                (false, false) => (128 * size as u32, size),
            };
            out.fill(((sum + (count / 2) as u32) / count as u32) as u8);
        }
        1 => {
            for row in 0..size {
                out[row * size..(row + 1) * size].copy_from_slice(&above[1..=size]);
            }
        }
        2 => {
            for row in 0..size {
                out[row * size..(row + 1) * size].fill(left[row]);
            }
        }
        3 => {
            for row in 0..size {
                for col in 0..size {
                    let i = row + col;
                    out[at(row, col)] = if i + 2 < 2 * size {
                        avg3(above[i + 1], above[i + 2], above[i + 3])
                    } else {
                        above[2 * size]
                    };
                }
            }
        }
        4 => {
            out[at(0, 0)] = avg3(left[0], a(-1), a(0));
            for col in 1..size {
                out[at(0, col)] = avg3(a(col as isize - 2), a(col as isize - 1), a(col as isize));
            }
            out[at(1, 0)] = avg3(a(-1), left[0], left[1]);
            for row in 2..size {
                out[at(row, 0)] = avg3(left[row - 2], left[row - 1], left[row]);
            }
            for row in 1..size {
                for col in 1..size {
                    out[at(row, col)] = out[at(row - 1, col - 1)];
                }
            }
        }
        5 => {
            for col in 0..size {
                out[at(0, col)] = avg2(a(col as isize - 1), a(col as isize));
            }
            out[at(1, 0)] = avg3(left[0], a(-1), a(0));
            for col in 1..size {
                out[at(1, col)] = avg3(a(col as isize - 2), a(col as isize - 1), a(col as isize));
            }
            out[at(2, 0)] = avg3(a(-1), left[0], left[1]);
            for row in 3..size {
                out[at(row, 0)] = avg3(left[row - 3], left[row - 2], left[row - 1]);
            }
            for row in 2..size {
                for col in 1..size {
                    out[at(row, col)] = out[at(row - 2, col - 1)];
                }
            }
        }
        6 => {
            out[at(0, 0)] = avg2(left[0], a(-1));
            for row in 1..size {
                out[at(row, 0)] = avg2(left[row - 1], left[row]);
            }
            out[at(0, 1)] = avg3(left[0], a(-1), a(0));
            out[at(1, 1)] = avg3(a(-1), left[0], left[1]);
            for row in 2..size {
                out[at(row, 1)] = avg3(left[row - 2], left[row - 1], left[row]);
            }
            for col in 2..size {
                out[at(0, col)] = avg3(a(col as isize - 3), a(col as isize - 2), a(col as isize - 1));
                for row in 1..size {
                    out[at(row, col)] = out[at(row - 1, col - 2)];
                }
            }
        }
        7 => {
            for col in 0..size {
                for row in 0..size {
                    out[at(row, col)] = if row == size - 1 {
                        left[size - 1]
                    } else if col == 0 {
                        avg2(left[row], left[row + 1])
                    } else if col == 1 {
                        avg3(left[row], left[row + 1], left[(row + 2).min(size - 1)])
                    } else {
                        out[at(row + 1, col - 2)]
                    };
                }
            }
        }
        8 => {
            for row in 0..size {
                for col in 0..size {
                    let k = row / 2 + col;
                    out[at(row, col)] = if row & 1 == 0 {
                        avg2(above[k + 1], above[k + 2])
                    } else {
                        avg3(above[k + 1], above[k + 2], above[k + 3])
                    };
                }
            }
        }
        9 => {
            for row in 0..size {
                for col in 0..size {
                    out[at(row, col)] = (i32::from(above[col + 1]) + i32::from(left[row])
                        - i32::from(a(-1))).clamp(0, 255) as u8;
                }
            }
        }
        _ => unreachable!(),
    }
    for row in 0..size {
        plane.pixels[(y + row) * plane.width + x..(y + row) * plane.width + x + size]
            .copy_from_slice(&out[row * size..(row + 1) * size]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_specified_missing_edge_values() {
        for (mode, value) in [(0, 128), (1, 127), (2, 129), (9, 129)] {
            let mut plane = Plane::new(4, 4);
            predict(&mut plane, 0, 0, 4, mode, false, false, false, 4, 4).unwrap();
            assert!(plane.pixels.iter().all(|pixel| *pixel == value), "mode {mode}");
        }
    }

    #[test]
    fn reconstructs_dc_and_directional_intra_samples() {
        let mut original = Plane::new(8, 8);
        original.pixels[0] = 5;
        original.pixels[1..5].copy_from_slice(&[10, 20, 30, 40]);
        for (row, value) in [60, 70, 80, 90].into_iter().enumerate() {
            original.pixels[(row + 1) * 8] = value;
        }
        for (mode, expected) in [
            (0, 50), (1, 10), (2, 60), (3, 20), (4, 20),
            (5, 8), (6, 33), (7, 65), (8, 15), (9, 65),
        ] {
            let mut plane = original.clone();
            predict(&mut plane, 1, 1, 4, mode, true, true, false, 8, 8).unwrap();
            assert_eq!(plane.pixels[9], expected, "mode {mode}");
        }
    }

    #[test]
    fn predicts_all_modes_at_every_transform_size() {
        for size in [4, 8, 16, 32] {
            for mode in 0..10 {
                let mut plane = Plane::new(size, size);
                predict(&mut plane, 0, 0, size, mode, false, false, false, size, size).unwrap();
            }
        }
    }

    #[test]
    fn extends_above_right_only_for_four_by_four_blocks() {
        let mut extended = Plane::new(12, 8);
        extended.pixels[1..9].copy_from_slice(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let mut repeated = extended.clone();
        predict(&mut extended, 1, 1, 4, 3, true, true, true, 12, 8).unwrap();
        predict(&mut repeated, 1, 1, 4, 3, true, true, false, 12, 8).unwrap();
        assert_eq!(extended.pixels[1 * 12 + 4], 50);
        assert_eq!(repeated.pixels[1 * 12 + 4], 40);
    }

    #[test]
    fn rejects_invalid_bounds_without_panicking() {
        let mut plane = Plane::new(4, 4);
        assert!(predict(&mut plane, 0, 0, 4, 1, false, true, false, 4, 4).is_err());
        assert!(predict(&mut plane, 0, 0, 4, 2, true, false, false, 4, 4).is_err());
        assert!(predict(&mut plane, 0, 0, 4, 0, false, false, false, 5, 4).is_err());
    }
}
