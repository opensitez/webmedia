//! VP8 intra predictors for 16x16, 8x8, and directional 4x4 blocks.

#[derive(Clone)]
pub(super) struct Plane {
    pub(super) width: usize,
    pub(super) pixels: Vec<u8>,
}

impl Plane {
    pub(super) fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            pixels: vec![0; width * height],
        }
    }

    fn top(&self, x: usize, y: usize) -> u8 {
        if y == 0 {
            127
        } else {
            self.pixels[(y - 1) * self.width + x.min(self.width - 1)]
        }
    }

    fn left(&self, x: usize, y: usize) -> u8 {
        if x == 0 {
            129
        } else {
            self.pixels[y * self.width + x - 1]
        }
    }

    fn top_left(&self, x: usize, y: usize) -> u8 {
        if y == 0 {
            127
        } else if x == 0 {
            129
        } else {
            self.pixels[(y - 1) * self.width + x - 1]
        }
    }

    pub(super) fn predict_large(&mut self, x: usize, y: usize, size: usize, mode: u8) {
        let mut above = [0u8; 16];
        let mut left = [0u8; 16];
        for i in 0..size {
            above[i] = self.top(x + i, y);
            left[i] = self.left(x, y + i);
        }
        let dc = if x == 0 && y == 0 {
            128
        } else if y == 0 {
            (left[..size].iter().map(|&p| u32::from(p)).sum::<u32>() + (size / 2) as u32)
                / size as u32
        } else if x == 0 {
            (above[..size].iter().map(|&p| u32::from(p)).sum::<u32>() + (size / 2) as u32)
                / size as u32
        } else {
            (above[..size]
                .iter()
                .chain(left[..size].iter())
                .map(|&p| u32::from(p))
                .sum::<u32>()
                + size as u32)
                / (2 * size) as u32
        };
        let corner = self.top_left(x, y);
        match mode {
            0 => {
                for row in 0..size {
                    let start = (y + row) * self.width + x;
                    self.pixels[start..start + size].fill(dc as u8);
                }
                return;
            }
            1 => {
                for row in 0..size {
                    let start = (y + row) * self.width + x;
                    self.pixels[start..start + size].copy_from_slice(&above[..size]);
                }
                return;
            }
            2 => {
                for row in 0..size {
                    let start = (y + row) * self.width + x;
                    self.pixels[start..start + size].fill(left[row]);
                }
                return;
            }
            3 => {}
            _ => unreachable!("invalid VP8 large intra mode"),
        }
        for row in 0..size {
            for col in 0..size {
                let predicted = clamp(i32::from(left[row]) + i32::from(above[col]) - i32::from(corner));
                self.pixels[(y + row) * self.width + x + col] = predicted;
            }
        }
    }

    pub(super) fn predict_small(
        &mut self,
        x: usize,
        y: usize,
        mode: u8,
        macroblock_x: usize,
        macroblock_y: usize,
    ) {
        let mut above = [0u8; 8];
        let mut left = [0u8; 4];
        for i in 0..4 {
            left[i] = self.left(x, y + i);
        }
        for i in 0..8 {
            // The upper-right reference for right-edge subblocks comes from
            // the row above the macroblock, even on lower subblock rows.
            above[i] = if x + i >= macroblock_x + 16 {
                self.top((x + i).min(self.width - 1), macroblock_y)
            } else {
                self.top(x + i, y)
            };
        }
        let corner = self.top_left(x, y);
        let edge = [
            left[3], left[2], left[1], left[0], corner, above[0], above[1], above[2], above[3],
        ];
        let dc = ((above[..4]
            .iter()
            .chain(left.iter())
            .map(|&v| u32::from(v))
            .sum::<u32>()
            + 4)
            >> 3) as u8;
        let mut predicted = [[0u8; 4]; 4];
        for row in 0..4 {
            for col in 0..4 {
                predicted[row][col] = match mode {
                    0 => dc,
                    1 => clamp(i32::from(left[row]) + i32::from(above[col]) - i32::from(corner)),
                    2 => avg3(
                        if col == 0 { corner } else { above[col - 1] },
                        above[col],
                        above[col + 1],
                    ),
                    3 => avg3(
                        if row == 0 { corner } else { left[row - 1] },
                        left[row],
                        left[(row + 1).min(3)],
                    ),
                    4 => {
                        let center = (row + col + 1).min(7);
                        avg3(above[center - 1], above[center], above[(center + 1).min(7)])
                    }
                    5 => {
                        let center = (4 + col) - row;
                        avg3(edge[center - 1], edge[center], edge[center + 1])
                    }
                    6 => predict_vertical_right(&edge, row, col),
                    7 => predict_vertical_left(&above, row, col),
                    8 => predict_horizontal_down(&edge, row, col),
                    9 => predict_horizontal_up(&left, row, col),
                    _ => unreachable!("invalid VP8 subblock mode"),
                };
            }
        }
        for row in 0..4 {
            self.pixels[(y + row) * self.width + x..(y + row) * self.width + x + 4]
                .copy_from_slice(&predicted[row]);
        }
    }

    pub(super) fn add_residual(&mut self, x: usize, y: usize, values: &[i32; 16]) {
        if values.iter().all(|&value| value == 0) { return; }
        for row in 0..4 {
            let start = (y + row) * self.width + x;
            for (pixel, &value) in self.pixels[start..start + 4].iter_mut()
                .zip(&values[row * 4..row * 4 + 4])
            {
                *pixel = clamp(i32::from(*pixel).saturating_add(value));
            }
        }
    }
}

fn clamp(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}
fn avg2(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b) + 1) >> 1) as u8
}
fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((u16::from(a) + 2 * u16::from(b) + u16::from(c) + 2) >> 2) as u8
}

fn predict_vertical_right(edge: &[u8; 9], row: usize, col: usize) -> u8 {
    let (center, three) = match (row, col) {
        (0, c) => (4 + c, false),
        (1, c) => (4 + c, true),
        (2, 0) => (3, true),
        (2, c) => (3 + c, false),
        (3, 0) => (2, true),
        (3, c) => (3 + c, true),
        _ => unreachable!(),
    };
    if three {
        avg3(edge[center - 1], edge[center], edge[center + 1])
    } else {
        avg2(edge[center], edge[center + 1])
    }
}

fn predict_vertical_left(above: &[u8; 8], row: usize, col: usize) -> u8 {
    let (center, three) = match (row, col) {
        (0, c) => (c, false),
        (1, c) => (c + 1, true),
        (2, c) if c < 3 => (c + 1, false),
        (2, _) => (5, true),
        (3, c) if c < 3 => (c + 2, true),
        (3, _) => (6, true),
        _ => unreachable!(),
    };
    if three {
        avg3(above[center - 1], above[center], above[center + 1])
    } else {
        avg2(above[center], above[center + 1])
    }
}

fn predict_horizontal_down(edge: &[u8; 9], row: usize, col: usize) -> u8 {
    let (center, three) = match (row, col) {
        (0, 0) => (3, false),
        (0, 1) => (4, true),
        (0, 2) => (5, true),
        (0, 3) => (6, true),
        (r, c) if c % 2 == 0 => (3 - r + c / 2, false),
        (r, c) => (3 - r + (c + 1) / 2, true),
    };
    if three {
        avg3(edge[center - 1], edge[center], edge[center + 1])
    } else {
        avg2(edge[center], edge[center + 1])
    }
}

fn predict_horizontal_up(left: &[u8; 4], row: usize, col: usize) -> u8 {
    let position = 2 * row + col;
    if position >= 6 {
        return left[3];
    }
    let center = position / 2;
    if position % 2 == 0 {
        avg2(left[center], left[center + 1])
    } else {
        avg3(left[center], left[center + 1], left[(center + 2).min(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_large_prediction(plane: &mut Plane, x: usize, y: usize, size: usize, mode: u8) {
        let above: Vec<_> = (0..size).map(|col| plane.top(x + col, y)).collect();
        let left: Vec<_> = (0..size).map(|row| plane.left(x, y + row)).collect();
        let (sum, count) = match (x == 0, y == 0) {
            (true, true) => (128 * size as u32, size),
            (false, true) => (left.iter().map(|&v| u32::from(v)).sum(), size),
            (true, false) => (above.iter().map(|&v| u32::from(v)).sum(), size),
            (false, false) => (above.iter().chain(&left).map(|&v| u32::from(v)).sum(), size * 2),
        };
        let dc = (sum + (count / 2) as u32) / count as u32;
        let corner = plane.top_left(x, y);
        for row in 0..size {
            for col in 0..size {
                let value = match mode {
                    0 => dc as u8,
                    1 => above[col],
                    2 => left[row],
                    _ => clamp(i32::from(left[row]) + i32::from(above[col]) - i32::from(corner)),
                };
                plane.pixels[(y + row) * plane.width + x + col] = value;
            }
        }
    }

    #[test]
    fn bulk_large_predictions_match_scalar_edges_and_clipping() {
        for size in [8, 16] {
            for x in [0, 8, 16] {
                for y in [0, 8, 16] {
                    for pattern in 0..4 {
                        let mut original = Plane::new(32, 32);
                        for (index, pixel) in original.pixels.iter_mut().enumerate() {
                            *pixel = match pattern {
                                0 => 0,
                                1 => 255,
                                2 => if index % 2 == 0 { 0 } else { 255 },
                                _ => ((index * 37 + 19) & 255) as u8,
                            };
                        }
                        for mode in 0..4 {
                            let mut actual = original.clone();
                            let mut expected = original.clone();
                            actual.predict_large(x, y, size, mode);
                            scalar_large_prediction(&mut expected, x, y, size, mode);
                            assert_eq!(actual.pixels, expected.pixels,
                                "size={size} x={x} y={y} mode={mode} pattern={pattern}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn default_edges_and_dc_modes() {
        let mut plane = Plane::new(32, 32);
        plane.predict_large(0, 0, 16, 0);
        assert!(plane.pixels[..16].iter().all(|&pixel| pixel == 128));
        plane.predict_small(0, 0, 0, 0, 0);
        assert!(plane.pixels[..4].iter().all(|&pixel| pixel == 128));
    }

    #[test]
    fn bulk_residual_matches_wide_clipped_addition() {
        let mut original = Plane::new(13, 9);
        for (index, pixel) in original.pixels.iter_mut().enumerate() {
            *pixel = (index * 73) as u8;
        }
        let extremes = [i32::MIN, -32768, -256, -255, -1, 0, 1, 255, 256, 32767, i32::MAX];
        for trial in 0..1024 {
            let values = std::array::from_fn(|index| {
                if trial < extremes.len() { extremes[(trial + index) % extremes.len()] }
                else { ((trial * 37 + index * 73) % 1025) as i32 - 512 }
            });
            let (x, y) = (trial % 10, trial % 6);
            let mut actual = original.clone();
            let mut expected = original.clone();
            actual.add_residual(x, y, &values);
            for row in 0..4 {
                for col in 0..4 {
                    let at = (y + row) * expected.width + x + col;
                    expected.pixels[at] = (i64::from(expected.pixels[at])
                        + i64::from(values[row * 4 + col])).clamp(0, 255) as u8;
                }
            }
            assert_eq!(actual.pixels, expected.pixels, "trial={trial}");
        }
        let mut actual = original.clone();
        actual.add_residual(9, 5, &[0; 16]);
        assert_eq!(actual.pixels, original.pixels);
    }

    #[test]
    #[ignore = "manual VP8 residual addition kernel timing"]
    fn benchmark_residual_addition() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut plane = Plane::new(13, 9);
        let values = std::array::from_fn(|index| (index * 7 % 13) as i32 - 6);
        for zero_every in [0, 4] {
            for trial in 0..5 {
                for bulk in if trial % 2 == 0 { [false, true] } else { [true, false] } {
                    let start = Instant::now();
                    for iteration in 0..1_000_000 {
                        let plane = black_box(&mut plane);
                        let values = black_box(if zero_every != 0 && iteration % zero_every == 0 {
                            &[0; 16]
                        } else { &values });
                        if bulk { plane.add_residual(3, 2, values); }
                        else {
                            for row in 0..4 {
                                for col in 0..4 {
                                    let at = (2 + row) * plane.width + 3 + col;
                                    plane.pixels[at] = clamp(i32::from(plane.pixels[at]) + values[row * 4 + col]);
                                }
                            }
                        }
                        black_box(&plane.pixels);
                    }
                    eprintln!("VP8 residual zero_every={zero_every} trial={trial} bulk={bulk} elapsed={:?}", start.elapsed());
                }
            }
        }
    }
}
