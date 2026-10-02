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
        for row in 0..size {
            for col in 0..size {
                let predicted = match mode {
                    0 => dc as u8,
                    1 => above[col],
                    2 => left[row],
                    3 => clamp(i32::from(left[row]) + i32::from(above[col]) - i32::from(corner)),
                    _ => unreachable!("invalid VP8 large intra mode"),
                };
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
        for row in 0..4 {
            for col in 0..4 {
                let at = (y + row) * self.width + x + col;
                self.pixels[at] = clamp(i32::from(self.pixels[at]) + values[row * 4 + col]);
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

    #[test]
    fn default_edges_and_dc_modes() {
        let mut plane = Plane::new(32, 32);
        plane.predict_large(0, 0, 16, 0);
        assert!(plane.pixels[..16].iter().all(|&pixel| pixel == 128));
        plane.predict_small(0, 0, 0, 0, 0);
        assert!(plane.pixels[..4].iter().all(|&pixel| pixel == 128));
    }
}
