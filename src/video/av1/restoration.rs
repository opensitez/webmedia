//! Wiener restoration units and stripe filtering, AV1 5.11.57-58 and 7.17.4-6.

use super::decoder::DecodedPlane;
use super::entropy::SymbolDecoder;
use super::syntax::{Error, IntraFrameHeader, SequenceHeader};

fn unit_count(size: usize, extent: usize) -> usize {
    ((extent + size / 2) / size).max(1)
}

fn signed_subexp(
    d: &mut SymbolDecoder<'_>,
    low: i32,
    high: i32,
    k: u8,
    reference: i32,
) -> Result<i32, Error> {
    let n = (high - low) as u32;
    let r = (reference - low) as u32;
    if r >= n {
        return Err(Error::Invalid("restoration coefficient reference"));
    }
    let (mut i, mut base) = (0u8, 0u32);
    let value = loop {
        let bits = if i == 0 { k } else { k + i - 1 };
        let range = 1 << bits;
        if n <= base + 3 * range {
            let remaining = n - base;
            let width = (32 - remaining.leading_zeros()) as u8;
            let m = (1 << width) - remaining;
            let v = if remaining == 1 {
                0
            } else {
                d.read_literal(width - 1)?
            };
            let v = if v < m {
                v
            } else {
                (v << 1) - m + u32::from(d.read_bool()?)
            };
            break base + v;
        }
        if !d.read_bool()? {
            break base + d.read_literal(bits)?;
        }
        i += 1;
        base += range;
    };
    let recenter = |r: u32, v: u32| {
        if v > 2 * r {
            v
        } else if v & 1 != 0 {
            r - v.div_ceil(2)
        } else {
            r + v / 2
        }
    };
    let v = if 2 * r <= n {
        recenter(r, value)
    } else {
        n - 1 - recenter(n - 1 - r, value)
    };
    Ok(low + v as i32)
}

pub(crate) struct Restoration {
    units: Vec<Vec<Option<[[i32; 3]; 2]>>>,
    dimensions: Vec<(usize, usize)>,
    previous: [[[i32; 3]; 2]; 3],
}

impl Restoration {
    pub(crate) fn new(s: &SequenceHeader, h: &IntraFrameHeader) -> Result<Self, Error> {
        if h.restoration_types.iter().any(|&r| r != 0 && r != 2) {
            return Err(Error::Unsupported("self-guided or switchable restoration"));
        }
        let mut dimensions = Vec::new();
        let mut units = Vec::new();
        for (plane, &kind) in h.restoration_types.iter().enumerate() {
            if kind == 0 {
                dimensions.push((0, 0));
                units.push(Vec::new());
                continue;
            }
            let sx = usize::from(plane > 0 && s.subsampling_x);
            let sy = usize::from(plane > 0 && s.subsampling_y);
            let size = h.restoration_unit_sizes[plane] as usize;
            if ![64, 128, 256].contains(&size) {
                return Err(Error::Invalid("restoration unit size"));
            }
            let cols = unit_count(size, ((h.upscaled_width as usize) + (1 << sx) / 2) >> sx);
            let rows = unit_count(size, ((h.height as usize) + (1 << sy) / 2) >> sy);
            dimensions.push((cols, rows));
            units.push(vec![None; cols * rows]);
        }
        Ok(Self {
            units,
            dimensions,
            previous: [[[3, -7, 15]; 2]; 3],
        })
    }

    pub(crate) fn read(
        &mut self,
        d: &mut SymbolDecoder<'_>,
        cdf: &mut [u16],
        s: &SequenceHeader,
        h: &IntraFrameHeader,
        row: usize,
        col: usize,
    ) -> Result<(), Error> {
        for plane in 0..self.units.len() {
            if h.restoration_types[plane] == 0 {
                continue;
            }
            let sx = usize::from(plane > 0 && s.subsampling_x);
            let sy = usize::from(plane > 0 && s.subsampling_y);
            let size = h.restoration_unit_sizes[plane] as usize;
            let (cols, rows) = self.dimensions[plane];
            let r0 = (row * (4 >> sy)).div_ceil(size);
            let r1 = ((row + 16) * (4 >> sy)).div_ceil(size).min(rows);
            let c0 = (col * (4 >> sx)).div_ceil(size);
            let c1 = ((col + 16) * (4 >> sx)).div_ceil(size).min(cols);
            for r in r0..r1 {
                for c in c0..c1 {
                    if d.read_symbol(cdf)? == 0 {
                        continue;
                    }
                    let mut coefficients = [[0; 3]; 2];
                    for pass in 0..2 {
                        for tap in usize::from(plane > 0)..3 {
                            let value = signed_subexp(
                                d,
                                [-5, -23, -17][tap],
                                [11, 9, 47][tap],
                                [1, 2, 3][tap],
                                self.previous[plane][pass][tap],
                            )?;
                            coefficients[pass][tap] = value;
                            self.previous[plane][pass][tap] = value;
                        }
                    }
                    self.units[plane][r * cols + c] = Some(coefficients);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn apply(
        &self,
        planes: &mut [DecodedPlane],
        deblocked: &[DecodedPlane],
        s: &SequenceHeader,
        h: &IntraFrameHeader,
    ) -> Result<(), Error> {
        let cdef = planes.to_vec();
        for (plane, out) in planes.iter_mut().enumerate() {
            if h.restoration_types[plane] == 0 {
                continue;
            }
            let sx = usize::from(plane > 0 && s.subsampling_x);
            let sy = usize::from(plane > 0 && s.subsampling_y);
            let width = ((h.upscaled_width as usize) + (1 << sx) / 2) >> sx;
            let height = ((h.height as usize) + (1 << sy) / 2) >> sy;
            let size = h.restoration_unit_sizes[plane] as usize;
            let (cols, rows) = self.dimensions[plane];
            for y in (0..height).step_by(4 >> sy) {
                for x in (0..width).step_by(4 >> sx) {
                    let stripe = ((y << sy) + 8) / 64;
                    let start = (-8 + stripe as i32 * 64) >> sy;
                    let end = start + (64 >> sy) - 1;
                    let unit_y = ((y + (8 >> sy)) / size).min(rows - 1);
                    let unit_x = (x / size).min(cols - 1);
                    let Some(coeff) = self.units[plane][unit_y * cols + unit_x] else {
                        continue;
                    };
                    let source = |xx: i32, yy: i32| -> i32 {
                        let xx = xx.clamp(0, width as i32 - 1) as usize;
                        let yy = yy.clamp(0, height as i32 - 1);
                        let (p, yy) = if yy < start {
                            (&deblocked[plane], yy.max(start - 2))
                        } else if yy > end {
                            (&deblocked[plane], yy.min(end + 2))
                        } else {
                            (&cdef[plane], yy)
                        };
                        i32::from(p.samples[yy as usize * p.stride + xx])
                    };
                    let make_filter = |c: [i32; 3]| {
                        [
                            c[0],
                            c[1],
                            c[2],
                            128 - 2 * (c[0] + c[1] + c[2]),
                            c[2],
                            c[1],
                            c[0],
                        ]
                    };
                    let vertical = make_filter(coeff[0]);
                    let horizontal = make_filter(coeff[1]);
                    let w = (4 >> sx).min(width - x);
                    let ht = (4 >> sy).min(height - y);
                    let round0 = if s.bit_depth == 12 { 5 } else { 3 };
                    let round1 = 14 - round0;
                    let offset = 1 << (s.bit_depth as u32 + 7 - round0 - 1);
                    let limit = (1 << (s.bit_depth as u32 + 1 + 7 - round0)) - 1;
                    let mut intermediate = vec![0; w * (ht + 6)];
                    for r in 0..ht + 6 {
                        for c in 0..w {
                            let sum = (0..7)
                                .map(|tap| {
                                    horizontal[tap]
                                        * source((x + c + tap) as i32 - 3, (y + r) as i32 - 3)
                                })
                                .sum::<i32>();
                            intermediate[r * w + c] = ((sum + (1 << (round0 - 1))) >> round0)
                                .clamp(-offset, limit - offset);
                        }
                    }
                    for r in 0..ht {
                        for c in 0..w {
                            let sum = (0..7)
                                .map(|tap| vertical[tap] * intermediate[(r + tap) * w + c])
                                .sum::<i32>();
                            out.samples[(y + r) * out.stride + x + c] =
                                ((sum + (1 << (round1 - 1))) >> round1)
                                    .clamp(0, (1 << s.bit_depth) - 1)
                                    as u16;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn symmetric_wiener_coefficients_preserve_dc_gain() {
        for a in -5..=10 {
            for b in -23..=8 {
                for c in -17..=46 {
                    let filter = [a, b, c, 128 - 2 * (a + b + c), c, b, a];
                    assert_eq!(filter.iter().sum::<i32>(), 128);
                }
            }
        }
    }
}
