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

    pub(crate) fn has_filter(&self) -> bool {
        #[cfg(test)]
        if RESTORATION_REFERENCE.with(|flag| flag.get()) {
            return true;
        }
        self.units.iter().flatten().any(Option::is_some)
    }

    pub(crate) fn apply(
        &self,
        planes: &mut [DecodedPlane],
        deblocked: &[DecodedPlane],
        s: &SequenceHeader,
        h: &IntraFrameHeader,
    ) -> Result<(), Error> {
        #[cfg(test)]
        if RESTORATION_REFERENCE.with(|flag| flag.get()) {
            return self.apply_reference(planes, deblocked, s, h);
        }
        if !self.has_filter() {
            return Ok(());
        }
        let input: Vec<_> = planes
            .iter()
            .enumerate()
            .map(|(plane, p)| {
                self.units[plane]
                    .iter()
                    .any(Option::is_some)
                    .then(|| p.clone())
            })
            .collect();
        let use_neon = std::env::var_os("AV1_DISABLE_NEON").is_none();
        let mut window = Vec::<i16>::new();
        let mut intermediate = Vec::<i16>::new();
        for (plane, out) in planes.iter_mut().enumerate() {
            let Some(cdef) = &input[plane] else {
                continue;
            };
            let sx = usize::from(plane > 0 && s.subsampling_x);
            let sy = usize::from(plane > 0 && s.subsampling_y);
            let width = ((h.upscaled_width as usize) + (1 << sx) / 2) >> sx;
            let height = ((h.height as usize) + (1 << sy) / 2) >> sy;
            let size = h.restoration_unit_sizes[plane] as usize;
            let (cols, rows) = self.dimensions[plane];
            let round0 = if s.bit_depth == 12 { 5 } else { 3 };
            let round1 = 14 - round0;
            let offset = 1 << (s.bit_depth as u32 + 7 - round0 - 1);
            let limit = (1 << (s.bit_depth as u32 + 1 + 7 - round0)) - 1;
            let mut y = 0;
            while y < height {
                let stripe = ((y << sy) + 8) / 64;
                let start = (-8 + stripe as i32 * 64) >> sy;
                let end = start + (64 >> sy) - 1;
                let ht = (end as usize + 1 - y).min(height - y);
                let unit_y = ((y + (8 >> sy)) / size).min(rows - 1);
                for unit_x in 0..cols {
                    let Some(coeff) = self.units[plane][unit_y * cols + unit_x] else {
                        continue;
                    };
                    let x = unit_x * size;
                    let w = if unit_x + 1 == cols {
                        width - x
                    } else {
                        size.min(width - x)
                    };
                    let source = |xx: i32, yy: i32| -> i16 {
                        let xx = xx.clamp(0, width as i32 - 1) as usize;
                        let yy = yy.clamp(0, height as i32 - 1);
                        let (p, yy) = if yy < start {
                            (&deblocked[plane], yy.max(start - 2))
                        } else if yy > end {
                            (&deblocked[plane], yy.min(end + 2))
                        } else {
                            (cdef, yy)
                        };
                        p.samples[yy as usize * p.stride + xx] as i16
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
                        .map(|v| v as i16)
                    };
                    let vertical = make_filter(coeff[0]);
                    let horizontal = make_filter(coeff[1]);
                    let halo_width = w + 6;
                    window.resize(window.len().max(halo_width * (ht + 6)), 0);
                    intermediate.resize(intermediate.len().max(w * (ht + 6)), 0);
                    // A stripe is also bounded by restoration-unit row edges.
                    // Snapshot its halo once; neighboring outputs share it.
                    for r in 0..ht + 6 {
                        for c in 0..halo_width {
                            window[r * halo_width + c] =
                                source((x + c) as i32 - 3, (y + r) as i32 - 3);
                        }
                        wiener_horizontal(
                            &window[r * halo_width..(r + 1) * halo_width],
                            &mut intermediate[r * w..(r + 1) * w],
                            &horizontal,
                            round0,
                            -offset,
                            limit - offset,
                            use_neon,
                        );
                    }
                    for r in 0..ht {
                        wiener_vertical(
                            &intermediate[..w * (ht + 6)],
                            r * w,
                            w,
                            &mut out.samples
                                [(y + r) * out.stride + x..(y + r) * out.stride + x + w],
                            &vertical,
                            round1,
                            (1 << s.bit_depth) - 1,
                            use_neon,
                        );
                    }
                }
                y += ht;
            }
        }
        Ok(())
    }
    #[cfg(test)]
    fn apply_reference(
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

#[allow(clippy::too_many_arguments)]
fn wiener_horizontal(
    source: &[i16],
    output: &mut [i16],
    weights: &[i16; 7],
    round: u32,
    low: i32,
    high: i32,
    use_neon: bool,
) {
    let mut x = 0;
    #[cfg(target_arch = "aarch64")]
    if use_neon {
        while x + 8 <= output.len() {
            // The source includes six halo samples beyond the output width.
            unsafe {
                horizontal_neon(
                    source.as_ptr().add(x),
                    output.as_mut_ptr().add(x),
                    weights,
                    round,
                    low,
                    high,
                );
            }
            x += 8;
        }
    }
    for x in x..output.len() {
        let sum: i32 = (0..7)
            .map(|tap| i32::from(weights[tap]) * i32::from(source[x + tap]))
            .sum();
        output[x] = ((sum + (1 << (round - 1))) >> round).clamp(low, high) as i16;
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = use_neon;
}

#[allow(clippy::too_many_arguments)]
fn wiener_vertical(
    source: &[i16],
    position: usize,
    stride: usize,
    output: &mut [u16],
    weights: &[i16; 7],
    round: u32,
    max: u16,
    use_neon: bool,
) {
    let mut x = 0;
    #[cfg(target_arch = "aarch64")]
    if use_neon {
        while x + 8 <= output.len() {
            // Seven complete intermediate rows and eight output lanes exist.
            unsafe {
                vertical_neon(
                    source.as_ptr().add(position + x),
                    stride,
                    output.as_mut_ptr().add(x),
                    weights,
                    round,
                    max,
                );
            }
            x += 8;
        }
    }
    for x in x..output.len() {
        let sum: i32 = (0..7)
            .map(|tap| i32::from(weights[tap]) * i32::from(source[position + tap * stride + x]))
            .sum();
        output[x] = ((sum + (1 << (round - 1))) >> round).clamp(0, i32::from(max)) as u16;
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = use_neon;
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn horizontal_neon(
    source: *const i16,
    output: *mut i16,
    weights: &[i16; 7],
    round: u32,
    low: i32,
    high: i32,
) {
    use std::arch::aarch64::*;
    unsafe {
        let mut lo = vdupq_n_s32(0);
        let mut hi = lo;
        for (tap, &weight) in weights.iter().enumerate() {
            if weight != 0 {
                let values = vld1q_s16(source.add(tap));
                lo = vmlal_n_s16(lo, vget_low_s16(values), weight);
                hi = vmlal_n_s16(hi, vget_high_s16(values), weight);
            }
        }
        let rounding = vdupq_n_s32(1 << (round - 1));
        let shift = vdupq_n_s32(-(round as i32));
        let lower = vdupq_n_s32(low);
        let upper = vdupq_n_s32(high);
        lo = vminq_s32(
            upper,
            vmaxq_s32(lower, vshlq_s32(vaddq_s32(lo, rounding), shift)),
        );
        hi = vminq_s32(
            upper,
            vmaxq_s32(lower, vshlq_s32(vaddq_s32(hi, rounding), shift)),
        );
        vst1q_s16(output, vcombine_s16(vmovn_s32(lo), vmovn_s32(hi)));
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vertical_neon(
    source: *const i16,
    stride: usize,
    output: *mut u16,
    weights: &[i16; 7],
    round: u32,
    max: u16,
) {
    use std::arch::aarch64::*;
    unsafe {
        let mut lo = vdupq_n_s32(0);
        let mut hi = lo;
        for (tap, &weight) in weights.iter().enumerate() {
            if weight != 0 {
                let values = vld1q_s16(source.add(tap * stride));
                lo = vmlal_n_s16(lo, vget_low_s16(values), weight);
                hi = vmlal_n_s16(hi, vget_high_s16(values), weight);
            }
        }
        let rounding = vdupq_n_s32(1 << (round - 1));
        let shift = vdupq_n_s32(-(round as i32));
        lo = vshlq_s32(vaddq_s32(lo, rounding), shift);
        hi = vshlq_s32(vaddq_s32(hi, rounding), shift);
        let result = vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi));
        vst1q_u16(output, vminq_u16(result, vdupq_n_u16(max)));
    }
}

#[cfg(test)]
std::thread_local! {
    static RESTORATION_REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
pub(crate) fn set_restoration_reference(enabled: bool) {
    RESTORATION_REFERENCE.with(|flag| flag.set(enabled));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stripe_scalar_and_neon_match_original_at_unit_edges() {
        let bytes = [
            0x12, 0x00, 0x0a, 0x0b, 0x02, 0x00, 0x00, 0x05, 0x15, 0x7f, 0xfc, 0x4a, 0xf9, 0x00,
            0x40, 0x32, 0x0c, 0x10, 0x00, 0xac, 0x02, 0x05, 0x14, 0x20, 0x81, 0x00, 0x00, 0x98,
            0x80,
        ];
        let mut stream = super::super::ObuStream::new();
        let obus = stream.push(&bytes).unwrap();
        let mut s =
            SequenceHeader::parse(&obus.iter().find(|o| o.kind == 1).unwrap().payload).unwrap();
        let obu = obus.iter().find(|o| o.kind == 6).unwrap();
        let mut h = IntraFrameHeader::parse(&obu.payload, &s, 0, 0).unwrap();
        h.restoration_types = vec![2; 3];
        for depth in [8, 10, 12] {
            s.bit_depth = depth;
            for (sx, sy) in [(false, false), (false, true), (true, false), (true, true)] {
                s.subsampling_x = sx;
                s.subsampling_y = sy;
                for size in [64, 128, 256] {
                    h.restoration_unit_sizes = [size; 3];
                    for (width, height) in [
                        (1usize, 1usize),
                        (17, 19),
                        (64, 64),
                        (73, 65),
                        (193, 137),
                        (257, 141),
                    ] {
                        h.width = width as u32;
                        h.upscaled_width = width as u32;
                        h.height = height as u32;
                        let mut filter = Restoration::new(&s, &h).unwrap();
                        let mut state = 73u32;
                        for (plane, units) in filter.units.iter_mut().enumerate() {
                            for (index, unit) in units.iter_mut().enumerate() {
                                if index % 5 == 4 {
                                    continue;
                                }
                                let mut coeff = [[0; 3]; 2];
                                for axis in &mut coeff {
                                    for tap in 0..3 {
                                        state =
                                            state.wrapping_mul(1664525).wrapping_add(1013904223);
                                        axis[tap] = [-5, -23, -17][tap]
                                            + ((state >> 16) % [16, 32, 64][tap]) as i32;
                                    }
                                    if plane > 0 {
                                        axis[0] = 0;
                                    }
                                }
                                *unit = Some(coeff);
                            }
                        }
                        let mut make_planes = || {
                            (0..3)
                                .map(|plane| {
                                    let px = usize::from(plane > 0 && sx);
                                    let py = usize::from(plane > 0 && sy);
                                    let w = width.div_ceil(8) * 8 >> px;
                                    let ht = height.div_ceil(8) * 8 >> py;
                                    let stride = w + 3;
                                    let samples = (0..stride * ht)
                                        .map(|_| {
                                            state = state
                                                .wrapping_mul(1664525)
                                                .wrapping_add(1013904223);
                                            ((state >> 16) & ((1 << depth) - 1)) as u16
                                        })
                                        .collect();
                                    DecodedPlane {
                                        width: w,
                                        height: ht,
                                        stride,
                                        samples,
                                    }
                                })
                                .collect::<Vec<_>>()
                        };
                        let input = make_planes();
                        let deblocked = make_planes();
                        let mut expected = input.clone();
                        filter
                            .apply_reference(&mut expected, &deblocked, &s, &h)
                            .unwrap();
                        let mut actual = input;
                        filter.apply(&mut actual, &deblocked, &s, &h).unwrap();
                        assert_eq!(
                            actual, expected,
                            "depth={depth} sx={sx} sy={sy} unit={size} size={width}x{height}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn inactive_units_do_not_need_deblocked_snapshot() {
        let filter = Restoration {
            units: vec![vec![None]],
            dimensions: vec![(1, 1)],
            previous: [[[3, -7, 15]; 2]; 3],
        };
        assert!(!filter.has_filter());
    }

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
