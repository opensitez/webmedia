//! Wiener, self-guided and switchable restoration, AV1 5.11.57-58 and 7.17.

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

#[derive(Clone, Copy)]
enum Unit {
    Wiener([[i32; 3]; 2]),
    SelfGuided { set: usize, weights: [i32; 2] },
}

const SGR_PARAMS: [[i32; 4]; 16] = [
    [2, 12, 1, 4], [2, 15, 1, 6], [2, 18, 1, 8], [2, 21, 1, 9],
    [2, 24, 1, 10], [2, 29, 1, 11], [2, 36, 1, 12], [2, 45, 1, 13],
    [2, 56, 1, 14], [2, 68, 1, 15], [0, 0, 1, 5], [0, 0, 1, 8],
    [0, 0, 1, 11], [0, 0, 1, 14], [2, 30, 0, 0], [2, 75, 0, 0],
];
const SGR_SGR_BITS: u32 = 8;
const SGR_RST_BITS: u32 = 4;
const SGR_PRJ_BITS: u32 = 7;
const SGR_MTABLE_BITS: u32 = 20;
const SGR_RECIP_BITS: u32 = 12;

fn round(value: i64, bits: u32) -> i64 {
    if bits == 0 { value } else { (value + (1 << (bits - 1))) >> bits }
}

fn self_guided(
    source: impl Fn(i32, i32) -> i16, width: usize, height: usize,
    depth: u8, set: usize, weights: [i32; 2], output: &mut [u16], stride: usize,
) {
    let mut filtered = [Vec::new(), Vec::new()];
    for pass in 0..2 {
        let radius = SGR_PARAMS[set][pass * 2];
        if radius == 0 { continue; }
        let n = i64::from((2 * radius + 1).pow(2));
        let n2e = n * n * i64::from(SGR_PARAMS[set][pass * 2 + 1]);
        let scale = ((1 << SGR_MTABLE_BITS) + n2e / 2) / n2e;
        let reciprocal = ((1 << SGR_RECIP_BITS) + n / 2) / n;
        let pitch = width + 2;
        let mut a = vec![0i64; pitch * (height + 2)];
        let mut b = vec![0i64; a.len()];
        for y in -1..=height as i32 {
            for x in -1..=width as i32 {
                let (mut squares, mut sum) = (0i64, 0i64);
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let sample = i64::from(source(x + dx, y + dy));
                        squares += sample * sample;
                        sum += sample;
                    }
                }
                squares = round(squares, 2 * u32::from(depth - 8));
                let reduced = round(sum, u32::from(depth - 8));
                let variance = (squares * n - reduced * reduced).max(0);
                let z = round(variance * scale, SGR_MTABLE_BITS);
                let coefficient = if z >= 255 { 256 } else if z == 0 { 1 }
                    else { ((z << SGR_SGR_BITS) + z / 2) / (z + 1) };
                let i = (y + 1) as usize * pitch + (x + 1) as usize;
                a[i] = coefficient;
                b[i] = round(((1 << SGR_SGR_BITS) - coefficient) * sum * reciprocal, SGR_RECIP_BITS);
            }
        }
        filtered[pass].resize(width * height, 0i64);
        for y in 0..height {
            let shift = if pass == 0 && y & 1 != 0 { 4 } else { 5 };
            for x in 0..width {
                let (mut aa, mut bb) = (0i64, 0i64);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let weight = if pass == 0 {
                            if (y as i32 + dy) & 1 == 0 { 0 } else if dx == 0 { 6 } else { 5 }
                        } else if dx == 0 || dy == 0 { 4 } else { 3 };
                        let i = (y as i32 + dy + 1) as usize * pitch + (x as i32 + dx + 1) as usize;
                        aa += weight * a[i];
                        bb += weight * b[i];
                    }
                }
                filtered[pass][y * width + x] = round(aa * i64::from(source(x as i32, y as i32)) + bb,
                    SGR_SGR_BITS + shift - SGR_RST_BITS);
            }
        }
    }
    let w0 = i64::from(weights[0]);
    let w1 = i64::from(weights[1]);
    let w2 = (1 << SGR_PRJ_BITS) - w0 - w1;
    for y in 0..height {
        for x in 0..width {
            let u = i64::from(source(x as i32, y as i32)) << SGR_RST_BITS;
            let f0 = if filtered[0].is_empty() { u } else { filtered[0][y * width + x] };
            let f1 = if filtered[1].is_empty() { u } else { filtered[1][y * width + x] };
            output[y * stride + x] = round(w0 * f0 + w1 * u + w2 * f1,
                SGR_RST_BITS + SGR_PRJ_BITS).clamp(0, (1 << depth) - 1) as u16;
        }
    }
}

pub(crate) struct Restoration {
    units: Vec<Vec<Option<Unit>>>,
    dimensions: Vec<(usize, usize)>,
    previous: [[[i32; 3]; 2]; 3],
    previous_sgr: [[i32; 2]; 3],
}

impl Restoration {
    pub(crate) fn begin_tile(&mut self) {
        self.previous = [[[3, -7, 15]; 2]; 3];
        self.previous_sgr = [[-32, 31]; 3];
    }

    pub(crate) fn new(s: &SequenceHeader, h: &IntraFrameHeader) -> Result<Self, Error> {
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
            previous_sgr: [[-32, 31]; 3],
        })
    }

    pub(crate) fn read(
        &mut self,
        d: &mut SymbolDecoder<'_>,
        cdfs: &mut super::inter::InterCdfs,
        s: &SequenceHeader,
        h: &IntraFrameHeader,
        row: usize,
        col: usize,
    ) -> Result<(), Error> {
        let superblock_step = if s.use_128x128_superblock { 32 } else { 16 };
        for plane in 0..self.units.len() {
            if h.restoration_types[plane] == 0 {
                continue;
            }
            let sx = usize::from(plane > 0 && s.subsampling_x);
            let sy = usize::from(plane > 0 && s.subsampling_y);
            let size = h.restoration_unit_sizes[plane] as usize;
            let (cols, rows) = self.dimensions[plane];
            let r0 = (row * (4 >> sy)).div_ceil(size);
            let r1 = ((row + superblock_step) * (4 >> sy)).div_ceil(size).min(rows);
            let c0 = (col * (4 >> sx)).div_ceil(size);
            let c1 = ((col + superblock_step) * (4 >> sx)).div_ceil(size).min(cols);
            for r in r0..r1 {
                for c in c0..c1 {
                    let kind = match h.restoration_types[plane] {
                        2 => usize::from(d.read_symbol(&mut cdfs.wiener)? != 0),
                        3 => 2 * usize::from(d.read_symbol(&mut cdfs.sgrproj)? != 0),
                        1 => d.read_symbol(&mut cdfs.restoration_type)?,
                        _ => return Err(Error::Invalid("restoration type")),
                    };
                    if kind == 0 {
                        continue;
                    }
                    if kind == 2 {
                        let set = d.read_literal(4)? as usize;
                        let mut weights = [0; 2];
                        for pass in 0..2 {
                            let (low, high) = if pass == 0 { (-96, 32) } else { (-32, 96) };
                            weights[pass] = if SGR_PARAMS[set][pass * 2] != 0 {
                                signed_subexp(d, low, high, 4, self.previous_sgr[plane][pass])?
                            } else if pass == 0 { 0 }
                            else { ((1 << SGR_PRJ_BITS) - self.previous_sgr[plane][0]).clamp(low, high - 1) };
                            self.previous_sgr[plane][pass] = weights[pass];
                        }
                        self.units[plane][r * cols + c] = Some(Unit::SelfGuided { set, weights });
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
                    self.units[plane][r * cols + c] = Some(Unit::Wiener(coefficients));
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
                    let Some(unit) = self.units[plane][unit_y * cols + unit_x] else {
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
                    let coeff = match unit {
                        Unit::Wiener(coeff) => coeff,
                        Unit::SelfGuided { set, weights } => {
                            self_guided(|xx, yy| source(x as i32 + xx, y as i32 + yy),
                                w, ht, s.bit_depth, set, weights,
                                &mut out.samples[y * out.stride + x..], out.stride);
                            continue;
                        }
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
                    let Some(unit) = self.units[plane][unit_y * cols + unit_x] else {
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
                    let coeff = match unit {
                        Unit::Wiener(coeff) => coeff,
                        Unit::SelfGuided { set, weights } => {
                            let w = (4 >> sx).min(width - x);
                            let ht = (4 >> sy).min(height - y);
                            self_guided(|xx, yy| source(x as i32 + xx, y as i32 + yy) as i16,
                                w, ht, s.bit_depth, set, weights,
                                &mut out.samples[y * out.stride + x..], out.stride);
                            continue;
                        }
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
    fn self_guided_stripes_match_small_blocks_for_every_parameter_set() {
        for depth in [8u8, 10, 12] {
            let max = (1 << depth) - 1;
            for set in 0..16 {
                for weights in [[-96, -32], [-32, 31], [31, 95]] {
                    let (width, height) = (19usize, 18usize);
                    let source = |x: i32, y: i32| {
                        let x = x.clamp(0, width as i32 - 1);
                        let y = y.clamp(0, height as i32 - 1);
                        ((x * 73 + y * 97 + 31) & max) as i16
                    };
                    let mut actual = vec![0; width * height];
                    self_guided(source, width, height, depth, set, weights, &mut actual, width);
                    let mut reference = vec![0; width * height];
                    for y in (0..height).step_by(4) {
                        for x in (0..width).step_by(4) {
                            self_guided(|xx, yy| source(x as i32 + xx, y as i32 + yy),
                                4.min(width - x), 4.min(height - y), depth, set, weights,
                                &mut reference[y * width + x..], width);
                        }
                    }
                    assert_eq!(actual, reference, "depth={depth} set={set} weights={weights:?}");
                    for value in [0, 1, max / 2, max] {
                        let mut uniform = vec![0; width * height];
                        self_guided(|_, _| value as i16, width, height, depth, set, weights,
                            &mut uniform, width);
                        // The normative reciprocal and high-depth rounding can
                        // move a uniform signal by one LSB; do not assume ideal DC gain.
                        let precision_round = |v: i64, bits: u32| {
                            if bits == 0 { v } else { (v + (1 << (bits - 1))) >> bits }
                        };
                        let mut f = [i64::from(value) << 4; 2];
                        for pass in 0..2 {
                            let radius = SGR_PARAMS[set][pass * 2];
                            if radius == 0 { continue; }
                            let n = i64::from((radius * 2 + 1).pow(2));
                            let sum = n * i64::from(value);
                            let squares = precision_round(n * i64::from(value).pow(2),
                                2 * u32::from(depth - 8));
                            let mean = precision_round(sum, u32::from(depth - 8));
                            let n2e = n * n * i64::from(SGR_PARAMS[set][pass * 2 + 1]);
                            let scale = ((1 << 20) + n2e / 2) / n2e;
                            let z = precision_round((squares * n - mean * mean).max(0) * scale, 20);
                            let a = if z >= 255 { 256 } else if z == 0 { 1 }
                                else { (z * 256 + z / 2) / (z + 1) };
                            let reciprocal = ((1 << 12) + n / 2) / n;
                            let b = precision_round((256 - a) * sum * reciprocal, 12);
                            f[pass] = precision_round(a * i64::from(value) + b, 4);
                        }
                        let w0 = i64::from(weights[0]);
                        let w1 = i64::from(weights[1]);
                        let expected = precision_round(w0 * f[0] + w1 * (i64::from(value) << 4)
                            + (128 - w0 - w1) * f[1], 11).clamp(0, i64::from(max)) as u16;
                        assert!(uniform.iter().all(|&sample| sample == expected),
                            "rounded DC depth={depth} set={set} value={value}");
                    }
                }
            }
        }
    }

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
                                *unit = Some(Unit::Wiener(coeff));
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
            previous_sgr: [[-32, 31]; 3],
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
