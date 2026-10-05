// Test-only draft copied from our decoder, never compiled for production.
//! Fixed 48-kHz, 20-ms CELT packet reconstruction from RFC 6716 sections
//! 4.3.2--4.3.7 and its normative Appendix A, with RFC 8251 energy limiting.
//! The codebooks and MDCT plans are retained across packets.

use super::{
    IdentificationHeader, Packet,
    celt::Layout,
    energy::BAND_MEANS,
    frame::{DraftShapeStart, FrameStart},
    invalid,
    pvq::Codebook,
    range::RangeDecoder,
    synthesis::{CeltSynthesis, SpectralFrame},
};
use crate::video::backend::{AudioSamples, MediaDecodeError};
use std::collections::BTreeMap;

const LOG_N: [i32; 21] = [
    0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 16, 16, 16, 21, 21, 24, 29, 34, 36,
];
const ORDER: [usize; 30] = [
    1, 0, 3, 0, 2, 1, 7, 0, 4, 3, 6, 1, 5, 2, 15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5,
];

struct PulseChoice {
    book: Codebook,
    bits: i32,
    pulses: usize,
    rotations: [SpreadRotation; 3],
}
#[derive(Clone, Copy)]
struct SpreadRotation {
    primary: (f64, f64),
    secondary: (f64, f64),
}

impl SpreadRotation {
    fn new(size: usize, pulses: usize, factor: f64) -> Self {
        let gain = size as f64 / (size as f64 + factor * pulses as f64);
        let theta = std::f64::consts::FRAC_PI_4 * gain * gain;
        Self {
            primary: theta.sin_cos(),
            secondary: (std::f64::consts::FRAC_PI_2 - theta).sin_cos(),
        }
    }
}
type Menus = BTreeMap<usize, Vec<PulseChoice>>;

fn nearest_pulse(choices: &[PulseChoice], budget: i32) -> usize {
    let upper = choices.partition_point(|choice| choice.bits < budget);
    if upper == choices.len() {
        choices.len() - 1
    } else if upper == 0 {
        0
    } else if budget - choices[upper - 1].bits <= choices[upper].bits - budget {
        upper - 1
    } else {
        upper
    }
}

// Conservative integer log2 in eighths, as specified by the normative PVQ
// cost calculation. Upward rounding at each squaring is intentional.
fn pulse_cost(entries: u32) -> i32 {
    let exponent = 32 - entries.leading_zeros();
    if entries.is_power_of_two() {
        return (exponent as i32 - 1) * 8;
    }
    let mut significand = if exponent > 16 {
        u64::from(entries).div_ceil(1u64 << (exponent - 16))
    } else {
        u64::from(entries) << (16 - exponent)
    };
    let mut result = (exponent as i32 - 1) * 8;
    for place in (0..=3).rev() {
        let carry = significand >> 16;
        result += (carry as i32) << place;
        significand = (significand + carry) >> carry;
        significand = (significand * significand + 32767) >> 15;
    }
    result + i32::from(significand > 32768)
}

fn menus() -> Result<Menus, MediaDecodeError> {
    let layout = Layout::from_configuration(31)?.ok_or(MediaDecodeError::Unsupported)?;
    let mut result = Menus::new();
    for band in 0..21 {
        let mut size = layout.band(band)?.len();
        while size > 1 {
            result.entry(size).or_insert_with(|| {
                let mut choices = Vec::new();
                for index in 0..=40 {
                    let pulses = if index < 8 {
                        index
                    } else {
                        (8 + (index & 7)) << ((index >> 3) - 1)
                    };
                    let Ok(book) = Codebook::new(size, pulses) else {
                        break;
                    };
                    let bits = if index == 0 {
                        0
                    } else {
                        pulse_cost(book.entries())
                    };
                    choices.push(PulseChoice {
                        book,
                        bits,
                        pulses,
                        rotations: [15.0, 10.0, 5.0]
                            .map(|factor| SpreadRotation::new(size, pulses, factor)),
                    });
                }
                choices
            });
            size /= 2;
        }
    }
    Ok(result)
}

fn q15(a: i32, b: i32) -> i32 {
    (16384 + a * b) >> 15
}
fn angle_cos(angle: i32) -> i32 {
    let square = (4096 + angle * angle) >> 13;
    32768 - square + q15(square, -7651 + q15(square, 8277 + q15(-626, square)))
}
fn log_ratio(sine: i32, cosine: i32) -> i32 {
    let s = 32 - sine.leading_zeros();
    let c = 32 - cosine.leading_zeros();
    let sine = sine << (15 - s);
    let cosine = cosine << (15 - c);
    (s as i32 - c as i32) * 2048 + q15(sine, q15(sine, -2597) + 7932)
        - q15(cosine, q15(cosine, -2597) + 7932)
}
fn unit(vector: &mut [f64], gain: f64) {
    let norm = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        super::simd::scale_in_place(vector, gain / norm);
    }
}
fn random(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    *seed
}
// Normative inverse rotation: PVQ blocks are contiguous, with positive
// orientation. This differs from the descriptive prose's forward matrix.
fn packet_spread(values: &mut [f64], choice: &PulseChoice, blocks: usize, mode: u8) {
    if mode == 0 || 2 * choice.pulses >= values.len() {
        return;
    }
    let rotation = choice.rotations[usize::from(mode - 1)];
    let length = values.len() / blocks;
    let rotate = |slice: &mut [f64], stride: usize, (sine, cosine): (f64, f64)| {
        let mut pair = |index: usize| {
            let a = slice[index];
            let b = slice[index + stride];
            slice[index] = cosine * a - sine * b;
            slice[index + stride] = sine * a + cosine * b;
        };
        for index in 0..length - stride {
            pair(index);
        }
        for index in (0..length.saturating_sub(2 * stride)).rev() {
            pair(index);
        }
    };
    for block in values.chunks_exact_mut(length) {
        if length >= 8 {
            let stride = (length as f64).sqrt().round() as usize;
            rotate(block, stride, rotation.secondary);
        }
        rotate(block, 1, rotation.primary);
    }
}

fn haar(values: &mut [f64], size: usize, stride: usize) {
    for offset in 0..stride {
        for pair in 0..size / 2 {
            let a = stride * 2 * pair + offset;
            let b = a + stride;
            let x = values[a] * std::f64::consts::FRAC_1_SQRT_2;
            let y = values[b] * std::f64::consts::FRAC_1_SQRT_2;
            values[a] = x + y;
            values[b] = x - y;
        }
    }
}
fn reorder(values: &mut [f64], stride: usize, sequency: bool, inverse: bool) {
    if stride == 1 {
        return;
    }
    let length = values.len() / stride;
    let mut output = vec![0.0; values.len()];
    for block in 0..stride {
        let ordered = if sequency {
            ORDER[stride - 2 + block]
        } else {
            block
        };
        for bin in 0..length {
            let a = bin * stride + block;
            let b = ordered * length + bin;
            if inverse {
                output[a] = values[b];
            } else {
                output[b] = values[a];
            }
        }
    }
    values.copy_from_slice(&output);
}

struct ShapeReader<'a, 'b> {
    entropy: &'b mut RangeDecoder<'a>,
    menus: &'b Menus,
    seed: &'b mut u32,
    remaining: i32,
    spread: u8,
    band: usize,
    intensity: usize,
    lm: i32,
}

impl ShapeReader<'_, '_> {
    fn theta(
        &mut self,
        size: usize,
        budget: i32,
        lm: i32,
        blocks: usize,
        stereo: bool,
    ) -> Result<(i32, i32, i32, bool), MediaDecodeError> {
        let cap = LOG_N[self.band] + lm * 8;
        let offset = cap / 2 - if stereo && size == 2 { 16 } else { 4 };
        let degrees = 2 * size as i32 - 1 - i32::from(stereo && size == 2);
        let resolution = (budget - cap - 32)
            .min((budget + degrees * offset) / degrees)
            .min(64);
        let levels = if resolution < 4 || (stereo && self.band >= self.intensity) {
            1
        } else {
            const EXP: [i32; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];
            let approximate = EXP[(resolution & 7) as usize] >> (14 - (resolution >> 3));
            ((approximate + 1) / 2) * 2
        };
        let before = self.entropy.tell_fractional();
        let mut inverted = false;
        let angle = if levels == 1 {
            if stereo && budget > 16 && self.remaining > 16 {
                inverted = self.entropy.bit(2)?;
            }
            0
        } else {
            let symbol = if stereo && size > 2 {
                let masses: Vec<u16> = (0..=levels)
                    .map(|i| if i <= levels / 2 { 3 } else { 1 })
                    .collect();
                self.entropy.symbol(&masses)? as i32
            } else if blocks > 1 || stereo {
                self.entropy.uniform((levels + 1) as u32)? as i32
            } else {
                let masses: Vec<u16> = (0..=levels)
                    .map(|i| (i + 1).min(levels + 1 - i) as u16)
                    .collect();
                self.entropy.symbol(&masses)? as i32
            };
            symbol * 16384 / levels
        };
        let cost = (self.entropy.tell_fractional() - before) as i32;
        self.remaining -= cost;
        let delta = match angle {
            0 => -16384,
            16384 => 16384,
            _ => q15(
                (size as i32 - 1) * 128,
                log_ratio(angle_cos(16384 - angle), angle_cos(angle)),
            ),
        };
        Ok((angle, delta, cost, inverted))
    }

    fn vector(
        &mut self,
        size: usize,
        budget: i32,
        blocks: usize,
        lm: i32,
        gain: f64,
        fold: Option<&[f64]>,
        fill: u32,
    ) -> Result<(Vec<f64>, u32), MediaDecodeError> {
        if size == 1 {
            let negative = if self.remaining >= 8 {
                self.remaining -= 8;
                self.entropy.raw_bits(1)? != 0
            } else {
                false
            };
            return Ok((vec![if negative { -gain } else { gain }], 1));
        }
        let choices = self
            .menus
            .get(&size)
            .ok_or_else(|| invalid("missing CELT pulse menu"))?;
        let maximum = choices
            .last()
            .ok_or_else(|| invalid("empty CELT pulse menu"))?
            .bits;
        if lm >= 0 && size > 2 && budget > maximum + 11 {
            let half = size / 2;
            let split_blocks = (blocks + 1) / 2;
            let (angle, mut delta, cost, _) = self.theta(half, budget, lm - 1, blocks, false)?;
            if blocks > 1 && angle != 0 && angle != 16384 {
                if angle > 8192 {
                    delta -= delta >> (5 - lm);
                } else {
                    delta = 0.min(delta + ((half as i32 * 8) >> (6 - lm)));
                }
            }
            let available = (budget - cost).max(0);
            let mut budgets = [((available - delta) / 2).clamp(0, available), 0];
            budgets[1] = available - budgets[0];
            let gains = [
                if angle == 16384 {
                    0.0
                } else {
                    angle_cos(angle) as f64 / 32768.0
                },
                if angle == 0 {
                    0.0
                } else {
                    angle_cos(16384 - angle) as f64 / 32768.0
                },
            ];
            let expanded = if blocks == 1 {
                (fill & 1) | (fill << 1)
            } else {
                fill
            };
            let allowed = if angle == 0 {
                expanded & ((1 << split_blocks) - 1)
            } else if angle == 16384 {
                expanded & (((1 << split_blocks) - 1) << split_blocks)
            } else {
                expanded
            };
            let mut pieces = [Vec::new(), Vec::new()];
            let mut masks = [0, 0];
            let first = usize::from(budgets[1] > budgets[0]);
            for index in [first, 1 - first] {
                let before = self.remaining;
                let lower = fold.map(|x| &x[index * half..(index + 1) * half]);
                let (v, mask) = self.vector(
                    half,
                    budgets[index],
                    split_blocks,
                    lm - 1,
                    gain * gains[index],
                    lower,
                    allowed >> (index * split_blocks),
                )?;
                pieces[index] = v;
                masks[index] = mask;
                if index == first {
                    let unused = budgets[index] - (before - self.remaining);
                    if unused > 24 && ((index == 0 && angle != 0) || (index == 1 && angle != 16384))
                    {
                        budgets[1 - index] += unused - 24;
                    }
                }
            }
            let upper = std::mem::take(&mut pieces[1]);
            pieces[0].extend(upper);
            return Ok((
                std::mem::take(&mut pieces[0]),
                masks[0] | (masks[1] << (blocks / 2)),
            ));
        }
        let mut chosen = nearest_pulse(choices, budget);
        while chosen > 0 && choices[chosen].bits > self.remaining {
            chosen -= 1;
        }
        let choice = &choices[chosen];
        self.remaining -= choice.bits;
        let mut values = vec![0.0; size];
        if chosen == 0 {
            let mask = fill & ((1 << blocks) - 1);
            if mask != 0 {
                for (index, value) in values.iter_mut().enumerate() {
                    let noise = random(self.seed);
                    *value = if let Some(source) = fold {
                        source[index]
                            + if noise & 0x8000 != 0 {
                                1.0 / 256.0
                            } else {
                                -1.0 / 256.0
                            }
                    } else {
                        (noise as i32 >> 20) as f64
                    };
                }
                unit(&mut values, gain);
            }
            return Ok((
                values,
                if fold.is_some() {
                    mask
                } else if mask != 0 {
                    (1 << blocks) - 1
                } else {
                    0
                },
            ));
        }
        let mut pulse_storage = [0; 176];
        let pulses = &mut pulse_storage[..size];
        choice
            .book
            .decode_pulses(self.entropy, pulses)
            .map_err(|error| {
                invalid(&format!(
                    "CELT band {} N={size} K={} budget={budget}: {error:?}",
                    self.band, choice.pulses
                ))
            })?;
        let length = size / blocks;
        let mask = pulses
            .chunks_exact(length)
            .enumerate()
            .fold(0, |mask, (b, part)| {
                mask | if part.iter().any(|&value| value != 0) {
                    1 << b
                } else {
                    0
                }
            });
        for (value, &pulse) in values.iter_mut().zip(pulses.iter()) {
            *value = pulse as f64;
        }
        unit(&mut values, gain);
        packet_spread(&mut values, choice, blocks, self.spread);
        Ok((values, mask))
    }

    fn mono(
        &mut self,
        size: usize,
        budget: i32,
        original_blocks: usize,
        adjustment: i8,
        gain: f64,
        fold: Option<&[f64]>,
        mut fill: u32,
    ) -> Result<(Vec<f64>, u32), MediaDecodeError> {
        let recombine = adjustment.max(0) as usize;
        let mut blocks = original_blocks >> recombine;
        let mut width = size / blocks;
        let mut folded = fold.map(|x| x.to_vec());
        for level in 0..recombine {
            if let Some(ref mut values) = folded {
                haar(values, size >> level, 1 << level);
            }
            let mut compressed = 0;
            for bit in 0..8 {
                if fill & (3 << (2 * bit)) != 0 {
                    compressed |= 1 << bit;
                }
            }
            fill = compressed;
        }
        let mut divides = 0;
        while adjustment < 0 && divides < adjustment.unsigned_abs() as usize && width % 2 == 0 {
            if let Some(ref mut values) = folded {
                haar(values, width, blocks);
            }
            fill |= fill << blocks;
            blocks *= 2;
            width /= 2;
            divides += 1;
        }
        let stride = blocks << recombine;
        let sequency = original_blocks == 1;
        if blocks > 1 {
            if let Some(ref mut values) = folded {
                reorder(values, stride, sequency, false);
            }
        }
        let (mut output, mut mask) =
            self.vector(size, budget, blocks, self.lm, gain, folded.as_deref(), fill)?;
        if blocks > 1 {
            reorder(&mut output, stride, sequency, true);
        }
        for _ in 0..divides {
            blocks /= 2;
            width *= 2;
            mask |= mask >> blocks;
            haar(&mut output, width, blocks);
        }
        for level in 0..recombine {
            let mut expanded = 0;
            for bit in 0..8 {
                if mask & (1 << bit) != 0 {
                    expanded |= 3 << (bit * 2);
                }
            }
            mask = expanded;
            haar(&mut output, size >> level, 1 << level);
        }
        Ok((output, mask & ((1 << original_blocks) - 1)))
    }

    fn stereo(
        &mut self,
        size: usize,
        budget: i32,
        blocks: usize,
        adjustment: i8,
        fold: Option<&[f64]>,
        fill: u32,
    ) -> Result<(Vec<f64>, Vec<f64>, u32, Vec<f64>), MediaDecodeError> {
        let (angle, delta, cost, inverted) = self.theta(size, budget, self.lm, blocks, true)?;
        let available = (budget - cost).max(0);
        let mut budgets = [((available - delta) / 2).clamp(0, available), 0];
        budgets[1] = available - budgets[0];
        let mid_gain = if angle == 16384 {
            0.0
        } else {
            angle_cos(angle) as f64 / 32768.0
        };
        let side_gain = if angle == 0 {
            0.0
        } else {
            angle_cos(16384 - angle) as f64 / 32768.0
        };
        let first = usize::from(budgets[1] > budgets[0]);
        let mut parts = [Vec::new(), Vec::new()];
        let mut mask = 0;
        for index in [first, 1 - first] {
            let before = self.remaining;
            let (values, cm) = self.mono(
                size,
                budgets[index],
                blocks,
                adjustment,
                if index == 0 { 1.0 } else { side_gain },
                if index == 0 { fold } else { None },
                if index == 0 && angle != 16384 {
                    fill
                } else {
                    0
                },
            )?;
            parts[index] = values;
            mask |= cm;
            if index == first {
                let unused = budgets[index] - (before - self.remaining);
                if unused > 24 && ((index == 0 && angle != 0) || (index == 1 && angle != 16384)) {
                    budgets[1 - index] += unused - 24;
                }
            }
        }
        let folded = parts[0].iter().map(|x| x * (size as f64).sqrt()).collect();
        let mut left = Vec::with_capacity(size);
        let mut right = Vec::with_capacity(size);
        for (mid, side) in parts[0].iter().zip(&parts[1]) {
            left.push(mid * mid_gain - side);
            right.push(mid * mid_gain + side);
        }
        if left.iter().map(|x| x * x).sum::<f64>() < 0.0006
            || right.iter().map(|x| x * x).sum::<f64>() < 0.0006
        {
            left = parts[0].clone();
            right = parts[0].clone();
        } else {
            unit(&mut left, 1.0);
            unit(&mut right, 1.0);
        }
        if inverted {
            for x in &mut right {
                *x = -*x;
            }
        }
        Ok((left, right, mask, folded))
    }
}

#[derive(Debug)]
pub(super) struct BandTrace {
    pub band: usize,
    pub tf: i8,
    pub fine_bits: u8,
    pub budget: i32,
    pub tell_before: u64,
    pub tell_after: u64,
    pub errors_before: u32,
    pub errors_after: u32,
}

pub struct MusicPacketDecoder {
    channels: usize,
    history: Vec<f64>,
    log1: Vec<f64>,
    log2: Vec<f64>,
    seed: u32,
    menus: Menus,
    synthesis: CeltSynthesis,
    output_gain: f64,
    failed: bool,
    uniform_errors: u32,
    shape: Vec<Vec<f64>>,
    norm: Vec<Vec<f64>>,
    pub(super) stage_positions: Option<[u64; 5]>,
    pub(super) band_trace: Vec<BandTrace>,
}

impl MusicPacketDecoder {
    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.log1.fill(-28.0);
        self.log2.fill(-28.0);
        self.seed = 0;
        self.synthesis.reset();
        self.failed = false;
        self.uniform_errors = 0;
        self.stage_positions = None;
        self.band_trace.clear();
        for channel in &mut self.shape {
            channel.fill(0.0);
        }
        for channel in &mut self.norm {
            channel.fill(0.0);
        }
    }

    pub fn new(header: &IdentificationHeader) -> Result<Self, MediaDecodeError> {
        if header.mapping_family != 0 || !(1..=2).contains(&header.channels) {
            return Err(MediaDecodeError::Unsupported);
        }
        let channels = usize::from(header.channels);
        Ok(Self {
            channels,
            history: vec![0.0; 21 * channels],
            log1: vec![-28.0; 21 * channels],
            log2: vec![-28.0; 21 * channels],
            seed: 0,
            menus: menus()?,
            synthesis: CeltSynthesis::new(channels)?,
            output_gain: 10.0f64.powf(f64::from(header.output_gain_q8) / (256.0 * 20.0)),
            failed: false,
            uniform_errors: 0,
            shape: vec![vec![0.0; 800]; channels],
            norm: vec![vec![0.0; 800]; channels],
            stage_positions: None,
            band_trace: Vec::with_capacity(21),
        })
    }

    pub fn decode_packet(&mut self, data: &[u8]) -> Result<AudioSamples, MediaDecodeError> {
        if self.failed {
            return Err(MediaDecodeError::Unsupported);
        }
        self.failed = true;
        self.stage_positions = None;
        self.band_trace.clear();
        let packet = Packet::parse(data)?;
        if !(28..=31).contains(&packet.configuration)
            || packet.frames.len() != 1
            || packet.stereo != (self.channels == 2)
        {
            return Err(MediaDecodeError::Unsupported);
        }
        let layout = packet.celt_layout()?.ok_or(MediaDecodeError::Unsupported)?;
        let lm = super::mode(layout)? as i32;
        let coded_bins = layout.band(20)?.end;
        let shape = &mut self.shape;
        for channel in shape.iter_mut() {
            channel.fill(0.0);
        }
        let pcm = if packet.frames[0] == [0xff, 0xfe] {
            self.history.fill(-28.0);
            let normalized = [
                &shape[0][..coded_bins],
                shape.get(1).map_or(&[][..], |p| &p[..coded_bins]),
            ];
            self.synthesis.synthesize(
                SpectralFrame {
                    layout,
                    transient: false,
                    normalized: &normalized[..self.channels],
                    log_amplitudes: &self.history,
                },
                None,
            )?
        } else {
            let mut frame = FrameStart::parse(&packet, 0)?;
            let (mut start, positions) = frame.start_shapes_draft(&self.history)?;
            self.stage_positions = Some(positions);
            let blocks = layout.blocks(frame.transient)?;
            let norm = &mut self.norm;
            for channel in norm.iter_mut() {
                channel.fill(0.0);
            }
            let mut masks = [0u32; 42];
            let total = frame.entropy.frame_bytes() as i32 * 64
                - if start.anti_collapse_reserved { 8 } else { 0 };
            let mut balance = start.allocation.balance_eighths;
            let mut low_offset = 0;
            let mut update = true;
            let mut dual = start.allocation.dual_stereo;
            for band in 0..21 {
                let bins = layout.band(band)?;
                let size = bins.len();
                let tell = frame.entropy.tell_fractional() as i32;
                let errors_before = frame.entropy.uniform_errors();
                if band > 0 {
                    balance -= tell;
                }
                let remaining = total - tell - 1;
                let budget = if band < start.allocation.coded_bands {
                    (start.allocation.shape_eighths[band]
                        + balance / (start.allocation.coded_bands - band).min(3) as i32)
                        .min(remaining + 1)
                        .clamp(0, 16383)
                } else {
                    0
                };
                if bins.start >= size && (update || low_offset == 0) {
                    low_offset = band;
                }
                let source = if low_offset > 0
                    && (start.spread != 3 || blocks > 1 || start.tf.adjustments[band] < 0)
                {
                    Some(layout.band(low_offset)?.start.saturating_sub(size))
                } else {
                    None
                };
                let mut fills = [(1 << blocks) - 1; 2];
                if let Some(position) = source {
                    fills.fill(0);
                    for old in 0..band {
                        let old_bins = layout.band(old)?;
                        if old_bins.start < position + size && old_bins.end > position {
                            for c in 0..self.channels {
                                fills[c] |= masks[old * self.channels + c];
                            }
                        }
                    }
                }
                if dual && band == start.allocation.intensity {
                    dual = false;
                    for index in 0..bins.start {
                        norm[0][index] = (norm[0][index] + norm[1][index]) * 0.5;
                    }
                }
                let mut reader = ShapeReader {
                    entropy: &mut frame.entropy,
                    menus: &self.menus,
                    seed: &mut self.seed,
                    remaining,
                    spread: start.spread,
                    band,
                    intensity: start.allocation.intensity,
                    lm,
                };
                if self.channels == 2 && !dual {
                    let (left, right, mask, folded) = reader.stereo(
                        size,
                        budget,
                        blocks,
                        start.tf.adjustments[band],
                        source.map(|p| &norm[0][p..p + size]),
                        fills[0] | fills[1],
                    )?;
                    shape[0][bins.clone()].copy_from_slice(&left);
                    shape[1][bins.clone()].copy_from_slice(&right);
                    norm[0][bins.clone()].copy_from_slice(&folded);
                    masks[2 * band] = mask;
                    masks[2 * band + 1] = mask;
                } else {
                    for c in 0..self.channels {
                        let (values, mask) = reader.mono(
                            size,
                            budget / self.channels as i32,
                            blocks,
                            start.tf.adjustments[band],
                            1.0,
                            source.map(|p| &norm[c][p..p + size]),
                            fills[c],
                        )?;
                        shape[c][bins.clone()].copy_from_slice(&values);
                        for (out, value) in norm[c][bins.clone()].iter_mut().zip(values) {
                            *out = value * (size as f64).sqrt();
                        }
                        masks[band * self.channels + c] = mask;
                    }
                }
                balance += start.allocation.shape_eighths[band] + tell;
                self.band_trace.push(BandTrace {
                    band,
                    tf: start.tf.adjustments[band],
                    fine_bits: start.allocation.fine_bits[band],
                    budget,
                    tell_before: tell as u64,
                    tell_after: frame.entropy.tell_fractional(),
                    errors_before,
                    errors_after: frame.entropy.uniform_errors(),
                });
                update = budget > size as i32 * 8;
            }
            let anti = start.anti_collapse_reserved && frame.entropy.raw_bits(1)? != 0;
            let remaining = (frame.entropy.frame_bytes() as u64 * 8)
                .checked_sub(frame.entropy.tell())
                .ok_or_else(|| invalid("CELT packet entropy exceeds capacity"))?;
            let eligible: [bool; 21] =
                std::array::from_fn(|band| start.allocation.fine_bits[band] < 8);
            start.energy.finalize(
                &mut frame.entropy,
                &start.allocation.final_priorities,
                &eligible,
                remaining as usize,
            )?;
            if anti {
                let mut seed = self.seed;
                for band in 0..21 {
                    let bins = layout.band(band)?;
                    let size = bins.len();
                    let depth = (1 + start.allocation.shape_eighths[band]) / size as i32;
                    let limit = 0.5 * (-0.125 * depth as f64).exp2();
                    for c in 0..self.channels {
                        let index = band * self.channels + c;
                        let difference = (start.energy.log_energies()[index]
                            - self.log1[index].min(self.log2[index]))
                        .max(0.0);
                        let amplitude = limit
                            .min(2.0 * std::f64::consts::SQRT_2 * (-difference).exp2())
                            / (size as f64).sqrt();
                        let mut changed = false;
                        for block in 0..blocks {
                            if masks[index] & (1 << block) == 0 {
                                for bin in 0..size / blocks {
                                    shape[c][bins.start + bin * blocks + block] =
                                        if random(&mut seed) & 0x8000 != 0 {
                                            amplitude
                                        } else {
                                            -amplitude
                                        };
                                }
                                changed = true;
                            }
                        }
                        if changed {
                            unit(&mut shape[c][bins.clone()], 1.0);
                        }
                    }
                }
            }
            let normalized = [
                &shape[0][..coded_bins],
                shape.get(1).map_or(&[][..], |p| &p[..coded_bins]),
            ];
            let pcm = self.synthesis.synthesize_refined(
                frame.transient,
                &normalized[..self.channels],
                &start.energy,
                &BAND_MEANS,
                frame.pitch,
            )?;
            self.history.copy_from_slice(start.energy.log_energies());
            if frame.transient {
                for (last, current) in self.log1.iter_mut().zip(&self.history) {
                    *last = last.min(*current);
                }
            } else {
                self.log2.clone_from(&self.log1);
                self.log1.clone_from(&self.history);
            }
            self.seed = frame.entropy.final_range();
            self.uniform_errors += frame.entropy.uniform_errors();
            pcm
        };
        self.failed = false;
        Ok(AudioSamples {
            sample_rate: 48000,
            channels: self.channels as u16,
            samples: pcm
                .into_iter()
                // The shared spectral synthesis carries a 1/2 IMDCT factor.
                // Restore the packet convention, then RFC Appendix A SCALEOUT.
                .map(|sample| (sample * 2.0 * self.output_gain / 32768.0) as f32)
                .collect(),
        })
    }

    pub fn uniform_errors(&self) -> u32 {
        self.uniform_errors
    }
}

#[cfg(test)]
mod stage_diagnostics {
    use super::*;

    fn two_coordinate_vector(index: u32, pulses: usize) -> [f64; 2] {
        // RFC 6716 4.3.4.2 reduces to V(2,K)=4K, V(1,K)=2.
        let half = 2 * pulses as u32 + 1;
        let (local, sign) = if index < half {
            (index, 1.0)
        } else {
            (index - half, -1.0)
        };
        let y = local.div_ceil(2) as f64;
        let x = sign * (pulses as f64 - y);
        let y = if local & 1 == 0 { -y } else { y };
        let norm = (x * x + y * y).sqrt();
        [x / norm, y / norm]
    }

    #[test]
    fn two_bin_mono_pvq_matches_independent_coordinates_and_entropy() {
        let menus = menus().unwrap();
        let mut cases = 0;
        for lm in 0..=3 {
            for blocks in [1, 2] {
                for (budget, pulses) in [(16, 1), (24, 2), (32, 4), (40, 8)] {
                    for fill in 0..=254 {
                        let bytes = [fill; 64];
                        let mut entropy = RangeDecoder::new(&bytes);
                        let mut expected_entropy = entropy.clone();
                        let index = expected_entropy.uniform(4 * pulses as u32).unwrap();
                        let expected = two_coordinate_vector(index, pulses);
                        let mut seed = 0;
                        let mut reader = ShapeReader {
                            entropy: &mut entropy,
                            menus: &menus,
                            seed: &mut seed,
                            remaining: 4000,
                            spread: 0,
                            band: 0,
                            intensity: 21,
                            lm,
                        };
                        let (actual, mask) =
                            reader.vector(2, budget, blocks, lm, 1.0, None, 0).unwrap();
                        assert_eq!(reader.remaining, 4000 - budget);
                        assert_eq!(
                            entropy.tell_fractional(),
                            expected_entropy.tell_fractional()
                        );
                        for (value, expected) in actual.iter().zip(expected) {
                            assert!(value.is_finite());
                            assert!((value - expected).abs() < 1e-14);
                        }
                        assert_ne!(mask, 0);
                        assert_eq!(entropy.uniform_errors(), 0);
                        assert_eq!(
                            entropy.raw_bits(5).unwrap(),
                            expected_entropy.raw_bits(5).unwrap()
                        );
                        assert_eq!(entropy.bit(2).unwrap(), expected_entropy.bit(2).unwrap());
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 8160);
    }

    #[test]
    fn two_bin_stereo_geometry_has_only_one_side_orientation_bit() {
        // Per-channel normalization implies (L+R).(R-L)=|R|^2-|L|^2=0.
        // In two dimensions the side direction is consequently one of the
        // two perpendicular directions, not another independent PVQ vector.
        // This explains RFC 6716 5.3.5's N>2 extra-degree condition; it does
        // not establish the missing special-case bitstream ordering.
        let mut cases = 0;
        for left_index in 0..16 {
            for right_index in 0..16 {
                let left = two_coordinate_vector(left_index, 4);
                let right = two_coordinate_vector(right_index, 4);
                let mid = [left[0] + right[0], left[1] + right[1]];
                let side = [right[0] - left[0], right[1] - left[1]];
                assert!((mid[0] * side[0] + mid[1] * side[1]).abs() < 1e-14);
                let determinant = mid[0] * side[1] - mid[1] * side[0];
                let mid_energy = mid[0] * mid[0] + mid[1] * mid[1];
                let side_energy = side[0] * side[0] + side[1] * side[1];
                assert!((determinant * determinant - mid_energy * side_energy).abs() < 1e-13);
                cases += 1;
            }
        }
        assert_eq!(cases, 256);
    }
}
