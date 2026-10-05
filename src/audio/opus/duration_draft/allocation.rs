// Test-only draft copied from our decoder, never compiled for production.
//! CELT allocation setup and static vectors, RFC 6716 section 4.3.3.
//! Exact caps are mandatory inputs, not inferred from a signal or a rate.
//! Skipping, final redistribution and fine/shape splitting remain incomplete.

use super::{celt::Layout, invalid, range::RangeDecoder};
use crate::video::backend::MediaDecodeError;

pub(super) fn shape_degrees(
    channels: usize,
    dimensions: i32,
    dual_stereo: bool,
    below_intensity: bool,
) -> i32 {
    // RFC 6716 section 5.3.5: only bands with more than two MDCT bins
    // require the additional mid-side degree of freedom.
    channels as i32 * dimensions
        + i32::from(channels == 2 && dimensions > 2 && !dual_stereo && below_intensity)
}

// RFC 6716 Appendix A, static_modes_float.h, cache_caps50 row 7 (LM=3,C=2).
use super::CAPS;
// Eighth-bit logarithms of 2.5-ms band widths, logN400 in the same appendix.
const LOG_WIDTH: [i32; 21] = [
    0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 16, 16, 16, 21, 21, 24, 29, 34, 36,
];

pub use super::super::allocation::BandAllocation;

const STATIC: [[u8; 11]; 21] = [
    [0, 90, 110, 118, 126, 134, 144, 152, 162, 172, 200],
    [0, 80, 100, 110, 119, 127, 137, 145, 155, 165, 200],
    [0, 75, 90, 103, 112, 120, 130, 138, 148, 158, 200],
    [0, 69, 84, 93, 104, 114, 124, 132, 142, 152, 200],
    [0, 63, 78, 86, 95, 103, 113, 123, 133, 143, 200],
    [0, 56, 71, 80, 89, 97, 107, 117, 127, 137, 200],
    [0, 49, 65, 75, 83, 91, 101, 111, 121, 131, 200],
    [0, 40, 58, 70, 78, 85, 95, 105, 115, 125, 200],
    [0, 34, 51, 65, 72, 78, 88, 98, 108, 118, 198],
    [0, 29, 45, 59, 66, 72, 82, 92, 102, 112, 193],
    [0, 20, 39, 53, 60, 66, 76, 86, 96, 106, 188],
    [0, 18, 32, 47, 54, 60, 70, 80, 90, 100, 183],
    [0, 10, 26, 40, 47, 54, 64, 74, 84, 94, 178],
    [0, 0, 20, 31, 39, 47, 57, 67, 77, 87, 173],
    [0, 0, 12, 23, 32, 41, 51, 61, 71, 81, 168],
    [0, 0, 0, 15, 25, 35, 45, 55, 65, 75, 163],
    [0, 0, 0, 4, 17, 29, 39, 49, 59, 69, 158],
    [0, 0, 0, 0, 12, 23, 33, 43, 53, 63, 153],
    [0, 0, 0, 0, 1, 16, 26, 36, 46, 56, 148],
    [0, 0, 0, 0, 0, 10, 15, 20, 30, 45, 129],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 20, 104],
];

/// RFC 6716 section 4.3.3 and normative LOG2_FRAC_TABLE: intensity has
/// n+1 choices for n coded bands. ceil(8*log2(n+1)); integer arithmetic
/// gives the exact ceiling, including power-of-two boundaries, without a cache.
fn intensity_reservation(bands: usize) -> Result<u8, MediaDecodeError> {
    if !(1..=21).contains(&bands) {
        return Err(invalid("invalid CELT intensity band count"));
    }
    let power = (bands as u64 + 1).pow(8);
    Ok((u64::BITS - (power - 1).leading_zeros()) as u8)
}

/// Decode allocation trim after band boosts, using Table 58. Capacity and boost
/// are in eighth-bit units; insufficient capacity selects the neutral trim.
pub fn decode_trim(
    decoder: &mut RangeDecoder<'_>,
    frame_bytes: usize,
    boost_eighths: u64,
) -> Result<u8, MediaDecodeError> {
    let capacity = u64::try_from(frame_bytes)
        .ok()
        .and_then(|bytes| bytes.checked_mul(64))
        .and_then(|bits| bits.checked_sub(boost_eighths))
        .ok_or_else(|| invalid("invalid CELT trim capacity"))?;
    if decoder.tell_fractional().saturating_add(48) > capacity {
        return Ok(5);
    }
    Ok(decoder.symbol(&[2, 2, 5, 10, 22, 46, 22, 10, 5, 2, 2])? as u8)
}

/// Unadjusted table row at an integer quality 0..=10, in eighths of a bit.
/// Return order follows the layout's active band range, including Hybrid.
pub fn static_vector(
    layout: Layout,
    channels: usize,
    quality: usize,
) -> Result<Vec<u32>, MediaDecodeError> {
    if !(1..=2).contains(&channels) || quality > 10 {
        return Err(invalid("invalid CELT static allocation input"));
    }
    layout
        .bands()
        .map(|band| {
            let width = layout.band(band)?.len() as u32;
            Ok(channels as u32 * width * u32::from(STATIC[band][quality]) / 4)
        })
        .collect()
}

/// Apply the normative cap conversion to one exact cache row selected by the
/// caller for this frame duration and channel count. No cache defaults exist.
pub fn caps_from_cache(
    layout: Layout,
    channels: usize,
    row: &[u8],
) -> Result<Vec<u16>, MediaDecodeError> {
    if !(1..=2).contains(&channels) || row.len() != 21 {
        return Err(invalid("invalid CELT allocation cap row"));
    }
    layout
        .bands()
        .map(|band| {
            let bins = layout.band(band)?.len();
            let cap = (usize::from(row[band]) + 64) * channels * bins / 4;
            u16::try_from(cap).map_err(|_| invalid("CELT allocation cap overflow"))
        })
        .collect()
}

/// Result of the RFC's allocation preparation, not final per-band shape bits.
/// The remaining allocation search must use these boosts, thresholds, trims,
/// caps and reservations rather than re-reading the symbols.
#[derive(Clone, Debug)]
pub struct AllocationPreparation {
    pub caps: Vec<u16>,
    pub boosts_eighths: Vec<u32>,
    pub total_boost_eighths: u64,
    pub trim: u8,
    pub thresholds_eighths: Vec<u32>,
    pub trim_offsets_eighths: Vec<i32>,
    pub remaining_eighths: u64,
    pub anti_collapse_reserved_eighths: u8,
    pub skip_reserved_eighths: u8,
    pub intensity_reserved_eighths: u8,
    pub dual_stereo_reserved_eighths: u8,
}

impl AllocationPreparation {
    pub fn decode_fixed(
        entropy: &mut RangeDecoder<'_>,
        layout: Layout,
        channels: usize,
        transient: bool,
    ) -> Result<Self, MediaDecodeError> {
        if !(1..=2).contains(&channels) {
            return Err(invalid("invalid draft channels"));
        }
        let row = &CAPS[2 * super::mode(layout)? + channels - 1];
        let caps = caps_from_cache(layout, channels, row)?;
        Self::decode(entropy, layout, channels, transient, &caps)
    }

    /// RFC 6716 section 4.3.3 and Appendix A's normative allocation rules.
    /// Integer search, explicit skip decisions, redistribution, then the fine
    /// energy share. Shape budgets remain eighth bits, not guessed pulse counts.
    pub fn finish(
        &self,
        entropy: &mut RangeDecoder<'_>,
        layout: Layout,
        channels: usize,
    ) -> Result<BandAllocation, MediaDecodeError> {
        if super::mode(layout).is_err()
            || layout.bands() != (0..21)
            || !(1..=2).contains(&channels)
            || self.caps.len() != 21
            || self.boosts_eighths.len() != 21
            || self.thresholds_eighths.len() != 21
            || self.trim_offsets_eighths.len() != 21
        {
            return Err(invalid("invalid CELT final allocation input"));
        }
        let lm = super::mode(layout)?;
        let scale = 1usize << lm;
        let widths: Vec<i32> = (0..21)
            .map(|band| layout.band(band).map(|r| (r.len() / scale) as i32))
            .collect::<Result<_, _>>()?;
        let floor = channels as i32 * 8;
        let mut total = i32::try_from(self.remaining_eighths)
            .map_err(|_| invalid("CELT allocation budget overflow"))?;
        let curve = |quality: usize, include_boost: bool| -> Vec<i32> {
            (0..21)
                .map(|band| {
                    let raw = if quality > 10 {
                        i32::from(self.caps[band])
                    } else {
                        (channels as i32
                            * widths[band]
                            * i32::from(STATIC[band][quality])
                            * scale as i32)
                            >> 2
                    };
                    let trimmed = if raw == 0 {
                        0
                    } else {
                        (raw + self.trim_offsets_eighths[band]).max(0)
                    };
                    trimmed
                        + if include_boost {
                            self.boosts_eighths[band] as i32
                        } else {
                            0
                        }
                })
                .collect()
        };
        let constrain = |values: &[i32]| -> Vec<i32> {
            let mut significant = false;
            let mut result = vec![0; 21];
            for band in (0..21).rev() {
                significant |= values[band] >= self.thresholds_eighths[band] as i32;
                result[band] = if significant {
                    values[band].min(i32::from(self.caps[band]))
                } else if values[band] >= floor {
                    floor.min(i32::from(self.caps[band]))
                } else {
                    0
                };
            }
            result
        };
        let mut upper = 1;
        while upper <= 10 && constrain(&curve(upper, true)).iter().sum::<i32>() <= total {
            upper += 1;
        }
        let lower = curve(upper - 1, upper > 1);
        let higher = curve(upper, true);
        let differences: Vec<_> = higher
            .iter()
            .zip(&lower)
            .map(|(h, l)| (h - l).max(0))
            .collect();
        let interpolate = |fraction: i32| -> Vec<i32> {
            lower
                .iter()
                .zip(&differences)
                .map(|(l, d)| l + ((fraction * d) >> 6))
                .collect()
        };
        let mut fraction = 0;
        for step in [32, 16, 8, 4, 2, 1] {
            if constrain(&interpolate(fraction + step)).iter().sum::<i32>() <= total {
                fraction += step;
            }
        }
        let mut bits = constrain(&interpolate(fraction));
        let mut used = bits.iter().sum::<i32>();
        let skip_limit = self
            .boosts_eighths
            .iter()
            .rposition(|&boost| boost > 0)
            .unwrap_or(0);
        let mut cursor = entropy.clone();
        let mut coded = 21;
        let mut intensity_reserve = i32::from(self.intensity_reserved_eighths);
        loop {
            let band = coded - 1;
            if band <= skip_limit {
                total += i32::from(self.skip_reserved_eighths);
                break;
            }
            let width_sum: i32 = widths[..coded].iter().sum();
            let left = total - used;
            let per_bin = left / width_sum;
            let remainder = left - width_sum * per_bin;
            let prefix: i32 = widths[..band].iter().sum();
            let mut available = bits[band] + per_bin * widths[band] + (remainder - prefix).max(0);
            if available >= (self.thresholds_eighths[band] as i32).max(floor + 8) {
                if cursor.bit(1)? {
                    break;
                }
                used += 8;
                available -= 8;
            }
            used -= bits[band] + intensity_reserve;
            if intensity_reserve > 0 {
                intensity_reserve = i32::from(intensity_reservation(band)?);
            }
            used += intensity_reserve;
            bits[band] = if available >= floor { floor } else { 0 };
            used += bits[band];
            coded -= 1;
        }
        let intensity = if intensity_reserve > 0 {
            cursor.uniform((coded + 1) as u32)? as usize
        } else {
            0
        };
        let dual_stereo = if intensity == 0 {
            total += i32::from(self.dual_stereo_reserved_eighths);
            false
        } else {
            self.dual_stereo_reserved_eighths > 0 && cursor.bit(1)?
        };
        let left = total - used;
        if left < 0 {
            return Err(invalid("CELT allocation overspent budget"));
        }
        let width_sum: i32 = widths[..coded].iter().sum();
        let per_bin = left / width_sum;
        let mut remainder = left % width_sum;
        for band in 0..coded {
            let extra = remainder.min(widths[band]);
            bits[band] += per_bin * widths[band] + extra;
            remainder -= extra;
        }
        let mut fine = vec![0u8; 21];
        let mut priority = vec![0u8; 21];
        let mut balance = 0;
        for band in 0..coded {
            bits[band] += balance;
            let mut excess = (bits[band] - i32::from(self.caps[band])).max(0);
            bits[band] -= excess;
            let dimensions = widths[band] * scale as i32;
            if dimensions == 1 {
                return Err(MediaDecodeError::Unsupported);
            }
            let degrees = shape_degrees(channels, dimensions, dual_stereo, band < intensity);
            let log_dimensions = degrees * (LOG_WIDTH[band] + lm as i32 * 8);
            let mut offset = (log_dimensions >> 1) - degrees * 21;
            if bits[band] + offset < degrees * 16 {
                offset += log_dimensions >> 2;
            } else if bits[band] + offset < degrees * 24 {
                offset += log_dimensions >> 3;
            }
            let chosen = ((bits[band] + offset + degrees * 4) / (degrees * 8))
                .max(0)
                .min(bits[band] / floor)
                .min(8);
            fine[band] = chosen as u8;
            priority[band] = u8::from(chosen * degrees * 8 >= bits[band] + offset);
            bits[band] -= floor * chosen;
            if excess > 0 {
                let more = (excess / floor).min(8 - chosen);
                fine[band] += more as u8;
                priority[band] = u8::from(more * floor >= excess - balance);
                excess -= more * floor;
            }
            balance = excess;
        }
        for band in coded..21 {
            fine[band] = (bits[band] / floor) as u8;
            priority[band] = u8::from(fine[band] < 1);
            bits[band] = 0;
        }
        *entropy = cursor;
        Ok(BandAllocation {
            shape_eighths: bits,
            fine_bits: fine,
            final_priorities: priority,
            coded_bands: coded,
            intensity,
            dual_stereo,
            balance_eighths: balance,
        })
    }

    /// Enter immediately after spread. Implements the prose's band-boost loop,
    /// trim symbol, thresholds and reservations. The caller supplies exact caps
    /// while the intensity reservation is derived from the coded band count.
    /// Restricted to 20-ms CELT to avoid guessing signed trim rounding in the
    /// smaller modes. Stereo boost interpretation still needs a PCM oracle once
    /// upstream coarse energy is available; no bit-exact full decoder claim.
    pub fn decode(
        entropy: &mut RangeDecoder<'_>,
        layout: Layout,
        channels: usize,
        transient: bool,
        caps: &[u16],
    ) -> Result<Self, MediaDecodeError> {
        if super::mode(layout).is_err() || layout.bands().start != 0 {
            return Err(MediaDecodeError::Unsupported);
        }
        if !(1..=2).contains(&channels)
            || caps.len() != layout.bands().len()
            || caps.iter().any(|&cap| cap > 32767)
        {
            return Err(invalid("invalid CELT allocation preparation input"));
        }
        let intensity = if channels == 2 {
            intensity_reservation(layout.bands().len())?
        } else {
            0
        };
        let capacity = entropy.frame_bytes() as u64 * 64;
        if entropy.tell_fractional() > capacity {
            return Err(invalid("CELT allocation starts beyond frame budget"));
        }
        let mut cursor = entropy.clone();
        let mut boosts = Vec::with_capacity(caps.len());
        let mut total_boost = 0u64;
        let mut logp = 6u8;
        for (index, band) in layout.bands().enumerate() {
            // N is the per-channel MDCT width specified in section 4.3.3.
            let n = layout.band(band)?.len() as u32 * channels as u32;
            let quantum = (8 * n).min(48u32.max(n));
            let mut boost = 0u32;
            let mut loop_logp = logp;
            // The prose's total_bits decreases by each quantum and its
            // total_boost increases by that quantum, so their sum is capacity.
            while cursor.tell_fractional() + u64::from(loop_logp) * 8
                < capacity.saturating_sub(total_boost)
                && boost < u32::from(caps[index])
            {
                if !cursor.bit(loop_logp)? {
                    break;
                }
                boost += quantum;
                total_boost += u64::from(quantum);
                loop_logp = 1;
            }
            if boost != 0 && logp > 2 {
                logp -= 1;
            }
            boosts.push(boost);
        }
        let trim = decode_trim(&mut cursor, entropy.frame_bytes(), total_boost)?;
        let mut remaining = capacity.saturating_sub(cursor.tell_fractional().saturating_add(1));
        let lm = super::mode(layout)?;
        let anti = if transient && lm > 1 && remaining >= (lm as u64 + 2) * 8 {
            8
        } else {
            0
        };
        remaining = remaining.saturating_sub(u64::from(anti));
        let skip = if remaining >= 8 { 8 } else { 0 };
        remaining -= u64::from(skip);
        let mut intensity_reserved = 0;
        let mut dual = 0;
        if channels == 2 && u64::from(intensity) <= remaining {
            intensity_reserved = intensity;
            remaining -= u64::from(intensity);
            if remaining >= 8 {
                dual = 8;
                remaining -= 8;
            }
        }
        let mut thresholds = Vec::with_capacity(caps.len());
        let mut trims = Vec::with_capacity(caps.len());
        let end = layout.bands().end;
        for band in layout.bands() {
            let n = layout.band(band)?.len() as i32;
            thresholds.push(((24 * n) / 16).max(8 * channels as i32) as u32);
            // RFC allocation trim uses floor division for the signed shift.
            let offset = ((i32::from(trim) - 5 - lm as i32)
                * channels as i32
                * n
                * (end - band - 1) as i32
                * 8)
                >> 6;
            trims.push(offset - if n == 1 { 8 * channels as i32 } else { 0 });
        }
        let result = Self {
            caps: caps.to_vec(),
            boosts_eighths: boosts,
            total_boost_eighths: total_boost,
            trim,
            thresholds_eighths: thresholds,
            trim_offsets_eighths: trims,
            remaining_eighths: remaining,
            anti_collapse_reserved_eighths: anti,
            skip_reserved_eighths: skip,
            intensity_reserved_eighths: intensity_reserved,
            dual_stereo_reserved_eighths: dual,
        };
        *entropy = cursor;
        Ok(result)
    }
}
