//! Frame-confined Opus entropy decoding, from RFC 6716 section 4.1.

use crate::video::backend::MediaDecodeError;

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.into())
}

#[derive(Clone, Debug)]
pub struct RangeDecoder<'a> {
    bytes: &'a [u8],
    front: usize,
    raw_used: usize,
    low_bit: u8,
    range: u32,
    value: u32,
    total_bits: u64,
    uniform_errors: u32,
}

impl<'a> RangeDecoder<'a> {
    pub fn frame_bytes(&self) -> usize {
        self.bytes.len()
    }

    pub fn final_range(&self) -> u32 {
        self.range
    }

    pub fn uniform_errors(&self) -> u32 {
        self.uniform_errors
    }

    pub fn new(bytes: &'a [u8]) -> Self {
        let first = bytes.first().copied().unwrap_or(0);
        let mut decoder = Self {
            bytes,
            front: 1,
            raw_used: 0,
            low_bit: first & 1,
            range: 128,
            value: 127 - u32::from(first >> 1),
            total_bits: 9,
            uniform_errors: 0,
        };
        decoder.normalize();
        decoder
    }

    fn normalize(&mut self) {
        while self.range <= 1 << 23 {
            self.range <<= 8;
            let next = self.bytes.get(self.front).copied().unwrap_or(0);
            self.front = self.front.saturating_add(1);
            let symbol = (self.low_bit << 7) | (next >> 1);
            self.low_bit = next & 1;
            self.value = (self.value.wrapping_shl(8) + u32::from(255 - symbol)) & 0x7fff_ffff;
            self.total_bits += 8;
        }
    }

    fn frequency(&self, total: u32) -> u32 {
        total - (self.value / (self.range / total) + 1).min(total)
    }

    fn update(&mut self, low: u32, high: u32, total: u32) {
        let unit = self.range / total;
        let tail = unit * (total - high);
        self.value -= tail;
        self.range = if low == 0 {
            self.range - tail
        } else {
            unit * (high - low)
        };
        self.normalize();
    }

    /// Decode a symbol from integer frequencies (including zero-frequency entries).
    pub fn symbol(&mut self, frequencies: &[u16]) -> Result<usize, MediaDecodeError> {
        let total = frequencies
            .iter()
            .try_fold(0u32, |sum, &frequency| {
                sum.checked_add(u32::from(frequency))
            })
            .ok_or_else(|| invalid("Opus frequency total overflow"))?;
        if total == 0 || total > 65535 {
            return Err(invalid("invalid Opus frequency total"));
        }
        let target = self.frequency(total);
        let mut low = 0;
        for (symbol, &frequency) in frequencies.iter().enumerate() {
            let high = low + u32::from(frequency);
            if target < high {
                self.update(low, high, total);
                return Ok(symbol);
            }
            low = high;
        }
        unreachable!("validated frequency interval covers the target")
    }

    /// Decode a rare one with probability 2^-log_probability.
    /// Integer Laplace PDF from RFC 6716 Appendix A, celt/laplace.c.
    /// Enumerate signed intervals in wire order; the tail retains unit mass.
    pub fn laplace(&mut self, zero: u16, decay: u16) -> Result<i16, MediaDecodeError> {
        let zero = u32::from(zero);
        let decay = u32::from(decay);
        if zero == 0 || zero > 32736 || decay >= 16384 {
            return Err(invalid("invalid Opus Laplace parameters"));
        }
        let target = self.frequency(32768);
        if target < zero {
            self.update(0, zero, 32768);
            return Ok(0);
        }
        let mut boundary = zero;
        let mut mass = ((32736 - zero) * (16384 - decay) >> 15) + 1;
        for magnitude in 1..=16384i16 {
            for value in [-magnitude, magnitude] {
                let end = (boundary + mass).min(32768);
                if target < end {
                    self.update(boundary, end, 32768);
                    return Ok(value);
                }
                boundary = end;
            }
            mass = ((2 * (mass - 1) * decay) >> 15) + 1;
        }
        Err(invalid("invalid Opus Laplace interval"))
    }

    /// Decode a rare one with probability 2^-log_probability.
    pub fn bit(&mut self, log_probability: u8) -> Result<bool, MediaDecodeError> {
        if !(1..=15).contains(&log_probability) {
            return Err(invalid("invalid Opus bit probability"));
        }
        let unit = self.range >> log_probability;
        let one = self.value < unit;
        if one {
            self.range = unit;
        } else {
            self.value -= unit;
            self.range -= unit;
        }
        self.normalize();
        Ok(one)
    }

    pub fn inverse_cdf(&mut self, cdf: &[u8], precision: u8) -> Result<usize, MediaDecodeError> {
        if !(1..=8).contains(&precision) {
            return Err(invalid("invalid Opus CDF precision"));
        }
        let total = 1u32 << precision;
        if cdf.last() != Some(&0)
            || cdf.first().is_none_or(|&first| u32::from(first) >= total)
            || cdf.windows(2).any(|pair| pair[0] < pair[1])
        {
            return Err(invalid("invalid Opus inverse CDF"));
        }
        let unit = self.range >> precision;
        let mut previous = self.range;
        for (symbol, &entry) in cdf.iter().enumerate() {
            let threshold = unit * u32::from(entry);
            if self.value >= threshold {
                self.value -= threshold;
                self.range = previous - threshold;
                self.normalize();
                return Ok(symbol);
            }
            previous = threshold;
        }
        unreachable!("validated CDF ends at zero")
    }

    /// Raw bits run backwards through the frame, least-significant bit first.
    /// Front entropy read-ahead is intentionally allowed to overlap this input.
    pub fn raw_bits(&mut self, count: u8) -> Result<u32, MediaDecodeError> {
        let end = self
            .raw_used
            .checked_add(usize::from(count))
            .ok_or_else(|| invalid("Opus raw bit offset overflow"))?;
        if count > 32 || end > self.bytes.len().saturating_mul(8) {
            return Err(invalid("truncated Opus raw bits"));
        }
        let mut result = 0u32;
        let mut written = 0;
        while self.raw_used < end {
            let offset = self.raw_used & 7;
            let take = (8 - offset).min(end - self.raw_used);
            let byte = self.bytes[self.bytes.len() - 1 - self.raw_used / 8];
            let mask = (1u32 << take) - 1;
            result |= ((u32::from(byte) >> offset) & mask) << written;
            self.raw_used += take;
            written += take;
        }
        self.total_bits += u64::from(count);
        Ok(result)
    }

    pub fn uniform(&mut self, total: u32) -> Result<u32, MediaDecodeError> {
        if total == 0 {
            return Err(invalid("zero Opus uniform total"));
        }
        if total == 1 {
            return Ok(0);
        }
        let extra = (32 - (total - 1).leading_zeros()).saturating_sub(8);
        let high_total = ((total - 1) >> extra) + 1;
        let high = self.frequency(high_total);
        self.update(high, high + 1, high_total);
        let result = (high << extra) | self.raw_bits(extra as u8)?;
        if result >= total {
            // RFC 6716 normative ec_dec_uint recovers to the final entry and
            // records a soft error. Preserve that evidence for packet checks.
            self.uniform_errors += 1;
            return Ok(total - 1);
        }
        Ok(result)
    }

    pub fn tell(&self) -> u64 {
        self.total_bits - u64::from(32 - self.range.leading_zeros())
    }

    /// Conservative consumed-bit estimate in eighths of a bit.
    pub fn tell_fractional(&self) -> u64 {
        let mut log = 32 - self.range.leading_zeros();
        let mut fraction = u64::from(self.range >> (log - 16));
        for _ in 0..3 {
            fraction = (fraction * fraction) >> 15;
            let bit = fraction >> 16;
            log = 2 * log + bit as u32;
            fraction >>= bit;
        }
        self.total_bits * 8 - u64::from(log)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn same_state(a: &RangeDecoder<'_>, b: &RangeDecoder<'_>) {
        assert_eq!(
            (a.range, a.value, a.front, a.total_bits),
            (b.range, b.value, b.front, b.total_bits)
        );
    }

    #[test]
    fn initialization_zero_extends_short_frames() {
        let data = [0xab, 0xcd, 0xef, 0x12];
        for length in 0..=4 {
            let mut padded = [0; 4];
            padded[..length].copy_from_slice(&data[..length]);
            let decoder = RangeDecoder::new(&data[..length]);
            assert_eq!(decoder.range, 1 << 31);
            assert_eq!(
                decoder.value,
                0x7fff_ffff - (u32::from_be_bytes(padded) >> 1)
            );
            assert_eq!(decoder.tell(), 1);
            assert_eq!(decoder.tell_fractional(), 8);
        }
    }

    #[test]
    fn optimized_contexts_match_general_frequencies() {
        for seed in 0u32..128 {
            let data: Vec<_> = (0..64).map(|i| (seed * 73 + i * 137) as u8).collect();
            for precision in 1..=15 {
                let mut fast = RangeDecoder::new(&data);
                let mut general = fast.clone();
                for _ in 0..64 {
                    let symbol = general
                        .symbol(&[((1u32 << precision) - 1) as u16, 1])
                        .unwrap();
                    assert_eq!(fast.bit(precision).unwrap(), symbol == 1);
                    same_state(&fast, &general);
                    assert_eq!((fast.tell_fractional() + 7) / 8, fast.tell());
                    assert!(fast.value < fast.range);
                }
            }
            for precision in 1..=8 {
                let total = 1u16 << precision;
                let frequencies = [1, total / 2 - 1, 0, total / 2];
                let cdf = [(total - 1) as u8, (total / 2) as u8, (total / 2) as u8, 0];
                let mut fast = RangeDecoder::new(&data);
                let mut general = fast.clone();
                for _ in 0..64 {
                    assert_eq!(
                        fast.inverse_cdf(&cdf, precision).unwrap(),
                        general.symbol(&frequencies).unwrap()
                    );
                    same_state(&fast, &general);
                }
            }
        }
    }

    #[test]
    fn raw_bits_overlap_entropy_read_ahead() {
        let mut decoder = RangeDecoder::new(&[0xab, 0xcd]);
        assert_eq!(decoder.raw_bits(4).unwrap(), 0xd);
        assert_eq!(decoder.raw_bits(8).unwrap(), 0xbc);
        assert_eq!(decoder.raw_bits(4).unwrap(), 0xa);
        assert_eq!(decoder.raw_bits(0).unwrap(), 0);
        assert!(decoder.raw_bits(1).is_err());
        let mut decoder = RangeDecoder::new(&[0x12, 0x34, 0x56, 0x78]);
        assert_eq!(decoder.raw_bits(32).unwrap(), 0x12345678);
    }

    #[test]
    fn uniform_matches_general_and_rejects_invalid_contexts() {
        let data = [0xa7; 128];
        for total in 1..=256 {
            let mut fast = RangeDecoder::new(&data);
            let mut general = fast.clone();
            for _ in 0..16 {
                let value = fast.uniform(total).unwrap();
                if total > 1 {
                    assert_eq!(
                        value as usize,
                        general.symbol(&vec![1; total as usize]).unwrap()
                    );
                }
                same_state(&fast, &general);
            }
        }
        let mut decoder = RangeDecoder::new(&data);
        let initial = decoder.clone();
        assert!(decoder.uniform(0).is_err());
        assert!(decoder.symbol(&[]).is_err());
        assert!(decoder.symbol(&[65535, 1]).is_err());
        assert!(decoder.bit(0).is_err());
        assert!(decoder.inverse_cdf(&[128, 0], 7).is_err());
        assert!(decoder.inverse_cdf(&[1, 2, 0], 8).is_err());
        assert!(decoder.raw_bits(33).is_err());
        same_state(&decoder, &initial);
    }
}
