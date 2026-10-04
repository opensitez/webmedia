//! Integer arithmetic symbol decoder, AV1 specification section 8.2.

use super::syntax::{Bits, Error};

pub struct SymbolDecoder<'a> {
    tile: &'a [u8],
    bits: Bits<'a>,
    range: u32,
    value: u32,
    remaining: i64,
    update_cdf: bool,
}

impl<'a> SymbolDecoder<'a> {
    pub fn new(tile: &'a [u8], update_cdf: bool) -> Result<Self, Error> {
        if tile.is_empty() || tile.len() > 64 * 1024 * 1024 {
            return Err(Error::Invalid("tile entropy size"));
        }
        let mut bits = Bits::new(tile);
        let n = (tile.len() * 8).min(15) as u8;
        let prefix = bits.read(n)? << (15 - n);
        Ok(Self {
            tile,
            bits,
            range: 32768,
            value: 32767 ^ prefix,
            remaining: tile.len() as i64 * 8 - 15,
            update_cdf,
        })
    }

    /// CDF is cumulative (ascending), with terminal 32768 and an adaptation count.
    pub fn read_symbol(&mut self, cdf: &mut [u16]) -> Result<usize, Error> {
        let n = cdf
            .len()
            .checked_sub(1)
            .ok_or(Error::Invalid("CDF length"))?;
        if !(2..=16).contains(&n)
            || cdf[n - 1] != 32768
            || cdf[n] > 32
            || cdf[..n].windows(2).any(|w| w[0] > w[1])
        {
            return Err(Error::Invalid("CDF values"));
        }
        let mut previous = self.range;
        let mut symbol = 0;
        let current = loop {
            let inverse_probability = 32768 - u32::from(cdf[symbol]);
            // EC_PROB_SHIFT=6 and EC_MIN_PROB=4, section 3.
            let threshold =
                ((self.range >> 8) * (inverse_probability >> 6) >> 1) + 4 * (n - symbol - 1) as u32;
            if self.value >= threshold {
                break threshold;
            }
            previous = threshold;
            symbol += 1;
        };
        self.range = previous - current;
        self.value -= current;
        let shift = (self.range.leading_zeros() - 16) as u8;
        self.range <<= shift;
        let available = self.remaining.max(0).min(i64::from(shift)) as u8;
        let padded = self.bits.read(available)? << (shift - available);
        self.value = padded ^ (((self.value + 1) << shift) - 1);
        self.remaining -= i64::from(shift);
        if self.remaining < -14 {
            return Err(Error::Truncated);
        }
        if self.update_cdf {
            let count = cdf[n];
            let rate = 3
                + u16::from(count > 15)
                + u16::from(count > 31)
                + (usize::BITS - 1 - n.leading_zeros()).min(2) as u16;
            for (i, value) in cdf[..n - 1].iter_mut().enumerate() {
                if i < symbol {
                    *value -= *value >> rate;
                } else {
                    *value += (32768 - *value) >> rate;
                }
            }
            cdf[n] += u16::from(count < 32);
        }
        Ok(symbol)
    }

    pub fn read_bool(&mut self) -> Result<bool, Error> {
        Ok(self.read_symbol(&mut [16384, 32768, 0])? != 0)
    }

    pub fn read_literal(&mut self, n: u8) -> Result<u32, Error> {
        if n > 32 {
            return Err(Error::Invalid("literal width"));
        }
        let mut value = 0;
        for _ in 0..n {
            value = (value << 1) | u32::from(self.read_bool()?);
        }
        Ok(value)
    }

    /// Validate tile termination only after all symbols have been consumed.
    pub fn finish(&self) -> Result<(), Error> {
        let tile = self.tile;
        if self.remaining < -14
            || tile.len() as i64 * 8 != self.bits.position as i64 + self.remaining.max(0)
        {
            return Err(Error::Invalid("entropy termination state"));
        }
        let trailing = self.bits.position as i64 - (self.remaining + 15).min(15);
        if trailing < 0 || trailing as usize >= tile.len() * 8 {
            return Err(Error::Truncated);
        }
        let mut b = Bits::new(tile);
        b.position = trailing as usize;
        if !b.flag()? {
            return Err(Error::Invalid("entropy trailing one bit"));
        }
        while b.position < tile.len() * 8 {
            if b.flag()? {
                return Err(Error::Invalid("entropy trailing zero bit"));
            }
        }
        Ok(())
    }
}

/// Default interior partition contexts from normative table 9.3. Context is
/// `above_smaller + 2 * left_smaller`; callers must handle frame-edge alphabets.
#[derive(Clone)]
pub struct PartitionCdfs {
    w8: [[u16; 5]; 4],
    w16: [[u16; 11]; 4],
    w32: [[u16; 11]; 4],
    w64: [[u16; 11]; 4],
    w128: [[u16; 9]; 4],
}

impl PartitionCdfs {
    pub(crate) fn reset_counts(&mut self) {
        for row in &mut self.w8 {
            row[4] = 0;
        }
        for rows in [&mut self.w16, &mut self.w32, &mut self.w64] {
            for row in rows {
                row[10] = 0;
            }
        }
        for row in &mut self.w128 {
            row[8] = 0;
        }
    }
}

impl Default for PartitionCdfs {
    fn default() -> Self {
        Self {
            w8: [
                [19132, 25510, 30392, 32768, 0],
                [13928, 19855, 28540, 32768, 0],
                [12522, 23679, 28629, 32768, 0],
                [9896, 18783, 25853, 32768, 0],
            ],
            w16: [
                [
                    15597, 20929, 24571, 26706, 27664, 28821, 29601, 30571, 31902, 32768, 0,
                ],
                [
                    7925, 11043, 16785, 22470, 23971, 25043, 26651, 28701, 29834, 32768, 0,
                ],
                [
                    5414, 13269, 15111, 20488, 22360, 24500, 25537, 26336, 32117, 32768, 0,
                ],
                [
                    2662, 6362, 8614, 20860, 23053, 24778, 26436, 27829, 31171, 32768, 0,
                ],
            ],
            w32: [
                [
                    18462, 20920, 23124, 27647, 28227, 29049, 29519, 30178, 31544, 32768, 0,
                ],
                [
                    7689, 9060, 12056, 24992, 25660, 26182, 26951, 28041, 29052, 32768, 0,
                ],
                [
                    6015, 9009, 10062, 24544, 25409, 26545, 27071, 27526, 32047, 32768, 0,
                ],
                [
                    1394, 2208, 2796, 28614, 29061, 29466, 29840, 30185, 31899, 32768, 0,
                ],
            ],
            w64: [
                [
                    20137, 21547, 23078, 29566, 29837, 30261, 30524, 30892, 31724, 32768, 0,
                ],
                [
                    6732, 7490, 9497, 27944, 28250, 28515, 28969, 29630, 30104, 32768, 0,
                ],
                [
                    5945, 7663, 8348, 28683, 29117, 29749, 30064, 30298, 32238, 32768, 0,
                ],
                [
                    870, 1212, 1487, 31198, 31394, 31574, 31743, 31881, 32332, 32768, 0,
                ],
            ],
            w128: [
                [27899, 28219, 28529, 32484, 32539, 32619, 32639, 32768, 0],
                [6607, 6990, 8268, 32060, 32219, 32338, 32371, 32768, 0],
                [5429, 6676, 7122, 32027, 32227, 32531, 32582, 32768, 0],
                [711, 966, 1172, 32448, 32538, 32617, 32664, 32768, 0],
            ],
        }
    }
}

impl PartitionCdfs {
    pub fn read_partition(
        &mut self,
        decoder: &mut SymbolDecoder<'_>,
        block_width: usize,
        context: usize,
        has_rows: bool,
        has_cols: bool,
    ) -> Result<usize, Error> {
        if block_width == 4 {
            return Ok(0);
        }
        if !has_rows && !has_cols {
            return Ok(3);
        }
        if has_rows && has_cols {
            return self.read_interior(decoder, block_width, context);
        }
        if context > 3 || block_width == 8 {
            return Err(Error::Invalid("edge partition context"));
        }
        let probabilities: &[u16] = match block_width {
            16 => &self.w16[context],
            32 => &self.w32[context],
            64 => &self.w64[context],
            128 => &self.w128[context],
            _ => return Err(Error::Invalid("partition width")),
        };
        let symbols: &[usize] = if has_cols {
            &[2, 3, 4, 6, 7, 9]
        } else {
            &[1, 3, 4, 5, 6, 8]
        };
        let mut mass = 0;
        for &symbol in symbols {
            if symbol >= probabilities.len() - 1 {
                continue;
            }
            mass += probabilities[symbol] - probabilities[symbol - 1];
        }
        let mut cdf = [32768 - mass, 32768, 0];
        let split = decoder.read_symbol(&mut cdf)?;
        Ok(if split != 0 {
            3
        } else if has_cols {
            1
        } else {
            2
        })
    }
    /// Return the normative PARTITION_* index; this does not decode block modes.
    pub fn read_interior(
        &mut self,
        decoder: &mut SymbolDecoder<'_>,
        block_width: usize,
        context: usize,
    ) -> Result<usize, Error> {
        if context > 3 {
            return Err(Error::Invalid("partition context"));
        }
        let cdf: &mut [u16] = match block_width {
            8 => &mut self.w8[context],
            16 => &mut self.w16[context],
            32 => &mut self.w32[context],
            64 => &mut self.w64[context],
            128 => &mut self.w128[context],
            _ => return Err(Error::Invalid("partition width")),
        };
        decoder.read_symbol(cdf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_first_symbol_exhaustive_integer_interval_oracle() {
        // Independent interval calculation: range=2^15, quantized half-range
        // plus the minimum-probability correction yields the split at 16388.
        for raw in 0u32..32768 {
            let bytes = ((raw << 1) as u16).to_be_bytes();
            let mut decoder = SymbolDecoder::new(&bytes, false).unwrap();
            assert_eq!(
                decoder.read_bool().unwrap(),
                32767 - raw < 16388,
                "prefix {raw}"
            );
        }
    }

    #[test]
    fn adaptation_count_and_frozen_cdf() {
        let mut cdf = [16384, 32768, 0];
        let mut d = SymbolDecoder::new(&[0; 128], true).unwrap();
        assert_eq!(d.read_symbol(&mut cdf).unwrap(), 0);
        assert_eq!(cdf, [17408, 32768, 1]);
        for _ in 0..64 {
            d.read_symbol(&mut cdf).unwrap();
        }
        assert_eq!(cdf[2], 32);
        let mut frozen = [1000, 2000, 32768, 0];
        let mut d = SymbolDecoder::new(&[0; 8], false).unwrap();
        d.read_symbol(&mut frozen).unwrap();
        assert_eq!(frozen, [1000, 2000, 32768, 0]);
    }

    #[test]
    fn malformed_cdf_and_exhausted_tile_are_errors() {
        let mut d = SymbolDecoder::new(&[0], false).unwrap();
        assert!(d.read_symbol(&mut [1, 0, 0]).is_err());
        assert!(d.read_literal(32).is_err());
    }

    #[test]
    fn first_spacewalk_partition_symbols() {
        // Compressed bytes immediately after the verified 16-byte frame header.
        // The first two symbols are independently checked with section 8.2.6
        // intervals: 8806 in [3224,9692), then 22331 in [11297,25872).
        let mut d =
            SymbolDecoder::new(&[0xbb, 0x32, 0x62, 0xf4, 0xfa, 0x57, 0x9a, 0x1e], true).unwrap();
        let mut cdfs = PartitionCdfs::default();
        assert_eq!(cdfs.read_interior(&mut d, 64, 0).unwrap(), 3); // SPLIT
        assert_eq!(cdfs.read_interior(&mut d, 32, 0).unwrap(), 0); // NONE
    }

    #[test]
    fn entropy_termination_checks_the_marker_and_padding() {
        assert!(SymbolDecoder::new(&[0x80], false).unwrap().finish().is_ok());
        for bytes in [[0x00], [0x81]] {
            assert!(SymbolDecoder::new(&bytes, false).unwrap().finish().is_err());
        }
    }
}
