//! Vorbis I bitpacking and Huffman codewords (specification sections 2 and 3).

use crate::video::backend::MediaDecodeError;
use std::collections::BTreeSet;

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.into())
}

#[derive(Clone, Debug)]
pub struct PacketBits<'a> {
    bytes: &'a [u8],
    position: usize,
    ended: bool,
}

impl<'a> PacketBits<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            position: 0,
            ended: false,
        }
    }

    /// None is the sticky end-of-packet condition, not synthesized zero padding.
    pub fn read(&mut self, width: u8) -> Option<u32> {
        let end = self.position.checked_add(usize::from(width));
        if self.ended
            || width > 32
            || end.is_none_or(|end| end > self.bytes.len().saturating_mul(8))
        {
            self.ended = true;
            return None;
        }
        let end = end.unwrap();
        let mut value = 0;
        let mut written = 0;
        while self.position < end {
            let offset = self.position & 7;
            let take = (8 - offset).min(end - self.position);
            value |= ((u32::from(self.bytes[self.position / 8]) >> offset) & ((1u32 << take) - 1))
                << written;
            written += take;
            self.position += take;
        }
        Some(value)
    }

    pub fn position(&self) -> usize {
        self.position
    }
    pub fn ended(&self) -> bool {
        self.ended
    }
}

#[derive(Clone, Debug, Default)]
struct Node {
    branches: [Option<usize>; 2],
    symbol: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct Huffman {
    nodes: Vec<Node>,
    single: Option<usize>,
}

impl Huffman {
    /// Length zero marks unused entries. Assignment preserves entry order,
    /// unlike the length-sorted canonical codes used by several image codecs.
    pub fn from_lengths(lengths: &[u8]) -> Result<Self, MediaDecodeError> {
        if lengths.iter().any(|&length| length > 32) {
            return Err(invalid("Vorbis codeword longer than 32 bits"));
        }
        let active = lengths.iter().filter(|&&length| length != 0).count();
        if active == 0 {
            return Err(invalid("empty Vorbis Huffman tree"));
        }
        if active == 1 {
            let symbol = lengths.iter().position(|&length| length != 0).unwrap();
            if lengths[symbol] != 1 {
                return Err(invalid("invalid single-entry Vorbis codebook"));
            }
            return Ok(Self {
                nodes: Vec::new(),
                single: Some(symbol),
            });
        }
        let mut free: [BTreeSet<u32>; 33] = std::array::from_fn(|_| BTreeSet::new());
        free[0].insert(0);
        let mut nodes = vec![Node::default()];
        for (symbol, &length) in lengths.iter().enumerate() {
            if length == 0 {
                continue;
            }
            let length = usize::from(length);
            // Select the leftmost free subtree that can hold this codeword.
            let (mut depth, mut prefix) = (0..=length)
                .filter_map(|depth| free[depth].first().copied().map(|prefix| (depth, prefix)))
                .min_by_key(|&(depth, prefix)| u64::from(prefix) << (32 - depth))
                .ok_or_else(|| invalid("overspecified Vorbis Huffman tree"))?;
            free[depth].remove(&prefix);
            while depth < length {
                prefix <<= 1;
                depth += 1;
                free[depth].insert(prefix | 1);
            }
            let mut node = 0;
            for bit_position in (0..length).rev() {
                let branch = ((prefix >> bit_position) & 1) as usize;
                node = match nodes[node].branches[branch] {
                    Some(next) => next,
                    None => {
                        let next = nodes.len();
                        nodes.push(Node::default());
                        nodes[node].branches[branch] = Some(next);
                        next
                    }
                };
            }
            nodes[node].symbol = Some(symbol);
        }
        if free.iter().any(|level| !level.is_empty()) {
            return Err(invalid("underspecified Vorbis Huffman tree"));
        }
        Ok(Self {
            nodes,
            single: None,
        })
    }

    /// A truncated audio packet propagates end-of-packet to the synthesis layer.
    pub fn decode(&self, bits: &mut PacketBits<'_>) -> Option<usize> {
        if let Some(symbol) = self.single {
            bits.read(1)?;
            return Some(symbol);
        }
        let mut node = 0;
        loop {
            if let Some(symbol) = self.nodes[node].symbol {
                return Some(symbol);
            }
            node = self.nodes[node].branches[bits.read(1)? as usize]?;
        }
    }
}

#[derive(Clone, Debug)]
struct Lookup {
    kind: u8,
    minimum: f64,
    delta: f64,
    sequence: bool,
    multiplicands: Vec<u16>,
}

#[derive(Clone, Debug)]
pub struct Codebook {
    pub dimensions: usize,
    pub entries: usize,
    pub huffman: Huffman,
    lookup: Option<Lookup>,
}

fn field(bits: &mut PacketBits<'_>, width: u8) -> Result<u32, MediaDecodeError> {
    bits.read(width)
        .ok_or_else(|| invalid("truncated Vorbis codebook"))
}

fn unpack_float(value: u32) -> f64 {
    let mantissa = f64::from(value & 0x1f_ffff);
    let sign = if value & 0x8000_0000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((value >> 21) & 0x3ff) as i32 - 788;
    sign * mantissa * 2f64.powi(exponent)
}

fn lattice_count(entries: usize, dimensions: usize) -> usize {
    let mut low = 1;
    let mut high = entries;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        let mut product = Some(1usize);
        for _ in 0..dimensions {
            product = product
                .and_then(|product| product.checked_mul(middle))
                .filter(|&product| product <= entries);
            if product.is_none() {
                break;
            }
        }
        if product.is_some() {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    low
}

impl Codebook {
    pub fn has_lookup(&self) -> bool {
        self.lookup.is_some()
    }

    /// Budgets are shared across a setup header and deducted before allocation.
    pub fn parse(
        bits: &mut PacketBits<'_>,
        entry_budget: &mut usize,
        lookup_budget: &mut usize,
    ) -> Result<Self, MediaDecodeError> {
        if field(bits, 24)? != 0x564342 {
            return Err(invalid("invalid Vorbis codebook signature"));
        }
        let dimensions = field(bits, 16)? as usize;
        let entries = field(bits, 24)? as usize;
        if dimensions == 0 || entries == 0 {
            return Err(invalid("empty Vorbis codebook dimensions or entries"));
        }
        *entry_budget = entry_budget
            .checked_sub(entries)
            .ok_or_else(|| invalid("Vorbis codebook entry budget exceeded"))?;
        let mut lengths = vec![0; entries];
        if field(bits, 1)? == 0 {
            let sparse = field(bits, 1)? != 0;
            for length in &mut lengths {
                if !sparse || field(bits, 1)? != 0 {
                    *length = field(bits, 5)? as u8 + 1;
                }
            }
        } else {
            let mut length = field(bits, 5)? as u8 + 1;
            let mut current = 0;
            while current < entries {
                if length > 32 {
                    return Err(invalid("ordered Vorbis codeword exceeds 32 bits"));
                }
                let width = usize::BITS - (entries - current).leading_zeros();
                let count = field(bits, width as u8)? as usize;
                if count > entries - current {
                    return Err(invalid("ordered Vorbis codebook run exceeds entry count"));
                }
                lengths[current..current + count].fill(length);
                current += count;
                length += 1;
            }
        }
        let huffman = Huffman::from_lengths(&lengths)?;
        let kind = field(bits, 4)? as u8;
        let lookup = match kind {
            0 => None,
            1 | 2 => {
                let minimum = unpack_float(field(bits, 32)?);
                let delta = unpack_float(field(bits, 32)?);
                let width = field(bits, 4)? as u8 + 1;
                let sequence = field(bits, 1)? != 0;
                let count = if kind == 1 {
                    lattice_count(entries, dimensions)
                } else {
                    entries
                        .checked_mul(dimensions)
                        .ok_or_else(|| invalid("Vorbis lookup size overflow"))?
                };
                *lookup_budget = lookup_budget
                    .checked_sub(count)
                    .ok_or_else(|| invalid("Vorbis lookup budget exceeded"))?;
                let mut multiplicands = Vec::with_capacity(count);
                for _ in 0..count {
                    multiplicands.push(field(bits, width)? as u16);
                }
                Some(Lookup {
                    kind,
                    minimum,
                    delta,
                    sequence,
                    multiplicands,
                })
            }
            _ => return Err(invalid("reserved Vorbis lookup type")),
        };
        Ok(Self {
            dimensions,
            entries,
            huffman,
            lookup,
        })
    }

    /// Materialize one vector into caller-owned storage rather than expanding
    /// the entire Cartesian lookup table during setup.
    pub fn vector(&self, entry: usize, output: &mut [f64]) -> Result<(), MediaDecodeError> {
        let lookup = self
            .lookup
            .as_ref()
            .ok_or_else(|| invalid("scalar Vorbis book used as a vector"))?;
        if entry >= self.entries || output.len() != self.dimensions {
            return Err(invalid("invalid Vorbis vector entry or dimensions"));
        }
        let mut quotient = entry;
        let mut last = 0.0;
        for (dimension, output) in output.iter_mut().enumerate() {
            let index = if lookup.kind == 1 {
                let index = quotient % lookup.multiplicands.len();
                quotient /= lookup.multiplicands.len();
                index
            } else {
                entry * self.dimensions + dimension
            };
            *output = f64::from(lookup.multiplicands[index]) * lookup.delta + lookup.minimum + last;
            if !output.is_finite() {
                return Err(invalid("nonfinite Vorbis vector"));
            }
            if lookup.sequence {
                last = *output;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(fields: &[(u32, u8)]) -> Vec<u8> {
        let mut output = vec![
            0;
            fields
                .iter()
                .map(|&(_, width)| usize::from(width))
                .sum::<usize>()
                .div_ceil(8)
        ];
        let mut position = 0;
        for &(value, width) in fields {
            for bit in 0..width {
                output[position / 8] |= (((value >> bit) & 1) as u8) << (position & 7);
                position += 1;
            }
        }
        output
    }

    #[test]
    fn reads_ordered_unordered_and_sparse_codebooks() {
        for fields in [
            vec![
                (0x564342, 24),
                (1, 16),
                (2, 24),
                (1, 1),
                (0, 5),
                (2, 2),
                (0, 4),
            ],
            vec![
                (0x564342, 24),
                (1, 16),
                (2, 24),
                (0, 1),
                (0, 1),
                (0, 5),
                (0, 5),
                (0, 4),
            ],
            vec![
                (0x564342, 24),
                (1, 16),
                (3, 24),
                (0, 1),
                (1, 1),
                (1, 1),
                (0, 5),
                (0, 1),
                (1, 1),
                (0, 5),
                (0, 4),
            ],
        ] {
            let data = pack(&fields);
            let mut entry_budget = 3;
            let book =
                Codebook::parse(&mut PacketBits::new(&data), &mut entry_budget, &mut 0).unwrap();
            assert_eq!(book.dimensions, 1);
            assert_eq!(entry_budget, 3 - book.entries);
            assert!(Codebook::parse(&mut PacketBits::new(&data), &mut 1, &mut 0).is_err());
            for length in 0..data.len() {
                assert!(
                    Codebook::parse(&mut PacketBits::new(&data[..length]), &mut 3, &mut 0).is_err()
                );
            }
        }
    }

    #[test]
    fn lookup_vectors_are_lazy_and_sequence_aware() {
        let one = 788u32 << 21 | 1;
        for kind in [1, 2] {
            for sequence in [0, 1] {
                let mut fields = vec![
                    (0x564342, 24),
                    (2, 16),
                    (4, 24),
                    (1, 1),
                    (1, 5),
                    (4, 3),
                    (kind, 4),
                    (one, 32),
                    (one, 32),
                    (3, 4),
                    (sequence, 1),
                ];
                let multiplicands = if kind == 1 {
                    vec![2, 3]
                } else {
                    vec![2, 3, 4, 5, 6, 7, 8, 9]
                };
                fields.extend(multiplicands.into_iter().map(|value| (value, 4)));
                let data = pack(&fields);
                let book = Codebook::parse(&mut PacketBits::new(&data), &mut 4, &mut 8).unwrap();
                let mut output = [0.0; 2];
                book.vector(3, &mut output).unwrap();
                let expected = if kind == 1 { [4.0, 4.0] } else { [9.0, 10.0] };
                assert_eq!(
                    output,
                    [
                        expected[0],
                        expected[1] + if sequence == 1 { expected[0] } else { 0.0 }
                    ]
                );
                assert!(book.vector(4, &mut output).is_err());
                assert!(Codebook::parse(&mut PacketBits::new(&data), &mut 4, &mut 0).is_err());
            }
        }
        for dimensions in 1..=16 {
            for entries in 1usize..=256 {
                let count = lattice_count(entries, dimensions);
                assert!(
                    count
                        .checked_pow(dimensions as u32)
                        .is_some_and(|power| power <= entries)
                );
                assert!(
                    (count + 1)
                        .checked_pow(dimensions as u32)
                        .is_none_or(|power| power > entries)
                );
            }
        }
    }

    #[test]
    fn specification_bitpacking_example_and_sticky_end() {
        let mut bits = PacketBits::new(&[0xfc, 0x48, 0xce, 0x06]);
        assert_eq!(bits.read(4), Some(12));
        assert_eq!(bits.read(3), Some(7));
        assert_eq!(bits.read(7), Some(17));
        assert_eq!(bits.read(13), Some(6969));
        assert_eq!(bits.read(5), Some(0));
        assert_eq!(bits.read(0), Some(0));
        assert_eq!(bits.read(1), None);
        assert_eq!(bits.read(0), None);
        assert!(bits.ended());
    }

    #[test]
    fn specification_entry_order_codewords() {
        let tree = Huffman::from_lengths(&[2, 4, 4, 4, 4, 2, 3, 3]).unwrap();
        let words = [
            (0b00, 2),
            (0b0100, 4),
            (0b0101, 4),
            (0b0110, 4),
            (0b0111, 4),
            (0b10, 2),
            (0b110, 3),
            (0b111, 3),
        ];
        for (symbol, (word, length)) in words.into_iter().enumerate() {
            let byte = ((word as u32).reverse_bits() >> (32 - length)) as u8;
            let mut bits = PacketBits::new(std::slice::from_ref(&byte));
            assert_eq!(tree.decode(&mut bits), Some(symbol));
            assert_eq!(bits.position(), length);
        }
    }

    #[test]
    fn sparse_and_single_entry_books_follow_errata() {
        let tree = Huffman::from_lengths(&[0, 1, 0, 1]).unwrap();
        let mut bits = PacketBits::new(&[2]);
        assert_eq!(tree.decode(&mut bits), Some(1));
        assert_eq!(tree.decode(&mut bits), Some(3));
        let tree = Huffman::from_lengths(&[0, 1, 0]).unwrap();
        let mut bits = PacketBits::new(&[255]);
        for _ in 0..8 {
            assert_eq!(tree.decode(&mut bits), Some(1));
        }
        assert_eq!(tree.decode(&mut bits), None);
        for lengths in [&[0][..], &[2], &[1, 1, 1], &[2, 2], &[33, 1]] {
            assert!(Huffman::from_lengths(lengths).is_err());
        }
    }

    #[test]
    fn long_codewords_and_malformed_books_are_bounded() {
        let mut lengths: Vec<_> = (1..=32).collect();
        lengths.push(32);
        let tree = Huffman::from_lengths(&lengths).unwrap();
        let mut bits = PacketBits::new(&[255; 4]);
        assert_eq!(tree.decode(&mut bits), Some(32));
        assert_eq!(bits.position(), 32);
        let mut bits = PacketBits::new(&[255; 3]);
        assert_eq!(tree.decode(&mut bits), None);
        assert!(bits.ended());
        let mut seed = 0xa123_b456u32;
        for _ in 0..1024 {
            let mut bytes = [0; 128];
            for byte in &mut bytes {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                *byte = (seed >> 24) as u8;
            }
            let lengths: Vec<_> = bytes.iter().take(64).map(|&byte| byte % 34).collect();
            if let Ok(tree) = Huffman::from_lengths(&lengths) {
                let mut bits = PacketBits::new(&bytes);
                while tree.decode(&mut bits).is_some() {}
                assert!(bits.ended());
            }
            bytes[..3].copy_from_slice(b"BCV");
            let _ = Codebook::parse(&mut PacketBits::new(&bytes), &mut 4096, &mut 4096);
        }
    }

    #[test]
    fn bit_reads_match_independent_bitwise_model() {
        let data: Vec<_> = (0..128).map(|i| (i * 137 + 71) as u8).collect();
        for offset in 0..8 {
            for width in 0..=32 {
                let mut bits = PacketBits::new(&data);
                bits.read(offset).unwrap();
                let mut expected = 0;
                for bit in 0..usize::from(width) {
                    let position = usize::from(offset) + bit;
                    expected |= u32::from((data[position / 8] >> (position & 7)) & 1) << bit;
                }
                assert_eq!(bits.read(width), Some(expected));
            }
        }
    }
}
