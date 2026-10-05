//! AAC noiseless decoding: ISO/IEC 13818-7:2004 section 9.3, tables 58-59.

use super::{Bits, tables::CODEBOOKS};
use crate::video::backend::MediaDecodeError;
use std::sync::OnceLock;

#[derive(Clone, Copy)]
struct Node {
    child: [usize; 2],
    symbol: Option<usize>,
}

impl Node {
    fn empty() -> Self {
        Self {
            child: [usize::MAX; 2],
            symbol: None,
        }
    }
}

fn trees() -> &'static [Vec<Node>; 12] {
    static TREES: OnceLock<[Vec<Node>; 12]> = OnceLock::new();
    TREES.get_or_init(|| {
        std::array::from_fn(|book| {
            let mut nodes = vec![Node::empty()];
            for (symbol, &(length, word)) in CODEBOOKS[book].iter().enumerate() {
                let mut node = 0;
                for shift in (0..length).rev() {
                    let bit = ((word >> shift) & 1) as usize;
                    if nodes[node].child[bit] == usize::MAX {
                        nodes[node].child[bit] = nodes.len();
                        nodes.push(Node::empty());
                    }
                    node = nodes[node].child[bit];
                }
                nodes[node].symbol = Some(symbol);
            }
            nodes
        })
    })
}

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.into())
}

fn symbol(bits: &mut Bits<'_>, book: usize) -> Result<usize, MediaDecodeError> {
    let nodes = &trees()[book];
    let mut node = 0;
    for _ in 0..19 {
        node = nodes[node].child[bits.read(1)? as usize];
        if node == usize::MAX {
            return Err(invalid("invalid AAC Huffman word"));
        }
        if let Some(value) = nodes[node].symbol {
            return Ok(value);
        }
    }
    Err(invalid("overlong AAC Huffman word"))
}

/// Differential scalefactor, before accumulation with global_gain.
pub(crate) fn scalefactor(bits: &mut Bits<'_>) -> Result<i32, MediaDecodeError> {
    Ok(symbol(bits, 0)? as i32 - 60)
}

/// Quantized tuple and dimension. Pair books use the first two array slots.
pub(crate) fn spectral(
    bits: &mut Bits<'_>,
    book: u8,
) -> Result<([i32; 4], usize), MediaDecodeError> {
    let (width, lav, signed) = match book {
        1 | 2 => (4, 1, true),
        3 | 4 => (4, 2, false),
        5 | 6 => (2, 4, true),
        7 | 8 => (2, 7, false),
        9 | 10 => (2, 12, false),
        11 => (2, 16, false),
        _ => return Err(invalid("AAC book is not a spectral Huffman book")),
    };
    let radix = if signed { 2 * lav + 1 } else { lav + 1 };
    let mut index = symbol(bits, usize::from(book))? as i32;
    let mut values = [0; 4];
    for value in values[..width].iter_mut().rev() {
        *value = index % radix - if signed { lav } else { 0 };
        index /= radix;
    }
    // All tuple signs precede all escape extensions (section 9.3).
    if !signed {
        for value in &mut values[..width] {
            if *value != 0 && bits.read(1)? != 0 {
                *value = -*value;
            }
        }
    }
    if book == 11 {
        for value in &mut values[..width] {
            if value.abs() == 16 {
                let mut width = 4;
                while bits.read(1)? != 0 {
                    width += 1;
                    // Section 9.3 limits escape_sequence to fewer than 22 bits.
                    if width > 12 {
                        return Err(invalid("overlong AAC escape sequence"));
                    }
                }
                let magnitude = (1i32 << width) + bits.read(width)? as i32;
                *value = if *value < 0 { -magnitude } else { magnitude };
            }
        }
    }
    Ok((values, width))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed(fields: &[(u32, usize)]) -> Vec<u8> {
        let mut bytes = vec![0; (fields.iter().map(|f| f.1).sum::<usize>() + 7) / 8];
        let mut position = 0;
        for &(value, width) in fields {
            for shift in (0..width).rev() {
                bytes[position / 8] |= (((value >> shift) & 1) as u8) << (7 - position % 8);
                position += 1;
            }
        }
        bytes
    }

    #[test]
    fn all_numeric_words_are_prefix_free_and_decode_exactly() {
        for (book, words) in CODEBOOKS.iter().enumerate() {
            let max = words.iter().map(|w| w.0).max().unwrap();
            assert_eq!(
                words.iter().map(|w| 1u32 << (max - w.0)).sum::<u32>(),
                1 << max
            );
            for (index, &(length, word)) in words.iter().enumerate() {
                for &(other_length, other_word) in words.iter() {
                    if other_length < length {
                        assert_ne!(word >> (length - other_length), other_word);
                    }
                }
                let bytes = packed(&[(word, usize::from(length)), (0x55, 7)]);
                let mut bits = Bits::new(&bytes);
                assert_eq!(symbol(&mut bits, book).unwrap(), index);
                assert_eq!(bits.read(7).unwrap(), 0x55);
            }
        }
    }

    #[test]
    fn scalefactor_range_and_center() {
        for index in 0..121 {
            let (length, word) = CODEBOOKS[0][index];
            let bytes = packed(&[(word, length as usize)]);
            assert_eq!(
                scalefactor(&mut Bits::new(&bytes)).unwrap(),
                index as i32 - 60
            );
        }
    }

    #[test]
    fn every_spectral_tuple_and_sign() {
        for book in 1..=11 {
            let (width, lav, signed) = match book {
                1 | 2 => (4, 1, true),
                3 | 4 => (4, 2, false),
                5 | 6 => (2, 4, true),
                7 | 8 => (2, 7, false),
                9 | 10 => (2, 12, false),
                _ => (2, 16, false),
            };
            let radix = if signed { 2 * lav + 1 } else { lav + 1 };
            for (index, &(length, word)) in CODEBOOKS[book].iter().enumerate() {
                let mut expected = [0; 4];
                for slot in 0..width {
                    let divisor = (radix as usize).pow((width - slot - 1) as u32);
                    expected[slot] =
                        (index / divisor % radix as usize) as i32 - if signed { lav } else { 0 };
                }
                let mut fields = vec![(word, length as usize)];
                if !signed {
                    for value in &mut expected[..width] {
                        if *value != 0 {
                            fields.push((1, 1));
                            *value = -*value;
                        }
                    }
                }
                if book == 11 {
                    for value in &expected[..width] {
                        if value.abs() == 16 {
                            fields.push((0, 5));
                        }
                    }
                }
                fields.push((0x55, 7));
                let bytes = packed(&fields);
                let mut bits = Bits::new(&bytes);
                assert_eq!(spectral(&mut bits, book as u8).unwrap(), (expected, width));
                assert_eq!(bits.read(7).unwrap(), 0x55);
            }
        }
    }

    #[test]
    fn escape_bounds_and_truncation() {
        let (length, word) = CODEBOOKS[11][16 * 17 + 16];
        let bytes = packed(&[
            (word, length as usize),
            (2, 2),
            (0, 1),
            (15, 4),
            (255, 8),
            (0, 1),
            (4095, 12),
        ]);
        assert_eq!(
            spectral(&mut Bits::new(&bytes), 11).unwrap(),
            ([-31, 8191, 0, 0], 2)
        );
        let bytes = packed(&[(word, length as usize), (0, 2), (511, 9)]);
        assert!(spectral(&mut Bits::new(&bytes), 11).is_err());
        assert!(scalefactor(&mut Bits::new(&[])).is_err());
        assert!(spectral(&mut Bits::new(&[]), 1).is_err());
        assert!(spectral(&mut Bits::new(&[]), 13).is_err());
    }
}
