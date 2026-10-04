//! Vorbis I residue formats 0, 1 and 2 (specification section 8.6).

use super::entropy::{Codebook, PacketBits};
use super::setup::Residue;
use crate::video::backend::MediaDecodeError;

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.into())
}

#[derive(Default)]
pub(super) struct Workspace {
    output: Vec<Vec<f64>>,
    channels: Vec<Vec<f64>>,
    classes: Vec<Vec<usize>>,
    vector: Vec<f64>,
}

pub(super) fn clear_vectors<T: Clone + Default>(
    vectors: &mut Vec<Vec<T>>,
    count: usize,
    size: usize,
) {
    vectors.resize_with(count, Vec::new);
    for vector in vectors {
        vector.resize(size, T::default());
        vector.fill(T::default());
    }
}

impl Residue {
    pub fn decode(
        &self,
        bits: &mut PacketBits<'_>,
        books: &[Codebook],
        do_decode: &[bool],
        bins: usize,
    ) -> Result<Vec<Vec<f64>>, MediaDecodeError> {
        let mut workspace = Workspace::default();
        self.decode_with_workspace(bits, books, do_decode, bins, &mut workspace)?;
        Ok(if self.kind == 2 {
            std::mem::take(&mut workspace.channels)
        } else {
            std::mem::take(&mut workspace.output)
        })
    }

    pub(super) fn decode_with_workspace<'a>(
        &self,
        bits: &mut PacketBits<'_>,
        books: &[Codebook],
        do_decode: &[bool],
        bins: usize,
        workspace: &'a mut Workspace,
    ) -> Result<&'a [Vec<f64>], MediaDecodeError> {
        if self.kind > 2
            || self.partition_size == 0
            || self.begin > self.end
            || do_decode.is_empty()
            || do_decode.len() > 255
            || bins > 4096
            || self.books.is_empty()
            || self.books.len() > 64
        {
            return Err(invalid("invalid Vorbis residue configuration"));
        }
        let classbook = books
            .get(self.classbook)
            .ok_or_else(|| invalid("invalid Vorbis residue classbook"))?;
        if classbook.dimensions == 0 || classbook.dimensions > 65535 {
            return Err(invalid("invalid Vorbis residue classword dimensions"));
        }
        let interleaved = self.kind == 2;
        let size = if interleaved {
            bins * do_decode.len()
        } else {
            bins
        };
        let interleaved_flags = [do_decode.iter().any(|&flag| flag)];
        let flags = if interleaved {
            &interleaved_flags[..]
        } else {
            do_decode
        };
        clear_vectors(&mut workspace.output, flags.len(), size);
        let output = &mut workspace.output;
        let begin = self.begin.min(size);
        let end = self.end.min(size);
        let partitions = (end - begin) / self.partition_size;
        clear_vectors(&mut workspace.classes, flags.len(), partitions);
        let classes = &mut workspace.classes;
        let vector = &mut workspace.vector;
        'passes: for pass in 0..8 {
            if !flags.iter().any(|&flag| flag) {
                break;
            }
            if pass != 0 && self.books.iter().all(|books| books[pass].is_none()) {
                continue;
            }
            let mut partition = 0;
            while partition < partitions {
                if pass == 0 {
                    for (channel, &enabled) in flags.iter().enumerate() {
                        if !enabled {
                            continue;
                        }
                        let Some(mut entry) = classbook.huffman.decode(bits) else {
                            break 'passes;
                        };
                        for word in (0..classbook.dimensions).rev() {
                            let class = entry % self.books.len();
                            entry /= self.books.len();
                            if word < partitions - partition {
                                classes[channel][partition + word] = class;
                            }
                        }
                    }
                }
                for _ in 0..classbook.dimensions.min(partitions - partition) {
                    for (channel, &enabled) in flags.iter().enumerate() {
                        if !enabled {
                            continue;
                        }
                        let Some(index) = self.books[classes[channel][partition]][pass] else {
                            continue;
                        };
                        let book = books
                            .get(index)
                            .ok_or_else(|| invalid("invalid Vorbis residue vector book"))?;
                        if book.dimensions == 0
                            || (self.kind != 0
                                && (book.dimensions > self.partition_size
                                    || self.partition_size % book.dimensions != 0))
                            || !book.has_lookup()
                        {
                            return Err(invalid(
                                "invalid Vorbis residue vector dimensions or lookup",
                            ));
                        }
                        vector.resize(book.dimensions, 0.0);
                        // Format zero uses integer division: any remainder is
                        // left untouched, and an oversized vector reads no VQ.
                        let steps = self.partition_size / book.dimensions;
                        let offset = begin + partition * self.partition_size;
                        for step in 0..steps {
                            let Some(entry) = book.huffman.decode(bits) else {
                                break 'passes;
                            };
                            book.vector(entry, vector)?;
                            for (dimension, &value) in vector.iter().enumerate() {
                                let target = if self.kind == 0 {
                                    step + dimension * steps
                                } else {
                                    step * book.dimensions + dimension
                                };
                                output[channel][offset + target] += value;
                            }
                        }
                    }
                    partition += 1;
                }
            }
        }
        if interleaved {
            clear_vectors(&mut workspace.channels, do_decode.len(), bins);
            let channels = &mut workspace.channels;
            for (bin, values) in output[0].chunks_exact(do_decode.len()).enumerate() {
                for (channel, &value) in channels.iter_mut().zip(values) {
                    channel[bin] = value;
                }
            }
            Ok(&workspace.channels)
        } else {
            Ok(&workspace.output)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(fields: &[(u32, u8)]) -> Vec<u8> {
        let mut bytes = vec![
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
                bytes[position / 8] |= ((value >> bit) as u8 & 1) << (position & 7);
                position += 1;
            }
        }
        bytes
    }

    fn books() -> Vec<Codebook> {
        let scalar = pack(&[
            (0x564342, 24),
            (1, 16),
            (1, 24),
            (0, 1),
            (0, 1),
            (0, 5),
            (0, 4),
        ]);
        let vector = pack(&[
            (0x564342, 24),
            (2, 16),
            (2, 24),
            (0, 1),
            (0, 1),
            (0, 5),
            (0, 5),
            (2, 4),
            (0, 32),
            ((788 << 21) | 1, 32),
            (3, 4),
            (0, 1),
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 4),
        ]);
        vec![
            Codebook::parse(&mut PacketBits::new(&scalar), &mut 1, &mut 0).unwrap(),
            Codebook::parse(&mut PacketBits::new(&vector), &mut 2, &mut 4).unwrap(),
        ]
    }

    fn residue(kind: u8) -> Residue {
        Residue {
            kind,
            begin: 0,
            end: 4,
            partition_size: 4,
            classbook: 0,
            books: vec![[Some(1), None, None, None, None, None, None, None]],
        }
    }

    #[test]
    fn format_zero_and_one_have_distinct_partition_layouts() {
        let books = books();
        let bytes = pack(&[(0, 1), (0, 1), (1, 1)]);
        let zero = residue(0)
            .decode(&mut PacketBits::new(&bytes), &books, &[true], 4)
            .unwrap();
        let one = residue(1)
            .decode(&mut PacketBits::new(&bytes), &books, &[true], 4)
            .unwrap();
        assert_eq!(zero[0], [1.0, 3.0, 2.0, 4.0]);
        assert_eq!(one[0], [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn format_zero_integer_stride_leaves_remainder_and_consumes_exact_words() {
        let books = books();
        let bytes = pack(&[(0, 1), (0, 1), (1, 1), (0, 1), (1, 1), (0, 1)]);
        let mut residue = residue(0);
        residue.end = 10;
        residue.partition_size = 5;
        let mut bits = PacketBits::new(&bytes);
        let output = residue.decode(&mut bits, &books, &[true], 10).unwrap();
        assert_eq!(
            output[0],
            [1.0, 3.0, 2.0, 4.0, 0.0, 3.0, 1.0, 4.0, 2.0, 0.0]
        );
        assert_eq!(bits.position(), 6);

        residue.end = 2;
        residue.partition_size = 1;
        let mut bits = PacketBits::new(&bytes);
        let output = residue.decode(&mut bits, &books, &[true], 2).unwrap();
        assert_eq!(output[0], [0.0; 2]);
        // Only the two classwords are consumed: floor(1 / 2) is zero.
        assert_eq!(bits.position(), 2);
    }

    #[test]
    fn format_two_decodes_all_channels_if_one_is_enabled() {
        let books = books();
        let bytes = pack(&[(0, 1), (0, 1), (1, 1)]);
        let output = residue(2)
            .decode(&mut PacketBits::new(&bytes), &books, &[false, true], 2)
            .unwrap();
        assert_eq!(output, [vec![1.0, 3.0], vec![2.0, 4.0]]);
        let mut bits = PacketBits::new(&bytes);
        let silent = residue(2)
            .decode(&mut bits, &books, &[false, false], 2)
            .unwrap();
        assert_eq!(silent, [vec![0.0; 2], vec![0.0; 2]]);
        assert_eq!(bits.position(), 0);
    }

    #[test]
    fn skips_disabled_vectors_and_adds_multiple_passes() {
        let books = books();
        let mut residue = residue(1);
        residue.books[0][1] = Some(1);
        let bytes = pack(&[(0, 1), (0, 1), (1, 1), (1, 1), (0, 1)]);
        let output = residue
            .decode(&mut PacketBits::new(&bytes), &books, &[false, true], 4)
            .unwrap();
        assert_eq!(output[0], [0.0; 4]);
        assert_eq!(output[1], [4.0, 6.0, 4.0, 6.0]);
        let output = residue
            .decode(&mut PacketBits::new(&[]), &books, &[true], 4)
            .unwrap();
        assert_eq!(output[0], [0.0; 4]);
    }

    #[test]
    fn workspace_reuse_clears_disabled_truncated_and_interleaved_vectors() {
        let books = books();
        let bytes = pack(&[(0, 1), (0, 1), (1, 1)]);
        let mut workspace = Workspace::default();
        for _ in 0..4 {
            let output = residue(0)
                .decode_with_workspace(
                    &mut PacketBits::new(&bytes),
                    &books,
                    &[true, false],
                    4,
                    &mut workspace,
                )
                .unwrap();
            assert_eq!(output, [vec![1.0, 3.0, 2.0, 4.0], vec![0.0; 4]]);
            let output = residue(2)
                .decode_with_workspace(
                    &mut PacketBits::new(&bytes),
                    &books,
                    &[false, true],
                    2,
                    &mut workspace,
                )
                .unwrap();
            assert_eq!(output, [vec![1.0, 3.0], vec![2.0, 4.0]]);
            let output = residue(1)
                .decode_with_workspace(
                    &mut PacketBits::new(&bytes),
                    &books,
                    &[false, true],
                    4,
                    &mut workspace,
                )
                .unwrap();
            assert_eq!(output, [vec![0.0; 4], vec![1.0, 2.0, 3.0, 4.0]]);
            let output = residue(2)
                .decode_with_workspace(
                    &mut PacketBits::new(&[]),
                    &books,
                    &[true, true],
                    4,
                    &mut workspace,
                )
                .unwrap();
            assert_eq!(output, [vec![0.0; 4], vec![0.0; 4]]);
            let mut bits = PacketBits::new(&bytes);
            let output = residue(2)
                .decode_with_workspace(&mut bits, &books, &[false, false], 2, &mut workspace)
                .unwrap();
            assert_eq!(output, [vec![0.0; 2], vec![0.0; 2]]);
            assert_eq!(bits.position(), 0);
        }
    }

    #[test]
    fn clamps_ranges_and_checks_untrusted_configurations() {
        let books = books();
        let mut residue = residue(0);
        residue.begin = 10;
        residue.end = 100;
        let mut bits = PacketBits::new(&[]);
        assert_eq!(
            residue.decode(&mut bits, &books, &[true], 4).unwrap()[0],
            [0.0; 4]
        );
        assert!(!bits.ended());
        residue.partition_size = 0;
        assert!(residue.decode(&mut bits, &books, &[true], 4).is_err());
        residue.partition_size = 4;
        assert!(residue.decode(&mut bits, &books, &[true], 4097).is_err());
        assert!(residue.decode(&mut bits, &books, &[], 4).is_err());
    }
}
