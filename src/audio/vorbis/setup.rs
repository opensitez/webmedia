//! Vorbis I setup header and audio mode/window selection, from sections 4, 6-8.

use super::entropy::{Codebook, PacketBits};
use crate::video::backend::MediaDecodeError;

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.into())
}

fn field(bits: &mut PacketBits<'_>, width: u8) -> Result<usize, MediaDecodeError> {
    bits.read(width)
        .map(|value| value as usize)
        .ok_or_else(|| invalid("truncated Vorbis setup or packet header"))
}

fn reference(bits: &mut PacketBits<'_>, count: usize) -> Result<usize, MediaDecodeError> {
    let index = field(bits, 8)?;
    if index >= count {
        return Err(invalid("invalid Vorbis setup reference"));
    }
    Ok(index)
}

pub(super) fn read_codebooks(
    bits: &mut PacketBits<'_>,
    entry_budget: &mut usize,
    lookup_budget: &mut usize,
) -> Result<Vec<Codebook>, MediaDecodeError> {
    let count = field(bits, 8)? + 1;
    let mut books = Vec::with_capacity(count);
    for _ in 0..count {
        books.push(Codebook::parse(bits, entry_budget, lookup_budget)?);
    }
    Ok(books)
}

#[derive(Clone, Debug)]
pub struct FloorZero {
    pub order: usize,
    pub rate: usize,
    pub bark_map_size: usize,
    pub amplitude_bits: u8,
    pub amplitude_offset: usize,
    pub books: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct FloorClass {
    pub dimensions: usize,
    pub subclasses: u8,
    pub masterbook: Option<usize>,
    pub books: Vec<Option<usize>>,
}

#[derive(Clone, Debug)]
pub struct FloorOne {
    pub partitions: Vec<usize>,
    pub classes: Vec<FloorClass>,
    pub multiplier: usize,
    pub x: Vec<usize>,
    pub neighbors: Vec<(usize, usize)>,
    pub sorted_points: Vec<usize>,
}

#[derive(Clone, Debug)]
pub enum Floor {
    Zero(FloorZero),
    One(FloorOne),
}

#[derive(Clone, Debug)]
pub struct Residue {
    pub kind: u8,
    pub begin: usize,
    pub end: usize,
    pub partition_size: usize,
    pub classbook: usize,
    pub books: Vec<[Option<usize>; 8]>,
}

#[derive(Clone, Debug)]
pub struct Submap {
    pub floor: usize,
    pub residue: usize,
}

#[derive(Clone, Debug)]
pub struct Mapping {
    pub coupling: Vec<(usize, usize)>,
    pub mux: Vec<usize>,
    pub submaps: Vec<Submap>,
}

#[derive(Clone, Debug)]
pub struct Mode {
    pub long: bool,
    pub mapping: usize,
}

#[derive(Clone, Debug)]
pub struct Setup {
    pub codebooks: Vec<Codebook>,
    pub floors: Vec<Floor>,
    pub residues: Vec<Residue>,
    pub mappings: Vec<Mapping>,
    pub modes: Vec<Mode>,
}

impl Floor {
    fn parse(bits: &mut PacketBits<'_>, books: &[Codebook]) -> Result<Self, MediaDecodeError> {
        match field(bits, 16)? {
            0 => {
                let order = field(bits, 8)?;
                let rate = field(bits, 16)?;
                let bark_map_size = field(bits, 16)?;
                let amplitude_bits = field(bits, 6)? as u8;
                let amplitude_offset = field(bits, 8)?;
                let count = field(bits, 4)? + 1;
                let mut floor_books = Vec::with_capacity(count);
                for _ in 0..count {
                    let book = reference(bits, books.len())?;
                    if !books[book].has_lookup() {
                        return Err(invalid("Vorbis floor zero requires vector books"));
                    }
                    floor_books.push(book);
                }
                if order == 0 || rate == 0 || bark_map_size == 0 {
                    return Err(invalid("invalid Vorbis floor zero parameters"));
                }
                Ok(Self::Zero(FloorZero {
                    order,
                    rate,
                    bark_map_size,
                    amplitude_bits,
                    amplitude_offset,
                    books: floor_books,
                }))
            }
            1 => {
                let count = field(bits, 5)?;
                let mut partitions = Vec::with_capacity(count);
                for _ in 0..count {
                    partitions.push(field(bits, 4)?);
                }
                let class_count = partitions.iter().max().map_or(0, |&maximum| maximum + 1);
                let mut classes = Vec::with_capacity(class_count);
                for _ in 0..class_count {
                    let dimensions = field(bits, 3)? + 1;
                    let subclasses = field(bits, 2)? as u8;
                    let masterbook = if subclasses != 0 {
                        Some(reference(bits, books.len())?)
                    } else {
                        None
                    };
                    let mut subclass_books = Vec::with_capacity(1 << subclasses);
                    for _ in 0..1 << subclasses {
                        let encoded = field(bits, 8)?;
                        if encoded > books.len() {
                            return Err(invalid("invalid Vorbis floor subclass book"));
                        }
                        subclass_books.push(encoded.checked_sub(1));
                    }
                    classes.push(FloorClass {
                        dimensions,
                        subclasses,
                        masterbook,
                        books: subclass_books,
                    });
                }
                let multiplier = field(bits, 2)? + 1;
                let range_bits = field(bits, 4)? as u8;
                let mut x = vec![0, 1 << range_bits];
                for &class in &partitions {
                    if x.len() + classes[class].dimensions > 65 {
                        return Err(invalid("Vorbis floor one has more than 65 points"));
                    }
                    for _ in 0..classes[class].dimensions {
                        let value = field(bits, range_bits)?;
                        if x.contains(&value) {
                            return Err(invalid("duplicate Vorbis floor point"));
                        }
                        x.push(value);
                    }
                }
                let mut neighbors = vec![(0, 0); 2];
                for index in 2..x.len() {
                    let low = (0..index)
                        .filter(|&prior| x[prior] < x[index])
                        .max_by_key(|&prior| x[prior])
                        .unwrap();
                    let high = (0..index)
                        .filter(|&prior| x[prior] > x[index])
                        .min_by_key(|&prior| x[prior])
                        .unwrap();
                    neighbors.push((low, high));
                }
                let mut sorted_points: Vec<_> = (0..x.len()).collect();
                sorted_points.sort_unstable_by_key(|&index| x[index]);
                Ok(Self::One(FloorOne {
                    partitions,
                    classes,
                    multiplier,
                    x,
                    neighbors,
                    sorted_points,
                }))
            }
            _ => Err(invalid("reserved Vorbis floor type")),
        }
    }
}

impl Residue {
    fn parse(bits: &mut PacketBits<'_>, codebooks: &[Codebook]) -> Result<Self, MediaDecodeError> {
        let kind = field(bits, 16)?;
        if kind > 2 {
            return Err(invalid("reserved Vorbis residue type"));
        }
        let begin = field(bits, 24)?;
        let end = field(bits, 24)?;
        let partition_size = field(bits, 24)? + 1;
        let classifications = field(bits, 6)? + 1;
        let classbook = reference(bits, codebooks.len())?;
        let mut combinations = 1usize;
        for _ in 0..codebooks[classbook].dimensions {
            combinations = combinations
                .checked_mul(classifications)
                .filter(|&count| count <= codebooks[classbook].entries)
                .ok_or_else(|| invalid("Vorbis residue classification book is too small"))?;
        }
        if begin > end {
            return Err(invalid("reversed Vorbis residue range"));
        }
        let mut cascades = Vec::with_capacity(classifications);
        for _ in 0..classifications {
            let low = field(bits, 3)?;
            let high = if field(bits, 1)? != 0 {
                field(bits, 5)?
            } else {
                0
            };
            cascades.push(low | (high << 3));
        }
        let mut books = Vec::with_capacity(classifications);
        for cascade in cascades {
            let mut passes = [None; 8];
            for (pass, book) in passes.iter_mut().enumerate() {
                if cascade & (1 << pass) != 0 {
                    let index = reference(bits, codebooks.len())?;
                    if !codebooks[index].has_lookup() {
                        return Err(invalid("Vorbis residue requires vector books"));
                    }
                    *book = Some(index);
                }
            }
            books.push(passes);
        }
        Ok(Self {
            kind: kind as u8,
            begin,
            end,
            partition_size,
            classbook,
            books,
        })
    }
}

impl Mapping {
    fn parse(
        bits: &mut PacketBits<'_>,
        channels: usize,
        floors: usize,
        residues: usize,
    ) -> Result<Self, MediaDecodeError> {
        if field(bits, 16)? != 0 {
            return Err(invalid("reserved Vorbis mapping type"));
        }
        let count = if field(bits, 1)? != 0 {
            field(bits, 4)? + 1
        } else {
            1
        };
        let mut coupling = Vec::new();
        if field(bits, 1)? != 0 {
            let steps = field(bits, 8)? + 1;
            let width = (usize::BITS - (channels - 1).leading_zeros()) as u8;
            for _ in 0..steps {
                let magnitude = field(bits, width)?;
                let angle = field(bits, width)?;
                if magnitude == angle || magnitude >= channels || angle >= channels {
                    return Err(invalid("invalid Vorbis coupled channels"));
                }
                coupling.push((magnitude, angle));
            }
        }
        if field(bits, 2)? != 0 {
            return Err(invalid("nonzero Vorbis mapping reserved bits"));
        }
        let mut mux = vec![0; channels];
        if count > 1 {
            for channel in &mut mux {
                *channel = field(bits, 4)?;
                if *channel >= count {
                    return Err(invalid("invalid Vorbis channel mux"));
                }
            }
        }
        let mut submaps = Vec::with_capacity(count);
        for _ in 0..count {
            field(bits, 8)?;
            submaps.push(Submap {
                floor: reference(bits, floors)?,
                residue: reference(bits, residues)?,
            });
        }
        Ok(Self {
            coupling,
            mux,
            submaps,
        })
    }
}

impl Setup {
    pub fn parse(
        bytes: &[u8],
        channels: u8,
        mut entry_budget: usize,
        mut lookup_budget: usize,
    ) -> Result<Self, MediaDecodeError> {
        if channels == 0 || !bytes.starts_with(b"\x05vorbis") {
            return Err(invalid("invalid Vorbis setup signature or channels"));
        }
        let mut bits = PacketBits::new(&bytes[7..]);
        let codebooks = read_codebooks(&mut bits, &mut entry_budget, &mut lookup_budget)?;
        let time_count = field(&mut bits, 6)? + 1;
        for _ in 0..time_count {
            if field(&mut bits, 16)? != 0 {
                return Err(invalid("nonzero Vorbis time transform"));
            }
        }
        let floor_count = field(&mut bits, 6)? + 1;
        let mut floors = Vec::with_capacity(floor_count);
        for _ in 0..floor_count {
            floors.push(Floor::parse(&mut bits, &codebooks)?);
        }
        let residue_count = field(&mut bits, 6)? + 1;
        let mut residues = Vec::with_capacity(residue_count);
        for _ in 0..residue_count {
            residues.push(Residue::parse(&mut bits, &codebooks)?);
        }
        let mapping_count = field(&mut bits, 6)? + 1;
        let mut mappings = Vec::with_capacity(mapping_count);
        for _ in 0..mapping_count {
            mappings.push(Mapping::parse(
                &mut bits,
                usize::from(channels),
                floors.len(),
                residues.len(),
            )?);
        }
        let mode_count = field(&mut bits, 6)? + 1;
        let mut modes = Vec::with_capacity(mode_count);
        for _ in 0..mode_count {
            let long = field(&mut bits, 1)? != 0;
            if field(&mut bits, 16)? != 0 || field(&mut bits, 16)? != 0 {
                return Err(invalid("reserved Vorbis mode window or transform"));
            }
            modes.push(Mode {
                long,
                mapping: reference(&mut bits, mappings.len())?,
            });
        }
        if field(&mut bits, 1)? != 1 {
            return Err(invalid("missing Vorbis setup framing bit"));
        }
        Ok(Self {
            codebooks,
            floors,
            residues,
            mappings,
            modes,
        })
    }

    /// Returns None for non-audio packets, which must be ignored.
    pub fn packet_header(
        &self,
        bits: &mut PacketBits<'_>,
    ) -> Result<Option<PacketHeader>, MediaDecodeError> {
        if field(bits, 1)? != 0 {
            return Ok(None);
        }
        if self.modes.is_empty() {
            return Err(invalid("Vorbis setup has no modes"));
        }
        let width = (usize::BITS - (self.modes.len() - 1).leading_zeros()) as u8;
        let index = field(bits, width)?;
        let mode = self
            .modes
            .get(index)
            .ok_or_else(|| invalid("invalid Vorbis audio mode"))?;
        let (previous_long, next_long) = if mode.long {
            (field(bits, 1)? != 0, field(bits, 1)? != 0)
        } else {
            (false, false)
        };
        Ok(Some(PacketHeader {
            mode: index,
            long: mode.long,
            previous_long,
            next_long,
        }))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PacketHeader {
    pub mode: usize,
    pub long: bool,
    pub previous_long: bool,
    pub next_long: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(fields: &[(u32, u8)]) -> Vec<u8> {
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

    fn minimal_setup() -> Vec<(u32, u8)> {
        vec![
            (0, 8), // One scalar codebook, one length-one entry.
            (0x564342, 24),
            (1, 16),
            (1, 24),
            (0, 1),
            (0, 1),
            (0, 5),
            (0, 4),
            (0, 6),
            (0, 16), // Time transform.
            (0, 6),
            (1, 16),
            (0, 5),
            (0, 2),
            (6, 4), // Floor one, no partitions.
            (0, 6),
            (0, 16),
            (0, 24),
            (32, 24),
            (7, 24),
            (0, 6),
            (0, 8),
            (0, 3),
            (0, 1),
            (0, 6),
            (0, 16),
            (0, 1),
            (0, 1),
            (0, 2),
            (0, 8),
            (0, 8),
            (0, 8),
            (1, 6), // Two modes, short then long.
            (0, 1),
            (0, 16),
            (0, 16),
            (0, 8),
            (1, 1),
            (0, 16),
            (0, 16),
            (0, 8),
            (1, 1),
        ]
    }

    fn header(fields: &[(u32, u8)]) -> Vec<u8> {
        let mut bytes = b"\x05vorbis".to_vec();
        bytes.extend(packet(fields));
        bytes
    }

    #[test]
    fn complete_setup_and_audio_window_flags() {
        let bytes = header(&minimal_setup());
        let setup = Setup::parse(&bytes, 2, 1, 0).unwrap();
        assert_eq!(setup.codebooks.len(), 1);
        assert_eq!(setup.floors.len(), 1);
        assert_eq!(setup.residues[0].partition_size, 8);
        assert_eq!(setup.mappings[0].mux, [0, 0]);
        for previous in [false, true] {
            for next in [false, true] {
                let bytes = packet(&[
                    (0, 1),
                    (1, 1),
                    (u32::from(previous), 1),
                    (u32::from(next), 1),
                ]);
                let header = setup
                    .packet_header(&mut PacketBits::new(&bytes))
                    .unwrap()
                    .unwrap();
                assert_eq!(
                    header,
                    PacketHeader {
                        mode: 1,
                        long: true,
                        previous_long: previous,
                        next_long: next
                    }
                );
            }
        }
        assert!(
            setup
                .packet_header(&mut PacketBits::new(&[1]))
                .unwrap()
                .is_none()
        );
        assert!(setup.packet_header(&mut PacketBits::new(&[])).is_err());
        for length in 0..bytes.len() {
            assert!(Setup::parse(&bytes[..length], 2, 1, 0).is_err());
        }
        assert!(Setup::parse(&bytes, 0, 1, 0).is_err());
        assert!(Setup::parse(&bytes, 2, 0, 0).is_err());
    }

    #[test]
    fn reserved_fields_references_and_framing_are_rejected() {
        let fields = minimal_setup();
        for (index, value) in [
            (9, 1),
            (11, 2),
            (16, 3),
            (21, 1),
            (25, 1),
            (28, 1),
            (30, 1),
            (31, 1),
            (34, 1),
            (35, 1),
            (36, 1),
            (38, 1),
            (39, 1),
            (40, 1),
            (41, 0),
        ] {
            let mut changed = fields.clone();
            changed[index].0 = value;
            assert!(
                Setup::parse(&header(&changed), 2, 1, 0).is_err(),
                "field {index}"
            );
        }
    }

    #[test]
    fn floor_points_and_coupling_channels_are_checked() {
        let book = Codebook::parse(
            &mut PacketBits::new(&packet(&[
                (0x564342, 24),
                (1, 16),
                (1, 24),
                (0, 1),
                (0, 1),
                (0, 5),
                (0, 4),
            ])),
            &mut 1,
            &mut 0,
        )
        .unwrap();
        let books = [book];
        let prefix = [
            (1, 16),
            (1, 5),
            (0, 4),
            (0, 3),
            (0, 2),
            (0, 8),
            (0, 2),
            (6, 4),
        ];
        for point in [0, 32] {
            let mut fields = prefix.to_vec();
            fields.push((point, 6));
            let bytes = packet(&fields);
            assert_eq!(
                Floor::parse(&mut PacketBits::new(&bytes), &books).is_ok(),
                point == 32
            );
        }
        for angle in [0, 1, 3] {
            let bytes = packet(&[
                (0, 16),
                (0, 1),
                (1, 1),
                (0, 8),
                (0, 2),
                (angle, 2),
                (0, 2),
                (0, 8),
                (0, 8),
                (0, 8),
            ]);
            assert_eq!(
                Mapping::parse(&mut PacketBits::new(&bytes), 3, 1, 1).is_ok(),
                angle == 1
            );
        }
    }
}
