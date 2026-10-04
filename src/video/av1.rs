//! Incremental AV1 low-overhead bitstream framing.
//!
//! Framing remains independent of the staged, clean-room decoder components.

#[path = "av1/blend.rs"]
mod blend;
#[path = "av1/blend_tables.rs"]
mod blend_tables;
#[path = "av1/coefficients.rs"]
mod coefficients;
#[path = "av1/decoder.rs"]
mod decoder;
#[path = "av1/entropy.rs"]
pub mod entropy;
#[path = "av1/filters.rs"]
mod filters;
#[path = "av1/inter.rs"]
mod inter;
#[path = "av1/inter_tables.rs"]
mod inter_tables;
#[path = "av1/intra.rs"]
mod intra;
#[path = "av1/motion.rs"]
mod motion;
#[path = "av1/motion_tables.rs"]
mod motion_tables;
#[path = "av1/prediction.rs"]
mod prediction;
#[cfg(test)]
#[path = "av1/profile.rs"]
mod profile;
#[path = "av1/reconstruction.rs"]
pub mod reconstruction;
#[path = "av1/restoration.rs"]
mod restoration;
#[path = "av1/syntax.rs"]
pub mod syntax;
#[path = "av1/tables.rs"]
mod tables;
#[path = "av1/temporal.rs"]
mod temporal;
#[path = "av1/transform.rs"]
mod transform;
#[path = "av1/warp.rs"]
mod warp;
#[path = "av1/warp_tables.rs"]
mod warp_tables;
pub use decoder::inspect_intra_decode;
pub use decoder::{Av1Decoder, DecodedFrame, DecodedIntraFrame, DecodedPlane, decode_intra_frame};
pub use intra::{IntraBlockInfo, IntraDecodeProgress};

const MAX_OBU_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Obu {
    pub kind: u8,
    pub temporal_id: u8,
    pub spatial_id: u8,
    pub payload: Vec<u8>,
}

/// Parsed coded intra frame, deliberately distinct from a decoded pixel frame.
pub struct CodedIntraFrame<'a> {
    pub header: syntax::IntraFrameHeader,
    pub tiles: Vec<(usize, &'a [u8])>,
}

impl<'a> CodedIntraFrame<'a> {
    pub fn parse(obu: &'a Obu, sequence: &syntax::SequenceHeader) -> Result<Self, syntax::Error> {
        if obu.kind != 6 {
            return Err(syntax::Error::Unsupported("expected combined OBU_FRAME"));
        }
        let header = syntax::IntraFrameHeader::parse(
            &obu.payload,
            sequence,
            obu.temporal_id,
            obu.spatial_id,
        )?;
        let tiles = syntax::tile_group(&obu.payload[header.header_bytes..], &header.tiles)?;
        if tiles.len() != header.tiles.count() {
            return Err(syntax::Error::Invalid(
                "combined frame must contain all tiles",
            ));
        }
        Ok(Self { header, tiles })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObuStreamError {
    InvalidHeader,
    InvalidSize,
    TooLarge,
    Incomplete,
}

#[derive(Default)]
pub struct ObuStream {
    pending: Vec<u8>,
}

impl ObuStream {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, input: &[u8]) -> Result<Vec<Obu>, ObuStreamError> {
        let mut complete = Vec::new();
        for chunk in input.chunks(16 * 1024) {
            self.pending.extend_from_slice(chunk);
            self.consume(&mut complete)?;
            if self.pending.len() > MAX_OBU_BYTES + 10 {
                return Err(ObuStreamError::TooLarge);
            }
        }
        Ok(complete)
    }

    pub fn finish(&self) -> Result<(), ObuStreamError> {
        if self.pending.is_empty() {
            Ok(())
        } else {
            Err(ObuStreamError::Incomplete)
        }
    }

    fn consume(&mut self, output: &mut Vec<Obu>) -> Result<(), ObuStreamError> {
        let mut offset = 0;
        while offset < self.pending.len() {
            let header = self.pending[offset];
            if header & 0x81 != 0 || header & 0x02 == 0 {
                return Err(ObuStreamError::InvalidHeader);
            }
            let kind = (header >> 3) & 0x0f;
            let extended = header & 0x04 != 0;
            let mut cursor = offset + 1;
            let (temporal_id, spatial_id) = if extended {
                let Some(&extension) = self.pending.get(cursor) else {
                    break;
                };
                if extension & 0x07 != 0 {
                    return Err(ObuStreamError::InvalidHeader);
                }
                cursor += 1;
                ((extension >> 5) & 0x07, (extension >> 3) & 0x03)
            } else {
                (0, 0)
            };
            let mut size = 0u64;
            let mut size_complete = false;
            for index in 0..8 {
                let Some(&byte) = self.pending.get(cursor) else {
                    break;
                };
                cursor += 1;
                size |= u64::from(byte & 0x7f) << (index * 7);
                if byte & 0x80 == 0 {
                    size_complete = true;
                    break;
                }
                if index == 7 {
                    return Err(ObuStreamError::InvalidSize);
                }
            }
            if !size_complete {
                break;
            }
            if size > u32::MAX as u64 {
                return Err(ObuStreamError::InvalidSize);
            }
            if size > MAX_OBU_BYTES as u64 {
                return Err(ObuStreamError::TooLarge);
            }
            let end = cursor + size as usize;
            if end > self.pending.len() {
                break;
            }
            output.push(Obu {
                kind,
                temporal_id,
                spatial_id,
                payload: self.pending[cursor..end].to_vec(),
            });
            offset = end;
        }
        if offset > 0 {
            self.pending.drain(..offset);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_units_as_soon_as_their_payloads_arrive() {
        let bytes = [0x0a, 0x03, 1, 2, 3, 0x36, 0x28, 0x01, 9];
        for split in 1..bytes.len() {
            let mut stream = ObuStream::new();
            let mut units = stream.push(&bytes[..split]).unwrap();
            units.extend(stream.push(&bytes[split..]).unwrap());
            assert_eq!(units.len(), 2, "split at {split}");
            assert_eq!(
                (units[0].kind, units[0].payload.as_slice()),
                (1, &[1, 2, 3][..])
            );
            assert_eq!(
                (units[1].kind, units[1].temporal_id, units[1].spatial_id),
                (6, 1, 1)
            );
            assert_eq!(units[1].payload, [9]);
            stream.finish().unwrap();
        }
    }

    #[test]
    fn waits_for_split_leb128_and_rejects_truncation() {
        let mut stream = ObuStream::new();
        assert!(stream.push(&[0x32, 0x81]).unwrap().is_empty());
        assert_eq!(stream.finish(), Err(ObuStreamError::Incomplete));
        let mut rest = vec![0x01];
        rest.extend(vec![7; 129]);
        let units = stream.push(&rest).unwrap();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].payload.len(), 129);
        stream.finish().unwrap();
    }

    #[test]
    fn rejects_invalid_headers_and_oversized_units() {
        for bytes in [&[0x80][..], &[0x08][..], &[0x0b][..], &[0x0e, 0x01][..]] {
            assert_eq!(
                ObuStream::new().push(bytes),
                Err(ObuStreamError::InvalidHeader)
            );
        }
        assert_eq!(
            ObuStream::new().push(&[0x0a, 0x80, 0x80, 0x80, 0x21]),
            Err(ObuStreamError::TooLarge)
        );
        assert_eq!(
            ObuStream::new().push(&[0x0a, 0x80, 0x80, 0x80, 0x80, 0x10]),
            Err(ObuStreamError::InvalidSize)
        );
    }
}
