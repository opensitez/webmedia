//! Vorbis I identification and Matroska codec-header unpacking.
//! Floor-zero and floor-one streams support PCM synthesis.

use super::webm::split_lace;
use crate::video::backend::MediaDecodeError;

pub mod entropy;
pub mod decoder;
pub mod floor;
pub mod floor_zero;
pub mod residue;
pub mod setup;
pub mod spectrum;
pub mod transform;

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.into())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IdentificationHeader {
    pub channels: u8,
    pub sample_rate: u32,
    pub block_sizes: [usize; 2],
}

impl IdentificationHeader {
    pub fn parse(bytes: &[u8]) -> Result<Self, MediaDecodeError> {
        if bytes.len() < 30 || &bytes[..7] != b"\x01vorbis" {
            return Err(invalid("invalid Vorbis identification header"));
        }
        if bytes[7..11] != [0, 0, 0, 0] {
            return Err(MediaDecodeError::Unsupported);
        }
        let channels = bytes[11];
        let sample_rate = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let small = bytes[28] & 15;
        let large = bytes[28] >> 4;
        if channels == 0
            || sample_rate == 0
            || !(6..=13).contains(&small)
            || !(small..=13).contains(&large)
            || bytes[29] & 1 == 0
        {
            return Err(invalid("invalid Vorbis identification parameters"));
        }
        Ok(Self {
            channels,
            sample_rate,
            block_sizes: [1 << small, 1 << large],
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Headers {
    pub identification: IdentificationHeader,
    pub comment: Vec<u8>,
    pub setup: Vec<u8>,
}

impl Headers {
    /// Decode the codebook prefix of setup. Floors, residues, mappings and modes
    /// follow at the returned bit offset and are not decoded by this method.
    pub fn codebooks(
        &self,
        mut entry_budget: usize,
        mut lookup_budget: usize,
    ) -> Result<(Vec<entropy::Codebook>, usize), MediaDecodeError> {
        if !self.setup.starts_with(b"\x05vorbis") {
            return Err(invalid("invalid Vorbis setup signature"));
        }
        let mut bits = entropy::PacketBits::new(&self.setup[7..]);
        let books = setup::read_codebooks(&mut bits, &mut entry_budget, &mut lookup_budget)?;
        Ok((books, bits.position()))
    }

    pub fn decode_setup(
        &self,
        entry_budget: usize,
        lookup_budget: usize,
    ) -> Result<setup::Setup, MediaDecodeError> {
        setup::Setup::parse(&self.setup, self.identification.channels, entry_budget, lookup_budget)
    }

    pub fn from_webm(bytes: &[u8]) -> Result<Self, MediaDecodeError> {
        if bytes.first() != Some(&2) {
            return Err(invalid("Vorbis needs three codec headers"));
        }
        let mut packets = split_lace(bytes, 2)?.into_iter();
        let identification = IdentificationHeader::parse(&packets.next().unwrap())?;
        let comment = packets.next().unwrap();
        let setup = packets.next().unwrap();
        if !comment.starts_with(b"\x03vorbis")
            || !setup.starts_with(b"\x05vorbis")
            || setup.len() < 8
        {
            return Err(invalid("invalid Vorbis header order or signature"));
        }
        Ok(Self {
            identification,
            comment,
            setup,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identification() -> Vec<u8> {
        let mut bytes = b"\x01vorbis".to_vec();
        bytes.extend(0u32.to_le_bytes());
        bytes.push(2);
        bytes.extend(48000u32.to_le_bytes());
        bytes.extend([0; 12]);
        bytes.extend([0xb8, 1]);
        bytes
    }

    #[test]
    fn parses_vorbis_identification_and_rejects_invalid_sizes() {
        let mut bytes = identification();
        assert_eq!(
            IdentificationHeader::parse(&bytes).unwrap(),
            IdentificationHeader {
                channels: 2,
                sample_rate: 48000,
                block_sizes: [256, 2048],
            }
        );
        for length in 0..30 {
            assert!(IdentificationHeader::parse(&bytes[..length]).is_err());
        }
        for sizes in [0x55, 0x6b, 0xee] {
            bytes[28] = sizes;
            assert!(IdentificationHeader::parse(&bytes).is_err());
        }
        bytes[28] = 0xb8;
        bytes[29] = 0;
        assert!(IdentificationHeader::parse(&bytes).is_err());
    }

    #[test]
    fn extracts_three_headers_and_checks_signatures() {
        let mut bytes = vec![2, 30, 7];
        bytes.extend(identification());
        bytes.extend(b"\x03vorbis");
        bytes.extend(b"\x05vorbis\x01");
        let headers = Headers::from_webm(&bytes).unwrap();
        assert_eq!(headers.identification.channels, 2);
        assert_eq!(headers.setup, b"\x05vorbis\x01");
        bytes[33] = 5;
        assert!(Headers::from_webm(&bytes).is_err());
        assert!(Headers::from_webm(&[2, 255]).is_err());
        let mut malformed = headers;
        malformed.setup.clear();
        assert!(malformed.codebooks(1024, 1024).is_err());
    }
}
