//! Incremental AV1 low-overhead bitstream framing.
//!
//! This is the byte-stream boundary for a local AV1 decoder, not a pixel decoder.

const MAX_OBU_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Obu {
    pub kind: u8,
    pub temporal_id: u8,
    pub spatial_id: u8,
    pub payload: Vec<u8>,
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
