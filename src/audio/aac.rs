//! AAC configuration from ISO/IEC 14496-3:2005, tables 1.13 and 4.1.

use crate::video::backend::MediaDecodeError;

mod bands;
mod decoder;
mod entropy;
mod frame;
mod noise;
mod prediction;
mod spectral;
mod synthesis;
mod tables;
pub use decoder::AacDecoder;
pub use frame::{AacChannel, AacFrame, IcsInfo, TnsFilterData};
pub use synthesis::AacSynthesis;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AacConfig {
    pub object_type: u8,
    pub sample_rate: u32,
    pub frequency_index: u8,
    pub channels: u16,
    pub frame_samples: usize,
}

pub(crate) struct Bits<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Bits<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub(crate) fn read(&mut self, width: usize) -> Result<u32, MediaDecodeError> {
        let end = self
            .position
            .checked_add(width)
            .filter(|_| width <= 32)
            .ok_or_else(|| invalid("invalid AAC bit width"))?;
        if end > self.bytes.len().saturating_mul(8) {
            return Err(invalid("truncated AAC bits"));
        }
        let mut value = 0;
        for bit in self.position..end {
            value = (value << 1) | u32::from((self.bytes[bit / 8] >> (7 - bit % 8)) & 1);
        }
        self.position = end;
        Ok(value)
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_mul(8) - self.position
    }

    fn skip(&mut self, mut width: usize) -> Result<(), MediaDecodeError> {
        if width > self.remaining() {
            return Err(invalid("truncated AAC payload"));
        }
        while width != 0 {
            let count = width.min(32);
            self.read(count)?;
            width -= count;
        }
        Ok(())
    }

    fn align(&mut self) -> Result<(), MediaDecodeError> {
        self.skip((8 - self.position % 8) % 8)
    }
}

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.into())
}

impl AacConfig {
    /// Main/LC configuration. Unsupported tools are rejected, not silently ignored.
    pub fn parse(bytes: &[u8]) -> Result<Self, MediaDecodeError> {
        let mut bits = Bits::new(bytes);
        let mut object_type = bits.read(5)?;
        if object_type == 31 {
            object_type = 32 + bits.read(6)?;
        }
        let frequency_index = bits.read(4)? as u8;
        let rates = [
            96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
        ];
        let sample_rate = if frequency_index == 15 {
            bits.read(24)?
        } else {
            *rates
                .get(usize::from(frequency_index))
                .ok_or_else(|| invalid("reserved AAC frequency index"))?
        };
        if sample_rate == 0 {
            return Err(invalid("zero AAC sample rate"));
        }
        let channel_configuration = bits.read(4)?;
        if !matches!(object_type, 1 | 2) {
            return Err(MediaDecodeError::Unsupported);
        }
        let channels = match channel_configuration {
            1 => 1,
            2 => 2,
            _ => return Err(MediaDecodeError::Unsupported),
        };
        let frame_samples = if bits.read(1)? == 0 { 1024 } else { 960 };
        if bits.read(1)? != 0 {
            bits.read(14)?;
            return Err(MediaDecodeError::Unsupported);
        }
        if bits.read(1)? != 0 && bits.read(1)? != 0 {
            return Err(MediaDecodeError::Unsupported);
        }
        if bits.remaining() >= 16 && bits.read(11)? == 0x2b7 {
            let extension_type = bits.read(5)?;
            if extension_type == 5 && bits.read(1)? != 0 {
                return Err(MediaDecodeError::Unsupported);
            }
        }
        Ok(Self {
            object_type: object_type as u8,
            sample_rate,
            frequency_index,
            channels,
            frame_samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_real_mp4_lc_configuration() {
        assert_eq!(
            AacConfig::parse(&[0x12, 0x10, 0x56, 0xe5, 0x00]).unwrap(),
            AacConfig {
                object_type: 2,
                sample_rate: 44100,
                frequency_index: 4,
                channels: 2,
                frame_samples: 1024
            }
        );
        assert_eq!(AacConfig::parse(&[0x11, 0x88]).unwrap().channels, 1);
        assert_eq!(AacConfig::parse(&[0x12, 0x14]).unwrap().frame_samples, 960);
        assert_eq!(AacConfig::parse(&[0x09, 0x88]).unwrap().object_type, 1);
    }

    #[test]
    fn rejects_missing_or_reserved_configuration() {
        for bytes in [&[][..], &[0x12][..], &[0x16, 0x90][..], &[0x12, 0][..]] {
            assert!(AacConfig::parse(bytes).is_err());
        }
        assert!(AacConfig::parse(&[0x2b, 0x92, 0x08]).is_err());
    }

    #[test]
    fn truncated_reads_do_not_advance() {
        let mut bits = Bits::new(&[0xa5]);
        assert!(bits.read(9).is_err());
        assert_eq!(bits.read(4).unwrap(), 10);
        assert!(bits.read(33).is_err());
        assert_eq!(bits.read(4).unwrap(), 5);
    }
}
