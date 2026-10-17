//! Container-independent interleaved PCM conversion.
//! Matroska mappings: https://www.matroska.org/technical/codec_specs.html

use crate::video::backend::{AudioSamples, MediaDecodeError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    Unsigned,
    SignedLittleEndian,
    SignedBigEndian,
    FloatLittleEndian,
    FloatBigEndian,
}

#[derive(Clone, Copy, Debug)]
pub struct Format {
    encoding: Encoding,
    bits: u16,
    channels: u16,
    sample_rate: u32,
}

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.into())
}

impl Format {
    pub fn new(encoding: Encoding, bits: u16, channels: u16, sample_rate: u32)
        -> Result<Self, MediaDecodeError>
    {
        if channels == 0 || sample_rate == 0 {
            return Err(invalid("zero PCM channel count or sample rate"));
        }
        let supported = match encoding {
            Encoding::Unsigned => bits == 8,
            Encoding::SignedLittleEndian | Encoding::SignedBigEndian =>
                bits != 0 && bits <= 64 && bits % 8 == 0,
            Encoding::FloatLittleEndian | Encoding::FloatBigEndian => matches!(bits, 32 | 64),
        };
        if !supported { return Err(MediaDecodeError::Unsupported); }
        Ok(Self { encoding, bits, channels, sample_rate })
    }

    pub fn sample_rate(self) -> u32 { self.sample_rate }

    pub fn frames(self, bytes: &[u8]) -> Result<usize, MediaDecodeError> {
        let alignment = usize::from(self.bits / 8) * usize::from(self.channels);
        if bytes.len() % alignment != 0 {
            return Err(invalid("truncated interleaved PCM frame"));
        }
        Ok(bytes.len() / alignment)
    }

    pub fn decode(self, bytes: &[u8]) -> Result<AudioSamples, MediaDecodeError> {
        self.frames(bytes)?;
        let width = usize::from(self.bits / 8);
        let integer_scale = 2f64.powi(i32::from(self.bits) - 1);
        let mut samples = Vec::new();
        samples.try_reserve_exact(bytes.len() / width)
            .map_err(|_| invalid("PCM output allocation failed"))?;
        for sample in bytes.chunks_exact(width) {
            let value = match self.encoding {
                Encoding::Unsigned => (f64::from(sample[0]) - 128.0) / 128.0,
                Encoding::SignedLittleEndian | Encoding::SignedBigEndian => {
                    let negative = if self.encoding == Encoding::SignedLittleEndian {
                        sample[width - 1] & 0x80 != 0
                    } else { sample[0] & 0x80 != 0 };
                    let mut expanded = [if negative { 255 } else { 0 }; 8];
                    let signed = if self.encoding == Encoding::SignedLittleEndian {
                        expanded[..width].copy_from_slice(sample);
                        i64::from_le_bytes(expanded)
                    } else {
                        expanded[8 - width..].copy_from_slice(sample);
                        i64::from_be_bytes(expanded)
                    };
                    signed as f64 / integer_scale
                }
                Encoding::FloatLittleEndian | Encoding::FloatBigEndian => {
                    let little = self.encoding == Encoding::FloatLittleEndian;
                    if self.bits == 32 {
                        let bytes: [u8; 4] = sample.try_into().unwrap();
                        f64::from(if little { f32::from_le_bytes(bytes) } else { f32::from_be_bytes(bytes) })
                    } else {
                        let bytes: [u8; 8] = sample.try_into().unwrap();
                        if little { f64::from_le_bytes(bytes) } else { f64::from_be_bytes(bytes) }
                    }
                }
            };
            if !value.is_finite() { return Err(invalid("nonfinite PCM float sample")); }
            samples.push(value.clamp(-1.0, 1.0) as f32);
        }
        Ok(AudioSamples { sample_rate: self.sample_rate, channels: self.channels, samples })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_widths_endianness_and_interleaving() {
        for bits in (8..=64).step_by(8) {
            let width = usize::from(bits / 8);
            for encoding in [Encoding::SignedLittleEndian, Encoding::SignedBigEndian] {
                let mut min = vec![0; width];
                min[if encoding == Encoding::SignedLittleEndian { width - 1 } else { 0 }] = 128;
                let mut bytes = min;
                bytes.extend(vec![0; width]);
                let format = Format::new(encoding, bits, 2, 48000).unwrap();
                assert_eq!(format.frames(&bytes).unwrap(), 1);
                assert_eq!(format.decode(&bytes).unwrap().samples, [-1.0, 0.0]);
                assert!(format.decode(&bytes[..bytes.len()-1]).is_err());
            }
        }
        assert_eq!(Format::new(Encoding::Unsigned, 8, 1, 8000).unwrap()
            .decode(&[0,128,255]).unwrap().samples, [-1.0,0.0,127.0/128.0]);
    }

    #[test]
    fn floats_endianness_saturation_and_invalid_formats() {
        for bits in [32,64] {
            for encoding in [Encoding::FloatLittleEndian, Encoding::FloatBigEndian] {
                let little = encoding == Encoding::FloatLittleEndian;
                let pack = |value: f64| if bits == 32 {
                    if little { (value as f32).to_le_bytes().to_vec() }
                    else { (value as f32).to_be_bytes().to_vec() }
                } else if little { value.to_le_bytes().to_vec() } else { value.to_be_bytes().to_vec() };
                let format = Format::new(encoding,bits,1,48000).unwrap();
                for (value,expected) in [(-2.0,-1.0),(-0.5,-0.5),(0.0,0.0),(2.0,1.0)] {
                    assert_eq!(format.decode(&pack(value)).unwrap().samples, [expected]);
                }
                for invalid in [f64::NAN,f64::INFINITY,f64::NEG_INFINITY] {
                    assert!(format.decode(&pack(invalid)).is_err());
                }
            }
        }
        assert!(Format::new(Encoding::Unsigned,16,1,48000).is_err());
        assert!(Format::new(Encoding::SignedLittleEndian,12,1,48000).is_err());
        assert!(Format::new(Encoding::SignedLittleEndian,16,0,48000).is_err());
        assert!(Format::new(Encoding::SignedLittleEndian,16,1,0).is_err());
    }
}
