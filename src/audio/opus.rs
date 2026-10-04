//! Opus stream identification (RFC 7845 section 5.1) and packet framing
//! (RFC 6716 section 3), entropy and CELT spectral synthesis stages.
//! `music::MusicPacketDecoder` reconstructs music and silence for mapping-family
//! zero, matching mono/stereo, single-frame configuration-31 packets (fullband
//! CELT, 20 ms at 48 kHz). Whole-track stereo Partyfire PCM is oracle-verified.
//! Other configurations, multi-frame packets, SILK/hybrid, and multistream
//! decoding remain unsupported. Container trimming is the caller's responsibility.

use crate::video::backend::MediaDecodeError;

#[path = "opus/allocation.rs"]
pub mod allocation;
#[path = "opus/celt.rs"]
pub mod celt;
#[path = "opus/energy.rs"]
pub mod energy;
#[path = "opus/frame.rs"]
pub mod frame;
#[path = "opus/music.rs"]
pub mod music;
#[path = "opus/pvq.rs"]
pub mod pvq;
#[path = "opus/range.rs"]
pub mod range;
#[path = "opus/residual.rs"]
pub mod residual;
#[path = "opus/silence.rs"]
pub mod silence;
#[path = "opus/simd.rs"]
pub mod simd;
#[path = "opus/synthesis.rs"]
pub mod synthesis;
#[path = "opus/tf.rs"]
pub mod tf;

#[cfg(test)]
#[path = "opus/fixtures.rs"]
mod fixtures;

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.into())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IdentificationHeader {
    pub channels: u8,
    pub pre_skip: u16,
    pub input_sample_rate: u32,
    pub output_gain_q8: i16,
    pub mapping_family: u8,
    pub streams: u8,
    pub coupled_streams: u8,
    pub channel_mapping: Vec<u8>,
}

impl IdentificationHeader {
    pub fn parse(bytes: &[u8]) -> Result<Self, MediaDecodeError> {
        if bytes.len() < 19 || &bytes[..8] != b"OpusHead" {
            return Err(invalid("invalid Opus identification header"));
        }
        if bytes[8] > 15 {
            return Err(MediaDecodeError::Unsupported);
        }
        let channels = bytes[9];
        if channels == 0 {
            return Err(invalid("Opus channel count is zero"));
        }
        let mapping_family = bytes[18];
        let (streams, coupled_streams, channel_mapping) = if mapping_family == 0 {
            if channels > 2 {
                return Err(invalid("invalid Opus family-zero channel count"));
            }
            (1, channels - 1, (0..channels).collect())
        } else {
            let mapping = bytes
                .get(19..21 + usize::from(channels))
                .ok_or_else(|| invalid("truncated Opus channel mapping"))?;
            let streams = mapping[0];
            let coupled = mapping[1];
            let encoded_channels = u16::from(streams) + u16::from(coupled);
            if streams == 0
                || coupled > streams
                || encoded_channels > 255
                || (mapping_family == 1 && channels > 8)
                || mapping[2..]
                    .iter()
                    .any(|&channel| channel != 255 && u16::from(channel) >= encoded_channels)
            {
                return Err(invalid("invalid Opus channel mapping"));
            }
            (streams, coupled, mapping[2..].to_vec())
        };
        Ok(Self {
            channels,
            pre_skip: u16::from_le_bytes(bytes[10..12].try_into().unwrap()),
            input_sample_rate: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            output_gain_q8: i16::from_le_bytes(bytes[16..18].try_into().unwrap()),
            mapping_family,
            streams,
            coupled_streams,
            channel_mapping,
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Packet<'a> {
    pub configuration: u8,
    pub stereo: bool,
    pub frame_samples_48khz: u16,
    pub frames: Vec<&'a [u8]>,
}

fn frame_length(bytes: &[u8], cursor: &mut usize, end: usize) -> Result<usize, MediaDecodeError> {
    if *cursor >= end {
        return Err(invalid("truncated Opus frame length"));
    }
    let first = usize::from(bytes[*cursor]);
    *cursor += 1;
    if first < 252 {
        return Ok(first);
    }
    if *cursor >= end {
        return Err(invalid("truncated Opus frame length"));
    }
    let length = first + 4 * usize::from(bytes[*cursor]);
    *cursor += 1;
    Ok(length)
}

impl<'a> Packet<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, MediaDecodeError> {
        let toc = *bytes.first().ok_or_else(|| invalid("empty Opus packet"))?;
        let configuration = toc >> 3;
        let frame_samples_48khz = match configuration {
            0..=11 => [480, 960, 1920, 2880][usize::from(configuration & 3)],
            12..=15 => [480, 960][usize::from(configuration & 1)],
            _ => [120, 240, 480, 960][usize::from(configuration & 3)],
        };
        let mut cursor = 1usize;
        let mut end = bytes.len();
        let mut sizes = [0usize; 48];
        let (count, variable) = match toc & 3 {
            0 => (1, false),
            1 => (2, false),
            2 => (2, true),
            _ => {
                let control = *bytes
                    .get(cursor)
                    .ok_or_else(|| invalid("missing Opus frame count"))?;
                cursor += 1;
                let count = usize::from(control & 63);
                if count == 0 || count > 48 {
                    return Err(invalid("invalid Opus frame count"));
                }
                if control & 64 != 0 {
                    let mut padding = 0usize;
                    loop {
                        let byte = *bytes
                            .get(cursor)
                            .ok_or_else(|| invalid("truncated Opus padding"))?;
                        cursor += 1;
                        padding = padding
                            .checked_add(if byte == 255 { 254 } else { usize::from(byte) })
                            .ok_or_else(|| invalid("Opus padding overflow"))?;
                        if byte != 255 {
                            break;
                        }
                    }
                    end = end
                        .checked_sub(padding)
                        .filter(|&end| end >= cursor)
                        .ok_or_else(|| invalid("Opus padding exceeds packet"))?;
                }
                (count, control & 128 != 0)
            }
        };
        if usize::from(frame_samples_48khz) * count > 5760 {
            return Err(invalid("Opus packet exceeds 120 milliseconds"));
        }
        if variable {
            let mut total = 0usize;
            for size in &mut sizes[..count - 1] {
                *size = frame_length(bytes, &mut cursor, end)?;
                total += *size;
            }
            sizes[count - 1] = end
                .checked_sub(cursor)
                .and_then(|remaining| remaining.checked_sub(total))
                .ok_or_else(|| invalid("Opus frame lengths exceed packet"))?;
        } else {
            let available = end - cursor;
            if available % count != 0 {
                return Err(invalid("unequal Opus CBR frames"));
            }
            sizes[..count].fill(available / count);
        }
        let mut frames = Vec::with_capacity(count);
        for &size in &sizes[..count] {
            if size > 1275 {
                return Err(invalid("Opus frame exceeds 1275 bytes"));
            }
            frames.push(&bytes[cursor..cursor + size]);
            cursor += size;
        }
        Ok(Self {
            configuration,
            stereo: toc & 4 != 0,
            frame_samples_48khz,
            frames,
        })
    }

    pub fn samples_48khz(&self) -> usize {
        usize::from(self.frame_samples_48khz) * self.frames.len()
    }

    pub fn celt_layout(&self) -> Result<Option<celt::Layout>, MediaDecodeError> {
        celt::Layout::from_configuration(self.configuration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_stereo_and_mapped_surround_headers() {
        let mut bytes = b"OpusHead\x01\x02\x38\x01\x80\xbb\x00\x00\x00\xff\x00".to_vec();
        let header = IdentificationHeader::parse(&bytes).unwrap();
        assert_eq!(
            (header.channels, header.pre_skip, header.input_sample_rate),
            (2, 312, 48000)
        );
        assert_eq!(header.output_gain_q8, -256);
        assert_eq!(header.channel_mapping, [0, 1]);
        bytes[9] = 6;
        bytes[18] = 1;
        bytes.extend([4, 2, 0, 4, 1, 2, 3, 5]);
        assert_eq!(
            IdentificationHeader::parse(&bytes).unwrap().channel_mapping,
            [0, 4, 1, 2, 3, 5]
        );
        for length in 0..bytes.len() {
            assert!(IdentificationHeader::parse(&bytes[..length]).is_err());
        }
        bytes[21] = 6;
        assert!(IdentificationHeader::parse(&bytes).is_err());
    }

    #[test]
    fn splits_all_opus_packing_codes_without_copying_frames() {
        assert_eq!(
            Packet::parse(&[0xf8, 10, 20]).unwrap().frames,
            [&[10, 20][..]]
        );
        assert_eq!(
            Packet::parse(&[0xf9, 10, 20, 30, 40]).unwrap().frames,
            [&[10, 20][..], &[30, 40][..]]
        );
        assert_eq!(
            Packet::parse(&[0xfa, 1, 10, 20, 30]).unwrap().frames,
            [&[10][..], &[20, 30][..]]
        );
        let packet = Packet::parse(&[0xfb, 0xc3, 1, 1, 2, 10, 20, 30, 40, 99]).unwrap();
        assert_eq!(packet.frames, [&[10][..], &[20, 30][..], &[40][..]]);
        assert_eq!(packet.samples_48khz(), 2880);
        assert_eq!(Packet::parse(&[0x83, 48]).unwrap().samples_48khz(), 5760);
        assert_eq!(Packet::parse(&[0xf8]).unwrap().frames, [&[][..]]);
    }

    #[test]
    fn validates_duration_padding_and_frame_size_limits() {
        for bytes in [
            &[][..],
            &[0xf9, 1][..],
            &[0xfa][..],
            &[0xfa, 10, 1][..],
            &[0xfb][..],
            &[0xfb, 0][..],
            &[0xfb, 7][..],
            &[0xfb, 65, 255][..],
            &[0xfb, 65, 5, 0][..],
        ] {
            assert!(Packet::parse(bytes).is_err(), "{bytes:?}");
        }
        assert_eq!(Packet::parse(&[0xfb, 129]).unwrap().frames, [&[][..]]);
        assert!(Packet::parse(&vec![0xf8; 1277]).is_err());
        let mut padded = vec![0xfb, 65, 255, 0, 7];
        padded.extend([1; 254]);
        assert_eq!(Packet::parse(&padded).unwrap().frames, [&[7][..]]);
        let mut vbr = vec![0xfa, 252, 0];
        vbr.extend([5; 252]);
        vbr.push(9);
        assert_eq!(
            Packet::parse(&vbr)
                .unwrap()
                .frames
                .iter()
                .map(|frame| frame.len())
                .collect::<Vec<_>>(),
            [252, 1]
        );
    }

    #[test]
    fn arbitrary_packets_never_panic_or_exceed_decoder_bounds() {
        let mut seed = 0x538a_f713u32;
        for length in 0..128 {
            for _ in 0..32 {
                let bytes: Vec<u8> = (0..length)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 17;
                        seed ^= seed << 5;
                        seed as u8
                    })
                    .collect();
                if let Ok(packet) = Packet::parse(&bytes) {
                    assert!(!packet.frames.is_empty() && packet.frames.len() <= 48);
                    assert!(packet.samples_48khz() <= 5760);
                    assert!(packet.frames.iter().all(|frame| frame.len() <= 1275));
                    assert!(
                        packet.frames.iter().map(|frame| frame.len()).sum::<usize>() <= bytes.len()
                    );
                }
                let _ = IdentificationHeader::parse(&bytes);
            }
        }
    }
}
