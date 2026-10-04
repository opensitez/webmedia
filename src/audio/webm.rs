//! WebM audio track metadata and bounded Matroska packet lacing.

use crate::video::backend::{AudioSamples, MediaDecodeError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WebmAudioCodec {
    Opus,
    Vorbis,
}

impl WebmAudioCodec {
    pub(crate) fn from_id(id: &[u8]) -> Option<Self> {
        match id {
            b"A_OPUS" => Some(Self::Opus),
            b"A_VORBIS" => Some(Self::Vorbis),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WebmAudioTrack {
    pub number: u64,
    pub codec: WebmAudioCodec,
    /// Container sampling frequency, not necessarily the decoder output rate.
    pub sampling_frequency: f64,
    pub channels: u16,
    pub codec_private: Vec<u8>,
    pub codec_delay_ns: u64,
    pub seek_pre_roll_ns: u64,
    pub default_duration_ns: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebmAudioBlock {
    /// Timestamp of the first packet, before CodecDelay is subtracted.
    pub timestamp_ns: i64,
    pub packets: Vec<Vec<u8>>,
    /// Positive values trim the end; negative values trim the beginning.
    pub discard_padding_ns: Option<i64>,
}

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.into())
}

/// Convert container padding to whole sample frames, rounding to the nearest sample.
pub fn discard_padding_frames(
    padding_ns: i64,
    sample_rate: u32,
) -> Result<usize, MediaDecodeError> {
    if sample_rate == 0 {
        return Err(invalid("invalid WebM PCM sample rate"));
    }
    let frames = (u128::from(padding_ns.unsigned_abs()) * u128::from(sample_rate) + 500_000_000)
        / 1_000_000_000;
    usize::try_from(frames).map_err(|_| invalid("WebM discard padding sample count overflow"))
}

/// Trim PCM and return the number of sample frames removed.
pub fn trim_discard_padding(
    audio: &mut AudioSamples,
    padding_ns: i64,
) -> Result<usize, MediaDecodeError> {
    let channels = usize::from(audio.channels);
    if channels == 0 || audio.sample_rate == 0 || audio.samples.len() % channels != 0 {
        return Err(invalid("invalid WebM PCM shape"));
    }
    let frames = discard_padding_frames(padding_ns, audio.sample_rate)?;
    if frames > audio.samples.len() / channels {
        return Err(invalid("WebM discard padding exceeds decoded block"));
    }
    let samples = frames * channels;
    if padding_ns < 0 {
        audio.samples.copy_within(samples.., 0);
    }
    audio.samples.truncate(audio.samples.len() - samples);
    Ok(frames)
}

fn lace_integer(bytes: &[u8], cursor: &mut usize) -> Result<(u64, usize), MediaDecodeError> {
    let first = *bytes
        .get(*cursor)
        .ok_or_else(|| invalid("truncated EBML lace"))?;
    if first == 0 {
        return Err(invalid("invalid EBML lace integer"));
    }
    let length = first.leading_zeros() as usize + 1;
    let end = cursor
        .checked_add(length)
        .ok_or_else(|| invalid("EBML lace overflow"))?;
    let value = bytes
        .get(*cursor..end)
        .ok_or_else(|| invalid("truncated EBML lace"))?;
    let mut result = u64::from(first & (0xffu16 >> length) as u8);
    for &byte in &value[1..] {
        result = (result << 8) | u64::from(byte);
    }
    *cursor = end;
    Ok((result, length))
}

pub(crate) fn split_lace(bytes: &[u8], flags: u8) -> Result<Vec<Vec<u8>>, MediaDecodeError> {
    let kind = (flags >> 1) & 3;
    if kind == 0 {
        return Ok(vec![bytes.to_vec()]);
    }
    let count = usize::from(*bytes.first().ok_or_else(|| invalid("missing lace count"))?) + 1;
    if count < 2 {
        return Err(invalid("lacing requires at least two packets"));
    }
    let mut cursor = 1usize;
    let mut sizes = Vec::with_capacity(count);
    if kind == 2 {
        let available = bytes.len() - cursor;
        if available % count != 0 {
            return Err(invalid("unequal fixed lace packet sizes"));
        }
        sizes.resize(count, available / count);
    } else {
        if kind == 1 {
            for _ in 0..count - 1 {
                let mut size = 0usize;
                loop {
                    let byte = *bytes
                        .get(cursor)
                        .ok_or_else(|| invalid("truncated Xiph lace"))?;
                    cursor += 1;
                    size = size
                        .checked_add(usize::from(byte))
                        .ok_or_else(|| invalid("Xiph lace overflow"))?;
                    if byte != 255 {
                        break;
                    }
                }
                sizes.push(size);
            }
        } else {
            let (first, _) = lace_integer(bytes, &mut cursor)?;
            let first = usize::try_from(first).map_err(|_| invalid("EBML lace size overflow"))?;
            sizes.push(first);
            for _ in 1..count - 1 {
                let (value, length) = lace_integer(bytes, &mut cursor)?;
                let bias = (1i128 << (7 * length - 1)) - 1;
                let size = *sizes.last().unwrap() as i128 + i128::from(value) - bias;
                sizes.push(usize::try_from(size).map_err(|_| invalid("invalid EBML lace delta"))?);
            }
        }
        let total = sizes
            .iter()
            .try_fold(0usize, |sum, &size| sum.checked_add(size))
            .ok_or_else(|| invalid("lace size overflow"))?;
        let last = bytes
            .len()
            .checked_sub(cursor)
            .and_then(|available| available.checked_sub(total))
            .ok_or_else(|| invalid("lace packets exceed block"))?;
        sizes.push(last);
    }
    let mut packets = Vec::with_capacity(count);
    for size in sizes {
        let end = cursor
            .checked_add(size)
            .ok_or_else(|| invalid("lace size overflow"))?;
        packets.push(
            bytes
                .get(cursor..end)
                .ok_or_else(|| invalid("truncated lace packet"))?
                .to_vec(),
        );
        cursor = end;
    }
    Ok(packets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discard_padding_trims_whole_interleaved_frames() {
        let original = AudioSamples {
            sample_rate: 48000,
            channels: 2,
            samples: (0..12).map(|value| value as f32).collect(),
        };
        let mut end = original.clone();
        assert_eq!(trim_discard_padding(&mut end, 41667).unwrap(), 2);
        assert_eq!(end.samples, original.samples[..8]);
        let mut start = original.clone();
        assert_eq!(trim_discard_padding(&mut start, -41667).unwrap(), 2);
        assert_eq!(start.samples, original.samples[4..]);
        assert!(trim_discard_padding(&mut start, i64::MIN).is_err());
        assert!(trim_discard_padding(&mut start, i64::MAX).is_err());
        assert_eq!(trim_discard_padding(&mut start, 0).unwrap(), 0);
        start.channels = 0;
        assert!(trim_discard_padding(&mut start, 0).is_err());
    }

    #[test]
    fn splits_all_matroska_lacing_modes() {
        let expected = vec![vec![1, 2], vec![3, 4, 5], vec![6]];
        assert_eq!(
            split_lace(&[2, 2, 3, 1, 2, 3, 4, 5, 6], 2).unwrap(),
            expected
        );
        assert_eq!(
            split_lace(&[2, 0x82, 0xc0, 1, 2, 3, 4, 5, 6], 6).unwrap(),
            expected
        );
        assert_eq!(
            split_lace(&[1, 1, 2, 3, 4], 4).unwrap(),
            vec![vec![1, 2], vec![3, 4]]
        );
        assert_eq!(split_lace(&[1, 2, 3], 0).unwrap(), vec![vec![1, 2, 3]]);
    }

    #[test]
    fn signed_ebml_lace_and_xiph_continuations_are_bounded() {
        assert_eq!(
            split_lace(&[2, 0x83, 0xbe, 1, 2, 3, 4, 5, 6], 6).unwrap(),
            vec![vec![1, 2, 3], vec![4, 5], vec![6]]
        );
        let mut xiph = vec![1, 255, 1];
        xiph.extend([7; 256]);
        xiph.push(9);
        assert_eq!(split_lace(&xiph, 2).unwrap(), vec![vec![7; 256], vec![9]]);
        for (bytes, flags) in [
            (&[][..], 2),
            (&[0][..], 4),
            (&[1, 255][..], 2),
            (&[1, 4, 1][..], 2),
            (&[1, 1][..], 4),
            (&[2, 0x81, 0x80][..], 6),
            (&[1, 0][..], 6),
            (&[1, 0x01][..], 6),
        ] {
            assert!(split_lace(bytes, flags).is_err(), "{bytes:?} flags={flags}");
        }
    }

    #[test]
    fn arbitrary_lacing_never_panics_or_expands_payloads() {
        let mut seed = 0x138d_73b5u32;
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
                for flags in [0, 2, 4, 6] {
                    if let Ok(packets) = split_lace(&bytes, flags) {
                        assert!(!packets.is_empty() && packets.len() <= 256);
                        assert!(packets.iter().map(Vec::len).sum::<usize>() <= bytes.len());
                    }
                }
            }
        }
    }
}
