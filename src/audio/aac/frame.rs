//! Bounded AAC Main/LC raw-data-block syntax, ISO 13818-7 tables 12-24.

use super::{AacConfig, Bits, bands, entropy, invalid, prediction};
use crate::video::backend::MediaDecodeError;

#[derive(Clone, Debug)]
pub struct IcsInfo {
    pub sequence: u8,
    pub shape: u8,
    pub max_sfb: usize,
    pub groups: Vec<usize>,
    pub offsets: &'static [u16],
    pub prediction_used: Vec<bool>,
    pub predictor_reset: Option<u8>,
}

#[derive(Clone, Debug)]
pub struct TnsFilterData {
    pub length: usize,
    pub reverse: bool,
    pub resolution: u8,
    pub compressed: bool,
    pub coefficients: Vec<u8>,
}

pub struct AacChannel {
    pub info: IcsInfo,
    pub codebooks: Vec<u8>,
    pub scalefactors: Vec<i16>,
    /// Window-major, after pulse modification and before inverse quantization.
    pub quantized: Vec<i32>,
    pub tns: Vec<Vec<TnsFilterData>>,
}

pub struct AacFrame {
    pub channels: Vec<AacChannel>,
    pub ms_mask_present: u8,
    pub ms_used: Vec<bool>,
}

impl IcsInfo {
    fn read(bits: &mut Bits<'_>, config: &AacConfig) -> Result<Self, MediaDecodeError> {
        if bits.read(1)? != 0 {
            return Err(invalid("reserved AAC ICS bit"));
        }
        let sequence = bits.read(2)? as u8;
        let shape = bits.read(1)? as u8;
        let short = sequence == 2;
        let max_sfb = bits.read(if short { 4 } else { 6 })? as usize;
        let offsets = bands::frame_bands(config.frequency_index, short, config.frame_samples)?;
        if max_sfb >= offsets.len() {
            return Err(invalid("AAC band count exceeds transform"));
        }
        let mut groups = vec![1];
        let mut prediction_used = Vec::new();
        let mut predictor_reset = None;
        if short {
            let grouping = bits.read(7)?;
            for bit in (0..7).rev() {
                if grouping & (1 << bit) != 0 {
                    *groups.last_mut().unwrap() += 1;
                } else {
                    groups.push(1);
                }
            }
        } else if bits.read(1)? != 0 {
            if config.object_type != 1 {
                return Err(MediaDecodeError::Unsupported);
            }
            if bits.read(1)? != 0 {
                let group = bits.read(5)? as u8;
                if !(1..=30).contains(&group) {
                    return Err(invalid("invalid AAC predictor reset group"));
                }
                predictor_reset = Some(group);
            }
            let limit = *prediction::MAX_BANDS
                .get(usize::from(config.frequency_index))
                .ok_or(MediaDecodeError::Unsupported)?;
            for _ in 0..max_sfb.min(limit) {
                prediction_used.push(bits.read(1)? != 0);
            }
        }
        Ok(Self {
            sequence,
            shape,
            max_sfb,
            groups,
            offsets,
            prediction_used,
            predictor_reset,
        })
    }

    pub fn short(&self) -> bool {
        self.sequence == 2
    }
}

impl AacFrame {
    pub fn parse(config: &AacConfig, packet: &[u8]) -> Result<Self, MediaDecodeError> {
        if !matches!(config.frame_samples, 1024 | 960) {
            return Err(MediaDecodeError::Unsupported);
        }
        let mut bits = Bits::new(packet);
        let mut channels = Vec::new();
        let mut ms_mask_present = 0;
        let mut ms_used = Vec::new();
        loop {
            match bits.read(3)? {
                0 => {
                    bits.read(4)?;
                    if channels.len() >= usize::from(config.channels) {
                        return Err(invalid("too many AAC channels"));
                    }
                    channels.push(channel(&mut bits, config, None)?);
                }
                1 => {
                    bits.read(4)?;
                    if config.channels != 2 || !channels.is_empty() {
                        return Err(invalid("unexpected AAC channel pair"));
                    }
                    let common = if bits.read(1)? != 0 {
                        let info = IcsInfo::read(&mut bits, config)?;
                        ms_mask_present = bits.read(2)? as u8;
                        if ms_mask_present == 3 {
                            return Err(invalid("reserved AAC MS mask"));
                        }
                        for _ in 0..info.groups.len() * info.max_sfb {
                            ms_used.push(
                                ms_mask_present == 2
                                    || (ms_mask_present == 1 && bits.read(1)? != 0),
                            );
                        }
                        Some(info)
                    } else {
                        None
                    };
                    channels.push(channel(&mut bits, config, common.clone())?);
                    channels.push(channel(&mut bits, config, common)?);
                }
                4 => {
                    bits.read(4)?;
                    let align = bits.read(1)? != 0;
                    let mut count = bits.read(8)? as usize;
                    if count == 255 {
                        count += bits.read(8)? as usize;
                    }
                    if align {
                        bits.align()?;
                    }
                    bits.skip(count * 8)?;
                }
                6 => {
                    let mut count = bits.read(4)? as usize;
                    if count == 15 {
                        count = 14 + bits.read(8)? as usize;
                    }
                    if count != 0 {
                        let extension = bits.read(4)?;
                        if extension == 13 || extension == 14 {
                            return Err(MediaDecodeError::Unsupported);
                        }
                        bits.skip(count * 8 - 4)?;
                    }
                }
                7 => {
                    bits.align()?;
                    break;
                }
                _ => return Err(MediaDecodeError::Unsupported),
            }
        }
        if channels.len() != usize::from(config.channels) {
            return Err(invalid("missing AAC channel"));
        }
        Ok(Self {
            channels,
            ms_mask_present,
            ms_used,
        })
    }
}

fn sections(bits: &mut Bits<'_>, info: &IcsInfo) -> Result<Vec<u8>, MediaDecodeError> {
    let width = if info.short() { 3 } else { 5 };
    let escape = (1 << width) - 1;
    let mut books = vec![0; info.groups.len() * info.max_sfb];
    for group in 0..info.groups.len() {
        let mut band = 0;
        while band < info.max_sfb {
            let book = bits.read(4)? as u8;
            if book == 12 {
                return Err(invalid("reserved AAC codebook"));
            }
            let mut length = 0;
            loop {
                let increment = bits.read(width)? as usize;
                length += increment;
                if length > info.max_sfb - band {
                    return Err(invalid("AAC section exceeds bands"));
                }
                if increment != escape {
                    break;
                }
            }
            if length == 0 {
                return Err(invalid("empty AAC section"));
            }
            books[group * info.max_sfb + band..group * info.max_sfb + band + length].fill(book);
            band += length;
        }
    }
    Ok(books)
}

fn channel(
    bits: &mut Bits<'_>,
    config: &AacConfig,
    common: Option<IcsInfo>,
) -> Result<AacChannel, MediaDecodeError> {
    let gain = bits.read(8)? as i32;
    let info = match common {
        Some(info) => info,
        None => IcsInfo::read(bits, config)?,
    };
    let codebooks = sections(bits, &info)?;
    let mut scale = gain;
    let mut intensity = 0;
    let mut noise = gain - 90;
    let mut first_noise = true;
    let mut scalefactors = Vec::with_capacity(codebooks.len());
    for &book in &codebooks {
        let value = match book {
            0 => 0,
            13 => {
                noise += if first_noise {
                    first_noise = false;
                    bits.read(9)? as i32 - 256
                } else {
                    entropy::scalefactor(bits)?
                };
                noise
            }
            14 | 15 => {
                intensity += entropy::scalefactor(bits)?;
                intensity
            }
            _ => {
                scale += entropy::scalefactor(bits)?;
                if !(0..=255).contains(&scale) {
                    return Err(invalid("invalid AAC spectral scalefactor"));
                }
                scale
            }
        };
        scalefactors.push(i16::try_from(value).map_err(|_| invalid("AAC scalefactor overflow"))?);
    }
    let mut pulses = Vec::new();
    if bits.read(1)? != 0 {
        if info.short() {
            return Err(invalid("AAC pulse on short window"));
        }
        let count = bits.read(2)? as usize + 1;
        let start = bits.read(6)? as usize;
        let mut position = usize::from(
            *info
                .offsets
                .get(start)
                .ok_or_else(|| invalid("invalid AAC pulse band"))?,
        );
        for _ in 0..count {
            position += bits.read(5)? as usize;
            let amplitude = bits.read(4)? as i32;
            if position >= config.frame_samples {
                return Err(invalid("AAC pulse exceeds spectrum"));
            }
            pulses.push((position, amplitude));
        }
    }
    let mut tns = vec![Vec::new(); if info.short() { 8 } else { 1 }];
    if bits.read(1)? != 0 {
        for filters in &mut tns {
            let count = bits.read(if info.short() { 1 } else { 2 })?;
            let resolution = if count != 0 {
                bits.read(1)? as u8 + 3
            } else {
                3
            };
            for _ in 0..count {
                let length = bits.read(if info.short() { 4 } else { 6 })? as usize;
                let order = bits.read(if info.short() { 3 } else { 5 })? as usize;
                if order
                    > if info.short() {
                        7
                    } else if config.object_type == 1 {
                        20
                    } else {
                        12
                    }
                {
                    return Err(invalid("AAC TNS order exceeds profile limit"));
                }
                let reverse = order != 0 && bits.read(1)? != 0;
                let compressed = order != 0 && bits.read(1)? != 0;
                let mut coefficients = Vec::with_capacity(order);
                for _ in 0..order {
                    coefficients
                        .push(bits.read(usize::from(resolution) - usize::from(compressed))? as u8);
                }
                filters.push(TnsFilterData {
                    length,
                    reverse,
                    resolution,
                    compressed,
                    coefficients,
                });
            }
        }
    }
    if bits.read(1)? != 0 {
        return Err(MediaDecodeError::Unsupported);
    }
    let mut quantized = vec![0; config.frame_samples];
    let window_size = config.frame_samples / if info.short() { 8 } else { 1 };
    let mut first_window = 0;
    for (group, &windows) in info.groups.iter().enumerate() {
        for band in 0..info.max_sfb {
            let book = codebooks[group * info.max_sfb + band];
            if !(1..=11).contains(&book) {
                continue;
            }
            for window in first_window..first_window + windows {
                let mut position = window * window_size + usize::from(info.offsets[band]);
                let end = window * window_size + usize::from(info.offsets[band + 1]);
                while position < end {
                    let (tuple, width) = entropy::spectral(bits, book)?;
                    if width > end - position {
                        return Err(invalid("AAC tuple exceeds band"));
                    }
                    quantized[position..position + width].copy_from_slice(&tuple[..width]);
                    position += width;
                }
            }
        }
        first_window += windows;
    }
    for (position, amplitude) in pulses {
        let value = &mut quantized[position];
        *value += if *value > 0 { amplitude } else { -amplitude };
        if value.abs() > 8191 {
            return Err(invalid("AAC pulse coefficient exceeds limit"));
        }
    }
    Ok(AacChannel {
        info,
        codebooks,
        scalefactors,
        quantized,
        tns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed(fields: &[(u32, usize)]) -> Vec<u8> {
        let mut bytes = vec![0; fields.iter().map(|f| f.1).sum::<usize>().div_ceil(8)];
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
    fn parses_transmitted_zero_band_channel_not_substitute_audio() {
        let config = AacConfig::parse(&[0x11, 0x88]).unwrap();
        let packet = packed(&[
            (0, 3),
            (0, 4),
            (100, 8),
            (0, 1),
            (0, 2),
            (0, 1),
            (0, 6),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (7, 3),
        ]);
        let frame = AacFrame::parse(&config, &packet).unwrap();
        assert_eq!(frame.channels.len(), 1);
        assert!(frame.channels[0].quantized.iter().all(|&value| value == 0));
        for end in 0..packet.len() {
            assert!(AacFrame::parse(&config, &packet[..end]).is_err());
        }
    }

    #[test]
    fn grouping_bits_form_eight_short_windows() {
        let config = AacConfig::parse(&[0x12, 0x10]).unwrap();
        let packet = packed(&[(0, 1), (2, 2), (1, 1), (14, 4), (0b1101010, 7)]);
        let info = IcsInfo::read(&mut Bits::new(&packet), &config).unwrap();
        assert_eq!(info.groups, [3, 2, 2, 1]);
        assert_eq!(info.max_sfb, 14);
    }

    #[test]
    fn shorter_frames_decode_coded_spectrum_for_all_window_sequences() {
        use super::super::{AacDecoder, AacSynthesis, tables::CODEBOOKS};

        let config = AacConfig::parse(&packed(&[(2, 5), (3, 4), (1, 4), (1, 1), (0, 2)])).unwrap();
        assert_eq!(config.frame_samples, 960);
        let (scale_length, scale_word) = CODEBOOKS[0][60];
        let (tuple_length, tuple_word) = CODEBOOKS[1][67]; // Signed tuple (1, 0, 0, 0).
        for sequence in 0..=3 {
            let short = sequence == 2;
            let mut fields = vec![(0, 3), (0, 4), (100, 8), (0, 1), (sequence, 2), (1, 1)];
            fields.push((1, if short { 4 } else { 6 }));
            fields.push(if short { (127, 7) } else { (0, 1) });
            fields.extend([
                (1, 4),
                (1, if short { 3 } else { 5 }),
                (scale_word, scale_length as usize),
            ]);
            fields.extend([(0, 1), (0, 1), (0, 1)]);
            for _ in 0..if short { 8 } else { 1 } {
                fields.push((tuple_word, tuple_length as usize));
            }
            fields.push((7, 3));
            let packet = packed(&fields);
            let frame = AacFrame::parse(&config, &packet).unwrap();
            let channel = &frame.channels[0];
            assert_eq!(channel.quantized.len(), 960);
            let mut spectrum = vec![0.0; 960];
            for window in 0..if short { 8 } else { 1 } {
                let bin = window * if short { 120 } else { 960 };
                spectrum[bin] = 1.0;
                assert_eq!(channel.quantized[bin], 1);
            }
            let mut reference = AacSynthesis::with_frame_samples(960).unwrap();
            let mut expected = vec![0.0; 960];
            let mut decoder = AacDecoder::new(config.clone()).unwrap();
            for _ in 0..3 {
                reference
                    .synthesize(&spectrum, sequence as u8, 1, &mut expected)
                    .unwrap();
                let actual = decoder.decode(&packet).unwrap();
                assert_eq!(actual.samples.len(), 960);
                assert_eq!(actual.sample_rate, 48000);
                assert!(actual.samples.iter().any(|value| *value != 0.0));
                for (actual, expected) in actual.samples.iter().zip(&expected) {
                    assert_eq!(*actual, (expected / 32768.0) as f32);
                }
            }
            for end in 0..packet.len() {
                assert!(AacFrame::parse(&config, &packet[..end]).is_err());
            }
        }
    }

    #[test]
    fn shorter_frame_band_limits_and_pulses_are_bounded() {
        let config = AacConfig::parse(&[0x11, 0x8c]).unwrap();
        assert_eq!(config.frame_samples, 960);
        // The last pulse band starts at 928; offset 31 reaches the last valid bin.
        let packet = |offset: u32| {
            packed(&[
                (0, 3),
                (0, 4),
                (100, 8),
                (0, 1),
                (0, 2),
                (0, 1),
                (0, 6),
                (0, 1),
                (1, 1),
                (0, 2),
                (48, 6),
                (offset, 5),
                (1, 4),
                (0, 1),
                (0, 1),
                (7, 3),
            ])
        };
        let frame = AacFrame::parse(&config, &packet(31)).unwrap();
        assert_eq!(frame.channels[0].quantized[959], -1);
        let invalid_pulse = packed(&[
            (0, 3),
            (0, 4),
            (100, 8),
            (0, 1),
            (0, 2),
            (0, 1),
            (0, 6),
            (0, 1),
            (1, 1),
            (0, 2),
            (49, 6),
            (0, 5),
            (1, 4),
            (0, 1),
            (0, 1),
            (7, 3),
        ]);
        assert!(AacFrame::parse(&config, &invalid_pulse).is_err());
        for index in 0..12 {
            let mut config = config.clone();
            config.frequency_index = index;
            for short in [false, true] {
                let offsets = bands::frame_bands(index, short, 960).unwrap();
                let max = (offsets.len() - 1) as u32;
                let fields = |max: u32| {
                    packed(&[
                        (0, 1),
                        (if short { 2 } else { 0 }, 2),
                        (0, 1),
                        (max, if short { 4 } else { 6 }),
                        (0, if short { 7 } else { 1 }),
                    ])
                };
                let bytes = fields(max);
                assert!(IcsInfo::read(&mut Bits::new(&bytes), &config).is_ok());
                if !short || max < 15 {
                    let bytes = fields(max + 1);
                    assert!(IcsInfo::read(&mut Bits::new(&bytes), &config).is_err());
                }
            }
        }
    }

    #[test]
    fn both_frame_lengths_decode_common_window_mid_side_stereo() {
        use super::super::{AacDecoder, AacSynthesis, tables::CODEBOOKS};

        let (scale_length, scale_word) = CODEBOOKS[0][60];
        let (tuple_length, tuple_word) = CODEBOOKS[1][67];
        for samples in [1024, 960] {
            let config = AacConfig::parse(&packed(&[
                (2, 5),
                (3, 4),
                (2, 4),
                (u32::from(samples == 960), 1),
                (0, 2),
            ]))
            .unwrap();
            for sequence in 0..=3 {
                let short = sequence == 2;
                let mut fields = vec![(1, 3), (0, 4), (1, 1), (0, 1), (sequence, 2), (0, 1)];
                fields.push((1, if short { 4 } else { 6 }));
                fields.push(if short { (127, 7) } else { (0, 1) });
                fields.push((2, 2)); // MS applies to every transmitted band.
                for _ in 0..2 {
                    fields.extend([
                        (100, 8),
                        (1, 4),
                        (1, if short { 3 } else { 5 }),
                        (scale_word, scale_length as usize),
                        (0, 1),
                        (0, 1),
                        (0, 1),
                    ]);
                    for _ in 0..if short { 8 } else { 1 } {
                        fields.push((tuple_word, tuple_length as usize));
                    }
                }
                fields.push((7, 3));
                let packet = packed(&fields);
                let mut decoder = AacDecoder::new(config.clone()).unwrap();
                let mut reference = AacSynthesis::with_frame_samples(samples).unwrap();
                let mut spectrum = vec![0.0; samples];
                for window in 0..if short { 8 } else { 1 } {
                    spectrum[window * if short { samples / 8 } else { samples }] = 2.0;
                }
                let mut expected = vec![0.0; samples];
                for _ in 0..3 {
                    reference
                        .synthesize(&spectrum, sequence as u8, 0, &mut expected)
                        .unwrap();
                    let actual = decoder.decode(&packet).unwrap();
                    assert_eq!(actual.channels, 2);
                    assert_eq!(actual.samples.len(), samples * 2);
                    for (pair, expected) in actual.samples.chunks_exact(2).zip(&expected) {
                        assert_eq!(pair[0], (expected / 32768.0) as f32);
                        assert_eq!(pair[1], 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn main_prediction_stream_resets_and_matches_optional_binary_oracle() {
        use super::super::{AacDecoder, tables::CODEBOOKS};

        let config = AacConfig::parse(&[0x09, 0x88]).unwrap();
        assert_eq!(config.object_type, 1);
        let mut decoder = AacDecoder::new(config.clone()).unwrap();
        let mut adts = Vec::new();
        let mut pcm = Vec::new();
        let (scale_length, scale_word) = CODEBOOKS[0][60];
        let gain = std::env::var("WEBMEDIA_AAC_MAIN_GAIN")
            .ok()
            .map(|gain| gain.parse::<u32>().unwrap())
            .unwrap_or(152);
        assert!(gain <= 255);
        for number in 0..120 {
            let sequence = match number {
                80 => 1,
                81 => 2,
                82 => 3,
                _ => 0,
            };
            let short = sequence == 2;
            let mut fields = vec![(0, 3), (0, 4), (gain, 8), (0, 1), (sequence, 2), (0, 1)];
            fields.push((1, if short { 4 } else { 6 }));
            if short {
                fields.push((127, 7));
            } else {
                fields.push((1, 1));
                if number % 8 == 0 {
                    fields.extend([(1, 1), (number / 8 % 4 + 1, 5)]);
                } else {
                    fields.push((0, 1));
                }
                fields.push((1, 1));
            }
            fields.extend([
                (1, 4),
                (1, if short { 3 } else { 5 }),
                (scale_word, scale_length as usize),
                (0, 1),
                (0, 1),
                (0, 1),
            ]);
            for window in 0..if short { 8 } else { 1 } {
                let index = if number % 5 == 0 {
                    40
                } else if (number + window) % 2 == 0 {
                    67
                } else {
                    13
                };
                let (length, word) = CODEBOOKS[1][index];
                fields.push((word, length as usize));
            }
            fields.push((7, 3));
            let packet = packed(&fields);
            let frame = AacFrame::parse(&config, &packet).unwrap();
            assert_eq!(
                frame.channels[0].info.prediction_used,
                if short { vec![] } else { vec![true] }
            );
            let decoded = decoder.decode(&packet).unwrap();
            assert_eq!(decoded.samples.len(), 1024);
            pcm.extend(decoded.samples);
            let length = packet.len() + 7;
            adts.extend([
                0xff,
                0xf1,
                3 << 2,
                (1 << 6) | (length >> 11) as u8,
                (length >> 3) as u8,
                ((length & 7) << 5) as u8 | 0x1f,
                0xfc,
            ]);
            adts.extend(packet);
        }
        assert!(pcm.iter().all(|value| value.is_finite()));
        assert!(pcm.iter().any(|value| *value != 0.0));
        main_binary_oracle(&adts, &pcm, "WEBMEDIA_AAC_MAIN");
        let packet = packed(&[(0, 1), (0, 2), (0, 1), (1, 6), (1, 1), (1, 1), (0, 5)]);
        assert!(IcsInfo::read(&mut Bits::new(&packet), &config).is_err());
        let packet = packed(&[(0, 1), (0, 2), (0, 1), (1, 6), (1, 1), (1, 1), (31, 5)]);
        assert!(IcsInfo::read(&mut Bits::new(&packet), &config).is_err());
    }

    fn main_binary_oracle(adts: &[u8], pcm: &[f32], prefix: &str) {
        if let Ok(path) = std::env::var(format!("{prefix}_ADTS_OUT")) {
            std::fs::write(path, adts).unwrap();
        }
        if let Ok(path) = std::env::var(format!("{prefix}_PCM_OUT")) {
            let bytes: Vec<_> = pcm.iter().flat_map(|value| value.to_le_bytes()).collect();
            std::fs::write(path, bytes).unwrap();
        }
        if let Ok(path) = std::env::var(format!("{prefix}_REFERENCE_PCM")) {
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(bytes.len(), pcm.len() * 4);
            let reference: Vec<_> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let maximum = pcm
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let peak = reference
                .iter()
                .map(|value| value.abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "AAC Main binary oracle: {} PCM samples, max error={maximum}",
                pcm.len()
            );
            assert!(maximum < (peak * 5e-6).max(1e-12));
        }
    }

    #[test]
    fn main_stereo_prediction_and_intensity_match_optional_binary_oracle() {
        use super::super::{AacDecoder, tables::CODEBOOKS};

        let config = AacConfig::parse(&[0x09, 0x90]).unwrap();
        let mut decoder = AacDecoder::new(config).unwrap();
        let mut adts = Vec::new();
        let mut pcm = Vec::new();
        for number in 0..100 {
            let intensity = (30..60).contains(&number);
            let mut fields = vec![
                (1, 3),
                (0, 4),
                (1, 1),
                (0, 1),
                (0, 2),
                (0, 1),
                (1, 6),
                (1, 1),
                (0, 1),
                (1, 1),
                (1, 2),
                (1, 1),
            ];
            for channel in 0..2 {
                let book = if channel == 1 && intensity { 15 } else { 1 };
                let (sf_length, sf_word) = CODEBOOKS[0][if book == 15 { 61 } else { 60 }];
                fields.extend([
                    (if channel == 0 { 152 } else { 148 }, 8),
                    (book, 4),
                    (1, 5),
                    (sf_word, sf_length as usize),
                    (0, 1),
                    (0, 1),
                    (0, 1),
                ]);
                if book == 1 {
                    let (length, word) =
                        CODEBOOKS[1][if (number + channel) % 2 == 0 { 67 } else { 13 }];
                    fields.push((word, length as usize));
                }
            }
            fields.push((7, 3));
            let packet = packed(&fields);
            pcm.extend(decoder.decode(&packet).unwrap().samples);
            let length = packet.len() + 7;
            adts.extend([
                0xff,
                0xf1,
                3 << 2,
                (2 << 6) | (length >> 11) as u8,
                (length >> 3) as u8,
                ((length & 7) << 5) as u8 | 0x1f,
                0xfc,
            ]);
            adts.extend(packet);
        }
        assert_eq!(pcm.len(), 204800);
        assert!(pcm.iter().all(|value| value.is_finite()));
        main_binary_oracle(&adts, &pcm, "WEBMEDIA_AAC_MAIN_STEREO");
    }

    #[test]
    fn main_accepts_twentieth_order_tns_but_lc_rejects_it() {
        use super::super::AacDecoder;

        let main = AacConfig::parse(&[0x09, 0x88]).unwrap();
        let lc = AacConfig::parse(&[0x11, 0x88]).unwrap();
        let mut fields = vec![
            (0, 3),
            (0, 4),
            (100, 8),
            (0, 1),
            (0, 2),
            (0, 1),
            (49, 6),
            (0, 1),
            (0, 4),
            (31, 5),
            (18, 5),
            (0, 1),
            (1, 1),
            (1, 2),
            (0, 1),
            (49, 6),
            (20, 5),
            (0, 1),
            (0, 1),
        ];
        fields.extend([(0, 3); 20]);
        fields.extend([(0, 1), (7, 3)]);
        let packet = packed(&fields);
        let parsed = AacFrame::parse(&main, &packet).unwrap();
        assert_eq!(parsed.channels[0].tns[0][0].coefficients.len(), 20);
        assert!(AacFrame::parse(&lc, &packet).is_err());
        let pcm = AacDecoder::new(main).unwrap().decode(&packet).unwrap();
        assert!(pcm.samples.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn main_twentieth_order_tns_matches_optional_binary_oracle() {
        use super::super::{AacDecoder, tables::CODEBOOKS};

        let mut decoder = AacDecoder::new(AacConfig::parse(&[0x09, 0x88]).unwrap()).unwrap();
        let mut pcm = Vec::new();
        let mut adts = Vec::new();
        let (scale_length, scale_word) = CODEBOOKS[0][60];
        for number in 0..32 {
            let mut fields = vec![
                (0, 3),
                (0, 4),
                (152, 8),
                (0, 1),
                (0, 2),
                (0, 1),
                (40, 6),
                (1, 1),
                (0, 1),
                (1, 1),
            ];
            fields.extend([(0, 1); 39]);
            fields.extend([
                (1, 4),
                (1, 5),
                (0, 4),
                (31, 5),
                (8, 5),
                (scale_word, scale_length as usize),
                (0, 1),
                (1, 1),
                (1, 2),
                (0, 1),
                (49, 6),
                (20, 5),
                (0, 1),
                (0, 1),
            ]);
            fields.extend([(0, 3); 19]);
            fields.extend([(1, 3), (0, 1)]);
            let (length, word) = CODEBOOKS[1][if number % 2 == 0 { 67 } else { 13 }];
            fields.extend([(word, length as usize), (7, 3)]);
            let packet = packed(&fields);
            pcm.extend(decoder.decode(&packet).unwrap().samples);
            let length = packet.len() + 7;
            adts.extend([
                0xff,
                0xf1,
                3 << 2,
                (1 << 6) | (length >> 11) as u8,
                (length >> 3) as u8,
                ((length & 7) << 5) as u8 | 0x1f,
                0xfc,
            ]);
            adts.extend(packet);
        }
        assert!(pcm.iter().any(|value| *value != 0.0));
        main_binary_oracle(&adts, &pcm, "WEBMEDIA_AAC_MAIN_TNS");
    }

    #[test]
    #[ignore = "set WEBMEDIA_AAC_MP4 to an AAC-LC MP4 fixture"]
    fn parses_all_fixture_access_units() {
        let bytes = std::fs::read(std::env::var("WEBMEDIA_AAC_MP4").unwrap()).unwrap();
        let index = crate::video::mp4::Mp4AudioIndex::parse_prefix(&bytes)
            .unwrap()
            .unwrap();
        let config = AacConfig::parse(&index.audio_specific_config).unwrap();
        let mut books = [0usize; 16];
        let mut sequences = [0usize; 4];
        let mut tns_filters = 0;
        for (number, sample) in index.samples.iter().enumerate() {
            let start = sample.offset as usize;
            let frame = AacFrame::parse(&config, &bytes[start..start + sample.size as usize])
                .unwrap_or_else(|error| panic!("packet {number}: {error:?}"));
            for channel in frame.channels {
                sequences[usize::from(channel.info.sequence)] += 1;
                for book in channel.codebooks {
                    books[usize::from(book)] += 1;
                }
                tns_filters += channel.tns.iter().map(Vec::len).sum::<usize>();
            }
        }
        eprintln!(
            "AAC fixture: codebooks={books:?} sequences={sequences:?} TNS filters={tns_filters}"
        );
    }
}
