//! AAC Main/LC spectral reconstruction and persistent per-channel synthesis.

use super::{
    AacConfig, AacFrame, AacSynthesis, bands, invalid, noise::Noise, prediction::Predictor,
    spectral,
};
use crate::video::backend::{AudioSamples, MediaDecodeError};
use std::ops::Range;

pub struct AacDecoder {
    config: AacConfig,
    synthesis: Vec<AacSynthesis>,
    spectra: Vec<Vec<f64>>,
    samples: Vec<Vec<f64>>,
    noise: Noise,
    prediction: Vec<Predictor>,
    #[cfg(test)]
    reference_quantizer: bool,
    #[cfg(test)]
    reference_tns: bool,
    #[cfg(test)]
    tns_time: std::time::Duration,
    #[cfg(test)]
    tns_windows: [usize; 2],
    #[cfg(test)]
    stage_times: [std::time::Duration; 5],
}

impl AacDecoder {
    pub fn new(config: AacConfig) -> Result<Self, MediaDecodeError> {
        if !matches!(config.frame_samples, 1024 | 960)
            || !matches!(config.object_type, 1 | 2)
            || config.sample_rate == 0
            || !(1..=2).contains(&config.channels)
        {
            return Err(MediaDecodeError::Unsupported);
        }
        bands::frame_bands(config.frequency_index, false, config.frame_samples)?;
        let channels = usize::from(config.channels);
        let samples = config.frame_samples;
        let synthesis = (0..channels)
            .map(|_| AacSynthesis::with_frame_samples(samples))
            .collect::<Result<_, _>>()?;
        let prediction = if config.object_type == 1 {
            (0..channels).map(|_| Predictor::new(samples)).collect()
        } else {
            Vec::new()
        };
        Ok(Self {
            config,
            synthesis,
            spectra: vec![vec![0.0; samples]; channels],
            samples: vec![vec![0.0; samples]; channels],
            noise: Noise::new(),
            prediction,
            #[cfg(test)]
            reference_quantizer: false,
            #[cfg(test)]
            reference_tns: true,
            #[cfg(test)]
            tns_time: std::time::Duration::ZERO,
            #[cfg(test)]
            tns_windows: [0; 2],
            #[cfg(test)]
            stage_times: [std::time::Duration::ZERO; 5],
        })
    }

    pub fn reset(&mut self) {
        for synthesis in &mut self.synthesis {
            synthesis.reset();
        }
        self.noise = Noise::new();
        for predictor in &mut self.prediction {
            predictor.reset();
        }
    }

    pub fn decode(&mut self, packet: &[u8]) -> Result<AudioSamples, MediaDecodeError> {
        #[cfg(test)]
        let mut stage_start = std::time::Instant::now();
        let frame = AacFrame::parse(&self.config, packet)?;
        #[cfg(test)]
        {
            self.stage_times[0] += stage_start.elapsed();
            stage_start = std::time::Instant::now();
        }
        let mut channel_bands = Vec::new();
        for (channel, spectrum) in frame.channels.iter().zip(&mut self.spectra) {
            let width = self.config.frame_samples / if channel.info.short() { 8 } else { 1 };
            let mut ranges = Vec::new();
            let mut factors = Vec::new();
            let mut all = Vec::new();
            let mut first_window = 0;
            for (group, &windows) in channel.info.groups.iter().enumerate() {
                for band in 0..channel.info.max_sfb {
                    let index = group * channel.info.max_sfb + band;
                    for window in first_window..first_window + windows {
                        let range = window * width + usize::from(channel.info.offsets[band])
                            ..window * width + usize::from(channel.info.offsets[band + 1]);
                        if (1..=11).contains(&channel.codebooks[index]) {
                            ranges.push(range.clone());
                            factors.push(channel.scalefactors[index]);
                        }
                        all.push((range, index));
                    }
                }
                first_window += windows;
            }
            #[cfg(test)]
            if self.reference_quantizer {
                spectral::inverse_quantize_reference(
                    &channel.quantized,
                    spectrum,
                    &ranges,
                    &factors,
                )?;
            } else {
                spectral::inverse_quantize(&channel.quantized, spectrum, &ranges, &factors)?;
            }
            #[cfg(not(test))]
            spectral::inverse_quantize(&channel.quantized, spectrum, &ranges, &factors)?;
            channel_bands.push(all);
        }
        #[cfg(test)]
        {
            self.stage_times[1] += stage_start.elapsed();
            stage_start = std::time::Instant::now();
        }
        for channel_index in 0..frame.channels.len() {
            let channel = &frame.channels[channel_index];
            for (range, band) in &channel_bands[channel_index] {
                if channel.codebooks[*band] != 13 {
                    continue;
                }
                let correlated = channel_index == 1
                    && frame.ms_used.get(*band) == Some(&true)
                    && frame.channels[0].codebooks.get(*band) == Some(&13)
                    && frame.channels[0].info.sequence == channel.info.sequence
                    && frame.channels[0].info.groups == channel.info.groups;
                if correlated {
                    let (left, right) = self.spectra.split_at_mut(1);
                    Noise::correlated(
                        &left[0][range.clone()],
                        &mut right[0][range.clone()],
                        frame.channels[0].scalefactors[*band],
                        channel.scalefactors[*band],
                    )?;
                } else {
                    self.noise.fill(
                        &mut self.spectra[channel_index][range.clone()],
                        channel.scalefactors[*band],
                    )?;
                }
            }
        }
        let mut intensity_ranges = Vec::new();
        let mut intensity_modes = Vec::new();
        if frame.channels.len() == 2 && !frame.ms_used.is_empty() {
            let left = &frame.channels[0];
            let right = &frame.channels[1];
            let mut ranges: Vec<Range<usize>> = Vec::new();
            let mut modes = Vec::new();
            for (range, band) in &channel_bands[1] {
                let book = right.codebooks[*band];
                let mode = if book == 14 || book == 15 {
                    spectral::StereoMode::Intensity {
                        position: right.scalefactors[*band],
                        out_of_phase: book == 14,
                        invert: frame.ms_mask_present == 1 && frame.ms_used[*band],
                    }
                } else if right.codebooks[*band] != 13
                    && left.codebooks.get(*band) != Some(&13)
                    && frame.ms_used[*band]
                {
                    spectral::StereoMode::MidSide
                } else {
                    spectral::StereoMode::Independent
                };
                ranges.push(range.clone());
                if self.config.object_type == 1
                    && matches!(mode, spectral::StereoMode::Intensity { .. })
                {
                    intensity_ranges.push(range.clone());
                    intensity_modes.push(mode);
                    modes.push(spectral::StereoMode::Independent);
                } else {
                    modes.push(mode);
                }
            }
            let (left, right) = self.spectra.split_at_mut(1);
            spectral::apply_stereo(&mut left[0], &mut right[0], &ranges, &modes)?;
        }
        for (index, predictor) in self.prediction.iter_mut().enumerate() {
            // Main intensity uses the reconstructed left spectrum, and the
            // right predictor adapts to that scaled value with prediction off.
            if index == 1 && !intensity_ranges.is_empty() {
                let (left, right) = self.spectra.split_at_mut(1);
                spectral::apply_stereo(
                    &mut left[0],
                    &mut right[0],
                    &intensity_ranges,
                    &intensity_modes,
                )?;
            }
            let channel = &frame.channels[index];
            predictor.apply(
                &mut self.spectra[index],
                &channel.info,
                &channel.codebooks,
                self.config.frequency_index,
            )?;
        }
        #[cfg(test)]
        {
            self.stage_times[2] += stage_start.elapsed();
            stage_start = std::time::Instant::now();
        }
        const LONG_TNS_BANDS: [usize; 12] = [31, 31, 34, 40, 42, 51, 46, 46, 42, 42, 42, 39];
        const SHORT_TNS_BANDS: [usize; 12] = [9, 9, 10, 14, 14, 14, 14, 14, 14, 14, 14, 14];
        for (index, channel) in frame.channels.iter().enumerate() {
            let short = channel.info.short();
            let width = self.config.frame_samples / if short { 8 } else { 1 };
            let offsets: Vec<_> = channel
                .info
                .offsets
                .iter()
                .map(|&value| usize::from(value))
                .collect();
            let max_bands = if short {
                SHORT_TNS_BANDS
            } else {
                LONG_TNS_BANDS
            }[usize::from(self.config.frequency_index)];
            for (window, filters) in channel.tns.iter().enumerate() {
                let filters: Vec<_> = filters
                    .iter()
                    .map(|filter| spectral::TnsFilter {
                        length_bands: filter.length,
                        reverse: filter.reverse,
                        resolution: filter.resolution,
                        compressed: filter.compressed,
                        coefficients: &filter.coefficients,
                    })
                    .collect();
                #[cfg(test)]
                {
                    self.tns_windows[0] +=
                        usize::from(filters.iter().all(|filter| filter.coefficients.is_empty()));
                    self.tns_windows[1] += 1;
                    let time = std::time::Instant::now();
                    let apply = if self.reference_tns {
                        spectral::apply_tns
                    } else {
                        spectral::apply_tns_candidate
                    };
                    apply(
                        &mut self.spectra[index][window * width..(window + 1) * width],
                        &offsets,
                        channel.info.max_sfb,
                        max_bands,
                        if short {
                            7
                        } else if self.config.object_type == 1 {
                            20
                        } else {
                            12
                        },
                        &filters,
                    )?;
                    self.tns_time += time.elapsed();
                }
                #[cfg(not(test))]
                spectral::apply_tns(
                    &mut self.spectra[index][window * width..(window + 1) * width],
                    &offsets,
                    channel.info.max_sfb,
                    max_bands,
                    if short {
                        7
                    } else if self.config.object_type == 1 {
                        20
                    } else {
                        12
                    },
                    &filters,
                )?;
            }
            self.synthesis[index].synthesize(
                &self.spectra[index],
                channel.info.sequence,
                channel.info.shape,
                &mut self.samples[index],
            )?;
        }
        #[cfg(test)]
        {
            self.stage_times[3] += stage_start.elapsed();
            stage_start = std::time::Instant::now();
        }
        let channels = usize::from(self.config.channels);
        let mut samples = Vec::with_capacity(self.config.frame_samples * channels);
        for time in 0..self.config.frame_samples {
            for channel in &self.samples {
                let value = channel[time] / 32768.0;
                if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
                    return Err(invalid("invalid AAC PCM"));
                }
                samples.push(value as f32);
            }
        }
        #[cfg(test)]
        {
            self.stage_times[4] += stage_start.elapsed();
        }
        Ok(AudioSamples {
            sample_rate: self.config.sample_rate,
            channels: self.config.channels,
            samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pns_test_packet(short: bool, mask: u32, books: [u32; 2]) -> Vec<u8> {
        let mut fields = vec![
            (1, 3),
            (0, 4),
            (1, 1),
            (0, 1),
            (if short { 2 } else { 0 }, 2),
            (0, 1),
        ];
        if short {
            fields.extend([(2, 4), (127, 7)]);
        } else {
            fields.extend([(2, 6), (0, 1)]);
        }
        fields.push((mask, 2));
        if mask == 1 {
            fields.extend([(1, 1), (0, 1)]);
        }
        let (length, word) = super::super::tables::CODEBOOKS[0][60];
        for (channel, &book) in books.iter().enumerate() {
            fields.extend([
                (100 + 4 * channel as u32, 8),
                (book, 4),
                (2, if short { 3 } else { 5 }),
            ]);
            if book == 13 {
                fields.extend([(256, 9), (word, length as usize)]);
            }
            fields.extend([(0, 1), (0, 1), (0, 1)]);
        }
        fields.push((7, 3));
        let mut bits = Vec::new();
        for (word, width) in fields {
            for bit in (0..width).rev() {
                bits.push(((word >> bit) & 1) as u8);
            }
        }
        let mut bytes = vec![0; bits.len().div_ceil(8)];
        for (position, bit) in bits.into_iter().enumerate() {
            bytes[position / 8] |= bit << (7 - position % 8);
        }
        bytes
    }

    #[test]
    fn pns_energy_and_stereo_reuse_follow_band_flags() {
        let config = AacConfig::parse(&[0x12, 0x10]).unwrap();
        for short in [false, true] {
            for mask in [0, 1, 2] {
                let packet = pns_test_packet(short, mask, [13, 13]);
                let frame = AacFrame::parse(&config, &packet).unwrap();
                assert_eq!(frame.channels[0].scalefactors, [10, 10]);
                assert_eq!(frame.channels[1].scalefactors, [14, 14]);
                let mut decoder = AacDecoder::new(config.clone()).unwrap();
                decoder.decode(&packet).unwrap();
                let width = if short { 128 } else { 1024 };
                for window in 0..if short { 8 } else { 1 } {
                    for band in 0..2 {
                        let offsets = frame.channels[0].info.offsets;
                        let start = window * width + usize::from(offsets[band]);
                        let end = window * width + usize::from(offsets[band + 1]);
                        let left = &decoder.spectra[0][start..end];
                        let right = &decoder.spectra[1][start..end];
                        for (values, energy) in [(left, 10), (right, 14)] {
                            let power: f64 = values.iter().map(|x| x * x).sum();
                            let target = 2.0_f64.powf(f64::from(energy) * 0.5);
                            // Existing Noise unit-test roundoff bound, not a
                            // PCM decoder-conformance acceptance tolerance.
                            assert!((power / target - 1.0).abs() < 1e-14);
                        }
                        if mask == 2 || (mask == 1 && band == 0) {
                            assert!(
                                left.iter()
                                    .zip(right)
                                    .all(|(&l, &r)| r.to_bits() == (l * 2.0).to_bits())
                            );
                        } else {
                            assert!(
                                left.iter()
                                    .zip(right)
                                    .any(|(&l, &r)| r.to_bits() != (l * 2.0).to_bits())
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn ms_flag_is_ignored_for_one_sided_pns() {
        let config = AacConfig::parse(&[0x12, 0x10]).unwrap();
        for short in [false, true] {
            for books in [[13, 0], [0, 13]] {
                let mut independent = AacDecoder::new(config.clone()).unwrap();
                let mut ms = AacDecoder::new(config.clone()).unwrap();
                let a = independent
                    .decode(&pns_test_packet(short, 0, books))
                    .unwrap();
                let b = ms.decode(&pns_test_packet(short, 2, books)).unwrap();
                assert!(
                    a.samples
                        .iter()
                        .zip(&b.samples)
                        .all(|(x, y)| x.to_bits() == y.to_bits())
                );
                let silent = usize::from(books[0] == 13);
                assert!(ms.spectra[silent].iter().all(|&x| x == 0.0));
            }
        }
    }

    #[test]
    fn reduced_mdct_filterbank_transitions_and_reset_match_retained() {
        for samples in [1024, 960] {
            let mut retained = AacSynthesis::with_frame_samples(samples).unwrap();
            let mut reduced = AacSynthesis::with_frame_samples(samples).unwrap();
            let mut expected = vec![0.0; samples];
            let mut actual = vec![0.0; samples];
            for (frame, sequence) in [0, 1, 2, 2, 3, 0, 1, 3].into_iter().enumerate() {
                let shape = (frame % 2) as u8;
                let spectrum: Vec<_> = (0..samples)
                    .map(|k| ((k * 137 + frame * 71) % 101) as f64 - 50.0)
                    .collect();
                retained
                    .synthesize(&spectrum, sequence, shape, &mut expected)
                    .unwrap();
                crate::audio::transform::with_reduced_mdct(|| {
                    reduced.synthesize(&spectrum, sequence, shape, &mut actual)
                })
                .unwrap();
                for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (a - b).abs() < 2e-12,
                        "{samples}/{frame}/{sequence}/{shape}/{i}: {a} != {b}"
                    );
                }
            }
            retained.reset();
            reduced.reset();
            let spectrum = vec![0.25; samples];
            retained.synthesize(&spectrum, 0, 1, &mut expected).unwrap();
            crate::audio::transform::with_reduced_mdct(|| {
                reduced.synthesize(&spectrum, 0, 1, &mut actual)
            })
            .unwrap();
            let mut fresh = AacSynthesis::with_frame_samples(samples).unwrap();
            let mut fresh_output = vec![0.0; samples];
            crate::audio::transform::with_reduced_mdct(|| {
                fresh.synthesize(&spectrum, 0, 1, &mut fresh_output)
            })
            .unwrap();
            assert!(
                actual
                    .iter()
                    .zip(&fresh_output)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| (a - b).abs() < 2e-12)
            );
        }
    }

    #[test]
    #[ignore = "set WEBMEDIA_AAC_MP4 for whole-fixture reduced MDCT numerical/PCM gate"]
    fn reduced_mdct_fixture_pcm_matches_retained() {
        let bytes = std::fs::read(std::env::var("WEBMEDIA_AAC_MP4").unwrap()).unwrap();
        let index = crate::video::mp4::Mp4AudioIndex::parse_prefix(&bytes)
            .unwrap()
            .unwrap();
        let config = AacConfig::parse(&index.audio_specific_config).unwrap();
        let mut retained = AacDecoder::new(config.clone()).unwrap();
        let mut reduced = AacDecoder::new(config).unwrap();
        retained.reference_tns = true;
        reduced.reference_tns = true;
        let mut values = 0usize;
        let mut bit_differences = 0usize;
        let mut peak = 0.0f64;
        let mut error = 0.0f64;
        let mut energy = 0.0f64;
        for (packet, sample) in index.samples.iter().enumerate() {
            let start = usize::try_from(sample.offset).unwrap();
            let data = &bytes[start..start + sample.size as usize];
            let expected = retained.decode(data).unwrap();
            let actual =
                crate::audio::transform::with_reduced_mdct(|| reduced.decode(data)).unwrap();
            assert_eq!(actual.sample_rate, expected.sample_rate);
            assert_eq!(actual.channels, expected.channels);
            assert_eq!(
                actual.samples.len(),
                expected.samples.len(),
                "packet {packet}"
            );
            for (&a, &b) in actual.samples.iter().zip(&expected.samples) {
                assert!(a.is_finite() && b.is_finite(), "packet {packet}");
                bit_differences += usize::from(a.to_bits() != b.to_bits());
                let difference = f64::from(a) - f64::from(b);
                peak = peak.max(difference.abs());
                error += difference * difference;
                energy += f64::from(b) * f64::from(b);
                values += 1;
            }
        }
        assert!(energy > 0.0);
        eprintln!(
            "AAC reduced MDCT: packets={} pcm_values={values} bit_differences={bit_differences} peak_error={peak} relative_rms={}",
            index.samples.len(),
            (error / energy).sqrt()
        );
        // No new PCM tolerance is authorized: keep exact parity as the
        // conservative acceptance gate while reporting numerical differences.
        assert_eq!(
            bit_differences, 0,
            "reduced MDCT PCM differs; numerical acceptance pending"
        );
    }

    #[test]
    #[ignore = "set WEBMEDIA_AAC_MP4 for whole-fixture TNS no-op/reference ABBA measurements"]
    fn tns_noop_fixture_abba_is_bit_exact() {
        let bytes = std::fs::read(std::env::var("WEBMEDIA_AAC_MP4").unwrap()).unwrap();
        let index = crate::video::mp4::Mp4AudioIndex::parse_prefix(&bytes)
            .unwrap()
            .unwrap();
        let config = AacConfig::parse(&index.audio_specific_config).unwrap();
        let mut expected = Vec::new();
        let mut decoder = AacDecoder::new(config.clone()).unwrap();
        decoder.reference_tns = true;
        for sample in &index.samples {
            let start = sample.offset as usize;
            expected.extend(
                decoder
                    .decode(&bytes[start..start + sample.size as usize])
                    .unwrap()
                    .samples,
            );
        }
        for reference in [true, false, false, true] {
            let mut decoder = AacDecoder::new(config.clone()).unwrap();
            decoder.reference_tns = reference;
            let mut offset = 0;
            let mut elapsed = std::time::Duration::ZERO;
            for sample in &index.samples {
                let start = sample.offset as usize;
                let time = std::time::Instant::now();
                let audio = decoder
                    .decode(&bytes[start..start + sample.size as usize])
                    .unwrap();
                elapsed += time.elapsed();
                for &value in &audio.samples {
                    assert_eq!(
                        value.to_bits(),
                        expected[offset].to_bits(),
                        "PCM offset {offset}"
                    );
                    offset += 1;
                }
            }
            assert_eq!(offset, expected.len());
            eprintln!(
                "AAC TNS ABBA reference={reference} packets={} pcm_values={offset} decode_ms={:.3} tns_ms={:.3} noop_windows={}/{} parse/IQ/stereo/TNS+synth/PCM_ms={:?}",
                index.samples.len(),
                elapsed.as_secs_f64() * 1000.0,
                decoder.tns_time.as_secs_f64() * 1000.0,
                decoder.tns_windows[0],
                decoder.tns_windows[1],
                decoder.stage_times.map(|time| time.as_secs_f64() * 1000.0)
            );
        }
    }

    #[test]
    #[ignore = "set WEBMEDIA_AAC_MP4 for same-binary cached/reference ABBA fixture measurements"]
    fn quantizer_cache_fixture_abba_is_bit_exact() {
        let bytes = std::fs::read(std::env::var("WEBMEDIA_AAC_MP4").unwrap()).unwrap();
        let index = crate::video::mp4::Mp4AudioIndex::parse_prefix(&bytes)
            .unwrap()
            .unwrap();
        let config = AacConfig::parse(&index.audio_specific_config).unwrap();
        let mut expected = Vec::new();
        let mut decoder = AacDecoder::new(config.clone()).unwrap();
        decoder.reference_quantizer = true;
        for sample in &index.samples {
            let start = sample.offset as usize;
            expected.extend(
                decoder
                    .decode(&bytes[start..start + sample.size as usize])
                    .unwrap()
                    .samples,
            );
        }
        for round in 0..1 {
            for reference in [true, false, false, true] {
                let mut decoder = AacDecoder::new(config.clone()).unwrap();
                decoder.reference_quantizer = reference;
                let mut offset = 0;
                let mut elapsed = std::time::Duration::ZERO;
                for sample in &index.samples {
                    let start = sample.offset as usize;
                    let time = std::time::Instant::now();
                    let audio = decoder
                        .decode(&bytes[start..start + sample.size as usize])
                        .unwrap();
                    elapsed += time.elapsed();
                    for &value in &audio.samples {
                        assert_eq!(
                            value.to_bits(),
                            expected[offset].to_bits(),
                            "PCM offset {offset}"
                        );
                        offset += 1;
                    }
                }
                assert_eq!(offset, expected.len());
                eprintln!(
                    "AAC ABBA round={round} reference={reference} packets={} decode_ms={:.3} parse/IQ/stereo/TNS+synth/PCM_ms={:?}",
                    index.samples.len(),
                    elapsed.as_secs_f64() * 1000.0,
                    decoder.stage_times.map(|time| time.as_secs_f64() * 1000.0)
                );
            }
        }
    }

    #[test]
    #[ignore = "set WEBMEDIA_AAC_MP4; optionally WEBMEDIA_AAC_PCM_OUT for raw interleaved f32 PCM"]
    fn decodes_fixture_audio_to_pcm() {
        let bytes = std::fs::read(std::env::var("WEBMEDIA_AAC_MP4").unwrap()).unwrap();
        let index = crate::video::mp4::Mp4AudioIndex::parse_prefix(&bytes)
            .unwrap()
            .unwrap();
        let config = AacConfig::parse(&index.audio_specific_config).unwrap();
        let mut decoder = AacDecoder::new(config).unwrap();
        let mut pcm = Vec::new();
        let start_time = std::time::Instant::now();
        for (number, sample) in index.samples.iter().enumerate() {
            let start = sample.offset as usize;
            let audio = decoder
                .decode(&bytes[start..start + sample.size as usize])
                .unwrap_or_else(|error| panic!("AAC packet {number}: {error:?}"));
            assert_eq!(audio.samples.len(), 2048);
            assert!(audio.samples.iter().all(|sample| sample.is_finite()));
            pcm.extend(audio.samples);
        }
        assert!(
            pcm.iter().any(|sample| sample.abs() > 1e-6),
            "actual nonzero audio required"
        );
        eprintln!(
            "AAC: {} packets, {} samples, {:.3}s decode, peak={}",
            index.samples.len(),
            pcm.len(),
            start_time.elapsed().as_secs_f64(),
            pcm.iter().map(|s| s.abs()).fold(0.0f32, f32::max)
        );
        if let Ok(path) = std::env::var("WEBMEDIA_AAC_PCM_OUT") {
            let bytes: Vec<_> = pcm.iter().flat_map(|sample| sample.to_le_bytes()).collect();
            std::fs::write(path, bytes).unwrap();
        }
    }
}
