//! Stateful Vorbis PCM synthesis. Container timing and discard padding stay outside.

use super::{Headers, setup::Setup, transform::MdctPlan};
use crate::video::backend::{AudioSamples, MediaDecodeError};
use std::f64::consts::FRAC_PI_2;

pub struct Decoder {
    setup: Setup,
    block_sizes: [usize; 2],
    sample_rate: u32,
    channels: usize,
    transforms: [MdctPlan; 2],
    windows: [Vec<f64>; 5],
    spectrum: super::spectrum::Workspace,
    current: Vec<Vec<f64>>,
    previous: Vec<Vec<f64>>,
    previous_size: Option<usize>,
}

impl Decoder {
    /// Committed overlap window length, or None before priming/after reset.
    pub fn previous_window_frames(&self) -> Option<usize> {
        self.previous_size
    }

    /// Read only the audio mode/window header, without decoding or advancing
    /// overlap state. Non-audio packets return None. The length is in frames,
    /// not interleaved samples, and includes both halves of the MDCT window.
    pub fn packet_window_frames(&self, packet: &[u8]) -> Result<Option<usize>, MediaDecodeError> {
        let mut bits = super::entropy::PacketBits::new(packet);
        Ok(self
            .setup
            .packet_header(&mut bits)?
            .map(|header| self.block_sizes[usize::from(header.long)]))
    }

    /// Elapsed frames for a packet from the current previous-window state.
    /// Priming consumes half a window but produces no PCM; subsequent audio
    /// packets produce (previous_window + current_window) / 4 PCM frames.
    /// This is a header preview, not validation of the packet's floor/residue.
    pub fn packet_duration_frames(&self, packet: &[u8]) -> Result<Option<usize>, MediaDecodeError> {
        Ok(self.packet_window_frames(packet)?.map(|size| {
            self.previous_size
                .map_or(size / 2, |previous| previous / 4 + size / 4)
        }))
    }

    /// Laced packets share a block's discard padding; trim only after decoding
    /// the complete block, not independently for each packet.
    pub fn decode_block(
        &mut self,
        block: &crate::audio::webm::WebmAudioBlock,
    ) -> Result<Option<AudioSamples>, MediaDecodeError> {
        Ok(self
            .decode_block_with_offset(block)?
            .map(|(samples, _)| samples))
    }

    /// PCM start offset in sample frames relative to the first laced packet.
    /// Priming packets and leading discard padding advance the first sample.
    pub fn decode_block_with_offset(
        &mut self,
        block: &crate::audio::webm::WebmAudioBlock,
    ) -> Result<Option<(AudioSamples, usize)>, MediaDecodeError> {
        let mut samples = Vec::new();
        let mut start_offset = 0usize;
        for packet in &block.packets {
            let unprimed = self.previous_size.is_none();
            if let Some(audio) = self.decode(packet)? {
                samples.extend(audio.samples);
            } else if unprimed && let Some(previous) = self.previous_size {
                start_offset += previous / 2;
            }
        }
        if samples.is_empty() {
            return Ok(None);
        }
        let mut audio = AudioSamples {
            sample_rate: self.sample_rate,
            channels: self.channels as u16,
            samples,
        };
        if let Some(padding) = block.discard_padding_ns {
            let trimmed = crate::audio::webm::trim_discard_padding(&mut audio, padding)?;
            if padding < 0 {
                start_offset += trimmed;
            }
        }
        Ok(Some((audio, start_offset)))
    }

    pub fn new(
        headers: &Headers,
        entry_budget: usize,
        lookup_budget: usize,
    ) -> Result<Self, MediaDecodeError> {
        let identification = &headers.identification;
        let block_sizes = identification.block_sizes;
        if identification.sample_rate == 0
            || identification.channels == 0
            || block_sizes[0] > block_sizes[1]
        {
            return Err(MediaDecodeError::InvalidData(
                "invalid Vorbis stream identification".into(),
            ));
        }
        let transforms = [
            MdctPlan::new(block_sizes[0])?,
            MdctPlan::new(block_sizes[1])?,
        ];
        let setup = headers.decode_setup(entry_budget, lookup_budget)?;
        let windows = std::array::from_fn(|index| {
            let long = index != 0;
            window(
                block_sizes[usize::from(long)],
                block_sizes[0],
                !long || (index - 1) & 1 != 0,
                !long || (index - 1) & 2 != 0,
            )
        });
        Ok(Self {
            setup,
            block_sizes,
            sample_rate: identification.sample_rate,
            channels: usize::from(identification.channels),
            transforms,
            windows,
            spectrum: super::spectrum::Workspace::default(),
            current: vec![vec![0.0; block_sizes[1]]; usize::from(identification.channels)],
            previous: vec![vec![0.0; block_sizes[1]]; usize::from(identification.channels)],
            previous_size: None,
        })
    }

    /// First audio packet primes overlap state and returns no PCM. Non-audio
    /// packets are ignored without disturbing the previous window.
    pub fn decode(&mut self, bytes: &[u8]) -> Result<Option<AudioSamples>, MediaDecodeError> {
        let Some(header) =
            self.setup
                .spectrum_with_workspace(bytes, self.block_sizes, &mut self.spectrum)?
        else {
            return Ok(None);
        };
        let size_index = usize::from(header.long);
        let size = self.block_sizes[size_index];
        let window_index = if header.long {
            1 + usize::from(header.previous_long) + 2 * usize::from(header.next_long)
        } else {
            0
        };
        if self.spectrum.channels.len() != self.channels {
            return Err(MediaDecodeError::InvalidData(
                "Vorbis spectrum channel count mismatch".into(),
            ));
        }
        let current = &mut self.current;
        for (channel, spectrum) in current.iter_mut().zip(&self.spectrum.channels) {
            channel.resize(size, 0.0);
            self.transforms[size_index].inverse(spectrum, channel)?;
            for (sample, &window) in channel.iter_mut().zip(&self.windows[window_index]) {
                *sample *= window;
            }
        }
        let samples = if let Some(previous_size) = self.previous_size {
            let previous = &self.previous;
            let count = previous_size / 4 + size / 4;
            let mut samples = Vec::with_capacity(count * self.channels);
            for time in 0..count {
                for (previous, current) in previous.iter().zip(current.iter()) {
                    let old = previous
                        .get(previous_size / 2 + time)
                        .copied()
                        .unwrap_or(0.0);
                    let current_index = (size / 2 + time).checked_sub(count);
                    let new = current_index
                        .and_then(|index| current.get(index))
                        .copied()
                        .unwrap_or(0.0);
                    let sample = (old + new) as f32;
                    if !sample.is_finite() {
                        return Err(MediaDecodeError::InvalidData(
                            "nonfinite Vorbis PCM sample".into(),
                        ));
                    }
                    samples.push(sample);
                }
            }
            Some(AudioSamples {
                sample_rate: self.sample_rate,
                channels: self.channels as u16,
                samples,
            })
        } else {
            None
        };
        // Scratch can change on failure, but the committed overlap and timing
        // state change only after every output sample has been checked.
        std::mem::swap(&mut self.previous, &mut self.current);
        self.previous_size = Some(size);
        Ok(samples)
    }

    pub fn reset(&mut self) {
        self.previous_size = None;
    }
}

fn window(size: usize, short: usize, previous_long: bool, next_long: bool) -> Vec<f64> {
    let (left_start, left_end) = if previous_long {
        (0, size / 2)
    } else {
        (size / 4 - short / 4, size / 4 + short / 4)
    };
    let (right_start, right_end) = if next_long {
        (size / 2, size)
    } else {
        (3 * size / 4 - short / 4, 3 * size / 4 + short / 4)
    };
    let mut output = vec![0.0; size];
    output[left_end..right_start].fill(1.0);
    for (offset, value) in output[left_start..left_end].iter_mut().enumerate() {
        let phase = FRAC_PI_2 * (offset as f64 + 0.5) / (left_end - left_start) as f64;
        *value = (FRAC_PI_2 * phase.sin().powi(2)).sin();
    }
    for (offset, value) in output[right_start..right_end].iter_mut().enumerate() {
        let phase =
            FRAC_PI_2 * (offset as f64 + 0.5) / (right_end - right_start) as f64 + FRAC_PI_2;
        *value = (FRAC_PI_2 * phase.sin().powi(2)).sin();
    }
    output
}

#[cfg(test)]
mod tests {
    use super::super::{
        entropy::{Codebook, PacketBits},
        setup::{Floor, FloorOne, FloorZero, Mapping, Mode, Residue, Submap},
    };
    use super::*;

    fn pack(fields: &[(u32, u8)]) -> Vec<u8> {
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
                bytes[position / 8] |= ((value >> bit) as u8 & 1) << (position % 8);
                position += 1;
            }
        }
        bytes
    }

    fn synthetic_decoder() -> Decoder {
        let scalar = pack(&[
            (0x564342, 24),
            (1, 16),
            (1, 24),
            (0, 1),
            (0, 1),
            (0, 5),
            (0, 4),
        ]);
        let vector = pack(&[
            (0x564342, 24),
            (2, 16),
            (1, 24),
            (0, 1),
            (0, 1),
            (0, 5),
            (2, 4),
            (0, 32),
            ((788 << 21) | 1, 32),
            (1, 4),
            (0, 1),
            (1, 2),
            (2, 2),
        ]);
        let codebooks = vec![
            Codebook::parse(&mut PacketBits::new(&scalar), &mut 1, &mut 0).unwrap(),
            Codebook::parse(&mut PacketBits::new(&vector), &mut 1, &mut 2).unwrap(),
        ];
        let setup = Setup {
            codebooks,
            floors: vec![Floor::One(FloorOne {
                partitions: vec![],
                classes: vec![],
                multiplier: 1,
                x: vec![0, 128],
                neighbors: vec![(0, 0); 2],
                sorted_points: vec![0, 1],
            })],
            residues: vec![Residue {
                kind: 2,
                begin: 0,
                end: 96,
                partition_size: 4,
                classbook: 0,
                books: vec![[Some(1), None, None, None, None, None, None, None]],
            }],
            mappings: vec![Mapping {
                coupling: vec![(0, 1)],
                mux: vec![0; 3],
                submaps: vec![Submap {
                    floor: 0,
                    residue: 0,
                }],
            }],
            // Three modes deliberately leave the two-bit mode value 3 invalid.
            modes: vec![
                Mode {
                    long: false,
                    mapping: 0,
                },
                Mode {
                    long: true,
                    mapping: 0,
                },
                Mode {
                    long: false,
                    mapping: 0,
                },
            ],
        };
        Decoder {
            setup,
            block_sizes: [64, 256],
            sample_rate: 48000,
            channels: 3,
            transforms: [MdctPlan::new(64).unwrap(), MdctPlan::new(256).unwrap()],
            windows: std::array::from_fn(|index| {
                window(
                    if index == 0 { 64 } else { 256 },
                    64,
                    index == 0 || (index - 1) & 1 != 0,
                    index == 0 || (index - 1) & 2 != 0,
                )
            }),
            spectrum: super::super::spectrum::Workspace::default(),
            current: vec![vec![0.0; 256]; 3],
            previous: vec![vec![0.0; 256]; 3],
            previous_size: None,
        }
    }

    fn packet(long: bool, previous: bool, next: bool) -> Vec<u8> {
        let mut fields = vec![(0, 1), (u32::from(long), 2)];
        if long {
            fields.extend([(u32::from(previous), 1), (u32::from(next), 1)]);
        }
        for _ in 0..3 {
            fields.extend([(1, 1), (255, 8), (255, 8)]);
        }
        // 24 partitions, each with one classword and two vector codewords.
        fields.extend([(0, 1); 72]);
        pack(&fields)
    }

    // Independent scalar MDCT and piecewise window, with no production helpers.
    fn reference_block(long: bool, previous: bool, next: bool) -> Vec<Vec<f64>> {
        let size = if long { 256 } else { 64 };
        let left_width = if long && !previous { 32 } else { size / 2 };
        let right_width = if long && !next { 32 } else { size / 2 };
        let left = size / 4 - left_width / 2;
        let right = 3 * size / 4 - right_width / 2;
        (0..3)
            .map(|channel| {
                (0..size)
                    .map(|time| {
                        let weight = if time < left || time >= right + right_width {
                            0.0
                        } else if time < left + left_width {
                            let phase = std::f64::consts::PI * (time - left) as f64
                                / (2 * left_width) as f64
                                + std::f64::consts::PI / (4 * left_width) as f64;
                            (FRAC_PI_2 * phase.sin().powi(2)).sin()
                        } else if time < right {
                            1.0
                        } else {
                            let phase = std::f64::consts::PI * (right + right_width - time) as f64
                                / (2 * right_width) as f64
                                - std::f64::consts::PI / (4 * right_width) as f64;
                            (FRAC_PI_2 * phase.sin().powi(2)).sin()
                        };
                        let sum: f64 = (0..32)
                            .map(|bin| {
                                let magnitude = (1 + (3 * bin) % 2) as f64;
                                let angle = (1 + (3 * bin + 1) % 2) as f64;
                                let value = match channel {
                                    0 => magnitude,
                                    1 => magnitude - angle,
                                    _ => (1 + (3 * bin + 2) % 2) as f64,
                                };
                                value
                                    * (std::f64::consts::TAU / size as f64
                                        * (time as f64 + 0.5 + size as f64 / 4.0)
                                        * (bin as f64 + 0.5))
                                        .cos()
                            })
                            .sum();
                        sum * weight
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn mixed_packet_windows_and_three_channel_residue_match_scalar_oracle() {
        for pattern in 0..16 {
            let modes: Vec<_> = (0..4).map(|bit| pattern & (1 << bit) != 0).collect();
            let mut decoder = synthetic_decoder();
            let mut previous: Option<Vec<Vec<f64>>> = None;
            for (index, &long) in modes.iter().enumerate() {
                let prior = index > 0 && modes[index - 1];
                let next = index + 1 < modes.len() && modes[index + 1];
                let current = reference_block(long, prior, next);
                let decoded = decoder.decode(&packet(long, prior, next)).unwrap();
                if let Some(previous) = previous {
                    let audio = decoded.unwrap();
                    let count = (previous[0].len() + current[0].len()) / 4;
                    assert_eq!(audio.samples.len(), count * 3);
                    for time in 0..count {
                        for channel in 0..3 {
                            let old = previous[channel]
                                .get(previous[0].len() / 2 + time)
                                .copied()
                                .unwrap_or(0.0);
                            let relative =
                                time as isize + current[0].len() as isize / 2 - count as isize;
                            let new = if relative < 0 {
                                0.0
                            } else {
                                current[channel]
                                    .get(relative as usize)
                                    .copied()
                                    .unwrap_or(0.0)
                            };
                            assert!(
                                (f64::from(audio.samples[time * 3 + channel]) - old - new).abs()
                                    < 2e-6,
                                "pattern {pattern} packet {index} time {time} channel {channel}"
                            );
                        }
                    }
                } else {
                    assert!(decoded.is_none());
                }
                previous = Some(current);
            }
        }
    }

    #[test]
    fn rejected_headers_and_non_audio_packets_preserve_overlap_and_reset_reprimes() {
        let mut decoder = synthetic_decoder();
        let mut control = synthetic_decoder();
        let first = packet(false, false, true);
        assert!(decoder.decode(&first).unwrap().is_none());
        assert!(control.decode(&first).unwrap().is_none());
        assert!(decoder.decode(&[]).is_err());
        assert!(decoder.decode(&[6]).is_err());
        assert!(decoder.decode(&[1]).unwrap().is_none());
        let next = packet(true, false, false);
        assert_eq!(
            decoder.decode(&next).unwrap().unwrap().samples,
            control.decode(&next).unwrap().unwrap().samples
        );
        decoder.reset();
        assert!(decoder.decode(&next).unwrap().is_none());
        // A peeled packet with a valid header is a real silent overlap block.
        assert!(decoder.decode(&[0]).unwrap().is_some());
    }

    #[test]
    fn mixed_lacing_preserves_priming_offset_and_leading_padding() {
        let packets = vec![
            vec![1],
            packet(false, false, true),
            packet(true, false, false),
            packet(false, true, false),
        ];
        let mut control = synthetic_decoder();
        let mut expected = Vec::new();
        for packet in &packets {
            if let Some(audio) = control.decode(packet).unwrap() {
                expected.extend(audio.samples);
            }
        }
        let block = crate::audio::webm::WebmAudioBlock {
            timestamp_ns: 0,
            packets,
            discard_padding_ns: Some(-41667),
        };
        let (audio, offset) = synthetic_decoder()
            .decode_block_with_offset(&block)
            .unwrap()
            .unwrap();
        assert_eq!(offset, 64 / 2 + 2);
        assert_eq!(audio.samples, expected[2 * 3..]);
        assert_eq!(audio.samples.len(), (80 + 80 - 2) * 3);
    }

    #[test]
    fn peeled_three_channel_residue_preserves_every_complete_vector() {
        let decoder = synthetic_decoder();
        let bytes = packet(false, false, false);
        // Three-bit header, then three 17-bit floors: residue starts at bit 54.
        for length in 1..=bytes.len() {
            let decoded = decoder
                .setup
                .spectrum(&bytes[..length], [64, 256])
                .unwrap()
                .unwrap();
            let available = (length * 8).saturating_sub(54).min(72);
            let scalars = (0..available).filter(|bit| bit % 3 != 0).count() * 2;
            let source = |index: usize| {
                if index < scalars {
                    (1 + index % 2) as f64
                } else {
                    0.0
                }
            };
            for bin in 0..32 {
                let magnitude = source(3 * bin);
                let angle = source(3 * bin + 1);
                assert_eq!(
                    decoded.channels[0][bin], magnitude,
                    "prefix {length} bin {bin}"
                );
                assert_eq!(
                    decoded.channels[1][bin],
                    magnitude - angle,
                    "prefix {length} bin {bin}"
                );
                assert_eq!(
                    decoded.channels[2][bin],
                    source(3 * bin + 2),
                    "prefix {length} bin {bin}"
                );
            }
        }
    }

    #[test]
    fn packet_timing_preview_is_read_only_and_includes_priming() {
        let mut decoder = synthetic_decoder();
        let short = packet(false, false, true);
        let long = packet(true, false, true);
        assert_eq!(decoder.previous_window_frames(), None);
        assert_eq!(decoder.packet_window_frames(&short).unwrap(), Some(64));
        assert_eq!(decoder.packet_window_frames(&long).unwrap(), Some(256));
        assert_eq!(decoder.packet_duration_frames(&short).unwrap(), Some(32));
        assert_eq!(decoder.packet_duration_frames(&long).unwrap(), Some(128));
        assert_eq!(decoder.packet_duration_frames(&[1]).unwrap(), None);
        assert!(decoder.packet_duration_frames(&[]).is_err());
        assert!(decoder.packet_duration_frames(&[6]).is_err());
        assert!(decoder.decode(&short).unwrap().is_none());
        assert_eq!(decoder.previous_window_frames(), Some(64));
        assert_eq!(decoder.packet_duration_frames(&long).unwrap(), Some(80));
        assert_eq!(
            decoder.decode(&long).unwrap().unwrap().samples.len(),
            80 * 3
        );
        assert_eq!(decoder.packet_duration_frames(&short).unwrap(), Some(80));
        assert_eq!(decoder.previous_window_frames(), Some(256));
        assert_eq!(decoder.packet_duration_frames(&long).unwrap(), Some(128));
        decoder.reset();
        assert_eq!(decoder.previous_window_frames(), None);
        assert_eq!(decoder.packet_duration_frames(&short).unwrap(), Some(32));
    }

    #[test]
    fn synthesis_failure_keeps_committed_overlap_and_overwrites_dirty_scratch() {
        let mut decoder = synthetic_decoder();
        let mut control = synthetic_decoder();
        let short = packet(false, false, true);
        let long = packet(true, false, false);
        decoder.decode(&short).unwrap();
        control.decode(&short).unwrap();
        let healthy = decoder.setup.codebooks[1].clone();
        // A finite spectrum whose synthesized samples overflow f32 forces an
        // error after time-domain scratch has already been written.
        let huge = pack(&[
            (0x564342, 24),
            (2, 16),
            (1, 24),
            (0, 1),
            (0, 1),
            (0, 5),
            (2, 4),
            (0, 32),
            ((1000 << 21) | 1, 32),
            (1, 4),
            (0, 1),
            (1, 2),
            (2, 2),
        ]);
        decoder.setup.codebooks[1] =
            Codebook::parse(&mut PacketBits::new(&huge), &mut 1, &mut 2).unwrap();
        assert!(decoder.decode(&long).is_err());
        assert_eq!(decoder.previous_window_frames(), Some(64));
        assert_eq!(decoder.previous, control.previous);
        decoder.setup.codebooks[1] = healthy;
        assert_eq!(
            decoder.decode(&long).unwrap().unwrap().samples,
            control.decode(&long).unwrap().unwrap().samples
        );
    }

    #[test]
    fn reused_spectrum_clears_partial_packets_and_double_buffers_survive_reset() {
        let first = packet(false, false, true);
        let full = packet(true, false, false);
        let mut reused = synthetic_decoder();
        let pointers: Vec<_> = reused
            .current
            .iter()
            .chain(&reused.previous)
            .map(|channel| channel.as_ptr())
            .collect();
        for length in 1..=full.len() {
            reused.reset();
            let mut fresh = synthetic_decoder();
            reused.decode(&first).unwrap();
            fresh.decode(&first).unwrap();
            assert_eq!(
                reused.decode(&full[..length]).unwrap().unwrap().samples,
                fresh.decode(&full[..length]).unwrap().unwrap().samples
            );
            assert_eq!(
                reused.decode(&first).unwrap().unwrap().samples,
                fresh.decode(&first).unwrap().unwrap().samples
            );
            for channel in reused.current.iter().chain(&reused.previous) {
                assert!(pointers.contains(&channel.as_ptr()));
            }
        }
    }

    #[test]
    fn floor_zero_workspace_handles_mixed_modes_partial_packets_and_curve_failure() {
        for order in 1usize..=3 {
            let make_decoder = || {
                let mut decoder = synthetic_decoder();
                decoder.setup.floors.push(Floor::Zero(FloorZero {
                    order,
                    rate: 48000,
                    bark_map_size: 32,
                    amplitude_bits: 6,
                    amplitude_offset: 60,
                    books: vec![1],
                }));
                let mut mapping = decoder.setup.mappings[0].clone();
                mapping.submaps[0].floor = 1;
                decoder.setup.mappings.push(mapping);
                decoder.setup.modes[1].mapping = 1;
                decoder
            };
            let mut fields = vec![(0, 1), (1, 2), (0, 1), (0, 1)];
            for _ in 0..3 {
                fields.extend([(31, 6), (0, 1)]);
                fields.extend(vec![(0, 1); order.div_ceil(2)]);
            }
            fields.extend([(0, 1); 72]);
            let long = pack(&fields);
            let short = packet(false, false, true);
            let mut reused = make_decoder();
            for length in 1..=long.len() {
                reused.reset();
                let mut fresh = make_decoder();
                reused.decode(&short).unwrap();
                fresh.decode(&short).unwrap();
                assert_eq!(
                    reused.decode(&long[..length]).unwrap().unwrap().samples,
                    fresh.decode(&long[..length]).unwrap().unwrap().samples
                );
                assert_eq!(
                    reused.decode(&short).unwrap().unwrap().samples,
                    fresh.decode(&short).unwrap().unwrap().samples
                );
            }
            if order == 2 {
                let mut control = make_decoder();
                reused.reset();
                reused.decode(&short).unwrap();
                control.decode(&short).unwrap();
                // Zero-valued LSP vectors yield a singular even-order DC
                // envelope, after all channel floor workspaces were decoded.
                let healthy = reused.setup.codebooks[1].clone();
                let singular = pack(&[
                    (0x564342, 24),
                    (2, 16),
                    (1, 24),
                    (0, 1),
                    (0, 1),
                    (0, 5),
                    (2, 4),
                    (0, 32),
                    (0, 32),
                    (0, 4),
                    (0, 1),
                    (0, 1),
                    (0, 1),
                ]);
                reused.setup.codebooks[1] =
                    Codebook::parse(&mut PacketBits::new(&singular), &mut 1, &mut 2).unwrap();
                assert!(reused.decode(&long).is_err());
                assert_eq!(reused.previous_window_frames(), Some(64));
                assert_eq!(reused.previous, control.previous);
                reused.setup.codebooks[1] = healthy;
                assert_eq!(
                    reused.decode(&long).unwrap().unwrap().samples,
                    control.decode(&long).unwrap().unwrap().samples
                );
            }
        }
    }

    #[test]
    fn same_size_window_preserves_overlap_energy() {
        for size in [64, 256, 2048, 8192] {
            let values = window(size, size, true, true);
            for index in 0..size / 2 {
                assert!(
                    (values[index].powi(2) + values[index + size / 2].powi(2) - 1.0).abs() < 1e-14
                );
            }
        }
    }

    #[test]
    fn mixed_windows_align_slopes_and_flat_regions() {
        let short = window(256, 256, true, true);
        let long = window(2048, 256, false, false);
        assert!(long[..448].iter().all(|&value| value == 0.0));
        assert_eq!(&long[448..576], &short[..128]);
        assert!(long[576..1472].iter().all(|&value| value == 1.0));
        assert_eq!(&long[1472..1600], &short[128..]);
        assert!(long[1600..].iter().all(|&value| value == 0.0));
    }

    #[cfg(feature = "video")]
    #[test]
    #[ignore = "requires VORBIS_FIXTURE_WEBM and VORBIS_FIXTURE_PCM binary oracle paths"]
    fn supplied_vorbis_full_stream_matches_oracle_and_reset_loops() {
        use crate::audio::webm::{WebmAudioCodec, trim_discard_padding};
        use crate::video::webm::{WebmPacket, WebmStream};
        use std::io::Read;

        let path = std::env::var("VORBIS_FIXTURE_WEBM").expect("missing Vorbis fixture path");
        let reference_path = std::env::var("VORBIS_FIXTURE_PCM").expect("missing PCM oracle path");
        let mut file = std::fs::File::open(path).unwrap();
        let mut stream = WebmStream::new();
        let mut bytes = [0u8; 4093];
        let mut blocks = Vec::new();
        loop {
            let count = file.read(&mut bytes).unwrap();
            if count == 0 {
                break;
            }
            for packet in stream.push_packets(&bytes[..count]).unwrap() {
                if let WebmPacket::Audio(block) = packet {
                    blocks.push(block);
                }
            }
        }
        stream.finish().unwrap();
        let track = stream.audio_track().unwrap();
        assert_eq!(
            track.codec,
            WebmAudioCodec::Vorbis,
            "fixture does not contain Vorbis audio"
        );
        let headers = Headers::from_webm(&track.codec_private).unwrap();
        let rate = headers.identification.sample_rate;
        let channels = u16::from(headers.identification.channels);
        assert_eq!(track.sampling_frequency, f64::from(rate));
        assert_eq!(track.channels, channels);
        assert!(!blocks.is_empty());
        let mut decoder = Decoder::new(&headers, 1 << 20, 1 << 20).unwrap();
        let decode_packets = |decoder: &mut Decoder| {
            let mut pcm = Vec::new();
            for (block_index, block) in blocks.iter().enumerate() {
                let mut audio = AudioSamples {
                    sample_rate: rate,
                    channels,
                    samples: Vec::new(),
                };
                for (packet_index, packet) in block.packets.iter().enumerate() {
                    let previous = decoder.previous_window_frames();
                    let duration = decoder.packet_duration_frames(packet).unwrap();
                    let decoded = decoder.decode(packet).unwrap_or_else(|error| {
                        panic!("block {block_index} packet {packet_index}: {error:?}")
                    });
                    if let Some(decoded) = decoded {
                        assert!(previous.is_some());
                        assert_eq!(decoded.sample_rate, rate);
                        assert_eq!(decoded.channels, channels);
                        assert_eq!(
                            Some(decoded.samples.len() / usize::from(channels)),
                            duration
                        );
                        assert!(decoded.samples.iter().all(|value| value.is_finite()));
                        audio.samples.extend(decoded.samples);
                    } else if let Some(duration) = duration {
                        assert!(previous.is_none());
                        assert_eq!(Some(duration * 2), decoder.previous_window_frames());
                    }
                }
                if !audio.samples.is_empty() {
                    if let Some(padding) = block.discard_padding_ns {
                        trim_discard_padding(&mut audio, padding).unwrap();
                    }
                    pcm.extend(audio.samples);
                }
            }
            pcm
        };
        let pcm = decode_packets(&mut decoder);
        let reference = std::fs::read(reference_path).unwrap();
        assert_eq!(reference.len(), pcm.len() * 4);
        assert!(!pcm.is_empty());
        let mut maximum = 0f32;
        for (&sample, bytes) in pcm.iter().zip(reference.chunks_exact(4)) {
            let expected = f32::from_le_bytes(bytes.try_into().unwrap());
            assert!(expected.is_finite());
            maximum = maximum.max((sample - expected).abs());
        }
        assert!(maximum < 3e-7, "maximum PCM oracle error {maximum:e}");
        let same_bits = |candidate: &[f32]| {
            candidate.len() == pcm.len()
                && candidate
                    .iter()
                    .zip(&pcm)
                    .all(|(left, right)| left.to_bits() == right.to_bits())
        };
        for pass in 1..=3 {
            decoder.reset();
            assert_eq!(decoder.previous_window_frames(), None);
            assert!(
                same_bits(&decode_packets(&mut decoder)),
                "PCM reset loop {pass} differs"
            );
        }
        decoder.reset();
        let mut block_pcm = Vec::new();
        for block in &blocks {
            if let Some((audio, _)) = decoder.decode_block_with_offset(block).unwrap() {
                block_pcm.extend(audio.samples);
            }
        }
        assert!(
            same_bits(&block_pcm),
            "block delivery differs from packet delivery"
        );
        eprintln!(
            "Vorbis rate={rate} channels={channels} blocks={} samples={} max_error={maximum:e}; three reset loops and block/packet delivery are bit-identical",
            blocks.len(),
            pcm.len()
        );
    }
}
