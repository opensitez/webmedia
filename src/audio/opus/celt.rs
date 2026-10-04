//! CELT spectral layouts from RFC 6716 tables 2 and 55.
//! These are synthesis building blocks, not a complete Opus PCM decoder.

use crate::audio::transform::MdctPlan;
use crate::video::backend::MediaDecodeError;
use std::ops::Range;

const BAND_BOUNDARIES: [usize; 22] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    log_blocks: u8,
    first_band: usize,
    end_band: usize,
}

impl Layout {
    pub fn from_configuration(configuration: u8) -> Result<Option<Self>, MediaDecodeError> {
        let (log_blocks, first_band, end_band) = match configuration {
            0..=11 => return Ok(None),
            12..=15 => (
                2 + (configuration & 1),
                17,
                if configuration < 14 { 19 } else { 21 },
            ),
            16..=31 => (
                configuration & 3,
                0,
                [13, 17, 19, 21][usize::from((configuration - 16) >> 2)],
            ),
            _ => {
                return Err(MediaDecodeError::InvalidData(
                    "invalid Opus configuration".into(),
                ));
            }
        };
        Ok(Some(Self {
            log_blocks,
            first_band,
            end_band,
        }))
    }

    pub fn samples(&self) -> usize {
        120 << self.log_blocks
    }
    pub fn bands(&self) -> Range<usize> {
        self.first_band..self.end_band
    }
    pub fn blocks(&self, transient: bool) -> Result<usize, MediaDecodeError> {
        if transient && self.log_blocks == 0 {
            return Err(MediaDecodeError::InvalidData(
                "transient in a single-block Opus frame".into(),
            ));
        }
        Ok(if transient { 1 << self.log_blocks } else { 1 })
    }

    /// Ranges include every interleaved short-transform coefficient in a band.
    pub fn band(&self, band: usize) -> Result<Range<usize>, MediaDecodeError> {
        if !self.bands().contains(&band) {
            return Err(MediaDecodeError::InvalidData(
                "inactive Opus spectral band".into(),
            ));
        }
        Ok((BAND_BOUNDARIES[band] << self.log_blocks)
            ..(BAND_BOUNDARIES[band + 1] << self.log_blocks))
    }

    /// Cache the returned plan for the stream rather than rebuilding per packet.
    pub fn transform(&self, transient: bool) -> Result<MdctPlan, MediaDecodeError> {
        MdctPlan::new(self.samples() / self.blocks(transient)? * 2)
    }
}

/// Final CELT de-emphasis, with independent channel histories (RFC 6716 4.3.7.2).
#[derive(Clone)]
pub struct Deemphasis {
    history: Vec<f64>,
    scratch: Vec<f64>,
}

impl Deemphasis {
    pub fn new(channels: usize) -> Result<Self, MediaDecodeError> {
        if !(1..=255).contains(&channels) {
            return Err(MediaDecodeError::InvalidData(
                "invalid Opus de-emphasis channel count".into(),
            ));
        }
        Ok(Self {
            history: vec![0.0; channels],
            scratch: vec![0.0; channels],
        })
    }

    pub fn reset(&mut self) {
        self.history.fill(0.0);
    }

    pub fn process(&mut self, samples: &mut [f64]) -> Result<(), MediaDecodeError> {
        if samples.len() % self.history.len() != 0 || samples.iter().any(|value| !value.is_finite())
        {
            return Err(MediaDecodeError::InvalidData(
                "invalid Opus de-emphasis samples".into(),
            ));
        }
        // Validate the complete recurrence before committing either PCM or
        // history, so an overflowing malformed block cannot poison the stream.
        self.scratch.copy_from_slice(&self.history);
        for frame in samples.chunks_exact(self.history.len()) {
            for (&value, history) in frame.iter().zip(&mut self.scratch) {
                *history = value + (27853.0 / 32768.0) * *history;
                if !history.is_finite() {
                    return Err(MediaDecodeError::InvalidData(
                        "nonfinite Opus de-emphasis output".into(),
                    ));
                }
            }
        }
        for frame in samples.chunks_exact_mut(self.history.len()) {
            for (value, history) in frame.iter_mut().zip(&mut self.history) {
                *value += (27853.0 / 32768.0) * *history;
                *history = *value;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_packet_configurations_map_to_the_normative_spectral_layout() {
        for configuration in 0..32 {
            let data = [configuration << 3, 0];
            let packet = crate::audio::opus::Packet::parse(&data).unwrap();
            let layout = packet.celt_layout().unwrap();
            if configuration < 12 {
                assert!(layout.is_none());
                continue;
            }
            let layout = layout.unwrap();
            assert_eq!(layout.samples(), usize::from(packet.frame_samples_48khz));
            let bandwidth = match configuration {
                12 | 13 | 24..=27 => 12000,
                14 | 15 | 28..=31 => 20000,
                16..=19 => 4000,
                _ => 8000,
            };
            let mut previous = if configuration < 16 {
                layout.samples() / 3
            } else {
                0
            };
            for band in layout.bands() {
                let bins = layout.band(band).unwrap();
                assert_eq!(bins.start, previous);
                assert!(bins.end > bins.start);
                previous = bins.end;
            }
            assert_eq!(previous * 24000 / layout.samples(), bandwidth);
            assert!(layout.band(layout.end_band).is_err());
            assert!(layout.transform(false).is_ok());
            if layout.log_blocks == 0 {
                assert!(layout.transform(true).is_err());
            } else {
                assert!(layout.transform(true).is_ok());
            }
        }
        assert!(Layout::from_configuration(32).is_err());
    }

    #[test]
    fn deemphasis_is_chunk_independent_and_does_not_mix_channels() {
        let input: Vec<_> = (0..96)
            .map(|index| if index == 0 { 1.0 } else { 0.0 })
            .collect();
        let mut complete = input.clone();
        Deemphasis::new(2).unwrap().process(&mut complete).unwrap();
        for frames in 1..=48 {
            let mut filter = Deemphasis::new(2).unwrap();
            let mut chunked = input.clone();
            for chunk in chunked.chunks_mut(frames * 2) {
                filter.process(chunk).unwrap();
            }
            assert_eq!(complete, chunked);
            filter.reset();
            let mut zeros = [0.0; 2];
            filter.process(&mut zeros).unwrap();
            assert_eq!(zeros, [0.0; 2]);
        }
        for (index, frame) in complete.chunks_exact(2).enumerate() {
            assert!((frame[0] - (27853.0f64 / 32768.0).powi(index as i32)).abs() < 1e-12);
            assert_eq!(frame[1], 0.0);
        }
    }

    #[test]
    fn invalid_deemphasis_blocks_do_not_commit_partial_state() {
        let mut filter = Deemphasis::new(1).unwrap();
        let mut overflow = [f64::MAX, f64::MAX];
        assert!(filter.process(&mut overflow).is_err());
        assert_eq!(overflow, [f64::MAX, f64::MAX]);
        assert_eq!(filter.history, [0.0]);
        assert!(filter.process(&mut [f64::NAN]).is_err());
        assert!(Deemphasis::new(0).is_err());
        assert!(Deemphasis::new(2).unwrap().process(&mut [0.0; 3]).is_err());
    }
}
