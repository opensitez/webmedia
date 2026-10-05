//! One bounded MP4 reader feeding independently scheduled codec workers.

use super::mp4::{Mp4AudioIndex, Mp4Error, Mp4VideoIndex, Sample};

const MAX_BUFFER_BYTES: usize = 64 * 1024 * 1024;
const MAX_PACKETS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Track {
    Audio,
    Video,
}

#[derive(Debug)]
pub struct Packet {
    pub track: Track,
    pub sample_number: usize,
    pub presentation_time: i64,
    pub timescale: u32,
    pub keyframe: bool,
    pub data: Vec<u8>,
}

#[derive(Default)]
pub struct Mp4Demux {
    bytes: Vec<u8>,
    base_offset: u64,
    indexed: bool,
    audio: Option<Mp4AudioIndex>,
    video: Option<Mp4VideoIndex>,
    next_audio: usize,
    next_video: usize,
    start_time: f64,
}

impl Mp4Demux {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_start_time(seconds: f64) -> Result<Self, Mp4Error> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(Mp4Error::Invalid("invalid media seek time"));
        }
        Ok(Self {
            start_time: seconds,
            ..Self::default()
        })
    }

    pub fn audio_index(&self) -> Option<&Mp4AudioIndex> {
        self.audio.as_ref()
    }

    pub fn video_index(&self) -> Option<&Mp4VideoIndex> {
        self.video.as_ref()
    }

    fn next(&self) -> Option<(Track, usize, u32, &Sample)> {
        let audio = self.audio.as_ref().and_then(|index| {
            index
                .samples
                .get(self.next_audio)
                .map(|sample| (Track::Audio, self.next_audio, index.timescale, sample))
        });
        let video = self.video.as_ref().and_then(|index| {
            index
                .samples
                .get(self.next_video)
                .map(|sample| (Track::Video, self.next_video, index.timescale, sample))
        });
        match (audio, video) {
            (Some(a), Some(v)) => Some(if a.3.offset <= v.3.offset { a } else { v }),
            (a, v) => a.or(v),
        }
    }

    /// Empty pushes drain ready packets. No codec work or playback pacing occurs here.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<Packet>, Mp4Error> {
        if bytes.len() > MAX_BUFFER_BYTES.saturating_sub(self.bytes.len()) {
            return Err(Mp4Error::TooLarge);
        }
        self.bytes.extend_from_slice(bytes);
        if !self.indexed {
            let audio = match Mp4AudioIndex::parse_prefix(&self.bytes) {
                Err(Mp4Error::Incomplete) => return Ok(Vec::new()),
                result => result?,
            };
            let video = match Mp4VideoIndex::parse_prefix(&self.bytes) {
                Ok(index) => Some(index),
                Err(Mp4Error::Invalid("no video track")) => None,
                Err(error) => return Err(error),
            };
            if audio.is_none() && video.is_none() {
                return Err(Mp4Error::Invalid("no supported media tracks"));
            }
            self.audio = audio;
            self.video = video;
            if self.start_time > 0.0 {
                if let Some(index) = &self.audio {
                    self.next_audio = index
                        .samples
                        .partition_point(|sample| {
                            sample.presentation_time as f64 / f64::from(index.timescale)
                                < self.start_time
                        })
                        .saturating_sub(2);
                }
                if let Some(index) = &self.video {
                    self.next_video = index
                        .samples
                        .iter()
                        .rposition(|sample| {
                            sample.keyframe
                                && sample.presentation_time as f64 / f64::from(index.timescale)
                                    <= self.start_time
                        })
                        .unwrap_or(0);
                }
            }
            self.indexed = true;
        }
        let mut packets = Vec::new();
        while packets.len() < MAX_PACKETS {
            let Some((track, sample_number, timescale, sample)) = self.next() else {
                break;
            };
            let start = sample
                .offset
                .checked_sub(self.base_offset)
                .ok_or(Mp4Error::Invalid("sample precedes demux buffer"))?;
            let start = usize::try_from(start).map_err(|_| Mp4Error::TooLarge)?;
            let end = start
                .checked_add(sample.size as usize)
                .ok_or(Mp4Error::TooLarge)?;
            let Some(data) = self.bytes.get(start..end) else {
                break;
            };
            packets.push(Packet {
                track,
                sample_number,
                timescale,
                presentation_time: sample.presentation_time,
                keyframe: sample.keyframe,
                data: data.to_vec(),
            });
            match track {
                Track::Audio => self.next_audio += 1,
                Track::Video => self.next_video += 1,
            }
        }
        let retain = self
            .next()
            .map(|next| next.3.offset)
            .unwrap_or(self.base_offset + self.bytes.len() as u64);
        let discard = retain
            .saturating_sub(self.base_offset)
            .min(self.bytes.len() as u64) as usize;
        if discard >= 256 * 1024 || self.next().is_none() {
            self.bytes.drain(..discard);
            self.base_offset += discard as u64;
        }
        Ok(packets)
    }

    pub fn finish(&self) -> Result<(), Mp4Error> {
        if !self.indexed || self.next().is_some() {
            Err(Mp4Error::Incomplete)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incomplete_input_is_not_success() {
        let mut demux = Mp4Demux::new();
        assert!(
            demux
                .push(&[0, 0, 0, 8, b'f', b't', b'y', b'p'])
                .unwrap()
                .is_empty()
        );
        assert_eq!(demux.finish(), Err(Mp4Error::Incomplete));
    }

    #[test]
    fn empty_movie_is_not_successful_media() {
        assert_eq!(
            Mp4Demux::new()
                .push(&[0, 0, 0, 8, b'm', b'o', b'o', b'v'])
                .unwrap_err(),
            Mp4Error::Invalid("no supported media tracks")
        );
    }

    #[test]
    fn rejects_invalid_seek_times() {
        for time in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(Mp4Demux::with_start_time(time).is_err());
        }
    }

    #[test]
    #[ignore = "set WEBMEDIA_AAC_MP4 to goodtimes_h264.mp4"]
    fn seeking_retains_audio_overlap_and_video_keyframe() {
        let bytes = std::fs::read(std::env::var("WEBMEDIA_AAC_MP4").unwrap()).unwrap();
        let mut demux = Mp4Demux::with_start_time(60.0).unwrap();
        let mut first_audio = None;
        let mut first_video = None;
        for chunk in bytes.chunks(65521) {
            for packet in demux.push(chunk).unwrap() {
                match packet.track {
                    Track::Audio => {
                        first_audio.get_or_insert(packet);
                    }
                    Track::Video => {
                        first_video.get_or_insert(packet);
                    }
                }
            }
            if first_audio.is_some() && first_video.is_some() {
                break;
            }
        }
        let audio = first_audio.unwrap();
        let time = audio.presentation_time as f64 / f64::from(audio.timescale);
        assert!(time > 59.94 && time < 60.0);
        let video = first_video.unwrap();
        assert!(video.keyframe);
        assert!(video.presentation_time as f64 / f64::from(video.timescale) <= 60.0);
        assert!(video.sample_number > 0);
        assert!(demux.bytes.len() < 256 * 1024);
    }

    #[test]
    #[ignore = "set WEBMEDIA_AAC_MP4 to an audiovisual MP4 fixture"]
    fn interleaved_packets_match_both_tracks() {
        let bytes = std::fs::read(std::env::var("WEBMEDIA_AAC_MP4").unwrap()).unwrap();
        let audio = Mp4AudioIndex::parse_prefix(&bytes).unwrap().unwrap();
        let video = Mp4VideoIndex::parse_prefix(&bytes).unwrap();
        let mut demux = Mp4Demux::new();
        let mut counts = [0, 0];
        let mut previous_offset = 0;
        let mut consume = |packets: Vec<Packet>| {
            for packet in packets {
                let (slot, samples) = match packet.track {
                    Track::Audio => (0, &audio.samples),
                    Track::Video => (1, &video.samples),
                };
                assert_eq!(packet.sample_number, counts[slot]);
                let sample = &samples[packet.sample_number];
                assert!(sample.offset >= previous_offset);
                previous_offset = sample.offset;
                assert_eq!(packet.presentation_time, sample.presentation_time);
                assert_eq!(
                    packet.data,
                    bytes[sample.offset as usize..sample.offset as usize + sample.size as usize]
                );
                counts[slot] += 1;
            }
        };
        for chunk in bytes.chunks(65521) {
            consume(demux.push(chunk).unwrap());
            loop {
                let packets = demux.push(&[]).unwrap();
                if packets.is_empty() {
                    break;
                }
                consume(packets);
            }
            assert!(demux.bytes.len() < MAX_BUFFER_BYTES);
        }
        demux.finish().unwrap();
        assert_eq!(counts, [audio.samples.len(), video.samples.len()]);
        assert!(demux.bytes.is_empty());
    }

    #[test]
    #[ignore = "set WEBMEDIA_AAC_MP4 to goodtimes_h264.mp4"]
    fn shared_demux_decodes_complete_trimmed_audio() {
        use crate::audio::mp4::Mp4AacDecoder;

        let bytes = std::fs::read(std::env::var("WEBMEDIA_AAC_MP4").unwrap()).unwrap();
        let mut demux = Mp4Demux::new();
        let mut decoder = None;
        let mut frames = 0;
        let mut last = 0.0;
        let mut has_signal = false;
        for chunk in bytes.chunks(65521) {
            let mut packets = demux.push(chunk).unwrap();
            if decoder.is_none() {
                if let Some(index) = demux.audio_index() {
                    decoder = Some(Mp4AacDecoder::new(index).unwrap());
                }
            }
            loop {
                for packet in packets {
                    if packet.track != Track::Audio {
                        continue;
                    }
                    let Some(decoded) = decoder
                        .as_mut()
                        .unwrap()
                        .decode(packet.presentation_time, &packet.data)
                        .unwrap()
                    else {
                        continue;
                    };
                    assert!((decoded.timestamp - last).abs() < 1e-9);
                    assert!(decoded.samples.samples.iter().all(|s| s.is_finite()));
                    has_signal |= decoded.samples.samples.iter().any(|s| s.abs() > 1e-6);
                    let count =
                        decoded.samples.samples.len() / usize::from(decoded.samples.channels);
                    frames += count;
                    last =
                        decoded.timestamp + count as f64 / f64::from(decoded.samples.sample_rate);
                }
                packets = demux.push(&[]).unwrap();
                if packets.is_empty() {
                    break;
                }
            }
        }
        demux.finish().unwrap();
        assert!(has_signal);
        assert_eq!(frames, 6_325_704);
        assert!((last - 143.44).abs() < 1e-9);
    }
}
