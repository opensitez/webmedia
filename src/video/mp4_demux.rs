//! One bounded MP4 reader feeding independently scheduled codec workers.

use super::mp4::{
    Mp4AudioIndex, Mp4Error, Mp4Fragment, Mp4FragmentDecodeTimes, Mp4FragmentInit, Mp4VideoIndex,
    Sample,
};

const MAX_BUFFER_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_MOVIE_BYTES: usize = 32 * 1024 * 1024;
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
    pub sample: Sample,
    pub fragmented: bool,
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
    fragments: Option<Mp4FragmentInit>,
    fragment_times: Mp4FragmentDecodeTimes,
    scan_offset: u64,
    open_ended_mdat: bool,
    data_ranges: Vec<(u64, u64)>,
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

    /// Seed classic MP4 from a separately fetched movie box. Sample offsets remain
    /// absolute file offsets; the caller starts input at `input_offset()`.
    /// Fragmented MP4 returns None and must use the ordinary incremental path.
    pub fn from_movie_box(movie: &[u8], file_size: u64, seconds: f64)
        -> Result<Option<Self>, Mp4Error>
    {
        if movie.len() > MAX_MOVIE_BYTES { return Err(Mp4Error::TooLarge); }
        let mut demux = Self::with_start_time(seconds)?;
        match Mp4FragmentInit::parse_prefix(movie) {
            Ok(_) => return Ok(None),
            Err(Mp4Error::Unsupported("not fragmented MP4")) => {},
            Err(error) => return Err(error),
        }
        demux.audio = Mp4AudioIndex::parse_prefix(movie)?;
        demux.video = match Mp4VideoIndex::parse_prefix(movie) {
            Ok(index) => Some(index),
            Err(Mp4Error::Invalid("no video track")) => None,
            Err(error) => return Err(error),
        };
        if demux.audio.is_none() && demux.video.is_none() {
            return Err(Mp4Error::Invalid("no supported media tracks"));
        }
        for (timescale, samples) in demux.audio.iter().map(|index| (index.timescale, &index.samples))
            .chain(demux.video.iter().map(|index| (index.timescale, &index.samples)))
        {
            if timescale == 0 { return Err(Mp4Error::Invalid("zero media timescale")); }
            let mut previous_end = 0;
            for sample in samples {
                let end = sample.offset.checked_add(u64::from(sample.size)).ok_or(Mp4Error::TooLarge)?;
                if sample.size as usize > MAX_BUFFER_BYTES { return Err(Mp4Error::TooLarge); }
                if sample.offset < previous_end || end > file_size {
                    return Err(Mp4Error::Invalid("sample outside file or nonmonotonic offsets"));
                }
                previous_end = end;
            }
        }
        demux.select_start();
        demux.indexed = true;
        demux.base_offset = demux.next().map_or(file_size, |next| next.3.offset);
        Ok(Some(demux))
    }

    /// Absolute offset at which the next push must begin.
    pub fn input_offset(&self) -> u64 { self.base_offset + self.bytes.len() as u64 }

    /// End of the final indexed sample, excluding a trailing movie box.
    pub fn media_end(&self) -> Option<u64> {
        self.audio.iter().flat_map(|index| &index.samples)
            .chain(self.video.iter().flat_map(|index| &index.samples))
            .filter_map(|sample| sample.offset.checked_add(u64::from(sample.size))).max()
    }

    fn select_start(&mut self) {
        if self.start_time <= 0.0 { return; }
        if let Some(index) = &self.audio {
            self.next_audio = index.samples.partition_point(|sample|
                sample.presentation_time as f64 / f64::from(index.timescale) < self.start_time)
                .saturating_sub(2);
        }
        if let Some(index) = &self.video {
            self.next_video = index.samples.iter().rposition(|sample|
                sample.keyframe && sample.presentation_time as f64 / f64::from(index.timescale)
                    <= self.start_time).unwrap_or(0);
        }
    }

    pub fn audio_index(&self) -> Option<&Mp4AudioIndex> {
        self.audio.as_ref()
    }

    pub fn video_index(&self) -> Option<&Mp4VideoIndex> {
        self.video.as_ref()
    }

    pub fn is_fragmented(&self) -> bool {
        self.fragments.is_some()
    }

    fn scan_fragments(&mut self) -> Result<(), Mp4Error> {
        if self.fragments.is_none() || self.open_ended_mdat {
            return Ok(());
        }
        let available_end = self.base_offset + self.bytes.len() as u64;
        while self.scan_offset < available_end {
            let at = usize::try_from(
                self.scan_offset
                    .checked_sub(self.base_offset)
                    .ok_or(Mp4Error::Invalid("discarded fragment header"))?,
            )
            .map_err(|_| Mp4Error::TooLarge)?;
            let Some(header) = self.bytes.get(at..at.saturating_add(8)) else {
                break;
            };
            let short = u32::from_be_bytes(header[..4].try_into().unwrap());
            let kind: [u8; 4] = header[4..8].try_into().unwrap();
            let (size, header_len) = if short == 1 {
                let Some(extended) = self.bytes.get(at + 8..at + 16) else {
                    break;
                };
                (u64::from_be_bytes(extended.try_into().unwrap()), 16u64)
            } else if short == 0 {
                if kind != *b"mdat" {
                    return Err(Mp4Error::Unsupported("open-ended non-media box"));
                }
                self.data_ranges.push((self.scan_offset + 8, u64::MAX));
                self.open_ended_mdat = true;
                self.scan_offset = u64::MAX;
                break;
            } else {
                (u64::from(short), 8u64)
            };
            if size < header_len {
                return Err(Mp4Error::Invalid("invalid box size"));
            }
            let end = self
                .scan_offset
                .checked_add(size)
                .ok_or(Mp4Error::TooLarge)?;
            if kind == *b"moof" {
                if end > available_end {
                    break;
                }
                let fragment = Mp4Fragment::parse_prefix(
                    &self.bytes[at..],
                    self.scan_offset,
                    self.fragments.as_ref().unwrap(),
                    self.fragment_times,
                )?;
                for (index, samples) in [
                    (
                        self.video.as_mut().map(|index| &mut index.samples),
                        &fragment.video_samples,
                    ),
                    (
                        self.audio.as_mut().map(|index| &mut index.samples),
                        &fragment.audio_samples,
                    ),
                ] {
                    if let Some(index) = index {
                        if samples.len() > 1_000_000usize.saturating_sub(index.len()) {
                            return Err(Mp4Error::TooLarge);
                        }
                        if samples
                            .iter()
                            .any(|sample| sample.offset < self.base_offset)
                        {
                            return Err(Mp4Error::Invalid(
                                "fragment sample precedes retained input",
                            ));
                        }
                        index.extend_from_slice(samples);
                    }
                }
                self.fragment_times = fragment.decode_times;
                if let Some(index) = &mut self.video {
                    index.duration_ticks = index.duration_ticks.max(self.fragment_times.video);
                }
                if let Some(index) = &mut self.audio {
                    index.duration_ticks = index.duration_ticks.max(self.fragment_times.audio);
                }
            } else if kind == *b"mdat" {
                self.data_ranges.push((self.scan_offset + header_len, end));
            }
            self.scan_offset = end;
        }
        Ok(())
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
            self.fragments = match Mp4FragmentInit::parse_prefix(&self.bytes) {
                Ok(init) => Some(init),
                Err(Mp4Error::Unsupported("not fragmented MP4")) => None,
                Err(Mp4Error::Incomplete) => return Ok(Vec::new()),
                Err(error) => return Err(error),
            };
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
            self.select_start();
            self.indexed = true;
        }
        self.scan_fragments()?;
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
            if self.fragments.is_some() {
                let sample_end = sample
                    .offset
                    .checked_add(u64::from(sample.size))
                    .ok_or(Mp4Error::TooLarge)?;
                if !self
                    .data_ranges
                    .iter()
                    .any(|&(begin, end)| sample.offset >= begin && sample_end <= end)
                {
                    if sample.offset >= self.scan_offset {
                        break;
                    }
                    return Err(Mp4Error::Invalid("fragment sample outside media data"));
                }
            }
            let Some(data) = self.bytes.get(start..end) else {
                break;
            };
            packets.push(Packet {
                track,
                sample_number,
                timescale,
                presentation_time: sample.presentation_time,
                keyframe: sample.keyframe,
                sample: sample.clone(),
                fragmented: self.fragments.is_some(),
                data: data.to_vec(),
            });
            match track {
                Track::Audio => self.next_audio += 1,
                Track::Video => self.next_video += 1,
            }
        }
        let mut retain = self
            .next()
            .map(|next| next.3.offset)
            .unwrap_or(self.base_offset + self.bytes.len() as u64);
        if self.fragments.is_some() {
            retain = retain.min(self.scan_offset);
        }
        let discard = retain
            .saturating_sub(self.base_offset)
            .min(self.bytes.len() as u64) as usize;
        if discard >= 256 * 1024 || self.next().is_none() {
            self.bytes.drain(..discard);
            self.base_offset += discard as u64;
            self.data_ranges.retain(|&(_, end)| end > self.base_offset);
        }
        Ok(packets)
    }

    pub fn finish(&self) -> Result<(), Mp4Error> {
        if !self.indexed
            || self.next().is_some()
            || (self.fragments.is_some()
                && !self.open_ended_mdat
                && self.scan_offset != self.base_offset + self.bytes.len() as u64)
        {
            Err(Mp4Error::Incomplete)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classic_movie(offset: u64) -> Vec<u8> {
        fn atom(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
            let mut bytes = ((data.len() + 8) as u32).to_be_bytes().to_vec();
            bytes.extend(kind);
            bytes.extend(data);
            bytes
        }
        fn words(values: &[u32]) -> Vec<u8> {
            values.iter().flat_map(|value| value.to_be_bytes()).collect()
        }
        let mut entry = vec![0; 78];
        entry[24..26].copy_from_slice(&2u16.to_be_bytes());
        entry[26..28].copy_from_slice(&2u16.to_be_bytes());
        let mut description = words(&[0, 1]);
        description.extend(atom(b"vp08", &entry));
        let mut offsets = words(&[0, 1]);
        offsets.extend(offset.to_be_bytes());
        let mut tables = atom(b"stsd", &description);
        for (kind, data) in [(b"stsz", words(&[0, 0, 2, 6, 6])),
            (b"stsc", words(&[0, 1, 1, 2, 1])), (b"stts", words(&[0, 1, 2, 100]))] {
            tables.extend(atom(kind, &data));
        }
        tables.extend(atom(b"co64", &offsets));
        let mut media = atom(b"mdhd", &words(&[0, 0, 0, 1000]));
        let mut handler = words(&[0, 0]);
        handler.extend(b"vide");
        media.extend(atom(b"hdlr", &handler));
        media.extend(atom(b"minf", &atom(b"stbl", &tables)));
        atom(b"moov", &atom(b"trak", &atom(b"mdia", &media)))
    }

    #[test]
    fn separately_fetched_trailing_movie_keeps_absolute_offsets_and_bounded_input() {
        let sample_offset = 99 * 1024 * 1024u64;
        let movie = classic_movie(sample_offset);
        let mut demux = Mp4Demux::from_movie_box(&movie, sample_offset + 12 + movie.len() as u64, 0.0)
            .unwrap().unwrap();
        assert_eq!(demux.input_offset(), sample_offset);
        assert_eq!(demux.media_end(), Some(sample_offset + 12));
        let mut packets = Vec::new();
        for chunk in [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].chunks(2) {
            packets.extend(demux.push(chunk).unwrap());
            assert!(demux.bytes.len() <= 12);
        }
        demux.finish().unwrap();
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].data, [1, 2, 3, 4, 5, 6]);
        assert_eq!(packets[1].data, [7, 8, 9, 10, 11, 12]);
        assert_eq!(packets[1].sample.offset, sample_offset + 6);
        assert_eq!(packets[1].sample_number, 1);
    }

    #[test]
    fn seeded_seek_starts_at_selected_keyframe_and_rejects_out_of_file_samples() {
        let movie = classic_movie(100_000_000);
        let mut demux = Mp4Demux::from_movie_box(&movie, 100_000_012, 0.1).unwrap().unwrap();
        assert_eq!(demux.input_offset(), 100_000_006);
        let packets = demux.push(&[7, 8, 9, 10, 11, 12]).unwrap();
        assert_eq!(packets.len(), 1);
        assert_eq!(packets[0].sample_number, 1);
        assert_eq!(packets[0].presentation_time, 100);
        demux.finish().unwrap();
        assert!(matches!(Mp4Demux::from_movie_box(&movie, 100_000_011, 0.0),
            Err(Mp4Error::Invalid("sample outside file or nonmonotonic offsets"))));
        assert!(Mp4Demux::from_movie_box(&movie, u64::MAX, f64::NAN).is_err());
        assert!(Mp4Demux::from_movie_box(&classic_movie(u64::MAX - 5), u64::MAX, 0.0).is_err());
    }

    #[test]
    fn fragmented_movie_seed_keeps_incremental_fragment_path() {
        let bytes = super::super::mp4::fragmented_test_file();
        assert!(Mp4Demux::from_movie_box(&bytes, bytes.len() as u64, 0.0).unwrap().is_none());
    }

    #[test]
    fn public_movie_seed_rejects_oversized_input_before_parsing() {
        let bytes = vec![0; MAX_MOVIE_BYTES + 1];
        assert!(matches!(Mp4Demux::from_movie_box(&bytes, u64::MAX, 0.0), Err(Mp4Error::TooLarge)));
    }

    fn collect_fragments(bytes: &[u8], chunk_size: usize) -> (Mp4Demux, Vec<Packet>) {
        let mut demux = Mp4Demux::new();
        let mut packets = Vec::new();
        for chunk in bytes.chunks(chunk_size) {
            packets.extend(demux.push(chunk).unwrap());
            loop {
                let ready = demux.push(&[]).unwrap();
                if ready.is_empty() {
                    break;
                }
                packets.extend(ready);
            }
        }
        (demux, packets)
    }

    #[test]
    fn fragmented_demux_streams_both_tracks_without_duplicate_commits() {
        let bytes = super::super::mp4::fragmented_test_file();
        for size in [1, 2, 7, 16, 63, 257, bytes.len()] {
            let (mut demux, packets) = collect_fragments(&bytes, size);
            demux.finish().unwrap();
            assert!(demux.is_fragmented());
            assert_eq!(packets.len(), 12, "chunk size {size}");
            let mut counts = [0; 2];
            let mut offset = 0;
            for packet in packets {
                let slot = usize::from(packet.track == Track::Video);
                assert_eq!(packet.sample_number, counts[slot]);
                counts[slot] += 1;
                assert!(packet.fragmented);
                assert!(packet.sample.offset >= offset);
                offset = packet.sample.offset;
                assert_eq!(packet.presentation_time, packet.sample.presentation_time);
                assert_eq!(
                    packet.data,
                    bytes[offset as usize..offset as usize + packet.sample.size as usize]
                );
            }
            assert_eq!(counts, [6, 6]);
            assert_eq!(demux.video_index().unwrap().duration_ticks, 60);
            assert_eq!(demux.audio_index().unwrap().duration_ticks, 6144);
            assert!(demux.bytes.is_empty());
            assert!(demux.push(&[]).unwrap().is_empty());
        }
    }

    #[test]
    fn fragmented_demux_rejects_truncated_headers_and_payloads() {
        let bytes = super::super::mp4::fragmented_test_file();
        for cut in 1..=7 {
            let (demux, _) = collect_fragments(&bytes[..bytes.len() - cut], 13);
            assert_eq!(demux.finish(), Err(Mp4Error::Incomplete));
        }
        for partial in 1..8 {
            let mut demux = Mp4Demux::new();
            demux.push(&bytes).unwrap();
            demux.push(&[0; 7][..partial]).unwrap();
            assert_eq!(demux.finish(), Err(Mp4Error::Incomplete));
        }
    }

    #[test]
    fn fragmented_demux_requires_samples_to_belong_to_media_payload() {
        let mut bytes = super::super::mp4::fragmented_test_file();
        let at = bytes.windows(4).position(|word| word == b"trun").unwrap();
        // Data offset points at the fragment header rather than its mdat payload.
        bytes[at + 12..at + 16].copy_from_slice(&0u32.to_be_bytes());
        assert_eq!(
            Mp4Demux::new().push(&bytes).unwrap_err(),
            Mp4Error::Invalid("fragment sample outside media data")
        );
    }

    #[test]
    fn fragmented_demux_accepts_final_open_ended_mdat() {
        let mut bytes = super::super::mp4::fragmented_test_file();
        let at = bytes.windows(4).rposition(|word| word == b"mdat").unwrap();
        bytes[at - 4..at].fill(0);
        let (demux, packets) = collect_fragments(&bytes, 11);
        assert_eq!(packets.len(), 12);
        demux.finish().unwrap();
    }

    #[test]
    #[ignore = "set WEBMEDIA_FRAGMENTED_MP4 to the downloaded CNN hero"]
    fn fragmented_cnn_demux_streams_all_audio_and_video_samples() {
        let bytes = std::fs::read(std::env::var("WEBMEDIA_FRAGMENTED_MP4").unwrap()).unwrap();
        let (demux, packets) = collect_fragments(&bytes, 4093);
        demux.finish().unwrap();
        let audio = packets.iter().filter(|p| p.track == Track::Audio).count();
        let video = packets.iter().filter(|p| p.track == Track::Video).count();
        // Independently counted with ffprobe on this fixed CNN clip.
        assert_eq!((video, audio), (240, 175));
        assert!(packets.iter().all(|p| p.fragmented && !p.data.is_empty()));
        assert!(demux.bytes.is_empty());
        eprintln!("CNN demux: {video} video and {audio} audio packets");
    }

    #[test]
    #[ignore = "set WEBMEDIA_FRAGMENTED_MP4 to the downloaded CNN hero"]
    fn fragmented_cnn_decodes_video_and_aac_through_existing_workers() {
        use super::super::mp4_video::Mp4VideoPackets;
        use crate::audio::mp4::Mp4AacDecoder;
        let bytes = std::fs::read(std::env::var("WEBMEDIA_FRAGMENTED_MP4").unwrap()).unwrap();
        let (demux, packets) = collect_fragments(&bytes, 4093);
        let mut metadata = demux.video_index().unwrap().clone();
        metadata.samples.clear();
        let mut video = Mp4VideoPackets::new_fragmented(metadata, 0).unwrap();
        let mut audio = Mp4AacDecoder::new(demux.audio_index().unwrap()).unwrap();
        let mut decoded = 0;
        let mut last_time = -1.0;
        let mut signal = false;
        let mut audio_frames = 0;
        for packet in packets {
            match packet.track {
                Track::Video if packet.sample_number < 18 => {
                    for frame in video
                        .push_sample(packet.sample_number, packet.sample, &packet.data)
                        .unwrap()
                    {
                        assert_eq!((frame.width, frame.height), (1440, 1080));
                        assert!(frame.timestamp >= last_time);
                        last_time = frame.timestamp;
                        decoded += 1;
                    }
                }
                Track::Audio if packet.sample_number < 32 => {
                    if let Some(frame) = audio
                        .decode(packet.presentation_time, &packet.data)
                        .unwrap()
                    {
                        audio_frames += frame.samples.samples.len();
                        signal |= frame
                            .samples
                            .samples
                            .iter()
                            .any(|sample| sample.abs() > 0.0001);
                    }
                }
                _ => {}
            }
        }
        assert!(
            decoded > 0,
            "fragment reorder must not hold the entire movie"
        );
        // The CNN loop's AAC track is digital silence (ffmpeg astats: -inf).
        assert!(!signal && audio_frames > 1024);
        eprintln!("CNN decoded {decoded} early video frames and {audio_frames} PCM samples");
    }

    #[test]
    #[ignore = "set WEBMEDIA_FRAGMENTED_LOCAL_MP4 to goodtimes_h264_fragmented.mp4"]
    fn fragmented_local_video_and_audible_audio_decode_to_eof() {
        use super::super::mp4_video::Mp4VideoPackets;
        use crate::audio::mp4::Mp4AacDecoder;
        let bytes = std::fs::read(std::env::var("WEBMEDIA_FRAGMENTED_LOCAL_MP4").unwrap()).unwrap();
        let (demux, packets) = collect_fragments(&bytes, 4093);
        demux.finish().unwrap();
        let expected = demux.video_index().unwrap().samples.len();
        let mut metadata = demux.video_index().unwrap().clone();
        metadata.samples.clear();
        let mut video = Mp4VideoPackets::new_fragmented(metadata, 0).unwrap();
        let mut audio_metadata = demux.audio_index().unwrap().clone();
        audio_metadata.duration_ticks = 0;
        let mut audio = Mp4AacDecoder::new(&audio_metadata).unwrap();
        let mut times = Vec::new();
        let mut pcm = 0;
        let mut signal = false;
        let start = std::time::Instant::now();
        for packet in packets {
            match packet.track {
                Track::Video => {
                    times.extend(
                        video
                            .push_sample(packet.sample_number, packet.sample, &packet.data)
                            .unwrap()
                            .into_iter()
                            .map(|frame| frame.timestamp),
                    );
                }
                Track::Audio => {
                    if let Some(frame) = audio
                        .decode(packet.presentation_time, &packet.data)
                        .unwrap()
                    {
                        pcm += frame.samples.samples.len();
                        signal |= frame
                            .samples
                            .samples
                            .iter()
                            .any(|sample| sample.abs() > 0.001);
                    }
                }
            }
        }
        loop {
            let tail = video.finish_input().unwrap();
            if tail.is_empty() {
                break;
            }
            times.extend(tail.into_iter().map(|frame| frame.timestamp));
        }
        assert_eq!(
            times.len(),
            expected,
            "every decoded picture must reach presentation"
        );
        assert!(times.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(signal && pcm > 44100 * 20);
        assert!(*times.last().unwrap() > 11.0);
        eprintln!(
            "Local fragmented decode: {} frames, {pcm} audible PCM samples in {:.2}s",
            times.len(),
            start.elapsed().as_secs_f64()
        );
    }

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
