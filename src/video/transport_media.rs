//! Stateful codec adapters for complete MPEG-TS PES elementary payloads.
//! Timestamps must already share the parent's unwrapped, nonnegative 90-kHz epoch.

use super::backend::{MediaDecodeError, MediaMetadata, MediaSample, VideoFrame};
use super::h264::{AvcConfig, AvcError, SequenceParameters, parse_pps_2005, parse_sps};
use super::mp4::{Mp4Index, Sample};
use super::mp4_avc::Mp4AvcPackets;
use super::mpeg_ts::{ElementaryPacket, StreamKind, MAX_PES_BYTES, TIMESTAMP_TIMESCALE};
use crate::audio::aac::{AacConfig, AacDecoder};
use std::collections::VecDeque;

const MAX_NALS: usize = 1024;
const MAX_AUDIO_FRAMES: usize = 256;
const MAX_ADTS_BYTES: usize = MAX_PES_BYTES + 8191;
const AAC_RATES: [u32; 13] = [96000, 88200, 64000, 48000, 44100, 32000, 24000,
    22050, 16000, 12000, 11025, 8000, 7350];

fn invalid(reason: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(reason.into())
}

fn avc_error(error: AvcError) -> MediaDecodeError {
    match error {
        AvcError::Unsupported(_) | AvcError::UnsupportedProfile(_) => MediaDecodeError::Unsupported,
        other => invalid(&format!("transport AVC: {other:?}")),
    }
}

fn packet_track(pid: &mut Option<u16>, packet: &ElementaryPacket) -> Result<(), MediaDecodeError> {
    if packet.pid > 0x1fff || packet.data.len() > MAX_PES_BYTES {
        return Err(invalid("transport packet bound exceeded"));
    }
    if pid.is_some_and(|pid| pid != packet.pid) { return Err(MediaDecodeError::Unsupported); }
    *pid = Some(packet.pid);
    Ok(())
}

#[derive(Default)]
pub struct TransportVideoDecoder {
    pid: Option<u16>,
    sps_bytes: Option<Vec<u8>>,
    sps: Option<SequenceParameters>,
    pps: Option<Vec<u8>>,
    decoder: Option<Mp4AvcPackets>,
    next_sample: usize,
    offset: u64,
    finished: bool,
    error: Option<MediaDecodeError>,
}

impl TransportVideoDecoder {
    pub fn new() -> Self { Self::default() }

    pub fn push(&mut self, packet: &ElementaryPacket) -> Result<Vec<VideoFrame>, MediaDecodeError> {
        if packet.kind != StreamKind::H264 { return Err(MediaDecodeError::Unsupported); }
        if packet.discontinuity { *self = Self::new(); }
        if let Some(error) = &self.error { return Err(error.clone()); }
        let result = self.push_inner(packet);
        if let Err(error) = &result { self.error = Some(error.clone()); }
        result
    }

    fn push_inner(&mut self, packet: &ElementaryPacket) -> Result<Vec<VideoFrame>, MediaDecodeError> {
        if self.finished { return Err(invalid("video PES after EOF")); }
        packet_track(&mut self.pid, packet)?;
        if packet.data.is_empty() { return Ok(Vec::new()); }
        let units = annex_b_units(&packet.data)?;
        let pictures: Vec<_> = units.iter().filter(|unit| unit.iter()
            .any(|nal| matches!(nal[0] & 31, 1 | 5))).collect();
        // AUD establishes boundaries, not timestamps for later pictures. Without
        // an exposed VUI/per-AU clock, multiple pictures cannot be timed safely.
        if pictures.len() > 1 { return Err(MediaDecodeError::Unsupported); }
        for unit in &units {
            for &nal in unit {
                match nal[0] & 31 {
                    7 => {
                        if self.decoder.is_some() && self.sps_bytes.as_deref() != Some(nal) {
                            return Err(MediaDecodeError::Unsupported);
                        }
                        self.sps = Some(parse_sps(nal).map_err(avc_error)?);
                        self.sps_bytes = Some(nal.to_vec());
                    }
                    8 => {
                        if self.decoder.is_some() && self.pps.as_deref() != Some(nal) {
                            return Err(MediaDecodeError::Unsupported);
                        }
                        parse_pps_2005(nal).map_err(avc_error)?;
                        self.pps = Some(nal.to_vec());
                    }
                    _ => {}
                }
            }
        }
        let Some(unit) = pictures.first() else { return Ok(Vec::new()); };
        let slice = unit.iter().find(|nal| matches!(nal[0] & 31, 1 | 5)).unwrap();
        let (first_mb, _, pps_id) = slice_prefix(slice)?;
        if first_mb != 0 {
            return Err(MediaDecodeError::Unsupported);
        }
        let sps = self.sps.as_ref().ok_or_else(|| invalid("AVC picture before SPS"))?;
        let pps = self.pps.as_ref().ok_or_else(|| invalid("AVC picture before PPS"))?;
        let parsed_pps = parse_pps_2005(pps).map_err(avc_error)?.core;
        if parsed_pps.id != pps_id || parsed_pps.sequence_id != sps.id {
            return Err(MediaDecodeError::Unsupported);
        }
        let keyframe = slice[0] & 31 == 5;
        let pts = packet.pts.ok_or_else(|| invalid("AVC picture without PTS"))?;
        let presentation_time = i64::try_from(pts).map_err(|_| invalid("AVC PTS overflow"))?;
        // PES may omit DTS when it equals PTS, including a reordered B picture.
        let decode_time = packet.dts.unwrap_or(pts);
        let mut data = Vec::new();
        for &nal in *unit {
            let length = u32::try_from(nal.len()).map_err(|_| invalid("AVC NAL too large"))?;
            if data.len().checked_add(nal.len() + 4).is_none_or(|size| size > MAX_PES_BYTES) {
                return Err(invalid("AVC access unit too large"));
            }
            data.extend_from_slice(&length.to_be_bytes());
            data.extend_from_slice(nal);
        }
        if self.decoder.is_none() {
            if !keyframe { return Err(MediaDecodeError::Unsupported); }
            self.decoder = Some(Mp4AvcPackets::new_fragmented(Mp4Index {
                timescale: TIMESTAMP_TIMESCALE, duration_ticks: 0,
                config: AvcConfig { nal_length_size: 4,
                    sequence_parameters: vec![sps.clone()], picture_parameter_sets: vec![pps.clone()] },
                samples: Vec::new(),
            }, 0)?);
        }
        let size = u32::try_from(data.len()).map_err(|_| invalid("AVC access unit too large"))?;
        let next_offset = self.offset.checked_add(u64::from(size)).ok_or_else(|| invalid("AVC offset overflow"))?;
        let frames = self.decoder.as_mut().unwrap().push_sample(self.next_sample, Sample {
            offset: self.offset, size, decode_time, presentation_time, keyframe,
        }, &data)?;
        self.next_sample = self.next_sample.checked_add(1).ok_or_else(|| invalid("AVC sample count overflow"))?;
        self.offset = next_offset;
        Ok(frames)
    }

    /// Definitive EOF only; ordinary HLS segment boundaries do not drain/reset.
    pub fn finish(&mut self) -> Result<Vec<VideoFrame>, MediaDecodeError> {
        if let Some(error) = &self.error { return Err(error.clone()); }
        if self.finished { return Ok(Vec::new()); }
        let mut frames = Vec::new();
        if let Some(decoder) = &mut self.decoder {
            loop {
                let tail = decoder.finish_input()?;
                if tail.is_empty() { break; }
                frames.extend(tail);
            }
        }
        self.finished = true;
        Ok(frames)
    }

    pub fn metadata(&self) -> Option<MediaMetadata> {
        let sps = self.sps.as_ref()?;
        Some(MediaMetadata { presentation_size: None, duration: None,
            width: Some(sps.width), height: Some(sps.height), sample_rate: None, channels: None })
    }
}

fn start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    for at in from..data.len().saturating_sub(2) {
        if data[at..].starts_with(&[0, 0, 0, 1]) { return Some((at, 4)); }
        if data[at..].starts_with(&[0, 0, 1]) { return Some((at, 3)); }
    }
    None
}

/// A bounded catch-up point carrying its own configuration, not an arbitrary
/// predicted picture. The normal shared decoder validates all syntax afterward.
pub fn independent_h264_packet(packet: &ElementaryPacket) -> Result<bool, MediaDecodeError> {
    if packet.kind != StreamKind::H264 || packet.data.len() > MAX_PES_BYTES { return Ok(false); }
    let units = annex_b_units(&packet.data)?;
    let mut sps = false;
    let mut pps = false;
    let mut idr = false;
    for nal in units.iter().flatten() {
        match nal[0] & 31 {
            7 => sps = true,
            8 => pps = true,
            5 => idr = true,
            1 => return Ok(false),
            _ => {},
        }
    }
    Ok(sps && pps && idr)
}

fn annex_b_units(data: &[u8]) -> Result<Vec<Vec<&[u8]>>, MediaDecodeError> {
    let (first, prefix) = start_code(data, 0).ok_or_else(|| invalid("missing Annex B start code"))?;
    if data[..first].iter().any(|&byte| byte != 0) { return Err(invalid("garbage before Annex B NAL")); }
    let mut from = first + prefix;
    let mut units = Vec::new();
    let mut unit = Vec::new();
    let mut vcl = false;
    let mut count = 0;
    loop {
        let next = start_code(data, from);
        let mut end = next.map_or(data.len(), |(at, _)| at);
        while end > from && data[end - 1] == 0 { end -= 1; }
        let nal = &data[from..end];
        if nal.is_empty() || nal[0] & 0x80 != 0 { return Err(invalid("invalid Annex B NAL header")); }
        count += 1;
        if count > MAX_NALS { return Err(invalid("too many Annex B NALs")); }
        match nal[0] & 31 {
            9 => {
                if nal.len() != 2 || nal[0] & 0x60 != 0 || nal[1] & 31 != 16 {
                    return Err(invalid("invalid access unit delimiter"));
                }
                if vcl { units.push(std::mem::take(&mut unit)); vcl = false; }
            }
            1 | 5 => {
                // The shared decoder accepts one complete slice per picture.
                if vcl { return Err(MediaDecodeError::Unsupported); }
                if slice_prefix(nal)?.0 != 0 { return Err(MediaDecodeError::Unsupported); }
                vcl = true;
            }
            2..=4 | 13..=31 | 0 => return Err(MediaDecodeError::Unsupported),
            6..=8 if vcl => return Err(MediaDecodeError::Unsupported),
            _ => {}
        }
        unit.push(nal);
        let Some((at, length)) = next else { break; };
        from = at + length;
    }
    if !unit.is_empty() { units.push(unit); }
    Ok(units)
}

// Only framing/parameter-set identity is inspected here; picture decoding stays
// in Mp4AvcPackets. Three bounded Exp-Golomb fields precede the full slice header.
fn slice_prefix(nal: &[u8]) -> Result<(u32, u32, u32), MediaDecodeError> {
    let mut prefix = Vec::new();
    let mut zeros = 0;
    for &byte in nal.get(1..).ok_or_else(|| invalid("missing slice header"))? {
        if zeros == 2 && byte == 3 { zeros = 0; continue; }
        prefix.push(byte);
        zeros = if byte == 0 { zeros + 1 } else { 0 };
        if prefix.len() == 32 { break; }
    }
    let mut bit = 0;
    let mut ue = || -> Result<u32, MediaDecodeError> {
        let mut zeros = 0;
        loop {
            let byte = prefix.get(bit / 8).ok_or_else(|| invalid("truncated slice prefix"))?;
            let value = byte >> (7 - bit % 8) & 1;
            bit += 1;
            if value != 0 { break; }
            zeros += 1;
            if zeros > 31 { return Err(invalid("excessive slice prefix")); }
        }
        let mut value = 1u32;
        for _ in 0..zeros {
            let byte = prefix.get(bit / 8).ok_or_else(|| invalid("truncated slice prefix"))?;
            value = (value << 1) | u32::from(byte >> (7 - bit % 8) & 1);
            bit += 1;
        }
        Ok(value - 1)
    };
    Ok((ue()?, ue()?, ue()?))
}

struct AudioClock {
    pts: u64,
    samples: u64,
}

#[derive(Default)]
pub struct TransportAudioDecoder {
    pid: Option<u16>,
    config: Option<AacConfig>,
    decoder: Option<AacDecoder>,
    pending: Vec<u8>,
    anchors: VecDeque<(usize, u64)>,
    clock: Option<AudioClock>,
    finished: bool,
    error: Option<MediaDecodeError>,
}

impl TransportAudioDecoder {
    pub fn new() -> Self { Self::default() }

    pub fn push(&mut self, packet: &ElementaryPacket) -> Result<Vec<MediaSample>, MediaDecodeError> {
        if packet.kind != StreamKind::Aac { return Err(MediaDecodeError::Unsupported); }
        if packet.discontinuity { *self = Self::new(); }
        if let Some(error) = &self.error { return Err(error.clone()); }
        let result = self.push_inner(packet);
        if let Err(error) = &result { self.error = Some(error.clone()); }
        result
    }

    fn push_inner(&mut self, packet: &ElementaryPacket) -> Result<Vec<MediaSample>, MediaDecodeError> {
        if self.finished { return Err(invalid("audio PES after EOF")); }
        packet_track(&mut self.pid, packet)?;
        if self.pending.len().checked_add(packet.data.len()).is_none_or(|size| size > MAX_ADTS_BYTES) {
            return Err(invalid("ADTS buffer bound exceeded"));
        }
        if !packet.data.is_empty() && let Some(pts) = packet.pts.or(packet.dts) {
            if self.anchors.len() >= MAX_AUDIO_FRAMES { return Err(invalid("too many AAC timestamp anchors")); }
            self.anchors.push_back((self.pending.len(), pts));
        }
        self.pending.extend_from_slice(&packet.data);
        let mut consumed = 0;
        let mut output = Vec::new();
        while self.pending.len() - consumed >= 7 {
            let (config, header, size) = adts_header(&self.pending[consumed..])?;
            if self.pending.len() - consumed < size { break; }
            if output.len() >= MAX_AUDIO_FRAMES { return Err(invalid("too many AAC frames in one PES")); }
            if self.config.as_ref().is_some_and(|known| known != &config) {
                return Err(MediaDecodeError::Unsupported);
            }
            while self.anchors.front().is_some_and(|&(at, _)| at <= consumed) {
                let (_, pts) = self.anchors.pop_front().unwrap();
                self.clock = Some(AudioClock { pts, samples: 0 });
            }
            let clock = self.clock.as_mut().ok_or_else(|| invalid("AAC frame without PTS"))?;
            let rate = u128::from(config.sample_rate);
            let numerator = (u128::from(clock.pts) * rate
                + u128::from(clock.samples) * u128::from(TIMESTAMP_TIMESCALE)) * 1_000_000_000;
            let timestamp_ns = i64::try_from(numerator / (u128::from(TIMESTAMP_TIMESCALE) * rate))
                .map_err(|_| invalid("AAC timestamp overflow"))?;
            if self.decoder.is_none() {
                self.decoder = Some(AacDecoder::new(config.clone())?);
                self.config = Some(config.clone());
            }
            let samples = self.decoder.as_mut().unwrap().decode(&self.pending[consumed + header..consumed + size])?;
            clock.samples = clock.samples.checked_add(config.frame_samples as u64)
                .ok_or_else(|| invalid("AAC sample clock overflow"))?;
            output.push(MediaSample::Audio { timestamp_ns, samples });
            consumed += size;
        }
        if consumed != 0 {
            self.pending.drain(..consumed);
            for (at, _) in &mut self.anchors { *at = at.saturating_sub(consumed); }
        }
        Ok(output)
    }

    pub fn finish(&mut self) -> Result<Vec<MediaSample>, MediaDecodeError> {
        if let Some(error) = &self.error { return Err(error.clone()); }
        if !self.pending.is_empty() { return Err(invalid("truncated ADTS frame at EOF")); }
        self.finished = true;
        Ok(Vec::new())
    }

    pub fn metadata(&self) -> Option<MediaMetadata> {
        let config = self.config.as_ref()?;
        Some(MediaMetadata { presentation_size: None, duration: None, width: None, height: None,
            sample_rate: Some(config.sample_rate), channels: Some(config.channels) })
    }
}

fn adts_header(bytes: &[u8]) -> Result<(AacConfig, usize, usize), MediaDecodeError> {
    if bytes.len() < 7 { return Err(invalid("truncated ADTS header")); }
    if bytes[0] != 0xff || bytes[1] & 0xf6 != 0xf0 { return Err(invalid("invalid ADTS sync/layer")); }
    let frequency_index = bytes[2] >> 2 & 15;
    let sample_rate = *AAC_RATES.get(usize::from(frequency_index)).ok_or(MediaDecodeError::Unsupported)?;
    let object_type = (bytes[2] >> 6) + 1;
    let channels = u16::from((bytes[2] & 1) << 2 | bytes[3] >> 6);
    if !matches!(object_type, 1 | 2) || !(1..=2).contains(&channels) || bytes[6] & 3 != 0 {
        return Err(MediaDecodeError::Unsupported);
    }
    let size = (usize::from(bytes[3] & 3) << 11) | (usize::from(bytes[4]) << 3)
        | usize::from(bytes[5] >> 5);
    let header = if bytes[1] & 1 != 0 { 7 } else { 9 };
    if size <= header { return Err(invalid("invalid ADTS frame length")); }
    Ok((AacConfig { object_type, sample_rate, frequency_index, channels, frame_samples: 1024 }, header, size))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(kind: StreamKind, data: Vec<u8>, pts: Option<u64>) -> ElementaryPacket {
        ElementaryPacket { pid: if kind == StreamKind::H264 { 0x101 } else { 0x102 },
            kind, pts, dts: pts, data, discontinuity: false }
    }

    fn avc_fixture() -> Vec<Vec<u8>> {
        let mut idr = vec![0x65, 0xb8, 0x40, 0xa0, 0xd0];
        idr.extend([235; 256]);
        idr.extend([128; 128]);
        idr.push(0x80);
        vec![vec![0x67, 0x42, 0x00, 0x0a, 0xf4, 0xf2], vec![0x68, 0xce, 0x3c, 0x80], idr]
    }

    fn annex(nals: &[Vec<u8>]) -> Vec<u8> {
        let mut data = Vec::new();
        for (number, nal) in nals.iter().enumerate() {
            data.extend_from_slice(if number % 2 == 0 { &[0, 0, 0, 1] } else { &[0, 0, 1] });
            data.extend_from_slice(nal);
        }
        data
    }

    fn length_prefixed(nals: &[Vec<u8>]) -> Vec<u8> {
        let mut data = Vec::new();
        for nal in nals {
            data.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            data.extend_from_slice(nal);
        }
        data
    }

    #[test]
    fn transport_video_matches_shared_avc_across_pes_and_drains_eof() {
        let nals = avc_fixture();
        let mut expected = Mp4AvcPackets::new_fragmented(Mp4Index {
            timescale: TIMESTAMP_TIMESCALE, duration_ticks: 0,
            config: AvcConfig { nal_length_size: 4, sequence_parameters: vec![parse_sps(&nals[0]).unwrap()],
                picture_parameter_sets: vec![nals[1].clone()] }, samples: Vec::new(),
        }, 0).unwrap();
        let mut actual = TransportVideoDecoder::new();
        let mut expected_frames = Vec::new();
        let mut actual_frames = Vec::new();
        let mut offset = 0;
        for (number, (pts, dts)) in [(90000, 90000), (99000, 93000)].into_iter().enumerate() {
            let unit = if number == 0 { nals.clone() } else { vec![nals[2].clone()] };
            let data = length_prefixed(&unit);
            expected_frames.extend(expected.push_sample(number, Sample { offset, size: data.len() as u32,
                decode_time: dts, presentation_time: pts as i64, keyframe: true }, &data).unwrap());
            offset += data.len() as u64;
            let mut pes = packet(StreamKind::H264, annex(&unit), Some(pts));
            pes.dts = Some(dts);
            actual_frames.extend(actual.push(&pes).unwrap());
        }
        loop {
            let tail = expected.finish_input().unwrap();
            if tail.is_empty() { break; }
            expected_frames.extend(tail);
        }
        actual_frames.extend(actual.finish().unwrap());
        assert_eq!(actual_frames, expected_frames);
        assert_eq!(actual_frames.len(), 2);
        assert_eq!(actual_frames.iter().map(|frame| frame.timestamp).collect::<Vec<_>>(), [1.0, 1.1]);
        assert_eq!(&actual_frames[0].rgba[..4], &[255, 255, 255, 255]);
        assert_eq!(actual.next_sample, 2);
        assert_eq!(actual.metadata().unwrap().duration, None);
        assert!(actual.finish().unwrap().is_empty());
        assert!(actual.push(&packet(StreamKind::H264, annex(&nals), Some(108000))).is_err());
    }

    #[test]
    fn transport_video_discontinuity_discards_tail_and_requires_fresh_idr_config() {
        let nals = avc_fixture();
        let mut decoder = TransportVideoDecoder::new();
        assert!(decoder.push(&packet(StreamKind::H264, annex(&nals), Some(90000))).unwrap().is_empty());
        let mut restart = packet(StreamKind::H264, annex(&nals), Some(180000));
        restart.discontinuity = true;
        let mut frames = decoder.push(&restart).unwrap();
        frames.extend(decoder.finish().unwrap());
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].timestamp, 2.0);
        restart.data = annex(&[nals[2].clone()]);
        assert!(decoder.push(&restart).is_err());
    }

    #[test]
    fn transport_video_validates_aud_boundaries_and_rejects_ambiguous_timing() {
        let mut nals = avc_fixture();
        nals.insert(2, vec![9, 0xf0]);
        let mut decoder = TransportVideoDecoder::new();
        decoder.push(&packet(StreamKind::H264, annex(&nals), Some(90000))).unwrap();
        assert_eq!(decoder.finish().unwrap().len(), 1);
        nals.extend([vec![9, 0xf0], avc_fixture()[2].clone()]);
        assert_eq!(annex_b_units(&annex(&nals)).unwrap().len(), 2);
        assert_eq!(TransportVideoDecoder::new().push(&packet(StreamKind::H264, annex(&nals), Some(90000))),
            Err(MediaDecodeError::Unsupported));
        let mut no_aud = avc_fixture();
        no_aud.push(no_aud[2].clone());
        assert_eq!(annex_b_units(&annex(&no_aud)), Err(MediaDecodeError::Unsupported));
        assert!(annex_b_units(&[0, 0, 1, 9]).is_err());
        assert!(annex_b_units(&[0, 0, 1]).is_err());
        assert!(annex_b_units(&[1, 0, 0, 1, 0x65, 0x80]).is_err());
    }

    #[test]
    fn transport_video_rejects_timestamp_track_and_slice_identity_errors() {
        let nals = avc_fixture();
        assert!(TransportVideoDecoder::new().push(&packet(StreamKind::H264, annex(&nals), None)).is_err());
        assert!(TransportVideoDecoder::new().push(&packet(StreamKind::H264, annex(&nals), Some(u64::MAX))).is_err());
        let mut decoder = TransportVideoDecoder::new();
        decoder.push(&packet(StreamKind::H264, annex(&nals), Some(90000))).unwrap();
        let mut other = packet(StreamKind::H264, annex(&nals), Some(93000));
        other.pid += 1;
        assert!(decoder.push(&other).is_err());
        let mut partial_slice = nals.clone();
        partial_slice[2][1] &= 0x7f;
        assert!(TransportVideoDecoder::new().push(&packet(StreamKind::H264,
            annex(&partial_slice), Some(90000))).is_err());
        let many: Vec<_> = (0..=MAX_NALS).map(|_| vec![12, 0x80]).collect();
        assert!(annex_b_units(&annex(&many)).is_err());
    }

    fn raw_aac() -> Vec<u8> {
        // Transmitted zero-band mono SCE, matching the existing AAC frame fixture.
        let fields = [(0u32, 3usize), (0, 4), (100, 8), (0, 1), (0, 2), (0, 1),
            (0, 6), (0, 1), (0, 1), (0, 1), (0, 1), (7, 3)];
        let mut bytes = vec![0; fields.iter().map(|field| field.1).sum::<usize>().div_ceil(8)];
        let mut at = 0;
        for (value, width) in fields {
            for shift in (0..width).rev() {
                bytes[at / 8] |= ((value >> shift & 1) as u8) << (7 - at % 8);
                at += 1;
            }
        }
        bytes
    }

    fn adts(raw: &[u8], crc: bool) -> Vec<u8> {
        let length = raw.len() + if crc { 9 } else { 7 };
        let mut data = vec![0xff, if crc { 0xf0 } else { 0xf1 }, 0x4c,
            0x40 | (length >> 11) as u8, (length >> 3) as u8,
            ((length & 7) << 5) as u8 | 0x1f, 0xfc];
        if crc { data.extend([0, 0]); }
        data.extend_from_slice(raw);
        data
    }

    #[test]
    fn transport_audio_matches_shared_aac_and_advances_rational_clock() {
        let raw = raw_aac();
        let frame = adts(&raw, false);
        let (config, _, _) = adts_header(&frame).unwrap();
        let mut reference = AacDecoder::new(config).unwrap();
        let mut decoder = TransportAudioDecoder::new();
        let mut data = frame.clone();
        data.extend_from_slice(&frame);
        let mut samples = decoder.push(&packet(StreamKind::Aac, data, Some(90000))).unwrap();
        samples.extend(decoder.push(&packet(StreamKind::Aac, frame, None)).unwrap());
        for (number, sample) in samples.into_iter().enumerate() {
            let MediaSample::Audio { timestamp_ns, samples } = sample else { panic!("not audio"); };
            assert_eq!(timestamp_ns, 1_000_000_000 + (number as i64 * 1024 * 1_000_000_000 / 48000));
            assert_eq!(samples, reference.decode(&raw).unwrap());
        }
        assert_eq!(decoder.metadata().unwrap().sample_rate, Some(48000));
        assert_eq!(decoder.metadata().unwrap().duration, None);
        assert!(decoder.finish().unwrap().is_empty());
    }

    #[test]
    fn transport_audio_buffers_adts_at_every_pes_split_including_crc_header() {
        for crc in [false, true] {
            let frame = adts(&raw_aac(), crc);
            for at in 0..=frame.len() {
                let mut decoder = TransportAudioDecoder::new();
                let mut samples = decoder.push(&packet(StreamKind::Aac, frame[..at].to_vec(), Some(90000))).unwrap();
                samples.extend(decoder.push(&packet(StreamKind::Aac, frame[at..].to_vec(),
                    (at == 0).then_some(90000))).unwrap());
                assert_eq!(samples.len(), 1, "split={at}, CRC={crc}");
                assert!(matches!(&samples[0], MediaSample::Audio { timestamp_ns: 1_000_000_000, .. }));
                decoder.finish().unwrap();
            }
        }
    }

    #[test]
    fn transport_audio_continuation_timestamp_anchors_the_next_started_frame() {
        let frame = adts(&raw_aac(), false);
        let mut decoder = TransportAudioDecoder::new();
        decoder.push(&packet(StreamKind::Aac, frame[..3].to_vec(), Some(90000))).unwrap();
        let mut rest = frame[3..].to_vec();
        rest.extend_from_slice(&frame);
        let output = decoder.push(&packet(StreamKind::Aac, rest, Some(91920))).unwrap();
        assert_eq!(output.len(), 2);
        assert!(matches!(&output[0], MediaSample::Audio { timestamp_ns: 1_000_000_000, .. }));
        assert!(matches!(&output[1], MediaSample::Audio { timestamp_ns: 1_021_333_333, .. }));
    }

    #[test]
    fn transport_audio_discontinuity_resets_partial_frame_and_clock() {
        let frame = adts(&raw_aac(), false);
        let mut decoder = TransportAudioDecoder::new();
        decoder.push(&packet(StreamKind::Aac, frame[..5].to_vec(), Some(90000))).unwrap();
        assert!(decoder.finish().is_err());
        let mut restart = packet(StreamKind::Aac, frame, Some(450000));
        restart.discontinuity = true;
        let output = decoder.push(&restart).unwrap();
        assert!(matches!(&output[0], MediaSample::Audio { timestamp_ns: 5_000_000_000, .. }));
        decoder.finish().unwrap();
    }

    #[test]
    fn transport_audio_rejects_invalid_lengths_configs_and_resource_bounds() {
        let frame = adts(&raw_aac(), false);
        assert!(TransportAudioDecoder::new().push(&packet(StreamKind::Aac, frame.clone(), None)).is_err());
        assert!(TransportAudioDecoder::new().push(&packet(StreamKind::Aac, frame.clone(), Some(u64::MAX))).is_err());
        for change in 0..4 {
            let mut bad = frame.clone();
            match change {
                0 => bad[0] = 0,
                1 => bad[2] = bad[2] & 0xc3 | 15 << 2,
                2 => bad[6] |= 1,
                _ => { bad[3] &= 0xfc; bad[4] = 0; bad[5] &= 0x1f; },
            }
            assert!(TransportAudioDecoder::new().push(&packet(StreamKind::Aac, bad, Some(90000))).is_err());
        }
        let mut decoder = TransportAudioDecoder::new();
        decoder.push(&packet(StreamKind::Aac, frame.clone(), Some(90000))).unwrap();
        let mut changed = frame.clone();
        changed[2] = changed[2] & 0xc3 | 4 << 2;
        assert_eq!(decoder.push(&packet(StreamKind::Aac, changed, Some(91920))), Err(MediaDecodeError::Unsupported));
        let many = frame.repeat(MAX_AUDIO_FRAMES + 1);
        assert!(TransportAudioDecoder::new().push(&packet(StreamKind::Aac, many, Some(90000))).is_err());
        assert!(TransportAudioDecoder::new().push(&packet(StreamKind::Aac,
            vec![0; MAX_PES_BYTES + 1], Some(90000))).is_err());
    }

    #[test]
    #[ignore = "local H264/AAC TS fixture: set WEBMEDIA_TS_FIXTURE to the playlist folder"]
    fn transport_local_hls_fixture_decodes_video_and_audio_across_segments() {
        use super::super::hls::{Playlist, parse_playlist};
        use super::super::mpeg_ts::TransportStream;

        #[derive(Default)]
        struct Checks {
            video_frames: usize,
            last_video: Option<f32>,
            nonblank_video: bool,
            audio_frames: usize,
            last_audio: Option<i64>,
            pcm_samples: usize,
            nonzero_pcm: usize,
        }
        impl Checks {
            fn video(&mut self, frames: Vec<VideoFrame>) {
                for frame in frames {
                    assert!(frame.timestamp.is_finite() && frame.timestamp >= 0.0);
                    assert!(self.last_video.is_none_or(|last| frame.timestamp >= last),
                        "video went backward: {:?} -> {}", self.last_video, frame.timestamp);
                    assert!(frame.width > 0 && frame.height > 0);
                    assert_eq!(frame.rgba.len(), frame.width as usize * frame.height as usize * 4);
                    self.nonblank_video |= frame.rgba.chunks_exact(4)
                        .any(|pixel| pixel[..3].iter().any(|&channel| channel != 0));
                    self.last_video = Some(frame.timestamp);
                    self.video_frames += 1;
                }
            }
            fn audio(&mut self, frames: Vec<MediaSample>) {
                for frame in frames {
                    let MediaSample::Audio { timestamp_ns, samples } = frame else {
                        panic!("transport audio returned a non-audio sample");
                    };
                    assert!(timestamp_ns >= 0);
                    assert!(self.last_audio.is_none_or(|last| timestamp_ns >= last),
                        "audio went backward: {:?} -> {timestamp_ns}", self.last_audio);
                    assert!(samples.sample_rate > 0 && samples.channels > 0);
                    assert_eq!(samples.samples.len() % usize::from(samples.channels), 0);
                    assert!(samples.samples.iter().all(|sample| sample.is_finite()));
                    self.pcm_samples += samples.samples.len();
                    self.nonzero_pcm += samples.samples.iter().filter(|sample| sample.abs() > 1e-7).count();
                    self.last_audio = Some(timestamp_ns);
                    self.audio_frames += 1;
                }
            }
        }

        let fixture = std::path::PathBuf::from(std::env::var_os("WEBMEDIA_TS_FIXTURE")
            .expect("WEBMEDIA_TS_FIXTURE is required for this ignored test"));
        let playlist_path = if fixture.is_dir() { fixture.join("index.m3u8") } else { fixture };
        let folder = playlist_path.parent().unwrap();
        let text = std::fs::read_to_string(&playlist_path).unwrap();
        let Playlist::Media(playlist) = parse_playlist(&text).unwrap() else {
            panic!("fixture must be a local TS media playlist");
        };
        assert!(playlist.segments.len() > 1, "fixture must cross segment boundaries");
        let mut demux = TransportStream::new();
        let mut packets = Vec::new();
        let mut bytes = 0usize;
        for segment in &playlist.segments {
            assert!(!segment.discontinuity && segment.map.is_none() && segment.byte_range.is_none(),
                "fixture expects continuous, complete TS segments");
            let uri = std::path::Path::new(&segment.uri);
            assert!(uri.components().all(|part| matches!(part,
                std::path::Component::Normal(_) | std::path::Component::CurDir)));
            let data = std::fs::read(folder.join(uri)).unwrap();
            bytes = bytes.checked_add(data.len()).unwrap();
            assert!(bytes <= 64 * 1024 * 1024, "local fixture exceeds test memory bound");
            for chunk in data.chunks(32 * 1024) {
                packets.extend(demux.push(chunk).unwrap());
            }
            packets.extend(demux.finish_segment().unwrap());
        }
        // This local fixture does not wrap. One shared DTS baseline preserves
        // A/V offsets; production wrap/discontinuity normalization belongs upstream.
        let baseline = packets.iter().filter_map(|packet| packet.dts.or(packet.pts)).min().unwrap();
        let mut video = TransportVideoDecoder::new();
        let mut audio = TransportAudioDecoder::new();
        let mut checks = Checks::default();
        for mut packet in packets {
            packet.pts = packet.pts.map(|pts| pts.checked_sub(baseline).expect("fixture PTS precedes shared DTS baseline"));
            packet.dts = packet.dts.map(|dts| dts.checked_sub(baseline).unwrap());
            let context = format!("PID={} kind={:?} PTS={:?} DTS={:?}", packet.pid, packet.kind, packet.pts, packet.dts);
            match packet.kind {
                StreamKind::H264 => checks.video(video.push(&packet).unwrap_or_else(|error| panic!("{context}: {error:?}"))),
                StreamKind::Aac => checks.audio(audio.push(&packet).unwrap_or_else(|error| panic!("{context}: {error:?}"))),
            }
        }
        checks.video(video.finish().unwrap());
        checks.audio(audio.finish().unwrap());
        assert!(checks.video_frames > 1 && checks.nonblank_video,
            "video frames={}, nonblank={}", checks.video_frames, checks.nonblank_video);
        assert!(checks.audio_frames > 1 && checks.pcm_samples >= 16000 && checks.nonzero_pcm >= 1024,
            "audio frames={}, PCM samples={}, nonzero={}", checks.audio_frames, checks.pcm_samples, checks.nonzero_pcm);
        assert_eq!(video.metadata().unwrap().duration, None);
        assert_eq!(audio.metadata().unwrap().duration, None);
        eprintln!("local TS fixture segments={} video_frames={} audio_frames={} PCM_samples={} nonzero_PCM={} video_end={:?} audio_end_ns={:?}",
            playlist.segments.len(), checks.video_frames, checks.audio_frames, checks.pcm_samples,
            checks.nonzero_pcm, checks.last_video, checks.last_audio);
    }
}
