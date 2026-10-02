//! Incremental classic MP4/AVC access-unit extraction.

use super::backend::{MediaDecodeError, MediaMetadata, StreamingVideoDecoder, VideoFrame};
use super::h264::{
    AvcError, MemoryManagement, NalStream, decode_intra_2003, frame_from_yuv420,
    parse_cabac_inter_slice, parse_pps_2003, parse_pps_2005, parse_slice_type,
};
#[cfg(test)]
use super::h264_high::decode_cabac_idr_2005;
use super::h264_high::{Yuv420Picture, decode_cabac_i_yuv_2005, decode_cabac_idr_yuv_2005};
use super::h264_inter::{decode_cabac_b_2005, decode_cabac_p_2005};
use super::mp4::{Mp4Error, Mp4Index};

const MAX_BUFFER_BYTES: usize = 64 * 1024 * 1024;
const MAX_FRAMES_PER_PUSH: usize = 4;

#[derive(Default)]
pub struct Mp4AvcStream {
    bytes: Vec<u8>,
    base_offset: u64,
    index: Option<Mp4Index>,
    next_sample: usize,
    pending_pictures: Vec<(i64, VideoFrame)>,
    reference_pictures: Vec<Yuv420Picture>,
    future_min_pts: Vec<i64>,
    decode_error: Option<MediaDecodeError>,
    dropped_nonreference_samples: usize,
    dropped_until_idr_samples: usize,
    waiting_for_idr: bool,
}

impl Mp4AvcStream {
    pub fn new() -> Self {
        Self::default()
    }

    fn invalid(error: impl std::fmt::Debug) -> MediaDecodeError {
        MediaDecodeError::InvalidData(format!("MP4/AVC: {error:?}"))
    }

    fn avc_error(error: AvcError) -> MediaDecodeError {
        if std::env::var_os("WEBMEDIA_TRACE_RECOVERY").is_some() {
            eprintln!("AVC decode error: {error:?}");
        }
        match error {
            AvcError::Unsupported(_) | AvcError::UnsupportedProfile(_) => {
                MediaDecodeError::Unsupported
            }
            other => Self::invalid(other),
        }
    }

    fn queue_picture(&mut self, presentation_time: i64, frame: VideoFrame) {
        let position = self
            .pending_pictures
            .partition_point(|(time, _)| *time <= presentation_time);
        self.pending_pictures
            .insert(position, (presentation_time, frame));
    }

    fn drain_presentable(&mut self, frames: &mut Vec<VideoFrame>) {
        let next_time = self.future_min_pts[self.next_sample];
        let ready = self
            .pending_pictures
            .partition_point(|(time, _)| *time <= next_time);
        for (_, frame) in self.pending_pictures.drain(..ready) {
            frames.push(frame);
        }
    }

    fn decode_sample(
        index: &Mp4Index,
        bytes: &[u8],
        base_offset: u64,
        sample_number: usize,
        references: &[Yuv420Picture],
    ) -> Result<
        (
            VideoFrame,
            Option<Yuv420Picture>,
            bool,
            Vec<MemoryManagement>,
        ),
        MediaDecodeError,
    > {
        let sample = &index.samples[sample_number];
        let start = usize::try_from(
            sample
                .offset
                .checked_sub(base_offset)
                .ok_or_else(|| Self::invalid("sample precedes buffered window"))?,
        )
        .map_err(Self::invalid)?;
        let end = start
            .checked_add(sample.size as usize)
            .ok_or(MediaDecodeError::Unsupported)?;
        let payload = bytes
            .get(start..end)
            .ok_or_else(|| Self::invalid("missing sample"))?;
        let mut stream = NalStream::new(index.config.nal_length_size).map_err(Self::invalid)?;
        let nals = stream.push(payload).map_err(Self::invalid)?;
        stream.finish().map_err(Self::invalid)?;
        let mut picture = None;
        let mut reference = None;
        let mut is_idr = false;
        let mut marking = Vec::new();
        for nal in nals {
            if std::env::var_os("WEBMEDIA_TRACE_SAMPLE").is_some() {
                eprintln!("sample {sample_number}: NAL type {}", nal[0] & 0x1f);
            }
            match nal[0] & 0x1f {
                5 => {
                    if picture.is_some() {
                        return Err(MediaDecodeError::Unsupported);
                    }
                    is_idr = true;
                    let sps = index
                        .config
                        .sequence_parameters
                        .first()
                        .ok_or(MediaDecodeError::Unsupported)?;
                    let pps = index
                        .config
                        .picture_parameter_sets
                        .first()
                        .ok_or(MediaDecodeError::Unsupported)?;
                    picture = Some(match sps.profile_idc {
                        66 => {
                            let pps = parse_pps_2003(pps).map_err(Self::avc_error)?;
                            decode_intra_2003(&nal, sps, &pps).map_err(Self::avc_error)?
                        }
                        100 => {
                            let pps = parse_pps_2005(pps).map_err(Self::avc_error)?;
                            let yuv = decode_cabac_idr_yuv_2005(&nal, sps, &pps)
                                .map_err(Self::avc_error)?;
                            let frame = frame_from_yuv420(sps, &yuv.luma, &yuv.cb, &yuv.cr);
                            reference = Some(yuv);
                            frame
                        }
                        _ => return Err(MediaDecodeError::Unsupported),
                    });
                }
                1 => {
                    if picture.is_some() {
                        return Err(MediaDecodeError::Unsupported);
                    }
                    let sps = index
                        .config
                        .sequence_parameters
                        .first()
                        .ok_or(MediaDecodeError::Unsupported)?;
                    let pps = index
                        .config
                        .picture_parameter_sets
                        .first()
                        .ok_or(MediaDecodeError::Unsupported)?;
                    let pps = parse_pps_2005(pps).map_err(Self::avc_error)?;
                    if parse_slice_type(&nal).map_err(Self::avc_error)? % 5 == 2 {
                        let (yuv, operations) =
                            decode_cabac_i_yuv_2005(&nal, sps, &pps, references.last())
                                .map_err(Self::avc_error)?;
                        marking = operations;
                        picture = Some(frame_from_yuv420(sps, &yuv.luma, &yuv.cb, &yuv.cr));
                        if nal[0] & 0x60 != 0 {
                            reference = Some(yuv);
                        }
                        continue;
                    }
                    let parsed = parse_cabac_inter_slice(&nal, sps, &pps.core);
                    if std::env::var_os("WEBMEDIA_TRACE_SAMPLE").is_some() {
                        if let Err(error) = &parsed {
                            eprintln!("sample {sample_number} slice header error: {error:?}");
                        }
                    }
                    let parsed = parsed.map_err(Self::avc_error)?;
                    marking = parsed.marking;
                    let slice_type = parsed.slice_type;
                    let yuv = match slice_type % 5 {
                        0 => {
                            let decoded = decode_cabac_p_2005(&nal, sps, &pps, references);
                            if std::env::var_os("WEBMEDIA_TRACE_P").is_some() {
                                if let Err(error) = &decoded {
                                    eprintln!("P sample {sample_number} decode error: {error:?}");
                                }
                            }
                            decoded.map_err(Self::avc_error)?
                        }
                        1 => {
                            let decoded = decode_cabac_b_2005(&nal, sps, &pps, references);
                            if std::env::var_os("WEBMEDIA_TRACE_B").is_some() {
                                if let Err(error) = &decoded {
                                    eprintln!("B sample {sample_number} decode error: {error:?}");
                                }
                            }
                            decoded.map_err(Self::avc_error)?
                        }
                        _ => return Err(MediaDecodeError::Unsupported),
                    };
                    picture = Some(frame_from_yuv420(sps, &yuv.luma, &yuv.cb, &yuv.cr));
                    if nal[0] & 0x60 != 0 {
                        reference = Some(yuv);
                    }
                }
                2..=4 => return Err(MediaDecodeError::Unsupported),
                _ => {}
            }
        }
        let mut picture = picture.ok_or(MediaDecodeError::Unsupported)?;
        if sample.presentation_time < 0 {
            return Err(MediaDecodeError::Unsupported);
        }
        picture.timestamp = sample.presentation_time as f32 / index.timescale as f32;
        Ok((picture, reference, is_idr, marking))
    }

    fn sample_video_header(
        index: &Mp4Index,
        bytes: &[u8],
        base_offset: u64,
        sample_number: usize,
    ) -> Option<u8> {
        let sample = &index.samples[sample_number];
        let Some(start) = sample
            .offset
            .checked_sub(base_offset)
            .and_then(|offset| usize::try_from(offset).ok())
        else {
            return None;
        };
        let Some(end) = start.checked_add(sample.size as usize) else {
            return None;
        };
        let Some(payload) = bytes.get(start..end) else {
            return None;
        };
        let Ok(mut nals) = NalStream::new(index.config.nal_length_size) else {
            return None;
        };
        let Ok(units) = nals.push(payload) else {
            return None;
        };
        if nals.finish().is_err() {
            return None;
        }
        let mut headers = units
            .iter()
            .filter(|nal| matches!(nal[0] & 0x1f, 1..=5))
            .map(|nal| nal[0]);
        let header = headers.next()?;
        headers
            .all(|other| other & 0x7f == header & 0x7f)
            .then_some(header)
    }

    fn discard_consumed_prefix(&mut self) {
        let keep_from = self
            .index
            .as_ref()
            .and_then(|index| index.samples.get(self.next_sample))
            .map(|sample| sample.offset)
            .unwrap_or(self.base_offset + self.bytes.len() as u64);
        let drop_count = keep_from
            .saturating_sub(self.base_offset)
            .min(self.bytes.len() as u64) as usize;
        self.bytes.drain(..drop_count);
        self.base_offset += drop_count as u64;
    }
}

impl StreamingVideoDecoder for Mp4AvcStream {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<VideoFrame>, MediaDecodeError> {
        if let Some(error) = &self.decode_error {
            return Err(error.clone());
        }
        if bytes.len() > MAX_BUFFER_BYTES.saturating_sub(self.bytes.len()) {
            return Err(MediaDecodeError::Unsupported);
        }
        self.bytes.extend_from_slice(bytes);
        if self.index.is_none() {
            match Mp4Index::parse_prefix(&self.bytes) {
                Ok(index) => self.index = Some(index),
                Err(Mp4Error::Incomplete) => return Ok(Vec::new()),
                Err(error) => return Err(Self::invalid(error)),
            }
        }
        if self.future_min_pts.is_empty() {
            let index = self.index.as_ref().unwrap();
            self.future_min_pts = vec![i64::MAX; index.samples.len() + 1];
            for sample in (0..index.samples.len()).rev() {
                self.future_min_pts[sample] =
                    self.future_min_pts[sample + 1].min(index.samples[sample].presentation_time);
            }
        }
        let mut frames = Vec::new();
        while frames.len() < MAX_FRAMES_PER_PUSH {
            let Some(sample) = self.index.as_ref().unwrap().samples.get(self.next_sample) else {
                break;
            };
            let end = sample
                .offset
                .checked_add(u64::from(sample.size))
                .ok_or(MediaDecodeError::Unsupported)?;
            if end > self.base_offset + self.bytes.len() as u64 {
                break;
            }
            if self.waiting_for_idr {
                let header = Self::sample_video_header(
                    self.index.as_ref().unwrap(),
                    &self.bytes,
                    self.base_offset,
                    self.next_sample,
                );
                if header.is_none_or(|header| header & 0x1f != 5) {
                    self.dropped_until_idr_samples += 1;
                    self.next_sample += 1;
                    continue;
                }
                self.reference_pictures.clear();
                self.waiting_for_idr = false;
            }
            let (frame, reference, is_idr, marking) = match Self::decode_sample(
                self.index.as_ref().unwrap(),
                &self.bytes,
                self.base_offset,
                self.next_sample,
                &self.reference_pictures,
            ) {
                Ok(frame) => frame,
                Err(error)
                    if Self::sample_video_header(
                        self.index.as_ref().unwrap(),
                        &self.bytes,
                        self.base_offset,
                        self.next_sample,
                    )
                    .is_some_and(|header| header & 0x1f == 1 && header & 0x60 == 0) =>
                {
                    if std::env::var_os("WEBMEDIA_TRACE_RECOVERY").is_some() {
                        eprintln!(
                            "dropping non-reference sample {}: {error:?}",
                            self.next_sample
                        );
                    }
                    self.dropped_nonreference_samples += 1;
                    self.next_sample += 1;
                    self.drain_presentable(&mut frames);
                    continue;
                }
                Err(error) if self.next_sample + 1 < self.index.as_ref().unwrap().samples.len() => {
                    if std::env::var_os("WEBMEDIA_TRACE_RECOVERY").is_some() {
                        eprintln!(
                            "resync after reference sample {}: {error:?}",
                            self.next_sample
                        );
                    }
                    self.waiting_for_idr = true;
                    self.dropped_until_idr_samples += 1;
                    self.next_sample += 1;
                    frames.extend(self.pending_pictures.drain(..).map(|(_, frame)| frame));
                    continue;
                }
                Err(error) if !frames.is_empty() => {
                    self.decode_error = Some(error);
                    break;
                }
                Err(error) => return Err(error),
            };
            let presentation_time = sample.presentation_time;
            self.next_sample += 1;
            if is_idr {
                self.reference_pictures.clear();
            }
            if let Some(reference) = reference {
                let max_frame_num = 1u32
                    << self.index.as_ref().unwrap().config.sequence_parameters[0].frame_num_bits;
                for operation in marking {
                    match operation {
                        MemoryManagement::ForgetShortTerm(distance) => {
                            let target = reference
                                .frame_num
                                .wrapping_add(max_frame_num)
                                .wrapping_sub(distance.wrapping_add(1) % max_frame_num)
                                % max_frame_num;
                            let position = self
                                .reference_pictures
                                .iter()
                                .position(|picture| picture.frame_num == target)
                                .ok_or_else(|| Self::invalid("missing short-term reference"))?;
                            self.reference_pictures.remove(position);
                        }
                    }
                }
                let max_references = self.index.as_ref().unwrap().config.sequence_parameters[0]
                    .max_num_ref_frames as usize;
                if max_references > 0 {
                    self.reference_pictures.push(reference);
                    if self.reference_pictures.len() > max_references {
                        self.reference_pictures.remove(0);
                    }
                }
            }
            self.queue_picture(presentation_time, frame);
            self.drain_presentable(&mut frames);
        }
        self.discard_consumed_prefix();
        Ok(frames)
    }

    fn metadata(&self) -> Option<MediaMetadata> {
        let index = self.index.as_ref()?;
        let sps = index.config.sequence_parameters.first()?;
        Some(MediaMetadata {
            duration: Some(index.duration_ticks as f32 / index.timescale as f32),
            width: Some(sps.width),
            height: Some(sps.height),
            sample_rate: None,
            channels: None,
        })
    }

    fn finish(&self) -> Result<(), MediaDecodeError> {
        if self
            .index
            .as_ref()
            .is_some_and(|index| self.next_sample == index.samples.len())
            && self.pending_pictures.is_empty()
        {
            Ok(())
        } else {
            Err(Self::invalid("truncated MP4/AVC stream"))
        }
    }

    fn has_buffered_samples(&self) -> bool {
        self.index
            .as_ref()
            .and_then(|index| index.samples.get(self.next_sample))
            .and_then(|sample| sample.offset.checked_add(u64::from(sample.size)))
            .is_some_and(|end| end <= self.base_offset + self.bytes.len() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::h264::{
        AvcConfig, parse_cabac_idr_i_slice, parse_cabac_inter_slice, parse_pps_2005, parse_sps,
        type0_pic_order_count,
    };
    use crate::video::h264_cabac::{
        CabacDecoder, ChromaDcContexts, CodedBlockContexts, InterMbContexts, IntraMbTypeContexts,
        IntraPredContexts, Luma4x4Contexts, Luma8x8Contexts, MbQpContexts, MotionVectorContexts,
        Transform8x8Contexts, intra4x4_modes, intra8x8_modes,
    };
    use crate::video::h264_inter::{
        b_reference_lists, predict_l0_16x16, predict_l1_16x16, spatial_direct_motion,
    };
    use crate::video::h264_intra::{
        reconstruct_chroma_dc_only, reconstruct_intra4x4_luma, reconstruct_intra8x8_luma,
    };
    use crate::video::h264_transform::{
        chroma_qp, inverse_4x4_frame_scan, inverse_4x4_residual, inverse_8x8_frame_scan,
        inverse_8x8_residual, inverse_chroma_dc,
    };
    use crate::video::mp4::Sample;

    #[test]
    fn presents_b_pictures_in_timestamp_order() {
        let mut decoder = Mp4AvcStream::new();
        decoder.future_min_pts = vec![0, 40, 40, 40, 80, i64::MAX];
        let mut delivered = Vec::new();
        for time in [0, 100, 60, 40, 80] {
            decoder.queue_picture(
                time,
                VideoFrame {
                    width: 1,
                    height: 1,
                    rgba: std::sync::Arc::new(vec![0, 0, 0, 255]),
                    timestamp: time as f32 / 1000.0,
                },
            );
            decoder.next_sample += 1;
            decoder.drain_presentable(&mut delivered);
        }
        assert_eq!(
            delivered
                .iter()
                .map(|frame| frame.timestamp)
                .collect::<Vec<_>>(),
            [0.0, 0.04, 0.06, 0.08, 0.1]
        );
        assert!(decoder.pending_pictures.is_empty());
    }

    #[test]
    fn emits_a_2003_baseline_frame_from_chunked_mp4_sample() {
        let sps = parse_sps(&[0x67, 0x42, 0x00, 0x0a, 0xf4, 0xf2]).unwrap();
        let pps = vec![0x68, 0xce, 0x3c, 0x80];
        parse_pps_2003(&pps).unwrap();
        let mut nal = vec![0x65, 0xb8, 0x40, 0xa0, 0xd0];
        nal.extend([235; 256]);
        nal.extend([128; 128]);
        nal.push(0x80);
        let mut sample = (nal.len() as u32).to_be_bytes().to_vec();
        sample.extend(nal);
        let index = Mp4Index {
            timescale: 1000,
            duration_ticks: 1000,
            config: AvcConfig {
                nal_length_size: 4,
                sequence_parameters: vec![sps],
                picture_parameter_sets: vec![pps],
            },
            samples: vec![Sample {
                offset: 0,
                size: sample.len() as u32,
                decode_time: 0,
                presentation_time: 0,
                keyframe: true,
            }],
        };
        let mut decoder = Mp4AvcStream::new();
        decoder.index = Some(index);
        let mut frames = Vec::new();
        for chunk in sample.chunks(7) {
            frames.extend(decoder.push(chunk).unwrap());
        }
        decoder.finish().unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].width, frames[0].height), (16, 16));
        assert_eq!(&frames[0].rgba[..4], &[255, 255, 255, 255]);
        assert_eq!(frames[0].timestamp, 0.0);
    }

    #[test]
    fn failed_later_sample_is_not_redecoded_for_each_network_chunk() {
        let sps = parse_sps(&[0x67, 0x42, 0x00, 0x0a, 0xf4, 0xf2]).unwrap();
        let pps = vec![0x68, 0xce, 0x3c, 0x80];
        let mut nal = vec![0x65, 0xb8, 0x40, 0xa0, 0xd0];
        nal.extend([235; 256]);
        nal.extend([128; 128]);
        nal.push(0x80);
        let mut bytes = (nal.len() as u32).to_be_bytes().to_vec();
        bytes.extend(nal);
        let first_size = bytes.len() as u32;
        bytes.extend([0, 0, 0, 1, 0xff]);
        let index = Mp4Index {
            timescale: 1000,
            duration_ticks: 2000,
            config: AvcConfig {
                nal_length_size: 4,
                sequence_parameters: vec![sps],
                picture_parameter_sets: vec![pps],
            },
            samples: vec![
                Sample {
                    offset: 0,
                    size: first_size,
                    decode_time: 0,
                    presentation_time: 0,
                    keyframe: true,
                },
                Sample {
                    offset: u64::from(first_size),
                    size: 5,
                    decode_time: 1000,
                    presentation_time: 1000,
                    keyframe: false,
                },
            ],
        };
        let mut decoder = Mp4AvcStream::new();
        decoder.index = Some(index);
        assert_eq!(decoder.push(&bytes).unwrap().len(), 1);
        let failed_at = decoder.next_sample;
        let buffered = decoder.bytes.len();
        let error = decoder.decode_error.clone().unwrap();
        assert_eq!(decoder.push(&[0; 16]).unwrap_err(), error);
        assert_eq!(decoder.next_sample, failed_at);
        assert_eq!(decoder.bytes.len(), buffered);
    }

    #[test]
    fn decodes_site_prefix_in_presentation_order() {
        let Ok(path) = std::env::var("WEBCORE_MP4_FIXTURE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let last = &index.samples[289];
        let prefix_end = last.offset as usize + last.size as usize;
        let mut decoder = Mp4AvcStream::new();
        assert!(decoder.metadata().is_none());
        let mut rejection = None;
        let mut delivered = Vec::new();
        for chunk in bytes[..prefix_end].chunks(16 * 1024) {
            match decoder.push(chunk) {
                Ok(frames) => delivered.extend(frames),
                Err(error) => {
                    rejection = Some(error);
                    break;
                }
            }
        }
        let metadata = decoder.metadata().unwrap();
        assert_eq!((metadata.width, metadata.height), (Some(1280), Some(720)));
        assert!(metadata.duration.unwrap() > 80.0);
        assert!(delivered.len() >= 5, "delivered {} frames", delivered.len());
        assert_eq!((delivered[0].width, delivered[0].height), (1280, 720));
        assert_eq!(delivered[0].timestamp, 0.0);
        assert!(
            delivered
                .windows(2)
                .all(|pair| pair[0].timestamp <= pair[1].timestamp)
        );
        if let Some((_, pending)) = decoder.pending_pictures.first() {
            assert!(pending.timestamp >= delivered.last().unwrap().timestamp);
        }
        assert!(
            decoder.next_sample >= 290,
            "decoded {} samples; rejection: {rejection:?}",
            decoder.next_sample
        );
        assert!(decoder.base_offset > 0);
        assert!(decoder.bytes.len() < 256 * 1024);
        assert!(decoder.reference_pictures.len() >= 2);
        assert_eq!(
            (
                decoder.reference_pictures[0].width,
                decoder.reference_pictures[0].height
            ),
            (1280, 720)
        );
        assert!(rejection.is_none(), "prefix rejection: {rejection:?}");
    }

    #[test]
    fn decodes_trailing_moov_fixture_first_frame() {
        let Ok(path) = std::env::var("WEBCORE_TRAILING_MOOV_FIXTURE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let sample = &index.samples[9];
        let end = (sample.offset + u64::from(sample.size)) as usize;
        let mut decoder = Mp4AvcStream::new();
        decoder.index = Some(index);
        let mut frames = Vec::new();
        for chunk in bytes[..end].chunks(16 * 1024) {
            frames.extend(decoder.push(chunk).unwrap());
        }
        assert_eq!(decoder.next_sample, 10);
        assert_eq!(decoder.dropped_nonreference_samples, 0);
        assert_eq!(decoder.dropped_until_idr_samples, 0);
        assert!(
            !frames.is_empty(),
            "pending={}, dropped B={}, dropped until IDR={}, next={}, first timestamps={:?}",
            decoder.pending_pictures.len(),
            decoder.dropped_nonreference_samples,
            decoder.dropped_until_idr_samples,
            decoder.next_sample,
            &decoder.index.as_ref().unwrap().samples[..10]
                .iter()
                .map(|sample| sample.presentation_time)
                .collect::<Vec<_>>()
        );
        assert_eq!((frames[0].width, frames[0].height), (1920, 1080));
    }

    #[test]
    fn streams_trailing_moov_fixture_in_bounded_batches() {
        let Ok(path) = std::env::var("WEBCORE_TRAILING_MOOV_FIXTURE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let mut decoder = Mp4AvcStream::new();
        let mut frames = Vec::new();
        for chunk in bytes.chunks(16 * 1024) {
            let batch = decoder.push(chunk).unwrap();
            assert!(batch.len() <= MAX_FRAMES_PER_PUSH + 4);
            frames.extend(batch);
        }
        let metadata = decoder.metadata().unwrap();
        assert_eq!((metadata.width, metadata.height), (Some(1920), Some(1080)));
        assert!(decoder.has_buffered_samples());
        while frames.len() < 4 && decoder.has_buffered_samples() {
            let batch = decoder.push(&[]).unwrap();
            assert!(batch.len() <= MAX_FRAMES_PER_PUSH + 4);
            frames.extend(batch);
        }
        assert!(!frames.is_empty());
        assert_eq!((frames[0].width, frames[0].height), (1920, 1080));
    }

    #[test]
    fn decodes_entire_site_video_in_presentation_order() {
        let Ok(path) = std::env::var("WEBCORE_MP4_FULL_FIXTURE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let mut decoder = Mp4AvcStream::new();
        let mut frames = 0usize;
        let mut last_timestamp = None;
        let mut reported = 0usize;
        let mut last_dropped_nonreference = 0usize;
        let mut last_dropped_until_idr = 0usize;
        for chunk in bytes.chunks(16 * 1024) {
            let decoded = decoder.push(chunk).unwrap_or_else(|error| {
                panic!("failed at sample {}: {error:?}", decoder.next_sample)
            });
            for frame in decoded {
                assert!(last_timestamp.is_none_or(|last| frame.timestamp >= last));
                last_timestamp = Some(frame.timestamp);
                frames += 1;
            }
            if decoder.next_sample / 200 > reported {
                reported = decoder.next_sample / 200;
                eprintln!(
                    "decoded {} / {} samples, frames={}, dropped B={} (+{}), until IDR={} (+{})",
                    decoder.next_sample,
                    index.samples.len(),
                    frames,
                    decoder.dropped_nonreference_samples,
                    decoder.dropped_nonreference_samples - last_dropped_nonreference,
                    decoder.dropped_until_idr_samples,
                    decoder.dropped_until_idr_samples - last_dropped_until_idr,
                );
                last_dropped_nonreference = decoder.dropped_nonreference_samples;
                last_dropped_until_idr = decoder.dropped_until_idr_samples;
            }
        }
        while decoder.has_buffered_samples() {
            let decoded = decoder.push(&[]).unwrap_or_else(|error| {
                panic!("failed at sample {}: {error:?}", decoder.next_sample)
            });
            for frame in decoded {
                assert!(last_timestamp.is_none_or(|last| frame.timestamp >= last));
                last_timestamp = Some(frame.timestamp);
                frames += 1;
            }
        }
        decoder.finish().unwrap();
        assert_eq!(decoder.next_sample, index.samples.len());
        assert_eq!(frames, index.samples.len());
    }

    #[test]
    fn decodes_site_gop_around_sample_722() {
        let Ok(path) = std::env::var("WEBCORE_MP4_FULL_FIXTURE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let target = std::env::var("WEBCORE_MP4_GOP_TARGET")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(722);
        let start = (0..=target)
            .rev()
            .find(|&sample| index.samples[sample].keyframe)
            .unwrap();
        let start_offset = index.samples[start].offset as usize;
        let end_sample = &index.samples[target];
        if std::env::var_os("WEBMEDIA_TRACE_RECOVERY").is_some() {
            eprintln!(
                "target sample {target} from IDR {start} pts={:.3} offset={} size={}",
                end_sample.presentation_time as f64 / index.timescale as f64,
                end_sample.offset,
                end_sample.size
            );
        }
        let end_offset = end_sample.offset as usize + end_sample.size as usize;
        let mut decoder = Mp4AvcStream::new();
        decoder.index = Some(index);
        decoder.next_sample = start;
        decoder.base_offset = start_offset as u64;
        decoder.push(&bytes[start_offset..end_offset]).unwrap();
        while decoder.next_sample <= target && decoder.has_buffered_samples() {
            decoder.push(&[]).unwrap();
        }
        assert_eq!(
            decoder.next_sample,
            target + 1,
            "target sample error: {:?}",
            decoder.decode_error
        );
        assert_eq!(decoder.dropped_nonreference_samples, 0);
        assert_eq!(decoder.dropped_until_idr_samples, 0);
    }

    #[test]
    fn site_target_reference_picture_error_profile() {
        let (Ok(fixture), Ok(raw), Ok(target)) = (
            std::env::var("WEBCORE_MP4_FULL_FIXTURE"),
            std::env::var("WEBCORE_H264_TARGET_REFERENCE_YUV"),
            std::env::var("WEBCORE_H264_TARGET_SAMPLE"),
        ) else {
            return;
        };
        let target: usize = target.parse().unwrap();
        let bytes = std::fs::read(fixture).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let sample = &index.samples[target];
        let start = (0..=target)
            .rev()
            .find(|&number| index.samples[number].keyframe)
            .unwrap();
        let presentation_index = index
            .samples
            .iter()
            .filter(|other| other.presentation_time < sample.presentation_time)
            .count();
        eprintln!("sample {target} presentation index {presentation_index} from keyframe {start}");
        let start_offset = index.samples[start].offset as usize;
        let end_offset = sample.offset as usize + sample.size as usize;
        let mut decoder = Mp4AvcStream::new();
        decoder.index = Some(index.clone());
        decoder.next_sample = start;
        decoder.base_offset = start_offset as u64;
        let preceding = if target == start {
            start_offset
        } else {
            let previous = &index.samples[target - 1];
            previous.offset as usize + previous.size as usize
        };
        decoder.push(&bytes[start_offset..preceding]).unwrap();
        assert_eq!(decoder.next_sample, target);
        let (actual, picture, _, _) = Mp4AvcStream::decode_sample(
            &index,
            &bytes[start_offset..end_offset],
            start_offset as u64,
            target,
            &decoder.reference_pictures,
        )
        .unwrap();
        let expected = std::fs::read(raw).unwrap();
        let frame_bytes = actual.width as usize * actual.height as usize * 3 / 2;
        assert_eq!(expected.len(), frame_bytes);
        let y_end = actual.width as usize * actual.height as usize;
        let cb_end = y_end + y_end / 4;
        if let Some(picture) = picture.as_ref() {
            for (name, actual_plane, expected_plane, plane_width, plane_height) in [
                (
                    "Y",
                    &picture.luma,
                    &expected[..y_end],
                    actual.width as usize,
                    actual.height as usize,
                ),
                (
                    "Cb",
                    &picture.cb,
                    &expected[y_end..cb_end],
                    actual.width as usize / 2,
                    actual.height as usize / 2,
                ),
                (
                    "Cr",
                    &picture.cr,
                    &expected[cb_end..],
                    actual.width as usize / 2,
                    actual.height as usize / 2,
                ),
            ] {
                let error: u64 = actual_plane
                    .chunks_exact(plane_width)
                    .take(plane_height)
                    .flatten()
                    .zip(expected_plane.iter())
                    .map(|(a, b)| u64::from(a.abs_diff(*b)))
                    .sum();
                eprintln!(
                    "sample {target} {name} MAE={:.3}",
                    error as f64 / expected_plane.len() as f64
                );
            }
        }
        let reference = frame_from_yuv420(
            &index.config.sequence_parameters[0],
            &expected[..y_end],
            &expected[y_end..cb_end],
            &expected[cb_end..],
        );
        if let Ok(path) = std::env::var("WEBCORE_H264_TARGET_DUMP_RGBA") {
            std::fs::write(path, &*actual.rgba).unwrap();
        }
        let error: u64 = actual
            .rgba
            .chunks_exact(4)
            .zip(reference.rgba.chunks_exact(4))
            .map(|(a, b)| {
                (0..3)
                    .map(|channel| u64::from(a[channel].abs_diff(b[channel])))
                    .sum::<u64>()
            })
            .sum();
        let mae = error as f64 / (y_end * 3) as f64;
        eprintln!(
            "sample {target} pts={:.3} RGB MAE={:.3}",
            sample.presentation_time as f64 / index.timescale as f64,
            mae
        );
        if let Ok(limit) = std::env::var("WEBCORE_H264_TARGET_MAX_RGB_MAE") {
            let limit: f64 = limit.parse().unwrap();
            assert!(mae < limit, "sample {target} RGB MAE {mae} exceeds {limit}");
        }
        let mut blocks = Vec::new();
        for my in 0..actual.height as usize / 16 {
            for mx in 0..actual.width as usize / 16 {
                let mut error = 0u64;
                for row in 0..16 {
                    for col in 0..16 {
                        let at = ((my * 16 + row) * actual.width as usize + mx * 16 + col) * 4;
                        error += u64::from(actual.rgba[at].abs_diff(reference.rgba[at]));
                    }
                }
                blocks.push((error, mx, my));
            }
        }
        blocks.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        eprintln!("worst luma macroblocks: {:?}", &blocks[..12]);
        if std::env::var_os("WEBMEDIA_TRACE_ERROR_MAP").is_some() {
            if let Some(picture) = picture.as_ref() {
                for &(_, mx, my) in blocks.iter().take(5) {
                    let index = my * (picture.width / 16) + mx;
                    eprintln!("motion ({mx},{my}): {:?}", picture.motion[index]);
                }
            }
        }
    }

    #[test]
    fn site_gop_frame_error_profile() {
        let (Ok(fixture), Ok(raw), Ok(first), Ok(last)) = (
            std::env::var("WEBCORE_MP4_FULL_FIXTURE"),
            std::env::var("WEBCORE_H264_GOP_REFERENCE_YUV"),
            std::env::var("WEBCORE_H264_GOP_FIRST_FRAME"),
            std::env::var("WEBCORE_H264_GOP_LAST_SAMPLE"),
        ) else {
            return;
        };
        let first: usize = first.parse().unwrap();
        let last: usize = last.parse().unwrap();
        let bytes = std::fs::read(fixture).unwrap();
        let raw = std::fs::read(raw).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let mut presentation_times: Vec<_> = index
            .samples
            .iter()
            .map(|sample| sample.presentation_time)
            .collect();
        presentation_times.sort_unstable();
        let start = (0..=last)
            .rev()
            .find(|&sample| index.samples[sample].keyframe)
            .unwrap();
        let width = index.config.sequence_parameters[0].width as usize;
        let height = index.config.sequence_parameters[0].height as usize;
        let y_size = width * height;
        let frame_size = y_size * 3 / 2;
        let start_offset = index.samples[start].offset as usize;
        let end = &index.samples[last];
        let end_offset = end.offset as usize + end.size as usize;
        let mut decoder = Mp4AvcStream::new();
        decoder.index = Some(index.clone());
        decoder.next_sample = start;
        decoder.base_offset = start_offset as u64;
        let mut frames = decoder.push(&bytes[start_offset..end_offset]).unwrap();
        while decoder.next_sample <= last && decoder.has_buffered_samples() {
            frames.extend(decoder.push(&[]).unwrap());
        }
        for frame in frames {
            let presentation_time = (frame.timestamp * index.timescale as f32).round() as i64;
            let number = presentation_times.partition_point(|&time| time < presentation_time);
            if number < first {
                continue;
            }
            let offset = (number - first) * frame_size;
            let Some(expected) = raw.get(offset..offset + frame_size) else {
                continue;
            };
            let reference = frame_from_yuv420(
                &index.config.sequence_parameters[0],
                &expected[..y_size],
                &expected[y_size..y_size + y_size / 4],
                &expected[y_size + y_size / 4..],
            );
            let error: u64 = frame
                .rgba
                .chunks_exact(4)
                .zip(reference.rgba.chunks_exact(4))
                .map(|(a, b)| {
                    (0..3)
                        .map(|channel| u64::from(a[channel].abs_diff(b[channel])))
                        .sum::<u64>()
                })
                .sum();
            eprintln!(
                "frame {number} t={:.3} RGB MAE={:.3}",
                frame.timestamp,
                error as f64 / (y_size * 3) as f64
            );
        }
    }

    #[test]
    fn site_samples_770_and_771_have_distinct_picture_headers() {
        let Ok(path) = std::env::var("WEBCORE_MP4_FULL_FIXTURE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let sps = &index.config.sequence_parameters[0];
        let pps = parse_pps_2005(&index.config.picture_parameter_sets[0]).unwrap();
        for sample_number in [770, 771] {
            let sample = &index.samples[sample_number];
            let start = sample.offset as usize;
            let mut nals = NalStream::new(index.config.nal_length_size).unwrap();
            let units = nals
                .push(&bytes[start..start + sample.size as usize])
                .unwrap();
            nals.finish().unwrap();
            let headers: Vec<_> = units
                .iter()
                .filter(|nal| nal[0] & 0x1f == 1)
                .map(|nal| parse_cabac_inter_slice(nal, sps, &pps.core).unwrap())
                .collect();
            eprintln!(
                "sample {sample_number}: pts={} nals={} ref_idc={:?} headers={:?}",
                sample.presentation_time,
                units.len(),
                units
                    .iter()
                    .map(|nal| (nal[0] >> 5) & 3)
                    .collect::<Vec<_>>(),
                headers
                    .iter()
                    .map(|h| (h.first_mb, h.frame_num, h.pic_order_cnt_lsb))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn site_first_ten_frame_pixel_error_profile() {
        let (Ok(fixture), Ok(raw)) = (
            std::env::var("WEBCORE_MP4_FIXTURE"),
            std::env::var("WEBCORE_H264_FIRST_TEN_REFERENCE_YUV"),
        ) else {
            return;
        };
        let bytes = std::fs::read(fixture).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let sps = &index.config.sequence_parameters[0];
        let mut decoder = Mp4AvcStream::new();
        let frames = decoder.push(&bytes).unwrap();
        let raw = std::fs::read(raw).unwrap();
        let frame_bytes = sps.width as usize * sps.height as usize * 3 / 2;
        assert_eq!(raw.len(), 10 * frame_bytes);
        assert!(frames.len() >= 10);
        for (number, actual) in frames.iter().take(10).enumerate() {
            let expected = &raw[number * frame_bytes..(number + 1) * frame_bytes];
            let y_end = sps.width as usize * sps.height as usize;
            let chroma_end = y_end + y_end / 4;
            let expected = frame_from_yuv420(
                sps,
                &expected[..y_end],
                &expected[y_end..chroma_end],
                &expected[chroma_end..],
            );
            let error: u64 = actual
                .rgba
                .chunks_exact(4)
                .zip(expected.rgba.chunks_exact(4))
                .map(|(a, b)| {
                    u64::from(a[0].abs_diff(b[0]))
                        + u64::from(a[1].abs_diff(b[1]))
                        + u64::from(a[2].abs_diff(b[2]))
                })
                .sum();
            eprintln!(
                "presentation frame {number}: t={:.3} RGB MAE={:.3}",
                actual.timestamp,
                error as f64 / (y_end * 3) as f64
            );
        }
    }

    #[test]
    fn site_idr_slice_reaches_original_cabac_engine() {
        let Ok(path) = std::env::var("WEBCORE_MP4_FIXTURE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let pps = parse_pps_2005(&index.config.picture_parameter_sets[0]).unwrap();
        assert!(pps.core.cabac);
        assert!(!index.config.sequence_parameters[0].scaling_matrices_present);
        assert!(!pps.scaling_matrices_present);
        let sample = &index.samples[0];
        let start = sample.offset as usize;
        let end = start + sample.size as usize;
        let mut nals = NalStream::new(index.config.nal_length_size).unwrap();
        let units = nals.push(&bytes[start..end]).unwrap();
        nals.finish().unwrap();
        let idr = units.iter().find(|nal| nal[0] & 0x1f == 5).unwrap();
        let slice =
            parse_cabac_idr_i_slice(idr, &index.config.sequence_parameters[0], &pps.core).unwrap();
        assert_eq!(slice.first_mb, 0);
        assert!((0..=51).contains(&slice.slice_qp));
        let mut decoder = CabacDecoder::new(&slice.rbsp[slice.data_byte_offset..]).unwrap();
        assert_eq!(decoder.consumed_bits(), 9);
        let mut contexts = IntraMbTypeContexts::new(slice.slice_qp).unwrap();
        let first_mb_type = decoder.intra_mb_type(&mut contexts, None, None).unwrap();
        assert_eq!(first_mb_type, 0);
        let mut transform_contexts = Transform8x8Contexts::new(slice.slice_qp).unwrap();
        let transform_8x8 = if pps.transform_8x8 {
            decoder
                .transform_size_8x8_flag(&mut transform_contexts, None, None)
                .unwrap()
        } else {
            false
        };
        let mut prediction_contexts = IntraPredContexts::new(slice.slice_qp).unwrap();
        let block_count = if transform_8x8 { 4 } else { 16 };
        let prediction_codes: Vec<_> = (0..block_count)
            .map(|_| {
                decoder
                    .intra_luma_pred_code(&mut prediction_contexts)
                    .unwrap()
            })
            .collect();
        let luma_modes =
            intra4x4_modes(prediction_codes.as_slice().try_into().unwrap(), None, None).unwrap();
        let chroma_mode = decoder
            .intra_chroma_pred_mode(&mut prediction_contexts, None, None)
            .unwrap();
        let mut coded_contexts = CodedBlockContexts::new(slice.slice_qp).unwrap();
        let coded = decoder
            .coded_block_pattern(&mut coded_contexts, None, None)
            .unwrap();
        let mut qp_contexts = MbQpContexts::new(slice.slice_qp).unwrap();
        let qp_delta = decoder.mb_qp_delta(&mut qp_contexts, false).unwrap();
        let mut luma_contexts = Luma4x4Contexts::new(slice.slice_qp).unwrap();
        let luma_blocks = decoder
            .luma4x4_macroblock(&mut luma_contexts, coded.luma, None, None)
            .unwrap();
        let mut chroma_contexts = ChromaDcContexts::new(slice.slice_qp).unwrap();
        let cb_dc = decoder
            .chroma_dc_coefficients(&mut chroma_contexts, None, None)
            .unwrap();
        let cr_dc = decoder
            .chroma_dc_coefficients(&mut chroma_contexts, None, None)
            .unwrap();
        assert_eq!(prediction_codes.len(), block_count);
        assert!(!transform_8x8);
        assert_eq!(chroma_mode, 0);
        assert_eq!((coded.luma, coded.chroma), (3, 1));
        assert_eq!(qp_delta, 0);
        assert_eq!(luma_modes[0], 2);
        let first_coefficients = inverse_4x4_frame_scan(&luma_blocks[0]);
        let first_residual =
            inverse_4x4_residual(&first_coefficients, slice.slice_qp + qp_delta, false).unwrap();
        assert!(
            first_residual
                .iter()
                .all(|&sample| sample == first_residual[0])
        );
        assert!(first_residual[0] < 0);
        let first_macroblock = reconstruct_intra4x4_luma(
            &luma_modes,
            &luma_blocks,
            slice.slice_qp + qp_delta,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            first_macroblock[0],
            (128 + first_residual[0]).clamp(0, 255) as u8
        );
        assert!(
            first_macroblock
                .iter()
                .any(|&pixel| pixel != first_macroblock[0])
        );
        assert_eq!(
            luma_blocks[0],
            [-47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert!(
            luma_blocks[8..]
                .iter()
                .all(|block| block.iter().all(|&level| level == 0))
        );
        assert_eq!(cb_dc, [-1, 0, 0, 0]);
        assert_eq!(cr_dc, [-1, 0, 0, 0]);
        let cb_qp = chroma_qp(slice.slice_qp + qp_delta, pps.core.chroma_qp_index_offset).unwrap();
        let cr_qp =
            chroma_qp(slice.slice_qp + qp_delta, pps.second_chroma_qp_index_offset).unwrap();
        assert_eq!(inverse_chroma_dc(&cb_dc, cb_qp).unwrap().len(), 4);
        assert_eq!(inverse_chroma_dc(&cr_dc, cr_qp).unwrap().len(), 4);
        let cb_pixels = reconstruct_chroma_dc_only(&cb_dc, cb_qp, None, None).unwrap();
        let cr_pixels = reconstruct_chroma_dc_only(&cr_dc, cr_qp, None, None).unwrap();
        assert!(cb_pixels.iter().all(|&pixel| pixel == cb_pixels[0]));
        assert!(cr_pixels.iter().all(|&pixel| pixel == cr_pixels[0]));
        assert!(!decoder.terminate().unwrap());
        let second_mb_type = decoder
            .intra_mb_type(&mut contexts, Some(first_mb_type), None)
            .unwrap();
        assert_eq!(second_mb_type, 0);
        let transform_8x8_2 = if pps.transform_8x8 {
            decoder
                .transform_size_8x8_flag(&mut transform_contexts, Some(transform_8x8), None)
                .unwrap()
        } else {
            false
        };
        assert!(transform_8x8_2);
        let prediction_codes_2: Vec<_> = (0..4)
            .map(|_| {
                decoder
                    .intra_luma_pred_code(&mut prediction_contexts)
                    .unwrap()
            })
            .collect();
        let left_modes = [luma_modes[5], luma_modes[13]];
        let luma_modes_2 = intra8x8_modes(
            prediction_codes_2.as_slice().try_into().unwrap(),
            Some(left_modes),
            None,
        )
        .unwrap();
        let chroma_mode_2 = decoder
            .intra_chroma_pred_mode(&mut prediction_contexts, Some(chroma_mode), None)
            .unwrap();
        let coded_2 = decoder
            .coded_block_pattern(&mut coded_contexts, Some(coded), None)
            .unwrap();
        let qp_delta_2 = if coded_2.luma != 0 || coded_2.chroma != 0 {
            decoder.mb_qp_delta(&mut qp_contexts, false).unwrap()
        } else {
            0
        };
        let mut luma8x8_contexts = Luma8x8Contexts::new(slice.slice_qp).unwrap();
        let mut luma8x8_blocks = [[0i32; 64]; 4];
        for (region, block) in luma8x8_blocks.iter_mut().enumerate() {
            if coded_2.luma & (1 << region) != 0 {
                *block = decoder.luma8x8_coefficients(&mut luma8x8_contexts).unwrap();
            }
        }
        assert_eq!(chroma_mode_2, 0);
        assert_eq!((coded_2.luma, coded_2.chroma), (15, 0));
        assert_eq!(qp_delta_2, 0);
        assert_eq!(luma_modes_2, [2, 2, 2, 8]);
        assert_eq!(&luma8x8_blocks[0][..8], &[11, -3, -1, 0, -1, 0, -1, 0]);
        let first_8x8_coefficients = inverse_8x8_frame_scan(&luma8x8_blocks[0]);
        let first_8x8_residual = inverse_8x8_residual(
            &first_8x8_coefficients,
            slice.slice_qp + qp_delta + qp_delta_2,
            &[16; 64],
        )
        .unwrap();
        assert!(first_8x8_residual.iter().any(|&sample| sample != 0));
        let second_macroblock = reconstruct_intra8x8_luma(
            &luma_modes_2,
            &luma8x8_blocks,
            slice.slice_qp + qp_delta + qp_delta_2,
            &[16; 64],
            None,
            Some(std::array::from_fn(|y| first_macroblock[y * 16 + 15])),
            None,
        )
        .unwrap();
        assert!(
            second_macroblock
                .iter()
                .any(|&pixel| pixel != second_macroblock[0])
        );
        let cb_pixels_2 = reconstruct_chroma_dc_only(
            &[0; 4],
            cb_qp,
            None,
            Some(std::array::from_fn(|y| cb_pixels[y * 8 + 7])),
        )
        .unwrap();
        let cr_pixels_2 = reconstruct_chroma_dc_only(
            &[0; 4],
            cr_qp,
            None,
            Some(std::array::from_fn(|y| cr_pixels[y * 8 + 7])),
        )
        .unwrap();
        assert!(!decoder.terminate().unwrap());
        let third_mb_type = decoder
            .intra_mb_type(&mut contexts, Some(second_mb_type), None)
            .unwrap();
        assert_eq!(third_mb_type, 0);
        let transform_8x8_3 = if pps.transform_8x8 {
            decoder
                .transform_size_8x8_flag(&mut transform_contexts, Some(transform_8x8_2), None)
                .unwrap()
        } else {
            false
        };
        let prediction_codes_3: Vec<_> = (0..if transform_8x8_3 { 4 } else { 16 })
            .map(|_| {
                decoder
                    .intra_luma_pred_code(&mut prediction_contexts)
                    .unwrap()
            })
            .collect();
        let chroma_mode_3 = decoder
            .intra_chroma_pred_mode(&mut prediction_contexts, Some(chroma_mode_2), None)
            .unwrap();
        let coded_3 = decoder
            .coded_block_pattern(&mut coded_contexts, Some(coded_2), None)
            .unwrap();
        let qp_delta_3 = if coded_3.luma != 0 || coded_3.chroma != 0 {
            decoder.mb_qp_delta(&mut qp_contexts, false).unwrap()
        } else {
            0
        };
        assert!(transform_8x8_3);
        let luma_modes_3 = intra8x8_modes(
            prediction_codes_3.as_slice().try_into().unwrap(),
            Some([luma_modes_2[1], luma_modes_2[3]]),
            None,
        )
        .unwrap();
        let mut luma8x8_blocks_3 = [[0i32; 64]; 4];
        for (region, block) in luma8x8_blocks_3.iter_mut().enumerate() {
            if coded_3.luma & (1 << region) != 0 {
                *block = decoder.luma8x8_coefficients(&mut luma8x8_contexts).unwrap();
            }
        }
        let cb_dc_3 = decoder
            .chroma_dc_coefficients(&mut chroma_contexts, Some(false), None)
            .unwrap();
        let cr_dc_3 = decoder
            .chroma_dc_coefficients(&mut chroma_contexts, Some(false), None)
            .unwrap();
        assert_eq!(luma_modes_3, [2, 2, 8, 0]);
        assert_eq!(chroma_mode_3, 0);
        assert_eq!((coded_3.luma, coded_3.chroma), (15, 1));
        assert_eq!(qp_delta_3, 0);
        assert_eq!(cb_dc_3, [0; 4]);
        assert_eq!(cr_dc_3, [1, 0, 0, 0]);
        let third_macroblock = reconstruct_intra8x8_luma(
            &luma_modes_3,
            &luma8x8_blocks_3,
            slice.slice_qp + qp_delta + qp_delta_2 + qp_delta_3,
            &[16; 64],
            None,
            Some(std::array::from_fn(|y| second_macroblock[y * 16 + 15])),
            None,
        )
        .unwrap();
        assert!(
            third_macroblock
                .iter()
                .any(|&pixel| pixel != third_macroblock[0])
        );
        let cb_pixels_3 = reconstruct_chroma_dc_only(
            &cb_dc_3,
            cb_qp,
            None,
            Some(std::array::from_fn(|y| cb_pixels_2[y * 8 + 7])),
        )
        .unwrap();
        let cr_pixels_3 = reconstruct_chroma_dc_only(
            &cr_dc_3,
            cr_qp,
            None,
            Some(std::array::from_fn(|y| cr_pixels_2[y * 8 + 7])),
        )
        .unwrap();
        assert_eq!(cb_pixels_3.len(), 64);
        assert_eq!(cr_pixels_3.len(), 64);
        assert!(!decoder.terminate().unwrap());
        let fourth_mb_type = decoder
            .intra_mb_type(&mut contexts, Some(third_mb_type), None)
            .unwrap();
        assert_eq!(fourth_mb_type, 0);
    }

    #[test]
    fn site_high_profile_idr_reconstructs_full_picture() {
        let Ok(path) = std::env::var("WEBCORE_MP4_FIXTURE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let sample = &index.samples[0];
        let start = sample.offset as usize;
        let mut nals = NalStream::new(index.config.nal_length_size).unwrap();
        let units = nals
            .push(&bytes[start..start + sample.size as usize])
            .unwrap();
        nals.finish().unwrap();
        let idr = units.iter().find(|nal| nal[0] & 0x1f == 5).unwrap();
        let pps = parse_pps_2005(&index.config.picture_parameter_sets[0]).unwrap();
        let frame = decode_cabac_idr_2005(idr, &index.config.sequence_parameters[0], &pps).unwrap();
        assert_eq!((frame.width, frame.height), (1280, 720));
        assert_eq!(frame.rgba.len(), 1280 * 720 * 4);
        assert!(frame.rgba.chunks_exact(4).all(|pixel| pixel[3] == 255));
        assert!(
            frame
                .rgba
                .chunks_exact(4)
                .any(|pixel| pixel[0] != frame.rgba[0])
        );
        if let Ok(path) = std::env::var("WEBCORE_H264_REFERENCE_RGBA") {
            let reference = std::fs::read(path).unwrap();
            assert_eq!(frame.rgba.len(), reference.len());
            let mut error = 0u64;
            for (actual, expected) in frame.rgba.chunks_exact(4).zip(reference.chunks_exact(4)) {
                for channel in 0..3 {
                    error += u64::from(actual[channel].abs_diff(expected[channel]));
                }
            }
            let mean_error = error as f64 / (frame.width * frame.height * 3) as f64;
            assert!(mean_error < 3.0, "IDR mean RGB error: {mean_error}");
        }
    }

    #[test]
    fn site_inter_picture_headers_follow_2003_syntax() {
        let Ok(path) = std::env::var("WEBCORE_MP4_FIXTURE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let pps = parse_pps_2005(&index.config.picture_parameter_sets[0]).unwrap();
        let mut kinds = Vec::new();
        for sample in index.samples.iter().skip(1).take(4) {
            let start = sample.offset as usize;
            let end = start + sample.size as usize;
            let mut stream = NalStream::new(index.config.nal_length_size).unwrap();
            let nals = stream.push(&bytes[start..end]).unwrap();
            stream.finish().unwrap();
            let nal = nals.iter().find(|nal| nal[0] & 0x1f == 1).unwrap();
            let slice =
                parse_cabac_inter_slice(nal, &index.config.sequence_parameters[0], &pps.core)
                    .unwrap();
            assert_eq!(slice.first_mb, 0);
            assert!(slice.data_byte_offset < slice.rbsp.len());
            kinds.push(slice.slice_type % 5);
        }
        assert_eq!(kinds, [0, 1, 1, 1]);
    }

    #[test]
    fn site_first_b_header_and_skip_flag() {
        let Ok(path) = std::env::var("WEBCORE_MP4_FIXTURE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let pps = parse_pps_2005(&index.config.picture_parameter_sets[0]).unwrap();
        let sample = &index.samples[2];
        assert!(index.config.sequence_parameters[0].direct_8x8_inference);
        assert!(!pps.core.pic_order_present);
        assert_eq!(sample.presentation_time, 5400);
        assert_eq!(index.timescale, 90000);
        let start = sample.offset as usize;
        let mut stream = NalStream::new(index.config.nal_length_size).unwrap();
        let nals = stream
            .push(&bytes[start..start + sample.size as usize])
            .unwrap();
        stream.finish().unwrap();
        let nal = nals.iter().find(|nal| nal[0] & 0x1f == 1).unwrap();
        let slice =
            parse_cabac_inter_slice(nal, &index.config.sequence_parameters[0], &pps.core).unwrap();
        assert_eq!(slice.slice_type % 5, 1);
        assert_eq!(slice.first_mb, 0);
        let mut decoder = CabacDecoder::new(&slice.rbsp[slice.data_byte_offset..]).unwrap();
        let mut contexts =
            InterMbContexts::new(slice.slice_qp, slice.cabac_init_idc, true).unwrap();
        let skipped = decoder
            .inter_mb_skip_flag(&mut contexts, None, None)
            .unwrap();
        assert_eq!(nal[0] >> 5, 2);
        assert_eq!((slice.ref_idx_l0, slice.ref_idx_l1), (1, 1));
        assert!(slice.direct_spatial_mv_pred);
        assert_eq!(slice.slice_qp, 28);
        assert!(!skipped);
        let kind = decoder.b_inter_mb_type(&mut contexts, None, None).unwrap();
        assert_eq!(kind, 2);
        let mut motion = MotionVectorContexts::new(slice.slice_qp, slice.cabac_init_idc).unwrap();
        let dx = decoder
            .motion_vector_difference(&mut motion, 0, None, None)
            .unwrap();
        let dy = decoder
            .motion_vector_difference(&mut motion, 1, None, None)
            .unwrap();
        let mut coded =
            CodedBlockContexts::new_inter(slice.slice_qp, slice.cabac_init_idc).unwrap();
        let pattern = decoder.coded_block_pattern(&mut coded, None, None).unwrap();
        assert_eq!((dx, dy), (-3, 0));
        assert_eq!((pattern.luma, pattern.chroma), (0, 0));
        assert!(!decoder.terminate().unwrap());
        let next_skipped = decoder
            .inter_mb_skip_flag(&mut contexts, Some(false), None)
            .unwrap();
        assert!(next_skipped);
        if let Ok(path) = std::env::var("WEBCORE_H264_B_REFERENCE_YUV") {
            let (_, Some(idr), _, _) =
                Mp4AvcStream::decode_sample(&index, &bytes, 0, 0, &[]).unwrap()
            else {
                panic!("IDR sample did not yield a reference picture");
            };
            let (_, Some(future), _, _) =
                Mp4AvcStream::decode_sample(&index, &bytes, 0, 1, std::slice::from_ref(&idr))
                    .unwrap()
            else {
                panic!("P sample did not yield a reference picture");
            };
            assert_eq!(future.pic_order_cnt, 8);
            assert_eq!(
                type0_pic_order_count(
                    slice.pic_order_cnt_lsb,
                    index.config.sequence_parameters[0]
                        .pic_order_cnt_lsb_bits
                        .unwrap(),
                    Some((future.pic_order_cnt_msb, future.pic_order_cnt_lsb)),
                )
                .unwrap()
                .1,
                4
            );
            let references = [idr, future];
            let (list0, list1) = b_reference_lists(&references, 4);
            assert_eq!(
                (list0.as_slice(), list1.as_slice()),
                ([0, 1].as_slice(), [1, 0].as_slice())
            );
            let future = &references[list1[0]];
            let block =
                predict_l1_16x16(future, 0, 0, [dx, dy], slice.weights.as_ref(), 0).unwrap();
            let expected = std::fs::read(path).unwrap();
            let frame_bytes = future.width * future.height * 3 / 2;
            assert_eq!(expected.len(), 5 * frame_bytes);
            let presentation_index = index
                .samples
                .iter()
                .take(5)
                .filter(|other| other.presentation_time < sample.presentation_time)
                .count();
            assert_eq!(presentation_index, 2);
            let frame =
                &expected[presentation_index * frame_bytes..(presentation_index + 1) * frame_bytes];
            let mut luma_error = 0u32;
            for row in 0..16 {
                for col in 0..16 {
                    luma_error += u32::from(
                        block.luma[row * 16 + col].abs_diff(frame[row * future.width + col]),
                    );
                }
            }
            assert!(luma_error < 1024, "first B luma block error: {luma_error}");
            let direct = spatial_direct_motion(
                Some(crate::video::h264_high::MotionCell {
                    l0: None,
                    l1: Some((0, [dx, dy])),
                }),
                None,
                None,
                None,
                future.motion[1][0],
                true,
            );
            assert_eq!(direct.l0, None);
            let (ref_index, vector) = direct.l1.unwrap();
            let skipped_block = predict_l1_16x16(
                future,
                1,
                0,
                vector,
                slice.weights.as_ref(),
                ref_index as usize,
            )
            .unwrap();
            let mut skipped_error = 0u32;
            for row in 0..16 {
                for col in 0..16 {
                    skipped_error += u32::from(
                        skipped_block.luma[row * 16 + col]
                            .abs_diff(frame[row * future.width + 16 + col]),
                    );
                }
            }
            assert!(
                skipped_error < 1024,
                "second B luma block error: {skipped_error}"
            );
        }
    }

    #[test]
    fn site_first_b_picture_matches_reference_pixels() {
        let (Ok(fixture), Ok(raw)) = (
            std::env::var("WEBCORE_MP4_FIXTURE"),
            std::env::var("WEBCORE_H264_B_REFERENCE_YUV"),
        ) else {
            return;
        };
        let bytes = std::fs::read(fixture).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let (_, Some(idr), _, _) = Mp4AvcStream::decode_sample(&index, &bytes, 0, 0, &[]).unwrap()
        else {
            panic!("IDR sample did not yield a reference picture");
        };
        let (_, Some(p), _, _) =
            Mp4AvcStream::decode_sample(&index, &bytes, 0, 1, std::slice::from_ref(&idr)).unwrap()
        else {
            panic!("P sample did not yield a reference picture");
        };
        let sample = &index.samples[2];
        let start = sample.offset as usize;
        let mut stream = NalStream::new(index.config.nal_length_size).unwrap();
        let units = stream
            .push(&bytes[start..start + sample.size as usize])
            .unwrap();
        stream.finish().unwrap();
        let nal = units.iter().find(|nal| nal[0] & 0x1f == 1).unwrap();
        let pps = parse_pps_2005(&index.config.picture_parameter_sets[0]).unwrap();
        let picture =
            decode_cabac_b_2005(nal, &index.config.sequence_parameters[0], &pps, &[idr, p])
                .unwrap();
        let expected = std::fs::read(raw).unwrap();
        let frame_bytes = picture.width * picture.height * 3 / 2;
        assert_eq!(expected.len(), 5 * frame_bytes);
        let presentation_index = index
            .samples
            .iter()
            .take(5)
            .filter(|other| other.presentation_time < sample.presentation_time)
            .count();
        let expected =
            &expected[presentation_index * frame_bytes..(presentation_index + 1) * frame_bytes];
        let actual = [&picture.luma[..], &picture.cb[..], &picture.cr[..]].concat();
        let error: u64 = actual
            .iter()
            .zip(expected)
            .map(|(&a, &b)| u64::from(a.abs_diff(b)))
            .sum();
        let mean_error = error as f64 / frame_bytes as f64;
        assert!(mean_error < 5.0, "first B mean YUV error: {mean_error}");
    }

    #[test]
    fn site_multi_reference_p_picture_matches_reference_pixels() {
        let (Ok(fixture), Ok(raw)) = (
            std::env::var("WEBCORE_MP4_FIXTURE"),
            std::env::var("WEBCORE_H264_P2_REFERENCE_YUV"),
        ) else {
            return;
        };
        let bytes = std::fs::read(fixture).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let sample = &index.samples[5];
        let end = sample.offset as usize + sample.size as usize;
        let mut decoder = Mp4AvcStream::new();
        decoder.index = Some(index.clone());
        decoder.push(&bytes[..end]).unwrap();
        assert_eq!(decoder.next_sample, 6);
        let picture = decoder.reference_pictures.last().unwrap();
        let frame_bytes = picture.width * picture.height * 3 / 2;
        let presentation_index = index
            .samples
            .iter()
            .filter(|other| other.presentation_time < sample.presentation_time)
            .count();
        let raw = std::fs::read(raw).unwrap();
        let expected = &raw[presentation_index * frame_bytes..][..frame_bytes];
        let absolute_error: u64 = picture
            .luma
            .iter()
            .zip(expected)
            .map(|(actual, expected)| u64::from(actual.abs_diff(*expected)))
            .sum();
        let mae = absolute_error as f64 / picture.luma.len() as f64;
        assert!(mae < 5.0, "multi-reference P luma MAE: {mae}");
    }

    #[test]
    fn site_b_intra_picture_matches_reference_pixels() {
        let (Ok(fixture), Ok(raw)) = (
            std::env::var("WEBCORE_MP4_FIXTURE"),
            std::env::var("WEBCORE_H264_B2_REFERENCE_YUV"),
        ) else {
            return;
        };
        let bytes = std::fs::read(fixture).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let sample = &index.samples[6];
        let end = sample.offset as usize + sample.size as usize;
        let mut decoder = Mp4AvcStream::new();
        decoder.index = Some(index.clone());
        decoder.push(&bytes[..end]).unwrap();
        assert_eq!(decoder.next_sample, 7);
        let picture = decoder.reference_pictures.last().unwrap();
        let frame_bytes = picture.width * picture.height * 3 / 2;
        let presentation_index = index
            .samples
            .iter()
            .filter(|other| other.presentation_time < sample.presentation_time)
            .count();
        let raw = std::fs::read(raw).unwrap();
        let expected = &raw[presentation_index * frame_bytes..][..frame_bytes];
        let absolute_error: u64 = picture
            .luma
            .iter()
            .zip(expected)
            .map(|(actual, expected)| u64::from(actual.abs_diff(*expected)))
            .sum();
        let mae = absolute_error as f64 / picture.luma.len() as f64;
        assert!(mae < 5.0, "B intra luma MAE: {mae}");
    }

    #[test]
    fn site_later_reference_picture_matches_reference_pixels() {
        let (Ok(fixture), Ok(raw)) = (
            std::env::var("WEBCORE_MP4_FIXTURE"),
            std::env::var("WEBCORE_H264_LATER_REFERENCE_YUV"),
        ) else {
            return;
        };
        let bytes = std::fs::read(fixture).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let sample = &index.samples[100];
        assert_eq!(
            sample.presentation_time as f64 / index.timescale as f64,
            2.0
        );
        let end = sample.offset as usize + sample.size as usize;
        let mut decoder = Mp4AvcStream::new();
        decoder.push(&bytes[..end]).unwrap();
        assert_eq!(decoder.next_sample, 101);
        let picture = decoder.reference_pictures.last().unwrap();
        if let Ok(path) = std::env::var("WEBCORE_H264_DUMP_RGBA") {
            let frame = frame_from_yuv420(
                &index.config.sequence_parameters[0],
                &picture.luma,
                &picture.cb,
                &picture.cr,
            );
            std::fs::write(path, &*frame.rgba).unwrap();
        }
        let expected = std::fs::read(raw).unwrap();
        let frame_bytes = picture.width * picture.height * 3 / 2;
        assert_eq!(expected.len(), frame_bytes);
        if std::env::var_os("WEBMEDIA_TRACE_ERROR_MAP").is_some() {
            let mut edge_error = 0u64;
            let mut edge_count = 0u64;
            let mut interior_error = 0u64;
            let mut interior_count = 0u64;
            let mut macroblocks = Vec::new();
            for my in 0..picture.height / 16 {
                for mx in 0..picture.width / 16 {
                    let mut error = 0u64;
                    for row in 0..16 {
                        for col in 0..16 {
                            let x = mx * 16 + col;
                            let y = my * 16 + row;
                            let at = y * picture.width + x;
                            let difference = u64::from(picture.luma[at].abs_diff(expected[at]));
                            error += difference;
                            if x % 4 == 0 || x % 4 == 3 || y % 4 == 0 || y % 4 == 3 {
                                edge_error += difference;
                                edge_count += 1;
                            } else {
                                interior_error += difference;
                                interior_count += 1;
                            }
                        }
                    }
                    macroblocks.push((error, mx, my));
                }
            }
            macroblocks.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            eprintln!(
                "luma edge MAE={:.3}, interior MAE={:.3}, worst macroblocks={:?}",
                edge_error as f64 / edge_count as f64,
                interior_error as f64 / interior_count as f64,
                &macroblocks[..10]
            );
        }
        for (name, actual, reference) in [
            (
                "Y",
                &picture.luma[..],
                &expected[..picture.width * picture.height],
            ),
            (
                "Cb",
                &picture.cb[..],
                &expected[picture.width * picture.height..picture.width * picture.height * 5 / 4],
            ),
            (
                "Cr",
                &picture.cr[..],
                &expected[picture.width * picture.height * 5 / 4..],
            ),
        ] {
            let error: u64 = actual
                .iter()
                .zip(reference)
                .map(|(&a, &b)| u64::from(a.abs_diff(b)))
                .sum();
            let mae = error as f64 / actual.len() as f64;
            eprintln!("sample 100 {name} MAE: {mae:.3}");
            let limit = if name == "Y" { 3.2 } else { 1.0 };
            assert!(mae < limit, "sample 100 {name} MAE: {mae}");
        }
    }

    #[test]
    fn site_b_sample_slice_layout() {
        let Ok(fixture) = std::env::var("WEBCORE_MP4_FIXTURE") else {
            return;
        };
        let bytes = std::fs::read(fixture).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let pps = parse_pps_2005(&index.config.picture_parameter_sets[0]).unwrap();
        for sample_number in 2..5 {
            let sample = &index.samples[sample_number];
            let mut stream = NalStream::new(index.config.nal_length_size).unwrap();
            let start = sample.offset as usize;
            let units = stream
                .push(&bytes[start..start + sample.size as usize])
                .unwrap();
            stream.finish().unwrap();
            let slices: Vec<_> = units
                .iter()
                .filter(|nal| nal[0] & 0x1f == 1)
                .map(|nal| {
                    parse_cabac_inter_slice(nal, &index.config.sequence_parameters[0], &pps.core)
                        .unwrap()
                        .first_mb
                })
                .collect();
            assert_eq!(slices, [0], "sample {sample_number} slice layout changed");
        }
    }

    #[test]
    fn site_first_p_macroblock_reaches_cabac_inter_syntax() {
        let Ok(path) = std::env::var("WEBCORE_MP4_FIXTURE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let pps = parse_pps_2005(&index.config.picture_parameter_sets[0]).unwrap();
        let sample = &index.samples[1];
        let start = sample.offset as usize;
        let mut nals = NalStream::new(index.config.nal_length_size).unwrap();
        let units = nals
            .push(&bytes[start..start + sample.size as usize])
            .unwrap();
        nals.finish().unwrap();
        let nal = units.iter().find(|nal| nal[0] & 0x1f == 1).unwrap();
        let slice =
            parse_cabac_inter_slice(nal, &index.config.sequence_parameters[0], &pps.core).unwrap();
        assert_eq!(slice.slice_type % 5, 0);
        assert_eq!(slice.slice_qp, 23);
        assert!(slice.weights.is_some());
        let mut decoder = CabacDecoder::new(&slice.rbsp[slice.data_byte_offset..]).unwrap();
        let mut contexts =
            InterMbContexts::new(slice.slice_qp, slice.cabac_init_idc, false).unwrap();
        let skipped = decoder
            .inter_mb_skip_flag(&mut contexts, None, None)
            .unwrap();
        assert!(!skipped);
        assert_eq!(slice.ref_idx_l0, 1);
        assert!(slice.reorder_l0.is_empty());
        assert_eq!(slice.cabac_init_idc, 0);
        assert_eq!(decoder.p_inter_mb_type(&mut contexts).unwrap(), 0);
        let mut motion = MotionVectorContexts::new(slice.slice_qp, slice.cabac_init_idc).unwrap();
        let dx = decoder
            .motion_vector_difference(&mut motion, 0, None, None)
            .unwrap();
        let dy = decoder
            .motion_vector_difference(&mut motion, 1, None, None)
            .unwrap();
        assert_eq!((dx, dy), (6, 1));
        let mut coded_contexts =
            CodedBlockContexts::new_inter(slice.slice_qp, slice.cabac_init_idc).unwrap();
        let coded = decoder
            .coded_block_pattern(&mut coded_contexts, None, None)
            .unwrap();
        assert_eq!((coded.luma, coded.chroma), (0, 0));
        let idr_sample = &index.samples[0];
        let idr_start = idr_sample.offset as usize;
        let mut idr_nals = NalStream::new(index.config.nal_length_size).unwrap();
        let idr_units = idr_nals
            .push(&bytes[idr_start..idr_start + idr_sample.size as usize])
            .unwrap();
        idr_nals.finish().unwrap();
        let idr = idr_units.iter().find(|nal| nal[0] & 0x1f == 5).unwrap();
        let reference =
            decode_cabac_idr_yuv_2005(idr, &index.config.sequence_parameters[0], &pps).unwrap();
        let block =
            predict_l0_16x16(&reference, 0, 0, [dx, dy], slice.weights.as_ref(), 0).unwrap();
        if let Ok(path) = std::env::var("WEBCORE_H264_P_REFERENCE_YUV") {
            let expected = std::fs::read(path).unwrap();
            assert_eq!(expected.len(), reference.width * reference.height * 3 / 2);
            let mut error = 0u32;
            for y in 0..16 {
                for x in 0..16 {
                    error += u32::from(
                        block.luma[y * 16 + x].abs_diff(expected[y * reference.width + x]),
                    );
                }
            }
            assert!(error < 256, "first P luma block error: {error}");
            let chroma_len = reference.width * reference.height / 4;
            for (plane, offset) in [
                (&block.cb, reference.width * reference.height),
                (&block.cr, reference.width * reference.height + chroma_len),
            ] {
                let mut error = 0u32;
                for y in 0..8 {
                    for x in 0..8 {
                        error += u32::from(
                            plane[y * 8 + x]
                                .abs_diff(expected[offset + y * (reference.width / 2) + x]),
                        );
                    }
                }
                assert!(error < 128, "first P chroma block error: {error}");
            }
        }
    }

    #[test]
    fn site_first_p_picture_reconstructs_full_frame() {
        let Ok(path) = std::env::var("WEBCORE_MP4_FIXTURE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        let pps = parse_pps_2005(&index.config.picture_parameter_sets[0]).unwrap();
        let first = &index.samples[0];
        let mut nals = NalStream::new(index.config.nal_length_size).unwrap();
        let units = nals
            .push(&bytes[first.offset as usize..first.offset as usize + first.size as usize])
            .unwrap();
        nals.finish().unwrap();
        let idr = units.iter().find(|nal| nal[0] & 0x1f == 5).unwrap();
        let reference =
            decode_cabac_idr_yuv_2005(idr, &index.config.sequence_parameters[0], &pps).unwrap();
        let second = &index.samples[1];
        let mut nals = NalStream::new(index.config.nal_length_size).unwrap();
        let units = nals
            .push(&bytes[second.offset as usize..second.offset as usize + second.size as usize])
            .unwrap();
        nals.finish().unwrap();
        let nal = units.iter().find(|nal| nal[0] & 0x1f == 1).unwrap();
        let frame = decode_cabac_p_2005(
            nal,
            &index.config.sequence_parameters[0],
            &pps,
            std::slice::from_ref(&reference),
        )
        .unwrap();
        assert_eq!((frame.width, frame.height), (1280, 720));
        assert_eq!(frame.motion[0][0].l0, Some((0, [6, 1])));
        assert_eq!(frame.motion.len(), (1280 / 16) * (720 / 16));
        if let Ok(path) = std::env::var("WEBCORE_H264_P_REFERENCE_YUV") {
            let expected = std::fs::read(path).unwrap();
            let actual = [&frame.luma[..], &frame.cb[..], &frame.cr[..]].concat();
            assert_eq!(actual.len(), expected.len());
            let error: u64 = actual
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| u64::from(a.abs_diff(b)))
                .sum();
            let mean_error = error as f64 / actual.len() as f64;
            assert!(mean_error < 5.0, "first P mean YUV error: {mean_error}");
        }
    }
}
