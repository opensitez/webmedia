//! Incremental classic MP4/AVC access-unit extraction.

use super::backend::{MediaDecodeError, MediaMetadata, StreamingVideoDecoder, VideoFrame};
use super::h264::{
    AvcError, NalStream, decode_intra_2003, frame_from_yuv420, parse_pps_2003, parse_pps_2005,
};
#[cfg(test)]
use super::h264_high::decode_cabac_idr_2005;
use super::h264_high::{Yuv420Picture, decode_cabac_idr_yuv_2005};
use super::h264_inter::decode_cabac_p_2005;
use super::mp4::{Mp4Error, Mp4Index};

const MAX_BUFFER_BYTES: usize = 64 * 1024 * 1024;

#[derive(Default)]
pub struct Mp4AvcStream {
    bytes: Vec<u8>,
    index: Option<Mp4Index>,
    next_sample: usize,
    pending_pictures: Vec<(i64, VideoFrame)>,
    reference_pictures: Vec<Yuv420Picture>,
    future_min_pts: Vec<i64>,
}

impl Mp4AvcStream {
    pub fn new() -> Self {
        Self::default()
    }

    fn invalid(error: impl std::fmt::Debug) -> MediaDecodeError {
        MediaDecodeError::InvalidData(format!("MP4/AVC: {error:?}"))
    }

    fn avc_error(error: AvcError) -> MediaDecodeError {
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
        sample_number: usize,
        references: &[Yuv420Picture],
    ) -> Result<(VideoFrame, Option<Yuv420Picture>, bool), MediaDecodeError> {
        let sample = &index.samples[sample_number];
        let start = usize::try_from(sample.offset).map_err(Self::invalid)?;
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
        for nal in nals {
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
                    if picture.is_some() || nal[0] & 0x60 == 0 {
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
                    let previous = references.last().ok_or(MediaDecodeError::Unsupported)?;
                    let yuv =
                        decode_cabac_p_2005(&nal, sps, &pps, previous).map_err(Self::avc_error)?;
                    picture = Some(frame_from_yuv420(sps, &yuv.luma, &yuv.cb, &yuv.cr));
                    reference = Some(yuv);
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
        Ok((picture, reference, is_idr))
    }
}

impl StreamingVideoDecoder for Mp4AvcStream {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<VideoFrame>, MediaDecodeError> {
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
        while let Some(sample) = self.index.as_ref().unwrap().samples.get(self.next_sample) {
            let end = sample
                .offset
                .checked_add(u64::from(sample.size))
                .ok_or(MediaDecodeError::Unsupported)?;
            if end > self.bytes.len() as u64 {
                break;
            }
            let (frame, reference, is_idr) = match Self::decode_sample(
                self.index.as_ref().unwrap(),
                &self.bytes,
                self.next_sample,
                &self.reference_pictures,
            ) {
                Ok(frame) => frame,
                Err(_) if !frames.is_empty() => break,
                Err(error) => return Err(error),
            };
            let presentation_time = sample.presentation_time;
            self.next_sample += 1;
            if is_idr {
                self.reference_pictures.clear();
            }
            if let Some(reference) = reference {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::h264::{
        AvcConfig, parse_cabac_idr_i_slice, parse_cabac_inter_slice, parse_pps_2005, parse_sps,
    };
    use crate::video::h264_cabac::{
        CabacDecoder, ChromaDcContexts, CodedBlockContexts, InterMbContexts, IntraMbTypeContexts,
        IntraPredContexts, Luma4x4Contexts, Luma8x8Contexts, MbQpContexts, MotionVectorContexts,
        Transform8x8Contexts, intra4x4_modes, intra8x8_modes,
    };
    use crate::video::h264_inter::predict_l0_16x16;
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
    fn indexes_site_stream_before_rejecting_unsupported_picture() {
        let Ok(path) = std::env::var("WEBCORE_MP4_FIXTURE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let mut decoder = Mp4AvcStream::new();
        assert!(decoder.metadata().is_none());
        let mut rejection = None;
        let mut delivered = Vec::new();
        for chunk in bytes.chunks(16 * 1024) {
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
        assert_eq!(delivered.len(), 1);
        assert_eq!((delivered[0].width, delivered[0].height), (1280, 720));
        assert_eq!(delivered[0].timestamp, 0.0);
        assert_eq!(decoder.reference_pictures.len(), 1);
        assert_eq!(
            (
                decoder.reference_pictures[0].width,
                decoder.reference_pictures[0].height
            ),
            (1280, 720)
        );
        assert_eq!(rejection, Some(MediaDecodeError::Unsupported));
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
}
