//! VP9 keyframe reconstruction from independently coded tiles.

use super::backend::MediaDecodeError;
use super::vp8_keyframe::YuvKeyFrame;
use super::vp8_predict::Plane;
use super::vp9::{FrameHeader, InterframeHeader, KeyframeLayout, LoopFilterState, ReferenceFrame, SegmentationState};
use super::vp9_loop_filter::FilterGrid;
use super::vp9_compressed::CompressedHeader;
use super::vp9_adapt::{CoefficientCounts, NonCoefficientCounts};
use super::vp9_motion::{predict_block, scaled_motion};
use super::vp9_tile::{decode_frame_tiles_counted, InterPrediction, TileBlock};
use super::vp9_transform::{add_inter_residual, reconstruct_intra};
use std::sync::Arc;

pub(super) struct Vp9Decoder {
    references: [Option<Arc<YuvKeyFrame>>; 8],
    contexts: [CompressedHeader; 4],
    segmentation: SegmentationState,
    segment_ids: Vec<u8>,
    loop_filter: LoopFilterState,
    previous_predictions: Option<Vec<Option<InterPrediction>>>,
    previous_shown: bool,
    previous_keyframe: bool,
    previous_dimensions: Option<(u32, u32)>,
}

impl Vp9Decoder {
    pub(super) fn new() -> Self {
        Self {
            references: std::array::from_fn(|_| None),
            contexts: std::array::from_fn(|_| CompressedHeader::default()),
            segmentation: SegmentationState::default(),
            segment_ids: Vec::new(),
            loop_filter: LoopFilterState::default(),
            previous_predictions: None,
            previous_shown: false,
            previous_keyframe: false,
            previous_dimensions: None,
        }
    }

    pub(super) fn decode(&mut self, data: &[u8]) -> Result<Option<Arc<YuvKeyFrame>>, MediaDecodeError> {
        let header = FrameHeader::parse(data)?;
        if let Some(slot) = header.show_existing_frame {
            return self.references[slot as usize].clone()
                .map(Some).ok_or(MediaDecodeError::Unsupported);
        }
        if header.key_frame {
            let layout = KeyframeLayout::parse(data)?;
            let probabilities = CompressedHeader::parse_keyframe(&layout)?;
            let (decoded, segment_ids, coef_counts) = decode_keyframe_with_segments(data)?;
            let frame = Arc::new(decoded);
            self.contexts = std::array::from_fn(|_| CompressedHeader::default());
            if layout.refresh_frame_context {
                self.contexts[0] = if layout.frame_parallel_decoding {
                    probabilities
                } else {
                    let mut adapted = CompressedHeader::default();
                    adapted.tx_mode = probabilities.tx_mode;
                    adapted.skip_probs = probabilities.skip_probs;
                    adapted.inter_probs.tx_8x8 = probabilities.inter_probs.tx_8x8;
                    adapted.inter_probs.tx_16x16 = probabilities.inter_probs.tx_16x16;
                    adapted.inter_probs.tx_32x32 = probabilities.inter_probs.tx_32x32;
                    coef_counts.adapt(&mut adapted, 112);
                    adapted
                };
            }
            self.segmentation = layout.segmentation_state(SegmentationState::default());
            self.segment_ids = segment_ids;
            self.loop_filter = layout.loop_filter;
            self.previous_predictions = None;
            self.previous_shown = header.show_frame;
            self.previous_keyframe = true;
            self.previous_dimensions = Some((frame.width as u32, frame.height as u32));
            self.references.fill(Some(frame.clone()));
            return Ok(header.show_frame.then_some(frame));
        }
        let reference_headers: [Option<ReferenceFrame>; 8] = std::array::from_fn(|index| {
            self.references[index].as_ref().map(|frame| ReferenceFrame {
                width: frame.width as u32, height: frame.height as u32, bit_depth: 8,
            })
        });
        let (inter, layout, segmentation) = KeyframeLayout::parse_interframe_with_filter(
            data, &reference_headers, self.segmentation,
            if header.error_resilient { LoopFilterState::default() } else { self.loop_filter },
        )?;
        let signaled_index = layout.frame_context_idx as usize;
        let independent = inter.intra_only || header.error_resilient;
        if independent {
            self.segment_ids.fill(0);
            if header.error_resilient || inter.reset_frame_context == 3 {
                self.contexts = std::array::from_fn(|_| CompressedHeader::default());
            } else if inter.reset_frame_context == 2 {
                self.contexts[signaled_index] = CompressedHeader::default();
            }
        }
        let index = if independent { 0 } else { signaled_index };
        let probabilities = CompressedHeader::parse_interframe(&layout, &inter, &self.contexts[index])?;
        if inter.intra_only {
            let (decoded, segment_ids, coef_counts) = decode_intra_with_segments(&layout, &probabilities)?;
            let frame = Arc::new(decoded);
            if layout.refresh_frame_context {
                let mut saved = probabilities.clone();
                if !layout.frame_parallel_decoding {
                    saved.coef_probs = self.contexts[index].coef_probs;
                    saved.inter_probs = self.contexts[index].inter_probs.clone();
                    saved.inter_probs.tx_8x8 = probabilities.inter_probs.tx_8x8;
                    saved.inter_probs.tx_16x16 = probabilities.inter_probs.tx_16x16;
                    saved.inter_probs.tx_32x32 = probabilities.inter_probs.tx_32x32;
                    coef_counts.adapt(&mut saved, 112);
                }
                self.contexts[index] = saved;
            }
            self.segmentation = segmentation;
            self.segment_ids = segment_ids;
            self.loop_filter = layout.loop_filter;
            self.previous_predictions = None;
            self.previous_shown = header.show_frame;
            self.previous_keyframe = false;
            self.previous_dimensions = Some((frame.width as u32, frame.height as u32));
            for slot in 0..8 {
                if inter.refresh_frame_flags & (1 << slot) != 0 {
                    self.references[slot] = Some(frame.clone());
                }
            }
            return Ok(header.show_frame.then_some(frame));
        }
        if std::env::var_os("WEBMEDIA_VP9_REPORT").is_some() {
            eprintln!("VP9 interframe: show={}, intra_only={}, ref_mode={}, segment={}, tx_mode={}, refs={:?}, refresh={:#04x}, context={}, refresh_context={}, parallel={}, tile_bytes={}",
                header.show_frame,
                inter.intra_only, probabilities.reference_mode, layout.segmentation_enabled,
                probabilities.tx_mode, inter.reference_indices, inter.refresh_frame_flags,
                index, layout.refresh_frame_context, layout.frame_parallel_decoding,
                layout.tile_partitions()?.iter().map(|tile| tile.len()).sum::<usize>());
        }
        let selected = |reference: usize| self.references[inter.reference_indices[reference] as usize]
            .as_deref().ok_or(MediaDecodeError::Unsupported);
        let previous = if self.previous_shown && !header.error_resilient
            && self.previous_dimensions == layout.header.width.zip(layout.header.height) {
            self.previous_predictions.as_deref()
        } else { None };
        let (decoded, predictions, segment_ids, coef_counts, noncoef_counts) = decode_interframe_with_segments(&layout, &inter, &probabilities,
            [selected(0)?, selected(1)?, selected(2)?], previous, &self.segment_ids)?;
        let first_after_keyframe = self.previous_keyframe;
        let frame = Arc::new(decoded);
        self.previous_predictions = Some(predictions);
        self.previous_shown = header.show_frame;
        self.previous_keyframe = false;
        self.previous_dimensions = Some((frame.width as u32, frame.height as u32));
        self.segmentation = segmentation;
        self.segment_ids = segment_ids;
        self.loop_filter = layout.loop_filter;
        if layout.refresh_frame_context {
            self.contexts[index] = if layout.frame_parallel_decoding {
                probabilities
            } else {
                let mut adapted = self.contexts[index].clone();
                adapted.tx_mode = probabilities.tx_mode;
                adapted.reference_mode = probabilities.reference_mode;
                coef_counts.adapt(&mut adapted, if first_after_keyframe { 128 } else { 112 });
                noncoef_counts.adapt(&mut adapted, inter.interpolation_filter == 4,
                    inter.allow_high_precision_mv);
                adapted
            };
        }
        for slot in 0..8 {
            if inter.refresh_frame_flags & (1 << slot) != 0 {
                self.references[slot] = Some(frame.clone());
            }
        }
        Ok(header.show_frame.then_some(frame))
    }
}

#[cfg(test)]
pub(super) fn decode_keyframe(data: &[u8]) -> Result<YuvKeyFrame, MediaDecodeError> {
    decode_keyframe_with_segments(data).map(|(frame, _, _)| frame)
}

fn decode_keyframe_with_segments(data: &[u8]) -> Result<(YuvKeyFrame, Vec<u8>, CoefficientCounts), MediaDecodeError> {
    let layout = KeyframeLayout::parse(data)?;
    let compressed = CompressedHeader::parse_keyframe(&layout)?;
    decode_intra_with_segments(&layout, &compressed)
}

fn decode_intra_with_segments(
    layout: &KeyframeLayout<'_>, compressed: &CompressedHeader,
) -> Result<(YuvKeyFrame, Vec<u8>, CoefficientCounts), MediaDecodeError> {
    if layout.header.bit_depth != Some(8) {
        return Err(MediaDecodeError::Unsupported);
    }
    let width = layout.header.width.ok_or(MediaDecodeError::Unsupported)? as usize;
    let height = layout.header.height.ok_or(MediaDecodeError::Unsupported)? as usize;
    let mut frame = empty_frame(width, height);
    let mut filter = FilterGrid::new(width, height);
    let mut segment_ids = vec![0; width.div_ceil(8) * height.div_ceil(8)];
    let (tiles, coef_counts, _) = decode_frame_tiles_counted(layout, compressed, None, None, None)?;
    let tile_cols = 1usize << layout.tile_cols_log2;
    let mi_cols = width.div_ceil(8);
    for (tile_index, blocks) in tiles.into_iter().enumerate() {
        let tile_col = tile_index % tile_cols;
        let tile_start = (((tile_col * mi_cols.div_ceil(8)) >> layout.tile_cols_log2) << 6)
            .min(width.div_ceil(8) * 8);
        for block in blocks {
            reconstruct_block(&mut frame, &layout, &block, tile_start)?;
            filter.record(&block);
            record_segment(&mut segment_ids, width.div_ceil(8), height.div_ceil(8), &block);
        }
    }
    filter.apply(&mut frame, &layout);
    Ok((frame, segment_ids, coef_counts))
}

fn record_segment(segments: &mut [u8], cols: usize, rows: usize, entry: &TileBlock) {
    for y in entry.y / 8..((entry.y + entry.height.max(8)) / 8).min(rows) {
        for x in entry.x / 8..((entry.x + entry.width.max(8)) / 8).min(cols) {
            segments[y * cols + x] = entry.block.segment_id;
        }
    }
}

fn empty_frame(width: usize, height: usize) -> YuvKeyFrame {
    let padded_width = width.div_ceil(64) * 64;
    let padded_height = height.div_ceil(64) * 64;
    YuvKeyFrame {
        width,
        height,
        y: Plane::new(padded_width, padded_height),
        u: Plane::new(padded_width / 2, padded_height / 2),
        v: Plane::new(padded_width / 2, padded_height / 2),
    }
}

#[cfg(test)]
pub(super) fn decode_interframe(
    layout: &KeyframeLayout<'_>, inter: &InterframeHeader,
    compressed: &CompressedHeader, references: [&YuvKeyFrame; 3],
    previous_predictions: Option<&[Option<InterPrediction>]>,
) -> Result<(YuvKeyFrame, Vec<Option<InterPrediction>>), MediaDecodeError> {
    let (frame, predictions, _, _, _) = decode_interframe_with_segments(
        layout, inter, compressed, references, previous_predictions, &[],
    )?;
    Ok((frame, predictions))
}

fn decode_interframe_with_segments(
    layout: &KeyframeLayout<'_>, inter: &InterframeHeader,
    compressed: &CompressedHeader, references: [&YuvKeyFrame; 3],
    previous_predictions: Option<&[Option<InterPrediction>]>, previous_segments: &[u8],
) -> Result<(YuvKeyFrame, Vec<Option<InterPrediction>>, Vec<u8>, CoefficientCounts, NonCoefficientCounts), MediaDecodeError> {
    let profile = std::env::var_os("WEBMEDIA_VP9_PROFILE").is_some();
    let started = profile.then(std::time::Instant::now);
    let width = layout.header.width.ok_or(MediaDecodeError::Unsupported)? as usize;
    let height = layout.header.height.ok_or(MediaDecodeError::Unsupported)? as usize;
    if layout.header.bit_depth != Some(8) {
        return Err(MediaDecodeError::Unsupported);
    }
    let mut frame = empty_frame(width, height);
    let mut filter = FilterGrid::new(width, height);
    let mi_cols = width.div_ceil(8);
    let mut predictions = vec![None; mi_cols * height.div_ceil(8)];
    let mut segment_ids = vec![0; mi_cols * height.div_ceil(8)];
    let mut compound_scratch = Plane::new(64, 64);
    let (tiles, coef_counts, noncoef_counts) = decode_frame_tiles_counted(layout, compressed,
        Some(inter), previous_predictions, Some(previous_segments))?;
    let entropy_done = profile.then(std::time::Instant::now);
    let tile_cols = 1usize << layout.tile_cols_log2;
    for (tile_index, blocks) in tiles.into_iter().enumerate() {
        let tile_col = tile_index % tile_cols;
        let tile_start = (((tile_col * mi_cols.div_ceil(8)) >> layout.tile_cols_log2) << 6)
            .min(mi_cols * 8);
        for entry in blocks {
            filter.record(&entry);
            record_segment(&mut segment_ids, mi_cols, height.div_ceil(8), &entry);
            if let Some(prediction) = entry.block.inter {
                for row in entry.y / 8..((entry.y + entry.height.max(8)) / 8).min(height.div_ceil(8)) {
                    for col in entry.x / 8..((entry.x + entry.width.max(8)) / 8).min(mi_cols) {
                        predictions[row * mi_cols + col] = Some(prediction);
                    }
                }
                reconstruct_inter_block(&mut frame, references, layout, &entry, &mut compound_scratch)?;
            } else {
                reconstruct_block(&mut frame, layout, &entry, tile_start)?;
            }
        }
    }
    let reconstruction_done = profile.then(std::time::Instant::now);
    filter.apply(&mut frame, layout);
    if let (Some(started), Some(entropy_done), Some(reconstruction_done)) =
        (started, entropy_done, reconstruction_done)
    {
        eprintln!("VP9 phases us: entropy={} reconstruction={} filter={}",
            entropy_done.duration_since(started).as_micros(),
            reconstruction_done.duration_since(entropy_done).as_micros(),
            reconstruction_done.elapsed().as_micros());
    }
    Ok((frame, predictions, segment_ids, coef_counts, noncoef_counts))
}

fn reconstruct_block(
    frame: &mut YuvKeyFrame,
    layout: &KeyframeLayout<'_>,
    entry: &TileBlock,
    tile_start: usize,
) -> Result<(), MediaDecodeError> {
    let block = &entry.block;
    let quantizer = match layout.segment_alt_q[block.segment_id as usize] {
        Some(delta) if layout.segmentation_abs_or_delta_update => i32::from(delta),
        Some(delta) => i32::from(layout.base_q_idx) + i32::from(delta),
        None => i32::from(layout.base_q_idx),
    };
    continue_luma(frame, layout, entry, tile_start, quantizer)?;
    let visible_width = frame.width.div_ceil(2);
    let visible_height = frame.height.div_ceil(2);
    let x0 = entry.x / 2;
    let y0 = entry.y / 2;
    let block_width = entry.width.max(8) / 2;
    let block_height = entry.height.max(8) / 2;
    for plane_index in 0..2 {
        let plane = if plane_index == 0 { &mut frame.u } else { &mut frame.v };
        let tx_size = block.tx_size.min((block_width.min(block_height) / 4).trailing_zeros() as u8);
        let transform_size = 4usize << tx_size;
        let transforms_wide = block_width / transform_size;
        let coefficients = &block.chroma_coefficients.as_ref().ok_or(MediaDecodeError::Unsupported)?[plane_index];
        for (index, transform) in coefficients.chunks_exact(transform_size * transform_size).enumerate() {
            let block_x = index % transforms_wide;
            let x = x0 + block_x * transform_size;
            let y = y0 + index / transforms_wide * transform_size;
            if x >= visible_width || y >= visible_height {
                continue;
            }
            reconstruct_intra(
                plane, x, y, transform_size, block.uv_mode, true,
                x > tile_start / 2, y != 0, block_x + 1 < transforms_wide,
                visible_width, visible_height, transform, 8,
                quantizer + i32::from(layout.delta_q_uv_dc),
                quantizer + i32::from(layout.delta_q_uv_ac),
                layout.lossless,
            )?;
        }
    }
    Ok(())
}

fn reconstruct_inter_block(
    frame: &mut YuvKeyFrame,
    references: [&YuvKeyFrame; 3],
    layout: &KeyframeLayout<'_>,
    entry: &TileBlock,
    compound_scratch: &mut Plane,
) -> Result<(), MediaDecodeError> {
    let prediction = entry.block.inter.ok_or(MediaDecodeError::Unsupported)?;
    let reference = *references.get(prediction.reference.saturating_sub(1) as usize)
        .ok_or(MediaDecodeError::Unsupported)?;
    let quantizer = match layout.segment_alt_q[entry.block.segment_id as usize] {
        Some(delta) if layout.segmentation_abs_or_delta_update => i32::from(delta),
        Some(delta) => i32::from(layout.base_q_idx) + i32::from(delta),
        None => i32::from(layout.base_q_idx),
    };
    for plane_index in 0..3 {
        let chroma = plane_index != 0;
        let scale = if chroma { 2 } else { 1 };
        let x = entry.x / scale;
        let y = entry.y / scale;
        let width = entry.width.max(8) / scale;
        let height = entry.height.max(8) / scale;
        let (destination, source) = match plane_index {
            0 => (&mut frame.y, &reference.y),
            1 => (&mut frame.u, &reference.u),
            _ => (&mut frame.v, &reference.v),
        };
        let visible_width = reference.width.div_ceil(scale);
        let visible_height = reference.height.div_ceil(scale);
        if !chroma {
            if let Some(sub_motions) = prediction.sub_motions {
                for sub_y in 0..2 {
                    for sub_x in 0..2 {
                        let sampling = scaled_motion(
                            (frame.width, frame.height), (reference.width, reference.height),
                            (x + 4 * sub_x, y + 4 * sub_y), (entry.x / 8, entry.y / 8),
                            (1, 1), sub_motions[sub_y * 2 + sub_x], false,
                        )?;
                        predict_block(destination, source, x + 4 * sub_x, y + 4 * sub_y, 4, 4,
                            sampling.start_x, sampling.start_y, sampling.step_x, sampling.step_y,
                            prediction.interpolation_filter, visible_width, visible_height)?;
                    }
                }
            } else {
                let sampling = scaled_motion(
                    (frame.width, frame.height), (reference.width, reference.height),
                    (x, y), (entry.x / 8, entry.y / 8),
                    (entry.width.max(8) / 8, entry.height.max(8) / 8), prediction.motion, false,
                )?;
                predict_block(destination, source, x, y, width, height,
                    sampling.start_x, sampling.start_y, sampling.step_x, sampling.step_y,
                    prediction.interpolation_filter, visible_width, visible_height)?;
            }
        } else {
            let motion = if let Some(sub_motions) = prediction.sub_motions {
                let sum_row: i32 = sub_motions.iter().map(|value| value.0).sum();
                let sum_col: i32 = sub_motions.iter().map(|value| value.1).sum();
                (round_chroma_motion(sum_row), round_chroma_motion(sum_col))
            } else { prediction.motion };
            let sampling = scaled_motion(
                (frame.width, frame.height), (reference.width, reference.height),
                (x, y), (entry.x / 8, entry.y / 8),
                (entry.width.max(8) / 8, entry.height.max(8) / 8), motion, true,
            )?;
            predict_block(destination, source, x, y, width, height,
                sampling.start_x, sampling.start_y, sampling.step_x, sampling.step_y,
                prediction.interpolation_filter, visible_width, visible_height)?;
        }
        if let Some(second) = prediction.second {
            let other = *references.get(second.reference.saturating_sub(1) as usize)
                .ok_or(MediaDecodeError::Unsupported)?;
            let other_source = match plane_index {
                0 => &other.y, 1 => &other.u, _ => &other.v,
            };
            compound_scratch.width = width;
            compound_scratch.pixels.resize(width * height, 0);
            let other_prediction = &mut *compound_scratch;
            let other_width = other.width.div_ceil(scale);
            let other_height = other.height.div_ceil(scale);
            if !chroma {
                if let Some(sub_motions) = second.sub_motions {
                    for sub_y in 0..2 {
                        for sub_x in 0..2 {
                            let sampling = scaled_motion(
                                (frame.width, frame.height), (other.width, other.height),
                                (x + 4 * sub_x, y + 4 * sub_y), (entry.x / 8, entry.y / 8),
                                (1, 1), sub_motions[sub_y * 2 + sub_x], false,
                            )?;
                            predict_block(other_prediction, other_source,
                                4 * sub_x, 4 * sub_y, 4, 4,
                                sampling.start_x, sampling.start_y, sampling.step_x, sampling.step_y,
                                prediction.interpolation_filter, other_width, other_height)?;
                        }
                    }
                } else {
                    let sampling = scaled_motion(
                        (frame.width, frame.height), (other.width, other.height),
                        (x, y), (entry.x / 8, entry.y / 8),
                        (entry.width.max(8) / 8, entry.height.max(8) / 8), second.motion, false,
                    )?;
                    predict_block(other_prediction, other_source, 0, 0, width, height,
                        sampling.start_x, sampling.start_y, sampling.step_x, sampling.step_y,
                        prediction.interpolation_filter, other_width, other_height)?;
                }
            } else {
                let motion = if let Some(sub_motions) = second.sub_motions {
                    let sum_row: i32 = sub_motions.iter().map(|value| value.0).sum();
                    let sum_col: i32 = sub_motions.iter().map(|value| value.1).sum();
                    (round_chroma_motion(sum_row), round_chroma_motion(sum_col))
                } else { second.motion };
                let sampling = scaled_motion(
                    (frame.width, frame.height), (other.width, other.height),
                    (x, y), (entry.x / 8, entry.y / 8),
                    (entry.width.max(8) / 8, entry.height.max(8) / 8), motion, true,
                )?;
                predict_block(other_prediction, other_source, 0, 0, width, height,
                    sampling.start_x, sampling.start_y, sampling.step_x, sampling.step_y,
                    prediction.interpolation_filter, other_width, other_height)?;
            }
            for row in 0..height {
                let to = (y + row) * destination.width + x;
                let source = &other_prediction.pixels[row * width..(row + 1) * width];
                for (a, &b) in destination.pixels[to..to + width].iter_mut().zip(source) {
                    *a = ((u16::from(*a) + u16::from(b) + 1) >> 1) as u8;
                }
            }
        }
        if entry.block.skip { continue; }
        let tx_size = entry.block.tx_size.min((width.min(height) / 4).trailing_zeros() as u8);
        let transform_size = 4usize << tx_size;
        let transforms_wide = width / transform_size;
        let dc = quantizer + if chroma { i32::from(layout.delta_q_uv_dc) }
            else { i32::from(layout.delta_q_y_dc) };
        let ac = quantizer + if chroma { i32::from(layout.delta_q_uv_ac) } else { 0 };
        let luma = entry.block.luma_coefficients.as_ref().ok_or(MediaDecodeError::Unsupported)?;
        let chroma_coefficients = entry.block.chroma_coefficients.as_ref().ok_or(MediaDecodeError::Unsupported)?;
        let luma_transforms = if chroma { &luma[..0] } else { luma.as_slice() };
        let chroma_transforms = if chroma { chroma_coefficients[plane_index - 1].as_slice() } else { &[] };
        let coefficients = luma_transforms.iter().map(Vec::as_slice)
            .chain(chroma_transforms.chunks_exact(transform_size * transform_size));
        for (index, transform) in coefficients.enumerate() {
            if transform.iter().all(|&value| value == 0) { continue; }
            add_inter_residual(destination,
                x + (index % transforms_wide) * transform_size,
                y + (index / transforms_wide) * transform_size,
                transform_size, transform, dc, ac, layout.lossless)?;
        }
    }
    Ok(())
}

fn round_chroma_motion(value: i32) -> i32 {
    (value + if value < 0 { -2 } else { 2 }) / 4
}

fn continue_luma(
    frame: &mut YuvKeyFrame,
    layout: &KeyframeLayout<'_>,
    entry: &TileBlock,
    tile_start: usize,
    quantizer: i32,
) -> Result<(), MediaDecodeError> {
    let block = &entry.block;
    let transform_size = 4usize << block.tx_size;
    let transforms_wide = entry.width.max(8) / transform_size;
    for (index, transform) in block.luma_coefficients.as_ref().ok_or(MediaDecodeError::Unsupported)?.iter().enumerate() {
        let block_x = index % transforms_wide;
        let block_y = index / transforms_wide;
        let x = entry.x + block_x * transform_size;
        let y = entry.y + block_y * transform_size;
        if x >= frame.width || y >= frame.height {
            continue;
        }
        let mode = block.sub_modes.map_or(block.y_mode, |modes| modes[block_y * 2 + block_x]);
        reconstruct_intra(
            &mut frame.y, x, y, transform_size, mode, false,
            x > tile_start, y != 0, block_x + 1 < transforms_wide,
            frame.width, frame.height, transform, 8,
            quantizer + i32::from(layout.delta_q_y_dc), quantizer,
            layout.lossless,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::vp9::split_superframe;
    use crate::video::webm::{WebmVideoCodec, WebmVideoStream};

    // Profile 0 keyframes and intra-only frames share compressed headers and intra tile syntax.
    fn hidden_intra_packet(keyframe: &[u8], refresh: u8, reset: u8, context: u8) -> Vec<u8> {
        let layout = KeyframeLayout::parse(keyframe).unwrap();
        assert_eq!(layout.header.profile, 0);
        assert!(!layout.header.error_resilient);
        assert_eq!(layout.header.width, layout.header.render_width);
        assert_eq!(layout.header.height, layout.header.render_height);
        let compressed_start = keyframe.len() - layout.compressed_header.len() - layout.tiles.len();
        let bit = |position: usize| (keyframe[position / 8] >> (7 - position % 8)) & 1;
        let header_end = (0..8).map(|padding| compressed_start * 8 - padding)
            .find(|&end| {
                (end - 16..end).fold(0usize, |value, position| (value << 1) | bit(position) as usize)
                    == layout.compressed_header.len()
            }).unwrap();
        let mut bits = Vec::new();
        let mut append = |value: u32, count: usize| {
            for shift in (0..count).rev() { bits.push(((value >> shift) & 1) as u8); }
        };
        append(0x84, 8);
        append(1, 1);
        append(u32::from(reset), 2);
        for byte in [0x49, 0x83, 0x42] { append(byte, 8); }
        append(u32::from(refresh), 8);
        append(layout.header.width.unwrap() - 1, 16);
        append(layout.header.height.unwrap() - 1, 16);
        append(0, 1);
        let body_start = bits.len();
        bits.extend((69..header_end).map(bit));
        bits[body_start + 2] = (context >> 1) & 1;
        bits[body_start + 3] = context & 1;
        let mut packet: Vec<u8> = bits.chunks(8).map(|chunk| {
            chunk.iter().fold(0u8, |value, &bit| (value << 1) | bit) << (8 - chunk.len())
        }).collect();
        packet.extend_from_slice(&keyframe[compressed_start..]);
        packet
    }

    #[test]
    fn hidden_intra_only_refreshes_selected_reference_and_can_be_shown() {
        let bytes = include_bytes!("../../tests/fixtures/vp9-aq.webm");
        let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
        let packets = stream.push(bytes).unwrap();
        let keyframe = split_superframe(&packets[0].data).unwrap()[0];
        let expected = decode_keyframe(keyframe).unwrap();
        let mut decoder = Vp9Decoder::new();
        decoder.contexts[1].skip_probs = [1; 3];
        assert!(decoder.decode(&hidden_intra_packet(keyframe, 1 << 3, 3, 0)).unwrap().is_none());
        for slot in 0..8 { assert_eq!(decoder.references[slot].is_some(), slot == 3); }
        assert_eq!(decoder.contexts[1], CompressedHeader::default());
        let shown = decoder.decode(&[0x8b]).unwrap().unwrap();
        assert_eq!(shown.y.pixels, expected.y.pixels);
        assert_eq!(shown.u.pixels, expected.u.pixels);
        assert_eq!(shown.v.pixels, expected.v.pixels);
    }

    #[test]
    fn intra_only_context_reset_preserves_other_saved_contexts() {
        let bytes = include_bytes!("../../tests/fixtures/vp9-aq.webm");
        let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
        let packets = stream.push(bytes).unwrap();
        let keyframe = split_superframe(&packets[0].data).unwrap()[0];
        let mut decoder = Vp9Decoder::new();
        decoder.contexts[1].skip_probs = [123; 3];
        decoder.contexts[2].skip_probs = [42; 3];
        assert!(decoder.decode(&hidden_intra_packet(keyframe, 0xff, 2, 2)).unwrap().is_none());
        assert_eq!(decoder.contexts[1].skip_probs, [123; 3]);
        assert_eq!(decoder.contexts[2], CompressedHeader::default());
    }

    #[test]
    fn interframes_continue_after_hidden_intra_only() {
        let bytes = include_bytes!("../../tests/fixtures/vp9-aq.webm");
        let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
        let mut original = Vp9Decoder::new();
        let mut hidden = Vp9Decoder::new();
        let mut coded = 0;
        for packet in stream.push(bytes).unwrap() {
            for data in split_superframe(&packet.data).unwrap() {
                let expected = original.decode(data).unwrap().unwrap();
                if coded == 0 {
                    assert!(hidden.decode(&hidden_intra_packet(data, 0xff, 3, 0)).unwrap().is_none());
                } else {
                    let frame = hidden.decode(data).unwrap().unwrap();
                    assert_eq!(frame.y.pixels, expected.y.pixels);
                    assert_eq!(frame.u.pixels, expected.u.pixels);
                    assert_eq!(frame.v.pixels, expected.v.pixels);
                }
                coded += 1;
            }
        }
        assert_eq!(coded, 60);
    }

    #[test]
    fn decodes_segmented_vp9_fixture_across_stream_chunks() {
        let bytes = include_bytes!("../../tests/fixtures/vp9-aq.webm");
        let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
        let mut decoder = Vp9Decoder::new();
        let mut coded = 0;
        let mut shown = 0;
        for chunk in bytes.chunks(1024) {
            for packet in stream.push(chunk).unwrap() {
                for data in split_superframe(&packet.data).unwrap() {
                    if coded == 0 {
                        assert!(KeyframeLayout::parse(data).unwrap().segmentation_enabled);
                    }
                    if decoder.decode(data).unwrap().is_some() { shown += 1; }
                    if coded == 0 {
                        assert!(decoder.segment_ids.iter().any(|&id| id != 0));
                    }
                    coded += 1;
                }
            }
        }
        stream.finish().unwrap();
        assert_eq!((coded, shown), (60, 60));
    }

    #[test]
    fn decodes_compound_altref_fixture_across_stream_chunks() {
        let bytes = include_bytes!("../../tests/fixtures/vp9-altref.webm");
        let reference = include_bytes!("../../tests/fixtures/vp9-altref-check.yuv");
        let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
        let mut decoder = Vp9Decoder::new();
        let mut coded = 0;
        let mut shown = 0;
        let frame_bytes = 320 * 240 * 3 / 2;
        assert_eq!(reference.len(), 2 * frame_bytes);
        for chunk in bytes.chunks(1024) {
            for packet in stream.push(chunk).unwrap() {
                for data in split_superframe(&packet.data).unwrap() {
                    if let Some(frame) = decoder.decode(data).unwrap() {
                        assert_eq!((frame.width, frame.height), (320, 240));
                        if let Some(check) = [5, 110].iter().position(|&index| index == shown) {
                            let reference = &reference[check * frame_bytes..(check + 1) * frame_bytes];
                            let mut offset = 0;
                            for (plane, width, height) in [
                                (&frame.y, 320, 240), (&frame.u, 160, 120), (&frame.v, 160, 120),
                            ] {
                                let mut error = 0u64;
                                for y in 0..height {
                                    for x in 0..width {
                                        error += u64::from(plane.pixels[y * plane.width + x]
                                            .abs_diff(reference[offset + y * width + x]));
                                    }
                                }
                                assert!(error < (width * height) as u64,
                                    "VP9 shown frame {shown} differs from reference");
                                offset += width * height;
                            }
                        }
                        shown += 1;
                    }
                    coded += 1;
                }
            }
        }
        stream.finish().unwrap();
        assert_eq!((coded, shown), (130, 120));
    }

    #[test]
    fn decodes_lossless_motion_pixel_exact_across_stream_chunks() {
        let bytes = include_bytes!("../../tests/fixtures/vp9-lossless.webm");
        let reference = include_bytes!("../../tests/fixtures/vp9-lossless.yuv");
        assert_lossless_clip(bytes, reference, 160, 96, &(0..30).collect::<Vec<_>>());
    }

    #[test]
    fn decodes_dependent_tile_rows_pixel_exact() {
        let bytes = include_bytes!("../../tests/fixtures/vp9-tile-rows.webm");
        let reference = include_bytes!("../../tests/fixtures/vp9-tile-rows-check.yuv");
        let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
        let packet = stream.push(bytes).unwrap().remove(0);
        let layout = KeyframeLayout::parse(&packet.data).unwrap();
        assert_eq!(layout.tile_rows_log2, 1);
        assert_eq!(layout.tile_partitions().unwrap().len(), 2);
        assert_lossless_clip(bytes, reference, 320, 192, &[0, 1, 29]);
    }

    #[test]
    fn decodes_padded_edge_contexts_pixel_exact() {
        assert_lossless_clip(include_bytes!("../../tests/fixtures/vp9-edges.webm"),
            include_bytes!("../../tests/fixtures/vp9-edges-check.yuv"), 162, 98, &[0, 5, 29]);
    }

    fn assert_lossless_clip(bytes: &[u8], reference: &[u8], width: usize, height: usize,
        check_frames: &[usize]) {
        let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
        let mut decoder = Vp9Decoder::new();
        let mut shown = 0;
        let frame_bytes = width * height + 2 * width.div_ceil(2) * height.div_ceil(2);
        assert_eq!(reference.len(), check_frames.len() * frame_bytes);
        for chunk in bytes.chunks(257) {
            for packet in stream.push(chunk).unwrap() {
                for data in split_superframe(&packet.data).unwrap() {
                    let frame = decoder.decode(data).unwrap().unwrap();
                    assert_eq!((frame.width, frame.height), (width, height));
                    let Some(check) = check_frames.iter().position(|&index| index == shown) else {
                        shown += 1;
                        continue;
                    };
                    let mut offset = check * frame_bytes;
                    for (plane_index, (plane, width, height)) in [
                        (&frame.y, width, height),
                        (&frame.u, width.div_ceil(2), height.div_ceil(2)),
                        (&frame.v, width.div_ceil(2), height.div_ceil(2)),
                    ].into_iter().enumerate() {
                        for row in 0..height {
                            assert_eq!(&plane.pixels[row * plane.width..row * plane.width + width],
                                &reference[offset..offset + width],
                                "lossless frame {shown}, plane {plane_index}, row {row}");
                            offset += width;
                        }
                    }
                    shown += 1;
                }
            }
        }
        stream.finish().unwrap();
        assert_eq!(shown, 30);
    }

    #[test]
    fn decodes_serial_probability_context_across_stream_chunks() {
        let bytes = include_bytes!("../../tests/fixtures/vp9-serial-static.webm");
        let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
        let mut decoder = Vp9Decoder::new();
        let mut coded = 0;
        let mut shown = 0;
        for chunk in bytes.chunks(257) {
            for packet in stream.push(chunk).unwrap() {
                for data in split_superframe(&packet.data).unwrap() {
                    let header = FrameHeader::parse(data).unwrap();
                    if header.show_existing_frame.is_none() {
                        let layout = if header.key_frame {
                            KeyframeLayout::parse(data).unwrap()
                        } else {
                            let references = std::array::from_fn(|slot| decoder.references[slot]
                                .as_ref().map(|frame| ReferenceFrame {
                                    width: frame.width as u32, height: frame.height as u32,
                                    bit_depth: 8,
                                }));
                            KeyframeLayout::parse_interframe(data, &references,
                                decoder.segmentation).unwrap().1
                        };
                        assert!(!layout.frame_parallel_decoding);
                    }
                    if let Some(frame) = decoder.decode(data).unwrap() {
                        assert_eq!((frame.width, frame.height), (320, 240));
                        shown += 1;
                    }
                    coded += 1;
                }
            }
        }
        stream.finish().unwrap();
        assert_eq!((coded, shown), (90, 90));
    }

    #[test]
    fn decodes_serial_motion_and_inferred_syntax_counts() {
        let bytes = include_bytes!("../../tests/fixtures/vp9-serial-motion.webm");
        let reference = include_bytes!("../../tests/fixtures/vp9-serial-motion-check.yuv");
        let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
        let mut decoder = Vp9Decoder::new();
        let mut coded = 0;
        let mut shown = 0;
        let frame_bytes = 320 * 240 * 3 / 2;
        assert_eq!(reference.len(), 2 * frame_bytes);
        for chunk in bytes.chunks(257) {
            for packet in stream.push(chunk).unwrap() {
                for data in split_superframe(&packet.data).unwrap() {
                    if let Some(frame) = decoder.decode(data).unwrap() {
                        assert_eq!((frame.width, frame.height), (320, 240));
                        if let Some(check) = [3, 89].iter().position(|&index| index == shown) {
                            let reference = &reference[check * frame_bytes..(check + 1) * frame_bytes];
                            let mut offset = 0;
                            for (plane, width, height) in [
                                (&frame.y, 320, 240), (&frame.u, 160, 120), (&frame.v, 160, 120),
                            ] {
                                let mut error = 0u64;
                                for y in 0..height {
                                    for x in 0..width {
                                        error += u64::from(plane.pixels[y * plane.width + x]
                                            .abs_diff(reference[offset + y * width + x]));
                                    }
                                }
                                assert!(error * 20 < (width * height) as u64,
                                    "serial VP9 frame {shown} differs from reference");
                                offset += width * height;
                            }
                        }
                        shown += 1;
                    }
                    coded += 1;
                }
            }
        }
        stream.finish().unwrap();
        assert_eq!((coded, shown), (90, 90));
    }

    #[test]
    fn decodes_consecutive_frames_from_supplied_vp9_clip() {
        let Ok(sample) = std::env::var("WEBMEDIA_VP9_SAMPLE") else { return };
        let mut source = std::fs::File::open(sample).unwrap();
        let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
        let mut decoder = Vp9Decoder::new();
        let mut buffer = [0u8; 16 * 1024];
        let mut decoded = 0usize;
        let target = std::env::var("WEBMEDIA_VP9_TEST_FRAMES").ok()
            .and_then(|value| value.parse::<usize>().ok()).unwrap_or(10);
        let reference = std::env::var("WEBMEDIA_VP9_TEN_FRAMES").ok()
            .map(|path| std::fs::read(path).unwrap());
        let reference_frames = std::env::var("WEBMEDIA_VP9_REFERENCE_FRAMES").ok()
            .map(|value| value.split(',').map(|frame| frame.parse::<usize>().unwrap())
                .collect::<Vec<_>>());
        let mut shown_count = 0usize;
        let mut checked_frames = Vec::new();
        let mut dimensions = None;
        let convert_rgba = std::env::var_os("WEBMEDIA_VP9_TEST_RGBA").is_some();
        while decoded < target {
            let count = std::io::Read::read(&mut source, &mut buffer).unwrap();
            assert!(count != 0, "VP9 sample ends before {target} coded frames");
            for packet in stream.push(&buffer[..count]).unwrap() {
                for data in split_superframe(&packet.data).unwrap() {
                    let shown = decoder.decode(data)
                        .unwrap_or_else(|error| panic!("VP9 coded frame {decoded}: {error:?}"));
                    if let Some(frame) = shown {
                        assert!(frame.width > 0 && frame.height > 0);
                        assert_eq!(*dimensions.get_or_insert((frame.width, frame.height)),
                            (frame.width, frame.height));
                        let reference_index = reference_frames.as_ref().map_or(Some(shown_count),
                            |frames| frames.iter().position(|&index| index == shown_count));
                        if let (Some(reference), Some(reference_index)) = (&reference, reference_index) {
                            let chroma_width = frame.width.div_ceil(2);
                            let chroma_height = frame.height.div_ceil(2);
                            let luma_bytes = frame.width * frame.height;
                            let chroma_bytes = chroma_width * chroma_height;
                            let frame_bytes = luma_bytes + 2 * chroma_bytes;
                            assert!(reference.len() >= (reference_index + 1) * frame_bytes);
                            let planes = [
                                (&frame.y, frame.width, frame.height, 0),
                                (&frame.u, chroma_width, chroma_height, luma_bytes),
                                (&frame.v, chroma_width, chroma_height, luma_bytes + chroma_bytes),
                            ];
                            for (plane_index, (plane, width, height, offset)) in planes.into_iter().enumerate() {
                                let mut error = 0u64;
                                let mut mismatches = Vec::new();
                                for y in 0..height {
                                    for x in 0..width {
                                        let actual = plane.pixels[y * plane.width + x];
                                        let expected = reference[reference_index * frame_bytes + offset + y * width + x];
                                        error += u64::from(actual.abs_diff(expected));
                                        if actual != expected && mismatches.len() < 12 {
                                            mismatches.push((x, y, actual, expected));
                                        }
                                    }
                                }
                                let mae = error as f64 / (width * height) as f64;
                                eprintln!("VP9 frame {shown_count} plane {plane_index}: MAE={mae:.3}");
                                if std::env::var_os("WEBMEDIA_VP9_TEST_DIFF").is_some() {
                                    eprintln!("First differing pixels: {mismatches:?}");
                                }
                                assert!(mae < 3.0, "VP9 frame {shown_count} drifts from reference");
                                if std::env::var_os("WEBMEDIA_VP9_TEST_EXACT").is_some() {
                                    assert_eq!(error, 0, "VP9 frame {shown_count} plane {plane_index} is not pixel-exact");
                                }
                            }
                            checked_frames.push(shown_count);
                        }
                        if convert_rgba { std::hint::black_box(frame.rgba()); }
                        shown_count += 1;
                    }
                    decoded += 1;
                    if decoded == target { break; }
                }
                if decoded == target { break; }
            }
        }
        if reference.is_some() {
            if let Some(frames) = reference_frames {
                for index in frames {
                    assert!(checked_frames.contains(&index), "VP9 reference frame {index} was not reached");
                }
            }
        }
    }

    #[test]
    fn reconstructs_supplied_vp9_keyframe() {
        let (Ok(sample), Ok(reference)) = (
            std::env::var("WEBMEDIA_VP9_SAMPLE"),
            std::env::var("WEBMEDIA_VP9_REFERENCE"),
        ) else { return };
        let bytes = std::fs::read(sample).unwrap();
        let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
        let packet = stream.push(&bytes).unwrap().into_iter().next().unwrap();
        let encoded = split_superframe(&packet.data).unwrap();
        let frame = decode_keyframe(encoded[0]).unwrap();
        let reference = std::fs::read(reference).unwrap();
        let chroma_width = frame.width.div_ceil(2);
        let chroma_height = frame.height.div_ceil(2);
        let offsets = [0, frame.width * frame.height, frame.width * frame.height + chroma_width * chroma_height];
        for (index, plane) in [&frame.y, &frame.u, &frame.v].into_iter().enumerate() {
            let width = if index == 0 { frame.width } else { chroma_width };
            let height = if index == 0 { frame.height } else { chroma_height };
            let mut total_error = 0u64;
            for y in 0..height {
                for x in 0..width {
                    total_error += u64::from(plane.pixels[y * plane.width + x].abs_diff(
                        reference[offsets[index] + y * width + x]));
                }
            }
            let mae = total_error as f64 / (width * height) as f64;
            eprintln!("VP9 keyframe plane {index}: MAE={mae:.3}");
            assert!(mae < 1.0, "VP9 keyframe differs from reference");
        }
    }

    #[test]
    fn reconstructs_all_keyframes_in_supplied_vp9_clip() {
        let Ok(sample) = std::env::var("WEBMEDIA_VP9_SAMPLE") else { return };
        let mut source = std::fs::File::open(sample).unwrap();
        let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
        let mut buffer = [0u8; 16 * 1024];
        let mut count = 0usize;
        loop {
            let size = std::io::Read::read(&mut source, &mut buffer).unwrap();
            if size == 0 {
                break;
            }
            for packet in stream.push(&buffer[..size]).unwrap() {
                for data in split_superframe(&packet.data).unwrap() {
                    if !super::super::vp9::FrameHeader::parse(data).unwrap().key_frame {
                        continue;
                    }
                    let frame = decode_keyframe(data)
                        .unwrap_or_else(|error| panic!("VP9 keyframe {count}: {error:?}"));
                    assert_eq!((frame.width, frame.height), (1920, 1080));
                    count += 1;
                }
            }
        }
        stream.finish().unwrap();
        assert!(count > 1);
    }

    #[test]
    fn predicts_first_interframe_block_against_reference() {
        let (Ok(sample), Ok(reference)) = (
            std::env::var("WEBMEDIA_VP9_SAMPLE"),
            std::env::var("WEBMEDIA_VP9_TWO_FRAMES"),
        ) else { return };
        let mut source = std::fs::File::open(sample).unwrap();
        let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
        let mut encoded = Vec::new();
        let mut buffer = [0u8; 16 * 1024];
        while encoded.len() < 2 {
            let count = std::io::Read::read(&mut source, &mut buffer).unwrap();
            assert!(count != 0, "VP9 sample has fewer than two frames");
            for packet in stream.push(&buffer[..count]).unwrap() {
                encoded.extend(split_superframe(&packet.data).unwrap().into_iter().map(Vec::from));
            }
        }
        let key = decode_keyframe(&encoded[0]).unwrap();
        let key_layout = KeyframeLayout::parse(&encoded[0]).unwrap();
        let key_probabilities = CompressedHeader::parse_keyframe(&key_layout).unwrap();
        let reference_slots = [Some(super::super::vp9::ReferenceFrame {
            width: key.width as u32, height: key.height as u32, bit_depth: 8,
        }); 8];
        let (inter, layout, _) = KeyframeLayout::parse_interframe(
            &encoded[1], &reference_slots, super::super::vp9::SegmentationState::default(),
        ).unwrap();
        let probabilities = CompressedHeader::parse_interframe(&layout, &inter, &key_probabilities).unwrap();
        let first = super::super::vp9_tile::first_inter_block_mode(
            &layout, &inter, &probabilities, layout.tile_partitions().unwrap()[0],
        ).unwrap();
        let tile_blocks = super::super::vp9_tile::decode_interframe_tile_prefix(
            &layout, &inter, &probabilities, layout.tile_partitions().unwrap()[0], 0, 1, None, None,
        ).unwrap();
        assert_eq!(tile_blocks.len(), 1);
        let parsed = tile_blocks[0].block.inter.unwrap();
        assert_eq!(tile_blocks[0].block.skip, first.skip);
        assert_eq!((parsed.reference, parsed.mode, parsed.interpolation_filter, parsed.motion),
            (first.reference, first.mode, first.interpolation_filter, first.motion));
        assert!(first.skip && first.reference == 1);
        let expected = std::fs::read(reference).unwrap();
        let frame_bytes = key.width * key.height * 3 / 2;
        assert!(expected.len() >= 2 * frame_bytes);
        let (decoded, _) = decode_interframe(&layout, &inter, &probabilities, [&key; 3], None).unwrap();
        for (index, (plane, width, height, offset)) in [
            (&decoded.y, key.width, key.height, 0),
            (&decoded.u, key.width / 2, key.height / 2, key.width * key.height),
            (&decoded.v, key.width / 2, key.height / 2,
                key.width * key.height + key.width * key.height / 4),
        ].into_iter().enumerate() {
            let mut error = 0u64;
            for y in 0..height {
                for x in 0..width {
                    error += u64::from(plane.pixels[y * plane.width + x].abs_diff(
                        expected[frame_bytes + offset + y * width + x]));
                }
            }
            let mae = error as f64 / (width * height) as f64;
            eprintln!("VP9 decoded interframe plane {index}: MAE={mae:.3}");
            assert!(mae < 1.0, "VP9 interframe differs from reference");
        }
        let planes = [
            (&key.y, key.width, key.height, 64usize, 0usize),
            (&key.u, key.width / 2, key.height / 2, 32, key.width * key.height),
            (&key.v, key.width / 2, key.height / 2, 32,
                key.width * key.height + key.width * key.height / 4),
        ];
        {
            let two = super::super::vp9_tile::decode_interframe_tile_prefix(
                &layout, &inter, &probabilities, layout.tile_partitions().unwrap()[0], 0, 2, None, None,
            ).unwrap();
            if let Some(second) = two.get(1) {
                    let second_mode = second.block.inter.unwrap();
                    let sampling = super::super::vp9_motion::scaled_motion(
                        (key.width, key.height), (key.width, key.height),
                        (second.x, second.y), (second.x / 8, second.y / 8), (8, 8),
                        second_mode.motion, false,
                    ).unwrap();
                    let mut predicted = Plane::new(64, 64);
                    super::super::vp9_motion::predict_block(
                        &mut predicted, &key.y, 0, 0, 64, 64,
                        sampling.start_x, sampling.start_y, sampling.step_x, sampling.step_y,
                        second_mode.interpolation_filter, key.width, key.height,
                    ).unwrap();
                    let error: u64 = (0..64).flat_map(|y| (0..64).map(move |x| (x, y)))
                        .map(|(x, y)| u64::from(predicted.pixels[y * 64 + x].abs_diff(
                            expected[frame_bytes + y * key.width + second.x + x],
                        ))).sum();
                    assert!(error < 4096, "VP9 second interframe block differs from reference");
            }
        }
        for (index, (reference_plane, width, height, size, offset)) in planes.into_iter().enumerate() {
            let mut predicted = Plane::new(size, size);
            let sampling = super::super::vp9_motion::scaled_motion(
                (key.width, key.height), (key.width, key.height), (0, 0), (0, 0), (8, 8),
                first.motion, index != 0,
            ).unwrap();
            super::super::vp9_motion::predict_block(
                &mut predicted, reference_plane, 0, 0, size, size,
                sampling.start_x, sampling.start_y, sampling.step_x, sampling.step_y,
                first.interpolation_filter, width, height,
            ).unwrap();
            let mut total_error = 0u64;
            for row in 0..size {
                for col in 0..size {
                    total_error += u64::from(predicted.pixels[row * size + col].abs_diff(
                        expected[frame_bytes + offset + row * width + col],
                    ));
                }
            }
            let mae = total_error as f64 / (size * size) as f64;
            eprintln!("VP9 first interframe block plane {index}: MAE={mae:.3}");
            assert!(mae < 1.0, "VP9 motion prediction differs from reference");
        }
    }
}
