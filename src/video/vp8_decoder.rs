//! VP8 reference-frame reconstruction across keyframes and interframes.

use super::backend::MediaDecodeError;
use super::vp8::FrameHeader;
#[cfg(test)]
use super::vp8::KeyFrameLayout;
use super::vp8_filter::{filter_frame, FilterMacroblock};
use super::vp8_inter::{InterFrameLayout, InterMacroblock, InterState};
use super::vp8_keyframe::{decode_keyframe_with_state, reconstruct_intra, YuvKeyFrame};
use super::vp8_motion::{chroma_vector, predict_block};
use super::vp8_residue::{ResidualMacroblock, ResidueDecoder};
use std::sync::Arc;

pub(super) struct Vp8Decoder {
    state: Option<InterState>,
    last: Option<Arc<YuvKeyFrame>>,
    golden: Option<Arc<YuvKeyFrame>>,
    alternate: Option<Arc<YuvKeyFrame>>,
    retired: Vec<Arc<YuvKeyFrame>>,
}

fn prefill_zero_reference(
    frame: &mut YuvKeyFrame,
    macroblocks: usize,
    counts: [usize; 4],
    references: [Option<&YuvKeyFrame>; 3],
) -> Option<u8> {
    if macroblocks == 0 { return None; }
    let selected = (1..4).max_by_key(|&reference| counts[reference]).unwrap();
    if counts[selected] < macroblocks - macroblocks / 4 { return None; }
    let reference = references[selected - 1]?;
    if reference.width != frame.width || reference.height != frame.height
        || reference.y.width != frame.y.width || reference.y.pixels.len() != frame.y.pixels.len()
        || reference.u.width != frame.u.width || reference.u.pixels.len() != frame.u.pixels.len()
        || reference.v.width != frame.v.width || reference.v.pixels.len() != frame.v.pixels.len()
    { return None; }
    // Intra prediction reads only completed raster neighbors, not these future cells.
    frame.y.pixels.copy_from_slice(&reference.y.pixels);
    frame.u.pixels.copy_from_slice(&reference.u.pixels);
    frame.v.pixels.copy_from_slice(&reference.v.pixels);
    Some(selected as u8)
}

fn add_prefilled_residuals(frame: &mut YuvKeyFrame, mb_x: usize, mb_y: usize,
    blocks: &ResidualMacroblock,
) {
    if !blocks.has_coefficients { return; }
    for row in 0..4 {
        for col in 0..4 {
            frame.y.add_residual(mb_x * 16 + col * 4, mb_y * 16 + row * 4,
                &blocks.y[row * 4 + col]);
        }
    }
    for (plane, values) in [(&mut frame.u, &blocks.u), (&mut frame.v, &blocks.v)] {
        for row in 0..2 {
            for col in 0..2 {
                plane.add_residual(mb_x * 8 + col * 4, mb_y * 8 + row * 4,
                    &values[row * 2 + col]);
            }
        }
    }
}

impl Vp8Decoder {
    pub(super) fn new() -> Self {
        Self {
            state: None,
            last: None,
            golden: None,
            alternate: None,
            retired: Vec::new(),
        }
    }

    pub(super) fn decode(&mut self, packet: &[u8]) -> Result<Arc<YuvKeyFrame>, MediaDecodeError> {
        #[cfg(test)]
        let profile = std::env::var_os("WEBMEDIA_VP8_PROFILE").is_some();
        #[cfg(test)]
        let started = profile.then(std::time::Instant::now);
        let header = FrameHeader::parse(packet)?;
        if header.version > 3 { return Err(MediaDecodeError::Unsupported); }
        if header.key_frame {
            let (frame, state) = decode_keyframe_with_state(packet)?;
            self.retired.clear();
            self.state = Some(state);
            let reference = Arc::new(frame);
            self.last = Some(reference.clone());
            self.golden = Some(reference.clone());
            self.alternate = Some(reference.clone());
            #[cfg(test)]
            if let Some(started) = started {
                eprintln!("VP8 keyframe total_us={}", started.elapsed().as_micros());
            }
            return Ok(reference);
        }
        let mut state = self.state.as_ref().ok_or_else(|| {
            MediaDecodeError::InvalidData("VP8 interframe before keyframe".into())
        })?.clone();
        let mut layout = InterFrameLayout::parse(packet, &mut state)?;
        let mb_width = state.mb_width;
        let mb_height = state.mb_height;
        let (modes, zero_references) = layout
            .read_macroblocks_counted(&mut state, mb_width, mb_height)
            .map_err(|error| {
                MediaDecodeError::InvalidData(format!("VP8 macroblock modes: {error:?}"))
            })?;
        #[cfg(test)]
        let modes_done = profile.then(std::time::Instant::now);
        #[cfg(test)]
        if std::env::var_os("WEBMEDIA_VP8_REPORT").is_some() {
            let mut counts = [0usize; 10];
            for mode in &modes {
                counts[mode.mode as usize] += 1;
            }
            eprintln!("VP8 modes {counts:?} token bytes {:?}", layout.token_partitions.iter().map(|part| part.len()).collect::<Vec<_>>());
            if mb_width <= 3 {
                eprintln!("VP8 block modes {:?}", modes.iter().map(|mode| (mode.mode, mode.reference, mode.skip_coefficients)).collect::<Vec<_>>());
            }
        }
        let mut residues = ResidueDecoder::new(&layout)?;
        let last = self.last.as_ref().unwrap();
        let (width, height) = (last.width, last.height);
        let mut frame = self.take_frame(width, height);
        let prefilled = prefill_zero_reference(&mut frame, modes.len(), zero_references,
            [self.last.as_deref(), self.golden.as_deref(), self.alternate.as_deref()]);
        let mut filter_settings = Vec::with_capacity(modes.len());
        #[cfg(test)]
        let setup_done = profile.then(std::time::Instant::now);
        #[cfg(test)]
        let (mut coefficients_ns, mut prediction_ns) = (0u128, 0u128);
        for mb_y in 0..mb_height {
            for mb_x in 0..mb_width {
                let index = mb_y * mb_width + mb_x;
                let mode = &modes[index];
                #[cfg(test)]
                let coefficients_started = profile.then(std::time::Instant::now);
                let blocks = residues
                    .decode(&layout, mode, mb_x, mb_y)
                    .map_err(|error| {
                        MediaDecodeError::InvalidData(format!(
                            "VP8 residual at ({mb_x}, {mb_y}), mode={} ref={} skip={}: {error:?}",
                            mode.mode, mode.reference, mode.skip_coefficients
                        ))
                    })?;
                #[cfg(test)]
                let prediction_started = profile.then(std::time::Instant::now);
                #[cfg(test)]
                if let (Some(before), Some(after)) = (coefficients_started, prediction_started) {
                    coefficients_ns += after.duration_since(before).as_nanos();
                }
                if mode.reference == 0 {
                    reconstruct_intra(
                        &mut frame,
                        mb_x,
                        mb_y,
                        mode.luma,
                        mode.chroma,
                        &mode.subblocks,
                        &blocks,
                    );
                } else {
                    let reference = match mode.reference {
                        1 => self.last.as_ref(),
                        2 => self.golden.as_ref(),
                        3 => self.alternate.as_ref(),
                        _ => None,
                    }
                    .ok_or_else(|| {
                        MediaDecodeError::InvalidData("missing VP8 reference frame".into())
                    })?;
                    if prefilled == Some(mode.reference) && mode.mode == 7 {
                        add_prefilled_residuals(&mut frame, mb_x, mb_y, &blocks);
                    } else {
                        reconstruct_inter(
                            &mut frame,
                            reference,
                            mb_x,
                            mb_y,
                            mode,
                            &blocks,
                            layout.version,
                        );
                    }
                }
                #[cfg(test)]
                if let Some(before) = prediction_started {
                    prediction_ns += before.elapsed().as_nanos();
                }
                filter_settings.push(FilterMacroblock {
                    level: layout.macroblock_filter_level(mode),
                    skip_inner: mode.mode != 4 && mode.mode != 9 && !blocks.has_coefficients,
                });
            }
        }
        #[cfg(test)]
        let filter_started = profile.then(std::time::Instant::now);
        filter_frame(
            &mut frame.y,
            &mut frame.u,
            &mut frame.v,
            mb_width,
            mb_height,
            &filter_settings,
            layout.sharpness,
            layout.simple_filter,
            false,
        );
        #[cfg(test)]
        if let (Some(started), Some(modes_done), Some(setup_done), Some(filter_started)) =
            (started, modes_done, setup_done, filter_started)
        {
            eprintln!("VP8 interframe modes_us={} setup_us={} coefficients_us={} prediction_us={} filter_us={} total_us={}",
                modes_done.duration_since(started).as_micros(), setup_done.duration_since(modes_done).as_micros(),
                coefficients_ns / 1000, prediction_ns / 1000,
                filter_started.elapsed().as_micros(), started.elapsed().as_micros());
        }
        let refreshed = Arc::new(frame);
        let old_last = self.last.clone();
        let old_golden = self.golden.clone();
        let old_alternate = self.alternate.clone();
        if layout.refresh_golden {
            self.golden = Some(refreshed.clone());
        } else {
            self.golden = match layout.copy_to_golden {
                1 => old_last.clone(),
                2 => old_alternate.clone(),
                _ => old_golden.clone(),
            };
        }
        if layout.refresh_alternate {
            self.alternate = Some(refreshed.clone());
        } else {
            self.alternate = match layout.copy_to_alternate {
                1 => old_last.clone(),
                2 => old_golden.clone(),
                _ => old_alternate.clone(),
            };
        }
        if layout.refresh_last {
            self.last = Some(refreshed.clone());
        }
        // Failed frames must not commit entropy, segmentation, or filter updates.
        self.state = Some(state);
        for frame in [old_last, old_golden, old_alternate].into_iter().flatten() {
            self.retire_frame(frame);
        }
        Ok(refreshed)
    }

    fn take_frame(&mut self, width: usize, height: usize) -> YuvKeyFrame {
        if let Some(index) = self.retired.iter().position(|frame| {
            frame.width == width && frame.height == height && Arc::strong_count(frame) == 1
        }) {
            match Arc::try_unwrap(self.retired.swap_remove(index)) {
                Ok(frame) => return frame,
                // A concurrent weak-reference upgrade must not expose a mutable frame.
                Err(frame) => self.retired.push(frame),
            }
        }
        YuvKeyFrame::new(width, height)
    }

    fn retire_frame(&mut self, frame: Arc<YuvKeyFrame>) {
        if [&self.last, &self.golden, &self.alternate].into_iter().flatten()
            .chain(self.retired.iter()).any(|other| Arc::ptr_eq(other, &frame)) {
            return;
        }
        // Bound retained storage even when the compositor holds older frames.
        if self.retired.len() == 3 { self.retired.remove(0); }
        self.retired.push(frame);
    }
}

fn reconstruct_inter(
    frame: &mut YuvKeyFrame,
    reference: &YuvKeyFrame,
    mb_x: usize,
    mb_y: usize,
    mode: &InterMacroblock,
    blocks: &ResidualMacroblock,
    version: u8,
) {
    let uniform = mode.motion.iter().all(|&motion| motion == mode.motion[0]);
    if uniform {
        predict_block(&mut frame.y, &reference.y, mb_x * 16, mb_y * 16, 16, 16,
            mode.motion[0], false, version != 0, reference.y.width,
            reference.y.pixels.len() / reference.y.width);
    }
    let partitioned = !uniform && mode.split_partition < 3;
    if partitioned {
        let (width, height, count) = match mode.split_partition {
            0 => (16, 8, 2),
            1 => (8, 16, 2),
            _ => (8, 8, 4),
        };
        for piece in 0..count {
            let (row, col) = match mode.split_partition {
                0 => (piece * 2, 0),
                1 => (0, piece * 2),
                _ => (piece / 2 * 2, piece % 2 * 2),
            };
            predict_block(&mut frame.y, &reference.y, mb_x * 16 + col * 4,
                mb_y * 16 + row * 4, width, height, mode.motion[row * 4 + col], false,
                version != 0, reference.y.width, reference.y.pixels.len() / reference.y.width);
        }
    }
    if !uniform || blocks.has_coefficients {
        for row in 0..4 {
            for col in 0..4 {
                let index = row * 4 + col;
                let x = mb_x * 16 + col * 4;
                let y = mb_y * 16 + row * 4;
                if !uniform && !partitioned { predict_block(
                    &mut frame.y, &reference.y, x, y, 4, 4, mode.motion[index], false,
                    version != 0, reference.y.width, reference.y.pixels.len() / reference.y.width,
                ); }
                if blocks.has_coefficients {
                    frame.y.add_residual(x, y, &blocks.y[index]);
                }
            }
        }
    }
    let mut chroma_motion = if uniform {
        [chroma_vector(&mode.motion, 0, 0); 4]
    } else {
        std::array::from_fn(|index| chroma_vector(&mode.motion, index / 2, index % 2))
    };
    if version == 3 {
        for motion in &mut chroma_motion {
            // RFC 6386 section 18.1 truncates signed chroma fractions to whole pixels.
            motion.row &= !7;
            motion.col &= !7;
        }
    }
    let chroma_uniform = uniform || chroma_motion[1..].iter().all(|&motion| motion == chroma_motion[0]);
    for (plane, reference_plane, residuals) in [
        (&mut frame.u, &reference.u, &blocks.u),
        (&mut frame.v, &reference.v, &blocks.v),
    ] {
        if chroma_uniform {
            predict_block(plane, reference_plane, mb_x * 8, mb_y * 8, 8, 8,
                chroma_motion[0], true, version != 0,
                reference_plane.width, reference_plane.pixels.len() / reference_plane.width);
        }
        let chroma_partitioned = !chroma_uniform && mode.split_partition < 2;
        if chroma_partitioned {
            for piece in 0..2 {
                let (row, col, width, height) = if mode.split_partition == 0 { (piece, 0, 8, 4) }
                    else { (0, piece, 4, 8) };
                predict_block(plane, reference_plane, mb_x * 8 + col * 4, mb_y * 8 + row * 4,
                    width, height, chroma_motion[row * 2 + col], true, version != 0,
                    reference_plane.width, reference_plane.pixels.len() / reference_plane.width);
            }
        }
        if (chroma_uniform || chroma_partitioned) && !blocks.has_coefficients { continue; }
        for row in 0..2 {
            for col in 0..2 {
                let index = row * 2 + col;
                let x = mb_x * 8 + col * 4;
                let y = mb_y * 8 + row * 4;
                if !chroma_uniform && !chroma_partitioned { predict_block(
                    plane,
                    reference_plane,
                    x,
                    y,
                    4,
                    4,
                    chroma_motion[index],
                    true,
                    version != 0,
                    reference_plane.width,
                    reference_plane.pixels.len() / reference_plane.width,
                ); }
                if blocks.has_coefficients {
                    plane.add_residual(x, y, &residuals[index]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::webm::WebmVp8Stream;

    #[test]
    fn prefill_threshold_and_reference_compatibility() {
        let mut references = [YuvKeyFrame::new(31, 29), YuvKeyFrame::new(31, 29), YuvKeyFrame::new(31, 29)];
        for (index, reference) in references.iter_mut().enumerate() {
            for plane in [&mut reference.y, &mut reference.u, &mut reference.v] {
                plane.pixels.fill((index * 53 + 17) as u8);
            }
        }
        let unchanged = |a: &YuvKeyFrame, b: &YuvKeyFrame| {
            assert_eq!(a.y.pixels, b.y.pixels); assert_eq!(a.u.pixels, b.u.pixels);
            assert_eq!(a.v.pixels, b.v.pixels);
        };
        for total in 0..=32 {
            for count in 0..=total {
                for selected in 1..=3 {
                    let mut frame = YuvKeyFrame::new(31, 29);
                    for plane in [&mut frame.y, &mut frame.u, &mut frame.v] { plane.pixels.fill(211); }
                    let prior = frame.clone();
                    let mut counts = [0; 4]; counts[selected] = count;
                    let result = prefill_zero_reference(&mut frame, total, counts, references.each_ref().map(Some));
                    if total != 0 && count >= total - total / 4 {
                        assert_eq!(result, Some(selected as u8));
                        unchanged(&frame, &references[selected - 1]);
                    } else { assert_eq!(result, None); unchanged(&frame, &prior); }
                }
            }
        }
        for fault in 0..9 {
            let mut bad = references[1].clone();
            match fault {
                0 => bad.width += 1,
                1 => bad.height += 1,
                2 => bad.y.width += 1,
                3 => bad.u.width += 1,
                4 => bad.v.width += 1,
                5 => { bad.y.pixels.pop(); },
                6 => { bad.u.pixels.pop(); },
                7 => { bad.v.pixels.pop(); },
                _ => {},
            }
            let mut frame = YuvKeyFrame::new(31, 29);
            for plane in [&mut frame.y, &mut frame.u, &mut frame.v] { plane.pixels.fill(211); }
            let prior = frame.clone();
            let golden = if fault == 8 { None } else { Some(&bad) };
            assert_eq!(prefill_zero_reference(&mut frame, 16, [0, 2, 12, 2],
                [Some(&references[0]), golden, Some(&references[2])]), None);
            unchanged(&frame, &prior);
            // A declined prefill leaves the original motion path available.
            let mode = InterMacroblock { reference: 1, mode: 7, ..Default::default() };
            let mut expected = prior;
            reconstruct_inter(&mut frame, &references[0], 0, 0, &mode, &ResidualMacroblock::default(), 0);
            reconstruct_inter_unbatched(&mut expected, &references[0], 0, 0, &mode, &ResidualMacroblock::default(), 0);
            unchanged(&frame, &expected);
        }
    }

    #[test]
    fn prefilled_mixed_raster_preserves_intra_residuals_and_padding() {
        let mut cases = 0;
        for (width, height) in [(64, 64), (61, 57)] {
            for version in 0..4 {
                for luma in 0..5 {
                    for chroma in 0..4 {
                        for pattern in 0..10 {
                            let mut reference = YuvKeyFrame::new(width, height);
                            for (p, plane) in [&mut reference.y, &mut reference.u, &mut reference.v].into_iter().enumerate() {
                                for (i, pixel) in plane.pixels.iter_mut().enumerate() {
                                    *pixel = ((i * 47 + p * 59 + pattern * 13) % 256) as u8;
                                }
                            }
                            let held = reference.clone();
                            let mut baseline = YuvKeyFrame::new(width, height);
                            let mut prefilled = YuvKeyFrame::new(width, height);
                            assert_eq!(prefill_zero_reference(&mut prefilled, 16, [0, 12, 0, 0],
                                [Some(&reference), None, None]), Some(1));
                            let mut poisoned = YuvKeyFrame::new(width, height);
                            for plane in [&mut poisoned.y, &mut poisoned.u, &mut poisoned.v] { plane.pixels.fill(255); }
                            for mb in 0..16 {
                                let (mx, my) = (mb % 4, mb / 4);
                                let mut residual = ResidualMacroblock::default();
                                residual.has_coefficients = pattern % 3 != 0;
                                if residual.has_coefficients {
                                    let values = [i32::MIN, -256, -1, 0, 1, 255, i32::MAX];
                                    residual.y[(mb + pattern) % 16][pattern % 16] = values[(mb + pattern) % 7];
                                    residual.u[pattern % 4][mb % 16] = values[(mb + 2) % 7];
                                    residual.v[mb % 4][pattern % 16] = values[(mb + 4) % 7];
                                    if pattern == 8 { for block in &mut residual.y { block.fill(17); } }
                                }
                                if [0, 3, 6, 13].contains(&mb) {
                                    let subblocks = std::array::from_fn(|i| ((i + pattern) % 10) as u8);
                                    for frame in [&mut baseline, &mut prefilled, &mut poisoned] {
                                        reconstruct_intra(frame, mx, my, luma, chroma, &subblocks, &residual);
                                    }
                                } else {
                                    let mode = InterMacroblock { reference: 1, mode: 7, ..Default::default() };
                                    reconstruct_inter(&mut baseline, &reference, mx, my, &mode, &residual, version);
                                    reconstruct_inter(&mut poisoned, &reference, mx, my, &mode, &residual, version);
                                    add_prefilled_residuals(&mut prefilled, mx, my, &residual);
                                }
                                for (a, b, c, size) in [(&baseline.y, &prefilled.y, &poisoned.y, 16),
                                    (&baseline.u, &prefilled.u, &poisoned.u, 8), (&baseline.v, &prefilled.v, &poisoned.v, 8)] {
                                    for row in my * size..(my + 1) * size {
                                        let start = row * a.width + mx * size;
                                        assert_eq!(&a.pixels[start..start + size], &b.pixels[start..start + size]);
                                        assert_eq!(&a.pixels[start..start + size], &c.pixels[start..start + size]);
                                    }
                                }
                            }
                            for (a, b, c, source, old) in [(&baseline.y, &prefilled.y, &poisoned.y, &reference.y, &held.y),
                                (&baseline.u, &prefilled.u, &poisoned.u, &reference.u, &held.u),
                                (&baseline.v, &prefilled.v, &poisoned.v, &reference.v, &held.v)] {
                                assert_eq!(a.pixels, b.pixels); assert_eq!(a.pixels, c.pixels);
                                assert_eq!(source.pixels, old.pixels);
                            }
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 1600);
    }

    fn reconstruct_inter_unbatched(frame: &mut YuvKeyFrame, reference: &YuvKeyFrame,
        mb_x: usize, mb_y: usize, mode: &InterMacroblock, blocks: &ResidualMacroblock, version: u8,
    ) {
        for plane_index in 0..3 {
            let (plane, source, residuals, size) = match plane_index {
                0 => (&mut frame.y, &reference.y, &blocks.y[..], 16),
                1 => (&mut frame.u, &reference.u, &blocks.u[..], 8),
                _ => (&mut frame.v, &reference.v, &blocks.v[..], 8),
            };
            for row in 0..size / 4 {
                for col in 0..size / 4 {
                    let index = row * (size / 4) + col;
                    let mut motion = if plane_index == 0 { mode.motion[index] }
                        else { chroma_vector(&mode.motion, row, col) };
                    if plane_index != 0 && version == 3 {
                        motion.row &= !7;
                        motion.col &= !7;
                    }
                    let (x, y) = (mb_x * size + col * 4, mb_y * size + row * 4);
                    predict_block(plane, source, x, y, 4, 4, motion, plane_index != 0,
                        version != 0, source.width, source.pixels.len() / source.width);
                    if blocks.has_coefficients { plane.add_residual(x, y, &residuals[index]); }
                }
            }
        }
    }

    #[test]
    fn batched_motion_matches_subblock_prediction() {
        use super::super::vp8_inter::MotionVector;
        let mut reference = YuvKeyFrame::new(32, 32);
        for plane in [&mut reference.y, &mut reference.u, &mut reference.v] {
            for (index, pixel) in plane.pixels.iter_mut().enumerate() {
                *pixel = (index * 73 ^ index / plane.width) as u8;
            }
        }
        let mut random = 73u32;
        for trial in 0..256 {
            let mut mode = InterMacroblock::default();
            mode.split_partition = match trial % 7 { 1 => 2, 4 => 0, 6 => 1, _ => 3 };
            mode.motion = std::array::from_fn(|index| {
                random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                let component = ((random >> 24) as i16 % 65) - 32;
                let value = match trial % 7 {
                    0 => 5,
                    1 => (index / 8 * 2 + index % 4 / 2) as i16 * 7 - 9,
                    2 => if index % 2 == 0 { 3 } else { -3 },
                    3 => component,
                    4 => if index < 8 { -5 } else { 13 },
                    5 => if index == 7 { component } else { 7 },
                    _ => if index % 4 < 2 { -5 } else { 13 },
                };
                MotionVector { row: value, col: -value }
            });
            for has_coefficients in [false, true] {
                let mut blocks = ResidualMacroblock::default();
                blocks.has_coefficients = has_coefficients;
                if has_coefficients {
                    for residual in blocks.y.iter_mut().chain(&mut blocks.u).chain(&mut blocks.v) {
                        for value in residual {
                            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                            *value = ((random >> 24) as i32 - 128) * 3;
                        }
                    }
                }
                for version in 0..4 {
                    for (mb_x, mb_y) in [(0, 0), (1, 1)] {
                        let mut actual = YuvKeyFrame::new(32, 32);
                        for plane in [&mut actual.y, &mut actual.u, &mut actual.v] { plane.pixels.fill(91); }
                        let mut expected = actual.clone();
                        reconstruct_inter(&mut actual, &reference, mb_x, mb_y, &mode, &blocks, version);
                        reconstruct_inter_unbatched(&mut expected, &reference, mb_x, mb_y, &mode, &blocks, version);
                        assert_eq!(actual.y.pixels, expected.y.pixels, "trial={trial} version={version}");
                        assert_eq!(actual.u.pixels, expected.u.pixels, "trial={trial} version={version}");
                        assert_eq!(actual.v.pixels, expected.v.pixels, "trial={trial} version={version}");
                    }
                }
            }
        }
    }

    #[test]
    fn frame_pool_preserves_held_frames_and_is_bounded() {
        let mut decoder = Vp8Decoder::new();
        let mut frame = YuvKeyFrame::new(16, 16);
        frame.y.pixels.fill(123);
        let held = Arc::new(frame);
        let pointer = held.y.pixels.as_ptr();
        decoder.retire_frame(held.clone());
        decoder.retire_frame(held.clone());
        assert_eq!(decoder.retired.len(), 1);
        let fresh = decoder.take_frame(16, 16);
        assert_ne!(fresh.y.pixels.as_ptr(), pointer);
        assert!(held.y.pixels.iter().all(|&pixel| pixel == 123));
        drop(held);
        let reused = decoder.take_frame(16, 16);
        assert_eq!(reused.y.pixels.as_ptr(), pointer);
        assert!(reused.y.pixels.iter().all(|&pixel| pixel == 123));
        assert!(decoder.retired.is_empty());
        for _ in 0..8 { decoder.retire_frame(Arc::new(YuvKeyFrame::new(16, 16))); }
        assert_eq!(decoder.retired.len(), 3);
        let fresh = decoder.take_frame(32, 16);
        assert_eq!(fresh.width, 32);
        assert_eq!(decoder.retired.len(), 3);
        let active = Arc::new(YuvKeyFrame::new(16, 16));
        decoder.last = Some(active.clone());
        decoder.retire_frame(active.clone());
        assert!(decoder.retired.iter().all(|frame| !Arc::ptr_eq(frame, &active)));
    }

    #[test]
    #[ignore = "manual split-motion reconstruction kernel timing"]
    fn benchmark_motion_batches() {
        use std::hint::black_box;
        use std::time::Instant;
        use super::super::vp8_inter::MotionVector;
        let mut reference = YuvKeyFrame::new(64, 64);
        for plane in [&mut reference.y, &mut reference.u, &mut reference.v] {
            for (index, pixel) in plane.pixels.iter_mut().enumerate() { *pixel = (index * 73) as u8; }
        }
        let mut frame = YuvKeyFrame::new(32, 32);
        let blocks = ResidualMacroblock::default();
        for grouped in [false, true] {
            let mut mode = InterMacroblock::default();
            mode.split_partition = if grouped { 2 } else { 3 };
            mode.motion = std::array::from_fn(|index| {
                let value = if grouped { (index / 8 * 2 + index % 4 / 2) as i16 * 7 - 9 }
                    else { (index * 13 % 47) as i16 - 23 };
                MotionVector { row: value, col: -value }
            });
            for trial in 0..5 {
                for batched in if trial % 2 == 0 { [false, true] } else { [true, false] } {
                    let start = Instant::now();
                    for _ in 0..20_000 {
                        if batched {
                            reconstruct_inter(black_box(&mut frame), black_box(&reference), 1, 1,
                                black_box(&mode), black_box(&blocks), 0);
                        } else {
                            reconstruct_inter_unbatched(black_box(&mut frame), black_box(&reference), 1, 1,
                                black_box(&mode), black_box(&blocks), 0);
                        }
                        black_box(&frame);
                    }
                    eprintln!("VP8 motion grouped={grouped} trial={trial} batched={batched} elapsed={:?}", start.elapsed());
                }
            }
        }
    }

    #[test]
    fn reused_dirty_planes_match_fresh_reconstruction() {
        let mut poisoned = 0;
        for bytes in [include_bytes!("../../tests/fixtures/vp8-motion.webm").as_slice(),
            include_bytes!("../../tests/fixtures/vp8-edges.webm").as_slice(),
            include_bytes!("../../tests/fixtures/vp8-odd-edges.webm").as_slice()] {
            let mut stream = WebmVp8Stream::new();
            let packets = stream.push(bytes).unwrap();
            let mut reused = Vp8Decoder::new();
            let mut fresh = Vp8Decoder::new();
            let mut held = Vec::new();
            for (index, packet) in packets.iter().enumerate() {
                for frame in &mut reused.retired {
                    if let Some(frame) = Arc::get_mut(frame) {
                        frame.y.pixels.fill(199);
                        frame.u.pixels.fill(31);
                        frame.v.pixels.fill(240);
                        poisoned += 1;
                    }
                }
                fresh.retired.clear();
                let actual = reused.decode(&packet.data).unwrap();
                let expected = fresh.decode(&packet.data).unwrap();
                assert_eq!(actual.y.pixels, expected.y.pixels, "frame={index}");
                assert_eq!(actual.u.pixels, expected.u.pixels, "frame={index}");
                assert_eq!(actual.v.pixels, expected.v.pixels, "frame={index}");
                if index % 11 == 0 {
                    held.push((actual.clone(), actual.y.pixels.clone(), actual.u.pixels.clone(), actual.v.pixels.clone()));
                }
                assert!(reused.retired.len() <= 3);
            }
            for (frame, y, u, v) in held {
                assert_eq!(frame.y.pixels, y);
                assert_eq!(frame.u.pixels, u);
                assert_eq!(frame.v.pixels, v);
            }
            reused.decode(&packets[0].data).unwrap();
            assert!(reused.retired.is_empty());
        }
        assert!(poisoned > 0);
    }

    #[test]
    fn segment_feature_modes_match_reference() {
        let key = include_bytes!("../../tests/fixtures/vp8-keyframe.ivf");
        let length = u32::from_le_bytes(key[32..36].try_into().unwrap()) as usize;
        let key = &key[44..44 + length];
        let mut decoder = Vp8Decoder::new();
        let mut packets = vec![key.to_vec()];
        let mut frames = vec![decoder.decode(key).unwrap()];
        let delta = (false, [-20, 0, 40, 90], [-20, 0, 20, 63]);
        let absolute = (true, [0, 20, 70, 120], [0, 10, 40, 63]);
        for (enabled, features, map_update) in [
            (true, Some(delta), true),
            (true, Some(absolute), false),
            (true, None, false),
            (false, None, false),
            (true, None, false),
            (true, Some((true, [0; 4], [0; 4])), false),
        ] {
            let packet = super::super::vp8_inter::test_segmented_interframe(
                decoder.state.as_ref().unwrap(), enabled, features, map_update);
            frames.push(decoder.decode(&packet).unwrap());
            packets.push(packet);
            let state = decoder.state.as_ref().unwrap();
            assert_eq!(state.segment_map.as_slice(), &[0, 1, 2, 3]);
            if let Some((absolute, quantizers, levels)) = features {
                assert_eq!(state.segment_absolute, absolute);
                assert_eq!(state.segment_quantizers, quantizers);
                assert_eq!(state.segment_filter_levels, levels);
            }
        }
        assert_ne!(frames[1].y.pixels, frames[2].y.pixels);
        for plane in [0, 1, 2] {
            let pixels = |index: usize| match plane {
                0 => &frames[index].y.pixels,
                1 => &frames[index].u.pixels,
                _ => &frames[index].v.pixels,
            };
            assert_eq!(pixels(2), pixels(3));
            assert_ne!(pixels(3), pixels(4));
            assert_eq!(pixels(3), pixels(5));
            assert_ne!(pixels(5), pixels(6));
        }
        let mut ivf = b"DKIF".to_vec();
        ivf.extend(0u16.to_le_bytes());
        ivf.extend(32u16.to_le_bytes());
        ivf.extend(b"VP80");
        ivf.extend(32u16.to_le_bytes());
        ivf.extend(32u16.to_le_bytes());
        ivf.extend(30u32.to_le_bytes());
        ivf.extend(1u32.to_le_bytes());
        ivf.extend((packets.len() as u32).to_le_bytes());
        ivf.extend(0u32.to_le_bytes());
        for (index, packet) in packets.iter().enumerate() {
            ivf.extend((packet.len() as u32).to_le_bytes());
            ivf.extend((index as u64).to_le_bytes());
            ivf.extend(packet);
        }
        assert_eq!(ivf.as_slice(), include_bytes!("../../tests/fixtures/vp8-segment-modes.ivf"));
        if let Ok(path) = std::env::var("WEBMEDIA_VP8_SEGMENT_TEST_IVF") {
            std::fs::write(path, ivf).unwrap();
        }
        {
            let expected = include_bytes!("../../tests/fixtures/vp8-segment-modes.yuv");
            let mut offset = 0;
            for (index, frame) in frames.iter().enumerate() {
                for (plane_index, plane) in [&frame.y, &frame.u, &frame.v].iter().enumerate() {
                    let reference = &expected[offset..offset + plane.pixels.len()];
                    if let Some(position) = plane.pixels.iter().zip(reference).position(|(a, b)| a != b) {
                        panic!("segmentation frame={index} plane={plane_index} x={} y={} actual={} expected={}",
                            position % plane.width, position / plane.width,
                            plane.pixels[position], reference[position]);
                    }
                    offset += plane.pixels.len();
                }
            }
            assert_eq!(offset, expected.len());
        }
    }

    #[test]
    fn reference_update_combinations_match_reference() {
        let mut stream = WebmVp8Stream::new();
        let input = stream.push(include_bytes!("../../tests/fixtures/vp8-motion.webm")).unwrap();
        let mut packets = Vec::new();
        let mut frames = Vec::new();
        for golden in 0..4 {
            for alternate in 0..4 {
                for refresh_last in [false, true] {
                    let mut decoder = Vp8Decoder::new();
                    for packet in &input[..2] {
                        frames.push(decoder.decode(&packet.data).unwrap());
                        packets.push(packet.data.clone());
                    }
                    let golden_before = decoder.golden.as_ref().unwrap().clone();
                    let last_before = decoder.last.as_ref().unwrap().clone();
                    assert_ne!(golden_before.y.pixels, last_before.y.pixels);
                    let flat = super::super::vp8_inter::test_interframe(
                        decoder.state.as_ref().unwrap(), None, false, 0, 3, false);
                    let flat_frame = decoder.decode(&flat).unwrap();
                    assert!(flat_frame.y.pixels.iter().all(|&pixel| pixel == 128));
                    assert_ne!(flat_frame.y.pixels, golden_before.y.pixels);
                    assert_ne!(flat_frame.y.pixels, last_before.y.pixels);
                    frames.push(flat_frame);
                    packets.push(flat);
                    let update = super::super::vp8_inter::test_interframe(
                        decoder.state.as_ref().unwrap(), None, true, golden, alternate, refresh_last);
                    let update_frame = decoder.decode(&update).unwrap();
                    assert!(update_frame.y.pixels.iter().all(|&pixel| pixel == 127));
                    assert_ne!(update_frame.y.pixels, golden_before.y.pixels);
                    assert_ne!(update_frame.y.pixels, last_before.y.pixels);
                    assert_ne!(update_frame.y.pixels, frames.last().unwrap().y.pixels);
                    frames.push(update_frame);
                    packets.push(update);
                    for reference in [2, 3, 1] {
                        let probe = super::super::vp8_inter::test_interframe(
                            decoder.state.as_ref().unwrap(), Some(reference), false, 0, 0, false);
                        frames.push(decoder.decode(&probe).unwrap());
                        packets.push(probe);
                    }
                }
            }
        }
        assert_eq!(frames.len(), 224);
        {
            let mut ivf = b"DKIF".to_vec();
            ivf.extend(0u16.to_le_bytes());
            ivf.extend(32u16.to_le_bytes());
            ivf.extend(b"VP80");
            ivf.extend(160u16.to_le_bytes());
            ivf.extend(96u16.to_le_bytes());
            ivf.extend(30u32.to_le_bytes());
            ivf.extend(1u32.to_le_bytes());
            ivf.extend((packets.len() as u32).to_le_bytes());
            ivf.extend(0u32.to_le_bytes());
            for (index, packet) in packets.iter().enumerate() {
                ivf.extend((packet.len() as u32).to_le_bytes());
                ivf.extend((index as u64).to_le_bytes());
                ivf.extend(packet);
            }
            assert_eq!(ivf.as_slice(), include_bytes!("../../tests/fixtures/vp8-reference-updates.ivf"));
            if let Ok(path) = std::env::var("WEBMEDIA_VP8_REFERENCE_TEST_IVF") {
                std::fs::write(path, ivf).unwrap();
            }
        }
        {
            let expected = include_bytes!("../../tests/fixtures/vp8-reference-updates.yuv");
            let mut offset = 0;
            for (index, frame) in frames.iter().enumerate() {
                for (plane, width, height) in [(&frame.y, 160, 96), (&frame.u, 80, 48), (&frame.v, 80, 48)] {
                    for row in 0..height {
                        assert_eq!(&plane.pixels[row * plane.width..row * plane.width + width],
                            &expected[offset..offset + width],
                            "frame={index} golden={} alternate={} refresh_last={} phase={}",
                            index / 56, (index / 14) % 4, (index / 7) % 2, index % 7);
                        offset += width;
                    }
                }
            }
            assert_eq!(offset, expected.len());
        }
    }

    #[test]
    fn unchanged_segment_map_is_shared_without_copying() {
        let mut stream = WebmVp8Stream::new();
        let packets = stream.push(include_bytes!("../../tests/fixtures/vp8-motion.webm")).unwrap();
        let mut decoder = Vp8Decoder::new();
        decoder.decode(&packets[0].data).unwrap();
        let segments = decoder.state.as_ref().unwrap().segment_map.clone();
        let mut state = decoder.state.as_ref().unwrap().clone();
        assert!(!InterFrameLayout::parse(&packets[1].data, &mut state).unwrap().segment_map_update);
        decoder.decode(&packets[1].data).unwrap();
        assert!(Arc::ptr_eq(&segments, &decoder.state.as_ref().unwrap().segment_map));
    }

    #[test]
    fn malformed_keyframe_preserves_committed_state_and_references() {
        let mut stream = WebmVp8Stream::new();
        let packets = stream.push(include_bytes!("../../tests/fixtures/vp8-motion.webm")).unwrap();
        let mut initial = Vp8Decoder::new();
        initial.decode(&packets[0].data).unwrap();
        initial.decode(&packets[1].data).unwrap();
        let mut invalid = packets[0].data[..10].to_vec();
        invalid[..3].copy_from_slice(&[0x10, 0, 0]);
        invalid[6..10].copy_from_slice(&[0xff, 0x3f, 0xff, 0x3f]);
        let mut decoder = Vp8Decoder {
            state: initial.state.clone(), last: initial.last.clone(),
            golden: initial.golden.clone(), alternate: initial.alternate.clone(),
            retired: Vec::new(),
        };
        assert!(decoder.decode(&invalid).is_err());
        assert_eq!(decoder.state, initial.state);
        for (actual, expected) in [(&decoder.last, &initial.last),
            (&decoder.golden, &initial.golden), (&decoder.alternate, &initial.alternate)]
        {
            assert!(Arc::ptr_eq(actual.as_ref().unwrap(), expected.as_ref().unwrap()));
        }
        let expected = initial.decode(&packets[2].data).unwrap();
        let actual = decoder.decode(&packets[2].data).unwrap();
        assert_eq!(actual.y.pixels, expected.y.pixels);
        assert_eq!(actual.u.pixels, expected.u.pixels);
        assert_eq!(actual.v.pixels, expected.v.pixels);
    }

    #[test]
    fn resolution_changes_reset_references_without_mutating_held_frames() {
        let mut decoder = Vp8Decoder::new();
        let mut held = Vec::new();
        for bytes in [
            include_bytes!("../../tests/fixtures/vp8-motion.webm").as_slice(),
            include_bytes!("../../tests/fixtures/vp8-odd-edges.webm").as_slice(),
            include_bytes!("../../tests/fixtures/vp8-altref.webm").as_slice(),
            include_bytes!("../../tests/fixtures/vp8-motion.webm").as_slice(),
        ] {
            let mut stream = WebmVp8Stream::new();
            let packets = stream.push(bytes).unwrap();
            stream.finish().unwrap();
            assert!(FrameHeader::parse(&packets[0].data).unwrap().key_frame);
            let mut fresh = Vp8Decoder::new();
            for (index, packet) in packets.iter().enumerate() {
                let expected = fresh.decode(&packet.data).unwrap();
                let actual = decoder.decode(&packet.data).unwrap();
                assert_eq!((actual.width, actual.height), (expected.width, expected.height));
                assert_eq!(actual.y.pixels, expected.y.pixels, "packet {index}");
                assert_eq!(actual.u.pixels, expected.u.pixels, "packet {index}");
                assert_eq!(actual.v.pixels, expected.v.pixels, "packet {index}");
                assert_eq!(decoder.state, fresh.state, "packet {index}");
                if index == 0 {
                    assert!(decoder.retired.is_empty());
                    for reference in [&decoder.last, &decoder.golden, &decoder.alternate] {
                        assert!(Arc::ptr_eq(reference.as_ref().unwrap(), &actual));
                    }
                    held.push((actual.clone(), actual.y.pixels.clone(),
                        actual.u.pixels.clone(), actual.v.pixels.clone()));
                }
            }
            for (frame, y, u, v) in &held {
                assert_eq!(&frame.y.pixels, y);
                assert_eq!(&frame.u.pixels, u);
                assert_eq!(&frame.v.pixels, v);
            }
        }
    }

    #[test]
    fn rejected_keyframe_does_not_poison_decoder_state() {
        let mut stream = WebmVp8Stream::new();
        let packets = stream.push(include_bytes!("../../tests/fixtures/vp8-motion.webm")).unwrap();
        let mut initial = Vp8Decoder::new();
        initial.decode(&packets[0].data).unwrap();
        initial.decode(&packets[1].data).unwrap();
        let mut clean = Vp8Decoder {
            state: initial.state.clone(), last: initial.last.clone(),
            golden: initial.golden.clone(), alternate: initial.alternate.clone(),
            retired: Vec::new(),
        };
        let expected = clean.decode(&packets[2].data).unwrap();
        let mut rejected = 0;
        for length in 0..packets[0].data.len() {
            let mut decoder = Vp8Decoder {
                state: initial.state.clone(), last: initial.last.clone(),
                golden: initial.golden.clone(), alternate: initial.alternate.clone(),
                retired: Vec::new(),
            };
            if decoder.decode(&packets[0].data[..length]).is_ok() { continue; }
            rejected += 1;
            assert_eq!(decoder.state, initial.state, "truncated length {length}");
            for (actual, reference) in [(&decoder.last, &initial.last),
                (&decoder.golden, &initial.golden), (&decoder.alternate, &initial.alternate)]
            {
                assert!(Arc::ptr_eq(actual.as_ref().unwrap(), reference.as_ref().unwrap()),
                    "truncated length {length}");
            }
            let recovered = decoder.decode(&packets[2].data).unwrap();
            assert_eq!(recovered.y.pixels, expected.y.pixels, "truncated length {length}");
            assert_eq!(recovered.u.pixels, expected.u.pixels, "truncated length {length}");
            assert_eq!(recovered.v.pixels, expected.v.pixels, "truncated length {length}");
        }
        assert!(rejected > 0);
    }

    #[test]
    fn rejected_interframe_does_not_poison_decoder_state() {
        let mut stream = WebmVp8Stream::new();
        let packets = stream.push(include_bytes!("../../tests/fixtures/vp8-segmentation.webm")).unwrap();
        let mut initial = Vp8Decoder::new();
        initial.decode(&packets[0].data).unwrap();
        let mut clean = Vp8Decoder::new();
        clean.decode(&packets[0].data).unwrap();
        let expected = clean.decode(&packets[1].data).unwrap();
        let mut rejected = 0;
        for length in 0..packets[1].data.len() {
            let mut decoder = Vp8Decoder {
                state: initial.state.clone(), last: initial.last.clone(),
                golden: initial.golden.clone(), alternate: initial.alternate.clone(),
                retired: Vec::new(),
            };
            if decoder.decode(&packets[1].data[..length]).is_ok() { continue; }
            rejected += 1;
            assert_eq!(decoder.state, initial.state, "truncated length {length}");
            assert!(Arc::ptr_eq(decoder.last.as_ref().unwrap(), initial.last.as_ref().unwrap()));
            assert!(Arc::ptr_eq(decoder.golden.as_ref().unwrap(), initial.golden.as_ref().unwrap()));
            assert!(Arc::ptr_eq(decoder.alternate.as_ref().unwrap(), initial.alternate.as_ref().unwrap()));
            let recovered = decoder.decode(&packets[1].data).unwrap();
            assert_eq!(recovered.y.pixels, expected.y.pixels, "truncated length {length}");
            assert_eq!(recovered.u.pixels, expected.u.pixels, "truncated length {length}");
            assert_eq!(recovered.v.pixels, expected.v.pixels, "truncated length {length}");
        }
        assert!(rejected > 0);
    }

    #[test]
    fn entropy_updates_are_frame_local_or_committed_and_rejected_frames_roll_back() {
        use super::super::vp8_inter::{test_entropy_interframe, test_interframe};
        let key = include_bytes!("../../tests/fixtures/vp8-keyframe.ivf");
        let size = u32::from_le_bytes(key[32..36].try_into().unwrap()) as usize;
        let packet = &key[44..44 + size];
        let mut initial = Vp8Decoder::new();
        let held = initial.decode(packet).unwrap();
        let held_pixels = [held.y.pixels.clone(), held.u.pixels.clone(), held.v.pixels.clone()];
        let initial_state = initial.state.as_ref().unwrap().clone();
        for refresh in [false, true] {
            let frame = test_entropy_interframe(&initial_state, refresh, true);
            let mut parsed_state = initial_state.clone();
            let mut layout = InterFrameLayout::parse(&frame, &mut parsed_state).unwrap();
            assert_ne!(layout.coeff_probs, initial_state.coeff_probs);
            assert_ne!(layout.mv_probs, initial_state.mv_probs);
            assert_ne!(layout.ymode_probs, initial_state.ymode_probs);
            assert_ne!(layout.uv_mode_probs, initial_state.uv_mode_probs);
            assert_eq!(layout.mv_probs[0][0], 1);
            assert_eq!(layout.ymode_probs, [11, 22, 33, 44]);
            assert_eq!(layout.uv_mode_probs, [55, 66, 77]);
            let modes = layout.read_macroblocks(&mut parsed_state,
                initial_state.mb_width, initial_state.mb_height).unwrap();
            assert!(modes.iter().all(|mode| mode.reference == 0 && mode.luma == 0
                && mode.chroma == 0 && mode.skip_coefficients));
            if refresh {
                assert_eq!(parsed_state.coeff_probs, layout.coeff_probs);
                assert_eq!(parsed_state.mv_probs, layout.mv_probs);
                assert_eq!(parsed_state.ymode_probs, layout.ymode_probs);
                assert_eq!(parsed_state.uv_mode_probs, layout.uv_mode_probs);
            } else { assert_eq!(parsed_state, initial_state); }
            let fresh = || Vp8Decoder {
                state: Some(initial_state.clone()), last: initial.last.clone(),
                golden: initial.golden.clone(), alternate: initial.alternate.clone(), retired: Vec::new(),
            };
            let mut clean = fresh();
            let expected = clean.decode(&frame).unwrap();
            assert_eq!(clean.state.as_ref().unwrap(), &parsed_state);
            assert!(Arc::ptr_eq(clean.last.as_ref().unwrap(), &expected));
            let following = test_interframe(clean.state.as_ref().unwrap(), None, true, 0, 0, true);
            let expected_following = clean.decode(&following).unwrap();
            let mut rejected = 0;
            let mut rejected_after_header = 0;
            for cut in 0..frame.len() {
                let mut decoder = fresh();
                if decoder.decode(&frame[..cut]).is_ok() { continue; }
                rejected += 1;
                let mut scratch = initial_state.clone();
                if InterFrameLayout::parse(&frame[..cut], &mut scratch).is_ok() {
                    rejected_after_header += 1;
                }
                assert_eq!(decoder.state.as_ref().unwrap(), &initial_state);
                for (actual, prior) in [(&decoder.last, &initial.last),
                    (&decoder.golden, &initial.golden), (&decoder.alternate, &initial.alternate)] {
                    assert!(Arc::ptr_eq(actual.as_ref().unwrap(), prior.as_ref().unwrap()));
                }
                let recovered = decoder.decode(&frame).unwrap();
                assert_eq!(decoder.state.as_ref().unwrap(), &parsed_state);
                let next = decoder.decode(&following).unwrap();
                for (a, b) in [(&recovered, &expected), (&next, &expected_following)] {
                    assert_eq!(a.y.pixels, b.y.pixels);
                    assert_eq!(a.u.pixels, b.u.pixels);
                    assert_eq!(a.v.pixels, b.v.pixels);
                }
            }
            assert!(rejected > 0);
            // All-skipped frames need no token bytes, so truncation need not fail after
            // a complete header. Force a residual-stage failure with explicit non-skip modes.
            let mut malformed = test_entropy_interframe(&initial_state, refresh, false);
            malformed.truncate(malformed.len() - 2);
            let mut scratch = initial_state.clone();
            let mut parsed = InterFrameLayout::parse(&malformed, &mut scratch).unwrap();
            let modes = parsed.read_macroblocks(&mut scratch,
                initial_state.mb_width, initial_state.mb_height).unwrap();
            assert!(modes.iter().all(|mode| !mode.skip_coefficients));
            assert_ne!(parsed.coeff_probs, initial_state.coeff_probs);
            let mut decoder = fresh();
            assert!(decoder.decode(&malformed).is_err());
            assert_eq!(decoder.state.as_ref().unwrap(), &initial_state);
            for (actual, prior) in [(&decoder.last, &initial.last),
                (&decoder.golden, &initial.golden), (&decoder.alternate, &initial.alternate)] {
                assert!(Arc::ptr_eq(actual.as_ref().unwrap(), prior.as_ref().unwrap()));
            }
            let recovered = decoder.decode(&frame).unwrap();
            assert_eq!(decoder.state.as_ref().unwrap(), &parsed_state);
            let next = decoder.decode(&following).unwrap();
            for (a, b) in [(&recovered, &expected), (&next, &expected_following)] {
                assert_eq!(a.y.pixels, b.y.pixels);
                assert_eq!(a.u.pixels, b.u.pixels);
                assert_eq!(a.v.pixels, b.v.pixels);
            }
            assert_eq!(held.y.pixels, held_pixels[0]);
            assert_eq!(held.u.pixels, held_pixels[1]);
            assert_eq!(held.v.pixels, held_pixels[2]);
            eprintln!("entropy refresh={refresh}: {rejected} rejected cuts, {rejected_after_header} after complete header");
        }
    }

    #[test]
    fn reserved_versions_are_rejected_without_updating_references() {
        let bytes = include_bytes!("../../tests/fixtures/vp8-motion.webm");
        let mut stream = WebmVp8Stream::new();
        let packets = stream.push(bytes).unwrap();
        assert!(FrameHeader::parse(&packets[0].data).unwrap().key_frame);
        assert!(!FrameHeader::parse(&packets[1].data).unwrap().key_frame);
        for version in 4..8 {
            let mut decoder = Vp8Decoder::new();
            let mut key = packets[0].data.clone();
            key[0] = (key[0] & !14) | (version << 1);
            assert!(matches!(decoder.decode(&key), Err(MediaDecodeError::Unsupported)));
            assert!(decoder.state.is_none());
            assert!(matches!(decode_keyframe_with_state(&key), Err(MediaDecodeError::Unsupported)));
            let reference = decoder.decode(&packets[0].data).unwrap();
            let mut inter = packets[1].data.clone();
            inter[0] = (inter[0] & !14) | (version << 1);
            assert!(matches!(decoder.decode(&inter), Err(MediaDecodeError::Unsupported)));
            assert!(Arc::ptr_eq(&reference, decoder.last.as_ref().unwrap()));
            decoder.decode(&packets[1].data).unwrap();
        }
    }

    #[test]
    fn reconstructs_supplied_webm_prefix() {
        let Ok(path) = std::env::var("WEBMEDIA_WEBM_SAMPLE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let reference = std::env::var("WEBMEDIA_VP8_SEQUENCE_REFERENCE")
            .ok()
            .map(|path| std::fs::read(path).unwrap());
        let limit = std::env::var("WEBMEDIA_VP8_FRAME_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(12);
        let max_mae = std::env::var("WEBMEDIA_VP8_MAX_MAE")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(1.0);
        let report = std::env::var_os("WEBMEDIA_VP8_REPORT").is_some();
        let reference_frames = std::env::var("WEBMEDIA_VP8_REFERENCE_FRAMES").ok()
            .map(|value| value.split(',').map(|index| index.parse::<usize>().unwrap()).collect::<Vec<_>>());
        verify_sequence(&bytes, reference.as_deref(), limit, max_mae, report, reference_frames.as_deref());
    }

    #[test]
    fn motion_sequence_matches_reference_pixel_exactly() {
        verify_sequence(include_bytes!("../../tests/fixtures/vp8-motion.webm"),
            Some(include_bytes!("../../tests/fixtures/vp8-motion.yuv")), 60, 0.0, false, None);
    }

    #[test]
    fn padded_reference_edges_match_pixel_exactly() {
        verify_sequence(include_bytes!("../../tests/fixtures/vp8-edges.webm"),
            Some(include_bytes!("../../tests/fixtures/vp8-edges.yuv")), 30, 0.0, false, None);
    }

    #[test]
    fn hidden_references_and_four_token_partitions_match_pixel_exactly() {
        let bytes = include_bytes!("../../tests/fixtures/vp8-altref.webm");
        let mut stream = WebmVp8Stream::new();
        let mut decoder = Vp8Decoder::new();
        let mut hidden = 0;
        let mut shown = 0;
        for packet in stream.push(bytes).unwrap() {
            let header = FrameHeader::parse(&packet.data).unwrap();
            let partitions = if header.key_frame {
                KeyFrameLayout::parse(&packet.data).unwrap().token_partitions.len()
            } else {
                let mut state = decoder.state.as_ref().unwrap().clone();
                InterFrameLayout::parse(&packet.data, &mut state).unwrap().token_partitions.len()
            };
            assert_eq!(partitions, 4);
            decoder.decode(&packet.data).unwrap();
            if header.show_frame { shown += 1; } else { hidden += 1; }
        }
        stream.finish().unwrap();
        assert_eq!(shown, 120);
        assert_eq!(hidden, 6);
        verify_sequence(bytes, Some(include_bytes!("../../tests/fixtures/vp8-altref.yuv")),
            shown, 0.0, false, None);
    }

    #[test]
    fn low_complexity_versions_match_pixel_exactly() {
        for (version, bytes, reference) in [
            (1, include_bytes!("../../tests/fixtures/vp8-version1.webm").as_slice(),
                include_bytes!("../../tests/fixtures/vp8-version1.yuv").as_slice()),
            (2, include_bytes!("../../tests/fixtures/vp8-version2.webm").as_slice(),
                include_bytes!("../../tests/fixtures/vp8-version2.yuv").as_slice()),
            (3, include_bytes!("../../tests/fixtures/vp8-version3.webm").as_slice(),
                include_bytes!("../../tests/fixtures/vp8-version3.yuv").as_slice()),
        ] {
            let mut stream = WebmVp8Stream::new();
            let packets = stream.push(bytes).unwrap();
            assert_eq!(packets.len(), 30);
            for packet in packets {
                assert_eq!(FrameHeader::parse(&packet.data).unwrap().version, version);
            }
            verify_sequence(bytes, Some(reference), 30, 0.0, false, None);
        }
    }

    #[test]
    fn active_segmentation_matches_pixel_exactly() {
        let bytes = include_bytes!("../../tests/fixtures/vp8-segmentation.webm");
        let mut stream = WebmVp8Stream::new();
        let packets = stream.push(bytes).unwrap();
        assert_eq!(packets.len(), 8);
        let mut decoder = Vp8Decoder::new();
        let mut active_features = false;
        let mut multiple_segments = false;
        for packet in packets {
            let header = FrameHeader::parse(&packet.data).unwrap();
            let partitions = if header.key_frame {
                KeyFrameLayout::parse(&packet.data).unwrap().token_partitions.len()
            } else {
                let mut state = decoder.state.as_ref().unwrap().clone();
                InterFrameLayout::parse(&packet.data, &mut state).unwrap().token_partitions.len()
            };
            assert_eq!(partitions, 2);
            decoder.decode(&packet.data).unwrap();
            let state = decoder.state.as_ref().unwrap();
            active_features |= state.segment_quantizers.iter().any(|&value| value != 0);
            multiple_segments |= state.segment_map.iter().any(|&value| value != state.segment_map[0]);
        }
        assert!(active_features, "fixture must use nonzero segment quantization");
        assert!(multiple_segments, "fixture must assign different macroblock segments");
        stream.finish().unwrap();
        verify_sequence(bytes, Some(include_bytes!("../../tests/fixtures/vp8-segmentation.yuv")),
            8, 0.0, false, None);
    }

    #[test]
    fn odd_dimensions_with_eight_token_partitions_match_pixel_exactly() {
        let bytes = include_bytes!("../../tests/fixtures/vp8-odd-edges.webm");
        let mut stream = WebmVp8Stream::new();
        let mut decoder = Vp8Decoder::new();
        let packets = stream.push(bytes).unwrap();
        assert_eq!(packets.len(), 20);
        for packet in packets {
            let header = FrameHeader::parse(&packet.data).unwrap();
            let partitions = if header.key_frame {
                assert_eq!((header.width, header.height), (Some(163), Some(91)));
                KeyFrameLayout::parse(&packet.data).unwrap().token_partitions.len()
            } else {
                let mut state = decoder.state.as_ref().unwrap().clone();
                let layout = InterFrameLayout::parse(&packet.data, &mut state).unwrap();
                assert_eq!(layout.sharpness, 7);
                layout.token_partitions.len()
            };
            assert_eq!(partitions, 8);
            decoder.decode(&packet.data).unwrap();
        }
        verify_sequence(bytes, Some(include_bytes!("../../tests/fixtures/vp8-odd-edges.yuv")),
            20, 0.0, false, None);
    }

    fn verify_sequence(bytes: &[u8], reference: Option<&[u8]>, limit: usize, max_mae: f64,
        report: bool, reference_frames: Option<&[usize]>)
    {
        let mut demuxer = WebmVp8Stream::new();
        let mut decoder = Vp8Decoder::new();
        let mut count = 0;
        for chunk in bytes.chunks(16384) {
            for packet in demuxer.push(chunk).unwrap() {
                let frame = decoder
                    .decode(&packet.data)
                    .unwrap_or_else(|error| panic!("VP8 frame {count}: {error:?}"));
                let header = FrameHeader::parse(&packet.data).unwrap();
                if header.key_frame {
                    assert!(Arc::ptr_eq(&frame, decoder.last.as_ref().unwrap()));
                    assert!(Arc::ptr_eq(&frame, decoder.golden.as_ref().unwrap()));
                    assert!(Arc::ptr_eq(&frame, decoder.alternate.as_ref().unwrap()));
                }
                if !header.show_frame { continue; }
                assert_eq!(
                    frame.width as u32,
                    demuxer.metadata().unwrap().width.unwrap()
                );
                let reference_index = reference_frames.map_or(Some(count),
                    |frames| frames.iter().position(|&index| index == count));
                if let (Some(reference), Some(reference_index)) = (reference, reference_index) {
                    let luma = frame.width * frame.height;
                    let chroma = frame.width.div_ceil(2) * frame.height.div_ceil(2);
                    let frame_size = luma + 2 * chroma;
                    assert!(reference.len() >= (reference_index + 1) * frame_size);
                    let expected = &reference[reference_index * frame_size..(reference_index + 1) * frame_size];
                    let error = |plane: &super::super::vp8_predict::Plane,
                                 offset: usize,
                                 width: usize,
                                 height: usize| {
                        let mut differences = Vec::new();
                        let total = (0..height)
                            .flat_map(|row| (0..width).map(move |col| (row, col)))
                            .map(|(row, col)| {
                                let actual = plane.pixels[row * plane.width + col];
                                let wanted = expected[offset + row * width + col];
                                if report && actual != wanted && differences.len() < 12 {
                                    differences.push((col, row, actual, wanted));
                                }
                                usize::from(actual.abs_diff(wanted))
                            })
                            .sum::<usize>();
                        if report && total != 0 { eprintln!("VP8 differing pixels: {differences:?}, total error={total}"); }
                        total as f64 / (width * height) as f64
                    };
                    for (name, plane, offset, width, height) in [
                        ("Y", &frame.y, 0, frame.width, frame.height),
                        (
                            "U",
                            &frame.u,
                            luma,
                            frame.width.div_ceil(2),
                            frame.height.div_ceil(2),
                        ),
                        (
                            "V",
                            &frame.v,
                            luma + chroma,
                            frame.width.div_ceil(2),
                            frame.height.div_ceil(2),
                        ),
                    ] {
                        let mae = error(plane, offset, width, height);
                        if report {
                            eprintln!("VP8 frame {count} {name} MAE {mae:.3}");
                        }
                        assert!(if max_mae == 0.0 { mae == 0.0 } else { mae < max_mae },
                            "VP8 frame {count} {name} MAE {mae:.6}");
                    }
                }
                count += 1;
                if count == limit {
                    break;
                }
            }
            if count == limit {
                break;
            }
        }
        assert_eq!(count, limit);
    }
}
