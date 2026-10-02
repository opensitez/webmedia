//! VP8 reference-frame reconstruction across keyframes and interframes.

use super::backend::MediaDecodeError;
use super::vp8::{FrameHeader, KeyFrameLayout};
use super::vp8_filter::{filter_frame, FilterMacroblock};
use super::vp8_inter::{InterFrameLayout, InterMacroblock, InterState};
use super::vp8_keyframe::{decode_keyframe, reconstruct_intra, YuvKeyFrame};
use super::vp8_motion::{chroma_vector, predict_block};
use super::vp8_residue::{ResidualMacroblock, ResidueDecoder};
use std::sync::Arc;

pub(super) struct Vp8Decoder {
    state: Option<InterState>,
    last: Option<Arc<YuvKeyFrame>>,
    golden: Option<Arc<YuvKeyFrame>>,
    alternate: Option<Arc<YuvKeyFrame>>,
}

impl Vp8Decoder {
    pub(super) fn new() -> Self {
        Self {
            state: None,
            last: None,
            golden: None,
            alternate: None,
        }
    }

    pub(super) fn decode(&mut self, packet: &[u8]) -> Result<YuvKeyFrame, MediaDecodeError> {
        let header = FrameHeader::parse(packet)?;
        if header.key_frame {
            let frame = decode_keyframe(packet)?;
            let mut layout = KeyFrameLayout::parse(packet)?;
            let mut state = InterState::after_keyframe(&layout);
            for segment in &mut state.segment_map {
                *segment = layout
                    .next_macroblock_mode()?
                    .ok_or_else(|| {
                        MediaDecodeError::InvalidData("truncated VP8 segment map".into())
                    })?
                    .segment;
            }
            self.state = Some(state);
            let reference = Arc::new(frame.clone());
            self.last = Some(reference.clone());
            self.golden = Some(reference.clone());
            self.alternate = Some(reference);
            return Ok(frame);
        }
        let state = self.state.as_mut().ok_or_else(|| {
            MediaDecodeError::InvalidData("VP8 interframe before keyframe".into())
        })?;
        let mut layout = InterFrameLayout::parse(packet, state)?;
        let mb_width = state.mb_width;
        let mb_height = state.mb_height;
        let modes = layout.read_macroblocks(state, mb_width, mb_height)?;
        let mut residues = ResidueDecoder::new(&layout)?;
        let last = self.last.as_ref().unwrap();
        let mut frame = YuvKeyFrame::new(last.width, last.height);
        let mut filter_settings = Vec::with_capacity(modes.len());
        for mb_y in 0..mb_height {
            for mb_x in 0..mb_width {
                let index = mb_y * mb_width + mb_x;
                let mode = &modes[index];
                let blocks = residues.decode(&layout, mode, mb_x, mb_y)?;
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
                filter_settings.push(FilterMacroblock {
                    level: layout.macroblock_filter_level(mode),
                    skip_inner: mode.mode != 4 && mode.mode != 9 && mode.skip_coefficients,
                });
            }
        }
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
        let refreshed = Arc::new(frame.clone());
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
                1 => old_last,
                2 => old_golden,
                _ => old_alternate,
            };
        }
        if layout.refresh_last {
            self.last = Some(refreshed);
        }
        Ok(frame)
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
    for row in 0..4 {
        for col in 0..4 {
            let index = row * 4 + col;
            let x = mb_x * 16 + col * 4;
            let y = mb_y * 16 + row * 4;
            predict_block(
                &mut frame.y,
                &reference.y,
                x,
                y,
                4,
                4,
                mode.motion[index],
                false,
                version != 0,
                reference.width,
                reference.height,
            );
            frame.y.add_residual(x, y, &blocks.y[index]);
        }
    }
    for (plane, reference_plane, residuals) in [
        (&mut frame.u, &reference.u, &blocks.u),
        (&mut frame.v, &reference.v, &blocks.v),
    ] {
        for row in 0..2 {
            for col in 0..2 {
                let index = row * 2 + col;
                let x = mb_x * 8 + col * 4;
                let y = mb_y * 8 + row * 4;
                predict_block(
                    plane,
                    reference_plane,
                    x,
                    y,
                    4,
                    4,
                    chroma_vector(&mode.motion, row, col),
                    true,
                    version != 0,
                    reference.width.div_ceil(2),
                    reference.height.div_ceil(2),
                );
                plane.add_residual(x, y, &residuals[index]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::webm::WebmVp8Stream;

    #[test]
    fn reconstructs_supplied_webm_prefix() {
        let Ok(path) = std::env::var("WEBMEDIA_WEBM_SAMPLE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let mut demuxer = WebmVp8Stream::new();
        let mut decoder = Vp8Decoder::new();
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
        let mut count = 0;
        for chunk in bytes.chunks(16384) {
            for packet in demuxer.push(chunk).unwrap() {
                let frame = decoder
                    .decode(&packet.data)
                    .unwrap_or_else(|error| panic!("VP8 frame {count}: {error:?}"));
                assert_eq!(
                    frame.width as u32,
                    demuxer.metadata().unwrap().width.unwrap()
                );
                if let Some(reference) = &reference {
                    let luma = frame.width * frame.height;
                    let chroma = frame.width.div_ceil(2) * frame.height.div_ceil(2);
                    let frame_size = luma + 2 * chroma;
                    assert!(reference.len() >= (count + 1) * frame_size);
                    let expected = &reference[count * frame_size..(count + 1) * frame_size];
                    let error = |plane: &super::super::vp8_predict::Plane,
                                 offset: usize,
                                 width: usize,
                                 height: usize| {
                        (0..height)
                            .flat_map(|row| (0..width).map(move |col| (row, col)))
                            .map(|(row, col)| {
                                usize::from(
                                    plane.pixels[row * plane.width + col]
                                        .abs_diff(expected[offset + row * width + col]),
                                )
                            })
                            .sum::<usize>() as f64
                            / (width * height) as f64
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
                        assert!(mae < max_mae, "VP8 frame {count} {name} MAE {mae:.3}");
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
        assert!(count > 1);
    }
}
