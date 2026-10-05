//! Stateful AV1 compressed-to-pixel decoding with explicit unsupported tools.

use std::sync::Arc;

use super::intra::{FrameCdfs, IntraDecodeProgress, IntraTile};
use super::syntax::{Bits, Error, IntraFrameHeader, SequenceHeader};
use super::{CodedIntraFrame, Obu};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedPlane {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub samples: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedIntraFrame {
    pub header: IntraFrameHeader,
    pub bit_depth: u8,
    /// Original parsed color, range, chroma position and subsampling metadata.
    pub sequence: SequenceHeader,
    /// Y, U, V in sequence precision, or only Y for monochrome.
    pub planes: Vec<DecodedPlane>,
}

/// Displayable decoded pixels and complete sequence color metadata.
pub type DecodedFrame = DecodedIntraFrame;

#[derive(Clone)]
struct ReferenceFrame {
    frame: Arc<DecodedFrame>,
    cdfs: FrameCdfs,
    motion: super::temporal::SavedMotion,
}

/// Persistent sequence, reference-slot and entropy state. Failed frames are not committed.
#[derive(Default, Clone)]
pub struct Av1Decoder {
    sequence: Option<SequenceHeader>,
    references: [Option<Arc<ReferenceFrame>>; 8],
}

impl Av1Decoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn sequence(&self) -> Option<&SequenceHeader> {
        self.sequence.as_ref()
    }

    /// Feed one complete low-overhead OBU. Hidden frames update state but return no output.
    pub fn decode_obu(&mut self, obu: &Obu) -> Result<Option<DecodedFrame>, Error> {
        #[cfg(test)]
        let old_lifecycle = LIFECYCLE_REFERENCE.with(|flag| flag.get());
        #[cfg(not(test))]
        let old_lifecycle = false;
        if obu.kind == 1 {
            let sequence = SequenceHeader::parse(&obu.payload)?;
            if self.sequence.as_ref() != Some(&sequence) {
                self.references = Default::default();
            }
            self.sequence = Some(sequence);
            return Ok(None);
        }
        if matches!(obu.kind, 2 | 5 | 15) {
            return Ok(None);
        }
        if !matches!(obu.kind, 3 | 6) {
            return Err(Error::Unsupported("AV1 OBU type"));
        }
        let sequence = self
            .sequence
            .as_ref()
            .ok_or(Error::Invalid("frame before sequence header"))?;
        if !sequence.reduced_still_picture_header {
            let mut bits = Bits::new(&obu.payload);
            if bits.flag()? {
                let slot = bits.read(3)? as usize;
                if sequence.frame_presentation_time_bits != 0 && !sequence.equal_picture_interval {
                    bits.read(sequence.frame_presentation_time_bits)?;
                }
                let display_id = bits.read(sequence.frame_id_bits)?;
                if obu.kind == 3 {
                    bits.trailing()?;
                } else {
                    bits.align()?;
                }
                if bits.position != obu.payload.len() * 8 {
                    return Err(Error::Invalid("show-existing trailing bytes"));
                }
                let reference = self.references[slot]
                    .as_ref()
                    .ok_or(Error::Invalid("missing show-existing slot"))?
                    .clone();
                if !reference.frame.header.showable_frame {
                    return Err(Error::Invalid("reference is not showable"));
                }
                if sequence.frame_id_bits != 0
                    && reference.frame.header.current_frame_id != display_id
                {
                    return Err(Error::Invalid("show-existing frame id"));
                }
                let mut output = (*reference.frame).clone();
                output.header.show_frame = true;
                if output.header.frame_type == 0 {
                    output.header.refresh_frame_flags = 255;
                    output.header.showable_frame = false;
                    let replacement = Arc::new(ReferenceFrame {
                        frame: Arc::new(output.clone()),
                        cdfs: reference.cdfs.clone(),
                        motion: reference.motion.clone(),
                    });
                    self.references.fill(Some(replacement));
                } else {
                    output.header.refresh_frame_flags = 0;
                }
                return Ok(Some(output));
            }
        }
        if obu.kind != 6 {
            return Err(Error::Unsupported("separate frame header and tile group"));
        }
        let refs = self
            .references
            .each_ref()
            .map(|r| r.as_ref().map(|r| &r.frame.header));
        let header = IntraFrameHeader::parse_with_refs(
            &obu.payload,
            sequence,
            obu.temporal_id,
            obu.spatial_id,
            &refs,
        )?;
        let tiles = super::syntax::tile_group(&obu.payload[header.header_bytes..], &header.tiles)?;
        if tiles.len() != header.tiles.count() {
            return Err(Error::Invalid("combined frame must contain all tiles"));
        }
        let mut tile = IntraTile::new(sequence, &header, tiles[0].1)?;
        if header.primary_ref_frame != 7 {
            let slot = header.ref_frame_idx[header.primary_ref_frame as usize] as usize;
            let previous = self.references[slot]
                .as_ref()
                .ok_or(Error::Invalid("missing primary CDF reference"))?;
            tile.load_cdfs(&previous.cdfs);
        }
        let pixel_refs = header.ref_frame_idx.map(|slot| {
            self.references[slot as usize]
                .as_ref()
                .map(|r| r.frame.as_ref())
        });
        tile.set_references(pixel_refs);
        if header.use_ref_frame_mvs {
            tile.set_motion_field(super::temporal::MotionField::new(
                sequence,
                &header,
                header
                    .ref_frame_idx
                    .map(|i| self.references[i as usize].as_ref().map(|r| &r.motion)),
            ));
        }
        let store_reference = header.refresh_frame_flags != 0;
        let initial = (old_lifecycle || store_reference && header.disable_frame_end_update_cdf)
            .then(|| tile.save_cdfs());
        #[cfg(not(test))]
        tile.run()?;
        #[cfg(test)]
        if super::profile::enabled() {
            let mut last = std::time::Instant::now();
            let mut stage = 0;
            tile.run_with_observer(|_, _| {
                let now = std::time::Instant::now();
                super::profile::record_stage(stage, now.duration_since(last).as_nanos());
                last = now;
                stage += 1;
            })?;
        } else {
            tile.run()?;
        }
        let saved = if old_lifecycle || store_reference {
            let mut cdfs = if header.disable_frame_end_update_cdf {
                initial.expect("initial reference CDFs")
            } else {
                tile.save_cdfs()
            };
            cdfs.reset_counts();
            Some((cdfs, tile.save_motion()))
        } else {
            None
        };
        let output = DecodedFrame {
            bit_depth: sequence.bit_depth,
            sequence: sequence.clone(),
            planes: tile.finish_planes(),
            header,
        };
        let refresh = output.header.refresh_frame_flags;
        let hints = output.header.ref_order_hints;
        let show = output.header.show_frame;
        let (output, reference) = if let Some((cdfs, motion)) = saved {
            if old_lifecycle {
                let reference = Arc::new(ReferenceFrame {
                    frame: Arc::new(output.clone()),
                    cdfs,
                    motion,
                });
                (show.then_some(output), Some(reference))
            } else {
                // Hidden references own the finished planes directly. Only a
                // displayed, refreshing frame needs a second owned plane set.
                let frame = Arc::new(output);
                let displayed = show.then(|| (*frame).clone());
                (
                    displayed,
                    Some(Arc::new(ReferenceFrame {
                        frame,
                        cdfs,
                        motion,
                    })),
                )
            }
        } else {
            (show.then_some(output), None)
        };
        for (i, slot) in self.references.iter_mut().enumerate() {
            if refresh & (1 << i) != 0 {
                *slot = Some(reference.as_ref().expect("refreshed reference").clone());
            } else if hints[i].is_some_and(|hint| {
                slot.as_ref()
                    .is_some_and(|r| r.frame.header.order_hint != hint)
            }) {
                *slot = None;
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
std::thread_local! {
    static LIFECYCLE_REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
pub(crate) fn lifecycle_reference() -> bool {
    LIFECYCLE_REFERENCE.with(|flag| flag.get())
}

/// Decode a combined OBU_FRAME without using any external decoder.
/// Currently handles bounded intra modes and DCT/ADST/identity coefficients; unsupported
/// content never returns placeholder pixels or a partial successful frame.
pub fn decode_intra_frame(
    obu: &Obu,
    sequence: &SequenceHeader,
) -> Result<DecodedIntraFrame, Error> {
    let coded = CodedIntraFrame::parse(obu, sequence)?;
    let mut tile = IntraTile::new(sequence, &coded.header, coded.tiles[0].1)?;
    tile.run()?;
    let planes = tile.finish_planes();
    Ok(DecodedIntraFrame {
        header: coded.header,
        bit_depth: sequence.bit_depth,
        sequence: sequence.clone(),
        planes,
    })
}

/// Diagnostic prefix status only. A stopped prefix must never be displayed as
/// a decoded frame; `decode_intra_frame` retains all-or-error semantics.
pub fn inspect_intra_decode(
    obu: &Obu,
    sequence: &SequenceHeader,
) -> Result<IntraDecodeProgress, Error> {
    let coded = CodedIntraFrame::parse(obu, sequence)?;
    let mut tile = IntraTile::new(sequence, &coded.header, coded.tiles[0].1)?;
    tile.progress.stopped = tile.run().err();
    Ok(tile.progress)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "later-clip phase distributions; requires AV1_STREAM_OBU"]
    fn spacewalk_later_frame_profile() {
        use std::time::Instant;
        let first = std::env::var("AV1_BENCH_START_DISPLAY")
            .ok()
            .map_or(256, |v| v.parse::<usize>().unwrap());
        let count = std::env::var("AV1_PROFILE_COUNT")
            .ok()
            .map_or(64, |v| v.parse::<usize>().unwrap());
        let bytes = std::fs::read(std::env::var("AV1_STREAM_OBU").unwrap()).unwrap();
        let mut stream = super::super::ObuStream::new();
        let mut decoder = Av1Decoder::new();
        let mut displayed = 0;
        let mut records = Vec::new();
        for obu in stream.push(&bytes).unwrap() {
            let coded = obu.kind == 6 && obu.payload.first().is_some_and(|b| b & 128 == 0);
            let sampling = coded && displayed >= first;
            let header = if sampling {
                let refs = decoder
                    .references
                    .each_ref()
                    .map(|r| r.as_ref().map(|r| &r.frame.header));
                Some(
                    IntraFrameHeader::parse_with_refs(
                        &obu.payload,
                        decoder.sequence.as_ref().unwrap(),
                        obu.temporal_id,
                        obu.spatial_id,
                        &refs,
                    )
                    .unwrap(),
                )
            } else {
                None
            };
            super::super::profile::reset(sampling);
            let start = Instant::now();
            let output = decoder.decode_obu(&obu).unwrap();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            if let Some(header) = header {
                let stages = super::super::profile::stages().map(|v| v as f64 / 1_000_000.0);
                let details = super::super::profile::times().map(|v| v as f64 / 1_000_000.0);
                records.push((
                    displayed,
                    header.order_hint,
                    header.show_frame,
                    elapsed,
                    stages,
                    details,
                ));
            }
            if output.is_some() {
                displayed += 1;
            }
            std::hint::black_box(&output);
            if records.len() == count {
                break;
            }
        }
        super::super::profile::reset(false);
        assert_eq!(records.len(), count);
        let percentile = |mut values: Vec<f64>, percent: usize| {
            values.sort_by(f64::total_cmp);
            values[((values.len() - 1) * percent / 100).min(values.len() - 1)]
        };
        for show in [false, true] {
            let group: Vec<_> = records.iter().filter(|r| r.2 == show).collect();
            if group.is_empty() {
                continue;
            }
            let times: Vec<_> = group.iter().map(|r| r.3).collect();
            let average = times.iter().sum::<f64>() / times.len() as f64;
            let stages = std::array::from_fn::<_, 4, _>(|i| {
                group.iter().map(|r| r.4[i]).sum::<f64>() / group.len() as f64
            });
            let details = std::array::from_fn::<_, 5, _>(|i| {
                group.iter().map(|r| r.5[i]).sum::<f64>() / group.len() as f64
            });
            eprintln!(
                "LATER first={first} show={show} count={} avg_ms={average:.3} p50_ms={:.3} p90_ms={:.3} max_ms={:.3} reconstruct={:.3} deblock={:.3} cdef={:.3} restoration={:.3} motion={:.3} coefficients={:.3} transform={:.3} intra={:.3} warp={:.3}",
                group.len(),
                percentile(times.clone(), 50),
                percentile(times.clone(), 90),
                percentile(times, 100),
                stages[0],
                stages[1],
                stages[2],
                stages[3],
                details[0],
                details[1],
                details[2],
                details[3],
                details[4]
            );
        }
        records.sort_by(|a, b| b.3.total_cmp(&a.3));
        for r in records.iter().take(12) {
            eprintln!(
                "SPIKE display={} hint={} show={} total={:.3} stages={:?} details={:?}",
                r.0, r.1, r.2, r.3, r.4, r.5
            );
        }
    }

    #[test]
    #[ignore = "same-state lifecycle A/B benchmark; requires AV1_STREAM_OBU"]
    fn spacewalk_lifecycle_same_frame_benchmark() {
        use std::time::Instant;
        let repeats = std::env::var("AV1_BENCH_REPEATS")
            .ok()
            .map_or(12, |v| v.parse::<usize>().unwrap());
        assert!(repeats >= 3);
        let first_display = std::env::var("AV1_BENCH_START_DISPLAY")
            .ok()
            .map_or(0, |v| v.parse::<usize>().unwrap());
        let bytes = std::fs::read(std::env::var("AV1_STREAM_OBU").unwrap()).unwrap();
        let mut stream = super::super::ObuStream::new();
        let mut decoder = Av1Decoder::new();
        let mut targets = vec![0, 16, 8, 1];
        let mut displayed = 0;
        for obu in stream.push(&bytes).unwrap() {
            if obu.kind == 6
                && obu.payload.first().is_some_and(|b| b & 128 == 0)
                && displayed >= first_display
            {
                let sequence = decoder.sequence.as_ref().unwrap();
                let refs = decoder
                    .references
                    .each_ref()
                    .map(|r| r.as_ref().map(|r| &r.frame.header));
                let h = IntraFrameHeader::parse_with_refs(
                    &obu.payload,
                    sequence,
                    obu.temporal_id,
                    obu.spatial_id,
                    &refs,
                )
                .unwrap();
                let target = if first_display != 0 {
                    Some(0)
                } else {
                    targets.iter().position(|&v| v == h.order_hint)
                };
                if let Some(target) = target {
                    let mut times: [Vec<f64>; 2] = Default::default();
                    let mut expected: Option<(Option<DecodedFrame>, Av1Decoder)> = None;
                    for repeat in 0..repeats * 2 + 4 {
                        let reference = [true, false, false, true][repeat % 4];
                        let mut trial = decoder.clone();
                        LIFECYCLE_REFERENCE.with(|flag| flag.set(reference));
                        let start = Instant::now();
                        let result = trial.decode_obu(&obu).unwrap();
                        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                        LIFECYCLE_REFERENCE.with(|flag| flag.set(false));
                        if let Some((output, saved)) = &expected {
                            assert_eq!(&result, output, "lifecycle changed output");
                            for (i, (actual, expected)) in
                                trial.references.iter().zip(&saved.references).enumerate()
                            {
                                match (actual, expected) {
                                    (Some(a), Some(b)) => {
                                        if !Arc::ptr_eq(a, b)
                                            && !saved.references[..i].iter().any(|r| {
                                                r.as_ref().is_some_and(|r| Arc::ptr_eq(r, b))
                                            })
                                        {
                                            assert_eq!(
                                                a.frame, b.frame,
                                                "lifecycle changed reference slot {i}"
                                            );
                                        }
                                    }
                                    (None, None) => {}
                                    _ => panic!("lifecycle changed slot presence {i}"),
                                }
                            }
                        } else {
                            expected = Some((result.clone(), trial.clone()));
                        }
                        if repeat >= 4 {
                            times[usize::from(reference)].push(elapsed);
                        }
                        std::hint::black_box(&result);
                    }
                    let medians = times.map(|mut values| {
                        values.sort_by(f64::total_cmp);
                        values[values.len() / 2]
                    });
                    eprintln!(
                        "LIFECYCLE display={displayed} hint={} show={} refresh={} repeats={repeats} optimized_ms={:.3} original_ms={:.3} speedup={:.3}",
                        h.order_hint,
                        h.show_frame,
                        h.refresh_frame_flags,
                        medians[0],
                        medians[1],
                        medians[1] / medians[0]
                    );
                    targets.remove(target);
                }
            }
            if decoder.decode_obu(&obu).unwrap().is_some() {
                displayed += 1;
            }
            if targets.is_empty() {
                return;
            }
        }
        panic!("lifecycle benchmark target frames missing");
    }

    #[test]
    fn persistent_decoder_preserves_references_on_failed_frame() {
        let mut stream = super::super::ObuStream::new();
        let obus = stream.push(CONSTANT_OBUS).unwrap();
        let mut decoder = Av1Decoder::new();
        assert!(decoder.decode_obu(&obus[0]).unwrap().is_none());
        let sequence = obus.iter().find(|o| o.kind == 1).unwrap();
        decoder.decode_obu(sequence).unwrap();
        let coded = obus.iter().find(|o| o.kind == 6).unwrap();
        let frame = decoder.decode_obu(coded).unwrap().unwrap();
        assert_eq!(
            frame,
            decode_intra_frame(coded, decoder.sequence().unwrap()).unwrap()
        );
        assert!(
            decoder
                .references
                .iter()
                .all(|r| r.as_ref().is_some_and(|r| *r.frame == frame))
        );
        let saved = decoder.references[0].as_ref().unwrap().clone();
        let mut broken = coded.clone();
        broken.payload.truncate(2);
        assert!(decoder.decode_obu(&broken).is_err());
        assert!(Arc::ptr_eq(&saved, decoder.references[0].as_ref().unwrap()));
        // A displayed keyframe is not a showable reference.
        let show_existing = Obu {
            kind: 3,
            temporal_id: 0,
            spatial_id: 0,
            payload: vec![0x88],
        };
        assert_eq!(
            decoder.decode_obu(&show_existing),
            Err(Error::Invalid("reference is not showable"))
        );
        assert!(Arc::ptr_eq(&saved, decoder.references[0].as_ref().unwrap()));
        decoder.reset();
        assert!(decoder.sequence().is_none());
        assert!(decoder.references.iter().all(Option::is_none));
    }

    #[test]
    #[ignore = "requires AV1_STREAM_OBU full Spacewalk binary fixture"]
    fn spacewalk_inter_header_walk() {
        let bytes = std::fs::read(std::env::var("AV1_STREAM_OBU").unwrap()).unwrap();
        let mut stream = super::super::ObuStream::new();
        let obus = stream.push(&bytes).unwrap();
        let mut sequence = None;
        let mut references: [Option<IntraFrameHeader>; 8] = Default::default();
        let mut coded = 0;
        for obu in obus {
            if obu.kind == 1 {
                sequence = Some(SequenceHeader::parse(&obu.payload).unwrap());
            }
            if obu.kind != 6 {
                continue;
            }
            if obu.payload[0] & 128 != 0 {
                continue;
            }
            let refs = references.each_ref().map(Option::as_ref);
            let h = IntraFrameHeader::parse_with_refs(
                &obu.payload,
                sequence.as_ref().unwrap(),
                0,
                0,
                &refs,
            )
            .unwrap();
            println!(
                "frame {coded}: type={} show={} hint={} primary={} refs={:?} refresh={} header={} q={} interp={} motion={} refmvs={} compound={} skip={:?}/{} warp={} gm={:?}",
                h.frame_type,
                h.show_frame,
                h.order_hint,
                h.primary_ref_frame,
                h.ref_frame_idx,
                h.refresh_frame_flags,
                h.header_bytes,
                h.base_q_idx,
                h.interpolation_filter,
                h.is_motion_mode_switchable,
                h.use_ref_frame_mvs,
                h.reference_select,
                h.skip_mode_frames,
                h.skip_mode_present,
                h.allow_warped_motion,
                h.global_motion_types
            );
            println!(
                "tools: screen={} intrabc={} segmentation={} qmatrix={:?} restoration={:?} delta_lf={:?}",
                h.allow_screen_content_tools,
                h.allow_intrabc,
                h.segmentation_enabled,
                h.quantizer_matrix_levels,
                h.restoration_types,
                h.delta_lf
            );
            let tiles =
                super::super::syntax::tile_group(&obu.payload[h.header_bytes..], &h.tiles).unwrap();
            assert_eq!(tiles.len(), 1);
            for (i, r) in references.iter_mut().enumerate() {
                if h.refresh_frame_flags & (1 << i) != 0 {
                    *r = Some(h.clone());
                }
            }
            coded += 1;
            if coded == 24 {
                break;
            }
        }
        assert_eq!(coded, 24);
    }

    #[test]
    #[ignore = "requires AV1_STREAM_OBU and AV1_STREAM_ORACLE binary fixtures"]
    fn spacewalk_first_hidden_filter_stages() {
        spacewalk_filter_stages(16);
    }

    #[test]
    #[ignore = "diagnostic only; requires AV1_STREAM_OBU and AV1_STREAM_ORACLE binary fixtures"]
    fn spacewalk_hint8_diagnostic_stages() {
        spacewalk_filter_stages(8);
    }

    #[test]
    #[ignore = "diagnostic only; requires coded stream, selected display oracle and hint occurrence"]
    fn spacewalk_selected_frame_filter_stages() {
        spacewalk_filter_stages(
            std::env::var("AV1_DIAGNOSTIC_HINT")
                .unwrap()
                .parse()
                .unwrap(),
        );
    }

    fn spacewalk_filter_stages(hint: u32) {
        let bytes = std::fs::read(std::env::var("AV1_STREAM_OBU").unwrap()).unwrap();
        let oracle = std::fs::read(std::env::var("AV1_STREAM_ORACLE").unwrap()).unwrap();
        let mut decoder = Av1Decoder::new();
        let mut stream = super::super::ObuStream::new();
        let occurrence = std::env::var("AV1_DIAGNOSTIC_OCCURRENCE")
            .ok()
            .map_or(1, |v| v.parse::<usize>().unwrap());
        let oracle_frame = std::env::var("AV1_STAGE_ORACLE_FRAME")
            .ok()
            .map_or(hint as usize, |v| v.parse::<usize>().unwrap());
        let mut matching = 0;
        for obu in stream.push(&bytes).unwrap() {
            if decoder.references[0].is_none() || obu.kind != 6 {
                decoder.decode_obu(&obu).unwrap();
                continue;
            }
            let sequence = decoder.sequence.as_ref().unwrap();
            let refs = decoder
                .references
                .each_ref()
                .map(|r| r.as_ref().map(|r| &r.frame.header));
            let header =
                IntraFrameHeader::parse_with_refs(&obu.payload, sequence, 0, 0, &refs).unwrap();
            if header.order_hint != hint {
                decoder.decode_obu(&obu).unwrap();
                continue;
            }
            matching += 1;
            if matching < occurrence {
                decoder.decode_obu(&obu).unwrap();
                continue;
            }
            eprintln!("target hint={hint} occurrence={matching} header={header:?}");
            let groups = super::super::syntax::tile_group(
                &obu.payload[header.header_bytes..],
                &header.tiles,
            )
            .unwrap();
            let mut tile = IntraTile::new(sequence, &header, groups[0].1).unwrap();
            tile.load_cdfs(
                &decoder.references
                    [header.ref_frame_idx[header.primary_ref_frame as usize] as usize]
                    .as_ref()
                    .unwrap()
                    .cdfs,
            );
            tile.set_references(header.ref_frame_idx.map(|i| {
                decoder.references[i as usize]
                    .as_ref()
                    .map(|r| r.frame.as_ref())
            }));
            if header.use_ref_frame_mvs {
                tile.set_motion_field(super::super::temporal::MotionField::new(
                    sequence,
                    &header,
                    header
                        .ref_frame_idx
                        .map(|i| decoder.references[i as usize].as_ref().map(|r| &r.motion)),
                ));
            }
            let result = tile.run_with_observer(|stage, tile| {
                let mut offset = oracle_frame * 3_110_400;
                for (plane, p) in tile.planes.iter().enumerate() {
                    let width = (header.width as usize)
                        .div_ceil(1 << usize::from(plane > 0 && sequence.subsampling_x));
                    let height = (header.height as usize)
                        .div_ceil(1 << usize::from(plane > 0 && sequence.subsampling_y));
                    let expected = &oracle[offset..offset + width * height];
                    let mut differing = 0;
                    let mut max_error = 0;
                    for y in 0..height {
                        for x in 0..width {
                            let error = (i32::from(p.samples[y * p.stride + x])
                                - i32::from(expected[y * width + x]))
                            .abs();
                            differing += usize::from(error != 0);
                            max_error = max_error.max(error);
                        }
                    }
                    eprintln!("stage={stage} plane={plane} differing={differing} max={max_error}");
                    offset += width * height;
                }
            });
            if let Err(error) = result {
                eprintln!("hint={hint} incomplete diagnostic: {error:?}");
                let p = &tile.planes[0];
                let expected =
                    &oracle[oracle_frame * 3_110_400..oracle_frame * 3_110_400 + 1920 * 1080];
                let mut printed = 0;
                for block in tile.progress.blocks.iter().filter(|b| b.reconstructed) {
                    let mut max_error = 0;
                    for y in block.y..(block.y + block.height).min(1080) {
                        for x in block.x..(block.x + block.width).min(1920) {
                            max_error = max_error.max(
                                (i32::from(p.samples[y * p.stride + x])
                                    - i32::from(expected[y * 1920 + x]))
                                .abs(),
                            );
                        }
                    }
                    if max_error > 16 {
                        eprintln!(
                            "first divergent reconstructed block {block:?} max={max_error} cell={:?}",
                            tile.test_motion_at(block.x, block.y)
                        );
                        printed += 1;
                    }
                    if printed == 10 {
                        break;
                    }
                }
                assert_eq!(hint, 8, "unexpected diagnostic decode failure");
            }
            return;
        }
        panic!("missing first hidden frame");
    }

    #[test]
    #[ignore = "requires AV1_STREAM_OBU and AV1_STREAM_ORACLE binary fixtures"]
    fn spacewalk_first_hidden_reference_oracle() {
        let bytes = std::fs::read(std::env::var("AV1_STREAM_OBU").unwrap()).unwrap();
        let oracle = std::fs::read(std::env::var("AV1_STREAM_ORACLE").unwrap()).unwrap();
        assert!(oracle.len() >= 17 * 3_110_400);
        let mut decoder = Av1Decoder::new();
        let mut stream = super::super::ObuStream::new();
        for obu in stream.push(&bytes).unwrap() {
            decoder.decode_obu(&obu).unwrap();
            if let Some(reference) = &decoder.references[1] {
                if reference.frame.header.order_hint != 16 {
                    continue;
                }
                let mut offset = 16 * 3_110_400;
                let mut total = 0;
                for (plane, p) in reference.frame.planes.iter().enumerate() {
                    let expected = &oracle[offset..offset + p.samples.len()];
                    let differences: Vec<_> = p
                        .samples
                        .iter()
                        .zip(expected)
                        .enumerate()
                        .filter(|(_, (a, b))| **a != u16::from(**b))
                        .collect();
                    let max_error = differences
                        .iter()
                        .map(|(_, (a, b))| (i32::from(**a) - i32::from(**b)).abs())
                        .max()
                        .unwrap_or(0);
                    let largest = differences
                        .iter()
                        .max_by_key(|(_, (a, b))| (i32::from(**a) - i32::from(**b)).abs());
                    let mut tiles = std::collections::BTreeMap::new();
                    for (i, _) in &differences {
                        *tiles
                            .entry((i / p.width / 64, i % p.width / 64))
                            .or_insert(0_usize) += 1;
                    }
                    eprintln!(
                        "hidden hint16 plane={plane} differing={} max={max_error} largest={largest:?} first={:?} tiles={tiles:?}",
                        differences.len(),
                        differences.first()
                    );
                    total += differences.len();
                    offset += p.samples.len();
                }
                assert_eq!(total, 0, "hidden hint16 reference");
                return;
            }
        }
        panic!("missing hidden hint16 reference");
    }

    #[test]
    #[ignore = "requires AV1_STREAM_OBU and AV1_STREAM_ORACLE binary fixtures"]
    fn spacewalk_full_first16_display_frames() {
        spacewalk_display_oracle(16);
    }

    #[test]
    #[ignore = "requires AV1_STREAM_OBU and AV1_STREAM_ORACLE first32 binary fixtures"]
    fn spacewalk_full_first32_display_frames() {
        spacewalk_display_oracle(32);
    }

    #[test]
    #[ignore = "test-only fused transform candidate; requires AV1_STREAM_OBU and first32 AV1_STREAM_ORACLE"]
    fn spacewalk_fused_columns_first32_binary_oracle() {
        super::super::transform::set_fused_columns_candidate(true);
        spacewalk_display_oracle(32);
        super::super::transform::set_fused_columns_candidate(false);
    }

    #[test]
    #[ignore = "test-only skip-generic prediction candidate; requires AV1_STREAM_OBU and first32 AV1_STREAM_ORACLE"]
    fn spacewalk_skip_generic_first32_binary_oracle() {
        super::super::intra::set_skip_generic_prediction_candidate(true);
        spacewalk_display_oracle(32);
        super::super::intra::set_skip_generic_prediction_candidate(false);
    }

    #[test]
    #[ignore = "test-only reused coefficient storage; requires AV1_STREAM_OBU and first32 AV1_STREAM_ORACLE"]
    fn spacewalk_reuse_coefficients_first32_binary_oracle() {
        super::super::intra::set_reuse_coefficients_candidate(true);
        spacewalk_display_oracle(32);
        super::super::intra::set_reuse_coefficients_candidate(false);
    }

    #[test]
    #[ignore = "production EOB-bounded dequantization; requires AV1_STREAM_OBU and first32 AV1_STREAM_ORACLE"]
    fn spacewalk_production_eob_first32_binary_oracle() {
        spacewalk_display_oracle(32);
    }

    #[test]
    #[ignore = "exact original/fast inter reconstruction plus selected 16-21s/28s binary pixels; requires AV1_STREAM_OBU and AV1_LATER_ORACLE"]
    fn spacewalk_inter_prediction_later_pixels() {
        let targets = [480, 510, 540, 570, 600, 630, 840, 860];
        let profile_phases = std::env::var_os("AV1_HEAVY_PHASE_PROFILE").is_some();
        let skip_generic = std::env::var_os("AV1_SKIP_GENERIC_PREDICTION_CANDIDATE").is_some();
        let reuse_coefficients = std::env::var_os("AV1_REUSE_COEFFICIENTS_CANDIDATE").is_some();
        assert!(usize::from(skip_generic) + usize::from(reuse_coefficients) <= 1);
        let oracle = std::fs::read(std::env::var("AV1_LATER_ORACLE").unwrap()).unwrap();
        assert_eq!(oracle.len(), targets.len() * 3_110_400);
        let bytes = std::fs::read(std::env::var("AV1_STREAM_OBU").unwrap()).unwrap();
        let mut stream = super::super::ObuStream::new();
        let mut fast = Av1Decoder::new();
        let mut reference = Av1Decoder::new();
        let mut displayed = 0;
        let mut checked = 0;
        for obu in stream.push(&bytes).unwrap() {
            super::super::intra::set_discarded_intra_reference(true);
            super::super::reconstruction::set_lazy_edges_reference(true);
            super::super::intra::set_eob_dequant_reference(true);
            let expected = reference.decode_obu(&obu);
            super::super::intra::set_eob_dequant_reference(false);
            super::super::intra::set_discarded_intra_reference(false);
            super::super::reconstruction::set_lazy_edges_reference(false);
            let counting = [480, 600, 840].contains(&displayed) && obu.kind == 6
                && obu.payload.first().is_some_and(|b| b & 128 == 0);
            super::super::intra::reset_reconstruction_counts(counting);
            super::super::profile::reset(counting && profile_phases);
            super::super::intra::set_skip_generic_prediction_candidate(skip_generic);
            super::super::intra::set_reuse_coefficients_candidate(reuse_coefficients);
            let actual = fast.decode_obu(&obu);
            super::super::intra::set_reuse_coefficients_candidate(false);
            super::super::intra::set_skip_generic_prediction_candidate(false);
            if counting && profile_phases {
                let stages = super::super::profile::stages().map(|n| n as f64 / 1_000_000.0);
                let parts = super::super::profile::times().map(|n| n as f64 / 1_000_000.0);
                eprintln!("HEAVY_PHASE display={displayed} diagnostic_wall_ms reconstruct={:.3} deblock={:.3} cdef={:.3} restoration={:.3} motion={:.3} coefficients={:.3} transform={:.3} base_intra={:.3} warp={:.3} reconstruct_other={:.3}",
                    stages[0], stages[1], stages[2], stages[3], parts[0], parts[1], parts[2],
                    parts[3], parts[4], stages[0] - parts.iter().sum::<f64>());
            }
            super::super::profile::reset(false);
            let counts = super::super::intra::take_reconstruction_counts();
            if counting {
                eprintln!("HEAVY_COUNTS display={displayed} frame_type={:?} {counts:?}",
                    actual.as_ref().ok().and_then(|f| f.as_ref()).map(|f| f.header.frame_type));
            }
            {
                for (expected, actual) in reference.references.iter().zip(&fast.references) {
                    match (expected, actual) {
                        (Some(expected), Some(actual)) => {
                            assert_eq!(actual.frame.planes, expected.frame.planes,
                                "display {displayed}: coefficient reference planes");
                            assert!(actual.cdfs.coefficient_state_matches(&expected.cdfs),
                                "display {displayed}: saved coefficient contexts");
                        }
                        (None, None) => {}
                        _ => panic!("coefficient reference presence changed"),
                    }
                }
            }
            match (expected, actual) {
                (Ok(Some(expected)), Ok(Some(actual))) => {
                    assert_eq!(actual.planes, expected.planes, "display {displayed}");
                    if targets.get(checked) == Some(&displayed) {
                        let mut offset = checked * 3_110_400;
                        for plane in &actual.planes {
                            for &sample in &plane.samples {
                                assert_eq!(sample, u16::from(oracle[offset]),
                                    "display {displayed} byte {offset}");
                                offset += 1;
                            }
                        }
                        assert_eq!(offset, (checked + 1) * 3_110_400);
                        eprintln!("exact later inter reconstruction display={displayed}");
                        checked += 1;
                        if checked == targets.len() { return; }
                    }
                    displayed += 1;
                }
                (Ok(None), Ok(None)) => {
                    for (expected, actual) in reference.references.iter().zip(&fast.references) {
                        match (expected, actual) {
                            (Some(expected), Some(actual)) => assert_eq!(actual.frame.planes,
                                expected.frame.planes, "hidden reference before display {displayed}"),
                            (None, None) => {}
                            _ => panic!("reference presence changed"),
                        }
                    }
                }
                (Err(expected), Err(actual)) => panic!("display {displayed} blocked: {expected:?}/{actual:?}"),
                _ => panic!("display {displayed} output presence changed"),
            }
        }
        panic!("selected later display frames missing");
    }

    #[test]
    #[ignore = "malformed inter-tile original/fast error and atomic reference parity; requires AV1_STREAM_OBU"]
    fn spacewalk_inter_prediction_malformed_parity() {
        let skip_generic = std::env::var_os("AV1_SKIP_GENERIC_PREDICTION_CANDIDATE").is_some();
        let reuse_coefficients = std::env::var_os("AV1_REUSE_COEFFICIENTS_CANDIDATE").is_some();
        assert!(usize::from(skip_generic) + usize::from(reuse_coefficients) <= 1);
        let selected_display = std::env::var("AV1_MALFORMED_DISPLAY").ok()
            .map(|s| s.parse::<usize>().unwrap());
        let bytes = std::fs::read(std::env::var("AV1_STREAM_OBU").unwrap()).unwrap();
        let mut stream = super::super::ObuStream::new();
        let mut base = Av1Decoder::new();
        let mut displayed = 0;
        for obu in stream.push(&bytes).unwrap() {
            if obu.kind == 6 && obu.payload.first().is_some_and(|b| b & 128 == 0)
                && base.sequence.is_some()
            {
                let refs = base.references.each_ref().map(|r| r.as_ref().map(|r| &r.frame.header));
                let h = IntraFrameHeader::parse_with_refs(&obu.payload,
                    base.sequence.as_ref().unwrap(), obu.temporal_id, obu.spatial_id, &refs).unwrap();
                if selected_display.map_or(h.frame_type == 1, |selected| displayed == selected) {
                    for point in [0, 1, h.header_bytes, h.header_bytes + 1,
                        obu.payload.len() / 2, obu.payload.len() - 1] {
                        for truncate in [false, true] {
                            let mut mutated = obu.clone();
                            if truncate { mutated.payload.truncate(point); }
                            else { mutated.payload[point] ^= 0x80; }
                            let mut original = base.clone();
                            let mut optimized = base.clone();
                            super::super::intra::set_discarded_intra_reference(true);
                            super::super::reconstruction::set_lazy_edges_reference(true);
                            super::super::intra::set_eob_dequant_reference(true);
                            let expected = original.decode_obu(&mutated);
                            super::super::intra::set_eob_dequant_reference(false);
                            super::super::intra::set_discarded_intra_reference(false);
                            super::super::reconstruction::set_lazy_edges_reference(false);
                            super::super::intra::set_skip_generic_prediction_candidate(skip_generic);
                            super::super::intra::set_reuse_coefficients_candidate(reuse_coefficients);
                            let actual = optimized.decode_obu(&mutated);
                            super::super::intra::set_reuse_coefficients_candidate(false);
                            super::super::intra::set_skip_generic_prediction_candidate(false);
                            match (expected, actual) {
                                (Err(expected), Err(actual)) => {
                                    assert_eq!(actual, expected);
                                    for decoder in [&original, &optimized] {
                                        for (before, after) in base.references.iter().zip(&decoder.references) {
                                            match (before, after) {
                                                (Some(before), Some(after)) => assert!(Arc::ptr_eq(before, after)),
                                                (None, None) => {}
                                                _ => panic!("failed inter frame changed reference presence"),
                                            }
                                        }
                                    }
                                }
                                (Ok(Some(expected)), Ok(Some(actual))) => assert_eq!(actual.planes, expected.planes),
                                (Ok(None), Ok(None)) => {}
                                _ => panic!("mutated inter-frame result changed"),
                            }
                            for (expected, actual) in original.references.iter().zip(&optimized.references) {
                                match (expected, actual) {
                                    (Some(expected), Some(actual)) => {
                                        assert_eq!(actual.frame.planes, expected.frame.planes);
                                        assert!(actual.cdfs.coefficient_state_matches(&expected.cdfs));
                                    }
                                    (None, None) => {}
                                    _ => panic!("mutated inter-frame references changed"),
                                }
                            }
                        }
                    }
                    return;
                }
            }
            if base.decode_obu(&obu).unwrap().is_some() { displayed += 1; }
        }
        panic!("missing first inter frame");
    }

    #[test]
    #[ignore = "same-state whole decode_obu later-frame ABBA CPU/wall; requires AV1_STREAM_OBU, exclusive timing slot"]
    fn spacewalk_discarded_intra_later_abba() {
        use std::time::Instant;
        #[cfg(target_os = "macos")]
        fn cpu_clock() -> u64 {
            unsafe extern "C" { fn clock_gettime_nsec_np(clock_id: i32) -> u64; }
            unsafe { clock_gettime_nsec_np(16) }
        }
        #[cfg(not(target_os = "macos"))]
        fn cpu_clock() -> u64 { 0 }
        let repeats = std::env::var("AV1_BENCH_REPEATS").ok()
            .map_or(12, |v| v.parse::<usize>().unwrap());
        assert!(repeats >= 4 && repeats % 2 == 0);
        let lazy_edges_ab = std::env::var_os("AV1_BENCH_LAZY_EDGES_AB").is_some();
        let skip_generic_ab = std::env::var_os("AV1_BENCH_SKIP_GENERIC_AB").is_some();
        let reuse_coefficients_ab = std::env::var_os("AV1_BENCH_REUSE_COEFFICIENTS_AB").is_some();
        let eob_dequant_ab = std::env::var_os("AV1_BENCH_EOB_DEQUANT_AB").is_some();
        assert!(usize::from(lazy_edges_ab) + usize::from(skip_generic_ab)
            + usize::from(reuse_coefficients_ab) + usize::from(eob_dequant_ab) <= 1);
        let mut targets = if lazy_edges_ab || skip_generic_ab || reuse_coefficients_ab || eob_dequant_ab { vec![480, 600, 840] }
            else { vec![480, 510, 540, 570, 600, 630, 840, 860] };
        let bytes = std::fs::read(std::env::var("AV1_STREAM_OBU").unwrap()).unwrap();
        let mut stream = super::super::ObuStream::new();
        let mut base = Av1Decoder::new();
        let mut displayed = 0;
        super::super::profile::reset(false);
        for obu in stream.push(&bytes).unwrap() {
            if obu.kind == 6 && obu.payload.first().is_some_and(|b| b & 128 == 0)
                && displayed >= targets[0]
            {
                let requested = targets.remove(0);
                let refs = base.references.each_ref().map(|r| r.as_ref().map(|r| &r.frame.header));
                let h = IntraFrameHeader::parse_with_refs(&obu.payload,
                    base.sequence.as_ref().unwrap(), obu.temporal_id, obu.spatial_id, &refs).unwrap();
                let mut cpu: [Vec<f64>; 2] = Default::default();
                let mut wall: [Vec<f64>; 2] = Default::default();
                let mut expected_output: Option<Option<Vec<DecodedPlane>>> = None;
                let mut expected_state: Option<Av1Decoder> = None;
                for trial in 0..4 + repeats * 2 {
                    let reference = [true, false, false, true][trial % 4];
                    super::super::intra::set_discarded_intra_reference(reference && !lazy_edges_ab && !skip_generic_ab && !reuse_coefficients_ab && !eob_dequant_ab);
                    super::super::reconstruction::set_lazy_edges_reference(reference && lazy_edges_ab);
                    super::super::intra::set_skip_generic_prediction_candidate(!reference && skip_generic_ab);
                    super::super::intra::set_reuse_coefficients_candidate(!reference && reuse_coefficients_ab);
                    super::super::intra::set_eob_dequant_reference(reference && eob_dequant_ab);
                    let mut decoder = base.clone();
                    let wall_start = Instant::now();
                    let cpu_start = cpu_clock();
                    let decoded = decoder.decode_obu(std::hint::black_box(&obu)).unwrap();
                    let cpu_elapsed = (cpu_clock() - cpu_start) as f64 / 1_000_000.0;
                    let wall_elapsed = wall_start.elapsed().as_secs_f64() * 1000.0;
                    if let Some(expected) = &expected_output {
                        assert_eq!(decoded.as_ref().map(|f| &f.planes), expected.as_ref(),
                            "display {displayed} output planes");
                    } else {
                        expected_output = Some(decoded.as_ref().map(|f| f.planes.clone()));
                        expected_state = Some(decoder.clone());
                    }
                    for (expected, actual) in expected_state.as_ref().unwrap().references.iter()
                        .zip(&decoder.references)
                    {
                        match (expected, actual) {
                            (Some(expected), Some(actual)) => {
                                assert_eq!(actual.frame.planes, expected.frame.planes,
                                    "display {displayed} reference planes");
                                assert!(actual.cdfs.coefficient_state_matches(&expected.cdfs),
                                    "display {displayed} saved coefficient contexts");
                            }
                            (None, None) => {}
                            _ => panic!("ABBA reference presence changed"),
                        }
                    }
                    if trial >= 4 {
                        cpu[usize::from(reference)].push(cpu_elapsed);
                        wall[usize::from(reference)].push(wall_elapsed);
                    }
                    std::hint::black_box(decoded);
                    std::hint::black_box(decoder);
                }
                super::super::intra::set_discarded_intra_reference(false);
                super::super::reconstruction::set_lazy_edges_reference(false);
                super::super::intra::set_skip_generic_prediction_candidate(false);
                super::super::intra::set_reuse_coefficients_candidate(false);
                super::super::intra::set_eob_dequant_reference(false);
                for mode in 0..2 {
                    cpu[mode].sort_by(f64::total_cmp);
                    wall[mode].sort_by(f64::total_cmp);
                    eprintln!("WHOLE_ABBA requested={requested} display={displayed} hint={} show={} reference={} repeats={repeats} thread_cpu={} cpu_ms={:.3} wall_ms={:.3}",
                        h.order_hint, h.show_frame, mode == 1, cfg!(target_os = "macos"),
                        cpu[mode][repeats / 2], wall[mode][repeats / 2]);
                }
            }
            super::super::intra::set_discarded_intra_reference(false);
            super::super::reconstruction::set_lazy_edges_reference(false);
            super::super::intra::set_skip_generic_prediction_candidate(false);
            super::super::intra::set_reuse_coefficients_candidate(false);
            super::super::intra::set_eob_dequant_reference(false);
            if base.decode_obu(&obu).unwrap().is_some() { displayed += 1; }
            if targets.is_empty() { return; }
        }
        panic!("later ABBA targets missing");
    }

    #[test]
    #[ignore = "controlled repeated-frame stage benchmark; requires AV1_STREAM_OBU"]
    fn spacewalk_repeated_stage_benchmark() {
        use std::time::Instant;
        #[cfg(target_os = "macos")]
        fn cpu_clock() -> u64 {
            unsafe extern "C" {
                fn clock_gettime_nsec_np(clock_id: i32) -> u64;
            }
            unsafe { clock_gettime_nsec_np(16) }
        }
        #[cfg(not(target_os = "macos"))]
        fn cpu_clock() -> u64 { 0 }
        let thread_cpu = cfg!(target_os = "macos")
            && std::env::var_os("AV1_BENCH_THREAD_CPU").is_some();
        let repeats = std::env::var("AV1_BENCH_REPEATS")
            .ok()
            .map_or(12, |v| v.parse::<usize>().unwrap());
        assert!(repeats >= 3);
        let cdef_ab = std::env::var_os("AV1_BENCH_CDEF_AB").is_some();
        let deblock_ab = std::env::var_os("AV1_BENCH_DEBLOCK_AB").is_some();
        let motion_ab = std::env::var_os("AV1_BENCH_MOTION_AB").is_some();
        let buffers_ab = std::env::var_os("AV1_BENCH_CDEF_BUFFERS_AB").is_some();
        let restoration_ab = std::env::var_os("AV1_BENCH_RESTORATION_AB").is_some();
        let scratch_ab = std::env::var_os("AV1_BENCH_INTER_SCRATCH_AB").is_some();
        let coefficients_ab = std::env::var_os("AV1_BENCH_COEFFICIENTS_AB").is_some();
        let fused_columns_ab = std::env::var_os("AV1_BENCH_FUSED_COLUMNS_AB").is_some();
        let discarded_intra_ab = std::env::var_os("AV1_BENCH_DISCARDED_INTRA_AB").is_some();
        let ab = cdef_ab || deblock_ab || motion_ab || buffers_ab || restoration_ab || scratch_ab
            || coefficients_ab || fused_columns_ab || discarded_intra_ab;
        let profile = std::env::var_os("AV1_BENCH_PROFILE").is_some();
        let first_display = std::env::var("AV1_BENCH_START_DISPLAY")
            .ok()
            .map_or(0, |v| v.parse::<usize>().unwrap());
        let bytes = std::fs::read(std::env::var("AV1_STREAM_OBU").unwrap()).unwrap();
        let mut stream = super::super::ObuStream::new();
        let mut decoder = Av1Decoder::new();
        let mut targets = vec![0, 16, 8, 1];
        let mut displayed = 0;
        for obu in stream.push(&bytes).unwrap() {
            if obu.kind == 6
                && obu.payload.first().is_some_and(|b| b & 128 == 0)
                && displayed >= first_display
            {
                let sequence = decoder.sequence.as_ref().unwrap();
                let refs = decoder
                    .references
                    .each_ref()
                    .map(|r| r.as_ref().map(|r| &r.frame.header));
                let h = IntraFrameHeader::parse_with_refs(
                    &obu.payload,
                    sequence,
                    obu.temporal_id,
                    obu.spatial_id,
                    &refs,
                )
                .unwrap();
                let target = if first_display != 0 {
                    Some(0)
                } else {
                    targets.iter().position(|&v| v == h.order_hint)
                };
                if let Some(target) = target {
                    let groups =
                        super::super::syntax::tile_group(&obu.payload[h.header_bytes..], &h.tiles)
                            .unwrap();
                    let mut timings: [[Vec<f64>; 5]; 2] = Default::default();
                    let mut details: [Vec<f64>; 5] = Default::default();
                    let mut expected = None;
                    let trials = if ab { repeats * 2 + 4 } else { repeats + 2 };
                    for repeat in 0..trials {
                        let reference = ab && [true, false, false, true][repeat % 4];
                        super::super::intra::set_discarded_intra_reference(reference && discarded_intra_ab);
                        super::super::transform::set_fused_columns_candidate(
                            !reference && fused_columns_ab,
                        );
                        super::super::coefficients::set_allocation_reference(
                            reference && coefficients_ab,
                        );
                        super::super::intra::set_inter_scratch_reference(reference && scratch_ab);
                        super::super::filters::set_cdef_reference(reference && cdef_ab);
                        super::super::filters::set_deblock_reference(reference && deblock_ab);
                        super::super::motion::set_motion_reference(reference && motion_ab);
                        super::super::filters::set_cdef_snapshot_reference(reference && buffers_ab);
                        super::super::restoration::set_restoration_reference(
                            reference && restoration_ab,
                        );
                        let mut tile = IntraTile::new(sequence, &h, groups[0].1).unwrap();
                        if h.primary_ref_frame != 7 {
                            tile.load_cdfs(
                                &decoder.references
                                    [h.ref_frame_idx[h.primary_ref_frame as usize] as usize]
                                    .as_ref()
                                    .unwrap()
                                    .cdfs,
                            );
                        }
                        tile.set_references(h.ref_frame_idx.map(|i| {
                            decoder.references[i as usize]
                                .as_ref()
                                .map(|r| r.frame.as_ref())
                        }));
                        if h.use_ref_frame_mvs {
                            tile.set_motion_field(super::super::temporal::MotionField::new(
                                sequence,
                                &h,
                                h.ref_frame_idx.map(|i| {
                                    decoder.references[i as usize].as_ref().map(|r| &r.motion)
                                }),
                            ));
                        }
                        super::super::profile::reset(profile);
                        let start = Instant::now();
                        let mut last = start;
                        let cpu_start = if thread_cpu { cpu_clock() } else { 0 };
                        let mut cpu_last = cpu_start;
                        let mut durations = [0.0; 5];
                        let mut stage = 0;
                        tile.run_with_observer(|_, _| {
                            let now = Instant::now();
                            durations[stage] = now.duration_since(last).as_secs_f64() * 1000.0;
                            if thread_cpu {
                                let cpu_now = cpu_clock();
                                durations[stage] = (cpu_now - cpu_last) as f64 / 1_000_000.0;
                                cpu_last = cpu_now;
                            }
                            last = now;
                            stage += 1;
                        })
                        .unwrap();
                        durations[4] = start.elapsed().as_secs_f64() * 1000.0;
                        if thread_cpu {
                            durations[4] = (cpu_clock() - cpu_start) as f64 / 1_000_000.0;
                        }
                        assert_eq!(stage, 4);
                        if let Some(expected) = &expected {
                            assert!(tile.planes == *expected, "repeated frame changed samples");
                        } else {
                            expected = Some(tile.planes.clone());
                        }
                        if repeat >= if ab { 4 } else { 2 } {
                            for (values, nanos) in
                                details.iter_mut().zip(super::super::profile::times())
                            {
                                values.push(nanos as f64 / 1_000_000.0);
                            }
                            for (values, duration) in
                                timings[usize::from(reference)].iter_mut().zip(durations)
                            {
                                values.push(duration);
                            }
                        }
                        std::hint::black_box(&tile.planes);
                    }
                    super::super::filters::set_cdef_reference(false);
                    super::super::filters::set_deblock_reference(false);
                    super::super::motion::set_motion_reference(false);
                    super::super::filters::set_cdef_snapshot_reference(false);
                    super::super::restoration::set_restoration_reference(false);
                    super::super::intra::set_inter_scratch_reference(false);
                    super::super::coefficients::set_allocation_reference(false);
                    super::super::transform::set_fused_columns_candidate(false);
                    super::super::intra::set_discarded_intra_reference(false);
                    super::super::profile::reset(false);
                    if profile {
                        let medians = details.map(|mut values| {
                            values.sort_by(f64::total_cmp);
                            values[values.len() / 2]
                        });
                        eprintln!(
                            "PROFILE hint={} motion_ms={:.3} coefficients_ms={:.3} transform_ms={:.3} intra_prediction_ms={:.3} warp_ms={:.3}",
                            h.order_hint,
                            medians[0],
                            medians[1],
                            medians[2],
                            medians[3],
                            medians[4]
                        );
                    }
                    for (mode, times) in timings.into_iter().enumerate() {
                        if times[0].is_empty() {
                            continue;
                        }
                        let medians = times.map(|mut values| {
                            values.sort_by(f64::total_cmp);
                            values[values.len() / 2]
                        });
                        eprintln!(
                            "BENCH hint={} reference={} repeats={repeats} reconstruct_ms={:.3} deblock_ms={:.3} cdef_ms={:.3} restoration_ms={:.3} total_ms={:.3}",
                            h.order_hint,
                            mode == 1,
                            medians[0],
                            medians[1],
                            medians[2],
                            medians[3],
                            medians[4]
                        );
                    }
                    targets.remove(target);
                }
            }
            if decoder.decode_obu(&obu).unwrap().is_some() {
                displayed += 1;
            }
            if targets.is_empty() {
                return;
            }
        }
        panic!("benchmark target frames missing");
    }

    #[test]
    #[ignore = "requires AV1_STREAM_OBU and random-access first256 display oracle"]
    fn spacewalk_hidden_references_binary_oracle() {
        use std::io::{Read, Seek, SeekFrom};
        let bytes = std::fs::read(std::env::var("AV1_STREAM_OBU").unwrap()).unwrap();
        let mut oracle = std::fs::File::open(std::env::var("AV1_STREAM_ORACLE").unwrap()).unwrap();
        let mut expected = vec![0; 3_110_400];
        let mut decoder = Av1Decoder::new();
        let mut stream = super::super::ObuStream::new();
        let mut displayed = 0;
        for obu in stream.push(&bytes).unwrap() {
            let previous = decoder.references.clone();
            let decoded = decoder.decode_obu(&obu).unwrap_or_else(|error| {
                panic!("coded OBU at display={displayed} failed: {error:?}")
            });
            let stored = decoder
                .references
                .iter()
                .zip(&previous)
                .find_map(|(after, before)| {
                    after
                        .as_ref()
                        .filter(|after| {
                            before
                                .as_ref()
                                .is_none_or(|before| !Arc::ptr_eq(after, before))
                        })
                        .map(|r| r.frame.as_ref())
                });
            if let Some(frame) = decoded.as_ref().or(stored) {
                let index = displayed as i32
                    + super::super::syntax::relative_dist(
                        &frame.sequence,
                        frame.header.order_hint,
                        displayed as u32,
                    );
                assert!((0..256).contains(&index), "reference oracle index {index}");
                oracle
                    .seek(SeekFrom::Start(index as u64 * expected.len() as u64))
                    .unwrap();
                oracle.read_exact(&mut expected).unwrap();
                let pixels: Vec<u16> = frame
                    .planes
                    .iter()
                    .flat_map(|p| p.samples.iter().copied())
                    .collect();
                assert_eq!(pixels.len(), expected.len());
                let difference = pixels
                    .iter()
                    .zip(&expected)
                    .position(|(a, b)| *a != u16::from(*b));
                if let Some(sample) = difference {
                    let count = pixels
                        .iter()
                        .zip(&expected)
                        .filter(|(a, b)| **a != u16::from(**b))
                        .count();
                    panic!(
                        "coded hint={} oracle index={index} show={} at display={displayed} differing={count} first sample={sample} decoded={} oracle={}",
                        frame.header.order_hint,
                        frame.header.show_frame,
                        pixels[sample],
                        expected[sample]
                    );
                }
            }
            if decoded.is_some() {
                displayed += 1;
            }
            if displayed % 32 == 0 && decoded.is_some() {
                eprintln!("displayed={displayed}; all earlier decoded references exact");
            }
            if displayed == 224 {
                return;
            }
        }
        panic!("missing display frames");
    }

    #[test]
    #[ignore = "requires AV1_STREAM_OBU and FFmpeg binary for the complete stream oracle"]
    fn spacewalk_complete_stream_binary_oracle() {
        use std::io::Read;
        use std::process::{Command, Stdio};

        let path = std::env::var("AV1_STREAM_OBU").unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let mut oracle =
            Command::new(std::env::var("AV1_FFMPEG").unwrap_or_else(|_| "ffmpeg".into()))
                .args([
                    "-v", "error", "-i", &path, "-map", "0:v:0", "-pix_fmt", "yuv420p", "-f",
                    "rawvideo", "pipe:1",
                ])
                .stdout(Stdio::piped())
                .spawn()
                .unwrap();
        let result = (|| -> Result<usize, String> {
            let mut output = oracle.stdout.take().unwrap();
            let mut decoder = Av1Decoder::new();
            let mut stream = super::super::ObuStream::new();
            let mut displayed = 0;
            let mut expected = vec![0; 3_110_400];
            for obu in stream
                .push(&bytes)
                .map_err(|e| format!("OBU stream: {e:?}"))?
            {
                let previous = decoder.references.clone();
                let decoded = decoder.decode_obu(&obu);
                if let Err(error) = &decoded {
                    for (before, after) in previous.iter().zip(&decoder.references) {
                        assert!(
                            match (before, after) {
                                (None, None) => true,
                                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                                _ => false,
                            },
                            "failed coded frame changed references"
                        );
                    }
                    return Err(format!("display frame {displayed} blocked: {error:?}"));
                }
                let Some(frame) = decoded.unwrap() else {
                    continue;
                };
                let pixels: Vec<u8> = frame
                    .planes
                    .iter()
                    .flat_map(|p| p.samples.iter().map(|&v| u8::try_from(v).unwrap()))
                    .collect();
                if (frame.header.width, frame.header.height) != (1920, 1080)
                    || pixels.len() != expected.len()
                {
                    return Err(format!("display frame {displayed} unexpected dimensions"));
                }
                output
                    .read_exact(&mut expected)
                    .map_err(|e| format!("oracle frame {displayed}: {e}"))?;
                if let Some(index) = pixels.iter().zip(&expected).position(|(a, b)| a != b) {
                    let differences: Vec<_> = pixels
                        .iter()
                        .zip(&expected)
                        .enumerate()
                        .filter(|(_, (a, b))| a != b)
                        .collect();
                    eprintln!(
                        "hint={} differing={} first={:?} last={:?}",
                        frame.header.order_hint,
                        differences.len(),
                        differences.first(),
                        differences.last()
                    );
                    return Err(format!(
                        "display frame {displayed} sample {index}: decoded={} oracle={}",
                        pixels[index], expected[index]
                    ));
                }
                displayed += 1;
                if displayed % 32 == 0 {
                    eprintln!("exact display frames: {displayed}");
                }
            }
            let mut extra = [0];
            if output.read(&mut extra).map_err(|e| e.to_string())? != 0 {
                return Err("decoder missed oracle display frames".into());
            }
            Ok(displayed)
        })();
        if result.is_err() {
            let _ = oracle.kill();
        }
        let status = oracle.wait().unwrap();
        let displayed = result.unwrap();
        assert!(status.success(), "FFmpeg oracle failed: {status}");
        assert!(displayed >= 32);
        eprintln!("complete stream: {displayed} exact display frames");
    }

    fn spacewalk_display_oracle(count: usize) {
        let bytes = std::fs::read(std::env::var("AV1_STREAM_OBU").unwrap()).unwrap();
        let oracle = std::fs::read(std::env::var("AV1_STREAM_ORACLE").unwrap()).unwrap();
        assert_eq!(oracle.len(), count * 3_110_400);
        let mut decoder = Av1Decoder::new();
        let mut stream = super::super::ObuStream::new();
        let mut offset = 0;
        let mut displayed = 0;
        for obu in stream.push(&bytes).unwrap() {
            let previous = decoder.references.clone();
            let decoded = decoder.decode_obu(&obu);
            if let Err(error) = &decoded {
                for (before, after) in previous.iter().zip(&decoder.references) {
                    assert!(
                        match (before, after) {
                            (None, None) => true,
                            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                            _ => false,
                        },
                        "failed coded frame changed references"
                    );
                }
                panic!("display frame {displayed} blocked: {error:?}");
            }
            let Some(frame) = decoded.unwrap() else {
                continue;
            };
            let pixels: Vec<u8> = frame
                .planes
                .iter()
                .flat_map(|p| p.samples.iter().map(|&v| u8::try_from(v).unwrap()))
                .collect();
            assert_eq!((frame.header.width, frame.header.height), (1920, 1080));
            assert_eq!(pixels.len(), 3_110_400);
            let expected = &oracle[offset..offset + pixels.len()];
            let first_difference = pixels.iter().zip(expected).position(|(a, b)| a != b);
            assert_eq!(
                first_difference, None,
                "display frame {displayed} first sample mismatch"
            );
            offset += pixels.len();
            displayed += 1;
            if displayed == count {
                break;
            }
        }
        assert_eq!(displayed, count);
        assert_eq!(offset, oracle.len());
    }

    #[test]
    fn complete_nonzero_dc_frame_matches_binary_oracle() {
        // FFmpeg/SVT-AV1 binary fixture, geq Y=100,U=V=128 at 64x64.
        let bytes = [
            0x12, 0x00, 0x0a, 0x0b, 0x02, 0x00, 0x00, 0x05, 0x15, 0x7f, 0xfc, 0x4a, 0xf9, 0x00,
            0x40, 0x32, 0x0e, 0x10, 0x00, 0xac, 0x02, 0x05, 0x14, 0x20, 0x81, 0x00, 0x00, 0x03,
            0x25, 0x10, 0x88,
        ];
        let mut stream = super::super::ObuStream::new();
        let obus = stream.push(&bytes).unwrap();
        let sequence =
            SequenceHeader::parse(&obus.iter().find(|o| o.kind == 1).unwrap().payload).unwrap();
        let frame =
            decode_intra_frame(obus.iter().find(|o| o.kind == 6).unwrap(), &sequence).unwrap();
        assert_eq!(frame.sequence, sequence);
        assert!(frame.planes[0].samples.iter().all(|&v| v == 100));
        assert!(
            frame.planes[1..]
                .iter()
                .all(|p| p.samples.iter().all(|&v| v == 128))
        );
    }

    // Generated fixture: FFmpeg binary, nullsrc 64x64, geq Y/U/V=128,
    // SVT-AV1 preset 12. Independent binary decoding yields 6144 samples=128.
    const CONSTANT_OBUS: &[u8] = &[
        0x12, 0x00, 0x0a, 0x0b, 0x02, 0x00, 0x00, 0x05, 0x15, 0x7f, 0xfc, 0x4a, 0xf9, 0x00, 0x40,
        0x32, 0x0c, 0x10, 0x00, 0xac, 0x02, 0x05, 0x14, 0x20, 0x81, 0x00, 0x00, 0x98, 0x80,
    ];

    #[test]
    fn malformed_entropy_tiles_never_panic_or_return_partial_planes() {
        let mut stream = super::super::ObuStream::new();
        let obus = stream.push(CONSTANT_OBUS).unwrap();
        let sequence =
            SequenceHeader::parse(&obus.iter().find(|o| o.kind == 1).unwrap().payload).unwrap();
        let original = obus.iter().find(|o| o.kind == 6).unwrap();
        let mut state = 73u32;
        for len in 0..32 {
            for _ in 0..16 {
                let mut mutated = original.clone();
                mutated.payload.truncate(10);
                for _ in 0..len {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    mutated.payload.push((state >> 24) as u8);
                }
                if let Ok(frame) = decode_intra_frame(&mutated, &sequence) {
                    assert_eq!(frame.planes.len(), 3);
                    for (p, n) in frame.planes.iter().zip([4096, 1024, 1024]) {
                        assert_eq!(p.samples.len(), n);
                        assert!(p.samples.iter().all(|&v| v < 256));
                    }
                }
            }
        }
    }

    #[test]
    fn complete_skipped_dc_frame_pixels() {
        let mut stream = super::super::ObuStream::new();
        let obus = stream.push(CONSTANT_OBUS).unwrap();
        stream.finish().unwrap();
        let s = SequenceHeader::parse(&obus.iter().find(|o| o.kind == 1).unwrap().payload).unwrap();
        let frame_obu = obus.iter().find(|o| o.kind == 6).unwrap();
        let frame = decode_intra_frame(frame_obu, &s).unwrap();
        assert_eq!(
            frame.planes.iter().map(|p| p.samples.len()).sum::<usize>(),
            6144
        );
        assert_eq!((frame.planes[0].width, frame.planes[0].height), (64, 64));
        assert_eq!((frame.planes[1].width, frame.planes[1].height), (32, 32));
        assert!(
            frame
                .planes
                .iter()
                .all(|p| p.samples.iter().all(|&v| v == 128))
        );
        let mut malformed = frame_obu.clone();
        *malformed.payload.last_mut().unwrap() ^= 1;
        assert!(decode_intra_frame(&malformed, &s).is_err());
    }

    #[test]
    fn complete_ten_bit_skipped_dc_frame_pixels() {
        // FFmpeg binary fixture with format=yuv420p10le before geq=512.
        let bytes = [
            0x12, 0x00, 0x0a, 0x0b, 0x02, 0x00, 0x00, 0x05, 0x15, 0x7f, 0xfc, 0x4a, 0xf9, 0x40,
            0x40, 0x32, 0x0c, 0x10, 0x00, 0xad, 0x02, 0x07, 0x1c, 0x30, 0xc1, 0x00, 0x00, 0x98,
            0x80,
        ];
        let mut stream = super::super::ObuStream::new();
        let obus = stream.push(&bytes).unwrap();
        let s = SequenceHeader::parse(&obus.iter().find(|o| o.kind == 1).unwrap().payload).unwrap();
        let frame = decode_intra_frame(obus.iter().find(|o| o.kind == 6).unwrap(), &s).unwrap();
        assert_eq!(frame.bit_depth, 10);
        assert!(
            frame
                .planes
                .iter()
                .all(|p| p.samples.iter().all(|&v| v == 512))
        );
    }

    #[test]
    fn large_target_never_returns_synthetic_pixels() {
        let sequence = SequenceHeader::parse(&[
            0x02, 0x00, 0x00, 0x42, 0x95, 0x5d, 0xfe, 0x1b, 0x8d, 0x5f, 0x32, 0x02, 0x02, 0x02,
            0x48,
        ])
        .unwrap();
        let obu = Obu {
            kind: 6,
            temporal_id: 0,
            spatial_id: 0,
            payload: vec![
                0x10, 0x00, 0x84, 0x00, 0x80, 0x41, 0x00, 0x00, 0x20, 0xbc, 0xf3, 0xcf, 0x80, 0x08,
                0x00, 0x20, 0xbb, 0x32,
            ],
        };
        assert!(decode_intra_frame(&obu, &sequence).is_err());
    }
}
