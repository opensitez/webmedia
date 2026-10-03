//! Bounded VP9 bitstream framing and uncompressed frame headers.

use super::backend::MediaDecodeError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    pub profile: u8,
    pub show_existing_frame: Option<u8>,
    pub key_frame: bool,
    pub show_frame: bool,
    pub error_resilient: bool,
    pub bit_depth: Option<u8>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub render_width: Option<u32>,
    pub render_height: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReferenceFrame {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InterframeHeader {
    pub header: FrameHeader,
    pub intra_only: bool,
    pub reset_frame_context: u8,
    pub refresh_frame_flags: u8,
    pub reference_indices: [u8; 3],
    pub reference_sign_bias: [bool; 3],
    pub allow_high_precision_mv: bool,
    pub interpolation_filter: u8,
    pub header_bits: usize,
}

impl InterframeHeader {
    pub fn parse(
        data: &[u8],
        references: &[Option<ReferenceFrame>; 8],
    ) -> Result<Self, MediaDecodeError> {
        let (mut header, mut bits) = FrameHeader::parse_with_bits(data)?;
        if header.key_frame || header.show_existing_frame.is_some() {
            return Err(invalid("VP9 frame is not an interframe"));
        }
        let intra_only = !header.show_frame && bits.read(1)? != 0;
        let reset_frame_context = if !header.error_resilient { bits.read(2)? as u8 } else { 0 };
        let mut reference_indices = [0; 3];
        let mut reference_sign_bias = [false; 3];
        let mut allow_high_precision_mv = false;
        let mut interpolation_filter = 0;
        let refresh_frame_flags;
        if intra_only {
            if [bits.read(8)?, bits.read(8)?, bits.read(8)?] != [0x49, 0x83, 0x42] {
                return Err(invalid("invalid VP9 intra-only frame sync code"));
            }
            header.bit_depth = Some(if header.profile > 0 {
                read_color_config(&mut bits, header.profile)?
            } else {
                8
            });
            refresh_frame_flags = bits.read(8)? as u8;
            let (width, height) = read_frame_size(&mut bits)?;
            header.width = Some(width);
            header.height = Some(height);
            (header.render_width, header.render_height) = read_render_size(&mut bits, width, height)?;
        } else {
            refresh_frame_flags = bits.read(8)? as u8;
            for index in 0..3 {
                reference_indices[index] = bits.read(3)? as u8;
                reference_sign_bias[index] = bits.read(1)? != 0;
            }
            let mut frame_size = None;
            for &index in &reference_indices {
                if bits.read(1)? != 0 {
                    frame_size = Some(references[index as usize]
                        .ok_or_else(|| invalid("VP9 frame references an unavailable slot"))?);
                    break;
                }
            }
            let frame = if let Some(frame) = frame_size {
                frame
            } else {
                let (width, height) = read_frame_size(&mut bits)?;
                ReferenceFrame {
                    width,
                    height,
                    bit_depth: references[reference_indices[0] as usize]
                        .ok_or_else(|| invalid("VP9 frame references an unavailable slot"))?
                        .bit_depth,
                }
            };
            header.width = Some(frame.width);
            header.height = Some(frame.height);
            header.bit_depth = Some(frame.bit_depth);
            (header.render_width, header.render_height) =
                read_render_size(&mut bits, frame.width, frame.height)?;
            allow_high_precision_mv = bits.read(1)? != 0;
            interpolation_filter = if bits.read(1)? != 0 {
                4 // SWITCHABLE
            } else {
                [1, 0, 2, 3][bits.read(2)? as usize]
            };
        }
        Ok(Self {
            header,
            intra_only,
            reset_frame_context,
            refresh_frame_flags,
            reference_indices,
            reference_sign_bias,
            allow_high_precision_mv,
            interpolation_filter,
            header_bits: bits.position,
        })
    }
}

fn read_frame_size(bits: &mut Bits<'_>) -> Result<(u32, u32), MediaDecodeError> {
    Ok((bits.read(16)? + 1, bits.read(16)? + 1))
}

fn read_render_size(
    bits: &mut Bits<'_>, width: u32, height: u32,
) -> Result<(Option<u32>, Option<u32>), MediaDecodeError> {
    if bits.read(1)? != 0 {
        Ok((Some(bits.read(16)? + 1), Some(bits.read(16)? + 1)))
    } else {
        Ok((Some(width), Some(height)))
    }
}

fn read_color_config(bits: &mut Bits<'_>, profile: u8) -> Result<u8, MediaDecodeError> {
    let bit_depth = if profile >= 2 {
        if bits.read(1)? != 0 { 12 } else { 10 }
    } else {
        8
    };
    let color_space = bits.read(3)?;
    if color_space == 7 {
        if profile != 1 && profile != 3 {
            return Err(invalid("VP9 RGB requires profile 1 or 3"));
        }
        if bits.read(1)? != 0 {
            return Err(invalid("nonzero VP9 RGB reserved bit"));
        }
    } else {
        bits.read(1)?;
        if profile == 1 || profile == 3 {
            bits.read(2)?;
            if bits.read(1)? != 0 {
                return Err(invalid("nonzero VP9 chroma reserved bit"));
            }
        }
    }
    Ok(bit_depth)
}

impl FrameHeader {
    pub fn parse(data: &[u8]) -> Result<Self, MediaDecodeError> {
        Self::parse_with_bits(data).map(|(header, _)| header)
    }

    fn parse_with_bits(data: &[u8]) -> Result<(Self, Bits<'_>), MediaDecodeError> {
        let mut bits = Bits { data, position: 0 };
        if bits.read(2)? != 2 {
            return Err(invalid("invalid VP9 frame marker"));
        }
        let profile = (bits.read(1)? | (bits.read(1)? << 1)) as u8;
        if profile == 3 && bits.read(1)? != 0 {
            return Err(invalid("nonzero VP9 reserved profile bit"));
        }
        if bits.read(1)? != 0 {
            return Ok((Self {
                profile,
                show_existing_frame: Some(bits.read(3)? as u8),
                key_frame: false,
                show_frame: true,
                error_resilient: false,
                bit_depth: None,
                width: None,
                height: None,
                render_width: None,
                render_height: None,
            }, bits));
        }
        let key_frame = bits.read(1)? == 0;
        let show_frame = bits.read(1)? != 0;
        let error_resilient = bits.read(1)? != 0;
        let mut header = Self {
            profile,
            show_existing_frame: None,
            key_frame,
            show_frame,
            error_resilient,
            bit_depth: None,
            width: None,
            height: None,
            render_width: None,
            render_height: None,
        };
        if !key_frame {
            return Ok((header, bits));
        }
        if [bits.read(8)?, bits.read(8)?, bits.read(8)?] != [0x49, 0x83, 0x42] {
            return Err(invalid("invalid VP9 frame sync code"));
        }
        let bit_depth = if profile >= 2 {
            if bits.read(1)? != 0 { 12 } else { 10 }
        } else {
            8
        };
        let color_space = bits.read(3)?;
        if color_space == 7 {
            if profile != 1 && profile != 3 {
                return Err(invalid("VP9 RGB requires profile 1 or 3"));
            }
            if bits.read(1)? != 0 {
                return Err(invalid("nonzero VP9 RGB reserved bit"));
            }
        } else {
            bits.read(1)?; // color range
            if profile == 1 || profile == 3 {
                bits.read(2)?; // chroma subsampling
                if bits.read(1)? != 0 {
                    return Err(invalid("nonzero VP9 chroma reserved bit"));
                }
            }
        }
        let width = bits.read(16)? + 1;
        let height = bits.read(16)? + 1;
        let (render_width, render_height) = if bits.read(1)? != 0 {
            (bits.read(16)? + 1, bits.read(16)? + 1)
        } else {
            (width, height)
        };
        header.bit_depth = Some(bit_depth);
        header.width = Some(width);
        header.height = Some(height);
        header.render_width = Some(render_width);
        header.render_height = Some(render_height);
        Ok((header, bits))
    }
}

pub struct KeyframeLayout<'a> {
    pub header: FrameHeader,
    pub refresh_frame_context: bool,
    pub frame_parallel_decoding: bool,
    pub frame_context_idx: u8,
    pub loop_filter_level: u8,
    pub loop_filter_sharpness: u8,
    pub loop_filter: LoopFilterState,
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_uv_dc: i8,
    pub delta_q_uv_ac: i8,
    pub lossless: bool,
    pub segmentation_enabled: bool,
    pub segmentation_update_map: bool,
    pub segment_tree_probs: [u8; 7],
    pub segment_pred_probs: [u8; 3],
    pub segmentation_temporal_update: bool,
    pub segment_skip: [bool; 8],
    pub segment_alt_l: [Option<i16>; 8],
    pub segment_ref: [Option<u8>; 8],
    pub segmentation_abs_or_delta_update: bool,
    pub segment_alt_q: [Option<i16>; 8],
    pub tile_cols_log2: u8,
    pub tile_rows_log2: u8,
    pub compressed_header: &'a [u8],
    pub tiles: &'a [u8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopFilterState {
    pub delta_enabled: bool,
    pub reference_deltas: [i8; 4],
    pub mode_deltas: [i8; 2],
}

impl Default for LoopFilterState {
    fn default() -> Self {
        Self { delta_enabled: true, reference_deltas: [1, 0, -1, -1], mode_deltas: [0; 2] }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentationState {
    pub tree_probs: [u8; 7],
    pub prediction_probs: [u8; 3],
    pub temporal_update: bool,
    pub skip: [bool; 8],
    pub alt_l: [Option<i16>; 8],
    pub reference: [Option<u8>; 8],
    pub abs_or_delta_update: bool,
    pub alt_q: [Option<i16>; 8],
}

impl Default for SegmentationState {
    fn default() -> Self {
        Self {
            tree_probs: [255; 7],
            prediction_probs: [255; 3],
            temporal_update: false,
            skip: [false; 8],
            alt_l: [None; 8],
            reference: [None; 8],
            abs_or_delta_update: false,
            alt_q: [None; 8],
        }
    }
}

impl<'a> KeyframeLayout<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, MediaDecodeError> {
        let (header, mut bits) = FrameHeader::parse_with_bits(data)?;
        if !header.key_frame || header.show_existing_frame.is_some() {
            return Err(invalid("VP9 packet is not a keyframe"));
        }
        let refresh_frame_context = !header.error_resilient && bits.read(1)? != 0;
        let frame_parallel_decoding = header.error_resilient || bits.read(1)? != 0;
        let frame_context_idx = bits.read(2)? as u8;
        Self::parse_body(data, header, bits, SegmentationState::default(), LoopFilterState::default(),
            refresh_frame_context, frame_parallel_decoding, frame_context_idx)
    }

    pub fn parse_interframe(
        data: &'a [u8],
        references: &[Option<ReferenceFrame>; 8],
        segmentation: SegmentationState,
    ) -> Result<(InterframeHeader, Self, SegmentationState), MediaDecodeError> {
        Self::parse_interframe_with_filter(data, references, segmentation, LoopFilterState::default())
    }

    pub fn parse_interframe_with_filter(
        data: &'a [u8],
        references: &[Option<ReferenceFrame>; 8],
        segmentation: SegmentationState,
        loop_filter: LoopFilterState,
    ) -> Result<(InterframeHeader, Self, SegmentationState), MediaDecodeError> {
        let inter = InterframeHeader::parse(data, references)?;
        let mut bits = Bits { data, position: inter.header_bits };
        let refresh_frame_context = !inter.header.error_resilient && bits.read(1)? != 0;
        let frame_parallel_decoding = inter.header.error_resilient || bits.read(1)? != 0;
        let frame_context_idx = bits.read(2)? as u8;
        let independent = inter.intra_only || inter.header.error_resilient;
        let layout = Self::parse_body(data, inter.header, bits,
            if independent { SegmentationState::default() } else { segmentation },
            if independent { LoopFilterState::default() } else { loop_filter },
            refresh_frame_context, frame_parallel_decoding, frame_context_idx)?;
        let updated = layout.segmentation_state(
            if independent { SegmentationState::default() } else { segmentation });
        Ok((inter, layout, updated))
    }

    pub fn segmentation_state(&self, mut previous: SegmentationState) -> SegmentationState {
        if self.segmentation_enabled {
            previous.tree_probs = self.segment_tree_probs;
            previous.prediction_probs = self.segment_pred_probs;
            previous.temporal_update = self.segmentation_temporal_update;
            previous.skip = self.segment_skip;
            previous.alt_l = self.segment_alt_l;
            previous.reference = self.segment_ref;
            previous.abs_or_delta_update = self.segmentation_abs_or_delta_update;
            previous.alt_q = self.segment_alt_q;
        }
        previous
    }

    fn parse_body(
        data: &'a [u8],
        header: FrameHeader,
        mut bits: Bits<'a>,
        previous: SegmentationState,
        mut loop_filter: LoopFilterState,
        refresh_frame_context: bool,
        frame_parallel_decoding: bool,
        frame_context_idx: u8,
    ) -> Result<Self, MediaDecodeError> {

        let loop_filter_level = bits.read(6)? as u8;
        let loop_filter_sharpness = bits.read(3)? as u8;
        loop_filter.delta_enabled = bits.read(1)? != 0;
        if loop_filter.delta_enabled && bits.read(1)? != 0 {
            for delta in loop_filter.reference_deltas.iter_mut().chain(loop_filter.mode_deltas.iter_mut()) {
                if bits.read(1)? != 0 {
                    let magnitude = bits.read(6)? as i8;
                    *delta = if bits.read(1)? != 0 { -magnitude } else { magnitude };
                }
            }
        }

        let base_q_idx = bits.read(8)? as u8;
        let mut deltas = [0i32; 3];
        for delta in &mut deltas {
            if bits.read(1)? != 0 {
                let value = bits.read(4)? as i32;
                *delta = if bits.read(1)? != 0 { -value } else { value };
            }
        }
        let lossless = base_q_idx == 0 && deltas.iter().all(|delta| *delta == 0);

        let segmentation_enabled = bits.read(1)? != 0;
        let mut segmentation_update_map = false;
        let mut segment_tree_probs = previous.tree_probs;
        let mut segment_pred_probs = previous.prediction_probs;
        let mut segmentation_temporal_update = previous.temporal_update;
        let mut segment_skip = previous.skip;
        let mut segment_alt_l = previous.alt_l;
        let mut segment_ref = previous.reference;
        let mut segmentation_abs_or_delta_update = previous.abs_or_delta_update;
        let mut segment_alt_q = previous.alt_q;
        if segmentation_enabled {
            segmentation_update_map = bits.read(1)? != 0;
            if segmentation_update_map {
                for probability in &mut segment_tree_probs {
                    *probability = read_probability(&mut bits)?;
                }
                segmentation_temporal_update = bits.read(1)? != 0;
                for probability in &mut segment_pred_probs {
                    *probability = if segmentation_temporal_update {
                        read_probability(&mut bits)?
                    } else {
                        255
                    };
                }
            }
            if bits.read(1)? != 0 {
                segmentation_abs_or_delta_update = bits.read(1)? != 0;
                for (segment, skip) in segment_skip.iter_mut().enumerate() {
                    segment_alt_q[segment] = None;
                    segment_alt_l[segment] = None;
                    segment_ref[segment] = None;
                    *skip = false;
                    for (feature, (width, signed)) in [(8, true), (6, true), (2, false), (0, false)].into_iter().enumerate() {
                        if bits.read(1)? != 0 {
                            let value = bits.read(width)? as i16;
                            let value = if signed && bits.read(1)? != 0 { -value } else { value };
                            if feature == 0 {
                                segment_alt_q[segment] = Some(value);
                            }
                            if feature == 1 {
                                segment_alt_l[segment] = Some(value);
                            }
                            if feature == 2 {
                                segment_ref[segment] = Some(value as u8);
                            }
                            if feature == 3 {
                                *skip = true;
                            }
                        }
                    }
                }
            }
        }

        let width = header.width.ok_or_else(|| invalid("missing VP9 keyframe width"))?;
        let sb_cols = width.div_ceil(64);
        let mut min_cols = 0u8;
        while (64u32 << min_cols) < sb_cols {
            min_cols += 1;
        }
        let mut max_cols = 0u8;
        while (sb_cols >> (max_cols + 1)) >= 4 {
            max_cols += 1;
        }
        let mut tile_cols_log2 = min_cols;
        while tile_cols_log2 < max_cols && bits.read(1)? != 0 {
            tile_cols_log2 += 1;
        }
        let mut tile_rows_log2 = bits.read(1)? as u8;
        if tile_rows_log2 != 0 {
            tile_rows_log2 += bits.read(1)? as u8;
        }
        let header_size = bits.read(16)? as usize;
        if header_size == 0 {
            return Err(invalid("empty VP9 compressed header"));
        }
        while bits.position % 8 != 0 {
            if bits.read(1)? != 0 {
                return Err(invalid("nonzero VP9 header padding"));
            }
        }
        let start = bits.position / 8;
        let end = start
            .checked_add(header_size)
            .ok_or_else(|| invalid("VP9 compressed header size overflow"))?;
        if end >= data.len() {
            return Err(invalid("truncated VP9 compressed header or tiles"));
        }
        Ok(Self {
            header,
            refresh_frame_context,
            frame_parallel_decoding,
            frame_context_idx,
            loop_filter_level,
            loop_filter_sharpness,
            loop_filter,
            base_q_idx,
            delta_q_y_dc: deltas[0] as i8,
            delta_q_uv_dc: deltas[1] as i8,
            delta_q_uv_ac: deltas[2] as i8,
            lossless,
            segmentation_enabled,
            segmentation_update_map,
            segment_tree_probs,
            segment_pred_probs,
            segmentation_temporal_update,
            segment_skip,
            segment_alt_l,
            segment_ref,
            segmentation_abs_or_delta_update,
            segment_alt_q,
            tile_cols_log2,
            tile_rows_log2,
            compressed_header: &data[start..end],
            tiles: &data[end..],
        })
    }

    pub fn tile_partitions(&self) -> Result<Vec<&'a [u8]>, MediaDecodeError> {
        let count = (1usize << self.tile_cols_log2) * (1usize << self.tile_rows_log2);
        let mut remaining = self.tiles;
        let mut partitions = Vec::with_capacity(count);
        for index in 0..count {
            let size = if index + 1 == count {
                remaining.len()
            } else {
                if remaining.len() < 4 {
                    return Err(invalid("truncated VP9 tile size"));
                }
                let bytes: [u8; 4] = remaining[..4].try_into().unwrap();
                remaining = &remaining[4..];
                u32::from_be_bytes(bytes) as usize
            };
            if size == 0 || size > remaining.len() {
                return Err(invalid("VP9 tile exceeds frame"));
            }
            partitions.push(&remaining[..size]);
            remaining = &remaining[size..];
        }
        Ok(partitions)
    }
}

fn read_probability(bits: &mut Bits<'_>) -> Result<u8, MediaDecodeError> {
    Ok(if bits.read(1)? != 0 { bits.read(8)? as u8 } else { 255 })
}

pub fn split_superframe(data: &[u8]) -> Result<Vec<&[u8]>, MediaDecodeError> {
    let Some(&marker) = data.last() else {
        return Err(invalid("empty VP9 packet"));
    };
    if marker & 0xe0 != 0xc0 {
        return Ok(vec![data]);
    }
    let frame_count = usize::from(marker & 7) + 1;
    let magnitude = usize::from((marker >> 3) & 3) + 1;
    let index_len = 2 + frame_count * magnitude;
    if data.len() < index_len || data[data.len() - index_len] != marker {
        return Err(invalid("invalid VP9 superframe index"));
    }
    let mut offset = 0usize;
    let mut frames = Vec::with_capacity(frame_count);
    let index = &data[data.len() - index_len + 1..];
    for entry in index.chunks_exact(magnitude).take(frame_count) {
        let size = entry
            .iter()
            .rev()
            .fold(0usize, |size, byte| (size << 8) | usize::from(*byte));
        let end = offset
            .checked_add(size)
            .ok_or_else(|| invalid("VP9 superframe size overflow"))?;
        if size == 0 || end > data.len() - index_len {
            return Err(invalid("VP9 superframe exceeds packet"));
        }
        frames.push(&data[offset..end]);
        offset = end;
    }
    if offset != data.len() - index_len {
        return Err(invalid("VP9 superframe sizes do not match packet"));
    }
    Ok(frames)
}

struct Bits<'a> {
    data: &'a [u8],
    position: usize,
}

impl Bits<'_> {
    fn read(&mut self, count: usize) -> Result<u32, MediaDecodeError> {
        if count > 16 || self.position.saturating_add(count) > self.data.len() * 8 {
            return Err(invalid("truncated VP9 frame header"));
        }
        let mut value = 0u32;
        for _ in 0..count {
            let byte = self.data[self.position / 8];
            value = (value << 1) | u32::from((byte >> (7 - self.position % 8)) & 1);
            self.position += 1;
        }
        Ok(value)
    }
}

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::vp9_compressed::CompressedHeader;
    use crate::video::webm::{WebmVideoCodec, WebmVideoStream};

    #[test]
    fn splits_superframe_and_rejects_corrupt_index() {
        let packet = [1, 2, 3, 4, 0xc1, 2, 2, 0xc1];
        assert_eq!(split_superframe(&packet).unwrap(), vec![&packet[..2], &packet[2..4]]);
        assert!(split_superframe(&[1, 2, 3, 4, 0xc0, 2, 2, 0xc1]).is_err());
        assert_eq!(split_superframe(&[0x82, 0, 0]).unwrap(), vec![&[0x82, 0, 0][..]]);
    }

    #[test]
    fn parses_profile_zero_keyframe_dimensions() {
        let mut bits = Vec::new();
        let mut append = |value: u32, count: usize| {
            for shift in (0..count).rev() {
                bits.push(((value >> shift) & 1) as u8);
            }
        };
        append(2, 2); // frame marker
        append(0, 2); // profile 0
        append(0, 1); // not show-existing
        append(0, 1); // keyframe
        append(1, 1); // show frame
        append(0, 1); // error resilient
        for byte in [0x49, 0x83, 0x42] {
            append(byte, 8);
        }
        append(1, 3); // BT.601
        append(0, 1); // limited range
        append(639, 16);
        append(359, 16);
        append(0, 1); // render size equals frame size
        let bytes: Vec<u8> = bits
            .chunks(8)
            .map(|chunk| chunk.iter().fold(0u8, |value, bit| (value << 1) | bit) << (8 - chunk.len()))
            .collect();
        let header = FrameHeader::parse(&bytes).unwrap();
        assert!(header.key_frame && header.show_frame);
        assert_eq!((header.width, header.height), (Some(640), Some(360)));
        assert_eq!(header.bit_depth, Some(8));
        assert!(FrameHeader::parse(&bytes[..4]).is_err());
    }

    #[test]
    fn short_and_malformed_packets_do_not_panic() {
        for first in 0u8..=255 {
            for second in 0u8..=255 {
                let bytes = [first, second];
                let _ = FrameHeader::parse(&bytes);
                let _ = split_superframe(&bytes);
            }
        }
    }

    #[test]
    fn parses_interframe_references_in_supplied_vp9_clip() {
        let Ok(sample) = std::env::var("WEBMEDIA_VP9_SAMPLE") else { return };
        let mut source = std::fs::File::open(sample).unwrap();
        let mut stream = WebmVideoStream::for_codec(WebmVideoCodec::Vp9);
        let mut buffer = [0u8; 16 * 1024];
        let mut references = [None; 8];
        let mut segmentation = SegmentationState::default();
        let mut probabilities = None;
        let mut keyframes = 0usize;
        let mut interframes = 0usize;
        let mut shown_existing = 0usize;
        loop {
            let count = std::io::Read::read(&mut source, &mut buffer).unwrap();
            if count == 0 {
                break;
            }
            for packet in stream.push(&buffer[..count]).unwrap() {
                for data in split_superframe(&packet.data).unwrap() {
                    let header = FrameHeader::parse(data).unwrap();
                    if let Some(index) = header.show_existing_frame {
                        assert!(references[index as usize].is_some());
                        shown_existing += 1;
                    } else if header.key_frame {
                        let layout = KeyframeLayout::parse(data).unwrap();
                        if std::env::var_os("WEBMEDIA_VP9_REPORT").is_some() {
                            eprintln!("VP9 keyframe segmentation: enabled={}, update_map={}, temporal={}, alt_q={:?}",
                                layout.segmentation_enabled, layout.segmentation_update_map,
                                layout.segmentation_temporal_update, layout.segment_alt_q);
                        }
                        assert!(!layout.tile_partitions().unwrap().is_empty());
                        probabilities = Some(CompressedHeader::parse_keyframe(&layout).unwrap());
                        segmentation = layout.segmentation_state(SegmentationState::default());
                        let frame = ReferenceFrame {
                            width: header.width.unwrap(),
                            height: header.height.unwrap(),
                            bit_depth: header.bit_depth.unwrap(),
                        };
                        references.fill(Some(frame));
                        keyframes += 1;
                    } else {
                        let (inter, layout, updated_segmentation) =
                            KeyframeLayout::parse_interframe(data, &references, segmentation)
                                .unwrap_or_else(|error| panic!("VP9 interframe {interframes}: {error:?}"));
                        let mut compressed = super::super::vp8::BoolDecoder::new(layout.compressed_header).unwrap();
                        assert!(!compressed.read_bit().unwrap(), "interframe {interframes} compressed marker");
                        for tile in layout.tile_partitions().unwrap() {
                            let mut reader = super::super::vp8::BoolDecoder::new(tile).unwrap();
                            assert!(!reader.read_bit().unwrap(), "interframe {interframes} tile marker");
                        }
                        let decoded_probabilities = CompressedHeader::parse_interframe(
                            &layout, &inter, probabilities.as_ref().unwrap(),
                        ).unwrap_or_else(|error| panic!("VP9 interframe {interframes} compressed header: {error:?}"));
                        if interframes == 0 && std::env::var_os("WEBMEDIA_VP9_REPORT").is_some() {
                            eprintln!("VP9 first interframe: {}x{}, q={}, tx_mode={}, ref_mode={}, filter={}, high_precision_mv={}, segmentation={}, tiles={}x{}, updates={}",
                                layout.header.width.unwrap(), layout.header.height.unwrap(),
                                layout.base_q_idx, decoded_probabilities.tx_mode,
                                decoded_probabilities.reference_mode, inter.interpolation_filter,
                                inter.allow_high_precision_mv, layout.segmentation_enabled,
                                1usize << layout.tile_cols_log2, 1usize << layout.tile_rows_log2,
                                decoded_probabilities.probability_updates);
                            eprintln!("VP9 segmentation: update_map={}, temporal={}, abs={}, alt_q={:?}, skip={:?}",
                                layout.segmentation_update_map, layout.segmentation_temporal_update,
                                layout.segmentation_abs_or_delta_update, layout.segment_alt_q,
                                layout.segment_skip);
                            if !layout.segmentation_enabled && !inter.intra_only {
                            let partition = super::super::vp9_tile::first_inter_partition(
                                layout.tile_partitions().unwrap()[0], &decoded_probabilities,
                            ).unwrap();
                            let block = super::super::vp9_tile::first_inter_block_mode(
                                &layout, &inter, &decoded_probabilities,
                                layout.tile_partitions().unwrap()[0],
                            ).unwrap();
                            eprintln!("VP9 first interframe first partition: {partition:?}");
                            eprintln!("VP9 first interframe first block: {block:?}");
                            }
                        }
                        probabilities = Some(decoded_probabilities);
                        segmentation = updated_segmentation;
                        let frame = ReferenceFrame {
                            width: inter.header.width.unwrap(),
                            height: inter.header.height.unwrap(),
                            bit_depth: inter.header.bit_depth.unwrap(),
                        };
                        for (index, reference) in references.iter_mut().enumerate() {
                            if inter.refresh_frame_flags & (1 << index) != 0 {
                                *reference = Some(frame);
                            }
                        }
                        interframes += 1;
                    }
                }
            }
        }
        stream.finish().unwrap();
        assert!(keyframes > 0 && interframes > 0);
        eprintln!("VP9 sample headers: keyframes={keyframes}, interframes={interframes}, shown_existing={shown_existing}");
    }
}
