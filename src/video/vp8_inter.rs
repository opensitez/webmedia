//! Persistent VP8 interframe header and entropy state (RFC 6386, sections 9 and 17).

use super::backend::MediaDecodeError;
use super::vp8::{read_bmode, BoolDecoder, FrameHeader, KeyFrameLayout};
use super::vp8_coeff::CoeffProbs;

const MV_UPDATE_PROBS: [[u8; 19]; 2] = [
    [237, 246, 253, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 250, 250, 252, 254, 254],
    [231, 243, 245, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 251, 251, 254, 254, 254],
];
const DEFAULT_MV_PROBS: [[u8; 19]; 2] = [
    [162, 128, 225, 146, 172, 147, 214, 39, 156, 128, 129, 132, 75, 145, 178, 206, 239, 254, 254],
    [164, 128, 204, 170, 119, 235, 140, 230, 228, 128, 130, 130, 74, 148, 180, 203, 236, 254, 254],
];
const MODE_CONTEXTS: [[u8; 4]; 6] = [
    [7, 1, 1, 143], [14, 18, 14, 107], [135, 64, 57, 68],
    [60, 56, 128, 65], [159, 134, 128, 34], [234, 188, 128, 28],
];
const SUBMODE_PROBS: [[u8; 3]; 5] = [
    [147, 136, 18], [106, 145, 1], [179, 121, 1], [223, 1, 34], [208, 1, 1],
];
const BMODE_PROBS: [u8; 9] = [120, 90, 79, 133, 87, 85, 80, 111, 151];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct MotionVector {
    pub(super) row: i16,
    pub(super) col: i16,
}

impl MotionVector {
    fn add(self, other: Self) -> Self {
        Self { row: self.row + other.row, col: self.col + other.col }
    }

    fn negated(self) -> Self {
        Self { row: -self.row, col: -self.col }
    }
}

#[derive(Clone, Debug)]
pub(super) struct InterMacroblock {
    pub(super) segment: u8,
    pub(super) skip_coefficients: bool,
    pub(super) reference: u8,
    pub(super) mode: u8,
    pub(super) luma: u8,
    pub(super) chroma: u8,
    pub(super) subblocks: [u8; 16],
    pub(super) motion: [MotionVector; 16],
}

impl Default for InterMacroblock {
    fn default() -> Self {
        Self {
            segment: 0, skip_coefficients: false, reference: 0, mode: 0,
            luma: 0, chroma: 0, subblocks: [0; 16],
            motion: [MotionVector::default(); 16],
        }
    }
}

#[derive(Clone)]
pub(super) struct InterState {
    pub(super) coeff_probs: CoeffProbs,
    pub(super) mv_probs: [[u8; 19]; 2],
    pub(super) ymode_probs: [u8; 4],
    pub(super) uv_mode_probs: [u8; 3],
    pub(super) segment_quantizers: [i16; 4],
    pub(super) segment_filter_levels: [i16; 4],
    pub(super) segment_absolute: bool,
    pub(super) segment_map: Vec<u8>,
    pub(super) mb_width: usize,
    pub(super) mb_height: usize,
    pub(super) reference_filter_deltas: [i16; 4],
    pub(super) mode_filter_deltas: [i16; 4],
}

impl InterState {
    pub(super) fn after_keyframe(layout: &KeyFrameLayout<'_>) -> Self {
        let (segment_quantizers, segment_filter_levels, segment_absolute) = layout.segment_features();
        let (reference_filter_deltas, mode_filter_deltas) = layout.filter_deltas();
        Self {
            coeff_probs: if layout.refresh_entropy_probs {
                layout.coeff_probabilities()
            } else {
                CoeffProbs::default()
            },
            mv_probs: DEFAULT_MV_PROBS,
            ymode_probs: [112, 86, 140, 37],
            uv_mode_probs: [162, 101, 204],
            segment_quantizers,
            segment_filter_levels,
            segment_absolute,
            segment_map: vec![0; layout.macroblocks_wide() * layout.macroblocks_high()],
            mb_width: layout.macroblocks_wide(),
            mb_height: layout.macroblocks_high(),
            reference_filter_deltas,
            mode_filter_deltas,
        }
    }
}

pub(super) struct InterFrameLayout<'a> {
    pub(super) control: BoolDecoder<'a>,
    pub(super) token_partitions: Vec<&'a [u8]>,
    pub(super) coeff_probs: CoeffProbs,
    pub(super) mv_probs: [[u8; 19]; 2],
    pub(super) ymode_probs: [u8; 4],
    pub(super) uv_mode_probs: [u8; 3],
    pub(super) quantizer: u8,
    pub(super) quant_deltas: [i8; 5],
    pub(super) segment_quantizers: [i16; 4],
    pub(super) segment_filter_levels: [i16; 4],
    pub(super) segment_absolute: bool,
    pub(super) mb_width: usize,
    pub(super) version: u8,
    pub(super) reference_filter_deltas: [i16; 4],
    pub(super) mode_filter_deltas: [i16; 4],
    pub(super) simple_filter: bool,
    pub(super) filter_level: u8,
    pub(super) sharpness: u8,
    pub(super) filter_adjustments: bool,
    pub(super) segment_enabled: bool,
    pub(super) segment_map_update: bool,
    pub(super) segment_probs: [u8; 3],
    pub(super) refresh_golden: bool,
    pub(super) refresh_alternate: bool,
    pub(super) copy_to_golden: u8,
    pub(super) copy_to_alternate: u8,
    pub(super) sign_bias: [bool; 2],
    pub(super) refresh_last: bool,
    pub(super) mb_no_skip_coeff: bool,
    pub(super) prob_skip_false: u8,
    pub(super) prob_intra: u8,
    pub(super) prob_last: u8,
    pub(super) prob_gf: u8,
}

impl<'a> InterFrameLayout<'a> {
    pub(super) fn parse(frame: &'a [u8], state: &mut InterState) -> Result<Self, MediaDecodeError> {
        let header = FrameHeader::parse(frame)?;
        if header.key_frame {
            return Err(MediaDecodeError::InvalidData("expected VP8 interframe".into()));
        }
        let mut control = BoolDecoder::new(header.control_partition(frame))?;
        let segment_enabled = control.read_bit()?;
        let mut segment_map_update = false;
        let mut segment_probs = [255; 3];
        if segment_enabled {
            segment_map_update = control.read_bit()?;
            if control.read_bit()? {
                state.segment_absolute = control.read_bit()?;
                for (bits, values) in [
                    (7, &mut state.segment_quantizers),
                    (6, &mut state.segment_filter_levels),
                ] {
                    for value in values {
                        if control.read_bit()? {
                            let magnitude = control.read_literal(bits)? as i16;
                            *value = if control.read_bit()? { -magnitude } else { magnitude };
                        }
                    }
                }
            }
            if segment_map_update {
                for probability in &mut segment_probs {
                    if control.read_bit()? {
                        *probability = control.read_literal(8)? as u8;
                    }
                }
            }
        }
        let simple_filter = control.read_bit()?;
        let filter_level = control.read_literal(6)? as u8;
        let sharpness = control.read_literal(3)? as u8;
        let filter_adjustments = control.read_bit()?;
        if filter_adjustments && control.read_bit()? {
            for values in [&mut state.reference_filter_deltas, &mut state.mode_filter_deltas] {
                for value in values {
                    if control.read_bit()? {
                        let magnitude = control.read_literal(6)? as i16;
                        *value = if control.read_bit()? { -magnitude } else { magnitude };
                    }
                }
            }
        }
        let partition_count = 1usize << control.read_literal(2)?;
        let quantizer = control.read_literal(7)? as u8;
        let mut quant_deltas = [0; 5];
        for value in &mut quant_deltas {
            if control.read_bit()? {
                let magnitude = control.read_literal(4)? as i8;
                *value = if control.read_bit()? { -magnitude } else { magnitude };
            }
        }
        let refresh_golden = control.read_bit()?;
        let refresh_alternate = control.read_bit()?;
        let copy_to_golden = if refresh_golden { 0 } else { control.read_literal(2)? as u8 };
        let copy_to_alternate = if refresh_alternate { 0 } else { control.read_literal(2)? as u8 };
        let sign_bias = [control.read_bit()?, control.read_bit()?];
        let refresh_entropy = control.read_bit()?;
        let refresh_last = control.read_bit()?;
        let previous_entropy = if refresh_entropy { None } else { Some((state.coeff_probs.clone(), state.mv_probs, state.ymode_probs, state.uv_mode_probs)) };
        state.coeff_probs.update(&mut control)?;
        let mb_no_skip_coeff = control.read_bit()?;
        let prob_skip_false = if mb_no_skip_coeff { control.read_literal(8)? as u8 } else { 0 };
        let prob_intra = control.read_literal(8)? as u8;
        let prob_last = control.read_literal(8)? as u8;
        let prob_gf = control.read_literal(8)? as u8;
        if control.read_bit()? {
            for probability in &mut state.ymode_probs {
                *probability = control.read_literal(8)? as u8;
            }
        }
        if control.read_bit()? {
            for probability in &mut state.uv_mode_probs {
                *probability = control.read_literal(8)? as u8;
            }
        }
        for (component, updates) in MV_UPDATE_PROBS.iter().enumerate() {
            for (index, probability) in updates.iter().enumerate() {
                if control.read(*probability)? {
                    let value = control.read_literal(7)? as u8;
                    state.mv_probs[component][index] = if value == 0 { 1 } else { value * 2 };
                }
            }
        }
        let token_partitions = split_partitions(frame, &header, partition_count)?;
        let coeff_probs = state.coeff_probs.clone();
        let mv_probs = state.mv_probs;
        let ymode_probs = state.ymode_probs;
        let uv_mode_probs = state.uv_mode_probs;
        // Entropy changes in a non-refreshing frame apply to this frame only.
        if let Some((coeff, mv, ymode, uv)) = previous_entropy {
            state.coeff_probs = coeff;
            state.mv_probs = mv;
            state.ymode_probs = ymode;
            state.uv_mode_probs = uv;
        }
        Ok(Self {
            control, token_partitions, coeff_probs, mv_probs, ymode_probs, uv_mode_probs,
            quantizer, quant_deltas,
            segment_quantizers: state.segment_quantizers,
            segment_filter_levels: state.segment_filter_levels,
            segment_absolute: state.segment_absolute,
            mb_width: state.mb_width,
            version: header.version,
            reference_filter_deltas: state.reference_filter_deltas,
            mode_filter_deltas: state.mode_filter_deltas,
            simple_filter,
            filter_level, sharpness, filter_adjustments, segment_enabled,
            segment_map_update, segment_probs, refresh_golden, refresh_alternate,
            copy_to_golden, copy_to_alternate, sign_bias, refresh_last,
            mb_no_skip_coeff, prob_skip_false, prob_intra, prob_last, prob_gf,
        })
    }

    pub(super) fn dequant_factors(&self, segment: u8) -> [i32; 6] {
        let base = if self.segment_absolute {
            self.segment_quantizers[segment as usize]
        } else {
            i16::from(self.quantizer) + self.segment_quantizers[segment as usize]
        };
        super::vp8_quant::factors(base, self.quant_deltas)
    }

    pub(super) fn macroblock_filter_level(&self, mb: &InterMacroblock) -> u8 {
        let level = if self.segment_absolute {
            self.segment_filter_levels[mb.segment as usize]
        } else {
            i16::from(self.filter_level) + self.segment_filter_levels[mb.segment as usize]
        };
        let level = if self.filter_adjustments {
            let mode = if mb.reference == 0 {
                usize::from(mb.mode == 4)
            } else if mb.mode == 7 { 1 } else if mb.mode == 9 { 3 } else { 2 };
            level + self.reference_filter_deltas[mb.reference as usize]
                + self.mode_filter_deltas[mode]
        } else {
            level
        };
        level.clamp(0, 63) as u8
    }

    pub(super) fn read_macroblocks(
        &mut self,
        state: &mut InterState,
        mb_width: usize,
        mb_height: usize,
    ) -> Result<Vec<InterMacroblock>, MediaDecodeError> {
        let mut result = Vec::with_capacity(mb_width * mb_height);
        if state.segment_map.len() != mb_width * mb_height {
            state.segment_map.resize(mb_width * mb_height, 0);
        }
        for y in 0..mb_height {
            for x in 0..mb_width {
                let index = y * mb_width + x;
                let mut mb = InterMacroblock::default();
                if self.segment_enabled {
                    if self.segment_map_update {
                        mb.segment = read_segment(&mut self.control, &self.segment_probs)?;
                        state.segment_map[index] = mb.segment;
                    } else {
                        mb.segment = state.segment_map[index];
                    }
                }
                mb.skip_coefficients = self.mb_no_skip_coeff && self.control.read(self.prob_skip_false)?;
                if !self.control.read(self.prob_intra)? {
                    read_intra_mode(&mut self.control, &self.ymode_probs, &self.uv_mode_probs, &mut mb)?;
                } else {
                    mb.reference = if !self.control.read(self.prob_last)? {
                        1
                    } else if !self.control.read(self.prob_gf)? {
                        2
                    } else {
                        3
                    };
                    let (nearest, near, best, counts) = near_vectors(&result, x, y, mb_width, mb.reference, self.sign_bias);
                    let probs = [
                        MODE_CONTEXTS[counts[0].min(5)][0],
                        MODE_CONTEXTS[counts[1].min(5)][1],
                        MODE_CONTEXTS[counts[2].min(5)][2],
                        MODE_CONTEXTS[counts[3].min(5)][3],
                    ];
                    mb.mode = read_inter_mode(&mut self.control, probs)?;
                    mb.luma = mb.mode;
                    let vector = match mb.mode {
                        5 => nearest,
                        6 => near,
                        7 => MotionVector::default(),
                        8 => best.add(read_motion_vector(&mut self.control, &self.mv_probs)?),
                        9 => {
                            read_split_vectors(&mut self.control, &self.mv_probs, &result, x, y, mb_width, best, &mut mb)?;
                            MotionVector::default()
                        }
                        _ => unreachable!(),
                    };
                    if mb.mode != 9 {
                        mb.motion.fill(vector);
                    }
                }
                result.push(mb);
            }
        }
        Ok(result)
    }
}

fn read_segment(control: &mut BoolDecoder<'_>, probs: &[u8; 3]) -> Result<u8, MediaDecodeError> {
    Ok(if control.read(probs[0])? {
        if control.read(probs[2])? { 3 } else { 2 }
    } else if control.read(probs[1])? { 1 } else { 0 })
}

fn read_intra_mode(
    control: &mut BoolDecoder<'_>, y_probs: &[u8; 4], uv_probs: &[u8; 3],
    mb: &mut InterMacroblock,
) -> Result<(), MediaDecodeError> {
    mb.luma = if !control.read(y_probs[0])? {
        0
    } else if !control.read(y_probs[1])? {
        if control.read(y_probs[2])? { 2 } else { 1 }
    } else if control.read(y_probs[3])? { 4 } else { 3 };
    mb.mode = mb.luma;
    if mb.luma == 4 {
        for mode in &mut mb.subblocks {
            *mode = read_bmode(control, &BMODE_PROBS)?;
        }
    }
    mb.chroma = if !control.read(uv_probs[0])? {
        0
    } else if !control.read(uv_probs[1])? {
        1
    } else if !control.read(uv_probs[2])? {
        2
    } else { 3 };
    Ok(())
}

fn read_inter_mode(control: &mut BoolDecoder<'_>, probs: [u8; 4]) -> Result<u8, MediaDecodeError> {
    Ok(if !control.read(probs[0])? { 7 }
    else if !control.read(probs[1])? { 5 }
    else if !control.read(probs[2])? { 6 }
    else if !control.read(probs[3])? { 8 } else { 9 })
}

fn near_vectors(
    decoded: &[InterMacroblock], x: usize, y: usize, width: usize,
    reference: u8, sign_bias: [bool; 2],
) -> (MotionVector, MotionVector, MotionVector, [usize; 4]) {
    let mut vectors = [MotionVector::default(); 3];
    let mut scores = [0usize; 3];
    let mut split_score = 0;
    for (location, weight) in [
        (if y > 0 { Some((x, y - 1)) } else { None }, 2),
        (if x > 0 { Some((x - 1, y)) } else { None }, 2),
        (if x > 0 && y > 0 { Some((x - 1, y - 1)) } else { None }, 1),
    ] {
        let Some((nx, ny)) = location else { continue };
        let neighbor = &decoded[ny * width + nx];
        if neighbor.reference == 0 { continue; }
        let mut vector = neighbor.motion[0];
        if neighbor.reference >= 2 && reference >= 2
            && sign_bias[(neighbor.reference - 2) as usize] != sign_bias[(reference - 2) as usize]
        {
            vector = vector.negated();
        }
        if let Some(index) = vectors.iter().position(|known| *known == vector) {
            scores[index] += weight;
        } else if let Some(index) = scores.iter().position(|score| *score == 0) {
            vectors[index] = vector;
            scores[index] = weight;
        }
        if neighbor.mode == 9 { split_score += weight; }
    }
    let mut nonzero = (0..3).filter(|&index| scores[index] != 0 && vectors[index] != MotionVector::default()).collect::<Vec<_>>();
    nonzero.sort_by_key(|&index| std::cmp::Reverse(scores[index]));
    let nearest = nonzero.first().map(|&index| vectors[index]).unwrap_or_default();
    let near = nonzero.get(1).map(|&index| vectors[index]).unwrap_or_default();
    let nearest_score = nonzero.first().map(|&index| scores[index]).unwrap_or(0);
    let near_score = nonzero.get(1).map(|&index| scores[index]).unwrap_or(0);
    let zero_score = (0..3).filter(|&index| vectors[index] == MotionVector::default()).map(|index| scores[index]).sum();
    let best = if nearest_score >= zero_score { nearest } else { MotionVector::default() };
    (nearest, near, best, [zero_score, nearest_score, near_score, split_score])
}

fn read_motion_vector(control: &mut BoolDecoder<'_>, probs: &[[u8; 19]; 2]) -> Result<MotionVector, MediaDecodeError> {
    Ok(MotionVector {
        row: read_motion_component(control, &probs[0])?,
        col: read_motion_component(control, &probs[1])?,
    })
}

fn read_motion_component(control: &mut BoolDecoder<'_>, p: &[u8; 19]) -> Result<i16, MediaDecodeError> {
    let magnitude = if control.read(p[0])? {
        let mut value = 0;
        for bit in 0..3 { value |= i16::from(control.read(p[9 + bit])?) << bit; }
        for bit in (4..=9).rev() { value |= i16::from(control.read(p[9 + bit])?) << bit; }
        if value < 16 || control.read(p[12])? { value |= 8; }
        value
    } else {
        let mut value = 0;
        if control.read(p[2])? { value |= 4; }
        if control.read(p[3 + usize::from(value >= 4) * 3])? { value |= 2; }
        if control.read(p[4 + usize::from(value >= 4) * 3 + usize::from(value & 2 != 0)])? { value |= 1; }
        value
    };
    Ok(if magnitude != 0 && control.read(p[1])? { -magnitude } else { magnitude })
}

fn read_split_vectors(
    control: &mut BoolDecoder<'_>, probs: &[[u8; 19]; 2],
    decoded: &[InterMacroblock], x: usize, y: usize, width: usize,
    best: MotionVector, mb: &mut InterMacroblock,
) -> Result<(), MediaDecodeError> {
    let partition = if !control.read(110)? { 3 }
        else if !control.read(111)? { 2 }
        else if !control.read(150)? { 0 } else { 1 };
    let pieces = match partition { 0 | 1 => 2, 2 => 4, _ => 16 };
    for piece in 0..pieces {
        let (row, col) = match partition {
            0 => (piece * 2, 0), 1 => (0, piece * 2),
            2 => ((piece / 2) * 2, (piece % 2) * 2),
            _ => (piece / 4, piece % 4),
        };
        let left = if col > 0 { mb.motion[row * 4 + col - 1] }
            else if x > 0 { decoded[y * width + x - 1].motion[row * 4 + 3] }
            else { MotionVector::default() };
        let above = if row > 0 { mb.motion[(row - 1) * 4 + col] }
            else if y > 0 { decoded[(y - 1) * width + x].motion[12 + col] }
            else { MotionVector::default() };
        let context = if left == above {
            if left == MotionVector::default() { 4 } else { 3 }
        } else if above == MotionVector::default() { 2 }
        else if left == MotionVector::default() { 1 } else { 0 };
        let p = SUBMODE_PROBS[context];
        let vector = if !control.read(p[0])? { left }
            else if !control.read(p[1])? { above }
            else if !control.read(p[2])? { MotionVector::default() }
            else { best.add(read_motion_vector(control, probs)?) };
        for sub_row in 0..4 {
            for sub_col in 0..4 {
                let included = match partition {
                    0 => sub_row / 2 == piece,
                    1 => sub_col / 2 == piece,
                    2 => (sub_row / 2) * 2 + sub_col / 2 == piece,
                    _ => sub_row * 4 + sub_col == piece,
                };
                if included { mb.motion[sub_row * 4 + sub_col] = vector; }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::webm::WebmVp8Stream;

    #[test]
    fn parses_every_interframe_header_in_supplied_webm() {
        let Ok(path) = std::env::var("WEBMEDIA_WEBM_SAMPLE") else {
            return;
        };
        let data = std::fs::read(path).unwrap();
        let mut stream = WebmVp8Stream::new();
        let mut packets = Vec::new();
        for chunk in data.chunks(4096) {
            packets.extend(stream.push(chunk).unwrap());
        }
        let mut state = None;
        let mut interframes = 0;
        let (mb_width, mb_height) = {
            let metadata = stream.metadata().unwrap();
            (metadata.width.unwrap().div_ceil(16) as usize, metadata.height.unwrap().div_ceil(16) as usize)
        };
        let mut decoded_modes = 0;
        for packet in &packets {
            if packet.key_frame {
                let layout = KeyFrameLayout::parse(&packet.data).unwrap();
                state = Some(InterState::after_keyframe(&layout));
            } else {
                let mut layout = InterFrameLayout::parse(&packet.data, state.as_mut().unwrap())
                    .unwrap_or_else(|error| panic!("interframe {interframes}: {error:?}"));
                assert!(!layout.token_partitions.is_empty());
                assert!(layout.token_partitions.iter().all(|part| !part.is_empty()));
                if interframes < 12 {
                    let modes = layout.read_macroblocks(state.as_mut().unwrap(), mb_width, mb_height)
                        .unwrap_or_else(|error| panic!("interframe {interframes} modes: {error:?}"));
                    assert_eq!(modes.len(), mb_width * mb_height);
                    assert!(modes.iter().any(|mb| mb.reference != 0));
                    let mut residue = super::super::vp8_residue::ResidueDecoder::new(&layout).unwrap();
                    for (index, mb) in modes.iter().enumerate() {
                        residue.decode(&layout, mb, index % mb_width, index / mb_width)
                            .unwrap_or_else(|error| panic!("interframe {interframes} residue {index}: {error:?}"));
                    }
                    decoded_modes += 1;
                }
                interframes += 1;
            }
        }
        assert!(interframes > 0);
        assert!(decoded_modes > 0);
    }
}

fn split_partitions<'a>(frame: &'a [u8], header: &FrameHeader, count: usize) -> Result<Vec<&'a [u8]>, MediaDecodeError> {
    let mut pos = 3 + header.first_partition_size;
    let table_end = pos + (count - 1) * 3;
    if table_end > frame.len() {
        return Err(MediaDecodeError::InvalidData("truncated VP8 partition sizes".into()));
    }
    let table = &frame[pos..table_end];
    pos = table_end;
    let mut result = Vec::with_capacity(count);
    for entry in table.chunks_exact(3) {
        let length = usize::from(entry[0]) | usize::from(entry[1]) << 8 | usize::from(entry[2]) << 16;
        let end = pos.checked_add(length).filter(|end| *end <= frame.len())
            .ok_or_else(|| MediaDecodeError::InvalidData("VP8 token partition exceeds frame".into()))?;
        result.push(&frame[pos..end]);
        pos = end;
    }
    result.push(&frame[pos..]);
    Ok(result)
}
