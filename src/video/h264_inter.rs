//! 2003 AVC inter-picture sample prediction for progressive 8-bit 4:2:0.

use super::h264::{
    AvcError, PictureParameters2005, PredictionWeightTable, RefPicReorder, SequenceParameters,
    parse_cabac_inter_slice, type0_pic_order_count,
};
use super::h264_cabac::{
    CabacDecoder, ChromaAcContexts, ChromaDcContexts, CodedBlockContexts, CodedBlockPattern,
    InterMbContexts, IntraPredCode, IntraPredContexts, Luma4x4Contexts, Luma8x8Contexts,
    Luma16x16AcContexts, Luma16x16DcContexts, MbQpContexts, MotionVectorContexts,
    ReferenceIndexContexts, Transform8x8Contexts, intra4x4_modes, intra8x8_modes,
};
use super::h264_deblock::{DeblockMb, filter_inter_picture};
use super::h264_high::{MotionCell, Yuv420Picture, left_edge, upper_edge};
use super::h264_intra::{
    reconstruct_chroma, reconstruct_intra4x4_luma, reconstruct_intra8x8_luma,
    reconstruct_intra16x16_luma,
};
use super::h264_transform::{
    chroma_qp, inverse_4x4_frame_scan, inverse_4x4_residual, inverse_8x8_frame_scan,
    inverse_8x8_residual, inverse_chroma_dc,
};

#[derive(Clone, Copy)]
struct InterMbState {
    qp: i32,
    skipped: bool,
    intra16: bool,
    intra_nxn: bool,
    transform8x8: bool,
    modes4: [u8; 16],
    modes8: [u8; 4],
    motion4: [[i32; 2]; 16],
    mvd4: [[i32; 2]; 16],
    refs4: [u8; 16],
    coded: CodedBlockPattern,
    coded_luma4: u16,
    chroma_mode: u8,
    luma_dc_coded: bool,
    luma_ac_right: [bool; 4],
    luma_ac_bottom: [bool; 4],
    chroma_dc_coded: [bool; 2],
    chroma_ac_right: [[bool; 2]; 2],
    chroma_ac_bottom: [[bool; 2]; 2],
}

enum InterLumaResidual {
    None,
    FourByFour([[i32; 16]; 16]),
    EightByEight([[i32; 64]; 4]),
}

#[derive(Clone, Copy)]
struct PSubPartitions {
    modes: [u8; 4],
    vectors: [[i32; 2]; 16],
    mvd: [[i32; 2]; 16],
    references: [u8; 16],
}

fn p_sub_partition_rect(mode: u8, sub: usize) -> (usize, usize, usize, usize) {
    match mode {
        0 => (0, 0, 2, 2),
        1 => (0, sub, 2, 1),
        2 => (sub, 0, 1, 2),
        3 => (sub % 2, sub / 2, 1, 1),
        _ => unreachable!(),
    }
}

fn b_sub_partition_mode(subtype: u8) -> Result<(u8, u8), AvcError> {
    let value = match subtype {
        0 => (0, 0),
        1 => (0, 1),
        2 => (0, 2),
        3 => (0, 3),
        4 => (1, 1),
        5 => (2, 1),
        6 => (1, 2),
        7 => (2, 2),
        8 => (1, 3),
        9 => (2, 3),
        10 => (3, 1),
        11 => (3, 2),
        12 => (3, 3),
        _ => return Err(AvcError::InvalidData("B sub-macroblock type")),
    };
    Ok(value)
}

fn intra4_edges(mb: &InterMbState, side: bool) -> [u8; 4] {
    if !mb.intra_nxn {
        return [2; 4];
    }
    if mb.transform8x8 {
        let indices = if side { [1, 1, 3, 3] } else { [2, 2, 3, 3] };
        indices.map(|index| mb.modes8[index])
    } else {
        let indices = if side {
            [5, 7, 13, 15]
        } else {
            [10, 11, 14, 15]
        };
        indices.map(|index| mb.modes4[index])
    }
}

fn intra8_edges(mb: &InterMbState, side: bool) -> [u8; 2] {
    if !mb.intra_nxn {
        [2; 2]
    } else if mb.transform8x8 {
        let indices = if side { [1, 3] } else { [2, 3] };
        indices.map(|index| mb.modes8[index])
    } else {
        let indices = if side { [5, 13] } else { [10, 14] };
        indices.map(|index| mb.modes4[index])
    }
}

fn median(a: i32, b: i32, c: i32) -> i32 {
    let mut values = [a, b, c];
    values.sort_unstable();
    values[1]
}

fn p_reference_list(
    references: &[Yuv420Picture],
    frame_num: u32,
    frame_num_bits: usize,
    changes: &[RefPicReorder],
) -> Result<Vec<usize>, AvcError> {
    let max_frame_num = 1u32 << frame_num_bits;
    let mut list: Vec<usize> = (0..references.len()).rev().collect();
    let mut predicted_pic_num = frame_num;
    for (position, change) in changes.iter().enumerate() {
        let target_pic_num = match change {
            RefPicReorder::Subtract(distance) => {
                predicted_pic_num = predicted_pic_num
                    .wrapping_add(max_frame_num)
                    .wrapping_sub(distance.wrapping_add(1))
                    % max_frame_num;
                predicted_pic_num
            }
            RefPicReorder::Add(distance) => {
                predicted_pic_num =
                    predicted_pic_num.wrapping_add(*distance).wrapping_add(1) % max_frame_num;
                predicted_pic_num
            }
            RefPicReorder::LongTerm(_) => {
                return Err(AvcError::Unsupported("P long-term reference"));
            }
        };
        let target = references
            .iter()
            .position(|picture| picture.frame_num == target_pic_num)
            .ok_or(AvcError::Unsupported("P reordered reference unavailable"))?;
        list.insert(position, target);
        let mut next = position + 1;
        while next < list.len() {
            if list[next] == target {
                list.remove(next);
            } else {
                next += 1;
            }
        }
    }
    Ok(list)
}

/// Initial B-slice lists for progressive, short-term reference frames (8.2.4.2.3).
pub(super) fn b_reference_lists(
    references: &[Yuv420Picture],
    current_poc: i32,
) -> (Vec<usize>, Vec<usize>) {
    let mut earlier = Vec::new();
    let mut equal = Vec::new();
    let mut later = Vec::new();
    for (index, picture) in references.iter().enumerate() {
        if picture.pic_order_cnt < current_poc {
            earlier.push(index);
        } else if picture.pic_order_cnt == current_poc {
            equal.push(index);
        } else {
            later.push(index);
        }
    }
    earlier.sort_by_key(|&index| std::cmp::Reverse(references[index].pic_order_cnt));
    later.sort_by_key(|&index| references[index].pic_order_cnt);
    let mut list0 = earlier.clone();
    list0.extend_from_slice(&equal);
    list0.extend_from_slice(&later);
    let mut list1 = later;
    list1.extend_from_slice(&equal);
    list1.extend_from_slice(&earlier);
    if list1.len() > 1 && list1 == list0 {
        list1.swap(0, 1);
    }
    (list0, list1)
}

fn motion_predictor(
    left: Option<[i32; 2]>,
    above: Option<[i32; 2]>,
    upper_right: Option<[i32; 2]>,
    upper_left: Option<[i32; 2]>,
) -> [i32; 2] {
    let upper_right = upper_right.or(upper_left);
    if above.is_none() && upper_right.is_none() {
        return left.unwrap_or([0, 0]);
    }
    let a = left.unwrap_or([0, 0]);
    let b = above.unwrap_or([0, 0]);
    let c = upper_right.unwrap_or([0, 0]);
    [median(a[0], b[0], c[0]), median(a[1], b[1], c[1])]
}

fn motion_predictor_for_ref(
    left: Option<(u8, [i32; 2])>,
    above: Option<(u8, [i32; 2])>,
    upper_right: Option<(u8, [i32; 2])>,
    upper_left: Option<(u8, [i32; 2])>,
    reference: u8,
) -> [i32; 2] {
    let right = upper_right.or(upper_left);
    let mut matching = None;
    let mut count = 0;
    for candidate in [left, above, right] {
        if let Some((index, vector)) = candidate
            && index == reference
        {
            matching = Some(vector);
            count += 1;
        }
    }
    if count == 1 {
        return matching.unwrap();
    }
    motion_predictor(
        left.map(|(_, vector)| vector),
        above.map(|(_, vector)| vector),
        right.map(|(_, vector)| vector),
        None,
    )
}

fn p_neighbor_motion(mb: Option<&InterMbState>, cell: usize) -> Option<(u8, [i32; 2])> {
    mb.map(|mb| {
        if mb.intra16 || mb.intra_nxn {
            // Intra neighbors are available for prediction with refIdx = -1 and mv = 0.
            (u8::MAX, [0, 0])
        } else {
            (mb.refs4[cell], mb.motion4[cell])
        }
    })
}

fn p_top_right_candidate(
    kind: u8,
    above: Option<&InterMbState>,
    upper_right: Option<&InterMbState>,
) -> Option<(u8, [i32; 2])> {
    // The left P_8x16 partition ends halfway across the macroblock.
    if kind == 2 {
        p_neighbor_motion(above, 14)
    } else {
        p_neighbor_motion(upper_right, 12)
    }
}

fn p_skip_motion(
    x: usize,
    y: usize,
    left: Option<(u8, [i32; 2])>,
    above: Option<(u8, [i32; 2])>,
    predicted: [i32; 2],
) -> [i32; 2] {
    if x == 0
        || y == 0
        || left.is_some_and(|(reference, vector)| reference == 0 && vector == [0, 0])
        || above.is_some_and(|(reference, vector)| reference == 0 && vector == [0, 0])
    {
        [0, 0]
    } else {
        predicted
    }
}

pub(super) fn spatial_direct_motion(
    left: Option<MotionCell>,
    above: Option<MotionCell>,
    upper_right: Option<MotionCell>,
    upper_left: Option<MotionCell>,
    colocated: MotionCell,
    colocated_short_term: bool,
) -> MotionCell {
    let right = upper_right.or(upper_left);
    let refs = [0, 1].map(|list| {
        [left, above, right]
            .into_iter()
            .flatten()
            .filter_map(|cell| if list == 0 { cell.l0 } else { cell.l1 }.map(|(index, _)| index))
            .min()
    });
    let direct_zero = refs[0].is_none() && refs[1].is_none();
    let colocated = colocated.l0.or(colocated.l1);
    let colocated_zero = colocated.is_some_and(|(index, vector)| {
        colocated_short_term
            && index == 0
            && vector[0].unsigned_abs() <= 1
            && vector[1].unsigned_abs() <= 1
    });
    let refs = if direct_zero {
        [Some(0), Some(0)]
    } else {
        refs
    };
    let vector = |list: usize, index: u8| {
        if direct_zero || (index == 0 && colocated_zero) {
            return [0, 0];
        }
        let from = |neighbor: Option<MotionCell>| {
            neighbor
                .map(|cell| if list == 0 { cell.l0 } else { cell.l1 }.unwrap_or((u8::MAX, [0; 2])))
        };
        motion_predictor_for_ref(
            from(left),
            from(above),
            from(upper_right),
            from(upper_left),
            index,
        )
    };
    MotionCell {
        l0: refs[0].map(|index| (index, vector(0, index))),
        l1: refs[1].map(|index| (index, vector(1, index))),
    }
}

// Progressive short-term case of 2005 8.4.1.2.3 (equations 8-188..8-199).
fn temporal_direct_motion(
    colocated: MotionCell,
    colocated_lists: &[Vec<i32>; 2],
    list0_pocs: &[i32],
    current_poc: i32,
    colocated_poc: i32,
) -> Result<MotionCell, AvcError> {
    let (reference, vector) = if let Some((index, vector)) = colocated.l0.or(colocated.l1) {
        let list = usize::from(colocated.l0.is_none());
        let poc = colocated_lists[list]
            .get(index as usize)
            .ok_or(AvcError::InvalidData(
                "co-located reference identity missing",
            ))?;
        let reference =
            list0_pocs
                .iter()
                .position(|value| value == poc)
                .ok_or(AvcError::InvalidData(
                    "co-located reference absent from list 0",
                ))?;
        (reference, vector)
    } else {
        (0, [0; 2])
    };
    let pic0 = *list0_pocs
        .get(reference)
        .ok_or(AvcError::InvalidData("temporal direct list 0 empty"))?;
    let td = (colocated_poc - pic0).clamp(-128, 127);
    let tb = (current_poc - pic0).clamp(-128, 127);
    let (mv0, mv1) = if td == 0 {
        (vector, [0; 2])
    } else {
        let tx = (16384 + (td / 2).abs()) / td;
        let scale = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
        let mv0 = vector.map(|v| ((i64::from(scale) * i64::from(v) + 128) >> 8) as i32);
        (mv0, [mv0[0] - vector[0], mv0[1] - vector[1]])
    };
    Ok(MotionCell {
        l0: Some((
            u8::try_from(reference)
                .map_err(|_| AvcError::InvalidData("reference index overflow"))?,
            mv0,
        )),
        l1: Some((0, mv1)),
    })
}

fn combine_16x8(top: InterMacroblock, bottom: InterMacroblock) -> InterMacroblock {
    let mut block = top;
    block.luma[128..].copy_from_slice(&bottom.luma[128..]);
    block.cb[32..].copy_from_slice(&bottom.cb[32..]);
    block.cr[32..].copy_from_slice(&bottom.cr[32..]);
    block
}

fn combine_8x16(left: InterMacroblock, right: InterMacroblock) -> InterMacroblock {
    let mut block = left;
    for row in 0..16 {
        let offset = row * 16 + 8;
        block.luma[offset..offset + 8].copy_from_slice(&right.luma[offset..offset + 8]);
    }
    for row in 0..8 {
        let offset = row * 8 + 4;
        block.cb[offset..offset + 4].copy_from_slice(&right.cb[offset..offset + 4]);
        block.cr[offset..offset + 4].copy_from_slice(&right.cr[offset..offset + 4]);
    }
    block
}

fn combine_8x8(blocks: &[InterMacroblock; 4]) -> InterMacroblock {
    let mut result = InterMacroblock {
        luma: [0; 256],
        cb: [0; 64],
        cr: [0; 64],
    };
    for (part, source) in blocks.iter().enumerate() {
        let x = part % 2;
        let y = part / 2;
        for row in 0..8 {
            let offset = (y * 8 + row) * 16 + x * 8;
            result.luma[offset..offset + 8].copy_from_slice(&source.luma[offset..offset + 8]);
        }
        for row in 0..4 {
            let offset = (y * 4 + row) * 8 + x * 4;
            result.cb[offset..offset + 4].copy_from_slice(&source.cb[offset..offset + 4]);
            result.cr[offset..offset + 4].copy_from_slice(&source.cr[offset..offset + 4]);
        }
    }
    result
}

fn add_chroma_residual(
    block: &mut InterMacroblock,
    dc_levels: &[i32; 4],
    ac_levels: &[[i32; 15]; 4],
    qp: i32,
    channel: usize,
) -> Result<(), AvcError> {
    let scaled = inverse_chroma_dc(dc_levels, qp)?;
    let plane = if channel == 0 {
        &mut block.cb
    } else {
        &mut block.cr
    };
    for quadrant in 0..4 {
        let mut scanned = [0; 16];
        scanned[1..].copy_from_slice(&ac_levels[quadrant]);
        let mut coefficients = inverse_4x4_frame_scan(&scanned);
        coefficients[0] = scaled[quadrant];
        let residual = inverse_4x4_residual(&coefficients, qp, true)?;
        let x0 = (quadrant % 2) * 4;
        let y0 = (quadrant / 2) * 4;
        for y in 0..4 {
            for x in 0..4 {
                let index = (y0 + y) * 8 + x0 + x;
                plane[index] = (i32::from(plane[index]) + residual[y * 4 + x]).clamp(0, 255) as u8;
            }
        }
    }
    Ok(())
}

fn coded_luma4_mask(residual: &InterLumaResidual) -> u16 {
    let mut mask = 0;
    for block in 0..16 {
        let coded = match residual {
            InterLumaResidual::None => false,
            InterLumaResidual::FourByFour(levels) => levels[block].iter().any(|&v| v != 0),
            InterLumaResidual::EightByEight(levels) => levels[block / 4].iter().any(|&v| v != 0),
        };
        if coded {
            let row = block / 8 * 2 + block % 4 / 2;
            let col = block / 4 % 2 * 2 + block % 2;
            mask |= 1 << (row * 4 + col);
        }
    }
    mask
}

fn add_luma_residual(
    block: &mut InterMacroblock,
    levels: &InterLumaResidual,
    qp: i32,
) -> Result<(), AvcError> {
    match levels {
        InterLumaResidual::None => {}
        InterLumaResidual::FourByFour(blocks) => {
            for (index, scanned) in blocks.iter().enumerate() {
                if scanned.iter().all(|&level| level == 0) {
                    continue;
                }
                let region = index / 4;
                let sub = index % 4;
                let x0 = (region % 2) * 8 + (sub % 2) * 4;
                let y0 = (region / 2) * 8 + (sub / 2) * 4;
                let coefficients = inverse_4x4_frame_scan(scanned);
                let residual = inverse_4x4_residual(&coefficients, qp, false)?;
                for row in 0..4 {
                    for col in 0..4 {
                        let pixel = (y0 + row) * 16 + x0 + col;
                        block.luma[pixel] = (i32::from(block.luma[pixel]) + residual[row * 4 + col])
                            .clamp(0, 255) as u8;
                    }
                }
            }
        }
        InterLumaResidual::EightByEight(blocks) => {
            for (index, scanned) in blocks.iter().enumerate() {
                if scanned.iter().all(|&level| level == 0) {
                    continue;
                }
                let x0 = (index % 2) * 8;
                let y0 = (index / 2) * 8;
                let coefficients = inverse_8x8_frame_scan(scanned);
                let residual = inverse_8x8_residual(&coefficients, qp, &[16; 64])?;
                for row in 0..8 {
                    for col in 0..8 {
                        let pixel = (y0 + row) * 16 + x0 + col;
                        block.luma[pixel] = (i32::from(block.luma[pixel]) + residual[row * 8 + col])
                            .clamp(0, 255) as u8;
                    }
                }
            }
        }
    }
    Ok(())
}

fn write_inter_block(picture: &mut Yuv420Picture, x: usize, y: usize, block: &InterMacroblock) {
    for row in 0..16 {
        let offset = (y * 16 + row) * picture.width + x * 16;
        picture.luma[offset..offset + 16].copy_from_slice(&block.luma[row * 16..row * 16 + 16]);
    }
    for (plane, data) in [(&mut picture.cb, &block.cb), (&mut picture.cr, &block.cr)] {
        for row in 0..8 {
            let offset = (y * 8 + row) * (picture.width / 2) + x * 8;
            plane[offset..offset + 8].copy_from_slice(&data[row * 8..row * 8 + 8]);
        }
    }
}

/// Decode one-reference P pictures with 16x16, 16x8, 8x16, 8x8 and Intra16x16 blocks.
/// Subpartitions smaller than 8x8 are not yet supported.
pub fn decode_cabac_p_2005(
    nal: &[u8],
    sps: &SequenceParameters,
    pps: &PictureParameters2005,
    references: &[Yuv420Picture],
) -> Result<Yuv420Picture, AvcError> {
    let reference = references
        .last()
        .ok_or(AvcError::Unsupported("P reference picture"))?;
    if sps.profile_idc != 100
        || sps.chroma_format_idc != 1
        || sps.bit_depth_luma != 8
        || sps.bit_depth_chroma != 8
        || !sps.frame_mbs_only
        || sps.scaling_matrices_present
        || pps.scaling_matrices_present
        || sps.width != sps.width_mbs * 16
        || sps.height == 0
        || sps.height > sps.frame_height_mbs * 16
        || reference.width != sps.width as usize
        || reference.height != (sps.frame_height_mbs * 16) as usize
    {
        return Err(AvcError::Unsupported("P picture format"));
    }
    let slice = parse_cabac_inter_slice(nal, sps, &pps.core)?;
    if std::env::var_os("WEBMEDIA_TRACE_P").is_some() {
        eprintln!(
            "P header: type={} first_mb={} active_refs={} available_refs={} frame_num={} poc_lsb={} qp={} cabac_init={} reorder={:?} reference_frames={:?}",
            slice.slice_type,
            slice.first_mb,
            slice.ref_idx_l0,
            references.len(),
            slice.frame_num,
            slice.pic_order_cnt_lsb,
            slice.slice_qp,
            slice.cabac_init_idc,
            slice.reorder_l0,
            references
                .iter()
                .map(|picture| picture.frame_num)
                .collect::<Vec<_>>()
        );
    }
    if slice.slice_type % 5 != 0 || slice.first_mb != 0 || slice.ref_idx_l0 == 0 {
        return Err(AvcError::Unsupported("P picture slice layout"));
    }
    let list0 = p_reference_list(
        references,
        slice.frame_num,
        sps.frame_num_bits,
        &slice.reorder_l0,
    )?;
    if list0.len() < slice.ref_idx_l0 as usize {
        return Err(AvcError::Unsupported("P reference list length"));
    }
    if references
        .iter()
        .any(|picture| picture.width != reference.width || picture.height != reference.height)
    {
        return Err(AvcError::Unsupported("P reference dimensions"));
    }
    let width_mbs = sps.width_mbs as usize;
    let height_mbs = sps.frame_height_mbs as usize;
    let (pic_order_cnt_msb, pic_order_cnt) = type0_pic_order_count(
        slice.pic_order_cnt_lsb,
        sps.pic_order_cnt_lsb_bits
            .ok_or(AvcError::Unsupported("POC type"))?,
        Some((reference.pic_order_cnt_msb, reference.pic_order_cnt_lsb)),
    )?;
    let mut picture = Yuv420Picture {
        width: reference.width,
        height: reference.height,
        frame_num: slice.frame_num,
        pic_order_cnt_lsb: slice.pic_order_cnt_lsb,
        pic_order_cnt_msb,
        pic_order_cnt,
        luma: vec![0; reference.luma.len()],
        cb: vec![0; reference.cb.len()],
        cr: vec![0; reference.cr.len()],
        motion: vec![[MotionCell::default(); 4]; width_mbs * height_mbs],
        reference_pocs: [
            list0.iter().map(|&i| references[i].pic_order_cnt).collect(),
            Vec::new(),
        ],
        luma_half: std::sync::OnceLock::new(),
    };
    let mut states: Vec<Option<InterMbState>> = vec![None; width_mbs * height_mbs];
    let mut decoder = CabacDecoder::new(&slice.rbsp[slice.data_byte_offset..])?;
    let mut inter = InterMbContexts::new(slice.slice_qp, slice.cabac_init_idc, false)?;
    let mut vectors = MotionVectorContexts::new(slice.slice_qp, slice.cabac_init_idc)?;
    let mut reference_indices = ReferenceIndexContexts::new(slice.slice_qp, slice.cabac_init_idc)?;
    let mut patterns = CodedBlockContexts::new_inter(slice.slice_qp, slice.cabac_init_idc)?;
    let mut chroma_contexts = ChromaDcContexts::new_inter(slice.slice_qp, slice.cabac_init_idc)?;
    let mut chroma_ac_contexts = ChromaAcContexts::new_inter(slice.slice_qp, slice.cabac_init_idc)?;
    let mut luma_dc_contexts =
        Luma16x16DcContexts::new_inter(slice.slice_qp, slice.cabac_init_idc)?;
    let mut luma_ac_contexts =
        Luma16x16AcContexts::new_inter(slice.slice_qp, slice.cabac_init_idc)?;
    let mut luma4_contexts = Luma4x4Contexts::new_inter(slice.slice_qp, slice.cabac_init_idc)?;
    let mut luma8_contexts = Luma8x8Contexts::new_inter(slice.slice_qp, slice.cabac_init_idc)?;
    let mut transform_contexts =
        Transform8x8Contexts::new_inter(slice.slice_qp, slice.cabac_init_idc)?;
    let mut intra_pred_contexts = IntraPredContexts::new(slice.slice_qp)?;
    let mut qp_contexts = MbQpContexts::new(slice.slice_qp)?;
    let mut qp = slice.slice_qp;
    let mut previous_qp_delta_nonzero = false;
    let trace_failing_p = std::env::var_os("WEBMEDIA_TRACE_P_MB").is_some()
        && slice.frame_num == 9
        && matches!(slice.pic_order_cnt_lsb, 2 | 40);
    for y in 0..height_mbs {
        for x in 0..width_mbs {
            let index = y * width_mbs + x;
            let left = (x > 0).then(|| states[index - 1].as_ref()).flatten();
            let above = (y > 0)
                .then(|| states[index - width_mbs].as_ref())
                .flatten();
            let upper_right = (y > 0 && x + 1 < width_mbs)
                .then(|| states[index - width_mbs + 1].as_ref())
                .flatten();
            let upper_left = (y > 0 && x > 0)
                .then(|| states[index - width_mbs - 1].as_ref())
                .flatten();
            let skipped = decoder.inter_mb_skip_flag(
                &mut inter,
                left.map(|mb| mb.skipped),
                above.map(|mb| mb.skipped),
            )?;
            if trace_failing_p
                && (slice.pic_order_cnt_lsb == 40 && index <= 130
                    || slice.pic_order_cnt_lsb == 2
                        && ((460..=470).contains(&index) || (580..=604).contains(&index)))
            {
                eprintln!(
                    "P MB {index}: skipped={skipped} bits={}",
                    decoder.consumed_bits()
                );
            }
            let left_reference = p_neighbor_motion(left, 3);
            let above_reference = p_neighbor_motion(above, 12);
            let upper_right_reference = p_neighbor_motion(upper_right, 12);
            let upper_left_reference = p_neighbor_motion(upper_left, 15);
            let upper_candidate = if y > 0 && x + 1 < width_mbs {
                upper_right_reference
            } else {
                upper_left_reference
            };
            let predictor =
                motion_predictor_for_ref(left_reference, above_reference, upper_candidate, None, 0);
            let mut sub_partitions = None;
            let (
                motion,
                mvd,
                ref_indices,
                coded,
                transform8x8,
                luma_residual,
                dc_levels,
                ac_levels,
                dc_coded,
                partition_kind,
            ) = if skipped {
                previous_qp_delta_nonzero = false;
                let vector = p_skip_motion(x, y, left_reference, above_reference, predictor);
                (
                    [vector; 4],
                    [[0, 0]; 4],
                    [0; 4],
                    CodedBlockPattern {
                        luma: 0,
                        chroma: 0,
                        pcm: false,
                    },
                    false,
                    InterLumaResidual::None,
                    [[0; 4]; 2],
                    [[[0; 15]; 4]; 2],
                    [false; 2],
                    0,
                )
            } else {
                let kind = decoder.p_inter_mb_type(&mut inter)?;
                if trace_failing_p
                    && (slice.pic_order_cnt_lsb == 40 && index <= 130
                        || slice.pic_order_cnt_lsb == 2
                            && ((460..=470).contains(&index) || (580..=604).contains(&index)))
                {
                    eprintln!("P MB {index}: kind={kind} bits={}", decoder.consumed_bits());
                }
                if kind >= 5 {
                    let intra_kind = kind - 5;
                    if intra_kind == 0 {
                        let transform8x8 = if pps.transform_8x8 {
                            decoder.transform_size_8x8_flag(
                                &mut transform_contexts,
                                left.map(|mb| mb.transform8x8),
                                above.map(|mb| mb.transform8x8),
                            )?
                        } else {
                            false
                        };
                        let mut modes4 = [2; 16];
                        let mut modes8 = [2; 4];
                        if transform8x8 {
                            let mut codes = [IntraPredCode {
                                use_predicted_mode: true,
                                remaining_mode: None,
                            }; 4];
                            for code in &mut codes {
                                *code = decoder.intra_luma_pred_code(&mut intra_pred_contexts)?;
                            }
                            modes8 = intra8x8_modes(
                                &codes,
                                left.map(|mb| intra8_edges(&mb, true)),
                                above.map(|mb| intra8_edges(&mb, false)),
                            )?;
                        } else {
                            let mut codes = [IntraPredCode {
                                use_predicted_mode: true,
                                remaining_mode: None,
                            }; 16];
                            for code in &mut codes {
                                *code = decoder.intra_luma_pred_code(&mut intra_pred_contexts)?;
                            }
                            modes4 = intra4x4_modes(
                                &codes,
                                left.map(|mb| intra4_edges(&mb, true)),
                                above.map(|mb| intra4_edges(&mb, false)),
                            )?;
                        }
                        let chroma_mode = decoder.intra_chroma_pred_mode(
                            &mut intra_pred_contexts,
                            left.map(|mb| mb.chroma_mode),
                            above.map(|mb| mb.chroma_mode),
                        )?;
                        let coded = decoder.coded_block_pattern(
                            &mut patterns,
                            left.map(|mb| mb.coded),
                            above.map(|mb| mb.coded),
                        )?;
                        if coded.luma != 0 || coded.chroma != 0 {
                            let delta =
                                decoder.mb_qp_delta(&mut qp_contexts, previous_qp_delta_nonzero)?;
                            previous_qp_delta_nonzero = delta != 0;
                            qp = (qp + delta).rem_euclid(52);
                        } else {
                            previous_qp_delta_nonzero = false;
                        }
                        let x0 = x * 16;
                        let y0 = y * 16;
                        let side = left_edge(&picture.luma, picture.width, x0, y0);
                        let corner = (x0 > 0 && y0 > 0)
                            .then(|| picture.luma[(y0 - 1) * picture.width + x0 - 1]);
                        let (luma, luma_ac_right, luma_ac_bottom) = if transform8x8 {
                            let mut blocks = [[0; 64]; 4];
                            for (region, levels) in blocks.iter_mut().enumerate() {
                                if coded.luma & (1 << region) != 0 {
                                    *levels = decoder.luma8x8_coefficients(&mut luma8_contexts)?;
                                }
                            }
                            let luma = reconstruct_intra8x8_luma(
                                &modes8,
                                &blocks,
                                qp,
                                &[16; 64],
                                upper_edge(&picture.luma, picture.width, x0, y0),
                                side,
                                corner,
                            )?;
                            let edge =
                                |region: usize| blocks[region].iter().any(|&level| level != 0);
                            (
                                luma,
                                [edge(1), edge(1), edge(3), edge(3)],
                                [edge(2), edge(2), edge(3), edge(3)],
                            )
                        } else {
                            let blocks = decoder.luma4x4_macroblock(
                                &mut luma4_contexts,
                                coded.luma,
                                left.map(|mb| mb.luma_ac_right),
                                above.map(|mb| mb.luma_ac_bottom),
                            )?;
                            let luma = reconstruct_intra4x4_luma(
                                &modes4,
                                &blocks,
                                qp,
                                upper_edge(&picture.luma, picture.width, x0, y0),
                                side,
                                corner,
                            )?;
                            let edge = |index: usize| blocks[index].iter().any(|&level| level != 0);
                            (luma, [5, 7, 13, 15].map(edge), [10, 11, 14, 15].map(edge))
                        };
                        let mut chroma_dc = [[0; 4]; 2];
                        if coded.chroma != 0 {
                            for (channel, levels) in chroma_dc.iter_mut().enumerate() {
                                *levels = decoder.chroma_dc_coefficients(
                                    &mut chroma_contexts,
                                    left.map(|mb| mb.chroma_dc_coded[channel]),
                                    above.map(|mb| mb.chroma_dc_coded[channel]),
                                )?;
                            }
                        }
                        let mut chroma_ac = [[[0; 15]; 4]; 2];
                        if coded.chroma == 2 {
                            for (channel, blocks) in chroma_ac.iter_mut().enumerate() {
                                *blocks = decoder.chroma_ac_macroblock(
                                    &mut chroma_ac_contexts,
                                    left.map(|mb| mb.chroma_ac_right[channel]),
                                    above.map(|mb| mb.chroma_ac_bottom[channel]),
                                )?;
                            }
                        }
                        let cx0 = x * 8;
                        let cy0 = y * 8;
                        let chroma_width = picture.width / 2;
                        let cb = reconstruct_chroma(
                            chroma_mode,
                            &chroma_dc[0],
                            &chroma_ac[0],
                            chroma_qp(qp, pps.core.chroma_qp_index_offset)?,
                            upper_edge(&picture.cb, chroma_width, cx0, cy0),
                            left_edge(&picture.cb, chroma_width, cx0, cy0),
                            (cx0 > 0 && cy0 > 0)
                                .then(|| picture.cb[(cy0 - 1) * chroma_width + cx0 - 1]),
                        )?;
                        let cr = reconstruct_chroma(
                            chroma_mode,
                            &chroma_dc[1],
                            &chroma_ac[1],
                            chroma_qp(qp, pps.second_chroma_qp_index_offset)?,
                            upper_edge(&picture.cr, chroma_width, cx0, cy0),
                            left_edge(&picture.cr, chroma_width, cx0, cy0),
                            (cx0 > 0 && cy0 > 0)
                                .then(|| picture.cr[(cy0 - 1) * chroma_width + cx0 - 1]),
                        )?;
                        write_inter_block(&mut picture, x, y, &InterMacroblock { luma, cb, cr });
                        states[index] = Some(InterMbState {
                            qp,
                            skipped: false,
                            intra16: false,
                            intra_nxn: true,
                            transform8x8,
                            modes4,
                            modes8,
                            motion4: [[0; 2]; 16],
                            mvd4: [[0; 2]; 16],
                            refs4: [0; 16],
                            coded,
                            coded_luma4: 0,
                            chroma_mode,
                            luma_dc_coded: false,
                            luma_ac_right,
                            luma_ac_bottom,
                            chroma_dc_coded: chroma_dc
                                .map(|levels| levels.iter().any(|&level| level != 0)),
                            chroma_ac_right: std::array::from_fn(|channel| {
                                [1, 3].map(|block| {
                                    chroma_ac[channel][block].iter().any(|&level| level != 0)
                                })
                            }),
                            chroma_ac_bottom: std::array::from_fn(|channel| {
                                [2, 3].map(|block| {
                                    chroma_ac[channel][block].iter().any(|&level| level != 0)
                                })
                            }),
                        });
                        let ended = decoder.terminate()?;
                        if ended != (index + 1 == states.len()) {
                            if std::env::var_os("WEBMEDIA_TRACE_P").is_some() {
                                eprintln!(
                                    "P slice ended={ended} at MB {index}, bits={}",
                                    decoder.consumed_bits()
                                );
                            }
                            return Err(AvcError::Unsupported(
                                "P picture has multiple or incomplete slices",
                            ));
                        }
                        continue;
                    }
                    if !(1..=24).contains(&intra_kind) {
                        return Err(AvcError::Unsupported("P intra macroblock mode"));
                    }
                    let coded = CodedBlockPattern {
                        luma: if intra_kind >= 13 { 15 } else { 0 },
                        chroma: ((intra_kind - 1) / 4) % 3,
                        pcm: false,
                    };
                    let chroma_mode = decoder.intra_chroma_pred_mode(
                        &mut intra_pred_contexts,
                        left.map(|mb| mb.chroma_mode),
                        above.map(|mb| mb.chroma_mode),
                    )?;
                    let delta = decoder.mb_qp_delta(&mut qp_contexts, previous_qp_delta_nonzero)?;
                    previous_qp_delta_nonzero = delta != 0;
                    qp = (qp + delta).rem_euclid(52);
                    let luma_dc = decoder.luma16x16_dc_coefficients(
                        &mut luma_dc_contexts,
                        left.map(|mb| mb.intra16 && mb.luma_dc_coded),
                        above.map(|mb| mb.intra16 && mb.luma_dc_coded),
                    )?;
                    let luma_ac = if coded.luma != 0 {
                        decoder.luma16x16_ac_macroblock(
                            &mut luma_ac_contexts,
                            left.map(|mb| mb.luma_ac_right),
                            above.map(|mb| mb.luma_ac_bottom),
                        )?
                    } else {
                        [[0; 15]; 16]
                    };
                    let mut chroma_dc = [[0; 4]; 2];
                    if coded.chroma != 0 {
                        for (channel, levels) in chroma_dc.iter_mut().enumerate() {
                            *levels = decoder.chroma_dc_coefficients(
                                &mut chroma_contexts,
                                left.map(|mb| mb.coded.chroma != 0 && mb.chroma_dc_coded[channel]),
                                above.map(|mb| mb.coded.chroma != 0 && mb.chroma_dc_coded[channel]),
                            )?;
                        }
                    }
                    let mut chroma_ac = [[[0; 15]; 4]; 2];
                    if coded.chroma == 2 {
                        for (channel, blocks) in chroma_ac.iter_mut().enumerate() {
                            *blocks = decoder.chroma_ac_macroblock(
                                &mut chroma_ac_contexts,
                                left.map(|mb| mb.chroma_ac_right[channel]),
                                above.map(|mb| mb.chroma_ac_bottom[channel]),
                            )?;
                        }
                    }
                    let x0 = x * 16;
                    let y0 = y * 16;
                    let cx0 = x * 8;
                    let cy0 = y * 8;
                    let chroma_width = picture.width / 2;
                    let luma = reconstruct_intra16x16_luma(
                        (intra_kind - 1) % 4,
                        &luma_dc,
                        &luma_ac,
                        qp,
                        upper_edge(&picture.luma, picture.width, x0, y0),
                        left_edge(&picture.luma, picture.width, x0, y0),
                        (x0 > 0 && y0 > 0).then(|| picture.luma[(y0 - 1) * picture.width + x0 - 1]),
                    )?;
                    let cb = reconstruct_chroma(
                        chroma_mode,
                        &chroma_dc[0],
                        &chroma_ac[0],
                        chroma_qp(qp, pps.core.chroma_qp_index_offset)?,
                        upper_edge(&picture.cb, chroma_width, cx0, cy0),
                        left_edge(&picture.cb, chroma_width, cx0, cy0),
                        (cx0 > 0 && cy0 > 0)
                            .then(|| picture.cb[(cy0 - 1) * chroma_width + cx0 - 1]),
                    )?;
                    let cr = reconstruct_chroma(
                        chroma_mode,
                        &chroma_dc[1],
                        &chroma_ac[1],
                        chroma_qp(qp, pps.second_chroma_qp_index_offset)?,
                        upper_edge(&picture.cr, chroma_width, cx0, cy0),
                        left_edge(&picture.cr, chroma_width, cx0, cy0),
                        (cx0 > 0 && cy0 > 0)
                            .then(|| picture.cr[(cy0 - 1) * chroma_width + cx0 - 1]),
                    )?;
                    write_inter_block(&mut picture, x, y, &InterMacroblock { luma, cb, cr });
                    states[index] = Some(InterMbState {
                        qp,
                        skipped: false,
                        intra16: true,
                        intra_nxn: false,
                        transform8x8: false,
                        modes4: [2; 16],
                        modes8: [2; 4],
                        motion4: [[0; 2]; 16],
                        mvd4: [[0; 2]; 16],
                        refs4: [0; 16],
                        coded,
                        coded_luma4: 0,
                        chroma_mode,
                        luma_dc_coded: luma_dc.iter().any(|&level| level != 0),
                        luma_ac_right: [5, 7, 13, 15]
                            .map(|block| luma_ac[block].iter().any(|&level| level != 0)),
                        luma_ac_bottom: [10, 11, 14, 15]
                            .map(|block| luma_ac[block].iter().any(|&level| level != 0)),
                        chroma_dc_coded: chroma_dc
                            .map(|levels| levels.iter().any(|&level| level != 0)),
                        chroma_ac_right: std::array::from_fn(|channel| {
                            [1, 3].map(|block| {
                                chroma_ac[channel][block].iter().any(|&level| level != 0)
                            })
                        }),
                        chroma_ac_bottom: std::array::from_fn(|channel| {
                            [2, 3].map(|block| {
                                chroma_ac[channel][block].iter().any(|&level| level != 0)
                            })
                        }),
                    });
                    let ended = decoder.terminate()?;
                    if ended != (index + 1 == states.len()) {
                        if std::env::var_os("WEBMEDIA_TRACE_P").is_some() {
                            eprintln!(
                                "P slice ended={ended} at MB {index}, bits={}",
                                decoder.consumed_bits()
                            );
                        }
                        return Err(AvcError::Unsupported(
                            "P picture has multiple or incomplete slices",
                        ));
                    }
                    continue;
                }
                if kind > 3 {
                    return Err(AvcError::Unsupported("P macroblock partition mode"));
                }
                let (motion, mvd, ref_indices) = if kind == 3 {
                    let mut modes = [0u8; 4];
                    for mode in &mut modes {
                        *mode = decoder.p_sub_mb_type(&mut inter)?;
                    }
                    let mut refs = [0u8; 4];
                    for part in 0..4 {
                        let left_ref = match part {
                            1 | 3 => Some(refs[part - 1]),
                            0 | 2 => left
                                .filter(|mb| !mb.intra16 && !mb.intra_nxn)
                                .map(|mb| mb.refs4[part / 2 * 8 + 3]),
                            _ => None,
                        };
                        let above_ref = match part {
                            2 | 3 => Some(refs[part - 2]),
                            0 | 1 => above
                                .filter(|mb| !mb.intra16 && !mb.intra_nxn)
                                .map(|mb| mb.refs4[12 + part % 2 * 2]),
                            _ => None,
                        };
                        if slice.ref_idx_l0 > 1 {
                            refs[part] = decoder.reference_index(
                                &mut reference_indices,
                                left_ref,
                                above_ref,
                                slice.ref_idx_l0,
                            ).map_err(|error| {
                                if std::env::var_os("WEBMEDIA_TRACE_P").is_some() {
                                    eprintln!("P sub-ref at mb={index} part={part} modes={modes:?} refs={refs:?} bits={} active={}: {error:?}", decoder.consumed_bits(), slice.ref_idx_l0);
                                }
                                error
                            })?;
                        }
                    }
                    let mut motion = [[0; 2]; 4];
                    let mut mvd = [[0; 2]; 4];
                    let mut cell_motion: [Option<[i32; 2]>; 16] = [None; 16];
                    let mut cell_mvd: [Option<[i32; 2]>; 16] = [None; 16];
                    let mut cell_refs = [0u8; 16];
                    for part in 0..4 {
                        let base_x = (part % 2) * 2;
                        let base_y = (part / 2) * 2;
                        let count = match modes[part] {
                            0 => 1,
                            1 | 2 => 2,
                            3 => 4,
                            _ => return Err(AvcError::InvalidData("P sub-macroblock type")),
                        };
                        for sub in 0..count {
                            let (dx, dy, width, height) = p_sub_partition_rect(modes[part], sub);
                            let sx = base_x + dx;
                            let sy = base_y + dy;
                            let neighbor = |cx: isize,
                                            cy: isize,
                                            cells: &[Option<[i32; 2]>; 16],
                                            mvd: bool| {
                                if (0..4).contains(&cx) && (0..4).contains(&cy) {
                                    cells[cy as usize * 4 + cx as usize]
                                } else if cx < 0 && (0..4).contains(&cy) {
                                    left.filter(|mb| !mb.intra16 && !mb.intra_nxn).map(|mb| {
                                        if mvd {
                                            mb.mvd4[cy as usize * 4 + 3]
                                        } else {
                                            mb.motion4[cy as usize * 4 + 3]
                                        }
                                    })
                                } else if cy < 0 && (0..4).contains(&cx) {
                                    above.filter(|mb| !mb.intra16 && !mb.intra_nxn).map(|mb| {
                                        if mvd {
                                            mb.mvd4[12 + cx as usize]
                                        } else {
                                            mb.motion4[12 + cx as usize]
                                        }
                                    })
                                } else if cx >= 4 && cy < 0 {
                                    upper_right
                                        .filter(|mb| !mb.intra16 && !mb.intra_nxn)
                                        .map(|mb| if mvd { mb.mvd4[12] } else { mb.motion4[12] })
                                } else if cx < 0 && cy < 0 {
                                    upper_left
                                        .filter(|mb| !mb.intra16 && !mb.intra_nxn)
                                        .map(|mb| if mvd { mb.mvd4[15] } else { mb.motion4[15] })
                                } else {
                                    None
                                }
                            };
                            let am = neighbor(sx as isize - 1, sy as isize, &cell_mvd, true);
                            let bm = neighbor(sx as isize, sy as isize - 1, &cell_mvd, true);
                            let delta = [
                                decoder.motion_vector_difference(
                                    &mut vectors,
                                    0,
                                    am.map(|v| v[0]),
                                    bm.map(|v| v[0]),
                                )?,
                                decoder.motion_vector_difference(
                                    &mut vectors,
                                    1,
                                    am.map(|v| v[1]),
                                    bm.map(|v| v[1]),
                                )?,
                            ];
                            let motion_with_ref = |cx: isize, cy: isize| {
                                if (0..4).contains(&cx) && (0..4).contains(&cy) {
                                    let cell = cy as usize * 4 + cx as usize;
                                    cell_motion[cell].map(|vector| (cell_refs[cell], vector))
                                } else if cx < 0 && (0..4).contains(&cy) {
                                    p_neighbor_motion(left, cy as usize * 4 + 3)
                                } else if cy < 0 && (0..4).contains(&cx) {
                                    p_neighbor_motion(above, 12 + cx as usize)
                                } else if cx >= 4 && cy < 0 {
                                    p_neighbor_motion(upper_right, 12)
                                } else if cx < 0 && cy < 0 {
                                    p_neighbor_motion(upper_left, 15)
                                } else {
                                    None
                                }
                            };
                            let predicted = motion_predictor_for_ref(
                                motion_with_ref(sx as isize - 1, sy as isize),
                                motion_with_ref(sx as isize, sy as isize - 1),
                                motion_with_ref((sx + width) as isize, sy as isize - 1),
                                motion_with_ref(sx as isize - 1, sy as isize - 1),
                                refs[part],
                            );
                            let vector = [
                                predicted[0].saturating_add(delta[0]),
                                predicted[1].saturating_add(delta[1]),
                            ];
                            for row in sy..sy + height {
                                for col in sx..sx + width {
                                    let cell = row * 4 + col;
                                    cell_motion[cell] = Some(vector);
                                    cell_mvd[cell] = Some(delta);
                                    cell_refs[cell] = refs[part];
                                }
                            }
                        }
                        motion[part] = cell_motion[base_y * 4 + base_x].unwrap_or([0, 0]);
                        mvd[part] = cell_mvd[base_y * 4 + base_x].unwrap_or([0, 0]);
                    }
                    sub_partitions = Some(PSubPartitions {
                        modes,
                        vectors: cell_motion.map(|cell| cell.unwrap_or([0, 0])),
                        mvd: cell_mvd.map(|cell| cell.unwrap_or([0, 0])),
                        references: cell_refs,
                    });
                    (motion, mvd, refs)
                } else {
                    let mut refs = [0u8; 4];
                    let part_count = if kind == 0 { 1 } else { 2 };
                    for part in 0..part_count {
                        let left_ref = if part == 1 && kind == 2 {
                            Some(refs[0])
                        } else if let Some(mb) = left.filter(|mb| !mb.intra16 && !mb.intra_nxn) {
                            Some(mb.refs4[if part == 1 { 11 } else { 3 }])
                        } else {
                            None
                        };
                        let above_ref = if part == 1 && kind == 1 {
                            Some(refs[0])
                        } else if let Some(mb) = above.filter(|mb| !mb.intra16 && !mb.intra_nxn) {
                            Some(mb.refs4[if part == 1 { 14 } else { 12 }])
                        } else {
                            None
                        };
                        if slice.ref_idx_l0 > 1 {
                            refs[part] = decoder.reference_index(
                                &mut reference_indices,
                                left_ref,
                                above_ref,
                                slice.ref_idx_l0,
                            )?;
                        }
                    }
                    refs = match kind {
                        1 => [refs[0], refs[0], refs[1], refs[1]],
                        2 => [refs[0], refs[1], refs[0], refs[1]],
                        _ => [refs[0]; 4],
                    };
                    let top_mvd = [
                        decoder.motion_vector_difference(
                            &mut vectors,
                            0,
                            left.map(|mb| mb.mvd4[3][0]),
                            above.map(|mb| mb.mvd4[12][0]),
                        )?,
                        decoder.motion_vector_difference(
                            &mut vectors,
                            1,
                            left.map(|mb| mb.mvd4[3][1]),
                            above.map(|mb| mb.mvd4[12][1]),
                        )?,
                    ];
                    let top_predictor = match kind {
                        1 if above_reference.is_some_and(|(index, _)| index == refs[0]) => {
                            above_reference.unwrap().1
                        }
                        2 if left_reference.is_some_and(|(index, _)| index == refs[0]) => {
                            left_reference.unwrap().1
                        }
                        _ => motion_predictor_for_ref(
                            left_reference,
                            above_reference,
                            p_top_right_candidate(kind, above, upper_right),
                            upper_left_reference,
                            refs[0],
                        ),
                    };
                    let top = [
                        top_predictor[0].saturating_add(top_mvd[0]),
                        top_predictor[1].saturating_add(top_mvd[1]),
                    ];
                    if kind == 1 || kind == 2 {
                        let second_left_mvd = if kind == 1 {
                            left.map(|mb| mb.mvd4[11][0])
                        } else {
                            Some(top_mvd[0])
                        };
                        let second_above_mvd = if kind == 1 {
                            Some(top_mvd[0])
                        } else {
                            above.map(|mb| mb.mvd4[14][0])
                        };
                        let second_left_mvd_y = if kind == 1 {
                            left.map(|mb| mb.mvd4[11][1])
                        } else {
                            Some(top_mvd[1])
                        };
                        let second_above_mvd_y = if kind == 1 {
                            Some(top_mvd[1])
                        } else {
                            above.map(|mb| mb.mvd4[14][1])
                        };
                        let bottom_mvd = [
                            decoder.motion_vector_difference(
                                &mut vectors,
                                0,
                                second_left_mvd,
                                second_above_mvd,
                            )?,
                            decoder.motion_vector_difference(
                                &mut vectors,
                                1,
                                second_left_mvd_y,
                                second_above_mvd_y,
                            )?,
                        ];
                        let (left_motion, above_motion, right_motion, upper_left_motion, reference) =
                            if kind == 1 {
                                (
                                    p_neighbor_motion(left, 11),
                                    Some((refs[0], top)),
                                    None,
                                    p_neighbor_motion(left, 7),
                                    refs[2],
                                )
                            } else {
                                (
                                    Some((refs[0], top)),
                                    p_neighbor_motion(above, 14),
                                    p_neighbor_motion(upper_right, 12),
                                    // D is at (7,-1), i.e. above raster cell (1,3).
                                    p_neighbor_motion(above, 13),
                                    refs[1],
                                )
                            };
                        let directional = if kind == 1 {
                            left_motion
                        } else {
                            right_motion.or(upper_left_motion)
                        };
                        let bottom_predictor = directional
                            .filter(|(index, _)| *index == reference)
                            .map(|(_, vector)| vector)
                            .unwrap_or_else(|| {
                                motion_predictor_for_ref(
                                    left_motion,
                                    above_motion,
                                    right_motion,
                                    upper_left_motion,
                                    reference,
                                )
                            });
                        let bottom = [
                            bottom_predictor[0].saturating_add(bottom_mvd[0]),
                            bottom_predictor[1].saturating_add(bottom_mvd[1]),
                        ];
                        if kind == 1 {
                            (
                                [top, top, bottom, bottom],
                                [top_mvd, top_mvd, bottom_mvd, bottom_mvd],
                                refs,
                            )
                        } else {
                            (
                                [top, bottom, top, bottom],
                                [top_mvd, bottom_mvd, top_mvd, bottom_mvd],
                                refs,
                            )
                        }
                    } else {
                        ([top; 4], [top_mvd; 4], refs)
                    }
                };
                let coded = decoder.coded_block_pattern(
                    &mut patterns,
                    left.map(|mb| mb.coded),
                    above.map(|mb| mb.coded),
                )?;
                let transform8x8 = if coded.luma != 0
                    && pps.transform_8x8
                    && sub_partitions.is_none_or(|parts| parts.modes.iter().all(|&mode| mode == 0))
                {
                    decoder.transform_size_8x8_flag(
                        &mut transform_contexts,
                        left.map(|mb| mb.transform8x8),
                        above.map(|mb| mb.transform8x8),
                    )?
                } else {
                    false
                };
                if coded.luma != 0 || coded.chroma != 0 {
                    let delta = decoder.mb_qp_delta(&mut qp_contexts, previous_qp_delta_nonzero)?;
                    previous_qp_delta_nonzero = delta != 0;
                    qp = (qp + delta).rem_euclid(52);
                } else {
                    previous_qp_delta_nonzero = false;
                }
                let luma_residual = if coded.luma == 0 {
                    InterLumaResidual::None
                } else if transform8x8 {
                    let mut blocks = [[0; 64]; 4];
                    for (region, levels) in blocks.iter_mut().enumerate() {
                        if coded.luma & (1 << region) != 0 {
                            *levels = decoder.luma8x8_coefficients(&mut luma8_contexts)?;
                        }
                    }
                    InterLumaResidual::EightByEight(blocks)
                } else {
                    InterLumaResidual::FourByFour(decoder.luma4x4_macroblock(
                        &mut luma4_contexts,
                        coded.luma,
                        Some(left.map_or([false; 4], |mb| mb.luma_ac_right)),
                        Some(above.map_or([false; 4], |mb| mb.luma_ac_bottom)),
                    )?)
                };
                let dc_levels = if coded.chroma != 0 {
                    let mut levels = [[0; 4]; 2];
                    for channel in 0..2 {
                        levels[channel] = decoder.chroma_dc_coefficients(
                            &mut chroma_contexts,
                            Some(left.is_some_and(|mb| mb.chroma_dc_coded[channel])),
                            Some(above.is_some_and(|mb| mb.chroma_dc_coded[channel])),
                        )?;
                    }
                    levels
                } else {
                    [[0; 4]; 2]
                };
                let mut ac_levels = [[[0; 15]; 4]; 2];
                if coded.chroma == 2 {
                    for (channel, blocks) in ac_levels.iter_mut().enumerate() {
                        *blocks = decoder.chroma_ac_macroblock(
                            &mut chroma_ac_contexts,
                            Some(left.map_or([false; 2], |mb| mb.chroma_ac_right[channel])),
                            Some(above.map_or([false; 2], |mb| mb.chroma_ac_bottom[channel])),
                        )?;
                    }
                }
                let dc_coded = dc_levels.map(|levels| levels.iter().any(|&level| level != 0));
                (
                    motion,
                    mvd,
                    ref_indices,
                    coded,
                    transform8x8,
                    luma_residual,
                    dc_levels,
                    ac_levels,
                    dc_coded,
                    kind,
                )
            };
            let predict = |region: usize| {
                let ref_index = ref_indices[region] as usize;
                let source = list0
                    .get(ref_index)
                    .and_then(|index| references.get(*index))
                    .ok_or(AvcError::Unsupported("P list0 index"))?;
                predict_l0_16x16(
                    source,
                    x,
                    y,
                    motion[region],
                    slice.weights.as_ref(),
                    ref_index,
                )
            };
            let mut block = if let Some(parts) = sub_partitions {
                predict_p_subpartitions(references, &list0, x, y, &parts, slice.weights.as_ref())?
            } else {
                let top = predict(0)?;
                match partition_kind {
                    0 => top,
                    1 => {
                        let bottom = predict(2)?;
                        combine_16x8(top, bottom)
                    }
                    2 => {
                        let right = predict(1)?;
                        combine_8x16(top, right)
                    }
                    3 => {
                        let blocks = [top, predict(1)?, predict(2)?, predict(3)?];
                        combine_8x8(&blocks)
                    }
                    _ => unreachable!(),
                }
            };
            add_luma_residual(&mut block, &luma_residual, qp)?;
            if coded.chroma != 0 {
                add_chroma_residual(
                    &mut block,
                    &dc_levels[0],
                    &ac_levels[0],
                    chroma_qp(qp, pps.core.chroma_qp_index_offset)?,
                    0,
                )?;
                add_chroma_residual(
                    &mut block,
                    &dc_levels[1],
                    &ac_levels[1],
                    chroma_qp(qp, pps.second_chroma_qp_index_offset)?,
                    1,
                )?;
            }
            write_inter_block(&mut picture, x, y, &block);
            let (luma_ac_right, luma_ac_bottom) = match &luma_residual {
                InterLumaResidual::None => ([false; 4], [false; 4]),
                InterLumaResidual::FourByFour(levels) => (
                    [5, 7, 13, 15].map(|i| levels[i].iter().any(|&level| level != 0)),
                    [10, 11, 14, 15].map(|i| levels[i].iter().any(|&level| level != 0)),
                ),
                InterLumaResidual::EightByEight(_) => (
                    [1, 1, 3, 3].map(|region| coded.luma & (1 << region) != 0),
                    [2, 2, 3, 3].map(|region| coded.luma & (1 << region) != 0),
                ),
            };
            states[index] = Some(InterMbState {
                qp,
                skipped,
                intra16: false,
                intra_nxn: false,
                transform8x8,
                modes4: [2; 16],
                modes8: [2; 4],
                motion4: sub_partitions.map_or_else(
                    || std::array::from_fn(|cell| motion[(cell / 8) * 2 + (cell % 4) / 2]),
                    |parts| parts.vectors,
                ),
                mvd4: sub_partitions.map_or_else(
                    || std::array::from_fn(|cell| mvd[(cell / 8) * 2 + (cell % 4) / 2]),
                    |parts| parts.mvd,
                ),
                refs4: sub_partitions.map_or_else(
                    || std::array::from_fn(|cell| ref_indices[(cell / 8) * 2 + (cell % 4) / 2]),
                    |parts| parts.references,
                ),
                coded,
                chroma_mode: 0,
                coded_luma4: coded_luma4_mask(&luma_residual),
                luma_dc_coded: false,
                luma_ac_right,
                luma_ac_bottom,
                chroma_dc_coded: dc_coded,
                chroma_ac_right: std::array::from_fn(|channel| {
                    [1, 3].map(|part| ac_levels[channel][part].iter().any(|&level| level != 0))
                }),
                chroma_ac_bottom: std::array::from_fn(|channel| {
                    [2, 3].map(|part| ac_levels[channel][part].iter().any(|&level| level != 0))
                }),
            });
            let state = states[index].as_ref().expect("decoded P macroblock");
            picture.motion[index] = [0, 3, 12, 15].map(|cell| MotionCell {
                l0: Some((state.refs4[cell], state.motion4[cell])),
                l1: None,
            });
            let ended = decoder.terminate()?;
            if ended != (index + 1 == states.len()) {
                if std::env::var_os("WEBMEDIA_TRACE_P").is_some() {
                    eprintln!(
                        "P slice ended={ended} at MB {index}, bits={}",
                        decoder.consumed_bits()
                    );
                }
                return Err(AvcError::Unsupported(
                    "P picture has multiple or incomplete slices",
                ));
            }
        }
    }
    if !slice.deblocking_disabled {
        let macroblocks: Vec<_> = states
            .iter()
            .map(|state| {
                let state = state.as_ref().expect("decoded P macroblock");
                DeblockMb {
                    qp: state.qp,
                    intra: state.intra16 || state.intra_nxn,
                    transform8x8: state.transform8x8,
                    coded_luma: state.coded_luma4,
                    motion: std::array::from_fn(|cell| MotionCell {
                        l0: Some((list0[state.refs4[cell] as usize] as u8, state.motion4[cell])),
                        l1: None,
                    }),
                }
            })
            .collect();
        filter_inter_picture(
            &mut picture.luma,
            &mut picture.cb,
            &mut picture.cr,
            picture.width,
            &macroblocks,
            [
                pps.core.chroma_qp_index_offset,
                pps.second_chroma_qp_index_offset,
            ],
            slice.alpha_offset,
            slice.beta_offset,
        )?;
    }
    Ok(picture)
}

#[derive(Clone, Copy)]
struct BMbState {
    qp: i32,
    skipped: bool,
    direct: bool,
    direct_regions: [bool; 4],
    intra16: bool,
    intra_nxn: bool,
    modes4: [u8; 16],
    modes8: [u8; 4],
    chroma_mode: u8,
    luma_dc: bool,
    coded: CodedBlockPattern,
    coded_luma4: u16,
    transform8x8: bool,
    mvd4: [[[i32; 2]; 16]; 2],
    motion4: [MotionCell; 16],
    luma_right: [bool; 4],
    luma_bottom: [bool; 4],
    chroma_dc: [bool; 2],
    chroma_right: [[bool; 2]; 2],
    chroma_bottom: [[bool; 2]; 2],
}

fn b_motion_neighbor(
    cx: isize,
    cy: isize,
    cells: &[Option<MotionCell>; 16],
    neighbors: [Option<&[MotionCell; 16]>; 4],
) -> Option<MotionCell> {
    let [left, above, upper_right, upper_left] = neighbors;
    if (0..4).contains(&cx) && (0..4).contains(&cy) {
        cells[cy as usize * 4 + cx as usize]
    } else if cx < 0 && (0..4).contains(&cy) {
        left.map(|motion| motion[cy as usize * 4 + 3])
    } else if cy < 0 && (0..4).contains(&cx) {
        above.map(|motion| motion[12 + cx as usize])
    } else if cx >= 4 && cy < 0 {
        upper_right.map(|motion| motion[12])
    } else if cx < 0 && cy < 0 {
        upper_left.map(|motion| motion[15])
    } else {
        None
    }
}

fn b_intra4_edges(mb: &BMbState, side: bool) -> [u8; 4] {
    if !mb.intra_nxn {
        return [2; 4];
    }
    if mb.transform8x8 {
        let indices = if side { [1, 1, 3, 3] } else { [2, 2, 3, 3] };
        indices.map(|index| mb.modes8[index])
    } else {
        let indices = if side {
            [5, 7, 13, 15]
        } else {
            [10, 11, 14, 15]
        };
        indices.map(|index| mb.modes4[index])
    }
}

fn b_intra8_edges(mb: &BMbState, side: bool) -> [u8; 2] {
    if !mb.intra_nxn {
        [2; 2]
    } else if mb.transform8x8 {
        let indices = if side { [1, 3] } else { [2, 3] };
        indices.map(|index| mb.modes8[index])
    } else {
        let indices = if side { [5, 13] } else { [10, 14] };
        indices.map(|index| mb.modes4[index])
    }
}

/// Decode progressive CABAC B slices with spatial or 8x8 temporal-direct prediction.
/// Explicit bipred weighting remains unsupported.
pub fn decode_cabac_b_2005(
    nal: &[u8],
    sps: &SequenceParameters,
    pps: &PictureParameters2005,
    references: &[Yuv420Picture],
) -> Result<Yuv420Picture, AvcError> {
    if sps.profile_idc != 100
        || sps.chroma_format_idc != 1
        || sps.bit_depth_luma != 8
        || sps.bit_depth_chroma != 8
        || !sps.frame_mbs_only
        || sps.scaling_matrices_present
        || pps.scaling_matrices_present
        || sps.width != sps.width_mbs * 16
        || sps.height == 0
        || sps.height > sps.frame_height_mbs * 16
        || pps.core.weighted_bipred_idc == 1
    {
        return Err(AvcError::Unsupported("B picture format"));
    }
    let slice = parse_cabac_inter_slice(nal, sps, &pps.core)?;
    if std::env::var_os("WEBMEDIA_TRACE_B").is_some() {
        eprintln!(
            "B header: type={} first_mb={} spatial={} refs=({}, {}) reorder=({}, {}) init={} qp={} bytes={}",
            slice.slice_type,
            slice.first_mb,
            slice.direct_spatial_mv_pred,
            slice.ref_idx_l0,
            slice.ref_idx_l1,
            slice.reorder_l0.len(),
            slice.reorder_l1.len(),
            slice.cabac_init_idc,
            slice.slice_qp,
            slice.rbsp.len() - slice.data_byte_offset,
        );
    }
    if slice.slice_type % 5 != 1 || slice.first_mb != 0 {
        return Err(AvcError::Unsupported("B picture slice layout"));
    }
    if !slice.reorder_l0.is_empty() || !slice.reorder_l1.is_empty() {
        return Err(AvcError::Unsupported("B motion or reference reordering"));
    }
    if !slice.direct_spatial_mv_pred && !sps.direct_8x8_inference {
        return Err(AvcError::Unsupported(
            "temporal direct without 8x8 inference",
        ));
    }
    let last_reference = references
        .last()
        .ok_or(AvcError::Unsupported("B reference picture"))?;
    let (poc_msb, poc) = type0_pic_order_count(
        slice.pic_order_cnt_lsb,
        sps.pic_order_cnt_lsb_bits
            .ok_or(AvcError::Unsupported("POC type"))?,
        Some((
            last_reference.pic_order_cnt_msb,
            last_reference.pic_order_cnt_lsb,
        )),
    )?;
    if std::env::var_os("WEBMEDIA_TRACE_B").is_some() {
        eprintln!("B picture poc={poc} poc_lsb={}", slice.pic_order_cnt_lsb);
    }
    let (list0, list1) = b_reference_lists(references, poc);
    if list0.len() < slice.ref_idx_l0 as usize || list1.len() < slice.ref_idx_l1 as usize {
        return Err(AvcError::Unsupported("B reference list length"));
    }
    if references.iter().any(|picture| {
        picture.width != sps.width as usize
            || picture.height != (sps.frame_height_mbs * 16) as usize
    }) {
        return Err(AvcError::Unsupported("B reference dimensions"));
    }
    let width_mbs = sps.width_mbs as usize;
    let height_mbs = sps.frame_height_mbs as usize;
    let mut picture = Yuv420Picture {
        width: sps.width as usize,
        height: (sps.frame_height_mbs * 16) as usize,
        frame_num: slice.frame_num,
        pic_order_cnt_lsb: slice.pic_order_cnt_lsb,
        pic_order_cnt_msb: poc_msb,
        pic_order_cnt: poc,
        luma: vec![0; last_reference.luma.len()],
        cb: vec![0; last_reference.cb.len()],
        cr: vec![0; last_reference.cr.len()],
        motion: vec![[MotionCell::default(); 4]; width_mbs * height_mbs],
        reference_pocs: [
            list0.iter().map(|&i| references[i].pic_order_cnt).collect(),
            list1.iter().map(|&i| references[i].pic_order_cnt).collect(),
        ],
        luma_half: std::sync::OnceLock::new(),
    };
    let mut states: Vec<Option<BMbState>> = vec![None; width_mbs * height_mbs];
    let mut decoder = CabacDecoder::new(&slice.rbsp[slice.data_byte_offset..])?;
    let mut inter = InterMbContexts::new(slice.slice_qp, slice.cabac_init_idc, true)?;
    let mut vectors = MotionVectorContexts::new(slice.slice_qp, slice.cabac_init_idc)?;
    let mut reference_indices = ReferenceIndexContexts::new(slice.slice_qp, slice.cabac_init_idc)?;
    let mut patterns = CodedBlockContexts::new_inter(slice.slice_qp, slice.cabac_init_idc)?;
    let mut chroma_dc_contexts = ChromaDcContexts::new_inter(slice.slice_qp, slice.cabac_init_idc)?;
    let mut chroma_ac_contexts = ChromaAcContexts::new_inter(slice.slice_qp, slice.cabac_init_idc)?;
    let mut luma4_contexts = Luma4x4Contexts::new_inter(slice.slice_qp, slice.cabac_init_idc)?;
    let mut luma8_contexts = Luma8x8Contexts::new_inter(slice.slice_qp, slice.cabac_init_idc)?;
    let mut luma_dc_contexts =
        Luma16x16DcContexts::new_inter(slice.slice_qp, slice.cabac_init_idc)?;
    let mut luma_ac_contexts =
        Luma16x16AcContexts::new_inter(slice.slice_qp, slice.cabac_init_idc)?;
    let mut intra_pred_contexts = IntraPredContexts::new(slice.slice_qp)?;
    let mut transform_contexts =
        Transform8x8Contexts::new_inter(slice.slice_qp, slice.cabac_init_idc)?;
    let mut qp_contexts = MbQpContexts::new(slice.slice_qp)?;
    let mut qp = slice.slice_qp;
    let mut previous_qp_delta_nonzero = false;
    let trace_b = std::env::var_os("WEBMEDIA_TRACE_B").is_some()
        && std::env::var("WEBMEDIA_TRACE_B_POC")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            .is_none_or(|target| target == poc);
    for y in 0..height_mbs {
        let mut type_row = String::new();
        for x in 0..width_mbs {
            let index = y * width_mbs + x;
            let mb_start_bits = if trace_b { decoder.consumed_bits() } else { 0 };
            let left = (x > 0).then(|| states[index - 1].as_ref()).flatten();
            let above = (y > 0)
                .then(|| states[index - width_mbs].as_ref())
                .flatten();
            let motion_neighbors = [
                left.map(|state| &state.motion4),
                above.map(|state| &state.motion4),
                (y > 0 && x + 1 < width_mbs)
                    .then(|| states[index - width_mbs + 1].as_ref())
                    .flatten()
                    .map(|state| &state.motion4),
                (y > 0 && x > 0)
                    .then(|| states[index - width_mbs - 1].as_ref())
                    .flatten()
                    .map(|state| &state.motion4),
            ];
            let skipped = decoder.inter_mb_skip_flag(
                &mut inter,
                left.map(|state| state.skipped),
                above.map(|state| state.skipped),
            )?;
            let kind = if skipped {
                0
            } else {
                decoder.b_inter_mb_type(
                    &mut inter,
                    left.map(|state| state.direct),
                    above.map(|state| state.direct),
                )?
            };
            if trace_b {
                type_row.push(match kind {
                    0 => 'd',
                    1 => '>',
                    2 => '<',
                    3 => 'X',
                    23..=47 => 'i',
                    _ => '*',
                });
            }
            if kind == 23 {
                let transform8x8 = if pps.transform_8x8 {
                    decoder.transform_size_8x8_flag(
                        &mut transform_contexts,
                        left.map(|state| state.transform8x8),
                        above.map(|state| state.transform8x8),
                    )?
                } else {
                    false
                };
                let mut modes4 = [2; 16];
                let mut modes8 = [2; 4];
                if transform8x8 {
                    let mut codes = [IntraPredCode {
                        use_predicted_mode: true,
                        remaining_mode: None,
                    }; 4];
                    for code in &mut codes {
                        *code = decoder.intra_luma_pred_code(&mut intra_pred_contexts)?;
                    }
                    modes8 = intra8x8_modes(
                        &codes,
                        left.map(|state| b_intra8_edges(&state, true)),
                        above.map(|state| b_intra8_edges(&state, false)),
                    )?;
                } else {
                    let mut codes = [IntraPredCode {
                        use_predicted_mode: true,
                        remaining_mode: None,
                    }; 16];
                    for code in &mut codes {
                        *code = decoder.intra_luma_pred_code(&mut intra_pred_contexts)?;
                    }
                    modes4 = intra4x4_modes(
                        &codes,
                        left.map(|state| b_intra4_edges(&state, true)),
                        above.map(|state| b_intra4_edges(&state, false)),
                    )?;
                }
                let chroma_mode = decoder.intra_chroma_pred_mode(
                    &mut intra_pred_contexts,
                    left.map(|state| state.chroma_mode),
                    above.map(|state| state.chroma_mode),
                )?;
                let coded = decoder.coded_block_pattern(
                    &mut patterns,
                    left.map(|state| state.coded),
                    above.map(|state| state.coded),
                )?;
                if coded.luma != 0 || coded.chroma != 0 {
                    let delta = decoder.mb_qp_delta(&mut qp_contexts, previous_qp_delta_nonzero)?;
                    previous_qp_delta_nonzero = delta != 0;
                    qp = (qp + delta).rem_euclid(52);
                } else {
                    previous_qp_delta_nonzero = false;
                }
                let x0 = x * 16;
                let y0 = y * 16;
                let side = left_edge(&picture.luma, picture.width, x0, y0);
                let corner =
                    (x0 > 0 && y0 > 0).then(|| picture.luma[(y0 - 1) * picture.width + x0 - 1]);
                let (luma, luma_right, luma_bottom) = if transform8x8 {
                    let mut blocks = [[0; 64]; 4];
                    for (region, levels) in blocks.iter_mut().enumerate() {
                        if coded.luma & (1 << region) != 0 {
                            *levels = decoder.luma8x8_coefficients(&mut luma8_contexts)?;
                        }
                    }
                    let luma = reconstruct_intra8x8_luma(
                        &modes8,
                        &blocks,
                        qp,
                        &[16; 64],
                        upper_edge(&picture.luma, picture.width, x0, y0),
                        side,
                        corner,
                    )?;
                    let edge = |region: usize| blocks[region].iter().any(|&level| level != 0);
                    (
                        luma,
                        [edge(1), edge(1), edge(3), edge(3)],
                        [edge(2), edge(2), edge(3), edge(3)],
                    )
                } else {
                    let blocks = decoder.luma4x4_macroblock(
                        &mut luma4_contexts,
                        coded.luma,
                        left.map(|state| state.luma_right),
                        above.map(|state| state.luma_bottom),
                    )?;
                    let luma = reconstruct_intra4x4_luma(
                        &modes4,
                        &blocks,
                        qp,
                        upper_edge(&picture.luma, picture.width, x0, y0),
                        side,
                        corner,
                    )?;
                    let edge = |block: usize| blocks[block].iter().any(|&level| level != 0);
                    (luma, [5, 7, 13, 15].map(edge), [10, 11, 14, 15].map(edge))
                };
                let mut chroma_dc = [[0; 4]; 2];
                if coded.chroma != 0 {
                    for (channel, levels) in chroma_dc.iter_mut().enumerate() {
                        *levels = decoder.chroma_dc_coefficients(
                            &mut chroma_dc_contexts,
                            left.map(|state| state.chroma_dc[channel]),
                            above.map(|state| state.chroma_dc[channel]),
                        )?;
                    }
                }
                let mut chroma_ac = [[[0; 15]; 4]; 2];
                if coded.chroma == 2 {
                    for (channel, blocks) in chroma_ac.iter_mut().enumerate() {
                        *blocks = decoder.chroma_ac_macroblock(
                            &mut chroma_ac_contexts,
                            left.map(|state| state.chroma_right[channel]),
                            above.map(|state| state.chroma_bottom[channel]),
                        )?;
                    }
                }
                let cx0 = x * 8;
                let cy0 = y * 8;
                let chroma_width = picture.width / 2;
                let cb = reconstruct_chroma(
                    chroma_mode,
                    &chroma_dc[0],
                    &chroma_ac[0],
                    chroma_qp(qp, pps.core.chroma_qp_index_offset)?,
                    upper_edge(&picture.cb, chroma_width, cx0, cy0),
                    left_edge(&picture.cb, chroma_width, cx0, cy0),
                    (cx0 > 0 && cy0 > 0).then(|| picture.cb[(cy0 - 1) * chroma_width + cx0 - 1]),
                )?;
                let cr = reconstruct_chroma(
                    chroma_mode,
                    &chroma_dc[1],
                    &chroma_ac[1],
                    chroma_qp(qp, pps.second_chroma_qp_index_offset)?,
                    upper_edge(&picture.cr, chroma_width, cx0, cy0),
                    left_edge(&picture.cr, chroma_width, cx0, cy0),
                    (cx0 > 0 && cy0 > 0).then(|| picture.cr[(cy0 - 1) * chroma_width + cx0 - 1]),
                )?;
                write_inter_block(&mut picture, x, y, &InterMacroblock { luma, cb, cr });
                states[index] = Some(BMbState {
                    qp,
                    skipped: false,
                    direct: false,
                    direct_regions: [false; 4],
                    intra16: false,
                    intra_nxn: true,
                    modes4,
                    modes8,
                    chroma_mode,
                    luma_dc: false,
                    coded,
                    transform8x8,
                    coded_luma4: 0,
                    mvd4: [[[0; 2]; 16]; 2],
                    motion4: [MotionCell::default(); 16],
                    luma_right,
                    luma_bottom,
                    chroma_dc: chroma_dc.map(|levels| levels.iter().any(|&level| level != 0)),
                    chroma_right: std::array::from_fn(|channel| {
                        [1, 3]
                            .map(|block| chroma_ac[channel][block].iter().any(|&level| level != 0))
                    }),
                    chroma_bottom: std::array::from_fn(|channel| {
                        [2, 3]
                            .map(|block| chroma_ac[channel][block].iter().any(|&level| level != 0))
                    }),
                });
                let ended = decoder.terminate()?;
                if ended != (index + 1 == states.len()) {
                    if trace_b {
                        eprintln!(
                            "B intra-nxn termination at {index}: ended={ended} bits={}",
                            decoder.consumed_bits()
                        );
                    }
                    return Err(AvcError::Unsupported(
                        "B picture has multiple or incomplete slices",
                    ));
                }
                continue;
            }
            if (24..=47).contains(&kind) {
                let intra_kind = kind - 23;
                let coded = CodedBlockPattern {
                    luma: if intra_kind >= 13 { 15 } else { 0 },
                    chroma: ((intra_kind - 1) / 4) % 3,
                    pcm: false,
                };
                let chroma_mode = decoder.intra_chroma_pred_mode(
                    &mut intra_pred_contexts,
                    left.map(|state| state.chroma_mode),
                    above.map(|state| state.chroma_mode),
                )?;
                let delta = decoder.mb_qp_delta(&mut qp_contexts, previous_qp_delta_nonzero)?;
                previous_qp_delta_nonzero = delta != 0;
                qp = (qp + delta).rem_euclid(52);
                let luma_dc = decoder.luma16x16_dc_coefficients(
                    &mut luma_dc_contexts,
                    left.map(|state| state.intra16 && state.luma_dc),
                    above.map(|state| state.intra16 && state.luma_dc),
                )?;
                let luma_ac = if coded.luma != 0 {
                    decoder.luma16x16_ac_macroblock(
                        &mut luma_ac_contexts,
                        left.map(|state| state.luma_right),
                        above.map(|state| state.luma_bottom),
                    )?
                } else {
                    [[0; 15]; 16]
                };
                let mut chroma_dc = [[0; 4]; 2];
                if coded.chroma != 0 {
                    for channel in 0..2 {
                        chroma_dc[channel] = decoder.chroma_dc_coefficients(
                            &mut chroma_dc_contexts,
                            left.map(|state| state.chroma_dc[channel]),
                            above.map(|state| state.chroma_dc[channel]),
                        )?;
                    }
                }
                let mut chroma_ac = [[[0; 15]; 4]; 2];
                if coded.chroma == 2 {
                    for channel in 0..2 {
                        chroma_ac[channel] = decoder.chroma_ac_macroblock(
                            &mut chroma_ac_contexts,
                            left.map(|state| state.chroma_right[channel]),
                            above.map(|state| state.chroma_bottom[channel]),
                        )?;
                    }
                }
                let x0 = x * 16;
                let y0 = y * 16;
                let cx0 = x * 8;
                let cy0 = y * 8;
                let chroma_width = picture.width / 2;
                let luma = reconstruct_intra16x16_luma(
                    (intra_kind - 1) % 4,
                    &luma_dc,
                    &luma_ac,
                    qp,
                    upper_edge(&picture.luma, picture.width, x0, y0),
                    left_edge(&picture.luma, picture.width, x0, y0),
                    (x0 > 0 && y0 > 0).then(|| picture.luma[(y0 - 1) * picture.width + x0 - 1]),
                )?;
                let cb = reconstruct_chroma(
                    chroma_mode,
                    &chroma_dc[0],
                    &chroma_ac[0],
                    chroma_qp(qp, pps.core.chroma_qp_index_offset)?,
                    upper_edge(&picture.cb, chroma_width, cx0, cy0),
                    left_edge(&picture.cb, chroma_width, cx0, cy0),
                    (cx0 > 0 && cy0 > 0).then(|| picture.cb[(cy0 - 1) * chroma_width + cx0 - 1]),
                )?;
                let cr = reconstruct_chroma(
                    chroma_mode,
                    &chroma_dc[1],
                    &chroma_ac[1],
                    chroma_qp(qp, pps.second_chroma_qp_index_offset)?,
                    upper_edge(&picture.cr, chroma_width, cx0, cy0),
                    left_edge(&picture.cr, chroma_width, cx0, cy0),
                    (cx0 > 0 && cy0 > 0).then(|| picture.cr[(cy0 - 1) * chroma_width + cx0 - 1]),
                )?;
                write_inter_block(&mut picture, x, y, &InterMacroblock { luma, cb, cr });
                states[index] = Some(BMbState {
                    qp,
                    skipped: false,
                    direct: false,
                    direct_regions: [false; 4],
                    intra16: true,
                    intra_nxn: false,
                    modes4: [2; 16],
                    modes8: [2; 4],
                    chroma_mode,
                    luma_dc: luma_dc.iter().any(|&level| level != 0),
                    coded,
                    transform8x8: false,
                    coded_luma4: 0,
                    mvd4: [[[0; 2]; 16]; 2],
                    motion4: [MotionCell::default(); 16],
                    luma_right: [5, 7, 13, 15]
                        .map(|part| luma_ac[part].iter().any(|&level| level != 0)),
                    luma_bottom: [10, 11, 14, 15]
                        .map(|part| luma_ac[part].iter().any(|&level| level != 0)),
                    chroma_dc: chroma_dc.map(|levels| levels.iter().any(|&level| level != 0)),
                    chroma_right: std::array::from_fn(|channel| {
                        [1, 3].map(|part| chroma_ac[channel][part].iter().any(|&level| level != 0))
                    }),
                    chroma_bottom: std::array::from_fn(|channel| {
                        [2, 3].map(|part| chroma_ac[channel][part].iter().any(|&level| level != 0))
                    }),
                });
                let ended = decoder.terminate()?;
                if ended != (index + 1 == states.len()) {
                    if trace_b {
                        eprintln!("B intra termination at {index}: ended={ended}");
                    }
                    return Err(AvcError::Unsupported(
                        "B picture has multiple or incomplete slices",
                    ));
                }
                continue;
            }
            if kind > 22 {
                if trace_b {
                    eprintln!("B macroblock {index} ({x},{y}) has unsupported type {kind}");
                }
                return Err(AvcError::Unsupported("B macroblock partition mode"));
            }
            let mut motion = [MotionCell::default(); 4];
            let mut direct_regions = [kind == 0; 4];
            let mut mvd = [[[0; 2]; 4]; 2];
            let mut mvd4 = [[[0; 2]; 16]; 2];
            let mut sub_motion4 = None;
            let mut decoded_motion4 = None;
            let mut no_small_subpart = kind != 0 || sps.direct_8x8_inference;
            if kind == 0 {
                let left_cell = (x > 0).then(|| picture.motion[index - 1][1]);
                let above_cell = (y > 0).then(|| picture.motion[index - width_mbs][2]);
                let upper_right =
                    (y > 0 && x + 1 < width_mbs).then(|| picture.motion[index - width_mbs + 1][2]);
                let upper_left = (y > 0 && x > 0).then(|| picture.motion[index - width_mbs - 1][3]);
                for region in 0..4 {
                    let colocated = references[list1[0]].motion[index][region];
                    motion[region] = if slice.direct_spatial_mv_pred {
                        spatial_direct_motion(
                            left_cell,
                            above_cell,
                            upper_right,
                            upper_left,
                            colocated,
                            true,
                        )
                    } else {
                        temporal_direct_motion(
                            colocated,
                            &references[list1[0]].reference_pocs,
                            &picture.reference_pocs[0],
                            poc,
                            references[list1[0]].pic_order_cnt,
                        )?
                    };
                }
            } else if kind == 22 {
                let mut subtypes = [0u8; 4];
                let mut submodes = [0u8; 4];
                let mut masks = [0u8; 4];
                for part in 0..4 {
                    subtypes[part] = decoder.b_sub_mb_type(&mut inter)?;
                    (submodes[part], masks[part]) = b_sub_partition_mode(subtypes[part])?;
                    direct_regions[part] = subtypes[part] == 0;
                }
                no_small_subpart = submodes.iter().all(|&mode| mode == 0)
                    && (sps.direct_8x8_inference || !subtypes.contains(&0));
                if trace_b && submodes.iter().any(|&mode| mode != 0) {
                    eprintln!(
                        "B subtypes at MB {index} ({x},{y}): {subtypes:?} bits={}",
                        decoder.consumed_bits()
                    );
                }
                let mut refs = [[0u8; 4]; 2];
                for list in 0..2 {
                    let active = if list == 0 {
                        slice.ref_idx_l0
                    } else {
                        slice.ref_idx_l1
                    };
                    for part in 0..4 {
                        if masks[part] & (1 << list) == 0 {
                            continue;
                        }
                        let left_ref = if part & 1 != 0 {
                            Some(refs[list][part - 1])
                        } else if x > 0 && !left.is_some_and(|state| state.direct_regions[part + 1])
                        {
                            let cell = picture.motion[index - 1][part + 1];
                            (if list == 0 { cell.l0 } else { cell.l1 }).map(|(r, _)| r)
                        } else {
                            None
                        };
                        let above_ref = if part >= 2 {
                            Some(refs[list][part - 2])
                        } else if y > 0
                            && !above.is_some_and(|state| state.direct_regions[part + 2])
                        {
                            let cell = picture.motion[index - width_mbs][part + 2];
                            (if list == 0 { cell.l0 } else { cell.l1 }).map(|(r, _)| r)
                        } else {
                            None
                        };
                        if active > 1 {
                            refs[list][part] = decoder.reference_index(
                                &mut reference_indices,
                                left_ref,
                                above_ref,
                                active,
                            ).map_err(|error| {
                                if trace_b {
                                    eprintln!(
                                        "B sub-ref error at MB {index} ({x},{y}) part={part} list={list} subtypes={subtypes:?} refs={refs:?} neighbors=({left_ref:?},{above_ref:?}) direct=({:?},{:?}) active={active} bits={}: {error:?}",
                                        left.map(|state| state.direct),
                                        above.map(|state| state.direct),
                                        decoder.consumed_bits()
                                    );
                                }
                                error
                            })?;
                        }
                    }
                }
                let mut deltas = [[[[0; 2]; 4]; 4]; 2];
                let mut mvd_cells: [[Option<[i32; 2]>; 16]; 2] = [[None; 16]; 2];
                for part in 0..4 {
                    if masks[part] == 0 {
                        let base = part / 2 * 8 + part % 2 * 2;
                        for offset in [0, 1, 4, 5] {
                            mvd_cells[0][base + offset] = Some([0, 0]);
                            mvd_cells[1][base + offset] = Some([0, 0]);
                        }
                    }
                }
                for list in 0..2 {
                    for part in 0..4 {
                        if masks[part] & (1 << list) == 0 {
                            continue;
                        }
                        let mode = submodes[part];
                        let count = [1, 2, 2, 4][mode as usize];
                        let base_x = part % 2 * 2;
                        let base_y = part / 2 * 2;
                        for sub in 0..count {
                            let (dx, dy, width, height) = p_sub_partition_rect(mode, sub);
                            let sx = base_x + dx;
                            let sy = base_y + dy;
                            let left_mvd = if sx > 0 {
                                mvd_cells[list][sy * 4 + sx - 1]
                            } else {
                                left.map(|state| state.mvd4[list][sy * 4 + 3])
                            };
                            let above_mvd = if sy > 0 {
                                mvd_cells[list][(sy - 1) * 4 + sx]
                            } else {
                                above.map(|state| state.mvd4[list][12 + sx])
                            };
                            let delta = [
                                decoder.motion_vector_difference(
                                    &mut vectors,
                                    0,
                                    left_mvd.map(|value| value[0]),
                                    above_mvd.map(|value| value[0]),
                                )?,
                                decoder.motion_vector_difference(
                                    &mut vectors,
                                    1,
                                    left_mvd.map(|value| value[1]),
                                    above_mvd.map(|value| value[1]),
                                )?,
                            ];
                            deltas[list][part][sub] = delta;
                            for row in sy..sy + height {
                                for col in sx..sx + width {
                                    mvd_cells[list][row * 4 + col] = Some(delta);
                                }
                            }
                        }
                        mvd[list][part] = deltas[list][part][0];
                    }
                    mvd4[list] = mvd_cells[list].map(|value| value.unwrap_or([0, 0]));
                }
                let mut cells = [None; 16];
                for part in 0..4 {
                    let base_x = part % 2 * 2;
                    let base_y = part / 2 * 2;
                    let count = [1, 2, 2, 4][submodes[part] as usize];
                    for sub in 0..count {
                        let (dx, dy, width, height) = p_sub_partition_rect(submodes[part], sub);
                        let sx = base_x + dx;
                        let sy = base_y + dy;
                        let neighbor = |cx: isize, cy: isize, cells: &[Option<MotionCell>; 16]| {
                            b_motion_neighbor(cx, cy, cells, motion_neighbors)
                        };
                        let a = neighbor(sx as isize - 1, sy as isize, &cells);
                        let b = neighbor(sx as isize, sy as isize - 1, &cells);
                        let c = neighbor((sx + width) as isize, sy as isize - 1, &cells);
                        let d = neighbor(sx as isize - 1, sy as isize - 1, &cells);
                        let mut cell = MotionCell::default();
                        if masks[part] == 0 {
                            // 8.4.1.2.2 uses mbPartIdx=0 even for B_Direct_8x8:
                            // all spatial-direct regions share the macroblock predictor.
                            cell = if slice.direct_spatial_mv_pred {
                                spatial_direct_motion(
                                    motion_neighbors[0].map(|motion| motion[3]),
                                    motion_neighbors[1].map(|motion| motion[12]),
                                    motion_neighbors[2].map(|motion| motion[12]),
                                    motion_neighbors[3].map(|motion| motion[15]),
                                    references[list1[0]].motion[index][part],
                                    true,
                                )
                            } else {
                                temporal_direct_motion(
                                    references[list1[0]].motion[index][part],
                                    &references[list1[0]].reference_pocs,
                                    &picture.reference_pocs[0],
                                    poc,
                                    references[list1[0]].pic_order_cnt,
                                )?
                            };
                        } else {
                            for list in 0..2 {
                                if masks[part] & (1 << list) == 0 {
                                    continue;
                                }
                                let from = |cell: MotionCell| {
                                    if list == 0 { cell.l0 } else { cell.l1 }
                                        .unwrap_or((u8::MAX, [0; 2]))
                                };
                                let reference = refs[list][part];
                                let predicted = motion_predictor_for_ref(
                                    a.map(from),
                                    b.map(from),
                                    c.map(from),
                                    d.map(from),
                                    reference,
                                );
                                let delta = deltas[list][part][sub];
                                let vector = [
                                    predicted[0].saturating_add(delta[0]),
                                    predicted[1].saturating_add(delta[1]),
                                ];
                                if list == 0 {
                                    cell.l0 = Some((reference, vector));
                                } else {
                                    cell.l1 = Some((reference, vector));
                                }
                            }
                        }
                        for row in sy..sy + height {
                            for col in sx..sx + width {
                                cells[row * 4 + col] = Some(cell);
                            }
                        }
                    }
                    motion[part] = cells[base_y * 4 + base_x].unwrap();
                }
                let cells = cells.map(|cell| cell.expect("decoded B subpartition"));
                decoded_motion4 = Some(cells);
                if submodes.iter().any(|&mode| mode != 0) {
                    sub_motion4 = Some(cells);
                }
            } else {
                let (vertical, modes): (bool, [u8; 2]) = match kind {
                    1 => (false, [1, 0]),
                    2 => (false, [2, 0]),
                    3 => (false, [3, 0]),
                    4..=21 => {
                        let pair = (kind - 4) / 2;
                        (
                            kind % 2 != 0,
                            [
                                [1, 2, 1, 2, 1, 2, 3, 3, 3][pair as usize],
                                [1, 2, 2, 1, 3, 3, 1, 2, 3][pair as usize],
                            ],
                        )
                    }
                    _ => unreachable!(),
                };
                let partition_count = if kind <= 3 { 1 } else { 2 };
                let partitions = if vertical {
                    [[0, 2, 0, 0], [1, 3, 0, 0]]
                } else {
                    [[0, 1, 2, 3], [2, 3, 0, 0]]
                };
                let mut selected_refs = [[0u8; 4]; 2];
                for list in 0..2 {
                    let active_count = if list == 0 {
                        slice.ref_idx_l0
                    } else {
                        slice.ref_idx_l1
                    };
                    for part in 0..partition_count {
                        if modes[part] & (1 << list) == 0 {
                            continue;
                        }
                        let left_ref = if part == 1 && vertical {
                            Some(selected_refs[list][0])
                        } else if left.is_some_and(|state| {
                            state.direct_regions[if part == 1 { 3 } else { 1 }]
                        }) {
                            None
                        } else if x > 0 {
                            let cell = picture.motion[index - 1][if part == 1 { 3 } else { 1 }];
                            if list == 0 { cell.l0 } else { cell.l1 }.map(|(index, _)| index)
                        } else {
                            None
                        };
                        let above_ref = if part == 1 && !vertical {
                            Some(selected_refs[list][0])
                        } else if above.is_some_and(|state| {
                            state.direct_regions[if part == 1 { 3 } else { 2 }]
                        }) {
                            None
                        } else if y > 0 {
                            let cell =
                                picture.motion[index - width_mbs][if part == 1 { 3 } else { 2 }];
                            if list == 0 { cell.l0 } else { cell.l1 }.map(|(index, _)| index)
                        } else {
                            None
                        };
                        let ref_index = if active_count > 1 {
                            decoder.reference_index(
                                &mut reference_indices,
                                left_ref,
                                above_ref,
                                active_count,
                            )?
                        } else {
                            0
                        };
                        let count = if partition_count == 1 { 4 } else { 2 };
                        for &region in &partitions[part][..count] {
                            selected_refs[list][region] = ref_index;
                        }
                    }
                }
                for list in 0..2 {
                    for part in 0..partition_count {
                        if modes[part] & (1 << list) == 0 {
                            continue;
                        }
                        let left_cell = if part == 1 && vertical {
                            Some(motion[0])
                        } else if x > 0 {
                            left.map(|state| {
                                state.motion4[if part == 1 && !vertical { 11 } else { 3 }]
                            })
                        } else {
                            None
                        };
                        let above_cell = if part == 1 && !vertical {
                            Some(motion[0])
                        } else if y > 0 {
                            above.map(|state| {
                                state.motion4[if part == 1 && vertical { 14 } else { 12 }]
                            })
                        } else {
                            None
                        };
                        let sx = if vertical { part as isize * 2 } else { 0 };
                        let sy = if !vertical { part as isize * 2 } else { 0 };
                        let width = if vertical { 2 } else { 4 };
                        let right_cell =
                            b_motion_neighbor(sx + width, sy - 1, &[None; 16], motion_neighbors);
                        let upper_left_cell =
                            b_motion_neighbor(sx - 1, sy - 1, &[None; 16], motion_neighbors);
                        let from = |cell: MotionCell| {
                            if list == 0 { cell.l0 } else { cell.l1 }.unwrap_or((u8::MAX, [0; 2]))
                        };
                        let ref_index = selected_refs[list][partitions[part][0]];
                        let left_motion = left_cell.map(from);
                        let above_motion = above_cell.map(from);
                        // 8.4.1.3.2 substitutes D for unavailable C before directional prediction.
                        let right_motion = right_cell.map(from).or(upper_left_cell.map(from));
                        let upper_left_motion = upper_left_cell.map(from);
                        let directional = |candidate: Option<(u8, [i32; 2])>| {
                            candidate
                                .filter(|(index, _)| *index == ref_index)
                                .map(|(_, vector)| vector)
                        };
                        let median_prediction = || {
                            motion_predictor_for_ref(
                                left_motion,
                                above_motion,
                                right_motion,
                                upper_left_motion,
                                ref_index,
                            )
                        };
                        let left_mvd = if part == 1 && vertical {
                            Some(mvd[list][0])
                        } else {
                            left.map(|state| state.mvd4[list][if part == 1 { 11 } else { 3 }])
                        };
                        let above_mvd = if part == 1 && !vertical {
                            Some(mvd[list][0])
                        } else {
                            above.map(|state| state.mvd4[list][if part == 1 { 14 } else { 12 }])
                        };
                        let difference = [
                            decoder.motion_vector_difference(
                                &mut vectors,
                                0,
                                left_mvd.map(|value| value[0]),
                                above_mvd.map(|value| value[0]),
                            )?,
                            decoder.motion_vector_difference(
                                &mut vectors,
                                1,
                                left_mvd.map(|value| value[1]),
                                above_mvd.map(|value| value[1]),
                            )?,
                        ];
                        let predicted = if partition_count == 2 && !vertical && part == 0 {
                            directional(above_motion).unwrap_or_else(median_prediction)
                        } else if partition_count == 2 && !vertical && part == 1 {
                            directional(left_motion).unwrap_or_else(median_prediction)
                        } else if partition_count == 2 && vertical && part == 0 {
                            directional(left_motion).unwrap_or_else(median_prediction)
                        } else if partition_count == 2 && vertical && part == 1 {
                            directional(right_motion).unwrap_or_else(median_prediction)
                        } else {
                            median_prediction()
                        };
                        let vector = [
                            predicted[0].saturating_add(difference[0]),
                            predicted[1].saturating_add(difference[1]),
                        ];
                        let region_count = if partition_count == 1 { 4 } else { 2 };
                        for &region in &partitions[part][..region_count] {
                            mvd[list][region] = difference;
                            if list == 0 {
                                motion[region].l0 = Some((ref_index, vector));
                            } else {
                                motion[region].l1 = Some((ref_index, vector));
                            }
                        }
                    }
                }
            }
            if kind != 22 {
                for list in 0..2 {
                    for cell in 0..16 {
                        let region = cell / 8 * 2 + cell % 4 / 2;
                        mvd4[list][cell] = mvd[list][region];
                    }
                }
            }
            let coded = if skipped {
                CodedBlockPattern {
                    luma: 0,
                    chroma: 0,
                    pcm: false,
                }
            } else {
                decoder.coded_block_pattern(
                    &mut patterns,
                    left.map(|state| state.coded),
                    above.map(|state| state.coded),
                )?
            };
            let transform8x8 = if coded.luma != 0 && pps.transform_8x8 && no_small_subpart {
                decoder.transform_size_8x8_flag(
                    &mut transform_contexts,
                    left.map(|state| state.transform8x8),
                    above.map(|state| state.transform8x8),
                )?
            } else {
                false
            };
            if coded.luma != 0 || coded.chroma != 0 {
                let delta = decoder.mb_qp_delta(&mut qp_contexts, previous_qp_delta_nonzero).map_err(|error| {
                    if trace_b {
                        eprintln!("B QP error at MB {index} ({x},{y}) kind={kind} coded={coded:?} transform8x8={transform8x8} bits={}: {error:?}", decoder.consumed_bits());
                    }
                    error
                })?;
                previous_qp_delta_nonzero = delta != 0;
                qp = (qp + delta).rem_euclid(52);
            } else {
                previous_qp_delta_nonzero = false;
            }
            let luma_residual = if coded.luma == 0 {
                InterLumaResidual::None
            } else if transform8x8 {
                let mut blocks = [[0; 64]; 4];
                for (region, levels) in blocks.iter_mut().enumerate() {
                    if coded.luma & (1 << region) != 0 {
                        *levels = decoder.luma8x8_coefficients(&mut luma8_contexts)?;
                    }
                }
                InterLumaResidual::EightByEight(blocks)
            } else {
                InterLumaResidual::FourByFour(decoder.luma4x4_macroblock(
                    &mut luma4_contexts,
                    coded.luma,
                    Some(left.map_or([false; 4], |state| state.luma_right)),
                    Some(above.map_or([false; 4], |state| state.luma_bottom)),
                )?)
            };
            let mut dc_levels = [[0; 4]; 2];
            let mut ac_levels = [[[0; 15]; 4]; 2];
            if coded.chroma != 0 {
                for channel in 0..2 {
                    dc_levels[channel] = decoder.chroma_dc_coefficients(
                        &mut chroma_dc_contexts,
                        Some(left.is_some_and(|state| state.chroma_dc[channel])),
                        Some(above.is_some_and(|state| state.chroma_dc[channel])),
                    )?;
                }
            }
            if coded.chroma == 2 {
                for channel in 0..2 {
                    ac_levels[channel] = decoder.chroma_ac_macroblock(
                        &mut chroma_ac_contexts,
                        Some(left.map_or([false; 2], |state| state.chroma_right[channel])),
                        Some(above.map_or([false; 2], |state| state.chroma_bottom[channel])),
                    )?;
                }
            }
            let mut block = if let Some(cells) = sub_motion4 {
                predict_b_subpartition_block(
                    &cells,
                    references,
                    &list0,
                    &list1,
                    x,
                    y,
                    poc,
                    pps.core.weighted_bipred_idc,
                )?
            } else if motion.iter().all(|&cell| cell == motion[0]) {
                let list0_block = if let Some((ref_index, vector)) = motion[0].l0 {
                    let source = list0
                        .get(ref_index as usize)
                        .ok_or(AvcError::Unsupported("B list0 index"))?;
                    Some(predict_inter_16x16(
                        &references[*source],
                        x,
                        y,
                        vector,
                        None,
                        0,
                        ref_index as usize,
                    )?)
                } else {
                    None
                };
                let list1_block = if let Some((ref_index, vector)) = motion[0].l1 {
                    let source = list1
                        .get(ref_index as usize)
                        .ok_or(AvcError::Unsupported("B list1 index"))?;
                    Some(predict_inter_16x16(
                        &references[*source],
                        x,
                        y,
                        vector,
                        None,
                        1,
                        ref_index as usize,
                    )?)
                } else {
                    None
                };
                match (list0_block, list1_block) {
                    (Some(block0), Some(block1)) => {
                        let weights = if pps.core.weighted_bipred_idc == 2 {
                            let source0 = list0[motion[0].l0.unwrap().0 as usize];
                            let source1 = list1[motion[0].l1.unwrap().0 as usize];
                            Some(implicit_b_weights(
                                poc,
                                references[source0].pic_order_cnt,
                                references[source1].pic_order_cnt,
                            ))
                        } else {
                            None
                        };
                        blend_b_macroblocks(&block0, &block1, weights)
                    }
                    (Some(block), None) | (None, Some(block)) => block,
                    (None, None) => {
                        return Err(AvcError::InvalidData("B block without prediction"));
                    }
                }
            } else {
                let mut predicted = [InterRegion {
                    luma: [0; 64],
                    cb: [0; 16],
                    cr: [0; 16],
                }; 4];
                for region in 0..4 {
                    let list0_block = if let Some((ref_index, vector)) = motion[region].l0 {
                        let source = list0
                            .get(ref_index as usize)
                            .ok_or(AvcError::Unsupported("B list0 index"))?;
                        Some(predict_inter_8x8_region(
                            &references[*source],
                            x,
                            y,
                            vector,
                            region,
                        )?)
                    } else {
                        None
                    };
                    let list1_block = if let Some((ref_index, vector)) = motion[region].l1 {
                        let source = list1
                            .get(ref_index as usize)
                            .ok_or(AvcError::Unsupported("B list1 index"))?;
                        Some(predict_inter_8x8_region(
                            &references[*source],
                            x,
                            y,
                            vector,
                            region,
                        )?)
                    } else {
                        None
                    };
                    predicted[region] = match (list0_block, list1_block) {
                        (Some(block0), Some(block1)) => {
                            let weights = if pps.core.weighted_bipred_idc == 2 {
                                let source0 = list0[motion[region].l0.unwrap().0 as usize];
                                let source1 = list1[motion[region].l1.unwrap().0 as usize];
                                Some(implicit_b_weights(
                                    poc,
                                    references[source0].pic_order_cnt,
                                    references[source1].pic_order_cnt,
                                ))
                            } else {
                                None
                            };
                            blend_b_compact(&block0, &block1, weights)
                        }
                        (Some(block), None) | (None, Some(block)) => block,
                        (None, None) => {
                            return Err(AvcError::InvalidData("B block without prediction"));
                        }
                    };
                }
                assemble_inter_regions(&predicted)
            };
            add_luma_residual(&mut block, &luma_residual, qp)?;
            if coded.chroma != 0 {
                add_chroma_residual(
                    &mut block,
                    &dc_levels[0],
                    &ac_levels[0],
                    chroma_qp(qp, pps.core.chroma_qp_index_offset)?,
                    0,
                )?;
                add_chroma_residual(
                    &mut block,
                    &dc_levels[1],
                    &ac_levels[1],
                    chroma_qp(qp, pps.second_chroma_qp_index_offset)?,
                    1,
                )?;
            }
            write_inter_block(&mut picture, x, y, &block);
            let motion4 = decoded_motion4
                .unwrap_or_else(|| std::array::from_fn(|cell| motion[cell / 8 * 2 + cell % 4 / 2]));
            picture.motion[index] = [0, 3, 12, 15].map(|cell| motion4[cell]);
            let (luma_right, luma_bottom) = match &luma_residual {
                InterLumaResidual::None => ([false; 4], [false; 4]),
                InterLumaResidual::FourByFour(levels) => (
                    [5, 7, 13, 15].map(|part| levels[part].iter().any(|&level| level != 0)),
                    [10, 11, 14, 15].map(|part| levels[part].iter().any(|&level| level != 0)),
                ),
                InterLumaResidual::EightByEight(_) => (
                    [1, 1, 3, 3].map(|region| coded.luma & (1 << region) != 0),
                    [2, 2, 3, 3].map(|region| coded.luma & (1 << region) != 0),
                ),
            };
            states[index] = Some(BMbState {
                qp,
                skipped,
                direct: kind == 0,
                direct_regions,
                intra16: false,
                intra_nxn: false,
                modes4: [2; 16],
                modes8: [2; 4],
                chroma_mode: 0,
                luma_dc: false,
                coded,
                transform8x8,
                coded_luma4: coded_luma4_mask(&luma_residual),
                mvd4,
                motion4,
                luma_right,
                luma_bottom,
                chroma_dc: dc_levels.map(|levels| levels.iter().any(|&level| level != 0)),
                chroma_right: std::array::from_fn(|channel| {
                    [1, 3].map(|part| ac_levels[channel][part].iter().any(|&level| level != 0))
                }),
                chroma_bottom: std::array::from_fn(|channel| {
                    [2, 3].map(|part| ac_levels[channel][part].iter().any(|&level| level != 0))
                }),
            });
            if trace_b && !skipped && std::env::var_os("WEBMEDIA_TRACE_B_MB").is_some() {
                eprintln!(
                    "B mb {index} ({x},{y}) kind={kind} skipped={skipped} qp={qp} cbp=({}, {}) refs={motion:?} bits={mb_start_bits}..{}",
                    coded.luma,
                    coded.chroma,
                    decoder.consumed_bits()
                );
            }
            let ended = decoder.terminate()?;
            if ended != (index + 1 == states.len()) {
                if trace_b {
                    eprintln!(
                        "B inter termination at {index}: kind={kind} skipped={skipped} ended={ended} bits={}/{}",
                        decoder.consumed_bits(),
                        (slice.rbsp.len() - slice.data_byte_offset) * 8
                    );
                }
                return Err(AvcError::Unsupported(
                    "B picture has multiple or incomplete slices",
                ));
            }
        }
        if trace_b && slice.ref_idx_l1 > 1 {
            eprintln!("B type row {y}: {type_row}");
        }
    }
    if !slice.deblocking_disabled {
        let macroblocks: Vec<_> = states
            .iter()
            .map(|state| {
                let state = state.as_ref().expect("decoded B macroblock");
                DeblockMb {
                    qp: state.qp,
                    intra: state.intra16 || state.intra_nxn,
                    transform8x8: state.transform8x8,
                    coded_luma: state.coded_luma4,
                    motion: state.motion4.map(|cell| MotionCell {
                        l0: cell.l0.map(|(r, mv)| (list0[r as usize] as u8, mv)),
                        l1: cell.l1.map(|(r, mv)| (list1[r as usize] as u8, mv)),
                    }),
                }
            })
            .collect();
        filter_inter_picture(
            &mut picture.luma,
            &mut picture.cb,
            &mut picture.cr,
            picture.width,
            &macroblocks,
            [
                pps.core.chroma_qp_index_offset,
                pps.second_chroma_qp_index_offset,
            ],
            slice.alpha_offset,
            slice.beta_offset,
        )?;
    }
    Ok(picture)
}

pub struct InterMacroblock {
    pub luma: [u8; 256],
    pub cb: [u8; 64],
    pub cr: [u8; 64],
}

#[derive(Clone, Copy)]
struct InterRegion {
    luma: [u8; 64],
    cb: [u8; 16],
    cr: [u8; 16],
}

fn assemble_inter_regions(regions: &[InterRegion; 4]) -> InterMacroblock {
    let mut block = InterMacroblock {
        luma: [0; 256],
        cb: [0; 64],
        cr: [0; 64],
    };
    for (region, part) in regions.iter().enumerate() {
        let luma_x = region % 2 * 8;
        let luma_y = region / 2 * 8;
        let chroma_x = region % 2 * 4;
        let chroma_y = region / 2 * 4;
        for row in 0..8 {
            let target = (luma_y + row) * 16 + luma_x;
            block.luma[target..target + 8].copy_from_slice(&part.luma[row * 8..row * 8 + 8]);
        }
        for row in 0..4 {
            let target = (chroma_y + row) * 8 + chroma_x;
            block.cb[target..target + 4].copy_from_slice(&part.cb[row * 4..row * 4 + 4]);
            block.cr[target..target + 4].copy_from_slice(&part.cr[row * 4..row * 4 + 4]);
        }
    }
    block
}

#[cfg(test)]
std::thread_local! {
    static B_BLEND_CACHE: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

#[cfg(test)]
#[allow(dead_code)] // Used by the standalone H.264 performance harness.
pub(super) fn with_b_blend_cache<T>(enabled: bool, decode: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            B_BLEND_CACHE.set(self.0);
        }
    }
    let _restore = Restore(B_BLEND_CACHE.replace(enabled));
    decode()
}

fn predict_b_subpartition_block(
    cells: &[MotionCell; 16],
    references: &[Yuv420Picture],
    list0: &[usize],
    list1: &[usize],
    x: usize,
    y: usize,
    poc: i32,
    weighted_bipred_idc: u32,
) -> Result<InterMacroblock, AvcError> {
    #[cfg(test)]
    if B_BLEND_CACHE.get() {
        return predict_b_subpartition_block_impl::<true>(
            cells,
            references,
            list0,
            list1,
            x,
            y,
            poc,
            weighted_bipred_idc,
        );
    }
    predict_b_subpartition_block_impl::<false>(
        cells,
        references,
        list0,
        list1,
        x,
        y,
        poc,
        weighted_bipred_idc,
    )
}

fn predict_b_subpartition_block_impl<const CACHE_BLEND: bool>(
    cells: &[MotionCell; 16],
    references: &[Yuv420Picture],
    list0: &[usize],
    list1: &[usize],
    x: usize,
    y: usize,
    poc: i32,
    weighted_bipred_idc: u32,
) -> Result<InterMacroblock, AvcError> {
    let mut block = InterMacroblock {
        luma: [0; 256],
        cb: [0; 64],
        cr: [0; 64],
    };
    let mut cached: [Option<(u8, [i32; 2], InterRegion)>; 8] = [None; 8];
    // A shared 8x8 motion region needs only one blend, not one per 4x4 cell.
    let mut blended: [Option<(MotionCell, InterRegion)>; 4] = [None; 4];
    for cy in 0..4 {
        for cx in 0..4 {
            let region = cy / 2 * 2 + cx / 2;
            let motion = cells[cy * 4 + cx];
            let pixels = if let Some((_, pixels)) =
                blended[region].filter(|(key, _)| CACHE_BLEND && *key == motion)
            {
                pixels
            } else {
                let mut predicted = [None; 2];
                for list in 0..2 {
                    let Some((ref_index, vector)) = (if list == 0 { motion.l0 } else { motion.l1 })
                    else {
                        continue;
                    };
                    let slot = &mut cached[region * 2 + list];
                    if !slot.is_some_and(|(cached_ref, cached_vector, _)| {
                        cached_ref == ref_index && cached_vector == vector
                    }) {
                        let source = (if list == 0 { list0 } else { list1 })
                            .get(ref_index as usize)
                            .ok_or(AvcError::Unsupported("B reference index"))?;
                        *slot = Some((
                            ref_index,
                            vector,
                            predict_inter_8x8_region(&references[*source], x, y, vector, region)?,
                        ));
                    }
                    predicted[list] = slot.map(|(_, _, pixels)| pixels);
                }
                let pixels = match (predicted[0], predicted[1]) {
                    (Some(first), Some(second)) => {
                        let weights = if weighted_bipred_idc == 2 {
                            let source0 = list0[motion.l0.unwrap().0 as usize];
                            let source1 = list1[motion.l1.unwrap().0 as usize];
                            Some(implicit_b_weights(
                                poc,
                                references[source0].pic_order_cnt,
                                references[source1].pic_order_cnt,
                            ))
                        } else {
                            None
                        };
                        blend_b_compact(&first, &second, weights)
                    }
                    (Some(pixels), None) | (None, Some(pixels)) => pixels,
                    (None, None) => {
                        return Err(AvcError::InvalidData("B block without prediction"));
                    }
                };
                if CACHE_BLEND {
                    blended[region] = Some((motion, pixels));
                }
                pixels
            };
            let local_x = cx % 2 * 4;
            let local_y = cy % 2 * 4;
            for row in 0..4 {
                let source = (local_y + row) * 8 + local_x;
                let target = (cy * 4 + row) * 16 + cx * 4;
                block.luma[target..target + 4].copy_from_slice(&pixels.luma[source..source + 4]);
            }
            for row in 0..2 {
                let source = (cy % 2 * 2 + row) * 4 + cx % 2 * 2;
                let target = (cy * 2 + row) * 8 + cx * 2;
                block.cb[target..target + 2].copy_from_slice(&pixels.cb[source..source + 2]);
                block.cr[target..target + 2].copy_from_slice(&pixels.cr[source..source + 2]);
            }
        }
    }
    Ok(block)
}

/// H.264 8.4.2.3.2, implicit weighting for two short-term frame references.
pub(super) fn implicit_b_weights(current_poc: i32, poc0: i32, poc1: i32) -> (i32, i32) {
    let td = (poc1 - poc0).clamp(-128, 127);
    if td == 0 {
        return (32, 32);
    }
    let tb = (current_poc - poc0).clamp(-128, 127);
    let tx = (16384 + (td / 2).abs()) / td;
    let scale = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
    let weight1 = scale >> 2;
    if !(-64..=128).contains(&weight1) {
        (32, 32)
    } else {
        (64 - weight1, weight1)
    }
}

pub(super) fn blend_b_macroblocks(
    list0: &InterMacroblock,
    list1: &InterMacroblock,
    weights: Option<(i32, i32)>,
) -> InterMacroblock {
    #[cfg(not(target_arch = "aarch64"))]
    let blend = |a: u8, b: u8| -> u8 {
        match weights {
            Some((w0, w1)) => {
                ((i32::from(a) * w0 + i32::from(b) * w1 + 32) >> 6).clamp(0, 255) as u8
            }
            None => (u16::from(a) + u16::from(b) + 1 >> 1) as u8,
        }
    };
    let mut block = InterMacroblock {
        luma: [0; 256],
        cb: [0; 64],
        cr: [0; 64],
    };
    for (output, first, second) in [
        (&mut block.luma[..], &list0.luma[..], &list1.luma[..]),
        (&mut block.cb[..], &list0.cb[..], &list1.cb[..]),
        (&mut block.cr[..], &list0.cr[..], &list1.cr[..]),
    ] {
        #[cfg(target_arch = "aarch64")]
        for start in (0..output.len()).step_by(8) {
            unsafe {
                blend_b_luma_row_neon(
                    &first[start..start + 8],
                    &second[start..start + 8],
                    &mut output[start..start + 8],
                    weights,
                );
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        for (target, (&a, &b)) in output.iter_mut().zip(first.iter().zip(second)) {
            *target = blend(a, b);
        }
    }
    block
}

#[cfg(test)]
fn blend_b_region(
    list0: &InterMacroblock,
    list1: &InterMacroblock,
    region: usize,
    weights: Option<(i32, i32)>,
) -> InterMacroblock {
    let blend = |a: u8, b: u8| -> u8 {
        match weights {
            Some((w0, w1)) => {
                ((i32::from(a) * w0 + i32::from(b) * w1 + 32) >> 6).clamp(0, 255) as u8
            }
            None => ((u16::from(a) + u16::from(b) + 1) >> 1) as u8,
        }
    };
    let mut block = InterMacroblock {
        luma: [0; 256],
        cb: [0; 64],
        cr: [0; 64],
    };
    let x = region % 2;
    let y = region / 2;
    for row in 0..8 {
        let start = (y * 8 + row) * 16 + x * 8;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            blend_b_luma_row_neon(
                &list0.luma[start..start + 8],
                &list1.luma[start..start + 8],
                &mut block.luma[start..start + 8],
                weights,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        for offset in start..start + 8 {
            block.luma[offset] = blend(list0.luma[offset], list1.luma[offset]);
        }
    }
    for row in 0..4 {
        let start = (y * 4 + row) * 8 + x * 4;
        for offset in start..start + 4 {
            block.cb[offset] = blend(list0.cb[offset], list1.cb[offset]);
            block.cr[offset] = blend(list0.cr[offset], list1.cr[offset]);
        }
    }
    block
}

fn blend_b_compact(
    list0: &InterRegion,
    list1: &InterRegion,
    weights: Option<(i32, i32)>,
) -> InterRegion {
    let blend = |a: u8, b: u8| -> u8 {
        match weights {
            Some((w0, w1)) => {
                ((i32::from(a) * w0 + i32::from(b) * w1 + 32) >> 6).clamp(0, 255) as u8
            }
            None => ((u16::from(a) + u16::from(b) + 1) >> 1) as u8,
        }
    };
    let mut block = InterRegion {
        luma: [0; 64],
        cb: [0; 16],
        cr: [0; 16],
    };
    for row in 0..8 {
        let start = row * 8;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            blend_b_luma_row_neon(
                &list0.luma[start..start + 8],
                &list1.luma[start..start + 8],
                &mut block.luma[start..start + 8],
                weights,
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        for index in start..start + 8 {
            block.luma[index] = blend(list0.luma[index], list1.luma[index]);
        }
    }
    for index in 0..16 {
        block.cb[index] = blend(list0.cb[index], list1.cb[index]);
        block.cr[index] = blend(list0.cr[index], list1.cr[index]);
    }
    block
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn blend_b_luma_row_neon(
    a: &[u8],
    b: &[u8],
    output: &mut [u8],
    weights: Option<(i32, i32)>,
) {
    use std::arch::aarch64::*;
    unsafe {
        let a = vld1_u8(a.as_ptr());
        let b = vld1_u8(b.as_ptr());
        let value = if let Some((w0, w1)) = weights {
            let a = vreinterpretq_s16_u16(vmovl_u8(a));
            let b = vreinterpretq_s16_u16(vmovl_u8(b));
            let low = vmlal_n_s16(
                vmull_n_s16(vget_low_s16(a), w0 as i16),
                vget_low_s16(b),
                w1 as i16,
            );
            let high = vmlal_n_s16(
                vmull_n_s16(vget_high_s16(a), w0 as i16),
                vget_high_s16(b),
                w1 as i16,
            );
            let rounded = |sum| vshrq_n_s32::<6>(vaddq_s32(sum, vdupq_n_s32(32)));
            vqmovun_s16(vcombine_s16(
                vqmovn_s32(rounded(low)),
                vqmovn_s32(rounded(high)),
            ))
        } else {
            vrhadd_u8(a, b)
        };
        vst1_u8(output.as_mut_ptr(), value);
    }
}

fn sample(plane: &[u8], width: usize, height: usize, x: i32, y: i32) -> i32 {
    let x = x.clamp(0, width as i32 - 1) as usize;
    let y = y.clamp(0, height as i32 - 1) as usize;
    i32::from(plane[y * width + x])
}

fn six_tap(values: [i32; 6]) -> i32 {
    values[0] - 5 * values[1] + 20 * values[2] + 20 * values[3] - 5 * values[4] + values[5]
}

fn horizontal(plane: &[u8], width: usize, height: usize, x: i32, y: i32) -> i32 {
    six_tap(std::array::from_fn(|i| {
        sample(plane, width, height, x + i as i32 - 2, y)
    }))
}

fn vertical(plane: &[u8], width: usize, height: usize, x: i32, y: i32) -> i32 {
    six_tap(std::array::from_fn(|i| {
        sample(plane, width, height, x, y + i as i32 - 2)
    }))
}

fn clip(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

fn average(a: i32, b: i32) -> u8 {
    ((a + b + 1) >> 1) as u8
}

pub(super) struct HalfPelPlanes {
    horizontal: Vec<u8>,
    vertical: Vec<u8>,
    diagonal: Vec<u8>,
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn six_tap_neon_u8(
    taps: [std::arch::aarch64::uint8x8_t; 6],
) -> std::arch::aarch64::int16x8_t {
    use std::arch::aarch64::*;
    let taps = taps.map(|tap| vreinterpretq_s16_u16(vmovl_u8(tap)));
    vaddq_s16(
        vsubq_s16(
            vaddq_s16(taps[0], taps[5]),
            vmulq_n_s16(vaddq_s16(taps[1], taps[4]), 5),
        ),
        vmulq_n_s16(vaddq_s16(taps[2], taps[3]), 20),
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn halfpel_rows_neon(
    plane: &[u8],
    width: usize,
    index: usize,
    raw: &mut [i16],
    horizontal_half: &mut [u8],
    vertical_half: &mut [u8],
) {
    use std::arch::aarch64::*;
    unsafe {
        let source = plane.as_ptr().add(index);
        let horizontal = six_tap_neon_u8([
            vld1_u8(source.sub(2)),
            vld1_u8(source.sub(1)),
            vld1_u8(source),
            vld1_u8(source.add(1)),
            vld1_u8(source.add(2)),
            vld1_u8(source.add(3)),
        ]);
        let vertical = six_tap_neon_u8([
            vld1_u8(source.sub(2 * width)),
            vld1_u8(source.sub(width)),
            vld1_u8(source),
            vld1_u8(source.add(width)),
            vld1_u8(source.add(2 * width)),
            vld1_u8(source.add(3 * width)),
        ]);
        vst1q_s16(raw.as_mut_ptr().add(index), horizontal);
        let rounded_horizontal = vshrq_n_s16::<5>(vaddq_s16(horizontal, vdupq_n_s16(16)));
        let rounded_vertical = vshrq_n_s16::<5>(vaddq_s16(vertical, vdupq_n_s16(16)));
        vst1_u8(
            horizontal_half.as_mut_ptr().add(index),
            vqmovun_s16(rounded_horizontal),
        );
        vst1_u8(
            vertical_half.as_mut_ptr().add(index),
            vqmovun_s16(rounded_vertical),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn diagonal_row_neon(raw: &[i16], width: usize, index: usize, output: &mut [u8]) {
    use std::arch::aarch64::*;
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn filter(taps: [int16x4_t; 6]) -> uint16x4_t {
        let taps = taps.map(|value| vmovl_s16(value));
        let value = vaddq_s32(
            vsubq_s32(
                vaddq_s32(taps[0], taps[5]),
                vmulq_n_s32(vaddq_s32(taps[1], taps[4]), 5),
            ),
            vmulq_n_s32(vaddq_s32(taps[2], taps[3]), 20),
        );
        vqmovun_s32(vshrq_n_s32::<10>(vaddq_s32(value, vdupq_n_s32(512))))
    }
    unsafe {
        let source = raw.as_ptr().add(index);
        let taps = [
            vld1q_s16(source.sub(2 * width)),
            vld1q_s16(source.sub(width)),
            vld1q_s16(source),
            vld1q_s16(source.add(width)),
            vld1q_s16(source.add(2 * width)),
            vld1q_s16(source.add(3 * width)),
        ];
        let low = filter(taps.map(|value| vget_low_s16(value)));
        let high = filter(taps.map(|value| vget_high_s16(value)));
        vst1_u8(
            output.as_mut_ptr().add(index),
            vqmovn_u16(vcombine_u16(low, high)),
        );
    }
}

impl HalfPelPlanes {
    fn new(reference: &Yuv420Picture) -> Self {
        let width = reference.width;
        let height = reference.height;
        let plane = &reference.luma;
        let mut horizontal_raw = vec![0i16; width * height];
        let mut horizontal_half = vec![0u8; width * height];
        let mut vertical_half = vec![0u8; width * height];
        let mut diagonal_half = vec![0u8; width * height];
        for y in 0..height {
            let mut x = 0;
            while x < width {
                let index = y * width + x;
                #[cfg(target_arch = "aarch64")]
                if y >= 2 && y + 3 < height && x >= 2 && x + 10 < width {
                    // SAFETY: the interior bounds cover every six-tap load and eight output lanes.
                    unsafe {
                        halfpel_rows_neon(
                            plane,
                            width,
                            index,
                            &mut horizontal_raw,
                            &mut horizontal_half,
                            &mut vertical_half,
                        );
                    }
                    x += 8;
                    continue;
                }
                let interior = x >= 2 && x + 3 < width && y >= 2 && y + 3 < height;
                let value = if interior {
                    six_tap([
                        i32::from(plane[index - 2]),
                        i32::from(plane[index - 1]),
                        i32::from(plane[index]),
                        i32::from(plane[index + 1]),
                        i32::from(plane[index + 2]),
                        i32::from(plane[index + 3]),
                    ])
                } else {
                    horizontal(plane, width, height, x as i32, y as i32)
                };
                // Six-tap output from u8 samples lies in [-2550, 10200].
                horizontal_raw[index] = value as i16;
                horizontal_half[index] = clip((value + 16) >> 5);
                let vertical_value = if interior {
                    six_tap([
                        i32::from(plane[index - 2 * width]),
                        i32::from(plane[index - width]),
                        i32::from(plane[index]),
                        i32::from(plane[index + width]),
                        i32::from(plane[index + 2 * width]),
                        i32::from(plane[index + 3 * width]),
                    ])
                } else {
                    vertical(plane, width, height, x as i32, y as i32)
                };
                vertical_half[index] = clip((vertical_value + 16) >> 5);
                x += 1;
            }
        }
        for y in 0..height {
            let mut x = 0;
            while x < width {
                let index = y * width + x;
                #[cfg(target_arch = "aarch64")]
                if y >= 2 && y + 3 < height && x + 8 <= width {
                    // SAFETY: all six source rows and eight output lanes are in range.
                    unsafe { diagonal_row_neon(&horizontal_raw, width, index, &mut diagonal_half) };
                    x += 8;
                    continue;
                }
                let taps = if y >= 2 && y + 3 < height {
                    six_tap([
                        i32::from(horizontal_raw[index - 2 * width]),
                        i32::from(horizontal_raw[index - width]),
                        i32::from(horizontal_raw[index]),
                        i32::from(horizontal_raw[index + width]),
                        i32::from(horizontal_raw[index + 2 * width]),
                        i32::from(horizontal_raw[index + 3 * width]),
                    ])
                } else {
                    six_tap(std::array::from_fn(|i| {
                        i32::from(
                            horizontal_raw[(y as i32 + i as i32 - 2).clamp(0, height as i32 - 1)
                                as usize
                                * width
                                + x],
                        )
                    }))
                };
                diagonal_half[y * width + x] = clip((taps + 512) >> 10);
                x += 1;
            }
        }
        Self {
            horizontal: horizontal_half,
            vertical: vertical_half,
            diagonal: diagonal_half,
        }
    }
}

fn luma_quarter_cached(reference: &Yuv420Picture, half: &HalfPelPlanes, x4: i32, y4: i32) -> u8 {
    let x = x4.div_euclid(4);
    let y = y4.div_euclid(4);
    let fx = x4.rem_euclid(4);
    let fy = y4.rem_euclid(4);
    let width = reference.width;
    let height = reference.height;
    if x < 0 || y < 0 || x >= width as i32 - 1 || y >= height as i32 - 1 {
        return luma_quarter(&reference.luma, width, height, x4, y4);
    }
    let at =
        |plane: &[u8], x: i32, y: i32| -> i32 { i32::from(plane[y as usize * width + x as usize]) };
    let full = at(&reference.luma, x, y);
    let b = || at(&half.horizontal, x, y);
    let h = || at(&half.vertical, x, y);
    let j = || at(&half.diagonal, x, y);
    let m = || at(&half.vertical, x + 1, y);
    let s = || at(&half.horizontal, x, y + 1);
    match (fx, fy) {
        (0, 0) => full as u8,
        (0, 1) => average(full, h()),
        (0, 2) => h() as u8,
        (0, 3) => average(at(&reference.luma, x, y + 1), h()),
        (1, 0) => average(full, b()),
        (1, 1) => average(b(), h()),
        (1, 2) => average(h(), j()),
        (1, 3) => average(h(), s()),
        (2, 0) => b() as u8,
        (2, 1) => average(b(), j()),
        (2, 2) => j() as u8,
        (2, 3) => average(j(), s()),
        (3, 0) => average(at(&reference.luma, x + 1, y), b()),
        (3, 1) => average(b(), m()),
        (3, 2) => average(j(), m()),
        (3, 3) => average(m(), s()),
        _ => unreachable!(),
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn average_luma_row_neon(a: &[u8], b: &[u8], output: &mut [u8]) {
    use std::arch::aarch64::*;
    unsafe {
        if output.len() == 16 {
            vst1q_u8(
                output.as_mut_ptr(),
                vrhaddq_u8(vld1q_u8(a.as_ptr()), vld1q_u8(b.as_ptr())),
            );
        } else {
            vst1_u8(
                output.as_mut_ptr(),
                vrhadd_u8(vld1_u8(a.as_ptr()), vld1_u8(b.as_ptr())),
            );
        }
    }
}

fn predict_cached_luma_region(
    reference: &Yuv420Picture,
    half: &HalfPelPlanes,
    mb_x: usize,
    mb_y: usize,
    region_x: usize,
    region_y: usize,
    span: usize,
    motion: [i32; 2],
    output: &mut [u8],
    output_stride: usize,
    output_x: usize,
    output_y: usize,
) -> bool {
    let width = reference.width;
    let height = reference.height;
    let source_x = (mb_x * 16 + region_x) as i64 + i64::from(motion[0] >> 2);
    let source_y = (mb_y * 16 + region_y) as i64 + i64::from(motion[1] >> 2);
    if source_x < 0
        || source_y < 0
        || source_x + span as i64 >= width as i64
        || source_y + span as i64 >= height as i64
    {
        return false;
    }
    let full = reference.luma.as_slice();
    let horizontal = half.horizontal.as_slice();
    let vertical = half.vertical.as_slice();
    let diagonal = half.diagonal.as_slice();
    let (first, first_offset, second): (&[u8], usize, Option<(&[u8], usize)>) =
        match (motion[0].rem_euclid(4), motion[1].rem_euclid(4)) {
            (0, 0) => (full, 0, None),
            (0, 1) => (full, 0, Some((vertical, 0))),
            (0, 2) => (vertical, 0, None),
            (0, 3) => (full, width, Some((vertical, 0))),
            (1, 0) => (full, 0, Some((horizontal, 0))),
            (1, 1) => (horizontal, 0, Some((vertical, 0))),
            (1, 2) => (vertical, 0, Some((diagonal, 0))),
            (1, 3) => (vertical, 0, Some((horizontal, width))),
            (2, 0) => (horizontal, 0, None),
            (2, 1) => (horizontal, 0, Some((diagonal, 0))),
            (2, 2) => (diagonal, 0, None),
            (2, 3) => (diagonal, 0, Some((horizontal, width))),
            (3, 0) => (full, 1, Some((horizontal, 0))),
            (3, 1) => (horizontal, 0, Some((vertical, 1))),
            (3, 2) => (diagonal, 0, Some((vertical, 1))),
            (3, 3) => (vertical, 1, Some((horizontal, width))),
            _ => unreachable!(),
        };
    for row in 0..span {
        let source = (source_y as usize + row) * width + source_x as usize;
        let target = (output_y + row) * output_stride + output_x;
        let first = &first[source + first_offset..source + first_offset + span];
        let destination = &mut output[target..target + span];
        if let Some((second, offset)) = second {
            let second = &second[source + offset..source + offset + span];
            #[cfg(target_arch = "aarch64")]
            unsafe {
                if span >= 8 {
                    average_luma_row_neon(first, second, destination);
                } else {
                    for (out, (&a, &b)) in destination.iter_mut().zip(first.iter().zip(second)) {
                        *out = (u16::from(a) + u16::from(b) + 1 >> 1) as u8;
                    }
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            for (out, (&a, &b)) in destination.iter_mut().zip(first.iter().zip(second)) {
                *out = (u16::from(a) + u16::from(b) + 1 >> 1) as u8;
            }
        } else {
            destination.copy_from_slice(first);
        }
    }
    true
}

/// x4/y4 are absolute quarter-luma-sample coordinates, including the motion vector.
pub fn luma_quarter(plane: &[u8], width: usize, height: usize, x4: i32, y4: i32) -> u8 {
    let x = x4.div_euclid(4);
    let y = y4.div_euclid(4);
    let fx = x4.rem_euclid(4);
    let fy = y4.rem_euclid(4);
    let full = sample(plane, width, height, x, y);
    if fx == 0 && fy == 0 {
        return full as u8;
    }
    let b = || i32::from(clip((horizontal(plane, width, height, x, y) + 16) >> 5));
    let h = || i32::from(clip((vertical(plane, width, height, x, y) + 16) >> 5));
    let j = || {
        let taps = six_tap(std::array::from_fn(|i| {
            horizontal(plane, width, height, x, y + i as i32 - 2)
        }));
        i32::from(clip((taps + 512) >> 10))
    };
    let m = || i32::from(clip((vertical(plane, width, height, x + 1, y) + 16) >> 5));
    let s = || i32::from(clip((horizontal(plane, width, height, x, y + 1) + 16) >> 5));
    match (fx, fy) {
        (0, 1) => average(full, h()),
        (0, 2) => h() as u8,
        (0, 3) => average(sample(plane, width, height, x, y + 1), h()),
        (1, 0) => average(full, b()),
        (1, 1) => average(b(), h()),
        (1, 2) => average(h(), j()),
        (1, 3) => average(h(), s()),
        (2, 0) => b() as u8,
        (2, 1) => average(b(), j()),
        (2, 2) => j() as u8,
        (2, 3) => average(j(), s()),
        (3, 0) => average(sample(plane, width, height, x + 1, y), b()),
        (3, 1) => average(b(), m()),
        (3, 2) => average(j(), m()),
        (3, 3) => average(m(), s()),
        _ => unreachable!(),
    }
}

/// x8/y8 are absolute eighth-chroma-sample coordinates for 4:2:0.
pub fn chroma_eighth(plane: &[u8], width: usize, height: usize, x8: i32, y8: i32) -> u8 {
    let x = x8.div_euclid(8);
    let y = y8.div_euclid(8);
    let fx = x8.rem_euclid(8);
    let fy = y8.rem_euclid(8);
    let (a, b, c, d) = if x >= 0 && y >= 0 && x + 1 < width as i32 && y + 1 < height as i32 {
        let index = y as usize * width + x as usize;
        (
            i32::from(plane[index]),
            i32::from(plane[index + 1]),
            i32::from(plane[index + width]),
            i32::from(plane[index + width + 1]),
        )
    } else {
        (
            sample(plane, width, height, x, y),
            sample(plane, width, height, x + 1, y),
            sample(plane, width, height, x, y + 1),
            sample(plane, width, height, x + 1, y + 1),
        )
    };
    (((8 - fx) * (8 - fy) * a + fx * (8 - fy) * b + (8 - fx) * fy * c + fx * fy * d + 32) >> 6)
        as u8
}

fn predict_chroma_region(
    reference: &Yuv420Picture,
    x: usize,
    y: usize,
    motion: [i32; 2],
    size: usize,
    cb: &mut [u8],
    cr: &mut [u8],
) -> bool {
    let width = reference.width / 2;
    let height = reference.height / 2;
    let base_x = x as i64 + i64::from(motion[0].div_euclid(8));
    let base_y = y as i64 + i64::from(motion[1].div_euclid(8));
    if base_x < 0
        || base_y < 0
        || base_x + size as i64 >= width as i64
        || base_y + size as i64 >= height as i64
    {
        return false;
    }
    let fx = motion[0].rem_euclid(8);
    let fy = motion[1].rem_euclid(8);
    let coefficients = [(8 - fx) * (8 - fy), fx * (8 - fy), (8 - fx) * fy, fx * fy];
    let base = base_y as usize * width + base_x as usize;
    #[cfg(target_arch = "aarch64")]
    if size == 8 {
        let weights = coefficients.map(|value| value as u16);
        unsafe {
            predict_chroma_rows_neon(&reference.cb, base, width, weights, cb);
            predict_chroma_rows_neon(&reference.cr, base, width, weights, cr);
        }
        return true;
    }
    for (plane, output) in [(&reference.cb[..], cb), (&reference.cr[..], cr)] {
        for row in 0..size {
            let source = base + row * width;
            let target = row * size;
            for col in 0..size {
                let offset = source + col;
                output[target + col] = ((coefficients[0] * i32::from(plane[offset])
                    + coefficients[1] * i32::from(plane[offset + 1])
                    + coefficients[2] * i32::from(plane[offset + width])
                    + coefficients[3] * i32::from(plane[offset + width + 1])
                    + 32)
                    >> 6) as u8;
            }
        }
    }
    true
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn predict_chroma_rows_neon(
    plane: &[u8],
    base: usize,
    width: usize,
    weights: [u16; 4],
    output: &mut [u8],
) {
    use std::arch::aarch64::*;
    unsafe {
        for row in 0..8 {
            let source = plane.as_ptr().add(base + row * width);
            let a = vmovl_u8(vld1_u8(source));
            let b = vmovl_u8(vld1_u8(source.add(1)));
            let c = vmovl_u8(vld1_u8(source.add(width)));
            let d = vmovl_u8(vld1_u8(source.add(width + 1)));
            let sum = vmlaq_n_u16(
                vmlaq_n_u16(
                    vmlaq_n_u16(vmulq_n_u16(a, weights[0]), b, weights[1]),
                    c,
                    weights[2],
                ),
                d,
                weights[3],
            );
            let rounded = vshrn_n_u16::<6>(vaddq_u16(sum, vdupq_n_u16(32)));
            vst1_u8(output.as_mut_ptr().add(row * 8), rounded);
        }
    }
}

/// Equation 8-270 for an 8-bit single-list weighted prediction sample.
pub fn weighted_single(value: u8, weight: i32, offset: i32, denom: u32) -> u8 {
    let round = if denom == 0 { 0 } else { 1 << (denom - 1) };
    clip(((i32::from(value) * weight + round) >> denom) + offset)
}

/// Predict a P_L0_16x16 macroblock with one 4:2:0 reference picture.
pub fn predict_l0_16x16(
    reference: &Yuv420Picture,
    mb_x: usize,
    mb_y: usize,
    motion: [i32; 2],
    weights: Option<&PredictionWeightTable>,
    ref_index: usize,
) -> Result<InterMacroblock, AvcError> {
    predict_inter_16x16(reference, mb_x, mb_y, motion, weights, 0, ref_index)
}

#[cfg(test)]
pub fn predict_l1_16x16(
    reference: &Yuv420Picture,
    mb_x: usize,
    mb_y: usize,
    motion: [i32; 2],
    weights: Option<&PredictionWeightTable>,
    ref_index: usize,
) -> Result<InterMacroblock, AvcError> {
    predict_inter_16x16(reference, mb_x, mb_y, motion, weights, 1, ref_index)
}

fn predict_inter_16x16(
    reference: &Yuv420Picture,
    mb_x: usize,
    mb_y: usize,
    motion: [i32; 2],
    weights: Option<&PredictionWeightTable>,
    list: usize,
    ref_index: usize,
) -> Result<InterMacroblock, AvcError> {
    if mb_x * 16 >= reference.width || mb_y * 16 >= reference.height {
        return Err(AvcError::InvalidData("inter macroblock out of bounds"));
    }
    let weight = weights
        .map(|table| {
            (if list == 0 {
                &table.list0
            } else {
                &table.list1
            })
            .get(ref_index)
            .map(|weight| (table, weight))
            .ok_or(AvcError::InvalidData("missing prediction weight"))
        })
        .transpose()?;
    let mut block = InterMacroblock {
        luma: [0; 256],
        cb: [0; 64],
        cr: [0; 64],
    };
    if motion == [0, 0] && weight.is_none() {
        for row in 0..16 {
            let source = (mb_y * 16 + row) * reference.width + mb_x * 16;
            block.luma[row * 16..row * 16 + 16]
                .copy_from_slice(&reference.luma[source..source + 16]);
        }
        for row in 0..8 {
            let source = (mb_y * 8 + row) * (reference.width / 2) + mb_x * 8;
            let target = row * 8..row * 8 + 8;
            block.cb[target.clone()].copy_from_slice(&reference.cb[source..source + 8]);
            block.cr[target].copy_from_slice(&reference.cr[source..source + 8]);
        }
        return Ok(block);
    }
    let half = (motion[0] & 3 != 0 || motion[1] & 3 != 0).then(|| {
        reference
            .luma_half
            .get_or_init(|| HalfPelPlanes::new(reference))
    });
    let integer_x = (mb_x * 16) as i64 + i64::from(motion[0] >> 2);
    let integer_y = (mb_y * 16) as i64 + i64::from(motion[1] >> 2);
    let luma_bulk = if motion[0] & 3 == 0
        && motion[1] & 3 == 0
        && integer_x >= 0
        && integer_y >= 0
        && integer_x + 16 <= reference.width as i64
        && integer_y + 16 <= reference.height as i64
    {
        let source_x = integer_x as usize;
        let source_y = integer_y as usize;
        for row in 0..16 {
            let source = (source_y + row) * reference.width + source_x;
            block.luma[row * 16..row * 16 + 16]
                .copy_from_slice(&reference.luma[source..source + 16]);
        }
        true
    } else if let Some(half) = half {
        predict_cached_luma_region(
            reference,
            half,
            mb_x,
            mb_y,
            0,
            0,
            16,
            motion,
            &mut block.luma,
            16,
            0,
            0,
        )
    } else {
        false
    };
    if luma_bulk {
        if let Some((table, weight)) = weight {
            for value in &mut block.luma {
                *value = weighted_single(
                    *value,
                    weight.luma_weight,
                    weight.luma_offset,
                    table.luma_denom,
                );
            }
        }
    } else {
        for y in 0..16 {
            for x in 0..16 {
                let x4 = ((mb_x * 16 + x) as i32) * 4 + motion[0];
                let y4 = ((mb_y * 16 + y) as i32) * 4 + motion[1];
                let value = match half {
                    Some(half) => luma_quarter_cached(reference, half, x4, y4),
                    None => {
                        luma_quarter(&reference.luma, reference.width, reference.height, x4, y4)
                    }
                };
                block.luma[y * 16 + x] = if let Some((table, weight)) = weight {
                    weighted_single(
                        value,
                        weight.luma_weight,
                        weight.luma_offset,
                        table.luma_denom,
                    )
                } else {
                    value
                };
            }
        }
    }
    let chroma_width = reference.width / 2;
    let chroma_height = reference.height / 2;
    let chroma_source_x = mb_x as i64 * 8 + i64::from(motion[0] >> 3);
    let chroma_source_y = mb_y as i64 * 8 + i64::from(motion[1] >> 3);
    let chroma_bulk = if motion[0] & 7 == 0
        && motion[1] & 7 == 0
        && chroma_source_x >= 0
        && chroma_source_y >= 0
        && chroma_source_x + 8 <= chroma_width as i64
        && chroma_source_y + 8 <= chroma_height as i64
    {
        let source_x = chroma_source_x as usize;
        let source_y = chroma_source_y as usize;
        for row in 0..8 {
            let source = (source_y + row) * chroma_width + source_x;
            let target = row * 8..row * 8 + 8;
            block.cb[target.clone()].copy_from_slice(&reference.cb[source..source + 8]);
            block.cr[target].copy_from_slice(&reference.cr[source..source + 8]);
        }
        true
    } else {
        predict_chroma_region(
            reference,
            mb_x * 8,
            mb_y * 8,
            motion,
            8,
            &mut block.cb,
            &mut block.cr,
        )
    };
    if chroma_bulk {
        if let Some((table, weight)) = weight {
            for (channel, target) in [&mut block.cb, &mut block.cr].into_iter().enumerate() {
                for value in target {
                    *value = weighted_single(
                        *value,
                        weight.chroma_weight[channel],
                        weight.chroma_offset[channel],
                        table.chroma_denom,
                    );
                }
            }
        }
        return Ok(block);
    }
    for y in 0..8 {
        for x in 0..8 {
            let x8 = ((mb_x * 8 + x) as i32) * 8 + motion[0];
            let y8 = ((mb_y * 8 + y) as i32) * 8 + motion[1];
            for (channel, (source, target)) in [
                (&reference.cb, &mut block.cb),
                (&reference.cr, &mut block.cr),
            ]
            .into_iter()
            .enumerate()
            {
                let value = chroma_eighth(source, chroma_width, chroma_height, x8, y8);
                target[y * 8 + x] = if let Some((table, weight)) = weight {
                    weighted_single(
                        value,
                        weight.chroma_weight[channel],
                        weight.chroma_offset[channel],
                        table.chroma_denom,
                    )
                } else {
                    value
                };
            }
        }
    }
    Ok(block)
}

fn predict_p_subpartitions(
    references: &[Yuv420Picture],
    list0: &[usize],
    mb_x: usize,
    mb_y: usize,
    parts: &PSubPartitions,
    weights: Option<&PredictionWeightTable>,
) -> Result<InterMacroblock, AvcError> {
    let mut block = InterMacroblock {
        luma: [0; 256],
        cb: [0; 64],
        cr: [0; 64],
    };
    for cy in 0..4 {
        for cx in 0..4 {
            let cell = cy * 4 + cx;
            let ref_index = parts.references[cell] as usize;
            let source = list0
                .get(ref_index)
                .and_then(|&index| references.get(index))
                .ok_or(AvcError::Unsupported("P list0 index"))?;
            let motion = parts.vectors[cell];
            let half = (motion[0] & 3 != 0 || motion[1] & 3 != 0)
                .then(|| source.luma_half.get_or_init(|| HalfPelPlanes::new(source)));
            let weight = weights
                .map(|table| {
                    table
                        .list0
                        .get(ref_index)
                        .map(|weight| (table, weight))
                        .ok_or(AvcError::InvalidData("missing prediction weight"))
                })
                .transpose()?;
            let region_x = cx * 4;
            let region_y = cy * 4;
            let integer_x = (mb_x * 16 + region_x) as i64 + i64::from(motion[0] >> 2);
            let integer_y = (mb_y * 16 + region_y) as i64 + i64::from(motion[1] >> 2);
            let luma_bulk = if motion[0] & 3 == 0
                && motion[1] & 3 == 0
                && integer_x >= 0
                && integer_y >= 0
                && integer_x + 4 <= source.width as i64
                && integer_y + 4 <= source.height as i64
            {
                for row in 0..4 {
                    let from = (integer_y as usize + row) * source.width + integer_x as usize;
                    let to = (region_y + row) * 16 + region_x;
                    block.luma[to..to + 4].copy_from_slice(&source.luma[from..from + 4]);
                }
                true
            } else if let Some(half) = half {
                predict_cached_luma_region(
                    source,
                    half,
                    mb_x,
                    mb_y,
                    region_x,
                    region_y,
                    4,
                    motion,
                    &mut block.luma,
                    16,
                    region_x,
                    region_y,
                )
            } else {
                false
            };
            for row in region_y..region_y + 4 {
                for col in region_x..region_x + 4 {
                    let value = if luma_bulk {
                        block.luma[row * 16 + col]
                    } else {
                        let x4 = ((mb_x * 16 + col) as i32) * 4 + motion[0];
                        let y4 = ((mb_y * 16 + row) as i32) * 4 + motion[1];
                        match half {
                            Some(half) => luma_quarter_cached(source, half, x4, y4),
                            None => luma_quarter(&source.luma, source.width, source.height, x4, y4),
                        }
                    };
                    block.luma[row * 16 + col] = if let Some((table, weight)) = weight {
                        weighted_single(
                            value,
                            weight.luma_weight,
                            weight.luma_offset,
                            table.luma_denom,
                        )
                    } else {
                        value
                    };
                }
            }
            let chroma_width = source.width / 2;
            let chroma_height = source.height / 2;
            for row in cy * 2..cy * 2 + 2 {
                for col in cx * 2..cx * 2 + 2 {
                    let x8 = ((mb_x * 8 + col) as i32) * 8 + motion[0];
                    let y8 = ((mb_y * 8 + row) as i32) * 8 + motion[1];
                    for (channel, (plane, target)) in
                        [(&source.cb, &mut block.cb), (&source.cr, &mut block.cr)]
                            .into_iter()
                            .enumerate()
                    {
                        let value = chroma_eighth(plane, chroma_width, chroma_height, x8, y8);
                        target[row * 8 + col] = if let Some((table, weight)) = weight {
                            weighted_single(
                                value,
                                weight.chroma_weight[channel],
                                weight.chroma_offset[channel],
                                table.chroma_denom,
                            )
                        } else {
                            value
                        };
                    }
                }
            }
        }
    }
    Ok(block)
}

fn predict_inter_8x8_region(
    reference: &Yuv420Picture,
    mb_x: usize,
    mb_y: usize,
    motion: [i32; 2],
    region: usize,
) -> Result<InterRegion, AvcError> {
    if mb_x * 16 >= reference.width || mb_y * 16 >= reference.height {
        return Err(AvcError::InvalidData("inter macroblock out of bounds"));
    }
    let mut block = InterRegion {
        luma: [0; 64],
        cb: [0; 16],
        cr: [0; 16],
    };
    if motion == [0, 0] {
        let luma_x = (region % 2) * 8;
        let luma_y = (region / 2) * 8;
        for row in 0..8 {
            let source = (mb_y * 16 + luma_y + row) * reference.width + mb_x * 16 + luma_x;
            block.luma[row * 8..row * 8 + 8].copy_from_slice(&reference.luma[source..source + 8]);
        }
        let chroma_x = (region % 2) * 4;
        let chroma_y = (region / 2) * 4;
        for row in 0..4 {
            let source = (mb_y * 8 + chroma_y + row) * (reference.width / 2) + mb_x * 8 + chroma_x;
            let target = row * 4..row * 4 + 4;
            block.cb[target.clone()].copy_from_slice(&reference.cb[source..source + 4]);
            block.cr[target].copy_from_slice(&reference.cr[source..source + 4]);
        }
        return Ok(block);
    }
    let half = (motion[0] & 3 != 0 || motion[1] & 3 != 0).then(|| {
        reference
            .luma_half
            .get_or_init(|| HalfPelPlanes::new(reference))
    });
    let luma_x = (region % 2) * 8;
    let luma_y = (region / 2) * 8;
    let integer_x = (mb_x * 16 + luma_x) as i64 + i64::from(motion[0] >> 2);
    let integer_y = (mb_y * 16 + luma_y) as i64 + i64::from(motion[1] >> 2);
    if motion[0] & 3 == 0
        && motion[1] & 3 == 0
        && integer_x >= 0
        && integer_y >= 0
        && integer_x + 8 <= reference.width as i64
        && integer_y + 8 <= reference.height as i64
    {
        let source_x = integer_x as usize;
        let source_y = integer_y as usize;
        for row in 0..8 {
            let source = (source_y + row) * reference.width + source_x;
            let target = row * 8;
            block.luma[target..target + 8].copy_from_slice(&reference.luma[source..source + 8]);
        }
    } else if !half.is_some_and(|half| {
        predict_cached_luma_region(
            reference,
            half,
            mb_x,
            mb_y,
            luma_x,
            luma_y,
            8,
            motion,
            &mut block.luma,
            8,
            0,
            0,
        )
    }) {
        for y in luma_y..luma_y + 8 {
            for x in luma_x..luma_x + 8 {
                let x4 = ((mb_x * 16 + x) as i32) * 4 + motion[0];
                let y4 = ((mb_y * 16 + y) as i32) * 4 + motion[1];
                block.luma[(y - luma_y) * 8 + x - luma_x] = match half {
                    Some(half) => luma_quarter_cached(reference, half, x4, y4),
                    None => {
                        luma_quarter(&reference.luma, reference.width, reference.height, x4, y4)
                    }
                };
            }
        }
    }
    let chroma_width = reference.width / 2;
    let chroma_height = reference.height / 2;
    let chroma_x = (region % 2) * 4;
    let chroma_y = (region / 2) * 4;
    let chroma_source_x = (mb_x * 8 + chroma_x) as i64 + i64::from(motion[0] >> 3);
    let chroma_source_y = (mb_y * 8 + chroma_y) as i64 + i64::from(motion[1] >> 3);
    if motion[0] & 7 == 0
        && motion[1] & 7 == 0
        && chroma_source_x >= 0
        && chroma_source_y >= 0
        && chroma_source_x + 4 <= chroma_width as i64
        && chroma_source_y + 4 <= chroma_height as i64
    {
        let source_x = chroma_source_x as usize;
        let source_y = chroma_source_y as usize;
        for row in 0..4 {
            let source = (source_y + row) * chroma_width + source_x;
            let target = row * 4;
            block.cb[target..target + 4].copy_from_slice(&reference.cb[source..source + 4]);
            block.cr[target..target + 4].copy_from_slice(&reference.cr[source..source + 4]);
        }
        return Ok(block);
    }
    if predict_chroma_region(
        reference,
        mb_x * 8 + chroma_x,
        mb_y * 8 + chroma_y,
        motion,
        4,
        &mut block.cb,
        &mut block.cr,
    ) {
        return Ok(block);
    }
    for y in chroma_y..chroma_y + 4 {
        for x in chroma_x..chroma_x + 4 {
            let x8 = ((mb_x * 8 + x) as i32) * 8 + motion[0];
            let y8 = ((mb_y * 8 + y) as i32) * 8 + motion[1];
            let target = (y - chroma_y) * 4 + x - chroma_x;
            block.cb[target] = chroma_eighth(&reference.cb, chroma_width, chroma_height, x8, y8);
            block.cr[target] = chroma_eighth(&reference.cr, chroma_width, chroma_height, x8, y8);
        }
    }
    Ok(block)
}

#[cfg(test)]
fn predict_inter_8x8(
    reference: &Yuv420Picture,
    mb_x: usize,
    mb_y: usize,
    motion: [i32; 2],
    region: usize,
) -> Result<InterMacroblock, AvcError> {
    let mut regions = [InterRegion {
        luma: [0; 64],
        cb: [0; 16],
        cr: [0; 16],
    }; 4];
    regions[region] = predict_inter_8x8_region(reference, mb_x, mb_y, motion, region)?;
    Ok(assemble_inter_regions(&regions))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blend_cache_references() -> Vec<Yuv420Picture> {
        (0..2)
            .map(|seed| Yuv420Picture {
                width: 48,
                height: 48,
                frame_num: seed,
                pic_order_cnt_lsb: seed * 8,
                pic_order_cnt_msb: 0,
                pic_order_cnt: seed as i32 * 8,
                luma: (0..48 * 48)
                    .map(|i| (i * 37 + seed as usize * 91) as u8)
                    .collect(),
                cb: (0..24 * 24)
                    .map(|i| (i * 19 + seed as usize * 51) as u8)
                    .collect(),
                cr: (0..24 * 24)
                    .map(|i| (i * 13 + seed as usize * 17) as u8)
                    .collect(),
                motion: Vec::new(),
                reference_pocs: [Vec::new(), Vec::new()],
                luma_half: std::sync::OnceLock::new(),
            })
            .collect()
    }

    #[test]
    fn b_blend_cache_matches_uncached_for_shared_and_mixed_motion() {
        let references = blend_cache_references();
        for pattern in 0..6 {
            let cells = std::array::from_fn(|i| {
                let key = match pattern {
                    0 => 0,
                    1 => i / 8 * 2 + i % 4 / 2,
                    2 => i % 2,
                    3 => i / 4 % 2,
                    _ => i,
                };
                MotionCell {
                    l0: (pattern != 4).then_some(((key % 2) as u8, [key as i32 - 7, 3])),
                    l1: (pattern != 5).then_some((((key + 1) % 2) as u8, [2, key as i32 - 9])),
                }
            });
            for (x, y) in [(0, 0), (1, 1), (2, 2)] {
                for poc in [-40, 0, 2, 4, 6, 48] {
                    for weighting in [0, 2] {
                        let a = predict_b_subpartition_block_impl::<true>(
                            &cells,
                            &references,
                            &[0, 1],
                            &[1, 0],
                            x,
                            y,
                            poc,
                            weighting,
                        )
                        .unwrap();
                        let b = predict_b_subpartition_block_impl::<false>(
                            &cells,
                            &references,
                            &[0, 1],
                            &[1, 0],
                            x,
                            y,
                            poc,
                            weighting,
                        )
                        .unwrap();
                        assert_eq!(a.luma, b.luma);
                        assert_eq!(a.cb, b.cb);
                        assert_eq!(a.cr, b.cr);
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "explicit cached versus repeated B-region blending benchmark"]
    fn benchmark_b_blend_cache() {
        let references = blend_cache_references();
        let cells = [MotionCell {
            l0: Some((0, [3, 7])),
            l1: Some((0, [-1, 5])),
        }; 16];
        for run in 0..6 {
            for cache in [run % 2 == 0, run % 2 != 0] {
                let start = std::time::Instant::now();
                for _ in 0..100_000 {
                    let cells = std::hint::black_box(&cells);
                    let result = if cache {
                        predict_b_subpartition_block_impl::<true>(
                            cells,
                            &references,
                            &[0],
                            &[1],
                            1,
                            1,
                            2,
                            2,
                        )
                    } else {
                        predict_b_subpartition_block_impl::<false>(
                            cells,
                            &references,
                            &[0],
                            &[1],
                            1,
                            1,
                            2,
                            2,
                        )
                    };
                    std::hint::black_box(result.unwrap());
                }
                eprintln!(
                    "run={run} cache={cache} ns/block={:.1}",
                    start.elapsed().as_nanos() as f64 / 100_000.0
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn benchmark_halfpel_generation_720p() {
        use std::hint::black_box;
        use std::time::Instant;

        let width = 1280;
        let height = 720;
        let reference = Yuv420Picture {
            width,
            height,
            frame_num: 0,
            pic_order_cnt_lsb: 0,
            pic_order_cnt_msb: 0,
            pic_order_cnt: 0,
            luma: (0..width * height)
                .map(|index| ((index * 37 + index / width * 17) & 255) as u8)
                .collect(),
            cb: vec![128; width * height / 4],
            cr: vec![128; width * height / 4],
            motion: vec![[MotionCell::default(); 4]; width * height / 256],
            reference_pocs: Default::default(),
            luma_half: std::sync::OnceLock::new(),
        };
        let mut times = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            black_box(HalfPelPlanes::new(black_box(&reference)));
            times.push(start.elapsed());
        }
        times.sort_unstable();
        eprintln!("720p half-pel generation median: {:?}", times[2]);
    }

    #[test]
    #[ignore]
    fn benchmark_integer_luma_row_copy() {
        use std::hint::black_box;
        use std::time::Instant;

        let width = 1280;
        let height = 720;
        let plane: Vec<u8> = (0..width * height)
            .map(|index| ((index * 37 + index / width * 17) & 255) as u8)
            .collect();
        let mut scalar_times = Vec::new();
        let mut bulk_times = Vec::new();
        for round in 0..6 {
            let run = |bulk: bool| {
                let start = Instant::now();
                for repeat in 0..16 {
                    for mb_y in 0..height / 16 - 1 {
                        for mb_x in 0..width / 16 - 1 {
                            let mut block = [0u8; 256];
                            let source_x = mb_x * 16 + 2 + repeat % 2;
                            let source_y = mb_y * 16 + 1;
                            if bulk {
                                for row in 0..16 {
                                    let source = (source_y + row) * width + source_x;
                                    block[row * 16..row * 16 + 16]
                                        .copy_from_slice(&plane[source..source + 16]);
                                }
                            } else {
                                for row in 0..16 {
                                    for col in 0..16 {
                                        block[row * 16 + col] = luma_quarter(
                                            &plane,
                                            width,
                                            height,
                                            ((source_x + col) * 4) as i32,
                                            ((source_y + row) * 4) as i32,
                                        );
                                    }
                                }
                            }
                            black_box(block);
                        }
                    }
                }
                start.elapsed()
            };
            if round % 2 == 0 {
                scalar_times.push(run(false));
                bulk_times.push(run(true));
            } else {
                bulk_times.push(run(true));
                scalar_times.push(run(false));
            }
        }
        scalar_times.sort_unstable();
        bulk_times.sort_unstable();
        eprintln!(
            "integer luma scalar={:?} bulk={:?}",
            scalar_times[scalar_times.len() / 2],
            bulk_times[bulk_times.len() / 2]
        );
    }

    #[test]
    fn cached_halfpel_matches_direct_interpolation_at_edges_and_all_phases() {
        let reference = Yuv420Picture {
            width: 16,
            height: 16,
            frame_num: 0,
            pic_order_cnt_lsb: 0,
            pic_order_cnt_msb: 0,
            pic_order_cnt: 0,
            luma: (0..256).map(|i| ((i * 37 + 19) & 255) as u8).collect(),
            cb: vec![128; 64],
            cr: vec![128; 64],
            motion: vec![[MotionCell::default(); 4]],
            reference_pocs: Default::default(),
            luma_half: std::sync::OnceLock::new(),
        };
        let half = HalfPelPlanes::new(&reference);
        for y4 in -12..76 {
            for x4 in -12..76 {
                assert_eq!(
                    luma_quarter_cached(&reference, &half, x4, y4),
                    luma_quarter(&reference.luma, 16, 16, x4, y4),
                    "sample ({x4}, {y4})"
                );
            }
        }
    }

    #[test]
    fn cached_halfpel_matches_direct_interpolation_across_simd_rows() {
        let width = 64;
        let height = 48;
        let reference = Yuv420Picture {
            width,
            height,
            frame_num: 0,
            pic_order_cnt_lsb: 0,
            pic_order_cnt_msb: 0,
            pic_order_cnt: 0,
            luma: (0..width * height)
                .map(|index| ((index * 73 + index / width * 19) & 255) as u8)
                .collect(),
            cb: vec![128; width * height / 4],
            cr: vec![128; width * height / 4],
            motion: vec![[MotionCell::default(); 4]; width * height / 256],
            reference_pocs: Default::default(),
            luma_half: std::sync::OnceLock::new(),
        };
        let half = HalfPelPlanes::new(&reference);
        for y4 in -8..height as i32 * 4 + 8 {
            for x4 in -8..width as i32 * 4 + 8 {
                assert_eq!(
                    luma_quarter_cached(&reference, &half, x4, y4),
                    luma_quarter(&reference.luma, width, height, x4, y4),
                    "sample ({x4}, {y4})"
                );
            }
        }
    }

    #[test]
    fn b_lists_order_short_term_frames_by_poc() {
        let picture = |poc| Yuv420Picture {
            width: 16,
            height: 16,
            frame_num: 0,
            pic_order_cnt_lsb: poc as u32,
            pic_order_cnt_msb: 0,
            pic_order_cnt: poc,
            luma: vec![0; 256],
            cb: vec![0; 64],
            cr: vec![0; 64],
            motion: vec![[MotionCell::default(); 4]],
            reference_pocs: Default::default(),
            luma_half: std::sync::OnceLock::new(),
        };
        let references = [picture(0), picture(8), picture(2), picture(10)];
        assert_eq!(
            b_reference_lists(&references, 4),
            (vec![2, 0, 1, 3], vec![1, 3, 2, 0])
        );
        assert_eq!(
            b_reference_lists(&references[..2], 4),
            (vec![0, 1], vec![1, 0])
        );
        assert_eq!(
            b_reference_lists(&references[..2], -2),
            (vec![0, 1], vec![1, 0])
        );
        assert_eq!(
            b_reference_lists(&[picture(4), picture(8), picture(0)], 4),
            (vec![2, 0, 1], vec![1, 0, 2])
        );
    }

    #[test]
    fn b_bidirectional_prediction_uses_default_and_implicit_weights() {
        assert_eq!(implicit_b_weights(4, 0, 8), (32, 32));
        assert_eq!(implicit_b_weights(2, 0, 8), (48, 16));
        assert_eq!(implicit_b_weights(2, 0, 0), (32, 32));
        let block0 = InterMacroblock {
            luma: [20; 256],
            cb: [21; 64],
            cr: [22; 64],
        };
        let block1 = InterMacroblock {
            luma: [100; 256],
            cb: [100; 64],
            cr: [100; 64],
        };
        let average = blend_b_macroblocks(&block0, &block1, None);
        assert_eq!(
            (average.luma[0], average.cb[0], average.cr[0]),
            (60, 61, 61)
        );
        let weighted = blend_b_macroblocks(&block0, &block1, Some((48, 16)));
        assert_eq!(
            (weighted.luma[0], weighted.cb[0], weighted.cr[0]),
            (40, 41, 42)
        );
    }

    #[test]
    fn b_region_blend_matches_scalar_for_signed_weights() {
        let mut first = InterMacroblock {
            luma: [0; 256],
            cb: [0; 64],
            cr: [0; 64],
        };
        let mut second = InterMacroblock {
            luma: [0; 256],
            cb: [0; 64],
            cr: [0; 64],
        };
        for index in 0..256 {
            first.luma[index] = (index * 37) as u8;
            second.luma[index] = (index * 91 + 13) as u8;
        }
        for region in 0..4 {
            for weights in [None, Some((48, 16)), Some((128, -64)), Some((-64, 128))] {
                let actual = blend_b_region(&first, &second, region, weights);
                let x = region % 2;
                let y = region / 2;
                let compact_first = InterRegion {
                    luma: std::array::from_fn(|index| {
                        first.luma[(y * 8 + index / 8) * 16 + x * 8 + index % 8]
                    }),
                    cb: std::array::from_fn(|index| {
                        first.cb[(y * 4 + index / 4) * 8 + x * 4 + index % 4]
                    }),
                    cr: std::array::from_fn(|index| {
                        first.cr[(y * 4 + index / 4) * 8 + x * 4 + index % 4]
                    }),
                };
                let compact_second = InterRegion {
                    luma: std::array::from_fn(|index| {
                        second.luma[(y * 8 + index / 8) * 16 + x * 8 + index % 8]
                    }),
                    cb: std::array::from_fn(|index| {
                        second.cb[(y * 4 + index / 4) * 8 + x * 4 + index % 4]
                    }),
                    cr: std::array::from_fn(|index| {
                        second.cr[(y * 4 + index / 4) * 8 + x * 4 + index % 4]
                    }),
                };
                let compact = blend_b_compact(&compact_first, &compact_second, weights);
                for row in 0..8 {
                    for col in 0..8 {
                        let index = (y * 8 + row) * 16 + x * 8 + col;
                        assert_eq!(compact.luma[row * 8 + col], actual.luma[index]);
                        let a = i32::from(first.luma[index]);
                        let b = i32::from(second.luma[index]);
                        let expected = match weights {
                            Some((w0, w1)) => ((a * w0 + b * w1 + 32) >> 6).clamp(0, 255) as u8,
                            None => ((a + b + 1) >> 1) as u8,
                        };
                        assert_eq!(
                            actual.luma[index], expected,
                            "region={region} weights={weights:?} index={index}"
                        );
                    }
                }
                for row in 0..4 {
                    for col in 0..4 {
                        let index = (y * 4 + row) * 8 + x * 4 + col;
                        assert_eq!(compact.cb[row * 4 + col], actual.cb[index]);
                        assert_eq!(compact.cr[row * 4 + col], actual.cr[index]);
                    }
                }
            }
        }
    }

    #[test]
    fn spatial_direct_uses_available_list_and_colocated_zero_rule() {
        let left = MotionCell {
            l0: None,
            l1: Some((0, [-3, 0])),
        };
        let colocated = MotionCell {
            l0: Some((0, [6, 1])),
            l1: None,
        };
        assert_eq!(
            spatial_direct_motion(Some(left), None, None, None, colocated, true),
            left
        );
        let colocated_zero = MotionCell {
            l0: Some((0, [1, 0])),
            l1: None,
        };
        assert_eq!(
            spatial_direct_motion(Some(left), None, None, None, colocated_zero, true).l1,
            Some((0, [0, 0]))
        );
        assert_eq!(
            spatial_direct_motion(Some(left), None, None, None, colocated_zero, false).l1,
            Some((0, [-3, 0]))
        );
        let no_neighbors = spatial_direct_motion(None, None, None, None, colocated, true);
        assert_eq!(no_neighbors.l0, Some((0, [0, 0])));
        assert_eq!(no_neighbors.l1, Some((0, [0, 0])));
    }

    #[test]
    fn temporal_direct_maps_reference_identity_and_scales_signed_vectors() {
        let lists = [vec![12, 0], vec![4]];
        let colocated = MotionCell {
            l0: Some((1, [16, -12])),
            l1: None,
        };
        let actual = temporal_direct_motion(colocated, &lists, &[4, 0, 12], 2, 8).unwrap();
        assert_eq!(actual.l0, Some((1, [4, -3])));
        assert_eq!(actual.l1, Some((0, [-12, 9])));
        let l1_only = MotionCell {
            l0: None,
            l1: Some((0, [-9, 7])),
        };
        let actual = temporal_direct_motion(l1_only, &lists, &[4, 0], 2, 8).unwrap();
        assert_eq!(actual.l0, Some((0, [5, -3])));
        assert_eq!(actual.l1, Some((0, [14, -10])));
        let intra = temporal_direct_motion(MotionCell::default(), &lists, &[0], 2, 8).unwrap();
        assert_eq!(intra.l0, Some((0, [0, 0])));
        assert_eq!(intra.l1, Some((0, [0, 0])));
        let equal_poc = temporal_direct_motion(colocated, &lists, &[0], 2, 0).unwrap();
        assert_eq!(equal_poc.l0, Some((0, [16, -12])));
        assert_eq!(equal_poc.l1, Some((0, [0, 0])));
        assert!(temporal_direct_motion(colocated, &lists, &[4], 2, 8).is_err());
    }

    #[test]
    fn b_partition_candidates_follow_four_by_four_neighbor_geometry() {
        let edge = |base| {
            std::array::from_fn(|i| MotionCell {
                l0: Some((0, [base + i as i32, 0])),
                l1: None,
            })
        };
        let left = edge(100);
        let above = edge(200);
        let upper_right = edge(300);
        let upper_left = edge(400);
        let neighbors = [
            Some(&left),
            Some(&above),
            Some(&upper_right),
            Some(&upper_left),
        ];
        let mut cells = [None; 16];
        // C for the left 8x16 partition is inside the above macroblock.
        assert_eq!(b_motion_neighbor(2, -1, &cells, neighbors), Some(above[14]));
        // C for the bottom 16x8 partition is unavailable; D is on the left.
        assert_eq!(b_motion_neighbor(4, 1, &cells, neighbors), None);
        assert_eq!(b_motion_neighbor(-1, 1, &cells, neighbors), Some(left[7]));
        assert_eq!(b_motion_neighbor(1, -1, &cells, neighbors), Some(above[13]));
        assert_eq!(b_motion_neighbor(-1, 2, &cells, neighbors), Some(left[11]));
        // A future subpartition is unavailable, not an available intra/unused-list cell.
        assert_eq!(b_motion_neighbor(2, 1, &cells, neighbors), None);
        cells[6] = Some(MotionCell::default());
        assert_eq!(
            b_motion_neighbor(2, 1, &cells, neighbors),
            Some(MotionCell::default())
        );
    }

    #[test]
    fn spatial_direct_median_includes_other_reference_vectors() {
        let left = MotionCell {
            l0: Some((0, [10, 0])),
            l1: None,
        };
        let above = MotionCell {
            l0: Some((0, [20, 0])),
            l1: None,
        };
        let upper_right = MotionCell {
            l0: Some((1, [30, 0])),
            l1: None,
        };
        let colocated = MotionCell {
            l0: Some((0, [8, 0])),
            l1: None,
        };
        let direct = spatial_direct_motion(
            Some(left),
            Some(above),
            Some(upper_right),
            None,
            colocated,
            true,
        );
        assert_eq!(direct.l0, Some((0, [20, 0])));
    }

    #[test]
    fn spatial_direct_does_not_replace_an_available_unused_list_neighbor() {
        let left = MotionCell {
            l0: Some((2, [4, 4])),
            l1: None,
        };
        let above = MotionCell {
            l0: Some((1, [12, 12])),
            l1: None,
        };
        let upper_left = MotionCell {
            l0: Some((1, [-100, -100])),
            l1: None,
        };
        let actual = spatial_direct_motion(
            Some(left),
            Some(above),
            Some(MotionCell::default()),
            Some(upper_left),
            MotionCell::default(),
            true,
        );
        assert_eq!(actual.l0, Some((1, [12, 12])));
        assert_eq!(actual.l1, None);
    }

    #[test]
    fn p_skip_zeros_only_for_unavailable_or_reference_zero_still_neighbors() {
        let predicted = [7, -3];
        assert_eq!(
            p_skip_motion(0, 1, None, Some((1, [0, 0])), predicted),
            [0, 0]
        );
        assert_eq!(
            p_skip_motion(1, 0, Some((1, [0, 0])), None, predicted),
            [0, 0]
        );
        assert_eq!(
            p_skip_motion(1, 1, Some((0, [0, 0])), Some((1, [0, 0])), predicted),
            [0, 0]
        );
        assert_eq!(
            p_skip_motion(1, 1, Some((1, [0, 0])), Some((1, [0, 0])), predicted),
            predicted
        );
        assert_eq!(
            p_skip_motion(1, 1, None, Some((1, [0, 0])), predicted),
            predicted
        );
    }

    #[test]
    fn list_one_prediction_uses_list_one_weights() {
        let reference = Yuv420Picture {
            width: 16,
            height: 16,
            frame_num: 0,
            pic_order_cnt_lsb: 0,
            pic_order_cnt_msb: 0,
            pic_order_cnt: 0,
            luma: vec![100; 256],
            cb: vec![80; 64],
            cr: vec![60; 64],
            motion: vec![[MotionCell::default(); 4]],
            reference_pocs: Default::default(),
            luma_half: std::sync::OnceLock::new(),
        };
        let weights = PredictionWeightTable {
            luma_denom: 0,
            chroma_denom: 0,
            list0: vec![],
            list1: vec![super::super::h264::PredictionWeight {
                luma_weight: 2,
                luma_offset: 0,
                chroma_weight: [2, 2],
                chroma_offset: [0, 0],
            }],
        };
        let block = predict_l1_16x16(&reference, 0, 0, [0, 0], Some(&weights), 0).unwrap();
        assert_eq!(block.luma, [200; 256]);
        assert_eq!(block.cb, [160; 64]);
        assert_eq!(block.cr, [120; 64]);
    }

    #[test]
    fn weighted_fractional_prediction_matches_scalar_samples() {
        let width = 48;
        let height = 48;
        let reference = Yuv420Picture {
            width,
            height,
            frame_num: 0,
            pic_order_cnt_lsb: 0,
            pic_order_cnt_msb: 0,
            pic_order_cnt: 0,
            luma: (0..width * height).map(|i| (i * 37 % 251) as u8).collect(),
            cb: (0..width * height / 4)
                .map(|i| (i * 19 % 253) as u8)
                .collect(),
            cr: (0..width * height / 4)
                .map(|i| (i * 29 % 247) as u8)
                .collect(),
            motion: vec![[MotionCell::default(); 4]; 9],
            reference_pocs: Default::default(),
            luma_half: std::sync::OnceLock::new(),
        };
        let weights = PredictionWeightTable {
            luma_denom: 1,
            chroma_denom: 1,
            list0: vec![super::super::h264::PredictionWeight {
                luma_weight: 3,
                luma_offset: -2,
                chroma_weight: [2, 3],
                chroma_offset: [1, -3],
            }],
            list1: vec![],
        };
        let weight = &weights.list0[0];
        for (mb_x, mb_y, motion) in [(1, 1, [1, 3]), (1, 1, [8, 0]), (0, 0, [-3, -5])] {
            let block =
                predict_l0_16x16(&reference, mb_x, mb_y, motion, Some(&weights), 0).unwrap();
            for y in 0..16 {
                for x in 0..16 {
                    let x4 = ((mb_x * 16 + x) as i32) * 4 + motion[0];
                    let y4 = ((mb_y * 16 + y) as i32) * 4 + motion[1];
                    let expected = weighted_single(
                        luma_quarter(&reference.luma, width, height, x4, y4),
                        weight.luma_weight,
                        weight.luma_offset,
                        weights.luma_denom,
                    );
                    assert_eq!(
                        block.luma[y * 16 + x],
                        expected,
                        "luma ({mb_x},{mb_y}) ({x},{y})"
                    );
                }
            }
            for y in 0..8 {
                for x in 0..8 {
                    let x8 = ((mb_x * 8 + x) as i32) * 8 + motion[0];
                    let y8 = ((mb_y * 8 + y) as i32) * 8 + motion[1];
                    for (channel, (source, actual)) in
                        [(&reference.cb, &block.cb), (&reference.cr, &block.cr)]
                            .into_iter()
                            .enumerate()
                    {
                        let expected = weighted_single(
                            chroma_eighth(source, width / 2, height / 2, x8, y8),
                            weight.chroma_weight[channel],
                            weight.chroma_offset[channel],
                            weights.chroma_denom,
                        );
                        assert_eq!(
                            actual[y * 8 + x],
                            expected,
                            "chroma {channel} ({mb_x},{mb_y}) ({x},{y})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn integer_motion_row_copy_matches_scalar_luma() {
        let reference = Yuv420Picture {
            width: 32,
            height: 32,
            frame_num: 0,
            pic_order_cnt_lsb: 0,
            pic_order_cnt_msb: 0,
            pic_order_cnt: 0,
            luma: (0..1024).map(|i| (i * 37 % 251) as u8).collect(),
            cb: vec![80; 256],
            cr: vec![60; 256],
            motion: vec![[MotionCell::default(); 4]; 4],
            reference_pocs: Default::default(),
            luma_half: std::sync::OnceLock::new(),
        };
        let motion = [8, 4];
        let block = predict_l0_16x16(&reference, 0, 0, motion, None, 0).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(
                    block.luma[y * 16 + x],
                    luma_quarter(
                        &reference.luma,
                        32,
                        32,
                        x as i32 * 4 + motion[0],
                        y as i32 * 4 + motion[1],
                    )
                );
            }
        }
        for region in 0..4 {
            let region_block = predict_inter_8x8(&reference, 0, 0, motion, region).unwrap();
            let x0 = region % 2 * 8;
            let y0 = region / 2 * 8;
            for y in y0..y0 + 8 {
                for x in x0..x0 + 8 {
                    assert_eq!(
                        region_block.luma[y * 16 + x],
                        luma_quarter(
                            &reference.luma,
                            32,
                            32,
                            x as i32 * 4 + motion[0],
                            y as i32 * 4 + motion[1],
                        )
                    );
                }
            }
        }
    }

    #[test]
    fn integer_chroma_motion_row_copy_matches_interpolation() {
        let reference = Yuv420Picture {
            width: 48,
            height: 48,
            frame_num: 0,
            pic_order_cnt_lsb: 0,
            pic_order_cnt_msb: 0,
            pic_order_cnt: 0,
            luma: vec![100; 48 * 48],
            cb: (0..24 * 24).map(|i| (i * 37 % 251) as u8).collect(),
            cr: (0..24 * 24).map(|i| (i * 19 % 253) as u8).collect(),
            motion: vec![[MotionCell::default(); 4]; 9],
            reference_pocs: Default::default(),
            luma_half: std::sync::OnceLock::new(),
        };
        for motion in [[8, 16], [-8, -8], [16, 0]] {
            let block = predict_l0_16x16(&reference, 1, 1, motion, None, 0).unwrap();
            for region in 0..4 {
                let part = predict_inter_8x8(&reference, 1, 1, motion, region).unwrap();
                for y in (region / 2) * 4..(region / 2) * 4 + 4 {
                    for x in (region % 2) * 4..(region % 2) * 4 + 4 {
                        let x8 = ((8 + x) as i32) * 8 + motion[0];
                        let y8 = ((8 + y) as i32) * 8 + motion[1];
                        let cb = chroma_eighth(&reference.cb, 24, 24, x8, y8);
                        let cr = chroma_eighth(&reference.cr, 24, 24, x8, y8);
                        assert_eq!(block.cb[y * 8 + x], cb);
                        assert_eq!(block.cr[y * 8 + x], cr);
                        assert_eq!(part.cb[y * 8 + x], cb);
                        assert_eq!(part.cr[y * 8 + x], cr);
                    }
                }
            }
        }
    }

    #[test]
    fn cached_luma_rows_match_every_fractional_phase() {
        let reference = Yuv420Picture {
            width: 64,
            height: 64,
            frame_num: 0,
            pic_order_cnt_lsb: 0,
            pic_order_cnt_msb: 0,
            pic_order_cnt: 0,
            luma: (0..64 * 64).map(|i| (i * 37 % 251) as u8).collect(),
            cb: vec![0; 32 * 32],
            cr: vec![0; 32 * 32],
            motion: vec![[MotionCell::default(); 4]; 16],
            reference_pocs: Default::default(),
            luma_half: std::sync::OnceLock::new(),
        };
        let half = HalfPelPlanes::new(&reference);
        for dy in -7..=7 {
            for dx in -7..=7 {
                for (region_x, region_y, span) in [(0, 0, 16), (8, 0, 8), (0, 8, 8), (8, 8, 8)] {
                    let mut output = [0; 256];
                    assert!(predict_cached_luma_region(
                        &reference,
                        &half,
                        1,
                        1,
                        region_x,
                        region_y,
                        span,
                        [dx, dy],
                        &mut output,
                        16,
                        region_x,
                        region_y,
                    ));
                    for y in region_y..region_y + span {
                        for x in region_x..region_x + span {
                            let x4 = ((16 + x) as i32) * 4 + dx;
                            let y4 = ((16 + y) as i32) * 4 + dy;
                            assert_eq!(
                                output[y * 16 + x],
                                luma_quarter_cached(&reference, &half, x4, y4),
                                "phase ({dx}, {dy}) at ({x}, {y})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn benchmark_fractional_luma_rows() {
        use std::hint::black_box;
        use std::time::Instant;

        let reference = Yuv420Picture {
            width: 1280,
            height: 720,
            frame_num: 0,
            pic_order_cnt_lsb: 0,
            pic_order_cnt_msb: 0,
            pic_order_cnt: 0,
            luma: (0..1280 * 720).map(|i| (i * 37 % 251) as u8).collect(),
            cb: vec![0; 640 * 360],
            cr: vec![0; 640 * 360],
            motion: vec![[MotionCell::default(); 4]; 80 * 45],
            reference_pocs: Default::default(),
            luma_half: std::sync::OnceLock::new(),
        };
        let half = HalfPelPlanes::new(&reference);
        let run = |bulk: bool| {
            let mut output = [0u8; 256];
            let start = Instant::now();
            for i in 0..15_000 {
                let mb_x = black_box(1 + i % 78);
                let mb_y = black_box(1 + i % 43);
                let motion = black_box([1, 3]);
                if bulk {
                    assert!(predict_cached_luma_region(
                        &reference,
                        &half,
                        mb_x,
                        mb_y,
                        0,
                        0,
                        16,
                        motion,
                        &mut output,
                        16,
                        0,
                        0,
                    ));
                } else {
                    for y in 0..16 {
                        for x in 0..16 {
                            let x4 = ((mb_x * 16 + x) as i32) * 4 + motion[0];
                            let y4 = ((mb_y * 16 + y) as i32) * 4 + motion[1];
                            output[y * 16 + x] = luma_quarter_cached(&reference, &half, x4, y4);
                        }
                    }
                }
                black_box(output[0]);
            }
            start.elapsed()
        };
        for _ in 0..3 {
            eprintln!("luma pixel={:?}, row={:?}", run(false), run(true));
        }
    }

    #[test]
    fn quarter_luma_uses_six_tap_and_rounds_quarters() {
        let plane: Vec<u8> = (0..16)
            .flat_map(|y| (0..16).map(move |x| (4 * x + 8 * y) as u8))
            .collect();
        let base = 4 * 6 + 8 * 6;
        assert_eq!(luma_quarter(&plane, 16, 16, 6 * 4, 6 * 4), base);
        assert_eq!(luma_quarter(&plane, 16, 16, 6 * 4 + 2, 6 * 4), base + 2);
        assert_eq!(luma_quarter(&plane, 16, 16, 6 * 4, 6 * 4 + 2), base + 4);
        assert_eq!(luma_quarter(&plane, 16, 16, 6 * 4 + 2, 6 * 4 + 2), base + 6);
        assert_eq!(luma_quarter(&plane, 16, 16, 6 * 4 + 1, 6 * 4 + 1), base + 3);
        for fy in 0..4 {
            for fx in 0..4 {
                assert_eq!(
                    luma_quarter(&plane, 16, 16, 6 * 4 + fx, 6 * 4 + fy),
                    base + fx as u8 + 2 * fy as u8,
                    "quarter-pixel phase ({fx}, {fy})"
                );
            }
        }
        assert_eq!(luma_quarter(&plane, 16, 16, -10, -10), 0);
    }

    #[test]
    fn chroma_interpolation_and_weighting_follow_2003_rounding() {
        let plane = [0, 64, 128, 255];
        assert_eq!(chroma_eighth(&plane, 2, 2, 4, 0), 32);
        assert_eq!(chroma_eighth(&plane, 2, 2, 0, 4), 64);
        assert_eq!(chroma_eighth(&plane, 2, 2, 4, 4), 112);
        assert_eq!(weighted_single(101, 3, -2, 1), 150);
        assert_eq!(weighted_single(255, 2, 0, 0), 255);
    }

    #[test]
    fn chroma_fast_path_matches_clamped_sampling() {
        let width = 11;
        let height = 9;
        let plane: Vec<u8> = (0..width * height)
            .map(|index| (index * 37 % 256) as u8)
            .collect();
        for y8 in -16..(height as i32 * 8 + 16) {
            for x8 in -16..(width as i32 * 8 + 16) {
                let x = x8.div_euclid(8);
                let y = y8.div_euclid(8);
                let fx = x8.rem_euclid(8);
                let fy = y8.rem_euclid(8);
                let a = sample(&plane, width, height, x, y);
                let b = sample(&plane, width, height, x + 1, y);
                let c = sample(&plane, width, height, x, y + 1);
                let d = sample(&plane, width, height, x + 1, y + 1);
                let expected = (((8 - fx) * (8 - fy) * a
                    + fx * (8 - fy) * b
                    + (8 - fx) * fy * c
                    + fx * fy * d
                    + 32)
                    >> 6) as u8;
                assert_eq!(chroma_eighth(&plane, width, height, x8, y8), expected);
            }
        }
    }

    #[test]
    fn chroma_region_matches_pixel_interpolation() {
        let reference = Yuv420Picture {
            width: 64,
            height: 64,
            frame_num: 0,
            pic_order_cnt_lsb: 0,
            pic_order_cnt_msb: 0,
            pic_order_cnt: 0,
            luma: vec![0; 64 * 64],
            cb: (0..32 * 32).map(|i| (i * 37) as u8).collect(),
            cr: (0..32 * 32).map(|i| (i * 91 + 13) as u8).collect(),
            motion: vec![[MotionCell::default(); 4]; 16],
            reference_pocs: Default::default(),
            luma_half: std::sync::OnceLock::new(),
        };
        for size in [2, 4, 8] {
            for y in [0, 4, 12, 24] {
                for x in [0, 4, 12, 24] {
                    for dy in -9..=9 {
                        for dx in -9..=9 {
                            let mut cb = vec![0; size * size];
                            let mut cr = vec![0; size * size];
                            if !predict_chroma_region(
                                &reference,
                                x,
                                y,
                                [dx, dy],
                                size,
                                &mut cb,
                                &mut cr,
                            ) {
                                continue;
                            }
                            for row in 0..size {
                                for col in 0..size {
                                    let x8 = ((x + col) as i32) * 8 + dx;
                                    let y8 = ((y + row) as i32) * 8 + dy;
                                    assert_eq!(
                                        cb[row * size + col],
                                        chroma_eighth(&reference.cb, 32, 32, x8, y8)
                                    );
                                    assert_eq!(
                                        cr[row * size + col],
                                        chroma_eighth(&reference.cr, 32, 32, x8, y8)
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "run explicitly when measuring chroma block prediction"]
    fn benchmark_chroma_region() {
        let reference = Yuv420Picture {
            width: 1280,
            height: 720,
            frame_num: 0,
            pic_order_cnt_lsb: 0,
            pic_order_cnt_msb: 0,
            pic_order_cnt: 0,
            luma: vec![0; 1280 * 720],
            cb: (0..640 * 360).map(|i| (i * 37) as u8).collect(),
            cr: (0..640 * 360).map(|i| (i * 91 + 13) as u8).collect(),
            motion: vec![[MotionCell::default(); 4]; 80 * 45],
            reference_pocs: Default::default(),
            luma_half: std::sync::OnceLock::new(),
        };
        for size in [2, 8] {
            let count = if size == 2 { 100_000 } else { 10_000 };
            for bulk in [false, true] {
                let start = std::time::Instant::now();
                let mut cb = [0u8; 64];
                let mut cr = [0u8; 64];
                for i in 0..count {
                    let x = 16 + i % 600;
                    let y = 16 + i % 320;
                    let motion = [3, 5];
                    if bulk {
                        assert!(predict_chroma_region(
                            &reference,
                            x,
                            y,
                            motion,
                            size,
                            &mut cb[..size * size],
                            &mut cr[..size * size],
                        ));
                    } else {
                        for row in 0..size {
                            for col in 0..size {
                                let x8 = ((x + col) as i32) * 8 + motion[0];
                                let y8 = ((y + row) as i32) * 8 + motion[1];
                                cb[row * size + col] =
                                    chroma_eighth(&reference.cb, 640, 360, x8, y8);
                                cr[row * size + col] =
                                    chroma_eighth(&reference.cr, 640, 360, x8, y8);
                            }
                        }
                    }
                    std::hint::black_box((&cb[..size * size], &cr[..size * size]));
                }
                eprintln!(
                    "chroma {count} {size}x{size} blocks bulk={bulk}: {:?}",
                    start.elapsed()
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn benchmark_chroma_interior_path() {
        use std::hint::black_box;
        use std::time::Instant;

        let width = 640;
        let height = 360;
        let plane: Vec<u8> = (0..width * height)
            .map(|index| (index * 37 % 256) as u8)
            .collect();
        let old = |x8: i32, y8: i32| {
            let x = x8.div_euclid(8);
            let y = y8.div_euclid(8);
            let fx = x8.rem_euclid(8);
            let fy = y8.rem_euclid(8);
            let a = sample(&plane, width, height, x, y);
            let b = sample(&plane, width, height, x + 1, y);
            let c = sample(&plane, width, height, x, y + 1);
            let d = sample(&plane, width, height, x + 1, y + 1);
            (((8 - fx) * (8 - fy) * a + fx * (8 - fy) * b + (8 - fx) * fy * c + fx * fy * d + 32)
                >> 6) as u8
        };
        let run = |fast: bool| {
            let start = Instant::now();
            let mut checksum = 0u64;
            for index in 0..2_000_000 {
                let x8 = black_box((index % 638) as i32 * 8 + 3);
                let y8 = black_box((index % 358) as i32 * 8 + 5);
                checksum += u64::from(if fast {
                    chroma_eighth(&plane, width, height, x8, y8)
                } else {
                    old(x8, y8)
                });
            }
            black_box(checksum);
            start.elapsed()
        };
        for _ in 0..3 {
            eprintln!("chroma original={:?}, fast={:?}", run(false), run(true));
        }
    }

    #[test]
    fn median_prediction_uses_left_only_exception_and_zero_for_missing_neighbors() {
        assert_eq!(motion_predictor(None, Some([9, -3]), None, None), [0, 0]);
        assert_eq!(motion_predictor(None, None, Some([9, -3]), None), [0, 0]);
        assert_eq!(motion_predictor(Some([9, -3]), None, None, None), [9, -3]);
        assert_eq!(
            motion_predictor_for_ref(
                Some((1, [17, 0])),
                Some((0, [2, -5])),
                Some((2, [-9, 4])),
                None,
                0,
            ),
            [2, -5]
        );
        let left = Some((0, [0, 0]));
        let above = Some((1, [10, 10]));
        let upper_left = Some((0, [200, 200]));
        assert_eq!(
            motion_predictor_for_ref(left, above, Some((u8::MAX, [0, 0])), upper_left, 0,),
            [0, 0]
        );
        assert_eq!(
            motion_predictor_for_ref(left, above, None, upper_left, 0),
            [10, 10]
        );
        assert_eq!(
            motion_predictor_for_ref(Some((1, [9, 0])), Some((u8::MAX, [0, 0])), None, None, 0,),
            [0, 0]
        );
        assert_eq!(
            motion_predictor_for_ref(Some((1, [9, 0])), None, None, None, 0),
            [9, 0]
        );
    }

    #[test]
    fn p_eight_by_sixteen_left_partition_uses_above_midpoint_as_c_neighbor() {
        let mut above = InterMbState {
            qp: 26,
            skipped: false,
            intra16: false,
            intra_nxn: false,
            transform8x8: false,
            modes4: [2; 16],
            modes8: [2; 4],
            motion4: [[0; 2]; 16],
            mvd4: [[0; 2]; 16],
            refs4: [0; 16],
            coded: CodedBlockPattern {
                luma: 0,
                chroma: 0,
                pcm: false,
            },
            coded_luma4: 0,
            chroma_mode: 0,
            luma_dc_coded: false,
            luma_ac_right: [false; 4],
            luma_ac_bottom: [false; 4],
            chroma_dc_coded: [false; 2],
            chroma_ac_right: [[false; 2]; 2],
            chroma_ac_bottom: [[false; 2]; 2],
        };
        above.motion4[14] = [3, -4];
        above.refs4[14] = 0;
        let mut upper_right = above;
        upper_right.motion4[12] = [71, 43];
        upper_right.refs4[12] = 1;
        assert_eq!(
            p_top_right_candidate(2, Some(&above), Some(&upper_right)),
            Some((0, [3, -4]))
        );
        assert_eq!(
            p_top_right_candidate(0, Some(&above), Some(&upper_right)),
            Some((1, [71, 43]))
        );
        above.intra16 = true;
        assert_eq!(
            p_top_right_candidate(2, Some(&above), Some(&upper_right)),
            Some((u8::MAX, [0, 0]))
        );
    }

    #[test]
    fn sixteen_by_eight_prediction_preserves_each_half() {
        let top = InterMacroblock {
            luma: [1; 256],
            cb: [2; 64],
            cr: [3; 64],
        };
        let bottom = InterMacroblock {
            luma: [4; 256],
            cb: [5; 64],
            cr: [6; 64],
        };
        let block = combine_16x8(top, bottom);
        assert_eq!(&block.luma[..128], &[1; 128]);
        assert_eq!(&block.luma[128..], &[4; 128]);
        assert_eq!(&block.cb[..32], &[2; 32]);
        assert_eq!(&block.cr[32..], &[6; 32]);
    }

    #[test]
    fn eight_by_sixteen_prediction_preserves_each_half() {
        let left = InterMacroblock {
            luma: [1; 256],
            cb: [2; 64],
            cr: [3; 64],
        };
        let right = InterMacroblock {
            luma: [4; 256],
            cb: [5; 64],
            cr: [6; 64],
        };
        let block = combine_8x16(left, right);
        for row in 0..16 {
            assert_eq!(&block.luma[row * 16..row * 16 + 8], &[1; 8]);
            assert_eq!(&block.luma[row * 16 + 8..row * 16 + 16], &[4; 8]);
        }
        for row in 0..8 {
            assert_eq!(&block.cb[row * 8..row * 8 + 4], &[2; 4]);
            assert_eq!(&block.cb[row * 8 + 4..row * 8 + 8], &[5; 4]);
            assert_eq!(&block.cr[row * 8..row * 8 + 4], &[3; 4]);
            assert_eq!(&block.cr[row * 8 + 4..row * 8 + 8], &[6; 4]);
        }
    }

    #[test]
    fn eight_by_eight_prediction_preserves_each_quadrant() {
        let blocks: [InterMacroblock; 4] = std::array::from_fn(|part| InterMacroblock {
            luma: [part as u8 + 1; 256],
            cb: [part as u8 + 5; 64],
            cr: [part as u8 + 9; 64],
        });
        let block = combine_8x8(&blocks);
        for part in 0..4 {
            let x = part % 2;
            let y = part / 2;
            for row in 0..8 {
                let offset = (y * 8 + row) * 16 + x * 8;
                assert_eq!(&block.luma[offset..offset + 8], &[part as u8 + 1; 8]);
            }
            for row in 0..4 {
                let offset = (y * 4 + row) * 8 + x * 4;
                assert_eq!(&block.cb[offset..offset + 4], &[part as u8 + 5; 4]);
                assert_eq!(&block.cr[offset..offset + 4], &[part as u8 + 9; 4]);
            }
        }
    }

    #[test]
    fn inter_luma_four_by_four_residual_stays_in_its_block() {
        let mut block = InterMacroblock {
            luma: [100; 256],
            cb: [100; 64],
            cr: [100; 64],
        };
        let mut levels = [[0; 16]; 16];
        levels[0][0] = 32;
        add_luma_residual(&mut block, &InterLumaResidual::FourByFour(levels), 20).unwrap();
        assert_ne!(block.luma[0], 100);
        assert_eq!(block.luma[4], 100);
        assert_eq!(block.luma[16 * 4], 100);
        assert_eq!(block.cb, [100; 64]);
    }

    #[test]
    fn deblock_nonzero_mask_maps_block_scan_to_raster_cells() {
        let mut levels = [[0; 16]; 16];
        levels[1][0] = 1;
        levels[2][3] = -1;
        assert_eq!(
            coded_luma4_mask(&InterLumaResidual::FourByFour(levels)),
            (1 << 1) | (1 << 4)
        );
        let mut levels8 = [[0; 64]; 4];
        levels8[1][17] = 1;
        assert_eq!(
            coded_luma4_mask(&InterLumaResidual::EightByEight(levels8)),
            (1 << 2) | (1 << 3) | (1 << 6) | (1 << 7)
        );
        assert_eq!(coded_luma4_mask(&InterLumaResidual::None), 0);
    }

    #[test]
    fn inter_luma_eight_by_eight_residual_stays_in_its_region() {
        let mut block = InterMacroblock {
            luma: [100; 256],
            cb: [100; 64],
            cr: [100; 64],
        };
        let mut levels = [[0; 64]; 4];
        levels[1][0] = 32;
        add_luma_residual(&mut block, &InterLumaResidual::EightByEight(levels), 20).unwrap();
        assert_eq!(block.luma[0], 100);
        assert_ne!(block.luma[8], 100);
        assert_eq!(block.luma[16 * 8], 100);
        assert_eq!(block.cr, [100; 64]);
    }

    #[test]
    fn chroma_dc_residual_modifies_only_the_selected_plane() {
        let mut block = InterMacroblock {
            luma: [90; 256],
            cb: [100; 64],
            cr: [110; 64],
        };
        add_chroma_residual(&mut block, &[0; 4], &[[0; 15]; 4], 20, 0).unwrap();
        assert_eq!(block.cb, [100; 64]);
        add_chroma_residual(&mut block, &[8, 0, 0, 0], &[[0; 15]; 4], 20, 1).unwrap();
        assert!(block.cr.iter().any(|&value| value != 110));
        assert_eq!(block.cb, [100; 64]);
        assert_eq!(block.luma, [90; 256]);
    }

    #[test]
    fn chroma_ac_residual_combines_with_dc_before_clamping() {
        let mut block = InterMacroblock {
            luma: [90; 256],
            cb: [100; 64],
            cr: [110; 64],
        };
        let mut ac = [[0; 15]; 4];
        ac[0][0] = 32;
        add_chroma_residual(&mut block, &[8, 0, 0, 0], &ac, 20, 0).unwrap();
        assert_ne!(block.cb[0], block.cb[3]);
        assert_eq!(block.cr, [110; 64]);
        assert_eq!(block.luma, [90; 256]);
    }
}
