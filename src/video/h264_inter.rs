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
    motion: [[i32; 2]; 4],
    motion4: [[i32; 2]; 16],
    mvd4: [[i32; 2]; 16],
    refs4: [u8; 16],
    coded: CodedBlockPattern,
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
    let available = [left, above, upper_right];
    if available.iter().filter(|mb| mb.is_some()).count() == 1 {
        return available.into_iter().flatten().next().unwrap();
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
            neighbor.and_then(|cell| if list == 0 { cell.l0 } else { cell.l1 })
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
        || sps.height != sps.frame_height_mbs * 16
        || reference.width != sps.width as usize
        || reference.height != sps.height as usize
    {
        return Err(AvcError::Unsupported("P picture format"));
    }
    let slice = parse_cabac_inter_slice(nal, sps, &pps.core)?;
    if std::env::var_os("WEBMEDIA_TRACE_P").is_some() {
        eprintln!(
            "P header: type={} first_mb={} active_refs={} available_refs={} frame_num={} reorder={:?} reference_frames={:?}",
            slice.slice_type,
            slice.first_mb,
            slice.ref_idx_l0,
            references.len(),
            slice.frame_num,
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
    for y in 0..height_mbs {
        for x in 0..width_mbs {
            let index = y * width_mbs + x;
            let left = (x > 0).then(|| states[index - 1]).flatten();
            let above = (y > 0).then(|| states[index - width_mbs]).flatten();
            let upper_right = (y > 0 && x + 1 < width_mbs)
                .then(|| states[index - width_mbs + 1])
                .flatten();
            let upper_left = (y > 0 && x > 0)
                .then(|| states[index - width_mbs - 1])
                .flatten();
            let skipped = decoder.inter_mb_skip_flag(
                &mut inter,
                left.map(|mb| mb.skipped),
                above.map(|mb| mb.skipped),
            )?;
            let above_bottom = above
                .filter(|mb| !mb.intra16 && !mb.intra_nxn)
                .map(|mb| mb.motion[2]);
            let upper_right_bottom = upper_right
                .filter(|mb| !mb.intra16 && !mb.intra_nxn)
                .map(|mb| mb.motion[2]);
            let left_reference = left
                .filter(|mb| !mb.intra16 && !mb.intra_nxn)
                .map(|mb| (mb.refs4[3], mb.motion4[3]));
            let above_reference = above
                .filter(|mb| !mb.intra16 && !mb.intra_nxn)
                .map(|mb| (mb.refs4[12], mb.motion4[12]));
            let upper_right_reference = upper_right
                .filter(|mb| !mb.intra16 && !mb.intra_nxn)
                .map(|mb| (mb.refs4[12], mb.motion4[12]));
            let upper_left_reference = upper_left
                .filter(|mb| !mb.intra16 && !mb.intra_nxn)
                .map(|mb| (mb.refs4[15], mb.motion4[15]));
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
                            motion: [[0; 2]; 4],
                            motion4: [[0; 2]; 16],
                            mvd4: [[0; 2]; 16],
                            refs4: [0; 16],
                            coded,
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
                        motion: [[0; 2]; 4],
                        motion4: [[0; 2]; 16],
                        mvd4: [[0; 2]; 16],
                        refs4: [0; 16],
                        coded,
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
                                    left.filter(|mb| !mb.intra16 && !mb.intra_nxn).map(|mb| {
                                        let cell = cy as usize * 4 + 3;
                                        (mb.refs4[cell], mb.motion4[cell])
                                    })
                                } else if cy < 0 && (0..4).contains(&cx) {
                                    above.filter(|mb| !mb.intra16 && !mb.intra_nxn).map(|mb| {
                                        let cell = 12 + cx as usize;
                                        (mb.refs4[cell], mb.motion4[cell])
                                    })
                                } else if cx >= 4 && cy < 0 {
                                    upper_right
                                        .filter(|mb| !mb.intra16 && !mb.intra_nxn)
                                        .map(|mb| (mb.refs4[12], mb.motion4[12]))
                                } else if cx < 0 && cy < 0 {
                                    upper_left
                                        .filter(|mb| !mb.intra16 && !mb.intra_nxn)
                                        .map(|mb| (mb.refs4[15], mb.motion4[15]))
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
                            upper_candidate,
                            None,
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
                        let bottom_predictor = if kind == 1 {
                            left.map_or(top, |mb| mb.motion[3])
                        } else {
                            upper_right_bottom.unwrap_or_else(|| {
                                motion_predictor(
                                    Some(top),
                                    above
                                        .filter(|mb| !mb.intra16 && !mb.intra_nxn)
                                        .map(|mb| mb.motion[3]),
                                    upper_right_bottom,
                                    above_bottom,
                                )
                            })
                        };
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
                motion,
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
            let state = states[index].expect("decoded P macroblock");
            picture.motion[index] = [0, 3, 12, 15].map(|cell| MotionCell {
                l0: Some((state.refs4[cell], state.motion4[cell])),
                l1: None,
            });
            let ended = decoder.terminate()?;
            if ended != (index + 1 == states.len()) {
                return Err(AvcError::Unsupported(
                    "P picture has multiple or incomplete slices",
                ));
            }
        }
    }
    if !slice.deblocking_disabled {
        let macroblocks: Vec<_> = states
            .iter()
            .enumerate()
            .map(|(index, state)| {
                let state = state.expect("decoded P macroblock");
                DeblockMb {
                    qp: state.qp,
                    intra: state.intra16 || state.intra_nxn,
                    transform8x8: state.transform8x8,
                    coded_luma: state.coded.luma as u8,
                    motion: picture.motion[index],
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
    intra16: bool,
    intra_nxn: bool,
    modes4: [u8; 16],
    modes8: [u8; 4],
    chroma_mode: u8,
    luma_dc: bool,
    coded: CodedBlockPattern,
    transform8x8: bool,
    mvd: [[[i32; 2]; 4]; 2],
    luma_right: [bool; 4],
    luma_bottom: [bool; 4],
    chroma_dc: [bool; 2],
    chroma_right: [[bool; 2]; 2],
    chroma_bottom: [[bool; 2]; 2],
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

/// Decode progressive CABAC B slices using 16x16 and spatial-direct partitions.
/// Other partition shapes and explicit bipred weighting remain unsupported.
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
        || sps.height != sps.frame_height_mbs * 16
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
    if !slice.direct_spatial_mv_pred || !slice.reorder_l0.is_empty() || !slice.reorder_l1.is_empty()
    {
        return Err(AvcError::Unsupported("B motion or reference reordering"));
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
    let (list0, list1) = b_reference_lists(references, poc);
    if list0.len() < slice.ref_idx_l0 as usize || list1.len() < slice.ref_idx_l1 as usize {
        return Err(AvcError::Unsupported("B reference list length"));
    }
    if references
        .iter()
        .any(|picture| picture.width != sps.width as usize || picture.height != sps.height as usize)
    {
        return Err(AvcError::Unsupported("B reference dimensions"));
    }
    let width_mbs = sps.width_mbs as usize;
    let height_mbs = sps.frame_height_mbs as usize;
    let mut picture = Yuv420Picture {
        width: sps.width as usize,
        height: sps.height as usize,
        frame_num: slice.frame_num,
        pic_order_cnt_lsb: slice.pic_order_cnt_lsb,
        pic_order_cnt_msb: poc_msb,
        pic_order_cnt: poc,
        luma: vec![0; last_reference.luma.len()],
        cb: vec![0; last_reference.cb.len()],
        cr: vec![0; last_reference.cr.len()],
        motion: vec![[MotionCell::default(); 4]; width_mbs * height_mbs],
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
            let left = (x > 0).then(|| states[index - 1]).flatten();
            let above = (y > 0).then(|| states[index - width_mbs]).flatten();
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
                    intra16: false,
                    intra_nxn: true,
                    modes4,
                    modes8,
                    chroma_mode,
                    luma_dc: false,
                    coded,
                    transform8x8,
                    mvd: [[[0; 2]; 4]; 2],
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
                    intra16: true,
                    intra_nxn: false,
                    modes4: [2; 16],
                    modes8: [2; 4],
                    chroma_mode,
                    luma_dc: luma_dc.iter().any(|&level| level != 0),
                    coded,
                    transform8x8: false,
                    mvd: [[[0; 2]; 4]; 2],
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
            if kind > 21 {
                if trace_b {
                    eprintln!("B macroblock {index} ({x},{y}) has unsupported type {kind}");
                }
                return Err(AvcError::Unsupported("B macroblock partition mode"));
            }
            let mut motion = [MotionCell::default(); 4];
            let mut mvd = [[[0; 2]; 4]; 2];
            if kind == 0 {
                let left_cell = (x > 0).then(|| picture.motion[index - 1][1]);
                let above_cell = (y > 0).then(|| picture.motion[index - width_mbs][2]);
                let upper_right =
                    (y > 0 && x + 1 < width_mbs).then(|| picture.motion[index - width_mbs + 1][2]);
                let upper_left = (y > 0 && x > 0).then(|| picture.motion[index - width_mbs - 1][3]);
                for region in 0..4 {
                    let colocated = references[list1[0]].motion[index][region];
                    motion[region] = spatial_direct_motion(
                        left_cell,
                        above_cell,
                        upper_right,
                        upper_left,
                        colocated,
                        true,
                    );
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
                        } else if left.is_some_and(|state| state.direct) {
                            None
                        } else if x > 0 {
                            let cell = picture.motion[index - 1][if part == 1 { 3 } else { 1 }];
                            if list == 0 { cell.l0 } else { cell.l1 }.map(|(index, _)| index)
                        } else {
                            None
                        };
                        let above_ref = if part == 1 && !vertical {
                            Some(selected_refs[list][0])
                        } else if above.is_some_and(|state| state.direct) {
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
                        let left_region = if part == 1 && vertical {
                            0
                        } else if part == 1 {
                            3
                        } else {
                            1
                        };
                        let above_region = if part == 1 && !vertical {
                            0
                        } else if part == 1 {
                            3
                        } else {
                            2
                        };
                        let left_cell = if part == 1 && vertical {
                            Some(motion[0])
                        } else if x > 0 {
                            Some(picture.motion[index - 1][left_region])
                        } else {
                            None
                        };
                        let above_cell = if part == 1 && !vertical {
                            Some(motion[0])
                        } else if y > 0 {
                            Some(picture.motion[index - width_mbs][above_region])
                        } else {
                            None
                        };
                        let right_cell = if y > 0 && x + 1 < width_mbs {
                            Some(picture.motion[index - width_mbs + 1][2])
                        } else {
                            None
                        };
                        let upper_left_cell = if x > 0 && y > 0 {
                            Some(picture.motion[index - width_mbs - 1][3])
                        } else {
                            None
                        };
                        let from = |cell: MotionCell| if list == 0 { cell.l0 } else { cell.l1 };
                        let ref_index = selected_refs[list][partitions[part][0]];
                        let vector_for_ref = |cell: MotionCell| {
                            from(cell)
                                .filter(|(index, _)| *index == ref_index)
                                .map(|(_, vector)| vector)
                        };
                        let left_motion = left_cell.and_then(vector_for_ref);
                        let above_motion = above_cell.and_then(vector_for_ref);
                        let right_motion = right_cell.and_then(vector_for_ref);
                        let upper_left_motion = upper_left_cell.and_then(vector_for_ref);
                        let left_mvd = if part == 1 && vertical {
                            Some(mvd[list][0])
                        } else {
                            left.map(|state| state.mvd[list][left_region])
                        };
                        let above_mvd = if part == 1 && !vertical {
                            Some(mvd[list][0])
                        } else {
                            above.map(|state| state.mvd[list][above_region])
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
                            above_motion.unwrap_or_else(|| {
                                motion_predictor(
                                    left_motion,
                                    above_motion,
                                    right_motion,
                                    upper_left_motion,
                                )
                            })
                        } else if partition_count == 2 && !vertical && part == 1 {
                            left_motion.unwrap_or_else(|| {
                                motion_predictor(
                                    left_motion,
                                    above_motion,
                                    right_motion,
                                    upper_left_motion,
                                )
                            })
                        } else if partition_count == 2 && vertical && part == 0 {
                            left_motion.unwrap_or_else(|| {
                                motion_predictor(
                                    left_motion,
                                    above_motion,
                                    right_motion,
                                    upper_left_motion,
                                )
                            })
                        } else if partition_count == 2 && vertical && part == 1 {
                            right_motion.unwrap_or_else(|| {
                                motion_predictor(
                                    left_motion,
                                    above_motion,
                                    right_motion,
                                    upper_left_motion,
                                )
                            })
                        } else {
                            motion_predictor(
                                left_motion,
                                above_motion,
                                right_motion,
                                upper_left_motion,
                            )
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
            let transform8x8 = if coded.luma != 0 && pps.transform_8x8 {
                decoder.transform_size_8x8_flag(
                    &mut transform_contexts,
                    left.map(|state| state.transform8x8),
                    above.map(|state| state.transform8x8),
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
            let mut predicted = [(); 4].map(|_| InterMacroblock {
                luma: [0; 256],
                cb: [0; 64],
                cr: [0; 64],
            });
            for region in 0..4 {
                let list0_block = if let Some((ref_index, vector)) = motion[region].l0 {
                    let source = list0
                        .get(ref_index as usize)
                        .ok_or(AvcError::Unsupported("B list0 index"))?;
                    Some(predict_inter_8x8(
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
                    Some(predict_inter_8x8(
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
                        blend_b_region(&block0, &block1, region, weights)
                    }
                    (Some(block), None) | (None, Some(block)) => block,
                    (None, None) => {
                        return Err(AvcError::InvalidData("B block without prediction"));
                    }
                };
            }
            let mut block = combine_8x8(&predicted);
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
            picture.motion[index] = motion;
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
                intra16: false,
                intra_nxn: false,
                modes4: [2; 16],
                modes8: [2; 4],
                chroma_mode: 0,
                luma_dc: false,
                coded,
                transform8x8,
                mvd,
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
            .enumerate()
            .map(|(index, state)| {
                let state = state.expect("decoded B macroblock");
                DeblockMb {
                    qp: state.qp,
                    intra: state.intra16 || state.intra_nxn,
                    transform8x8: state.transform8x8,
                    coded_luma: state.coded.luma as u8,
                    motion: picture.motion[index],
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

#[cfg(test)]
pub(super) fn blend_b_macroblocks(
    list0: &InterMacroblock,
    list1: &InterMacroblock,
    weights: Option<(i32, i32)>,
) -> InterMacroblock {
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
    for (output, (&a, &b)) in block
        .luma
        .iter_mut()
        .zip(list0.luma.iter().zip(&list1.luma))
    {
        *output = blend(a, b);
    }
    for (output, (&a, &b)) in block.cb.iter_mut().zip(list0.cb.iter().zip(&list1.cb)) {
        *output = blend(a, b);
    }
    for (output, (&a, &b)) in block.cr.iter_mut().zip(list0.cr.iter().zip(&list1.cr)) {
        *output = blend(a, b);
    }
    block
}

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

impl HalfPelPlanes {
    fn new(reference: &Yuv420Picture) -> Self {
        let width = reference.width;
        let height = reference.height;
        let plane = &reference.luma;
        let mut horizontal_raw = vec![0i32; width * height];
        let mut horizontal_half = vec![0u8; width * height];
        let mut vertical_half = vec![0u8; width * height];
        let mut diagonal_half = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                let value = horizontal(plane, width, height, x as i32, y as i32);
                horizontal_raw[index] = value;
                horizontal_half[index] = clip((value + 16) >> 5);
                vertical_half[index] =
                    clip((vertical(plane, width, height, x as i32, y as i32) + 16) >> 5);
            }
        }
        for y in 0..height {
            for x in 0..width {
                let taps = six_tap(std::array::from_fn(|i| {
                    horizontal_raw
                        [(y as i32 + i as i32 - 2).clamp(0, height as i32 - 1) as usize * width + x]
                }));
                diagonal_half[y * width + x] = clip((taps + 512) >> 10);
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
    let at = |plane: &[u8], x: i32, y: i32| -> i32 {
        let x = x.clamp(0, width as i32 - 1) as usize;
        let y = y.clamp(0, height as i32 - 1) as usize;
        i32::from(plane[y * width + x])
    };
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
    let a = sample(plane, width, height, x, y);
    let b = sample(plane, width, height, x + 1, y);
    let c = sample(plane, width, height, x, y + 1);
    let d = sample(plane, width, height, x + 1, y + 1);
    (((8 - fx) * (8 - fy) * a + fx * (8 - fy) * b + (8 - fx) * fy * c + fx * fy * d + 32) >> 6)
        as u8
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
    for y in 0..16 {
        for x in 0..16 {
            let x4 = ((mb_x * 16 + x) as i32) * 4 + motion[0];
            let y4 = ((mb_y * 16 + y) as i32) * 4 + motion[1];
            let value = match half {
                Some(half) => luma_quarter_cached(reference, half, x4, y4),
                None => luma_quarter(&reference.luma, reference.width, reference.height, x4, y4),
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
    let chroma_width = reference.width / 2;
    let chroma_height = reference.height / 2;
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
            for row in cy * 4..cy * 4 + 4 {
                for col in cx * 4..cx * 4 + 4 {
                    let x4 = ((mb_x * 16 + col) as i32) * 4 + motion[0];
                    let y4 = ((mb_y * 16 + row) as i32) * 4 + motion[1];
                    let value = match half {
                        Some(half) => luma_quarter_cached(source, half, x4, y4),
                        None => luma_quarter(&source.luma, source.width, source.height, x4, y4),
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

fn predict_inter_8x8(
    reference: &Yuv420Picture,
    mb_x: usize,
    mb_y: usize,
    motion: [i32; 2],
    region: usize,
) -> Result<InterMacroblock, AvcError> {
    if mb_x * 16 >= reference.width || mb_y * 16 >= reference.height {
        return Err(AvcError::InvalidData("inter macroblock out of bounds"));
    }
    let mut block = InterMacroblock {
        luma: [0; 256],
        cb: [0; 64],
        cr: [0; 64],
    };
    if motion == [0, 0] {
        let luma_x = (region % 2) * 8;
        let luma_y = (region / 2) * 8;
        for row in luma_y..luma_y + 8 {
            let source = (mb_y * 16 + row) * reference.width + mb_x * 16 + luma_x;
            block.luma[row * 16 + luma_x..row * 16 + luma_x + 8]
                .copy_from_slice(&reference.luma[source..source + 8]);
        }
        let chroma_x = (region % 2) * 4;
        let chroma_y = (region / 2) * 4;
        for row in chroma_y..chroma_y + 4 {
            let source = (mb_y * 8 + row) * (reference.width / 2) + mb_x * 8 + chroma_x;
            let target = row * 8 + chroma_x..row * 8 + chroma_x + 4;
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
    for y in luma_y..luma_y + 8 {
        for x in luma_x..luma_x + 8 {
            let x4 = ((mb_x * 16 + x) as i32) * 4 + motion[0];
            let y4 = ((mb_y * 16 + y) as i32) * 4 + motion[1];
            block.luma[y * 16 + x] = match half {
                Some(half) => luma_quarter_cached(reference, half, x4, y4),
                None => luma_quarter(&reference.luma, reference.width, reference.height, x4, y4),
            };
        }
    }
    let chroma_width = reference.width / 2;
    let chroma_height = reference.height / 2;
    let chroma_x = (region % 2) * 4;
    let chroma_y = (region / 2) * 4;
    for y in chroma_y..chroma_y + 4 {
        for x in chroma_x..chroma_x + 4 {
            let x8 = ((mb_x * 8 + x) as i32) * 8 + motion[0];
            let y8 = ((mb_y * 8 + y) as i32) * 8 + motion[1];
            block.cb[y * 8 + x] = chroma_eighth(&reference.cb, chroma_width, chroma_height, x8, y8);
            block.cr[y * 8 + x] = chroma_eighth(&reference.cr, chroma_width, chroma_height, x8, y8);
        }
    }
    Ok(block)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn median_prediction_uses_the_only_available_reference() {
        let mb = InterMbState {
            qp: 26,
            skipped: false,
            intra16: false,
            intra_nxn: false,
            transform8x8: false,
            modes4: [2; 16],
            modes8: [2; 4],
            motion: [[9, -3]; 4],
            motion4: [[9, -3]; 16],
            mvd4: [[0, 0]; 16],
            refs4: [0; 16],
            coded: CodedBlockPattern {
                luma: 0,
                chroma: 0,
                pcm: false,
            },
            chroma_mode: 0,
            luma_dc_coded: false,
            luma_ac_right: [false; 4],
            luma_ac_bottom: [false; 4],
            chroma_dc_coded: [false; 2],
            chroma_ac_right: [[false; 2]; 2],
            chroma_ac_bottom: [[false; 2]; 2],
        };
        assert_eq!(
            motion_predictor(None, Some(mb.motion[2]), None, None),
            [9, -3]
        );
        assert_eq!(
            motion_predictor(Some(mb.motion[1]), None, None, None),
            [9, -3]
        );
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
