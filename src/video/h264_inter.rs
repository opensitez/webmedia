//! 2003 AVC inter-picture sample prediction for progressive 8-bit 4:2:0.

use super::h264::{
    AvcError, PictureParameters2005, PredictionWeightTable, SequenceParameters,
    parse_cabac_inter_slice,
};
use super::h264_cabac::{
    CabacDecoder, ChromaDcContexts, CodedBlockContexts, CodedBlockPattern, InterMbContexts,
    MbQpContexts, MotionVectorContexts,
};
use super::h264_high::Yuv420Picture;
use super::h264_transform::{chroma_qp, inverse_4x4_residual, inverse_chroma_dc};

#[derive(Clone, Copy)]
struct InterMbState {
    skipped: bool,
    motion: [[i32; 2]; 4],
    mvd: [[i32; 2]; 4],
    coded: CodedBlockPattern,
    chroma_dc_coded: [bool; 2],
}

fn median(a: i32, b: i32, c: i32) -> i32 {
    let mut values = [a, b, c];
    values.sort_unstable();
    values[1]
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

fn combine_16x8(top: InterMacroblock, bottom: InterMacroblock) -> InterMacroblock {
    let mut block = top;
    block.luma[128..].copy_from_slice(&bottom.luma[128..]);
    block.cb[32..].copy_from_slice(&bottom.cb[32..]);
    block.cr[32..].copy_from_slice(&bottom.cr[32..]);
    block
}

fn add_chroma_dc(
    block: &mut InterMacroblock,
    levels: &[i32; 4],
    qp: i32,
    channel: usize,
) -> Result<(), AvcError> {
    let scaled = inverse_chroma_dc(levels, qp)?;
    let plane = if channel == 0 {
        &mut block.cb
    } else {
        &mut block.cr
    };
    for quadrant in 0..4 {
        let mut coefficients = [0; 16];
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

/// Decode complete one-reference P pictures for the supported 2003 inter modes.
/// Mixed intra macroblocks and remaining partitions/residuals are rejected.
pub fn decode_cabac_p_2005(
    nal: &[u8],
    sps: &SequenceParameters,
    pps: &PictureParameters2005,
    reference: &Yuv420Picture,
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
        || reference.width != sps.width as usize
        || reference.height != sps.height as usize
    {
        return Err(AvcError::Unsupported("P picture format"));
    }
    let slice = parse_cabac_inter_slice(nal, sps, &pps.core)?;
    if slice.slice_type % 5 != 0 || slice.first_mb != 0 || slice.ref_idx_l0 != 1 {
        return Err(AvcError::Unsupported("P picture slice layout"));
    }
    if !slice.reorder_l0.is_empty() {
        return Err(AvcError::Unsupported("P reference list reordering"));
    }
    let width_mbs = sps.width_mbs as usize;
    let height_mbs = sps.frame_height_mbs as usize;
    let mut picture = Yuv420Picture {
        width: reference.width,
        height: reference.height,
        frame_num: slice.frame_num,
        pic_order_cnt_lsb: slice.pic_order_cnt_lsb,
        luma: vec![0; reference.luma.len()],
        cb: vec![0; reference.cb.len()],
        cr: vec![0; reference.cr.len()],
    };
    let mut states: Vec<Option<InterMbState>> = vec![None; width_mbs * height_mbs];
    let mut decoder = CabacDecoder::new(&slice.rbsp[slice.data_byte_offset..])?;
    let mut inter = InterMbContexts::new(slice.slice_qp, slice.cabac_init_idc, false)?;
    let mut vectors = MotionVectorContexts::new(slice.slice_qp, slice.cabac_init_idc)?;
    let mut patterns = CodedBlockContexts::new_inter(slice.slice_qp, slice.cabac_init_idc)?;
    let mut chroma_contexts = ChromaDcContexts::new_inter(slice.slice_qp, slice.cabac_init_idc)?;
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
            let left_top = left.map(|mb| mb.motion[1]);
            let above_bottom = above.map(|mb| mb.motion[2]);
            let upper_right_bottom = upper_right.map(|mb| mb.motion[2]);
            let upper_left_bottom = upper_left.map(|mb| mb.motion[3]);
            let predictor = motion_predictor(
                left_top,
                above_bottom,
                upper_right_bottom,
                upper_left_bottom,
            );
            let (motion, mvd, coded, dc_levels, dc_coded) = if skipped {
                previous_qp_delta_nonzero = false;
                let zero_neighbor = left_top.is_none_or(|vector| vector == [0, 0])
                    || above_bottom.is_none_or(|vector| vector == [0, 0]);
                let vector = if zero_neighbor { [0, 0] } else { predictor };
                (
                    [vector; 4],
                    [[0, 0]; 4],
                    CodedBlockPattern {
                        luma: 0,
                        chroma: 0,
                        pcm: false,
                    },
                    [[0; 4]; 2],
                    [false; 2],
                )
            } else {
                let kind = decoder.p_inter_mb_type(&mut inter)?;
                if kind > 1 {
                    return Err(AvcError::Unsupported("P macroblock partition mode"));
                }
                let top_mvd = [
                    decoder.motion_vector_difference(
                        &mut vectors,
                        0,
                        left.map(|mb| mb.mvd[1][0]),
                        above.map(|mb| mb.mvd[2][0]),
                    )?,
                    decoder.motion_vector_difference(
                        &mut vectors,
                        1,
                        left.map(|mb| mb.mvd[1][1]),
                        above.map(|mb| mb.mvd[2][1]),
                    )?,
                ];
                let top_predictor = if kind == 1 {
                    above_bottom.unwrap_or(predictor)
                } else {
                    predictor
                };
                let top = [
                    top_predictor[0].saturating_add(top_mvd[0]),
                    top_predictor[1].saturating_add(top_mvd[1]),
                ];
                let (motion, mvd) = if kind == 1 {
                    let bottom_mvd = [
                        decoder.motion_vector_difference(
                            &mut vectors,
                            0,
                            left.map(|mb| mb.mvd[3][0]),
                            Some(top_mvd[0]),
                        )?,
                        decoder.motion_vector_difference(
                            &mut vectors,
                            1,
                            left.map(|mb| mb.mvd[3][1]),
                            Some(top_mvd[1]),
                        )?,
                    ];
                    let bottom_predictor = left.map_or(top, |mb| mb.motion[3]);
                    let bottom = [
                        bottom_predictor[0].saturating_add(bottom_mvd[0]),
                        bottom_predictor[1].saturating_add(bottom_mvd[1]),
                    ];
                    (
                        [top, top, bottom, bottom],
                        [top_mvd, top_mvd, bottom_mvd, bottom_mvd],
                    )
                } else {
                    ([top; 4], [top_mvd; 4])
                };
                let coded = decoder.coded_block_pattern(
                    &mut patterns,
                    left.map(|mb| mb.coded),
                    above.map(|mb| mb.coded),
                )?;
                if coded.luma != 0 || coded.chroma > 1 {
                    return Err(AvcError::Unsupported("P macroblock residual"));
                }
                let dc_levels = if coded.chroma == 1 {
                    let delta = decoder.mb_qp_delta(&mut qp_contexts, previous_qp_delta_nonzero)?;
                    previous_qp_delta_nonzero = delta != 0;
                    qp = (qp + delta).rem_euclid(52);
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
                    previous_qp_delta_nonzero = false;
                    [[0; 4]; 2]
                };
                let dc_coded = dc_levels.map(|levels| levels.iter().any(|&level| level != 0));
                (motion, mvd, coded, dc_levels, dc_coded)
            };
            let top = predict_l0_16x16(reference, x, y, motion[0], slice.weights.as_ref(), 0)?;
            let mut block = if motion[2] == motion[0] {
                top
            } else {
                let bottom =
                    predict_l0_16x16(reference, x, y, motion[2], slice.weights.as_ref(), 0)?;
                combine_16x8(top, bottom)
            };
            if coded.chroma == 1 {
                add_chroma_dc(
                    &mut block,
                    &dc_levels[0],
                    chroma_qp(qp, pps.core.chroma_qp_index_offset)?,
                    0,
                )?;
                add_chroma_dc(
                    &mut block,
                    &dc_levels[1],
                    chroma_qp(qp, pps.second_chroma_qp_index_offset)?,
                    1,
                )?;
            }
            write_inter_block(&mut picture, x, y, &block);
            states[index] = Some(InterMbState {
                skipped,
                motion,
                mvd,
                coded,
                chroma_dc_coded: dc_coded,
            });
            let ended = decoder.terminate()?;
            if ended != (index + 1 == states.len()) {
                return Err(AvcError::Unsupported(
                    "P picture has multiple or incomplete slices",
                ));
            }
        }
    }
    Ok(picture)
}

pub struct InterMacroblock {
    pub luma: [u8; 256],
    pub cb: [u8; 64],
    pub cr: [u8; 64],
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
    let b = i32::from(clip((horizontal(plane, width, height, x, y) + 16) >> 5));
    let h = i32::from(clip((vertical(plane, width, height, x, y) + 16) >> 5));
    let j1 = six_tap(std::array::from_fn(|i| {
        horizontal(plane, width, height, x, y + i as i32 - 2)
    }));
    let j = i32::from(clip((j1 + 512) >> 10));
    let m = i32::from(clip((vertical(plane, width, height, x + 1, y) + 16) >> 5));
    let s = i32::from(clip((horizontal(plane, width, height, x, y + 1) + 16) >> 5));
    match (fx, fy) {
        (0, 1) => average(full, h),
        (0, 2) => h as u8,
        (0, 3) => average(sample(plane, width, height, x, y + 1), h),
        (1, 0) => average(full, b),
        (1, 1) => average(b, h),
        (1, 2) => average(h, j),
        (1, 3) => average(h, s),
        (2, 0) => b as u8,
        (2, 1) => average(b, j),
        (2, 2) => j as u8,
        (2, 3) => average(j, s),
        (3, 0) => average(sample(plane, width, height, x + 1, y), b),
        (3, 1) => average(b, m),
        (3, 2) => average(j, m),
        (3, 3) => average(m, s),
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
    if mb_x * 16 >= reference.width || mb_y * 16 >= reference.height {
        return Err(AvcError::InvalidData("inter macroblock out of bounds"));
    }
    let weight = weights
        .map(|table| {
            table
                .list0
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
    for y in 0..16 {
        for x in 0..16 {
            let x4 = ((mb_x * 16 + x) as i32) * 4 + motion[0];
            let y4 = ((mb_y * 16 + y) as i32) * 4 + motion[1];
            let value = luma_quarter(&reference.luma, reference.width, reference.height, x4, y4);
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

#[cfg(test)]
mod tests {
    use super::*;

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
            skipped: false,
            motion: [[9, -3]; 4],
            mvd: [[0, 0]; 4],
            coded: CodedBlockPattern {
                luma: 0,
                chroma: 0,
                pcm: false,
            },
            chroma_dc_coded: [false; 2],
        };
        assert_eq!(
            motion_predictor(None, Some(mb.motion[2]), None, None),
            [9, -3]
        );
        assert_eq!(
            motion_predictor(Some(mb.motion[1]), None, None, None),
            [9, -3]
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
    fn chroma_dc_residual_modifies_only_the_selected_plane() {
        let mut block = InterMacroblock {
            luma: [90; 256],
            cb: [100; 64],
            cr: [110; 64],
        };
        add_chroma_dc(&mut block, &[0; 4], 20, 0).unwrap();
        assert_eq!(block.cb, [100; 64]);
        add_chroma_dc(&mut block, &[8, 0, 0, 0], 20, 1).unwrap();
        assert!(block.cr.iter().any(|&value| value != 110));
        assert_eq!(block.cb, [100; 64]);
        assert_eq!(block.luma, [90; 256]);
    }
}
