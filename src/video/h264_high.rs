//! Progressive 8-bit, 4:2:0 CABAC IDR picture reconstruction (2005 profile).

#[cfg(test)]
use super::VideoFrame;
#[cfg(test)]
use super::h264::frame_from_yuv420;
use super::h264::{
    AvcError, MemoryManagement, PictureParameters2005, SequenceParameters, parse_cabac_i_slice,
    type0_pic_order_count,
};
use super::h264_cabac::{
    CabacDecoder, ChromaAcContexts, ChromaDcContexts, CodedBlockContexts, CodedBlockPattern,
    IntraMbTypeContexts, IntraPredContexts, Luma4x4Contexts, Luma8x8Contexts, Luma16x16AcContexts,
    Luma16x16DcContexts, MbQpContexts, Transform8x8Contexts, intra4x4_modes, intra8x8_modes,
};
use super::h264_deblock::filter_intra_picture;
use super::h264_intra::{
    reconstruct_chroma, reconstruct_intra4x4_luma, reconstruct_intra8x8_luma,
    reconstruct_intra16x16_luma,
};
use super::h264_transform::chroma_qp;

pub struct Yuv420Picture {
    pub width: usize,
    pub height: usize,
    pub frame_num: u32,
    pub pic_order_cnt_lsb: u32,
    pub pic_order_cnt_msb: i32,
    pub pic_order_cnt: i32,
    pub luma: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
    pub motion: Vec<[MotionCell; 4]>,
    // Reference identities at decode time, retained for temporal-direct mapping.
    pub(super) reference_pocs: [Vec<i32>; 2],
    pub(super) luma_half: std::sync::OnceLock<super::h264_inter::HalfPelPlanes>,
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct MotionCell {
    pub l0: Option<(u8, [i32; 2])>,
    pub l1: Option<(u8, [i32; 2])>,
}

#[derive(Clone, Copy)]
struct IntraMb {
    kind: u8,
    transform_8x8: bool,
    modes_4x4: [u8; 16],
    modes_8x8: [u8; 4],
    chroma_mode: u8,
    coded: CodedBlockPattern,
    luma_coded_right: [bool; 4],
    luma_coded_bottom: [bool; 4],
    chroma_dc_coded: [bool; 2],
    chroma_ac_right: [[bool; 2]; 2],
    chroma_ac_bottom: [[bool; 2]; 2],
    luma_dc_coded: bool,
}

fn luma_edges(mb: &IntraMb, side: bool) -> [u8; 4] {
    if mb.kind != 0 {
        [2; 4]
    } else if mb.transform_8x8 {
        let indices = if side { [1, 1, 3, 3] } else { [2, 2, 3, 3] };
        indices.map(|index| mb.modes_8x8[index])
    } else {
        let indices = if side {
            [5, 7, 13, 15]
        } else {
            [10, 11, 14, 15]
        };
        indices.map(|index| mb.modes_4x4[index])
    }
}

fn luma8_edges(mb: &IntraMb, side: bool) -> [u8; 2] {
    if mb.kind != 0 {
        [2; 2]
    } else if mb.transform_8x8 {
        if side {
            [mb.modes_8x8[1], mb.modes_8x8[3]]
        } else {
            [mb.modes_8x8[2], mb.modes_8x8[3]]
        }
    } else if side {
        [mb.modes_4x4[5], mb.modes_4x4[13]]
    } else {
        [mb.modes_4x4[10], mb.modes_4x4[14]]
    }
}

pub(super) fn upper_edge<const N: usize>(
    plane: &[u8],
    stride: usize,
    x: usize,
    y: usize,
) -> Option<[u8; N]> {
    (y > 0).then(|| std::array::from_fn(|i| plane[(y - 1) * stride + (x + i).min(stride - 1)]))
}

pub(super) fn left_edge<const N: usize>(
    plane: &[u8],
    stride: usize,
    x: usize,
    y: usize,
) -> Option<[u8; N]> {
    (x > 0).then(|| std::array::from_fn(|i| plane[(y + i) * stride + x - 1]))
}

fn write_block(plane: &mut [u8], stride: usize, x: usize, y: usize, block: &[u8], size: usize) {
    for row in 0..size {
        let start = (y + row) * stride + x;
        plane[start..start + size].copy_from_slice(&block[row * size..(row + 1) * size]);
    }
}

#[cfg(test)]
pub fn decode_cabac_idr_2005(
    nal: &[u8],
    sps: &SequenceParameters,
    pps: &PictureParameters2005,
) -> Result<VideoFrame, AvcError> {
    let picture = decode_cabac_idr_yuv_2005(nal, sps, pps)?;
    Ok(frame_from_yuv420(
        sps,
        &picture.luma,
        &picture.cb,
        &picture.cr,
    ))
}

pub fn decode_cabac_idr_yuv_2005(
    nal: &[u8],
    sps: &SequenceParameters,
    pps: &PictureParameters2005,
) -> Result<Yuv420Picture, AvcError> {
    if nal.first().is_none_or(|header| header & 0x1f != 5) {
        return Err(AvcError::InvalidData("expected IDR I slice"));
    }
    decode_cabac_i_yuv_2005(nal, sps, pps, None).map(|(picture, _)| picture)
}

pub fn decode_cabac_i_yuv_2005(
    nal: &[u8],
    sps: &SequenceParameters,
    pps: &PictureParameters2005,
    previous_reference: Option<&Yuv420Picture>,
) -> Result<(Yuv420Picture, Vec<MemoryManagement>), AvcError> {
    #[cfg(test)]
    let _profile_picture = super::h264_inter::profiling::scope(super::h264_inter::profiling::Stage::Other);
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
    {
        return Err(AvcError::Unsupported("High-profile I picture format"));
    }
    let slice = parse_cabac_i_slice(nal, sps, &pps.core)?;
    if slice.first_mb != 0 {
        return Err(AvcError::Unsupported("I picture starts in a later slice"));
    }
    let coded_height = sps.frame_height_mbs * 16;
    let pixels = u64::from(sps.width) * u64::from(coded_height);
    if pixels > 8 * 1024 * 1024 {
        return Err(AvcError::TooLarge);
    }
    let width = sps.width as usize;
    let height = coded_height as usize;
    let mut luma = vec![0; width * height];
    let mut cb = vec![0; width * height / 4];
    let mut cr = vec![0; width * height / 4];
    let mut states: Vec<IntraMb> =
        Vec::with_capacity((sps.width_mbs * sps.frame_height_mbs) as usize);
    let mut macroblock_qps = Vec::with_capacity(states.capacity());
    let mut macroblock_transform8x8 = Vec::with_capacity(states.capacity());
    let mut decoder = CabacDecoder::new(&slice.rbsp[slice.data_byte_offset..])?;
    let mut type_contexts = IntraMbTypeContexts::new(slice.slice_qp)?;
    let mut transform_contexts = Transform8x8Contexts::new(slice.slice_qp)?;
    let mut pred_contexts = IntraPredContexts::new(slice.slice_qp)?;
    let mut coded_contexts = CodedBlockContexts::new(slice.slice_qp)?;
    let mut qp_contexts = MbQpContexts::new(slice.slice_qp)?;
    let mut luma4_contexts = Luma4x4Contexts::new(slice.slice_qp)?;
    let mut luma8_contexts = Luma8x8Contexts::new(slice.slice_qp)?;
    let mut luma16_dc_contexts = Luma16x16DcContexts::new(slice.slice_qp)?;
    let mut luma16_ac_contexts = Luma16x16AcContexts::new(slice.slice_qp)?;
    let mut chroma_contexts = ChromaDcContexts::new(slice.slice_qp)?;
    let mut chroma_ac_contexts = ChromaAcContexts::new(slice.slice_qp)?;
    let mut qp = slice.slice_qp;
    let mut previous_qp_delta_nonzero = false;
    let mb_width = sps.width_mbs as usize;
    let mb_count = mb_width * sps.frame_height_mbs as usize;

    for mb in 0..mb_count {
        #[cfg(test)]
        let _profile_mb = super::h264_inter::profiling::scope(super::h264_inter::profiling::Stage::EntropyState);
        let mb_x = mb % mb_width;
        let mb_y = mb / mb_width;
        let left = (mb_x > 0).then(|| &states[mb - 1]);
        let above = (mb_y > 0).then(|| &states[mb - mb_width]);
        let kind = decoder.intra_mb_type(
            &mut type_contexts,
            left.map(|mb| mb.kind),
            above.map(|mb| mb.kind),
        )?;
        if kind == 25 {
            return Err(AvcError::Unsupported("High-profile PCM macroblock"));
        }
        let transform_8x8 = if kind == 0 && pps.transform_8x8 {
            decoder.transform_size_8x8_flag(
                &mut transform_contexts,
                left.map(|mb| mb.transform_8x8),
                above.map(|mb| mb.transform_8x8),
            )?
        } else {
            false
        };
        let mut modes_4x4 = [2; 16];
        let mut modes_8x8 = [2; 4];
        if kind != 0 {
            // Intra16x16 has no luma prediction-mode syntax elements.
        } else if transform_8x8 {
            let mut codes = [super::h264_cabac::IntraPredCode {
                use_predicted_mode: true,
                remaining_mode: None,
            }; 4];
            for code in &mut codes {
                *code = decoder.intra_luma_pred_code(&mut pred_contexts)?;
            }
            modes_8x8 = intra8x8_modes(
                &codes,
                left.map(|mb| luma8_edges(mb, true)),
                above.map(|mb| luma8_edges(mb, false)),
            )?;
        } else {
            let mut codes = [super::h264_cabac::IntraPredCode {
                use_predicted_mode: true,
                remaining_mode: None,
            }; 16];
            for code in &mut codes {
                *code = decoder.intra_luma_pred_code(&mut pred_contexts)?;
            }
            modes_4x4 = intra4x4_modes(
                &codes,
                left.map(|mb| luma_edges(mb, true)),
                above.map(|mb| luma_edges(mb, false)),
            )?;
        }
        let chroma_mode = decoder.intra_chroma_pred_mode(
            &mut pred_contexts,
            left.map(|mb| mb.chroma_mode),
            above.map(|mb| mb.chroma_mode),
        )?;
        let coded = if kind == 0 {
            decoder.coded_block_pattern(
                &mut coded_contexts,
                left.map(|mb| mb.coded),
                above.map(|mb| mb.coded),
            )?
        } else {
            CodedBlockPattern {
                luma: if kind >= 13 { 15 } else { 0 },
                chroma: ((kind - 1) / 4) % 3,
                pcm: false,
            }
        };
        let qp_delta = if kind != 0 || coded.luma != 0 || coded.chroma != 0 {
            decoder.mb_qp_delta(&mut qp_contexts, previous_qp_delta_nonzero)?
        } else {
            0
        };
        previous_qp_delta_nonzero = qp_delta != 0;
        qp = (qp + qp_delta).rem_euclid(52);
        let x0 = mb_x * 16;
        let y0 = mb_y * 16;
        let corner = (x0 > 0 && y0 > 0).then(|| luma[(y0 - 1) * width + x0 - 1]);
        let mut luma_dc_coded = false;
        let (luma_block, luma_coded_right, luma_coded_bottom) = if kind != 0 {
            let dc = decoder.luma16x16_dc_coefficients(
                &mut luma16_dc_contexts,
                left.map(|mb| mb.luma_dc_coded),
                above.map(|mb| mb.luma_dc_coded),
            )?;
            luma_dc_coded = dc.iter().any(|&level| level != 0);
            let ac = if coded.luma != 0 {
                decoder.luma16x16_ac_macroblock(
                    &mut luma16_ac_contexts,
                    left.map(|mb| mb.luma_coded_right),
                    above.map(|mb| mb.luma_coded_bottom),
                )?
            } else {
                [[0; 15]; 16]
            };
            #[cfg(test)]
            let _profile_reconstruct = super::h264_inter::profiling::scope(super::h264_inter::profiling::Stage::Reconstruction);
            let pixels = reconstruct_intra16x16_luma(
                (kind - 1) % 4,
                &dc,
                &ac,
                qp,
                upper_edge(&luma, width, x0, y0),
                left_edge(&luma, width, x0, y0),
                corner,
            )?;
            let coded_block = |index: usize| ac[index].iter().any(|&level| level != 0);
            (
                pixels,
                [5, 7, 13, 15].map(coded_block),
                [10, 11, 14, 15].map(coded_block),
            )
        } else if transform_8x8 {
            let mut blocks = [[0; 64]; 4];
            for (region, block) in blocks.iter_mut().enumerate() {
                if coded.luma & (1 << region) != 0 {
                    *block = decoder.luma8x8_coefficients(&mut luma8_contexts)?;
                }
            }
            #[cfg(test)]
            let _profile_reconstruct = super::h264_inter::profiling::scope(super::h264_inter::profiling::Stage::Reconstruction);
            let pixels = reconstruct_intra8x8_luma(
                &modes_8x8,
                &blocks,
                qp,
                &[16; 64],
                upper_edge(&luma, width, x0, y0),
                left_edge(&luma, width, x0, y0),
                corner,
            )?;
            let edge_coded = |region: usize| blocks[region].iter().any(|&level| level != 0);
            (
                pixels,
                [edge_coded(1), edge_coded(1), edge_coded(3), edge_coded(3)],
                [edge_coded(2), edge_coded(2), edge_coded(3), edge_coded(3)],
            )
        } else {
            let blocks = decoder.luma4x4_macroblock(
                &mut luma4_contexts,
                coded.luma,
                left.map(|mb| mb.luma_coded_right),
                above.map(|mb| mb.luma_coded_bottom),
            )?;
            #[cfg(test)]
            let _profile_reconstruct = super::h264_inter::profiling::scope(super::h264_inter::profiling::Stage::Reconstruction);
            let pixels = reconstruct_intra4x4_luma(
                &modes_4x4,
                &blocks,
                qp,
                upper_edge(&luma, width, x0, y0),
                left_edge(&luma, width, x0, y0),
                corner,
            )?;
            let coded_block = |index: usize| blocks[index].iter().any(|&level| level != 0);
            (
                pixels,
                [5, 7, 13, 15].map(coded_block),
                [10, 11, 14, 15].map(coded_block),
            )
        };
        let cb_dc = if coded.chroma != 0 {
            decoder.chroma_dc_coefficients(
                &mut chroma_contexts,
                left.map(|mb| mb.chroma_dc_coded[0]),
                above.map(|mb| mb.chroma_dc_coded[0]),
            )?
        } else {
            [0; 4]
        };
        let cr_dc = if coded.chroma != 0 {
            decoder.chroma_dc_coefficients(
                &mut chroma_contexts,
                left.map(|mb| mb.chroma_dc_coded[1]),
                above.map(|mb| mb.chroma_dc_coded[1]),
            )?
        } else {
            [0; 4]
        };
        let mut chroma_ac = [[[0; 15]; 4]; 2];
        if coded.chroma == 2 {
            for plane in 0..2 {
                chroma_ac[plane] = decoder.chroma_ac_macroblock(
                    &mut chroma_ac_contexts,
                    left.map(|mb| mb.chroma_ac_right[plane]),
                    above.map(|mb| mb.chroma_ac_bottom[plane]),
                )?;
            }
        }
        let cx0 = mb_x * 8;
        let cy0 = mb_y * 8;
        let chroma_width = width / 2;
        let cb_qp = chroma_qp(qp, pps.core.chroma_qp_index_offset)?;
        let cr_qp = chroma_qp(qp, pps.second_chroma_qp_index_offset)?;
        let cb_corner = (cx0 > 0 && cy0 > 0).then(|| cb[(cy0 - 1) * chroma_width + cx0 - 1]);
        let cr_corner = (cx0 > 0 && cy0 > 0).then(|| cr[(cy0 - 1) * chroma_width + cx0 - 1]);
        #[cfg(test)]
        let profile_reconstruct = super::h264_inter::profiling::scope(super::h264_inter::profiling::Stage::Reconstruction);
        let cb_block = reconstruct_chroma(
            chroma_mode,
            &cb_dc,
            &chroma_ac[0],
            cb_qp,
            upper_edge(&cb, chroma_width, cx0, cy0),
            left_edge(&cb, chroma_width, cx0, cy0),
            cb_corner,
        )?;
        let cr_block = reconstruct_chroma(
            chroma_mode,
            &cr_dc,
            &chroma_ac[1],
            cr_qp,
            upper_edge(&cr, chroma_width, cx0, cy0),
            left_edge(&cr, chroma_width, cx0, cy0),
            cr_corner,
        )?;
        write_block(&mut luma, width, x0, y0, &luma_block, 16);
        write_block(&mut cb, chroma_width, cx0, cy0, &cb_block, 8);
        write_block(&mut cr, chroma_width, cx0, cy0, &cr_block, 8);
        #[cfg(test)]
        drop(profile_reconstruct);
        states.push(IntraMb {
            kind,
            transform_8x8,
            modes_4x4,
            modes_8x8,
            chroma_mode,
            coded,
            luma_coded_right,
            luma_coded_bottom,
            chroma_dc_coded: [
                cb_dc.iter().any(|&level| level != 0),
                cr_dc.iter().any(|&level| level != 0),
            ],
            chroma_ac_right: std::array::from_fn(|plane| {
                [1, 3].map(|block| chroma_ac[plane][block].iter().any(|&level| level != 0))
            }),
            chroma_ac_bottom: std::array::from_fn(|plane| {
                [2, 3].map(|block| chroma_ac[plane][block].iter().any(|&level| level != 0))
            }),
            luma_dc_coded,
        });
        macroblock_qps.push(qp);
        macroblock_transform8x8.push(transform_8x8);
        let end = decoder.terminate()?;
        if end != (mb + 1 == mb_count) {
            return Err(AvcError::Unsupported("incomplete single-slice I picture"));
        }
    }
    if !slice.deblocking_disabled {
        #[cfg(test)]
        let _profile_deblock = super::h264_inter::profiling::scope(super::h264_inter::profiling::Stage::Deblock);
        filter_intra_picture(
            &mut luma,
            &mut cb,
            &mut cr,
            width,
            &macroblock_qps,
            &macroblock_transform8x8,
            [
                pps.core.chroma_qp_index_offset,
                pps.second_chroma_qp_index_offset,
            ],
            slice.alpha_offset,
            slice.beta_offset,
        )?;
    }
    let previous_poc = if nal[0] & 0x1f == 5 {
        None
    } else {
        previous_reference.map(|picture| (picture.pic_order_cnt_msb, picture.pic_order_cnt_lsb))
    };
    let (pic_order_cnt_msb, pic_order_cnt) = type0_pic_order_count(
        slice.pic_order_cnt_lsb,
        sps.pic_order_cnt_lsb_bits
            .ok_or(AvcError::Unsupported("POC type"))?,
        previous_poc,
    )?;
    Ok((
        Yuv420Picture {
            width,
            height,
            frame_num: slice.frame_num,
            pic_order_cnt_lsb: slice.pic_order_cnt_lsb,
            pic_order_cnt_msb,
            pic_order_cnt,
            luma,
            cb,
            cr,
            motion: vec![[MotionCell::default(); 4]; mb_count],
            reference_pocs: Default::default(),
            luma_half: std::sync::OnceLock::new(),
        },
        slice.marking,
    ))
}
