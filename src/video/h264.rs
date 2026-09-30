//! Bounded AVC (H.264) bitstream input for the 2003/2005 profiles.
//!
//! This parses MP4 `avcC` configuration and length-prefixed NAL units. Picture
//! reconstruction covers bounded intra and inter subsets of the 2003/2005
//! profiles; callers must not advertise general H.264 playback yet.

const MAX_NAL_BYTES: usize = 32 * 1024 * 1024;
const MAX_DIMENSION: u32 = 8192;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AvcError {
    InvalidData(&'static str),
    Unsupported(&'static str),
    UnsupportedProfile(u8),
    TooLarge,
    Incomplete,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceParameters {
    pub id: u32,
    pub profile_idc: u8,
    pub level_idc: u8,
    pub width: u32,
    pub height: u32,
    pub chroma_format_idc: u32,
    pub bit_depth_luma: u32,
    pub bit_depth_chroma: u32,
    pub scaling_matrices_present: bool,
    pub width_mbs: u32,
    pub frame_height_mbs: u32,
    pub frame_mbs_only: bool,
    pub frame_num_bits: usize,
    pub max_num_ref_frames: u32,
    pub pic_order_cnt_type: u32,
    pub pic_order_cnt_lsb_bits: Option<usize>,
}

/// Fields from the original 2003 picture parameter set (7.3.2.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PictureParameters2003 {
    pub id: u32,
    pub sequence_id: u32,
    pub cabac: bool,
    pub pic_order_present: bool,
    pub slice_groups: u32,
    pub slice_group_map_type: Option<u32>,
    pub ref_idx_l0: u32,
    pub ref_idx_l1: u32,
    pub weighted_pred: bool,
    pub weighted_bipred_idc: u32,
    pub pic_init_qp: i32,
    pub chroma_qp_index_offset: i32,
    pub deblocking_filter_control_present: bool,
    pub constrained_intra_pred: bool,
    pub redundant_pic_cnt_present: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PictureParameters2005 {
    pub core: PictureParameters2003,
    pub transform_8x8: bool,
    pub scaling_matrices_present: bool,
    pub second_chroma_qp_index_offset: i32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CabacIdrISlice {
    pub first_mb: u32,
    pub frame_num: u32,
    pub pic_order_cnt_lsb: u32,
    pub slice_qp: i32,
    pub rbsp: Vec<u8>,
    pub data_byte_offset: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CabacInterSlice {
    pub first_mb: u32,
    pub slice_type: u32,
    pub frame_num: u32,
    pub pic_order_cnt_lsb: u32,
    pub ref_idx_l0: u32,
    pub ref_idx_l1: u32,
    pub reorder_l0: Vec<RefPicReorder>,
    pub reorder_l1: Vec<RefPicReorder>,
    pub direct_spatial_mv_pred: bool,
    pub cabac_init_idc: u32,
    pub slice_qp: i32,
    pub weights: Option<PredictionWeightTable>,
    pub rbsp: Vec<u8>,
    pub data_byte_offset: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RefPicReorder {
    Subtract(u32),
    Add(u32),
    LongTerm(u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredictionWeight {
    pub luma_weight: i32,
    pub luma_offset: i32,
    pub chroma_weight: [i32; 2],
    pub chroma_offset: [i32; 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredictionWeightTable {
    pub luma_denom: u32,
    pub chroma_denom: u32,
    pub list0: Vec<PredictionWeight>,
    pub list1: Vec<PredictionWeight>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AvcConfig {
    pub nal_length_size: usize,
    pub sequence_parameters: Vec<SequenceParameters>,
    pub picture_parameter_sets: Vec<Vec<u8>>,
}

impl AvcConfig {
    pub fn parse(data: &[u8]) -> Result<Self, AvcError> {
        if data.len() < 7 || data[0] != 1 || data[4] & 0xfc != 0xfc || data[5] & 0xe0 != 0xe0 {
            return Err(AvcError::InvalidData("invalid avcC header"));
        }
        let nal_length_size = usize::from((data[4] & 3) + 1);
        if nal_length_size == 3 {
            return Err(AvcError::InvalidData("invalid NAL length size"));
        }
        let mut offset = 6;
        let mut sequence_parameters = Vec::new();
        for _ in 0..(data[5] & 31) {
            let nal = take_nal(data, &mut offset)?;
            let parameters = parse_sps(nal)?;
            if parameters.profile_idc != data[1] || parameters.level_idc != data[3] {
                return Err(AvcError::InvalidData("avcC and SPS disagree"));
            }
            sequence_parameters.push(parameters);
        }
        let count = *data.get(offset).ok_or(AvcError::Incomplete)?;
        offset += 1;
        let mut picture_parameter_sets = Vec::new();
        for _ in 0..count {
            let nal = take_nal(data, &mut offset)?;
            if nal
                .first()
                .is_none_or(|byte| byte & 0x1f != 8 || byte & 0x80 != 0)
            {
                return Err(AvcError::InvalidData("invalid PPS NAL"));
            }
            picture_parameter_sets.push(nal.to_vec());
        }
        if sequence_parameters.is_empty() || picture_parameter_sets.is_empty() {
            return Err(AvcError::InvalidData("missing parameter sets"));
        }
        Ok(Self {
            nal_length_size,
            sequence_parameters,
            picture_parameter_sets,
        })
    }
}

fn take_nal<'a>(data: &'a [u8], offset: &mut usize) -> Result<&'a [u8], AvcError> {
    let size = data.get(*offset..*offset + 2).ok_or(AvcError::Incomplete)?;
    let size = u16::from_be_bytes([size[0], size[1]]) as usize;
    *offset += 2;
    if size == 0 {
        return Err(AvcError::InvalidData("empty parameter set"));
    }
    let nal = data
        .get(*offset..*offset + size)
        .ok_or(AvcError::Incomplete)?;
    *offset += size;
    Ok(nal)
}

/// MP4 stores AVC NAL units with a length prefix instead of Annex B start codes.
pub struct NalStream {
    length_size: usize,
    pending: Vec<u8>,
}

impl NalStream {
    pub fn new(length_size: usize) -> Result<Self, AvcError> {
        if !matches!(length_size, 1 | 2 | 4) {
            return Err(AvcError::InvalidData("invalid NAL length size"));
        }
        Ok(Self {
            length_size,
            pending: Vec::new(),
        })
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<Vec<u8>>, AvcError> {
        let mut units = Vec::new();
        for chunk in bytes.chunks(16 * 1024) {
            self.pending.extend_from_slice(chunk);
            let mut consumed = 0;
            while self.pending.len() - consumed >= self.length_size {
                let mut size = 0usize;
                for byte in &self.pending[consumed..consumed + self.length_size] {
                    size = (size << 8) | usize::from(*byte);
                }
                if size == 0 || size > MAX_NAL_BYTES {
                    return Err(AvcError::TooLarge);
                }
                let end = consumed + self.length_size + size;
                if end > self.pending.len() {
                    break;
                }
                let nal = &self.pending[consumed + self.length_size..end];
                if nal[0] & 0x80 != 0 || nal[0] & 0x1f == 0 {
                    return Err(AvcError::InvalidData("invalid NAL header"));
                }
                units.push(nal.to_vec());
                consumed = end;
            }
            self.pending.drain(..consumed);
            if self.pending.len() > MAX_NAL_BYTES + self.length_size {
                return Err(AvcError::TooLarge);
            }
        }
        Ok(units)
    }

    pub fn finish(&self) -> Result<(), AvcError> {
        if self.pending.is_empty() {
            Ok(())
        } else {
            Err(AvcError::Incomplete)
        }
    }
}

struct Bits<'a> {
    bytes: &'a [u8],
    bit: usize,
}

impl Bits<'_> {
    fn read(&mut self, count: usize) -> Result<u32, AvcError> {
        if count > 32
            || self
                .bit
                .checked_add(count)
                .is_none_or(|end| end > self.bytes.len() * 8)
        {
            return Err(AvcError::Incomplete);
        }
        let mut value = 0;
        for _ in 0..count {
            value = (value << 1) | u32::from((self.bytes[self.bit / 8] >> (7 - self.bit % 8)) & 1);
            self.bit += 1;
        }
        Ok(value)
    }

    fn ue(&mut self) -> Result<u32, AvcError> {
        let mut zeros = 0;
        while self.read(1)? == 0 {
            zeros += 1;
            if zeros > 30 {
                return Err(AvcError::InvalidData("Exp-Golomb value too large"));
            }
        }
        Ok(((1u32 << zeros) - 1) + self.read(zeros)?)
    }

    fn se(&mut self) -> Result<i32, AvcError> {
        let code = self.ue()?;
        Ok(if code & 1 == 0 {
            -(code as i32 / 2)
        } else {
            (code as i32 + 1) / 2
        })
    }

    fn finish_rbsp(&mut self) -> Result<(), AvcError> {
        if self.read(1)? != 1 {
            return Err(AvcError::InvalidData("missing RBSP stop bit"));
        }
        while self.bit % 8 != 0 {
            if self.read(1)? != 0 {
                return Err(AvcError::InvalidData("invalid RBSP padding"));
            }
        }
        if self.bit != self.bytes.len() * 8 {
            return Err(AvcError::InvalidData("unexpected RBSP data"));
        }
        Ok(())
    }

    fn more_rbsp_data(&self) -> Result<bool, AvcError> {
        let remaining = self.bytes.len() * 8 - self.bit;
        if remaining == 0 {
            return Err(AvcError::Incomplete);
        }
        if remaining > 8 {
            return Ok(true);
        }
        let mut probe = Bits {
            bytes: self.bytes,
            bit: self.bit,
        };
        if probe.read(1)? != 1 {
            return Ok(true);
        }
        for _ in 1..remaining {
            if probe.read(1)? != 0 {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

fn rbsp_from_nal(nal: &[u8], expected_type: u8) -> Result<Vec<u8>, AvcError> {
    if nal
        .first()
        .is_none_or(|byte| byte & 0x80 != 0 || byte & 0x1f != expected_type)
    {
        return Err(AvcError::InvalidData("invalid parameter-set NAL"));
    }
    let mut rbsp = Vec::with_capacity(nal.len().saturating_sub(1));
    for &byte in &nal[1..] {
        if rbsp.len() >= 2 && rbsp[rbsp.len() - 2..] == [0, 0] && byte == 3 {
            continue;
        }
        rbsp.push(byte);
    }
    Ok(rbsp)
}

pub fn parse_pps_2003(nal: &[u8]) -> Result<PictureParameters2003, AvcError> {
    let rbsp = rbsp_from_nal(nal, 8)?;
    let mut bits = Bits {
        bytes: &rbsp,
        bit: 0,
    };
    let pps = parse_pps_fields(&mut bits)?;
    bits.finish_rbsp()?;
    Ok(pps)
}

pub fn parse_pps_2005(nal: &[u8]) -> Result<PictureParameters2005, AvcError> {
    let rbsp = rbsp_from_nal(nal, 8)?;
    let mut bits = Bits {
        bytes: &rbsp,
        bit: 0,
    };
    let core = parse_pps_fields(&mut bits)?;
    let mut transform_8x8 = false;
    let mut scaling_matrices_present = false;
    let mut second_chroma_qp_index_offset = core.chroma_qp_index_offset;
    if bits.more_rbsp_data()? {
        transform_8x8 = bits.read(1)? != 0;
        scaling_matrices_present = bits.read(1)? != 0;
        if scaling_matrices_present {
            for index in 0..(6 + if transform_8x8 { 2 } else { 0 }) {
                if bits.read(1)? != 0 {
                    skip_scaling_list(&mut bits, if index < 6 { 16 } else { 64 })?;
                }
            }
        }
        second_chroma_qp_index_offset = bits.se()?;
        if !(-12..=12).contains(&second_chroma_qp_index_offset) {
            return Err(AvcError::InvalidData("chroma QP offset out of range"));
        }
    }
    bits.finish_rbsp()?;
    Ok(PictureParameters2005 {
        core,
        transform_8x8,
        scaling_matrices_present,
        second_chroma_qp_index_offset,
    })
}

fn parse_pps_fields(bits: &mut Bits<'_>) -> Result<PictureParameters2003, AvcError> {
    let id = bits.ue()?;
    let sequence_id = bits.ue()?;
    if id > 255 || sequence_id > 31 {
        return Err(AvcError::InvalidData("parameter-set ID out of range"));
    }
    let cabac = bits.read(1)? != 0;
    let pic_order_present = bits.read(1)? != 0;
    let slice_groups = bits.ue()?.checked_add(1).ok_or(AvcError::TooLarge)?;
    if slice_groups > 8 {
        return Err(AvcError::TooLarge);
    }
    let slice_group_map_type = if slice_groups > 1 {
        let map_type = bits.ue()?;
        match map_type {
            0 => {
                for _ in 0..slice_groups {
                    bits.ue()?;
                }
            }
            1 => {}
            2 => {
                for _ in 1..slice_groups {
                    bits.ue()?;
                    bits.ue()?;
                }
            }
            3..=5 => {
                bits.read(1)?;
                bits.ue()?;
            }
            6 => {
                let count = bits.ue()?.checked_add(1).ok_or(AvcError::TooLarge)?;
                if count > 65_536 {
                    return Err(AvcError::TooLarge);
                }
                let width = (32 - (slice_groups - 1).leading_zeros()) as usize;
                for _ in 0..count {
                    if bits.read(width)? >= slice_groups {
                        return Err(AvcError::InvalidData("slice group ID out of range"));
                    }
                }
            }
            _ => return Err(AvcError::InvalidData("invalid slice group map type")),
        }
        Some(map_type)
    } else {
        None
    };
    let ref_idx_l0 = bits.ue()?.checked_add(1).ok_or(AvcError::TooLarge)?;
    let ref_idx_l1 = bits.ue()?.checked_add(1).ok_or(AvcError::TooLarge)?;
    if ref_idx_l0 > 32 || ref_idx_l1 > 32 {
        return Err(AvcError::TooLarge);
    }
    let weighted_pred = bits.read(1)? != 0;
    let weighted_bipred_idc = bits.read(2)?;
    if weighted_bipred_idc == 3 {
        return Err(AvcError::InvalidData("invalid weighted bipred mode"));
    }
    let pic_init_qp = 26 + bits.se()?;
    bits.se()?; // pic_init_qs_minus26
    let chroma_qp_index_offset = bits.se()?;
    if !(-12..=12).contains(&chroma_qp_index_offset) {
        return Err(AvcError::InvalidData("chroma QP offset out of range"));
    }
    let deblocking_filter_control_present = bits.read(1)? != 0;
    let constrained_intra_pred = bits.read(1)? != 0;
    let redundant_pic_cnt_present = bits.read(1)? != 0;
    Ok(PictureParameters2003 {
        id,
        sequence_id,
        cabac,
        pic_order_present,
        slice_groups,
        slice_group_map_type,
        ref_idx_l0,
        ref_idx_l1,
        weighted_pred,
        weighted_bipred_idc,
        pic_init_qp,
        chroma_qp_index_offset,
        deblocking_filter_control_present,
        constrained_intra_pred,
        redundant_pic_cnt_present,
    })
}

/// Locate CABAC bytes after the original 2003 IDR I-slice header syntax.
pub fn parse_cabac_idr_i_slice(
    nal: &[u8],
    sps: &SequenceParameters,
    pps: &PictureParameters2003,
) -> Result<CabacIdrISlice, AvcError> {
    if !sps.frame_mbs_only || sps.pic_order_cnt_type != 0 || pps.slice_groups != 1 || !pps.cabac {
        return Err(AvcError::Unsupported("CABAC IDR header shape"));
    }
    if pps.sequence_id != sps.id {
        return Err(AvcError::InvalidData("PPS refers to another SPS"));
    }
    if nal.first().is_none_or(|header| header & 0x60 == 0) {
        return Err(AvcError::InvalidData("IDR is not a reference picture"));
    }
    let rbsp = rbsp_from_nal(nal, 5)?;
    let mut bits = Bits {
        bytes: &rbsp,
        bit: 0,
    };
    let first_mb = bits.ue()?;
    if first_mb >= sps.width_mbs * sps.frame_height_mbs {
        return Err(AvcError::InvalidData("first macroblock out of bounds"));
    }
    if !matches!(bits.ue()?, 2 | 7) || bits.ue()? != pps.id {
        return Err(AvcError::Unsupported("expected IDR I slice for PPS"));
    }
    let frame_num = bits.read(sps.frame_num_bits)?;
    bits.ue()?; // idr_pic_id
    let pic_order_cnt_lsb = bits.read(
        sps.pic_order_cnt_lsb_bits
            .ok_or(AvcError::Unsupported("POC type"))?,
    )?;
    if pps.pic_order_present {
        bits.se()?;
    }
    if pps.redundant_pic_cnt_present {
        bits.ue()?;
    }
    bits.read(2)?; // IDR reference picture marking flags
    let slice_qp = pps.pic_init_qp + bits.se()?;
    if !(0..=51).contains(&slice_qp) {
        return Err(AvcError::InvalidData("slice QP out of range"));
    }
    if pps.deblocking_filter_control_present {
        let disable_idc = bits.ue()?;
        if disable_idc > 2 {
            return Err(AvcError::InvalidData("invalid deblocking mode"));
        }
        if disable_idc != 1 {
            bits.se()?;
            bits.se()?;
        }
    }
    while bits.bit % 8 != 0 {
        if bits.read(1)? != 1 {
            return Err(AvcError::InvalidData("invalid CABAC alignment"));
        }
    }
    let data_byte_offset = bits.bit / 8;
    if data_byte_offset >= rbsp.len() {
        return Err(AvcError::Incomplete);
    }
    Ok(CabacIdrISlice {
        first_mb,
        frame_num,
        pic_order_cnt_lsb,
        slice_qp,
        rbsp,
        data_byte_offset,
    })
}

fn parse_ref_pic_list_reordering(bits: &mut Bits<'_>) -> Result<Vec<RefPicReorder>, AvcError> {
    if bits.read(1)? == 0 {
        return Ok(Vec::new());
    }
    let mut changes = Vec::new();
    for _ in 0..32 {
        match bits.ue()? {
            0 => changes.push(RefPicReorder::Subtract(bits.ue()?)),
            1 => changes.push(RefPicReorder::Add(bits.ue()?)),
            2 => changes.push(RefPicReorder::LongTerm(bits.ue()?)),
            3 => return Ok(changes),
            _ => {
                return Err(AvcError::InvalidData(
                    "invalid reference picture reordering",
                ));
            }
        }
    }
    Err(AvcError::TooLarge)
}

fn parse_prediction_weight_table(
    bits: &mut Bits<'_>,
    is_b: bool,
    ref_idx_l0: u32,
    ref_idx_l1: u32,
) -> Result<PredictionWeightTable, AvcError> {
    let luma_denom = bits.ue()?;
    let chroma_denom = bits.ue()?;
    if luma_denom > 7 || chroma_denom > 7 {
        return Err(AvcError::InvalidData(
            "prediction weight denominator out of range",
        ));
    }
    let read_list = |bits: &mut Bits<'_>, count: u32| -> Result<Vec<PredictionWeight>, AvcError> {
        let mut list = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let mut weight = PredictionWeight {
                luma_weight: 1 << luma_denom,
                luma_offset: 0,
                chroma_weight: [1 << chroma_denom; 2],
                chroma_offset: [0; 2],
            };
            if bits.read(1)? != 0 {
                weight.luma_weight = bits.se()?;
                weight.luma_offset = bits.se()?;
            }
            if bits.read(1)? != 0 {
                for channel in 0..2 {
                    weight.chroma_weight[channel] = bits.se()?;
                    weight.chroma_offset[channel] = bits.se()?;
                }
            }
            list.push(weight);
        }
        Ok(list)
    };
    let list0 = read_list(bits, ref_idx_l0)?;
    let list1 = if is_b {
        read_list(bits, ref_idx_l1)?
    } else {
        Vec::new()
    };
    Ok(PredictionWeightTable {
        luma_denom,
        chroma_denom,
        list0,
        list1,
    })
}

/// Parse the original 2003 P/B slice header through CABAC alignment.
pub fn parse_cabac_inter_slice(
    nal: &[u8],
    sps: &SequenceParameters,
    pps: &PictureParameters2003,
) -> Result<CabacInterSlice, AvcError> {
    if !sps.frame_mbs_only || sps.pic_order_cnt_type != 0 || pps.slice_groups != 1 || !pps.cabac {
        return Err(AvcError::Unsupported("CABAC inter header shape"));
    }
    if pps.sequence_id != sps.id {
        return Err(AvcError::InvalidData(
            "inter picture PPS refers to another SPS",
        ));
    }
    let rbsp = rbsp_from_nal(nal, 1)?;
    let mut bits = Bits {
        bytes: &rbsp,
        bit: 0,
    };
    let first_mb = bits.ue()?;
    if first_mb >= sps.width_mbs * sps.frame_height_mbs {
        return Err(AvcError::InvalidData("first macroblock out of bounds"));
    }
    let slice_type = bits.ue()?;
    if !matches!(slice_type, 0 | 1 | 5 | 6) || bits.ue()? != pps.id {
        return Err(AvcError::Unsupported("expected P or B slice for PPS"));
    }
    let is_b = slice_type % 5 == 1;
    let frame_num = bits.read(sps.frame_num_bits)?;
    let pic_order_cnt_lsb = bits.read(
        sps.pic_order_cnt_lsb_bits
            .ok_or(AvcError::Unsupported("POC type"))?,
    )?;
    if pps.pic_order_present {
        bits.se()?;
    }
    if pps.redundant_pic_cnt_present {
        bits.ue()?;
    }
    let direct_spatial_mv_pred = is_b && bits.read(1)? != 0;
    let mut ref_idx_l0 = pps.ref_idx_l0;
    let mut ref_idx_l1 = pps.ref_idx_l1;
    if bits.read(1)? != 0 {
        ref_idx_l0 = bits.ue()?.checked_add(1).ok_or(AvcError::TooLarge)?;
        if is_b {
            ref_idx_l1 = bits.ue()?.checked_add(1).ok_or(AvcError::TooLarge)?;
        }
    }
    if ref_idx_l0 > 32 || ref_idx_l1 > 32 {
        return Err(AvcError::TooLarge);
    }
    let reorder_l0 = parse_ref_pic_list_reordering(&mut bits)?;
    let reorder_l1 = if is_b {
        parse_ref_pic_list_reordering(&mut bits)?
    } else {
        Vec::new()
    };
    let weights = if (pps.weighted_pred && !is_b) || (pps.weighted_bipred_idc == 1 && is_b) {
        Some(parse_prediction_weight_table(
            &mut bits, is_b, ref_idx_l0, ref_idx_l1,
        )?)
    } else {
        None
    };
    if nal[0] & 0x60 != 0 && bits.read(1)? != 0 {
        return Err(AvcError::Unsupported("adaptive reference picture marking"));
    }
    let cabac_init_idc = bits.ue()?;
    if cabac_init_idc > 2 {
        return Err(AvcError::InvalidData("CABAC initialization out of range"));
    }
    let slice_qp = pps.pic_init_qp + bits.se()?;
    if !(0..=51).contains(&slice_qp) {
        return Err(AvcError::InvalidData("slice QP out of range"));
    }
    if pps.deblocking_filter_control_present {
        let disable_idc = bits.ue()?;
        if disable_idc > 2 {
            return Err(AvcError::InvalidData("invalid deblocking mode"));
        }
        if disable_idc != 1 {
            bits.se()?;
            bits.se()?;
        }
    }
    while bits.bit % 8 != 0 {
        if bits.read(1)? != 1 {
            return Err(AvcError::InvalidData("invalid CABAC alignment"));
        }
    }
    let data_byte_offset = bits.bit / 8;
    if data_byte_offset >= rbsp.len() {
        return Err(AvcError::Incomplete);
    }
    Ok(CabacInterSlice {
        first_mb,
        slice_type,
        frame_num,
        pic_order_cnt_lsb,
        ref_idx_l0,
        ref_idx_l1,
        reorder_l0,
        reorder_l1,
        direct_spatial_mv_pred,
        cabac_init_idc,
        slice_qp,
        weights,
        rbsp,
        data_byte_offset,
    })
}

fn skip_scaling_list(bits: &mut Bits<'_>, count: usize) -> Result<(), AvcError> {
    let mut last = 8i32;
    let mut next = 8i32;
    for _ in 0..count {
        if next != 0 {
            next = (last + bits.se()? + 256).rem_euclid(256);
        }
        if next != 0 {
            last = next;
        }
    }
    Ok(())
}

pub fn parse_sps(nal: &[u8]) -> Result<SequenceParameters, AvcError> {
    let rbsp = rbsp_from_nal(nal, 7)?;
    let mut bits = Bits {
        bytes: &rbsp,
        bit: 0,
    };
    let profile_idc = bits.read(8)? as u8;
    // Only profiles present in the original and March 2005 recommendations.
    if !matches!(profile_idc, 66 | 77 | 88 | 100 | 110 | 122 | 144) {
        return Err(AvcError::UnsupportedProfile(profile_idc));
    }
    let flags = bits.read(8)?;
    let reserved_mask = if matches!(profile_idc, 66 | 77 | 88) {
        0x1f
    } else {
        0x0f
    };
    if flags & reserved_mask != 0 {
        return Err(AvcError::InvalidData("nonzero reserved SPS bits"));
    }
    let level_idc = bits.read(8)? as u8;
    let id = bits.ue()?;
    if id > 31 {
        return Err(AvcError::InvalidData("SPS ID out of range"));
    }
    let mut chroma_format_idc = 1;
    let mut bit_depth_luma = 8;
    let mut bit_depth_chroma = 8;
    let mut scaling_matrices_present = false;
    if matches!(profile_idc, 100 | 110 | 122 | 144) {
        chroma_format_idc = bits.ue()?;
        if chroma_format_idc > 3 {
            return Err(AvcError::InvalidData("invalid chroma format"));
        }
        if chroma_format_idc == 3 {
            bits.read(1)?; // residual_colour_transform_flag in the 2005 edition
        }
        bit_depth_luma = 8 + bits.ue()?;
        bit_depth_chroma = 8 + bits.ue()?;
        if bit_depth_luma > 14 || bit_depth_chroma > 14 {
            return Err(AvcError::InvalidData("invalid bit depth"));
        }
        bits.read(1)?; // qpprime_y_zero_transform_bypass_flag
        scaling_matrices_present = bits.read(1)? != 0;
        if scaling_matrices_present {
            for index in 0..8 {
                if bits.read(1)? != 0 {
                    skip_scaling_list(&mut bits, if index < 6 { 16 } else { 64 })?;
                }
            }
        }
    }
    let frame_num_bits = bits.ue()?.checked_add(4).ok_or(AvcError::TooLarge)? as usize;
    if frame_num_bits > 16 {
        return Err(AvcError::InvalidData("frame number field too wide"));
    }
    let pic_order_cnt_type = bits.ue()?;
    let pic_order_cnt_lsb_bits = match pic_order_cnt_type {
        0 => {
            let count = bits.ue()?.checked_add(4).ok_or(AvcError::TooLarge)? as usize;
            if count > 16 {
                return Err(AvcError::InvalidData("POC field too wide"));
            }
            Some(count)
        }
        1 => {
            bits.read(1)?;
            bits.se()?;
            bits.se()?;
            let count = bits.ue()?;
            if count > 255 {
                return Err(AvcError::TooLarge);
            }
            for _ in 0..count {
                bits.se()?;
            }
            None
        }
        2 => None,
        _ => return Err(AvcError::InvalidData("invalid picture order count type")),
    };
    let max_num_ref_frames = bits.ue()?;
    if max_num_ref_frames > 16 {
        return Err(AvcError::TooLarge);
    }
    bits.read(1)?; // gaps_in_frame_num_value_allowed_flag
    let width_mbs = bits.ue()?.checked_add(1).ok_or(AvcError::TooLarge)?;
    let height_map_units = bits.ue()?.checked_add(1).ok_or(AvcError::TooLarge)?;
    let frame_mbs_only = bits.read(1)? != 0;
    if !frame_mbs_only {
        bits.read(1)?;
    }
    bits.read(1)?; // direct_8x8_inference_flag
    let crop = if bits.read(1)? != 0 {
        [bits.ue()?, bits.ue()?, bits.ue()?, bits.ue()?]
    } else {
        [0; 4]
    };
    let frame_height_units = if frame_mbs_only { 1u32 } else { 2 };
    let (sub_x, sub_y) = match chroma_format_idc {
        0 => (1, 1),
        1 => (2, 2),
        2 => (2, 1),
        _ => (1, 1),
    };
    let width = width_mbs
        .checked_mul(16)
        .and_then(|n| n.checked_sub((crop[0] + crop[1]).checked_mul(sub_x)?))
        .ok_or(AvcError::TooLarge)?;
    let height = height_map_units
        .checked_mul(16)
        .and_then(|n| n.checked_mul(frame_height_units))
        .and_then(|n| n.checked_sub((crop[2] + crop[3]).checked_mul(sub_y * frame_height_units)?))
        .ok_or(AvcError::TooLarge)?;
    if width == 0 || height == 0 || width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(AvcError::TooLarge);
    }
    Ok(SequenceParameters {
        id,
        profile_idc,
        level_idc,
        width,
        height,
        chroma_format_idc,
        bit_depth_luma,
        bit_depth_chroma,
        scaling_matrices_present,
        width_mbs,
        frame_height_mbs: height_map_units * frame_height_units,
        frame_mbs_only,
        frame_num_bits,
        max_num_ref_frames,
        pic_order_cnt_type,
        pic_order_cnt_lsb_bits,
    })
}

fn intra_dc(plane: &[u8], stride: usize, x: usize, y: usize, size: usize) -> u8 {
    let top = (y > 0).then(|| {
        (0..size)
            .map(|i| u32::from(plane[(y - 1) * stride + x + i]))
            .sum::<u32>()
    });
    let left = (x > 0).then(|| {
        (0..size)
            .map(|i| u32::from(plane[(y + i) * stride + x - 1]))
            .sum::<u32>()
    });
    match (top, left) {
        (Some(top), Some(left)) => ((top + left + size as u32) / (2 * size as u32)) as u8,
        (Some(sum), None) | (None, Some(sum)) => ((sum + (size / 2) as u32) / size as u32) as u8,
        (None, None) => 128,
    }
}

fn chroma_dc(plane: &[u8], stride: usize, x: usize, y: usize, quadrant: usize) -> u8 {
    let qx = quadrant % 2;
    let qy = quadrant / 2;
    let top = (y > 0).then(|| {
        (0..4)
            .map(|i| u32::from(plane[(y - 1) * stride + x + qx * 4 + i]))
            .sum::<u32>()
    });
    let left = (x > 0).then(|| {
        (0..4)
            .map(|i| u32::from(plane[(y + qy * 4 + i) * stride + x - 1]))
            .sum::<u32>()
    });
    let (top, left) = match (qx, qy) {
        (1, 0) => (top, None),
        (0, 1) => (None, left),
        _ => (top, left),
    };
    match (top, left) {
        (Some(top), Some(left)) => ((top + left + 4) >> 3) as u8,
        (Some(sum), None) | (None, Some(sum)) => ((sum + 2) >> 2) as u8,
        (None, None) => 128,
    }
}

/// Decodes a complete 2003 Baseline IDR I-slice with I_PCM and zero-residual
/// Intra16x16 DC macroblocks. Other macroblock forms are not reconstructed.
pub fn decode_intra_2003(
    nal: &[u8],
    sps: &SequenceParameters,
    pps: &PictureParameters2003,
) -> Result<super::VideoFrame, AvcError> {
    if sps.profile_idc != 66
        || sps.chroma_format_idc != 1
        || sps.bit_depth_luma != 8
        || sps.bit_depth_chroma != 8
        || !sps.frame_mbs_only
        || sps.pic_order_cnt_type != 0
        || pps.cabac
        || pps.slice_groups != 1
        || pps.pic_order_present
        || pps.redundant_pic_cnt_present
        || pps.sequence_id != sps.id
        || sps.width != sps.width_mbs * 16
        || sps.height != sps.frame_height_mbs * 16
    {
        return Err(AvcError::Unsupported(
            "I_PCM decoder requires uncropped progressive Baseline 4:2:0",
        ));
    }
    if nal
        .first()
        .is_none_or(|byte| byte & 0x1f != 5 || byte & 0x80 != 0 || byte & 0x60 == 0)
    {
        return Err(AvcError::InvalidData("expected reference IDR slice"));
    }
    let rbsp = rbsp_from_nal(nal, 5)?;
    let mut bits = Bits {
        bytes: &rbsp,
        bit: 0,
    };
    let first_mb = bits.ue()?;
    let slice_type = bits.ue()?;
    let pps_id = bits.ue()?;
    if first_mb != 0 || !matches!(slice_type, 2 | 7) || pps_id != pps.id {
        return Err(AvcError::Unsupported("expected a complete I slice"));
    }
    bits.read(sps.frame_num_bits)?;
    bits.ue()?; // idr_pic_id
    bits.read(
        sps.pic_order_cnt_lsb_bits
            .ok_or(AvcError::Unsupported("POC type"))?,
    )?;
    bits.read(2)?; // no_output_of_prior_pics_flag, long_term_reference_flag
    bits.se()?; // slice_qp_delta
    if pps.deblocking_filter_control_present {
        if bits.ue()? != 1 {
            bits.se()?;
            bits.se()?;
        }
    }
    let pixels = u64::from(sps.width) * u64::from(sps.height);
    if pixels > 8 * 1024 * 1024 {
        return Err(AvcError::TooLarge);
    }
    let luma_stride = sps.width as usize;
    let chroma_stride = luma_stride / 2;
    let mut luma = vec![0u8; pixels as usize];
    let mut cb = vec![0u8; pixels as usize / 4];
    let mut cr = vec![0u8; pixels as usize / 4];
    let macroblocks = sps
        .width_mbs
        .checked_mul(sps.frame_height_mbs)
        .ok_or(AvcError::TooLarge)?;
    for mb in 0..macroblocks {
        let mb_x = (mb % sps.width_mbs) as usize;
        let mb_y = (mb / sps.width_mbs) as usize;
        let x0 = mb_x * 16;
        let y0 = mb_y * 16;
        let cx0 = mb_x * 8;
        let cy0 = mb_y * 8;
        match bits.ue()? {
            25 => {
                while bits.bit % 8 != 0 {
                    if bits.read(1)? != 0 {
                        return Err(AvcError::InvalidData("invalid PCM alignment"));
                    }
                }
                for y in 0..16 {
                    for x in 0..16 {
                        luma[(y0 + y) * luma_stride + x0 + x] = bits.read(8)? as u8;
                    }
                }
                for plane in [&mut cb, &mut cr] {
                    for y in 0..8 {
                        for x in 0..8 {
                            plane[(cy0 + y) * chroma_stride + cx0 + x] = bits.read(8)? as u8;
                        }
                    }
                }
            }
            3 => {
                if bits.ue()? != 0 || bits.se()? != 0 || bits.read(1)? != 1 {
                    return Err(AvcError::Unsupported(
                        "Intra16x16 DC requires zero residual and chroma DC",
                    ));
                }
                let dc = intra_dc(&luma, luma_stride, x0, y0, 16);
                for y in 0..16 {
                    luma[(y0 + y) * luma_stride + x0..(y0 + y) * luma_stride + x0 + 16].fill(dc);
                }
                for plane in [&mut cb, &mut cr] {
                    for quadrant in 0..4 {
                        let value = chroma_dc(plane, chroma_stride, cx0, cy0, quadrant);
                        let qx = quadrant % 2;
                        let qy = quadrant / 2;
                        for y in 0..4 {
                            let start = (cy0 + qy * 4 + y) * chroma_stride + cx0 + qx * 4;
                            plane[start..start + 4].fill(value);
                        }
                    }
                }
            }
            _ => return Err(AvcError::Unsupported("unsupported intra macroblock")),
        }
    }
    bits.finish_rbsp()?;
    Ok(frame_from_yuv420(sps, &luma, &cb, &cr))
}

pub(super) fn frame_from_yuv420(
    sps: &SequenceParameters,
    luma: &[u8],
    cb: &[u8],
    cr: &[u8],
) -> super::VideoFrame {
    let luma_stride = sps.width as usize;
    let chroma_stride = luma_stride / 2;
    let mut rgba = vec![0u8; luma.len() * 4];
    for y in 0..sps.height as usize {
        for x in 0..luma_stride {
            let value = i32::from(luma[y * luma_stride + x]) - 16;
            let uv = (y / 2) * chroma_stride + x / 2;
            let u = i32::from(cb[uv]) - 128;
            let v = i32::from(cr[uv]) - 128;
            let c = value.max(0) * 298;
            let index = (y * luma_stride + x) * 4;
            rgba[index] = ((c + 409 * v + 128) >> 8).clamp(0, 255) as u8;
            rgba[index + 1] = ((c - 100 * u - 208 * v + 128) >> 8).clamp(0, 255) as u8;
            rgba[index + 2] = ((c + 516 * u + 128) >> 8).clamp(0, 255) as u8;
            rgba[index + 3] = 255;
        }
    }
    super::VideoFrame {
        width: sps.width,
        height: sps.height,
        rgba: std::sync::Arc::new(rgba),
        timestamp: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct BitWriter(Vec<bool>);

    impl BitWriter {
        fn bits(&mut self, value: u32, count: usize) {
            for shift in (0..count).rev() {
                self.0.push(value & (1 << shift) != 0);
            }
        }

        fn ue(&mut self, value: u32) {
            let code = value + 1;
            let width = (32 - code.leading_zeros()) as usize;
            self.bits(0, width - 1);
            self.bits(code, width);
        }

        fn se(&mut self, value: i32) {
            let code = if value <= 0 {
                (-value as u32) * 2
            } else {
                value as u32 * 2 - 1
            };
            self.ue(code);
        }

        fn align_zero(&mut self) {
            while !self.0.len().is_multiple_of(8) {
                self.bits(0, 1);
            }
        }

        fn bytes(&mut self, data: &[u8]) {
            assert!(self.0.len().is_multiple_of(8));
            for byte in data {
                self.bits(u32::from(*byte), 8);
            }
        }

        fn finish(mut self) -> Vec<u8> {
            self.bits(1, 1);
            while !self.0.len().is_multiple_of(8) {
                self.bits(0, 1);
            }
            self.0
                .chunks(8)
                .map(|chunk| {
                    chunk
                        .iter()
                        .fold(0, |byte, bit| (byte << 1) | u8::from(*bit))
                })
                .collect()
        }
    }

    fn high_720p_sps() -> Vec<u8> {
        let mut bits = BitWriter(Vec::new());
        bits.bits(100, 8); // High Profile (March 2005)
        bits.bits(0, 8); // constraint flags
        bits.bits(31, 8); // level 3.1
        bits.ue(0); // SPS id
        bits.ue(1); // 4:2:0
        bits.ue(0); // 8-bit luma
        bits.ue(0); // 8-bit chroma
        bits.bits(0, 2); // transform bypass, scaling matrices
        bits.ue(0); // frame number bits minus four
        bits.ue(0); // picture order count type
        bits.ue(0); // POC LSB bits minus four
        bits.ue(1); // one reference frame
        bits.bits(0, 1); // frame number gaps disallowed
        bits.ue(79); // 80 macroblocks wide
        bits.ue(44); // 45 macroblocks high
        bits.bits(1, 1); // frame-only pictures
        bits.bits(1, 1); // direct 8x8 inference
        bits.bits(0, 1); // no crop
        bits.bits(0, 1); // no VUI
        let mut nal = vec![0x67];
        nal.extend(bits.finish());
        nal
    }

    #[test]
    fn parses_2005_high_profile_config() {
        let sps = high_720p_sps();
        let pps = [0x68, 0xce, 0x06, 0xe2];
        let mut avcc = vec![1, 100, 0, 31, 0xff, 0xe1];
        avcc.extend((sps.len() as u16).to_be_bytes());
        avcc.extend(&sps);
        avcc.push(1);
        avcc.extend((pps.len() as u16).to_be_bytes());
        avcc.extend(pps);
        let config = AvcConfig::parse(&avcc).unwrap();
        assert_eq!(config.nal_length_size, 4);
        assert_eq!(config.sequence_parameters[0].width, 1280);
        assert_eq!(config.sequence_parameters[0].height, 720);
        assert_eq!(config.sequence_parameters[0].profile_idc, 100);
        assert_eq!(config.picture_parameter_sets, vec![pps.to_vec()]);
        avcc[3] = 30;
        assert_eq!(
            AvcConfig::parse(&avcc),
            Err(AvcError::InvalidData("avcC and SPS disagree"))
        );
    }

    #[test]
    fn parses_original_2003_picture_parameter_set() {
        let mut bits = BitWriter(Vec::new());
        bits.ue(0); // PPS id
        bits.ue(0); // SPS id
        bits.bits(0, 2); // CAVLC, no bottom-field order count
        bits.ue(0); // one slice group
        bits.ue(0); // one L0 reference
        bits.ue(0); // one L1 reference
        bits.bits(0, 3); // no weighted prediction
        bits.se(0); // QP = 26
        bits.se(0); // QS = 26
        bits.se(0); // chroma QP offset
        bits.bits(1, 1); // deblocking filter control
        bits.bits(0, 2); // no constrained intra or redundant pictures
        let mut nal = vec![0x68];
        nal.extend(bits.finish());
        let pps = parse_pps_2003(&nal).unwrap();
        assert_eq!((pps.id, pps.sequence_id, pps.pic_init_qp), (0, 0, 26));
        assert!(!pps.cabac);
        assert_eq!(pps.slice_groups, 1);
        assert!(pps.deblocking_filter_control_present);
        nal.push(0);
        assert_eq!(
            parse_pps_2003(&nal),
            Err(AvcError::InvalidData("unexpected RBSP data"))
        );
    }

    #[test]
    fn parses_2003_disperse_slice_group_map_without_extra_fields() {
        let mut bits = BitWriter(Vec::new());
        bits.ue(0);
        bits.ue(0);
        bits.bits(0, 2);
        bits.ue(1); // two slice groups
        bits.ue(1); // dispersed map has no further syntax
        bits.ue(0);
        bits.ue(0);
        bits.bits(0, 3);
        bits.se(0);
        bits.se(0);
        bits.se(0);
        bits.bits(1, 1);
        bits.bits(0, 2);
        let mut nal = vec![0x68];
        nal.extend(bits.finish());
        let pps = parse_pps_2003(&nal).unwrap();
        assert_eq!((pps.slice_groups, pps.slice_group_map_type), (2, Some(1)));
    }

    #[test]
    fn decodes_original_2003_pcm_and_intra_dc_idr_into_pixels() {
        let mut sps_bits = BitWriter(Vec::new());
        sps_bits.bits(66, 8); // Baseline
        sps_bits.bits(0, 8); // constraints
        sps_bits.bits(10, 8); // level 1.0
        sps_bits.ue(0); // SPS ID
        sps_bits.ue(0); // four-bit frame number
        sps_bits.ue(0); // picture order count type 0
        sps_bits.ue(0); // four-bit POC LSB
        sps_bits.ue(1); // one reference frame
        sps_bits.bits(0, 1); // no frame number gaps
        sps_bits.ue(1); // two macroblocks wide
        sps_bits.ue(0); // one macroblock high
        sps_bits.bits(1, 1); // progressive
        sps_bits.bits(1, 1); // direct 8x8 inference
        sps_bits.bits(0, 2); // no crop or VUI
        let mut sps_nal = vec![0x67];
        sps_nal.extend(sps_bits.finish());
        let sps = parse_sps(&sps_nal).unwrap();

        let mut pps_bits = BitWriter(Vec::new());
        pps_bits.ue(0); // PPS ID
        pps_bits.ue(0); // SPS ID
        pps_bits.bits(0, 2); // CAVLC, no bottom field POC
        pps_bits.ue(0); // one slice group
        pps_bits.ue(0); // one L0 reference
        pps_bits.ue(0); // one L1 reference
        pps_bits.bits(0, 3); // no weighted prediction
        pps_bits.se(0);
        pps_bits.se(0);
        pps_bits.se(0);
        pps_bits.bits(1, 1); // deblock setting in slice
        pps_bits.bits(0, 2);
        let mut pps_nal = vec![0x68];
        pps_nal.extend(pps_bits.finish());
        let pps = parse_pps_2003(&pps_nal).unwrap();

        let mut slice = BitWriter(Vec::new());
        slice.ue(0); // first macroblock
        slice.ue(2); // I slice
        slice.ue(0); // PPS ID
        slice.bits(0, 4); // frame number
        slice.ue(0); // IDR picture ID
        slice.bits(0, 4); // POC LSB
        slice.bits(0, 2); // IDR reference marking
        slice.se(0); // QP delta
        slice.ue(1); // deblocking disabled
        slice.ue(25); // I_PCM macroblock
        slice.align_zero();
        let mut pcm = vec![235; 256];
        pcm.extend([128; 128]);
        slice.bytes(&pcm);
        slice.ue(3); // I_16x16_2_0_0 (DC, no coded chroma or luma AC)
        slice.ue(0); // chroma DC prediction
        slice.se(0); // macroblock QP delta
        slice.bits(1, 1); // zero luma DC coefficients, Table 9-5 (nC = 0)
        let mut slice_nal = vec![0x65];
        slice_nal.extend(slice.finish());
        let frame = decode_intra_2003(&slice_nal, &sps, &pps).unwrap();
        assert_eq!((frame.width, frame.height), (32, 16));
        assert_eq!(&frame.rgba[..4], &[255, 255, 255, 255]);
        assert_eq!(&frame.rgba[frame.rgba.len() - 4..], &[255, 255, 255, 255]);
    }

    #[test]
    fn parses_maroc_mp4_high_profile_configuration() {
        // avcC from the site's MP4, captured as a small metadata-only fixture.
        let avcc = [
            0x01, 0x64, 0x00, 0x20, 0xff, 0xe1, 0x00, 0x1d, 0x67, 0x64, 0x00, 0x20, 0xac, 0xd9,
            0x40, 0x50, 0x05, 0xbb, 0xff, 0x04, 0x40, 0x04, 0x41, 0x10, 0x00, 0x00, 0x03, 0x00,
            0x10, 0x00, 0x00, 0x06, 0x48, 0xf1, 0x83, 0x19, 0x60, 0x01, 0x00, 0x06, 0x68, 0xeb,
            0xe3, 0xcb, 0x22, 0xc0, 0xfd, 0xf8, 0xf8, 0x00,
        ];
        let config = AvcConfig::parse(&avcc).unwrap();
        assert_eq!(config.nal_length_size, 4);
        assert_eq!(config.sequence_parameters[0].profile_idc, 100);
        assert_eq!(config.sequence_parameters[0].level_idc, 32);
        assert_eq!(
            (
                config.sequence_parameters[0].width,
                config.sequence_parameters[0].height
            ),
            (1280, 720)
        );
    }

    #[test]
    fn length_prefixed_nals_arrive_across_arbitrary_chunks() {
        let bytes = [0, 0, 0, 2, 0x65, 0x99, 0, 0, 0, 3, 0x41, 0x12, 0x34];
        for split in 1..bytes.len() {
            let mut stream = NalStream::new(4).unwrap();
            let mut units = stream.push(&bytes[..split]).unwrap();
            units.extend(stream.push(&bytes[split..]).unwrap());
            assert_eq!(units, vec![vec![0x65, 0x99], vec![0x41, 0x12, 0x34]]);
            stream.finish().unwrap();
        }
    }

    #[test]
    fn rejects_truncated_and_unbounded_nals() {
        let mut stream = NalStream::new(4).unwrap();
        assert!(stream.push(&[0, 0, 0, 2, 0x65]).unwrap().is_empty());
        assert_eq!(stream.finish(), Err(AvcError::Incomplete));
        assert_eq!(stream.push(&[0x99]).unwrap(), vec![vec![0x65, 0x99]]);
        stream.finish().unwrap();
        assert_eq!(
            NalStream::new(3).err(),
            Some(AvcError::InvalidData("invalid NAL length size"))
        );
        assert_eq!(
            NalStream::new(4).unwrap().push(&[0xff, 0xff, 0xff, 0xff]),
            Err(AvcError::TooLarge)
        );
        assert_eq!(
            parse_sps(&[0x67, 118]),
            Err(AvcError::UnsupportedProfile(118))
        );
    }
}
