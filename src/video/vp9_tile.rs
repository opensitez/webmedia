//! VP9 keyframe tile partition syntax.

use super::backend::MediaDecodeError;
use super::vp8::BoolDecoder;
use super::vp9::{InterframeHeader, KeyframeLayout};
use super::vp9_adapt::{CoefficientCounts, NonCoefficientCounts};
use super::vp9_compressed::CompressedHeader;
use super::vp9_coef_probs::PARETO_TABLE;
use super::vp9_mode_probs::KEYFRAME_LUMA_MODE_PROBS;
use super::vp9_motion::{read_motion_difference, read_motion_difference_counted, use_high_precision};
use super::vp9_scan::coefficient_scan;

const INTRA_MODE_TREE: [i8; 18] = [
    0, 2, -9, 4, -1, 6, 8, 12, -2, 10, -4, -5, -3, 14, -8, 16, -6, -7,
];
const INTER_MODE_TREE: [i8; 6] = [-2, 2, 0, 4, -1, -3];
const INTERP_FILTER_TREE: [i8; 4] = [0, 2, -1, -2];
const MV_REF_BLOCKS: [[(isize, isize); 8]; 13] = [
    [(-1, 0), (0, -1), (-1, -1), (-2, 0), (0, -2), (-2, -1), (-1, -2), (-2, -2)],
    [(-1, 0), (0, -1), (-1, -1), (-2, 0), (0, -2), (-2, -1), (-1, -2), (-2, -2)],
    [(-1, 0), (0, -1), (-1, -1), (-2, 0), (0, -2), (-2, -1), (-1, -2), (-2, -2)],
    [(-1, 0), (0, -1), (-1, -1), (-2, 0), (0, -2), (-2, -1), (-1, -2), (-2, -2)],
    [(0, -1), (-1, 0), (1, -1), (-1, -1), (0, -2), (-2, 0), (-2, -1), (-1, -2)],
    [(-1, 0), (0, -1), (-1, 1), (-1, -1), (-2, 0), (0, -2), (-1, -2), (-2, -1)],
    [(-1, 0), (0, -1), (-1, 1), (1, -1), (-1, -1), (-3, 0), (0, -3), (-3, -3)],
    [(0, -1), (-1, 0), (2, -1), (-1, -1), (-1, 1), (0, -3), (-3, 0), (-3, -3)],
    [(-1, 0), (0, -1), (-1, 2), (-1, -1), (1, -1), (-3, 0), (0, -3), (-3, -3)],
    [(-1, 1), (1, -1), (-1, 2), (2, -1), (-1, -1), (-3, 0), (0, -3), (-3, -3)],
    [(0, -1), (-1, 0), (4, -1), (-1, 2), (-1, -1), (0, -3), (-3, 0), (2, -1)],
    [(-1, 0), (0, -1), (-1, 4), (2, -1), (-1, -1), (-3, 0), (0, -3), (-1, 2)],
    [(-1, 3), (3, -1), (-1, 4), (4, -1), (-1, -1), (-1, 0), (0, -1), (-1, 6)],
];
const SEGMENT_TREE: [i8; 14] = [2, 4, 6, 8, 10, 12, 0, -1, -2, -3, -4, -5, -6, -7];
const TOKEN_TREE: [i8; 20] = [0, 2, -1, 4, 6, 10, -2, 8, -3, -4, 12, 14, -5, -6, 16, 18, -7, -8, -9, -10];
const KEYFRAME_UV_MODE_PROBS: [[u8; 9]; 10] = [
    [144, 11, 54, 157, 195, 130, 46, 58, 108],
    [118, 15, 123, 148, 131, 101, 44, 93, 131],
    [113, 12, 23, 188, 226, 142, 26, 32, 125],
    [120, 11, 50, 123, 163, 135, 64, 77, 103],
    [113, 9, 36, 155, 111, 157, 32, 44, 161],
    [116, 9, 55, 176, 76, 96, 37, 61, 149],
    [115, 9, 28, 141, 161, 167, 21, 25, 193],
    [120, 12, 32, 145, 195, 142, 32, 38, 86],
    [116, 12, 64, 120, 140, 125, 49, 115, 121],
    [102, 19, 66, 162, 182, 122, 35, 59, 128],
];
pub(super) const INTERFRAME_UV_MODE_PROBS: [[u8; 9]; 10] = [
    [120, 7, 76, 176, 208, 126, 28, 54, 103],
    [48, 12, 154, 155, 139, 90, 34, 117, 119],
    [67, 6, 25, 204, 243, 158, 13, 21, 96],
    [97, 5, 44, 131, 176, 139, 48, 68, 97],
    [83, 5, 42, 156, 111, 152, 26, 49, 152],
    [80, 5, 58, 178, 74, 83, 33, 62, 145],
    [86, 5, 32, 154, 192, 168, 14, 22, 163],
    [85, 5, 32, 156, 216, 148, 19, 29, 73],
    [77, 7, 64, 116, 132, 122, 37, 126, 120],
    [101, 21, 107, 181, 192, 103, 19, 67, 125],
];

const KEYFRAME_PARTITION_PROBS: [[u8; 3]; 16] = [
    [158, 97, 94], [93, 24, 99], [85, 119, 44], [62, 59, 67],
    [149, 53, 53], [94, 20, 48], [83, 53, 24], [52, 18, 18],
    [150, 40, 39], [78, 12, 26], [67, 33, 11], [24, 7, 5],
    [174, 35, 49], [68, 11, 27], [57, 15, 9], [12, 3, 3],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Partition {
    None,
    Horizontal,
    Vertical,
    Split,
}

fn block_size_index(width: usize, height: usize) -> Option<usize> {
    const SIZES: [(usize, usize); 13] = [
        (4, 4), (4, 8), (8, 4), (8, 8), (8, 16), (16, 8),
        (16, 16), (16, 32), (32, 16), (32, 32), (32, 64),
        (64, 32), (64, 64),
    ];
    SIZES.iter().position(|&size| size == (width, height))
}

fn single_reference_contexts(
    above_available: bool, left_available: bool,
    above: Option<InterPrediction>, left: Option<InterPrediction>,
) -> (usize, usize) {
    let a = reference_pair(above);
    let l = reference_pair(left);
    let has = |refs: [u8; 2], reference: u8| refs.contains(&reference);
    let single = |refs: [u8; 2]| refs[1] == 0;
    let one = |refs: [u8; 2]| {
        if refs[0] == 0 { (2, 2) }
        else if single(refs) {
            (4 * usize::from(refs[0] == 1),
                if refs[0] == 1 { 2 } else { 4 * usize::from(refs[0] == 2) })
        } else {
            (1 + usize::from(has(refs, 1)), 3 * usize::from(has(refs, 2)))
        }
    };
    match (above_available, left_available) {
        (true, true) if a[0] == 0 && l[0] == 0 => (2, 2),
        (true, true) if a[0] == 0 || l[0] == 0 => {
            let refs = if a[0] == 0 { l } else { a };
            if single(refs) {
                (4 * usize::from(refs[0] == 1),
                    if refs[0] == 1 { 3 } else { 4 * usize::from(refs[0] == 2) })
            } else {
                (1 + usize::from(has(refs, 1)), 1 + 2 * usize::from(has(refs, 2)))
            }
        }
        (true, true) if single(a) && single(l) => {
            let p1 = 2 * usize::from(a[0] == 1) + 2 * usize::from(l[0] == 1);
            let p2 = if a[0] == 1 && l[0] == 1 { 3 }
                else if a[0] == 1 { 4 * usize::from(l[0] == 2) }
                else if l[0] == 1 { 4 * usize::from(a[0] == 2) }
                else { 2 * usize::from(a[0] == 2) + 2 * usize::from(l[0] == 2) };
            (p1, p2)
        }
        (true, true) if !single(a) && !single(l) => {
            let p1 = 1 + usize::from(has(a, 1) || has(l, 1));
            let p2 = if a == l { 3 * usize::from(has(a, 2)) } else { 2 };
            (p1, p2)
        }
        (true, true) => {
            let (s, c) = if single(a) { (a, l) } else { (l, a) };
            let p1 = if s[0] == 1 { 3 + usize::from(has(c, 1)) }
                else { usize::from(has(c, 1)) };
            let p2 = if s[0] == 2 { 3 + usize::from(has(c, 2)) }
                else if s[0] == 3 { usize::from(has(c, 2)) }
                else { 1 + 2 * usize::from(has(c, 2)) };
            (p1, p2)
        }
        (true, false) => one(a),
        (false, true) => one(l),
        (false, false) => (2, 2),
    }
}

fn compound_references(frame: &InterframeHeader) -> (u8, [u8; 2]) {
    let bias = frame.reference_sign_bias;
    if bias[0] == bias[1] { (3, [1, 2]) }
    else if bias[0] == bias[2] { (2, [1, 3]) }
    else { (1, [2, 3]) }
}

fn reference_pair(prediction: Option<InterPrediction>) -> [u8; 2] {
    prediction.map_or([0, 0], |value| [value.reference,
        value.second.map_or(0, |second| second.reference)])
}

fn compound_mode_context(above_available: bool, left_available: bool,
    above: Option<InterPrediction>, left: Option<InterPrediction>, fixed: u8,
) -> usize {
    let a = reference_pair(above);
    let l = reference_pair(left);
    let a_single = a[1] == 0;
    let l_single = l[1] == 0;
    match (above_available, left_available) {
        (true, true) if a_single && l_single => usize::from((a[0] == fixed) ^ (l[0] == fixed)),
        (true, true) if a_single => 2 + usize::from(a[0] == fixed || a[0] == 0),
        (true, true) if l_single => 2 + usize::from(l[0] == fixed || l[0] == 0),
        (true, true) => 4,
        (true, false) => if a_single { usize::from(a[0] == fixed) } else { 3 },
        (false, true) => if l_single { usize::from(l[0] == fixed) } else { 3 },
        (false, false) => 1,
    }
}

fn compound_ref_context(above_available: bool, left_available: bool,
    above: Option<InterPrediction>, left: Option<InterPrediction>,
    fixed: u8, variable: [u8; 2], fixed_index: usize,
) -> usize {
    let a = reference_pair(above);
    let l = reference_pair(left);
    let a_single = a[1] == 0;
    let l_single = l[1] == 0;
    let variable_index = 1 - fixed_index;
    let a_var = if a_single { a[0] } else { a[variable_index] };
    let l_var = if l_single { l[0] } else { l[variable_index] };
    match (above_available, left_available) {
        (true, true) if a[0] == 0 && l[0] == 0 => 2,
        (true, true) if l[0] == 0 => 1 + 2 * usize::from(a_var != variable[1]),
        (true, true) if a[0] == 0 => 1 + 2 * usize::from(l_var != variable[1]),
        (true, true) if a_var == l_var && a_var == variable[1] => 0,
        (true, true) if a_single && l_single => {
            if (a_var == fixed && l_var == variable[0])
                || (l_var == fixed && a_var == variable[0]) { 4 }
            else if a_var == l_var { 3 } else { 1 }
        }
        (true, true) if a_single || l_single => {
            let single = if a_single { a_var } else { l_var };
            let compound = if a_single { l_var } else { a_var };
            if compound == variable[1] && single != variable[1] { 1 }
            else if single == variable[1] && compound != variable[1] { 2 }
            else { 4 }
        }
        (true, true) if a_var == l_var => 4,
        (true, true) => 2,
        (true, false) if a[0] == 0 => 2,
        (true, false) if a_single => 3 * usize::from(a_var != variable[1]),
        (true, false) => 4 * usize::from(a_var != variable[1]),
        (false, true) if l[0] == 0 => 2,
        (false, true) if l_single => 3 * usize::from(l_var != variable[1]),
        (false, true) => 4 * usize::from(l_var != variable[1]),
        (false, false) => 2,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FirstBlock {
    pub partition: Partition,
    pub block_size: usize,
    pub segment_id: u8,
    pub skip: bool,
    pub tx_size: u8,
    pub y_mode: u8,
    pub uv_mode: u8,
    pub sub_modes: Option<[u8; 4]>,
    pub first_luma_token: Option<u8>,
    pub first_luma_coefficient: Option<i32>,
    pub luma_coefficients: Option<Vec<Vec<i32>>>,
    pub chroma_coefficients: Option<[Vec<i32>; 2]>,
    pub first_luma_eob: usize,
    pub luma_nonzero: [bool; 4],
    pub chroma_nonzero: [bool; 2],
    pub inter: Option<InterPrediction>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterPrediction {
    pub reference: u8,
    pub mode: u8,
    pub interpolation_filter: u8,
    pub motion: (i32, i32),
    pub sub_motions: Option<[(i32, i32); 4]>,
    pub second: Option<SecondReference>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SecondReference {
    pub reference: u8,
    pub motion: (i32, i32),
    pub sub_motions: Option<[(i32, i32); 4]>,
}

struct ParsedTransform {
    coefficients: Vec<i32>,
    first_token: Option<u8>,
    first_coefficient: Option<i32>,
    eob: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileBlock {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub block: FirstBlock,
}

pub fn decode_keyframe_tile_prefix(
    layout: &KeyframeLayout<'_>,
    compressed: &CompressedHeader,
    tile: &[u8],
    tile_index: usize,
    max_blocks: usize,
) -> Result<Vec<TileBlock>, MediaDecodeError> {
    decode_keyframe_tile_counted(layout, compressed, tile, tile_index, max_blocks)
        .map(|(blocks, _)| blocks)
}

pub(super) fn decode_keyframe_tile_counted(
    layout: &KeyframeLayout<'_>,
    compressed: &CompressedHeader,
    tile: &[u8],
    tile_index: usize,
    max_blocks: usize,
) -> Result<(Vec<TileBlock>, CoefficientCounts), MediaDecodeError> {
    let width = layout.header.width.ok_or(MediaDecodeError::Unsupported)? as usize;
    let height = layout.header.height.ok_or(MediaDecodeError::Unsupported)? as usize;
    if layout.header.bit_depth != Some(8) {
        return Err(MediaDecodeError::Unsupported);
    }
    let mut reader = KeyframeTileReader::new(tile, width, height, tile_index, layout, compressed)?;
    for row in (reader.row_start..reader.row_end).step_by(8) {
        reader.left_partition.fill(0);
        for plane in &mut reader.left_nonzero {
            plane.fill(false);
        }
        for col in (reader.col_start..reader.col_end).step_by(8) {
            if reader.blocks.len() >= max_blocks {
                return Ok((reader.blocks, reader.coef_counts));
            }
            if let Err(error) = reader.decode_partition(row, col, 64, max_blocks) {
                if std::env::var_os("WEBMEDIA_VP9_REPORT").is_some() {
                    eprintln!("VP9 tile decode failed at superblock ({row}, {col}), blocks={}, last={:?}: {error:?}",
                        reader.blocks.len(), reader.blocks.last().map(|block| (block.x, block.y,
                            block.width, block.height, block.block.inter)));
                }
                return Err(error);
            }
        }
    }
    Ok((reader.blocks, reader.coef_counts))
}

pub fn decode_interframe_tile_prefix(
    layout: &KeyframeLayout<'_>,
    frame: &InterframeHeader,
    compressed: &CompressedHeader,
    tile: &[u8],
    tile_index: usize,
    max_blocks: usize,
    previous_predictions: Option<&[Option<InterPrediction>]>,
    previous_segments: Option<&[u8]>,
) -> Result<Vec<TileBlock>, MediaDecodeError> {
    decode_interframe_tile_counted(layout, frame, compressed, tile, tile_index, max_blocks,
        previous_predictions, previous_segments).map(|(blocks, _, _)| blocks)
}

pub(super) fn decode_interframe_tile_counted(
    layout: &KeyframeLayout<'_>,
    frame: &InterframeHeader,
    compressed: &CompressedHeader,
    tile: &[u8],
    tile_index: usize,
    max_blocks: usize,
    previous_predictions: Option<&[Option<InterPrediction>]>,
    previous_segments: Option<&[u8]>,
) -> Result<(Vec<TileBlock>, CoefficientCounts, NonCoefficientCounts), MediaDecodeError> {
    if frame.intra_only {
        return Err(MediaDecodeError::Unsupported);
    }
    let width = layout.header.width.ok_or(MediaDecodeError::Unsupported)? as usize;
    let height = layout.header.height.ok_or(MediaDecodeError::Unsupported)? as usize;
    let mut reader = KeyframeTileReader::new(tile, width, height, tile_index, layout, compressed)?;
    reader.inter = Some(frame);
    reader.previous_predictions = previous_predictions;
    reader.previous_segments = previous_segments;
    for row in (reader.row_start..reader.row_end).step_by(8) {
        reader.left_partition.fill(0);
        reader.left_seg_pred.fill(false);
        for plane in &mut reader.left_nonzero {
            plane.fill(false);
        }
        for col in (reader.col_start..reader.col_end).step_by(8) {
            if reader.blocks.len() >= max_blocks {
                return Ok((reader.blocks, reader.coef_counts, reader.noncoef_counts));
            }
            if let Err(error) = reader.decode_partition(row, col, 64, max_blocks) {
                if std::env::var_os("WEBMEDIA_VP9_REPORT").is_some() {
                    eprintln!("VP9 tile decode failed at superblock ({row}, {col}), blocks={}, last={:?}: {error:?}",
                        reader.blocks.len(), reader.blocks.last().map(|block| (block.x, block.y,
                            block.width, block.height, block.block.inter)));
                }
                return Err(error);
            }
        }
    }
    Ok((reader.blocks, reader.coef_counts, reader.noncoef_counts))
}

pub(super) fn decode_frame_tiles_counted<'a, 'b>(
    layout: &'b KeyframeLayout<'a>, compressed: &'b CompressedHeader,
    frame: Option<&'b InterframeHeader>,
    previous_predictions: Option<&'b [Option<InterPrediction>]>,
    previous_segments: Option<&'b [u8]>,
) -> Result<(Vec<Vec<TileBlock>>, CoefficientCounts, NonCoefficientCounts), MediaDecodeError> {
    let width = layout.header.width.ok_or(MediaDecodeError::Unsupported)? as usize;
    let height = layout.header.height.ok_or(MediaDecodeError::Unsupported)? as usize;
    let tiles = layout.tile_partitions()?;
    let bounds: Vec<_> = (0..tiles.len()).map(|index| tile_bounds(width, height, index, layout))
        .collect::<Result<_, _>>()?;
    let first = bounds.iter().position(|&(row_start, row_end, col_start, col_end)|
        row_start < row_end && col_start < col_end)
        .ok_or_else(|| MediaDecodeError::InvalidData("empty VP9 tile grid".into()))?;
    let mut reader = KeyframeTileReader::new(tiles[first], width, height, first, layout, compressed)?;
    reader.inter = frame;
    reader.previous_predictions = previous_predictions;
    reader.previous_segments = previous_segments;
    let mut blocks = Vec::with_capacity(tiles.len());
    for (index, tile) in tiles.iter().enumerate() {
        let (row_start, row_end, col_start, col_end) = bounds[index];
        if row_start == row_end || col_start == col_end {
            blocks.push(Vec::new());
            continue;
        }
        if index != first {
            reader.bits = BoolDecoder::new(tile)?;
            if reader.bits.read_bit()? {
                return Err(MediaDecodeError::InvalidData("invalid VP9 tile marker".into()));
            }
            reader.row_start = row_start;
            reader.row_end = row_end;
            reader.col_start = col_start;
            reader.col_end = col_end;
        }
        // Only the arithmetic coder restarts at a tile row; above contexts span the frame.
        for row in (row_start..row_end).step_by(8) {
            reader.left_partition.fill(0);
            reader.left_seg_pred.fill(false);
            for plane in &mut reader.left_nonzero { plane.fill(false); }
            for col in (col_start..col_end).step_by(8) {
                reader.decode_partition(row, col, 64, usize::MAX)?;
            }
        }
        blocks.push(std::mem::take(&mut reader.blocks));
    }
    Ok((blocks, reader.coef_counts, reader.noncoef_counts))
}

fn tile_bounds(width: usize, height: usize, index: usize, layout: &KeyframeLayout<'_>)
    -> Result<(usize, usize, usize, usize), MediaDecodeError> {
    let cols = 1usize << layout.tile_cols_log2;
    let rows = 1usize << layout.tile_rows_log2;
    if index >= cols * rows {
        return Err(MediaDecodeError::InvalidData("VP9 tile index out of range".into()));
    }
    let offset = |tile: usize, count: usize, log2: u8|
        (((tile * count.div_ceil(8)) >> log2) << 3).min(count);
    let mi_cols = width.div_ceil(8);
    let mi_rows = height.div_ceil(8);
    Ok((offset(index / cols, mi_rows, layout.tile_rows_log2),
        offset(index / cols + 1, mi_rows, layout.tile_rows_log2),
        offset(index % cols, mi_cols, layout.tile_cols_log2),
        offset(index % cols + 1, mi_cols, layout.tile_cols_log2)))
}

struct KeyframeTileReader<'a, 'b> {
    bits: BoolDecoder<'a>,
    layout: &'b KeyframeLayout<'a>,
    compressed: &'b CompressedHeader,
    inter: Option<&'b InterframeHeader>,
    previous_predictions: Option<&'b [Option<InterPrediction>]>,
    previous_segments: Option<&'b [u8]>,
    mi_cols: usize,
    mi_rows: usize,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    above_partition: Vec<u8>,
    left_partition: Vec<u8>,
    above_seg_pred: Vec<bool>,
    left_seg_pred: Vec<bool>,
    sub_modes: Vec<[u8; 4]>,
    skips: Vec<bool>,
    tx_sizes: Vec<u8>,
    inter_predictions: Vec<Option<InterPrediction>>,
    above_nonzero: [Vec<bool>; 3],
    left_nonzero: [Vec<bool>; 3],
    blocks: Vec<TileBlock>,
    coef_counts: CoefficientCounts,
    noncoef_counts: NonCoefficientCounts,
}

impl<'a, 'b> KeyframeTileReader<'a, 'b> {
    fn new(
        tile: &'a [u8],
        width: usize,
        height: usize,
        tile_index: usize,
        layout: &'b KeyframeLayout<'a>,
        compressed: &'b CompressedHeader,
    ) -> Result<Self, MediaDecodeError> {
        let mut bits = BoolDecoder::new(tile)?;
        if bits.read_bit()? {
            return Err(MediaDecodeError::InvalidData("invalid VP9 tile marker".into()));
        }
        let mi_cols = width.div_ceil(8);
        let mi_rows = height.div_ceil(8);
        let (row_start, row_end, col_start, col_end) = tile_bounds(width, height, tile_index, layout)?;
        Ok(Self {
            bits, layout, compressed, inter: None, previous_predictions: None,
            previous_segments: None,
            mi_cols, mi_rows, row_start, row_end, col_start, col_end,
            above_partition: vec![0; mi_cols],
            left_partition: vec![0; mi_rows],
            above_seg_pred: vec![false; mi_cols],
            left_seg_pred: vec![false; mi_rows],
            sub_modes: vec![[0; 4]; mi_cols * mi_rows],
            skips: vec![false; mi_cols * mi_rows],
            tx_sizes: vec![0; mi_cols * mi_rows],
            inter_predictions: vec![None; mi_cols * mi_rows],
            above_nonzero: std::array::from_fn(|plane| vec![false; mi_cols * if plane == 0 { 2 } else { 1 }]),
            left_nonzero: std::array::from_fn(|plane| vec![false; mi_rows * if plane == 0 { 2 } else { 1 }]),
            blocks: Vec::new(),
            coef_counts: CoefficientCounts::default(),
            noncoef_counts: NonCoefficientCounts::default(),
        })
    }

    fn decode_partition(
        &mut self,
        row: usize,
        col: usize,
        size: usize,
        max_blocks: usize,
    ) -> Result<(), MediaDecodeError> {
        if row >= self.mi_rows || col >= self.mi_cols || self.blocks.len() >= max_blocks {
            return Ok(());
        }
        let mi_width = size / 8;
        let half = mi_width / 2;
        let bit = 1u8 << (3 - mi_width.trailing_zeros());
        let above = self.above_partition[col..(col + mi_width).min(self.mi_cols)]
            .iter().fold(0, |mask, value| mask | value) & bit != 0;
        let left = self.left_partition[row..(row + mi_width).min(self.mi_rows)]
            .iter().fold(0, |mask, value| mask | value) & bit != 0;
        let context = mi_width.trailing_zeros() as usize * 4 + usize::from(left) * 2 + usize::from(above);
        let has_rows = row + half < self.mi_rows;
        let has_cols = col + half < self.mi_cols;
        let partition = if self.inter.is_some() {
            read_partition_with_probabilities(&mut self.bits,
                self.compressed.inter_probs.partition[context], has_rows, has_cols)?
        } else {
            read_partition(&mut self.bits, context, has_rows, has_cols)?
        };
        if self.inter.is_some() {
            self.noncoef_counts.partition[context][partition as usize] += 1;
        }
        let (block_width, block_height) = match partition {
            Partition::None => (size, size),
            Partition::Horizontal => (size, size / 2),
            Partition::Vertical => (size / 2, size),
            Partition::Split => (size / 2, size / 2),
        };
        match partition {
            Partition::None => self.decode_block(row, col, block_width, block_height, max_blocks)?,
            Partition::Horizontal | Partition::Vertical => {
                self.decode_block(row, col, block_width, block_height, max_blocks)?;
                if size > 8 {
                    let next_row = row + if partition == Partition::Horizontal { half } else { 0 };
                    let next_col = col + if partition == Partition::Vertical { half } else { 0 };
                    if next_row < self.mi_rows && next_col < self.mi_cols {
                        self.decode_block(next_row, next_col, block_width, block_height, max_blocks)?;
                    }
                }
            }
            Partition::Split if size == 8 => self.decode_block(row, col, 4, 4, max_blocks)?,
            Partition::Split => {
                for (dy, dx) in [(0, 0), (0, half), (half, 0), (half, half)] {
                    self.decode_partition(row + dy, col + dx, size / 2, max_blocks)?;
                }
            }
        }
        if size == 8 || partition != Partition::Split {
            let width_log2 = (block_width / 4).trailing_zeros();
            let height_log2 = (block_height / 4).trailing_zeros();
            let above_mask = 15 >> width_log2;
            let left_mask = 15 >> height_log2;
            self.above_partition[col..(col + mi_width).min(self.mi_cols)].fill(above_mask);
            self.left_partition[row..(row + mi_width).min(self.mi_rows)].fill(left_mask);
        }
        Ok(())
    }

    fn decode_block(
        &mut self,
        row: usize,
        col: usize,
        width: usize,
        height: usize,
        max_blocks: usize,
    ) -> Result<(), MediaDecodeError> {
        if self.blocks.len() >= max_blocks {
            return Ok(());
        }
        if let Some(frame) = self.inter {
            return self.decode_inter_block(frame, row, col, width, height);
        }
        let size = width.max(height).max(8);
        if !matches!(size, 8 | 16 | 32 | 64) {
            return Err(MediaDecodeError::Unsupported);
        }
        let segment_id = if self.layout.segmentation_enabled && self.layout.segmentation_update_map {
            read_tree(&mut self.bits, &SEGMENT_TREE, &self.layout.segment_tree_probs)?
        } else {
            0
        };
        let above_index = (row > 0).then(|| (row - 1) * self.mi_cols + col);
        let left_index = (col > self.col_start).then(|| row * self.mi_cols + col - 1);
        let skip_context = above_index.map_or(0, |index| usize::from(self.skips[index]))
            + left_index.map_or(0, |index| usize::from(self.skips[index]));
        let skip = self.layout.segment_skip[segment_id as usize]
            || self.bits.read(self.compressed.skip_probs[skip_context])?;
        let tx_size = self.read_block_tx_size(row, col, width, height, skip, false)?;
        let mut sub_modes = [0u8; 4];
        let mut is_sub8 = false;
        let y_mode = if width < 8 || height < 8 {
            is_sub8 = true;
            let step_x = width / 4;
            let step_y = height / 4;
            let mut last = 0;
            for sub_y in (0..2).step_by(step_y) {
                for sub_x in (0..2).step_by(step_x) {
                    let above_mode = if sub_y != 0 { sub_modes[sub_x] } else {
                        above_index.map_or(0, |index| self.sub_modes[index][2 + sub_x])
                    } as usize;
                    let left_mode = if sub_x != 0 { sub_modes[sub_y * 2] } else {
                        left_index.map_or(0, |index| self.sub_modes[index][1 + sub_y * 2])
                    } as usize;
                    last = read_tree(&mut self.bits, &INTRA_MODE_TREE,
                        &KEYFRAME_LUMA_MODE_PROBS[above_mode][left_mode])?;
                    for y in sub_y..sub_y + step_y {
                        for x in sub_x..sub_x + step_x {
                            sub_modes[y * 2 + x] = last;
                        }
                    }
                }
            }
            last
        } else {
            let above_mode = above_index.map_or(0, |index| self.sub_modes[index][2]) as usize;
            let left_mode = left_index.map_or(0, |index| self.sub_modes[index][1]) as usize;
            let mode = read_tree(&mut self.bits, &INTRA_MODE_TREE,
                &KEYFRAME_LUMA_MODE_PROBS[above_mode][left_mode])?;
            sub_modes.fill(mode);
            mode
        };
        let uv_mode = read_tree(&mut self.bits, &INTRA_MODE_TREE, &KEYFRAME_UV_MODE_PROBS[y_mode as usize])?;
        let mut luma_coefficients = Vec::new();
        let mut chroma_coefficients = [Vec::new(), Vec::new()];
        let mut luma_nonzero = [false; 4];
        let mut chroma_nonzero = [false; 2];
        let mut first_token = None;
        let mut first_coefficient = None;
        let mut first_eob = 0;
        for plane in 0..3 {
            let plane_width = if plane == 0 { width.max(8) } else { width.max(8) / 2 };
            let plane_height = if plane == 0 { height.max(8) } else { height.max(8) / 2 };
            let plane_tx = tx_size.min((plane_width.min(plane_height) / 4).trailing_zeros() as u8);
            let transform_size = 4usize << plane_tx;
            let transforms_wide = plane_width / transform_size;
            let transforms_high = plane_height / transform_size;
            for block_y in 0..transforms_high {
                for block_x in 0..transforms_wide {
                    let x4 = (col * 8 / if plane == 0 { 1 } else { 2 } + block_x * transform_size) / 4;
                    let y4 = (row * 8 / if plane == 0 { 1 } else { 2 } + block_y * transform_size) / 4;
                    let max_x4 = self.mi_cols * if plane == 0 { 2 } else { 1 };
                    let max_y4 = self.mi_rows * if plane == 0 { 2 } else { 1 };
                    let span = 1usize << plane_tx;
                    let on_screen = x4 < max_x4 && y4 < max_y4;
                    let above = on_screen && self.above_nonzero[plane][x4..(x4 + span).min(self.above_nonzero[plane].len())]
                        .iter().any(|&value| value);
                    let left = on_screen && self.left_nonzero[plane][y4..(y4 + span).min(self.left_nonzero[plane].len())]
                        .iter().any(|&value| value);
                    let transform = if skip || !on_screen {
                        ParsedTransform { coefficients: vec![0; transform_size * transform_size], first_token: None,
                            first_coefficient: None, eob: 0 }
                    } else {
                        let mode = if plane == 0 && is_sub8 {
                            sub_modes[block_y * 2 + block_x]
                        } else { y_mode };
                        read_transform_with_reference(&mut self.bits, self.compressed, plane_tx, 8,
                            usize::from(plane != 0), 0, usize::from(above) + usize::from(left),
                            if self.layout.lossless { 0 } else { mode },
                            Some(&mut self.coef_counts))?
                    };
                    let nonzero = transform.eob != 0;
                    let above_end = (x4 + span).min(self.above_nonzero[plane].len());
                    let left_end = (y4 + span).min(self.left_nonzero[plane].len());
                    if x4 < above_end {
                        self.above_nonzero[plane][x4..above_end].fill(nonzero);
                    }
                    if y4 < left_end {
                        self.left_nonzero[plane][y4..left_end].fill(nonzero);
                    }
                    if plane == 0 {
                        let index = block_y * transforms_wide + block_x;
                        if let Some(value) = luma_nonzero.get_mut(index) {
                            *value = nonzero;
                        }
                        if index == 0 {
                            first_token = transform.first_token;
                            first_coefficient = transform.first_coefficient;
                            first_eob = transform.eob;
                        }
                        luma_coefficients.push(transform.coefficients);
                    } else {
                        chroma_nonzero[plane - 1] |= nonzero;
                        chroma_coefficients[plane - 1].extend(transform.coefficients);
                    }
                }
            }
        }
        for y in row..(row + height.max(8) / 8).min(self.mi_rows) {
            for x in col..(col + width.max(8) / 8).min(self.mi_cols) {
                let index = y * self.mi_cols + x;
                self.sub_modes[index] = sub_modes;
                self.skips[index] = skip;
                self.tx_sizes[index] = tx_size;
            }
        }
        self.blocks.push(TileBlock {
            x: col * 8, y: row * 8, width, height,
            block: FirstBlock {
                partition: Partition::None, block_size: size, segment_id, skip, tx_size, y_mode, uv_mode,
                sub_modes: is_sub8.then_some(sub_modes),
                first_luma_token: first_token, first_luma_coefficient: first_coefficient,
                luma_coefficients: Some(luma_coefficients), chroma_coefficients: Some(chroma_coefficients),
                first_luma_eob: first_eob, luma_nonzero, chroma_nonzero, inter: None,
            },
        });
        Ok(())
    }

    fn read_block_tx_size(
        &mut self, row: usize, col: usize, width: usize, height: usize,
        skip: bool, is_inter: bool,
    ) -> Result<u8, MediaDecodeError> {
        if self.layout.lossless || width < 8 || height < 8 {
            return Ok(0);
        }
        let maximum = ((width.min(height) / 4).trailing_zeros() as u8).min(3);
        if self.compressed.tx_mode != 4 {
            return Ok(self.compressed.tx_mode.min(maximum));
        }
        if skip && is_inter {
            return Ok(maximum);
        }
        let above_index = (row > 0).then(|| (row - 1) * self.mi_cols + col);
        let left_index = (col > self.col_start).then(|| row * self.mi_cols + col - 1);
        let mut above = above_index.filter(|&index| !self.skips[index])
            .map_or(maximum, |index| self.tx_sizes[index]);
        let mut left = left_index.filter(|&index| !self.skips[index])
            .map_or(maximum, |index| self.tx_sizes[index]);
        if above_index.is_none() { above = left; }
        if left_index.is_none() { left = above; }
        let context = usize::from(above + left > maximum);
        let probabilities: &[u8] = match maximum {
            0 => return Ok(0),
            1 => &self.compressed.inter_probs.tx_8x8[context],
            2 => &self.compressed.inter_probs.tx_16x16[context],
            _ => &self.compressed.inter_probs.tx_32x32[context],
        };
        for (size, &probability) in probabilities.iter().enumerate() {
            if !self.bits.read(probability)? {
                if self.inter.is_some() {
                    self.noncoef_counts.tx_size[maximum as usize][context][size] += 1;
                }
                return Ok(size as u8);
            }
        }
        if self.inter.is_some() {
            self.noncoef_counts.tx_size[maximum as usize][context][maximum as usize] += 1;
        }
        Ok(maximum)
    }

    fn read_inter_segment_id(
        &mut self, row: usize, col: usize, width: usize, height: usize,
    ) -> Result<u8, MediaDecodeError> {
        if !self.layout.segmentation_enabled { return Ok(0); }
        let rows = (row + height.max(8) / 8).min(self.mi_rows);
        let cols = (col + width.max(8) / 8).min(self.mi_cols);
        let predicted = self.previous_segments
            .filter(|segments| segments.len() == self.mi_rows * self.mi_cols)
            .map_or(0, |segments| {
                let mut minimum = 7;
                for y in row..rows {
                    for x in col..cols {
                        minimum = minimum.min(segments[y * self.mi_cols + x]);
                    }
                }
                minimum
            });
        if !self.layout.segmentation_update_map { return Ok(predicted); }
        if self.layout.segmentation_temporal_update {
            let context = usize::from(self.above_seg_pred[col])
                + usize::from(self.left_seg_pred[row]);
            let use_previous = self.bits.read(self.layout.segment_pred_probs[context])?;
            self.above_seg_pred[col..cols].fill(use_previous);
            self.left_seg_pred[row..rows].fill(use_previous);
            if use_previous { return Ok(predicted); }
        }
        read_tree(&mut self.bits, &SEGMENT_TREE, &self.layout.segment_tree_probs)
    }

    fn decode_inter_block(
        &mut self,
        frame: &InterframeHeader,
        row: usize,
        col: usize,
        width: usize,
        height: usize,
    ) -> Result<(), MediaDecodeError> {
        let segment_id = self.read_inter_segment_id(row, col, width, height)?;
        let above_index = (row > 0).then(|| (row - 1) * self.mi_cols + col);
        let left_index = (col > self.col_start).then(|| row * self.mi_cols + col - 1);
        let above = above_index.and_then(|index| self.inter_predictions[index]);
        let left = left_index.and_then(|index| self.inter_predictions[index]);
        let skip_context = above_index.map_or(0, |index| usize::from(self.skips[index]))
            + left_index.map_or(0, |index| usize::from(self.skips[index]));
        let skip = if self.layout.segment_skip[segment_id as usize] { true } else {
            let skip = self.bits.read(self.compressed.skip_probs[skip_context])?;
            self.noncoef_counts.skip[skip_context][usize::from(skip)] += 1;
            skip
        };
        let is_inter_context = match (above_index, left_index) {
            (Some(_), Some(_)) => if above.is_none() && left.is_none() { 3 }
                else { usize::from(above.is_none() || left.is_none()) },
            (Some(_), None) => 2 * usize::from(above.is_none()),
            (None, Some(_)) => 2 * usize::from(left.is_none()),
            (None, None) => 0,
        };
        let is_inter = match self.layout.segment_ref[segment_id as usize] {
            Some(reference) => reference != 0,
            None => {
                let value = self.bits.read(self.compressed.inter_probs.is_inter[is_inter_context])?;
                self.noncoef_counts.is_inter[is_inter_context][usize::from(value)] += 1;
                value
            }
        };
        let tx_size = self.read_block_tx_size(row, col, width, height, skip, is_inter)?;
        if !is_inter {
            let group = match width.min(height) {
                0..=4 => 0,
                5..=8 => 1,
                9..=16 => 2,
                _ => 3,
            };
            let mut sub_modes = [0u8; 4];
            let y_mode = if width < 8 || height < 8 {
                let step_x = width / 4;
                let step_y = height / 4;
                let mut last = 0;
                for sub_y in (0..2).step_by(step_y) {
                    for sub_x in (0..2).step_by(step_x) {
                        last = read_tree(&mut self.bits, &INTRA_MODE_TREE,
                            &self.compressed.inter_probs.y_mode[0])?;
                        self.noncoef_counts.y_mode[0][last as usize] += 1;
                        for y in sub_y..sub_y + step_y {
                            for x in sub_x..sub_x + step_x {
                                sub_modes[y * 2 + x] = last;
                            }
                        }
                    }
                }
                last
            } else {
                let mode = read_tree(&mut self.bits, &INTRA_MODE_TREE,
                    &self.compressed.inter_probs.y_mode[group])?;
                self.noncoef_counts.y_mode[group][mode as usize] += 1;
                mode
            };
            let uv_mode = read_tree(&mut self.bits, &INTRA_MODE_TREE,
                &self.compressed.inter_probs.uv_mode[y_mode as usize])?;
            self.noncoef_counts.uv_mode[y_mode as usize][uv_mode as usize] += 1;
            let (luma, chroma, luma_nonzero, chroma_nonzero, _) =
                self.read_inter_residual(row, col, width, height, tx_size, skip, false,
                    y_mode, (width < 8 || height < 8).then_some(sub_modes))?;
            for y in row..(row + height.max(8) / 8).min(self.mi_rows) {
                for x in col..(col + width.max(8) / 8).min(self.mi_cols) {
                    self.skips[y * self.mi_cols + x] = skip;
                    self.tx_sizes[y * self.mi_cols + x] = tx_size;
                }
            }
            self.blocks.push(TileBlock {
                x: col * 8, y: row * 8, width, height,
                block: FirstBlock {
                    partition: Partition::None, block_size: width.max(height), segment_id,
                    skip, tx_size, y_mode, uv_mode,
                    sub_modes: (width < 8 || height < 8).then_some(sub_modes),
                    first_luma_token: None, first_luma_coefficient: None,
                    luma_coefficients: Some(luma), chroma_coefficients: Some(chroma),
                    first_luma_eob: 0, luma_nonzero, chroma_nonzero, inter: None,
                },
            });
            return Ok(());
        }
        let (single_ref_p1, single_ref_p2) = single_reference_contexts(
            above_index.is_some(), left_index.is_some(), above, left,
        );
        let (fixed, variable) = compound_references(frame);
        let fixed_index = usize::from(frame.reference_sign_bias[(fixed - 1) as usize]);
        let forced_reference = self.layout.segment_ref[segment_id as usize];
        let compound = if forced_reference.is_some() { false } else {
            match self.compressed.reference_mode {
                0 => false,
                1 => true,
                2 => {
                    let context = compound_mode_context(above_index.is_some(), left_index.is_some(),
                        above, left, fixed);
                    let value = self.bits.read(self.compressed.inter_probs.comp_mode[context])?;
                    self.noncoef_counts.comp_mode[context][usize::from(value)] += 1;
                    value
                },
                _ => return Err(MediaDecodeError::Unsupported),
            }
        };
        let (reference, second_reference) = if compound {
            let context = compound_ref_context(above_index.is_some(), left_index.is_some(),
                above, left, fixed, variable, fixed_index);
            let choice = self.bits.read(self.compressed.inter_probs.comp_ref[context])?;
            self.noncoef_counts.comp_ref[context][usize::from(choice)] += 1;
            let variable_ref = variable[usize::from(choice)];
            if fixed_index == 0 { (fixed, Some(variable_ref)) }
            else { (variable_ref, Some(fixed)) }
        } else if let Some(reference) = forced_reference {
            (reference, None)
        } else {
            let first = self.bits.read(self.compressed.inter_probs.single_ref[single_ref_p1][0])?;
            self.noncoef_counts.single_ref[single_ref_p1][0][usize::from(first)] += 1;
            if first {
                let second = self.bits.read(self.compressed.inter_probs.single_ref[single_ref_p2][1])?;
                self.noncoef_counts.single_ref[single_ref_p2][1][usize::from(second)] += 1;
                (if second { 3 } else { 2 }, None)
            } else { (1, None) }
        };
        let neighbor_mode = |prediction: InterPrediction| match prediction.mode {
            0 | 1 => 0usize, 2 => 3, 3 => 1, _ => 9,
        };
        let size_index = block_size_index(width, height).ok_or(MediaDecodeError::Unsupported)?;
        let candidates: [(bool, Option<InterPrediction>); 8] = MV_REF_BLOCKS[size_index].map(|(dr, dc)| {
            let candidate_row = row as isize + dr;
            let candidate_col = col as isize + dc;
            if candidate_row < 0 || candidate_row >= self.mi_rows as isize
                || candidate_col < self.col_start as isize || candidate_col >= self.col_end as isize {
                (false, None)
            } else {
                (true, self.inter_predictions[candidate_row as usize * self.mi_cols + candidate_col as usize])
            }
        });
        let context_counter = candidates[..2].iter().map(|&(available, prediction)| {
            if available { prediction.map_or(9, neighbor_mode) } else { 0 }
        }).sum::<usize>();
        let inter_mode_context = [2, 3, 4, 1, 3, 5, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6]
            [context_counter.min(18)];
        let sub8 = width < 8 || height < 8;
        let mut mode = if sub8 || self.layout.segment_skip[segment_id as usize] { 2 } else {
            let value = read_tree(&mut self.bits, &INTER_MODE_TREE,
                &self.compressed.inter_probs.inter_mode[inter_mode_context])?;
            self.noncoef_counts.inter_mode[inter_mode_context][value as usize] += 1;
            value
        };
        let above_filter = above.map_or(3, |neighbor| neighbor.interpolation_filter);
        let left_filter = left.map_or(3, |neighbor| neighbor.interpolation_filter);
        let filter_context = if above_filter == left_filter { above_filter }
            else if above_filter == 3 { left_filter }
            else if left_filter == 3 { above_filter }
            else { 3 } as usize;
        let interpolation_filter = if frame.interpolation_filter == 4 {
            let value = read_tree(&mut self.bits, &INTERP_FILTER_TREE,
                &self.compressed.inter_probs.interp_filter[filter_context])?;
            self.noncoef_counts.interp_filter[filter_context][value as usize] += 1;
            value
        } else { frame.interpolation_filter };
        let references = [reference, second_reference.unwrap_or(0)];
        let mut best = [[(0, 0); 2]; 2];
        for list in 0..(1 + usize::from(second_reference.is_some())) {
            best[list] = self.motion_candidates(&candidates, frame, references[list],
                row, col, width, height, None);
        }
        let mut sub_motions = [None; 2];
        let mut motions = [(0, 0); 2];
        if sub8 {
            let mut values = [[(0, 0); 4]; 2];
            let step_x = width / 4;
            let step_y = height / 4;
            for sub_y in (0..2).step_by(step_y) {
                for sub_x in (0..2).step_by(step_x) {
                    mode = if self.layout.segment_skip[segment_id as usize] { 2 } else {
                        let value = read_tree(&mut self.bits, &INTER_MODE_TREE,
                            &self.compressed.inter_probs.inter_mode[inter_mode_context])?;
                        self.noncoef_counts.inter_mode[inter_mode_context][value as usize] += 1;
                        value
                    };
                    let block_index = sub_y * 2 + sub_x;
                    for list in 0..(1 + usize::from(second_reference.is_some())) {
                        let refs = self.motion_candidates(&candidates, frame, references[list],
                            row, col, width, height, Some(block_index));
                        let mut sub_refs = [(0, 0); 2];
                        let mut count = 0;
                        if block_index == 1 || block_index == 2 {
                            push_unique_motion(&mut sub_refs, &mut count, values[list][0]);
                        } else if block_index == 3 {
                            push_unique_motion(&mut sub_refs, &mut count, values[list][2]);
                            push_unique_motion(&mut sub_refs, &mut count, values[list][1]);
                            push_unique_motion(&mut sub_refs, &mut count, values[list][0]);
                        }
                        for candidate in refs {
                            push_unique_motion(&mut sub_refs, &mut count, candidate);
                        }
                        let motion = read_inter_motion(&mut self.bits, self.compressed,
                            frame.allow_high_precision_mv, mode, sub_refs[0], sub_refs[1], best[list][0],
                            Some(&mut self.noncoef_counts))?;
                        for y in sub_y..sub_y + step_y {
                            for x in sub_x..sub_x + step_x {
                                values[list][y * 2 + x] = motion;
                            }
                        }
                    }
                }
            }
            for list in 0..(1 + usize::from(second_reference.is_some())) {
                sub_motions[list] = Some(values[list]);
                motions[list] = values[list][3];
            }
        } else {
            for list in 0..(1 + usize::from(second_reference.is_some())) {
                motions[list] = read_inter_motion(&mut self.bits, self.compressed,
                    frame.allow_high_precision_mv, mode, best[list][0], best[list][1], best[list][0],
                    Some(&mut self.noncoef_counts))?;
            }
        }
        let (luma, chroma, luma_nonzero, chroma_nonzero, any_eob) =
            self.read_inter_residual(row, col, width, height, tx_size, skip, true, 0, None)?;
        let skip = skip || (!sub8 && !any_eob);
        let prediction = InterPrediction {
            reference, mode, interpolation_filter, motion: motions[0], sub_motions: sub_motions[0],
            second: second_reference.map(|reference| SecondReference {
                reference, motion: motions[1], sub_motions: sub_motions[1],
            }),
        };
        for y in row..(row + height.max(8) / 8).min(self.mi_rows) {
            for x in col..(col + width.max(8) / 8).min(self.mi_cols) {
                self.skips[y * self.mi_cols + x] = skip;
                self.tx_sizes[y * self.mi_cols + x] = tx_size;
                self.inter_predictions[y * self.mi_cols + x] = Some(prediction);
            }
        }
        self.blocks.push(TileBlock {
            x: col * 8, y: row * 8, width, height,
            block: FirstBlock {
                partition: Partition::None, block_size: width.max(height), segment_id,
                skip, tx_size, y_mode: 0, uv_mode: 0, sub_modes: None,
                first_luma_token: None, first_luma_coefficient: None,
                luma_coefficients: Some(luma), chroma_coefficients: Some(chroma),
                first_luma_eob: 0, luma_nonzero, chroma_nonzero,
                inter: Some(prediction),
            },
        });
        Ok(())
    }

    fn motion_candidates(
        &self, candidates: &[(bool, Option<InterPrediction>); 8],
        frame: &InterframeHeader, reference: u8, row: usize, col: usize,
        width: usize, height: usize, block: Option<usize>,
    ) -> [(i32, i32); 2] {
        const SUBBLOCK_NEIGHBOR: [[usize; 2]; 4] = [[1, 2], [1, 3], [3, 2], [3, 3]];
        let mut result = [(0, 0); 2];
        let mut count = 0;
        let candidate_motion = |index: usize, prediction: InterPrediction,
                                second: bool, use_subblock: bool| {
            let subblock = if use_subblock && index < 2 {
                block.map(|block| SUBBLOCK_NEIGHBOR[block][usize::from(index == 0)])
                    .unwrap_or(3)
            } else { 3 };
            if second {
                prediction.second.map_or((0, 0), |value| value.sub_motions
                    .map_or(value.motion, |motions| motions[subblock]))
            } else {
                prediction.sub_motions.map_or(prediction.motion, |motions| motions[subblock])
            }
        };
        for (index, (_, prediction)) in candidates.iter().enumerate() {
            if let Some(prediction) = prediction {
                if prediction.reference == reference {
                    push_unique_motion(&mut result, &mut count, candidate_motion(index, *prediction, false, true));
                } else if prediction.second.is_some_and(|value| value.reference == reference) {
                    push_unique_motion(&mut result, &mut count, candidate_motion(index, *prediction, true, true));
                }
            }
        }
        let previous = self.previous_predictions
            .and_then(|predictions| predictions.get(row * self.mi_cols + col))
            .copied().flatten();
        if let Some(prediction) = previous {
            if prediction.reference == reference {
                push_unique_motion(&mut result, &mut count, prediction.motion);
            } else if let Some(second) = prediction.second.filter(|value| value.reference == reference) {
                push_unique_motion(&mut result, &mut count, second.motion);
            }
        }
        if candidates[..2].iter().any(|candidate| candidate.0) {
            for (index, (_, prediction)) in candidates.iter().enumerate() {
                if let Some(prediction) = prediction {
                    for (candidate_ref, second) in [(prediction.reference, false),
                        (prediction.second.map_or(0, |value| value.reference), true)] {
                        if candidate_ref == 0 || candidate_ref == reference { continue; }
                        if second && candidate_motion(index, *prediction, false, false)
                            == candidate_motion(index, *prediction, true, false) { continue; }
                        let mut motion = candidate_motion(index, *prediction, second, false);
                        if frame.reference_sign_bias[(candidate_ref - 1) as usize]
                            != frame.reference_sign_bias[(reference - 1) as usize] {
                            motion = (-motion.0, -motion.1);
                        }
                        push_unique_motion(&mut result, &mut count, motion);
                    }
                }
            }
        }
        if let Some(prediction) = previous {
            for (candidate_ref, mut motion) in [(prediction.reference, prediction.motion),
                prediction.second.map_or((0, (0, 0)), |value| (value.reference, value.motion))] {
                if candidate_ref == 0 || candidate_ref == reference { continue; }
                if candidate_ref != prediction.reference && prediction.second.is_some()
                    && prediction.motion == motion { continue; }
                if frame.reference_sign_bias[(candidate_ref - 1) as usize]
                    != frame.reference_sign_bias[(reference - 1) as usize] {
                    motion = (-motion.0, -motion.1);
                }
                push_unique_motion(&mut result, &mut count, motion);
            }
        }
        let block_rows = height.max(8) / 8;
        let block_cols = width.max(8) / 8;
        let top = -(row as i32 * 64);
        let bottom = (self.mi_rows as i32 - block_rows as i32 - row as i32) * 64;
        let left = -(col as i32 * 64);
        let right = (self.mi_cols as i32 - block_cols as i32 - col as i32) * 64;
        for motion in &mut result {
            motion.0 = motion.0.clamp(top - 128, bottom + 128);
            motion.1 = motion.1.clamp(left - 128, right + 128);
            // append_sub8x8_mvs uses find_mv_refs directly, without find_best_ref_mvs.
            if block.is_none() && (!frame.allow_high_precision_mv || !use_high_precision(*motion)) {
                for component in [&mut motion.0, &mut motion.1] {
                    if *component & 1 != 0 {
                        *component += if *component > 0 { -1 } else { 1 };
                    }
                }
            }
            if block.is_none() {
                motion.0 = motion.0.clamp(top - 1248, bottom + 1248);
                motion.1 = motion.1.clamp(left - 1248, right + 1248);
            }
        }
        result
    }

    fn read_inter_residual(
        &mut self, row: usize, col: usize, width: usize, height: usize,
        tx_size: u8, skip: bool, is_inter: bool, y_mode: u8,
        sub_modes: Option<[u8; 4]>,
    ) -> Result<(Vec<Vec<i32>>, [Vec<i32>; 2], [bool; 4], [bool; 2], bool), MediaDecodeError> {
        if is_inter && skip {
            for plane in 0..3 {
                let scale = if plane == 0 { 1 } else { 2 };
                let x4 = col * 2 / scale;
                let y4 = row * 2 / scale;
                let above_end = (x4 + width.max(8) / (4 * scale)).min(self.above_nonzero[plane].len());
                let left_end = (y4 + height.max(8) / (4 * scale)).min(self.left_nonzero[plane].len());
                if x4 < above_end { self.above_nonzero[plane][x4..above_end].fill(false); }
                if y4 < left_end { self.left_nonzero[plane][y4..left_end].fill(false); }
            }
            return Ok((Vec::new(), [Vec::new(), Vec::new()], [false; 4], [false; 2], false));
        }
        let mut luma = Vec::new();
        let mut chroma = [Vec::new(), Vec::new()];
        let mut luma_nonzero = [false; 4];
        let mut chroma_nonzero = [false; 2];
        let mut any_eob = false;
        for plane in 0..3 {
            let plane_width = if plane == 0 { width.max(8) } else { width.max(8) / 2 };
            let plane_height = if plane == 0 { height.max(8) } else { height.max(8) / 2 };
            let plane_tx = tx_size.min((plane_width.min(plane_height) / 4).trailing_zeros() as u8);
            let transform_size = 4usize << plane_tx;
            let transforms_wide = plane_width / transform_size;
            let transforms_high = plane_height / transform_size;
            for block_y in 0..transforms_high {
                for block_x in 0..transforms_wide {
                    let x4 = (col * 8 / if plane == 0 { 1 } else { 2 }
                        + block_x * transform_size) / 4;
                    let y4 = (row * 8 / if plane == 0 { 1 } else { 2 }
                        + block_y * transform_size) / 4;
                    let max_x4 = self.mi_cols * if plane == 0 { 2 } else { 1 };
                    let max_y4 = self.mi_rows * if plane == 0 { 2 } else { 1 };
                    let span = 1usize << plane_tx;
                    let on_screen = x4 < max_x4 && y4 < max_y4;
                    let above = on_screen && self.above_nonzero[plane]
                        [x4..(x4 + span).min(self.above_nonzero[plane].len())]
                        .iter().any(|&value| value);
                    let left = on_screen && self.left_nonzero[plane]
                        [y4..(y4 + span).min(self.left_nonzero[plane].len())]
                        .iter().any(|&value| value);
                    let transform = if skip || !on_screen {
                        ParsedTransform { coefficients: vec![0; transform_size * transform_size],
                            first_token: None, first_coefficient: None, eob: 0 }
                    } else {
                        let mode = if plane == 0 {
                            sub_modes.map_or(y_mode, |modes| modes[block_y * 2 + block_x])
                        } else { y_mode };
                        read_transform_with_reference(&mut self.bits, self.compressed, plane_tx, 8,
                            usize::from(plane != 0), usize::from(is_inter),
                            usize::from(above) + usize::from(left),
                            if self.layout.lossless { 0 } else { mode }, Some(&mut self.coef_counts))?
                    };
                    let nonzero = transform.eob != 0;
                    any_eob |= nonzero;
                    let above_end = (x4 + span).min(self.above_nonzero[plane].len());
                    let left_end = (y4 + span).min(self.left_nonzero[plane].len());
                    if x4 < above_end {
                        self.above_nonzero[plane][x4..above_end].fill(nonzero);
                    }
                    if y4 < left_end {
                        self.left_nonzero[plane][y4..left_end].fill(nonzero);
                    }
                    if plane == 0 {
                        let index = block_y * transforms_wide + block_x;
                        if let Some(value) = luma_nonzero.get_mut(index) {
                            *value = nonzero;
                        }
                        luma.push(transform.coefficients);
                    } else {
                        chroma_nonzero[plane - 1] |= nonzero;
                        chroma[plane - 1].extend(transform.coefficients);
                    }
                }
            }
        }
        Ok((luma, chroma, luma_nonzero, chroma_nonzero, any_eob))
    }
}

fn push_unique_motion(
    motions: &mut [(i32, i32); 2], count: &mut usize, candidate: (i32, i32),
) {
    if *count < 2 && (*count == 0 || motions[0] != candidate) {
        motions[*count] = candidate;
        *count += 1;
    }
}

fn read_inter_motion(
    bits: &mut BoolDecoder<'_>, compressed: &CompressedHeader, high_precision: bool,
    mode: u8, nearest: (i32, i32), near: (i32, i32), best: (i32, i32),
    counts: Option<&mut NonCoefficientCounts>,
) -> Result<(i32, i32), MediaDecodeError> {
    match mode {
        0 => Ok(nearest),
        1 => Ok(near),
        2 => Ok((0, 0)),
        3 => {
            let high_precision = high_precision && use_high_precision(best);
            let delta = read_motion_difference_counted(bits, &compressed.inter_probs, high_precision, counts)?;
            Ok((best.0 + delta.0, best.1 + delta.1))
        }
        _ => Err(MediaDecodeError::Unsupported),
    }
}

pub fn first_block(
    layout: &KeyframeLayout<'_>,
    compressed: &CompressedHeader,
    tile: &[u8],
) -> Result<Option<FirstBlock>, MediaDecodeError> {
    let mut bits = BoolDecoder::new(tile)?;
    if bits.read_bit()? {
        return Err(MediaDecodeError::InvalidData("invalid VP9 tile marker".into()));
    }
    let partition = read_partition(&mut bits, 12, true, true)?;
    if partition != Partition::None {
        return Ok(None);
    }
    read_block_64x64(&mut bits, layout, compressed, None, None).map(Some)
}

pub fn first_tile_row_prefix(
    layout: &KeyframeLayout<'_>,
    compressed: &CompressedHeader,
    tile: &[u8],
    max_blocks: usize,
) -> Result<(Vec<FirstBlock>, Option<Partition>), MediaDecodeError> {
    let (blocks, next_partition, _) = read_first_tile_row_prefix(layout, compressed, tile, max_blocks)?;
    Ok((blocks, next_partition))
}

fn read_first_tile_row_prefix<'a>(
    layout: &KeyframeLayout<'_>,
    compressed: &CompressedHeader,
    tile: &'a [u8],
    max_blocks: usize,
) -> Result<(Vec<FirstBlock>, Option<Partition>, BoolDecoder<'a>), MediaDecodeError> {
    let mut bits = BoolDecoder::new(tile)?;
    if bits.read_bit()? {
        return Err(MediaDecodeError::InvalidData("invalid VP9 tile marker".into()));
    }
    let mut blocks = Vec::new();
    for _ in 0..max_blocks.min(1024) {
        let partition = read_partition(&mut bits, 12, true, true)?;
        if partition != Partition::None {
            return Ok((blocks, Some(partition), bits));
        }
        let block = read_block_64x64(&mut bits, layout, compressed, None, blocks.last())?;
        blocks.push(block);
    }
    Ok((blocks, None, bits))
}

pub fn first_split_top_left_path(
    layout: &KeyframeLayout<'_>,
    compressed: &CompressedHeader,
    tile: &[u8],
    max_blocks: usize,
) -> Result<Option<Vec<Partition>>, MediaDecodeError> {
    let (_, next_partition, mut bits) = read_first_tile_row_prefix(layout, compressed, tile, max_blocks)?;
    if next_partition != Some(Partition::Split) {
        return Ok(None);
    }
    let mut path = Vec::with_capacity(3);
    for context in [8, 4, 0] {
        let partition = read_partition(&mut bits, context, true, true)?;
        path.push(partition);
        if partition != Partition::Split {
            break;
        }
    }
    Ok(Some(path))
}

pub fn first_split_top_left_16x16(
    layout: &KeyframeLayout<'_>,
    compressed: &CompressedHeader,
    tile: &[u8],
    max_blocks: usize,
) -> Result<Option<FirstBlock>, MediaDecodeError> {
    let (blocks, next_partition, mut bits) = read_first_tile_row_prefix(layout, compressed, tile, max_blocks)?;
    if next_partition != Some(Partition::Split)
        || read_partition(&mut bits, 8, true, true)? != Partition::Split
        || read_partition(&mut bits, 4, true, true)? != Partition::None
    {
        return Ok(None);
    }
    read_block_16x16(&mut bits, layout, compressed, blocks.last()).map(Some)
}

fn read_block_16x16(
    bits: &mut BoolDecoder<'_>,
    layout: &KeyframeLayout<'_>,
    compressed: &CompressedHeader,
    left: Option<&FirstBlock>,
) -> Result<FirstBlock, MediaDecodeError> {
    let segment_id = if layout.segmentation_enabled && layout.segmentation_update_map {
        read_tree(bits, &SEGMENT_TREE, &layout.segment_tree_probs)?
    } else {
        0
    };
    let skip = if layout.segment_skip[segment_id as usize] {
        true
    } else {
        bits.read(compressed.skip_probs[usize::from(left.is_some_and(|block| block.skip))])?
    };
    let tx_size = if compressed.tx_mode == 4 {
        return Err(MediaDecodeError::InvalidData("VP9 selected transform size not decoded yet".into()));
    } else if layout.lossless {
        0
    } else {
        compressed.tx_mode.min(2)
    };
    if tx_size != 2 {
        return Err(MediaDecodeError::Unsupported);
    }
    let left_mode = left.map_or(0, |block| block.y_mode) as usize;
    let y_mode = read_tree(bits, &INTRA_MODE_TREE, &KEYFRAME_LUMA_MODE_PROBS[0][left_mode])?;
    let uv_mode = read_tree(bits, &INTRA_MODE_TREE, &KEYFRAME_UV_MODE_PROBS[y_mode as usize])?;
    let bit_depth = layout.header.bit_depth.unwrap_or(8);
    let luma_context = usize::from(left.is_some_and(|block| block.luma_nonzero[1]));
    let luma = if skip {
        ParsedTransform { coefficients: vec![0; 256], first_token: None,
            first_coefficient: None, eob: 0 }
    } else {
        read_transform(bits, compressed, tx_size, bit_depth, 0, luma_context)?
    };
    let mut chroma_coefficients = [vec![0; 64], vec![0; 64]];
    let mut chroma_nonzero = [false; 2];
    if !skip {
        for (index, coefficients) in chroma_coefficients.iter_mut().enumerate() {
            let context = usize::from(left.is_some_and(|block| block.chroma_nonzero[index]));
            let transform = read_transform(bits, compressed, 1, bit_depth, 1, context)?;
            chroma_nonzero[index] = transform.eob != 0;
            *coefficients = transform.coefficients;
        }
    }
    Ok(FirstBlock {
        partition: Partition::None, block_size: 16, segment_id, skip, tx_size, y_mode, uv_mode,
        first_luma_token: luma.first_token,
        first_luma_coefficient: luma.first_coefficient,
        first_luma_eob: luma.eob,
        sub_modes: None,
        luma_nonzero: [luma.eob != 0, false, false, false],
        luma_coefficients: Some(vec![luma.coefficients]),
        chroma_coefficients: Some(chroma_coefficients),
        chroma_nonzero,
        inter: None,
    })
}

fn read_block_64x64(
    bits: &mut BoolDecoder<'_>,
    layout: &KeyframeLayout<'_>,
    compressed: &CompressedHeader,
    above: Option<&FirstBlock>,
    left: Option<&FirstBlock>,
) -> Result<FirstBlock, MediaDecodeError> {
    let segment_id = if layout.segmentation_enabled && layout.segmentation_update_map {
        read_tree(bits, &SEGMENT_TREE, &layout.segment_tree_probs)?
    } else {
        0
    };
    let skip = if layout.segment_skip[segment_id as usize] {
        true
    } else {
        let skip_context = usize::from(above.is_some_and(|block| block.skip))
            + usize::from(left.is_some_and(|block| block.skip));
        bits.read(compressed.skip_probs[skip_context])?
    };
    let tx_size = if compressed.tx_mode == 4 {
        return Err(MediaDecodeError::InvalidData("VP9 selected transform size not decoded yet".into()));
    } else if layout.lossless {
        0
    } else {
        compressed.tx_mode.min(3)
    };
    if tx_size != 3 {
        return Err(MediaDecodeError::Unsupported);
    }
    let above_mode = above.map_or(0, |block| block.y_mode) as usize;
    let left_mode = left.map_or(0, |block| block.y_mode) as usize;
    let y_mode = read_tree(bits, &INTRA_MODE_TREE,
        &KEYFRAME_LUMA_MODE_PROBS[above_mode][left_mode])?;
    let uv_mode = read_tree(
        bits,
        &INTRA_MODE_TREE,
        &KEYFRAME_UV_MODE_PROBS[y_mode as usize],
    )?;
    let bit_depth = layout.header.bit_depth.unwrap_or(8);
    let mut luma_coefficients = (tx_size == 3).then(|| Vec::with_capacity(4));
    let mut first_luma_token = None;
    let mut first_luma_coefficient = None;
    let mut first_luma_eob = 0;
    let transforms = if tx_size == 3 { 4 } else { 1 };
    let mut nonzero = [false; 4];
    for index in 0..transforms {
        let above_nonzero = if index >= 2 { nonzero[index - 2] }
            else { above.is_some_and(|block| block.luma_nonzero[index + 2]) };
        let left_nonzero = if index % 2 != 0 { nonzero[index - 1] }
            else { left.is_some_and(|block| block.luma_nonzero[index + 1]) };
        let context = usize::from(above_nonzero) + usize::from(left_nonzero);
        let transform = if skip {
            ParsedTransform {
                coefficients: vec![0; if tx_size == 3 { 1024 } else { 1 }],
                first_token: None,
                first_coefficient: None,
                eob: 0,
            }
        } else {
            read_transform(bits, compressed, tx_size, bit_depth, 0, context)?
        };
        nonzero[index] = transform.eob != 0;
        if index == 0 {
            first_luma_token = transform.first_token;
            first_luma_coefficient = transform.first_coefficient;
            first_luma_eob = transform.eob;
        }
        if let Some(coefficients) = &mut luma_coefficients {
            coefficients.push(transform.coefficients);
        }
    }
    let mut chroma_coefficients = None;
    let mut chroma_nonzero = [false; 2];
    if tx_size == 3 {
        let mut planes = [vec![0; 1024], vec![0; 1024]];
        if !skip {
            for (index, plane) in planes.iter_mut().enumerate() {
                let context = usize::from(above.is_some_and(|block| block.chroma_nonzero[index]))
                    + usize::from(left.is_some_and(|block| block.chroma_nonzero[index]));
                let transform = read_transform(bits, compressed, 3, bit_depth, 1, context)?;
                chroma_nonzero[index] = transform.eob != 0;
                *plane = transform.coefficients;
            }
        }
        chroma_coefficients = Some(planes);
    }
    Ok(FirstBlock {
        partition: Partition::None, block_size: 64, segment_id, skip, tx_size, y_mode, uv_mode,
        first_luma_token, first_luma_coefficient, luma_coefficients,
        chroma_coefficients, first_luma_eob, luma_nonzero: nonzero, chroma_nonzero,
        sub_modes: None,
        inter: None,
    })
}

fn read_transform(
    bits: &mut BoolDecoder<'_>,
    compressed: &CompressedHeader,
    tx_size: u8,
    bit_depth: u8,
    block_type: usize,
    initial_context: usize,
) -> Result<ParsedTransform, MediaDecodeError> {
    read_transform_with_mode(bits, compressed, tx_size, bit_depth, block_type, initial_context, 0)
}

fn read_transform_with_mode(
    bits: &mut BoolDecoder<'_>,
    compressed: &CompressedHeader,
    tx_size: u8,
    bit_depth: u8,
    block_type: usize,
    initial_context: usize,
    mode: u8,
) -> Result<ParsedTransform, MediaDecodeError> {
    read_transform_with_reference(bits, compressed, tx_size, bit_depth, block_type,
        0, initial_context, mode, None)
}

fn read_transform_with_reference(
    bits: &mut BoolDecoder<'_>,
    compressed: &CompressedHeader,
    tx_size: u8,
    bit_depth: u8,
    block_type: usize,
    reference_type: usize,
    initial_context: usize,
    mode: u8,
    mut counts: Option<&mut CoefficientCounts>,
) -> Result<ParsedTransform, MediaDecodeError> {
    let mut token_cache = [0u8; 1024];
    let mut check_eob = true;
    let count = match tx_size { 3 => 1024, 2 => 256, 1 => 64, _ => 16 };
    let mut coefficients = vec![0; count];
    let mut first_token = None;
    let mut first_coefficient = None;
    let mut eob = 0;
    let directional_scan = if reference_type != 0 || block_type != 0 || tx_size == 3 {
        0
    } else {
        match mode {
            1 | 5 | 8 => 1,
            2 | 6 | 7 => 2,
            _ => 0,
        }
    };
    for (coefficient_index, scan) in coefficient_scan(tx_size, directional_scan).iter().enumerate() {
        let position = usize::from(scan.position);
        let context = if coefficient_index == 0 {
            initial_context
        } else {
            ((1 + token_cache[usize::from(scan.first)] + token_cache[usize::from(scan.second)]) >> 1) as usize
        };
        let band = usize::from(scan.band);
        let probabilities = compressed.coef_probs[tx_size as usize][block_type][reference_type][band][context];
        if check_eob {
            let more = bits.read(probabilities[0])?;
            if let Some(counts) = counts.as_deref_mut() {
                counts.record_more(tx_size as usize, block_type, reference_type, band, context, more);
            }
            if !more { break; }
        }
        let token = read_token(bits, probabilities)?;
        if let Some(counts) = counts.as_deref_mut() {
            counts.record_token(tx_size as usize, block_type, reference_type, band, context, token);
        }
        token_cache[position] = [0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5][token as usize];
        let coefficient = if token == 0 {
            check_eob = false;
            0
        } else {
            check_eob = true;
            read_coefficient(bits, token, bit_depth)?
        };
        if coefficient_index == 0 {
            first_token = Some(token);
            first_coefficient = Some(coefficient);
        }
        coefficients[position] = coefficient;
        eob = coefficient_index + 1;
    }
    Ok(ParsedTransform { coefficients, first_token, first_coefficient, eob })
}

fn read_coefficient(bits: &mut BoolDecoder<'_>, token: u8, bit_depth: u8) -> Result<i32, MediaDecodeError> {
    const BASE: [i32; 11] = [0, 1, 2, 3, 4, 5, 7, 11, 19, 35, 67];
    const CAT_PROBS: [&[u8]; 11] = [
        &[], &[], &[], &[], &[], &[159], &[165, 145],
        &[173, 148, 140], &[176, 155, 140, 135],
        &[180, 157, 141, 134, 130],
        &[254, 254, 254, 252, 249, 243, 230, 196, 177, 153, 140, 133, 130, 129],
    ];
    let mut coefficient = BASE[token as usize];
    if token == 10 {
        for extra in 0..bit_depth.saturating_sub(8) {
            coefficient += i32::from(bits.read(255)?) << (5 + bit_depth - extra);
        }
    }
    let probabilities = CAT_PROBS[token as usize];
    for (index, &probability) in probabilities.iter().enumerate() {
        coefficient += i32::from(bits.read(probability)?) << (probabilities.len() - 1 - index);
    }
    Ok(if bits.read_bit()? { -coefficient } else { coefficient })
}

fn read_token(bits: &mut BoolDecoder<'_>, probabilities: [u8; 3]) -> Result<u8, MediaDecodeError> {
    let mut node = 0usize;
    loop {
        let probability = if node < 4 {
            probabilities[1 + node / 2]
        } else {
            let base = probabilities[2];
            let row = (usize::from(base) - 1) / 2;
            let column = node / 2 - 2;
            if base & 1 != 0 {
                PARETO_TABLE[row][column]
            } else {
                ((u16::from(PARETO_TABLE[row][column])
                    + u16::from(PARETO_TABLE[row + 1][column])) >> 1) as u8
            }
        };
        let branch = usize::from(bits.read(probability)?);
        let next = TOKEN_TREE[node + branch];
        if next <= 0 {
            return Ok((-next) as u8);
        }
        node = next as usize;
    }
}

fn read_tree(bits: &mut BoolDecoder<'_>, tree: &[i8], probabilities: &[u8]) -> Result<u8, MediaDecodeError> {
    let mut node = 0usize;
    loop {
        let branch = usize::from(bits.read(probabilities[node / 2])?);
        let next = tree[node + branch];
        if next <= 0 {
            return Ok((-next) as u8);
        }
        node = next as usize;
    }
}

pub fn first_partition(tile: &[u8]) -> Result<Partition, MediaDecodeError> {
    let mut bits = BoolDecoder::new(tile)?;
    if bits.read_bit()? {
        return Err(MediaDecodeError::InvalidData("invalid VP9 tile marker".into()));
    }
    read_partition(&mut bits, 12, true, true)
}

pub fn first_inter_partition(
    tile: &[u8], compressed: &CompressedHeader,
) -> Result<Partition, MediaDecodeError> {
    let mut bits = BoolDecoder::new(tile)?;
    if bits.read_bit()? {
        return Err(MediaDecodeError::InvalidData("invalid VP9 tile marker".into()));
    }
    read_partition_with_probabilities(
        &mut bits, compressed.inter_probs.partition[12], true, true,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FirstInterBlockMode {
    pub skip: bool,
    pub reference: u8,
    pub mode: u8,
    pub interpolation_filter: u8,
    pub motion: (i32, i32),
}

pub fn first_inter_block_mode(
    layout: &KeyframeLayout<'_>,
    frame: &super::vp9::InterframeHeader,
    compressed: &CompressedHeader,
    tile: &[u8],
) -> Result<FirstInterBlockMode, MediaDecodeError> {
    if layout.segmentation_enabled || frame.intra_only || compressed.tx_mode == 4 {
        return Err(MediaDecodeError::Unsupported);
    }
    let mut bits = BoolDecoder::new(tile)?;
    if bits.read_bit()? {
        return Err(MediaDecodeError::InvalidData("invalid VP9 tile marker".into()));
    }
    if read_partition_with_probabilities(&mut bits, compressed.inter_probs.partition[12], true, true)?
        != Partition::None
    {
        return Err(MediaDecodeError::Unsupported);
    }
    let skip = bits.read(compressed.skip_probs[0])?;
    if !bits.read(compressed.inter_probs.is_inter[0])? {
        return Err(MediaDecodeError::Unsupported);
    }
    if compressed.reference_mode != 0 {
        return Err(MediaDecodeError::Unsupported);
    }
    let reference = if bits.read(compressed.inter_probs.single_ref[2][0])? {
        if bits.read(compressed.inter_probs.single_ref[2][1])? { 3 } else { 2 }
    } else {
        1
    };
    let mode = read_tree(&mut bits, &INTER_MODE_TREE, &compressed.inter_probs.inter_mode[2])?;
    let interpolation_filter = if frame.interpolation_filter == 4 {
        read_tree(&mut bits, &INTERP_FILTER_TREE, &compressed.inter_probs.interp_filter[3])?
    } else {
        frame.interpolation_filter
    };
    let motion = if mode == 3 {
        read_motion_difference(&mut bits, &compressed.inter_probs, frame.allow_high_precision_mv)?
    } else {
        (0, 0)
    };
    Ok(FirstInterBlockMode { skip, reference, mode, interpolation_filter, motion })
}

fn read_partition(
    bits: &mut BoolDecoder<'_>,
    context: usize,
    has_rows: bool,
    has_cols: bool,
) -> Result<Partition, MediaDecodeError> {
    read_partition_with_probabilities(bits, KEYFRAME_PARTITION_PROBS[context], has_rows, has_cols)
}

fn read_partition_with_probabilities(
    bits: &mut BoolDecoder<'_>, probabilities: [u8; 3], has_rows: bool, has_cols: bool,
) -> Result<Partition, MediaDecodeError> {
    if has_rows && has_cols {
        if !bits.read(probabilities[0])? {
            Ok(Partition::None)
        } else if !bits.read(probabilities[1])? {
            Ok(Partition::Horizontal)
        } else if !bits.read(probabilities[2])? {
            Ok(Partition::Vertical)
        } else {
            Ok(Partition::Split)
        }
    } else if has_cols {
        Ok(if bits.read(probabilities[1])? { Partition::Split } else { Partition::Horizontal })
    } else if has_rows {
        Ok(if bits.read(probabilities[2])? { Partition::Split } else { Partition::Vertical })
    } else {
        Ok(Partition::Split)
    }
}
