//! Raster-context partition traversal and bounded intra block reconstruction.

use super::coefficients::{CoefficientState, tx_index};
use super::decoder::{DecodedFrame, DecodedPlane};
use super::entropy::{PartitionCdfs, SymbolDecoder};
use super::inter::{InterBlock, MotionCell};
use super::reconstruction::{
    IntraMode, add_residual, inverse_lossless_4x4, predict_cfl, predict_intra,
};
use super::syntax::{Error, IntraFrameHeader, SequenceHeader};
use super::tables;
use super::transform::inverse_transform;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntraBlockInfo {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub skip: bool,
    pub y_mode: usize,
    pub uv_mode: usize,
    pub qindex: u8,
    pub reconstructed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires AV1_PREFIX_OBU and AV1_PREFIX_ORACLE binary fixtures"]
    fn spacewalk_prefix_binary_oracle() {
        let bytes = std::fs::read(std::env::var("AV1_PREFIX_OBU").unwrap()).unwrap();
        let oracle = std::fs::read(std::env::var("AV1_PREFIX_ORACLE").unwrap()).unwrap();
        let mut stream = super::super::ObuStream::new();
        let obus = stream.push(&bytes).unwrap();
        let sequence =
            SequenceHeader::parse(&obus.iter().find(|o| o.kind == 1).unwrap().payload).unwrap();
        let frame = super::super::CodedIntraFrame::parse(
            obus.iter().find(|o| o.kind == 6).unwrap(),
            &sequence,
        )
        .unwrap();
        if let Ok(path) = std::env::var("AV1_PREFIX_ZERO_CDEF") {
            let mut expected = frame.header.clone();
            expected.cdef_strengths.iter_mut().for_each(|s| *s = [0; 4]);
            let mut modified = obus.clone();
            let frame_obu = modified.iter_mut().find(|o| o.kind == 6).unwrap();
            let span = frame.header.cdef_strengths.len() * 12;
            let mut found = false;
            for start in 0..frame.header.header_bytes * 8 - span {
                let mut candidate = frame_obu.payload.clone();
                for bit in start..start + span {
                    candidate[bit / 8] &= !(1 << (7 - bit % 8));
                }
                if IntraFrameHeader::parse(&candidate, &sequence, 0, 0)
                    .ok()
                    .as_ref()
                    == Some(&expected)
                {
                    frame_obu.payload = candidate;
                    found = true;
                    println!("neutralized CDEF strength bits {start}..{}", start + span);
                    break;
                }
            }
            assert!(found, "could not locate CDEF strength span");
            // With both luma levels zero, the chroma levels are absent, not zero bits.
            if std::env::var_os("AV1_PREFIX_ZERO_LOOP_FILTER").is_some() {
                expected.loop_filter_levels = [0; 4];
                expected.header_bytes = (frame.header.header_bytes * 8 - 12).div_ceil(8);
                let original = frame_obu.payload.clone();
                let bits: Vec<bool> = (0..frame.header.header_bytes * 8)
                    .map(|i| original[i / 8] >> (7 - i % 8) & 1 != 0)
                    .collect();
                let mut found = false;
                for start in 0..bits.len() - 24 {
                    let remaining: Vec<bool> = bits
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &v)| {
                            if (start + 12..start + 24).contains(&i) {
                                None
                            } else {
                                Some(if (start..start + 12).contains(&i) {
                                    false
                                } else {
                                    v
                                })
                            }
                        })
                        .collect();
                    let mut header = vec![0u8; remaining.len().div_ceil(8)];
                    for (i, &v) in remaining.iter().enumerate() {
                        if v {
                            header[i / 8] |= 1 << (7 - i % 8);
                        }
                    }
                    let parsed = IntraFrameHeader::parse(&header, &sequence, 0, 0).ok();
                    if let Some(parsed) = &parsed {
                        expected.header_bytes = parsed.header_bytes;
                    }
                    if parsed.as_ref() == Some(&expected) {
                        header.truncate(expected.header_bytes);
                        header.extend_from_slice(&original[frame.header.header_bytes..]);
                        frame_obu.payload = header;
                        found = true;
                        println!("neutralized loop filter levels at bit {start}");
                        break;
                    }
                }
                assert!(found, "could not locate loop filter level span");
            }
            let mut encoded = Vec::new();
            for obu in modified {
                assert_eq!((obu.temporal_id, obu.spatial_id), (0, 0));
                encoded.push(obu.kind << 3 | 2);
                let mut n = obu.payload.len();
                loop {
                    encoded.push((n & 127) as u8 | if n > 127 { 128 } else { 0 });
                    n >>= 7;
                    if n == 0 {
                        break;
                    }
                }
                encoded.extend_from_slice(&obu.payload);
            }
            std::fs::write(path, encoded).unwrap();
            return;
        }
        let mut tile = IntraTile::new(&sequence, &frame.header, frame.tiles[0].1).unwrap();
        let stopped = tile.run().err();
        assert!(tile.progress.blocks[0].reconstructed);
        assert!(tile.progress.blocks.len() >= 10);
        let mut offset = 0;
        let mut prefix_maximum = 0;
        for (plane, p) in tile.planes.iter().enumerate() {
            let sx = usize::from(plane > 0 && sequence.subsampling_x);
            let sy = usize::from(plane > 0 && sequence.subsampling_y);
            let width = (frame.header.width as usize).div_ceil(1 << sx);
            let height = (frame.header.height as usize).div_ceil(1 << sy);
            let mut differences = 0;
            let mut maximum = 0;
            let mut compared = 0;
            let mut first_bad = 0;
            for block in tile.progress.blocks.iter().filter(|b| b.reconstructed) {
                let before = differences;
                for row in (block.y >> sy)..((block.y + block.height) >> sy).min(height) {
                    for col in (block.x >> sx)..((block.x + block.width) >> sx).min(width) {
                        let difference = (i32::from(p.samples[row * p.stride + col])
                            - i32::from(oracle[offset + row * width + col]))
                        .unsigned_abs();
                        differences += usize::from(difference > 0);
                        maximum = maximum.max(difference);
                        compared += 1;
                    }
                }
                if before != differences && first_bad < 8 {
                    println!(
                        "plane {plane} first differing block {block:?}: {}",
                        differences - before
                    );
                    first_bad += 1;
                }
            }
            println!(
                "prefix plane {plane}: compared={compared}, differing={differences}, maximum={maximum}, stop={stopped:?}"
            );
            prefix_maximum = prefix_maximum.max(maximum);
            offset += width * height;
        }
        if stopped.is_none()
            || (frame.header.cdef_strengths.iter().all(|s| *s == [0; 4])
                && frame.header.loop_filter_levels == [0; 4])
        {
            assert_eq!(
                prefix_maximum, 0,
                "unfiltered prefix differs from binary oracle"
            );
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IntraDecodeProgress {
    pub blocks: Vec<IntraBlockInfo>,
    /// A prefix is diagnostic only; it is never a displayable decoded frame.
    pub stopped: Option<Error>,
}

#[derive(Clone, Copy, Default)]
struct Cell {
    width: u8,
    height: u8,
    y_mode: u8,
    uv_mode: u8,
    skip: bool,
    tx_width: u8,
    tx_height: u8,
}

pub(crate) struct IntraTile<'a> {
    s: &'a SequenceHeader,
    h: &'a IntraFrameHeader,
    decoder: SymbolDecoder<'a>,
    partitions: PartitionCdfs,
    cells: Vec<Cell>,
    cols: usize,
    rows: usize,
    y_cdf: Vec<u16>,
    uv_no_cfl: Vec<u16>,
    uv_cfl: Vec<u16>,
    skip_cdf: [[u16; 3]; 3],
    filter_cdf: [[u16; 3]; 22],
    filter_mode_cdf: [u16; 6],
    angle_cdf: [[u16; 8]; 8],
    tx_cdf: [[[u16; 4]; 3]; 3],
    tx8_cdf: [[u16; 3]; 3],
    delta_q_cdf: [u16; 5],
    cfl_sign: [u16; 9],
    cfl_alpha: Vec<u16>,
    coefficients: CoefficientState,
    inter_cdfs: super::inter::InterCdfs,
    references: [Option<&'a DecodedFrame>; 7],
    motion_cells: Vec<Option<MotionCell>>,
    motion_field: Option<super::temporal::MotionField>,
    tx_types: Vec<usize>,
    restoration: super::restoration::Restoration,
    qindex: u8,
    read_deltas: bool,
    cdef_index: Option<u8>,
    decoded: Vec<Vec<bool>>,
    filter_tx: Vec<Vec<(u8, u8)>>,
    cdef_indices: Vec<i16>,
    pub(crate) planes: Vec<DecodedPlane>,
    pub(crate) progress: IntraDecodeProgress,
}

#[derive(Clone)]
pub(crate) struct FrameCdfs {
    partitions: PartitionCdfs,
    y: Vec<u16>,
    uv_no_cfl: Vec<u16>,
    uv_cfl: Vec<u16>,
    skip: [[u16; 3]; 3],
    filter: [[u16; 3]; 22],
    filter_mode: [u16; 6],
    angle: [[u16; 8]; 8],
    tx: [[[u16; 4]; 3]; 3],
    tx8: [[u16; 3]; 3],
    delta_q: [u16; 5],
    cfl_sign: [u16; 9],
    cfl_alpha: Vec<u16>,
    coefficients: CoefficientState,
    inter: super::inter::InterCdfs,
}

impl FrameCdfs {
    pub(crate) fn reset_counts(&mut self) {
        fn reset(values: &mut [u16], n: usize) {
            for row in values.chunks_exact_mut(n) {
                row[n - 1] = 0;
            }
        }
        self.partitions.reset_counts();
        reset(&mut self.y, 14);
        reset(&mut self.uv_no_cfl, 14);
        reset(&mut self.uv_cfl, 15);
        for row in &mut self.skip {
            row[2] = 0;
        }
        for row in &mut self.filter {
            row[2] = 0;
        }
        self.filter_mode[5] = 0;
        for row in &mut self.angle {
            row[7] = 0;
        }
        for group in &mut self.tx {
            for row in group {
                row[3] = 0;
            }
        }
        for row in &mut self.tx8 {
            row[2] = 0;
        }
        self.delta_q[4] = 0;
        self.cfl_sign[8] = 0;
        reset(&mut self.cfl_alpha, 17);
        self.coefficients.reset_counts();
        self.inter.reset_counts();
    }
}

impl<'a> IntraTile<'a> {
    pub(crate) fn set_references(&mut self, refs: [Option<&'a DecodedFrame>; 7]) {
        self.references = refs;
    }

    fn read_var_tx(
        &mut self,
        origin: [usize; 2],
        pos: [usize; 2],
        size: [usize; 2],
        block: [usize; 2],
        depth: u8,
        out: &mut Vec<(usize, usize, usize, usize)>,
    ) -> Result<(), Error> {
        let [row, col] = pos;
        let [th, tw] = size;
        if row >= self.rows || col >= self.cols {
            return Ok(());
        }
        let above = if row == origin[0] && row == 0 {
            64
        } else if row == origin[0]
            && self.cells[(row - 1) * self.cols + col].skip
            && self.motion_cells[(row - 1) * self.cols + col].is_some_and(|c| c.refs[0] > 0)
        {
            self.cells[(row - 1) * self.cols + col].width as usize
        } else {
            self.filter_tx[0][(row - 1) * self.cols + col].0 as usize
        };
        let left = if col == origin[1] && col == 0 {
            64
        } else if col == origin[1]
            && self.cells[row * self.cols + col - 1].skip
            && self.motion_cells[row * self.cols + col - 1].is_some_and(|c| c.refs[0] > 0)
        {
            self.cells[row * self.cols + col - 1].height as usize
        } else {
            self.filter_tx[0][row * self.cols + col - 1].1 as usize
        };
        let max = block[0].max(block[1]).min(64).ilog2() as usize - 2;
        let upper = th.max(tw).ilog2() as usize - 2;
        let ctx = usize::from(upper != max) * 3
            + (4 - max) * 6
            + usize::from(above < tw)
            + usize::from(left < th);
        let split = th.max(tw) > 4
            && depth < 2
            && self
                .decoder
                .read_symbol(&mut self.inter_cdfs.split[ctx * 3..ctx * 3 + 3])?
                != 0;
        if split {
            let i = [0, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 7, 8, 9, 10][tx_index(tw, th)?];
            let (sw, sh) = super::coefficients::TX_DIMENSIONS[i];
            for dy in (0..th / 4).step_by(sh / 4) {
                for dx in (0..tw / 4).step_by(sw / 4) {
                    self.read_var_tx(
                        origin,
                        [row + dy, col + dx],
                        [sh, sw],
                        block,
                        depth + 1,
                        out,
                    )?;
                }
            }
        } else {
            out.push(((col - origin[1]) * 4, (row - origin[0]) * 4, tw, th));
            for y in row..(row + th / 4).min(self.rows) {
                for x in col..(col + tw / 4).min(self.cols) {
                    self.filter_tx[0][y * self.cols + x] = (tw as u8, th as u8);
                }
            }
        }
        Ok(())
    }

    fn predict_inter(
        &self,
        info: &InterBlock,
        plane: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    ) -> Result<Vec<u16>, Error> {
        let sx = plane > 0 && self.s.subsampling_x;
        let sy = plane > 0 && self.s.subsampling_y;
        let compound = info.cell.refs[1] > 0;
        let mut predictions = Vec::new();
        let mut post = 0;
        for list in 0..if compound { 2 } else { 1 } {
            let r = info.cell.refs[list] as usize - 1;
            let reference = self.references[r].ok_or(Error::Invalid("missing pixel reference"))?;
            let location = super::motion::sampling(
                [y, x],
                info.cell.mvs[list],
                [sy, sx],
                [self.h.height, self.h.width],
                [reference.header.height, reference.header.upscaled_width],
            )?;
            let (samples, bits) = if let Some(params) = info
                .warp
                .filter(|_| w >= 8 && h >= 8 && !self.h.force_integer_mv)
            {
                super::warp::predict(
                    &reference.planes[plane],
                    [y, x],
                    [h, w],
                    [sy, sx],
                    self.s.bit_depth,
                    params,
                    compound,
                )?
            } else {
                super::motion::predict(
                    &reference.planes[plane],
                    w,
                    h,
                    self.s.bit_depth,
                    location,
                    info.cell.filters,
                    compound,
                )?
            };
            post = bits;
            predictions.push(samples);
        }
        let weights = if compound && info.distance_weighted {
            let distances = info.cell.refs.map(|r| {
                let reference = self.references[r as usize - 1].unwrap();
                super::syntax::relative_dist(self.s, reference.header.order_hint, self.h.order_hint)
                    .abs()
                    .min(31)
            });
            let (d0, d1) = (distances[1], distances[0]);
            let order = usize::from(d0 <= d1);
            let mut i = 3;
            if d0 != 0 && d1 != 0 {
                for n in 0..3 {
                    let c = [[2, 3], [2, 5], [2, 7]][n];
                    if if order != 0 {
                        d0 * c[order] > d1 * c[1 - order]
                    } else {
                        d0 * c[order] < d1 * c[1 - order]
                    } {
                        i = n;
                        break;
                    }
                }
            }
            let lookup = [[9, 7], [11, 5], [12, 4], [13, 3]][i];
            [lookup[order], lookup[1 - order]]
        } else {
            [1, 1]
        };
        let shift = post + if info.distance_weighted { 4 } else { 1 };
        let max = (1 << self.s.bit_depth) - 1;
        let mut output: Vec<u16> = (0..w * h)
            .map(|i| {
                let value = if !compound {
                    predictions[0][i]
                } else {
                    (weights[0] * predictions[0][i]
                        + weights[1] * predictions[1][i]
                        + (1 << (shift - 1)))
                        >> shift
                };
                value.clamp(0, max) as u16
            })
            .collect();
        if let Some(mode) = info.inter_intra {
            let p = &self.planes[plane];
            let above = (y > 0).then(|| {
                (0..w)
                    .map(|i| p.samples[(y - 1) * p.stride + (x + i).min(p.width - 1)])
                    .collect::<Vec<_>>()
            });
            let left = (x > 0).then(|| {
                (0..h)
                    .map(|i| p.samples[(y + i).min(p.height - 1) * p.stride + x - 1])
                    .collect::<Vec<_>>()
            });
            let kind = match mode {
                1 => IntraMode::Vertical,
                2 => IntraMode::Horizontal,
                _ => IntraMode::Dc,
            };
            let mut intra = predict_intra(
                kind,
                w,
                h,
                self.s.bit_depth,
                above.as_deref(),
                left.as_deref(),
                None,
            )?;
            if mode == 3 {
                let mid = 1 << (self.s.bit_depth - 1);
                let top = above.unwrap_or_else(|| vec![left.as_ref().map_or(mid - 1, |v| v[0]); w]);
                let side = left.unwrap_or_else(|| vec![top[0]; h]);
                intra = super::prediction::smooth(w, h, 9, &top, &side)?;
            }
            let mask = if let Some((index, sign)) = info.wedge {
                let lw = info.cell.width * 4;
                let lh = info.cell.height * 4;
                let luma = super::blend::wedge(lw, lh, index, sign)?;
                (0..w * h)
                    .map(|i| super::blend::subsample(&luma, lw, i % w, i / w, [sy, sx]) as u8)
                    .collect::<Vec<_>>()
            } else {
                super::blend::intra_mask(w, h, mode)
            };
            for i in 0..w * h {
                let m = i32::from(mask[i]);
                output[i] =
                    ((m * i32::from(intra[i]) + (64 - m) * i32::from(output[i]) + 32) >> 6) as u16;
            }
        }
        if info.obmc {
            self.blend_obmc(info, plane, [y, x], [h, w], &mut output)?;
        }
        Ok(output)
    }

    fn blend_obmc(
        &self,
        info: &InterBlock,
        plane: usize,
        origin: [usize; 2],
        size: [usize; 2],
        output: &mut [u16],
    ) -> Result<(), Error> {
        const MASK2: [i32; 2] = [45, 64];
        const MASK4: [i32; 4] = [39, 50, 59, 64];
        const MASK8: [i32; 8] = [36, 42, 48, 53, 57, 61, 64, 64];
        const MASK16: [i32; 16] = [
            34, 37, 40, 43, 46, 49, 52, 54, 56, 58, 60, 61, 64, 64, 64, 64,
        ];
        const MASK32: [i32; 32] = [
            33, 35, 36, 38, 40, 41, 43, 44, 45, 47, 48, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 60,
            61, 62, 64, 64, 64, 64, 64, 64, 64, 64,
        ];
        let mask = |n| -> Result<&[i32], Error> {
            match n {
                2 => Ok(&MASK2),
                4 => Ok(&MASK4),
                8 => Ok(&MASK8),
                16 => Ok(&MASK16),
                32 => Ok(&MASK32),
                _ => Err(Error::Invalid("OBMC overlap size")),
            }
        };
        let [h, w] = size;
        let [y, x] = origin;
        let sx = usize::from(plane > 0 && self.s.subsampling_x);
        let sy = usize::from(plane > 0 && self.s.subsampling_y);
        let [row, col] = info.origin;
        let max = (1_i32 << self.s.bit_depth) - 1;
        for pass in 0..2 {
            if (pass == 0 && (row == 0 || w.min(h) < 8)) || (pass == 1 && col == 0) {
                continue;
            }
            let (start, extent, bound) = if pass == 0 {
                (col, info.cell.width, self.cols)
            } else {
                (row, info.cell.height, self.rows)
            };
            let limit = extent.ilog2().min(4) as usize;
            let mut position = start;
            let mut count = 0;
            while position < (start + extent).min(bound) && count < limit {
                let (cr, cc) = if pass == 0 {
                    (row - 1, position | 1)
                } else {
                    (position | 1, col - 1)
                };
                let Some(cell) = self
                    .motion_cells
                    .get(cr * self.cols + cc)
                    .copied()
                    .flatten()
                else {
                    position += 2;
                    continue;
                };
                let step = if pass == 0 { cell.width } else { cell.height }.clamp(2, 16);
                if cell.refs[0] > 0 {
                    count += 1;
                    let offset = ((position - start) * 4) >> if pass == 0 { sx } else { sy };
                    let (pw, ph) = if pass == 0 {
                        (
                            ((step * 4) >> sx).min(w.saturating_sub(offset)),
                            (h / 2).min(32 >> sy),
                        )
                    } else {
                        (
                            (w / 2).min(32 >> sx),
                            ((step * 4) >> sy).min(h.saturating_sub(offset)),
                        )
                    };
                    let weights = mask(if pass == 0 { ph } else { pw })?;
                    let (py, px) = if pass == 0 {
                        (y, x + offset)
                    } else {
                        (y + offset, x)
                    };
                    let reference = self.references[cell.refs[0] as usize - 1]
                        .ok_or(Error::Invalid("missing OBMC reference"))?;
                    let location = super::motion::sampling(
                        [py, px],
                        cell.mvs[0],
                        [sy != 0, sx != 0],
                        [self.h.height, self.h.width],
                        [reference.header.height, reference.header.upscaled_width],
                    )?;
                    let (prediction, _) = super::motion::predict(
                        &reference.planes[plane],
                        pw,
                        ph,
                        self.s.bit_depth,
                        location,
                        cell.filters,
                        false,
                    )?;
                    for iy in 0..ph {
                        for ix in 0..pw {
                            let index = (iy + if pass == 1 { offset } else { 0 }) * w
                                + ix
                                + if pass == 0 { offset } else { 0 };
                            let weight = weights[if pass == 0 { iy } else { ix }];
                            output[index] = ((weight * i32::from(output[index])
                                + (64 - weight) * prediction[iy * pw + ix].clamp(0, max)
                                + 32)
                                >> 6) as u16;
                        }
                    }
                }
                position += step;
            }
        }
        Ok(())
    }

    pub(crate) fn save_cdfs(&self) -> FrameCdfs {
        FrameCdfs {
            partitions: self.partitions.clone(),
            y: self.y_cdf.clone(),
            uv_no_cfl: self.uv_no_cfl.clone(),
            uv_cfl: self.uv_cfl.clone(),
            skip: self.skip_cdf,
            filter: self.filter_cdf,
            filter_mode: self.filter_mode_cdf,
            angle: self.angle_cdf,
            tx: self.tx_cdf,
            tx8: self.tx8_cdf,
            delta_q: self.delta_q_cdf,
            cfl_sign: self.cfl_sign,
            cfl_alpha: self.cfl_alpha.clone(),
            coefficients: self.coefficients.clone(),
            inter: self.inter_cdfs.clone(),
        }
    }

    pub(crate) fn load_cdfs(&mut self, saved: &FrameCdfs) {
        let mut c = saved.clone();
        c.reset_counts();
        self.partitions = c.partitions;
        self.y_cdf = c.y;
        self.uv_no_cfl = c.uv_no_cfl;
        self.uv_cfl = c.uv_cfl;
        self.skip_cdf = c.skip;
        self.filter_cdf = c.filter;
        self.filter_mode_cdf = c.filter_mode;
        self.angle_cdf = c.angle;
        self.tx_cdf = c.tx;
        self.tx8_cdf = c.tx8;
        self.delta_q_cdf = c.delta_q;
        self.cfl_sign = c.cfl_sign;
        self.cfl_alpha = c.cfl_alpha;
        self.coefficients.load_cdfs(&c.coefficients);
        self.inter_cdfs = c.inter;
    }

    pub(crate) fn new(
        s: &'a SequenceHeader,
        h: &'a IntraFrameHeader,
        tile: &'a [u8],
    ) -> Result<Self, Error> {
        if s.use_128x128_superblock || h.tiles.count() != 1 || h.width != h.upscaled_width {
            return Err(Error::Unsupported(
                "128x128 superblocks, multiple tiles or superresolution",
            ));
        }
        if h.allow_intrabc
            || h.allow_screen_content_tools
            || h.segmentation_enabled
            || h.quantizer_matrix_levels.is_some()
        {
            return Err(Error::Unsupported(
                "intra block tools, quantizer matrix or restoration",
            ));
        }
        if h.width as u64 * h.height as u64 > 16 * 1024 * 1024 {
            return Err(Error::Unsupported("AV1 pixel allocation limit"));
        }
        let cols = 2 * h.width.div_ceil(8) as usize;
        let rows = 2 * h.height.div_ceil(8) as usize;
        let dims: Vec<(usize, usize)> = (0..if s.monochrome { 1 } else { 3 })
            .map(|plane| {
                let sx = usize::from(plane > 0 && s.subsampling_x);
                let sy = usize::from(plane > 0 && s.subsampling_y);
                ((cols * 4) >> sx, (rows * 4) >> sy)
            })
            .collect();
        let mut filter_cdf = [[0; 3]; 22];
        for (cdf, p) in filter_cdf.iter_mut().zip([
            4621, 6743, 5893, 7866, 12551, 9394, 12408, 14301, 12756, 22343, 16384, 16384, 16384,
            16384, 16384, 16384, 12770, 10368, 20229, 18101, 16384, 16384,
        ]) {
            *cdf = [p, 32768, 0];
        }
        Ok(Self {
            s,
            h,
            decoder: SymbolDecoder::new(tile, !h.disable_cdf_update)?,
            partitions: PartitionCdfs::default(),
            cells: vec![Cell::default(); cols * rows],
            cols,
            rows,
            y_cdf: tables::Y_MODE.to_vec(),
            uv_no_cfl: tables::UV_NO_CFL.to_vec(),
            uv_cfl: tables::UV_CFL.to_vec(),
            skip_cdf: [[31671, 32768, 0], [16515, 32768, 0], [4576, 32768, 0]],
            filter_cdf,
            filter_mode_cdf: [8949, 12776, 17211, 29558, 32768, 0],
            angle_cdf: [
                [2180, 5032, 7567, 22776, 26989, 30217, 32768, 0],
                [2301, 5608, 8801, 23487, 26974, 30330, 32768, 0],
                [3780, 11018, 13699, 19354, 23083, 31286, 32768, 0],
                [4581, 11226, 15147, 17138, 21834, 28397, 32768, 0],
                [1737, 10927, 14509, 19588, 22745, 28823, 32768, 0],
                [2664, 10176, 12485, 17650, 21600, 30495, 32768, 0],
                [2240, 11096, 15453, 20341, 22561, 28917, 32768, 0],
                [3605, 10428, 12459, 17676, 21244, 30655, 32768, 0],
            ],
            tx8_cdf: [[19968, 32768, 0], [19968, 32768, 0], [24320, 32768, 0]],
            tx_cdf: [
                [
                    [12272, 30172, 32768, 0],
                    [12272, 30172, 32768, 0],
                    [18677, 30848, 32768, 0],
                ],
                [
                    [12986, 15180, 32768, 0],
                    [12986, 15180, 32768, 0],
                    [24302, 25602, 32768, 0],
                ],
                [
                    [5782, 11475, 32768, 0],
                    [5782, 11475, 32768, 0],
                    [16803, 22759, 32768, 0],
                ],
            ],
            delta_q_cdf: [28160, 32120, 32677, 32768, 0],
            cfl_sign: tables::CFL_SIGN,
            cfl_alpha: tables::CFL_ALPHA.to_vec(),
            coefficients: CoefficientState::new(h.base_q_idx, &dims),
            inter_cdfs: super::inter::InterCdfs::default(),
            references: [None; 7],
            motion_cells: vec![None; cols * rows],
            motion_field: None,
            tx_types: vec![0; cols * rows],
            restoration: super::restoration::Restoration::new(s, h)?,
            qindex: h.base_q_idx,
            read_deltas: false,
            cdef_index: None,
            decoded: dims
                .iter()
                .map(|&(w, h)| vec![false; w / 4 * (h / 4)])
                .collect(),
            filter_tx: dims
                .iter()
                .map(|&(w, h)| vec![(4, 4); w / 4 * (h / 4)])
                .collect(),
            cdef_indices: vec![-1; cols.div_ceil(16) * rows.div_ceil(16)],
            planes: dims
                .into_iter()
                .map(|(width, height)| DecodedPlane {
                    width,
                    height,
                    stride: width,
                    samples: vec![0; width * height],
                })
                .collect(),
            progress: IntraDecodeProgress::default(),
        })
    }

    pub(crate) fn run(&mut self) -> Result<(), Error> {
        self.run_with_observer(|_, _| {})
    }

    #[cfg(test)]
    pub(crate) fn test_motion_at(&self, x: usize, y: usize) -> Option<MotionCell> {
        self.motion_cells[(y / 4) * self.cols + x / 4]
    }

    pub(crate) fn run_with_observer(
        &mut self,
        mut observer: impl FnMut(&str, &Self),
    ) -> Result<(), Error> {
        for y in (0..self.rows).step_by(16) {
            for x in (0..self.cols).step_by(16) {
                self.read_deltas = self.h.delta_q_resolution.is_some();
                self.cdef_index = None;
                self.restoration.read(
                    &mut self.decoder,
                    &mut self.inter_cdfs.wiener,
                    self.s,
                    self.h,
                    y,
                    x,
                )?;
                self.partition(x, y, 64)?;
                self.cdef_indices[(y / 16) * self.cols.div_ceil(16) + x / 16] =
                    self.cdef_index.map_or(-1, i16::from);
            }
        }
        self.decoder.finish()?;
        observer("reconstructed", self);
        let skips: Vec<bool> = self.cells.iter().map(|c| c.skip).collect();
        super::filters::deblock(
            &mut self.planes,
            self.s,
            self.h,
            &self.filter_tx,
            &self.motion_cells,
            &skips,
            self.cols,
        )?;
        observer("deblocked", self);
        let deblocked = self.planes.clone();
        super::filters::cdef(
            &mut self.planes,
            self.s,
            self.h,
            &skips,
            self.cols,
            &self.cdef_indices,
        )?;
        observer("CDEF", self);
        self.restoration
            .apply(&mut self.planes, &deblocked, self.s, self.h)?;
        observer("restored", self);
        Ok(())
    }

    fn neighbors(&self, x: usize, y: usize) -> (Option<Cell>, Option<Cell>) {
        (
            if y > 0 {
                Some(self.cells[(y - 1) * self.cols + x])
            } else {
                None
            },
            if x > 0 {
                Some(self.cells[y * self.cols + x - 1])
            } else {
                None
            },
        )
    }

    fn partition(&mut self, x: usize, y: usize, size: usize) -> Result<(), Error> {
        if x >= self.cols || y >= self.rows {
            return Ok(());
        }
        let (above, left) = self.neighbors(x, y);
        let context = usize::from(above.is_some_and(|c| usize::from(c.width) < size))
            + 2 * usize::from(left.is_some_and(|c| usize::from(c.height) < size));
        let half = size / 2;
        let step = half / 4;
        let rows = y + step < self.rows;
        let cols = x + step < self.cols;
        let partition =
            self.partitions
                .read_partition(&mut self.decoder, size, context, rows, cols)?;
        let quarter = size / 4;
        match partition {
            0 => self.block(x, y, size, size),
            1 => {
                self.block(x, y, size, half)?;
                if rows {
                    self.block(x, y + step, size, half)?;
                }
                Ok(())
            }
            2 => {
                self.block(x, y, half, size)?;
                if cols {
                    self.block(x + step, y, half, size)?;
                }
                Ok(())
            }
            3 => {
                for (dx, dy) in [(0, 0), (step, 0), (0, step), (step, step)] {
                    self.partition(x + dx, y + dy, half)?;
                }
                Ok(())
            }
            4 => {
                self.block(x, y, half, half)?;
                self.block(x + step, y, half, half)?;
                self.block(x, y + step, size, half)
            }
            5 => {
                self.block(x, y, size, half)?;
                self.block(x, y + step, half, half)?;
                self.block(x + step, y + step, half, half)
            }
            6 => {
                self.block(x, y, half, half)?;
                self.block(x, y + step, half, half)?;
                self.block(x + step, y, half, size)
            }
            7 => {
                self.block(x, y, half, size)?;
                self.block(x + step, y, half, half)?;
                self.block(x + step, y + step, half, half)
            }
            8 => {
                for i in 0..4 {
                    self.block(x, y + i * quarter / 4, size, quarter)?;
                }
                Ok(())
            }
            9 => {
                for i in 0..4 {
                    self.block(x + i * quarter / 4, y, quarter, size)?;
                }
                Ok(())
            }
            _ => Err(Error::Invalid("partition symbol")),
        }
    }

    fn angle(&mut self, mode: usize, block_index: usize) -> Result<i32, Error> {
        if (1..=8).contains(&mode) && block_index >= 3 {
            return Ok(self.decoder.read_symbol(&mut self.angle_cdf[mode - 1])? as i32 - 3);
        }
        Ok(0)
    }

    fn block(&mut self, x: usize, y: usize, w: usize, h: usize) -> Result<(), Error> {
        if x >= self.cols || y >= self.rows {
            return Ok(());
        }
        const BLOCK_DIMS: [(usize, usize); 22] = [
            (4, 4),
            (4, 8),
            (8, 4),
            (8, 8),
            (8, 16),
            (16, 8),
            (16, 16),
            (16, 32),
            (32, 16),
            (32, 32),
            (32, 64),
            (64, 32),
            (64, 64),
            (64, 128),
            (128, 64),
            (128, 128),
            (4, 16),
            (16, 4),
            (8, 32),
            (32, 8),
            (16, 64),
            (64, 16),
        ];
        let block_index = BLOCK_DIMS
            .iter()
            .position(|&d| d == (w, h))
            .ok_or(Error::Invalid("block dimensions"))?;
        let (above, left) = self.neighbors(x, y);
        let inter_frame = matches!(self.h.frame_type, 1 | 3);
        let motion_above = if y > 0 {
            self.motion_cells[(y - 1) * self.cols + x]
        } else {
            None
        };
        let motion_left = if x > 0 {
            self.motion_cells[y * self.cols + x - 1]
        } else {
            None
        };
        let skip_mode = inter_frame
            && self.h.skip_mode_present
            && w.min(h) >= 8
            && self
                .inter_cdfs
                .read_skip_mode(&mut self.decoder, motion_above, motion_left)?;
        let skip_ctx =
            usize::from(above.is_some_and(|c| c.skip)) + usize::from(left.is_some_and(|c| c.skip));
        let skip = skip_mode || self.decoder.read_symbol(&mut self.skip_cdf[skip_ctx])? != 0;
        if !skip && !self.h.coded_lossless && self.s.enable_cdef && self.cdef_index.is_none() {
            let n = self.h.cdef_strengths.len().ilog2() as u8;
            self.cdef_index = Some(self.decoder.read_literal(n)? as u8);
        }
        if !(w == 64 && h == 64 && skip) && self.read_deltas {
            let mut magnitude = self.decoder.read_symbol(&mut self.delta_q_cdf)? as i32;
            if magnitude == 3 {
                let n = self.decoder.read_literal(3)? as u8 + 1;
                magnitude = self.decoder.read_literal(n)? as i32 + (1 << n) + 1;
            }
            if magnitude > 0 && self.decoder.read_bool()? {
                magnitude = -magnitude;
            }
            self.qindex = (i32::from(self.qindex)
                + (magnitude << self.h.delta_q_resolution.unwrap()))
            .clamp(1, 255) as u8;
            if self.h.delta_lf.is_some() {
                return Err(Error::Unsupported("block delta loop filter"));
            }
        }
        self.read_deltas = false;
        let is_inter = inter_frame
            && (skip_mode
                || self
                    .inter_cdfs
                    .read_is_inter(&mut self.decoder, motion_above, motion_left)?);
        let info = if is_inter {
            let mut hints = [0; 8];
            for (i, r) in self.references.iter().enumerate() {
                hints[i + 1] = r
                    .ok_or(Error::Invalid("missing motion reference"))?
                    .header
                    .order_hint;
            }
            Some(self.inter_cdfs.read_block(
                &mut self.decoder,
                self.s,
                self.h,
                &self.motion_cells,
                [self.rows, self.cols],
                [y, x],
                [h / 4, w / 4],
                block_index,
                skip_mode,
                hints,
                self.motion_field.as_ref(),
            )?)
        } else {
            None
        };
        const MODE_CTX: [usize; 13] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];
        let ac = above.map_or(0, |c| MODE_CTX.get(c.y_mode as usize).copied().unwrap_or(0));
        let lc = left.map_or(0, |c| MODE_CTX.get(c.y_mode as usize).copied().unwrap_or(0));
        let start = (ac * 5 + lc) * 14;
        let y_mode = if let Some(info) = &info {
            info.cell.mode
        } else if inter_frame {
            let group = (w.min(h).ilog2() as usize - 2).min(3);
            self.decoder
                .read_symbol(&mut self.inter_cdfs.y_mode[group * 14..group * 14 + 14])?
        } else {
            self.decoder
                .read_symbol(&mut self.y_cdf[start..start + 14])?
        };
        let y_angle = if is_inter {
            0
        } else {
            self.angle(y_mode, block_index)?
        };
        let has_chroma = !self.s.monochrome
            && !(h == 4 && self.s.subsampling_y && y % 2 == 0)
            && !(w == 4 && self.s.subsampling_x && x % 2 == 0);
        let mut uv_mode = 0;
        let mut uv_angle = 0;
        let mut cfl_alpha = [0; 2];
        if has_chroma && !is_inter {
            let uv_w = (w >> usize::from(self.s.subsampling_x)).max(4);
            let uv_h = (h >> usize::from(self.s.subsampling_y)).max(4);
            let cfl_allowed = if self.h.coded_lossless {
                uv_w == 4 && uv_h == 4
            } else {
                w.max(h) <= 32
            };
            let (table, n) = if cfl_allowed {
                (&mut self.uv_cfl, 15)
            } else {
                (&mut self.uv_no_cfl, 14)
            };
            let start = y_mode * n;
            uv_mode = self.decoder.read_symbol(&mut table[start..start + n])?;
            if uv_mode == 13 {
                let signs = self.decoder.read_symbol(&mut self.cfl_sign)? + 1;
                let sign_u = signs / 3;
                let sign_v = signs % 3;
                for (plane, (sign, other)) in
                    [(sign_u, sign_v), (sign_v, sign_u)].into_iter().enumerate()
                {
                    if sign != 0 {
                        let ctx = (sign - 1) * 3 + other;
                        let start = ctx * 17;
                        let alpha = self
                            .decoder
                            .read_symbol(&mut self.cfl_alpha[start..start + 17])?
                            as i32
                            + 1;
                        cfl_alpha[plane] = if sign == 1 { -alpha } else { alpha };
                    }
                }
            }
            uv_angle = self.angle(uv_mode, block_index)?;
        }
        let mut filter_mode = None;
        if !is_inter
            && self.s.enable_filter_intra
            && y_mode == 0
            && w.max(h) <= 32
            && self
                .decoder
                .read_symbol(&mut self.filter_cdf[block_index])?
                != 0
        {
            filter_mode = Some(self.decoder.read_symbol(&mut self.filter_mode_cdf)?);
        }
        let (mut tw, mut th) = if self.h.coded_lossless {
            (4, 4)
        } else {
            (w, h)
        };
        if !is_inter && !self.h.coded_lossless && self.h.tx_mode_select && block_index > 0 {
            let ctx = usize::from(above.is_some_and(|c| {
                usize::from(if motion_above.is_some_and(|m| m.refs[0] > 0) {
                    c.width
                } else {
                    c.tx_width
                }) >= w
            })) + usize::from(left.is_some_and(|c| {
                usize::from(if motion_left.is_some_and(|m| m.refs[0] > 0) {
                    c.height
                } else {
                    c.tx_height
                }) >= h
            }));
            let max_depth = w.max(h).ilog2() as usize - 2;
            let depth = if max_depth == 1 {
                self.decoder.read_symbol(&mut self.tx8_cdf[ctx])?
            } else {
                self.decoder
                    .read_symbol(&mut self.tx_cdf[max_depth - 2][ctx])?
            };
            for _ in 0..depth {
                let split =
                    [0, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 7, 8, 9, 10][tx_index(tw, th)?];
                (tw, th) = super::coefficients::TX_DIMENSIONS[split];
            }
        }
        let mut inter_txs = Vec::new();
        if is_inter && !skip && !self.h.coded_lossless && self.h.tx_mode_select && w.max(h) > 4 {
            self.read_var_tx([y, x], [y, x], [th, tw], [h, w], 0, &mut inter_txs)?;
        }
        let index = self.progress.blocks.len();
        self.progress.blocks.push(IntraBlockInfo {
            x: x * 4,
            y: y * 4,
            width: w,
            height: h,
            skip,
            y_mode,
            uv_mode,
            qindex: self.qindex,
            reconstructed: false,
        });
        // CFL uses complete reconstructed transforms, including their non-visible edge samples.
        let mut cfl_window = if has_chroma && uv_mode == 13 {
            let sx = usize::from(self.s.subsampling_x);
            let sy = usize::from(self.s.subsampling_y);
            let bx = ((x >> sx) * 4) << sx;
            let by = ((y >> sy) * 4) << sy;
            let width = (w >> sx).max(4) << sx;
            let height = (h >> sy).max(4) << sy;
            let luma = &self.planes[0];
            let mut samples = vec![0; width * height];
            for row in 0..height.min(luma.height.saturating_sub(by)) {
                let count = width.min(luma.width.saturating_sub(bx));
                samples[row * width..row * width + count].copy_from_slice(
                    &luma.samples
                        [(by + row) * luma.stride + bx..(by + row) * luma.stride + bx + count],
                );
            }
            Some((bx, by, width, height, samples, [0, 0]))
        } else {
            None
        };
        for plane in 0..if has_chroma { 3 } else { 1 } {
            let sx = usize::from(plane > 0 && self.s.subsampling_x);
            let sy = usize::from(plane > 0 && self.s.subsampling_y);
            let px = (x >> sx) * 4;
            let py = (y >> sy) * 4;
            let pw = (w >> sx).max(4);
            let ph = (h >> sy).max(4);
            let (ptw, pth) = if plane == 0 {
                (tw, th)
            } else if self.h.coded_lossless {
                (4, 4)
            } else {
                if pw == 64 || ph == 64 {
                    if pw == 16 {
                        (16, 32)
                    } else if ph == 16 {
                        (32, 16)
                    } else {
                        (32, 32)
                    }
                } else {
                    (pw, ph)
                }
            };
            if skip {
                self.coefficients.reset(plane, px, py, pw, ph);
            }
            let inter_prediction = info
                .as_ref()
                .map(|i| self.predict_inter(i, plane, px, py, pw, ph))
                .transpose()?;
            let transforms = if plane == 0 && !inter_txs.is_empty() {
                inter_txs.clone()
            } else {
                (0..ph)
                    .step_by(pth)
                    .flat_map(|dy| (0..pw).step_by(ptw).map(move |dx| (dx, dy, ptw, pth)))
                    .collect()
            };
            for (dx, dy, ptw, pth) in transforms {
                let ax = px + dx;
                let ay = py + dy;
                if ax >= self.planes[plane].width || ay >= self.planes[plane].height {
                    continue;
                }
                let mode = if plane == 0 { y_mode } else { uv_mode };
                let mode_kind = match mode {
                    0 | 13 => IntraMode::Dc,
                    1 => IntraMode::Vertical,
                    2 => IntraMode::Horizontal,
                    12 => IntraMode::Paeth,
                    _ => IntraMode::Dc,
                };
                let p = &self.planes[plane];
                let decoded = |cx: usize, cy: usize| -> bool {
                    cx < p.width / 4
                        && cy < p.height / 4
                        && self.decoded[plane][cy * (p.width / 4) + cx]
                };
                let above_right = ay > 0 && decoded(ax / 4 + ptw / 4, ay / 4 - 1);
                let below_left = ax > 0 && decoded(ax / 4 - 1, ay / 4 + pth / 4);
                let above = if ay > 0 {
                    Some(
                        (0..ptw + pth)
                            .map(|i| {
                                p.samples[(ay - 1) * p.stride
                                    + (ax + i).min(
                                        (ax + if above_right { 2 * ptw } else { ptw } - 1)
                                            .min(p.width - 1),
                                    )]
                            })
                            .collect::<Vec<_>>(),
                    )
                } else {
                    None
                };
                let left = if ax > 0 {
                    Some(
                        (0..ptw + pth)
                            .map(|i| {
                                p.samples[(ay + i).min(
                                    (ay + if below_left { 2 * pth } else { pth } - 1)
                                        .min(p.height - 1),
                                ) * p.stride
                                    + ax
                                    - 1]
                            })
                            .collect::<Vec<_>>(),
                    )
                } else {
                    None
                };
                let corner = if ax > 0 && ay > 0 {
                    Some(p.samples[(ay - 1) * p.stride + ax - 1])
                } else {
                    None
                };
                let mut prediction = predict_intra(
                    mode_kind,
                    ptw,
                    pth,
                    self.s.bit_depth,
                    above.as_deref(),
                    left.as_deref(),
                    corner,
                )?;
                let mid = 1u16 << (self.s.bit_depth - 1);
                let corner_value = corner.unwrap_or_else(|| {
                    above
                        .as_ref()
                        .map(|a| a[0])
                        .or_else(|| left.as_ref().map(|a| a[0]))
                        .unwrap_or(mid)
                });
                let top = above
                    .clone()
                    .unwrap_or_else(|| vec![left.as_ref().map_or(mid - 1, |l| l[0]); ptw + pth]);
                let side = left
                    .clone()
                    .unwrap_or_else(|| vec![above.as_ref().map_or(mid + 1, |a| a[0]); ptw + pth]);
                if plane == 0 && filter_mode.is_some() {
                    prediction = super::prediction::recursive(
                        ptw,
                        pth,
                        filter_mode.unwrap(),
                        self.s.bit_depth,
                        &top,
                        &side,
                        corner_value,
                    )?;
                } else if (1..=8).contains(&mode) {
                    let smooth_neighbor = [
                        above.as_ref().and_then(|_| {
                            if ay > 0 {
                                self.cells.get((y.saturating_sub(1)) * self.cols + x)
                            } else {
                                None
                            }
                        }),
                        if x > 0 {
                            self.cells.get(y * self.cols + x - 1)
                        } else {
                            None
                        },
                    ]
                    .into_iter()
                    .flatten()
                    .any(|c| (9..=11).contains(&if plane == 0 { c.y_mode } else { c.uv_mode }));
                    prediction = super::prediction::directional(
                        ptw,
                        pth,
                        self.s.bit_depth,
                        mode,
                        if plane == 0 { y_angle } else { uv_angle },
                        Some(top.as_slice()).filter(|_| ay > 0),
                        Some(side.as_slice()).filter(|_| ax > 0),
                        corner_value,
                        self.s.enable_intra_edge_filter,
                        smooth_neighbor,
                        p.width - ax,
                        p.height - ay,
                    )?;
                } else if (9..=11).contains(&mode) {
                    prediction = super::prediction::smooth(ptw, pth, mode, &top, &side)?;
                }
                if plane > 0 && uv_mode == 13 {
                    let (bx, by, stride, _, luma, max) = cfl_window
                        .as_ref()
                        .ok_or(Error::Invalid("missing CFL luma window"))?;
                    predict_cfl(
                        &mut prediction,
                        ptw,
                        pth,
                        self.s.bit_depth,
                        luma,
                        *stride,
                        max[0],
                        max[1],
                        ax - (bx >> sx),
                        ay - (by >> sy),
                        self.s.subsampling_x,
                        self.s.subsampling_y,
                        cfl_alpha[plane - 1],
                    )?;
                }
                if let Some(p) = &inter_prediction {
                    for row in 0..pth {
                        prediction[row * ptw..(row + 1) * ptw]
                            .copy_from_slice(&p[(dy + row) * pw + dx..(dy + row) * pw + dx + ptw]);
                    }
                }
                if !skip {
                    let inherited = if !is_inter {
                        None
                    } else if plane == 0 {
                        Some(0)
                    } else {
                        let lx = (ax << sx).max(x * 4) / 4;
                        let ly = (ay << sy).max(y * 4) / 4;
                        Some(self.tx_types[ly * self.cols + lx])
                    };
                    let decoded = self.coefficients.read(
                        &mut self.decoder,
                        plane,
                        ax,
                        ay,
                        ptw,
                        pth,
                        pw,
                        ph,
                        if plane == 0 {
                            filter_mode.map_or(mode, |m| [0, 1, 2, 6, 0][m])
                        } else {
                            mode
                        },
                        self.h.coded_lossless,
                        self.h.reduced_tx_set,
                        self.h.base_q_idx,
                        inherited,
                    )?;
                    if plane == 0 {
                        for row in ay / 4..((ay + pth) / 4).min(self.rows) {
                            for col in ax / 4..((ax + ptw) / 4).min(self.cols) {
                                self.tx_types[row * self.cols + col] = decoded.tx_type;
                            }
                        }
                    }
                    let mut coefficients = decoded.values;
                    if coefficients.iter().any(|&v| v != 0) {
                        let dc_delta = self.h.quantizer_deltas[if plane == 0 {
                            0
                        } else if plane == 1 {
                            1
                        } else {
                            3
                        }];
                        let dc_quant = tables::DC_QUANT[((self.s.bit_depth - 8) / 2) as usize
                            * 256
                            + (i32::from(self.qindex) + dc_delta).clamp(0, 255) as usize];
                        let ac_delta = if plane == 0 {
                            0
                        } else {
                            self.h.quantizer_deltas[if plane == 1 { 2 } else { 4 }]
                        };
                        let ac_quant = tables::AC_QUANT[((self.s.bit_depth - 8) / 2) as usize
                            * 256
                            + (i32::from(self.qindex) + ac_delta).clamp(0, 255) as usize];
                        let denom = match tx_index(ptw, pth)? {
                            3 | 9 | 10 | 17 | 18 => 2,
                            4 | 11 | 12 => 4,
                            _ => 1,
                        };
                        let bound = 1i64 << (7 + self.s.bit_depth);
                        for (i, coefficient) in coefficients.iter_mut().enumerate() {
                            let quant = if i == 0 { dc_quant } else { ac_quant };
                            let product = i64::from(*coefficient) * i64::from(quant);
                            let dequant = ((product.abs() & 0xffffff) / denom) * product.signum();
                            *coefficient = dequant.clamp(-bound, bound - 1) as i32;
                        }
                        let residual = if self.h.coded_lossless {
                            inverse_lossless_4x4(
                                coefficients
                                    .as_slice()
                                    .try_into()
                                    .map_err(|_| Error::Invalid("lossless transform dimensions"))?,
                                self.s.bit_depth,
                            )?
                            .to_vec()
                        } else {
                            inverse_transform(
                                ptw,
                                pth,
                                &coefficients,
                                self.s.bit_depth,
                                decoded.tx_type,
                            )?
                        };
                        add_residual(&mut prediction, &residual, self.s.bit_depth)?;
                    }
                }
                if plane == 0 {
                    if let Some((bx, by, stride, height, samples, max)) = &mut cfl_window {
                        let dx = ax - *bx;
                        let dy = ay - *by;
                        max[0] = (dx + ptw).min(*stride);
                        max[1] = (dy + pth).min(*height);
                        for row in 0..pth.min(*height - dy) {
                            let count = ptw.min(*stride - dx);
                            samples[(dy + row) * *stride + dx..(dy + row) * *stride + dx + count]
                                .copy_from_slice(&prediction[row * ptw..row * ptw + count]);
                        }
                    }
                }
                let p = &mut self.planes[plane];
                for row in 0..pth.min(p.height - ay) {
                    let n = ptw.min(p.width - ax);
                    p.samples[(ay + row) * p.stride + ax..(ay + row) * p.stride + ax + n]
                        .copy_from_slice(&prediction[row * ptw..row * ptw + n]);
                }
                for row in ay / 4..((ay + pth) / 4).min(p.height / 4) {
                    for col in ax / 4..((ax + ptw) / 4).min(p.width / 4) {
                        self.decoded[plane][row * (p.width / 4) + col] = true;
                        self.filter_tx[plane][row * (p.width / 4) + col] = (ptw as u8, pth as u8);
                    }
                }
            }
        }
        self.progress.blocks[index].reconstructed = true;
        for row in y..(y + h / 4).min(self.rows) {
            for col in x..(x + w / 4).min(self.cols) {
                self.cells[row * self.cols + col] = Cell {
                    width: w as u8,
                    height: h as u8,
                    y_mode: y_mode as u8,
                    uv_mode: uv_mode as u8,
                    skip,
                    tx_width: tw as u8,
                    tx_height: th as u8,
                };
                self.motion_cells[row * self.cols + col] = Some(info.as_ref().map_or(
                    MotionCell {
                        width: w / 4,
                        height: h / 4,
                        refs: [0, -1],
                        mode: y_mode,
                        ..Default::default()
                    },
                    |i| i.cell,
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn set_motion_field(&mut self, field: super::temporal::MotionField) {
        self.motion_field = Some(field);
    }

    pub(crate) fn save_motion(&self) -> super::temporal::SavedMotion {
        let mut hints = [0; 8];
        for (i, reference) in self.references.iter().enumerate() {
            hints[i + 1] = reference.map_or(0, |r| r.header.order_hint);
        }
        super::temporal::SavedMotion::capture(self.s, self.h, &self.motion_cells, hints)
    }

    pub(crate) fn finish_planes(mut self) -> Vec<DecodedPlane> {
        for (plane, p) in self.planes.iter_mut().enumerate() {
            let sx = usize::from(plane > 0 && self.s.subsampling_x);
            let sy = usize::from(plane > 0 && self.s.subsampling_y);
            let width = (self.h.width as usize).div_ceil(1 << sx);
            let height = (self.h.height as usize).div_ceil(1 << sy);
            if width != p.width || height != p.height {
                p.samples = (0..height)
                    .flat_map(|y| {
                        p.samples[y * p.stride..y * p.stride + width]
                            .iter()
                            .copied()
                    })
                    .collect();
                p.width = width;
                p.height = height;
                p.stride = width;
            }
        }
        self.planes
    }
}
