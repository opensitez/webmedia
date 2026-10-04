//! Inter block entropy contexts and spatial motion prediction, AV1 7.10.2 and 8.3.2.

use super::entropy::SymbolDecoder;
use super::inter_tables as tables;
use super::motion::{MvCdfs, lower_precision};
use super::syntax::{Error, IntraFrameHeader, SequenceHeader, relative_dist};

pub(crate) const GLOBAL_GLOBAL: usize = 23;
const NEW_NEW: usize = 24;

fn component_mode(mode: usize, list: usize) -> usize {
    if mode < 17 {
        mode
    } else if list == 0 {
        match mode {
            20 | 22 | NEW_NEW => 16,
            17 | 19 => 13,
            18 | 21 => 14,
            _ => 15,
        }
    } else {
        match mode {
            19 | 21 | NEW_NEW => 16,
            17 | 20 => 13,
            18 | 22 => 14,
            _ => 15,
        }
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct MotionCell {
    pub width: usize,
    pub height: usize,
    pub refs: [i8; 2],
    pub mvs: [[i32; 2]; 2],
    pub mode: usize,
    pub filters: [u8; 2],
    pub skip_mode: bool,
    pub group: u8,
    pub compound_idx: u8,
}

pub(crate) struct InterBlock {
    pub cell: MotionCell,
    pub distance_weighted: bool,
    pub warp: Option<[i32; 6]>,
    pub obmc: bool,
    pub origin: [usize; 2],
    pub inter_intra: Option<usize>,
    pub wedge: Option<(usize, bool)>,
}

macro_rules! cdf_fields {
    ($( $field:ident: $table:ident / $row:literal ),* $(,)?) => {
        #[derive(Clone)]
        pub(crate) struct InterCdfs {
            $(pub(crate) $field: Vec<u16>,)*
            pub(crate) mv: MvCdfs,
        }
        impl Default for InterCdfs {
            fn default() -> Self { Self { $($field: tables::$table.to_vec(),)* mv: MvCdfs::default() } }
        }
        impl InterCdfs {
            pub(crate) fn reset_counts(&mut self) {
                $(for row in self.$field.chunks_exact_mut($row) { row[$row - 1] = 0; })*
                self.mv.reset_counts();
            }
        }
    };
}

cdf_fields! {
    wiener: WIENER / 3,
    y_mode: Y_MODE / 14, split: TXFM_SPLIT / 3,
    new_mv: NEW_MV / 3, zero_mv: ZERO_MV / 3, ref_mv: REF_MV / 3, drl: DRL_MODE / 3,
    is_inter: IS_INTER / 3, comp_mode: COMP_MODE / 3, skip_mode: SKIP_MODE / 3,
    comp_ref: COMP_REF / 3, comp_bwd_ref: COMP_BWD_REF / 3, single_ref: SINGLE_REF / 3,
    compound_mode: COMPOUND_MODE / 9, interp: INTERP_FILTER / 4, motion_mode: MOTION_MODE / 4,
    compound_idx: COMPOUND_IDX / 3, comp_group: COMP_GROUP_IDX / 3,
    compound_type: COMPOUND_TYPE / 3, inter_intra: INTER_INTRA / 3,
    inter_intra_mode: INTER_INTRA_MODE / 5, wedge: WEDGE_INDEX / 17,
    wedge_inter_intra: WEDGE_INTER_INTRA / 3, obmc: USE_OBMC / 3,
    comp_ref_type: COMP_REF_TYPE / 3, uni_comp_ref: UNI_COMP_REF / 3,
}

fn symbol(
    d: &mut SymbolDecoder<'_>,
    table: &mut [u16],
    row: usize,
    n: usize,
) -> Result<usize, Error> {
    let cdf = table
        .get_mut(row * n..(row + 1) * n)
        .ok_or(Error::Invalid("inter CDF context"))?;
    d.read_symbol(cdf)
}

impl InterCdfs {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_block(
        &mut self,
        d: &mut SymbolDecoder<'_>,
        s: &SequenceHeader,
        h: &IntraFrameHeader,
        cells: &[Option<MotionCell>],
        dims: [usize; 2],
        origin: [usize; 2],
        size: [usize; 2],
        block_index: usize,
        skip_mode: bool,
        order_hints: [u32; 8],
        field: Option<&super::temporal::MotionField>,
    ) -> Result<InterBlock, Error> {
        let at = |row: isize, col: isize| {
            if row < 0 || col < 0 || row >= dims[0] as isize || col >= dims[1] as isize {
                None
            } else {
                cells[row as usize * dims[1] + col as usize]
            }
        };
        let above = at(origin[0] as isize - 1, origin[1] as isize);
        let left = at(origin[0] as isize, origin[1] as isize - 1);
        let mut refs = if skip_mode {
            h.skip_mode_frames
                .ok_or(Error::Invalid("skip mode references"))?
                .map(|r| r as i8)
        } else {
            self.read_refs(
                d,
                above,
                left,
                h.reference_select && size[0].min(size[1]) >= 2,
            )?
        };
        let compound = refs[1] > 0;
        let mut types = [0; 8];
        types[1..].copy_from_slice(&h.global_motion_types);
        let mut global = [[0; 2]; 2];
        for list in 0..if compound { 2 } else { 1 } {
            let index = refs[list] as usize - 1;
            let p = h.global_motion_params[index];
            let kind = h.global_motion_types[index];
            let round =
                |v: i64, bits: u32| (v.signum() * ((v.abs() + (1 << (bits - 1))) >> bits)) as i32;
            global[list] = if kind == 0 {
                [0; 2]
            } else if kind == 1 {
                [p[0] >> 13, p[1] >> 13]
            } else {
                let x = origin[1] as i64 * 4 + size[1] as i64 * 2 - 1;
                let y = origin[0] as i64 * 4 + size[0] as i64 * 2 - 1;
                let xc = i64::from(p[2] - (1 << 16)) * x + i64::from(p[3]) * y + i64::from(p[0]);
                let yc = i64::from(p[4]) * x + i64::from(p[5] - (1 << 16)) * y + i64::from(p[1]);
                if h.allow_high_precision_mv {
                    [round(yc, 13), round(xc, 13)]
                } else {
                    [round(yc, 14) * 2, round(xc, 14) * 2]
                }
            };
            global[list] =
                lower_precision(global[list], h.force_integer_mv, h.allow_high_precision_mv);
        }
        let bias = order_hints.map(|hint| relative_dist(s, hint, h.order_hint) > 0);
        if h.use_ref_frame_mvs && field.is_none() {
            return Err(Error::Invalid("missing projected temporal motion field"));
        }
        let stack = MvStack::build(
            cells,
            dims,
            origin,
            size,
            refs,
            global,
            types,
            bias,
            h.force_integer_mv,
            h.allow_high_precision_mv,
            field,
        );
        let mode = if skip_mode {
            17
        } else if compound {
            let ctx = [[0, 1, 1, 1, 1], [1, 2, 3, 4, 4], [4, 4, 5, 6, 7]][stack.ref_ctx >> 1]
                [stack.new_ctx.min(4)];
            17 + symbol(d, &mut self.compound_mode, ctx, 9)?
        } else if symbol(d, &mut self.new_mv, stack.new_ctx, 3)? == 0 {
            16
        } else if symbol(d, &mut self.zero_mv, stack.zero_ctx, 3)? == 0 {
            15
        } else {
            13 + symbol(d, &mut self.ref_mv, stack.ref_ctx, 3)?
        };
        let mut ref_index = 0;
        if mode == 16 || mode == NEW_NEW {
            for idx in 0..2 {
                if stack.count > idx + 1 {
                    if symbol(d, &mut self.drl, stack.drl[idx], 3)? == 0 {
                        ref_index = idx;
                        break;
                    }
                    ref_index = idx + 1;
                }
            }
        } else if matches!(mode, 14 | 18 | 21 | 22) {
            ref_index = 1;
            for idx in 1..3 {
                if stack.count > idx + 1 {
                    if symbol(d, &mut self.drl, stack.drl[idx], 3)? == 0 {
                        ref_index = idx;
                        break;
                    }
                    ref_index = idx + 1;
                }
            }
        }
        let mut mvs = [[0; 2]; 2];
        for list in 0..if compound { 2 } else { 1 } {
            let cm = component_mode(mode, list);
            let pos = if cm == 13 || (cm == 16 && stack.count <= 1) {
                0
            } else {
                ref_index
            };
            let predicted = if cm == 15 {
                global[list]
            } else {
                stack.candidates[pos].mvs[list]
            };
            mvs[list] = if cm == 16 {
                self.mv
                    .read(d, predicted, h.force_integer_mv, h.allow_high_precision_mv)?
            } else {
                predicted
            };
        }
        let group_size = (size[0].min(size[1]).ilog2() as usize)
            .saturating_sub(1)
            .min(2);
        let mut inter_intra = None;
        let mut wedge = None;
        if !skip_mode
            && s.enable_interintra_compound
            && !compound
            && (3..=9).contains(&block_index)
            && symbol(d, &mut self.inter_intra, group_size, 3)? != 0
        {
            inter_intra = Some(symbol(d, &mut self.inter_intra_mode, group_size, 5)?);
            refs[1] = 0;
            if symbol(d, &mut self.wedge_inter_intra, block_index, 3)? != 0 {
                wedge = Some((symbol(d, &mut self.wedge, block_index, 17)?, false));
            }
        }
        let mut overlaps = false;
        if size[0].min(size[1]) >= 2
            && h.is_motion_mode_switchable
            && !skip_mode
            && !compound
            && inter_intra.is_none()
        {
            for x in (origin[1]..(origin[1] + size[1]).min(dims[1])).step_by(2) {
                overlaps |=
                    at(origin[0] as isize - 1, (x | 1) as isize).is_some_and(|c| c.refs[0] > 0);
            }
            for y in (origin[0]..(origin[0] + size[0]).min(dims[0])).step_by(2) {
                overlaps |=
                    at((y | 1) as isize, origin[1] as isize - 1).is_some_and(|c| c.refs[0] > 0);
            }
        }
        let mut warp = None;
        let mut obmc = false;
        let mut local_warp = false;
        if overlaps
            && !((mode == 15 || mode == GLOBAL_GLOBAL)
                && !h.force_integer_mv
                && types[refs[0] as usize] > 1)
        {
            let samples = super::warp::samples(cells, dims, origin, size, refs[0], mvs[0]);
            let motion = if h.force_integer_mv || samples.is_empty() || !h.allow_warped_motion {
                symbol(d, &mut self.obmc, block_index, 3)?
            } else {
                symbol(d, &mut self.motion_mode, block_index, 4)?
            };
            obmc = motion == 1;
            local_warp = motion == 2;
            if motion == 2 {
                warp = super::warp::estimate(&samples, origin, size, mvs[0]);
            }
        }
        let mut group = 0;
        let mut compound_idx = 1;
        if compound && !skip_mode {
            if s.enable_masked_compound {
                let ctx = [above, left]
                    .into_iter()
                    .flatten()
                    .map(|c| {
                        if c.refs[1] > 0 {
                            c.group as usize
                        } else {
                            3 * usize::from(c.refs[0] == 7)
                        }
                    })
                    .sum::<usize>()
                    .min(5);
                group = symbol(d, &mut self.comp_group, ctx, 3)? as u8;
            }
            if group == 0 && s.enable_jnt_comp {
                let first = relative_dist(s, order_hints[refs[0] as usize], h.order_hint).abs();
                let second = relative_dist(s, order_hints[refs[1] as usize], h.order_hint).abs();
                let mut ctx = if first == second { 3 } else { 0 };
                for c in [above, left].into_iter().flatten() {
                    ctx += if c.refs[1] > 0 {
                        c.compound_idx as usize
                    } else {
                        usize::from(c.refs[0] == 7)
                    };
                }
                compound_idx = symbol(d, &mut self.compound_idx, ctx, 3)? as u8;
            } else if group != 0 {
                return Err(Error::Unsupported("masked compound prediction"));
            }
        }
        let mut filters = [h.interpolation_filter; 2];
        if h.interpolation_filter == 4 {
            let needs = !skip_mode
                && !local_warp
                && !(size[0].min(size[1]) >= 2
                    && (mode == 15 || mode == GLOBAL_GLOBAL)
                    && types[refs[0] as usize] != 1
                    && (!compound || types[refs[1] as usize] != 1));
            for dir in 0..if s.enable_dual_filter { 2 } else { 1 } {
                filters[dir] = if !needs {
                    0
                } else {
                    let neighbor = |c: Option<MotionCell>| {
                        c.filter(|c| c.refs.contains(&refs[0]))
                            .map_or(3, |c| c.filters[dir])
                    };
                    let a = neighbor(above);
                    let l = neighbor(left);
                    let sub = if a == l {
                        a
                    } else if l == 3 {
                        a
                    } else if a == 3 {
                        l
                    } else {
                        3
                    };
                    let ctx = ((dir & 1) * 2 + usize::from(compound)) * 4 + sub as usize;
                    symbol(d, &mut self.interp, ctx, 4)? as u8
                };
            }
            if !s.enable_dual_filter {
                filters[1] = filters[0];
            }
        }
        if (mode == 15 || mode == GLOBAL_GLOBAL) && types[refs[0] as usize] > 1 {
            return Err(Error::Unsupported("global warped prediction"));
        }
        Ok(InterBlock {
            cell: MotionCell {
                width: size[1],
                height: size[0],
                refs,
                mvs,
                mode,
                filters,
                skip_mode,
                group,
                compound_idx,
            },
            distance_weighted: compound_idx == 0,
            warp,
            obmc,
            origin,
            inter_intra,
            wedge,
        })
    }

    pub(crate) fn read_is_inter(
        &mut self,
        d: &mut SymbolDecoder<'_>,
        above: Option<MotionCell>,
        left: Option<MotionCell>,
    ) -> Result<bool, Error> {
        let ctx = match (above, left) {
            (Some(a), Some(l)) => {
                if a.refs[0] <= 0 && l.refs[0] <= 0 {
                    3
                } else {
                    usize::from(a.refs[0] <= 0 || l.refs[0] <= 0)
                }
            }
            (Some(c), None) | (None, Some(c)) => 2 * usize::from(c.refs[0] <= 0),
            _ => 0,
        };
        Ok(symbol(d, &mut self.is_inter, ctx, 3)? != 0)
    }

    pub(crate) fn read_skip_mode(
        &mut self,
        d: &mut SymbolDecoder<'_>,
        above: Option<MotionCell>,
        left: Option<MotionCell>,
    ) -> Result<bool, Error> {
        let ctx = usize::from(above.is_some_and(|c| c.skip_mode))
            + usize::from(left.is_some_and(|c| c.skip_mode));
        Ok(symbol(d, &mut self.skip_mode, ctx, 3)? != 0)
    }

    pub(crate) fn read_refs(
        &mut self,
        d: &mut SymbolDecoder<'_>,
        above: Option<MotionCell>,
        left: Option<MotionCell>,
        compound_allowed: bool,
    ) -> Result<[i8; 2], Error> {
        let backward = |r: i8| usize::from(r >= 5);
        let single = |c: MotionCell| c.refs[1] <= 0;
        let ctx = match (above, left) {
            (Some(a), Some(l)) if single(a) && single(l) => {
                backward(a.refs[0]) ^ backward(l.refs[0])
            }
            (Some(a), Some(_)) if single(a) => {
                2 + usize::from(a.refs[0] <= 0 || backward(a.refs[0]) != 0)
            }
            (Some(_), Some(l)) if single(l) => {
                2 + usize::from(l.refs[0] <= 0 || backward(l.refs[0]) != 0)
            }
            (Some(_), Some(_)) => 4,
            (Some(c), None) | (None, Some(c)) => {
                if single(c) {
                    backward(c.refs[0])
                } else {
                    3
                }
            }
            _ => 1,
        };
        let compound = compound_allowed && symbol(d, &mut self.comp_mode, ctx, 3)? != 0;
        let mut counts = [0usize; 8];
        for c in [above, left].into_iter().flatten() {
            for r in c.refs {
                if r > 0 {
                    counts[r as usize] += 1;
                }
            }
        }
        let count_ctx = |a: usize, b: usize| {
            if a < b {
                0
            } else if a == b {
                1
            } else {
                2
            }
        };
        let contexts = [
            count_ctx(
                counts[1] + counts[2] + counts[3] + counts[4],
                counts[5] + counts[6] + counts[7],
            ),
            count_ctx(counts[5] + counts[6], counts[7]),
            count_ctx(counts[1] + counts[2], counts[3] + counts[4]),
            count_ctx(counts[1], counts[2]),
            count_ctx(counts[3], counts[4]),
            count_ctx(counts[5], counts[6]),
        ];
        if !compound {
            let mut read =
                |branch| symbol(d, &mut self.single_ref, contexts[branch] * 6 + branch, 3);
            let r = if read(0)? != 0 {
                if read(1)? != 0 { 7 } else { 5 + read(5)? as i8 }
            } else if read(2)? != 0 {
                3 + read(4)? as i8
            } else {
                1 + read(3)? as i8
            };
            return Ok([r, -1]);
        }
        let same_dir = |a: i8, b: i8| (a >= 5) == (b >= 5);
        let a = above.unwrap_or_default();
        let l = left.unwrap_or_default();
        let ac = above.is_some() && a.refs[0] > 0 && !single(a);
        let lc = left.is_some() && l.refs[0] > 0 && !single(l);
        let au = ac && same_dir(a.refs[0], a.refs[1]);
        let lu = lc && same_dir(l.refs[0], l.refs[1]);
        let ctx = if above.is_some() && left.is_some() && a.refs[0] > 0 && l.refs[0] > 0 {
            let same = usize::from(same_dir(a.refs[0], l.refs[0]));
            if !ac && !lc {
                1 + 2 * same
            } else if !ac {
                if !lu { 1 } else { 3 + same }
            } else if !lc {
                if !au { 1 } else { 3 + same }
            } else if !au && !lu {
                0
            } else if !au || !lu {
                2
            } else {
                3 + usize::from((a.refs[0] == 5) == (l.refs[0] == 5))
            }
        } else if above.is_some() && left.is_some() {
            if ac {
                1 + 2 * usize::from(au)
            } else if lc {
                1 + 2 * usize::from(lu)
            } else {
                2
            }
        } else if ac {
            4 * usize::from(au)
        } else if lc {
            4 * usize::from(lu)
        } else {
            2
        };
        if symbol(d, &mut self.comp_ref_type, ctx, 3)? == 0 {
            if symbol(d, &mut self.uni_comp_ref, contexts[0] * 3, 3)? != 0 {
                return Ok([5, 7]);
            }
            let ctx = count_ctx(counts[2], counts[3] + counts[4]);
            if symbol(d, &mut self.uni_comp_ref, ctx * 3 + 1, 3)? == 0 {
                return Ok([1, 2]);
            }
            return Ok([
                1,
                3 + symbol(d, &mut self.uni_comp_ref, contexts[4] * 3 + 2, 3)? as i8,
            ]);
        }
        let first = if symbol(d, &mut self.comp_ref, contexts[2] * 3, 3)? == 0 {
            1 + symbol(d, &mut self.comp_ref, contexts[3] * 3 + 1, 3)? as i8
        } else {
            3 + symbol(d, &mut self.comp_ref, contexts[4] * 3 + 2, 3)? as i8
        };
        let second = if symbol(d, &mut self.comp_bwd_ref, contexts[1] * 2, 3)? != 0 {
            7
        } else {
            5 + symbol(d, &mut self.comp_bwd_ref, contexts[5] * 2 + 1, 3)? as i8
        };
        Ok([first, second])
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Candidate {
    pub mvs: [[i32; 2]; 2],
    pub weight: usize,
}

pub(crate) struct MvStack {
    pub candidates: Vec<Candidate>,
    pub count: usize,
    pub new_ctx: usize,
    pub ref_ctx: usize,
    pub zero_ctx: usize,
    pub drl: Vec<usize>,
}

struct StackBuilder<'a> {
    cells: &'a [Option<MotionCell>],
    cols: usize,
    rows: usize,
    origin: [usize; 2],
    size: [usize; 2],
    refs: [i8; 2],
    global: [[i32; 2]; 2],
    global_types: [u8; 8],
    integer: bool,
    high: bool,
    candidates: Vec<Candidate>,
    new_count: usize,
}

fn has_new(mode: usize) -> bool {
    matches!(mode, 16 | 19 | 20 | 21 | 22 | NEW_NEW)
}

impl StackBuilder<'_> {
    fn cell(&self, row: isize, col: isize) -> Option<MotionCell> {
        if row < 0 || col < 0 || row >= self.rows as isize || col >= self.cols as isize {
            None
        } else {
            self.cells[row as usize * self.cols + col as usize]
        }
    }
    fn add(&mut self, row: isize, col: isize, weight: usize) -> bool {
        let Some(c) = self.cell(row, col).filter(|c| c.refs[0] > 0) else {
            return false;
        };
        let compound = self.refs[1] > 0;
        let mut found = false;
        let count = if compound { 1 } else { 2 };
        for list in 0..count {
            if compound && c.refs != self.refs || !compound && c.refs[list] != self.refs[0] {
                continue;
            }
            let mut mvs = if compound {
                c.mvs
            } else {
                [c.mvs[list], [0; 2]]
            };
            for r in 0..if compound { 2 } else { 1 } {
                if (c.mode == 15 || c.mode == GLOBAL_GLOBAL)
                    && self.global_types[self.refs[r] as usize] > 1
                    && c.width.min(c.height) >= 2
                {
                    mvs[r] = self.global[r];
                }
                mvs[r] = lower_precision(mvs[r], self.integer, self.high);
            }
            found = true;
            self.new_count += usize::from(has_new(c.mode));
            if let Some(candidate) = self.candidates.iter_mut().find(|p| p.mvs == mvs) {
                candidate.weight += weight;
            } else if self.candidates.len() < 8 {
                self.candidates.push(Candidate { mvs, weight });
            }
        }
        found
    }
    fn scan(&mut self, axis: usize, delta: isize) -> bool {
        let primary = self.origin[axis];
        let along = self.origin[1 - axis];
        let length = self.size[1 - axis];
        let end = length
            .min(if axis == 0 {
                self.cols - along
            } else {
                self.rows - along
            })
            .min(16);
        let adjusted = if delta.abs() > 1 {
            delta + (primary & 1) as isize
        } else {
            delta
        };
        let offset = if delta.abs() > 1 {
            1 - (along & 1) as isize
        } else {
            0
        };
        let mut i = 0;
        let mut found = false;
        while i < end {
            let (row, col) = if axis == 0 {
                (
                    primary as isize + adjusted,
                    along as isize + offset + i as isize,
                )
            } else {
                (
                    along as isize + offset + i as isize,
                    primary as isize + adjusted,
                )
            };
            let Some(c) = self.cell(row, col) else {
                break;
            };
            let mut len = length
                .min(if axis == 0 { c.width } else { c.height })
                .max(1);
            if adjusted.abs() > 1 {
                len = len.max(2);
            }
            if length >= 16 {
                len = len.max(4);
            }
            found |= self.add(row, col, len * 2);
            i += len;
        }
        found
    }
    fn point(&mut self, row: isize, col: isize) -> bool {
        self.add(
            self.origin[0] as isize + row,
            self.origin[1] as isize + col,
            4,
        )
    }
}

impl MvStack {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build(
        cells: &[Option<MotionCell>],
        dims: [usize; 2],
        origin: [usize; 2],
        size: [usize; 2],
        refs: [i8; 2],
        global: [[i32; 2]; 2],
        global_types: [u8; 8],
        sign_bias: [bool; 8],
        integer: bool,
        high: bool,
        field: Option<&super::temporal::MotionField>,
    ) -> Self {
        let mut b = StackBuilder {
            cells,
            cols: dims[1],
            rows: dims[0],
            origin,
            size,
            refs,
            global,
            global_types,
            integer,
            high,
            candidates: Vec::new(),
            new_count: 0,
        };
        let mut above = b.scan(0, -1);
        let mut left = b.scan(1, -1);
        if size[0].max(size[1]) <= 16 {
            above |= b.point(-1, size[1] as isize);
        }
        let close = usize::from(above) + usize::from(left);
        let nearest = b.candidates.len();
        let num_new = b.new_count;
        for c in &mut b.candidates {
            c.weight += 640;
        }
        let mut zero_ctx = 0;
        if let Some(field) = field {
            let mut sample = |dy: isize, dx: isize| {
                let row = (origin[0] as isize + dy) | 1;
                let col = (origin[1] as isize + dx) | 1;
                if row < 0 || col < 0 || row >= dims[0] as isize || col >= dims[1] as isize {
                    return;
                }
                let center = dy == 0 && dx == 0;
                if center {
                    zero_ctx = 1;
                }
                let mut mvs = [[0; 2]; 2];
                for list in 0..if refs[1] > 0 { 2 } else { 1 } {
                    let Some(mv) = field.sample(row as usize, col as usize, refs[list]) else {
                        return;
                    };
                    mvs[list] = lower_precision(mv, integer, high);
                }
                if center {
                    zero_ctx = usize::from((0..if refs[1] > 0 { 2 } else { 1 }).any(|list| {
                        (0..2).any(|axis| (mvs[list][axis] - global[list][axis]).abs() >= 16)
                    }));
                }
                if let Some(candidate) = b.candidates.iter_mut().find(|c| c.mvs == mvs) {
                    candidate.weight += 2;
                } else if b.candidates.len() < 8 {
                    b.candidates.push(Candidate { mvs, weight: 2 });
                }
            };
            for dy in (0..size[0].min(16)).step_by(if size[0] >= 16 { 4 } else { 2 }) {
                for dx in (0..size[1].min(16)).step_by(if size[1] >= 16 { 4 } else { 2 }) {
                    sample(dy as isize, dx as isize);
                }
            }
            if size[0] >= 2 && size[0] < 16 && size[1] >= 2 && size[1] < 16 {
                for (dy, dx) in [
                    (size[0] as isize, -2),
                    (size[0] as isize, size[1] as isize),
                    (size[0] as isize - 2, size[1] as isize),
                ] {
                    let row = (origin[0] & 15) as isize + dy;
                    let col = (origin[1] & 15) as isize + dx;
                    if (0..16).contains(&row) && (0..16).contains(&col) {
                        sample(dy, dx);
                    }
                }
            }
        }
        above |= b.point(-1, -1);
        above |= b.scan(0, -3);
        left |= b.scan(1, -3);
        if size[0] > 1 {
            above |= b.scan(0, -5);
        }
        if size[1] > 1 {
            left |= b.scan(1, -5);
        }
        let total = usize::from(above) + usize::from(left);
        b.candidates[..nearest].sort_by_key(|c| std::cmp::Reverse(c.weight));
        b.candidates[nearest..].sort_by_key(|c| std::cmp::Reverse(c.weight));
        if b.candidates.len() < 2 {
            let mut same: [Vec<[i32; 2]>; 2] = Default::default();
            let mut different: [Vec<[i32; 2]>; 2] = Default::default();
            let end = size[0]
                .min(size[1])
                .min(16)
                .min(dims[0] - origin[0])
                .min(dims[1] - origin[1]);
            for pass in 0..2 {
                let mut i = 0;
                while i < end && b.candidates.len() < 2 {
                    let (row, col) = if pass == 0 {
                        (origin[0] as isize - 1, (origin[1] + i) as isize)
                    } else {
                        ((origin[0] + i) as isize, origin[1] as isize - 1)
                    };
                    let Some(c) = b.cell(row, col) else {
                        break;
                    };
                    for list in 0..2 {
                        let r = c.refs[list];
                        if r <= 0 {
                            continue;
                        }
                        for dest in 0..if refs[1] > 0 { 2 } else { 1 } {
                            let mut mv = c.mvs[list];
                            if refs[1] > 0 && r == refs[dest] && same[dest].len() < 2 {
                                same[dest].push(mv);
                            } else {
                                if sign_bias[r as usize] != sign_bias[refs[dest] as usize] {
                                    mv = mv.map(|v| -v);
                                }
                                if refs[1] > 0 {
                                    if different[dest].len() < 2 {
                                        different[dest].push(mv);
                                    }
                                } else if !b.candidates.iter().any(|p| p.mvs[0] == mv) {
                                    b.candidates.push(Candidate {
                                        mvs: [mv, [0; 2]],
                                        weight: 2,
                                    });
                                }
                            }
                        }
                    }
                    i += if pass == 0 {
                        c.width.max(1)
                    } else {
                        c.height.max(1)
                    };
                }
            }
            if refs[1] > 0 {
                let mut combined = [[[0; 2]; 2]; 2];
                for list in 0..2 {
                    let choices: Vec<_> = same[list]
                        .iter()
                        .chain(&different[list])
                        .copied()
                        .take(2)
                        .collect();
                    for i in 0..2 {
                        combined[i][list] = choices.get(i).copied().unwrap_or(global[list]);
                    }
                }
                if b.candidates.len() == 1 {
                    let i = usize::from(combined[0] == b.candidates[0].mvs);
                    b.candidates.push(Candidate {
                        mvs: combined[i],
                        weight: 2,
                    });
                } else {
                    for mvs in combined {
                        b.candidates.push(Candidate { mvs, weight: 2 });
                    }
                }
            }
        }
        let count = b.candidates.len();
        let mut drl = Vec::new();
        for i in 0..count {
            let ctx = if i + 1 == count {
                0
            } else if b.candidates[i].weight < 640 {
                2
            } else {
                usize::from(b.candidates[i + 1].weight < 640)
            };
            drl.push(ctx);
            for list in 0..if refs[1] > 0 { 2 } else { 1 } {
                for axis in 0..2 {
                    let border = 128 + size[axis] as i32 * 32;
                    b.candidates[i].mvs[list][axis] = b.candidates[i].mvs[list][axis].clamp(
                        -(origin[axis] as i32 * 32) - border,
                        (dims[axis] as i32 - origin[axis] as i32 - size[axis] as i32) * 32 + border,
                    );
                }
            }
        }
        while b.candidates.len() < 2 {
            b.candidates.push(Candidate {
                mvs: global,
                weight: 0,
            });
        }
        let (new_ctx, ref_ctx) = match close {
            0 => (total.min(1), total),
            1 => (3 - num_new.min(1), 2 + total),
            _ => (5 - num_new.min(1), 5),
        };
        Self {
            candidates: b.candidates,
            count,
            new_ctx,
            ref_ctx,
            zero_ctx,
            drl,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normative_compound_mode_components() {
        let expected = [
            [13, 13],
            [14, 14],
            [13, 16],
            [16, 13],
            [14, 16],
            [16, 14],
            [15, 15],
            [16, 16],
        ];
        for (symbol, components) in expected.into_iter().enumerate() {
            let mode = 17 + symbol;
            assert_eq!(
                [component_mode(mode, 0), component_mode(mode, 1)],
                components
            );
            assert_eq!(has_new(mode), components.contains(&16));
        }
        assert_eq!(GLOBAL_GLOBAL, 17 + 6);
        assert_eq!(NEW_NEW, 17 + 7);
    }

    #[test]
    fn empty_single_stack_preserves_zero_candidate_count() {
        let stack = MvStack::build(
            &vec![None; 64],
            [8, 8],
            [0, 0],
            [4, 4],
            [1, -1],
            [[0; 2]; 2],
            [0; 8],
            [false; 8],
            false,
            false,
            None,
        );
        assert_eq!(stack.count, 0);
        assert_eq!(stack.candidates.len(), 2);
        assert_eq!((stack.new_ctx, stack.ref_ctx), (0, 0));
    }

    #[test]
    fn nearest_spatial_matches_precede_distant_candidates() {
        let mut cells = vec![None; 64];
        let mut cell = MotionCell {
            width: 2,
            height: 2,
            refs: [1, -1],
            mvs: [[8, 16], [0; 2]],
            mode: 16,
            ..Default::default()
        };
        for x in 2..4 {
            cells[8 + x] = Some(cell);
        }
        cell.mode = 13;
        cell.mvs[0] = [24, 32];
        for y in 2..4 {
            cells[y * 8 + 1] = Some(cell);
        }
        let stack = MvStack::build(
            &cells,
            [8, 8],
            [2, 2],
            [2, 2],
            [1, -1],
            [[0; 2]; 2],
            [0; 8],
            [false; 8],
            false,
            false,
            None,
        );
        assert_eq!(stack.count, 2);
        assert_eq!(stack.candidates[0].mvs[0], [8, 16]);
        assert_eq!(stack.candidates[1].mvs[0], [24, 32]);
        assert_eq!((stack.new_ctx, stack.ref_ctx), (4, 5));
        assert_eq!(stack.drl, vec![0, 0]);
    }

    #[test]
    fn normative_cdfs_reset_only_adaptation_counts() {
        let mut cdfs = InterCdfs::default();
        for row in cdfs.new_mv.chunks_exact_mut(3) {
            row[2] = 32;
        }
        let first = cdfs.new_mv[0];
        cdfs.reset_counts();
        assert_eq!(cdfs.new_mv[0], first);
        assert!(cdfs.new_mv.chunks_exact(3).all(|r| r[2] == 0));
    }
}
