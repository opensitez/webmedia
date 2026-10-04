//! Saved and projected motion fields, AV1 sections 7.9 and 7.19.

use super::inter::MotionCell;
use super::syntax::{IntraFrameHeader, SequenceHeader, relative_dist};

const INVALID: [i32; 2] = [-32768; 2];
const DIV_MULT: [i32; 32] = [
    0, 16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638, 1489, 1365, 1260, 1170, 1092,
    1024, 963, 910, 862, 819, 780, 744, 712, 682, 655, 630, 606, 585, 564, 546, 528,
];

#[derive(Clone)]
pub(crate) struct SavedMotion {
    dims: [usize; 2],
    kind: u8,
    hint: u32,
    hints: [u32; 8],
    cells: Vec<Option<(usize, [i32; 2])>>,
}

impl SavedMotion {
    pub(crate) fn capture(
        s: &SequenceHeader,
        h: &IntraFrameHeader,
        cells: &[Option<MotionCell>],
        hints: [u32; 8],
    ) -> Self {
        let dims = [h.height.div_ceil(8) as usize, h.width.div_ceil(8) as usize];
        let mut saved = Vec::with_capacity(dims[0] * dims[1]);
        for y in 0..dims[0] {
            for x in 0..dims[1] {
                let mut selected = None;
                if let Some(cell) = cells[(y * 2 + 1) * dims[1] * 2 + x * 2 + 1] {
                    for list in 0..2 {
                        let r = cell.refs[list];
                        if r > 0
                            && relative_dist(s, hints[r as usize], h.order_hint) < 0
                            && cell.mvs[list].iter().all(|v| v.abs() <= 4095)
                        {
                            selected = Some((r as usize, cell.mvs[list]));
                        }
                    }
                }
                saved.push(selected);
            }
        }
        Self {
            dims,
            kind: h.frame_type,
            hint: h.order_hint,
            hints,
            cells: saved,
        }
    }
}

pub(crate) struct MotionField {
    dims: [usize; 2],
    cells: Vec<[[i32; 2]; 7]>,
}

fn project_mv(mv: [i32; 2], numerator: i32, denominator: i32) -> [i32; 2] {
    let factor =
        i64::from(numerator.clamp(-31, 31)) * i64::from(DIV_MULT[denominator.min(31) as usize]);
    mv.map(|v| {
        let scaled = i64::from(v) * factor;
        (scaled.signum() * ((scaled.abs() + 8192) >> 14)).clamp(-16383, 16383) as i32
    })
}

fn position(v: usize, delta: i32, sign: i32, bound: usize, offset: i32) -> Option<usize> {
    let base = (v & !7) as i32;
    let target = v as i32 + sign * (delta / 64);
    (target >= 0 && target < bound as i32 && target >= base - offset && target < base + 8 + offset)
        .then_some(target as usize)
}

impl MotionField {
    pub(crate) fn new(
        s: &SequenceHeader,
        h: &IntraFrameHeader,
        refs: [Option<&SavedMotion>; 7],
    ) -> Self {
        let dims = [h.height.div_ceil(8) as usize, h.width.div_ceil(8) as usize];
        let mut field = Self {
            dims,
            cells: vec![[INVALID; 7]; dims[0] * dims[1]],
        };
        let hints = refs.map(|r| r.map_or(0, |r| r.hint));
        let mut project = |src: usize, sign: i32| -> bool {
            let Some(reference) = refs[src] else {
                return false;
            };
            if reference.dims != dims || matches!(reference.kind, 0 | 2) {
                return false;
            }
            let to_current = relative_dist(s, reference.hint, h.order_hint);
            for y in 0..dims[0] {
                for x in 0..dims[1] {
                    let Some((r, mv)) = reference.cells[y * dims[1] + x] else {
                        continue;
                    };
                    let distance = relative_dist(s, reference.hint, reference.hints[r]);
                    if to_current.abs() > 31 || distance <= 0 || distance > 31 {
                        continue;
                    }
                    let projected = project_mv(mv, to_current * sign, distance);
                    let Some(py) = position(y, projected[0], sign, dims[0], 0) else {
                        continue;
                    };
                    let Some(px) = position(x, projected[1], sign, dims[1], 8) else {
                        continue;
                    };
                    for dst in 0..7 {
                        field.cells[py * dims[1] + px][dst] =
                            project_mv(mv, relative_dist(s, h.order_hint, hints[dst]), distance);
                    }
                }
            }
            true
        };
        if refs[0].is_some_and(|r| r.hints[7] != hints[3]) {
            project(0, -1);
        }
        let mut stamp = 1;
        for src in [4, 5, 6] {
            if relative_dist(s, hints[src], h.order_hint) > 0
                && (src != 6 || stamp >= 0)
                && project(src, 1)
            {
                stamp -= 1;
            }
        }
        if stamp >= 0 {
            project(1, -1);
        }
        field
    }

    pub(crate) fn sample(&self, row: usize, col: usize, reference: i8) -> Option<[i32; 2]> {
        if reference <= 0 || row / 2 >= self.dims[0] || col / 2 >= self.dims[1] {
            return None;
        }
        let mv = self.cells[(row / 2) * self.dims[1] + col / 2][reference as usize - 1];
        (mv != INVALID).then_some(mv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_projection_and_superblock_bounds() {
        assert_eq!(project_mv([64, -128], 3, 2), [96, -192]);
        assert_eq!(project_mv([1, -1], 1, 2), [1, -1]);
        assert_eq!(position(8, -63, 1, 32, 0), Some(8));
        assert_eq!(position(8, -64, 1, 32, 0), None);
        assert_eq!(position(8, -64, 1, 32, 8), Some(7));
        assert_eq!(project_mv([20000, -20000], 31, 1), [16383, -16383]);
    }
}
