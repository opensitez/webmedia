//! H.264 (05/2003) section 9.3.3 arithmetic decoding engine.
//!
//! Binarization, context selection, and picture reconstruction are separate
//! codec stages. Input here is already a byte-aligned RBSP slice-data region.

use super::h264::AvcError;

// Tables 9-33 and 9-34 of ITU-T H.264 (05/2003).
const RANGE_LPS: [[u16; 4]; 64] = [
    [128, 176, 208, 240],
    [128, 167, 197, 227],
    [128, 158, 187, 216],
    [123, 150, 178, 205],
    [116, 142, 169, 195],
    [111, 135, 160, 185],
    [105, 128, 152, 175],
    [100, 122, 144, 166],
    [95, 116, 137, 158],
    [90, 110, 130, 150],
    [85, 104, 123, 142],
    [81, 99, 117, 135],
    [77, 94, 111, 128],
    [73, 89, 105, 122],
    [69, 85, 100, 116],
    [66, 80, 95, 110],
    [62, 76, 90, 104],
    [59, 72, 86, 99],
    [56, 69, 81, 94],
    [53, 65, 77, 89],
    [51, 62, 73, 85],
    [48, 59, 69, 80],
    [46, 56, 66, 76],
    [43, 53, 63, 72],
    [41, 50, 59, 69],
    [39, 48, 56, 65],
    [37, 45, 54, 62],
    [35, 43, 51, 59],
    [33, 41, 48, 56],
    [32, 39, 46, 53],
    [30, 37, 43, 50],
    [29, 35, 41, 48],
    [27, 33, 39, 45],
    [26, 31, 37, 43],
    [24, 30, 35, 41],
    [23, 28, 33, 39],
    [22, 27, 32, 37],
    [21, 26, 30, 35],
    [20, 24, 29, 33],
    [19, 23, 27, 31],
    [18, 22, 26, 30],
    [17, 21, 25, 28],
    [16, 20, 23, 27],
    [15, 19, 22, 25],
    [14, 18, 21, 24],
    [14, 17, 20, 23],
    [13, 16, 19, 22],
    [12, 15, 18, 21],
    [12, 14, 17, 20],
    [11, 14, 16, 19],
    [11, 13, 15, 18],
    [10, 12, 15, 17],
    [10, 12, 14, 16],
    [9, 11, 13, 15],
    [9, 11, 12, 14],
    [8, 10, 12, 14],
    [8, 9, 11, 13],
    [7, 9, 11, 12],
    [7, 9, 10, 12],
    [7, 8, 10, 11],
    [6, 8, 9, 11],
    [6, 7, 9, 10],
    [6, 7, 8, 9],
    [2, 2, 2, 2],
];

const TRANS_LPS: [u8; 64] = [
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12, 13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21,
    21, 22, 22, 23, 24, 24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33, 33, 33, 34,
    34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

const TRANS_MPS: [u8; 64] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Context {
    pub state: u8,
    pub mps: u8,
}

impl Context {
    pub fn init(m: i32, n: i32, slice_qp: i32) -> Result<Self, AvcError> {
        if !(0..=51).contains(&slice_qp) || !(-128..=127).contains(&m) || !(-128..=127).contains(&n)
        {
            return Err(AvcError::InvalidData("CABAC initialization out of range"));
        }
        let pre = ((m * slice_qp) >> 4).saturating_add(n).clamp(1, 126);
        Ok(if pre <= 63 {
            Self {
                state: (63 - pre) as u8,
                mps: 0,
            }
        } else {
            Self {
                state: (pre - 64) as u8,
                mps: 1,
            }
        })
    }
}

pub struct CabacDecoder<'a> {
    rbsp: &'a [u8],
    bit: usize,
    range: u32,
    offset: u32,
}

/// CABAC contexts 3..10 for I-slice `mb_type` (2003 Tables 9-12 and 9-29).
pub struct IntraMbTypeContexts([Context; 8]);

/// Inter-slice CABAC contexts 11..13 (P skip), 24..26 (B skip), and 14..17 (P type).
pub struct InterMbContexts {
    skip: [Context; 3],
    p_type: [Context; 4],
}

/// CABAC contexts 40..53 for horizontal and vertical motion differences.
pub struct MotionVectorContexts([Context; 14]);

impl MotionVectorContexts {
    pub fn new(slice_qp: i32, cabac_init_idc: u32) -> Result<Self, AvcError> {
        const VALUES: [[(i32, i32); 14]; 3] = [
            [
                (-3, 69),
                (-6, 81),
                (-11, 96),
                (6, 55),
                (7, 67),
                (-5, 86),
                (2, 88),
                (0, 58),
                (-3, 76),
                (-10, 94),
                (5, 54),
                (4, 69),
                (-3, 81),
                (0, 88),
            ],
            [
                (-2, 69),
                (-5, 82),
                (-10, 96),
                (2, 59),
                (2, 75),
                (-3, 87),
                (-3, 100),
                (1, 56),
                (-3, 74),
                (-6, 85),
                (0, 59),
                (-3, 81),
                (-7, 86),
                (-5, 95),
            ],
            [
                (-11, 89),
                (-15, 103),
                (-21, 116),
                (19, 57),
                (20, 58),
                (4, 84),
                (6, 96),
                (1, 63),
                (-5, 85),
                (-13, 106),
                (5, 63),
                (6, 75),
                (-3, 90),
                (-1, 101),
            ],
        ];
        let values = VALUES
            .get(cabac_init_idc as usize)
            .ok_or(AvcError::InvalidData("CABAC initialization out of range"))?;
        let mut contexts = [Context { state: 0, mps: 0 }; 14];
        for (context, &(m, n)) in contexts.iter_mut().zip(values) {
            *context = Context::init(m, n, slice_qp)?;
        }
        Ok(Self(contexts))
    }
}

impl InterMbContexts {
    pub fn new(slice_qp: i32, cabac_init_idc: u32, is_b: bool) -> Result<Self, AvcError> {
        let init = match (cabac_init_idc, is_b) {
            (0, false) => [(23, 33), (23, 2), (21, 0)],
            (1, false) => [(22, 25), (34, 0), (16, 0)],
            (2, false) => [(29, 16), (25, 0), (14, 0)],
            (0, true) => [(18, 64), (9, 43), (29, 0)],
            (1, true) => [(26, 34), (19, 22), (40, 0)],
            (2, true) => [(20, 40), (20, 10), (29, 0)],
            _ => return Err(AvcError::InvalidData("CABAC initialization out of range")),
        };
        let p_type = match cabac_init_idc {
            0 => [(1, 9), (0, 49), (-37, 118), (5, 57)],
            1 => [(-2, 9), (4, 41), (-29, 118), (2, 65)],
            2 => [(-10, 51), (-3, 62), (-27, 99), (26, 16)],
            _ => unreachable!(),
        };
        Ok(Self {
            skip: [
                Context::init(init[0].0, init[0].1, slice_qp)?,
                Context::init(init[1].0, init[1].1, slice_qp)?,
                Context::init(init[2].0, init[2].1, slice_qp)?,
            ],
            p_type: [
                Context::init(p_type[0].0, p_type[0].1, slice_qp)?,
                Context::init(p_type[1].0, p_type[1].1, slice_qp)?,
                Context::init(p_type[2].0, p_type[2].1, slice_qp)?,
                Context::init(p_type[3].0, p_type[3].1, slice_qp)?,
            ],
        })
    }
}

/// Prediction syntax shared by 2003 Intra_4x4 and 2005 Intra_8x8.
pub struct IntraPredContexts {
    luma: [Context; 2],
    chroma: [Context; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntraPredCode {
    pub use_predicted_mode: bool,
    pub remaining_mode: Option<u8>,
}

/// Resolve the 2003 Intra_4x4 prediction codes in inverse block-scan order.
/// External edge modes are `None` when the neighbouring macroblock is unavailable.
pub fn intra4x4_modes(
    codes: &[IntraPredCode; 16],
    left_edge: Option<[u8; 4]>,
    above_edge: Option<[u8; 4]>,
) -> Result<[u8; 16], AvcError> {
    let mut modes = [0u8; 16];
    let mut grid = [[0u8; 4]; 4];
    for (block, code) in codes.iter().enumerate() {
        let region = block / 4;
        let sub = block % 4;
        let x = (region % 2) * 2 + sub % 2;
        let y = (region / 2) * 2 + sub / 2;
        let left = if x == 0 {
            left_edge.map(|edge| edge[y])
        } else {
            Some(grid[y][x - 1])
        };
        let above = if y == 0 {
            above_edge.map(|edge| edge[x])
        } else {
            Some(grid[y - 1][x])
        };
        let predicted = match (left, above) {
            (Some(a), Some(b)) if a <= 8 && b <= 8 => a.min(b),
            (None, _) | (_, None) => 2,
            _ => return Err(AvcError::InvalidData("invalid neighbouring intra mode")),
        };
        let mode = if code.use_predicted_mode {
            if code.remaining_mode.is_some() {
                return Err(AvcError::InvalidData("unexpected remaining intra mode"));
            }
            predicted
        } else {
            let rem = code
                .remaining_mode
                .ok_or(AvcError::InvalidData("missing remaining intra mode"))?;
            if rem > 7 {
                return Err(AvcError::InvalidData("remaining intra mode out of range"));
            }
            rem + u8::from(rem >= predicted)
        };
        modes[block] = mode;
        grid[y][x] = mode;
    }
    Ok(modes)
}

/// Resolve the 2005 Intra_8x8 prediction codes in 8x8 block-scan order.
/// External edge modes correspond to the top/bottom or left/right 8x8 blocks.
pub fn intra8x8_modes(
    codes: &[IntraPredCode; 4],
    left_edge: Option<[u8; 2]>,
    above_edge: Option<[u8; 2]>,
) -> Result<[u8; 4], AvcError> {
    let mut modes = [0u8; 4];
    for (block, code) in codes.iter().enumerate() {
        let x = block % 2;
        let y = block / 2;
        let left = if x == 0 {
            left_edge.map(|edge| edge[y])
        } else {
            Some(modes[block - 1])
        };
        let above = if y == 0 {
            above_edge.map(|edge| edge[x])
        } else {
            Some(modes[block - 2])
        };
        let predicted = match (left, above) {
            (Some(a), Some(b)) if a <= 8 && b <= 8 => a.min(b),
            (None, _) | (_, None) => 2,
            _ => return Err(AvcError::InvalidData("invalid neighbouring intra mode")),
        };
        modes[block] = if code.use_predicted_mode {
            if code.remaining_mode.is_some() {
                return Err(AvcError::InvalidData("unexpected remaining intra mode"));
            }
            predicted
        } else {
            let rem = code
                .remaining_mode
                .ok_or(AvcError::InvalidData("missing remaining intra mode"))?;
            if rem > 7 {
                return Err(AvcError::InvalidData("remaining intra mode out of range"));
            }
            rem + u8::from(rem >= predicted)
        };
    }
    Ok(modes)
}

impl IntraPredContexts {
    pub fn new(slice_qp: i32) -> Result<Self, AvcError> {
        Ok(Self {
            luma: [
                Context::init(13, 41, slice_qp)?,
                Context::init(3, 62, slice_qp)?,
            ],
            chroma: [
                Context::init(-9, 83, slice_qp)?,
                Context::init(4, 86, slice_qp)?,
                Context::init(0, 97, slice_qp)?,
                Context::init(-7, 72, slice_qp)?,
            ],
        })
    }
}

/// 2005 High-profile contexts 399..401 for transform_size_8x8_flag.
pub struct Transform8x8Contexts([Context; 3]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodedBlockPattern {
    pub luma: u8,
    pub chroma: u8,
    pub pcm: bool,
}

pub struct CodedBlockContexts([Context; 12]);

pub struct MbQpContexts([Context; 4]);

/// I-slice 4x4 luma residual contexts for block category 2 (2003 Tables 9-18..9-21).
pub struct Luma4x4Contexts {
    coded: [Context; 4],
    significant: [Context; 15],
    last: [Context; 15],
    magnitude: [Context; 10],
}

/// I-slice Intra16x16 luma DC contexts (block category 0, Tables 9-18..9-21).
pub struct Luma16x16DcContexts {
    coded: [Context; 4],
    significant: [Context; 15],
    last: [Context; 15],
    magnitude: [Context; 10],
}

/// I-slice Intra16x16 luma AC contexts (block category 1, Tables 9-18..9-21).
pub struct Luma16x16AcContexts {
    coded: [Context; 4],
    significant: [Context; 14],
    last: [Context; 14],
    magnitude: [Context; 10],
}

impl Luma16x16AcContexts {
    pub fn new(slice_qp: i32) -> Result<Self, AvcError> {
        fn init<const N: usize>(
            values: [(i32, i32); N],
            qp: i32,
        ) -> Result<[Context; N], AvcError> {
            let mut contexts = [Context { state: 0, mps: 0 }; N];
            for (context, (m, n)) in contexts.iter_mut().zip(values) {
                *context = Context::init(m, n, qp)?;
            }
            Ok(contexts)
        }
        Ok(Self {
            coded: init([(-12, 63), (-2, 68), (-15, 84), (-13, 104)], slice_qp)?,
            significant: init(
                [
                    (1, 50),
                    (7, 52),
                    (10, 35),
                    (0, 44),
                    (11, 38),
                    (1, 45),
                    (0, 46),
                    (5, 44),
                    (31, 17),
                    (1, 51),
                    (7, 50),
                    (28, 19),
                    (16, 33),
                    (14, 62),
                ],
                slice_qp,
            )?,
            last: init(
                [
                    (12, 38),
                    (11, 45),
                    (15, 39),
                    (11, 42),
                    (13, 44),
                    (16, 45),
                    (12, 41),
                    (10, 49),
                    (30, 34),
                    (18, 42),
                    (10, 55),
                    (17, 51),
                    (17, 46),
                    (0, 89),
                ],
                slice_qp,
            )?,
            magnitude: init(
                [
                    (-5, 67),
                    (-5, 27),
                    (-3, 39),
                    (-2, 44),
                    (0, 46),
                    (-16, 64),
                    (-8, 68),
                    (-10, 78),
                    (-6, 77),
                    (-10, 86),
                ],
                slice_qp,
            )?,
        })
    }
}

impl Luma16x16DcContexts {
    pub fn new(slice_qp: i32) -> Result<Self, AvcError> {
        fn init<const N: usize>(
            values: [(i32, i32); N],
            qp: i32,
        ) -> Result<[Context; N], AvcError> {
            let mut contexts = [Context { state: 0, mps: 0 }; N];
            for (context, (m, n)) in contexts.iter_mut().zip(values) {
                *context = Context::init(m, n, qp)?;
            }
            Ok(contexts)
        }
        Ok(Self {
            coded: init([(-17, 123), (-12, 115), (-16, 122), (-11, 115)], slice_qp)?,
            significant: init(
                [
                    (-7, 93),
                    (-11, 87),
                    (-3, 77),
                    (-5, 71),
                    (-4, 63),
                    (-4, 68),
                    (-12, 84),
                    (-7, 62),
                    (-7, 65),
                    (8, 61),
                    (5, 56),
                    (-2, 66),
                    (1, 64),
                    (0, 61),
                    (-2, 78),
                ],
                slice_qp,
            )?,
            last: init(
                [
                    (24, 0),
                    (15, 9),
                    (8, 25),
                    (13, 18),
                    (15, 9),
                    (13, 19),
                    (10, 37),
                    (12, 18),
                    (6, 29),
                    (20, 33),
                    (15, 30),
                    (4, 45),
                    (1, 58),
                    (0, 62),
                    (7, 61),
                ],
                slice_qp,
            )?,
            magnitude: init(
                [
                    (-3, 71),
                    (-6, 42),
                    (-5, 50),
                    (-3, 54),
                    (-2, 62),
                    (0, 58),
                    (1, 63),
                    (-2, 72),
                    (-1, 74),
                    (-9, 91),
                ],
                slice_qp,
            )?,
        })
    }
}

/// High-profile I-slice 8x8 luma residual contexts (2005 Table 9-24).
pub struct Luma8x8Contexts {
    significant: [Context; 15],
    last: [Context; 9],
    magnitude: [Context; 10],
}

// 2005 Table 9-34, frame-coded 8x8 luma blocks.
const SIG_8X8_MAP: [usize; 63] = [
    0, 1, 2, 3, 4, 5, 5, 4, 4, 3, 3, 4, 4, 4, 5, 5, 4, 4, 4, 4, 3, 3, 6, 7, 7, 7, 8, 9, 10, 9, 8,
    7, 7, 6, 11, 12, 13, 11, 6, 7, 8, 9, 14, 10, 9, 8, 6, 11, 12, 13, 11, 6, 9, 14, 10, 9, 11, 12,
    13, 11, 14, 10, 12,
];
const LAST_8X8_MAP: [usize; 63] = [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8,
];

impl Luma8x8Contexts {
    pub fn new(slice_qp: i32) -> Result<Self, AvcError> {
        const SIGNIFICANT: [(i32, i32); 15] = [
            (-17, 120),
            (-20, 112),
            (-18, 114),
            (-11, 85),
            (-15, 92),
            (-14, 89),
            (-26, 71),
            (-15, 81),
            (-14, 80),
            (0, 68),
            (-14, 70),
            (-24, 56),
            (-23, 68),
            (-24, 50),
            (-11, 74),
        ];
        const LAST: [(i32, i32); 9] = [
            (23, -13),
            (26, -13),
            (40, -15),
            (49, -14),
            (44, 3),
            (45, 6),
            (44, 34),
            (33, 54),
            (19, 82),
        ];
        const MAGNITUDE: [(i32, i32); 10] = [
            (-3, 75),
            (-1, 23),
            (1, 34),
            (1, 43),
            (0, 54),
            (-2, 55),
            (0, 61),
            (1, 64),
            (0, 68),
            (-9, 92),
        ];
        fn init<const N: usize>(
            values: [(i32, i32); N],
            qp: i32,
        ) -> Result<[Context; N], AvcError> {
            let mut contexts = [Context { state: 0, mps: 0 }; N];
            for (context, (m, n)) in contexts.iter_mut().zip(values) {
                *context = Context::init(m, n, qp)?;
            }
            Ok(contexts)
        }
        Ok(Self {
            significant: init(SIGNIFICANT, slice_qp)?,
            last: init(LAST, slice_qp)?,
            magnitude: init(MAGNITUDE, slice_qp)?,
        })
    }
}

/// I-slice 4:2:0 chroma DC residual contexts (2003 block category 3).
pub struct ChromaDcContexts {
    coded: [Context; 4],
    significant: [Context; 3],
    last: [Context; 3],
    magnitude: [Context; 10],
}

/// I-slice chroma AC residual contexts (block category 4, Tables 9-18..9-21).
pub struct ChromaAcContexts {
    coded: [Context; 4],
    significant: [Context; 14],
    last: [Context; 14],
    magnitude: [Context; 10],
}

impl ChromaAcContexts {
    pub fn new(slice_qp: i32) -> Result<Self, AvcError> {
        fn init<const N: usize>(
            values: [(i32, i32); N],
            qp: i32,
        ) -> Result<[Context; N], AvcError> {
            let mut contexts = [Context { state: 0, mps: 0 }; N];
            for (context, (m, n)) in contexts.iter_mut().zip(values) {
                *context = Context::init(m, n, qp)?;
            }
            Ok(contexts)
        }
        Ok(Self {
            coded: init([(-4, 56), (-5, 82), (-7, 76), (-22, 125)], slice_qp)?,
            significant: init(
                [
                    (-4, 75),
                    (2, 72),
                    (-11, 75),
                    (-3, 71),
                    (15, 46),
                    (-13, 69),
                    (0, 62),
                    (0, 65),
                    (21, 37),
                    (-15, 72),
                    (9, 57),
                    (16, 54),
                    (0, 62),
                    (12, 72),
                ],
                slice_qp,
            )?,
            last: init(
                [
                    (37, -16),
                    (35, -4),
                    (38, -8),
                    (38, -3),
                    (37, 3),
                    (38, 5),
                    (42, 0),
                    (35, 16),
                    (39, 22),
                    (14, 48),
                    (27, 37),
                    (21, 60),
                    (12, 68),
                    (2, 97),
                ],
                slice_qp,
            )?,
            magnitude: init(
                [
                    (-8, 78),
                    (-5, 33),
                    (-4, 48),
                    (-2, 53),
                    (-3, 62),
                    (-13, 71),
                    (-10, 79),
                    (-12, 86),
                    (-13, 90),
                    (-14, 97),
                ],
                slice_qp,
            )?,
        })
    }
}

impl ChromaDcContexts {
    pub fn new(slice_qp: i32) -> Result<Self, AvcError> {
        fn init<const N: usize>(
            values: [(i32, i32); N],
            qp: i32,
        ) -> Result<[Context; N], AvcError> {
            let mut contexts = [Context { state: 0, mps: 0 }; N];
            for (context, (m, n)) in contexts.iter_mut().zip(values) {
                *context = Context::init(m, n, qp)?;
            }
            Ok(contexts)
        }
        Ok(Self {
            coded: init([(-1, 74), (-6, 97), (-7, 91), (-20, 127)], slice_qp)?,
            significant: init([(-8, 102), (-15, 100), (0, 95)], slice_qp)?,
            last: init([(30, -6), (27, 3), (26, 22)], slice_qp)?,
            magnitude: init(
                [
                    (-11, 97),
                    (-20, 84),
                    (-11, 79),
                    (-6, 73),
                    (-4, 74),
                    (-13, 86),
                    (-13, 96),
                    (-11, 97),
                    (-19, 117),
                    (-8, 78),
                ],
                slice_qp,
            )?,
        })
    }

    pub fn new_inter(slice_qp: i32, cabac_init_idc: u32) -> Result<Self, AvcError> {
        const CODED: [[(i32, i32); 4]; 3] = [
            [(5, 54), (6, 60), (6, 59), (6, 69)],
            [(3, 55), (7, 56), (7, 55), (8, 61)],
            [(0, 65), (-2, 79), (0, 72), (-4, 92)],
        ];
        const SIGNIFICANT: [[(i32, i32); 3]; 3] = [
            [(3, 64), (1, 61), (9, 63)],
            [(-4, 71), (0, 58), (7, 61)],
            [(3, 65), (-7, 69), (8, 77)],
        ];
        const LAST: [[(i32, i32); 3]; 3] = [
            [(1, 67), (5, 59), (9, 67)],
            [(0, 75), (2, 72), (8, 77)],
            [(20, 34), (19, 31), (27, 44)],
        ];
        const MAGNITUDE: [[(i32, i32); 10]; 3] = [
            [
                (0, 70),
                (-4, 29),
                (5, 31),
                (7, 42),
                (1, 59),
                (-2, 58),
                (-3, 72),
                (-3, 81),
                (-11, 97),
                (0, 58),
            ],
            [
                (2, 66),
                (-9, 34),
                (1, 32),
                (11, 31),
                (5, 52),
                (-2, 55),
                (-2, 67),
                (0, 73),
                (-8, 89),
                (3, 52),
            ],
            [
                (-4, 79),
                (-22, 69),
                (-16, 75),
                (-2, 58),
                (1, 58),
                (-13, 78),
                (-9, 83),
                (-4, 81),
                (-13, 99),
                (-13, 81),
            ],
        ];
        let idc = cabac_init_idc as usize;
        if idc > 2 {
            return Err(AvcError::InvalidData("CABAC initialization out of range"));
        }
        fn init<const N: usize>(
            values: [(i32, i32); N],
            qp: i32,
        ) -> Result<[Context; N], AvcError> {
            let mut contexts = [Context { state: 0, mps: 0 }; N];
            for (context, (m, n)) in contexts.iter_mut().zip(values) {
                *context = Context::init(m, n, qp)?;
            }
            Ok(contexts)
        }
        Ok(Self {
            coded: init(CODED[idc], slice_qp)?,
            significant: init(SIGNIFICANT[idc], slice_qp)?,
            last: init(LAST[idc], slice_qp)?,
            magnitude: init(MAGNITUDE[idc], slice_qp)?,
        })
    }
}

impl Luma4x4Contexts {
    pub fn new(slice_qp: i32) -> Result<Self, AvcError> {
        const CODED: [(i32, i32); 4] = [(-3, 70), (-8, 93), (-10, 90), (-30, 127)];
        const SIGNIFICANT: [(i32, i32); 15] = [
            (-13, 108),
            (-15, 100),
            (-13, 101),
            (-13, 91),
            (-12, 94),
            (-10, 88),
            (-16, 84),
            (-10, 86),
            (-7, 83),
            (-13, 87),
            (-19, 94),
            (1, 70),
            (0, 72),
            (-5, 74),
            (18, 59),
        ];
        const LAST: [(i32, i32); 15] = [
            (26, -19),
            (22, -17),
            (26, -17),
            (30, -25),
            (28, -20),
            (33, -23),
            (37, -27),
            (33, -23),
            (40, -28),
            (38, -17),
            (33, -11),
            (40, -15),
            (41, -6),
            (38, 1),
            (41, 17),
        ];
        const MAGNITUDE: [(i32, i32); 10] = [
            (-12, 92),
            (-15, 55),
            (-10, 60),
            (-6, 62),
            (-4, 65),
            (-12, 73),
            (-8, 76),
            (-7, 80),
            (-9, 88),
            (-17, 110),
        ];
        fn init<const N: usize>(
            values: [(i32, i32); N],
            qp: i32,
        ) -> Result<[Context; N], AvcError> {
            let mut contexts = [Context { state: 0, mps: 0 }; N];
            for (context, (m, n)) in contexts.iter_mut().zip(values) {
                *context = Context::init(m, n, qp)?;
            }
            Ok(contexts)
        }
        Ok(Self {
            coded: init(CODED, slice_qp)?,
            significant: init(SIGNIFICANT, slice_qp)?,
            last: init(LAST, slice_qp)?,
            magnitude: init(MAGNITUDE, slice_qp)?,
        })
    }
}

impl MbQpContexts {
    pub fn new(slice_qp: i32) -> Result<Self, AvcError> {
        Ok(Self([
            Context::init(0, 41, slice_qp)?,
            Context::init(0, 63, slice_qp)?,
            Context::init(0, 63, slice_qp)?,
            Context::init(0, 63, slice_qp)?,
        ]))
    }
}

impl CodedBlockContexts {
    pub fn new(slice_qp: i32) -> Result<Self, AvcError> {
        const M: [i32; 12] = [-17, -13, 0, -7, -21, -27, -31, -24, -18, -27, -21, -30];
        const N: [i32; 12] = [127, 102, 82, 74, 107, 127, 127, 127, 95, 127, 114, 127];
        let mut contexts = [Context { state: 0, mps: 0 }; 12];
        for (index, context) in contexts.iter_mut().enumerate() {
            *context = Context::init(M[index], N[index], slice_qp)?;
        }
        Ok(Self(contexts))
    }

    pub fn new_inter(slice_qp: i32, cabac_init_idc: u32) -> Result<Self, AvcError> {
        const VALUES: [[(i32, i32); 12]; 3] = [
            [
                (-27, 126),
                (-28, 98),
                (-25, 101),
                (-23, 67),
                (-28, 82),
                (-20, 94),
                (-16, 83),
                (-22, 110),
                (-21, 91),
                (-18, 102),
                (-13, 93),
                (-29, 127),
            ],
            [
                (-39, 127),
                (-18, 91),
                (-17, 96),
                (-26, 81),
                (-35, 98),
                (-24, 102),
                (-23, 97),
                (-27, 119),
                (-24, 99),
                (-21, 110),
                (-18, 102),
                (-36, 127),
            ],
            [
                (-36, 127),
                (-17, 91),
                (-14, 95),
                (-25, 84),
                (-25, 86),
                (-12, 89),
                (-17, 91),
                (-31, 127),
                (-14, 76),
                (-18, 103),
                (-13, 90),
                (-37, 127),
            ],
        ];
        let values = VALUES
            .get(cabac_init_idc as usize)
            .ok_or(AvcError::InvalidData("CABAC initialization out of range"))?;
        let mut contexts = [Context { state: 0, mps: 0 }; 12];
        for (context, &(m, n)) in contexts.iter_mut().zip(values) {
            *context = Context::init(m, n, slice_qp)?;
        }
        Ok(Self(contexts))
    }
}

impl Transform8x8Contexts {
    pub fn new(slice_qp: i32) -> Result<Self, AvcError> {
        Ok(Self([
            Context::init(31, 21, slice_qp)?,
            Context::init(31, 31, slice_qp)?,
            Context::init(25, 50, slice_qp)?,
        ]))
    }
}

impl IntraMbTypeContexts {
    pub fn new(slice_qp: i32) -> Result<Self, AvcError> {
        const M: [i32; 8] = [20, 2, 3, -28, -23, -6, -1, 7];
        const N: [i32; 8] = [-15, 54, 74, 127, 104, 53, 54, 51];
        let mut contexts = [Context { state: 0, mps: 0 }; 8];
        for (index, context) in contexts.iter_mut().enumerate() {
            *context = Context::init(M[index], N[index], slice_qp)?;
        }
        Ok(Self(contexts))
    }
}

impl CabacDecoder<'_> {
    pub fn luma16x16_ac_macroblock(
        &mut self,
        contexts: &mut Luma16x16AcContexts,
        left_edge: Option<[bool; 4]>,
        above_edge: Option<[bool; 4]>,
    ) -> Result<[[i32; 15]; 16], AvcError> {
        let mut blocks = [[0; 15]; 16];
        let mut coded = [[false; 4]; 4];
        for block in 0..16 {
            let region = block / 4;
            let sub = block % 4;
            let x = (region % 2) * 2 + sub % 2;
            let y = (region / 2) * 2 + sub / 2;
            let left = if x == 0 {
                left_edge.map(|edge| edge[y])
            } else {
                Some(coded[y][x - 1])
            };
            let above = if y == 0 {
                above_edge.map(|edge| edge[x])
            } else {
                Some(coded[y - 1][x])
            };
            blocks[block] = self.residual_coefficients(
                &mut contexts.coded,
                &mut contexts.significant,
                &mut contexts.last,
                &mut contexts.magnitude,
                left,
                above,
            )?;
            coded[y][x] = blocks[block].iter().any(|&level| level != 0);
        }
        Ok(blocks)
    }

    pub fn chroma_ac_macroblock(
        &mut self,
        contexts: &mut ChromaAcContexts,
        left_edge: Option<[bool; 2]>,
        above_edge: Option<[bool; 2]>,
    ) -> Result<[[i32; 15]; 4], AvcError> {
        let mut blocks = [[0; 15]; 4];
        let mut coded = [false; 4];
        for block in 0..4 {
            let x = block % 2;
            let y = block / 2;
            let left = if x == 0 {
                left_edge.map(|edge| edge[y])
            } else {
                Some(coded[block - 1])
            };
            let above = if y == 0 {
                above_edge.map(|edge| edge[x])
            } else {
                Some(coded[block - 2])
            };
            blocks[block] = self.residual_coefficients(
                &mut contexts.coded,
                &mut contexts.significant,
                &mut contexts.last,
                &mut contexts.magnitude,
                left,
                above,
            )?;
            coded[block] = blocks[block].iter().any(|&level| level != 0);
        }
        Ok(blocks)
    }

    pub fn luma16x16_dc_coefficients(
        &mut self,
        contexts: &mut Luma16x16DcContexts,
        left_coded: Option<bool>,
        above_coded: Option<bool>,
    ) -> Result<[i32; 16], AvcError> {
        self.residual_coefficients(
            &mut contexts.coded,
            &mut contexts.significant,
            &mut contexts.last,
            &mut contexts.magnitude,
            left_coded,
            above_coded,
        )
    }

    pub fn luma4x4_macroblock(
        &mut self,
        contexts: &mut Luma4x4Contexts,
        coded_pattern_luma: u8,
        left_edge: Option<[bool; 4]>,
        above_edge: Option<[bool; 4]>,
    ) -> Result<[[i32; 16]; 16], AvcError> {
        if coded_pattern_luma > 15 {
            return Err(AvcError::InvalidData("luma coded pattern out of range"));
        }
        let mut blocks = [[0i32; 16]; 16];
        let mut coded = [[false; 4]; 4];
        for block in 0..16 {
            let region = block / 4;
            if coded_pattern_luma & (1 << region) == 0 {
                continue;
            }
            let sub = block % 4;
            let x = (region % 2) * 2 + sub % 2;
            let y = (region / 2) * 2 + sub / 2;
            let left = if x == 0 {
                left_edge.map(|edge| edge[y])
            } else {
                Some(coded[y][x - 1])
            };
            let above = if y == 0 {
                above_edge.map(|edge| edge[x])
            } else {
                Some(coded[y - 1][x])
            };
            blocks[block] = self.luma4x4_coefficients(contexts, left, above)?;
            coded[y][x] = blocks[block].iter().any(|&level| level != 0);
        }
        Ok(blocks)
    }

    pub fn luma4x4_coefficients(
        &mut self,
        contexts: &mut Luma4x4Contexts,
        left_coded: Option<bool>,
        above_coded: Option<bool>,
    ) -> Result<[i32; 16], AvcError> {
        self.residual_coefficients(
            &mut contexts.coded,
            &mut contexts.significant,
            &mut contexts.last,
            &mut contexts.magnitude,
            left_coded,
            above_coded,
        )
    }

    pub fn luma8x8_coefficients(
        &mut self,
        contexts: &mut Luma8x8Contexts,
    ) -> Result<[i32; 64], AvcError> {
        self.residual_coefficients_body(
            &mut contexts.significant,
            &mut contexts.last,
            &mut contexts.magnitude,
            Some(&SIG_8X8_MAP),
            Some(&LAST_8X8_MAP),
        )
    }

    pub fn chroma_dc_coefficients(
        &mut self,
        contexts: &mut ChromaDcContexts,
        left_coded: Option<bool>,
        above_coded: Option<bool>,
    ) -> Result<[i32; 4], AvcError> {
        self.residual_coefficients(
            &mut contexts.coded,
            &mut contexts.significant,
            &mut contexts.last,
            &mut contexts.magnitude,
            left_coded,
            above_coded,
        )
    }

    fn residual_coefficients<const N: usize>(
        &mut self,
        coded: &mut [Context; 4],
        significant_contexts: &mut [Context],
        last_contexts: &mut [Context],
        magnitude_contexts: &mut [Context; 10],
        left_coded: Option<bool>,
        above_coded: Option<bool>,
    ) -> Result<[i32; N], AvcError> {
        // For an intra block, an unavailable neighbour contributes one.
        let coded_context =
            usize::from(left_coded.unwrap_or(true)) + 2 * usize::from(above_coded.unwrap_or(true));
        if self.decision(&mut coded[coded_context])? == 0 {
            return Ok([0; N]);
        }
        self.residual_coefficients_body(
            significant_contexts,
            last_contexts,
            magnitude_contexts,
            None,
            None,
        )
    }

    fn residual_coefficients_body<const N: usize>(
        &mut self,
        significant_contexts: &mut [Context],
        last_contexts: &mut [Context],
        magnitude_contexts: &mut [Context; 10],
        significant_map: Option<&[usize]>,
        last_map: Option<&[usize]>,
    ) -> Result<[i32; N], AvcError> {
        let mut levels = [0i32; N];
        let mut significant = [false; N];
        let mut last_position = N - 1;
        significant[N - 1] = true;
        for position in 0..N - 1 {
            let sig_context = significant_map.map_or(position, |map| map[position]);
            let last_context = last_map.map_or(position, |map| map[position]);
            if self.decision(&mut significant_contexts[sig_context])? != 0 {
                significant[position] = true;
                if self.decision(&mut last_contexts[last_context])? != 0 {
                    significant[N - 1] = false;
                    last_position = position;
                    break;
                }
            }
        }
        let mut count_eq1 = 0usize;
        let mut count_gt1 = 0usize;
        for position in (0..=last_position).rev() {
            if !significant[position] {
                continue;
            }
            let first_context = if count_gt1 != 0 {
                0
            } else {
                (1 + count_eq1).min(4)
            };
            let later_context = 5 + count_gt1.min(4);
            let mut magnitude_minus1 = 0u32;
            loop {
                let context = if magnitude_minus1 == 0 {
                    first_context
                } else {
                    later_context
                };
                if self.decision(&mut magnitude_contexts[context])? == 0 {
                    break;
                }
                magnitude_minus1 += 1;
                if magnitude_minus1 == 14 {
                    let mut extra_bits = 0u32;
                    while self.bypass()? != 0 {
                        extra_bits += 1;
                        if extra_bits > 20 {
                            return Err(AvcError::InvalidData("CABAC coefficient too large"));
                        }
                    }
                    let mut suffix = 0u32;
                    for _ in 0..extra_bits {
                        suffix = (suffix << 1) | u32::from(self.bypass()?);
                    }
                    magnitude_minus1 += ((1u32 << extra_bits) - 1) + suffix;
                    break;
                }
            }
            if magnitude_minus1 > 1 << 20 {
                return Err(AvcError::InvalidData("CABAC coefficient too large"));
            }
            let magnitude = (magnitude_minus1 + 1) as i32;
            levels[position] = if self.bypass()? == 0 {
                magnitude
            } else {
                -magnitude
            };
            if magnitude == 1 {
                count_eq1 += 1;
            } else {
                count_gt1 += 1;
            }
        }
        Ok(levels)
    }

    pub fn mb_qp_delta(
        &mut self,
        contexts: &mut MbQpContexts,
        previous_nonzero: bool,
    ) -> Result<i32, AvcError> {
        let first_context = usize::from(previous_nonzero);
        if self.decision(&mut contexts.0[first_context])? == 0 {
            return Ok(0);
        }
        let mut mapped = 1i32;
        loop {
            let context = if mapped == 1 { 2 } else { 3 };
            if self.decision(&mut contexts.0[context])? == 0 {
                return Ok(if mapped & 1 == 1 {
                    (mapped + 1) / 2
                } else {
                    -mapped / 2
                });
            }
            mapped += 1;
            if mapped > 104 {
                return Err(AvcError::InvalidData("macroblock QP delta out of range"));
            }
        }
    }

    pub fn coded_block_pattern(
        &mut self,
        contexts: &mut CodedBlockContexts,
        left: Option<CodedBlockPattern>,
        above: Option<CodedBlockPattern>,
    ) -> Result<CodedBlockPattern, AvcError> {
        let mut luma = 0u8;
        for block in 0..4 {
            let x = block & 1;
            let y = block >> 1;
            let a = if x == 0 {
                left.filter(|neighbor| !neighbor.pcm)
                    .map(|neighbor| (neighbor.luma >> (y * 2 + 1)) & 1 == 0)
                    .unwrap_or(false)
            } else {
                luma & (1 << (block - 1)) == 0
            };
            let b = if y == 0 {
                above
                    .filter(|neighbor| !neighbor.pcm)
                    .map(|neighbor| (neighbor.luma >> (2 + x)) & 1 == 0)
                    .unwrap_or(false)
            } else {
                luma & (1 << (block - 2)) == 0
            };
            let context = usize::from(a) + 2 * usize::from(b);
            luma |= self.decision(&mut contexts.0[context])? << block;
        }
        let mut chroma = 0;
        for bin in 0..2 {
            let a = left.is_some_and(|neighbor| {
                neighbor.pcm
                    || if bin == 0 {
                        neighbor.chroma != 0
                    } else {
                        neighbor.chroma == 2
                    }
            });
            let b = above.is_some_and(|neighbor| {
                neighbor.pcm
                    || if bin == 0 {
                        neighbor.chroma != 0
                    } else {
                        neighbor.chroma == 2
                    }
            });
            let context = 4 + bin * 4 + usize::from(a) + 2 * usize::from(b);
            if self.decision(&mut contexts.0[context])? == 0 {
                break;
            }
            chroma += 1;
        }
        Ok(CodedBlockPattern {
            luma,
            chroma,
            pcm: false,
        })
    }

    pub fn transform_size_8x8_flag(
        &mut self,
        contexts: &mut Transform8x8Contexts,
        left: Option<bool>,
        above: Option<bool>,
    ) -> Result<bool, AvcError> {
        let index = usize::from(left.unwrap_or(false)) + usize::from(above.unwrap_or(false));
        Ok(self.decision(&mut contexts.0[index])? != 0)
    }

    pub fn intra_luma_pred_code(
        &mut self,
        contexts: &mut IntraPredContexts,
    ) -> Result<IntraPredCode, AvcError> {
        if self.decision(&mut contexts.luma[0])? != 0 {
            return Ok(IntraPredCode {
                use_predicted_mode: true,
                remaining_mode: None,
            });
        }
        let mut remaining = 0;
        for bit in 0..3 {
            remaining |= self.decision(&mut contexts.luma[1])? << bit;
        }
        Ok(IntraPredCode {
            use_predicted_mode: false,
            remaining_mode: Some(remaining),
        })
    }

    pub fn intra_chroma_pred_mode(
        &mut self,
        contexts: &mut IntraPredContexts,
        left: Option<u8>,
        above: Option<u8>,
    ) -> Result<u8, AvcError> {
        let index = usize::from(left.is_some_and(|mode| mode != 0))
            + usize::from(above.is_some_and(|mode| mode != 0));
        if self.decision(&mut contexts.chroma[index])? == 0 {
            return Ok(0);
        }
        for mode in 1..=2 {
            if self.decision(&mut contexts.chroma[3])? == 0 {
                return Ok(mode);
            }
        }
        Ok(3)
    }

    /// Decodes Table 9-26's I-slice macroblock type (0..=25).
    /// Neighbours must be available macroblocks from this slice; an I_4x4
    /// neighbour contributes zero, while any other type contributes one.
    pub fn intra_mb_type(
        &mut self,
        contexts: &mut IntraMbTypeContexts,
        left: Option<u8>,
        above: Option<u8>,
    ) -> Result<u8, AvcError> {
        const CODES: [&str; 26] = [
            "0", "100000", "100001", "100010", "100011", "1001000", "1001001", "1001010",
            "1001011", "1001100", "1001101", "1001110", "1001111", "101000", "101001", "101010",
            "101011", "1011000", "1011001", "1011010", "1011011", "1011100", "1011101", "1011110",
            "1011111", "11",
        ];
        let first_context = usize::from(left.is_some_and(|kind| kind != 0))
            + usize::from(above.is_some_and(|kind| kind != 0));
        let first = self.decision(&mut contexts.0[first_context])?;
        if first == 0 {
            return Ok(0);
        }
        if self.terminate()? {
            return Ok(25);
        }
        let mut code = String::from("10");
        while code.len() < 7 {
            let bin_idx = code.len();
            let ctx_idx = match bin_idx {
                2 => 6,
                3 => 7,
                4 => {
                    if code.as_bytes()[3] == b'1' {
                        8
                    } else {
                        9
                    }
                }
                5 => {
                    if code.as_bytes()[3] == b'1' {
                        9
                    } else {
                        10
                    }
                }
                _ => 10,
            };
            let bit = self.decision(&mut contexts.0[ctx_idx - 3])?;
            code.push(char::from(b'0' + bit));
            if let Some(kind) = CODES.iter().position(|candidate| *candidate == code) {
                return Ok(kind as u8);
            }
            if !CODES.iter().any(|candidate| candidate.starts_with(&code)) {
                return Err(AvcError::InvalidData("invalid CABAC I macroblock type"));
            }
        }
        Err(AvcError::InvalidData("CABAC I macroblock type too long"))
    }
}

impl<'a> CabacDecoder<'a> {
    pub fn motion_vector_difference(
        &mut self,
        contexts: &mut MotionVectorContexts,
        component: usize,
        left: Option<i32>,
        above: Option<i32>,
    ) -> Result<i32, AvcError> {
        if component > 1 {
            return Err(AvcError::InvalidData("invalid motion vector component"));
        }
        let neighbours = left
            .unwrap_or(0)
            .unsigned_abs()
            .saturating_add(above.unwrap_or(0).unsigned_abs());
        let initial = if neighbours < 3 {
            0
        } else if neighbours > 32 {
            2
        } else {
            1
        };
        let base = component * 7;
        let mut magnitude = 0i32;
        for bin in 0..9 {
            let context = if bin == 0 { initial } else { (bin + 2).min(6) };
            if self.decision(&mut contexts.0[base + context])? == 0 {
                break;
            }
            magnitude += 1;
        }
        if magnitude == 9 {
            let mut order = 3u32;
            loop {
                if order > 15 {
                    return Err(AvcError::TooLarge);
                }
                if self.bypass()? == 0 {
                    let mut suffix = 0i32;
                    for _ in 0..order {
                        suffix = (suffix << 1) | i32::from(self.bypass()?);
                    }
                    magnitude += suffix;
                    break;
                }
                magnitude += 1 << order;
                order += 1;
            }
        }
        if magnitude != 0 && self.bypass()? != 0 {
            magnitude = -magnitude;
        }
        Ok(magnitude)
    }

    pub fn inter_mb_skip_flag(
        &mut self,
        contexts: &mut InterMbContexts,
        left_skipped: Option<bool>,
        above_skipped: Option<bool>,
    ) -> Result<bool, AvcError> {
        let index =
            usize::from(left_skipped == Some(false)) + usize::from(above_skipped == Some(false));
        Ok(self.decision(&mut contexts.skip[index])? != 0)
    }

    /// The inter-only prefix of Table 9-28; intra types use a separate suffix.
    pub fn p_inter_mb_type(&mut self, contexts: &mut InterMbContexts) -> Result<u8, AvcError> {
        if self.decision(&mut contexts.p_type[0])? != 0 {
            return Err(AvcError::Unsupported("intra macroblock in P slice"));
        }
        let second = self.decision(&mut contexts.p_type[1])?;
        let third = self.decision(&mut contexts.p_type[2 + second as usize])?;
        Ok(match (second, third) {
            (0, 0) => 0,
            (0, 1) => 3,
            (1, 0) => 2,
            (1, 1) => 1,
            _ => unreachable!(),
        })
    }

    pub fn new(rbsp: &'a [u8]) -> Result<Self, AvcError> {
        let mut decoder = Self {
            rbsp,
            bit: 0,
            range: 0x1fe,
            offset: 0,
        };
        for _ in 0..9 {
            decoder.offset = (decoder.offset << 1) | u32::from(decoder.read_bit()?);
        }
        Ok(decoder)
    }

    fn read_bit(&mut self) -> Result<u8, AvcError> {
        let byte = *self.rbsp.get(self.bit / 8).ok_or(AvcError::Incomplete)?;
        let bit = (byte >> (7 - self.bit % 8)) & 1;
        self.bit += 1;
        Ok(bit)
    }

    fn renormalize(&mut self) -> Result<(), AvcError> {
        while self.range < 0x100 {
            self.range <<= 1;
            self.offset = (self.offset << 1) | u32::from(self.read_bit()?);
        }
        Ok(())
    }

    pub fn decision(&mut self, context: &mut Context) -> Result<u8, AvcError> {
        if context.state >= 64 || context.mps > 1 {
            return Err(AvcError::InvalidData("invalid CABAC context"));
        }
        let index = ((self.range >> 6) & 3) as usize;
        let lps_range = u32::from(RANGE_LPS[context.state as usize][index]);
        self.range -= lps_range;
        let value = if self.offset >= self.range {
            self.offset -= self.range;
            self.range = lps_range;
            let value = 1 - context.mps;
            if context.state == 0 {
                context.mps ^= 1;
            }
            context.state = TRANS_LPS[context.state as usize];
            value
        } else {
            context.state = TRANS_MPS[context.state as usize];
            context.mps
        };
        self.renormalize()?;
        Ok(value)
    }

    pub fn bypass(&mut self) -> Result<u8, AvcError> {
        self.offset = (self.offset << 1) | u32::from(self.read_bit()?);
        if self.offset >= self.range {
            self.offset -= self.range;
            Ok(1)
        } else {
            Ok(0)
        }
    }

    pub fn terminate(&mut self) -> Result<bool, AvcError> {
        self.range -= 2;
        if self.offset >= self.range {
            Ok(true)
        } else {
            self.renormalize()?;
            Ok(false)
        }
    }

    pub fn consumed_bits(&self) -> usize {
        self.bit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intra4x4_modes_use_dc_for_missing_neighbours() {
        let mut codes = [IntraPredCode {
            use_predicted_mode: true,
            remaining_mode: None,
        }; 16];
        assert_eq!(intra4x4_modes(&codes, None, None).unwrap(), [2; 16]);
        codes[3] = IntraPredCode {
            use_predicted_mode: false,
            remaining_mode: Some(2),
        };
        let modes = intra4x4_modes(&codes, None, None).unwrap();
        assert_eq!(modes[3], 3);
        assert!(modes.iter().all(|&mode| mode <= 8));
    }

    #[test]
    fn intra8x8_modes_use_2005_neighbour_derivation() {
        let codes = [IntraPredCode {
            use_predicted_mode: true,
            remaining_mode: None,
        }; 4];
        assert_eq!(intra8x8_modes(&codes, None, None).unwrap(), [2; 4]);
        assert_eq!(
            intra8x8_modes(&codes, Some([1, 3]), Some([0, 4])).unwrap(),
            [0, 0, 0, 0]
        );
    }

    #[test]
    fn context_initialization_uses_original_pre_state_rule() {
        assert_eq!(
            Context::init(20, -15, 26).unwrap(),
            Context { state: 46, mps: 0 }
        );
        assert_eq!(
            Context::init(0, 80, 26).unwrap(),
            Context { state: 16, mps: 1 }
        );
        assert_eq!(
            Context::init(0, 0, 52),
            Err(AvcError::InvalidData("CABAC initialization out of range"))
        );
    }

    #[test]
    fn decision_switches_mps_at_state_zero() {
        let mut decoder = CabacDecoder::new(&[0xc8, 0x00]).unwrap();
        let mut context = Context { state: 0, mps: 0 };
        assert_eq!(decoder.decision(&mut context).unwrap(), 1);
        assert_eq!(context, Context { state: 0, mps: 1 });
        assert_eq!(decoder.consumed_bits(), 10);
    }

    #[test]
    fn bypass_and_termination_use_separate_arithmetic_paths() {
        let mut decoder = CabacDecoder::new(&[0x00, 0x00]).unwrap();
        assert_eq!(decoder.bypass().unwrap(), 0);
        assert!(!decoder.terminate().unwrap());
        assert_eq!(decoder.consumed_bits(), 10);
        assert!(matches!(CabacDecoder::new(&[0]), Err(AvcError::Incomplete)));
    }
}
