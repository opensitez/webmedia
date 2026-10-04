//! Coefficient-context state and DCT coefficient entropy reconstruction,
//! sections 5.11.39 and 8.3.2.

use super::entropy::SymbolDecoder;
use super::syntax::Error;
use super::tables;

pub(crate) const TX_DIMENSIONS: [(usize, usize); 19] = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (4, 8),
    (8, 4),
    (8, 16),
    (16, 8),
    (16, 32),
    (32, 16),
    (32, 64),
    (64, 32),
    (4, 16),
    (16, 4),
    (8, 32),
    (32, 8),
    (16, 64),
    (64, 16),
];

const INTER_TX_SET1: [usize; 16] = [9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 4, 5, 3, 6, 7, 8];
const INTER_TX_SET2: [usize; 12] = [9, 10, 11, 0, 1, 2, 4, 5, 3, 6, 7, 8];
const INTER_TX_SET3: [usize; 2] = [9, 0];

pub(crate) fn tx_index(w: usize, h: usize) -> Result<usize, Error> {
    TX_DIMENSIONS
        .iter()
        .position(|&d| d == (w, h))
        .ok_or(Error::Unsupported("transform dimensions"))
}

#[derive(Clone, Default)]
struct Boundary {
    above_level: Vec<u8>,
    left_level: Vec<u8>,
    above_dc: Vec<u8>,
    left_dc: Vec<u8>,
}

#[derive(Clone)]
pub(crate) struct CoefficientState {
    boundaries: Vec<Boundary>,
    skip: Vec<u16>,
    eob: [Vec<u16>; 7],
    base_eob: Vec<u16>,
    base: Vec<u16>,
    eob_extra: Vec<u16>,
    br: Vec<u16>,
    dc_sign: [[[u16; 3]; 3]; 2],
    tx1: Vec<u16>,
    tx2: Vec<u16>,
    inter_tx1: Vec<u16>,
    inter_tx2: Vec<u16>,
    inter_tx3: Vec<u16>,
}

pub(crate) struct DecodedCoefficients {
    pub values: Vec<i32>,
    pub tx_type: usize,
}

impl CoefficientState {
    pub(crate) fn load_cdfs(&mut self, saved: &Self) {
        let boundaries = std::mem::take(&mut self.boundaries);
        *self = saved.clone();
        self.boundaries = boundaries;
        self.reset_counts();
    }

    pub(crate) fn reset_counts(&mut self) {
        fn reset(values: &mut [u16], row_size: usize) {
            for row in values.chunks_exact_mut(row_size) {
                row[row_size - 1] = 0;
            }
        }
        reset(&mut self.skip, 3);
        for (i, eob) in self.eob.iter_mut().enumerate() {
            reset(eob, i + 6);
        }
        reset(&mut self.base_eob, 4);
        reset(&mut self.base, 5);
        reset(&mut self.eob_extra, 3);
        reset(&mut self.br, 5);
        for plane in &mut self.dc_sign {
            for row in plane {
                row[2] = 0;
            }
        }
        reset(&mut self.tx1, 8);
        reset(&mut self.tx2, 6);
        reset(&mut self.inter_tx1, 17);
        reset(&mut self.inter_tx2, 13);
        reset(&mut self.inter_tx3, 3);
    }

    pub(crate) fn new(qindex: u8, plane_dims: &[(usize, usize)]) -> Self {
        let q = if qindex <= 20 {
            0
        } else if qindex <= 60 {
            1
        } else if qindex <= 120 {
            2
        } else {
            3
        };
        fn group(table: &[u16], q: usize) -> Vec<u16> {
            let n = table.len() / 4;
            table[q * n..(q + 1) * n].to_vec()
        }
        Self {
            boundaries: plane_dims
                .iter()
                .map(|&(w, h)| Boundary {
                    above_level: vec![0; w.div_ceil(4)],
                    left_level: vec![0; h.div_ceil(4)],
                    above_dc: vec![0; w.div_ceil(4)],
                    left_dc: vec![0; h.div_ceil(4)],
                })
                .collect(),
            skip: group(&tables::TXB_SKIP, q),
            eob: [
                group(&tables::EOB16, q),
                group(&tables::EOB32, q),
                group(&tables::EOB64, q),
                group(&tables::EOB128, q),
                group(&tables::EOB256, q),
                group(&tables::EOB512, q),
                group(&tables::EOB1024, q),
            ],
            base_eob: group(&tables::BASE_EOB, q),
            base: group(&tables::COEFF_BASE, q),
            eob_extra: group(&tables::EOB_EXTRA, q),
            br: group(&tables::COEFF_BR, q),
            dc_sign: [
                [[16000, 32768, 0], [13056, 32768, 0], [18816, 32768, 0]],
                [[15232, 32768, 0], [12928, 32768, 0], [17280, 32768, 0]],
            ],
            tx1: tables::INTRA_TX1.to_vec(),
            tx2: tables::INTRA_TX2.to_vec(),
            inter_tx1: super::inter_tables::TX1.to_vec(),
            inter_tx2: super::inter_tables::TX2.to_vec(),
            inter_tx3: super::inter_tables::TX3.to_vec(),
        }
    }

    pub(crate) fn reset(&mut self, plane: usize, x: usize, y: usize, w: usize, h: usize) {
        self.update(plane, x, y, w, h, 0, 0);
    }

    fn update(&mut self, plane: usize, x: usize, y: usize, w: usize, h: usize, level: u8, dc: u8) {
        let b = &mut self.boundaries[plane];
        let xe = (x + w).div_ceil(4).min(b.above_level.len());
        let ye = (y + h).div_ceil(4).min(b.left_level.len());
        b.above_level[x / 4..xe].fill(level);
        b.above_dc[x / 4..xe].fill(dc);
        b.left_level[y / 4..ye].fill(level);
        b.left_dc[y / 4..ye].fill(dc);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read(
        &mut self,
        decoder: &mut SymbolDecoder<'_>,
        plane: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        block_w: usize,
        block_h: usize,
        mode: usize,
        lossless: bool,
        reduced_tx_set: bool,
        base_q: u8,
        inter_luma_type: Option<usize>,
    ) -> Result<DecodedCoefficients, Error> {
        #[cfg(test)]
        let _measure = super::profile::measure(1);
        let tx = tx_index(w, h)?;
        let min_log = w.min(h).ilog2() as usize - 2;
        let max_log = w.max(h).ilog2() as usize - 2;
        let ctx = (min_log + max_log + 1) / 2;
        let b = &self.boundaries[plane];
        let top = &b.above_level[x / 4..((x + w) / 4).min(b.above_level.len())];
        let left = &b.left_level[y / 4..((y + h) / 4).min(b.left_level.len())];
        let dc_top = &b.above_dc[x / 4..((x + w) / 4).min(b.above_dc.len())];
        let dc_left = &b.left_dc[y / 4..((y + h) / 4).min(b.left_dc.len())];
        let skip_ctx = if plane == 0 {
            let top = top.iter().copied().max().unwrap_or(0);
            let left = left.iter().copied().max().unwrap_or(0);
            if block_w == w && block_h == h {
                0
            } else if top == 0 && left == 0 {
                1
            } else if top == 0 || left == 0 {
                2 + usize::from(top.max(left) > 3)
            } else if top.max(left) <= 3 {
                4
            } else if top.min(left) <= 3 {
                5
            } else {
                6
            }
        } else {
            let above = top.iter().chain(dc_top).copied().fold(0, |a, b| a | b) != 0;
            let left = left.iter().chain(dc_left).copied().fold(0, |a, b| a | b) != 0;
            7 + usize::from(above)
                + usize::from(left)
                + if block_w * block_h > w * h { 3 } else { 0 }
        };
        let sign_sum: i32 = dc_top
            .iter()
            .chain(dc_left)
            .map(|&v| {
                if v == 1 {
                    -1
                } else if v == 2 {
                    1
                } else {
                    0
                }
            })
            .sum();
        let start = (ctx * 13 + skip_ctx) * 3;
        if decoder.read_symbol(&mut self.skip[start..start + 3])? != 0 {
            self.update(plane, x, y, w, h, 0, 0);
            return Ok(DecodedCoefficients {
                values: vec![0; w * h],
                tx_type: 0,
            });
        }
        let mut tx_type = 0;
        if inter_luma_type.is_some() && !lossless && max_log <= 3 {
            let set = if reduced_tx_set || max_log == 3 {
                3
            } else if min_log == 2 {
                2
            } else {
                1
            };
            if plane == 0 && base_q > 0 {
                let (table, start, n, inv): (&mut [u16], usize, usize, &[usize]) = match set {
                    1 => (&mut self.inter_tx1, min_log * 17, 17, &INTER_TX_SET1),
                    2 => (&mut self.inter_tx2, 0, 13, &INTER_TX_SET2),
                    _ => (&mut self.inter_tx3, min_log * 3, 3, &INTER_TX_SET3),
                };
                tx_type = inv[decoder.read_symbol(&mut table[start..start + n])?];
            } else if plane > 0 {
                let inherited = inter_luma_type.unwrap();
                if inherited >= 16 {
                    return Err(Error::Invalid("inherited inter transform type"));
                }
                let allowed = match set {
                    1 => true,
                    2 => inherited < 12,
                    _ => inherited == 0 || inherited == 9,
                };
                tx_type = if allowed { inherited } else { 0 };
            }
        } else if inter_luma_type.is_none() && plane == 0 && !lossless && max_log < 3 && base_q > 0
        {
            let set2 = reduced_tx_set || min_log == 2;
            let (table, n, inv): (&mut [u16], usize, &[usize]) = if set2 {
                (&mut self.tx2, 6, &[9, 0, 3, 1, 2])
            } else {
                (&mut self.tx1, 8, &[9, 0, 10, 11, 3, 1, 2])
            };
            let start = (min_log * 13 + mode) * n;
            let symbol = decoder.read_symbol(&mut table[start..start + n])?;
            tx_type = inv[symbol];
        }
        if inter_luma_type.is_none() && plane > 0 && !lossless && max_log <= 3 && mode != 0 {
            // Intra sets contain only DCT at 32, and ADST/identity at smaller sizes.
            if max_log < 3 {
                tx_type = [0, 1, 2, 0, 3, 1, 2, 2, 1, 3, 1, 2, 3, 0][mode];
            }
        }
        let multi = w.min(32).ilog2() as usize + h.min(32).ilog2() as usize - 4;
        let class = match tx_type {
            10 | 12 | 14 => 2,
            11 | 13 | 15 => 1,
            _ => 0,
        };
        let ptype = usize::from(plane > 0);
        let n = 6 + multi;
        let start = if multi < 5 {
            (ptype * 2 + usize::from(class != 0)) * n
        } else {
            ptype * n
        };
        let pt = decoder.read_symbol(&mut self.eob[multi][start..start + n])? + 1;
        let mut eob = if pt < 2 { pt } else { (1usize << (pt - 2)) + 1 };
        if pt >= 3 {
            let start = ((ctx * 2 + ptype) * 9 + pt - 3) * 3;
            eob += decoder.read_symbol(&mut self.eob_extra[start..start + 3])? << (pt - 3);
            eob += decoder.read_literal((pt - 3) as u8)? as usize;
        }
        let aw = w.min(32);
        let ah = h.min(32);
        let scan: Vec<u16> = if class == 2 {
            (0..aw * ah).map(|i| i as u16).collect()
        } else if class == 1 {
            (0..aw * ah)
                .map(|i| ((i % ah) * aw + i / ah) as u16)
                .collect()
        } else {
            default_scan(aw, ah)?.to_vec()
        };
        if eob > scan.len() {
            return Err(Error::Invalid("coefficient end-of-block range"));
        }
        let mut quant = vec![0i32; aw * ah];
        for c in (0..eob).rev() {
            let pos = usize::from(scan[c]);
            let row = pos / aw;
            let col = pos % aw;
            let neighbor = |dr: usize, dc: usize, cap: i32| {
                if row + dr < ah && col + dc < aw {
                    quant[(row + dr) * aw + col + dc].min(cap)
                } else {
                    0
                }
            };
            let mut level = if c == eob - 1 {
                let ectx = if c == 0 {
                    0
                } else if c <= aw * ah / 8 {
                    1
                } else if c <= aw * ah / 4 {
                    2
                } else {
                    3
                };
                let start = ((ctx * 2 + ptype) * 4 + ectx) * 4;
                decoder.read_symbol(&mut self.base_eob[start..start + 4])? as i32 + 1
            } else {
                let offsets = match class {
                    1 => [(0, 1), (1, 0), (0, 2), (0, 3), (0, 4)],
                    2 => [(0, 1), (1, 0), (2, 0), (3, 0), (4, 0)],
                    _ => [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)],
                };
                let mag: i32 = offsets
                    .into_iter()
                    .map(|(dr, dc)| neighbor(dr, dc, 3))
                    .sum();
                let bctx = if class != 0 {
                    ((mag + 1) / 2).min(4) as usize
                        + [26, 31, 36][if class == 2 { row } else { col }.min(2)]
                } else if pos == 0 {
                    0
                } else {
                    ((mag + 1) / 2).min(4) as usize
                        + usize::from(
                            tables::BASE_CTX_OFFSET[tx * 25 + row.min(4) * 5 + col.min(4)],
                        )
                };
                let start = ((ctx * 2 + ptype) * 42 + bctx) * 5;
                decoder.read_symbol(&mut self.base[start..start + 5])? as i32
            };
            if level > 2 {
                let last = if class == 1 {
                    neighbor(0, 2, 15)
                } else if class == 2 {
                    neighbor(2, 0, 15)
                } else {
                    neighbor(1, 1, 15)
                };
                let mag =
                    ((neighbor(0, 1, 15) + neighbor(1, 0, 15) + last + 1) / 2).min(6) as usize;
                let bctx = mag
                    + if pos == 0 {
                        0
                    } else if (class == 0 && row < 2 && col < 2)
                        || (class == 1 && col == 0)
                        || (class == 2 && row == 0)
                    {
                        7
                    } else {
                        14
                    };
                let start = ((ctx.min(3) * 2 + ptype) * 21 + bctx) * 5;
                for _ in 0..4 {
                    let extra = decoder.read_symbol(&mut self.br[start..start + 5])? as i32;
                    level += extra;
                    if extra < 3 {
                        break;
                    }
                }
            }
            quant[pos] = level;
        }
        let sign_ctx = if sign_sum < 0 {
            1
        } else if sign_sum > 0 {
            2
        } else {
            0
        };
        let mut sum = 0i32;
        let mut dc_category = 0;
        for c in 0..eob {
            let pos = usize::from(scan[c]);
            let mut magnitude = quant[pos];
            let negative = if magnitude == 0 {
                false
            } else if c == 0 {
                decoder.read_symbol(&mut self.dc_sign[ptype][sign_ctx])? != 0
            } else {
                decoder.read_bool()?
            };
            if magnitude > 14 {
                let mut length = 0;
                while !decoder.read_bool()? {
                    length += 1;
                    if length > 20 {
                        return Err(Error::Invalid("coefficient Golomb length"));
                    }
                }
                magnitude = ((1u32 << length) | decoder.read_literal(length)?) as i32 + 14;
            }
            if pos == 0 && magnitude > 0 {
                dc_category = if negative { 1 } else { 2 };
            }
            magnitude &= 0xfffff;
            sum += magnitude;
            quant[pos] = if negative { -magnitude } else { magnitude };
        }
        self.update(plane, x, y, w, h, sum.min(63) as u8, dc_category);
        let mut output = vec![0; w * h];
        for row in 0..ah {
            output[row * w..row * w + aw].copy_from_slice(&quant[row * aw..(row + 1) * aw]);
        }
        Ok(DecodedCoefficients {
            values: output,
            tx_type,
        })
    }
}

fn default_scan(w: usize, h: usize) -> Result<&'static [u16], Error> {
    Ok(match (w, h) {
        (4, 4) => &tables::SCAN4,
        (8, 8) => &tables::SCAN8,
        (16, 16) => &tables::SCAN16,
        (32, 32) => &tables::SCAN32,
        (4, 8) => &tables::SCAN4X8,
        (8, 4) => &tables::SCAN8X4,
        (8, 16) => &tables::SCAN8X16,
        (16, 8) => &tables::SCAN16X8,
        (16, 32) => &tables::SCAN16X32,
        (32, 16) => &tables::SCAN32X16,
        (4, 16) => &tables::SCAN4X16,
        (16, 4) => &tables::SCAN16X4,
        (8, 32) => &tables::SCAN8X32,
        (32, 8) => &tables::SCAN32X8,
        _ => return Err(Error::Unsupported("coefficient scan dimensions")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normative_inter_transform_symbol_order() {
        assert_eq!(&INTER_TX_SET1[12..], [3, 6, 7, 8]);
        assert_eq!(&INTER_TX_SET2[8..], [3, 6, 7, 8]);
        let mut all = INTER_TX_SET1;
        all.sort_unstable();
        assert_eq!(all, std::array::from_fn::<_, 16, _>(|i| i));
        for (symbol, &ty) in INTER_TX_SET1.iter().enumerate() {
            let mut tile = vec![0; 32];
            tile[0] = (symbol * 16 + 8) as u8;
            let mut d = SymbolDecoder::new(&tile, false).unwrap();
            let mut cdf: Vec<u16> = (1..=16).map(|i| i * 2048).chain([0]).collect();
            let actual = d.read_symbol(&mut cdf).unwrap();
            assert_eq!(INTER_TX_SET1[actual], ty);
        }
    }
}
