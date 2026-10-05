//! Coefficient-context state and DCT coefficient entropy reconstruction,
//! sections 5.11.39 and 8.3.2.

use super::entropy::SymbolDecoder;
use super::syntax::Error;
use super::tables;
use std::sync::OnceLock;

#[cfg(test)]
thread_local! {
    static ALLOCATION_REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn set_allocation_reference(enabled: bool) {
    ALLOCATION_REFERENCE.set(enabled);
}

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
#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
struct Boundary {
    above_level: Vec<u8>,
    left_level: Vec<u8>,
    above_dc: Vec<u8>,
    left_dc: Vec<u8>,
}

#[derive(Clone)]
#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
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
    #[cfg(test)]
    pub scan_prefix: &'static [u16],
}

impl CoefficientState {
    pub(crate) fn snapshot_cdfs(&self) -> Self {
        #[cfg(test)]
        if super::decoder::lifecycle_reference() {
            return self.clone();
        }
        // Neighbor contexts belong to the tile, not the saved frame CDFs.
        Self {
            boundaries: Vec::new(),
            skip: self.skip.clone(),
            eob: self.eob.clone(),
            base_eob: self.base_eob.clone(),
            base: self.base.clone(),
            eob_extra: self.eob_extra.clone(),
            br: self.br.clone(),
            dc_sign: self.dc_sign,
            tx1: self.tx1.clone(),
            tx2: self.tx2.clone(),
            inter_tx1: self.inter_tx1.clone(),
            inter_tx2: self.inter_tx2.clone(),
            inter_tx3: self.inter_tx3.clone(),
        }
    }

    pub(crate) fn load_cdfs(&mut self, saved: &Self) {
        #[cfg(test)]
        if super::decoder::lifecycle_reference() {
            let boundaries = std::mem::take(&mut self.boundaries);
            *self = saved.clone();
            self.boundaries = boundaries;
            self.reset_counts();
            return;
        }
        self.skip.clone_from(&saved.skip);
        for (target, source) in self.eob.iter_mut().zip(&saved.eob) {
            target.clone_from(source);
        }
        self.base_eob.clone_from(&saved.base_eob);
        self.base.clone_from(&saved.base);
        self.eob_extra.clone_from(&saved.eob_extra);
        self.br.clone_from(&saved.br);
        self.dc_sign = saved.dc_sign;
        self.tx1.clone_from(&saved.tx1);
        self.tx2.clone_from(&saved.tx2);
        self.inter_tx1.clone_from(&saved.inter_tx1);
        self.inter_tx2.clone_from(&saved.inter_tx2);
        self.inter_tx3.clone_from(&saved.inter_tx3);
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
        self.read_impl::<false>(decoder, plane, x, y, w, h, block_w, block_h, mode,
            lossless, reduced_tx_set, base_q, inter_luma_type, &mut Vec::new(), &mut Vec::new())
    }

    #[cfg(test)]
    // Recycle returned values into quant_storage for <=32 axes, output_storage otherwise.
    pub(crate) fn read_into(
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
        quant_storage: &mut Vec<i32>,
        output_storage: &mut Vec<i32>,
    ) -> Result<DecodedCoefficients, Error> {
        self.read_impl::<true>(decoder, plane, x, y, w, h, block_w, block_h, mode,
            lossless, reduced_tx_set, base_q, inter_luma_type, quant_storage, output_storage)
    }

    fn read_impl<const REUSE: bool>(
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
        quant_storage: &mut Vec<i32>,
        output_storage: &mut Vec<i32>,
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
                values: zeroed_coefficients::<REUSE>(
                    if w <= 32 && h <= 32 { quant_storage } else { output_storage }, w * h),
                tx_type: 0,
                #[cfg(test)]
                scan_prefix: &[],
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
        let scan = coefficient_scan(class, aw, ah)?;
        #[cfg(test)]
        let active_scan = scan;
        #[cfg(test)]
        let reference_scan = ALLOCATION_REFERENCE.get().then(|| scan.to_vec());
        #[cfg(test)]
        let scan = reference_scan.as_deref().unwrap_or(scan);
        if eob > scan.len() {
            return Err(Error::Invalid("coefficient end-of-block range"));
        }
        let mut quant = zeroed_coefficients::<REUSE>(quant_storage, aw * ah);
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
        let reuse_quant = w == aw && h == ah;
        #[cfg(test)]
        let reuse_quant = reuse_quant && !ALLOCATION_REFERENCE.get();
        if reuse_quant {
            return Ok(DecodedCoefficients {
                values: quant,
                tx_type,
                #[cfg(test)]
                scan_prefix: &active_scan[..eob],
            });
        }
        let mut output = zeroed_coefficients::<REUSE>(output_storage, w * h);
        for row in 0..ah {
            output[row * w..row * w + aw].copy_from_slice(&quant[row * aw..(row + 1) * aw]);
        }
        if REUSE {
            *quant_storage = quant;
        }
        Ok(DecodedCoefficients {
            values: output,
            tx_type,
            #[cfg(test)]
            scan_prefix: &active_scan[..eob],
        })
    }
}

#[inline(always)]
fn zeroed_coefficients<const REUSE: bool>(storage: &mut Vec<i32>, length: usize) -> Vec<i32> {
    if REUSE {
        let mut values = std::mem::take(storage);
        values.resize(length, 0);
        values.fill(0);
        values
    } else {
        vec![0; length]
    }
}

pub(crate) fn coefficient_scan(class: usize, w: usize, h: usize) -> Result<&'static [u16], Error> {
    static ROW: [u16; 1024] = {
        let mut scan = [0; 1024];
        let mut i = 0;
        while i < scan.len() {
            scan[i] = i as u16;
            i += 1;
        }
        scan
    };
    // Each legal column scan is initialized once; default scans already live in tables.
    static COLUMN: [OnceLock<Vec<u16>>; 16] = [const { OnceLock::new() }; 16];
    match class {
        2 => Ok(&ROW[..w * h]),
        1 => {
            let index = (w.ilog2() as usize - 2) * 4 + h.ilog2() as usize - 2;
            Ok(COLUMN[index]
                .get_or_init(|| (0..w * h).map(|i| ((i % h) * w + i / h) as u16).collect()))
        }
        _ => default_scan(w, h),
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
    fn reused_coefficients_preserve_clear_layout_errors_cursor_and_all_contexts() {
        let mut random = 0x6512_98abu32;
        let mut expected_state = CoefficientState::new(64, &[(64, 64); 3]);
        let mut actual_state = expected_state.clone();
        let mut quant = vec![i32::MAX; 4096];
        let mut padded = vec![i32::MIN; 4096];
        let mut successes = 0;
        let mut failures = 0;
        let mut zero_results = 0;
        let mut padded_nonzero_results = 0;
        for trial in 0..12 {
            for (w, h) in TX_DIMENSIONS {
                for plane in 0..3 {
                    let length = [8192, 8192, 8192, 2, 4, 17][trial % 6];
                    let mut tile = vec![0; length];
                    if trial % 6 != 0 {
                        for byte in &mut tile {
                            random ^= random << 13;
                            random ^= random >> 17;
                            random ^= random << 5;
                            *byte = random as u8;
                        }
                    }
                    let mut expected_decoder = SymbolDecoder::new(&tile, true).unwrap();
                    let mut actual_decoder = SymbolDecoder::new(&tile, true).unwrap();
                    quant.fill(i32::MAX);
                    padded.fill(i32::MIN);
                    let inherited = (trial % 2 != 0).then_some(trial % 16);
                    let expected = expected_state.read(
                        &mut expected_decoder, plane, 0, 0, w, h, 64, 64,
                        trial % 14, false, trial % 3 == 0, 128, inherited,
                    );
                    let actual = actual_state.read_into(
                        &mut actual_decoder, plane, 0, 0, w, h, 64, 64,
                        trial % 14, false, trial % 3 == 0, 128, inherited,
                        &mut quant, &mut padded,
                    );
                    match (expected, actual) {
                        (Ok(expected), Ok(actual)) => {
                            successes += 1;
                            assert_eq!(actual.tx_type, expected.tx_type);
                            assert_eq!(actual.scan_prefix, expected.scan_prefix);
                            assert_eq!(actual.values, expected.values,
                                "{w}x{h} plane={plane} trial={trial}");
                            let zero = actual.values.iter().all(|&value| value == 0);
                            let mut active = vec![false; w * h];
                            for &position in actual.scan_prefix {
                                let position = usize::from(position);
                                active[(position / w.min(32)) * w + position % w.min(32)] = true;
                            }
                            for (index, &value) in actual.values.iter().enumerate() {
                                if !active[index] { assert_eq!(value, 0); }
                            }
                            zero_results += usize::from(zero);
                            if w > 32 || h > 32 {
                                padded_nonzero_results += usize::from(!zero);
                                for y in 0..h {
                                    for x in 0..w {
                                        if x >= 32 || y >= 32 {
                                            assert_eq!(actual.values[y * w + x], 0);
                                        }
                                    }
                                }
                                padded = actual.values;
                            } else {
                                quant = actual.values;
                            }
                        }
                        (Err(expected), Err(actual)) => {
                            failures += 1;
                            assert_eq!(actual, expected);
                        }
                        _ => panic!("reused coefficients changed success/error boundary"),
                    }
                    assert_eq!(actual_state, expected_state,
                        "{w}x{h} plane={plane} trial={trial}: complete CDF/boundary state");
                    for _ in 0..4 {
                        assert_eq!(actual_decoder.read_literal(8), expected_decoder.read_literal(8));
                    }
                }
            }
        }
        assert!(successes > 0 && failures > 0);
        assert!(zero_results > 0 && padded_nonzero_results > 0);
    }

    #[test]
    fn reused_coefficients_keep_storage_and_clear_grown_shrunk_prefixes() {
        let mut storage = Vec::with_capacity(4096);
        let pointer = storage.as_ptr();
        for length in [1024, 16, 4096, 64, 256, 0, 32] {
            let mut values = zeroed_coefficients::<true>(&mut storage, length);
            assert_eq!(values.as_ptr(), pointer);
            assert_eq!(values.len(), length);
            assert!(values.iter().all(|&value| value == 0));
            values.fill(-197);
            storage = values;
        }
    }

    #[test]
    fn cached_scans_match_original_for_every_transform_and_class() {
        for (w, h) in TX_DIMENSIONS {
            let (w, h) = (w.min(32), h.min(32));
            for class in 0..3 {
                let original: Vec<u16> = match class {
                    2 => (0..w * h).map(|i| i as u16).collect(),
                    1 => (0..w * h).map(|i| ((i % h) * w + i / h) as u16).collect(),
                    _ => default_scan(w, h).unwrap().to_vec(),
                };
                let scan = coefficient_scan(class, w, h).unwrap();
                assert_eq!(scan, original);
                assert_eq!(
                    scan.as_ptr(),
                    coefficient_scan(class, w, h).unwrap().as_ptr()
                );
            }
        }
    }

    #[test]
    fn allocation_fast_path_preserves_coefficients_and_entropy_cursor() {
        let mut random = 0x1834_5678u32;
        for (w, h) in TX_DIMENSIONS {
            for trial in 0..16 {
                let mut tile = vec![0; 8192];
                if trial != 0 {
                    for byte in &mut tile {
                        random ^= random << 13;
                        random ^= random >> 17;
                        random ^= random << 5;
                        *byte = random as u8;
                    }
                }
                let mut expected_state = CoefficientState::new(64, &[(64, 64); 3]);
                let mut actual_state = expected_state.clone();
                let mut expected_decoder = SymbolDecoder::new(&tile, true).unwrap();
                let mut actual_decoder = SymbolDecoder::new(&tile, true).unwrap();
                set_allocation_reference(true);
                let expected = expected_state.read(
                    &mut expected_decoder,
                    0,
                    0,
                    0,
                    w,
                    h,
                    w,
                    h,
                    0,
                    false,
                    false,
                    128,
                    Some(0),
                );
                set_allocation_reference(false);
                let actual = actual_state.read(
                    &mut actual_decoder,
                    0,
                    0,
                    0,
                    w,
                    h,
                    w,
                    h,
                    0,
                    false,
                    false,
                    128,
                    Some(0),
                );
                match (expected, actual) {
                    (Ok(expected), Ok(actual)) => {
                        assert_eq!(expected.tx_type, actual.tx_type);
                        assert_eq!(expected.values, actual.values);
                        assert_eq!(
                            expected_decoder.read_literal(8),
                            actual_decoder.read_literal(8)
                        );
                        assert_eq!(expected_state.base, actual_state.base);
                        assert_eq!(expected_state.br, actual_state.br);
                        assert_eq!(expected_state.dc_sign, actual_state.dc_sign);
                    }
                    (Err(expected), Err(actual)) => assert_eq!(expected, actual),
                    _ => panic!("coefficient allocation path changed decode result"),
                }
            }
        }
    }

    #[test]
    #[ignore = "isolated coefficient allocation ABBA benchmark, not whole-frame throughput"]
    fn coefficient_allocation_benchmark() {
        use std::hint::black_box;
        use std::time::Instant;
        #[cfg(target_os = "macos")]
        fn cpu_time() -> u64 {
            unsafe extern "C" {
                fn clock_gettime_nsec_np(clock_id: i32) -> u64;
            }
            unsafe { clock_gettime_nsec_np(16) }
        }
        let tile = vec![0; 8192];
        for (w, h) in [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)] {
            let mut timings: [Vec<f64>; 2] = Default::default();
            let mut expected = None;
            for trial in 0..20 {
                let reference = [true, false, false, true][trial % 4];
                set_allocation_reference(reference);
                let mut state = CoefficientState::new(64, &[(64, 64); 3]);
                let mut checksum = 0i64;
                let start = Instant::now();
                #[cfg(target_os = "macos")]
                let cpu_start = cpu_time();
                for _ in 0..20_000 {
                    let mut decoder = SymbolDecoder::new(black_box(&tile), false).unwrap();
                    let result = state
                        .read(
                            &mut decoder,
                            0,
                            0,
                            0,
                            w,
                            h,
                            w,
                            h,
                            0,
                            false,
                            false,
                            0,
                            Some(0),
                        )
                        .unwrap();
                    assert!(result.values.iter().any(|&v| v != 0));
                    checksum += i64::from(black_box(result.values[0]));
                    black_box(result);
                }
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                #[cfg(target_os = "macos")]
                let elapsed = {
                    black_box(elapsed);
                    (cpu_time() - cpu_start) as f64 / 1_000_000.0
                };
                assert_eq!(*expected.get_or_insert(checksum), checksum);
                if trial >= 4 {
                    timings[usize::from(reference)].push(elapsed);
                }
            }
            set_allocation_reference(false);
            let median = timings.map(|mut values| {
                values.sort_by(f64::total_cmp);
                values[values.len() / 2]
            });
            eprintln!(
                "COEFFICIENT_ALLOC {w}x{h} blocks=20000 thread_cpu={} original_ms={:.3} cached_ms={:.3}",
                cfg!(target_os = "macos"),
                median[1],
                median[0]
            );
        }
    }

    #[test]
    fn saved_cdfs_exclude_boundaries_and_load_reuses_local_storage() {
        let mut source = CoefficientState::new(128, &[(64, 64), (32, 32), (32, 32)]);
        source.boundaries[0].above_level.fill(23);
        source.skip[2] = 31;
        source.base[4] = 19;
        let saved = source.snapshot_cdfs();
        assert!(saved.boundaries.is_empty());
        let mut target = CoefficientState::new(64, &[(64, 64), (32, 32), (32, 32)]);
        target.boundaries[0].above_level.fill(7);
        let skip_storage = target.skip.as_ptr();
        let base_storage = target.base.as_ptr();
        let boundary_storage = target.boundaries[0].above_level.as_ptr();
        target.load_cdfs(&saved);
        assert_eq!(target.skip.as_ptr(), skip_storage);
        assert_eq!(target.base.as_ptr(), base_storage);
        assert_eq!(target.boundaries[0].above_level.as_ptr(), boundary_storage);
        assert!(target.boundaries[0].above_level.iter().all(|&v| v == 7));
        source.reset_counts();
        assert_eq!(target.skip, source.skip);
        assert_eq!(target.eob, source.eob);
        assert_eq!(target.base_eob, source.base_eob);
        assert_eq!(target.base, source.base);
        assert_eq!(target.eob_extra, source.eob_extra);
        assert_eq!(target.br, source.br);
        assert_eq!(target.dc_sign, source.dc_sign);
        assert_eq!(target.tx1, source.tx1);
        assert_eq!(target.tx2, source.tx2);
        assert_eq!(target.inter_tx1, source.inter_tx1);
        assert_eq!(target.inter_tx2, source.inter_tx2);
        assert_eq!(target.inter_tx3, source.inter_tx3);
    }
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
