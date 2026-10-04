//! AV1 specification sections 4.10, 5.5 and 5.9. No external codec dependency.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Truncated,
    Invalid(&'static str),
    Unsupported(&'static str),
}

pub(crate) struct Bits<'a> {
    data: &'a [u8],
    pub(crate) position: usize,
}

impl<'a> Bits<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }
    pub(crate) fn read(&mut self, n: u8) -> Result<u32, Error> {
        if n > 32
            || self
                .position
                .checked_add(n as usize)
                .is_none_or(|end| end > self.data.len().saturating_mul(8))
        {
            return Err(Error::Truncated);
        }
        if n == 0 { return Ok(0); }
        let offset = self.position % 8;
        let count = (offset + usize::from(n)).div_ceil(8);
        let first = self.position / 8;
        let mut word = 0u64;
        for &byte in &self.data[first..first + count] {
            word = (word << 8) | u64::from(byte);
        }
        let value = (word >> (count * 8 - offset - usize::from(n)))
            & ((1u64 << n) - 1);
        self.position += usize::from(n);
        Ok(value as u32)
    }
    pub(crate) fn flag(&mut self) -> Result<bool, Error> {
        Ok(self.read(1)? != 0)
    }
    fn signed(&mut self, n: u8) -> Result<i32, Error> {
        let value = self.read(n)? as i32;
        Ok((value << (32 - n)) >> (32 - n))
    }
    fn uvlc(&mut self) -> Result<u32, Error> {
        let mut zeros = 0;
        while !self.flag()? {
            zeros += 1;
            if zeros == 32 {
                return Ok(u32::MAX);
            }
        }
        Ok(((1u32 << zeros) - 1) + self.read(zeros)?)
    }
    fn ns(&mut self, n: u32) -> Result<u32, Error> {
        if n == 0 {
            return Err(Error::Invalid("empty ns alphabet"));
        }
        if n == 1 {
            return Ok(0);
        }
        let width = (32 - n.leading_zeros()) as u8;
        let m = (1u32 << width) - n;
        let value = self.read(width - 1)?;
        if value < m {
            Ok(value)
        } else {
            Ok((value << 1) - m + self.read(1)?)
        }
    }
    fn subexp(&mut self, symbols: u32) -> Result<u32, Error> {
        let (mut i, mut base) = (0, 0);
        loop {
            let bits = if i == 0 { 3 } else { 2 + i };
            let range = 1 << bits;
            if symbols <= base + 3 * range {
                return Ok(base + self.ns(symbols - base)?);
            }
            if !self.flag()? {
                return Ok(base + self.read(bits as u8)?);
            }
            i += 1;
            base += range;
        }
    }
    fn signed_subexp(&mut self, low: i32, high: i32, reference: i32) -> Result<i32, Error> {
        let n = (high - low) as u32;
        let r = (reference - low) as u32;
        if r >= n {
            return Err(Error::Invalid("global motion reference"));
        }
        let value = self.subexp(n)?;
        fn recenter(r: u32, v: u32) -> u32 {
            if v > 2 * r {
                v
            } else if v & 1 != 0 {
                r - v.div_ceil(2)
            } else {
                r + v / 2
            }
        }
        let result = if 2 * r <= n {
            recenter(r, value)
        } else {
            n - 1 - recenter(n - 1 - r, value)
        };
        Ok(low + result as i32)
    }
    pub(crate) fn align(&mut self) -> Result<(), Error> {
        while self.position % 8 != 0 {
            if self.flag()? {
                return Err(Error::Invalid("nonzero alignment bit"));
            }
        }
        Ok(())
    }
    pub(crate) fn trailing(&mut self) -> Result<(), Error> {
        if !self.flag()? {
            return Err(Error::Invalid("missing trailing one bit"));
        }
        while self.position < self.data.len() * 8 {
            if self.flag()? {
                return Err(Error::Invalid("nonzero trailing bit"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatingPoint {
    pub idc: u16,
    pub level: u8,
    pub tier: bool,
    pub decoder_model_present: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceHeader {
    pub profile: u8,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub operating_points: Vec<OperatingPoint>,
    pub max_width: u32,
    pub max_height: u32,
    pub width_bits: u8,
    pub height_bits: u8,
    pub frame_id_bits: u8,
    pub delta_frame_id_bits: u8,
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    pub enable_order_hint: bool,
    pub order_hint_bits: u8,
    pub force_screen_content_tools: u8,
    pub force_integer_mv: u8,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub bit_depth: u8,
    pub monochrome: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub full_range: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub chroma_sample_position: u8,
    pub separate_uv_delta_q: bool,
    pub film_grain_params_present: bool,
    pub equal_picture_interval: bool,
    pub buffer_removal_time_bits: u8,
    pub frame_presentation_time_bits: u8,
}

impl SequenceHeader {
    pub fn parse(payload: &[u8]) -> Result<Self, Error> {
        let mut b = Bits::new(payload);
        let profile = b.read(3)? as u8;
        if profile > 2 {
            return Err(Error::Invalid("reserved sequence profile"));
        }
        let still_picture = b.flag()?;
        let reduced = b.flag()?;
        if reduced && !still_picture {
            return Err(Error::Invalid("reduced header requires still picture"));
        }
        let mut equal_picture_interval = false;
        let mut delay_bits = 0;
        let mut removal_bits = 0;
        let mut presentation_bits = 0;
        let mut ops = Vec::new();
        if reduced {
            ops.push(OperatingPoint {
                idc: 0,
                level: b.read(5)? as u8,
                tier: false,
                decoder_model_present: false,
            });
        } else {
            if b.flag()? {
                if b.read(32)? == 0 || b.read(32)? == 0 {
                    return Err(Error::Invalid("zero timing rate"));
                }
                equal_picture_interval = b.flag()?;
                if equal_picture_interval {
                    b.uvlc()?;
                }
                if b.flag()? {
                    delay_bits = b.read(5)? as u8 + 1;
                    if b.read(32)? == 0 {
                        return Err(Error::Invalid("zero decoding tick"));
                    }
                    removal_bits = b.read(5)? as u8 + 1;
                    presentation_bits = b.read(5)? as u8 + 1;
                }
            }
            let display_delay = b.flag()?;
            let count = b.read(5)? + 1;
            for _ in 0..count {
                let idc = b.read(12)? as u16;
                let level = b.read(5)? as u8;
                let tier = level > 7 && b.flag()?;
                let model = delay_bits != 0 && b.flag()?;
                if model {
                    b.read(delay_bits)?;
                    b.read(delay_bits)?;
                    b.flag()?;
                }
                if display_delay && b.flag()? {
                    b.read(4)?;
                }
                ops.push(OperatingPoint {
                    idc,
                    level,
                    tier,
                    decoder_model_present: model,
                });
            }
        }
        let width_bits = b.read(4)? as u8 + 1;
        let height_bits = b.read(4)? as u8 + 1;
        let max_width = b.read(width_bits)? + 1;
        let max_height = b.read(height_bits)? + 1;
        let mut delta_frame_id_bits = 0;
        let frame_id_bits = if !reduced && b.flag()? {
            let delta = b.read(4)? as u8 + 2;
            delta_frame_id_bits = delta;
            let extra = b.read(3)? as u8 + 1;
            if delta + extra > 16 {
                return Err(Error::Invalid("frame id exceeds 16 bits"));
            }
            delta + extra
        } else {
            0
        };
        let use_128x128_superblock = b.flag()?;
        let enable_filter_intra = b.flag()?;
        let enable_intra_edge_filter = b.flag()?;
        let (mut order, mut order_bits, mut screen, mut integer) = (false, 0, 2, 2);
        let mut inter = [false; 6];
        if !reduced {
            for flag in &mut inter[..4] {
                *flag = b.flag()?;
            }
            order = b.flag()?;
            if order {
                inter[4] = b.flag()?;
                inter[5] = b.flag()?;
            }
            screen = if b.flag()? { 2 } else { b.read(1)? as u8 };
            if screen > 0 {
                integer = if b.flag()? { 2 } else { b.read(1)? as u8 };
            }
            if order {
                order_bits = b.read(3)? as u8 + 1;
            }
        }
        let enable_superres = b.flag()?;
        let enable_cdef = b.flag()?;
        let enable_restoration = b.flag()?;
        let high = b.flag()?;
        let bit_depth = if profile == 2 && high {
            if b.flag()? { 12 } else { 10 }
        } else if high {
            10
        } else {
            8
        };
        let monochrome = profile != 1 && b.flag()?;
        let (cp, tc, mc) = if b.flag()? {
            (b.read(8)? as u8, b.read(8)? as u8, b.read(8)? as u8)
        } else {
            (2, 2, 2)
        };
        let (full_range, sx, sy, chroma_position, separate) = if monochrome {
            (b.flag()?, true, true, 0, false)
        } else {
            let identity_rgb = cp == 1 && tc == 13 && mc == 0;
            let full = identity_rgb || b.flag()?;
            let (sx, sy) = if identity_rgb || profile == 1 {
                (false, false)
            } else if profile == 0 {
                (true, true)
            } else if bit_depth == 12 {
                let x = b.flag()?;
                (x, x && b.flag()?)
            } else {
                (true, false)
            };
            let position = if sx && sy { b.read(2)? as u8 } else { 0 };
            if position == 3 {
                return Err(Error::Invalid("reserved chroma sample position"));
            }
            if mc == 0 && (sx || sy) {
                return Err(Error::Invalid("identity matrix with subsampling"));
            }
            (full, sx, sy, position, b.flag()?)
        };
        let film_grain_params_present = b.flag()?;
        b.trailing()?;
        Ok(Self {
            profile,
            still_picture,
            reduced_still_picture_header: reduced,
            operating_points: ops,
            max_width,
            max_height,
            width_bits,
            height_bits,
            frame_id_bits,
            delta_frame_id_bits,
            use_128x128_superblock,
            enable_filter_intra,
            enable_intra_edge_filter,
            enable_interintra_compound: inter[0],
            enable_masked_compound: inter[1],
            enable_warped_motion: inter[2],
            enable_dual_filter: inter[3],
            enable_jnt_comp: inter[4],
            enable_ref_frame_mvs: inter[5],
            enable_order_hint: order,
            order_hint_bits: order_bits,
            force_screen_content_tools: screen,
            force_integer_mv: integer,
            enable_superres,
            enable_cdef,
            enable_restoration,
            bit_depth,
            monochrome,
            color_primaries: cp,
            transfer_characteristics: tc,
            matrix_coefficients: mc,
            full_range,
            subsampling_x: sx,
            subsampling_y: sy,
            chroma_sample_position: chroma_position,
            separate_uv_delta_q: separate,
            film_grain_params_present,
            equal_picture_interval,
            buffer_removal_time_bits: removal_bits,
            frame_presentation_time_bits: presentation_bits,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileLayout {
    pub column_starts: Vec<u32>,
    pub row_starts: Vec<u32>,
    pub column_log2: u8,
    pub row_log2: u8,
    pub size_bytes: u8,
    pub context_update_tile: u32,
}

fn tile_log2(block: u32, target: u32) -> u8 {
    let mut n = 0;
    while (u64::from(block) << n) < u64::from(target) {
        n += 1;
    }
    n
}

impl TileLayout {
    fn parse(b: &mut Bits<'_>, s: &SequenceHeader, width: u32, height: u32) -> Result<Self, Error> {
        let mi_cols = 2 * width.div_ceil(8);
        let mi_rows = 2 * height.div_ceil(8);
        let shift = if s.use_128x128_superblock { 5 } else { 4 };
        let cols = mi_cols.div_ceil(1 << shift);
        let rows = mi_rows.div_ceil(1 << shift);
        let max_width = 4096 >> (shift + 2);
        let max_area = (4096 * 2304) >> (2 * (shift + 2));
        let min_cols = tile_log2(max_width, cols);
        let min_tiles = min_cols.max(tile_log2(max_area, rows * cols));
        let max_cols = tile_log2(1, cols.min(64));
        let max_rows = tile_log2(1, rows.min(64));
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        let (cl, rl);
        if b.flag()? {
            let mut c = min_cols;
            while c < max_cols && b.flag()? {
                c += 1;
            }
            let mut r = min_tiles.saturating_sub(c);
            while r < max_rows && b.flag()? {
                r += 1;
            }
            let cw = cols.div_ceil(1 << c);
            let rh = rows.div_ceil(1 << r);
            xs.extend((0..cols).step_by(cw as usize).map(|x| x << shift));
            ys.extend((0..rows).step_by(rh as usize).map(|y| y << shift));
            cl = c;
            rl = r;
        } else {
            let (mut start, mut widest) = (0, 0);
            while start < cols {
                if xs.len() == 64 {
                    return Err(Error::Invalid("too many tile columns"));
                }
                xs.push(start << shift);
                let size = b.ns((cols - start).min(max_width))? + 1;
                widest = widest.max(size);
                start += size;
            }
            cl = tile_log2(1, xs.len() as u32);
            let area = if min_tiles > 0 {
                (rows * cols) >> (min_tiles + 1)
            } else {
                rows * cols
            };
            let max_height = (area / widest).max(1);
            start = 0;
            while start < rows {
                if ys.len() == 64 {
                    return Err(Error::Invalid("too many tile rows"));
                }
                ys.push(start << shift);
                start += b.ns((rows - start).min(max_height))? + 1;
            }
            rl = tile_log2(1, ys.len() as u32);
        }
        let count = xs.len() * ys.len();
        xs.push(mi_cols);
        ys.push(mi_rows);
        let (context, size) = if cl + rl > 0 {
            (b.read(cl + rl)?, b.read(2)? as u8 + 1)
        } else {
            (0, 0)
        };
        if context as usize >= count {
            return Err(Error::Invalid("context update tile index"));
        }
        Ok(Self {
            column_starts: xs,
            row_starts: ys,
            column_log2: cl,
            row_log2: rl,
            size_bytes: size,
            context_update_tile: context,
        })
    }
    pub fn count(&self) -> usize {
        self.column_starts.len().saturating_sub(1) * self.row_starts.len().saturating_sub(1)
    }
}

/// Complete uncompressed intra header for streams without applied film grain.
/// `header_bytes` is the tile-group offset for OBU_FRAME, not decoded pixel data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntraFrameHeader {
    pub frame_type: u8,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub width: u32,
    pub height: u32,
    pub upscaled_width: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub disable_cdf_update: bool,
    pub disable_frame_end_update_cdf: bool,
    pub order_hint: u32,
    pub refresh_frame_flags: u8,
    pub error_resilient_mode: bool,
    pub current_frame_id: u32,
    pub primary_ref_frame: u8,
    pub ref_frame_idx: [u8; 7],
    pub ref_order_hints: [Option<u32>; 8],
    pub force_integer_mv: bool,
    pub allow_high_precision_mv: bool,
    /// 0..=3 for fixed filters; 4 for switchable.
    pub interpolation_filter: u8,
    pub is_motion_mode_switchable: bool,
    pub use_ref_frame_mvs: bool,
    pub reference_select: bool,
    pub skip_mode_frames: Option<[u8; 2]>,
    pub skip_mode_present: bool,
    pub allow_warped_motion: bool,
    pub global_motion_types: [u8; 7],
    pub global_motion_params: [[i32; 6]; 7],
    pub allow_intrabc: bool,
    pub allow_screen_content_tools: bool,
    pub tiles: TileLayout,
    pub base_q_idx: u8,
    pub quantizer_deltas: [i32; 5],
    pub quantizer_matrix_levels: Option<[u8; 3]>,
    pub segment_features: [[i32; 8]; 8],
    pub segmentation_enabled: bool,
    pub segment_feature_enabled: [[bool; 8]; 8],
    pub delta_q_resolution: Option<u8>,
    pub delta_lf: Option<(u8, bool)>,
    pub coded_lossless: bool,
    pub loop_filter_levels: [u8; 4],
    pub loop_filter_delta_enabled: bool,
    pub loop_filter_sharpness: u8,
    pub loop_filter_ref_deltas: [i32; 8],
    pub loop_filter_mode_deltas: [i32; 2],
    pub cdef_damping: u8,
    pub cdef_strengths: Vec<[u8; 4]>,
    pub restoration_types: Vec<u8>,
    pub restoration_unit_sizes: [u32; 3],
    pub tx_mode_select: bool,
    pub reduced_tx_set: bool,
    pub header_bytes: usize,
}

fn delta_q(b: &mut Bits<'_>) -> Result<i32, Error> {
    if b.flag()? { b.signed(7) } else { Ok(0) }
}

pub(crate) fn relative_dist(s: &SequenceHeader, a: u32, b: u32) -> i32 {
    if !s.enable_order_hint {
        return 0;
    }
    let diff = a.wrapping_sub(b) as i32;
    let sign = 1 << (s.order_hint_bits - 1);
    (diff & (sign - 1)) - (diff & sign)
}

fn set_frame_refs(
    s: &SequenceHeader,
    refs: &[Option<&IntraFrameHeader>; 8],
    hint: u32,
    last: u8,
    gold: u8,
) -> Result<[u8; 7], Error> {
    let mut shifted = [0; 8];
    let current = 1 << (s.order_hint_bits - 1);
    for (i, r) in refs.iter().enumerate() {
        shifted[i] = current
            + relative_dist(
                s,
                r.ok_or(Error::Invalid("short reference signaling slot"))?
                    .order_hint,
                hint,
            );
    }
    if shifted[last as usize] >= current || shifted[gold as usize] >= current {
        return Err(Error::Invalid("short reference signaling order"));
    }
    let mut result = [None; 7];
    result[0] = Some(last);
    result[3] = Some(gold);
    let mut used = [false; 8];
    used[last as usize] = true;
    used[gold as usize] = true;
    for (dest, latest) in [(6, true), (4, false), (5, false)] {
        let mut best: Option<usize> = None;
        for i in 0..8 {
            if !used[i]
                && shifted[i] >= current
                && best.is_none_or(|j| {
                    if latest {
                        shifted[i] >= shifted[j]
                    } else {
                        shifted[i] < shifted[j]
                    }
                })
            {
                best = Some(i);
            }
        }
        if let Some(i) = best {
            result[dest] = Some(i as u8);
            used[i] = true;
        }
    }
    for dest in [1, 2, 4, 5, 6] {
        if result[dest].is_some() {
            continue;
        }
        let mut best: Option<usize> = None;
        for i in 0..8 {
            if !used[i] && shifted[i] < current && best.is_none_or(|j| shifted[i] >= shifted[j]) {
                best = Some(i);
            }
        }
        if let Some(i) = best {
            result[dest] = Some(i as u8);
            used[i] = true;
        }
    }
    let mut earliest = 0;
    for i in 1..8 {
        if shifted[i] < shifted[earliest] {
            earliest = i;
        }
    }
    Ok(result.map(|r| r.unwrap_or(earliest as u8)))
}

fn derive_skip_mode(
    s: &SequenceHeader,
    refs: &[Option<&IntraFrameHeader>; 8],
    indices: &[u8; 7],
    hint: u32,
) -> Option<[u8; 2]> {
    let mut forward: Option<(usize, u32)> = None;
    let mut backward: Option<(usize, u32)> = None;
    for (i, &slot) in indices.iter().enumerate() {
        let h = refs[slot as usize]?.order_hint;
        let d = relative_dist(s, h, hint);
        if d < 0 && forward.is_none_or(|(_, old)| relative_dist(s, h, old) > 0) {
            forward = Some((i, h));
        }
        if d > 0 && backward.is_none_or(|(_, old)| relative_dist(s, h, old) < 0) {
            backward = Some((i, h));
        }
    }
    let (first, forward_hint) = forward?;
    let second = if let Some((i, _)) = backward {
        i
    } else {
        let mut other: Option<(usize, u32)> = None;
        for (i, &slot) in indices.iter().enumerate() {
            let h = refs[slot as usize]?.order_hint;
            if relative_dist(s, h, forward_hint) < 0
                && other.is_none_or(|(_, old)| relative_dist(s, h, old) > 0)
            {
                other = Some((i, h));
            }
        }
        other?.0
    };
    Some([1 + first.min(second) as u8, 1 + first.max(second) as u8])
}

impl IntraFrameHeader {
    /// Parse the uncompressed header and zero alignment inside OBU_FRAME (kind 6).
    /// A standalone OBU_FRAME_HEADER has different trailing-bit syntax.
    pub fn parse(
        payload: &[u8],
        s: &SequenceHeader,
        temporal_id: u8,
        spatial_id: u8,
    ) -> Result<Self, Error> {
        let header = Self::parse_with_refs(payload, s, temporal_id, spatial_id, &[None; 8])?;
        if header.frame_type != 0 && header.frame_type != 2 {
            return Err(Error::Unsupported("inter frame header"));
        }
        Ok(header)
    }

    pub(crate) fn parse_with_refs(
        payload: &[u8],
        s: &SequenceHeader,
        temporal_id: u8,
        spatial_id: u8,
        refs: &[Option<&Self>; 8],
    ) -> Result<Self, Error> {
        if temporal_id > 7 || spatial_id > 3 {
            return Err(Error::Invalid("layer id"));
        }
        let mut b = Bits::new(payload);
        let (kind, show, showable, resilient) = if s.reduced_still_picture_header {
            (0, true, false, true)
        } else {
            if b.flag()? {
                return Err(Error::Unsupported(
                    "show-existing frame requires reference frames",
                ));
            }
            let kind = b.read(2)? as u8;
            let show = b.flag()?;
            if show && s.frame_presentation_time_bits != 0 && !s.equal_picture_interval {
                b.read(s.frame_presentation_time_bits)?;
            }
            let showable = if show { kind != 0 } else { b.flag()? };
            let resilient = kind == 3 || (kind == 0 && show) || b.flag()?;
            (kind, show, showable, resilient)
        };
        let intra = kind == 0 || kind == 2;
        let disable_cdf_update = b.flag()?;
        let screen = if s.force_screen_content_tools == 2 {
            b.flag()?
        } else {
            s.force_screen_content_tools != 0
        };
        let signaled_integer_mv = screen
            && if s.force_integer_mv == 2 {
                b.flag()?
            } else {
                s.force_integer_mv != 0
            };
        let force_integer_mv = intra || signaled_integer_mv;
        let current_frame_id = b.read(s.frame_id_bits)?;
        let override_size = kind == 3 || (!s.reduced_still_picture_header && b.flag()?);
        let order_hint = b.read(s.order_hint_bits)?;
        let primary_ref_frame = if intra || resilient {
            7
        } else {
            b.read(3)? as u8
        };
        if s.buffer_removal_time_bits != 0 && b.flag()? {
            for op in &s.operating_points {
                if op.decoder_model_present
                    && (op.idc == 0
                        || ((op.idc >> temporal_id) & 1 != 0
                            && (op.idc >> (spatial_id + 8)) & 1 != 0))
                {
                    b.read(s.buffer_removal_time_bits)?;
                }
            }
        }
        let refresh = if (kind == 0 && show) || kind == 3 {
            255
        } else {
            b.read(8)? as u8
        };
        let mut ref_order_hints = [None; 8];
        if (!intra || refresh != 255) && resilient && s.enable_order_hint {
            for hint in &mut ref_order_hints {
                *hint = Some(b.read(s.order_hint_bits)?);
            }
        }
        let mut ref_frame_idx = [0; 7];
        if !intra {
            let short = s.enable_order_hint && b.flag()?;
            if short {
                let last = b.read(3)? as u8;
                let gold = b.read(3)? as u8;
                ref_frame_idx = set_frame_refs(s, refs, order_hint, last, gold)?;
            }
            for index in &mut ref_frame_idx {
                if !short {
                    *index = b.read(3)? as u8;
                }
                let reference =
                    refs[*index as usize].ok_or(Error::Invalid("missing reference frame"))?;
                if s.frame_id_bits != 0 {
                    let delta = b.read(s.delta_frame_id_bits)? + 1;
                    let expected =
                        current_frame_id.wrapping_sub(delta) & ((1 << s.frame_id_bits) - 1);
                    if reference.current_frame_id != expected {
                        return Err(Error::Invalid("reference frame id"));
                    }
                }
                if ref_order_hints[*index as usize].is_some_and(|hint| hint != reference.order_hint)
                {
                    return Err(Error::Invalid("invalidated reference order hint"));
                }
            }
        }
        let mut size_ref = None;
        if !intra && override_size && !resilient {
            for index in ref_frame_idx {
                if b.flag()? {
                    size_ref = refs[index as usize];
                    break;
                }
            }
        }
        let (upscaled, height) = if let Some(r) = size_ref {
            (r.upscaled_width, r.height)
        } else if override_size {
            (b.read(s.width_bits)? + 1, b.read(s.height_bits)? + 1)
        } else {
            (s.max_width, s.max_height)
        };
        if upscaled > s.max_width || height > s.max_height {
            return Err(Error::Invalid("frame exceeds sequence size"));
        }
        let denom = if s.enable_superres && b.flag()? {
            b.read(3)? + 9
        } else {
            8
        };
        let width = (upscaled * 8 + denom / 2) / denom;
        let (render_width, render_height) = if let Some(r) = size_ref {
            (r.render_width, r.render_height)
        } else if b.flag()? {
            (b.read(16)? + 1, b.read(16)? + 1)
        } else {
            (upscaled, height)
        };
        let allow_intrabc = intra && screen && upscaled == width && b.flag()?;
        let mut allow_high_precision_mv = false;
        let mut interpolation_filter = 0;
        let mut is_motion_mode_switchable = false;
        let mut use_ref_frame_mvs = false;
        if !intra {
            allow_high_precision_mv = !force_integer_mv && b.flag()?;
            interpolation_filter = if b.flag()? { 4 } else { b.read(2)? as u8 };
            is_motion_mode_switchable = b.flag()?;
            use_ref_frame_mvs = !resilient && s.enable_ref_frame_mvs && b.flag()?;
        }
        let end_cdf = s.reduced_still_picture_header || disable_cdf_update || b.flag()?;
        let tiles = TileLayout::parse(&mut b, s, width, height)?;
        let base_q_idx = b.read(8)? as u8;
        let mut deltas = [0; 5];
        deltas[0] = delta_q(&mut b)?;
        if !s.monochrome {
            let different = s.separate_uv_delta_q && b.flag()?;
            deltas[1] = delta_q(&mut b)?;
            deltas[2] = delta_q(&mut b)?;
            deltas[3] = if different {
                delta_q(&mut b)?
            } else {
                deltas[1]
            };
            deltas[4] = if different {
                delta_q(&mut b)?
            } else {
                deltas[2]
            };
        }
        let quantizer_matrix_levels = if b.flag()? {
            let y = b.read(4)? as u8;
            let u = b.read(4)? as u8;
            let v = if s.separate_uv_delta_q {
                b.read(4)? as u8
            } else {
                u
            };
            Some([y, u, v])
        } else {
            None
        };
        let mut features = [[0; 8]; 8];
        let mut enabled = [[false; 8]; 8];
        let segmentation_enabled = b.flag()?;
        if segmentation_enabled {
            if primary_ref_frame != 7 {
                return Err(Error::Unsupported("inter segmentation header"));
            }
            for segment in 0..8 {
                for feature in 0..8 {
                    enabled[segment][feature] = b.flag()?;
                    if enabled[segment][feature] {
                        let bits = [8, 6, 6, 6, 6, 3, 0, 0][feature];
                        let limit = [255, 63, 63, 63, 63, 7, 0, 0][feature];
                        features[segment][feature] = if feature < 5 {
                            b.signed(bits + 1)?.clamp(-limit, limit)
                        } else {
                            (b.read(bits)? as i32).min(limit)
                        };
                    }
                }
            }
        }
        let delta_q_resolution = if base_q_idx != 0 && b.flag()? {
            Some(b.read(2)? as u8)
        } else {
            None
        };
        let delta_lf = if delta_q_resolution.is_some() && !allow_intrabc && b.flag()? {
            Some((b.read(2)? as u8, b.flag()?))
        } else {
            None
        };
        let coded_lossless = deltas.iter().all(|&x| x == 0)
            && features
                .iter()
                .all(|x| (i32::from(base_q_idx) + x[0]).clamp(0, 255) == 0);
        let mut levels = [0; 4];
        let mut sharpness = 0;
        let previous = if primary_ref_frame == 7 {
            None
        } else {
            refs[ref_frame_idx[primary_ref_frame as usize] as usize]
        };
        let mut ref_deltas =
            previous.map_or([1, 0, 0, 0, -1, 0, -1, -1], |r| r.loop_filter_ref_deltas);
        let mut mode_deltas = previous.map_or([0; 2], |r| r.loop_filter_mode_deltas);
        let mut loop_filter_delta_enabled = false;
        if !coded_lossless && !allow_intrabc {
            levels[0] = b.read(6)? as u8;
            levels[1] = b.read(6)? as u8;
            if !s.monochrome && (levels[0] != 0 || levels[1] != 0) {
                levels[2] = b.read(6)? as u8;
                levels[3] = b.read(6)? as u8;
            }
            sharpness = b.read(3)? as u8;
            loop_filter_delta_enabled = b.flag()?;
            if loop_filter_delta_enabled && b.flag()? {
                for value in ref_deltas.iter_mut().chain(mode_deltas.iter_mut()) {
                    if b.flag()? {
                        *value = b.signed(7)?;
                    }
                }
            }
        }
        let mut cdef = Vec::new();
        let mut cdef_damping = 3;
        if !coded_lossless && !allow_intrabc && s.enable_cdef {
            cdef_damping = b.read(2)? as u8 + 3;
            let count = 1 << b.read(2)?;
            for _ in 0..count {
                let mut strength = [0; 4];
                for plane in 0..if s.monochrome { 1 } else { 2 } {
                    strength[plane * 2] = b.read(4)? as u8;
                    let sec = b.read(2)? as u8;
                    strength[plane * 2 + 1] = if sec == 3 { 4 } else { sec };
                }
                cdef.push(strength);
            }
        }
        let mut restoration = vec![0; if s.monochrome { 1 } else { 3 }];
        let mut restoration_unit_sizes = [0; 3];
        if !(coded_lossless && width == upscaled) && !allow_intrabc && s.enable_restoration {
            for r in &mut restoration {
                *r = b.read(2)? as u8;
            }
            if restoration.iter().any(|&x| x != 0) {
                let mut shift = b.read(1)?;
                if s.use_128x128_superblock {
                    shift += 1;
                } else if shift != 0 {
                    shift += b.read(1)?;
                }
                let uv_shift = if s.subsampling_x
                    && s.subsampling_y
                    && restoration[1..].iter().any(|&x| x != 0)
                {
                    b.read(1)?
                } else {
                    0
                };
                restoration_unit_sizes[0] = 256 >> (2 - shift);
                restoration_unit_sizes[1] = restoration_unit_sizes[0] >> uv_shift;
                restoration_unit_sizes[2] = restoration_unit_sizes[1];
            }
        }
        let tx_mode_select = !coded_lossless && b.flag()?;
        let reference_select = !intra && b.flag()?;
        let skip_mode_frames = if !intra && reference_select && s.enable_order_hint {
            derive_skip_mode(s, refs, &ref_frame_idx, order_hint)
        } else {
            None
        };
        let skip_mode_present = skip_mode_frames.is_some() && b.flag()?;
        let allow_warped_motion = !intra && !resilient && s.enable_warped_motion && b.flag()?;
        let reduced_tx_set = b.flag()?;
        let mut global_motion_types = [0; 7];
        let identity = [0, 0, 1 << 16, 0, 0, 1 << 16];
        let mut global_motion_params = [identity; 7];
        if !intra {
            for i in 0..7 {
                let kind = if !b.flag()? {
                    0
                } else if b.flag()? {
                    2
                } else if b.flag()? {
                    1
                } else {
                    3
                };
                global_motion_types[i] = kind;
                let prev = previous.map_or(identity, |r| r.global_motion_params[i]);
                let mut read_param = |idx: usize| -> Result<i32, Error> {
                    let (abs, prec) = if idx >= 2 {
                        (12, 15)
                    } else if kind == 1 {
                        (
                            9 - u8::from(!allow_high_precision_mv),
                            3 - u8::from(!allow_high_precision_mv),
                        )
                    } else {
                        (12, 6)
                    };
                    let diff = 16 - prec;
                    let round = if idx % 3 == 2 { 1 << 16 } else { 0 };
                    let sub = if idx % 3 == 2 { 1 << prec } else { 0 };
                    let reference = (prev[idx] >> diff) - sub;
                    Ok((b.signed_subexp(-(1 << abs), (1 << abs) + 1, reference)? << diff) + round)
                };
                if kind >= 2 {
                    global_motion_params[i][2] = read_param(2)?;
                    global_motion_params[i][3] = read_param(3)?;
                    if kind == 3 {
                        global_motion_params[i][4] = read_param(4)?;
                        global_motion_params[i][5] = read_param(5)?;
                    } else {
                        global_motion_params[i][4] = -global_motion_params[i][3];
                        global_motion_params[i][5] = global_motion_params[i][2];
                    }
                }
                if kind >= 1 {
                    global_motion_params[i][0] = read_param(0)?;
                    global_motion_params[i][1] = read_param(1)?;
                }
            }
        }
        if s.film_grain_params_present && (show || showable) && b.flag()? {
            return Err(Error::Unsupported("applied film grain header"));
        }
        b.align()?;
        Ok(Self {
            frame_type: kind,
            show_frame: show,
            showable_frame: showable,
            width,
            height,
            upscaled_width: upscaled,
            render_width,
            render_height,
            disable_cdf_update,
            disable_frame_end_update_cdf: end_cdf,
            order_hint,
            refresh_frame_flags: refresh,
            error_resilient_mode: resilient,
            current_frame_id,
            primary_ref_frame,
            ref_frame_idx,
            ref_order_hints,
            force_integer_mv,
            allow_high_precision_mv,
            interpolation_filter,
            is_motion_mode_switchable,
            use_ref_frame_mvs,
            reference_select,
            skip_mode_frames,
            skip_mode_present,
            allow_warped_motion,
            global_motion_types,
            global_motion_params,
            allow_intrabc,
            allow_screen_content_tools: screen,
            tiles,
            base_q_idx,
            quantizer_deltas: deltas,
            quantizer_matrix_levels,
            segment_features: features,
            segmentation_enabled,
            segment_feature_enabled: enabled,
            delta_q_resolution,
            delta_lf,
            coded_lossless,
            loop_filter_levels: levels,
            loop_filter_delta_enabled,
            loop_filter_sharpness: sharpness,
            loop_filter_ref_deltas: ref_deltas,
            loop_filter_mode_deltas: mode_deltas,
            cdef_damping,
            cdef_strengths: cdef,
            restoration_types: restoration,
            restoration_unit_sizes,
            tx_mode_select,
            reduced_tx_set,
            header_bytes: b.position / 8,
        })
    }
}

/// Bounded extraction of arithmetic-coded tile slices (spec section 5.11.1).
pub fn tile_group<'a>(
    payload: &'a [u8],
    layout: &TileLayout,
) -> Result<Vec<(usize, &'a [u8])>, Error> {
    let count = layout.count();
    if count == 0
        || count > 4096
        || layout.column_log2 > 6
        || layout.row_log2 > 6
        || (count > 1 && !(1..=4).contains(&layout.size_bytes))
    {
        return Err(Error::Invalid("tile layout"));
    }
    let mut b = Bits::new(payload);
    let (start, end) = if count > 1 && b.flag()? {
        let n = layout.column_log2 + layout.row_log2;
        (b.read(n)? as usize, b.read(n)? as usize)
    } else {
        (0, count - 1)
    };
    if start > end || end >= count {
        return Err(Error::Invalid("tile group range"));
    }
    b.align()?;
    let mut offset = b.position / 8;
    let mut tiles = Vec::new();
    for index in start..=end {
        let len = if index == end {
            payload.len() - offset
        } else {
            let size_end = offset
                .checked_add(layout.size_bytes as usize)
                .ok_or(Error::Truncated)?;
            let bytes = payload.get(offset..size_end).ok_or(Error::Truncated)?;
            let mut size = 0u64;
            for (i, &byte) in bytes.iter().enumerate() {
                size |= u64::from(byte) << (8 * i);
            }
            offset = size_end;
            usize::try_from(size + 1).map_err(|_| Error::Invalid("tile size overflow"))?
        };
        if len == 0 {
            return Err(Error::Invalid("empty tile"));
        }
        let end = offset.checked_add(len).ok_or(Error::Truncated)?;
        tiles.push((index, payload.get(offset..end).ok_or(Error::Truncated)?));
        offset = end;
    }
    Ok(tiles)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_bitwise(bits: &mut Bits<'_>, n: u8) -> Result<u32, Error> {
        if n > 32 || bits.position.checked_add(usize::from(n))
            .is_none_or(|end| end > bits.data.len().saturating_mul(8))
        { return Err(Error::Truncated); }
        let mut value = 0;
        for _ in 0..n {
            value = (value << 1)
                | u32::from((bits.data[bits.position / 8] >> (7 - bits.position % 8)) & 1);
            bits.position += 1;
        }
        Ok(value)
    }

    #[test]
    fn bulk_bit_reader_matches_bitwise_values_cursors_and_truncation() {
        let data: Vec<u8> = (0..48u32).map(|i| i.wrapping_mul(197).wrapping_add(37) as u8).collect();
        for len in 0..=data.len() {
            for position in (0..=len * 8 + 1).chain([usize::MAX - 32, usize::MAX]) {
                for n in 0..=40 {
                    let mut actual = Bits { data: &data[..len], position };
                    let mut expected = Bits { data: &data[..len], position };
                    assert_eq!(actual.read(n), read_bitwise(&mut expected, n),
                        "length={len} position={position} width={n}");
                    assert_eq!(actual.position, expected.position);
                }
            }
        }
    }

    #[test]
    #[ignore = "same-binary ABBA bulk bit-reader benchmark"]
    fn benchmark_bulk_bit_reader() {
        use std::{hint::black_box, time::Instant};
        #[inline(never)]
        fn bulk(bits: &mut Bits<'_>, n: u8) -> Result<u32, Error> { bits.read(n) }
        #[inline(never)]
        fn scalar(bits: &mut Bits<'_>, n: u8) -> Result<u32, Error> { read_bitwise(bits, n) }
        let bytes: Vec<_> = (0..8192u32).map(|i| i.wrapping_mul(197) as u8).collect();
        for widths in [&[1, 2, 3, 4, 5, 6, 7, 8][..], &[9, 12, 15, 16, 24, 32][..]] {
            let mut times = [std::time::Duration::ZERO; 2];
            let mut checksums = [None; 2];
            for arm in [0, 1, 1, 0, 1, 0, 0, 1] {
                let read = if arm == 0 { scalar } else { bulk };
                let start = Instant::now();
                let mut bits = Bits::new(black_box(&bytes));
                let mut sum = 0u64;
                for index in 0..1_000_000usize {
                    if bits.position + 32 > bytes.len() * 8 { bits.position = 0; }
                    sum = sum.wrapping_add(u64::from(read(&mut bits, black_box(widths[index % widths.len()])).unwrap()));
                }
                times[arm] += start.elapsed();
                if let Some(expected) = checksums[arm] { assert_eq!(sum, expected); }
                checksums[arm] = Some(black_box(sum));
            }
            assert_eq!(checksums[0], checksums[1]);
            eprintln!("bit widths={widths:?} bitwise={:?} bulk={:?} speedup={:.3}", times[0], times[1],
                times[0].as_secs_f64() / times[1].as_secs_f64());
        }
    }

    #[derive(Default)]
    struct Writer {
        bytes: Vec<u8>,
        position: usize,
    }
    impl Writer {
        fn put(&mut self, value: u32, n: u8) {
            for bit in (0..n).rev() {
                if self.position % 8 == 0 {
                    self.bytes.push(0);
                }
                self.bytes[self.position / 8] |=
                    (((value >> bit) & 1) as u8) << (7 - self.position % 8);
                self.position += 1;
            }
        }
        fn pad(&mut self) {
            while self.position % 8 != 0 {
                self.put(0, 1);
            }
        }
    }

    #[test]
    fn reduced_still_headers_cover_profiles_and_monochrome() {
        for (profile, depth, mono) in [(0, 8, false), (0, 10, true), (1, 10, false), (2, 12, false)]
        {
            let mut w = Writer::default();
            w.put(profile, 3);
            w.put(1, 1);
            w.put(1, 1);
            w.put(0, 5);
            w.put(3, 4);
            w.put(3, 4);
            w.put(15, 4);
            w.put(15, 4);
            w.put(0, 1);
            w.put(0, 1);
            w.put(1, 1);
            w.put(0, 3); // superres, CDEF, restoration
            w.put(u32::from(depth > 8), 1);
            if profile == 2 && depth > 8 {
                w.put(u32::from(depth == 12), 1);
            }
            if profile != 1 {
                w.put(u32::from(mono), 1);
            }
            w.put(0, 1); // unspecified color description
            w.put(1, 1); // full range
            if !mono {
                if profile == 0 {
                    w.put(0, 2);
                }
                // chroma sample position
                else if profile == 2 && depth == 12 {
                    w.put(1, 1);
                    w.put(0, 1);
                } // 4:2:2
                w.put(0, 1); // separate UV quantizer delta
            }
            w.put(0, 1);
            w.put(1, 1);
            w.pad();
            let s = SequenceHeader::parse(&w.bytes).unwrap();
            assert_eq!(
                (s.profile, s.bit_depth, s.monochrome),
                (profile as u8, depth, mono)
            );
            let mut f = Writer::default();
            f.put(1, 1); // disable CDF update
            f.put(0, 1); // no screen content
            f.put(0, 1); // render dimensions equal
            f.put(1, 1); // uniform single tile
            f.put(0, 8); // lossless quantizer
            f.put(0, 1); // Y DC delta
            if !mono {
                f.put(0, 2);
            } // UV deltas
            f.put(0, 1);
            f.put(0, 1); // matrix and segmentation disabled
            f.put(1, 1); // reduced transform set
            f.pad();
            let frame = IntraFrameHeader::parse(&f.bytes, &s, 0, 0).unwrap();
            assert!(frame.coded_lossless);
            assert!(frame.disable_frame_end_update_cdf);
            assert_eq!(
                (frame.width, frame.height, frame.tiles.count()),
                (16, 16, 1)
            );
        }
    }

    // Header bytes of the local spacewalk fixture, remuxed by the FFmpeg binary.
    const SPACEWALK_SEQUENCE: &[u8] = &[
        0x02, 0x00, 0x00, 0x42, 0x95, 0x5d, 0xfe, 0x1b, 0x8d, 0x5f, 0x32, 0x02, 0x02, 0x02, 0x48,
    ];
    const SPACEWALK_FRAME: &[u8] = &[
        0x10, 0x00, 0x84, 0x00, 0x80, 0x41, 0x00, 0x00, 0x20, 0xbc, 0xf3, 0xcf, 0x80, 0x08, 0x00,
        0x20,
    ];

    #[test]
    fn target_sequence_and_intra_header() {
        let s = SequenceHeader::parse(SPACEWALK_SEQUENCE).unwrap();
        assert_eq!(
            (s.profile, s.max_width, s.max_height, s.bit_depth),
            (0, 1920, 1080, 8)
        );
        assert_eq!(
            (s.subsampling_x, s.subsampling_y, s.full_range),
            (true, true, false)
        );
        assert_eq!(
            (
                s.color_primaries,
                s.transfer_characteristics,
                s.matrix_coefficients
            ),
            (1, 1, 1)
        );
        assert_eq!(s.order_hint_bits, 7);
        let f = IntraFrameHeader::parse(SPACEWALK_FRAME, &s, 0, 0).unwrap();
        assert_eq!(
            (f.width, f.height, f.header_bytes, f.base_q_idx),
            (1920, 1080, 16, 32)
        );
        assert_eq!(f.tiles.count(), 1);
        assert_eq!(f.tiles.column_starts, [0, 480]);
        assert_eq!(f.tiles.row_starts, [0, 270]);
        assert_eq!(f.loop_filter_levels, [1, 1, 0, 0]);
        assert_eq!(
            f.cdef_strengths,
            [[0, 2, 15, 0], [15, 0, 15, 0], [15, 2, 0, 0], [0, 2, 0, 0]]
        );
        assert_eq!(f.restoration_types, [0, 0, 0]);
        assert_eq!(f.delta_q_resolution, Some(0));
        assert!(f.tx_mode_select);
        assert!(!f.coded_lossless);
    }

    #[test]
    fn truncated_headers_and_bad_trailing_bits() {
        for n in 0..SPACEWALK_SEQUENCE.len() {
            assert!(
                SequenceHeader::parse(&SPACEWALK_SEQUENCE[..n]).is_err(),
                "sequence prefix {n}"
            );
        }
        let s = SequenceHeader::parse(SPACEWALK_SEQUENCE).unwrap();
        for n in 0..SPACEWALK_FRAME.len() {
            assert!(
                IntraFrameHeader::parse(&SPACEWALK_FRAME[..n], &s, 0, 0).is_err(),
                "frame prefix {n}"
            );
        }
        let mut bad = SPACEWALK_SEQUENCE.to_vec();
        *bad.last_mut().unwrap() |= 1;
        assert!(matches!(
            SequenceHeader::parse(&bad),
            Err(Error::Invalid(_))
        ));
        let mut bad = SPACEWALK_FRAME.to_vec();
        *bad.last_mut().unwrap() |= 1;
        assert!(matches!(
            IntraFrameHeader::parse(&bad, &s, 0, 0),
            Err(Error::Invalid(_))
        ));
    }

    #[test]
    fn non_symmetric_integer_coding() {
        for n in 1u32..100 {
            let width = 32 - n.leading_zeros();
            let m = (1 << width) - n;
            for expected in 0..n {
                let (code, len) = if expected < m {
                    (expected, width - 1)
                } else {
                    (expected + m, width)
                };
                let bytes = if len == 0 {
                    [0; 4]
                } else {
                    (code << (32 - len)).to_be_bytes()
                };
                assert_eq!(Bits::new(&bytes).ns(n).unwrap(), expected);
            }
        }
    }

    #[test]
    fn tile_sizes_and_group_ranges_are_bounded() {
        let layout = TileLayout {
            column_starts: vec![0, 16, 32],
            row_starts: vec![0, 16],
            column_log2: 1,
            row_log2: 0,
            size_bytes: 1,
            context_update_tile: 0,
        };
        assert_eq!(
            tile_group(&[0, 1, 0xaa, 0xbb, 0xcc], &layout).unwrap(),
            [(0, &[0xaa, 0xbb][..]), (1, &[0xcc][..])]
        );
        assert_eq!(
            tile_group(&[0xe0, 0xab], &layout).unwrap(),
            [(1, &[0xab][..])]
        );
        assert!(tile_group(&[0, 255, 1], &layout).is_err());
        assert!(tile_group(&[0xc0, 1], &layout).is_err());
    }

    #[test]
    fn malformed_payloads_do_not_panic() {
        let s = SequenceHeader::parse(SPACEWALK_SEQUENCE).unwrap();
        let mut state = 971u32;
        for len in 0..64 {
            for _ in 0..32 {
                let bytes: Vec<u8> = (0..len)
                    .map(|_| {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        (state >> 24) as u8
                    })
                    .collect();
                let _ = SequenceHeader::parse(&bytes);
                let _ = IntraFrameHeader::parse(&bytes, &s, 0, 0);
            }
        }
    }
}
