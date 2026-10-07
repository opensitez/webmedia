//! AV1 palette mode, color-cache and diagonal token decoding (sections 5.11/7.12).
use super::entropy::SymbolDecoder;
use super::palette_tables as tables;
use super::syntax::Error;

#[derive(Clone, Default)]
pub(crate) struct PaletteColors {
    pub planes: [Vec<u16>; 3],
}

pub(crate) struct PaletteBlock {
    pub colors: PaletteColors,
    maps: [Vec<u8>; 2],
    widths: [usize; 2],
}

impl PaletteBlock {
    pub(crate) fn has_luma(&self) -> bool {
        !self.colors.planes[0].is_empty()
    }

    pub(crate) fn prediction(&self, plane: usize, x: usize, y: usize, w: usize, h: usize) -> Option<Vec<u16>> {
        let colors = &self.colors.planes[plane];
        if colors.is_empty() { return None; }
        let map_plane = usize::from(plane > 0);
        let stride = self.widths[map_plane];
        let mut samples = Vec::with_capacity(w * h);
        for row in y..y + h {
            for col in x..x + w {
                samples.push(colors[self.maps[map_plane][row * stride + col] as usize]);
            }
        }
        Some(samples)
    }
}

#[derive(Clone)]
pub(crate) struct PaletteCdfs {
    y_mode: Vec<u16>,
    uv_mode: Vec<u16>,
    y_size: Vec<u16>,
    uv_size: Vec<u16>,
    color: [[Vec<u16>; 7]; 2],
}

impl Default for PaletteCdfs {
    fn default() -> Self {
        Self {
            y_mode: tables::PALETTE_Y_MODE_CDF.to_vec(),
            uv_mode: tables::PALETTE_UV_MODE_CDF.to_vec(),
            y_size: tables::PALETTE_Y_SIZE_CDF.to_vec(),
            uv_size: tables::PALETTE_UV_SIZE_CDF.to_vec(),
            color: [
                [tables::PALETTE_SIZE_2_Y_COLOR_CDF.to_vec(), tables::PALETTE_SIZE_3_Y_COLOR_CDF.to_vec(), tables::PALETTE_SIZE_4_Y_COLOR_CDF.to_vec(), tables::PALETTE_SIZE_5_Y_COLOR_CDF.to_vec(), tables::PALETTE_SIZE_6_Y_COLOR_CDF.to_vec(), tables::PALETTE_SIZE_7_Y_COLOR_CDF.to_vec(), tables::PALETTE_SIZE_8_Y_COLOR_CDF.to_vec()],
                [tables::PALETTE_SIZE_2_UV_COLOR_CDF.to_vec(), tables::PALETTE_SIZE_3_UV_COLOR_CDF.to_vec(), tables::PALETTE_SIZE_4_UV_COLOR_CDF.to_vec(), tables::PALETTE_SIZE_5_UV_COLOR_CDF.to_vec(), tables::PALETTE_SIZE_6_UV_COLOR_CDF.to_vec(), tables::PALETTE_SIZE_7_UV_COLOR_CDF.to_vec(), tables::PALETTE_SIZE_8_UV_COLOR_CDF.to_vec()],
            ],
        }
    }
}

impl PaletteCdfs {
    pub(crate) fn reset_counts(&mut self) {
        for table in [&mut self.y_mode, &mut self.uv_mode] {
            for row in table.chunks_exact_mut(3) { row[2] = 0; }
        }
        for table in [&mut self.y_size, &mut self.uv_size] {
            for row in table.chunks_exact_mut(8) { row[7] = 0; }
        }
        for plane in &mut self.color {
            for (size, table) in plane.iter_mut().enumerate() {
                for row in table.chunks_exact_mut(size + 3) { row[size + 2] = 0; }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_modes(&mut self, d: &mut SymbolDecoder<'_>, depth: u8, dimensions: [usize; 2], modes: [usize; 2], has_chroma: bool, above: Option<&PaletteColors>, left: Option<&PaletteColors>, use_above_cache: bool) -> Result<PaletteBlock, Error> {
        let [w, h] = dimensions;
        let size_context = (w.ilog2() + h.ilog2() - 6) as usize;
        let mut colors = PaletteColors::default();
        if modes[0] == 0 {
            let context = usize::from(above.is_some_and(|p| !p.planes[0].is_empty()))
                + usize::from(left.is_some_and(|p| !p.planes[0].is_empty()));
            let start = (size_context * 3 + context) * 3;
            if d.read_symbol(&mut self.y_mode[start..start + 3])? != 0 {
                let start = size_context * 8;
                let size = d.read_symbol(&mut self.y_size[start..start + 8])? + 2;
                colors.planes[0] = read_colors(d, depth, size, 0, above.filter(|_| use_above_cache), left)?;
            }
        }
        if has_chroma && modes[1] == 0 {
            let start = usize::from(!colors.planes[0].is_empty()) * 3;
            if d.read_symbol(&mut self.uv_mode[start..start + 3])? != 0 {
                let start = size_context * 8;
                let size = d.read_symbol(&mut self.uv_size[start..start + 8])? + 2;
                colors.planes[1] = read_colors(d, depth, size, 1, above.filter(|_| use_above_cache), left)?;
                let max = 1i32 << depth;
                if d.read_bool()? {
                    let bits = depth - 4 + d.read_literal(2)? as u8;
                    colors.planes[2].push(d.read_literal(depth)? as u16);
                    for i in 1..size {
                        let mut delta = d.read_literal(bits)? as i32;
                        if delta != 0 && d.read_bool()? { delta = -delta; }
                        colors.planes[2].push((i32::from(colors.planes[2][i - 1]) + delta).rem_euclid(max) as u16);
                    }
                } else {
                    for _ in 0..size { colors.planes[2].push(d.read_literal(depth)? as u16); }
                }
            }
        }
        Ok(PaletteBlock { colors, maps: Default::default(), widths: [0; 2] })
    }

    pub(crate) fn read_tokens(&mut self, d: &mut SymbolDecoder<'_>, block: &mut PaletteBlock, dimensions: [usize; 2], onscreen: [usize; 2], subsampling: [bool; 2]) -> Result<(), Error> {
        for plane in 0..2 {
            let size = block.colors.planes[plane].len();
            if size == 0 { continue; }
            let sx = usize::from(plane > 0 && subsampling[0]);
            let sy = usize::from(plane > 0 && subsampling[1]);
            let w = (dimensions[0] >> sx).max(4);
            let h = (dimensions[1] >> sy).max(4);
            let visible_w = (onscreen[0] >> sx).max(4).min(w);
            let visible_h = (onscreen[1] >> sy).max(4).min(h);
            let mut map = vec![0; w * h];
            map[0] = read_uniform(d, size)? as u8;
            for diagonal in 1..visible_w + visible_h - 1 {
                for col in (diagonal.saturating_sub(visible_h - 1)..=diagonal.min(visible_w - 1)).rev() {
                    let row = diagonal - col;
                    let (context, order) = color_context(&map, w, row, col, size)?;
                    let table = &mut self.color[plane][size - 2];
                    let start = context * (size + 1);
                    let symbol = d.read_symbol(&mut table[start..start + size + 1])?;
                    map[row * w + col] = order[symbol];
                }
            }
            for row in 0..visible_h {
                let last = map[row * w + visible_w - 1];
                map[row * w + visible_w..(row + 1) * w].fill(last);
            }
            for row in visible_h..h {
                map.copy_within((visible_h - 1) * w..visible_h * w, row * w);
            }
            block.widths[plane] = w;
            block.maps[plane] = map;
        }
        Ok(())
    }
}

fn read_uniform(d: &mut SymbolDecoder<'_>, size: usize) -> Result<usize, Error> {
    let bits = size.next_power_of_two().ilog2() as u8;
    let threshold = (1 << bits) - size;
    let value = d.read_literal(bits - 1)? as usize;
    Ok(if value < threshold { value } else { (value << 1) - threshold + usize::from(d.read_bool()?) })
}

fn color_cache(plane: usize, above: Option<&PaletteColors>, left: Option<&PaletteColors>) -> Vec<u16> {
    let mut cache: Vec<_> = above.into_iter().chain(left).flat_map(|p| p.planes[plane].iter().copied()).collect();
    cache.sort_unstable();
    cache.dedup();
    cache
}

fn read_colors(d: &mut SymbolDecoder<'_>, depth: u8, size: usize, plane: usize, above: Option<&PaletteColors>, left: Option<&PaletteColors>) -> Result<Vec<u16>, Error> {
    let mut colors = Vec::with_capacity(size);
    for color in color_cache(plane, above, left) {
        if colors.len() == size { break; }
        if d.read_bool()? { colors.push(color); }
    }
    if colors.len() < size { colors.push(d.read_literal(depth)? as u16); }
    if colors.len() < size {
        let mut bits = depth - 3 + d.read_literal(2)? as u8;
        let increment = usize::from(plane == 0);
        let max = (1usize << depth) - 1;
        while colors.len() < size {
            let next = (usize::from(*colors.last().unwrap()) + d.read_literal(bits)? as usize + increment).min(max);
            colors.push(next as u16);
            let range = (1usize << depth) - next - increment;
            let required = if range <= 1 { 0 } else { (range - 1).ilog2() as u8 + 1 };
            bits = bits.min(required);
        }
    }
    colors.sort_unstable();
    Ok(colors)
}

fn color_context(map: &[u8], stride: usize, row: usize, col: usize, size: usize) -> Result<(usize, [u8; 8]), Error> {
    let mut scores = [0u8; 8];
    let mut order = std::array::from_fn(|i| i as u8);
    if col > 0 { scores[map[row * stride + col - 1] as usize] += 2; }
    if row > 0 && col > 0 { scores[map[(row - 1) * stride + col - 1] as usize] += 1; }
    if row > 0 { scores[map[(row - 1) * stride + col] as usize] += 2; }
    for i in 0..3 {
        let mut best = i;
        for j in i + 1..size { if scores[j] > scores[best] { best = j; } }
        scores[i..=best].rotate_right(1);
        order[i..=best].rotate_right(1);
    }
    let hash: usize = scores[..3].iter().zip(tables::PALETTE_COLOR_HASH_MULTIPLIERS).map(|(&s,m)| usize::from(s) * usize::from(m)).sum();
    let context = *tables::PALETTE_COLOR_CONTEXT.get(hash).ok_or(Error::Invalid("palette context hash"))?;
    if context < 0 { return Err(Error::Invalid("palette context")); }
    Ok((context as usize, order))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caches_merge_unique_colors_without_reordering_v() {
        let above = PaletteColors { planes: [vec![2,7,11], vec![4,8], vec![10,3]] };
        let left = PaletteColors { planes: [vec![1,7,12], vec![3,8], vec![9,5]] };
        assert_eq!(color_cache(0, Some(&above), Some(&left)), [1,2,7,11,12]);
        assert_eq!(color_cache(1, Some(&above), Some(&left)), [3,4,8]);
        assert_eq!(above.planes[2], [10,3]);
    }

    #[test]
    fn color_contexts_match_normative_neighbor_hashes() {
        for size in 2..=8 {
            for left in 0..size {
                for top in 0..size {
                    for diagonal in 0..size {
                        let map = [diagonal as u8, top as u8, left as u8, 0];
                        let (context, order) = color_context(&map, 2, 1, 1, size).unwrap();
                        assert!(context < 5);
                        let mut permutation = order[..size].to_vec();
                        permutation.sort_unstable();
                        assert_eq!(permutation, (0..size as u8).collect::<Vec<_>>());
                    }
                }
            }
        }
        assert_eq!(color_context(&[1,0], 2, 0, 1, 2).unwrap(), (0,[1,0,2,3,4,5,6,7]));
    }
}
