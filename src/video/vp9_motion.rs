//! VP9 separable sub-pixel inter prediction (bitstream specification, section 8.5.2.4).

use super::backend::MediaDecodeError;
use super::vp8::BoolDecoder;
use super::vp8_predict::Plane;
use super::vp9_inter_probs::InterframeProbabilities;
use super::vp9_adapt::NonCoefficientCounts;

const MV_JOINT_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];
const MV_CLASS_TREE: [i8; 20] = [
    0, 2, -1, 4, 6, 8, -2, -3, 10, 12, -4, -5, -6, 14, 16, 18, -7, -8, -9, -10,
];
const MV_FR_TREE: [i8; 6] = [0, 2, -1, 4, -2, -3];

// Syntax type order: regular, smooth, sharp, bilinear.
const SUBPEL_FILTERS: [[[i16; 8]; 16]; 4] = [
    [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [0, 1, -5, 126, 8, -3, 1, 0],
        [-1, 3, -10, 122, 18, -6, 2, 0],
        [-1, 4, -13, 118, 27, -9, 3, -1],
        [-1, 4, -16, 112, 37, -11, 4, -1],
        [-1, 5, -18, 105, 48, -14, 4, -1],
        [-1, 5, -19, 97, 58, -16, 5, -1],
        [-1, 6, -19, 88, 68, -18, 5, -1],
        [-1, 6, -19, 78, 78, -19, 6, -1],
        [-1, 5, -18, 68, 88, -19, 6, -1],
        [-1, 5, -16, 58, 97, -19, 5, -1],
        [-1, 4, -14, 48, 105, -18, 5, -1],
        [-1, 4, -11, 37, 112, -16, 4, -1],
        [-1, 3, -9, 27, 118, -13, 4, -1],
        [0, 2, -6, 18, 122, -10, 3, -1],
        [0, 1, -3, 8, 126, -5, 1, 0],
    ],
    [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [-3, -1, 32, 64, 38, 1, -3, 0],
        [-2, -2, 29, 63, 41, 2, -3, 0],
        [-2, -2, 26, 63, 43, 4, -4, 0],
        [-2, -3, 24, 62, 46, 5, -4, 0],
        [-2, -3, 21, 60, 49, 7, -4, 0],
        [-1, -4, 18, 59, 51, 9, -4, 0],
        [-1, -4, 16, 57, 53, 12, -4, -1],
        [-1, -4, 14, 55, 55, 14, -4, -1],
        [-1, -4, 12, 53, 57, 16, -4, -1],
        [0, -4, 9, 51, 59, 18, -4, -1],
        [0, -4, 7, 49, 60, 21, -3, -2],
        [0, -4, 5, 46, 62, 24, -3, -2],
        [0, -4, 4, 43, 63, 26, -2, -2],
        [0, -3, 2, 41, 63, 29, -2, -2],
        [0, -3, 1, 38, 64, 32, -1, -3],
    ],
    [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [-1, 3, -7, 127, 8, -3, 1, 0],
        [-2, 5, -13, 125, 17, -6, 3, -1],
        [-3, 7, -17, 121, 27, -10, 5, -2],
        [-4, 9, -20, 115, 37, -13, 6, -2],
        [-4, 10, -23, 108, 48, -16, 8, -3],
        [-4, 10, -24, 100, 59, -19, 9, -3],
        [-4, 11, -24, 90, 70, -21, 10, -4],
        [-4, 11, -23, 80, 80, -23, 11, -4],
        [-4, 10, -21, 70, 90, -24, 11, -4],
        [-3, 9, -19, 59, 100, -24, 10, -4],
        [-3, 8, -16, 48, 108, -23, 10, -4],
        [-2, 6, -13, 37, 115, -20, 9, -4],
        [-2, 5, -10, 27, 121, -17, 7, -3],
        [-1, 3, -6, 17, 125, -13, 5, -2],
        [0, 1, -3, 8, 127, -7, 3, -1],
    ],
    [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [0, 0, 0, 120, 8, 0, 0, 0],
        [0, 0, 0, 112, 16, 0, 0, 0],
        [0, 0, 0, 104, 24, 0, 0, 0],
        [0, 0, 0, 96, 32, 0, 0, 0],
        [0, 0, 0, 88, 40, 0, 0, 0],
        [0, 0, 0, 80, 48, 0, 0, 0],
        [0, 0, 0, 72, 56, 0, 0, 0],
        [0, 0, 0, 64, 64, 0, 0, 0],
        [0, 0, 0, 56, 72, 0, 0, 0],
        [0, 0, 0, 48, 80, 0, 0, 0],
        [0, 0, 0, 40, 88, 0, 0, 0],
        [0, 0, 0, 32, 96, 0, 0, 0],
        [0, 0, 0, 24, 104, 0, 0, 0],
        [0, 0, 0, 16, 112, 0, 0, 0],
        [0, 0, 0, 8, 120, 0, 0, 0],
    ],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MotionSampling {
    pub start_x: i32,
    pub start_y: i32,
    pub step_x: i32,
    pub step_y: i32,
}

pub(super) fn use_high_precision(reference_mv: (i32, i32)) -> bool {
    (reference_mv.0.unsigned_abs() >> 3) < 8 && (reference_mv.1.unsigned_abs() >> 3) < 8
}

pub(super) fn read_motion_difference(
    bits: &mut BoolDecoder<'_>,
    probabilities: &InterframeProbabilities,
    high_precision: bool,
) -> Result<(i32, i32), MediaDecodeError> {
    read_motion_difference_counted(bits, probabilities, high_precision, None)
}

pub(super) fn read_motion_difference_counted(
    bits: &mut BoolDecoder<'_>, probabilities: &InterframeProbabilities,
    high_precision: bool, mut counts: Option<&mut NonCoefficientCounts>,
) -> Result<(i32, i32), MediaDecodeError> {
    let joint = read_tree(bits, &MV_JOINT_TREE, &probabilities.mv_joint)?;
    if let Some(counts) = counts.as_deref_mut() { counts.mv_joint[joint as usize] += 1; }
    let row = if joint == 2 || joint == 3 {
        read_motion_component(bits, probabilities, 0, high_precision, counts.as_deref_mut())?
    } else {
        0
    };
    let col = if joint == 1 || joint == 3 {
        read_motion_component(bits, probabilities, 1, high_precision, counts.as_deref_mut())?
    } else {
        0
    };
    Ok((row, col))
}

fn read_motion_component(
    bits: &mut BoolDecoder<'_>,
    probabilities: &InterframeProbabilities,
    axis: usize,
    high_precision: bool,
    mut counts: Option<&mut NonCoefficientCounts>,
) -> Result<i32, MediaDecodeError> {
    let negative = bits.read(probabilities.mv_sign[axis])?;
    if let Some(counts) = counts.as_deref_mut() { counts.mv_sign[axis][usize::from(negative)] += 1; }
    let class = read_tree(bits, &MV_CLASS_TREE, &probabilities.mv_class[axis])?;
    if let Some(counts) = counts.as_deref_mut() { counts.mv_class[axis][class as usize] += 1; }
    let magnitude = if class == 0 {
        let class0 = usize::from(bits.read(probabilities.mv_class0_bit[axis])?);
        if let Some(counts) = counts.as_deref_mut() { counts.mv_class0_bit[axis][class0] += 1; }
        let fraction = i32::from(read_tree(
            bits, &MV_FR_TREE, &probabilities.mv_class0_fr[axis][class0],
        )?);
        if let Some(counts) = counts.as_deref_mut() { counts.mv_class0_fr[axis][class0][fraction as usize] += 1; }
        let high = if high_precision {
            let bit = bits.read(probabilities.mv_class0_hp[axis])?;
            i32::from(bit)
        } else {
            1
        };
        if let Some(counts) = counts.as_deref_mut() {
            counts.mv_class0_hp[axis][high as usize] += 1;
        }
        (((class0 as i32) << 3) | (fraction << 1) | high) + 1
    } else {
        let mut offset = 0i32;
        for index in 0..class as usize {
            let bit = bits.read(probabilities.mv_bits[axis][index])?;
            if let Some(counts) = counts.as_deref_mut() { counts.mv_bits[axis][index][usize::from(bit)] += 1; }
            if bit {
                offset |= 1 << index;
            }
        }
        let fraction = i32::from(read_tree(bits, &MV_FR_TREE, &probabilities.mv_fr[axis])?);
        if let Some(counts) = counts.as_deref_mut() { counts.mv_fr[axis][fraction as usize] += 1; }
        let high = if high_precision {
            let bit = bits.read(probabilities.mv_hp[axis])?;
            i32::from(bit)
        } else {
            1
        };
        if let Some(counts) = counts.as_deref_mut() {
            counts.mv_hp[axis][high as usize] += 1;
        }
        (2 << (class as i32 + 2)) + ((offset << 3) | (fraction << 1) | high) + 1
    };
    Ok(if negative { -magnitude } else { magnitude })
}

fn read_tree(
    bits: &mut BoolDecoder<'_>, tree: &[i8], probabilities: &[u8],
) -> Result<u8, MediaDecodeError> {
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

pub(super) fn scaled_motion(
    frame_size: (usize, usize),
    reference_size: (usize, usize),
    position: (usize, usize),
    mi_position: (usize, usize),
    mi_size: (usize, usize),
    mv: (i32, i32),
    chroma: bool,
) -> Result<MotionSampling, MediaDecodeError> {
    let (frame_width, frame_height) = frame_size;
    let (reference_width, reference_height) = reference_size;
    if frame_width == 0 || frame_height == 0 || reference_width == 0 || reference_height == 0
        || frame_width > 65_536 || frame_height > 65_536
        || reference_width > 65_536 || reference_height > 65_536
        || 2 * frame_width < reference_width || 2 * frame_height < reference_height
        || frame_width > 16 * reference_width || frame_height > 16 * reference_height
        || mi_size.0 == 0 || mi_size.1 == 0
    {
        return Err(MediaDecodeError::InvalidData("invalid VP9 reference scaling".into()));
    }
    let subsampling = i64::from(chroma);
    let mi_cols = frame_width.div_ceil(8) as i64;
    let mi_rows = frame_height.div_ceil(8) as i64;
    let (mi_col, mi_row) = (mi_position.0 as i64, mi_position.1 as i64);
    let (block_width, block_height) = (mi_size.0 as i64, mi_size.1 as i64);
    if mi_col >= mi_cols || mi_row >= mi_rows || block_width > 8 || block_height > 8 {
        return Err(MediaDecodeError::InvalidData("invalid VP9 motion block".into()));
    }
    let top = -(mi_row * 8 * 16) >> subsampling;
    let bottom = ((mi_rows - block_height - mi_row) * 8 * 16) >> subsampling;
    let left = -(mi_col * 8 * 16) >> subsampling;
    let right = ((mi_cols - block_width - mi_col) * 8 * 16) >> subsampling;
    let extend_y = (4 + ((block_height * 8) >> subsampling)) << 4;
    let extend_x = (4 + ((block_width * 8) >> subsampling)) << 4;
    let mv_row = ((2 * i64::from(mv.0)) >> subsampling)
        .clamp(top - extend_y, bottom + extend_y - 16);
    let mv_col = ((2 * i64::from(mv.1)) >> subsampling)
        .clamp(left - extend_x, right + extend_x - 16);
    let x_scale = ((reference_width as i64) << 14) / frame_width as i64;
    let y_scale = ((reference_height as i64) << 14) / frame_height as i64;
    let (x, y) = (position.0 as i64, position.1 as i64);
    let base_x = (x * x_scale) >> 14;
    let base_y = (y * y_scale) >> 14;
    let fraction_x = ((16 * (x << subsampling) * x_scale) >> 14) & 15;
    let fraction_y = ((16 * (y << subsampling) * y_scale) >> 14) & 15;
    let start_x = (base_x << 4) + ((mv_col * x_scale) >> 14) + fraction_x;
    let start_y = (base_y << 4) + ((mv_row * y_scale) >> 14) + fraction_y;
    let step_x = (16 * x_scale) >> 14;
    let step_y = (16 * y_scale) >> 14;
    Ok(MotionSampling {
        start_x: i32::try_from(start_x).map_err(|_| MediaDecodeError::Unsupported)?,
        start_y: i32::try_from(start_y).map_err(|_| MediaDecodeError::Unsupported)?,
        step_x: i32::try_from(step_x).map_err(|_| MediaDecodeError::Unsupported)?,
        step_y: i32::try_from(step_y).map_err(|_| MediaDecodeError::Unsupported)?,
    })
}

pub(super) fn predict_block(
    destination: &mut Plane,
    reference: &Plane,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    start_x: i32,
    start_y: i32,
    step_x: i32,
    step_y: i32,
    filter: u8,
    visible_width: usize,
    visible_height: usize,
) -> Result<(), MediaDecodeError> {
    if reference.width == 0 || destination.width == 0
        || usize::from(filter) >= SUBPEL_FILTERS.len()
        || visible_width == 0 || visible_height == 0
        || visible_width > reference.width
        || visible_height > reference.pixels.len() / reference.width
        || x.checked_add(width).is_none_or(|end| end > destination.width)
        || y.checked_add(height).is_none_or(|end| end > destination.pixels.len() / destination.width)
        || width == 0 || height == 0 || width > 64 || height > 64
        || !(1..=80).contains(&step_x) || !(1..=80).contains(&step_y)
    {
        return Err(MediaDecodeError::InvalidData("invalid VP9 prediction bounds".into()));
    }
    let source_x = start_x.div_euclid(16);
    let source_y = start_y.div_euclid(16);
    if step_x == 16 && step_y == 16 && start_x.rem_euclid(16) == 0 && start_y.rem_euclid(16) == 0
        && source_x >= 0 && source_y >= 0
        && source_x as usize + width <= visible_width
        && source_y as usize + height <= visible_height
    {
        for row in 0..height {
            let from = (source_y as usize + row) * reference.width + source_x as usize;
            let to = (y + row) * destination.width + x;
            destination.pixels[to..to + width].copy_from_slice(&reference.pixels[from..from + width]);
        }
        return Ok(());
    }
    let intermediate_height = (((height - 1) as i32 * step_y + 15) >> 4) as usize + 8;
    let mut scratch = [0u8; 64 * 72];
    let mut scaled_scratch = Vec::new();
    let scratch_len = width * intermediate_height;
    let intermediate = if scratch_len <= scratch.len() {
        &mut scratch[..scratch_len]
    } else {
        scaled_scratch.resize(scratch_len, 0);
        scaled_scratch.as_mut_slice()
    };
    let filters = &SUBPEL_FILTERS[filter as usize];
    let mut columns = [[0usize; 8]; 64];
    let mut phases = [0usize; 64];
    for col in 0..width {
        let position = start_x + step_x * col as i32;
        phases[col] = (position & 15) as usize;
        for tap in 0..8 {
            columns[col][tap] = ((position >> 4) + tap as i32 - 3)
                .clamp(0, visible_width as i32 - 1) as usize;
        }
    }
    let contiguous = step_x == 16 && source_x >= 3
        && source_x as usize + width + 4 <= visible_width;
    for row in 0..intermediate_height {
        let source_row = (start_y >> 4) + row as i32 - 3;
        let source_row = source_row.clamp(0, visible_height as i32 - 1) as usize;
        let source = &reference.pixels[source_row * reference.width..][..visible_width];
        if contiguous {
            let target = &mut intermediate[row * width..(row + 1) * width];
            let phase = phases[0];
            if phase == 0 {
                target.copy_from_slice(&source[source_x as usize..source_x as usize + width]);
            } else {
                let mut sums = [0i32; 64];
                for (tap, &coefficient) in filters[phase].iter().enumerate() {
                    if coefficient == 0 { continue; }
                    let from = source_x as usize + tap - 3;
                    for (sum, &sample) in sums[..width].iter_mut().zip(&source[from..from + width]) {
                        *sum += i32::from(sample) * i32::from(coefficient);
                    }
                }
                for (sample, sum) in target.iter_mut().zip(sums) {
                    *sample = ((sum + 64) >> 7).clamp(0, 255) as u8;
                }
            }
            continue;
        }
        for col in 0..width {
            let phase = phases[col];
            if phase == 0 {
                intermediate[row * width + col] = source[columns[col][3]];
                continue;
            }
            let mut sum = 0i32;
            for (tap, coefficient) in filters[phase].iter().enumerate() {
                sum += i32::from(source[columns[col][tap]]) * i32::from(*coefficient);
            }
            intermediate[row * width + col] = ((sum + 64) >> 7).clamp(0, 255) as u8;
        }
    }
    for row in 0..height {
        let position = (start_y & 15) + step_y * row as i32;
        let phase = (position & 15) as usize;
        let base = (position >> 4) as usize;
        if phase == 0 {
            let to = (y + row) * destination.width + x;
            destination.pixels[to..to + width]
                .copy_from_slice(&intermediate[(base + 3) * width..(base + 4) * width]);
            continue;
        }
        let mut sums = [0i32; 64];
        for (tap, &coefficient) in filters[phase].iter().enumerate() {
            if coefficient == 0 { continue; }
            let source = &intermediate[(base + tap) * width..(base + tap + 1) * width];
            for (sum, &sample) in sums[..width].iter_mut().zip(source) {
                *sum += i32::from(sample) * i32::from(coefficient);
            }
        }
        let to = (y + row) * destination.width + x;
        for (sample, sum) in destination.pixels[to..to + width].iter_mut().zip(sums) {
            *sample = ((sum + 64) >> 7).clamp(0, 255) as u8;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_prediction(reference: &Plane, width: usize, height: usize,
        start_x: i32, start_y: i32, step_x: i32, step_y: i32, filter: u8) -> Vec<u8>
    {
        let reference_height = reference.pixels.len() / reference.width;
        let intermediate_height = (((height - 1) as i32 * step_y + 15) >> 4) as usize + 8;
        let filters = &SUBPEL_FILTERS[filter as usize];
        let mut intermediate = vec![0u8; width * intermediate_height];
        for row in 0..intermediate_height {
            let source_row = ((start_y >> 4) + row as i32 - 3)
                .clamp(0, reference_height as i32 - 1) as usize;
            for col in 0..width {
                let position = start_x + step_x * col as i32;
                let mut sum = 0i32;
                for (tap, coefficient) in filters[(position & 15) as usize].iter().enumerate() {
                    let source_col = ((position >> 4) + tap as i32 - 3)
                        .clamp(0, reference.width as i32 - 1) as usize;
                    sum += i32::from(reference.pixels[source_row * reference.width + source_col])
                        * i32::from(*coefficient);
                }
                intermediate[row * width + col] = ((sum + 64) >> 7).clamp(0, 255) as u8;
            }
        }
        let mut output = vec![0u8; width * height];
        for row in 0..height {
            let position = (start_y & 15) + step_y * row as i32;
            for col in 0..width {
                let sum: i32 = filters[(position & 15) as usize].iter().enumerate()
                    .map(|(tap, coefficient)| i32::from(intermediate[((position >> 4) as usize + tap) * width + col])
                        * i32::from(*coefficient)).sum();
                output[row * width + col] = ((sum + 64) >> 7).clamp(0, 255) as u8;
            }
        }
        output
    }

    #[test]
    fn cached_prediction_matches_scalar_at_edges_and_scaled_references() {
        let mut reference = Plane::new(80, 80);
        for (index, value) in reference.pixels.iter_mut().enumerate() {
            *value = (index.wrapping_mul(73) ^ (index >> 3)) as u8;
        }
        for size in [4, 8, 16, 32, 64] {
            for filter in 0..4 {
                for (start_x, start_y) in [(-23, -17), (0, 0), (9, 7), (64, 64), (73, 71), (1152, 1216)] {
                    for (step_x, step_y) in [(16, 16), (7, 31), (80, 80)] {
                        let expected = scalar_prediction(&reference, size, size,
                            start_x, start_y, step_x, step_y, filter);
                        let mut actual = Plane::new(size, size);
                        predict_block(&mut actual, &reference, 0, 0, size, size,
                            start_x, start_y, step_x, step_y, filter, 80, 80).unwrap();
                        assert_eq!(actual.pixels, expected,
                            "size={size} filter={filter} start=({start_x},{start_y}) step=({step_x},{step_y})");
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "manual release-mode motion prediction benchmark"]
    fn benchmark_cached_motion_prediction() {
        let mut reference = Plane::new(80, 80);
        for (index, value) in reference.pixels.iter_mut().enumerate() {
            *value = index.wrapping_mul(73) as u8;
        }
        let mut actual = Plane::new(32, 32);
        for _ in 0..3 {
            let start = std::time::Instant::now();
            for i in 0..4000 {
                std::hint::black_box(scalar_prediction(std::hint::black_box(&reference),
                    32, 32, 64 + (i & 15), 64 + ((i >> 4) & 15), 16, 16, 0));
            }
            let scalar = start.elapsed();
            let start = std::time::Instant::now();
            for i in 0..4000 {
                predict_block(std::hint::black_box(&mut actual), std::hint::black_box(&reference),
                    0, 0, 32, 32, 64 + (i & 15), 64 + ((i >> 4) & 15), 16, 16, 0, 80, 80).unwrap();
                std::hint::black_box(&actual.pixels);
            }
            eprintln!("VP9 motion: scalar={scalar:?} cached={:?}", start.elapsed());
        }
    }

    #[test]
    fn filter_rows_preserve_constant_samples() {
        for filter in SUBPEL_FILTERS {
            for phase in filter {
                assert_eq!(phase.iter().map(|&value| i32::from(value)).sum::<i32>(), 128);
            }
        }
        let mut reference = Plane::new(16, 16);
        reference.pixels.fill(73);
        let mut destination = Plane::new(16, 16);
        for filter in 0..4 {
            predict_block(&mut destination, &reference, 0, 0, 16, 16, -7, 5, 16, 16, filter, 16, 16).unwrap();
            assert!(destination.pixels.iter().all(|&sample| sample == 73));
        }
    }

    #[test]
    fn whole_pixel_copy_and_bilinear_half_pixel() {
        let mut reference = Plane::new(16, 16);
        for row in 0..16 {
            for col in 0..16 {
                reference.pixels[row * 16 + col] = (col * 10 + row) as u8;
            }
        }
        let mut destination = Plane::new(16, 16);
        predict_block(&mut destination, &reference, 4, 4, 4, 4, 4 * 16, 4 * 16, 16, 16, 1, 16, 16).unwrap();
        assert_eq!(destination.pixels[4 * 16 + 4], 44);
        predict_block(&mut destination, &reference, 4, 4, 4, 4, 4 * 16 + 8, 4 * 16, 16, 16, 3, 16, 16).unwrap();
        assert_eq!(destination.pixels[4 * 16 + 4], 49);
    }

    #[test]
    fn same_size_motion_uses_sixteenth_sample_coordinates() {
        let luma = scaled_motion((1920, 1080), (1920, 1080), (32, 16),
            (4, 2), (8, 8), (4, -4), false).unwrap();
        assert_eq!(luma, MotionSampling { start_x: 32 * 16 - 8, start_y: 16 * 16 + 8,
            step_x: 16, step_y: 16 });
        let chroma = scaled_motion((1920, 1080), (1920, 1080), (16, 8),
            (4, 2), (8, 8), (4, -4), true).unwrap();
        assert_eq!(chroma, MotionSampling { start_x: 16 * 16 - 4, start_y: 8 * 16 + 4,
            step_x: 16, step_y: 16 });
    }

    #[test]
    fn zero_motion_joint_and_high_precision_threshold() {
        let probabilities = InterframeProbabilities::default();
        let mut bits = BoolDecoder::new(&[0; 8]).unwrap();
        assert!(!bits.read_bit().unwrap());
        assert_eq!(read_motion_difference(&mut bits, &probabilities, true).unwrap(), (0, 0));
        assert!(use_high_precision((63, -63)));
        assert!(!use_high_precision((64, 0)));
    }
}
