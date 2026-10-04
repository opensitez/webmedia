//! CELT pyramid-vector decoding from RFC 6716 section 4.3.4.2.

use super::range::RangeDecoder;
use crate::video::backend::MediaDecodeError;

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.into())
}

/// A bounded pulse codebook. Large spectral vectors must be split before coding.
pub struct Codebook {
    dimensions: usize,
    pulses: usize,
    counts: Vec<u64>,
    entries: u32,
}

impl Codebook {
    pub fn new(dimensions: usize, pulses: usize) -> Result<Self, MediaDecodeError> {
        if !(1..=176).contains(&dimensions) || pulses > 32767 {
            return Err(invalid("invalid Opus pulse-vector dimensions"));
        }
        let columns = pulses + 1;
        let cells = (dimensions + 1) * columns;
        if cells > 1 << 18 {
            return Err(invalid("Opus pulse codebook exceeds memory budget"));
        }
        let mut counts = vec![0u64; cells];
        counts[0] = 1;
        // Saturated counts can never occur in an ancestor of a valid 32-bit
        // codebook, and avoid overflow when rejecting unsplit large vectors.
        let limit = u64::from(u32::MAX) + 1;
        for n in 1..=dimensions {
            let row = n * columns;
            let previous = row - columns;
            counts[row] = 1;
            for k in 1..=pulses {
                counts[row + k] =
                    (counts[previous + k] + counts[row + k - 1] + counts[previous + k - 1])
                        .min(limit);
            }
        }
        let entries = counts[dimensions * columns + pulses];
        if entries > u64::from(u32::MAX) {
            return Err(invalid("Opus pulse codebook requires spectral splitting"));
        }
        Ok(Self {
            dimensions,
            pulses,
            counts,
            entries: entries as u32,
        })
    }

    pub fn entries(&self) -> u32 {
        self.entries
    }

    fn count(&self, dimensions: usize, pulses: usize) -> u64 {
        self.counts[dimensions * (self.pulses + 1) + pulses]
    }

    pub fn unrank(&self, index: u32, output: &mut [i32]) -> Result<(), MediaDecodeError> {
        if index >= self.entries || output.len() != self.dimensions {
            return Err(invalid("invalid Opus pulse-vector index or output size"));
        }
        let mut index = u64::from(index);
        let mut pulses = self.pulses;
        for (position, value) in output.iter_mut().enumerate() {
            let dimensions = self.dimensions - position;
            let remainder = self.count(dimensions - 1, pulses);
            let mut boundary = (remainder + self.count(dimensions, pulses)) / 2;
            let sign = if index < boundary {
                1
            } else {
                index -= boundary;
                -1
            };
            let previous = pulses;
            boundary -= remainder;
            while boundary > index {
                pulses -= 1;
                boundary -= self.count(dimensions - 1, pulses);
            }
            *value = sign * (previous - pulses) as i32;
            index -= boundary;
        }
        debug_assert_eq!(pulses, 0);
        debug_assert_eq!(index, 0);
        Ok(())
    }

    pub fn decode_pulses(
        &self,
        entropy: &mut RangeDecoder<'_>,
        output: &mut [i32],
    ) -> Result<(), MediaDecodeError> {
        if output.len() != self.dimensions {
            return Err(invalid("invalid Opus pulse-vector output size"));
        }
        let index = entropy.uniform(self.entries)?;
        self.unrank(index, output)
    }

    /// Zero-pulse bands need spectral folding or noise, not normalization.
    pub fn decode_shape(
        &self,
        entropy: &mut RangeDecoder<'_>,
        output: &mut [f64],
    ) -> Result<(), MediaDecodeError> {
        if self.pulses == 0 || output.len() != self.dimensions {
            return Err(invalid("invalid Opus normalized pulse-vector shape"));
        }
        let mut vector = vec![0; self.dimensions];
        self.decode_pulses(entropy, &mut vector)?;
        let energy: f64 = vector.iter().map(|&value| f64::from(value).powi(2)).sum();
        let gain = energy.sqrt().recip();
        for (sample, value) in output.iter_mut().zip(vector) {
            *sample = f64::from(value) * gain;
        }
        Ok(())
    }
}

/// Rotate normalized shapes to spread quantization noise (RFC 6716 4.3.4.3).
/// Short-transform coefficients are interleaved by time block.
pub fn spread(
    vector: &mut [f64],
    pulses: u32,
    blocks: usize,
    mode: u8,
) -> Result<(), MediaDecodeError> {
    if vector.is_empty()
        || vector.len() > 176
        || !blocks.is_power_of_two()
        || blocks > 64
        || vector.len() % blocks != 0
        || mode > 3
        || vector.iter().any(|value| !value.is_finite())
    {
        return Err(invalid("invalid Opus spreading shape"));
    }
    if mode == 0 {
        return Ok(());
    }
    let factor = [0.0, 15.0, 10.0, 5.0][mode as usize];
    let gain = vector.len() as f64 / (vector.len() as f64 + factor * f64::from(pulses));
    let angle = std::f64::consts::FRAC_PI_4 * gain * gain;
    let length = vector.len() / blocks;
    let mut output = vector.to_vec();
    for block in 0..blocks {
        if length >= 8 {
            let stride = (length as f64).sqrt().round() as usize;
            for offset in 0..stride {
                let count = (length - offset).div_ceil(stride);
                rotate_chain(
                    &mut output,
                    block + offset * blocks,
                    stride * blocks,
                    count,
                    std::f64::consts::FRAC_PI_2 - angle,
                );
            }
        }
        rotate_chain(&mut output, block, blocks, length, angle);
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(invalid("Opus spreading overflow"));
    }
    vector.copy_from_slice(&output);
    Ok(())
}

fn rotate_chain(vector: &mut [f64], start: usize, stride: usize, count: usize, angle: f64) {
    let (sine, cosine) = angle.sin_cos();
    let mut pair = |index: usize| {
        let left = start + index * stride;
        let right = left + stride;
        let (x, y) = (vector[left], vector[right]);
        vector[left] = cosine * x + sine * y;
        vector[right] = -sine * x + cosine * y;
    };
    for index in 0..count.saturating_sub(1) {
        pair(index);
    }
    for index in (0..count.saturating_sub(2)).rev() {
        pair(index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn enumerate(
        dimensions: usize,
        pulses: i32,
        prefix: &mut Vec<i32>,
        vectors: &mut HashSet<Vec<i32>>,
    ) {
        if dimensions == 0 {
            if pulses == 0 {
                vectors.insert(prefix.clone());
            }
            return;
        }
        for value in -pulses..=pulses {
            prefix.push(value);
            enumerate(dimensions - 1, pulses - value.abs(), prefix, vectors);
            prefix.pop();
        }
    }

    #[test]
    fn exhaustive_small_codebooks_cover_every_vector_once() {
        for dimensions in 1..=6 {
            for pulses in 0..=6 {
                let book = Codebook::new(dimensions, pulses).unwrap();
                let mut expected = HashSet::new();
                enumerate(dimensions, pulses as i32, &mut Vec::new(), &mut expected);
                assert_eq!(book.entries() as usize, expected.len());
                let mut found = HashSet::new();
                for index in 0..book.entries() {
                    let mut output = vec![0; dimensions];
                    book.unrank(index, &mut output).unwrap();
                    assert_eq!(
                        output
                            .iter()
                            .map(|value| value.abs() as usize)
                            .sum::<usize>(),
                        pulses
                    );
                    assert!(found.insert(output), "duplicate index {index}");
                }
                assert_eq!(found, expected);
            }
        }
    }

    #[test]
    fn indexing_matches_normative_sign_and_coordinate_order() {
        let book = Codebook::new(2, 2).unwrap();
        for (index, expected) in [
            [2, 0],
            [1, 1],
            [1, -1],
            [0, 2],
            [0, -2],
            [-2, 0],
            [-1, 1],
            [-1, -1],
        ]
        .into_iter()
        .enumerate()
        {
            let mut output = [0; 2];
            book.unrank(index as u32, &mut output).unwrap();
            assert_eq!(output, expected);
        }
        for pulses in [1, 2, 8, 1024, 32767] {
            assert_eq!(Codebook::new(1, pulses).unwrap().entries(), 2);
            assert_eq!(
                Codebook::new(2, pulses).unwrap().entries(),
                4 * pulses as u32
            );
        }
        assert_eq!(Codebook::new(176, 1).unwrap().entries(), 352);
    }

    #[test]
    fn range_decoded_vectors_have_unit_energy() {
        for seed in 0u32..64 {
            let bytes: Vec<_> = (0..64)
                .map(|position| (seed * 73 + position * 137) as u8)
                .collect();
            for dimensions in 1..=16 {
                let book = Codebook::new(dimensions, 3).unwrap();
                let mut entropy = RangeDecoder::new(&bytes);
                let mut output = vec![0.0; dimensions];
                let mut reference = entropy.clone();
                let Ok(index) = reference.uniform(book.entries()) else {
                    assert!(book.decode_shape(&mut entropy, &mut output).is_err());
                    continue;
                };
                book.decode_shape(&mut entropy, &mut output).unwrap();
                let mut expected = vec![0; dimensions];
                book.unrank(index, &mut expected).unwrap();
                let norm = expected
                    .iter()
                    .map(|&value| f64::from(value).powi(2))
                    .sum::<f64>()
                    .sqrt();
                for (&sample, value) in output.iter().zip(expected) {
                    assert!((sample - f64::from(value) / norm).abs() < 1e-12);
                }
                assert!(
                    (output.iter().map(|value| value * value).sum::<f64>() - 1.0).abs() < 1e-12
                );
            }
        }
    }

    #[test]
    fn rejects_unsplit_or_invalid_codebooks_without_unbounded_allocation() {
        for (dimensions, pulses) in [(0, 1), (177, 1), (1, 32768), (176, 32767), (176, 10)] {
            assert!(Codebook::new(dimensions, pulses).is_err());
        }
        let book = Codebook::new(2, 1).unwrap();
        assert!(book.unrank(4, &mut [0; 2]).is_err());
        assert!(book.unrank(0, &mut [0; 3]).is_err());
        let zero = Codebook::new(2, 0).unwrap();
        let mut output = [1; 2];
        zero.decode_pulses(&mut RangeDecoder::new(&[]), &mut output)
            .unwrap();
        assert_eq!(output, [0; 2]);
        assert!(
            zero.decode_shape(&mut RangeDecoder::new(&[]), &mut [0.0; 2])
                .is_err()
        );
    }

    #[test]
    fn spreading_matches_two_dimensional_rotation() {
        for mode in 1..=3 {
            let factor = [0.0, 15.0, 10.0, 5.0][mode as usize];
            let gain: f64 = 2.0 / (2.0 + factor * 3.0);
            let angle = std::f64::consts::FRAC_PI_4 * gain * gain;
            let mut vector = [0.6, -0.8];
            spread(&mut vector, 3, 1, mode).unwrap();
            assert!((vector[0] - (angle.cos() * 0.6 - angle.sin() * 0.8)).abs() < 1e-12);
            assert!((vector[1] - (-angle.sin() * 0.6 - angle.cos() * 0.8)).abs() < 1e-12);
        }
    }

    #[test]
    fn spreading_matches_composed_rotation_matrices_in_each_block() {
        // Independent dense matrix construction from RFC 6716 section
        // 4.3.4.3's elementary rotation and its ordered pair list.
        for length in [3usize, 7, 8, 11, 22] {
            for blocks in [1, 2, 4, 8, 16] {
                if length * blocks > 176 {
                    continue;
                }
                for (mode, factor) in [(1, 15.0), (2, 10.0), (3, 5.0)] {
                    let dimensions = (length * blocks) as f64;
                    let gain = dimensions / (dimensions + factor * 7.0);
                    let angle = std::f64::consts::FRAC_PI_4 * gain * gain;
                    let mut matrix = vec![vec![0.0; length]; length];
                    for (row, values) in matrix.iter_mut().enumerate() {
                        values[row] = 1.0;
                    }
                    let mut chains: Vec<(Vec<usize>, f64)> = Vec::new();
                    if length >= 8 {
                        let stride = (length as f64).sqrt().round() as usize;
                        for offset in 0..stride {
                            chains.push((
                                (offset..length).step_by(stride).collect(),
                                std::f64::consts::FRAC_PI_2 - angle,
                            ));
                        }
                    }
                    chains.push(((0..length).collect(), angle));
                    for (indices, theta) in chains {
                        let forward: Vec<_> = indices.windows(2).map(|p| (p[0], p[1])).collect();
                        let reverse: Vec<_> = forward.iter().rev().skip(1).copied().collect();
                        for (a, b) in forward.into_iter().chain(reverse) {
                            let row_a = matrix[a].clone();
                            let row_b = matrix[b].clone();
                            for column in 0..length {
                                matrix[a][column] =
                                    theta.cos() * row_a[column] + theta.sin() * row_b[column];
                                matrix[b][column] =
                                    -theta.sin() * row_a[column] + theta.cos() * row_b[column];
                            }
                        }
                    }
                    let input: Vec<_> = (0..length * blocks)
                        .map(|i| (i as f64 * 0.71 + 0.13).sin())
                        .collect();
                    let mut actual = input.clone();
                    spread(&mut actual, 7, blocks, mode).unwrap();
                    for block in 0..blocks {
                        for row in 0..length {
                            let expected: f64 = (0..length)
                                .map(|column| matrix[row][column] * input[column * blocks + block])
                                .sum();
                            assert!((actual[row * blocks + block] - expected).abs() < 1e-12);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn spreading_overflow_preserves_the_input() {
        let mut vector = [f64::MAX; 2];
        assert!(spread(&mut vector, 1, 1, 3).is_err());
        assert_eq!(vector, [f64::MAX; 2]);
    }

    #[test]
    fn spreading_preserves_each_time_blocks_energy() {
        for blocks in [1, 2, 4, 8] {
            for length in [1, 2, 7, 8, 11, 22] {
                for mode in 0..=3 {
                    let mut vector: Vec<_> = (0..blocks * length)
                        .map(|index| ((index * 37 % 101) as f64 - 50.0) / 50.0)
                        .collect();
                    let original = vector.clone();
                    spread(&mut vector, 7, blocks, mode).unwrap();
                    for block in 0..blocks {
                        let energy = |samples: &[f64]| {
                            samples[block..]
                                .iter()
                                .step_by(blocks)
                                .map(|value| value * value)
                                .sum::<f64>()
                        };
                        assert!((energy(&original) - energy(&vector)).abs() < 1e-12);
                    }
                    if mode == 0 || length == 1 {
                        assert_eq!(vector, original);
                    }
                }
            }
        }
        let mut vector = [1.0, 2.0, 3.0];
        for (blocks, mode) in [(0, 1), (2, 1), (1, 4)] {
            assert!(spread(&mut vector, 1, blocks, mode).is_err());
            assert_eq!(vector, [1.0, 2.0, 3.0]);
        }
        assert!(spread(&mut [f64::NAN], 1, 1, 1).is_err());
    }
}
