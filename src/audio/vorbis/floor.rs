//! Vorbis floor-one packet amplitudes and wrapped prediction (section 7.2).

use super::entropy::{Codebook, PacketBits};
use super::setup::FloorOne;
use crate::video::backend::MediaDecodeError;

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.into())
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Amplitudes {
    pub values: Vec<u16>,
    pub active: Vec<bool>,
}

impl FloorOne {
    /// Write the linear-amplitude envelope into caller-owned spectral storage.
    pub fn render_curve(
        &self,
        amplitudes: &Amplitudes,
        output: &mut [f64],
    ) -> Result<(), MediaDecodeError> {
        if !(1..=4).contains(&self.multiplier)
            || output.len() > 4096
            || amplitudes.values.len() != self.x.len()
            || amplitudes.active.len() != self.x.len()
            || self.sorted_points.len() != self.x.len()
            || self.x.len() < 2
        {
            return Err(invalid("invalid Vorbis floor curve configuration"));
        }
        let mut previous_x = None;
        for &point in &self.sorted_points {
            let x = self
                .x
                .get(point)
                .copied()
                .ok_or_else(|| invalid("invalid Vorbis floor sort index"))?;
            if x > 32768 || previous_x.is_some_and(|previous| previous >= x) {
                return Err(invalid("unsorted or duplicate Vorbis floor curve points"));
            }
            if usize::from(amplitudes.values[point]) * self.multiplier > 255 {
                return Err(invalid("invalid Vorbis floor curve amplitude"));
            }
            previous_x = Some(x);
        }
        if self.x[0] != 0 || !amplitudes.active[0] || !amplitudes.active[1] {
            return Err(invalid("missing Vorbis floor curve endpoints"));
        }
        let mut left_x = 0;
        let mut left_y = i32::from(amplitudes.values[0]) * self.multiplier as i32;
        for &point in self.sorted_points.iter().skip(1) {
            if !amplitudes.active[point] {
                continue;
            }
            let right_x = self.x[point];
            let right_y = i32::from(amplitudes.values[point]) * self.multiplier as i32;
            render_line(left_x, left_y, right_x, right_y, output);
            left_x = right_x;
            left_y = right_y;
        }
        if left_x < output.len() {
            output[left_x..].fill(f64::from(INVERSE_DB[left_y as usize]));
        }
        Ok(())
    }

    /// None denotes an unused floor, including nominally truncated audio packets.
    pub fn decode_amplitudes(
        &self,
        bits: &mut PacketBits<'_>,
        books: &[Codebook],
    ) -> Result<Option<Amplitudes>, MediaDecodeError> {
        let mut amplitudes = Amplitudes::default();
        let mut values = Vec::new();
        Ok(self
            .decode_amplitudes_into(bits, books, &mut amplitudes, &mut values)?
            .then_some(amplitudes))
    }

    pub(super) fn decode_amplitudes_into(
        &self,
        bits: &mut PacketBits<'_>,
        books: &[Codebook],
        amplitudes: &mut Amplitudes,
        values: &mut Vec<i64>,
    ) -> Result<bool, MediaDecodeError> {
        if bits.read(1) != Some(1) {
            return Ok(false);
        }
        let range = match self.multiplier {
            1 => 256i64,
            2 => 128,
            3 => 86,
            4 => 64,
            _ => return Err(invalid("invalid Vorbis floor multiplier")),
        };
        if !(2..=65).contains(&self.x.len())
            || self.neighbors.len() != self.x.len()
            || self.x.iter().any(|&value| value > 32768)
        {
            return Err(invalid("invalid Vorbis floor point configuration"));
        }
        let width = (64 - (range as u64 - 1).leading_zeros()) as u8;
        values.clear();
        values.reserve(self.x.len());
        for _ in 0..2 {
            let Some(value) = bits.read(width) else {
                return Ok(false);
            };
            values.push(i64::from(value).min(range - 1));
        }
        for &class in &self.partitions {
            let class = self
                .classes
                .get(class)
                .ok_or_else(|| invalid("invalid Vorbis floor class"))?;
            if !(1..=8).contains(&class.dimensions)
                || class.subclasses > 3
                || class.books.len() != 1 << class.subclasses
                || values.len() + class.dimensions > self.x.len()
            {
                return Err(invalid("invalid Vorbis floor subclass configuration"));
            }
            let mut selection = if class.subclasses != 0 {
                let book = class
                    .masterbook
                    .and_then(|index| books.get(index))
                    .ok_or_else(|| invalid("invalid Vorbis floor masterbook"))?;
                let Some(value) = book.huffman.decode(bits) else {
                    return Ok(false);
                };
                value
            } else {
                0
            };
            let mask = (1 << class.subclasses) - 1;
            for _ in 0..class.dimensions {
                let value = if let Some(index) = class.books[selection & mask] {
                    let book = books
                        .get(index)
                        .ok_or_else(|| invalid("invalid Vorbis floor subclass book"))?;
                    let Some(value) = book.huffman.decode(bits) else {
                        return Ok(false);
                    };
                    value as i64
                } else {
                    0
                };
                values.push(value);
                selection >>= class.subclasses;
            }
        }
        if values.len() != self.x.len() {
            return Err(invalid("incomplete Vorbis floor point list"));
        }
        amplitudes.active.resize(values.len(), false);
        amplitudes.active.fill(false);
        let active = &mut amplitudes.active;
        active[..2].fill(true);
        for index in 2..values.len() {
            let (low, high) = self.neighbors[index];
            if low >= index
                || high >= index
                || self.x[low] >= self.x[index]
                || self.x[high] <= self.x[index]
            {
                return Err(invalid("invalid Vorbis floor prediction neighbors"));
            }
            let predicted = values[low]
                + (values[high] - values[low]) * (self.x[index] - self.x[low]) as i64
                    / (self.x[high] - self.x[low]) as i64;
            let value = values[index];
            if value == 0 {
                values[index] = predicted;
                continue;
            }
            active[low] = true;
            active[high] = true;
            active[index] = true;
            let highroom = range - predicted;
            let lowroom = predicted;
            let room = 2 * highroom.min(lowroom);
            values[index] = if value >= room {
                if highroom > lowroom {
                    value - lowroom + predicted
                } else {
                    predicted - value + highroom - 1
                }
            } else if value & 1 != 0 {
                predicted - (value + 1) / 2
            } else {
                predicted + value / 2
            };
            values[index] = values[index].clamp(0, range - 1);
        }
        amplitudes.values.clear();
        amplitudes
            .values
            .extend(values.iter().map(|&value| value as u16));
        Ok(true)
    }
}

fn render_line(x0: usize, y0: i32, x1: usize, y1: i32, output: &mut [f64]) {
    if x0 >= output.len() {
        return;
    }
    let distance = (x1 - x0) as i32;
    let delta = y1 - y0;
    let base = delta / distance;
    let step = base + if delta < 0 { -1 } else { 1 };
    let remainder = delta.abs() - base.abs() * distance;
    let mut error = 0;
    let mut y = y0;
    let end = x1.min(output.len());
    for value in &mut output[x0..end] {
        *value = f64::from(INVERSE_DB[y as usize]);
        error += remainder;
        if error >= distance {
            error -= distance;
            y += step;
        } else {
            y += base;
        }
    }
}

// Normative Vorbis I section 10.1 data, not decoder implementation code.
const INVERSE_DB: [f32; 256] = [
    1.0649863e-07,
    1.1341951e-07,
    1.2079015e-07,
    1.2863978e-07,
    1.3699951e-07,
    1.4590251e-07,
    1.5538408e-07,
    1.6548181e-07,
    1.7623575e-07,
    1.8768855e-07,
    1.9988561e-07,
    2.1287530e-07,
    2.2670913e-07,
    2.4144197e-07,
    2.5713223e-07,
    2.7384213e-07,
    2.9163793e-07,
    3.1059021e-07,
    3.3077411e-07,
    3.5226968e-07,
    3.7516214e-07,
    3.9954229e-07,
    4.2550680e-07,
    4.5315863e-07,
    4.8260743e-07,
    5.1396998e-07,
    5.4737065e-07,
    5.8294187e-07,
    6.2082472e-07,
    6.6116941e-07,
    7.0413592e-07,
    7.4989464e-07,
    7.9862701e-07,
    8.5052630e-07,
    9.0579828e-07,
    9.6466216e-07,
    1.0273513e-06,
    1.0941144e-06,
    1.1652161e-06,
    1.2409384e-06,
    1.3215816e-06,
    1.4074654e-06,
    1.4989305e-06,
    1.5963394e-06,
    1.7000785e-06,
    1.8105592e-06,
    1.9282195e-06,
    2.0535261e-06,
    2.1869758e-06,
    2.3290978e-06,
    2.4804557e-06,
    2.6416497e-06,
    2.8133190e-06,
    2.9961443e-06,
    3.1908506e-06,
    3.3982101e-06,
    3.6190449e-06,
    3.8542308e-06,
    4.1047004e-06,
    4.3714470e-06,
    4.6555282e-06,
    4.9580707e-06,
    5.2802740e-06,
    5.6234160e-06,
    5.9888572e-06,
    6.3780469e-06,
    6.7925283e-06,
    7.2339451e-06,
    7.7040476e-06,
    8.2047000e-06,
    8.7378876e-06,
    9.3057248e-06,
    9.9104632e-06,
    1.0554501e-05,
    1.1240392e-05,
    1.1970856e-05,
    1.2748789e-05,
    1.3577278e-05,
    1.4459606e-05,
    1.5399272e-05,
    1.6400004e-05,
    1.7465768e-05,
    1.8600792e-05,
    1.9809576e-05,
    2.1096914e-05,
    2.2467911e-05,
    2.3928002e-05,
    2.5482978e-05,
    2.7139006e-05,
    2.8902651e-05,
    3.0780908e-05,
    3.2781225e-05,
    3.4911534e-05,
    3.7180282e-05,
    3.9596466e-05,
    4.2169667e-05,
    4.4910090e-05,
    4.7828601e-05,
    5.0936773e-05,
    5.4246931e-05,
    5.7772202e-05,
    6.1526565e-05,
    6.5524908e-05,
    6.9783085e-05,
    7.4317983e-05,
    7.9147585e-05,
    8.4291040e-05,
    8.9768747e-05,
    9.5602426e-05,
    0.00010181521,
    0.00010843174,
    0.00011547824,
    0.00012298267,
    0.00013097477,
    0.00013948625,
    0.00014855085,
    0.00015820453,
    0.00016848555,
    0.00017943469,
    0.00019109536,
    0.00020351382,
    0.00021673929,
    0.00023082423,
    0.00024582449,
    0.00026179955,
    0.00027881276,
    0.00029693158,
    0.00031622787,
    0.00033677814,
    0.00035866388,
    0.00038197188,
    0.00040679456,
    0.00043323036,
    0.00046138411,
    0.00049136745,
    0.00052329927,
    0.00055730621,
    0.00059352311,
    0.00063209358,
    0.00067317058,
    0.00071691700,
    0.00076350630,
    0.00081312324,
    0.00086596457,
    0.00092223983,
    0.00098217216,
    0.0010459992,
    0.0011139742,
    0.0011863665,
    0.0012634633,
    0.0013455702,
    0.0014330129,
    0.0015261382,
    0.0016253153,
    0.0017309374,
    0.0018434235,
    0.0019632195,
    0.0020908006,
    0.0022266726,
    0.0023713743,
    0.0025254795,
    0.0026895994,
    0.0028643847,
    0.0030505286,
    0.0032487691,
    0.0034598925,
    0.0036847358,
    0.0039241906,
    0.0041792066,
    0.0044507950,
    0.0047400328,
    0.0050480668,
    0.0053761186,
    0.0057254891,
    0.0060975636,
    0.0064938176,
    0.0069158225,
    0.0073652516,
    0.0078438871,
    0.0083536271,
    0.0088964928,
    0.009474637,
    0.010090352,
    0.010746080,
    0.011444421,
    0.012188144,
    0.012980198,
    0.013823725,
    0.014722068,
    0.015678791,
    0.016697687,
    0.017782797,
    0.018938423,
    0.020169149,
    0.021479854,
    0.022875735,
    0.024362330,
    0.025945531,
    0.027631618,
    0.029427276,
    0.031339626,
    0.033376252,
    0.035545228,
    0.037855157,
    0.040315199,
    0.042935108,
    0.045725273,
    0.048696758,
    0.051861348,
    0.055231591,
    0.058820850,
    0.062643361,
    0.066714279,
    0.071049749,
    0.075666962,
    0.080584227,
    0.085821044,
    0.091398179,
    0.097337747,
    0.10366330,
    0.11039993,
    0.11757434,
    0.12521498,
    0.13335215,
    0.14201813,
    0.15124727,
    0.16107617,
    0.17154380,
    0.18269168,
    0.19456402,
    0.20720788,
    0.22067342,
    0.23501402,
    0.25028656,
    0.26655159,
    0.28387361,
    0.30232132,
    0.32196786,
    0.34289114,
    0.36517414,
    0.38890521,
    0.41417847,
    0.44109412,
    0.46975890,
    0.50028648,
    0.53279791,
    0.56742212,
    0.60429640,
    0.64356699,
    0.68538959,
    0.72993007,
    0.77736504,
    0.82788260,
    0.88168307,
    0.9389798,
    1.0,
];

#[cfg(test)]
mod tests {
    use super::super::setup::FloorClass;
    use super::*;

    #[test]
    fn integer_line_matches_independent_point_prediction() {
        for distance in 1..=65 {
            for (start, end) in [(0, 255), (255, 0), (127, 128), (203, 18), (31, 251)] {
                let mut output = vec![0.0; distance];
                render_line(0, start, distance, end, &mut output);
                for (x, &value) in output.iter().enumerate() {
                    let y = start + (end - start) * x as i32 / distance as i32;
                    assert_eq!(value, f64::from(INVERSE_DB[y as usize]));
                }
            }
        }
    }

    #[test]
    fn floor_curve_fills_flat_tail_and_ignores_inactive_points() {
        let floor = FloorOne {
            partitions: vec![],
            classes: vec![],
            multiplier: 1,
            x: vec![0, 4, 2],
            neighbors: vec![(0, 0), (0, 0), (0, 1)],
            sorted_points: vec![0, 2, 1],
        };
        let amplitudes = Amplitudes {
            values: vec![10, 14, 200],
            active: vec![true, true, false],
        };
        let mut output = [0.0; 8];
        floor.render_curve(&amplitudes, &mut output).unwrap();
        for (index, &value) in output.iter().enumerate() {
            assert_eq!(value, f64::from(INVERSE_DB[10 + index.min(4)]));
        }
        let mut corrupted = amplitudes;
        corrupted.values[0] = 256;
        assert!(floor.render_curve(&corrupted, &mut output).is_err());
    }

    fn bits(fields: &[(u32, usize)]) -> Vec<u8> {
        let mut bytes = vec![
            0;
            fields
                .iter()
                .map(|&(_, width)| width)
                .sum::<usize>()
                .div_ceil(8)
        ];
        let mut position = 0;
        for &(value, width) in fields {
            for bit in 0..width {
                bytes[position / 8] |= ((value >> bit) as u8 & 1) << (position & 7);
                position += 1;
            }
        }
        bytes
    }

    #[test]
    fn absent_truncated_and_flat_floors() {
        let floor = FloorOne {
            partitions: vec![],
            classes: vec![],
            multiplier: 1,
            x: vec![0, 64],
            neighbors: vec![(0, 0); 2],
            sorted_points: vec![0, 1],
        };
        for bytes in [&[][..], &[0], &[1], &[1, 2]] {
            assert!(
                floor
                    .decode_amplitudes(&mut PacketBits::new(bytes), &[])
                    .unwrap()
                    .is_none()
            );
        }
        let bytes = bits(&[(1, 1), (17, 8), (230, 8)]);
        let decoded = floor
            .decode_amplitudes(&mut PacketBits::new(&bytes), &[])
            .unwrap()
            .unwrap();
        assert_eq!(decoded.values, [17, 230]);
        assert_eq!(decoded.active, [true, true]);
    }

    #[test]
    fn zero_residual_keeps_prediction_but_skips_final_line_point() {
        let floor = FloorOne {
            partitions: vec![0],
            classes: vec![FloorClass {
                dimensions: 1,
                subclasses: 0,
                masterbook: None,
                books: vec![None],
            }],
            multiplier: 1,
            x: vec![0, 64, 32],
            neighbors: vec![(0, 0), (0, 0), (0, 1)],
            sorted_points: vec![0, 2, 1],
        };
        let bytes = bits(&[(1, 1), (100, 8), (201, 8)]);
        let decoded = floor
            .decode_amplitudes(&mut PacketBits::new(&bytes), &[])
            .unwrap()
            .unwrap();
        assert_eq!(decoded.values, [100, 201, 150]);
        assert_eq!(decoded.active, [true, true, false]);
        let bytes = bits(&[(1, 1), (201, 8), (100, 8)]);
        let decoded = floor
            .decode_amplitudes(&mut PacketBits::new(&bytes), &[])
            .unwrap()
            .unwrap();
        assert_eq!(decoded.values, [201, 100, 151]);
    }

    #[test]
    fn vector_entropy_is_not_needed_for_scalar_floor_decode() {
        // A floor's subclass books decode entry indices, never VQ vectors.
        let book = super::super::entropy::Codebook::parse(
            &mut PacketBits::new(&bits(&[
                (0x564342, 24),
                (1, 16),
                (2, 24),
                (0, 1),
                (0, 1),
                (0, 5),
                (0, 5),
                (0, 4),
            ])),
            &mut 2,
            &mut 0,
        )
        .unwrap();
        let floor = FloorOne {
            partitions: vec![0],
            classes: vec![FloorClass {
                dimensions: 1,
                subclasses: 0,
                masterbook: None,
                books: vec![Some(0)],
            }],
            multiplier: 1,
            x: vec![0, 64, 32],
            neighbors: vec![(0, 0), (0, 0), (0, 1)],
            sorted_points: vec![0, 2, 1],
        };
        let bytes = bits(&[(1, 1), (100, 8), (100, 8), (1, 1)]);
        let decoded = floor
            .decode_amplitudes(&mut PacketBits::new(&bytes), &[book])
            .unwrap()
            .unwrap();
        assert_eq!(decoded.values, [100, 100, 99]);
        assert_eq!(decoded.active, [true; 3]);
    }
}
