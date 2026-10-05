//! Inverse MDCT via a zero-padded Fourier transform, without codec dependencies.

use crate::video::backend::MediaDecodeError;
use std::f64::consts::TAU;

#[cfg(test)]
std::thread_local! {
    static REDUCED_MDCT_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn with_reduced_mdct<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            REDUCED_MDCT_ENABLED.with(|enabled| enabled.set(self.0));
        }
    }
    let _restore = Restore(REDUCED_MDCT_ENABLED.with(|enabled| enabled.replace(true)));
    run()
}

pub struct MdctPlan {
    size: usize,
    roots: Vec<(f64, f64)>,
    permutation: Vec<usize>,
    scratch: Vec<(f64, f64)>,
    mixed: Option<MixedRadixPlan>,
    #[cfg(test)]
    reduced: Option<ReducedMdctPlan>,
}

impl MdctPlan {
    pub fn new(size: usize) -> Result<Self, MediaDecodeError> {
        if !(64..=8192).contains(&size)
            || (!size.is_power_of_two() && ![240, 480, 960, 1920].contains(&size))
        {
            return Err(MediaDecodeError::InvalidData(
                "invalid audio MDCT size".into(),
            ));
        }
        let length = size * 4;
        if !size.is_power_of_two() {
            return Ok(Self {
                size,
                roots: Vec::new(),
                permutation: Vec::new(),
                scratch: vec![(0.0, 0.0); length],
                mixed: Some(MixedRadixPlan::new(length)),
                #[cfg(test)]
                reduced: None,
            });
        }
        let roots = (0..length / 2)
            .map(|index| {
                let (sin, cos) = (-TAU * index as f64 / length as f64).sin_cos();
                (cos, sin)
            })
            .collect();
        let shift = usize::BITS - length.trailing_zeros();
        let permutation = (0..length)
            .map(|index| index.reverse_bits() >> shift)
            .collect();
        Ok(Self {
            size,
            roots,
            permutation,
            scratch: vec![(0.0, 0.0); length],
            mixed: None,
            #[cfg(test)]
            reduced: None,
        })
    }

    pub fn inverse(
        &mut self,
        spectrum: &[f64],
        output: &mut [f64],
    ) -> Result<(), MediaDecodeError> {
        if spectrum.len() != self.size / 2
            || output.len() != self.size
            || spectrum.iter().any(|value| !value.is_finite())
        {
            return Err(MediaDecodeError::InvalidData(
                "invalid audio MDCT input".into(),
            ));
        }
        #[cfg(test)]
        if REDUCED_MDCT_ENABLED.with(|enabled| enabled.get()) {
            if self.reduced.is_none() {
                self.reduced = Some(ReducedMdctPlan::new(self.size)?);
            }
            return self.reduced.as_mut().unwrap().inverse(spectrum, output);
        }
        self.scratch.fill((0.0, 0.0));
        if let Some(mixed) = &mut self.mixed {
            for (index, &value) in spectrum.iter().enumerate() {
                self.scratch[2 * index + 1].0 = value;
            }
            let transformed = mixed.forward(&self.scratch);
            for (index, value) in output.iter_mut().enumerate() {
                *value = transformed[2 * index + 1 + self.size / 2].0;
            }
            return Ok(());
        }
        for (index, &value) in spectrum.iter().enumerate() {
            self.scratch[self.permutation[2 * index + 1]].0 = value;
        }
        let length = self.scratch.len();
        let mut width = 2;
        while width <= length {
            let half = width / 2;
            let root_step = length / width;
            for start in (0..length).step_by(width) {
                for offset in 0..half {
                    let (real, imag) = self.scratch[start + half + offset];
                    let (cos, sin) = self.roots[offset * root_step];
                    let rotated = (real * cos - imag * sin, real * sin + imag * cos);
                    let (real, imag) = self.scratch[start + offset];
                    self.scratch[start + offset] = (real + rotated.0, imag + rotated.1);
                    self.scratch[start + half + offset] = (real - rotated.0, imag - rotated.1);
                }
            }
            width *= 2;
        }
        for (index, value) in output.iter_mut().enumerate() {
            *value = self.scratch[2 * index + 1 + self.size / 2].0;
        }
        Ok(())
    }
}

struct FourierStage {
    length: usize,
    radix: usize,
    roots: Vec<(f64, f64)>,
}

struct MixedRadixPlan {
    stages: Vec<FourierStage>,
    output: Vec<(f64, f64)>,
    scratch: Vec<(f64, f64)>,
}

impl MixedRadixPlan {
    fn new(length: usize) -> Self {
        let mut stages = Vec::new();
        let mut remaining = length;
        while remaining > 1 {
            let radix = if remaining % 2 == 0 {
                2
            } else if remaining % 3 == 0 {
                3
            } else {
                5
            };
            debug_assert_eq!(remaining % radix, 0);
            let roots = (0..remaining)
                .map(|index| {
                    let (sine, cosine) = (-TAU * index as f64 / remaining as f64).sin_cos();
                    (cosine, sine)
                })
                .collect();
            stages.push(FourierStage {
                length: remaining,
                radix,
                roots,
            });
            remaining /= radix;
        }
        Self {
            stages,
            output: vec![(0.0, 0.0); length],
            scratch: vec![(0.0, 0.0); length],
        }
    }

    fn forward(&mut self, input: &[(f64, f64)]) -> &[(f64, f64)] {
        fourier(
            &self.stages,
            input,
            0,
            1,
            &mut self.output,
            &mut self.scratch,
        );
        &self.output
    }
}

fn fourier(
    stages: &[FourierStage],
    input: &[(f64, f64)],
    start: usize,
    stride: usize,
    output: &mut [(f64, f64)],
    scratch: &mut [(f64, f64)],
) {
    let Some((stage, next)) = stages.split_first() else {
        output[0] = input[start];
        return;
    };
    let width = stage.length / stage.radix;
    for (branch, (output, scratch)) in output
        .chunks_exact_mut(width)
        .zip(scratch.chunks_exact_mut(width))
        .enumerate()
    {
        fourier(
            next,
            input,
            start + branch * stride,
            stride * stage.radix,
            output,
            scratch,
        );
    }
    // Child transforms are contiguous; combine into reusable scratch before
    // replacing them, so no packet-time allocation or trigonometry is needed.
    for (index, value) in scratch.iter_mut().enumerate() {
        let bin = index % width;
        let mut sum = output[bin];
        for branch in 1..stage.radix {
            let (real, imaginary) = output[branch * width + bin];
            let (cosine, sine) = stage.roots[branch * index % stage.length];
            sum.0 += real * cosine - imaginary * sine;
            sum.1 += real * sine + imaginary * cosine;
        }
        *value = sum;
    }
    output.copy_from_slice(scratch);
}

// Experimental factorization of the normative cosine sum; production keeps
// the original zero-padded transform until codec-level numerical gates pass.
#[cfg(test)]
pub(crate) struct ReducedMdctPlan {
    size: usize,
    pre: Vec<(f64, f64)>,
    post: Vec<(f64, f64)>,
    input: Vec<(f64, f64)>,
    fourier: MixedRadixPlan,
}

#[cfg(test)]
impl ReducedMdctPlan {
    pub(crate) fn new(size: usize) -> Result<Self, MediaDecodeError> {
        if !(64..=8192).contains(&size)
            || (!size.is_power_of_two() && ![240, 480, 960, 1920].contains(&size))
        {
            return Err(MediaDecodeError::InvalidData(
                "invalid audio MDCT size".into(),
            ));
        }
        let phase = |position: f64| {
            let (sin, cos) = (-std::f64::consts::PI * position / size as f64).sin_cos();
            (cos, sin)
        };
        Ok(Self {
            size,
            pre: (0..size / 2).map(|k| phase(k as f64)).collect(),
            post: (0..size / 2).map(|n| phase(n as f64 + 0.5)).collect(),
            input: vec![(0.0, 0.0); size],
            fourier: MixedRadixPlan::new(size),
        })
    }

    pub(crate) fn inverse(
        &mut self,
        spectrum: &[f64],
        output: &mut [f64],
    ) -> Result<(), MediaDecodeError> {
        if spectrum.len() != self.size / 2
            || output.len() != self.size
            || spectrum.iter().any(|value| !value.is_finite())
        {
            return Err(MediaDecodeError::InvalidData(
                "invalid audio MDCT input".into(),
            ));
        }
        self.input[self.size / 2..].fill((0.0, 0.0));
        for ((target, &value), &(cos, sin)) in self.input.iter_mut().zip(spectrum).zip(&self.pre) {
            *target = (value * cos, value * sin);
        }
        let transformed = self.fourier.forward(&self.input);
        let half = self.size / 2;
        for (time, value) in output.iter_mut().enumerate() {
            let r = time + half / 2;
            let (index, sign) = if r < half {
                (r, 1.0)
            } else if r < 2 * half {
                (2 * half - 1 - r, -1.0)
            } else {
                (r - 2 * half, -1.0)
            };
            let (real, imag) = transformed[index];
            let (cos, sin) = self.post[index];
            let result = sign * (real * cos - imag * sin);
            *value = if result == 0.0 { 0.0 } else { result };
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn supported_sizes() -> impl Iterator<Item = usize> {
        (6..=13)
            .map(|shift| 1 << shift)
            .chain([240, 480, 960, 1920])
    }

    #[test]
    fn reduced_dispatch_is_scoped_cached_and_panic_safe() {
        let mut plan = MdctPlan::new(64).unwrap();
        let spectrum = [0.25; 32];
        let mut output = [0.0; 64];
        plan.inverse(&spectrum, &mut output).unwrap();
        let retained = output;
        assert!(plan.reduced.is_none());
        with_reduced_mdct(|| {
            plan.inverse(&spectrum, &mut output).unwrap();
            let cached = plan.reduced.as_ref().unwrap().input.as_ptr();
            with_reduced_mdct(|| plan.inverse(&spectrum, &mut output).unwrap());
            assert_eq!(cached, plan.reduced.as_ref().unwrap().input.as_ptr());
            assert!(REDUCED_MDCT_ENABLED.with(|enabled| enabled.get()));
            std::thread::spawn(|| {
                assert!(!REDUCED_MDCT_ENABLED.with(|enabled| enabled.get()));
            })
            .join()
            .unwrap();
        });
        assert!(!REDUCED_MDCT_ENABLED.with(|enabled| enabled.get()));
        let panic = std::panic::catch_unwind(|| with_reduced_mdct(|| panic!("scope test")));
        assert!(panic.is_err());
        assert!(!REDUCED_MDCT_ENABLED.with(|enabled| enabled.get()));
        plan.inverse(&spectrum, &mut output).unwrap();
        assert!(
            output
                .iter()
                .zip(retained)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    #[test]
    fn reduced_mdct_matches_normative_sum_and_retained_fft() {
        for size in supported_sizes() {
            let mut reduced = ReducedMdctPlan::new(size).unwrap();
            let mut retained = MdctPlan::new(size).unwrap();
            let mut actual = vec![0.0; size];
            let mut reference = vec![0.0; size];
            for pattern in 0..6 {
                let spectrum: Vec<_> = (0..size / 2)
                    .map(|k| match pattern {
                        0 => f64::from(u8::from(k == 0)),
                        1 => f64::from(u8::from(k == size / 2 - 1)),
                        2 => {
                            if k % 2 == 0 {
                                0.5
                            } else {
                                -0.5
                            }
                        }
                        3 => ((k * 137 + 71) % 100) as f64 / 100.0 - 0.5,
                        4 => {
                            if k % 37 == 0 {
                                0.25
                            } else {
                                0.0
                            }
                        }
                        _ => 0.0,
                    })
                    .collect();
                reduced.inverse(&spectrum, &mut actual).unwrap();
                retained.inverse(&spectrum, &mut reference).unwrap();
                for (time, (&value, &baseline)) in actual.iter().zip(&reference).enumerate() {
                    assert!(
                        (value - baseline).abs() < 1e-10,
                        "retained: size={size} pattern={pattern} time={time}: {value} != {baseline}"
                    );
                    // All samples for small sizes; bounded stratified samples
                    // for large sizes, including both reflection boundaries.
                    if size <= 256
                        || time % (size / 16) == 0
                        || [
                            size / 4 - 1,
                            size / 4,
                            size * 3 / 4 - 1,
                            size * 3 / 4,
                            size - 1,
                        ]
                        .contains(&time)
                    {
                        let expected: f64 = spectrum
                            .iter()
                            .enumerate()
                            .map(|(k, &x)| {
                                // The normative phase is pi/(2N) times this
                                // integer product. Reduce exactly before converting
                                // to floating point, rather than rounding a large
                                // angle and relying on libm argument reduction.
                                let phase = ((2 * time + 1 + size / 2) * (2 * k + 1)) % (4 * size);
                                x * (std::f64::consts::PI * phase as f64 / (2 * size) as f64).cos()
                            })
                            .sum();
                        assert!(
                            (value - expected).abs() < 1e-10,
                            "cosine: size={size} pattern={pattern} time={time}: {value} != {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn reduced_mdct_validation_zero_and_overflow_contract() {
        for size in supported_sizes() {
            let mut reduced = ReducedMdctPlan::new(size).unwrap();
            let mut retained = MdctPlan::new(size).unwrap();
            let mut actual = vec![7.0; size];
            let mut reference = actual.clone();
            for zero in [0.0, -0.0] {
                let input = vec![zero; size / 2];
                reduced.inverse(&input, &mut actual).unwrap();
                retained.inverse(&input, &mut reference).unwrap();
                assert!(
                    actual
                        .iter()
                        .zip(&reference)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
            for bad in [
                vec![0.0; size / 2 - 1],
                vec![f64::NAN; size / 2],
                vec![f64::INFINITY; size / 2],
                vec![f64::NEG_INFINITY; size / 2],
            ] {
                actual.fill(-0.0);
                reference.copy_from_slice(&actual);
                let a = reduced.inverse(&bad, &mut actual);
                let b = retained.inverse(&bad, &mut reference);
                assert_eq!(format!("{a:?}"), format!("{b:?}"));
                assert!(actual.iter().all(|x| x.to_bits() == (-0.0f64).to_bits()));
            }
            let input = vec![0.0; size / 2];
            let mut wrong = vec![1.25; size - 1];
            assert!(reduced.inverse(&input, &mut wrong).is_err());
            assert!(wrong.iter().all(|&x| x == 1.25));
            // Existing API accepts finite inputs even when arithmetic
            // overflows. Do not add a candidate-only rejection or clamp.
            let huge = vec![f64::MAX; size / 2];
            assert!(reduced.inverse(&huge, &mut actual).is_ok());
            assert!(retained.inverse(&huge, &mut reference).is_ok());
            assert!(actual.iter().any(|x| !x.is_finite()));
            assert!(reference.iter().any(|x| !x.is_finite()));
            reduced.inverse(&input, &mut actual).unwrap();
            assert!(actual.iter().all(|&x| x == 0.0));
        }
        for size in [0, 32, 65, 239, 241, 16384, usize::MAX] {
            assert_eq!(
                format!("{:?}", ReducedMdctPlan::new(size).err()),
                format!("{:?}", MdctPlan::new(size).err())
            );
        }
    }

    #[test]
    fn fourier_mdct_matches_independent_cosine_sum() {
        for size in [64, 128, 256, 240, 480, 960, 1920] {
            let spectrum: Vec<_> = (0..size / 2)
                .map(|i| ((i * 137 + 71) % 100) as f64 / 100.0 - 0.5)
                .collect();
            let mut plan = MdctPlan::new(size).unwrap();
            let mut output = vec![0.0; size];
            for _ in 0..2 {
                plan.inverse(&spectrum, &mut output).unwrap();
                for (time, &value) in output.iter().enumerate() {
                    let expected: f64 = spectrum
                        .iter()
                        .enumerate()
                        .map(|(frequency, &value)| {
                            value
                                * (TAU / size as f64
                                    * (time as f64 + 0.5 + size as f64 / 4.0)
                                    * (frequency as f64 + 0.5))
                                    .cos()
                        })
                        .sum();
                    assert!(
                        (expected - value).abs() < 1e-10,
                        "{size}/{time}: {expected} != {value}"
                    );
                }
            }
        }
    }

    #[test]
    fn zero_input_and_invalid_shapes_are_bounded() {
        let mut plan = MdctPlan::new(64).unwrap();
        let mut output = [1.0; 64];
        plan.inverse(&[0.0; 32], &mut output).unwrap();
        assert_eq!(output, [0.0; 64]);
        assert!(plan.inverse(&[0.0; 31], &mut output).is_err());
        assert!(plan.inverse(&[f64::NAN; 32], &mut output).is_err());
        for size in [0, 32, 65, 16384, usize::MAX] {
            assert!(MdctPlan::new(size).is_err());
        }
    }

    #[test]
    fn mixed_radix_fourier_matches_complex_definition() {
        for length in [6, 10, 15, 30, 60, 120] {
            let input: Vec<_> = (0..length)
                .map(|index| ((index * 37 % 11) as f64, (index * 13 % 7) as f64))
                .collect();
            let mut plan = MixedRadixPlan::new(length);
            let output = plan.forward(&input);
            for (bin, &actual) in output.iter().enumerate() {
                let expected =
                    input
                        .iter()
                        .enumerate()
                        .fold((0.0, 0.0), |sum, (time, &(real, imaginary))| {
                            let (sine, cosine) =
                                (-TAU * (time * bin) as f64 / length as f64).sin_cos();
                            (
                                sum.0 + real * cosine - imaginary * sine,
                                sum.1 + real * sine + imaginary * cosine,
                            )
                        });
                assert!((actual.0 - expected.0).abs() < 1e-10);
                assert!((actual.1 - expected.1).abs() < 1e-10);
            }
            let zeros = vec![(0.0, 0.0); length];
            assert!(
                plan.forward(&zeros)
                    .iter()
                    .all(|&value| value == (0.0, 0.0))
            );
        }
    }
}
