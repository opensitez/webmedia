//! PNS energy reconstruction, ISO/IEC 14496-3:2005 section 4.6.13.3.

use super::invalid;
use crate::video::backend::MediaDecodeError;

pub(super) struct Noise {
    state: u32,
}

impl Noise {
    pub(super) fn new() -> Self {
        Self { state: 1 }
    }

    pub(super) fn fill(&mut self, output: &mut [f64], energy: i16) -> Result<(), MediaDecodeError> {
        if output.is_empty() {
            return Err(invalid("empty AAC noise band"));
        }
        let target = 2.0_f64.powf(f64::from(energy) * 0.25);
        if !target.is_finite() || target == 0.0 {
            return Err(invalid("invalid AAC noise energy"));
        }
        let mut power = 0.0;
        for value in output.iter_mut() {
            // Choice of random generator is not standardized; use a full-period LCG.
            self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
            *value = f64::from(self.state as i32) / 2147483648.0;
            power += *value * *value;
        }
        if power == 0.0 {
            return Err(invalid("zero AAC noise vector"));
        }
        let scale = target / power.sqrt();
        for value in output {
            *value *= scale;
        }
        Ok(())
    }

    pub(super) fn correlated(
        left: &[f64],
        right: &mut [f64],
        left_energy: i16,
        right_energy: i16,
    ) -> Result<(), MediaDecodeError> {
        let ratio = 2.0_f64.powf((f64::from(right_energy) - f64::from(left_energy)) * 0.25);
        if left.len() != right.len()
            || !ratio.is_finite()
            || left.iter().any(|value| !(value * ratio).is_finite())
        {
            return Err(invalid("invalid correlated AAC noise"));
        }
        for (right, left) in right.iter_mut().zip(left) {
            *right = left * ratio;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_power_matches_signaled_energy_for_all_band_lengths() {
        let mut noise = Noise::new();
        for width in [4, 8, 12, 32, 64, 128] {
            for energy in [-100, 0, 40, 100] {
                let mut samples = vec![0.0; width];
                noise.fill(&mut samples, energy).unwrap();
                let power: f64 = samples.iter().map(|s| s * s).sum();
                let target = 2.0_f64.powf(f64::from(energy) * 0.5);
                assert!((power / target - 1.0).abs() < 1e-14);
                assert!(samples.iter().any(|s| *s != 0.0));
            }
        }
    }

    #[test]
    fn correlated_bands_reuse_vector_with_independent_energy() {
        let mut noise = Noise::new();
        let mut left = [0.0; 32];
        let mut right = [0.0; 32];
        noise.fill(&mut left, 20).unwrap();
        Noise::correlated(&left, &mut right, 20, 24).unwrap();
        for (l, r) in left.iter().zip(right) {
            assert_eq!(r, l * 2.0);
        }
    }
}
