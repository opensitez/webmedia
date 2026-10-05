//! Scale-factor boundaries from ISO/IEC 13818-7:2004 tables 45-57.

use super::tables::*;
use crate::video::backend::MediaDecodeError;
use std::sync::OnceLock;

/// Boundaries for a 1024-sample frame (128 samples per short window).
pub(crate) fn bands(index: u8, short: bool) -> Result<&'static [u16], MediaDecodeError> {
    Ok(match (index, short) {
        (0 | 1, false) => BANDS_56,
        (2, false) => BANDS_54,
        (3 | 4, false) => BANDS_45,
        (5, false) => BANDS_47,
        (6 | 7, false) => BANDS_52,
        (8..=10, false) => BANDS_50,
        (11, false) => BANDS_48,
        (0 | 1, true) => BANDS_57,
        (2, true) => BANDS_55,
        (3..=5, true) => BANDS_46,
        (6 | 7, true) => BANDS_53,
        (8..=10, true) => BANDS_51,
        (11, true) => BANDS_49,
        _ => return Err(MediaDecodeError::Unsupported),
    })
}

pub(crate) fn frame_bands(
    index: u8,
    short: bool,
    samples: usize,
) -> Result<&'static [u16], MediaDecodeError> {
    let original = bands(index, short)?;
    match samples {
        1024 => Ok(original),
        960 => {
            // ISO/IEC 14496-3:2001 tables 4.73-4.85 give the shorter
            // terminal boundary in brackets; bands beyond it disappear.
            static SHORTER: OnceLock<[[Vec<u16>; 2]; 12]> = OnceLock::new();
            let tables = SHORTER.get_or_init(|| {
                std::array::from_fn(|rate| {
                    std::array::from_fn(|short| {
                        let limit = if short == 0 { 960 } else { 120 };
                        let mut offsets: Vec<_> = bands(rate as u8, short != 0)
                            .unwrap()
                            .iter()
                            .copied()
                            .take_while(|&offset| offset < limit)
                            .collect();
                        offsets.push(limit);
                        offsets
                    })
                })
            });
            Ok(&tables[usize::from(index)][usize::from(short)])
        }
        _ => Err(MediaDecodeError::Unsupported),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_verified_rate_boundaries() {
        let long_counts = [41, 41, 47, 49, 49, 51, 47, 47, 43, 43, 43, 40];
        let short_counts = [12, 12, 12, 14, 14, 14, 15, 15, 15, 15, 15, 15];
        for index in 0..12 {
            for short in [false, true] {
                let offsets = bands(index, short).unwrap();
                assert_eq!(
                    offsets.len() - 1,
                    if short {
                        short_counts[index as usize]
                    } else {
                        long_counts[index as usize]
                    }
                );
                assert_eq!(offsets[0], 0);
                assert_eq!(*offsets.last().unwrap(), if short { 128 } else { 1024 });
                assert!(offsets.windows(2).all(|w| w[0] < w[1]));
                assert!(offsets.iter().all(|x| x % 4 == 0));
            }
        }
        assert!(bands(12, false).is_err());
        assert!(bands(15, true).is_err());
    }

    #[test]
    fn shorter_boundaries_use_normative_terminal_values() {
        let long_counts = [40, 40, 46, 49, 49, 49, 46, 46, 42, 42, 42, 40];
        for index in 0..12 {
            for short in [false, true] {
                let original = bands(index, short).unwrap();
                let shorter = frame_bands(index, short, 960).unwrap();
                assert_eq!(
                    shorter.len() - 1,
                    if short {
                        original.len() - 1
                    } else {
                        long_counts[index as usize]
                    }
                );
                assert_eq!(*shorter.last().unwrap(), if short { 120 } else { 960 });
                assert_eq!(
                    &shorter[..shorter.len() - 1],
                    &original[..shorter.len() - 1]
                );
                assert!(shorter.windows(2).all(|pair| pair[0] < pair[1]));
                assert!(shorter.iter().all(|offset| offset % 4 == 0));
            }
        }
        assert!(frame_bands(12, false, 960).is_err());
        assert!(frame_bands(3, false, 480).is_err());
    }
}
