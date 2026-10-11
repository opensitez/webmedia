//! Intra segmentation prediction and per-segment tools, AV1 sections 5.11 and 7.12.

use super::entropy::SymbolDecoder;
use super::syntax::{Error, IntraFrameHeader};

pub(super) const DEFAULT_CDFS: [[u16; 9]; 3] = [
    [5622, 7893, 16093, 18233, 27809, 28373, 32533, 32768, 0],
    [14274, 18230, 22557, 24935, 29980, 30851, 32344, 32768, 0],
    [27527, 28487, 28723, 28890, 32397, 32647, 32679, 32768, 0],
];

pub(super) fn pre_skip(header: &IntraFrameHeader) -> bool {
    header.segmentation_enabled && header.segment_feature_enabled.iter()
        .any(|features| features[5..].iter().any(|&enabled| enabled))
}

pub(super) fn qindex(header: &IntraFrameHeader, segment: u8, current: u8) -> u8 {
    let segment = usize::from(segment);
    let delta = if header.segmentation_enabled && header.segment_feature_enabled[segment][0] {
        header.segment_features[segment][0]
    } else { 0 };
    (i32::from(current) + delta).clamp(0, 255) as u8
}

pub(super) fn lossless(header: &IntraFrameHeader, segment: u8) -> bool {
    qindex(header, segment, header.base_q_idx) == 0
        && header.quantizer_deltas.iter().all(|&delta| delta == 0)
}

pub(super) fn skip(header: &IntraFrameHeader, segment: u8) -> bool {
    header.segmentation_enabled && header.segment_feature_enabled[usize::from(segment)][6]
}

fn predicted(upper_left: Option<u8>, upper: Option<u8>, left: Option<u8>) -> (u8, usize) {
    let prediction = match (upper, left) {
        (None, None) => 0,
        (None, Some(left)) => left,
        (Some(upper), None) => upper,
        (Some(upper), Some(left)) => if upper_left == Some(upper) { upper } else { left },
    };
    let context = if upper_left.is_none() { 0 }
        else if upper_left == upper && upper_left == left { 2 }
        else if upper_left == upper || upper_left == left || upper == left { 1 }
        else { 0 };
    (prediction, context)
}

fn deinterleave(diff: usize, prediction: usize, count: usize) -> Result<u8, Error> {
    if diff >= count || prediction >= count {
        return Err(Error::Invalid("segment id range"));
    }
    let value = if prediction == 0 { diff }
        else if prediction >= count - 1 { count - diff - 1 }
        else if diff <= 2 * prediction.min(count - prediction - 1) {
            if diff & 1 != 0 { prediction + diff.div_ceil(2) } else { prediction - diff / 2 }
        } else if 2 * prediction < count { diff }
        else { count - diff - 1 };
    Ok(value as u8)
}

pub(super) fn read(
    decoder: &mut SymbolDecoder<'_>, cdfs: &mut [[u16; 9]; 3],
    header: &IntraFrameHeader, neighbors: [Option<u8>; 3], skipped: bool,
) -> Result<u8, Error> {
    let (prediction, context) = predicted(neighbors[0], neighbors[1], neighbors[2]);
    if skipped { return Ok(prediction); }
    let count = header.segment_feature_enabled.iter().rposition(|features|
        features.iter().any(|&enabled| enabled)).map_or(1, |last| last + 1);
    let diff = decoder.read_symbol(&mut cdfs[context])?;
    deinterleave(diff, usize::from(prediction), count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_deinterleaving_is_a_permutation_for_every_active_range() {
        for count in 1..=8 {
            for prediction in 0..count {
                let mut values: Vec<_> = (0..count).map(|diff|
                    deinterleave(diff, prediction, count).unwrap()).collect();
                assert_eq!(values[0], prediction as u8);
                values.sort_unstable();
                assert_eq!(values, (0..count as u8).collect::<Vec<_>>());
            }
        }
        assert!(deinterleave(8, 0, 8).is_err());
    }

    #[test]
    fn prediction_respects_tile_boundaries_and_neighbor_agreement() {
        assert_eq!(predicted(None, None, None), (0, 0));
        assert_eq!(predicted(None, Some(4), None), (4, 0));
        assert_eq!(predicted(None, None, Some(3)), (3, 0));
        assert_eq!(predicted(Some(4), Some(4), Some(4)), (4, 2));
        assert_eq!(predicted(Some(4), Some(4), Some(3)), (4, 1));
        assert_eq!(predicted(Some(2), Some(4), Some(3)), (3, 0));
    }
}
