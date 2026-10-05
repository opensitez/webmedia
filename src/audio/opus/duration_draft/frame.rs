//! Test-only connection of duration-specific energy/allocation to our frame prefix.
pub(super) use super::super::frame::FrameStart;
use super::super::frame::ShapeStart;
use super::{MediaDecodeError, draft_allocation, energy, tf};

pub(super) trait DraftShapeStart {
    fn start_shapes_draft(
        &mut self,
        previous: &[f64],
    ) -> Result<(ShapeStart, [u64; 5]), MediaDecodeError>;
}

impl DraftShapeStart for FrameStart<'_> {
    fn start_shapes_draft(
        &mut self,
        previous: &[f64],
    ) -> Result<(ShapeStart, [u64; 5]), MediaDecodeError> {
        let mut cursor = self.entropy.clone();
        let mut positions = [cursor.tell_fractional(); 5];
        let coarse = super::coarse(
            &mut cursor,
            self.layout,
            self.channels,
            self.intra,
            previous,
        )?;
        positions[1] = cursor.tell_fractional();
        let tf = tf::TfResolution::decode(&mut cursor, self.layout, self.transient)?;
        positions[2] = cursor.tell_fractional();
        let spread = if cursor.tell() + 4 <= cursor.frame_bytes() as u64 * 8 {
            cursor.inverse_cdf(&[25, 23, 2, 0], 5)? as u8
        } else {
            2
        };
        let preparation = draft_allocation::AllocationPreparation::decode_fixed(
            &mut cursor,
            self.layout,
            self.channels,
            self.transient,
        )?;
        let allocation = preparation.finish(&mut cursor, self.layout, self.channels)?;
        positions[3] = cursor.tell_fractional();
        let energy = energy::FineEnergy::decode(
            &mut cursor,
            self.layout,
            self.channels,
            &coarse,
            &allocation.fine_bits,
        )?;
        positions[4] = cursor.tell_fractional();
        self.entropy = cursor;
        Ok((
            ShapeStart {
                energy,
                tf,
                spread,
                allocation,
                anti_collapse_reserved: preparation.anti_collapse_reserved_eighths != 0,
            },
            positions,
        ))
    }
}
