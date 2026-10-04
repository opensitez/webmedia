//! CELT frame prefix in RFC 6716 Table 56 and sections 4.3.1/4.3.7.1.
//! Parsing stops at coarse energy; it never emits PCM for an undecoded frame.

use super::{Packet, celt::Layout, range::RangeDecoder, synthesis::PitchParameters};
use crate::video::backend::MediaDecodeError;

pub struct FrameStart<'a> {
    pub layout: Layout,
    pub channels: usize,
    pub transient: bool,
    pub intra: bool,
    pub pitch: Option<PitchParameters>,
    /// Positioned at coarse energy, not at TF, fine energy or residual shape.
    pub entropy: RangeDecoder<'a>,
}

pub struct ShapeStart {
    pub energy: super::energy::FineEnergy,
    pub tf: super::tf::TfResolution,
    pub spread: u8,
    pub allocation: super::allocation::BandAllocation,
    pub anti_collapse_reserved: bool,
}

impl<'a> FrameStart<'a> {
    /// Advance actual fixed-mode packet syntax through fine energy to PVQ.
    /// Previous energies must be final refined history, not coarse estimates.
    pub fn start_shapes(&mut self, previous: &[f64]) -> Result<ShapeStart, MediaDecodeError> {
        let mut cursor = self.entropy.clone();
        let coarse = super::energy::decode_coarse_20ms(
            &mut cursor,
            self.layout,
            self.channels,
            self.intra,
            previous,
        )?;
        let tf = super::tf::TfResolution::decode(&mut cursor, self.layout, self.transient)?;
        let spread = if cursor.tell() + 4 <= cursor.frame_bytes() as u64 * 8 {
            cursor.inverse_cdf(&[25, 23, 2, 0], 5)? as u8
        } else {
            2
        };
        let preparation = super::allocation::AllocationPreparation::decode_fixed(
            &mut cursor,
            self.layout,
            self.channels,
            self.transient,
        )?;
        let allocation = preparation.finish(&mut cursor, self.layout, self.channels)?;
        let energy = super::energy::FineEnergy::decode(
            &mut cursor,
            self.layout,
            self.channels,
            &coarse,
            &allocation.fine_bits,
        )?;
        self.entropy = cursor;
        Ok(ShapeStart {
            energy,
            tf,
            spread,
            allocation,
            anti_collapse_reserved: preparation.anti_collapse_reserved_eighths != 0,
        })
    }

    pub fn decode_coarse(&mut self, previous: &[f64]) -> Result<Vec<f64>, MediaDecodeError> {
        super::energy::decode_coarse_20ms(
            &mut self.entropy,
            self.layout,
            self.channels,
            self.intra,
            previous,
        )
    }

    /// Decode one sufficiently provisioned, non-silent CELT-only frame. The
    /// 32-byte restriction excludes small-budget syntax gates not specified
    /// precisely by the prose; it is not a claim that smaller frames are bad.
    pub fn parse(packet: &Packet<'a>, frame_index: usize) -> Result<Self, MediaDecodeError> {
        let frame = packet
            .frames
            .get(frame_index)
            .ok_or_else(|| super::invalid("invalid CELT frame index"))?;
        let layout = packet.celt_layout()?.ok_or(MediaDecodeError::Unsupported)?;
        if layout.bands().start != 0 || frame.len() < 32 {
            return Err(MediaDecodeError::Unsupported);
        }
        let mut entropy = RangeDecoder::new(frame);
        if entropy.bit(15)? {
            // Silence has its own constrained packet-to-PCM path. Do not run
            // ordinary syntax through its intentionally absent coding fields.
            return Err(MediaDecodeError::Unsupported);
        }
        let pitch = PitchParameters::decode(&mut entropy)?;
        let transient = layout.samples() > 120 && entropy.bit(3)?;
        let intra = entropy.bit(3)?;
        Ok(Self {
            layout,
            channels: if packet.stereo { 2 } else { 1 },
            transient,
            intra,
            pitch,
            entropy,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_matches_independent_table_56_probability_intervals() {
        for config in 16..=31 {
            for first in 0..=254u8 {
                let mut data = [0x96; 65];
                data[0] = (config << 3) | 4;
                data[1] = first;
                let packet = Packet::parse(&data).unwrap();
                let parsed = FrameStart::parse(&packet, 0).unwrap();
                let mut expected = RangeDecoder::new(packet.frames[0]);
                assert_eq!(expected.symbol(&[32767, 1]).unwrap(), 0);
                let pitch = PitchParameters::decode(&mut expected).unwrap();
                let transient = config & 3 != 0 && expected.symbol(&[7, 1]).unwrap() != 0;
                let intra = expected.symbol(&[7, 1]).unwrap() != 0;
                assert_eq!(
                    (
                        parsed.channels,
                        parsed.transient,
                        parsed.intra,
                        parsed.pitch
                    ),
                    (2, transient, intra, pitch)
                );
                assert_eq!(parsed.entropy.tell_fractional(), expected.tell_fractional());
            }
        }
        let data = [0xfc, 0xff, 0xfe];
        assert!(matches!(
            FrameStart::parse(&Packet::parse(&data).unwrap(), 0),
            Err(MediaDecodeError::Unsupported)
        ));
    }
}
