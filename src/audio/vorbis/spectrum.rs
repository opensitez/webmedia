//! Vorbis packet reconstruction through the frequency-domain spectrum.

use super::entropy::PacketBits;
use super::floor::Amplitudes;
use super::floor_zero::Lsp;
use super::setup::{Floor, PacketHeader, Setup};
use crate::video::backend::MediaDecodeError;

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.into())
}

#[derive(Debug)]
pub struct Spectrum {
    pub header: PacketHeader,
    pub channels: Vec<Vec<f64>>,
}

#[derive(Default)]
struct DecodedFloor {
    present: bool,
    one: Amplitudes,
    zero: Lsp,
}

#[derive(Default)]
pub(super) struct Workspace {
    pub channels: Vec<Vec<f64>>,
    floors: Vec<DecodedFloor>,
    floor_values: Vec<i64>,
    floor_vector: Vec<f64>,
    floor_cosines: Vec<f64>,
    enabled: Vec<bool>,
    indices: Vec<usize>,
    flags: Vec<bool>,
    curve: Vec<f64>,
    residue: super::residue::Workspace,
}

impl Setup {
    pub fn spectrum(
        &self,
        packet: &[u8],
        block_sizes: [usize; 2],
    ) -> Result<Option<Spectrum>, MediaDecodeError> {
        let mut workspace = Workspace::default();
        Ok(self
            .spectrum_with_workspace(packet, block_sizes, &mut workspace)?
            .map(|header| Spectrum {
                header,
                channels: workspace.channels,
            }))
    }

    pub(super) fn spectrum_with_workspace(
        &self,
        packet: &[u8],
        block_sizes: [usize; 2],
        workspace: &mut Workspace,
    ) -> Result<Option<PacketHeader>, MediaDecodeError> {
        if block_sizes
            .iter()
            .any(|&size| !(64..=8192).contains(&size) || !size.is_power_of_two())
            || block_sizes[0] > block_sizes[1]
        {
            return Err(invalid("invalid Vorbis block sizes"));
        }
        let mut bits = PacketBits::new(packet);
        let Some(header) = self.packet_header(&mut bits)? else {
            return Ok(None);
        };
        let mode = &self.modes[header.mode];
        let mapping = self
            .mappings
            .get(mode.mapping)
            .ok_or_else(|| invalid("invalid Vorbis mapping"))?;
        let bins = block_sizes[usize::from(header.long)] / 2;
        if mapping.mux.is_empty() || mapping.mux.len() > 255 {
            return Err(invalid("invalid Vorbis channel count"));
        }
        workspace
            .floors
            .resize_with(mapping.mux.len(), DecodedFloor::default);
        let floors = &mut workspace.floors;
        for (&mux, decoded) in mapping.mux.iter().zip(floors.iter_mut()) {
            decoded.present = false;
            let submap = mapping
                .submaps
                .get(mux)
                .ok_or_else(|| invalid("invalid Vorbis submap"))?;
            let floor = self
                .floors
                .get(submap.floor)
                .ok_or_else(|| invalid("invalid Vorbis floor"))?;
            decoded.present = match floor {
                Floor::One(floor) => floor.decode_amplitudes_into(
                    &mut bits,
                    &self.codebooks,
                    &mut decoded.one,
                    &mut workspace.floor_values,
                )?,
                Floor::Zero(floor) => floor.decode_lsp_into(
                    &mut bits,
                    &self.codebooks,
                    &mut decoded.zero,
                    &mut workspace.floor_vector,
                )?,
            };
        }
        super::residue::clear_vectors(&mut workspace.channels, mapping.mux.len(), bins);
        let channels = &mut workspace.channels;
        if bits.ended() {
            return Ok(Some(header));
        }
        workspace.enabled.clear();
        workspace
            .enabled
            .extend(floors.iter().map(|floor| floor.present));
        let enabled = &mut workspace.enabled;
        for &(magnitude, angle) in &mapping.coupling {
            if magnitude == angle || magnitude >= enabled.len() || angle >= enabled.len() {
                return Err(invalid("invalid Vorbis coupling channels"));
            }
            if enabled[magnitude] || enabled[angle] {
                enabled[magnitude] = true;
                enabled[angle] = true;
            }
        }
        for (index, submap) in mapping.submaps.iter().enumerate() {
            workspace.indices.clear();
            workspace.indices.extend(
                mapping
                    .mux
                    .iter()
                    .enumerate()
                    .filter_map(|(channel, &mux)| (mux == index).then_some(channel)),
            );
            let indices = &workspace.indices;
            if indices.is_empty() {
                continue;
            }
            let residue = self
                .residues
                .get(submap.residue)
                .ok_or_else(|| invalid("invalid Vorbis residue"))?;
            workspace.flags.clear();
            workspace
                .flags
                .extend(indices.iter().map(|&channel| enabled[channel]));
            let vectors = residue.decode_with_workspace(
                &mut bits,
                &self.codebooks,
                &workspace.flags,
                bins,
                &mut workspace.residue,
            )?;
            for (&channel, vector) in indices.iter().zip(vectors) {
                channels[channel].copy_from_slice(vector);
            }
        }
        for &(magnitude, angle) in mapping.coupling.iter().rev() {
            let (magnitude, angle) = if magnitude < angle {
                let (left, right) = channels.split_at_mut(angle);
                (&mut left[magnitude], &mut right[0])
            } else {
                let (left, right) = channels.split_at_mut(magnitude);
                (&mut right[0], &mut left[angle])
            };
            uncouple(magnitude, angle);
        }
        workspace.curve.resize(bins, 0.0);
        let curve = &mut workspace.curve;
        for (channel, decoded) in floors.iter().enumerate() {
            if !decoded.present {
                channels[channel].fill(0.0);
                continue;
            }
            let floor = &self.floors[mapping.submaps[mapping.mux[channel]].floor];
            match floor {
                Floor::One(floor) => floor.render_curve(&decoded.one, curve)?,
                Floor::Zero(floor) => floor.render_curve_with_workspace(
                    &decoded.zero,
                    curve,
                    &mut workspace.floor_cosines,
                )?,
            }
            for (value, &envelope) in channels[channel].iter_mut().zip(curve.iter()) {
                *value *= envelope;
                if !value.is_finite() {
                    return Err(invalid("nonfinite Vorbis spectrum"));
                }
            }
        }
        Ok(Some(header))
    }
}

fn uncouple(magnitude: &mut [f64], angle: &mut [f64]) {
    for (magnitude, angle) in magnitude.iter_mut().zip(angle) {
        let (m, a) = (*magnitude, *angle);
        // Quantization can leave magnitude zero with a nonzero angle. Use the
        // nonnegative quadrant at this boundary (covered by PCM oracle tests).
        (*magnitude, *angle) = if m >= 0.0 {
            if a > 0.0 { (m, m - a) } else { (m + a, m) }
        } else if a > 0.0 {
            (m, m + a)
        } else {
            (m - a, m)
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_coupling_covers_all_sign_quadrants_and_zero() {
        let mut magnitude = [3.0, 3.0, -3.0, -3.0, 0.0, 0.0];
        let mut angle = [2.0, -2.0, 2.0, -2.0, 2.0, -2.0];
        uncouple(&mut magnitude, &mut angle);
        assert_eq!(magnitude, [3.0, 1.0, -3.0, -1.0, 0.0, -2.0]);
        assert_eq!(angle, [1.0, 3.0, -1.0, -3.0, -2.0, 0.0]);
    }
}
