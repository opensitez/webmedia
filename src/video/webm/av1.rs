//! WebM AV1 routing. Pixel reconstruction stays in the shared AV1 decoder.

use crate::video::av1::{self, Obu, ObuStream, syntax::SequenceHeader};
use crate::video::backend::{MediaDecodeError, VideoFrame};
use crate::video::vp8_keyframe::{YuvKeyFrame, YuvMatrix};
use std::{collections::VecDeque, sync::Arc};

pub(super) struct Av1PacketDecoder {
    configuration: [u8; 4],
    sequence: Option<SequenceHeader>,
    decoder: av1::Av1Decoder,
    pending: VecDeque<Obu>,
    conversion: Option<YuvKeyFrame>,
}

fn invalid(message: &str) -> MediaDecodeError {
    MediaDecodeError::InvalidData(message.into())
}

fn syntax_error(error: av1::syntax::Error) -> MediaDecodeError {
    match error {
        av1::syntax::Error::Unsupported(_) => MediaDecodeError::Unsupported,
        _ => invalid(&format!("AV1 syntax: {error:?}")),
    }
}

fn pack_eight_bit_row(source: &[u16], target: &mut [u8]) -> bool {
    if source.len() != target.len() { return false; }
    let mut offset = 0;
    // Equal lengths and the eight-sample bounds cover every vector load/store.
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        while offset + 8 <= source.len() {
            let values = vld1q_u16(source.as_ptr().add(offset));
            if vmaxvq_u16(values) > 255 { return false; }
            vst1_u8(target.as_mut_ptr().add(offset), vmovn_u16(values));
            offset += 8;
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        let high_bits = _mm_set1_epi16(-256);
        let zero = _mm_setzero_si128();
        while offset + 8 <= source.len() {
            let values = _mm_loadu_si128(source.as_ptr().add(offset).cast());
            if _mm_movemask_epi8(_mm_cmpeq_epi16(_mm_and_si128(values, high_bits), zero)) != 0xffff {
                return false;
            }
            _mm_storel_epi64(target.as_mut_ptr().add(offset).cast(), _mm_packus_epi16(values, zero));
            offset += 8;
        }
    }
    for (&value, out) in source[offset..].iter().zip(&mut target[offset..]) {
        let Ok(value) = u8::try_from(value) else { return false; };
        *out = value;
    }
    true
}

fn obus(bytes: &[u8]) -> Result<Vec<Obu>, MediaDecodeError> {
    let mut stream = ObuStream::new();
    let result = stream
        .push(bytes)
        .map_err(|error| invalid(&format!("AV1 OBU: {error:?}")))?;
    stream
        .finish()
        .map_err(|error| invalid(&format!("AV1 OBU: {error:?}")))?;
    Ok(result)
}

impl Av1PacketDecoder {
    pub(super) fn new(private: &[u8]) -> Result<Self, MediaDecodeError> {
        let header: [u8; 4] = private
            .get(..4)
            .ok_or_else(|| invalid("missing AV1 configuration"))?
            .try_into()
            .unwrap();
        if header[0] != 0x81
            || header[3] & 0xe0 != 0
            || (header[3] & 0x10 == 0 && header[3] & 0x0f != 0)
        {
            return Err(invalid("invalid AV1 configuration record"));
        }
        let mut decoder = Self {
            configuration: header,
            sequence: None,
            decoder: av1::Av1Decoder::new(),
            pending: VecDeque::new(),
            conversion: None,
        };
        for (index, obu) in obus(&private[4..])?.into_iter().enumerate() {
            match obu.kind {
                1 if index == 0 => decoder.set_sequence(&obu)?,
                5 | 15 => {}
                _ => return Err(MediaDecodeError::Unsupported),
            }
        }
        Ok(decoder)
    }

    fn set_sequence(&mut self, obu: &Obu) -> Result<(), MediaDecodeError> {
        let sequence = SequenceHeader::parse(&obu.payload).map_err(syntax_error)?;
        let operating = sequence
            .operating_points
            .first()
            .ok_or_else(|| invalid("missing AV1 operating point"))?;
        let header = self.configuration;
        if header[1] >> 5 != sequence.profile
            || header[1] & 31 != operating.level
            || (header[2] & 128 != 0) != operating.tier
            || (header[2] & 64 != 0) != (sequence.bit_depth > 8)
            || (header[2] & 32 != 0) != (sequence.bit_depth == 12)
            || (header[2] & 16 != 0) != sequence.monochrome
            || (header[2] & 8 != 0) != sequence.subsampling_x
            || (header[2] & 4 != 0) != sequence.subsampling_y
            || header[2] & 3 != sequence.chroma_sample_position
        {
            return Err(invalid("AV1 configuration disagrees with sequence"));
        }
        self.decoder.decode_obu(obu).map_err(syntax_error)?;
        self.sequence = Some(sequence);
        Ok(())
    }

    pub(super) fn begin_packet(&mut self, bytes: &[u8]) -> Result<(), MediaDecodeError> {
        if !self.pending.is_empty() {
            return Err(invalid("undrained AV1 temporal unit"));
        }
        self.pending.extend(obus(bytes)?);
        Ok(())
    }

    pub(super) fn next_frame(
        &mut self,
        timestamp: f32,
    ) -> Result<(Option<VideoFrame>, bool), MediaDecodeError> {
        while let Some(obu) = self.pending.pop_front() {
            match obu.kind {
                1 => self.set_sequence(&obu)?,
                2 | 5 | 15 => {}
                3 | 6 => {
                    let sequence = self
                        .sequence
                        .as_ref()
                        .ok_or_else(|| invalid("AV1 frame before sequence"))?;
                    let Some(decoded) = self.decoder.decode_obu(&obu).map_err(syntax_error)? else { continue; };
                    // Reuse the existing accelerated limited-range 4:2:0 converter.
                    // Other precision/color formats remain explicit errors until
                    // their conversion path is implemented, not silently coerced.
                    if decoded.bit_depth != 8
                        || sequence.full_range
                        || !matches!(sequence.matrix_coefficients, 1 | 2 | 5 | 6)
                        || !matches!(sequence.color_primaries, 1 | 2 | 5 | 6)
                        || !matches!(sequence.transfer_characteristics, 1 | 2 | 6)
                        || (!sequence.monochrome
                            && !(sequence.subsampling_x && sequence.subsampling_y))
                    {
                        return Err(MediaDecodeError::Unsupported);
                    }
                    let width = decoded.header.width as usize;
                    let height = decoded.header.height as usize;
                    if decoded.planes.len() != if sequence.monochrome { 1 } else { 3 } {
                        return Err(invalid("invalid AV1 decoded plane count"));
                    }
                    let yuv = conversion_buffer(&mut self.conversion, width, height);
                    if sequence.monochrome {
                        yuv.u.pixels.fill(128);
                        yuv.v.pixels.fill(128);
                    }
                    for (index, (source, target)) in decoded
                        .planes
                        .iter()
                        .zip([&mut yuv.y, &mut yuv.u, &mut yuv.v])
                        .enumerate()
                    {
                        let expected = if index == 0 {
                            (width, height)
                        } else {
                            (width.div_ceil(2), height.div_ceil(2))
                        };
                        if (source.width, source.height) != expected
                            || source.stride < source.width
                            || source
                                .stride
                                .checked_mul(source.height)
                                .is_none_or(|n| n > source.samples.len())
                        {
                            return Err(invalid("invalid AV1 decoded plane shape"));
                        }
                        for row in 0..source.height {
                            if !pack_eight_bit_row(
                                &source.samples[row * source.stride..][..source.width],
                                &mut target.pixels[row * target.width..][..source.width],
                            ) { return Err(invalid("AV1 sample exceeds bit depth")); }
                        }
                    }
                    return Ok((
                        Some(VideoFrame {
                            width: width as u32,
                            height: height as u32,
                            rgba: Arc::new(yuv.rgba_with_matrix(if sequence.matrix_coefficients == 1 {
                                YuvMatrix::Bt709
                            } else {
                                YuvMatrix::Bt601
                            })),
                            timestamp,
                        }),
                        self.pending.is_empty(),
                    ));
                }
                _ => return Err(MediaDecodeError::Unsupported),
            }
        }
        Ok((None, true))
    }
}

fn conversion_buffer(buffer: &mut Option<YuvKeyFrame>, width: usize, height: usize) -> &mut YuvKeyFrame {
    if buffer.as_ref().is_none_or(|frame| frame.width != width || frame.height != height) {
        *buffer = Some(YuvKeyFrame::new(width, height));
    }
    buffer.as_mut().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_storage_is_reused_until_dimensions_change() {
        let mut buffer = None;
        let frame = conversion_buffer(&mut buffer, 65, 33);
        frame.y.pixels.fill(17);
        let pointers = (frame.y.pixels.as_ptr(), frame.u.pixels.as_ptr(), frame.v.pixels.as_ptr());
        let frame = conversion_buffer(&mut buffer, 65, 33);
        assert_eq!(pointers, (frame.y.pixels.as_ptr(), frame.u.pixels.as_ptr(), frame.v.pixels.as_ptr()));
        assert!(frame.y.pixels.iter().all(|&sample| sample == 17));
        let frame = conversion_buffer(&mut buffer, 128, 64);
        assert_eq!((frame.width, frame.height), (128, 64));
        assert!(frame.y.pixels.iter().all(|&sample| sample == 0));
    }

    #[test]
    fn checked_plane_packing_preserves_all_values_and_odd_tails() {
        for width in 0..=65 {
            for base in 0..=255u16 {
                let source: Vec<_> = (0..width).map(|index| (base + index as u16 * 37) & 255).collect();
                let mut target = vec![0xa5; width + 2];
                assert!(pack_eight_bit_row(&source, &mut target[1..width + 1]));
                assert_eq!(target[0], 0xa5);
                assert_eq!(target[width + 1], 0xa5);
                assert!(target[1..width + 1].iter().zip(source).all(|(&out, value)| u16::from(out) == value));
            }
        }
    }

    #[test]
    fn checked_plane_packing_rejects_overflow_in_every_lane() {
        for value in [256, 32768, 65535] {
            for index in 0..25 {
                let mut source = vec![255; 25];
                source[index] = value;
                assert!(!pack_eight_bit_row(&source, &mut [0; 25]));
            }
        }
        assert!(!pack_eight_bit_row(&[0; 8], &mut [0; 7]));
    }

    #[test]
    #[ignore = "manual optimized-build checked plane-packing benchmark"]
    fn benchmark_checked_plane_packing() {
        use std::{hint::black_box, time::Instant};
        #[inline(never)]
        fn scalar(source: &[u16], target: &mut [u8]) -> bool {
            for (&value, out) in source.iter().zip(target) {
                let Ok(value) = u8::try_from(value) else { return false; };
                *out = value;
            }
            true
        }
        let source: Vec<u16> = (0..1920).map(|index| ((index * 37) & 255) as u16).collect();
        let mut target = vec![0; source.len()];
        for round in 0..5 {
            let start = Instant::now();
            for _ in 0..8192 { assert!(scalar(black_box(&source), black_box(&mut target))); }
            let scalar_time = start.elapsed();
            let expected = target.clone();
            let start = Instant::now();
            for _ in 0..8192 { assert!(pack_eight_bit_row(black_box(&source), black_box(&mut target))); }
            let packed_time = start.elapsed();
            assert_eq!(target, expected);
            eprintln!("round {round}: scalar {scalar_time:?}, checked bulk {packed_time:?}");
        }
    }
}
