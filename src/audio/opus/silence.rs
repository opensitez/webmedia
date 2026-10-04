//! Deliberately restricted, binary-verified canonical CELT silence decoding.
//!
//! RFC 6716 Table 56 codes the silence flag with probabilities {32767,1}/32768.
//! Support is limited to family-zero mono/stereo configuration 31, a single
//! 20-ms frame, and the canonical two-byte silence frame FF FE. Other silence
//! encodings, duration changes, concealment and all coded audio are unsupported.
//! No inference about silence is made from absent bits, decode failure or PCM
//! amplitude. A failed packet permanently prevents this instance from emitting
//! later PCM, since the skipped packet's synthesis state is unknown.

use super::{IdentificationHeader, Packet, range::RangeDecoder};
use crate::video::backend::{AudioSamples, MediaDecodeError};

/// This is not a general Opus decoder. The stream must start at reset and remain
/// inside the supported silence-only subset. Pre-skip, container end trimming,
/// timestamps and seek/reset policy are the caller's responsibility.
pub struct SilencePacketDecoder {
    channels: u8,
    failed: bool,
}

impl SilencePacketDecoder {
    pub fn new(header: &IdentificationHeader) -> Result<Self, MediaDecodeError> {
        if !(1..=2).contains(&header.channels)
            || header.mapping_family != 0
            || header.streams != 1
            || header.coupled_streams != header.channels - 1
            || header.channel_mapping != (0..header.channels).collect::<Vec<_>>()
        {
            return Err(MediaDecodeError::Unsupported);
        }
        Ok(Self {
            channels: header.channels,
            failed: false,
        })
    }

    /// Return exactly 960 frames at 48 kHz, before pre-skip and end trimming.
    /// Header output gain leaves genuine zero samples unchanged. An unsupported
    /// packet returns no samples and invalidates this instance for the stream.
    pub fn decode_packet(&mut self, bytes: &[u8]) -> Result<AudioSamples, MediaDecodeError> {
        if self.failed {
            return Err(MediaDecodeError::Unsupported);
        }
        self.failed = true;
        let packet = Packet::parse(bytes)?;
        if bytes.len() != 3
            || packet.configuration != 31
            || packet.stereo != (self.channels == 2)
            || packet.frames.len() != 1
            || packet.frames[0] != [0xff, 0xfe]
        {
            return Err(MediaDecodeError::Unsupported);
        }
        let mut entropy = RangeDecoder::new(packet.frames[0]);
        if !entropy.bit(15)? {
            return Err(MediaDecodeError::Unsupported);
        }
        self.failed = false;
        Ok(AudioSamples {
            sample_rate: 48000,
            channels: u16::from(self.channels),
            samples: vec![0.0; packet.samples_48khz() * usize::from(self.channels)],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(channels: u8) -> IdentificationHeader {
        let mut bytes = b"OpusHead\x01\x02\x38\x01\x80\xbb\x00\x00\x00\x00\x00".to_vec();
        bytes[9] = channels;
        IdentificationHeader::parse(&bytes).unwrap()
    }

    #[test]
    fn canonical_packets_have_real_silence_flag_and_exact_duration() {
        for channels in 1..=2 {
            let bytes = [if channels == 1 { 0xf8 } else { 0xfc }, 0xff, 0xfe];
            let mut decoder = SilencePacketDecoder::new(&header(channels)).unwrap();
            let mut range = RangeDecoder::new(&bytes[1..]);
            assert!(range.bit(15).unwrap());
            assert_eq!(range.tell(), 16);
            for _ in 0..10 {
                let pcm = decoder.decode_packet(&bytes).unwrap();
                assert_eq!(
                    (pcm.sample_rate, pcm.channels, pcm.samples.len()),
                    (48000, u16::from(channels), 960 * usize::from(channels))
                );
                assert!(pcm.samples.iter().all(|sample| sample.to_bits() == 0));
            }
        }
    }

    #[test]
    fn unsupported_packets_never_become_silence_and_poison_the_stream() {
        for channels in 1..=2 {
            let valid = [if channels == 1 { 0xf8 } else { 0xfc }, 0xff, 0xfe];
            for position in 0..3 {
                for value in 0..=255 {
                    let mut mutated = valid;
                    mutated[position] = value;
                    if mutated == valid {
                        continue;
                    }
                    let mut decoder = SilencePacketDecoder::new(&header(channels)).unwrap();
                    assert!(decoder.decode_packet(&mutated).is_err());
                    assert!(matches!(
                        decoder.decode_packet(&valid),
                        Err(MediaDecodeError::Unsupported)
                    ));
                }
            }
            for bytes in [
                &[][..],
                &valid[..1],
                &valid[..2],
                &[valid[0], 0xff, 0xfe, 0][..],
            ] {
                let mut decoder = SilencePacketDecoder::new(&header(channels)).unwrap();
                assert!(decoder.decode_packet(bytes).is_err());
                assert!(decoder.decode_packet(&valid).is_err());
            }
        }
        let mut mapped = header(2);
        mapped.mapping_family = 1;
        assert!(SilencePacketDecoder::new(&mapped).is_err());
    }

    fn binary(name: &str, arguments: &[&str], input: Option<&[u8]>) -> Vec<u8> {
        use std::{
            io::Write,
            process::{Command, Stdio},
        };
        let mut child = Command::new(name)
            .args(arguments)
            .stdin(if input.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("FFmpeg/ffprobe binary required");
        if let Some(input) = input {
            child.stdin.take().unwrap().write_all(input).unwrap();
        }
        let result = child.wait_with_output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        result.stdout
    }

    fn ffmpeg(arguments: &[&str], input: Option<&[u8]>) -> Vec<u8> {
        binary("ffmpeg", arguments, input)
    }

    /// Binary use is exclusively fixture creation, packet extraction, and PCM
    /// oracle comparison. No FFmpeg/libopus implementation is read or linked.
    #[test]
    #[ignore = "requires FFmpeg binary fixture and PCM oracle"]
    fn canonical_silence_packets_match_binary_oracle_exactly() {
        for channels in 1..=2 {
            let source = if channels == 1 {
                "anullsrc=r=48000:cl=mono"
            } else {
                "anullsrc=r=48000:cl=stereo"
            };
            let ogg = ffmpeg(
                &[
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    source,
                    "-t",
                    "0.06",
                    "-c:a",
                    "libopus",
                    "-application",
                    "audio",
                    "-frame_duration",
                    "20",
                    "-f",
                    "ogg",
                    "pipe:1",
                ],
                None,
            );
            let header_offset = ogg
                .windows(8)
                .position(|bytes| bytes == b"OpusHead")
                .unwrap();
            let header =
                IdentificationHeader::parse(&ogg[header_offset..header_offset + 19]).unwrap();
            assert_eq!(header.channels, channels);
            assert_eq!(header.pre_skip, 312);
            let packets = ffmpeg(
                &[
                    "-v", "error", "-f", "ogg", "-i", "pipe:0", "-map", "0:a:0", "-c:a", "copy",
                    "-f", "data", "pipe:1",
                ],
                Some(&ogg),
            );
            let oracle = ffmpeg(
                &[
                    "-v",
                    "error",
                    "-f",
                    "ogg",
                    "-i",
                    "pipe:0",
                    "-f",
                    "f32le",
                    "-acodec",
                    "pcm_f32le",
                    "pipe:1",
                ],
                Some(&ogg),
            );
            assert_eq!(packets.len(), 12);
            let mut decoder = SilencePacketDecoder::new(&header).unwrap();
            let mut decoded = Vec::new();
            for packet in packets.chunks_exact(3) {
                decoded.extend(decoder.decode_packet(packet).unwrap().samples);
            }
            // The fixture is exactly 60 ms; remove the actual identification
            // pre-skip and trim the encoder's final padding to that duration.
            let start = usize::from(header.pre_skip) * usize::from(channels);
            let frames = 2880 * usize::from(channels);
            let decoded: Vec<_> = decoded[start..start + frames]
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect();
            assert_eq!(oracle.len(), frames * 4);
            assert_eq!(decoded, oracle);
            eprintln!(
                "canonical silence oracle: channels={channels} packets=4 frames=2880 max_error=0 byte_exact=true"
            );

            let channel_text = channels.to_string();
            let music = ffmpeg(
                &[
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=48000",
                    "-t",
                    "0.06",
                    "-ac",
                    &channel_text,
                    "-c:a",
                    "libopus",
                    "-application",
                    "audio",
                    "-frame_duration",
                    "20",
                    "-f",
                    "ogg",
                    "pipe:1",
                ],
                None,
            );
            let sizes = binary(
                "ffprobe",
                &[
                    "-v",
                    "error",
                    "-show_packets",
                    "-show_entries",
                    "packet=size",
                    "-of",
                    "default=noprint_wrappers=1",
                    "-f",
                    "ogg",
                    "-i",
                    "pipe:0",
                ],
                Some(&music),
            );
            let size: usize = String::from_utf8(sizes)
                .unwrap()
                .lines()
                .find_map(|line| line.strip_prefix("size="))
                .unwrap()
                .parse()
                .unwrap();
            let packets = ffmpeg(
                &[
                    "-v", "error", "-f", "ogg", "-i", "pipe:0", "-map", "0:a:0", "-c:a", "copy",
                    "-f", "data", "pipe:1",
                ],
                Some(&music),
            );
            let music_pcm = ffmpeg(
                &[
                    "-v",
                    "error",
                    "-f",
                    "ogg",
                    "-i",
                    "pipe:0",
                    "-f",
                    "f32le",
                    "-acodec",
                    "pcm_f32le",
                    "pipe:1",
                ],
                Some(&music),
            );
            assert!(
                music_pcm
                    .chunks_exact(4)
                    .any(|sample| f32::from_le_bytes(sample.try_into().unwrap()) != 0.0)
            );
            let mut unsupported = SilencePacketDecoder::new(&header).unwrap();
            assert!(matches!(
                unsupported.decode_packet(&packets[..size]),
                Err(MediaDecodeError::Unsupported)
            ));
            let silence_packet = [if channels == 1 { 0xf8 } else { 0xfc }, 0xff, 0xfe];
            assert!(matches!(
                unsupported.decode_packet(&silence_packet),
                Err(MediaDecodeError::Unsupported)
            ));
            eprintln!(
                "coded tone rejection: channels={channels} first_packet_bytes={size} oracle_nonzero=true unsupported=true subsequent_silence_rejected=true"
            );
        }
    }
}
