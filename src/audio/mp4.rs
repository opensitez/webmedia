//! Incremental AAC access-unit extraction. PCM synthesis is codec-owned.

use crate::video::mp4::{Mp4AudioIndex, Mp4Error};

const MAX_BUFFER_BYTES: usize = 64 * 1024 * 1024;
const MAX_PACKETS_PER_PUSH: usize = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AacPacket {
    pub presentation_time: i64,
    pub timescale: u32,
    pub data: Vec<u8>,
}

#[derive(Default)]
pub struct Mp4AudioStream {
    bytes: Vec<u8>,
    base_offset: u64,
    index: Option<Mp4AudioIndex>,
    indexed: bool,
    next_sample: usize,
}

impl Mp4AudioStream {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn index(&self) -> Option<&Mp4AudioIndex> {
        self.index.as_ref()
    }

    /// Empty pushes drain buffered packets without reading more input.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<AacPacket>, Mp4Error> {
        if self.indexed && self.index.is_none() {
            return Ok(Vec::new());
        }
        if bytes.len() > MAX_BUFFER_BYTES.saturating_sub(self.bytes.len()) {
            return Err(Mp4Error::TooLarge);
        }
        self.bytes.extend_from_slice(bytes);
        if !self.indexed {
            match Mp4AudioIndex::parse_prefix(&self.bytes) {
                Ok(index) => {
                    self.index = index;
                    self.indexed = true;
                }
                Err(Mp4Error::Incomplete) => return Ok(Vec::new()),
                Err(error) => return Err(error),
            }
        }
        let Some(index) = &self.index else {
            self.bytes.clear();
            return Ok(Vec::new());
        };
        let mut packets = Vec::new();
        while packets.len() < MAX_PACKETS_PER_PUSH {
            let Some(sample) = index.samples.get(self.next_sample) else {
                break;
            };
            let start = sample
                .offset
                .checked_sub(self.base_offset)
                .ok_or(Mp4Error::Invalid("audio sample precedes stream buffer"))?;
            let start = usize::try_from(start).map_err(|_| Mp4Error::TooLarge)?;
            let end = start
                .checked_add(sample.size as usize)
                .ok_or(Mp4Error::TooLarge)?;
            let Some(data) = self.bytes.get(start..end) else {
                break;
            };
            packets.push(AacPacket {
                presentation_time: sample.presentation_time,
                timescale: index.timescale,
                data: data.to_vec(),
            });
            self.next_sample += 1;
        }
        // Keep a small overlap buffer, not the complete downloaded file.
        let retain_offset = index
            .samples
            .get(self.next_sample)
            .map(|sample| sample.offset)
            .unwrap_or(self.base_offset + self.bytes.len() as u64);
        let discard = retain_offset
            .saturating_sub(self.base_offset)
            .min(self.bytes.len() as u64) as usize;
        if discard >= 256 * 1024 || self.next_sample == index.samples.len() {
            self.bytes.drain(..discard);
            self.base_offset += discard as u64;
        }
        Ok(packets)
    }

    /// Call after draining with empty pushes until no more packets are returned.
    pub fn finish(&self) -> Result<(), Mp4Error> {
        if !self.indexed
            || self
                .index
                .as_ref()
                .is_some_and(|index| self.next_sample != index.samples.len())
        {
            return Err(Mp4Error::Incomplete);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncated_movie_is_not_successful_audio() {
        let mut stream = Mp4AudioStream::new();
        assert!(
            stream
                .push(&[0, 0, 0, 8, b'f', b't', b'y', b'p'])
                .unwrap()
                .is_empty()
        );
        assert_eq!(stream.finish(), Err(Mp4Error::Incomplete));
    }

    #[test]
    #[ignore = "set WEBMEDIA_AAC_MP4 to an AAC MP4 fixture"]
    fn streamed_packets_match_file_samples() {
        let bytes = std::fs::read(std::env::var("WEBMEDIA_AAC_MP4").unwrap()).unwrap();
        let index = Mp4AudioIndex::parse_prefix(&bytes).unwrap().unwrap();
        let mut stream = Mp4AudioStream::new();
        let mut packets = Vec::new();
        for chunk in bytes.chunks(8191) {
            packets.extend(stream.push(chunk).unwrap());
        }
        loop {
            let next = stream.push(&[]).unwrap();
            if next.is_empty() {
                break;
            }
            packets.extend(next);
        }
        stream.finish().unwrap();
        assert_eq!(packets.len(), index.samples.len());
        for (packet, sample) in packets.iter().zip(&index.samples) {
            assert_eq!(packet.presentation_time, sample.presentation_time);
            assert_eq!(packet.timescale, index.timescale);
            assert_eq!(
                packet.data,
                bytes[sample.offset as usize..sample.offset as usize + sample.size as usize]
            );
        }
        assert!(stream.bytes.len() < 256 * 1024);
    }
}
