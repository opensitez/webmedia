//! Bounded ISO base media file parser for classic AVC-in-MP4 sample tables.
//!
//! A front-loaded `moov` can be indexed before `mdat` arrives. Files with a
//! trailing `moov` are indexed once the bounded input buffer reaches it.
//! Fragmented MP4 and external data references are not handled here.

use super::h264::{AvcConfig, AvcError};

const MAX_MOOV_BYTES: u64 = 32 * 1024 * 1024;
const MAX_SAMPLES: usize = 1_000_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mp4Error {
    Incomplete,
    Invalid(&'static str),
    Unsupported(&'static str),
    TooLarge,
    Avc(AvcError),
}

impl From<AvcError> for Mp4Error {
    fn from(error: AvcError) -> Self {
        Self::Avc(error)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sample {
    pub offset: u64,
    pub size: u32,
    pub decode_time: u64,
    pub presentation_time: i64,
    pub keyframe: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mp4Index {
    pub timescale: u32,
    pub duration_ticks: u64,
    pub config: AvcConfig,
    pub samples: Vec<Sample>,
}

#[derive(Clone, Copy)]
struct Atom<'a> {
    kind: [u8; 4],
    data: &'a [u8],
    size: usize,
}

fn u32_at(data: &[u8], offset: usize) -> Result<u32, Mp4Error> {
    let bytes = data.get(offset..offset + 4).ok_or(Mp4Error::Incomplete)?;
    Ok(u32::from_be_bytes(bytes.try_into().unwrap()))
}

fn u64_at(data: &[u8], offset: usize) -> Result<u64, Mp4Error> {
    let bytes = data.get(offset..offset + 8).ok_or(Mp4Error::Incomplete)?;
    Ok(u64::from_be_bytes(bytes.try_into().unwrap()))
}

fn atom_at(data: &[u8], offset: usize) -> Result<Atom<'_>, Mp4Error> {
    let header = data.get(offset..offset + 8).ok_or(Mp4Error::Incomplete)?;
    let kind = header[4..8].try_into().unwrap();
    let short_size = u32::from_be_bytes(header[..4].try_into().unwrap());
    let (size, header_size) = match short_size {
        0 => (data.len() - offset, 8),
        1 => (
            usize::try_from(u64_at(data, offset + 8)?).map_err(|_| Mp4Error::TooLarge)?,
            16,
        ),
        n => (n as usize, 8),
    };
    if size < header_size {
        return Err(Mp4Error::Invalid("invalid box size"));
    }
    let end = offset.checked_add(size).ok_or(Mp4Error::TooLarge)?;
    let payload = data
        .get(offset + header_size..end)
        .ok_or(Mp4Error::Incomplete)?;
    Ok(Atom {
        kind,
        data: payload,
        size,
    })
}

fn children(data: &[u8]) -> Result<Vec<Atom<'_>>, Mp4Error> {
    let mut atoms = Vec::new();
    let mut offset = 0;
    while offset < data.len() {
        let atom = atom_at(data, offset)?;
        offset += atom.size;
        atoms.push(atom);
    }
    Ok(atoms)
}

fn child<'a>(data: &'a [u8], kind: &[u8; 4]) -> Result<Atom<'a>, Mp4Error> {
    children(data)?
        .into_iter()
        .find(|atom| &atom.kind == kind)
        .ok_or(Mp4Error::Invalid("missing required box"))
}

fn path<'a>(mut data: &'a [u8], kinds: &[[u8; 4]]) -> Result<Atom<'a>, Mp4Error> {
    let mut last = None;
    for kind in kinds {
        let atom = child(data, kind)?;
        data = atom.data;
        last = Some(atom);
    }
    last.ok_or(Mp4Error::Invalid("empty box path"))
}

fn table_u32(data: &[u8], entry_width: usize) -> Result<(usize, usize), Mp4Error> {
    let count = u32_at(data, 4)? as usize;
    if count > MAX_SAMPLES {
        return Err(Mp4Error::TooLarge);
    }
    let len = count
        .checked_mul(entry_width)
        .and_then(|n| n.checked_add(8))
        .ok_or(Mp4Error::TooLarge)?;
    if data.len() < len {
        return Err(Mp4Error::Incomplete);
    }
    Ok((count, 8))
}

impl Mp4Index {
    /// Parse a complete `moov` from a file prefix. A trailing `moov` requires
    /// the preceding `mdat` to be present; sample offsets refer to the full file.
    pub fn parse_prefix(prefix: &[u8]) -> Result<Self, Mp4Error> {
        let mut offset = 0;
        let moov = loop {
            let header = prefix.get(offset..offset + 8).ok_or(Mp4Error::Incomplete)?;
            let size = u32::from_be_bytes(header[..4].try_into().unwrap()) as u64;
            if &header[4..8] == b"moov" {
                if size > MAX_MOOV_BYTES {
                    return Err(Mp4Error::TooLarge);
                }
                break atom_at(prefix, offset)?;
            }
            let atom = atom_at(prefix, offset)?;
            offset = offset.checked_add(atom.size).ok_or(Mp4Error::TooLarge)?;
        };

        for track in children(moov.data)?
            .into_iter()
            .filter(|atom| atom.kind == *b"trak")
        {
            let media = child(track.data, b"mdia")?;
            let handler = child(media.data, b"hdlr")?;
            if handler.data.get(8..12) != Some(&b"vide"[..]) {
                continue;
            }
            return Self::parse_video_track(media.data);
        }
        Err(Mp4Error::Invalid("no video track"))
    }

    fn parse_video_track(media: &[u8]) -> Result<Self, Mp4Error> {
        let mdhd = child(media, b"mdhd")?;
        let timescale = u32_at(
            mdhd.data,
            if mdhd.data.first() == Some(&1) {
                20
            } else {
                12
            },
        )?;
        if timescale == 0 {
            return Err(Mp4Error::Invalid("zero media timescale"));
        }
        let stbl = path(media, &[*b"minf", *b"stbl"])?;

        let stsd = child(stbl.data, b"stsd")?;
        if u32_at(stsd.data, 4)? != 1 {
            return Err(Mp4Error::Unsupported("multiple sample descriptions"));
        }
        let entry = atom_at(stsd.data, 8)?;
        if entry.kind != *b"avc1" || entry.data.len() < 78 {
            return Err(Mp4Error::Unsupported("video sample entry is not avc1"));
        }
        let avcc = child(&entry.data[78..], b"avcC")?;
        let config = AvcConfig::parse(avcc.data)?;

        let stsz = child(stbl.data, b"stsz")?;
        let sample_size = u32_at(stsz.data, 4)?;
        let sample_count = u32_at(stsz.data, 8)? as usize;
        if sample_count > MAX_SAMPLES {
            return Err(Mp4Error::TooLarge);
        }
        let sizes = if sample_size == 0 {
            let end = 12usize
                .checked_add(sample_count.checked_mul(4).ok_or(Mp4Error::TooLarge)?)
                .ok_or(Mp4Error::TooLarge)?;
            if stsz.data.len() < end {
                return Err(Mp4Error::Incomplete);
            }
            (0..sample_count)
                .map(|n| u32_at(stsz.data, 12 + n * 4).unwrap())
                .collect::<Vec<_>>()
        } else {
            vec![sample_size; sample_count]
        };

        let offsets = if let Ok(stco) = child(stbl.data, b"stco") {
            let (count, _) = table_u32(stco.data, 4)?;
            (0..count)
                .map(|n| u64::from(u32_at(stco.data, 8 + n * 4).unwrap()))
                .collect::<Vec<_>>()
        } else {
            let co64 = child(stbl.data, b"co64")?;
            let (count, _) = table_u32(co64.data, 8)?;
            (0..count)
                .map(|n| u64_at(co64.data, 8 + n * 8).unwrap())
                .collect::<Vec<_>>()
        };
        let stsc = child(stbl.data, b"stsc")?;
        let (stsc_count, _) = table_u32(stsc.data, 12)?;
        if stsc_count == 0 || offsets.is_empty() {
            return Err(Mp4Error::Invalid("empty chunk table"));
        }
        let mut chunks = Vec::with_capacity(stsc_count);
        for n in 0..stsc_count {
            let at = 8 + n * 12;
            let first = u32_at(stsc.data, at)?;
            let per_chunk = u32_at(stsc.data, at + 4)?;
            let description = u32_at(stsc.data, at + 8)?;
            if first == 0
                || per_chunk == 0
                || description != 1
                || chunks.last().is_some_and(|(last, _)| first <= *last)
            {
                return Err(Mp4Error::Invalid("invalid sample-to-chunk table"));
            }
            chunks.push((first, per_chunk));
        }
        if chunks[0].0 != 1 {
            return Err(Mp4Error::Invalid("first chunk not mapped"));
        }

        let stts = child(stbl.data, b"stts")?;
        let (timing_count, _) = table_u32(stts.data, 8)?;
        let mut decode_times = Vec::with_capacity(sample_count);
        let mut clock = 0u64;
        for n in 0..timing_count {
            let count = u32_at(stts.data, 8 + n * 8)? as usize;
            let delta = u32_at(stts.data, 12 + n * 8)?;
            if count > sample_count - decode_times.len() {
                return Err(Mp4Error::Invalid("timing count exceeds samples"));
            }
            for _ in 0..count {
                decode_times.push(clock);
                clock = clock
                    .checked_add(u64::from(delta))
                    .ok_or(Mp4Error::TooLarge)?;
            }
        }
        if decode_times.len() != sample_count {
            return Err(Mp4Error::Invalid("timing count mismatch"));
        }

        let mut composition_offsets = vec![0i64; sample_count];
        if let Ok(ctts) = child(stbl.data, b"ctts") {
            let (count, _) = table_u32(ctts.data, 8)?;
            let mut sample = 0;
            for n in 0..count {
                let run = u32_at(ctts.data, 8 + n * 8)? as usize;
                let raw = u32_at(ctts.data, 12 + n * 8)?;
                let value = if ctts.data[0] == 1 {
                    i64::from(raw as i32)
                } else {
                    i64::from(raw)
                };
                if run > sample_count - sample {
                    return Err(Mp4Error::Invalid("composition count exceeds samples"));
                }
                composition_offsets[sample..sample + run].fill(value);
                sample += run;
            }
            if sample != sample_count {
                return Err(Mp4Error::Invalid("composition count mismatch"));
            }
        }
        let mut sync = vec![true; sample_count];
        if let Ok(stss) = child(stbl.data, b"stss") {
            sync.fill(false);
            let (count, _) = table_u32(stss.data, 4)?;
            for n in 0..count {
                let index = u32_at(stss.data, 8 + n * 4)? as usize;
                if index == 0 || index > sample_count {
                    return Err(Mp4Error::Invalid("sync sample out of range"));
                }
                sync[index - 1] = true;
            }
        }

        let mut samples = Vec::with_capacity(sample_count);
        let mut sample = 0;
        let mut mapping = 0;
        for (n, chunk_offset) in offsets.into_iter().enumerate() {
            let chunk_index = n as u32 + 1;
            while mapping + 1 < chunks.len() && chunks[mapping + 1].0 <= chunk_index {
                mapping += 1;
            }
            let mut offset = chunk_offset;
            for _ in 0..chunks[mapping].1 {
                if sample >= sample_count {
                    return Err(Mp4Error::Invalid("chunk count exceeds samples"));
                }
                let size = sizes[sample];
                let presentation_time = i64::try_from(decode_times[sample])
                    .map_err(|_| Mp4Error::TooLarge)?
                    .checked_add(composition_offsets[sample])
                    .ok_or(Mp4Error::TooLarge)?;
                samples.push(Sample {
                    offset,
                    size,
                    decode_time: decode_times[sample],
                    presentation_time,
                    keyframe: sync[sample],
                });
                offset = offset
                    .checked_add(u64::from(size))
                    .ok_or(Mp4Error::TooLarge)?;
                sample += 1;
            }
        }
        if sample != sample_count {
            return Err(Mp4Error::Invalid("chunk count mismatch"));
        }
        Ok(Self {
            timescale,
            duration_ticks: clock,
            config,
            samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::h264::NalStream;

    fn boxed(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(data.len() + 8);
        result.extend(((data.len() + 8) as u32).to_be_bytes());
        result.extend(kind);
        result.extend(data);
        result
    }

    fn full_table(entries: &[u32], fields_per_entry: usize) -> Vec<u8> {
        let mut data = vec![0; 4];
        assert!(entries.len().is_multiple_of(fields_per_entry));
        data.extend(((entries.len() / fields_per_entry) as u32).to_be_bytes());
        for entry in entries {
            data.extend(entry.to_be_bytes());
        }
        data
    }

    #[test]
    fn indexes_classic_avc_samples_with_composition_offsets() {
        let avcc = [
            0x01, 0x64, 0x00, 0x20, 0xff, 0xe1, 0x00, 0x1d, 0x67, 0x64, 0x00, 0x20, 0xac, 0xd9,
            0x40, 0x50, 0x05, 0xbb, 0xff, 0x04, 0x40, 0x04, 0x41, 0x10, 0x00, 0x00, 0x03, 0x00,
            0x10, 0x00, 0x00, 0x06, 0x48, 0xf1, 0x83, 0x19, 0x60, 0x01, 0x00, 0x06, 0x68, 0xeb,
            0xe3, 0xcb, 0x22, 0xc0, 0xfd, 0xf8, 0xf8, 0x00,
        ];
        let mut entry = vec![0; 78];
        entry.extend(boxed(b"avcC", &avcc));
        let entry = boxed(b"avc1", &entry);
        let mut stsd = vec![0; 4];
        stsd.extend(1u32.to_be_bytes());
        stsd.extend(entry);
        let mut stsz = vec![0; 8];
        stsz.extend(2u32.to_be_bytes());
        stsz.extend(6u32.to_be_bytes());
        stsz.extend(6u32.to_be_bytes());
        let stsc = full_table(&[1, 2, 1], 3);
        let stts = full_table(&[2, 100], 2);
        let ctts = full_table(&[1, 150, 1, 0], 2);
        let mut stss = vec![0; 4];
        stss.extend(1u32.to_be_bytes());
        stss.extend(2u32.to_be_bytes());
        let mut stco = vec![0; 4];
        stco.extend(1u32.to_be_bytes());
        stco.extend(0u32.to_be_bytes());
        let make_moov = |stco: &[u8]| {
            let mut stbl = Vec::new();
            for (kind, data) in [
                (b"stsd", stsd.as_slice()),
                (b"stsz", stsz.as_slice()),
                (b"stsc", stsc.as_slice()),
                (b"stts", stts.as_slice()),
                (b"ctts", ctts.as_slice()),
                (b"stss", stss.as_slice()),
                (b"stco", stco),
            ] {
                stbl.extend(boxed(kind, data));
            }
            let minf = boxed(b"minf", &boxed(b"stbl", &stbl));
            let mut mdhd = vec![0; 12];
            mdhd.extend(1000u32.to_be_bytes());
            let mut hdlr = vec![0; 8];
            hdlr.extend(b"vide");
            let mut mdia = boxed(b"mdhd", &mdhd);
            mdia.extend(boxed(b"hdlr", &hdlr));
            mdia.extend(minf);
            boxed(b"moov", &boxed(b"trak", &boxed(b"mdia", &mdia)))
        };
        let ftyp = boxed(b"ftyp", b"isom");
        let first_moov = make_moov(&stco);
        let first_offset = (ftyp.len() + first_moov.len() + 8) as u32;
        let truncated_prefix_len = ftyp.len() + first_moov.len() - 1;
        stco[8..12].copy_from_slice(&first_offset.to_be_bytes());
        let moov = make_moov(&stco);
        let mut file = ftyp;
        file.extend(moov);
        file.extend(boxed(b"mdat", &[0; 12]));
        let index = Mp4Index::parse_prefix(&file).unwrap();
        assert_eq!(index.timescale, 1000);
        assert_eq!(index.duration_ticks, 200);
        assert_eq!(index.samples.len(), 2);
        assert_eq!(index.samples[0].offset, u64::from(first_offset));
        assert_eq!(index.samples[1].offset, u64::from(first_offset + 6));
        assert_eq!(index.samples[0].decode_time, 0);
        assert_eq!(index.samples[1].decode_time, 100);
        assert_eq!(index.samples[0].presentation_time, 150);
        assert_eq!(index.samples[1].presentation_time, 100);
        assert!(!index.samples[0].keyframe);
        assert!(index.samples[1].keyframe);
        assert_eq!(
            Mp4Index::parse_prefix(&file[..truncated_prefix_len]),
            Err(Mp4Error::Incomplete)
        );
    }

    #[test]
    fn indexes_maroc_video_prefix_when_fixture_is_available() {
        let Ok(path) = std::env::var("WEBCORE_MP4_FIXTURE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        assert_eq!(index.config.sequence_parameters[0].profile_idc, 100);
        assert_eq!(
            (
                index.config.sequence_parameters[0].width,
                index.config.sequence_parameters[0].height
            ),
            (1280, 720)
        );
        assert!(index.samples.len() > 1000);
        assert!(!index.samples[0].keyframe);
        assert!(index.samples.iter().any(|sample| sample.keyframe));
        assert!(index.samples[0].offset < bytes.len() as u64);
        let first = &index.samples[0];
        let start = first.offset as usize;
        let end = start + first.size as usize;
        let mut nal_stream = NalStream::new(index.config.nal_length_size).unwrap();
        let units = nal_stream.push(&bytes[start..end]).unwrap();
        nal_stream.finish().unwrap();
        assert!(units.iter().any(|nal| nal[0] & 0x1f == 5));
        assert!(
            index
                .samples
                .windows(2)
                .all(|pair| pair[0].decode_time <= pair[1].decode_time)
        );
    }

    #[test]
    fn indexes_trailing_moov_when_fixture_is_available() {
        let Ok(path) = std::env::var("WEBCORE_TRAILING_MOOV_FIXTURE") else {
            return;
        };
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(&bytes[0x2c..0x30], b"mdat");
        let mdat_size = u32_at(&bytes, 0x28).unwrap() as usize;
        let moov_offset = 0x28 + mdat_size;
        assert_eq!(&bytes[moov_offset + 4..moov_offset + 8], b"moov");
        assert_eq!(
            Mp4Index::parse_prefix(&bytes[..moov_offset]),
            Err(Mp4Error::Incomplete)
        );
        let index = Mp4Index::parse_prefix(&bytes).unwrap();
        assert_eq!(index.config.sequence_parameters[0].profile_idc, 100);
        assert_eq!(
            (
                index.config.sequence_parameters[0].width,
                index.config.sequence_parameters[0].height
            ),
            (1920, 1080)
        );
        assert!(index.samples.len() > 100);
    }
}
