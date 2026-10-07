//! Bounded ISO base media parser for classic tables and movie fragments.
//!
//! A front-loaded `moov` can be indexed before `mdat` arrives. Files with a
//! trailing `moov` are indexed once the bounded input buffer reaches it.
//! Fragment metadata is indexed separately; external data references are unsupported.

use super::h264::{AvcConfig, AvcError};

const MAX_MOOV_BYTES: u64 = 32 * 1024 * 1024;
const MAX_SAMPLES: usize = 1_000_000;
// Generous access-unit bound also caps the compressed audio worker queue.
const MAX_AAC_SAMPLE_BYTES: u32 = 256 * 1024;

#[cfg(test)]
pub(crate) use tests::fragmented_test_file;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mp4VideoCodec {
    Avc(AvcConfig),
    Vp8,
    Vp9,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mp4VideoIndex {
    pub timescale: u32,
    pub duration_ticks: u64,
    pub codec: Mp4VideoCodec,
    pub width: u32,
    pub height: u32,
    pub samples: Vec<Sample>,
}

/// AAC access units and their AudioSpecificConfig, independent of video codec.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mp4AudioIndex {
    pub timescale: u32,
    pub duration_ticks: u64,
    pub audio_specific_config: Vec<u8>,
    pub samples: Vec<Sample>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Mp4FragmentDefaults {
    pub track_id: u32,
    pub sample_description_index: u32,
    pub duration: u32,
    pub size: u32,
    pub flags: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Mp4FragmentInit {
    pub video: Option<(u32, Mp4VideoIndex)>,
    pub audio: Option<(u32, Mp4AudioIndex)>,
    pub defaults: Vec<Mp4FragmentDefaults>,
    audio_media_start: i64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Mp4FragmentDecodeTimes {
    pub audio: u64,
    pub video: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Mp4Fragment {
    pub consumed_bytes: usize,
    pub audio_samples: Vec<Sample>,
    pub video_samples: Vec<Sample>,
    pub decode_times: Mp4FragmentDecodeTimes,
}

impl Mp4FragmentInit {
    pub fn parse_prefix(prefix: &[u8]) -> Result<Self, Mp4Error> {
        let moov = movie_box(prefix)?;
        let movie_children = children(moov.data)?;
        let mvex = movie_children
            .iter()
            .find(|atom| atom.kind == *b"mvex")
            .ok_or(Mp4Error::Unsupported("not fragmented MP4"))?;
        let mut defaults = Vec::new();
        for trex in children(mvex.data)?
            .into_iter()
            .filter(|atom| atom.kind == *b"trex")
        {
            if u32_at(trex.data, 0)? != 0 {
                return Err(Mp4Error::Unsupported("unsupported trex version or flags"));
            }
            let track_id = u32_at(trex.data, 4)?;
            if track_id == 0
                || defaults
                    .iter()
                    .any(|d: &Mp4FragmentDefaults| d.track_id == track_id)
            {
                return Err(Mp4Error::Invalid("invalid fragment track defaults"));
            }
            defaults.push(Mp4FragmentDefaults {
                track_id,
                sample_description_index: u32_at(trex.data, 8)?,
                duration: u32_at(trex.data, 12)?,
                size: u32_at(trex.data, 16)?,
                flags: u32_at(trex.data, 20)?,
            });
            if trex.data.len() != 24 {
                return Err(Mp4Error::Invalid("invalid trex length"));
            }
        }
        let mut video_id = None;
        let mut audio_id = None;
        let mut audio_media_start = 0;
        for track in movie_children.iter().filter(|atom| atom.kind == *b"trak") {
            let media = child(track.data, b"mdia")?;
            let handler = child(media.data, b"hdlr")?;
            let kind = handler.data.get(8..12).ok_or(Mp4Error::Incomplete)?;
            if kind != b"vide" && kind != b"soun" {
                continue;
            }
            let tkhd = child(track.data, b"tkhd")?;
            let offset = match tkhd.data.first() {
                Some(0) => 12,
                Some(1) => 20,
                _ => return Err(Mp4Error::Unsupported("unsupported track header version")),
            };
            let id = u32_at(tkhd.data, offset)?;
            if id == 0 || !defaults.iter().any(|d| d.track_id == id) {
                return Err(Mp4Error::Invalid("missing fragment track defaults"));
            }
            if kind == b"vide" {
                if video_id.replace(id).is_some() {
                    return Err(Mp4Error::Unsupported("multiple fragmented video tracks"));
                }
            } else {
                if audio_id.replace(id).is_some() {
                    return Err(Mp4Error::Unsupported("multiple fragmented audio tracks"));
                }
                audio_media_start = audio_edit(track.data)?.0;
            }
        }
        if video_id.is_some() && video_id == audio_id {
            return Err(Mp4Error::Invalid("duplicate fragment track ID"));
        }
        let video = video_id
            .map(|id| Mp4VideoIndex::parse_prefix(prefix).map(|index| (id, index)))
            .transpose()?;
        let audio = if let Some(id) = audio_id {
            Some((
                id,
                Mp4AudioIndex::parse_prefix(prefix)?
                    .ok_or(Mp4Error::Invalid("missing fragmented audio track"))?,
            ))
        } else {
            None
        };
        if video.is_none() && audio.is_none() {
            return Err(Mp4Error::Invalid("no supported media tracks"));
        }
        Ok(Self {
            video,
            audio,
            defaults,
            audio_media_start,
        })
    }
}

fn fragment_u32(data: &[u8], cursor: &mut usize) -> Result<u32, Mp4Error> {
    let value = u32_at(data, *cursor)?;
    *cursor = cursor.checked_add(4).ok_or(Mp4Error::TooLarge)?;
    Ok(value)
}

fn fragment_offset(base: u64, delta: i32) -> Result<u64, Mp4Error> {
    base.checked_add_signed(i64::from(delta))
        .ok_or(Mp4Error::Invalid("fragment data offset out of range"))
}

impl Mp4Fragment {
    /// `prefix` starts at a moof header; samples refer to absolute file offsets.
    /// No mdat payload is required, and incomplete input never mutates state.
    pub fn parse_prefix(
        prefix: &[u8],
        absolute_offset: u64,
        init: &Mp4FragmentInit,
        decode_times: Mp4FragmentDecodeTimes,
    ) -> Result<Self, Mp4Error> {
        // A zero-sized box depends on EOF and cannot be committed incrementally.
        let declared = u32_at(prefix, 0)?;
        if declared == 0 {
            return Err(Mp4Error::Unsupported("open-ended movie fragment"));
        }
        let declared_size = if declared == 1 {
            u64_at(prefix, 8)?
        } else {
            u64::from(declared)
        };
        if declared_size > MAX_MOOV_BYTES {
            return Err(Mp4Error::TooLarge);
        }
        let moof = atom_at(prefix, 0)?;
        if moof.kind != *b"moof" {
            return Err(Mp4Error::Invalid("expected movie fragment"));
        }
        if moof.size as u64 > MAX_MOOV_BYTES {
            return Err(Mp4Error::TooLarge);
        }
        Self::parse_complete(moof, absolute_offset, init, decode_times).map_err(|error| {
            if error == Mp4Error::Incomplete {
                Mp4Error::Invalid("truncated movie fragment")
            } else {
                error
            }
        })
    }

    fn parse_complete(
        moof: Atom<'_>,
        absolute_offset: u64,
        init: &Mp4FragmentInit,
        decode_times: Mp4FragmentDecodeTimes,
    ) -> Result<Self, Mp4Error> {
        let moof_children = children(moof.data)?;
        let header = child(moof.data, b"mfhd")?;
        if u32_at(header.data, 0)? != 0 {
            return Err(Mp4Error::Unsupported("unsupported fragment header"));
        }
        u32_at(header.data, 4)?;
        let mut result = Self {
            consumed_bytes: moof.size,
            audio_samples: Vec::new(),
            video_samples: Vec::new(),
            decode_times,
        };
        let mut previous_data_end = absolute_offset;
        let mut first_traf = true;
        for traf in moof_children
            .into_iter()
            .filter(|atom| atom.kind == *b"traf")
        {
            let boxes = children(traf.data)?;
            if boxes.iter().filter(|atom| atom.kind == *b"tfhd").count() != 1
                || boxes.iter().filter(|atom| atom.kind == *b"tfdt").count() > 1
            {
                return Err(Mp4Error::Invalid(
                    "duplicate or missing fragment track header",
                ));
            }
            if boxes
                .iter()
                .any(|atom| matches!(&atom.kind, b"senc" | b"saiz" | b"saio"))
            {
                return Err(Mp4Error::Unsupported("encrypted movie fragment"));
            }
            let tfhd = child(traf.data, b"tfhd")?;
            let full = u32_at(tfhd.data, 0)?;
            let flags = full & 0xffffff;
            if full >> 24 != 0 || flags & !0x03003b != 0 {
                return Err(Mp4Error::Unsupported("unsupported tfhd version or flags"));
            }
            let track_id = u32_at(tfhd.data, 4)?;
            let defaults = init
                .defaults
                .iter()
                .find(|d| d.track_id == track_id)
                .ok_or(Mp4Error::Invalid("unknown fragment track"))?;
            let audio = init.audio.as_ref().is_some_and(|(id, _)| *id == track_id);
            if !audio && !init.video.as_ref().is_some_and(|(id, _)| *id == track_id) {
                return Err(Mp4Error::Unsupported("unselected fragment track"));
            }
            let mut cursor = 8;
            let base = if flags & 1 != 0 {
                let base = u64_at(tfhd.data, cursor)?;
                cursor += 8;
                base
            } else if flags & 0x020000 != 0 || first_traf {
                absolute_offset
            } else {
                previous_data_end
            };
            first_traf = false;
            let description = if flags & 2 != 0 {
                fragment_u32(tfhd.data, &mut cursor)?
            } else {
                defaults.sample_description_index
            };
            if description != 1 {
                return Err(Mp4Error::Unsupported(
                    "multiple fragment sample descriptions",
                ));
            }
            let duration = if flags & 8 != 0 {
                fragment_u32(tfhd.data, &mut cursor)?
            } else {
                defaults.duration
            };
            let size = if flags & 16 != 0 {
                fragment_u32(tfhd.data, &mut cursor)?
            } else {
                defaults.size
            };
            let sample_flags = if flags & 32 != 0 {
                fragment_u32(tfhd.data, &mut cursor)?
            } else {
                defaults.flags
            };
            if cursor != tfhd.data.len() {
                return Err(Mp4Error::Invalid("invalid tfhd length"));
            }
            let mut clock = if audio {
                result.decode_times.audio
            } else {
                result.decode_times.video
            };
            if let Some(tfdt) = boxes.iter().find(|atom| atom.kind == *b"tfdt") {
                let full = u32_at(tfdt.data, 0)?;
                if full & 0xffffff != 0 {
                    return Err(Mp4Error::Invalid("invalid tfdt flags"));
                }
                clock = match full >> 24 {
                    0 if tfdt.data.len() == 8 => u64::from(u32_at(tfdt.data, 4)?),
                    1 if tfdt.data.len() == 12 => u64_at(tfdt.data, 4)?,
                    0 | 1 => return Err(Mp4Error::Invalid("invalid tfdt length")),
                    _ => return Err(Mp4Error::Unsupported("unsupported tfdt version")),
                };
            }
            let mut data_end = base;
            let mut traf_data_end: Option<u64> = None;
            for trun in boxes.iter().filter(|atom| atom.kind == *b"trun") {
                let full = u32_at(trun.data, 0)?;
                let version = full >> 24;
                let run_flags = full & 0xffffff;
                if version > 1 || run_flags & !0x000f05 != 0 {
                    return Err(Mp4Error::Unsupported("unsupported trun version or flags"));
                }
                if run_flags & 4 != 0 && run_flags & 0x400 != 0 {
                    return Err(Mp4Error::Invalid("conflicting fragment sample flags"));
                }
                let count = u32_at(trun.data, 4)? as usize;
                let total = result
                    .audio_samples
                    .len()
                    .checked_add(result.video_samples.len())
                    .and_then(|n| n.checked_add(count))
                    .ok_or(Mp4Error::TooLarge)?;
                if total > MAX_SAMPLES {
                    return Err(Mp4Error::TooLarge);
                }
                if flags & 0x010000 != 0 && count != 0 {
                    return Err(Mp4Error::Invalid("samples in empty fragment"));
                }
                let mut cursor = 8;
                if run_flags & 1 != 0 {
                    data_end = fragment_offset(base, fragment_u32(trun.data, &mut cursor)? as i32)?;
                }
                let first_flags = if run_flags & 4 != 0 {
                    Some(fragment_u32(trun.data, &mut cursor)?)
                } else {
                    None
                };
                let fields = (run_flags & 0xf00).count_ones() as usize;
                let required = count
                    .checked_mul(fields * 4)
                    .and_then(|n| n.checked_add(cursor))
                    .ok_or(Mp4Error::TooLarge)?;
                if required > trun.data.len() {
                    return Err(Mp4Error::Incomplete);
                }
                if required != trun.data.len() {
                    return Err(Mp4Error::Invalid("invalid trun length"));
                }
                for index in 0..count {
                    let duration = if run_flags & 0x100 != 0 {
                        fragment_u32(trun.data, &mut cursor)?
                    } else {
                        duration
                    };
                    let size = if run_flags & 0x200 != 0 {
                        fragment_u32(trun.data, &mut cursor)?
                    } else {
                        size
                    };
                    let sample_flags = if run_flags & 0x400 != 0 {
                        fragment_u32(trun.data, &mut cursor)?
                    } else if index == 0 {
                        first_flags.unwrap_or(sample_flags)
                    } else {
                        sample_flags
                    };
                    let composition = if run_flags & 0x800 != 0 {
                        let raw = fragment_u32(trun.data, &mut cursor)?;
                        if version == 1 {
                            i64::from(raw as i32)
                        } else {
                            i64::from(raw)
                        }
                    } else {
                        0
                    };
                    if audio && size > MAX_AAC_SAMPLE_BYTES {
                        return Err(Mp4Error::TooLarge);
                    }
                    let presentation_time = i64::try_from(clock)
                        .map_err(|_| Mp4Error::TooLarge)?
                        .checked_add(composition)
                        .and_then(|time| {
                            time.checked_sub(if audio { init.audio_media_start } else { 0 })
                        })
                        .ok_or(Mp4Error::TooLarge)?;
                    let sample = Sample {
                        offset: data_end,
                        size,
                        decode_time: clock,
                        presentation_time,
                        keyframe: sample_flags & 0x00010000 == 0,
                    };
                    if audio {
                        result.audio_samples.push(sample);
                    } else {
                        result.video_samples.push(sample);
                    }
                    data_end = data_end
                        .checked_add(u64::from(size))
                        .ok_or(Mp4Error::TooLarge)?;
                    clock = clock
                        .checked_add(u64::from(duration))
                        .ok_or(Mp4Error::TooLarge)?;
                }
                if count != 0 {
                    traf_data_end = Some(traf_data_end.map_or(data_end, |end| end.max(data_end)));
                }
            }
            previous_data_end = traf_data_end.unwrap_or(base);
            if audio {
                result.decode_times.audio = clock;
            } else {
                result.decode_times.video = clock;
            }
        }
        Ok(result)
    }
}

impl Mp4AudioIndex {
    pub fn parse_prefix(prefix: &[u8]) -> Result<Option<Self>, Mp4Error> {
        let moov = movie_box(prefix)?;
        for track in children(moov.data)?
            .into_iter()
            .filter(|atom| atom.kind == *b"trak")
        {
            let media = child(track.data, b"mdia")?;
            if child(media.data, b"hdlr")?.data.get(8..12) != Some(&b"soun"[..]) {
                continue;
            }
            let mdhd = child(media.data, b"mdhd")?;
            let timescale = u32_at(
                mdhd.data,
                if mdhd.data.first() == Some(&1) {
                    20
                } else {
                    12
                },
            )?;
            if timescale == 0 {
                return Err(Mp4Error::Invalid("zero audio timescale"));
            }
            let stbl = path(media.data, &[*b"minf", *b"stbl"])?;
            let stsd = child(stbl.data, b"stsd")?;
            if u32_at(stsd.data, 4)? != 1 {
                return Err(Mp4Error::Unsupported("multiple audio sample descriptions"));
            }
            let entry = atom_at(stsd.data, 8)?;
            if entry.kind != *b"mp4a" {
                return Err(Mp4Error::Unsupported("unsupported audio sample entry"));
            }
            if entry.data.len() < 28 {
                return Err(Mp4Error::Incomplete);
            }
            if entry.data[8..10] != [0, 0] {
                return Err(Mp4Error::Unsupported("versioned audio sample entry"));
            }
            let esds = child(&entry.data[28..], b"esds")?;
            let audio_specific_config = aac_decoder_config(esds.data)?.to_vec();
            let fragmented = children(moov.data)?
                .iter()
                .any(|atom| atom.kind == *b"mvex");
            let (mut samples, duration_ticks) = parse_samples(stbl.data, fragmented)?;
            if samples
                .iter()
                .any(|sample| sample.size > MAX_AAC_SAMPLE_BYTES)
            {
                return Err(Mp4Error::TooLarge);
            }
            let (media_start, edit_duration) = audio_edit(track.data)?;
            for sample in &mut samples {
                sample.presentation_time = sample
                    .presentation_time
                    .checked_sub(media_start)
                    .ok_or(Mp4Error::TooLarge)?;
            }
            let duration_ticks = if let Some(duration) = edit_duration {
                let header = child(moov.data, b"mvhd")?;
                let movie_timescale = u32_at(
                    header.data,
                    if header.data.first() == Some(&1) {
                        20
                    } else {
                        12
                    },
                )?;
                if movie_timescale == 0 {
                    return Err(Mp4Error::Invalid("zero movie timescale"));
                }
                u64::try_from(
                    u128::from(duration) * u128::from(timescale) / u128::from(movie_timescale),
                )
                .map_err(|_| Mp4Error::TooLarge)?
            } else {
                duration_ticks.saturating_sub(media_start as u64)
            };
            return Ok(Some(Self {
                timescale,
                duration_ticks,
                audio_specific_config,
                samples,
            }));
        }
        Ok(None)
    }
}

fn audio_edit(track: &[u8]) -> Result<(i64, Option<u64>), Mp4Error> {
    let Some(edits) = children(track)?
        .into_iter()
        .find(|atom| atom.kind == *b"edts")
    else {
        return Ok((0, None));
    };
    let edit = child(edits.data, b"elst")?;
    if u32_at(edit.data, 4)? != 1 {
        return Err(Mp4Error::Unsupported("multiple audio timeline edits"));
    }
    let (start, duration, rate_offset) = match edit.data.first() {
        Some(0) => (
            i64::from(u32_at(edit.data, 12)? as i32),
            u64::from(u32_at(edit.data, 8)?),
            16,
        ),
        Some(1) => (u64_at(edit.data, 16)? as i64, u64_at(edit.data, 8)?, 24),
        _ => return Err(Mp4Error::Invalid("invalid edit list version")),
    };
    if start < 0 || u32_at(edit.data, rate_offset)? != 0x0001_0000 {
        return Err(Mp4Error::Unsupported("empty or rate-adjusted audio edit"));
    }
    Ok((start, Some(duration)))
}

fn descriptor(data: &[u8]) -> Result<(u8, &[u8]), Mp4Error> {
    let tag = *data.first().ok_or(Mp4Error::Incomplete)?;
    let mut size = 0usize;
    for index in 1..=4 {
        let byte = *data.get(index).ok_or(Mp4Error::Incomplete)?;
        size = (size << 7) | usize::from(byte & 127);
        if byte & 128 == 0 {
            let start = index + 1;
            return Ok((
                tag,
                data.get(start..start + size).ok_or(Mp4Error::Incomplete)?,
            ));
        }
    }
    Err(Mp4Error::Invalid("oversized descriptor length"))
}

fn aac_decoder_config(esds: &[u8]) -> Result<&[u8], Mp4Error> {
    let (tag, es) = descriptor(esds.get(4..).ok_or(Mp4Error::Incomplete)?)?;
    if tag != 3 {
        return Err(Mp4Error::Invalid("missing ES descriptor"));
    }
    let flags = *es.get(2).ok_or(Mp4Error::Incomplete)?;
    let mut offset = 3;
    if flags & 128 != 0 {
        offset += 2;
    }
    if flags & 64 != 0 {
        offset += 1 + usize::from(*es.get(offset).ok_or(Mp4Error::Incomplete)?);
    }
    if flags & 32 != 0 {
        offset += 2;
    }
    let (tag, decoder) = descriptor(es.get(offset..).ok_or(Mp4Error::Incomplete)?)?;
    if tag != 4 {
        return Err(Mp4Error::Invalid("missing decoder descriptor"));
    }
    let object = *decoder.first().ok_or(Mp4Error::Incomplete)?;
    if ![0x40, 0x66, 0x67, 0x68].contains(&object) {
        return Err(Mp4Error::Unsupported("audio is not MPEG AAC"));
    }
    if decoder.get(1).map(|byte| byte >> 2) != Some(5) {
        return Err(Mp4Error::Invalid("decoder is not an audio stream"));
    }
    let (tag, config) = descriptor(decoder.get(13..).ok_or(Mp4Error::Incomplete)?)?;
    if tag != 5 || config.is_empty() {
        return Err(Mp4Error::Invalid("missing AudioSpecificConfig"));
    }
    Ok(config)
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
    pub fn parse_prefix(prefix: &[u8]) -> Result<Self, Mp4Error> {
        let index = Mp4VideoIndex::parse_prefix(prefix)?;
        let Mp4VideoCodec::Avc(config) = index.codec else {
            return Err(Mp4Error::Unsupported("video sample entry is not avc1"));
        };
        Ok(Self {
            timescale: index.timescale,
            duration_ticks: index.duration_ticks,
            config,
            samples: index.samples,
        })
    }
}

impl Mp4VideoIndex {
    /// Parse a complete `moov` from a file prefix. A trailing `moov` requires
    /// the preceding `mdat` to be present; sample offsets refer to the full file.
    pub fn parse_prefix(prefix: &[u8]) -> Result<Self, Mp4Error> {
        let moov = movie_box(prefix)?;

        for track in children(moov.data)?
            .into_iter()
            .filter(|atom| atom.kind == *b"trak")
        {
            let media = child(track.data, b"mdia")?;
            let handler = child(media.data, b"hdlr")?;
            if handler.data.get(8..12) != Some(&b"vide"[..]) {
                continue;
            }
            let fragmented = children(moov.data)?
                .iter()
                .any(|atom| atom.kind == *b"mvex");
            return Self::parse_video_track(media.data, fragmented);
        }
        Err(Mp4Error::Invalid("no video track"))
    }

    fn parse_video_track(media: &[u8], fragmented: bool) -> Result<Self, Mp4Error> {
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
        if entry.data.len() < 78 {
            return Err(Mp4Error::Invalid("short visual sample entry"));
        }
        let width = u32::from(u16::from_be_bytes(entry.data[24..26].try_into().unwrap()));
        let height = u32::from(u16::from_be_bytes(entry.data[26..28].try_into().unwrap()));
        let codec = if entry.kind == *b"avc1" {
            let avcc = child(&entry.data[78..], b"avcC")?;
            Mp4VideoCodec::Avc(AvcConfig::parse(avcc.data)?)
        } else if entry.kind == *b"vp08" {
            Mp4VideoCodec::Vp8
        } else if entry.kind == *b"vp09" {
            Mp4VideoCodec::Vp9
        } else {
            return Err(Mp4Error::Unsupported("unsupported video sample entry"));
        };

        let (samples, clock) = parse_samples(stbl.data, fragmented)?;
        Ok(Self {
            timescale,
            duration_ticks: clock,
            codec,
            width,
            height,
            samples,
        })
    }
}

fn movie_box(prefix: &[u8]) -> Result<Atom<'_>, Mp4Error> {
    let mut offset = 0;
    loop {
        let header = prefix.get(offset..offset + 8).ok_or(Mp4Error::Incomplete)?;
        if &header[4..8] == b"moov" {
            let size = match u32_at(prefix, offset)? {
                0 => (prefix.len() - offset) as u64,
                1 => u64_at(prefix, offset + 8)?,
                size => u64::from(size),
            };
            if size > MAX_MOOV_BYTES {
                return Err(Mp4Error::TooLarge);
            }
            return atom_at(prefix, offset);
        }
        offset = offset
            .checked_add(atom_at(prefix, offset)?.size)
            .ok_or(Mp4Error::TooLarge)?;
    }
}

fn parse_samples(table: &[u8], fragmented: bool) -> Result<(Vec<Sample>, u64), Mp4Error> {
    let stsz = child(table, b"stsz")?;
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

    let offsets = if let Ok(stco) = child(table, b"stco") {
        let (count, _) = table_u32(stco.data, 4)?;
        (0..count)
            .map(|n| u64::from(u32_at(stco.data, 8 + n * 4).unwrap()))
            .collect::<Vec<_>>()
    } else {
        let co64 = child(table, b"co64")?;
        let (count, _) = table_u32(co64.data, 8)?;
        (0..count)
            .map(|n| u64_at(co64.data, 8 + n * 8).unwrap())
            .collect::<Vec<_>>()
    };
    let stsc = child(table, b"stsc")?;
    let (stsc_count, _) = table_u32(stsc.data, 12)?;
    if fragmented && sample_count == 0 && stsc_count == 0 && offsets.is_empty() {
        let stts = child(table, b"stts")?;
        if table_u32(stts.data, 8)?.0 != 0 {
            return Err(Mp4Error::Invalid("timing count mismatch"));
        }
        for atom in children(table)? {
            if (atom.kind == *b"ctts" && table_u32(atom.data, 8)?.0 != 0)
                || (atom.kind == *b"stss" && table_u32(atom.data, 4)?.0 != 0)
            {
                return Err(Mp4Error::Invalid("nonempty fragment initialization table"));
            }
        }
        return Ok((Vec::new(), 0));
    }
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

    let stts = child(table, b"stts")?;
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
    if let Ok(ctts) = child(table, b"ctts") {
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
    if let Ok(stss) = child(table, b"stss") {
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
    Ok((samples, clock))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video::h264::NalStream;

    #[test]
    fn aac_descriptor_lengths_and_flags_are_bounded() {
        let mut decoder = vec![0x40, 0x15];
        decoder.resize(13, 0);
        decoder.extend([5, 2, 0x12, 0x10]);
        let mut es = vec![0, 1, 0xe0, 0, 2, 3, b'a', b'b', b'c', 0, 3];
        es.extend([4, decoder.len() as u8]);
        es.extend(decoder);
        let mut esds = vec![0; 4];
        esds.extend([3, es.len() as u8]);
        esds.extend(es);
        assert_eq!(aac_decoder_config(&esds).unwrap(), [0x12, 0x10]);
        for end in 0..esds.len() {
            assert!(aac_decoder_config(&esds[..end]).is_err());
        }
        assert_eq!(
            descriptor(&[5, 0x80, 0x80, 0x80, 2, 0x12, 0x10]).unwrap(),
            (5, &[0x12, 0x10][..])
        );
        assert!(descriptor(&[5, 0x80, 0x80, 0x80, 0x80]).is_err());
    }

    #[test]
    fn audio_priming_edit_preserves_negative_first_packet_time() {
        let mut edit = vec![0; 4];
        edit.extend(1u32.to_be_bytes());
        edit.extend(1000u32.to_be_bytes());
        edit.extend(1024u32.to_be_bytes());
        edit.extend(0x0001_0000u32.to_be_bytes());
        let track = boxed(b"edts", &boxed(b"elst", &edit));
        assert_eq!(audio_edit(&track).unwrap(), (1024, Some(1000)));
        for end in 0..track.len() {
            if end == 0 {
                continue;
            }
            assert!(audio_edit(&track[..end]).is_err());
        }
        edit[12..16].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(audio_edit(&boxed(b"edts", &boxed(b"elst", &edit))).is_err());
    }

    #[test]
    #[ignore = "set WEBMEDIA_AAC_MP4 to an AAC-LC MP4 fixture"]
    fn indexes_real_aac_access_units() {
        let bytes = std::fs::read(std::env::var("WEBMEDIA_AAC_MP4").unwrap()).unwrap();
        let audio = Mp4AudioIndex::parse_prefix(&bytes).unwrap().unwrap();
        assert_eq!(audio.timescale, 44100);
        assert_eq!(audio.audio_specific_config[0] >> 3, 2);
        assert!(!audio.samples.is_empty());
        for sample in &audio.samples {
            assert!(sample.size > 0);
            assert!(
                sample.offset.checked_add(u64::from(sample.size)).unwrap() <= bytes.len() as u64
            );
        }
        assert!(
            audio
                .samples
                .windows(2)
                .all(|pair| pair[0].decode_time < pair[1].decode_time)
        );
    }

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

    fn fragment_words(kind: &[u8; 4], words: &[u32]) -> Vec<u8> {
        boxed(
            kind,
            &words
                .iter()
                .flat_map(|word| word.to_be_bytes())
                .collect::<Vec<_>>(),
        )
    }

    fn fragment_test_init() -> Mp4FragmentInit {
        Mp4FragmentInit {
            video: Some((
                1,
                Mp4VideoIndex {
                    timescale: 1000,
                    duration_ticks: 0,
                    width: 1440,
                    height: 1080,
                    codec: Mp4VideoCodec::Avc(AvcConfig {
                        nal_length_size: 4,
                        sequence_parameters: Vec::new(),
                        picture_parameter_sets: Vec::new(),
                    }),
                    samples: Vec::new(),
                },
            )),
            audio: Some((
                2,
                Mp4AudioIndex {
                    timescale: 48000,
                    duration_ticks: 0,
                    audio_specific_config: vec![0x11, 0x90],
                    samples: Vec::new(),
                },
            )),
            defaults: vec![
                Mp4FragmentDefaults {
                    track_id: 1,
                    sample_description_index: 1,
                    duration: 10,
                    size: 4,
                    flags: 0x10000,
                },
                Mp4FragmentDefaults {
                    track_id: 2,
                    sample_description_index: 1,
                    duration: 1024,
                    size: 3,
                    flags: 0,
                },
            ],
            audio_media_start: 0,
        }
    }

    fn fragment_movie(trafs: &[Vec<u8>]) -> Vec<u8> {
        let mut contents = fragment_words(b"mfhd", &[0, 1]);
        for traf in trafs {
            contents.extend(boxed(b"traf", traf));
        }
        boxed(b"moof", &contents)
    }

    #[test]
    fn fragmented_samples_resolve_cnn_defaults_signed_offsets_and_both_tracks() {
        let mut video = fragment_words(b"tfhd", &[0x39, 1, 0, 1000, 10, 4, 0x10000]);
        video.extend(fragment_words(b"tfdt", &[0x01000000, 0, 100]));
        video.extend(fragment_words(
            b"trun",
            &[0x01000a05, 2, 200, 0, 5, (-20i32) as u32, 6, 30],
        ));
        let mut audio = fragment_words(b"tfhd", &[0x020000, 2]);
        audio.extend(fragment_words(b"tfdt", &[0, 2048]));
        audio.extend(fragment_words(b"trun", &[1, 2, 600]));
        let moof = fragment_movie(&[video, audio]);
        let mut init = fragment_test_init();
        init.audio_media_start = 1024;
        let fragment = Mp4Fragment::parse_prefix(&moof, 5000, &init, Default::default()).unwrap();
        assert_eq!(fragment.consumed_bytes, moof.len());
        assert_eq!(
            fragment.video_samples,
            vec![
                Sample {
                    offset: 1200,
                    size: 5,
                    decode_time: 100,
                    presentation_time: 80,
                    keyframe: true
                },
                Sample {
                    offset: 1205,
                    size: 6,
                    decode_time: 110,
                    presentation_time: 140,
                    keyframe: false
                },
            ]
        );
        assert_eq!(
            fragment.audio_samples,
            vec![
                Sample {
                    offset: 5600,
                    size: 3,
                    decode_time: 2048,
                    presentation_time: 1024,
                    keyframe: true
                },
                Sample {
                    offset: 5603,
                    size: 3,
                    decode_time: 3072,
                    presentation_time: 2048,
                    keyframe: true
                },
            ]
        );
        assert_eq!(
            fragment.decode_times,
            Mp4FragmentDecodeTimes {
                audio: 4096,
                video: 120
            }
        );
        for end in 0..moof.len() {
            assert_eq!(
                Mp4Fragment::parse_prefix(&moof[..end], 5000, &init, Default::default()),
                Err(Mp4Error::Incomplete),
                "prefix length {end}"
            );
        }
        assert_eq!(
            fragment,
            Mp4Fragment::parse_prefix(&moof, 5000, &init, Default::default()).unwrap()
        );
    }

    #[test]
    fn fragmented_runs_continue_offsets_and_clocks_with_implicit_traf_base() {
        let mut video = fragment_words(b"tfhd", &[0x020000, 1]);
        video.extend(fragment_words(b"trun", &[1, 1, (-8i32) as u32]));
        video.extend(fragment_words(b"trun", &[0, 2]));
        let mut audio = fragment_words(b"tfhd", &[0, 2]);
        audio.extend(fragment_words(b"trun", &[0, 1]));
        let moof = fragment_movie(&[video, audio]);
        let times = Mp4FragmentDecodeTimes {
            audio: 4096,
            video: 120,
        };
        let fragment =
            Mp4Fragment::parse_prefix(&moof, 1000, &fragment_test_init(), times).unwrap();
        assert_eq!(
            fragment
                .video_samples
                .iter()
                .map(|s| s.offset)
                .collect::<Vec<_>>(),
            [992, 996, 1000]
        );
        assert_eq!(
            fragment
                .video_samples
                .iter()
                .map(|s| s.decode_time)
                .collect::<Vec<_>>(),
            [120, 130, 140]
        );
        assert_eq!(fragment.audio_samples[0].offset, 1004);
        assert_eq!(fragment.audio_samples[0].decode_time, 4096);
        assert_eq!(
            fragment.decode_times,
            Mp4FragmentDecodeTimes {
                audio: 5120,
                video: 150
            }
        );
    }

    #[test]
    fn fragmented_version_zero_composition_is_unsigned_and_per_sample_fields_win() {
        let mut video = fragment_words(b"tfhd", &[0x020000, 1]);
        video.extend(fragment_words(b"tfdt", &[0x01000000, 1, 0]));
        video.extend(fragment_words(
            b"trun",
            &[0xf01, 2, 300, 7, 11, 0, u32::MAX, 8, 12, 0x10000, 0],
        ));
        let result = Mp4Fragment::parse_prefix(
            &fragment_movie(&[video]),
            100,
            &fragment_test_init(),
            Default::default(),
        )
        .unwrap();
        assert_eq!(result.video_samples[0].presentation_time, 0x1ffffffff);
        assert_eq!(result.video_samples[1].decode_time, 0x100000007);
        assert_eq!(result.video_samples[1].offset, 411);
        assert_eq!(result.video_samples[1].size, 12);
        assert!(!result.video_samples[1].keyframe);
        assert_eq!(result.decode_times.video, 0x10000000f);
    }

    #[test]
    fn fragmented_parser_rejects_invalid_flags_offsets_counts_and_overflows() {
        let init = fragment_test_init();
        for (tfhd, trun, offset, expected) in [
            (
                vec![0x020000, 1],
                vec![0x405, 1, 0, 0, 0],
                100,
                Mp4Error::Invalid("conflicting fragment sample flags"),
            ),
            (
                vec![0x020000, 1],
                vec![1, 1, (-101i32) as u32],
                100,
                Mp4Error::Invalid("fragment data offset out of range"),
            ),
            (
                vec![0x020000, 1],
                vec![0, MAX_SAMPLES as u32 + 1],
                100,
                Mp4Error::TooLarge,
            ),
            (
                vec![0x020000, 1],
                vec![0x200, 1],
                100,
                Mp4Error::Invalid("truncated movie fragment"),
            ),
            (
                vec![0x020002, 1, 2],
                vec![0, 1],
                100,
                Mp4Error::Unsupported("multiple fragment sample descriptions"),
            ),
            (
                vec![0x030000, 1],
                vec![0, 1],
                100,
                Mp4Error::Invalid("samples in empty fragment"),
            ),
            (
                vec![0x020000, 2],
                vec![0x200, 1, MAX_AAC_SAMPLE_BYTES + 1],
                100,
                Mp4Error::TooLarge,
            ),
            (
                vec![1, 1, u32::MAX, u32::MAX],
                vec![0, 1],
                100,
                Mp4Error::TooLarge,
            ),
        ] {
            let mut traf = fragment_words(b"tfhd", &tfhd);
            traf.extend(fragment_words(b"trun", &trun));
            assert_eq!(
                Mp4Fragment::parse_prefix(
                    &fragment_movie(&[traf]),
                    offset,
                    &init,
                    Default::default()
                ),
                Err(expected)
            );
        }
        let mut traf = fragment_words(b"tfhd", &[0x020000, 1]);
        traf.extend(fragment_words(b"tfdt", &[0x01000000, u32::MAX, u32::MAX]));
        traf.extend(fragment_words(b"trun", &[0, 1]));
        assert_eq!(
            Mp4Fragment::parse_prefix(&fragment_movie(&[traf]), 0, &init, Default::default()),
            Err(Mp4Error::TooLarge)
        );
        let huge = fragment_words(b"moof", &[]);
        let mut huge = huge;
        huge[..4].copy_from_slice(&((MAX_MOOV_BYTES + 1) as u32).to_be_bytes());
        assert_eq!(
            Mp4Fragment::parse_prefix(&huge, 0, &init, Default::default()),
            Err(Mp4Error::TooLarge)
        );
    }

    fn fragmented_test_track(id: u32, audio: bool) -> Vec<u8> {
        let mut entry = if audio { vec![0; 28] } else { vec![0; 78] };
        let kind = if audio {
            let mut decoder = vec![0x40, 0x15];
            decoder.resize(13, 0);
            decoder.extend([5, 2, 0x11, 0x90]);
            let mut es = vec![0, 1, 0, 4, decoder.len() as u8];
            es.extend(decoder);
            let mut esds = vec![0, 0, 0, 0, 3, es.len() as u8];
            esds.extend(es);
            entry.extend(boxed(b"esds", &esds));
            b"mp4a"
        } else {
            entry[24..26].copy_from_slice(&1440u16.to_be_bytes());
            entry[26..28].copy_from_slice(&1080u16.to_be_bytes());
            let avcc = [
                0x01, 0x64, 0x00, 0x20, 0xff, 0xe1, 0x00, 0x1d, 0x67, 0x64, 0x00, 0x20, 0xac, 0xd9,
                0x40, 0x50, 0x05, 0xbb, 0xff, 0x04, 0x40, 0x04, 0x41, 0x10, 0x00, 0x00, 0x03, 0x00,
                0x10, 0x00, 0x00, 0x06, 0x48, 0xf1, 0x83, 0x19, 0x60, 0x01, 0x00, 0x06, 0x68, 0xeb,
                0xe3, 0xcb, 0x22, 0xc0, 0xfd, 0xf8, 0xf8, 0x00,
            ];
            entry.extend(boxed(b"avcC", &avcc));
            b"avc1"
        };
        let mut stsd = vec![0; 4];
        stsd.extend(1u32.to_be_bytes());
        stsd.extend(boxed(kind, &entry));
        let mut stbl = boxed(b"stsd", &stsd);
        stbl.extend(boxed(b"stsz", &[0; 12]));
        for kind in [b"stts", b"stsc", b"stco"] {
            stbl.extend(boxed(kind, &[0; 8]));
        }
        let mut mdhd = vec![0; 12];
        mdhd.extend((if audio { 48000u32 } else { 1000u32 }).to_be_bytes());
        let mut hdlr = vec![0; 8];
        hdlr.extend(if audio { b"soun" } else { b"vide" });
        let mut mdia = boxed(b"mdhd", &mdhd);
        mdia.extend(boxed(b"hdlr", &hdlr));
        mdia.extend(boxed(b"minf", &boxed(b"stbl", &stbl)));
        let mut tkhd = vec![0; 12];
        tkhd.extend(id.to_be_bytes());
        let mut trak = boxed(b"tkhd", &tkhd);
        trak.extend(boxed(b"mdia", &mdia));
        boxed(b"trak", &trak)
    }
    fn fragmented_test_initialization(fragmented: bool) -> Vec<u8> {
        let mut tracks = fragmented_test_track(1, false);
        tracks.extend(fragmented_test_track(2, true));
        if fragmented {
            let mut defaults = fragment_words(b"trex", &[0, 1, 1, 10, 4, 0x10000]);
            defaults.extend(fragment_words(b"trex", &[0, 2, 1, 1024, 3, 0]));
            tracks.extend(boxed(b"mvex", &defaults));
        }
        boxed(b"moov", &tracks)
    }

    /// Parser fixture only: distinct sample bytes are not encoded access units.
    pub(crate) fn fragmented_test_file() -> Vec<u8> {
        let mut file = boxed(b"ftyp", b"isom\0\0\0\0isomiso6");
        file.extend(fragmented_test_initialization(true));
        for sequence in 0..3u32 {
            let make_moof = |data_offset| {
                let mut video = fragment_words(b"tfhd", &[0x020000, 1]);
                video.extend(fragment_words(b"tfdt", &[0, sequence * 20]));
                video.extend(fragment_words(b"trun", &[5, 2, data_offset, 0]));
                let mut audio = fragment_words(b"tfhd", &[0x020000, 2]);
                audio.extend(fragment_words(b"tfdt", &[0, sequence * 2048]));
                audio.extend(fragment_words(b"trun", &[1, 2, data_offset + 8]));
                let mut contents = fragment_words(b"mfhd", &[0, sequence + 1]);
                contents.extend(boxed(b"traf", &video));
                contents.extend(boxed(b"traf", &audio));
                boxed(b"moof", &contents)
            };
            let data_offset = make_moof(0).len() as u32 + 8;
            file.extend(make_moof(data_offset));
            let payload: Vec<_> = (0..14)
                .map(|index| (sequence * 16 + index + 1) as u8)
                .collect();
            file.extend(boxed(b"mdat", &payload));
        }
        file
    }

    #[test]
    fn fragmented_portable_fixture_has_six_distinct_samples_per_track() {
        let file = fragmented_test_file();
        let init = Mp4FragmentInit::parse_prefix(&file).unwrap();
        let mut offset = 0;
        let mut clocks = Mp4FragmentDecodeTimes::default();
        let mut counts = (0, 0);
        let mut sequence = 0u8;
        while offset < file.len() {
            let atom = atom_at(&file, offset).unwrap();
            if atom.kind == *b"moof" {
                let result =
                    Mp4Fragment::parse_prefix(&file[offset..], offset as u64, &init, clocks)
                        .unwrap();
                assert_eq!(
                    (result.video_samples.len(), result.audio_samples.len()),
                    (2, 2)
                );
                for (index, sample) in result
                    .video_samples
                    .iter()
                    .chain(&result.audio_samples)
                    .enumerate()
                {
                    let local = [0, 4, 8, 11][index];
                    let expected: Vec<_> = (0..sample.size)
                        .map(|byte| sequence * 16 + local + byte as u8 + 1)
                        .collect();
                    assert_eq!(
                        &file
                            [sample.offset as usize..sample.offset as usize + sample.size as usize],
                        expected
                    );
                }
                counts.0 += result.video_samples.len();
                counts.1 += result.audio_samples.len();
                clocks = result.decode_times;
                sequence += 1;
            }
            offset += atom.size;
        }
        assert_eq!(counts, (6, 6));
        assert_eq!(
            clocks,
            Mp4FragmentDecodeTimes {
                video: 60,
                audio: 6144
            }
        );
    }

    #[test]
    fn fragmented_empty_tables_require_mvex_and_metadata_keeps_track_defaults() {
        let classic = fragmented_test_initialization(false);
        assert_eq!(
            Mp4VideoIndex::parse_prefix(&classic),
            Err(Mp4Error::Invalid("empty chunk table"))
        );
        assert_eq!(
            Mp4AudioIndex::parse_prefix(&classic),
            Err(Mp4Error::Invalid("empty chunk table"))
        );
        assert_eq!(
            Mp4FragmentInit::parse_prefix(&classic),
            Err(Mp4Error::Unsupported("not fragmented MP4"))
        );
        let file = fragmented_test_initialization(true);
        let init = Mp4FragmentInit::parse_prefix(&file).unwrap();
        assert_eq!(init.defaults, fragment_test_init().defaults);
        assert_eq!(init.video.as_ref().unwrap().0, 1);
        assert!(matches!(
            &init.video.as_ref().unwrap().1.codec,
            Mp4VideoCodec::Avc(_)
        ));
        assert_eq!(
            (
                init.video.as_ref().unwrap().1.width,
                init.video.as_ref().unwrap().1.height
            ),
            (1440, 1080)
        );
        assert!(init.video.as_ref().unwrap().1.samples.is_empty());
        assert_eq!(
            init.audio.as_ref().unwrap().1.audio_specific_config,
            [0x11, 0x90]
        );
        for end in 0..file.len() {
            assert_eq!(
                Mp4FragmentInit::parse_prefix(&file[..end]),
                Err(Mp4Error::Incomplete),
                "prefix {end}"
            );
        }
    }

    #[test]
    #[ignore = "set WEBMEDIA_FRAGMENTED_MP4 to the local CNN fixture"]
    fn fragmented_cnn_file_indexes_both_tracks_and_payload_bounds() {
        let bytes = std::fs::read(std::env::var("WEBMEDIA_FRAGMENTED_MP4").unwrap()).unwrap();
        let init = Mp4FragmentInit::parse_prefix(&bytes).unwrap();
        assert!(matches!(
            &init.video.as_ref().unwrap().1.codec,
            Mp4VideoCodec::Avc(_)
        ));
        assert!(init.audio.is_some());
        let mut times = Mp4FragmentDecodeTimes::default();
        let mut offset = 0;
        let mut counts = (0, 0);
        let mut mdat_ranges = Vec::new();
        let mut scan = 0;
        while scan < bytes.len() {
            let atom = atom_at(&bytes, scan).unwrap();
            if atom.kind == *b"mdat" {
                mdat_ranges.push((
                    (scan + atom.size - atom.data.len()) as u64,
                    (scan + atom.size) as u64,
                ));
            }
            scan += atom.size;
        }
        while offset < bytes.len() {
            let atom = atom_at(&bytes, offset).unwrap();
            if atom.kind == *b"moof" {
                let fragment =
                    Mp4Fragment::parse_prefix(&bytes[offset..], offset as u64, &init, times)
                        .unwrap();
                for sample in fragment.audio_samples.iter().chain(&fragment.video_samples) {
                    assert!(sample.size > 0);
                    assert!(sample.offset + u64::from(sample.size) <= bytes.len() as u64);
                    assert!(
                        mdat_ranges
                            .iter()
                            .any(|(start, end)| sample.offset >= *start
                                && sample.offset + u64::from(sample.size) <= *end)
                    );
                }
                if counts == (0, 0) {
                    assert_eq!(
                        fragment.video_samples[0],
                        Sample {
                            offset: 1488,
                            size: 57148,
                            decode_time: 0,
                            presentation_time: 0,
                            keyframe: true,
                        }
                    );
                    assert_eq!(fragment.audio_samples[0].offset, 58636);
                }
                counts.0 += fragment.video_samples.len();
                counts.1 += fragment.audio_samples.len();
                times = fragment.decode_times;
            }
            offset += atom.size;
        }
        // Independently counted by ffprobe on the downloaded CNN fixture.
        assert_eq!(counts, (240, 175));
        assert!(times.video > 0 && times.audio > 0);
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
