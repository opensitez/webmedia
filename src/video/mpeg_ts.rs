//! Bounded TS188/PES demuxing for single-program HLS (RFC 8216 section 3.2).
//! Wire syntax follows ITU-T H.222.0 sections 2.4.3 and 2.4.4; no codec decoding.
//! Input starts at a transport packet boundary. Timestamps remain modulo 2^33.

use super::backend::MediaDecodeError;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const TS_PACKET_BYTES: usize = 188;
pub const TIMESTAMP_TIMESCALE: u32 = 90_000;
pub const MAX_PUSH_BYTES: usize = 256 * 1024;
pub const MAX_PES_BYTES: usize = 4 * 1024 * 1024;
const MAX_BUFFERED_PES_BYTES: usize = 16 * 1024 * 1024;
const MAX_STREAMS: usize = 16;
const MAX_TABLE_BYTES: usize = 64 * 1024;
const MAX_OUTPUT_BYTES: usize = 16 * 1024 * 1024;
const MAX_OUTPUT_PACKETS: usize = 2048;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamKind {
    H264,
    Aac,
}

type ElementaryCodec = StreamKind;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElementaryPacket {
    pub pid: u16,
    pub kind: StreamKind,
    /// Raw 33-bit MPEG timestamps in 90-kHz units, not seconds or unwrapped time.
    pub pts: Option<u64>,
    pub dts: Option<u64>,
    /// A transport loss, signalled discontinuity, or changed stream map preceded this PES.
    pub discontinuity: bool,
    /// Complete PES elementary payload: Annex B H.264 or ADTS AAC, possibly multiple frames.
    pub data: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TsError {
    Invalid(&'static str),
    Unsupported(&'static str),
    TooLarge,
    Truncated,
}

impl fmt::Display for TsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(reason) => write!(f, "invalid MPEG-TS: {reason}"),
            Self::Unsupported(reason) => write!(f, "unsupported MPEG-TS: {reason}"),
            Self::TooLarge => f.write_str("MPEG-TS security bound exceeded"),
            Self::Truncated => f.write_str("truncated MPEG-TS unit"),
        }
    }
}

impl std::error::Error for TsError {}

impl From<TsError> for MediaDecodeError {
    fn from(error: TsError) -> Self {
        match error {
            TsError::Unsupported(_) => Self::Unsupported,
            _ => Self::InvalidData(error.to_string()),
        }
    }
}

/// HLS-facing API. Packet framing and validated maps survive segment boundaries.
#[derive(Default)]
pub struct TransportStream {
    demux: MpegTsDemux,
}

impl TransportStream {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<ElementaryPacket>, MediaDecodeError> {
        self.demux.push(bytes).map_err(Into::into)
    }
    pub fn finish_segment(&mut self) -> Result<Vec<ElementaryPacket>, MediaDecodeError> {
        self.demux.finish().map_err(Into::into)
    }
    pub fn reset(&mut self) {
        self.demux.reset();
    }
    pub fn streams(&self) -> impl Iterator<Item = (u16, StreamKind)> + '_ {
        self.demux.streams()
    }
    pub fn program_number(&self) -> Option<u16> {
        self.demux.program_number()
    }
}

#[derive(Default)]
struct Continuity {
    last: Option<[u8; TS_PACKET_BYTES]>,
}

impl Continuity {
    // Only payload packets advance CC. Identical retransmissions (including
    // legal PCR-only variation) must not duplicate elementary bytes.
    fn payload(&mut self, packet: &[u8; TS_PACKET_BYTES]) -> (bool, bool) {
        let mut lost = false;
        if let Some(last) = &self.last {
            let old = last[3] & 15;
            let new = packet[3] & 15;
            if old == new && duplicate_packet(last, packet) {
                return (true, false);
            }
            lost = new != ((old + 1) & 15);
        }
        self.last = Some(*packet);
        (false, lost)
    }
}

fn duplicate_packet(a: &[u8; 188], b: &[u8; 188]) -> bool {
    if a == b {
        return true;
    }
    a[..6] == b[..6] && a[3] & 0x20 != 0 && a[4] >= 7 && a[5] & 0x10 != 0 && a[12..] == b[12..]
}

#[derive(Default)]
struct Psi {
    bytes: Vec<u8>,
    continuity: Continuity,
}

impl Psi {
    fn feed(&mut self, payload: &[u8], start: bool) -> Result<Vec<Vec<u8>>, TsError> {
        let mut out = Vec::new();
        let mut rest = payload;
        if start {
            let (&pointer, following) = payload
                .split_first()
                .ok_or(TsError::Invalid("missing PSI pointer"))?;
            let pointer = usize::from(pointer);
            if pointer > following.len() {
                return Err(TsError::Invalid("PSI pointer out of bounds"));
            }
            if !self.bytes.is_empty() {
                let used = self.append(&following[..pointer], &mut out)?;
                if !self.bytes.is_empty() || following[used..pointer].iter().any(|b| *b != 0xff) {
                    return Err(TsError::Invalid("unfinished PSI before pointer boundary"));
                }
            }
            rest = &following[pointer..];
        } else if self.bytes.is_empty() {
            // Join mid-section only after a PUSI establishes a section boundary.
            return Ok(out);
        }
        while !rest.is_empty() {
            if self.bytes.is_empty() && rest[0] == 0xff {
                if rest.iter().any(|b| *b != 0xff) {
                    return Err(TsError::Invalid("non-stuffing after PSI stuffing"));
                }
                break;
            }
            let used = self.append(rest, &mut out)?;
            rest = &rest[used..];
            if !self.bytes.is_empty() {
                break;
            }
        }
        Ok(out)
    }

    fn append(&mut self, bytes: &[u8], out: &mut Vec<Vec<u8>>) -> Result<usize, TsError> {
        let mut used = 0;
        if self.bytes.len() < 3 {
            let n = (3 - self.bytes.len()).min(bytes.len());
            self.bytes.extend_from_slice(&bytes[..n]);
            used += n;
        }
        if self.bytes.len() < 3 {
            return Ok(used);
        }
        if self.bytes[1] & 0xf0 != 0xb0 {
            return Err(TsError::Invalid("invalid PSI syntax/reserved bits"));
        }
        let length = usize::from(u16::from_be_bytes([self.bytes[1] & 15, self.bytes[2]]));
        if !(9..=1021).contains(&length) {
            return Err(TsError::Invalid("invalid PSI section length"));
        }
        let n = (3 + length - self.bytes.len()).min(bytes.len() - used);
        self.bytes.extend_from_slice(&bytes[used..used + n]);
        used += n;
        if self.bytes.len() == 3 + length {
            if crc32_mpeg(&self.bytes) != 0 {
                return Err(TsError::Invalid("PSI CRC mismatch"));
            }
            out.push(std::mem::take(&mut self.bytes));
        }
        Ok(used)
    }
}

struct Table {
    extension: u16,
    version: u8,
    last: u8,
    bytes: usize,
    sections: Vec<Option<Vec<u8>>>,
}

impl Table {
    fn accept(slot: &mut Option<Self>, section: Vec<u8>) -> Result<Option<Vec<Vec<u8>>>, TsError> {
        if section[5] & 0xc0 != 0xc0 || section[6] > section[7] {
            return Err(TsError::Invalid("invalid PSI version/section numbering"));
        }
        if section[5] & 1 == 0 {
            return Ok(None);
        }
        let extension = u16::from_be_bytes([section[3], section[4]]);
        let version = (section[5] >> 1) & 31;
        let last = section[7];
        if slot
            .as_ref()
            .is_none_or(|t| t.extension != extension || t.version != version)
        {
            *slot = Some(Self {
                extension,
                version,
                last,
                bytes: 0,
                sections: vec![None; usize::from(last) + 1],
            });
        }
        let table = slot.as_mut().unwrap();
        if table.last != last {
            return Err(TsError::Invalid("inconsistent PSI last_section_number"));
        }
        let index = usize::from(section[6]);
        if let Some(prior) = &table.sections[index] {
            if prior != &section {
                return Err(TsError::Invalid("conflicting repeated PSI section"));
            }
        } else {
            if section.len() > MAX_TABLE_BYTES.saturating_sub(table.bytes) {
                return Err(TsError::TooLarge);
            }
            table.bytes += section.len();
            table.sections[index] = Some(section);
        }
        if table.sections.iter().all(Option::is_some) {
            Ok(Some(
                slot.take()
                    .unwrap()
                    .sections
                    .into_iter()
                    .map(Option::unwrap)
                    .collect(),
            ))
        } else {
            Ok(None)
        }
    }
}

#[derive(Clone, Copy)]
struct PesHeader {
    offset: usize,
    length: Option<usize>,
    pts: Option<u64>,
    dts: Option<u64>,
}

struct Pes {
    bytes: Vec<u8>,
    header: Option<PesHeader>,
    discontinuity: bool,
}

struct Stream {
    codec: ElementaryCodec,
    continuity: Continuity,
    pes: Option<Pes>,
    discontinuity: bool,
}

impl Stream {
    fn lose(&mut self) {
        self.pes = None;
        self.discontinuity = true;
    }

    fn emit(&mut self, pid: u16, out: &mut Vec<ElementaryPacket>) -> Result<(), TsError> {
        let pes = self.pes.take().unwrap();
        let header = pes.header.ok_or(TsError::Truncated)?;
        if header
            .length
            .is_some_and(|length| length != pes.bytes.len())
        {
            return Err(TsError::Truncated);
        }
        let data = pes.bytes[header.offset..].to_vec();
        if out.len() >= MAX_OUTPUT_PACKETS
            || data.len() > MAX_OUTPUT_BYTES.saturating_sub(out.iter().map(|p| p.data.len()).sum())
        {
            return Err(TsError::TooLarge);
        }
        out.push(ElementaryPacket {
            pid,
            kind: self.codec,
            pts: header.pts,
            dts: header.dts,
            discontinuity: pes.discontinuity,
            data,
        });
        Ok(())
    }

    fn feed(
        &mut self,
        pid: u16,
        payload: &[u8],
        start: bool,
        out: &mut Vec<ElementaryPacket>,
    ) -> Result<(), TsError> {
        if start {
            if self
                .pes
                .as_ref()
                .is_some_and(|p| p.header.is_some_and(|h| h.length.is_none()))
            {
                self.emit(pid, out)?;
            } else if self.pes.is_some() {
                self.lose();
            }
            self.pes = Some(Pes {
                bytes: Vec::new(),
                header: None,
                discontinuity: std::mem::take(&mut self.discontinuity),
            });
        }
        let Some(pes) = &mut self.pes else {
            return Ok(());
        };
        if payload.len() > MAX_PES_BYTES.saturating_sub(pes.bytes.len()) {
            return Err(TsError::TooLarge);
        }
        pes.bytes.extend_from_slice(payload);
        if pes.header.is_none() {
            pes.header = parse_pes_header(&pes.bytes, self.codec)?;
        }
        if let Some(length) = pes.header.and_then(|h| h.length) {
            if pes.bytes.len() >= length {
                if pes.bytes[length..].iter().any(|b| *b != 0xff) {
                    return Err(TsError::Invalid("non-stuffing beyond PES length"));
                }
                pes.bytes.truncate(length);
                self.emit(pid, out)?;
            }
        }
        Ok(())
    }
}

/// On any returned error all parser state is reset; no partial result is returned.
/// Call `push` with at most MAX_PUSH_BYTES per call and drain its returned packets.
#[derive(Default)]
pub struct MpegTsDemux {
    partial: Vec<u8>,
    pat: Psi,
    pmt: Psi,
    pat_table: Option<Table>,
    pmt_table: Option<Table>,
    program: Option<(u16, u16, u16)>, // transport_stream_id, program_number, PMT PID
    streams: BTreeMap<u16, Stream>,
}

impl MpegTsDemux {
    pub fn new() -> Self {
        Self::default()
    }

    /// For EXT-X-DISCONTINUITY, seeking, or an entirely new transport stream.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn program_number(&self) -> Option<u16> {
        self.program.map(|p| p.1)
    }

    pub fn streams(&self) -> impl Iterator<Item = (u16, ElementaryCodec)> + '_ {
        self.streams
            .iter()
            .map(|(&pid, stream)| (pid, stream.codec))
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<ElementaryPacket>, TsError> {
        let result = self.push_inner(bytes);
        if result.is_err() {
            self.reset();
        }
        result
    }

    fn push_inner(&mut self, mut bytes: &[u8]) -> Result<Vec<ElementaryPacket>, TsError> {
        if bytes.len() > MAX_PUSH_BYTES {
            return Err(TsError::TooLarge);
        }
        let mut out = Vec::new();
        if !self.partial.is_empty() {
            let n = (TS_PACKET_BYTES - self.partial.len()).min(bytes.len());
            self.partial.extend_from_slice(&bytes[..n]);
            bytes = &bytes[n..];
            if self.partial.len() == TS_PACKET_BYTES {
                let packet: [u8; TS_PACKET_BYTES] = self.partial.as_slice().try_into().unwrap();
                self.partial.clear();
                self.packet(&packet, &mut out)?;
            }
        }
        while bytes.len() >= TS_PACKET_BYTES {
            self.packet(bytes[..TS_PACKET_BYTES].try_into().unwrap(), &mut out)?;
            bytes = &bytes[TS_PACKET_BYTES..];
        }
        self.partial.extend_from_slice(bytes);
        Ok(out)
    }

    /// Close a segment/end-of-input. Flushes length-zero video PES, rejects
    /// incomplete TS/PSI/bounded PES, and retains validated maps and continuity.
    pub fn finish(&mut self) -> Result<Vec<ElementaryPacket>, TsError> {
        let result = self.finish_inner();
        if result.is_err() {
            self.reset();
        }
        result
    }

    fn finish_inner(&mut self) -> Result<Vec<ElementaryPacket>, TsError> {
        if !self.partial.is_empty()
            || !self.pat.bytes.is_empty()
            || !self.pmt.bytes.is_empty()
            || self.pat_table.is_some()
            || self.pmt_table.is_some()
        {
            return Err(TsError::Truncated);
        }
        let mut out = Vec::new();
        for (&pid, stream) in &mut self.streams {
            if stream.pes.is_some() {
                stream.emit(pid, &mut out)?;
            }
        }
        Ok(out)
    }

    fn lose(&mut self, pid: u16) {
        if pid == 0 {
            self.pat = Psi::default();
            self.pat_table = None;
        } else if self.program.is_some_and(|p| p.2 == pid) {
            self.pmt = Psi::default();
            self.pmt_table = None;
        } else if let Some(stream) = self.streams.get_mut(&pid) {
            stream.lose();
            stream.continuity = Continuity::default();
        }
    }

    fn packet(
        &mut self,
        packet: &[u8; 188],
        out: &mut Vec<ElementaryPacket>,
    ) -> Result<(), TsError> {
        if packet[0] != 0x47 {
            return Err(TsError::Invalid("TS188 sync byte"));
        }
        let pid = u16::from_be_bytes([packet[1] & 31, packet[2]]);
        let control = (packet[3] >> 4) & 3;
        if control == 0 {
            return Err(TsError::Invalid("reserved adaptation_field_control"));
        }
        let (offset, discontinuity) = adaptation(packet, control)?;
        if pid == 0x1fff {
            return Ok(());
        }
        if packet[1] & 0x80 != 0 {
            self.lose(pid);
            return Ok(());
        }
        if packet[3] & 0xc0 != 0 {
            return Err(TsError::Unsupported("scrambled transport payload"));
        }
        if discontinuity {
            self.lose(pid);
        }
        if control & 1 == 0 {
            let last = if pid == 0 {
                self.pat.continuity.last.as_ref()
            } else if self.program.is_some_and(|p| p.2 == pid) {
                self.pmt.continuity.last.as_ref()
            } else {
                self.streams
                    .get(&pid)
                    .and_then(|s| s.continuity.last.as_ref())
            };
            if last.is_some_and(|last| last[3] & 15 != packet[3] & 15) {
                self.lose(pid);
            }
            return Ok(());
        }
        let start = packet[1] & 0x40 != 0;
        let payload = &packet[offset..];
        if pid == 0 || self.program.is_some_and(|p| p.2 == pid) {
            let psi = if pid == 0 {
                &mut self.pat
            } else {
                &mut self.pmt
            };
            let (duplicate, lost) = psi.continuity.payload(packet);
            if duplicate {
                return Ok(());
            }
            if lost {
                psi.bytes.clear();
                if pid == 0 {
                    self.pat_table = None;
                } else {
                    self.pmt_table = None;
                }
            }
            let sections = psi.feed(payload, start)?;
            for section in sections {
                if pid == 0 {
                    self.pat_section(section)?;
                } else {
                    self.pmt_section(section)?;
                }
            }
        } else if let Some(stream) = self.streams.get_mut(&pid) {
            let (duplicate, lost) = stream.continuity.payload(packet);
            if duplicate {
                return Ok(());
            }
            if lost {
                stream.lose();
            }
            stream.feed(pid, payload, start, out)?;
            if self
                .streams
                .values()
                .filter_map(|s| s.pes.as_ref())
                .map(|p| p.bytes.len())
                .sum::<usize>()
                > MAX_BUFFERED_PES_BYTES
            {
                return Err(TsError::TooLarge);
            }
        }
        Ok(())
    }

    fn pat_section(&mut self, section: Vec<u8>) -> Result<(), TsError> {
        if section[0] != 0 || (section.len() - 12) % 4 != 0 {
            return Err(TsError::Invalid("invalid PAT layout"));
        }
        let Some(sections) = Table::accept(&mut self.pat_table, section)? else {
            return Ok(());
        };
        let transport = u16::from_be_bytes([sections[0][3], sections[0][4]]);
        let mut programs = BTreeMap::new();
        for section in sections {
            for entry in section[8..section.len() - 4].chunks_exact(4) {
                let number = u16::from_be_bytes([entry[0], entry[1]]);
                let pid = u16::from_be_bytes([entry[2] & 31, entry[3]]);
                if entry[2] & 0xe0 != 0xe0
                    || !(0x10..0x1fff).contains(&pid)
                    || programs.insert(number, pid).is_some()
                {
                    return Err(TsError::Invalid("invalid/duplicate PAT program"));
                }
            }
        }
        programs.remove(&0); // network PID, not a program
        if programs.len() > 1 {
            return Err(TsError::Unsupported("multiple HLS transport programs"));
        }
        let program = programs
            .into_iter()
            .next()
            .map(|(number, pid)| (transport, number, pid));
        if self.program != program {
            self.program = program;
            self.pmt = Psi::default();
            self.pmt_table = None;
            self.streams.clear();
        }
        Ok(())
    }

    fn pmt_section(&mut self, section: Vec<u8>) -> Result<(), TsError> {
        let (_, program, pmt_pid) = self.program.ok_or(TsError::Invalid("PMT without PAT"))?;
        if section[0] != 2
            || section.len() < 16
            || section[6] != 0
            || section[7] != 0
            || u16::from_be_bytes([section[3], section[4]]) != program
        {
            return Err(TsError::Invalid("invalid PMT identity/section numbering"));
        }
        let Some(mut sections) = Table::accept(&mut self.pmt_table, section)? else {
            return Ok(());
        };
        let section = sections.remove(0);
        let end = section.len() - 4;
        if section[8] & 0xe0 != 0xe0 || section[10] & 0xf0 != 0xf0 {
            return Err(TsError::Invalid("PMT reserved bits"));
        }
        let pcr_pid = u16::from_be_bytes([section[8] & 31, section[9]]);
        if pcr_pid < 0x10 {
            return Err(TsError::Invalid("reserved PCR PID"));
        }
        let info = usize::from(u16::from_be_bytes([section[10] & 15, section[11]]));
        let mut at = 12 + info;
        if at > end {
            return Err(TsError::Invalid("PMT program descriptors out of bounds"));
        }
        descriptors(&section[12..at])?;
        let mut streams = BTreeMap::new();
        let mut pids = BTreeSet::new();
        while at < end {
            if end - at < 5 {
                return Err(TsError::Invalid("truncated PMT stream entry"));
            }
            let kind = section[at];
            let pid = u16::from_be_bytes([section[at + 1] & 31, section[at + 2]]);
            if section[at + 1] & 0xe0 != 0xe0
                || section[at + 3] & 0xf0 != 0xf0
                || !(0x10..0x1fff).contains(&pid)
                || pid == pmt_pid
                || !pids.insert(pid)
            {
                return Err(TsError::Invalid("invalid/duplicate PMT elementary PID"));
            }
            let info = usize::from(u16::from_be_bytes([section[at + 3] & 15, section[at + 4]]));
            let next = at + 5 + info;
            if next > end {
                return Err(TsError::Invalid("PMT stream descriptors out of bounds"));
            }
            descriptors(&section[at + 5..next])?;
            let codec = match kind {
                0x1b => Some(ElementaryCodec::H264),
                0x0f => Some(ElementaryCodec::Aac),
                _ => None,
            };
            if let Some(codec) = codec {
                if streams.len() == MAX_STREAMS {
                    return Err(TsError::TooLarge);
                }
                streams.insert(pid, codec);
            }
            at = next;
        }
        if !self
            .streams
            .iter()
            .map(|(&pid, s)| (pid, s.codec))
            .eq(streams.iter().map(|(&pid, &codec)| (pid, codec)))
        {
            self.streams = streams
                .into_iter()
                .map(|(pid, codec)| {
                    (
                        pid,
                        Stream {
                            codec,
                            continuity: Continuity::default(),
                            pes: None,
                            discontinuity: true,
                        },
                    )
                })
                .collect();
        }
        Ok(())
    }
}

fn descriptors(mut bytes: &[u8]) -> Result<(), TsError> {
    while !bytes.is_empty() {
        if bytes.len() < 2 || usize::from(bytes[1]) > bytes.len() - 2 {
            return Err(TsError::Invalid("truncated PSI descriptor"));
        }
        if bytes[0] == 9 {
            return Err(TsError::Unsupported("conditional access descriptor"));
        }
        bytes = &bytes[2 + usize::from(bytes[1])..];
    }
    Ok(())
}

fn adaptation(packet: &[u8; 188], control: u8) -> Result<(usize, bool), TsError> {
    if control & 2 == 0 {
        return Ok((4, false));
    }
    let length = usize::from(packet[4]);
    if length > 183 || control == 2 && length != 183 || control == 3 && length > 182 {
        return Err(TsError::Invalid("adaptation length"));
    }
    let end = 5 + length;
    if length == 0 {
        return Ok((end, false));
    }
    let flags = packet[5];
    let mut at = 6;
    for flag in [0x10, 0x08] {
        if flags & flag != 0 {
            let pcr = take(&packet[..end], &mut at, 6)?;
            if pcr[4] & 0x7e != 0x7e || u16::from_be_bytes([pcr[4] & 1, pcr[5]]) >= 300 {
                return Err(TsError::Invalid("invalid PCR"));
            }
        }
    }
    if flags & 4 != 0 {
        take(&packet[..end], &mut at, 1)?;
    }
    for flag in [2, 1] {
        if flags & flag != 0 {
            let n = usize::from(take(&packet[..end], &mut at, 1)?[0]);
            let field = take(&packet[..end], &mut at, n)?;
            if flag == 1 && !field.is_empty() {
                let mut pos = 1;
                if field[0] & 0x80 != 0 {
                    take(field, &mut pos, 2)?;
                }
                if field[0] & 0x40 != 0 {
                    take(field, &mut pos, 3)?;
                }
                if field[0] & 0x20 != 0 {
                    timestamp(take(field, &mut pos, 5)?, None)?;
                }
            }
        }
    }
    if packet[at..end].iter().any(|b| *b != 0xff) {
        return Err(TsError::Invalid("adaptation stuffing"));
    }
    Ok((end, flags & 0x80 != 0))
}

fn take<'a>(bytes: &'a [u8], at: &mut usize, n: usize) -> Result<&'a [u8], TsError> {
    let end = at.checked_add(n).ok_or(TsError::TooLarge)?;
    let result = bytes
        .get(*at..end)
        .ok_or(TsError::Invalid("optional field out of bounds"))?;
    *at = end;
    Ok(result)
}

fn timestamp(bytes: &[u8], prefix: Option<u8>) -> Result<u64, TsError> {
    if bytes.len() != 5
        || prefix.is_some_and(|p| bytes[0] >> 4 != p)
        || bytes[0] & 1 == 0
        || bytes[2] & 1 == 0
        || bytes[4] & 1 == 0
    {
        return Err(TsError::Invalid("PES timestamp prefix/marker bits"));
    }
    Ok((u64::from((bytes[0] >> 1) & 7) << 30)
        | (u64::from(bytes[1]) << 22)
        | (u64::from(bytes[2] >> 1) << 15)
        | (u64::from(bytes[3]) << 7)
        | u64::from(bytes[4] >> 1))
}

fn parse_pes_header(bytes: &[u8], codec: ElementaryCodec) -> Result<Option<PesHeader>, TsError> {
    if bytes.len() < 6 {
        return Ok(None);
    }
    if bytes[..3] != [0, 0, 1]
        || !match codec {
            ElementaryCodec::H264 => (0xe0..=0xef).contains(&bytes[3]),
            ElementaryCodec::Aac => (0xc0..=0xdf).contains(&bytes[3]),
        }
    {
        return Err(TsError::Invalid("PES start code/stream id"));
    }
    let size = usize::from(u16::from_be_bytes([bytes[4], bytes[5]]));
    if size == 0 && codec != ElementaryCodec::H264 {
        return Err(TsError::Invalid("length-zero non-video PES"));
    }
    if size != 0 && size < 3 {
        return Err(TsError::Invalid("short PES length"));
    }
    if bytes.len() < 9 {
        return Ok(None);
    }
    if bytes[6] & 0xc0 != 0x80 {
        return Err(TsError::Invalid("PES fixed marker bits"));
    }
    if bytes[6] & 0x30 != 0 {
        return Err(TsError::Unsupported("scrambled PES"));
    }
    let offset = 9 + usize::from(bytes[8]);
    if size != 0 && offset > size + 6 {
        return Err(TsError::Invalid("PES header exceeds declared length"));
    }
    if bytes.len() < offset {
        return Ok(None);
    }
    let header = &bytes[..offset];
    let mut at = 9;
    let (pts, dts) = match bytes[7] >> 6 {
        0 => (None, None),
        2 => (Some(timestamp(take(header, &mut at, 5)?, Some(2))?), None),
        3 => (
            Some(timestamp(take(header, &mut at, 5)?, Some(3))?),
            Some(timestamp(take(header, &mut at, 5)?, Some(1))?),
        ),
        _ => return Err(TsError::Invalid("DTS without PTS")),
    };
    for (flag, n) in [(0x20, 6), (0x10, 3), (8, 1), (4, 1), (2, 2)] {
        if bytes[7] & flag != 0 {
            take(header, &mut at, n)?;
        }
    }
    if bytes[7] & 1 != 0 {
        let flags = take(header, &mut at, 1)?[0];
        if flags & 0x80 != 0 {
            take(header, &mut at, 16)?;
        }
        if flags & 0x40 != 0 {
            let n = usize::from(take(header, &mut at, 1)?[0]);
            take(header, &mut at, n)?;
        }
        if flags & 0x20 != 0 {
            take(header, &mut at, 2)?;
        }
        if flags & 0x10 != 0 {
            take(header, &mut at, 2)?;
        }
        if flags & 1 != 0 {
            let length = take(header, &mut at, 1)?[0];
            if length & 0x80 == 0 {
                return Err(TsError::Invalid("PES extension marker"));
            }
            take(header, &mut at, usize::from(length & 0x7f))?;
        }
    }
    if header[at..].iter().any(|b| *b != 0xff) {
        return Err(TsError::Invalid("PES header stuffing"));
    }
    Ok(Some(PesHeader {
        offset,
        length: (size != 0).then_some(size + 6),
        pts,
        dts,
    }))
}

fn crc32_mpeg(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for byte in bytes {
        crc ^= u32::from(*byte) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ 0x04c1_1db7
            } else {
                crc << 1
            };
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn section(mut bytes: Vec<u8>) -> Vec<u8> {
        let length = bytes.len() + 4 - 3;
        bytes[1] = 0xb0 | (length >> 8) as u8;
        bytes[2] = length as u8;
        let crc = crc32_mpeg(&bytes);
        bytes.extend_from_slice(&crc.to_be_bytes());
        assert_eq!(crc32_mpeg(&bytes), 0);
        bytes
    }

    fn pat(number: u8, last: u8, entries: &[(u16, u16)]) -> Vec<u8> {
        let mut bytes = vec![0, 0, 0, 0, 1, 0xc1, number, last];
        for &(program, pid) in entries {
            bytes.extend_from_slice(&program.to_be_bytes());
            bytes.extend_from_slice(&(0xe000 | pid).to_be_bytes());
        }
        section(bytes)
    }

    fn pmt(entries: &[(u8, u16)]) -> Vec<u8> {
        let mut bytes = vec![2, 0, 0, 0, 1, 0xc1, 0, 0, 0xe1, 1, 0xf0, 0];
        for &(kind, pid) in entries {
            bytes.push(kind);
            bytes.extend_from_slice(&(0xe000 | pid).to_be_bytes());
            bytes.extend_from_slice(&[0xf0, 0]);
        }
        section(bytes)
    }

    fn ts(pid: u16, cc: u8, start: bool, payload: &[u8], flags: u8) -> [u8; 188] {
        assert!(payload.len() <= 184);
        let mut bytes = [0xff; 188];
        bytes[0] = 0x47;
        bytes[1] = (pid >> 8) as u8 | if start { 0x40 } else { 0 };
        bytes[2] = pid as u8;
        bytes[3] = cc & 15;
        if payload.len() == 184 {
            assert_eq!(flags, 0);
            bytes[3] |= 0x10;
        } else {
            bytes[3] |= 0x30;
            bytes[4] = (183 - payload.len()) as u8;
            if bytes[4] > 0 {
                bytes[5] = flags;
            } else {
                assert_eq!(flags, 0);
            }
        }
        bytes[188 - payload.len()..].copy_from_slice(payload);
        bytes
    }

    fn psi_ts(pid: u16, cc: u8, section: &[u8]) -> [u8; 188] {
        let mut payload = vec![0];
        payload.extend_from_slice(section);
        ts(pid, cc, true, &payload, 0)
    }

    fn stamp(value: u64, prefix: u8) -> [u8; 5] {
        [
            (prefix << 4) | ((((value >> 30) & 7) as u8) << 1) | 1,
            (value >> 22) as u8,
            ((((value >> 15) & 127) as u8) << 1) | 1,
            (value >> 7) as u8,
            (((value & 127) as u8) << 1) | 1,
        ]
    }

    fn pes(
        video: bool,
        unbounded: bool,
        pts: Option<u64>,
        dts: Option<u64>,
        data: &[u8],
    ) -> Vec<u8> {
        let mut header = Vec::new();
        if let Some(pts) = pts {
            header.extend_from_slice(&stamp(pts, if dts.is_some() { 3 } else { 2 }));
        }
        if let Some(dts) = dts {
            header.extend_from_slice(&stamp(dts, 1));
        }
        let length = if unbounded {
            0
        } else {
            (3 + header.len() + data.len()) as u16
        };
        let mut bytes = vec![0, 0, 1, if video { 0xe0 } else { 0xc0 }];
        bytes.extend_from_slice(&length.to_be_bytes());
        bytes.extend_from_slice(&[
            0x80,
            if dts.is_some() {
                0xc0
            } else if pts.is_some() {
                0x80
            } else {
                0
            },
            header.len() as u8,
        ]);
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(data);
        bytes
    }

    fn ready() -> MpegTsDemux {
        let mut demux = MpegTsDemux::new();
        assert!(
            demux
                .push(&psi_ts(0, 0, &pat(0, 0, &[(1, 0x100)])))
                .unwrap()
                .is_empty()
        );
        assert!(
            demux
                .push(&psi_ts(0x100, 0, &pmt(&[(0x1b, 0x101), (0x0f, 0x102)])))
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            demux.streams().collect::<Vec<_>>(),
            vec![(0x101, StreamKind::H264), (0x102, StreamKind::Aac)]
        );
        demux
    }

    #[test]
    fn every_byte_split_and_chunk_size_preserves_complete_pes_and_33_bit_timestamps() {
        let video = [0, 0, 1, 0x65, 4, 5, 6];
        let audio = [0xff, 0xf1, 0x50, 0x80, 1, 0x3f, 0xfc, 8];
        let max = (1u64 << 33) - 1;
        let bytes: Vec<u8> = [
            psi_ts(0, 0, &pat(0, 0, &[(1, 0x100)])),
            psi_ts(0x100, 0, &pmt(&[(0x1b, 0x101), (0x0f, 0x102)])),
            ts(
                0x101,
                15,
                true,
                &pes(true, true, Some(max), Some(max - 9), &video),
                0,
            ),
            ts(0x102, 0, true, &pes(false, false, Some(3), None, &audio), 0),
        ]
        .into_iter()
        .flatten()
        .collect();
        let mut baseline = TransportStream::new();
        let mut expected = baseline.push(&bytes).unwrap();
        expected.extend(baseline.finish_segment().unwrap());
        assert_eq!(expected.len(), 2);
        assert_eq!(expected[0].kind, StreamKind::Aac);
        assert_eq!(expected[0].data, audio);
        assert_eq!(expected[1].data, video);
        assert_eq!(expected[1].pts, Some(max));
        assert_eq!(expected[1].dts, Some(max - 9));
        for cut in 0..=bytes.len() {
            let mut demux = TransportStream::new();
            let mut actual = demux.push(&bytes[..cut]).unwrap();
            actual.extend(demux.push(&bytes[cut..]).unwrap());
            actual.extend(demux.finish_segment().unwrap());
            assert_eq!(actual, expected, "split {cut}");
        }
        for size in [1, 2, 3, 7, 31, 187, 188, 189, 256] {
            let mut demux = TransportStream::new();
            let mut actual = Vec::new();
            for chunk in bytes.chunks(size) {
                actual.extend(demux.push(chunk).unwrap());
            }
            actual.extend(demux.finish_segment().unwrap());
            assert_eq!(actual, expected, "chunk {size}");
        }
    }

    #[test]
    fn pes_header_can_span_transport_packets_and_counter_wraps() {
        let mut demux = ready();
        let bytes = pes(true, false, Some(123), Some(100), &[0, 0, 1, 0x65, 9]);
        let mut out = Vec::new();
        for (index, piece) in bytes.chunks(2).enumerate() {
            out.extend(
                demux
                    .push(&ts(0x101, (15 + index) as u8, index == 0, piece, 0))
                    .unwrap(),
            );
        }
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].pts, Some(123));
        assert_eq!(out[0].dts, Some(100));
        assert_eq!(out[0].data, [0, 0, 1, 0x65, 9]);
        assert!(demux.finish().unwrap().is_empty());
    }

    #[test]
    fn psi_pointer_finishes_previous_section_and_commits_multisection_pat_atomically() {
        let mut demux = MpegTsDemux::new();
        let first = pat(0, 1, &[(0, 0x20)]);
        let second = pat(1, 1, &[(1, 0x100)]);
        let mut first_payload = vec![0];
        first_payload.extend_from_slice(&first[..10]);
        demux.push(&ts(0, 0, true, &first_payload, 0)).unwrap();
        assert_eq!(demux.program_number(), None);
        let mut next = vec![(first.len() - 10) as u8];
        next.extend_from_slice(&first[10..]);
        next.extend_from_slice(&second);
        next.extend_from_slice(&[0xff; 3]);
        demux.push(&ts(0, 1, true, &next, 0)).unwrap();
        assert_eq!(demux.program_number(), Some(1));
        demux
            .push(&psi_ts(0x100, 0, &pmt(&[(0x1b, 0x101)])))
            .unwrap();
        assert_eq!(demux.streams().count(), 1);
        let mut together = vec![0];
        together.extend_from_slice(&first);
        together.extend_from_slice(&second);
        demux.push(&ts(0, 2, true, &together, 0)).unwrap();
        assert_eq!(demux.streams().count(), 1);
    }

    #[test]
    fn psi_pointer_can_skip_lost_prefix_but_cannot_hide_incomplete_section() {
        let mut demux = MpegTsDemux::new();
        let valid = pat(0, 0, &[(1, 0x100)]);
        let mut recovered = vec![3, 0, 1, 2];
        recovered.extend_from_slice(&valid);
        demux.push(&ts(0, 0, true, &recovered, 0)).unwrap();
        assert_eq!(demux.program_number(), Some(1));
        let mut beginning = vec![0];
        beginning.extend_from_slice(&valid[..5]);
        demux.push(&ts(0, 1, true, &beginning, 0)).unwrap();
        assert!(demux.push(&ts(0, 2, true, &[0], 0)).is_err());
        assert_eq!(demux.program_number(), None);
        assert!(demux.push(&ts(0, 0, true, &[3, 1], 0)).is_err());
    }

    #[test]
    fn continuity_loss_drops_partial_pes_and_exact_duplicate_does_not_repeat_bytes() {
        let mut demux = ready();
        let first = ts(0x101, 0, true, &pes(true, true, Some(1), None, &[1, 2]), 0);
        demux.push(&first).unwrap();
        assert!(demux.push(&first).unwrap().is_empty());
        demux.push(&ts(0x101, 2, false, &[3, 4], 0)).unwrap();
        demux
            .push(&ts(
                0x101,
                3,
                true,
                &pes(true, true, Some(2), None, &[5, 6]),
                0,
            ))
            .unwrap();
        let out = demux.finish().unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].data, [5, 6]);
        assert!(out[0].discontinuity);
    }

    #[test]
    fn adaptation_only_keeps_counter_and_discontinuity_discards_pending_pes() {
        let mut demux = ready();
        demux
            .push(&ts(0x101, 0, true, &pes(true, true, None, None, &[1]), 0))
            .unwrap();
        let mut only = ts(0x101, 0, false, &[], 0);
        only[3] = 0x20;
        demux.push(&only).unwrap();
        demux.push(&ts(0x101, 1, false, &[2], 0)).unwrap();
        assert_eq!(demux.finish().unwrap()[0].data, [1, 2]);
        demux
            .push(&ts(0x101, 2, true, &pes(true, true, None, None, &[3]), 0))
            .unwrap();
        only[3] = 0x22;
        only[5] = 0x80;
        demux.push(&only).unwrap();
        demux
            .push(&ts(0x101, 9, true, &pes(true, false, None, None, &[4]), 0))
            .map(|out| {
                assert_eq!(out.len(), 1);
                assert!(out[0].discontinuity);
                assert_eq!(out[0].data, [4]);
            })
            .unwrap();
    }

    #[test]
    fn segment_flush_preserves_tracks_and_continuity_and_repeated_pmt_preserves_pending_pes() {
        let mut demux = ready();
        demux
            .push(&ts(0x101, 0, true, &pes(true, true, None, None, &[1]), 0))
            .unwrap();
        demux
            .push(&psi_ts(0x100, 1, &pmt(&[(0x1b, 0x101), (0x0f, 0x102)])))
            .unwrap();
        assert_eq!(demux.finish().unwrap()[0].data, [1]);
        let out = demux
            .push(&ts(
                0x101,
                1,
                true,
                &pes(true, false, Some(1), None, &[2]),
                0,
            ))
            .unwrap();
        assert_eq!(out[0].data, [2]);
        assert!(!out[0].discontinuity);
        assert_eq!(demux.streams().count(), 2);
        demux.reset();
        assert_eq!(demux.streams().count(), 0);
    }

    #[test]
    fn invalid_crc_reserved_bits_identity_descriptors_and_multisection_headers_fail_closed() {
        for mutation in 0..7 {
            let mut demux = ready();
            let mut bytes = pmt(&[(0x1b, 0x101)]);
            bytes.truncate(bytes.len() - 4);
            match mutation {
                0 => bytes[3] = 1,
                1 => bytes[6] = 1,
                2 => bytes[7] = 1,
                3 => bytes[10] = 0,
                4 => bytes[11] = 0xff,
                5 => bytes[16] = 0xff,
                _ => bytes[5] = 1,
            }
            let bytes = section(bytes);
            assert!(
                demux.push(&psi_ts(0x100, 1, &bytes)).is_err(),
                "mutation {mutation}"
            );
            assert_eq!(demux.streams().count(), 0);
        }
        let mut demux = ready();
        let mut bad = pat(0, 0, &[(1, 0x100)]);
        bad[10] ^= 1;
        assert!(demux.push(&psi_ts(0, 1, &bad)).is_err());
        let mut demux = MpegTsDemux::new();
        demux.push(&psi_ts(0, 0, &pat(0, 1, &[(0, 0x20)]))).unwrap();
        assert!(
            demux
                .push(&psi_ts(0, 1, &pat(1, 2, &[(1, 0x100)])))
                .is_err()
        );
        let mut demux = MpegTsDemux::new();
        assert_eq!(
            demux
                .push(&psi_ts(0, 0, &pat(0, 0, &[(1, 0x100), (2, 0x200)])))
                .unwrap_err(),
            TsError::Unsupported("multiple HLS transport programs")
        );
    }

    #[test]
    fn timestamp_markers_header_length_and_audio_zero_length_are_rejected() {
        for mutation in 0..5 {
            let mut demux = ready();
            let mut bytes = pes(true, false, Some(123), Some(100), &[1]);
            match mutation {
                0 => bytes[9] &= !1,
                1 => bytes[14] = (bytes[14] & 15) | 0x20,
                2 => bytes[7] = 0x40,
                3 => bytes[8] = 255,
                _ => bytes[6] = 0,
            }
            assert!(
                demux.push(&ts(0x101, 0, true, &bytes, 0)).is_err(),
                "mutation {mutation}"
            );
        }
        let mut demux = ready();
        let bytes = pes(false, true, None, None, &[1]);
        assert!(demux.push(&ts(0x102, 0, true, &bytes, 0)).is_err());
    }

    #[test]
    fn truncation_and_security_limits_reset_state() {
        let mut demux = ready();
        demux.push(&[0x47]).unwrap();
        assert_eq!(demux.finish().unwrap_err(), TsError::Truncated);
        assert_eq!(demux.streams().count(), 0);
        let mut demux = ready();
        let bytes = pes(true, false, None, None, &[1, 2]);
        demux.push(&ts(0x101, 0, true, &bytes[..10], 0)).unwrap();
        assert_eq!(demux.finish().unwrap_err(), TsError::Truncated);
        assert_eq!(
            demux.push(&vec![0; MAX_PUSH_BYTES + 1]).unwrap_err(),
            TsError::TooLarge
        );
        let mut stream = Stream {
            codec: StreamKind::H264,
            continuity: Continuity::default(),
            pes: None,
            discontinuity: false,
        };
        assert_eq!(
            stream
                .feed(0x101, &vec![0; MAX_PES_BYTES + 1], true, &mut Vec::new())
                .unwrap_err(),
            TsError::TooLarge
        );
        let mut slot = None;
        for number in 0..64u8 {
            let mut bytes = vec![0; 1024];
            bytes[5] = 0xc1;
            bytes[6] = number;
            bytes[7] = 255;
            assert!(Table::accept(&mut slot, bytes).unwrap().is_none());
        }
        let mut bytes = vec![0; 1024];
        bytes[5] = 0xc1;
        bytes[6] = 64;
        bytes[7] = 255;
        assert_eq!(
            Table::accept(&mut slot, bytes).unwrap_err(),
            TsError::TooLarge
        );
    }

    #[test]
    fn malformed_adaptation_and_random_packets_never_panic_or_retain_unbounded_input() {
        let mut bad = ts(0x101, 0, true, &[1], 0);
        bad[4] = 184;
        assert!(ready().push(&bad).is_err());
        let mut bad = ts(0x101, 0, true, &[1], 0);
        bad[5] = 0x10;
        bad[4] = 1;
        assert!(ready().push(&bad).is_err());
        let mut rng = 17u32;
        let mut demux = MpegTsDemux::new();
        for _ in 0..2048 {
            let mut bytes = [0u8; 188];
            for byte in &mut bytes {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                *byte = (rng >> 24) as u8;
            }
            bytes[0] = 0x47;
            let _ = demux.push(&bytes);
            assert!(demux.partial.len() < TS_PACKET_BYTES);
            assert!(demux.streams.len() <= MAX_STREAMS);
        }
    }
}
