//! Bounded complete HLS playlists (RFC 8216, sections 4 and 7).
//! URI references are retained verbatim; the host resolves them against the playlist URL.
//! No fetching, decryption, delta updates, or low-latency partial segments live here.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use super::backend::MediaDecodeError;

pub const MAX_PLAYLIST_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_LINE_BYTES: usize = 16 * 1024;
pub const MAX_SEGMENTS: usize = 10_000;
pub const MAX_VARIANTS: usize = 128;
pub const MAX_RENDITIONS: usize = 128;
const MAX_ATTRIBUTES: usize = 32;
const MAX_CODECS: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub enum Playlist {
    Master(MasterPlaylist),
    Media(MediaPlaylist),
}

#[derive(Clone, Debug, PartialEq)]
pub struct MasterPlaylist {
    pub version: u32,
    pub independent_segments: bool,
    pub variants: Vec<Variant>,
    pub renditions: Vec<Rendition>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Variant {
    pub uri: String,
    pub bandwidth: u64,
    pub average_bandwidth: Option<u64>,
    pub codecs: Vec<String>,
    pub resolution: Option<(u32, u32)>,
    pub frame_rate: Option<f64>,
    pub audio: Option<String>,
    /// True when the referenced audio group has separately fetched renditions.
    pub external_audio: bool,
    pub video: Option<String>,
    pub subtitles: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenditionType {
    Audio,
    Video,
    Subtitles,
    ClosedCaptions,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rendition {
    pub kind: RenditionType,
    pub group_id: String,
    pub name: String,
    pub uri: Option<String>,
    pub language: Option<String>,
    pub instream_id: Option<String>,
    pub default: bool,
    pub autoselect: bool,
    pub forced: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaylistType {
    Event,
    Vod,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MediaPlaylist {
    pub version: u32,
    pub independent_segments: bool,
    pub target_duration: f64,
    pub media_sequence: u64,
    pub discontinuity_sequence: u64,
    pub playlist_type: Option<PlaylistType>,
    pub segments: Vec<Segment>,
    pub end_list: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub uri: String,
    pub duration: f64,
    pub title: String,
    pub sequence: u64,
    pub discontinuity: bool,
    pub discontinuity_sequence: u64,
    pub map: Option<Arc<InitializationMap>>,
    pub byte_range: Option<ByteRange>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InitializationMap {
    pub uri: String,
    pub byte_range: Option<ByteRange>,
}

/// Resolved half-open byte range. `offset + length` is checked during parsing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteRange {
    pub offset: u64,
    pub length: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HlsError {
    /// One-based source line, or zero for a whole-input size/encoding failure.
    pub line: usize,
    pub kind: HlsErrorKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HlsErrorKind {
    Invalid(&'static str),
    LimitExceeded(&'static str),
    UnsupportedEncryption,
    Unsupported(&'static str),
}

impl std::fmt::Display for HlsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HLS line {}: {:?}", self.line, self.kind)
    }
}

impl std::error::Error for HlsError {}

fn invalid(line: usize, reason: &'static str) -> HlsError {
    HlsError {
        line,
        kind: HlsErrorKind::Invalid(reason),
    }
}

fn limit(line: usize, reason: &'static str) -> HlsError {
    HlsError {
        line,
        kind: HlsErrorKind::LimitExceeded(reason),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Master,
    Media,
}

fn set_kind(kind: &mut Option<Kind>, next: Kind, line: usize) -> Result<(), HlsError> {
    if kind.is_some_and(|kind| kind != next) {
        return Err(invalid(line, "mixed master and media tags"));
    }
    *kind = Some(next);
    Ok(())
}

fn integer(value: &str, line: usize) -> Result<u64, HlsError> {
    if value.is_empty() || value.len() > 20 || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid(line, "invalid decimal integer"));
    }
    value.parse().map_err(|_| invalid(line, "integer overflow"))
}

fn positive(value: &str, line: usize) -> Result<u64, HlsError> {
    let value = integer(value, line)?;
    if value == 0 {
        return Err(invalid(line, "expected positive integer"));
    }
    Ok(value)
}

fn decimal(value: &str, line: usize) -> Result<f64, HlsError> {
    if value.is_empty()
        || !value.bytes().any(|byte| byte.is_ascii_digit())
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || byte == b'.')
        || value.bytes().filter(|&byte| byte == b'.').count() > 1
    {
        return Err(invalid(line, "invalid decimal duration or rate"));
    }
    let number: f64 = value
        .parse()
        .map_err(|_| invalid(line, "invalid decimal"))?;
    if !number.is_finite() || number <= 0.0 {
        return Err(invalid(
            line,
            "duration or rate must be finite and positive",
        ));
    }
    Ok(number)
}

fn reference(value: &str, line: usize) -> Result<String, HlsError> {
    if value.is_empty() || value.chars().any(char::is_whitespace) {
        return Err(invalid(line, "empty URI or URI containing whitespace"));
    }
    if value.contains("{$") {
        return Err(HlsError {
            line,
            kind: HlsErrorKind::Unsupported("URI variables"),
        });
    }
    Ok(value.to_string())
}

struct Attributes<'a> {
    values: BTreeMap<&'a str, (&'a str, bool)>,
    line: usize,
}

impl<'a> Attributes<'a> {
    fn parse(mut text: &'a str, line: usize) -> Result<Self, HlsError> {
        let mut values = BTreeMap::new();
        if text.is_empty() {
            return Err(invalid(line, "empty attribute list"));
        }
        loop {
            if values.len() >= MAX_ATTRIBUTES {
                return Err(limit(line, "attributes"));
            }
            let (name, rest) = text
                .split_once('=')
                .ok_or_else(|| invalid(line, "attribute missing '='"))?;
            if name.is_empty()
                || !name
                    .bytes()
                    .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'-')
            {
                return Err(invalid(line, "invalid attribute name"));
            }
            let (value, quoted, remainder) = if let Some(rest) = rest.strip_prefix('"') {
                let end = rest
                    .find('"')
                    .ok_or_else(|| invalid(line, "unterminated quoted attribute"))?;
                (&rest[..end], true, &rest[end + 1..])
            } else {
                let end = rest.find(',').unwrap_or(rest.len());
                let value = &rest[..end];
                if value.is_empty() || value.chars().any(|ch| ch.is_whitespace() || ch == '"') {
                    return Err(invalid(line, "invalid unquoted attribute"));
                }
                (value, false, &rest[end..])
            };
            if values.insert(name, (value, quoted)).is_some() {
                return Err(invalid(line, "duplicate attribute"));
            }
            if remainder.is_empty() {
                break;
            }
            text = remainder
                .strip_prefix(',')
                .ok_or_else(|| invalid(line, "expected attribute comma"))?;
            if text.is_empty() {
                return Err(invalid(line, "trailing attribute comma"));
            }
        }
        Ok(Self { values, line })
    }

    fn optional(&self, name: &str, quoted: bool) -> Result<Option<&'a str>, HlsError> {
        self.values
            .get(name)
            .map(|&(value, actual)| {
                if actual != quoted {
                    Err(invalid(self.line, "incorrect attribute quoting"))
                } else {
                    Ok(value)
                }
            })
            .transpose()
    }

    fn required(&self, name: &str, quoted: bool) -> Result<&'a str, HlsError> {
        self.optional(name, quoted)?
            .ok_or_else(|| invalid(self.line, "missing required attribute"))
    }

    fn yes_no(&self, name: &str) -> Result<bool, HlsError> {
        match self.optional(name, false)? {
            None | Some("NO") => Ok(false),
            Some("YES") => Ok(true),
            _ => Err(invalid(self.line, "expected YES or NO")),
        }
    }
}

fn variant(text: &str, line: usize) -> Result<Variant, HlsError> {
    let attrs = Attributes::parse(text, line)?;
    let codecs = attrs
        .optional("CODECS", true)?
        .map(|value| {
            if value.split(',').count() > MAX_CODECS {
                return Err(limit(line, "codecs"));
            }
            value
                .split(',')
                .map(|codec| {
                    if codec.is_empty() || codec.chars().any(char::is_whitespace) {
                        Err(invalid(line, "empty or invalid codec"))
                    } else {
                        Ok(codec.to_string())
                    }
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();
    let resolution = attrs
        .optional("RESOLUTION", false)?
        .map(|value| {
            let (width, height) = value
                .split_once('x')
                .ok_or_else(|| invalid(line, "invalid resolution"))?;
            let width = u32::try_from(positive(width, line)?)
                .map_err(|_| invalid(line, "resolution overflow"))?;
            let height = u32::try_from(positive(height, line)?)
                .map_err(|_| invalid(line, "resolution overflow"))?;
            Ok::<_, HlsError>((width, height))
        })
        .transpose()?;
    // Retain rendition group references rather than silently dropping separate audio.
    Ok(Variant {
        uri: String::new(),
        bandwidth: positive(attrs.required("BANDWIDTH", false)?, line)?,
        average_bandwidth: attrs
            .optional("AVERAGE-BANDWIDTH", false)?
            .map(|value| positive(value, line))
            .transpose()?,
        codecs,
        resolution,
        frame_rate: attrs
            .optional("FRAME-RATE", false)?
            .map(|value| decimal(value, line))
            .transpose()?,
        audio: attrs.optional("AUDIO", true)?.map(str::to_string),
        external_audio: false,
        video: attrs.optional("VIDEO", true)?.map(str::to_string),
        subtitles: attrs.optional("SUBTITLES", true)?.map(str::to_string),
    })
}

fn rendition(text: &str, line: usize) -> Result<Rendition, HlsError> {
    let attrs = Attributes::parse(text, line)?;
    let kind = match attrs.required("TYPE", false)? {
        "AUDIO" => RenditionType::Audio,
        "VIDEO" => RenditionType::Video,
        "SUBTITLES" => RenditionType::Subtitles,
        "CLOSED-CAPTIONS" => RenditionType::ClosedCaptions,
        _ => return Err(invalid(line, "invalid rendition type")),
    };
    let uri = attrs
        .optional("URI", true)?
        .map(|value| reference(value, line))
        .transpose()?;
    let instream_id = attrs.optional("INSTREAM-ID", true)?.map(str::to_string);
    if (kind == RenditionType::Subtitles && uri.is_none())
        || (kind == RenditionType::ClosedCaptions && (uri.is_some() || instream_id.is_none()))
        || (kind != RenditionType::ClosedCaptions && instream_id.is_some())
    {
        return Err(invalid(line, "invalid rendition URI or INSTREAM-ID"));
    }
    let default = attrs.yes_no("DEFAULT")?;
    let autoselect = attrs.yes_no("AUTOSELECT")?;
    let forced = attrs.yes_no("FORCED")?;
    if (default && attrs.values.contains_key("AUTOSELECT") && !autoselect)
        || (kind != RenditionType::Subtitles && attrs.values.contains_key("FORCED"))
    {
        return Err(invalid(line, "invalid rendition selection flags"));
    }
    let group_id = attrs.required("GROUP-ID", true)?;
    let name = attrs.required("NAME", true)?;
    if group_id.is_empty() || name.is_empty() {
        return Err(invalid(line, "empty rendition group or name"));
    }
    Ok(Rendition {
        kind,
        group_id: group_id.to_string(),
        name: name.to_string(),
        uri,
        language: attrs.optional("LANGUAGE", true)?.map(str::to_string),
        instream_id,
        default,
        autoselect,
        forced,
    })
}

fn range_spec(text: &str, line: usize) -> Result<(u64, Option<u64>), HlsError> {
    let (length, offset) = match text.split_once('@') {
        Some((length, offset)) => (length, Some(integer(offset, line)?)),
        None => (text, None),
    };
    Ok((positive(length, line)?, offset))
}

fn resolve_range(
    spec: (u64, Option<u64>),
    uri: &str,
    previous: Option<&Segment>,
    line: usize,
) -> Result<ByteRange, HlsError> {
    let offset = match spec.1 {
        Some(offset) => offset,
        None => {
            let previous = previous
                .filter(|segment| segment.uri == uri)
                .and_then(|segment| segment.byte_range)
                .ok_or_else(|| {
                    invalid(
                        line,
                        "implicit range requires previous ranged segment of same URI",
                    )
                })?;
            previous
                .offset
                .checked_add(previous.length)
                .ok_or_else(|| invalid(line, "byte range overflow"))?
        }
    };
    offset
        .checked_add(spec.0)
        .ok_or_else(|| invalid(line, "byte range overflow"))?;
    Ok(ByteRange {
        offset,
        length: spec.0,
    })
}

/// Parse one complete UTF-8 playlist. Unknown tags are ignored per RFC 8216.
/// Implicit ranges require identical URI spelling; URL normalization is host-owned.
pub fn parse_playlist(text: &str) -> Result<Playlist, MediaDecodeError> {
    parse_playlist_inner(text).map_err(|error| match error.kind {
        HlsErrorKind::UnsupportedEncryption | HlsErrorKind::Unsupported(_) => {
            MediaDecodeError::Unsupported
        }
        _ => MediaDecodeError::InvalidData(error.to_string()),
    })
}

fn parse_playlist_inner(text: &str) -> Result<Playlist, HlsError> {
    if text.len() > MAX_PLAYLIST_BYTES {
        return Err(limit(0, "playlist bytes"));
    }
    if text.starts_with('\u{feff}')
        || text.chars().any(|ch| {
            matches!(ch, '\u{0}'..='\u{1f}' | '\u{7f}'..='\u{9f}') && ch != '\r' && ch != '\n'
        })
    {
        return Err(invalid(0, "BOM or forbidden control character"));
    }
    let mut lines = text.split('\n').enumerate();
    let header = lines
        .next()
        .map(|(_, line)| line.strip_suffix('\r').unwrap_or(line));
    if header != Some("#EXTM3U") {
        return Err(invalid(1, "missing EXTM3U header"));
    }
    let mut master = MasterPlaylist {
        version: 1,
        independent_segments: false,
        variants: Vec::new(),
        renditions: Vec::new(),
    };
    let mut media = MediaPlaylist {
        version: 1,
        independent_segments: false,
        target_duration: 0.0,
        media_sequence: 0,
        discontinuity_sequence: 0,
        playlist_type: None,
        segments: Vec::new(),
        end_list: false,
    };
    let mut kind = None;
    let mut seen = HashSet::new();
    let mut pending_variant: Option<Variant> = None;
    let mut pending_duration: Option<(f64, String)> = None;
    let mut pending_range: Option<(u64, Option<u64>)> = None;
    let mut pending_discontinuity = false;
    let mut discontinuity_sequence = 0u64;
    let mut map: Option<Arc<InitializationMap>> = None;
    let mut min_version = 1;
    let mut last_line = 1;
    for (index, raw) in lines {
        let line = index + 1;
        last_line = line;
        let value = raw.strip_suffix('\r').unwrap_or(raw);
        if value.len() > MAX_LINE_BYTES {
            return Err(limit(line, "line bytes"));
        }
        if value.contains('\r') {
            return Err(invalid(line, "bare carriage return"));
        }
        if value.is_empty() {
            continue;
        }
        if !value.starts_with('#') {
            let uri = reference(value, line)?;
            if let Some(mut next) = pending_variant.take() {
                set_kind(&mut kind, Kind::Master, line)?;
                if master.variants.len() >= MAX_VARIANTS {
                    return Err(limit(line, "variants"));
                }
                next.uri = uri;
                master.variants.push(next);
            } else {
                set_kind(&mut kind, Kind::Media, line)?;
                let (duration, title) = pending_duration
                    .take()
                    .ok_or_else(|| invalid(line, "segment missing EXTINF"))?;
                if media.segments.len() >= MAX_SEGMENTS {
                    return Err(limit(line, "segments"));
                }
                let byte_range = pending_range
                    .take()
                    .map(|spec| resolve_range(spec, &uri, media.segments.last(), line))
                    .transpose()?;
                let sequence = media
                    .media_sequence
                    .checked_add(media.segments.len() as u64)
                    .ok_or_else(|| invalid(line, "media sequence overflow"))?;
                media.segments.push(Segment {
                    uri,
                    duration,
                    title,
                    sequence,
                    byte_range,
                    discontinuity: std::mem::take(&mut pending_discontinuity),
                    discontinuity_sequence,
                    map: map.clone(),
                });
            }
            continue;
        }
        if !value.starts_with("#EXT") {
            continue;
        }
        let (tag, argument) = value
            .split_once(':')
            .map(|(tag, arg)| (tag, Some(arg)))
            .unwrap_or((value, None));
        let arg = || argument.ok_or_else(|| invalid(line, "tag missing value"));
        if matches!(
            tag,
            "#EXTM3U"
                | "#EXT-X-VERSION"
                | "#EXT-X-TARGETDURATION"
                | "#EXT-X-MEDIA-SEQUENCE"
                | "#EXT-X-DISCONTINUITY-SEQUENCE"
                | "#EXT-X-ENDLIST"
                | "#EXT-X-PLAYLIST-TYPE"
                | "#EXT-X-INDEPENDENT-SEGMENTS"
        ) && !seen.insert(tag)
        {
            return Err(invalid(line, "duplicate singleton tag"));
        }
        if matches!(
            tag,
            "#EXTM3U" | "#EXT-X-ENDLIST" | "#EXT-X-DISCONTINUITY" | "#EXT-X-INDEPENDENT-SEGMENTS"
        ) && argument.is_some()
        {
            return Err(invalid(line, "tag must not have a value"));
        }
        match tag {
            "#EXTM3U" => return Err(invalid(line, "repeated EXTM3U header")),
            "#EXT-X-VERSION" => {
                let version = u32::try_from(positive(arg()?, line)?)
                    .map_err(|_| invalid(line, "version overflow"))?;
                if version > 7 {
                    return Err(HlsError {
                        line,
                        kind: HlsErrorKind::Unsupported("protocol version newer than RFC 8216"),
                    });
                }
                master.version = version;
                media.version = version;
            }
            "#EXT-X-INDEPENDENT-SEGMENTS" => {
                master.independent_segments = true;
                media.independent_segments = true;
            }
            "#EXT-X-STREAM-INF" => {
                set_kind(&mut kind, Kind::Master, line)?;
                if pending_variant.is_some() {
                    return Err(invalid(line, "variant missing URI"));
                }
                pending_variant = Some(variant(arg()?, line)?);
            }
            "#EXT-X-MEDIA" => {
                set_kind(&mut kind, Kind::Master, line)?;
                if master.renditions.len() >= MAX_RENDITIONS {
                    return Err(limit(line, "renditions"));
                }
                let next = rendition(arg()?, line)?;
                if master.renditions.iter().any(|old| {
                    old.kind == next.kind && old.group_id == next.group_id && old.name == next.name
                }) {
                    return Err(invalid(line, "duplicate rendition name in group"));
                }
                if next
                    .instream_id
                    .as_deref()
                    .is_some_and(|id| id.starts_with("SERVICE"))
                {
                    min_version = min_version.max(7);
                }
                master.renditions.push(next);
            }
            "#EXT-X-SESSION-KEY" | "#EXT-X-KEY" => {
                set_kind(
                    &mut kind,
                    if tag == "#EXT-X-KEY" {
                        Kind::Media
                    } else {
                        Kind::Master
                    },
                    line,
                )?;
                let attrs = Attributes::parse(arg()?, line)?;
                let method = attrs.required("METHOD", false)?;
                if method != "NONE" {
                    return Err(HlsError {
                        line,
                        kind: HlsErrorKind::UnsupportedEncryption,
                    });
                }
                if tag == "#EXT-X-SESSION-KEY" || attrs.values.len() != 1 {
                    return Err(invalid(line, "invalid METHOD=NONE key"));
                }
            }
            "#EXT-X-TARGETDURATION" => {
                set_kind(&mut kind, Kind::Media, line)?;
                media.target_duration = positive(arg()?, line)? as f64;
            }
            "#EXT-X-MEDIA-SEQUENCE" => {
                set_kind(&mut kind, Kind::Media, line)?;
                if !media.segments.is_empty() {
                    return Err(invalid(line, "late media sequence"));
                }
                media.media_sequence = integer(arg()?, line)?;
            }
            "#EXT-X-DISCONTINUITY-SEQUENCE" => {
                set_kind(&mut kind, Kind::Media, line)?;
                if !media.segments.is_empty() || pending_discontinuity {
                    return Err(invalid(line, "late discontinuity sequence"));
                }
                discontinuity_sequence = integer(arg()?, line)?;
                media.discontinuity_sequence = discontinuity_sequence;
            }
            "#EXTINF" => {
                set_kind(&mut kind, Kind::Media, line)?;
                if pending_duration.is_some() {
                    return Err(invalid(line, "EXTINF missing segment URI"));
                }
                let (duration, title) = arg()?
                    .split_once(',')
                    .ok_or_else(|| invalid(line, "EXTINF missing comma"))?;
                if duration.contains('.') {
                    min_version = min_version.max(3);
                }
                pending_duration = Some((decimal(duration, line)?, title.to_string()));
            }
            "#EXT-X-BYTERANGE" => {
                set_kind(&mut kind, Kind::Media, line)?;
                if pending_range.is_some() {
                    return Err(invalid(line, "duplicate segment byte range"));
                }
                pending_range = Some(range_spec(arg()?, line)?);
                min_version = min_version.max(4);
            }
            "#EXT-X-DISCONTINUITY" => {
                set_kind(&mut kind, Kind::Media, line)?;
                if pending_discontinuity {
                    return Err(invalid(line, "duplicate pending discontinuity"));
                }
                pending_discontinuity = true;
                discontinuity_sequence = discontinuity_sequence
                    .checked_add(1)
                    .ok_or_else(|| invalid(line, "discontinuity sequence overflow"))?;
            }
            "#EXT-X-MAP" => {
                set_kind(&mut kind, Kind::Media, line)?;
                let attrs = Attributes::parse(arg()?, line)?;
                let uri = reference(attrs.required("URI", true)?, line)?;
                let byte_range = attrs
                    .optional("BYTERANGE", true)?
                    .map(|value| {
                        resolve_range(range_spec(value, line)?, &uri, media.segments.last(), line)
                    })
                    .transpose()?;
                map = Some(Arc::new(InitializationMap { uri, byte_range }));
                min_version = min_version.max(6);
            }
            "#EXT-X-PLAYLIST-TYPE" => {
                set_kind(&mut kind, Kind::Media, line)?;
                media.playlist_type = Some(match arg()? {
                    "EVENT" => PlaylistType::Event,
                    "VOD" => PlaylistType::Vod,
                    _ => return Err(invalid(line, "invalid playlist type")),
                });
            }
            "#EXT-X-ENDLIST" => {
                set_kind(&mut kind, Kind::Media, line)?;
                media.end_list = true;
            }
            "#EXT-X-PROGRAM-DATE-TIME" | "#EXT-X-DATERANGE" => {
                set_kind(&mut kind, Kind::Media, line)?;
                if arg()?.is_empty() {
                    return Err(invalid(line, "empty media metadata"));
                }
            }
            "#EXT-X-SESSION-DATA" => {
                set_kind(&mut kind, Kind::Master, line)?;
                Attributes::parse(arg()?, line)?;
            }
            "#EXT-X-I-FRAMES-ONLY"
            | "#EXT-X-I-FRAME-STREAM-INF"
            | "#EXT-X-START"
            | "#EXT-X-DEFINE"
            | "#EXT-X-PART"
            | "#EXT-X-PART-INF"
            | "#EXT-X-SKIP"
            | "#EXT-X-PRELOAD-HINT"
            | "#EXT-X-SERVER-CONTROL"
            | "#EXT-X-RENDITION-REPORT" => {
                return Err(HlsError {
                    line,
                    kind: HlsErrorKind::Unsupported(
                        "I-frame, start-offset, variable, delta, or low-latency playlist",
                    ),
                });
            }
            // Metadata and unknown extension tags do not affect segment fetching.
            _ => {}
        }
    }
    if pending_variant.is_some()
        || pending_duration.is_some()
        || pending_range.is_some()
        || pending_discontinuity
    {
        return Err(invalid(last_line, "unfinished variant or segment"));
    }
    if media.version < min_version {
        return Err(invalid(
            last_line,
            "protocol version too low for playlist tags",
        ));
    }
    if kind == Some(Kind::Master) {
        if master.variants.is_empty() {
            return Err(invalid(last_line, "master has no variants"));
        }
        for variant in &mut master.variants {
            for (group, kind) in [
                (&variant.audio, RenditionType::Audio),
                (&variant.video, RenditionType::Video),
                (&variant.subtitles, RenditionType::Subtitles),
            ] {
                if group.as_ref().is_some_and(|group| {
                    !master
                        .renditions
                        .iter()
                        .any(|rendition| rendition.kind == kind && rendition.group_id == *group)
                }) {
                    return Err(invalid(
                        last_line,
                        "variant references missing rendition group",
                    ));
                }
            }
            variant.external_audio = variant.audio.as_ref().is_some_and(|group| {
                master.renditions.iter().any(|rendition| {
                    rendition.kind == RenditionType::Audio
                        && rendition.group_id == *group
                        && rendition.uri.is_some()
                })
            });
        }
        return Ok(Playlist::Master(master));
    }
    if media.target_duration == 0.0 {
        return Err(invalid(last_line, "missing target duration"));
    }
    if media
        .segments
        .iter()
        .any(|segment| segment.duration.round() > media.target_duration)
    {
        return Err(invalid(last_line, "segment exceeds target duration"));
    }
    if media.playlist_type == Some(PlaylistType::Vod) && !media.end_list {
        return Err(invalid(last_line, "VOD missing ENDLIST"));
    }
    Ok(Playlist::Media(media))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn media(text: &str) -> MediaPlaylist {
        let Playlist::Media(playlist) = parse_playlist(text).unwrap() else {
            panic!("expected media playlist");
        };
        playlist
    }

    fn master(text: &str) -> MasterPlaylist {
        let Playlist::Master(playlist) = parse_playlist(text).unwrap() else {
            panic!("expected master playlist");
        };
        playlist
    }

    fn invalid_playlist(text: &str) {
        assert!(
            matches!(parse_playlist(text), Err(MediaDecodeError::InvalidData(_))),
            "{text:?}"
        );
    }

    #[test]
    fn master_retains_relative_variants_bandwidth_and_codec_lists() {
        let playlist = master(concat!(
            "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-INDEPENDENT-SEGMENTS\n",
            "#EXT-X-STREAM-INF:BANDWIDTH=1280000,AVERAGE-BANDWIDTH=1000000,CODECS=\"avc1.4d401f,mp4a.40.2\",RESOLUTION=1280x720,FRAME-RATE=29.970\n",
            "../720/live.m3u8?token=a\n",
            "#EXT-X-STREAM-INF:BANDWIDTH=640000,CODECS=\"avc1.42e01e,mp4a.40.2\"\n",
            "360/live.m3u8\n"
        ));
        assert_eq!(playlist.variants.len(), 2);
        assert_eq!(playlist.variants[0].uri, "../720/live.m3u8?token=a");
        assert_eq!(playlist.variants[0].bandwidth, 1_280_000);
        assert_eq!(playlist.variants[0].average_bandwidth, Some(1_000_000));
        assert_eq!(playlist.variants[0].codecs, ["avc1.4d401f", "mp4a.40.2"]);
        assert_eq!(playlist.variants[0].resolution, Some((1280, 720)));
        assert_eq!(playlist.variants[0].frame_rate, Some(29.97));
        assert!(!playlist.variants[0].external_audio);
        assert!(playlist.independent_segments);
    }

    #[test]
    fn master_marks_external_audio_but_not_in_band_audio() {
        for (uri, external) in [(",URI=\"audio/en.m3u8\"", true), ("", false)] {
            let playlist = master(&format!(
                "#EXTM3U\n#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"a\",NAME=\"English\",DEFAULT=YES,AUTOSELECT=YES{uri}\n#EXT-X-STREAM-INF:BANDWIDTH=123,CODECS=\"avc1.4d401f,mp4a.40.2\",AUDIO=\"a\"\nvideo.m3u8\n"
            ));
            assert_eq!(playlist.variants[0].audio.as_deref(), Some("a"));
            assert_eq!(playlist.variants[0].external_audio, external);
            assert_eq!(playlist.renditions[0].uri.is_some(), external);
        }
        invalid_playlist("#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=12,AUDIO=\"missing\"\nx.m3u8\n");
    }

    #[test]
    fn live_window_sequences_durations_titles_and_discontinuities() {
        let playlist = media(concat!(
            "#EXTM3U\r\n#EXT-X-VERSION:3\r\n#EXT-X-TARGETDURATION:8\r\n",
            "#EXT-X-MEDIA-SEQUENCE:2680\r\n#EXT-X-DISCONTINUITY-SEQUENCE:7\r\n",
            "#EXTINF:7.975,First, with comma\r\n../segments/one.ts\r\n",
            "#EXT-X-DISCONTINUITY\r\n#EXTINF:7.941,Second\r\ntwo.ts\r\n"
        ));
        assert_eq!(playlist.target_duration, 8.0);
        assert_eq!(playlist.media_sequence, 2680);
        assert_eq!(playlist.segments[0].sequence, 2680);
        assert_eq!(playlist.segments[1].sequence, 2681);
        assert_eq!(playlist.segments[0].duration, 7.975);
        assert_eq!(playlist.segments[0].title, "First, with comma");
        assert_eq!(playlist.segments[0].uri, "../segments/one.ts");
        assert!(!playlist.segments[0].discontinuity);
        assert_eq!(playlist.segments[0].discontinuity_sequence, 7);
        assert!(playlist.segments[1].discontinuity);
        assert_eq!(playlist.segments[1].discontinuity_sequence, 8);
        assert!(!playlist.end_list);
        let refreshed = media(
            "#EXTM3U\n#EXT-X-TARGETDURATION:8\n#EXT-X-MEDIA-SEQUENCE:2681\n#EXTINF:8,\ntwo.ts\n#EXTINF:8,\nthree.ts",
        );
        assert_eq!(
            refreshed.segments[0].sequence,
            playlist.segments[1].sequence
        );
        assert_eq!(refreshed.segments[1].sequence, 2682);
    }

    #[test]
    fn endlist_defaults_empty_live_window_and_unknown_metadata() {
        let playlist = media(
            "#EXTM3U\n#EXT-X-TARGETDURATION:5\n#EXT-X-ENDLIST\n#comment\n#EXT-X-UNKNOWN:value\n#EXTINF:5,\na.ts\n",
        );
        assert!(playlist.end_list);
        assert_eq!(playlist.media_sequence, 0);
        assert_eq!(playlist.segments[0].sequence, 0);
        assert_eq!(playlist.version, 1);
        assert!(
            media("#EXTM3U\n#EXT-X-TARGETDURATION:5\n")
                .segments
                .is_empty()
        );
    }

    #[test]
    fn byte_ranges_resolve_only_previous_ranged_same_resource() {
        let playlist = media(
            "#EXTM3U\n#EXT-X-VERSION:4\n#EXT-X-TARGETDURATION:2\n#EXTINF:2,\n#EXT-X-BYTERANGE:20@100\nmedia.ts\n#EXTINF:2,\n#EXT-X-BYTERANGE:30\nmedia.ts\n#EXTINF:2,\nwhole.ts\n",
        );
        assert_eq!(
            playlist.segments[0].byte_range,
            Some(ByteRange {
                offset: 100,
                length: 20
            })
        );
        assert_eq!(
            playlist.segments[1].byte_range,
            Some(ByteRange {
                offset: 120,
                length: 30
            })
        );
        assert_eq!(playlist.segments[2].byte_range, None);
        for prefix in [
            "",
            "#EXTINF:2,\nmedia.ts\n",
            "#EXTINF:2,\n#EXT-X-BYTERANGE:20@100\nother.ts\n",
        ] {
            invalid_playlist(&format!(
                "#EXTM3U\n#EXT-X-VERSION:4\n#EXT-X-TARGETDURATION:2\n{prefix}#EXTINF:2,\n#EXT-X-BYTERANGE:20\nmedia.ts\n"
            ));
        }
        for range in [
            "0@0",
            "-1@0",
            "1@18446744073709551615",
            "18446744073709551616@0",
            "1@0@1",
        ] {
            invalid_playlist(&format!(
                "#EXTM3U\n#EXT-X-VERSION:4\n#EXT-X-TARGETDURATION:2\n#EXTINF:2,\n#EXT-X-BYTERANGE:{range}\nmedia.ts\n"
            ));
        }
    }

    #[test]
    fn maps_persist_share_allocation_and_replace_on_discontinuity() {
        let playlist = media(concat!(
            "#EXTM3U\n#EXT-X-VERSION:6\n#EXT-X-TARGETDURATION:2\n",
            "#EXT-X-MAP:URI=\"../init.mp4\",BYTERANGE=\"100@20\"\n",
            "#EXTINF:2,\none.m4s\n#EXTINF:2,\ntwo.m4s\n",
            "#EXT-X-DISCONTINUITY\n#EXT-X-MAP:URI=\"second-init.mp4\"\n",
            "#EXTINF:2,\nthree.m4s\n#EXT-X-ENDLIST\n"
        ));
        let first = playlist.segments[0].map.as_ref().unwrap();
        let second = playlist.segments[1].map.as_ref().unwrap();
        let third = playlist.segments[2].map.as_ref().unwrap();
        assert!(Arc::ptr_eq(first, second));
        assert!(!Arc::ptr_eq(first, third));
        assert_eq!(first.uri, "../init.mp4");
        assert_eq!(
            first.byte_range,
            Some(ByteRange {
                offset: 20,
                length: 100
            })
        );
        assert_eq!(third.uri, "second-init.mp4");
        assert_eq!(third.byte_range, None);
        assert!(playlist.segments[2].discontinuity);
        invalid_playlist(
            "#EXTM3U\n#EXT-X-VERSION:6\n#EXT-X-TARGETDURATION:2\n#EXT-X-MAP:URI=\"init.mp4\",BYTERANGE=\"100\"\n#EXTINF:2,\none.m4s\n",
        );
    }

    #[test]
    fn encryption_is_explicitly_unsupported_even_before_none_reset() {
        for method in ["AES-128", "SAMPLE-AES", "OTHER"] {
            let text = format!(
                "#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXT-X-KEY:METHOD={method},URI=\"key.bin\"\n#EXT-X-KEY:METHOD=NONE\n#EXTINF:2,\na.ts\n"
            );
            assert_eq!(parse_playlist(&text), Err(MediaDecodeError::Unsupported));
            assert_eq!(
                parse_playlist_inner(&text).unwrap_err().kind,
                HlsErrorKind::UnsupportedEncryption
            );
        }
        assert!(
            media("#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXT-X-KEY:METHOD=NONE\n#EXTINF:2,\na.ts\n")
                .segments[0]
                .map
                .is_none()
        );
        invalid_playlist("#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXT-X-KEY:METHOD=NONE,URI=\"key\"\n");
        assert_eq!(
            parse_playlist("#EXTM3U\n#EXT-X-SESSION-KEY:METHOD=AES-128,URI=\"key\"\n"),
            Err(MediaDecodeError::Unsupported)
        );
        invalid_playlist("#EXTM3U\n#EXT-X-SESSION-KEY:METHOD=NONE\n");
    }

    #[test]
    fn malformed_attributes_numbers_headers_and_unfinished_entries_fail() {
        for text in [
            "",
            "\n#EXTM3U\n",
            "\u{feff}#EXTM3U\n",
            "#EXTM3U\n#comment\0\n",
            "#EXTM3U\n#comment\rbroken\n",
            "#EXTM3U\n#EXT-X-TARGETDURATION:2\nURI.ts\n",
            "#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXTINF:2,\n",
            "#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXTINF:2\na.ts\n",
            "#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXT-X-DISCONTINUITY\n",
            "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=100\n",
            "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=100\n#EXT-X-TARGETDURATION:2\n",
            "#EXTM3U\n#EXTM3U\n",
        ] {
            invalid_playlist(text);
        }
        for attrs in [
            "BANDWIDTH=1,BANDWIDTH=2",
            "BANDWIDTH=1,",
            "BANDWIDTH=\"1\"",
            "BANDWIDTH=+1",
            "BANDWIDTH=0",
            "bandwidth=1",
            "BANDWIDTH=1,CODECS=avc1",
            "BANDWIDTH=1,CODECS=\"avc1",
            "BANDWIDTH=1,CODECS=\"avc1\"junk",
            "BANDWIDTH=1, CODECS=\"avc1\"",
            "BANDWIDTH=1,RESOLUTION=0x720",
        ] {
            invalid_playlist(&format!("#EXTM3U\n#EXT-X-STREAM-INF:{attrs}\nx.m3u8\n"));
        }
        for duration in ["NaN", "inf", "+1", "-1", "0", "1e2", "1..2"] {
            invalid_playlist(&format!(
                "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n#EXTINF:{duration},\na.ts\n"
            ));
        }
    }

    #[test]
    fn singleton_order_version_target_and_sequence_overflow_checks() {
        for tags in [
            "#EXT-X-TARGETDURATION:2\n#EXT-X-TARGETDURATION:2\n",
            "#EXT-X-VERSION:3\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n",
            "#EXT-X-TARGETDURATION:2\n#EXTINF:2,\na.ts\n#EXT-X-MEDIA-SEQUENCE:1\n",
            "#EXT-X-TARGETDURATION:2\n#EXT-X-DISCONTINUITY\n#EXT-X-DISCONTINUITY-SEQUENCE:1\n",
            "#EXT-X-TARGETDURATION:2\n#EXT-X-MEDIA-SEQUENCE:18446744073709551615\n#EXTINF:2,\na.ts\n#EXTINF:2,\nb.ts\n",
            "#EXT-X-TARGETDURATION:2\n#EXT-X-DISCONTINUITY-SEQUENCE:18446744073709551615\n#EXT-X-DISCONTINUITY\n",
            "#EXT-X-TARGETDURATION:2\n#EXT-X-ENDLIST:yes\n",
            "#EXT-X-TARGETDURATION:2\n#EXTINF:1.5,\na.ts\n",
            "#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n#EXTINF:2,\n#EXT-X-BYTERANGE:1@0\na.ts\n",
        ] {
            invalid_playlist(&format!("#EXTM3U\n{tags}"));
        }
        assert_eq!(
            media("#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:6\n#EXTINF:6.49,\na.ts\n")
                .segments[0]
                .duration,
            6.49
        );
        invalid_playlist(
            "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:6\n#EXTINF:6.5,\na.ts\n",
        );
        assert_eq!(
            parse_playlist_inner("#EXTM3U\n#EXT-X-TARGETDURATION:+2\n")
                .unwrap_err()
                .line,
            2
        );
    }

    #[test]
    fn bounded_input_attributes_variants_renditions_and_segments() {
        assert!(matches!(
            parse_playlist_inner(&"x".repeat(MAX_PLAYLIST_BYTES + 1))
                .unwrap_err()
                .kind,
            HlsErrorKind::LimitExceeded("playlist bytes")
        ));
        assert!(matches!(
            parse_playlist_inner(&format!("#EXTM3U\n#{}", "x".repeat(MAX_LINE_BYTES)))
                .unwrap_err()
                .kind,
            HlsErrorKind::LimitExceeded("line bytes")
        ));
        let attrs = (0..MAX_ATTRIBUTES)
            .map(|index| format!(",X{index}=1"))
            .collect::<String>();
        assert!(matches!(
            parse_playlist_inner(&format!(
                "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1{attrs}\nx.m3u8\n"
            ))
            .unwrap_err()
            .kind,
            HlsErrorKind::LimitExceeded("attributes")
        ));
        let text = format!(
            "#EXTM3U\n{}",
            "#EXT-X-STREAM-INF:BANDWIDTH=1\nx.m3u8\n".repeat(MAX_VARIANTS + 1)
        );
        assert!(matches!(
            parse_playlist_inner(&text).unwrap_err().kind,
            HlsErrorKind::LimitExceeded("variants")
        ));
        let text = format!(
            "#EXTM3U\n#EXT-X-TARGETDURATION:1\n{}",
            "#EXTINF:1,\na.ts\n".repeat(MAX_SEGMENTS + 1)
        );
        assert!(matches!(
            parse_playlist_inner(&text).unwrap_err().kind,
            HlsErrorKind::LimitExceeded("segments")
        ));
        let renditions = (0..=MAX_RENDITIONS)
            .map(|index| format!("#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"a\",NAME=\"{index}\"\n"))
            .collect::<String>();
        assert!(matches!(
            parse_playlist_inner(&format!("#EXTM3U\n{renditions}"))
                .unwrap_err()
                .kind,
            HlsErrorKind::LimitExceeded("renditions")
        ));
    }

    #[test]
    fn unsupported_delta_partial_segments_and_uri_variables_are_not_ignored() {
        for tag in [
            "#EXT-X-SKIP:SKIPPED-SEGMENTS=2",
            "#EXT-X-PART:DURATION=0.5,URI=\"p.ts\"",
            "#EXT-X-I-FRAMES-ONLY",
        ] {
            assert_eq!(
                parse_playlist(&format!("#EXTM3U\n{tag}\n")),
                Err(MediaDecodeError::Unsupported)
            );
        }
        assert_eq!(
            parse_playlist("#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXTINF:2,\n{$host}/a.ts\n"),
            Err(MediaDecodeError::Unsupported)
        );
    }
}
