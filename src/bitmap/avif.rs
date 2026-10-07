//! AVIF item containers (AVIF 1.0, HEIF item metadata) over the shared AV1 core.
//! Metadata and complete item extents can be used before the file finishes loading.

use super::RasterImage;
use crate::av1::{Av1Decoder, DecodedFrame, ObuStream};
use std::collections::BTreeMap;
use std::fmt;

const MAX_INPUT: usize = 64 * 1024 * 1024;
const MAX_PIXELS: usize = 16 * 1024 * 1024;
const MAX_ITEMS: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Incomplete,
    Invalid(&'static str),
    Unsupported(&'static str),
    Limit,
    Codec(crate::av1::syntax::Error),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AVIF: {self:?}")
    }
}
impl std::error::Error for Error {}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.at.checked_add(n).ok_or(Error::Limit)?;
        let out = self.bytes.get(self.at..end).ok_or(Error::Incomplete)?;
        self.at = end;
        Ok(out)
    }
    fn uint(&mut self, n: usize) -> Result<u64, Error> {
        if n > 8 {
            return Err(Error::Invalid("integer width"));
        }
        Ok(self
            .take(n)?
            .iter()
            .fold(0, |v, b| (v << 8) | u64::from(*b)))
    }
    fn full(&mut self) -> Result<(u8, u32), Error> {
        Ok((self.uint(1)? as u8, self.uint(3)? as u32))
    }
}

struct BoxRef<'a> {
    kind: [u8; 4],
    data: &'a [u8],
    payload_offset: usize,
}
fn boxes(mut data: &[u8], mut offset: usize) -> Result<Vec<BoxRef<'_>>, Error> {
    let mut out = Vec::new();
    while !data.is_empty() {
        if out.len() == MAX_ITEMS {
            return Err(Error::Limit);
        }
        let mut r = Reader::new(data);
        let size = r.uint(4)?;
        let kind = r.take(4)?.try_into().unwrap();
        let size = match size {
            0 => data.len(),
            1 => usize::try_from(r.uint(8)?).map_err(|_| Error::Limit)?,
            n => n as usize,
        };
        if size < r.at {
            return Err(Error::Invalid("box size"));
        }
        let complete = data.get(..size).ok_or(Error::Incomplete)?;
        out.push(BoxRef {
            kind,
            data: &complete[r.at..],
            payload_offset: offset + r.at,
        });
        offset = offset.checked_add(size).ok_or(Error::Limit)?;
        data = &data[size..];
    }
    Ok(out)
}

pub fn is_avif(bytes: &[u8]) -> bool {
    let Some(header) = bytes.get(..16) else {
        return false;
    };
    if &header[4..8] != b"ftyp" {
        return false;
    }
    let size = u32::from_be_bytes(header[..4].try_into().unwrap()) as usize;
    if size < 16 {
        return false;
    }
    &header[8..12] == b"avif"
        || &header[8..12] == b"avis"
        || bytes
            .get(16..size.min(bytes.len()))
            .is_some_and(|brands| brands.chunks_exact(4).any(|b| b == b"avif" || b == b"avis"))
}

#[derive(Clone, Default)]
struct Item {
    kind: [u8; 4],
    method: u8,
    extents: Vec<(usize, usize)>,
    properties: Vec<(usize, bool)>,
    refs: Vec<([u8; 4], Vec<u32>)>,
}
#[derive(Clone)]
enum Property {
    Size(u32, u32),
    Config(Vec<u8>),
    Color(u16, u16, u16, bool),
    Alpha,
    Rotate(u8),
    Mirror(u8),
    Known,
    Unknown,
}
struct Index {
    primary: u32,
    items: BTreeMap<u32, Item>,
    properties: Vec<Property>,
    idat: Option<(usize, usize)>,
}
impl Index {
    fn item(&self, id: u32) -> Result<&Item, Error> {
        self.items.get(&id).ok_or(Error::Invalid("missing item"))
    }
    fn dimensions(&self, id: u32) -> Result<(u32, u32), Error> {
        for &(p, _) in &self.item(id)?.properties {
            if let Some(Property::Size(w, h)) = self.properties.get(p) {
                return Ok((*w, *h));
            }
        }
        Err(Error::Invalid("missing spatial extents"))
    }
    fn display_dimensions(&self, id: u32) -> Result<(u32, u32), Error> {
        let (mut width, mut height) = self.dimensions(id)?;
        for &(property, _) in &self.item(id)?.properties {
            if matches!(self.properties.get(property), Some(Property::Rotate(turns)) if turns % 2 == 1) {
                std::mem::swap(&mut width, &mut height);
            }
        }
        Ok((width, height))
    }
    fn payload(&self, bytes: &[u8], id: u32) -> Result<Vec<u8>, Error> {
        let item = self.item(id)?;
        let base = match item.method {
            0 => 0,
            1 => self.idat.ok_or(Error::Invalid("missing idat"))?.0,
            _ => return Err(Error::Unsupported("item construction method")),
        };
        let total = item.extents.iter().try_fold(0usize, |sum, (_, n)| {
            sum.checked_add(*n)
                .filter(|n| *n <= MAX_INPUT)
                .ok_or(Error::Limit)
        })?;
        let mut out = Vec::with_capacity(total);
        for &(offset, len) in &item.extents {
            if item.method == 1 && offset.checked_add(len).is_none_or(|end| end > self.idat.unwrap().1) {
                return Err(Error::Invalid("extent outside idat"));
            }
            let start = base.checked_add(offset).ok_or(Error::Limit)?;
            let end = start.checked_add(len).ok_or(Error::Limit)?;
            out.extend_from_slice(bytes.get(start..end).ok_or(Error::Incomplete)?);
        }
        if out.is_empty() {
            return Err(Error::Invalid("empty item"));
        }
        Ok(out)
    }
}

fn property(b: &BoxRef<'_>) -> Result<Property, Error> {
    let mut r = Reader::new(b.data);
    Ok(match &b.kind {
        b"ispe" => {
            r.full()?;
            let w = r.uint(4)? as u32;
            let h = r.uint(4)? as u32;
            pixel_count(w, h)?;
            Property::Size(w, h)
        }
        b"av1C" => {
            if r.uint(1)? != 0x81 {
                return Err(Error::Invalid("AV1 configuration version"));
            }
            r.take(3)?;
            Property::Config(b.data[4..].to_vec())
        }
        b"colr" => {
            if r.take(4)? != b"nclx" {
                return Ok(Property::Unknown);
            }
            Property::Color(
                r.uint(2)? as u16,
                r.uint(2)? as u16,
                r.uint(2)? as u16,
                r.uint(1)? & 128 != 0,
            )
        }
        b"auxC" => {
            r.full()?;
            if r.bytes[r.at..].split(|b| *b == 0).next()
                == Some(b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha".as_slice())
            {
                Property::Alpha
            } else {
                Property::Unknown
            }
        }
        b"irot" => Property::Rotate(r.uint(1)? as u8 & 3),
        b"imir" => Property::Mirror(r.uint(1)? as u8 & 1),
        b"pixi" => {
            r.full()?;
            let n = r.uint(1)? as usize;
            r.take(n)?;
            Property::Known
        }
        b"pasp" => {
            if r.uint(4)? != r.uint(4)? {
                return Err(Error::Unsupported("non-square pixels"));
            }
            Property::Known
        }
        _ => Property::Unknown,
    })
}

fn parse_meta(data: &[u8], offset: usize) -> Result<Index, Error> {
    let mut r = Reader::new(data);
    if r.full()?.0 != 0 {
        return Err(Error::Unsupported("meta version"));
    }
    let children = boxes(&data[4..], offset + 4)?;
    let mut index = Index {
        primary: 0,
        items: BTreeMap::new(),
        properties: Vec::new(),
        idat: None,
    };
    for b in &children {
        let mut r = Reader::new(b.data);
        match &b.kind {
            b"pitm" => {
                let v = r.full()?.0;
                if v > 1 {
                    return Err(Error::Unsupported("primary item version"));
                }
                index.primary = r.uint(if v == 0 { 2 } else { 4 })? as u32;
            }
            b"idat" => index.idat = Some((b.payload_offset, b.data.len())),
            b"iinf" => {
                let v = r.full()?.0;
                let count = r.uint(if v == 0 { 2 } else { 4 })? as usize;
                if count > MAX_ITEMS {
                    return Err(Error::Limit);
                }
                for info in boxes(&b.data[r.at..], b.payload_offset + r.at)? {
                    if &info.kind != b"infe" {
                        continue;
                    }
                    let mut q = Reader::new(info.data);
                    let v = q.full()?.0;
                    if !matches!(v, 2 | 3) {
                        return Err(Error::Unsupported("item info version"));
                    }
                    let id = q.uint(if v == 2 { 2 } else { 4 })? as u32;
                    if q.uint(2)? != 0 {
                        return Err(Error::Unsupported("protected item"));
                    }
                    index.items.entry(id).or_default().kind = q.take(4)?.try_into().unwrap();
                }
            }
            b"iloc" => {
                let v = r.full()?.0;
                if v > 2 {
                    return Err(Error::Unsupported("item location version"));
                }
                let a = r.uint(1)? as usize;
                let b = r.uint(1)? as usize;
                let (off, len, base, ext_index) =
                    (a >> 4, a & 15, b >> 4, if v == 0 { 0 } else { b & 15 });
                let count = r.uint(if v == 2 { 4 } else { 2 })? as usize;
                if count > MAX_ITEMS {
                    return Err(Error::Limit);
                }
                for _ in 0..count {
                    let id = r.uint(if v == 2 { 4 } else { 2 })? as u32;
                    let method = if v == 0 { 0 } else { (r.uint(2)? & 15) as u8 };
                    if r.uint(2)? != 0 {
                        return Err(Error::Unsupported("external item data"));
                    }
                    let base = r.uint(base)?;
                    let n = r.uint(2)? as usize;
                    if n > MAX_ITEMS {
                        return Err(Error::Limit);
                    }
                    let item = index.items.entry(id).or_default();
                    item.method = method;
                    for _ in 0..n {
                        r.uint(ext_index)?;
                        let offset = base.checked_add(r.uint(off)?).ok_or(Error::Limit)?;
                        let length = r.uint(len)?;
                        if length == 0 {
                            return Err(Error::Unsupported("unspecified extent length"));
                        }
                        item.extents.push((
                            usize::try_from(offset).map_err(|_| Error::Limit)?,
                            usize::try_from(length).map_err(|_| Error::Limit)?,
                        ));
                    }
                }
            }
            b"iprp" => {
                let props = boxes(b.data, b.payload_offset)?;
                for p in &props {
                    if &p.kind == b"ipco" {
                        for p in boxes(p.data, p.payload_offset)? {
                            index.properties.push(property(&p)?);
                        }
                    }
                }
                for p in &props {
                    if &p.kind != b"ipma" {
                        continue;
                    }
                    let mut q = Reader::new(p.data);
                    let (v, flags) = q.full()?;
                    if v > 1 {
                        return Err(Error::Unsupported("property association version"));
                    }
                    let count = q.uint(4)? as usize;
                    if count > MAX_ITEMS {
                        return Err(Error::Limit);
                    }
                    for _ in 0..count {
                        let id = q.uint(if v == 0 { 2 } else { 4 })? as u32;
                        let n = q.uint(1)? as usize;
                        for _ in 0..n {
                            let wide = flags & 1 != 0;
                            let value = q.uint(if wide { 2 } else { 1 })? as usize;
                            let essential = value & if wide { 0x8000 } else { 0x80 } != 0;
                            let p = value & if wide { 0x7fff } else { 0x7f };
                            if p != 0 {
                                index
                                    .items
                                    .entry(id)
                                    .or_default()
                                    .properties
                                    .push((p - 1, essential));
                            }
                        }
                    }
                }
            }
            b"iref" => {
                let v = r.full()?.0;
                if v > 1 {
                    return Err(Error::Unsupported("item reference version"));
                }
                for reference in boxes(&b.data[4..], b.payload_offset + 4)? {
                    let mut q = Reader::new(reference.data);
                    let width = if v == 0 { 2 } else { 4 };
                    let from = q.uint(width)? as u32;
                    let n = q.uint(2)? as usize;
                    if n > MAX_ITEMS {
                        return Err(Error::Limit);
                    }
                    let mut to = Vec::with_capacity(n);
                    for _ in 0..n {
                        to.push(q.uint(width)? as u32);
                    }
                    index
                        .items
                        .entry(from)
                        .or_default()
                        .refs
                        .push((reference.kind, to));
                }
            }
            _ => {}
        }
    }
    if index.primary == 0 {
        return Err(Error::Invalid("missing primary item"));
    }
    if index.items.len() > MAX_ITEMS {
        return Err(Error::Limit);
    }
    index.dimensions(index.primary)?;
    Ok(index)
}

fn find_index(bytes: &[u8]) -> Result<Index, Error> {
    let mut offset = 0usize;
    for _ in 0..MAX_ITEMS {
        let mut r = Reader::new(bytes.get(offset..).ok_or(Error::Incomplete)?);
        let size = r.uint(4)?;
        let kind = r.take(4)?;
        let size = match size {
            0 => bytes.len() - offset,
            1 => usize::try_from(r.uint(8)?).map_err(|_| Error::Limit)?,
            n => n as usize,
        };
        if size < r.at {
            return Err(Error::Invalid("box size"));
        }
        let end = offset.checked_add(size).ok_or(Error::Limit)?;
        if kind == b"meta" {
            return parse_meta(
                bytes.get(offset + r.at..end).ok_or(Error::Incomplete)?,
                offset + r.at,
            );
        }
        if end > bytes.len() {
            return Err(Error::Incomplete);
        }
        offset = end;
    }
    Err(Error::Limit)
}

fn pixel_count(w: u32, h: u32) -> Result<usize, Error> {
    let n = (w as usize).checked_mul(h as usize).ok_or(Error::Limit)?;
    if n == 0 || n > MAX_PIXELS {
        return Err(Error::Limit);
    }
    Ok(n)
}

fn coded_frame(index: &Index, bytes: &[u8], id: u32) -> Result<DecodedFrame, Error> {
    let item = index.item(id)?;
    if &item.kind != b"av01" {
        return Err(Error::Unsupported("derived image item"));
    }
    let mut decoder = Av1Decoder::new();
    let mut config_found = false;
    for &(p, essential) in &item.properties {
        let property = index
            .properties
            .get(p)
            .ok_or(Error::Invalid("property index"))?;
        if essential && matches!(property, Property::Unknown) {
            return Err(Error::Unsupported("essential image property"));
        }
        if let Property::Config(config) = property {
            config_found = true;
            let mut stream = ObuStream::new();
            for obu in stream
                .push(config)
                .map_err(|_| Error::Invalid("configuration OBU"))?
            {
                decoder.decode_obu(&obu).map_err(Error::Codec)?;
            }
            stream
                .finish()
                .map_err(|_| Error::Invalid("configuration OBU framing"))?;
        }
    }
    if !config_found {
        return Err(Error::Invalid("missing AV1 configuration"));
    }
    let coded = index.payload(bytes, id)?;
    let mut stream = ObuStream::new();
    let mut frame = None;
    for obu in stream
        .push(&coded)
        .map_err(|_| Error::Invalid("item OBU framing"))?
    {
        if let Some(decoded) = decoder.decode_obu(&obu).map_err(Error::Codec)? {
            if frame.replace(decoded).is_some() {
                return Err(Error::Unsupported("multiple display frames in an item"));
            }
        }
    }
    stream
        .finish()
        .map_err(|_| Error::Invalid("truncated item OBU"))?;
    let frame = frame.ok_or(Error::Invalid("item has no display frame"))?;
    if (frame.header.width, frame.header.height) != index.dimensions(id)? {
        return Err(Error::Invalid("coded dimensions disagree with ispe"));
    }
    Ok(frame)
}

fn rgba(frame: &DecodedFrame, color: Option<(u16, u16, u16, bool)>) -> Result<RasterImage, Error> {
    let s = &frame.sequence;
    let (primaries, transfer, matrix, full) = color.unwrap_or((
        s.color_primaries as u16,
        s.transfer_characteristics as u16,
        s.matrix_coefficients as u16,
        s.full_range,
    ));
    if !matches!(primaries, 1 | 2 | 5 | 6) || !matches!(transfer, 1 | 2 | 6 | 13) {
        return Err(Error::Unsupported("color primaries or transfer function"));
    }
    let w = frame.header.width;
    let h = frame.header.height;
    let n = pixel_count(w, h)?;
    if !matches!(frame.bit_depth, 8 | 10 | 12) {
        return Err(Error::Unsupported("sample precision"));
    }
    let (kr, kb) = match matrix {
        0 | 1 => (0.2126, 0.0722),
        2 | 5 | 6 => (0.299, 0.114),
        _ => return Err(Error::Unsupported("color matrix")),
    };
    let scale = (1u32 << (frame.bit_depth - 8)) as f64;
    let maximum = ((1u32 << frame.bit_depth) - 1) as f64;
    let sample = |plane: usize, x: usize, y: usize| -> Result<f64, Error> {
        let p = frame
            .planes
            .get(plane)
            .ok_or(Error::Invalid("missing plane"))?;
        let offset = y
            .checked_mul(p.stride)
            .and_then(|n| n.checked_add(x))
            .ok_or(Error::Limit)?;
        if x >= p.width || y >= p.height {
            return Err(Error::Invalid("plane dimensions"));
        }
        let v = *p
            .samples
            .get(offset)
            .ok_or(Error::Invalid("plane storage"))? as f64;
        if v > maximum {
            return Err(Error::Invalid("sample range"));
        }
        Ok(v)
    };
    let mut pixels = Vec::with_capacity(n * 4);
    for y in 0..h as usize {
        for x in 0..w as usize {
            let raw_y = sample(0, x, y)?;
            let luma = if full {
                raw_y / maximum
            } else {
                (raw_y - 16.0 * scale) / (219.0 * scale)
            };
            let rgb = if s.monochrome {
                [luma; 3]
            } else {
                let cx = x >> usize::from(s.subsampling_x);
                let cy = y >> usize::from(s.subsampling_y);
                let u = sample(1, cx, cy)?;
                let v = sample(2, cx, cy)?;
                if matrix == 0 {
                    if !full {
                        return Err(Error::Unsupported("limited-range identity matrix"));
                    }
                    [v / maximum, raw_y / maximum, u / maximum]
                } else {
                    let midpoint = (1u32 << (frame.bit_depth - 1)) as f64;
                    let divisor = if full { maximum } else { 224.0 * scale };
                    let cb = (u - midpoint) / divisor;
                    let cr = (v - midpoint) / divisor;
                    [
                        luma + 2.0 * (1.0 - kr) * cr,
                        luma - 2.0 * kb * (1.0 - kb) / (1.0 - kr - kb) * cb
                            - 2.0 * kr * (1.0 - kr) / (1.0 - kr - kb) * cr,
                        luma + 2.0 * (1.0 - kb) * cb,
                    ]
                }
            };
            pixels.extend(rgb.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));
            pixels.push(255);
        }
    }
    Ok(RasterImage {
        width: w,
        height: h,
        rgba: pixels,
    })
}

fn decode_index(index: &Index, bytes: &[u8]) -> Result<RasterImage, Error> {
    let frame = coded_frame(index, bytes, index.primary)?;
    let item = index.item(index.primary)?;
    let color = item
        .properties
        .iter()
        .find_map(|&(p, _)| match index.properties.get(p) {
            Some(Property::Color(a, b, c, d)) => Some((*a, *b, *c, *d)),
            _ => None,
        });
    let mut image = rgba(&frame, color)?;
    for (&id, auxiliary) in &index.items {
        if !auxiliary
            .refs
            .iter()
            .any(|(kind, to)| kind == b"auxl" && to.contains(&index.primary))
        {
            continue;
        }
        if !auxiliary
            .properties
            .iter()
            .any(|&(p, _)| matches!(index.properties.get(p), Some(Property::Alpha)))
        {
            continue;
        }
        let alpha = coded_frame(index, bytes, id)?;
        if !alpha.sequence.monochrome
            || (alpha.header.width, alpha.header.height) != (image.width, image.height)
        {
            return Err(Error::Invalid("alpha plane dimensions"));
        }
        let plane = alpha
            .planes
            .first()
            .ok_or(Error::Invalid("missing alpha plane"))?;
        let maximum = (1u32 << alpha.bit_depth) - 1;
        for y in 0..image.height as usize {
            for x in 0..image.width as usize {
                let a = *plane
                    .samples
                    .get(y * plane.stride + x)
                    .ok_or(Error::Invalid("alpha storage"))? as u32;
                let a = ((a * 255 + maximum / 2) / maximum) as u8;
                let pixel = &mut image.rgba[(y * image.width as usize + x) * 4..][..4];
                for value in &mut pixel[..3] {
                    *value = ((*value as u32 * a as u32 + 127) / 255) as u8;
                }
                pixel[3] = a;
            }
        }
    }
    for &(p, _) in &item.properties {
        let Some(property) = index.properties.get(p) else {
            return Err(Error::Invalid("property index"));
        };
        let (rotation, mirror) = match property {
            Property::Rotate(n) => (*n, None),
            Property::Mirror(axis) => (0, Some(*axis)),
            _ => continue,
        };
        let (w, h) = (image.width as usize, image.height as usize);
        let (nw, nh) = if rotation % 2 == 1 { (h, w) } else { (w, h) };
        let mut out = vec![0; image.rgba.len()];
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = match rotation {
                    1 => (y, w - 1 - x),
                    2 => (w - 1 - x, h - 1 - y),
                    3 => (h - 1 - y, x),
                    _ => match mirror {
                        Some(0) => (w - 1 - x, y),
                        Some(1) => (x, h - 1 - y),
                        _ => (x, y),
                    },
                };
                out[(dy * nw + dx) * 4..][..4].copy_from_slice(&image.rgba[(y * w + x) * 4..][..4]);
            }
        }
        image = RasterImage {
            width: nw as u32,
            height: nh as u32,
            rgba: out,
        };
    }
    Ok(image)
}

#[derive(Default)]
pub struct AvifStream {
    bytes: Vec<u8>,
    index: Option<Index>,
    done: bool,
    dimensions_reported: bool,
}
impl AvifStream {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        self.index
            .as_ref()
            .and_then(|i| i.display_dimensions(i.primary).ok())
    }
    pub fn push(&mut self, chunk: &[u8]) -> Result<Option<RasterImage>, Error> {
        self.push_with_dimensions(chunk, |_, _| {})
    }
    pub fn push_with_dimensions(&mut self, chunk: &[u8], mut on_dimensions: impl FnMut(u32,u32)) -> Result<Option<RasterImage>, Error> {
        if self.done {
            return Ok(None);
        }
        if self
            .bytes
            .len()
            .checked_add(chunk.len())
            .is_none_or(|n| n > MAX_INPUT)
        {
            return Err(Error::Limit);
        }
        self.bytes.extend_from_slice(chunk);
        if self.index.is_none() {
            match find_index(&self.bytes) {
                Ok(index) => self.index = Some(index),
                Err(Error::Incomplete) => return Ok(None),
                Err(error) => return Err(error),
            }
        }
        let index = self.index.as_ref().unwrap();
        if !is_avif(&self.bytes) { return Err(Error::Invalid("AVIF file type")); }
        if !self.dimensions_reported {
            let (w,h) = index.display_dimensions(index.primary)?;
            on_dimensions(w,h);
            self.dimensions_reported = true;
        }
        // Do not retry expensive pixel decoding until every referenced extent is available.
        for (&id,item) in index.items.iter().filter(|(id,item)| **id == index.primary || item.refs.iter().any(|(kind,to)| kind == b"auxl" && to.contains(&index.primary))) {
            if id != index.primary && !item.properties.iter().any(|&(p,_)| matches!(index.properties.get(p),Some(Property::Alpha))) { continue; }
            let base = if item.method == 1 {
                index.idat.ok_or(Error::Invalid("missing idat"))?.0
            } else {
                0
            };
            for &(offset, len) in &item.extents {
                if base
                    .checked_add(offset)
                    .and_then(|n| n.checked_add(len))
                    .ok_or(Error::Limit)?
                    > self.bytes.len()
                {
                    return Ok(None);
                }
            }
        }
        let image = decode_index(index, &self.bytes)?;
        self.done = true;
        self.bytes.clear();
        self.bytes.shrink_to_fit();
        Ok(Some(image))
    }
}

pub fn decode_stream(mut source: impl std::io::Read, mut on_dimensions: impl FnMut(u32,u32)) -> Result<RasterImage, Error> {
    let mut stream = AvifStream::new();
    let mut chunk = [0; 16 * 1024];
    loop {
        let count = source.read(&mut chunk).map_err(|_| Error::Invalid("input read"))?;
        if count == 0 { return Err(Error::Incomplete); }
        if let Some(image) = stream.push_with_dimensions(&chunk[..count], &mut on_dimensions)? { return Ok(image); }
    }
}

pub fn decode(bytes: &[u8]) -> Result<RasterImage, Error> {
    if bytes.len() > MAX_INPUT {
        return Err(Error::Limit);
    }
    if !is_avif(bytes) {
        return Err(Error::Invalid("AVIF file type"));
    }
    decode_index(&find_index(bytes)?, bytes)
}

/// Parsed primary image coding information, without reconstructing pixels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub tile_count: usize,
    pub superblock_size: u16,
    pub superresolution: bool,
    pub screen_content: bool,
    pub segmentation: bool,
}

pub fn inspect(bytes: &[u8]) -> Result<ImageInfo, Error> {
    if !is_avif(bytes) { return Err(Error::Invalid("AVIF file type")); }
    let index = find_index(bytes)?;
    let coded = index.payload(bytes, index.primary)?;
    let mut sequence = None;
    let mut stream = ObuStream::new();
    for obu in stream.push(&coded).map_err(|_| Error::Invalid("item OBU framing"))? {
        if obu.kind == 1 {
            sequence = Some(crate::av1::syntax::SequenceHeader::parse(&obu.payload).map_err(Error::Codec)?);
        } else if obu.kind == 6 {
            let sequence = sequence.as_ref().ok_or(Error::Invalid("frame before sequence"))?;
            let frame = crate::av1::CodedIntraFrame::parse(&obu, sequence).map_err(Error::Codec)?;
            return Ok(ImageInfo { width: frame.header.width, height: frame.header.height,
                bit_depth: sequence.bit_depth, tile_count: frame.header.tiles.count(),
                superblock_size: if sequence.use_128x128_superblock { 128 } else { 64 },
                superresolution: frame.header.width != frame.header.upscaled_width,
                screen_content: frame.header.allow_screen_content_tools,
                segmentation: frame.header.segmentation_enabled });
        }
    }
    Err(Error::Invalid("missing primary frame"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streaming_dimensions_follow_ordered_orientation() {
        let mut index = Index {
            primary: 1,
            items: BTreeMap::from([(1, Item {
                properties: vec![(0, true), (1, true), (2, true)],
                ..Item::default()
            })]),
            properties: vec![Property::Size(320, 180), Property::Mirror(0), Property::Rotate(1)],
            idat: None,
        };
        assert_eq!(index.dimensions(1).unwrap(), (320, 180));
        assert_eq!(index.display_dimensions(1).unwrap(), (180, 320));
        index.properties.push(Property::Rotate(3));
        index.items.get_mut(&1).unwrap().properties.push((3, true));
        let stream = AvifStream { index: Some(index), ..AvifStream::default() };
        assert_eq!(stream.dimensions(), Some((320, 180)));
    }
    #[test]
    fn malformed_boxes_never_panic() {
        for length in 0..128 {
            assert!(find_index(&vec![255; length]).is_err());
        }
        assert!(boxes(&[0, 0, 0, 4, b'm', b'e', b't', b'a'], 0).is_err());
    }
    #[test]
    #[ignore = "requires WEBMEDIA_AVIF_SAMPLE; exercises the actual local image"]
    fn local_avif_sample() {
        let path = std::env::var_os("WEBMEDIA_AVIF_SAMPLE").expect("set WEBMEDIA_AVIF_SAMPLE");
        let bytes = std::fs::read(path).unwrap();
        eprintln!("AVIF metadata: {:?}", inspect(&bytes).unwrap());
        let index = find_index(&bytes).unwrap();
        let frame = coded_frame(&index, &bytes, index.primary).unwrap();
        if let Some(path) = std::env::var_os("WEBMEDIA_AVIF_YUV_ORACLE") {
            let expected = std::fs::read(path).unwrap();
            assert_eq!(frame.bit_depth, 8, "oracle uses 8-bit native planar samples");
            let mut offset = 0;
            for (number, plane) in frame.planes.iter().enumerate() {
                for row in 0..plane.height {
                    let actual = &plane.samples[row * plane.stride..][..plane.width];
                    let reference = expected.get(offset..offset + plane.width).expect("oracle length");
                    let mismatch = actual.iter().zip(reference).position(|(&a, &b)| a != u16::from(b));
                    assert_eq!(mismatch, None, "native plane {number}, row {row}");
                    offset += plane.width;
                }
            }
            assert_eq!(offset, expected.len());
        }
        eprintln!(
            "AVIF {}x{} depth={} matrix={} full={}",
            frame.header.width,
            frame.header.height,
            frame.bit_depth,
            frame.sequence.matrix_coefficients,
            frame.sequence.full_range
        );
        let image = decode(&bytes).unwrap();
        assert_eq!(
            image.rgba.len(),
            image.width as usize * image.height as usize * 4
        );
        let mut streamed = AvifStream::new();
        let mut output = None;
        for chunk in bytes.chunks(113) {
            if let Some(image) = streamed.push(chunk).unwrap() {
                assert!(output.replace(image).is_none());
            }
        }
        assert_eq!(output, Some(image));
    }
}
