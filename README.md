# webmedia

`webmedia` provides bitmap, video, and web-font container decoding without a
DOM, network stack, or renderer. A browser or other host owns loading,
scheduling, and presentation.

The `bitmap` module decodes PNG, JPEG, GIF, WebP, and BMP bytes into
premultiplied RGBA8 pixels. Its complete-buffer API returns the first frame of
an animated file. The `video` module offers incremental Y4M and MP4/AVC video
decoders, plus AV1 OBU framing. AV1 framing is **not** an AV1 pixel decoder.
MP4/AVC decoding supports a subset of 2003 baseline and 2005 high-profile
features; unsupported streams return `MediaDecodeError::Unsupported`.
The optional `font` module decodes WOFF1 and WOFF2 to sfnt bytes and provides
EOT parsing and page-scoped decoding, including MTX-compressed EOT.

SVG is not included yet. WebCore's current SVG parser, animation, and painter
are coupled to its CSS, canvas, and DOM; they need to be separated before an
independent `webmedia::svg` module can be published.

**Dual-licensed:** GPLv3-or-later for compatible open-source projects, or a
separate commercial license for proprietary use. See [License](#license).

## Quick Start

```toml
[dependencies]
webmedia = "0.1"
```

```rust
use webmedia::video::{StreamingVideoDecoder, y4m::Y4mStream};

let mut decoder = Y4mStream::new();
let frames = decoder.push(b"YUV4MPEG2 W2 H2 F25:1 C420\nFRAME\n\x10\x10\x10\x10\x80\x80")?;
assert_eq!((frames[0].width, frames[0].height), (2, 2));
decoder.finish()?;
# Ok::<(), webmedia::video::MediaDecodeError>(())
```

For bitmap decoding, pass complete encoded bytes to
`webmedia::bitmap::decode_raster`. The returned pixels are row-major,
premultiplied RGBA8. Feed incremental video bytes to `push`; it returns frames
as they become available. Call `finish` at end of input to detect truncation.

`default-features = false` with `features = ["bitmap"]`,
`features = ["video"]`, or `features = ["font"]` keeps dependencies limited
to the required formats. `audio-symphonia` enables the optional audio backend.
Font selection, shaping, and text layout belong to the embedding application.
Use `webmedia::font::decode(&woff_bytes)` for WOFF1/WOFF2 and
`webmedia::font::eot::decode_for_page(&eot_bytes, page_url)` for EOT when
embedding rights and root-string restrictions must be checked.

## Copyright

Copyright (c) 2026 OpenSitez.com and Youness El Andaloussi. All rights reserved.

## License

`webmedia` is **dual-licensed**. Choose the license that fits your project:

**Open source (GPLv3 or later).** If you are building an open-source application
under a compatible license, you may use `webmedia` under the terms of the
[GNU General Public License v3.0](LICENSE-GPL) or later.

**Commercial license.** To use `webmedia` in a proprietary, closed-source
product without the requirements of GPLv3, you must purchase a separate
commercial license. Contact OpenSitez.com for pricing and terms.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `webmedia` by you, as defined in the Apache-2.0 license,
shall be dedicated to the public domain (or equivalent, such as the CC0 1.0
Universal public domain dedication). This allows contributions to be used in
both the GPLv3 and commercial releases.
