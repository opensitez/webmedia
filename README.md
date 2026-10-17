# webmedia

`webmedia` is the media decoding layer for the webcore web browser library.
It provides bitmap, video, and web-font container decoding without a DOM,
DOM, network stack, or renderer. Thhe browser or other host owns loading,
scheduling, and presentation.

The `bitmap` module decodes PNG, JPEG, JPEG XL, GIF, WebP, and BMP bytes into
premultiplied RGBA8 pixels. Its complete-buffer API returns the first frame of
an animated file. JPEG XL uses the BSD-3-Clause `jxl` (jxl-rs) library.
`bitmap::jpeg_xl::decode_stream` accepts a non-seekable reader, reports dimensions
before pixel decoding, and publishes changed progressive pixels while keeping
the decoder state between input chunks. Output is premultiplied RGBA8 with a
bounded sample/output allocation; animation currently yields only its first frame.
The `video` module offers incremental Y4M, WebM (VP8/VP9),
and MP4 (AVC/VP8/VP9) video decoders. The clean-room AV1 decoder reconstructs
supported single-tile key, intra, and inter frames, including hidden references
and show-existing frames. It implements coefficient reconstruction, integer
transforms, intra/CFL and motion prediction, compound/inter-intra prediction,
OBMC/local warp, deblocking, CDEF, and Wiener restoration. WebM routes these
streams through the same video/media APIs, including sequence headers in codec
configuration or packets. Its current RGBA
output supports 8-bit limited-range BT.601 4:2:0; other output formats return
`Unsupported` rather than being silently converted incorrectly.
Multiple tiles, separate frame-header/tile-group OBUs, 128x128 superblocks,
superresolution, screen-content/intrabc, segmentation, quantizer matrices,
block delta loop filtering, masked compound prediction, global warped prediction,
applied film grain, and self-guided/switchable restoration remain unsupported.
Supported streams are playable, but heavy-frame decoding remains performance-limited;
AV1 support is not complete.
The `audio` module (enabled by `video`) provides Opus identification, packet
framing, range entropy decoding, and CELT pulse-vector/spreading primitives;
Vorbis identification, codec-header unpacking,
bitpacking, Huffman codebooks, lazy vector lookups, complete setup-header parsing
and floor-zero/floor-one PCM synthesis (residue formats 0/1/2, inverse MDCT, windowing and
overlap-add); and WebM audio track
types. `video::webm::WebmStream::push_packets` incrementally emits video packets
and Opus/Vorbis audio blocks through the same container parser. Audio blocks
may also carry Matroska PCM integer little-/big-endian or IEEE little-endian
float samples. Their mandatory BitDepth metadata is retained, and the same
packet lacing, timestamps, codec delay and discard padding machinery applies.
The shared container accepts Matroska's document type for these codec mappings;
this is not a claim of complete Matroska format/codec support.
retain laced packet boundaries, signed nanosecond timestamps, and discard
padding; track metadata retains codec delay and seek pre-roll.
Opus PCM synthesis supports mapping-family-zero mono/stereo,
configuration-31 packets (fullband CELT, 20 ms at 48 kHz). Other configurations,
SILK/hybrid and multistream decoding remain unsupported. Multi-frame packets
for the supported configuration decode in order through the same persistent
synthesis state; all four framing codes retain the RFC's 120 ms packet bound.
Vorbis block decoding applies WebM discard padding across laced packets.
The shared audio inverse-MDCT plan supports both Vorbis power-of-two sizes and
CELT sizes using cached mixed-radix Fourier stages. Opus synthesis building blocks
also include per-mode band layouts and stateful channel-independent de-emphasis.
The existing video-only `push` API skips audio payloads without copying them.
`WebmMediaDecoder` implements `StreamingMediaDecoder`, emitting timestamped
Vorbis and supported Opus PCM and VP8/VP9/supported AV1 video samples through one
demuxer. Audio-track failure is reported separately without halting video. The
legacy video-only decoder uses the same packet reconstruction code.
`audio::pcm::Format` converts interleaved unsigned 8-bit, signed byte-aligned
8-64-bit little-/big-endian and IEEE float32/float64 samples to normalized PCM.
It rejects partial channel frames and nonfinite floats. WAV uses this same
converter rather than maintaining a duplicate sample-conversion loop.
`WebmVideoDecoder` and its VP8-only wrapper decode at most one coded frame per
`push`, including hidden reference frames that emit no image. While
`has_buffered_samples()` is true, drain with `push(&[])` before supplying more
input or calling `finish()`. Finishing with undrained packets returns an error.
MP4/AVC decoding supports a subset of 2003 baseline and 2005 high-profile
features; unsupported streams return `MediaDecodeError::Unsupported`.
`audio::aac::AacDecoder` provides in-house AAC-LC mono/stereo PCM synthesis
for 1024- and 960-sample blocks, with spectral Huffman decoding, intensity/mid-side
stereo, perceptual noise substitution, temporal noise shaping, and persistent
long/short-window overlap. `audio::mp4::Mp4AacStream` extracts incremental MP4
access units and applies container priming and end padding. For independent
audio/video workers, `video::mp4_demux::Mp4Demux` emits both tracks from one
bounded reader; `Mp4AacDecoder` and `Mp4VideoPackets` consume its access units
without duplicating codec reconstruction. AAC Main prediction is experimental:
it uses bounded per-channel predictor banks, cached reciprocal tables, the
specified float-state quantization and reset groups, and Main's 20th-order TNS
limit. Constructed mono prediction streams and a real Main stereo clip agree
with the FFmpeg binary oracle, including nonzero 20th-order TNS. A constructed
transition out of intensity stereo differs: the oracle matches a test model
which leaves right prediction enabled on a zero residual during intensity,
rather than the standard's disabled prediction adapting from reconstructed
right-channel values. The normative rule is retained; the test-only history
diagnostic records the disagreement. Main support is not considered complete
until broader conformance coverage is established. SSR/LTP, SBR/HE-AAC, coupling-channel elements,
and program-configured multichannel layouts are not yet supported.
`Mp4Demux` supports classic sample tables and incrementally indexed fragmented
MP4 (`mvex/trex`, `moof/traf`, `tfhd/tfdt/trun`, and `mdat`) for both tracks.
Fragments commit once their headers are complete; sample payloads can arrive
later. Signed composition offsets, 64-bit base offsets, defaults, and sample
flags are checked before packets enter the existing codec workers. Fragmented
video uses `Mp4VideoPackets::new_fragmented` and `push_sample`, followed by
repeated `finish_input` calls at EOF to drain reordered pictures. A fragmented
AAC decoder uses duration zero for an unknown final trim boundary. The older
standalone file-decoder wrappers still use classic sample tables; use the shared
demux/packet APIs for fragmented input. Encrypted fragments are unsupported.
VP8 regression fixtures cover versions 0-3, hidden alternate-reference frames,
one, two, four, and eight token partitions, delta/absolute segmentation features,
segment-map persistence, odd-sized pictures, resolution-changing keyframes with
compositor-held older frames,
and byte-by-byte streaming. Filter-delta regressions cover partial updates,
disable/re-enable persistence, keyframe reset, and reconstructed-pixel checks
against a binary FFmpeg oracle. Consumed Y2 and motion entropy updates are tested
with both refresh settings. Rejected keyframes and interframes leave committed entropy,
segmentation, filter settings, and reference frames unchanged.
Arithmetic-bitstream regressions cover both signs at every coefficient-category
boundary, all planes and initial contexts, both starting offsets, and every
remaining zigzag position, including exact consumption of trailing bits.
Mixed-reference motion tests cover all sign-bias combinations, weighted neighbors,
duplicate/zero-vector ties, parsed motion modes, and pixel checks against FFmpeg.
MP4 VP8 sample playback decodes hidden alternate-reference pictures without
presenting them. Packet and incremental sample-stream regressions cover six
hidden pictures and 120 visible pictures; hidden pictures count toward
the per-push coded-sample work limit.
The public MP4 decoder is tested with both front and trailing `moov` tables,
one-byte and bulk input, exact sample-payload preservation, visible-frame pixel
and timestamp parity, metadata, and end-of-stream draining.
Reserved VP8 versions return `MediaDecodeError::Unsupported` from reconstruction
and compressed-layout parsing before entropy state changes. Raw frame-header
inspection still reports the version field.
Undefined reference-copy selectors return `MediaDecodeError::InvalidData`
without committing state or replacing reference pictures; recovery on the next
valid interframe is covered. VP8 frame headers preserve display-scale hints,
expose their rational scale factors and provide an unrounded display extent.
Reconstruction and emitted frame dimensions remain at the coded resolution.
Optional `presentation_size` on shared metadata and video frames carries the
display extent independently of prediction and reference-buffer dimensions.
VP8 hints persist across interframes, reset on accepted keyframes, and remain
unchanged when a keyframe is rejected. WebM video/media and MP4 playback expose
them. WebCore uses display dimensions for layout and object-fit, coded dimensions
for pixel buffers, and a common centered crop for native layers and capture.
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

The default `acceleration` feature uses `accelerate` for shared pixel operations
and presentation surfaces. `RasterImage::into_surface()` consumes the decoded
allocation without copying pixels; `VideoFrame::surface()` retains its existing
`Arc`. Both return the same validated `RgbaSurface`, usable by a compositor's
software drawing path or `WindowSurface::present_surface()`. Coded dimensions
remain separate from video display dimensions, and streaming previews can use
the same surface API. Conversion does not create a window or initialize a GPU;
decoding stays on its independent worker threads. Disable default features to
keep codec-only builds free of the acceleration dependency.
Use `webmedia::font::decode(&woff_bytes)` for WOFF1/WOFF2 and
`webmedia::font::eot::decode_for_page(&eot_bytes, page_url)` for EOT when
embedding rights and root-string restrictions must be checked.

## VP8 Validation

The opt-in full-stream pixel regression compares every visible YUV sample with
the FFmpeg **binary**, not a linked decoder library. It streams the oracle
through a pipe instead of creating a large raw-video file. From the repository
root, use the existing build target and a local VP8 WebM fixture:

```sh
WEBMEDIA_WEBM_SAMPLE=/absolute/path/sample.webm cargo test \
  --manifest-path crates/webmedia/Cargo.toml \
  --target-dir crates/webcore/target --release --lib --features video \
  supplied_vp8_stream_matches_binary_pixel_oracle -- --ignored --nocapture
```

FFmpeg must be available on `PATH`. The test checks complete visible-frame
output and exact YUV pixels, including reconstruction after hidden reference
frames. It is a correctness gate, not a browser frame-rate benchmark or proof
of exhaustive VP8 conformance. The separate `streams_entire_supplied_webm`
test uses the same fixture variable to check streaming delivery and draining
through end-of-stream.

Both clamping-header values have a complete minimal keyframe regression with
known neutral-gray pixels. Reserved color-space rejection must preserve the
previous references and entropy state. The optional independent binary check
for those keyframes is:

```sh
WEBMEDIA_VP8_HEADER_ORACLE=1 cargo test \
  --manifest-path crates/webmedia/Cargo.toml \
  --target-dir crates/webcore/target --lib keyframe_clamping_flags -- --nocapture
```

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
