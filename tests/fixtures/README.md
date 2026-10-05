# VP8 and VP9 Fixtures

`vp8-opus-startup.webm` is a 6,174-byte synthetic audio/video fixture for
playback startup, failure, and clock handoff tests. It contains a 32x32 red
picture and a 440 Hz tone, with stereo 48 kHz Opus and two VP8 pictures.
The FFmpeg binary generated the fixture; tests use the in-house decoders and
do not require FFmpeg at runtime. Regenerate from this directory with:

```sh
ffmpeg -hide_banner -loglevel error \
  -f lavfi -i color=c=red:s=32x32:r=10:d=0.2 \
  -f lavfi -i sine=frequency=440:sample_rate=48000:duration=0.2 \
  -c:v libvpx -deadline best -cpu-used 0 -b:v 20k -g 2 \
  -c:a libopus -application audio -b:a 192k -vbr off \
  -frame_duration 20 -ac 2 -shortest -n vp8-opus-startup.webm
```

`vp8-empty-opus-track.webm` contains the same two VP8 pictures and a declared
stereo Opus track with no audio packets (802 bytes). It checks that empty audio
does not stall video or cause fabricated PCM. Generate it with the command
above, adding `-af aselect=0` and replacing `-shortest` with `-t 0.2`, using
`vp8-empty-opus-track.webm` as the output name.

`vp8-motion.webm` and `vp8-motion.yuv` exercise VP8 keyframes, interframe
references, mode-dependent filter deltas, and zero-level deblocking. All 60
displayed frames are compared pixel-exactly against the reference decoded by
the FFmpeg binary. Regenerate them with:

```sh
ffmpeg -f lavfi -i testsrc2=size=160x96:rate=30 -frames:v 60 \
  -c:v libvpx -deadline good -cpu-used 2 -b:v 350k -g 60 \
  -auto-alt-ref 1 -lag-in-frames 25 -an -y vp8-motion.webm
ffmpeg -i vp8-motion.webm -pix_fmt yuv420p -f rawvideo -y vp8-motion.yuv
```

`vp8-edges.webm` uses a 160x90 visible crop in 160x96 coded planes, exercising
motion prediction into padded reference rows. All 30 frames are compared exactly:

```sh
ffmpeg -f lavfi -i testsrc2=size=160x90:rate=30 -frames:v 30 \
  -c:v libvpx -deadline good -cpu-used 2 -b:v 350k -g 30 -an -y vp8-edges.webm
ffmpeg -i vp8-edges.webm -pix_fmt yuv420p -f rawvideo -y vp8-edges.yuv
```

`vp8-altref.webm` contains 126 coded frames (120 displayed and 6 hidden), each
with four token partitions. All displayed frames match `vp8-altref.yuv` exactly.
Both WebM decoder entry points also check identical output and timestamps when
fed chunks of 1, 17, 257, or 4096 bytes, with output before the transfer ends.

```sh
ffmpeg -f lavfi -i testsrc2=size=160x96:rate=30 -frames:v 120 \
  -c:v libvpx -b:v 250k -pass 1 -passlogfile vp8-altref \
  -auto-alt-ref 1 -lag-in-frames 25 -slices 4 -f webm -y /dev/null
ffmpeg -f lavfi -i testsrc2=size=160x96:rate=30 -frames:v 120 \
  -c:v libvpx -b:v 250k -pass 2 -passlogfile vp8-altref \
  -auto-alt-ref 1 -lag-in-frames 25 -slices 4 -y vp8-altref.webm
ffmpeg -i vp8-altref.webm -pix_fmt yuv420p -f rawvideo -y vp8-altref.yuv
```

`vp8-version1`, `vp8-version2`, and `vp8-version3` exercise the low-complexity
versions, including simple deblocking and version-3 whole-pixel chroma vectors.
Every pixel in all 30 frames is compared with the corresponding `.yuv` file.
Regenerate each pair with `VERSION` set to 1, 2, or 3:

```sh
ffmpeg -f lavfi -i testsrc2=size=160x90:rate=30 -frames:v 30 \
  -c:v libvpx -profile:v "$VERSION" -b:v 250k -g 30 -an \
  -y "vp8-version$VERSION.webm"
ffmpeg -i "vp8-version$VERSION.webm" -pix_fmt yuv420p -f rawvideo \
  -y "vp8-version$VERSION.yuv"
```

`vp8-odd-edges.webm` checks 163x91 visible dimensions, eight token partitions,
error-resilient partition coding, and maximum filter sharpness. All 20 frames
match the reference exactly:

```sh
ffmpeg -f lavfi -i testsrc=size=163x91:rate=15 -frames:v 20 \
  -pix_fmt yuv420p -c:v libvpx -b:v 250k -g 20 -slices 8 \
  -error-resilient partitions -sharpness 7 -an -y vp8-odd-edges.webm
ffmpeg -i vp8-odd-edges.webm -pix_fmt yuv420p -f rawvideo -y vp8-odd-edges.yuv
```

`vp9-aq.webm` is a synthetic 320x240, 60-frame Profile 0 clip encoded with
libvpx-vp9 to exercise the segment map and per-segment quantizers. It can be
regenerated with:

```sh
ffmpeg -f lavfi -i testsrc2=s=320x240:r=30 -frames:v 60 \
  -c:v libvpx-vp9 -b:v 0 -crf 32 -aq-mode 1 -lag-in-frames 15 vp9-aq.webm
```

`vp8-keyframe.ivf` provides one encoded VP8 frame for MP4 sample-routing tests.
The IVF header is discarded before the frame is fed to the in-house decoder.
Regenerate it with:

```sh
ffmpeg -f lavfi -i testsrc2=s=32x32:r=1 -frames:v 1 \
  -c:v libvpx -deadline best -b:v 40k -f ivf vp8-keyframe.ivf
```

`vp9-keyframe.mp4` is a one-frame MP4 fixture for VP sample entry selection
and incremental MP4 routing:

```sh
ffmpeg -f lavfi -i testsrc2=s=32x32:r=1 -frames:v 1 \
  -c:v libvpx-vp9 -b:v 40k -an -movflags +faststart vp9-keyframe.mp4
```

`vp9-altref.webm` exercises compound references and sub-8x8 motion-vector
fallbacks over 130 coded frames (120 displayed). `vp9-altref-check.yuv` contains shown frames
5 and 110 decoded by FFmpeg for pixel comparison. Regenerate both with:

```sh
ffmpeg -f lavfi -i testsrc2=s=320x240:r=30 -frames:v 120 \
  -c:v libvpx-vp9 -b:v 300k -pass 1 -passlogfile vp9-altref \
  -auto-alt-ref 1 -lag-in-frames 25 -f webm -y /dev/null
ffmpeg -f lavfi -i testsrc2=s=320x240:r=30 -frames:v 120 \
  -c:v libvpx-vp9 -b:v 300k -pass 2 -passlogfile vp9-altref \
  -auto-alt-ref 1 -lag-in-frames 25 -y vp9-altref.webm
ffmpeg -i vp9-altref.webm -vf 'select=eq(n\,5)+eq(n\,110)' \
  -fps_mode passthrough -pix_fmt yuv420p -f rawvideo -y vp9-altref-check.yuv
```

`vp9-serial-static.webm` exercises backward probability adaptation with
frame-parallel decoding disabled across 90 coded frames:

```sh
ffmpeg -f lavfi -i color=c=steelblue:s=320x240:r=30 -frames:v 90 \
  -c:v libvpx-vp9 -b:v 300k -frame-parallel 0 -auto-alt-ref 0 \
  -lag-in-frames 0 -y vp9-serial-static.webm
```

`vp8-reference-updates.ivf` exercises all 32 combinations of golden/alternate
keep, last-frame copy, other-reference copy, and current-frame refresh, together
with last-frame refresh on/off. Each case resets with two frames from
`vp8-motion.webm`, creates distinct reference pictures, applies the updates,
and probes all three references with zero motion. All 224 displayed frames
are compared pixel-exactly against `vp8-reference-updates.yuv`.

The test-only builder accumulates arithmetic interval endpoints independently
and validates round trips for mixed probabilities. No external encoder or
decoder source is used. To reproduce the encoded fixture and reference:

```sh
WEBMEDIA_VP8_REFERENCE_TEST_IVF=/tmp/vp8-reference-updates.ivf \
  cargo test --manifest-path crates/webmedia/Cargo.toml --release --lib \
  reference_update_combinations_match_reference
ffmpeg -i /tmp/vp8-reference-updates.ivf -pix_fmt yuv420p \
  -f rawvideo -y /tmp/vp8-reference-updates.yuv
```

`vp8-segment-modes.ivf` uses the independent test-only arithmetic writer to
exercise delta and absolute quantizer/filter features in all four segments,
nonzero Y2 and chroma DC residuals, an unchanged segment map, unchanged feature
tables, disabled/re-enabled segmentation, and clearing feature values. All
seven frames are compared pixel-exactly against `vp8-segment-modes.yuv`.
The explicit `libvpx` decoder selection below is intentional: FFmpeg's default
VP8 decoder produces a different picture in the re-enabled-map case. This uses
only the permitted FFmpeg executable, not encoder/decoder source or linkage.

```sh
WEBMEDIA_VP8_SEGMENT_TEST_IVF=/tmp/vp8-segment-modes.ivf \
  cargo test --manifest-path crates/webmedia/Cargo.toml --release --lib \
  segment_feature_modes_match_reference
ffmpeg -c:v libvpx -i /tmp/vp8-segment-modes.ivf -pix_fmt yuv420p \
  -f rawvideo -y /tmp/vp8-segment-modes.yuv
```

`vp8-segmentation.webm` exercises active segmentation with region-of-interest
quantization and two token partitions. The test requires nonzero segment
features, multiple segment IDs, and two partitions in every frame, and compares
all eight displayed frames pixel-exactly:

```sh
ffmpeg -f lavfi -i testsrc2=size=160x96:rate=15 -frames:v 8 \
  -vf addroi=x=0:y=0:w=80:h=48:qoffset=-0.25 \
  -c:v libvpx -b:v 150k -g 8 -slices 2 -threads 1 -an -y vp8-segmentation.webm
ffmpeg -i vp8-segmentation.webm -pix_fmt yuv420p -f rawvideo -y vp8-segmentation.yuv
```

`vp9-lossless.webm` and `vp9-lossless.yuv` exercise the lossless transform,
fixed interpolation filter mapping, signed chroma motion averaging, and sub-8x8
fractional motion preservation. All 30 frames are compared pixel-exactly:

```sh
ffmpeg -f lavfi -i testsrc2=size=160x96:rate=15 -t 2 \
  -c:v libvpx-vp9 -lossless 1 -g 30 -auto-alt-ref 0 -frame-parallel 0 \
  -threads 1 -pix_fmt yuv420p -y vp9-lossless.webm
ffmpeg -i vp9-lossless.webm -pix_fmt yuv420p -f rawvideo -y vp9-lossless.yuv
```

`vp9-serial-motion.webm` exercises serial adaptation with moving content,
including inferred high-precision motion symbols. The reference contains
frames 3 and 89 from FFmpeg:

```sh
ffmpeg -f lavfi -i testsrc2=s=320x240:r=30 -frames:v 90 \
  -c:v libvpx-vp9 -b:v 300k -frame-parallel 0 -auto-alt-ref 0 \
  -lag-in-frames 0 -y vp9-serial-motion.webm
ffmpeg -i vp9-serial-motion.webm -vf 'select=eq(n\,3)+eq(n\,89)' \
  -fps_mode passthrough -pix_fmt yuv420p -f rawvideo -y vp9-serial-motion-check.yuv
```

`vp9-tile-rows.webm` exercises two tile rows over three superblock rows
(the encoder clamps the requested row count), with above-row context continuity. The reference contains
frames 0, 1, and 29, compared pixel-exactly:

```sh
ffmpeg -f lavfi -i testsrc2=size=320x192:rate=15 -frames:v 30 \
  -c:v libvpx-vp9 -lossless 1 -tile-rows 2 -tile-columns 0 \
  -auto-alt-ref 0 -frame-parallel 0 -threads 1 -y vp9-tile-rows.webm
ffmpeg -i vp9-tile-rows.webm -vf 'select=eq(n\,0)+eq(n\,1)+eq(n\,29)' \
  -fps_mode passthrough -pix_fmt yuv420p -f rawvideo -y vp9-tile-rows-check.yuv
```

`vp9-edges.webm` checks coefficient contexts in a padded edge grid for a 162x98
picture. Frames 0, 5, and 29 are compared pixel-exactly:

```sh
ffmpeg -f lavfi -i testsrc=size=162x98:rate=15 -frames:v 30 \
  -pix_fmt yuv420p -c:v libvpx-vp9 -lossless 1 -auto-alt-ref 0 -threads 1 -y vp9-edges.webm
ffmpeg -i vp9-edges.webm -vf 'select=eq(n\,0)+eq(n\,5)+eq(n\,29)' \
  -fps_mode passthrough -pix_fmt yuv420p -f rawvideo -y vp9-edges-check.yuv
```
# JPEG XL

`jxl-alpha.png` is the webcore interlaced RGBA regression fixture.
`jxl-alpha.jxl` is its lossless JPEG XL encoding, generated with
`cjxl jxl-alpha.png jxl-alpha.jxl --distance=0 --effort=3` (libjxl 0.12.0).
The PNG is the independent pixel oracle for incremental JPEG XL decoding.
`jxl-progressive.jxl` encodes the same image with `--distance=1 --effort=3
--progressive --container=1 --num_threads=1`, exercising progressive VarDCT
pixels and JPEG XL container detection.
