# AV1 128x128 Superblock Fixture

This synthetic test pattern was generated locally, not copied from a website.
The OBU file exercises the shared AV1 decoder; the raw YUV file is an independent
FFmpeg decoding oracle. These tools are test-fixture generators, not runtime
dependencies.

Regenerate with libaom aomenc and FFmpeg:

```sh
ffmpeg -f lavfi -i 'testsrc2=size=256x192:rate=1,noise=alls=20:allf=t:all_seed=1' -frames:v 1 -pix_fmt yuv420p pattern.y4m
aomenc --obu --limit=1 --passes=1 --cpu-used=6 --sb-size=128 --force-video-mode=1 --enable-palette=0 --enable-intrabc=0 --enable-restoration=0 --deltaq-mode=0 --end-usage=q --cq-level=35 -o avif-128.obu pattern.y4m
ffmpeg -i avif-128.obu -frames:v 1 -f rawvideo -pix_fmt yuv420p avif-128.yuv
```

Original encoder: aomenc 3.14.1. Deterministic texture avoids screen-content
classification; the test asserts the parsed tools before reconstruction. This
fixture isolates superblock sizing, not palette or full AVIF support.
