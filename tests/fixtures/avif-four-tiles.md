# AVIF/AV1 Streaming and Tile Fixtures

These synthetic images were generated locally. They contain no website assets.
FFmpeg and libaom are fixture-generation tools, not decoder dependencies.

The four-tile fixture has two tile columns and two tile rows; the native YUV
oracle checks every sample, including prediction across internal block edges
and isolation at tile boundaries.

```sh
ffmpeg -f lavfi -i 'testsrc2=size=512x256:rate=1,noise=alls=20:allf=t:all_seed=1' -frames:v 1 -pix_fmt yuv420p pattern.y4m
aomenc --obu --limit=1 --passes=1 --cpu-used=6 --sb-size=64 --tile-columns=1 --tile-rows=1 --force-video-mode=1 --enable-palette=0 --enable-intrabc=0 --enable-restoration=0 --deltaq-mode=0 --end-usage=q --cq-level=35 -o avif-four-tiles.obu pattern.y4m
ffmpeg -i avif-four-tiles.obu -frames:v 1 -pix_fmt yuv420p -f rawvideo avif-four-tiles.yuv
ffmpeg -f lavfi -i 'color=c=red:s=64x64' -frames:v 1 -c:v libsvtav1 -preset 12 -f avif avif-stream-red.avif
```

Original generators: libaom 3.14.1 and SVT-AV1 4.2.0. The red AVIF fixture
tests metadata delivery and pixel publication before trailing container data.
