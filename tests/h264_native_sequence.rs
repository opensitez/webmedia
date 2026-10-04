//! Standalone native H.264 benchmark/oracle harness (no RGBA conversion).
//! Compile with `rustc -O --edition=2024 ... -o /private/tmp/h264-sequence`;
//! run with `INPUT.mp4 [SAMPLE_COUNT] [oracle]`. Omit SAMPLE_COUNT for a full oracle.
//! For paired timing, compile with `--test`, set H264_WORKER_FIXTURE (and optionally
//! H264_WORKER_SAMPLES), then run benchmark_actual_same_frame_blend_cache --ignored --nocapture.
#![allow(dead_code)]
#[path = "../src/video/backend.rs"]
mod backend;
pub use backend::VideoFrame;
#[cfg(test)]
mod video {
    pub use crate::h264;
}
#[path = "../src/video/h264.rs"]
pub mod h264;
#[path = "../src/video/h264_cabac.rs"]
mod h264_cabac;
#[path = "../src/video/h264_deblock.rs"]
mod h264_deblock;
#[path = "../src/video/h264_high.rs"]
mod h264_high;
#[path = "../src/video/h264_inter.rs"]
mod h264_inter;
#[path = "../src/video/h264_intra.rs"]
mod h264_intra;
#[path = "../src/video/h264_transform.rs"]
mod h264_transform;
#[path = "../src/video/mp4.rs"]
mod mp4;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    run_sequence(
        &args[1],
        args.get(2).map(|v| v.parse().unwrap()),
        args.get(3).is_some_and(|v| v == "oracle"),
    );
}

fn run_sequence(path: &str, count: Option<usize>, oracle: bool) -> std::time::Duration {
    use h264::*;
    use std::io::Read;
    let bytes = std::fs::read(path).unwrap();
    let index = mp4::Mp4Index::parse_prefix(&bytes).unwrap();
    let count = count.unwrap_or(index.samples.len());
    assert!(count > 0 && count <= index.samples.len());
    let sps = &index.config.sequence_parameters[0];
    let pps = parse_pps_2005(&index.config.picture_parameter_sets[0]).unwrap();
    let mut ffmpeg = oracle.then(|| {
        std::process::Command::new("ffmpeg")
            .args([
                "-v", "error", "-i", path, "-map", "0:v:0", "-pix_fmt", "yuv420p", "-f",
                "rawvideo", "-",
            ])
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap()
    });
    let mut output = ffmpeg.as_mut().map(|p| p.stdout.take().unwrap());
    let mut order: Vec<_> = (0..count).collect();
    order.sort_by_key(|&i| index.samples[i].presentation_time);
    let mut ranks = vec![0; count];
    for (rank, &sample) in order.iter().enumerate() {
        ranks[sample] = rank;
    }
    let mut pending = std::collections::BTreeMap::new();
    let mut checked = 0;
    let mut refs = Vec::new();
    let mut times = [std::time::Duration::ZERO; 3];
    let mut counts = [0usize; 3];
    let mut raw = vec![0; sps.width as usize * sps.height as usize * 3 / 2];
    for (i, sample) in index.samples[..count].iter().enumerate() {
        let payload = &bytes[sample.offset as usize..sample.offset as usize + sample.size as usize];
        let mut stream = NalStream::new(index.config.nal_length_size).unwrap();
        let nals = stream.push(payload).unwrap();
        stream.finish().unwrap();
        let mut pictures = nals.iter().filter(|n| matches!(n[0] & 31, 1 | 5));
        let nal = pictures.next().unwrap();
        assert!(pictures.next().is_none());
        let kind = parse_slice_type(nal).unwrap() % 5;
        let idr = nal[0] & 31 == 5;
        let marking = if kind != 2 {
            parse_cabac_inter_slice(nal, sps, &pps.core)
                .unwrap()
                .marking
        } else {
            Vec::new()
        };
        let start = std::time::Instant::now();
        let (picture, marking) = match kind {
            0 => (
                h264_inter::decode_cabac_p_2005(nal, sps, &pps, &refs),
                marking,
            ),
            1 => (
                h264_inter::decode_cabac_b_2005(nal, sps, &pps, &refs),
                marking,
            ),
            2 if idr => (
                h264_high::decode_cabac_idr_yuv_2005(nal, sps, &pps),
                Vec::new(),
            ),
            2 => {
                let (p, m) =
                    h264_high::decode_cabac_i_yuv_2005(nal, sps, &pps, refs.last()).unwrap();
                (Ok(p), m)
            }
            _ => panic!("slice type {kind}"),
        };
        let picture = picture.unwrap_or_else(|e| panic!("sample {i}: {e:?}"));
        let slot = if kind == 2 {
            0
        } else if kind == 0 {
            1
        } else {
            2
        };
        times[slot] += start.elapsed();
        counts[slot] += 1;
        if let Some(output) = output.as_mut() {
            let mut pixels = Vec::with_capacity(raw.len());
            for (plane, width, height, stride) in [
                (
                    &picture.luma,
                    sps.width as usize,
                    sps.height as usize,
                    picture.width,
                ),
                (
                    &picture.cb,
                    sps.width as usize / 2,
                    sps.height as usize / 2,
                    picture.width / 2,
                ),
                (
                    &picture.cr,
                    sps.width as usize / 2,
                    sps.height as usize / 2,
                    picture.width / 2,
                ),
            ] {
                for row in plane.chunks_exact(stride).take(height) {
                    pixels.extend_from_slice(&row[..width]);
                }
            }
            pending.insert(ranks[i], pixels);
            while let Some(pixels) = pending.remove(&checked) {
                output.read_exact(&mut raw).unwrap();
                if pixels != raw {
                    let differences = pixels.iter().zip(&raw).filter(|(a, b)| a != b).count();
                    panic!(
                        "presentation frame {checked} sample {}: {differences} differing bytes",
                        order[checked]
                    );
                }
                checked += 1;
            }
        }
        if idr {
            refs.clear();
        }
        if nal[0] & 0x60 != 0 {
            for op in marking {
                let MemoryManagement::ForgetShortTerm(distance) = op;
                let maximum = 1u32 << sps.frame_num_bits;
                let target = (picture.frame_num + maximum - (distance + 1) % maximum) % maximum;
                let at = refs.iter().position(|p| p.frame_num == target).unwrap();
                refs.remove(at);
            }
            refs.push(picture);
            if refs.len() > sps.max_num_ref_frames as usize {
                refs.remove(0);
            }
        }
        if i % 500 == 499 {
            eprintln!("decoded {} samples", i + 1);
        }
    }
    if let Some(mut child) = ffmpeg {
        if count == index.samples.len() {
            assert!(child.wait().unwrap().success());
        } else {
            drop(output);
            child.wait().unwrap();
        }
        assert_eq!(checked, count);
    }
    let total: std::time::Duration = times.iter().sum();
    eprintln!(
        "samples={count} checked={checked} native_ms/frame={:.4} counts={counts:?} totals_ms={:?}",
        total.as_secs_f64() * 1000.0 / count as f64,
        times.map(|t| t.as_secs_f64() * 1000.0)
    );
    total
}

#[test]
#[ignore = "explicit same-frame paired native decode benchmark with independent DPBs"]
fn benchmark_actual_same_frame_blend_cache() {
    use h264::*;
    let path = std::env::var("H264_WORKER_FIXTURE").expect("H264_WORKER_FIXTURE required");
    let count = std::env::var("H264_WORKER_SAMPLES")
        .ok()
        .map(|v| v.parse().unwrap())
        .unwrap_or(338);
    let bytes = std::fs::read(path).unwrap();
    let index = mp4::Mp4Index::parse_prefix(&bytes).unwrap();
    let sps = &index.config.sequence_parameters[0];
    let pps = parse_pps_2005(&index.config.picture_parameter_sets[0]).unwrap();
    assert!(count > 0 && count <= index.samples.len());
    let inputs: Vec<_> = index.samples[..count]
        .iter()
        .map(|sample| {
            let mut stream = NalStream::new(index.config.nal_length_size).unwrap();
            let nals = stream
                .push(&bytes[sample.offset as usize..sample.offset as usize + sample.size as usize])
                .unwrap();
            stream.finish().unwrap();
            let mut pictures = nals.into_iter().filter(|nal| matches!(nal[0] & 31, 1 | 5));
            let nal = pictures.next().unwrap();
            assert!(pictures.next().is_none());
            nal
        })
        .collect();
    let mut results = [Vec::new(), Vec::new()];
    for run in 0..3 {
        let mut references: [Vec<h264_high::Yuv420Picture>; 2] = [Vec::new(), Vec::new()];
        let mut times = [[std::time::Duration::ZERO; 3]; 2];
        for (sample, nal) in inputs.iter().enumerate() {
            let kind = parse_slice_type(nal).unwrap() % 5;
            let idr = nal[0] & 31 == 5;
            let marking = if kind != 2 {
                parse_cabac_inter_slice(nal, sps, &pps.core)
                    .unwrap()
                    .marking
            } else {
                Vec::new()
            };
            let mut decoded = [None, None];
            for turn in 0..2 {
                let mode = (turn + sample + run) % 2;
                let result = h264_inter::with_b_blend_cache(mode == 1, || {
                    let start = std::time::Instant::now();
                    let result = match kind {
                        0 => (
                            h264_inter::decode_cabac_p_2005(nal, sps, &pps, &references[mode])
                                .unwrap(),
                            marking.clone(),
                        ),
                        1 => (
                            h264_inter::decode_cabac_b_2005(nal, sps, &pps, &references[mode])
                                .unwrap(),
                            marking.clone(),
                        ),
                        2 if idr => (
                            h264_high::decode_cabac_idr_yuv_2005(nal, sps, &pps).unwrap(),
                            Vec::new(),
                        ),
                        2 => h264_high::decode_cabac_i_yuv_2005(
                            nal,
                            sps,
                            &pps,
                            references[mode].last(),
                        )
                        .unwrap(),
                        _ => panic!("unsupported slice type"),
                    };
                    times[mode][if kind == 2 {
                        0
                    } else if kind == 0 {
                        1
                    } else {
                        2
                    }] += start.elapsed();
                    result
                });
                decoded[mode] = Some(result);
            }
            let a = &decoded[0].as_ref().unwrap().0;
            let b = &decoded[1].as_ref().unwrap().0;
            assert!(
                a.luma == b.luma && a.cb == b.cb && a.cr == b.cr,
                "sample {sample}: YUV differs"
            );
            assert_eq!(a.motion, b.motion);
            assert_eq!(a.reference_pocs, b.reference_pocs);
            assert_eq!(
                (a.frame_num, a.pic_order_cnt),
                (b.frame_num, b.pic_order_cnt)
            );
            for mode in 0..2 {
                let (picture, marking) = decoded[mode].take().unwrap();
                if idr {
                    references[mode].clear();
                }
                if nal[0] & 0x60 != 0 {
                    for operation in marking {
                        let MemoryManagement::ForgetShortTerm(distance) = operation;
                        let maximum = 1u32 << sps.frame_num_bits;
                        let target =
                            (picture.frame_num + maximum - (distance + 1) % maximum) % maximum;
                        let at = references[mode]
                            .iter()
                            .position(|p| p.frame_num == target)
                            .unwrap();
                        references[mode].remove(at);
                    }
                    references[mode].push(picture);
                    if references[mode].len() > sps.max_num_ref_frames as usize {
                        references[mode].remove(0);
                    }
                }
            }
        }
        for mode in 0..2 {
            let total: std::time::Duration = times[mode].iter().sum();
            results[mode].push(total);
            eprintln!(
                "same-frame run={run} cache={} samples={count} native_ms/frame={:.4} totals_ms={:?}",
                mode == 1,
                total.as_secs_f64() * 1000.0 / count as f64,
                times[mode].map(|t| t.as_secs_f64() * 1000.0)
            );
        }
    }
    for times in &mut results {
        times.sort();
    }
    let before = results[0][1].as_secs_f64();
    let after = results[1][1].as_secs_f64();
    eprintln!(
        "same-frame median before_ms/frame={:.4} after_ms/frame={:.4} improvement={:.2}%",
        before * 1000.0 / count as f64,
        after * 1000.0 / count as f64,
        (1.0 - after / before) * 100.0
    );
}
