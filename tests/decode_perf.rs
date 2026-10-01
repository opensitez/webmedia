#![cfg(all(feature = "bitmap", feature = "font"))]

use std::hint::black_box;
use std::time::{Duration, Instant};

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn measure(mut decode: impl FnMut(), iterations: usize) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        decode();
    }
    start.elapsed() / iterations as u32
}

#[test]
#[ignore]
fn extracted_bitmap_vs_direct_image_crate() {
    let source = image::RgbaImage::from_fn(512, 512, |x, y| {
        image::Rgba([(x * 17) as u8, (y * 19) as u8, (x ^ y) as u8, 128])
    });
    for format in [image::ImageFormat::Png, image::ImageFormat::WebP] {
        let mut output = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(source.clone())
            .write_to(&mut output, format)
            .unwrap();
        let bytes = output.into_inner();
        let direct = || {
            let decoded = black_box(image::load_from_memory(black_box(&bytes)).unwrap());
            let has_alpha = decoded.color().has_alpha();
            let mut rgba = decoded.into_rgba8().into_raw();
            if has_alpha {
                webmedia::bitmap::premultiply_rgba(&mut rgba);
            }
            black_box(rgba);
        };
        let extracted = || {
            black_box(webmedia::bitmap::decode_raster(black_box(&bytes)).unwrap());
        };
        direct();
        extracted();
        let mut direct_times = Vec::new();
        let mut extracted_times = Vec::new();
        for round in 0..5 {
            if round % 2 == 0 {
                direct_times.push(measure(direct, 10));
                extracted_times.push(measure(extracted, 10));
            } else {
                extracted_times.push(measure(extracted, 10));
                direct_times.push(measure(direct, 10));
            }
        }
        eprintln!(
            "{format:?}: direct={:?}, webmedia={:?}",
            median(direct_times),
            median(extracted_times)
        );
    }
}

#[test]
#[ignore]
fn font_dispatch_vs_woff2_decoder() {
    let bytes = include_bytes!("fixtures/fonts/bootstrap-icons-1.11.3.woff2");
    let direct = || {
        black_box(webmedia::font::woff2::decode(black_box(bytes)).unwrap());
    };
    let dispatched = || {
        black_box(webmedia::font::decode(black_box(bytes)).unwrap());
    };
    direct();
    dispatched();
    let mut direct_times = Vec::new();
    let mut dispatched_times = Vec::new();
    for round in 0..5 {
        if round % 2 == 0 {
            direct_times.push(measure(direct, 10));
            dispatched_times.push(measure(dispatched, 10));
        } else {
            dispatched_times.push(measure(dispatched, 10));
            direct_times.push(measure(direct, 10));
        }
    }
    eprintln!(
        "WOFF2: direct={:?}, webmedia={:?}",
        median(direct_times),
        median(dispatched_times)
    );
}
