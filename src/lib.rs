//! Image and video decoders without a DOM, network stack, or renderer.

#[cfg(feature = "acceleration")]
pub use accelerate::{InvalidRgbaSurface, RgbaSurface};

#[cfg(feature = "bitmap")]
pub mod bitmap;

#[cfg(any(feature = "bitmap", feature = "video"))]
#[path = "video/av1.rs"]
pub mod av1;

#[cfg(feature = "video")]
pub mod video;

#[cfg(feature = "video")]
pub mod audio;

#[cfg(feature = "font")]
pub mod font;
