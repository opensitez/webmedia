//! Decode PNG, JPEG, JPEG XL, GIF, WebP, and BMP into premultiplied RGBA8.

pub mod jpeg_xl;
pub mod avif;

/// A decoded raster image. Pixels are premultiplied RGBA8, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RasterImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[cfg(feature = "acceleration")]
impl RasterImage {
    pub fn into_surface(self) -> Result<crate::RgbaSurface, crate::InvalidRgbaSurface> {
        crate::RgbaSurface::new(std::sync::Arc::new(self.rgba), self.width, self.height)
    }
}

/// Decode a complete raster image. Animated formats yield their first frame.
/// SVG is a document format and is not decoded here.
#[inline]
pub fn decode_raster(bytes: &[u8]) -> Result<RasterImage, image::ImageError> {
    if avif::is_avif(bytes) {
        return avif::decode(bytes).map_err(|error| {
            image::ImageError::IoError(std::io::Error::new(std::io::ErrorKind::InvalidData, error))
        });
    }
    if jpeg_xl::is_jpeg_xl(bytes) {
        return jpeg_xl::decode_stream(bytes, |_, _| {}, |_| {}).map_err(|error| {
            image::ImageError::IoError(std::io::Error::new(std::io::ErrorKind::InvalidData, error))
        });
    }
    let image = image::load_from_memory(bytes)?;
    let has_alpha = image.color().has_alpha();
    let rgba = image.into_rgba8();
    let (width, height) = rgba.dimensions();
    let mut rgba = rgba.into_raw();
    if has_alpha {
        premultiply_rgba(&mut rgba);
    }
    Ok(RasterImage {
        width,
        height,
        rgba,
    })
}

/// Convert straight RGBA8 pixels to premultiplied RGBA8 in place.
#[inline]
pub fn premultiply_rgba(rgba: &mut [u8]) {
    #[cfg(feature = "acceleration")]
    {
        accelerate::software::premultiply_rgba(rgba);
    }
    #[cfg(not(feature = "acceleration"))]
    for pixel in rgba.chunks_exact_mut(4) {
        let alpha = pixel[3] as u16;
        if alpha < 255 {
            for channel in &mut pixel[..3] {
                *channel = ((*channel as u16 * alpha) / 255) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "acceleration")]
    #[test]
    fn raster_surface_preserves_the_decoder_allocation() {
        let image = RasterImage { width: 1, height: 1, rgba: vec![40, 80, 120, 255] };
        let address = image.rgba.as_ptr();
        let surface = image.into_surface().unwrap();
        assert_eq!(surface.pixels().as_ptr(), address);
        assert_eq!(surface.dimensions(), (1, 1));
        assert_eq!(surface.layer_frame([0., 0., 10., 10.]).rgba.as_ptr(), address);
    }

    #[test]
    fn transparent_png_is_premultiplied() {
        let pixels = image::RgbaImage::from_raw(1, 1, vec![200, 100, 50, 128]).unwrap();
        let mut output = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(pixels)
            .write_to(&mut output, image::ImageFormat::Png)
            .unwrap();
        let decoded = decode_raster(output.get_ref()).unwrap();
        assert_eq!((decoded.width, decoded.height), (1, 1));
        assert_eq!(decoded.rgba, [100, 50, 25, 128]);
    }
}
