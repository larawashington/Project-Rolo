//! Alpha hit-mask for Rolo's sprite. Used so hover only triggers when the
//! cursor is over visible pixels, not the transparent area of the window.
//!
//! At startup we load the idle PNGs and OR their alpha channels into a single
//! binary mask — any pixel that's opaque in *any* idle frame counts as a hit.
//! That gives a stable silhouette that covers breathing bobs without needing
//! to know which frame is currently on screen.
//!
//! The mask is sampled in normalized coordinates so it works regardless of
//! window scale or retina factor.

use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Minimum alpha to count as "opaque enough for a hover". Anti-aliased edges
/// get a low alpha but are visually part of Rolo, so the threshold is low.
const ALPHA_THRESHOLD: u8 = 16;

pub struct HitMask {
    width: u32,
    height: u32,
    bits: Vec<u8>, // row-major, 0 or 1
}

impl HitMask {
    /// Merge alpha channels from the given PNG paths into a single binary mask.
    /// Returns an error if any PNG fails to decode or sizes don't match.
    pub fn from_png_paths<P: AsRef<Path>>(paths: &[P]) -> Result<Self, String> {
        if paths.is_empty() {
            return Err("hitmask: no PNG paths provided".into());
        }

        let mut combined: Option<Vec<u8>> = None;
        let mut dims: Option<(u32, u32)> = None;

        for path in paths {
            let alpha = decode_alpha(path.as_ref())?;
            let (w, h) = alpha.size;
            match dims {
                None => {
                    dims = Some((w, h));
                    combined = Some(vec![0u8; (w * h) as usize]);
                }
                Some((cw, ch)) if (cw, ch) == (w, h) => {}
                Some((cw, ch)) => {
                    return Err(format!(
                        "hitmask: PNG size mismatch — expected {}x{}, got {}x{}",
                        cw, ch, w, h,
                    ));
                }
            }
            let bits = combined.as_mut().expect("combined set above");
            for (i, &a) in alpha.values.iter().enumerate() {
                if a >= ALPHA_THRESHOLD {
                    bits[i] = 1;
                }
            }
        }

        let (width, height) = dims.ok_or("hitmask: no dimensions")?;
        let bits = combined.ok_or("hitmask: no bits")?;
        Ok(HitMask {
            width,
            height,
            bits,
        })
    }

    /// Sample the mask with normalized coordinates in `[0.0, 1.0]`.
    /// Values outside that range return false.
    pub fn contains_normalized(&self, u: f64, v: f64) -> bool {
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return false;
        }
        let x = ((u * self.width as f64) as i64).clamp(0, self.width as i64 - 1) as u32;
        let y = ((v * self.height as f64) as i64).clamp(0, self.height as i64 - 1) as u32;
        let idx = (y * self.width + x) as usize;
        self.bits.get(idx).copied().unwrap_or(0) != 0
    }
}

struct Alpha {
    size: (u32, u32),
    values: Vec<u8>,
}

fn decode_alpha(path: &Path) -> Result<Alpha, String> {
    let mut file = File::open(path).map_err(|e| format!("open {:?}: {}", path, e))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .map_err(|e| format!("read {:?}: {}", path, e))?;

    let decoder = png::Decoder::new(buf.as_slice());
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("png info {:?}: {}", path, e))?;
    let info = reader.info().clone();
    let mut raw = vec![0; reader.output_buffer_size()];
    let frame = reader
        .next_frame(&mut raw)
        .map_err(|e| format!("png decode {:?}: {}", path, e))?;
    let data = &raw[..frame.buffer_size()];

    let (w, h) = (info.width, info.height);
    let pixel_count = (w * h) as usize;
    let mut alpha = vec![255u8; pixel_count];

    use png::ColorType;
    match info.color_type {
        ColorType::Rgba => {
            for i in 0..pixel_count {
                alpha[i] = data[i * 4 + 3];
            }
        }
        ColorType::GrayscaleAlpha => {
            for i in 0..pixel_count {
                alpha[i] = data[i * 2 + 1];
            }
        }
        ColorType::Rgb | ColorType::Grayscale => {
            // No alpha channel — treat all pixels as opaque. This is not
            // what we want for a sprite but keeps the loader permissive.
        }
        ColorType::Indexed => {
            // Decoder should expand indexed to RGBA when a tRNS chunk exists.
            // Fall through assuming expansion; if alpha absent, stays opaque.
        }
    }

    Ok(Alpha {
        size: (w, h),
        values: alpha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_paths_returns_error() {
        let paths: Vec<&Path> = Vec::new();
        assert!(HitMask::from_png_paths(&paths).is_err());
    }

    #[test]
    fn normalized_contains_respects_bounds() {
        let mask = HitMask {
            width: 4,
            height: 4,
            bits: vec![0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        };
        assert!(mask.contains_normalized(0.5, 0.5));
        assert!(!mask.contains_normalized(0.0, 0.0));
        assert!(!mask.contains_normalized(-0.1, 0.5));
        assert!(!mask.contains_normalized(0.5, 1.5));
    }
}
