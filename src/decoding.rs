use ndarray::Array3;

use crate::types::{JvolDtype, RleData};
use crate::wavelet::{dwt3d_inverse, WaveletType};

/// Decode RLE-compressed wavelet coefficients back into a 3D array.
#[allow(clippy::too_many_arguments)]
pub fn decode_array(
    rle: &RleData,
    shape: [usize; 3],
    wavelet: WaveletType,
    levels: usize,
    step: f64,
    intercept: f64,
    slope: f64,
    quality: u8,
    dtype: JvolDtype,
) -> Array3<f64> {
    let lossless = quality == 0;

    // RLE decode
    let quantized = run_length_decode(rle);

    // Dequantize
    let dequantized: Vec<f64> = if lossless {
        quantized.iter().map(|&v| v as f64).collect()
    } else {
        quantized.iter().map(|&v| v as f64 * step).collect()
    };

    // Reshape to 3D
    let mut data =
        Array3::from_shape_vec((shape[0], shape[1], shape[2]), dequantized).expect("Shape mismatch");

    // Inverse 3D DWT
    dwt3d_inverse(&mut data, wavelet, levels);

    // Denormalize for lossy mode
    if !lossless && slope > 0.0 {
        let scale = slope / 255.0;
        let offset = 128.0 * scale + intercept;
        data.mapv_inplace(|v| v * scale + offset);
    }

    // Clip to dtype range if integer
    if let Some((min_val, max_val)) = dtype.iinfo() {
        data.mapv_inplace(|v| v.max(min_val).min(max_val));
    }

    data
}

/// Expand RLE data back into a flat sequence.
pub fn run_length_decode(rle: &RleData) -> Vec<i32> {
    let total_len: usize = rle.counts.iter().map(|&c| c as usize).sum();
    let mut result = Vec::with_capacity(total_len);
    for (&value, &count) in rle.values.iter().zip(rle.counts.iter()) {
        for _ in 0..count {
            result.push(value);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::run_length_encode;

    #[test]
    fn test_rle_roundtrip() {
        let original = vec![0, 0, 0, 5, 5, -1, -1, -1, -1, 0];
        let rle = run_length_encode(&original);
        let decoded = run_length_decode(&rle);
        assert_eq!(original, decoded);
    }
}
