use ndarray::ArrayView3;

use crate::entropy::{lorenzo_predict_3d, rice_encode_subband};
use crate::subbands::{compute_subbands, extract_subband_i32};
use crate::types::{EncodedSubband, JvolDtype};
use crate::wavelet::{compute_max_levels, dwt3d_forward, WaveletType};

/// Result of encoding a single 3D channel.
pub struct EncodeResult {
    /// For lossy: per-subband Rice-coded data.
    /// For lossless: single entry with prediction residuals or raw bytes.
    pub subbands: Vec<EncodedSubband>,
    pub intercept: f64,
    pub slope: f64,
    pub step: f64,
    pub levels: usize,
    pub wavelet: WaveletType,
}

/// Compute quantization step size from quality (1-100). Lower quality = larger step.
fn compute_step(quality: u8) -> f64 {
    200.0 * (0.0025_f64).powf((quality as f64 - 1.0) / 99.0)
}

/// Encode a 3D array.
/// quality=0: lossless. quality>0: lossy (3D DWT + quantization + per-subband Rice coding).
pub fn encode_array(array: &ArrayView3<f64>, quality: u8, dtype: JvolDtype) -> EncodeResult {
    if quality == 0 {
        encode_lossless(array, dtype)
    } else {
        encode_lossy(array, quality)
    }
}

/// Lossless encoding, dtype-aware:
/// - Integer dtypes: 3D Lorenzo prediction + byte-shuffle → great compression
/// - Float dtypes: raw native-dtype bytes → zstd handles compression
fn encode_lossless(array: &ArrayView3<f64>, dtype: JvolDtype) -> EncodeResult {
    let shape = [array.shape()[0], array.shape()[1], array.shape()[2]];
    let num_voxels = shape[0] * shape[1] * shape[2];

    let encoded_data = match dtype {
        JvolDtype::U8 | JvolDtype::U16 | JvolDtype::I16 | JvolDtype::I32 => {
            // Integer: Lorenzo prediction (Fortran order for better compression) + byte-shuffle
            let data_i32 = flatten_fortran_i32(array, shape);
            // Lorenzo needs the shape in Fortran iteration order [nk, nj, ni]
            let fortran_shape = [shape[2], shape[1], shape[0]];
            let residuals = lorenzo_predict_3d(&data_i32, fortran_shape);
            byte_shuffle_i32(&residuals)
        }
        JvolDtype::F32 => {
            // f32: Fortran-order raw bytes (matches NIfTI layout for better compression)
            let mut bytes = Vec::with_capacity(num_voxels * 4);
            let [ni, nj, nk] = shape;
            for k in 0..nk {
                for j in 0..nj {
                    for i in 0..ni {
                        bytes.extend_from_slice(&(array[[i, j, k]] as f32).to_le_bytes());
                    }
                }
            }
            bytes
        }
        JvolDtype::F64 => {
            // f64: Fortran-order raw bytes
            let mut bytes = Vec::with_capacity(num_voxels * 8);
            let [ni, nj, nk] = shape;
            for k in 0..nk {
                for j in 0..nj {
                    for i in 0..ni {
                        bytes.extend_from_slice(&array[[i, j, k]].to_le_bytes());
                    }
                }
            }
            bytes
        }
    };

    // rice_k=255 for integer Lorenzo, rice_k=254 for raw float bytes
    let marker = match dtype {
        JvolDtype::U8 | JvolDtype::U16 | JvolDtype::I16 | JvolDtype::I32 => 255,
        JvolDtype::F32 | JvolDtype::F64 => 254,
    };

    EncodeResult {
        subbands: vec![EncodedSubband {
            rice_k: marker,
            num_values: num_voxels as u32,
            data: encoded_data,
        }],
        intercept: 0.0,
        slope: 1.0,
        step: 1.0,
        levels: 0,
        wavelet: WaveletType::LeGall53,
    }
}

/// Flatten array to i32 in Fortran order (column-major, matching NIfTI layout).
fn flatten_fortran_i32(array: &ArrayView3<f64>, shape: [usize; 3]) -> Vec<i32> {
    let [ni, nj, nk] = shape;
    let mut data = Vec::with_capacity(ni * nj * nk);
    for k in 0..nk {
        for j in 0..nj {
            for i in 0..ni {
                data.push(array[[i, j, k]].round() as i32);
            }
        }
    }
    data
}

/// Byte-shuffle i32 values: separate into 4 byte planes.
/// For small values, high byte planes are mostly zeros → compresses well with zstd.
fn byte_shuffle_i32(values: &[i32]) -> Vec<u8> {
    let n = values.len();
    let mut out = vec![0u8; n * 4];
    for (i, &v) in values.iter().enumerate() {
        let bytes = v.to_le_bytes();
        out[i] = bytes[0]; // plane 0 (LSB)
        out[n + i] = bytes[1]; // plane 1
        out[2 * n + i] = bytes[2]; // plane 2
        out[3 * n + i] = bytes[3]; // plane 3 (MSB)
    }
    out
}

/// Lossy encoding: 3D DWT + quantization + per-subband Rice coding.
fn encode_lossy(array: &ArrayView3<f64>, quality: u8) -> EncodeResult {
    let wavelet = WaveletType::CDF97;
    let shape = [array.shape()[0], array.shape()[1], array.shape()[2]];
    let levels = compute_max_levels(shape);

    let mut data = array.to_owned();

    // Normalize to [-128, 127]
    let (min_val, max_val) = data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });
    let slope = max_val - min_val;
    let intercept = min_val;
    if slope > 0.0 {
        let inv_slope = 255.0 / slope;
        data.mapv_inplace(|v| (v - min_val) * inv_slope - 128.0);
    }

    // Forward 3D DWT
    dwt3d_forward(&mut data, wavelet, levels);

    // Dead-zone quantization
    let step = compute_step(quality);
    let inv_step = 1.0 / step;
    data.mapv_inplace(|v| {
        let s = if v >= 0.0 { 1.0 } else { -1.0 };
        s * (v.abs() * inv_step).floor()
    });

    // Per-subband Rice coding
    let subband_infos = compute_subbands(shape, levels);
    let mut encoded_subbands = Vec::with_capacity(subband_infos.len());

    for info in &subband_infos {
        let coefficients = extract_subband_i32(&data, info);
        let num_values = coefficients.len() as u32;
        let (encoded_data, rice_k) = rice_encode_subband(&coefficients);
        encoded_subbands.push(EncodedSubband {
            rice_k,
            num_values,
            data: encoded_data,
        });
    }

    EncodeResult {
        subbands: encoded_subbands,
        intercept,
        slope,
        step,
        levels,
        wavelet,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_lossless_integer_encode_decode() {
        let array = Array3::from_shape_fn((16, 16, 16), |(i, j, k)| (i * 100 + j * 10 + k) as f64);
        let result = encode_array(&array.view(), 0, JvolDtype::I16);

        let decoded = crate::decoding::decode_array(
            &result.subbands,
            [16, 16, 16],
            result.wavelet,
            result.levels,
            result.step,
            result.intercept,
            result.slope,
            0,
            JvolDtype::I16,
        );
        for (a, b) in array.iter().zip(decoded.iter()) {
            assert!(
                (*a - *b).abs() < 1e-10,
                "Lossless roundtrip failed: {} != {}",
                a,
                b,
            );
        }
    }

    #[test]
    fn test_lossless_float_encode_decode() {
        let array = Array3::from_shape_fn((8, 8, 8), |(i, j, k)| {
            i as f64 * 100.5 + j as f64 * 10.3 + k as f64 * 1.7
        });
        let result = encode_array(&array.view(), 0, JvolDtype::F32);

        let decoded = crate::decoding::decode_array(
            &result.subbands,
            [8, 8, 8],
            result.wavelet,
            result.levels,
            result.step,
            result.intercept,
            result.slope,
            0,
            JvolDtype::F32,
        );
        // f64 → f32 → f64 roundtrip may lose some precision
        for (a, b) in array.iter().zip(decoded.iter()) {
            let a_f32 = *a as f32 as f64;
            assert!(
                (a_f32 - *b).abs() < 1e-6,
                "Float lossless roundtrip failed: {} != {}",
                a_f32,
                b,
            );
        }
    }

    #[test]
    fn test_lossy_encode_decode() {
        let array = Array3::from_shape_fn((16, 16, 16), |(i, j, k)| (i * 100 + j * 10 + k) as f64);
        let result = encode_array(&array.view(), 60, JvolDtype::I16);

        let decoded = crate::decoding::decode_array(
            &result.subbands,
            [16, 16, 16],
            result.wavelet,
            result.levels,
            result.step,
            result.intercept,
            result.slope,
            60,
            JvolDtype::I16,
        );
        let max_err: f64 = array
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 200.0, "Max error too large: {}", max_err);
    }
}
