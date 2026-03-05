use ndarray::ArrayView3;

use crate::types::RleData;
use crate::wavelet::{compute_max_levels, dwt3d_forward, WaveletType};

/// Result of encoding a single 3D channel.
pub struct EncodeResult {
    pub rle: RleData,
    pub intercept: f64,
    pub slope: f64,
    pub step: f64,
    pub levels: usize,
    pub wavelet: WaveletType,
}

/// Compute quantization step size from quality (1-100). Lower quality = larger step.
fn compute_step(quality: u8) -> f64 {
    // step(1) = 200, step(100) ≈ 0.5
    200.0 * (0.0025_f64).powf((quality as f64 - 1.0) / 99.0)
}

/// Encode a 3D array using wavelet transform.
/// quality=0 means lossless (5/3 wavelet, step=1).
pub fn encode_array(array: &ArrayView3<f64>, quality: u8) -> EncodeResult {
    let lossless = quality == 0;
    let wavelet = if lossless {
        WaveletType::LeGall53
    } else {
        WaveletType::CDF97
    };
    let shape = [array.shape()[0], array.shape()[1], array.shape()[2]];
    let levels = compute_max_levels(shape);

    let mut data = array.to_owned();

    // Normalize for lossy mode
    let (intercept, slope) = if !lossless {
        let (min_val, max_val) = data.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(mn, mx), &v| (mn.min(v), mx.max(v)),
        );
        let slope = max_val - min_val;
        if slope > 0.0 {
            let inv_slope = 255.0 / slope;
            data.mapv_inplace(|v| (v - min_val) * inv_slope - 128.0);
        }
        (min_val, slope)
    } else {
        (0.0, 1.0)
    };

    // Forward 3D DWT
    dwt3d_forward(&mut data, wavelet, levels);

    // Quantize
    let step = if lossless { 1.0 } else { compute_step(quality) };
    let quantized: Vec<i32> = if lossless {
        data.iter().map(|&v| v.round() as i32).collect()
    } else {
        data.iter()
            .map(|&v| {
                let s = if v >= 0.0 { 1.0 } else { -1.0 };
                (s * (v.abs() / step).floor()) as i32
            })
            .collect()
    };

    let rle = run_length_encode(&quantized);

    EncodeResult {
        rle,
        intercept,
        slope,
        step,
        levels,
        wavelet,
    }
}

/// Run-length encode a sequence of integers.
pub fn run_length_encode(sequence: &[i32]) -> RleData {
    if sequence.is_empty() {
        return RleData {
            values: vec![],
            counts: vec![],
        };
    }

    let mut values = Vec::new();
    let mut counts = Vec::new();
    let mut current_value = sequence[0];
    let mut current_count: u32 = 1;

    for &val in &sequence[1..] {
        if val == current_value {
            current_count += 1;
        } else {
            values.push(current_value);
            counts.push(current_count);
            current_value = val;
            current_count = 1;
        }
    }
    values.push(current_value);
    counts.push(current_count);

    RleData { values, counts }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rle_roundtrip() {
        let seq = vec![0, 0, 0, 1, 1, 2, 0, 0];
        let rle = run_length_encode(&seq);
        assert_eq!(rle.values, vec![0, 1, 2, 0]);
        assert_eq!(rle.counts, vec![3, 2, 1, 2]);
    }

    #[test]
    fn test_lossless_encode_decode() {
        use ndarray::Array3;
        let array = Array3::from_shape_fn((16, 16, 16), |(i, j, k)| {
            (i * 100 + j * 10 + k) as f64
        });
        let result = encode_array(&array.view(), 0); // lossless
        let decoded = crate::decoding::decode_array(
            &result.rle,
            [16, 16, 16],
            result.wavelet,
            result.levels,
            result.step,
            result.intercept,
            result.slope,
            0,
            crate::types::JvolDtype::I16,
        );
        for (a, b) in array.iter().zip(decoded.iter()) {
            assert!(
                (*a - *b).abs() < 1e-10,
                "Lossless roundtrip failed: {} != {}",
                a,
                b
            );
        }
    }
}
