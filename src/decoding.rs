use ndarray::Array3;

use crate::entropy::{lorenzo_reconstruct_3d, rice_decode_subband};
use crate::subbands::{compute_subbands, inject_subband_i32};
use crate::types::{EncodedSubband, JvolDtype};
use crate::wavelet::{dwt3d_inverse, WaveletType};

/// Decode encoded subbands back into a 3D array.
#[allow(clippy::too_many_arguments)]
pub fn decode_array(
    subbands: &[EncodedSubband],
    shape: [usize; 3],
    wavelet: WaveletType,
    levels: usize,
    step: f64,
    intercept: f64,
    slope: f64,
    quality: u8,
    dtype: JvolDtype,
) -> Array3<f64> {
    if quality == 0 {
        decode_lossless(subbands, shape, dtype)
    } else {
        decode_lossy(
            subbands, shape, wavelet, levels, step, intercept, slope, dtype,
        )
    }
}

/// Lossless decode, dtype-aware.
fn decode_lossless(
    subbands: &[EncodedSubband],
    shape: [usize; 3],
    dtype: JvolDtype,
) -> Array3<f64> {
    assert_eq!(
        subbands.len(),
        1,
        "Lossless mode expects single encoded block"
    );
    let sub = &subbands[0];
    let num_voxels = sub.num_values as usize;
    let [ni, nj, nk] = shape;

    let mut array = Array3::zeros((ni, nj, nk));

    match sub.rice_k {
        255 => {
            // Integer Lorenzo: reverse byte-shuffle → Lorenzo reconstruct (Fortran order)
            let residuals = byte_unshuffle_i32(&sub.data, num_voxels);
            let fortran_shape = [nk, nj, ni];
            let data_i32 = lorenzo_reconstruct_3d(&residuals, fortran_shape);
            // Unflatten from Fortran order
            let mut idx = 0;
            for k in 0..nk {
                for j in 0..nj {
                    for i in 0..ni {
                        array[[i, j, k]] = data_i32[idx] as f64;
                        idx += 1;
                    }
                }
            }
        }
        254 => {
            // Raw float bytes in Fortran order
            match dtype {
                JvolDtype::F32 => {
                    let mut idx = 0;
                    for k in 0..nk {
                        for j in 0..nj {
                            for i in 0..ni {
                                let offset = idx * 4;
                                let v = f32::from_le_bytes([
                                    sub.data[offset],
                                    sub.data[offset + 1],
                                    sub.data[offset + 2],
                                    sub.data[offset + 3],
                                ]);
                                array[[i, j, k]] = v as f64;
                                idx += 1;
                            }
                        }
                    }
                }
                JvolDtype::F64 => {
                    let mut idx = 0;
                    for k in 0..nk {
                        for j in 0..nj {
                            for i in 0..ni {
                                let offset = idx * 8;
                                let v = f64::from_le_bytes([
                                    sub.data[offset],
                                    sub.data[offset + 1],
                                    sub.data[offset + 2],
                                    sub.data[offset + 3],
                                    sub.data[offset + 4],
                                    sub.data[offset + 5],
                                    sub.data[offset + 6],
                                    sub.data[offset + 7],
                                ]);
                                array[[i, j, k]] = v;
                                idx += 1;
                            }
                        }
                    }
                }
                _ => unreachable!("rice_k=254 only used for float dtypes"),
            }
        }
        _ => panic!("Unknown lossless marker: rice_k={}", sub.rice_k),
    }

    // Clip to dtype range for integer types
    if let Some((min_val, max_val)) = dtype.iinfo() {
        array.mapv_inplace(|v| v.max(min_val).min(max_val));
    }

    array
}

/// Reverse byte-shuffle: 4 planes of n bytes → n i32 values.
fn byte_unshuffle_i32(data: &[u8], n: usize) -> Vec<i32> {
    let mut values = vec![0i32; n];
    for i in 0..n {
        let bytes = [data[i], data[n + i], data[2 * n + i], data[3 * n + i]];
        values[i] = i32::from_le_bytes(bytes);
    }
    values
}

/// Lossy decode: Rice decode → inverse DWT → denormalize.
#[allow(clippy::too_many_arguments)]
fn decode_lossy(
    subbands: &[EncodedSubband],
    shape: [usize; 3],
    wavelet: WaveletType,
    levels: usize,
    step: f64,
    intercept: f64,
    slope: f64,
    dtype: JvolDtype,
) -> Array3<f64> {
    let subband_infos = compute_subbands(shape, levels);
    assert_eq!(
        subbands.len(),
        subband_infos.len(),
        "Subband count mismatch"
    );

    let mut data = Array3::zeros((shape[0], shape[1], shape[2]));

    for (encoded, info) in subbands.iter().zip(subband_infos.iter()) {
        let coefficients =
            rice_decode_subband(&encoded.data, encoded.num_values as usize, encoded.rice_k);
        inject_subband_i32(&mut data, info, &coefficients);
    }

    // Dequantize
    data.mapv_inplace(|v| v * step);

    // Inverse 3D DWT
    dwt3d_inverse(&mut data, wavelet, levels);

    // Denormalize
    if slope > 0.0 {
        let scale = slope / 255.0;
        let offset = 128.0 * scale + intercept;
        data.mapv_inplace(|v| v * scale + offset);
    }

    // Clip to dtype range
    if let Some((min_val, max_val)) = dtype.iinfo() {
        data.mapv_inplace(|v| v.max(min_val).min(max_val));
    }

    data
}
