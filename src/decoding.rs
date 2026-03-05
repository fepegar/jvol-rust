use ndarray::Array3;

use crate::entropy::rice_decode_subband;
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

/// Lossless decode, dtype-aware: reverse byte-unshuffle + delta/XOR-delta decode.
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
    let [ni, nj, nk] = shape;

    let mut array = Array3::zeros((ni, nj, nk));

    match dtype {
        JvolDtype::U8 => {
            let mut vals = sub.data.clone();
            delta_decode_u8(&mut vals);
            let mut idx = 0;
            for k in 0..nk {
                for j in 0..nj {
                    for i in 0..ni {
                        array[[i, j, k]] = vals[idx] as f64;
                        idx += 1;
                    }
                }
            }
        }
        JvolDtype::U16 => {
            let unshuffled = byte_unshuffle(&sub.data, 2);
            let mut vals = from_le_bytes_u16(&unshuffled);
            delta_decode_u16(&mut vals);
            let mut idx = 0;
            for k in 0..nk {
                for j in 0..nj {
                    for i in 0..ni {
                        array[[i, j, k]] = vals[idx] as f64;
                        idx += 1;
                    }
                }
            }
        }
        JvolDtype::I16 => {
            let unshuffled = byte_unshuffle(&sub.data, 2);
            let mut vals = from_le_bytes_i16(&unshuffled);
            delta_decode_i16(&mut vals);
            let mut idx = 0;
            for k in 0..nk {
                for j in 0..nj {
                    for i in 0..ni {
                        array[[i, j, k]] = vals[idx] as f64;
                        idx += 1;
                    }
                }
            }
        }
        JvolDtype::I32 => {
            let unshuffled = byte_unshuffle(&sub.data, 4);
            let mut vals = from_le_bytes_i32(&unshuffled);
            delta_decode_i32(&mut vals);
            let mut idx = 0;
            for k in 0..nk {
                for j in 0..nj {
                    for i in 0..ni {
                        array[[i, j, k]] = vals[idx] as f64;
                        idx += 1;
                    }
                }
            }
        }
        JvolDtype::F32 => {
            // Raw f32 bytes in Fortran order
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
            // Raw f64 bytes in Fortran order
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
    }

    array
}

// --- Byte unshuffle ---

/// Reverse byte-shuffle: N planes of 1-byte → N-byte elements.
fn byte_unshuffle(data: &[u8], elem_size: usize) -> Vec<u8> {
    let n = data.len() / elem_size;
    let mut out = vec![0u8; data.len()];
    for i in 0..n {
        for b in 0..elem_size {
            out[i * elem_size + b] = data[b * n + i];
        }
    }
    out
}

// --- Delta decode (prefix sum with wrapping arithmetic) ---

fn delta_decode_u8(data: &mut [u8]) {
    for i in 1..data.len() {
        data[i] = data[i].wrapping_add(data[i - 1]);
    }
}

fn delta_decode_u16(data: &mut [u16]) {
    for i in 1..data.len() {
        data[i] = data[i].wrapping_add(data[i - 1]);
    }
}

fn delta_decode_i16(data: &mut [i16]) {
    for i in 1..data.len() {
        data[i] = data[i].wrapping_add(data[i - 1]);
    }
}

fn delta_decode_i32(data: &mut [i32]) {
    for i in 1..data.len() {
        data[i] = data[i].wrapping_add(data[i - 1]);
    }
}

// --- Bytes-to-type conversion ---

fn from_le_bytes_u16(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn from_le_bytes_i16(data: &[u8]) -> Vec<i16> {
    data.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn from_le_bytes_i32(data: &[u8]) -> Vec<i32> {
    data.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
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
