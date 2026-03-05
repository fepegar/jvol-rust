use std::sync::Arc;

use ndarray::{Array3, ArrayView3};
use rayon::prelude::*;
use rustdct::{Dct2, DctPlanner};

use crate::quantization::get_quantization_table;
use crate::types::{BlockShape, RleData, VolumeShape};

/// Result of encoding a 3D array.
pub struct EncodeResult {
    pub dc_rle: RleData,
    pub ac_rle: RleData,
    pub quantization_table: Vec<f32>,
    pub intercept: f64,
    pub slope: f64,
}

/// Pre-planned DCT objects for a given block shape (reusable across blocks).
struct DctPlans {
    dct2_i: Arc<dyn Dct2<f64>>,
    dct2_j: Arc<dyn Dct2<f64>>,
    dct2_k: Arc<dyn Dct2<f64>>,
    scratch_len_i: usize,
    scratch_len_j: usize,
    scratch_len_k: usize,
}

impl DctPlans {
    fn new(block_shape: BlockShape) -> Self {
        let [si, sj, sk] = block_shape;
        let mut planner = DctPlanner::new();
        let dct2_i = planner.plan_dct2(si);
        let dct2_j = planner.plan_dct2(sj);
        let dct2_k = planner.plan_dct2(sk);
        Self {
            scratch_len_i: dct2_i.get_scratch_len(),
            scratch_len_j: dct2_j.get_scratch_len(),
            scratch_len_k: dct2_k.get_scratch_len(),
            dct2_i,
            dct2_j,
            dct2_k,
        }
    }
}

/// Reusable scratch buffers for DCT computation (one per rayon thread).
struct DctScratch {
    scratch_i: Vec<f64>,
    scratch_j: Vec<f64>,
    scratch_k: Vec<f64>,
    temp_i: Vec<f64>,
    temp_j: Vec<f64>,
    block_buf: Vec<f64>,
}

impl DctScratch {
    fn new(block_shape: BlockShape, plans: &DctPlans) -> Self {
        let [si, sj, sk] = block_shape;
        Self {
            scratch_i: vec![0.0; plans.scratch_len_i],
            scratch_j: vec![0.0; plans.scratch_len_j],
            scratch_k: vec![0.0; plans.scratch_len_k],
            temp_i: vec![0.0; si],
            temp_j: vec![0.0; sj],
            block_buf: vec![0.0; si * sj * sk],
        }
    }
}

/// Encode a 3D array into compressed DC and AC RLE sequences.
pub fn encode_array(array: &ArrayView3<f64>, block_size: usize, quality: u8) -> EncodeResult {
    let block_shape: BlockShape = [block_size, block_size, block_size];
    let quantization_table = get_quantization_table(block_shape, quality);
    let bs = block_size;
    let bt = bs * bs * bs;

    // Single-pass min/max
    let (min_val, max_val) = array.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(mn, mx), &v| (mn.min(v), mx.max(v)),
    );
    let intercept = min_val;
    let slope = max_val - min_val;
    let inv_slope = if slope > 0.0 { 255.0 / slope } else { 0.0 };
    let pad_val = if slope > 0.0 {
        -intercept * inv_slope - 128.0
    } else {
        0.0
    };

    let shape = [array.shape()[0], array.shape()[1], array.shape()[2]];
    let ps = padded_shape(shape, block_shape);
    let ni = ps[0] / bs;
    let nj = ps[1] / bs;
    let nk = ps[2] / bs;
    let num_blocks = ni * nj * nk;

    let plans = DctPlans::new(block_shape);

    // Pre-allocate flat output for all quantized blocks
    let mut quantized_flat: Vec<i32> = vec![0; num_blocks * bt];

    // Process blocks in parallel with thread-local scratch buffers
    quantized_flat
        .par_chunks_mut(bt)
        .enumerate()
        .for_each_init(
            || DctScratch::new(block_shape, &plans),
            |scratch, (idx, out)| {
                let bi = idx / (nj * nk);
                let rem = idx % (nj * nk);
                let bj = rem / nk;
                let bk = rem % nk;

                let i0 = bi * bs;
                let j0 = bj * bs;
                let k0 = bk * bs;

                // Extract block with on-the-fly normalization and zero-padding
                let buf = &mut scratch.block_buf;
                for li in 0..bs {
                    let gi = i0 + li;
                    for lj in 0..bs {
                        let gj = j0 + lj;
                        for lk in 0..bs {
                            let gk = k0 + lk;
                            let flat = li * bs * bs + lj * bs + lk;
                            buf[flat] =
                                if gi < shape[0] && gj < shape[1] && gk < shape[2] {
                                    (array[[gi, gj, gk]] - intercept) * inv_slope - 128.0
                                } else {
                                    pad_val
                                };
                        }
                    }
                }

                // In-place 3D DCT using reusable scratch
                dct3d_inplace(block_shape, &plans, scratch);

                // Quantize directly into pre-allocated output
                let buf = &scratch.block_buf;
                for i in 0..bt {
                    out[i] = (buf[i] / quantization_table[i] as f64).round() as i32;
                }
            },
        );

    // Build DC/AC sequences from flat buffer
    let scan_indices = get_scan_indices(block_shape);
    let (dc_seq, ac_seq) =
        flat_blocks_to_sequence(&quantized_flat, &scan_indices, block_shape, num_blocks);

    EncodeResult {
        dc_rle: run_length_encode(&dc_seq),
        ac_rle: run_length_encode(&ac_seq),
        quantization_table,
        intercept,
        slope,
    }
}

/// 3D DCT-II in-place on scratch.block_buf using reusable scratch buffers.
fn dct3d_inplace(block_shape: BlockShape, plans: &DctPlans, scratch: &mut DctScratch) {
    let [si, sj, sk] = block_shape;

    // DCT along axis 2 (k) — contiguous in memory
    for i in 0..si {
        for j in 0..sj {
            let offset = i * sj * sk + j * sk;
            plans.dct2_k.process_dct2_with_scratch(
                &mut scratch.block_buf[offset..offset + sk],
                &mut scratch.scratch_k,
            );
        }
    }

    // DCT along axis 1 (j)
    for i in 0..si {
        for k in 0..sk {
            for j in 0..sj {
                scratch.temp_j[j] = scratch.block_buf[i * sj * sk + j * sk + k];
            }
            plans
                .dct2_j
                .process_dct2_with_scratch(&mut scratch.temp_j, &mut scratch.scratch_j);
            for j in 0..sj {
                scratch.block_buf[i * sj * sk + j * sk + k] = scratch.temp_j[j];
            }
        }
    }

    // DCT along axis 0 (i)
    for j in 0..sj {
        for k in 0..sk {
            for i in 0..si {
                scratch.temp_i[i] = scratch.block_buf[i * sj * sk + j * sk + k];
            }
            plans
                .dct2_i
                .process_dct2_with_scratch(&mut scratch.temp_i, &mut scratch.scratch_i);
            for i in 0..si {
                scratch.block_buf[i * sj * sk + j * sk + k] = scratch.temp_i[i];
            }
        }
    }
}

/// Extract DC and AC sequences from a flat quantized buffer.
fn flat_blocks_to_sequence(
    quantized_flat: &[i32],
    scan_indices: &[[usize; 3]],
    block_shape: BlockShape,
    num_blocks: usize,
) -> (Vec<i32>, Vec<i32>) {
    let [_si, sj, sk] = block_shape;
    let bt = scan_indices.len();

    let dc_seq: Vec<i32> = (0..num_blocks)
        .map(|b| quantized_flat[b * bt])
        .collect();

    let mut ac_seq = Vec::with_capacity(num_blocks * (bt - 1));
    for idx in &scan_indices[1..] {
        let flat_idx = idx[0] * sj * sk + idx[1] * sk + idx[2];
        for b in 0..num_blocks {
            ac_seq.push(quantized_flat[b * bt + flat_idx]);
        }
    }

    (dc_seq, ac_seq)
}

/// Generate scan indices sorted by L2 distance from the origin.
pub fn get_scan_indices(block_shape: BlockShape) -> Vec<[usize; 3]> {
    let [si, sj, sk] = block_shape;
    let mut indices: Vec<[usize; 3]> = Vec::with_capacity(si * sj * sk);
    for i in 0..si {
        for j in 0..sj {
            for k in 0..sk {
                indices.push([i, j, k]);
            }
        }
    }
    indices.sort_by(|a, b| {
        let da = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) as f64;
        let db = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]) as f64;
        da.partial_cmp(&db).unwrap()
    });
    indices
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

/// Compute the padded shape for a given volume shape and block shape.
pub fn padded_shape(shape: VolumeShape, block_shape: BlockShape) -> VolumeShape {
    let mut result = [0usize; 3];
    for d in 0..3 {
        let remainder = shape[d] % block_shape[d];
        if remainder == 0 {
            result[d] = shape[d];
        } else {
            result[d] = shape[d] + (block_shape[d] - remainder);
        }
    }
    result
}

/// Pad the array so each dimension is divisible by the block size.
pub fn pad_array(array: &ArrayView3<f64>, block_shape: BlockShape) -> Array3<f64> {
    let shape = array.shape();
    let ps = padded_shape([shape[0], shape[1], shape[2]], block_shape);

    if ps[0] == shape[0] && ps[1] == shape[1] && ps[2] == shape[2] {
        return array.to_owned();
    }

    let mut padded = Array3::<f64>::zeros((ps[0], ps[1], ps[2]));
    padded
        .slice_mut(ndarray::s![..shape[0], ..shape[1], ..shape[2]])
        .assign(array);
    padded
}

/// Split a padded 3D array into non-overlapping blocks.
pub fn split_into_blocks(array: &Array3<f64>, block_shape: BlockShape) -> Vec<Vec<f64>> {
    let shape = array.shape();
    let ni = shape[0] / block_shape[0];
    let nj = shape[1] / block_shape[1];
    let nk = shape[2] / block_shape[2];

    let mut blocks = Vec::with_capacity(ni * nj * nk);
    for bi in 0..ni {
        for bj in 0..nj {
            for bk in 0..nk {
                let i0 = bi * block_shape[0];
                let j0 = bj * block_shape[1];
                let k0 = bk * block_shape[2];
                let block_view = array.slice(ndarray::s![
                    i0..i0 + block_shape[0],
                    j0..j0 + block_shape[1],
                    k0..k0 + block_shape[2]
                ]);
                blocks.push(block_view.iter().copied().collect());
            }
        }
    }
    blocks
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
    fn test_pad_array() {
        let arr = Array3::<f64>::zeros((10, 10, 10));
        let block_shape = [8, 8, 8];
        let padded = pad_array(&arr.view(), block_shape);
        assert_eq!(padded.shape(), &[16, 16, 16]);
    }

    #[test]
    fn test_scan_indices_dc_first() {
        let indices = get_scan_indices([4, 4, 4]);
        assert_eq!(indices[0], [0, 0, 0]);
    }

    #[test]
    fn test_split_blocks() {
        let arr = Array3::<f64>::zeros((8, 8, 8));
        let blocks = split_into_blocks(&arr, [4, 4, 4]);
        assert_eq!(blocks.len(), 8); // 2*2*2
        assert_eq!(blocks[0].len(), 64); // 4*4*4
    }
}
