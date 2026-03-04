use std::sync::Arc;

use ndarray::Array3;
use rayon::prelude::*;
use rustdct::{Dct3, DctPlanner};

use crate::encoding::{get_scan_indices, padded_shape};
use crate::types::{BlockShape, JvolDtype, RleData, VolumeShape};

/// Pre-planned inverse DCT objects for a given block shape.
struct IdctPlans {
    dct3_i: Arc<dyn Dct3<f64>>,
    dct3_j: Arc<dyn Dct3<f64>>,
    dct3_k: Arc<dyn Dct3<f64>>,
    scratch_len_i: usize,
    scratch_len_j: usize,
    scratch_len_k: usize,
}

impl IdctPlans {
    fn new(block_shape: BlockShape) -> Self {
        let [si, sj, sk] = block_shape;
        let mut planner = DctPlanner::new();
        let dct3_i = planner.plan_dct3(si);
        let dct3_j = planner.plan_dct3(sj);
        let dct3_k = planner.plan_dct3(sk);
        Self {
            scratch_len_i: dct3_i.get_scratch_len(),
            scratch_len_j: dct3_j.get_scratch_len(),
            scratch_len_k: dct3_k.get_scratch_len(),
            dct3_i,
            dct3_j,
            dct3_k,
        }
    }
}

/// Decode RLE-compressed DC and AC sequences back into a 3D array.
pub fn decode_array(
    dc_rle: &RleData,
    ac_rle: &RleData,
    quantization_table: &[f32],
    target_shape: VolumeShape,
    block_shape: BlockShape,
    intercept: f64,
    slope: f64,
    dtype: JvolDtype,
) -> Array3<f64> {
    // RLE decode
    let dc_sequence = run_length_decode(dc_rle);
    let ac_sequence = run_length_decode(ac_rle);

    // Reconstruct blocks
    let scan_indices = get_scan_indices(block_shape);
    let blocks = sequence_to_blocks(&dc_sequence, &ac_sequence, &scan_indices, block_shape);

    // Pre-plan iDCT
    let plans = IdctPlans::new(block_shape);

    // Parallel dequantize + inverse DCT
    let reconstructed_blocks: Vec<Vec<f64>> = blocks
        .par_iter()
        .map(|block| {
            let dequantized: Vec<f64> = block
                .iter()
                .zip(quantization_table.iter())
                .map(|(&v, &q)| v as f64 * q as f64)
                .collect();
            idct3d_with_plans(&dequantized, block_shape, &plans)
        })
        .collect();

    // Reassemble blocks into volume
    let padded = padded_shape(target_shape, block_shape);
    let mut array = assemble_blocks(&reconstructed_blocks, padded, block_shape);

    // Rescale: undo the [-128, 127] normalization
    array.mapv_inplace(|v| {
        let rescaled = (v + 128.0) / 255.0;
        rescaled * slope + intercept
    });

    // Crop to original shape
    let cropped = array
        .slice(ndarray::s![
            ..target_shape[0],
            ..target_shape[1],
            ..target_shape[2]
        ])
        .to_owned();

    // Clip to dtype range if integer
    if let Some((min_val, max_val)) = dtype.iinfo() {
        let mut result = cropped;
        result.mapv_inplace(|v| v.max(min_val).min(max_val));
        return result;
    }

    cropped
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

/// Reconstruct 3D blocks from DC and AC sequences using scan ordering.
fn sequence_to_blocks(
    dc_sequence: &[i32],
    ac_sequence: &[i32],
    scan_indices: &[[usize; 3]],
    block_shape: BlockShape,
) -> Vec<Vec<i32>> {
    let [_si, sj, sk] = block_shape;
    let block_size = scan_indices.len();
    let num_blocks = dc_sequence.len();

    let mut blocks: Vec<Vec<i32>> = vec![vec![0i32; block_size]; num_blocks];

    // DC component
    for (b, &dc) in dc_sequence.iter().enumerate() {
        blocks[b][0] = dc;
    }

    // AC components
    for (scan_pos, idx) in scan_indices[1..].iter().enumerate() {
        let flat_idx = idx[0] * sj * sk + idx[1] * sk + idx[2];
        let ac_offset = scan_pos * num_blocks;
        for b in 0..num_blocks {
            blocks[b][flat_idx] = ac_sequence[ac_offset + b];
        }
    }

    blocks
}

/// Compute 3D inverse DCT (DCT-III) using pre-planned objects.
fn idct3d_with_plans(block: &[f64], block_shape: BlockShape, plans: &IdctPlans) -> Vec<f64> {
    let [si, sj, sk] = block_shape;
    let mut data = block.to_vec();

    let mut scratch_i = vec![0.0f64; plans.scratch_len_i];
    let mut scratch_j = vec![0.0f64; plans.scratch_len_j];
    let mut scratch_k = vec![0.0f64; plans.scratch_len_k];
    let mut temp_i = vec![0.0f64; si];
    let mut temp_j = vec![0.0f64; sj];

    // iDCT along axis 0 (i)
    for j in 0..sj {
        for k in 0..sk {
            for i in 0..si {
                temp_i[i] = data[i * sj * sk + j * sk + k];
            }
            plans.dct3_i.process_dct3_with_scratch(&mut temp_i, &mut scratch_i);
            for i in 0..si {
                data[i * sj * sk + j * sk + k] = temp_i[i];
            }
        }
    }

    // iDCT along axis 1 (j)
    for i in 0..si {
        for k in 0..sk {
            for j in 0..sj {
                temp_j[j] = data[i * sj * sk + j * sk + k];
            }
            plans.dct3_j.process_dct3_with_scratch(&mut temp_j, &mut scratch_j);
            for j in 0..sj {
                data[i * sj * sk + j * sk + k] = temp_j[j];
            }
        }
    }

    // iDCT along axis 2 (k) - contiguous in memory
    for i in 0..si {
        for j in 0..sj {
            let offset = i * sj * sk + j * sk;
            plans.dct3_k.process_dct3_with_scratch(
                &mut data[offset..offset + sk],
                &mut scratch_k,
            );
        }
    }

    data
}

/// Reassemble blocks into a 3D array.
fn assemble_blocks(
    blocks: &[Vec<f64>],
    padded_shape: VolumeShape,
    block_shape: BlockShape,
) -> Array3<f64> {
    let nj = padded_shape[1] / block_shape[1];
    let nk = padded_shape[2] / block_shape[2];

    let mut array = Array3::<f64>::zeros((padded_shape[0], padded_shape[1], padded_shape[2]));

    for (idx, block) in blocks.iter().enumerate() {
        let bi = idx / (nj * nk);
        let remainder = idx % (nj * nk);
        let bj = remainder / nk;
        let bk = remainder % nk;

        let i0 = bi * block_shape[0];
        let j0 = bj * block_shape[1];
        let k0 = bk * block_shape[2];

        for li in 0..block_shape[0] {
            for lj in 0..block_shape[1] {
                for lk in 0..block_shape[2] {
                    let flat_idx = li * block_shape[1] * block_shape[2]
                        + lj * block_shape[2]
                        + lk;
                    array[[i0 + li, j0 + lj, k0 + lk]] = block[flat_idx];
                }
            }
        }
    }

    array
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
