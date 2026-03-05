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

/// Reusable scratch buffers for inverse DCT (one per rayon thread).
struct IdctScratch {
    scratch_i: Vec<f64>,
    scratch_j: Vec<f64>,
    scratch_k: Vec<f64>,
    temp_i: Vec<f64>,
    temp_j: Vec<f64>,
}

impl IdctScratch {
    fn new(block_shape: BlockShape, plans: &IdctPlans) -> Self {
        let [si, sj, _sk] = block_shape;
        Self {
            scratch_i: vec![0.0; plans.scratch_len_i],
            scratch_j: vec![0.0; plans.scratch_len_j],
            scratch_k: vec![0.0; plans.scratch_len_k],
            temp_i: vec![0.0; si],
            temp_j: vec![0.0; sj],
        }
    }
}

/// Decode RLE-compressed DC and AC sequences back into a 3D array.
#[allow(clippy::too_many_arguments)]
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
    let dc_sequence = run_length_decode(dc_rle);
    let ac_sequence = run_length_decode(ac_rle);

    let scan_indices = get_scan_indices(block_shape);
    let bs = block_shape[0];
    let bt = bs * bs * bs;
    let num_blocks = dc_sequence.len();

    // Pre-allocate flat buffer and fill with dequantized values (fused)
    let [_si, sj, sk] = block_shape;
    let mut block_data: Vec<f64> = vec![0.0; num_blocks * bt];

    // DC component
    for (b, &dc) in dc_sequence.iter().enumerate() {
        block_data[b * bt] = dc as f64 * quantization_table[0] as f64;
    }

    // AC components — dequantize during placement
    for (scan_pos, idx) in scan_indices[1..].iter().enumerate() {
        let flat_idx = idx[0] * sj * sk + idx[1] * sk + idx[2];
        let q = quantization_table[flat_idx] as f64;
        let ac_offset = scan_pos * num_blocks;
        for b in 0..num_blocks {
            block_data[b * bt + flat_idx] = ac_sequence[ac_offset + b] as f64 * q;
        }
    }

    // Parallel iDCT in-place with thread-local scratch
    let plans = IdctPlans::new(block_shape);
    block_data
        .par_chunks_mut(bt)
        .for_each_init(
            || IdctScratch::new(block_shape, &plans),
            |scratch, block| {
                idct3d_inplace(block, block_shape, &plans, scratch);
            },
        );

    // Direct assembly with rescaling — no intermediate padded array, no crop copy
    let ps = padded_shape(target_shape, block_shape);
    let nj_b = ps[1] / bs;
    let nk_b = ps[2] / bs;
    let scale = slope / 255.0;
    let offset = 128.0 * scale + intercept;

    let mut array = Array3::<f64>::zeros((target_shape[0], target_shape[1], target_shape[2]));

    for idx in 0..num_blocks {
        let bi = idx / (nj_b * nk_b);
        let rem = idx % (nj_b * nk_b);
        let bj = rem / nk_b;
        let bk = rem % nk_b;

        let i0 = bi * bs;
        let j0 = bj * bs;
        let k0 = bk * bs;
        let block = &block_data[idx * bt..(idx + 1) * bt];

        let i_end = (i0 + bs).min(target_shape[0]);
        let j_end = (j0 + bs).min(target_shape[1]);
        let k_end = (k0 + bs).min(target_shape[2]);

        for gi in i0..i_end {
            for gj in j0..j_end {
                for gk in k0..k_end {
                    let li = gi - i0;
                    let lj = gj - j0;
                    let lk = gk - k0;
                    let v = block[li * bs * bs + lj * bs + lk];
                    array[[gi, gj, gk]] = v * scale + offset;
                }
            }
        }
    }

    // Clip to dtype range if integer
    if let Some((min_val, max_val)) = dtype.iinfo() {
        array.mapv_inplace(|v| v.max(min_val).min(max_val));
    }

    array
}

/// 3D inverse DCT (DCT-III) in-place using reusable scratch buffers.
fn idct3d_inplace(
    data: &mut [f64],
    block_shape: BlockShape,
    plans: &IdctPlans,
    scratch: &mut IdctScratch,
) {
    let [si, sj, sk] = block_shape;

    // iDCT along axis 0 (i)
    for j in 0..sj {
        for k in 0..sk {
            for i in 0..si {
                scratch.temp_i[i] = data[i * sj * sk + j * sk + k];
            }
            plans
                .dct3_i
                .process_dct3_with_scratch(&mut scratch.temp_i, &mut scratch.scratch_i);
            for i in 0..si {
                data[i * sj * sk + j * sk + k] = scratch.temp_i[i];
            }
        }
    }

    // iDCT along axis 1 (j)
    for i in 0..si {
        for k in 0..sk {
            for j in 0..sj {
                scratch.temp_j[j] = data[i * sj * sk + j * sk + k];
            }
            plans
                .dct3_j
                .process_dct3_with_scratch(&mut scratch.temp_j, &mut scratch.scratch_j);
            for j in 0..sj {
                data[i * sj * sk + j * sk + k] = scratch.temp_j[j];
            }
        }
    }

    // iDCT along axis 2 (k) — contiguous in memory
    for i in 0..si {
        for j in 0..sj {
            let off = i * sj * sk + j * sk;
            plans
                .dct3_k
                .process_dct3_with_scratch(&mut data[off..off + sk], &mut scratch.scratch_k);
        }
    }
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
