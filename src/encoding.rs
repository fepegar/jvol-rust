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

/// Encode a 3D array into compressed DC and AC RLE sequences.
pub fn encode_array(
    array: &ArrayView3<f64>,
    block_size: usize,
    quality: u8,
) -> EncodeResult {
    let block_shape: BlockShape = [block_size, block_size, block_size];
    let quantization_table = get_quantization_table(block_shape, quality);

    // Normalize intensities to [0, 255] then shift to [-128, 127]
    let min_val = array.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = array.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let intercept = min_val;
    let slope = max_val - min_val;

    let mut normalized = array.to_owned();
    if slope > 0.0 {
        normalized.mapv_inplace(|v| ((v - intercept) / slope) * 255.0 - 128.0);
    } else {
        normalized.mapv_inplace(|_| 0.0);
    }

    let padded = pad_array(&normalized.view(), block_shape);
    let blocks = split_into_blocks(&padded, block_shape);

    // Pre-plan DCT (shared across threads via Arc)
    let plans = DctPlans::new(block_shape);

    // Parallel DCT + quantize in one pass
    let quantized_blocks: Vec<Vec<i32>> = blocks
        .par_iter()
        .map(|block| {
            let data = dct3d_with_plans(block, block_shape, &plans);
            data.iter()
                .zip(quantization_table.iter())
                .map(|(&v, &q)| (v / q as f64).round() as i32)
                .collect()
        })
        .collect();

    // Generate scan indices (zigzag by distance from origin)
    let scan_indices = get_scan_indices(block_shape);

    // Separate DC and AC components
    let (dc_sequence, ac_sequence) =
        blocks_to_sequence(&quantized_blocks, &scan_indices, block_shape);

    let dc_rle = run_length_encode(&dc_sequence);
    let ac_rle = run_length_encode(&ac_sequence);

    EncodeResult {
        dc_rle,
        ac_rle,
        quantization_table,
        intercept,
        slope,
    }
}

/// Pad the array so each dimension is divisible by the block size.
pub fn pad_array(array: &ArrayView3<f64>, block_shape: BlockShape) -> Array3<f64> {
    let shape = array.shape();
    let mut padded_shape = [0usize; 3];
    let mut needs_padding = false;

    for d in 0..3 {
        let remainder = shape[d] % block_shape[d];
        if remainder == 0 {
            padded_shape[d] = shape[d];
        } else {
            padded_shape[d] = shape[d] + (block_shape[d] - remainder);
            needs_padding = true;
        }
    }

    if !needs_padding {
        return array.to_owned();
    }

    let mut padded = Array3::<f64>::zeros((padded_shape[0], padded_shape[1], padded_shape[2]));
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

/// Compute 3D DCT-II using pre-planned DCT objects.
fn dct3d_with_plans(block: &[f64], block_shape: BlockShape, plans: &DctPlans) -> Vec<f64> {
    let [si, sj, sk] = block_shape;
    let mut data = block.to_vec();

    // Thread-local scratch buffers
    let mut scratch_k = vec![0.0f64; plans.scratch_len_k];
    let mut scratch_j = vec![0.0f64; plans.scratch_len_j];
    let mut scratch_i = vec![0.0f64; plans.scratch_len_i];
    let mut temp_j = vec![0.0f64; sj];
    let mut temp_i = vec![0.0f64; si];

    // DCT along axis 2 (k) - contiguous in memory
    for i in 0..si {
        for j in 0..sj {
            let offset = i * sj * sk + j * sk;
            plans.dct2_k.process_dct2_with_scratch(
                &mut data[offset..offset + sk],
                &mut scratch_k,
            );
        }
    }

    // DCT along axis 1 (j)
    for i in 0..si {
        for k in 0..sk {
            for j in 0..sj {
                temp_j[j] = data[i * sj * sk + j * sk + k];
            }
            plans.dct2_j.process_dct2_with_scratch(&mut temp_j, &mut scratch_j);
            for j in 0..sj {
                data[i * sj * sk + j * sk + k] = temp_j[j];
            }
        }
    }

    // DCT along axis 0 (i)
    for j in 0..sj {
        for k in 0..sk {
            for i in 0..si {
                temp_i[i] = data[i * sj * sk + j * sk + k];
            }
            plans.dct2_i.process_dct2_with_scratch(&mut temp_i, &mut scratch_i);
            for i in 0..si {
                data[i * sj * sk + j * sk + k] = temp_i[i];
            }
        }
    }

    data
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

/// Convert quantized blocks into DC and AC sequences using the scan ordering.
fn blocks_to_sequence(
    blocks: &[Vec<i32>],
    scan_indices: &[[usize; 3]],
    block_shape: BlockShape,
) -> (Vec<i32>, Vec<i32>) {
    let [_si, sj, sk] = block_shape;
    let num_blocks = blocks.len();

    // DC: first element (0,0,0) of each block
    let dc_sequence: Vec<i32> = blocks.iter().map(|b| b[0]).collect();

    // AC: all other elements in scan order
    let block_size = scan_indices.len();
    let mut ac_sequence = Vec::with_capacity(num_blocks * (block_size - 1));
    for idx in &scan_indices[1..] {
        let flat_idx = idx[0] * sj * sk + idx[1] * sk + idx[2];
        for block in blocks {
            ac_sequence.push(block[flat_idx]);
        }
    }

    (dc_sequence, ac_sequence)
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
