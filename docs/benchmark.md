# Benchmark

Comparison of the Python [JVol](https://github.com/fepegar/jvol) implementation
against this Rust implementation.

## Setup

- **Test data**: Colin27 2008 T1 brain MRI (`colin27_t1_tal_hires.nii.gz`)
- **Volume shape**: 362 × 434 × 362 voxels (~57M voxels)
- **Parameters**: quality = 60, block size = 8
- **Hardware**: Apple Silicon (M-series)

## Results

|            | Python    | Rust      | Speedup   |
|------------|-----------|-----------|-----------|
| **Encode** | ~5.5 s    | 0.56 s    | **~10×**  |
| **Decode** | ~1.5 s    | 0.76 s    | **~2×**   |

!!! note "What's measured"
    Times measure the core algorithm only (no NIfTI file I/O) for a fair
    comparison.

## Optimization techniques

The Rust implementation uses several layers of optimization to achieve
these results.

### 1. Zero per-block heap allocations

The original approach allocated multiple `Vec` buffers for every block
(~15K blocks for the standard volume, ~56K for hires). Each block
triggered allocations for DCT scratch buffers, quantized output, and
intermediate results — adding up to hundreds of MB of malloc/free churn.

The optimized version uses `rayon::for_each_init` to create **one set of
scratch buffers per thread** that are reused across all blocks processed
by that thread:

```rust
quantized_flat
    .par_chunks_mut(bt)
    .enumerate()
    .for_each_init(
        || DctScratch::new(block_shape, &plans),
        |scratch, (idx, out)| {
            // scratch.block_buf, scratch.temp_i, etc. are reused
            // across all blocks on this thread
        },
    );
```

### 2. Fused normalize + pad + extract

The original code made three full passes over the data:

1. Copy array → normalized array
2. Copy normalized → padded array
3. Split padded → individual block `Vec`s

The optimized version does all three in one step: block data is extracted
directly from the source array with on-the-fly normalization and
implicit zero-padding (out-of-bounds indices return a pre-computed
padding value):

```rust
buf[flat] = if gi < shape[0] && gj < shape[1] && gk < shape[2] {
    (array[[gi, gj, gk]] - intercept) * inv_slope - 128.0
} else {
    pad_val  // pre-computed normalized padding value
};
```

This eliminates ~180 MB of intermediate allocations for the hires volume.

### 3. In-place DCT with reusable scratch

The DCT operates directly on `scratch.block_buf` rather than allocating a
new `Vec` per block. Scratch buffers for the 1D DCT transforms
(`scratch_i/j/k`, `temp_i/j`) are also reused across blocks.

### 4. Fused dequantize + sequence reconstruction (decode)

Instead of first reconstructing `Vec<Vec<i32>>` blocks from DC/AC
sequences and then dequantizing each block, the decode path writes
dequantized `f64` values directly into a single flat buffer:

```rust
for (b, &dc) in dc_sequence.iter().enumerate() {
    block_data[b * bt] = dc as f64 * quantization_table[0] as f64;
}
```

### 5. Direct assembly without crop (decode)

The original decode path allocated a full padded array, wrote all blocks
into it, ran a rescaling pass, then copied the cropped region. The
optimized version writes rescaled values directly into the target-sized
output array, skipping padded voxels entirely:

```rust
let i_end = (i0 + bs).min(target_shape[0]);
// ...
array[[gi, gj, gk]] = v * scale + offset;
```

### 6. Buffered NIfTI I/O

NIfTI writes previously called `write_f32` for each individual voxel
(~57M calls through a `GzEncoder`). Now all voxel data is converted to a
byte buffer first, then written in a single `write_all` call.

### 7. Pre-planned DCT shared via Arc

DCT plans are computed once and shared across rayon threads via `Arc`,
eliminating per-block FFT planning overhead.

### 8. Single-pass min/max

Two separate folds over the array for min and max are combined into one.

### 9. Release profile with LTO

Link-time optimization is enabled in the release profile for maximum
cross-crate inlining.

## Reproducing

```bash
cargo build --release
./target/release/jvol-rust bench ~/.cache/torchio/mni_colin27_2008_nifti/colin27_t1_tal_hires.nii.gz
```
