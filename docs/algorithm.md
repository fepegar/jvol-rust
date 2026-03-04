# Algorithm

JVol Rust implements a 3D variant of the JPEG compression algorithm, adapted for
volumetric medical images. This page describes each stage of the pipeline.

## Overview

The compression pipeline follows the classic JPEG approach, extended from 2D to 3D:

```
┌─────────┐   ┌────────┐   ┌───────┐   ┌──────────┐   ┌────────┐   ┌─────┐
│  Input  │──▶│  Pad   │──▶│ Split │──▶│ 3D DCT  │──▶│Quantize│──▶│ RLE │
│ Volume  │   │ Array  │   │Blocks │   │(per blk) │   │        │   │     │
└─────────┘   └────────┘   └───────┘   └──────────┘   └────────┘   └─────┘
```

## Encoding

### 1. Intensity normalization

The input array is linearly rescaled to the range `[-128, 127]`:

```
normalized = (value - min) / (max - min) × 255 - 128
```

The original `min` (intercept) and `max - min` (slope) are stored as metadata
for lossless reversal during decoding.

### 2. Padding

The volume is zero-padded so that each spatial dimension is divisible by the
block size. For example, a 181 × 217 × 181 volume with block size 8 is padded
to 184 × 224 × 184.

### 3. Block splitting

The padded volume is split into non-overlapping cubic blocks. With block size 8,
each block is 8 × 8 × 8 = 512 voxels.

### 4. 3D Discrete Cosine Transform (DCT)

A **separable 3D DCT-II** is applied to each block independently. "Separable" means
the 3D transform is decomposed into three sequential 1D DCTs along each axis:

1. DCT along the k-axis (innermost, contiguous in memory)
2. DCT along the j-axis
3. DCT along the i-axis

This is mathematically equivalent to the full 3D DCT but much more efficient.
The implementation uses the [`rustdct`](https://crates.io/crates/rustdct) crate
for optimized 1D DCT computation.

!!! tip "Parallelism"
    Blocks are processed in parallel using [Rayon](https://crates.io/crates/rayon).
    DCT plans are pre-computed once and shared across threads via `Arc`, avoiding
    redundant FFT planning overhead.

### 5. Quantization

Each DCT coefficient is divided by the corresponding value in a **quantization table**
and rounded to the nearest integer:

```
quantized[i][j][k] = round(dct[i][j][k] / qtable[i][j][k])
```

The quantization table is generated based on the L2 distance of each frequency index
from the origin (DC component at `[0, 0, 0]`). Higher frequencies get larger divisors,
causing more aggressive rounding — this is where the lossy compression happens.

The `quality` parameter (1–100) controls a multiplier on the quantization table:

- **Quality 1**: very aggressive quantization → small file, low fidelity
- **Quality 100**: minimal quantization → large file, high fidelity

### 6. Zigzag scan

The 3D block indices are sorted by their L2 distance from the origin `[0, 0, 0]`.
This ordering ensures that low-frequency (high-energy) coefficients come first,
followed by high-frequency (near-zero) coefficients — maximizing RLE effectiveness.

The DC component (index `[0, 0, 0]`) is always first.

### 7. Sequence separation

After scanning, the coefficients are split into two sequences:

- **DC sequence**: one value per block (the `[0, 0, 0]` coefficient)
- **AC sequence**: all remaining coefficients in scan order

### 8. Run-length encoding (RLE)

Both the DC and AC sequences are run-length encoded: consecutive identical values
are stored as `(value, count)` pairs. Since quantization produces many zeros,
especially in high-frequency AC coefficients, RLE achieves significant compression.

## Decoding

Decoding reverses each step:

1. **RLE decode** — expand `(value, count)` pairs back into full sequences
2. **Sequence to blocks** — reconstruct 3D blocks from DC + AC sequences
3. **Inverse quantization** — multiply by the quantization table
4. **3D inverse DCT** (DCT-III) — transform back to spatial domain
5. **Reassemble** — place blocks back into the full volume grid
6. **Rescale** — undo the intensity normalization using stored intercept and slope
7. **Crop** — remove padding to restore the original volume shape
8. **Clip** — clamp values to the valid range for the original data type

## File format

The `.jvol` file is a **gzip-compressed bincode** archive containing:

| Field                | Type       | Description                              |
|----------------------|------------|------------------------------------------|
| `shape`              | `[u64; 3]` | Original volume dimensions               |
| `ijk_to_ras`         | `[[f64; 4]; 4]` | Affine transformation matrix        |
| `dtype`              | enum       | Original data type (U8, I16, F32, etc.)  |
| `intercept`          | `f64`      | Minimum intensity value                  |
| `slope`              | `f64`      | Intensity range (max - min)              |
| `block_shape`        | `[u64; 3]` | Block dimensions                         |
| `quality`            | `u8`       | Quality parameter used for encoding      |
| `quantization_table` | `Vec<f32>` | Flattened quantization table             |
| `dc_rle`             | RLE data   | Run-length encoded DC coefficients       |
| `ac_rle`             | RLE data   | Run-length encoded AC coefficients       |
