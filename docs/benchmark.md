# Benchmark

Comparison of the Python [JVol](https://github.com/fepegar/jvol) implementation
against this Rust implementation.

## Setup

- **Test data**: Colin27 1998 T1 brain MRI (`colin27_t1_tal_lin.nii.gz`)
- **Volume shape**: 181 × 217 × 181 voxels
- **File size**: 24.1 MB (gzip-compressed NIfTI)
- **Parameters**: quality = 60, block size = 8
- **Runs**: 5 iterations, median reported
- **Hardware**: Apple Silicon (M-series)

## Results

|            | Python   | Rust     | Speedup  |
|------------|----------|----------|----------|
| **Encode** | 0.686 s  | 0.106 s  | **6.5×** |
| **Decode** | 0.193 s  | 0.076 s  | **2.5×** |

!!! note "What's measured"
    - **Rust** times measure the core algorithm only (no NIfTI file I/O)
    - **Python** times include `.npz` file I/O but not NIfTI I/O
    - Both exclude the time to load/write NIfTI files for fair comparison

## Key optimizations in the Rust implementation

1. **Pre-planned DCT** — DCT plans are computed once and shared across threads
   via `Arc`, eliminating per-block FFT planning overhead
2. **Rayon parallelism** — all blocks are processed in parallel for both
   encoding (DCT + quantization) and decoding (inverse DCT + dequantization)
3. **Fused passes** — DCT and quantization are combined in a single parallel
   pass, reducing memory allocations
4. **Release-mode LTO** — link-time optimization enabled for maximum
   single-threaded performance

## Reproducing

Run the benchmark script from the repository root:

```bash
uv run benchmark.py
```

This requires:

- The Rust binary built in release mode (`cargo build --release`)
- The Colin27 1998 T1 NIfTI file at
  `~/.cache/torchio/mni_colin27_1998_nifti/colin27_t1_tal_lin.nii.gz`

The script will install the Python `jvol` package automatically via `uv`.
