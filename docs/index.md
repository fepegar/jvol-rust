# JVol Rust

Lightning-fast JPEG-like compression for 3D medical images, written in Rust.

---

JVol Rust is a high-performance reimplementation of [JVol](https://github.com/fepegar/jvol) that compresses 3D medical images (NIfTI volumes) using a JPEG-inspired algorithm adapted for volumetric data.

## Features

- **Fast** — 6.5× faster encoding and 2.5× faster decoding than the Python implementation
- **Parallel** — leverages all CPU cores via [Rayon](https://github.com/rayon-rs/rayon) for block-level DCT computation
- **Compact** — achieves ~49× compression on typical brain MRI volumes
- **Simple CLI** — encode and decode with a single command
- **NIfTI support** — reads `.nii` and `.nii.gz` files directly

## Quick start

```bash
# Encode a NIfTI volume to .jvol
jvol-rust encode brain.nii.gz brain.jvol --quality 60 --block-size 8

# Decode back to NIfTI
jvol-rust decode brain.jvol brain_decoded.nii.gz
```

## Benchmark results

Tested on the Colin27 1998 T1 brain MRI (181 × 217 × 181 voxels):

|            | Python | Rust   | Speedup |
|------------|--------|--------|---------|
| **Encode** | 0.686s | 0.106s | **6.5×** |
| **Decode** | 0.193s | 0.076s | **2.5×** |

See the [Benchmark](benchmark.md) page for full methodology and results.

## How it works

The algorithm applies a 3D variant of the JPEG compression pipeline:

1. **Pad** the volume so each dimension is divisible by the block size
2. **Split** into non-overlapping 3D blocks (e.g., 8×8×8)
3. **3D DCT** (Discrete Cosine Transform) on each block
4. **Quantize** the DCT coefficients using a quality-dependent table
5. **Zigzag scan** coefficients by distance from origin (DC → high-frequency AC)
6. **Run-length encode** the DC and AC sequences separately
7. **Serialize** with metadata (affine, dtype, intercept, slope)

Read more in the [Algorithm](algorithm.md) page.
