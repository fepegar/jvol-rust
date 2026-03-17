# JVol Rust

<div align="center">
  <img src="images/logo.png" alt="JVol logo" width="200">
</div>

Lightning-fast wavelet compression for 3D medical images, written in Rust.

---

JVol Rust compresses 3D medical images (NIfTI volumes) using a wavelet-based
codec with optional lossless mode. It provides both a Rust CLI and a Python
package.

## Features

- **Fast** — sub-second encoding on typical brain MRI volumes
- **Lossless** — exact roundtrip for integer data types, beats gzip on `uint16`/`int16` volumes
- **Lossy** — up to 1000× compression at low quality settings
- **Simple CLI** — available as `jvol-rust` (Rust) or `jvol` (Python)
- **NIfTI support** — reads `.nii` and `.nii.gz` files directly

## Quick start

Examples below use the Python CLI (`jvol`). If you installed the Rust binary,
replace `jvol` with `jvol-rust`.

```bash
# Lossy encode (default quality=60)
jvol encode brain.nii.gz brain.jvol

# Lossless encode
jvol encode brain.nii.gz brain.jvol --lossless

# Decode back to NIfTI
jvol decode brain.jvol brain_decoded.nii
```

## Benchmark highlights

| Image | Uncompressed | gzip | JVol lossless | JVol lossy q=60 |
|-------|-------------|------|---------------|-----------------|
| Colin 1998 (f32) | 27.1 MB | 22.9 MB | **22.2 MB** ✓ | 569 KB (49×) |
| Colin 2008 (f32) | 217.0 MB | 98.1 MB | 106.3 MB | 1.7 MB (126×) |
| FPG T1 (u16) | 22.0 MB | 10.4 MB | **9.4 MB** ✓ | 1.2 MB (19×) |

✓ = beats NIfTI + gzip. See the [Benchmark](benchmark.md) page for full results.

## How it works

JVol uses a **3D wavelet transform** (CDF 9/7) for lossy compression and
**dtype-aware prediction** for lossless mode:

- **Lossy:** 3D DWT → dead-zone quantization → per-subband Rice coding → zstd
- **Lossless (integer):** delta prediction + byte-shuffle → zstd
- **Lossless (float):** raw Fortran-order bytes → zstd

Read more in the [Algorithm](algorithm.md) page.
