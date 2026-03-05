# JVol Rust

<div align="center">
  <img src="docs/images/logo.png" alt="JVol logo" width="200">
</div>

<p align="center">
  Lightning-fast wavelet compression for 3D medical images, written in Rust.
</p>

---

JVol Rust compresses 3D medical images (NIfTI volumes) using a wavelet-based
codec with optional lossless mode. It provides both a Rust CLI and a Python
package with Rust bindings.

## Features

- **Fast** — sub-second encoding on typical brain MRI volumes
- **Lossless** — exact roundtrip for integer data types, beats gzip on `uint16`/`int16` volumes
- **Lossy** — up to 1000× compression at low quality settings
- **Simple CLI** — available as `jvol-rust` (Rust) or `jvol` (Python)
- **NIfTI support** — reads `.nii` and `.nii.gz` files directly

## Quick start

```bash
pip install jvol

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

✓ = beats NIfTI + gzip

## Documentation

📖 **[fepegar.github.io/jvol-rust](https://fepegar.github.io/jvol-rust/)**
