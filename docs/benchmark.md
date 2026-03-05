# Benchmark

Compression ratio and speed benchmarks across three brain MRI volumes.

## Test data

| Image | Shape | Data type | Raw size | NIfTI + gzip |
|-------|-------|-----------|----------|--------------|
| [Colin 1998](https://www.bic.mni.mcgill.ca/ServicesAtlases/Colin27) | 181 × 217 × 181 | float32 | 27.1 MB | 22.9 MB |
| [Colin 2008](https://www.bic.mni.mcgill.ca/ServicesAtlases/Colin27) | 362 × 434 × 362 | float32 | 217.0 MB | 98.1 MB |
| [FPG T1](https://torchio.readthedocs.io/) | 256 × 256 × 176 | uint16 | 22.0 MB | 10.4 MB |

**Hardware:** Apple Silicon (M-series)

## Compression ratios

### Colin 1998

| Format | Size | Ratio |
|--------|------|-------|
| Uncompressed NIfTI | 27.1 MB | 1.0× |
| NIfTI + gzip | 22.9 MB | 1.2× |
| **JVol lossless** | **22.9 MB** | **1.2×** |
| JVol lossy q=100 | 2.9 MB | 9.3× |
| JVol lossy q=80 | 1.4 MB | 19.7× |
| JVol lossy q=60 | 568 KB | 48.9× |
| JVol lossy q=40 | 189 KB | 146.9× |
| JVol lossy q=20 | 56.6 KB | 490.5× |

### Colin 2008

| Format | Size | Ratio |
|--------|------|-------|
| Uncompressed NIfTI | 217.0 MB | 1.0× |
| NIfTI + gzip | 98.1 MB | 2.2× |
| **JVol lossless** | **106.8 MB** | **2.0×** |
| JVol lossy q=100 | 11.3 MB | 19.1× |
| JVol lossy q=80 | 4.8 MB | 44.8× |
| JVol lossy q=60 | 1.7 MB | 125.6× |
| JVol lossy q=40 | 642 KB | 346.1× |
| JVol lossy q=20 | 218 KB | 1017.7× |

### FPG T1

| Format | Size | Ratio |
|--------|------|-------|
| Uncompressed NIfTI | 22.0 MB | 1.0× |
| NIfTI + gzip | 10.4 MB | 2.1× |
| **JVol lossless** | **14.0 MB** | **1.6×** |
| JVol lossy q=100 | 5.2 MB | 4.3× |
| JVol lossy q=80 | 2.7 MB | 8.2× |
| JVol lossy q=60 | 1.2 MB | 18.8× |
| JVol lossy q=40 | 364 KB | 62.0× |
| JVol lossy q=20 | 74.1 KB | 304.1× |

## Speed

### Algorithm-only times

Core encode/decode times excluding all file I/O (NIfTI reading/writing,
gzip, zstd):

| Image | Mode | Encode | Decode |
|-------|------|--------|--------|
| Colin 1998 | Lossless | 23 ms | 20 ms |
| Colin 1998 | Lossy q=60 | 75 ms | 69 ms |
| Colin 2008 | Lossless | 325 ms | 616 ms |
| Colin 2008 | Lossy q=60 | 835 ms | 439 ms |
| FPG T1 | Lossless | 116 ms | 180 ms |
| FPG T1 | Lossy q=60 | 139 ms | 135 ms |

### End-to-end times

Full pipeline including NIfTI I/O and entropy coding.

Decode times depend on the output format: writing `.nii.gz` requires
gzip recompression, which dominates the total time.

| Image | Mode | Encode | Decode (.nii) | Decode (.nii.gz) |
|-------|------|--------|---------------|------------------|
| Colin 1998 | Lossless | 1.24 s | 0.20 s | 1.26 s |
| Colin 1998 | Lossy q=60 | 0.48 s | — | 1.24 s |
| Colin 2008 | Lossless | 7.55 s | 1.49 s | 11.14 s |
| Colin 2008 | Lossy q=60 | 4.04 s | — | 10.05 s |
| FPG T1 | Lossless | 1.44 s | 0.39 s | 5.48 s |
| FPG T1 | Lossy q=60 | 0.54 s | — | 1.95 s |

!!! tip "For fastest decode, write to `.nii`"
    Decoding to uncompressed `.nii` avoids gzip recompression and is
    **5–7× faster** than writing `.nii.gz`.

### Comparison with gzip

| Image | gzip compress | gzip decompress |
|-------|---------------|-----------------|
| Colin 1998 | 1.40 s | 0.14 s |
| Colin 2008 | 9.81 s | 0.90 s |
| FPG T1 | 2.05 s | 0.10 s |

## Notes

### Lossless compression on float data

The Colin volumes use `float32` data with a NIfTI intensity scaling
factor (`scl_slope`), resulting in non-integer values with high entropy.
General-purpose compressors like gzip and zstd achieve only ~2× on this
data. JVol lossless matches gzip for float data and can outperform it
on integer-typed volumes (e.g., `int16`, `uint8`) where the 3D Lorenzo
predictor exploits spatial correlation.

### No block size parameter

Unlike block-based codecs (e.g., JPEG), JVol's wavelet codec operates
on the **full 3D volume** without spatial blocking. This eliminates
block artifacts and allows the wavelet transform to capture correlations
across the entire volume. The quality parameter (1–100) controls the
quantization step size.

## Reproducing

```bash
# Build optimized binary
mise run build

# Algorithm-only benchmark
./target/release/jvol-rust bench <input.nii.gz> -q 60
./target/release/jvol-rust bench <input.nii.gz> --lossless

# End-to-end encode/decode
uv run jvol encode input.nii.gz output.jvol -q 60 -v
uv run jvol decode output.jvol decoded.nii -v
```
