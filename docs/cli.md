# CLI Reference

JVol provides two CLIs: a Python package (`jvol`) and a Rust binary
(`jvol-rust`). Both share the same interface.

## `encode`

Encode a NIfTI file into a compressed `.jvol` file.

```
jvol encode [OPTIONS] <INPUT> <OUTPUT>
```

### Arguments

| Argument   | Description                            |
|------------|----------------------------------------|
| `<INPUT>`  | Input NIfTI file (`.nii` or `.nii.gz`) |
| `<OUTPUT>` | Output `.jvol` file path               |

### Options

| Option              | Default | Description                                         |
|---------------------|---------|-----------------------------------------------------|
| `-q`, `--quality`   | `60`    | Quality level (1–100 for lossy, 0 for lossless)     |
| `-l`, `--lossless`  | off     | Lossless mode (overrides quality to 0)              |
| `-v`, `--verbose`   | off     | Print encoding details and timing                   |

### Examples

```bash
# Lossy encode at default quality (60)
jvol encode brain.nii.gz brain.jvol

# High-quality lossy
jvol encode brain.nii.gz brain.jvol -q 90

# Lossless
jvol encode brain.nii.gz brain.jvol --lossless

# Verbose output with timing
jvol encode brain.nii.gz brain.jvol -l -v
```

---

## `decode`

Decode a `.jvol` file back to NIfTI format.

```
jvol decode [OPTIONS] <INPUT> <OUTPUT>
```

### Arguments

| Argument   | Description                             |
|------------|-----------------------------------------|
| `<INPUT>`  | Input `.jvol` file path                 |
| `<OUTPUT>` | Output NIfTI file (`.nii` or `.nii.gz`) |

### Options

| Option            | Default | Description                       |
|-------------------|---------|-----------------------------------|
| `-v`, `--verbose` | off     | Print decoding details and timing |

### Examples

```bash
# Decode to uncompressed NIfTI (fastest)
jvol decode brain.jvol brain_decoded.nii

# Decode to gzip-compressed NIfTI
jvol decode brain.jvol brain_decoded.nii.gz
```

!!! tip
    Decoding to `.nii` is **5–7× faster** than `.nii.gz` because it skips
    gzip recompression.

---

## `bench` (Rust CLI only)

Benchmark the core encode and decode algorithms on a NIfTI file.
Times only the codec (no file I/O), making it ideal for performance
comparisons.

```
jvol-rust bench [OPTIONS] <INPUT>
```

### Arguments

| Argument  | Description                            |
|-----------|----------------------------------------|
| `<INPUT>` | Input NIfTI file (`.nii` or `.nii.gz`) |

### Options

| Option              | Default | Description                                     |
|---------------------|---------|-------------------------------------------------|
| `-q`, `--quality`   | `60`    | Quality level (1–100 for lossy, 0 for lossless) |
| `-l`, `--lossless`  | off     | Lossless mode                                   |

### Example

```bash
jvol-rust bench brain.nii.gz --lossless
jvol-rust bench brain.nii.gz -q 60
```

Outputs two space-separated floats: `encode_time decode_time` (in seconds).
