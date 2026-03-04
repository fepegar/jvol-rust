# CLI Reference

JVol Rust provides three subcommands: `encode`, `decode`, and `bench`.

## `jvol-rust encode`

Encode a NIfTI file into a compressed `.jvol` file.

```
jvol-rust encode [OPTIONS] <INPUT> <OUTPUT>
```

### Arguments

| Argument   | Description                         |
|------------|-------------------------------------|
| `<INPUT>`  | Input NIfTI file (`.nii` or `.nii.gz`) |
| `<OUTPUT>` | Output `.jvol` file path            |

### Options

| Option              | Default | Description                                  |
|---------------------|---------|----------------------------------------------|
| `-q`, `--quality`   | `60`    | JPEG quality (1–100, higher = better quality) |
| `-b`, `--block-size`| `8`     | Block size for 3D DCT                        |
| `-v`, `--verbose`   | off     | Print encoding details and timing            |

### Example

```bash
jvol-rust encode brain.nii.gz brain.jvol --quality 80 --block-size 8 --verbose
```

Outputs the elapsed time in seconds to stdout (for scripting/benchmarking).

---

## `jvol-rust decode`

Decode a `.jvol` file back to NIfTI format.

```
jvol-rust decode [OPTIONS] <INPUT> <OUTPUT>
```

### Arguments

| Argument   | Description                              |
|------------|------------------------------------------|
| `<INPUT>`  | Input `.jvol` file path                  |
| `<OUTPUT>` | Output NIfTI file (`.nii` or `.nii.gz`)  |

### Options

| Option            | Default | Description                        |
|-------------------|---------|------------------------------------|
| `-v`, `--verbose` | off     | Print decoding details and timing  |

### Example

```bash
jvol-rust decode brain.jvol brain_decoded.nii.gz --verbose
```

---

## `jvol-rust bench`

Benchmark the core encode and decode algorithms on a NIfTI file. This subcommand
times only the core algorithm (no file I/O overhead), making it ideal for fair
performance comparisons.

```
jvol-rust bench [OPTIONS] <INPUT>
```

### Arguments

| Argument  | Description                                 |
|-----------|---------------------------------------------|
| `<INPUT>` | Input NIfTI file (`.nii` or `.nii.gz`)      |

### Options

| Option              | Default | Description                                  |
|---------------------|---------|----------------------------------------------|
| `-q`, `--quality`   | `60`    | JPEG quality (1–100)                         |
| `-b`, `--block-size`| `8`     | Block size for 3D DCT                        |

### Example

```bash
jvol-rust bench brain.nii.gz --quality 60 --block-size 8
```

Outputs two space-separated floats: `encode_time decode_time` (in seconds).
