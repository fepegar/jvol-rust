# Getting Started

## Installation

### Python package (recommended)

```bash
pip install jvol
```

### Pre-built binaries

Download the latest release for your platform from the
[Releases page](https://github.com/fepegar/jvol-rust/releases).

=== "macOS (Apple Silicon)"

    ```bash
    curl -LO https://github.com/fepegar/jvol-rust/releases/latest/download/jvol-rust-aarch64-apple-darwin.tar.gz
    tar xzf jvol-rust-aarch64-apple-darwin.tar.gz
    sudo mv jvol-rust /usr/local/bin/
    ```

=== "macOS (Intel)"

    ```bash
    curl -LO https://github.com/fepegar/jvol-rust/releases/latest/download/jvol-rust-x86_64-apple-darwin.tar.gz
    tar xzf jvol-rust-x86_64-apple-darwin.tar.gz
    sudo mv jvol-rust /usr/local/bin/
    ```

=== "Linux (x86_64)"

    ```bash
    curl -LO https://github.com/fepegar/jvol-rust/releases/latest/download/jvol-rust-x86_64-unknown-linux-gnu.tar.gz
    tar xzf jvol-rust-x86_64-unknown-linux-gnu.tar.gz
    sudo mv jvol-rust /usr/local/bin/
    ```

### Building from source

Requires [Rust](https://rustup.rs/) 1.80 or later.

```bash
git clone https://github.com/fepegar/jvol-rust.git
cd jvol-rust
cargo build --release
```

The binary will be at `target/release/jvol-rust`.

## Basic usage

### Lossy encode (default)

```bash
jvol encode brain.nii.gz brain.jvol
```

This uses quality 60 by default, giving ~50× compression on typical brain MRI.

### Lossless encode

```bash
jvol encode brain.nii.gz brain.jvol --lossless
```

Exact roundtrip — no information loss. Beats gzip on integer-typed volumes.

### Decode back to NIfTI

```bash
jvol decode brain.jvol brain_decoded.nii
```

!!! tip
    Decode to `.nii` (not `.nii.gz`) for fastest performance — avoids gzip
    recompression.

### Adjust quality

Higher quality means better fidelity but larger file size (1–100, default: 60):

```bash
jvol encode brain.nii.gz brain.jvol --quality 80
```

## File format

JVol uses a custom binary format (`.jvol`):

- The file is a **zstd-compressed bincode** archive
- Contains: metadata (shape, affine, dtype, wavelet type, quality) and
  per-channel encoded data (Rice-coded subbands for lossy, delta-coded
  bytes for lossless)

!!! note
    The Rust `.jvol` format is **not** compatible with the Python `jvol` format
    (which uses NumPy `.npz` archives). Each implementation reads/writes its own format.
