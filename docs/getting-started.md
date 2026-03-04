# Getting Started

## Installation

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

### Encode a NIfTI volume

```bash
jvol-rust encode brain.nii.gz brain.jvol
```

### Decode back to NIfTI

```bash
jvol-rust decode brain.jvol brain_decoded.nii.gz
```

### Adjust quality

Higher quality means better fidelity but larger file size (1–100, default: 60):

```bash
jvol-rust encode brain.nii.gz brain.jvol --quality 80
```

### Adjust block size

Larger blocks improve quality at the cost of compression ratio (default: 8):

```bash
jvol-rust encode brain.nii.gz brain.jvol --block-size 16
```

## File format

JVol Rust uses a custom binary format (`.jvol`):

- The file is a **gzip-compressed bincode** archive
- Contains: metadata (shape, affine, dtype, intercept, slope, block shape, quality),
  quantization table, and RLE-encoded DC/AC coefficient sequences

!!! note
    The Rust `.jvol` format is **not** compatible with the Python `jvol` format
    (which uses NumPy `.npz` archives). Each implementation reads/writes its own format.
