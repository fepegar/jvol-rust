"""Benchmark comparing Python jvol vs Rust jvol-rust.

Usage:
    uv run benchmark.py
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jvol @ git+https://github.com/fepegar/jvol.git",
#     "numpy",
# ]
# ///

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

NIFTI_PATH = Path.home() / ".cache/torchio/mni_colin27_1998_nifti/colin27_t1_tal_lin.nii.gz"
RUST_BINARY = Path(__file__).parent / "target/release/jvol-rust"
QUALITY = 60
BLOCK_SIZE = 8
N_RUNS = 5


def benchmark_python_encode(array, ijk_to_ras, output_path):
    """Time the Python jvol encoding."""
    from jvol import JpegVolume

    jv = JpegVolume(array, ijk_to_ras)
    start = time.perf_counter()
    jv.save(output_path, block_size=BLOCK_SIZE, quality=QUALITY)
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_python_decode(jvol_path):
    """Time the Python jvol decoding."""
    from jvol import JpegVolume

    start = time.perf_counter()
    jv = JpegVolume.open(jvol_path)
    _ = jv.array  # force the decode
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_rust(input_path):
    """Time the Rust jvol-rust core encode+decode (no file I/O)."""
    result = subprocess.run(
        [
            str(RUST_BINARY),
            "bench",
            str(input_path),
            "--quality",
            str(QUALITY),
            "--block-size",
            str(BLOCK_SIZE),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Rust bench error: {result.stderr}")
        return None, None
    parts = result.stdout.strip().split()
    return float(parts[0]), float(parts[1])


def main():
    if not NIFTI_PATH.exists():
        print(f"Error: NIfTI file not found: {NIFTI_PATH}")
        return

    if not RUST_BINARY.exists():
        print(f"Error: Rust binary not found: {RUST_BINARY}")
        print("Run 'cargo build --release' first.")
        return

    print("=" * 70)
    print("JVol Benchmark: Python vs Rust")
    print("=" * 70)
    print(f"Input file:  {NIFTI_PATH}")
    print(f"File size:   {NIFTI_PATH.stat().st_size / 1e6:.1f} MB")
    print(f"Quality:     {QUALITY}")
    print(f"Block size:  {BLOCK_SIZE}")
    print(f"Runs:        {N_RUNS}")
    print()

    # Load the NIfTI file for Python
    from jvol.io import open_itk_image

    array, ijk_to_ras = open_itk_image(NIFTI_PATH)
    print(f"Array shape: {array.shape}")
    print(f"Array dtype: {array.dtype}")
    print()

    py_encode_times = []
    py_decode_times = []
    rust_encode_times = []
    rust_decode_times = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        py_jvol = tmpdir / "python.jvol"

        print(f"{'Run':<6} {'Py Enc (s)':<14} {'Rs Enc (s)':<14} {'Py Dec (s)':<14} {'Rs Dec (s)':<14}")
        print("-" * 62)

        for run in range(N_RUNS):
            # Python encode
            t_py_enc = benchmark_python_encode(array, ijk_to_ras, py_jvol)
            py_encode_times.append(t_py_enc)

            # Python decode
            t_py_dec = benchmark_python_decode(py_jvol)
            py_decode_times.append(t_py_dec)

            # Rust bench (core algorithm only, no file I/O overhead)
            t_rs_enc, t_rs_dec = benchmark_rust(NIFTI_PATH)
            rust_encode_times.append(t_rs_enc)
            rust_decode_times.append(t_rs_dec)

            print(
                f"{run+1:<6} "
                f"{t_py_enc:<14.4f} "
                f"{t_rs_enc:<14.4f} "
                f"{t_py_dec:<14.4f} "
                f"{t_rs_dec:<14.4f}"
            )

    print()
    print("=" * 70)
    print("Results (median of {} runs)".format(N_RUNS))
    print("=" * 70)

    py_enc_med = np.median(py_encode_times)
    rs_enc_med = np.median(rust_encode_times)
    py_dec_med = np.median(py_decode_times)
    rs_dec_med = np.median(rust_decode_times)

    print(f"{'':20} {'Python':>12} {'Rust':>12} {'Speedup':>12}")
    print("-" * 56)
    print(f"{'Encode (s)':20} {py_enc_med:12.4f} {rs_enc_med:12.4f} {py_enc_med/rs_enc_med:11.1f}x")
    print(f"{'Decode (s)':20} {py_dec_med:12.4f} {rs_dec_med:12.4f} {py_dec_med/rs_dec_med:11.1f}x")
    print()
    print("Note: Rust times measure core algorithm only (no NIfTI I/O).")
    print("      Python times include npz file I/O but not NIfTI I/O.")


if __name__ == "__main__":
    main()
