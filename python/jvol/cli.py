"""CLI entry point for jvol."""

import argparse
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="jvol",
        description="Lightning-fast JPEG compression for 3D medical images",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # encode
    enc = subparsers.add_parser("encode", help="Encode a NIfTI file to .jvol")
    enc.add_argument("input", help="Input NIfTI file (.nii or .nii.gz)")
    enc.add_argument("output", help="Output .jvol file")
    enc.add_argument("-q", "--quality", type=int, default=60)
    enc.add_argument("-b", "--block-size", type=int, default=8)
    enc.add_argument("-v", "--verbose", action="store_true")

    # decode
    dec = subparsers.add_parser("decode", help="Decode a .jvol file to NIfTI")
    dec.add_argument("input", help="Input .jvol file")
    dec.add_argument("output", help="Output NIfTI file (.nii or .nii.gz)")
    dec.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    from jvol._jvol_rust import decode, encode

    if args.command == "encode":
        if not Path(args.input).exists():
            print(f"Error: input file '{args.input}' not found", file=sys.stderr)
            sys.exit(1)
        if args.verbose:
            print(f"Encoding '{args.input}' -> '{args.output}'", file=sys.stderr)
        start = time.perf_counter()
        encode(args.input, args.output, args.quality, args.block_size)
        elapsed = time.perf_counter() - start
        if args.verbose:
            print(f"Done in {elapsed:.3f}s", file=sys.stderr)

    elif args.command == "decode":
        if not Path(args.input).exists():
            print(f"Error: input file '{args.input}' not found", file=sys.stderr)
            sys.exit(1)
        if args.verbose:
            print(f"Decoding '{args.input}' -> '{args.output}'", file=sys.stderr)
        start = time.perf_counter()
        decode(args.input, args.output)
        elapsed = time.perf_counter() - start
        if args.verbose:
            print(f"Done in {elapsed:.3f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
