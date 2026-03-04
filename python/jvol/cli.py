"""CLI entry point for jvol."""

import sys
import time
from pathlib import Path

import typer

app = typer.Typer(
    name="jvol",
    help="Lightning-fast JPEG compression for 3D medical images",
    add_completion=False,
)


@app.command()
def encode(
    input: Path = typer.Argument(help="Input NIfTI file (.nii or .nii.gz)"),
    output: Path = typer.Argument(help="Output .jvol file"),
    quality: int = typer.Option(60, "--quality", "-q", help="JPEG quality (1-100)"),
    block_size: int = typer.Option(8, "--block-size", "-b", help="Block size for 3D DCT"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Encode a NIfTI file to .jvol."""
    from jvol._jvol_rust import encode as _encode

    if not input.exists():
        typer.echo(f"Error: input file '{input}' not found", err=True)
        raise typer.Exit(1)
    if verbose:
        typer.echo(f"Encoding '{input}' -> '{output}'", err=True)
    start = time.perf_counter()
    _encode(str(input), str(output), quality, block_size)
    elapsed = time.perf_counter() - start
    if verbose:
        typer.echo(f"Done in {elapsed:.3f}s", err=True)


@app.command()
def decode(
    input: Path = typer.Argument(help="Input .jvol file"),
    output: Path = typer.Argument(help="Output NIfTI file (.nii or .nii.gz)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Decode a .jvol file to NIfTI."""
    from jvol._jvol_rust import decode as _decode

    if not input.exists():
        typer.echo(f"Error: input file '{input}' not found", err=True)
        raise typer.Exit(1)
    if verbose:
        typer.echo(f"Decoding '{input}' -> '{output}'", err=True)
    start = time.perf_counter()
    _decode(str(input), str(output))
    elapsed = time.perf_counter() - start
    if verbose:
        typer.echo(f"Done in {elapsed:.3f}s", err=True)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
