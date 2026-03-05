"""CLI entry point for jvol."""

import sys
import time
from pathlib import Path

import typer

app = typer.Typer(
    name="jvol",
    help="Lightning-fast wavelet compression for 3D medical images",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def main_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


@app.command()
def encode(
    input: Path = typer.Argument(help="Input NIfTI file (.nii or .nii.gz)"),
    output: Path = typer.Argument(help="Output .jvol file"),
    quality: int = typer.Option(60, "--quality", "-q", help="Quality (1-100 lossy, 0 for lossless)"),
    lossless: bool = typer.Option(False, "--lossless", "-l", help="Lossless mode (overrides quality)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Encode a NIfTI file to .jvol."""
    from jvol._jvol_rust import encode as _encode

    if not input.exists():
        typer.echo(f"Error: input file '{input}' not found", err=True)
        raise typer.Exit(1)
    if verbose:
        mode = "lossless" if lossless else f"lossy (quality={quality})"
        typer.echo(f"Encoding '{input}' -> '{output}' [{mode}]", err=True)
    start = time.perf_counter()
    _encode(str(input), str(output), quality, lossless)
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
