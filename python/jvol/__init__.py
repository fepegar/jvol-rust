"""jvol: Lightning-fast JPEG-like compression for 3D medical images."""

from jvol._jvol_rust import (
    decode,
    decode_bytes_to_array,
    encode,
    encode_array_to_bytes,
    read_nifti_array,
)

__all__ = [
    "decode",
    "decode_bytes_to_array",
    "encode",
    "encode_array_to_bytes",
    "read_nifti_array",
]
