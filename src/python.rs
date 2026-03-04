use std::path::Path;

use numpy::ndarray::Ix2;
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::decoding::decode_array;
use crate::encoding::encode_array;
use crate::io::read_nifti;
use crate::types::*;

/// Encode a NIfTI file to a .jvol file.
#[pyfunction]
#[pyo3(signature = (input, output, quality=60, block_size=8))]
fn encode(input: &str, output: &str, quality: u8, block_size: usize) -> PyResult<()> {
    crate::io::encode_nifti_to_jvol(Path::new(input), Path::new(output), block_size, quality)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Decode a .jvol file to a NIfTI file.
#[pyfunction]
fn decode(input: &str, output: &str) -> PyResult<()> {
    crate::io::decode_jvol_to_nifti(Path::new(input), Path::new(output))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Read a NIfTI file and return (array, affine) as numpy arrays.
#[pyfunction]
fn read_nifti_array(
    py: Python<'_>,
    path: &str,
) -> PyResult<(Py<PyArray3<f64>>, Py<PyArray2<f64>>)> {
    let (array, affine) = read_nifti(Path::new(path))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let affine_nd =
        ndarray::Array2::from_shape_vec((4, 4), affine.iter().flatten().copied().collect())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        array.into_pyarray(py).into(),
        affine_nd.into_pyarray(py).into(),
    ))
}

/// Encode a numpy array to .jvol bytes.
#[pyfunction]
#[pyo3(signature = (array, affine, quality=60, block_size=8))]
fn encode_array_to_bytes(
    py: Python<'_>,
    array: PyReadonlyArray3<f64>,
    affine: numpy::PyReadonlyArray2<f64>,
    quality: u8,
    block_size: usize,
) -> PyResult<Py<PyBytes>> {
    let arr = array.as_array();
    let aff = affine.as_array();

    let shape = [arr.shape()[0], arr.shape()[1], arr.shape()[2]];
    let result = encode_array(&arr, block_size, quality);

    let mut affine_4x4: Affine4x4 = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            affine_4x4[i][j] = aff[[i, j]];
        }
    }

    let encoded = EncodedVolume {
        metadata: JvolMetadata {
            shape,
            ijk_to_ras: affine_4x4,
            dtype: JvolDtype::F64,
            intercept: result.intercept,
            slope: result.slope,
            block_shape: [block_size, block_size, block_size],
            quality,
        },
        quantization_table: result.quantization_table,
        dc_rle: result.dc_rle,
        ac_rle: result.ac_rle,
    };

    let serialized = bincode::serialize(&encoded)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    let mut gz = GzEncoder::new(Vec::new(), Compression::default());
    gz.write_all(&serialized)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let compressed = gz
        .finish()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyBytes::new(py, &compressed).into())
}

/// Decode .jvol bytes to (array, affine) numpy arrays.
#[pyfunction]
fn decode_bytes_to_array(
    py: Python<'_>,
    data: &[u8],
) -> PyResult<(Py<PyArray3<f64>>, Py<PyArray2<f64>>)> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let mut decoder = GzDecoder::new(data);
    let mut buf = Vec::new();
    decoder
        .read_to_end(&mut buf)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let encoded: EncodedVolume = bincode::deserialize(&buf)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let meta = &encoded.metadata;
    let array = decode_array(
        &encoded.dc_rle,
        &encoded.ac_rle,
        &encoded.quantization_table,
        meta.shape,
        meta.block_shape,
        meta.intercept,
        meta.slope,
        meta.dtype,
    );

    let affine_nd = ndarray::Array2::from_shape_vec(
        (4, 4),
        meta.ijk_to_ras.iter().flatten().copied().collect(),
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        array.into_pyarray(py).into(),
        affine_nd.into_pyarray(py).into(),
    ))
}

/// Python module definition.
#[pymodule]
fn _jvol_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(read_nifti_array, m)?)?;
    m.add_function(wrap_pyfunction!(encode_array_to_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(decode_bytes_to_array, m)?)?;
    Ok(())
}
