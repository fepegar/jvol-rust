use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use ndarray::Array3;
use nifti::{IntoNdArray, NiftiObject, ReaderOptions};

use crate::decoding::decode_array;
use crate::encoding::encode_array;
use crate::types::*;

/// Read a NIfTI file and return the 3D array (as f64) and affine matrix.
pub fn read_nifti(path: &Path) -> Result<(Array3<f64>, Affine4x4), Box<dyn std::error::Error>> {
    let obj = ReaderOptions::new().read_file(path)?;
    let header = obj.header();

    // Build affine from srow_x, srow_y, srow_z (method 3)
    let affine = if header.sform_code > 0 {
        let sx = header.srow_x;
        let sy = header.srow_y;
        let sz = header.srow_z;
        [
            [sx[0] as f64, sx[1] as f64, sx[2] as f64, sx[3] as f64],
            [sy[0] as f64, sy[1] as f64, sy[2] as f64, sy[3] as f64],
            [sz[0] as f64, sz[1] as f64, sz[2] as f64, sz[3] as f64],
            [0.0, 0.0, 0.0, 1.0],
        ]
    } else if header.qform_code > 0 {
        // Use quaternion-based method (method 2)
        let b = header.quatern_b as f64;
        let c = header.quatern_c as f64;
        let d = header.quatern_d as f64;
        let a = (1.0 - b * b - c * c - d * d).max(0.0).sqrt();

        let r = [
            [
                a * a + b * b - c * c - d * d,
                2.0 * (b * c - a * d),
                2.0 * (b * d + a * c),
            ],
            [
                2.0 * (b * c + a * d),
                a * a + c * c - b * b - d * d,
                2.0 * (c * d - a * b),
            ],
            [
                2.0 * (b * d - a * c),
                2.0 * (c * d + a * b),
                a * a + d * d - b * b - c * c,
            ],
        ];

        let pixdim = &header.pixdim;
        let qfac = if pixdim[0] < 0.0 { -1.0 } else { 1.0 };

        [
            [
                r[0][0] * pixdim[1] as f64,
                r[0][1] * pixdim[2] as f64,
                r[0][2] * pixdim[3] as f64 * qfac,
                header.quatern_x as f64,
            ],
            [
                r[1][0] * pixdim[1] as f64,
                r[1][1] * pixdim[2] as f64,
                r[1][2] * pixdim[3] as f64 * qfac,
                header.quatern_y as f64,
            ],
            [
                r[2][0] * pixdim[1] as f64,
                r[2][1] * pixdim[2] as f64,
                r[2][2] * pixdim[3] as f64 * qfac,
                header.quatern_z as f64,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    } else {
        // Identity with pixdim scaling
        let pixdim = &header.pixdim;
        [
            [pixdim[1] as f64, 0.0, 0.0, 0.0],
            [0.0, pixdim[2] as f64, 0.0, 0.0],
            [0.0, 0.0, pixdim[3] as f64, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    };

    let volume = obj.into_volume();
    let ndarray_data = volume.into_ndarray::<f64>()?;

    let shape = ndarray_data.shape();
    if shape.len() < 3 {
        return Err("Expected at least 3 dimensions".into());
    }

    // The nifti crate returns Fortran-order (column-major) ndarray.
    // Convert to standard C-order layout to match our encoding pipeline.
    let array = ndarray_data
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| format!("Expected 3D volume: {}", e))?
        .as_standard_layout()
        .into_owned();

    Ok((array, affine))
}

/// Write a 3D array as a NIfTI file.
pub fn write_nifti(
    array: &Array3<f64>,
    affine: &Affine4x4,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use byteorder::{LittleEndian, WriteBytesExt};

    let shape = array.shape();
    let (ni, nj, nk) = (shape[0], shape[1], shape[2]);

    // Build a minimal NIfTI-1 header (348 bytes)
    let mut header = vec![0u8; 348];
    let mut cursor = std::io::Cursor::new(&mut header[..]);

    // sizeof_hdr
    cursor.write_i32::<LittleEndian>(348)?;

    // Skip to dim (offset 40)
    cursor.set_position(40);
    cursor.write_i16::<LittleEndian>(3)?; // ndim
    cursor.write_i16::<LittleEndian>(ni as i16)?;
    cursor.write_i16::<LittleEndian>(nj as i16)?;
    cursor.write_i16::<LittleEndian>(nk as i16)?;
    cursor.write_i16::<LittleEndian>(1)?; // dim[4]
    cursor.write_i16::<LittleEndian>(1)?; // dim[5]
    cursor.write_i16::<LittleEndian>(1)?; // dim[6]
    cursor.write_i16::<LittleEndian>(1)?; // dim[7]

    // Skip to datatype (offset 70)
    cursor.set_position(70);
    cursor.write_i16::<LittleEndian>(16)?; // DT_FLOAT32
    cursor.write_i16::<LittleEndian>(32)?; // bitpix

    // pixdim (offset 76) — column norms of the rotation/scaling submatrix
    cursor.set_position(76);
    cursor.write_f32::<LittleEndian>(1.0)?; // qfac
    #[allow(clippy::needless_range_loop)]
    for col in 0..3 {
        let pixdim_val = (0..3)
            .map(|row| affine[row][col] * affine[row][col])
            .sum::<f64>()
            .sqrt() as f32;
        cursor.write_f32::<LittleEndian>(pixdim_val)?;
    }

    // vox_offset (offset 108)
    cursor.set_position(108);
    cursor.write_f32::<LittleEndian>(352.0)?; // data starts at byte 352

    // sform_code (offset 254)
    cursor.set_position(254);
    cursor.write_i16::<LittleEndian>(1)?; // NIFTI_XFORM_SCANNER_ANAT

    // srow_x (offset 280), srow_y (offset 296), srow_z (offset 312)
    cursor.set_position(280);
    for row in &affine[..3] {
        for &val in row {
            cursor.write_f32::<LittleEndian>(val as f32)?;
        }
    }

    // magic (offset 344)
    cursor.set_position(344);
    cursor.write_all(b"n+1\0")?;

    // Write the file
    let is_gz = path
        .to_str()
        .map(|s| s.ends_with(".nii.gz"))
        .unwrap_or(false);

    // NIfTI stores data in Fortran order (first dimension varies fastest)
    let write_data = |writer: &mut dyn Write| -> Result<(), Box<dyn std::error::Error>> {
        writer.write_all(&header)?;
        writer.write_all(&[0u8; 4])?; // 4-byte extension pad
        for k in 0..nk {
            for j in 0..nj {
                for i in 0..ni {
                    writer.write_f32::<LittleEndian>(array[[i, j, k]] as f32)?;
                }
            }
        }
        Ok(())
    };

    if is_gz {
        let file = File::create(path)?;
        let mut gz = GzEncoder::new(BufWriter::new(file), Compression::default());
        write_data(&mut gz)?;
        gz.finish()?;
    } else {
        let mut file = BufWriter::new(File::create(path)?);
        write_data(&mut file)?;
    }

    Ok(())
}

/// Determine the JvolDtype from the NIfTI datatype code.
pub fn dtype_from_nifti_code(datatype: i16) -> JvolDtype {
    match datatype {
        2 => JvolDtype::U8,
        4 => JvolDtype::I16,
        8 => JvolDtype::I32,
        16 => JvolDtype::F32,
        64 => JvolDtype::F64,
        512 => JvolDtype::U16,
        _ => JvolDtype::F64, // fallback
    }
}

/// Save an encoded volume to a .jvol file (custom binary format).
pub fn save_jvol(encoded: &EncodedVolume, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let serialized = bincode::serialize(encoded)?;
    let file = File::create(path)?;
    let mut gz = GzEncoder::new(BufWriter::new(file), Compression::default());
    gz.write_all(&serialized)?;
    gz.finish()?;
    Ok(())
}

/// Load an encoded volume from a .jvol file.
pub fn open_jvol(path: &Path) -> Result<EncodedVolume, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut decoder = GzDecoder::new(BufReader::new(file));
    let mut buf = Vec::new();
    decoder.read_to_end(&mut buf)?;
    let encoded: EncodedVolume = bincode::deserialize(&buf)?;
    Ok(encoded)
}

/// Encode a NIfTI file and save as .jvol.
pub fn encode_nifti_to_jvol(
    input_path: &Path,
    output_path: &Path,
    block_size: usize,
    quality: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    let (array, affine) = read_nifti(input_path)?;
    let shape = [array.shape()[0], array.shape()[1], array.shape()[2]];

    let header = nifti::ReaderOptions::new().read_file(input_path)?;
    let nifti_dtype = dtype_from_nifti_code(header.header().datatype);

    let result = encode_array(&array.view(), block_size, quality);

    let encoded = EncodedVolume {
        metadata: JvolMetadata {
            shape,
            ijk_to_ras: affine,
            dtype: nifti_dtype,
            intercept: result.intercept,
            slope: result.slope,
            block_shape: [block_size, block_size, block_size],
            quality,
        },
        quantization_table: result.quantization_table,
        dc_rle: result.dc_rle,
        ac_rle: result.ac_rle,
    };

    save_jvol(&encoded, output_path)?;
    Ok(())
}

/// Decode a .jvol file and save as NIfTI.
pub fn decode_jvol_to_nifti(
    input_path: &Path,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let encoded = open_jvol(input_path)?;
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

    write_nifti(&array, &meta.ijk_to_ras, output_path)?;
    Ok(())
}
