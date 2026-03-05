use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use flate2::write::GzEncoder;
use flate2::Compression;
use ndarray::Array3;
use nifti::{IntoNdArray, NiftiObject, ReaderOptions};

use crate::decoding::decode_array;
use crate::encoding::encode_array;
use crate::types::*;

/// Read a NIfTI file and return one or more 3D channels (as f64) and affine matrix.
/// 3D files return a single channel. 4D files return one channel per volume.
#[allow(clippy::type_complexity)]
pub fn read_nifti(
    path: &Path,
) -> Result<(Vec<Array3<f64>>, Affine4x4), Box<dyn std::error::Error>> {
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

    let ndim = ndarray_data.ndim();
    if ndim < 3 {
        return Err("Expected at least 3 dimensions".into());
    }

    let channels = if ndim == 3 {
        let array = ndarray_data
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| format!("Expected 3D volume: {}", e))?
            .as_standard_layout()
            .into_owned();
        vec![array]
    } else {
        let shape = ndarray_data.shape().to_vec();
        let nc = shape[3];
        let standard = ndarray_data.as_standard_layout().into_owned();
        let ni = shape[0];
        let nj = shape[1];
        let nk = shape[2];
        (0..nc)
            .map(|c| {
                Array3::from_shape_fn((ni, nj, nk), |(i, j, k)| standard[[i, j, k, c].as_ref()])
            })
            .collect()
    };

    Ok((channels, affine))
}

/// Write one or more 3D channels as a NIfTI file.
pub fn write_nifti(
    channels: &[Array3<f64>],
    affine: &Affine4x4,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use byteorder::{LittleEndian, WriteBytesExt};

    if channels.is_empty() {
        return Err("No channels to write".into());
    }

    let shape = channels[0].shape();
    let (ni, nj, nk) = (shape[0], shape[1], shape[2]);
    let nc = channels.len();
    let ndim = if nc > 1 { 4 } else { 3 };

    let mut header = vec![0u8; 348];
    let mut cursor = std::io::Cursor::new(&mut header[..]);

    cursor.write_i32::<LittleEndian>(348)?;

    cursor.set_position(40);
    cursor.write_i16::<LittleEndian>(ndim as i16)?;
    cursor.write_i16::<LittleEndian>(ni as i16)?;
    cursor.write_i16::<LittleEndian>(nj as i16)?;
    cursor.write_i16::<LittleEndian>(nk as i16)?;
    cursor.write_i16::<LittleEndian>(nc as i16)?;
    cursor.write_i16::<LittleEndian>(1)?;
    cursor.write_i16::<LittleEndian>(1)?;
    cursor.write_i16::<LittleEndian>(1)?;

    cursor.set_position(70);
    cursor.write_i16::<LittleEndian>(16)?; // DT_FLOAT32
    cursor.write_i16::<LittleEndian>(32)?; // bitpix

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

    cursor.set_position(108);
    cursor.write_f32::<LittleEndian>(352.0)?;

    cursor.set_position(254);
    cursor.write_i16::<LittleEndian>(1)?;

    cursor.set_position(280);
    for row in &affine[..3] {
        for &val in row {
            cursor.write_f32::<LittleEndian>(val as f32)?;
        }
    }

    cursor.set_position(344);
    cursor.write_all(b"n+1\0")?;

    // NIfTI stores data in Fortran order
    let total_voxels = ni * nj * nk * nc;
    let mut data_buf: Vec<u8> = Vec::with_capacity(total_voxels * 4);
    for ch in &channels[..nc] {
        for k in 0..nk {
            for j in 0..nj {
                for i in 0..ni {
                    data_buf.extend_from_slice(&(ch[[i, j, k]] as f32).to_le_bytes());
                }
            }
        }
    }

    let write_all = |writer: &mut dyn Write| -> Result<(), Box<dyn std::error::Error>> {
        writer.write_all(&header)?;
        writer.write_all(&[0u8; 4])?;
        writer.write_all(&data_buf)?;
        Ok(())
    };

    let is_gz = path
        .to_str()
        .map(|s| s.ends_with(".nii.gz"))
        .unwrap_or(false);

    if is_gz {
        let file = File::create(path)?;
        let mut gz = GzEncoder::new(BufWriter::new(file), Compression::default());
        write_all(&mut gz)?;
        gz.finish()?;
    } else {
        let mut file = BufWriter::new(File::create(path)?);
        write_all(&mut file)?;
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
        _ => JvolDtype::F64,
    }
}

/// Save an encoded volume to a .jvol file (bincode + zstd).
pub fn save_jvol(encoded: &EncodedVolume, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let serialized = bincode::serialize(encoded)?;
    let file = File::create(path)?;
    // Level 6 balances compression ratio and speed well for our data
    let mut encoder = zstd::Encoder::new(BufWriter::new(file), 6)?;
    encoder.write_all(&serialized)?;
    encoder.finish()?;
    Ok(())
}

/// Load an encoded volume from a .jvol file (bincode + zstd).
pub fn open_jvol(path: &Path) -> Result<EncodedVolume, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut decoder = zstd::Decoder::new(BufReader::new(file))?;
    let mut buf = Vec::new();
    decoder.read_to_end(&mut buf)?;
    let encoded: EncodedVolume = bincode::deserialize(&buf)?;
    Ok(encoded)
}

/// Encode a NIfTI file and save as .jvol.
pub fn encode_nifti_to_jvol(
    input_path: &Path,
    output_path: &Path,
    quality: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    let (channels, affine) = read_nifti(input_path)?;
    let shape = [
        channels[0].shape()[0],
        channels[0].shape()[1],
        channels[0].shape()[2],
    ];

    let header = nifti::ReaderOptions::new().read_file(input_path)?;
    let nifti_dtype = dtype_from_nifti_code(header.header().datatype);

    let mut encoded_channels = Vec::with_capacity(channels.len());
    let mut wavelet = WaveletType::LeGall53;
    let mut levels = 0;

    for ch in &channels {
        let result = encode_array(&ch.view(), quality, nifti_dtype);
        wavelet = result.wavelet;
        levels = result.levels;
        encoded_channels.push(EncodedChannel {
            subbands: result.subbands,
            intercept: result.intercept,
            slope: result.slope,
            step: result.step,
        });
    }

    let encoded = EncodedVolume {
        metadata: JvolMetadata {
            shape,
            num_channels: channels.len(),
            ijk_to_ras: affine,
            dtype: nifti_dtype,
            wavelet,
            levels,
            quality,
        },
        channels: encoded_channels,
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

    let mut channels = Vec::with_capacity(encoded.channels.len());
    for ch in &encoded.channels {
        let array = decode_array(
            &ch.subbands,
            meta.shape,
            meta.wavelet,
            meta.levels,
            ch.step,
            ch.intercept,
            ch.slope,
            meta.quality,
            meta.dtype,
        );
        channels.push(array);
    }

    write_nifti(&channels, &meta.ijk_to_ras, output_path)?;
    Ok(())
}
