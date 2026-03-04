use serde::{Deserialize, Serialize};

/// Affine transformation matrix (4x4) mapping voxel indices to RAS+ coordinates.
pub type Affine4x4 = [[f64; 4]; 4];

/// Shape of the 3D volume.
pub type VolumeShape = [usize; 3];

/// Shape of a 3D block.
pub type BlockShape = [usize; 3];

/// Metadata stored alongside the compressed data in a .jvol file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JvolMetadata {
    pub shape: VolumeShape,
    pub ijk_to_ras: Affine4x4,
    pub dtype: JvolDtype,
    pub intercept: f64,
    pub slope: f64,
    pub block_shape: BlockShape,
    pub quality: u8,
}

/// Supported data types for volume arrays.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum JvolDtype {
    U8,
    U16,
    I16,
    I32,
    F32,
    F64,
}

impl JvolDtype {
    /// Get the min/max range for integer dtypes.
    pub fn iinfo(&self) -> Option<(f64, f64)> {
        match self {
            JvolDtype::U8 => Some((0.0, 255.0)),
            JvolDtype::U16 => Some((0.0, 65535.0)),
            JvolDtype::I16 => Some((-32768.0, 32767.0)),
            JvolDtype::I32 => Some((-2147483648.0, 2147483647.0)),
            JvolDtype::F32 | JvolDtype::F64 => None,
        }
    }
}

/// Run-length encoded data: parallel arrays of values and counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RleData {
    pub values: Vec<i32>,
    pub counts: Vec<u32>,
}

/// The full encoded representation of a volume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedVolume {
    pub metadata: JvolMetadata,
    pub quantization_table: Vec<f32>,
    pub dc_rle: RleData,
    pub ac_rle: RleData,
}
