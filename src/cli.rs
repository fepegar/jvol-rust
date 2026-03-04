use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "jvol-rust", about = "Lightning-fast JPEG compression for 3D medical images")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Encode a NIfTI file into a compressed .jvol file
    Encode {
        /// Input NIfTI file path (.nii or .nii.gz)
        input: String,
        /// Output .jvol file path
        output: String,
        /// JPEG quality (1-100, higher = better quality)
        #[arg(short, long, default_value_t = 60)]
        quality: u8,
        /// Block size for 3D DCT
        #[arg(short, long, default_value_t = 8)]
        block_size: usize,
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    /// Decode a .jvol file back to NIfTI
    Decode {
        /// Input .jvol file path
        input: String,
        /// Output NIfTI file path (.nii or .nii.gz)
        output: String,
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },
    /// Benchmark encode+decode on a NIfTI file (times core algorithm only)
    Bench {
        /// Input NIfTI file path
        input: String,
        /// JPEG quality (1-100)
        #[arg(short, long, default_value_t = 60)]
        quality: u8,
        /// Block size for 3D DCT
        #[arg(short, long, default_value_t = 8)]
        block_size: usize,
    },
}
