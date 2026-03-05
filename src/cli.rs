use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "jvol-rust",
    about = "Lightning-fast wavelet compression for 3D medical images"
)]
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
        /// Quality (1-100 lossy, or 0 for lossless)
        #[arg(short, long, default_value_t = 60)]
        quality: u8,
        /// Lossless mode (overrides quality to 0)
        #[arg(short, long)]
        lossless: bool,
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
        /// Quality (1-100 lossy, or 0 for lossless)
        #[arg(short, long, default_value_t = 60)]
        quality: u8,
        /// Lossless mode (overrides quality to 0)
        #[arg(short, long)]
        lossless: bool,
    },
}
