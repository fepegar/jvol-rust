pub mod cli;
pub mod decoding;
pub mod encoding;
pub mod io;
pub mod types;
pub mod wavelet;

#[cfg(feature = "python")]
mod python;
