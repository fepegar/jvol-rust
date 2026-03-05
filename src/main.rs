use std::path::Path;
use std::time::Instant;

use clap::Parser;
use nifti::NiftiObject;

use jvol_rust::cli::{Cli, Commands};
use jvol_rust::decoding::decode_array;
use jvol_rust::encoding::encode_array;
use jvol_rust::io::{
    decode_jvol_to_nifti, dtype_from_nifti_code, encode_nifti_to_jvol, read_nifti,
};

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Encode {
            input,
            output,
            quality,
            lossless,
            verbose,
        } => {
            let quality = if lossless { 0 } else { quality };
            let input_path = Path::new(&input);
            let output_path = Path::new(&output);

            if !input_path.exists() {
                eprintln!("Error: input file '{}' not found", input);
                std::process::exit(1);
            }

            if verbose {
                eprintln!("Encoding '{}' -> '{}'", input, output);
                if quality == 0 {
                    eprintln!("Mode: lossless (LeGall 5/3 wavelet)");
                } else {
                    eprintln!("Mode: lossy, quality: {} (CDF 9/7 wavelet)", quality);
                }
            }

            let start = Instant::now();
            match encode_nifti_to_jvol(input_path, output_path, quality) {
                Ok(()) => {
                    let elapsed = start.elapsed();
                    if verbose {
                        let input_size =
                            std::fs::metadata(input_path).map(|m| m.len()).unwrap_or(0);
                        let output_size =
                            std::fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);
                        eprintln!(
                            "Done in {:.3}s ({} -> {} bytes, {:.1}x compression)",
                            elapsed.as_secs_f64(),
                            input_size,
                            output_size,
                            input_size as f64 / output_size as f64,
                        );
                    }
                    println!("{:.6}", elapsed.as_secs_f64());
                }
                Err(e) => {
                    eprintln!("Error encoding: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Decode {
            input,
            output,
            verbose,
        } => {
            let input_path = Path::new(&input);
            let output_path = Path::new(&output);

            if !input_path.exists() {
                eprintln!("Error: input file '{}' not found", input);
                std::process::exit(1);
            }

            if verbose {
                eprintln!("Decoding '{}' -> '{}'", input, output);
            }

            let start = Instant::now();
            match decode_jvol_to_nifti(input_path, output_path) {
                Ok(()) => {
                    let elapsed = start.elapsed();
                    if verbose {
                        eprintln!("Done in {:.3}s", elapsed.as_secs_f64());
                    }
                    println!("{:.6}", elapsed.as_secs_f64());
                }
                Err(e) => {
                    eprintln!("Error decoding: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Commands::Bench {
            input,
            quality,
            lossless,
        } => {
            let quality = if lossless { 0 } else { quality };
            let input_path = Path::new(&input);
            if !input_path.exists() {
                eprintln!("Error: input file '{}' not found", input);
                std::process::exit(1);
            }

            eprintln!("Loading NIfTI file...");
            let (channels, _affine) = read_nifti(input_path).unwrap_or_else(|e| {
                eprintln!("Error reading NIfTI: {}", e);
                std::process::exit(1);
            });
            let shape = channels[0].shape();
            eprintln!(
                "Array shape: [{}, {}, {}] ({} channel{})",
                shape[0],
                shape[1],
                shape[2],
                channels.len(),
                if channels.len() > 1 { "s" } else { "" }
            );
            if quality == 0 {
                eprintln!("Mode: lossless");
            } else {
                eprintln!("Mode: lossy, quality: {}", quality);
            }

            let nifti_obj = nifti::ReaderOptions::new().read_file(input_path).unwrap();
            let nifti_dtype = dtype_from_nifti_code(nifti_obj.header().datatype);

            // Time encode only (first channel)
            let enc_start = Instant::now();
            let result = encode_array(&channels[0].view(), quality, nifti_dtype);
            let enc_elapsed = enc_start.elapsed();

            let target_shape = [shape[0], shape[1], shape[2]];

            // Report encoded size
            let encoded_bytes: usize = result.subbands.iter().map(|s| s.data.len()).sum();
            let raw_bytes = shape[0] * shape[1] * shape[2] * 4; // i32 per voxel
            eprintln!(
                "Encoded: {} bytes (from {} raw, {:.1}x)",
                encoded_bytes,
                raw_bytes,
                raw_bytes as f64 / encoded_bytes as f64,
            );

            // Time decode only
            let dec_start = Instant::now();
            let _decoded = decode_array(
                &result.subbands,
                target_shape,
                result.wavelet,
                result.levels,
                result.step,
                result.intercept,
                result.slope,
                quality,
                nifti_dtype,
            );
            let dec_elapsed = dec_start.elapsed();

            println!(
                "{:.6} {:.6}",
                enc_elapsed.as_secs_f64(),
                dec_elapsed.as_secs_f64()
            );
        }
    }
}
