use std::path::Path;
use std::time::Instant;

use clap::Parser;
use nifti::NiftiObject;

use jvol_rust::cli::{Cli, Commands};
use jvol_rust::decoding::decode_array;
use jvol_rust::encoding::encode_array;
use jvol_rust::io::{decode_jvol_to_nifti, encode_nifti_to_jvol, read_nifti, dtype_from_nifti_code};

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Encode {
            input,
            output,
            quality,
            block_size,
            verbose,
        } => {
            let input_path = Path::new(&input);
            let output_path = Path::new(&output);

            if !input_path.exists() {
                eprintln!("Error: input file '{}' not found", input);
                std::process::exit(1);
            }

            if verbose {
                eprintln!("Encoding '{}' -> '{}'", input, output);
                eprintln!("Quality: {}, Block size: {}", quality, block_size);
            }

            let start = Instant::now();
            match encode_nifti_to_jvol(input_path, output_path, block_size, quality) {
                Ok(()) => {
                    let elapsed = start.elapsed();
                    if verbose {
                        let input_size = std::fs::metadata(input_path)
                            .map(|m| m.len())
                            .unwrap_or(0);
                        let output_size = std::fs::metadata(output_path)
                            .map(|m| m.len())
                            .unwrap_or(0);
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
            block_size,
        } => {
            let input_path = Path::new(&input);
            if !input_path.exists() {
                eprintln!("Error: input file '{}' not found", input);
                std::process::exit(1);
            }

            // Read NIfTI (not timed)
            eprintln!("Loading NIfTI file...");
            let (array, _affine) = read_nifti(input_path).unwrap_or_else(|e| {
                eprintln!("Error reading NIfTI: {}", e);
                std::process::exit(1);
            });
            let shape = array.shape();
            eprintln!("Array shape: [{}, {}, {}]", shape[0], shape[1], shape[2]);

            let nifti_obj = nifti::ReaderOptions::new().read_file(input_path).unwrap();
            let nifti_dtype = dtype_from_nifti_code(nifti_obj.header().datatype);

            // Time encode only
            let enc_start = Instant::now();
            let result = encode_array(&array.view(), block_size, quality);
            let enc_elapsed = enc_start.elapsed();

            let block_shape = [block_size, block_size, block_size];
            let target_shape = [shape[0], shape[1], shape[2]];

            // Time decode only
            let dec_start = Instant::now();
            let _decoded = decode_array(
                &result.dc_rle,
                &result.ac_rle,
                &result.quantization_table,
                target_shape,
                block_shape,
                result.intercept,
                result.slope,
                nifti_dtype,
            );
            let dec_elapsed = dec_start.elapsed();

            // Output: encode_time decode_time
            println!("{:.6} {:.6}", enc_elapsed.as_secs_f64(), dec_elapsed.as_secs_f64());
        }
    }
}
