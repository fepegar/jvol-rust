use crate::types::BlockShape;

/// Compute the quality multiplier using exponential decay.
///
/// Maps quality (1–100) to a multiplier that scales the quantization table.
/// Lower quality → higher multiplier → more aggressive quantization.
pub fn get_multiplier(quality: u8) -> f64 {
    let max_multiplier: f64 = 100.0;
    let spread: f64 = 15.0;

    // Compute multipliers for all qualities 1..=100
    let mut multipliers = [0.0f64; 100];
    for i in 0..100 {
        let q = i as f64; // quality - 1 (0-indexed)
        multipliers[i] = 1.0 + (max_multiplier - 1.0) * (-q / spread).exp();
    }

    // Rescale to [1, max_multiplier]
    let min_val = multipliers
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_val = multipliers
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    for m in multipliers.iter_mut() {
        *m -= min_val;
        *m /= max_val - min_val;
        *m *= max_multiplier - 1.0;
        *m += 1.0;
    }

    multipliers[(quality - 1) as usize]
}

/// Generate a 3D quantization table for the given block shape and quality.
///
/// The table values increase with distance from the origin (DC component),
/// modulated by the quality setting.
pub fn get_quantization_table(shape: BlockShape, quality: u8) -> Vec<f32> {
    let [si, sj, sk] = shape;
    let total = si * sj * sk;
    let mut norms = Vec::with_capacity(total);

    // Compute L2 distance from origin for each index in the block
    for i in 0..si {
        for j in 0..sj {
            for k in 0..sk {
                let norm =
                    ((i * i + j * j + k * k) as f64).sqrt();
                norms.push(norm);
            }
        }
    }

    let max_norm = norms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_norm == 0.0 {
        return vec![1.0; total];
    }

    // Normalize to [0, 1]
    let qtm_unit: Vec<f64> = norms.iter().map(|n| n / max_norm).collect();

    // Scale to JPEG-like range (quality 50 baseline)
    let qtm_min_50: f64 = 10.0;
    let qtm_max_50: f64 = 120.0;
    let qtm_range_50 = qtm_max_50 - qtm_min_50;

    let multiplier = get_multiplier(quality);

    qtm_unit
        .iter()
        .map(|&u| {
            let qtm_50 = u * qtm_range_50 + qtm_min_50;
            let qtm = 2.0 * qtm_50;
            (qtm * multiplier) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiplier_range() {
        let m1 = get_multiplier(1);
        let m100 = get_multiplier(100);
        assert!(m1 > m100, "quality 1 should have higher multiplier than 100");
        assert!(m100 >= 1.0, "multiplier should be >= 1");
    }

    #[test]
    fn test_quantization_table_shape() {
        let table = get_quantization_table([8, 8, 8], 60);
        assert_eq!(table.len(), 512);
        // DC coefficient should be the smallest
        assert!(table[0] <= table[1]);
    }
}
