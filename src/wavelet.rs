use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Wavelet types supported by jvol.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WaveletType {
    /// LeGall 5/3 — reversible integer wavelet (lossless).
    LeGall53,
    /// CDF 9/7 — irreversible wavelet (lossy).
    CDF97,
}

// CDF 9/7 lifting coefficients
const ALPHA: f64 = -1.586134342059924;
const BETA: f64 = -0.052980118572961;
const GAMMA: f64 = 0.882911075530934;
const DELTA: f64 = 0.443506852043971;
const K: f64 = 1.230174104914001;
const K_INV: f64 = 1.0 / K;

/// Forward 1D DWT in-place. After transform: data[0..half] = low, data[half..n] = high.
/// `temp` must be at least length `n`.
pub fn dwt1d_forward(data: &mut [f64], temp: &mut [f64], wavelet: WaveletType) {
    let n = data.len();
    if n < 2 {
        return;
    }
    let half = n.div_ceil(2);
    let nh = n / 2;

    // Deinterleave: even → low, odd → high
    for i in 0..half {
        temp[i] = data[2 * i];
    }
    for i in 0..nh {
        temp[half + i] = data[2 * i + 1];
    }

    let (low, rest) = temp[..n].split_at_mut(half);
    let high = &mut rest[..nh];

    match wavelet {
        WaveletType::LeGall53 => lifting_53_forward(low, high),
        WaveletType::CDF97 => lifting_97_forward(low, high),
    }

    data[..n].copy_from_slice(&temp[..n]);
}

/// Inverse 1D DWT in-place.
/// `temp` must be at least length `n`.
pub fn dwt1d_inverse(data: &mut [f64], temp: &mut [f64], wavelet: WaveletType) {
    let n = data.len();
    if n < 2 {
        return;
    }
    let half = n.div_ceil(2);
    let nh = n / 2;

    temp[..n].copy_from_slice(&data[..n]);
    let (low, rest) = temp[..n].split_at_mut(half);
    let high = &mut rest[..nh];

    match wavelet {
        WaveletType::LeGall53 => lifting_53_inverse(low, high),
        WaveletType::CDF97 => lifting_97_inverse(low, high),
    }

    // Interleave back
    for i in 0..half {
        data[2 * i] = low[i];
    }
    for i in 0..nh {
        data[2 * i + 1] = high[i];
    }
}

// --- LeGall 5/3 lifting ---

fn lifting_53_forward(low: &mut [f64], high: &mut [f64]) {
    let nl = low.len();
    let nh = high.len();
    if nh == 0 {
        return;
    }

    // Predict
    for n in 0..nh {
        let l_next = if n + 1 < nl { low[n + 1] } else { low[nl - 1] };
        high[n] -= ((low[n] + l_next) / 2.0).floor();
    }

    // Update
    for n in 0..nl {
        let h_prev = if n > 0 { high[n - 1] } else { high[0] };
        let h_curr = if n < nh { high[n] } else { high[nh - 1] };
        low[n] += ((h_prev + h_curr + 2.0) / 4.0).floor();
    }
}

fn lifting_53_inverse(low: &mut [f64], high: &mut [f64]) {
    let nl = low.len();
    let nh = high.len();
    if nh == 0 {
        return;
    }

    // Undo update
    for n in 0..nl {
        let h_prev = if n > 0 { high[n - 1] } else { high[0] };
        let h_curr = if n < nh { high[n] } else { high[nh - 1] };
        low[n] -= ((h_prev + h_curr + 2.0) / 4.0).floor();
    }

    // Undo predict
    for n in 0..nh {
        let l_next = if n + 1 < nl { low[n + 1] } else { low[nl - 1] };
        high[n] += ((low[n] + l_next) / 2.0).floor();
    }
}

// --- CDF 9/7 lifting ---

fn lifting_97_forward(low: &mut [f64], high: &mut [f64]) {
    let nl = low.len();
    let nh = high.len();
    if nh == 0 {
        return;
    }

    // Step 1
    for n in 0..nh {
        let l_next = if n + 1 < nl { low[n + 1] } else { low[nl - 1] };
        high[n] += ALPHA * (low[n] + l_next);
    }
    // Step 2
    for n in 0..nl {
        let h_prev = if n > 0 { high[n - 1] } else { high[0] };
        let h_curr = if n < nh { high[n] } else { high[nh - 1] };
        low[n] += BETA * (h_prev + h_curr);
    }
    // Step 3
    for n in 0..nh {
        let l_next = if n + 1 < nl { low[n + 1] } else { low[nl - 1] };
        high[n] += GAMMA * (low[n] + l_next);
    }
    // Step 4
    for n in 0..nl {
        let h_prev = if n > 0 { high[n - 1] } else { high[0] };
        let h_curr = if n < nh { high[n] } else { high[nh - 1] };
        low[n] += DELTA * (h_prev + h_curr);
    }
    // Scaling
    for v in low.iter_mut() {
        *v *= K;
    }
    for v in high.iter_mut() {
        *v *= K_INV;
    }
}

fn lifting_97_inverse(low: &mut [f64], high: &mut [f64]) {
    let nl = low.len();
    let nh = high.len();
    if nh == 0 {
        return;
    }

    // Undo scaling
    for v in low.iter_mut() {
        *v *= K_INV;
    }
    for v in high.iter_mut() {
        *v *= K;
    }
    // Undo step 4
    for n in 0..nl {
        let h_prev = if n > 0 { high[n - 1] } else { high[0] };
        let h_curr = if n < nh { high[n] } else { high[nh - 1] };
        low[n] -= DELTA * (h_prev + h_curr);
    }
    // Undo step 3
    for n in 0..nh {
        let l_next = if n + 1 < nl { low[n + 1] } else { low[nl - 1] };
        high[n] -= GAMMA * (low[n] + l_next);
    }
    // Undo step 2
    for n in 0..nl {
        let h_prev = if n > 0 { high[n - 1] } else { high[0] };
        let h_curr = if n < nh { high[n] } else { high[nh - 1] };
        low[n] -= BETA * (h_prev + h_curr);
    }
    // Undo step 1
    for n in 0..nh {
        let l_next = if n + 1 < nl { low[n + 1] } else { low[nl - 1] };
        high[n] -= ALPHA * (low[n] + l_next);
    }
}

// --- 3D DWT with rayon parallelism ---

/// Apply `transform_fn` along `axis` for the subregion `[0..extent[0], 0..extent[1], 0..extent[2]]`.
/// Parallelises over the outermost independent loop.
///
/// SAFETY: Each parallel iteration accesses disjoint memory regions partitioned by the outer loop index.
#[allow(clippy::needless_range_loop)]
fn par_transform_axis(
    ptr: *mut f64,
    extent: [usize; 3],
    strides: [usize; 3],
    axis: usize,
    wavelet: WaveletType,
    transform: fn(&mut [f64], &mut [f64], WaveletType),
) {
    // Convert to usize so closures can capture it (usize is Send+Sync)
    let base_ptr = ptr as usize;
    let axis_len = extent[axis];

    match axis {
        2 => {
            let [ni, nj, _] = extent;
            (0..ni).into_par_iter().for_each_init(
                || (vec![0.0; axis_len], vec![0.0; axis_len]),
                |(buf, temp), i| {
                    let p = base_ptr as *mut f64;
                    for j in 0..nj {
                        let base = i * strides[0] + j * strides[1];
                        for k in 0..axis_len {
                            buf[k] = unsafe { *p.add(base + k) };
                        }
                        transform(&mut buf[..axis_len], &mut temp[..axis_len], wavelet);
                        for k in 0..axis_len {
                            unsafe { *p.add(base + k) = buf[k] };
                        }
                    }
                },
            );
        }
        1 => {
            let [ni, _, nk] = extent;
            (0..ni).into_par_iter().for_each_init(
                || (vec![0.0; axis_len], vec![0.0; axis_len]),
                |(buf, temp), i| {
                    let p = base_ptr as *mut f64;
                    for k in 0..nk {
                        let base = i * strides[0] + k;
                        for j in 0..axis_len {
                            buf[j] = unsafe { *p.add(base + j * strides[1]) };
                        }
                        transform(&mut buf[..axis_len], &mut temp[..axis_len], wavelet);
                        for j in 0..axis_len {
                            unsafe { *p.add(base + j * strides[1]) = buf[j] };
                        }
                    }
                },
            );
        }
        0 => {
            let [_, nj, nk] = extent;
            (0..nj).into_par_iter().for_each_init(
                || (vec![0.0; axis_len], vec![0.0; axis_len]),
                |(buf, temp), j| {
                    let p = base_ptr as *mut f64;
                    for k in 0..nk {
                        let base = j * strides[1] + k;
                        for i in 0..axis_len {
                            buf[i] = unsafe { *p.add(base + i * strides[0]) };
                        }
                        transform(&mut buf[..axis_len], &mut temp[..axis_len], wavelet);
                        for i in 0..axis_len {
                            unsafe { *p.add(base + i * strides[0]) = buf[i] };
                        }
                    }
                },
            );
        }
        _ => unreachable!(),
    }
}

/// Forward multi-level 3D DWT in-place (parallelized with rayon).
pub fn dwt3d_forward(data: &mut ndarray::Array3<f64>, wavelet: WaveletType, levels: usize) {
    assert!(data.is_standard_layout());
    let total = [data.shape()[0], data.shape()[1], data.shape()[2]];
    let strides = [total[1] * total[2], total[2], 1];
    let ptr = data.as_mut_ptr();
    let mut extent = total;

    for _level in 0..levels {
        let [ni, nj, nk] = extent;
        // DWT along axis 2, then 1, then 0
        par_transform_axis(ptr, extent, strides, 2, wavelet, dwt1d_forward);
        par_transform_axis(ptr, extent, strides, 1, wavelet, dwt1d_forward);
        par_transform_axis(ptr, extent, strides, 0, wavelet, dwt1d_forward);
        extent = [ni.div_ceil(2), nj.div_ceil(2), nk.div_ceil(2)];
    }
}

/// Inverse multi-level 3D DWT in-place (parallelized with rayon).
pub fn dwt3d_inverse(data: &mut ndarray::Array3<f64>, wavelet: WaveletType, levels: usize) {
    assert!(data.is_standard_layout());
    let total = [data.shape()[0], data.shape()[1], data.shape()[2]];
    let strides = [total[1] * total[2], total[2], 1];
    let ptr = data.as_mut_ptr();

    // Compute extents for each level
    let mut extents = Vec::with_capacity(levels);
    let mut ext = total;
    for _ in 0..levels {
        extents.push(ext);
        ext = [ext[0].div_ceil(2), ext[1].div_ceil(2), ext[2].div_ceil(2)];
    }

    // Coarsest to finest
    for level in (0..levels).rev() {
        let extent = extents[level];
        // Inverse: axis 0, then 1, then 2 (reverse of forward)
        par_transform_axis(ptr, extent, strides, 0, wavelet, dwt1d_inverse);
        par_transform_axis(ptr, extent, strides, 1, wavelet, dwt1d_inverse);
        par_transform_axis(ptr, extent, strides, 2, wavelet, dwt1d_inverse);
    }
}

/// Maximum decomposition levels for a given shape (minimum dimension must be >= 2 at each level).
pub fn compute_max_levels(shape: [usize; 3]) -> usize {
    let min_dim = *shape.iter().min().unwrap();
    if min_dim < 2 {
        return 0;
    }
    // Each level halves dimensions; stop when any dimension would be < 2
    let mut levels = 0;
    let mut d = min_dim;
    while d >= 4 {
        d = d.div_ceil(2);
        levels += 1;
    }
    levels.min(6) // cap at 6 levels
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_53_roundtrip_1d() {
        let original = vec![1.0, 5.0, 3.0, 8.0, 2.0, 7.0, 4.0, 6.0];
        let mut data = original.clone();
        let mut temp = vec![0.0; 8];
        dwt1d_forward(&mut data, &mut temp, WaveletType::LeGall53);
        dwt1d_inverse(&mut data, &mut temp, WaveletType::LeGall53);
        for (a, b) in original.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-10, "5/3 roundtrip failed: {} != {}", a, b);
        }
    }

    #[test]
    fn test_97_roundtrip_1d() {
        let original = vec![1.0, 5.0, 3.0, 8.0, 2.0, 7.0, 4.0, 6.0];
        let mut data = original.clone();
        let mut temp = vec![0.0; 8];
        dwt1d_forward(&mut data, &mut temp, WaveletType::CDF97);
        dwt1d_inverse(&mut data, &mut temp, WaveletType::CDF97);
        for (a, b) in original.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-10, "9/7 roundtrip failed: {} != {}", a, b);
        }
    }

    #[test]
    fn test_53_roundtrip_3d() {
        let original = Array3::from_shape_fn((16, 16, 16), |(i, j, k)| {
            (i * 100 + j * 10 + k) as f64
        });
        let mut data = original.clone();
        dwt3d_forward(&mut data, WaveletType::LeGall53, 3);
        dwt3d_inverse(&mut data, WaveletType::LeGall53, 3);
        for (a, b) in original.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-10, "3D 5/3 roundtrip failed");
        }
    }

    #[test]
    fn test_97_roundtrip_3d() {
        let original = Array3::from_shape_fn((16, 16, 16), |(i, j, k)| {
            (i * 100 + j * 10 + k) as f64
        });
        let mut data = original.clone();
        dwt3d_forward(&mut data, WaveletType::CDF97, 3);
        dwt3d_inverse(&mut data, WaveletType::CDF97, 3);
        for (a, b) in original.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-8, "3D 9/7 roundtrip failed");
        }
    }

    #[test]
    fn test_odd_dimensions() {
        let original = Array3::from_shape_fn((15, 17, 13), |(i, j, k)| {
            (i * 100 + j * 10 + k) as f64
        });
        let mut data = original.clone();
        dwt3d_forward(&mut data, WaveletType::LeGall53, 2);
        dwt3d_inverse(&mut data, WaveletType::LeGall53, 2);
        for (a, b) in original.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-10, "Odd-dim 5/3 roundtrip failed");
        }
    }

    #[test]
    fn test_max_levels() {
        assert_eq!(compute_max_levels([256, 256, 256]), 6);
        assert_eq!(compute_max_levels([16, 16, 16]), 3);
        assert_eq!(compute_max_levels([3, 3, 3]), 0);
    }
}
