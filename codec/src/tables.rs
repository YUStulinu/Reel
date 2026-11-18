//! Constant tables: the integer DCT basis, quantizer scales, scan orders and
//! the deblocking thresholds.

/// 64·√2·cos(π·m/64) for m = 0..=32, rounded and tuned for near-orthogonality
/// (the HEVC values); index 0 is used for the DC row only (64).
const COS64: [i32; 33] = [
    64, 90, 90, 90, 89, 88, 87, 85, 83, 82, 80, 78, 75, 73, 70, 67, 64, 61, 57, 54, 50, 46, 43, 38, 36, 31, 25, 22, 18, 13,
    9, 4, 0,
];

fn cos_at(a: usize) -> i32 {
    // the angle π·a/64, a in 0..128
    let a = a % 128;
    match a {
        0..=32 => COS64[a],
        33..=64 => -COS64[64 - a],
        65..=96 => -COS64[a - 64],
        _ => COS64[128 - a],
    }
}

/// the N×N integer DCT-II basis (rows are frequencies), N in {4, 8, 16}
pub fn dct_matrix(n: usize) -> Vec<i32> {
    let step = 32 / n;
    let mut m = vec![0; n * n];
    for k in 0..n {
        for j in 0..n {
            m[k * n + j] = if k == 0 { 64 } else { cos_at(k * (2 * j + 1) * step) };
        }
    }
    m
}

pub struct Bases {
    pub m4: Vec<i32>,
    pub m8: Vec<i32>,
    pub m16: Vec<i32>,
}

pub fn bases() -> &'static Bases {
    static B: std::sync::OnceLock<Bases> = std::sync::OnceLock::new();
    B.get_or_init(|| Bases { m4: dct_matrix(4), m8: dct_matrix(8), m16: dct_matrix(16) })
}

/// forward quantizer scale (Q14) and dequantizer scale (Q6) per QP mod 6:
/// Qstep doubles every 6 QP steps, Qstep(4) = 1
pub const QUANT: [i32; 6] = [26214, 23302, 20560, 18396, 16384, 14564];
pub const DEQUANT: [i32; 6] = [40, 45, 51, 57, 64, 72];
pub const MAX_QP: i32 = 51;

/// zig-zag scan of an n×n block: scan position -> raster index
pub fn zigzag(n: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(n * n);
    for s in 0..(2 * n - 1) {
        let range: Vec<usize> = (0..n).filter(|&i| s >= i && s - i < n).collect();
        if s % 2 == 0 {
            for &i in range.iter().rev() {
                out.push(((i) * n + (s - i)) as u16); // up-right: row decreasing
            }
        } else {
            for &i in range.iter() {
                out.push(((i) * n + (s - i)) as u16);
            }
        }
    }
    out
}

pub struct Scans {
    pub z4: Vec<u16>,
    pub z8: Vec<u16>,
    pub z16: Vec<u16>,
}

pub fn scans() -> &'static Scans {
    static S: std::sync::OnceLock<Scans> = std::sync::OnceLock::new();
    S.get_or_init(|| Scans { z4: zigzag(4), z8: zigzag(8), z16: zigzag(16) })
}

/// deblocking edge thresholds indexed by QP (alpha: edge step, beta: side flatness)
pub const ALPHA: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20, 22, 25, 28, 32, 36, 40, 45, 50,
    56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226, 255, 255,
];
pub const BETA: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
    13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];

/// the clipping bound of the normal deblocking filter for boundary strength 1..3
pub fn tc0(qp: i32, bs: u8) -> i32 {
    let b = BETA[qp.clamp(0, 51) as usize] as i32;
    match bs {
        1 => (b + 2) / 4,
        2 => (b + 1) / 3 + (b > 0) as i32,
        _ => (b + 1) / 2 + (b > 0) as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bases_are_nearly_orthogonal() {
        for n in [4, 8, 16] {
            let m = dct_matrix(n);
            for a in 0..n {
                for b in 0..n {
                    let dot: i64 = (0..n).map(|j| (m[a * n + j] * m[b * n + j]) as i64).sum();
                    let want = if a == b { 4096 * n as i64 } else { 0 };
                    assert!((dot - want).abs() <= 2 * n as i64 * 16, "n={n} rows {a},{b}: {dot}");
                }
            }
        }
    }

    #[test]
    fn zigzag_is_a_permutation() {
        for n in [4, 8, 16] {
            let mut z = zigzag(n);
            assert_eq!(&z[..3], &[0, 1, n as u16]);
            z.sort();
            assert!(z.iter().enumerate().all(|(i, &v)| i == v as usize));
        }
    }

    #[test]
    fn quant_and_dequant_scales_invert() {
        for i in 0..6 {
            let p = QUANT[i] as i64 * DEQUANT[i] as i64;
            assert!((p - (1 << 20)).abs() < 1 << 11, "{p}");
        }
    }
}
