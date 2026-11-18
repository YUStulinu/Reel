//! The integer DCT (4×4, 8×8, 16×16) and the quantizer. Everything is exact
//! integer arithmetic, so the encoder, the decoder, the C/AVX2 version and the
//! WebAssembly build reconstruct the same pixels.
use crate::tables::{bases, DEQUANT, QUANT};

fn basis(n: usize) -> &'static [i32] {
    let b = bases();
    match n {
        4 => &b.m4,
        8 => &b.m8,
        16 => &b.m16,
        _ => unreachable!("transform size {n}"),
    }
}

fn log2(n: usize) -> u32 {
    n.trailing_zeros()
}

/// residual (n×n, raster) -> coefficients
pub fn forward_scalar(n: usize, x: &[i32], y: &mut [i32]) {
    let m = basis(n);
    let s1 = log2(n) - 1;
    let s2 = log2(n) + 6;
    let mut t = [0i32; 256];
    // columns: t = M · X
    for k in 0..n {
        for c in 0..n {
            let mut acc = 0i32;
            for j in 0..n {
                acc += m[k * n + j] * x[j * n + c];
            }
            t[k * n + c] = (acc + (1 << (s1 - 1))) >> s1;
        }
    }
    // rows: Y = t · Mᵀ
    for r in 0..n {
        for k in 0..n {
            let mut acc = 0i64;
            for j in 0..n {
                acc += (t[r * n + j] * m[k * n + j]) as i64;
            }
            y[r * n + k] = ((acc + (1 << (s2 - 1))) >> s2) as i32;
        }
    }
}

/// coefficients -> residual
pub fn inverse_scalar(n: usize, y: &[i32], x: &mut [i32]) {
    let m = basis(n);
    let mut t = [0i32; 256];
    // columns: t = Mᵀ · Y
    for j in 0..n {
        for c in 0..n {
            let mut acc = 0i32;
            for k in 0..n {
                acc += m[k * n + j] * y[k * n + c];
            }
            t[j * n + c] = ((acc + 64) >> 7).clamp(-32768, 32767);
        }
    }
    // rows: X = t · M
    for r in 0..n {
        for j in 0..n {
            let mut acc = 0i32;
            for k in 0..n {
                acc += t[r * n + k] * m[k * n + j];
            }
            x[r * n + j] = (acc + 2048) >> 12;
        }
    }
}

/// the forward transform, through AVX2 when available
pub fn forward(n: usize, x: &[i32], y: &mut [i32]) {
    if crate::simd::forward(n, x, y) {
        return;
    }
    forward_scalar(n, x, y);
}

pub fn inverse(n: usize, y: &[i32], x: &mut [i32]) {
    if crate::simd::inverse(n, y, x) {
        return;
    }
    inverse_scalar(n, y, x);
}

/// quantizes in place; `deadzone` is the rounding offset in 1/64ths of a step
/// (32 = round to nearest). Returns the number of nonzero levels.
pub fn quantize(n: usize, qp: i32, deadzone: i32, c: &mut [i32]) -> usize {
    let shift = 14 + qp / 6 + 7 - log2(n) as i32;
    let q = QUANT[(qp % 6) as usize] as i64;
    let off = (deadzone as i64) << (shift - 6);
    let mut nz = 0;
    for v in c[..n * n].iter_mut() {
        let a = (((v.unsigned_abs() as i64 * q + off) >> shift) as i32).min(32767);
        *v = if *v < 0 { -a } else { a };
        nz += (a != 0) as usize;
    }
    nz
}

/// the reconstruction value of a level
#[inline]
pub fn dequant_one(n: usize, qp: i32, level: i32) -> i32 {
    let s = log2(n) as i32 - 1;
    let v = ((level as i64 * DEQUANT[(qp % 6) as usize] as i64) << (qp / 6)) + ((1i64 << s) >> 1);
    (v >> s).clamp(-32768, 32767) as i32
}

pub fn dequantize(n: usize, qp: i32, levels: &[i32], out: &mut [i32]) {
    for i in 0..n * n {
        out[i] = if levels[i] == 0 { 0 } else { dequant_one(n, qp, levels[i]) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> i32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed % 511) as i32 - 255
    }

    #[test]
    fn transform_round_trip_is_near_lossless() {
        let mut s = 0x1234_5678_9abc_def1u64;
        for n in [4, 8, 16] {
            for _ in 0..50 {
                let x: Vec<i32> = (0..n * n).map(|_| rng(&mut s)).collect();
                let mut y = vec![0; n * n];
                let mut back = vec![0; n * n];
                forward_scalar(n, &x, &mut y);
                // the forward output carries a 128/n gain; undo it the way dequantization would at QP 4
                let lv: Vec<i32> = y.iter().map(|&v| v).collect();
                let mut dq = vec![0; n * n];
                for i in 0..n * n {
                    // QP 4: Qstep 1, so the level equals the coefficient divided by the gain
                    dq[i] = lv[i];
                }
                inverse_scalar(n, &dq, &mut back);
                let err = x.iter().zip(&back).map(|(a, b)| (a - b).abs()).max().unwrap();
                assert!(err <= if n == 16 { 3 } else { 2 }, "n={n} max error {err}");
            }
        }
    }

    #[test]
    fn quantization_at_qp4_keeps_values() {
        // a flat block of 10: DC only
        for n in [4, 8, 16] {
            let x = vec![10; n * n];
            let mut y = vec![0; n * n];
            forward_scalar(n, &x, &mut y);
            quantize(n, 4, 32, &mut y);
            assert_eq!(y[0], 10 * n as i32, "n={n}: the DC level of a flat block is mean·n");
            let mut d = vec![0; n * n];
            let mut back = vec![0; n * n];
            dequantize(n, 4, &y, &mut d);
            inverse_scalar(n, &d, &mut back);
            assert!(back.iter().all(|&v| v == 10), "n={n}: {:?}", &back[..4]);
        }
    }
}
