//! Intra prediction from the reconstructed neighbours of a block: DC,
//! vertical, horizontal and plane, for 4×4 to 16×16 blocks.
use crate::frame::Plane;

pub const DC: u8 = 0;
pub const VERT: u8 = 1;
pub const HORZ: u8 = 2;
pub const PLANE: u8 = 3;
pub const MODES: u8 = 4;

/// the mode actually used: modes whose neighbours are missing fall back to DC
pub fn effective(mode: u8, x: usize, y: usize) -> u8 {
    match mode {
        VERT if y == 0 => DC,
        HORZ if x == 0 => DC,
        PLANE if x == 0 || y == 0 => DC,
        m => m,
    }
}

/// predicts the n×n block at (x, y) of `p` into `out` (raster, n×n)
pub fn predict(p: &Plane, x: usize, y: usize, n: usize, mode: u8, out: &mut [u8]) {
    let has_top = y > 0;
    let has_left = x > 0;
    let top = |i: usize| p.at(x + i, y - 1);
    let left = |i: usize| p.at(x - 1, y + i);
    match effective(mode, x, y) {
        VERT => {
            for r in 0..n {
                for c in 0..n {
                    out[r * n + c] = top(c);
                }
            }
        }
        HORZ => {
            for r in 0..n {
                out[r * n..r * n + n].fill(left(r));
            }
        }
        PLANE => {
            let tl = p.at(x - 1, y - 1) as i32;
            let t = |i: isize| if i < 0 { tl } else { top(i as usize) as i32 };
            let l = |i: isize| if i < 0 { tl } else { left(i as usize) as i32 };
            let half = (n / 2) as isize;
            let mut hs = 0;
            let mut vs = 0;
            let mut sq = 0;
            for i in 1..=half {
                hs += i as i32 * (t(half - 1 + i) - t(half - 1 - i));
                vs += i as i32 * (l(half - 1 + i) - l(half - 1 - i));
                sq += (i * i) as i32;
            }
            let k = (1024 + sq / 2) / sq;
            let b = (k * hs + 32) >> 6;
            let c = (k * vs + 32) >> 6;
            let a = 16 * (l(n as isize - 1) + t(n as isize - 1));
            let o = half as i32 - 1;
            for r in 0..n {
                for cc in 0..n {
                    let v = (a + b * (cc as i32 - o) + c * (r as i32 - o) + 16) >> 5;
                    out[r * n + cc] = v.clamp(0, 255) as u8;
                }
            }
        }
        _ => {
            let mut sum = 0u32;
            let mut cnt = 0u32;
            if has_top {
                sum += (0..n).map(|i| top(i) as u32).sum::<u32>();
                cnt += n as u32;
            }
            if has_left {
                sum += (0..n).map(|i| left(i) as u32).sum::<u32>();
                cnt += n as u32;
            }
            let dc = if cnt == 0 { 128 } else { ((sum + cnt / 2) / cnt) as u8 };
            out[..n * n].fill(dc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_reproduces_a_gradient() {
        let mut p = Plane::new(32, 32, 8);
        for y in 0..32 {
            for x in 0..32 {
                p.set(x, y, (40 + 3 * x + 2 * y) as u8);
            }
        }
        for n in [4, 8, 16] {
            let mut out = vec![0; n * n];
            predict(&p, 8, 8, n, PLANE, &mut out);
            for r in 0..n {
                for c in 0..n {
                    let want = 40 + 3 * (8 + c) + 2 * (8 + r);
                    assert!((out[r * n + c] as i32 - want as i32).abs() <= 1, "n={n} ({r},{c}) {} vs {want}", out[r * n + c]);
                }
            }
        }
    }

    #[test]
    fn missing_neighbours_fall_back_to_dc() {
        let p = Plane::new(16, 16, 8);
        let mut out = vec![0; 16];
        predict(&p, 0, 0, 4, PLANE, &mut out);
        assert!(out.iter().all(|&v| v == 128));
    }
}
