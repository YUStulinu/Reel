//! Motion estimation: SAD/SATD costs, a hexagon search from predicted
//! candidates, an exhaustive search (the reference for the CUDA kernel) and
//! half/quarter-pel refinement.
use crate::frame::Plane;
use crate::mc;

/// sum of absolute differences of a w×h block of `a` at (ax, ay) and `b` at (bx, by)
#[inline]
pub fn sad(a: &Plane, ax: usize, ay: usize, b: &Plane, bx: isize, by: isize, w: usize, h: usize) -> u32 {
    if let Some(s) = crate::simd::sad(a, ax, ay, b, bx, by, w, h) {
        return s;
    }
    sad_scalar(a, ax, ay, b, bx, by, w, h)
}

pub fn sad_scalar(a: &Plane, ax: usize, ay: usize, b: &Plane, bx: isize, by: isize, w: usize, h: usize) -> u32 {
    let mut s = 0u32;
    for r in 0..h {
        let ia = a.idx(ax as isize, (ay + r) as isize);
        let ib = b.idx(bx, by + r as isize);
        for c in 0..w {
            s += (a.data[ia + c] as i32 - b.data[ib + c] as i32).unsigned_abs();
        }
    }
    s
}

/// the sum of absolute 8×8 Hadamard-transformed differences, divided by 4
pub fn satd8(diff: &[i32; 64]) -> u32 {
    let mut t = *diff;
    for r in 0..8 {
        hadamard8(&mut t[r * 8..r * 8 + 8]);
    }
    let mut s = 0u32;
    for c in 0..8 {
        let mut col = [0i32; 8];
        for r in 0..8 {
            col[r] = t[r * 8 + c];
        }
        hadamard8(&mut col);
        s += col.iter().map(|v| v.unsigned_abs()).sum::<u32>();
    }
    (s + 2) >> 2
}

fn hadamard8(v: &mut [i32]) {
    let mut h = 1;
    while h < 8 {
        for i in (0..8).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

/// SATD of a w×h source block against a prediction (w, h multiples of 8)
pub fn satd(src: &Plane, x: usize, y: usize, pred: &[u8], w: usize, h: usize) -> u32 {
    let mut s = 0;
    for by in (0..h).step_by(8) {
        for bx in (0..w).step_by(8) {
            let mut d = [0i32; 64];
            for r in 0..8 {
                let i = src.idx((x + bx) as isize, (y + by + r) as isize);
                for c in 0..8 {
                    d[r * 8 + c] = src.data[i + c] as i32 - pred[(by + r) * w + bx + c] as i32;
                }
            }
            s += crate::simd::satd8(&d).unwrap_or_else(|| satd8(&d));
        }
    }
    s
}

/// approximate bits of a motion vector difference (quarter pel), in 1/16 bit
pub fn mv_bits16(d: [i32; 2]) -> u32 {
    let one = |v: i32| -> u32 {
        if v == 0 {
            16
        } else {
            let a = v.unsigned_abs();
            16 * (2 + 2 * (32 - a.leading_zeros()))
        }
    };
    one(d[0]) + one(d[1])
}

/// integer-pel bounds that keep a block (and its interpolation taps) inside the padded reference
pub struct Bounds {
    pub min: [i32; 2],
    pub max: [i32; 2],
}

impl Bounds {
    pub fn new(p: &Plane, x: usize, y: usize, w: usize, h: usize) -> Self {
        let m = p.pad as i32 - 8;
        Bounds { min: [-m - x as i32, -m - y as i32], max: [p.w as i32 + m - (x + w) as i32, p.h as i32 + m - (y + h) as i32] }
    }
    pub fn clamp(&self, v: [i32; 2]) -> [i32; 2] {
        [v[0].clamp(self.min[0], self.max[0]), v[1].clamp(self.min[1], self.max[1])]
    }
}

/// the half-pel samples of a reference plane, computed once per reference
/// with the motion-compensation filter: `h[x, y]` lies between pixels x and
/// x+1, `v` between rows, `hv` in the middle of four pixels
pub struct HalfPlanes {
    pub h: Plane,
    pub v: Plane,
    pub hv: Plane,
}

impl HalfPlanes {
    pub fn new(p: &Plane) -> Self {
        const F: [i32; 8] = [-1, 4, -11, 40, 40, -11, 4, -1];
        let (mut h, mut v, mut hv) = (p.clone(), p.clone(), p.clone());
        let m = p.pad as isize - 8;
        let (x0, x1, y0, y1) = (-m, p.w as isize + m, -m, p.h as isize + m);
        let s = p.stride as isize;
        let at = |x: isize, y: isize| p.data[p.idx(x, y)] as i32;
        let wide = (x1 - x0) as usize;
        let mut tmp = vec![0i32; wide * (y1 - y0 + 8) as usize];
        // horizontal half samples, unshifted, for rows y0-3 .. y1+4 (the 2-D case needs them)
        for (r, y) in (y0 - 3..y1 + 5).enumerate() {
            for (c, x) in (x0..x1).enumerate() {
                let i = p.idx(x - 3, y);
                tmp[r * wide + c] = (0..8).map(|k| F[k] * p.data[i + k] as i32).sum();
            }
        }
        for (r, y) in (y0..y1).enumerate() {
            for (c, x) in (x0..x1).enumerate() {
                let i = p.idx(x, y);
                let th = tmp[(r + 3) * wide + c];
                h.data[i] = ((th + 32) >> 6).clamp(0, 255) as u8;
                let sv: i32 = (0..8).map(|k| F[k] * at(x, y - 3 + k as isize)).sum();
                v.data[i] = ((sv + 32) >> 6).clamp(0, 255) as u8;
                let shv: i32 = (0..8).map(|k| F[k] * tmp[(r + k) * wide + c]).sum();
                hv.data[i] = (((shv >> 6) + 32) >> 6).clamp(0, 255) as u8;
            }
        }
        let _ = s;
        HalfPlanes { h, v, hv }
    }

    /// a sample at quarter-pel position (qx, qy) (absolute, in quarter pixels):
    /// exact at full and half positions, the average of the two nearest
    /// half-grid samples at quarter positions (an approximation for searching)
    #[inline]
    fn half_grid(&self, full: &Plane, hx: isize, hy: isize) -> i32 {
        // (hx, hy) in half pixels
        let (x, y) = (hx >> 1, hy >> 1);
        let p = match (hx & 1, hy & 1) {
            (0, 0) => full,
            (1, 0) => &self.h,
            (0, _) => &self.v,
            _ => &self.hv,
        };
        p.data[p.idx(x, y)] as i32
    }

    pub fn predict(&self, full: &Plane, x: usize, y: usize, w: usize, h: usize, mv: [i32; 2], out: &mut [u8]) {
        let (fx, fy) = (mv[0] & 1, mv[1] & 1);
        for r in 0..h {
            for c in 0..w {
                let qx = (x + c) as isize * 4 + mv[0] as isize;
                let qy = (y + r) as isize * 4 + mv[1] as isize;
                let a = self.half_grid(full, qx >> 1, qy >> 1);
                let v = if fx == 0 && fy == 0 {
                    a
                } else {
                    let b = self.half_grid(full, (qx + fx as isize) >> 1, (qy + fy as isize) >> 1);
                    (a + b + 1) >> 1
                };
                out[r * w + c] = v as u8;
            }
        }
    }
}

pub struct Search<'a> {
    pub src: &'a Plane,
    pub refp: &'a Plane,
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    /// the predicted vector (quarter pel), for the vector cost
    pub pmv: [i32; 2],
    /// λ for SAD/SATD + λ·bits, in 1/16
    pub lambda16: u32,
    /// precomputed half-pel samples of `refp` (fast sub-pel search)
    pub half: Option<&'a HalfPlanes>,
}

impl Search<'_> {
    fn cost_int(&self, m: [i32; 2]) -> u32 {
        let s = sad(self.src, self.x, self.y, self.refp, self.x as isize + m[0] as isize, self.y as isize + m[1] as isize, self.w, self.h);
        s + (self.lambda16 * mv_bits16([m[0] * 4 - self.pmv[0], m[1] * 4 - self.pmv[1]]) >> 8)
    }

    /// hexagon search around the best of the candidates (integer pel); returns
    /// the vector in quarter pel and its cost
    pub fn hexagon(&self, candidates: &[[i32; 2]], range: i32) -> ([i32; 2], u32) {
        let b = Bounds::new(self.refp, self.x, self.y, self.w, self.h);
        let lim = |v: [i32; 2]| -> [i32; 2] {
            let v = b.clamp(v);
            [v[0].clamp(-range, range), v[1].clamp(-range, range)]
        };
        let mut best = lim([0, 0]);
        let mut bc = self.cost_int(best);
        for c in candidates {
            let v = lim([(c[0] + 2) >> 2, (c[1] + 2) >> 2]);
            let k = self.cost_int(v);
            if k < bc {
                bc = k;
                best = v;
            }
        }
        const HEX: [[i32; 2]; 6] = [[-2, 0], [-1, -2], [1, -2], [2, 0], [1, 2], [-1, 2]];
        for _ in 0..64 {
            let mut moved = false;
            let center = best;
            for d in HEX {
                let v = lim([center[0] + d[0], center[1] + d[1]]);
                if v == center {
                    continue;
                }
                let k = self.cost_int(v);
                if k < bc {
                    bc = k;
                    best = v;
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
        // a final small diamond
        loop {
            let center = best;
            for d in [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]] {
                let v = lim([center[0] + d[0], center[1] + d[1]]);
                let k = self.cost_int(v);
                if k < bc {
                    bc = k;
                    best = v;
                }
            }
            if best == center {
                break;
            }
        }
        ([best[0] * 4, best[1] * 4], bc)
    }

    /// half- then quarter-pel refinement with SATD; returns the vector and its SATD cost
    pub fn subpel(&self, start: [i32; 2]) -> ([i32; 2], u32) {
        let mut pred = [0u8; 256];
        let cost = |m: [i32; 2], pred: &mut [u8; 256]| -> u32 {
            // half-pel positions come exactly from the precomputed planes;
            // quarter-pel ones need the real filter
            match self.half {
                Some(hp) if m[0] & 1 == 0 && m[1] & 1 == 0 => hp.predict(self.refp, self.x, self.y, self.w, self.h, m, pred),
                _ => mc::luma(self.refp, self.x, self.y, self.w, self.h, m[0], m[1], pred),
            }
            satd(self.src, self.x, self.y, &pred[..], self.w, self.h) + (self.lambda16 * mv_bits16([m[0] - self.pmv[0], m[1] - self.pmv[1]]) >> 8)
        };
        let mut best = start;
        let mut bc = cost(best, &mut pred);
        let b = Bounds::new(self.refp, self.x, self.y, self.w, self.h);
        for step in [2, 1] {
            let center = best;
            for d in [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]] {
                let v = [center[0] + d[0] * step, center[1] + d[1] * step];
                if v[0] < b.min[0] * 4 || v[1] < b.min[1] * 4 || v[0] > b.max[0] * 4 || v[1] > b.max[1] * 4 {
                    continue;
                }
                let k = cost(v, &mut pred);
                if k < bc {
                    bc = k;
                    best = v;
                }
            }
        }
        (best, bc)
    }
}

/// the exhaustive integer search result of one macroblock
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FullResult {
    /// best 16×16 vector (integer pel) and its cost
    pub mv16: [i32; 2],
    pub cost16: u32,
    /// best vector of each 8×8 quadrant
    pub mv8: [[i32; 2]; 4],
}

/// the cost the exhaustive search minimizes: SAD + λ·(|x|+|y|), ties broken by
/// scan order (dy, then dx); the CUDA kernel computes exactly the same
#[inline]
pub fn full_cost(sad: u32, m: [i32; 2], lam: u32) -> u32 {
    sad + ((lam * (m[0].unsigned_abs() + m[1].unsigned_abs())) >> 4)
}

/// exhaustive search over ±range for every macroblock (the CPU reference)
pub fn full_search(src: &Plane, refp: &Plane, range: i32, lam: u32) -> Vec<FullResult> {
    let (mbw, mbh) = (src.w / 16, src.h / 16);
    let mut out = vec![FullResult::default(); mbw * mbh];
    for mby in 0..mbh {
        for mbx in 0..mbw {
            let (x, y) = (mbx * 16, mby * 16);
            let mut best16 = (u64::MAX, [0, 0]);
            let mut best8 = [(u64::MAX, [0i32, 0i32]); 4];
            let mut idx = 0u64;
            for dy in -range..=range {
                for dx in -range..=range {
                    let mut s8 = [0u32; 4];
                    for q in 0..4 {
                        let (qx, qy) = ((q % 2) * 8, (q / 2) * 8);
                        s8[q] = sad_scalar(src, x + qx, y + qy, refp, (x + qx) as isize + dx as isize, (y + qy) as isize + dy as isize, 8, 8);
                    }
                    let m = [dx, dy];
                    let k16 = (full_cost(s8.iter().sum(), m, lam) as u64) << 24 | idx;
                    if k16 < best16.0 {
                        best16 = (k16, m);
                    }
                    for q in 0..4 {
                        let k = (full_cost(s8[q], m, lam) as u64) << 24 | idx;
                        if k < best8[q].0 {
                            best8[q] = (k, m);
                        }
                    }
                    idx += 1;
                }
            }
            out[mby * mbw + mbx] = FullResult { mv16: best16.1, cost16: (best16.0 >> 24) as u32, mv8: [best8[0].1, best8[1].1, best8[2].1, best8[3].1] };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn textured(w: usize, h: usize, shift: (i32, i32)) -> Plane {
        let mut p = Plane::new(w, h, 80);
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = (x as i32 + shift.0, y as i32 + shift.1);
                let v = ((sx * 7 + sy * 13) ^ (sx * sy / 3)) & 255;
                p.set(x, y, v as u8);
            }
        }
        p.extend_edges();
        p
    }

    #[test]
    fn hexagon_finds_a_global_shift() {
        let refp = textured(64, 64, (0, 0));
        let src = textured(64, 64, (3, -2)); // src(x) = ref(x + 3, y - 2)
        let s = Search { src: &src, refp: &refp, x: 24, y: 24, w: 16, h: 16, pmv: [0, 0], lambda16: 16, half: None };
        let (mv, _) = s.hexagon(&[[0, 0]], 32);
        assert_eq!(mv, [12, -8]);
        let (mv, _) = s.subpel(mv);
        assert_eq!(mv, [12, -8]);
    }

    #[test]
    fn full_search_finds_a_global_shift() {
        let refp = textured(64, 48, (0, 0));
        let src = textured(64, 48, (-5, 4));
        let r = full_search(&src, &refp, 8, 4);
        // a macroblock away from the borders
        assert_eq!(r[1 * 4 + 1].mv16, [-5, 4]);
        assert_eq!(r[1 * 4 + 1].mv8, [[-5, 4]; 4]);
    }

    #[test]
    fn half_planes_match_motion_compensation() {
        let p = textured(64, 48, (1, 2));
        let hp = HalfPlanes::new(&p);
        let mut a = [0u8; 256];
        let mut b = [0u8; 256];
        // full and half positions are exact
        for mv in [[0, 0], [2, 0], [0, 2], [2, 2], [-6, 10], [14, -2]] {
            crate::mc::luma(&p, 16, 16, 16, 16, mv[0], mv[1], &mut a);
            hp.predict(&p, 16, 16, 16, 16, mv, &mut b);
            assert_eq!(a, b, "{mv:?}");
        }
        // quarter positions are close
        crate::mc::luma(&p, 16, 16, 16, 16, 1, 3, &mut a);
        hp.predict(&p, 16, 16, 16, 16, [1, 3], &mut b);
        // (a noise-like texture: compare the mean, single samples can differ a lot)
        let err = a.iter().zip(&b).map(|(x, y)| (*x as i32 - *y as i32).abs()).sum::<i32>() as f64 / 256.0;
        assert!(err < 12.0, "{err}");
    }

    #[test]
    fn satd_of_a_flat_difference_is_its_dc() {
        let d = [3i32; 64];
        assert_eq!(satd8(&d), (3 * 64 + 2) / 4);
    }
}
