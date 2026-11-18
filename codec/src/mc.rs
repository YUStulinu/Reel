//! Motion compensation: quarter-pel luma with 8-tap filters and eighth-pel
//! chroma with 4-tap filters (separable, exact integer arithmetic).
use crate::frame::Plane;

const LUMA: [[i32; 8]; 4] = [
    [0, 0, 0, 64, 0, 0, 0, 0],
    [-1, 4, -10, 58, 17, -5, 1, 0],
    [-1, 4, -11, 40, 40, -11, 4, -1],
    [0, 1, -5, 17, 58, -10, 4, -1],
];
const CHROMA: [[i32; 4]; 8] = [
    [0, 64, 0, 0],
    [-2, 58, 10, -2],
    [-4, 54, 16, -2],
    [-6, 46, 28, -4],
    [-4, 36, 36, -4],
    [-4, 28, 46, -6],
    [-2, 16, 54, -4],
    [-2, 10, 58, -2],
];

/// copies the window [x0, x0+w) × [y0, y0+h) of `p` into `win` (w wide),
/// replicating edges for anything outside the padded plane
fn window(p: &Plane, x0: isize, y0: isize, w: usize, h: usize, win: &mut [i16]) {
    let pad = p.pad as isize;
    let inside = x0 >= -pad && y0 >= -pad && x0 + w as isize <= p.w as isize + pad && y0 + h as isize <= p.h as isize + pad;
    for r in 0..h {
        if inside {
            let i = p.idx(x0, y0 + r as isize);
            for c in 0..w {
                win[r * w + c] = p.data[i + c] as i16;
            }
        } else {
            for c in 0..w {
                win[r * w + c] = p.clamped(x0 + c as isize, y0 + r as isize) as i16;
            }
        }
    }
}

fn filter<const T: usize>(p: &Plane, x: isize, y: isize, w: usize, h: usize, fx: [i32; T], fy: [i32; T], out: &mut [u8]) {
    let half = (T / 2 - 1) as isize; // taps before the sample
    let ww = w + T - 1;
    let wh = h + T - 1;
    let mut win = [0i16; (16 + 7) * (16 + 7)];
    window(p, x - half, y - half, ww, wh, &mut win);
    let hx = fx[T / 2 - 1] == 64;
    let hy = fy[T / 2 - 1] == 64;
    match (hx, hy) {
        (true, true) => {
            for r in 0..h {
                for c in 0..w {
                    out[r * w + c] = win[(r + half as usize) * ww + c + half as usize] as u8;
                }
            }
        }
        (false, true) => {
            for r in 0..h {
                let row = &win[(r + half as usize) * ww..];
                for c in 0..w {
                    let s: i32 = (0..T).map(|k| fx[k] * row[c + k] as i32).sum();
                    out[r * w + c] = ((s + 32) >> 6).clamp(0, 255) as u8;
                }
            }
        }
        (true, false) => {
            for r in 0..h {
                for c in 0..w {
                    let s: i32 = (0..T).map(|k| fy[k] * win[(r + k) * ww + c + half as usize] as i32).sum();
                    out[r * w + c] = ((s + 32) >> 6).clamp(0, 255) as u8;
                }
            }
        }
        (false, false) => {
            let mut tmp = [0i32; (16 + 7) * 16];
            for r in 0..wh {
                for c in 0..w {
                    tmp[r * w + c] = (0..T).map(|k| fx[k] * win[r * ww + c + k] as i32).sum();
                }
            }
            for r in 0..h {
                for c in 0..w {
                    let s: i32 = (0..T).map(|k| fy[k] * tmp[(r + k) * w + c]).sum();
                    out[r * w + c] = (((s >> 6) + 32) >> 6).clamp(0, 255) as u8;
                }
            }
        }
    }
}

/// predicts a w×h luma block whose top-left is (x, y), displaced by the
/// quarter-pel vector (mx, my)
pub fn luma(p: &Plane, x: usize, y: usize, w: usize, h: usize, mx: i32, my: i32, out: &mut [u8]) {
    let ix = x as isize + (mx >> 2) as isize;
    let iy = y as isize + (my >> 2) as isize;
    filter::<8>(p, ix, iy, w, h, LUMA[(mx & 3) as usize], LUMA[(my & 3) as usize], out);
}

/// predicts a w×h chroma block at chroma (x, y) for the luma vector (mx, my)
pub fn chroma(p: &Plane, x: usize, y: usize, w: usize, h: usize, mx: i32, my: i32, out: &mut [u8]) {
    let ix = x as isize + (mx >> 3) as isize;
    let iy = y as isize + (my >> 3) as isize;
    filter::<4>(p, ix, iy, w, h, CHROMA[(mx & 7) as usize], CHROMA[(my & 7) as usize], out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp() -> Plane {
        let mut p = Plane::new(64, 64, 16);
        for y in 0..64 {
            for x in 0..64 {
                p.set(x, y, (2 * x + y) as u8);
            }
        }
        p.extend_edges();
        p
    }

    #[test]
    fn integer_vectors_copy() {
        let p = ramp();
        let mut out = [0u8; 64];
        luma(&p, 16, 16, 8, 8, 4 * 3, -4 * 2, &mut out);
        assert_eq!(out[0], p.at(19, 14));
        assert_eq!(out[63], p.at(26, 21));
    }

    #[test]
    fn fractional_vectors_interpolate_a_ramp() {
        let p = ramp();
        let mut out = [0u8; 64];
        // half a pixel right and a quarter down on a linear ramp: +1 and +0.25
        luma(&p, 16, 16, 8, 8, 2, 1, &mut out);
        let want = 2 * 16 + 16 + 1; // 49.25
        assert!((out[0] as i32 - want).abs() <= 1, "{}", out[0]);
        let mut c = [0u8; 16];
        chroma(&p, 16, 16, 4, 4, 4, 0, &mut c); // half a chroma pixel
        assert_eq!(c[0], 2 * 16 + 16 + 1);
    }

    #[test]
    fn far_vectors_stay_in_bounds() {
        let p = ramp();
        let mut out = [0u8; 256];
        luma(&p, 0, 0, 16, 16, -4000, 9000, &mut out);
        assert_eq!(out[0], p.clamped(-1000, 2000));
    }
}
