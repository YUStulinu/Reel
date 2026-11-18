//! The AVX2 kernels of simd/reel_simd.c, used when the CPU has AVX2. Each
//! returns exactly what the Rust scalar version returns (checked by tests);
//! `None`/`false` means "not available, use the scalar code".
use crate::frame::Plane;

#[cfg(reel_simd)]
mod ffi {
    extern "C" {
        pub fn reel_forward_avx2(n: i32, m: *const i32, x: *const i32, y: *mut i32);
        pub fn reel_inverse_avx2(n: i32, m: *const i32, y: *const i32, x: *mut i32);
        pub fn reel_sad_avx2(a: *const u8, sa: isize, b: *const u8, sb: isize, w: i32, h: i32) -> u32;
        pub fn reel_satd8_avx2(d: *const i32) -> u32;
    }
}

/// AVX2 is present and not disabled with REEL_NO_SIMD
pub fn enabled() -> bool {
    #[cfg(reel_simd)]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("REEL_NO_SIMD").is_none() && std::is_x86_feature_detected!("avx2"))
    }
    #[cfg(not(reel_simd))]
    false
}

#[cfg(reel_simd)]
fn basis(n: usize) -> &'static [i32] {
    let b = crate::tables::bases();
    match n {
        4 => &b.m4,
        8 => &b.m8,
        _ => &b.m16,
    }
}

pub fn forward(n: usize, x: &[i32], y: &mut [i32]) -> bool {
    #[cfg(reel_simd)]
    if enabled() {
        assert!(x.len() >= n * n && y.len() >= n * n);
        unsafe { ffi::reel_forward_avx2(n as i32, basis(n).as_ptr(), x.as_ptr(), y.as_mut_ptr()) };
        return true;
    }
    let _ = (n, x, y);
    false
}

pub fn inverse(n: usize, y: &[i32], x: &mut [i32]) -> bool {
    #[cfg(reel_simd)]
    if enabled() {
        assert!(x.len() >= n * n && y.len() >= n * n);
        unsafe { ffi::reel_inverse_avx2(n as i32, basis(n).as_ptr(), y.as_ptr(), x.as_mut_ptr()) };
        return true;
    }
    let _ = (n, y, x);
    false
}

pub fn sad(a: &Plane, ax: usize, ay: usize, b: &Plane, bx: isize, by: isize, w: usize, h: usize) -> Option<u32> {
    #[cfg(reel_simd)]
    if enabled() && (w == 16 || w == 8) {
        let ia = a.idx(ax as isize, ay as isize);
        let ib = b.idx(bx, by);
        // the last row must lie inside each buffer
        assert!(ia + (h - 1) * a.stride + w <= a.data.len() && ib + (h - 1) * b.stride + w <= b.data.len());
        return Some(unsafe { ffi::reel_sad_avx2(a.data.as_ptr().add(ia), a.stride as isize, b.data.as_ptr().add(ib), b.stride as isize, w as i32, h as i32) });
    }
    let _ = (a, ax, ay, b, bx, by, w, h);
    None
}

pub fn satd8(d: &[i32; 64]) -> Option<u32> {
    #[cfg(reel_simd)]
    if enabled() {
        return Some(unsafe { ffi::reel_satd8_avx2(d.as_ptr()) });
    }
    let _ = d;
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dct, me};

    fn rng(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    #[test]
    fn avx2_matches_scalar() {
        if !enabled() {
            eprintln!("AVX2 kernels not available: nothing to compare");
            return;
        }
        let mut s = 42u64;
        for n in [4usize, 8, 16] {
            for _ in 0..200 {
                let x: Vec<i32> = (0..n * n).map(|_| (rng(&mut s) % 511) as i32 - 255).collect();
                let (mut a, mut b) = (vec![0; n * n], vec![0; n * n]);
                dct::forward_scalar(n, &x, &mut a);
                assert!(forward(n, &x, &mut b));
                assert_eq!(a, b, "forward {n}");
                // big coefficients exercise the 16-bit clipping of the inverse
                let y: Vec<i32> = (0..n * n).map(|_| (rng(&mut s) % 65535) as i32 - 32767).collect();
                dct::inverse_scalar(n, &y, &mut a);
                assert!(inverse(n, &y, &mut b));
                assert_eq!(a, b, "inverse {n}");
            }
        }
        for _ in 0..200 {
            let mut d = [0i32; 64];
            for v in d.iter_mut() {
                *v = (rng(&mut s) % 511) as i32 - 255;
            }
            assert_eq!(satd8(&d), Some(me::satd8(&d)));
        }
        let mut p = Plane::new(64, 64, 16);
        let mut q = Plane::new(64, 64, 16);
        for v in p.data.iter_mut() {
            *v = rng(&mut s) as u8;
        }
        for v in q.data.iter_mut() {
            *v = rng(&mut s) as u8;
        }
        for (w, h) in [(16, 16), (8, 8), (16, 8), (8, 16)] {
            for &(x, y, dx, dy) in &[(0usize, 0usize, -5isize, 3isize), (16, 24, 7, -9), (40, 40, 0, 0)] {
                let want = me::sad_scalar(&p, x, y, &q, x as isize + dx, y as isize + dy, w, h);
                assert_eq!(sad(&p, x, y, &q, x as isize + dx, y as isize + dy, w, h), Some(want));
            }
        }
    }
}
