//! Exhaustive motion search on the GPU (cuda/me.cu), built with `--features cuda`.
use crate::frame::Plane;
use crate::me::FullResult;

#[cfg(feature = "cuda")]
extern "C" {
    fn reel_cuda_full_search(
        src: *const u8,
        refp: *const u8,
        bytes: usize,
        stride: i32,
        origin: i32,
        w: i32,
        h: i32,
        range: i32,
        lam: u32,
        out: *mut i32,
        err: *mut u8,
        errlen: i32,
    ) -> i32;
}

/// the same result as `me::full_search`, computed on the GPU
#[cfg(feature = "cuda")]
pub fn full_search(src: &Plane, refp: &Plane, range: i32, lam: u32) -> Result<Vec<FullResult>, String> {
    assert!(src.stride == refp.stride && src.data.len() == refp.data.len() && src.pad as i32 >= range + 8);
    let mbs = (src.w / 16) * (src.h / 16);
    let mut out = vec![0i32; mbs * 11];
    let mut err = [0u8; 256];
    let rc = unsafe {
        reel_cuda_full_search(
            src.data.as_ptr(),
            refp.data.as_ptr(),
            src.data.len(),
            src.stride as i32,
            src.idx(0, 0) as i32,
            src.w as i32,
            src.h as i32,
            range,
            lam,
            out.as_mut_ptr(),
            err.as_mut_ptr(),
            err.len() as i32,
        )
    };
    if rc != 0 {
        let n = err.iter().position(|&b| b == 0).unwrap_or(err.len());
        return Err(String::from_utf8_lossy(&err[..n]).into_owned());
    }
    Ok(out
        .chunks(11)
        .map(|o| FullResult { mv16: [o[0], o[1]], cost16: o[2] as u32, mv8: [[o[3], o[4]], [o[5], o[6]], [o[7], o[8]], [o[9], o[10]]] })
        .collect())
}

#[cfg(not(feature = "cuda"))]
pub fn full_search(_src: &Plane, _refp: &Plane, _range: i32, _lam: u32) -> Result<Vec<FullResult>, String> {
    Err("this build has no CUDA support (build with --features cuda)".into())
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::frame::LUMA_PAD;

    #[test]
    fn gpu_matches_the_cpu_reference() {
        let (w, h) = (160, 96);
        let mut s = 1u64;
        let mut make = |shift: i32| {
            let mut p = Plane::new(w, h, LUMA_PAD);
            for y in 0..h {
                for x in 0..w {
                    let (sx, sy) = (x as i32 + shift, y as i32 - shift / 2);
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let noise = (s >> 60) as i32;
                    p.set(x, y, (((sx * 5 + sy * 11) ^ (sx * sy / 7)) & 255).wrapping_add(noise) as u8);
                }
            }
            p.extend_edges();
            p
        };
        let refp = make(0);
        let src = make(3);
        for (range, lam) in [(8, 0), (16, 40), (32, 7)] {
            let cpu = crate::me::full_search(&src, &refp, range, lam);
            let gpu = full_search(&src, &refp, range, lam).expect("CUDA");
            assert_eq!(cpu, gpu, "range {range} lam {lam}");
        }
    }
}
