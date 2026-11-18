//! Reconstruction shared by the encoder and the decoder: prediction plus the
//! dequantized, inverse-transformed residual.
use crate::dct;
use crate::frame::Frame;
use crate::intra;
use crate::mc;
use crate::syntax::{chroma_tsize, luma_blocks, partitions, Mb, DIR_L0, DIR_L1};

/// adds the residual of one n×n transform block to `pred` (stride `ps`),
/// writing the clipped result to `out` (stride `os`)
pub fn add_residual(n: usize, qp: i32, levels: &[i32], pred: &[u8], ps: usize, out: &mut [u8], os: usize) {
    if levels[..n * n].iter().all(|&v| v == 0) {
        for r in 0..n {
            out[r * os..r * os + n].copy_from_slice(&pred[r * ps..r * ps + n]);
        }
        return;
    }
    let mut co = [0i32; 256];
    let mut res = [0i32; 256];
    dct::dequantize(n, qp, levels, &mut co);
    dct::inverse(n, &co, &mut res);
    for r in 0..n {
        for c in 0..n {
            out[r * os + c] = (pred[r * ps + c] as i32 + res[r * n + c]).clamp(0, 255) as u8;
        }
    }
}

/// the reference pictures of a frame: list 0 (past) and list 1 (future, B frames)
pub type Refs<'a> = [Option<&'a Frame>; 2];

/// motion-compensated prediction of a whole inter macroblock; bi-predicted
/// partitions average both lists with rounding
pub fn inter_pred(refs: Refs, mbx: usize, mby: usize, mb: &Mb, luma: &mut [u8; 256], chroma: &mut [[u8; 64]; 2]) {
    let (x0, y0) = (mbx * 16, mby * 16);
    let mut tmp = [[0u8; 256]; 2];
    for &(q, wb, hb) in partitions(mb.part) {
        let (px, py) = ((q % 2) * 8, (q / 2) * 8);
        let (w, h) = (wb * 8, hb * 8);
        let dir = mb.dir[q];
        let lists: Vec<usize> = [DIR_L0, DIR_L1].iter().enumerate().filter(|(_, d)| dir & **d != 0).map(|(l, _)| l).collect();
        let mv = |l: usize| if l == 0 { mb.mv[q] } else { mb.mv1[q] };
        let refp = |l: usize| refs[l].expect("a missing reference picture");
        let blend = |dst: &mut [u8], stride: usize, ox: usize, oy: usize, bw: usize, bh: usize, t: &[[u8; 256]; 2]| {
            for r in 0..bh {
                for c in 0..bw {
                    dst[(oy + r) * stride + ox + c] = if lists.len() == 2 {
                        ((t[0][r * bw + c] as u16 + t[1][r * bw + c] as u16 + 1) >> 1) as u8
                    } else {
                        t[0][r * bw + c]
                    };
                }
            }
        };
        for (i, &l) in lists.iter().enumerate() {
            let [mx, my] = mv(l);
            mc::luma(&refp(l).planes[0], x0 + px, y0 + py, w, h, mx, my, &mut tmp[i]);
        }
        blend(&mut luma[..], 16, px, py, w, h, &tmp);
        let (cw, ch) = (w / 2, h / 2);
        for k in 0..2 {
            for (i, &l) in lists.iter().enumerate() {
                let [mx, my] = mv(l);
                mc::chroma(&refp(l).planes[1 + k], x0 / 2 + px / 2, y0 / 2 + py / 2, cw, ch, mx, my, &mut tmp[i]);
            }
            blend(&mut chroma[k][..], 8, px / 2, py / 2, cw, ch, &tmp);
        }
    }
}

fn write_block(frame: &mut Frame, plane: usize, x: usize, y: usize, n: usize, data: &[u8]) {
    let p = &mut frame.planes[plane];
    for r in 0..n {
        let i = p.idx(x as isize, (y + r) as isize);
        p.data[i..i + n].copy_from_slice(&data[r * n..r * n + n]);
    }
}

/// chroma residual (both planes) on top of an 8×8 prediction per plane
pub fn chroma_residual(mb: &Mb, qp: i32, pred: &[[u8; 64]; 2], out: &mut [[u8; 64]; 2]) {
    let cn = chroma_tsize(mb.tsize);
    for k in 0..2 {
        if cn == 8 {
            add_residual(8, qp, &mb.chroma[k], &pred[k], 8, &mut out[k], 8);
        } else {
            for t in 0..4 {
                let (x, y) = ((t % 2) * 4, (t / 2) * 4);
                let o = y * 8 + x;
                add_residual(4, qp, &mb.chroma[k][t * 16..], &pred[k][o..], 8, &mut out[k][o..], 8);
            }
        }
    }
}

/// luma residual of the transform blocks lying inside the 16×16 region
/// selected by `only` (a quadrant, or None for all)
pub fn luma_residual(mb: &Mb, qp: i32, pred: &[u8], out: &mut [u8], only: Option<usize>) {
    let n = mb.tsize as usize;
    for (x, y, off) in luma_blocks(mb.tsize) {
        if let Some(q) = only {
            if (y / 8) * 2 + x / 8 != q {
                continue;
            }
        }
        let o = y * 16 + x;
        add_residual(n, qp, &mb.luma[off..], &pred[o..], 16, &mut out[o..], 16);
    }
}

/// reconstructs one macroblock into `frame` (the picture being decoded)
pub fn reconstruct(frame: &mut Frame, refs: Refs, mbx: usize, mby: usize, mb: &Mb, qp: i32) {
    let (x0, y0) = (mbx * 16, mby * 16);
    let mut lp = [0u8; 256];
    let mut cp = [[0u8; 64]; 2];
    let mut lo = [0u8; 256];
    let mut co = [[0u8; 64]; 2];
    if mb.intra {
        if mb.i16 {
            intra::predict(&frame.planes[0], x0, y0, 16, mb.lmode[0], &mut lp);
            luma_residual(mb, qp, &lp, &mut lo, None);
            write_block(frame, 0, x0, y0, 16, &lo);
        } else {
            // 8×8 blocks in order: each predicts from the ones before it
            for q in 0..4 {
                let (qx, qy) = ((q % 2) * 8, (q / 2) * 8);
                let mut p8 = [0u8; 64];
                intra::predict(&frame.planes[0], x0 + qx, y0 + qy, 8, mb.lmode[q], &mut p8);
                for r in 0..8 {
                    lp[(qy + r) * 16 + qx..(qy + r) * 16 + qx + 8].copy_from_slice(&p8[r * 8..r * 8 + 8]);
                }
                luma_residual(mb, qp, &lp, &mut lo, Some(q));
                let mut b = [0u8; 64];
                for r in 0..8 {
                    b[r * 8..r * 8 + 8].copy_from_slice(&lo[(qy + r) * 16 + qx..(qy + r) * 16 + qx + 8]);
                }
                write_block(frame, 0, x0 + qx, y0 + qy, 8, &b);
            }
        }
        for k in 0..2 {
            intra::predict(&frame.planes[1 + k], x0 / 2, y0 / 2, 8, mb.cmode, &mut cp[k]);
        }
    } else {
        inter_pred(refs, mbx, mby, mb, &mut lp, &mut cp);
        if mb.skip {
            lo = lp;
            co = cp;
        } else {
            luma_residual(mb, qp, &lp, &mut lo, None);
        }
        write_block(frame, 0, x0, y0, 16, &lo);
    }
    if !mb.skip {
        chroma_residual(mb, qp, &cp, &mut co);
    }
    for k in 0..2 {
        write_block(frame, 1 + k, x0 / 2, y0 / 2, 8, &co[k]);
    }
}
