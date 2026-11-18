//! The in-loop deblocking filter: smooths block edges whose step is small
//! enough to be a coding artifact rather than a real edge. Boundary strength
//! comes from intra coding, coded coefficients and motion vector differences.
use crate::frame::{Frame, Plane};
use crate::syntax::Grid;
use crate::tables::{tc0, ALPHA, BETA};

/// boundary strength between luma 4×4 blocks p (left/above) and q
fn strength(g: &Grid, p: (usize, usize), q: (usize, usize), mb_edge: bool) -> u8 {
    let mbp = &g.mbs[(p.1 / 4) * g.mbw + p.0 / 4];
    let mbq = &g.mbs[(q.1 / 4) * g.mbw + q.0 / 4];
    if mbp.intra || mbq.intra {
        return if mb_edge { 4 } else { 3 };
    }
    let nzp = mbp.nz >> ((p.1 % 4) * 4 + p.0 % 4) & 1;
    let nzq = mbq.nz >> ((q.1 % 4) * 4 + q.0 % 4) & 1;
    if nzp | nzq != 0 {
        return 2;
    }
    let w8 = g.mbw * 2;
    let (ip, iq) = ((p.1 / 2) * w8 + p.0 / 2, (q.1 / 2) * w8 + q.0 / 2);
    if g.dir[ip] != g.dir[iq] {
        return 1;
    }
    for l in 0..2 {
        if g.has_mv[l][ip] {
            let (mp, mq) = (g.mv[l][ip], g.mv[l][iq]);
            if (mp[0] - mq[0]).abs() >= 4 || (mp[1] - mq[1]).abs() >= 4 {
                return 1;
            }
        }
    }
    0
}

/// filters one line of samples across an edge: `i` is q0, `step` the distance
/// between samples across the edge
#[inline]
fn line(d: &mut [u8], i: usize, step: usize, bs: u8, qp: i32, luma: bool) {
    let (alpha, beta) = (ALPHA[qp as usize] as i32, BETA[qp as usize] as i32);
    let (p0, p1, q0, q1) = (d[i - step] as i32, d[i - 2 * step] as i32, d[i] as i32, d[i + step] as i32);
    // the third and fourth samples exist for luma only (chroma edges are 4 samples apart)
    let (p2, p3, q2, q3) = if luma {
        (d[i - 3 * step] as i32, d[i - 4 * step] as i32, d[i + 2 * step] as i32, d[i + 3 * step] as i32)
    } else {
        (0, 0, 0, 0)
    };
    if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
        return;
    }
    if bs == 4 {
        if luma {
            let small = (p0 - q0).abs() < (alpha >> 2) + 2;
            if small && (p2 - p0).abs() < beta {
                d[i - step] = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) as u8;
                d[i - 2 * step] = ((p2 + p1 + p0 + q0 + 2) >> 2) as u8;
                d[i - 3 * step] = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3) as u8;
            } else {
                d[i - step] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
            }
            if small && (q2 - q0).abs() < beta {
                d[i] = ((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3) as u8;
                d[i + step] = ((p0 + q0 + q1 + q2 + 2) >> 2) as u8;
                d[i + 2 * step] = ((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3) as u8;
            } else {
                d[i] = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;
            }
        } else {
            d[i - step] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
            d[i] = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;
        }
        return;
    }
    let t0 = tc0(qp, bs);
    let mut tc = t0;
    let (ap, aq) = if luma { ((p2 - p0).abs(), (q2 - q0).abs()) } else { (beta, beta) };
    if luma {
        tc += (ap < beta) as i32 + (aq < beta) as i32;
    } else {
        tc += 1;
    }
    let delta = (((q0 - p0) * 4 + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
    d[i - step] = (p0 + delta).clamp(0, 255) as u8;
    d[i] = (q0 - delta).clamp(0, 255) as u8;
    if luma {
        if ap < beta {
            d[i - 2 * step] = (p1 + ((p2 + ((p0 + q0 + 1) >> 1) - 2 * p1) >> 1).clamp(-t0, t0)) as u8;
        }
        if aq < beta {
            d[i + step] = (q1 + ((q2 + ((p0 + q0 + 1) >> 1) - 2 * q1) >> 1).clamp(-t0, t0)) as u8;
        }
    }
}

/// is there a transform or partition edge `e` (0, 4, 8, 12) inside macroblock `m`?
fn internal_edge(g: &Grid, m: usize, e: usize) -> bool {
    let s = &g.mbs[m];
    match s.tsize {
        4 => true,
        8 => e == 8,
        _ => e == 8 && !s.intra, // partition edges of inter macroblocks
    }
}

fn filter_plane(p: &mut Plane, g: &Grid, qp: i32, chroma: bool) {
    let sub = if chroma { 2 } else { 1 }; // pixels per luma pixel
    let mbsz = 16 / sub;
    let stride = p.stride;
    for mby in 0..g.mbh {
        for mbx in 0..g.mbw {
            let m = mby * g.mbw + mbx;
            for dir in 0..2 {
                // dir 0: vertical edges (filter horizontally), 1: horizontal edges
                for e in (0..16).step_by(4) {
                    let at_mb = e == 0;
                    if at_mb && (if dir == 0 { mbx == 0 } else { mby == 0 }) {
                        continue;
                    }
                    if !at_mb && !internal_edge(g, m, e) {
                        continue;
                    }
                    if chroma && e % 8 != 0 {
                        continue; // chroma: 8×8 blocks of 4×4 chroma pixels -> edges at 0 and 8 luma
                    }
                    for seg in 0..4 {
                        // luma 4×4 coordinates of the blocks on each side of this 4-pixel segment
                        let (qb, pb) = if dir == 0 {
                            let q = (mbx * 4 + e / 4, mby * 4 + seg);
                            (q, (q.0 - 1, q.1))
                        } else {
                            let q = (mbx * 4 + seg, mby * 4 + e / 4);
                            (q, (q.0, q.1 - 1))
                        };
                        let bs = strength(g, pb, qb, at_mb);
                        if bs == 0 {
                            continue;
                        }
                        let n = 4 / sub; // samples along the segment
                        for k in 0..n {
                            let (x, y) = if dir == 0 {
                                (mbx * mbsz + e / sub, mby * mbsz + seg * n + k)
                            } else {
                                (mbx * mbsz + seg * n + k, mby * mbsz + e / sub)
                            };
                            let i = p.idx(x as isize, y as isize);
                            let step = if dir == 0 { 1 } else { stride };
                            line(&mut p.data, i, step, bs, qp, !chroma);
                        }
                    }
                }
            }
        }
    }
}

pub fn deblock(f: &mut Frame, g: &Grid, qp: i32) {
    let qp = qp.clamp(0, 51);
    if ALPHA[qp as usize] == 0 {
        return;
    }
    let [y, u, v] = &mut f.planes;
    filter_plane(y, g, qp, false);
    filter_plane(u, g, qp, true);
    filter_plane(v, g, qp, true);
}
