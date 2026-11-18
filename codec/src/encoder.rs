//! The encoder: motion search, a rate-distortion choice among skip, inter
//! partitionings (and, in B frames, past/future/bi-directional prediction),
//! intra 16×16 and intra 8×8, each with 4×4, 8×8 or 16×16 transforms; then
//! entropy coding and the same reconstruction and deblocking as the decoder.
//!
//! Frames are grouped as I/P anchors with `bframes` B frames between them
//! (display order I B B P B B P …, coded as I P B B P B B …). Groups never
//! cross a keyframe, so every keyframe starts an independently decodable run.
use crate::bitstream::{write_packet, StreamInfo, TYPE_B, TYPE_I, TYPE_P};
use crate::dct;
use crate::deblock::deblock;
use crate::frame::{align16, Frame, Plane};
use crate::intra;
use crate::me::{self, FullResult, HalfPlanes, Search};
use crate::range::{Encoder as RangeEncoder, Estimator};
use crate::rc::{self, RateControl};
use crate::recon::{chroma_residual, inter_pred, luma_residual, reconstruct, Refs};
use crate::syntax::{
    self, chroma_tsize, luma_blocks, Ctx, Grid, Mb, MbState, DIR_BI, DIR_L0, DIR_L1, KIND_B, KIND_I, KIND_P, PART_16X16, PART_16X8, PART_8X16, PART_8X8,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MeMethod {
    /// hexagon search from predicted candidates
    Hex,
    /// exhaustive ±range search on the CPU, then refinement
    Full,
    /// exhaustive ±range search on the GPU (CUDA), then refinement
    Cuda,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub rate: rc::Mode,
    pub keyint: usize,
    pub me: MeMethod,
    pub range: i32,
    /// B frames between anchors
    pub bframes: usize,
    /// return the reconstructed picture with every frame
    pub keep_recon: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config { rate: rc::Mode::Cqp(30), keyint: 60, me: MeMethod::Hex, range: 32, bframes: 1, keep_recon: false }
    }
}

#[derive(Clone, Debug)]
pub struct FrameStats {
    /// position in display order
    pub display: usize,
    /// 0 I, 1 P, 2 B
    pub kind: u8,
    pub qp: i32,
    pub bytes: usize,
    /// PSNR of Y, U, V against the source
    pub psnr: [f64; 3],
    /// macroblock decisions: skip, inter, intra
    pub modes: [usize; 3],
}

pub struct Coded {
    /// the packet with its length prefix
    pub packet: Vec<u8>,
    pub stats: FrameStats,
    /// the decoded picture (packed 4:2:0), with `keep_recon`
    pub recon: Option<Vec<u8>>,
}

pub struct Encoder {
    pub cfg: Config,
    pub info: StreamInfo,
    mbw: usize,
    mbh: usize,
    src: Frame,
    cur: Frame,
    past: Option<Frame>,
    last: Option<Frame>,
    /// display positions of the two anchors
    past_at: usize,
    last_at: usize,
    grid: Grid,
    rc: RateControl,
    received: usize,
    queue: Vec<(usize, Vec<u8>)>,
    /// list-0 vectors of the last anchor's macroblocks
    anchor_mv: Vec<[i32; 2]>,
    /// time spent in exhaustive motion search (seconds), for reporting
    pub me_seconds: f64,
}

/// the Lagrange multiplier of SSD + λ·bits (0.6 tuned on three CIF clips, see the README)
fn lambda(qp: i32) -> f64 {
    0.6 * 2f64.powf((qp as f64 - 12.0) / 3.0)
}

fn ssd_block(src: &Plane, x: usize, y: usize, n: usize, rec: &[u8]) -> u64 {
    let mut s = 0u64;
    for r in 0..n {
        let i = src.idx(x as isize, (y + r) as isize);
        for c in 0..n {
            let d = src.data[i + c] as i32 - rec[r * n + c] as i32;
            s += (d * d) as u64;
        }
    }
    s
}

/// residual -> quantized levels for an n×n block of `src` at (x, y) against `pred` (stride ps)
fn quantize_block(src: &Plane, x: usize, y: usize, n: usize, pred: &[u8], ps: usize, qp: i32, deadzone: i32, out: &mut [i32]) {
    let mut d = [0i32; 256];
    for r in 0..n {
        let i = src.idx(x as isize, (y + r) as isize);
        for c in 0..n {
            d[r * n + c] = src.data[i + c] as i32 - pred[r * ps + c] as i32;
        }
    }
    dct::forward(n, &d, out);
    dct::quantize(n, qp, deadzone, out);
}

pub fn psnr(sse: u64, count: usize) -> f64 {
    if sse == 0 {
        return 100.0;
    }
    10.0 * (255.0f64 * 255.0 * count as f64 / sse as f64).log10()
}

fn scale_mv(v: [i32; 2], num: i32, den: i32) -> [i32; 2] {
    let den = den.max(1);
    [(v[0] * num) / den, (v[1] * num) / den]
}

impl Encoder {
    pub fn new(cfg: Config, info: StreamInfo, fps: f64) -> Self {
        let (w, h) = (align16(info.w), align16(info.h));
        let (mbw, mbh) = (w / 16, h / 16);
        Encoder {
            rc: RateControl::new(cfg.rate, fps, w, h),
            cfg,
            info,
            mbw,
            mbh,
            src: Frame::new(w, h),
            cur: Frame::new(w, h),
            past: None,
            last: None,
            past_at: 0,
            last_at: 0,
            grid: Grid::new(mbw, mbh),
            received: 0,
            queue: Vec::new(),
            anchor_mv: vec![[0; 2]; mbw * mbh],
            me_seconds: 0.0,
        }
    }

    /// takes the next picture (display order); returns the frames coded so far, in decode order
    pub fn push(&mut self, yuv: &[u8]) -> Vec<Coded> {
        let at = self.received;
        self.received += 1;
        let mut out = vec![];
        if at % self.cfg.keyint == 0 {
            out.extend(self.flush());
            out.push(self.code(KIND_I, yuv, at));
            return out;
        }
        self.queue.push((at, yuv.to_vec()));
        if self.queue.len() > self.cfg.bframes {
            let (pa, anchor) = self.queue.pop().unwrap();
            out.push(self.code(KIND_P, &anchor, pa));
            for (ba, b) in std::mem::take(&mut self.queue) {
                out.push(self.code(KIND_B, &b, ba));
            }
        }
        out
    }

    /// codes the pictures still queued (as P frames)
    pub fn flush(&mut self) -> Vec<Coded> {
        std::mem::take(&mut self.queue).into_iter().map(|(a, f)| self.code(KIND_P, &f, a)).collect()
    }

    fn code(&mut self, kind: u8, yuv: &[u8], at: usize) -> Coded {
        self.src.load(self.info.w, self.info.h, yuv);
        self.src.extend_edges();
        let qp = self.rc.qp(kind);
        let past = self.past.take();
        let last = self.last.take();
        let refs: Refs = match kind {
            KIND_P => [last.as_ref(), None],
            KIND_B => [past.as_ref(), last.as_ref()],
            _ => [None, None],
        };
        let full = if kind != KIND_P || self.cfg.me == MeMethod::Hex {
            None
        } else {
            let t0 = std::time::Instant::now();
            let lam = (lambda(qp).sqrt() * 16.0) as u32;
            let rp = &refs[0].unwrap().planes[0];
            let r = match self.cfg.me {
                MeMethod::Cuda => crate::gpu::full_search(&self.src.planes[0], rp, self.cfg.range, lam).unwrap_or_else(|e| panic!("CUDA motion search: {e}")),
                _ => me::full_search(&self.src.planes[0], rp, self.cfg.range, lam),
            };
            self.me_seconds += t0.elapsed().as_secs_f64();
            Some(r)
        };
        let halves = [refs[0].map(|f| HalfPlanes::new(&f.planes[0])), refs[1].map(|f| HalfPlanes::new(&f.planes[0]))];
        // temporal distances for scaling the anchor's vectors into B-frame candidates
        let tb = at as i32 - self.past_at as i32;
        let td = self.last_at as i32 - self.past_at as i32;
        let mut enc = RangeEncoder::default();
        let mut cx = Ctx::default();
        self.grid.reset();
        self.grid.tb = tb;
        self.grid.td = td;
        syntax::frame_header(&mut enc, kind, &mut self.grid);
        let mut modes = [0usize; 3];
        let mut new_mv = vec![[0i32; 2]; self.mbw * self.mbh];
        for mby in 0..self.mbh {
            for mbx in 0..self.mbw {
                let col = self.anchor_mv[mby * self.mbw + mbx];
                let mut mb = self.decide(&mut cx, refs, &halves, mbx, mby, kind, qp, full.as_ref().map(|f| f[mby * self.mbw + mbx]), col, (tb, td));
                self.clear_mb(mbx, mby);
                syntax::macroblock(&mut enc, &mut cx, &mut self.grid, mbx, mby, kind, &mut mb);
                self.grid.mbs[mby * self.mbw + mbx] = MbState { skip: mb.skip, intra: mb.intra, cbp: mb.cbp, tsize: mb.tsize, nz: syntax::nz4_of(&mb) };
                reconstruct(&mut self.cur, refs, mbx, mby, &mb, qp);
                new_mv[mby * self.mbw + mbx] = if mb.intra { [0, 0] } else { mb.mv[0] };
                modes[if mb.skip { 0 } else if mb.intra { 2 } else { 1 }] += 1;
            }
        }
        deblock(&mut self.cur, &self.grid, qp);
        let payload = enc.finish();
        let mut packet = Vec::with_capacity(payload.len() + 6);
        write_packet(&mut packet, [TYPE_I, TYPE_P, TYPE_B][kind as usize], qp as u8, &payload);
        let mut psnr_v = [0.0; 3];
        for k in 0..3 {
            let (w, h) = if k == 0 { (self.info.w, self.info.h) } else { ((self.info.w + 1) / 2, (self.info.h + 1) / 2) };
            let (a, b) = (&self.src.planes[k], &self.cur.planes[k]);
            let mut sse = 0u64;
            for y in 0..h {
                for (p, q) in a.row(y)[..w].iter().zip(&b.row(y)[..w]) {
                    let d = *p as i32 - *q as i32;
                    sse += (d * d) as u64;
                }
            }
            psnr_v[k] = psnr(sse, w * h);
        }
        let recon = self.cfg.keep_recon.then(|| self.cur.store(self.info.w, self.info.h));
        self.rc.update(kind, qp, (packet.len() * 8) as f64);
        if kind == KIND_B {
            self.past = past;
            self.last = last;
        } else {
            // the new anchor: rotate like the decoder
            self.cur.extend_edges();
            self.grid.keep_as_anchor();
            let spare = past.unwrap_or_else(|| self.cur.clone());
            let new = std::mem::replace(&mut self.cur, spare);
            self.past = last;
            self.last = Some(new);
            self.past_at = self.last_at;
            self.last_at = at;
            self.anchor_mv = new_mv;
        }
        let stats = FrameStats { display: at, kind, qp, bytes: packet.len(), psnr: psnr_v, modes };
        Coded { packet, stats, recon }
    }

    fn clear_mb(&mut self, mbx: usize, mby: usize) {
        let w8 = self.mbw * 2;
        for q in 0..4 {
            let i = (mby * 2 + q / 2) * w8 + mbx * 2 + q % 2;
            self.grid.done[i] = false;
            self.grid.has_mv[0][i] = false;
            self.grid.has_mv[1][i] = false;
        }
    }

    /// bits (1/256) of a candidate, from the current models
    fn rate(&mut self, cx: &mut Ctx, mbx: usize, mby: usize, kind: u8, mb: &Mb) -> u64 {
        self.clear_mb(mbx, mby);
        let mut est = Estimator::default();
        let mut m = mb.clone();
        syntax::macroblock(&mut est, cx, &mut self.grid, mbx, mby, kind, &mut m);
        est.cost as u64
    }

    fn chroma_ssd(&self, mbx: usize, mby: usize, rec: &[[u8; 64]; 2]) -> u64 {
        (0..2).map(|k| ssd_block(&self.src.planes[1 + k], mbx * 8, mby * 8, 8, &rec[k])).sum()
    }

    /// fills the residual levels of `mb` for its transform size against the given prediction
    fn residual(&self, mb: &mut Mb, mbx: usize, mby: usize, qp: i32, lp: &[u8; 256], cp: &[[u8; 64]; 2], only_luma: Option<usize>) {
        let (x0, y0) = (mbx * 16, mby * 16);
        let dz = if mb.intra { 21 } else { 11 };
        let n = mb.tsize as usize;
        for (x, y, off) in luma_blocks(mb.tsize) {
            if let Some(q) = only_luma {
                if (y / 8) * 2 + x / 8 != q {
                    continue;
                }
            }
            quantize_block(&self.src.planes[0], x0 + x, y0 + y, n, &lp[y * 16 + x..], 16, qp, dz, &mut mb.luma[off..off + n * n]);
        }
        if only_luma.is_some() {
            return;
        }
        let cn = chroma_tsize(mb.tsize);
        for k in 0..2 {
            if cn == 8 {
                quantize_block(&self.src.planes[1 + k], x0 / 2, y0 / 2, 8, &cp[k], 8, qp, dz, &mut mb.chroma[k]);
            } else {
                for t in 0..4 {
                    let (x, y) = ((t % 2) * 4, (t / 2) * 4);
                    quantize_block(&self.src.planes[1 + k], x0 / 2 + x, y0 / 2 + y, 4, &cp[k][y * 8 + x..], 8, qp, dz, &mut mb.chroma[k][t * 16..t * 16 + 16]);
                }
            }
        }
    }

    /// J = SSD·256 + λ·bits·256 of a fully specified candidate against its prediction
    fn cost(&mut self, cx: &mut Ctx, mbx: usize, mby: usize, kind: u8, qp: i32, lam: f64, m: &Mb, lp: &[u8; 256], cp: &[[u8; 64]; 2]) -> f64 {
        let mut lo = [0u8; 256];
        let mut co = [[0u8; 64]; 2];
        luma_residual(m, qp, lp, &mut lo, None);
        chroma_residual(m, qp, cp, &mut co);
        let d = ssd_block(&self.src.planes[0], mbx * 16, mby * 16, 16, &lo) + self.chroma_ssd(mbx, mby, &co);
        let r = self.rate(cx, mbx, mby, kind, m);
        d as f64 * 256.0 + lam * r as f64
    }

    /// the best transform size for a prediction independent of the residual,
    /// then whether dropping the residual altogether is cheaper
    fn eval(&mut self, cx: &mut Ctx, mbx: usize, mby: usize, kind: u8, qp: i32, lam: f64, base: &Mb, lp: &[u8; 256], cp: &[[u8; 64]; 2], tsizes: &[u8]) -> (f64, Mb) {
        let mut best = (f64::MAX, base.clone());
        for &t in tsizes {
            let mut m = base.clone();
            m.tsize = t;
            self.residual(&mut m, mbx, mby, qp, lp, cp, None);
            let j = self.cost(cx, mbx, mby, kind, qp, lam, &m, lp, cp);
            if j < best.0 {
                best = (j, m);
            }
        }
        if best.1.luma.iter().any(|&x| x != 0) || best.1.chroma.iter().flatten().any(|&x| x != 0) {
            let mut z = best.1.clone();
            z.luma = [0; 256];
            z.chroma = [[0; 64]; 2];
            let j = self.cost(cx, mbx, mby, kind, qp, lam, &z, lp, cp);
            if j < best.0 {
                best = (j, z);
            }
        }
        best
    }

    fn chroma_mode(&self, mbx: usize, mby: usize) -> (u8, [[u8; 64]; 2]) {
        let mut best = (u32::MAX, 0u8, [[0u8; 64]; 2]);
        for mode in 0..intra::MODES {
            if intra::effective(mode, mbx * 8, mby * 8) != mode {
                continue;
            }
            let mut cp = [[0u8; 64]; 2];
            let mut s = 0;
            for k in 0..2 {
                intra::predict(&self.cur.planes[1 + k], mbx * 8, mby * 8, 8, mode, &mut cp[k]);
                s += me::satd(&self.src.planes[1 + k], mbx * 8, mby * 8, &cp[k], 8, 8);
            }
            if s < best.0 {
                best = (s, mode, cp);
            }
        }
        (best.1, best.2)
    }

    /// motion search of one partition against `refp`; returns the vector and its SATD + λ·bits cost
    fn motion(&self, refp: &Plane, half: &HalfPlanes, mbx: usize, mby: usize, q: usize, wb: usize, hb: usize, pmv: [i32; 2], cands: &[[i32; 2]], lam16: u32) -> ([i32; 2], u32) {
        let s = Search { src: &self.src.planes[0], refp, x: mbx * 16 + (q % 2) * 8, y: mby * 16 + (q / 2) * 8, w: wb * 8, h: hb * 8, pmv, lambda16: lam16, half: Some(half) };
        let (mv, _) = s.hexagon(cands, self.cfg.range.max(16) * 2);
        s.subpel(mv)
    }

    #[allow(clippy::too_many_arguments)]
    fn decide(&mut self, cx: &mut Ctx, refs: Refs, halves: &[Option<HalfPlanes>; 2], mbx: usize, mby: usize, kind: u8, qp: i32, full: Option<FullResult>, col: [i32; 2], (tb, td): (i32, i32)) -> Mb {
        let lam = lambda(qp);
        let lam16 = (lam.sqrt() * 16.0) as u32;
        let mut best = (f64::MAX, Mb::default());
        let (x0, y0) = (mbx * 16, mby * 16);
        let mut inter_satd = u32::MAX;
        let mut lp = [0u8; 256];
        let mut cp = [[0u8; 64]; 2];
        if kind != KIND_I {
            self.clear_mb(mbx, mby);
            let pmv0 = self.grid.predict_mv(0, mbx * 2, mby * 2, 2);
            let pmv1 = self.grid.predict_mv(1, mbx * 2, mby * 2, 2);
            // skip: P uses the predicted vector, B bi-predicts from both predicted vectors
            let skip = if kind == KIND_B {
                let (d0, d1) = self.grid.direct(mbx, mby);
                Mb { skip: true, direct: true, part: PART_8X8, mv: d0, mv1: d1, dir: [DIR_BI; 4], ..Mb::default() }
            } else {
                Mb { skip: true, mv: [pmv0; 4], ..Mb::default() }
            };
            inter_pred(refs, mbx, mby, &skip, &mut lp, &mut cp);
            let d = ssd_block(&self.src.planes[0], x0, y0, 16, &lp) + self.chroma_ssd(mbx, mby, &cp);
            let r = self.rate(cx, mbx, mby, kind, &skip);
            best = (d as f64 * 256.0 + lam * r as f64, skip.clone());
            if kind == KIND_B {
                // direct prediction with a residual (same prediction as skip)
                let m = Mb { skip: false, ..skip };
                let (j, cand) = self.eval(cx, mbx, mby, kind, qp, lam, &m, &lp, &cp, &[4, 8]);
                if j < best.0 {
                    best = (j, cand);
                }
                inter_satd = me::satd(&self.src.planes[0], x0, y0, &lp, 16, 16);
            }

            let r0 = &refs[0].unwrap().planes[0];
            let h0 = halves[0].as_ref().unwrap();
            let mut cands = vec![pmv0, [0, 0], col];
            if let Some(f) = full {
                cands.push([f.mv16[0] * 4, f.mv16[1] * 4]);
            }
            if kind == KIND_B {
                cands.push(scale_mv(col, tb, td));
            }
            let (mv16, c16) = self.motion(r0, h0, mbx, mby, 0, 2, 2, pmv0, &cands, lam16);
            let mut tries: Vec<Mb> = vec![Mb { part: PART_16X16, mv: [mv16; 4], ..Mb::default() }];
            inter_satd = inter_satd.min(c16);
            if kind == KIND_P {
                // split partitions: tried in full only when their motion cost beats 16×16
                let mut mv8 = [[0i32; 2]; 4];
                let mut c8 = 0;
                for q in 0..4 {
                    let mut c = vec![mv16, pmv0];
                    if let Some(f) = full {
                        c.push([f.mv8[q][0] * 4, f.mv8[q][1] * 4]);
                    }
                    let (v, k) = self.motion(r0, h0, mbx, mby, q, 1, 1, pmv0, &c, lam16);
                    mv8[q] = v;
                    c8 += k;
                }
                for (part, a, b, wb, hb) in [(PART_16X8, 0, 2, 2, 1), (PART_8X16, 0, 1, 1, 2)] {
                    let (ma, ca) = self.motion(r0, h0, mbx, mby, a, wb, hb, pmv0, &[mv16, mv8[a]], lam16);
                    let (mb_, cb) = self.motion(r0, h0, mbx, mby, b, wb, hb, pmv0, &[mv16, mv8[b]], lam16);
                    if ma != mb_ && ca + cb < c16 {
                        let mut mv = [ma; 4];
                        for q in 0..4 {
                            if (part == PART_16X8 && q >= 2) || (part == PART_8X16 && q % 2 == 1) {
                                mv[q] = mb_;
                            }
                        }
                        tries.push(Mb { part, mv, ..Mb::default() });
                        inter_satd = inter_satd.min(ca + cb);
                    }
                }
                if c8 < c16 && mv8.iter().any(|v| *v != mv8[0]) {
                    tries.push(Mb { part: PART_8X8, mv: mv8, ..Mb::default() });
                    inter_satd = inter_satd.min(c8);
                }
            } else {
                // B: the future reference, and both averaged
                let r1 = &refs[1].unwrap().planes[0];
                let h1 = halves[1].as_ref().unwrap();
                let mut c1 = vec![pmv1, [0, 0], scale_mv(col, tb - td, td)];
                c1.push([-mv16[0], -mv16[1]]);
                let (mv1, k1) = self.motion(r1, h1, mbx, mby, 0, 2, 2, pmv1, &c1, lam16);
                tries.push(Mb { part: PART_16X16, mv1: [mv1; 4], dir: [DIR_L1; 4], ..Mb::default() });
                tries.push(Mb { part: PART_16X16, mv: [mv16; 4], mv1: [mv1; 4], dir: [DIR_BI; 4], ..Mb::default() });
                inter_satd = inter_satd.min(k1);
                tries[0].dir = [DIR_L0; 4];
            }
            for m in tries {
                inter_pred(refs, mbx, mby, &m, &mut lp, &mut cp);
                if kind == KIND_B && m.dir[0] == DIR_BI {
                    inter_satd = inter_satd.min(me::satd(&self.src.planes[0], x0, y0, &lp, 16, 16));
                }
                let ts: &[u8] = if m.part == PART_8X8 { &[4, 8] } else { &[4, 8, 16] };
                let (j, cand) = self.eval(cx, mbx, mby, kind, qp, lam, &m, &lp, &cp, ts);
                if j < best.0 {
                    best = (j, cand);
                }
            }
        }
        // intra 16×16: the mode with the lowest SATD
        let mut i16best = (u32::MAX, 0u8, [0u8; 256]);
        for mode in 0..intra::MODES {
            if intra::effective(mode, x0, y0) != mode {
                continue;
            }
            let mut p = [0u8; 256];
            intra::predict(&self.cur.planes[0], x0, y0, 16, mode, &mut p);
            let s = me::satd(&self.src.planes[0], x0, y0, &p, 16, 16);
            if s < i16best.0 {
                i16best = (s, mode, p);
            }
        }
        // in P and B frames intra is tried in full only when it looks competitive
        if kind != KIND_I && i16best.0 > inter_satd + inter_satd / 4 {
            return best.1;
        }
        let (cmode, cpi) = self.chroma_mode(mbx, mby);
        let m = Mb { intra: true, i16: true, lmode: [i16best.1; 4], cmode, ..Mb::default() };
        let (j, cand) = self.eval(cx, mbx, mby, kind, qp, lam, &m, &i16best.2, &cpi, &[4, 8, 16]);
        if j < best.0 {
            best = (j, cand);
        }
        // intra 8×8: block by block, each predicted from the reconstruction of the previous ones
        for t in [4u8, 8] {
            let mut m = Mb { intra: true, i16: false, cmode, tsize: t, ..Mb::default() };
            let mut lp = [0u8; 256];
            let mut lo = [0u8; 256];
            for q in 0..4 {
                let (qx, qy) = (x0 + (q % 2) * 8, y0 + (q / 2) * 8);
                let mut bm = (u32::MAX, 0u8, [0u8; 64]);
                for mode in 0..intra::MODES {
                    if intra::effective(mode, qx, qy) != mode {
                        continue;
                    }
                    let mut p8 = [0u8; 64];
                    intra::predict(&self.cur.planes[0], qx, qy, 8, mode, &mut p8);
                    let s = me::satd(&self.src.planes[0], qx, qy, &p8, 8, 8);
                    if s < bm.0 {
                        bm = (s, mode, p8);
                    }
                }
                m.lmode[q] = bm.1;
                let (ox, oy) = ((q % 2) * 8, (q / 2) * 8);
                for r in 0..8 {
                    lp[(oy + r) * 16 + ox..(oy + r) * 16 + ox + 8].copy_from_slice(&bm.2[r * 8..r * 8 + 8]);
                }
                self.residual(&mut m, mbx, mby, qp, &lp, &cpi, Some(q));
                luma_residual(&m, qp, &lp, &mut lo, Some(q));
                let p = &mut self.cur.planes[0];
                for r in 0..8 {
                    let i = p.idx(qx as isize, (qy + r) as isize);
                    p.data[i..i + 8].copy_from_slice(&lo[(oy + r) * 16 + ox..(oy + r) * 16 + ox + 8]);
                }
            }
            let mut tmp = m.clone();
            self.residual(&mut tmp, mbx, mby, qp, &lp, &cpi, None);
            m.chroma = tmp.chroma;
            let j = self.cost(cx, mbx, mby, kind, qp, lam, &m, &lp, &cpi);
            if j < best.0 {
                best = (j, m);
            }
        }
        best.1
    }
}

/// encodes a whole sequence; returns the stream bytes and the per-frame stats in decode order
pub fn encode_all(cfg: Config, info: StreamInfo, fps: f64, frames: &[Vec<u8>]) -> (Vec<u8>, Vec<FrameStats>) {
    let mut out = Vec::new();
    crate::bitstream::write_header(&mut out, &info);
    let mut e = Encoder::new(cfg, info, fps);
    let mut stats = vec![];
    let mut take = |coded: Vec<Coded>, out: &mut Vec<u8>| {
        for c in coded {
            out.extend_from_slice(&c.packet);
            stats.push(c.stats);
        }
    };
    for f in frames {
        let c = e.push(f);
        take(c, &mut out);
    }
    let c = e.flush();
    take(c, &mut out);
    (out, stats)
}
