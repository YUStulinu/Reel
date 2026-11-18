//! Macroblock syntax. Every element is written once, generically over
//! `Coder`: the encoder passes values in, the decoder gets them out, and the
//! estimator counts their bits for rate-distortion decisions.
use crate::range::{Coder, Prob};
use crate::tables::scans;

pub const PART_16X16: u8 = 0;
pub const PART_16X8: u8 = 1;
pub const PART_8X16: u8 = 2;
pub const PART_8X8: u8 = 3;

/// frame kinds
pub const KIND_I: u8 = 0;
pub const KIND_P: u8 = 1;
pub const KIND_B: u8 = 2;

/// prediction directions of an inter partition (bit 0: list 0 = past, bit 1: list 1 = future)
pub const DIR_L0: u8 = 1;
pub const DIR_L1: u8 = 2;
pub const DIR_BI: u8 = 3;

/// the syntax elements of one 16×16 macroblock
#[derive(Clone)]
pub struct Mb {
    pub skip: bool,
    pub intra: bool,
    /// intra 16×16 prediction (else four 8×8 blocks)
    pub i16: bool,
    pub lmode: [u8; 4],
    pub cmode: u8,
    pub part: u8,
    /// list-0 motion vectors (quarter pel) of the four 8×8 quadrants, raster order
    pub mv: [[i32; 2]; 4],
    /// list-1 vectors (B frames)
    pub mv1: [[i32; 2]; 4],
    /// prediction direction of each quadrant (DIR_*)
    pub dir: [u8; 4],
    /// B frames: temporal direct prediction (vectors derived from the future anchor)
    pub direct: bool,
    /// luma transform size: 4, 8 or 16 (chroma uses 4, or 8 when luma is 8 or 16)
    pub tsize: u8,
    /// bits 0-3: luma quadrants with coefficients; bit 4: U; bit 5: V
    pub cbp: u8,
    /// luma levels, transform blocks in quadrant order, each n×n raster
    pub luma: [i32; 256],
    pub chroma: [[i32; 64]; 2],
}

impl Default for Mb {
    fn default() -> Self {
        Mb { skip: false, intra: false, i16: true, lmode: [0; 4], cmode: 0, part: 0, mv: [[0; 2]; 4], mv1: [[0; 2]; 4], dir: [DIR_L0; 4], direct: false, tsize: 8, cbp: 0, luma: [0; 256], chroma: [[0; 64]; 2] }
    }
}

/// the partitions of a mode: (first quadrant, width, height) in 8×8 units
pub fn partitions(part: u8) -> &'static [(usize, usize, usize)] {
    match part {
        PART_16X16 => &[(0, 2, 2)],
        PART_16X8 => &[(0, 2, 1), (2, 2, 1)],
        PART_8X16 => &[(0, 1, 2), (1, 1, 2)],
        _ => &[(0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1)],
    }
}

pub fn chroma_tsize(tsize: u8) -> usize {
    if tsize == 4 {
        4
    } else {
        8
    }
}

/// what later macroblocks see of an earlier one (contexts, prediction, deblocking)
#[derive(Clone, Copy, Default)]
pub struct MbState {
    pub skip: bool,
    pub intra: bool,
    pub cbp: u8,
    pub tsize: u8,
    /// luma 4×4 blocks with nonzero coefficients, bit = row*4 + col
    pub nz: u16,
}

/// per-frame neighbour information
pub struct Grid {
    pub mbw: usize,
    pub mbh: usize,
    pub mbs: Vec<MbState>,
    /// motion vectors on the 8×8 grid, per list
    pub mv: [Vec<[i32; 2]>; 2],
    /// the 8×8 block uses that list
    pub has_mv: [Vec<bool>; 2],
    /// prediction direction per 8×8 block (0: intra)
    pub dir: Vec<u8>,
    /// the 8×8 block is already decoded in this frame
    pub done: Vec<bool>,
    /// B frames: the list-0 vectors of the future anchor (8×8 grid, intra = 0)
    /// and the temporal distances past→B (tb) and past→future (td)
    pub col: Vec<[i32; 2]>,
    pub tb: i32,
    pub td: i32,
}

impl Grid {
    pub fn new(mbw: usize, mbh: usize) -> Self {
        let n8 = mbw * mbh * 4;
        Grid {
            mbw,
            mbh,
            mbs: vec![MbState::default(); mbw * mbh],
            mv: [vec![[0; 2]; n8], vec![[0; 2]; n8]],
            has_mv: [vec![false; n8], vec![false; n8]],
            dir: vec![0; n8],
            done: vec![false; n8],
            col: vec![[0; 2]; n8],
            tb: 1,
            td: 2,
        }
    }
    pub fn reset(&mut self) {
        self.mbs.fill(MbState::default());
        for l in 0..2 {
            self.mv[l].fill([0; 2]);
            self.has_mv[l].fill(false);
        }
        self.dir.fill(0);
        self.done.fill(false);
    }
    fn b8(&self, bx: isize, by: isize) -> Option<usize> {
        let w = self.mbw as isize * 2;
        if bx < 0 || by < 0 || bx >= w || by >= self.mbh as isize * 2 {
            return None;
        }
        Some((by * w + bx) as usize)
    }
    /// the median motion vector predictor of a partition whose top-left 8×8
    /// block is (bx, by) and which is `wb` blocks wide
    pub fn predict_mv(&self, list: usize, bx: usize, by: usize, wb: usize) -> [i32; 2] {
        let (bx, by) = (bx as isize, by as isize);
        let get = |x: isize, y: isize| -> Option<[i32; 2]> {
            let i = self.b8(x, y)?;
            if !self.done[i] {
                return None;
            }
            Some(if self.has_mv[list][i] { self.mv[list][i] } else { [0, 0] })
        };
        let a = get(bx - 1, by);
        let b = get(bx, by - 1);
        let c = get(bx + wb as isize, by - 1).or_else(|| get(bx - 1, by - 1));
        let avail: Vec<[i32; 2]> = [a, b, c].iter().flatten().copied().collect();
        match avail.len() {
            0 => [0, 0],
            1 => avail[0],
            _ => {
                let (a, b, c) = (a.unwrap_or([0, 0]), b.unwrap_or([0, 0]), c.unwrap_or([0, 0]));
                let med = |x: i32, y: i32, z: i32| x.max(y).min(x.min(y).max(z));
                [med(a[0], b[0], c[0]), med(a[1], b[1], c[1])]
            }
        }
    }
    /// records an MB's quadrant vectors (intra: none)
    pub fn set_mvs(&mut self, mbx: usize, mby: usize, mb: &Mb, quadrants: &[usize]) {
        for &q in quadrants {
            let i = self.b8((mbx * 2 + q % 2) as isize, (mby * 2 + q / 2) as isize).unwrap();
            self.mv[0][i] = mb.mv[q];
            self.mv[1][i] = mb.mv1[q];
            self.has_mv[0][i] = !mb.intra && mb.dir[q] & DIR_L0 != 0;
            self.has_mv[1][i] = !mb.intra && mb.dir[q] & DIR_L1 != 0;
            self.dir[i] = if mb.intra { 0 } else { mb.dir[q] };
            self.done[i] = true;
        }
    }
    /// temporal direct vectors of a macroblock's quadrants: the co-located
    /// anchor vector scaled to the past (list 0) and future (list 1) pictures
    pub fn direct(&self, mbx: usize, mby: usize) -> ([[i32; 2]; 4], [[i32; 2]; 4]) {
        let w8 = self.mbw * 2;
        let mut l0 = [[0; 2]; 4];
        let mut l1 = [[0; 2]; 4];
        let td = self.td.max(1);
        for q in 0..4 {
            let c = self.col[(mby * 2 + q / 2) * w8 + mbx * 2 + q % 2];
            for k in 0..2 {
                l0[q][k] = (c[k] * self.tb) / td;
                l1[q][k] = l0[q][k] - c[k];
            }
        }
        (l0, l1)
    }
    /// keeps this frame's list-0 vectors as the co-located field of later B frames
    pub fn keep_as_anchor(&mut self) {
        for i in 0..self.col.len() {
            self.col[i] = if self.has_mv[0][i] { self.mv[0][i] } else { [0, 0] };
        }
    }
    pub fn mb(&self, mbx: isize, mby: isize) -> Option<&MbState> {
        if mbx < 0 || mby < 0 || mbx >= self.mbw as isize || mby >= self.mbh as isize {
            return None;
        }
        Some(&self.mbs[mby as usize * self.mbw + mbx as usize])
    }
}

#[derive(Clone)]
pub struct CoefCtx {
    pub sig: [[Prob; 16]; 3],
    pub last: [[Prob; 16]; 3],
    pub gt1: [Prob; 5],
    pub lvl: [Prob; 5],
}

impl Default for CoefCtx {
    fn default() -> Self {
        CoefCtx { sig: [[Prob::default(); 16]; 3], last: [[Prob::default(); 16]; 3], gt1: [Prob::default(); 5], lvl: [Prob::default(); 5] }
    }
}

/// all adaptive models; reset at the start of every frame
#[derive(Clone)]
pub struct Ctx {
    pub skip: [Prob; 3],
    pub intra: [Prob; 3],
    pub i16: [Prob; 2],
    pub lmode: [Prob; 3],
    pub cmode: [Prob; 3],
    pub part: [Prob; 3],
    pub tsize: [Prob; 4],
    pub mvd: [[Prob; 10]; 2],
    pub bdir: [Prob; 2],
    pub direct: [Prob; 3],
    pub cbp: [Prob; 4],
    pub cbpc: [Prob; 6],
    pub coded4: [[Prob; 3]; 2],
    /// luma, chroma
    pub coef: [CoefCtx; 2],
}

impl Default for Ctx {
    fn default() -> Self {
        let p = Prob::default();
        Ctx {
            skip: [p; 3],
            intra: [p; 3],
            i16: [p; 2],
            lmode: [p; 3],
            cmode: [p; 3],
            part: [p; 3],
            tsize: [p; 4],
            mvd: [[p; 10]; 2],
            bdir: [p; 2],
            direct: [p; 3],
            cbp: [p; 4],
            cbpc: [p; 6],
            coded4: [[p; 3]; 2],
            coef: [CoefCtx::default(), CoefCtx::default()],
        }
    }
}

/// k-th order Exp-Golomb, bypass coded
pub fn exp_golomb<C: Coder>(c: &mut C, k: u32, v: &mut u32) {
    if c.decoding() {
        let mut k = k;
        let mut base = 0u32;
        while c.bypass(false) {
            base += 1 << k;
            k += 1;
            if k > 24 {
                break; // a corrupt stream
            }
        }
        *v = base + c.bypass_bits(k, 0);
    } else {
        let mut k = k;
        let mut x = *v;
        while x >= 1 << k {
            c.bypass(true);
            x -= 1 << k;
            k += 1;
        }
        c.bypass(false);
        c.bypass_bits(k, x);
    }
}

fn two_bits<C: Coder>(c: &mut C, p: &mut [Prob; 3], v: &mut u8) {
    let hi = c.bit(&mut p[0], *v & 2 != 0) as u8;
    let lo = c.bit(&mut p[1 + hi as usize], *v & 1 != 0) as u8;
    *v = hi << 1 | lo;
}

fn mvd<C: Coder>(c: &mut C, p: &mut [Prob; 10], v: &mut i32) {
    if !c.bit(&mut p[0], *v != 0) {
        *v = 0;
        return;
    }
    let mut m = v.unsigned_abs() - 1;
    let mut u = 0u32;
    while u < 8 && c.bit(&mut p[1 + u as usize], m > u) {
        u += 1;
    }
    if u == 8 {
        let mut rest = m.saturating_sub(8);
        exp_golomb(c, 3, &mut rest);
        m = 8 + rest;
    } else {
        m = u;
    }
    let neg = c.bypass(*v < 0);
    let mag = (m + 1).min(1 << 20) as i32;
    *v = if neg { -mag } else { mag };
}

fn size_class(n: usize) -> usize {
    match n {
        4 => 0,
        8 => 1,
        _ => 2,
    }
}

/// codes one transform block that has at least one nonzero level
pub fn block<C: Coder>(c: &mut C, cx: &mut CoefCtx, n: usize, lv: &mut [i32]) {
    let s = scans();
    let scan: &[u16] = match n {
        4 => &s.z4,
        8 => &s.z8,
        _ => &s.z16,
    };
    let total = n * n;
    let sc = size_class(n);
    let cls = |i: usize| -> usize {
        match n {
            4 => i.min(15),
            8 => i / 4,
            _ => i / 16,
        }
    };
    let dec = c.decoding();
    let last = if dec { 0 } else { (0..total).rev().find(|&i| lv[scan[i] as usize] != 0).unwrap_or(0) };
    let mut sig_pos = [0u16; 256];
    let mut ns = 0;
    let mut i = 0;
    loop {
        if i == total - 1 {
            sig_pos[ns] = i as u16;
            ns += 1;
            break;
        }
        let k = cls(i);
        let sig = c.bit(&mut cx.sig[sc][k], !dec && lv[scan[i] as usize] != 0);
        if sig {
            sig_pos[ns] = i as u16;
            ns += 1;
            if c.bit(&mut cx.last[sc][k], i == last) {
                break;
            }
        }
        i += 1;
    }
    if dec {
        lv[..total].fill(0);
    }
    let (mut eq1, mut gt1) = (0usize, 0usize);
    for &p in sig_pos[..ns].iter().rev() {
        let r = scan[p as usize] as usize;
        let a = lv[r].unsigned_abs();
        let g = c.bit(&mut cx.gt1[if gt1 > 0 { 0 } else { (1 + eq1).min(4) }], a > 1);
        let mut abs = 1u32;
        if g {
            let ctx = &mut cx.lvl[gt1.min(4)];
            let mut u = 0u32;
            while u < 13 && c.bit(ctx, a.saturating_sub(2) > u) {
                u += 1;
            }
            let mut rest = a.saturating_sub(15);
            if u == 13 {
                exp_golomb(c, 0, &mut rest);
                abs = 15 + rest;
            } else {
                abs = 2 + u;
            }
            gt1 += 1;
        } else {
            eq1 += 1;
        }
        let neg = c.bypass(lv[r] < 0);
        let abs = abs.min(1 << 15) as i32;
        lv[r] = if neg { -abs } else { abs };
    }
}

/// luma transform blocks: (x, y, levels offset) inside the macroblock, quadrant order
pub fn luma_blocks(tsize: u8) -> Vec<(usize, usize, usize)> {
    match tsize {
        16 => vec![(0, 0, 0)],
        8 => (0..4).map(|q| ((q % 2) * 8, (q / 2) * 8, q * 64)).collect(),
        _ => (0..16).map(|t| {
            let (q, s) = (t / 4, t % 4);
            ((q % 2) * 8 + (s % 2) * 4, (q / 2) * 8 + (s / 2) * 4, t * 16)
        }).collect(),
    }
}

pub fn nz4_of(mb: &Mb) -> u16 {
    let mut nz = 0u16;
    for (x, y, off) in luma_blocks(mb.tsize) {
        let n = mb.tsize as usize;
        if mb.luma[off..off + n * n].iter().any(|&v| v != 0) {
            for yy in (y..y + n).step_by(4) {
                for xx in (x..x + n).step_by(4) {
                    nz |= 1 << ((yy / 4) * 4 + xx / 4);
                }
            }
        }
    }
    nz
}

/// the frame header inside the entropy-coded payload: B frames carry the
/// temporal distances used by direct prediction
pub fn frame_header<C: Coder>(c: &mut C, kind: u8, g: &mut Grid) {
    if kind == KIND_B {
        let tb = c.bypass_bits(4, g.tb as u32);
        let td = c.bypass_bits(4, g.td as u32);
        g.tb = tb as i32;
        g.td = (td as i32).max(1);
    }
}

/// codes a whole macroblock. `p_frame` selects the skip/intra flags; the motion
/// vector grid is updated as partitions are coded (prediction needs it).
pub fn macroblock<C: Coder>(c: &mut C, cx: &mut Ctx, g: &mut Grid, mbx: usize, mby: usize, kind: u8, mb: &mut Mb) {
    let dec = c.decoding();
    let left = g.mb(mbx as isize - 1, mby as isize).copied();
    let top = g.mb(mbx as isize, mby as isize - 1).copied();
    if kind != KIND_I {
        let k = left.map_or(0, |s| s.skip as usize) + top.map_or(0, |s| s.skip as usize);
        mb.skip = c.bit(&mut cx.skip[k], mb.skip);
        if mb.skip {
            if dec {
                *mb = Mb { skip: true, tsize: 8, ..Mb::default() };
            }
            // P: the predicted vector; B: bi-prediction from both predicted vectors
            if kind == KIND_B {
                (mb.mv, mb.mv1) = g.direct(mbx, mby);
                mb.dir = [DIR_BI; 4];
                mb.direct = true;
                mb.part = PART_8X8;
            } else {
                mb.mv = [g.predict_mv(0, mbx * 2, mby * 2, 2); 4];
                mb.dir = [DIR_L0; 4];
                mb.part = PART_16X16;
            }
            mb.intra = false;
            mb.cbp = 0;
            g.set_mvs(mbx, mby, mb, &[0, 1, 2, 3]);
            return;
        }
        let k = left.map_or(0, |s| s.intra as usize) + top.map_or(0, |s| s.intra as usize);
        mb.intra = c.bit(&mut cx.intra[k], mb.intra);
    } else {
        mb.skip = false;
        mb.intra = true;
    }
    if mb.intra {
        mb.i16 = c.bit(&mut cx.i16[0], mb.i16);
        let count = if mb.i16 { 1 } else { 4 };
        for i in 0..count {
            two_bits(c, &mut cx.lmode, &mut mb.lmode[i]);
        }
        if mb.i16 {
            mb.lmode = [mb.lmode[0]; 4];
        }
        two_bits(c, &mut cx.cmode, &mut mb.cmode);
        g.set_mvs(mbx, mby, mb, &[0, 1, 2, 3]);
    } else if kind == KIND_B && {
        let k = left.map_or(0, |s| s.skip as usize) + top.map_or(0, |s| s.skip as usize);
        mb.direct = c.bit(&mut cx.direct[k], mb.direct);
        mb.direct
    } {
        // temporal direct with a residual
        (mb.mv, mb.mv1) = g.direct(mbx, mby);
        mb.dir = [DIR_BI; 4];
        mb.part = PART_8X8;
        g.set_mvs(mbx, mby, mb, &[0, 1, 2, 3]);
    } else {
        mb.direct = false;
        two_bits(c, &mut cx.part, &mut mb.part);
        for &(q, wb, hb) in partitions(mb.part) {
            if kind == KIND_B {
                let bi = c.bit(&mut cx.bdir[0], mb.dir[q] == DIR_BI);
                mb.dir[q] = if bi {
                    DIR_BI
                } else if c.bit(&mut cx.bdir[1], mb.dir[q] == DIR_L1) {
                    DIR_L1
                } else {
                    DIR_L0
                };
            } else {
                mb.dir[q] = DIR_L0;
            }
            for list in 0..2 {
                if mb.dir[q] & (1 << list) == 0 {
                    continue;
                }
                let pmv = g.predict_mv(list, mbx * 2 + q % 2, mby * 2 + q / 2, wb);
                let v = if list == 0 { &mut mb.mv[q] } else { &mut mb.mv1[q] };
                for comp in 0..2 {
                    let mut d = v[comp] - pmv[comp];
                    mvd(c, &mut cx.mvd[comp], &mut d);
                    v[comp] = pmv[comp] + d;
                }
            }
            let quads: Vec<usize> = (0..hb).flat_map(|dy| (0..wb).map(move |dx| q + dy * 2 + dx)).collect();
            for &qq in &quads {
                mb.mv[qq] = mb.mv[q];
                mb.mv1[qq] = mb.mv1[q];
                mb.dir[qq] = mb.dir[q];
            }
            g.set_mvs(mbx, mby, mb, &quads);
        }
    }
    // transform size: 16 is allowed unless the prediction is split into 8×8 blocks
    let allow16 = if mb.intra { mb.i16 } else { mb.part != PART_8X8 };
    let t4 = c.bit(&mut cx.tsize[0], mb.tsize == 4);
    mb.tsize = if t4 {
        4
    } else if allow16 && c.bit(&mut cx.tsize[1 + mb.intra as usize], mb.tsize == 16) {
        16
    } else {
        8
    };
    // coded block pattern
    if !dec {
        let n = mb.tsize as usize;
        let mut cbp = 0u8;
        for (x, y, off) in luma_blocks(mb.tsize) {
            if mb.luma[off..off + n * n].iter().any(|&v| v != 0) {
                cbp |= if n == 16 { 15 } else { 1 << ((y / 8) * 2 + x / 8) };
            }
        }
        for k in 0..2 {
            if mb.chroma[k].iter().any(|&v| v != 0) {
                cbp |= 16 << k;
            }
        }
        mb.cbp = cbp;
    }
    let mut cbp = 0u8;
    if mb.tsize == 16 {
        let k = left.map_or(0, |s| (s.cbp & 2 != 0) as usize) + top.map_or(0, |s| (s.cbp & 4 != 0) as usize) * 2;
        if c.bit(&mut cx.cbp[k], mb.cbp & 15 != 0) {
            cbp = 15;
        }
    } else {
        for q in 0..4 {
            let l = if q % 2 == 1 { cbp & (1 << (q - 1)) != 0 } else { left.map_or(false, |s| s.cbp & (1 << (q + 1)) != 0) };
            let t = if q >= 2 { cbp & (1 << (q - 2)) != 0 } else { top.map_or(false, |s| s.cbp & (1 << (q + 2)) != 0) };
            if c.bit(&mut cx.cbp[l as usize + 2 * t as usize], mb.cbp & (1 << q) != 0) {
                cbp |= 1 << q;
            }
        }
    }
    for k in 0..2 {
        let nb = left.map_or(0, |s| (s.cbp >> (4 + k) & 1) as usize) + top.map_or(0, |s| (s.cbp >> (4 + k) & 1) as usize);
        if c.bit(&mut cx.cbpc[k * 3 + nb], mb.cbp & (16 << k) != 0) {
            cbp |= 16 << k;
        }
    }
    mb.cbp = cbp;
    // luma residual
    let n = mb.tsize as usize;
    let mut nzmap = 0u16; // 4×4 coded flags inside this MB, for contexts
    for (x, y, off) in luma_blocks(mb.tsize) {
        let q = (y / 8) * 2 + x / 8;
        let lv = &mut mb.luma[off..off + n * n];
        if cbp & (1 << q) == 0 {
            if dec {
                lv.fill(0);
            }
            continue;
        }
        let coded = if n == 4 {
            // per-4×4 coded flag, context from the left and top 4×4 blocks
            let (bx, by) = (x / 4, y / 4);
            let l = if bx > 0 { nzmap >> (by * 4 + bx - 1) & 1 } else { left.map_or(0, |s| s.nz >> (by * 4 + 3) & 1) };
            let t = if by > 0 { nzmap >> ((by - 1) * 4 + bx) & 1 } else { top.map_or(0, |s| s.nz >> (12 + bx) & 1) };
            let has = lv.iter().any(|&v| v != 0);
            c.bit(&mut cx.coded4[0][(l + t) as usize], has)
        } else {
            true
        };
        if coded {
            block(c, &mut cx.coef[0], n, lv);
            nzmap |= 1 << ((y / 4) * 4 + x / 4);
        } else if dec {
            lv.fill(0);
        }
    }
    // chroma residual
    let cn = chroma_tsize(mb.tsize);
    for k in 0..2 {
        let lv = &mut mb.chroma[k];
        if cbp & (16 << k) == 0 {
            if dec {
                lv.fill(0);
            }
            continue;
        }
        if cn == 8 {
            block(c, &mut cx.coef[1], 8, &mut lv[..]);
        } else {
            let mut prev = 0usize;
            for t in 0..4 {
                let b = &mut lv[t * 16..t * 16 + 16];
                let has = b.iter().any(|&v| v != 0);
                let coded = c.bit(&mut cx.coded4[1][prev.min(2)], has);
                if coded {
                    block(c, &mut cx.coef[1], 4, b);
                    prev += 1;
                } else if dec {
                    b.fill(0);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range::{Decoder, Encoder};

    fn sample_mbs(kind: u8) -> Vec<Mb> {
        let mut v = vec![];
        let mut s = 7u64;
        let mut r = |m: u64| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 33) % m
        };
        for i in 0..60 {
            let mut mb = Mb::default();
            mb.skip = i % 7 == 3;
            mb.intra = i % 5 == 0;
            mb.i16 = r(2) == 0;
            mb.lmode = [r(4) as u8, r(4) as u8, r(4) as u8, r(4) as u8];
            if mb.i16 {
                mb.lmode = [mb.lmode[0]; 4];
            }
            mb.cmode = r(4) as u8;
            mb.part = r(4) as u8;
            for q in 0..4 {
                mb.mv[q] = [r(400) as i32 - 200, r(60) as i32 - 30];
                mb.mv1[q] = [r(100) as i32 - 50, r(40) as i32 - 20];
                mb.dir[q] = if kind == KIND_B { 1 + r(3) as u8 } else { DIR_L0 };
            }
            // vectors must agree inside each partition
            for &(q, wb, hb) in partitions(mb.part) {
                for dy in 0..hb {
                    for dx in 0..wb {
                        mb.mv[q + dy * 2 + dx] = mb.mv[q];
                        mb.mv1[q + dy * 2 + dx] = mb.mv1[q];
                        mb.dir[q + dy * 2 + dx] = mb.dir[q];
                    }
                }
            }
            let allow16 = if mb.intra { mb.i16 } else { mb.part != PART_8X8 };
            mb.tsize = [4, 8, 16][r(if allow16 { 3 } else { 2 }) as usize];
            for v in mb.luma.iter_mut() {
                *v = if r(6) == 0 { r(41) as i32 - 20 } else { 0 };
            }
            if i % 4 == 0 {
                mb.luma = [0; 256];
                mb.luma[3] = 900;
            }
            for k in 0..2 {
                for v in mb.chroma[k].iter_mut() {
                    *v = if r(9) == 0 { r(7) as i32 - 3 } else { 0 };
                }
            }
            if mb.skip {
                mb.luma = [0; 256];
                mb.chroma = [[0; 64]; 2];
            }
            v.push(mb);
        }
        v
    }

    #[test]
    fn macroblocks_round_trip() {
        for kind in [KIND_P, KIND_B] {
            round_trip(kind);
        }
    }

    fn round_trip(kind: u8) {
        let mbs = sample_mbs(kind);
        let (mbw, mbh) = (10, 6);
        let mut e = Encoder::default();
        let mut cx = Ctx::default();
        let mut g = Grid::new(mbw, mbh);
        let mut sent = vec![];
        for (i, mb) in mbs.iter().enumerate() {
            let mut m = mb.clone();
            macroblock(&mut e, &mut cx, &mut g, i % mbw, i / mbw, kind, &mut m);
            g.mbs[i] = MbState { skip: m.skip, intra: m.intra, cbp: m.cbp, tsize: m.tsize, nz: nz4_of(&m) };
            sent.push(m);
        }
        let data = e.finish();
        let mut d = Decoder::new(&data);
        let mut cx = Ctx::default();
        let mut g = Grid::new(mbw, mbh);
        for (i, want) in sent.iter().enumerate() {
            let mut m = Mb::default();
            macroblock(&mut d, &mut cx, &mut g, i % mbw, i / mbw, kind, &mut m);
            g.mbs[i] = MbState { skip: m.skip, intra: m.intra, cbp: m.cbp, tsize: m.tsize, nz: nz4_of(&m) };
            assert_eq!((m.skip, m.intra), (want.skip, want.intra), "mb {i}");
            if m.skip {
                assert_eq!((m.mv, m.dir), (want.mv, want.dir), "mb {i} skip vector");
                if kind == KIND_B {
                    assert_eq!(m.mv1, want.mv1, "mb {i} skip list-1 vector");
                }
                continue;
            }
            if m.intra {
                assert_eq!((m.i16, m.lmode, m.cmode), (want.i16, want.lmode, want.cmode), "mb {i}");
            } else {
                assert_eq!((m.part, m.dir), (want.part, want.dir), "mb {i}");
                for q in 0..4 {
                    if m.dir[q] & DIR_L0 != 0 {
                        assert_eq!(m.mv[q], want.mv[q], "mb {i} list 0");
                    }
                    if m.dir[q] & DIR_L1 != 0 {
                        assert_eq!(m.mv1[q], want.mv1[q], "mb {i} list 1");
                    }
                }
            }
            assert_eq!((m.tsize, m.cbp), (want.tsize, want.cbp), "mb {i}");
            assert_eq!(&m.luma[..], &want.luma[..], "mb {i} luma");
            assert_eq!(m.chroma, want.chroma, "mb {i} chroma");
        }
    }
}

#[cfg(test)]
mod block_tests {
    use super::*;
    use crate::range::{Decoder, Encoder};

    #[test]
    fn blocks_round_trip() {
        let mut s = 3u64;
        let mut r = |m: u64| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 33) % m
        };
        for n in [4usize, 8, 16] {
            let mut blocks = vec![];
            for _ in 0..200 {
                let mut b = vec![0i32; n * n];
                for v in b.iter_mut() {
                    *v = if r(6) == 0 { r(41) as i32 - 20 } else { 0 };
                }
                if b.iter().all(|&v| v == 0) {
                    b[r((n * n) as u64) as usize] = 1;
                }
                blocks.push(b);
            }
            let mut e = Encoder::default();
            let mut cx = CoefCtx::default();
            for b in &blocks {
                let mut c = b.clone();
                block(&mut e, &mut cx, n, &mut c);
            }
            let data = e.finish();
            let mut d = Decoder::new(&data);
            let mut cx = CoefCtx::default();
            for (i, b) in blocks.iter().enumerate() {
                let mut c = vec![0i32; n * n];
                block(&mut d, &mut cx, n, &mut c);
                assert_eq!(&c, b, "n={n} block {i}");
            }
        }
    }
}
