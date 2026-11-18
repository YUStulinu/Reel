//! Pictures: three 8-bit planes (4:2:0) with a replicated border, so motion
//! compensation can read outside the picture without bounds checks.

#[derive(Clone)]
pub struct Plane {
    pub w: usize,
    pub h: usize,
    pub pad: usize,
    pub stride: usize,
    pub data: Vec<u8>,
}

impl Plane {
    pub fn new(w: usize, h: usize, pad: usize) -> Self {
        let stride = w + 2 * pad;
        Plane { w, h, pad, stride, data: vec![0; stride * (h + 2 * pad)] }
    }
    #[inline]
    pub fn idx(&self, x: isize, y: isize) -> usize {
        ((y + self.pad as isize) as usize) * self.stride + (x + self.pad as isize) as usize
    }
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.data[(y + self.pad) * self.stride + x + self.pad]
    }
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: u8) {
        let i = (y + self.pad) * self.stride + x + self.pad;
        self.data[i] = v;
    }
    /// a clamped read: any coordinate, the nearest edge pixel outside the border
    #[inline]
    pub fn clamped(&self, x: isize, y: isize) -> u8 {
        let p = self.pad as isize;
        let x = x.clamp(-p, self.w as isize + p - 1);
        let y = y.clamp(-p, self.h as isize + p - 1);
        self.data[self.idx(x, y)]
    }
    /// fills the border by replicating the edge pixels
    pub fn extend_edges(&mut self) {
        let (w, h, p, s) = (self.w, self.h, self.pad, self.stride);
        for y in 0..h {
            let row = (y + p) * s;
            let l = self.data[row + p];
            let r = self.data[row + p + w - 1];
            self.data[row..row + p].fill(l);
            self.data[row + p + w..row + s].fill(r);
        }
        for y in 0..p {
            let (top, bot) = (p * s, (p + h - 1) * s);
            self.data.copy_within(top..top + s, y * s);
            self.data.copy_within(bot..bot + s, (p + h + y) * s);
        }
    }
    pub fn row(&self, y: usize) -> &[u8] {
        let i = (y + self.pad) * self.stride + self.pad;
        &self.data[i..i + self.w]
    }
    pub fn row_mut(&mut self, y: usize) -> &mut [u8] {
        let i = (y + self.pad) * self.stride + self.pad;
        &mut self.data[i..i + self.w]
    }
}

pub const LUMA_PAD: usize = 80;
pub const CHROMA_PAD: usize = 40;

#[derive(Clone)]
pub struct Frame {
    pub planes: [Plane; 3],
}

impl Frame {
    /// a picture of w×h luma pixels (both multiples of 16)
    pub fn new(w: usize, h: usize) -> Self {
        Frame { planes: [Plane::new(w, h, LUMA_PAD), Plane::new(w / 2, h / 2, CHROMA_PAD), Plane::new(w / 2, h / 2, CHROMA_PAD)] }
    }
    pub fn w(&self) -> usize {
        self.planes[0].w
    }
    pub fn h(&self) -> usize {
        self.planes[0].h
    }
    pub fn extend_edges(&mut self) {
        for p in &mut self.planes {
            p.extend_edges();
        }
    }
    /// copies a w×h 4:2:0 picture in (padding to the frame size by edge replication)
    pub fn load(&mut self, w: usize, h: usize, yuv: &[u8]) {
        let (cw, ch) = ((w + 1) / 2, (h + 1) / 2);
        let planes = [(0, w, h), (w * h, cw, ch), (w * h + cw * ch, cw, ch)];
        for (k, &(off, pw, ph)) in planes.iter().enumerate() {
            let p = &mut self.planes[k];
            for y in 0..p.h {
                let sy = y.min(ph - 1);
                for x in 0..p.w {
                    let sx = x.min(pw - 1);
                    p.set(x, y, yuv[off + sy * pw + sx]);
                }
            }
        }
    }
    /// the top-left w×h of the picture as packed 4:2:0
    pub fn store(&self, w: usize, h: usize) -> Vec<u8> {
        let (cw, ch) = ((w + 1) / 2, (h + 1) / 2);
        let mut out = Vec::with_capacity(w * h + 2 * cw * ch);
        for (k, (pw, ph)) in [(w, h), (cw, ch), (cw, ch)].into_iter().enumerate() {
            for y in 0..ph {
                out.extend_from_slice(&self.planes[k].row(y)[..pw]);
            }
        }
        out
    }
}

/// rounds up to a multiple of 16
pub fn align16(v: usize) -> usize {
    (v + 15) & !15
}
