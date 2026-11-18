//! Rate control: constant QP, or one-pass average bitrate with a per-frame-type
//! complexity model (bits × Qstep ≈ constant) and a virtual buffer that pulls
//! the running rate back to the target.

#[derive(Clone, Copy, Debug)]
pub enum Mode {
    /// constant QP for P frames; I frames use QP − 3, B frames QP + 4
    Cqp(i32),
    /// average bitrate in bits per second
    Abr(f64),
}

pub struct RateControl {
    mode: Mode,
    fps: f64,
    pixels: f64,
    /// complexity estimates (bits · Qstep) for I, P and B frames
    cplx: [Option<f64>; 3],
    last_qp: [Option<i32>; 3],
    /// bits produced minus bits budgeted so far
    pub debt: f64,
}

pub fn qstep(qp: i32) -> f64 {
    2f64.powf((qp as f64 - 4.0) / 6.0)
}

fn qp_of(step: f64) -> i32 {
    (4.0 + 6.0 * step.max(1e-3).log2()).round() as i32
}

impl RateControl {
    pub fn new(mode: Mode, fps: f64, w: usize, h: usize) -> Self {
        RateControl { mode, fps, pixels: (w * h) as f64, cplx: [None; 3], last_qp: [None; 3], debt: 0.0 }
    }

    /// `kind`: 0 I, 1 P, 2 B
    pub fn qp(&self, kind: u8) -> i32 {
        let t = kind as usize;
        let intra = kind == 0;
        match self.mode {
            Mode::Cqp(q) => [q - 3, q, q + 4][t].clamp(0, 51),
            Mode::Abr(bps) => {
                let per_frame = bps / self.fps;
                // spread the accumulated error over about a second of frames
                let base = (per_frame - self.debt / self.fps.max(1.0)).max(per_frame * 0.25);
                // relative budgets of the frame kinds
                let want = base * [4.0, 1.3, 0.5][t];
                let qp = match self.cplx[t] {
                    Some(c) => qp_of(c / want),
                    None => match self.cplx[0] {
                        // a first P or B frame: assume a fraction of the I-frame complexity
                        Some(c) if !intra => qp_of(c / [1.0, 4.0, 8.0][t] / want),
                        _ => {
                            let bpp = per_frame / self.pixels;
                            (46.0 - 6.0 * (bpp / 0.02).log2()).round() as i32
                        }
                    },
                };
                let qp = match self.last_qp[t] {
                    Some(l) => qp.clamp(l - 3, l + 3),
                    None => qp,
                };
                qp.clamp(1, 51)
            }
        }
    }

    pub fn update(&mut self, kind: u8, qp: i32, bits: f64) {
        let t = kind as usize;
        let c = bits * qstep(qp);
        self.cplx[t] = Some(match self.cplx[t] {
            Some(old) => 0.5 * old + 0.5 * c,
            None => c,
        });
        self.last_qp[t] = Some(qp);
        if let Mode::Abr(bps) = self.mode {
            self.debt += bits - bps / self.fps;
        }
    }
}
