//! The decoder: entropy decoding of every macroblock, reconstruction, then the
//! deblocking filter.
//!
//! Frames arrive in decode order. I and P frames are anchors (references);
//! B frames sit between two anchors in display order and are decoded after
//! both. So a B frame is displayed as soon as it is decoded, and an anchor is
//! displayed when the next anchor is decoded (or at `flush`).
use crate::bitstream::{StreamInfo, TYPE_B, TYPE_I, TYPE_P};
use crate::deblock::deblock;
use crate::frame::{align16, Frame};
use crate::range;
use crate::recon::reconstruct;
use crate::syntax::{self, Ctx, Grid, Mb, MbState, KIND_B, KIND_I, KIND_P};

pub struct Decoder {
    pub info: StreamInfo,
    cur: Frame,
    /// the older and the newer anchor
    past: Option<Frame>,
    last: Option<Frame>,
    /// the newer anchor has not been displayed yet
    held: bool,
    grid: Grid,
}

impl Decoder {
    pub fn new(info: StreamInfo) -> Self {
        let (w, h) = (align16(info.w), align16(info.h));
        Decoder { info, cur: Frame::new(w, h), past: None, last: None, held: false, grid: Grid::new(w / 16, h / 16) }
    }

    /// decodes one packet (type byte, qp byte, payload); returns the picture
    /// to display now, if any
    pub fn decode(&mut self, packet: &[u8]) -> Result<Option<&Frame>, String> {
        if packet.len() < 2 {
            return Err("empty packet".into());
        }
        let (ftype, qp) = (packet[0], packet[1] as i32);
        if qp > crate::tables::MAX_QP {
            return Err(format!("bad qp {qp}"));
        }
        let kind = match ftype {
            TYPE_I => KIND_I,
            TYPE_P if self.last.is_some() => KIND_P,
            TYPE_B if self.past.is_some() && self.last.is_some() => KIND_B,
            TYPE_P | TYPE_B => return Err("a predicted frame without its reference".into()),
            t => return Err(format!("unknown frame type {t}")),
        };
        let refs = match kind {
            KIND_P => [self.last.as_ref(), None],
            KIND_B => [self.past.as_ref(), self.last.as_ref()],
            _ => [None, None],
        };
        let mut rd = range::Decoder::new(&packet[2..]);
        let mut cx = Ctx::default();
        self.grid.reset();
        syntax::frame_header(&mut rd, kind, &mut self.grid);
        let (mbw, mbh) = (self.grid.mbw, self.grid.mbh);
        let mut mb = Mb::default();
        for mby in 0..mbh {
            for mbx in 0..mbw {
                syntax::macroblock(&mut rd, &mut cx, &mut self.grid, mbx, mby, kind, &mut mb);
                if rd.overrun() {
                    return Err(format!("stream ends inside macroblock {mbx},{mby}"));
                }
                self.grid.mbs[mby * mbw + mbx] = MbState { skip: mb.skip, intra: mb.intra, cbp: mb.cbp, tsize: mb.tsize, nz: syntax::nz4_of(&mb) };
                reconstruct(&mut self.cur, refs, mbx, mby, &mb, qp);
            }
        }
        deblock(&mut self.cur, &self.grid, qp);
        if kind == KIND_B {
            return Ok(Some(&self.cur));
        }
        self.cur.extend_edges();
        self.grid.keep_as_anchor();
        // rotate: the new anchor becomes `last`, the old `last` becomes `past`,
        // and the old `past` (no longer referenced) is reused for decoding
        let spare = self.past.take().unwrap_or_else(|| self.cur.clone());
        let new = std::mem::replace(&mut self.cur, spare);
        self.past = self.last.take();
        self.last = Some(new);
        let show = self.held;
        self.held = true;
        Ok(if show { self.past.as_ref() } else { None })
    }

    /// the anchor still waiting to be displayed (call at the end of a stream or segment)
    pub fn flush(&mut self) -> Option<&Frame> {
        if !self.held {
            return None;
        }
        self.held = false;
        self.last.as_ref()
    }

    /// the visible part of a picture as packed 4:2:0
    pub fn visible(&self, f: &Frame) -> Vec<u8> {
        f.store(self.info.w, self.info.h)
    }

    /// decodes a whole stream, returning the pictures in display order
    pub fn decode_stream(data: &[u8]) -> Result<Vec<Vec<u8>>, String> {
        let info = crate::bitstream::read_header(data)?;
        let mut d = Decoder::new(info);
        let mut out = vec![];
        for (off, len) in crate::bitstream::packets(data)? {
            if let Some(f) = d.decode(&data[off..off + len])? {
                out.push(f.store(info.w, info.h));
            }
        }
        if let Some(f) = d.flush() {
            out.push(f.store(info.w, info.h));
        }
        Ok(out)
    }
}
