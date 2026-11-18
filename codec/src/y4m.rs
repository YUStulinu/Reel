//! YUV4MPEG2 (.y4m) reading and writing, 8-bit 4:2:0 only.
use std::io::{BufRead, BufReader, Read, Write};

#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub w: usize,
    pub h: usize,
    pub fps_num: u32,
    pub fps_den: u32,
}

impl Header {
    pub fn frame_bytes(&self) -> usize {
        let (cw, ch) = ((self.w + 1) / 2, (self.h + 1) / 2);
        self.w * self.h + 2 * cw * ch
    }
}

pub struct Reader<R: Read> {
    r: BufReader<R>,
    pub header: Header,
}

impl<R: Read> Reader<R> {
    pub fn new(r: R) -> Result<Self, String> {
        let mut r = BufReader::with_capacity(1 << 20, r);
        let mut line = String::new();
        r.read_line(&mut line).map_err(|e| e.to_string())?;
        let mut it = line.trim_end().split(' ');
        if it.next() != Some("YUV4MPEG2") {
            return Err("not a YUV4MPEG2 file".into());
        }
        let mut h = Header { w: 0, h: 0, fps_num: 30, fps_den: 1 };
        for tok in it {
            let (k, v) = tok.split_at(1);
            match k {
                "W" => h.w = v.parse().map_err(|_| "bad width")?,
                "H" => h.h = v.parse().map_err(|_| "bad height")?,
                "F" => {
                    let (a, b) = v.split_once(':').ok_or("bad frame rate")?;
                    h.fps_num = a.parse().map_err(|_| "bad frame rate")?;
                    h.fps_den = b.parse().map_err(|_| "bad frame rate")?;
                }
                "C" if !v.starts_with("420") => return Err(format!("only 4:2:0 is supported, not C{v}")),
                _ => {}
            }
        }
        if h.w == 0 || h.h == 0 {
            return Err("missing size".into());
        }
        Ok(Reader { r, header: h })
    }

    /// the next picture as packed 4:2:0, or None at the end
    pub fn next_frame(&mut self) -> Result<Option<Vec<u8>>, String> {
        let mut line = Vec::new();
        let n = self.r.read_until(b'\n', &mut line).map_err(|e| e.to_string())?;
        if n == 0 {
            return Ok(None);
        }
        if !line.starts_with(b"FRAME") {
            return Err("expected FRAME".into());
        }
        let mut buf = vec![0; self.header.frame_bytes()];
        self.r.read_exact(&mut buf).map_err(|e| e.to_string())?;
        Ok(Some(buf))
    }
}

pub fn write_header(w: &mut impl Write, h: &Header) -> std::io::Result<()> {
    writeln!(w, "YUV4MPEG2 W{} H{} F{}:{} Ip A1:1 C420jpeg", h.w, h.h, h.fps_num, h.fps_den)
}

pub fn write_frame(w: &mut impl Write, yuv: &[u8]) -> std::io::Result<()> {
    w.write_all(b"FRAME\n")?;
    w.write_all(yuv)
}
