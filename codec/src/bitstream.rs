//! The .reel container: a small header, then one length-prefixed packet per
//! frame. Keyframes (I) restart prediction, so a stream can be cut there.
pub const MAGIC: &[u8; 4] = b"REEL";
pub const VERSION: u8 = 1;
pub const HEADER_LEN: usize = 4 + 1 + 2 + 2 + 4 + 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamInfo {
    pub w: usize,
    pub h: usize,
    pub fps_num: u32,
    pub fps_den: u32,
}

pub fn write_header(out: &mut Vec<u8>, s: &StreamInfo) {
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.extend_from_slice(&(s.w as u16).to_le_bytes());
    out.extend_from_slice(&(s.h as u16).to_le_bytes());
    out.extend_from_slice(&s.fps_num.to_le_bytes());
    out.extend_from_slice(&s.fps_den.to_le_bytes());
}

pub fn read_header(b: &[u8]) -> Result<StreamInfo, String> {
    if b.len() < HEADER_LEN || &b[..4] != MAGIC {
        return Err("not a Reel stream".into());
    }
    if b[4] != VERSION {
        return Err(format!("unsupported version {}", b[4]));
    }
    let u16_at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]) as usize;
    let u32_at = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
    let s = StreamInfo { w: u16_at(5), h: u16_at(7), fps_num: u32_at(9), fps_den: u32_at(13) };
    if s.w == 0 || s.h == 0 || s.w > 8192 || s.h > 8192 || s.fps_den == 0 {
        return Err("bad stream header".into());
    }
    Ok(s)
}

pub const TYPE_I: u8 = 0;
pub const TYPE_P: u8 = 1;
/// a bi-predicted frame: not a reference, displayed before the P frame decoded just ahead of it
pub const TYPE_B: u8 = 2;

/// appends a frame packet: u32 length, type, qp, entropy-coded payload
pub fn write_packet(out: &mut Vec<u8>, ftype: u8, qp: u8, payload: &[u8]) {
    out.extend_from_slice(&((payload.len() + 2) as u32).to_le_bytes());
    out.push(ftype);
    out.push(qp);
    out.extend_from_slice(payload);
}

/// the packets of a stream: (offset of the type byte, length incl. type and qp)
pub fn packets(b: &[u8]) -> Result<Vec<(usize, usize)>, String> {
    read_header(b)?;
    let mut out = vec![];
    let mut p = HEADER_LEN;
    while p < b.len() {
        if p + 4 > b.len() {
            return Err("truncated packet length".into());
        }
        let n = u32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]) as usize;
        if n < 2 || p + 4 + n > b.len() {
            return Err("truncated packet".into());
        }
        out.push((p + 4, n));
        p += 4 + n;
    }
    Ok(out)
}
