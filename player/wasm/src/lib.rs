//! The Reel decoder for the browser: a C-style API over linear memory, no
//! imports. JavaScript writes packets into memory from `reel_alloc`, calls
//! `reel_decode`, then reads RGBA pixels from `reel_rgba`.
use reel::bitstream::{read_header, StreamInfo};
use reel::decoder::Decoder;

struct State {
    dec: Decoder,
    rgba: Vec<u8>,
    /// FNV-1a of the last shown picture (visible YUV)
    sum: u32,
}

static mut STATE: Option<State> = None;

#[allow(static_mut_refs)]
fn state() -> Option<&'static mut State> {
    // single-threaded wasm: no other reference can exist
    unsafe { STATE.as_mut() }
}

#[no_mangle]
pub extern "C" fn reel_alloc(n: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(n);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

#[no_mangle]
pub unsafe extern "C" fn reel_free(p: *mut u8, n: usize) {
    drop(Vec::from_raw_parts(p, 0, n));
}

/// starts a stream from its 17-byte header; returns 0 on success
#[no_mangle]
pub unsafe extern "C" fn reel_open(p: *const u8, n: usize) -> i32 {
    let h = std::slice::from_raw_parts(p, n);
    match read_header(h) {
        Ok(info) => {
            STATE = Some(State { dec: Decoder::new(info), rgba: vec![255; info.w * info.h * 4], sum: 0 });
            0
        }
        Err(_) => -1,
    }
}

fn info() -> StreamInfo {
    state().map(|s| s.dec.info).unwrap_or(StreamInfo { w: 0, h: 0, fps_num: 0, fps_den: 1 })
}

#[no_mangle]
pub extern "C" fn reel_width() -> u32 {
    info().w as u32
}
#[no_mangle]
pub extern "C" fn reel_height() -> u32 {
    info().h as u32
}

fn to_rgba(rgba: &mut [u8], frame: &reel::frame::Frame, w: usize, h: usize) {
    // BT.601 limited range, integer
    let [yp, up, vp] = &frame.planes;
    for y in 0..h {
        let (yr, ur, vr) = (yp.row(y), up.row(y / 2), vp.row(y / 2));
        let o = y * w * 4;
        for x in 0..w {
            let c = (yr[x] as i32 - 16) * 298;
            let d = ur[x / 2] as i32 - 128;
            let e = vr[x / 2] as i32 - 128;
            let px = &mut rgba[o + x * 4..o + x * 4 + 3];
            px[0] = ((c + 409 * e + 128) >> 8).clamp(0, 255) as u8;
            px[1] = ((c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255) as u8;
            px[2] = ((c + 516 * d + 128) >> 8).clamp(0, 255) as u8;
        }
    }
}

fn shown(s: &mut State, frame: Option<&reel::frame::Frame>) -> i32 {
    match frame {
        Some(f) => {
            let (w, h) = (s.dec.info.w, s.dec.info.h);
            let mut hsh = 0x811c9dc5u32;
            for b in f.store(w, h) {
                hsh = (hsh ^ b as u32).wrapping_mul(0x01000193);
            }
            s.sum = hsh;
            to_rgba(&mut s.rgba, f, w, h);
            1
        }
        None => 0,
    }
}

/// decodes one packet (type byte, qp byte, payload). Returns 1 when a picture
/// is ready to show (RGBA in `reel_rgba`), 0 when the decoder holds it back
/// (frames come in decode order), negative on errors.
#[no_mangle]
pub unsafe extern "C" fn reel_decode(p: *const u8, n: usize) -> i32 {
    let Some(s) = state() else { return -1 };
    let data = std::slice::from_raw_parts(p, n);
    let f = match s.dec.decode(data) {
        Ok(f) => f.map(|f| f.clone()),
        Err(_) => return -2,
    };
    shown(s, f.as_ref())
}

/// the picture still held at the end of a stream or segment: 1 if there was one
#[no_mangle]
pub extern "C" fn reel_flush() -> i32 {
    let Some(s) = state() else { return -1 };
    let f = s.dec.flush().cloned();
    shown(s, f.as_ref())
}

#[no_mangle]
pub extern "C" fn reel_rgba() -> *const u8 {
    state().map_or(std::ptr::null(), |s| s.rgba.as_ptr())
}

/// FNV-1a of the visible Y, U and V planes of the last shown picture
/// (lets tests compare the wasm decoder with the native one)
#[no_mangle]
pub extern "C" fn reel_checksum() -> u32 {
    state().map_or(0, |s| s.sum)
}
