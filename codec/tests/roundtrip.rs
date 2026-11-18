//! End-to-end: encode synthetic video, decode it, and check that the decoder
//! reproduces the encoder's reconstruction bit for bit.
use reel::bitstream::{self, StreamInfo};
use reel::decoder::Decoder;
use reel::encoder::{encode_all, Config, Encoder, MeMethod};
use reel::rc;

/// a moving, textured 4:2:0 sequence (size deliberately not a multiple of 16)
fn clip(w: usize, h: usize, frames: usize) -> Vec<Vec<u8>> {
    let (cw, ch) = ((w + 1) / 2, (h + 1) / 2);
    (0..frames)
        .map(|t| {
            let mut f = vec![0u8; w * h + 2 * cw * ch];
            for y in 0..h {
                for x in 0..w {
                    // a pattern panning right and down, and a square moving the other way
                    let (sx, sy) = (x as i32 - 2 * t as i32, y as i32 - t as i32);
                    let mut v = 128.0 + 60.0 * ((sx as f64) * 0.21).sin() * ((sy as f64) * 0.13).cos() + ((sx * 7 + sy * 3) & 15) as f64;
                    let (bx, by) = (40 + 3 * t as i32, 10 + t as i32);
                    if (x as i32 - bx).abs() < 8 && (y as i32 - by).abs() < 8 {
                        v = 230.0;
                    }
                    f[y * w + x] = v.clamp(0.0, 255.0) as u8;
                }
            }
            for k in 0..2 {
                for y in 0..ch {
                    for x in 0..cw {
                        f[w * h + k * cw * ch + y * cw + x] = (100 + 40 * k + ((x + y + t) % 20)) as u8;
                    }
                }
            }
            f
        })
        .collect()
}

fn decode_all(stream: &[u8]) -> Vec<Vec<u8>> {
    Decoder::decode_stream(stream).unwrap()
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let sse: f64 = a.iter().zip(b).map(|(x, y)| ((*x as f64) - (*y as f64)).powi(2)).sum();
    10.0 * (255.0f64.powi(2) * a.len() as f64 / sse.max(1e-9)).log10()
}

#[test]
fn decoder_matches_encoder_reconstruction() {
    let (w, h) = (100, 60);
    let frames = clip(w, h, 14);
    let info = StreamInfo { w, h, fps_num: 25, fps_den: 1 };
    for (me, bframes) in [(MeMethod::Hex, 0), (MeMethod::Hex, 2), (MeMethod::Full, 3)] {
        for qp in [12, 30, 45] {
            let cfg = Config { rate: rc::Mode::Cqp(qp), keyint: 6, me, range: 16, bframes, keep_recon: true };
            let mut e = Encoder::new(cfg, info, 25.0);
            let mut stream = Vec::new();
            bitstream::write_header(&mut stream, &info);
            let mut recons = vec![None; frames.len()];
            let mut kinds = vec![];
            for f in &frames {
                for c in e.push(f) {
                    stream.extend_from_slice(&c.packet);
                    kinds.push(c.stats.kind);
                    recons[c.stats.display] = c.recon;
                }
            }
            for c in e.flush() {
                stream.extend_from_slice(&c.packet);
                kinds.push(c.stats.kind);
                recons[c.stats.display] = c.recon;
            }
            assert_eq!(kinds.iter().filter(|&&k| k == 2).count() > 0, bframes > 0, "B frames used iff enabled: {kinds:?}");
            let decoded = decode_all(&stream);
            assert_eq!(decoded.len(), frames.len());
            for (i, (d, r)) in decoded.iter().zip(&recons).enumerate() {
                assert!(Some(d) == r.as_ref(), "{me:?} b{bframes} qp {qp} frame {i}: decoder differs from the encoder");
            }
            let q = psnr(&frames[11], &decoded[11]);
            let floor = match qp {
                12 => 44.0,
                30 => 32.0,
                _ => 24.0,
            };
            assert!(q > floor, "{me:?} b{bframes} qp {qp}: psnr {q:.2}");
        }
    }
}

#[test]
fn lower_qp_costs_more_bits_and_gives_more_quality() {
    let (w, h) = (64, 48);
    let frames = clip(w, h, 6);
    let info = StreamInfo { w, h, fps_num: 25, fps_den: 1 };
    let mut last = (usize::MAX, 0.0);
    for qp in [20, 28, 36, 44] {
        let (s, stats) = encode_all(Config { rate: rc::Mode::Cqp(qp), ..Config::default() }, info, 25.0, &frames);
        let q = stats.iter().map(|s| s.psnr[0]).sum::<f64>() / stats.len() as f64;
        assert!(s.len() < last.0, "qp {qp}: {} bytes", s.len());
        assert!(q < last.1 || last.1 == 0.0, "qp {qp}: {q}");
        last = (s.len(), q);
    }
}

#[test]
fn rate_control_hits_the_target() {
    let (w, h) = (128, 96);
    let frames = clip(w, h, 50);
    let info = StreamInfo { w, h, fps_num: 25, fps_den: 1 };
    for kbps in [150.0, 400.0] {
        let (s, _) = encode_all(Config { rate: rc::Mode::Abr(kbps * 1000.0), keyint: 25, ..Config::default() }, info, 25.0, &frames);
        let got = (s.len() - bitstream::HEADER_LEN) as f64 * 8.0 / 2.0 / 1000.0;
        assert!((got - kbps).abs() / kbps < 0.15, "target {kbps} kbps, got {got:.1}");
    }
}

#[test]
fn corrupt_streams_fail_cleanly() {
    let (w, h) = (48, 32);
    let frames = clip(w, h, 3);
    let info = StreamInfo { w, h, fps_num: 25, fps_den: 1 };
    let (s, _) = encode_all(Config::default(), info, 25.0, &frames);
    let packets = bitstream::packets(&s).unwrap();
    let mut d = Decoder::new(info);
    // a P frame first is an error, not a panic
    let (o, n) = packets[1];
    assert!(d.decode(&s[o..o + n]).is_err());
    // flipped bits decode to something (or an error) without panicking
    for flip in [3usize, 17, 40] {
        let (o, n) = packets[0];
        let mut p = s[o..o + n].to_vec();
        let i = (flip % (n - 2)) + 2;
        p[i] ^= 0x5A;
        let mut d = Decoder::new(info);
        let _ = d.decode(&p);
    }
    assert!(bitstream::read_header(b"NOPE").is_err());
}
