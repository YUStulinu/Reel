//! reel encode in.y4m -o out.reel [--qp N | --bitrate KBPS] [--keyint N] [--bframes N] [--me hex|full|cuda] [--range N] [--frames N] [--recon r.y4m] [-v]
//! reel decode in.reel [-o out.y4m]
//! reel info in.reel
//! reel checksum in.reel   (FNV-1a of each decoded picture)
use reel::bitstream::{self, StreamInfo};
use reel::decoder::Decoder;
use reel::encoder::{Config, Encoder, MeMethod};
use reel::rc;
use reel::y4m;
use std::io::Write;
use std::time::Instant;

fn die(msg: &str) -> ! {
    eprintln!("reel: {msg}");
    std::process::exit(1)
}

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

fn num<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    arg(args, name).map(|v| v.parse().unwrap_or_else(|_| die(&format!("bad value for {name}"))))
}

fn encode(args: &[String]) {
    let input = args.get(2).unwrap_or_else(|| die("missing input"));
    let output = arg(args, "-o").unwrap_or_else(|| die("missing -o output"));
    let file = std::fs::File::open(input).unwrap_or_else(|e| die(&format!("{input}: {e}")));
    let mut rd = y4m::Reader::new(file).unwrap_or_else(|e| die(&e));
    let h = rd.header.clone();
    let fps = h.fps_num as f64 / h.fps_den as f64;
    let mut cfg = Config::default();
    if let Some(q) = num::<i32>(args, "--qp") {
        cfg.rate = rc::Mode::Cqp(q.clamp(0, 51));
    }
    if let Some(k) = num::<f64>(args, "--bitrate") {
        cfg.rate = rc::Mode::Abr(k * 1000.0);
    }
    if let Some(k) = num(args, "--keyint") {
        cfg.keyint = k;
    }
    if let Some(r) = num(args, "--range") {
        cfg.range = r;
    }
    if let Some(b) = num(args, "--bframes") {
        cfg.bframes = b;
    }
    cfg.me = match arg(args, "--me").unwrap_or("hex") {
        "hex" => MeMethod::Hex,
        "full" => MeMethod::Full,
        "cuda" => MeMethod::Cuda,
        m => die(&format!("unknown motion search {m}")),
    };
    let limit: usize = num(args, "--frames").unwrap_or(usize::MAX);
    let info = StreamInfo { w: h.w, h: h.h, fps_num: h.fps_num, fps_den: h.fps_den };
    let mut out = Vec::new();
    bitstream::write_header(&mut out, &info);
    cfg.keep_recon = arg(args, "--recon").is_some();
    let mut enc = Encoder::new(cfg, info, fps);
    let mut recon = arg(args, "--recon").map(|p| {
        let mut f = std::io::BufWriter::new(std::fs::File::create(p).unwrap_or_else(|e| die(&format!("{p}: {e}"))));
        y4m::write_header(&mut f, &h).unwrap();
        f
    });
    let verbose = args.iter().any(|a| a == "-v");
    let (mut n, mut coded, mut psnr, mut secs) = (0usize, 0usize, [0.0f64; 3], 0.0f64);
    // reconstructed pictures arrive in decode order: write them in display order
    let mut pending: std::collections::BTreeMap<usize, Vec<u8>> = Default::default();
    let mut next_out = 0usize;
    let mut sink = |batch: Vec<reel::encoder::Coded>, out: &mut Vec<u8>, coded: &mut usize, psnr: &mut [f64; 3]| {
        for c in batch {
            out.extend_from_slice(&c.packet);
            *coded += 1;
            let st = &c.stats;
            for k in 0..3 {
                psnr[k] += st.psnr[k];
            }
            if verbose {
                eprintln!(
                    "frame {:4} {} qp {:2} {:7} bytes  Y {:.2} U {:.2} V {:.2}  skip/inter/intra {:?}",
                    st.display,
                    ["I", "P", "B"][st.kind as usize],
                    st.qp,
                    st.bytes,
                    st.psnr[0],
                    st.psnr[1],
                    st.psnr[2],
                    st.modes
                );
            }
            if let (Some(f), Some(pic)) = (recon.as_mut(), c.recon) {
                pending.insert(st.display, pic);
                while let Some(p) = pending.remove(&next_out) {
                    y4m::write_frame(f, &p).unwrap();
                    next_out += 1;
                }
            }
        }
    };
    while n < limit {
        let Some(pic) = rd.next_frame().unwrap_or_else(|e| die(&e)) else { break };
        let t0 = Instant::now();
        let batch = enc.push(&pic);
        secs += t0.elapsed().as_secs_f64();
        sink(batch, &mut out, &mut coded, &mut psnr);
        n += 1;
    }
    let t0 = Instant::now();
    let batch = enc.flush();
    secs += t0.elapsed().as_secs_f64();
    sink(batch, &mut out, &mut coded, &mut psnr);
    assert_eq!(coded, n);
    if n == 0 {
        die("no frames");
    }
    if let Some(mut f) = recon {
        f.flush().unwrap();
    }
    std::fs::write(output, &out).unwrap_or_else(|e| die(&format!("{output}: {e}")));
    let kbps = (out.len() - bitstream::HEADER_LEN) as f64 * 8.0 / (n as f64 / fps) / 1000.0;
    // one machine-readable summary line
    println!(
        "frames {n} kbps {kbps:.2} psnr_y {:.4} psnr_u {:.4} psnr_v {:.4} fps {:.2} me_seconds {:.3}",
        psnr[0] / n as f64,
        psnr[1] / n as f64,
        psnr[2] / n as f64,
        n as f64 / secs,
        enc.me_seconds
    );
}

fn decode(args: &[String]) {
    let input = args.get(2).unwrap_or_else(|| die("missing input"));
    let data = std::fs::read(input).unwrap_or_else(|e| die(&format!("{input}: {e}")));
    let info = bitstream::read_header(&data).unwrap_or_else(|e| die(&e));
    let packets = bitstream::packets(&data).unwrap_or_else(|e| die(&e));
    let mut dec = Decoder::new(info);
    let mut out: Option<std::io::BufWriter<std::fs::File>> = arg(args, "-o").map(|p| {
        let mut f = std::io::BufWriter::new(std::fs::File::create(p).unwrap_or_else(|e| die(&format!("{p}: {e}"))));
        y4m::write_header(&mut f, &y4m::Header { w: info.w, h: info.h, fps_num: info.fps_num, fps_den: info.fps_den }).unwrap();
        f
    });
    let t0 = Instant::now();
    for (i, &(off, len)) in packets.iter().enumerate() {
        let shown = dec.decode(&data[off..off + len]).unwrap_or_else(|e| die(&format!("frame {i}: {e}")));
        if let (Some(f), Some(pic)) = (out.as_mut(), shown) {
            y4m::write_frame(f, &pic.store(info.w, info.h)).unwrap();
        }
    }
    if let (Some(f), Some(pic)) = (out.as_mut(), dec.flush()) {
        y4m::write_frame(f, &pic.store(info.w, info.h)).unwrap();
    }
    let secs = t0.elapsed().as_secs_f64();
    if let Some(mut f) = out {
        f.flush().unwrap();
    }
    println!("frames {} fps {:.2}", packets.len(), packets.len() as f64 / secs);
}

/// FNV-1a of every decoded picture (the wasm decoder test compares against these)
fn checksum(args: &[String]) {
    let input = args.get(2).unwrap_or_else(|| die("missing input"));
    let data = std::fs::read(input).unwrap_or_else(|e| die(&format!("{input}: {e}")));
    for pic in Decoder::decode_stream(&data).unwrap_or_else(|e| die(&e)) {
        let mut h = 0x811c9dc5u32;
        for b in pic {
            h = (h ^ b as u32).wrapping_mul(0x01000193);
        }
        println!("{h}");
    }
}

fn info(args: &[String]) {
    let input = args.get(2).unwrap_or_else(|| die("missing input"));
    let data = std::fs::read(input).unwrap_or_else(|e| die(&format!("{input}: {e}")));
    let s = bitstream::read_header(&data).unwrap_or_else(|e| die(&e));
    let packets = bitstream::packets(&data).unwrap_or_else(|e| die(&e));
    let keys = packets.iter().filter(|&&(o, _)| data[o] == bitstream::TYPE_I).count();
    println!("{}x{} at {}/{} fps, {} frames, {} keyframes, {} bytes", s.w, s.h, s.fps_num, s.fps_den, packets.len(), keys, data.len());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("encode") => encode(&args),
        Some("decode") => decode(&args),
        Some("info") => info(&args),
        Some("checksum") => checksum(&args),
        _ => die("usage: reel encode|decode|info ..."),
    }
}
