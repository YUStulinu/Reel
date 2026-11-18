"""Rate-distortion and speed evaluation of Reel against x264, x265 and libaom.

    python eval/evaluate.py quality [--clips a,b] [--jobs N]   # all RD points, in parallel
    python eval/evaluate.py speed                              # encode/decode fps, one job at a time
    python eval/evaluate.py report                             # results/summary.md + RD plots

The baselines run through ffmpeg (Ubuntu's apt package) inside WSL on Windows,
or directly on Linux. Every encoder gets the same frames, one keyframe at the
start, and four rate points; quality is measured here (eval/metrics.py) on
the decoded output, rate from the size of the raw elementary stream.
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(ROOT, "data", "clips")
WORK = os.path.join(ROOT, "run", "eval")
RESULTS = os.path.join(ROOT, "results")

# clip -> frames used (CIF: 150 of 300; 720p: 60 of 500, to bound the run time)
CLIP_FRAMES = {
    "foreman_cif": 150,
    "akiyo_cif": 150,
    "mobile_cif": 150,
    "news_cif": 150,
    "park_joy_420_720p50": 60,
    "in_to_tree_420_720p50": 60,
}

# encoder -> (rate parameter values, output extension)
ENCODERS = {
    "reel": ([22, 27, 32, 37], "reel"),
    "x264": ([22, 27, 32, 37], "264"),
    "x264-P": ([22, 27, 32, 37], "264"),  # x264 limited to Reel's tools: no B frames, one reference
    "x265": ([22, 27, 32, 37], "hevc"),
    "libaom": ([24, 36, 48, 60], "ivf"),
}

ON_WINDOWS = platform.system() == "Windows"


def reel_bin():
    for p in [os.environ.get("REEL", ""), "C:/rt/release/reel.exe", os.path.join(ROOT, "codec", "target", "release", "reel"),
              os.path.join(ROOT, "codec", "target", "release", "reel.exe")]:
        if p and os.path.exists(p):
            return p
    sys.exit("build the codec first (cd codec && cargo build --release)")


def unix(path):
    """a path as the ffmpeg side sees it"""
    p = os.path.abspath(path).replace("\\", "/")
    if ON_WINDOWS:
        return "/mnt/" + p[0].lower() + p[2:]
    return p


def ffmpeg(args, capture=True):
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y"] + args
    if ON_WINDOWS:
        cmd = ["wsl", "-d", "Ubuntu-24.04", "--"] + cmd
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    r = subprocess.run(cmd, capture_output=capture, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stderr


def baseline_args(enc, param, frames):
    common = ["-frames:v", str(frames), "-g", "100000"]
    if enc == "x264":
        return common + ["-c:v", "libx264", "-preset", "medium", "-tune", "psnr", "-qp", str(param), "-threads", "1", "-f", "h264"]
    if enc == "x264-P":
        return common + ["-c:v", "libx264", "-preset", "medium", "-tune", "psnr", "-qp", str(param), "-bf", "0", "-refs", "1", "-threads", "1", "-f", "h264"]
    if enc == "x265":
        return common + ["-c:v", "libx265", "-preset", "medium", "-tune", "psnr",
                         "-x265-params", f"qp={param}:keyint=100000:pools=1:frame-threads=1:log-level=error", "-f", "hevc"]
    if enc == "libaom":
        return common + ["-c:v", "libaom-av1", "-cpu-used", "4", "-crf", str(param), "-b:v", "0", "-threads", "1", "-row-mt", "0", "-f", "ivf"]
    raise ValueError(enc)


BENCH = re.compile(r"rtime=([0-9.]+)s")


def encode_point(clip, enc, param, frames):
    """encodes, decodes and measures one rate point; returns a result row"""
    src = os.path.join(CLIPS, clip + ".y4m")
    _, _, fps = metrics.y4m_info(src)
    d = os.path.join(WORK, clip)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, f"{enc}_{param}.{ENCODERS[enc][1]}")
    dec = os.path.join(d, f"{enc}_{param}.y4m")
    t0 = time.time()
    if enc == "reel":
        extra = os.environ.get("REEL_ARGS", "").split()  # for experiments
        r = subprocess.run([reel_bin(), "encode", src, "-o", out, "--qp", str(param), "--keyint", "100000", "--frames", str(frames)] + extra,
                           check=True, capture_output=True, text=True)
        enc_secs = frames / float(re.search(r"fps ([0-9.]+)", r.stdout).group(1))
        r = subprocess.run([reel_bin(), "decode", out, "-o", dec], check=True, capture_output=True, text=True)
        bytes_ = os.path.getsize(out) - 17 - 4 * frames  # the elementary stream: no container header, no length prefixes
    else:
        err = ffmpeg(["-benchmark", "-i", unix(src)] + baseline_args(enc, param, frames) + [unix(out)])
        enc_secs = float(BENCH.findall(err)[-1])
        ffmpeg(["-i", unix(out), "-f", "yuv4mpegpipe", "-pix_fmt", "yuv420p", unix(dec)])
        bytes_ = os.path.getsize(out)
        if enc == "libaom":
            bytes_ -= 32 + 12 * frames  # the IVF file and frame headers
    q = metrics.compare(src, dec, frames)
    os.remove(dec)  # decoded video is large; keep only the bitstream
    row = {"clip": clip, "encoder": enc, "param": param, "frames": q["frames"], "kbps": bytes_ * 8 / 1000 / (frames / fps),
           "enc_fps_parallel": frames / enc_secs, **{k: q[k] for k in ("psnr_y", "psnr_u", "psnr_v", "psnr_yuv", "ssim_y")}}
    print(f"{clip:24s} {enc:7s} {param:3d}  {row['kbps']:9.1f} kbps  Y {row['psnr_y']:.3f}  SSIM {row['ssim_y']:.4f}  ({time.time() - t0:.0f} s)", flush=True)
    return row


def quality(args):
    clips = args.clips.split(",") if args.clips else list(CLIP_FRAMES)
    encs = args.encoders.split(",") if args.encoders else list(ENCODERS)
    jobs = [(c, e, p, args.frames or CLIP_FRAMES[c]) for c in clips for e in encs for p in ENCODERS[e][0]]
    path = os.path.join(RESULTS, "quality.json")
    old = json.load(open(path)) if os.path.exists(path) else []
    keep = [r for r in old if not (r["clip"] in clips and r["encoder"] in encs)]
    with ThreadPoolExecutor(args.jobs) as ex:
        rows = list(ex.map(lambda j: encode_point(*j), jobs))
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(keep + rows, open(path, "w"), indent=1)


def speed(args):
    """single-threaded encode and decode speed, one process at a time"""
    rows = []
    for clip, frames in [("foreman_cif", 150), ("park_joy_420_720p50", 60)]:
        src = os.path.join(CLIPS, clip + ".y4m")
        _, _, fps = metrics.y4m_info(src)
        d = os.path.join(WORK, "speed")
        os.makedirs(d, exist_ok=True)
        for enc in ENCODERS:
            param = ENCODERS[enc][0][2]
            out = os.path.join(d, f"{clip}_{enc}.{ENCODERS[enc][1]}")
            variants = [(enc, [])]
            if enc == "reel":
                variants = [("reel", ["--me", "hex"]), ("reel full-search CPU", ["--me", "full"]), ("reel full-search CUDA", ["--me", "cuda"])]
            for name, extra in variants:
                if enc == "reel":
                    exe = os.environ.get("REEL_CUDA", "C:/rtc/release/reel.exe") if "CUDA" in name else reel_bin()
                    if not os.path.exists(exe):
                        continue
                    r = subprocess.run([exe, "encode", src, "-o", out, "--qp", str(param), "--keyint", "100000", "--frames", str(frames)] + extra,
                                       capture_output=True, text=True)
                    if r.returncode != 0:
                        print(f"skip {name}: {r.stderr.strip()}")
                        continue
                    e_fps = float(re.search(r" fps ([0-9.]+)", r.stdout).group(1))
                    me_s = float(re.search(r"me_seconds ([0-9.]+)", r.stdout).group(1))
                    r = subprocess.run([reel_bin(), "decode", out], check=True, capture_output=True, text=True)
                    d_fps = float(re.search(r"fps ([0-9.]+)", r.stdout).group(1))
                else:
                    err = ffmpeg(["-benchmark", "-i", unix(src)] + baseline_args(enc, param, frames) + [unix(out)])
                    e_fps = frames / float(BENCH.findall(err)[-1])
                    me_s = None
                    err = ffmpeg(["-benchmark", "-threads", "1", "-i", unix(out), "-f", "null", "-"])
                    d_fps = frames / float(BENCH.findall(err)[-1])
                row = {"clip": clip, "encoder": name, "param": param, "enc_fps": e_fps, "dec_fps": d_fps, "me_seconds": me_s}
                print(f"{clip:24s} {name:24s} encode {e_fps:8.2f} fps  decode {d_fps:8.1f} fps" + (f"  (motion search {me_s:.2f} s)" if me_s else ""), flush=True)
                rows.append(row)
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(rows, open(os.path.join(RESULTS, "speed.json"), "w"), indent=1)


def curves(rows, clip, enc, metric):
    pts = sorted((r["kbps"], r[metric]) for r in rows if r["clip"] == clip and r["encoder"] == enc)
    return [p[0] for p in pts], [p[1] for p in pts]


def report(args):
    import numpy as np

    rows = json.load(open(os.path.join(RESULTS, "quality.json")))
    clips = [c for c in CLIP_FRAMES if any(r["clip"] == c for r in rows)]
    baselines = [e for e in ENCODERS if e != "reel" and any(r["encoder"] == e for r in rows)]
    lines = ["# Reel evaluation", "",
             "BD-rate of Reel against each baseline (positive: Reel needs that much more bitrate for the same quality).", ""]
    table = {}
    for metric, label in [("psnr_y", "PSNR-Y"), ("psnr_yuv", "PSNR-YUV 6:1:1"), ("ssim_y", "SSIM-Y")]:
        lines += [f"## BD-rate, {label}", "", "| clip | " + " | ".join(baselines) + " |", "|---" * (len(baselines) + 1) + "|"]
        sums = {b: [] for b in baselines}
        for c in clips:
            ra, qa = curves(rows, c, "reel", metric)
            cells = []
            for b in baselines:
                rb, qb = curves(rows, c, b, metric)
                v = metrics.bd_rate(np.array(rb), np.array(qb), np.array(ra), np.array(qa)) if len(ra) >= 4 and len(rb) >= 4 else None
                if v is not None:
                    sums[b].append(v)
                cells.append("n/a" if v is None else f"{v:+.1f}%")
                table[(metric, c, b)] = v
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines.append("| **mean** | " + " | ".join(f"**{np.mean(sums[b]):+.1f}%**" if sums[b] else "n/a" for b in baselines) + " |")
        lines.append("")
    lines += ["## Rate points", "", "| clip | encoder | param | kbps | PSNR-Y | PSNR-YUV | SSIM-Y |", "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (list(CLIP_FRAMES).index(r["clip"]), list(ENCODERS).index(r["encoder"]), r["param"])):
        lines.append(f"| {r['clip']} | {r['encoder']} | {r['param']} | {r['kbps']:.1f} | {r['psnr_y']:.3f} | {r['psnr_yuv']:.3f} | {r['ssim_y']:.4f} |")
    sp = os.path.join(RESULTS, "speed.json")
    if os.path.exists(sp):
        lines += ["", "## Speed (single thread)", "", "| clip | encoder | encode fps | decode fps | motion search s |", "|---|---|---|---|---|"]
        for r in json.load(open(sp)):
            me = "" if r["me_seconds"] is None else f"{r['me_seconds']:.2f}"
            lines.append(f"| {r['clip']} | {r['encoder']} | {r['enc_fps']:.2f} | {r['dec_fps']:.1f} | {me} |")
    open(os.path.join(RESULTS, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: no plots")
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    colors = {"reel": "#c2410c", "x264": "#2563eb", "x264-P": "#60a5fa", "x265": "#16a34a", "libaom": "#7c3aed"}
    for ax, c in zip(axes.flat, clips):
        for e in ["reel"] + baselines:
            r, q = curves(rows, c, e, "psnr_y")
            if r:
                ax.plot(r, q, "o-", ms=4, lw=2 if e == "reel" else 1.4, color=colors.get(e), label=e)
        ax.set_xscale("log")
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("kbps")
        ax.set_ylabel("PSNR-Y (dB)")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "rd_curves.png"), dpi=110)
    print("wrote results/rd_curves.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["quality", "speed", "report"])
    ap.add_argument("--clips", default="")
    ap.add_argument("--encoders", default="")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    a = ap.parse_args()
    {"quality": quality, "speed": speed, "report": report}[a.command](a)


if __name__ == "__main__":
    main()
