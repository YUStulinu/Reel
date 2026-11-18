"""Encodes a clip at several bitrates with aligned keyframes, for the
streaming server (server/media/<name>/<kbps>k.reel).

    python scripts/ladder.py data/clips/foreman_cif.y4m --kbps 150 300 600 1200 [--keyint 30] [--frames N]
"""
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--kbps", type=int, nargs="+", default=[150, 300, 600, 1200])
    ap.add_argument("--keyint", type=int, default=30)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--reel", default=os.environ.get("REEL", "C:/rt/release/reel.exe"))
    a = ap.parse_args()
    name = a.name or os.path.basename(a.clip).split(".")[0]
    out = os.path.join(ROOT, "server", "media", name)
    os.makedirs(out, exist_ok=True)

    def one(k):
        cmd = [a.reel, "encode", a.clip, "-o", os.path.join(out, f"{k:05d}k.reel"), "--bitrate", str(k), "--keyint", str(a.keyint)]
        if a.frames:
            cmd += ["--frames", str(a.frames)]
        r = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return k, r.stdout.strip()

    with ThreadPoolExecutor(len(a.kbps)) as ex:
        for k, line in ex.map(one, a.kbps):
            print(f"{k:5d} kbps target: {line}")


if __name__ == "__main__":
    main()
