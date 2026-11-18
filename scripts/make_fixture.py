"""Builds player/test/fixture.reel (a small synthetic clip encoded by Reel) and
the per-frame checksums of the native decoder, which the wasm test must match.

    python scripts/make_fixture.py [--reel PATH]
"""
import argparse
import json
import math
import os
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def synthetic_y4m(path, w=96, h=64, frames=8):
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{w} H{h} F25:1 Ip A1:1 C420jpeg\n".encode())
        for t in range(frames):
            y = bytearray(w * h)
            for r in range(h):
                for c in range(w):
                    v = 128 + 70 * math.sin((c - 2 * t) * 0.19) * math.cos((r + t) * 0.11) + ((c * 5 + r * 3 + t) & 7)
                    if abs(c - 30 - 4 * t) < 9 and abs(r - 20 - t) < 9:
                        v = 235
                    y[r * w + c] = max(0, min(255, int(v)))
            uv = bytes((90 + ((c + r + t) % 30)) for r in range(h // 2) for c in range(w // 2))
            f.write(b"FRAME\n" + bytes(y) + uv + uv[::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reel", default=os.environ.get("REEL", "C:/rt/release/reel.exe"))
    a = ap.parse_args()
    out = os.path.join(ROOT, "player", "test", "fixture.reel")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "clip.y4m")
        synthetic_y4m(src)
        subprocess.run([a.reel, "encode", src, "-o", out, "--qp", "30", "--keyint", "4"], check=True, capture_output=True)
    sums = subprocess.run([a.reel, "checksum", out], check=True, capture_output=True, text=True).stdout.split()
    json.dump({"checksums": [int(s) for s in sums]}, open(os.path.join(ROOT, "player", "test", "fixture.json"), "w"), indent=1)
    print(f"{out}: {os.path.getsize(out)} bytes, {len(sums)} frames")


if __name__ == "__main__":
    main()
