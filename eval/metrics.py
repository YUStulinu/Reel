"""Quality metrics and Bjøntegaard deltas, written out here (numpy only).

  psnr      per plane, from the mean squared error over the whole sequence
  ssim      Wang et al. 2004: 11x11 Gaussian window (sigma 1.5), K1 0.01, K2 0.03, luma
  bd_rate   average bitrate difference at equal quality (cubic fit of log rate)
"""
import numpy as np


def read_y4m(path, limit=None):
    """yields (Y, U, V) uint8 arrays of an 8-bit 4:2:0 .y4m file"""
    with open(path, "rb") as f:
        header = f.readline().decode().split()
        if header[0] != "YUV4MPEG2":
            raise ValueError(f"{path}: not a y4m file")
        params = {t[0]: t[1:] for t in header[1:]}
        if "C" in params and not params["C"].startswith("420"):
            raise ValueError(f"{path}: only 4:2:0 is supported")
        w, h = int(params["W"]), int(params["H"])
        cw, ch = (w + 1) // 2, (h + 1) // 2
        size = w * h + 2 * cw * ch
        n = 0
        while limit is None or n < limit:
            line = f.readline()
            if not line:
                return
            if not line.startswith(b"FRAME"):
                raise ValueError(f"{path}: expected FRAME")
            buf = np.frombuffer(f.read(size), dtype=np.uint8)
            if buf.size < size:
                return
            yield (buf[: w * h].reshape(h, w), buf[w * h: w * h + cw * ch].reshape(ch, cw), buf[w * h + cw * ch:].reshape(ch, cw))
            n += 1


def y4m_info(path):
    with open(path, "rb") as f:
        params = {t[0]: t[1:] for t in f.readline().decode().split()[1:]}
    num, den = (int(x) for x in params.get("F", "30:1").split(":"))
    return int(params["W"]), int(params["H"]), num / den


def psnr_from_mse(mse):
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def _gauss(sigma=1.5, size=11):
    x = np.arange(size) - size // 2
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _filter(img, g):
    """separable 'valid' filtering of a float image with a 1-D kernel"""
    k = len(g)
    rows = sum(g[i] * img[:, i: img.shape[1] - k + 1 + i] for i in range(k))
    return sum(g[i] * rows[i: rows.shape[0] - k + 1 + i, :] for i in range(k))


def ssim(a, b):
    """mean SSIM of two single-plane images"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    g = _gauss()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = _filter(a, g), _filter(b, g)
    saa = _filter(a * a, g) - mu_a ** 2
    sbb = _filter(b * b, g) - mu_b ** 2
    sab = _filter(a * b, g) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return float(m.mean())


def compare(ref_path, test_path, limit=None):
    """sequence PSNR (Y, U, V, and YUV weighted 6:1:1) and mean SSIM (Y)"""
    sse = np.zeros(3)
    count = np.zeros(3)
    ssims = []
    n = 0
    for r, t in zip(read_y4m(ref_path, limit), read_y4m(test_path, limit)):
        for k in range(3):
            d = r[k].astype(np.int32) - t[k].astype(np.int32)
            sse[k] += float((d * d).sum())
            count[k] += d.size
        ssims.append(ssim(r[0], t[0]))
        n += 1
    if n == 0:
        raise ValueError("no frames to compare")
    p = [psnr_from_mse(sse[k] / count[k]) for k in range(3)]
    return {"frames": n, "psnr_y": p[0], "psnr_u": p[1], "psnr_v": p[2], "psnr_yuv": (6 * p[0] + p[1] + p[2]) / 8, "ssim_y": float(np.mean(ssims))}


def _integral(rates, quals, lo, hi):
    """integral over quality [lo, hi] of a cubic fit of log10(rate) as a function of quality"""
    coef = np.polyfit(quals, np.log10(rates), 3)
    prim = np.polyint(coef)
    return np.polyval(prim, hi) - np.polyval(prim, lo)


def bd_rate(rates_a, quals_a, rates_b, quals_b):
    """percent bitrate of B relative to A at equal quality (negative: B is better)"""
    lo = max(min(quals_a), min(quals_b))
    hi = min(max(quals_a), max(quals_b))
    if hi <= lo:
        return None  # the curves do not overlap
    da = _integral(rates_a, quals_a, lo, hi) / (hi - lo)
    db = _integral(rates_b, quals_b, lo, hi) / (hi - lo)
    return (10 ** (db - da) - 1) * 100


def bd_quality(rates_a, quals_a, rates_b, quals_b):
    """average quality difference of B minus A at equal bitrate"""
    la, lb = np.log10(rates_a), np.log10(rates_b)
    lo, hi = max(la.min(), lb.min()), min(la.max(), lb.max())
    if hi <= lo:
        return None
    ia = np.polyval(np.polyint(np.polyfit(la, quals_a, 3)), [lo, hi])
    ib = np.polyval(np.polyint(np.polyfit(lb, quals_b, 3)), [lo, hi])
    return ((ib[1] - ib[0]) - (ia[1] - ia[0])) / (hi - lo)
