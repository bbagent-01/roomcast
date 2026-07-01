#!/usr/bin/env python3
"""RoomCast ANGLE — honest, deterministic vertical-straighten (no generative repaint, no new vantage).
Per-room roll + keystone (measured from each photo's long wall lines), applied as a homography to
both the before and the lit versions, then a black-free inscribed crop. Bathroom also gets an honest
doorframe crop. Hand-set per room because straightening is inherently per-photo; auto line-estimation
was unreliable across these varied wide-angle shots."""
import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import to_43
from upright import _build_H

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets", "loren-apt")

# roll (deg, +CCW), k (vertical keystone); doorframe -> crop the doorway out & zoom into the interior
ANGLE = {
    "1-living":   dict(roll=-1.1, k=-0.0004),
    "2-kitchen":  dict(roll=-7.0, k=-0.0002),
    "3-kitchen2": dict(roll=-5.3, k=-0.0001),
    "4-hall":     dict(roll=-1.0, k=-0.0004),
    "5-bath":     dict(roll=-1.3, k=-0.0001, doorframe=dict(max_frac=0.26, dark_ratio=0.72, zoom=1.06)),
}


def warp_pair(before, lit, roll, k):
    """Apply the same homography to both; crop a centered margin that scales with the roll so the
    rotated/keystoned edges (replicate smear) are removed. No black borders, no invented pixels."""
    h, w = before.shape[:2]
    H = _build_H(w, h, roll, k)
    b = cv2.warpPerspective(before, H, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    l = cv2.warpPerspective(lit, H, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    frac = 0.05 + 1.15 * np.sin(np.radians(abs(roll))) + 220.0 * abs(k)
    frac = min(frac, 0.28)
    mx, my = int(frac * w), int(frac * h)
    return b[my:h - my, mx:w - mx], l[my:h - my, mx:w - mx]


def doorframe_crop(im, max_frac=0.26, dark_ratio=0.72):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape
    thr = dark_ratio * float(g.mean())
    col = g.mean(axis=0); row = g.mean(axis=1)
    def adv(p, cap):
        i = 0
        while i < cap and p[i] < thr:
            i += 1
        return i
    x0 = adv(col, int(max_frac * W)); x1 = W - adv(col[::-1], int(max_frac * W))
    y0 = adv(row, int(max_frac * H)); y1 = H - adv(row[::-1], int(max_frac * H))
    if x1 - x0 < 0.6 * W: x0, x1 = 0, W
    if y1 - y0 < 0.6 * H: y0, y1 = 0, H
    return x0, x1, y0, y1


def main(stems=None):
    for stem, p in ANGLE.items():
        if stems and stem not in stems:
            continue
        before = cv2.imread(os.path.join(ASSETS, f"{stem}-before.webp"))
        lit = cv2.imread(os.path.join(ASSETS, f"{stem}-lit.webp"))
        b2, l2 = warp_pair(before, lit, p["roll"], p["k"])
        z = 1.0
        df = p.get("doorframe")
        if df:
            x0, x1, y0, y1 = doorframe_crop(b2, df["max_frac"], df["dark_ratio"])
            b2 = b2[y0:y1, x0:x1]; l2 = l2[y0:y1, x0:x1]; z = df.get("zoom", 1.0)
        cv2.imwrite(os.path.join(ASSETS, f"{stem}-angle.webp"), to_43(b2, 1600, 1200, z), [cv2.IMWRITE_WEBP_QUALITY, 92])
        cv2.imwrite(os.path.join(ASSETS, f"{stem}-lit-angle.webp"), to_43(l2, 1600, 1200, z), [cv2.IMWRITE_WEBP_QUALITY, 92])
        print(f"{stem:12s} roll {p['roll']:+.2f} k {p['k']:+.5f}{'  +doorframe' if df else ''}  APPLIED", flush=True)
    print("ANGLE COMPLETE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
