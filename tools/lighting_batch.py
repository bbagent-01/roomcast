#!/usr/bin/env python3
"""RoomCast LIGHTING pass — HONEST, deterministic relight (no generative repaint).
Operates only on existing pixels. Cannot add/remove/swap anything. Same native framing.

relight_pro: adaptive (bright rooms get a light touch, dim rooms more lift), a white/black
point re-anchor to kill the 'muddy gray whites' a flat gray-world WB caused, gentle local
contrast, and a mild unsharp pass to counter softness."""
import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import gray_world_wb, lmean

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets", "loren-apt")
STEMS = ["1-living", "2-kitchen", "3-kitchen2", "4-hall", "5-bath"]


def relight_pro(im):
    meanL0 = lmean(im)
    # 1) gentle WB — keep warmth, avoid the gray mud a strong gray-world caused
    im = gray_world_wb(im, 0.28)

    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0

    # 2) adaptive shadow/mid lift — dim rooms get more, already-bright rooms barely any
    shadow = float(np.clip(0.55 - (meanL0 - 100.0) / 120.0, 0.12, 0.55))
    L = np.power(np.clip(L, 0, 1), 1.0 / (1.0 + shadow))

    # 3) window/highlight recovery — pull near-blown REAL pixels back (never invents a view)
    thr = 0.88
    hi = L > thr
    L[hi] = thr + (L[hi] - thr) * (1.0 - 0.70)

    # 4) white/black-point re-anchor — restores clean whites + punch (fixes 'muddy/flat')
    lo = np.percentile(L, 1.5)
    hiP = np.percentile(L, 99.0)
    if hiP - lo > 1e-3:
        L = (L - lo) / (hiP - lo)
        L = 0.02 + np.clip(L, 0, 1) * 0.95          # gentle, leaves headroom

    L8 = np.clip(L * 255, 0, 255).astype(np.uint8)

    # 5) light local contrast (dimensional, not blotchy)
    L8 = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(8, 8)).apply(L8)

    # 6) soft S-curve pop
    Ls = L8.astype(np.float32) / 255.0
    Ls = np.clip(0.5 + (Ls - 0.5) * 1.10, 0, 1)
    lab[:, :, 0] = Ls * 255.0

    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    # 7) mild vibrance
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.10, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 8) mild unsharp — counters the softness reviewers flagged
    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    out = cv2.addWeighted(out, 1.45, blur, -0.45, 0)
    return out


def main():
    for stem in STEMS:
        inp = os.path.join(ASSETS, f"{stem}-before.webp")
        out = os.path.join(ASSETS, f"{stem}-lit.webp")   # fresh name — never a stale generative cache hit
        im = cv2.imread(inp, cv2.IMREAD_COLOR)
        if im is None:
            print(f"  SKIP {stem}: cannot read {inp}", flush=True); continue
        l0 = lmean(im)
        outim = relight_pro(im)
        cv2.imwrite(out, outim, [cv2.IMWRITE_WEBP_QUALITY, 92])
        print(f"{stem}: {im.shape[1]}x{im.shape[0]}  L {l0:.1f} -> {lmean(outim):.1f}", flush=True)
    print("LIGHTING BATCH COMPLETE", flush=True)


if __name__ == "__main__":
    main()
