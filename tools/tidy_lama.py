#!/usr/bin/env python3
"""RoomCast TIDY engine — honest, remove-only erase via LaMa (local, free).

Masked content-aware fill: paint the clutter white in a mask, LaMa reconstructs the surface behind it
from surrounding pixels. Everything OUTSIDE the mask is pixel-identical -> honest by construction
(can't add decor, can't swap furniture; it only removes). This is the correct Tidy engine after the
generative -clean assets were rejected for reinventing the room.

Produces {stem}-tidy.webp (the FULLY tidied layer) + a before/after compare. The web app reveals this
layer only inside the user's drag-selected boxes, so the user clears exactly what they point at.

Usage:  ~/.roomcast-ml/bin/python tools/tidy_lama.py <base-stem>   e.g. 1-living-lit-angleL
"""
import os, sys
import cv2, numpy as np, torch
from PIL import Image
# big-lama.pt was traced on CUDA; force the jit load onto CPU on this Mac.
_ORIG_JIT_LOAD = torch.jit.load
torch.jit.load = lambda *a, **k: _ORIG_JIT_LOAD(*a, **{**k, "map_location": "cpu"})
from simple_lama_inpainting import SimpleLama

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "assets", "loren-apt")
SCRATCH = "/private/tmp/claude-501/-Users-lorenpolster-Claude-Projects-roomcast/7d42c730-3fa8-4885-b304-3dea4c119ad7/scratchpad"
WEBP_Q = 92

# clutter masks per base image: list of (x0,y0,x1,y1) rects (feathered), in that image's pixel space
MASKS = {
    "1-living-lit-angleL": [
        (416, 668, 694, 792),   # magazines/papers strewn on rug + front table edge
        (556, 686, 688, 726),   # items on the marble table top (bottle, papers)
    ],
}


def build_mask(shape, rects, feather=9):
    m = np.zeros(shape[:2], np.uint8)
    for x0, y0, x1, y1 in rects:
        cv2.rectangle(m, (x0, y0), (x1, y1), 255, -1)
    if feather:
        m = cv2.dilate(m, np.ones((feather, feather), np.uint8), 1)
        m = cv2.GaussianBlur(m, (0, 0), feather / 3.0)
        m = (m > 40).astype(np.uint8) * 255
    return m


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else "1-living-lit-angleL"
    base_p = os.path.join(ASSETS, f"{stem}.webp")
    base = cv2.imread(base_p)
    rects = MASKS[stem]
    mask = build_mask(base.shape, rects)

    lama = SimpleLama(torch.device("cpu"))
    img_pil = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask)
    fill_pil = lama(img_pil, mask_pil)
    fill = cv2.cvtColor(np.array(fill_pil), cv2.COLOR_RGB2BGR)
    fill = cv2.resize(fill, (base.shape[1], base.shape[0]))   # LaMa pads to /8; restore exact size

    # COMPOSITE: keep original pixels everywhere; paste LaMa's fill ONLY inside the mask.
    # Honest by construction — outside the mask is byte-for-byte the original (can't add/alter anything).
    alpha = (cv2.GaussianBlur(mask, (0, 0), 4.0).astype(np.float32) / 255.0)[..., None]
    out = (base.astype(np.float32) * (1 - alpha) + fill.astype(np.float32) * alpha).astype(np.uint8)

    # honesty check: outside the (feathered) mask must be identical
    m3 = (cv2.GaussianBlur(mask, (0, 0), 4.0) > 8)
    diff = np.abs(base.astype(int) - out.astype(int)).sum(2)
    outside_changed = int((diff[~m3] > 2).sum())
    outside_total = int((~m3).sum())

    out_p = os.path.join(ASSETS, f"{stem}-tidy.webp")
    cv2.imwrite(out_p, out, [cv2.IMWRITE_WEBP_QUALITY, WEBP_Q])

    # compare sheet + mask overlay
    def lab(im, t):
        im = im.copy(); cv2.rectangle(im, (0, 0), (im.shape[1], 30), (250, 247, 241), -1)
        cv2.putText(im, t, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 34, 28), 2); return im
    ov = base.copy(); ov[m3] = (0.4 * ov[m3] + 0.6 * np.array([40, 40, 230])).astype(np.uint8)
    h = 460
    def rs(im): return cv2.resize(im, (int(h * im.shape[1] / im.shape[0]), h))
    sheet = np.hstack([lab(rs(ov), "MASK (clutter)"), np.full((h, 6, 3), 230, np.uint8),
                       lab(rs(out), "TIDIED (remove-only)")])
    sp = os.path.join(SCRATCH, f"TIDYLAMA_{stem}.png")
    cv2.imwrite(sp, sheet)

    print(f"wrote {out_p}")
    print(f"outside-mask changed px: {outside_changed}/{outside_total} "
          f"({100*outside_changed/outside_total:.3f}%)  [want ~0 => honest]")
    print(f"sheet: {sp}")


if __name__ == "__main__":
    main()
