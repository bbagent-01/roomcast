#!/usr/bin/env python3
"""RoomCast ANGLE corrector v2 — conservative, guarded, verified on the FINAL image.

Done the industry way: robust vertical vanishing-point => keystone (fixes BOTH sides' convergence,
not just a roll) + residual roll level. KEYSTONE is applied PARTIALLY (full convergence-correction
over-warps); roll is applied near-fully (leveling is safe). DECLINES when there aren't enough lines.
DO-NO-HARM: measures lean on the final cropped image and reverts to the original if it isn't plumber.
Same homography is applied to the lit version so Lighting+Angle stay aligned.
"""
import cv2, numpy as np
from upright import _build_H, vanishing_vertical
from verify_angle import lean, long_verticals

KEYSTONE_PARTIAL = 0.80     # convergence correction is eased (full = over-warp)
ROLL_PARTIAL = 0.92         # leveling is safe to apply nearly fully
MIN_LINES = 6               # fewer than this -> decline
DO_NO_HARM = 0.3            # must improve lean by at least this many deg, else revert


def _lw_roll(im):
    lv = long_verticals(im)
    if not lv:
        return 0.0
    num = sum(a * L for a, L, _ in lv); den = sum(L for a, L, _ in lv)
    return num / den if den else 0.0


def _half_leans(im):
    """Length-weighted signed lean of long verticals in the LEFT vs RIGHT image half.
    Opposite signs = genuine perspective convergence (keystone); same sign = a roll."""
    w = im.shape[1]
    lv = long_verticals(im)
    def lw(side):
        sel = [(a, L) for a, L, c in lv if (((c[0] + c[2]) / 2 < w / 2) == side)]
        den = sum(L for _, L in sel)
        return (sum(a * L for a, L in sel) / den) if den else None
    return lw(True), lw(False)


def _rot(im, deg):
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def correct_params(im):
    """Find the roll that minimizes a SMOOTHED lean (robust to the metric's noise), or decline.
    A single noisy lean reading can't be trusted; the minimum of a densely-sampled, smoothed curve can.
    Keystone only when the two halves genuinely converge (opposite leans)."""
    h, w = im.shape[:2]; cx, cy = w / 2.0, h / 2.0
    if len(long_verticals(im)) < MIN_LINES:
        return None
    rolls = np.arange(-10.0, 6.01, 0.5)
    leans = np.array([lean(_rot(im, float(r)))[0] for r in rolls])
    sm = np.convolve(leans, np.array([0.25, 0.5, 0.25]), mode="same")   # smooth out the noise
    best_roll = float(rolls[int(np.argmin(sm))])
    # genuine-convergence check -> add a partial keystone
    k = 0.0
    lhh, rhh = _half_leans(im)
    if lhh is not None and rhh is not None and lhh * rhh < 0 and min(abs(lhh), abs(rhh)) > 1.5:
        vp, inliers, n = vanishing_vertical(im)
        if vp is not None and len(inliers) >= 4 and abs(vp[2]) > 1e-12:
            vyc = vp[1] / vp[2] - cy
            if abs(vyc) > 0.5 * h:
                k = float(np.clip(-1.0 / vyc, -0.0035, 0.0035)) * KEYSTONE_PARTIAL
    return best_roll * ROLL_PARTIAL, k


def _crop(im, roll, k):
    frac = min(0.05 + 1.15 * np.sin(np.radians(abs(roll))) + 220.0 * abs(k), 0.26)
    h, w = im.shape[:2]; mx, my = int(frac * w), int(frac * h)
    return im[my:h - my, mx:w - mx]


def straighten(before, lit):
    """Returns (before_out, lit_out, status). status in {corrected, declined, already-level, reverted}."""
    p = correct_params(before)
    if p is None:
        return None, None, "declined (too few/weak lines)"
    roll, k = p
    if abs(roll) < 0.4 and abs(k) < 5e-5:
        return None, None, "already-level"
    h, w = before.shape[:2]
    H = _build_H(w, h, roll, k)
    b = _crop(cv2.warpPerspective(before, H, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), roll, k)
    l = _crop(cv2.warpPerspective(lit, H, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), roll, k)
    before_lean = lean(before)[0]; after_lean = lean(b)[0]
    if after_lean > before_lean - DO_NO_HARM:          # not plumber -> do no harm
        return None, None, f"reverted (lean {before_lean:.1f}->{after_lean:.1f}, no gain)"
    return b, l, f"corrected roll{roll:+.2f} k{k:+.5f}  lean {before_lean:.1f}->{after_lean:.1f}"
