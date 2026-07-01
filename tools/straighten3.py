#!/usr/bin/env python3
"""RoomCast ANGLE engine v3 — GeoCalib-grounded roll leveling (replaces hand-rolled line math).

Pipeline per room:
  1. GeoCalib.calibrate(before)  -> recover roll, pitch, focal + uncertainties (the camera MODEL,
     not a noisy Hough scalar). One trained model, same meaning across every photo.
  2. CONSERVATIVE roll-leveling homography (industry rules):
       - PARTIAL: ease off as |roll| grows (full at small angles, tapers toward big ones).
       - VERTICALS + LEVEL only (a pure in-plane rotation about the principal point; we do NOT
         force horizontals / add keystone — that is the 3-point case the industry declines).
       - DECLINE (return original untouched) when GeoCalib is low-confidence, or the tilt is
         extreme / not a clean roll case.
  3. DO-NO-HARM: build the FINAL image (warp -> crop warped border -> cover-crop to 4:3 via
     pipeline.to_43), measure verify_angle.lean on THAT exact image; if it is not at least
     ~0.3 deg plumber than the original's lean, REVERT to the untouched original.
     Worst case is therefore "unchanged", never "worse".
  4. The SAME homography is applied to the -before and the -lit twin so they stay pixel-aligned;
     written to FRESH filenames {stem}-angle3 / {stem}-lit-angle3 (never overwrites -angle/-lit-angle).

Run:  python tools/straighten3.py
"""
import os, sys, hashlib
import cv2, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pipeline import to_43                       # cover-crop to 4:3 (the exact shipped framing)
import verify_angle                              # lean() + sheet() — judged on the FINAL image

from geocalib import GeoCalib

ASSETS = os.path.join(ROOT, "assets", "loren-apt")
SCRATCH = "/private/tmp/claude-501/-Users-lorenpolster-Claude-Projects-roomcast/7d42c730-3fa8-4885-b304-3dea4c119ad7/scratchpad"
ROOMS = ["1-living", "2-kitchen", "3-kitchen2", "4-hall", "5-bath"]

# ---- conservative gates (industry-aligned) ----
ROLL_DEADZONE   = 0.25   # deg — below this the photo is already plumb; do nothing.
ROLL_EXTREME    = 12.0   # deg — above this it is not a routine roll fix; DECLINE.
ROLL_UNC_MAX    = 0.06   # rad — GeoCalib roll uncertainty above this = low confidence; DECLINE.
HARM_MARGIN     = 0.30   # deg — final image must beat the original's lean by at least this.
CROP_FRAC       = 0.06   # trim the replicate-smeared warped border before the 4:3 cover-crop.


def partial_strength(roll_abs):
    """Ease off as the tilt grows (Lightroom/Capture One default is partial, not 100%).
    Full correction for small tilts, tapering down so big tilts are only gently nudged."""
    if roll_abs <= 4.0:
        return 1.00
    if roll_abs >= ROLL_EXTREME:
        return 0.55
    # linear taper 1.00 -> 0.55 across [4, 12] deg
    return 1.00 - 0.45 * (roll_abs - 4.0) / (ROLL_EXTREME - 4.0)


def roll_homography(w, h, cx, cy, roll_deg):
    """Pure in-plane rotation by -roll about the principal point (a 2-point, verticals-only level).
    No keystone term — we never force horizontals."""
    th = np.radians(-roll_deg)
    c, s = np.cos(th), np.sin(th)
    T  = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float64)
    R  = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)
    Ti = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], np.float64)
    return Ti @ R @ T


def final_from(im, H):
    """Apply homography -> crop the warped border -> cover-crop to 4:3. THIS is what ships."""
    h, w = im.shape[:2]
    warped = cv2.warpPerspective(im, H, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    m = int(CROP_FRAC * max(h, w))
    warped = warped[m:h - m, m:w - m]
    return to_43(warped)


WEBP_Q = 92

def write_webp(path, im):
    cv2.imwrite(path, im, [cv2.IMWRITE_WEBP_QUALITY, WEBP_Q])

def webp_roundtrip(im):
    """Encode->decode as webp q92 so we measure the EXACT pixels that get shipped, not the
    in-memory buffer (the lossy re-encode shifts edges enough to move the Hough metric)."""
    ok, buf = cv2.imencode(".webp", im, [cv2.IMWRITE_WEBP_QUALITY, WEBP_Q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def md5_bytes(im):
    ok, buf = cv2.imencode(".png", im)
    return hashlib.md5(buf.tobytes()).hexdigest()[:10]


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = GeoCalib(weights="pinhole").to(dev)
    results = []

    for stem in ROOMS:
        before_p = os.path.join(ASSETS, f"{stem}-before.webp")
        lit_p    = os.path.join(ASSETS, f"{stem}-lit.webp")
        before = cv2.imread(before_p)
        lit    = cv2.imread(lit_p)
        h, w = before.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # 1) GeoCalib calibration on the ORIGINAL before image.
        img_t = model.load_image(before_p).to(dev)
        res = model.calibrate(img_t)
        roll  = float(np.degrees(res["gravity"].roll.item()))
        pitch = float(np.degrees(res["gravity"].pitch.item()))
        roll_unc = float(res["roll_uncertainty"].item())

        ident = np.eye(3)
        out_before = os.path.join(ASSETS, f"{stem}-angle3.webp")
        out_lit    = os.path.join(ASSETS, f"{stem}-lit-angle3.webp")

        # The original's lean is measured on the ORIGINAL's FINAL framing, ENCODED THE SAME WAY
        # the shipped file is (webp q92) — the lossy re-encode shifts edges enough to move the
        # Hough metric, so we must compare apples-to-apples on the post-webp pixels, never an
        # in-memory buffer (the exact "measure an image nobody ships" trap that sank v1).
        orig_final = final_from(before, ident)
        orig_final_webp = webp_roundtrip(orig_final)        # what "unchanged" actually ships as
        lean_before, nb = verify_angle.lean(orig_final_webp)

        roll_abs = abs(roll)
        status = None
        note = ""

        # 2) DECLINE gates (low confidence / dead zone / extreme).
        if roll_unc > ROLL_UNC_MAX:
            status, note = "declined", f"low confidence (roll_unc {roll_unc:.3f} > {ROLL_UNC_MAX})"
        elif roll_abs < ROLL_DEADZONE:
            status, note = "declined", f"already plumb (|roll| {roll_abs:.2f} < {ROLL_DEADZONE})"
        elif roll_abs > ROLL_EXTREME:
            status, note = "declined", f"extreme tilt (|roll| {roll_abs:.2f} > {ROLL_EXTREME})"

        if status == "declined":
            # Ship the unchanged original framing (as webp), both twins identity-transformed.
            write_webp(out_before, orig_final)
            write_webp(out_lit, final_from(lit, ident))
            lean_after = lean_before
            applied_roll = 0.0
        else:
            # 3) Conservative partial roll-leveling homography.
            strength = partial_strength(roll_abs)
            applied_roll = roll * strength
            H = roll_homography(w, h, cx, cy, applied_roll)
            cand_before = final_from(before, H)

            # DO-NO-HARM judged on the EXACT SHIPPED PIXELS: encode the candidate as webp,
            # re-read it, and measure lean on that — the same bytes the user receives.
            cand_webp = webp_roundtrip(cand_before)
            lean_after, na = verify_angle.lean(cand_webp)

            if lean_after <= lean_before - HARM_MARGIN:
                status = "corrected"
                note = f"strength {strength:.2f}, applied roll {applied_roll:+.2f}"
                write_webp(out_before, cand_before)
                write_webp(out_lit, final_from(lit, H))
            else:
                status = "reverted"
                note = (f"do-no-harm: shipped lean {lean_after:.2f} not >= "
                        f"{HARM_MARGIN} plumber than {lean_before:.2f}")
                write_webp(out_before, orig_final)            # revert to unchanged original
                write_webp(out_lit, final_from(lit, ident))
                lean_after = lean_before
                applied_roll = 0.0

        # 5) Contact sheet built from the EXACT shipped files (re-read from disk), md5 cache-bust.
        shipped = cv2.imread(out_before)
        md5 = md5_bytes(shipped)
        sheet_path = os.path.join(SCRATCH, f"G3_{stem}_{md5}.png")
        title = (f"{stem} | roll {roll:+.2f} pitch {pitch:+.2f} | {status} | "
                 f"lean {lean_before:.2f}->{lean_after:.2f}")
        verify_angle.sheet(orig_final_webp, shipped, title, sheet_path)

        results.append(dict(stem=stem, roll=roll, pitch=pitch, roll_unc=roll_unc,
                            status=status, note=note, lean_before=lean_before,
                            lean_after=lean_after, n_before=nb, applied_roll=applied_roll,
                            sheet=sheet_path))

    # ---- report table ----
    print("\n" + "=" * 96)
    print(f"{'room':11} {'roll':>6} {'pitch':>6} {'r_unc':>6} {'status':>10} "
          f"{'lean_b':>7} {'lean_a':>7} {'applied':>8}")
    print("-" * 96)
    for r in results:
        print(f"{r['stem']:11} {r['roll']:6.2f} {r['pitch']:6.2f} {r['roll_unc']:6.3f} "
              f"{r['status']:>10} {r['lean_before']:7.2f} {r['lean_after']:7.2f} "
              f"{r['applied_roll']:+8.2f}")
    print("=" * 96)
    for r in results:
        print(f"{r['stem']:11} {r['note']}")
        print(f"            sheet: {r['sheet']}")
    return results


if __name__ == "__main__":
    main()
