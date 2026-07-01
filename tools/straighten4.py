#!/usr/bin/env python3
"""RoomCast ANGLE engine v4 — pitch-driven UPRIGHT (two-point vertical / keystone) correction.

The v3 diagnosis: these wide-angle phone shots are NOT roll-tilted — GeoCalib reports roll ~0-2 deg
on all five. The walls CONVERGE because the camera is PITCHED DOWN 4-15 deg. A roll fix can't touch
that. v4 fixes the right thing: the standard "Vertical" / upright correction (what Lightroom's
Vertical mode does), grounded in GeoCalib's REAL recovered camera — no hand-guessed keystone scalar.

Per room:
  1. GeoCalib.calibrate(before) -> camera K + gravity (roll, pitch, and the rotation R that aligns
     the camera to gravity / straight-down).
  2. UPRIGHT homography H = K . R_partial . K^-1, where R_partial is the rotation that undoes a
     FRACTION of the measured roll+pitch (built from GeoCalib's own Gravity.from_rp, a true
     rotation — not a lerped matrix). Re-levels the camera so verticals become parallel/vertical.
  3. PARTIAL + adaptive: bigger pitch => gentler.  base 0.6, eased for large pitch:
        eff = 0.6 * clamp(1 - (|pitch| - 6)/30, 0.4, 1)
     Full pitch removal on a 12-15 deg downward tilt over-warps and over-crops; stay conservative.
  4. Same H to -before and -lit; crop the warped border (proportional to the warp magnitude);
     cover-crop 4:3 via pipeline.to_43.  Fresh names {stem}-angleK / {stem}-lit-angleK.
  5. DO-NO-HARM on the SHIPPED WebP bytes (encode->decode->measure — the v3 lesson). verify_angle.lean
     measures convergence, so a correct keystone SHOULD reduce it; if it doesn't, REVERT. Decline on
     low-confidence / dead-zone pitch.

Run:  python tools/straighten4.py
"""
import os, sys, hashlib
import cv2, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pipeline import to_43
import verify_angle

from geocalib import GeoCalib
from geocalib.gravity import Gravity

ASSETS = os.path.join(ROOT, "assets", "loren-apt")
SCRATCH = "/private/tmp/claude-501/-Users-lorenpolster-Claude-Projects-roomcast/7d42c730-3fa8-4885-b304-3dea4c119ad7/scratchpad"
ROOMS = ["1-living", "2-kitchen", "3-kitchen2", "4-hall", "5-bath"]

# ---- gates ----
PITCH_DEADZONE = 1.5     # deg — below this there is no meaningful convergence; decline.
PITCH_EXTREME  = 22.0    # deg — beyond this an upright fix over-warps badly; decline.
PITCH_UNC_MAX  = 0.08    # rad — GeoCalib pitch uncertainty above this = low confidence; decline.
HARM_MARGIN    = 0.30    # deg — shipped lean must beat the original by at least this.
WEBP_Q         = 92
REPORT_FACTORS = [0.5, 0.65, 0.8]   # tradeoff readout
CHOSEN_NOTE    = "base 0.6 eased by pitch"


def effective_partial(pitch_abs, base=0.6):
    """base 0.6, eased further for large pitch so a steep downward tilt is only gently uprighted."""
    return base * float(np.clip(1.0 - (pitch_abs - 6.0) / 30.0, 0.4, 1.0))


def upright_H(K, roll_rad, pitch_rad, frac):
    """H = K . R^T . K^-1.  GeoCalib's Gravity.R is the camera-from-world rotation (world gravity
    expressed in the camera frame); to re-render the image as the world-upright camera would see
    it we apply the INVERSE rotation R^T (world-from-camera). Verified directly on the convergence
    metric: R^T collapses vertical convergence, plain R amplifies it. R is built from a FRACTION of
    the measured roll+pitch via GeoCalib's own gravity parametrisation (a true rotation)."""
    g = Gravity.from_rp(roll_rad * frac, pitch_rad * frac)
    R = g.R.numpy().reshape(3, 3)
    H = K @ R.T @ np.linalg.inv(K)
    return H / H[2, 2]


def crop_frac_for(frac, pitch_abs):
    """Crop the warped border proportional to the warp magnitude (more upright => more border)."""
    return float(np.clip(0.04 + 0.006 * pitch_abs * frac, 0.04, 0.16))


def final_from(im, H, crop_frac):
    h, w = im.shape[:2]
    warped = cv2.warpPerspective(im, H, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    m = int(crop_frac * max(h, w))
    warped = warped[m:h - m, m:w - m]
    return to_43(warped)


def write_webp(path, im):
    cv2.imwrite(path, im, [cv2.IMWRITE_WEBP_QUALITY, WEBP_Q])

def webp_roundtrip(im):
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

        # 1) calibrate
        res = model.calibrate(model.load_image(before_p).to(dev))
        g = res["gravity"].cpu()
        roll_rad  = float(g.roll.item())
        pitch_rad = float(g.pitch.item())
        roll  = float(np.degrees(roll_rad))
        pitch = float(np.degrees(pitch_rad))
        pitch_abs = abs(pitch)
        pitch_unc = float(res["pitch_uncertainty"].item())
        K = res["camera"].K.cpu().numpy().reshape(3, 3)

        out_before = os.path.join(ASSETS, f"{stem}-angleK.webp")
        out_lit    = os.path.join(ASSETS, f"{stem}-lit-angleK.webp")

        # original final framing, measured as the SHIPPED webp would be (the v3 lesson)
        orig_final = to_43(before)
        orig_final_webp = webp_roundtrip(orig_final)
        lean_before, nb = verify_angle.lean(orig_final_webp)

        # tradeoff readout: lean at 0.5 / 0.65 / 0.8 (shipped-webp pixels)
        factor_leans = {}
        for f in REPORT_FACTORS:
            Hf = upright_H(K, roll_rad, pitch_rad, f)
            cf = final_from(before, Hf, crop_frac_for(f, pitch_abs))
            factor_leans[f] = round(verify_angle.lean(webp_roundtrip(cf))[0], 2)

        # 2) decline gates
        eff = effective_partial(pitch_abs)
        status = None; note = ""
        if pitch_unc > PITCH_UNC_MAX:
            status, note = "declined", f"low confidence (pitch_unc {pitch_unc:.3f} > {PITCH_UNC_MAX})"
        elif pitch_abs < PITCH_DEADZONE:
            status, note = "declined", f"no convergence (|pitch| {pitch_abs:.2f} < {PITCH_DEADZONE})"
        elif pitch_abs > PITCH_EXTREME:
            status, note = "declined", f"extreme pitch (|pitch| {pitch_abs:.2f} > {PITCH_EXTREME})"

        if status == "declined":
            write_webp(out_before, orig_final)
            write_webp(out_lit, to_43(lit))
            lean_after = lean_before
            eff_used = 0.0
        else:
            cf = crop_frac_for(eff, pitch_abs)
            H = upright_H(K, roll_rad, pitch_rad, eff)
            cand_before = final_from(before, H, cf)
            lean_after, na = verify_angle.lean(webp_roundtrip(cand_before))

            if lean_after <= lean_before - HARM_MARGIN:
                status = "corrected"
                note = f"eff {eff:.2f} ({CHOSEN_NOTE}), crop {cf:.3f}"
                write_webp(out_before, cand_before)
                write_webp(out_lit, final_from(lit, H, cf))
                eff_used = eff
            else:
                status = "reverted"
                note = (f"do-no-harm: shipped lean {lean_after:.2f} not >= "
                        f"{HARM_MARGIN} plumber than {lean_before:.2f}")
                write_webp(out_before, orig_final)
                write_webp(out_lit, to_43(lit))
                lean_after = lean_before
                eff_used = 0.0

        shipped = cv2.imread(out_before)
        md5 = md5_bytes(shipped)
        sheet_path = os.path.join(SCRATCH, f"G4_{stem}_{md5}.png")
        title = (f"{stem} | pitch {pitch:+.1f} | eff {eff_used:.2f} | {status} | "
                 f"lean {lean_before:.2f}->{lean_after:.2f}")
        verify_angle.sheet(orig_final_webp, shipped, title, sheet_path)

        results.append(dict(stem=stem, roll=roll, pitch=pitch, pitch_unc=pitch_unc,
                            eff=eff_used, status=status, note=note,
                            lean_before=lean_before, lean_after=lean_after,
                            factor_leans=factor_leans, sheet=sheet_path))

    # ---- report ----
    print("\n" + "=" * 104)
    print(f"{'room':11} {'pitch':>6} {'p_unc':>6} {'eff':>5} {'status':>10} "
          f"{'lean_b':>7} {'lean_a':>7} | {'f0.5':>6} {'f0.65':>6} {'f0.8':>6}")
    print("-" * 104)
    for r in results:
        fl = r["factor_leans"]
        print(f"{r['stem']:11} {r['pitch']:6.1f} {r['pitch_unc']:6.3f} {r['eff']:5.2f} "
              f"{r['status']:>10} {r['lean_before']:7.2f} {r['lean_after']:7.2f} | "
              f"{fl[0.5]:6.2f} {fl[0.65]:6.2f} {fl[0.8]:6.2f}")
    print("=" * 104)
    for r in results:
        print(f"{r['stem']:11} {r['note']}")
        print(f"            sheet: {r['sheet']}")
    return results


if __name__ == "__main__":
    main()
