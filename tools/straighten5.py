#!/usr/bin/env python3
"""RoomCast ANGLE engine v5 — decoupled upright + empirical residual LEVEL (verticals AND horizontals).

Loren's ask after approving v4 ("better and much more consistent"): push the verticals a touch
straighter, AND rotate so the HORIZONTALS are horizontal too — the combined straighten+level he's
seen work perfectly before. v4 applied GeoCalib's rotation at ONE partial strength to roll+pitch
together, which under-levels horizontals (roll leveling is safe near-full) and leaves verticals shy.

v5 does two things v4 did not:
  1. DECOUPLE roll vs pitch. Upright warp = K . R(roll*roll_frac, pitch*pitch_frac)^T . K^-1 with
     roll_frac ~1.0 (leveling is cheap/safe) and pitch_frac bumped a touch over v4 (verticals a bit
     more vertical). Pitch stays partial+eased so a steep downward tilt never over-warps.
  2. EMPIRICAL RESIDUAL LEVEL. After the upright warp, densely sweep a small in-plane rotation and
     pick the one that minimises a length-weighted, smoothed COMBINED off-axis deviation of BOTH long
     near-vertical lines (want vertical) AND long near-horizontal lines (want horizontal). Fully
     measured on the rotated pixels — no sign derivation, no camera-model guess. This is the literal
     "rotate the angle so the horizontals are also horizontal" step.

DO-NO-HARM on the SHIPPED WebP bytes: never ship verticals worse; require a net gain (verticals
better, or verticals no-worse and horizontals better). Fresh names {stem}-angleL / {stem}-lit-angleL
so the approved v4 -angleK set is never overwritten (stale-cache lesson).

Run:  ~/.roomcast-ml/bin/python tools/straighten5.py
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
PITCH_DEADZONE = 1.5
PITCH_EXTREME  = 22.0
PITCH_UNC_MAX  = 0.08
WEBP_Q         = 92

# ---- strengths ----
ROLL_FRAC   = 1.00      # leveling is safe -> apply GeoCalib's full recovered roll
LEVEL_RANGE = 3.5       # deg — residual-level search half-range (clamped so it can't run away)
LEVEL_STEP  = 0.25
LEVEL_VGUARD = 0.5      # deg — a leveling rotation may not raise vertical lean past this
NEAR_AXIS   = 20.0      # deg — a line counts as "near vertical/horizontal" within this


def pitch_frac_for(pitch_abs, base=0.72):
    """Bumped over v4's 0.6 and eased less: verticals a touch more vertical, still gentle when steep."""
    return base * float(np.clip(1.0 - (pitch_abs - 6.0) / 40.0, 0.55, 1.0))


def upright_H(K, roll_rad, pitch_rad, roll_frac, pitch_frac):
    g = Gravity.from_rp(roll_rad * roll_frac, pitch_rad * pitch_frac)
    R = g.R.numpy().reshape(3, 3)
    H = K @ R.T @ np.linalg.inv(K)
    return H / H[2, 2]


# ---- line analysis (both axes) ----
def _long_lines(im, minfrac=0.15):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, w = im.shape[:2]
    lines = cv2.HoughLinesP(cv2.Canny(g, 50, 150), 1, np.pi / 180, 80,
                            minLineLength=int(minfrac * min(h, w)), maxLineGap=12)
    V, Hz = [], []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            L = float(np.hypot(x2 - x1, y2 - y1))
            av = np.degrees(np.arctan2(x2 - x1, y2 - y1))   # 0 = vertical
            av = (av + 90) % 180 - 90
            ah = np.degrees(np.arctan2(y2 - y1, x2 - x1))   # 0 = horizontal
            ah = (ah + 90) % 180 - 90
            if abs(av) < NEAR_AXIS:
                V.append((av, L, (x1, y1, x2, y2)))
            elif abs(ah) < NEAR_AXIS:
                Hz.append((ah, L, (x1, y1, x2, y2)))
    return V, Hz


def combined_dev(im):
    """Length-weighted mean |off-axis| over long near-vertical + near-horizontal lines (deg)."""
    V, Hz = _long_lines(im)
    num = sum(abs(a) * L for a, L, _ in V) + sum(abs(a) * L for a, L, _ in Hz)
    den = sum(L for _, L, _ in V) + sum(L for _, L, _ in Hz)
    return (num / den if den else 0.0)


def axis_leans(im):
    """(vertical lean, horizontal lean) — length-weighted mean |deviation| per axis, deg."""
    V, Hz = _long_lines(im)
    def lw(sel):
        den = sum(L for _, L, _ in sel)
        return (sum(abs(a) * L for a, L, _ in sel) / den) if den else 0.0
    return lw(V), lw(Hz)


def _rotate(im, deg):
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    return cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def find_level(warped_pre, pitch_abs, pf):
    """Find the small in-plane rotation that LEVELS THE HORIZONTALS without degrading verticals.
    The upright warp already handled vertical convergence; a rotation cannot fix convergence, so we
    must NOT let the (still slightly converging) verticals drive it — that hijacked v5a to the clamp
    and tilted the horizontals worse. Here horizontals are primary; verticals are a hard guard + a
    tiny tiebreak; smallest adequate angle wins. Graded on the EXACT final framing (crop + to_43) so
    the leveler optimises the pixels we actually ship (no pre-crop/post-crop mismatch). Sign-proof."""
    h, w = warped_pre.shape[:2]
    cf = crop_frac_for(pitch_abs, pf, 2.0)             # constant framing across the sweep
    m = int(cf * max(h, w))
    def final_at(a):
        r = _rotate(warped_pre, float(a)) if abs(a) > 1e-3 else warped_pre
        return to_43(r[m:h - m, m:w - m])
    v0, _ = axis_leans(final_at(0.0))
    angs = np.arange(-LEVEL_RANGE, LEVEL_RANGE + 1e-6, LEVEL_STEP)
    best, best_cost = 0.0, 1e9
    for a in angs:
        vl, hl = axis_leans(final_at(a))
        if vl > v0 + LEVEL_VGUARD:                     # never worsen verticals
            continue
        cost = hl + 0.20 * vl + 0.05 * abs(a)          # level H first; nudge V; prefer small angle
        if cost < best_cost:
            best_cost, best = cost, float(a)
    return best


def crop_frac_for(pitch_abs, pitch_frac, level_deg):
    return float(np.clip(0.04 + 0.006 * pitch_abs * pitch_frac + 0.010 * abs(level_deg), 0.04, 0.18))


def build_final(im, H, level_deg, crop_frac):
    h, w = im.shape[:2]
    warped = cv2.warpPerspective(im, H, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if abs(level_deg) > 1e-3:
        warped = _rotate(warped, level_deg)
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


# ---- richer overlay: both grids + detected lines colored by axis ----
CYAN = (210, 180, 40); RED = (40, 40, 230); GRN = (60, 190, 60); INK = (40, 34, 28)

def overlay2(im):
    o = im.copy(); h, w = o.shape[:2]
    grid = o.copy()
    for i in range(1, 8):
        cv2.line(grid, (int(i * w / 8), 0), (int(i * w / 8), h), CYAN, 1)
        cv2.line(grid, (0, int(i * h / 8)), (w, int(i * h / 8)), CYAN, 1)
    cv2.addWeighted(grid, 0.4, o, 0.6, 0, o)
    V, Hz = _long_lines(im)
    for a, L, (x1, y1, x2, y2) in V:
        cv2.line(o, (x1, y1), (x2, y2), RED, 3)      # verticals red
    for a, L, (x1, y1, x2, y2) in Hz:
        cv2.line(o, (x1, y1), (x2, y2), GRN, 3)      # horizontals green
    vl, hl = axis_leans(im)
    cv2.rectangle(o, (0, 0), (w, 40), (250, 247, 241), -1)
    cv2.putText(o, f"V lean {vl:.1f}  H lean {hl:.1f}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, INK, 2, cv2.LINE_AA)
    return o

def sheet2(before, after, title, out_path, disp_w=560):
    def prep(im, label):
        o = overlay2(im); h, w = o.shape[:2]
        o = cv2.resize(o, (disp_w, int(disp_w * h / w)))
        bar = np.full((32, disp_w, 3), 250, np.uint8)
        cv2.putText(bar, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INK, 2, cv2.LINE_AA)
        return np.vstack([bar, o])
    a = prep(before, "ORIGINAL"); b = prep(after, "V5  straighten + level")
    H = max(a.shape[0], b.shape[0])
    a = np.pad(a, ((0, H - a.shape[0]), (0, 0), (0, 0)), constant_values=250)
    b = np.pad(b, ((0, H - b.shape[0]), (0, 0), (0, 0)), constant_values=250)
    gap = np.full((H, 8, 3), 230, np.uint8)
    strip = np.hstack([a, gap, b])
    tb = np.full((38, strip.shape[1], 3), 255, np.uint8)
    cv2.putText(tb, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, INK, 2, cv2.LINE_AA)
    cv2.imwrite(out_path, np.vstack([tb, strip]))
    return out_path


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = GeoCalib(weights="pinhole").to(dev)
    results = []

    for stem in ROOMS:
        before_p = os.path.join(ASSETS, f"{stem}-before.webp")
        lit_p    = os.path.join(ASSETS, f"{stem}-lit.webp")
        before = cv2.imread(before_p)
        lit    = cv2.imread(lit_p)

        res = model.calibrate(model.load_image(before_p).to(dev))
        g = res["gravity"].cpu()
        roll_rad  = float(g.roll.item()); pitch_rad = float(g.pitch.item())
        roll  = float(np.degrees(roll_rad)); pitch = float(np.degrees(pitch_rad))
        pitch_abs = abs(pitch)
        pitch_unc = float(res["pitch_uncertainty"].item())
        K = res["camera"].K.cpu().numpy().reshape(3, 3)

        out_before = os.path.join(ASSETS, f"{stem}-angleL.webp")
        out_lit    = os.path.join(ASSETS, f"{stem}-lit-angleL.webp")

        orig_final = to_43(before)
        orig_webp  = webp_roundtrip(orig_final)
        vlean_b, hlean_b = axis_leans(orig_webp)

        pf = pitch_frac_for(pitch_abs)
        status = None; note = ""; level = 0.0; eff_pf = 0.0
        if pitch_unc > PITCH_UNC_MAX:
            status, note = "declined", f"low confidence (pitch_unc {pitch_unc:.3f})"
        elif pitch_abs > PITCH_EXTREME:
            status, note = "declined", f"extreme pitch (|pitch| {pitch_abs:.2f})"

        if status == "declined":
            write_webp(out_before, orig_final); write_webp(out_lit, to_43(lit))
            vlean_a, hlean_a = vlean_b, hlean_b
        else:
            # upright warp (decoupled roll/pitch), then find residual level on the WARPED before
            H = upright_H(K, roll_rad, pitch_rad, ROLL_FRAC, pf)
            hb, wb = before.shape[:2]
            warped_pre = cv2.warpPerspective(before, H, (wb, hb), flags=cv2.INTER_CUBIC,
                                             borderMode=cv2.BORDER_REPLICATE)
            level = find_level(warped_pre, pitch_abs, pf)
            cf = crop_frac_for(pitch_abs, pf, level)
            cand_before = build_final(before, H, level, cf)
            cand_webp = webp_roundtrip(cand_before)
            vlean_a, hlean_a = axis_leans(cand_webp)

            v_ok    = vlean_a <= vlean_b + 0.15                 # never make verticals worse
            net_gain = (vlean_a <= vlean_b - 0.30) or \
                       (v_ok and hlean_a <= hlean_b - 0.30)      # improve V, or level H w/o hurting V
            if v_ok and net_gain:
                status = "corrected"
                note = f"pitch_frac {pf:.2f}, roll_frac {ROLL_FRAC:.2f}, level {level:+.2f}, crop {cf:.3f}"
                write_webp(out_before, cand_before)
                write_webp(out_lit, build_final(lit, H, level, cf))
                eff_pf = pf
            else:
                status = "reverted"
                note = (f"do-no-harm: V {vlean_b:.2f}->{vlean_a:.2f}, H {hlean_b:.2f}->{hlean_a:.2f} "
                        f"(no net gain)")
                write_webp(out_before, orig_final); write_webp(out_lit, to_43(lit))
                vlean_a, hlean_a = vlean_b, hlean_b; level = 0.0

        shipped = cv2.imread(out_before)
        md5 = md5_bytes(shipped)
        sheet_path = os.path.join(SCRATCH, f"G5_{stem}_{md5}.png")
        title = (f"{stem} | pitch {pitch:+.1f} roll {roll:+.1f} | pf {eff_pf:.2f} lvl {level:+.2f} | "
                 f"{status} | V {vlean_b:.1f}->{vlean_a:.1f}  H {hlean_b:.1f}->{hlean_a:.1f}")
        sheet2(orig_webp, shipped, title, sheet_path)

        results.append(dict(stem=stem, roll=roll, pitch=pitch, pitch_unc=pitch_unc,
                            pf=eff_pf, level=level, status=status, note=note,
                            vlean_b=vlean_b, vlean_a=vlean_a, hlean_b=hlean_b, hlean_a=hlean_a,
                            sheet=sheet_path))

    print("\n" + "=" * 108)
    print(f"{'room':11} {'pitch':>6} {'roll':>6} {'pf':>5} {'level':>6} {'status':>10} "
          f"{'V_b':>6} {'V_a':>6} {'H_b':>6} {'H_a':>6}")
    print("-" * 108)
    for r in results:
        print(f"{r['stem']:11} {r['pitch']:6.1f} {r['roll']:6.1f} {r['pf']:5.2f} {r['level']:6.2f} "
              f"{r['status']:>10} {r['vlean_b']:6.2f} {r['vlean_a']:6.2f} {r['hlean_b']:6.2f} {r['hlean_a']:6.2f}")
    print("=" * 108)
    for r in results:
        print(f"{r['stem']:11} {r['note']}")
        print(f"            sheet: {r['sheet']}")
    return results


if __name__ == "__main__":
    main()
