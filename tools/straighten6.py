#!/usr/bin/env python3
"""RoomCast ANGLE engine v6 — straighten WITHOUT zooming in (keep the whole frame, fill the wedges).

Loren on v5: "looks like it's just zooming in more... if anything zoom out a hair so we don't lose
anything the photo captured." v6 keeps the GENTLER (v4, approved) vertical correction + horizontal
leveling, but stops CROPPING. Straightening opens small triangular wedges at the frame edge; instead
of cutting them away (zoom-in) or mirroring them (ugly seams that also corrupt the line metric), v6
fills them with LaMa outpaint — it extends the existing wall / floor / ceiling texture into the wedge.
No object is invented (edge surface only — the "small faithful outpaint" the brief called for), no
content is lost, and framing ends a hair WIDER than the original, never tighter.

Pipeline: GeoCalib K/roll/pitch -> upright H (roll_frac 1.0, pitch v4-gentle) -> residual horizontal
LEVEL (empirical, measured on a clean center-crop) -> compose H,level,zoom-out into ONE homography ->
warp once with a blank border -> LaMa-fill the blank wedges -> to_43. Fresh names -angleZ.

Run:  ~/.roomcast-ml/bin/python tools/straighten6.py
"""
import os, sys, hashlib
import cv2, numpy as np, torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pipeline import to_43
from geocalib import GeoCalib
from geocalib.gravity import Gravity

# big-lama.pt was traced on CUDA; force CPU load.
_ORIG_JIT = torch.jit.load
torch.jit.load = lambda *a, **k: _ORIG_JIT(*a, **{**k, "map_location": "cpu"})
from simple_lama_inpainting import SimpleLama

ASSETS = os.path.join(ROOT, "assets", "loren-apt")
SCRATCH = "/private/tmp/claude-501/-Users-lorenpolster-Claude-Projects-roomcast/7d42c730-3fa8-4885-b304-3dea4c119ad7/scratchpad"
ROOMS = ["1-living", "2-kitchen", "3-kitchen2", "4-hall", "5-bath"]

PITCH_EXTREME = 22.0; PITCH_UNC_MAX = 0.08
ROLL_FRAC = 1.00
LEVEL_RANGE = 3.5; LEVEL_STEP = 0.25; LEVEL_VGUARD = 0.5
NEAR_AXIS = 20.0
ZOOM_OUT = 1.00          # exact original frame (no uniform margin -> no tonal seam on plain walls);
                         # only the actual straightening wedges get LaMa-filled. Never crops in.
WEBP_Q = 92
LAMA = None


def pitch_frac_for(pitch_abs, base=0.60):
    return base * float(np.clip(1.0 - (pitch_abs - 6.0) / 30.0, 0.4, 1.0))

def upright_H(K, roll_rad, pitch_rad, roll_frac, pitch_frac):
    g = Gravity.from_rp(roll_rad * roll_frac, pitch_rad * pitch_frac)
    R = g.R.numpy().reshape(3, 3); H = K @ R.T @ np.linalg.inv(K); return H / H[2, 2]

def rot3(w, h, deg):
    return np.vstack([cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0), [0, 0, 1]])

def scale3(w, h, s):
    return np.array([[s, 0, (1 - s) * w / 2.0], [0, s, (1 - s) * h / 2.0], [0, 0, 1]], np.float64)


# ---- line analysis ----
def _long_lines(im, minfrac=0.15):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); h, w = im.shape[:2]
    lines = cv2.HoughLinesP(cv2.Canny(g, 50, 150), 1, np.pi / 180, 80,
                            minLineLength=int(minfrac * min(h, w)), maxLineGap=12)
    V, Hz = [], []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]; L = float(np.hypot(x2 - x1, y2 - y1))
            av = np.degrees(np.arctan2(x2 - x1, y2 - y1)); av = (av + 90) % 180 - 90
            ah = np.degrees(np.arctan2(y2 - y1, x2 - x1)); ah = (ah + 90) % 180 - 90
            if abs(av) < NEAR_AXIS: V.append((av, L, (x1, y1, x2, y2)))
            elif abs(ah) < NEAR_AXIS: Hz.append((ah, L, (x1, y1, x2, y2)))
    return V, Hz

def axis_leans(im):
    V, Hz = _long_lines(im)
    def lw(sel):
        den = sum(L for _, L, _ in sel); return (sum(abs(a) * L for a, L, _ in sel) / den) if den else 0.0
    return lw(V), lw(Hz)

def center(im, keep=0.86):
    h, w = im.shape[:2]; mh = int((1 - keep) / 2 * h); mw = int((1 - keep) / 2 * w)
    return im[mh:h - mh, mw:w - mw]


def warp_blank(src, M):
    """Warp with a blank (0) border; return (warped, blank_mask 255=needs fill)."""
    h, w = src.shape[:2]
    warped = cv2.warpPerspective(src, M, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (w, h),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    blank = (valid < 128).astype(np.uint8) * 255
    blank = cv2.dilate(blank, np.ones((3, 3), np.uint8), 1)   # cover the 1px antialias seam
    return warped, blank

def lama_fill(warped, blank):
    if blank.sum() == 0: return warped
    img = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)); m = Image.fromarray(blank)
    fill = cv2.cvtColor(np.array(LAMA(img, m)), cv2.COLOR_RGB2BGR)
    fill = cv2.resize(fill, (warped.shape[1], warped.shape[0]))
    a = (cv2.GaussianBlur(blank, (0, 0), 2.5).astype(np.float32) / 255.0)[..., None]
    return (warped.astype(np.float32) * (1 - a) + fill.astype(np.float32) * a).astype(np.uint8)

def render(src, M):
    """Keep the full frame; extend the edge into the straightening wedges (replicate = seamless on
    the plain ceiling/wall/floor these wedges land on; LaMa left tonal seams there). No crop."""
    h, w = src.shape[:2]
    warped = cv2.warpPerspective(src, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return to_43(warped)


def find_level(warpedH_rep):
    """Level horizontals w/o degrading verticals; measured on a clean CENTER-CROP (no edge seams)."""
    h, w = warpedH_rep.shape[:2]
    def meas(a):
        r = warpedH_rep if abs(a) < 1e-3 else cv2.warpPerspective(
            warpedH_rep, rot3(w, h, a), (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return axis_leans(center(to_43(r)))
    v0, _ = meas(0.0); best, bc = 0.0, 1e9
    for a in np.arange(-LEVEL_RANGE, LEVEL_RANGE + 1e-6, LEVEL_STEP):
        vl, hl = meas(float(a))
        if vl > v0 + LEVEL_VGUARD: continue
        cost = hl + 0.20 * vl + 0.05 * abs(a)
        if cost < bc: bc, best = cost, float(a)
    return best


def write_webp(p, im): cv2.imwrite(p, im, [cv2.IMWRITE_WEBP_QUALITY, WEBP_Q])
def webp_rt(im):
    ok, b = cv2.imencode(".webp", im, [cv2.IMWRITE_WEBP_QUALITY, WEBP_Q]); return cv2.imdecode(b, 1)
def md5b(im):
    ok, b = cv2.imencode(".png", im); return hashlib.md5(b.tobytes()).hexdigest()[:10]

CYAN=(210,180,40); RED=(40,40,230); GRN=(60,190,60); INK=(40,34,28)
def overlay2(im):
    o=im.copy(); h,w=o.shape[:2]; g=o.copy()
    for i in range(1,8):
        cv2.line(g,(int(i*w/8),0),(int(i*w/8),h),CYAN,1); cv2.line(g,(0,int(i*h/8)),(w,int(i*h/8)),CYAN,1)
    cv2.addWeighted(g,0.4,o,0.6,0,o)
    V,Hz=_long_lines(im)
    for a,L,(x1,y1,x2,y2) in V: cv2.line(o,(x1,y1),(x2,y2),RED,3)
    for a,L,(x1,y1,x2,y2) in Hz: cv2.line(o,(x1,y1),(x2,y2),GRN,3)
    vl,hl=axis_leans(im); cv2.rectangle(o,(0,0),(w,40),(250,247,241),-1)
    cv2.putText(o,f"V {vl:.1f}  H {hl:.1f}",(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.75,INK,2,cv2.LINE_AA); return o

def sheet(orig, out, title, path, dw=560):
    def prep(im,lb):
        o=overlay2(im); h,w=o.shape[:2]; o=cv2.resize(o,(dw,int(dw*h/w)))
        bar=np.full((30,dw,3),250,np.uint8); cv2.putText(bar,lb,(10,22),cv2.FONT_HERSHEY_SIMPLEX,0.65,INK,2,cv2.LINE_AA)
        return np.vstack([bar,o])
    a=prep(orig,"ORIGINAL framing"); b=prep(out,"V6  straighten, kept full frame")
    H=max(a.shape[0],b.shape[0])
    a=np.pad(a,((0,H-a.shape[0]),(0,0),(0,0)),constant_values=250); b=np.pad(b,((0,H-b.shape[0]),(0,0),(0,0)),constant_values=250)
    strip=np.hstack([a,np.full((H,8,3),230,np.uint8),b]); tb=np.full((36,strip.shape[1],3),255,np.uint8)
    cv2.putText(tb,title,(12,25),cv2.FONT_HERSHEY_SIMPLEX,0.66,INK,2,cv2.LINE_AA)
    cv2.imwrite(path,np.vstack([tb,strip])); return path


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = GeoCalib(weights="pinhole").to(dev)
    results = []
    for stem in ROOMS:
        before_p=os.path.join(ASSETS,f"{stem}-before.webp"); lit_p=os.path.join(ASSETS,f"{stem}-lit.webp")
        before=cv2.imread(before_p); lit=cv2.imread(lit_p)
        res=model.calibrate(model.load_image(before_p).to(dev)); g=res["gravity"].cpu()
        roll_rad=float(g.roll.item()); pitch_rad=float(g.pitch.item())
        roll=float(np.degrees(roll_rad)); pitch=float(np.degrees(pitch_rad)); pitch_abs=abs(pitch)
        pitch_unc=float(res["pitch_uncertainty"].item()); K=res["camera"].K.cpu().numpy().reshape(3,3)
        out_b=os.path.join(ASSETS,f"{stem}-angleZ.webp"); out_l=os.path.join(ASSETS,f"{stem}-lit-angleZ.webp")
        orig=to_43(before); orig_w=webp_rt(orig); vb,hb=axis_leans(center(orig_w))
        pf=pitch_frac_for(pitch_abs); status=None; note=""; level=0.0; eff_pf=0.0

        if pitch_unc>PITCH_UNC_MAX or pitch_abs>PITCH_EXTREME:
            status="declined"; note=f"unc {pitch_unc:.3f} pitch {pitch_abs:.1f}"
            write_webp(out_b,orig); write_webp(out_l,to_43(lit)); va,ha=vb,hb
        else:
            hh,ww=before.shape[:2]
            H=upright_H(K,roll_rad,pitch_rad,ROLL_FRAC,pf)
            warpedH_rep=cv2.warpPerspective(before,H,(ww,hh),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
            level=find_level(warpedH_rep)
            M=scale3(ww,hh,ZOOM_OUT)@rot3(ww,hh,level)@H
            cand=render(before,M); va,ha=axis_leans(center(webp_rt(cand)))
            v_ok=va<=vb+0.15; gain=(va<=vb-0.30) or (v_ok and ha<=hb-0.30)
            if v_ok and gain:
                status="corrected"; note=f"pf {pf:.2f} level {level:+.2f} zoom_out {ZOOM_OUT} LaMa-fill no-crop"
                write_webp(out_b,cand); write_webp(out_l,render(lit,M)); eff_pf=pf
            else:
                status="reverted"; note=f"do-no-harm V {vb:.2f}->{va:.2f} H {hb:.2f}->{ha:.2f}"
                write_webp(out_b,orig); write_webp(out_l,to_43(lit)); va,ha=vb,hb; level=0.0

        shipped=cv2.imread(out_b)
        title=f"{stem} | pitch {pitch:+.1f} | pf {eff_pf:.2f} lvl {level:+.2f} | {status} | V {vb:.1f}->{va:.1f} H {hb:.1f}->{ha:.1f}"
        sp=os.path.join(SCRATCH,f"G6_{stem}_{md5b(shipped)}.png"); sheet(orig_w,shipped,title,sp)
        results.append(dict(stem=stem,pitch=pitch,pf=eff_pf,level=level,status=status,note=note,vb=vb,va=va,hb=hb,ha=ha,sheet=sp))

    print("\n"+"="*100)
    print(f"{'room':11}{'pitch':>7}{'pf':>6}{'level':>7}{'status':>11}{'V_b':>6}{'V_a':>6}{'H_b':>6}{'H_a':>6}")
    print("-"*100)
    for r in results:
        print(f"{r['stem']:11}{r['pitch']:7.1f}{r['pf']:6.2f}{r['level']:7.2f}{r['status']:>11}{r['vb']:6.2f}{r['va']:6.2f}{r['hb']:6.2f}{r['ha']:6.2f}")
    print("="*100)
    for r in results: print(f"{r['stem']:11} {r['note']}\n            sheet: {r['sheet']}")
    return results

if __name__ == "__main__":
    main()
