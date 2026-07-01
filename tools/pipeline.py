#!/usr/bin/env python3
"""RoomCast deterministic pipeline: straighten (geometry) + relight (tone-map).
Same room guaranteed — no generative repaint. CLI param-driven for best-of-N tuning."""
import cv2, numpy as np, argparse, sys

# ---------- measurement ----------
def measure_tilt(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h = im.shape[0]
    edges = cv2.Canny(g, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=int(0.18*h), maxLineGap=12)
    if lines is None: return 0.0, 0
    angs = []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        ang = np.degrees(np.arctan2(x2-x1, y2-y1))  # 0 = vertical
        a = abs(ang)
        if a > 90: a = 180-a
        if a < 25: angs.append(a*np.sign(ang if abs(ang)<=90 else (180-abs(ang))*np.sign(ang)))
    if not angs: return 0.0, 0
    return float(np.median(np.abs(angs))), len(angs)

def signed_tilt(im):
    """median SIGNED tilt of near-vertical lines (deg), + = leaning one way."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); h = im.shape[0]
    edges = cv2.Canny(g,50,150)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,80,minLineLength=int(0.18*h),maxLineGap=12)
    if lines is None: return 0.0,0
    s=[]
    for l in lines:
        x1,y1,x2,y2=l[0]
        ang=np.degrees(np.arctan2(x2-x1,y2-y1))
        if abs(ang)<25: s.append(ang)
        elif abs(ang)>155: s.append(ang-180*np.sign(ang))
    if not s: return 0.0,0
    return float(np.median(s)), len(s)

# ---------- geometry ----------
def warp(im, theta_deg, k):
    h,w = im.shape[:2]
    cx,cy = w/2,h/2
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th),-np.sin(th),0],[np.sin(th),np.cos(th),0],[0,0,1]],np.float64)
    T  = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]],np.float64)
    Ti = np.array([[1,0,cx],[0,1,cy],[0,0,1]],np.float64)
    K = np.array([[1,0,0],[0,1,0],[0,k,1]],np.float64)
    H = Ti @ K @ R @ T
    return cv2.warpPerspective(im,H,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

def straighten(im, strength=1.0):
    """grid-search roll+gentle keystone, minimize tilt with a distortion penalty, then crop."""
    st,_ = signed_tilt(im)
    base_t,_ = measure_tilt(im)
    best = im; best_score = base_t; best_t=base_t; best_args=(0,0)
    thetas = np.linspace(-st-2, -st+2, 9) if abs(st)>0.3 else np.linspace(-2,2,9)
    ks = np.linspace(-0.0004,0.0004,11)*strength      # allow real keystone for convergence
    for th in thetas:
        for k in ks:
            cand = warp(im, th, k)
            t,n = measure_tilt(cand)
            if n < 6: continue
            score = t + 600*abs(k) + 0.04*abs(th)     # light distortion penalty
            if score < best_score - 0.03:
                best_score=score; best_t=t; best=cand; best_args=(round(th,2),round(k,5))
    h,w = best.shape[:2]
    m = int(0.07*max(h,w))                            # crop to hide warped edges
    best = best[m:h-m, m:w-m]
    return best, best_t, best_args

def to_43(im, W=1200, H=900, zoom=1.0):
    h,w = im.shape[:2]; tar=W/H; cur=w/h
    if cur>tar: nw=int(h*tar); x=(w-nw)//2; im=im[:,x:x+nw]
    elif cur<tar: nh=int(w/tar); y=(h-nh)//2; im=im[y:y+nh,:]
    if zoom>1.0:
        h2,w2=im.shape[:2]; cw,ch=int(w2/zoom),int(h2/zoom)
        x=(w2-cw)//2; y=(h2-ch)//2; im=im[y:y+ch, x:x+cw]
    return cv2.resize(im,(W,H),interpolation=cv2.INTER_AREA)

# ---------- relight ----------
def gray_world_wb(im, amt=0.6):
    b,g,r = cv2.split(im.astype(np.float32))
    mb,mg,mr = b.mean(),g.mean(),r.mean(); mg_all=(mb+mg+mr)/3
    b*= (1+amt*((mg_all/ (mb+1e-6))-1)); g*=(1+amt*((mg_all/(mg+1e-6))-1)); r*=(1+amt*((mg_all/(mr+1e-6))-1))
    return cv2.merge([np.clip(b,0,255),np.clip(g,0,255),np.clip(r,0,255)]).astype(np.uint8)

def relight(im, shadow=0.45, clahe=1.3, hi_recover=0.6, wb=0.5, vibrance=0.12, con=0.12, lift_floor=0.0):
    if wb>0: im = gray_world_wb(im, wb)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:,:,0]/255.0
    # shadow/mid lift via gamma (compresses highlights naturally) — "bright & airy"
    g = 1.0/(1.0+shadow)
    Llift = np.power(np.clip(L+lift_floor,0,1), g)
    # window / highlight recovery: pull near-blown pixels down to recover character
    thr=0.90; hi = Llift>thr
    Llift[hi] = thr + (Llift[hi]-thr)*(1.0-hi_recover)
    L8 = np.clip(Llift*255,0,255).astype(np.uint8)
    # gentle local contrast (dimensional, lit look) — low clip avoids blotchy walls
    cl = cv2.createCLAHE(clipLimit=clahe, tileGridSize=(8,8))
    L8 = cl.apply(L8)
    # soft S-curve for pro "pop" without crushing
    Ls = L8.astype(np.float32)/255.0
    Ls = np.clip(0.5+(Ls-0.5)*(1.0+con),0,1)
    lab[:,:,0] = (Ls*255.0)
    out = cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    if vibrance>0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1]=np.clip(hsv[:,:,1]*(1+vibrance),0,255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out

def lmean(im): return float(cv2.cvtColor(im,cv2.COLOR_BGR2LAB)[:,:,0].mean())

# ---------- main ----------
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--in",dest="inp",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--straighten",type=int,default=1)
    ap.add_argument("--strength",type=float,default=1.0)
    ap.add_argument("--shadow",type=float,default=0.45)
    ap.add_argument("--clahe",type=float,default=1.3)
    ap.add_argument("--hi",type=float,default=0.6)
    ap.add_argument("--wb",type=float,default=0.5)
    ap.add_argument("--vib",type=float,default=0.12)
    ap.add_argument("--con",type=float,default=0.12)
    ap.add_argument("--zoom",type=float,default=1.0)
    ap.add_argument("--q",type=int,default=92)
    a=ap.parse_args()
    im=cv2.imread(a.inp,cv2.IMREAD_COLOR)
    if im is None: sys.exit("cannot read "+a.inp)
    t0,_=measure_tilt(im); l0=lmean(im); args=(0,0); t1=t0
    if a.straighten:
        im,t1,args=straighten(im,a.strength)
    im=to_43(im, zoom=a.zoom)
    im=relight(im, shadow=a.shadow, clahe=a.clahe, hi_recover=a.hi, wb=a.wb, vibrance=a.vib, con=a.con)
    if a.out.endswith(".webp"): cv2.imwrite(a.out, im,[cv2.IMWRITE_WEBP_QUALITY,a.q])
    else: cv2.imwrite(a.out, im)
    print(f"{a.out}: tilt {t0:.2f}->{t1:.2f} (roll {args[0]:+.2f},k {args[1]:+.5f})  L {l0:.1f}->{lmean(im):.1f}")
