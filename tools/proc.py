import cv2, numpy as np, sys

def load(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: raise SystemExit("cannot read "+p)
    return im

def to_43(im, W=1200, H=900):
    h,w = im.shape[:2]
    tar = W/H
    cur = w/h
    if cur > tar:   # too wide -> crop sides
        nw = int(h*tar); x=(w-nw)//2; im=im[:, x:x+nw]
    elif cur < tar: # too tall -> crop top/bottom
        nh = int(w/tar); y=(h-nh)//2; im=im[y:y+nh, :]
    return cv2.resize(im,(W,H),interpolation=cv2.INTER_AREA)

def exposure_pin(pro, clean):
    # additive LAB mean shift: match pro's global brightness/color to clean
    a = cv2.cvtColor(pro,  cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(clean,cv2.COLOR_BGR2LAB).astype(np.float32)
    for i in range(3):
        a[:,:,i] += (b[:,:,i].mean() - a[:,:,i].mean())
    a = np.clip(a,0,255).astype(np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_LAB2BGR)

def tilt(im):
    # median absolute tilt of near-vertical lines, in degrees
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h = im.shape[0]
    edges = cv2.Canny(g,50,150)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=80,
                            minLineLength=int(0.18*h),maxLineGap=12)
    if lines is None: return None,0
    angs=[]
    for l in lines:
        x1,y1,x2,y2=l[0]
        ang=np.degrees(np.arctan2(x2-x1, y2-y1))  # 0 = perfectly vertical
        a=abs(ang)
        if a>90: a=180-a
        if a<22: angs.append(a)
    if not angs: return None,0
    return float(np.median(angs)), len(angs)

def lmean(im):
    return float(cv2.cvtColor(im,cv2.COLOR_BGR2LAB)[:,:,0].mean())

def save_webp(im, path, q=90):
    cv2.imwrite(path, im, [cv2.IMWRITE_WEBP_QUALITY, q])

jobs = [
    ("living-clean.png",  "living-pro.jpeg", "1-living"),
    ("bath-clean.jpeg",   "bath-pro.png",    "5-bath"),
]
OUT="/Users/lorenpolster/Claude/Projects/roomcast/assets/loren-apt"
for cf, pf, name in jobs:
    clean = to_43(load(cf))
    pro   = to_43(load(pf))
    t0,n0 = tilt(pro)
    pinned = exposure_pin(pro, clean)
    t1,n1 = tilt(pinned)
    save_webp(clean,  f"{OUT}/{name}-clean.webp")
    save_webp(pinned, f"{OUT}/{name}-pro.webp")
    print(f"{name}: clean L={lmean(clean):.1f}  pro L raw={lmean(pro):.1f} -> pinned={lmean(pinned):.1f}  "
          f"(jump {lmean(pinned)-lmean(clean):+.1f})   pro tilt={t1:.2f}deg (n={n1})")
print("done")
