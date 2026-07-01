#!/usr/bin/env python3
"""RoomCast TIDY — living room, honest, tested. Fixes v1: (1) tight masks so the table itself is
NEVER erased (only the clutter on it); (2) tidy MORE than one spot so several selections do something.

Two honest operations, both composited back into the original so everything outside is untouched:
  - ERASE (LaMa remove-only): magazines/items on the coffee table, loose clutter on the desk.
  - DE-WRINKLE (edge-preserving smooth): soften cushion creases on the couch — removes wrinkle detail,
    adds nothing. Gentle; if it reads plastic we drop it.

Output: 1-living-lit-angleZ-tidy.webp + a verification sheet (mask overlay | tidied) + honesty stats.
"""
import os, cv2, numpy as np, torch
from PIL import Image
_ORIG=torch.jit.load; torch.jit.load=lambda *a,**k: _ORIG(*a,**{**k,"map_location":"cpu"})
from simple_lama_inpainting import SimpleLama

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
ASSETS=os.path.join(ROOT,"assets","loren-apt")
SCRATCH="/private/tmp/claude-501/-Users-lorenpolster-Claude-Projects-roomcast/7d42c730-3fa8-4885-b304-3dea4c119ad7/scratchpad"
STEM="1-living-lit-angleZ"; WEBP_Q=92

ERASE=[(500,604,590,674),(630,620,672,656),(492,446,570,484)]   # table magazines, table item, desk clutter
SMOOTH=[(668,560,1185,822)]                                       # couch cushions (de-wrinkle)


def mask_from(shape,rects,feather):
    m=np.zeros(shape[:2],np.uint8)
    for x0,y0,x1,y1 in rects: cv2.rectangle(m,(x0,y0),(x1,y1),255,-1)
    if feather:
        m=cv2.dilate(m,np.ones((feather,feather),np.uint8),1)
        m=cv2.GaussianBlur(m,(0,0),feather/3.0); m=(m>40).astype(np.uint8)*255
    return m


def main():
    base=cv2.imread(os.path.join(ASSETS,f"{STEM}.webp")); out=base.copy()

    # 1) ERASE clutter with LaMa, composite back only inside the erase mask
    em=mask_from(base.shape,ERASE,9)
    lama=SimpleLama(torch.device("cpu"))
    fill=cv2.cvtColor(np.array(lama(Image.fromarray(cv2.cvtColor(base,cv2.COLOR_BGR2RGB)),Image.fromarray(em))),cv2.COLOR_RGB2BGR)
    fill=cv2.resize(fill,(base.shape[1],base.shape[0]))
    ea=(cv2.GaussianBlur(em,(0,0),4).astype(np.float32)/255)[...,None]
    out=(out.astype(np.float32)*(1-ea)+fill.astype(np.float32)*ea).astype(np.uint8)

    # 2) DE-WRINKLE couch: edge-preserving smooth, composited only inside the couch mask
    sm=mask_from(base.shape,SMOOTH,15)
    soft=out.copy()
    for _ in range(2):
        soft=cv2.bilateralFilter(soft,13,42,14)
    sa=(cv2.GaussianBlur(sm,(0,0),8).astype(np.float32)/255*0.8)[...,None]   # 0.8 = keep some texture
    out=(out.astype(np.float32)*(1-sa)+soft.astype(np.float32)*sa).astype(np.uint8)

    # honesty: outside the union of masks must be identical
    union=cv2.max(em,sm); m3=(cv2.GaussianBlur(union,(0,0),5)>8)
    diff=np.abs(base.astype(int)-out.astype(int)).sum(2)
    outside=int((diff[~m3]>2).sum()); tot=int((~m3).sum())

    cv2.imwrite(os.path.join(ASSETS,f"{STEM}-tidy.webp"),out,[cv2.IMWRITE_WEBP_QUALITY,WEBP_Q])

    ov=base.copy()
    ov[em>0]=(0.45*ov[em>0]+0.55*np.array([40,40,230])).astype(np.uint8)   # erase = red
    ov[sm>0]=(0.6*ov[sm>0]+0.4*np.array([230,150,40])).astype(np.uint8)     # smooth = blue
    def lab(im,t):
        im=im.copy();cv2.rectangle(im,(0,0),(im.shape[1],30),(250,247,241),-1);cv2.putText(im,t,(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.7,(40,34,28),2);return im
    h=560
    def rs(im): return cv2.resize(im,(int(h*im.shape[1]/im.shape[0]),h))
    sheet=np.hstack([lab(rs(ov),"MASKS: red=erase blue=de-wrinkle"),np.full((h,6,3),230,np.uint8),lab(rs(out),"TIDIED")])
    sp=os.path.join(SCRATCH,f"TIDYLIV_{STEM}.png"); cv2.imwrite(sp,sheet)
    print(f"wrote {STEM}-tidy.webp  outside-mask changed {outside}/{tot} ({100*outside/tot:.3f}%)\nsheet: {sp}")


if __name__=="__main__": main()
