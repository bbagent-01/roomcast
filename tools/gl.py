#!/usr/bin/env python3
"""RoomCast geometry-lock beautify (local, free).
SDXL img2img + dual ControlNet (depth + canny) locks the real room's geometry while
the prompt relights/tidies. Structure can't be swapped; one strength dial = faithful<->dramatic."""
import argparse, sys, numpy as np, cv2
from PIL import Image
import torch
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel, AutoencoderKL

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def load_resize(path, W=1024, H=768):
    im = Image.open(path).convert("RGB")
    # cover-crop to 4:3 then resize
    w,h = im.size; tar=W/H; cur=w/h
    if cur>tar:
        nw=int(h*tar); x=(w-nw)//2; im=im.crop((x,0,x+nw,h))
    elif cur<tar:
        nh=int(w/tar); y=(h-nh)//2; im=im.crop((0,y,w,y+nh))
    return im.resize((W,H), Image.LANCZOS)

def canny_pil(pil, lo=80, hi=180):
    a=np.array(pil); g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
    e=cv2.Canny(g,lo,hi); e=np.stack([e]*3,-1)
    return Image.fromarray(e)

_pipe=None;_midas=None
def get_pipe(dtype):
    global _pipe,_midas
    if _pipe is None:
        from controlnet_aux import MidasDetector
        _midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
        depth_cn = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=dtype)
        canny_cn = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=dtype)
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
        _pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=[depth_cn, canny_cn], vae=vae, torch_dtype=dtype, variant="fp16" if dtype==torch.float16 else None)
        _pipe.to(DEVICE)
        _pipe.set_progress_bar_config(disable=True)
    return _pipe,_midas

PROMPT=("professional real estate interior listing photograph, {room}, bright airy natural daylight, "
        "soft even professional lighting, clean tidy and decluttered, crisp smooth upholstery, "
        "magazine quality, photorealistic, sharp, high detail")
NEG=("cluttered, messy, dark, dim, underexposed, blurry, distorted, warped, deformed furniture, "
     "extra furniture, different room, cartoon, illustration, lowres, watermark, text, oversaturated")

def run(a):
    dtype = torch.float32 if a.fp32 else torch.float16
    pipe,midas = get_pipe(dtype)
    init = load_resize(a.inp)
    depth = midas(init).resize(init.size)
    canny = canny_pil(init, a.canny_lo, a.canny_hi)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    room = a.room or "room"
    out = pipe(
        prompt=PROMPT.format(room=room), negative_prompt=NEG,
        image=init, control_image=[depth, canny],
        strength=a.strength, num_inference_steps=a.steps, guidance_scale=a.guidance,
        controlnet_conditioning_scale=[a.depth, a.canny], generator=g).images[0]
    out.save(a.out, quality=92)
    print(f"{a.out}  strength={a.strength} depth={a.depth} canny={a.canny} seed={a.seed} dev={DEVICE} dtype={dtype}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--in",dest="inp",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--room",default="living room")
    ap.add_argument("--strength",type=float,default=0.5)
    ap.add_argument("--depth",type=float,default=0.8)
    ap.add_argument("--canny",type=float,default=0.55)
    ap.add_argument("--steps",type=int,default=28)
    ap.add_argument("--guidance",type=float,default=6.0)
    ap.add_argument("--canny_lo",type=int,default=80)
    ap.add_argument("--canny_hi",type=int,default=180)
    ap.add_argument("--seed",type=int,default=1)
    ap.add_argument("--fp32",action="store_true")
    run(ap.parse_args())
