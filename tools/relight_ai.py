#!/usr/bin/env python3
"""RoomCast LIGHTING — AI fill-light, honestly applied.

The AI (SDXL+dualControlNet) generates a professionally-lit *reference* of the room.
We DISCARD its pixels and keep only the smooth LOW-FREQUENCY light field — i.e. WHERE
the pro lighting adds illumination. That light field is multiplied onto the REAL photo.

Why this is honest: a smooth, heavily-blurred multiplicative light map cannot add a
window, swap a fixture, or change a view — those live in the high-frequency detail of
the ORIGINAL, which is preserved untouched. Only the broad illumination changes — exactly
what bringing lights into the real room would do. Light comes from the AI; truth from the photo.
Fill-only: light is never subtracted, and highlights are protected so windows don't blow.
"""
import os, sys, types, tempfile, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import gl

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets", "loren-apt")
ROOMS = {
    "1-living": "living room", "2-kitchen": "kitchen", "3-kitchen2": "kitchen",
    "4-hall": "entryway hallway", "5-bath": "bathroom",
}

LIGHT_PROMPT = ("professional real estate interior photograph, {room}, bright and airy, "
                "abundant soft natural daylight filling the room, evenly and beautifully lit, "
                "luminous, magazine quality, well-lit, photorealistic")


def ai_reference(stem, room, strength=0.6):
    """Generate a strongly, professionally lit reference (pixels discarded; light kept)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".webp", delete=False).name
    a = types.SimpleNamespace(
        inp=os.path.join(ASSETS, f"{stem}-before.webp"), out=tmp, room=room,
        strength=strength, depth=0.85, canny=0.62, steps=28, guidance=6.0,
        canny_lo=80, canny_hi=180, seed=1, fp32=False)
    # temporarily swap gl's prompt for a light-forward one
    saved = gl.PROMPT
    gl.PROMPT = LIGHT_PROMPT
    try:
        gl.run(a)
    finally:
        gl.PROMPT = saved
    return cv2.imread(tmp, cv2.IMREAD_COLOR)


def apply_light_field(orig, ref, sigma_frac=0.05, fill=1.1, ai_w=0.7, sh_pow=1.5,
                      bloom=0.10, hi_thr=215.0):
    """Fill the real photo with light. Shadow-targeted (windows protected), MODULATED by where
    the AI says light falls, plus a subtle bloom for the luminous 'shot-with-lights' feel.
    Fill-only (gain >= 1): shadows bloom up, highlights hold — like a pro fill light, not a slider.
    Params locked from the living-room sweep (meanL ~147, ~0% blown)."""
    H, W = orig.shape[:2]
    ref = cv2.resize(ref, (W, H), interpolation=cv2.INTER_CUBIC)
    Ol = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    Rl = cv2.cvtColor(ref,  cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    sig = sigma_frac * max(H, W)
    Ob = cv2.GaussianBlur(Ol, (0, 0), sig)
    Rb = cv2.GaussianBlur(Rl, (0, 0), sig)

    shadowfill = np.clip((200.0 - Ob) / 200.0, 0, 1) ** sh_pow   # only genuine shadow gets strong fill
    ai_extra = np.clip((Rb - Ob) / 110.0, 0, 1)                  # where the AI added light
    weight = np.clip(shadowfill * (1.0 + ai_w * ai_extra), 0, 1.2)
    gain = 1.0 + fill * weight                                   # >= 1 everywhere: never darkens
    gain = cv2.GaussianBlur(gain, (0, 0), sig * 0.6)             # SMOOTH light field (no detail = can't fake)

    out = orig.astype(np.float32) * gain[:, :, None]
    out = np.where(out > hi_thr, hi_thr + (out - hi_thr) * 0.35, out)  # soft highlight rolloff

    # subtle bloom: light spilling from bright areas — luminous, professional, content-safe
    if bloom > 0:
        glow = cv2.GaussianBlur(np.clip(out - 205, 0, 255), (0, 0), sig * 0.5)
        out = np.clip(out + bloom * glow, 0, 255)

    out = np.clip(out, 0, 255).astype(np.uint8)
    # gentle vibrance only — no contrast curve, keep it airy (no darker darks)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.07, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def lmean(im):
    return float(cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[:, :, 0].mean())


def main(stems):
    for stem in stems:
        room = ROOMS[stem]
        orig = cv2.imread(os.path.join(ASSETS, f"{stem}-before.webp"), cv2.IMREAD_COLOR)
        ref = ai_reference(stem, room)
        out = apply_light_field(orig, ref)
        dst = os.path.join(ASSETS, f"{stem}-lit.webp")
        cv2.imwrite(dst, out, [cv2.IMWRITE_WEBP_QUALITY, 92])
        print(f"{stem}: L {lmean(orig):.1f} -> {lmean(out):.1f}", flush=True)
    print("AI FILL-LIGHT COMPLETE", flush=True)


if __name__ == "__main__":
    stems = sys.argv[1:] or list(ROOMS.keys())
    main(stems)
