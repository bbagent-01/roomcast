#!/usr/bin/env python3
"""Batch the geometry-lock Lighting pass over the remaining rooms.
Loads the SDXL+dualControlNet pipeline ONCE (cached globally in gl.get_pipe), then
runs each room at the validated Balanced setting. Free/local on M4 Max."""
import os, sys, types, time
sys.path.insert(0, os.path.dirname(__file__))
import gl

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets", "loren-apt")

# (before-stem, room-label-for-prompt)
ROOMS = [
    ("2-kitchen",  "kitchen"),
    ("3-kitchen2", "kitchen"),
    ("4-hall",     "entryway hallway"),
    ("5-bath",     "bathroom"),
]

# Validated Balanced settings (from the living-room proof)
BALANCED = dict(strength=0.38, depth=0.85, canny=0.62, steps=28, guidance=6.0,
                canny_lo=80, canny_hi=180, seed=1, fp32=False)

def main():
    for stem, room in ROOMS:
        inp = os.path.join(ASSETS, f"{stem}-before.webp")
        out = os.path.join(ASSETS, f"{stem}-gl-balanced.webp")
        a = types.SimpleNamespace(inp=inp, out=out, room=room, **BALANCED)
        t0 = time.time()
        gl.run(a)
        print(f"  done {stem} in {time.time()-t0:.0f}s", flush=True)
    print("BATCH COMPLETE", flush=True)

if __name__ == "__main__":
    main()
