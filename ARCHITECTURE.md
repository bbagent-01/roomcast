# RoomCast — the right architecture (v2, after deep research 2026-06-26)

## ✅ SHIPPED LIGHTING ENGINE (Loren-approved 2026-06-27) — supersedes the generative plan below for LIGHTING
**Lighting = AI fill-light** (`tools/relight_ai.py`), live on all 5 rooms at roomcast.bbase.ai/apartment.
- The generative SDXL+dualControlNet pass makes a professionally-lit *reference*, then we KEEP ONLY its smooth low-frequency **light field** and multiply that onto the REAL photo (fill-only gain ≥1, shadow-targeted, windows protected, subtle bloom; NO contrast S-curve). Light from the AI, truth from the photo.
- **Honest by construction:** a blurred multiplicative light map carries no detail → cannot add a window or swap a fixture (those are the original's high-freq pixels, untouched). Geometry NCC ~0.97; meanL ~146–164; ~0% blown. "Brightness" is one dial (`fill`, currently 1.1).
- **REJECTED for Lighting/fidelity:** full-frame generative img2img (even depth+canny locked at strength 0.38) — it reinvents fixtures/views every run (fabric curtain→glass door, added microwave+dishwasher, re-rendered skyline). Used now ONLY as the throwaway light-field source.
- **Process lesson:** iterate image outputs to FRESH filenames (`-lit.webp`) — overwriting a reused path serves stale cached bytes to Read AND to review subagents (cost an hour of phantom "reinvention" failures). Cache-bust by copying to a unique path before trusting any visual review.

## Root cause of every failed round (confirmed by research)
Nano Banana / Gemini-edit / Flux Kontext are **instruction editors with NO geometric anchor**. They rebuild the whole image from a semantic understanding, so any strong "make it beautiful" transform is *free* to swap the sofa/rug/windows/art. **No prompt fixes this** — it's inherent to instruction editors. Feeding the cleaned base instead of the original helps but doesn't truly lock geometry.

## THE FIX: geometry-locked generation (what real virtual-staging companies use)
Condition generation on the photo's OWN geometry (depth + edge maps), so structure is locked outside the model's discretion. Beautify (relight, smooth, tidy, polish) happens *within* the locked room. One faithful↔dramatic dial = ControlNet scale up + denoise down.

## Recommended pipeline — all callable on fal.ai (the approved backend), ~$0.20–0.40/photo
1. **Faithful straighten + reframe** (NOT a new camera angle — that always hallucinates a different room):
   - deterministic vertical/horizon + lens-distortion correction (OpenCV vanishing-point homography)
   - then **small faithful outpaint** (`fal-ai/bria/expand`, ~$0.04) to refill the wedges the warp creates → keeps a wide *composed* frame instead of an ugly crop. Cap ~15%, never invent a whole new wall.
2. **Geometry-locked beautify** — `fal-ai/flux-general/image-to-image` with **dual ControlNet (depth scale ~0.8 + canny/lineart ~0.6)** at **strength ~0.5**. Relight to bright/airy, smooth/de-wrinkle upholstery, tidy — locked to the real room. (SDXL ControlNet Union is the fallback.)
3. **Surgical declutter** — `fal-ai/sam-3/image` (text→mask, $0.005) → `fal-ai/bria/eraser` ($0.04/object). Outside-mask is pixel-identical by construction. Self-host LaMa (IOPaint) for cheap small removals on plain walls.
4. **De-wrinkle** (if step 2 isn't enough) — SAM-3 mask the sofa → `fal-ai/flux-pro/v1/fill` at low denoise (~0.25). Keeps shape/identity; only texture changes.
5. **Final polish** — `fal-ai/clarity-upscaler` at **creativity ~0.15 / resemblance ~1.2** = magazine crispness, no hallucination. (Avoid Magnific/SUPIR — they invent texture.)

## The angle debate — SETTLED
A genuinely new viewpoint must invent geometry the photo never captured → different room. The "magazine-composed" feel that's faithful = straighten + level + lens-fix + **outpaint the gaps back out**. Frame to Loren: "we straighten the room and rebalance the frame like a magazine layout — we don't move the camera."

## Tools to AVOID (research-flagged)
- IC-Light for whole-room relight (built for foreground subjects; distorts complex scenes).
- Magnific / SUPIR at default creativity (hallucinate texture/proportions).
- Depth ControlNet alone (locks layout but still swaps objects — must add an edge control).
- Any single-pass instruction editor for a *strong global* transform (the trap we were in).

## BUSINESS / LEGAL GUARDRAIL (governs the whole product)
Rule: **"correct, don't reconstruct."** (NAR Article 12 "true picture"; **California AB 723, effective 2026-01-01, makes non-disclosure of digitally-altered listing photos a misdemeanor** — requires a conspicuous label + link to the unaltered original.)
- **Always free, no disclosure:** exposure/white-balance/color/denoise/sharpen, lens + perspective correction, straighten, crop, removing small genuinely-removable clutter.
- **Disclosure required:** virtual staging (added furniture), sky replacement → label on image + show original alongside.
- **NEVER (no disclosure saves you):** hide defects (cracks/stains/damage), add/remove permanent features (windows, pools, fireplaces), fake views, AI landscaping, change paint/flooring/fixtures, distort dimensions.
- RoomCast's safe lane = relight + straighten + declutter-removable + de-wrinkle. That lane is also ~90% of the perceived quality jump, so safe ≈ impressive. "Honest enhancement" is a selling point.
- Honest ceiling: true window-pull needs bracketed exposures (can't recover a blown window's real view from one phone frame); large-object removal is a quality + legal landmine.

## Reference services (real APIs, for buy-vs-build): Autoenhance.ai (faithful correction pipeline), Pedra, AI HomeDesign, REimagineHome (structure-locked staging), virtualstagingai.app. Restb.ai = compliance-checking "eyes" (could be our guardrail).

## What this needs
fal.ai API key (billing account — Loren's to create). Everything else (pipeline, caps, one-photo cost test) is on me.

## ✅ CONFIRMED BUILD SPEC (Loren, 2026-06-26 — after council-review)
- **THREE INDEPENDENT, USER-SELECTABLE enhancements — ANY SUBSET** (Lighting only / Tidy only / Angle only / any pair / all three). NOT all-or-nothing, NOT one global intensity. Each is its OWN pass on its OWN mechanism:
  - **Lighting** = geometry-locked relight (SDXL+dual ControlNet), CAPPED so it can't move walls or change room proportions.
  - **Tidy** = masked erase, **REMOVE-ONLY, never ADD** (SAM-3 text-mask -> Bria/LaMa erase); show what was removed.
  - **Angle** = MEASURE tilt first, only correct when it genuinely improves (kills "inverted-but-still-bad"); faithful crop + straighten + doorframe-removal of ALREADY-CAPTURED pixels. Honest line: zoom-in/crop/straighten = YES; widen / new vantage / invent geometry = NO.
- **AI-ENHANCED BADGE on every output image -> click reveals the ORIGINAL unaltered photo.** This is BOTH the transparency moat AND the AB723 legal-disclosure mechanism (conspicuous label + access to the unaltered original). Build from day one.
- **Honesty enforced by the ENGINE (hard caps), not user judgment.** Before/after is the headline feature.
- **Cost:** free local now (M4 Max); pennies/photo cloud for the lean pipeline. Business model (consumer free-tier vs B2B/agent SaaS) DEFERRED until proven on Loren's own listings — do not pivot the business prematurely.
- **DO THIS FIRST (council "do this first"):** run geometry-lock on the other 4 rooms free/local at Balanced (0.38) — finishes the apartment AND is the diverse-room fidelity test. Then layer Tidy + Angle as separate honest passes. Verify: every room stays unmistakably the same room; tidy only removes; angle only levels.
