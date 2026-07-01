#!/usr/bin/env python3
"""RoomCast ANGLE verification harness — judge straightness on the EXACT image the user receives.

The core past failure: metric ran on the raw warped buffer, the agent eyeballed a tiny preview, and
the user saw the final cropped+relit image — three different pictures. This harness ends that:
- overlay(): draws a TRUE-VERTICAL cyan grid + horizon, and the DETECTED long wall lines in red, on
  the final image. Red parallel to cyan = plumb; red splaying = tilted. Straightness becomes visible.
- lean(): length-weighted absolute deviation of long architectural verticals (deg) — for the
  do-no-harm guard (compare after vs before; never ship worse).
- sheet(): before|after contact sheet of overlays for human / redundant-VLM grading.
"""
import os, sys, cv2, numpy as np

CYAN = (210, 180, 40)   # BGR — true-vertical reference grid
RED = (40, 40, 230)     # BGR — detected wall lines
INK = (40, 34, 28)


def long_verticals(im, minfrac=0.18):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h = im.shape[0]
    lines = cv2.HoughLinesP(cv2.Canny(g, 50, 150), 1, np.pi / 180, 80,
                            minLineLength=int(minfrac * h), maxLineGap=12)
    out = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            ang = np.degrees(np.arctan2(x2 - x1, y2 - y1))   # 0 = vertical
            if abs(ang) < 22:
                out.append((ang, np.hypot(x2 - x1, y2 - y1), (x1, y1, x2, y2)))
    return out


def lean(im):
    """Length-weighted |deviation from vertical| of long wall lines, in degrees. Lower = plumber."""
    lv = long_verticals(im)
    if not lv:
        return 0.0, 0
    num = sum(abs(a) * L for a, L, _ in lv)
    den = sum(L for a, L, _ in lv)
    return (num / den if den else 0.0), len(lv)


def overlay(im):
    o = im.copy()
    h, w = o.shape[:2]
    grid = o.copy()
    for i in range(1, 8):                          # true-vertical reference lines
        x = int(i * w / 8)
        cv2.line(grid, (x, 0), (x, h), CYAN, 2)
    cv2.line(grid, (0, h // 2), (w, h // 2), CYAN, 2)   # horizon
    cv2.addWeighted(grid, 0.45, o, 0.55, 0, o)
    for a, L, (x1, y1, x2, y2) in long_verticals(im):   # detected wall lines
        cv2.line(o, (x1, y1), (x2, y2), RED, 3)
    lv_lean, n = lean(im)
    cv2.rectangle(o, (0, 0), (w, 42), (250, 247, 241), -1)
    cv2.putText(o, f"wall lean {lv_lean:.1f} deg   ({n} lines)", (12, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, INK, 2, cv2.LINE_AA)
    return o


def sheet(before, after, title, out_path, disp_w=560):
    """Side-by-side overlays of the before vs the corrected FINAL image."""
    def prep(im, label):
        o = overlay(im)
        h, w = o.shape[:2]
        nh = int(disp_w * h / w)
        o = cv2.resize(o, (disp_w, nh))
        bar = np.full((34, disp_w, 3), 250, np.uint8)
        cv2.putText(bar, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, INK, 2, cv2.LINE_AA)
        return np.vstack([bar, o])
    a = prep(before, "ORIGINAL")
    b = prep(after, "CORRECTED")
    H = max(a.shape[0], b.shape[0])
    a = np.pad(a, ((0, H - a.shape[0]), (0, 0), (0, 0)), constant_values=250)
    b = np.pad(b, ((0, H - b.shape[0]), (0, 0), (0, 0)), constant_values=250)
    gap = np.full((H, 8, 3), 230, np.uint8)
    sheet = np.hstack([a, gap, b])
    title_bar = np.full((40, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(title_bar, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, INK, 2, cv2.LINE_AA)
    cv2.imwrite(out_path, np.vstack([title_bar, sheet]))
    return out_path


if __name__ == "__main__":
    # quick self-test: overlay a given image
    inp = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "overlay.png"
    im = cv2.imread(inp)
    cv2.imwrite(out, overlay(im))
    print(out, lean(im))
