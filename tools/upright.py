#!/usr/bin/env python3
"""RoomCast vertical-straighten via the vertical VANISHING POINT (true keystone + roll).

The old approach only rotated (roll) — it leveled the average but left verticals converging,
so it read as "just zoomed" or even inverted the lean. This finds where the vertical lines
actually converge and applies the perspective correction that makes them PLUMB and PARALLEL,
then a residual roll level. Honest: pure homography on captured pixels + crop (no new pixels).
The same homography is applied to the lit version so Lighting+Angle stay aligned.
"""
import numpy as np, cv2


def _build_H(w, h, theta_deg, k):
    """Same homography family as pipeline.warp: center, roll, vertical keystone, uncenter."""
    cx, cy = w / 2.0, h / 2.0
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]], np.float64)
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float64)
    Ti = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], np.float64)
    K = np.array([[1, 0, 0], [0, 1, 0], [0, k, 1]], np.float64)
    return Ti @ K @ R @ T


def _vertical_lines(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h = im.shape[0]
    edges = cv2.Canny(g, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=int(0.12 * h), maxLineGap=15)
    out = []
    if lines is None:
        return out
    for l in lines:
        x1, y1, x2, y2 = l[0]
        ang = np.degrees(np.arctan2(x2 - x1, y2 - y1))   # 0 = perfectly vertical
        a = abs(ang)
        if a > 90:
            a = 180 - a
        if a < 30:                                       # near-vertical only
            out.append(((x1, y1), (x2, y2)))
    return out


def _dir(p1, p2):
    d = np.array([p2[0] - p1[0], p2[1] - p1[1]], float)
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else d


def _dir_to_vp(mid, vp):
    if abs(vp[2]) < 1e-9:
        d = np.array([vp[0], vp[1]], float)
    else:
        d = np.array([vp[0] / vp[2] - mid[0], vp[1] / vp[2] - mid[1]], float)
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else d


def vanishing_vertical(im, tol_deg=2.0):
    """RANSAC: find the dominant vertical vanishing point, robust to furniture/plant edges."""
    lines = _vertical_lines(im)
    if len(lines) < 8:
        return None, [], len(lines)
    homo = [np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0]) for (p1, p2) in lines]
    mids = [((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0) for (p1, p2) in lines]
    dirs = [_dir(p1, p2) for (p1, p2) in lines]
    n = len(lines)
    tol = np.cos(np.radians(tol_deg))
    # deterministic candidate set: all pairs if small, else a fixed stride sampling
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    if len(pairs) > 1200:
        pairs = pairs[::max(1, len(pairs) // 1200)]
    best_inliers = []
    for (i, j) in pairs:
        vp = np.cross(homo[i], homo[j])
        if np.linalg.norm(vp) < 1e-9:
            continue
        inl = []
        for t in range(n):
            dv = _dir_to_vp(mids[t], vp)
            if abs(float(np.dot(dv, dirs[t]))) >= tol:
                inl.append(t)
        if len(inl) > len(best_inliers):
            best_inliers = inl
    if len(best_inliers) < 4:
        return None, [], n
    # refit vp on inliers via SVD (normalized lines)
    L = []
    for t in best_inliers:
        l = homo[t] / (np.linalg.norm(homo[t][:2]) + 1e-12)
        L.append(l)
    _, _, Vt = np.linalg.svd(np.array(L))
    return Vt[-1], best_inliers, n


def robust_roll(im, minfrac=0.20):
    """LENGTH-WEIGHTED lean of LONG near-vertical lines (deg). The long architectural walls are
    what the eye reads as 'vertical' — a median over all little edges hid the kitchens' ~7deg roll."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h = im.shape[0]
    edges = cv2.Canny(g, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=int(minfrac * h), maxLineGap=12)
    if lines is None:
        return 0.0
    num = den = 0.0
    for l in lines:
        x1, y1, x2, y2 = l[0]
        ang = np.degrees(np.arctan2(x2 - x1, y2 - y1))
        if abs(ang) < 22:
            L = np.hypot(x2 - x1, y2 - y1)
            num += ang * L; den += L
    return float(num / den) if den > 0 else 0.0


def upright_params(im, kmax=0.0034, rollmax=9.0):
    """Return (theta_deg, k) that make verticals plumb. k from the vertical vanishing point."""
    h, w = im.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    vp, inliers, nlines = vanishing_vertical(im)
    k = 0.0
    if vp is not None and abs(vp[2]) > 1e-12:
        vy = vp[1] / vp[2]
        vyc = vy - cy
        if abs(vyc) > 0.55 * h:          # far enough to be real convergence, not degenerate noise
            k = float(np.clip(-1.0 / vyc, -kmax, kmax))
    # apply keystone, then measure residual roll to level
    from pipeline import warp
    im1 = warp(im, 0, k)
    theta = float(np.clip(-robust_roll(im1), -rollmax, rollmax))
    return theta, k, nlines


def vertical_error(im, minfrac=0.20):
    """LENGTH-WEIGHTED mean |angle from vertical| of LONG lines (deg). Consistent with robust_roll
    so the improvement guard judges the walls, not the swarm of tiny edges. Lower = more plumb."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h = im.shape[0]
    edges = cv2.Canny(g, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=int(minfrac * h), maxLineGap=12)
    if lines is None:
        return 0.0, 0
    num = den = 0.0; n = 0
    for l in lines:
        x1, y1, x2, y2 = l[0]
        ang = np.degrees(np.arctan2(x2 - x1, y2 - y1))
        if abs(ang) < 22:
            L = np.hypot(x2 - x1, y2 - y1)
            num += abs(ang) * L; den += L; n += 1
    return (float(num / den) if den > 0 else 0.0), n


def process(before, lit):
    """Compute upright homography on `before`, apply to both, crop to the black-free interior.
    Returns (before_out, lit_out, info) or (None, None, info) if it doesn't genuinely improve."""
    h, w = before.shape[:2]
    theta, k, nlines = upright_params(before)
    H1 = _build_H(w, h, 0, k)
    H2 = _build_H(w, h, theta, 0)
    Htot = H2 @ H1
    pre_err, _ = vertical_error(before)

    def apply(im):
        return cv2.warpPerspective(im, Htot, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    b2 = apply(before)
    post_err, _ = vertical_error(b2)
    # only keep if it genuinely made verticals more plumb (and a real change happened)
    improved = (post_err < pre_err - 0.25) and (abs(theta) > 0.3 or abs(k) > 1e-4)
    info = dict(theta=theta, k=k, nlines=nlines, pre_err=pre_err, post_err=post_err, improved=improved)
    if not improved:
        return None, None, info

    l2 = apply(lit)
    # black-free inscribed crop from a validity mask warped the same way
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), Htot, (w, h),
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = mask > 0.5
    xs = np.where(valid.all(axis=0))[0]
    ys = np.where(valid.all(axis=1))[0]
    if len(xs) and len(ys):
        x0, x1, y0, y1 = xs[0], xs[-1] + 1, ys[0], ys[-1] + 1
    else:
        m = int(0.08 * max(h, w)); x0, x1, y0, y1 = m, w - m, m, h - m
    return b2[y0:y1, x0:x1], l2[y0:y1, x0:x1], info
