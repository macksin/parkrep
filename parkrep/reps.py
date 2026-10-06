"""From a tracked point to reps: smoothing, main axis, hysteresis turns, per-rep metrics.

Units are image pixels and seconds. Nothing here is calibrated to the real world;
velocity loss is a *relative* comparison against the first reps of the same clip.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

SMOOTH_HALF_S = 0.10      # temporal median half-window
VEL_HALF_S = 0.15         # local linear-fit half-window for velocity
MAX_BRIDGE_S = 0.35       # gaps up to this long are bridged (marked as inferred)
PROMINENCE_FRAC = 0.35    # reversal must exceed this share of the motion range
MIN_CONCENTRIC_S = 0.25
REFERENCE_REPS = 3        # the first N full reps form the fixed reference
COMPLETE_ROM = 0.60       # a rep counts if it covers >= 60% of reference ROM


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, bool)
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def find_turns(s: np.ndarray, prominence: float) -> list[tuple[int, str]]:
    """Alternating low/high turning points with prominence hysteresis.

    A turn is confirmed only once the signal has moved `prominence` away from it,
    so small wobbles at a sticking point do not create extra reps.
    """
    s = np.asarray(s, float)
    if len(s) < 3 or prominence <= 0:
        return []
    turns: list[tuple[int, str]] = []
    lo = hi = 0
    direction = 0
    for j in range(1, len(s)):
        if direction == 0:
            if s[j] < s[lo]:
                lo = j
            if s[j] > s[hi]:
                hi = j
            if s[j] - s[lo] >= prominence:
                turns.append((lo, "low"))
                direction, hi = 1, j
            elif s[hi] - s[j] >= prominence:
                turns.append((hi, "high"))
                direction, lo = -1, j
        elif direction == 1:
            if s[j] > s[hi]:
                hi = j
            elif s[hi] - s[j] >= prominence:
                turns.append((hi, "high"))
                direction, lo = -1, j
        else:
            if s[j] < s[lo]:
                lo = j
            elif s[j] - s[lo] >= prominence:
                turns.append((lo, "low"))
                direction, hi = 1, j
    if direction == 1:
        turns.append((hi, "high"))
    elif direction == -1:
        turns.append((lo, "low"))
    return turns


def smooth(t: np.ndarray, xy: np.ndarray, ok: np.ndarray) -> np.ndarray:
    out = np.full_like(xy, np.nan, dtype=float)
    for a, b in runs(ok):
        tt = t[a:b]
        lo = np.searchsorted(tt, tt - SMOOTH_HALF_S)
        hi = np.searchsorted(tt, tt + SMOOTH_HALF_S, side="right")
        for j in range(b - a):
            out[a + j] = np.median(xy[a + lo[j]:a + hi[j]], axis=0)
    return out


def velocity(t: np.ndarray, s: np.ndarray) -> np.ndarray:
    out = np.full(len(t), np.nan)
    for a, b in runs(np.isfinite(s)):
        tt = t[a:b]
        lo = np.searchsorted(tt, tt - VEL_HALF_S)
        hi = np.searchsorted(tt, tt + VEL_HALF_S, side="right")
        for j in range(b - a):
            x = tt[lo[j]:hi[j]] - tt[j]
            y = s[a + lo[j]:a + hi[j]]
            if len(x) >= 3:
                xc = x - x.mean()
                out[a + j] = np.dot(xc, y - y.mean()) / np.dot(xc, xc)
    return out


def bridge(t: np.ndarray, xy: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly bridge short tracking gaps; returns (xy, inferred_mask)."""
    xy = xy.copy()
    inferred = np.zeros(len(t), bool)
    for a, b in runs(~ok):
        if a == 0 or b == len(t) or t[b] - t[a - 1] > MAX_BRIDGE_S:
            continue
        w = (t[a:b] - t[a - 1]) / (t[b] - t[a - 1])
        xy[a:b] = xy[a - 1] + w[:, None] * (xy[b] - xy[a - 1])
        inferred[a:b] = True
    return xy, inferred


@dataclass
class Rep:
    number: int                 # 1-based among counted reps
    start: int                  # frame index of the bottom (concentric start)
    peak: int                   # frame index of the top (concentric end)
    end: int                    # next bottom, or peak if the set ends at the top
    start_s: float
    peak_s: float
    concentric_s: float
    rom_px: float
    mean_speed: float           # px/s over the concentric phase
    peak_speed: float
    velocity_loss_pct: float | None = None   # vs reference mean speed (+ = slower)
    rom_pct: float | None = None             # vs reference ROM
    reference: bool = False


@dataclass
class Analysis:
    fps: float
    t: np.ndarray
    xy: np.ndarray              # bridged + smoothed point, NaN on long gaps
    observed: np.ndarray        # model-visible frames
    inferred: np.ndarray        # bridged frames
    s: np.ndarray               # position along main axis, + = up on screen
    v: np.ndarray               # px/s along main axis
    axis: np.ndarray
    prominence: float
    reps: list[Rep]
    ignored: int                # concentric moves too short to count
    reference_speed: float | None
    reference_rom: float | None
    phase: list[str] = field(repr=False, default_factory=list)

    def summary(self) -> dict:
        losses = [r.velocity_loss_pct for r in self.reps if r.velocity_loss_pct is not None]
        last2 = losses[-2:]
        return {
            "reps": len(self.reps),
            "ignored_partial_moves": self.ignored,
            "reference_reps": [r.number for r in self.reps if r.reference],
            "reference_mean_speed_px_s": self.reference_speed,
            "reference_rom_px": self.reference_rom,
            "last_reps_velocity_loss_pct": float(np.median(last2)) if last2 else None,
            "max_velocity_loss_pct": max(losses) if losses else None,
            "tracked_share": float(self.observed.mean()),
            "per_rep": [asdict(r) for r in self.reps],
        }


def analyze(t: np.ndarray, xy: np.ndarray, observed: np.ndarray, fps: float) -> Analysis:
    t = np.asarray(t, float)
    observed = np.asarray(observed, bool) & np.isfinite(xy).all(axis=1)
    if observed.sum() < 10:
        raise ValueError("the tracked point is visible in too few frames to analyse")
    bridged, inferred = bridge(t, xy, observed)
    ok = observed | inferred
    sm = smooth(t, bridged, ok)

    pts = sm[ok]
    center = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - center, full_matrices=False)
    axis = vt[0] if vt[0][1] < 0 else -vt[0]          # point "up" on screen
    s = (sm - center) @ axis
    v = velocity(t, s)

    span = float(np.nanpercentile(s, 95) - np.nanpercentile(s, 5))
    noise = float(np.nanmedian(np.abs(np.diff(s)))) * 3 + 1.0
    prominence = max(noise, PROMINENCE_FRAC * span)

    candidates = []
    for a, b in runs(ok):
        turns = find_turns(s[a:b], prominence)
        turns = [(i + a, k) for i, k in turns]
        for j, (i0, k0) in enumerate(turns[:-1]):
            i1, k1 = turns[j + 1]
            if k0 != "low" or k1 != "high":
                continue
            end = turns[j + 2][0] if j + 2 < len(turns) else i1
            candidates.append((i0, i1, end))

    moves = []
    for i0, i1, end in candidates:
        dur = t[i1] - t[i0]
        rom = s[i1] - s[i0]
        if dur < MIN_CONCENTRIC_S or rom < prominence:
            continue
        seg = v[i0:i1 + 1]
        moves.append(dict(start=i0, peak=i1, end=end, dur=dur, rom=rom,
                          mean=rom / dur, peak_v=float(np.nanmax(seg)) if np.isfinite(seg).any() else rom / dur))

    ref = moves[:REFERENCE_REPS]
    ref_rom = float(np.median([m["rom"] for m in ref])) if ref else None
    counted = [m for m in moves if ref_rom is None or m["rom"] >= COMPLETE_ROM * ref_rom]
    ignored = len(moves) - len(counted)
    ref = counted[:REFERENCE_REPS]
    ref_speed = float(np.median([m["mean"] for m in ref])) if len(ref) >= 2 else None
    ref_rom = float(np.median([m["rom"] for m in ref])) if len(ref) >= 2 else None

    reps = []
    for n, m in enumerate(counted, 1):
        reps.append(Rep(
            number=n, start=int(m["start"]), peak=int(m["peak"]), end=int(m["end"]),
            start_s=float(t[m["start"]]), peak_s=float(t[m["peak"]]),
            concentric_s=float(m["dur"]), rom_px=float(m["rom"]),
            mean_speed=float(m["mean"]), peak_speed=float(m["peak_v"]),
            velocity_loss_pct=None if ref_speed is None else float((1 - m["mean"] / ref_speed) * 100),
            rom_pct=None if ref_rom is None else float(m["rom"] / ref_rom * 100),
            reference=n <= REFERENCE_REPS and ref_speed is not None))

    phase = ["SETUP"] * len(t)
    for r in reps:
        for i in range(r.start, r.peak + 1):
            phase[i] = "CONCENTRIC"
        for i in range(r.peak + 1, r.end + 1):
            phase[i] = "ECCENTRIC"
    if reps:
        for i in range(reps[-1].end + 1, len(t)):
            phase[i] = "SET DONE"
        for a, b in zip(reps[:-1], reps[1:]):
            for i in range(a.end + 1, b.start):
                phase[i] = "HOLD"
    for i in np.flatnonzero(~ok):
        phase[i] = "TRACKING GAP"

    out_xy = sm.copy()
    out_xy[~ok] = np.nan
    return Analysis(fps=fps, t=t, xy=out_xy, observed=observed, inferred=inferred, s=s, v=v,
                    axis=axis, prominence=prominence, reps=reps, ignored=ignored,
                    reference_speed=ref_speed, reference_rom=ref_rom, phase=phase)
