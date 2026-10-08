"""From a tracked point to reps: smoothing, main axis, hysteresis turns, per-rep metrics.

Units are image pixels and seconds. Nothing here is calibrated to the real world;
velocity loss is a *relative* comparison against an early reference (reps 2..4) of the same clip.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

SMOOTH_HALF_S = 0.10      # temporal median half-window
VEL_HALF_S = 0.15         # local linear-fit half-window for velocity
MAX_BRIDGE_S = 0.35       # gaps up to this long are bridged (marked as inferred)
PROMINENCE_FRAC = 0.35    # reversal must exceed this share of the motion range
MIN_CONCENTRIC_S = 0.25
REFERENCE_REPS = 3        # the reference is N reps: reps 2..4 when 4+ reps, else the first N
RUN_ROM_LO = 0.60         # main run: a move's ROM must be within this band of the median ROM
RUN_ROM_HI = 2.00
RUN_LEVEL = 0.35          # main run: bottom or top level (and return) within this share of the median ROM
RUN_GAP = 3.50            # main run: consecutive moves start at most this many median periods apart (rest-pause allowed)
RUN_MIN_CHAIN = 3         # main run: chains shorter than this (isolated moves) are adjustments
LOW_COVERAGE = 0.80       # warn when the point is visible in less of the clip than this
GAP_WINDOW_S = 3.0        # window length for local tracking-loss warnings
GAP_WINDOW_MIN = 0.50     # a window with less visible share than this is reported
MIN_ROM_FRAC = 0.04       # reps shorter than this share of the frame's long side are suspicious
MAX_INFERRED_SHARE = 0.30 # a move whose start..peak is mostly bridged is not counted
EDGE_FRAC = 0.05          # concentric timing: bottom/top edge band as a share of the ROM (plateau-safe)


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


def _cross_time(t: np.ndarray, s: np.ndarray, a: int, b: int, level: float) -> float:
    """Time at which s, moving linearly from frame a to adjacent frame b, reaches `level`."""
    d = s[b] - s[a]
    f = 0.0 if d <= 0 else float(np.clip((level - s[a]) / d, 0.0, 1.0))
    return float(t[a] + f * (t[b] - t[a]))


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
    warnings: list[str] = field(default_factory=list)
    adjustments: list[dict] = field(default_factory=list)   # moves outside the main run, with a reason

    def summary(self) -> dict:
        losses = [r.velocity_loss_pct for r in self.reps if r.velocity_loss_pct is not None]
        last2 = losses[-2:]
        return {
            "reps": len(self.reps),
            "ignored_partial_moves": self.ignored,
            "adjustments": self.adjustments,
            "reference_reps": [r.number for r in self.reps if r.reference],
            "reference_mean_speed_px_s": self.reference_speed,
            "reference_rom_px": self.reference_rom,
            "last_reps_velocity_loss_pct": float(np.median(last2)) if last2 else None,
            "max_velocity_loss_pct": max(losses) if losses else None,
            "tracked_share": float(self.observed.mean()),
            "warnings": self.warnings,
            "per_rep": [asdict(r) for r in self.reps],
        }


def visibility_spans(t: np.ndarray, observed: np.ndarray, t0: float) -> list[tuple[float, float]]:
    """Merged time spans covered by a GAP_WINDOW_S window whose visible share is too low."""
    spans: list[list[float]] = []
    for a in np.arange(t0, max(t[-1] - GAP_WINDOW_S, t0) + 1e-9, 1.0):
        m = (t >= a) & (t < a + GAP_WINDOW_S)
        if not m.any() or observed[m].mean() >= GAP_WINDOW_MIN:
            continue
        if spans and a <= spans[-1][1] + 1e-9:
            spans[-1][1] = a + GAP_WINDOW_S
        else:
            spans.append([a, a + GAP_WINDOW_S])
    return [(a, b) for a, b in spans]


def analyze(t: np.ndarray, xy: np.ndarray, observed: np.ndarray, fps: float,
            frame_size: tuple[int, int] | None = None) -> Analysis:
    """frame_size=(width, height) in px enables the absolute amplitude check."""
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
        s_lo, s_hi = s[i0], s[i1]
        rom = s_hi - s_lo
        if rom < prominence:
            continue
        # refine timing inside i0..i1 so a noisy plateau at either turn does not inflate the concentric
        band = EDGE_FRAC * rom
        k_end = i0 + int(np.flatnonzero(s[i0:i1 + 1] >= s_hi - band)[0])          # first index near the top
        k_start = i0 + int(np.flatnonzero(s[i0:k_end + 1] <= s_lo + band)[-1])    # last index still at the bottom
        k_hold = k_end + int(np.flatnonzero(s[k_end:end + 1] >= s_hi - band)[-1]) # last index of the top pause
        # sub-frame edge times: linear interpolation of each band crossing, so identical reps get identical durations
        t_s = _cross_time(t, s, k_start, k_start + 1, s_lo + band) if k_start < k_end else t[k_start]
        t_e = _cross_time(t, s, k_end - 1, k_end, s_hi - band) if k_end > k_start else t[k_end]
        dur = max(t_e - t_s, 1 / fps)
        if dur < MIN_CONCENTRIC_S:
            continue
        seg = v[k_start:k_end + 1]
        moves.append(dict(start=k_start, peak=k_end, hold=k_hold, end=end, dur=dur, rom=rom,
                          base=float(s_lo), mean=rom / dur,
                          peak_v=float(np.nanmax(seg)) if np.isfinite(seg).any() else rom / dur))

    # B6: a move that starts in a gap or whose start..peak is mostly bridged is not counted
    for m in moves:
        m["gap"] = bool(not ok[m["start"]] or inferred[m["start"]:m["peak"] + 1].mean() > MAX_INFERRED_SHARE)
    gap_moves = sum(m["gap"] for m in moves)

    # main run: the longest chain of mutually regular moves; every other move is an adjustment
    ok_moves = [m for m in moves if not m["gap"]]
    R = float(np.median([m["rom"] for m in ok_moves])) if ok_moves else 0.0
    B = float(np.median([m["base"] for m in ok_moves])) if ok_moves else 0.0
    T = float(np.median([m["base"] + m["rom"] for m in ok_moves])) if ok_moves else 0.0
    P = float(np.median(np.diff([t[m["start"]] for m in ok_moves]))) if len(ok_moves) >= 2 else None
    # return = deepest observed point between this peak and the next move (the eccentric is often
    # partly out of view, so an unobserved return is "unknown", not "does not return")
    for j, m in enumerate(moves):
        stop = moves[j + 1]["start"] if j + 1 < len(moves) else len(t)
        after = s[m["peak"]:stop]
        m["drop"] = float(s[m["peak"]] - np.nanmin(after)) if np.isfinite(after).any() else None
    for m in ok_moves:
        if not RUN_ROM_LO * R <= m["rom"] <= RUN_ROM_HI * R:
            m["why"] = "different range"
        # a partial keeps the bottom, a rep pulled from a stretch keeps the top; standing up keeps neither
        elif abs(m["base"] - B) > RUN_LEVEL * R and abs(m["base"] + m["rom"] - T) > RUN_LEVEL * R:
            m["why"] = "different start level"
        else:
            # the last move is exempt: after the final rep people rack, release or stand up, so
            # "no return" there does not tell a real rep from an adjustment
            returns = m["drop"] is None or m["drop"] >= (1 - RUN_LEVEL) * min(m["rom"], R)
            m["why"] = "" if returns or m is moves[-1] else "does not return"
    consistent = [m for m in ok_moves if not m["why"]]
    chains: list[list[dict]] = []
    for m in consistent:
        if chains and (P is None or t[m["start"]] - t[chains[-1][-1]["start"]] <= RUN_GAP * P):
            chains[-1].append(m)
        else:
            chains.append([m])
    # every chain of RUN_MIN_CHAIN+ moves belongs to the set (a rest-pause splits a set into chains);
    # only short isolated chains are adjustments. With no long chain, keep the longest one.
    long_chains = [c for c in chains if len(c) >= RUN_MIN_CHAIN]
    run = [m for c in long_chains for m in c] if long_chains else max(chains, key=len, default=[])

    # with 4+ counted reps the reference is reps 2..4; otherwise the first REFERENCE_REPS
    first = 1 if len(run) >= 4 else 0
    ref = run[first:first + REFERENCE_REPS]
    ref_speed = float(np.median([m["mean"] for m in ref])) if len(ref) >= 2 else None
    ref_rom = float(np.median([m["rom"] for m in ref])) if len(ref) >= 2 else None
    ref_ids = {id(m) for m in ref} if ref_speed is not None else set()

    run_ids = {id(m) for m in run}
    adjustments = []
    for m in ok_moves:
        if id(m) not in run_ids:
            reason = m["why"] or "out of rhythm"
            adjustments.append(dict(start=int(m["start"]), end=int(m["end"]), start_s=float(t[m["start"]]),
                                    end_s=float(t[m["end"]]), reason=reason))
    ignored = gap_moves + len(adjustments)

    reps = []
    for n, m in enumerate(run, 1):
        reps.append(Rep(
            number=n, start=int(m["start"]), peak=int(m["peak"]), end=int(m["end"]),
            start_s=float(t[m["start"]]), peak_s=float(t[m["peak"]]),
            concentric_s=float(m["dur"]), rom_px=float(m["rom"]),
            mean_speed=float(m["mean"]), peak_speed=float(m["peak_v"]),
            velocity_loss_pct=None if ref_speed is None else float((1 - m["mean"] / ref_speed) * 100),
            rom_pct=None if ref_rom is None else float(m["rom"] / ref_rom * 100),
            reference=id(m) in ref_ids))

    warnings = []
    if observed.mean() < LOW_COVERAGE:
        warnings.append(f"point visible in only {observed.mean():.0%} of frames; rep count may be low")
    t0 = float(t[moves[0]["start"]]) if moves else float(t[0])   # tracking loss counts from the first move, not the run
    spans = visibility_spans(t, observed, t0)
    if spans:
        ranges = ", ".join(f"{a:.0f}-{b:.0f} s" for a, b in spans)
        warnings.append(f"tracking lost around {ranges}; reps there may be missing")
    if frame_size and run:
        med_rom = float(np.median([m["rom"] for m in run]))
        long_side = max(frame_size)
        if med_rom < MIN_ROM_FRAC * long_side:
            warnings.append(f"reps move only {med_rom:.0f} px ({med_rom / long_side:.1%} of the frame); "
                            "count and speeds are unreliable")
    if adjustments:
        reasons = sorted({a["reason"] for a in adjustments})
        warnings.append(f"{len(adjustments)} adjustment moves not counted ({', '.join(reasons)})")
    if len(run) < 3:
        warnings.append("no regular set found (fewer than 3 consistent reps); check the point")
    if gap_moves:
        warnings.append(f"{gap_moves} moves overlapped tracking gaps and were not counted")

    hold_at = {int(m["start"]): int(m["hold"]) for m in run}
    phase = ["SETUP"] * len(t)
    for r in reps:
        for i in range(r.start, r.peak + 1):
            phase[i] = "CONCENTRIC"
        for i in range(r.peak + 1, hold_at[r.start] + 1):   # pause at the top
            phase[i] = "HOLD"
        for i in range(hold_at[r.start] + 1, r.end + 1):
            phase[i] = "ECCENTRIC"
    if reps:
        for i in range(reps[-1].end + 1, len(t)):
            phase[i] = "SET DONE"
        for a, b in zip(reps[:-1], reps[1:]):
            for i in range(a.end + 1, b.start):
                phase[i] = "HOLD"
    for a in adjustments:                                   # only over setup/hold/done, never over a counted rep
        for i in range(a["start"], a["end"] + 1):
            if phase[i] in ("SETUP", "HOLD", "SET DONE"):
                phase[i] = "ADJUST"
    for i in np.flatnonzero(~ok):
        phase[i] = "TRACKING GAP"

    out_xy = sm.copy()
    out_xy[~ok] = np.nan
    return Analysis(fps=fps, t=t, xy=out_xy, observed=observed, inferred=inferred, s=s, v=v,
                    axis=axis, prominence=prominence, reps=reps, ignored=ignored,
                    reference_speed=ref_speed, reference_rom=ref_rom, phase=phase, warnings=warnings,
                    adjustments=adjustments)
