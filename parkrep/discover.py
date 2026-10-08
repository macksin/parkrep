"""Light 'mesh' pass: Lucas-Kanade on a texture-adapted grid finds the point worth tracking.

The mesh only proposes *where* and *when* to query the neural tracker. Its own
trajectories never become the final track.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .reps import find_turns
from .video import VideoInfo, read_frames

LONG_SIDE = 384          # discovery resolution (long side, px)
GRID_STEP = 14           # px between seeded points at discovery resolution
RESEED_S = 1.5           # top up the mesh this often
MIN_TRACK_S = 2.0        # shorter trajectories are ignored
FB_MAX_ERR = 1.0         # forward-backward LK consistency, px
BORDER_PENALTY = 0.3     # score multiplier for tracks sitting at the frame edge (e.g. a knee cut off by the border)
BORDER_FRAC = 0.05      # "at the edge" = median position within this fraction of width/height
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


@dataclass
class Candidate:
    query_frame: int
    query_xy: tuple[float, float]   # full-resolution upright pixels
    start: int
    end: int
    amplitude_px: float             # full-resolution pixels along main axis
    reversals: int
    score: float


def _seed(gray: np.ndarray, existing: np.ndarray, roi_mask: np.ndarray | None) -> np.ndarray:
    mask = np.full(gray.shape, 255, np.uint8) if roi_mask is None else roi_mask.copy()
    for x, y in existing:
        cv2.circle(mask, (int(x), int(y)), GRID_STEP // 2, 0, -1)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=1500, qualityLevel=0.005,
                                  minDistance=GRID_STEP * 0.7, mask=mask, blockSize=5)
    return np.zeros((0, 2), np.float32) if pts is None else pts.reshape(-1, 2)


def _main_axis(xy: np.ndarray) -> tuple[np.ndarray, float]:
    centered = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    s = centered @ vt[0]
    return s, float(np.percentile(s, 95) - np.percentile(s, 5))


def discover(info: VideoInfo, roi: tuple[int, int, int, int] | None = None,
             top_k: int = 5, progress=None) -> list[Candidate]:
    scale = LONG_SIDE / max(info.width, info.height)
    w, h = round(info.width * scale / 2) * 2, round(info.height * scale / 2) * 2
    sx, sy = info.width / w, info.height / h
    roi_mask = None
    if roi:
        rx, ry, rw, rh = roi
        roi_mask = np.zeros((h, w), np.uint8)
        roi_mask[int(ry / sy):int((ry + rh) / sy), int(rx / sx):int((rx + rw) / sx)] = 255

    n = info.frames
    reseed_every = max(1, round(RESEED_S * info.fps))
    tracks: list[dict] = []          # {'start': int, 'xy': list[(x, y)]}
    active: list[int] = []           # indices into tracks
    camera = np.zeros((n, 2))        # cumulative global motion (median flow)
    prev = None
    for i, gray in enumerate(read_frames(info, (w, h), gray=True)):
        if prev is not None and active:
            p0 = np.array([tracks[k]["xy"][-1] for k in active], np.float32)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, **LK)
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, p1, None, **LK)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(p0 - p0r, axis=1) < FB_MAX_ERR)
            ok &= (p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h)
            flow = p1[ok] - p0[ok]
            camera[i] = camera[i - 1] + (np.median(flow, axis=0) if len(flow) > 20 else 0)
            still = []
            for k, good, xy in zip(active, ok, p1):
                if good:
                    tracks[k]["xy"].append((float(xy[0]), float(xy[1])))
                    still.append(k)
            active = still
        elif i > 0:
            camera[i] = camera[i - 1]
        if i % reseed_every == 0:
            current = np.array([tracks[k]["xy"][-1] for k in active]).reshape(-1, 2)
            for x, y in _seed(gray, current, roi_mask):
                tracks.append({"start": i, "xy": [(float(x), float(y))]})
                active.append(len(tracks) - 1)
        prev = gray
        if progress and (i % 30 == 0 or i == n - 1):
            progress(i + 1, n)

    min_len = max(8, round(MIN_TRACK_S * info.fps))
    cands = []
    for tr in tracks:
        xy = np.asarray(tr["xy"])
        if len(xy) < min_len:
            continue
        a, b = tr["start"], tr["start"] + len(xy)
        # Remove global (camera / handheld) motion before judging the point.
        # A real mover moves both before and after compensation; using the smaller
        # amplitude rejects static points that only "move" because a large body
        # dominated the camera estimate, and background that only shakes.
        rel = (xy - (camera[a:b] - camera[a])) * np.array([sx, sy])
        s, amp = _main_axis(rel)
        _, raw_amp = _main_axis(xy * np.array([sx, sy]))
        amp = min(amp, raw_amp)
        if amp < 0.04 * max(info.width, info.height) * 0.25:
            continue
        jitter = float(np.median(np.abs(np.diff(s, 2)))) + 1e-3
        turns = find_turns(s, prominence=0.4 * amp)
        reversals = len(turns)
        if reversals < 2:
            continue
        coverage = len(xy) / n
        score = amp * np.sqrt(min(reversals, 30)) * (0.3 + coverage) / (1 + jitter / 2)
        q = a + len(xy) // 2
        qx, qy = (xy[len(xy) // 2] + 0.5) * np.array([sx, sy]) - 0.5
        mx, my = (np.median(xy, axis=0) + 0.5) * np.array([sx, sy]) - 0.5
        if min(mx, info.width - mx) < BORDER_FRAC * info.width or min(my, info.height - my) < BORDER_FRAC * info.height:
            score *= BORDER_PENALTY
        cands.append(Candidate(q, (float(qx), float(qy)), a, b - 1, amp, reversals, float(score)))

    cands.sort(key=lambda c: -c.score)
    picked: list[Candidate] = []
    for c in cands:  # avoid near-duplicates of the same body part
        if all(np.hypot(c.query_xy[0] - p.query_xy[0], c.query_xy[1] - p.query_xy[1]) > 0.05 * info.width
               or abs(c.query_frame - p.query_frame) > info.fps * 3 for p in picked):
            picked.append(c)
        if len(picked) >= top_k:
            break
    return picked
