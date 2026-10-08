"""LocoTrack-S point tracking over the whole clip, in bounded-memory chunks."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .video import VideoInfo, read_frames

MODEL_RES = 256
CHUNK = 240          # frames per encoder pass (~3.5 GB peak for LocoTrack-S fp32)
OVERLAP = 24         # frames shared by consecutive chunks; used to hand the query over
MIN_SCORE = 0.4      # (1 - p_occluded) * (1 - p_far); uncalibrated
CKPT_URL = "https://huggingface.co/datasets/hamacojr/LocoTrack-pytorch-weights/resolve/main/locotrack_small.ckpt"
CKPT_SHA256 = "da023594e6d6c05ecad9644efc1467545481cfa899e20730bd9fdce778ffa5ac"
DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "weights" / "locotrack_small.ckpt"


@dataclass
class Track:
    xy: np.ndarray       # (n, 2) upright full-res pixels, NaN where not visible
    raw_xy: np.ndarray   # (n, 2) model output everywhere
    score: np.ndarray    # (n,)
    visible: np.ndarray  # (n,) bool


def load_model(checkpoint: Path, device: str):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    from .locotrack.locotrack_model import LocoTrack

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"LocoTrack-S weights not found at {checkpoint}. Run: scripts/fetch_weights.sh")
    payload = checkpoint.read_bytes()
    if hashlib.sha256(payload).hexdigest() != CKPT_SHA256:
        raise ValueError(f"unexpected SHA-256 for {checkpoint}; re-download it")
    import io
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {k.removeprefix("model."): v for k, v in state.items()}
    model = LocoTrack(model_size="small", feature_extractor_chunk_size=16)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def pick_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _chunk_video(frames: np.ndarray, device: str):
    import torch
    return torch.from_numpy(frames).to(device, dtype=torch.float32)[None] / 127.5 - 1.0


def _run_queries(model, video, grids, queries: list[tuple[int, np.ndarray]], device: str):
    """queries: (chunk-relative frame, model-raster xy). Returns xy (k, t, 2), score (k, t)."""
    import torch
    q = torch.tensor([[[f, xy[1], xy[0]] for f, xy in queries]], device=device, dtype=torch.float32)
    with torch.inference_mode():
        out = model(video, q, feature_grids=grids, query_chunk_size=len(queries))
        xy = out["tracks"][0].float().cpu().numpy()
        score = ((1 - torch.sigmoid(out["occlusion"])) * (1 - torch.sigmoid(out["expected_dist"])))[0]
    return xy, score.float().cpu().numpy()


def track_points(info: VideoInfo, queries: list[tuple[int, tuple[float, float]]],
                 checkpoint: Path = DEFAULT_CKPT, device: str = "auto", progress=None) -> list[Track]:
    """Track several (frame, (x, y)) queries through the whole clip.

    The clip is cut into overlapping chunks so memory stays bounded. Each chunk is
    encoded once and shared by all queries. A point is handed from one chunk to
    the next through the overlap: the best-scoring overlap frame becomes the new query.
    """
    import torch
    device = pick_device(device)
    model = load_model(checkpoint, device)
    n, k = info.frames, len(queries)
    to_model = np.array([MODEL_RES / info.width, MODEL_RES / info.height])
    frames = np.empty((n, MODEL_RES, MODEL_RES, 3), np.uint8)
    for i, f in enumerate(read_frames(info, (MODEL_RES, MODEL_RES))):
        frames[i] = f

    step = CHUNK - OVERLAP
    starts = list(range(0, max(1, n - OVERLAP), step))
    bounds = [(a, min(n, a + CHUNK)) for a in starts]
    home = [max(c for c, (a, _) in enumerate(bounds) if a <= qf) for qf, _ in queries]
    raw = np.full((k, n, 2), np.nan)
    score = np.zeros((k, n))
    chunk_xy: dict[tuple[int, int], np.ndarray] = {}
    done = [0]
    backward = sum(1 for c in range(len(bounds) - 1) if any(c < h for h in home))
    total = len(bounds) - min(home) + backward

    def process(c, items):  # items: list of (query index, absolute frame, model xy)
        if not items:
            return
        a, b = bounds[c]
        video = _chunk_video(frames[a:b], device)
        with torch.inference_mode():
            grids = model.get_feature_grids(video, is_training=False)
        xy, sc = _run_queries(model, video, grids, [(f - a, q) for _, f, q in items], device)
        for row, (qi, _, _) in enumerate(items):
            better = np.isnan(raw[qi, a:b, 0]) | (sc[row] > score[qi, a:b])
            raw[qi, a:b][better] = xy[row][better]
            score[qi, a:b][better] = sc[row][better]
            chunk_xy[qi, c] = xy[row]
        del video, grids
        if device == "mps":
            torch.mps.empty_cache()
        done[0] += 1
        if progress:
            progress(done[0], total)

    def handover(qi, c_from, lo, hi):
        a = bounds[c_from][0]
        j = lo + int(np.argmax(score[qi, lo:hi]))
        return j, chunk_xy[qi, c_from][j - a]

    for c in range(len(bounds)):  # forward sweep
        items = []
        for qi, (qf, qxy) in enumerate(queries):
            if c == home[qi]:
                items.append((qi, qf, (np.asarray(qxy) + 0.5) * to_model - 0.5))
            elif c > home[qi]:
                items.append((qi, *handover(qi, c - 1, bounds[c][0], bounds[c - 1][1])))
        process(c, items)
    for c in range(len(bounds) - 2, -1, -1):  # backward sweep
        items = [(qi, *handover(qi, c + 1, bounds[c + 1][0], bounds[c][1]))
                 for qi in range(k) if c < home[qi]]
        process(c, items)

    tracks = []
    for qi in range(k):
        full = (raw[qi] + 0.5) / to_model - 0.5
        visible = (score[qi] >= MIN_SCORE) & np.isfinite(full).all(axis=1)
        visible &= (full[:, 0] >= 0) & (full[:, 0] < info.width) & (full[:, 1] >= 0) & (full[:, 1] < info.height)
        xy = full.copy()
        xy[~visible] = np.nan
        tracks.append(Track(xy=xy, raw_xy=full, score=score[qi], visible=visible))
    return tracks
