import json
from types import SimpleNamespace

import numpy as np
import pytest

from parkrep import video

FPS_RAW = "494400/16721"  # realistic iPhone avg_frame_rate (~29.57 fps)
FPS = 494400 / 16721


def fake_ffprobe(monkeypatch, stream, timestamps):
    """Make video.probe see `stream` and frames with the given timestamps (strings, like ffprobe)."""
    payload = {
        "streams": [stream],
        "frames": [{"best_effort_timestamp_time": ts} for ts in timestamps],
    }
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=json.dumps(payload))

    monkeypatch.setattr(video, "_tool", lambda name: name)
    monkeypatch.setattr(video.subprocess, "run", fake_run)
    return calls


def ten_frame_timestamps(start=0.0):
    return [f"{start + i / FPS:.6f}" for i in range(10)]


def test_rotation_from_display_matrix_survives_other_side_data(monkeypatch):
    # Real iPhone layout: the rotation entry is not the last side_data entry.
    stream = {
        "width": 1280,
        "height": 720,
        "avg_frame_rate": FPS_RAW,
        "side_data_list": [
            {"side_data_type": "DOVI configuration record"},
            {"side_data_type": "Display Matrix", "rotation": -90},
            {"side_data_type": "Ambient viewing environment"},
        ],
    }
    fake_ffprobe(monkeypatch, stream, ten_frame_timestamps())

    info = video.probe("clip.mov")

    assert (info.width, info.height) == (720, 1280)


def test_rotation_from_tags_rotate_swaps(monkeypatch):
    stream = {
        "width": 1920,
        "height": 1080,
        "avg_frame_rate": FPS_RAW,
        "tags": {"rotate": "90"},
    }
    fake_ffprobe(monkeypatch, stream, ten_frame_timestamps())

    info = video.probe("clip.mov")

    assert (info.width, info.height) == (1080, 1920)


def test_no_rotation_keeps_dimensions(monkeypatch):
    stream = {"width": 1920, "height": 1080, "avg_frame_rate": FPS_RAW}
    fake_ffprobe(monkeypatch, stream, ten_frame_timestamps())

    info = video.probe("clip.mov")

    assert (info.width, info.height) == (1920, 1080)
    assert info.frames == 10
    assert info.fps == pytest.approx(FPS)


def test_duplicate_and_non_monotonic_timestamps_are_fixed(monkeypatch):
    stream = {"width": 1920, "height": 1080, "avg_frame_rate": FPS_RAW}
    # Starts at 1.2s (must be shifted to 0), with a duplicate and a backwards step.
    raw = [1.2, 1.2338, 1.2338, 1.2676, 1.2610, 1.3014, 1.3352, 1.3352, 1.3690, 1.4028]
    fake_ffprobe(monkeypatch, stream, [f"{t:.6f}" for t in raw])

    info = video.probe("clip.mov")
    pts = info.pts

    assert len(pts) == 10
    assert pts[0] == 0.0
    assert np.all(np.diff(pts) > 0), f"timestamps not strictly increasing: {pts}"
    # A repaired duplicate is pushed forward by exactly one frame interval.
    assert pts[2] == pytest.approx(pts[1] + 1.0 / FPS)
