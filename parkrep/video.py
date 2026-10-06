"""ffmpeg/ffprobe helpers. Frames are decoded upright (rotation metadata applied)."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} not found on PATH (brew install ffmpeg)")
    return path


@dataclass
class VideoInfo:
    path: Path
    width: int  # upright (display) width
    height: int
    fps: float
    pts: np.ndarray  # seconds, one per decoded frame, starting at 0

    @property
    def frames(self) -> int:
        return len(self.pts)

    @property
    def duration(self) -> float:
        return float(self.pts[-1]) if len(self.pts) else 0.0


def probe(path: Path) -> VideoInfo:
    out = subprocess.run(
        [_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate:stream_side_data=rotation:stream_tags=rotate",
         "-show_entries", "frame=best_effort_timestamp_time",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    w, h = int(stream["width"]), int(stream["height"])
    rotation = 0
    for side in stream.get("side_data_list", []):
        rotation = int(float(side.get("rotation", 0)))
    rotation = rotation or int(stream.get("tags", {}).get("rotate", 0))
    if abs(rotation) % 180 == 90:
        w, h = h, w
    num, den = stream["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    pts = np.array([float(f["best_effort_timestamp_time"]) for f in data["frames"]
                    if f.get("best_effort_timestamp_time") not in (None, "N/A")])
    if len(pts) < 2:
        raise ValueError(f"could not read frame timestamps from {path}")
    pts = pts - pts[0]
    # Guard against duplicate/non-monotonic timestamps from some phones.
    for i in range(1, len(pts)):
        if pts[i] <= pts[i - 1]:
            pts[i] = pts[i - 1] + 1.0 / fps
    return VideoInfo(Path(path), w, h, fps, pts)


def read_frames(info: VideoInfo, size: tuple[int, int] | None = None, gray: bool = False):
    """Yield upright frames as uint8 arrays, optionally resized to (w, h)."""
    w, h = size or (info.width, info.height)
    channels = 1 if gray else 3
    argv = [_tool("ffmpeg"), "-v", "error", "-i", str(info.path), "-map", "0:v:0", "-an",
            "-vsync", "passthrough"]
    if size:
        argv += ["-vf", f"scale={w}:{h}:flags=area"]
    argv += ["-f", "rawvideo", "-pix_fmt", "gray" if gray else "rgb24", "-"]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    nbytes = w * h * channels
    try:
        for _ in range(info.frames):
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            frame = np.frombuffer(buf, np.uint8)
            yield frame.reshape(h, w) if gray else frame.reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


class VideoWriter:
    """H.264 MP4 writer fed with RGB frames through a pipe."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float, crf: int = 20):
        self.size = size
        self.proc = subprocess.Popen(
            [_tool("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:.6f}", "-i", "-",
             "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding the output video")
