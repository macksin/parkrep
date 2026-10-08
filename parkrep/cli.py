"""parkrep VIDEO -> HUD video + summary, all on this machine."""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(f"[parkrep] {msg}", file=sys.stderr, flush=True)


def bar(label: str):
    def update(i, n):
        width = 28
        k = int(width * i / max(n, 1))
        print(f"\r[parkrep] {label:<10} [{'#' * k}{'.' * (width - k)}] {i}/{n}", end="", file=sys.stderr, flush=True)
        if i >= n:
            print(file=sys.stderr)
    return update


def track_quality(a) -> float:
    """Prefer the track whose main run of reps is long, large, regular and clean."""
    reps = a.reps
    n = len(reps)
    if n < 2:
        return 0.0
    roms = np.array([r.rom_px for r in reps], float)
    rom = float(np.median(roms))
    cv_rom = float(np.std(roms) / np.mean(roms))
    gaps = np.diff([r.start_s for r in reps])
    cv_period = float(np.std(gaps) / np.mean(gaps)) if len(gaps) >= 2 else 0.0
    coverage_run = float(np.asarray(a.observed)[reps[0].start:reps[-1].end + 1].mean())
    regularity = max(0.2, 1 - cv_rom - cv_period)
    purity = n / (n + len(getattr(a, "adjustments", [])))
    return n * rom * coverage_run * regularity * purity


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="parkrep", description=__doc__)
    p.add_argument("video", type=Path, help="phone video of one set")
    p.add_argument("-o", "--out", type=Path, default=Path("out"), help="output folder (default: ./out)")
    p.add_argument("--title", help="exercise name shown in the HUD (default: file name)")
    p.add_argument("--point", type=float, nargs=2, metavar=("X", "Y"),
                   help="track this pixel instead of auto-selecting (upright video pixels)")
    p.add_argument("--frame", type=int, default=0, help="frame where --point is located (default 0)")
    p.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                   help="only look for the moving point inside this box")
    p.add_argument("--candidates", type=int, default=5, help="points tried by the neural tracker (1-8)")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    p.add_argument("--hold", type=float, default=4.0, help="seconds the summary card stays on screen")
    p.add_argument("--save-candidates", action="store_true",
                   help="also save every candidate track to <stem>_candidates.npz (offline evaluation)")
    p.add_argument("--no-video", action="store_true", help="skip the HUD video and summary card")
    args = p.parse_args(argv)

    from .discover import discover
    from .hud import HUD
    from .reps import analyze
    from .track import track_points
    from .video import VideoWriter, probe, read_frames

    started = time.perf_counter()
    if not args.video.is_file():
        p.error(f"no such file: {args.video}")
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem
    info = probe(args.video)
    log(f"{args.video.name}: {info.width}x{info.height}, {info.fps:.1f} fps, {info.duration:.1f} s")

    if args.point:
        queries = [(args.frame, tuple(args.point))]
        log(f"using the point you gave: ({args.point[0]:.0f}, {args.point[1]:.0f}) @ frame {args.frame}")
    else:
        cands = discover(info, roi=tuple(args.roi) if args.roi else None,
                         top_k=max(1, min(8, args.candidates)), progress=bar("mesh"))
        if not cands:
            log("the mesh found nothing that moves back and forth; try --roi or --point")
            return 1
        queries = [(c.query_frame, c.query_xy) for c in cands]
        log(f"mesh proposed {len(cands)} candidate points")

    tracks = track_points(info, queries, device=args.device, progress=bar("locotrack"))
    if args.save_candidates:
        np.savez_compressed(args.out / f"{stem}_candidates.npz", t=info.pts, fps=info.fps,
                            size=np.array([info.width, info.height]),
                            queries=np.array([[f, x, y] for f, (x, y) in queries], float),
                            xy=np.stack([tr.xy for tr in tracks]), visible=np.stack([tr.visible for tr in tracks]))
    extra = {"frame_size": (info.width, info.height)} if "frame_size" in inspect.signature(analyze).parameters else {}
    best = None
    for k, tr in enumerate(tracks):
        try:
            a = analyze(info.pts, tr.xy, tr.visible, info.fps, **extra)
        except ValueError:
            continue
        q = track_quality(a)
        log(f"  candidate {k + 1}: {len(a.reps)} reps, visible {tr.visible.mean():.0%}, score {q:.0f}")
        if best is None or q > best[0]:
            best = (q, k, a, tr)
    if best is None:
        log("no candidate produced a usable track")
        return 1
    _, k, a, tr = best
    warnings = list(getattr(a, "warnings", []) or [])
    log(f"selected candidate {k + 1}: {len(a.reps)} reps")
    for w in warnings:
        log(f"warning: {w}")

    title = args.title or stem.replace("_", " ")
    qf, (qx, qy) = queries[k]
    hud = HUD(a, info.width, info.height, title,
              f"POINT ({qx:.0f}, {qy:.0f}) @ {info.pts[qf]:.1f}s  |  {len(queries)} CANDIDATES  |  TRACKED {tr.visible.mean():.0%}")
    video_path = args.out / f"{stem}_parkrep.mp4"
    if not args.no_video:
        writer = VideoWriter(video_path, (720, 1280), info.fps)
        progress = bar("render")
        last = None
        for i, rgb in enumerate(read_frames(info)):
            last = hud.draw(rgb, i)
            writer.write(np.asarray(last))
            if i % 15 == 0 or i == info.frames - 1:
                progress(i + 1, info.frames)
        card = hud.summary_card(last)
        for _ in range(round(args.hold * info.fps)):
            writer.write(np.asarray(card))
        writer.close()
        card.save(args.out / f"{stem}_summary.png")

    summary = dict(video=str(args.video), title=title, fps=info.fps, frames=info.frames,
                   query=dict(frame=int(qf), x=float(qx), y=float(qy)), candidates=len(queries),
                   tracker="LocoTrack-S", **a.summary())
    (args.out / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    np.savez_compressed(args.out / f"{stem}_track.npz", t=info.pts, xy=tr.raw_xy, score=tr.score,
                        visible=tr.visible, s=a.s, v=a.v)
    log(f"done in {time.perf_counter() - started:.0f} s -> {video_path}")
    print(json.dumps({**{k: summary[k] for k in ("reps", "last_reps_velocity_loss_pct", "max_velocity_loss_pct")},
                      "warnings": warnings}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
