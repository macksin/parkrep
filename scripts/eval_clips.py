"""Offline evaluation of rep counting and candidate selection from cached candidate tracks.

Reads a session folder with labels.json (ground truth per clip) and cands/IMG_XXXX_candidates.npz
(written by `parkrep --save-candidates`). For every clip, runs parkrep.reps.analyze on each candidate
track, picks the best one the same way parkrep.cli.main does (highest track_quality), and compares
the counted reps with the labelled Hevy reps. No GPU or video needed.

Usage:
    uv run python scripts/eval_clips.py ~/Downloads/TREINOSHOJE
    uv run python scripts/eval_clips.py ~/Downloads/TREINOSHOJE -v            # per-candidate tables
    uv run python scripts/eval_clips.py ~/Downloads/TREINOSHOJE --json out.json  # full rows for diffing
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np

from parkrep.cli import track_quality
from parkrep.reps import analyze


def _get(obj, name, default=None):
    """Attribute or dict-key lookup, so adjustments may be dataclasses or plain dicts."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _round(x, nd=1):
    return None if x is None else round(float(x), nd)


def _accepts_frame_size() -> bool:
    return "frame_size" in inspect.signature(analyze).parameters


def _run_candidate(k, npz, takes_size):
    """Analyse one candidate. Returns (record, analysis-or-None)."""
    t = npz["t"]
    xy = npz["xy"][k]
    vis = npz["visible"][k].astype(bool)
    fps = float(npz["fps"])
    qf, qx, qy = (float(v) for v in npz["queries"][k])
    rec = dict(k=k + 1, query=dict(frame=int(qf), x=_round(qx), y=_round(qy)),
               coverage=_round(vis.mean(), 3), reps=None, score=None, error=None)
    kwargs = {"frame_size": tuple(int(v) for v in npz["size"])} if takes_size else {}
    try:
        a = analyze(t, xy, vis, fps, **kwargs)
    except ValueError as e:
        rec["error"] = str(e)
        return rec, None
    rec["reps"] = len(a.reps)
    rec["score"] = _round(track_quality(a), 0)
    return rec, a


def evaluate_clip(name, label, npz, takes_size, detail=True):
    """Build one result row for a clip. Returns a dict (JSON-serialisable)."""
    cands, best = [], None
    for k in range(npz["queries"].shape[0]):
        rec, a = _run_candidate(k, npz, takes_size)
        cands.append(rec)
        if a is None:
            continue
        q = track_quality(a)
        if best is None or q > best[0]:
            best = (q, k, a)

    hevy = label.get("reps")
    row = dict(clip=name, exercise=label.get("exercise", ""), hevy_reps=hevy, rpe=label.get("rpe"),
               keep_slow_tail=bool(label.get("keep_slow_tail", False)), status="ok", candidates=cands)
    if best is None:
        row.update(status="no_candidate", counted=None, ok=False, chosen=None, warnings=[],
                   adjustments=[], last_reps_velocity_loss_pct=None, per_rep=[], slow_tail_concentric_s=[])
        return row

    _, k, a = best
    summ = a.summary()
    counted = len(a.reps)
    if isinstance(hevy, (list, tuple)):
        lo, hi = float(hevy[0]), float(hevy[1])
    else:
        lo = hi = float(hevy) if hevy is not None else float("nan")
    ok = bool(lo <= counted <= hi)
    warnings = list(getattr(a, "warnings", []) or [])
    adjustments = [dict(reason=_get(adj, "reason", str(adj)), start_s=_round(_get(adj, "start_s"), 2))
                   for adj in (getattr(a, "adjustments", []) or [])]
    per_rep = [dict(n=r.number, start_s=_round(r.start_s, 2), concentric_s=_round(r.concentric_s, 2),
                    rom_px=_round(r.rom_px, 0), velocity_loss_pct=_round(r.velocity_loss_pct, 1),
                    reference=bool(r.reference)) for r in a.reps]
    tail = [r["concentric_s"] for r in per_rep[-2:]] if row["keep_slow_tail"] else []
    row.update(status="ok", counted=counted, ok=ok, warnings=warnings, adjustments=adjustments,
               last_reps_velocity_loss_pct=_round(summ["last_reps_velocity_loss_pct"]),
               per_rep=per_rep, slow_tail_concentric_s=tail,
               chosen=dict(cands[k], index=k + 1))
    return row


def _fmt_reps(hevy):
    if isinstance(hevy, (list, tuple)):
        return f"{hevy[0]}-{hevy[1]}"
    return "-" if hevy is None else str(hevy)


def print_row(row, detail):
    mark = "✓" if row["ok"] else "✗"
    if row["status"] != "ok":
        print(f"{row['clip']:<10} {row['exercise']:<22} hevy {_fmt_reps(row['hevy_reps']):<6} NO USABLE CANDIDATE")
    else:
        ch = row["chosen"]
        q = ch["query"]
        vl = "-" if row["last_reps_velocity_loss_pct"] is None else f"{row['last_reps_velocity_loss_pct']:.1f}%"
        adj = "; ".join(f"{a['reason']}@{a['start_s']}s" for a in row["adjustments"]) or "-"
        tail = ""
        if row["keep_slow_tail"]:
            tail = "  tail_concentric_s=" + (",".join(f"{x:.2f}" for x in row["slow_tail_concentric_s"]) or "-")
        print(f"{row['clip']:<10} {row['exercise']:<22} hevy {_fmt_reps(row['hevy_reps']):<6} "
              f"counted {row['counted']:<3} {mark}  cand {ch['k']}/{len(row['candidates'])} "
              f"q=({q['frame']},{q['x']},{q['y']}) cov {ch['coverage']:.0%}  vloss_last2 {vl}  "
              f"adj {len(row['adjustments'])} [{adj}]  warn {len(row['warnings'])}{tail}")
        for w in row["warnings"]:
            print(f"    warning: {w}")
    if detail:
        print(f"    {'k':>2} {'frame':>5} {'x':>7} {'y':>7} {'cov':>5} {'reps':>4} {'score':>7}  error")
        for c in row["candidates"]:
            q = c["query"]
            mark_c = "*" if row["status"] == "ok" and c["k"] == row["chosen"]["k"] else " "
            err = c["error"] or ""
            print(f"  {mark_c} {c['k']:>2} {q['frame']:>5} {q['x']:>7} {q['y']:>7} "
                  f"{(c['coverage'] or 0):>5.0%} {'-' if c['reps'] is None else c['reps']:>4} {'-' if c['score'] is None else c['score']:>7}  {err[:60]}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Offline rep-count evaluation from cached candidate tracks.")
    p.add_argument("session", type=Path, help="session folder with labels.json and cands/")
    p.add_argument("-v", "--verbose", action="store_true", help="print a per-candidate table under each clip")
    p.add_argument("--json", type=Path, metavar="PATH", help="write all rows to PATH as JSON")
    args = p.parse_args(argv)

    labels_path = args.session / "labels.json"
    if not labels_path.is_file():
        p.error(f"no labels.json in {args.session}")
    labels = json.loads(labels_path.read_text())
    cdir = args.session / "cands"
    takes_size = _accepts_frame_size()

    rows, missing = [], []
    print(f"session: {labels.get('session', args.session.name)}")
    print(f"{'clip':<10} {'exercise':<22} {'hevy':<11} result")
    for name, label in labels.get("clips", {}).items():
        f = cdir / f"{name}_candidates.npz"
        if not f.is_file():
            missing.append(name)
            print(f"{name:<10} {label.get('exercise', ''):<22} hevy {_fmt_reps(label.get('reps')):<6} (no candidates yet, skipped)")
            continue
        with np.load(f) as npz:
            data = {key: npz[key] for key in npz.files}
        row = evaluate_clip(name, label, data, takes_size)
        rows.append(row)
        print_row(row, args.verbose)

    matches = sum(1 for r in rows if r["ok"])
    wrong_silent = [r["clip"] for r in rows if not r["ok"] and not r["warnings"]]
    total_adj = sum(len(r["adjustments"]) for r in rows)
    print()
    print(f"matches {matches}/{len(rows)} clips" + (f" ({len(missing)} missing: {', '.join(missing)})" if missing else ""))
    print(f"wrong count with no warning: {', '.join(wrong_silent) if wrong_silent else 'none'}")
    print(f"total adjustments: {total_adj}")

    if args.json:
        args.json.write_text(json.dumps(dict(session=labels.get("session"), matches=matches, total=len(rows),
                                             missing=missing, wrong_without_warning=wrong_silent,
                                             total_adjustments=total_adj, rows=rows),
                                        indent=2, default=float, ensure_ascii=False))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
