# ParkRep

Film your set at the park with your phone. Back home, one command turns the clip into an
Evangelion-style HUD video: reps counted, speed per rep, and how much you slowed down by the end.
Everything runs locally on a laptop (tested on a MacBook Air M3, 16 GB). Nothing is uploaded.

```bash
scripts/fetch_weights.sh            # LocoTrack-S weights, ~33 MB, checksum verified
uv sync
uv run parkrep my_pullups.mov --title "Pull-ups"
# -> out/my_pullups_parkrep.mp4, out/my_pullups_summary.json, out/my_pullups_summary.png
```

Requires `ffmpeg` (`brew install ffmpeg`) and Python 3.11–3.12.

## How it works

1. **Mesh (light model).** Lucas–Kanade optical flow on a texture-adapted grid of points
   follows everything that moves. Each trajectory is scored by how much it moves back and forth
   along one axis, after removing camera shake. The top candidates become queries.
2. **Neural tracker (open weights).** [LocoTrack-S](https://github.com/cvlab-kaist/locotrack)
   (Apache-2.0) tracks every candidate through the whole clip on Apple's GPU (MPS). The clip is
   cut into overlapping chunks so memory stays bounded; each chunk is encoded once and shared by
   all candidates. The track with the most consistent reps wins.
3. **Rep logic.** The point is smoothed, projected on its main axis (+ = up on screen) and split
   into reps with prominence hysteresis, so wobbles at a sticking point do not add reps.
   The first three reps are the fixed reference; every later rep is compared to them:
   velocity loss = 1 − mean concentric speed / reference speed.
4. **HUD.** Each frame only shows what was knowable at that moment: a rep's numbers appear when
   its top is reached. A summary card closes the video.

## Options

| Flag | Meaning |
| --- | --- |
| `--title` | Exercise name in the HUD |
| `--point X Y --frame N` | Track this pixel instead of auto-selecting |
| `--roi X Y W H` | Only look for the moving point inside this box |
| `--candidates N` | Points tried by the neural tracker (1–8, default 5) |
| `--device auto/mps/cpu` | Torch device |

## Limits

Speeds are image-plane pixels per second, not m/s; compare reps within one clip only.
Keep the phone still and the moving part (bar, hands, hips) visible. Velocity loss describes
the video; it is not a diagnosis of failure, RIR or RPE.

## License

Apache-2.0. `parkrep/locotrack/` is vendored LocoTrack model code (Apache-2.0, commit 3385a23).

## Future improvements

Found by running ParkRep on a real gym session (14 clips) and checking against the logged sets
in Hevy: 12 of 14 rep counts matched, and velocity loss rose with the logged RPE.

- **Robust reference.** The first "rep" can include getting set up (a 3.3 s concentric in a
  warmup clip), which skews every comparison. Use the median of reps 2–4, or drop a first rep
  that is far off the pattern.
- **Concentric direction.** The code assumes "up on screen" is the concentric phase. With a
  low camera under a leg extension this flips and mixes in pauses (a −237% velocity loss).
  Detect the direction from phase speed, or add `--concentric up|down`.
- **Low-coverage warning.** Both wrong counts (leg curl, 4 of 11–12; triceps, 3 of 12) had the
  point visible in only 58–60% of frames because the limb left the frame. Below ~80% coverage,
  warn instead of reporting a confident count.
- **Trim idle time.** Several clips start with walking in or end with an empty machine.
  Ignore motion outside the main run of reps.
