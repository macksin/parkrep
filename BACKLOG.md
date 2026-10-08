# ParkRep backlog

Findings from running ParkRep on the 2026-10-07 gym session (Hevy: "D3 Smart Fit · Lats + RDL",
11 clips, 1280x720 HEVC with −90° rotation). Rep counts matched Hevy in 7 of 11 clips.
Evidence images live in `docs/backlog/` (contact sheets; the yellow label is the time in seconds).

Clip → set mapping (checked visually, frame by frame): 2767 lat pulldown 75 kg (11) ·
2768 row warmup (6) · 2769 row 70 kg (11) · 2770 row 70 kg (8) · 2771 cable lateral raise (15) ·
2772 lateral raise (13) · 2773 hammer incline warmup (6) · 2774, 2775 hammer incline (8, 8) ·
2776, 2777 RDL (8, 8).

| Clip | Hevy | ParkRep (before fixes) | Matches |
| --- | --- | --- | --- |
| 2767 | 11 | 3 | no |
| 2768 | 6 | 1 | no |
| 2769 | 11 | 11 | yes |
| 2770 | 8 | 9 | no |
| 2771 | 15 | 11 | no |
| 2772–2777 | 13, 6, 8, 8, 8, 8 | same | yes |

> Correction (same day): a first manual review called 2767 a 6-rep warmup and placed the 2771 point
> near the hand. A second frame-by-frame check (0.5 s sampling) found ~11–12 pulldowns in 2767 and
> the 2771 point on the sweatpants at mid-thigh. Both are corrected below.

Status legend: **DONE** = merged in the working tree with tests; **IMPLEMENTED** = code and
synthetic tests in place, not yet re-validated on the real clips; **OPEN** = not started.

## B0. Rotation lost when a clip has extra side data — DONE
`probe()` in `parkrep/video.py` overwrote `rotation` with 0 for every `side_data_list` entry
without a rotation. iPhone clips with Dolby Vision + "Ambient viewing environment" listed after the
Display Matrix were treated as 1280x720 instead of 720x1280; frames were decoded with the wrong width
and the HUD background came out as horizontal streaks.
- Fix: only assign when `"rotation" in side`. Regression test: `tests/test_video.py`.

## Re-run after B0, B1, B2, B5, B6, B9 (outputs in `sessions/2026-10-07/parkrep_v2`)

| Clip | Hevy | Reps before → after | Last-reps loss before → after | Warnings after |
| --- | --- | --- | --- | --- |
| 2767 lat | 11 | 3 → 3 | −9% → −1% | coverage 65%; tracking lost 5–11 s |
| 2768 row wu | 6 | 1 → 1 | n/a | **none** (see B10) |
| 2769 row | 11 | 11 → 11 | 55% → 51% | coverage 76% |
| 2770 row | 8 | 9 → **8** | −9% → −32% | none |
| 2771 lateral | 15 | 11 → 9 | 76% → 0% | reps move only 20 px (1.6% of frame) |
| 2772 lateral | 13 | 13 → 13 | 5% → 13% | none |
| 2773 hammer wu | 6 | 6 → 6 | 2% → 4% | none |
| 2774 hammer, RPE 7 | 8 | 8 → 8 | 7% → 27% | none |
| 2775 hammer, RPE 8 | 8 | 8 → 8 | 42% → 51% | none |
| 2776 RDL, RPE 8.5 | 8 | 8 → 8 | 24% → 25% | tracking lost 23–27 s |
| 2777 RDL, RPE 9 | 8 | 8 → 8 | 16% → 30% | none |

- Counts: 8 of 11 match (2770 fixed). Every clip with a wrong count now carries a warning except 2768.
- Timing is cleaner. Hammer 2774 before: 1.10, 1.17, 0.97, 0.90, 0.97, 1.00, 1.00, 1.30 s, with
  −15% to −22% in mid-set (artifacts). After: 0.68, 0.62, 0.58, 0.62, 0.63, 0.69, 0.73, 0.90 s, a flat
  middle and a steady slowdown at the end. The RDL order now follows RPE (8.5 → 25%, 9 → 30%; before
  it was reversed), but both last reps still include the walk to the rack (B7).
- 2770 still shows −32% last-reps loss with the right count; the point is on the chin (B3).
- Concentric times are now measured between 5% and 95% of the range, so absolute seconds read
  ~30% shorter than the full motion; ratios (velocity loss) are unaffected. Say so in the HUD/README.

## B11. Adjustment moves counted as reps ("main run") — IMPLEMENTED
User insight: in many clips the problem is that adjustment moves (setup, sitting back, standing up,
releasing the handle, an isolated move) are counted, which also skews candidate choice and the
velocity reference.
- Rule (`parkrep/reps.py`): a move belongs to the set when its range is 0.6–2.0× the median, its
  start level **or** top level matches the set (a partial keeps the bottom, a rep pulled from a
  stretch keeps the top, standing up keeps neither), and it comes back down (the last move is
  exempt: after the final rep people rack or release). Consistent moves form chains when they start
  ≤ 3.5 median periods apart (rest-pause allowed); every chain of 3+ moves counts, shorter isolated
  chains are adjustments. **No duration rule**: slow reps are the signal.
- Candidate choice (`track_quality` in `parkrep/cli.py`): reps × median range × coverage inside the
  set × regularity (1 − CV of range − CV of period) × share of moves that are reps. `discover.py`
  penalizes candidates within 5% of the frame border.
- Output: `adjustments` in the JSON, phase **ADJUST** in the HUD, a warning, and
  "no regular set found" when fewer than 3 consistent reps (covers the old B10, 2768).
- Evaluation (`scripts/eval_clips.py`, cached 8-candidate tracks, labels from Hevy):

  | Session | Old code | New code | Wrong count without a warning |
  | --- | --- | --- | --- |
  | 07/out (11 clips) | 8/11 | 8/11 | old 0 → new 0 |
  | 05/out (14 clips) | 11/14 | 12/14 | old 1 (2738) → new 0 |

  Slow tails preserved: 2775 reps 7–8, 2769 rep 11 (2.0 s), RDL last reps, 2742, 2744.
- Rejected on evidence: a "setup" rule for a slow first rep. On 05/out it dropped real first reps
  in the rear-delt rows (2726, 2730) and never caught a real setup on either session. Picking only
  the longest chain split a leg-extension set with a rest-pause (2739: 10 → 6).
- Caveat: 05/out was used to find those two problems, so it is no longer a held-out test. The
  next recorded session is the clean check.
- Fresh GPU run with the border penalty (`sessions/2026-10-07/parkrep_v3`): 2767 picked a new
  point on the cable/handle (100% coverage) and counts 11 = Hevy. **Partly a coincidence**: reps 1–10
  are real (every ~1.9 s up to 20.3 s), but rep 11 at 26.7 s (concentric 3.76 s) is the person
  standing up, let in by the last-move exemption; one or two real reps around 21–24 s are missing.
  Open: tell a final stand-up from a final grind (both are slow and do not come back down).
- Every adjustment raises the LOW CONFIDENCE badge (2770 shows it for one partial rep at the
  start, which was handled correctly). Consider showing the badge only for coverage/amplitude/
  no-set warnings.

## B10. No warning when fewer than 3 reps are found (2768) — DONE (via B11)

## B9. Concentric time inflated by pauses at the top or bottom — IMPLEMENTED
Found while reviewing B5. On a synthetic set of 8 identical reps (every concentric 0.6 s) the code
reports rep 6 = 0.73 s (+19% loss) and rep 8 = 1.17 s (+48% loss). The turn found by the hysteresis
lands anywhere on a flat, noisy stretch, often its far end, so a lockout hold or the rest after the
last rep is added to the concentric phase. This directly inflates "velocity loss, last reps", the
headline number.
- Fix: keep the turns for segmentation, but time the concentric between the 5% band crossings at
  the bottom and top, interpolated between frames; a pause at the top is labelled HOLD. Tests check
  ratios (identical reps within ±5%, a 0.8 s top hold changes nothing, slowing set within 5 points
  of the analytic loss).

## P1 – wrong result that looks confident

### B1. Point on the wrong body part, reps accepted with almost no amplitude (2771: 11 of 15) — IMPLEMENTED (warning)
The query point (frame 571, x=403, y=779) is on the grey sweatpants at mid-thigh, not on the arm or
handle. Over the whole 38 s clip it stays inside a ~70x45 px box (x 380–451, y 744–788) while the arm
sweeps to shoulder height ~15 times. Reps 1–11 have a 13–24 px range; reps 8 and 9 "last" 4.13 s and
6.37 s (pauses, not reps). Velocity loss reads 76–91% at 100% coverage, so it looks trustworthy and is not.
- Implemented: warning when the median rep range is below 4% of the frame's long side; ROM filter
  relative to the median of all moves instead of the first three.
- Still open: the root cause is candidate selection (see B3).
- Evidence: `2771_query_point.jpg`, `2771_lateral.jpg`, `2771_summary.jpg`.

### B2. Silent undercount: "SET DONE" while the person keeps training (2767: 3 of ~11) — IMPLEMENTED (warning)
Reps are found only in the first 5.4 s; the HUD says SET DONE from 6 s to the end while the person
keeps pulling, with bottoms around 1.5, 4, 5.5, 7, 9.5, 11, 13, 15, 17, 19, 21.5, 23 s. The point
(query at frame 528, 17.6 s) is lost for long stretches: visibility by 3 s window 89, 69, 47, 54, 92,
52, 52, 74, 61, 0 %; overall 65%.
- Implemented: warning below 80% coverage, and a warning naming the time spans where tracking was lost.
- Still open: counting the missed reps needs a better point (B3).
- Evidence: `2767_start.jpg`, `2767_late.jpg`, `2767_reps_8-15s.jpg`.

### B3. Candidate point selection picks static or off-limb points (2768, 2771, 2767) — OPEN
- 2768 (row warmup, 1 of 6): query at (31, 549) on the knee at the left border; the person is mostly
  out of frame and the single "rep" is them sitting back at 12–14 s.
- 2771: point on the thigh (B1).
- 2767: point lost in about half the frames (B2).
- Idea: penalize candidates in a border band and with a small range; prefer tracks whose range is a
  real share of the frame; save a debug frame with all candidates so the user can pick `--point`.
- Evidence: `2768_row.jpg`, `2771_query_point.jpg`.

## P2 – reference and direction

### B4. Concentric direction assumes "up on screen" — OPEN
Rows are horizontal; lateral raises filmed from the front barely move vertically. The README already
notes the reverse case (leg extension). Add `--concentric up|down|left|right` or detect the axis and
sign from phase speeds. 2768 shows the case (track range 293 px in x, 27 px in y).

### B5. Reference includes setup / partial reps (2767, 2769, 2770) — DONE (reference = reps 2–4; setup handling replaced by B11)
- 2767: rep 1 starts at t = 0 (clip begins mid-movement) and lasts 1.73 s; reps 2–3 last 0.87 s and 0.47 s.
- 2769: rep 1 has 389 px and 1.43 s, others ~200 px and 0.5–0.9 s.
- 2770: rep 1 has 72 px and 0.53 s (partial, right after a tracking gap) vs 106–150 px; the
  distorted reference gives −27% and −40% losses, and the count is 9 vs 8 in Hevy.
- Implemented: with 4+ reps the reference is reps 2–4; rep 1 is dropped (with a warning) when its
  range or duration is far from reps 2–4 and it starts at the clip start or right after a gap.
- Evidence: `2767_start.jpg`, `2769_row_side_end.jpg`, `2770_row_front.jpg`.

### B6. Reps lost or invented around tracking gaps (2770) — IMPLEMENTED (partial)
Between 14.2 s and 17.1 s the person keeps pulling but no rep is registered; coverage for 18–20 s is 5%.
- Implemented: a move that starts inside a gap or is >30% bridged is not counted (warning).
- Still open: recovering the missed rep needs a better track.
- Evidence: `2770_row_front.jpg` (frames 15–17 s).

## P3 – interpretation

### B7. Last RDL "rep" includes walking the bar to the rack (2776, 2777) — OPEN
Checked every 0.25 s:
- 2776: last rep 21.2–23.0 s. Bar to the floor at 21.25–21.5 s, up by 23.0 s, then walked to the rack
  and set down at 23.25–24.75 s. Its concentric is 1.87 s vs ~1 s (+44%).
- 2777: last rep 23.6–25.7 s. A real but slow pull from the floor (23.5–25.0 s), then walking to the
  rack (25.25–25.5 s); the detected top lands in the walk. 2.17 s (+43%).
- B9 explains part of this (the top lands at the end of a flat stretch). Also ignore motion after
  the main run of reps ("Trim idle time" in the README).
- 2769 (row): rep 11 lasts 2.3 s (+73%) during a tracking gap (21–24 s); low confidence.
- Evidence: `2776_last_rep_rack.jpg`, `2777_last_rep.jpg`, `2776_rdl_end.jpg`.

### B8. Do not read velocity loss vs RPE from this session yet
Hammer incline: RPE 7 → 7% and RPE 8 → 42%; RDL: RPE 8.5 → 24% and RPE 9 → 16%. Until B9 and B7
are fixed and the clips re-run, say which clips were excluded (2767, 2768, 2770, 2771) and do not
publish a velocity-loss-vs-RPE claim.

## Ideas to revisit (after the DEV submission)

### B12. Track a rigid group of points instead of one point — IDEA
Inspired by the local-rigidity prior in Dynamic 3D Gaussians (Luiten et al., 2023,
dynamic3dgaussians.github.io): neighbouring points on the same body part move together. The paper
itself does not fit ParkRep (it needs a synchronized multi-camera dome and per-scene optimization).
Cheap version for ParkRep:
- Seed a small cluster (e.g. 8–16 points) around the chosen candidate on the limb/bar; LocoTrack
  already tracks queries in batch.
- Use the median motion of the cluster; drop points that disagree with the group or leave the frame.
- Expected effect on B3/B2: one lost or drifting point no longer kills the set (2767, 2738, 2743
  lost the point; 2771 sat on the thigh). Measure with `scripts/eval_clips.py`.

### B13. Evaluate a stronger tracker (CoTracker3) — IDEA
CoTracker3 tracks many points jointly and handles occlusion better than LocoTrack-S, which may help
when the limb leaves the frame. **Check the license first**: CoTracker is, as far as remembered,
CC BY-NC (non-commercial), which conflicts with ParkRep's Apache-2.0; it may only be usable as an
optional, separately installed backend. Also check speed/memory on the M3 16 GB.

## Suggested order
1. B3 candidate selection: all remaining wrong counts (2767, 2768, 2771, 2738, 2743) come from the
   point leaving the frame or sitting on the wrong body part. Needs a GPU re-run to test the border
   penalty (the cache was made before it).
2. B4 direction.
3. B7 racking after the last rep.
