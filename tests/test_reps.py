import numpy as np

from parkrep.reps import analyze, find_turns


def synthetic(reps=8, fps=30.0, slow=0.0, gap=None, seed=0, amp=200.0, first_dur=1.0, top_hold=0.0, noise=0.5,
              setup_dur=None, sit_back=False, iso_rest=None):
    """Vertical bar path: rest, then `reps` cycles that get slower by `slow` per rep.

    `amp` is the path length in px, `first_dur` stretches the concentric phase of rep 1,
    `top_hold` adds a pause (seconds) at the top after each concentric.
    `setup_dur` adds a slow setup concentric before rep 1, `iso_rest` a long rest then one
    isolated move after the set, `sit_back` a lowering then a one-way rise that stays up.
    """
    rng = np.random.default_rng(seed)
    t, y = [], []
    now = 0.0
    def add(dur, y0, y1):
        nonlocal now
        k = max(2, int(dur * fps))
        for j in range(k):
            t.append(now + j / fps)
            y.append(y0 + (y1 - y0) * (1 - np.cos(np.pi * j / k)) / 2)
        now += k / fps
    add(1.0, 600, 600)
    if setup_dur:
        add(0.8, 600, 600 - amp)
        add(setup_dur, 600 - amp, 600)
    for r in range(reps):
        add(0.8, 600, 600 - amp)              # eccentric: bar goes down on screen (+y)... inverted below
        add(0.6 * (1 + slow * r) * (first_dur if r == 0 else 1), 600 - amp, 600)
        if top_hold:
            add(top_hold, 600, 600)
    if sit_back:
        add(0.8, 600, 600 - amp)
        add(1.0, 600 - amp, 600 - amp)
        add(0.8, 600 - amp, 600 + 1.5 * amp)              # stands up: rises 2.5 ROM and stays there
        add(2.5, 600, 600)
    elif iso_rest:
        add(iso_rest, 600, 600)
        add(0.8, 600, 600 - amp)
        add(0.6, 600 - amp, 600)
    add(1.0, 600, 600)
    t = np.array(t)
    y = 1000 - np.array(y)                     # bottom at y=600 -> screen y=400 is "down"
    xy = np.c_[np.full_like(y, 300.0) + rng.normal(0, noise, len(y)), y + rng.normal(0, noise, len(y))]
    ok = np.ones(len(t), bool)
    if gap:
        ok[gap[0]:gap[1]] = False
        xy[~ok] = np.nan
    return t, xy, ok


def test_turns_alternate():
    s = np.sin(np.linspace(0, 6 * np.pi, 300))
    kinds = [k for _, k in find_turns(s, 0.5)]
    assert all(a != b for a, b in zip(kinds, kinds[1:]))


def test_counts_reps():
    t, xy, ok = synthetic(reps=8)
    a = analyze(t, xy, ok, 30.0)
    assert len(a.reps) == 8


def test_velocity_loss_grows_when_slowing():
    slow = 0.08
    t, xy, ok = synthetic(reps=8, slow=slow)
    a = analyze(t, xy, ok, 30.0)
    losses = [r.velocity_loss_pct for r in a.reps]
    assert len(a.reps) == 8
    # truth: rep r (0-based) has concentric 0.6*(1+slow*r), so speed ∝ 1/(1+slow*r).
    # reference = median speed of reps 2..4 (r=1..3), the middle one being r=2 -> 1/(1+2*slow).
    # last rep r=7: loss = 100*(1 - (1+2*slow)/(1+7*slow)) = 25.6% for slow=0.08
    truth = 100 * (1 - (1 + 2 * slow) / (1 + 7 * slow))
    assert abs(losses[-1] - truth) < 5
    assert abs(losses[2]) < 5                           # rep 3 is the reference median: about zero


def test_constant_speed_has_no_fake_loss():
    t, xy, ok = synthetic(reps=8)
    a = analyze(t, xy, ok, 30.0)
    assert len(a.reps) == 8
    med = np.median([r.concentric_s for r in a.reps])
    assert all(abs(r.concentric_s / med - 1) < 0.05 for r in a.reps)
    assert all(abs(r.velocity_loss_pct) < 5 for r in a.reps)


def test_top_hold_does_not_inflate_concentric():
    # noise-free pair: with noise, the extra hold frames change the random draws of every later frame
    base = analyze(*synthetic(reps=8, noise=0.0), 30.0)
    held = analyze(*synthetic(reps=8, top_hold=0.8, noise=0.0), 30.0)
    assert len(held.reps) == 8
    for r, b in zip(held.reps, base.reps):              # same reps with and without the pause
        assert abs(r.concentric_s / b.concentric_s - 1) < 0.05
    a = analyze(*synthetic(reps=8, top_hold=0.8), 30.0)  # noisy run for losses and labels
    assert len(a.reps) == 8
    assert all(abs(r.velocity_loss_pct) < 5 for r in a.reps)
    # the pause is labelled HOLD: about 0.8 s x 30 fps per rep, well above the bottom-edge HOLD frames
    assert a.phase.count("HOLD") - base.phase.count("HOLD") >= 8 * 20


def test_short_gap_is_bridged():
    t, xy, ok = synthetic(reps=6, gap=(120, 126))
    a = analyze(t, xy, ok, 30.0)
    assert len(a.reps) == 6
    assert a.inferred[120:126].all()


def test_low_coverage_window_warns():
    t, xy, ok = synthetic(reps=8, gap=(150, 270))       # 4 s without the point
    a = analyze(t, xy, ok, 30.0)
    assert not ok[150:270].any()
    assert any("point visible in only" in w for w in a.warnings)
    assert any(w.startswith("tracking lost around") for w in a.warnings)
    assert "tracked_share" in a.summary() and a.summary()["warnings"] == a.warnings


def test_tiny_amplitude_warns_with_frame_size():
    t, xy, ok = synthetic(reps=8, amp=20)
    a = analyze(t, xy, ok, 30.0, frame_size=(720, 1280))
    assert any("count and speeds are unreliable" in w for w in a.warnings)
    b = analyze(t, xy, ok, 30.0)                        # without frame size: no absolute check
    assert not any("count and speeds are unreliable" in w for w in b.warnings)


def test_slow_first_rep_counts_but_is_not_reference():
    # real sets often start with a slower first rep; it is a rep, but the reference is reps 2..4
    t, xy, ok = synthetic(reps=6, first_dur=2.5)
    a = analyze(t, xy, ok, 30.0)
    assert len(a.reps) == 6 and not a.adjustments
    assert a.summary()["reference_reps"] == [2, 3, 4]
    assert a.reps[0].velocity_loss_pct > 30
    assert all(abs(r.velocity_loss_pct) < 10 for r in a.reps[1:])


def test_sitting_back_after_set_is_not_counted():
    t, xy, ok = synthetic(reps=6, sit_back=True)        # lowered, then one rise by 2.5 ROM that stays up
    a = analyze(t, xy, ok, 30.0)
    assert len(a.reps) == 6
    assert [x["reason"] for x in a.adjustments] == ["different range"]
    assert any("adjustment moves not counted (different range)" in w for w in a.warnings)


def test_last_rep_followed_by_racking_is_counted():
    # a final rep whose eccentric does not come back (bar racked / handle released) is still a rep
    t, xy, ok = synthetic(reps=6)
    p = analyze(t, xy, ok, 30.0).reps[-1].peak
    cut = slice(0, p + 6)                                # the clip ends early in the last eccentric
    a = analyze(t[cut], xy[cut], ok[cut], 30.0)
    assert len(a.reps) == 6 and not a.adjustments


def test_isolated_move_after_long_rest_is_out_of_rhythm():
    t, xy, ok = synthetic(reps=6, iso_rest=4.2)         # about 3 rep periods (1.4 s each) of rest, then one move
    a = analyze(t, xy, ok, 30.0)
    assert len(a.reps) == 6
    assert [x["reason"] for x in a.adjustments] == ["out of rhythm"]


def test_slowing_set_keeps_every_rep_and_no_adjustments():
    a = analyze(*synthetic(reps=8, slow=0.08), 30.0)
    assert len(a.reps) == 8
    assert a.adjustments == [] and a.ignored == 0


def test_single_move_clip_warns_no_regular_set():
    a = analyze(*synthetic(reps=1), 30.0)
    assert any(w.startswith("no regular set found") for w in a.warnings)


def test_move_starting_in_gap_is_not_counted():
    clean = analyze(*synthetic(reps=6), 30.0)
    s0 = clean.reps[2].start
    gap = (s0 - 2, s0 + 7)                              # 0.3 s gap across the 3rd bottom: bridged, inferred
    t, xy, ok = synthetic(reps=6, gap=gap)
    a = analyze(t, xy, ok, 30.0)
    assert any("overlapped tracking gaps" in w for w in a.warnings)
    assert len(a.reps) == 5
    assert not any(gap[0] <= r.start < gap[1] for r in a.reps)
