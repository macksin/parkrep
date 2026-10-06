import numpy as np

from parkrep.reps import analyze, find_turns


def synthetic(reps=8, fps=30.0, slow=0.0, gap=None, seed=0):
    """Vertical bar path: rest, then `reps` cycles that get slower by `slow` per rep."""
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
    for r in range(reps):
        add(0.8, 600, 600 - 200)              # eccentric: bar goes down on screen (+y)... inverted below
        add(0.6 * (1 + slow * r), 400, 600)
    add(1.0, 600, 600)
    t = np.array(t)
    y = 1000 - np.array(y)                     # bottom at y=600 -> screen y=400 is "down"
    xy = np.c_[np.full_like(y, 300.0) + rng.normal(0, 0.5, len(y)), y + rng.normal(0, 0.5, len(y))]
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
    t, xy, ok = synthetic(reps=8, slow=0.08)
    a = analyze(t, xy, ok, 30.0)
    losses = [r.velocity_loss_pct for r in a.reps]
    assert len(a.reps) == 8
    assert losses[-1] > 25 and abs(losses[0]) < 10


def test_short_gap_is_bridged():
    t, xy, ok = synthetic(reps=6, gap=(120, 126))
    a = analyze(t, xy, ok, 30.0)
    assert len(a.reps) == 6
    assert a.inferred[120:126].all()
