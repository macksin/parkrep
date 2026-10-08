from types import SimpleNamespace

import numpy as np

from parkrep.cli import track_quality

FPS = 10
N = 300


def make_analysis(roms, starts_s, adjustments=0, observed=None):
    reps = []
    for rom, s in zip(roms, starts_s):
        start = round(s * FPS)
        reps.append(SimpleNamespace(start=start, end=start + 15, start_s=s, rom_px=rom))
    obs = np.ones(N, bool) if observed is None else observed
    return SimpleNamespace(reps=reps, observed=obs, adjustments=[{}] * adjustments)


REGULAR_ROM = [100.0] * 8
REGULAR_T = [2.0 * i for i in range(8)]
IRREGULAR_ROM = [40.0, 160.0, 60.0, 140.0, 50.0, 170.0, 45.0, 155.0]
IRREGULAR_T = [0.0, 1.0, 4.0, 5.5, 8.0, 8.5, 12.0, 15.0]


def test_regular_track_beats_irregular_track():
    regular = make_analysis(REGULAR_ROM, REGULAR_T)
    irregular = make_analysis(IRREGULAR_ROM, IRREGULAR_T)
    assert track_quality(regular) > track_quality(irregular) > 0


def test_single_rep_scores_zero():
    assert track_quality(make_analysis([100.0], [0.0])) == 0.0


def test_adjustments_lower_score():
    clean = make_analysis(REGULAR_ROM, REGULAR_T, adjustments=0)
    dirty = make_analysis(REGULAR_ROM, REGULAR_T, adjustments=2)
    assert track_quality(dirty) < track_quality(clean)


def test_unobserved_run_lowers_score():
    seen = make_analysis(REGULAR_ROM, REGULAR_T)
    gappy = np.ones(N, bool)
    gappy[20:60] = False
    assert track_quality(make_analysis(REGULAR_ROM, REGULAR_T, observed=gappy)) < track_quality(seen)
