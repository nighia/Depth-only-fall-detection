import warnings
import numpy as np
import pytest
from fall_detection.regions.background import DepthBackgroundExtractor

EMPTY = np.full((120, 160), 3.0, np.float32)


def test_warmup_emits_nothing_then_detects_object():
    ex = DepthBackgroundExtractor(warmup_frames=10, min_area=100)
    assert all(ex.update(EMPTY) == [] for _ in range(9)) and not ex.ready
    assert ex.update(EMPTY) == [] and ex.ready          # 10th frame completes warm-up
    d = EMPTY.copy(); d[40:100, 60:90] = 2.0           # 1 m closer
    regions = ex.update(d)
    assert ex.ready and len(regions) == 1 and regions[0].h >= 55


def test_unstable_warmup_warns():
    ex = DepthBackgroundExtractor(warmup_frames=10, max_unstable_fraction=0.02)
    rng = np.random.default_rng(0)
    with pytest.warns(RuntimeWarning, match='non-static'):
        for i in range(10):
            d = EMPTY.copy(); d[20:100, 20:140] = 3.0 + (i % 2)   # large region flickers 1 m
            ex.update(d)
    assert ex.unstable_fraction > 0.02


def test_static_warmup_does_not_warn():
    ex = DepthBackgroundExtractor(warmup_frames=10)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for _ in range(10): ex.update(EMPTY)
    assert ex.unstable_fraction == 0.0


def test_relative_threshold_ignores_far_noise_but_keeps_far_objects():
    far = np.full((120, 160), 12.0, np.float32)
    ex = DepthBackgroundExtractor(warmup_frames=10, min_area=100)
    for _ in range(10): ex.update(far)
    noisy = far + 0.2                       # 1.7% of range: sensor noise level, not an object
    assert ex.update(noisy) == []
    obj = far.copy(); obj[40:100, 60:90] = 9.0          # 3 m closer at 12 m range
    assert len(ex.update(obj)) == 1
