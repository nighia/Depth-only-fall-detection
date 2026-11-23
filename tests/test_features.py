import numpy as np
from fall_detection.schemas import Region
from fall_detection.features.depth_features import DepthFeatureExtractor, FEATURE_NAMES


def test_schema_has_14_features_and_normalized_sizes():
    d = np.full((100, 200), 3.0, np.float32)
    f = DepthFeatureExtractor().extract(1, Region(20, 10, 40, 80, 3200), d, 0.0)
    assert len(FEATURE_NAMES) == 14 and list(f) == FEATURE_NAMES
    assert f['bbox_w_norm'] == 0.2 and f['bbox_h_norm'] == 0.8
    assert f['depth_median'] == 3.0 and f['valid_depth_fraction'] == 1.0
    assert f['vertical_velocity'] == 0.0            # first sample


def test_valid_fraction_and_nan_when_no_depth():
    d = np.zeros((100, 200), np.float32)
    f = DepthFeatureExtractor().extract(1, Region(0, 0, 50, 50, 2500), d, 0.0)
    assert f['valid_depth_fraction'] == 0.0 and np.isnan(f['depth_median'])


def test_rates_and_gap_reset():
    d = np.full((100, 200), 3.0, np.float32); fe = DepthFeatureExtractor(max_gap_s=1.0)
    fe.extract(1, Region(0, 0, 40, 40, 1600), d, 0.0)
    f = fe.extract(1, Region(0, 20, 40, 40, 1600), d, 0.1)
    assert abs(f['vertical_velocity'] - 2.0) < 1e-6        # 0.2 of frame height in 0.1 s
    f2 = fe.extract(1, Region(0, 60, 40, 40, 1600), d, 5.0)   # long gap
    assert f2['vertical_velocity'] == 0.0
