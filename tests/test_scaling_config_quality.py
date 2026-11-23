import numpy as np
import pytest
from fall_detection.config import load_config
from fall_detection.data.scaling import FeatureScaler
from fall_detection.depth.quality import temporal_flicker


def test_scaler_fits_on_train_only_and_imputes_nan():
    tr = np.array([[[1., 10.], [3., 10.]]]); s = FeatureScaler().fit(tr)
    te = np.array([[[np.nan, 10.], [100., 10.]]])
    out = s.transform(te)
    assert np.isfinite(out).all() and out.dtype == np.float32
    assert s.median[0] == 2.0 and s.std[1] == 1.0          # constant column guarded


def test_scaler_roundtrip():
    s = FeatureScaler().fit(np.random.rand(5, 4, 3))
    s2 = FeatureScaler.from_state_dict(s.state_dict())
    x = np.random.rand(2, 4, 3)
    assert np.allclose(s.transform(x), s2.transform(x))


def test_base_config_loads():
    cfg = load_config('configs/base.yaml')
    assert cfg['project']['feature_schema_version'] == '0.2' and cfg['depth']['imgsz'] == 768
    with pytest.raises(FileNotFoundError): load_config('configs/nope.yaml')


def test_flicker_detects_global_scale_wobble():
    base = np.full((30, 20, 20), 3.0, np.float32)
    steady = temporal_flicker(base)
    wobble = base * np.exp(np.random.default_rng(0).normal(0, 0.05, 30))[:, None, None]
    assert steady['scale_drift_std'] < 1e-6 and steady['pixel_std_p95_m'] < 1e-6
    assert temporal_flicker(wobble)['scale_drift_std'] > 0.02
