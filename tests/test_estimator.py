import numpy as np
import pytest
from fall_detection.depth.estimator import YOLO26DepthEstimator


class _Tensor:
    def __init__(self, a): self.a = a
    def cpu(self): return self
    def numpy(self): return self.a


def _estimator(depth):
    class DepthMap: data = _Tensor(depth)
    class Result: depth = DepthMap()
    class Model:
        def predict(self, **kw): return [Result()]
    e = YOLO26DepthEstimator('x'); e.model = Model()
    return e


def test_reads_documented_result_depth_data():
    d = np.random.rand(48, 64).astype(np.float32) + 1
    out = _estimator(d).predict_depth(np.zeros((48, 64, 3), np.uint8))
    assert out.dtype == np.float32 and out.shape == (48, 64)


def test_rejects_shape_mismatch():
    with pytest.raises(RuntimeError):
        _estimator(np.ones((10, 10), np.float32)).predict_depth(np.zeros((48, 64, 3), np.uint8))


def test_auto_device_becomes_none():
    assert YOLO26DepthEstimator('x', 'auto').device is None
