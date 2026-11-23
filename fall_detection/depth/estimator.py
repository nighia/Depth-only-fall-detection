import numpy as np

DEFAULT_IMGSZ = 768  # released YOLO26-Depth weights are trained at 768


def _to_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value)


class YOLO26DepthEstimator:
    """Adapter for Ultralytics YOLO26 Depth (needs ultralytics>=8.4.104).

    predict_depth returns float32 depth in meters with the same (H, W) as the
    input frame. Absolute scale depends on calibration (model.calibrate()).
    """

    def __init__(self, weights, device='auto', imgsz=DEFAULT_IMGSZ):
        self.weights = str(weights)
        self.device = None if device in (None, 'auto') else device
        self.imgsz = imgsz
        self.model = None

    def load(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('Install ultralytics>=8.4.104 (YOLO26 Depth)') from exc
        self.model = YOLO(self.weights)
        return self

    def predict_depth(self, bgr_frame):
        if self.model is None:
            self.load()
        result = self.model.predict(source=bgr_frame, imgsz=self.imgsz,
                                    device=self.device, verbose=False)[0]
        depth_map = getattr(result, 'depth', None)
        if depth_map is None:
            raise RuntimeError('Result has no .depth; is this a YOLO26 -depth checkpoint?')
        arr = _to_numpy(depth_map.data).squeeze().astype(np.float32)
        if arr.ndim != 2:
            raise RuntimeError(f'Expected a 2D depth map, got shape {arr.shape}')
        if arr.shape != bgr_frame.shape[:2]:
            raise RuntimeError(f'Depth shape {arr.shape} != frame shape {bgr_frame.shape[:2]}')
        if not np.isfinite(arr).any():
            raise RuntimeError('Depth map has no finite values')
        return arr


def save_depth(path, depth, dtype=np.float32):
    import os
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.save(path, np.asarray(depth, dtype=dtype))


def runtime_metadata():
    import platform
    meta = {'python': platform.python_version(), 'numpy': np.__version__}
    for name in ('ultralytics', 'torch', 'cv2'):
        try:
            mod = __import__(name)
            meta[name] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            meta[name] = None
    return meta
