import json
import sys
import cv2
import numpy as np
import pandas as pd
import fall_detection.cli as cli


class FakeEstimator:
    def __init__(self, *a, **k): self.weights, self.imgsz = 'fake.pt', 768
    def predict_depth(self, frame):
        return np.full(frame.shape[:2], 3.0, np.float32)


def _video(path, n=60, fps=30):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (64, 48))
    for _ in range(n): w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()


def test_infer_depth_subsamples_and_writes_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'YOLO26DepthEstimator', FakeEstimator)
    vid, out = tmp_path / 'v.avi', tmp_path / 'depth'
    _video(vid)
    monkeypatch.setattr(sys, 'argv', ['x', '--config', 'configs/base.yaml', 'infer-depth',
                                      '--source', str(vid), '--output-dir', str(out), '--fp16'])
    cli.main()
    ts = pd.read_csv(out / 'timestamps.csv')
    assert 18 <= len(ts) <= 22                                   # 2 s at ~10 fps
    meta = json.loads((out / 'meta.json').read_text())
    assert meta['units'] == 'meters' and meta['saved_dtype'] == 'float16'
    assert np.load(out / 'frame_000000.npy').dtype == np.float16


def test_extract_features_end_to_end(tmp_path, monkeypatch):
    d = tmp_path / 'depth'; d.mkdir()
    n = 45 + 20
    pd.DataFrame({'timestamp_s': np.arange(n) / 10}).to_csv(d / 'timestamps.csv', index=False)
    for i in range(n):
        f = np.full((120, 160), 3.0, np.float32)
        if i >= 45: f[40:100, 60:90] = 2.0
        np.save(d / f'frame_{i:06d}.npy', f)
    out = tmp_path / 'f.csv'
    monkeypatch.setattr(sys, 'argv', ['x', '--config', 'configs/base.yaml', 'extract-features',
                                      '--depth-dir', str(d), '--output', str(out)])
    cli.main()
    df = pd.read_csv(out)
    assert len(df) == 20 and df['track_id'].nunique() == 1
    assert df['frame_index'].min() == 44 or df['frame_index'].min() == 45
