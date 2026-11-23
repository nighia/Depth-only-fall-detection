import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from fall_detection.config import load_config
from fall_detection.video import read_video, video_fps
from fall_detection.depth.estimator import YOLO26DepthEstimator, save_depth, runtime_metadata
from fall_detection.depth.quality import temporal_flicker
from fall_detection.regions.background import DepthBackgroundExtractor
from fall_detection.tracking.iou_tracker import IoUTracker
from fall_detection.features.depth_features import (
    DepthFeatureExtractor, FEATURE_SCHEMA_VERSION)

DEFAULT_CONFIG = 'configs/base.yaml'


def infer_depth(a):
    cfg = load_config(a.config)
    target_fps = a.target_fps or cfg['temporal']['target_fps']
    estimator = YOLO26DepthEstimator(a.weights or cfg['depth']['weights'],
                                     a.device or cfg['depth']['device'],
                                     cfg['depth'].get('imgsz', 768))
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if a.fp16 else np.float32
    step, next_t, rows, shape = 1.0 / target_fps, 0.0, [], None
    for idx, ts, frame in read_video(a.source):
        if ts + 1e-6 < next_t:
            continue
        while next_t <= ts + 1e-6:
            next_t += step
        depth = estimator.predict_depth(frame)
        shape = depth.shape
        save_depth(out / f'frame_{len(rows):06d}.npy', depth, dtype)
        rows.append({'depth_index': len(rows), 'source_frame': idx, 'timestamp_s': ts})
        if len(rows) % 100 == 1:
            print(f'depth_index={len(rows)-1}, t={ts:.2f}s, shape={shape}')
    pd.DataFrame(rows).to_csv(out / 'timestamps.csv', index=False)
    meta = {'feature_schema_version': FEATURE_SCHEMA_VERSION, 'weights': str(estimator.weights),
            'imgsz': estimator.imgsz, 'units': 'meters', 'saved_dtype': np.dtype(dtype).name,
            'source': str(a.source), 'source_fps': video_fps(a.source),
            'target_fps': target_fps, 'depth_shape': list(shape) if shape else None,
            'versions': runtime_metadata()}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Wrote {len(rows)} depth maps + timestamps.csv + meta.json to {out}')


def extract_features(a):
    cfg = load_config(a.config)
    paths = sorted(Path(a.depth_dir).glob('frame_*.npy'))
    if not paths:
        raise FileNotFoundError(f'No depth .npy files in {a.depth_dir}')
    ts_path = Path(a.depth_dir) / 'timestamps.csv'
    if ts_path.is_file():
        timestamps = pd.read_csv(ts_path)['timestamp_s'].to_numpy()
        if len(timestamps) != len(paths):
            raise ValueError('timestamps.csv length != number of depth files')
    else:
        warnings.warn(f'No timestamps.csv; assuming constant {a.fps} fps')
        timestamps = np.arange(len(paths)) / a.fps
    r, t, f = cfg['regions'], cfg['tracking'], cfg.get('features', {})
    extractor = DepthBackgroundExtractor(
        r['background_warmup_frames'], r['depth_difference_threshold'], r['min_area_pixels'],
        r.get('max_unstable_fraction', 0.02), r.get('stable_std_m', 0.05),
        relative_threshold=r.get('depth_difference_relative', 0.06),
        stable_relative=r.get('stable_relative', 0.05))
    tracker = IoUTracker(t['iou_threshold'], t['max_missed_frames'], t.get('centroid_gate_factor', 0.5))
    fe = DepthFeatureExtractor(f.get('max_gap_s', 1.0))
    rows = []
    for idx, path in enumerate(paths):
        depth = np.load(path).astype(np.float32)
        ts = float(timestamps[idx])
        for tid, region in tracker.update(extractor.update(depth)):
            rows.append({'frame_index': idx, 'timestamp_s': ts, 'track_id': tid,
                         **fe.extract(tid, region, depth, ts)})
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output, index=False)
    print(f'Wrote {len(rows)} feature rows (schema {FEATURE_SCHEMA_VERSION}); '
          f'first {r["background_warmup_frames"]} frames are background warm-up. '
          f'Labels are not inferred; annotate separately.')


def depth_flicker(a):
    paths = sorted(Path(a.depth_dir).glob('frame_*.npy'))[:a.max_frames]
    if len(paths) < 3:
        raise FileNotFoundError(f'Need >=3 depth files in {a.depth_dir}')
    stack = np.stack([np.load(p).astype(np.float32) for p in paths])
    for k, v in temporal_flicker(stack).items():
        print(f'{k}: {v:.4f}')
    print('Run on an EMPTY static scene. Compare pixel_std_p95_m with the depth '
          'difference threshold before trusting it.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=DEFAULT_CONFIG)
    sub = p.add_subparsers(required=True)
    q = sub.add_parser('infer-depth')
    q.add_argument('--source', required=True); q.add_argument('--weights')
    q.add_argument('--device'); q.add_argument('--output-dir', required=True)
    q.add_argument('--target-fps', type=float)
    q.add_argument('--fp16', action='store_true', help='save float16 (half the disk)')
    q.set_defaults(func=infer_depth)
    q = sub.add_parser('extract-features')
    q.add_argument('--depth-dir', required=True); q.add_argument('--output', required=True)
    q.add_argument('--fps', type=float, default=10); q.set_defaults(func=extract_features)
    q = sub.add_parser('depth-flicker')
    q.add_argument('--depth-dir', required=True); q.add_argument('--max-frames', type=int, default=300)
    q.set_defaults(func=depth_flicker)
    a = p.parse_args(); a.func(a)


if __name__ == '__main__':
    main()
