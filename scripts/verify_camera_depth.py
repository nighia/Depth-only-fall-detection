"""Phase 0-1 check: camera/video/image -> YOLO26 Depth -> numeric depth map.

Run from the repo root, e.g.
  python scripts/verify_camera_depth.py --camera 0 --weights weights/yolo26n-depth.pt
  python scripts/verify_camera_depth.py --video data/raw/clip.mp4
  python scripts/verify_camera_depth.py --image sample.jpg
Exit code 0 = all hard checks passed (warnings do not fail the run).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fall_detection.depth.estimator import YOLO26DepthEstimator, runtime_metadata  # noqa: E402

DOC_RANGE_M = (0.02, 150.0)   # documented output range of the log-depth head


def open_source(a):
    if a.image:
        img = cv2.imread(a.image)
        if img is None:
            raise RuntimeError(f'Could not read image: {a.image}')
        return ('image', img, None)
    if a.video:
        cap = cv2.VideoCapture(a.video)
        kind = 'video'
    else:
        backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
        cap = cv2.VideoCapture(a.camera, backend)
        kind = 'camera'
        if a.width: cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        if a.height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open {kind}: {a.video or a.camera}. '
                           'Close other apps using the camera / check the index (try --camera 1).')
    return (kind, None, cap)


def next_frame(kind, img, cap):
    if kind == 'image':
        return True, img.copy()
    return cap.read()


def preview(frame, depth, path):
    """Side-by-side PREVIEW ONLY (colourised inverse depth). Computation uses raw metres."""
    valid = np.isfinite(depth) & (depth > 0)
    inv = np.zeros_like(depth)
    inv[valid] = 1.0 / depth[valid]
    lo, hi = np.percentile(inv[valid], [2, 98]) if valid.any() else (0, 1)
    norm = np.clip((inv - lo) / max(hi - lo, 1e-9), 0, 1)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    color[~valid] = 0
    cv2.imwrite(str(path), np.hstack([frame, color]))


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument('--camera', type=int, default=0)
    g.add_argument('--video'); g.add_argument('--image')
    p.add_argument('--weights', default='weights/yolo26n-depth.pt')
    p.add_argument('--device', default='auto'); p.add_argument('--imgsz', type=int, default=768)
    p.add_argument('--frames', type=int, default=60)
    p.add_argument('--warmup', type=int, default=5, help='frames discarded (camera auto-exposure)')
    p.add_argument('--width', type=int); p.add_argument('--height', type=int)
    p.add_argument('--out', default='artifacts/verify')
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    kind, img, cap = open_source(a)
    est = YOLO26DepthEstimator(a.weights, a.device, a.imgsz)
    t0 = time.perf_counter(); est.load(); load_s = time.perf_counter() - t0
    reported_fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None

    for _ in range(a.warmup):
        next_frame(kind, img, cap)

    cap_ms, inf_ms, stamps, med, valid_frac, lo, hi = [], [], [], [], [], [], []
    problems, first = [], None
    for i in range(a.frames):
        t = time.perf_counter(); ok, frame = next_frame(kind, img, cap); t1 = time.perf_counter()
        if not ok or frame is None:
            problems.append(f'frame {i}: read failed (end of video / camera dropped)'); break
        try:
            depth = est.predict_depth(frame)
        except Exception as exc:                                  # adapter checks shape/finite
            problems.append(f'frame {i}: {type(exc).__name__}: {exc}'); break
        t2 = time.perf_counter()
        cap_ms.append((t1 - t) * 1e3); inf_ms.append((t2 - t1) * 1e3); stamps.append(t1)
        v = np.isfinite(depth) & (depth > 0)
        valid_frac.append(float(v.mean()))
        if v.any():
            med.append(float(np.median(depth[v]))); lo.append(float(np.percentile(depth[v], 1)))
            hi.append(float(np.percentile(depth[v], 99)))
        if first is None:
            first = (frame, depth)
    if cap is not None: cap.release()

    if first is None:
        print('FAIL: no frame was processed'); [print(' -', q) for q in problems]; sys.exit(1)
    frame, depth = first
    cv2.imwrite(str(out / 'frame_first.jpg'), frame)
    np.save(out / 'depth_first.npy', depth)
    preview(frame, depth, out / 'preview_first.png')

    n = len(inf_ms)
    intervals = np.diff(stamps) * 1e3 if n > 1 else np.array([0.0])
    stats = {
        'source': kind, 'frames_processed': n, 'frame_shape': list(frame.shape),
        'camera_reported_fps': reported_fps, 'model_load_s': round(load_s, 2),
        'depth_dtype': str(depth.dtype), 'depth_shape': list(depth.shape),
        'valid_fraction_mean': round(float(np.mean(valid_frac)), 4),
        'depth_p1_m': round(float(np.mean(lo)), 3), 'depth_median_m': round(float(np.mean(med)), 3),
        'depth_p99_m': round(float(np.mean(hi)), 3),
        'per_frame_median_std_m': round(float(np.std(med)), 4),
        'capture_ms_p50': round(float(np.percentile(cap_ms, 50)), 1),
        'inference_ms_p50': round(float(np.percentile(inf_ms, 50)), 1),
        'inference_ms_p95': round(float(np.percentile(inf_ms, 95)), 1),
        'end_to_end_fps': round(1000.0 / (np.mean(cap_ms) + np.mean(inf_ms)), 2),
        'frame_interval_ms_p95': round(float(np.percentile(intervals, 95)), 1),
        'versions': runtime_metadata(), 'weights': a.weights, 'imgsz': a.imgsz,
    }
    (out / 'verify_report.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')

    hard = [('float32 dtype', depth.dtype == np.float32),
            ('depth shape == frame shape', depth.shape == frame.shape[:2]),
            ('finite + positive values (valid fraction > 0.5)', stats['valid_fraction_mean'] > 0.5),
            ('no read/inference errors', not problems)]
    warn = []
    if not (DOC_RANGE_M[0] <= stats['depth_p1_m'] and stats['depth_p99_m'] <= DOC_RANGE_M[1]):
        warn.append(f'depth p1/p99 outside documented {DOC_RANGE_M} m range')
    if kind != 'image' and stats['per_frame_median_std_m'] > 0.3:
        warn.append('per-frame median depth varies >0.3 m (fine if people move; for a static '
                    'scene it indicates scale instability: run the depth-flicker command)')
    if stats['end_to_end_fps'] < 5:
        warn.append('end-to-end < 5 FPS: use a GPU or lower --imgsz for realtime')

    print(json.dumps({k: v for k, v in stats.items() if k != 'versions'}, indent=2))
    print('\nChecks:')
    for name, ok in hard: print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    for w in warn: print(f'  [WARN] {w}')
    for q in problems: print(f'  [ERR] {q}')
    print(f'\nSaved: {out}/preview_first.png (visual check), depth_first.npy, verify_report.json')
    print('Look at preview_first.png: near objects should be bright, walls/background darker.')
    sys.exit(0 if all(ok for _, ok in hard) else 1)


if __name__ == '__main__':
    main()
