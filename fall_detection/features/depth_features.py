import numpy as np

FEATURE_SCHEMA_VERSION = '0.2'
FEATURE_NAMES = [
    'bbox_w_norm', 'bbox_h_norm', 'bbox_aspect', 'foreground_area_norm',
    'centroid_x_norm', 'centroid_y_norm', 'depth_median', 'depth_iqr',
    'depth_variance', 'vertical_velocity', 'area_rate', 'aspect_rate',
    'depth_rate', 'valid_depth_fraction',
]


class DepthFeatureExtractor:
    """Per-track geometric/depth proxies (not body pose).

    Sizes are normalized by frame size so features transfer across resolutions.
    Rate features are zero on a track's first sample and after a gap longer than
    max_gap_s. Depth stats are NaN when the crop has no valid depth (kept for
    audit; the scaler imputes them from training data).
    """

    def __init__(self, max_gap_s=1.0):
        self.previous = {}
        self.max_gap_s = max_gap_s

    def extract(self, track_id, region, depth, timestamp_s):
        h, w = depth.shape
        x1, y1 = max(0, region.x), max(0, region.y)
        x2, y2 = min(w, region.x + region.w), min(h, region.y + region.h)
        crop = depth[y1:y2, x1:x2]
        valid = np.isfinite(crop) & (crop > 0)
        vals = crop[valid]
        valid_fraction = float(vals.size / crop.size) if crop.size else 0.0
        if vals.size:
            med = float(np.median(vals))
            q1, q3 = np.percentile(vals, [25, 75])
            iqr, var = float(q3 - q1), float(np.var(vals))
        else:
            med = iqr = var = float('nan')
        cx = (region.x + region.w / 2) / max(w, 1)
        cy = (region.y + region.h / 2) / max(h, 1)
        aspect = region.w / max(region.h, 1)
        area = float(region.area) / max(w * h, 1)
        prev = self.previous.get(track_id)
        dt = timestamp_s - prev['t'] if prev else 0.0
        if prev and 1e-6 < dt <= self.max_gap_s:
            vy = (cy - prev['cy']) / dt
            ar = (area - prev['area']) / dt
            asp = (aspect - prev['aspect']) / dt
            ok = np.isfinite(med) and np.isfinite(prev['med'])
            dr = (med - prev['med']) / dt if ok else 0.0
        else:
            vy = ar = asp = dr = 0.0
        self.previous[track_id] = {'t': timestamp_s, 'cy': cy, 'area': area,
                                   'aspect': aspect, 'med': med}
        return dict(zip(FEATURE_NAMES, [
            region.w / max(w, 1), region.h / max(h, 1), float(aspect), area, float(cx), float(cy),
            med, iqr, var, float(vy), float(ar), float(asp), float(dr), valid_fraction,
        ]))
