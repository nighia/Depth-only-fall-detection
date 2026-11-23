import numpy as np


def temporal_flicker(stack, min_valid_fraction=0.8):
    """Stability of depth on a STATIC scene (no people), stack shape [T, H, W] in meters.

    Returns:
      pixel_std_median_m / pixel_std_p95_m: per-pixel temporal std (meters).
      rel_std_median / rel_std_p95: the same as a fraction of range -- noise grows
        with distance, so choose the relative foreground threshold above rel_std_p95.
      scale_drift_std: std over frames of the median log-ratio to the reference
        frame-median; a global per-frame scale wobble shows up here even when
        per-pixel std looks small.
      valid_fraction: share of pixels valid in >= min_valid_fraction of frames.
    """
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim != 3 or stack.shape[0] < 3:
        raise ValueError('stack must be [T>=3, H, W]')
    valid = np.isfinite(stack) & (stack > 0)
    keep = valid.mean(axis=0) >= min_valid_fraction
    if not keep.any():
        raise ValueError('No pixel is valid in enough frames')
    vals = np.where(valid, stack, np.nan)
    ref = np.nanmedian(vals, axis=0)
    std = np.nanstd(vals, axis=0)[keep]
    rel_std = std / ref[keep]
    ratios = np.log(vals[:, keep] / ref[keep])
    frame_scale = np.nanmedian(ratios, axis=1)
    return {
        'pixel_std_median_m': float(np.median(std)),
        'pixel_std_p95_m': float(np.percentile(std, 95)),
        'scale_drift_std': float(np.std(frame_scale)),
        'valid_fraction': float(keep.mean()),
    }
