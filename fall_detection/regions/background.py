import warnings

import cv2
import numpy as np
from fall_detection.schemas import Region


class DepthBackgroundExtractor:
    """Fixed-camera baseline; this is not a semantic person detector.

    Warm-up policy: the first `warmup_frames` frames build a median background
    and emit no regions (`ready` is False). The scene MUST be empty and static
    during warm-up; `unstable_fraction` reports how many pixels varied more than
    `stable_std` meters, and a warning is raised above `max_unstable_fraction`.
    """

    def __init__(self, warmup_frames=45, threshold=0.05, min_area=500,
                 max_unstable_fraction=0.02, stable_std=0.05,
                 relative_threshold=0.06, stable_relative=0.05):
        # Monocular depth noise grows with distance (~1-3% of range), so a foreground
        # pixel needs |depth - bg| > max(threshold [m, floor], relative_threshold * bg).
        self.warmup_frames, self.threshold, self.min_area = warmup_frames, threshold, min_area
        self.max_unstable_fraction, self.stable_std = max_unstable_fraction, stable_std
        self.relative_threshold, self.stable_relative = relative_threshold, stable_relative
        self.samples, self.background = [], None
        self.unstable_fraction = None

    @property
    def ready(self):
        return self.background is not None

    def _build_background(self):
        stack = np.stack(self.samples)
        self.samples.clear()
        count = np.isfinite(stack).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            bg = np.nanmedian(stack, axis=0)
            std = np.nanstd(stack, axis=0)
        bg[count < 0.5 * stack.shape[0]] = np.nan  # too few valid samples
        usable = np.isfinite(std)
        limit = np.maximum(self.stable_std, self.stable_relative * np.abs(bg))
        self.unstable_fraction = float((std[usable] > limit[usable]).mean()) if usable.any() else 1.0
        if self.unstable_fraction > self.max_unstable_fraction:
            warnings.warn(f'Background warm-up looks non-static: {self.unstable_fraction:.1%} of '
                          f'pixels vary more than max({self.stable_std} m, '
                          f'{self.stable_relative:.0%} of range). Is the scene empty?', RuntimeWarning)
        self.background = bg

    def update(self, depth):
        depth = np.asarray(depth, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        if self.background is None:
            self.samples.append(np.where(valid, depth, np.nan))
            if len(self.samples) < self.warmup_frames:
                return []
            self._build_background()
        limit = np.maximum(self.threshold, self.relative_threshold * np.abs(self.background))
        mask = valid & np.isfinite(self.background) & (np.abs(depth - self.background) > limit)
        mask = mask.astype('uint8') * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = int(cv2.contourArea(contour))
            if area >= self.min_area:
                regions.append(Region(x, y, w, h, area, 0.5))
        return regions
