import numpy as np


class FeatureScaler:
    """Median-impute NaNs, then standardize. Fit on TRAIN only; reuse unchanged everywhere."""

    def __init__(self):
        self.median = self.mean = self.std = None

    def fit(self, x):
        flat = np.asarray(x, dtype=np.float64).reshape(-1, np.shape(x)[-1])
        median = np.nanmedian(flat, axis=0)
        self.median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(np.isfinite(flat), flat, self.median)
        self.mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        self.std = np.where(std < 1e-6, 1.0, std)
        return self

    def transform(self, x):
        if self.median is None:
            raise RuntimeError('FeatureScaler is not fitted')
        x = np.asarray(x, dtype=np.float64)
        x = np.where(np.isfinite(x), x, self.median)
        return ((x - self.mean) / self.std).astype(np.float32)

    def state_dict(self):
        return {'median': self.median.tolist(), 'mean': self.mean.tolist(), 'std': self.std.tolist()}

    @classmethod
    def from_state_dict(cls, state):
        s = cls()
        s.median, s.mean, s.std = (np.asarray(state[k]) for k in ('median', 'mean', 'std'))
        return s
