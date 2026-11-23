
import numpy as np

def make_windows(features, labels, window_size, stride=1):
    # Call separately per video and track; never concatenate independent clips.
    if len(features) != len(labels):
        raise ValueError('features/labels length mismatch')
    xs, ys = [], []
    for start in range(0, len(features)-window_size+1, stride):
        end = start + window_size
        xs.append(features[start:end])
        vals, counts = np.unique(labels[start:end], return_counts=True)
        ys.append(vals[np.argmax(counts)])
    if not xs:
        return np.empty((0, window_size, features.shape[-1])), np.empty((0,), dtype=labels.dtype)
    return np.stack(xs), np.asarray(ys)
