from dataclasses import dataclass
from fall_detection.schemas import Region


def iou(a, b):
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x+a.w, b.x+b.w), min(a.y+a.h, b.y+b.h)
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = a.w*a.h + b.w*b.h - inter
    return inter / union if union > 0 else 0.0


def centroid_distance(a, b):
    ax, ay = a.x + a.w / 2, a.y + a.h / 2
    bx, by = b.x + b.w / 2, b.y + b.h / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


@dataclass
class Track:
    region: Region
    missed: int = 0


class IoUTracker:
    """Greedy baseline tracker; not suitable for claiming robust crowded-scene tracking.

    A pair matches if IoU >= threshold OR the centroid lies within
    gate_factor * max(track w, track h). The centroid gate keeps the ID when a
    box flips from upright to horizontal during a fall (IoU alone collapses to
    ~the threshold there). Set gate_factor=0 to get plain IoU matching.
    """

    def __init__(self, threshold=0.2, max_missed=15, gate_factor=0.5):
        self.threshold, self.max_missed, self.gate_factor = threshold, max_missed, gate_factor
        self.tracks, self.next_id = {}, 1

    def _candidates(self, regions):
        for tid, t in self.tracks.items():
            gate = self.gate_factor * max(t.region.w, t.region.h)
            for idx, r in enumerate(regions):
                o, d = iou(t.region, r), centroid_distance(t.region, r)
                in_gate = gate > 0 and d <= gate
                if o >= self.threshold or in_gate:
                    yield (o + (0.5 * (1 - d / gate) if in_gate else 0.0), tid, idx)

    def update(self, regions):
        pairs = sorted(self._candidates(regions), reverse=True)
        used_t, used_r, output = set(), set(), []
        for score, tid, idx in pairs:
            if tid in used_t or idx in used_r:
                continue
            self.tracks[tid].region = regions[idx]
            self.tracks[tid].missed = 0
            used_t.add(tid); used_r.add(idx)
            output.append((tid, regions[idx]))
        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.max_missed:
                    del self.tracks[tid]
        for idx, region in enumerate(regions):
            if idx not in used_r:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = Track(region)
                output.append((tid, region))
        return output
