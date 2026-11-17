"""Gated global matching with constant-velocity prediction, at most ten people.

No appearance descriptors, images, or biometric identity. An ID is session-local.
Ambiguous crossings can still switch identity; expired tracks never reuse IDs.
"""
from functools import lru_cache
import math


class PersonTracker:
    def __init__(self, max_gap=0.5, gate_metres=1.0):
        self.max_gap = max_gap
        self.gate = gate_metres
        self.tracks = {}
        self.next_id = 1
        self.last_t = None

    def update(self, detections, t):
        if not math.isfinite(t) or (self.last_t is not None and t <= self.last_t):
            raise ValueError("tracker timestamps must increase")
        if len(detections) > 10 or any(
                len(p) != 2 or not all(math.isfinite(v) for v in p) for p in detections):
            raise ValueError("tracker accepts at most ten finite X/Z positions")
        self.last_t = t
        self.tracks = {pid: tr for pid, tr in self.tracks.items()
                       if t - tr["t"] <= self.max_gap}
        ids = sorted(self.tracks)
        costs = []
        for pid in ids:
            tr = self.tracks[pid]
            dt = t - tr["t"]
            prediction = [tr["p"][i] + tr["v"][i]*dt for i in range(2)]
            costs.append([math.dist(prediction, p) for p in detections])

        @lru_cache(None)
        def match(row, used):
            if row == len(ids):
                return 0, 0.0, ()
            count, cost, assignment = match(row + 1, used)
            best = (count, cost, (-1,) + assignment)
            for j, distance in enumerate(costs[row]):
                if used & (1 << j) or distance > self.gate:
                    continue
                n, c, tail = match(row + 1, used | (1 << j))
                candidate = (n + 1, c + distance, (j,) + tail)
                if (-candidate[0], candidate[1]) < (-best[0], best[1]):
                    best = candidate
            return best

        assignment = match(0, 0)[2]
        result = [None] * len(detections)
        for row, j in enumerate(assignment):
            if j >= 0:
                result[j] = ids[row]
        for j, point in enumerate(detections):
            pid = result[j]
            velocity = (0.0, 0.0)
            if pid is None:
                pid = self.next_id
                self.next_id += 1
                result[j] = pid
            else:
                tr = self.tracks[pid]
                dt = t - tr["t"]
                velocity = tuple((point[i] - tr["p"][i]) / dt for i in range(2))
            self.tracks[pid] = {"p": tuple(point), "v": velocity, "t": t}
        return result
