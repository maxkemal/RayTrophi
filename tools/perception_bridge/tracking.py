"""Gated global matching with constant-velocity prediction, at most ten people.

No appearance descriptors, images, or biometric identity. An ID is session-local.
Ambiguous crossings can still switch identity; expired tracks never reuse IDs.
"""
import math


def _smoothing_factor(dt, cutoff):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter3D:
    """Adaptive low-pass filter for 3D joint positions."""

    def __init__(self, min_cutoff=1.5, beta=1.0, d_cutoff=1.0):
        if min_cutoff <= 0 or beta < 0 or d_cutoff <= 0:
            raise ValueError("filter cutoffs must be positive and beta nonnegative")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = [0.0, 0.0, 0.0]
        self.t_prev = None

    def filter(self, value, t):
        x = [float(value[0]), float(value[1]), float(value[2])]
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = [0.0, 0.0, 0.0]
            self.t_prev = t
            return list(x)
        dt = t - self.t_prev
        if dt < 0:
            raise ValueError("filter timestamps must not decrease")
        if dt == 0:
            return list(self.x_prev)
        a_d = _smoothing_factor(dt, self.d_cutoff)
        dx = [(x[i] - self.x_prev[i]) / dt for i in range(3)]
        dx_hat = [a_d * dx[i] + (1.0 - a_d) * self.dx_prev[i] for i in range(3)]
        x_hat = []
        for i in range(3):
            cutoff = self.min_cutoff + self.beta * abs(dx_hat[i])
            a = _smoothing_factor(dt, cutoff)
            x_hat.append(a * x[i] + (1.0 - a) * self.x_prev[i])
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return list(x_hat)


class PersonTracker:
    def __init__(self, max_gap=0.5, gate_metres=1.0, min_cutoff=1.5, beta=1.0, d_cutoff=1.0):
        self.max_gap = max_gap
        self.gate = gate_metres
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.tracks = {}
        self.joint_filters = {}
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
        for pid in list(self.joint_filters):
            if pid not in self.tracks:
                del self.joint_filters[pid]
        ids = sorted(self.tracks)
        costs = []
        for pid in ids:
            tr = self.tracks[pid]
            dt = t - tr["t"]
            prediction = [tr["p"][i] + tr["v"][i]*dt for i in range(2)]
            costs.append([math.dist(prediction, p) for p in detections])

        memo = {}

        def match(row, used):
            state = (row, used)
            if state in memo:
                return memo[state]
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
            memo[state] = best
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

    def smooth_joints(self, pid, joints, t):
        filters = self.joint_filters.setdefault(pid, {})
        smoothed = {}
        for name, value in joints.items():
            flt = filters.get(name)
            if flt is None:
                flt = OneEuroFilter3D(
                    min_cutoff=self.min_cutoff,
                    beta=self.beta,
                    d_cutoff=self.d_cutoff,
                )
                filters[name] = flt
            xyz = flt.filter(value[:3], t)
            smoothed[name] = [xyz[0], xyz[1], xyz[2], value[3]]
        return smoothed

    def update_poses(self, persons, t):
        ids = self.update([(p["joints"]["pelvis"][0], p["joints"]["pelvis"][2])
                           for p in persons], t)
        for person, pid in zip(persons, ids):
            person["id"] = pid
            person["joints"] = self.smooth_joints(pid, person["joints"], t)
        return ids
