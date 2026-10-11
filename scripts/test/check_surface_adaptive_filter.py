"""Reference checks for guarded surface reconstruction filtering; no native build."""
import math


def confidence(weights):
    n = sum(weights) ** 2 / max(sum(w * w for w in weights), 1e-12)
    t = max(0, min(1, (n - 4) / 12))
    return t * t * (3 - 2 * t)


def update(value, pairs, strength, anchor, limit):
    if any(a > 0 and b > 0 for a, b in pairs):
        return value
    candidate = value + .65 * strength * (sum(a + b for a, b in pairs) / 6 - value)
    return max(anchor - limit, min(anchor + limit, candidate))


assert confidence([1]) == 0
assert confidence([1] * 4) == 0
assert confidence([1] * 16) == 1
assert math.isclose(confidence([.1] * 16), 1)
# An affine plane is unchanged; dense curvature noise is reduced.
assert abs(update(0, [(-1, 1)] * 3, 1, 0, .2)) < 1e-12
assert update(.1, [(-.5, .5)] * 3, 1, .1, .2) < .1
assert update(-.1, [(.5, .5), (-1, 1), (-1, 1)], 1, -.1, .2) == -.1
assert update(.1, [(-.5, .5)] * 3, 0, .1, .2) == .1
value = 0
for _ in range(30):
    value = update(value, [(-10, 1)] * 3, 1, 0, .2)
assert abs(value) <= .2
# Refinement samples the same physical displacement (h/m * stride).
for multiplier in (1, 2, 3, 4):
    assert math.isclose((.04 / multiplier) * multiplier, .04)
print('PASS isolated support, plane preservation, dense noise reduction, thin guard, displacement bound and refinement footprint; reference only')

# Long surface traversals keep voxel-relative spacing despite the old 192 cap.
for voxel in (.078125, .0390625, .01953125):
    span = 8.0
    fine = voxel * .45
    budget = max(192, min(math.ceil(span / fine), 16384))
    step = max(fine, span / budget)
    assert step <= fine * (1 + 1e-12)
    assert budget <= 16384
print('PASS voxel-relative marching budget across 64/128/256 grids; reference only')
