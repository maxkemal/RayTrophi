"""Reference checks for volume-preserving covariance surface axes (no build)."""
import math


def axes(variance, stretch):
    floor = max(variance) / (stretch * stretch)
    lengths = [math.sqrt(max(v, floor)) for v in variance]
    volume = math.prod(lengths) ** (1 / 3)
    return [v / volume for v in lengths]


for variance in ([1, 1, 1], [9, 1, 1], [9, 9, 0], [0, 0, 9]):
    shape = axes(variance, 3)
    assert math.isclose(math.prod(shape), 1, rel_tol=1e-12)
    assert max(shape) / min(shape) <= 3 + 1e-12
    radius = 0.7
    for axis in range(3):
        distance = radius * shape[axis]
        residual = distance / shape[axis] - radius
        assert abs(residual) < 1e-12
assert axes([9, 9, 0], 3)[2] < axes([9, 9, 0], 3)[0]
assert axes([0, 0, 9], 3)[2] > axes([0, 0, 9], 3)[0]
print('PASS axis aspect ratio, volume, sheet/stream orientation and ellipsoid crossings; no native execution')
