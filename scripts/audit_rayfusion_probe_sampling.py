"""Non-build sampling contract checks; numerical reference, NOT GPU execution."""
from itertools import product
from math import floor, isfinite
from pathlib import Path


def visibility(mean, square, distance):
    if not all(isfinite(v) and v >= 0 for v in (mean, square, distance)):
        return 0
    if square + 1e-5 * max(mean * mean, 1) < mean * mean:
        return 0
    if distance <= mean:
        return 1
    variance = max(square - mean * mean, 0)
    return (variance / (variance + (distance - mean) ** 2)) ** 3


def neighbours(position, spacing=3, lo=(-2, -1, -2), counts=(4, 2, 4)):
    q = [p / spacing - 0.5 for p in position]
    base = [floor(p) for p in q]
    t = [p - b for p, b in zip(q, base)]
    result = []
    for offset in product(range(2), repeat=3):
        cell = tuple(b + o for b, o in zip(base, offset))
        if not all(l <= c < l + n for c, l, n in zip(cell, lo, counts)):
            continue
        w = 1
        for o, f in zip(offset, t):
            w *= f if o else 1 - f
        if w:
            result.append((cell, w))
    return result


# Producer centres must reproduce their own sample, including negative cells.
for cell in product(range(-2, 2), range(-1, 1), range(-2, 2)):
    assert neighbours(tuple((c + 0.5) * 3 for c in cell)) == [(cell, 1)]
middle = neighbours((0, 0, 0))
assert len(middle) == 8 and all(w == 0.125 for _, w in middle)
# Grid borders drop out-of-window cells rather than wrapping their contents.
edge = neighbours((-5.9, 0, 0))
assert all(c[0] == -2 for c, _ in edge)
assert 0 < sum(w for _, w in edge) < 1
# Constant sky is invariant even with missing neighbours/normal weights.
for samples in (middle, edge, middle[:3]):
    weights = [w * (i + 1) / 8 for i, (_, w) in enumerate(samples)]
    assert abs(sum(7 * w for w in weights) / sum(weights) - 7) < 1e-12
# Occlusion cannot normalize back into full sky, even with a sole neighbour.
assert visibility(1, 1, 2) == 0
assert visibility(2, 5, 3) == 0.125
assert visibility(2, 1, 1) == 0
assert visibility(float('nan'), 1, 1) == 0
assert visibility(10000, 100000000, 10) == 1
assert 7 * visibility(2, 5, 3) == 0.875
# Spatial interpolation is continuous across the OLD cell boundary at x=0.
def ramp(x):
    return sum((c[0] + 0.5) * 3 * w for c, w in neighbours((x, 0, 0)))
assert abs(ramp(-1e-6) - ramp(1e-6)) < 3e-6
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from rt_repo_root import repo_root as _repo_root
root = _repo_root()
shader = (root / 'RayTrophiStudio/source/shaders/probe_field.glsl').read_text(encoding='utf-8')
# Link the reference's essential contracts to the actual source, without
# presenting these text assertions as shader compilation or visual validation.
for contract in (
    'worldPos / spacing - vec3(0.5)',
    'rfDirectionalIndex(slot, direction)].distance.xy',
    'rfDirectionalIndex(slot, n)',
    'baseWeightSum += weight;',
    'weight * visibility',
    'irradiance /= baseWeightSum;',
):
    assert contract in shader, contract
assert shader.index('lessThan(neighbour, lo)') < shader.index('rfSlotFor(neighbour)')
print('PASS: probe sampling numerical contracts and source wiring (no build/GPU test)')
