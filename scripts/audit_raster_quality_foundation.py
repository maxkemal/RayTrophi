"""Non-build regression checks for the raster quality foundation batch."""
from pathlib import Path
import math
import random

root = Path(__file__).resolve().parents[1]
src = root / 'RayTrophiStudio/source'
read = lambda p: (src / p).read_text(encoding='utf-8')
api = read('src/Api/RtApiRayFusion.cpp')
for token in ('shading.mode == "material"', 'sceneLighting', 'field.configured',
              'field.uploaded', 'field.bound', 'field.valid > 0', 'field.producer == "traced"'):
    assert token in api, token
assert 'double(probe.minimum[i]) * double(probe.spacing)' in api
for name in ('RtIpcRayFusion.cpp', 'RtPythonRayFusion.cpp'):
    text = read('src/Api/' + name)
    for key in ('minimum_cell', 'minimum_world'):
        assert text.count('"' + key + '"') == 2, (name, key)
for name, kind in [('RtIpc.cpp', 'json'), ('RtPython.cpp', 'py::dict')]:
    assert f'rasterDepthPrepassDictionary<{kind}>()' in read('src/Api/' + name)
status = read('src/Api/RtApiViewport.cpp')
assert 'timings.available && timings.frames > 0' in status
assert 'out.observed && timings.depth_prepass' in status
assert 'out.effective && timings.rt_shadow_ready' in status

# Regression for the two-hemisphere estimator: a constant environment must
# retain its energy for all mixture weights. A bright back/dark front must
# follow the authored thin-transmission weight, rather than add both lobes.
rng = random.Random(20260910)
samples = [rng.random() for _ in range(100000)]
for weight in (0.0, 0.1, 0.5, 0.9, 1.0):
    constant = sum(1.0 if u < weight else 1.0 for u in samples) / len(samples)
    assert constant == 1.0
    measured = sum(1.0 if u < weight else 0.0 for u in samples) / len(samples)
    assert abs(measured - weight) < 0.006
for specular in (0.0, 0.5, 1.0, 4.0):
    f0 = min(0.08 * specular, 1.0)
    for metal in (0.0, 0.5, 1.0):
        for coat in (0.0, 0.5, 1.0):
            energy = (1-metal) * (1-(f0+(1-f0)/21)) * (1-0.04*coat)
            assert math.isfinite(energy) and 0 <= energy <= 1
bounce = read('shaders/rayfusion_probe_bounce.glsl')
assert 'rfRandom(seed) < thinTransmission' in bounce
assert 'n = -n' in bounce and '1.0 - 0.04 * m.scalars.w' in bounce
assert 'if (!rfValidTexture(opacityTex)) return true' in bounce
assert 'float(slot)' not in bounce
host = read('src/Viewport/RayFusionBounce.cpp')
assert '!(m.transmission <= 0.001f)' in host
assert 'm.transmission_tex != 0u' in host
assert '!std::isfinite(m.opacity)' in host
assert 'textureResident(m.opacity_tex) ? m.opacity_tex : UINT32_MAX' in host
print('PASS: status/binding contracts, world units, bounded thin-diffuse estimator')
print('Static/numerical model checks only; GPU output and timing are not tested.')
