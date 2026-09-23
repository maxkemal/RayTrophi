"""Non-build reference and cross-file contracts; does not execute GLSL."""
from pathlib import Path
import re
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from rt_repo_root import repo_root as _repo_root
root = _repo_root()
src = root / 'RayTrophiStudio/source'
read = lambda p: (src / p).read_text(encoding='utf-8')
shader = read('shaders/rayfusion_specular_visibility.glsl')
frag = read('shaders/material_preview_frag.frag')
host = read('src/Viewport/MaterialPreviewProbeField.cpp')
trace = read('src/Viewport/RayFusionProbeTrace.cpp')
code = re.sub(r'//[^\n]*', '', shader)
assert 'RayFusion::kProbeTraceDistance' in trace
assert 'state->activeProducer == "traced"' in host
assert 'params.spacing[1]' in host and 'RayFusion::kProbeTraceDistance : 0.0f' in host
assert 'packet.irradiance.rgb' not in code
assert 'rfDirectionalIndex(slot, r)' in code
assert 'rfMomentVisibility(packet.distance.xy, horizon * 0.999)' in code
assert code.index('lessThan(neighbour, lo)') < code.index('rfSlotFor(neighbour)')
assert 'if (!sceneReflectionValid && lightingPreset == 3u)' in frag
assert 'envSpecular *= rfSpecularSkyVisibility(vWorldPos, N, reflect(-V, N));' in frag
# R belongs to the earlier lighting branches, not the final composition scope.
assert 'rfSpecularSkyVisibility(vWorldPos, N, R)' not in frag
assert '+ specularLit + envSpecular' in frag

# Independent endpoint/mixture reference: no geometry -> unchanged sky,
# enclosing deterministic wall -> zero sky, blocked probe connection -> zero.
def moment(mean, square, distance):
    if distance <= mean:
        return 1.0
    variance = max(square - mean * mean, 0.0)
    return (variance / (variance + (distance - mean)**2))**3

assert moment(200, 40000, 199.8) == 1
assert moment(3, 9, 199.8) == 0
assert moment(2, 4, 3) * moment(200, 40000, 199.8) == 0
# A wide moment lobe mixes wall hits and misses: explicitly an estimate, not
# exact aperture coverage. Keep bounded/monotonic, don't call it visibility RT.
values = []
for miss_fraction in (0, 0.25, 0.5, 0.75, 1):
    mean = 3 * (1-miss_fraction) + 200 * miss_fraction
    square = 9 * (1-miss_fraction) + 40000 * miss_fraction
    values.append(moment(mean, square, 199.8))
assert all(0 <= v <= 1 for v in values)
assert values == sorted(values)
# Spatial weights normalize; occlusion never does. A lone blocked neighbour
# cannot become unoccluded merely because it is the only available sample.
assert (0.25 * 0.0) / 0.25 == 0
print('PASS: specular visibility endpoint/mixture and source contracts; GPU untested')
