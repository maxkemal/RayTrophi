"""Non-build overlay/scroll reference and integration checks. No GPU execution."""
from pathlib import Path
from itertools import product
import math
import re
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from rt_repo_root import repo_root as _repo_root
root = _repo_root()
src = root / 'RayTrophiStudio/source'
read = lambda p: (src / p).read_text(encoding='utf-8')
overlay = read('src/Viewport/RayFusionProbeOverlay.cpp')
viewport = read('src/Backend/VulkanViewportBackend.cpp')
field = read('src/Viewport/MaterialPreviewProbeField.cpp')
api = read('src/Api/RtIpcRayFusion.cpp')
py = read('src/Api/RtPythonRayFusion.cpp')
ui = read('include/UI/rayfusion_status_panel.hpp')
assert 'ds.depthTestEnable = VK_TRUE' in overlay
assert 'ds.depthWriteEnable = VK_FALSE' in overlay
assert 'VK_COMPARE_OP_LESS_OR_EQUAL' in overlay
assert 'VK_DYNAMIC_STATE_SCISSOR' in overlay
assert 'GetForegroundDrawList' not in overlay and 'GetForegroundDrawList' not in ui
assert 'recordRayFusionProbeOverlay(cmd, viewProj);\n    vkCmdEndRenderPass(cmd);' in viewport
assert viewport.index('recordRasterPostPass(cmd,') < viewport.index('recordRayFusionProbeOverlay(cmd, viewProj);')
assert 'm_probeOverlay.reset();' in viewport
assert overlay.index('if (!rayFusionProbeOverlayRequested()) return;') < overlay.index('auto markers =')
assert 'vkCmdCopy' not in overlay and 'vkDeviceWaitIdle' not in overlay
assert 'sizeof(Push) == 96' in overlay
assert 'matrix[col * 4 + row] = viewProj.m[row][col]' in overlay
for name, service in [('overlay', 'Overlay'), ('follow_camera', 'FollowCamera')]:
    assert 'rayfusion.set_probe_' + name in api
    assert '"set_probe_' + name + '"' in py
    assert 'setRayFusionProbe' + service in ui
assert py.count('py::arg("enabled").noconvert()') >= 3
assert api.count('if (params.size() != 1)') >= 3
# Compiler derives output from stem: don't overwrite the vertex SPV with frag.
for name in ['rayfusion_probe_overlay.vert', 'rayfusion_probe_overlay_frag.frag']:
    assert (src / 'shaders' / name).exists()
    assert Path(name).stem + '.spv' in overlay
assert 'state->field.scroll(minimum, error)' in field
# The published window must be read back from the APPLIED grid. Pinning the
# old spelling here broke the moment the grid became a runtime value, so pin
# the INTENT: the params handed to the shader come from field.grid(), and the
# scroll target is a stored request rather than a build constant.
publish = field[field.index('if (accepted == 0u) return;'):]
assert 'const auto& grid = state->field.grid();' in publish
assert 'params.minimum[axis] = static_cast<int32_t>(grid.minimum[axis]);' in publish
assert 'params.spacing[0] = grid.spacing;' in publish
assert 'state->pump = state->field.stats().pending > 0u' in field
assert '!rayFusionProbeUpdatesPending()' in viewport
assert 'rayFusionProbeUpdatesPending() ||' in read('include/Backend/VulkanBackend.h')

# Independent spatial invariants: an X translation changes 8 of 32 cells;
# unchanged cells retain their toroidal slot. A distant jump replaces all.
counts = (4, 2, 4)
def cells(lo):
    return set(product(*(range(a, a+n) for a,n in zip(lo, counts))))
def slot(c):
    x,y,z = (v % n for v,n in zip(c, counts))
    return (z*2+y)*4+x
old = cells((-2,-1,-2)); new = cells((-1,-1,-2))
assert len(old & new) == 24 and len(new-old) == 8
assert len({slot(c) for c in new}) == 32
assert not (old & cells((20,20,20)))
# 8 triangle faces, indices in the octahedron corner table, bounded radius.
vert = read('shaders/rayfusion_probe_overlay.vert')
indices = list(map(int,re.search(r'int\[24\]\((.*?)\);',vert,re.S)[1].replace('\n','').split(',')))
assert len(indices) == 24 and set(indices) == set(range(6))
# Camera rotation is absent from the scroll block.
scroll = field[field.index('if (state->followCamera)'):field.index('auto budget =')]
assert 'm_camera.origin' in scroll and 'm_camera.lookAt' not in scroll
print('PASS: overlay depth/lifetime/controls and scroll contracts (no build/GPU test)')
