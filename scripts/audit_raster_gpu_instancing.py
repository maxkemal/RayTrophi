"""Raster GPU instancing: one shared build tail, and a switchable A/B lever.

Source audit only. It compiles nothing and renders nothing -- and for THIS
defect that limit matters more than usual: the fault it guards against has no
visual symptom at all. The scene draws correctly with culling off; it is only
slow. Numbers, not pictures, are the whole evidence base here.
"""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
src = root / 'RayTrophiStudio/source'
read = lambda p: (src / p).read_text(encoding='utf-8')
raster = read('src/Backend/VulkanBackend_Raster.cpp')
viewport = read('src/Backend/VulkanViewportBackend.cpp')
backend_h = read('include/Backend/VulkanBackend.h')
ibackend = read('include/Backend/IBackend.h')
api_h = read('include/Api/RtApi.h')
api = read('src/Api/RtApiViewport.cpp')
ipc = read('src/Api/RtIpc.cpp')
py = read('src/Api/RtPython.cpp')
panel = read('include/UI/viewport_realtime_quality_panel.hpp')
descriptors = read('src/Api/RtIpcMethodDescriptors.cpp')

# ── ★★★★ The defect: the realtime viewport's buildRasterGeometry override kept
#    its own COPY of the base class's tail and dropped the layout call, so the
#    global instance buffer could never turn on there. Both builds must now end
#    in the SAME body -- a copy is how the call went missing.
assert 'void VulkanBackendAdapter::refreshRasterInstanceLayout()' in raster
assert raster.count('refreshRasterInstanceLayout();') >= 2   # definition + base tail
assert 'refreshRasterInstanceLayout();' in viewport, \
    'the realtime viewport override must end in the shared layout body'

# rebuildRasterInstanceLayout is the PRODUCER of m_rasterUseGlobalInstBuffer and
# must be reached only through the shared body, never inlined into a build tail.
calls = [m.start() for m in re.finditer(r'(?<!void VulkanBackendAdapter::)\brebuildRasterInstanceLayout\(\)',
                                        raster)]
shared = raster.index('void VulkanBackendAdapter::refreshRasterInstanceLayout()')
shared_end = raster.index('bool VulkanBackendAdapter::setRasterGpuInstancing')
assert all(shared < c < shared_end for c in calls), \
    'rebuildRasterInstanceLayout() is called outside the shared body again'
assert 'rebuildRasterInstanceLayout' not in viewport

# The per-mesh fallback upload loop must live in the shared body only. A second
# copy is the same defect wearing different clothes.
assert viewport.count('uploadRasterInstanceBuffer(mesh);') == 0, \
    'the viewport override is uploading per-mesh again instead of using the shared body'

# ── The lever, and why it has to exist ──────────────────────────────────────
# This path had never run in the realtime viewport. A fix that cannot be turned
# off destroys the measurement that would judge it.
assert 'm_rasterGpuInstancingAllowed' in backend_h
assert 'setRasterGpuInstancing' in ibackend and 'rasterGpuInstancingAllowed' in ibackend
assert '!m_rasterGpuInstancingAllowed' in raster, 'the lever must gate the layout builder'
assert 'Result setRasterGpuInstancing' in api and 'setRasterGpuInstancing' in api_h
assert '"viewport.set_raster_gpu_instancing"' in ipc
assert '"set_raster_gpu_instancing"' in py
assert '"viewport.set_raster_gpu_instancing", "viewport"' in descriptors
# viewport.* is Render by namespace; the audit_ipc_capabilities mirror covers it.

# Turning it OFF must leave a searchable trace: the scene stays CORRECT and only
# gets slow, so months later there is no other evidence.
off = api[api.index('Result setRasterGpuInstancing'):]
off = off[:off.index('\nbool rasterGpuInstancing()')]
assert 'SCENE_LOG_WARN' in off

# ── Panel: the MEASUREMENT is shown, the switch deliberately is not ─────────
# Same reasoning as viewport.set_scene_load_guard: leaving the faulty path
# selectable from a menu turns a rule into a preference.
assert 'gpu_culling' in panel and 'visible_triangles' in panel
assert 'scatter_triangle_target' in panel
# Comments are stripped first: the panel NAMES the method in the note that
# explains why it is not offered as a widget, and an audit that cannot tell
# code from prose would have to be silenced by deleting the explanation.
panel_code = re.sub(r'//[^\n]*', '', panel)
assert 'set_raster_gpu_instancing' not in panel_code, \
    'the faulty-path switch must not become a menu preference'

# ── The consumer side was already complete; only the producer call was missing.
# If these ever disappear, enabling the flag would draw nothing at all.
for needed in ('m_rasterGpuCullActive = true;', 'm_rasterGlobalInstBuf->vkBuffer()',
               'rebuildRasterCullBindings();'):
    assert needed in viewport, needed
# ...and the fallback that runs when the cull fails must still be reported, not
# silent: with the global buffer on, uploadVisibleRasterInstances early-returns,
# so a failed cull draws EVERY instance.
assert 'GPU culling kurulamadi' in viewport

# ── The drain-free write is what the flag actually buys per frame ───────────
fast = raster[raster.index('if (m_rasterUseGlobalInstBuffer && m_rasterGlobalInstBuf'):]
fast = fast[:fast.index('drainInteractiveViewportInFlight();')]
assert 'NO DRAIN' in fast and 'return;' in fast, \
    'the global path must still return before the draining per-mesh upload'

print('PASS: one shared instance-layout body, lever wired across API/IPC/Python,')
print('      panel reports the measurement, consumer path intact.')
print('NOT tested: build, GPU culling correctness, or any frame time.')
