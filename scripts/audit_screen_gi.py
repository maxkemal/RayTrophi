"""Source/ABI and numerical regression checks; never invokes a compiler or app."""
from pathlib import Path
import itertools
import math
import re
import struct
import sys
import json
import xml.etree.ElementTree as ET
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from rt_repo_root import repo_root as _repo_root
ROOT = _repo_root()
SRC=ROOT/'RayTrophiStudio/source'
read=lambda p:(SRC/p).read_text(encoding='utf-8')
host=read('src/Viewport/ScreenGi.cpp')
trace=read('shaders/screen_gi_trace.comp')
filter_shader=read('shaders/screen_gi_filter.comp')
consumer=read('shaders/material_preview_screen_gi.glsl')
main=read('shaders/material_preview_frag.frag')
bounce=read('shaders/rayfusion_probe_bounce.glsl')
shadow=read('src/Viewport/MaterialPreviewRtShadow.cpp')
assert struct.calcsize('<16f4f4f4I')==112
assert struct.calcsize('<4f4f')==32 and struct.calcsize('<4I')==16
assert 'sizeof(GiPush)==112' in host and 'kGiPixel=32' in host
assert 'kGiHeader=16' in host
assert 'binding=24' in consumer and 'write.dstBinding=24' in host
for p in ('src/Backend/VulkanBackend.cpp','src/Backend/VulkanViewportBackend.cpp'):
    code=read(p)
    assert 'mpDslBindings[25]' in code and 'mpBindingFlags[25]' in code,p
    assert 'binding <= 24' in code and 'mpDslci.bindingCount = 25' in code,p
    assert 'mpPoolSizes[0].descriptorCount = 12' in code,p
assert 'kStageCount = 13' in read('include/Viewport/RasterStageTimings.h')
assert 'ScreenGiTrace' in read('include/Viewport/RasterGpuTimers.h')
assert 'ScreenGiFilter' in read('src/Viewport/RasterGpuTimers.cpp')
# Separate buffers, barriers and current-frame invalidation. No in-place filter.
assert '{s.raw.buffer,0,VK_WHOLE_SIZE}' in host
assert '{s.output.buffer,0,VK_WHOLE_SIZE}' in host
assert 'readonly buffer RawGi' in filter_shader
assert 'writeonly buffer FilteredGi' in filter_shader
assert 'filtered[i]=center' in filter_shader
assert shadow.index('resetScreenGiFrame(cmd,width,height)') < shadow.index('if(!m_rtShadowAllowed || !eligible)')
assert shadow.index('markRasterStage(cmd,RasterStage::RtShadow,true)') < shadow.index('recordScreenGi(cmd,viewProj)')
assert host.index('vkCmdDispatch(cmd,(s.width+7)/8,(s.height+7)/8,1)') < host.index('0x53474931u')
assert 'maxStorageBufferRange' in host
assert 'supportedSamples==0u' in trace
assert 'float(supportedSamples)/float(pc.shape.z)' in trace
assert 'mat.diffuse.w<0.5) continue;' in trace
assert 'center.light.w<=0.0' in filter_shader
assert 'vec4(sum/weights,center.light.w)' in filter_shader
assert 's.light.w<=0.0' in consumer and 'out float confidence' in consumer
assert 'mix(fallbackIrradiance, irradiance, screenGiConfidence)' in main
assert 'screenGiConfidence < 1.0' in main
assert 'gl_RayFlagsNoOpaqueEXT' in trace
assert '#include "rayfusion_probe_bounce.glsl"' in trace
assert 'RF_ENV_SCALE pc.params2.y' in trace
assert '* RF_ENV_SCALE' in bounce
assert 'm_cachedWorld.env_intensity' in host
assert 'copy' not in re.sub(r'//.*','',filter_shader).lower()
for forbidden in ('vkQueueWaitIdle','vkDeviceWaitIdle','downloadBuffer','previousFrame','historyBuffer'):
    assert forbidden not in host+trace+filter_shader,forbidden

# Four samples with one unknown must retain the other three contributions.
# Old all-or-nothing rejection returned fallback for every mixed pattern.
fallback=0.75
def compose(samples):
    known=[x for x in samples if x is not None]
    if not known: return fallback
    confidence=len(known)/len(samples)
    return sum(known)/len(known)*confidence+fallback*(1-confidence)
for count in (1,2,4):
    for samples in itertools.product((None,0.0,0.2,1.4),repeat=count):
        expected=sum(fallback if x is None else x for x in samples)/count
        assert math.isclose(compose(samples),expected,abs_tol=1e-12)
        assert math.isclose(compose(samples*2),expected,abs_tol=1e-12)
assert compose((0.0,))==0.0 # valid darkness must never become a fallback hole
assert compose((None,))==fallback
assert compose((0.2,None,0.6,1.0))!=fallback
# Spatial weights cannot bridge a distant parallel surface or flipped normal.
def weight(normal_dot,plane,footprint,confidence):
    return max(normal_dot,0)**32*math.exp(-plane/(footprint*0.5))*confidence
assert weight(1,0,0.01,0.25)>0
assert weight(1,10,0.01,1)<1e-12
assert weight(-1,0,0.01,1)==0
for confidence in (0.25,0.5,0.75,1):
    values=[(2.0,confidence),(2.0,0.25)]
    assert sum(c*w for c,w in values)/sum(w for _,w in values)==2.0

# Both external interfaces and UI use the same validation/service.
for path in ('src/Api/RtIpcScreenGi.cpp','src/Api/RtPythonScreenGi.cpp','include/UI/ScreenGiUI.h'):
    assert 'rtapi::setScreenGi(' in read(path),path
assert 'validateScreenGi(settings,error)' in host
assert 'validateScreenGi(settings, error)' in read('src/Api/RtApiScreenGi.cpp')
assert 'rayfusion.screen_gi' in read('src/Api/RtIpcSecurity.cpp')
assert 'rayfusion.set_screen_gi' in read('src/Api/RtIpcSecurity.cpp')
assert 'out.screen_gi = screenGiStatus()' in read('src/Viewport/RasterStageInstrumentation.cpp')
assert 'a.screen_gi.settings.samples == b.screen_gi.settings.samples' in read('src/Viewport/RasterStageTimings.cpp')
sys.path.insert(0,str(ROOT/'scripts'))
import gen_ipc_descriptors as gen
records=gen.build(json.loads(Path(gen.OVERLAY).read_text(encoding='utf-8')),gen.namespace_table(gen.read(gen.SECURITY)))
selected=[r for r in records if r['method'] in ('rayfusion.screen_gi','rayfusion.set_screen_gi')]
assert len(selected)==2
assert len(next(r for r in selected if r['method']=='rayfusion.set_screen_gi')['params'])==4
fragment=gen.emit(selected)[len(gen.HEADER):].split('} // namespace')[0].strip()
assert fragment in read('src/Api/RtIpcMethodDescriptors.cpp')
project=ROOT/'RayTrophiStudio/RayTrophiStudio.vcxproj'
ET.parse(project)
project_text=project.read_text(encoding='utf-8')
for name in ('ScreenGi.cpp','RtApiScreenGi.cpp','RtIpcScreenGi.cpp','RtPythonScreenGi.cpp'):
    assert name in project_text,name
visited=set()
def includes(name):
    if name in visited:return
    visited.add(name)
    text=read('shaders/'+name)
    for child in re.findall(r'^\s*#include\s+"([^"]+)"',text,re.M):includes(child)
for name in ('screen_gi_trace.comp','screen_gi_filter.comp','material_preview_screen_gi.glsl'):includes(name)
print('PASS: screen GI ABI, descriptor paths, lifecycle/barriers, timing and API wiring')
print('PASS: partial-sample fallback, valid darkness, spatial edges and constant-light regressions')
print('Shader/C++ compilation and corrected GPU output were NOT tested.')
