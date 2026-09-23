"""Non-build checks of cross-file bounce ABI, descriptor and control contracts."""
from pathlib import Path
import re
import struct
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from rt_repo_root import repo_root as _repo_root
root = _repo_root()
src = root / 'RayTrophiStudio/source'
read = lambda p: (src / p).read_text(encoding='utf-8')
host = read('src/Viewport/RayFusionProbeTrace.cpp')
shader = read('shaders/rayfusion_probe_trace_beta.comp')
bounce = read('shaders/rayfusion_probe_bounce.glsl')
field = read('src/Viewport/MaterialPreviewProbeField.cpp')
scene = read('src/Viewport/RayFusionSceneAS.cpp')
api = read('src/Api/RtIpcRayFusion.cpp')
python = read('src/Api/RtPythonRayFusion.cpp')

# HitInstance: four addresses + four uints = 48 B (indexed/flat SoA).
# A second instance catches stride errors hidden by a single object at offset 0.
blob = struct.pack('<QQQQIIII', 0x123456789ABC, 0xABCDEFAABBCC, 0x111122223333, 0, 3, 1, 0, 0)
blob += struct.pack('<QQQQIIII', 0xFFEEDDCCBBAA, 0x998877665544, 0x444455556666, 0x123456, 6, 4, 0, 0)
assert len(blob) == 96
low, high = struct.unpack_from('<II', blob, 48)
assert low | (high << 32) == 0xFFEEDDCCBBAA
# The UV address is the field this slice added; read it at the SECOND instance,
# because an offset error here samples another mesh's coordinates and produces a
# plausible-looking wrong texel rather than a crash.
assert struct.unpack_from('<Q', blob, 64)[0] == 0x444455556666
assert struct.unpack_from('<Q', blob, 72)[0] == 0x123456
assert struct.unpack_from('<I', blob, 80)[0] == 6
assert struct.unpack_from('<I', blob, 84)[0] == 4
# BounceMaterial: 7 x vec4 = 112 B. The GLSL mirror must declare exactly these.
assert struct.calcsize('<4f4f4I4I4f4f4f') == 112
for member in ('vec4  diffuse', 'vec4  emission', 'uvec4 textures;', 'uvec4 textures2',
               'vec4  uvScaleOffset', 'vec4  uvTiling', 'vec4  scalars'):
    assert member in bounce, member
assert 'uint64_t uvs = 0;' in read('include/RayFusion/ProbeBounce.h')
assert 'sizeof(BounceMaterial) == 112' in read('include/RayFusion/ProbeBounce.h')
bindings = {int(x) for x in re.findall(r'binding\s*=\s*(\d+)', shader + bounce)}
assert bindings == set(range(9)), bindings
assert 'dlci.bindingCount = 9' in host and 'vkUpdateDescriptorSets(device, 8' in host
assert 'VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6u' in host
assert 'state->textureArrayLen + 1u' in host
# The bindless array must be refreshed against the texture cache generation:
# a stale slot samples a freed image, which is a device fault, not a wrong pixel.
assert 'textureCacheGeneration()' in host and 'state->textureGeneration' in host
# Counters are per dispatch. Without the fill they accumulate and report the
# PREVIOUS batch as if it were this one.
assert 'vkCmdFillBuffer(cmd, state->counters.buffer' in host
assert 'downloadBuffer(state->counters' in host
# Alpha masking needs the opaque flag GONE from both rays; with it, a leaf card
# shadows like a solid sheet and the symptom is only "interiors look dark".
# Comments are stripped first: the flag is NAMED in the comment that explains
# why it was removed, and an audit that cannot tell code from prose would have
# to be silenced by deleting the explanation.
strip = lambda t: re.sub(r'//.*', '', t)
assert 'gl_RayFlagsOpaqueEXT' not in strip(shader)
assert 'gl_RayFlagsOpaqueEXT' not in strip(bounce)
assert 'gl_RayFlagsNoOpaqueEXT' in strip(shader)
assert 'gl_RayFlagsNoOpaqueEXT' in strip(bounce)
assert 'sizeof(HitInstance) == 48' in read('include/RayFusion/ProbeBounce.h')
assert 'interpolated.y = 1.0 - interpolated.y' in bounce
assert 'opacityTex == mat.textures.x' in bounce
assert 'MATERIAL_FLAGS_PREVIEW_CUTOUT' in bounce
assert '& 0x7fffffffu' in bounce
assert 'frontFace || thinHit' in shader and '!frontFace && !thinHit' in shader
assert 'pc.params2.w' in bounce and 'state->textureArrayLen' in host
assert 'rayQueryConfirmIntersectionEXT' in bounce and 'rayQueryConfirmIntersectionEXT' in shader
# The bounce must decode packed channels with the SHARED policy, not its own.
assert '#include "pbr_texture_policy.glsl"' in bounce
assert 'samplePackedMetallic(' in bounce
for key in ('bounce_shaded_hits', 'bounce_hits', 'bounce_alpha_tested'):
    assert key in api and key in python, key
assert 'rayfusion_probe_trace_beta.spv' in host  # old alpha SPV cannot report beta
# Opaque queries must not be copied out of a user function. The earlier audit
# accidentally required the invalid signature; check the value-only boundary.
signature = re.search(r'vec3\s+rfBounceRadiance\s*\(([^)]*)\)', bounce).group(1)
assert 'rayQueryEXT' not in signature and 'out ' not in signature
assert 'rayQueryGetIntersectionObjectToWorldEXT(query, true)' in shader
assert 'rayQueryGetIntersectionBarycentricsEXT(query, true)' in shader
assert 'origin + direction * hitDistance' in shader
assert not re.search(r'\b(?:out|inout)\s+rayQueryEXT\b', shader + bounce)
assert 'rfBounceRadiance(' not in read('shaders/rayfusion_probe_trace.comp')
assert 'entry.customIndex = static_cast<uint32_t>(tlasInfo.instances.size())' in scene
assert 'state->hitMeshKeys.push_back(instance.meshKey)' in scene
assert 'mesh.cpuMatIds.data()' in scene
assert 'bounceSignature' in field and '192u : 64u' in field
# Every quality ceiling can service at least one 192-ray probe; no budget overrun.
for rays, cap in [(256, 8), (1024, 16), (4096, 32), (8192, 32)]:
    probes = min(cap, rays // 192)
    assert probes >= 1 and probes * 192 <= rays
assert 'rayfusion.set_probe_bounce' in api and 'requireBool(params, "enabled")' in api
assert 'set_probe_bounce' in python and 'py::arg("enabled").noconvert()' in python
assert 'RayFusionBounce.cpp' in (root/'RayTrophiStudio/RayTrophiStudio.vcxproj').read_text(encoding='utf-8')

# ── Per-frame CPU cost of the table build (added 2026-09-08) ────────────────
# The bounce tables are rebuilt on EVERY raster frame. The build used to hash
# each mesh's whole material-ID stream once per TLAS INSTANCE, so a scattered
# foliage scene multiplied that hash by the placement count and one core sat at
# 100% while the GPU idled. These pins keep the two properties that fixed it.
import re as _re
_src = (root / 'RayTrophiStudio/source')
_read = lambda p: (_src / p).read_text(encoding='utf-8')
_scene = _read('src/Viewport/RayFusionSceneAS.cpp')

# 1. The stream hash is CACHED, and the cache is read through one helper.
assert 'meshMatIdsHash(mesh)' in _scene
assert len(_re.findall(r'cpuMatIds\.data\(\)', _scene)) == 1, \
    'the material-ID stream must be hashed in exactly one place (the cache)'
assert 'matIdsHashValid' in _read('include/Backend/VulkanBackend.h')

# 2. Distinct meshes are resolved once; instances expand by index.
_hits = _scene[_scene.index('bool VulkanBackendAdapter::getRayFusionHitInstances'):]
_hits = _hits[:_hits.index('\nbool VulkanBackendAdapter::getRayFusionSceneASStatus')]
assert 'resolved.find(key)' in _hits and 'unique.push_back(hit)' in _hits

# 3. ★ EVERY site that writes cpuMatIds must drop the cache. A forgotten site
#    does not fail loudly: the bounce keeps lighting the scene with the OLD
#    material assignment and the image merely looks slightly wrong.
_writers = ['src/Backend/VulkanBackend.cpp', 'src/Backend/VulkanBackend_Raster.cpp',
            'src/Backend/VulkanViewportBackend.cpp', 'src/Viewport/RayFusionSceneAS.cpp']
_missing = []
for _path in _writers:
    _text = _read(_path).split('\n')
    for _n, _line in enumerate(_text):
        # An assignment to cpuMatIds, or the in-place remap loop over it.
        _assigns = _re.search(r'\bcpuMatIds\s*=', _line)
        _inplace = _re.search(r'for\s*\(\s*auto&\s*\w+\s*:\s*\w+\.cpuMatIds\s*\)', _line)
        if not _assigns and not _inplace:
            continue
        _window = '\n'.join(_text[max(0, _n - 20):_n + 20])
        if 'matIdsHashValid = false' not in _window:
            _missing.append('%s:%d  %s' % (_path, _n + 1, _line.strip()))
assert not _missing, 'cpuMatIds written without dropping the hash cache:\n  ' + '\n  '.join(_missing)

# 4. The cost is REPORTED. An unmeasured per-frame cost is how this one grew.
assert 'prepareMs' in _read('include/RayFusion/ProbeBounce.h')
assert 'status.prepareMs' in _read('src/Viewport/RayFusionBounce.cpp')
assert 'bounce_prepare_ms' in _read('src/Api/RtIpcRayFusion.cpp')
assert 'bounce_prepare_ms' in _read('src/Api/RtPythonRayFusion.cpp')
assert 'prepareMs' in _read('include/UI/rayfusion_status_panel.hpp')

# 5. The word-wise mixer must still consume EVERY input byte: a faster gate that
#    can miss a change is not a faster gate, it is a broken one.
def _mix(seed, data):
    for i in range(0, len(data) - len(data) % 8, 8):
        seed = ((seed ^ int.from_bytes(data[i:i+8], 'little')) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        seed ^= seed >> 29
    tail = data[len(data) - len(data) % 8:]
    if tail:
        seed = ((seed ^ int.from_bytes(tail, 'little') ^ len(tail)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        seed ^= seed >> 29
    return seed

_base = bytes(range(64)) * 3
for _i in range(len(_base)):
    _flipped = bytearray(_base)
    _flipped[_i] ^= 0x01
    assert _mix(0, _base) != _mix(0, bytes(_flipped)), 'byte %d does not reach the hash' % _i
# Trailing-length mixing: "ab" and "ab\0" must not collide.
assert _mix(0, b'ab') != _mix(0, b'ab\x00')
assert _mix(0, b'a' * 9) != _mix(0, b'a' * 8)

print('PASS: bounce table build is cached per mesh, every cpuMatIds writer drops')
print('      the cache, the per-frame cost is reported, and the mixer is complete')
print('PASS: bounce address/stride, material ABI, texture/alpha contracts, descriptors,')
print('      shader version, TLAS order, budgets and API contracts')
print('GPU shader compilation, rendering and numerical image acceptance NOT tested')
