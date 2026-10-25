"""Non-build contract check for the Vulkan RT gas march LOD.

    python scripts/test/check_volume_march_lod_contracts.py

docs/dev/RT_GAS_SECONDARY_MARCH.md §8. The setting travels as ONE packed world
word (volumeMarchLod, ex _aerialPad0) from Backend::packVolumeMarchLod to the
volume closest-hit. None of these breaks loudly when it drifts:
  * a shader mirror that still declares the old float keeps the offset, but a
    future reader would see garbage;
  * a pack/decode bit mismatch silently selects another stride or step cap;
  * a cached shadow still gated on `step % stride == 0` can go stale for a
    whole ray once lighting runs only on every k-th sample.
Source check only; the shader build is the user's.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "RayTrophiStudio", "source")
SH = os.path.join(SRC, "shaders")
fails = []


def read(*p):
    with open(os.path.join(*p), encoding="utf-8", errors="replace") as f:
        return f.read()


def check(cond, msg):
    if not cond:
        fails.append(msg)


world = read(SRC, "include", "Backend", "vulkan_world_data.h")
check("uint32_t volumeMarchLod;" in world, "vulkan_world_data.h lacks volumeMarchLod")
check(re.search(r"float\s+_aerialPad0;", world) is None, "vulkan_world_data.h still declares _aerialPad0")
for name in ("closesthit.rchit", "hair_closesthit.rchit", "raygen.rgen", "rt_sky.glsl",
             "volume_closesthit.rchit"):
    text = read(SH, name)
    check("volumeMarchLod" in text and "float _aerialPad0" not in text,
          name + ": world mirror not renamed to volumeMarchLod")

vol = read(SRC, "src", "Backend", "VulkanBackend_Volumes.cpp")
pack = vol[vol.index("uint32_t packVolumeMarchLod"):]
pack = pack[:pack.index("\n}\n")]
for frag in ("camera_light_stride, 1, 16) - 1u)", "secondary_light_stride, 1, 16) - 1u) << 4)",
             "secondary_step_scale, 1, 16) - 1u) << 8)", "secondary_max_steps, 0, 511) << 12)",
             "((lod.secondary_radiance_cache ? 1u : 0u) << 21)"):
    check(frag in pack, "packVolumeMarchLod layout changed: " + frag)

vch = read(SH, "volume_closesthit.rchit")
for frag in ("int(marchLod & 15u) + 1", "int((marchLod >> 4) & 15u) + 1",
             "float(((marchLod >> 8) & 15u) + 1u)", "int((marchLod >> 12) & 511u)",
             "((marchLod >> 21) & 1u) != 0u"):
    check(frag in vch, "volume_closesthit decode differs from the pack: " + frag)
check("volumeCameraSegment = (pathFlags & PL_CAMERA_SEGMENT) != 0u;" in vch,
      "volume_closesthit main() does not classify the segment")
check(re.search(r"\(step % (shadowStride|ambientShadowStride)\) == 0", vch) is None,
      "a cached shadow is still gated on step % stride == 0 (stale under light stride)")
check(vch.count("volumeShadowDue(step,") == 4, "expected 4 volumeShadowDue call sites")
check("heldInscatterStep = step;" in vch, "in-scatter hold not recorded")
# The first-sample adaptive stride was removed (§12); it must not come back
# without a majorant bound.
# Both variable strides were removed after live A/B (§12 density at segment
# start, §14 spent transmittance). One must not come back without a majorant.
check("segStride" not in vch and "adaptiveStepping" not in vch and
      "growStride" not in vch and "growMax" not in vch,
      "a removed variable march stride is back")
check("float dt = min(baseStep, tFar - t);" in vch and vch.count("volumeShadowDue(step, ") == 4,
      "march segment or shadow refresh changed from the fixed-lattice form")
# Radiance cache (§13).
check('#include "volume_radiance_cache.glsl"' in vch and
      "volumeRadianceCacheMarch(volIdx, vol, rayOrigin, rayDir, tNear, tFar," in vch and
      "if (cached) t = tFar;" in vch, "bounce rays are not routed through the radiance cache")
lc = read(SH, "volume_light_cache.glsl")
check("const uint VLC_LIGHT_CHANNELS = 6u;" in lc and
      "lightIndex >= int(VLC_LIGHT_CHANNELS) - 1" in lc,
      "light channel mapping follows the total channel count (lights would land on radiance channels)")
rc = read(SH, "volume_radiance_cache.glsl")
pub = rc.index("entries.e[iD] = (uint64_t(floatBitsToUint(d))")
check(all(rc.index("entries.e[iD + %du] = " % k) < pub for k in (1, 2, 3)),
      "radiance vertex publishes density (the claim channel) before its source channels")
check("atomicCompSwap(words.w[iD * 2u], filling, seen);" in rc,
      "an aborted radiance fill does not release its claim (vertex stuck as 'filling')")

backend = read(SRC, "src", "Backend", "VulkanBackend.cpp")
check("gw.volumeMarchLod = packVolumeMarchLod(volumeMarchLod());" in backend,
      "setWorldData does not upload the march LOD")

api = read(SRC, "src", "Api", "RtApi.cpp")
check("refreshVolumeMarchLod()" in api, "setVolumeMarchLod does not refresh the backends")
ipc = read(SRC, "src", "Api", "RtIpc.cpp")
check('"render.set_volume_march_lod"' in ipc and '"render.volume_march_lod"' in ipc,
      "IPC methods missing")
py = read(SRC, "src", "Api", "RtPython.cpp")
check('"set_volume_march_lod"' in py and '"volume_march_lod"' in py, "Python bindings missing")
ui = read(SRC, "src", "UI", "scene_ui_volume_performance.cpp")
check("rtapi::setVolumeMarchLod(" in ui, "panel does not drive the core setting")

if fails:
    print("FAIL")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("OK - gas march LOD: world word, pack/decode, shadow refresh, API layers agree.")
