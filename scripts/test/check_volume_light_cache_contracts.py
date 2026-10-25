"""Non-build contract check for the Vulkan RT volume light cache.

    python scripts/test/check_volume_light_cache_contracts.py

docs/dev/VOLUME_LIGHT_CACHE.md. The host writes a header the shaders parse,
the world buffer carries the address in two renamed pad words, and the
instrumentation buffer grew by four counters. None of these fail loudly when
the two sides disagree - a wrong channel count reads as "cache never hits",
a wrong word count shifts every counter after it - so they are compared here.

This is a source check, not GLSL execution; the shader build is the user's.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "RayTrophiStudio", "source")
SH = os.path.join(SRC, "shaders")

failures = []


def read(*parts):
    with open(os.path.join(*parts), encoding="utf-8", errors="replace") as f:
        return f.read()


def check(cond, message):
    if not cond:
        failures.append(message)


def const(text, name):
    m = re.search(r"\b" + name + r"\s*=\s*(\d+)u?", text)
    return int(m.group(1)) if m else None


glsl = read(SH, "volume_light_cache.glsl")
fill = read(SH, "volume_light_cache_fill.glsl")
host = read(SRC, "src", "Backend", "VulkanBackend_Volumes.cpp")

# 1. Header layout constants agree.
for g, h in (("VLC_CHANNELS", "kVlcChannels"), ("VLC_MAX_SLOTS", "kVlcMaxSlots"),
             ("VLC_ENTRY_BASE_BYTES", "kVlcEntryBaseBytes")):
    gv, hv = const(glsl, g), const(host, h)
    check(gv is not None and gv == hv, "%s=%s but host %s=%s" % (g, gv, h, hv))
# Header table must fit before the entry base: 16 B + slots * 16 B.
check(16 + const(host, "kVlcMaxSlots") * 16 <= const(host, "kVlcEntryBaseBytes"),
      "slot table overruns the entry base")

# 2. World buffer: the two words exist in C++ and every shader copy that READS them.
world_h = read(SRC, "include", "Backend", "vulkan_world_data.h")
check("uint32_t volumeLightCacheLo;" in world_h and "uint32_t volumeLightCacheHi;" in world_h,
      "vulkan_world_data.h lacks volumeLightCacheLo/Hi")
for name in ("closesthit.rchit", "volume_closesthit.rchit"):
    text = read(SH, name)
    check("volumeLightCacheLo" in text and "volumeLightCacheHi" in text,
          name + " world struct lacks volumeLightCacheLo/Hi")
    check("_aerialPad1" not in text and "_aerialPad2" not in text,
          name + " still declares the retired _aerialPad1/_aerialPad2")

# 3. Instrumentation: GLSL uint/float words == C++ static_assert size / 4.
instr = read(SH, "volume_instrumentation.glsl")
block = re.search(r"VolumeInstrumentationBuffer\s*\{(.*?)\}\s*volumeInstrumentation;", instr, re.S)
words = len(re.findall(r"^\s*(?:uint|float)\s+\w+\s*;", block.group(1), re.M)) if block else -1
instr_h = read(SRC, "include", "Backend", "VulkanVolumeInstrumentation.h")
m = re.search(r"sizeof\(VolumePerformanceStats\)\s*==\s*(\d+)", instr_h)
check(m and words * 4 == int(m.group(1)),
      "instrumentation GLSL has %d words, C++ asserts %s bytes" % (words, m.group(1) if m else "?"))
for field in ("lightCacheHits", "lightCacheMisses", "lightCacheFills", "lightCacheFillSamples"):
    check(field in block.group(1) if block else False, "GLSL instrumentation lacks " + field)
    check(field in instr_h, "C++ instrumentation lacks " + field)

# 4. Consumers pass a light identity; no direction guessing.
bsdf = read(SH, "bsdf_scatter.glsl")
ch = read(SH, "closesthit.rchit")
surf = read(SH, "volume_surface_shadow.glsl")
vch = read(SH, "volume_closesthit.rchit")
check("computeVolumeShadowTransmittance(shadowOrigin, L, tMax, lightIndex)" in bsdf,
      "bsdf_scatter surface shadow does not pass lightIndex")
check("computeVolumeShadowTransmittance(shadowOrigin, wi, tmax, lightIdx)" in ch,
      "closesthit surface shadow does not pass lightIdx")
check("volumeLightCacheRead(uint(vi), vol, lightIndex" in surf,
      "volume_surface_shadow does not read the cache")
check(ch.index('#include "volume_light_cache.glsl"') < ch.index('#include "volume_surface_shadow.glsl"'),
      "closesthit includes the cache after its consumer")
check(vch.count("volumeLightCacheFetch(volIdx, vol,") == 2,
      "volume_closesthit should fetch for scene lights and the sun (2 sites)")

# 5. The fill uses the shading sampler, never a private copy.
check("sampleDensityAcc(vol," in fill, "fill does not use sampleDensityAcc")
check("atomicCompSwap" in fill, "fill lost its claim")
check("(h & ~1u) | 2u" in glsl,
      "tag no longer forces bit1=1/bit0=0 (zeroed memory could match)")

# 7. The A/B switch was removed after acceptance (2026-10-25). A miss already
# runs the old march, so an off switch would be a second path with no purpose.
api_h = read(SRC, "include", "Api", "RtApi.h")
check("Result setVolumeLightCache(int max_cells);" in api_h,
      "setVolumeLightCache grew an argument again (enabled switch back?)")
check("VolumeLightCacheConfig" not in read(SRC, "include", "Backend", "VulkanBackend.h"),
      "VolumeLightCacheConfig (with its enabled flag) is back")

# 6. Host never allocates the cache on an adapter without a TLAS.
check("m_tlasVolumeSlotCount != 0" in host, "cache allocation is not gated on a TLAS")
check("vkCmdFillBuffer(cmd, m_volumeLightCache.buffer" in host, "cache is not zeroed on allocation")

if failures:
    print("FAIL")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("OK - volume light cache header, world words, counters and consumers agree.")
