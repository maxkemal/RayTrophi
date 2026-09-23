#!/usr/bin/env python3
"""Static and virtual-profile checks for independent Vulkan feature tiers."""

from dataclasses import dataclass
from pathlib import Path
import sys
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from rt_repo_root import repo_root as _repo_root
ROOT = _repo_root()
HEADER = (
    ROOT / "RayTrophiStudio/source/include/Backend/VulkanBackend.h"
).read_text(encoding="utf-8")
DEVICE = (
    ROOT / "RayTrophiStudio/source/src/Backend/VulkanBackend.cpp"
).read_text(encoding="utf-8")
SDF_PASS = (
    ROOT / "RayTrophiStudio/source/src/Viewport/MaterialPreviewSdfSurface.cpp"
).read_text(encoding="utf-8")
VOLUME_PASS = (
    ROOT / "RayTrophiStudio/source/src/Viewport/MaterialPreviewVolume.cpp"
).read_text(encoding="utf-8")
SCENE_AS = (
    ROOT / "RayTrophiStudio/source/src/Viewport/RayFusionSceneAS.cpp"
).read_text(encoding="utf-8")
PROBE_TRACE = (
    ROOT / "RayTrophiStudio/source/src/Viewport/RayFusionProbeTrace.cpp"
).read_text(encoding="utf-8")
BOUNCE = (
    ROOT / "RayTrophiStudio/source/src/Viewport/RayFusionBounce.cpp"
).read_text(encoding="utf-8")
PROBE_FIELD = (
    ROOT / "RayTrophiStudio/source/src/Viewport/MaterialPreviewProbeField.cpp"
).read_text(encoding="utf-8")


@dataclass(frozen=True)
class Profile:
    bda: bool
    shader_int64: bool
    acceleration_structure: bool
    rt_pipeline: bool
    ray_query: bool

    @property
    def native_volume(self) -> bool:
        return self.bda and self.shader_int64

    @property
    def hardware_rt(self) -> bool:
        return self.bda and self.acceleration_structure and self.rt_pipeline

    @property
    def ray_query_effects(self) -> bool:
        return self.hardware_rt and self.ray_query


profiles = {
    "strong Vulkan compute without RT": (
        Profile(True, True, False, False, False),
        (True, False, False),
    ),
    "RT pipeline without ray query": (
        Profile(True, True, True, True, False),
        (True, True, False),
    ),
    "full RT device": (
        Profile(True, True, True, True, True),
        (True, True, True),
    ),
    "legacy device without BDA": (
        Profile(False, True, False, False, False),
        (False, False, False),
    ),
}

checks = {
    "native volume predicate is BDA plus shaderInt64": (
        "return supportsBufferDeviceAddress && supportsShaderInt64;" in HEADER
    ),
    "Vulkan 1.2 core BDA is accepted": (
        "deviceProperties.apiVersion >= VK_API_VERSION_1_2" in DEVICE
        and "hasBDAExtension || hasCoreBDA" in DEVICE
    ),
    "BDA is latched independently from RT mode": (
        "bool enabledBDA = (result == VK_SUCCESS) && canUseBDA;" in DEVICE
        and "m_capabilities.supportsBufferDeviceAddress = enabledBDA;" in DEVICE
    ),
    "shaderInt64 is queried, enabled and latched": (
        "supportedFeatures.features.shaderInt64 == VK_TRUE" in DEVICE
        and "features2.features.shaderInt64" in DEVICE
        and "m_capabilities.supportsShaderInt64 = enabledShaderInt64;" in DEVICE
    ),
    "core and extension BDA entry points are accepted": (
        '"vkGetBufferDeviceAddressKHR"' in DEVICE
        and '"vkGetBufferDeviceAddress"' in DEVICE
    ),
    "ray query has a real feature gate": (
        "VkPhysicalDeviceRayQueryFeaturesKHR supportedRayQueryFeatures" in DEVICE
        and "supportedRayQueryFeatures.rayQuery == VK_TRUE" in DEVICE
        and "m_capabilities.supportsRayQuery = enabledRayQuery;" in DEVICE
    ),
    "native SDF uses the volume capability tier": (
        "supportsNativeVolumeRaymarch()" in SDF_PASS
    ),
    "native gas uses the volume capability tier": (
        "supportsNativeVolumeRaymarch()" in VOLUME_PASS
    ),
    "scene acceleration remains hardware RT only": (
        "!m_device->hasHardwareRT()" in SCENE_AS
        and "supportsRayQuery" in SCENE_AS
    ),
    "RayFusion ray passes require the ray-query tier": (
        "supportsRayQuery" in PROBE_TRACE
        and "supportsRayQuery" in BOUNCE
    ),
    "RayFusion retains its RT-free probe producer": (
        "if (!tracedThisBatch)" in PROBE_FIELD
        and 'state->activeProducer = "sky_bake"' in PROBE_FIELD
    ),
    "runtime test can disable RT without disabling BDA": (
        'std::getenv("RAYTROPHI_DISABLE_HARDWARE_RT")' in DEVICE
        and "m_device->initialize(preferHardwareRT, validation)" in DEVICE
    ),
}

for name, (profile, expected) in profiles.items():
    actual = (
        profile.native_volume,
        profile.hardware_rt,
        profile.ray_query_effects,
    )
    checks[f"virtual profile: {name}"] = actual == expected

failed = [name for name, passed in checks.items() if not passed]
for name, passed in checks.items():
    print(f"[{'OK' if passed else 'FAIL'}] {name}")

if failed:
    print("Vulkan feature-tier audit failed: " + ", ".join(failed), file=sys.stderr)
    sys.exit(1)

print("Vulkan feature-tier static and virtual-profile audit passed.")
