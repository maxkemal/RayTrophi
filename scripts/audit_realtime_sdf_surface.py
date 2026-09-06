#!/usr/bin/env python3
"""Static contract checks for the raster SurfaceSDF bridge (no build needed)."""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
shader = (ROOT / "RayTrophiStudio/source/shaders/material_preview_sdf_surface.frag").read_text(
    encoding="utf-8"
)
header = (ROOT / "RayTrophiStudio/source/include/Backend/vulkan_volume_types.h").read_text(
    encoding="utf-8"
)
main = (ROOT / "RayTrophiStudio/source/src/Core/Main.cpp").read_text(encoding="utf-8")
particle_bridge = (
    ROOT / "RayTrophiStudio/source/src/Physics/ParticleRenderBridge.cpp"
).read_text(encoding="utf-8")
volume_backend = (
    ROOT / "RayTrophiStudio/source/src/Backend/VulkanBackend_Volumes.cpp"
).read_text(encoding="utf-8")
transmission_pass = (
    ROOT / "RayTrophiStudio/source/src/Viewport/MaterialPreviewTransmission.cpp"
).read_text(encoding="utf-8")
viewport_backend = (
    ROOT / "RayTrophiStudio/source/src/Backend/VulkanViewportBackend.cpp"
).read_text(encoding="utf-8")

checks = {
    "624-byte host ABI": "static_assert(sizeof(VkVolumeInstance) == 624" in header,
    "156-word raw stride": re.search(r"VOLUME_STRIDE_WORDS\s*=\s*156u", shader) is not None,
    "volume descriptor binding": "binding = 20" in shader,
    "transform offset": "transformPoint(vi, 184u" in shader,
    "bounds offsets": "volVec3(vi, 48u)" in shader and "volVec3(vi, 60u)" in shader,
    "grid address offset": "volAddress(vi, 232u)" in shader,
    "source type offset": "volWord(vi, 428u)" in shader,
    "surface IOR offset": "volFloat(nearestVolume, 464u)" in shader,
    "surface roughness offset": "volFloat(nearestVolume, 468u)" in shader,
    "material index offset": "volFloat(nearestVolume, 556u)" in shader,
    "depth output": "gl_FragDepth = clamp(depth, 0.0, 1.0)" in shader,
    "bounded volume loop": "min(pc.materialMeta.w, 16u)" in shader,
    "sub-voxel stable surface march": (
        "voxel * 0.45" in shader and "quality == 2u ? 512 : 1024" in shader
    ),
    "RT boundary hysteresis": (
        "isoHysteresis = 0.05" in shader
        and "d0 = min(d0, iso - 1e-4)" in shader
    ),
    "depth equality tolerance": "VK_COMPARE_OP_LESS_OR_EQUAL" in (
        ROOT / "RayTrophiStudio/source/src/Viewport/MaterialPreviewSdfSurface.cpp"
    ).read_text(encoding="utf-8"),
    "dielectric material extension": (
        "materialsExt[mi]" in shader
        and "previewBeerExtinction(interiorTint)" in shader
        and "mat.specular_tex" in shader
    ),
    "opaque scene refraction": (
        "binding = 17" in shader
        and "binding = 18" in shader
        and "projectOpaqueRefraction" in shader
        and "sceneThrough * transmissionWeight" in shader
    ),
    "snapshot-before-SDF ordering": (
        re.search(
            r"vkCmdCopyImage[\s\S]+recordMaterialPreviewSdfSurfacePass\("
            r"[\s\S]+width, height, true\)",
            transmission_pass,
        )
        is not None
        and "useMaterialPreview && !m_materialPreviewTransmission"
        in viewport_backend
    ),
    "live preview volume consumption": (
        "nativeSdfRasterActive" in main
        and "scene, activeViewportBackend, &wd" in main
    ),
    "no duplicate preview sphere proxy": (
        "g_solid_viewport_active && !g_material_preview_viewport_active"
        in particle_bridge
    ),
    "empty packet invalidates stale frame": re.search(
        r"m_publishedVolumeKeyOrder\.clear\(\);\s*"
        r"(?://[^\n]*\n\s*)*resetAccumulation\(\);\s*return;",
        volume_backend,
    )
    is not None,
}

failed = [name for name, ok in checks.items() if not ok]
for name, ok in checks.items():
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
if failed:
    print("Realtime SurfaceSDF audit failed: " + ", ".join(failed), file=sys.stderr)
    sys.exit(1)
print("Realtime SurfaceSDF static contract audit passed.")
