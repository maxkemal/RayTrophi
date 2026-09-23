#!/usr/bin/env python3
"""Static contract checks for the raster SurfaceSDF bridge (no build needed)."""

from pathlib import Path
import re
import sys
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from rt_repo_root import repo_root as _repo_root
ROOT = _repo_root()
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
rt_shadow = (
    ROOT / "RayTrophiStudio/source/shaders/rayfusion_rt_shadow.comp"
).read_text(encoding="utf-8")
sdf_shadow_query = (
    ROOT / "RayTrophiStudio/source/shaders/surface_sdf_shadow_query.glsl"
).read_text(encoding="utf-8")
rt_shadow_host = (
    ROOT / "RayTrophiStudio/source/src/Viewport/MaterialPreviewRtShadow.cpp"
).read_text(encoding="utf-8")
volume_shader = (
    ROOT / "RayTrophiStudio/source/shaders/material_preview_volume.frag"
).read_text(encoding="utf-8")
volume_pass = (
    ROOT / "RayTrophiStudio/source/src/Viewport/MaterialPreviewVolume.cpp"
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
    "native raster modes share SurfaceSDF": (
        "g_native_surface_sdf_viewport_available" in particle_bridge
        and "ViewportMode::Solid" in viewport_backend
        and "ViewportMode::Matcap" in viewport_backend
    ),
    "native SDF live update gate covers raster modes": (
        "isInteractiveViewportShadingMode(" in main
        and "g_native_surface_sdf_viewport_available" in main
        and re.search(
            r"g_native_surface_sdf_viewport_available\s*&&\s*"
            r"g_gas_volumes_dirty\)\s*\{\s*start_render = true;",
            main,
        ) is not None
    ),
    "grid restores solid pipeline after SDF": re.search(
        r"recordMaterialPreviewSdfSurfacePass\([\s\S]+"
        r"Grid drawing must restore the solid[\s\S]+"
        r"vkCmdBindPipeline\([\s\S]+solidPipeline",
        viewport_backend,
    ) is not None,
    "RayFusion shadows query native SurfaceSDF": (
        "binding = 6" in sdf_shadow_query
        and "RF_VOLUME_STRIDE_WORDS = 156u" in sdf_shadow_query
        and "rfVolAddress(vi, 232u)" in sdf_shadow_query
        and "rfVolWord(vi, 428u)" in sdf_shadow_query
        and "rfVolFloat(vi, 556u)" in sdf_shadow_query
        and "rfSurfaceSdfOccludes(" in rt_shadow
        and "pc.volumeMeta.x" in rt_shadow
        and "m_device->m_volumeBuffer.buffer" in rt_shadow_host
        and "m_device->m_volumeCount" in rt_shadow_host
    ),
    "Solid and Matcap share the native gas volume": (
        "VkPipeline previewPipeline" in volume_pass
        and "VkPipeline solidPipeline" in volume_pass
        and "ViewportMode::Solid" in volume_pass
        and "ViewportMode::Matcap" in volume_pass
        and "push.cameraPos[3] = previewMode ? 0.0f : 1.0f" in volume_pass
        and "depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL" in volume_pass
        and "m_interactiveViewport.renderPass, 1u, true" in volume_pass
        and "bool workbench=pc.cameraPos.w>0.5" in volume_shader
        and "if(workbench)" in volume_shader
        and "gl_FragDepth=clamp(firstDepth,0.0,1.0)" in volume_shader
        and "recordMaterialPreviewVolumePass(" in viewport_backend
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
