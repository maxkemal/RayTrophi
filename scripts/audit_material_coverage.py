"""Source-only checks for coverage authoring and exact-prepass shading. No builds."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'RayTrophiStudio/source'
def read(name):
    return (SRC / name).read_text(encoding='utf-8-sig')

core = read('include/MaterialCoverage.h')
assert 'bool alphaCutout = false;' in core
assert 'value == 0.0f || value == 1.0f' in core
api = read('src/Api/RtApi.cpp')
assert all(s in api for s in ['name == "alpha_cutout"', 'MaterialCoverage::validCutoutValue',
                              'material.coverage.setCutout(value.scalar)', 'std::isfinite(value.scalar)'])
for name in ('src/Api/RtPyScene.cpp', 'src/Api/RtIpc.cpp', 'include/UI/MaterialCoverageUI.h'):
    assert 'setMaterialParamByName' in read(name), name
assert 'alpha_cutout|base_color' in read('src/Api/RtIpcMethodDescriptors.cpp')
serialization = read('src/Scene/MaterialManager.cpp')
assert 'matJson["alpha_cutout"] = pbsdf->coverage.alphaCutout' in serialization
assert 'matJson.value("alpha_cutout", false)' in serialization
assert read('src/Scene/PrincipledBSDF.cpp').count('coverage = other.coverage;') == 2
snapshot = read('include/PBRMaterialSnapshot.h')
assert 'gpu.flags |= MATERIAL_FLAG_ALPHA_CUTOUT' in snapshot
assert 'gpu.flags &= ~MATERIAL_FLAG_ALPHA_CUTOUT' in snapshot
assert 'data.flags |= MATERIAL_FLAG_ALPHA_CUTOUT' in snapshot

frag = read('shaders/material_preview_frag.frag')
prepass = read('shaders/material_preview_shadow_frag.frag')
for shader in (frag, prepass):
    assert '#include "material_preview_opacity.glsl"' in shader
    assert 'previewSurfaceOpacity(mat,' in shader
assert '#ifdef PREVIEW_COVERED_SHADING\nlayout(early_fragment_tests) in;\n#endif' in frag
coverage = read('src/Viewport/MaterialPreviewCoverage.cpp')
assert 'depth.depthWriteEnable = VK_FALSE' in coverage
assert 'depth.depthCompareOp = VK_COMPARE_OP_EQUAL' in coverage
assert 'if (depthPrepassActive &&' in coverage
assert 'rasterMeshHasExactCoverage' in coverage
assert 'return m_interactiveViewport.materialPreviewPipeline;' in coverage
policy = read('include/Viewport/RasterMaterialVisibility.h')
exact = policy[policy.index('bool rasterMeshHasExactCoverage'):]
for guard in ('externalMaterials', 'programs.hasProgram(id)', 'm.tile_break_strength',
              'm.transmission_tex', 'm.opacity_tex', 'MATERIAL_FLAGS_PREVIEW_CUTOUT'):
    assert guard in exact, guard
for shader in ('shaders/shadow_anyhit.rahit', 'shaders/rayfusion_rt_shadow.comp',
               'shaders/rayfusion_probe_bounce.glsl'):
    assert 'materialCoverageOpacity' in read(shader), shader
for cuda in ('src/Device/ray_color.cuh', 'src/Device/hitgroup_kernels.cu',
             'src/Device/material_scatter.cuh'):
    assert 'SurfaceCoverage::materialCoverageOpacity' in read(cuda), cuda
vp = read('src/Backend/VulkanViewportBackend.cpp')
assert 'materialPreviewShadingPipeline(rmb, depthPrepassActive)' in vp
assert 'createMaterialPreviewCoveredPipeline(mpPCI, shaderDir)' in vp
assert 'destroyMaterialPreviewCoveredPipeline();' in vp
project = (ROOT / 'RayTrophiStudio/RayTrophiStudio.vcxproj').read_text(encoding='utf-8-sig')
assert 'source\\src\\Viewport\\MaterialPreviewCoverage.cpp' in project
batch = (ROOT / 'RayTrophiStudio/compile_shaders.bat').read_text(encoding='utf-8-sig')
assert '-DPREVIEW_COVERED_SHADING=1' in batch and 'material_preview_covered.spv' in batch
print('PASS: coverage default/copy/persistence; shared UI/Python/IPC service; GPU flags; exact-depth guards/lifecycle; shader/CUDA coverage wiring')
