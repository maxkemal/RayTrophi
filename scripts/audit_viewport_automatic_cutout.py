"""Source wiring checks only; no build and no live GPU verification."""
from pathlib import Path
root = Path(__file__).resolve().parents[1] / 'RayTrophiStudio/source'
def read(p): return (root / p).read_text(encoding='utf-8-sig')
assert 'bool viewport_automatic_cutout = true;' in read('include/globals.h')
project = read('src/Core/ProjectManager.cpp')
assert 'j["viewport_automatic_cutout"] = settings.viewport_automatic_cutout' in project
assert 'j.value("viewport_automatic_cutout", true)' in project
backend = read('src/Backend/VulkanBackend.cpp')
assert backend.count('applyAutomaticViewportCutout(gm, ::render_settings.viewport_automatic_cutout)') == 2
for file in ('material_preview_frag.frag', 'material_preview_opacity.glsl',
             'material_preview_shadow_frag.frag', 'rayfusion_rt_shadow.comp'):
    assert 'MATERIAL_FLAGS_PREVIEW_CUTOUT' in read('shaders/' + file), file
assert 'MATERIAL_FLAGS_PREVIEW_CUTOUT' in read('include/Viewport/RasterMaterialVisibility.h')
for file in ('closesthit.rchit', 'shadow_anyhit.rahit'):
    assert 'MATERIAL_FLAGS_PREVIEW_CUTOUT' not in read('shaders/' + file), file
api = read('src/Api/RtIpcViewportCutout.cpp')
assert 'g_materials_dirty = true;' in api and 'markModified()' in api
assert api.count('rtapi::setViewportAutomaticCutout(') == 2
assert 'rtapi::setViewportAutomaticCutout(' in read('include/UI/ViewportCutoutUI.h')
assert 'is_boolean()' in api and 'pybind11::bool_' in api
print('PASS: viewport-only coverage wiring, default/persistence, material refresh, UI/Python/IPC')
