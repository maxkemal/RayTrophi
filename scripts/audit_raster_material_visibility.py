"""Non-build wiring audit. Behavioral C++ test is delivered separately."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "RayTrophiStudio/source"

def read(path):
    return (SRC / path).read_text(encoding="utf-8-sig")

def code(text):
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)

sites = 0
for path in (SRC / "src").rglob("*.cpp"):
    text = code(path.read_text(encoding="utf-8-sig", errors="replace"))
    for match in re.finditer(r"(\w+)\.matIdsHashValid\s*=\s*false\s*;", text):
        owner = match[1]
        before = text[max(0, match.start() - 120):match.start()]
        assert f"{owner}.materialUsage.invalidate();" in before, (path, owner)
        sites += 1
    for match in re.finditer(r"(\w+)\.cpuMatIds\s*=(?!=)", text):
        owner = match[1]
        after = text[match.end():match.end() + 240]
        assert f"{owner}.materialUsage.invalidate();" in after, (path, owner)
assert sites == 11, f"Review material-ID writer inventory: {sites} sites"

frag = code(read("shaders/material_preview_frag.frag"))
coverage = frag.index("float opacity = previewSurfaceOpacity")
assert frag.index("uv = pd_tileBreak") < coverage
assert coverage < frag.index("if ((mat.flags & FLAG_TERRAIN)")
assert coverage < frag.index("vec4 texAlbedo = texture")
assert "if (graphOffset == MATPROG_NONE)" in frag[coverage:coverage + 230]
assert "if (opacity == 0.0) discard;" in frag[coverage:coverage + 230]
assert frag.count("float opacity =") == 1
assert "(graphWritten & MP_SLOT_OPACITY)" in frag
assert "imageAtomicMax(previewTransmissionBackDepth" in frag
assert "#ifdef PREVIEW_COVERED_SHADING\nlayout(early_fragment_tests) in;\n#endif" in frag
assert '#include "raster_material_policy.h"' in frag
assert '#include "material_preview_opacity.glsl"' in frag

replay = code(read("src/Viewport/MaterialPreviewTransmission.cpp"))
assert replay.count("materialPreviewMeshNeedsTransmission(") == 3
assert "m_device->m_volumeCount > 0u" in replay
loop = replay[replay.index("for (const auto& [key, mesh]"):]
assert loop.index("materialPreviewMeshNeedsTransmission(mesh)") < loop.index("vkCmdBindVertexBuffers")
for token in ("compactedInstanceBuffer()", "mesh.cullOutBase", "vkCmdDrawIndexedIndirect",
              "vkCmdDrawIndirect", "recordMaterialPreviewVolumePass"):
    assert token in loop, token
backend = code(read("src/Backend/VulkanBackend.cpp"))
assert "m_rasterMaterialPrograms.update(words);" in backend
print(f"PASS: {sites} ID invalidation sites; early alpha/graph guards; shared policy; culled replay/volume wiring")
