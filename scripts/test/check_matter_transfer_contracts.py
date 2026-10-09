"""Non-build audit of the C4a identity/transfer integration."""
import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    repo = Path(__file__).resolve().parents[2]
    project = repo / "RayTrophiStudio"
    source = project / "source"

    def read(path):
        return (source / path).read_text(encoding="utf-8-sig")

    particles = read("include/Fluid/FluidParticles.h")
    for text in ["particle_id.push_back(identity)", "particle_id[i] = particle_id[last]",
                 "fn(a.particle_id, b.particle_id)", "particle_id.resize(n)",
                 "particle_id.clear()", "particle_id.reserve(n)"]:
        assert text in particles, text
    clear = particles.split("void clear()", 1)[1].split("size_t size()", 1)[0]
    assert "next_particle_id =" not in clear
    memory = read("include/SimFrameCacheMemory.h")
    assert "allocationBytes(particles.particle_id)" in memory
    assert "allocationBytes(particles.rest_mass_kg)" in memory
    cache = read("src/Physics/SimCache.cpp")
    assert "writePod(os, d.particles.next_particle_id)" in cache
    assert "readPod(is, d.particles.next_particle_id)" in cache
    assert "sorted.back() >= next_identity" in cache
    assert "sorted.front() == 0" in cache
    assert "std::adjacent_find(sorted.begin(), sorted.end())" in cache
    assert "readParticleIdentities(is, d.particles.particle_id," in cache
    assert "kVersion = 12u" in read("include/SimCache.h")
    frame_cache = read("include/scene_data.h")
    assert "c.meta = st;" in frame_cache
    assert "particles.particle_id.clear" not in frame_cache
    assert "collapseIfUniform(pt.particle_id" not in frame_cache

    cpp_paths = [
        r"source\src\Physics\Fluid\MatterTransfer.cpp",
        r"source\src\Physics\Fluid\MatterModelService.cpp",
        r"source\src\UI\MatterModelControls.cpp",
        r"source\src\Api\RtApiMatterModels.cpp",
        r"source\src\Api\RtIpcMatterModels.cpp",
        r"source\src\Api\RtPythonMatterModels.cpp",
    ]
    for filename in ["RayTrophiStudio.vcxproj", "RayTrophiStudio.vcxproj.filters"]:
        tree = ET.parse(project / filename)
        compiled = [node.attrib["Include"] for node in tree.iter()
                    if node.tag.endswith("ClCompile") and "Include" in node.attrib]
        for path in cpp_paths:
            assert compiled.count(path) == 1, (filename, path)
    for path in cpp_paths:
        text = (project / Path(path.replace("\\", "/"))).read_text(encoding="utf-8")
        assert len(text.splitlines()) < 2000
        assert all(len(line) <= 110 for line in text.splitlines()), path
    core = read("src/Physics/Fluid/MatterTransfer.cpp")
    assert "model_lane > 1 || !include_transfer" in core
    assert "auto cells = frame.cells;" in core
    assert "fluid.momentum = add(fluid.momentum, scale(impulse, -1.0));" in core
    assert "granular.momentum = add(granular.momentum, impulse);" in core
    assert "inspectMatterModels(" in read("src/UI/MatterModelControls.cpp")
    assert "inspectMatterModels(" in read("src/Api/RtApiMatterModels.cpp")
    assert "getMatterModels(" in read("src/Api/RtPythonMatterModels.cpp")
    assert "getMatterModels(" in read("src/Api/RtIpcMatterModels.cpp")
    assert '"fluid.matter_models") return Read' in read("src/Api/RtIpcSecurity.cpp")
    assert "registerMatterModelBindings(fluid)" in read("src/Api/RtPython.cpp")
    assert "dispatchMatterModelIpc(" in read("src/Api/RtIpc.cpp")
    descriptor = json.loads((repo / "scripts/ipc_descriptor_overlay.json").read_text(
        encoding="utf-8"))
    assert "fluid.matter_models" in descriptor
    assert '"fluid.matter_models", "fluid"' in read("src/Api/RtIpcMethodDescriptors.cpp")
    ast.parse((repo / "scripts/test/rt_test_matter_models_ipc.py").read_text())
    print("PASS: identities, cache v11/v12, separate transfer/contact, UI/API/IPC, project XML")


if __name__ == "__main__":
    main()
