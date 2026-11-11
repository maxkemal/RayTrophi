"""Non-build checks for shared report integration and API documentation."""

import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RayTrophiStudio/source"


def read(path):
    return (SOURCE / path).read_text(encoding="utf-8-sig")


def main():
    api = read("src/Api/RtApiMatterModels.cpp")
    ui = read("src/UI/MatterMaterialResolutionUI.cpp")
    solver = read("src/Physics/Fluid/FluidDomainSubstance.cpp")
    core = read("src/Physics/Fluid/MatterMaterialResolution.cpp")
    assert '"material_resolution", material_resolution' in api
    for consumer in (api, ui):
        assert "inspectMatterMaterialResolution(" in consumer
        assert "kUnresolvedViscosityDiffusion" not in consumer
    assert "resolveSubstanceViscosity(*s, voxel_size," in solver
    assert "nu * kViscosityReferenceDt" not in solver
    assert "collectMatterStaticTags(domain, static_tags)" in core
    assert "substanceTransportOwner(tag, model, params)" in core
    assert "inspectMatterDomainOwners(particles, domain)" in core
    assert 'fluid.def("matter_models"' in read("src/Api/RtPythonMatterModels.cpp")
    assert 'method == "fluid.matter_models"' in read("src/Api/RtIpcMatterModels.cpp")
    assert "drawMatterMaterialResolution(domain, state);" in read(
        "src/UI/MatterModelControls.cpp")

    registered = {node.attrib.get("Include", "").replace("\\", "/")
                  for node in ET.parse(ROOT / "RayTrophiStudio/RayTrophiStudio.vcxproj")
                  .getroot().iter()}
    new_sources = (
        "include/Fluid/MatterMaterialResolution.h",
        "src/Physics/Fluid/MatterMaterialResolution.cpp",
        "src/UI/MatterMaterialResolutionUI.h",
        "src/UI/MatterMaterialResolutionUI.cpp",
    )
    for path in new_sources:
        assert "source/" + path in registered, path
        data = (SOURCE / path).read_bytes()
        assert b"\xef\xbb\xbf" not in data[3:], path
        assert len(data.splitlines()) < 2000, path

    overlay = json.loads((ROOT / "scripts/ipc_descriptor_overlay.json").read_text(
        encoding="utf-8-sig"))
    assert "material_resolution" in overlay["fluid.matter_models"]["notes"]
    descriptor = read("src/Api/RtIpcMethodDescriptors.cpp")
    start = descriptor.index("static const MethodDescriptor desc_fluid_matter_models")
    assert "material_resolution" in descriptor[start:descriptor.index("};", start)]
    ast.parse((ROOT / "scripts/test/rt_test_material_resolution_ipc.py").read_text())
    print("PASS material resolution: shared policy, UI/API/IPC/script integration, "
          "project registration, descriptors and Python syntax")


if __name__ == "__main__":
    main()
