"""Non-build audit of shared flat-only domain creation wiring."""
from pathlib import Path
import ast
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "RayTrophiStudio/source"
def read(path):
    return (SRC / path).read_text(encoding="utf-8-sig")

core = read("src/Api/RtApiDomainCreation.cpp")
assert "dynamic_pointer_cast<TriangleMesh>" in core
assert 'get_attribute_data<Vec3>("P_orig")' in core
assert 'get_attribute_data<Vec3>("P")' in core
assert "getFinal()" in core
assert "dynamic_pointer_cast<Triangle>" not in core
assert ".selected.object" not in core
assert "resolveObjectBoundsForSimulation" not in core
assert core.index("if (!source.empty())") < core.index("addSimulationGridDomain(descriptor)")
ui = read("src/UI/DomainCreationUI.cpp")
assert 'create("")' in ui and "create(source)" in ui
assert "createPlacedMatterDomain(" in ui
panel = read("src/UI/scene_ui_simulation_domains.cpp")
assert "drawDomainCreationActions(resetSimulationNow, selected_domain_index)" in panel
creation = panel[:panel.index("// --- Row 2: Cleanup Actions Toolbar ---")]
assert "selected.object->getNodeName()" not in creation
assert "Vec3 center(0.0f, 1.0f, 0.0f)" not in panel
assert '"fluid.create_placed_domain"' in read("src/Api/RtIpcMatterSeedSources.cpp")
assert '"create_placed_domain"' in read("src/Api/RtPythonMatterSeedSources.cpp")
overlay = json.loads((ROOT / "scripts/ipc_descriptor_overlay.json").read_text())
assert "fluid.create_placed_domain" in overlay
entries = {node.attrib.get("Include", "").replace("\\", "/") for node in
           ET.parse(ROOT / "RayTrophiStudio/RayTrophiStudio.vcxproj").getroot().iter()}
for path in ("source/src/Api/RtApiDomainCreation.cpp", "source/src/UI/DomainCreationUI.cpp"):
    assert path in entries
ast.parse((ROOT / "scripts/test/rt_test_domain_placement_ipc.py").read_text())
print("PASS shared domain placement, flat SoA-only bounds, UI/script/IPC and project wiring")
