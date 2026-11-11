"""Non-build catalogue publication/API integration checks."""
import ast
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RayTrophiStudio/source"


def read(path):
    return (SOURCE / path).read_text(encoding="utf-8-sig")


def main():
    builtin = read("src/Physics/MaterialStateField.cpp")
    assert builtin.count("Fluid::configureBuiltinPresetPhysics(out);") == 1
    assert "appendFoundationSubstances" not in builtin
    assert "appendFoundationSubstances" not in read("src/Physics/SubstanceCatalogue.cpp")
    named = re.findall(r'p\.name\s*=\s*"([^"]+)"', builtin)
    assert len(named) + builtin.count("add(Fluid::makeMud") + builtin.count(
        "add(Fluid::makeDenseMud") == 26
    catalogue = read("src/Physics/SubstanceCatalogue.cpp")
    assert "Generic " not in catalogue
    entries = catalogue[catalogue.index("constexpr Foundation foundations[]"):
                        catalogue.index("std::pair<std::string")]
    names = re.findall(r'\{"([^"]+)"', entries)
    assert len(names) == len(set(names)) == 12
    assert set(names) <= set(named + ["Mud", "Mud Slurry"])
    assert "entry.example" not in catalogue
    core = read("src/Physics/Fluid/BuiltinPresetPhysics.cpp")
    assert "water.liquid_kinematic_viscosity" in core
    assert "liquid_cold_viscosity = std::max" in core
    assert "liquid_cold_viscosity < p.liquid_kinematic_viscosity" in read(
        "src/Physics/SubstanceLibrary.cpp")
    assert "model_notes" in read("src/UI/scene_ui_substance_editor.hpp")
    assert "substanceCatalogueInfo" in read("src/Api/RtApiSubstance.cpp")
    overlay = json.loads((ROOT / "scripts/ipc_descriptor_overlay.json").read_text(
        encoding="utf-8-sig"))
    assert "model_notes" in overlay["substance.list"]["notes"]
    registered = {node.attrib.get("Include", "").replace("\\", "/") for node in
                  ET.parse(ROOT / "RayTrophiStudio/RayTrophiStudio.vcxproj").getroot().iter()}
    for path in ("source/include/Fluid/BuiltinPresetPhysics.h",
                 "source/src/Physics/Fluid/BuiltinPresetPhysics.cpp"):
        assert path in registered
    for name in ("rt_test_substance_catalogue_ipc.py", "rt_test_substance_profiles_ipc.py"):
        ast.parse((ROOT / "scripts/test" / name).read_text(encoding="utf-8-sig"))
    print("PASS 26 unique presets, 12 real starting profiles, physical finalizer, "
          "shared validation, UI/API metadata and project registration")


if __name__ == "__main__":
    main()
