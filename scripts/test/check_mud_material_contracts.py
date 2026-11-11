"""Non-build Mud preset/authoring wiring check; does not claim live acceptance."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RayTrophiStudio/source"

def read(name):
    data = (SOURCE / name).read_bytes()
    bom = bytes.fromhex("efbbbf")
    assert bom not in (data[3:] if data.startswith(bom) else data), name
    return data.decode("utf-8-sig")


def main():
    presets = read("src/Physics/Fluid/MudMaterial.cpp")
    dense = presets.split("SubstanceProfile makeDenseMud()", 1)[1].split(
        "SubstanceProfile makeMudSlurry()", 1)[0]
    assert "MatterConstitutiveModel::Elastic" in dense
    for field in ("elastic_yield_strain", "elastic_viscosity_pa_s",
                  "elastic_fracture_strain", "elastic_damage_rate"):
        assert "p." + field + " = " in dense
    assert "p.elastic_relaxation_s = 0.0f;" in dense
    assert 'p.name = "Mud Slurry";' in presets
    assert "MatterConstitutiveModel::Fluid" in presets
    builtin = read("src/Physics/MaterialStateField.cpp")
    for call in ("add(Fluid::makeMudSlurry());", "add(Fluid::makeDenseMud());"):
        assert builtin.count(call) == 1
    # Existing library authoring surfaces cover the same material fields.
    library = read("src/Physics/SubstanceLibrary.cpp")
    mapping = read("src/Physics/Fluid/FluidDomainSubstance.cpp")
    for field in ("elastic_yield_strain", "elastic_fracture_strain"):
        assert "RT_FLOAT(" + field in library
        assert "substance." + field in mapping
    widget = read("src/UI/ContactAdhesionControl.h")
    assert "ImGui::SliderFloat(" in widget
    assert "ImGui::InputFloat(" in widget
    assert "maximum_pa = 1.0e6f" in widget
    assert "ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp" in widget
    force_ui = read("src/UI/scene_ui_forcefield.hpp")
    assert "drawContactAdhesionControl(c.contact_adhesion_pa)" in force_ui
    assert 'ImGui::DragFloat("Contact Adhesion' not in force_ui
    # The exact existing field remains available to both bindings.
    for name in ("src/Api/RtIpc.cpp", "src/Api/RtPython.cpp"):
        assert "contact_adhesion_pa" in read(name)
    includes = {e.attrib.get("Include", "").replace("\\", "/") for e in
        ET.parse(ROOT / "RayTrophiStudio/RayTrophiStudio.vcxproj").getroot().iter()}
    assert "source/src/Physics/Fluid/MudMaterial.cpp" in includes
    print("PASS Mud EVP/slurry registration, existing material API and Pa adhesion controls")


if __name__ == "__main__":
    main()
