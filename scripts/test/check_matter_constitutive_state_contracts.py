"""Non-build audit of the canonical birth/fusion selection integration."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RayTrophiStudio/source"


def read(path):
    return (SOURCE / path).read_text(encoding="utf-8-sig")


def main():
    state = read("src/Physics/Fluid/MatterSubstanceState.cpp")
    birth = state[state.index("MatterConstitutiveModel substanceBirthModel("):
                  state.index("float substanceAmbientBirthKelvin(")]
    refresh = state[state.index("void refreshMatterConstitutiveModels("):
                    state.index("bool isMatterObstacle(")]
    assert "resolveMatterConstitutiveState(" in birth
    assert "resolveMatterConstitutiveState(" in refresh
    assert "solid[index] >= kFusionSolidThreshold" not in refresh
    assert "return hasMatterSolidSkeleton(profile);" in state
    header = read("include/Fluid/MatterConstitutiveState.h")
    assert "kFusionSolidThreshold = 0.5f" in header
    assert "constexpr float kFusionSolidThreshold" not in read(
        "include/Fluid/MatterSubstanceState.h")
    core = read("src/Physics/Fluid/MatterConstitutiveState.cpp")
    assert "supported_pinned_carrier" in core and "not_defined" in core
    report = read("src/Physics/Fluid/MatterMaterialResolution.cpp")
    ui = read("src/UI/MatterMaterialResolutionUI.cpp")
    for key in ("phase_mechanics", "phase_counts", "selection_basis_counts",
                "model_mismatch_particles"):
        assert '"' + key + '"' in report
        assert '"' + key + '"' in ui
    registered = {node.attrib.get("Include", "").replace("\\", "/") for node in
                  ET.parse(ROOT / "RayTrophiStudio/RayTrophiStudio.vcxproj").getroot().iter()}
    for name in ("source/include/Fluid/MatterConstitutiveState.h",
                 "source/src/Physics/Fluid/MatterConstitutiveState.cpp"):
        assert name in registered
    print("PASS canonical birth/fusion model selection and UI/API report integration")


if __name__ == "__main__":
    main()
