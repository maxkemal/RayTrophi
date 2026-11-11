"""Non-build persistence/first-sync wiring audit."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RayTrophiStudio/source"


def read(path):
    return (SOURCE / path).read_text(encoding="utf-8-sig")


def main():
    project = read("src/Core/ProjectManager.cpp")
    scene = read("src/Utils/SceneSerializer.cpp")
    assert 'domain.fluid_seed_box_initialized = f.contains("seed_min") && f.contains("seed_max")' in project
    assert 'd.fluid_seed_box_initialized = j.contains("fluid_seed_min") && j.contains("fluid_seed_max")' in scene
    helper = read("include/Fluid/FluidSeedBox.h")
    assert helper.index("if (domain.fluid_seed_box_initialized)") < helper.index(
        "domain.fluid_seed_min =")
    sync = read("src/Physics/Fluid/MatterDomainSynchronization.inl")
    assert "Fluid::initializeFluidSeedBox(domain, mn, mx);" in sync
    assert "domain.fluid_seed_min = seed_center" not in sync
    author = read("src/Physics/Fluid/FluidDomainParticleAuthoring.cpp")
    assert "domain.fluid_seed_box_initialized = true;" in author
    assert '#include "Fluid/FluidSeedBox.h"' in read("src/Physics/ParticleSimulation.cpp")
    registered = {node.attrib.get("Include", "").replace("\\", "/") for node in
                  ET.parse(ROOT / "RayTrophiStudio/RayTrophiStudio.vcxproj").getroot().iter()}
    assert "source/include/Fluid/FluidSeedBox.h" in registered
    print("PASS seed-box persistence: both loaders, API authorship, guarded first sync")


if __name__ == "__main__":
    main()
