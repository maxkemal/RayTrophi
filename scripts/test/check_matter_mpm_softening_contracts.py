"""Non-build heat -> MPM softening -> CPU/GPU stress wiring audit."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RayTrophiStudio/source"


def read(path):
    return (SOURCE / path).read_text(encoding="utf-8-sig")


def main():
    step = read("src/Physics/Fluid/FluidDomainStep.inl")
    update = step.index("Fluid::updateMatterMpmSoftening(state.particles, fluid_params);")
    assert step.index("Fluid::diffuseParticleTemperature(") < update
    assert update < step.index("ok = runMatterGpuStep(")
    assert "Fluid::Granular::updateSoftening(state.particles, softening_params)" not in step
    core = read("src/Physics/Fluid/MatterMpmSoftening.cpp")
    assert "MatterConstitutiveModel::Elastic" in core
    assert "MatterTransportOwner::Mpm" in core
    assert "substanceBirthModel(tag, params, kelvin)" in core
    assert "Granular::updateSoftening(particles, softening, &mask)" in core
    assert "if (params.granular_enabled)" not in core
    cpu = read("include/Fluid/GranularConstitutive.h")
    gpu = read("shaders/sim_fluid_granular_stress_update.glsl")
    assert "elasticThermalScale(soft, pc.elastic_thermal_viscosity_power)" in cpu
    assert "elasticThermalScale(soft, pc.elastic_thermal_viscosity_power)" in gpu
    assert "pc.young_modulus * soft" in cpu
    assert "Granular::uploadState(*compute, particles" in read(
        "src/Physics/Fluid/MatterGpuStep.inl")
    registered = {node.attrib.get("Include", "").replace("\\", "/") for node in
                  ET.parse(ROOT / "RayTrophiStudio/RayTrophiStudio.vcxproj").getroot().iter()}
    assert "source/src/Physics/Fluid/MatterMpmSoftening.cpp" in registered
    assert "source/include/Fluid/MatterMpmSoftening.h" in registered
    print("PASS elastic/mixed MPM softening, conduction order, shared curve and stress upload")


if __name__ == "__main__":
    main()
