"""Non-build audit: shared authoring, CPU/GPU propagation and scalar ABI."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "RayTrophiStudio/source"


def read(name):
    return (S / name).read_text(encoding="utf-8-sig")


def main():
    fields = ["elastic_thermal_viscosity_power", "elastic_thermal_yield_power",
              "elastic_thermal_relaxation_power"]
    paths = ["include/MaterialStateField.h", "include/Fluid/APICFluidSolver.h",
             "include/Fluid/GranularConstitutive.h", "include/Fluid/GranularGpuDispatch.h",
             "src/Physics/SubstanceLibrary.cpp", "src/Physics/Fluid/FluidDomainSubstance.cpp",
             "src/Physics/Fluid/APICFluidStep.inl", "src/Physics/Fluid/FluidGpuTransferStages.inl",
             "shaders/sim_fluid_granular_stress_update.glsl"]
    for path in paths:
        source = read(path)
        for field in fields:
            assert field in source, (field, path)
    constants = read("include/Fluid/GranularGpuDispatch.h")
    assert "sizeof(StressUpdateConstants) == 124" in constants
    assert "alignas(16) StressUpdateConstants" not in constants
    assert '"sim_matter_stress_update.spv", 18, 128' in read(
        "src/Device/SimulationComputeVulkan.cpp")
    assert '"sim_fluid_granular_stress_update.spv", 15, 124' in read(
        "src/Device/SimulationComputeVulkan.cpp")
    for path in ["include/Fluid/GranularConstitutive.h",
                 "shaders/sim_fluid_granular_stress_update.glsl"]:
        source = read(path)
        for field in fields:
            assert "elasticThermalScale(soft, pc." + field + ")" in source
    for name in ["rt_test_substance_profiles_ipc.py", "rt_test_material_resolution_ipc.py"]:
        ast.parse((ROOT / "scripts/test" / name).read_text(encoding="utf-8-sig"))
    print("PASS elastic thermal authoring/CPU/GPU wiring and 124/128-byte stress ABI")


if __name__ == "__main__":
    main()
