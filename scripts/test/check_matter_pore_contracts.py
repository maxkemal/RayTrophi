"""Non-build C5 lifecycle, publication and shared authoring source audit."""
import ast
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / "RayTrophiStudio/source"

    def read(name):
        return (source / name).read_text(encoding="utf-8-sig")

    particles = read("include/Fluid/FluidParticles.h")
    cache = read("src/Physics/SimCache.cpp")
    memory = read("include/SimFrameCacheMemory.h")
    for field in ["pore_water_mass_kg", "pore_capacity_kg", "pore_porosity",
                  "pore_water_energy_j"]:
        for operation in ["clear()", "reserve(n)", "resize(n)"]:
            assert f"{field}.{operation}" in particles, (field, operation)
        assert f"copy({field}, other.{field})" in particles, field
        assert f"{field}[i] = {field}[last]" in particles, field
        assert f"allocationBytes(particles.{field})" in memory, field
        assert cache.count(field) >= 3, field
    assert "kVersion = 12u" in read("include/SimCache.h")
    assert "version == 11u || version == kVersion" in read("include/SimCache.h")
    core = read("src/Physics/Fluid/MatterPoreExchange.cpp")
    assert core.index("working > budget_bytes") < core.index("auto candidate = particles")
    assert core.index("C5 conservation gate rejected") < core.index("particles = std::move(candidate)")
    assert "compute.beginTransferBatch()" in core and "compute.endTransferBatch()" in core
    assert "MatterExchangeKind::Absorption" in core and "MatterExchangeKind::Drainage" in core
    shader = read("shaders/sim_matter_pores.comp")
    assert "birth_slots[receiver] = 2" in core
    assert "std::vector<uint32_t> drainage_receiver(cells, outside)" in core
    assert "++report.drainage_refills" in core
    assert core.index("// GPU records are exchange results") < core.index("candidate.emit(birth")
    assert "release_scale * pore[i]" in shader
    assert "output_room - rest[i] * fraction[i] * (1.0 - debit)" in shader
    assert '"drainage_refills"' in read("src/Api/RtApiMatterModels.cpp")
    mass = read("src/Physics/Fluid/FluidPhysicalMass.cpp")
    assert "model == MatterConstitutiveModel::Granular ? profile->density" in mass
    assert "rest_mass <= 0.0f" in mass
    assert "mixed_legacy_granular);" in read("src/Physics/Fluid/FluidDomainStep.inl")
    assert "fluid_params.grain.enabled || fluid_params.pore_exchange.enabled ||" in read(
        "src/Physics/Fluid/FluidDomainStep.inl")
    driver = read("src/Physics/Fluid/MatterGpuStep.inl")
    assert driver.index("exchangeMatterPoresGpu") < driver.index("particles = std::move(liquid_result)")
    assert driver.index("particles = std::move(liquid_result)") < driver.index("ledger.record")
    assert "view.mass_fraction = runtime.transport_fraction" in driver
    assert "matterPoreSettingsHash(fp.pore_exchange)" in read("include/scene_data.h")
    for name in ["src/UI/MatterPoreControls.cpp", "src/Api/RtIpcMatterModels.cpp",
                 "src/Api/RtPythonMatterModels.cpp"]:
        assert "setMatterPoreExchange" in read(name), name
    for name in ["src/Core/ProjectManager.cpp", "src/Utils/SceneSerializer.cpp"]:
        assert "matterPoreParamsFromJson" in read(name), name
        assert "matterPoreParamsToJson" in read(name), name
    probe = root / "scripts/test/rt_test_matter_pores_ipc.py"
    ast.parse(probe.read_text())
    print("PASS C5 sidecar lifecycle/cache, budget/conservation publication and shared authoring")


if __name__ == "__main__":
    main()
