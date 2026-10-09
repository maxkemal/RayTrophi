"""C6 source-only lifecycle/ABI/parity audit. No C++/shader build or app run."""
from pathlib import Path
import ast
import json
import xml.etree.ElementTree as ET


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / "RayTrophiStudio/source"
    def read(name):
        return (source / name).read_text(encoding="utf-8-sig")
    keys = ["wet_response_enabled", "wet_appearance_enabled", "wet_friction_scale",
            "wet_dilatancy_scale", "capillary_cohesion_pa", "pore_pressure_scale",
            "wet_color_scale", "wet_roughness_scale", "wet_appearance_full_saturation"]
    metadata = json.loads((root / "scripts/ipc_descriptor_overlay.json").read_text(encoding="utf-8"))
    for key in keys:
        assert key in read("include/Fluid/MatterPoreExchange.h")
        assert key in read("src/Physics/Fluid/MatterPoreAuthoring.cpp")
        assert key in read("src/UI/MatterPoreControls.cpp")
        assert key in metadata["fluid.set_pore_exchange"]["params"]
    core = read("src/Physics/Fluid/MatterWetResponse.cpp")
    assert "4.0f * s * (1.0f - s)" in core
    assert "voxel_size, 0.0f) * s * s" in core
    assert "mass > capacity * 1.00001f" in core
    assert "profile->default_constitutive_model" in core
    shader = read("shaders/sim_fluid_granular_stress_update.glsl")
    assert "matter_dry_volume[id]" in read("shaders/sim_fluid_granular_stress_p2g.glsl")
    assert "binding = 17" in shader and "wet_response[i]" in shader
    assert shader.count("max(pressure - wet.w, 0.0) * friction_tangent") == 3
    # Snow compaction scales bonds too (exp(xi (pv - 1)), 1 when off).
    assert "bond_scale * compaction_scale + wet.z" in shader
    assert "exp(-dilatancy_tangent * dp)" in shader
    step = read("src/Physics/Fluid/MatterGpuStep.inl")
    assert step.index("buildMatterWetResponses") < step.index("for (int substep")
    assert "view.wet_response = runtime.wet_response" in step
    assert "view.dry_volume = runtime.dry_volume" in step
    assert "destroyBuffer(runtime->wet_response)" in read("src/Physics/Fluid/MatterGpuRelease.inl")
    assert "buffers.push_back(model.wet_response)" in read("src/Physics/Fluid/MatterGpuModelView.cpp")
    appearance = read("src/Physics/Fluid/MatterWetAppearance.cpp")
    assert "matterParticleSaturation(particles, particle)" in appearance
    assert "manager.generation()" in appearance
    assert "matterWetAppearanceBand(saturation, appearance_full_saturation)" in appearance
    assert "params.wet_appearance_full_saturation" in appearance
    assert "std::ceil(" in core and "saturation <= 0.0f" in core
    assert "matterWetAppearanceBand(" in read("src/Physics/Fluid/MatterAcceptanceMetrics.cpp")
    assert "Triangle" not in appearance
    bridge = read("src/Physics/ParticleRenderBridge.cpp")
    assert "wet_palette.sourceIndex(" in bridge and "wet_palette.appendMaterialKeys(" in bridge
    assert "wet_appearance_enabled" in read("src/UI/ParticleBillboardBuilder.cpp")
    for name in ["src/Core/ProjectManager.cpp", "src/Utils/SceneSerializer.cpp"]:
        assert "matterPoreParamsToJson" in read(name) and "matterPoreParamsFromJson" in read(name)
    for name in ["src/Api/RtPythonMatterModels.cpp", "src/Api/RtIpcMatterModels.cpp"]:
        assert "setMatterPoreExchange" in read(name)
    for name in ["RayTrophiStudio.vcxproj", "RayTrophiStudio.vcxproj.filters"]:
        project = root / "RayTrophiStudio" / name
        ET.parse(project)
        text = project.read_text(encoding="utf-8-sig")
        for module in ["MatterWetResponse", "MatterWetAppearance", "MatterAcceptanceMetrics"]:
            assert f"{module}.cpp" in text and f"{module}.h" in text
    ast.parse((root / "scripts/test/rt_test_matter_wet_ipc.py").read_text(encoding="utf-8"))
    ast.parse((root / "scripts/test/rt_test_matter_snapshot_ipc.py").read_text(encoding="utf-8"))
    assert "inspectMatterAcceptanceMetrics" in read("src/Api/RtApiMatterModels.cpp")
    print("PASS C6 source contracts: ABI, canonical saturation, appearance, parity, persistence")


if __name__ == "__main__":
    main()
