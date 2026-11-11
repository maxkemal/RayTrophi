"""Non-build checks for layered seed publication, API surfaces and persistence."""
import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RayTrophiStudio/source"


def read(name):
    return (SOURCE / name).read_text(encoding="utf-8-sig")


def main():
    core = read("src/Physics/Fluid/MatterSeedSources.cpp")
    assert "FluidParticles initial;" in core
    commit = core.index("state.particles = std::move(initial)")
    assert core.index("validateMatterSeedSources(domain") < commit
    for primitive in ("seedBox(initial", "seedMatterGrainBox(initial",
                      "initializeMatterBirthModels(initial", "ensureFluidParticleRestMasses(initial"):
        assert primitive in core
    assert "source.grain_saturation" in core and "source.velocity" in core
    assert core.index("source.mesh_object") < commit
    assert "state.settle_stats.pending" in core
    sync = read("src/Physics/Fluid/MatterDomainSynchronization.inl")
    assert "Fluid::applyMatterSeedSources(domain, state," in sync
    assert sync.index("Fluid::applyMatterSeedSources(") < sync.index("state.particles.clear();")
    assert "domain.matter_seed_sources_enabled" in sync
    for name in ("src/Core/ProjectManager.cpp", "src/Utils/SceneSerializer.cpp"):
        source = read(name)
        assert "writeMatterSeedSources(" in source and "readMatterSeedSources(" in source
    api = read("src/Api/RtApiMatterSeedSources.cpp")
    assert "adoptSeedAsSimulationStart()" in api
    assert "validateMatterSeedSources(candidate" in api
    assert "pending_recipe_changes" in api
    planner = read("src/Physics/Fluid/MatterSeedPlanning.cpp")
    assert "liquidGrid(grid_domain_states_[index])" in planner
    assert "substanceBirthModel(" in planner
    assert "matterGrainMaximumRadius(" in planner
    assert "VK_EXT_MEMORY_BUDGET_EXTENSION_NAME" in planner
    assert core.index("checkMatterSeedMemory(plan, error)") < commit
    assert core.index("domain.fluid_max_particles = capacity") > commit
    assert "matter_seed_auto_capacity" in core and "matter_seed_memory_guard" in core
    assert "suggested_capacity" in planner and '"selected_source_id"' in api
    viewport = read("src/UI/MatterSeedViewport.cpp")
    assert 'rtapi::matterSeedSources(domain, "select"' in viewport
    assert '"update", id' in viewport
    assert "ImGuizmo::Manipulate(" in viewport
    assert "manipulateSelectedMatterSeed(ctx" in read("src/UI/scene_ui_gizmos.cpp")
    assert "matter_seed_sources" in read("src/Api/RtApiSubstance.cpp")
    assert "matterSeedAuthoringSignature(d)" in read("include/scene_data.h")
    assert "matterSeedSourcesUseObject(domain, node_name)" in read("include/scene_data.h")
    ui = read("src/UI/MatterSeedSourcesUI.cpp")
    assert "rtapi::matterSeedSources(" in ui
    assert "rtapi::setDomainEnvironment(" in ui
    assert "rtapi::setFluidThermal(" in ui
    assert "drawMatterSeedSourcesControls(domain)" in read("src/UI/scene_ui_simulation_domains.cpp")
    assert "drawMatterSeedAuthoring(ctx, projectScreen, drawAABB)" in read("src/UI/scene_ui_gizmos.cpp")
    assert "drawMatterSeedSourceBounds(domain, draw_box)" in viewport
    methods = ("get_seed_sources", "add_seed_source", "update_seed_source", "remove_seed_source",
               "apply_seed_sources", "set_seed_source_settings", "select_seed_source")
    overlay = json.loads((ROOT / "scripts/ipc_descriptor_overlay.json").read_text(encoding="utf-8-sig"))
    ipc = read("src/Api/RtIpcMatterSeedSources.cpp")
    python = read("src/Api/RtPythonMatterSeedSources.cpp")
    for method in methods:
        assert '"fluid.' + method + '"' in ipc
        assert '"' + method + '"' in python
        assert "fluid." + method in overlay
    assert "dispatchMatterSeedSourcesIpc(" in read("src/Api/RtIpc.cpp")
    assert "registerMatterSeedSources(fluid)" in read("src/Api/RtPython.cpp")
    registered = {node.attrib.get("Include", "").replace("\\", "/") for node in
                  ET.parse(ROOT / "RayTrophiStudio/RayTrophiStudio.vcxproj").getroot().iter()}
    for path in ("source/src/Physics/Fluid/MatterSeedSources.cpp",
                 "source/src/Api/RtApiMatterSeedSources.cpp",
                 "source/src/Api/RtIpcMatterSeedSources.cpp",
                 "source/src/Api/RtPythonMatterSeedSources.cpp",
                 "source/src/UI/MatterSeedSourcesUI.cpp",
                 "source/src/UI/MatterSeedViewport.cpp",
                 "source/src/Physics/Fluid/MatterSeedPlanning.cpp"):
        assert path in registered
        assert len((ROOT / "RayTrophiStudio" / path).read_text().splitlines()) < 2000
    ast.parse((ROOT / "scripts/test/rt_test_matter_seed_sources_ipc.py").read_text())
    print("PASS layered seed atomic staging, birth primitives, reset hash, persistence, "
          "UI/script/IPC and registration")


if __name__ == "__main__":
    main()
