"""Non-build wiring checks. Numerical C++ regression is domain_wall_contact_test.cpp."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RayTrophiStudio/source"
def read(path):
    return (SOURCE / path).read_text(encoding="utf-8")


def main():
    core = read("include/Fluid/DomainWallContact.h")
    settings = read("include/Fluid/DomainWallSettings.h")
    assert "std::max(normal_change, support)" in core
    assert "vn > 1.0e-6f" in core
    # MPM grid support includes arbitrary constitutive stress: no restitution.
    assert "grid_wall.restitution = 0.0f;" in core
    assert "grid_wall, acceleration, dt), component)" in core
    grid_shader = read("shaders/sim_domain_wall_grid.glsl")
    assert "max(pc.adhesion_pa, 0.0) * pc.dt / max(pc.density * pc.h, 1e-6)" in grid_shader
    assert "velocity[axis] = wallSnapshot(axis, f);" not in grid_shader
    assert "pc.restitution" not in grid_shader

    assert settings.index("validateDomainWallContact(wall, error)") < settings.index(
        "params.wall_friction = wall.friction")
    assert "unknown wall setting" in settings
    assert "wall.restitution = 0.0f;" in settings
    assert "wall.friction = 0.0f;" in settings  # MPM tail must not reapply traction.
    for source in ("src/UI/DomainWallControls.h", "src/Api/RtApiDomainWalls.cpp"):
        assert "patchDomainWallSettings(" in read(source)
    for source in ("src/Api/RtPythonDomainWalls.cpp", "src/Api/RtIpcDomainWalls.cpp"):
        for call in ("getDomainWallContact", "setDomainWallContact"):
            assert call in read(source)
    for name in ("get_wall_contact", "set_wall_contact"):
        assert name in read("src/Api/RtPythonDomainWalls.cpp")
        assert "fluid." + name in read("src/Api/RtIpcMethodDescriptors.cpp")
    assert "registerDomainWallBindings(fluid)" in read("src/Api/RtPython.cpp")
    assert "dispatchDomainWallsIpc(" in read("src/Api/RtIpc.cpp")
    cpu = read("src/Physics/Fluid/APICFluidSolver.cpp")
    assert "applyDomainWallGridContact(grid, params, dt)" in cpu
    assert "resolveDomainWallTail(pn, pv, lo, hi, params, sub_dt, grid.voxel_size)" in cpu
    assert "enforceSolidBoundaries(grid, params, dt, false)" in read(
        "src/Physics/Fluid/APICFluidStep.inl")
    assert "solid_faces(lane, lane_dt, false)" in read(
        "src/Physics/Fluid/MatterGpuStep.inl")
    assert "(pc.flags & 3u) != 3u" in read("shaders/sim_domain_wall_grid.glsl")
    assert 'result = domainWallFace(result, component, f);' in read(
        "shaders/sim_matter_mpm_boundary.comp")
    stage = read("src/Physics/Fluid/FluidGpuTransferStages.inl")
    assert "params.granular_enabled ? 0.0f : params.wall_friction" in stage
    assert "params.granular_enabled ? 0.0f : params.wall_damping" in stage
    assert "motion.domain_wall = Fluid::domainWallContact(params)" in read(
        "src/Physics/Fluid/MatterGrainStep.inl")
    dem = read("src/Physics/Fluid/MatterGrainGpu.cpp")
    assert "(2 * capacity + 8) * sizeof(float)" in dem
    assert "std::vector<float> masses(2 * count + 8)" in dem
    assert "sleep_context.data() + 24, masses.data() + 2 * count + 4" in dem
    shader = read("shaders/sim_matter_grain.glsl")
    assert "(key & PATCH_KEY) == WALL_KEY" in shader
    assert "domain_wall ? masses[2u * pc.meta.x + 4u] : pc.step_contact.w" in shader
    assert "domain_wall ? masses[2u * pc.meta.x + 5u] : pc.step_contact.y" in shader
    for source in ("src/Core/ProjectManager.cpp", "src/Utils/SceneSerializer.cpp"):
        assert read(source).count('"wall_friction"') >= 2
    assert "fb(fp.wall_friction)" in read("include/scene_data.h")
    assert "params.wall_damping = s.solver_wall_damping" not in read(
        "src/Physics/Fluid/FluidDomainSubstance.cpp")
    # Every new TU is registered in the explicit Visual Studio source list.
    project = ET.parse(ROOT / "RayTrophiStudio/RayTrophiStudio.vcxproj")
    includes = {item.attrib.get("Include", "").replace("\\", "/")
                for item in project.getroot().iter()}
    for name in ("RtApiDomainWalls", "RtIpcDomainWalls", "RtPythonDomainWalls"):
        assert "source/src/Api/" + name + ".cpp" in includes
    print("PASS shared domain-wall settings, UI/script/IPC, persistence, CPU/GPU and DEM wiring")


if __name__ == "__main__":
    main()
