"""Non-build audit of host/shader registration and public diagnostic surfaces."""

import re
from pathlib import Path
import xml.etree.ElementTree as ET
from check_sparse_mac_canonical_contracts import entry, bindings, push_bytes


ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "RayTrophiStudio/source").is_dir())
SOURCE = ROOT / "RayTrophiStudio" / "source"


def read(path):
    return (SOURCE / path).read_text(encoding="utf-8-sig")


def main():
    shader = read("shaders/sim_sparse_pressure.glsl")
    host = read("src/Physics/Fluid/SparsePressureGpu.cpp")
    registry = read("src/Device/SimulationComputeVulkan.cpp")
    compile_script = read("shaders/compile_sim_shaders.bat")
    stages = ("clear", "mark", "init", "jacobi", "copy", "spmv", "axpy", "zpby",
              "scatter", "dense_clear")
    assert bindings(entry("sim_sparse_pressure_init")) == 16
    push = shader.split("uniform PC {", 1)[1].split("} pc;", 1)[0]
    assert len(re.findall(r"\b(?:int|uint|float)\s+\w+\s*;", push)) * 4 == 80
    assert 'sizeof(Constants) == 80' in host
    for stage in stages:
        wrapper = read(f"shaders/sim_sparse_pressure_{stage}.comp")
        assert wrapper.startswith("#version 450")
        assert f"#define SPARSE_{stage.upper()}" in wrapper
        assert f'"sim_sparse_pressure_{stage}.spv", 16, 80' in registry
        assert stage in compile_script
    for stage in ("init", "spmv"):
        kernel = f"sim_sparse_pressure_{stage}_mac_weights"
        expanded = entry(kernel)
        assert bindings(expanded) == 18
        assert push_bytes(expanded) == 80
        assert f'"{kernel}.spv", 18, 80' in registry
        assert kernel in compile_script and kernel in host
        assert "macSolidWeight(axis, face.x, face.y, face.z)" in expanded
        assert "(slot - 1u) * 576u" in expanded
    assert "compact_weights && !buffers.sparse_mac_transfer.solid_weights_ready" in host
    assert "command.buffer_count = compact_init || compact_spmv ? 18 : 16;" in host
    assert 'kernel.rfind("sim_sparse_pressure_", 0) == 0' in registry
    assert 'return group * 256u + gl_LocalInvocationID.x;' in shader
    assert 'group >= (count + 255u) / 256u' in shader
    assert 'gl_WorkGroupID.y * gl_NumWorkGroups.x' in shader
    for project in ("RayTrophiStudio.vcxproj", "RayTrophiStudio.vcxproj.filters"):
        entries = ET.parse(ROOT / "RayTrophiStudio" / project)
        for path in (r"source\include\Fluid\SparsePressureGpu.h",
                     r"source\src\Physics\Fluid\SparsePressureGpu.cpp"):
            assert sum(node.attrib.get("Include") == path for node in entries.iter()) == 1
    fields = ("pressure_sparse_used", "pressure_sparse_active_tiles",
              "pressure_sparse_allocated_tiles", "pressure_sparse_resident_bytes")
    for path in ("include/Fluid/APICFluidSolver.h", "include/Api/RtApi.h",
                 "src/Api/RtApiParticle.cpp", "src/Api/RtIpc.cpp", "src/Api/RtPython.cpp",
                 "include/Fluid/FluidActiveWindow.h", "src/Physics/Fluid/FluidDomainStep.inl"):
        source = read(path)
        for field in fields:
            assert field in source, (path, field)
    driver = read("src/Physics/ParticleSimulation.cpp")
    assert "releaseSparsePressure(compute, buffers.sparse_pressure)" in driver
    print("PASS sparse pressure dense-weight 16/80 + compact-weight 18/80 ABIs, "
          "ten shared stages + two weight twins, 2D lanes/reductions, lifecycle and API/UI")


if __name__ == "__main__":
    main()
