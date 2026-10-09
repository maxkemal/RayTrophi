"""Source-only ABI, shared-physics and public-surface audit; does not compile."""

import re
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RayTrophiStudio/source"


def read(path):
    return (SOURCE / path).read_text(encoding="utf-8-sig")


def expand(text, defines):
    output = []
    stack = []
    active = True
    for line in text.splitlines():
        part = line.strip()
        if part.startswith("#ifdef "):
            condition = part.split()[1] in defines
            stack.append([active, condition])
            active = active and condition
        elif part.startswith("#elif defined("):
            condition = re.search(r"defined\((\w+)\)", part).group(1) in defines
            parent, taken = stack[-1]
            active = parent and not taken and condition
            stack[-1][1] = taken or condition
        elif part == "#else":
            parent, taken = stack[-1]
            active = parent and not taken
            stack[-1][1] = True
        elif part == "#endif":
            active = stack.pop()[0]
        elif active:
            output.append(line)
    assert not stack
    return "\n".join(output)


def main():
    body = read("shaders/sim_fluid_viscosity.glsl")
    registry = read("src/Device/SimulationComputeVulkan.cpp")
    # Both entry points compile this exact physical operator, not a copied one.
    assert len(re.findall(r"\bbool relax\(", body)) == 1
    assert len(re.findall(r"\bvoid classify\(", body)) == 1
    assert 'group >= (pc.compact_faces + 255u) / 256u' in body
    for stage in (None, "clear", "mark", "capture", "sweep"):
        filename = "sim_fluid_viscosity_rbgs" if stage is None else f"sim_sparse_viscosity_{stage}"
        wrapper = read(f"shaders/{filename}.comp")
        assert wrapper.startswith("#version 450")
        assert '#include "sim_fluid_viscosity.glsl"' in wrapper
        defines = set(re.findall(r"#define\s+(\w+)", wrapper))
        expanded = expand(body, defines)
        count, push_bytes = (11, 52) if stage is None else (13, 68)
        assert sorted(map(int, re.findall(r"binding\s*=\s*(\d+)", expanded))) == list(range(count))
        push = expanded.split("uniform PC {", 1)[1].split("} pc;", 1)[0]
        push = re.sub(r"//[^\n]*", "", push)
        assert len(re.findall(r"\b(?:int|uint|float)\s+\w+\s*;", push)) * 4 == push_bytes
        assert re.search(rf'"{filename}\.spv",\s*{count},\s*{push_bytes}', registry)
    host = read("src/Physics/Fluid/SparseViscosityGpu.cpp")
    assert 'sizeof(Constants) == 68' in host
    assert 'offsetof(Constants, tiles_x) == sizeof(ViscosityGpuConstants)' in host
    driver = read("src/Physics/ParticleSimulation.cpp")
    assert 'using FluidViscosityGpuConstants = Fluid::ViscosityGpuConstants;' in driver
    assert 'releaseSparseViscosity(compute, buffers.sparse_viscosity)' in driver
    for field in ("viscosity_sparse_used", "viscosity_sparse_active_tiles",
                  "viscosity_sparse_allocated_tiles", "viscosity_sparse_resident_bytes"):
        for path in ("include/Fluid/APICFluidSolver.h", "include/Api/RtApi.h",
                     "src/Api/RtApiParticle.cpp", "src/Api/RtIpc.cpp", "src/Api/RtPython.cpp",
                     "include/Fluid/FluidActiveWindow.h", "src/Physics/Fluid/FluidDomainStep.inl"):
            assert field in read(path), (path, field)
    for name in ("RayTrophiStudio.vcxproj", "RayTrophiStudio.vcxproj.filters"):
        document = ET.parse(ROOT / "RayTrophiStudio" / name)
        for path in (r"source\include\Fluid\SparseViscosityGpu.h",
                     r"source\src\Physics\Fluid\SparseViscosityGpu.cpp"):
            assert sum(node.attrib.get("Include") == path for node in document.iter()) == 1
    print("PASS dense 11/52 + sparse 13/68 viscosity ABIs, shared operator, lifecycle, API/UI")


if __name__ == "__main__":
    main()
