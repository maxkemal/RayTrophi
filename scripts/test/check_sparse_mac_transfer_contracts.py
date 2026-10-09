"""Source-only sparse MAC ABI/lifecycle/public-surface audit; no compiler/GPU."""

import re
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "RayTrophiStudio/source").is_dir())
SOURCE = ROOT / "RayTrophiStudio/source"


def read(path):
    return (SOURCE / path).read_text(encoding="utf-8-sig")


def expand(filename, defines=None):
    defines = set() if defines is None else defines
    output, stack = [], []
    active = True
    for line in read("shaders/" + filename).splitlines():
        part = line.strip()
        if part.startswith(("#ifdef ", "#ifndef ")):
            condition = part.split()[1] in defines
            if part.startswith("#ifndef "):
                condition = not condition
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
        elif active and part.startswith("#define "):
            defines.add(part.split()[1])
        elif active and part.startswith('#include "'):
            output.append(expand(part.split('"')[1], defines))
        elif active:
            output.append(line)
    assert not stack, filename
    return "\n".join(output)


def main():
    table = read("src/Device/SimulationComputeVulkan.cpp")
    batch = read("shaders/compile_sim_shaders.bat")
    specs = {op: (8, 36) for op in
             ("clear", "mark", "reset", "normalize", "publish", "capture", "gather")}
    specs.update(p2g=(7, 36), matter_p2g=(12, 40), g2p=(11, 68), matter_g2p=(13, 72))
    for name, (bindings, push_bytes) in specs.items():
        kernel = "sim_sparse_mac_" + name
        source = expand(kernel + ".comp")
        actual = sorted(map(int, re.findall(r"binding\s*=\s*(\d+)", source)))
        assert actual == list(range(bindings)), (kernel, actual)
        push = re.search(r"layout\(push_constant\).*?\{(.*?)\}\s*pc;",
                         source, re.S).group(1)
        push = re.sub(r"//[^\n]*", "", push)
        assert len(re.findall(r"\b(?:int|uint|float)\s+\w+", push)) * 4 == push_bytes
        assert re.search(rf'"{kernel}\.spv",\s*{bindings},\s*{push_bytes}', table)
        assert kernel in batch
        assert "simLane256(" in source
        assert source.count("#version 450") == 1
    p2g = expand("sim_sparse_mac_matter_p2g.comp")
    assert "matter_rest[id] * matter_fraction[id]" in p2g
    assert "matter_gradient[dense_fi * 3]" in p2g
    assert "fi = int((slot - 1u) * 576u" in p2g
    g2p = expand("sim_sparse_mac_matter_g2p.comp")
    assert "vel_x_post[address]" in g2p and "vel_x_pre[address]" in g2p
    assert "use_solid_flip_limiter" in g2p
    host = read("src/Physics/Fluid/SparseMacTransferGpu.cpp")
    assert "&active, sizeof(active)" in host
    # The transfer itself reads back one tile count; the compact canonical
    # field's frame publication (S1) is its own explicit function.
    transfer_host = host.split("bool publishCompactMacToHost")[0]
    assert transfer_host.count("downloadBuffer(") == 1
    assert "bool publishCompactMacToHost(" in host
    assert "max_storage_buffer_bytes" in host
    assert "mixed_working_set_budget_bytes" in host
    assert "lookup_bytes + page_fields * bytes > budget" in host
    assert "canonical && params.variational_solids ? 14u : 11u" in host
    assert "compute.getBufferSize(handle)" in host
    assert "storage.flip_gather_used" in host
    assert "releaseSparseMacTransfer(compute, storage)" in host
    assert "releaseSparseMacTransfer(compute, buffers.sparse_mac_transfer)" in read(
        "src/Physics/ParticleSimulation.cpp")
    pure = read("src/Physics/Fluid/FluidDomainStep.inl")
    mixed = read("src/Physics/Fluid/MatterGpuStep.inl")
    assert pure.index("captureSparseMacFlip(") < pure.index("runGpuFluidViscosity(")
    assert mixed.index("runGpuFluidZeroSolidFaces(grid") < mixed.index("captureSparseMacFlip(")
    assert mixed.index("captureSparseMacFlip(") < mixed.index("runGpuFluidViscosity(")
    fields = ("transfer_sparse_used", "flip_sparse_used", "transfer_sparse_active_tiles",
              "transfer_sparse_allocated_tiles", "transfer_sparse_resident_bytes",
              "transfer_sparse_status")
    for path in ("include/Fluid/APICFluidSolver.h", "include/Api/RtApi.h",
                 "src/Api/RtApiParticle.cpp", "src/Api/RtIpc.cpp", "src/Api/RtPython.cpp"):
        for field in fields:
            assert field in read(path), (path, field)
    assert "Sparse transfer; dense solver publication" in read("include/Fluid/FluidActiveWindow.h")
    for filename in ("RayTrophiStudio.vcxproj", "RayTrophiStudio.vcxproj.filters"):
        document = ET.parse(ROOT / "RayTrophiStudio" / filename)
        for path in (r"source\include\Fluid\SparseMacTransferGpu.h",
                     r"source\src\Physics\Fluid\SparseMacTransferGpu.cpp"):
            assert sum(node.attrib.get("Include") == path for node in document.iter()) == 1
    print("PASS sparse MAC transfer ABIs, shared APIC/mass/FLIP, lifecycle and API/UI wiring")


if __name__ == "__main__":
    main()
