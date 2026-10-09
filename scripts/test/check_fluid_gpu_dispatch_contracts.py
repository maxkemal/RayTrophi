"""Source-only audit of Vulkan 2D transfer/copy indexing and unchanged ABIs."""

import re
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RayTrophiStudio/source"


def read(path):
    return (SOURCE / path).read_text(encoding="utf-8-sig")


def main():
    shader = read("shaders/sim_dispatch.glsl")
    assert shader.index("if (group >= logical_groups)") < shader.index("return group * 256u")
    assert "count / 256u" in shader and "count % 256u" in shader
    assert "gl_WorkGroupID.y * gl_NumWorkGroups.x" in shader
    for path in ("sim_fluid_p2g_scatter.glsl", "sim_fluid_g2p.glsl",
                 "sim_fluid_clear_float.comp", "sim_fluid_p2g_normalize.comp",
                 "sim_fluid_normalize_window.comp", "sim_matter_copy.comp",
                 "sim_matter_clear.comp", "sim_matter_contact.glsl"):
        source = read("shaders/" + path)
        assert '#include "sim_dispatch.glsl"' in source, path
        assert "simLane256(" in source, path
    for path in ("include/Fluid/FluidGpuFlipSnapshot.h", "include/Fluid/MatterGpuModelView.h",
                 "src/Physics/Fluid/FluidGpuP2G.inl", "src/Physics/Fluid/FluidGpuTransferStages.inl",
                 "src/Physics/Fluid/SparsePressureGpu.cpp", "src/Physics/Fluid/SparseViscosityGpu.cpp",
                 "src/Physics/Fluid/MatterGpuStep.inl"):
        assert "FluidGpuDispatch::groups256" in read(path), path
    for path in ("sim_fluid_p2g_scatter.glsl", "sim_fluid_g2p.glsl"):
        body = read("shaders/" + path)
        assert body.index("if (lane >= uint(pc.particle_count))") < body.index("int id = int(lane)")
    registry = read("src/Device/SimulationComputeVulkan.cpp")
    for kernel, bindings, push in (("sim_fluid_p2g_scatter", 5, 36),
                                   ("sim_fluid_g2p", 10, 68), ("sim_matter_p2g", 10, 40),
                                   ("sim_matter_g2p", 12, 72), ("sim_matter_copy", 2, 4)):
        assert re.search(rf'"{kernel}\.spv",\s*{bindings},\s*{push}', registry), kernel
    for name in ("RayTrophiStudio.vcxproj", "RayTrophiStudio.vcxproj.filters"):
        project = ET.parse(ROOT / "RayTrophiStudio" / name)
        assert sum(node.attrib.get("Include") == r"source\include\Fluid\FluidGpuDispatch.h"
                   for node in project.iter()) == 1
    print("PASS 2D transfer/copy shader-host wiring, padded-lane guards and unchanged ABIs")


if __name__ == "__main__":
    main()
