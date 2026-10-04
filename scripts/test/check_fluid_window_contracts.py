"""Non-build source contracts for the Vulkan fluid window dispatch bundle."""

import ast
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2] / "RayTrophiStudio"
    shaders = root / "source/shaders"
    header = (root / "source/include/Fluid/FluidActivePressure.h").read_text(encoding="utf-8")
    registry = (root / "source/src/Device/SimulationComputeVulkan.cpp").read_text(encoding="utf-8")
    compiler = (shaders / "compile_fluid_window_shaders.bat").read_text(encoding="utf-8")
    pairs = re.findall(r'\{"(sim_fluid_\w+)", "(sim_fluid_\w+_window)"\}', header)
    assert len(pairs) == 17
    for full, window in pairs:
        assert window == full + "_window"
        source = (shaders / (full + ".comp")).read_text(encoding="utf-8")
        assert '#include "fluid_pressure_window.glsl"' in source, full
        assert "#ifdef RT_FLUID_WINDOW" in source, full
        assert "fluidPressureCellIndex()" in source, full
        assert "gl_GlobalInvocationID.x" not in source, full
        original = re.search(r'\{\s*"' + full + r'",\s*"[^"]+",\s*(\d+),\s*52\s*\}', registry)
        variant = re.search(r'\{\s*"' + window + r'",\s*"[^"]+",\s*(\d+),\s*76\s*\}', registry)
        assert original and variant, full
        assert original.group(1) == variant.group(1), full
        assert re.search(r'\b' + full + r'\b', compiler), full
        if "double" in source:
            gate = registry.split("if (!m_has_shader_float64 &&", 1)[1].split("return false;", 1)[0]
            assert '"' + window + '"' in gate, window
        if "shared double" in source:
            # Padded reduction lanes must reach barriers with zero contribution.
            main_body = source.split("void main()", 1)[1]
            assert "barrier();" in main_body and "return;" not in main_body, full
    for full in ("sim_fluid_cg_scalar_step", "sim_fluid_cg_residual_init",
                 "sim_fluid_subtract_gradient", "sim_fluid_subtract_gradient_var"):
        assert full not in dict(pairs), full
    pressure = (root / "source/src/Physics/Fluid/FluidGpuPressure.inl").read_text(encoding="utf-8")
    assert "dot_blocks = window_dispatch.groups(cell_groups)" in pressure
    assert "c.iterations = static_cast<int>(dot_blocks)" in pressure
    assert "window_dispatch.dispatch(*compute, cmd, c)" in pressure
    assert "uploadPreparedMask" in pressure
    assert "gpu_buffers.fluid_mask_device_valid" in pressure
    for project in ("RayTrophiStudio.vcxproj", "RayTrophiStudio.vcxproj.filters"):
        ET.parse(root / project)
    for script in ("rt_test_fluid_active_window_ipc.py", "rt_probe_fluid_active_window_matrix_ipc.py"):
        ast.parse((Path(__file__).parent / script).read_text(encoding="utf-8"))
    print("PASS: 17 window variants, ABI/bindings, float64 gates, reduction tails, project XML")


if __name__ == "__main__":
    main()
