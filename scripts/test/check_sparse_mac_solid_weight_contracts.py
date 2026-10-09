"""Non-build S1b solid-weight audit. Does not execute C++ or GPU kernels.

The C++ packer test is sparse_mac_solid_weights_test.cpp (user builds it).
Live dense/compact pressure parity remains an external IPC acceptance gate.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from check_sparse_mac_canonical_contracts import entry, bindings, push_bytes, read

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "RayTrophiStudio/source").is_dir())


def main():
    helper = read("shaders/sim_mac_solid_weight.glsl")
    assert "uint address = macAddress(component, i, j, k);" in helper
    assert re.search(r"if \(address == MAC_ABSENT\) \{\s*return 1\.0;", helper)
    for name in ("sim_fluid_divergence_var", "sim_fluid_divergence_porous",
                 "sim_fluid_subtract_gradient_var"):
        body = read(f"shaders/{name}.glsl")
        assert '#include "sim_mac_solid_weight.glsl"' in body
        for axis in range(3):
            assert f"return macSolidWeight({axis}, i, j, k);" in body
        assert not re.search(r"return (?:uw|vw|ww)\[", body)
    for stage in ("init", "spmv"):
        kernel = f"sim_sparse_pressure_{stage}_mac_weights"
        expanded = entry(kernel)
        assert bindings(expanded) == 18 and push_bytes(expanded) == 80
        assert "(slot - 1u) * 576u" in expanded
        # Both map conventions coexist; cell CG pages keep slot/512.
        assert "slot * 512u" in expanded and "const uint EMPTY = 0xffffffffu" in expanded
    transfer = read("src/Physics/Fluid/SparseMacTransferGpu.cpp")
    assert "storage.solid_weights_ready = false;" in transfer
    assert "canonical && params.variational_solids ? 14u : 11u" in transfer
    assert "lookup_bytes + page_fields * bytes > budget" in transfer
    assert "binding.solid_weight = {pool[11], pool[12], pool[13]}" in transfer
    upload = read("src/Physics/Fluid/SparseMacSolidWeightsGpu.cpp")
    assert "list[0] != active" in upload and "storage.nx != grid.nx" in upload
    assert "storage.solid_weights_ready = false;" in upload
    assert upload.index("compute.endTransferBatch() && ok;", upload.index("uploadBuffer(")) < \
        upload.index("storage.solid_weights_ready = true;")
    assert "createBuffer(" not in upload  # P2G owns budget and page allocation.
    assert "storage.owned[11u + axis]" in upload
    pack = read("src/Physics/Fluid/SparseMacSolidWeights.cpp")
    assert "FluidSim::FluidGrid::weightToFloat" in pack
    assert "owner_key != key" in pack and "page.assign(values, 1.0f)" in pack
    assert "std::adjacent_find" in pack and "pages = std::move(candidate)" in pack
    assert "cell_count > uint64_t(std::numeric_limits<int32_t>::max()) / dims[axis]" in pack
    pressure = read("src/Physics/Fluid/FluidGpuPressure.inl")
    assert pressure.index("uploadSparseMacSolidWeights(") < \
        pressure.index("gpu_buffers.matter_model.pressure_statics_uploaded = true;")
    assert "if (is_variational && mac.compact)" in pressure
    for axis in range(3):
        assert f"fluid_divergence_bufs[{5 + axis}] = mac.solid_weight[{axis}]" in pressure
        assert f"gradient_buffers[{5 + axis}] = mac.solid_weight[{axis}]" in pressure
    for project in ("RayTrophiStudio.vcxproj", "RayTrophiStudio.vcxproj.filters"):
        nodes = list(ET.parse(ROOT / "RayTrophiStudio" / project).iter())
        for path in (r"source\include\Fluid\SparseMacSolidWeightsGpu.h",
                     r"source\src\Physics\Fluid\SparseMacSolidWeights.cpp",
                     r"source\src\Physics\Fluid\SparseMacSolidWeightsGpu.cpp"):
            assert sum(node.attrib.get("Include") == path for node in nodes) == 1, path
    print("PASS S1b solid weights: shared face reads, 18/80 pressure twins, "
          "pool budget/lifecycle, per-topology packing, atomic validation, project registration")


if __name__ == "__main__":
    main()
