"""Non-build audit of shared host/device topology publication contracts."""

import re
from pathlib import Path
import xml.etree.ElementTree as ET

from check_sparse_mac_transfer_contracts import expand

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RayTrophiStudio/source"


def read(path):
    return (SOURCE / path).read_text(encoding="utf-8-sig")


def main():
    host = read("src/Physics/Fluid/SparseGridStorage.cpp")
    device = read("src/Physics/Fluid/SparseGridGpu.cpp")
    header = read("include/Fluid/SparseGridStorage.h")
    assert "std::shared_ptr<const Topology> topology;" in host
    assert "std::array<Pages, channelCount> pages" in host
    assert "std::make_shared<State>(*state_)" in host
    assert host.index("Sparse topology transaction would discard") < host.index(
        "state_.swap(candidate)")
    assert "float background" in host and "isBackground(" in host
    assert "compare_exchange_weak" in host and "fetch_sub" in host
    assert host.index("reserve(values * sizeof(float))") < host.index(
        "page->values.assign(values, background)")
    assert "retained_value_bytes" in header and "value_budget_bytes" in header
    assert "gasPressureTopology" in host
    assert "std::vector<CellBox>{{{0, 0, 0}, dimensions}}" in host
    assert "density" not in host.split("gasPressureTopology", 1)[1]
    assert "Sparse page padding must contain its background" in host
    assert "std::max<uint64_t>(page * constants.new_active, 1u)" in device
    assert "destination.resident_bytes, working_budget_bytes" in device
    assert "grid.resident_bytes, working_budget_bytes" in device
    assert device.index('"sparse retirement would discard') < device.index(
        "candidate.generation = grid.generation + 1u")
    assert "&invalid, sizeof(invalid)" in device
    assert "GridStorage candidate = host;" in device
    assert device.index("candidate.importPage(") < device.index("host = std::move(candidate)")
    assert "downloadBuffer(" not in device.split("bool rebindGrid", 1)[0]
    table = read("src/Device/SimulationComputeVulkan.cpp")
    batch = read("shaders/compile_sim_shaders.bat")
    for stage in ("map_clear", "map_seed", "retire", "remap"):
        name = "sim_sparse_grid_" + stage
        source = expand(name + ".comp")
        assert sorted(map(int, re.findall(r"binding\s*=\s*(\d+)", source))) == list(range(7))
        pc = source.split("uniform PC {", 1)[1].split("} pc;", 1)[0]
        assert len(re.findall(r"\b(?:int|uint|float)\s+\w+", pc)) * 4 == 48
        assert re.search(rf'"{name}\.spv",\s*7,\s*48', table)
        assert name in batch and "simLane256(" in source
    for filename in ("RayTrophiStudio.vcxproj", "RayTrophiStudio.vcxproj.filters"):
        document = ET.parse(ROOT / "RayTrophiStudio" / filename)
        for path in (r"source\include\Fluid\SparseGridStorage.h",
                     r"source\src\Physics\Fluid\SparseGridStorage.cpp",
                     r"source\include\Fluid\SparseGridGpu.h",
                     r"source\src\Physics\Fluid\SparseGridGpu.cpp"):
            assert sum(node.attrib.get("Include") == path for node in document.iter()) == 1
    print("PASS shared sparse topology, atomic remap/snapshots, budget ledger and GPU 7/48 ABIs")


if __name__ == "__main__":
    main()
