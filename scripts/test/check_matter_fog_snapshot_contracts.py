"""Read-only source contracts, not live VDB or GPU allocation acceptance."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'RayTrophiStudio/source'


def read(path):
    return (ROOT / path).read_text(encoding='utf-8')


def main():
    publisher = read('src/Physics/Fluid/MatterFogPublisher.cpp')
    snapshot = read('src/Physics/Fluid/MatterFogSnapshot.cpp')
    header = read('include/Fluid/MatterFogSnapshot.h')
    assert 'MatterFogSnapshotPurpose::Publication' in publisher
    assert 'purpose = MatterFogSnapshotPurpose::Inspection' in header
    assert 'budget - retained_snapshot_bytes' in snapshot
    assert snapshot.index('matterFogSnapshotBudgetBytes') < snapshot.index('std::vector<float> density;')
    cells = 222**3
    assert cells * 36 > 128 * 1024**2
    assert cells * (8 + 24) + 100000 < 1024**3
    # Both public bindings carry the exact same core-produced simulation census.
    ipc = read('src/Api/RtIpc.cpp')
    python = read('src/Api/RtPython.cpp')
    assert 'json::parse(r.simulation_memory_json)' in ipc
    assert 'attr("loads")(r.simulation_memory_json)' in python
    backend = read('src/Device/SimulationComputeVulkan.cpp')
    assert 'allocation.allocation_bytes = req.size;' in backend
    assert 'report.device_local_bytes += buffer.allocation_bytes' in backend
    assert 'report.allocation_bytes += buffer.allocation_bytes' in backend
    assert 'backend_->memoryReport()' in read('src/Physics/SimulationCompute.cpp')
    print('PASS distinct inspection/publication budgets, retained snapshot charge '
          'and shared compute allocation census; no GPU execution')


if __name__ == '__main__':
    main()
