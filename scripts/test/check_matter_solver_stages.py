"""Non-build audit of the model-local CPU projection/contact/gather boundary."""
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    project = Path(__file__).resolve().parents[2] / "RayTrophiStudio"
    source = project / "source"
    solver = (source / "src/Physics/Fluid/APICFluidSolver.cpp").read_text(encoding="utf-8-sig")
    step = (source / "src/Physics/Fluid/APICFluidStep.inl").read_text(encoding="utf-8-sig")
    stages = (source / "src/Physics/Fluid/MatterSolverStages.cpp").read_text(encoding="utf-8-sig")
    header = (source / "include/Fluid/APICFluidSolver.h").read_text(encoding="utf-8-sig")
    assert '#include "APICFluidStep.inl"' in solver
    assert "void step(FluidParticles& particles," not in solver
    assert len(step.splitlines()) < 2000
    assert "bool stop_after_projection = false;" in header
    assert "bool pressure_precomputed = false;" in header
    assert "const APICFlipSnapshot* model_flip_snapshot = nullptr;" in header
    stop = step.index("if (params.stop_after_projection)")
    gather = step.index("gridToParticle(particles,", stop)
    advect = step.index("advectParticles(particles,", gather)
    assert stop < gather < advect
    assert "if (params.pressure_precomputed)" in step
    assert "want_flip && !params.model_flip_snapshot" in step
    for axis in "xyz":
        assert f"params.model_flip_snapshot->{axis}.data()" in step
        assert f"candidate.flip.{axis}.assign(getLastFlipPreSnapshot{axis.upper()}()" in stages
    for flag in ["p2g_precomputed", "external_forces_preintegrated",
                 "viscosity_precomputed", "pressure_precomputed"]:
        assert f"params.{flag} = true;" in stages
    assert "workspace.identities != particles.particle_id" in stages
    assert "workspace.ready = false;" in stages
    assert "stats.p2g_on_gpu = false;" in stages
    batch = (source / "src/Physics/Fluid/MatterModelBatch.cpp").read_text(encoding="utf-8-sig")
    assert batch.index("prepareMatterModelGrid(") < batch.index("if (!contact(models")
    assert batch.index("if (!contact(models") < batch.index("finishMatterModelGrid(")
    assert batch.index("finishMatterModelGrid(") < batch.index("particles = std::move(merged)")
    assert "merged.next_particle_id = particles.next_particle_id;" in batch
    assert "target.copyParticleFrom" in batch
    assert "if (!contact)" in batch
    for filename in ["RayTrophiStudio.vcxproj", "RayTrophiStudio.vcxproj.filters"]:
        tree = ET.parse(project / filename)
        paths = [node.attrib["Include"] for node in tree.iter()
                 if "Include" in node.attrib]
        for path in [r"source\src\Physics\Fluid\MatterSolverStages.cpp",
                     r"source\src\Physics\Fluid\APICFluidStep.inl",
                     r"source\include\Fluid\MatterSolverStages.h",
                     r"source\include\Fluid\MatterModelBatch.h",
                     r"source\src\Physics\Fluid\MatterModelBatch.cpp"]:
            assert paths.count(path) == 1, (filename, path)
    print("PASS: projection/gather split, per-model FLIP, topology gate, project XML")


if __name__ == "__main__":
    main()
