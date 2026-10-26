"""Non-build numerical/wiring checks, not compiled C++ or GPU acceptance."""
from pathlib import Path
import math
import random
import re
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RayTrophiStudio/source"


def read(path):
    return (SOURCE / path).read_text(encoding="utf-8-sig")


def reconcile(values, target, policy):
    """Independent scalar oracle for density/fuel partition semantics."""
    density = math.fsum(d for d, f in values)
    fuel = math.fsum(f for d, f in values)
    result = []
    for index, (d, f) in enumerate(values):
        residual = float(index == 0)
        share = f / fuel if policy == 1 and fuel else residual
        next_d = (d * target[0] / density if density else residual * target[0]) \
            if density > target[0] or policy == 2 \
            else d + max(0, target[0] - density) * share
        next_f = (f * target[1] / fuel if fuel else residual * target[1]) \
            if fuel > target[1] or policy == 2 \
            else f + max(0, target[1] - fuel) * residual
        result.append((next_d, next_f))
    return result


class GasSourceContracts(unittest.TestCase):
    def test_randomized_partition_and_permutation(self):
        rng = random.Random(918241)
        for _ in range(4000):
            values = [(rng.random() * 10, rng.random() * 10)
                      for _ in range(rng.randrange(2, 10))]
            target = (rng.random() * 100, rng.random() * 100)
            for policy in range(3):
                result = reconcile(values, target, policy)
                self.assertTrue(all(d >= 0 and f >= 0 for d, f in result))
                self.assertAlmostEqual(math.fsum(d for d, f in result), target[0], places=10)
                self.assertAlmostEqual(math.fsum(f for d, f in result), target[1], places=10)
                reverse = reconcile(values[:1] + values[:0:-1], target, policy)
                self.assertEqual(result[:1] + result[:0:-1], reverse)

    def test_generated_vapor_stays_unattributed(self):
        self.assertEqual(reconcile([(1, 0), (2, 1), (3, 2)], (10, 3), 0),
                         [(5, 0), (2, 1), (3, 2)])

    def test_combustion_products_follow_mixed_fuel(self):
        result = reconcile([(1, 0), (2, 1), (3, 3)], (10, 2), 1)
        self.assertEqual(result, [(1, 0), (3, .5), (6, 1.5)])

    def test_empty_and_extinguished_cells(self):
        self.assertEqual(reconcile([(0, 0)] * 3, (4, 5), 2),
                         [(4, 5), (0, 0), (0, 0)])
        self.assertEqual(reconcile([(1, 2), (3, 4)], (0, 0), 0), [(0, 0), (0, 0)])

    def test_transport_uses_existing_solver_kernels(self):
        core = read("src/Physics/Fluid/GasSourceProvenance.cpp")
        self.assertIn('command.kernel = "sim_grid_advect_scalar"', core)
        self.assertIn('command.kernel = "sim_grid_maccormack_scalar"', core)
        self.assertIn("GridFluid::advectPassiveScalarField", core)
        self.assertIn("buffers.gas_solid_mask", core)
        # Healthy transport does not download tracer fields.
        gpu = core.split("void gpuStage(", 1)[1].split("void gasSourceStage(", 1)[0]
        self.assertNotIn("downloadBuffer", gpu)
        self.assertIn("deviceHasCurrent", core)
        self.assertIn("hostHasCurrent", core)

    def test_deposit_identity_and_stage_order(self):
        sources = read("src/Physics/Fluid/MatterDomainSources.inl")
        self.assertIn("source.timeline_uid, source.name", sources)
        self.assertIn("density_amount * weight", sources)
        self.assertIn("fuel_amount * weight", sources)
        simulation = read("src/Physics/ParticleSimulation.cpp")
        chain = simulation.split("if (gpu_velocity_advection_pre_run &&", 1)[1]
        self.assertLess(chain.index("GasSourceStage::Prepare"), chain.index("runGpuScalarAdvection("))
        self.assertLess(chain.index("GasSourceStage::Advect"), chain.index("runGpuScalarAdvection("))
        self.assertLess(chain.index("runGpuScalarAdvection("), chain.index("GasSourceStage::Transported"))
        self.assertLess(chain.index("runGpuCombustion("), chain.index("GasSourceStage::Reacted"))
        host = read("src/Physics/GridFluidSolver.cpp")
        host = host.split("// 1) Transport.", 1)[1]
        self.assertLess(host.index("GasSourceStage::Transported"), host.index("addSurfaceDust("))

    def test_shared_temperature_and_no_full_density_duplication(self):
        fields = read("src/Physics/Fluid/MatterFogFields.cpp")
        self.assertIn("component.temperature_kelvin = temperature", fields)
        self.assertIn("image.density[cell] * target / sum", fields)
        self.assertIn("uid == 0 ? float(target - sum) : 0.0f", fields)
        snapshot = read("src/Physics/Fluid/MatterFogSnapshot.cpp")
        self.assertIn("snapshotGasSources(source_snapshot)", snapshot)
        self.assertIn("airKelvinFromVirtual", snapshot)

    def test_lifecycle_and_cache(self):
        simulation = read("src/Physics/ParticleSimulation.cpp")
        self.assertIn("!Fluid::snapshotGasSources(state)", simulation)
        self.assertIn("Fluid::releaseGasSources(buffers)", simulation)
        self.assertIn("Fluid::materializeGasSources(state)", simulation)
        self.assertIn("Fluid::releaseGasSourceDomains", read("src/Physics/ParticleDeviceResidency.cpp"))
        self.assertIn("state.gas_source_images.clear()", read("src/Physics/Fluid/MatterPhaseConfig.cpp"))
        self.assertIn("kVersion = 18u", read("include/SimCache.h"))
        cache = read("src/Physics/SimCache.cpp")
        self.assertIn("Fluid::writeGasSourceCache(os, d)", cache)
        self.assertIn("version >= 18u && !Fluid::readGasSourceCache(is, d)", cache)
        codec = read("src/Physics/Fluid/GasSourceCache.cpp")
        self.assertIn("images.count(uid)", codec)
        self.assertIn("!std::isfinite(value) || value < 0.0f", codec)

    def test_material_keys_and_script_ipc_parity(self):
        streams = read("src/Physics/Fluid/MatterFogStreams.cpp")
        self.assertIn('"gas_source:" + std::to_string(key.producer_uid)', streams)
        self.assertIn('row.value("producer_uid", uint64_t{0})', streams)
        bindings = read("src/Api/RtVolumeMaterialBindings.cpp")
        for method in ("volume.list_materials", "volume.get_material", "volume.set_material"):
            self.assertIn(method, bindings)
        self.assertIn('d["producer_uid"] = s.producer_uid', read("src/Api/RtPython.cpp"))
        self.assertIn('{"producer_uid", s.producer_uid}', read("src/Api/RtIpc.cpp"))

    def test_shader_and_project_registration(self):
        shader = read("shaders/sim_gas_source_partition.comp")
        self.assertEqual(sorted(set(map(int, re.findall(r"binding\s*=\s*(\d+)", shader)))),
                         list(range(7)))
        self.assertIn('"sim_gas_source_partition.spv", 7, 16',
                      read("src/Device/SimulationComputeVulkan.cpp"))
        project = ET.parse(ROOT / "RayTrophiStudio/RayTrophiStudio.vcxproj")
        entries = {node.attrib.get("Include", "") for node in project.iter()}
        for name in ("GasSourceProvenance", "GasSourceCache"):
            self.assertIn(f"source\\src\\Physics\\Fluid\\{name}.cpp", entries)
            self.assertIn(f"source\\include\\Fluid\\{name}.h", entries)


if __name__ == "__main__":
    unittest.main()
