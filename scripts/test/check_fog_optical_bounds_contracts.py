"""Non-build proof checks for conservative material-aware Fog bounds.

Actual C++ core cases are in fog_optical_bounds_test.cpp for the user to run.
"""
from pathlib import Path
import math
import random
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "RayTrophiStudio/source"


class FogOpticalBoundsContracts(unittest.TestCase):
    def test_live_panel_edits_domain_material_and_culling_invalidates(self):
        volume = (SRC / "include/VDBVolume.h").read_text(encoding="utf-8")
        getter = volume.split("getOrCreateShader() {", 1)[1].split("\n    }", 1)[0]
        self.assertLess(getter.index("return authored_shader;"),
                        getter.index("return volume_shader;"))
        self.assertIn("*authored_shader = *shader;", volume)
        self.assertIn("authored_shader = authored;", volume)
        header = (SRC / "include/Fluid/FogOpticalBounds.h").read_text(encoding="utf-8")
        key = header.split("fogOpticalBoundsKey(", 1)[1].split(".dump();", 1)[0]
        for dependency in ("scattering.coefficient", "absorption.coefficient", "emission.mode"):
            self.assertIn(dependency, key)
        publisher = (SRC / "src/Physics/Fluid/MatterFogPublisher.cpp").read_text(encoding="utf-8")
        publish = publisher.split("void SceneData::syncMatterFogStreams", 1)[1]
        self.assertLess(publish.index("cullInvisibleFogBlocks(field.density"),
                        publish.index("manager.registerOrUpdateLiveVolume("))

    def test_monotone_curve_after_trilinear_never_exceeds_corner_maximum(self):
        rng = random.Random(20261028)
        for _ in range(3000):
            curve = [0.] + sorted(rng.random() for _ in range(7))
            low = rng.uniform(-.3, .3)
            width = rng.uniform(.01, 3.)
            def mapped(raw):
                x = max((raw-low)/width, 0.)
                if x >= 1: return x*curve[-1]
                t = x*7
                i = min(int(t), 6)
                return curve[i] + (curve[i+1]-curve[i])*(t-i)
            corners = [rng.random()*3 for _ in range(8)]
            f = [rng.random() for _ in range(3)]
            raw = sum(corners[i]*math.prod(f[a] if i & (1 << a) else 1-f[a]
                                         for a in range(3)) for i in range(8))
            self.assertLessEqual(mapped(raw), max(map(mapped, corners))+1e-12)

    def test_block_box_contains_full_corner_support_at_edges(self):
        for size in (1, 7, 8, 9, 32, 86):
            for corner in range(size):
                # C++ integer division truncates toward zero at corner zero.
                first = max(0, int((corner-1)/8)*8)
                end = min(size, ((corner+9)//8)*8)
                self.assertLessEqual(first, max(0, corner-1))
                self.assertGreaterEqual(end, min(size, corner+1))
                self.assertLess(first, end)

    def test_production_wiring_and_invalidation(self):
        header = (SRC / "include/Fluid/FogOpticalBounds.h").read_text(encoding="utf-8")
        for guard in ("shader->material_graph.empty()", "shader->material_program.active",
                      "shader->material_program.density_noise_enabled", "d.edge_falloff",
                      "mapped(0.0f)", "d.curve.values[i - 1]", "std::isfinite(raw)"):
            self.assertIn(guard, header)
        publisher = (SRC / "src/Physics/Fluid/MatterFogPublisher.cpp").read_text(encoding="utf-8")
        current = publisher.split("const bool current", 1)[1].split("if (current)", 1)[0]
        self.assertIn("stream.optical_bounds_key == fogOpticalBoundsKey(", current)
        self.assertIn("fogOpticalBounds(field.density", publisher)
        self.assertNotIn("shrinkLiveBoundsToOccupied(field.density", publisher)
        self.assertIn("stream.populated && stream.optically_populated && appearance.visible", publisher)
        inspection = (SRC / "src/Api/RtApiMatterFogInspection.cpp").read_text(encoding="utf-8")
        self.assertIn("fogOpticalBounds(field.density", inspection)
        self.assertIn('row["optical_bounds"]', inspection)
        bindings = (SRC / "src/Api/RtVolumeMaterialBindings.cpp").read_text(encoding="utf-8")
        self.assertIn('volume.set_material', bindings)
        self.assertIn('volume.def("set_material"', bindings)

    def test_project_registration_is_valid_xml(self):
        for name in ("RayTrophiStudio.vcxproj", "RayTrophiStudio.vcxproj.filters"):
            project = ROOT / "RayTrophiStudio" / name
            tree = ET.parse(project)
            includes = [node.attrib.get("Include", "") for node in tree.iter()]
            self.assertEqual(includes.count(r"source\include\Fluid\FogOpticalBounds.h"), 1)


if __name__ == "__main__":
    unittest.main()
