"""External cloud-material acceptance, after the user builds C++ and shaders.

Default: inspect schema without editing the scene.
--apply: test edits, GPU density scaling and rejection; restore cloud settings
in finally. This intentionally resets accumulation. Never run inside Studio.
"""

import argparse
import copy
import math

from rt_ipc import RtIpc, RtIpcError


def density(ipc, points, mode="density", backend="viewport"):
    return ipc.call("world.sample_clouds", mode=mode, points=points,
                    backend=backend)["values"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backend", choices=("viewport", "render"), default="viewport")
    args = parser.parse_args()
    ipc = RtIpc()
    original = None
    edited = False
    try:
        original = ipc.call("world.get_clouds")
        assert "material" in original, "Build containing CloudMaterial is not running"
        expected = {"density_multiplier", "scattering_coefficient", "scattering_color",
                    "absorption_coefficient", "override_anisotropy", "anisotropy",
                    "emission_color", "emission_strength", "scattering_saturation",
                    "emission_saturation", "emission_exposure_ev"}
        assert expected <= original["material"].keys()
        if not args.apply:
            print("PASS material schema; no scene edits. Use --apply for live acceptance.")
            return
        layers = [layer for layer in original["layers"] if layer["enabled"]]
        assert layers, "Enable a volumetric cloud layer before GPU density acceptance"
        layer = layers[0]
        size = layer["cell_size_m"]
        points = [[x * size, layer["base_altitude_m"] + h * layer["thickness_m"], z * size]
                  for x in range(-12, 13) for z in range(-12, 13)
                  for h in (0.15, 0.35, 0.55)]
        neutral = {"density_multiplier": 1.0, "scattering_coefficient": 1.0,
                   "scattering_color": [1.0, 1.0, 1.0], "absorption_coefficient": 0.0,
                   "override_anisotropy": False, "emission_strength": 0.0,
                   "scattering_saturation": 1.0, "emission_saturation": 1.0,
                   "emission_exposure_ev": 0.0}
        edited = True
        ipc.call("world.set_clouds", material=neutral)
        baseline = density(ipc, points, backend=args.backend)
        assert any(v > 1e-6 for v in baseline), "No clouds at probe points; choose another view"
        assert all(math.isfinite(v) and v >= 0 for v in baseline)
        ipc.call("world.set_clouds", material={"scattering_saturation": 0.0,
                                              "emission_saturation": 2.0,
                                              "emission_exposure_ev": 16.0})
        graded_density = density(ipc, points, backend=args.backend)
        assert all(math.isclose(b, g, rel_tol=2e-5, abs_tol=1e-7)
                   for b, g in zip(baseline, graded_density)), "Grading changed extinction"
        ipc.call("world.set_clouds", material={"density_multiplier": 2.0})
        doubled = density(ipc, points, backend=args.backend)
        assert all(math.isclose(b * 2, d, rel_tol=2e-5, abs_tol=1e-7)
                   for b, d in zip(baseline, doubled))
        ipc.call("world.set_clouds", material={"density_multiplier": 1.0,
                                              "absorption_coefficient": 1.0})
        absorbing = density(ipc, points, backend=args.backend)
        assert all(math.isclose(b * 2, a, rel_tol=2e-5, abs_tol=1e-7)
                   for b, a in zip(baseline, absorbing))
        base = density(ipc, points, "base_density", args.backend)
        assert all(0 <= v <= b + 1e-7 for v, b in zip(absorbing, base))
        authored = {"scattering_color": [0.8, 0.7, 0.6], "override_anisotropy": True,
                    "anisotropy": -0.3, "emission_color": [0.4, 0.5, 0.6],
                    "emission_strength": 0.25, "scattering_saturation": 0.7,
                    "emission_saturation": 1.2, "emission_exposure_ev": 1.0}
        ipc.call("world.set_clouds", material=authored)
        current = ipc.call("world.get_clouds")
        for key, value in authored.items():
            actual = current["material"][key]
            if isinstance(value, list):
                assert all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(actual, value))
            elif isinstance(value, bool):
                assert actual is value
            else:
                assert math.isclose(actual, value, abs_tol=1e-6)
        # Existing merge-patch reset semantics remain intact.
        ipc.call("world.set_clouds", material={"emission_exposure_ev": None})
        reset = ipc.call("world.get_clouds")["material"]
        assert reset["emission_exposure_ev"] == 0.0
        assert math.isclose(reset["scattering_saturation"], 0.7, abs_tol=1e-6)
        local_grade = {"scattering_saturation": 0.4, "emission_saturation": 1.5,
                       "emission_exposure_ev": -1.0}
        ipc.call("world.set_clouds", layers=[{"material": local_grade}])
        local = ipc.call("world.get_clouds")["layers"][0]["material"]
        for key, value in local_grade.items():
            assert math.isclose(local[key], value, abs_tol=1e-6)
        before_layers = copy.deepcopy(ipc.call("world.get_clouds")["layers"])
        try:
            ipc.call("world.set_clouds", layers=[{"material": {"emission_exposure_ev": 17}}])
        except RtIpcError:
            pass
        else:
            raise AssertionError("Invalid layer grading accepted")
        assert ipc.call("world.get_clouds")["layers"] == before_layers
        for invalid in ({"density_multiplier": -1}, {"absorption_coefficient": -0.1},
                        {"scattering_color": [1, 2, 1]}, {"scattering_color": [1, 1]},
                        {"anisotropy": 1.0}, {"emission_strength": -1},
                        {"scattering_saturation": -0.01}, {"scattering_saturation": 2.01},
                        {"emission_saturation": -1}, {"emission_saturation": 3},
                        {"emission_exposure_ev": -17}, {"emission_exposure_ev": 17},
                        {"scattering_saturation": "grey"},
                        {"emission_saturation": []}, {"emission_exposure_ev": True},
                        {"scattering_saturation": 0.5, "emission_exposure_ev": 17}):
            before = copy.deepcopy(ipc.call("world.get_clouds")["material"])
            try:
                ipc.call("world.set_clouds", material=invalid)
            except RtIpcError:
                pass
            else:
                raise AssertionError(f"Invalid patch accepted: {invalid}")
            assert ipc.call("world.get_clouds")["material"] == before
        ipc.call("world.set_clouds", material={"density_multiplier": 0})
        assert not ipc.call("world.get_clouds")["any_enabled"]
        assert all(v == 0 for v in density(ipc, points, backend=args.backend))
        print("PASS GPU scaling, density bound, material round-trip and atomic rejection.")
    finally:
        try:
            if edited and original is not None:
                ipc.call("world.set_clouds", **original)
                restored = ipc.call("world.get_clouds")
                for key in ("material", "layers", "cirrus", "weather", "quality",
                            "derive_from_climate", "precipitation"):
                    assert restored[key] == original[key], f"Restoration failed: {key}"
                print("Original cloud settings restored; accumulation restarted.")
        finally:
            ipc.close()


if __name__ == "__main__":
    main()
