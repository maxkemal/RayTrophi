"""External cloud-region acceptance after C++/shader compilation.

Default is read-only schema inspection. --apply tests a storm preset and layer
material effects, then restores the original cloud state. It resets accumulation.
"""

import argparse
import math

from rt_ipc import RtIpc, RtIpcError


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
        required = {"formation_role", "formation_group", "use_custom_material", "material"}
        assert all(required <= layer.keys() for layer in original["layers"])
        if not args.apply:
            print("PASS layer formation/material schema; no scene edits.")
            return
        edited = True
        ipc.call("world.apply_cloud_preset", preset="storm")
        storm = ipc.call("world.get_clouds")
        layers = storm["layers"]
        assert all(l["enabled"] and l["formation_group"] == 1 for l in layers)
        assert [l["formation_role"] for l in layers] == [1, 2, 3]
        assert all(l["use_custom_material"] for l in layers)
        for i in range(2):
            assert layers[i]["base_altitude_m"] + layers[i]["thickness_m"] == \
                   layers[i + 1]["base_altitude_m"]
        ipc.call("world.set_clouds", material={"density_multiplier": 1,
                                              "scattering_coefficient": 1,
                                              "absorption_coefficient": 0})
        tower = layers[1]
        period = layers[0]["cell_size_m"]
        points = [[x * period * 0.25,
                   tower["base_altitude_m"] + h * tower["thickness_m"],
                   z * period * 0.25]
                  for x in range(-20, 21) for z in range(-20, 21)
                  for h in (0.1, 0.35, 0.65)]

        def values(mode="density"):
            return ipc.call("world.sample_clouds", points=points, mode=mode,
                            backend=args.backend)["values"]

        baseline = values()
        assert all(math.isfinite(v) and v >= 0 for v in baseline)
        assert any(v > 1e-6 for v in baseline), "No storm tower at probe points"
        ipc.call("world.set_clouds", layers=[{}, {"material": {"density_multiplier": 2}}])
        doubled = values()
        assert all(math.isclose(a * 2, b, rel_tol=3e-5, abs_tol=2e-7)
                   for a, b in zip(baseline, doubled))
        cheap = values("base_density")
        assert all(0 <= a <= b + 2e-7 for a, b in zip(doubled, cheap))
        # The shared material remains a master even with local modifiers.
        ipc.call("world.set_clouds", material={"density_multiplier": 0})
        assert not ipc.call("world.get_clouds")["any_enabled"]
        assert all(v == 0 for v in values())
        for invalid in ({"formation_role": 4}, {"formation_role": 1.5},
                        {"formation_group": -1},
                        {"material": {"density_multiplier": -1}},
                        {"material": {"scattering_color": [1, 2, 1]}}):
            before = ipc.call("world.get_clouds")
            try:
                ipc.call("world.set_clouds", layers=[{}, invalid])
            except RtIpcError:
                pass
            else:
                raise AssertionError(f"Invalid layer patch accepted: {invalid}")
            after = ipc.call("world.get_clouds")
            assert after["layers"] == before["layers"]
            assert after["material"] == before["material"]
        print("PASS linked regions, local GPU density, master control and atomic rejection.")
    finally:
        try:
            if edited and original is not None:
                ipc.call("world.set_clouds", **original)
                restored = ipc.call("world.get_clouds")
                for key in ("layers", "cirrus", "material", "weather", "quality",
                            "derive_from_climate", "precipitation"):
                    assert restored[key] == original[key], f"Restore failed: {key}"
                print("Cloud settings restored; accumulation restarted.")
        finally:
            ipc.close()


if __name__ == "__main__":
    main()
