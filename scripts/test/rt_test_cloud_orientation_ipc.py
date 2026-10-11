"""External orientation acceptance after building C++ and all cloud shaders.

Default checks the schema without editing. --apply checks angle round-trips,
full-turn GPU equivalence and rejection, then restores settings. Accumulation
restarts. Run from a separate process, never inside Studio.
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
        assert all("rotation_degrees" in l for l in original["layers"])
        assert "rotation_degrees" in original["cirrus"]
        assert "towering_cumulus" in original["presets"]
        if not args.apply:
            print("PASS layer/cirrus orientation schema and towering preset; no scene edits.")
            return
        enabled = [i for i, l in enumerate(original["layers"]) if l["enabled"]]
        assert enabled, "Enable a volumetric cloud before GPU orientation acceptance"
        index = enabled[0]
        layer = original["layers"][index]
        size = layer["cell_size_m"]
        points = [[x * size, layer["base_altitude_m"] + h * layer["thickness_m"], z * size]
                  for x in range(-8, 9) for z in range(-8, 9) for h in (0.2, 0.4)]

        def set_angle(angle):
            patch = [{} for _ in range(index + 1)]
            patch[index] = {"rotation_degrees": angle}
            ipc.call("world.set_clouds", layers=patch, cirrus={"rotation_degrees": angle})

        def values():
            return ipc.call("world.sample_clouds", points=points, mode="density",
                            backend=args.backend)["values"]

        edited = True
        set_angle(0)
        baseline = values()
        assert any(v > 1e-6 for v in baseline), "No cloud at probe points"
        set_angle(360)
        full_turn = values()
        assert all(math.isfinite(v) and v >= 0 for v in full_turn)
        # float sine/cosine and texture coordinates differ slightly at 2*pi.
        assert all(math.isclose(a, b, rel_tol=2e-3, abs_tol=2e-5)
                   for a, b in zip(baseline, full_turn))
        set_angle(37)
        current = ipc.call("world.get_clouds")
        assert current["layers"][index]["rotation_degrees"] == 37
        assert current["cirrus"]["rotation_degrees"] == 37
        turned = values()
        assert any(abs(a - b) > 1e-6 for a, b in zip(baseline, turned))
        for patch in ({"layers": [{"rotation_degrees": 361}]},
                      {"cirrus": {"rotation_degrees": -361}}):
            before = ipc.call("world.get_clouds")
            try:
                ipc.call("world.set_clouds", **patch)
            except RtIpcError:
                pass
            else:
                raise AssertionError(f"Out-of-range rotation accepted: {patch}")
            after = ipc.call("world.get_clouds")
            assert after["layers"] == before["layers"]
            assert after["cirrus"] == before["cirrus"]
        print("PASS full-turn equivalence, rotated field, angle round-trip and rejection.")
    finally:
        try:
            if edited and original is not None:
                ipc.call("world.set_clouds", **original)
                after = ipc.call("world.get_clouds")
                for key in ("layers", "cirrus", "material", "weather", "quality",
                            "derive_from_climate", "precipitation"):
                    assert after[key] == original[key], f"Restore failed: {key}"
                print("Cloud settings restored; accumulation restarted.")
        finally:
            ipc.close()


if __name__ == "__main__":
    main()
