"""External material schema/atomic-rejection probe. Default is read-only.

After the user builds and opens Studio, pass --target from list_materials.
--apply tests writes and restores the material in finally; resets accumulation.
"""

import argparse
from rt_ipc import RtIpc, RtIpcError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    ipc = RtIpc()
    original = None
    edited = False
    try:
        listing = ipc.call("volume.list_materials")
        print(listing)
        if not args.target:
            assert not args.apply, "--apply requires --target"
            return
        original = ipc.call("volume.get_material", target=args.target)
        assert {"response_enabled", "temperature_power", "density_power",
                "emission_exposure_ev", "emission_saturation", "internal_light_steps",
                "internal_light_stride", "density_curve"} <= original.keys()
        if not args.apply:
            print("PASS read-only material schema")
            return
        edited = True
        current = ipc.call("volume.set_material", target=args.target, settings={
            "response_enabled": True, "temperature_power": 2.0,
            "density_power": 0.5, "emission_exposure_ev": 1.0,
            "internal_light_steps": 0, "internal_light_stride": 4,
            "density_curve": {"enabled": True,
                "values": [0, .03, .1, .25, .45, .65, .85, 1]}})
        assert current["temperature_power"] == 2.0
        assert current["density_power"] == 0.5
        assert current["internal_light_steps"] == 0
        assert current["density_curve"]["enabled"]
        assert current["scatter_color"] == original["scatter_color"]
        for invalid in ({"density_curve": {"enabled": True, "values": [0, .5, .1, .3, .4, .5, .6, 1]}},
                        {"density_power": 5}, {"emission_saturation": True},
                        {"unknown": 1}, {"temperature_power": -1},
                        {"preset": "physical_cloud", "emission_exposure_ev": 17},
                        {"internal_light_steps": 33}, {"internal_light_stride": 0}):
            before = ipc.call("volume.get_material", target=args.target)
            try:
                ipc.call("volume.set_material", target=args.target, settings=invalid)
            except RtIpcError:
                pass
            else:
                raise AssertionError(f"Invalid material accepted: {invalid}")
            assert ipc.call("volume.get_material", target=args.target) == before
        print("PASS material overlay and atomic rejection")
    finally:
        try:
            if edited and original is not None:
                ipc.call("volume.set_material", target=args.target, settings=original)
                assert ipc.call("volume.get_material", target=args.target) == original
        finally:
            ipc.close()


if __name__ == "__main__":
    main()
