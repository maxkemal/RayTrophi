"""Post-build material contract probe. Run externally with Studio open."""
import argparse

from rt_ipc import RtIpc, RtIpcError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True,
                        help="Exact target from volume.list_materials")
    args = parser.parse_args()
    client = RtIpc()
    original = client.call("volume.get_material", target=args.target)
    if "temperature_units" not in original:
        client.close()
        raise RuntimeError("This binary does not expose the new temperature policy")
    try:
        for units in (1, 2, 0):
            client.call("volume.set_material", target=args.target,
                        settings={"temperature_units": units})
            actual = client.call("volume.get_material", target=args.target)
            expected = dict(original, temperature_units=units)
            assert actual == expected, "Unexpected setting changed"
        before = client.call("volume.get_material", target=args.target)
        for invalid in (-1, 3, 1.5, True, "kelvin", 4294967296):
            try:
                client.call("volume.set_material", target=args.target,
                            settings={"temperature_units": invalid})
            except RtIpcError:
                pass
            else:
                raise AssertionError("Invalid unit policy accepted: " + repr(invalid))
            assert client.call("volume.get_material", target=args.target) == before
        print("PASS: unit modes round-trip; invalid patches reject atomically")
    finally:
        try:
            client.call("volume.set_material", target=args.target,
                        settings={"temperature_units": original["temperature_units"]})
            assert client.call("volume.get_material", target=args.target) == original
            print("Original material restored; project not saved")
        finally:
            client.close()


if __name__ == "__main__":
    main()
