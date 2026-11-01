"""Post-build atomic mask API probe. Run in an external terminal with Studio open."""
import argparse
from rt_ipc import RtIpc, RtIpcError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    client = RtIpc()
    original = client.call("volume.get_material", target=args.target)
    keys = ("emission_mask_enabled", "emission_mask_center",
            "emission_mask_width", "emission_mask_strength")
    if not all(k in original for k in keys):
        client.close()
        raise RuntimeError("Binary does not expose density emission masks")
    restore = {k: original[k] for k in keys}
    try:
        patch = dict(zip(keys, (True, 0.25, 0.125, 0.75)))
        client.call("volume.set_material", target=args.target, settings=patch)
        expected = dict(original, **patch)
        assert client.call("volume.get_material", target=args.target) == expected
        for patch in ({"emission_mask_enabled": 1}, {"emission_mask_center": -1},
                      {"emission_mask_width": 0}, {"emission_mask_width": -1},
                      {"emission_mask_strength": 1.01}, {"emission_mask_strength": -0.1},
                      {"emission_mask_center": 0.5, "emission_mask_width": 0}):
            try:
                client.call("volume.set_material", target=args.target, settings=patch)
            except RtIpcError:
                pass
            else:
                raise AssertionError("Invalid mask patch accepted: " + repr(patch))
            assert client.call("volume.get_material", target=args.target) == expected
        print("PASS: mask fields round-trip; invalid multi-field edits reject atomically")
    finally:
        try:
            client.call("volume.set_material", target=args.target, settings=restore)
            assert client.call("volume.get_material", target=args.target) == original
            print("Original material restored; project not saved")
        finally:
            client.close()


if __name__ == "__main__":
    main()
