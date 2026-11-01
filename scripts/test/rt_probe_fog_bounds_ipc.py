"""Read-only live Fog optical bounds diagnostic; run outside the app."""
import argparse
import json
import math
from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="Physics Domain 1")
    args = parser.parse_args()
    client = RtIpc()
    try:
        data = client.call("fluid.inspect_fog_fields", domain=args.domain)
        total = math.prod(data["grid"])
        for field in data["fields"]:
            bounds = field.get("optical_bounds")
            row = {"target": field["material_target"], "visible": field.get("visible"),
                   "raw_active_fraction": field["active_cells"] / max(total, 1),
                   "optical_bounds": bounds}
            if bounds:
                volume = 0 if bounds["empty"] else math.prod(
                    hi-lo for lo, hi in zip(bounds["first_voxel"], bounds["end_voxel"]))
                row["render_box_fraction"] = volume / max(total, 1)
            print(json.dumps(row), flush=True)
        slots = client.call("render.volume_slots")
        for backend in slots.get("backends", []):
            for slot in backend.get("slots", []):
                print(json.dumps({"backend": backend["role"], "slot": slot["slot"],
                    "density_device_local": slot.get("density_device_local"),
                    "temperature_device_local": slot.get("temperature_device_local")}), flush=True)
        print(json.dumps(slots), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
