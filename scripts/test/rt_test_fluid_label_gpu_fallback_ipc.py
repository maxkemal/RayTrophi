"""Legacy filename: dense (>64 per bin) linked-label GPU acceptance.

Use a separate empty paused scene. No fixed-slot overflow or CPU fallback
should be needed after building the three linked-label shader entry points.
"""
import json
import uuid
from pathlib import Path

from rt_ipc import RtIpc
from rt_test_fluid_labels_ipc import check_report


def main():
    client = RtIpc()
    name = "LabelOverflow_" + uuid.uuid4().hex[:8]
    path = Path(__file__).resolve().parents[2] / ".tmp" / "fluid_label_gpu_fallback.json"
    created = False
    result = {}
    errors = []
    try:
        assert client.call("fluid.list_domains")["domains"] == [], "use an empty scene"
        assert not client.call("sim.control_state")["playing"], "pause first"
        result["frame_before"] = client.call("timeline.get_frame")
        client.call("fluid.create_domain", name=name, type="fluid",
                    domain_min=[5, 0, 0], domain_max=[6, 1, 1], voxel_size=0.1)
        created = True
        client.call("fluid.set_param", domain=name, backend="vulkan",
                    boundary="closed", default_substance="Water", visible=False)
        client.call("fluid.seed", domain=name, seed_min=[5.05, 0.05, 0.05],
                    seed_max=[5.95, 0.95, 0.95], particles_per_cell=32,
                    replace=True, persistent=False)
        client.call("fluid.step", dt=1 / 60)
        info = client.call("fluid.get", domain=name)
        report = check_report(info, True)
        result.update({"particles": info["particle_count"],
                       "labels": report["primary"],
                       "last_step": report["last_step"]})
        assert report["last_step"]["on_gpu"] is True, (
            "linked classifier did not run; rebuild C++ and linked-label shaders")
        occupied = report["last_step"]["occupied_bins"]
        assert occupied > 0 and info["particle_count"] > 64 * occupied, (
            "fixture did not exceed the old 64-entry capacity", occupied)
        assert report["primary"]["unknown"] == 0
        result["passed"] = True
        print(json.dumps(result), flush=True)
    finally:
        if created:
            try:
                client.call("fluid.remove_domain", domain=name)
            except Exception as error:
                errors.append(str(error))
        try:
            result["frame_after"] = client.call("timeline.get_frame")
            result["remaining_domains"] = client.call("fluid.list_domains")["domains"]
        except Exception as error:
            errors.append(str(error))
        client.close()
        result["cleanup_errors"] = errors
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    assert not errors and not result["remaining_domains"]


if __name__ == "__main__":
    main()
