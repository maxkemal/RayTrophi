"""Closed-tank count regression. Run externally with the timeline paused.

Requires an otherwise empty scene and a paused timeline. Creates a falling
block and steps without changing the timeline frame or clearing the cache.
Reports every step, including the first count change and reseed accounting.
"""
import argparse
import json
import time
import uuid
from pathlib import Path

from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--ppc", type=int, default=8)
    parser.add_argument("--backend", choices=("cpu", "vulkan"), default="vulkan")
    args = parser.parse_args()
    if args.steps < 1 or args.ppc < 1:
        parser.error("steps and ppc must be positive")
    client = RtIpc()
    name = "Conservation_" + uuid.uuid4().hex[:8]
    report = {"backend": args.backend, "ppc": args.ppc, "steps": []}
    created = False
    errors = []
    path = Path(__file__).resolve().parents[2] / ".tmp" / "closed_tank_conservation.json"
    try:
        report["frame_before"] = client.call("timeline.get_frame")
        time.sleep(0.25)
        assert client.call("timeline.get_frame") == report["frame_before"], "pause the timeline"
        domains = client.call("fluid.list_domains")["domains"]
        assert not domains, "use an otherwise empty scene; fluid.step advances all systems"
        client.call("fluid.create_domain", name=name, type="fluid",
                    domain_min=[20, 0, -1], domain_max=[22, 3, 1], voxel_size=0.08)
        created = True
        client.call("fluid.set_param", domain=name, backend=args.backend,
                    boundary="closed", preset="water", visible=False)
        client.call("fluid.set_whitewater", domain=name, enabled=False)
        client.call("fluid.seed", domain=name, seed_min=[20.55, 2, -0.45],
                    seed_max=[21.45, 2.6, 0.45], particles_per_cell=args.ppc,
                    replace=True, persistent=False)
        initial = client.call("fluid.get", domain=name)["particle_count"]
        assert initial > 0, "seed was empty"
        report["initial"] = initial
        previous = initial
        for step in range(1, args.steps + 1):
            assert client.call("timeline.get_frame") == report["frame_before"], "timeline advanced"
            client.call("fluid.step", dt=1 / 60)
            info = client.call("fluid.get", domain=name)
            row = {"step": step, "count": info["particle_count"],
                   "added": info["reseed_added_particles"],
                   "removed": info["reseed_removed_particles"]}
            row["delta"] = row["count"] - previous
            row["accounted"] = row["delta"] == row["added"] - row["removed"]
            report["steps"].append(row)
            if row["count"] != initial and "first_change" not in report:
                report["first_change"] = row
                print("FIRST CHANGE", row, flush=True)
            if step % 20 == 0 or step == 1:
                print(row, flush=True)
            previous = row["count"]
        report["passed"] = all(row["count"] == initial for row in report["steps"])
    finally:
        def attempt(action):
            try:
                action()
            except Exception as exc:
                errors.append(str(exc))
        if created:
            attempt(lambda: client.call("fluid.remove_domain", domain=name))
        try:
            report["frame_unchanged"] = (
                client.call("timeline.get_frame") == report.get("frame_before"))
        except Exception as exc:
            errors.append(str(exc))
        client.close()
        report["cleanup_errors"] = errors
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Report:", path, flush=True)
    assert not errors, errors
    assert report["frame_unchanged"], "timeline advanced during the test"
    assert report["passed"], "closed-tank particle count changed; see report"


if __name__ == "__main__":
    main()
