"""Exercise label classification in a temporary liquid domain via external IPC.

Run with the timeline paused. Temporarily disables pre-existing enabled domains,
restores their enabled flags, and removes only the uniquely named test domain.
Does not change the timeline frame, save the project or launch the application.
"""

import argparse
import json
import time
import uuid
from pathlib import Path

from rt_ipc import RtIpc
from rt_domain_material import domain_material  # noqa: E402
from rt_test_fluid_labels_ipc import check_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true",
                        help="100,000 parcels / four timeline frames; needs the updated build")
    parser.add_argument("--backend", choices=("cpu", "vulkan"), default="vulkan")
    args = parser.parse_args()
    client = RtIpc()
    name = "LabelProbe_" + uuid.uuid4().hex[:10]
    disabled = []
    created = False
    results = {"domain": name, "backend": args.backend, "cases": []}
    output = Path((".tmp/fluid_labels_benchmark_" + args.backend + ".json")
                  if args.benchmark else ".tmp/fluid_labels_live.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    def record(case, require=True):
        info = client.call("fluid.get", domain=name)
        report = check_report(info, require)
        results["cases"].append({"case": case, "labels": report,
                                 "thermal_frozen": info["thermal_frozen_particles"],
                                 "render_mode": info["render_mode"]})
        print(case, json.dumps(results["cases"][-1]), flush=True)
        return info

    def step(count=1):
        for _ in range(count):
            client.call("fluid.step", dt=1.0 / 60.0)

    try:
        original = client.call("fluid.list_domains")["domains"]
        results["original_domains"] = [d["name"] for d in original]
        results["frame_before"] = client.call("timeline.get_frame")
        for domain in original:
            if domain["enabled"]:
                client.call("fluid.set_param", domain=domain["name"], enabled=False)
                disabled.append(domain["name"])
        client.call("fluid.create_domain", name=name, type="fluid",
                    domain_min=[5.0, 0.0, 0.0], domain_max=[7.0, 2.0, 2.0],
                    voxel_size=0.05 if args.benchmark else 0.1)
        created = True
        client.call("fluid.set_param", domain=name, backend=args.backend,
                    boundary="closed", default_substance="Water", render_mode="surface")
        if args.benchmark:
            client.call("fluid.set_param", domain=name, visible=False)
        client.call("fluid.seed", domain=name,
                    seed_min=[5.25, 0.25, 0.25] if args.benchmark else [5.5, 0.5, 0.5],
                    seed_max=[6.75, 1.75, 1.75] if args.benchmark else [6.5, 1.5, 1.5],
                    particles_per_cell=4,
                    replace=True, persistent=True)
        record("new_seed", require=False)
        for i in range(4 if args.benchmark else 8):
            if args.benchmark:
                client.call("timeline.set_frame", frame=results["frame_before"] + i + 1)
                time.sleep(0.4)
            else:
                step()
            info = record("dense_step_" + str(i + 1))
            assert info["particle_labels"]["primary"]["body"] > 0
            if args.benchmark:
                assert info["particle_count"] == 100000, "not the baseline workload"
                assert info["particle_labels"]["last_step"]["on_gpu"] == (
                    args.backend == "vulkan"), "unexpected classifier backend"
        if args.benchmark:
            results["step_stats"] = client.call("fluid.step_stats", domain=name)
            results["passed"] = True
            return
        listed = client.call("fluid.list_domains")["domains"]
        check_report(next(d for d in listed if d["name"] == name), True)

        before = client.call("fluid.get", domain=name)["particle_labels"]["primary"]
        for mode in ("particles", "fog", "surface"):
            client.call("fluid.set_param", domain=name, render_mode=mode)
            time.sleep(0.15)
            info = record("display_" + mode)
            assert info["particle_labels"]["primary"] == before, "display changed labels"

        client.call("fluid.seed", domain=name, seed_min=[6.0, 1.0, 1.0],
                    seed_max=[6.01, 1.01, 1.01], particles_per_cell=1,
                    replace=True, persistent=True)
        record("isolated_seed", require=False)
        step()
        info = record("isolated_step")
        assert info["particle_labels"]["primary"]["spray"] > 0

        material = domain_material(client.call, 'T: probe_fluid_labels_live', 'Water',
            thermal_freeze_kelvin=330.0, thermal_cold_viscosity=0.001)
        client.call("fluid.set_param", domain=name, thermal_liquid_enabled=True,
                    default_substance=material)
        client.call("fluid.seed", domain=name, seed_min=[5.5, 0.01, 0.5],
                    seed_max=[6.5, 0.19, 1.5], particles_per_cell=4,
                    replace=True, persistent=True)
        step(3)
        info = record("cold_supported")
        frozen = info["particle_labels"]["primary"]["frozen"]
        assert frozen > 0, "supported cold liquid did not produce frozen labels"
        assert frozen == info["thermal_frozen_particles"], "thermal/label disagreement"
        client.call("fluid.set_param", domain=name, thermal_liquid_enabled=False)
        step()
        info = record("thermal_disabled")
        assert info["particle_labels"]["primary"]["frozen"] == 0

        client.call("fluid.clear", domain=name, clear_seed=True)
        info = record("cleared", require=False)
        assert info["particle_count"] == 0
        results["passed"] = True
    except Exception as error:
        results["error"] = str(error)
        raise
    finally:
        cleanup_errors = []
        if created:
            try:
                client.call("fluid.remove_domain", domain=name)
            except Exception as error:
                cleanup_errors.append(str(error))
        if args.benchmark and "frame_before" in results:
            try:
                client.call("timeline.set_frame", frame=results["frame_before"])
            except Exception as error:
                cleanup_errors.append(str(error))
        for domain in disabled:
            try:
                client.call("fluid.set_param", domain=domain, enabled=True)
            except Exception as error:
                cleanup_errors.append(str(error))
        results["cleanup_errors"] = cleanup_errors
        try:
            results["frame_after"] = client.call("timeline.get_frame")
            results["remaining_domains"] = [d["name"] for d in
                client.call("fluid.list_domains")["domains"]]
        finally:
            client.close()
            output.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print("Report:", output, "cleanup errors:", cleanup_errors, flush=True)
        if cleanup_errors:
            raise RuntimeError("test cleanup failed: " + "; ".join(cleanup_errors))


if __name__ == "__main__":
    main()
