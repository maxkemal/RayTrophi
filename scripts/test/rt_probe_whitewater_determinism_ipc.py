"""Faz 2-W gate: does enabling whitewater change the PRIMARY liquid?

Three timeline runs of the same temporary dam break (fluid.state_digest at
the last frame):

  A  whitewater off
  B  whitewater off again      -> is the solver itself run-to-run deterministic?
  C  whitewater on             -> are the primary parcels untouched by it?

Reading the result:
  A == B (hashes) : the solver is bit-deterministic; the W2 gate can be
                    "C hashes == A hashes" and any difference is a leak.
  A != B          : bit-exact gating is impossible on this path (e.g. float
                    atomics in P2G). The probe then reports the tolerant view
                    (centroid / mean speed drift) and the gate must be set
                    against the A-vs-B noise floor, not zero.
  C must also report whitewater_particles > 0, or the "no effect" is only
  "whitewater never ran".

Run with the timeline paused. Temporarily disables pre-existing enabled
domains and restores them; removes only its own domain. Restores frame.
"""

import argparse
import json
import time
import uuid
from pathlib import Path

from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--backend", choices=("cpu", "vulkan"), default="vulkan")
    args = parser.parse_args()
    client = RtIpc()
    name = "WWDeterminism_" + uuid.uuid4().hex[:8]
    report_path = (Path(__file__).resolve().parents[2] / ".tmp" /
                   ("whitewater_determinism_" + args.backend + ".json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"domain": name, "frames": args.frames, "backend": args.backend}
    disabled, created = [], False

    def run(label):
        # Drop RAM frames so the timeline re-simulates instead of replaying.
        client.call("sim_cache.clear", ram_only=True)
        client.call("timeline.set_frame", frame=0)
        time.sleep(0.3)
        for f in range(1, args.frames + 1):
            client.call("timeline.set_frame", frame=f)
        time.sleep(0.5)
        dg = client.call("fluid.state_digest", domain=name)
        ww = client.call("fluid.get_whitewater", domain=name)["stats"]
        results[label] = {"digest": dg, "whitewater_stats": ww}
        print(label, json.dumps(dg), "ww_alive", ww["alive"], flush=True)
        return dg

    try:
        results["frame_before"] = client.call("timeline.get_frame")
        for domain in client.call("fluid.list_domains")["domains"]:
            if domain["enabled"]:
                client.call("fluid.set_param", domain=domain["name"], enabled=False)
                disabled.append(domain["name"])
        client.call("fluid.create_domain", name=name, type="fluid",
                    domain_min=[5.0, 0.0, -1.0], domain_max=[8.0, 2.0, 1.0], voxel_size=0.08)
        created = True
        client.call("fluid.set_param", domain=name, backend=args.backend, boundary="closed",
                    preset="water", render_mode="surface", visible=False)
        client.call("fluid.seed", domain=name, seed_min=[5.05, 0.05, -0.95],
                    seed_max=[6.0, 1.6, 0.95], particles_per_cell=4,
                    replace=True, persistent=True)
        client.call("fluid.set_whitewater", domain=name, enabled=False)

        a = run("A_off")
        b = run("B_off_again")
        client.call("fluid.set_whitewater", domain=name, enabled=True)
        c = run("C_on")

        same = lambda x, y: (x["particles"] == y["particles"] and
                             x["position_hash"] == y["position_hash"] and
                             x["velocity_hash"] == y["velocity_hash"])
        drift = lambda x, y: max(abs(x["centroid"][i] - y["centroid"][i]) for i in range(3))
        results["solver_deterministic"] = same(a, b)
        results["primary_unchanged_by_whitewater"] = same(a, c)
        results["noise_floor_centroid"] = drift(a, b)
        results["whitewater_centroid_drift"] = drift(a, c)
        results["whitewater_ran"] = c["whitewater_particles"] > 0
        print("solver_deterministic", results["solver_deterministic"],
              "| primary_unchanged_by_whitewater", results["primary_unchanged_by_whitewater"],
              "| centroid drift A-B %.3g A-C %.3g" % (results["noise_floor_centroid"],
                                                     results["whitewater_centroid_drift"]),
              "| whitewater_ran", results["whitewater_ran"], flush=True)
        assert results["whitewater_ran"], "whitewater produced no particles: the C run proves nothing"
        results["passed"] = True
    except Exception as error:
        results["error"] = str(error)
        print("FAIL", error, flush=True)
        raise
    finally:
        cleanup_errors = []

        def attempt(fn):
            try:
                fn()
            except Exception as error:
                cleanup_errors.append(str(error))
        if created:
            attempt(lambda: client.call("fluid.remove_domain", domain=name))
        for domain in disabled:
            attempt(lambda d=domain: client.call("fluid.set_param", domain=d, enabled=True))
        if "frame_before" in results:
            attempt(lambda: client.call("timeline.set_frame", frame=results["frame_before"]))
        client.close()
        results["cleanup_errors"] = cleanup_errors
        report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("Report:", report_path, "cleanup errors:", cleanup_errors, flush=True)
        if cleanup_errors:
            raise RuntimeError("test cleanup failed: " + "; ".join(cleanup_errors))


if __name__ == "__main__":
    main()
