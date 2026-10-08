"""Particle state labels must survive the DISK bake (SimCache v6).

Bakes a temporary liquid domain to disk, drops the RAM timeline frames while
keeping the disk bake bound (sim_cache.clear ram_only), then reads a baked frame
back. That read can only come from disk, so the labels it reports are the ones
the file carries.

Silent failure this guards against: before v6 the reader zeroed flags, so a
disk-replayed frame reported every parcel as `unknown`. The RAM cache carries
flags, so a test that does not drop RAM first passes on a broken file.

Run with the timeline paused. Temporarily disables pre-existing enabled domains
and restores them; removes only its own domain and cache directory.
"""

import json
import shutil
import time
import uuid
from pathlib import Path

from rt_ipc import RtIpc
from rt_test_fluid_labels_ipc import check_report

BAKE_END = 6
READ_FRAME = 5


def main():
    client = RtIpc()
    name = "LabelCache_" + uuid.uuid4().hex[:10]
    cache_dir = (Path(__file__).resolve().parents[2] / ".tmp" / ("label_cache_" + name)).as_posix()
    report_path = Path(__file__).resolve().parents[2] / ".tmp" / "fluid_labels_cache.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"domain": name, "cache_dir": cache_dir}
    disabled, created, baked = [], False, False
    try:
        results["frame_before"] = client.call("timeline.get_frame")
        for domain in client.call("fluid.list_domains")["domains"]:
            if domain["enabled"]:
                client.call("fluid.set_param", domain=domain["name"], enabled=False)
                disabled.append(domain["name"])
        client.call("fluid.create_domain", name=name, type="fluid",
                    domain_min=[5.0, 0.0, 0.0], domain_max=[7.0, 2.0, 2.0], voxel_size=0.1)
        created = True
        client.call("fluid.set_param", domain=name, backend="vulkan",
                    boundary="closed", default_substance="Water", render_mode="surface")
        client.call("fluid.seed", domain=name, seed_min=[5.5, 0.5, 0.5],
                    seed_max=[6.5, 1.5, 1.5], particles_per_cell=4,
                    replace=True, persistent=True)

        client.call("sim_cache.bake", cache_dir=cache_dir, start_frame=0,
                    end_frame=BAKE_END, fps=24.0)
        baked = True
        status = client.call("sim_cache.status")
        results["status_after_bake"] = status
        assert status["valid"], "bake did not leave a valid cache bound"

        client.call("sim_cache.clear", ram_only=True)
        status = client.call("sim_cache.status")
        results["status_after_ram_drop"] = status
        assert status["valid"], "ram_only clear unbound the disk bake"
        assert status["ram_frames"] == 0, "RAM frames survived; the read would not test disk"

        client.call("timeline.set_frame", frame=READ_FRAME)
        time.sleep(0.4)
        info = client.call("fluid.get", domain=name)
        # require_classified=False: the classifier does not run on playback by
        # design (last_step stays at the bake's last live pass or empty). The
        # labels must come from the FILE; completeness is asserted below.
        labels = check_report(info, False)
        assert info["particle_labels"]["primary_complete"], "disk frame has unlabelled parcels"
        results["disk_frame"] = {"frame": READ_FRAME, "particle_count": info["particle_count"],
                                 "labels": labels}
        primary = info["particle_labels"]["primary"]
        assert info["particle_count"] > 0, "disk frame restored no particles"
        assert primary["unknown"] == 0, "disk frame lost labels (reader zeroed flags)"
        assert primary["body"] > 0, "dense liquid replayed without body labels"
        results["passed"] = True
        print("PASS disk frame", READ_FRAME, json.dumps(primary), flush=True)
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
        if baked:
            attempt(lambda: client.call("sim_cache.clear"))
        if created:
            attempt(lambda: client.call("fluid.remove_domain", domain=name))
        for domain in disabled:
            attempt(lambda d=domain: client.call("fluid.set_param", domain=d, enabled=True))
        if "frame_before" in results:
            attempt(lambda: client.call("timeline.set_frame", frame=results["frame_before"]))
        client.close()
        shutil.rmtree(cache_dir, ignore_errors=True)
        results["cleanup_errors"] = cleanup_errors
        report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("Report:", report_path, "cleanup errors:", cleanup_errors, flush=True)
        if cleanup_errors:
            raise RuntimeError("test cleanup failed: " + "; ".join(cleanup_errors))


if __name__ == "__main__":
    main()
