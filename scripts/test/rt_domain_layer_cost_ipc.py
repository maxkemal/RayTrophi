"""Per-volume cost split for a domain the camera looks into (Vulkan RT).

    python scripts/test/rt_domain_layer_cost_ipc.py [--output file.json]

For every volume material target (volume.list_materials) it measures:
  hidden:<t>  the target is not published at all (no TLAS box)
  inert:<t>   published but density_multiplier 0 - the intersection shader
              should reject it from metadata (volumeOpticallyInert)
plus `baseline` and `all_hidden`. The pairs separate three costs that look
the same on screen:
  baseline - inert  = marching the content (density, lighting, shadows)
  inert - hidden    = merely having the box in the TLAS (intersection shader
                      calls; non-zero means inertness is NOT being detected)
  hidden - all_hidden = the other volumes

Wall ms per natural Vulkan RT sample (8 -> 40, counters off), then a short
counted window. Restores every material (incl. visible) in `finally`.
Never calls viewport.render_frames (CPU path).
"""
import argparse
import json
import os
import tempfile
import time

from rt_ipc import RtIpc


def wait_samples(ipc, minimum, frame, timeout=120):
    deadline = time.perf_counter() + timeout
    while True:
        assert ipc.call("timeline.get_frame") == frame, "Timeline changed during probe"
        status = ipc.call("viewport.status")
        if status["samples"] >= minimum:
            return time.perf_counter(), status["samples"]
        if status["accumulation_complete"]:
            raise RuntimeError("Accumulation finished before the measurement window")
        if time.perf_counter() > deadline:
            raise RuntimeError("RT sample progress timed out")
        time.sleep(0.015)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=os.path.join(tempfile.gettempdir(),
                                                         "rt_domain_layer_cost.json"))
    args = parser.parse_args()
    ipc = RtIpc()
    originals = {}
    prior = None
    report = {"rows": []}
    try:
        assert ipc.call("viewport.shading")["mode"] == "rendered", "Rendered viewport required"
        assert ipc.call("render.rt_pipeline_status")["rt_pipeline_state"] == "ready"
        frame = ipc.call("timeline.get_frame")
        prior = ipc.call("render.volume_stats")
        # ★ Skip vdb:<id> rows. Live simulation volumes (Matter streams, liquid
        # domains) re-register on every republish and get a NEW id, so the
        # second case cannot find the first one's target. Matter streams are
        # covered by their stable matter:<domain>:<field> targets; a liquid
        # SurfaceSDF has no stable material target and stays on in every case.
        rows = ipc.call("volume.list_materials")["targets"]
        targets = [t["target"] for t in rows
                   if t["has_material"] and not t["target"].startswith("vdb:")]
        for t in targets:
            originals[t] = ipc.call("volume.get_material", target=t)
        report.update(frame=frame, targets=targets, slots=ipc.call("render.volume_slots"))

        def apply(patches):
            for t in targets:
                s = dict(originals[t])
                s.update(patches.get(t, {}))
                ipc.call("volume.set_material", target=t, settings=s)

        cases = [("baseline", {})]
        for t in targets:
            cases.append(("hidden:" + t, {t: {"visible": False}}))
            cases.append(("inert:" + t, {t: {"density_multiplier": 0.0}}))
        cases.append(("all_hidden", {t: {"visible": False} for t in targets}))
        cases.append(("baseline_repeat", {}))
        for label, patches in cases:
            ipc.call("render.volume_counters", enabled=False)
            apply(patches)
            start, s0 = wait_samples(ipc, 8, frame)
            end, s1 = wait_samples(ipc, s0 + 32, frame)
            row = {"case": label, "ms": (end - start) * 1000.0 / (s1 - s0)}
            ipc.call("render.volume_counters", enabled=True)
            _, c0 = wait_samples(ipc, 0, frame)
            wait_samples(ipc, c0 + 8, frame)
            st = ipc.call("render.volume_stats")
            paths = max(st["paths_traced"], 1)
            rays = max(st["volume_rays"], 1)
            row.update(volume_rays_per_path=st["volume_rays"] / paths,
                       density_per_volume_ray=st["density_samples"] / rays,
                       shadow_per_volume_ray=st["shadow_density_samples"] / rays,
                       probes_per_volume_ray=st["solid_probe_runs"] / rays,
                       probe_hits_per_volume_ray=st["solid_probe_hits"] / rays,
                       gas_handoffs_per_path=st["gas_handoffs"] / paths,
                       medium_passes_per_path=st["medium_passes"] / paths,
                       empty_skips_per_volume_ray=st["empty_segments_skipped"] / rays,
                       arbiter_samples_per_volume_ray=st["sdf_arbiter_samples"] / rays,
                       free_passes_per_path=st["free_passes"] / paths)
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
    finally:
        try:
            for t, m in originals.items():
                ipc.call("volume.set_material", target=t, settings=m)
            if prior is not None:
                ipc.call("render.volume_counters", enabled=prior["enabled"], region=prior["region"])
            report["restored"] = all(ipc.call("volume.get_material", target=t) == m
                                     for t, m in originals.items())
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(json.dumps({"restored": report.get("restored"), "output": args.output}))
        finally:
            ipc.close()


if __name__ == "__main__":
    main()
