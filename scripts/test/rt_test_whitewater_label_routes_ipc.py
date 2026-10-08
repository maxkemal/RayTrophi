"""Faz 2-W / W1: one table routes BOTH the solver's labels and the whitewater.

Before W1 the whitewater had its own render_mode (spheres | volume) while the
Particle State Views table routed only the solver's spray / foam / bubble
parcels: two authorities for the same names, and the panel showed the table.
Now the label routes decide where each whitewater type is drawn too.

Checks, on a temporary dam break with whitewater on:
  1. defaults (spray/foam/bubble -> splat): every live whitewater particle is
     in splat, and fluid.get views[splat].whitewater agrees.
  2. all three -> hidden: every whitewater particle is hidden, no view reports
     whitewater. (Nothing is resimulated: routes are render-only, so the same
     particles are counted in every configuration.)
  3. spray hidden / foam sdf / bubble fog: fluid.get_whitewater views report
     exactly that, the per-view counts add up to alive, and a fog volume is
     published when whitewater is routed there.
  4. set_whitewater render_mode is REJECTED (removed key), nothing changes.

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
    args = parser.parse_args()
    client = RtIpc()
    name = "WWRoutes_" + uuid.uuid4().hex[:8]
    report_path = Path(__file__).resolve().parents[2] / ".tmp" / "whitewater_label_routes.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"domain": name, "frames": args.frames, "checks": {}}
    disabled, created = [], False

    def settle():
        # Routes are applied by the render resync of the next frames.
        time.sleep(0.6)

    def snapshot(label):
        ww = client.call("fluid.get_whitewater", domain=name)
        info = client.call("fluid.get", domain=name)
        views = {v["view"]: v for v in info.get("views", [])}
        entry = {"views": ww["views"], "stats": ww["stats"],
                 "fluid_views": {k: {"particles": v.get("particles"),
                                     "whitewater": v.get("whitewater"),
                                     "vdb_id": v.get("vdb_id")} for k, v in views.items()}}
        results[label] = entry
        print(label, json.dumps(entry["views"]), json.dumps(
            {k: ww["stats"][k] for k in ("alive", "spray", "foam", "bubble",
                                         "in_sdf", "in_splat", "in_fog", "hidden")}), flush=True)
        return ww, views

    def check(key, ok, detail=""):
        results["checks"][key] = {"ok": bool(ok), "detail": detail}
        print(("PASS " if ok else "FAIL ") + key + (" - " + detail if detail else ""), flush=True)
        return ok

    try:
        results["frame_before"] = client.call("timeline.get_frame")
        for domain in client.call("fluid.list_domains")["domains"]:
            if domain["enabled"]:
                client.call("fluid.set_param", domain=domain["name"], enabled=False)
                disabled.append(domain["name"])
        client.call("fluid.create_domain", name=name, type="fluid",
                    domain_min=[5.0, 0.0, -1.0], domain_max=[8.0, 2.0, 1.0], voxel_size=0.08)
        created = True
        client.call("fluid.set_param", domain=name, backend="vulkan", boundary="closed",
                    default_substance="Water", render_mode="surface")
        client.call("fluid.seed", domain=name, seed_min=[5.05, 0.05, -0.95],
                    seed_max=[6.0, 1.6, 0.95], particles_per_cell=4,
                    replace=True, persistent=True)
        client.call("fluid.set_whitewater", domain=name, enabled=True,
                    trapped_air_rate=200.0, wave_crest_rate=200.0)
        client.call("fluid.set_label_views", domain=name, routes={}, reset=True)

        client.call("sim_cache.clear", ram_only=True)
        client.call("timeline.set_frame", frame=0)
        time.sleep(0.3)
        for f in range(1, args.frames + 1):
            client.call("timeline.set_frame", frame=f)
        settle()

        # 1. defaults
        ww, views = snapshot("defaults")
        st = ww["stats"]
        alive = st["alive"]
        if not check("whitewater_ran", alive > 0, "alive=%d" % alive):
            raise RuntimeError("whitewater produced no particles: the routing checks prove nothing")
        check("defaults_all_splat",
              ww["views"] == {"spray": "splat", "foam": "splat", "bubble": "splat"}
              and st["in_splat"] == alive, json.dumps(ww["views"]))
        check("defaults_fluid_get_agrees",
              views.get("splat", {}).get("whitewater") == alive,
              "views[splat].whitewater=%s alive=%d" % (views.get("splat", {}).get("whitewater"), alive))

        # 2. all hidden
        client.call("fluid.set_label_views", domain=name,
                    routes={"spray": "hidden", "foam": "hidden", "bubble": "hidden"})
        settle()
        ww, views = snapshot("all_hidden")
        st = ww["stats"]
        check("hidden_counts", st["hidden"] == st["alive"] and st["alive"] == alive,
              "hidden=%d alive=%d (before %d)" % (st["hidden"], st["alive"], alive))
        check("hidden_no_view_reports_whitewater",
              all((v.get("whitewater") or 0) == 0 for v in views.values()))

        # 3. mixed
        client.call("fluid.set_label_views", domain=name,
                    routes={"spray": "hidden", "foam": "sdf", "bubble": "fog"})
        settle()
        ww, views = snapshot("mixed")
        st = ww["stats"]
        check("mixed_views_reported",
              ww["views"] == {"spray": "hidden", "foam": "sdf", "bubble": "fog"}, json.dumps(ww["views"]))
        total = st["in_sdf"] + st["in_splat"] + st["in_fog"] + st["hidden"]
        check("mixed_counts_add_up", total == st["alive"] and st["in_splat"] == 0,
              "sdf=%d splat=%d fog=%d hidden=%d alive=%d" % (
                  st["in_sdf"], st["in_splat"], st["in_fog"], st["hidden"], st["alive"]))
        check("mixed_fluid_get_agrees",
              views.get("sdf", {}).get("whitewater") == st["in_sdf"] and
              views.get("fog", {}).get("whitewater") == st["in_fog"])
        if st["in_fog"] > 0:
            fog_id = views.get("fog", {}).get("vdb_id", -1)
            check("fog_volume_published_for_whitewater", fog_id is not None and fog_id >= 0,
                  "fog vdb_id=%s with %d bubbles routed" % (fog_id, st["in_fog"]))
        else:
            results["checks"]["fog_volume_published_for_whitewater"] = {
                "ok": None, "detail": "no live bubble: not measured"}
            print("SKIP fog_volume_published_for_whitewater - no live bubble", flush=True)

        # 4. removed key
        rejected = False
        try:
            client.call("fluid.set_whitewater", domain=name, render_mode="volume")
        except Exception as error:
            rejected = "render_mode was removed" in str(error)
            results["render_mode_error"] = str(error)
        after = client.call("fluid.get_whitewater", domain=name)
        check("render_mode_rejected", rejected and "render_mode" not in after)

        failed = [k for k, v in results["checks"].items() if v["ok"] is False]
        results["passed"] = not failed
        if failed:
            raise AssertionError("failed: " + ", ".join(failed))
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
