"""State labels route parcels to views (Faz 2): a surface liquid's spray is splat.

Builds a temporary surface-mode liquid with a dense block (body) and a few
isolated parcels (spray), steps it, and checks through fluid.get:

  - default table: body parcels are in the sdf view, spray parcels in the
    splat view (views[].particles / views[].labels);
  - routing spray to hidden removes it from every view (hidden_particles);
  - reset restores the default table;
  - unknown may only follow, and a bad name fails without changing anything.

The per-view counts come from the same per-parcel rule the level set, the fog
and the splat bridge filter with (FluidViewPlan::viewForParticle), so this
checks the routing decision, not the picture. Silent failure it guards
against: spray labelled correctly but still built into the surface, which
reads as "labels work, nothing changed on screen".

Run with the timeline paused. Temporarily disables pre-existing enabled
domains and restores them; removes only its own domain.
"""

import json
import uuid
from pathlib import Path

from rt_ipc import RtIpc


def views_by_name(info):
    return {v["view"]: v for v in info["views"]}


def main():
    client = RtIpc()
    name = "LabelViews_" + uuid.uuid4().hex[:10]
    report_path = Path(__file__).resolve().parents[2] / ".tmp" / "fluid_label_views.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"domain": name, "cases": []}
    disabled, created = [], False

    def snapshot(case):
        info = client.call("fluid.get", domain=name)
        labels = info["particle_labels"]["primary"]
        entry = {"case": case, "particle_count": info["particle_count"],
                 "labels": labels, "label_routes": info["label_routes"],
                 "hidden_particles": info["hidden_particles"],
                 "views": [{k: v[k] for k in ("view", "labels", "particles", "live")}
                           for v in info["views"]]}
        results["cases"].append(entry)
        print(case, json.dumps(entry), flush=True)
        return info

    try:
        for domain in client.call("fluid.list_domains")["domains"]:
            if domain["enabled"]:
                client.call("fluid.set_param", domain=domain["name"], enabled=False)
                disabled.append(domain["name"])
        client.call("fluid.create_domain", name=name, type="fluid",
                    domain_min=[5.0, 0.0, 0.0], domain_max=[7.0, 2.0, 2.0], voxel_size=0.1)
        created = True
        client.call("fluid.set_param", domain=name, backend="vulkan",
                    boundary="closed", default_substance="Water", render_mode="surface")
        client.call("fluid.seed", domain=name, seed_min=[5.2, 0.1, 0.2],
                    seed_max=[6.0, 0.9, 1.8], particles_per_cell=4,
                    replace=True, persistent=False)
        # Isolated parcels far from the block and from each other.
        for x in (6.5, 6.8):
            client.call("fluid.seed", domain=name, seed_min=[x, 1.5, 1.0],
                        seed_max=[x + 0.01, 1.51, 1.01], particles_per_cell=1,
                        replace=False, persistent=False)
        client.call("fluid.step", dt=1.0 / 60.0)

        info = snapshot("default_routes")
        labels = info["particle_labels"]["primary"]
        assert info["label_routes"]["spray"] == "splat", info["label_routes"]
        assert info["label_routes"]["unknown"] == "follow"
        assert labels["body"] > 0, "no body parcels: the dense block did not classify"
        assert labels["spray"] > 0, "no spray parcels: isolated seeds did not classify"
        views = views_by_name(info)
        assert "sdf" in views and views["sdf"]["particles"] == labels["body"] + labels["frozen"], \
            "sdf view is not exactly the body parcels"
        assert "splat" in views, "spray is live but no splat view was allocated"
        assert views["splat"]["particles"] == labels["spray"], "splat view is not the spray"
        assert "spray" in views["splat"]["labels"]
        assert "spray" not in views["sdf"]["labels"], "spray still builds the surface"

        client.call("fluid.set_label_views", domain=name, routes={"spray": "hidden"})
        info = snapshot("spray_hidden")
        views = views_by_name(info)
        assert info["hidden_particles"] == labels["spray"], "hidden count != spray"
        assert views.get("splat", {"particles": 0})["particles"] == 0
        assert views["sdf"]["particles"] == labels["body"] + labels["frozen"]

        client.call("fluid.set_label_views", domain=name, reset=True)
        info = snapshot("reset")
        assert info["label_routes"]["spray"] == "splat" and info["hidden_particles"] == 0

        for bad in ({"unknown": "splat"}, {"spray": "sideways"}, {"splash": "splat"}):
            try:
                client.call("fluid.set_label_views", domain=name, routes=bad)
            except Exception as error:
                results.setdefault("rejected", []).append({"routes": bad, "error": str(error)})
                continue
            raise AssertionError("accepted an invalid route: %s" % bad)
        info = snapshot("after_rejections")
        assert info["label_routes"]["spray"] == "splat", "a rejected call changed the table"
        results["passed"] = True
        print("PASS", flush=True)
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
        client.close()
        results["cleanup_errors"] = cleanup_errors
        report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("Report:", report_path, "cleanup errors:", cleanup_errors, flush=True)
        if cleanup_errors:
            raise RuntimeError("test cleanup failed: " + "; ".join(cleanup_errors))


if __name__ == "__main__":
    main()
