"""K0: procedural sphere cloud vs near-camera grain meshes, same particles.

docs/dev/KUM_TANE_GEOMETRISI_VE_MINERAL_KARISIMI.md section 6b. The question is
whether the cost of drawing grains as hardware triangle instances scales with
the INSTANCE count (expected) or with the triangle count (not expected), next
to the procedural sphere cloud's software intersection shader.

Read-only unless --apply. With --apply it:
  1. builds two exemplar libraries far below the scene (--library-y):
     - low-poly grains: --library boxes (default; flat cube primitives,
       12 triangles, baked aspect/tilt) or --library fracture (Voronoi
       shards; needs the build in which fracture emits flat meshes),
     - plain sphere primitives (same instance count, many more triangles);
  2. measures, on a PAUSED frame, with the RT pipeline ready and the viewport
     in Rendered mode:
       spheres            near meshes off (baseline)
       lowpoly_D          low-poly grains inside --distance
       lowpoly_4D         low-poly grains inside 4 x --distance (instance scaling)
       spheremesh_D       sphere-primitive library inside --distance
                          (same instances as lowpoly_D, more triangles)
       spheres_repeat     baseline again (drift check)
  3. restores the domain's near-mesh settings, and deletes the libraries
     unless --keep-library.

Timing: natural Vulkan sample progress (samples 8 -> 8+N), wall time including
UI/scheduler. viewport.render_frames is NOT used - it drives the CPU renderer.
Run from a separate terminal with the app open: python scripts\\test\\rt_k0_grain_mesh_near_ipc.py --domain "<name>" --apply
"""
import argparse
import json
import time
from pathlib import Path

from rt_ipc import RtIpc


def wait_samples(ipc, minimum, frame, timeout=60):
    deadline = time.perf_counter() + timeout
    while True:
        assert ipc.call("timeline.get_frame") == frame, "Timeline changed during probe"
        status = ipc.call("viewport.status")
        if status["samples"] >= minimum:
            return time.perf_counter(), status["samples"]
        if status["accumulation_complete"]:
            raise RuntimeError("Accumulation finished before the measurement window; "
                               "raise the viewport sample limit")
        if time.perf_counter() > deadline:
            raise RuntimeError("RT sample progress timed out")
        time.sleep(.015)


def wait_split(ipc, domain, want_active, timeout=30):
    """The split runs on the frame loop; wait until fluid.get reports it."""
    deadline = time.perf_counter() + timeout
    while True:
        near = ipc.call("fluid.get", domain=domain)["grain_mesh_near"]
        if (near["status"] == "active") == want_active:
            return near
        if want_active and near["status"] not in ("active", "off") and \
                not near["status"].startswith("not yet split"):
            raise RuntimeError("near split refused: " + near["status"])
        if time.perf_counter() > deadline:
            raise RuntimeError("near split did not settle: " + json.dumps(near))
        time.sleep(.05)


def build_library(ipc, y, sites, kind):
    """Returns (low_poly_names, sphere_mesh_names, created_objects, fracture_seed).

    kind 'boxes': flat cube primitives (12 triangles) with baked non-uniform
    scale and rotation - angular low-poly grains. kind 'fracture': Voronoi
    shards, one flat TriangleMesh each since the 2026-10-22 fracture
    migration; an older exe still emits facades and the split reports
    "missing source".
    """
    created = []
    seed = None
    if kind == "fracture":
        # A per-run seed name: a library kept by an earlier --keep-library run
        # lost its shards to this run's cleanup when both seeds were "K0GrainSeed".
        seed = ipc.call("scene.add_primitive", type="sphere",
                        name="K0GrainSeed%d" % (int(time.time()) % 100000), size=0.5)
        created.append(seed)
        ipc.call("scene.set_transform", name=seed, translation=[0.0, y, 0.0])
        frac = ipc.call("physics.fracture_object", object=seed, site_count=sites,
                        exact_surface=False, preview_gap=0.05, seed=1337)
        low = list(frac["shard_objects"])
    else:
        low = []
        for i in range(sites):
            b = ipc.call("scene.add_primitive", type="cube", name="K0GrainBox", size=0.5)
            # Deterministic aspect + tilt per exemplar; baked into the shape
            # because the gatherer reads P_orig through the transform.
            sx = 0.7 + 0.6 * ((i * 37) % 11) / 10.0
            sy = 0.5 + 0.5 * ((i * 53) % 7) / 6.0
            sz = 0.8 + 0.4 * ((i * 71) % 5) / 4.0
            ipc.call("scene.set_transform", name=b, translation=[-2.0 - i * 1.5, y, 0.0],
                     rotation=[(i * 47) % 90, (i * 29) % 90, (i * 13) % 90],
                     scale=[sx, sy, sz])
            low.append(b)
            created.append(b)
    spheres = []
    for i in range(min(len(low), 8)):
        s = ipc.call("scene.add_primitive", type="sphere", name="K0GrainSphereMesh", size=0.5)
        ipc.call("scene.set_transform", name=s, translation=[2.0 + i * 1.5, y, 0.0])
        spheres.append(s)
        created.append(s)
    return low, spheres, created, seed


def perf_accel(ipc):
    return {s["name"]: {k: s[k] for k in ("last_ms", "max_ms", "total_ms", "count")}
            for s in ipc.call("perf.list") if s["name"].startswith("accel.")}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", required=True, help="Matter domain name (fluid.list_domains)")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--distance", type=float, default=1.5, help="near distance, metres")
    parser.add_argument("--budget", type=int, default=2000000)
    parser.add_argument("--sites", type=int, default=12, help="low-poly exemplar count")
    parser.add_argument("--library", choices=("boxes", "fracture"), default="boxes")
    parser.add_argument("--samples", type=int, default=32, help="RT samples per timing window")
    parser.add_argument("--library-y", type=float, default=-500.0)
    parser.add_argument("--keep-library", action="store_true")
    parser.add_argument("--output", default="docs/dev/k0_grain_mesh_near_20261022.json")
    args = parser.parse_args()

    ipc = RtIpc()
    original_near = None
    created = []
    seed = None
    report = {"method": "Natural Vulkan RT sample progress; UI/scheduler-inclusive "
                        "wall time, not a GPU timestamp", "rows": []}
    try:
        info = ipc.call("fluid.get", domain=args.domain)
        original_near = info["grain_mesh_near"]
        frame = ipc.call("timeline.get_frame")
        report.update(domain=args.domain, frame=frame, original_near=original_near,
                      splat_geometry=info.get("splat_geometry"),
                      primary_visual_children=info.get("primary_visual_children"),
                      shading=ipc.call("viewport.shading"),
                      pipeline=ipc.call("render.rt_pipeline_status"),
                      camera=ipc.call("camera.get"),
                      instance_stats=ipc.call("render.instance_stats"))
        if not args.apply:
            print(json.dumps(report, indent=2))
            return
        assert report["shading"]["mode"] == "rendered", "Rendered viewport required"
        assert report["pipeline"]["rt_pipeline_state"] == "ready", "Ready RT pipeline required"
        assert report["splat_geometry"] == "icosphere", \
            "Domain must use procedural spheres (fluid.set_splat_geometry geometry='icosphere')"

        shards, spheres, created, seed = build_library(ipc, args.library_y, args.sites, args.library)
        report["library"] = {"kind": args.library, "low_poly": shards, "sphere_meshes": spheres}
        assert shards, "no low-poly exemplars were created"

        cases = [
            ("spheres", {"near_mesh_enabled": False}),
            ("lowpoly_D", {"near_mesh_enabled": True, "near_mesh_sources": shards,
                          "near_mesh_distance": args.distance}),
            ("lowpoly_4D", {"near_mesh_enabled": True, "near_mesh_sources": shards,
                           "near_mesh_distance": args.distance * 4.0}),
            ("spheremesh_D", {"near_mesh_enabled": True, "near_mesh_sources": spheres,
                              "near_mesh_distance": args.distance}),
            ("spheres_repeat", {"near_mesh_enabled": False}),
        ]
        for label, patch in cases:
            ipc.call("fluid.set_splat_geometry", domain=args.domain,
                     near_mesh_budget=args.budget, near_mesh_shape="scene",
                     near_mesh_source_materials=False, **patch)
            near = wait_split(ipc, args.domain, patch["near_mesh_enabled"])
            # Let the structural rebuild land before the window opens.
            wait_samples(ipc, 2, frame)
            ipc.call("perf.reset")
            start, s0 = wait_samples(ipc, 8, frame)
            end, s1 = wait_samples(ipc, s0 + args.samples, frame)
            near = ipc.call("fluid.get", domain=args.domain)["grain_mesh_near"]
            row = {"case": label, "patch": {k: v for k, v in patch.items()
                                            if k != "near_mesh_sources"},
                   "sample_start": s0, "sample_end": s1,
                   "wall_ms_per_rt_sample": (end - start) * 1000.0 / max(s1 - s0, 1),
                   "near": near,
                   "instance_stats": ipc.call("render.instance_stats"),
                   "accel_perf": perf_accel(ipc),
                   "gpu_memory": ipc.call("perf.get_gpu_memory")}
            report["rows"].append(row)
            Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
            render_row = next((b for b in row["instance_stats"]["backends"]
                               if b["role"] == "render"), {})
            print(json.dumps({"case": label,
                              "ms_per_sample": round(row["wall_ms_per_rt_sample"], 2),
                              "near_instances": near["near_instances"],
                              "sphere_instances": near["sphere_instances"],
                              "exemplar_triangles": near["exemplar_triangles"],
                              "tlas_instances": render_row.get("instance_count"),
                              "sphere_records": render_row.get("sphere_records"),
                              "status": near["status"]}), flush=True)
    finally:
        try:
            if original_near is not None and args.apply:
                ipc.call("fluid.set_splat_geometry", domain=args.domain,
                         near_mesh_enabled=original_near["enabled"],
                         near_mesh_distance=max(original_near["distance"], 0.01),
                         near_mesh_budget=original_near["budget"],
                         near_mesh_shape=original_near.get("shape", "scene"),
                         near_mesh_source_materials=original_near.get("source_materials", False))
                # Sources too, and BEFORE the library is deleted: the last case
                # leaves the library named here, and a domain still naming a
                # deleted node falls back to spheres with "missing source".
                # Restoring only the flags left a user's own exemplar list
                # replaced by this run's library.
                try:
                    ipc.call("fluid.set_splat_geometry", domain=args.domain,
                             near_mesh_sources=original_near["sources"])
                except Exception as exc:  # report, never mask the measurement
                    report["source_restore_error"] = str(exc)
                restored = ipc.call("fluid.get", domain=args.domain)["grain_mesh_near"]
                report["near_restored"] = (restored["enabled"] == original_near["enabled"] and
                                           restored["budget"] == original_near["budget"] and
                                           restored["sources"] == original_near["sources"] and
                                           restored.get("shape", "scene") == original_near.get("shape", "scene") and
                                           restored.get("source_materials", False) ==
                                           original_near.get("source_materials", False))
            if created and not args.keep_library:
                if seed:
                    try:
                        ipc.call("physics.unfracture_object", object=seed)
                    except Exception as exc:
                        report["unfracture_error"] = str(exc)
                for name in created:
                    try:
                        ipc.call("scene.delete", name=name)
                    except Exception as exc:
                        report.setdefault("delete_errors", []).append(f"{name}: {exc}")
            if report["rows"] or "near_restored" in report:
                report["final_frame"] = ipc.call("timeline.get_frame")
                Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(json.dumps({k: report.get(k) for k in
                                  ("near_restored", "final_frame", "delete_errors")}), flush=True)
        finally:
            ipc.close()


if __name__ == "__main__":
    main()
