"""Object visibility (outliner eye) over IPC: hide, verify, restore.

    python scripts/test/rt_object_visibility_ipc.py --object Plane_1
    python scripts/test/rt_object_visibility_ipc.py --object Plane_1 --rt-counters

Without --object the first visible object from scene.list_objects is used.
Phases:
  baseline  object visible; scene flag, object_info, backend masks, pick_gpu.
  hidden    scene.set_visible false -> flags false, every backend mask 0,
            pick_gpu no longer returns the object.
  rebuilt   ★ field case 2026-10-25: while hidden, a temporary primitive is
            added and deleted so the backends REBUILD. Rebuilds skip hidden
            objects entirely, so the object's instances are gone afterwards.
  restored  scene.set_visible true -> the instances must come BACK (polled,
            the rebuild runs on the frame loop) and be unmasked; pick_gpu
            hits the object again.
  refused   an unknown name is refused, not silently accepted.
With --rt-counters the RT result itself is measured too: surface bounces per
path (charged_diffuse + charged_specular from render.volume_stats). Hiding a
large surface must change it; restoring must bring it back near baseline.
This is what actually failed in the field: the flag said visible and the RT
frame was identical to the hidden one.
The object's original visibility is restored on exit, also on failure.
★ A backend that held N instances at baseline and 0 after restore is a FAIL,
not a SKIP: "no instance" is exactly what the dropped-by-rebuild bug looks like.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_ipc import RtIpc, RtIpcError  # noqa: E402

MASKS = (("viewport_raster", "viewport_raster_instances", "viewport_raster_visible"),
         ("viewport_rt", "viewport_rt_instances", "viewport_rt_visible"),
         ("render_rt", "render_rt_instances", "render_rt_visible"))
TEMP_PRIMITIVE = "__visibility_probe_tmp"

failures = []


def report(ok, label, detail=""):
    tag = "PASS" if ok is True else ("SKIP" if ok is None else "FAIL")
    print(f"[{tag}] {label}{(' - ' + detail) if detail else ''}", flush=True)
    if ok is False:
        failures.append(label)


def find_pick_point(ipc, name):
    for j in range(9):
        for i in range(9):
            u, v = (i + 0.5) / 9.0, (j + 0.5) / 9.0
            r = ipc.call("scene.pick_gpu", u=u, v=v)
            if r.get("ok") and r.get("hit") and r.get("object") == name:
                return u, v
    return None


def surface_bounces_per_path(ipc):
    """RT result: surface bounces per traced path over a short window."""
    ipc.call("render.volume_counters", enabled=False)
    ipc.call("reset_accumulation")
    time.sleep(1.0)
    ipc.call("render.volume_counters", enabled=True)
    time.sleep(2.0)
    st = ipc.call("render.volume_stats")
    ipc.call("render.volume_counters", enabled=False)
    if not st.get("available") or st.get("paths_traced", 0) <= 0:
        return None
    return (st["charged_diffuse"] + st["charged_specular"]) / st["paths_traced"]


def wait_for_instances(ipc, name, baseline, timeout=15.0):
    """Poll until every backend that held the object at baseline holds it again."""
    deadline = time.time() + timeout
    vis = ipc.call("scene.get_visible", name=name)
    while time.time() < deadline:
        if all(vis[total] >= baseline[total] for _, total, _ in MASKS if baseline[total] > 0):
            break
        time.sleep(0.5)
        vis = ipc.call("scene.get_visible", name=name)
    return vis


def check_state(ipc, name, expect_visible, pick, phase, baseline=None, vis=None):
    if vis is None:
        time.sleep(0.5)  # let the frame loop apply raster/TLAS uploads
        vis = ipc.call("scene.get_visible", name=name)
    report(vis["visible"] == expect_visible, f"{phase}: scene flag",
           f"visible={vis['visible']} entries={vis['visible_entries']}/{vis['entries']}")
    info = ipc.call("scene.object_info", name=name)
    report(info.get("visible") == expect_visible, f"{phase}: object_info.visible",
           f"{info.get('visible')}")
    for label, total_key, vis_key in MASKS:
        total, shown = vis[total_key], vis[vis_key]
        if total < 0:
            report(None, f"{phase}: {label} mask", "no Vulkan read-back for this backend")
            continue
        if total == 0:
            had = baseline[total_key] if baseline else 0
            if expect_visible and had > 0:
                report(False, f"{phase}: {label} mask",
                       f"0 instances, baseline had {had} (dropped by a rebuild and never recreated)")
            else:
                report(None, f"{phase}: {label} mask", "backend holds no instance of the object")
            continue
        want = total if expect_visible else 0
        report(shown == want, f"{phase}: {label} mask", f"{shown}/{total} visible, expected {want}")
    if pick is None:
        report(None, f"{phase}: pick_gpu", "object not on screen")
    else:
        r = ipc.call("scene.pick_gpu", u=pick[0], v=pick[1])
        hit_it = bool(r.get("ok") and r.get("hit") and r.get("object") == name)
        report(hit_it == expect_visible, f"{phase}: pick_gpu at {pick[0]:.3f},{pick[1]:.3f}",
               f"hit={r.get('hit')} object={r.get('object')!r} reason={r.get('reason')!r}")
    return vis


def force_rebuild(ipc):
    """Add and delete a primitive so the backends rebuild their geometry."""
    try:
        ipc.call("scene.add_primitive", type="cube", name=TEMP_PRIMITIVE, size=0.01)
        time.sleep(1.5)
        ipc.call("scene.delete", name=TEMP_PRIMITIVE)
        time.sleep(2.0)
        return True
    except RtIpcError as e:
        report(None, "rebuilt: forced rebuild", f"could not add/delete a primitive: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="")
    ap.add_argument("--rt-counters", action="store_true",
                    help="also measure surface bounces per path (needs Vulkan RT accumulating)")
    args = ap.parse_args()
    ipc = RtIpc()

    name = args.object
    if not name:
        for candidate in ipc.call("scene.list_objects"):
            if ipc.call("scene.get_visible", name=candidate)["visible"]:
                name = candidate
                break
    if not name:
        print("no visible object in the scene", flush=True)
        return 2
    print(f"object: {name}", flush=True)

    original = ipc.call("scene.get_visible", name=name)["visible"]
    try:
        if not original:
            ipc.call("scene.set_visible", name=name, visible=True)
            wait_for_instances(ipc, name, {k: 1 for _, k, _ in MASKS}, timeout=10.0)
        pick = find_pick_point(ipc, name)
        baseline = check_state(ipc, name, True, pick, "baseline")
        rt_base = surface_bounces_per_path(ipc) if args.rt_counters else None

        ipc.call("scene.set_visible", name=name, visible=False)
        check_state(ipc, name, False, pick, "hidden")
        rt_hidden = surface_bounces_per_path(ipc) if args.rt_counters else None

        if force_rebuild(ipc):
            check_state(ipc, name, False, pick, "rebuilt")

        ipc.call("scene.set_visible", name=name, visible=True)
        restored = wait_for_instances(ipc, name, baseline)
        check_state(ipc, name, True, pick, "restored", baseline=baseline, vis=restored)
        rt_restored = surface_bounces_per_path(ipc) if args.rt_counters else None

        if args.rt_counters:
            if None in (rt_base, rt_hidden, rt_restored):
                report(None, "rt: surface bounces per path",
                       "volume counters unavailable or no paths traced")
            else:
                detail = f"baseline={rt_base:.3f} hidden={rt_hidden:.3f} restored={rt_restored:.3f}"
                changed = abs(rt_hidden - rt_base) > 0.05 * max(rt_base, 1e-6)
                back = abs(rt_restored - rt_base) <= 0.1 * max(rt_base, 1e-6)
                if not changed:
                    report(None, "rt: hiding changed the RT result",
                           detail + " (object too small or off-screen to move the counter)")
                else:
                    report(True, "rt: hiding changed the RT result", detail)
                    report(back, "rt: restore brought the RT result back", detail)

        try:
            ipc.call("scene.set_visible", name="__no_such_object__", visible=False)
            report(False, "refused: unknown name", "call succeeded")
        except RtIpcError as e:
            report(True, "refused: unknown name", str(e))
    finally:
        ipc.call("scene.set_visible", name=name, visible=original)

    print(f"{'FAIL' if failures else 'PASS'}: {len(failures)} failure(s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
