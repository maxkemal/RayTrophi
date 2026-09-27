"""IPC test: volume slot identity invariants (render.volume_slots).

Faz 0 item 3 of docs/dev/BIRLESIK_MADDE_DOMAIN_TASARIMI.md. The unified domain
will give one domain more than one volume; these are the invariants that
repeatedly broke in this repo with a single volume per domain, checked BEFORE
touching that code:

  1. Nothing changes -> the liquid volume keeps its slot identity (stable_key).
  2. Particles mode -> no ACTIVE liquid volume on either backend.
  3. Back to surface -> active again, with a density grid.
  4. An unrelated scene edit (TLAS rebuild) does not evict the SDF volume.
  5. Foam off -> the SDF slot carries no temperature channel (foam rides it).
  6. The domain's volume name / vdb id match a slot on every Vulkan backend.

Self-contained: creates domain 'SlotProbe' + a primitive, removes both on exit.
Run with the app open and the timeline paused:

    python scripts\\test\\rt_test_volume_slot_identity_ipc.py
"""

import ctypes
import ctypes.wintypes as wintypes
import json
import sys
import time

PIPE_NAME = r"\\.\pipe\RayTrophiStudio"
DOMAIN = "SlotProbe"
PROP = "SlotProbeProp"
SETTLE_S = 1.5
# "solid" exercises the raster viewport (packet order, no TLAS); "rendered"
# fills the render backend too (TLAS order, stable_key identity).
SHADING = sys.argv[1] if len(sys.argv) > 1 else "solid"


class Ipc:
    def __init__(self):
        self.k32 = ctypes.windll.kernel32
        self.handle = self.k32.CreateFileW(
            PIPE_NAME, 0x80000000 | 0x40000000, 0, None, 3, 0, None)
        invalid = wintypes.HANDLE(-1).value & 0xFFFFFFFFFFFFFFFF
        if self.handle == -1 or (self.handle & 0xFFFFFFFFFFFFFFFF) == invalid:
            raise SystemExit("Cannot connect to {} (error {}). Is RayTrophi Studio running?"
                             .format(PIPE_NAME, self.k32.GetLastError()))
        mode = wintypes.DWORD(0x00000002)
        self.k32.SetNamedPipeHandleState(self.handle, ctypes.byref(mode), None, None)
        self.request_id = 0

    def call(self, method, params=None):
        self.request_id += 1
        request = {"id": self.request_id, "method": method}
        if params:
            request["params"] = params
        payload = json.dumps(request).encode("utf-8")
        written = wintypes.DWORD(0)
        if not self.k32.WriteFile(self.handle, payload, len(payload),
                                  ctypes.byref(written), None):
            raise OSError("WriteFile failed ({})".format(self.k32.GetLastError()))
        chunks = []
        while True:
            buf = ctypes.create_string_buffer(65536)
            read = wintypes.DWORD(0)
            ok = self.k32.ReadFile(self.handle, buf, len(buf), ctypes.byref(read), None)
            chunks.append(buf.raw[:read.value])
            if ok:
                break
            if self.k32.GetLastError() != 234:
                raise OSError("ReadFile failed ({})".format(self.k32.GetLastError()))
        response = json.loads(b"".join(chunks).decode("utf-8"))
        if "error" in response:
            raise RuntimeError("{} failed: {}".format(method, response["error"]))
        return response.get("result")


failures = []
skipped = set()


def check(ok, what, detail=""):
    print("  [{}] {}{}".format("PASS" if ok else "FAIL", what, " -- " + detail if detail else ""))
    if not ok:
        failures.append(what)


def settle(rt):
    # A frame change is what reliably re-runs the volume route on a paused
    # timeline (fluid.step does not); nudge the camera so raster redraws too.
    frame = rt.call("timeline.get_frame")
    frame = int(frame.get("frame", 0) if isinstance(frame, dict) else frame)
    rt.call("timeline.set_frame", {"frame": frame + 1})
    rt.call("camera.orbit", {"yaw": 0.2, "pitch": 0.0})
    time.sleep(SETTLE_S)
    rt.call("camera.orbit", {"yaw": -0.2, "pitch": 0.0})
    time.sleep(SETTLE_S)


def is_ours(s, name, vdb_id):
    # A backend without a TLAS (raster viewport) publishes in packet order: its
    # slots carry no object name, only the vdb id they were filled from.
    if s.get("packet_order"):
        return vdb_id is not None and vdb_id >= 0 and s["vdb_id"] == vdb_id
    return bool(name) and s["name"] == name


def identity(s):
    # stable_key is object identity; packet-order slots only have the resource
    # id, a weaker signal (it churns by design on rebinds).
    return ("vdb", s["vdb_id"]) if s.get("packet_order") else ("key", s["stable_key"])


def snapshot(rt):
    t = rt.call("render.volume_slots")
    dom = next((d for d in t["domains"] if d["domain"] == DOMAIN), None)
    rows = {}
    for b in t["backends"]:
        if not b["is_vulkan"]:
            continue
        # The render backend is filled lazily (only once Rendered has run), so
        # in Solid it legitimately holds nothing. Say so instead of failing or
        # silently dropping it.
        if not b["slots"]:
            skipped.add(b["role"])
            continue
        name = dom["volume_name"] if dom and dom["has_volume"] else None
        vdb_id = dom["vdb_id"] if dom and dom["has_volume"] else None
        rows[b["role"]] = {
            "serial": b["upload_serial"],
            "slot": next((s for s in b["slots"] if is_ours(s, name, vdb_id)), None),
        }
    return t, dom, rows


def describe(rows):
    out = []
    for role, r in rows.items():
        s = r["slot"]
        out.append("{}: {}".format(role, "none" if not s else
                   "slot={} {}={} active={} src={} pub={}".format(
                       s["slot"], *identity(s), s["is_active"], s["source"], s["published"])))
    return "; ".join(out)


def main():
    rt = Ipc()
    t = rt.call("render.volume_slots")
    if not t.get("available"):
        raise SystemExit("no Vulkan backend -- nothing to test")
    shading_before = rt.call("viewport.shading").get("mode", "solid")
    try:
        rt.call("fluid.create_domain", {"name": DOMAIN, "type": "fluid",
                                        "domain_min": [-1, 0, -1], "domain_max": [1, 2, 1],
                                        "voxel_size": 0.05})
        rt.call("fluid.seed", {"domain": DOMAIN, "seed_min": [-0.6, 0.05, -0.6],
                               "seed_max": [0.2, 0.8, 0.2], "particles_per_cell": 4})
        rt.call("fluid.set_param", {"domain": DOMAIN, "render_mode": "surface"})
        rt.call("viewport.set_shading", {"mode": SHADING})
        settle(rt)

        print("0. setup ({})".format(SHADING))
        _, dom, rows = snapshot(rt)
        check(dom is not None and dom["has_volume"], "domain owns a volume in surface mode",
              str(dom))
        check(any(r["slot"] for r in rows.values()), "at least one backend holds the volume",
              "backends checked: {}".format(sorted(rows)))
        if not dom or not dom["has_volume"]:
            return

        print("1. identity is stable when nothing changes")
        _, _, a = snapshot(rt)
        settle(rt)
        _, _, b = snapshot(rt)
        for role in a:
            sa, sb = a[role]["slot"], b[role]["slot"]
            check(sa is not None and sb is not None and identity(sa) == identity(sb),
                  "{}: same identity across frames".format(role),
                  "{} -> {}".format(sa and identity(sa), sb and identity(sb)))

        print("6. scene side matches every backend")
        for role, r in b.items():
            s = r["slot"]
            check(s is not None and s["vdb_id"] == dom["vdb_id"],
                  "{}: slot named '{}' with vdb_id {}".format(role, dom["volume_name"], dom["vdb_id"]),
                  describe({role: r}))

        print("5. SDF slot has no temperature channel with foam off")
        for role, r in b.items():
            s = r["slot"]
            if s:
                check(s["source"] == "sdf" and not s["has_temperature"],
                      "{}: source sdf, has_temperature false".format(role),
                      "source={} has_temperature={}".format(s["source"], s["has_temperature"]))

        print("4. unrelated edit does not evict the SDF")
        rt.call("scene.add_primitive", {"type": "cube", "name": PROP, "size": 0.5})
        rt.call("scene.set_transform", {"name": PROP, "translation": [4.0, 0.25, 4.0]})
        settle(rt)
        _, dom4, c = snapshot(rt)
        for role, r in c.items():
            s = r["slot"]
            check(s is not None and s["is_active"] == 1 and s["has_density"],
                  "{}: SDF still active with density after TLAS change".format(role),
                  describe({role: r}))
            before = b[role]["slot"]
            check(s is not None and before is not None and identity(s) == identity(before),
                  "{}: SDF kept its identity".format(role),
                  "{} -> {}".format(before and identity(before), s and identity(s)))

        print("2. particles mode: no active liquid volume")
        rt.call("fluid.set_param", {"domain": DOMAIN, "render_mode": "particles"})
        settle(rt)
        t2 = rt.call("render.volume_slots")
        name, vdb_id = dom["volume_name"], dom["vdb_id"]
        for bk in t2["backends"]:
            if not bk["is_vulkan"]:
                continue
            live = [s for s in bk["slots"] if is_ours(s, name, vdb_id) and s["is_active"] == 1]
            check(not live, "{}: liquid volume not drawn".format(bk["role"]),
                  "active slots: {}".format([s["slot"] for s in live]))

        print("3. back to surface: active again")
        rt.call("fluid.set_param", {"domain": DOMAIN, "render_mode": "surface"})
        settle(rt)
        _, dom3, d = snapshot(rt)
        check(dom3 is not None and dom3["has_volume"], "domain owns a volume again")
        for role, r in d.items():
            s = r["slot"]
            check(s is not None and s["is_active"] == 1 and s["has_density"],
                  "{}: SDF active with density".format(role), describe({role: r}))
    finally:
        for method, params in (("fluid.remove_domain", {"domain": DOMAIN}),
                               ("scene.delete", {"name": PROP}),
                               ("viewport.set_shading", {"mode": shading_before})):
            try:
                rt.call(method, params)
            except RuntimeError:
                pass

    print()
    for role in sorted(skipped):
        print("  [SKIP] {}: backend held no volumes in this shading (not an error)".format(role))
    if failures:
        print("FAIL: {} check(s): {}".format(len(failures), "; ".join(failures)))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
