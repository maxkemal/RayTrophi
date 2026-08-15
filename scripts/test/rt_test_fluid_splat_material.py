"""IPC smoke test for fluid splat-material authoring.

Run from a separate terminal while RayTrophi Studio is open:

    python scripts\test\rt_test_fluid_splat_material.py

This checks the shared core/API/IPC path and read-back. The Vulkan SDF/Splat
visibility transition still needs the short visual checklist printed at PASS.
"""

import ctypes
import ctypes.wintypes as wintypes
import json


PIPE_NAME = r"\\.\pipe\RayTrophiStudio"
DOMAIN = "SplatMaterialTest"
MATERIAL = "SplatMaterialProbe"


class Ipc:
    def __init__(self):
        self.k32 = ctypes.windll.kernel32
        self.handle = self.k32.CreateFileW(
            PIPE_NAME, 0x80000000 | 0x40000000, 0, None, 3, 0, None)
        invalid = wintypes.HANDLE(-1).value & 0xFFFFFFFFFFFFFFFF
        if self.handle == -1 or (self.handle & 0xFFFFFFFFFFFFFFFF) == invalid:
            raise SystemExit(
                "Cannot connect to {} (error {}). Is RayTrophi Studio running?"
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
            ok = self.k32.ReadFile(self.handle, buf, len(buf),
                                   ctypes.byref(read), None)
            chunks.append(buf.raw[:read.value])
            if ok:
                break
            if self.k32.GetLastError() != 234:
                raise OSError("ReadFile failed ({})".format(self.k32.GetLastError()))
        response = json.loads(b"".join(chunks).decode("utf-8"))
        if "error" in response:
            raise RuntimeError("{} failed: {}".format(method, response["error"]))
        return response.get("result")


def ensure_rig(rt):
    materials = rt.call("material.list") or []
    names = [m if isinstance(m, str) else m.get("name") for m in materials]
    if MATERIAL not in names:
        rt.call("material.create", {"type": "principled", "name": MATERIAL})

    domains = rt.call("fluid.list_domains")["domains"]
    if DOMAIN not in [domain["name"] for domain in domains]:
        rt.call("fluid.create_domain", {
            "name": DOMAIN,
            "type": "fluid",
            "domain_min": [-1.0, 0.0, -1.0],
            "domain_max": [1.0, 2.0, 1.0],
            "voxel_size": 0.08,
        })
    rt.call("fluid.set_param", {
        "domain": DOMAIN,
        "backend": "vulkan",
        "render_mode": "splat",
    })


def main():
    rt = Ipc()
    ensure_rig(rt)

    rt.call("fluid.set_splat_material", {
        "domain": DOMAIN,
        "material": MATERIAL,
    })
    assigned = rt.call("fluid.get", {"domain": DOMAIN}).get("splat_material")
    print("assigned splat_material={!r}".format(assigned))
    if assigned != MATERIAL:
        raise SystemExit("FAIL: assigned material did not round-trip")

    rt.call("fluid.set_splat_material", {"domain": DOMAIN, "material": ""})
    cleared = rt.call("fluid.get", {"domain": DOMAIN}).get("splat_material")
    print("cleared splat_material={!r}".format(cleared))
    if cleared != "":
        raise SystemExit("FAIL: clearing the material did not restore inheritance")

    print("PASS")
    print("Visual 1: assign the probe in Built-in Icosphere mode, then toggle")
    print("SDF -> Splat -> SDF in Vulkan RT. Exactly one representation must")
    print("be visible after every switch; SDF must not flash away or return stale.")
    print("Visual 2: choose a scene mesh as Splat Geometry, then delete it.")
    print("The hierarchy row must disappear, the combo must say 'Missing: <name>',")
    print("splats must fall back to icospheres, and no ImGui assertion may appear.")
    print("Undo the delete: the row, scene-mesh splats and original face materials")
    print("must return without switching Water/Sand or rebuilding the scene manually.")


if __name__ == "__main__":
    main()
