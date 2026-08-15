"""Regression test for delayed fluid particle growth after seed + reset.

Run in an otherwise empty scene while RayTrophi Studio is open:

    python scripts\test\rt_test_fluid_reseed_conservation.py

The falling block spreads over the floor, which used to make dynamic reseeding
amplify a thin frontier until the domain suddenly consumed its particle cap.
"""

import ctypes
import ctypes.wintypes as wintypes
import json


PIPE_NAME = r"\\.\pipe\RayTrophiStudio"
DOMAIN = "FluidReseedConservationTest"
DT = 1.0 / 60.0
STEPS = 180


class Ipc:
    def __init__(self):
        self.k32 = ctypes.windll.kernel32
        self.handle = self.k32.CreateFileW(
            PIPE_NAME, 0x80000000 | 0x40000000, 0, None, 3, 0, None)
        invalid = wintypes.HANDLE(-1).value & 0xFFFFFFFFFFFFFFFF
        if self.handle == -1 or (self.handle & 0xFFFFFFFFFFFFFFFF) == invalid:
            raise SystemExit("Cannot connect to RayTrophi Studio IPC")
        mode = wintypes.DWORD(0x00000002)
        self.k32.SetNamedPipeHandleState(self.handle, ctypes.byref(mode), None, None)
        self.request_id = 0

    def call(self, method, params=None):
        self.request_id += 1
        request = {"id": self.request_id, "method": method}
        if params is not None:
            request["params"] = params
        payload = json.dumps(request).encode("utf-8")
        written = wintypes.DWORD(0)
        if not self.k32.WriteFile(self.handle, payload, len(payload),
                                  ctypes.byref(written), None):
            raise OSError("WriteFile failed")

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
                raise OSError("ReadFile failed")
        response = json.loads(b"".join(chunks).decode("utf-8"))
        if "error" in response:
            raise RuntimeError("{} failed: {}".format(method, response["error"]))
        return response.get("result")


def domain_info(rt):
    return rt.call("fluid.get", {"domain": DOMAIN})


def particle_count(rt):
    return int(domain_info(rt)["particle_count"])


def main():
    rt = Ipc()
    existing = rt.call("fluid.list_domains")["domains"]
    if DOMAIN in [domain["name"] for domain in existing]:
        rt.call("fluid.remove_domain", {"domain": DOMAIN})

    rt.call("fluid.create_domain", {
        "name": DOMAIN,
        "type": "fluid",
        "domain_min": [-1.0, 0.0, -1.0],
        "domain_max": [1.0, 3.0, 1.0],
        "voxel_size": 0.08,
    })
    rt.call("fluid.set_param", {
        "domain": DOMAIN,
        "backend": "vulkan",
        "render_mode": "splat",
    })
    rt.call("fluid.clear", {"domain": DOMAIN, "clear_seed": True})
    rt.call("fluid.seed", {
        "domain": DOMAIN,
        "seed_min": [-0.45, 2.0, -0.45],
        "seed_max": [0.45, 2.6, 0.45],
        "particles_per_cell": 8,
        "replace": True,
        "persistent": True,
    })

    seeded = particle_count(rt)
    rt.call("fluid.reset")
    reset_count = particle_count(rt)
    print("seeded={} reset={}".format(seeded, reset_count))
    if seeded <= 0 or reset_count != seeded:
        raise SystemExit("FAIL: persistent seed did not reproduce exactly on reset")

    minimum = reset_count
    maximum = reset_count
    total_added = 0
    total_removed = 0
    for step in range(1, STEPS + 1):
        rt.call("fluid.step", {"dt": DT})
        info = domain_info(rt)
        count = int(info["particle_count"])
        added = int(info["reseed_added_particles"])
        removed = int(info["reseed_removed_particles"])
        total_added += added
        total_removed += removed
        if added > removed:
            raise SystemExit(
                "FAIL: step {} reseed created mass: +{} / -{}"
                .format(step, added, removed))
        minimum = min(minimum, count)
        maximum = max(maximum, count)
        if step % 15 == 0:
            print("step {:3d}: particles={} reseed=+{} / -{}".format(
                step, count, added, removed))

    print("range: {}..{} (initial {})".format(minimum, maximum, reset_count))
    print("reseed totals: +{} / -{}".format(total_added, total_removed))
    if maximum > reset_count:
        raise SystemExit(
            "FAIL: particle count grew without an emitter: {} -> {}"
            .format(reset_count, maximum))
    print("PASS: seed/reset stayed count-conservative for {} steps".format(STEPS))


if __name__ == "__main__":
    main()
