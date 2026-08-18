"""Granular preset contract — roadmap Faz 6 smoke test.

    python scripts\test\rt_test_granular_presets.py [DomainName]

Sets each granular preset over IPC and reads it back. No stepping, no seeding:
this is a PARAMETER contract test, so it stays fast and cannot be blamed for a
physics regression.

WHAT IT PINS DOWN
-----------------
1. Every granular preset is reachable BY NAME from a script. A preset that only
   exists in the UI combo is, by this project's first rule, untestable.
2. The values actually land on the domain the solver reads. There are two
   parameter stores behind a fluid domain (the legacy editor mirror and the grid
   descriptor); fluid.get reports the grid descriptor, which is the one the
   solver uses.
3. ★ THE DEAD DIALS STAY DEAD. internal_friction and flip_blend must read 0 on
   every granular preset. Both were once non-zero and reached nothing: the G2P
   paths zero internal_friction when granular_enabled, and FLIP is forced off
   because the elastic stress enters the grid BEFORE the FLIP snapshot, so any
   non-zero blend subtracts it straight back out. A preset shipping 0.92 there
   is not a preference, it is the collapse bug wearing a preset's name.
4. ★★ E IS CHECKED AGAINST PILE DEPTH, not against a magic number. The
   corotational predictor needs E >= 10*rho*g*h to keep the bottom layer inside
   small strain, so each preset asserts the depth it claims to be honest up to.
   A preset that quietly stops satisfying its own claim is the failure this
   catches — nothing about it looks wrong on screen.
5. get -> set round trips, "custom" included. Rejecting a value this very API
   reports is how a script that reads a domain and writes it back breaks.
"""

import ctypes
from ctypes import wintypes
import json
import sys


PIPE_NAME = r"\\.\pipe\RayTrophiStudio"
DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "GranularPresetTest"

GRANULAR_DENSITY = 1600.0   # kGranularDensity
GRAVITY = 9.81
LOAD_RATIO = 10.0           # kGranularStiffnessLoadRatio

# preset -> (expected friction, cohesion, tensile, substep ceiling, claimed depth m)
EXPECTED = {
    "sand":          (35.0,     0.0,    0.0, 32, 1.27),
    "wet_sand":      (37.0,  1500.0,  400.0, 32, 1.59),
    "gravel":        (43.0,     0.0,    0.0, 40, 1.91),
    "cohesive_soil": (20.0, 12000.0, 3000.0, 32, 0.76),
}


class Ipc:
    def __init__(self):
        self.k32 = ctypes.windll.kernel32
        self.handle = self.k32.CreateFileW(
            PIPE_NAME, 0x80000000 | 0x40000000, 0, None, 3, 0, None)
        invalid = wintypes.HANDLE(-1).value & 0xFFFFFFFFFFFFFFFF
        if self.handle == -1 or (self.handle & 0xFFFFFFFFFFFFFFFF) == invalid:
            raise SystemExit(
                "Cannot connect to the RayTrophi Studio IPC pipe.\n"
                "  - Is the app open, and does the SceneLog say 'IPC server started'?\n"
                "  - The pipe serves ONE client: a PowerShell session that never\n"
                "    called Disconnect-RtIpc still holds it.")
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
            raise OSError("IPC WriteFile failed")
        chunks = []
        while True:
            buffer = ctypes.create_string_buffer(65536)
            read = wintypes.DWORD(0)
            ok = self.k32.ReadFile(self.handle, buffer, len(buffer),
                                   ctypes.byref(read), None)
            chunks.append(buffer.raw[:read.value])
            if ok:
                break
            if self.k32.GetLastError() != 234:
                raise OSError("IPC ReadFile failed")
        response = json.loads(b"".join(chunks).decode("utf-8"))
        if "error" in response:
            raise RuntimeError("{}: {}".format(method, response["error"]))
        return response.get("result")


def close(a, b, tol=1e-3):
    return abs(float(a) - float(b)) <= tol


def main():
    rt = Ipc()
    names = [d["name"] for d in rt.call("fluid.list_domains")["domains"]]
    if DOMAIN not in names:
        rt.call("fluid.create_domain", {
            "name": DOMAIN, "type": "fluid",
            "domain_min": [-1.0, 0.0, -1.0], "domain_max": [1.0, 3.0, 1.0],
            "voxel_size": 0.05,
        })

    failures = []
    for preset, (friction, cohesion, tensile, ceiling, depth) in EXPECTED.items():
        rt.call("fluid.set_param", {"domain": DOMAIN, "preset": preset})
        got = rt.call("fluid.get", {"domain": DOMAIN})

        if got.get("preset") != preset:
            failures.append("{}: reported back as {!r}".format(preset, got.get("preset")))
        if not got.get("granular_enabled"):
            failures.append("{}: granular_enabled did not land on the grid domain "
                            "(the store the solver reads)".format(preset))
        for key, want in (("granular_friction_angle", friction),
                          ("granular_cohesion", cohesion),
                          ("granular_tensile_cutoff", tensile)):
            if not close(got.get(key, -1), want):
                failures.append("{}: {}={} expected {}".format(
                    preset, key, got.get(key), want))
        if got.get("granular_max_solver_substeps") != ceiling:
            failures.append("{}: substep ceiling {} expected {}".format(
                preset, got.get("granular_max_solver_substeps"), ceiling))

        # ★ The dead dials.
        for key in ("internal_friction", "flip_blend"):
            value = got.get(key)
            if value is not None and not close(value, 0.0):
                failures.append(
                    "{}: {}={} — a granular preset must ship 0 here; both paths "
                    "ignore internal_friction and FLIP cancels the elastic "
                    "stress".format(preset, key, value))

        # ★★ Does the stiffness still meet the depth this preset claims?
        needed = LOAD_RATIO * GRANULAR_DENSITY * GRAVITY * depth
        E = float(got.get("granular_young_modulus", 0.0))
        if E + 1.0 < needed:
            failures.append(
                "{}: E={:.0f} Pa no longer covers its claimed {:.2f} m pile "
                "({:.0f} Pa needed). Either raise E or correct the claim — the "
                "doc comment and the tooltip both state this depth.".format(
                    preset, E, depth, needed))

        print("{:14s} E={:>7.0f} Pa  friction={:>4.1f}  cohesion={:>7.0f}  "
              "tensile={:>6.0f}  ceiling={:>2d}  honest to {:.2f} m".format(
                  preset, E, got.get("granular_friction_angle", 0.0),
                  got.get("granular_cohesion", 0.0),
                  got.get("granular_tensile_cutoff", 0.0),
                  got.get("granular_max_solver_substeps", 0), depth))

    # ── get -> set round trip, including the value a tuned domain reports ────
    rt.call("fluid.set_param", {"domain": DOMAIN, "granular_cohesion": 777.0})
    tuned = rt.call("fluid.get", {"domain": DOMAIN})
    if tuned.get("preset") != "custom":
        failures.append("editing a field did not mark the domain custom: {!r}".format(
            tuned.get("preset")))
    try:
        rt.call("fluid.set_param", {"domain": DOMAIN, "preset": tuned["preset"]})
    except RuntimeError as exc:
        failures.append("get -> set round trip rejected its own value: {}".format(exc))
    back = rt.call("fluid.get", {"domain": DOMAIN})
    if not close(back.get("granular_cohesion", -1), 777.0):
        failures.append("writing preset='custom' overwrote the tuned material "
                        "(cohesion {}, expected 777)".format(back.get("granular_cohesion")))

    # A liquid preset must still turn granular OFF.
    rt.call("fluid.set_param", {"domain": DOMAIN, "preset": "water"})
    if rt.call("fluid.get", {"domain": DOMAIN}).get("granular_enabled"):
        failures.append("switching to water left granular_enabled set")

    if failures:
        print("FAIL")
        for f in failures:
            print(" - " + f)
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
