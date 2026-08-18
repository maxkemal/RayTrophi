"""Granular CPU vs Vulkan parity — the roadmap's Faz 3 acceptance gate.

    python scripts\test\rt_test_granular_backend_parity.py

Runs the SAME seeded scene twice, once on each backend, and compares the
material response. Until the CPU constitutive path existed this test could not
be written at all: a granular domain without the Vulkan backend fell through to
the incompressible liquid solver, so "CPU" and "Vulkan" were not two
implementations of one model, they were two different materials.

WHAT PARITY MEANS HERE
----------------------
Not bit equality. The device accumulates P2G with float atomics, so the summation
ORDER differs run to run and between backends; demanding exact equality would
produce a test that fails for reasons no one can act on. What must match is the
PHYSICS: how much material yielded, how much compacted, how much broke, and
whether either backend reached a state the other did not.

★ THE ASSERTION THAT MATTERS MOST IS THE FLAG ONE. A tolerance on a mean can be
satisfied by two runs that are quietly doing different things; `invalid` or
`strain_limited` appearing on ONE backend only means the two are not running the
same solver, however close their averages look.

★★ A run where BOTH backends report zero of everything also passes every
tolerance while proving nothing. The exercise gate below refuses that case
explicitly — the scene has to actually load the material.
"""

import ctypes
from ctypes import wintypes
import json
import sys


PIPE_NAME = r"\\.\pipe\RayTrophiStudio"
# Defaults to its own scratch domain; pass a name to run against an
# existing one (the material params are overwritten either way).
DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "GranularParityTest"
DT = 1.0 / 24.0
STEPS = 120


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


MATERIAL = {
    "preset": "sand",
    "boundary": "closed",
    "enabled": True,
    "granular_enabled": True,
    "granular_friction_angle": 35.0,
    "granular_cohesion": 0.0,
    "granular_dilatancy": 5.0,
    "granular_young_modulus": 2.0e5,
    "granular_poisson_ratio": 0.25,
    "granular_tensile_cutoff": 0.0,
    "granular_hardening": 0.0,
    "granular_rebonding": False,
    "granular_max_solver_substeps": 32,
}

SEED = {
    "seed_min": [-0.35, 1.30, -0.35],
    "seed_max": [0.35, 2.10, 0.35],
    "particles_per_cell": 4,
    "replace": True,
    "persistent": False,
}


def require_isolated_scene(rt):
    domains = rt.call("fluid.list_domains")["domains"]
    if DOMAIN not in [d["name"] for d in domains]:
        rt.call("fluid.create_domain", {
            "name": DOMAIN, "type": "fluid",
            "domain_min": [-1.0, 0.0, -1.0], "domain_max": [1.0, 3.0, 1.0],
            "voxel_size": 0.05,
        })
        domains = rt.call("fluid.list_domains")["domains"]
    if len([d for d in domains if d["name"] == DOMAIN]) > 1:
        raise SystemExit("Duplicate domains named {!r}".format(DOMAIN))
    others = [d for d in domains if d["name"] != DOMAIN and d.get("enabled", True)]
    if others:
        raise SystemExit(
            "fluid.step advances every enabled domain; disable the others first: " +
            ", ".join(d["name"] for d in others))


def run_backend(rt, backend):
    payload = dict(MATERIAL)
    payload["domain"] = DOMAIN
    payload["backend"] = backend
    rt.call("fluid.set_param", payload)
    rt.call("fluid.clear", {"domain": DOMAIN, "clear_seed": True})
    rt.call("fluid.reset")
    rt.call("fluid.seed", dict(SEED, domain=DOMAIN))
    initial = rt.call("fluid.get", {"domain": DOMAIN})["particle_count"]
    worst_invalid = 0
    worst_strain_limited = 0
    for _ in range(STEPS):
        rt.call("fluid.step", {"dt": DT})
        s = rt.call("fluid.get", {"domain": DOMAIN})
        worst_invalid = max(worst_invalid, s["granular_invalid"])
        worst_strain_limited = max(worst_strain_limited, s["granular_strain_limited"])
    final = rt.call("fluid.get", {"domain": DOMAIN})
    final["_initial_particles"] = initial
    final["_worst_invalid"] = worst_invalid
    final["_worst_strain_limited"] = worst_strain_limited
    return final


def main():
    rt = Ipc()
    require_isolated_scene(rt)

    cpu = run_backend(rt, "cpu")
    gpu = run_backend(rt, "vulkan")

    failures = []

    if cpu["_initial_particles"] != gpu["_initial_particles"]:
        failures.append("seeds differ: cpu={} vulkan={}".format(
            cpu["_initial_particles"], gpu["_initial_particles"]))
    for name, state in (("cpu", cpu), ("vulkan", gpu)):
        if state["particle_count"] != state["_initial_particles"]:
            failures.append("{}: particle count changed {} -> {}".format(
                name, state["_initial_particles"], state["particle_count"]))

    # ── The scene has to actually load the material ──────────────────────────
    if cpu["granular_yielded"] == 0 and gpu["granular_yielded"] == 0:
        failures.append(
            "neither backend yielded a single particle - the pile never loaded, "
            "so every tolerance below passes vacuously")

    # ── Flags: a state one backend reaches and the other does not ────────────
    for label, a, b in (("det(F) reset", cpu["_worst_invalid"], gpu["_worst_invalid"]),
                        ("dt*C clamp", cpu["_worst_strain_limited"],
                         gpu["_worst_strain_limited"])):
        if (a > 0) != (b > 0):
            failures.append(
                "{} fired on one backend only (cpu={}, vulkan={}) - the two are "
                "not running the same solver".format(label, a, b))

    # ── Physics tolerances. Fractions of the particle count, not raw counts,
    # so the gate does not tighten as the scene grows.
    n = max(cpu["_initial_particles"], 1)
    def fraction(state, key):
        return float(state[key]) / n

    for key, tol, what in (
        ("granular_yielded", 0.15, "yielded fraction"),
        ("granular_detached", 0.20, "detached fraction"),
        ("granular_compaction_capped", 0.15, "plastic compaction fraction"),
    ):
        delta = abs(fraction(cpu, key) - fraction(gpu, key))
        if delta > tol:
            failures.append("{} differs by {:.3f} (cpu={:.3f} vulkan={:.3f}, tol {:.2f})".format(
                what, delta, fraction(cpu, key), fraction(gpu, key), tol))

    for key, tol, what in (
        ("granular_mean_damage", 0.05, "mean damage"),
        ("granular_overburden_pressure", 0.30, "settled column load (relative)"),
    ):
        a, b = float(cpu[key]), float(gpu[key])
        if what.endswith("(relative)"):
            scale = max(abs(a), abs(b), 1.0)
            delta = abs(a - b) / scale
        else:
            delta = abs(a - b)
        if delta > tol:
            failures.append("{} differs by {:.3f} (cpu={:.3f} vulkan={:.3f}, tol {:.2f})".format(
                what, delta, a, b, tol))

    for label, state in (("cpu", cpu), ("vulkan", gpu)):
        print("{:7s} particles={} yielded={} detached={} compacted={} "
              "invalid_peak={} clamp_peak={}".format(
                  label, state["particle_count"], state["granular_yielded"],
                  state["granular_detached"], state["granular_compaction_capped"],
                  state["_worst_invalid"], state["_worst_strain_limited"]))
        print("        mean_damage={:.4f} overburden={:.0f} Pa E_eff={:.0f} Pa "
              "substeps={}".format(
                  state["granular_mean_damage"], state["granular_overburden_pressure"],
                  state["granular_effective_young_modulus"],
                  state["granular_solver_substeps"]))

    if failures:
        print("FAIL")
        for f in failures:
            print(" - " + f)
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
