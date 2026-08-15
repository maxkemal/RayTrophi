"""Vulkan granular material contract and cohesive-fracture smoke test.

Run from a separate terminal while RayTrophi Studio is open:

    python scripts\test\rt_test_granular_damage.py

The test uses a dedicated domain named ``GranularDamageTest``. It verifies that
IPC set/get/list expose the same clamped material values, then drops a brittle
cohesive block onto the closed domain floor. A passing physics run reports
non-zero damage, zero invalid particles and a constant particle count.
"""

import ctypes
from ctypes import wintypes
import json


PIPE_NAME = r"\\.\pipe\RayTrophiStudio"
DOMAIN = "GranularDamageTest"
DT = 1.0 / 60.0


class Ipc:
    def __init__(self):
        self.k32 = ctypes.windll.kernel32
        self.handle = self.k32.CreateFileW(
            PIPE_NAME, 0x80000000 | 0x40000000, 0, None, 3, 0, None)
        invalid = wintypes.HANDLE(-1).value & 0xFFFFFFFFFFFFFFFF
        if self.handle == -1 or (self.handle & 0xFFFFFFFFFFFFFFFF) == invalid:
            raise SystemExit("Cannot connect to RayTrophi Studio IPC pipe.")
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


def close_enough(actual, expected, tolerance=1.0e-5):
    return abs(float(actual) - float(expected)) <= tolerance


def main():
    rt = Ipc()
    domains = rt.call("fluid.list_domains")["domains"]
    if DOMAIN not in [item["name"] for item in domains]:
        rt.call("fluid.create_domain", {
            "name": DOMAIN,
            "type": "fluid",
            "domain_min": [-1.0, 0.0, -1.0],
            "domain_max": [1.0, 3.0, 1.0],
            "voxel_size": 0.05,
        })

    # fluid.step intentionally advances SimulationWorld, not just the named
    # domain. Refuse a shared scene instead of silently stepping/rendering a
    # second liquid or gas domain and making two unrelated fields look like an
    # interpenetrating pair. We do not temporarily disable domains: toggling a
    # grid domain invalidates/rebuilds its runtime grid and would mutate the
    # user's scene merely to run a test.
    domains = rt.call("fluid.list_domains")["domains"]
    same_name = [item for item in domains if item["name"] == DOMAIN]
    if len(same_name) > 1:
        raise SystemExit(
            "Granular damage test found {} domains named {!r}. Remove the "
            "duplicate test-domain/preset system before retrying; name-based "
            "seed/get would otherwise be ambiguous.".format(len(same_name), DOMAIN))
    active_others = [item for item in domains
                     if item["name"] != DOMAIN and item.get("enabled", True)]
    if active_others:
        details = ", ".join("{} ({}, {} particles)".format(
            item["name"], item.get("type", "domain"),
            item.get("particle_count", 0)) for item in active_others)
        raise SystemExit(
            "Granular damage test requires an isolated scene because fluid.step "
            "advances every enabled domain. Disable/remove the other domains "
            "and retry. Active others: " + details)

    authored = {
        "domain": DOMAIN,
        "backend": "vulkan",
        "boundary": "closed",
        "preset": "sand",
        "enabled": True,
        "granular_enabled": True,
        "granular_friction_angle": 32.0,
        "granular_cohesion": 1500.0,
        "granular_dilatancy": 4.0,
        "granular_young_modulus": 350000.0,
        "granular_poisson_ratio": 0.22,
        "granular_tensile_cutoff": 500.0,
        "granular_hardening": 1.5,
        "granular_fracture_strain": 0.01,
        "granular_damage_rate": 12.0,
        "granular_healing_rate": 0.0,
        "granular_rebonding": False,
        "granular_max_solver_substeps": 16,
    }
    rt.call("fluid.set_param", authored)

    got = rt.call("fluid.get", {"domain": DOMAIN})
    print("test domain: id={} name={} bounds={}..{} voxel={}".format(
        got.get("id"), got.get("name", DOMAIN), got.get("domain_min"),
        got.get("domain_max"), got.get("voxel_size")))
    expected = {key: value for key, value in authored.items()
                if key.startswith("granular_")}
    failures = []
    for key, value in expected.items():
        actual = got.get(key)
        equal = actual == value if isinstance(value, bool) else close_enough(actual, value)
        if not equal:
            failures.append("get {}={!r}, expected {!r}".format(key, actual, value))

    listed = next(item for item in rt.call("fluid.list_domains")["domains"]
                  if item["name"] == DOMAIN)
    for key, value in expected.items():
        actual = listed.get(key)
        equal = actual == value if isinstance(value, bool) else close_enough(actual, value)
        if not equal:
            failures.append("list {}={!r}, expected {!r}".format(key, actual, value))

    seed_payload = {
        "domain": DOMAIN,
        "seed_min": [-0.32, 1.9, -0.32],
        "seed_max": [0.32, 2.5, 0.32],
        "particles_per_cell": 4,
        "replace": True,
    }

    # Seed lifecycle contract: a one-shot seed must disappear on reset; a
    # persistent seed must return; clear_seed must disarm that recipe. This also
    # catches the retired duplicate FluidObject seed because fluid.get/list now
    # read only the authoritative grid-domain particle state.
    rt.call("fluid.clear", {"domain": DOMAIN, "clear_seed": True})
    rt.call("fluid.seed", dict(seed_payload, persistent=False))
    one_shot_count = rt.call("fluid.get", {"domain": DOMAIN})["particle_count"]
    rt.call("fluid.reset")
    after_one_shot_reset = rt.call("fluid.get", {"domain": DOMAIN})["particle_count"]
    if one_shot_count <= 0:
        failures.append("one-shot seed spawned no particles")
    elif after_one_shot_reset != 0:
        failures.append("one-shot seed survived reset: {} -> {}".format(
            one_shot_count, after_one_shot_reset))

    rt.call("fluid.seed", dict(seed_payload, persistent=True))
    persistent_count = rt.call("fluid.get", {"domain": DOMAIN})["particle_count"]
    rt.call("fluid.reset")
    after_persistent_reset = rt.call("fluid.get", {"domain": DOMAIN})["particle_count"]
    if persistent_count <= 0 or after_persistent_reset != persistent_count:
        failures.append("persistent seed did not reproduce on reset: {} -> {}".format(
            persistent_count, after_persistent_reset))

    rt.call("fluid.clear", {"domain": DOMAIN, "clear_seed": True})
    rt.call("fluid.reset")
    after_disarm_reset = rt.call("fluid.get", {"domain": DOMAIN})["particle_count"]
    if after_disarm_reset != 0:
        failures.append("clear_seed failed to disarm reset recipe: {} particles".format(
            after_disarm_reset))
    print("seed lifecycle: one-shot {} -> {}; persistent {} -> {}; disarmed -> {}".format(
        one_shot_count, after_one_shot_reset,
        persistent_count, after_persistent_reset,
        after_disarm_reset))

    rt.call("fluid.seed", dict(seed_payload, persistent=False))
    initial_count = rt.call("fluid.get", {"domain": DOMAIN})["particle_count"]
    checkpoints = []
    checkpoint_steps = {15, 30, 45, 60, 75, 90}
    for step in range(1, 91):
        rt.call("fluid.step", {"dt": DT})
        if step in checkpoint_steps:
            checkpoints.append((step, rt.call("fluid.get", {"domain": DOMAIN})))
    final = checkpoints[-1][1]

    if final["particle_count"] != initial_count:
        failures.append("particle count changed: {} -> {}".format(
            initial_count, final["particle_count"]))
    if final["granular_invalid"] != 0:
        failures.append("granular_invalid={}".format(final["granular_invalid"]))
    if final["granular_damaged"] == 0 or final["granular_max_damage"] <= 0.0:
        failures.append("impact produced no bond damage")
    if final["granular_damage_over_50"] > int(0.50 * max(initial_count, 1)):
        failures.append("severe damage became domain-wide instead of localizing near cracks")
    if final["granular_damage_over_90"] > int(0.25 * max(initial_count, 1)):
        failures.append("more than a quarter of the block reached near-total damage")
    if final["granular_detached"] > int(0.50 * max(initial_count, 1)):
        failures.append("more than half the cohesive block fully detached")
    preimpact = next(state for step, state in checkpoints if step == 30)
    if preimpact["granular_damaged"] != 0:
        failures.append("bond damage began during uniform free fall before impact")
    # Sub-threshold elastic motion is recoverable and is exactly what
    # Fracture Strain separates from damage. Keep a safety margin below that
    # physical onset instead of requiring an unrealistically exact zero from a
    # particle-grid transfer with jittered quadrature.
    preimpact_mean_limit = 0.50 * authored["granular_fracture_strain"]
    preimpact_max_limit = 0.90 * authored["granular_fracture_strain"]
    if preimpact["granular_mean_fracture_history"] > preimpact_mean_limit:
        failures.append("mean bond-opening history approached fracture during free fall: "
                        "{:.6f} > {:.6f}".format(
                            preimpact["granular_mean_fracture_history"],
                            preimpact_mean_limit))
    if preimpact["granular_max_fracture_history"] > preimpact_max_limit:
        failures.append("peak bond-opening history approached fracture during free fall: "
                        "{:.6f} > {:.6f}".format(
                            preimpact["granular_max_fracture_history"],
                            preimpact_max_limit))
    expected_solver_substeps = min(
        final["granular_required_substeps"], authored["granular_max_solver_substeps"])
    if final["granular_solver_substeps"] != expected_solver_substeps:
        failures.append("solver ran {} elastic substeps, expected {}".format(
            final["granular_solver_substeps"], expected_solver_substeps))
    if (final["granular_required_substeps"] <= authored["granular_max_solver_substeps"]
            and final["granular_effective_young_modulus"]
            < 0.95 * final["granular_requested_young_modulus"]):
        failures.append("full subcycling did not recover the authored Young modulus")

    print("particles: {} -> {}".format(initial_count, final["particle_count"]))
    print("fracture timeline:")
    for step, state in checkpoints:
        print("  step {:2d}: damaged={} damaged10={} detached={} bond_mean={:.6f} "
              "bond_max={:.6f}".format(
                  step, state["granular_damaged"],
                  state["granular_damage_over_10"],
                  state["granular_detached"],
                  state["granular_mean_fracture_history"],
                  state["granular_max_fracture_history"]))
    print("yielded={}, detached={}, damaged={}, invalid={}, max_damage={:.4f}".format(
        final["granular_yielded"], final["granular_detached"],
        final["granular_damaged"], final["granular_invalid"],
        final["granular_max_damage"]))
    print("damage bands: >=10% {}, >=50% {}, >=90% {}, mean={:.4f}".format(
        final["granular_damage_over_10"], final["granular_damage_over_50"],
        final["granular_damage_over_90"], final["granular_mean_damage"]))
    print("plastic strain: increment={:.6f}, accumulated={:.6f}".format(
        final["granular_max_plastic"],
        final["granular_max_accumulated_plastic"]))
    print("plastic strain mean accumulated={:.6f}".format(
        final["granular_mean_accumulated_plastic"]))
    print("maximum bond-opening history: max={:.6f}, mean={:.6f}".format(
        final["granular_max_fracture_history"],
        final["granular_mean_fracture_history"]))
    print("Young modulus: requested={:.0f} Pa, effective={:.0f} Pa, "
          "required_substeps={}, solver_substeps={}, capped={}".format(
              final["granular_requested_young_modulus"],
              final["granular_effective_young_modulus"],
              final["granular_required_substeps"],
              final["granular_solver_substeps"],
              final["granular_stiffness_capped"]))
    if failures:
        print("FAIL")
        for failure in failures:
            print(" - " + failure)
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
