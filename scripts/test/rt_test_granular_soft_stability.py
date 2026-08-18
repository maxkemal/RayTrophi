"""Granular soft-material stability regression.

Run from a separate terminal while RayTrophi Studio is open:

    python scripts\test\rt_test_granular_soft_stability.py

WHAT THIS PINS DOWN
-------------------
The elastic subcycle used to be sized from the elastic wave CFL alone. That
limit demands MORE substeps as the Young modulus goes UP, so it left a hole
exactly where the material is soft: any E at or below rho*(0.35h/dt)^2 -- about
1.1 kPa for h = 0.1 m at 24 fps -- returned a single substep at the full frame
dt. The stress kernel then integrated the deformation gradient as (I + dt*C)*F
with dt*||C|| of order one. That error is multiplicative, so nothing looked
wrong for a while: strain accumulated quietly frame over frame and then
discharged through the kernel's det(F) reset as a velocity burst. Reported as
"below 1000 the pile suddenly flings its particles".

The fix is two-layer, and this test checks BOTH layers plus the honesty of the
report:

  1. elasticStepInfo now also honours a strain-rate CFL measured from the
     previous frame's affine field, so a soft material gets a real subcycle.
  2. The stress kernel clamps dt*C itself and raises flag bit 16, so a step the
     subcycle could not cover survives AND says so.

★ THE SNEAKY FAILURE THIS GUARDS: a soft run that merely looks calm. Bounded
positions are not enough -- an under-resolved subcycle can sit quiet for
hundreds of steps. granular_strain_rate is the reading that moves first, so the
test watches its trend, not just its final value.

WHAT IT DOES NOT CLAIM
----------------------
It does not claim 800 Pa sand is physical. A pile that deep needs roughly
200 kPa to hold itself inside the small-strain regime, and the run is EXPECTED
to report granular_stiffness_below_load = True. Numerical stability and
material validity are separate questions; conflating them is what let the
original bug read as a mystery.
"""

import ctypes
from ctypes import wintypes
import json


PIPE_NAME = r"\\.\pipe\RayTrophiStudio"
DOMAIN = "GranularSoftStabilityTest"
DT = 1.0 / 24.0
STEPS = 240
SAMPLE_EVERY = 30

# Only used to turn a reported overburden back into the column height it came
# from, for the printed line. Mirrors kGranularDensity / the solver's gravity.
GRANULAR_DENSITY = 1600.0
GRAVITY = 9.81

# The regime the report came from: below the old wave-CFL threshold, so it used
# to receive exactly one full-frame substep.
SOFT_YOUNG = 800.0
# The shipped Sand default, as the control. Same scene, same assertions.
STIFF_YOUNG = 2.0e5


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


def require_isolated_scene(rt):
    """fluid.step advances every enabled domain, not just the named one."""
    domains = rt.call("fluid.list_domains")["domains"]
    if DOMAIN not in [item["name"] for item in domains]:
        rt.call("fluid.create_domain", {
            "name": DOMAIN,
            "type": "fluid",
            "domain_min": [-1.0, 0.0, -1.0],
            "domain_max": [1.0, 3.0, 1.0],
            "voxel_size": 0.05,
        })
        domains = rt.call("fluid.list_domains")["domains"]
    same_name = [item for item in domains if item["name"] == DOMAIN]
    if len(same_name) > 1:
        raise SystemExit(
            "Found {} domains named {!r}; name-based get would be ambiguous. "
            "Remove the duplicate and retry.".format(len(same_name), DOMAIN))
    others = [item for item in domains
              if item["name"] != DOMAIN and item.get("enabled", True)]
    if others:
        details = ", ".join("{} ({})".format(item["name"], item.get("type", "domain"))
                            for item in others)
        raise SystemExit(
            "This test needs an isolated scene because fluid.step advances every "
            "enabled domain. Active others: " + details)


def run_case(rt, label, young_modulus):
    rt.call("fluid.set_param", {
        "domain": DOMAIN,
        "backend": "vulkan",
        "boundary": "closed",
        "preset": "sand",
        "enabled": True,
        "granular_enabled": True,
        "granular_friction_angle": 35.0,
        "granular_cohesion": 0.0,
        "granular_dilatancy": 5.0,
        "granular_young_modulus": young_modulus,
        "granular_poisson_ratio": 0.25,
        "granular_tensile_cutoff": 0.0,
        "granular_hardening": 0.0,
        "granular_rebonding": False,
        "granular_max_solver_substeps": 16,
    })
    rt.call("fluid.clear", {"domain": DOMAIN, "clear_seed": True})
    rt.call("fluid.reset")
    rt.call("fluid.seed", {
        "domain": DOMAIN,
        "seed_min": [-0.40, 1.20, -0.40],
        "seed_max": [0.40, 2.20, 0.40],
        "particles_per_cell": 4,
        "replace": True,
        "persistent": False,
    })

    initial = rt.call("fluid.get", {"domain": DOMAIN})
    # ★ Sample EVERY step, not every Nth. The det(F) discharge this test exists
    # to catch is a one-step event; a stride of 30 would step right over it and
    # report a clean run. Only the printed table is thinned.
    samples = []
    for step in range(1, STEPS + 1):
        rt.call("fluid.step", {"dt": DT})
        samples.append((step, rt.call("fluid.get", {"domain": DOMAIN})))

    failures = []
    final = samples[-1][1]
    worst_invalid = max(s["granular_invalid"] for _, s in samples)
    peak_compaction = max(s["granular_compaction_capped"] for _, s in samples)

    if final["particle_count"] != initial["particle_count"]:
        failures.append("{}: particle count changed {} -> {}".format(
            label, initial["particle_count"], final["particle_count"]))
    # ★★★ THE REGRESSION. The stored-strain cap turns "this particle cannot
    # hold its stress" into permanent compaction instead of a one-step dump to
    # zero. A soft pile used to show 22 of these in 240 steps, and each one was
    # a small burst leaving the surface. Checked across ALL steps, not just the
    # last, because the run ends calm either way.
    if worst_invalid != 0:
        failures.append(
            "{}: {} particles hit the NaN/det(F) reset path (peak over the run). "
            "Stress is being discharged in one step - the pop is back".format(
                label, worst_invalid))

    # ── Layer 1: the subcycle actually covered the motion ────────────────────
    granted = final["granular_solver_substeps"]
    needed = final["granular_required_substeps"]
    if granted < min(needed, 16):
        failures.append("{}: solver ran {} substeps, needed {}".format(
            label, granted, needed))
    # ★ The regression itself. Before the fix a soft material returned
    # required_substeps == 1 because only the wave CFL was consulted; a settling
    # pile always has a non-zero velocity gradient, so the strain limb must ask
    # for more than one substep once the material is moving.
    if final["granular_strain_rate"] > 0.0 and final["granular_strain_substeps"] < 1:
        failures.append("{}: strain-rate limb reported no substep demand".format(label))
    if young_modulus <= 1.0e3 and needed <= 1 and final["granular_strain_rate"] > 5.0:
        failures.append(
            "{}: |C| = {:.1f} 1/s but the subcycle asked for one full-frame step "
            "- the wave-CFL-only hole is back".format(
                label, final["granular_strain_rate"]))

    # ── Layer 2: the shader clamp is a backstop, not the working mechanism ───
    # ★ Transient clamping AT IMPACT is expected and is not a failure. The host
    # sizes the subcycle from the PREVIOUS frame's C, and during free fall that
    # reading is legitimately zero -- the collision happens inside the frame the
    # measurement could not see. (Measured: 1993 particles clamped on the impact
    # step while |C| still read 0.00.) What must not happen is clamping that
    # persists once the pile is settled, which means the ceiling is too low.
    settled = samples[len(samples) // 2:]
    sustained_clamp = min(s["granular_strain_limited"] for _, s in settled)
    if sustained_clamp > 0:
        failures.append(
            "{}: {} particles were still being dt*C clamped on every settled "
            "step; the subcycle never covered the motion (raise "
            "granular_max_solver_substeps)".format(label, sustained_clamp))

    # ── Plastic compaction is the REPLACEMENT mechanism, so it must be live ──
    # A soft pile carrying 26 kPa on a 1 kPa skeleton has to compact somewhere.
    # If neither the reset nor the cap ever fires, the stress is going somewhere
    # unaccounted for and this test is measuring nothing.
    if young_modulus <= 1.0e3 and peak_compaction == 0 and worst_invalid == 0:
        failures.append(
            "{}: neither plastic compaction nor a reset ever fired on material "
            "far too soft for its load - the strain path is not being "
            "exercised, so this run proves nothing".format(label))

    # ── The explosion signature: |C| trending up without bound ──────────────
    # A settling pile's velocity gradient decays. A pile storing integration
    # error does the opposite, and does it long before positions look wrong.
    strain_rates = [state["granular_strain_rate"] for _, state in samples]
    early = max(strain_rates[:len(strain_rates) // 2])
    late = max(strain_rates[len(strain_rates) // 2:])
    shown = strain_rates[::SAMPLE_EVERY]
    if late > max(4.0 * early, 50.0):
        failures.append(
            "{}: |C| grew from {:.1f} to {:.1f} 1/s over the run - energy is "
            "accumulating, not settling".format(label, early, late))

    print("[{}] E={:.0f} Pa  particles {} -> {}".format(
        label, young_modulus, initial["particle_count"], final["particle_count"]))
    print("  substeps: wave={} strain={} required={} run={}".format(
        final["granular_wave_substeps"], final["granular_strain_substeps"],
        needed, granted))
    print("  |C|: {}".format(" -> ".join("{:.1f}".format(v) for v in shown)))
    print("  peak det(F) resets={}  peak plastic compaction={}  "
          "settled dt*C clamp={}  final yielded={}".format(
              worst_invalid, peak_compaction, sustained_clamp,
              final["granular_yielded"]))
    # ★★★ THE LOAD GATE MUST BE READ OVER THE RUN, NOT ON THE LAST FRAME.
    # overburden = column_height * rho * g, and column_height is the extent of
    # the MATERIAL. A pile that fails does not stay a pile: measured 2026-08-17,
    # the 800 Pa column went 1.05 m -> 0.0003 m over 120 steps, so its overburden
    # decayed 16480 Pa -> 5 Pa and the gate switched itself OFF at step ~75 --
    # exactly when the failure it predicts had finished happening.
    #
    # Asserting on the final frame therefore tested the collapsed puddle, not the
    # pile, and read a correct instrument as broken.
    peak_load = max(s["granular_overburden_pressure"] for _, s in samples)
    ever_below = any(s["granular_stiffness_below_load"] for _, s in samples)
    final_height = final["granular_overburden_pressure"] / (GRANULAR_DENSITY * GRAVITY)
    peak_height = peak_load / (GRANULAR_DENSITY * GRAVITY)
    print("  load: peak overburden={:.0f} Pa (h={:.2f} m) -> final {:.0f} Pa "
          "(h={:.4f} m), below_load ever={} final={}".format(
              peak_load, peak_height,
              final["granular_overburden_pressure"], final_height,
              ever_below, final["granular_stiffness_below_load"]))
    return failures, final, ever_below


def main():
    rt = Ipc()
    require_isolated_scene(rt)

    failures = []
    soft_failures, soft, soft_ever_below = run_case(rt, "soft", SOFT_YOUNG)
    failures += soft_failures
    stiff_failures, stiff, stiff_ever_below = run_case(rt, "stiff", STIFF_YOUNG)
    failures += stiff_failures

    # The validity gate must SEPARATE these two, otherwise it is decoration.
    # Soft is genuinely too weak to hold a metre of material and must say so;
    # the shipped default must not be dragged into the same warning.
    #
    # ★ Read over the RUN, not on the last frame: the soft pile collapses, and a
    # collapsed pile carries no overburden, so the gate correctly stops warning
    # about a load that no longer exists. The claim being tested is "it warned
    # while there was something to warn about".
    if not soft_ever_below:
        failures.append(
            "soft case never reported stiffness_below_load; the validity gate is "
            "not separating 'stable' from 'physical'")
    if stiff_ever_below:
        failures.append(
            "the shipped Sand default tripped the below-load gate - the threshold "
            "is too strict and will be ignored as noise")

    if failures:
        print("FAIL")
        for failure in failures:
            print(" - " + failure)
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
