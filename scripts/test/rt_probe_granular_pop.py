"""Read-only probe: what IS the small burst a soft granular pile releases?

    python scripts\test\rt_probe_granular_pop.py [DomainName]

Play the simulation in the app; this polls fluid.get and prints one row per
sample. It NEVER writes: no set_param, no seed, no step, no reset. Safe to point
at a scene you are working in.

WHY THIS EXISTS
---------------
A soft pile (E ~ 1 kPa, no cohesion) settles like wet mud and then, every so
often, lets something go with a small pop. Three different mechanisms produce
that same picture, and the eye cannot separate them:

  (a) A REAL compaction pocket. K = E/(3(1-2nu)) is a few hundred Pa, so the
      material must compress enormously before it pushes back. Regions load up
      and relieve. This is foam behaviour, and at E = 1 kPa it is the honest
      answer -- the material really is that soft.
  (b) The dt*C CLAMP in the stress kernel (flag bit 16). The subcycle failed to
      cover the motion and the shader held the step together. Numerical.
  (c) The det(F) RESET (flag bit 4). A particle compacted past the kernel's
      floor, so its stress was dumped to zero and its deformation gradient set
      back to identity, in one step. Numerical, and a step-function discharge --
      which is what actually reads as a "pop".

★ THE DISCRIMINATOR IS COINCIDENCE IN TIME, NOT MAGNITUDE. (a) shows no spike
in either counter. (b) and (c) name themselves. A row where invalid or
strain-clamped jumps on the same sample as the visible burst settles the
question; a burst with both counters flat means the physics is doing it and the
only fix is a stiffer material.

★★ A COUNTER THAT READS ZERO EVERY SAMPLE IS ALSO A RESULT -- it is what rules
(b) and (c) out. Do not stop polling because "nothing is happening".
"""

import ctypes
from ctypes import wintypes
import json
import sys
import time


PIPE_NAME = r"\\.\pipe\RayTrophiStudio"
POLL_SECONDS = 0.25


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
                "  - The pipe serves ONE client: a PowerShell session that never called\n"
                "    Disconnect-RtIpc still holds it. Close that session and retry.")
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


def pick_domain(rt, requested):
    domains = rt.call("fluid.list_domains")["domains"]
    granular = [d for d in domains if d.get("granular_enabled")]
    if requested:
        match = [d for d in domains if d["name"] == requested]
        if not match:
            raise SystemExit("No domain named {!r}. Present: {}".format(
                requested, ", ".join(d["name"] for d in domains) or "(none)"))
        return match[0]["name"]
    if not granular:
        raise SystemExit(
            "No granular domain in the scene. Present: " +
            (", ".join(d["name"] for d in domains) or "(none)"))
    if len(granular) > 1:
        raise SystemExit("Several granular domains; name one: " +
                         ", ".join(d["name"] for d in granular))
    return granular[0]["name"]


def main():
    requested = sys.argv[1] if len(sys.argv) > 1 else None
    rt = Ipc()
    domain = pick_domain(rt, requested)

    state = rt.call("fluid.get", {"domain": domain})
    if "granular_strain_limited" not in state:
        raise SystemExit(
            "This build predates the strain-rate instrumentation "
            "(granular_strain_limited missing). Rebuild before probing.")

    print("domain: {}".format(domain))
    print("E requested={:.0f} Pa  effective={:.0f} Pa  poisson={:.2f}  cohesion={:.0f} Pa".format(
        state["granular_requested_young_modulus"],
        state["granular_effective_young_modulus"],
        state.get("granular_poisson_ratio", 0.0),
        state.get("granular_cohesion", 0.0)))
    print("overburden={:.0f} Pa  E needed for small strain={:.0f} Pa  below_load={}".format(
        state["granular_overburden_pressure"],
        state["granular_young_modulus_for_load"],
        state["granular_stiffness_below_load"]))
    print()
    print("  time   parts   |C|   sub(w/s/run)  yield  detach  INVALID  CLAMPED   note")
    print("  " + "-" * 74)

    started = time.time()
    previous = None
    samples = 0
    invalid_events = 0
    clamp_events = 0
    quiet_bursts = 0
    peak_strain_rate = 0.0
    try:
        while True:
            state = rt.call("fluid.get", {"domain": domain})
            strain_rate = state["granular_strain_rate"]
            peak_strain_rate = max(peak_strain_rate, strain_rate)
            invalid = state["granular_invalid"]
            clamped = state["granular_strain_limited"]

            note = ""
            if previous is not None:
                # A burst is a jump in the velocity gradient. Attribute it by
                # what else moved on the SAME sample.
                burst = strain_rate > max(2.0 * previous["rate"], 1.0)
                grew_invalid = invalid > previous["invalid"]
                grew_clamped = clamped > previous["clamped"]
                if grew_invalid:
                    invalid_events += 1
                if grew_clamped:
                    clamp_events += 1
                if burst and grew_invalid:
                    note = "<< POP = det(F) RESET (c) - numerical"
                elif burst and grew_clamped:
                    note = "<< POP = dt*C CLAMP (b) - subcycle too coarse"
                elif burst:
                    quiet_bursts += 1
                    note = "<< POP with both counters flat = COMPACTION (a)"
                elif grew_invalid:
                    note = "det(F) reset (no visible burst)"
                elif grew_clamped:
                    note = "dt*C clamped (no visible burst)"

            print("  {:5.1f}s  {:6d}  {:5.1f}   {:2d}/{:2d}/{:2d}      {:5d}  {:6d}  {:7d}  {:7d}   {}".format(
                time.time() - started, state["particle_count"], strain_rate,
                state["granular_wave_substeps"], state["granular_strain_substeps"],
                state["granular_solver_substeps"],
                state["granular_yielded"], state["granular_detached"],
                invalid, clamped, note))

            previous = {"rate": strain_rate, "invalid": invalid, "clamped": clamped}
            samples += 1
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print()
        print("samples={}  peak |C|={:.1f} 1/s".format(samples, peak_strain_rate))
        print("det(F) resets seen on {} samples, dt*C clamps on {}, "
              "bursts with neither on {}".format(
                  invalid_events, clamp_events, quiet_bursts))
        if invalid_events == 0 and clamp_events == 0:
            print("VERDICT: no numerical discharge in this window. The pops are "
                  "compaction relief (a) - the material really is this soft, and "
                  "only a stiffer E changes it.")
        elif invalid_events > 0:
            print("VERDICT: det(F) resets are firing. Stress is being dumped to "
                  "zero in one step. Raise E toward granular_young_modulus_for_load; "
                  "if that is not acceptable, the compaction floor in "
                  "sim_fluid_granular_stress_update.comp needs a real volumetric "
                  "plasticity cap instead of a reset.")
        else:
            print("VERDICT: the dt*C clamp is carrying the step. Raise "
                  "granular_max_solver_substeps until CLAMPED stays 0.")


if __name__ == "__main__":
    main()
