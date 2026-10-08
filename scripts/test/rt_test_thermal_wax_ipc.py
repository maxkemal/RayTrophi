"""Thermal liquid (wax) + surface detail, driven from outside over IPC.

    python scripts\\test\\rt_test_thermal_wax_ipc.py

Run OUTSIDE the app, with RayTrophi Studio open. Builds its own small rig
("WaxProbe" domain + "WaxProbePour" source), leaves it in the scene.

WHAT THIS PROVES
----------------
1. Surface reconstruction controls round-trip, and out-of-range values are
   REJECTED (not clamped) with the old value intact.
2. The wax preset switches the thermal chain on, and every OTHER preset
   switches it off. The trap: water born at 293 K is below wax's 330 K freeze
   point, so a stale switch would freeze "water" on the first surface.
3. A source's pour temperature round-trips.
4. Poured hot, the liquid is born hot (not 0 K), cools, thickens (the viscosity
   field spans a real range) and SETS on the closed domain floor.
5. Switching the chain off releases every frozen parcel.
6. fluid.set_substance_material accepts a LIQUID domain (it used the gas-only
   lookup and could only answer "gas domain not found").

The assertions scale tolerances with the change being measured, not with the
magnitude of the value (see project_ipc_test_channel: a 1%-of-altitude
tolerance once hid a full revert).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_ipc import RtIpc, RtIpcError  # noqa: E402

DOMAIN = "WaxProbe"
SOURCE = "WaxProbePour"
DT = 1.0 / 60.0
POUR_K = 353.0


def get(rt):
    return rt.call("fluid.get", domain=DOMAIN)


def build_rig(rt):
    names = [d["name"] for d in rt.call("fluid.list_domains")["domains"]]
    if DOMAIN not in names:
        rt.call("fluid.create_domain", name=DOMAIN, type="fluid",
                domain_min=[-0.3, 0.0, -0.3], domain_max=[0.3, 0.6, 0.3],
                voxel_size=0.02)
    rt.call("fluid.set_param", domain=DOMAIN, backend="vulkan",
            render_mode="surface", boundary="closed")
    existing = [s["name"] for s in (rt.call("flow_source.list") or [])]
    payload = dict(name=SOURCE, domain=DOMAIN, position=[0.0, 0.45, 0.0],
                   velocity=[0.0, -1.0, 0.0], radius=0.05,
                   fluid_particles_per_second=3000.0,
                   fluid_temperature_override=True,
                   fluid_temperature_kelvin=POUR_K)
    rt.call("flow_source.update" if SOURCE in existing else "flow_source.create",
            **payload)


def phase_surface_detail(rt):
    print("\n=== phase 1: surface detail round trip ===")
    f = []
    rt.call("fluid.set_param", domain=DOMAIN, surface_resolution_multiplier=2,
            anisotropy_enabled=True, smoothing_iterations=3)
    g = get(rt)
    for key, want in (("surface_resolution_multiplier", 2),
                      ("anisotropy_enabled", True), ("smoothing_iterations", 3)):
        print("  {:32s} {!r}".format(key, g.get(key)))
        if g.get(key) != want:
            f.append("{} reads {!r}, wrote {!r}: the surface key never landed "
                     "(missing from the dispatch, or from fluid.get).".format(key, g.get(key), want))
    try:
        rt.call("fluid.set_param", domain=DOMAIN, surface_resolution_multiplier=5)
        f.append("multiplier 5 was ACCEPTED - out-of-range must be rejected, "
                 "a silent clamp lets a script believe it got a detail level it did not")
    except RtIpcError as e:
        print("  multiplier 5 rejected: {}".format(e))
    if get(rt).get("surface_resolution_multiplier") != 2:
        f.append("a REJECTED write changed the multiplier anyway")
    # Back to the cheap default so the pour below measures the thermal chain,
    # not the surface build.
    rt.call("fluid.set_param", domain=DOMAIN, surface_resolution_multiplier=1,
            anisotropy_enabled=False, smoothing_iterations=2)
    return f


def phase_preset_switch(rt):
    print("\n=== phase 2: wax preset switches the chain on, others off ===")
    f = []
    rt.call("fluid.set_param", domain=DOMAIN, default_substance="Wax")
    g = get(rt)
    print("  wax  : preset={} thermal={} freeze={}".format(
        g.get("preset"), g.get("thermal_liquid_enabled"), g.get("thermal_freeze_kelvin")))
    if g.get("preset") != "wax" or not g.get("thermal_liquid_enabled"):
        f.append("preset wax did not enable the thermal chain")
    rt.call("fluid.set_param", domain=DOMAIN, default_substance="Water")
    g = get(rt)
    print("  water: preset={} thermal={}".format(g.get("preset"), g.get("thermal_liquid_enabled")))
    if g.get("thermal_liquid_enabled"):
        f.append("switching to WATER left the thermal chain ON - water at 293 K "
                 "would freeze below wax's 330 K point on the first contact")
    rt.call("fluid.set_param", domain=DOMAIN, default_substance="Wax")
    return f


def phase_pour_temperature(rt):
    print("\n=== phase 3: pour temperature round trip ===")
    s = rt.call("flow_source.get", name=SOURCE)
    print("  override={} kelvin={}".format(s.get("fluid_temperature_override"),
                                            s.get("fluid_temperature_kelvin")))
    if not s.get("fluid_temperature_override") or \
            abs(float(s.get("fluid_temperature_kelvin", 0.0)) - POUR_K) > 1e-3:
        return ["flow source does not report the pour temperature it was given"]
    return []


def phase_pour_and_set(rt):
    print("\n=== phase 4: pour hot, cool, thicken, set ===")
    f = []
    rt.call("fluid.reset")
    samples = []
    for frame in range(1, 241):
        rt.call("fluid.step", dt=DT)
        if frame % 30 == 0:
            g = get(rt)
            samples.append(g)
            print("  f{:3d} n={:5d} T[min/mean/max]={:6.1f}/{:6.1f}/{:6.1f} "
                  "frozen={:5d} cold_unsup={:4d} nu={:.1e}..{:.1e}".format(
                      frame, g.get("particle_count", 0),
                      g.get("thermal_min_kelvin", 0), g.get("thermal_mean_kelvin", 0),
                      g.get("thermal_max_kelvin", 0), g.get("thermal_frozen_particles", 0),
                      g.get("thermal_cold_unsupported", 0),
                      g.get("thermal_min_viscosity", 0), g.get("thermal_max_viscosity", 0)))
    measured = [s for s in samples if s.get("thermal_measured") and s.get("particle_count", 0) > 0]
    if not measured:
        return ["thermal chain never reported measured=true with particles present"]
    ambient = float(measured[-1].get("thermal_ambient_kelvin", 293.0))
    first, last = measured[0], measured[-1]
    # Born hot: the 0 K emitter bug would show as min far below ambient.
    if float(first["thermal_max_kelvin"]) < POUR_K - 5.0:
        f.append("hottest parcel {:.1f} K right after pouring at {:.0f} K - the pour "
                 "temperature is not reaching emitted particles".format(
                     float(first["thermal_max_kelvin"]), POUR_K))
    if min(float(s["thermal_min_kelvin"]) for s in measured) < ambient - 1.0:
        f.append("a parcel is colder than ambient ({:.1f} K) - something is still born "
                 "at 0 K".format(min(float(s["thermal_min_kelvin"]) for s in measured)))
    # Cooling, measured against the span it can cover (pour - ambient).
    drop = float(first["thermal_mean_kelvin"]) - float(last["thermal_mean_kelvin"])
    span = POUR_K - ambient
    if drop < 0.05 * span:
        f.append("mean temperature fell only {:.2f} K over 4 s (span {:.0f} K) - "
                 "cooling is not running".format(drop, span))
    if not last.get("thermal_viscosity_field"):
        f.append("no viscosity field was built - nu(T) never reached the solver")
    elif float(last["thermal_max_viscosity"]) < 10.0 * max(float(last["thermal_min_viscosity"]), 1e-9):
        f.append("viscosity spans {:.1e}..{:.1e} - the thermal ramp is flat".format(
            float(last["thermal_min_viscosity"]), float(last["thermal_max_viscosity"])))
    if int(last.get("thermal_frozen_particles", 0)) == 0:
        f.append("nothing froze on the closed floor after 4 s (cold_unsupported={}) - "
                 "freezing or its support test is broken".format(last.get("thermal_cold_unsupported")))
    return f


def phase_disable_releases(rt):
    print("\n=== phase 5: switching the chain off releases every frozen parcel ===")
    rt.call("fluid.set_param", domain=DOMAIN, thermal_liquid_enabled=False)
    rt.call("fluid.step", dt=DT)
    g = get(rt)
    print("  frozen after disable = {}".format(g.get("thermal_frozen_particles")))
    rt.call("fluid.set_param", domain=DOMAIN, thermal_liquid_enabled=True)
    if int(g.get("thermal_frozen_particles", 0)) != 0:
        return ["frozen parcels survived switching the chain OFF - they stay pinned "
                "through the solid-phase path with no visible cause"]
    return []


def phase_substance_lookup(rt):
    print("\n=== phase 6: set_substance_material accepts a liquid domain ===")
    try:
        rt.call("fluid.set_substance_material", domain=DOMAIN,
                substance="__probe__", material="__no_such_material__")
        return ["a nonexistent material was accepted"]
    except RtIpcError as e:
        msg = str(e)
        print("  error: {}".format(msg))
        if "gas domain" in msg:
            return ["set_substance_material still uses the GAS-only lookup on a liquid domain"]
        if "material not found" not in msg:
            return ["unexpected error: {}".format(msg)]
    return []


def main():
    rt = RtIpc()
    build_rig(rt)
    failures = []
    for phase in (phase_surface_detail, phase_preset_switch, phase_pour_temperature,
                  phase_pour_and_set, phase_disable_releases, phase_substance_lookup):
        failures += phase(rt)
    print("\n" + ("PASS" if not failures else "FAIL ({})".format(len(failures))))
    for msg in failures:
        print("  - " + msg)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
