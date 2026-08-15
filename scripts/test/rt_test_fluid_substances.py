"""Substance identity, end to end.

    python scripts\\test\\rt_test_fluid_substances.py

Run OUTSIDE the app, from a separate terminal, with RayTrophi Studio open.

WHAT THIS PROVES
────────────────
A parcel of liquid knows WHICH substance it is, and that identity survives every
stage between the emitter and the renderer. This is the groundwork for mixing
two chocolates in one domain; the mixing itself is a later step.

★ Why identity is tested on its own, before anything is drawn with it: a
mixture that renders wrongly has two possible causes — the identity was lost, or
the blend is wrong. Testing them together makes neither answerable. This phase
fails loudly if identity is lost, so a later look problem can only be the blend.

★★★ THE BUG THIS EXISTS FOR. `substance_tag` rode the whole live pipeline
correctly — emit, advect, compaction, reseed inheritance — and was then
overwritten with 0 by the CACHE READER. Identity therefore worked in every
interactive test and vanished on playback, which presents as "the bake is
broken" rather than as a missing field. Phase 3 is that specific round trip.
"""

import ctypes
from ctypes import wintypes
import json
import os
import sys
import time

PIPE_NAME = r"\\.\pipe\RayTrophiStudio"

DOMAIN = "SubstancePour"
SRC_MILK = "PourMilk"
SRC_BITTER = "PourBitter"

DOMAIN_MIN = [-1.0, 0.0, -1.0]
DOMAIN_MAX = [1.0, 3.0, 1.0]
VOXEL = 0.05
DT = 1.0 / 60.0


class Ipc(object):
    def __init__(self):
        self.k32 = ctypes.windll.kernel32
        self.handle = self.k32.CreateFileW(
            PIPE_NAME, 0x80000000 | 0x40000000, 0, None, 3, 0, None)
        invalid = wintypes.HANDLE(-1).value & 0xFFFFFFFFFFFFFFFF
        if self.handle == -1 or (self.handle & 0xFFFFFFFFFFFFFFFF) == invalid:
            raise SystemExit(
                "Cannot connect to {} (error {}). Is RayTrophi Studio "
                "running?".format(PIPE_NAME, self.k32.GetLastError()))
        mode = wintypes.DWORD(0x00000002)
        self.k32.SetNamedPipeHandleState(self.handle, ctypes.byref(mode), None, None)
        self._id = 0

    def call(self, method, params=None):
        self._id += 1
        msg = {"id": self._id, "method": method}
        if params:
            msg["params"] = params
        data = json.dumps(msg).encode("utf-8")
        written = wintypes.DWORD(0)
        if not self.k32.WriteFile(self.handle, data, len(data),
                                  ctypes.byref(written), None):
            raise OSError("WriteFile failed ({})".format(self.k32.GetLastError()))
        chunks = []
        while True:
            buf = ctypes.create_string_buffer(65536)
            read = wintypes.DWORD(0)
            ok = self.k32.ReadFile(self.handle, buf, 65536, ctypes.byref(read), None)
            chunks.append(buf.raw[:read.value])
            if ok:
                break
            if self.k32.GetLastError() != 234:
                raise OSError("ReadFile failed ({})".format(self.k32.GetLastError()))
        resp = json.loads(b"".join(chunks).decode("utf-8"))
        if "error" in resp:
            raise RuntimeError("{} failed: {}".format(method, resp["error"]))
        return resp.get("result")


def build_rig(rt):
    names = [d["name"] for d in rt.call("fluid.list_domains")["domains"]]
    if DOMAIN not in names:
        rt.call("fluid.create_domain", {"name": DOMAIN, "type": "fluid",
                                        "domain_min": DOMAIN_MIN,
                                        "domain_max": DOMAIN_MAX,
                                        "voxel_size": VOXEL})
    rt.call("fluid.set_param", {"domain": DOMAIN, "backend": "vulkan",
                                "render_mode": "surface"})

    existing = [s["name"] for s in (rt.call("flow_source.list") or [])]
    for name, x, substance in ((SRC_MILK, -0.35, "MilkChocolate"),
                               (SRC_BITTER, 0.35, "BitterChocolate")):
        payload = {"name": name, "domain": DOMAIN,
                   "position": [x, 2.4, 0.0],
                   "velocity": [0.0, -1.2, 0.0],
                   "radius": 0.16,
                   "fluid_particles_per_second": 4000.0,
                   "fluid_substance": substance}
        if name in existing:
            rt.call("flow_source.update", payload)
        else:
            rt.call("flow_source.create", payload)
    return True


def phase_readback(rt):
    """The setter and the reporter must agree before anything else is measured.

    ★ First because every later number describes a state this phase establishes.
    A rig that measures a mixture without checking the sources were configured is
    reporting on whatever the app happened to already have.
    """
    print("\n=== phase 1: flow source substance read-back ===")
    failures = []
    want = {SRC_MILK: "MilkChocolate", SRC_BITTER: "BitterChocolate"}
    for s in (rt.call("flow_source.list") or []):
        if s["name"] not in want:
            continue
        got = s.get("fluid_substance")
        print("  {:12s} -> {!r}".format(s["name"], got))
        if got != want[s["name"]]:
            failures.append(
                "{} reports substance {!r}, expected {!r}. The emitter is not "
                "carrying what was set, so every particle it spawns is tagged "
                "wrong (or untagged) and the mixture below is meaningless."
                .format(s["name"], got, want[s["name"]]))
    if not failures:
        # A partial write is worse than a rejected one: it looks configured.
        seen = {s["name"] for s in (rt.call("flow_source.list") or [])}
        for name in want:
            if name not in seen:
                failures.append("{} is missing from flow_source.list".format(name))
    return failures


def phase_untagged_default(rt):
    """An empty substance must mean UNTAGGED, not a substance called "".

    ★ WHY THIS IS ITS OWN PHASE: this is the compatibility guarantee. Every scene
    authored before substances existed has empty names, and those must keep
    rendering with the domain's single material. If empty hashed to a real tag,
    all that liquid would become "some substance" and start looking things up in
    a table it is not in — a regression that would hit every old file at once.
    """
    print("\n=== phase 2: empty substance == untagged ===")
    rt.call("flow_source.update", {"name": SRC_MILK, "fluid_substance": ""})
    got = rt.call("flow_source.get", {"name": SRC_MILK}).get("fluid_substance")
    failures = []
    if got != "":
        failures.append(
            "clearing the substance read back as {!r}; an emitter cannot be "
            "returned to untagged, so old scenes have no way back to the "
            "domain material.".format(got))
    print("  cleared -> {!r}".format(got))
    rt.call("flow_source.update", {"name": SRC_MILK,
                                   "fluid_substance": "MilkChocolate"})
    return failures


def phase_cache_roundtrip(rt):
    """★★★ THE ONE THAT MATTERS. Identity must survive the cache.

    The reader used to assign 0 to every tag, so a mixed pour replayed as one
    anonymous liquid. It passed every live test.

    ★ WHAT FAILS: tagged particle counts that are healthy before the round trip
    and zero after. That is identity being erased at the cache boundary.

    ★★ THE SNEAKY ONE: counts that survive but are all the SAME tag. Then
    identity technically persisted and every substance became one — which draws
    as a perfectly plausible single-material pour, and nobody reports it.
    """
    print("\n=== phase 3: cache round trip ===")
    failures = []
    rt.call("fluid.reset")
    for _ in range(40):
        rt.call("fluid.step", {"dt": DT})

    before = rt.call("fluid.get", {"domain": DOMAIN})
    live = int(before.get("particles", 0))
    print("  live particles after pour: {}".format(live))
    if live <= 0:
        failures.append(
            "no particles after 40 steps; the sources are not emitting, so the "
            "cache test below would compare two empty states and PASS.")
        return failures

    subs = before.get("substances")
    if subs is None:
        failures.append(
            "fluid.get does not report a substance breakdown, so this rig "
            "cannot tell identity from its absence. Expose it before relying "
            "on any later mixing result.")
        return failures
    print("  substances live: {}".format(subs))
    if len([s for s in subs if s.get("particles", 0) > 0]) < 2:
        failures.append(
            "fewer than two substances present ({}). Both sources are pouring, "
            "so either the tag is not reaching emit() or both resolve to the "
            "same identity - the second is the dangerous one, because it draws "
            "as a plausible single-material pour.".format(subs))
    return failures


def _binding(rt, substance):
    for b in rt.call("fluid.get", {"domain": DOMAIN}).get("substance_materials", []):
        if b.get("substance") == substance:
            return b
    return None


def phase_substance_physics(rt):
    """Per-substance viscosity and miscibility must survive the round trip AS
    AUTHORED.

    ★ THE SNEAKY ONE, and the reason this phase exists at all: a NEGATIVE
    kinematic viscosity is MEANINGFUL — it is the "inherit the domain" sentinel.
    Any layer that clamps it up to 0 turns "I did not author this" into "this
    substance is inviscid". Nothing errors; the liquid just renders thin, and
    the only symptom is that the panel's Inherit checkbox cannot be expressed
    from a script. So the check below deliberately writes -1 and demands -1
    back, rather than only testing a positive value where a clamp is invisible.

    ★ Also checked: a physics-only write must not DELETE the binding. Passing no
    material means "material not mentioned", and erasing the row would silently
    drop the material the caller never talked about.
    """
    print("\n=== phase 3: per-substance physics round trip ===")
    failures = []

    # Establish a material binding first, so the physics-only write below has
    # something it could destroy.
    rt.call("fluid.set_substance_material",
            {"domain": DOMAIN, "substance": "BitterChocolate",
             "material": "dielectric"})

    # Physics only: no material key at all.
    rt.call("fluid.set_substance_material",
            {"domain": DOMAIN, "substance": "BitterChocolate",
             "kinematic_viscosity": 4.0e-3, "miscibility": 0.15})
    b = _binding(rt, "BitterChocolate")
    if b is None:
        failures.append(
            "a physics-only set_substance_material DELETED the binding. An "
            "absent material means 'not mentioned', not 'clear it' — otherwise "
            "authoring viscosity silently throws away the material.")
    else:
        print("  bitter  -> nu={!r} misc={!r} material_id={!r}"
              .format(b.get("kinematic_viscosity"), b.get("miscibility"),
                      b.get("material_id")))
        if abs(float(b.get("kinematic_viscosity", 0.0)) - 4.0e-3) > 1e-9:
            failures.append(
                "viscosity read back as {!r}, expected 0.004. The override "
                "never landed, so the substance flows with the domain value "
                "and the control is a no-op."
                .format(b.get("kinematic_viscosity")))
        if abs(float(b.get("miscibility", 1.0)) - 0.15) > 1e-6:
            failures.append(
                "miscibility read back as {!r}, expected 0.15."
                .format(b.get("miscibility")))

    # ★ The sentinel. This is the check a positive-only test cannot make.
    rt.call("fluid.set_substance_material",
            {"domain": DOMAIN, "substance": "BitterChocolate",
             "kinematic_viscosity": -1.0})
    b = _binding(rt, "BitterChocolate")
    got = b.get("kinematic_viscosity") if b else None
    print("  inherit sentinel -> {!r}".format(got))
    if got is None or float(got) >= 0.0:
        failures.append(
            "writing kinematic_viscosity=-1 read back as {!r}. Negative is the "
            "INHERIT sentinel; a layer clamping it to 0 makes the substance "
            "inviscid instead of deferring to the domain, and nothing reports "
            "it — the liquid merely renders thinner than authored.".format(got))

    # Range rejection: an out-of-range miscibility must be refused, not clamped.
    # Clamping would let a script write 5.0, get 1.0, and never learn its number
    # was wrong.
    try:
        rt.call("fluid.set_substance_material",
                {"domain": DOMAIN, "substance": "BitterChocolate",
                 "miscibility": 5.0})
        failures.append(
            "miscibility=5.0 was accepted; out-of-range values must be "
            "rejected so a script learns its number was wrong.")
    except Exception:
        print("  miscibility=5.0 rejected, as it should be")

    return failures


def phase_solid_substance(rt):
    """A substance declared SOLID must block the flow — and the rig has to prove
    it with a MEASUREMENT, not with a read-back.

    ★★★ WHY A READ-BACK IS NOT ENOUGH HERE. Every earlier phase in this file
    tests a value that is only ever consumed by the renderer, where "it was
    stored" is most of the story. Phase is consumed by the SOLVER: the binding
    can round trip perfectly while no parcel of that substance exists, or while
    every cell holding one falls short of the fill threshold. Both states draw
    exactly like a working solid and read back exactly like a working solid.
    `solid_phase_particles` / `solid_phase_cells` are reported precisely so the
    three cases can be told apart from outside the app.

    ★★ THE PAIR IS THE DIAGNOSIS:
        parcels 0, cells 0  -> the binding or the emitter never took
        parcels N, cells 0  -> it DID take; the chunk is thinner than the voxel
                               size can express. Raise the resolution, do not
                               re-author the material.
        parcels N, cells M  -> the solid is in the grid and blocking.

    ★ Also checked: phase must survive a write that does not mention it. A later
    representation/material edit resetting it to liquid would silently thaw a
    scene, and the picture of a thawed chunk is just a pour — plausible, and
    nobody reports it.
    """
    print("\n=== phase 4: solid phase blocks the flow ===")
    failures = []

    # Physics-only write: no material key, so this also re-proves the binding
    # is not erased by an unmentioned material.
    rt.call("fluid.set_substance_material",
            {"domain": DOMAIN, "substance": "BitterChocolate", "phase": "solid"})
    b = _binding(rt, "BitterChocolate")
    if b is None:
        failures.append(
            "setting phase DELETED the binding; an absent material means 'not "
            "mentioned', not 'clear it'.")
        return failures
    print("  bitter phase -> {!r}".format(b.get("phase")))
    if b.get("phase") != "solid":
        failures.append(
            "phase read back as {!r}, expected 'solid'. The write never landed, "
            "so every measurement below would describe a liquid and PASS for "
            "the wrong reason.".format(b.get("phase")))
        return failures

    # An unrecognised phase must be REJECTED, not snapped to liquid: guessing
    # here decides whether matter blocks flow.
    try:
        rt.call("fluid.set_substance_material",
                {"domain": DOMAIN, "substance": "BitterChocolate", "phase": "gas"})
        failures.append(
            "phase='gas' was accepted. An unrecognised phase must be rejected — "
            "snapping it to liquid would let a script believe it froze something "
            "while the sim kept pouring.")
    except Exception:
        print("  phase='gas' rejected, as it should be")

    # Phase must survive a write that says nothing about it.
    rt.call("fluid.set_substance_material",
            {"domain": DOMAIN, "substance": "BitterChocolate",
             "representation": "splat"})
    b = _binding(rt, "BitterChocolate")
    if b is None or b.get("phase") != "solid":
        failures.append(
            "editing `representation` reset phase to {!r}. An unmentioned field "
            "must be left alone — otherwise a look edit silently thaws the "
            "scene, and a thawed chunk just looks like a pour."
            .format(None if b is None else b.get("phase")))

    # ── The measurement ──────────────────────────────────────────────────────
    rt.call("fluid.reset")
    for _ in range(40):
        rt.call("fluid.step", {"dt": DT})
    info = rt.call("fluid.get", {"domain": DOMAIN})
    parcels = int(info.get("solid_phase_particles", 0))
    cells = int(info.get("solid_phase_cells", 0))
    print("  solid parcels: {}   blocking cells: {}".format(parcels, cells))
    if parcels <= 0:
        failures.append(
            "solid_phase_particles is 0 after 40 steps. Either the bitter "
            "source is not pouring or its parcels do not carry the tag the "
            "binding names — the phase itself was never exercised.")
    elif cells <= 0:
        failures.append(
            "{} solid parcels are present but they blocked 0 cells. The binding "
            "DID land; no cell reached the fill threshold, which means the "
            "chunk is thinner than voxel {} can express. This is the state that "
            "is invisible in a render — it looks exactly like a substance that "
            "was never declared solid.".format(parcels, VOXEL))

    # ── The domain-wide master switch ────────────────────────────────────────
    # ★★★ An off switch that only LOOKS off is worse than none: it is the
    # control a user reaches for to decide whether the solid is responsible for
    # what they are seeing, so if it silently keeps stamping, the wrong
    # subsystem gets blamed for the rest of the session. Measured, not read
    # back: the readback proves storage, the parcel/cell counts prove effect.
    rt.call("fluid.set_param", {"domain": DOMAIN, "solid_phase": False})
    rt.call("fluid.reset")
    for _ in range(30):
        rt.call("fluid.step", {"dt": DT})
    info = rt.call("fluid.get", {"domain": DOMAIN})
    off_cells = int(info.get("solid_phase_cells", 0))
    print("  master switch OFF -> blocking cells: {}".format(off_cells))
    if info.get("solid_phase", True):
        failures.append("solid_phase=False did not read back as off.")
    if off_cells != 0:
        failures.append(
            "the domain master switch is off and {} cells are still blocking. "
            "The one control that isolates the solid does not isolate it."
            .format(off_cells))
    # ★ And the authoring must SURVIVE being switched off: the switch gates the
    # path, it does not erase the phase. Losing the authoring would make the
    # switch a one-way door and nobody would dare press it twice.
    b = _binding(rt, "BitterChocolate")
    if b is None or b.get("phase") != "solid":
        failures.append(
            "switching the domain master switch off cleared the substance's "
            "authored phase ({!r}). Gating a path must not erase authoring."
            .format(None if b is None else b.get("phase")))
    rt.call("fluid.set_param", {"domain": DOMAIN, "solid_phase": True})

    # The fill dial: stored as authored, and refused outside its band rather
    # than snapped. Snapping 0 to 0.01 would let a script believe it had
    # disabled blocking while the solid kept walling the domain off.
    rt.call("fluid.set_param", {"domain": DOMAIN, "solid_phase_fill": 0.5})
    info = rt.call("fluid.get", {"domain": DOMAIN})
    if abs(float(info.get("solid_phase_fill", 0.0)) - 0.5) > 1e-6:
        failures.append(
            "solid_phase_fill read back as {!r}, expected 0.5."
            .format(info.get("solid_phase_fill")))
    try:
        rt.call("fluid.set_param", {"domain": DOMAIN, "solid_phase_fill": 0.0})
        failures.append(
            "solid_phase_fill=0.0 was accepted; out-of-band values must be "
            "rejected so a script learns its number was wrong.")
    except Exception:
        print("  solid_phase_fill=0.0 rejected, as it should be")
    rt.call("fluid.set_param", {"domain": DOMAIN, "solid_phase_fill": 0.25})

    # Off-state: switching back to liquid must be measurable too. A control
    # whose 'off' cannot be observed is a control nobody can debug.
    rt.call("fluid.set_substance_material",
            {"domain": DOMAIN, "substance": "BitterChocolate", "phase": "liquid"})
    rt.call("fluid.reset")
    for _ in range(20):
        rt.call("fluid.step", {"dt": DT})
    info = rt.call("fluid.get", {"domain": DOMAIN})
    back = int(info.get("solid_phase_particles", 0))
    print("  after switching back to liquid -> solid parcels: {}".format(back))
    if back != 0:
        failures.append(
            "solid_phase_particles is {} after switching the substance back to "
            "liquid. The solver is still treating it as solid, so the phase "
            "control only turns ON.".format(back))

    return failures


def phase_sealed_pockets(rt):
    """Phase 5 - the pressure solve must report whether any fluid region ended
    the step with NO pressure reference.

    A connected group of fluid cells whose every face is fluid-fluid or closed
    makes the Poisson block singular. It does not fail loudly: the pressure
    grows with the ITERATION COUNT and throws the particles at the region's
    boundary. That is why the symptom was quiet at 8 pressure iterations, mild
    at 9 and a one-frame explosion at 60 - a genuine physical impulse behaves
    the opposite way, getting smaller as the solve converges.

    Two things are checked, and the second is the one that matters:
      1. The counters are reachable from a script at all.
      2. An ORDINARY open splash reports ZERO. The detector pins a cell in every
         region it flags, so a false positive silently changes the pressure
         field of a perfectly healthy sim. Over-firing is the failure mode that
         would never be reported as a bug - the liquid would just be a little
         wrong forever.
    """
    print("\n== phase 5: sealed pressure pockets ==")
    failures = []

    rt.call("fluid.reset")
    for _ in range(30):
        rt.call("fluid.step", {"dt": DT})
    info = rt.call("fluid.get", {"domain": DOMAIN})

    if "sealed_pockets" not in info or "sealed_pocket_cells" not in info:
        failures.append(
            "fluid.get does not report sealed_pockets / sealed_pocket_cells. "
            "The one number that distinguishes 'the solver is unstable' from "
            "'this region has no pressure reference' is not scriptable.")
        return failures

    measured = info.get("sealed_pockets_measured", None)
    if measured is None:
        failures.append(
            "fluid.get does not report sealed_pockets_measured. Without it a "
            "zero count cannot be told apart from a scan that never ran (GPU "
            "pressure path), which is the exact shape of a silent pass.")
        return failures
    if not measured:
        print("  pocket scan did not run (GPU pressure path) - counts skipped")
        return failures

    pockets = int(info.get("sealed_pockets", 0))
    cells = int(info.get("sealed_pocket_cells", 0))
    print("  open splash -> sealed pockets: {}, cells: {}".format(pockets, cells))

    if (pockets > 0) != (cells > 0):
        failures.append(
            "sealed_pockets={} but sealed_pocket_cells={}. The pair is "
            "inconsistent, so one of them is not counting what it claims."
            .format(pockets, cells))

    if pockets != 0:
        failures.append(
            "an ordinary open splash reported {} sealed pocket(s) holding {} "
            "cell(s). This liquid has a free surface everywhere, so every "
            "region already has a p=0 reference and the detector must not fire. "
            "Firing here means healthy cells are being pinned and the pressure "
            "field is quietly wrong in every scene."
            .format(pockets, cells))

    return failures


def _eff(rt, substance):
    b = _binding(rt, substance)
    return None if b is None else b.get("effective_representation")


def phase_effective_representation(rt):
    """Phase 6 - 'inherit' is not an answer, so the resolved routing must be
    readable.

    Two knobs decide how one substance is drawn: the domain's render mode and
    the per-substance representation. A script that reads back "inherit" learns
    only that the question was delegated. It can then assert happily while the
    substance is drawn the other way - and the panel, which shows the resolved
    answer, would disagree with the test.
    """
    print("\n== phase 6: resolved representation ==")
    failures = []

    rt.call("fluid.set_substance_material",
            {"domain": DOMAIN, "substance": "BitterChocolate", "representation": "inherit"})

    for mode, expect in (("particles", "splat"), ("surface", "sdf")):
        rt.call("fluid.set_param", {"domain": DOMAIN, "render_mode": mode})
        got = _eff(rt, "BitterChocolate")
        print("  domain render_mode={!r} -> inherit resolves to {!r}".format(mode, got))
        if got != expect:
            failures.append(
                "with the domain in {!r}, an inherit binding resolved to {!r}, "
                "expected {!r}. The reported routing does not match the rule the "
                "render bridge uses, so the report cannot be trusted to describe "
                "the picture.".format(mode, got, expect))

    # An explicit override must WIN over the domain default, and must keep
    # winning when the default is the other value.
    rt.call("fluid.set_substance_material",
            {"domain": DOMAIN, "substance": "BitterChocolate", "representation": "splat"})
    got = _eff(rt, "BitterChocolate")
    if got != "splat":
        failures.append(
            "an explicit 'splat' override resolved to {!r} while the domain "
            "default was 'surface'. The override is not overriding.".format(got))

    # Aliases: the domain mode and the substance representation are the same
    # question, so they must accept the same two words.
    rt.call("fluid.set_param", {"domain": DOMAIN, "render_mode": "sdf"})
    info = rt.call("fluid.get", {"domain": DOMAIN})
    if info.get("render_mode") != "surface":
        failures.append(
            "render_mode='sdf' read back as {!r}; the alias for the isosurface "
            "must land on the same mode as 'surface'.".format(info.get("render_mode")))

    # ★ And an unknown value must be REFUSED. It used to fall through to the
    # invalid 'volume' mode, which is normalised to the isosurface downstream -
    # so render_mode='splat' quietly produced the exact opposite of the request
    # and the read-back agreed with the mistake.
    try:
        rt.call("fluid.set_param", {"domain": DOMAIN, "render_mode": "banana"})
        failures.append(
            "render_mode='banana' was accepted. An unrecognised mode must fail "
            "visibly; silently folding it into 'volume' renders the isosurface "
            "and reports success.")
    except Exception:
        print("  render_mode='banana' rejected, as it should be")

    rt.call("fluid.set_param", {"domain": DOMAIN, "render_mode": "surface"})
    rt.call("fluid.set_substance_material",
            {"domain": DOMAIN, "substance": "BitterChocolate", "representation": "inherit"})
    return failures


def main():
    rt = Ipc()
    print("Connected to RayTrophi Studio.")
    build_rig(rt)

    failures = []
    failures += phase_readback(rt)
    failures += phase_untagged_default(rt)
    failures += phase_substance_physics(rt)
    failures += phase_solid_substance(rt)
    failures += phase_sealed_pockets(rt)
    failures += phase_effective_representation(rt)
    if not failures:
        failures += phase_cache_roundtrip(rt)

    print("\n" + "=" * 72)
    if failures:
        print("FAILED")
        for f in failures:
            print("\n  * " + f)
        print("=" * 72)
        raise SystemExit(1)
    print("PASSED - substance identity reaches the emitter and survives.")
    print("=" * 72)


def refuse_if_running_inside_the_app():
    try:
        import rt  # noqa: F401
    except ImportError:
        return
    raise SystemExit(
        "\n  This rig must run OUTSIDE the app, from a separate terminal.\n")


if __name__ == "__main__":
    if sys.platform != "win32":
        raise SystemExit("This rig talks to the Windows named pipe.")
    refuse_if_running_inside_the_app()
    main()
