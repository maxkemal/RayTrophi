# Simulation node graph â€” Faz N7, render binding.
#
# WHAT THIS PINS DOWN
# -------------------
# â˜…â˜… A LOOK is authored state too, so binding it from a graph must be reversible
# like everything else. And "empty" is a MEANINGFUL value here â€” empty surface
# material means "built-in dielectric" â€” so a graph must be able to write it
# back, not just overwrite it with something non-empty.
#
# â˜… Smoke and fire are ONE node with a preset, not two nodes. They write the
# same authored struct; two nodes would let a graph hold both and have the
# second silently overwrite the first.
import os
import sys

import rt

sys.path.insert(0, os.path.join("scripts", "test"))
import rt_testlog  # noqa: E402

rt_testlog.start("sim_render")
log = rt_testlog.log

FAIL = []
UNVERIFIED = []


def check(label, ok, detail=""):
    log(("  OK   " + label) if ok else ("  FAIL " + label +
        ((" -- " + detail) if detail else "")))
    if not ok:
        FAIL.append(label)


def vacuous(label, reason):
    log("  ????  " + label + " -- NOT VERIFIED: " + reason)
    UNVERIFIED.append(label)


domains = rt.fluid.list_domains()
gases = [d["name"] for d in domains if d.get("type") == "gas"]
fluids = [d["name"] for d in domains if d.get("type") == "fluid"]
log("gas domains: %s   fluid domains: %s" % (gases, fluids))

if not gases:
    vacuous("volume look is bound and reversible",
            "no gas domain in the scene; GasShaderSettings has nothing to bind")
else:
    gas = gases[0]
    try:
        authored = rt.gas.get_shader(gas)
    except Exception as exc:                                   # noqa: BLE001
        authored = None
        vacuous("volume look is bound and reversible",
                "cannot read the authored gas shader settings (%s)" % exc)
    if authored is not None:
        log("authored gas shader: preset=%r scattering=%.3f" % (
            authored["preset"], authored["scattering_coefficient"]))

        # A domain's LOOK belongs to that domain, so the graph is owned by it.
        dom = rt_testlog.fresh_graph(rt, "domain", gas)
        mat = rt.sim_graph.add_node("domain", gas, "sim.material_volume")
        rt.sim_graph.connect("domain", gas, dom, mat)
        target_preset = "fire" if authored["preset"] != "fire" else "smoke"
        rt.sim_graph.set_node("domain", gas, mat, "preset", target_preset)

        result = rt.sim_graph.evaluate("domain", gas)
        kinds = [c["kind"] for c in result["commands"]]
        check("render commands emitted", "set_render" in kinds, "%s" % (kinds,))
        # â˜…â˜…â˜… Preset ALONE by default. A node that also pushed its placeholder
        # sliders would install the recipe and immediately overwrite every
        # number that makes it that recipe â€” the preset would be a label with no
        # visible effect. Measured 2026-08-17 through rt.gas.set_shader, which
        # reported success and changed nothing.
        check("preset alone is emitted by default",
              sum(1 for c in result["commands"] if c["kind"] == "set_render") == 1,
              "%s" % ([c["key"] for c in result["commands"]],))

        applied = rt.sim_graph.apply("domain", gas)
        log("   apply -> %s" % (applied,))
        check("apply reported no failures", not applied["failed"],
              "%s" % (applied["failed"],))
        now = rt.gas.get_shader(gas)
        check("preset actually changed", now["preset"] == target_preset,
              "%r" % (now["preset"],))
        # â˜…â˜… And the recipe's OWN numbers survived the switch.
        check("the preset brought its own values",
              now["scattering_coefficient"] != authored["scattering_coefficient"] or
              now["blackbody_intensity"] != authored["blackbody_intensity"] or
              now["temperature_max"] != authored["temperature_max"],
              "nothing changed but the label: %s" % (now,))

        log("   -- now with explicit value overrides --")
        rt.sim_graph.set_node_value("domain", gas, mat, "override_values", 1.0)
        rt.sim_graph.set_node_value("domain", gas, mat, "scattering", 0.77)
        rt.sim_graph.apply("domain", gas)
        overridden = rt.gas.get_shader(gas)
        check("explicit override reaches the shader",
              abs(overridden["scattering_coefficient"] - 0.77) < 1e-4,
              "%.4f" % overridden["scattering_coefficient"])

        rt.sim_graph.clear_overrides()
        restored = rt.gas.get_shader(gas)
        check("authored preset restored exactly",
              restored["preset"] == authored["preset"],
              "%r != %r" % (restored["preset"], authored["preset"]))
        check("authored look restored exactly",
              abs(restored["scattering_coefficient"] -
                  authored["scattering_coefficient"]) < 1e-6,
              "%.6f != %.6f" % (restored["scattering_coefficient"],
                                authored["scattering_coefficient"]))
        check("no overrides held after clear", rt.sim_graph.override_count() == 0)
        rt.sim_graph.clear("domain", gas)

if not fluids:
    vacuous("liquid material is bound and reversible", "no fluid domain")
else:
    fluid = fluids[0]
    authored_surface = [d for d in domains if d["name"] == fluid][0]["surface_material"]
    log("authored surface material: %r" % authored_surface)

    try:
        materials = rt.material.list()
    except Exception:                                          # noqa: BLE001
        materials = []
    names = [m["name"] if isinstance(m, dict) else m for m in materials]
    target = next((n for n in names if n != authored_surface), None)
    if not target:
        vacuous("liquid material is bound and reversible",
                "the scene has no second material to bind")
    else:
        dom = rt_testlog.fresh_graph(rt, "domain", fluid)
        mat = rt.sim_graph.add_node("domain", fluid, "sim.material_liquid")
        rt.sim_graph.connect("domain", fluid, dom, mat)
        rt.sim_graph.set_node("domain", fluid, mat, "surface_material", target)

        applied = rt.sim_graph.apply("domain", fluid)
        log("   apply -> %s" % (applied,))
        now = [d for d in rt.fluid.list_domains() if d["name"] == fluid][0]
        check("surface material actually changed",
              now["surface_material"] == target,
              "%r" % (now["surface_material"],))

        rt.sim_graph.clear_overrides()
        restored = [d for d in rt.fluid.list_domains()
                    if d["name"] == fluid][0]["surface_material"]
        # â˜…â˜… Empty is a real value ("built-in dielectric"), so restoring it must
        # work â€” a layer that only restores non-empty names would leave the
        # liquid permanently wearing whatever the graph put on it.
        check("authored surface material restored exactly",
              restored == authored_surface,
              "%r != %r" % (restored, authored_surface))

# ★ Two graphs were built, so two get cleared. There is no "clear everything"
# call on purpose: a caller that cannot name what it is clearing is the same
# ambiguity the scope argument removes.
# * Read from the domain lists, not from `gas`/`fluid`: those names only exist
# if their branch ran, and a NameError here would erase a completed test run.
for _owner in ([gases[0]] if gases else []) + ([fluids[0]] if fluids else []):
    rt.sim_graph.clear("domain", _owner)

log("")
log("NOTE: there is deliberately no Foam Material node (FoamParams has no")
log("      scripting surface, same gap as the N4 foam coupling) and no Char")
log("      node (char colour is DERIVED from the substance).")

log("")
if FAIL:
    log("RESULT: %d FAILED: %s" % (len(FAIL), FAIL))
elif UNVERIFIED:
    log("RESULT: PASSED SO FAR, but %d claim(s) NOT VERIFIED: %s" %
        (len(UNVERIFIED), UNVERIFIED))
else:
    log("RESULT: ALL PASSED")
