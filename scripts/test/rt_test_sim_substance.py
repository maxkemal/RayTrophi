# Simulation node graph — Faz N5, substance and chemistry nodes.
#
# WHAT THIS PINS DOWN
# -------------------
# ★★★ An object's MATERIAL is authored state, so writing it from a graph has to
# be reversible exactly like every other override (plan B.5). Substance is the
# harshest case: it is a NAME, and a name has no numeric fallback -- if the
# capture is lost, there is nothing to reconstruct "it used to be steel" from.
#
# ★★ It also pins the two nodes that are deliberately MISSING. Moisture has no
# authored knob (it is written by fluid contact), and thermal transfer has no
# scripting surface at all yet. Both are exposed or refused honestly rather than
# given an invented authoring surface.
import os
import sys

import rt

sys.path.insert(0, os.path.join("scripts", "test"))
import rt_testlog  # noqa: E402

rt_testlog.start("sim_substance")
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


colliders = rt.collider.list()
if not colliders:
    log("no simulation collider in the scene -- "
        "run rt_setup_sim_substance_scene.py first")
    raise SystemExit(0)
obj = colliders[0]["name"]
log("object: %s" % obj)

authored = rt.collider.get(obj)
log("authored substance=%r ignite_on_contact=%s melt_spread=%.3f" % (
    authored["msf_substance"], authored["gas_ignite_on_contact"],
    authored["msf_melt_spread"]))

log("== build an object chain ==")
# ★★★ OBJECT scope, not domain: what an object is made of is a property of the
# object, so it lives on the object's own graph. This is the second scope, and
# the reason scopes are separate graph kinds rather than a flag.
SCOPE = "object"
OWNER = obj
node_obj = rt_testlog.fresh_graph(rt, SCOPE, OWNER)
check("the object graph opens with an owner node", node_obj != 0)
node_sub = rt.sim_graph.add_node(SCOPE, OWNER, "sim.substance")
node_pyro = rt.sim_graph.add_node(SCOPE, OWNER, "sim.pyrolysis")
rt.sim_graph.connect(SCOPE, OWNER, node_obj, node_sub)
rt.sim_graph.connect(SCOPE, OWNER, node_sub, node_pyro)

TARGET = "Iron" if authored["msf_substance"] != "Iron" else "Steel"
rt.sim_graph.set_node(SCOPE, OWNER, node_sub, "substance", TARGET)
rt.sim_graph.set_node_value(SCOPE, OWNER, node_sub, "burn_rate_scale", 2.5)
rt.sim_graph.set_node_value(SCOPE, OWNER, node_pyro, "active",
                            0.0 if authored["gas_ignite_on_contact"] else 1.0)

log("== object and domain scopes are SEPARATE graphs ==")
# ★★ Same owner name in a different scope must be a different canvas. If the two
# storages ever collapsed into one, an object graph and a domain graph that
# happened to share a name would overwrite each other silently.
scoped = rt.sim_graph.list()
mine = [g for g in scoped if g["scope"] == SCOPE and g["owner"] == OWNER]
check("the object graph is listed under the object scope", len(mine) == 1,
      "%s" % (scoped,))

result = rt.sim_graph.evaluate(SCOPE, OWNER)
kinds = [c["kind"] for c in result["commands"]]
log("   commands: %s" % (kinds,))
check("object bind emitted", "bind_object" in kinds, "%s" % (kinds,))
check("surface writes emitted", kinds.count("set_surface") >= 6, "%s" % (kinds,))
# ★ The commands carry the OBJECT name, not a handle -- same identity rule as
# every other node. A pointer here would not survive a rebuild or an IPC hop.
targets = {c["target"] for c in result["commands"]}
check("commands name the object", targets == {obj}, "%s" % (targets,))

log("== apply, then read the AUTHORED state back ==")
applied = rt.sim_graph.apply(SCOPE, OWNER)
log("   apply -> %s" % (applied,))
check("apply reported no failures", not applied["failed"], "%s" % (applied["failed"],))

now = rt.collider.get(obj)
check("substance actually changed", now["msf_substance"] == TARGET,
      "%r" % (now["msf_substance"],))
check("numeric override actually changed",
      abs(now["msf_burn_rate_scale"] - 2.5) < 1e-4,
      "%.4f" % now["msf_burn_rate_scale"])
check("pyrolysis switch actually changed",
      now["gas_ignite_on_contact"] != authored["gas_ignite_on_contact"],
      "%s -> %s" % (authored["gas_ignite_on_contact"], now["gas_ignite_on_contact"]))
# ★★ The shape of the object must be untouched. updateSimulationCollider takes a
# COMPLETE descriptor, so a writer that did not read first would silently reset
# the collider's geometry and rates to their defaults while the substance write
# "succeeded".
check("unrelated collider fields survived the write",
      now["source_mode"] == authored["source_mode"] and
      abs(now["friction"] - authored["friction"]) < 1e-6 and
      abs(now["restitution"] - authored["restitution"]) < 1e-6,
      "mode=%s friction=%s restitution=%s" % (
          now["source_mode"], now["friction"], now["restitution"]))

log("== and it must be REVERSIBLE, name included ==")
rt.sim_graph.clear_overrides()
restored = rt.collider.get(obj)
# ★★★ A substance is a NAME. If the capture is lost there is no numeric
# fallback to rebuild it from -- "it used to be Steel" is unrecoverable.
check("authored substance restored exactly",
      restored["msf_substance"] == authored["msf_substance"],
      "%r != %r" % (restored["msf_substance"], authored["msf_substance"]))
check("authored scale restored exactly",
      abs(restored["msf_burn_rate_scale"] - authored["msf_burn_rate_scale"]) < 1e-6,
      "%.6f != %.6f" % (restored["msf_burn_rate_scale"],
                        authored["msf_burn_rate_scale"]))
check("authored pyrolysis switch restored",
      restored["gas_ignite_on_contact"] == authored["gas_ignite_on_contact"])
check("no overrides held after clear", rt.sim_graph.override_count() == 0)

log("== an unknown substance must be REFUSED, not silently accepted ==")
node_obj = rt_testlog.fresh_graph(rt, SCOPE, OWNER)
node_sub = rt.sim_graph.add_node(SCOPE, OWNER, "sim.substance")
rt.sim_graph.connect(SCOPE, OWNER, node_obj, node_sub)
rt.sim_graph.set_node(SCOPE, OWNER, node_sub, "substance", "Unobtainium")
bad = rt.sim_graph.apply(SCOPE, OWNER)
log("   apply -> %s" % (bad,))
# A typo must not quietly turn a steel beam into oak.
check("unknown substance reported as a failure", bool(bad["failed"]),
      "%s" % (bad,))
check("object kept its substance",
      rt.collider.get(obj)["msf_substance"] == authored["msf_substance"],
      "%r" % rt.collider.get(obj)["msf_substance"])
rt.sim_graph.clear_overrides()
rt.sim_graph.clear(SCOPE, OWNER)

log("== an object has TWO names, and both must work ==")
# ★★★ The authored material lives on the COLLIDER; the measured MSF field is
# keyed by the collider's SOURCE OBJECT. Resolving only one of them would make a
# node silently work under one name and silently do nothing under the other,
# with no error either way. Measured 2026-08-17: naming the collider read back an
# empty attribute list while a 99072-element field sat right there.
source_object = authored.get("source_object") or ""
if not source_object:
    vacuous("both names resolve to the same surface",
            "this collider is analytic (no source object), so there is only one "
            "name to resolve")
else:
    by_collider = rt.attr.list("object", obj)
    by_object = rt.attr.list("object", source_object)
    log("   collider %r -> %d attrs, object %r -> %d attrs" % (
        obj, len(by_collider), source_object, len(by_object)))
    check("both names resolve to the same surface", by_collider == by_object,
          "%s != %s" % (by_collider, by_object))

log("== the naming layer reaches per-TEXEL MSF data ==")
attrs = rt.attr.list("object", obj)
log("   surface attributes: %s" % (attrs,))
if not attrs:
    # ★ Legitimate before the object has ever been stepped with a live MSF --
    # and NOT a pass. Saying "no attributes" and "never measured" with the same
    # empty list is the confusion this project keeps paying for.
    vacuous("surface attribute naming layer resolves real arrays",
            "the object carries no Material State Field yet; step a gas domain "
            "with this object inside it first")
else:
    check("moisture is exposed as a READING", "moisture" in attrs,
          "%s" % (attrs,))
    check("melt is exposed as a READING", "melt" in attrs, "%s" % (attrs,))

    node_obj = rt_testlog.fresh_graph(rt, SCOPE, OWNER)
    insp = rt.sim_graph.add_node(SCOPE, OWNER, "sim.surface_inspect")
    rt.sim_graph.set_node(SCOPE, OWNER, insp, "channel", "temperature")
    rt.sim_graph.connect(SCOPE, OWNER, node_obj, insp)
    rt.sim_graph.evaluate(SCOPE, OWNER)
    for n in rt.sim_graph.nodes(SCOPE, OWNER):
        if n["type"] != "sim.surface_inspect":
            continue
        if not n.get("stats_available"):
            vacuous("surface inspector reports real statistics",
                    "no MSF elements, or the channel name is unknown")
            break
        log("   channel=%s n=%d host_fresh=%s min=%.2f max=%.2f mean=%.2f" % (
            n["channel"], n["particle_count"], n.get("host_fresh"),
            n["min_value"], n["max_value"], n["mean_value"]))
        check("min <= mean <= max",
              n["min_value"] <= n["mean_value"] <= n["max_value"])
        check("element count is positive", n["particle_count"] > 0)
        # ★★★ The reading must SAY whether it describes the current device state.
        # The host mirror is only refreshed by a readback; before one, the array
        # still holds its initialisation values (fuel_remaining = -1 means "not
        # seeded", not "negative fuel"). A number without this flag is a default
        # wearing a measurement's clothes.
        check("the reading reports its freshness", "host_fresh" in n,
              "%s" % (sorted(n.keys()),))
    rt.sim_graph.clear(SCOPE, OWNER)

log("")
log("NOTE: there is deliberately no Moisture setter node and no Thermal")
log("      Transfer node. Moisture has no authored knob (fluid contact writes")
log("      it), and WorldThermalState has no scripting surface at all yet.")

log("")
if FAIL:
    log("RESULT: %d FAILED: %s" % (len(FAIL), FAIL))
elif UNVERIFIED:
    log("RESULT: PASSED SO FAR, but %d claim(s) NOT VERIFIED: %s" %
        (len(UNVERIFIED), UNVERIFIED))
else:
    log("RESULT: ALL PASSED")
