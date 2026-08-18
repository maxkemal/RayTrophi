# Simulation node graph — Faz N4, coupling nodes.
#
# WHAT THIS PINS DOWN
# -------------------
# ★★★ The graph DECLARES couplings; stepGridDomains decides the order they run
# in. This test exists to make sure the two are reported SEPARATELY and compared,
# because a graph that displayed a chosen order while the solver ran a different
# one would look like control and be a lie -- and "producer != consumer" is
# already a recurring failure class in this repository.
#
# ★★ It also pins the honest gaps. A coupling with no script-writable switch
# (foam) must REPORT that, not quietly succeed; and `traced` must separate
# "stepped, nothing coupled" from "the solver was never asked".
import os
import sys

import rt

sys.path.insert(0, os.path.join("scripts", "test"))
import rt_testlog  # noqa: E402

rt_testlog.start("sim_couplings")
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
fluids = [d["name"] for d in domains if d.get("type") == "fluid"]
gases = [d["name"] for d in domains if d.get("type") == "gas"]
if not fluids:
    log("no fluid domain -- run rt_setup_sim_graph_scene.py first")
    raise SystemExit(0)
fluid_name = fluids[0]
gas_name = gases[0] if gases else ""
log("fluid: %s   gas: %s" % (fluid_name, gas_name or "(none)"))

log("== build a coupling chain ==")
# The chain lives in the FLUID domain's graph: it is the source of the coupling.
# ★ The far end is still an ordinary Domain node inside that graph -- a coupling
# names two ends, and only one of them owns the canvas.
SCOPE = "domain"
OWNER = fluid_name
src = rt_testlog.fresh_graph(rt, SCOPE, OWNER)
check("the graph opens with an owner node", src != 0)
dst = rt.sim_graph.add_node(SCOPE, OWNER, "sim.domain_ref")
rt.sim_graph.set_node(SCOPE, OWNER, dst, "domain", gas_name or fluid_name)

# Chained through the Source pass-through, so the command order follows the
# chain rather than node creation order. That order is what gets compared.
burn = rt.sim_graph.add_node(SCOPE, OWNER, "sim.couple_fluid_to_gas")
foam = rt.sim_graph.add_node(SCOPE, OWNER, "sim.couple_foam")
rt.sim_graph.connect(SCOPE, OWNER, src, burn)          # Source -> Source
rt.sim_graph.connect(SCOPE, OWNER, dst, burn, 0, 1)    # Domain -> Target
rt.sim_graph.connect(SCOPE, OWNER, burn, foam)

report = rt.sim_graph.couplings()
declared = [c["coupling"] for c in report["declared"]]
log("   declared: %s" % (declared,))
log("   actual:   %s" % ([c["coupling"] for c in report["actual"]],))
log("   traced=%s order_matches=%s" % (report["traced"], report["order_matches"]))
check("both couplings declared", declared == ["fluid_to_gas", "foam_from_fluid"],
      "%s" % (declared,))
check("declared coupling names its two ends",
      report["declared"][0]["source_domain"] == fluid_name and
      report["declared"][0]["target_domain"] == (gas_name or fluid_name),
      "%s" % (report["declared"][0],))
# ★★★ The coupling report spans EVERY scope on purpose: a coupling joins two
# domains, so it belongs to neither graph alone. Scoping this report to one
# graph would hide exactly the cross-domain declarations it exists to show, so
# each declaration has to say which graph it came from.
check("each declaration names the graph it came from",
      all(c.get("scope") and c.get("owner") for c in report["declared"]),
      "%s" % (report["declared"],))

log("== the solver's OWN report, not a copy of the graph ==")
# ★★★ `traced` false means the solver was never asked. An empty `actual` alone
# cannot tell that apart from "stepped and nothing coupled", and reading it as
# zero is exactly the confusion the flag exists to prevent.
if not report["traced"]:
    vacuous("solver reports the couplings it ran",
            "no particle system exists; the trace was never taken")
else:
    rt.fluid.step(0.0166)
    stepped = rt.sim_graph.couplings()
    actual = [c["coupling"] for c in stepped["actual"]]
    log("   after one step, actual: %s" % (actual,))
    if not actual:
        # Legitimate: nothing in this scene couples. Still NOT a pass -- the
        # claim that the trace reflects reality went untested.
        vacuous("solver trace reflects a running coupling",
                "the scene has no coupling that actually runs (needs a burning "
                "liquid overlapping a gas domain, or foam enabled)")
    else:
        check("every traced coupling names its producer and consumer",
              all(c["producer"] and c["consumer"] for c in stepped["actual"]),
              "%s" % (stepped["actual"],))

log("== a declaration the solver never ran must be REPORTED ==")
# ★★ This is the assertion that makes the whole phase honest. The graph above
# declares foam on a domain with no foam running; if that silently disappeared,
# a user would read the graph as a description of the step and be wrong.
report = rt.sim_graph.couplings()
log("   declared_not_running: %s" % (report["declared_not_running"],))
log("   running_not_declared: %s" % (report["running_not_declared"],))
check("a declared-but-not-running coupling is named",
      any("foam_from_fluid" in s for s in report["declared_not_running"]) or
      any(c["coupling"] == "foam_from_fluid" for c in report["actual"]),
      "neither declared_not_running nor actual mentions foam")

log("== foam has no script-writable switch, and must SAY so ==")
applied = rt.sim_graph.apply(SCOPE, OWNER)
log("   apply -> %s" % (applied,))
# ★ A coupling that cannot be switched must fail loudly. Reporting success here
# would leave a user believing a graph edit took effect when nothing changed.
check("foam coupling reports its missing switch",
      any("foam" in f for f in applied["failed"]),
      "%s" % (applied["failed"],))

log("== a coupling switch is REVERSIBLE, like every other override ==")
# ★★★ Drop the previous section's override FIRST. Without this the "authored"
# reading below is taken while an override is still in force, and the test then
# demands that clear_overrides restore a value that was never authored. The
# first version of this test did exactly that and read the override layer's
# correct behaviour as a failure -- the same confusion the layer exists to
# prevent, committed by the test that guards it.
rt.sim_graph.clear_overrides()
try:
    before = rt.fluid.get_combustion(fluid_name)
except Exception as exc:                                   # noqa: BLE001
    before = None
    vacuous("coupling switch is reversible",
            "this domain has no combustible-fluid settings to switch (%s)" % exc)

if before is not None:
    log("   authored combustion enabled=%s auto_ignite=%s" %
        (before["enabled"], before["auto_ignite"]))

    src = rt_testlog.fresh_graph(rt, SCOPE, OWNER)
    burn = rt.sim_graph.add_node(SCOPE, OWNER, "sim.couple_fluid_to_gas")
    rt.sim_graph.connect(SCOPE, OWNER, src, burn)
    rt.sim_graph.set_node_value(SCOPE, OWNER, burn, "active",
                                0.0 if before["enabled"] else 1.0)

    applied = rt.sim_graph.apply(SCOPE, OWNER)
    log("   apply -> %s" % (applied,))
    during = rt.fluid.get_combustion(fluid_name)
    check("coupling switch actually changed",
          during["enabled"] != before["enabled"],
          "%s -> %s" % (before["enabled"], during["enabled"]))

    rt.sim_graph.clear_overrides()
    after = rt.fluid.get_combustion(fluid_name)
    # ★★★ Same contract as N3: the authored value must come back exactly. If it
    # does not, the graph wrote into authored data and the original is gone.
    check("authored coupling state restored",
          after["enabled"] == before["enabled"] and
          after["auto_ignite"] == before["auto_ignite"],
          "%s / %s" % (after["enabled"], after["auto_ignite"]))
    check("no overrides held after clear", rt.sim_graph.override_count() == 0)

rt.sim_graph.clear(SCOPE, OWNER)

log("")
if FAIL:
    log("RESULT: %d FAILED: %s" % (len(FAIL), FAIL))
elif UNVERIFIED:
    log("RESULT: PASSED SO FAR, but %d claim(s) NOT VERIFIED: %s" %
        (len(UNVERIFIED), UNVERIFIED))
    log("        A coupling that never runs cannot prove the trace is real.")
else:
    log("RESULT: ALL PASSED")
