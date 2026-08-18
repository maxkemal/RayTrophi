# Simulation node graph smoke test (Faz N0/N1/N2/N3).
#
# The point of N2 is that the graph can be BUILT and READ from a script before
# anything is allowed to write. N3 adds writes, but only as a reversible
# override layer.
import os
import sys

import rt

# scripts/ is on sys.path, scripts/test/ is not -- and `test` is a stdlib
# package name, so importing it as a package would be a coin flip.
sys.path.insert(0, os.path.join("scripts", "test"))
import rt_testlog  # noqa: E402

rt_testlog.start("sim_graph")
log = rt_testlog.log

FAIL = []
UNVERIFIED = []


def check(label, ok, detail=""):
    # Detail only on failure: printing it next to "OK" produced lines like
    # "OK  authored value restored exactly -- 0.000000 != 0.000000", which reads
    # as a contradiction of the very thing that passed.
    log(("  OK   " + label) if ok else ("  FAIL " + label +
        ((" -- " + detail) if detail else "")))
    if not ok:
        FAIL.append(label)


def vacuous(label, reason):
    """An assertion that cannot fail in the current scene.

    ★ A check whose preconditions are absent has NOT passed -- it did not run.
    Counting it as green is how a batch reports ALL PASSED while its most
    important claim was never tested. Say so instead."""
    log("  ????  " + label + " -- NOT VERIFIED: " + reason)
    UNVERIFIED.append(label)


domains = rt.fluid.list_domains()
if not domains:
    log("no fluid domain in the scene -- run rt_setup_sim_graph_scene.py first")
    raise SystemExit(0)
# ★ Prefer a domain that actually HAS particles. Taking domains[0] made this
# whole file vacuous the moment the scene grew a second, empty domain: every
# central claim reported NOT VERIFIED while a perfectly good seeded domain sat
# next to it. The rig must pick the domain it can measure, not the first one.
live = [d for d in domains if d.get("particle_count", 0) > 0]
domain_name = (live[0] if live else domains[0])["name"]
log("domain: %s (%d particles)" % (
    domain_name,
    next(d.get("particle_count", 0) for d in domains if d["name"] == domain_name)))

log("== attribute discovery (the naming layer) ==")
attrs = rt.sim_graph.attributes(domain_name)
log("   %s" % (attrs,))
if not attrs:
    # Legitimate for a domain that has never been stepped — but then the naming
    # layer was never exercised, and saying nothing would let the batch look
    # fully green while N1's real payload went untested.
    vacuous("attribute naming layer resolves real arrays",
            "domain has no live particle state; nothing to name yet")
else:
    check("known attributes are exposed", "temperature" in attrs,
          "got %s" % (attrs,))

log("== build graph ==")
rt.sim_graph.clear()
dom = rt.sim_graph.add_node("sim.domain_ref")
insp = rt.sim_graph.add_node("sim.field_inspect")
rt.sim_graph.set_node(dom, "domain", domain_name)
rt.sim_graph.set_node(insp, "channel", attrs[0] if attrs else "temperature")
rt.sim_graph.connect(dom, insp)
nodes = rt.sim_graph.nodes()
check("two nodes exist", len(nodes) == 2, "got %d" % len(nodes))

log("== evaluate (produces INTENT, applies nothing) ==")
result = rt.sim_graph.evaluate()
check("graph evaluated", result["evaluated"])
log("   commands: %s" % (result["commands"],))
binds = [c for c in result["commands"] if c["kind"] == "bind_domain"]
check("domain bind emitted", len(binds) == 1 and binds[0]["target"] == domain_name)
check("no restart demanded by read-only nodes", not result["restart_requests"],
      "%s" % (result["restart_requests"],))

log("== inspect stats ==")
for n in rt.sim_graph.nodes():
    if n["type"] != "sim.field_inspect":
        continue
    if not n.get("stats_available"):
        # ★ Not a failure — stats_available false means the value could not be
        # measured, and treating that as a zero is the confusion the flag exists
        # to prevent. But it is also not a pass: the inspector never ran.
        vacuous("field inspector reports real statistics",
                "no live particles, or the attribute name is unknown")
        break
    log("   channel=%s n=%d array=%d min=%.4f max=%.4f mean=%.4f" % (
        n["channel"], n["particle_count"], n.get("array_size", -1),
        n["min_value"], n["max_value"], n["mean_value"]))
    check("min <= mean <= max", n["min_value"] <= n["mean_value"] <= n["max_value"])
    check("particle count is positive", n["particle_count"] > 0)
    # ★★★ An attribute array longer than the particle count means a removal path
    # shortened some arrays and not this one -- which also means this array's
    # entry i no longer describes particle i. Measured 2026-08-17: the reseed
    # trim compacted the primary arrays and left every granular_* array behind.
    check("attribute array is in sync with the particle count",
          n.get("array_in_sync", True),
          "array=%d particles=%d" % (n.get("array_size", -1), n["particle_count"]))

log("== evaluating twice must not disturb the solver ==")
def particle_count(name):
    for d in rt.fluid.list_domains():
        if d["name"] == name:
            return d["particle_count"]
    return 0


before = particle_count(domain_name)
rt.sim_graph.evaluate()
after = particle_count(domain_name)
# ★★★ The whole N0 contract in one assertion: evaluating a simulation graph must
# never reset or advance anything. GraphBase::evaluate() would have cleared the
# cache and marked everything dirty; evaluateSimulation deliberately does not.
#
# ★★ But 0 -> 0 proves NOTHING. Resetting an empty simulation looks exactly like
# not resetting it, so on an unstepped domain this check cannot fail and must
# not be reported as green. Step the sim before trusting it.
if before == 0:
    vacuous("particle count unchanged by evaluation",
            "domain has no particles; a reset of an empty sim is invisible. "
            "Seed and step the domain, then re-run.")
else:
    check("particle count unchanged by evaluation", before == after,
          "%d -> %d" % (before, after))

log("== N3: parameter override, and it must be REVERSIBLE ==")


def read_param(name, key):
    for d in rt.fluid.list_domains():
        if d["name"] == name:
            return d[key]
    return None


KEY = "pore_amount"
authored = read_param(domain_name, KEY)
log("   authored %s = %.4f" % (KEY, authored))

rt.sim_graph.clear()
dom = rt.sim_graph.add_node("sim.domain_ref")
setter = rt.sim_graph.add_node("sim.set_parameter")
rt.sim_graph.set_node(dom, "domain", domain_name)
rt.sim_graph.set_node(setter, "key", KEY)
rt.sim_graph.set_node_value(setter, "value", authored + 0.25)
rt.sim_graph.connect(dom, setter)

applied = rt.sim_graph.apply()
log("   apply -> %s" % (applied,))
check("one override applied", applied["applied"] == 1, "%s" % (applied,))
check("authored value captured", applied["overrides_held"] == 1)
now = read_param(domain_name, KEY)
check("parameter actually changed", abs(now - (authored + 0.25)) < 1e-4,
      "%.4f" % now)

rt.sim_graph.clear_overrides()
restored = read_param(domain_name, KEY)
# ★★★ The whole N3 contract: an override must be reversible. If this fails the
# graph has written into authored data and the original is gone -- which is
# exactly what plan B.5 forbids, and it cannot be undone after the fact.
check("authored value restored exactly", abs(restored - authored) < 1e-6,
      "%.6f != %.6f" % (restored, authored))
check("no overrides held after clear", rt.sim_graph.override_count() == 0)

log("== restart-requiring parameter must be REFUSED, not applied ==")
rt.sim_graph.clear()
dom = rt.sim_graph.add_node("sim.domain_ref")
setter = rt.sim_graph.add_node("sim.set_parameter")
rt.sim_graph.set_node(dom, "domain", domain_name)
rt.sim_graph.set_node(setter, "key", "voxel_size")
rt.sim_graph.set_node_value(setter, "value", 0.2)
rt.sim_graph.connect(dom, setter)

voxel_before = read_param(domain_name, "voxel_size")
result = rt.sim_graph.evaluate()
check("node reports the restart", len(result["restart_requests"]) == 1,
      "%s" % (result["restart_requests"],))
guarded = rt.sim_graph.apply()          # allow_restart defaults to False
check("refused without permission", len(guarded["refused"]) == 1 and guarded["applied"] == 0,
      "%s" % (guarded,))
check("voxel size untouched", read_param(domain_name, "voxel_size") == voxel_before)
rt.sim_graph.clear()

log("")
if FAIL:
    log("RESULT: %d FAILED: %s" % (len(FAIL), FAIL))
elif UNVERIFIED:
    # ★★★ Never print ALL PASSED while a claim went untested. A green batch whose
    # central assertion never ran is worse than a red one: nobody re-runs it.
    log("RESULT: PASSED SO FAR, but %d claim(s) NOT VERIFIED: %s" %
        (len(UNVERIFIED), UNVERIFIED))
    log("        Seed and step the domain, then run this again -- the N0")
    log("        contract cannot be tested on an empty simulation.")
else:
    log("RESULT: ALL PASSED")
