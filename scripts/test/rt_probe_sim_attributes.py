# Does the field inspector READ, or does it just return zero?
#
# ★ rt_test_sim_graph.py saw temperature min=max=mean=0 across 3200 particles.
# That is a legitimate reading for an unheated water domain -- and it is also
# exactly what a zero-filled buffer, a wrong stride, or an unresolved attribute
# name would look like. "A default is not a measurement": the only way to tell
# them apart is to inspect an attribute that MUST vary.
import os
import sys

import rt

sys.path.insert(0, os.path.join("scripts", "test"))
import rt_testlog  # noqa: E402

rt_testlog.start("sim_attributes")
log = rt_testlog.log

domains = rt.fluid.list_domains()
if not domains:
    log("no fluid domain -- run rt_setup_sim_graph_scene.py first")
    raise SystemExit(0)
name = domains[0]["name"]
log("domain: %s (%d particles)" % (name, domains[0]["particle_count"]))

attrs = rt.sim_graph.attributes(name)
log("attributes: %s" % (attrs,))

rt.sim_graph.clear()
dom = rt.sim_graph.add_node("sim.domain_ref")
insp = rt.sim_graph.add_node("sim.field_inspect")
rt.sim_graph.set_node(dom, "domain", name)
rt.sim_graph.connect(dom, insp)

varying = 0
unavailable = []
out_of_sync = []
readings = {}
for channel in attrs:
    rt.sim_graph.set_node(insp, "channel", channel)
    rt.sim_graph.evaluate()
    stats = None
    for n in rt.sim_graph.nodes():
        if n["type"] == "sim.field_inspect":
            stats = n
            break
    if not stats or not stats.get("stats_available"):
        # Not zero -- unmeasured. Kept separate on purpose.
        unavailable.append(channel)
        log("  %-24s NOT MEASURABLE" % channel)
        continue
    spread = stats["max_value"] - stats["min_value"]
    if spread > 0.0:
        varying += 1
    readings[channel] = (stats["min_value"], stats["max_value"])
    # array_in_sync false: the backing array outlived particles that were
    # removed. Not a reading error -- a fact about the solver's arrays.
    synced = stats.get("array_in_sync", True)
    if not synced:
        out_of_sync.append("%s(%d)" % (channel, stats.get("array_size", 0)))
    log("  %-24s n=%-6d min=%-16.6f max=%-16.6f mean=%-16.6f%s%s" % (
        channel, stats["particle_count"], stats["min_value"], stats["max_value"],
        stats["mean_value"],
        "  <- varies" if spread > 0.0 else "",
        "  <- ARRAY OUT OF SYNC (%d)" % stats.get("array_size", 0) if not synced else ""))

# An unknown name must report NOT MEASURABLE, never a confident zero.
rt.sim_graph.set_node(insp, "channel", "no_such_attribute_xyz")
rt.sim_graph.evaluate()
bogus = None
for n in rt.sim_graph.nodes():
    if n["type"] == "sim.field_inspect":
        bogus = n
        break
bogus_ok = bool(bogus) and not bogus.get("stats_available")
log("")
log("unknown attribute reports unmeasured: %s" % ("OK" if bogus_ok else "FAIL"))
log("channels with real spread: %d / %d (unmeasurable: %s)" %
    (varying, len(attrs), unavailable or "none"))
if out_of_sync:
    log("arrays longer than the live particle count: %s" % ", ".join(out_of_sync))

# ★ Two separate discriminators, because "flat" alone proves nothing. A domain
# at rest legitimately has flat channels -- water that was never heated really
# does read temperature 0 everywhere. What a zero-filled buffer or a wrong
# pointer CANNOT do is return different, semantically correct constants per
# channel (mass_fraction 1.0 next to granular_damage 0.0).
distinct = len({v for v in readings.values()})
if varying > 0:
    log("VERDICT: inspector reads live per-particle data (a channel varies).")
elif distinct > 1:
    log("VERDICT: no channel varies, but channels disagree with each other "
        "(%d distinct readings) -- the arrays are distinct and being read. "
        "A domain at rest is expected to read flat." % distinct)
else:
    log("VERDICT: every channel reads the SAME flat value -- consistent with a "
        "zero-filled buffer or an unresolved pointer. Do not trust N2 until a "
        "channel is made to vary.")
rt.sim_graph.clear()
