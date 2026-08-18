# Does the surface inspector READ, or does it return zero?
#
# ★ Same discriminator as rt_probe_sim_attributes.py, one level down: a channel
# reading flat proves nothing on its own (an unheated crate really is at 0), but
# a zero-filled buffer or a wrong stride CANNOT return different, semantically
# correct constants per channel.
import os
import sys

import rt

sys.path.insert(0, os.path.join("scripts", "test"))
import rt_testlog  # noqa: E402

rt_testlog.start("surface_attributes")
log = rt_testlog.log

colliders = rt.collider.list()
if not colliders:
    log("no collider -- run rt_setup_sim_substance_scene.py first")
    raise SystemExit(0)
obj = colliders[0]["name"]
attrs = rt.sim_graph.surface_attributes(obj)
log("object: %s" % obj)
log("attributes: %s" % (attrs,))
if not attrs:
    log("no Material State Field on this object -- nothing to measure")
    raise SystemExit(0)

rt.sim_graph.clear()
node = rt.sim_graph.add_node("sim.object_ref")
rt.sim_graph.set_node(node, "object", obj)
insp = rt.sim_graph.add_node("sim.surface_inspect")
rt.sim_graph.connect(node, insp)

readings = {}
for channel in attrs:
    rt.sim_graph.set_node(insp, "channel", channel)
    rt.sim_graph.evaluate()
    stats = None
    for n in rt.sim_graph.nodes():
        if n["type"] == "sim.surface_inspect":
            stats = n
            break
    if not stats or not stats.get("stats_available"):
        log("  %-20s NOT MEASURABLE" % channel)
        continue
    spread = stats["max_value"] - stats["min_value"]
    readings[channel] = (stats["min_value"], stats["max_value"])
    fresh = stats.get("host_fresh", True)
    log("  %-20s n=%-7d min=%-12.6f max=%-12.6f mean=%-12.6f%s%s" % (
        channel, stats["particle_count"], stats["min_value"],
        stats["max_value"], stats["mean_value"],
        "  <- varies" if spread > 0.0 else "",
        "" if fresh else "  <- STALE HOST MIRROR"))

rt.sim_graph.set_node(insp, "channel", "no_such_channel_xyz")
rt.sim_graph.evaluate()
bogus = None
for n in rt.sim_graph.nodes():
    if n["type"] == "sim.surface_inspect":
        bogus = n
        break
log("")
log("unknown channel reports unmeasured: %s" %
    ("OK" if bogus and not bogus.get("stats_available") else "FAIL"))

varying = sum(1 for lo, hi in readings.values() if hi > lo)
distinct = len(set(readings.values()))

# ★★★ Before judging the numbers, say whether they describe the CURRENT state.
# fuel_remaining == -1 everywhere is the "not seeded yet" sentinel, not a fuel
# level, and a reader who takes it for a measurement is being lied to by a
# default. This is reported FIRST because it changes what every line above means.
fuel = readings.get("fuel_remaining")
if fuel and fuel[0] == fuel[1] == -1.0:
    log("")
    log("★ fuel_remaining reads -1 everywhere: that is the 'not seeded yet' "
        "sentinel. The field exists but its fuel has never been initialised — "
        "step a gas domain that actually heats this object.")

if varying:
    log("VERDICT: inspector reads live per-element data (%d channel(s) vary)." % varying)
elif distinct > 1:
    log("VERDICT: no channel varies, but channels disagree with each other "
        "(%d distinct readings) -- the arrays are distinct and being read. An "
        "unheated object is expected to read flat." % distinct)
else:
    log("VERDICT: every channel reads the SAME flat value -- consistent with a "
        "zero-filled buffer or a wrong stride. Do not trust the inspector until "
        "a channel is made to vary.")
rt.sim_graph.clear()
