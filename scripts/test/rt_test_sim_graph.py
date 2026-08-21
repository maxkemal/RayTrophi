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
attrs = rt.attr.list("domain", domain_name)
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

log("== scoped graphs: a graph belongs to a named entity ==")
# There is no global simulation graph any more, and no active-domain default:
# every call below says which entity it means.
SCOPE = "domain"
OWNER = domain_name

log("== a scope with no graph must REFUSE, not invent one ==")
# ★★★ The failure this guards is the worst shape available here: a typo'd owner
# that silently created a second, empty graph would accept every later edit and
# drive nothing, while every call returned success.
try:
    rt.sim_graph.nodes(SCOPE, "no_such_domain_xyz")
    check("unknown owner is refused", False, "the call succeeded")
except Exception as exc:
    log("   refused: %s" % exc)
    check("unknown owner is refused", True)
try:
    rt.sim_graph.create(SCOPE, "no_such_domain_xyz")
    check("creating a graph for a nonexistent entity is refused", False,
          "the call succeeded")
except Exception as exc:
    log("   refused: %s" % exc)
    check("creating a graph for a nonexistent entity is refused", True)

log("== build graph ==")
# ★ The owner node comes WITH the graph -- it is not authored here. That is the
# whole point of the scope: the canvas already knows whose it is.
dom = rt_testlog.fresh_graph(rt, SCOPE, OWNER)
check("the graph opens with an owner node", dom != 0)
insp = rt.sim_graph.add_node(SCOPE, OWNER, "sim.field_inspect")
rt.sim_graph.set_node(SCOPE, OWNER, insp, "channel", attrs[0] if attrs else "temperature")
rt.sim_graph.connect(SCOPE, OWNER, dom, insp)
nodes = rt.sim_graph.nodes(SCOPE, OWNER)
check("two nodes exist", len(nodes) == 2, "got %d" % len(nodes))

log("== the owner node cannot be retargeted ==")
# ★★ A graph filed under one entity that drove another would make every later
# reading agree with the wrong one, silently.
try:
    rt.sim_graph.set_node(SCOPE, OWNER, dom, "domain", "no_such_domain_xyz")
    check("retargeting the owner node is refused", False, "the call succeeded")
except Exception as exc:
    log("   refused: %s" % exc)
    check("retargeting the owner node is refused", True)
still = [n for n in rt.sim_graph.nodes(SCOPE, OWNER) if n["id"] == dom]
check("the refused retarget changed nothing",
      bool(still) and still[0].get("domain") == OWNER,
      "%s" % (still,))

log("== the graph is listed under its scope ==")
listed = [g for g in rt.sim_graph.list()
          if g["scope"] == SCOPE and g["owner"] == OWNER]
check("the graph appears in sim_graph.list", len(listed) == 1, "%s" % (rt.sim_graph.list(),))
if listed:
    check("its owner still exists", not listed[0]["owner_missing"],
          "owner_missing on a domain that is in the scene")

log("== evaluate (produces INTENT, applies nothing) ==")
result = rt.sim_graph.evaluate(SCOPE, OWNER)
check("graph evaluated", result["evaluated"])
log("   commands: %s" % (result["commands"],))
binds = [c for c in result["commands"] if c["kind"] == "bind_domain"]
check("domain bind emitted", len(binds) == 1 and binds[0]["target"] == domain_name)
check("no restart demanded by read-only nodes", not result["restart_requests"],
      "%s" % (result["restart_requests"],))

log("== inspect stats ==")
for n in rt.sim_graph.nodes(SCOPE, OWNER):
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

log("== rt.attr unifies discovery/measurement with the node-based reading ==")
# ★ Differential test: rt.attr.list/stats must not just exist, they must AGREE
# with the Field Inspect node reading taken above -- an instrument that only
# ever agrees with itself is not a measurement.
attr_list_domain = rt.attr.list("domain", domain_name)
check("rt.attr.list('domain', ...) matches the naming layer's own list",
      set(attr_list_domain) == set(attrs), "%s != %s" % (attr_list_domain, attrs))

node_reading = None
for n in rt.sim_graph.nodes(SCOPE, OWNER):
    if n["type"] == "sim.field_inspect":
        node_reading = n
        break
if node_reading and node_reading.get("stats_available"):
    s = rt.attr.stats("domain", domain_name, node_reading["channel"])
    check("rt.attr.stats is available for a channel the node measured",
          s["available"], "%s" % (s,))
    check("rt.attr.stats matches the Field Inspect node exactly",
          s["available"] and
          abs(s["min_value"] - node_reading["min_value"]) < 1e-6 and
          abs(s["max_value"] - node_reading["max_value"]) < 1e-6 and
          abs(s["mean_value"] - node_reading["mean_value"]) < 1e-6,
          "%s vs node %s" % (s, node_reading))
else:
    vacuous("rt.attr.stats matches the Field Inspect node exactly",
            "the node itself reported no measurable reading")

unknown = rt.attr.stats("domain", domain_name, "no_such_attribute_xyz")
check("an unknown attribute name reports unavailable, not zero",
      not unknown["available"], "%s" % (unknown,))

log("== evaluating twice must not disturb the solver ==")
def particle_count(name):
    for d in rt.fluid.list_domains():
        if d["name"] == name:
            return d["particle_count"]
    return 0


before = particle_count(domain_name)
rt.sim_graph.evaluate(SCOPE, OWNER)
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

dom = rt_testlog.fresh_graph(rt, SCOPE, OWNER)
setter = rt.sim_graph.add_node(SCOPE, OWNER, "sim.set_parameter")
rt.sim_graph.set_node(SCOPE, OWNER, setter, "key", KEY)
rt.sim_graph.set_node_value(SCOPE, OWNER, setter, "value", authored + 0.25)
rt.sim_graph.connect(SCOPE, OWNER, dom, setter)

applied = rt.sim_graph.apply(SCOPE, OWNER)
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

log("== Solver node: every field is OPT-IN ==")
# ★★★ The claim under test is the one that costs the most if it breaks: a field
# the user never ticked must NOT be written. A node that wrote all of its fields
# on every apply would flatten authored values with numbers nobody chose, and
# the override layer would then faithfully "restore" the graph's inventions.
authored_visc = read_param(domain_name, "kinematic_viscosity")
authored_slip = read_param(domain_name, "viscosity_wall_slip")
log("   authored viscosity=%.4f wall_slip=%.4f" % (authored_visc, authored_slip))

dom = rt_testlog.fresh_graph(rt, SCOPE, OWNER)
solver = rt.sim_graph.add_node(SCOPE, OWNER, "sim.solver")
rt.sim_graph.connect(SCOPE, OWNER, dom, solver)

fields = None
for n in rt.sim_graph.nodes(SCOPE, OWNER):
    if n["type"] == "sim.solver":
        fields = n.get("fields")
check("the solver node publishes its fields", bool(fields), "%s" % (fields,))
if fields:
    # Discoverability: a script must be able to LEARN which dials exist instead
    # of reading the solver source.
    keys = [f["key"] for f in fields]
    check("kinematic_viscosity is offered", "kinematic_viscosity" in keys, "%s" % (keys,))
    check("no field starts out in use",
          not any(f["in_use"] for f in fields), "%s" % (fields,))
    check("voxel_size declares that it needs a restart",
          any(f["key"] == "voxel_size" and f["requires_restart"] for f in fields),
          "%s" % (fields,))

# Touch exactly ONE field.
rt.sim_graph.set_node_value(SCOPE, OWNER, solver, "kinematic_viscosity",
                            authored_visc + 0.5)
applied = rt.sim_graph.apply(SCOPE, OWNER)
log("   apply -> %s" % (applied,))
check("only the touched field was written", applied["applied"] == 1, "%s" % (applied,))
check("the touched field actually changed",
      abs(read_param(domain_name, "kinematic_viscosity") - (authored_visc + 0.5)) < 1e-4,
      "%.4f" % read_param(domain_name, "kinematic_viscosity"))
# ★★★ THE ONE THAT MATTERS: an untouched field must be untouched. If this fails,
# every unticked dial is silently overwriting an authored value with a default
# and nothing in the app would report it.
check("an UNTOUCHED field was not written",
      abs(read_param(domain_name, "viscosity_wall_slip") - authored_slip) < 1e-6,
      "%.6f != %.6f" % (read_param(domain_name, "viscosity_wall_slip"), authored_slip))

log("   -- switching a field off stops it being written --")
rt.sim_graph.clear_overrides()
rt.sim_graph.set_node_value(SCOPE, OWNER, solver, "kinematic_viscosity.use", 0.0)
off = rt.sim_graph.apply(SCOPE, OWNER)
check("a field switched off writes nothing", off["applied"] == 0, "%s" % (off,))
check("the authored value is back",
      abs(read_param(domain_name, "kinematic_viscosity") - authored_visc) < 1e-6,
      "%.6f != %.6f" % (read_param(domain_name, "kinematic_viscosity"), authored_visc))

log("== an integer parameter must round-trip, not drift ==")
# ★★ viscosity_sweeps is an int carried through a float. A truncating write
# would read back one lower than the value just written, and clear_overrides
# would then restore the wrong authored number.
authored_sweeps = read_param(domain_name, "viscosity_sweeps")
rt.sim_graph.set_node_value(SCOPE, OWNER, solver, "viscosity_sweeps",
                            authored_sweeps + 3)
rt.sim_graph.apply(SCOPE, OWNER)
check("an integer parameter round-trips exactly",
      read_param(domain_name, "viscosity_sweeps") == authored_sweeps + 3,
      "%s -> %s" % (authored_sweeps + 3, read_param(domain_name, "viscosity_sweeps")))
rt.sim_graph.clear_overrides()
check("the authored integer is restored exactly",
      read_param(domain_name, "viscosity_sweeps") == authored_sweeps,
      "%s != %s" % (read_param(domain_name, "viscosity_sweeps"), authored_sweeps))

log("== Emitter node: names a flow source, does NOT rebind its domain ==")
try:
    sources = rt.flow_source.list()
except Exception as exc:                                       # noqa: BLE001
    sources = []
    log("   flow_source.list unavailable: %s" % exc)
if not sources:
    vacuous("emitter overrides are reversible",
            "the scene has no flow source; create one to exercise this")
else:
    emitter_name = sources[0]["name"] if isinstance(sources[0], dict) else sources[0]
    authored_src = rt.flow_source.get(emitter_name)
    log("   emitter %r radius=%.3f domain=%r" % (
        emitter_name, authored_src["radius"], authored_src.get("domain")))

    dom = rt_testlog.fresh_graph(rt, SCOPE, OWNER)
    emit_node = rt.sim_graph.add_node(SCOPE, OWNER, "sim.emitter")
    rt.sim_graph.set_node(SCOPE, OWNER, emit_node, "emitter", emitter_name)
    rt.sim_graph.set_node_value(SCOPE, OWNER, emit_node, "radius",
                                authored_src["radius"] * 2.0 + 0.1)

    # ★★★ UNWIRED FIRST. A node whose effect does not follow its wiring makes
    # the canvas unreadable: the reader sees an isolated box, the solver sees a
    # command. Reported by the user against the first cut, which had no Domain
    # pin at all -- so the rule is asserted before the wired case, not after.
    ev = rt.sim_graph.evaluate(SCOPE, OWNER)
    kinds = [c["kind"] for c in ev["commands"]]
    check("an UNCONNECTED emitter emits nothing", "set_emitter" not in kinds,
          "%s" % (kinds,))

    rt.sim_graph.connect(SCOPE, OWNER, dom, emit_node)
    ev = rt.sim_graph.evaluate(SCOPE, OWNER)
    kinds = [c["kind"] for c in ev["commands"]]
    check("emitter command emitted once wired", "set_emitter" in kinds, "%s" % (kinds,))

    applied = rt.sim_graph.apply(SCOPE, OWNER)
    log("   apply -> %s" % (applied,))
    now = rt.flow_source.get(emitter_name)
    check("emitter radius actually changed",
          abs(now["radius"] - (authored_src["radius"] * 2.0 + 0.1)) < 1e-4,
          "%.4f" % now["radius"])
    # ★★★ Read-modify-write, not build-fresh. SimulationFlowSourceInfo carries
    # ~25 authored fields; a node that rebuilt one to set a single value would
    # reset the rest, and the symptom would be "changing the rate also moved the
    # emitter and dropped its substance".
    check("the emitter's other authored fields survived",
          now.get("domain") == authored_src.get("domain") and
          abs(now["density"] - authored_src["density"]) < 1e-6 and
          now.get("source_mode") == authored_src.get("source_mode"),
          "domain=%r density=%s mode=%r" % (
              now.get("domain"), now["density"], now.get("source_mode")))

    rt.sim_graph.clear_overrides()
    restored = rt.flow_source.get(emitter_name)
    check("authored emitter radius restored exactly",
          abs(restored["radius"] - authored_src["radius"]) < 1e-6,
          "%.6f != %.6f" % (restored["radius"], authored_src["radius"]))

log("== restart-requiring parameter must be REFUSED, not applied ==")
dom = rt_testlog.fresh_graph(rt, SCOPE, OWNER)
setter = rt.sim_graph.add_node(SCOPE, OWNER, "sim.set_parameter")
rt.sim_graph.set_node(SCOPE, OWNER, setter, "key", "voxel_size")
rt.sim_graph.set_node_value(SCOPE, OWNER, setter, "value", 0.2)
rt.sim_graph.connect(SCOPE, OWNER, dom, setter)

voxel_before = read_param(domain_name, "voxel_size")
result = rt.sim_graph.evaluate(SCOPE, OWNER)
check("node reports the restart", len(result["restart_requests"]) == 1,
      "%s" % (result["restart_requests"],))
guarded = rt.sim_graph.apply(SCOPE, OWNER)   # allow_restart defaults to False
check("refused without permission", len(guarded["refused"]) == 1 and guarded["applied"] == 0,
      "%s" % (guarded,))
check("voxel size untouched", read_param(domain_name, "voxel_size") == voxel_before)
rt.sim_graph.clear(SCOPE, OWNER)

log("== World scope: WorldThermalNode overrides the ambient thermal state ==")
# Decision record: docs/dev/SIMULATION_NODE_OBJECT_MODEL.md section 7 item 1
# (closed) and section 8 step 4.
authored_thermal = rt.world.get_thermal()
log("   authored world thermal = %s" % (authored_thermal,))

wdom = rt_testlog.fresh_graph(rt, "world", "")
check("the world graph opens with NO owner node (there is exactly one world, "
      "so there is no identity to name)", wdom == 0, "got %s" % wdom)

wnode = rt.sim_graph.add_node("world", "", "sim.world_thermal")

wfields = None
for n in rt.sim_graph.nodes("world", ""):
    if n["type"] == "sim.world_thermal":
        wfields = n.get("fields")
check("the world thermal node publishes its fields", bool(wfields), "%s" % (wfields,))
if wfields:
    wkeys = [f["key"] for f in wfields]
    check("ambient_kelvin is offered", "ambient_kelvin" in wkeys, "%s" % (wkeys,))
    check("no field starts out in use",
          not any(f["in_use"] for f in wfields), "%s" % (wfields,))

# Touch exactly ONE field, same "opt-in" claim as the Solver test above.
new_ambient = authored_thermal["ambient_kelvin"] + 12.0
rt.sim_graph.set_node_value("world", "", wnode, "ambient_kelvin", new_ambient)

ev = rt.sim_graph.evaluate("world", "")
check("world graph emits a scoped set_parameter command",
      any(c["kind"] == "set_parameter" for c in ev["commands"]),
      "%s" % (ev["commands"],))

applied = rt.sim_graph.apply("world", "")
log("   apply -> %s" % (applied,))
check("exactly one world override applied", applied["applied"] == 1, "%s" % (applied,))

# ★★★ THE DIFFERENTIAL TEST that section 8 step 4 asked for: the graph path
# must land on the EXACT number the direct rt.world.set_thermal/get_thermal
# path would produce. An instrument that only ever agrees with itself is not a
# measurement -- the reference has to come from outside the node layer.
now_direct = rt.world.get_thermal()
check("graph-applied ambient_kelvin matches the direct API reading exactly",
      abs(now_direct["ambient_kelvin"] - new_ambient) < 1e-4,
      "%.4f != %.4f" % (now_direct["ambient_kelvin"], new_ambient))
# ★★★ THE ONE THAT MATTERS, same as the Solver test: an untouched field must
# stay untouched, or every unticked World Thermal dial silently overwrites the
# scene's ambient condition with a number nobody chose.
check("an UNTOUCHED world field was not written",
      abs(now_direct["kelvin_per_unit"] - authored_thermal["kelvin_per_unit"]) < 1e-6,
      "%.6f != %.6f" % (now_direct["kelvin_per_unit"], authored_thermal["kelvin_per_unit"]))

rt.sim_graph.clear_overrides()
restored = rt.world.get_thermal()
check("authored ambient_kelvin restored exactly",
      abs(restored["ambient_kelvin"] - authored_thermal["ambient_kelvin"]) < 1e-6,
      "%.6f != %.6f" % (restored["ambient_kelvin"], authored_thermal["ambient_kelvin"]))
check("no overrides held after clear", rt.sim_graph.override_count() == 0)
rt.sim_graph.clear("world", "")

log("== rt.attr on world scope: a single scalar, not a population ==")
world_attrs = rt.attr.list("world", "")
check("world exposes its four thermal fields",
      set(world_attrs) == {"ambient_kelvin", "kelvin_per_unit",
                           "convection_coefficient", "oxygen_availability"},
      "%s" % (world_attrs,))
ws = rt.attr.stats("world", "", "ambient_kelvin")
check("world attr stats matches rt.world.get_thermal() exactly",
      ws["available"] and abs(ws["mean_value"] - restored["ambient_kelvin"]) < 1e-6,
      "%s vs %.6f" % (ws, restored["ambient_kelvin"]))
check("a scalar world field reads min == max == mean",
      ws["min_value"] == ws["max_value"] == ws["mean_value"], "%s" % (ws,))

log("== sim_graph.domain_intersections: MEASURED, not declared (section 9.6) ==")
# ★ Neither ForceFieldInfo nor SimulationColliderInfo carries a `domain` field
# on purpose (section 1.3/9.1) -- so the only way to know a force reaches a
# domain is to overlap their world-space bounds. Two fields, one clearly
# inside the domain's box and one clearly far outside, prove the report tells
# them apart rather than always reporting true or always false.
dbox = rt.fluid.get(domain_name)
dmin, dmax = dbox["domain_min"], dbox["domain_max"]
center = tuple((dmin[i] + dmax[i]) * 0.5 for i in range(3))
far = tuple(dmin[i] - 1000.0 for i in range(3))
near_name = "IntersectTestNear_%d" % os.getpid()
far_name = "IntersectTestFar_%d" % os.getpid()
rt.forcefield.create("wind", near_name)
rt.forcefield.set_param(near_name, shape="sphere", position=center, falloff_radius=0.1)
rt.forcefield.create("wind", far_name)
rt.forcefield.set_param(far_name, shape="sphere", position=far, falloff_radius=0.1)
try:
    report = rt.sim_graph.domain_intersections(domain_name)
    by_name = {e["name"]: e for e in report}
    check("a force field placed at the domain's center is reported intersecting",
          near_name in by_name and by_name[near_name]["measurable"]
          and by_name[near_name]["intersects"], "%s" % (by_name.get(near_name),))
    check("a force field placed far outside is reported NOT intersecting",
          far_name in by_name and by_name[far_name]["measurable"]
          and not by_name[far_name]["intersects"], "%s" % (by_name.get(far_name),))
finally:
    rt.forcefield.remove(near_name)
    rt.forcefield.remove(far_name)

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
