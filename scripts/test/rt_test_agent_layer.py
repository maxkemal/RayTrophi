# Agent discovery layer smoke test.
#
# What this rig is really guarding: the discovery layer is the surface an agent
# BELIEVES. Its failure mode is not a crash, it is a confident wrong answer -
# a method that exists with no schema, a recipe naming a call that was renamed,
# a probe reporting 0.0 when nothing was measured. Every check below is aimed
# at one of those.
import os
import sys

import rt

sys.path.insert(0, os.path.join("scripts", "test"))
import rt_testlog  # noqa: E402

rt_testlog.start("agent_layer")
log = rt_testlog.log

FAIL = []
UNVERIFIED = []


def check(label, ok, detail=""):
    log(("  OK   " + label) if ok else
        ("  FAIL " + label + ((" -- " + detail) if detail else "")))
    if not ok:
        FAIL.append(label)


def vacuous(label, reason):
    # A check whose preconditions are absent has NOT passed.
    log("  ????  " + label + " -- NOT VERIFIED: " + reason)
    UNVERIFIED.append(label)


# ---------------------------------------------------------------------------
log("1. discover")
# ---------------------------------------------------------------------------
info = rt.agent.discover()
registered = info.get("registered_methods", 0)
documented = info.get("documented_methods", 0)
coverage = info.get("documented_coverage", 0.0)
log("   %s %s | %d methods | %d documented (%.0f%%)"
    % (info.get("app"), info.get("version"), registered, documented, 100.0 * coverage))

check("discover reports methods", registered > 100, "got %d" % registered)
check("discover lists domains", len(info.get("domains", [])) > 10)
check("no invented totals", "total_methods_expected" not in info,
      "a hand-kept dispatch total is a lie waiting to happen")
check("documented_coverage is a real fraction", 0.0 <= coverage <= 1.0,
      "got %r" % coverage)
if coverage < 1.0:
    log("   note: %d methods carry a schema but no prose" % (registered - documented))

# ---------------------------------------------------------------------------
log("2. list_methods / describe agree")
# ---------------------------------------------------------------------------
listing = rt.agent.list_methods()
methods = [entry["method"] for entry in listing.get("methods", [])]
check("list_methods returns the whole registry", len(methods) == registered,
      "%d vs %d" % (len(methods), registered))

fluid_only = rt.agent.list_methods("fluid")
fluid_names = [entry["method"] for entry in fluid_only.get("methods", [])]
check("domain filter filters", bool(fluid_names) and
      all(name.startswith("fluid.") for name in fluid_names))

# ★ The descriptor half that must never drift: parameters come from dispatch.
# If someone renames a parameter in RtIpc*.cpp without regenerating, this is
# where an agent would start sending a key nobody reads -- and the call would
# SUCCEED, silently using the default.
spot = {
    "fluid.create_domain": ["name", "type", "domain_min", "domain_max", "voxel_size"],
    "scene.add_primitive": ["type", "name", "size"],
    "render.start": ["output_path", "spp"],
    "gas.set_settings": ["domain", "fire_enabled", "ignition_temperature"],
    "physics.fracture_object": ["object", "site_count"],
    "flow_source.create": ["domain", "position", "temperature"],
}
for method, expected in spot.items():
    described = rt.agent.describe(method)
    params = described.get("params", {})
    missing = [key for key in expected if key not in params]
    check("describe(%s) knows its parameters" % method, not missing,
          "missing " + ", ".join(missing))

described = rt.agent.describe("scene.add_primitive")
check("required flags survive", described["params"]["type"]["required"] is True)
check("optional flags survive", described["params"]["size"]["required"] is False)
check("defaults are reported", "default" in described["params"]["size"])
check("enum hints reach the agent", "enum" in described["params"]["type"],
      "cube|sphere|plane|cylinder|torus should be discoverable")

# ---------------------------------------------------------------------------
log("3. access and capability tell the truth")
# ---------------------------------------------------------------------------
# ★ These were all reported as "write" in the first cut, including
# scene.list_objects. An agent told that reading writes will refuse to look.
for method in ("scene.list_objects", "fluid.list_domains", "camera.get",
               "timeline.get_frame", "agent.discover"):
    described = rt.agent.describe(method)
    check("%s is read access" % method, described.get("access") == "read",
          "got %r" % described.get("access"))
check("render.start is not scene-write",
      rt.agent.describe("render.start").get("capability") == "Render")
check("agent.chat_send is not read-only",
      rt.agent.describe("agent.chat_send").get("capability") == "AgentChat",
      "posting into the user's panel must not ride on a Read token")

# ---------------------------------------------------------------------------
log("4. search finds the recipe, and the recipe is real")
# ---------------------------------------------------------------------------
found = rt.agent.search("make a wooden object burn")
workflows = [w["workflow"] for w in found.get("relevant_workflows", [])]
check("burning wood finds the combustion recipe", "combustion_setup" in workflows,
      "got " + ", ".join(workflows) or "nothing")

found_pour = rt.agent.search("pour water into a container")
check("pouring finds the liquid recipe",
      "liquid_pour" in [w["workflow"] for w in found_pour.get("relevant_workflows", [])])

check("search returns method hits too", bool(found.get("relevant_methods")))

# ★ Recipe rot: a recipe naming a method that no longer exists is worse than no
# recipe. The agent follows it, the call fails, and the failure looks like the
# engine's fault.
known = set(methods)
for query in ("burn", "pour", "shatter", "terrain", "material", "render", "scatter"):
    for workflow in rt.agent.search(query).get("relevant_workflows", []):
        unknown = [m for m in workflow.get("key_methods", []) if m not in known]
        check("recipe %s only names real methods" % workflow["workflow"], not unknown,
              "unknown: " + ", ".join(unknown))

# ---------------------------------------------------------------------------
log("5. examples are generated from the schema, not written by hand")
# ---------------------------------------------------------------------------
example = rt.agent.examples(workflow="combustion_setup")
calls = example.get("calls", [])
check("combustion example has calls", bool(calls))
check("every example call is a real method",
      all(call["method"] in known for call in calls))

single = rt.agent.examples(method="fluid.create_domain")
check("method example carries a call", "example_call" in single)
check("method example knows its workflows", bool(single.get("used_by_workflows")))

# ---------------------------------------------------------------------------
log("6. unknown names are answered with directions")
# ---------------------------------------------------------------------------
try:
    rt.agent.describe("fluid.create_domian")   # typo on purpose
    check("typo is rejected", False, "describe accepted a name that does not exist")
except Exception as exc:  # noqa: BLE001
    check("typo is rejected", "no method named" in str(exc), str(exc))

# ---------------------------------------------------------------------------
log("7. state summary: absence of measurement is reported as absence")
# ---------------------------------------------------------------------------
rt.viewport.capture(False)
summary = rt.agent.state_summary(include_probe=True)
probe = summary.get("viewport", {}).get("probe")
# ★★★ The single most important assertion in this file. With capture off the
# probe must NOT come back as a set of zeros: mean_luminance 0.0 reads as "the
# scene is black", and an agent would go looking for a lighting bug that does
# not exist.
check("probe with capture off says unavailable", isinstance(probe, str),
      "got %r" % (probe,))

rt.viewport.capture(True)
rt.viewport.render_frames(4)
summary = rt.agent.state_summary(include_probe=True)
probe = summary.get("viewport", {}).get("probe")
if isinstance(probe, dict):
    check("probe with capture on returns numbers", "mean_luminance" in probe)
    check("probe counts pixels", probe.get("pixels", 0) > 0)
else:
    vacuous("probe with capture on returns numbers",
            "no frame was captured even after render_frames -- probe said: %r" % (probe,))

check("summary counts objects",
      summary.get("scene", {}).get("object_count") == len(rt.scene.objects()))
check("summary reports the timeline frame", "frame" in summary.get("timeline", {}))

# Domains: particle_count must be a number ONLY when the solver state is live.
for domain in summary.get("simulation_domains", []):
    value = domain.get("particle_count")
    check("domain %s reports particle_count honestly" % domain.get("name"),
          isinstance(value, int) or isinstance(value, str),
          "got %r" % (value,))

# ---------------------------------------------------------------------------
log("8. chat surface")
# ---------------------------------------------------------------------------
rt.agent.chat_send("agent layer test running", sender="Test", type="activity")
polled = rt.agent.chat_poll()
check("chat_poll answers with a prompt list", "prompts" in polled)

try:
    rt.agent.chat_send("bad type", sender="Test", type="not_a_type")
    check("chat_send validates its type", False, "an unknown type was accepted")
except Exception as exc:  # noqa: BLE001
    check("chat_send validates its type", "reply|activity" in str(exc), str(exc))

# ---------------------------------------------------------------------------
log("")
if FAIL:
    log("FAILED (%d): %s" % (len(FAIL), "; ".join(FAIL)))
elif UNVERIFIED:
    log("PASSED, but %d check(s) NOT VERIFIED: %s"
        % (len(UNVERIFIED), "; ".join(UNVERIFIED)))
else:
    log("ALL PASSED")
