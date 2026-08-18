# Simulation node graph — Faz N6, cache / bake.
#
# WHAT THIS PINS DOWN
# -------------------
# ★★★ THE STALE CACHE. A bake that still exists but was built from a DIFFERENT
# authored config keeps serving frames, and those frames describe a scene that no
# longer exists. Nothing else tells it apart from a healthy cache — which is
# exactly how stale physics reaches a render and nobody reports a bug.
#
# ★★ And three states that all look like "nothing usable" must stay apart:
# nothing baked, a bake running, and a bake invalidated.
#
# ★ Evaluating the graph must NEVER bake. A bake walks the whole simulation;
# if evaluation triggered it, merely inspecting a graph would run the sim.
import os
import sys
import time

import rt

sys.path.insert(0, os.path.join("scripts", "test"))
import rt_testlog  # noqa: E402

rt_testlog.start("sim_cache")
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


domains = [d for d in rt.fluid.list_domains() if d.get("enabled", True)]
if not domains:
    log("no enabled domain -- run rt_setup_sim_graph_scene.py first")
    raise SystemExit(0)
domain = domains[0]["name"]
log("domain: %s" % domain)

log("== the three 'nothing usable' states are reported separately ==")
status = rt.sim_cache.status()
log("   %s" % (status,))
for key in ("valid", "baking", "ram_frames", "config_signature"):
    check("status reports %s" % key, key in status, "%s" % (sorted(status),))

log("== evaluating a Cache node must NOT bake ==")
# The cache belongs to the domain being baked, so the graph is owned by it.
SCOPE = "domain"
OWNER = domain
dom = rt_testlog.fresh_graph(rt, SCOPE, OWNER)
check("the graph opens with an owner node", dom != 0)
cache = rt.sim_graph.add_node(SCOPE, OWNER, "sim.cache")
rt.sim_graph.connect(SCOPE, OWNER, dom, cache)

before = rt.sim_cache.status()
started = time.time()
rt.sim_graph.evaluate(SCOPE, OWNER)
elapsed = time.time() - started
after = rt.sim_cache.status()
log("   evaluate took %.3f s" % elapsed)
# ★ A bake of any real range cannot finish in a few milliseconds, so a fast
# evaluation is evidence it did not start one; the frame counts confirm it.
check("evaluation did not start a bake", not after["baking"])
check("evaluation did not change the cache",
      after["ram_frames"] == before["ram_frames"] and
      after["valid"] == before["valid"],
      "%s -> %s" % (before, after))

log("== the node REPORTS the cache instead of guessing ==")
node = None
for n in rt.sim_graph.nodes(SCOPE, OWNER):
    if n["type"] == "sim.cache":
        node = n
        break
check("cache node exists", node is not None)
if node is not None:
    log("   node: %s" % ({k: v for k, v in node.items()
                          if k.startswith("cache_")},))
    check("node reports cache validity", "cache_valid" in node,
          "%s" % (sorted(node),))
    check("node reports staleness separately from validity",
          "cache_stale" in node, "%s" % (sorted(node),))
    check("node agrees with sim_cache.status",
          node.get("cache_valid") == after["valid"] and
          node.get("cache_ram_frames") == after["ram_frames"],
          "node=%s status=%s" % (node.get("cache_valid"), after["valid"]))

log("== ★★★ a config change must make an existing cache read STALE ==")
# ★ Bake a real range rather than scrubbing. Scrubbing populates the RAM cache
# only while the timeline is actually driving the simulation; an explicit bake is
# the operation whose result this phase is about, and it also exercises the bake
# surface itself. Small range on purpose — this test is about the SIGNATURE, not
# about throughput.
BAKE_DIR = os.path.join("scripts", "test", "_bake_cache_test")
try:
    rt.sim_cache.bake(BAKE_DIR, end_frame=4, start_frame=0, fps=24.0)
except Exception as exc:                                       # noqa: BLE001
    log("   bake refused: %s" % exc)

warm = rt.sim_cache.status()
log("   after bake: valid=%s ram_frames=%d signature=%s" % (
    warm["valid"], warm["ram_frames"], warm["config_signature"]))

if warm["ram_frames"] == 0 and not warm["valid"]:
    # ★ Nothing cached means the staleness claim cannot fail, and a check that
    # cannot fail has not passed. Say so rather than printing green.
    vacuous("a config change marks the cache stale",
            "the bake produced nothing, so there is no cache to invalidate — "
            "this scene may have no bakeable system")
else:
    # Change the authored config. voxel_size is the canonical example: it
    # reallocates the field the simulation lives in.
    live = [d for d in rt.fluid.list_domains() if d["name"] == domain][0]
    rt.fluid.set_param(domain, voxel_size=live["voxel_size"] * 1.5)
    rt.sim_graph.evaluate(SCOPE, OWNER)
    changed = None
    for n in rt.sim_graph.nodes(SCOPE, OWNER):
        if n["type"] == "sim.cache":
            changed = n
            break
    after_change = rt.sim_cache.status()
    log("   after the edit: ram_frames=%d signature=%s stale=%s" % (
        after_change["ram_frames"], after_change["config_signature"],
        changed.get("cache_stale") if changed else None))
    # Either the app dropped the cache outright (also correct — the edit
    # invalidated it) or the cache survives and MUST read stale. What must never
    # happen is a surviving cache that still reads healthy.
    dropped = after_change["ram_frames"] == 0 and not after_change["valid"]
    check("a surviving cache reads stale, or the cache was dropped",
          dropped or bool(changed and changed.get("cache_stale")),
          "dropped=%s stale=%s" % (dropped, changed.get("cache_stale") if changed else None))
    rt.fluid.set_param(domain, voxel_size=live["voxel_size"])

rt.sim_graph.clear(SCOPE, OWNER)

log("")
if FAIL:
    log("RESULT: %d FAILED: %s" % (len(FAIL), FAIL))
elif UNVERIFIED:
    log("RESULT: PASSED SO FAR, but %d claim(s) NOT VERIFIED: %s" %
        (len(UNVERIFIED), UNVERIFIED))
else:
    log("RESULT: ALL PASSED")
