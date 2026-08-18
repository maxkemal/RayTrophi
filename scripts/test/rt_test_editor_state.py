# Editor view state (rt.editor) — the test that makes the Nodes tab verifiable.
#
# ★★★ Why this file exists at all: panel DRAWING is unscriptable and always will
# be, but "which editor is open" is a value, and leaving it unreadable made the
# one thing this repo keeps getting wrong — the panel disagreeing with the core —
# structurally invisible to an agent. This checks the values, never the pixels.
import os
import sys

import rt

sys.path.insert(0, os.path.join("scripts", "test"))
import rt_testlog  # noqa: E402

rt_testlog.start("editor_state")
log = rt_testlog.log

FAIL = []


def check(label, ok, detail=""):
    log(("  OK   " + label) if ok else ("  FAIL " + label +
        ((" -- " + detail) if detail else "")))
    if not ok:
        FAIL.append(label)


def state():
    return rt.editor.get_state()


log("== the reader reports SOMETHING, and reports it completely ==")
s = state()
log("   %s" % (s,))
for key in ("bottom_editor", "node_editor_domain", "node_editor_open", "open_editors"):
    check("get_state carries '%s'" % key, key in s, "%s" % (s,))

log("== opening the Nodes editor on the simulation domain ==")
rt.editor.set_bottom_editor("simulation")
s = state()
check("bottom_editor is simulation", s["bottom_editor"] == "simulation", s["bottom_editor"])
check("the Nodes window is open", s["node_editor_open"] is True)
check("the selector agrees with what is open",
      s["node_editor_domain"] == "simulation", s["node_editor_domain"])

log("== switching domain MOVES the editor, it does not stack ==")
rt.editor.set_node_domain("geometry")
s = state()
log("   %s" % (s,))
# ★★★ The central claim. A selector naming one graph while another is on screen
# is exactly the shape of "the panel lies", and it would be worse arriving from a
# script because nobody is looking at the screen when it happens.
check("the selector followed the switch", s["node_editor_domain"] == "geometry",
      s["node_editor_domain"])
check("the geometry editor is what is open", s["bottom_editor"] == "geometry",
      s["bottom_editor"])
check("the simulation window closed", s["node_editor_open"] is False)

log("== exclusivity: exactly ONE bottom editor at a time ==")
for name in ("console", "assets", "simulation", "material", "dope_sheet",
             "graph_editor", "terrain", "anim_graph"):
    rt.editor.set_bottom_editor(name)
    s = state()
    # ★★ `open_editors` exists for precisely this assertion. `bottom_editor`
    # names ONE, so a reader that only returned it could never report two panels
    # open at once -- it would answer "healthy" exactly when it was not.
    check("only one editor open after set_bottom_editor(%s)" % name,
          len(s["open_editors"]) == 1, "%s" % (s["open_editors"],))
    check("set_bottom_editor(%s) opened that one" % name,
          s["bottom_editor"] == name, s["bottom_editor"])

log("== 'none' closes everything ==")
rt.editor.set_bottom_editor("none")
s = state()
check("nothing is open", s["open_editors"] == [], "%s" % (s["open_editors"],))
check("bottom_editor reads none", s["bottom_editor"] == "none", s["bottom_editor"])

log("== an unknown name is REFUSED, not silently ignored ==")
rt.editor.set_bottom_editor("simulation")
before = state()
refused = False
try:
    rt.editor.set_bottom_editor("no_such_editor")
except Exception as exc:                       # noqa: BLE001 - the point is that it raises
    refused = True
    log("   refused: %s" % exc)
check("unknown editor name raises", refused)
after = state()
# ★ A rejected call must leave the state alone. A "failure" that already closed
# everything before noticing the name was bad is a silent side effect, and the
# caller would have no way to tell it apart from success.
check("a refused call changed nothing",
      after["bottom_editor"] == before["bottom_editor"] and
      after["open_editors"] == before["open_editors"],
      "%s -> %s" % (before, after))

refused = False
try:
    rt.editor.set_node_domain("no_such_domain")
except Exception as exc:                       # noqa: BLE001
    refused = True
    log("   refused: %s" % exc)
check("unknown node domain raises", refused)

log("== which SCOPED graph the canvas is on is a VALUE ==")
# ★★★ The canvas shows one scoped graph at a time, so "which one" is a reading
# an agent must be able to take. Without it, every later assertion about the
# panel would be about an unknown graph -- the panel-lies shape again, one level
# up.
state = rt.editor.get_state()
for key in ("sim_graph_scope", "sim_graph_owner"):
    check("get_state carries %r" % key, key in state, "%s" % (sorted(state),))

domains = rt.fluid.list_domains()
if not domains:
    log("   no fluid domain -- scope selection checks need one")
else:
    domain_name = domains[0]["name"]
    rt.editor.set_sim_graph_scope("domain", domain_name)
    state = rt.editor.get_state()
    check("the scope selection is reported back",
          state["sim_graph_scope"] == "domain" and
          state["sim_graph_owner"] == domain_name,
          "%s / %s" % (state["sim_graph_scope"], state["sim_graph_owner"]))

    # ★★ Selecting a scope with NO graph must be allowed: the panel draws an
    # explicit empty state, and that is how graph creation is reached at all.
    # Refusing here would make the empty case unreachable from UI and script.
    if any(g["scope"] == "domain" and g["owner"] == domain_name
           for g in rt.sim_graph.list()):
        rt.sim_graph.delete("domain", domain_name)
    rt.editor.set_sim_graph_scope("domain", domain_name)
    check("selecting a scope with no graph is allowed",
          rt.editor.get_state()["sim_graph_owner"] == domain_name)

    log("== world scope carries no owner ==")
    rt.editor.set_sim_graph_scope("world", "ignored")
    state = rt.editor.get_state()
    # ★ There is one world, so an owner name here would be a value that means
    # nothing and could disagree with itself between readings.
    check("world scope drops the owner name",
          state["sim_graph_scope"] == "world" and state["sim_graph_owner"] == "",
          "%s / %r" % (state["sim_graph_scope"], state["sim_graph_owner"]))

    log("== an unknown scope is REFUSED, and changes nothing ==")
    before = rt.editor.get_state()
    refused = False
    try:
        rt.editor.set_sim_graph_scope("no_such_scope", "x")
    except Exception as exc:                       # noqa: BLE001
        refused = True
        log("   refused: %s" % exc)
    check("unknown scope raises", refused)
    after = rt.editor.get_state()
    # ★★★ Same trap as set_bottom_editor: a refused call that had already moved
    # the selection would leave the canvas somewhere the caller was told it had
    # not gone. The user sees the error; nobody sees the side effect.
    check("a refused scope change moved nothing",
          after["sim_graph_scope"] == before["sim_graph_scope"] and
          after["sim_graph_owner"] == before["sim_graph_owner"],
          "%s -> %s" % (before, after))

log("== the panel and the script edit the SAME graph ==")
# ★★★ This is what makes the Nodes panel legitimate under CLAUDE.md rule 1. The
# panel draws rtapi::simulationGraph(scope, owner); if it kept a copy, a
# script-built graph would not be the one on screen and neither side could check
# the other.
rt.editor.set_bottom_editor("simulation")
if domains:
    domain_name = domains[0]["name"]
    rt.editor.set_sim_graph_scope("domain", domain_name)
    node = rt_testlog.fresh_graph(rt, "domain", domain_name)
    nodes = rt.sim_graph.nodes("domain", domain_name)
    # One node, and it is the OWNER node the graph was created with -- not an
    # authored one. A scoped graph is never ownerless.
    check("the graph the panel draws opens with its owner node",
          len(nodes) == 1 and nodes[0]["id"] == node and nodes[0]["owner_node"],
          "%s" % (nodes,))
    check("the owner node names this graph's owner",
          nodes and nodes[0].get("domain") == domain_name, "%s" % (nodes,))
    check("every simulation node type is offered by the registry",
          all(n["type"].startswith("sim.") for n in nodes), "%s" % (nodes,))
    rt.sim_graph.clear("domain", domain_name)

log("")
if FAIL:
    log("RESULT: %d FAILED: %s" % (len(FAIL), FAIL))
else:
    log("RESULT: ALL PASSED")
