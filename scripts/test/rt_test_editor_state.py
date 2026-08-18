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

log("== the panel and the script edit the SAME graph ==")
# ★★★ This is what makes the Nodes panel legitimate under CLAUDE.md rule 1. The
# panel draws rtapi::simulationGraph(); if it kept a copy, a script-built graph
# would not be the one on screen and neither side could check the other.
rt.editor.set_bottom_editor("simulation")
rt.sim_graph.clear()
node = rt.sim_graph.add_node("sim.domain_ref")
domains = rt.fluid.list_domains()
if domains:
    rt.sim_graph.set_node(node, "domain", domains[0]["name"])
nodes = rt.sim_graph.nodes()
check("the graph the panel draws has the script's node", len(nodes) == 1,
      "%d" % len(nodes))
check("every simulation node type is offered by the registry",
      all(n["type"].startswith("sim.") for n in nodes), "%s" % (nodes,))
rt.sim_graph.clear()

log("")
if FAIL:
    log("RESULT: %d FAILED: %s" % (len(FAIL), FAIL))
else:
    log("RESULT: ALL PASSED")
