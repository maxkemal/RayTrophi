# Test output that an agent can READ from outside the app.
#
# ★ Python print() inside RayTrophi goes to the in-app console
# (rtpython::consoleOutputSnapshot), which is not exposed over IPC. So a test
# driven through `script.run_file` runs blind: the agent sees `true` and nothing
# else. Mirroring every line into a file makes the result readable without a
# human reading the console panel.
#
# Paths are relative to the process working directory (x64/Release), which is
# also where the app reads its scripts from.
import os

_STATE = {"path": None, "lines": []}


def start(name):
    """Begin a run; truncates the result file so a stale pass cannot be re-read."""
    directory = os.path.join("scripts", "test")
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        pass
    _STATE["path"] = os.path.join(directory, "_%s_result.txt" % name)
    _STATE["lines"] = []
    _flush()


def fresh_graph(rt, scope, owner):
    """Create-or-empty the scoped graph and return its OWNER NODE id.

    ★★★ Every test used to open with clear() + add_node("sim.domain_ref") +
    set_node(..., "domain", name). That triple is now one call: a scoped graph
    is created already naming its owner, and clear() re-seeds that node rather
    than leaving an ownerless canvas.

    ★ Returns 0 for the World scope, which has no owner node -- there is
    exactly one world, so there is no identity to name (unlike a Domain or
    Object graph, which can hold more than one). A caller that needs the id
    must check, not assume.
    """
    rt.sim_graph.create(scope, owner)
    rt.sim_graph.clear(scope, owner)
    for node in rt.sim_graph.nodes(scope, owner):
        if node.get("owner_node"):
            return node["id"]
    return 0


def log(message=""):
    text = str(message)
    print(text)
    _STATE["lines"].append(text)
    _flush()


def _flush():
    if not _STATE["path"]:
        return
    # Written after every line, not at exit: a test that dies mid-way must leave
    # behind everything it had already proven, otherwise the crash erases the
    # evidence of what worked.
    try:
        with open(_STATE["path"], "w", encoding="utf-8") as handle:
            handle.write("\n".join(_STATE["lines"]) + "\n")
    except OSError:
        pass
