# Does a graph whose owner is GONE say so?
#
# ★★★ A stranded graph keeps drawing and keeps accepting edits while driving
# nothing — the shape that let fracture UI state outlive a scene change. Domain
# removal drops the graph at its one call site; object deletion reaches the
# scene through several paths, so instead of claiming they are all hooked the
# condition is MEASURED. This probe is that measurement.
import os
import sys

import rt

sys.path.insert(0, os.path.join("scripts", "test"))
import rt_testlog  # noqa: E402

rt_testlog.start("orphan_graph")
log = rt_testlog.log
FAIL = []


def check(label, ok, detail=""):
    log(("  OK   " + label) if ok else ("  FAIL " + label +
        ((" -- " + detail) if detail else "")))
    if not ok:
        FAIL.append(label)


def graph_row(scope, owner):
    for g in rt.sim_graph.list():
        if g["scope"] == scope and g["owner"] == owner:
            return g
    return None


log("== a domain graph goes WITH its domain ==")
NAME = "OrphanProbeDomain"
if any(d["name"] == NAME for d in rt.fluid.list_domains()):
    rt.fluid.remove_domain(NAME)
rt.fluid.create_domain(NAME, domain_min=(-1, 0, -1), domain_max=(1, 2, 1),
                       voxel_size=0.2)
rt.sim_graph.create("domain", NAME)
check("the domain graph exists", graph_row("domain", NAME) is not None)
rt.fluid.remove_domain(NAME)
check("removing the domain dropped its graph", graph_row("domain", NAME) is None,
      "%s" % (graph_row("domain", NAME),))

log("== an object graph is REPORTED when its owner disappears ==")
objects = rt.scene.objects()
names = [o["name"] if isinstance(o, dict) else o for o in objects]
if not names:
    log("   no object in the scene -- cannot exercise the object half")
else:
    obj = names[0]
    log("   object: %s" % obj)
    rt.sim_graph.create("object", obj)
    row = graph_row("object", obj)
    check("the object graph exists", row is not None)
    check("a live owner is not reported missing",
          row is not None and not row["owner_missing"], "%s" % (row,))

    rt.scene.delete(obj)
    row = graph_row("object", obj)
    # ★★★ The graph is EXPECTED to still be here — object deletion is not
    # hooked. What must not happen is it reading healthy.
    if row is None:
        log("   the graph was dropped outright (also correct)")
        check("a stranded object graph cannot read healthy", True)
    else:
        check("a stranded object graph reports owner_missing",
              row["owner_missing"], "%s" % (row,))

log("")
log("RESULT: %s" % ("%d FAILED: %s" % (len(FAIL), FAIL) if FAIL else "ALL PASSED"))
