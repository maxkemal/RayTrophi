"""RayTrophi visual demo — DESTRUCTIVE on purpose (leaves geometry in the scene).

Unlike rt_api_smoke_test.py (which undoes every mutation so the scene ends empty),
this script CREATES and KEEPS its results so you can actually SEE the modifier and
scatter working in the viewport. Run it from the Python Console on an empty scene.

Undo (Ctrl+Z) repeatedly to roll it back when you're done inspecting.
"""

import rt

print(f"[rt-demo] API version: {rt.version()}")

# ── 1. Catmull-Clark subdivision — KEEP the modifier so the cube stays smooth ──
sub_cube = rt.scene.add_primitive("cube", name="DemoSubdivCube", size=2.0)
rt.scene.set_transform(sub_cube, translation=(-3.0, 0.0, 0.0))

mod = rt.modifiers.add(sub_cube, type="catmull_clark", name="DemoSubdiv",
                       levels=2, render_levels=3)
stack = rt.modifiers.get_stack(sub_cube)
assert len(stack) == 1 and stack[0]["type"] == "catmull_clark"
print(f"[rt-demo] subdivided cube kept at x=-3 (levels={stack[0]['levels']}, "
      f"render_levels={stack[0]['render_levels']}) — should look like a smooth blob")

# ── 2. Scatter — plane target, small-cube source, KEEP the filled instances ────
target = rt.scene.add_primitive("plane", name="DemoScatterPlane", size=10.0)
rt.scene.set_transform(target, translation=(3.0, 0.0, 0.0))
source = rt.scene.add_primitive("cube", name="DemoScatterSource", size=0.4)

rt.scatter.create_group("DemoScatterGrp", target_node=target, target_type="mesh")
rt.scatter.add_source("DemoScatterGrp", source, weight=1.0, scale_min=0.5, scale_max=1.5)
rt.scatter.set_settings("DemoScatterGrp", target_count=80, min_distance=0.3)

spawned = rt.scatter.fill("DemoScatterGrp")
assert spawned > 0, "scatter fill spawned nothing"
print(f"[rt-demo] scattered {spawned} cubes onto the plane at x=+3 — should look like a field of cubes")

# Nudge a fresh render/rebuild so both results show immediately.
rt.reset_accumulation()
rt.request_render()

print("[rt-demo] DONE — left subdivided cube (x=-3) + scattered cubes (x=+3). Ctrl+Z to undo.")
