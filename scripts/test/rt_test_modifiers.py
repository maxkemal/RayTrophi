"""rt.modifiers — comprehensive test / demo (mesh modifier stack API).

MODE — set KEEP below:
  KEEP = False  Non-destructive validation. Exercises EVERY rt.modifiers function
                with assertions, then deletes the objects it created so the scene
                ends exactly as it started. Use for CI / smoke runs.
  KEEP = True   Same validation, but at the end leaves ONE visible subdivided cube
                in the scene so you can SEE the modifier working. Delete it (or
                Ctrl+Z) when you are done inspecting.

Run from the Python Console on ANY scene: it only ever touches objects it creates
itself (all names are prefixed), so it never disturbs your existing geometry.

Coverage: get_stack, add (catmull_clark / simple / smooth), set_param (levels,
render_levels, enabled, name), stacking, apply (bake), remove, and error paths —
plus real geometry checks (triangle count grows on subdivide, reverts when disabled).

This file is the TEMPLATE for the per-feature rt_test_*.py scripts.
"""

import rt

KEEP = True   # ← flip to False for non-destructive (CI) validation

PREFIX = "RTTestMod_"     # unique names so we never touch the user's objects
_created = []             # objects we made, for cleanup at the end


# ── helpers ─────────────────────────────────────────────────────────────────
def _make_cube(suffix, size=2.0, x=0.0):
    name = PREFIX + suffix
    rt.scene.add_primitive("cube", name=name, size=size)
    if x:
        rt.scene.set_transform(name, translation=(x, 0.0, 0.0))
    _created.append(name)
    return name


def _tri_count(name):
    for o in rt.scene.objects():
        if o["name"] == name:
            return o["triangles"]
    return 0


def _expect_error(fn, what):
    try:
        fn()
    except (RuntimeError, ValueError):
        return
    raise AssertionError("expected an error but none was raised: " + what)


def _cleanup():
    for name in _created:
        while rt.modifiers.get_stack(name):
            rt.modifiers.remove(name, index=0)
        rt.scene.delete(name)
    _created.clear()


print("[rt.modifiers] === comprehensive test (KEEP=%s) ===" % KEEP)

# ── 1. Empty stack on a fresh object ────────────────────────────────────────
cube = _make_cube("A", x=-3.0)
base_tris = _tri_count(cube)
assert base_tris > 0, "fresh cube reports no triangles"
assert rt.modifiers.get_stack(cube) == [], "fresh object must have an empty modifier stack"
print("  [1] fresh object empty stack: OK (base=%d tris)" % base_tris)

# ── 2. add catmull_clark + verify returned dict AND stack + geometry ────────
mod = rt.modifiers.add(cube, type="catmull_clark", name="Subdiv", levels=2, render_levels=3)
assert mod["type"] == "catmull_clark"
assert mod["name"] == "Subdiv"
assert mod["levels"] == 2 and mod["render_levels"] == 3
assert mod["enabled"]
assert mod["index"] == 0
stack = rt.modifiers.get_stack(cube)
assert len(stack) == 1 and stack[0]["type"] == "catmull_clark"
sub_tris = _tri_count(cube)
assert sub_tris > base_tris, "catmull_clark did not subdivide (%d !> %d)" % (sub_tris, base_tris)
print("  [2] add catmull_clark: OK (%d -> %d tris)" % (base_tris, sub_tris))

# ── 3. set_param — levels, render_levels, name ──────────────────────────────
rt.modifiers.set_param(cube, index=0, levels=3)
assert rt.modifiers.get_stack(cube)[0]["levels"] == 3
tris_l3 = _tri_count(cube)
assert tris_l3 > sub_tris, "raising levels 2->3 did not subdivide further (%d !> %d)" % (tris_l3, sub_tris)
rt.modifiers.set_param(cube, index=0, render_levels=4, name="SubdivRenamed")
s0 = rt.modifiers.get_stack(cube)[0]
assert s0["render_levels"] == 4 and s0["name"] == "SubdivRenamed"
print("  [3] set_param levels/render_levels/name: OK (levels 3 -> %d tris)" % tris_l3)

# ── 4. enabled toggle — disabling reverts to base geometry ──────────────────
rt.modifiers.set_param(cube, index=0, enabled=False)
assert not rt.modifiers.get_stack(cube)[0]["enabled"]
assert _tri_count(cube) == base_tris, "disabled modifier must show base geometry"
rt.modifiers.set_param(cube, index=0, enabled=True)
assert rt.modifiers.get_stack(cube)[0]["enabled"]
assert _tri_count(cube) > base_tris
print("  [4] enabled toggle reverts/restores geometry: OK")

# ── 5. stacking — add a second modifier on top ──────────────────────────────
mod2 = rt.modifiers.add(cube, type="simple", name="Simple2", levels=1)
stack = rt.modifiers.get_stack(cube)
assert len(stack) == 2, "stack must hold two modifiers"
assert stack[1]["type"] == "simple" and mod2["index"] == 1
print("  [5] stack two modifiers (catmull_clark + simple): OK")

# ── 6. apply (bake) index 0 → collapses into base, stack shrinks ────────────
before_apply = rt.modifiers.get_stack(cube)
rt.modifiers.apply(cube, index=0)
after_apply = rt.modifiers.get_stack(cube)
assert len(after_apply) == len(before_apply) - 1, "apply must remove the applied modifier"
print("  [6] apply/bake index 0: OK (%d -> %d modifiers)" % (len(before_apply), len(after_apply)))

# ── 7. remove remaining modifiers → back to empty ───────────────────────────
while rt.modifiers.get_stack(cube):
    rt.modifiers.remove(cube, index=0)
assert rt.modifiers.get_stack(cube) == []
print("  [7] remove all modifiers: OK")

# ── 8. every subdivision type actually produces geometry ────────────────────
for t in ("catmull_clark", "simple", "smooth"):
    obj = _make_cube("T_" + t)
    before = _tri_count(obj)
    rt.modifiers.add(obj, type=t, levels=2)
    after = _tri_count(obj)
    assert after > before, "%s produced no extra geometry (%d !> %d)" % (t, after, before)
print("  [8] catmull_clark / simple / smooth all subdivide: OK")

# ── 9. error paths ──────────────────────────────────────────────────────────
assert rt.modifiers.get_stack(PREFIX + "missing") == []      # missing object -> empty, no raise
_expect_error(lambda: rt.modifiers.add(PREFIX + "missing", type="catmull_clark"), "add on missing object")
_expect_error(lambda: rt.modifiers.add(cube, type="bogus_type"), "add with unsupported type")
_expect_error(lambda: rt.modifiers.remove(cube, index=999), "remove with out-of-range index")
_expect_error(lambda: rt.modifiers.set_param(cube, index=999, levels=2), "set_param with out-of-range index")
print("  [9] error paths (missing obj / bad type / bad index): OK")

# ── done: always clean up intermediates; KEEP leaves ONE showcase ───────────
_cleanup()

if KEEP:
    demo = PREFIX + "SHOWCASE"
    rt.scene.add_primitive("cube", name=demo, size=2.0)
    rt.modifiers.add(demo, type="catmull_clark", name="DemoSubdiv", levels=2, render_levels=3)
    rt.reset_accumulation()
    rt.request_render()
    print("[rt.modifiers] PASS — showcase '%s' left in scene (subdivided). Delete when done." % demo)
else:
    print("[rt.modifiers] PASS — non-destructive, scene restored.")
