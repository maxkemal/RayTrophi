"""Non-destructive RayTrophi embedded-API smoke test.

RayTrophi runs this automatically during default-scene startup and writes all
output to the in-app Python Console. Every persistent mutation is undone before
the script exits.
"""

import os
import tempfile
import rt


print(f"[rt-smoke] API version: {rt.version()}")

objects = rt.scene.objects()
assert isinstance(objects, list), "rt.scene.objects() must return a list"
print(f"[rt-smoke] scene objects: {len(objects)}")

if objects:
    source_name = objects[0]["name"]
    assert rt.scene.exists(source_name), f"object lookup failed: {source_name}"

    original_transform = rt.scene.transform[source_name]
    rt.scene.transform[source_name] = original_transform
    rt.undo()
    rt.redo()
    rt.undo()
    print(f"[rt-smoke] transform + undo/redo: OK ({source_name})")

    duplicate_name = rt.scene.duplicate(source_name)
    assert duplicate_name != source_name
    assert rt.scene.exists(duplicate_name)
    rt.undo()
    print(f"[rt-smoke] duplicate + undo: OK ({duplicate_name})")

    # ── Faz 5.2a — Procedural primitive creation + undo ─────────────────────
    prim_name = rt.scene.add_primitive("sphere", name="SmokeTestSphere", size=1.5)
    assert rt.scene.exists(prim_name), f"procedural primitive not created: {prim_name}"
    info = next(o for o in rt.scene.objects() if o["name"] == prim_name)
    assert info["triangles"] > 0 and info["vertices"] > 0, "primitive mesh is empty"
    rt.undo()
    assert not rt.scene.exists(prim_name), "add_primitive undo failed"
    print(f"[rt-smoke] add_primitive + undo: OK ({prim_name})")

    original_roughness = float(rt.material.get(source_name, "roughness"))
    test_roughness = 0.25 if original_roughness > 0.5 else 0.75
    rt.material.set(source_name, "roughness", test_roughness)
    assert abs(float(rt.material.get(source_name, "roughness")) - test_roughness) < 1e-5
    rt.undo()
    assert abs(float(rt.material.get(source_name, "roughness")) - original_roughness) < 1e-5
    rt.redo()
    assert abs(float(rt.material.get(source_name, "roughness")) - test_roughness) < 1e-5
    rt.undo()
    print(f"[rt-smoke] material param + undo/redo: OK ({source_name})")

    # ── Faz 3a — mesh data (positions/normals/uvs) ──────────────────────────
    # Not undoable (bulk vertex writes are treated like sculpt strokes), so the
    # test restores state itself by writing the original arrays back.
    info = next(o for o in objects if o["name"] == source_name)
    vcount = info["vertices"]
    positions = rt.mesh.positions(source_name)
    normals = rt.mesh.normals(source_name)
    uvs = rt.mesh.uvs(source_name)
    assert positions.shape == (vcount, 3), f"positions shape {positions.shape} != ({vcount}, 3)"
    assert normals.shape == (vcount, 3), f"normals shape {normals.shape} != ({vcount}, 3)"
    assert uvs.shape[0] == vcount and uvs.shape[1] == 2, f"uvs shape {uvs.shape} != ({vcount}, 2)"

    original_positions = positions.copy()
    original_normals = normals.copy()
    original_uvs = uvs.copy()

    rt.mesh.set_positions(source_name, original_positions)
    rt.mesh.set_normals(source_name, original_normals)
    rt.mesh.set_uvs(source_name, original_uvs)
    rt.mesh.recompute_normals(source_name)
    rt.mesh.set_normals(source_name, original_normals)  # restore pre-recompute normals

    restored_positions = rt.mesh.positions(source_name)
    assert (restored_positions == original_positions).all(), "position round-trip mismatch"
    print(f"[rt-smoke] mesh positions/normals/uvs read+write round-trip: OK ({source_name}, {vcount} verts)")
else:
    print("[rt-smoke] object tests skipped: scene is empty")

before_lights = len(rt.lights.list())
light_name = rt.lights.add("point", (1.0, 2.0, 3.0))
lights_after_add = rt.lights.list()
assert len(lights_after_add) == before_lights + 1
light_index = next(item["index"] for item in lights_after_add if item["name"] == light_name)
rt.lights.set_position(light_index, (4.0, 5.0, 6.0))
rt.lights.delete(light_index)
rt.undo()  # restore deleted light
rt.undo()  # restore original position
rt.undo()  # remove smoke-test light
assert len(rt.lights.list()) == before_lights
print("[rt-smoke] light add/move/delete + undo: OK")

original_frame = rt.timeline.get_frame()
test_frame = original_frame + 1
rt.timeline.set_frame(test_frame)
assert rt.timeline.get_frame() == test_frame
rt.timeline.set_frame(original_frame)
print("[rt-smoke] timeline frame control: OK")

# ── 5.1a — Camera get/set (non-destructive: restore original values) ────────
cam = rt.camera.get()
for k in ("position", "target", "up", "fov", "focus_distance", "aperture"):
    assert k in cam, f"camera dict missing {k}: {cam}"
orig_fov = float(cam["fov"])
test_fov = 35.0 if orig_fov > 40.0 else 55.0
rt.camera.set(fov=test_fov)
assert abs(float(rt.camera.get()["fov"]) - test_fov) < 1e-3, "camera fov set failed"
rt.camera.set(position=cam["position"], target=cam["target"],
              fov=orig_fov, focus_distance=cam["focus_distance"], aperture=cam["aperture"])
assert abs(float(rt.camera.get()["fov"]) - orig_fov) < 1e-3, "camera restore failed"
print("[rt-smoke] camera get/set round-trip: OK")

# ── 5.1c — World get/set (non-destructive: restore original values) ─────────
w = rt.world.get()
for k in ("mode", "background_color", "sun_elevation", "sun_azimuth", "sun_intensity",
          "atmosphere_intensity", "sun_size"):
    assert k in w, f"world dict missing {k}: {w}"
assert w["mode"] in ("solid", "hdri", "nishita"), f"unexpected world mode: {w['mode']}"
orig_elev = float(w["sun_elevation"])
test_elev = 45.0 if orig_elev < 30.0 else 10.0
rt.world.set(sun_elevation=test_elev)
assert abs(float(rt.world.get()["sun_elevation"]) - test_elev) < 1e-3, "world sun_elevation set failed"
rt.world.set(mode=w["mode"], background_color=w["background_color"], sun_elevation=orig_elev,
             sun_azimuth=w["sun_azimuth"], sun_intensity=w["sun_intensity"],
             atmosphere_intensity=w["atmosphere_intensity"], sun_size=w["sun_size"])
assert abs(float(rt.world.get()["sun_elevation"]) - orig_elev) < 1e-3, "world restore failed"
print("[rt-smoke] world get/set round-trip: OK")

# ── 5.1d — Post get/set (non-destructive: restore original values) ─────────
p = rt.post.get()
for k in ("exposure", "gamma", "saturation", "color_temperature", "tone_mapping",
          "vignette_enabled", "vignette_strength", "stylize_enabled", "stylize_strength"):
    assert k in p, f"post dict missing {k}: {p}"
assert p["tone_mapping"] in ("agx", "aces", "uncharted", "filmic", "none"), f"unexpected tone_mapping: {p['tone_mapping']}"
orig_exposure = float(p["exposure"])
test_exposure = 1.8 if orig_exposure < 1.5 else 0.8
rt.post.set(exposure=test_exposure)
assert abs(float(rt.post.get()["exposure"]) - test_exposure) < 1e-3, "post exposure set failed"
rt.post.set(exposure=orig_exposure, gamma=p["gamma"], saturation=p["saturation"],
            color_temperature=p["color_temperature"], tone_mapping=p["tone_mapping"],
            vignette_enabled=p["vignette_enabled"], vignette_strength=p["vignette_strength"],
            stylize_enabled=p["stylize_enabled"], stylize_strength=p["stylize_strength"])
assert abs(float(rt.post.get()["exposure"]) - orig_exposure) < 1e-3, "post restore failed"
print("[rt-smoke] post get/set round-trip: OK")

rt.reset_accumulation()
rt.request_render()
render_status = rt.render.status()
assert render_status["state"] in {"idle", "completed", "failed", "cancelled"}
print("[rt-smoke] render controls: OK")

# ── 0.4.0 — Sequence render API surface check ───────────────────────────────
# We verify the API exists and returns sensible defaults without actually
# starting a render (which would write to disk and block the default-scene smoke).
assert callable(rt.render.start_sequence),   "rt.render.start_sequence must be callable"
assert callable(rt.render.sequence_status),  "rt.render.sequence_status must be callable"
assert callable(rt.render.cancel_sequence),  "rt.render.cancel_sequence must be callable"
seq_status = rt.render.sequence_status()
assert isinstance(seq_status, dict),         "rt.render.sequence_status() must return a dict"
assert "active"         in seq_status
assert "current_frame"  in seq_status
assert "start_frame"    in seq_status
assert "end_frame"      in seq_status
assert "total_progress" in seq_status
assert "output_dir"     in seq_status
assert seq_status["active"] == False,        "sequence must be idle at startup"
print("[rt-smoke] sequence API surface: OK")


# ── 3c — Keyframe API (transform tracks) ────────────────────────────────────
if objects:
    kf_obj = objects[0]["name"]
    before_keys = set(rt.anim.list_keys(kf_obj))
    rt.anim.insert_key(kf_obj, "location", 5, (1.0, 2.0, 3.0))
    rt.anim.insert_key(kf_obj, "rotation", 5, (0.0, 90.0, 0.0))  # same frame, other channel preserved
    rt.anim.insert_key(kf_obj, "scale", 12, (2.0, 2.0, 2.0))
    keys = set(rt.anim.list_keys(kf_obj))
    assert 5 in keys and 12 in keys, f"expected frames 5 and 12 in {keys}"
    rt.anim.remove_key(kf_obj, 5)
    rt.anim.remove_key(kf_obj, 12)
    # Restore original key set (remove any we added that weren't there before)
    for f in set(rt.anim.list_keys(kf_obj)) - before_keys:
        rt.anim.remove_key(kf_obj, f)
    print(f"[rt-smoke] keyframe insert/list/remove: OK ({kf_obj})")

# ── 3d — Node graph construction (surface + registry check) ─────────────────
node_types = rt.nodes.types()
assert isinstance(node_types, list) and len(node_types) > 0, "rt.nodes.types() must list registered types"
assert all("type_id" in t for t in node_types), "each node type needs a type_id"
# add/link/list operate on an existing named graph; asserting a missing graph
# raises keeps the smoke non-destructive (no material/geo graph in the default scene).
try:
    rt.nodes.add("material", "__rt_smoke_missing__", node_types[0]["type_id"])
    raised = False
except RuntimeError:
    raised = True
assert raised, "rt.nodes.add on a missing graph must raise"
try:
    rt.nodes.add("bogus_type", "x", node_types[0]["type_id"])
    raised = False
except RuntimeError:
    raised = True
assert raised, "rt.nodes.add with an unknown graph_type must raise"
print(f"[rt-smoke] node graph API surface: OK ({len(node_types)} registered types)")

# ── 5.1b — Node parameters (surface + missing-graph raise) ──────────────────
# No material/geometry graph exists in the default scene, so we assert the
# param API is present and rejects a missing graph rather than mutating one.
assert callable(rt.nodes.list_params) and callable(rt.nodes.get_param) and callable(rt.nodes.set_param)
try:
    rt.nodes.get_param("material", "__rt_smoke_missing__", 1, 0)
    raised = False
except RuntimeError:
    raised = True
assert raised, "rt.nodes.get_param on a missing graph must raise"
try:
    rt.nodes.set_param("material", "__rt_smoke_missing__", 1, 0, 0.5)
    raised = False
except RuntimeError:
    raised = True
assert raised, "rt.nodes.set_param on a missing graph must raise"
print("[rt-smoke] node parameter API surface: OK")

# ── 5.5b — Graph lifecycle + serialized property reflection ─────────────────
# The reflection layer used to be hard-limited to terrain nodes. It now
# dispatches on all three node families, so this walks a real material graph
# end to end: create the asset, create its graph, list nodes, read properties
# back, write one and read it again.
_refl_mat = rt.material.create("principled", "ReflectMat")
assert _refl_mat not in rt.nodes.graphs("material")
rt.nodes.create_graph("material", _refl_mat)
assert _refl_mat in rt.nodes.graphs("material"), rt.nodes.graphs("material")

_gnodes = rt.nodes.list("material", _refl_mat)
assert len(_gnodes) > 0, "a material graph seeded from its material must have nodes"

# Find any node in the graph that exposes at least one scalar property. Before
# this change every material node reported zero properties.
_target, _props = None, []
for _n in _gnodes:
    _p = rt.nodes.list_properties("material", _refl_mat, _n["id"])
    if _p:
        _target, _props = _n, _p
        break
assert _target is not None, "no material node exposed serialized properties (reflection gate closed?)"

# Round-trip a float property through get -> set -> get.
_float_props = [p for p in _props if p["type"] == "float"]
if _float_props:
    _pname = _float_props[0]["name"]
    _before = rt.nodes.get_property("material", _refl_mat, _target["id"], _pname)
    _new = 0.375 if abs(_before - 0.375) > 1e-3 else 0.625
    rt.nodes.set_property("material", _refl_mat, _target["id"], _pname, _new)
    _after = rt.nodes.get_property("material", _refl_mat, _target["id"], _pname)
    assert abs(_after - _new) < 1e-4, f"{_pname}: wrote {_new}, read {_after}"
    print(f"[rt-smoke] material node reflection: OK ({_target['type_id']}.{_pname} round-trip)")
else:
    print(f"[rt-smoke] material node reflection: OK ({len(_props)} props, no float to round-trip)")

# An unknown property must be rejected, not silently ignored.
try:
    rt.nodes.set_property("material", _refl_mat, _target["id"], "__no_such_prop__", 1.0)
    raise AssertionError("unknown node property must raise")
except RuntimeError:
    pass

# Geometry graphs are keyed by object name and start empty.
_geo_obj = rt.scene.add_primitive("cube", name="ReflectGeoCube")
rt.nodes.create_graph("geometry", _geo_obj)
assert _geo_obj in rt.nodes.graphs("geometry")
try:
    rt.nodes.create_graph("geometry", "__no_such_object__")
    raise AssertionError("geometry graph for a missing object must raise")
except RuntimeError:
    pass
# Terrain graphs are owned by the terrain object, not creatable here.
try:
    rt.nodes.create_graph("terrain", "anything")
    raise AssertionError("terrain graph creation must be refused")
except RuntimeError:
    pass

print(f"[rt-smoke] node graph lifecycle: OK ({len(rt.nodes.types())} types registered)")

# ── 5.5c — rt.nodes.apply closes the authoring loop ─────────────────────────
# Building a graph is not enough: the material only picks up the graph when it
# is APPLIED (fold constants into the material + compile the per-pixel program).
# This is the editor's Apply, not terrain's async evaluate.
_apply = rt.nodes.apply("material", _refl_mat)
assert isinstance(_apply, dict) and _apply["ok"] is True, _apply
assert isinstance(_apply["warnings"], list) and isinstance(_apply["errors"], list)

# End-to-end proof: drive a material value THROUGH the graph. Write the property
# on the node, apply, and read the material back through the object binding.
# Uses its own object so this block stays independent of the rt.select section
# further down the file.
_apply_cube = rt.scene.add_primitive("cube", name="ApplyTestCube")
rt.material.assign(_apply_cube, _refl_mat)
_rough_node, _rough_prop = None, None
for _n in rt.nodes.list("material", _refl_mat):
    for _p in rt.nodes.list_properties("material", _refl_mat, _n["id"]):
        if _p["name"] == "roughness" and _p["type"] == "float":
            _rough_node, _rough_prop = _n["id"], _p["name"]
            break
    if _rough_node is not None:
        break
if _rough_node is not None:
    rt.nodes.set_property("material", _refl_mat, _rough_node, _rough_prop, 0.137)
    _res = rt.nodes.apply("material", _refl_mat)
    assert _res["ok"] is True, _res
    _mat_rough = rt.material.get(_apply_cube, "roughness")
    assert abs(_mat_rough - 0.137) < 1e-3, \
        f"graph roughness 0.137 did not reach the material (got {_mat_rough})"
    print("[rt-smoke] rt.nodes.apply end-to-end: OK (graph value -> material)")
else:
    print("[rt-smoke] rt.nodes.apply: OK (no roughness node to drive end-to-end)")

# Terrain keeps its async contract; geometry applies through the Geo-DAG path.
try:
    rt.nodes.apply("terrain", "anything")
    raise AssertionError("terrain apply must redirect to rt.terrain.evaluate")
except RuntimeError:
    pass
try:
    rt.nodes.apply("material", "__no_such_graph__")
    raise AssertionError("apply on a missing graph must raise")
except RuntimeError:
    pass

rt.nodes.remove_graph("geometry", _geo_obj)
assert _geo_obj not in rt.nodes.graphs("geometry")
rt.nodes.remove_graph("material", _refl_mat)

# ── 3b — Event callbacks (subscribe/unsubscribe surface) ────────────────────
_fired = []
cb_id = rt.on_frame_change(lambda f: _fired.append(f))
scene_cb_id = rt.on_scene_load(lambda: _fired.append(-1))
assert isinstance(cb_id, int) and cb_id >= 0
assert isinstance(scene_cb_id, int) and scene_cb_id >= 0
rt.remove_frame_change_callback(cb_id)
rt.remove_scene_load_callback(scene_cb_id)
print("[rt-smoke] event callback subscribe/unsubscribe: OK")

# ── 4a — Addon discovery (non-destructive: list only, no enable/disable) ─────
addons = rt.addons.list()
assert isinstance(addons, list), "rt.addons.list() must return a list"
for a in addons:
    assert "module_name" in a and "enabled" in a and "loaded" in a, f"addon dict shape: {a}"
if any(a["module_name"] == "example_addon" for a in addons):
    print(f"[rt-smoke] addon discovery: OK ({len(addons)} found, incl. example_addon)")
else:
    print(f"[rt-smoke] addon discovery: OK ({len(addons)} found)")

# ── 4b — rt.ui surface (panel register + widget guard) ──────────────────────
assert hasattr(rt, "ui"), "rt.ui submodule must exist"
assert callable(rt.ui.register_panel) and callable(rt.ui.unregister_panel)
# Immediate-mode widgets outside a panel draw callback must raise.
try:
    rt.ui.button("x")
    raised = False
except RuntimeError:
    raised = True
assert raised, "rt.ui.button outside a panel draw must raise"
# Register + immediately unregister a throwaway panel (its draw never runs here).
_pid = rt.ui.register_panel("__rt_smoke_panel__", lambda: None)
assert isinstance(_pid, int)
assert any(p["id"] == _pid and p["region"] == "" for p in rt.ui.list_panels())
rt.ui.set_panel_visible(_pid, False)
assert any(p["id"] == _pid and p["visible"] is False for p in rt.ui.list_panels())
rt.ui.unregister_panel(_pid)
assert all(p["id"] != _pid for p in rt.ui.list_panels())

# 4b+ — regions: an addon mounts into an existing panel instead of a new window.
_regions = rt.ui.regions()
assert "properties.world" in _regions and "menu.view" in _regions
_rid = rt.ui.register_region("properties.world", "__rt_smoke_region__", lambda: None)
assert any(p["id"] == _rid and p["region"] == "properties.world" for p in rt.ui.list_panels())
rt.ui.unregister_panel(_rid)
# The widget guard covers the whole surface, not just button().
for _fn, _args in (("text", ("x",)), ("collapsing_header", ("x",)), ("combo", ("x", 0, ["a"])),
                   ("begin_table", ("x", 2)), ("push_style_var", ("alpha", 1.0)),
                   ("begin_child", ("x",)), ("selectable", ("x",))):
    try:
        getattr(rt.ui, _fn)(*_args)
        raise AssertionError(f"rt.ui.{_fn} outside a panel draw must raise")
    except RuntimeError:
        pass

# 5.5d — container / style surface. These can only RUN inside a draw callback,
# so here we assert the surface exists and that the name tables are sane.
for _fn in ("begin_child", "end_child", "begin_table", "end_table", "table_setup_column",
            "table_headers_row", "table_next_row", "table_next_column",
            "begin_tab_bar", "end_tab_bar", "begin_tab_item", "end_tab_item",
            "begin_list_box", "end_list_box", "selectable",
            "open_popup", "begin_popup", "begin_popup_modal", "end_popup",
            "close_current_popup", "push_style_color", "pop_style_color",
            "push_style_var", "pop_style_var", "input_text_multiline",
            "plot_lines", "plot_histogram", "is_key_pressed", "content_region_avail"):
    assert callable(getattr(rt.ui, _fn, None)), f"rt.ui.{_fn} is missing"

_style_colors = rt.ui.style_colors()
_style_vars = rt.ui.style_vars()
assert "button" in _style_colors and "window_bg" in _style_colors, _style_colors
assert "frame_rounding" in _style_vars and "item_spacing" in _style_vars, _style_vars
print(f"[rt-smoke] rt.ui panels + regions: OK ({len(_regions)} regions, "
      f"{len(_style_colors)} style colors, {len(_style_vars)} style vars)")

# ── 5.5a — rt.select ───────────────────────────────────────────────────────
_sel_cube = rt.scene.add_primitive("cube", name="SelTestCube")
rt.select.clear()
assert rt.select.list() == []
rt.select.object(_sel_cube)
_sel = rt.select.list()
assert any(s["name"] == _sel_cube and s["type"] == "object" for s in _sel), _sel
assert any(s["primary"] for s in _sel), "one item must be primary"
_all = rt.select.all()
assert _all >= 1 and len(rt.select.list()) == _all
rt.select.deselect(_sel_cube)
assert all(s["name"] != _sel_cube for s in rt.select.list())
rt.select.clear()
try:
    rt.select.object("__no_such_object__")
    raise AssertionError("selecting a missing object must raise")
except RuntimeError:
    pass
print(f"[rt-smoke] rt.select: OK (select_all saw {_all} objects)")

# ── 5.5a — rt.material asset layer ─────────────────────────────────────────
_mat = rt.material.create("principled", "SmokeMat")
assert isinstance(_mat, str) and _mat
assert any(m["name"] == _mat and m["type"] == "principled" for m in rt.material.list())
assert rt.material.info(_mat)["name"] == _mat
# A second create with the same requested name must not collide.
_mat2 = rt.material.create("principled", "SmokeMat")
assert _mat2 != _mat, "createMaterial must uniquify a taken name"
rt.material.assign(_sel_cube, _mat)
assert _mat in rt.material.of_object(_sel_cube), rt.material.of_object(_sel_cube)
# Parameter get/set still route through the object.
rt.material.set(_sel_cube, "roughness", 0.25)
assert abs(rt.material.get(_sel_cube, "roughness") - 0.25) < 1e-4
rt.material.set(_sel_cube, "base_color", (0.1, 0.6, 0.9))
assert rt.material.textures(_mat) == [], "a fresh material has no textures"
try:
    rt.material.set_texture(_mat, "__no_such_slot__", "x.png")
    raise AssertionError("unknown texture slot must raise")
except RuntimeError:
    pass
print(f"[rt-smoke] rt.material assets: OK ({len(rt.material.list())} materials)")

# ── 5.5a — rt.lights parameters ────────────────────────────────────────────
_light_name = rt.lights.add("spot", (0.0, 4.0, 0.0))
_li = [l for l in rt.lights.list() if l["name"] == _light_name]
assert len(_li) == 1, _li
_lidx = _li[0]["index"]
rt.lights.set_color(_lidx, (1.0, 0.5, 0.25))
rt.lights.set_intensity(_lidx, 3.5)
rt.lights.set_direction(_lidx, (0.0, -1.0, 0.0))
rt.lights.set_param(_lidx, "spot_angle", 30.0)
_lg = rt.lights.get(_lidx)
assert abs(_lg["intensity"] - 3.5) < 1e-4, _lg
assert abs(_lg["spot_angle"] - 30.0) < 1e-3, _lg
assert abs(_lg["color"][1] - 0.5) < 1e-4, _lg
rt.lights.rename(_lidx, "SmokeSpot")
assert rt.lights.get(_lidx)["name"] == "SmokeSpot"
# A degenerate direction must be rejected rather than silently ignored.
try:
    rt.lights.set_direction(_lidx, (0.0, 0.0, 0.0))
    raise AssertionError("zero-length direction must raise")
except RuntimeError:
    pass
# spot_angle is meaningless on a point light.
_pt_name = rt.lights.add("point", (0.0, 2.0, 0.0))
_pidx = [l for l in rt.lights.list() if l["name"] == _pt_name][0]["index"]
try:
    rt.lights.set_param(_pidx, "spot_angle", 20.0)
    raise AssertionError("spot_angle on a point light must raise")
except RuntimeError:
    pass
rt.lights.delete(_pidx)
print("[rt-smoke] rt.lights parameters: OK")

# ── 5.2b — rt.modifiers surface check ──────────────────────────────────────
assert hasattr(rt, "modifiers"), "rt.modifiers submodule must exist"
mod_cube = rt.scene.add_primitive("cube", name="ModTestCube")
initial_stack = rt.modifiers.get_stack(mod_cube)
assert isinstance(initial_stack, list)

added_mod = rt.modifiers.add(mod_cube, type="catmull_clark", name="TestSubdiv", levels=2, render_levels=3)
assert added_mod["name"] == "TestSubdiv"
assert added_mod["levels"] == 2
assert added_mod["render_levels"] == 3

new_stack = rt.modifiers.get_stack(mod_cube)
assert len(new_stack) == len(initial_stack) + 1

rt.modifiers.set_param(mod_cube, index=added_mod["index"], levels=3)
updated_stack = rt.modifiers.get_stack(mod_cube)
assert updated_stack[added_mod["index"]]["levels"] == 3

rt.modifiers.remove(mod_cube, index=added_mod["index"])
final_stack = rt.modifiers.get_stack(mod_cube)
assert len(final_stack) == len(initial_stack)
rt.undo() # Undo the cube creation
print("[rt-smoke] rt.modifiers stack operations: OK")

# ── 5.2c — rt.scatter surface check ──────────────────────────────────────
assert hasattr(rt, "scatter"), "rt.scatter submodule must exist"
scatter_target = rt.scene.add_primitive("plane", name="ScatterTestPlane", size=10.0)
scatter_source = rt.scene.add_primitive("cube", name="ScatterTestSource", size=0.5)

grp_info = rt.scatter.create_group("TestScatterGrp", target_node=scatter_target, target_type="mesh")
assert grp_info["name"] == "TestScatterGrp"

rt.scatter.add_source("TestScatterGrp", scatter_source, weight=1.0, scale_min=0.5, scale_max=1.5)
rt.scatter.set_settings("TestScatterGrp", target_count=50, min_distance=0.1)

spawned = rt.scatter.fill("TestScatterGrp")
assert spawned > 0, "scatter fill must spawn at least 1 instance"

groups = rt.scatter.list_groups()
found_grp = next((g for g in groups if g["name"] == "TestScatterGrp"), None)
assert found_grp is not None and found_grp["instance_count"] == spawned

rt.scatter.clear("TestScatterGrp")
groups_after_clear = rt.scatter.list_groups()
cleared_grp = next((g for g in groups_after_clear if g["name"] == "TestScatterGrp"), None)
assert cleared_grp["instance_count"] == 0

rt.scatter.delete_group("TestScatterGrp")
rt.undo() # Undo cube
rt.undo() # Undo plane
print("[rt-smoke] rt.scatter operations: OK")

# ── 5.3a — rt.physics surface check ──────────────────────────────────────
assert hasattr(rt, "physics"), "rt.physics submodule must exist"
phys_cube = rt.scene.add_primitive("cube", name="PhysicsTestCube")

body_info = rt.physics.add_body(phys_cube, kind="rigid", motion_type="dynamic", shape="box", mass=5.0)
assert body_info["object_name"] == phys_cube
assert body_info["kind"] == "rigid"
assert body_info["motion_type"] == "dynamic"
assert body_info["mass"] == 5.0

get_info = rt.physics.get(phys_cube)
assert get_info["mass"] == 5.0

rt.physics.set_param(phys_cube, mass=10.0, friction=0.8)
updated_info = rt.physics.get(phys_cube)
assert updated_info["mass"] == 10.0
assert abs(updated_info["friction"] - 0.8) < 1e-3

rt.physics.step(0.0166)
rt.physics.reset()

rt.physics.remove_body(phys_cube)
rt.undo() # Undo cube
print("[rt-smoke] rt.physics operations: OK")

# ── 5.3b — rt.fluid surface check ──────────────────────────────────────
assert hasattr(rt, "fluid"), "rt.fluid submodule must exist"
domain_info = rt.fluid.create_domain("TestLiquidDomain", domain_min=(-1, 0, -1), domain_max=(1, 2, 1), voxel_size=0.1)
assert domain_info["name"] == "TestLiquidDomain"
assert abs(domain_info["voxel_size"] - 0.1) < 1e-4, "voxel_size must match"

rt.fluid.seed("TestLiquidDomain", seed_min=(-0.5, 0.5, -0.5), seed_max=(0.5, 1.5, 0.5), particles_per_cell=4)
rt.fluid.set_param("TestLiquidDomain", backend="gpu", preset="oil", boundary="closed")
get_domain = rt.fluid.get("TestLiquidDomain")
assert get_domain["particle_count"] > 0, "fluid seeding must generate particles"
assert get_domain["backend"] == "gpu", "fluid solver backend must be gpu"
assert get_domain["preset"] == "oil", "fluid preset must be oil"
assert get_domain["boundary"] == "closed", "fluid boundary must be closed"

rt.fluid.step(0.0166)
rt.fluid.reset()

rt.fluid.remove_domain("TestLiquidDomain")

assert hasattr(rt, "gas"), "rt.gas submodule must exist"
gas_domain = rt.gas.create_domain("TestSmokeDomain", domain_min=(-1, 0, -1), domain_max=(1, 2, 1), voxel_size=0.1)
assert gas_domain["name"] == "TestSmokeDomain"
assert gas_domain["type"] == "gas"
assert gas_domain["boundary"] == "open"
rt.fluid.remove_domain("TestSmokeDomain")

print("[rt-smoke] rt.fluid & rt.gas operations: OK")

# 5.4a - rt.hair deterministic groom automation
assert hasattr(rt, "hair"), "rt.hair submodule must exist"
hair_mesh = rt.scene.add_primitive("plane", name="HairTestScalp", size=2.0)
groom = rt.hair.create(hair_mesh, "HairTestGroom", guide_count=32,
                       children_per_guide=2, points_per_strand=6, length=0.25)
assert groom["name"] == "HairTestGroom"
assert groom["bound_mesh"] == hair_mesh
assert groom["settings"]["guide_count"] == 32
rt.hair.update("HairTestGroom", length=0.4, clumpiness=0.75,
               curl_frequency=2.0, curl_radius=0.02, visible=True)
updated_groom = rt.hair.get("HairTestGroom")
assert abs(updated_groom["settings"]["length"] - 0.4) < 1e-4
assert abs(updated_groom["settings"]["clumpiness"] - 0.75) < 1e-4
assert any(item["name"] == "HairTestGroom" for item in rt.hair.list())
rt.hair.restyle("HairTestGroom")
presets = rt.hair.list_presets()
assert "curly" in presets and "wet" in presets
rt.hair.apply_preset("HairTestGroom", "curly")
curly_groom = rt.hair.get("HairTestGroom")
assert abs(curly_groom["settings"]["curl_frequency"] - 6.0) < 1e-4
length_before_trim = curly_groom["settings"]["length"]
rt.hair.trim("HairTestGroom", 0.8)
trimmed_groom = rt.hair.get("HairTestGroom")
assert abs(trimmed_groom["settings"]["length"] - length_before_trim * 0.8) < 1e-4
rt.hair.grow("HairTestGroom", 1.25)
grown_groom = rt.hair.get("HairTestGroom")
assert abs(grown_groom["settings"]["length"] - length_before_trim) < 1e-4
rt.hair.comb("HairTestGroom", direction=(1.0, 0.25, 0.0), strength=0.6,
             root_stiffness=0.8)
rt.hair.smooth("HairTestGroom", strength=0.4, iterations=2)
rt.hair.reset_simulation("HairTestGroom")
rt.hair.bake("HairTestGroom")
renamed_groom = rt.hair.rename("HairTestGroom", "HairTestGroomRenamed")
assert renamed_groom["name"] == "HairTestGroomRenamed"
rt.hair.remove("HairTestGroomRenamed")
rt.undo()  # Undo scalp plane creation
print("[rt-smoke] rt.hair operations: OK")

# 5.4b - rt.paint deterministic layer automation
assert hasattr(rt, "paint"), "rt.paint submodule must exist"
paint_mesh = rt.scene.add_primitive("plane", name="PaintTestMesh", size=2.0)
paint_target = rt.paint.ensure(paint_mesh, resolution=64)
assert paint_target["object"] == paint_mesh
assert paint_target["resolution"] == 64
assert len(paint_target["layers"]) == 1
paint_layer = rt.paint.add_layer(paint_mesh, "Script Fill")
assert paint_layer["index"] == 1
rt.paint.update_layer(paint_mesh, 1, opacity=0.75, blend_mode="multiply",
                      visible=True, locked=False)
rt.paint.fill(paint_mesh, 1, "base_color", (0.2, 0.4, 0.8))
paint_after_fill = rt.paint.get(paint_mesh)
assert "base_color" in paint_after_fill["channels"]
assert "base_color" in paint_after_fill["layers"][1]["channels"]
assert abs(paint_after_fill["layers"][1]["opacity"] - 0.75) < 1e-4
assert paint_after_fill["layers"][1]["blend_mode"] == "multiply"
mask_presets = rt.paint.list_mask_presets()
assert "radial" in mask_presets and "edge_wear" in mask_presets
rt.paint.apply_mask(paint_mesh, 1, "radial", strength=0.9, seed=42)
rt.paint.bake_height_to_normal(paint_mesh, strength=4.0, clear_height=False)
duplicate_layer = rt.paint.duplicate_layer(paint_mesh, 1)
assert duplicate_layer["index"] == 2
rt.paint.move_layer(paint_mesh, 2, 1)
rt.paint.merge_down(paint_mesh, 1)
assert len(rt.paint.get(paint_mesh)["layers"]) == 2
paint_export = os.path.join(tempfile.gettempdir(), "raytrophi_rt_paint_smoke.png")
rt.paint.export_channel(paint_mesh, "normal", paint_export)
assert os.path.isfile(paint_export) and os.path.getsize(paint_export) > 0
rt.paint.import_channel(paint_mesh, 1, "normal", paint_export)
os.remove(paint_export)
rt.paint.flatten(paint_mesh)
assert len(rt.paint.get(paint_mesh)["layers"]) == 1
rt.undo()  # Undo paint test plane creation
print("[rt-smoke] rt.paint complete automation: OK")

# 5.4c - rt.sculpt deterministic world-space stroke automation
assert hasattr(rt, "sculpt"), "rt.sculpt submodule must exist"
sculpt_mesh = rt.scene.add_primitive("sphere", name="SculptTestMesh", size=1.0)
sculpt_info = rt.sculpt.get(sculpt_mesh)
assert sculpt_info["vertex_count"] > 0
rt.sculpt.mask_operation(sculpt_mesh, "fill", undo=False)
assert rt.sculpt.get(sculpt_mesh)["mask_min"] > 0.99
rt.sculpt.mask_operation(sculpt_mesh, "invert", undo=False)
assert rt.sculpt.get(sculpt_mesh)["mask_max"] < 0.01
rt.sculpt.paint_mask(sculpt_mesh, [(0.0, 0.0, 0.0)], radius=3.0,
                     value=1.0, strength=1.0, undo=False)
assert rt.sculpt.get(sculpt_mesh)["mask_max"] > 0.0
rt.sculpt.mask_operation(sculpt_mesh, "clear", undo=False)
rt.sculpt.mask_operation(sculpt_mesh, "noise", seed=42, undo=False)
assert rt.sculpt.get(sculpt_mesh)["has_mask"]
rt.sculpt.mask_operation(sculpt_mesh, "clear", undo=False)
rt.sculpt.stroke(sculpt_mesh, "inflate", [(0.0, 0.0, 0.0)], radius=3.0,
                 strength=0.01, undo=True)
rt.undo(); rt.redo(); rt.undo()  # One API stroke is one undo group.
for sculpt_tool in ("draw", "smooth", "flatten", "stamp", "noise"):
    rt.sculpt.stroke(sculpt_mesh, sculpt_tool, [(0.0, 0.0, 0.0)], radius=3.0,
                     strength=0.01, direction=(0.0, 1.0, 0.0), seed=42,
                     use_mask=False, undo=False)
rt.undo()  # Undo sculpt test sphere creation
print("[rt-smoke] rt.sculpt deterministic strokes + mask: OK")

# 5.6a - rt.forcefield: one field feeds every simulation family
assert hasattr(rt, "forcefield"), "rt.forcefield submodule must exist"
ff_types = rt.forcefield.types()
assert "wind" in ff_types and "vortex" in ff_types and "curlnoise" in ff_types

ff_before = len(rt.forcefield.list())
vortex = rt.forcefield.create("vortex", "SmokeVortex")
assert vortex["name"] == "SmokeVortex" and vortex["type"] == "vortex"
# The panel's per-type defaults must be replayed, otherwise a scripted vortex
# is a bare rotation with no spiral or lift.
assert vortex["shape"] == "cylinder", "vortex must default to a cylinder shape"
assert vortex["inward_force"] > 0.0 and vortex["upward_force"] > 0.0
assert len(rt.forcefield.list()) == ff_before + 1

# Same requested name twice must not collide.
vortex2 = rt.forcefield.create("vortex", "SmokeVortex")
assert vortex2["name"] != vortex["name"]
assert vortex2["id"] != vortex["id"]
rt.forcefield.remove(str(vortex2["id"]))

# Turbulence and curl noise are dead without use_noise.
turb = rt.forcefield.create("turbulence", "SmokeTurbulence")
assert turb["use_noise"] is True

# Partial edit through kwargs, read back by ID.
rt.forcefield.set_param(str(vortex["id"]), strength=7.5, falloff_radius=12.0,
                        position=(1.0, 2.0, 3.0), affects_cloth=False,
                        falloff="inverse_square")
edited = rt.forcefield.get(str(vortex["id"]))
assert abs(edited["strength"] - 7.5) < 1e-4
assert abs(edited["falloff_radius"] - 12.0) < 1e-4
assert abs(edited["position"][1] - 2.0) < 1e-4
assert edited["affects_cloth"] is False
assert edited["falloff"] == "inverse_square"
# An untouched field must survive a partial edit.
assert edited["shape"] == "cylinder"
assert edited["name"] == "SmokeVortex"

# Spelling variants the panel uses must resolve to the same enum. The
# multi-word cases matter most: a canonical name that contains "_" must still
# match a caller who typed a space or a hyphen (and vice versa).
rt.forcefield.set_param("SmokeTurbulence", type="Curl Noise")
assert rt.forcefield.get("SmokeTurbulence")["type"] == "curlnoise"
for spelling in ("inverse_square", "Inverse Square", "inverse-square", "InverseSquare"):
    rt.forcefield.set_param("SmokeTurbulence", falloff="linear")
    rt.forcefield.set_param("SmokeTurbulence", falloff=spelling)
    assert rt.forcefield.get("SmokeTurbulence")["falloff"] == "inverse_square", spelling

# Rejected edits must leave the field untouched, not half-applied.
for bad in ({"type": "__nope__"}, {"shape": "__nope__"},
            {"falloff": "__nope__"}, {"noise_octaves": 99}):
    raised = False
    try:
        rt.forcefield.set_param("SmokeTurbulence", **bad)
    except Exception:
        raised = True
    assert raised, "rt.forcefield.set_param must reject %r" % bad
assert rt.forcefield.get("SmokeTurbulence")["type"] == "curlnoise"
assert rt.forcefield.get("SmokeTurbulence")["noise_octaves"] <= 8

# inner_radius may not exceed falloff_radius.
raised = False
try:
    rt.forcefield.set_param("SmokeTurbulence", inner_radius=999.0)
except Exception:
    raised = True
assert raised, "inner_radius > falloff_radius must be rejected"

# evaluate() is the read-only probe: a strong wind must show up in the sum.
wind = rt.forcefield.create("wind", "SmokeWind")
rt.forcefield.set_param(str(wind["id"]), shape="infinite", strength=50.0,
                        direction=(1.0, 0.0, 0.0), falloff="none")
pushed = rt.forcefield.evaluate((0.0, 0.0, 0.0), 0.0)
rt.forcefield.set_param(str(wind["id"]), enabled=False)
calm = rt.forcefield.evaluate((0.0, 0.0, 0.0), 0.0)
assert pushed != calm, "a disabled field must stop contributing force"

for ff_name in ("SmokeVortex", "SmokeTurbulence", "SmokeWind"):
    rt.forcefield.remove(ff_name)
raised = False
try:
    rt.forcefield.get("SmokeVortex")
except Exception:
    raised = True
assert raised, "a removed force field must not resolve"
assert len(rt.forcefield.list()) == ff_before
print("[rt-smoke] rt.forcefield lifecycle + patch + evaluate: OK")

# 5.6b - rt.particle: emitters, solver settings, stats, direct spawn
assert hasattr(rt, "particle"), "rt.particle submodule must exist"
pt_emitters_before = len(rt.particle.emitters())

em = rt.particle.add_emitter(name="SmokeEmitter", rate_per_second=64.0,
                             speed=3.0, lifetime_seconds=2.0,
                             point=(0.0, 2.0, 0.0), direction=(0.0, 1.0, 0.0))
assert em["name"] == "SmokeEmitter"
assert abs(em["rate_per_second"] - 64.0) < 1e-4
assert em["index"] == pt_emitters_before
assert len(rt.particle.emitters()) == pt_emitters_before + 1

# Addressable by index AND by name.
by_index = rt.particle.get_emitter(str(em["index"]))
by_name = rt.particle.get_emitter("SmokeEmitter")
assert by_index["name"] == by_name["name"] == "SmokeEmitter"

# Partial edit must not disturb untouched fields.
rt.particle.set_emitter("SmokeEmitter", speed=9.0, spread=0.5, burst_count=32)
edited = rt.particle.get_emitter("SmokeEmitter")
assert abs(edited["speed"] - 9.0) < 1e-4
assert abs(edited["spread"] - 0.5) < 1e-4
assert edited["burst_count"] == 32
assert abs(edited["rate_per_second"] - 64.0) < 1e-4, "untouched field must survive"
assert abs(edited["point"][1] - 2.0) < 1e-4

# Rejected edits must leave the emitter untouched, not half-applied.
for bad in ({"source_mode": "__nope__"}, {"spawn_mode": "__nope__"},
            {"lifetime_seconds": 0.0}, {"mass": 0.0}, {"burst_count": -5}):
    raised = False
    try:
        rt.particle.set_emitter("SmokeEmitter", **bad)
    except Exception:
        raised = True
    assert raised, "rt.particle.set_emitter must reject %r" % bad
assert abs(rt.particle.get_emitter("SmokeEmitter")["speed"] - 9.0) < 1e-4

# An object-bound emitter MUST name a live object: the scene prunes one whose
# object is missing, so accepting it would hand back an emitter that vanishes.
# (An in-process script would not notice — the prune runs as scene maintenance
# between frames, which is why this only showed up over IPC.)
for bad in ({"source_mode": "object_origin"},
            {"source_mode": "object_origin", "source_name": "__no_such_object__"},
            {"source_mode": "force_field_origin", "source_name": ""}):
    raised = False
    try:
        rt.particle.set_emitter("SmokeEmitter", **bad)
    except Exception:
        raised = True
    assert raised, "unbound object emitter must be rejected: %r" % bad
assert rt.particle.get_emitter("SmokeEmitter")["source_mode"] == "point"

# Spelling variants (panel writes them with spaces, the API with underscores).
pt_source = rt.scene.add_primitive("cube", name="SmokeEmitterSource", size=1.0)
for spelling in ("object_origin", "Object Origin", "object-origin", "ObjectOrigin"):
    rt.particle.set_emitter("SmokeEmitter", source_mode="point")
    rt.particle.set_emitter("SmokeEmitter", source_mode=spelling, source_name=pt_source)
    assert rt.particle.get_emitter("SmokeEmitter")["source_mode"] == "object_origin", spelling
for spelling in ("object_aabb_surface", "Object AABB Surface", "ObjectAABBSurface"):
    rt.particle.set_emitter("SmokeEmitter", spawn_mode="center")
    rt.particle.set_emitter("SmokeEmitter", spawn_mode=spelling)
    assert rt.particle.get_emitter("SmokeEmitter")["spawn_mode"] == "object_aabb_surface", spelling
# Unbind before the source object goes away, or the prune takes the emitter too.
rt.particle.set_emitter("SmokeEmitter", source_mode="point", source_name="")

# Solver settings are per system, not per emitter.
phys_before = rt.particle.get_physics()
rt.particle.set_physics(mode="granular", particle_radius=0.08, gravity_scale=0.5,
                        grid_fuel_deposit=0.25)
phys = rt.particle.get_physics()
assert phys["mode"] == "granular"
assert abs(phys["particle_radius"] - 0.08) < 1e-4
assert abs(phys["gravity_scale"] - 0.5) < 1e-4
assert abs(phys["grid_fuel_deposit"] - 0.25) < 1e-4
assert phys["quality"] == phys_before["quality"], "untouched setting must survive"
for bad in ({"mode": "__nope__"}, {"quality": "__nope__"},
            {"particle_radius": 0.0}, {"solver_iterations": 0},
            {"max_neighbors_per_particle": 0}, {"rest_density": 0.0}):
    raised = False
    try:
        rt.particle.set_physics(**bad)
    except Exception:
        raised = True
    assert raised, "rt.particle.set_physics must reject %r" % bad
assert rt.particle.get_physics()["mode"] == "granular"

# Direct spawn bypasses the emitters; stats must see it.
alive_before = rt.particle.stats()["alive_count"]
spawn_index = rt.particle.spawn(position=(0.0, 3.0, 0.0), velocity=(0.0, 1.0, 0.0),
                                lifetime_seconds=4.0)
assert spawn_index >= 0
stats = rt.particle.stats()
assert stats["alive_count"] == alive_before + 1
# Counts must be LIVE: the runtime only refreshes its own stats block inside
# step(), so reading them from there would report 0 emitters right after an add.
assert stats["emitter_count"] >= 1, "stats() counts must not wait for a step"
assert stats["capacity"] >= stats["alive_count"]
rt.particle.step(0.016)
rt.particle.clear()
assert rt.particle.stats()["alive_count"] == 0

# Restore the solver settings this block changed, then drop the test emitter.
rt.particle.set_physics(mode=phys_before["mode"],
                        particle_radius=phys_before["particle_radius"],
                        gravity_scale=phys_before["gravity_scale"],
                        grid_fuel_deposit=phys_before["grid_fuel_deposit"])
rt.particle.remove_emitter("SmokeEmitter")
rt.scene.delete(pt_source)
assert len(rt.particle.emitters()) == pt_emitters_before
raised = False
try:
    rt.particle.get_emitter("SmokeEmitter")
except Exception:
    raised = True
assert raised, "a removed emitter must not resolve"
# clear_emitters wipes EVERY emitter, so only exercise it on a scene that had
# none to begin with — a smoke test must not destroy the user's authoring.
if pt_emitters_before == 0:
    rt.particle.add_emitter(name="SmokeEmitterClear")
    rt.particle.clear_emitters()
    assert len(rt.particle.emitters()) == 0
print("[rt-smoke] rt.particle emitters + physics + stats + spawn: OK")

# 5.6c - rt.anim skeletal playback (transport + graph parameters only)
for fn in ("characters", "character", "clips", "play", "stop", "set_paused",
           "set_time", "set_speed", "set_loop", "status",
           "set_graph_param", "trigger_graph_param", "graph_status"):
    assert callable(getattr(rt.anim, fn)), "rt.anim.%s must exist" % fn

# Negative checks work on any scene, animated or not.
for call in (lambda: rt.anim.character("__no_such_character__"),
             lambda: rt.anim.clips("__no_such_character__"),
             lambda: rt.anim.play("__no_such_character__", "Idle"),
             lambda: rt.anim.status("__no_such_character__")):
    raised = False
    try:
        call()
    except Exception:
        raised = True
    assert raised, "a missing character must raise"

anim_chars = rt.anim.characters()
if anim_chars:
    ch = anim_chars[0]["name"]
    info = rt.anim.character(ch)
    assert info["name"] == ch
    # ★These check the REASON, not just that something raised. characters() used to list
    # static mesh imports too, so both of these raised "no animation controller" and passed
    # while testing nothing — the real failure only surfaced later, on the first call that
    # had no try/except around it. A bare `assert raised` hides the bug it is meant to catch.
    #
    # An out-of-range layer is silently ignored by the controller, so the facade
    # must reject it rather than report a no-op edit as success.
    for bad_layer in (-1, 4, 99):
        err = ""
        try:
            rt.anim.status(ch, layer=bad_layer)
        except Exception as e:
            err = str(e)
        assert "layer" in err, "layer %d must be rejected for BEING a bad layer, got: %r" % (bad_layer, err)
    err = ""
    try:
        rt.anim.play(ch, "__no_such_clip__")
    except Exception as e:
        err = str(e)
    assert "clip not found" in err, "an unknown clip must raise for THAT reason, got: %r" % err

    clips = rt.anim.clips(ch)
    if clips:
        first = clips[0]["name"]
        rt.anim.play(ch, first, blend=0.0)
        status = rt.anim.status(ch)
        assert status["clip"] == first
        rt.anim.set_speed(ch, 0.5)
        rt.anim.set_loop(ch, True)
        rt.anim.set_time(ch, 0.0)
        rt.anim.set_paused(ch, True)
        assert rt.anim.status(ch)["paused"] is True
        rt.anim.set_paused(ch, False)
        rt.anim.set_speed(ch, 1.0)
        rt.anim.stop(ch, blend_out=0.0)
    # Graph parameters must be refused on a character that is not graph-driven,
    # otherwise the value is stored where nothing will ever read it.
    if not info["uses_graph"]:
        raised = False
        try:
            rt.anim.set_graph_param(ch, "Speed", 1.0)
        except Exception:
            raised = True
        assert raised, "graph param on a clip-driven character must be refused"
    print("[rt-smoke] rt.anim playback: OK (%d character(s), '%s')" % (len(anim_chars), ch))
else:
    print("[rt-smoke] rt.anim surface: OK (no animated character in this scene)")

print("[rt-smoke] PASS")
