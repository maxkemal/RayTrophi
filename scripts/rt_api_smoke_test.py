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
rt.ui.unregister_panel(_pid)
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

print("[rt-smoke] PASS")
