"""Read-only smoke test for the embedded `rt.templates` scripting surface."""

import rt


rt.templates.refresh()
all_entries = rt.templates.list(include_invalid=True)
valid_entries = rt.templates.list()

assert isinstance(all_entries, list)
assert isinstance(valid_entries, list)
assert len(valid_entries) <= len(all_entries)

sort_keys = [
    (item["sort_order"], item["display_name"], item["id"], item["manifest_path"])
    for item in all_entries
]
assert sort_keys == sorted(sort_keys), "template registry order is not deterministic"

ids = [item["id"] for item in all_entries if item["id"]]
assert len(ids) == len(set(ids)), "duplicate template IDs escaped registry validation"
assert "raytrophi.start.empty" in ids, "built-in Empty template was not discovered"
assert "raytrophi.start.general_scene" in ids, "built-in General Scene template was not discovered"

for item in all_entries:
    assert isinstance(item["valid"], bool)
    assert isinstance(item["errors"], list)
    fetched = rt.templates.get(item["id"])
    checked = rt.templates.validate(item["id"])
    assert fetched["manifest_path"] == item["manifest_path"]
    assert checked["valid"] == item["valid"]
    assert checked["errors"] == item["errors"]

missing_plan = rt.templates.prepare("raytrophi.missing.template")
assert not missing_plan["ready"]
assert missing_plan["code"] == "template_not_found"

empty_plan = rt.templates.prepare("raytrophi.start.empty", conflict_policy="discard")
assert empty_plan["ready"], empty_plan["errors"]
assert empty_plan["state"] == "ready"
assert empty_plan["code"] == "ready"
assert empty_plan["scene_type"] == "recipe"

general_plan = rt.templates.prepare("raytrophi.start.general_scene", conflict_policy="discard")
assert general_plan["ready"], general_plan["errors"]
assert general_plan["scene_type"] == "recipe"

try:
    rt.templates.prepare("raytrophi.missing.template", conflict_policy="invalid")
except ValueError:
    pass
else:
    raise AssertionError("invalid conflict policy must be rejected")

opened = rt.templates.open("raytrophi.start.empty", conflict_policy="discard")
assert opened["opened"], opened["errors"]
assert opened["state"] == "opened"
assert opened["code"] == "opened"
assert opened["ui_state_applied"]
assert rt.camera.get()["fov"] > 0.0, "Empty template did not retain a viewport camera"

general_open = rt.templates.open("raytrophi.start.general_scene", conflict_policy="discard")
assert general_open["opened"], general_open["errors"]
general_objects = rt.scene.objects()
assert "Default_Cube" in [item["name"] for item in general_objects]
general_cube = next(item for item in general_objects if item["name"] == "Default_Cube")
assert general_cube["triangles"] == 12
general_materials = rt.material.list()
general_material_names = [
    item if isinstance(item, str) else item.get("name") for item in general_materials
]
assert "Default_Cube_Material" in general_material_names
assert len(rt.lights.list()) == 1
assert rt.select.list()[0]["name"] == "Default_Cube"

opened = rt.templates.open("raytrophi.start.empty", conflict_policy="discard")
assert opened["opened"], opened["errors"]
assert rt.scene.objects() == []

invalid_open = rt.templates.open("raytrophi.start.empty", conflict_policy="invalid")
assert not invalid_open["opened"]
assert invalid_open["code"] == "invalid_parameter"

# Preflight/commit parity. `prepare` promises exactly one thing: ready means
# open() will accept it. This held only by convention while the supported-preset
# list existed in two places (loader + stager); a preset added to one and not the
# other produced a `ready` plan whose open() then failed with
# recipe_commit_not_available. The list is now canonical in TemplateRecipeStager
# and this assertion is the guard that keeps it that way as presets are added.
checked_recipes = 0
for item in valid_entries:
    plan = rt.templates.prepare(item["id"], conflict_policy="discard")
    if plan["scene_type"] != "recipe" or not plan["ready"]:
        continue
    result = rt.templates.open(item["id"], conflict_policy="discard")
    assert result["opened"], (
        "prepare() reported ready but open() refused %s: %s"
        % (item["id"], result["errors"])
    )
    checked_recipes += 1
assert checked_recipes >= 2, "expected at least the two built-in recipe templates"

# Project-backed templates must never reach a destructive commit. Whether they
# fail preflight or the staging gate, the active scene has to survive intact.
for item in valid_entries:
    plan = rt.templates.prepare(item["id"], conflict_policy="discard")
    if plan["scene_type"] != "project":
        continue
    before = [entry["name"] for entry in rt.scene.objects()]
    result = rt.templates.open(item["id"], conflict_policy="discard")
    assert not result["opened"], "project template committed without staging support"
    assert [entry["name"] for entry in rt.scene.objects()] == before, (
        "rejected project template mutated the active scene: %s" % item["id"]
    )

opened = rt.templates.open("raytrophi.start.empty", conflict_policy="discard")
assert opened["opened"], opened["errors"]

print("[rt.templates] PASS - %d discovered, %d valid, %d recipe parity"
      % (len(all_entries), len(valid_entries), checked_recipes))
