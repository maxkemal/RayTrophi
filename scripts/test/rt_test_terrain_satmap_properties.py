"""SatMap scripting/IPC reflection smoke test.

Run inside RayTrophi Studio after the user build. The generic rt.nodes property
surface is shared with IPC, so this verifies that the new user-facing SatMap
controls are not UI-only and survive the node's JSON contract.
"""
import rt


NAME = "SatMapPropertyTest"
for terrain in rt.terrain.list():
    if terrain["name"] == NAME:
        rt.terrain.remove(NAME)

rt.terrain.create(name=NAME, resolution=64, size=128.0, height_scale=32.0)
paint_info = rt.terrain.set_paint_resolution(NAME, 128)
assert paint_info["paint_grid"] == (128, 128), paint_info
assert paint_info["surface_semantic_channels"] == (
    "flow", "wetness", "ice", "hardness"), paint_info
assert paint_info["has_surface_semantic"] is False, paint_info
rt.terrain.apply_preset(
    NAME, "biome_temperate", replace_graph=True, add_satmap=True)

library_presets = rt.terrain.list_satmap_presets()
library_ids = {item["id"] for item in library_presets}
assert {"alpine_flow_ridges", "temperate_wetland", "river_network_detailed"} <= library_ids
assert all(item["layer_count"] >= 4 for item in library_presets), library_presets
recipe_warnings = rt.terrain.apply_satmap_preset(NAME, "temperate_wetland")
# This graph has no Hydraulic Erosion Channel Width output, so only the
# width-dependent fine-flow layer is skipped; the remaining recipe still builds.
assert recipe_warnings, recipe_warnings
recipe_nodes = rt.nodes.list("terrain", NAME)
assert any(node["type_id"] == "Terrain.SatMapBlend" for node in recipe_nodes)
assert any(node["type_id"] == "Terrain.SurfaceMasks" for node in recipe_nodes)
assert any(node["type_id"] == "Terrain.PaintMaskCombine" for node in recipe_nodes)

satmap_nodes = [node for node in rt.nodes.list("terrain", NAME)
                if node["type_id"] == "Terrain.SatMapColorRamp"]
assert satmap_nodes, satmap_nodes
node_id = satmap_nodes[0]["id"]
assert satmap_nodes[0]["inputs"] == 9, satmap_nodes[0]

properties = {item["name"] for item in
              rt.nodes.list_properties("terrain", NAME, node_id)}
required = {
    "preset", "autoNormalize", "autoDeriveMasks",
    "normalizeLowPercentile", "normalizeHighPercentile",
    "detailStrength", "detailScale", "debugView",
    "slopeBlend", "flowBlend", "soilBlend", "grassBlend",
    "snowBlend", "meltBlend", "avalancheBlend",
}
missing = sorted(required - properties)
assert not missing, "SatMap properties missing from scripting/IPC reflection: %s" % missing

rt.nodes.set_property("terrain", NAME, node_id, "detailStrength", 0.125)
rt.nodes.set_property("terrain", NAME, node_id, "debugView", 3)
assert abs(rt.nodes.get_property("terrain", NAME, node_id, "detailStrength") - 0.125) < 1e-6
assert rt.nodes.get_property("terrain", NAME, node_id, "debugView") == 3

# Preset selection must call the same distribution path as the UI instead of
# changing only the label while leaving the previous ramps behind.
rt.nodes.set_property("terrain", NAME, node_id, "preset", "Alpine")
assert rt.nodes.get_property("terrain", NAME, node_id, "preset") == "Alpine"
assert abs(rt.nodes.get_property(
    "terrain", NAME, node_id, "normalizeHighPercentile") - 90.0) < 1e-6
assert abs(rt.nodes.get_property(
    "terrain", NAME, node_id, "slopeBlend") - 0.72) < 1e-6

# Layer presets produce color only; SatMap Blend owns their spatial coverage.
rt.nodes.set_property("terrain", NAME, node_id, "preset", "Layer: Flow")
assert rt.nodes.get_property("terrain", NAME, node_id, "preset") == "Layer: Flow"
assert rt.nodes.get_property("terrain", NAME, node_id, "autoDeriveMasks") is False
assert abs(rt.nodes.get_property("terrain", NAME, node_id, "flowBlend")) < 1e-6

blend_id = rt.nodes.add("terrain", NAME, "Terrain.SatMapBlend")
blend = next(item for item in rt.nodes.list("terrain", NAME)
             if item["id"] == blend_id)
assert blend["inputs"] == 3 and blend["outputs"] == 1, blend
blend_properties = {item["name"] for item in
                    rt.nodes.list_properties("terrain", NAME, blend_id)}
assert {"opacity", "maskPower", "invertMask"} <= blend_properties
rt.nodes.set_property("terrain", NAME, blend_id, "maskPower", 1.75)
assert abs(rt.nodes.get_property(
    "terrain", NAME, blend_id, "maskPower") - 1.75) < 1e-6

grass_id = rt.nodes.add("terrain", NAME, "Terrain.GrassMask")
grass = next(item for item in rt.nodes.list("terrain", NAME)
             if item["id"] == grass_id)
assert grass["inputs"] == 6 and grass["outputs"] == 1, grass
grass_properties = {item["name"] for item in
                    rt.nodes.list_properties("terrain", NAME, grass_id)}
required_grass = {
    "preset", "density", "maxSlope", "slopeSoftness", "soilInfluence",
    "flowAvoidance", "wetnessPreference", "wetnessRange",
    "hardnessAvoidance", "patchiness", "detailScale", "seed",
}
assert required_grass <= grass_properties, sorted(required_grass - grass_properties)
rt.nodes.set_property("terrain", NAME, grass_id, "density", 0.67)
assert abs(rt.nodes.get_property(
    "terrain", NAME, grass_id, "density") - 0.67) < 1e-6

surface_masks_id = rt.nodes.add("terrain", NAME, "Terrain.SurfaceMasks")
surface_masks = next(item for item in rt.nodes.list("terrain", NAME)
                     if item["id"] == surface_masks_id)
assert surface_masks["inputs"] == 5 and surface_masks["outputs"] == 3, surface_masks
surface_mask_properties = {item["name"] for item in
                           rt.nodes.list_properties("terrain", NAME, surface_masks_id)}
assert {"preset", "cavityPower", "mudStrength", "mossStrength",
        "slopeSuppression", "detailScale", "seed"} <= surface_mask_properties

rt.terrain.remove(NAME)
print("[satmap] property reflection and JSON-backed controls: OK")
