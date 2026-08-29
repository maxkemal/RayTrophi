"""Structural Hardness + Surface Relief scripting smoke test.

Run inside RayTrophi Studio after the user build. The generic rt.nodes service
is the same service used by IPC, so this checks type registration, JSON-backed
properties and the ready-terrain wiring contract without a UI-only path.
"""
import rt


NAME = "TerrainSurfaceReliefTest"
for terrain in rt.terrain.list():
    if terrain["name"] == NAME:
        rt.terrain.remove(NAME)

rt.terrain.create(name=NAME, resolution=64, mesh_resolution=64,
                  size=256.0, height_scale=64.0)
applied = rt.terrain.apply_preset(
    NAME, "snowy_mountain_valley", replace_graph=True)
assert not applied, applied

nodes = rt.nodes.list("terrain", NAME)
by_type = {node["type_id"]: node for node in nodes}
for required in (
        "TerrainV2.NoiseGenerator",
        "TerrainV2.StructuralHardness",
        "TerrainV2.HydraulicErosion",
        "TerrainV2.SurfaceRelief",
        "TerrainV2.HardnessOutput"):
    assert required in by_type, (required, sorted(by_type))

hardness = by_type["TerrainV2.StructuralHardness"]
relief = by_type["TerrainV2.SurfaceRelief"]
assert hardness["inputs"] == 3 and hardness["outputs"] == 3, hardness
assert relief["inputs"] == 7 and relief["outputs"] == 4, relief

hardness_properties = {
    item["name"] for item in rt.nodes.list_properties(
        "terrain", NAME, hardness["id"])
}
assert {
    "baseHardness", "slopeStartDegrees", "slopeFullDegrees",
    "exposureHardening", "convexityInfluence", "sharpnessInfluence",
    "heterogeneity", "fractureStrength", "fractureSoftening", "seed",
} <= hardness_properties, hardness_properties

relief_properties = {
    item["name"] for item in rt.nodes.list_properties(
        "terrain", NAME, relief["id"])
}
assert {
    "featureSizeMeters", "rockAmplitudeMeters", "soilAmplitudeMeters",
    "rillDepthMeters", "directionStretch", "flowInfluence",
    "samplesPerFeature", "maxAddedSlopeDegrees", "seed",
} <= relief_properties, relief_properties

rt.nodes.set_property(
    "terrain", NAME, relief["id"], "rockAmplitudeMeters", 0.625)
assert abs(rt.nodes.get_property(
    "terrain", NAME, relief["id"], "rockAmplitudeMeters") - 0.625) < 1e-6

# Ground Detail was a uniform generator-wide displacement. Its removal from
# serialized reflection is the compatibility gate for the new architecture.
noise_properties = {
    item["name"] for item in rt.nodes.list_properties(
        "terrain", NAME, by_type["TerrainV2.NoiseGenerator"]["id"])
}
assert "groundDetailMeters" not in noise_properties, noise_properties

rt.terrain.remove(NAME)
print("[terrain surface relief] node registration and script reflection: OK")
