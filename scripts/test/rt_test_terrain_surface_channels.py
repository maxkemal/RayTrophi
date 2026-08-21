"""Terrain material/semantic channel scripting and IPC reflection smoke test.

Run inside RayTrophi Studio after the user build. The generic node service is
shared by Python and IPC, so pin counts and links verify the same core contract.
"""
import rt


NAME = "TerrainSurfaceChannelTest"
for terrain in rt.terrain.list():
    if terrain["name"] == NAME:
        rt.terrain.remove(NAME)

rt.terrain.create(name=NAME, resolution=64, size=128.0, height_scale=32.0)
composer_id = rt.nodes.add("terrain", NAME, "TerrainV2.SurfaceComposer")
auto_splat_id = rt.nodes.add("terrain", NAME, "TerrainV2.AutoSplat")
output_id = rt.nodes.add("terrain", NAME, "TerrainV2.SplatOutput")

nodes = {node["id"]: node for node in rt.nodes.list("terrain", NAME)}
assert nodes[composer_id]["outputs"] == 3, nodes[composer_id]
assert nodes[auto_splat_id]["outputs"] == 2, nodes[auto_splat_id]
assert nodes[output_id]["inputs"] == 2, nodes[output_id]

# Surface Composer output 1 is normalized material RGBA. Output 2 is the
# independent semantic RGBA map; Splat Output receives them at inputs 0 and 1.
rt.nodes.link("terrain", NAME, composer_id, 1, output_id, 0)
rt.nodes.link("terrain", NAME, composer_id, 2, output_id, 1)

info = next(item for item in rt.terrain.list() if item["name"] == NAME)
assert info["surface_semantic_channels"] == (
    "flow", "wetness", "ice", "hardness"), info

rt.terrain.remove(NAME)
print("[terrain surface] 4 material + 4 semantic node contract: OK")
