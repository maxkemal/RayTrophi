# Terrain Surface Channel Contract

Terrain surface authoring uses two paint-resolution RGBA textures. Material
coverage and physical/semantic controls must never be packed into the same
normalized weight set.

## Material RGBA

| Channel | Meaning | Contract |
| --- | --- | --- |
| R | Grass | Material coverage |
| G | Rock | Material coverage |
| B | Snow | Material coverage |
| A | Soil | Material coverage |

The four channels are normalized material weights. Flow is not a material
channel and must not be merged into Soil.

## Semantic RGBA

| Channel | Meaning | Default shading use |
| --- | --- | --- |
| R | Flow | Wet-channel darkening and roughness response |
| G | Wetness | Wet-surface darkening and roughness response |
| B | Ice | Cold tint and lower roughness |
| A | Hardness | Small roughness modulation |

Semantic channels are independent `[0, 1]` controls and are not normalized.
Surface Composer emits both textures. Auto Splat emits material RGBA plus a
semantic texture whose R channel contains its optional Flow input. Splat Output
publishes both maps to the terrain at the configured paint resolution.

Existing serialized graphs are migrated on load: when Surface Composer or Auto
Splat already drives the material input of Splat Output and its semantic input
is empty, the matching semantic output is connected automatically. Terrain
serialization format v3 stores the semantic texture as a separate PNG.

## Derived authoring masks

Mud, Moss and Cavity are authoring masks, not additional normalized material
channels. Mud should reinforce Soil coverage and Wetness; Moss can reinforce
Grass coverage and SatMap tint; Cavity/Concavity should drive SatMap breakup,
roughness or the choice between Rock and Soil. Keeping them outside material
RGBA avoids expanding the terrain contract to eight normalized layers and
prevents physical signals from competing with visible material coverage.

## Build verification

The Vulkan terrain-layer struct changed from 48 to 64 bytes. Both
`closesthit.rchit` and `material_preview_frag.frag` must be rebuilt; stale SPIR-V
is incompatible with the new CPU buffer layout.

Manual verification:

1. Use a terrain with mesh resolution 1024 and paint resolution 4096.
2. Evaluate a graph containing Surface Composer and Splat Output.
3. Confirm material A remains Soil while increasing Flow.
4. Confirm Flow/Wetness darken and smooth the surface without replacing Soil.
5. Confirm Grass, Soil and Flow boundaries are visibly distinct.
6. Save and reload the project; confirm the semantic map and graph cable remain.
7. Check Vulkan RT, viewport preview, CPU and OptiX paths for matching response.
