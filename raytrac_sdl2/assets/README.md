# Asset Layout

Runtime assets under `assets/` follow a folder-first structure.

## Standard

- `scenes/default/scene.glb`: default startup scene
- `scenes/default/asset.json`: metadata for the default scene
- `vegetation/trees/<asset_id>/`: tree assets
- `vegetation/bushes/<asset_id>/`: bush assets
- `vegetation/grass/<asset_id>/`: grass assets
- `vegetation/flowers/<asset_id>/`: flower assets
- `rocks/<asset_id>/`: rocks and debris
- `materials/<asset_id>/`: reusable material assets
- `prefabs/<prefab_id>/`: grouped scene assets

## Per-Asset Folder

Use one folder per asset:

```txt
assets/vegetation/trees/oak_01/
  model.glb
  preview.png
  asset.json
```

Supported entry files:

- `scene.glb`
- `model.glb`
- first discovered `.glb`, `.gltf`, `.fbx`, or `.obj`

Supported preview files:

- `preview.png`
- `preview.jpg`
- `preview.jpeg`
- `preview.webp`
- `thumbnail.png`

## Metadata

`asset.json` is optional. If it is missing, the browser derives:

- `id` from folder name
- `name` from folder name
- `category` and `subcategory` from the path
- `entry` from the first supported model file
- `tags` from path segments and file name

The Asset Browser can generate a starter `asset.json` automatically.

Legacy compatibility:

- `assets/default.glb` is kept as a fallback for older builds.
