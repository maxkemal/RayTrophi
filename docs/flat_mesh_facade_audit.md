# Flat Mesh / Triangle Facade Audit

## Current selection contract

- Object identity is `TriangleMesh` plus the hit face index.
- `SelectableItem::object` is only a temporary, single-triangle compatibility handle.
- A legacy `Triangle` selection is normalized to its `parentMesh` automatically.
- Transform, bounds, multi-selection equality/removal, selection rebinding, duplication,
  and selected-only export now prefer the canonical mesh.

## Facade use that remains

### 1. UI lookup bridge

`SceneUI::mesh_cache`, `direct_mesh_nodes`, and `direct_mesh_rep_by_ptr` still keep one
representative `Triangle` for a flat mesh. Hierarchy, timeline, force-field, modifier,
material, gizmo, and several property panels still read the selected node name or
transform through this handle. This does not materialize every face, but it keeps the
old API dependency alive.

Recommended replacement: expose common object accessors on `SelectableItem` (name,
transform, visibility, material target) and migrate panels to those accessors before
removing the representative maps.

### 2. Edit-mode control cages and topology undo

`base_mesh_cache`, edit display meshes, and topology undo commands still store
`vector<shared_ptr<Triangle>>`. The cage-less edit/sculpt path already writes the flat
SoA mesh directly, but subdivision control-cage editing intentionally falls back to
facades.

Recommended replacement: introduce a welded control-mesh snapshot containing vertex,
index, corner/UV, and material arrays. Move topology undo to snapshots/deltas before
removing the cage facade path.

### 3. Foliage/scatter source storage

Terrain foliage and Scatter Brush sources now retain `TriangleMesh` references rather
than a facade for every source face. Project-load source resolution follows the same
contract. The legacy `triangles` vector remains only for old triangle-soup geometry.

The centered runtime/backend bridge still expands faces while building its shared BLAS
source. Its temporary input facade set is released immediately after the centered
runtime geometry exists, so flat sources no longer retain both copies.

Recommended replacement: teach `HittableInstance` and all render backends to consume a
centered indexed `TriangleMesh` directly; that removes the final runtime facade set.

### 4. Per-face temporary expansion

Target-surface sampling, material/UV editing, shading operations, and a few modifier/edit
operations temporarily create a `Triangle` per face. These are not selection handles;
they are compatibility adapters for algorithms that still accept triangle vectors.

Recommended replacement: migrate these algorithms independently to
`GeometryDetail`/indexed SoA spans. Avoid changing all of them in one patch because
material corners, welded topology, and terrain ownership have different invariants.

### 5. Legacy triangle-soup producers

Some primitive creation and import paths still construct standalone triangles. These
are real legacy scene geometry rather than flat-mesh facades, so selection migration
alone cannot remove them.

Animation-clip-only import now skips geometry, material, texture, and per-face facade
extraction entirely. Full model import still has a temporary facade peak before its
post-import flat collapse.

Recommended replacement: make primitive/import output a `TriangleMesh` at the source,
then keep triangle-soup loading only as a project-compatibility conversion path.

## Already protected flat consumers

- Vulkan/CPU hit records carry `tri_mesh` and `tri_face`.
- Direct flat duplication deep-copies the whole mesh.
- Selected-only export receives the whole flat mesh.
- Soft delete hides the actual parent mesh, not only its representative.
- Material and edit paths contain explicit flat-SoA branches where a one-facade loop
  would otherwise affect only face zero.

## Suggested migration order

1. Shared `SelectableItem` object accessors and UI panel conversion.
2. Hierarchy, gizmo, timeline, force-field, and material selection consumers.
3. Indexed control-mesh edit/undo representation.
4. Per-face algorithm adapters.
5. Legacy primitive/import triangle-soup producers.
