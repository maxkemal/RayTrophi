# Geometry Cache Deformation

## Contract

`GeometryCacheClip` is a scene-level, source-independent fixed-topology deformation
cache. It targets a named canonical flat `TriangleMesh` and stores only local-space
vertex positions for sampled frames.

The target mesh remains authoritative for:

- vertex/index topology;
- UV and material ID attributes;
- custom attributes and skin metadata;
- object transform and hierarchy.

Every clip records a hash of vertex count plus the complete index buffer. Bake aborts
as soon as this topology changes. Playback also validates the hash and refuses stale
data instead of indexing the wrong vertex layout.

## Authoring and playback

Geometry Graph toolbar `Bake Cache` evaluates the timeline range with expensive
backend/BVH publication deferred, captures `P`, then publishes the completed result
once. An enabled cache suppresses live curve graph regeneration for its host and
drives `P/P_orig` directly. Normals are derived from the unchanged index buffer, so
they do not consume cache storage.

Final animation rendering applies the same clip before per-frame backend geometry
sync. Vulkan and OptiX use the existing flat-mesh BLAS refit path. Cache disable/clear
restores live Geometry Graph evaluation without deleting the graph or source splines.

Full project saves place tightly packed `Vec3` samples in the binary sidecar. Scene
JSON exchange has a JSON fallback for portability. Cache deletion follows object
deletion, and a source graph signature exposes when rebaking is required.

## Shared surfaces

Python:

```python
rt.geometry_cache.bake("CableHost", 0, 120, frame_step=1)
rt.geometry_cache.status("CableHost")
rt.geometry_cache.set_enabled("CableHost", True)
rt.geometry_cache.clear("CableHost")
```

IPC uses the matching `geometry_cache.bake/status/set_enabled/clear` methods. Both
surfaces call the same `rtapi` implementation used by the Geometry Graph toolbar.

## Deliberate boundaries

- Variable topology is rejected. A later full-geometry sequence cache is a separate,
  more expensive data type.
- Shape keys are not geometry-cache samples. They can later feed the same deformation
  playback layer as authored sparse targets with animated weights.
- Version one stores integer-frame samples and linearly interpolates skipped frames.
  Motion-blur/subframe sampling can extend the sample time representation later.
- Position compression/quantization is deferred until real scene measurements justify
  its quality and decode cost.

## Release validation

Development validation targets the Release executable. `geometry_cache.self_test`
checks capture, interpolation, packed-binary round-trip, topology rejection and memory
accounting. `scripts/test/rt_test_spline_curve_ipc.ps1` exercises spline animation,
Curve-to-Mesh, cache bake/status/toggle/clear and temporary-object cleanup through IPC.
