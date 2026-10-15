# Scene acoustics: direct-path diagnostic

2026-10-15 source delivery. No build, application launch, listening acceptance or
new test script was performed. The user requested that scripts remain at the end
of structural work. This adds a geometry/material diagnostic, not audible
occlusion, reflected voices, convolution or a completed wall-transmission solver.

The optional [session listening preview](SCENE_ACOUSTICS_PREVIEW.md) now consumes
these traces. The trace operation itself remains read-only. `object_barrier` is
also accepted as an explicit one-coefficient-per-object assembly approximation.

## Shared operation and UI

`rt.audio.acoustics_trace(...)` and external IPC `audio.acoustics_trace` enter the
same scene-bound AudioAuthoring -> querySceneAcoustics -> traceAcousticSegment
path. The selected Audio Source inspector exposes **Scene acoustics / Direct
path** and **Trace direct path**. Its UI uses the same operation with the source
ID and current camera. It reports a snapshot, not an automatically updated acoustic
state. Changes to source/listener pose, scene geometry generation, timeline frame,
material-library revision, object assignments/legacy collider material changes,
pending deletion, unit scale or audio generation hide a stale UI result.
`materials_revision` is a transient fingerprint, not a persistent material ID or
cryptographic integrity guarantee.

```python
# Authored emitter world position to the current camera:
result = rt.audio.acoustics_trace(id="audio_source_1", meters_per_unit=1)

# Explicit positions, no camera required:
result = rt.audio.acoustics_trace(
    source_position=[0, 1, 0], listener_position=[5, 1, 0],
    sound_speed_mps=343, boundary_mode="diagnostic",
)
# IPC has the same parameters with method audio.acoustics_trace.
```

Exactly one of id/source_position is required. ID lookup uses the emitter's
authored worldPose including its parent transform; it does not pretend to sample
reactive follow-position voices. An omitted listener_position uses the camera;
explicit null or malformed vectors are rejected. Positions contain three finite
numbers within +/-1e9 scene units. Endpoints must be at least 0.1 mm apart after
unit conversion. Sound speed is explicitly authored here, not automatically read
from atmosphere/thermal state.

| Option | Default | Bounds / meaning |
|---|---|---|
| meters_per_unit | 1 | Finite 1e-6..1e6 |
| sound_speed_mps | 343 | Finite 100..2000 m/s |
| max_triangles | 100000 | Integer 1..500000; overflow rejects the query |
| max_hits | 128 | Integer 1..256 distinct crossings |
| include_hidden | true | Boolean; render visibility does not suppress acoustic walls by default |
| boundary_mode | diagnostic | diagnostic, thin_surface or object_barrier |

There is also a fixed 2048 distinct-node snapshot budget and an eight-times-max_hits
raw intersection budget before duplicate removal. Limits reject instead of
silently sampling/truncating geometry. Unknown parameters, invalid types/numbers,
missing source/camera, bad mesh indices or unavailable position buffers fail via
the existing audio error/Python exception path. Authoring data is unchanged.

The operation requires Read, with no FilesRead/SceneWrite, and does not decode
assets, select sources, open a device, enable telemetry, create files or register
physics consumers. It is an explicit synchronous diagnostic; snapshot capture and
BVH construction run on the caller/main thread. No realtime performance budget or
background acoustic scheduling is claimed by this initial path.

## Geometry and material snapshot

The adapter reads canonical TriangleMesh/DNA positions and flat index buffers
directly. It follows the CPU flat-mesh position convention: skinned live local P,
otherwise P_orig, with current object transform applied once. Missing P_orig falls
back to P. Index checks use actual attribute-buffer lengths rather than trusting
the nominal vertex count. World positions must be finite. No Triangle facade
collection, renderer hit cache or material shader transparency becomes authority.

Faces, positions and resolved object-material metadata are owned by the snapshot.
A median-split bounded BVH performs double-precision segment/triangle queries.
Hits are sorted by distance; coincident coplanar crossings on the same node are
deduplicated so adjacent triangles do not count as independent walls. Different
normals at a coincident same-node boundary are flagged ambiguous. Endpoint contacts
within 10 micrometres, or the normalized double-precision tolerance on very long
segments, are excluded. Degenerate triangles have no intersection; grazing/coplanar
propagation is not a diffraction/edge-contact model.

Coverage is explicitly `authored_flat_mesh_only`. Transient meshes and encountered
non-flat Hittables are counted as skipped; pending-deleted flat nodes are excluded.
Intentional render-hidden exclusions are counted. `snapshot_complete=false` makes
aggregate transmission unknown if any such skipped entries exist. This flag is
about the enumerated snapshot scope, not a guarantee of full rendered-world
coverage: external instance managers, volume boundaries and other representations
are not enumerated by this adapter. The UI always labels its flat-mesh coverage.

## Result and transmission semantics

Results include distance_meters, delay_seconds = distance/sound_speed,
geometrically_obstructed, triangle/coverage counters and ordered hits with world
position, normal, distance and object material provenance. Material metadata comes
from the shared object-substance resolver, including legacy collider inheritance,
conflicts and the configured acoustic energy response. A crossing is a geometric
obstruction report, not a claim that all sound is blocked.

`transmission_energy` is either null or six energy fractions at 125, 250, 500,
1000, 2000 and 4000 Hz. Status is:

- clear: no crossings in a complete supported snapshot; all six fractions are 1.
- requires_boundary_model: known materials cross the path in diagnostic mode;
  no aggregate loss is invented for a thick/closed/layered wall.
- unknown_material: an encountered response is unassigned, conflicting or undefined.
- unsupported_geometry: snapshot omissions prevent a complete estimate.
- ambiguous_boundary: coincident noncoplanar same-node crossings are ambiguous.
- evaluated_thin_surface: user explicitly selected independent thin boundaries
  and every crossed material is defined; multiply their transmission ENERGY
  coefficients per band. This is not an amplitude/pressure gain or complete
  propagation attenuation.

`thin_surface` is an explicit modeling assumption that every retained crossing is
an independent zero-thickness boundary. Do not use it to treat the entry and exit
of a closed wall as independent assemblies. Default diagnostic mode deliberately
does not make that mistake. Closed-shell intervals, measured assembly thickness,
layering and incidence-dependent transmission are subsequent work. There is no
distance/air absorption, reflected path, audible filter or automatic playback
modulation in this delivery.

## Deferred acceptance

Final user build: Studio C++ only, no new shader. Add to the final combined pass:
unobstructed path/units/delay; a single flat wall; coplanar triangle-edge duplicate
removal; rotated/scaled/skinned/deformed meshes and position-buffer bounds;
explicit/legacy/conflicting/unknown acoustic materials; diagnostic vs thin-boundary
semantics; closed-wall entry/exit reporting; ambiguous/grazing/end-point cases;
hidden/pending-deleted/unsupported/transient coverage; resource failures without
state mutation; UI/script/IPC parity and Read-only authorization; stale UI results.

Static checks cover source delimiters/size/whitespace, VS source registration,
all nine dynamic binding parameters and IPC capability/security mirror. They do
not establish C++ compilation or acoustic numerical/runtime acceptance.
