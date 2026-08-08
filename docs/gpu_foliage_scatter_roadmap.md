# GPU Foliage Scatter Roadmap

## Invariants

- Vulkan RT and Solid consume one canonical GPU instance set.
- Terrain fill, mesh fill, brush add and brush erase use identical include/exclude semantics.
- Exclusion is a veto; missing include fields resolve to one and missing exclusions to zero.
- Candidate RNG depends only on seed, candidate ID and stream ID.
- Every phase retains the current CPU implementation as a failure fallback.

## Phase 1 — Contract and parity gate

Deliverables: fixed std430 CPU/GPU ABI, deterministic RNG, shared rejection bits,
synthetic parity compute kernel and per-reason mismatch telemetry.

Exit test: at least one million synthetic candidates produce zero CPU/GPU decision
or RNG mismatches on the target Vulkan device.

## Phase 2 — Terrain and brush production

Deliverables: terrain height/normal and named-field buffers, splat buffers, terrain
fill, brush add, brush erase, alive mask, deterministic prefix compaction and exact
minimum-distance spatial grid.

Exit test: identical seed/settings give stable saved transforms; include/exclusion,
slope, height, curvature, direction and edge tests match the CPU reference. A 50k
terrain fill completes below 100 ms on the reference scene, excluding TLAS build.

## Phase 3 — Vulkan RT canonical consumption

Deliverables: RT instance preparation reads the canonical buffer directly, topology
BUILD and transform UPDATE use the same buffer, and per-instance CPU packing/upload is
removed.

Exit test: RT instance count and transforms match Phase 2 output; telemetry reports
zero source-upload bytes for an unchanged GPU-resident group.

## Phase 4 — Vulkan Solid GPU-driven draw

Deliverables: frustum/distance/LOD culling, visible-instance compaction, indirect draw
commands and shared instance transforms with RT.

Exit test: RT/Solid instance parity is exact before visibility culling; camera motion
causes no CPU matrix rebuild or per-mesh instance upload.

## Phase 5 — Mesh scatter

Deliverables: triangle area alias/CDF sampling, barycentric density/exclusion/scale
attributes, mesh brush projection and the common acceptance/compaction path.

Exit test: statistical triangle-area distribution and every mask decision match the
CPU reference; multi-material flat source meshes preserve source selection.

## Phase 6 — Persistence and backend switching

Deliverables: revisioned asynchronous readback for save/undo, GPU restoration on load,
and lazy `FoliageInstanceBatch` construction when entering a CPU renderer.

Exit test: save/load and undo preserve deterministic order; Vulkan-to-CPU-to-Vulkan
round trips do not change transforms or require scene-object facades.

## Phase 7 — Stress and fallback

Deliverables: telemetry for candidate, mask, compaction, upload, TLAS and raster passes;
memory budgets; graceful CPU fallback; 50k/500k/1m/2m benchmark presets.

Exit test: no validation errors, stale descriptor use, device loss or silent parity
failure across repeated edits and backend switches.
