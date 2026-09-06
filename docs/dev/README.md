# RayTrophi — engineering notes index

This folder holds the working documents behind RayTrophi Studio: roadmaps,
architecture decisions, acceptance-test protocols and root-cause postmortems.
`docs/` itself is the user-facing HTML manual; everything an implementer needs
lives here.

**Most of these are written in Turkish.** The project has been developed by one
person for about two and a half years, and the notes were written to be useful
during that work rather than to be published. They are kept in the repository
anyway, because a roadmap explains *why* and *what is next* — which the code
cannot. If you are reading these to contribute and a document matters to you,
open an issue and it will be translated.

## How to read the status line

Every document starts with a `> **Durum:**` line. Four values:

| Durum | Meaning |
|---|---|
| **AKTIF** | Live plan. Work is in progress or queued here. Read before touching that area. |
| **REFERANS** | Implemented. Records a contract, rule or architecture that is in force. Still binding. |
| **ARSIV** | Finished or closed. Historical record — usually a postmortem, kept because the root cause is worth remembering. |
| **TASLAK** | Proposed. Not implemented; the design may still change. |
| **CANLI** | Rewritten every work batch. Only the latest revision is meaningful. |

`REFERANS` is not "old". A completed postmortem is often the most useful file in
this folder: the root cause was paid for once, and the note is what stops it
being paid for twice.

---

## Active plans (AKTIF)

| Document | Area |
|---|---|
| [IMPORT_EXPORT_OPEN_DEBTS.md](IMPORT_EXPORT_OPEN_DEBTS.md) | **Open debts left by Faz 3.** Assimp is gone and verified; this is what was deliberately left out. Top item is a rule-1 breach: `scene.import_model` returns nothing although ImportStats already measures every phase. Also 4.39M duplicate triangles from shared meshes, unread KHR_texture_transform, and the writer's non-conformant skin contract that the reader compensates for by sniffing `generator`. |
| [REALTIME_VOLUME_TABLE_NEVER_PUBLISHED.md](REALTIME_VOLUME_TABLE_NEVER_PUBLISHED.md) | Realtime viewport drew no volume because the volume packet was published to the render backend only — the raster viewport is a second VulkanBackendAdapter with its own VkDevice, and the SSBO is per device. Also: why the tripwire's `volumeCount=0` was misread |
| [API_SCRIPTING_ROADMAP.md](API_SCRIPTING_ROADMAP.md) | Scripting/IPC API waves and their status table |
| [PHYSICS_VALIDATION.md](PHYSICS_VALIDATION.md) | Is the solver RIGHT, not merely running — analytical cases, and why a script test is blind to the frame loop |
| [IPC_TEST_CHANNEL.md](IPC_TEST_CHANNEL.md) | The second test channel: drives the app from outside so it can see the frame loop — and the first thing it found is that physics.step is reverted 100% before you can read it |
| [SIMULATION_NODE_CONCEPTUAL_MODEL.md](SIMULATION_NODE_CONCEPTUAL_MODEL.md) | What the simulation node layer IS — read this before the object-model report |
| [PROFILE_SPLINE_NEXT_PHASE_ROADMAP.md](PROFILE_SPLINE_NEXT_PHASE_ROADMAP.md) | 2D profile spline authoring core; Faz 3.6 generalizes it into a shared 3D curve for road/scatter-mask/mesh-surface consumers, and Faz 3.7 makes drawing on a surface a shared service so River becomes a consumer of `SplineObject` instead of owning a private curve |
| [SIMULATION_NODE_OBJECT_MODEL.md](SIMULATION_NODE_OBJECT_MODEL.md) | Scope-based object model for the simulation node layer — steps 1-6 done, compiled, and live-verified |
| [AGENT_VIEWPORT_MEASUREMENT_PLAN.md](AGENT_VIEWPORT_MEASUREMENT_PLAN.md) | Driving the viewport and reading render DATA over IPC — the substrate agents verify with |
| [AGENT_DISCOVERY_LAYER_PLAN.md](AGENT_DISCOVERY_LAYER_PLAN.md) | Self-describing API layer — agent.* discovery, method registry, workflow recipes |
| [DESCRIPTOR_TRUTH_VERIFICATION.md](DESCRIPTOR_TRUTH_VERIFICATION.md) | Whether the descriptor prose is TRUE — claim grounding, capability mirror drift |
| [LOCAL_MODEL_HORIZON.md](LOCAL_MODEL_HORIZON.md) | Where a local 8B model breaks down on a long task, and what the loop does about it |
| [BUG_DELETED_NAME_REUSE_GHOST.md](BUG_DELETED_NAME_REUSE_GHOST.md) | OPEN: re-adding a deleted object name yields a half-existing object |
| [TEMPLATE_HUB_UX_ROADMAP.md](TEMPLATE_HUB_UX_ROADMAP.md) | Canonical product direction for startup / template / guided-scene UX |
| [GRANULAR_SIMULATION_ROADMAP.md](GRANULAR_SIMULATION_ROADMAP.md) | Sand and granular MPM/APIC constitutive solver |
| [GRANULAR_GPU_FIRST_PLAN.md](GRANULAR_GPU_FIRST_PLAN.md) | GPU-first execution plan for the granular solver |
| [GRANULAR_GPU_TEST_PROTOCOL.md](GRANULAR_GPU_TEST_PROTOCOL.md) | Vulkan granular acceptance test |
| [GRANULAR_COHESIVE_DAMAGE_TEST.md](GRANULAR_COHESIVE_DAMAGE_TEST.md) | Bonded/cohesive granular damage acceptance test |
| [material_transformation_fracture_roadmap.md](material_transformation_fracture_roadmap.md) | Burning, melting, mass transfer, fracture |
| [NODE_SIMULATION_ARCHITECTURE_PLAN.md](NODE_SIMULATION_ARCHITECTURE_PLAN.md) | Node-based simulation and thermochemistry layer |
| [VULKAN_PRODUCTION_VOLUMETRICS_ROADMAP.md](VULKAN_PRODUCTION_VOLUMETRICS_ROADMAP.md) | Production volumetric path and its release gates |
| [VULKAN_GPU_FORCE_FIELD_SIMULATION_ROADMAP.md](VULKAN_GPU_FORCE_FIELD_SIMULATION_ROADMAP.md) | Force-field evaluation on Vulkan Compute |
| [VOLUME_SHADER_GRAPH_ROADMAP.md](VOLUME_SHADER_GRAPH_ROADMAP.md) | Volume output of the unified material graph |
| [volumetric_cloud_layer_roadmap.md](volumetric_cloud_layer_roadmap.md) | Layered procedural cloud volumes |
| [gpu_foliage_scatter_roadmap.md](gpu_foliage_scatter_roadmap.md) | GPU foliage scatter migration |
| [hydraulic_multipass_next_steps.md](hydraulic_multipass_next_steps.md) | Terrain hydraulic erosion multi-pass |
| [ASSIMP_IMPORT_REPLACEMENT_BRIEF.md](ASSIMP_IMPORT_REPLACEMENT_BRIEF.md) | Assimp import'tan çıkış. **Faz 0 YAZILDI, DERLENMEDİ (2026-09-05):** animasyon anahtar tipi Assimp'ten ayrıştırıldı (`RayTrophi::VectorKey`/`QuatKey`), sarkan `aiNode*` alanı ve ölü `AnimatedObject` söküldü, `.rtp` şeması değişmedi. Faz 1 (doğrudan glTF okuyucu) başlamadı. Kabul aleti: `anim.source_clips` + `scripts/probe_import_export_parity.py` |
| [SCENE_EXPORT_DIRECT_GLTF.md](SCENE_EXPORT_DIRECT_GLTF.md) | Why the Assimp export round trip cost minutes and 15 GB on one giant mesh, and the plan-then-write glTF writer that replaced it — plus `scene.export_gltf`, the script surface export never had |
| [TERRAIN_NODE_CONTRACT_REDESIGN.md](TERRAIN_NODE_CONTRACT_REDESIGN.md) | Compact terrain ports, authoritative field ownership, node pruning and setup migration |
| [TERRAIN_LANDFORM_SHAPE.md](TERRAIN_LANDFORM_SHAPE.md) | Measured: the Noise Generator was producing a homogeneous fractal, not a landscape - no landform wider than a quarter tile, 4.4% flat ground, gaussian hypsometry. Feature Size sized to the terrain, dissection driven by an authored lowland area, and terrain.landform_stats to judge it |
| [TERRAIN_BUILD_CAP_SCOPE.md](TERRAIN_BUILD_CAP_SCOPE.md) | Measured: maxDepositionMeters bounds only the route pass, talus escapes it and never saturates — plus the two instruments that were reporting zeros as success |
| [TERRAIN_PERF_HANDOFF.md](TERRAIN_PERF_HANDOFF.md) | Terrain build cost: what was measured, what is verified, what is NOT, and the ordered next steps — start here before touching terrain performance |
| [TERRAIN_SATMAP_COLORIZER_ROADMAP.md](TERRAIN_SATMAP_COLORIZER_ROADMAP.md) | Gaea-style SatMap macro colour over the existing 4-layer splat blend — why it must modulate rather than replace the albedo, and the Faz 0 split of field / mesh / paint resolution that has to land first |
| [refactoring_implementation_plan.md](refactoring_implementation_plan.md) | DNA / flat-SoA data-oriented core migration |
| [flat_mesh_facade_audit.md](flat_mesh_facade_audit.md) | Remaining `Triangle` facade call sites |

## Rules and architecture in force (REFERANS)

| Document | Area |
|---|---|
| [DEVELOPMENT_PRINCIPLES.md](DEVELOPMENT_PRINCIPLES.md) | Binding engineering rules (file size, ownership, splitting) |
| [BUG_FOLIAGE_ASSET_TEXTURE_ALIASING.md](BUG_FOLIAGE_ASSET_TEXTURE_ALIASING.md) | Why two Asset Library plants wore each other's textures on every backend: one shared import name disarmed the embedded-texture ABA guard. Plus the material-ID generation rule for anything that caches a resolved ID |
| [FOLIAGE_MASK_CONTRACT.md](FOLIAGE_MASK_CONTRACT.md) | Foliage placement masks: why include and exclude are deliberately asymmetric, why the slot count does NOT grow, and the Publish Field node that makes composing masks in the graph the answer instead |
| [VULKAN_SIMULATION_RESET_SAFETY.md](VULKAN_SIMULATION_RESET_SAFETY.md) | Reset/rewind safety rule — TDR avoidance |
| [BUG_SKINNING_DESCRIPTOR_POOL_LEAK.md](BUG_SKINNING_DESCRIPTOR_POOL_LEAK.md) | Why GPU skinning fell back to the CPU the moment you added an object — and why Vulkan RT lost it at the same time: destroyRasterMesh dropped the skinning descriptor set instead of freeing it, and raster + RT share one fixed pool. Also: the RT path is slower by construction (BLAS refit), and the `dispatchSkinningAll` batching it advertised never existed |
| [BUG_TIMELINE_PLAYBACK_LOCKED_TO_RENDER_RATE.md](BUG_TIMELINE_PLAYBACK_LOCKED_TO_RENDER_RATE.md) | Why the timeline played at whatever speed the viewport computed: the 1-frame cap banked a time debt it could never repay, so playback pinned itself to the render rate permanently. The cap was the deliberate part; the carry underneath it was the bug |
| [simulation_physics_foundation_plan.md](simulation_physics_foundation_plan.md) | Field-first physics stack direction |
| [IPC_SECURITY_PERFORMANCE.md](IPC_SECURITY_PERFORMANCE.md) | IPC transport and local security model |
| [AGENT_RUNTIME_PHASE4_REVIEW.md](AGENT_RUNTIME_PHASE4_REVIEW.md) | Review of the agent runtime + discovery layer before Phase 4 — what is real, what only looks real |
| [REMOTE_IPC_GATEWAY.md](REMOTE_IPC_GATEWAY.md) | Remote gateway boundary (see also `docs/remote_ipc_gateway_openapi.yaml`) |
| [TEMPLATE_REGISTRY_API.md](TEMPLATE_REGISTRY_API.md) | Template Registry script/IPC surface |
| [INTERIOR_VOLUME.md](INTERIOR_VOLUME.md) | Procedural interior appearance model |
| [VULKAN_GAS_FLUID_LAYERING.md](VULKAN_GAS_FLUID_LAYERING.md) | Coincident gas + fluid surface layering |
| [VOLUME_BOX_REENTRY_POSTMORTEM.md](VOLUME_BOX_REENTRY_POSTMORTEM.md) | Volume-box re-entry: black band + cost explosion, and why a distance epsilon cannot fix it |
| [VULKAN_VOLUME_TEMPORAL_INSTRUMENTATION.md](VULKAN_VOLUME_TEMPORAL_INSTRUMENTATION.md) | Volume temporal stability and instrumentation |
| [FOLIAGE_WIND_PIPELINE.md](FOLIAGE_WIND_PIPELINE.md) | Foliage wind runtime modes |
| [terrain_river_lake_snow_notes.md](terrain_river_lake_snow_notes.md) | Snowmelt water budget and river visibility |
| [fire_burns_down_a_structure_recipe.md](fire_burns_down_a_structure_recipe.md) | End-to-end scenario recipe, built from panels only |
| [raytrophi_manifesto_and_plan.md](raytrophi_manifesto_and_plan.md) | Project direction manifesto — the "world kernel" idea |
| [TERRAIN_EROSION_FLUVIAL_REWRITE.md](TERRAIN_EROSION_FLUVIAL_REWRITE.md) | Landscape-evolution cycle: drainage-area feedback, lake spill and sediment transport — live-tested and closed 2026-08-23 |
| [TERRAIN_FLAT_DRAINAGE.md](TERRAIN_FLAT_DRAINAGE.md) | Why flat and pit-floor channels came out straight and angular — a filled flat has no gradient, so the conditioning ladder WAS the drainage pattern, and a flat could never incise to change it |
| [TERRAIN_DEPOSITION_MODEL.md](TERRAIN_DEPOSITION_MODEL.md) | Why the LEM transport pass was kept over the redundant channel carvers, and the avulsion + alluvial-spreading passes added on top of it |

## Closed — postmortems and audits (ARSIV)

| Document | What it records |
|---|---|
| [FAZ3_DEVIR_NOTU.md](FAZ3_DEVIR_NOTU.md) | **ARCHIVE — Faz 3 is done.** Assimp was removed on 2026-09-06 after the user verified FBX, GLB and OBJ through the direct readers. §3 is still live reference: the traps already paid for when writing an importer here. |
| [UFBX_INCREMENT1.md](UFBX_INCREMENT1.md) | **ARCHIVE.** Static FBX reader and its dispatch. Superseded by increments 2 (skinning + animation) and 3 (OBJ + Assimp removal), both user-verified. |
| [IMPORT_GLTF_CHECKS_ARCHIVE.md](IMPORT_GLTF_CHECKS_ARCHIVE.md) | glTF import/export acceptance history through batch 9; §7/§8/§9(a) verified, §9(b) shared geometry storage remains a separate optimization. |
| [GRANULAR_STABILITY_POSTMORTEM.md](GRANULAR_STABILITY_POSTMORTEM.md) | Granular blow-up at low Young modulus — four root causes, and the CPU/Vulkan stage contract |
| [VULKAN_HAIR_PAUSE_DOUBLE_FREE_POSTMORTEM.md](VULKAN_HAIR_PAUSE_DOUBLE_FREE_POSTMORTEM.md) | Double-free on the second pause/play, Vulkan RT hair |
| [VULKAN_PARTICLE_PRESET_PAUSE_TDR.md](VULKAN_PARTICLE_PRESET_PAUSE_TDR.md) | Particle preset pause/resume TDR |
| [VOLUME_RAYMARCH_FOUNDATION_AUDIT.md](VOLUME_RAYMARCH_FOUNDATION_AUDIT.md) | Three-backend volume parity audit, 2026-07-23 |
| [fluid_material_coordinate_quality_notes.md](fluid_material_coordinate_quality_notes.md) | Fluid UVW quality — why the remaining difference is the method, not a bug |
| [oidn_optimization_report.md](oidn_optimization_report.md) | OIDN denoiser CPU fallback and CUDA path |

## Proposed, not implemented (TASLAK)

| Document | Area |
|---|---|
| [REALTIME_RENDERER_ROADMAP.md](REALTIME_RENDERER_ROADMAP.md) | Vulkan realtime viewport: product contract, frame graph and bounded quality profiles. **Separate Realtime mode was removed 2026-08-31** — the analysis holds, the ORDER was wrong; read the decision section first. Raster speed-up (presentation copies, GPU-driven culling) comes before scene lighting, which lands as an option in material preview. **2026-09-01**: carries the measured answer to "continuous LOD or scene lighting" — reaching pathtrace speed is the wrong target (RT cost is ~independent of instance count, raster's is linear), there is no mesh decimator in this repo, and the dominant cause of "crude proxy" was that proxies never reached the material pipeline at all. **2026-09-01**: material preview gained a `scene` lighting preset that reads the SAME light buffer the renderer reads (no second light list, so the two cannot disagree about which lights exist); shadows and world ambient are still absent and are reported as values rather than left to be inferred |
| [TERRAIN_ROAD_NETWORK_ROADMAP.md](TERRAIN_ROAD_NETWORK_ROADMAP.md) | Road/path üst katmanı: Faz 3.6 Curve to Mask + Road Carve üzerine profil, ölçülebilir field publication, foliage, crossing ve otomatik routing |
| [TERRAIN_ROAD_BUILD_CHECKS.md](TERRAIN_ROAD_BUILD_CHECKS.md) | Curve Input, Curve to Mask ve Road Carve faz-sonu derleme/canlı IPC doğrulama paketi |
| [AGENT_PIPELINE_ARCHITECTURE.md](AGENT_PIPELINE_ARCHITECTURE.md) | Multi-instance agent/human production pipeline — which decisions are irreversible |
| [photon_caustics_plan.md](photon_caustics_plan.md) | Photon caustic pass, Vulkan RT first |
| [stylize_mode_architecture.md](stylize_mode_architecture.md) | Stylize mode as an AOV-driven post layer |
| [stylize_pipeline_notes.md](stylize_pipeline_notes.md) | Stylize working notes |
| [mesh_paint_fill_layer_smart_masks_report.md](mesh_paint_fill_layer_smart_masks_report.md) | Non-destructive fill layers and smart masks |
| [mesh_paint_tablet_support_plan.md](mesh_paint_tablet_support_plan.md) | Tablet/pen pressure input |

## Rewritten every batch (CANLI)

- [NEXT_BUILD_CHECKS.md](NEXT_BUILD_CHECKS.md) — the ordered check list for the
  most recent batch of work. It is overwritten each time, so it describes only
  what is waiting to be verified right now.

---

Related material outside this folder:

- [`docs/`](../) — the user-facing HTML manual
- [`docs/template_hub/`](../template_hub/) — template manifest schema and examples
- [`CLAUDE.md`](../../CLAUDE.md) / [`AGENTS.md`](../../AGENTS.md) — working rules for agents and contributors
