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
| [API_SCRIPTING_ROADMAP.md](API_SCRIPTING_ROADMAP.md) | Scripting/IPC API waves and their status table |
| [PHYSICS_VALIDATION.md](PHYSICS_VALIDATION.md) | Is the solver RIGHT, not merely running — analytical cases, and why a script test is blind to the frame loop |
| [IPC_TEST_CHANNEL.md](IPC_TEST_CHANNEL.md) | The second test channel: drives the app from outside so it can see the frame loop — and the first thing it found is that physics.step is reverted 100% before you can read it |
| [SIMULATION_NODE_CONCEPTUAL_MODEL.md](SIMULATION_NODE_CONCEPTUAL_MODEL.md) | What the simulation node layer IS — read this before the object-model report |
| [SIMULATION_NODE_OBJECT_MODEL.md](SIMULATION_NODE_OBJECT_MODEL.md) | Scope-based object model for the simulation node layer — scopes + World thermal landed (steps 1-4), steps 5-6 open |
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
| [TERRAIN_PERF_HANDOFF.md](TERRAIN_PERF_HANDOFF.md) | Terrain build cost: what was measured, what is verified, what is NOT, and the ordered next steps — start here before touching terrain performance |
| [TERRAIN_SATMAP_COLORIZER_ROADMAP.md](TERRAIN_SATMAP_COLORIZER_ROADMAP.md) | Gaea-style SatMap macro colour over the existing 4-layer splat blend — why it must modulate rather than replace the albedo, and the Faz 0 split of field / mesh / paint resolution that has to land first |
| [refactoring_implementation_plan.md](refactoring_implementation_plan.md) | DNA / flat-SoA data-oriented core migration |
| [flat_mesh_facade_audit.md](flat_mesh_facade_audit.md) | Remaining `Triangle` facade call sites |

## Rules and architecture in force (REFERANS)

| Document | Area |
|---|---|
| [DEVELOPMENT_PRINCIPLES.md](DEVELOPMENT_PRINCIPLES.md) | Binding engineering rules (file size, ownership, splitting) |
| [VULKAN_SIMULATION_RESET_SAFETY.md](VULKAN_SIMULATION_RESET_SAFETY.md) | Reset/rewind safety rule — TDR avoidance |
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

## Closed — postmortems and audits (ARSIV)

| Document | What it records |
|---|---|
| [GRANULAR_STABILITY_POSTMORTEM.md](GRANULAR_STABILITY_POSTMORTEM.md) | Granular blow-up at low Young modulus — four root causes, and the CPU/Vulkan stage contract |
| [VULKAN_HAIR_PAUSE_DOUBLE_FREE_POSTMORTEM.md](VULKAN_HAIR_PAUSE_DOUBLE_FREE_POSTMORTEM.md) | Double-free on the second pause/play, Vulkan RT hair |
| [VULKAN_PARTICLE_PRESET_PAUSE_TDR.md](VULKAN_PARTICLE_PRESET_PAUSE_TDR.md) | Particle preset pause/resume TDR |
| [VOLUME_RAYMARCH_FOUNDATION_AUDIT.md](VOLUME_RAYMARCH_FOUNDATION_AUDIT.md) | Three-backend volume parity audit, 2026-07-23 |
| [fluid_material_coordinate_quality_notes.md](fluid_material_coordinate_quality_notes.md) | Fluid UVW quality — why the remaining difference is the method, not a bug |
| [oidn_optimization_report.md](oidn_optimization_report.md) | OIDN denoiser CPU fallback and CUDA path |

## Proposed, not implemented (TASLAK)

| Document | Area |
|---|---|
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
