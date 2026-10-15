# Scene acoustics: shared-material direction

2026-10-15: user extended the audio target to object interaction, obstruction,
transmission and physical echoes. Shared object materials precede propagation;
see OBJECT_SUBSTANCE_BINDINGS.md for the implemented source checkpoint. Test
scripts remain deferred to the final acceptance stage by user instruction.

Latest source checkpoint: [direct-path diagnostic](SCENE_ACOUSTICS_TRACE.md) and
[session listening preview](SCENE_ACOUSTICS_PREVIEW.md) connect bounded flat-mesh
traces to source volume through UI/Python/IPC. The first scene acceptance checklist
is ready for the user's build. Object-barrier mode counts a named object once as
an explicit assembly approximation; closed-wall thickness, spectral filtering,
background solving and reflections remain incomplete. The 2026-10-16 external IPC/native PCM acceptance passed for the direct-path
preview; this does not validate the remaining physics stages.

1. **Shared material authority:** object -> SubstanceLibrary profile and explicit
   overrides; domain-independent binding, MSF compatibility, project/history and
   UI/Python/IPC. Acoustic band data and energy-conserving response are in source.
   No measured built-in acoustic table or live acceptance is claimed yet.
2. **Geometric direct paths:** immutable flat TriangleMesh/DNA geometry snapshots,
   bounded acceleration structure and source/listener segment queries. Report
   blockers, material provenance, unknown responses and path length/delay. Treat
   thin surfaces and closed/layered walls explicitly; do not multiply entry/exit
   losses as if they were independent walls. Moving/deforming geometry needs
   revision ownership and bounded updates before it can be live authority.
3. **Early reflections:** validated source -> surface -> listener paths, visibility
   on every segment, band response, reflection order/path budgets and delay. Begin
   with specular paths; diffuse scattering needs energy accounting and deterministic
   sampling. A material name alone cannot produce a room response.
4. **Audible propagation:** shared filtered direct path and delayed reflection /
   impulse-response processing. Keep render/analysis work outside the device
   callback; smoothly replace responses with cancellation/latest-result ownership.
   UI, Python, IPC and offline export must use the same acoustic model and timing.
5. **Broader physical coverage:** air absorption and environment-derived sound
   speed, source directivity/listener orientation, motion/Doppler, diffraction at
   openings/edges, multilayer transmission and temperature/wetness-dependent
   materials. Geometric reflection rays are not a complete wave solver; explicitly
   identify approximations and the frequency/scene scales they support.
6. **Late reverberation and closure:** bounded diffuse/multiple-bounce response,
   deterministic project/export, reference scenes, physical-unit calibration,
   listening/performance budgets and the final combined script/manual acceptance.

No per-face Triangle collection may become geometry authority. Acoustic state must
not silently enable domain telemetry or simulation, infer physical data from PBR
appearance, duplicate SubstanceLibrary records or add a permanent UI shelf.
