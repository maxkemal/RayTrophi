# Acoustic surface debug (view 15)

Render Settings > Acoustic Energy (six-band) now runs a dedicated GPU compute
visualizer over the existing first-hit position/normal AOVs. Beauty continues
accumulating underneath; source/settings changes refresh presentation without
resetting beauty. The old solid-proxy shader is replaced.

- Selected source or **Sum sources (max 4)**: incoherent authored energy sum;
  spatial manual sources, including stopped sources. Gain/range/cone apply.
- Band -1 averages six final band energies; 0..5 selects 125, 250, 500, 1000,
  2000, 4000 Hz. Transmission multiplies once per canonical object along a path.
- **First-bounce samples 0..2** adds a cosine-sampled, Lambertian single diffuse
  reflection using diffuse share (1 - absorption - transmission) * scattering.
  The remaining specular share uses finite-triangle image-source candidates,
  validating both legs and using total reflected path length for source falloff.
  Repeated coplanar candidates cannot count the same source path twice.
  Candidate discovery is incomplete, not a complete specular-path solver.
  This is a noisy, deterministic spatial estimator without temporal accumulation.
  It adds surface energy only: no delayed audible echo, refraction or phase-resolved pressure,
  diffraction, phase interference, source-spectrum analysis or calibrated SPL.
- Receiving-surface incidence cosine applies. Purple/yellow spans -60..0 dB
  relative authored energy. Magenta means missing material/unsupported intersection
  or exhausted ray budget; grey means no source or unavailable/stale scene data.

CPU geometry is never rebuilt: canonical flat-object coverage and current GPU AS
metadata are reused. At most 4 sources, 32 unique barriers and 256 candidates per
ray; reflected probes stop at 10000 scene units. At two reflection samples, at most
30 ray queries/pixel. Ray-query support is required only by the optional compute
pipeline. Two small mapped frame-slot uploads reuse render fences; no new waits,
submissions or readback. Geometry metadata refresh is throttled to 250 ms and
revalidated against the active AS before dispatch. Normal rendering modes add no
surface queries. Gas/SDF/transient splats are excluded by the authored-solid mask.

Shared UI / Python / IPC controls:

```python
rt.audio.workspace_update(acoustic_all_sources=True, acoustic_band=2,
                          acoustic_reflection_samples=1)
rt.render.set_settings(debug_view=15)
state = rt.audio.acoustics_debug(id="audio_source_2")["surface_visualizer"]
```

Workspace validation is atomic: boolean source-sum, integer band -1..5, integer
reflection samples 0..2. Settings are session-only. The read query reports model,
budgets, settings and backend error without dispatching work.

User build: rebuild Studio, raygen and new acoustic_surface.comp. Check a known
six-band wall; compare selected vs summed sources, band changes, reflection 0 vs 1,
unknown-material marking and source motion after convergence. No builds/live tests
were run by the agent; only static source checks are performed in this workspace.

## Optional medium / interface probe

Substance fields `acoustic_medium_density` (effective kg/m3), `acoustic_medium_speed`
(m/s), `acoustic_medium_loss_125` .. `_4000` (dB/m) and `acoustic_medium_defined`
are shared by the substance UI, Python/IPC, inheritance and project serialization.
Built-ins remain undefined. Configure density/speed before enabling; invalid
profiles reject atomically. Density is an acoustic effective property, not an
implicit edit to simulation/thermal density.

Render Settings > Acoustic interface probe calls the same read-only service as
`rt.audio.acoustics_boundary(from_substance="Medium A", to_substance="Medium B",
incidence_degrees=30, transmitted_distance_meters=1)`. This uses lossless planar
fluid-interface Snell/impedance conditions, complex pressure coefficients, normal
energy flux ratios and post-interface band attenuation/delay/phase. It handles
total internal reflection without fabricating a propagating ray. Coefficients
are nondispersive at the interface; only bulk attenuation varies by band.
No elastic shear conversion, diffraction, layered-wall resonance or GPU refracted
path construction is claimed. GPU maps continue using authored band responses.
This is a testable physical-interface kernel, not a complete wave-field solver.
Reference: [intensity and phase conditions](https://doc.comsol.com/6.3/doc/com.comsol.help.aco/aco_ug_geometrical_aco.11.46.html).
