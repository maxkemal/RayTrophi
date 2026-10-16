# Acoustic surface render visualizer

The acoustic debug view lives in Render Settings beside Normal, Depth and Bounce
Count. The previous screen-space information card has been removed.

Choose **Acoustic Energy (solid proxy)** (view 15), then choose an **Acoustic
source** in the same panel. This selects the source through the ordinary shared
`audio.select` operation. Selecting a different kind of object disables the source
field until an eligible source is selected again. This first version visualizes
one spatial source with a manual reaction driver, on Vulkan RT only. Playback and
acoustic listener-preview activation are optional.

## Meaning of the image

At each visible first-hit surface, the shader evaluates the authored source gain,
inverse-distance-clamped attenuation and cone gain. The amplitude is squared to
obtain relative energy. An extra finite source-to-surface ray checks active-AS
authored solids as **solid blockers**; optical transparency is deliberately not reused
as acoustic transmission. No camera-frustum culling is applied to that ray.

- Purple to yellow represents -60 to 0 dB relative to unit amplitude, after Debug
  Exposure scales energy. This is a normalized authored-level field, not measured
  PCM loudness or calibrated sound pressure level.
- Black means blocked/zero energy or no primary surface. Grey surfaces mean no
  eligible source. Overlay Beauty mixes the result with the ordinary render.
- Source max distance clamps attenuation; it does not silently become a hard
  cutoff. Source gain and directivity apply; bus/master mixing and native playback
  meters do not drive the field.

This is a direct-field **solid geometry proxy**, distinct from the listener
preview's six-band material-transmission solver. It does not yet show material
transmission, reflections, diffraction, interference, pressure waves, band-specific
energy or SDF/volume occlusion. Hidden/excluded geometry absent from the active
render AS cannot block this view. Surface values are point samples of the relative
field, not deposited acoustic irradiance. The query uses the existing authored-solid
visibility bit (0x01); ordinary procedural solids can block, while transient splats
(0x04), gas/fog (0x02) and SurfaceSDF (0x08) are excluded.

The existing progressive rendering and display post chain present this view.
Camera depth of field, antialiasing and display post controls can therefore affect
its appearance, as with the existing raygen debug modes.

## Cost and synchronization

Off and other debug modes execute no additional acoustic surface rays. View 15
adds at most one finite, terminate-on-first-hit ray per visible surface sample.
It reuses the active TLAS and existing renderer submission/fences; no new AS,
collider cooking, CPU BVH, per-pixel CPU download, fence or queue submission is
introduced. It can cost significantly more than the four listener-preview rays.

One 56-byte selected-source snapshot is appended to the existing camera push
constants. The scalar host/shader ABI is 252 bytes inside the existing 256-byte
RT pipeline range. No new descriptor, storage buffer or shader binary is required;
`raygen.rgen` must be rebuilt by the normal user shader build because it now includes
`acoustic_surface_debug.glsl`.

Source changes restart the debug accumulation without synchronous GPU image
clears: frame zero ignores old adaptive statistics and overwrites all pixels/AOVs
through the existing frame path. Normal mode/camera changes retain the renderer's
existing reset behavior.

## Shared Python and IPC

```python
rt.audio.select(id="audio_source_2")
rt.render.set_settings(debug_view=15)
state = rt.audio.acoustics_debug(id="audio_source_2")
rt.render.set_settings(debug_view=0)
```

External IPC uses the same `audio.select`, `render.set_settings`,
`render.get_settings` and `audio.acoustics_debug` methods. Both render panels call
the shared render settings service. The read-only debug query requires exactly one
string source ID, an existing source and a camera; invalid inputs do not mutate
state. Its `surface_visualizer` object reports active mode, whether the queried
source is selected/eligible, proxy model, relative dB range and zero extra readback.
The existing `preview` entry remains a separate listener-path result.
`result_pose_matches` checks its endpoint positions only, not complete scene
freshness. The old overlay workspace toggle is removed.

Render settings accept debug views 0..15; values outside that range fail before
mutating bounce settings or debug state. Mode 15 is named `acoustic_surface` in
`render.get_settings`. Render writes keep SceneWrite permissions; the debug query
requires Read and no file access.

## User-build acceptance

1. Build Studio and the normal RT shaders, including the changed raygen shader.
2. In Vulkan RT, choose view 15 in either Render Settings entry point. Select a
   spatial manual source from its source dropdown. Verify that the surface image
   changes rather than adding a viewport information card.
3. Put a wall between source and a visible receiving surface. Verify a black
   obstruction region; move the wall aside and verify the field returns. Optical
   glass is also solid in this explicitly approximate mode.
4. Move the source or its parent, alter gain/range/cone, and verify the image
   updates after convergence. Gain zero must give black; no eligible selection
   must give grey surfaces. Stopped playback must still show the authored field.
5. Compare at two distances with unit gain, rolloff 1 and distances between min/max:
   doubling distance gives approximately -6.02 dB relative energy. Beyond max,
   attenuation clamps instead of becoming silent.
6. Toggle Normal, Bounce Count, view 15 and Off; verify original views and beauty
   return. Verify Adaptive Sampling does not retain a previous source's pixels.
7. Repeat source selection and mode switching through Python/external IPC. The
   existing `rt_test_acoustics_preview_ipc.py` covers read contract, mode name,
   invalid mode 16 rejection and mode restoration on its disposable audio-empty
   test scene; it does not certify rendered pixel colors.

2026-10-16 user-build checkpoint: the user confirmed that the surface heatmap
works, then confirmed the source-selection focus fix. Automated rendered pixel
acceptance and surface-debug GPU performance measurement remain open. The agent
performed static checks only and did not compile or launch Studio.
