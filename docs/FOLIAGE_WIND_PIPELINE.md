# Foliage Wind Pipeline

## Runtime Modes

- `CPU fallback`
  - Always available.
  - Updates instance transforms safely for every foliage group.
  - Supports per-source wind profiles.

- `CUDA deform`
  - Enabled only when all active foliage groups share one safe global wind profile.
  - Disabled automatically when source-specific wind profiles are active or when populated groups have wind disabled.
  - CPU fallback remains active even when CUDA deform is disabled.

## Wind Profile Rules

### Group profile

Stored on `InstanceGroup::WindSettings`.

- `speed`
- `strength`
- `direction`
- `turbulence`
- `wave_size`
- `use_source_profiles`
- `allow_gpu_deform`

### Source profile

Stored on `ScatterSource::SourceSettings`.

- `wind_strength_scale`
- `wind_speed_scale`
- `wind_turbulence_scale`
- `wind_bend_limit_scale`
- `wind_phase_offset`

These values let one foliage layer mix different plant behaviors safely on CPU.

## Asset Authoring Standard

### Pivot

- Trees and bushes should use a bottom-center pivot.
- Grass patches should use a ground-contact pivot.

### Scale

- Export in real-world scale when possible.
- Keep the trunk base close to local `Y = 0`.

### Vertex Colors

Recommended convention for future CUDA/vertex deform work:

- `R`: stiffness mask
  - `0.0` fully flexible
  - `1.0` rigid trunk/branch
- `G`: branch secondary mask
  - reserved for secondary bend or gust response
- `B`: leaf flutter mask
  - `0.0` branch only
  - `1.0` leaf-heavy flutter

### Export Notes

- Preferred runtime format: `GLB`
- Recommended content path:
  - `PlantFactory -> FBX -> Blender cleanup -> GLB`

## Safe CUDA Conditions

CUDA deform should be considered safe only when:

- all enabled foliage groups use the same group wind profile
- `allow_gpu_deform` is enabled for those groups
- no populated foliage group is excluded from wind while others use CUDA deform
- source-specific wind profile overrides are not active

If any of these conditions are violated, the engine should stay on CPU fallback.
