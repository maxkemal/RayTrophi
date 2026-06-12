# RayTrophi Stylize Mode Architecture

Stylize Mode is an optional art-direction layer for RayTrophi. It should not replace the existing renderer, terrain sculpt, mesh paint, foliage scatter, VDB, force field, or animation systems. It reads their outputs and adds a controllable visual language on top.

## Goals

- Let artists restyle a scene quickly without depending on expensive closed tools.
- Keep the base realistic/path-traced scene intact when Stylize Mode is disabled.
- Use profiles for fast results, with enough controls for serious art direction.
- Connect style to world behavior over time: sky motion, foliage wind, volume drift, and force fields should be able to influence painterly motion later.

## First Layer

- `Stylize::StylizeModeState` owns the active profile and controls.
- `Renderer::stylizeMode` exposes this state without changing existing render behavior.
- `SceneUI::drawStylizePanel` gives artists a dedicated control surface.
- CPU render reuses the OIDN auxiliary path as Stylize AOV data:
  - color
  - albedo
  - normal
  - depth
  - material id
  - derived edge strength from depth/normal/material discontinuities
- Painterly material response should use smooth, normal-guided brush fields rather than block/hash grain. Large brush scales must read as broad strokes, not droplets or pixel islands.
- The brush layer is a discrete daub field (`StylizeCore::brushStrokeField`): oriented elliptical stamps on a jittered grid in stroke space, composited by per-daub random priority so overlapping daubs keep crisp painter-ordered boundaries. Each daub carries its own tonal offset, bristle streaks, and a signed impasto rim; `Dry Brush` removes daubs (bare canvas sinks toward the shadow palette). Two layers (base + finer detail) run per pixel; the field is pure ALU + integer-hash noise, identical across CPU/CUDA (StylizeCore.h) and Vulkan (stylize.comp).
- Palette influence is independent from brush response. At `0.0`, the current render/material colors are only pushed through brush strokes; at higher values, the active style palette restyles the surface.
- Material color preservation protects small props and material variations from collapsing into the same palette value. Style palettes should tint material identity, not erase it by default.
- The Wet Oil Model reuses the mesh paint brush language (`Body`, `Load`, `Pickup`, `Deposit`, `Buildup`) as a cheap screen-space stroke model. It does not run the full UV wet simulation in the stylize pass, so the cost stays close to a few coherent noise and blend operations per pixel.

## Future Render Hooks

1. Tune CPU AOV-driven profiles so painterly modes read albedo/normal/depth instead of final color only.
2. GPU tonemap/post constants for realtime preview parity.
3. Sky adapter that maps style sky controls to gradient, painterly cloud, haze, and sun glow behavior.
4. Material adapter that reads mesh paint and material layers for pigment, dry brush, and stroke scale.
5. World adapters for terrain strokes, foliage cluster simplification, VDB grain, and force-field-driven style motion.

## Profiles

Initial profiles are:

- Painterly Oil
- Gouache
- Ink + Wash
- Graphic Toon
- Clay / Maquette
- Dreamy Sunset

These profiles are starting points, not hard modes. Artists should be able to override every important value and eventually save project/local presets.
