# Interior Volume — a procedural interior appearance model

*Technical note, RayTrophi Studio, July 2026. Status: shipped in the Vulkan RT backend; CPU/OptiX carry only the base depth-absorption layer.*

---

## What it is

Interior Volume gives a material an **embedded interior** — depth-tinted resin or
see-through glass filled with dust clouds, dirt specks, air bubbles and colored
glass shards — **without any interior geometry, volume data, or extra scene
rays**. The entire interior is synthesized analytically inside the closest-hit
shader at the first surface crossing, from hash-derived procedural fields.

Two modes share one implementation:

| Mode | Trigger | Behaviour |
|---|---|---|
| **Resin coat** | `Interior Depth > 0` | refractive absorbing layer over an **opaque** base; the base albedo is read with a refraction-parallax UV offset |
| **Glass marble** | `Transmission > 0.5`, `Interior Depth = 0` | real see-through glass; the interior structure is sampled on the entry shell, then the ray refracts through normally |

## Why this is worth describing as a system

None of the ingredients are new; the combination and one property of it are.

- **Hypertexture** (Perlin & Hoffert 1989) established ray-marched procedural
  density inside objects. Our dust phase is bounded hypertexture with
  Beer-Lambert absorption.
- **Interior Mapping** (van Dongen 2008) established per-cell analytic fake
  interiors behind a surface, with zero geometry. Our speck lattice is a 3D
  generalization: one analytic primitive per hash cell.
- **Parallax (occlusion) mapping** is where the refracted base-albedo lookup
  comes from.
- **Aggregate appearance models** for granular media (e.g. Meng et al. 2015)
  established replacing thousands of explicit grains with a statistical proxy.
  That is the closest conceptual relative: what microfacet BSDFs do to
  microgeometry, this does to interior macro-inclusions.

The property we have not seen elsewhere in this combination: the fake interior
**participates in light transport**. The same evaluation runs for photon rays
in the caustics pass, so a marble's shard colors tint the photons crossing it —
the interior produces **stained-glass caustics** on receiving surfaces. Fake
interiors in the real-time tradition are camera-only and never feed
illumination.

## How it works

### Phase A — dust (stochastic, converging)

A 12-step march along the refracted interior ray, jittered per sample and
integrated by the renderer's normal progressive accumulation — i.e. a Monte
Carlo estimate of the density integral, converging with the same contract as
the rest of the path tracer.

- Extinction: `absorb *= exp(-dt · (ext_base + dust·k))`, dust from billowy
  two-frequency fbm shaped with `pow(x,2)` for sparse wispy cores.
- The dust is **visible**, not just darkening: a coverage-weighted color is
  mixed into the result (milky in-scatter approximation).
- Four selectable styles: **Nebula** (auto tint derived from the interior
  color and its hue-rotated complement), **Billow 2-color**, **Wispy streaks**
  (anisotropically stretched ridged filaments), **Paint swirl** (domain-warped
  fbm; the color field is warped by the same flow, so two user-chosen pigments
  fold into each other like ink stirred into water).

### Phase B — specks (deterministic, exact)

Inclusions must be *sharp*; sampling them with the dust march blurred them
(a speck was only found when a sample landed in its cell — hit/miss averaging
across accumulation produced translucent blobs). Phase B therefore walks the
speck lattice with a **3D voxel DDA**: every noise cell the ray crosses is
visited **in order**, independent of step counts.

- One candidate inclusion per cell; an existence hash gated by the amount
  knobs controls population density.
- **Dirt**: analytic ray-sphere intersection — the silhouette is exact and
  identical every pass. First hit terminates the walk (correct ordering
  against everything behind it). Shaded as a micro-sphere using the sampled
  NEE light direction.
- **Bubbles**: a fixed slice of the dirt population; bright rim where the ray
  grazes the shell.
- **Glass shards**: translucent color chips. They tint the transmittance
  behind them (stained glass) *and* carry an additive visible color body so
  they read over an opaque resin base. Two shapes: round chips, or elongated
  (~2.6×) ellipsoids intersected in squashed space with the surface normal
  quantized into a per-shard facet lattice — flat faces that flash as the
  light or the object turns. Palette: a base hue ± hashed spread, or a full
  rainbow.
- Transmittance at a speck's depth is reconstructed channelwise as
  `absorb_total^(t/t_end)` — consistent with Phase A without re-marching.

### Anchoring

The fields evaluate in **object space** by default (`gl_WorldToObjectEXT`;
the interior travels and rotates with the mesh — the sampled light direction
is rotated into the same frame so speck shading stays consistent). A **world**
option keeps the legacy behaviour deliberately: a moving object appears to
pass through a frozen medium.

## What is honestly approximate

This is an **appearance model**, not physically-based volumetric transport:

1. **Single entry interface.** The interior is sampled once, at the first
   crossing. Internal reflections do not see it again.
2. **No interior self-shadowing.** Specks are lit by the sampled NEE
   *direction* only — a shadow ray from inside an opaque-based resin would
   always self-occlude to black, so none is traced. Dense dust does not
   shadow the specks either.
3. **Reciprocity is broken** by the additive shard-glow term. Light going the
   other way does not see the inverse of it.
4. **Transmittance interpolation** (`absorb^(t/t_end)`) is exact only for
   homogeneous media; heterogeneous dust makes it an approximation.
5. **No multiple scattering.** Milkiness is approximated by whitening the
   attenuation, not by simulating it.
6. Speck spheres near a cell boundary can extend slightly into neighbouring
   cells; a ray clipping that overhang through the neighbour is missed.
7. In object-anchor mode, feature *density* stays fixed in world terms under
   non-unit object scale (positions anchor, sizes don't rescale).

None of these have been observed to matter visually at typical settings; they
are listed so nobody mistakes the model for ground truth.

## Cost model

- Everything runs **only on materials with the feature enabled** — no cost to
  the rest of the scene, no BVH growth, no memory beyond ~8 material floats.
- Phase A: 12 steps × (1–5 fbm evaluations depending on dust style). Paint
  swirl is the most expensive style (~5 fbm/step).
- Phase B: ≤48 DDA cells × 1 hash (empty cell) or ~2 hashes + one quadratic
  (occupied). Cheaper than the 27-hash worley lookups it replaced.
- Measured on the development scene (RTX-class GPU, 1080p): a full-screen
  marble with dust + shards renders interactively at the same order as plain
  glass; the feature has never been the frame-time bottleneck in our tests.
  (No formal benchmark yet — see roadmap.)

## Interaction with photon caustics

Photon rays execute the same closest-hit code, so:

- a marble's shards tint crossing photons → **colored caustics**;
- dense dust attenuates photons → milky marbles correctly produce weaker
  caustics;
- the resin *coat* mode is opaque-based, transmits nothing, and is therefore
  correctly **not** a caustic caster.

## Parameters

`Interior Depth`, `Interior Tint`, `Coat Gloss`, `Dust` (+ `Dust Style`,
`Dust Color A/B`), `Dirt Specks` (+ `Dirt Color`), `Glass Shards`
(+ `Shard Shape`, `Rainbow`/`Shard Hue`), `Inclusion Scale`, `Anchor`,
plus nine curated interior presets (Clear, Dusty Amber, Nebula, Galaxy Marble,
Cat's Eye, Crystal Geode, Ink in Water, Riverbed Epoxy, Champagne). All values
serialize to the scene JSON; legacy scenes load with identical defaults.

## Roadmap

Planned, in rough priority order — none of it promised on a date:

1. **Formal benchmark** — publish honest frame-time numbers for each dust
   style and speck density on a reference scene.
2. **OptiX / CPU port** — the material fields already travel through the
   shared structs; the march itself is Vulkan-only today.
3. **Interior on internal reflections** — re-sample the interior when a ray
   exits via total internal reflection (fixes limitation 1 for marbles).
4. **Cheap interior self-shadowing** — a second short DDA toward the light
   through the speck lattice only (no scene rays), so dense inclusions shade
   each other.
5. **Density from textures / VDB** — replace or modulate the procedural dust
   with an artist-authored 3D texture or a simulation field (shares the
   volumetric-caustics V3 machinery).
6. **Per-shard refraction sparkle** — one extra analytic bounce inside
   faceted crystals for internal fire.
7. **Animated interiors** — time-warped paint-swirl flow (needs an
   accumulation-aware strategy; naive animation would be averaged away).
8. **Energy audit** — bound the additive glow terms so the model can be
   toggled into a strictly energy-conserving mode.

## References (informal)

- K. Perlin, E. M. Hoffert, *Hypertexture*, SIGGRAPH 1989.
- J. van Dongen, *Interior Mapping*, CGI 2008.
- J. Meng, M. Papas, R. Habel, C. Dachsbacher, S. Marschner, M. Gross,
  W. Jarosz, *Multi-Scale Modeling and Rendering of Granular Materials*,
  SIGGRAPH 2015.
- T. Kaneko et al., *Detailed Shape Representation with Parallax Mapping*,
  ICAT 2001 (and the POM family that followed).
