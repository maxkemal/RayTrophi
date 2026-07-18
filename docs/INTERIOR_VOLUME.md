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
| **Resin coat** | `Interior Depth > 0`, coat lobe | refractive absorbing layer over an **opaque** base; the base albedo is read with a refraction-parallax UV offset |
| **Glass marble** | glass lobe | real see-through glass; the interior structure is sampled on the entry shell, then the ray refracts through normally; depth absorption (Beer-Lambert with the interior tint) is applied directly to the path throughput on interior segments |

When both `Interior Depth` and `Transmission` are non-zero, the lobe is chosen
**stochastically per sample** with probability `Transmission` — the estimator
converges to `t·marble + (1−t)·coat`, a continuous blend between translucent
stone and coated resin (no hard threshold; earlier builds switched at
`Transmission = 0.5`, which produced a visible seam).

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
  mixed into the result.
- The dust is **lit as a volume** (directional single scattering): each
  occupied step marches 3 short jittered steps *toward the sampled light*
  through the same density field, and weights its contribution by
  `transmittance · phase`. The phase is a fixed dual lobe — 65% forward HG
  (g = 0.55) + 35% isotropic — so backlit dust glows with the classic
  "silver lining" while side/back lighting never goes fully dark. Light
  absorbed on the way re-emerges partially as an energy-limited diffuse
  floor (the cheap stand-in for multiple scattering). One mechanism buys
  two behaviours: dense cores shadow **themselves** and shadow the **specks**
  suspended below them. The light transmittance is cached for two camera
  steps (the field is low-frequency at that scale) to halve the cost.
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
2. **Interior self-shadowing is heuristic, not traced.** No scene shadow
   rays (from inside an opaque-based resin they would always self-occlude to
   black). Instead: specks shadow each other via a short lattice DDA toward
   the light (dirt blocks, shards tint the shadow like stained glass), and
   the dust field shadows both itself and the specks via a 3-step light
   march through the density field. Both see only the sampled NEE
   *direction*, not the full light set.
3. **Reciprocity is broken** by the additive shard-glow and dust-glow
   (forward-scatter excess) terms. Light going the other way does not see
   the inverse of them.
4. **Transmittance interpolation** (`absorb^(t/t_end)`) is exact only for
   homogeneous media; heterogeneous dust makes it an approximation.
5. **Multiple scattering is approximated, not simulated.** Single scattering
   is directional (dual-lobe phase × marched light transmittance); the
   multiply-scattered remainder is a diffuse floor proportional to the light
   absorbed on the way — energy-limited, but with no real angular or spatial
   diffusion. Deep milky media remain SSS's domain, deliberately: a random
   walk here would rebuild SSS at higher cost.
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
  swirl is the most expensive style (~5 fbm/step). The lit-volume light
  march adds ≤6 cached light marches × 3 density evaluations, and only on
  steps that actually hold dust — empty steps skip it entirely.
- Phase B: ≤48 DDA cells × 1 hash (empty cell) or ~2 hashes + one quadratic
  (occupied). Cheaper than the 27-hash worley lookups it replaced.
- Measured on the development scene (RTX-class GPU, 1080p, July 2026) with
  a shader-bisect kit (interior march compiled out vs. in, speck self-shadow
  off vs. on, and the pre-feature closest-hit as baseline): the interior
  march is **within measurement noise of plain glass** — near-zero cost at
  `Interior Depth = 0` and no meaningful frame-time regression with dust and
  specks enabled. When subsurface scattering is enabled on the same material,
  **SSS dominates the frame time by a wide margin**; the interior march is
  not the bottleneck in any configuration we tested. These are qualitative
  bisect results, not a per-style ms/sample table — that table remains on
  the roadmap.

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

1. **Formal benchmark** — publish honest per-style ms/sample numbers on a
   reference scene. (A first bisect pass is done — see the cost model above;
   the per-style table is still owed.)
2. **OptiX / CPU port** — the material fields already travel through the
   shared structs; the march itself is Vulkan-only today.
3. **Interior on internal reflections** — re-sample the interior when a ray
   exits via total internal reflection (fixes limitation 1 for marbles).
4. ~~**Cheap interior self-shadowing**~~ — done: lattice DDA toward the
   light (specks shade each other, shards tint the shadow) plus a 3-step
   dust light march (dense dust shades itself and the specks).
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
