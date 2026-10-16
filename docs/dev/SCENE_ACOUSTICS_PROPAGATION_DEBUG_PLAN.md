# Acoustic relevance, proxies and debug views

2026-10-16 discussion checkpoint after distant-source measurements. Planning only;
no propagation/proxy implementation is claimed by this document. The first selected-source
debug slice is now implemented; see [diagnostics contract](SCENE_ACOUSTICS_DEBUG.md).

## Relevance comes before detailed intersection

Rendering visibility and acoustic influence are separate. Existing direct-path
queries already use a finite source-listener segment: geometry beyond its endpoints
does not obstruct that segment. A surface behind the camera or source can still
contribute through a reflected/diffracted path, which needs a separate path model.
Do not use camera frustum visibility as acoustic relevance.

Skip sources outside an explicitly hard audible range when that range is the
authored model. For other sources, use band energy, source directivity, accumulated
path length, delay horizon, material losses and a declared audibility threshold.
Do not derive a universal hard acoustic radius from distance alone.

For one-bounce candidates, a path-length limit L bounds possible reflection points
by |source-X| + |X-listener| <= L. Source/listener spheres and conservative bounds
can cheaply reject candidates before exact tests; they must not reject an object
whose bounds overlap that region merely because its centre lies outside it.

## Bounds first, optional SDF, exact geometry where required

Reuse existing canonical object bounds. World AABBs are a cheap broad phase;
local bounds with inverse object transform give a tighter OBB-style test for
rotated geometry. A bounds intersection does not establish a physical wall hit.
Bounds of a building can cover doors, windows, courtyards and disconnected parts.

An explicit acoustic proxy option may treat a coarse box as the object, with
visible approximation/error semantics and a user-controlled scope. Do not silently
replace accepted triangle geometry with boxes. An optional hierarchy is:

1. Source/path energy, range and time-horizon rejection.
2. Conservative bounds candidate testing using existing scene spatial organization.
3. Selected representation: existing valid SDF, explicit box proxy or flat GPU mesh.
4. Per-path spectral response, timing, visibility and energy accounting.

Hardware AS traversal already performs internal bounding-box tests. Adding a full
CPU scan or a duplicate per-triangle bbox structure can cost more than a direct
GPU segment query. Benchmark the extra hierarchy for reflection sampling/probes;
do not assume it helps the current four direct segments.

Keep triangle -> instance -> canonical object -> Substance/proxy bindings. Existing
SDF reuse can provide local signed distance/gradient and support interior/exterior
or approximate thickness work without rebuilding another representation. It does
not establish source -> surface -> listener visibility or solve reflections with
zero additional queries. Locating an exit still requires bounded SDF sampling or
other geometry work; nonuniform transforms and distance units need explicit rules.

Current ParticleColliderDesc exposes cached grid, origin/extents/dimensions and
bake serial/busy guards. Its header explicitly states automatic recooking is
disabled in the legacy melt-refresh field. Validate geometry/pose/bake ownership,
grid coverage, sign and resolution before acoustic reuse; thin walls/openings may
be below grid resolution. Domain-local grids and kinematic proxy stamps are not
automatically complete object SDFs. Never enable simulation or cook a missing SDF
silently. Flat TriangleMesh/DNA remains geometry authority; proxy/SDF data is derived.

## Debugging must identify the underlying model

Current source slice: Render Settings surface field (solid geometry proxy), documented
in [the visualizer contract](SCENE_ACOUSTICS_DEBUG.md). The earlier screen-space card
is removed. The following remains the target for a richer solver-backed overlay,
in parallel with relevance/budget metadata. Show source/listener, audible bounds,
accepted/rejected paths, proxy type, band energy, path length/delay, unknown material,
stale results and query/transfer budgets. Draw GPU-resident bounded debug buffers
with the same submission/fence ownership; CPU reads summaries only. Debug defaults
off, uses selected sources/regions and does not add a permanent UI shelf.

Then add sampled per-band energy heatmaps on explicit slices/probe grids, followed
by verified early-reflection paths and arrival-time displays. A drawn expanding
sphere is an illustrative propagation marker, not a solved pressure wave. Heatmaps
must state sampled/interpolated/stale coverage and distinguish energy from pressure.
Do not draw reflection/diffraction/refraction paths as physical results before
their solver produces them. Refraction requires a spatial sound-speed model;
edge diffraction and surface reflection are separate mechanisms.

Six energy fractions do not retain phase. True coherent interference requires
complex pressure or signed time-domain responses, path delay and phase/boundary
response; unrelated sources do not automatically form stable interference fringes.
Plan a hybrid: geometric paths for suitable scales, bounded wave/hybrid treatment
for low-frequency diffraction/interference. SDF geometry alone is not a wave solver.

All controls, budgets, diagnostic results and proxy authoring must share service
logic through UI, Python and IPC, with validation/project/history semantics where
persistent. Deliver final regression/reference scenes and numerical/listening tests
with the relevant source slice. This plan does not close any existing acceptance gate.
