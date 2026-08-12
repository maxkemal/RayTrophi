/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          FractureGenerator.h
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 *
 * Voronoi pre-fracture (Faz 1 of the destruction pipeline).
 *
 * Given a closed mesh's world-space triangle soup, this scatters N "sites"
 * inside the mesh and cuts it by each site's Voronoi cell — the intersection of
 * the bisector half-spaces between that site and every other.
 *
 * It is geometry only: no physics, no scene mutation. The caller (SceneUI)
 * turns each FractureShard into a scene node + (later) a rigid body.
 *
 * ★ A shard's RENDER mesh and its COLLIDER are not the same thing. Exact-mode
 * shards can be concave, while the rigid-body path builds a Jolt ConvexHull, so
 * such a shard collides as its own hull. That is the intended arrangement — the
 * visible break keeps its cavities and its texturing, and the physics keeps the
 * cheap, robust shape — but it does mean a deliberately hollow shard does not
 * collide hollow. Switching those bodies to a mesh collider is a separate call.
 *
 * Two cutting modes, same Voronoi sites and the same half-spaces:
 *
 *   - CONVEX (default): clip the mesh's convex hull. Fast and always closed, but
 *     a hollow or concave asset fills in and every surface UV is lost.
 *   - EXACT (`FractureParams::exact_surface`): clip the SOURCE TRIANGLE SOUP.
 *     Cavities, concave profiles, UVs and material IDs survive, because the
 *     surviving surface IS the original surface. Cut cross-sections can be
 *     concave and multi-loop, so they are sealed by chaining the cut edges into
 *     loops, bridging holes into their outer loop, and ear-clipping the result.
 *     Needs a closed input; a cell whose cut will not seal falls back to that
 *     cell's convex form rather than emitting a shard with a hole in it.
 *
 * OpenVDB LevelSetFracture stays a separate opt-in mode for solid/organic
 * assets, where a resampled isosurface (and the UVs it destroys) is an
 * acceptable trade.
 *
 * Cell boundaries that lie on the original surface keep the hull faces; cell
 * boundaries created by a cut are flagged `interior` so a different inner
 * material can be assigned later.
 */

#pragma once

#include "Vec2.h"
#include "Vec3.h"
#include <vector>
#include <cstdint>

namespace RayTrophiSim {

// One world-space source triangle. Positions alone build the hull; the UVs and
// material are what EXACT clipping carries into the shards, and without them a
// shard cannot find the char mask that decided where it broke.
struct FractureInputTri {
    Vec3 a, b, c;
    Vec2 ua = Vec2(0.0f, 0.0f), ub = Vec2(0.0f, 0.0f), uc = Vec2(0.0f, 0.0f);
    uint16_t material = 0;
};

// One triangle of a generated shard (world space). The flat face normal is
// stored on all three corners (shards shade faceted, like real broken stone).
struct FractureShardTri {
    Vec3 a, b, c;
    Vec3 n;            // flat face normal (outward)
    bool interior;     // true: this face was created by a cut (candidate inner material)
    // Source UVs, interpolated along every cut. Meaningful on exterior faces;
    // on `interior` faces there is no source surface to inherit from and these
    // stay (0,0) — the caller must not let an interior face sample the char
    // mask, because a fresh break is not burnt.
    Vec2 ua = Vec2(0.0f, 0.0f), ub = Vec2(0.0f, 0.0f), uc = Vec2(0.0f, 0.0f);
    uint16_t material = 0;   // inherited from the source triangle it came from
};

// One shard: convex in hull mode, possibly concave in exact mode.
struct FractureShard {
    std::vector<FractureShardTri> tris;
    Vec3  centroid = Vec3(0.0f, 0.0f, 0.0f);  // world-space centre of mass
    float volume = 0.0f;                        // world-space volume
    // ★ STRUCTURAL CLUSTER INDEX — which group of shards this one shatters WITH.
    //
    // A fracture group is the unit the impulse consumer breaks, so putting every
    // shard of an object in one group means a blast anywhere detaches the entire
    // object at once. On a water tower that is the difference between "one leg
    // was blown off" and "the tower vanished", and no amount of cut quality
    // upstream can fix it — it is decided here, by the grouping.
    int   cluster = 0;
};

// A point on the object that has already taken thermal damage, used to steer
// where it breaks. `weight` is 0 (pristine) .. 1 (fully lost), typically
// 1 - integrity for one MSF element.
struct FractureDamageSample {
    Vec3  position = Vec3(0.0f, 0.0f, 0.0f);
    float weight = 0.0f;
};

enum class FracturePattern : int {
    Uniform = 0,          // sites scattered uniformly through the hull
    ImpactClustered = 1,  // sites densified near `impact_point` (finer near the hit)
    // ★ Sites densified where the object has actually BURNT.
    //
    // This is the pattern the whole material-transformation chain exists to
    // make possible: MSF knows, per surface element, how much mass that patch
    // has lost to pyrolysis, melting and transfer. Feeding that back as seed
    // density means a charred beam breaks along the char line instead of along
    // an arbitrary noise field — the burn and the fracture stop being two
    // effects that merely happen to the same object.
    ThermalWeakened = 2,
};

struct FractureParams {
    int      site_count = 12;        // number of shards requested (some may be culled)
    uint32_t seed = 1337u;
    FracturePattern pattern = FracturePattern::Uniform;
    Vec3     impact_point = Vec3(0.0f, 0.0f, 0.0f);  // ImpactClustered cluster centre (world)
    float    impact_radius = 0.5f;                    // cluster spread (world units)
    float    min_shard_volume_ratio = 1.0e-4f;        // drop shards smaller than this × hull volume
    // Number of structural clusters to partition the shards into. 1 reproduces
    // the old all-or-nothing behaviour; clamped to the surviving shard count.
    int      cluster_count = 1;
    // ThermalWeakened input. Empty (or all-zero weights) falls back to Uniform
    // rather than producing nothing — an object that has not burnt yet must
    // still be fracturable.
    std::vector<FractureDamageSample> damage_samples;
    // How strongly damage biases seeding. 0 = ignore damage entirely,
    // 1 = seed density proportional to damage.
    float    damage_bias = 1.0f;
    // ★ EXACT SURFACE CLIPPING: clip the SOURCE TRIANGLE SOUP by the bisector
    // half-spaces instead of clipping the hull.
    //
    // The hull path answers "what is the convex outside of this object", so a
    // cavity, a concave profile, and every UV on the surface are gone before the
    // first cut is made. Clipping the soup never asks what is inside: the
    // surface that survives IS the original surface, carrying its UVs and
    // material, and only the CUT faces are new.
    //
    // Costs more than the hull path and needs a CLOSED input to seal its cuts.
    // A cell whose cut cannot be sealed falls back to that cell's convex form
    // rather than emitting a shard with a hole in it.
    bool     exact_surface = false;
    // ★ REPLAY: use exactly these sites and scatter none.
    //
    // A project file cannot store shard geometry (there is no general geometry
    // section in .rtp — everything is re-imported or re-generated), so a saved
    // fracture is restored by CUTTING AGAIN. Re-running the scatter would not
    // reproduce it: the thermal pattern samples the MSF damage field as it stood
    // at authoring time, and that state is gone by the time the project reopens.
    // The sites are the smallest thing that pins the result down exactly, so
    // they are what gets saved.
    //
    // Taken VERBATIM — no inside-hull rejection, no top-up to `site_count`. Those
    // filters would re-run in float against a hull rebuilt from re-imported
    // geometry, and a single rejected site changes the shard count, which
    // renames every shard after it and orphans the rigid bodies bound to them.
    std::vector<Vec3> explicit_sites;
};

// ★ WHICH PATH EACH CELL ACTUALLY TOOK.
//
// Exact mode falls back to the convex form per cell, and a fallen-back cell is
// INDISTINGUISHABLE ON SCREEN from one that was never asked to be exact: both
// are a convex blob. Without this counter, "I turned it on and it still looks
// wrong" cannot be told apart from "the mesh is open, so every cut refused to
// seal" — and those two have completely different fixes. Report it, do not make
// the user guess from the picture.
struct FractureStats {
    int cells_exact = 0;         // every cut sealed against real geometry
    // Cut crossed a hole in the surface and its loose ends were joined. Still
    // the real surface with real UVs — but the cross-section there is a guess,
    // so the shard's volume (and therefore its mass) is approximate. Normal and
    // expected on game assets, which routinely have hidden faces deleted.
    int cells_approximated = 0;
    int cells_unsealed = 0;      // no usable cross-section at all -> hull
    int cells_hull = 0;          // convex mode, or fell back for any reason
    // ★ The sites this run actually used, which is the run's REPLAY RECORD.
    //
    // Reported rather than recomputed because the caller cannot derive it: the
    // scatter rejects candidates outside the hull and tops up from a second
    // uniform pass, so "seed + pattern + count" does not determine the result on
    // its own. Feed this back through FractureParams::explicit_sites to cut the
    // same object the same way after a reload.
    std::vector<Vec3> sites;
};

// Build Voronoi shards from `source`. Returns false (and leaves `out_shards`
// empty) when the input is degenerate (fewer than 4 non-coplanar vertices, i.e.
// no 3D hull) or no shard survives. `out_shards` is cleared first.
bool generateFracture(const std::vector<FractureInputTri>& source,
                      const FractureParams& params,
                      std::vector<FractureShard>& out_shards,
                      FractureStats* out_stats = nullptr);

} // namespace RayTrophiSim
