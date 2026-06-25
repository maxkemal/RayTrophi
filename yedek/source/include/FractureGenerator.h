/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          FractureGenerator.h
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 *
 * Convex Voronoi pre-fracture (Faz 1 of the destruction pipeline).
 *
 * Given a closed mesh's world-space triangle soup, this scatters N "sites"
 * inside the mesh and clips the mesh's CONVEX HULL by each site's Voronoi cell
 * (the intersection of the bisector half-spaces with the hull). The result is a
 * set of CONVEX shards — exactly the shape Jolt wants for a dynamic ConvexHull
 * rigid body, so each shard drops straight into the existing rigid-body system.
 *
 * It is geometry only: no physics, no scene mutation. The caller (SceneUI)
 * turns each FractureShard into a scene node + (later) a rigid body.
 *
 * Limitation (Faz 1): concave inputs are fractured as their convex hull. True
 * solid/concave fracture (OpenVDB LevelSetFracture) is a later phase. Cell
 * boundaries that lie on the original surface keep the hull faces; cell
 * boundaries created by a cut are flagged `interior` so a different inner
 * material can be assigned later.
 */

#pragma once

#include "Vec3.h"
#include <vector>
#include <cstdint>

namespace RayTrophiSim {

// One world-space source triangle (only positions are needed to build the hull).
struct FractureInputTri {
    Vec3 a, b, c;
};

// One triangle of a generated shard (world space). The flat face normal is
// stored on all three corners (shards shade faceted, like real broken stone).
struct FractureShardTri {
    Vec3 a, b, c;
    Vec3 n;            // flat face normal (outward)
    bool interior;     // true: this face was created by a cut (candidate inner material)
};

// One convex shard.
struct FractureShard {
    std::vector<FractureShardTri> tris;
    Vec3  centroid = Vec3(0.0f, 0.0f, 0.0f);  // world-space centre of mass
    float volume = 0.0f;                        // world-space volume
};

enum class FracturePattern : int {
    Uniform = 0,          // sites scattered uniformly through the hull
    ImpactClustered = 1,  // sites densified near `impact_point` (finer near the hit)
};

struct FractureParams {
    int      site_count = 12;        // number of shards requested (some may be culled)
    uint32_t seed = 1337u;
    FracturePattern pattern = FracturePattern::Uniform;
    Vec3     impact_point = Vec3(0.0f, 0.0f, 0.0f);  // ImpactClustered cluster centre (world)
    float    impact_radius = 0.5f;                    // cluster spread (world units)
    float    min_shard_volume_ratio = 1.0e-4f;        // drop shards smaller than this × hull volume
};

// Build convex Voronoi shards from `source`. Returns false (and leaves
// `out_shards` empty) when the input is degenerate (fewer than 4 non-coplanar
// vertices, i.e. no 3D hull) or no shard survives. `out_shards` is cleared first.
bool generateConvexFracture(const std::vector<FractureInputTri>& source,
                            const FractureParams& params,
                            std::vector<FractureShard>& out_shards);

} // namespace RayTrophiSim
