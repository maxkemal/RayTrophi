#pragma once

#include "Vec3.h"

#include <cstdint>
#include <vector>

namespace RayTrophiSim {

struct MeltSurfaceFlowSettings {
    // 0 = runs like water and levels out in a few iterations,
    // 1 = liquefies but stays where it is.
    float viscosity = 0.5f;
    // How far a vertex that has lost ALL of its material sinks, as a fraction of
    // the object's own rest height. Scaling by the object keeps the same
    // substance behaving the same on a coffee cup and on a cathedral.
    float maximum_height_loss = 0.85f;
    // Cap on the outward push applied where liquid pools. Purely a silhouette
    // control: the volume accounting is done on the vertical axis.
    float maximum_lateral_gain = 1.50f;
    // Downhill transport iterations. More iterations = liquid travels further
    // per solve, not more liquid.
    uint32_t flow_iterations = 12;
};

// Quasi-static molten-shell flow, solved on the REST pose.
//
// ★ WHAT THIS IS, PRECISELY. Each vertex owns a share of the object's surface
// material. `local_mass_fraction` is how much of that share is still attached
// (pyrolysis and APIC transfer have taken the rest); `melt` is how much of what
// remains is liquid. The liquid part is MOBILE and is transported downhill
// through the triangle-connectivity graph, conserving total volume exactly. The
// surface then sits at whatever height the material under it adds up to.
//
// Two properties follow, and they are the reason this replaces the previous
// masked-smoothing version:
//   - Material actually FLOWS. The old pass diffused the melt VALUE but gated
//     it on `melt[i] > 0`, so it could never spread into a cold neighbour: it
//     smoothed the melted region without ever moving anything out of it.
//   - Geometric volume loss EQUALS the mass that left. Because thickness is
//     driven by `local_mass_fraction`, a kilogram handed to APIC removes exactly
//     its own volume from the mesh, instead of the two being tuned separately
//     and drifting apart.
//
// Rest-pose based, so it is idempotent, monotonic in the inputs, and replays
// identically from the timeline cache without any state of its own.
//
// Lateral spread is the one approximation: pooling pushes outward from the
// object's vertical axis rather than along a real surface tangent. The volume
// ACCOUNTING stays exact; only the silhouette of a puddle is approximate.
//
// ★ `sampled` marks which vertices the caller could actually read the field for.
// A vertex whose UV lands on no island has NO answer, and that is not the same
// as an answer of zero: leaving it at melt = 0 while its neighbours sank left it
// standing as a spike on the melted surface. Unsampled vertices are filled from
// their sampled neighbours through the mesh, which invents no field value — it
// only says the surface is continuous where the UV layout is not. A component
// that is unmapped end to end stays at its rest pose, which is still the honest
// outcome for geometry nothing can be known about. Pass it empty to treat every
// vertex as sampled.
bool solveMeltSurfaceFlow(const std::vector<Vec3>& rest_positions,
                          const std::vector<uint32_t>& triangle_indices,
                          const std::vector<float>& melt,
                          const std::vector<float>& local_mass_fraction,
                          const std::vector<uint8_t>& sampled,
                          const MeltSurfaceFlowSettings& settings,
                          std::vector<Vec3>& out_positions);

} // namespace RayTrophiSim
