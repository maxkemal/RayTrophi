#pragma once
// ===========================================================================
// Landscape Evolution Model (LEM) cycle for terrain erosion.
// ===========================================================================
//
// WHY THIS EXISTS
//
// The droplet solver (terrain_hydraulic_droplet.comp / TerrainManager::
// hydraulicErosion) is a Monte-Carlo random walk. Every droplet starts with
// the same water, sees only the LOCAL slope, and dies after a fixed lifetime.
// Three structural consequences follow, and no amount of parameter tuning
// removes any of them:
//
//   1. NO DRAINAGE-AREA FEEDBACK.  A valley floor crossed by ten thousand
//      droplets and a hillside crossed by three obey identical physics per
//      droplet; the only difference between them is linear in the visit count.
//      Real fluvial hierarchy comes from E ~ A^m S^n with m near 0.5: a
//      channel that captures slightly more area incises faster and captures
//      more, which is a superlinear positive feedback. Without it, erosion is
//      statistically isotropic -- the "too symmetric" result.
//
//   2. NO DEPRESSION HANDLING.  A droplet that enters a pit dies there and
//      drops its load. Nothing spills, so a closed basin high on a mountain is
//      permanent. In reality a lake is a TEMPORARY base level: it fills, spills
//      over its lowest saddle, the outlet incises, and the lake drains.
//
//   3. NO LONG-RANGE SEDIMENT TRANSPORT.  A droplet lifetime of ~64 steps is
//      about 1.5% of a 4K map, so no grain can travel from a ridge to the
//      coast. A delta is by definition the load of an entire catchment
//      arriving at standing water, so deltas were not merely absent, they were
//      unreachable.
//
// This cycle supplies all three, plus mass wasting inside the loop (so valley
// walls collapse as channels incise, instead of being smoothed once at the
// end) and hillslope creep (which gives the convex-ridge / concave-valley
// signature that a uniform blur cannot).
//
// PASS ORDER, one iteration:
//   drainage solve (every `drainageRefreshInterval` iterations)
//     depression fill -> MFD weights -> rain-weighted area accumulation
//   incision      stream power on the real slope, lakes excluded, outlets not
//   transport     `sedimentRouteSteps` cell-lengths of advection + settling
//   mass wasting  `massWastingSteps` gather-form talus relaxations
//   creep         stability-substepped hillslope diffusion
//
// GPU / CPU CONTRACT
//   The morphology passes (incision, transport, mass wasting, creep) are the
//   same explicit schemes with the same limits on both paths and agree closely.
//   The DRAINAGE SOLVE does not: the CPU path uses an exact priority-flood and
//   an exact topological accumulation, the GPU path uses a cascadic pyramid of
//   Planchon-Darboux and Jacobi relaxations with a bounded pass budget. The CPU
//   result is therefore the reference and the GPU result is an approximation of
//   it. This is flagged rather than hidden: an under-converged GPU fill shows
//   up as spurious lakes, and an under-converged accumulation as trunk rivers
//   that are weaker than their tributaries deserve. Both are visible in
//   HydraulicErosionStats.
// ===========================================================================

#include "TerrainManager.h"
#include "SimulationCompute.h"

#include <functional>
#include <vector>

namespace TerrainLem {

// Full-resolution fields owned by the caller. `height` and `heightAlt` are a
// ping-pong pair: the cycle swaps them internally and the caller must re-read
// both members afterwards, because the live surface may have moved.
struct GpuFields {
    RayTrophiSim::ComputeBufferHandle height;
    RayTrophiSim::ComputeBufferHandle heightAlt;
    RayTrophiSim::ComputeBufferHandle hardness;
    RayTrophiSim::ComputeBufferHandle mask;
    RayTrophiSim::ComputeBufferHandle erosion;
    RayTrophiSim::ComputeBufferHandle deposition;
    RayTrophiSim::ComputeBufferHandle discharge;
    RayTrophiSim::ComputeBufferHandle channelWidth;
    RayTrophiSim::ComputeBufferHandle waterDepth;
    RayTrophiSim::ComputeBufferHandle waterLevel;
    RayTrophiSim::ComputeBufferHandle directionX;
    RayTrophiSim::ComputeBufferHandle directionY;
};

// Buffers owned by the cycle. They stay alive between the main cycle and the
// post-droplet polish so the polish can reuse the converged drainage and so
// sediment still in transit is carried across rather than silently dropped.
struct GpuState {
    // The depression-fill ping-pong pair. After a drainage solve, `lakeSurface`
    // and `route` below name whichever half holds what.
    RayTrophiSim::ComputeBufferHandle filledA, filledB;

    // ★★★ TWO SURFACES, and conflating them is what the flat artifact was made
    // of. They are ALIASES of filledA/filledB, not separate allocations, so
    // destroyGpuState must not free them.
    //
    //   lakeSurface -- the depression-filled surface, exactly level over a
    //                  flat or a lake. This is the one that means "standing
    //                  water reaches here": lakeDepth = lakeSurface - bed.
    //   route       -- lakeSurface plus the Garbrecht-Martz flat gradient.
    //                  This is the one flow STEERS by.
    //
    // They used to be one field carrying an epsilon ladder, which made it a
    // bad lake surface (the ladder counts as standing water, which is why
    // lakeCells could read a quarter of the map and mean nothing) and a worse
    // routing surface (the ladder is a square distance field). Splitting them
    // costs no memory and lets each be correct.
    RayTrophiSim::ComputeBufferHandle lakeSurface, route;

    // 16 bytes of GPU-written diagnostics, cleared at the start of every
    // drainage solve. Slot 0 counts flat cells the outlet front never reached.
    // A probe that is absent is not a probe that read zero.
    RayTrophiSim::ComputeBufferHandle diag;
    // Packed MFD outflow weights, one byte per direction (8 bytes per cell).
    // Rebuilt on every drainage refresh and read by both the accumulation and
    // the sediment routing. Without it those passes recompute a neighbour's
    // weight row from the neighbour's own neighbours, which is 64+ scattered
    // loads per cell per pass across thousands of passes -- the single reason
    // the first version of this cycle was far more expensive than the solver
    // it replaced.
    RayTrophiSim::ComputeBufferHandle weights;
    RayTrophiSim::ComputeBufferHandle rainArea;
    RayTrophiSim::ComputeBufferHandle areaA, areaB;
    RayTrophiSim::ComputeBufferHandle fluxA, fluxB;
    // Loose, unconsolidated sediment thickness (normalized), ping-ponged by
    // the alluvial spreading pass. Deliberately NOT the deposition ledger:
    // the ledger is a cumulative record used by the closing mass check, while
    // this is a live "how much material here can still move" field. Sharing
    // one buffer would make spreading double-book itself and the ledger check
    // would fail by exactly the amount the fan spread -- a leak report with no
    // leak behind it.
    RayTrophiSim::ComputeBufferHandle alluviumA, alluviumB;
    RayTrophiSim::ComputeBufferHandle exported;
    RayTrophiSim::ComputeBufferHandle lakeDepth;
    // The cycle keeps its OWN erosion/deposition accumulators rather than
    // writing into the caller's published fields. Sharing them would mix
    // droplet mass into the sediment ledger, and a ledger that cannot close
    // by construction is not a measurement -- it is a warning that fires every
    // run and gets ignored. The caller adds these into the published fields
    // after the ledger has been checked.
    RayTrophiSim::ComputeBufferHandle erosionLedger, depositionLedger;
    int width = 0;
    int height = 0;
    bool valid = false;

    // Drainage area in m^2, for the droplet stage's area-aware erosion cap.
    RayTrophiSim::ComputeBufferHandle liveArea() const { return areaA; }
};

bool createGpuState(RayTrophiSim::ISimulationComputeBackend* backend,
                    int width, int height, GpuState& state);
void destroyGpuState(RayTrophiSim::ISimulationComputeBackend* backend, GpuState& state);

// Runs `iterations` morphology steps. `publishFields` writes the hydrology
// products (discharge, channel width, water depth/level, flow direction, lake
// depth) from the FINAL surface and should be set only on the last call.
// `iterations == 0 && publishFields` is the publish-only contract: condition
// and accumulate the current surface without applying more morphology.
bool runGpu(RayTrophiSim::ISimulationComputeBackend* backend,
            const HydraulicErosionParams& params,
            int width, int height, float cellSize, float heightScale,
            int iterations, bool publishFields,
            GpuFields& fields, GpuState& state);

// CPU reference. Operates directly on terrain->heightmap.data and accumulates
// into `fields` when supplied. Exact drainage solve; see the contract note at
// the top of this file.
void runCpu(TerrainObject* terrain, const HydraulicErosionParams& params,
            const std::vector<float>& mask,
            int iterations, bool publishFields,
            HydraulicErosionFields* fields,
            HydraulicErosionStats& stats,
            const std::function<void(float)>& progressCallback);

// Shared post-run diagnostics: mass ledger closure, lake coverage, drainage
// density. Called by both paths so the numbers mean the same thing.
void summarize(const std::vector<float>& erosion,
               const std::vector<float>& deposition,
               const std::vector<float>& exported,
               const std::vector<float>& carried,
               const std::vector<float>& drainageArea,
               const std::vector<float>& lakeDepth,   // METRES, both callers
               float headwaterAreaKm2,
               float cellAreaM2,
               float heightScaleMeters,
               HydraulicErosionStats& stats);

// Convert the conservative gross deposition ledger into the spatial product
// artists and material nodes need: net positive height change from the input
// surface. Gross mass totals in stats are deliberately preserved; only the
// deposit shape fields are recomputed.
void publishNetAggradation(const std::vector<float>& inputHeight,
                           const std::vector<float>& outputHeight,
                           float heightScaleMeters,
                           std::vector<float>& deposition,
                           HydraulicErosionStats& stats);

}  // namespace TerrainLem
