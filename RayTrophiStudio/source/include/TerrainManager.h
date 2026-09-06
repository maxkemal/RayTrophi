/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          TerrainManager.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

#include "TerrainSystem.h"
#include "json.hpp"
#include <vector>
#include <string>
#include <functional>
#include "FoliageFwd.h"
#include "Transform.h"

struct SceneData; // Forward decl
class Material;
class Texture;

// Serialization version - increment when format changes
static constexpr int TERRAIN_SERIALIZATION_VERSION = 3;

// ===========================================================================
// EROSION PARAMETERS
// ===========================================================================

enum class ErosionBoundaryMode : int {
    Preserve = 0, // Blend the simulation back to the authored border heights.
    Open = 1,     // Water/sediment may leave the domain; do not reshape geometry.
    SeaLevel = 2  // Blend border geometry toward an explicit normalized level.
};

struct HydraulicErosionParams {
    int iterations = 50000;        // Number of 'hits' (Determines passes in GPU)
    int dropletLifetime = 64;      // Max steps per droplet
    float inertia = 0.05f;         // direction momentum (0-1)
    float sedimentCapacity = 2.0f; // Sediment carrying capacity
    float minSlope = 0.005f;       // Minimal slope for flow
    float erodeSpeed = 0.05f;      // Erosion rate
    float depositSpeed = 0.1f;     // Deposit rate
    float evaporateSpeed = 0.01f;  // Evaporation rate
    float gravity = 9.8f;          // Gravitational acceleration
    int erosionRadius = 2;         // Default channel width
    float initialWater = 1.0f;     // Water carried by a newly spawned droplet
    float initialSpeed = 1.0f;     // Initial droplet velocity
    float uphillErosion = 0.3f;    // Momentum-driven erosion while climbing a cell
    float flatSettling = 1.0f;     // Multiplier for low-slope sediment settling
    float velocitySettling = 1.0f; // Multiplier for deceleration-driven settling
    float minWater = 0.01f;        // Droplet termination water threshold
    float minSpeed = 0.01f;        // Droplet termination velocity threshold
    bool removeSpikes = true;
    bool fillPits = true;
    // Broad 3x3 smoothing erases first-order tributaries after the LEM has
    // protected and incised them. Channel-aware hillslopeDiffusion is the
    // physical smoother; keep this legacy cleanup opt-in.
    bool smoothSurface = false;
    unsigned int seed = 1337u;     // Deterministic CPU/GPU droplet distribution
    ErosionBoundaryMode boundaryMode = ErosionBoundaryMode::Preserve;
    int boundaryWidth = 0;         // Cells; 0 selects a resolution-aware width
    float boundaryLevel = 0.0f;    // Normalized height used by SeaLevel mode
    bool channelEvolution = true;  // Mature accumulated runoff without a second Fluvial solve
    int channelIterations = 12;    // Race-free grid transport/evolution passes
    float channelErosion = 0.18f;  // Bed incision response to accumulated discharge
    float channelDeposition = 0.22f; // Low-energy sediment settling
    float channelWidthScale = 1.0f;  // Hydraulic geometry width multiplier
    float channelDepthScale = 1.0f;  // Hydraulic geometry depth multiplier
    // DEPRECATED (2026-08-23): the macro stage carves a fixed valley depth from
    // a one-shot coarse catchment solve. It is a decoration, not a feedback --
    // the carved valley never changes the flow that carved it. `fluvialCycle`
    // supersedes it and the two are mutually exclusive; the macro path is kept
    // only until the LEM cycle has been validated on real projects.
    bool macroDrainage = true;
    float macroValleyScaleMeters = 140.0f;
    float macroHeadwaterAreaKm2 = 0.012f;
    float macroValleyDepthMeters = 10.0f;
    float macroValleyFloor = 0.35f;

    // ---------------------------------------------------------------------
    // Landscape Evolution Model cycle
    // ---------------------------------------------------------------------
    // The droplet solver sees only the local slope, so a valley floor and a
    // hillside obey identical physics per droplet and the difference between
    // them stays linear -- that is the whole reason eroded terrain came out
    // radially symmetric. This cycle adds the three couplings a real landscape
    // has and a Monte-Carlo droplet walk structurally cannot have:
    //   1. drainage-area feedback  (E ~ A^m S^n : superlinear, builds hierarchy)
    //   2. depression conditioning (lakes spill, the outlet incises, they drain)
    //   3. downstream sediment transport (load reaches standing water -> delta)
    // plus in-loop mass wasting and hillslope creep.
    bool  fluvialCycle = true;
    int   fluvialIterations = 16;
    float fluvialTimeStep = 1.0f;     // whole-cycle multiplier; dt = this/iterations

    // Rain. rainRate is a runoff depth per unit time; only its ratio to
    // settlingVelocity has physical meaning. Orographic rain is the second
    // symmetry breaker after drainage area: uniform rain gives every catchment
    // identical input, so only terrain shape can distinguish them.
    float rainRate = 1.0f;
    float orographicRain = 0.0f;      // 0 = uniform
    float rainWindDegrees = 45.0f;

    // Stream power. incisionK is metres of incision at A = 1 km^2 and slope 1
    // over the WHOLE cycle (dt already divides by the iteration count, so
    // changing fluvialIterations refines the solve without changing how much
    // material moves). It scales with terrain size because A is physical.
    float incisionK = 150.0f;
    float streamPowerM = 0.5f;
    float streamPowerN = 1.0f;
    float slopeMin = 1.0e-4f;         // pow() guard, not a physical floor
    float slopeMax = 4.0f;

    // Sediment. transportK is the capacity coefficient in the same units as
    // incisionK; sedimentCover is the strength of the cover effect (a bed
    // already carrying its capacity is armoured -> alluvial valley floors).
    float transportK = 240.0f;
    float sedimentCover = 0.6f;
    float settlingVelocity = 0.35f;   // larger -> sediment drops sooner
    int   sedimentRouteSteps = 96;    // cells of downstream travel per iteration

    // ★★★ AVULSION. Route steps between rebuilds of the MFD weights from the
    // LIVE bed (max(conditioned, bed)) instead of the conditioned surface the
    // drainage solve produced. 0 disables it and restores the old behaviour.
    //
    // Without this the weights are frozen for a whole refresh interval while
    // the bed aggrades beneath them, so flow keeps using a channel it has
    // already buried. That is not a small error: an alluvial fan is MADE of
    // avulsions -- the channel silts up, stops being the lowest path, swings
    // aside, and the abandoned lobe is left behind. With frozen weights the
    // load stacks into one ridge, and "a new channel cutting through its own
    // deposit" cannot happen at all. It is cheap: one local dispatch, no
    // drainage re-solve, and discharge deliberately stays frozen (a catchment
    // does not change because a bar grew; a flow path does).
    int   avulsionInterval = 8;

    // ★★★ ALLUVIAL SPREADING. Fresh deposit relaxes toward a stable low
    // angle instead of standing where the water happened to drop it. This is
    // the depositional half of water erosion, and it was absent: the route
    // pass put material exactly under the channel, so a fan came out as a
    // narrow ridge and a footslope apron never formed. Real fan surfaces sit
    // at roughly 1-5 degrees.
    //
    // Only loose material moves (bounded by the alluvium thickness). Without
    // that bound the pass is "relax everything to 2 degrees", which does not
    // look slightly wrong -- it dissolves the mountains.
    float alluviumSlopeDegrees = 2.0f;
    float alluviumRate = 0.5f;
    int   alluviumSteps = 4;          // 0 disables spreading
    float alluviumConsolidation = 0.15f;  // mobility lost per spread step

    // Drainage solve. The fill and accumulation are relaxations on the GPU, so
    // these are convergence budgets, not quality dials: too low leaves basins
    // over-filled (spurious lakes) and long trunks under-counted (weak main
    // rivers). The CPU path solves both exactly and is the reference.
    int   drainageRefreshInterval = 6;
    int   drainageFillPasses = 96;
    int   drainageAccumulatePasses = 192;
    int   drainageCoarsestSize = 128;

    // ★★★★ FLAT ROUTING GRADIENT (Garbrecht & Martz 1997).
    //
    // In a flat or a filled depression the terrain has NO gradient, so the
    // direction flow takes there is a property of the conditioning algorithm,
    // not of the landscape. The old scheme raised each flood ring by a few
    // ulp, which is a geodesic distance field from the outlet -- and because
    // the fill charged a diagonal the same as a cardinal step, the level sets
    // of that field are SQUARES. Steepest descent over squares runs along 0
    // and 45 degrees: straight, angular channels that never merge, on exactly
    // the low-gradient ground where real rivers braid and converge. Dithering
    // the ladder cannot fix it (the dither is summed along the path, so its
    // relative spread falls as 1/sqrt(N) and a long flat converges back to a
    // linear ramp); only building the gradient from a SECOND distance field --
    // distance from where higher ground meets the flat -- makes tributaries
    // enter where they actually arrive and converge instead of running
    // parallel to the outlet rings.
    //
    // ★★★ This is a SLOPE IN m/m, not an epsilon, and the magnitude is load
    // bearing. A few ulp of drop is numerically zero drop, and incision is
    // clamped to incisionSafety * (drop to the receiver) -- so with the old
    // ladder a flat could not incise by any amount, for any parameter setting.
    // No first cut means no A^m feedback, which means the pattern the first
    // drainage solve drew on the flat was the FINAL pattern. At a real
    // floodplain gradient the channel cuts a bed, the bed takes over, and the
    // synthetic ramp stops mattering. That handover is the point.
    //
    // Raising it makes flat drainage more decisive and more incised; lowering
    // it toward zero restores the frozen behaviour. 0 disables the ramp.
    float flatGradient = 2.0e-4f;     // m/m, a real low-gradient floodplain

    // GPU only. The geodesic fronts advance one cell per pass, so this is
    // literally "the widest flat, in cells, that can be resolved". Cells the
    // front never reaches keep no gradient and stay terminal sinks -- they are
    // COUNTED (stats.unresolvedFlatCells), never silently treated as drained,
    // because an unresolved flat renders perfectly plausibly while truncating
    // every catchment upstream of it. The CPU path solves both fronts exactly
    // and ignores this budget.
    int   flatResolvePasses = 256;

    // Mass wasting and creep.
    bool  massWasting = true;
    float reposeAngleDegrees = 34.0f;
    float massWastingRate = 0.5f;
    int   massWastingSteps = 4;
    float hillslopeDiffusion = 0.02f; // D in m^2 per whole cycle.
                                      // 0.35 was erasing small channels every iteration:
                                      // incision opened them, diffusion closed them, net = blur.
                                      // 0.02 preserves physical creep on hillslopes without
                                      // competing against incisionK on tributary channels.
    float channelRefAreaKm2 = 0.005f; // creep is halved at this catchment area.
                                      // 0.05 km^2 left tributaries unprotected (full diffusion);
                                      // 0.005 km^2 shields streams from ~5000 m^2 upward.

    // Numerical limits. These are the difference between a landscape and a
    // field of spikes, so they are exposed rather than hidden constants.
    float incisionSafety = 0.5f;      // never cut past this fraction of the
                                      // drop to the receiver (anti-pit)
    // ★★★ Total height one cell may gain over the whole run. The per-pass
    // accommodation limit is `depositionSafety * rise + depositFloor`, and on
    // FLAT ground `rise` is zero - which is exactly where the anti-dam guard
    // was supposed to bite. `depositFloor` (1 % of a cell) then applies
    // unconditionally, every route pass: 16 iterations x 96 route steps x 3
    // erosion stages is roughly 45 m of unopposed building. Meanwhile incision
    // is capped at `incisionSafety * dropNorm`, which is also zero on a flat.
    // A flat cell can therefore only ever GAIN height - a ratchet - and the
    // river dams itself with its own load, closing a depression upstream.
    // Measured on the scene that exposed this: deepest fill 38.5 m, a quarter
    // of the map inside closed depressions.
    float maxDepositionMeters = 4.0f;
    float depositionSafety = 0.5f;    // never build past this fraction of the
                                      // rise to the donor (anti-spike)
    float maxStepMeters = 0.0f;       // 0 = auto (half a cell)
    float lakeEpsilonMeters = 0.01f;  // below this a fill residue is not a lake

    // Hydraulic geometry for the published fields.
    float fluvialWidthScale = 1.0f;
    float fluvialDepthScale = 1.0f;
    float fluvialHeadwaterAreaKm2 = 0.01f;
};

// Convergence budget presets. These set ONLY the cost dials -- iteration count,
// relaxation pass budgets, transport steps, pyramid depth -- and never the
// shape dials, so switching quality refines the same landscape instead of
// producing a different one. Everything they touch is still individually
// reachable from the panel and from script under Custom.
enum class FluvialQuality : int { Draft = 0, Balanced = 1, High = 2, Custom = 3 };
const char* fluvialQualityName(FluvialQuality quality);
void applyFluvialQuality(HydraulicErosionParams& params, FluvialQuality quality);
// Reports which preset a parameter set corresponds to, or Custom when it
// matches none. Used so a loaded project does not claim a preset it has since
// been edited away from.
FluvialQuality detectFluvialQuality(const HydraulicErosionParams& params);

// Mass ledger and shape diagnostics for one erosion run. This exists because
// the failure mode of every stage below is silent: a leak in transport, an
// under-converged fill, a talus pass that quietly does nothing, all produce a
// plausible-looking heightfield. Nobody reports those as bugs.
struct HydraulicErosionStats {
    double eroded = 0.0;        // normalized height units, summed over cells
    double deposited = 0.0;
    double exported = 0.0;      // left the domain through the open boundary
    double carried = 0.0;       // still in transit when the cycle ended
    double massError = 0.0;     // eroded - (deposited + exported + carried)
    double massErrorFraction = 0.0;
    int    lakeCells = 0;       // cells still under standing water at the end
    int    closedDepressions = 0;
    float  lakeAreaFraction = 0.0f;
    float  maxDrainageAreaKm2 = 0.0f;
    // ★★★ The largest catchment as a FRACTION of the map. km2 alone cannot be
    // read - 0.03 km2 is a shredded network on a 1 km terrain and a healthy
    // trunk on a 100 m one. Single digits mean no trunk river exists, however
    // convincing the render looks.
    float  maxDrainageAreaFraction = 0.0f;

    // ★★★ SHAPE OF THE DEPOSIT. The point of alluvial spreading is that the
    // load stops standing where the water dropped it, and nothing in the mass
    // ledger can tell those apart: a ridge and a fan of the same volume book
    // identically. These three do tell them apart, and they must be read
    // TOGETHER - one of them alone says nothing:
    //
    //   * A ridge is a few cells, thick. Small depositedAreaFraction, and
    //     deepestDepositMeters many times meanDepositMeters (ratio 10+).
    //   * A fan is many cells, thin. Larger area fraction and a ratio near 2-4,
    //     because a cone's peak is a small multiple of its mean thickness.
    //
    // ★ The insidious reading is a HIGH area fraction with a LOW ratio across
    // the whole map: that is not a fan, that is the spreading pass having
    // escaped its loose-material bound and started smoothing bedrock. Check it
    // against erosion patterns, not against how good the render looks.
    int    depositedCells = 0;            // deposit thicker than 1 cm
    float  depositedAreaFraction = 0.0f;
    float  deepestDepositMeters = 0.0f;
    float  meanDepositMeters = 0.0f;      // averaged over depositedCells only
    // lakeCells counts ANY standing water, including the conditioning ladder's
    // few-ulp lift, so it can read a quarter of the map and mean nothing. These
    // three carry a physical threshold and are the ones to act on.
    int    deepLakeCells = 0;        // deeper than 10 cm
    float  deepLakeAreaFraction = 0.0f;
    float  deepestLakeMeters = 0.0f;
    float  drainageDensity = 0.0f;   // channel cells / total cells

    // ★★★ Flat cells the outlet front never reached, so they carry no routing
    // gradient and are terminal sinks for drainage area. NOT a cosmetic
    // number: an unresolved flat renders as perfectly ordinary ground while
    // silently truncating every catchment upstream of it, which reads
    // downstream as "the trunk river is weaker than its tributaries deserve".
    // Non-zero means flatResolvePasses is below the widest flat on the map.
    // GPU path only; the CPU reference resolves every flat exactly.
    int    unresolvedFlatCells = 0;
    float  unresolvedFlatFraction = 0.0f;

    float  cycleMilliseconds = 0.0f;
    int    cycleIterations = 0;
    bool   gpuPath = false;

    void reset() { *this = HydraulicErosionStats(); }
};

// Optional transient products emitted by the hydraulic solver. These are
// accumulated by the simulation itself (not reconstructed from height deltas).
struct HydraulicErosionFields {
    int width = 0;
    int height = 0;
    std::vector<float> erosion;
    std::vector<float> deposition;
    std::vector<float> discharge;
    std::vector<float> sediment;
    std::vector<float> directionX;
    std::vector<float> directionY;
    std::vector<float> channelWidth;
    std::vector<float> waterDepth;
    std::vector<float> waterLevel;
    // LEM products. drainageArea is in m^2 (resolution independent), lakeDepth
    // in metres of standing water above the bed.
    std::vector<float> drainageArea;
    std::vector<float> lakeDepth;
    HydraulicErosionStats stats;

    void reset(int w, int h) {
        width = w; height = h;
        const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
        erosion.assign(n, 0.0f); deposition.assign(n, 0.0f);
        discharge.assign(n, 0.0f); sediment.assign(n, 0.0f);
        directionX.assign(n, 0.0f); directionY.assign(n, 0.0f);
        channelWidth.assign(n, 0.0f); waterDepth.assign(n, 0.0f);
        waterLevel.assign(n, 0.0f);
        drainageArea.assign(n, 0.0f); lakeDepth.assign(n, 0.0f);
        stats.reset();
    }
};

struct ThermalErosionParams {
    int iterations = 50;          // Moderate default
    float talusAngle = 0.5f;       // ~27 degrees
    float erosionAmount = 0.3f;    // Less aggressive
    float anisotropy = 0.0f;       // Directional thermal stress (0 isotropic)
    float anisotropyDirection = 0.0f; // Degrees in terrain XZ
    float talusSettling = 1.0f;    // Mobility of unstable debris
    float sedimentRemoval = 0.0f;  // Fraction of transported mass leaving domain
    bool fineDetail = false;
    float debrisSizeMeters = 1.0f;
};

class TerrainManager {
public:
    static TerrainManager& getInstance() {
        static TerrainManager instance;
        return instance;
    }
    // Keyframe Animation
    // We pass the track directly to avoid Scene dependency in this header
    void captureKeyframeToTrack(TerrainObject* terrain, struct ObjectAnimationTrack& track, int frame);
    void applyKeyframe(TerrainObject* terrain, const struct TerrainKeyframe& keyframe);
    
    // Helper to interpolate between two keyframes manually if needed (usually KeyframeSystem handles this)
    void updateFromTrack(TerrainObject* terrain, const struct ObjectAnimationTrack& track, int currentFrame);

    // Create a flat terrain grid
    // ★ height_scale is a CREATION parameter, not a post-edit. Callers used to
    // create the terrain, then assign heightmap.scale_y and call
    // updateTerrainMesh() a second time -- a full extra vertex+normal pass
    // (156 ms measured at 4096^2) whose only job was to re-apply a scalar the
    // first pass could have used.
    // ★ mesh_resolution is a CREATION parameter too, for the same reason
    // height_scale is: creating at the field resolution and decimating after
    // means paying the full acceleration-structure build once (5.6 s at 4096^2)
    // before saving anything. 0 = vertex grid follows the field.
    TerrainObject* createTerrain(SceneData& scene, int resolution, float size,
                                 float height_scale = 10.0f, int mesh_resolution = 0);
    
    // Create terrain from heightmap image (using stb_image)
    TerrainObject* createTerrainFromHeightmap(SceneData& scene, const std::string& filepath, float size, float maxHeight, int max_resolution = 1024);
    
    // Update mesh vertices based on heightmap (Call after sculpting)
    void updateTerrainMesh(TerrainObject* terrain, bool signalRebuild = true);

    // Rebuild mesh topology (Call when resolution changes)
    void rebuildTerrainMesh(SceneData& scene, TerrainObject* terrain);
    
    // Update only dirty sectors (incremental update for performance)
    void updateDirtySectors(TerrainObject* terrain, bool clearRegion = true);
    
    // ===========================================================================
    // NORMAL CALCULATION
    // ===========================================================================
    Vec3 calculateNormal(TerrainObject* terrain, int x, int y);  // Uses terrain->normal_quality
    Vec3 calculateSobelNormal(TerrainObject* terrain, int x, int y);  // 8-neighbor Sobel filter
    Vec3 calculateFastNormal(TerrainObject* terrain, int x, int y);   // 4-neighbor central difference
    
   
    // Sculpting
    // mode: 0=Raise, 1=Lower, 2=Flatten, 3=Smooth, 4=Stamp
    void sculpt(TerrainObject* terrain, const Vec3& hitPoint, int mode, float radius, float strength, float dt,
                float curve = 2.0f, float targetHeight = 0.0f,
                std::shared_ptr<class Texture> stampTexture = nullptr, float rotation = 0.0f,
                bool signalHeavyRebuild = true);
    
    void smoothTerrain(TerrainObject* terrain, int iterations);

    // Layer System & Painting
    void initLayers(TerrainObject* terrain);
    // channel: 0=R, 1=G, 2=B, 3=A
    void paintSplatMap(TerrainObject* terrain, const Vec3& hitPoint, int channel, float radius, float strength, float dt);
    // Auto-generate mask based on slope and height
    void autoMask(TerrainObject* terrain, float slopeWeight, float heightWeight, float heightMin, float heightMax, float slopeSteepness);
    
    // Internal helper to sync CPU splat data to GPU texture
    void updateSplatMapTexture(TerrainObject* terrain);
    // resizePaintMaps resizes both splatMap and macroColorMap to paintGridWidth/Height.
    // This is the canonical name; resizeSplatMap below is a compatibility forwarder.
    void resizePaintMaps(TerrainObject* terrain);
    inline void resizeSplatMap(TerrainObject* terrain) { resizePaintMaps(terrain); }
    void exportSplatMap(TerrainObject* terrain, const std::string& filepath);
    void importSplatMap(TerrainObject* terrain, const std::string& filepath);
    void exportHeightmap(TerrainObject* terrain, const std::string& filepath);

    // Flow Analysis (New)
    void calculateFlowMap(TerrainObject* terrain);

    // ===========================================================================
    // FOLIAGE SYSTEM
    // ===========================================================================    // Foliage
    void updateFoliage(TerrainObject* terrain, OptixWrapper* optix);
    void clearFoliage(TerrainObject* terrain, OptixWrapper* optix);
    void reapplyAllFoliage(OptixWrapper* optix); // Re-adds persistence after rebuild
    int migrateLegacyFoliageToInstanceGroups(SceneData& scene, bool clearLegacy = true);
    bool hasLegacyFoliage() const;
    
    // Serialization===========================================================================
    // EROSION SYSTEM
    // ===========================================================================
    // progressCallback (optional): invoked periodically with a 0..1 fraction of
    // iterations completed. These CPU loops are single-threaded and can run for
    // tens of seconds at default iteration counts on a background evaluate
    // thread — without this, the node-editor progress bar shows no movement for
    // the whole duration a single erosion node is running.
    void hydraulicErosion(TerrainObject* terrain, const HydraulicErosionParams& params, const std::vector<float>& mask = {}, const std::function<void(float)>& progressCallback = nullptr, HydraulicErosionFields* fields = nullptr);
    void hydraulicErosionAdvanced(TerrainObject* terrain, const HydraulicErosionParams& params, const std::vector<float>& mask = {});
    void fluvialErosion(TerrainObject* terrain, const HydraulicErosionParams& params,
                        const std::vector<float>& mask = {},
                        const std::function<void(float)>& progressCallback = nullptr,
                        const std::vector<float>& flowGuide = {});
    void fluvialErosionGPU(TerrainObject* terrain, const HydraulicErosionParams& params,
                           const std::vector<float>& mask = {},
                           const std::vector<float>& flowGuide = {});
    void hydraulicErosionGPU(TerrainObject* terrain, const HydraulicErosionParams& params, const std::vector<float>& mask = {}, HydraulicErosionFields* fields = nullptr);
    void hydraulicErosionMultiPass(TerrainObject* terrain,
                                   const std::vector<HydraulicErosionParams>& stages,
                                   bool useGPU, const std::vector<float>& mask = {},
                                   const std::function<void(float)>& progressCallback = nullptr,
                                   HydraulicErosionFields* fields = nullptr);
    void thermalErosionGPU(TerrainObject* terrain, const ThermalErosionParams& params, const std::vector<float>& mask = {});
    void thermalErosion(TerrainObject* terrain, const ThermalErosionParams& params, const std::vector<float>& mask = {}, const std::function<void(float)>& progressCallback = nullptr);

    void windErosion(TerrainObject* terrain, float strength, float direction, int iterations, const std::vector<float>& mask = {}, const std::function<void(float)>& progressCallback = nullptr);
    void windErosionGPU(TerrainObject* terrain, float strength, float direction, int iterations, const std::vector<float>& mask = {});
    
    // Edge preservation helpers (prevents cliffs/walls at terrain boundaries)
    void preserveEdges(TerrainObject* terrain, const std::vector<float>& originalHeights, int fadeWidth);
    int getEdgeFadeWidth(TerrainObject* terrain);
    
    // ===========================================================================
    // HARDNESS SYSTEM (for realistic erosion)
    // ===========================================================================
    void initHardnessMap(TerrainObject* terrain, float defaultHardness = 0.3f);
    void autoGenerateHardness(TerrainObject* terrain, float slopeWeight = 0.7f, float noiseAmount = 0.2f);
    void paintHardness(TerrainObject* terrain, const Vec3& hitPoint, float radius, float strength, float dt, bool increase);
    
    // Combined Wizard Process (Thermal -> Fluvial -> Wind)
    void applyCombinedErosion(TerrainObject* terrain, int iterations, float strength, bool useGPU = false);
    
    // Progress callback type for long-running operations
    using ProgressCallback = std::function<void(float progress, const std::string& stage)>;
    
    // Erosion with progress reporting
    void applyCombinedErosionWithProgress(TerrainObject* terrain, int iterations, float strength, ProgressCallback callback);
    
    // ===========================================================================
    // HEIGHTMAP EXPORT/IMPORT
    // ===========================================================================
   
    void importMaskChannel(TerrainObject* terrain, const std::string& filepath, int channel);
    
    // ===========================================================================
    // SERIALIZATION
    // ===========================================================================
    /**
     * @brief Serialize all terrains to JSON + binary files
     * @param terrainDir Directory to save terrain data
     * @return JSON object containing terrain metadata
     */
    nlohmann::json serialize(const std::string& terrainDir) const;
    
    /**
     * @brief Deserialize terrains from JSON + binary files
     * @param data JSON object containing terrain metadata
     * @param terrainDir Directory where terrain data is stored
     * @param scene SceneData reference to add triangles
     */
    void deserialize(const nlohmann::json& data, const std::string& terrainDir, SceneData& scene);
    
    /**
     * @brief Save heightmap to binary file (float32 raw format)
     */
    void saveHeightmapBinary(const TerrainObject* terrain, const std::string& filepath) const;
    
    /**
     * @brief Load heightmap from binary file
     */
    void loadHeightmapBinary(TerrainObject* terrain, const std::string& filepath);
    
    // Getters
    std::vector<TerrainObject>& getTerrains() { return terrains; }

    TerrainObject* getTerrain(int id);
    TerrainObject* getTerrainByName(const std::string& name);

    // ===========================================================================
    // HEIGHT SAMPLING & RAYCAST
    // ===========================================================================
    
    // Ray-Terrain Intersection (Ignores all other objects, perfect for sculpting/painting)
    // Returns true if hit, populates t_out and normal_out
    bool intersectRay(TerrainObject* terrain, const Ray& r, float& t_out, Vec3& normal_out, float t_min = 0.001f, float t_max = 1e9f);
    
    // Check if any terrain exists
    bool hasActiveTerrain() const { return !terrains.empty(); }
    
    // Sample height at world XZ coordinate
    float sampleHeight(float worldX, float worldZ) const {
        if (terrains.empty()) return 0.0f;
        
        // Find terrain containing this point
        for (const auto& terrain : terrains) {
            const Heightmap& hm = terrain.heightmap;
            if (hm.data.empty() || hm.width <= 0 || hm.height <= 0) continue;
            
            // 1. Transform World position to Local terrain space
            Vec3 localPos(worldX, 0, worldZ);
            if (terrain.transform) {
                Matrix4x4 inv = terrain.transform->getFinal().inverse();
                localPos = inv.multiplyVector(Vec4(worldX, 0, worldZ, 1.0f)).xyz();
            }

            // 2. Check if local position is within terrain bounds [0, scale_xz]
            if (localPos.x < 0 || localPos.x > hm.scale_xz || localPos.z < 0 || localPos.z > hm.scale_xz) {
                continue; // Not this terrain
            }

            // 3. Convert local position to heightmap grid coordinates
            float normalizedX = localPos.x / hm.scale_xz;
            float normalizedZ = localPos.z / hm.scale_xz;
            
            // Clamp to valid range (redundant due to bounds check but safer)
            normalizedX = std::clamp(normalizedX, 0.0f, 1.0f);
            normalizedZ = std::clamp(normalizedZ, 0.0f, 1.0f);
            
            // Get grid coordinates
            float gx = normalizedX * (hm.width - 1);
            float gz = normalizedZ * (hm.height - 1);
            
            int x0 = (int)std::floor(gx);
            int z0 = (int)std::floor(gz);
            int x1 = (std::min)(x0 + 1, hm.width - 1);
            int z1 = (std::min)(z0 + 1, hm.height - 1);
            
            float fx = gx - x0;
            float fz = gz - z0;
            
            // Bilinear interpolation
            float h00 = hm.data[z0 * hm.width + x0];
            float h10 = hm.data[z0 * hm.width + x1];
            float h01 = hm.data[z1 * hm.width + x0];
            float h11 = hm.data[z1 * hm.width + x1];
            
            float h0 = h00 * (1.0f - fx) + h10 * fx;
            float h1 = h01 * (1.0f - fx) + h11 * fx;
            float local_height = (h0 * (1.0f - fz) + h1 * fz) * hm.scale_y;
            
            // 4. Transform local height back to world space
            if (terrain.transform) {
                 Vec3 worldPos = terrain.transform->getFinal().multiplyVector(Vec4(localPos.x, local_height, localPos.z, 1.0f)).xyz();
                 return worldPos.y;
            }
            
            return local_height;
        }
        
        return 0.0f;
    }

    // Sample normal at world XZ coordinate
    Vec3 sampleNormal(float worldX, float worldZ) const;

    // Sample splat map channel value (0..1) at world XZ coordinate for the terrain containing the point
    // channel: 0=R,1=G,2=B,3=A. Returns -1.0f if no splat data or out of bounds.
    float sampleSplatChannel(float worldX, float worldZ, int channel) const;
    // Samples a named, graph-published terrain field (terrain.slope,
    // terrain.valley, terrain.wetness, ...). Returns 1 when no field is
    // requested and -1 when the requested field cannot be sampled.
    float sampleAnalysisField(float worldX, float worldZ, const std::string& fieldName) const;
    
    // ===========================================================================
    // RIVER BED CARVING (for River System integration)
    // ===========================================================================
    
    /**
     * @brief Natural carve parameters for realistic river bed generation
     */
    struct NaturalCarveParams {
        bool enableNoise = true;           // Noise-based edge irregularity
        float noiseScale = 0.15f;          // Noise frequency
        float noiseStrength = 0.3f;        // Noise intensity
        
        bool enableDeepPools = true;       // Random deep pools
        float poolFrequency = 0.15f;       // Pool occurrence rate
        float poolDepthMult = 1.8f;        // Pool depth multiplier
        
        bool enableRiffles = true;         // Shallow riffle zones
        float riffleFrequency = 0.2f;      // Riffle occurrence rate
        float riffleDepthMult = 0.4f;      // Riffle depth multiplier
        
        bool enableAsymmetry = true;       // Asymmetric bank profiles
        float asymmetryStrength = 0.6f;    // Inner/outer bank difference
        
        bool enablePointBars = true;       // Point bar deposits on inner bends
        float pointBarStrength = 0.4f;     // Point bar elevation amount
    };
    
    /**
     * @brief Carve a river bed into the terrain along a path
     * @param terrainId Target terrain ID (-1 for first terrain)
     * @param points Vector of world-space points along the river center
     * @param widths Width at each point
     * @param depths Depth at each point (how deep to carve)
     * @param smoothness Edge smoothing factor (0-1)
     * @param scene SceneData for mesh update
     */
    void carveRiverBed(int terrainId, 
                       const std::vector<Vec3>& points,
                       const std::vector<float>& widths,
                       const std::vector<float>& depths,
                       float smoothness,
                       SceneData& scene);
    
    /**
     * @brief Carve a natural river bed with advanced features
     * @param terrainId Target terrain ID (-1 for first terrain)
     * @param points Vector of world-space points along the river center
     * @param widths Width at each point
     * @param depths Depth at each point
     * @param smoothness Edge smoothing factor
     * @param naturalParams Natural carve parameters (noise, pools, etc.)
     * @param scene SceneData for mesh update
     */
    void carveRiverBedNatural(int terrainId, 
                              const std::vector<Vec3>& points,
                              const std::vector<float>& widths,
                              const std::vector<float>& depths,
                              float smoothness,
                              const NaturalCarveParams& naturalParams,
                              SceneData& scene);
    
    /**
     * @brief Lower terrain height at a world position
     * @param worldX World X coordinate
     * @param worldZ World Z coordinate
     * @param amount Amount to lower (positive = deeper)
     * @param radius Falloff radius
     * @param terrainId Target terrain (-1 for first)
     */
    void lowerHeightAt(float worldX, float worldZ, float amount, float radius, int terrainId = -1);
    
    // Management
    void removeTerrain(SceneData& scene, int id);
    void removeAllTerrains(SceneData& scene);

private:
    TerrainManager() = default;
    
    std::vector<TerrainObject> terrains;

    int next_id = 1;

    // =====================================================================
    // DEPRECATED (2026-08-23): CUDA erosion back end.
    // ---------------------------------------------------------------------
    // Vulkan compute is the primary GPU path for this project and now carries
    // every erosion stage, including the LEM cycle, which was never ported to
    // CUDA and will not be. Nothing in the UI selects CUDA; it is reachable
    // only as a fallback after the Vulkan path has already failed, which means
    // it is effectively untested on every run. Keeping two solvers alive "just
    // in case" is exactly how this codebase has produced silent divergence
    // before, so these handles, initCuda(), erosion_kernels.cu and the
    // erosion_kernels.ptx build step are all scheduled for removal.
    //
    // REMOVAL PLAN: once the LEM cycle has been validated on real projects,
    // delete this block, the `cuda*`/`*KernelFunc` members, initCuda(), every
    // `if (cudaInitialized)` branch in TerrainManager.cpp, erosion_kernels.cu
    // and its .ptx packaging. Do not add anything new below this line.
    // =====================================================================
    void* cudaModule = nullptr;
    void* erosionKernelFunc = nullptr;
    void* smoothKernelFunc = nullptr;
    void* thermalKernelFunc = nullptr;
    // Fluvial Kernels
    void* fluvRainKernelFunc = nullptr;
    void* fluvFluxKernelFunc = nullptr;
    void* fluvWaterKernelFunc = nullptr;
    void* fluvErodeKernelFunc = nullptr;
    void* streamPowerKernelFunc = nullptr;
    void* applyStreamPowerKernelFunc = nullptr;
    void* windKernelFunc = nullptr;
    // Post-processing kernels (for CPU-GPU parity)
    void* pitFillKernelFunc = nullptr;
    void* spikeRemovalKernelFunc = nullptr;
    void* edgePreservationKernelFunc = nullptr;
    void* thermalWithHardnessKernelFunc = nullptr;
    bool cudaInitialized = false;
    [[deprecated("CUDA erosion is deprecated; Vulkan compute is the primary and only "
                 "maintained GPU erosion path. Scheduled for removal.")]]
    void initCuda();

    // Ledger of the most recent erosion run, published so a script can assert
    // on mass balance and drainage shape instead of eyeballing a render.
    HydraulicErosionStats lastStats;

public:
    const HydraulicErosionStats& lastErosionStats() const { return lastStats; }
    // A terrain node may refine the public, spatial diagnostics after the
    // solver returns (for example, net aggradation rather than the gross
    // transport ledger). Keep RTAPI's last-run report aligned with the node
    // products without changing the conservative mass totals.
    void publishLastErosionStats(const HydraulicErosionStats& stats) { lastStats = stats; }
};
