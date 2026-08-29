// Landscape Evolution Model cycle. See TerrainErosionLem.h for why it exists
// and what it fixes; this file is the two implementations of that contract.
//
// Everything below works in NORMALIZED height units (the units of
// heightmap.data) and converts to metres only where physics needs it, via
// heightScale. Sediment flux is carried in the same units, meaning "the depth
// this material would occupy if spread over one cell". Cells all have the same
// area, so moving flux between cells conserves volume by construction and the
// mass ledger closes exactly -- which is the point, because every failure mode
// in this file is otherwise silent.

#include "TerrainErosionLem.h"
#include "globals.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <deque>
#include <limits>
#include <cstring>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

namespace TerrainLem {
namespace {

using RayTrophiSim::ComputeBufferDesc;
using RayTrophiSim::ComputeBufferHandle;
using RayTrophiSim::ComputeDispatch;
using RayTrophiSim::ISimulationComputeBackend;

constexpr int kNeighborX[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
constexpr int kNeighborY[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
const float kNeighborDist[8] = { 1.41421356f, 1.0f, 1.41421356f, 1.0f,
                                 1.0f, 1.41421356f, 1.0f, 1.41421356f };

// Depression-fill increment, in normalized height. It has to be small enough
// that a long flat run does not accumulate into a fake lake (a 4000-cell run
// accumulates 4000 * eps) and large enough to stay above float resolution near
// h = 0.5, where one ulp is about 6e-8. Three ulp is the compromise.
constexpr float kFillEpsilon = 2.0e-7f;

struct Derived {
    float dt = 1.0f;
    float cellArea = 1.0f;
    float maxStep = 0.0f;        // normalized
    float lakeEps = 0.0f;        // normalized
    float depositFloor = 0.0f;   // normalized
    float maxBuildNorm = 0.0f;   // normalized cap on TOTAL deposition per cell
    float minDischarge = 1.0f;
    float tanRepose = 0.7f;
    float tanAlluvium = 0.035f;  // stable slope of a loose wet deposit
    float diffuseCoeff = 0.0f;   // per sub-step, <= 0.25
    int   diffuseSteps = 0;
    float channelRefArea = 1.0f; // m^2
    float windX = 1.0f, windY = 0.0f;
};


Derived derive(const HydraulicErosionParams& p, float cellSize, float heightScale,
               int iterations) {
    Derived d;
    const float safeScale = (std::max)(heightScale, 1.0e-3f);
    d.dt = p.fluvialTimeStep / (float)(std::max)(iterations, 1);
    d.cellArea = cellSize * cellSize;
    const float maxStepMeters = p.maxStepMeters > 0.0f ? p.maxStepMeters : cellSize * 0.5f;
    d.maxStep = maxStepMeters / safeScale;
    d.lakeEps = (std::max)(p.lakeEpsilonMeters, 0.0f) / safeScale;
    d.depositFloor = (cellSize * 0.01f) / safeScale;
    d.maxBuildNorm = (std::max)(p.maxDepositionMeters, 0.0f) / safeScale;
    // Unit-width discharge of a single unfed cell, quartered. Below this the
    // settling exponent saturates and fdep is 1 anyway; the floor exists to
    // keep the division defined, not to shape anything.
    d.minDischarge = (std::max)(cellSize * p.rainRate * 0.25f, 1.0e-6f);
    d.tanRepose = std::tan(std::clamp(p.reposeAngleDegrees, 1.0f, 80.0f) * 3.14159265f / 180.0f);
    // Clamped well below the bedrock repose angle on purpose: an "alluvial"
    // angle set to 30 degrees would let this pass move loose material about as
    // readily as talus does, and the fan would go back to being a ridge.
    d.tanAlluvium = std::tan(std::clamp(p.alluviumSlopeDegrees, 0.1f, 20.0f) * 3.14159265f / 180.0f);
    d.channelRefArea = (std::max)(p.channelRefAreaKm2, 1.0e-6f) * 1.0e6f;

    // Explicit diffusion is stable only for D*dt/dx^2 <= 0.25. Sub-step to
    // honour that instead of clamping the result: a diffusion scheme run past
    // its stability limit does not look slightly wrong, it oscillates into a
    // checkerboard of spikes.
    const float total = (std::max)(p.hillslopeDiffusion, 0.0f) * d.dt /
                        (std::max)(cellSize * cellSize, 1.0e-6f);
    if (total > 1.0e-9f) {
        d.diffuseSteps = std::clamp((int)std::ceil(total / 0.2f), 1, 32);
        d.diffuseCoeff = (std::min)(total / (float)d.diffuseSteps, 0.25f);
    }

    const float radians = p.rainWindDegrees * 3.14159265f / 180.0f;
    d.windX = std::cos(radians);
    d.windY = std::sin(radians);
    return d;
}

// ---------------------------------------------------------------------------
// GPU push-constant mirrors. Each must match its .comp byte for byte; the
// static_asserts are the only thing standing between a layout drift and a
// solver that silently reads garbage parameters.
// ---------------------------------------------------------------------------
struct RestrictPc { int fineWidth, fineHeight, coarseWidth, coarseHeight, mode; };
static_assert(sizeof(RestrictPc) == 20, "must match terrain_lem_restrict.comp");

struct ProlongatePc { int coarseWidth, coarseHeight, fineWidth, fineHeight, mode; };
static_assert(sizeof(ProlongatePc) == 20, "must match terrain_lem_prolongate.comp");

struct RainPc {
    int width, height;
    float cellArea, orographic, windX, windY, cellSize, heightScale;
};
static_assert(sizeof(RainPc) == 32, "must match terrain_lem_rain.comp");

// useBedMax != 0 routes on max(conditioned, bed) rather than the conditioned
// surface -- see the avulsion note in terrain_lem_weights.comp.
struct WeightsPc { int width, height; float cellSize; int useBedMax; };
static_assert(sizeof(WeightsPc) == 16, "must match terrain_lem_weights.comp");

struct AccumulatePc { int width, height; };
static_assert(sizeof(AccumulatePc) == 8, "must match terrain_lem_accumulate.comp");

struct FillPc { int mapWidth, mapHeight; float eps, noiseAmplitude; };
static_assert(sizeof(FillPc) == 16, "must match terrain_flow_fill.comp");

struct IncisePc {
    int width, height;
    float cellSize, heightScale, dt;
    float incisionK, exponentM, exponentN;
    float minSlope, maxSlope;
    float coverFactor, transportK;
    float incisionSafety, maxStep;
    float lakeEps;
};
static_assert(sizeof(IncisePc) == 60, "must match terrain_lem_incise.comp");

struct RoutePc {
    int width, height;
    float cellSize, heightScale;
    float settlingVelocity, rainRate, depositionSafety, minDischarge, lakeEps, depositFloor;
    float maxBuildNorm;
};
static_assert(sizeof(RoutePc) == 44, "must match terrain_lem_route.comp");

struct TalusPc {
    int width, height;
    float cellSize, heightScale, tanRepose, rate, hardnessRepose, maxStep;
};
static_assert(sizeof(TalusPc) == 32, "must match terrain_lem_talus.comp");

struct AlluviumPc {
    int width, height;
    float cellSize, heightScale, tanAlluvium, rate, maxStep, consolidation;
};
static_assert(sizeof(AlluviumPc) == 32, "must match terrain_lem_alluvium.comp");

struct DiffusePc { int width, height; float coeff, channelRefArea, lakeEps; };
static_assert(sizeof(DiffusePc) == 20, "must match terrain_lem_diffuse.comp");

struct FinalizePc {
    int width, height;
    float cellSize, heightScale, rainRate, widthScale, depthScale;
    float headwaterAreaKm2, lakeEps, dischargeScale;
};
static_assert(sizeof(FinalizePc) == 40, "must match terrain_lem_finalize.comp");

// Batches dispatches into one command buffer. The backend already emits a full
// shader-write -> shader-read barrier after every dispatch, so a synchronize()
// between passes buys nothing but a queue round trip.
//
// TWO independent reasons to flush anyway, and mixing them up cost a TDR:
//
//  1. The descriptor pool holds 512 sets and is only reset on synchronize().
//     This cycle issues thousands of dispatches, so a COUNT bound is required.
//
//  2. ★ A submission must not monopolise the queue. The first version bounded
//     only the count, which is a bound on dispatches, not on WORK: 384 passes
//     over a 4K grid is billions of cell updates in one submit -- seconds of
//     GPU time, which starves the renderer and trips the Windows TDR watchdog.
//     The droplet solver above already learned this exact lesson and bounds its
//     batches by particle-steps; this one has to bound by cell-updates. The
//     same pass count is cheap at 1K and lethal at 4K, which is why the bound
//     has to be on the work and not on the pass.
struct Batch {
    ISimulationComputeBackend* backend = nullptr;
    int pending = 0;
    uint64_t pendingWork = 0;
    bool ok = true;
    static constexpr int kFlushInterval = 384;
    // Cell-updates per submission. Every pass here is memory bound at roughly
    // 40 bytes per cell, so this is about ten gigabytes of traffic -- tens of
    // milliseconds on any card that runs this renderer, two orders of magnitude
    // under the watchdog.
    static constexpr uint64_t kWorkPerSubmission = 256ull * 1024ull * 1024ull;

    void flush() {
        if (pending > 0) { backend->synchronize(); pending = 0; pendingWork = 0; }
    }
    bool run(const char* kernel, uint32_t groups, const ComputeBufferHandle* bufs,
             uint32_t bufCount, const void* pc, size_t pcSize, uint64_t workCells) {
        if (!ok) return false;
        ComputeDispatch cmd;
        cmd.kernel = kernel;
        cmd.groups.groups_x = groups;
        cmd.buffers = bufs;
        cmd.buffer_count = bufCount;
        cmd.constants = pc;
        cmd.constants_size = pcSize;
        ok = backend->dispatch(cmd);
        if (!ok) return false;
        ++pending;
        pendingWork += workCells;
        if (pending >= kFlushInterval || pendingWork >= kWorkPerSubmission) flush();
        return true;
    }
};

uint32_t groupsFor(int cells) { return (uint32_t)((cells + 255) / 256); }

// One pyramid level. Level 0 borrows the caller's and the state's buffers;
// coarser levels own theirs.
struct Level {
    int w = 0, h = 0;
    ComputeBufferHandle height, filledA, filledB, weights, rain, areaA, areaB;
    bool owned = false;
    int cells() const { return w * h; }
};

// Two uints of packed byte weights per cell.
size_t weightBytes(int cells) { return (size_t)cells * 2u * sizeof(uint32_t); }

// Exact priority-flood depression fill (Barnes et al.). The GPU relaxes toward
// this answer; here it is computed outright, which is why the CPU path is the
// reference for lake extent.
//
// ★★★ The ladder is DITHERED, and that is not cosmetic.
//
// A bare flood fills a flat with `parent + eps` and the queue breaks ties by
// linear index, so the front sweeps a flat in row-major order and the epsilon
// ramp comes out an almost perfect inclined PLANE. Steepest descent over a
// plane sends every cell the same way: dense parallel straight channels that
// never merge - the "power line" artifact, which reads as a printed circuit
// board over the valley floors while the sloping ground either side looks
// perfect (real gradient, no ramp, no artifact).
//
// The dither is applied to the ladder's MAGNITUDE, never to the surface.
// Perturbing the surface is the obvious move and it is wrong: it creates
// dither pits mid-flat, cells with no strictly lower neighbour, which
// buildWeights then writes as an all-zero row - a terminal sink that cuts the
// river in the middle of the plain. Scaling the step keeps every cell at least
// one full ladder above its parent, so a strictly lower neighbour always
// exists, while the varying step breaks the plane and lets paths merge.
//
// The sibling WatershedAnalysisNode carries the same term, and its comment
// records that a raw LCG product kept a constant difference between adjacent
// indices and laid sawtooth ramps that steered drainage into repeating
// diagonal sweeps. Hence an avalanche mix, not a multiply.
void priorityFlood(const std::vector<float>& height, int w, int h,
                   std::vector<float>& filled) {
    filled = height;
    std::vector<uint8_t> visited((size_t)w * h, 0u);
    using Cell = std::pair<float, int>;
    std::priority_queue<Cell, std::vector<Cell>, std::greater<Cell>> open;
    auto seed = [&](int i) {
        if (!visited[i]) { visited[i] = 1u; open.push({ filled[i], i }); }
    };
    for (int x = 0; x < w; ++x) { seed(x); seed((h - 1) * w + x); }
    for (int y = 1; y < h - 1; ++y) { seed(y * w); seed(y * w + w - 1); }

    while (!open.empty()) {
        const auto [level, i] = open.top(); open.pop();
        const int x = i % w, y = i / w;
        for (int d = 0; d < 8; ++d) {
            const int nx = x + kNeighborX[d], ny = y + kNeighborY[d];
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            const int ni = ny * w + nx;
            if (visited[ni]) continue;
            visited[ni] = 1u;
            // The ladder has to stay above float resolution AT THE WORKING
            // HEIGHT. kFillEpsilon is about three ulp near h = 0.5 but only
            // 1.7 near h = 1, and over a wide basin `level + eps == level`
            // ends the ladder entirely: every cell of the basin lands on one
            // value, buildWeights finds no strictly lower neighbour, leaves an
            // all-zero row, and the basin becomes a terminal sink for drainage
            // area. Flow then dies at the rim - on the CPU reference too,
            // which is why both paths cut flow at the same kind of boundary.
            const float ladder = (std::max)(kFillEpsilon,
                                            std::fabs(level) * 4.0f * 1.1920929e-7f);
            uint32_t tie = (uint32_t)ni;
            tie ^= tie >> 16; tie *= 0x7feb352du;
            tie ^= tie >> 15; tie *= 0x846ca68bu;
            tie ^= tie >> 16;
            const float jitter = 1.0f + 0.5f * ((float)(tie & 0xffffu) / 65535.0f);
            filled[ni] = (std::max)(height[ni], level + ladder * jitter);
            open.push({ filled[ni], ni });
        }
    }
}

// Normalized MFD outflow weights, slope^1.5, on the conditioned surface. The
// table is built once per drainage refresh and then reused by both the exact
// accumulation sweep and the Jacobi sediment routing, so the two cannot drift
// apart in the way a recomputed-per-consumer weight would.
// `bed` (optional) is the LIVE surface. When supplied the weights are derived
// from max(filled, bed) instead of the conditioned surface, which is what makes
// avulsion possible -- see the long note in terrain_lem_weights.comp. Where
// nothing has aggraded the two are identical by construction, so passing it
// cannot perturb an un-deposited landscape.
void buildWeights(const std::vector<float>& filled, int w, int h, float cellSize,
                  std::vector<float>& weights, const std::vector<float>* bed = nullptr) {
    weights.assign((size_t)w * h * 8, 0.0f);
    auto surface = [&](int idx) {
        const float f = filled[(size_t)idx];
        return bed ? (std::max)(f, (*bed)[(size_t)idx]) : f;
    };
#pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int i = y * w + x;
            float power[8]{};
            float total = 0.0f;
            for (int d = 0; d < 8; ++d) {
                const int nx = x + kNeighborX[d], ny = y + kNeighborY[d];
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                const float slope = (surface(i) - surface(ny * w + nx)) /
                                    (kNeighborDist[d] * (std::max)(cellSize, 1.0e-3f));
                if (slope > 0.0f) { power[d] = std::pow(slope, 1.5f); total += power[d]; }
            }
            if (total > 0.0f)
                for (int d = 0; d < 8; ++d) weights[(size_t)i * 8 + d] = power[d] / total;
        }
    }
}

// Exact accumulation: one sweep in descending conditioned order.
void accumulateExact(const std::vector<float>& filled, const std::vector<float>& weights,
                     const std::vector<float>& rainArea, int w, int h,
                     std::vector<int>& order, std::vector<float>& area) {
    const int n = w * h;
    // ★★★ Kahn over the MFD graph, NOT a sort by conditioned height.
    //
    // Sorting looks equivalent and is not, and the difference is measurable:
    // priority-flood raises a pit by kFillEpsilon per ring, and over a wide
    // basin that ladder falls under float resolution (0.5f + 1e-7f == 0.5f),
    // so every cell of the basin compares EQUAL. std::sort then orders them by
    // whatever the introsort happens to do, a river's accumulated area is
    // handed to cells that were already visited, and it never re-emerges past
    // the basin. Flow dies at the basin rim.
    //
    // On the scene that exposed this, a quarter of the map was inside filled
    // depressions and the largest catchment came out at 3 % of a 1 km terrain -
    // there was no trunk river anywhere, on either the CPU or the GPU path.
    //
    // The sibling FlowMaskNode was fixed this way in 2026; the LEM reference
    // was not, and promoting it to be the GPU's authority made both paths wrong
    // in the same way instead of one.
    //
    // The graph is a DAG by construction - a weight exists only where the
    // slope is STRICTLY positive, so no two cells can feed each other - which
    // is what makes Kahn exact regardless of how many cells tie in height.
    std::vector<int> indegree((size_t)n, 0);
    for (int i = 0; i < n; ++i) {
        const int x = i % w, y = i / w;
        for (int d = 0; d < 8; ++d) {
            if (weights[(size_t)i * 8 + d] <= 0.0f) continue;
            const int nx = x + kNeighborX[d], ny = y + kNeighborY[d];
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            ++indegree[(size_t)(ny * w + nx)];
        }
    }
    order.clear();
    order.reserve((size_t)n);
    for (int i = 0; i < n; ++i) if (indegree[(size_t)i] == 0) order.push_back(i);
    for (size_t cursor = 0; cursor < order.size(); ++cursor) {
        const int i = order[cursor];
        const int x = i % w, y = i / w;
        for (int d = 0; d < 8; ++d) {
            if (weights[(size_t)i * 8 + d] <= 0.0f) continue;
            const int nx = x + kNeighborX[d], ny = y + kNeighborY[d];
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            const int ni = ny * w + nx;
            if (--indegree[(size_t)ni] == 0) order.push_back(ni);
        }
    }
    if ((int)order.size() != n) {
        // Cannot happen while the strict-slope invariant holds. If it ever
        // does, say so instead of silently dropping the unemitted cells'
        // catchment, which would read as a plausible but too-small river.
        SCENE_LOG_WARN("[LEM] accumulation order covered " + std::to_string(order.size()) +
                       " of " + std::to_string(n) + " cells; the weight graph has a cycle.");
        std::vector<uint8_t> emitted((size_t)n, 0u);
        for (int i : order) emitted[(size_t)i] = 1u;
        for (int i = 0; i < n; ++i) if (!emitted[(size_t)i]) order.push_back(i);
    }
    area = rainArea;
    for (int i : order) {
        const int x = i % w, y = i / w;
        const float have = area[i];
        if (have <= 0.0f) continue;
        for (int d = 0; d < 8; ++d) {
            const float weight = weights[(size_t)i * 8 + d];
            if (weight <= 0.0f) continue;
            const int nx = x + kNeighborX[d], ny = y + kNeighborY[d];
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            area[ny * w + nx] += have * weight;
        }
    }
}


// Condition + route + accumulate as ONE step. Three call sites used to repeat
// this sequence and a flat-routing pass added to only some of them would be
// invisible: the odd one out would keep the radial fan and nothing would say so.
void conditionAndAccumulate(const std::vector<float>& height, int w, int h,
                            float cellSize, const std::vector<float>& rainArea,
                            std::vector<float>& filled, std::vector<float>& weights,
                            std::vector<int>& order, std::vector<float>& area) {
    priorityFlood(height, w, h, filled);
    buildWeights(filled, w, h, cellSize, weights);
    accumulateExact(filled, weights, rainArea, w, h, order, area);
}

}  // namespace

// ===========================================================================
// GPU path
// ===========================================================================

bool createGpuState(ISimulationComputeBackend* backend, int width, int height,
                    GpuState& state) {
    if (!backend || width <= 2 || height <= 2) return false;
    const int cells = width * height;
    ComputeBufferDesc d;
    d.size_bytes = (size_t)cells * sizeof(float);

    state.filledA = backend->createBuffer(d);
    state.filledB = backend->createBuffer(d);
    ComputeBufferDesc wd;
    wd.size_bytes = (size_t)cells * 2u * sizeof(uint32_t);
    state.weights = backend->createBuffer(wd);
    state.rainArea = backend->createBuffer(d);
    state.areaA = backend->createBuffer(d);
    state.areaB = backend->createBuffer(d);
    state.fluxA = backend->createBuffer(d);
    state.fluxB = backend->createBuffer(d);
    state.alluviumA = backend->createBuffer(d);
    state.alluviumB = backend->createBuffer(d);
    state.exported = backend->createBuffer(d);
    state.lakeDepth = backend->createBuffer(d);
    state.erosionLedger = backend->createBuffer(d);
    state.depositionLedger = backend->createBuffer(d);
    state.width = width;
    state.height = height;
    state.valid = state.filledA.valid() && state.filledB.valid() &&
                  state.weights.valid() && state.rainArea.valid() &&
                  state.areaA.valid() && state.areaB.valid() && state.fluxA.valid() &&
                  state.fluxB.valid() && state.alluviumA.valid() && state.alluviumB.valid() &&
                  state.exported.valid() && state.lakeDepth.valid() &&
                  state.erosionLedger.valid() && state.depositionLedger.valid();

    if (state.valid) {
        const std::vector<float> zeros((size_t)cells, 0.0f);
        const size_t bytes = (size_t)cells * sizeof(float);
        state.valid = backend->uploadBuffer(state.fluxA, zeros.data(), bytes) &&
                      backend->uploadBuffer(state.fluxB, zeros.data(), bytes) &&
                      backend->uploadBuffer(state.alluviumA, zeros.data(), bytes) &&
                      backend->uploadBuffer(state.alluviumB, zeros.data(), bytes) &&
                      backend->uploadBuffer(state.exported, zeros.data(), bytes) &&
                      backend->uploadBuffer(state.areaA, zeros.data(), bytes) &&
                      backend->uploadBuffer(state.areaB, zeros.data(), bytes) &&
                      backend->uploadBuffer(state.lakeDepth, zeros.data(), bytes) &&
                      backend->uploadBuffer(state.erosionLedger, zeros.data(), bytes) &&
                      backend->uploadBuffer(state.depositionLedger, zeros.data(), bytes);
    }
    if (!state.valid) destroyGpuState(backend, state);
    return state.valid;
}

void destroyGpuState(ISimulationComputeBackend* backend, GpuState& state) {
    if (!backend) return;
    auto kill = [&](ComputeBufferHandle& h) { if (h.valid()) backend->destroyBuffer(h); h = {}; };
    kill(state.filledA); kill(state.filledB); kill(state.weights); kill(state.rainArea);
    kill(state.areaA); kill(state.areaB);
    kill(state.fluxA); kill(state.fluxB);
    kill(state.alluviumA); kill(state.alluviumB);
    kill(state.exported); kill(state.lakeDepth);
    kill(state.erosionLedger); kill(state.depositionLedger);
    state.valid = false;
}

bool runGpu(ISimulationComputeBackend* backend, const HydraulicErosionParams& p,
            int width, int height, float cellSize, float heightScale,
            int iterations, bool publishFields, GpuFields& fields, GpuState& state) {
    // iterations == 0 is the publish-only path: re-condition the final
    // surface and emit hydrology without applying a second morphology cycle.
    if (!backend || !state.valid || iterations < 0 || (iterations == 0 && !publishFields)) return false;
    if (state.width != width || state.height != height) return false;

    const int cells = width * height;
    const uint32_t fullGroups = groupsFor(cells);
    const Derived d = derive(p, cellSize, heightScale, iterations);

    Batch batch;
    batch.backend = backend;

    // ---- Build the cascadic pyramid ------------------------------------
    std::vector<Level> levels;
    levels.push_back(Level{ width, height, fields.height, state.filledA, state.filledB,
                            state.weights, state.rainArea, state.areaA, state.areaB, false });
    const int coarsest = std::clamp(p.drainageCoarsestSize, 32, 512);
    {
        int cw = width, ch = height;
        while ((std::max)(cw, ch) > coarsest && levels.size() < 10) {
            cw = (std::max)((cw + 1) / 2, 8);
            ch = (std::max)((ch + 1) / 2, 8);
            Level lv; lv.w = cw; lv.h = ch; lv.owned = true;
            ComputeBufferDesc ld; ld.size_bytes = (size_t)(cw * ch) * sizeof(float);
            ComputeBufferDesc lw; lw.size_bytes = weightBytes(cw * ch);
            lv.height = backend->createBuffer(ld);
            lv.filledA = backend->createBuffer(ld);
            lv.filledB = backend->createBuffer(ld);
            lv.weights = backend->createBuffer(lw);
            lv.rain = backend->createBuffer(ld);
            lv.areaA = backend->createBuffer(ld);
            lv.areaB = backend->createBuffer(ld);
            if (!lv.height.valid() || !lv.filledA.valid() || !lv.filledB.valid() ||
                !lv.weights.valid() || !lv.rain.valid() || !lv.areaA.valid() || !lv.areaB.valid()) {
                SCENE_LOG_WARN("[LEM] Could not allocate pyramid level " +
                               std::to_string(levels.size()) + "; drainage solve aborted.");
                for (auto& b : { lv.height, lv.filledA, lv.filledB, lv.weights,
                                 lv.rain, lv.areaA, lv.areaB })
                    if (b.valid()) backend->destroyBuffer(b);
                lv = Level{};
                break;
            }
            levels.push_back(lv);
            if ((int)levels.size() >= 10) break;
        }
    }
    auto releaseLevels = [&]() {
        for (auto& lv : levels) {
            if (!lv.owned) continue;
            for (auto& b : { lv.height, lv.filledA, lv.filledB, lv.weights,
                             lv.rain, lv.areaA, lv.areaB })
                if (b.valid()) backend->destroyBuffer(b);
            lv.owned = false;
        }
    };

    const int deepest = (int)levels.size() - 1;
    std::vector<float> infinity((size_t)levels[deepest].cells(), 1.0e18f);

    // ---- Drainage solve -------------------------------------------------
    auto solveDrainage = [&]() -> bool {
        // The morphology passes ping-pong the height buffer, so level 0 must be
        // re-pointed at whichever half currently holds the live surface. Getting
        // this wrong conditions the PREVIOUS surface, which looks entirely
        // plausible and is off by one whole iteration.
        levels[0].height = fields.height;

        RainPc rainPc{ width, height, d.cellArea, (std::max)(p.orographicRain, 0.0f),
                       d.windX, d.windY, cellSize, heightScale };
        {
            const ComputeBufferHandle bufs[2] = { levels[0].height, levels[0].rain };
            if (!batch.run("terrain_lem_rain", fullGroups, bufs, 2, &rainPc, sizeof(rainPc), (uint64_t)cells))
                return false;
        }

        for (int l = 1; l <= deepest; ++l) {
            RestrictPc hPc{ levels[l - 1].w, levels[l - 1].h, levels[l].w, levels[l].h, 0 };
            const ComputeBufferHandle hb[2] = { levels[l - 1].height, levels[l].height };
            if (!batch.run("terrain_lem_restrict", groupsFor(levels[l].cells()), hb, 2,
                           &hPc, sizeof(hPc), (uint64_t)levels[l].cells())) return false;
            RestrictPc rPc{ levels[l - 1].w, levels[l - 1].h, levels[l].w, levels[l].h, 1 };
            const ComputeBufferHandle rb[2] = { levels[l - 1].rain, levels[l].rain };
            if (!batch.run("terrain_lem_restrict", groupsFor(levels[l].cells()), rb, 2,
                           &rPc, sizeof(rPc), (uint64_t)levels[l].cells())) return false;
        }

        // Coarsest level: Planchon-Darboux from +infinity, run to convergence.
        batch.flush();
        infinity.assign((size_t)levels[deepest].cells(), 1.0e18f);
        if (!backend->uploadBuffer(levels[deepest].filledA, infinity.data(),
                                   infinity.size() * sizeof(float))) return false;
        // Submit the staged transfer before any kernel reads it: the per-
        // dispatch barrier this backend emits is COMPUTE -> COMPUTE and does
        // not cover a pending TRANSFER write.
        backend->synchronize();

        for (int l = deepest; l >= 0; --l) {
            Level& lv = levels[l];
            const uint32_t g = groupsFor(lv.cells());
            int passes;
            if (l == deepest) {
                passes = std::clamp((std::max)(lv.w, lv.h) * 2, 128, 1024);
            } else {
                // Seed from the coarser solve. Nearest lift + max against own
                // height keeps the seed a valid UPPER bound of the true filled
                // surface, which is what lets a decreasing-only relaxation
                // start here instead of at +infinity.
                ProlongatePc pPc{ levels[l + 1].w, levels[l + 1].h, lv.w, lv.h, 0 };
                const ComputeBufferHandle pb[3] = { levels[l + 1].filledA, lv.height, lv.filledA };
                if (!batch.run("terrain_lem_prolongate", g, pb, 3, &pPc, sizeof(pPc), (uint64_t)lv.cells()))
                    return false;
                // Each finer level starts from the level above, so it only has
                // to correct fine-scale detail -- and it costs four times as
                // much per pass. A flat budget spent most of the solve on the
                // level that needed it least.
                passes = std::clamp(p.drainageFillPasses >> (deepest - 1 - l),
                                    12, (std::max)(p.drainageFillPasses, 12));
            }
            FillPc fPc{ lv.w, lv.h, kFillEpsilon, 0.0f };
            for (int pass = 0; pass < passes; ++pass) {
                const ComputeBufferHandle fb[3] = { lv.height, lv.filledA, lv.filledB };
                if (!batch.run("terrain_flow_fill", g, fb, 3, &fPc, sizeof(fPc), (uint64_t)lv.cells())) return false;
                std::swap(lv.filledA, lv.filledB);
            }
            // The conditioned surface is final for this level, so tabulate its
            // outflow weights once instead of re-deriving them inside every
            // accumulation and routing pass.
            // useBedMax = 0: the drainage solve must see the CONDITIONED
            // surface. Avulsion routing is a morphology-loop concern and
            // feeding it in here would let a bar in one basin re-route the
            // accumulation of the whole pyramid.
            WeightsPc wPc{ lv.w, lv.h, cellSize, 0 };
            const ComputeBufferHandle wb[3] = { lv.filledA, lv.weights, lv.height };
            if (!batch.run("terrain_lem_weights", g, wb, 3, &wPc, sizeof(wPc), (uint64_t)lv.cells())) return false;
        }

        // Accumulation, coarse to fine. Same cascadic argument, except the seed
        // needs no bound property: area is in physical m^2 so a bilinear lift is
        // already the right magnitude at the finer level.
        for (int l = deepest; l >= 0; --l) {
            Level& lv = levels[l];
            const uint32_t g = groupsFor(lv.cells());
            if (l == deepest) {
                // Same-size "restrict" with sum pooling is a copy; this seeds
                // the fixed point at A = rain rather than adding a copy kernel.
                RestrictPc cPc{ lv.w, lv.h, lv.w, lv.h, 1 };
                const ComputeBufferHandle cb[2] = { lv.rain, lv.areaA };
                if (!batch.run("terrain_lem_restrict", g, cb, 2, &cPc, sizeof(cPc), (uint64_t)lv.cells()))
                    return false;
            } else {
                ProlongatePc pPc{ levels[l + 1].w, levels[l + 1].h, lv.w, lv.h, 1 };
                const ComputeBufferHandle pb[3] = { levels[l + 1].areaA, levels[l + 1].areaA,
                                                    lv.areaA };
                if (!batch.run("terrain_lem_prolongate", g, pb, 3, &pPc, sizeof(pPc), (uint64_t)lv.cells()))
                    return false;
            }
            const int passes = (l == deepest)
                ? std::clamp((std::max)(lv.w, lv.h) * 2, 128, 1024)
                : std::clamp(p.drainageAccumulatePasses >> (deepest - 1 - l),
                             16, (std::max)(p.drainageAccumulatePasses, 16));
            AccumulatePc aPc{ lv.w, lv.h };
            for (int pass = 0; pass < passes; ++pass) {
                const ComputeBufferHandle ab[4] = { lv.weights, lv.rain, lv.areaA, lv.areaB };
                if (!batch.run("terrain_lem_accumulate", g, ab, 4, &aPc, sizeof(aPc), (uint64_t)lv.cells()))
                    return false;
                std::swap(lv.areaA, lv.areaB);
            }
        }
        // Level 0 borrows the state's buffers; publish whichever half won.
        state.filledA = levels[0].filledA; state.filledB = levels[0].filledB;
        state.weights = levels[0].weights;
        state.areaA = levels[0].areaA;     state.areaB = levels[0].areaB;
        return true;
    };

    // ---- Morphology -----------------------------------------------------
    const int refresh = std::clamp(p.drainageRefreshInterval, 1, 64);
    const int routeSteps = std::clamp(p.sedimentRouteSteps, 0, 4096);
    const int talusSteps = p.massWasting ? std::clamp(p.massWastingSteps, 0, 64) : 0;

    IncisePc incisePc{ width, height, cellSize, heightScale, d.dt,
                       (std::max)(p.incisionK, 0.0f), std::clamp(p.streamPowerM, 0.0f, 2.0f),
                       std::clamp(p.streamPowerN, 0.1f, 4.0f),
                       (std::max)(p.slopeMin, 1.0e-6f), (std::max)(p.slopeMax, 1.0e-3f),
                       std::clamp(p.sedimentCover, 0.0f, 1.0f), (std::max)(p.transportK, 0.0f),
                       std::clamp(p.incisionSafety, 0.0f, 0.95f), d.maxStep, d.lakeEps };
    RoutePc routePc{ width, height, cellSize, heightScale,
                     (std::max)(p.settlingVelocity, 0.0f), (std::max)(p.rainRate, 1.0e-6f),
                     std::clamp(p.depositionSafety, 0.0f, 0.95f), d.minDischarge,
                     d.lakeEps, d.depositFloor, d.maxBuildNorm };
    const int avulsion = (std::max)(p.avulsionInterval, 0);
    const int alluviumSteps = std::clamp(p.alluviumSteps, 0, 64);
    WeightsPc avulsionPc{ width, height, cellSize, 1 };
    AlluviumPc alluviumPc{ width, height, cellSize, heightScale, d.tanAlluvium,
                           std::clamp(p.alluviumRate, 0.0f, 1.0f), d.maxStep,
                           std::clamp(p.alluviumConsolidation, 0.0f, 1.0f) };
    TalusPc talusPc{ width, height, cellSize, heightScale, d.tanRepose,
                     std::clamp(p.massWastingRate, 0.0f, 1.0f), 0.6f, d.maxStep };
    DiffusePc diffusePc{ width, height, d.diffuseCoeff, d.channelRefArea, d.lakeEps };

    bool ok = true;
    for (int it = 0; ok && it < iterations; ++it) {
        if (it % refresh == 0) {
            ok = solveDrainage();
            if (!ok) break;
        }

        {
            const ComputeBufferHandle bufs[9] = { fields.height, fields.heightAlt,
                                                  state.filledA, state.areaA,
                                                  fields.hardness, fields.mask,
                                                  state.fluxA, state.fluxB, fields.erosion };
            ok = batch.run("terrain_lem_incise", fullGroups, bufs, 9, &incisePc, sizeof(incisePc), (uint64_t)cells);
            std::swap(state.fluxA, state.fluxB);
            std::swap(fields.height, fields.heightAlt);
        }

        for (int s = 0; ok && s < routeSteps; ++s) {
            // ★★★ Avulsion: re-derive the flow directions from the LIVE bed.
            // The weights the drainage solve produced describe a bed that no
            // longer exists once incision and this loop's own deposition have
            // run, and a channel that keeps being fed after it has silted up
            // is the reason deposits came out as ridges instead of fans.
            // Discharge (areaA) stays frozen on purpose: a bar growing in a
            // channel changes where the water goes, not how much of it there
            // is, and re-solving accumulation here would cost the whole
            // multigrid pass per route step.
            if (avulsion > 0 && (s % avulsion) == 0) {
                const ComputeBufferHandle ab[3] = { state.filledA, state.weights, fields.height };
                ok = batch.run("terrain_lem_weights", fullGroups, ab, 3,
                               &avulsionPc, sizeof(avulsionPc), (uint64_t)cells);
                if (!ok) break;
            }

            const ComputeBufferHandle bufs[10] = { state.weights, state.filledA, state.areaA,
                                                   state.fluxA, state.fluxB,
                                                   fields.height, fields.heightAlt,
                                                   fields.deposition, state.exported,
                                                   state.alluviumA };
            ok = batch.run("terrain_lem_route", fullGroups, bufs, 10, &routePc, sizeof(routePc), (uint64_t)cells * 2u);
            std::swap(state.fluxA, state.fluxB);
            std::swap(fields.height, fields.heightAlt);
        }

        // --- Alluvial spreading ---------------------------------------
        // Runs after transport and before talus: the water drops its load,
        // the fresh deposit relaxes to a fan slope, and only then does dry
        // mass wasting act on the resulting surface.
        for (int s = 0; ok && s < alluviumSteps; ++s) {
            const ComputeBufferHandle bufs[6] = { fields.height, fields.heightAlt,
                                                  state.alluviumA, state.alluviumB,
                                                  fields.mask, fields.deposition };
            ok = batch.run("terrain_lem_alluvium", fullGroups, bufs, 6,
                           &alluviumPc, sizeof(alluviumPc), (uint64_t)cells * 8u);
            std::swap(fields.height, fields.heightAlt);
            std::swap(state.alluviumA, state.alluviumB);
        }

        for (int s = 0; ok && s < talusSteps; ++s) {
            const ComputeBufferHandle bufs[6] = { fields.height, fields.heightAlt,
                                                  fields.hardness, fields.mask,
                                                  fields.erosion, fields.deposition };
            ok = batch.run("terrain_lem_talus", fullGroups, bufs, 6, &talusPc, sizeof(talusPc), (uint64_t)cells * 8u);
            std::swap(fields.height, fields.heightAlt);
        }

        for (int s = 0; ok && s < d.diffuseSteps; ++s) {
            const ComputeBufferHandle bufs[5] = { fields.height, fields.heightAlt,
                                                  state.areaA, fields.mask, state.filledA };
            ok = batch.run("terrain_lem_diffuse", fullGroups, bufs, 5, &diffusePc, sizeof(diffusePc), (uint64_t)cells);
            std::swap(fields.height, fields.heightAlt);
        }
    }

    // ---- Publish --------------------------------------------------------
    if (ok && publishFields) {
        // Re-condition on the FINAL surface. Without this the lake and channel
        // fields describe the surface as it was `refresh` iterations ago, which
        // is exactly the sort of plausible-looking stale reading that never
        // gets reported as a bug.
        ok = solveDrainage();
        if (ok) {
            FinalizePc finPc{ width, height, cellSize, heightScale,
                              (std::max)(p.rainRate, 1.0e-6f),
                              (std::max)(p.fluvialWidthScale, 0.0f),
                              (std::max)(p.fluvialDepthScale, 0.0f),
                              (std::max)(p.fluvialHeadwaterAreaKm2, 1.0e-6f),
                              d.lakeEps, 1.0f };
            const ComputeBufferHandle bufs[10] = { fields.height, state.filledA, state.areaA,
                                                   fields.discharge, fields.channelWidth,
                                                   fields.waterDepth, fields.waterLevel,
                                                   fields.directionX, fields.directionY,
                                                   state.lakeDepth };
            ok = batch.run("terrain_lem_finalize", fullGroups, bufs, 10, &finPc, sizeof(finPc), (uint64_t)cells);
        }
    }

    batch.flush();
    releaseLevels();
    if (!ok) SCENE_LOG_WARN("[LEM] GPU cycle failed; caller will fall back.");
    return ok;
}

// ===========================================================================
// CPU reference
// ===========================================================================
void runCpu(TerrainObject* terrain, const HydraulicErosionParams& p,
            const std::vector<float>& mask, int iterations, bool publishFields,
            HydraulicErosionFields* fields, HydraulicErosionStats& stats,
            const std::function<void(float)>& progressCallback) {
    // iterations == 0 is a deliberate publish-only solve. It is used after
    // droplet/post-processing geometry so the exported flow fields describe
    // the final terrain without integrating fluvialTimeStep a second time.
    if (!terrain || iterations < 0 || (iterations == 0 && !publishFields)) return;
    const int w = terrain->heightmap.width;
    const int h = terrain->heightmap.height;
    if (w <= 2 || h <= 2) return;
    const int n = w * h;
    std::vector<float>& data = terrain->heightmap.data;
    if ((int)data.size() != n) return;

    const float cellSize = (std::max)(terrain->heightmap.scale_xz / (float)w, 1.0e-3f);
    const float heightScale = (std::max)((float)terrain->heightmap.scale_y, 1.0e-3f);
    const Derived d = derive(p, cellSize, heightScale, iterations);

    const bool hasHardness = terrain->hardnessMap.size() == (size_t)n;
    const bool hasMask = mask.size() == (size_t)n;
    auto hardnessAt = [&](int i) { return hasHardness ? std::clamp(terrain->hardnessMap[i], 0.0f, 1.0f) : 0.0f; };
    auto maskAt = [&](int i) { return hasMask ? std::clamp(mask[i], 0.0f, 1.0f) : 1.0f; };

    std::vector<float> filled, weights, area, rainArea((size_t)n, d.cellArea);
    std::vector<float> flux((size_t)n, 0.0f), fluxNext((size_t)n, 0.0f);
    std::vector<float> exported((size_t)n, 0.0f), scratch((size_t)n, 0.0f);
    std::vector<int> order;

    std::vector<float> erosion((size_t)n, 0.0f), deposition((size_t)n, 0.0f);
    // Loose material available to alluvial spreading. Mirrors state.alluviumA
    // on the GPU and is deliberately separate from the deposition ledger.
    std::vector<float> alluvium((size_t)n, 0.0f), alluviumNext((size_t)n, 0.0f);

    const float orographic = (std::max)(p.orographicRain, 0.0f);
    auto computeRain = [&]() {
        if (orographic <= 0.0f) { rainArea.assign((size_t)n, d.cellArea); return; }
#pragma omp parallel for
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int i = y * w + x;
                float rain = 1.0f;
                if (x > 0 && y > 0 && x < w - 1 && y < h - 1) {
                    const float inv = 1.0f / cellSize;
                    const float gx = (data[i + 1] - data[i - 1]) * 0.5f * heightScale * inv;
                    const float gy = (data[i + w] - data[i - w]) * 0.5f * heightScale * inv;
                    const float ascent = std::clamp((d.windX * gx + d.windY * gy) * 4.0f, -1.0f, 1.0f);
                    rain = std::clamp(1.0f + orographic * ascent, 0.05f, 4.0f);
                }
                rainArea[i] = rain * d.cellArea;
            }
        }
    };

    const int refresh = std::clamp(p.drainageRefreshInterval, 1, 64);
    const int routeSteps = std::clamp(p.sedimentRouteSteps, 0, 4096);
    const int talusSteps = p.massWasting ? std::clamp(p.massWastingSteps, 0, 64) : 0;
    const float exponentM = std::clamp(p.streamPowerM, 0.0f, 2.0f);
    const float exponentN = std::clamp(p.streamPowerN, 0.1f, 4.0f);
    const float slopeMin = (std::max)(p.slopeMin, 1.0e-6f);
    const float slopeMax = (std::max)(p.slopeMax, 1.0e-3f);
    const float cover = std::clamp(p.sedimentCover, 0.0f, 1.0f);
    const float inciseSafety = std::clamp(p.incisionSafety, 0.0f, 0.95f);
    const float depositSafety = std::clamp(p.depositionSafety, 0.0f, 0.95f);
    const float talusRate = std::clamp(p.massWastingRate, 0.0f, 1.0f);
    const float rainRate = (std::max)(p.rainRate, 1.0e-6f);
    const int avulsion = (std::max)(p.avulsionInterval, 0);
    const int alluviumSteps = std::clamp(p.alluviumSteps, 0, 64);
    const float alluviumRate = std::clamp(p.alluviumRate, 0.0f, 1.0f);
    const float alluviumKeep = 1.0f - std::clamp(p.alluviumConsolidation, 0.0f, 1.0f);

    for (int it = 0; it < iterations; ++it) {
        if (progressCallback) progressCallback((float)it / (float)iterations);
        if (it % refresh == 0) {
            computeRain();
            conditionAndAccumulate(data, w, h, cellSize, rainArea, filled, weights, order, area);
        }

        // --- Incision -----------------------------------------------------
#pragma omp parallel for
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int i = y * w + x;
                const float carried = (std::max)(flux[i], 0.0f);
                const float hCur = data[i];
                // Mirrors the shader: incision reads a frozen surface so the
                // anti-pit clamp cannot be handed a receiver height that has
                // already been lowered this pass.
                scratch[i] = hCur;
                if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) { fluxNext[i] = carried; continue; }
                if (filled[i] - hCur > d.lakeEps) { fluxNext[i] = carried; continue; }

                int receiver = -1;
                float bestGrade = 0.0f, receiverDist = 1.0f;
                for (int k = 0; k < 8; ++k) {
                    const int ni = (y + kNeighborY[k]) * w + (x + kNeighborX[k]);
                    const float grade = (filled[i] - filled[ni]) / kNeighborDist[k];
                    if (grade > bestGrade) { bestGrade = grade; receiver = ni; receiverDist = kNeighborDist[k]; }
                }
                if (receiver < 0) { fluxNext[i] = carried; continue; }

                const float run = (std::max)(receiverDist * cellSize, 1.0e-3f);
                const float dropNorm = (std::max)(hCur - data[receiver], 0.0f);
                const float slope = std::clamp(dropNorm * heightScale / run, slopeMin, slopeMax);
                const float areaKm2 = (std::max)(area[i], 0.0f) * 1.0e-6f;
                const float shape = std::pow((std::max)(areaKm2, 1.0e-9f), exponentM) *
                                    std::pow(slope, exponentN);

                float erode = p.incisionK * shape * d.dt / heightScale;
                if (cover > 0.0f) {
                    const float capacity = p.transportK * shape * d.dt / heightScale;
                    erode *= (capacity > 1.0e-9f)
                        ? std::clamp(1.0f - cover * carried / capacity, 0.0f, 1.0f) : 0.0f;
                }
                erode *= (1.0f - hardnessAt(i) * 0.9f) * maskAt(i);
                erode = (std::min)(erode, inciseSafety * dropNorm);
                erode = (std::min)(erode, d.maxStep);
                if (!(erode > 0.0f) || !std::isfinite(erode)) erode = 0.0f;

                scratch[i] = hCur - erode;
                erosion[i] += erode;
                fluxNext[i] = carried + erode;
            }
        }
        data.swap(scratch);
        flux.swap(fluxNext);

        // --- Sediment transport -------------------------------------------
        // Same Jacobi advection the GPU runs, with the same pass budget, so
        // the two paths move sediment the same distance per iteration. An
        // exact downstream sweep would be cheaper here but would silently make
        // the CPU produce deltas an iteration earlier than the GPU.
        for (int s = 0; s < routeSteps; ++s) {
            // Avulsion, same cadence and same routing surface as the GPU. The
            // drainage area is deliberately NOT re-accumulated here: aggradation
            // moves the flow path, not the size of the catchment feeding it.
            if (avulsion > 0 && (s % avulsion) == 0)
                buildWeights(filled, w, h, cellSize, weights, &data);
#pragma omp parallel for
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const int i = y * w + x;
                    float outWeight = 0.0f;
                    for (int k = 0; k < 8; ++k) outWeight += weights[(size_t)i * 8 + k];
                    const float selfRetain = std::clamp(1.0f - std::clamp(outWeight, 0.0f, 1.0f), 0.0f, 1.0f);
                    float arriving = (std::max)(flux[i], 0.0f) * selfRetain;
                    for (int k = 0; k < 8; ++k) {
                        const int nx = x + kNeighborX[k], ny = y + kNeighborY[k];
                        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                        const int ni = ny * w + nx;
                        arriving += (std::max)(flux[ni], 0.0f) * weights[(size_t)ni * 8 + (7 - k)];
                    }

                    const float hCur = data[i];
                    if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) {
                        scratch[i] = hCur;
                        fluxNext[i] = 0.0f;
                        exported[i] += arriving;
                        continue;
                    }

                    const float lakeDepth = (std::max)(filled[i] - hCur, 0.0f);
                    const float q = (std::max)(area[i], 0.0f) * rainRate / cellSize;
                    const float e = std::clamp(p.settlingVelocity * cellSize /
                                               (std::max)(q, d.minDischarge), 0.0f, 60.0f);
                    const float throughFlowDeposit =
                        std::clamp(1.0f - std::exp(-e), 0.0f, 1.0f);
                    float fdep = throughFlowDeposit;
                    if (lakeDepth > d.lakeEps) {
                        // Match terrain_lem_route.comp: a numerical or shallow
                        // hollow is a partial settling basin, while established
                        // standing water approaches complete capture.
                        const float lakeDepthMeters = lakeDepth * heightScale;
                        const float lakeEpsMeters = d.lakeEps * heightScale;
                        const float fullCaptureMeters = (std::max)(1.0f, lakeEpsMeters * 5.0f);
                        const float raw = std::clamp(
                            (lakeDepthMeters - lakeEpsMeters) /
                            (std::max)(fullCaptureMeters - lakeEpsMeters, 1.0e-6f),
                            0.0f, 1.0f);
                        const float lakeCapture = raw * raw * (3.0f - 2.0f * raw);
                        fdep = 1.0f - (1.0f - throughFlowDeposit) * (1.0f - lakeCapture);
                    }
                    float deposit = arriving * fdep;
                    float room;
                    if (lakeDepth > d.lakeEps) {
                        room = lakeDepth;
                    } else {
                        // See terrain_lem_route.comp: `rise` bounds aggradation
                        // against what feeds the cell, `fall` against what it
                        // drains into. Only the second one stops a deposit
                        // from building over its own outlet and manufacturing
                        // the pit that the fdep=1.0 lake branch then fills with
                        // the entire catchment's load.
                        float rise = 0.0f;
                        float fall = 0.0f;
                        for (int k = 0; k < 8; ++k) {
                            const float nh = data[(y + kNeighborY[k]) * w + (x + kNeighborX[k])];
                            rise = (std::max)(rise, nh - hCur);
                            fall = (std::max)(fall, hCur - nh);
                        }
                        room = depositSafety * (std::min)(rise, fall) + d.depositFloor;
                    }
                    // Bound TOTAL build-up, not just this pass. Without it the
                    // unconditional floor ratchets a flat cell upward forever.
                    if (d.maxBuildNorm > 0.0f)
                        room = (std::min)(room, (std::max)(d.maxBuildNorm - deposition[i], 0.0f));
                    deposit = std::clamp(deposit, 0.0f, (std::max)(room, 0.0f));
                    scratch[i] = hCur + deposit;
                    deposition[i] += deposit;
                    alluvium[i] += deposit;
                    fluxNext[i] = (std::max)(arriving - deposit, 0.0f);
                }
            }
            data.swap(scratch);
            flux.swap(fluxNext);
        }

        // --- Alluvial spreading --------------------------------------------
        // Fresh deposit relaxes toward a low, stable fan slope. Mirrors
        // terrain_lem_alluvium.comp exactly, including the two bounds that
        // make it a sediment process rather than a smoothing filter: only
        // LOOSE material may move, and never more of it than is there.
        for (int s = 0; s < alluviumSteps; ++s) {
            auto shedAlluvium = [&](int cx, int cy, int target) -> float {
                if (cx < 1 || cy < 1 || cx >= w - 1 || cy >= h - 1) return 0.0f;
                const int ci = cy * w + cx;
                const float loose = (std::max)(alluvium[(size_t)ci], 0.0f);
                if (loose <= 0.0f) return 0.0f;   // bedrock does not flow
                const float hc = data[ci];
                float sumExcess = 0.0f, maxExcess = 0.0f, want = 0.0f;
                for (int k = 0; k < 8; ++k) {
                    const int ni = (cy + kNeighborY[k]) * w + (cx + kNeighborX[k]);
                    const float stable = d.tanAlluvium * kNeighborDist[k] * cellSize / heightScale;
                    const float excess = (std::max)((hc - data[ni]) - stable, 0.0f);
                    sumExcess += excess;
                    maxExcess = (std::max)(maxExcess, excess);
                    if (k == target) want = excess;
                }
                if (sumExcess <= 0.0f) return 0.0f;
                const float total = (std::min)((std::min)(alluviumRate * 0.5f * maxExcess, d.maxStep),
                                               loose) * maskAt(ci);
                return target < 0 ? total : total * (want / sumExcess);
            };
#pragma omp parallel for
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const int i = y * w + x;
                    float inflow = 0.0f;
                    for (int k = 0; k < 8; ++k) {
                        const int nx = x + kNeighborX[k], ny = y + kNeighborY[k];
                        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                        inflow += shedAlluvium(nx, ny, 7 - k);
                    }
                    const float outflow = shedAlluvium(x, y, -1);
                    scratch[i] = data[i] + inflow - outflow;
                    const float relocated = inflow - outflow;
                    alluviumNext[i] = (std::max)(alluvium[(size_t)i] + relocated, 0.0f) * alluviumKeep;
                    // Relocate the spatial ledger with the height. Its global
                    // sum is unchanged, so mass conservation still closes,
                    // while deposit masks and build caps follow the fan.
                    deposition[(size_t)i] = (std::max)(deposition[(size_t)i] + relocated, 0.0f);
                }
            }
            data.swap(scratch);
            alluvium.swap(alluviumNext);
        }

        // --- Mass wasting --------------------------------------------------
        for (int s = 0; s < talusSteps; ++s) {
            auto reposeDrop = [&](int idx, float dist) {
                return d.tanRepose * (1.0f + hardnessAt(idx) * 0.6f) * dist * cellSize / heightScale;
            };
            auto shed = [&](int cx, int cy, int target) -> float {
                if (cx < 1 || cy < 1 || cx >= w - 1 || cy >= h - 1) return 0.0f;
                const int ci = cy * w + cx;
                const float hc = data[ci];
                float sumExcess = 0.0f, maxExcess = 0.0f, want = 0.0f;
                for (int k = 0; k < 8; ++k) {
                    const int ni = (cy + kNeighborY[k]) * w + (cx + kNeighborX[k]);
                    const float excess = (std::max)((hc - data[ni]) - reposeDrop(ci, kNeighborDist[k]), 0.0f);
                    sumExcess += excess;
                    maxExcess = (std::max)(maxExcess, excess);
                    if (k == target) want = excess;
                }
                if (sumExcess <= 0.0f) return 0.0f;
                const float total = (std::min)(talusRate * 0.5f * maxExcess, d.maxStep) * maskAt(ci);
                return target < 0 ? total : total * (want / sumExcess);
            };
#pragma omp parallel for
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const int i = y * w + x;
                    // Border cells never shed but must still accept inflow;
                    // see the note in terrain_lem_talus.comp -- dropping it
                    // leaks mass with no symptom but a ledger residual.
                    float inflow = 0.0f;
                    for (int k = 0; k < 8; ++k) {
                        const int nx = x + kNeighborX[k], ny = y + kNeighborY[k];
                        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                        inflow += shed(nx, ny, 7 - k);
                    }
                    const float outflow = shed(x, y, -1);
                    scratch[i] = data[i] + inflow - outflow;
                    if (outflow > 0.0f) erosion[i] += outflow;
                    if (inflow > 0.0f) deposition[i] += inflow;
                }
            }
            data.swap(scratch);
        }

        // --- Hillslope creep -----------------------------------------------
        for (int s = 0; s < d.diffuseSteps; ++s) {
#pragma omp parallel for
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    const int i = y * w + x;
                    const float hCur = data[i];
                    if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) { scratch[i] = hCur; continue; }
                    if (filled[i] - hCur > d.lakeEps) { scratch[i] = hCur; continue; }
                    const float lap = data[i - 1] + data[i + 1] + data[i - w] + data[i + w] - 4.0f * hCur;
                    const float channel = 1.0f / (1.0f + (std::max)(area[i], 0.0f) / d.channelRefArea);
                    scratch[i] = hCur + (std::min)(d.diffuseCoeff, 0.25f) * maskAt(i) * channel * lap;
                }
            }
            data.swap(scratch);
        }
    }

    // Re-condition on the final surface, but only when the hydrology fields are
    // actually going out (see the GPU path's note: stale fields look entirely
    // plausible). The main cycle publishes nothing, so paying for a full
    // priority-flood plus a topological sweep there was pure waste.
    if (publishFields) {
        computeRain();
        conditionAndAccumulate(data, w, h, cellSize, rainArea, filled, weights, order, area);
    }

    if (fields) {
        if ((int)fields->erosion.size() != n) fields->reset(w, h);
        for (int i = 0; i < n; ++i) {
            fields->erosion[i] += erosion[i];
            fields->deposition[i] += deposition[i];
        }
    }
    if (fields && publishFields) {
        const float widthScale = (std::max)(p.fluvialWidthScale, 0.0f);
        const float depthScale = (std::max)(p.fluvialDepthScale, 0.0f);
        const float headwater = (std::max)(p.fluvialHeadwaterAreaKm2, 1.0e-6f);
#pragma omp parallel for
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const int i = y * w + x;
                const float hCur = data[i];
                const float lake = (std::max)(filled[i] - hCur, 0.0f);
                const float areaM2 = (std::max)(area[i], 0.0f);
                const float areaKm2 = areaM2 * 1.0e-6f;
                const float discharge = areaM2 * rainRate;
                fields->discharge[i] = discharge;
                fields->drainageArea[i] = areaM2;
                fields->lakeDepth[i] = lake * heightScale;

                float t = std::clamp((areaKm2 - headwater) / (headwater * 4.0f), 0.0f, 1.0f);
                t = t * t * (3.0f - 2.0f * t);
                const float qref = (std::max)(discharge, 1.0e-6f);
                const float channelW = std::clamp(widthScale * 2.5f * std::sqrt(qref) * t,
                                                  0.0f, cellSize * 512.0f);
                const float channelD = std::clamp(depthScale * 0.30f * std::pow(qref, 0.40f) * t,
                                                  0.0f, heightScale);
                if (lake > d.lakeEps) {
                    fields->channelWidth[i] = (std::max)(channelW, cellSize);
                    fields->waterDepth[i] = lake * heightScale;
                    fields->waterLevel[i] = filled[i];
                } else {
                    fields->channelWidth[i] = channelW;
                    fields->waterDepth[i] = channelD;
                    fields->waterLevel[i] = hCur + channelD / heightScale;
                }

                float dirX = 0.0f, dirY = 0.0f;
                if (x > 0 && y > 0 && x < w - 1 && y < h - 1) {
                    float best = 0.0f;
                    for (int k = 0; k < 8; ++k) {
                        const int ni = (y + kNeighborY[k]) * w + (x + kNeighborX[k]);
                        const float grade = (filled[i] - filled[ni]) / kNeighborDist[k];
                        if (grade > best) {
                            best = grade;
                            dirX = kNeighborX[k] / kNeighborDist[k];
                            dirY = kNeighborY[k] / kNeighborDist[k];
                        }
                    }
                }
                fields->directionX[i] = dirX;
                fields->directionY[i] = dirY;
            }
        }
    }

    std::vector<float> lakeField((size_t)n, 0.0f);
    for (int i = 0; i < n; ++i) lakeField[i] = (std::max)(filled[i] - data[i], 0.0f) * heightScale;
    if (iterations > 0) {
        summarize(erosion, deposition, exported, flux, area, lakeField,
                  (std::max)(p.fluvialHeadwaterAreaKm2, 1.0e-6f), cellSize * cellSize,
                  heightScale, stats);
        stats.gpuPath = false;
        stats.cycleIterations = iterations;
    }
    if (fields) fields->stats = stats;
}

// ===========================================================================
// Diagnostics
// ===========================================================================

void summarize(const std::vector<float>& erosion, const std::vector<float>& deposition,
               const std::vector<float>& exported, const std::vector<float>& carried,
               const std::vector<float>& drainageArea, const std::vector<float>& lakeDepth,
               float headwaterAreaKm2, float cellAreaM2, float heightScaleMeters,
               HydraulicErosionStats& stats) {
    const size_t n = erosion.size();
    double eroded = 0.0, deposited = 0.0, out = 0.0, transit = 0.0;
    double maxArea = 0.0;
    long long lakeCells = 0, channelCells = 0, deepLakeCells = 0;
    double deepestLake = 0.0;
    // Deposit shape. `deposition` is in normalized units like `erosion`, so it
    // needs the height scale to become metres - the ledger sums above do not,
    // which is why the two are kept apart here rather than folded together.
    const double heightScaleM = (double)(std::max)(heightScaleMeters, 1.0e-3f);
    const double depositFloorMeters = 0.01;
    long long depositedCells = 0;
    double deepestDeposit = 0.0, depositedThickness = 0.0;
    const double headwaterM2 = (double)headwaterAreaKm2 * 1.0e6;
    // ★ An unthresholded counter is not an instrument. `lakeDepth > 0` counts
    // the conditioning ladder's few-ulp lift as a lake, so a quarter of a map
    // can read "flooded" and mean nothing. Ten centimetres of standing water is
    // the smallest thing worth calling a lake.
    const double lakeDepthFloorMeters = 0.10;

    for (size_t i = 0; i < n; ++i) {
        eroded += erosion[i];
        if (i < deposition.size()) deposited += deposition[i];
        if (i < exported.size()) out += exported[i];
        if (i < carried.size()) transit += carried[i];
        if (i < drainageArea.size()) {
            const double a = drainageArea[i];
            if (a > maxArea) maxArea = a;
            if (a >= headwaterM2) ++channelCells;
        }
        if (i < deposition.size() && deposition[i] > 0.0f) {
            const double thickness = (double)deposition[i] * heightScaleM;
            if (thickness > deepestDeposit) deepestDeposit = thickness;
            // Same discipline as the lake counters: an unthresholded count
            // would call one micron of settled load a deposit and report the
            // whole floodplain as covered.
            if (thickness > depositFloorMeters) {
                ++depositedCells;
                depositedThickness += thickness;
            }
        }
        if (i < lakeDepth.size() && lakeDepth[i] > 0.0f) {
            ++lakeCells;
            // ★ lakeDepth arrives in METRES from both callers - runCpu scales
            // by heightScale before the call, terrain_lem_finalize scales in
            // the shader. Multiplying again here would have made every lake
            // read hundreds of metres deep, which is exactly the "read the
            // consumer's unit" trap this file has hit before.
            const double depthMeters = (double)lakeDepth[i];
            if (depthMeters > deepestLake) deepestLake = depthMeters;
            if (depthMeters > lakeDepthFloorMeters) ++deepLakeCells;
        }
    }

    stats.eroded = eroded;
    stats.deposited = deposited;
    stats.exported = out;
    stats.carried = transit;
    stats.massError = eroded - (deposited + out + transit);
    stats.massErrorFraction = eroded > 1.0e-9 ? stats.massError / eroded : 0.0;
    stats.lakeCells = (int)(std::min)(lakeCells, (long long)INT_MAX);
    stats.lakeAreaFraction = n ? (float)((double)lakeCells / (double)n) : 0.0f;
    stats.drainageDensity = n ? (float)((double)channelCells / (double)n) : 0.0f;
    stats.maxDrainageAreaKm2 = (float)(maxArea * 1.0e-6);
    stats.deepLakeCells = (int)(std::min)(deepLakeCells, (long long)INT_MAX);
    stats.deepLakeAreaFraction = n ? (float)((double)deepLakeCells / (double)n) : 0.0f;
    stats.deepestLakeMeters = (float)deepestLake;
    stats.depositedCells = (int)(std::min)(depositedCells, (long long)INT_MAX);
    stats.depositedAreaFraction = n ? (float)((double)depositedCells / (double)n) : 0.0f;
    stats.deepestDepositMeters = (float)deepestDeposit;
    stats.meanDepositMeters = depositedCells
        ? (float)(depositedThickness / (double)depositedCells) : 0.0f;
    // ★★★ The decision-relevant number. maxDrainageAreaKm2 on its own cannot be
    // read: 0.03 km2 is a shredded network on a 1 km map and a healthy trunk on
    // a 100 m one. As a fraction it says outright whether a trunk river exists
    // - single digits mean the drainage graph is in fragments no matter how
    // convincing the render looks.
    const double mapAreaM2 = (double)n * (double)cellAreaM2;
    stats.maxDrainageAreaFraction = mapAreaM2 > 0.0 ? (float)(maxArea / mapAreaM2) : 0.0f;

    // This ledger covers detachment and transport only, and it is meant to
    // close: mass wasting books equal erosion and deposition (its border
    // handling is conservative), and hillslope creep is deliberately outside
    // the ledger because it moves the surface without producing sediment
    // flux. A residual therefore means a transport leak, not bookkeeping
    // slack. Half a percent sits above float accumulation noise on 16M cells
    // and well below anything a real leak produces.
    if (std::abs(stats.massErrorFraction) > 0.005) {
        SCENE_LOG_WARN("[LEM] Sediment ledger does not close: eroded " +
                       std::to_string(stats.eroded) + ", deposited " +
                       std::to_string(stats.deposited) + ", exported " +
                       std::to_string(stats.exported) + ", in transit " +
                       std::to_string(stats.carried) + " (" +
                       std::to_string(stats.massErrorFraction * 100.0) + "% unaccounted).");
    }
}

void publishNetAggradation(const std::vector<float>& inputHeight,
                           const std::vector<float>& outputHeight,
                           float heightScaleMeters,
                           std::vector<float>& deposition,
                           HydraulicErosionStats& stats) {
    if (inputHeight.size() != outputHeight.size() ||
        deposition.size() != outputHeight.size()) return;

    const float verticalScale = (std::max)(std::abs(heightScaleMeters), 1.0e-6f);
    int depositedCells = 0;
    double depositedThicknessMeters = 0.0;
    float deepestDepositMeters = 0.0f;
    for (std::size_t index = 0; index < outputHeight.size(); ++index) {
        const float net = (std::max)(outputHeight[index] - inputHeight[index], 0.0f);
        deposition[index] = net;
        const float meters = net * verticalScale;
        deepestDepositMeters = (std::max)(deepestDepositMeters, meters);
        if (meters > 0.01f) {
            ++depositedCells;
            depositedThicknessMeters += meters;
        }
    }
    stats.deepestDepositMeters = deepestDepositMeters;
    stats.depositedCells = depositedCells;
    stats.depositedAreaFraction = outputHeight.empty()
        ? 0.0f
        : static_cast<float>(depositedCells) / static_cast<float>(outputHeight.size());
    stats.meanDepositMeters = depositedCells > 0
        ? static_cast<float>(depositedThicknessMeters / depositedCells) : 0.0f;
}

}  // namespace TerrainLem

// Declared at global scope in TerrainManager.h so the node panel and the
// scripting layer can reach them without depending on the solver header,
// which is why they live outside namespace TerrainLem.
const char* fluvialQualityName(FluvialQuality quality) {
    switch (quality) {
        case FluvialQuality::Draft: return "Draft";
        case FluvialQuality::Balanced: return "Balanced";
        case FluvialQuality::High: return "High";
        default: return "Custom";
    }
}

void applyFluvialQuality(HydraulicErosionParams& p, FluvialQuality quality) {
    switch (quality) {
        case FluvialQuality::Draft:
            p.fluvialIterations = 10; p.drainageRefreshInterval = 8;
            p.drainageFillPasses = 48; p.drainageAccumulatePasses = 96;
            p.sedimentRouteSteps = 48; p.drainageCoarsestSize = 96;
            p.avulsionInterval = 12; p.alluviumSteps = 2;
            break;
        case FluvialQuality::Balanced:
            p.fluvialIterations = 16; p.drainageRefreshInterval = 6;
            p.drainageFillPasses = 96; p.drainageAccumulatePasses = 192;
            p.sedimentRouteSteps = 96; p.drainageCoarsestSize = 128;
            p.avulsionInterval = 8; p.alluviumSteps = 4;
            break;
        case FluvialQuality::High:
            p.fluvialIterations = 28; p.drainageRefreshInterval = 4;
            p.drainageFillPasses = 160; p.drainageAccumulatePasses = 320;
            p.sedimentRouteSteps = 160; p.drainageCoarsestSize = 192;
            p.avulsionInterval = 5; p.alluviumSteps = 6;
            break;
        case FluvialQuality::Custom:
            break;
    }
}

FluvialQuality detectFluvialQuality(const HydraulicErosionParams& p) {
    for (int i = 0; i < 3; ++i) {
        HydraulicErosionParams probe = p;
        applyFluvialQuality(probe, static_cast<FluvialQuality>(i));
        if (probe.fluvialIterations == p.fluvialIterations &&
            probe.drainageRefreshInterval == p.drainageRefreshInterval &&
            probe.drainageFillPasses == p.drainageFillPasses &&
            probe.drainageAccumulatePasses == p.drainageAccumulatePasses &&
            probe.sedimentRouteSteps == p.sedimentRouteSteps &&
            probe.drainageCoarsestSize == p.drainageCoarsestSize &&
            probe.avulsionInterval == p.avulsionInterval &&
            probe.alluviumSteps == p.alluviumSteps)
            return static_cast<FluvialQuality>(i);
    }
    return FluvialQuality::Custom;
}
