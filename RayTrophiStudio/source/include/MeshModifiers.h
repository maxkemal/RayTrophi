#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <map>
#include <array>
#include <cstdint>
#include "Triangle.h"
#include "json.hpp"

namespace MeshModifiers {

    // Subdivides a mesh evenly (1 level = each triangle split into 4)
    // Performs a linear subdivision without smoothing (flat subdivide).
    std::vector<std::shared_ptr<Triangle>> SubdivideSubD(const std::vector<std::shared_ptr<Triangle>>& inputMesh, int levels = 1);

    // Subdivides a mesh and applies Laplacian smoothing to the new vertices.
    // Approximates evaluating a smooth subdivision surface (like Loop subdivision for triangles).
    // NOTE: superseded by CatmullClarkSubD for the Smooth Subdivision modifier; kept for reference.
    std::vector<std::shared_ptr<Triangle>> SmoothSubD(const std::vector<std::shared_ptr<Triangle>>& inputMesh, int levels = 1, float smoothAngle = 0.5f);

    // Per-edge crease sharpness provider: given the two edge endpoints in LOCAL space,
    // returns sharpness in [0,1] (0 = smooth, 1 = fully sharp). Seeds level-1 creases;
    // sharpness is then propagated to child edges across levels. Boundary edges are
    // always sharp regardless. An empty function => smooth everywhere except boundaries.
    using EdgeCreaseFn = std::function<float(const Vec3&, const Vec3&)>;

    // True Catmull-Clark subdivision surface. The original vertices act as a control
    // cage and each level converges toward the CC limit surface (Blender's Subdivision
    // Surface behaviour). Operates on reconstructed topology, so it is immune to the
    // position-quantize normal cancellation of the old SmoothSubD path.
    std::vector<std::shared_ptr<Triangle>> CatmullClarkSubD(
        const std::vector<std::shared_ptr<Triangle>>& inputMesh,
        int levels = 1,
        const EdgeCreaseFn& creaseLookup = {});

    // ---- Catmull-Clark stencil engine (OpenSubdiv-style prepass + refine) ----
    // CC position rules are LINEAR in the current-level positions: every level is a
    // sparse matrix apply P_{L+1} = S_L * P_L whose weights depend only on topology +
    // crease sharpness, NOT on positions. CCSubdivPlan captures that topology + the
    // per-level sparse stencils + the static final connectivity/attributes ONCE; then
    // evaluateCCPositions re-applies the stencils to (possibly edited) cage positions.
    // This is what makes a LIVE, non-destructive CC modifier affordable: a cage edit
    // (sculpt/gizmo) only re-runs the cheap position apply, never a topology rebuild.
    struct CCSubdivPlan {
        int cageVertCount = 0;     // level-0 welded control vertices
        int finalVertCount = 0;

        // Level-0 welded cage positions (default control points). evaluateCCPositions
        // expects cagePositions in THIS welded order/size; CatmullClarkSubD passes
        // cageP0, while a live edit supplies the re-welded edited positions.
        std::vector<Vec3> cageP0;
        // Maps each input cage triangle corner (3*triIndex + c) to its welded vertex id,
        // so a live editor can re-gather welded positions after moving the soup.
        std::vector<int> cornerToWelded;

        // Per-level sparse stencil in CSR form:
        //   outPos[i] = sum_{k in [off[i], off[i+1])} w[k] * inPos[idx[k]]
        struct StencilLevel {
            std::vector<int>   off;   // size outCount+1
            std::vector<int>   idx;   // size off.back()
            std::vector<float> w;     // size off.back()
            int inCount = 0;
            int outCount = 0;
        };
        std::vector<StencilLevel> levels;

        // Static final triangulated geometry (for output Triangles + GPU BLAS):
        std::vector<uint32_t> triIndices;   // final triangles, 3 vertex ids per tri
        std::vector<Vec2>     triUV;        // per-CORNER uv (size == triIndices.size(); CC keeps seams)
        std::vector<uint16_t> triMat;       // per-tri material id (size == triIndices.size()/3)
        std::vector<int>      triFace;      // per-tri source FACE id (size == triIndices.size()/3) so the
                                            // materialized Triangles carry their polygon (quad/ngon) grouping
                                            // in Triangle::faceIndex — lets sculpt/shading compute ONE normal
                                            // per polygon (loop normals) instead of per split triangle.

        // Final FACE connectivity (quads/tris/ngons, CSR) — normals are Newell-summed over
        // these faces (NOT the split triangles) to match the reference CC exactly.
        std::vector<int> faceVertOff;       // size numFaces+1
        std::vector<int> faceVertIdx;       // vertex ids, size faceVertOff.back()
        // Final vertex -> incident final-FACE CSR, for an atomic-free normal gather.
        std::vector<int> vfaceOff;          // size finalVertCount+1
        std::vector<int> vfaceIdx;          // face ids, size vfaceOff.back()

        bool valid() const { return finalVertCount > 0 && !triIndices.empty(); }
    };

    // Build the position-independent plan from a control cage (triangle soup). Mirrors
    // the prepass + per-level rules of CatmullClarkSubD, storing weights instead of
    // evaluating positions. Returns an invalid plan on degenerate input.
    CCSubdivPlan buildCCSubdivPlan(
        const std::vector<std::shared_ptr<Triangle>>& cageMesh,
        int levels,
        const EdgeCreaseFn& creaseLookup = {});

    // Apply the plan's stencils to the given cage positions (one per cageVertCount) to
    // produce the final-level positions, plus area-weighted smooth normals. Runs on the
    // GPU compute backend when available + large enough, else on the CPU (identical
    // result up to float rounding).
    void evaluateCCPositions(
        const CCSubdivPlan& plan,
        const std::vector<Vec3>& cagePositions,
        std::vector<Vec3>& outPositions,
        std::vector<Vec3>& outNormals);

    // Drop-in stencil-engine equivalent of CatmullClarkSubD: build the plan, evaluate the
    // default cage positions, and materialize Triangle objects. Numerically identical to
    // CatmullClarkSubD (used to validate the engine, and the path the live modifier uses).
    std::vector<std::shared_ptr<Triangle>> catmullClarkSubDStencil(
        const std::vector<std::shared_ptr<Triangle>>& inputMesh,
        int levels = 1,
        const EdgeCreaseFn& creaseLookup = {});

    // ---- Phase 3d: device-resident CC geometry (zero-copy BLAS) ----
    // Result of evaluating a CC plan straight into a GPU buffer laid out as the RT BLAS
    // build expects (non-indexed expanded soup: vert | norm | uv | mat, no index block).
    // No host download — the buffer stays on the device and feeds createBLAS via
    // geometryDeviceAddress. Opaque ids keep this header free of the compute backend API.
    struct CCDeviceGeometry {
        uint64_t bufferId = 0;       // shared mesh-compute backend buffer handle id
        uint64_t deviceAddress = 0;  // -> BLASCreateInfo.geometryDeviceAddress
        uint32_t vertexCount = 0;    // triCount * 3 (expanded)
        uint32_t triCount = 0;
        bool valid() const { return bufferId != 0 && deviceAddress != 0 && triCount > 0; }
    };

    // Evaluate the plan's stencils + normals on the GPU and expand into a device-resident
    // BLAS-layout buffer (no host round-trip). Returns false (out left invalid) when the
    // compute backend is unavailable or any GPU step fails — caller then uses the host
    // (Triangle) path. Caller owns out.bufferId and must releaseCCDeviceGeometry() it.
    bool evaluateCCToDeviceGeometry(
        const CCSubdivPlan& plan,
        const std::vector<Vec3>& cagePositions,
        CCDeviceGeometry& out);

    // Destroy the device buffer behind a CCDeviceGeometry (no-op if invalid).
    void releaseCCDeviceGeometry(CCDeviceGeometry& geo);

    enum class ModifierType {
        FlatSubdivision,
        SmoothSubdivision,
        CatmullClark        // LIVE non-destructive Catmull-Clark (stencil engine + crease)
    };

    struct ModifierData {
        std::string name = "Modifier";
        ModifierType type = ModifierType::FlatSubdivision;
        bool enabled = true;
        int levels = 1;          // Blender-style VIEWPORT subdivision level (Solid / edit display)
        int renderLevels = 2;    // RENDER subdivision level (Rendered viewport / final quality)
        float smoothAngle = 0.5f;

        void serialize(nlohmann::json& j) const;
        void deserialize(const nlohmann::json& j);
    };

    struct ModifierStack {
        std::vector<ModifierData> modifiers;

        // Per-edge crease sharpness in [0,1], authored on the control cage and consumed
        // by the Catmull-Clark evaluator. Keyed by the quantized LOCAL positions of the
        // two edge endpoints, so it is topology-independent: it survives re-welding and
        // maps cleanly onto a future GPU subdivider that rebuilds its own connectivity.
        std::map<std::array<int, 6>, float> edgeCreases;

        // Evaluate modifiers sequentially on a base mesh. Creases (if any) are applied
        // automatically inside Catmull-Clark — no extra plumbing at the call sites.
        // forRender selects each subdivision modifier's renderLevels (Rendered / final
        // quality) instead of its viewport levels (Solid / edit display).
        std::vector<std::shared_ptr<Triangle>> evaluate(const std::vector<std::shared_ptr<Triangle>>& baseMesh, bool forRender = false) const;

        // Crease authoring (positions in mesh LOCAL space). weight<=0 clears the edge.
        static std::array<int, 6> makeCreaseKey(const Vec3& a, const Vec3& b);
        void setEdgeCrease(const Vec3& a, const Vec3& b, float weight);
        float getEdgeCrease(const Vec3& a, const Vec3& b) const;

        void serialize(nlohmann::json& j) const;
        void deserialize(const nlohmann::json& j);
    };

} 
