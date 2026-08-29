/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          TerrainSystem.h
* Author:        Kemal Demirtas
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include "Vec3.h"
#include "TriangleMesh.h"
#include "FoliageFwd.h"
#include "WaterSystem.h" // For WaterWaveParams

namespace TerrainNodesV2 {
    class TerrainNodeGraphV2;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// TERRAIN DATA STRUCTURES
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// Normal calculation quality levels
enum class NormalQuality { 
    Fast,        // 4-neighbor central difference (fastest)
    Sobel,       // 8-neighbor Sobel filter (balanced)
    HighQuality  // Weighted 8-neighbor with edge detection
};

// Sector-based dirty region tracking for incremental updates
struct DirtyRegion {
    static constexpr int SECTOR_GRID_SIZE = 16;  // 16x16 = 256 sectors
    bool sectors[SECTOR_GRID_SIZE][SECTOR_GRID_SIZE] = {{false}};
    bool has_any_dirty = false;
    
    void markDirty(int gridX, int gridZ, int terrainWidth, int terrainHeight) {
        if (terrainWidth <= 0 || terrainHeight <= 0) return;
        int sectorX = (gridX * SECTOR_GRID_SIZE) / terrainWidth;
        int sectorZ = (gridZ * SECTOR_GRID_SIZE) / terrainHeight;
        sectorX = std::clamp(sectorX, 0, SECTOR_GRID_SIZE - 1);
        sectorZ = std::clamp(sectorZ, 0, SECTOR_GRID_SIZE - 1);
        sectors[sectorX][sectorZ] = true;
        has_any_dirty = true;
    }
    
    void markAllDirty() {
        memset(sectors, true, sizeof(sectors));
        has_any_dirty = true;
    }
    
    void clear() {
        memset(sectors, 0, sizeof(sectors));
        has_any_dirty = false;
    }
    
    int countDirtySectors() const {
        int count = 0;
        for (int y = 0; y < SECTOR_GRID_SIZE; y++)
            for (int x = 0; x < SECTOR_GRID_SIZE; x++)
                if (sectors[x][y]) count++;
        return count;
    }
};

struct Heightmap {
    std::vector<float> data; // Row-major: y * width + x
    int width = 0;
    int height = 0;
    float scale_y = 10.0f;   // Maximum height
    float scale_xz = 100.0f; // World size (Total width/depth)
    // Derived raw-data bounds. Procedural metre-authored graphs may exceed
    // 0..1 or become negative, so picking cannot assume normalized heights.
    float min_value = 0.0f;
    float max_value = 0.0f;
    
    // Get height at grid coordinate (clamped)
    float getHeight(int x, int y) const {
        if (data.empty()) return 0.0f;
        if (x < 0) x = 0; if (x >= width) x = width - 1;
        if (y < 0) y = 0; if (y >= height) y = height - 1;
        return data[y * width + x] * scale_y;
    }
    
    // Set height (0.0 - 1.0 range usually stored, but we store raw normalized)
    void setHeight(int x, int y, float v) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            data[y * width + x] = v;
        }
    }
};

struct TerrainObject {
    int id = -1;
    std::string name;
    
    Heightmap heightmap;

    // Flat (SoA) mesh representation for raytracing/raster. One Hittable for the
    // whole terrain grid instead of a facade Triangle per triangle. Rebuilt (new
    // TriangleMesh) when resolution/topology changes, updated in place (direct
    // SoA writes) when only heights change.
    std::shared_ptr<TriangleMesh> flatMesh;
    
    // Reference to a transform (usually identity for terrain, but can be moved)
    std::shared_ptr<Transform> transform;
    
    uint16_t material_id = 0;
    
    bool dirty_mesh = false; // Flag to rebuild mesh from heightmap

    // Terrain node evaluation may run its expensive height phase on a worker.
    // Erosion routines historically rebuilt the render mesh as a side effect;
    // during graph evaluation that would create/replace flatMesh off-thread,
    // before it can be registered in SceneData.  The graph enables this guard
    // while computing and performs the single authoritative mesh update during
    // its main-thread finalize phase.
    bool defer_mesh_updates = false;
    
    // Terrain Layer System
    std::shared_ptr<class Texture> splatMap;   // RGBA Splat Map (Control Texture)
    // Non-normalized paint controls: R=Flow, G=Wetness, B=Ice, A=Hardness.
    // Kept separate from normalized material weights so semantic fields remain
    // independently recoverable by shading, scatter and export consumers.
    std::shared_ptr<class Texture> surfaceSemanticMap;

    // -------------------------------------------------------------------
    // Layer slots
    //
    // Slots 0-3 are driven by the splat map and *partition* the surface:
    // their weights are normalized to sum to 1, so adding one takes cover
    // away from the others. Slots 4-7 are driven by the semantic map and
    // *overlay* that blend, each composited by its own unnormalized weight.
    //
    // The two families cannot share one normalization. A semantic weight is
    // not a share of the surface - Flow=0.8 means "strong current here", not
    // "80% of this pixel is river" - and a river bed material covers the
    // substrate rather than competing with it for area.
    //
    // An empty overlay slot keeps the built-in shading tweak for that
    // channel, so a terrain authored before overlays looks unchanged.
    // -------------------------------------------------------------------
    static constexpr int kSplatLayerSlots = 4;      ///< 0-3: Grass/Rock/Snow/Soil by convention
    static constexpr int kSemanticLayerSlots = 4;   ///< 4-7: Flow/Wetness/Ice/Hardness
    static constexpr int kMaxLayerSlots = kSplatLayerSlots + kSemanticLayerSlots;

    std::vector<std::shared_ptr<class Material>> layers; // kMaxLayerSlots slots
    std::vector<float> layer_uv_scales;      // UV tiling scale for each layer
    /// Overlay strength for slots 4-7. Ignored by the splat-driven slots,
    /// which are normalized against each other instead.
    std::vector<float> layer_overlay_strength;
    /// Per-slot opt-out of snow burial, for slots 4-7.
    ///
    /// A semantic channel value is a MEASUREMENT; an overlay's coverage is a
    /// VISIBILITY decision. Flow can read 0.9 under two metres of snow - the
    /// measurement is right, and erosion and soil capacity must keep reading
    /// it. What is wrong is painting it, which is how a river ended up drawn
    /// across a snow-filled valley with nothing to mask it. So overlays are
    /// composited UNDER the snow the splat map already placed.
    ///
    /// Setting this for a slot says the author means it to show through:
    /// open water cutting a snowfield, say. Ice (slot 6) is a cover in its
    /// own right and is never buried.
    std::vector<uint8_t> layer_overlay_ignore_cover;

    /// Human-readable name of a slot, for panels, logs and the script API.
    /// Hardness is the odd channel and the name says so.
    ///
    /// Flow, Wetness and Ice are SURFACE CONDITIONS - water, moisture and rime
    /// are literally at the surface, so painting a material by their weight is
    /// meaningful. Hardness is a SUBSTRATE PROPERTY: it says how resistant the
    /// bedrock is, not that bedrock is visible. Hardness 0.8 under two metres
    /// of valley soil is still 0.8, and a material bound to it used to paint
    /// granite across meadows.
    ///
    /// So the Hardness overlay is gated by ROCK EXPOSURE (splat G), which
    /// turns it into what it is actually useful for: a bedrock VARIANT, hard
    /// granite outcrops against soft shale ones, appearing only where rock
    /// shows. As a control it keeps its full unmasked value - erosion
    /// resistance and soil capacity read the field, not the paint.
    static const char* layerSlotName(int slot) {
        switch (slot) {
            case 0: return "Splat R";
            case 1: return "Splat G";
            case 2: return "Splat B";
            case 3: return "Splat A";
            case 4: return "Semantic Flow";
            case 5: return "Semantic Wetness";
            case 6: return "Semantic Ice";
            case 7: return "Semantic Hardness (exposed rock)";
            default: return "Invalid";
        }
    }
    static bool isSemanticLayerSlot(int slot) {
        return slot >= kSplatLayerSlots && slot < kMaxLayerSlots;
    }

    // =========================================================================
    // MACRO COLOR MAP (SatMap Colorizer — Faz 1 / Faz 2)
    // =========================================================================
    // Low-frequency color modulator evaluated at paint_resolution. Produced by
    // the terrain node graph (SatMapNode → ColorOutputNode) and applied in the
    // shader AFTER the splat-blend, preserving tile detail via luma ratio:
    //   relit = macro * (blendAlbedo / luminance(blendAlbedo))
    // macro_color_strength = 0 (default) → zero render cost, old scenes untouched.
    std::shared_ptr<class Texture> macroColorMap; // RGBA8, paint_resolution
    float macro_color_strength = 0.0f;            // 0 = off; 1 = full macro color
    
    // Foliage System
    std::vector<TerrainFoliageLayer> foliageLayers;
    
    // Hardness map for erosion: 0.0 = soft (sand/soil), 1.0 = hard (bedrock)
    std::vector<float> hardnessMap;  // Same resolution as heightmap

    // Flow accumulation map: Higher values indicate streams/rivers
    std::vector<float> flowMap;      // Same resolution as heightmap

    // Packed erosion helper map: RGBA = erosion / deposition / flow / influence
    std::vector<float> erosionMapRGBA; // Same resolution as heightmap, 4 channels

    // Derived named fields published by explicit terrain graph output nodes.
    // updateTerrainMesh mirrors valid fields to flatMesh vertex attributes on
    // the main thread so foliage/scatter can sample them barycentrically.
    std::unordered_map<std::string, std::shared_ptr<std::vector<float>>> analysisFields;

    // Analytical river/lake descriptions produced by terrain hydrology nodes.
    // These remain independent from generated WaterSurface meshes so a water
    // body can be regenerated, LODed or simulated without losing its identity.
    std::vector<WaterBodyData> waterBodies;

    // Non-destructive editing support (Node Graph)
    std::vector<float> original_heightmap_data; // Initial state before node graph evaluation
    
    // =========================================================================
    // WATER RENDERING INTEGRATION
    // =========================================================================
    bool renderAsWater = false; // If true, treats this terrain as a water surface (mesh)
    WaterWaveParams waterParams; // Wave parameters if rendered as water
    int waterSurfaceId = -1;     // Internal ID for WaterManager integration
    
    // =========================================================================
    // QUALITY & OPTIMIZATION SETTINGS
    // =========================================================================
    
    // Dirty region tracking for incremental mesh updates
    DirtyRegion dirty_region;
    
    // =========================================================================
    // THREE-WAY RESOLUTION SPLIT (Faz 0 — fully separated 2026-08-21)
    // =========================================================================
    //
    // Three numbers used to share one value (heightmap.width). Each serves a
    // different optimum:
    //
    //   FIELD   — high: flow accumulation, erosion, valley/wetness analysis
    //   MESH    — low:  BVH/BLAS is linear in triangle count; 1024 mesh over a
    //             4096 field is ~18× cheaper with no analysis quality loss
    //             (measured 2026-08-21: 6.4 s → 0.42 s Vulkan solid raster)
    //   PAINT   — independent: splat + macro-color evaluation resolution.
    //             Raising it only buys new information when procedural nodes
    //             (noise, warp, threshold) run at that resolution.
    //
    // Convention: 0 means "follow the field" for both mesh and paint.
    // Serialized as separate keys so old projects read 0 and stay unchanged.

    // MESH RESOLUTION — vertex grid, separate from the field
    // 0 = same as field (historical default, existing scenes untouched).
    int mesh_resolution = 0;

    int meshGridWidth() const {
        return mesh_resolution >= 2 ? (std::min)(mesh_resolution, heightmap.width) : heightmap.width;
    }
    int meshGridHeight() const {
        return mesh_resolution >= 2 ? (std::min)(mesh_resolution, heightmap.height) : heightmap.height;
    }
    // True when the vertex grid no longer matches the field grid.
    bool meshMatchesField() const {
        return meshGridWidth() == heightmap.width && meshGridHeight() == heightmap.height;
    }

    // PAINT RESOLUTION — splat map + macroColorMap evaluation grid.
    // 0 = follow the field (max(512, field_width)), which is the historical
    // behaviour of resizeSplatMap and keeps existing scenes unchanged.
    // Values > field_resolution are valid: procedural nodes evaluate at paint
    // resolution, buying real new frequency. Pure analysis chains (no procedural
    // nodes) gain nothing beyond field resolution — the panel warns about this.
    int paint_resolution = 0;

    int paintGridWidth() const {
        if (paint_resolution >= 2) return paint_resolution;
        return (std::max)(512, heightmap.width);
    }
    int paintGridHeight() const {
        if (paint_resolution >= 2) return paint_resolution;
        return (std::max)(512, heightmap.height);
    }

    // Normal calculation quality
    NormalQuality normal_quality = NormalQuality::Sobel;
    float normal_strength = 1.0f;  // Multiplier for normal intensity (0.1 - 3.0)
    
    // Procedural Auto-Mask Settings (Persistent)
    float am_height_min = 5.0f;
    float am_height_max = 20.0f;
    float am_slope = 5.0f;
    float am_flow_threshold = 5.0f; // Threshold for flow accumulation masking

    // Node Graph for non-destructive editing
    std::shared_ptr<TerrainNodesV2::TerrainNodeGraphV2> nodeGraph;

    // Helper to mark a heightmap cell as dirty
    void markCellDirty(int gridX, int gridZ) {
        dirty_region.markDirty(gridX, gridZ, heightmap.width, heightmap.height);
    }
};
