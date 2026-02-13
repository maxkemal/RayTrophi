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
#include "Vec3.h"
#include "Triangle.h"
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
    
    // Mesh representation for Raytracing
    // We maintain a list of triangles to update them when heightmap changes
    std::vector<std::shared_ptr<Triangle>> mesh_triangles;
    
    // Reference to a transform (usually identity for terrain, but can be moved)
    std::shared_ptr<Transform> transform;
    
    uint16_t material_id = 0;
    
    bool dirty_mesh = false; // Flag to rebuild mesh from heightmap
    
    // Terrain Layer System
    std::shared_ptr<class Texture> splatMap;   // RGBA Splat Map (Control Texture)
    std::vector<std::shared_ptr<class Material>> layers; // Up to 4 layers
    std::vector<float> layer_uv_scales;      // UV tiling scale for each layer
    
    // Foliage System
    std::vector<TerrainFoliageLayer> foliageLayers;
    
    // Hardness map for erosion: 0.0 = soft (sand/soil), 1.0 = hard (bedrock)
    std::vector<float> hardnessMap;  // Same resolution as heightmap

    // Flow accumulation map: Higher values indicate streams/rivers
    std::vector<float> flowMap;      // Same resolution as heightmap

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

