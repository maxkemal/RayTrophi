/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          FluidGrid.h
* Author:        Kemal Demirtas
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file FluidGrid.h
 * @brief MAC (Marker-And-Cell) staggered grid for fluid simulation
 * 
 * This implementation uses a staggered grid where:
 * - Velocity components are stored on cell faces
 * - Scalar quantities (density, temperature, pressure) are cell-centered
 * 
 * This approach prevents the checkerboard instability common in collocated grids.
 */

#include "Vec3.h"
#include <vector>
#include <cmath>
#include <algorithm>

// Forward declaration for CUDA compatibility
#ifdef __CUDACC__
#define FLUID_FUNC __host__ __device__
#else
#define FLUID_FUNC
#endif

namespace FluidSim {

/**
 * @brief 3D index helper for flat array access
 */
struct GridIndex {
    int i, j, k;
    
    FLUID_FUNC GridIndex(int _i = 0, int _j = 0, int _k = 0) 
        : i(_i), j(_j), k(_k) {}
    
    FLUID_FUNC bool isValid(int nx, int ny, int nz) const {
        return i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz;
    }
};

/**
 * @brief MAC (Marker-And-Cell) staggered grid for stable fluid simulation
 * 
 * Grid layout:
 * - vel_x: stored at (i+0.5, j, k) - X-face of each cell
 * - vel_y: stored at (i, j+0.5, k) - Y-face of each cell
 * - vel_z: stored at (i, j, k+0.5) - Z-face of each cell
 * - density/temperature/pressure: stored at cell centers (i+0.5, j+0.5, k+0.5)
 */

// ═══════════════════════════════════════════════════════════════════════════════
// ACTIVE TILES SYSTEM (VDB-style Sparse Grid Optimization)
// ═══════════════════════════════════════════════════════════════════════════════
// Tiles are 8x8x8 blocks of cells. Only "active" tiles are processed.
// This can provide 50-90% speedup when smoke occupies small portion of grid.

constexpr int TILE_SIZE = 8;  // 8x8x8 = 512 cells per tile

struct ActiveTile {
    int tx, ty, tz;        // Tile coordinates
    int start_x, start_y, start_z;  // Cell start indices
    float max_density;     // Max density in tile (for LOD)
    float max_velocity;    // Max velocity in tile (for CFL)
    bool has_emitter;      // Tile contains an emitter
    
    FLUID_FUNC ActiveTile(int _tx = 0, int _ty = 0, int _tz = 0)
        : tx(_tx), ty(_ty), tz(_tz)
        , start_x(_tx * TILE_SIZE), start_y(_ty * TILE_SIZE), start_z(_tz * TILE_SIZE)
        , max_density(0.0f), max_velocity(0.0f), has_emitter(false) {}
};

class FluidGrid {
public:
    // ═══════════════════════════════════════════════════════════════════════════
    // GRID DIMENSIONS
    // ═══════════════════════════════════════════════════════════════════════════
    int nx, ny, nz;           // Number of cells in each dimension
    float voxel_size;         // Size of each voxel in world units
    Vec3 origin;              // World position of grid origin (corner 0,0,0)
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ACTIVE TILES (Sparse Grid Optimization)
    // ═══════════════════════════════════════════════════════════════════════════
    int tiles_x, tiles_y, tiles_z;           // Number of tiles in each dimension
    std::vector<ActiveTile> active_tiles;    // List of currently active tiles
    std::vector<uint8_t> tile_active_mask;   // Fast lookup: is tile active?
    bool sparse_mode_enabled = true;         // Enable/disable sparse optimization
    float sparse_threshold = 0.001f;         // Minimum density to consider tile active
    
    // Statistics
    mutable int total_tiles = 0;
    mutable int active_tile_count = 0;
    mutable float sparse_efficiency = 0.0f;  // % of tiles skipped
    
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    // VELOCITY FIELD (Staggered MAC grid)
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    std::vector<float> vel_x;  // X velocity on X-faces, size: (nx+1) * ny * nz
    std::vector<float> vel_y;  // Y velocity on Y-faces, size: nx * (ny+1) * nz
    std::vector<float> vel_z;  // Z velocity on Z-faces, size: nx * ny * (nz+1)
    
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    // SCALAR FIELDS (Cell-centered)
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    std::vector<float> density;      // Smoke/gas density
    std::vector<float> temperature;  // Temperature field
    std::vector<float> fuel;         // Fuel field (for combustion/fire)
    std::vector<float> interaction;  // Reaction/Flame intensity field
    std::vector<float> pressure;     // Pressure field (for projection)
    std::vector<float> divergence;   // Velocity divergence (temp buffer)
    
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    // SOLID/BOUNDARY FIELD
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    std::vector<uint8_t> solid;     // 0 = fluid, 1 = solid boundary
    
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    // CONSTRUCTION
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    FluidGrid() : nx(0), ny(0), nz(0), voxel_size(1.0f) {}
    
    FluidGrid(int _nx, int _ny, int _nz, float _voxel_size = 0.1f, const Vec3& _origin = Vec3(0,0,0))
        : nx(_nx), ny(_ny), nz(_nz), voxel_size(_voxel_size), origin(_origin) {
        allocate();
    }
    
    void resize(int _nx, int _ny, int _nz, float _voxel_size = 0.1f, const Vec3& _origin = Vec3(0,0,0)) {
        nx = _nx; ny = _ny; nz = _nz;
        voxel_size = _voxel_size;
        origin = _origin;
        allocate();
    }
    
    void allocate() {
        size_t cell_count = static_cast<size_t>(nx) * ny * nz;
        
        // Staggered velocity grids
        vel_x.resize((nx + 1) * ny * nz, 0.0f);
        vel_y.resize(nx * (ny + 1) * nz, 0.0f);
        vel_z.resize(nx * ny * (nz + 1), 0.0f);
        
        // Cell-centered scalars
        density.resize(cell_count, 0.0f);
        temperature.resize(cell_count, 0.0f);
        fuel.resize(cell_count, 0.0f);
        interaction.resize(cell_count, 0.0f);
        pressure.resize(cell_count, 0.0f);
        divergence.resize(cell_count, 0.0f);
        solid.resize(cell_count, 0);
        
        // Initialize tile system
        tiles_x = (nx + TILE_SIZE - 1) / TILE_SIZE;
        tiles_y = (ny + TILE_SIZE - 1) / TILE_SIZE;
        tiles_z = (nz + TILE_SIZE - 1) / TILE_SIZE;
        total_tiles = tiles_x * tiles_y * tiles_z;
        tile_active_mask.resize(total_tiles, 0);
        active_tiles.clear();
        active_tiles.reserve(total_tiles);
    }
    
    void clear() {
        std::fill(vel_x.begin(), vel_x.end(), 0.0f);
        std::fill(vel_y.begin(), vel_y.end(), 0.0f);
        std::fill(vel_z.begin(), vel_z.end(), 0.0f);
        std::fill(density.begin(), density.end(), 0.0f);
        std::fill(temperature.begin(), temperature.end(), 0.0f);
        std::fill(fuel.begin(), fuel.end(), 0.0f);
        std::fill(interaction.begin(), interaction.end(), 0.0f);
        std::fill(pressure.begin(), pressure.end(), 0.0f);
        std::fill(divergence.begin(), divergence.end(), 0.0f);
        
        // Clear tile activity
        std::fill(tile_active_mask.begin(), tile_active_mask.end(), 0);
        active_tiles.clear();
        active_tile_count = 0;
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ACTIVE TILE MANAGEMENT (VDB-style Sparse Optimization)
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// @brief Get tile index from tile coordinates
    FLUID_FUNC size_t tileIndex(int tx, int ty, int tz) const {
        return static_cast<size_t>(tx) + static_cast<size_t>(ty) * tiles_x + static_cast<size_t>(tz) * tiles_x * tiles_y;
    }
    
    /// @brief Get tile coordinates from cell indices
    FLUID_FUNC void cellToTile(int i, int j, int k, int& tx, int& ty, int& tz) const {
        tx = i / TILE_SIZE;
        ty = j / TILE_SIZE;
        tz = k / TILE_SIZE;
    }
    
    /// @brief Check if a tile is active
    FLUID_FUNC bool isTileActive(int tx, int ty, int tz) const {
        if (tx < 0 || tx >= tiles_x || ty < 0 || ty >= tiles_y || tz < 0 || tz >= tiles_z) return false;
        return tile_active_mask[tileIndex(tx, ty, tz)] != 0;
    }
    
    /// @brief Update active tiles based on density/velocity content
    void updateActiveTiles(float ambient_temp = 293.0f) {
        if (!sparse_mode_enabled) {
            // All tiles active when sparse mode disabled
            active_tiles.clear();
            for (int tz = 0; tz < tiles_z; ++tz) {
                for (int ty = 0; ty < tiles_y; ++ty) {
                    for (int tx = 0; tx < tiles_x; ++tx) {
                        active_tiles.emplace_back(tx, ty, tz);
                        tile_active_mask[tileIndex(tx, ty, tz)] = 1;
                    }
                }
            }
            active_tile_count = total_tiles;
            sparse_efficiency = 0.0f;
            return;
        }
        
        active_tiles.clear();
        std::fill(tile_active_mask.begin(), tile_active_mask.end(), 0);
        
        // Scan each tile for activity
        for (int tz = 0; tz < tiles_z; ++tz) {
            for (int ty = 0; ty < tiles_y; ++ty) {
                for (int tx = 0; tx < tiles_x; ++tx) {
                    bool is_active = scanTileForActivity(tx, ty, tz, ambient_temp);
                    
                    if (is_active) {
                        active_tiles.emplace_back(tx, ty, tz);
                        tile_active_mask[tileIndex(tx, ty, tz)] = 1;
                    }
                }
            }
        }
        
        // Also activate neighbors of active tiles (for advection/diffusion)
        std::vector<ActiveTile> expanded;
        expanded.reserve(active_tiles.size() * 2);
        
        for (const auto& tile : active_tiles) {
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int ntx = tile.tx + dx;
                        int nty = tile.ty + dy;
                        int ntz = tile.tz + dz;
                        
                        if (ntx >= 0 && ntx < tiles_x && 
                            nty >= 0 && nty < tiles_y && 
                            ntz >= 0 && ntz < tiles_z) {
                            size_t idx = tileIndex(ntx, nty, ntz);
                            if (tile_active_mask[idx] == 0) {
                                tile_active_mask[idx] = 1;
                                expanded.emplace_back(ntx, nty, ntz);
                            }
                        }
                    }
                }
            }
        }
        
        active_tiles.insert(active_tiles.end(), expanded.begin(), expanded.end());
        active_tile_count = static_cast<int>(active_tiles.size());
        sparse_efficiency = total_tiles > 0 ? 100.0f * (1.0f - (float)active_tile_count / total_tiles) : 0.0f;
    }
    
    /// @brief Check if tile has meaningful content
    bool scanTileForActivity(int tx, int ty, int tz, float ambient_temp) const {
        int start_i = tx * TILE_SIZE;
        int start_j = ty * TILE_SIZE;
        int start_k = tz * TILE_SIZE;
        int end_i = std::min(start_i + TILE_SIZE, nx);
        int end_j = std::min(start_j + TILE_SIZE, ny);
        int end_k = std::min(start_k + TILE_SIZE, nz);
        
        for (int k = start_k; k < end_k; ++k) {
            for (int j = start_j; j < end_j; ++j) {
                for (int i = start_i; i < end_i; ++i) {
                    size_t idx = cellIndex(i, j, k);
                    // Check density, fuel, or temperature above ambient
                    if (density[idx] > sparse_threshold || 
                        fuel[idx] > sparse_threshold ||
                        std::abs(temperature[idx] - ambient_temp) > 10.0f) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
    /// @brief Mark tile as active (e.g., for emitters)
    void activateTile(int tx, int ty, int tz) {
        if (tx < 0 || tx >= tiles_x || ty < 0 || ty >= tiles_y || tz < 0 || tz >= tiles_z) return;
        size_t idx = tileIndex(tx, ty, tz);
        if (tile_active_mask[idx] == 0) {
            tile_active_mask[idx] = 1;
            active_tiles.emplace_back(tx, ty, tz);
            active_tile_count = static_cast<int>(active_tiles.size());
        }
    }
    
    /// @brief Activate tiles around a world position (for emitters)
    void activateTilesAroundPosition(const Vec3& world_pos, float radius) {
        float fi, fj, fk;
        worldToGrid(world_pos, fi, fj, fk);
        
        int radius_cells = static_cast<int>(std::ceil(radius / voxel_size));
        int radius_tiles = (radius_cells + TILE_SIZE - 1) / TILE_SIZE + 1;
        
        int center_tx = static_cast<int>(fi) / TILE_SIZE;
        int center_ty = static_cast<int>(fj) / TILE_SIZE;
        int center_tz = static_cast<int>(fk) / TILE_SIZE;
        
        for (int dz = -radius_tiles; dz <= radius_tiles; ++dz) {
            for (int dy = -radius_tiles; dy <= radius_tiles; ++dy) {
                for (int dx = -radius_tiles; dx <= radius_tiles; ++dx) {
                    activateTile(center_tx + dx, center_ty + dy, center_tz + dz);
                }
            }
        }
    }
    
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    // INDEX HELPERS
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    
    /// @brief Linear index for cell-centered quantities
    FLUID_FUNC size_t cellIndex(int i, int j, int k) const {
        return static_cast<size_t>(i) + static_cast<size_t>(j) * nx + static_cast<size_t>(k) * nx * ny;
    }
    
    /// @brief Linear index for X-velocity (on X-faces)
    FLUID_FUNC size_t velXIndex(int i, int j, int k) const {
        return static_cast<size_t>(i) + static_cast<size_t>(j) * (nx + 1) + static_cast<size_t>(k) * (nx + 1) * ny;
    }
    
    /// @brief Linear index for Y-velocity (on Y-faces)
    FLUID_FUNC size_t velYIndex(int i, int j, int k) const {
        return static_cast<size_t>(i) + static_cast<size_t>(j) * nx + static_cast<size_t>(k) * nx * (ny + 1);
    }
    
    /// @brief Linear index for Z-velocity (on Z-faces)
    FLUID_FUNC size_t velZIndex(int i, int j, int k) const {
        return static_cast<size_t>(i) + static_cast<size_t>(j) * nx + static_cast<size_t>(k) * nx * ny;
    }
    
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    // COORDINATE CONVERSION
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    
    /// @brief Convert world position to grid indices
    void worldToGrid(const Vec3& world, float& fi, float& fj, float& fk) const {
        Vec3 local = (world - origin) / voxel_size;
        fi = local.x;
        fj = local.y;
        fk = local.z;
    }
    
    /// @brief Convert grid indices to world position (cell center)
    Vec3 gridToWorld(int i, int j, int k) const {
        return origin + Vec3(
            (i + 0.5f) * voxel_size,
            (j + 0.5f) * voxel_size,
            (k + 0.5f) * voxel_size
        );
    }
    
    /// @brief Get world-space bounding box
    void getWorldBounds(Vec3& min_out, Vec3& max_out) const {
        min_out = origin;
        max_out = origin + Vec3(nx * voxel_size, ny * voxel_size, nz * voxel_size);
    }
    
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    // SAMPLING (Trilinear interpolation)
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    
    /// @brief Sample density at world position using trilinear interpolation
    float sampleDensity(const Vec3& world_pos) const {
        return sampleCellCentered(density, world_pos);
    }
    
    /// @brief Sample temperature at world position
    float sampleTemperature(const Vec3& world_pos) const {
        return sampleCellCentered(temperature, world_pos);
    }
    
    /// @brief Sample fuel at world position
    float sampleFuel(const Vec3& world_pos) const {
        return sampleCellCentered(fuel, world_pos);
    }
    
    /// @brief Sample interaction (flame intensity) at world position
    float sampleInteraction(const Vec3& world_pos) const {
        return sampleCellCentered(interaction, world_pos);
    }
    
    /// @brief Sample velocity at world position (interpolated from staggered grid)
    Vec3 sampleVelocity(const Vec3& world_pos) const {
        Vec3 local = (world_pos - origin) / voxel_size;
        
        // Sample each velocity component at its staggered location
        float vx = sampleVelX(local.x, local.y - 0.5f, local.z - 0.5f);
        float vy = sampleVelY(local.x - 0.5f, local.y, local.z - 0.5f);
        float vz = sampleVelZ(local.x - 0.5f, local.y - 0.5f, local.z);
        
        return Vec3(vx, vy, vz);
    }
    
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    // CELL ACCESS
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    
    float& densityAt(int i, int j, int k) { return density[cellIndex(i, j, k)]; }
    float densityAt(int i, int j, int k) const { return density[cellIndex(i, j, k)]; }
    
    float& temperatureAt(int i, int j, int k) { return temperature[cellIndex(i, j, k)]; }
    float temperatureAt(int i, int j, int k) const { return temperature[cellIndex(i, j, k)]; }
    
    float& fuelAt(int i, int j, int k) { return fuel[cellIndex(i, j, k)]; }
    float fuelAt(int i, int j, int k) const { return fuel[cellIndex(i, j, k)]; }

    float& reactionAt(int i, int j, int k) { return interaction[cellIndex(i, j, k)]; }
    float reactionAt(int i, int j, int k) const { return interaction[cellIndex(i, j, k)]; }

    float& pressureAt(int i, int j, int k) { return pressure[cellIndex(i, j, k)]; }
    float pressureAt(int i, int j, int k) const { return pressure[cellIndex(i, j, k)]; }
    
    float& velXAt(int i, int j, int k) { return vel_x[velXIndex(i, j, k)]; }
    float velXAt(int i, int j, int k) const { return vel_x[velXIndex(i, j, k)]; }
    
    float& velYAt(int i, int j, int k) { return vel_y[velYIndex(i, j, k)]; }
    float velYAt(int i, int j, int k) const { return vel_y[velYIndex(i, j, k)]; }
    
    float& velZAt(int i, int j, int k) { return vel_z[velZIndex(i, j, k)]; }
    float velZAt(int i, int j, int k) const { return vel_z[velZIndex(i, j, k)]; }
    
    bool isSolid(int i, int j, int k) const {
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return true;
        return solid[cellIndex(i, j, k)] != 0;
    }
    
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    // STATISTICS
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    
    float getTotalDensity() const {
        float total = 0.0f;
        for (float d : density) total += d;
        return total;
    }
    
    float getMaxDensity() const {
        float max_val = 0.0f;
        for (float d : density) max_val = std::max(max_val, d);
        return max_val;
    }
    
    float getMaxVelocity() const {
        float max_vel = 0.0f;
        for (float v : vel_x) max_vel = std::max(max_vel, std::abs(v));
        for (float v : vel_y) max_vel = std::max(max_vel, std::abs(v));
        for (float v : vel_z) max_vel = std::max(max_vel, std::abs(v));
        return max_vel;
    }
    
    int getActiveVoxelCount(float threshold = 0.001f) const {
        int count = 0;
        for (float d : density) {
            if (d > threshold) count++;
        }
        return count;
    }
    
    size_t getCellCount() const { return static_cast<size_t>(nx) * ny * nz; }
    size_t getMemoryUsage() const {
        return vel_x.size() * sizeof(float) +
               vel_y.size() * sizeof(float) +
               vel_z.size() * sizeof(float) +
               density.size() * sizeof(float) +
               temperature.size() * sizeof(float) +
               fuel.size() * sizeof(float) +
               interaction.size() * sizeof(float) +
               pressure.size() * sizeof(float) +
               divergence.size() * sizeof(float) +
               solid.size() * sizeof(uint8_t);
    }
    float sampleCellCentered(const std::vector<float>& field, const Vec3& world_pos) const {
        Vec3 local = (world_pos - origin) / voxel_size - Vec3(0.5f, 0.5f, 0.5f);

        int i0 = static_cast<int>(std::floor(local.x));
        int j0 = static_cast<int>(std::floor(local.y));
        int k0 = static_cast<int>(std::floor(local.z));

        float fx = local.x - i0;
        float fy = local.y - j0;
        float fz = local.z - k0;

        return trilinear(field, i0, j0, k0, fx, fy, fz, nx, ny, nz);
    }
private:
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    // INTERNAL SAMPLING HELPERS
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    
    /// @brief Trilinear interpolation for cell-centered quantities
   
    
    /// @brief Sample X-velocity component
    float sampleVelX(float fi, float fj, float fk) const {
        int i0 = static_cast<int>(std::floor(fi));
        int j0 = static_cast<int>(std::floor(fj));
        int k0 = static_cast<int>(std::floor(fk));
        
        float fx = fi - i0;
        float fy = fj - j0;
        float fz = fk - k0;
        
        return trilinearVelX(i0, j0, k0, fx, fy, fz);
    }
    
    float sampleVelY(float fi, float fj, float fk) const {
        int i0 = static_cast<int>(std::floor(fi));
        int j0 = static_cast<int>(std::floor(fj));
        int k0 = static_cast<int>(std::floor(fk));
        
        float fx = fi - i0;
        float fy = fj - j0;
        float fz = fk - k0;
        
        return trilinearVelY(i0, j0, k0, fx, fy, fz);
    }
    
    float sampleVelZ(float fi, float fj, float fk) const {
        int i0 = static_cast<int>(std::floor(fi));
        int j0 = static_cast<int>(std::floor(fj));
        int k0 = static_cast<int>(std::floor(fk));
        
        float fx = fi - i0;
        float fy = fj - j0;
        float fz = fk - k0;
        
        return trilinearVelZ(i0, j0, k0, fx, fy, fz);
    }
    
    /// @brief Trilinear interpolation helper
    static float trilinear(const std::vector<float>& field, 
                           int i0, int j0, int k0, 
                           float fx, float fy, float fz,
                           int nx, int ny, int nz) {
        
        // OPEN BOUNDARY: If any part of the sample kernel is outside, we return 0.
        // This causes the fluid to "disappear" at the edges.
        if (i0 < 0 || i0 >= nx - 1 || j0 < 0 || j0 >= ny - 1 || k0 < 0 || k0 >= nz - 1) {
            return 0.0f;
        }

        auto idx = [nx, ny](int i, int j, int k) { 
            return static_cast<size_t>(i) + static_cast<size_t>(j) * nx + static_cast<size_t>(k) * nx * ny; 
        };
        
        int i1 = i0, i2 = i0 + 1;
        int j1 = j0, j2 = j0 + 1;
        int k1 = k0, k2 = k0 + 1;
        
        float c000 = field[idx(i1, j1, k1)];
        float c100 = field[idx(i2, j1, k1)];
        float c010 = field[idx(i1, j2, k1)];
        float c110 = field[idx(i2, j2, k1)];
        float c001 = field[idx(i1, j1, k2)];
        float c101 = field[idx(i2, j1, k2)];
        float c011 = field[idx(i1, j2, k2)];
        float c111 = field[idx(i2, j2, k2)];
        
        float c00 = c000 * (1 - fx) + c100 * fx;
        float c10 = c010 * (1 - fx) + c110 * fx;
        float c01 = c001 * (1 - fx) + c101 * fx;
        float c11 = c011 * (1 - fx) + c111 * fx;
        
        float c0 = c00 * (1 - fy) + c10 * fy;
        float c1 = c01 * (1 - fy) + c11 * fy;
        
        return c0 * (1 - fz) + c1 * fz;
    }
    
    float trilinearVelX(int i0, int j0, int k0, float fx, float fy, float fz) const {
        auto clamp = [](int v, int max_val) { return std::max(0, std::min(v, max_val - 1)); };
        
        int i1 = clamp(i0, nx + 1), i2 = clamp(i0 + 1, nx + 1);
        int j1 = clamp(j0, ny), j2 = clamp(j0 + 1, ny);
        int k1 = clamp(k0, nz), k2 = clamp(k0 + 1, nz);
        
        float c000 = vel_x[velXIndex(i1, j1, k1)];
        float c100 = vel_x[velXIndex(i2, j1, k1)];
        float c010 = vel_x[velXIndex(i1, j2, k1)];
        float c110 = vel_x[velXIndex(i2, j2, k1)];
        float c001 = vel_x[velXIndex(i1, j1, k2)];
        float c101 = vel_x[velXIndex(i2, j1, k2)];
        float c011 = vel_x[velXIndex(i1, j2, k2)];
        float c111 = vel_x[velXIndex(i2, j2, k2)];
        
        float c00 = c000 * (1 - fx) + c100 * fx;
        float c10 = c010 * (1 - fx) + c110 * fx;
        float c01 = c001 * (1 - fx) + c101 * fx;
        float c11 = c011 * (1 - fx) + c111 * fx;
        
        float c0 = c00 * (1 - fy) + c10 * fy;
        float c1 = c01 * (1 - fy) + c11 * fy;
        
        return c0 * (1 - fz) + c1 * fz;
    }
    
    float trilinearVelY(int i0, int j0, int k0, float fx, float fy, float fz) const {
        auto clamp = [](int v, int max_val) { return std::max(0, std::min(v, max_val - 1)); };
        
        int i1 = clamp(i0, nx), i2 = clamp(i0 + 1, nx);
        int j1 = clamp(j0, ny + 1), j2 = clamp(j0 + 1, ny + 1);
        int k1 = clamp(k0, nz), k2 = clamp(k0 + 1, nz);
        
        float c000 = vel_y[velYIndex(i1, j1, k1)];
        float c100 = vel_y[velYIndex(i2, j1, k1)];
        float c010 = vel_y[velYIndex(i1, j2, k1)];
        float c110 = vel_y[velYIndex(i2, j2, k1)];
        float c001 = vel_y[velYIndex(i1, j1, k2)];
        float c101 = vel_y[velYIndex(i2, j1, k2)];
        float c011 = vel_y[velYIndex(i1, j2, k2)];
        float c111 = vel_y[velYIndex(i2, j2, k2)];
        
        float c00 = c000 * (1 - fx) + c100 * fx;
        float c10 = c010 * (1 - fx) + c110 * fx;
        float c01 = c001 * (1 - fx) + c101 * fx;
        float c11 = c011 * (1 - fx) + c111 * fx;
        
        float c0 = c00 * (1 - fy) + c10 * fy;
        float c1 = c01 * (1 - fy) + c11 * fy;
        
        return c0 * (1 - fz) + c1 * fz;
    }
    
    float trilinearVelZ(int i0, int j0, int k0, float fx, float fy, float fz) const {
        auto clamp = [](int v, int max_val) { return std::max(0, std::min(v, max_val - 1)); };
        
        int i1 = clamp(i0, nx), i2 = clamp(i0 + 1, nx);
        int j1 = clamp(j0, ny), j2 = clamp(j0 + 1, ny);
        int k1 = clamp(k0, nz + 1), k2 = clamp(k0 + 1, nz + 1);
        
        float c000 = vel_z[velZIndex(i1, j1, k1)];
        float c100 = vel_z[velZIndex(i2, j1, k1)];
        float c010 = vel_z[velZIndex(i1, j2, k1)];
        float c110 = vel_z[velZIndex(i2, j2, k1)];
        float c001 = vel_z[velZIndex(i1, j1, k2)];
        float c101 = vel_z[velZIndex(i2, j1, k2)];
        float c011 = vel_z[velZIndex(i1, j2, k2)];
        float c111 = vel_z[velZIndex(i2, j2, k2)];
        
        float c00 = c000 * (1 - fx) + c100 * fx;
        float c10 = c010 * (1 - fx) + c110 * fx;
        float c01 = c001 * (1 - fx) + c101 * fx;
        float c11 = c011 * (1 - fx) + c111 * fx;
        
        float c0 = c00 * (1 - fy) + c10 * fy;
        float c1 = c01 * (1 - fy) + c11 * fy;
        
        return c0 * (1 - fz) + c1 * fz;
    }
};

} // namespace FluidSim

