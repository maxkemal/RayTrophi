/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          FluidSimulationSystem.cpp
 * Author: Kemal Demirtas
 * License:       MIT
 * =========================================================================
 */

// Enable OpenVDB support for Fluid Simulation exports
#ifndef OPENVDB_ENABLED
#define OPENVDB_ENABLED
#endif

// Fix for linker errors: Ensure Imath/OpenEXR are treated as DLLs
#ifndef IMATH_DLL
#define IMATH_DLL
#endif

#include "Fluid/FluidSimulationSystem.h"
#include "Fluid/APICFluidSolver.h"
#include "Fluid/FluidObject.h"
#include <fstream>
#include <iostream>
#include <filesystem>

#ifdef OPENVDB_ENABLED
#include <openvdb/openvdb.h>
#endif

namespace RayTrophiSim {
namespace Fluid {

// Helper to load VDB frame into fluid grids directly from disk cache
#ifdef OPENVDB_ENABLED
bool loadVDBFrameIntoFluid(FluidObject& obj, const std::string& filepath) {
    try {
        openvdb::initialize();
        if (!std::filesystem::exists(filepath)) return false;
        
        openvdb::io::File file(filepath);
        file.open();
        
        openvdb::GridPtrVecPtr grids = file.getGrids();
        if (!grids || grids->empty()) {
            file.close();
            return false;
        }
        
        openvdb::FloatGrid::Ptr density_grid;
        openvdb::FloatGrid::Ptr sdf_grid;
        
        for (size_t i = 0; i < grids->size(); ++i) {
            openvdb::GridBase::Ptr grid = (*grids)[i];
            if (grid->getName() == "density") {
                density_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
            } else if (grid->getName() == "sdf") {
                sdf_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
            }
        }
        file.close();
        
        if (!sdf_grid) return false;
        
        // Reallocate/resize Fluid grid to match VDB metadata if needed
        // Keep bounds same and copy SDF data back into obj.sdf!
        obj.sdf.assign(obj.grid.getCellCount(), obj.level_set_params.narrow_band_voxels * obj.grid.voxel_size);
        
        obj.particles.clear();
        openvdb::FloatGrid::ConstAccessor accessor = sdf_grid->getConstAccessor();
        for (int k = 0; k < obj.grid.nz; ++k) {
            for (int j = 0; j < obj.grid.ny; ++j) {
                for (int i = 0; i < obj.grid.nx; ++i) {
                    openvdb::Coord coord(i, j, k);
                    float val = accessor.getValue(coord);
                    obj.sdf[obj.grid.cellIndex(i, j, k)] = val;
                    
                    // If in dynamic level-set boundary (inside fluid), reconstruct particle for sphere rendering
                    if (val <= 0.0f) {
                        Vec3 pos = obj.grid.origin + Vec3(
                            (static_cast<float>(i) + 0.5f) * obj.grid.voxel_size,
                            (static_cast<float>(j) + 0.5f) * obj.grid.voxel_size,
                            (static_cast<float>(k) + 0.5f) * obj.grid.voxel_size
                        );
                        obj.particles.emit(pos, Vec3(0.0f, 0.0f, 0.0f));
                    }
                }
            }
        }
        
        return true;
    } catch (...) {
        return false;
    }
}
#else
bool loadVDBFrameIntoFluid(FluidObject& obj, const std::string& filepath) {
    // Fallback binary reader
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return false;
    
    int32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x464c5544) return false;
    
    int32_t nx = 0, ny = 0, nz = 0;
    float voxel_size = 0.0f;
    file.read(reinterpret_cast<char*>(&nx), sizeof(nx));
    file.read(reinterpret_cast<char*>(&ny), sizeof(ny));
    file.read(reinterpret_cast<char*>(&nz), sizeof(nz));
    file.read(reinterpret_cast<char*>(&voxel_size), sizeof(voxel_size));
    
    if (nx != obj.grid.nx || ny != obj.grid.ny || nz != obj.grid.nz) {
        return false;
    }
    
    obj.sdf.assign(obj.grid.getCellCount(), obj.level_set_params.narrow_band_voxels * obj.grid.voxel_size);
    file.read(reinterpret_cast<char*>(obj.sdf.data()), obj.sdf.size() * sizeof(float));
    
    obj.particles.clear();
    for (int k = 0; k < obj.grid.nz; ++k) {
        for (int j = 0; j < obj.grid.ny; ++j) {
            for (int i = 0; i < obj.grid.nx; ++i) {
                float val = obj.sdf[obj.grid.cellIndex(i, j, k)];
                if (val <= 0.0f) {
                    Vec3 pos = obj.grid.origin + Vec3(
                        (static_cast<float>(i) + 0.5f) * obj.grid.voxel_size,
                        (static_cast<float>(j) + 0.5f) * obj.grid.voxel_size,
                        (static_cast<float>(k) + 0.5f) * obj.grid.voxel_size
                    );
                    obj.particles.emit(pos, Vec3(0.0f, 0.0f, 0.0f));
                }
            }
        }
    }
    return true;
}
#endif

bool FluidObject::exportToVDB(const std::string& filepath) const {
#ifdef OPENVDB_ENABLED
    openvdb::initialize();
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) return false;
    
    // Build the SDF fresh at its (possibly refined) resolution so the bake
    // captures the full surface detail and we know the exact grid dims. The
    // live obj.sdf may be refined (size != sim cells, surface_resolution_
    // multiplier > 1), so NEVER index it with the sim-grid stride — drive every
    // loop and the VDB transform off the build's effective dims/voxel.
    std::vector<float> built_sdf;
    LevelSetStats ls{};
    buildLevelSet(particles, grid, level_set_params, built_sdf, &ls);
    if (built_sdf.empty() || ls.eff_nx <= 0) return false;

    const int   enx = ls.eff_nx;
    const int   eny = ls.eff_ny;
    const int   enz = ls.eff_nz;
    const float evoxel = ls.eff_voxel;
    auto eIndex = [enx, eny](int i, int j, int k) -> std::size_t {
        return static_cast<std::size_t>(i) +
               static_cast<std::size_t>(j) * static_cast<std::size_t>(enx) +
               static_cast<std::size_t>(k) * static_cast<std::size_t>(enx) *
               static_cast<std::size_t>(eny);
    };

    // density proxy; grad_width stays PHYSICAL (sim voxel) — phi is a physical
    // distance regardless of the refinement, so the band is invariant.
    const float grad_width = std::max(1.0f, surface_band_voxels) * grid.voxel_size;
    const float inv_w = 0.5f / grad_width;

    openvdb::FloatGrid::Ptr density_grid = openvdb::FloatGrid::create();
    density_grid->setName("density");
    density_grid->setGridClass(openvdb::GRID_FOG_VOLUME);
    openvdb::FloatGrid::Accessor density_accessor = density_grid->getAccessor();

    openvdb::FloatGrid::Ptr distance_grid = openvdb::FloatGrid::create();
    distance_grid->setName("sdf");
    distance_grid->setGridClass(openvdb::GRID_LEVEL_SET);
    openvdb::FloatGrid::Accessor distance_accessor = distance_grid->getAccessor();

    for (int k = 0; k < enz; ++k) {
        for (int j = 0; j < eny; ++j) {
            for (int i = 0; i < enx; ++i) {
                const float phi = built_sdf[eIndex(i, j, k)];
                float dproxy = 0.5f - phi * inv_w;
                if (dproxy < 0.0f) dproxy = 0.0f;
                if (dproxy > 1.0f) dproxy = 1.0f;
                if (dproxy > 0.0001f) density_accessor.setValue(openvdb::Coord(i, j, k), dproxy);
                distance_accessor.setValue(openvdb::Coord(i, j, k), phi);
            }
        }
    }

    // Set transforms aligned with MAC grid origin, at the refined voxel size.
    openvdb::math::Transform::Ptr vdb_transform =
        openvdb::math::Transform::createLinearTransform(evoxel);
    vdb_transform->postTranslate(openvdb::Vec3d(grid.origin.x, grid.origin.y, grid.origin.z));
    
    density_grid->setTransform(vdb_transform);
    distance_grid->setTransform(vdb_transform);
    
    openvdb::GridPtrVec grids;
    grids.push_back(density_grid);
    grids.push_back(distance_grid);
    
    try {
        openvdb::io::File file(filepath);
        file.write(grids);
        file.close();
        return true;
    } catch (...) {
        return false;
    }
#else
    std::ofstream file(filepath, std::ios::binary);
    if (!file) return false;

    // Build at the (possibly refined) resolution and write THOSE dims so the
    // reader reconstructs the exact grid the SDF was baked on.
    std::vector<float> built_sdf;
    LevelSetStats ls{};
    buildLevelSet(particles, grid, level_set_params, built_sdf, &ls);
    if (built_sdf.empty() || ls.eff_nx <= 0) { file.close(); return false; }

    int32_t magic = 0x464c5544; // "FLUD"
    file.write(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&ls.eff_nx), sizeof(ls.eff_nx));
    file.write(reinterpret_cast<const char*>(&ls.eff_ny), sizeof(ls.eff_ny));
    file.write(reinterpret_cast<const char*>(&ls.eff_nz), sizeof(ls.eff_nz));
    file.write(reinterpret_cast<const char*>(&ls.eff_voxel), sizeof(ls.eff_voxel));
    file.write(reinterpret_cast<const char*>(built_sdf.data()), built_sdf.size() * sizeof(float));
    file.close();
    return true;
#endif
}

} // namespace Fluid

void FluidSimulationSystem::setObjects(std::vector<Fluid::FluidObject>* objects) {
    objects_ = objects;
}

bool FluidSimulationSystem::enabled() const {
    return objects_ != nullptr && !objects_->empty();
}

void FluidSimulationSystem::step(const SimulationContext& context) {
    if (!objects_) return;

    for (auto& obj : *objects_) {
        if (!obj.enabled) continue;
        obj.ensureGrid();

        // ── VDB Sequence Cache Bridge ──
        if (obj.use_vdb_cache && !obj.vdb_cache_pattern.empty()) {
            std::string filepath = obj.vdb_cache_pattern;
            std::string placeholder(obj.vdb_cache_digits, '#');
            size_t pos = filepath.find(placeholder);
            if (pos != std::string::npos) {
                char buf[32];
                std::string fmt = "%0" + std::to_string(obj.vdb_cache_digits) + "d";
                snprintf(buf, sizeof(buf), fmt.c_str(), context.frame);
                filepath.replace(pos, obj.vdb_cache_digits, buf);
            }
            Fluid::loadVDBFrameIntoFluid(obj, filepath);
            continue;
        }

        // Apply any pending seed request (set by the UI).
        if (obj.pending_seed) {
            if (obj.replace_on_seed) {
                obj.particles.clear();
                obj.grid.clear();
                obj.ensureGrid();
            }
            Fluid::seedBox(obj.particles,
                           obj.grid,
                           obj.seed_min,
                           obj.seed_max,
                           obj.seed_particles_per_cell,
                           /*seed=*/static_cast<uint32_t>(obj.id) * 2654435761u,
                           obj.particles.size() < obj.max_particles
                               ? obj.max_particles - obj.particles.size()
                               : 0u);
            obj.pending_seed = false;
        }

        if (obj.particles.empty()) {
            obj.stats = Fluid::APICSolverStats{};
            continue;
        }

        Fluid::step(obj.particles,
                    obj.grid,
                    obj.params,
                    context.dt,
                    context.force_snapshot,
                    context.time_seconds,
                    &obj.stats);
    }
}

} // namespace RayTrophiSim
