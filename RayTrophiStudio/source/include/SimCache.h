/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          SimCache.h
* Author:        Kemal Demirtas
* License:       MIT
* =========================================================================
*/
#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// SimCache — render-only on-disk point cache for baked simulations.
// ─────────────────────────────────────────────────────────────────────────────
// A baked sequence is persisted as one small binary file per (system, frame).
// We store each grid domain's RENDER source-of-truth only:
//   • Fluid → particle positions + foam (position/type/lifetime)
//   • Gas   → grid scalar fields (density/temperature/fuel/flame)
//   • plus grid metadata (bounds / resolution / voxel / origin) for both.
// Velocity / APIC-affine / grid velocity fields are intentionally NOT stored —
// the simulation cannot be RESUMED from this cache, but every render
// representation (SurfaceSDF, NanoVDB volume, particle splat, foam spheres) is
// regenerated on load by the existing syncSimulationRenderVolumes + render
// bridge, exactly as the live sim does. This keeps files ~half size and the
// reader free of solver state.
//
// Layout on disk (next to the project file):
//   <project>.simcache/
//     manifest.json                 (config signature + frame range + fps)
//     sys<id>_f<######>.rtfc        (one binary frame per particle system)
//
// SimCache is pure I/O: the config signature it stores in the manifest is
// computed by the caller (SceneData) so this module stays decoupled from the
// scene/serializer layer.

#include "ParticleSimulation.h"   // RayTrophiSim::SimulationGridDomainState
#include "RigidBodySystem.h"      // RayTrophiSim::RigidBodyFrameState

#include <string>
#include <vector>
#include <cstdint>

namespace RayTrophiSim {
namespace SimCache {

constexpr uint32_t kMagic   = 0x43465452u; // 'RTFC'
constexpr uint32_t kVersion = 1u;

// Absolute path of the per-system, per-frame binary file inside cache_dir.
std::string frameFilePath(const std::string& cache_dir, uint32_t system_id, int frame);

// True if a frame file for (system_id, frame) exists in cache_dir.
bool frameExists(const std::string& cache_dir, uint32_t system_id, int frame);

// Write one system's domain states (render-only fields) for `frame`.
// Creates cache_dir if needed. Returns false on any I/O failure.
bool writeSystemFrame(const std::string& cache_dir, uint32_t system_id, int frame,
                      const std::vector<SimulationGridDomainState>& domains);

// Read one system's domain states back (render-only) for `frame`. Reconstructs
// grid metadata + present scalar fields + particles/foam; velocity & affine are
// left zero-sized-safe (resized to match counts, value 0). Returns false if the
// file is missing / corrupt / version-mismatched.
bool readSystemFrame(const std::string& cache_dir, uint32_t system_id, int frame,
                     std::vector<SimulationGridDomainState>& out_domains);

// ── Soft / cloth bodies ──────────────────────────────────────────────────────
// Soft-body deformation is mesh-resident (not a pose), so it is baked as one file
// per frame holding every soft body's deformed UNIQUE world vertices, keyed by the
// source object's node name. Lives in the same cache_dir as the fluid frames
// (soft_f<######>.rtfc). Render-only, like the rest of the cache.
struct SoftBodyFrame {
    std::string       name;       // source object nodeName
    std::vector<Vec3> vertices;   // welded/unique deformed positions (world space)
};

std::string softFrameFilePath(const std::string& cache_dir, int frame);
bool softFrameExists(const std::string& cache_dir, int frame);
bool writeSoftFrame(const std::string& cache_dir, int frame,
                    const std::vector<SoftBodyFrame>& bodies);
bool readSoftFrame(const std::string& cache_dir, int frame,
                   std::vector<SoftBodyFrame>& out_bodies);

// Dynamic rigid body poses live beside fluid/soft frames in the same cache dir.
// This makes disk replay match the RAM timeline cache instead of re-simulating
// rigid bodies against a restored fluid frame.
std::string rigidFrameFilePath(const std::string& cache_dir, int frame);
bool rigidFrameExists(const std::string& cache_dir, int frame);
bool writeRigidFrame(const std::string& cache_dir, int frame,
                     const std::vector<RigidBodyFrameState>& bodies);
bool readRigidFrame(const std::string& cache_dir, int frame,
                    std::vector<RigidBodyFrameState>& out_bodies);

// ── Manifest ────────────────────────────────────────────────────────────────
struct SystemManifest {
    uint32_t id = 0;
    uint64_t config_hash = 0;  // caller-computed signature of the system's config
    int      domain_count = 0;
};

struct Manifest {
    uint32_t version = kVersion;
    int      start_frame = 0;
    int      end_frame   = 0;
    float    fps = 24.0f;
    std::vector<SystemManifest> systems;
};

bool writeManifest(const std::string& cache_dir, const Manifest& m);
bool readManifest(const std::string& cache_dir, Manifest& out);

// Delete every cached frame + manifest under cache_dir (Clear Bake).
bool clearCache(const std::string& cache_dir);

} // namespace SimCache
} // namespace RayTrophiSim
