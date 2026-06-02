/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          SimCache.cpp
* Author:        Kemal Demirtas
* License:       MIT
* =========================================================================
*/
#include "SimCache.h"

#include "json.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

namespace RayTrophiSim {
namespace SimCache {

// ── Low-level binary helpers (little-endian host assumed: x86/x64) ───────────
namespace {

template <typename T>
inline void writePod(std::ostream& os, const T& v) {
    static_assert(std::is_trivially_copyable<T>::value, "POD only");
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
inline bool readPod(std::istream& is, T& v) {
    static_assert(std::is_trivially_copyable<T>::value, "POD only");
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    return static_cast<bool>(is);
}

// Vec3 is written component-wise so we never depend on its in-memory padding.
inline void writeVec3(std::ostream& os, const Vec3& v) {
    writePod(os, v.x); writePod(os, v.y); writePod(os, v.z);
}
inline bool readVec3(std::istream& is, Vec3& v) {
    return readPod(is, v.x) && readPod(is, v.y) && readPod(is, v.z);
}

// A float scalar array: u64 count followed by count floats. count==0 → absent.
inline void writeFloatArray(std::ostream& os, const std::vector<float>& a) {
    const uint64_t n = a.size();
    writePod(os, n);
    if (n) os.write(reinterpret_cast<const char*>(a.data()), n * sizeof(float));
}
inline bool readFloatArray(std::istream& is, std::vector<float>& a) {
    uint64_t n = 0;
    if (!readPod(is, n)) return false;
    a.resize(static_cast<size_t>(n));
    if (n) is.read(reinterpret_cast<char*>(a.data()), n * sizeof(float));
    return static_cast<bool>(is);
}

inline std::string framePath(const std::string& dir, uint32_t system_id, int frame) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "sys%u_f%06d.rtfc", system_id, frame);
    return (fs::path(dir) / buf).string();
}

inline std::string manifestPath(const std::string& dir) {
    return (fs::path(dir) / "manifest.json").string();
}

} // namespace

std::string frameFilePath(const std::string& cache_dir, uint32_t system_id, int frame) {
    return framePath(cache_dir, system_id, frame);
}

bool frameExists(const std::string& cache_dir, uint32_t system_id, int frame) {
    std::error_code ec;
    return fs::exists(framePath(cache_dir, system_id, frame), ec);
}

// ─────────────────────────────────────────────────────────────────────────────
// Write
// ─────────────────────────────────────────────────────────────────────────────
bool writeSystemFrame(const std::string& cache_dir, uint32_t system_id, int frame,
                      const std::vector<SimulationGridDomainState>& domains) {
    std::error_code ec;
    fs::create_directories(cache_dir, ec);

    std::ofstream os(framePath(cache_dir, system_id, frame), std::ios::binary | std::ios::trunc);
    if (!os) return false;

    writePod(os, kMagic);
    writePod(os, kVersion);
    writePod(os, static_cast<uint32_t>(domains.size()));

    for (const auto& d : domains) {
        writePod(os, static_cast<uint32_t>(d.type));
        writePod(os, d.channels);
        writePod(os, static_cast<uint8_t>(d.valid ? 1 : 0));
        writePod(os, d.max_density);
        writePod(os, static_cast<uint64_t>(d.active_density_cells));

        writeVec3(os, d.bounds_min);
        writeVec3(os, d.bounds_max);
        writePod(os, d.resolution_x);
        writePod(os, d.resolution_y);
        writePod(os, d.resolution_z);
        writePod(os, d.voxel_size);
        writeVec3(os, d.domain_motion_delta);

        // Grid metadata (may differ slightly from state-level res; both restored).
        writePod(os, d.grid.nx);
        writePod(os, d.grid.ny);
        writePod(os, d.grid.nz);
        writePod(os, d.grid.voxel_size);
        writeVec3(os, d.grid.origin);

        // Scalar fields — only the ones the render path consumes. Each may be
        // empty (count 0), e.g. a fluid domain whose scratch grid was cleared.
        writeFloatArray(os, d.grid.density);
        writeFloatArray(os, d.grid.temperature);
        writeFloatArray(os, d.grid.fuel);
        writeFloatArray(os, d.grid.interaction);

        // Fluid particles — positions only (render source of truth).
        const uint64_t pcount = d.particles.position.size();
        writePod(os, pcount);
        for (uint64_t i = 0; i < pcount; ++i) writeVec3(os, d.particles.position[i]);

        // Foam — position + type + remaining lifetime.
        const uint64_t fcount = d.foam.position.size();
        writePod(os, fcount);
        for (uint64_t i = 0; i < fcount; ++i) writeVec3(os, d.foam.position[i]);
        if (fcount) {
            os.write(reinterpret_cast<const char*>(d.foam.type.data()), fcount * sizeof(uint8_t));
            os.write(reinterpret_cast<const char*>(d.foam.lifetime.data()), fcount * sizeof(float));
        }
    }

    return static_cast<bool>(os);
}

// ─────────────────────────────────────────────────────────────────────────────
// Read
// ─────────────────────────────────────────────────────────────────────────────
bool readSystemFrame(const std::string& cache_dir, uint32_t system_id, int frame,
                     std::vector<SimulationGridDomainState>& out_domains) {
    std::ifstream is(framePath(cache_dir, system_id, frame), std::ios::binary);
    if (!is) return false;

    uint32_t magic = 0, version = 0, domain_count = 0;
    if (!readPod(is, magic) || magic != kMagic) return false;
    if (!readPod(is, version) || version != kVersion) return false;
    if (!readPod(is, domain_count)) return false;

    out_domains.clear();
    out_domains.resize(domain_count);

    for (uint32_t di = 0; di < domain_count; ++di) {
        SimulationGridDomainState& d = out_domains[di];

        uint32_t type_u = 0;
        uint8_t valid_u = 0;
        if (!readPod(is, type_u)) return false;
        d.type = static_cast<SimulationDomainType>(type_u);
        if (!readPod(is, d.channels)) return false;
        if (!readPod(is, valid_u)) return false;
        d.valid = (valid_u != 0);
        if (!readPod(is, d.max_density)) return false;
        uint64_t active_cells = 0;
        if (!readPod(is, active_cells)) return false;
        d.active_density_cells = static_cast<size_t>(active_cells);

        if (!readVec3(is, d.bounds_min)) return false;
        if (!readVec3(is, d.bounds_max)) return false;
        if (!readPod(is, d.resolution_x)) return false;
        if (!readPod(is, d.resolution_y)) return false;
        if (!readPod(is, d.resolution_z)) return false;
        if (!readPod(is, d.voxel_size)) return false;
        if (!readVec3(is, d.domain_motion_delta)) return false;

        int gnx = 0, gny = 0, gnz = 0;
        float gvoxel = 0.1f;
        Vec3 gorigin(0.0f);
        if (!readPod(is, gnx) || !readPod(is, gny) || !readPod(is, gnz)) return false;
        if (!readPod(is, gvoxel)) return false;
        if (!readVec3(is, gorigin)) return false;

        // Reconstruct the grid (allocates all fields zero-filled), then overwrite
        // the present scalar fields. Velocity stays zero — unused for rendering.
        d.grid.resize(gnx, gny, gnz, gvoxel, gorigin);

        std::vector<float> density, temperature, fuel, interaction;
        if (!readFloatArray(is, density))     return false;
        if (!readFloatArray(is, temperature)) return false;
        if (!readFloatArray(is, fuel))        return false;
        if (!readFloatArray(is, interaction)) return false;
        if (!density.empty())     d.grid.density     = std::move(density);
        if (!temperature.empty()) d.grid.temperature = std::move(temperature);
        if (!fuel.empty())        d.grid.fuel        = std::move(fuel);
        if (!interaction.empty()) d.grid.interaction = std::move(interaction);

        // Fluid particles — positions restored; velocity/affine/flags zeroed to
        // match the count so any consumer iterating them in lockstep stays valid.
        uint64_t pcount = 0;
        if (!readPod(is, pcount)) return false;
        d.particles.clear();
        d.particles.position.resize(static_cast<size_t>(pcount));
        for (uint64_t i = 0; i < pcount; ++i) {
            if (!readVec3(is, d.particles.position[i])) return false;
        }
        d.particles.velocity.assign(static_cast<size_t>(pcount), Vec3(0.0f));
        d.particles.affine.assign(static_cast<size_t>(pcount), Fluid::AffineC{});
        d.particles.flags.assign(static_cast<size_t>(pcount), 0u);

        // Foam — position + type + lifetime restored; velocity zeroed.
        uint64_t fcount = 0;
        if (!readPod(is, fcount)) return false;
        d.foam.clear();
        d.foam.position.resize(static_cast<size_t>(fcount));
        for (uint64_t i = 0; i < fcount; ++i) {
            if (!readVec3(is, d.foam.position[i])) return false;
        }
        d.foam.type.resize(static_cast<size_t>(fcount));
        d.foam.lifetime.resize(static_cast<size_t>(fcount));
        if (fcount) {
            is.read(reinterpret_cast<char*>(d.foam.type.data()), fcount * sizeof(uint8_t));
            is.read(reinterpret_cast<char*>(d.foam.lifetime.data()), fcount * sizeof(float));
        }
        d.foam.velocity.assign(static_cast<size_t>(fcount), Vec3(0.0f));

        if (!is) return false;
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Manifest (JSON)
// ─────────────────────────────────────────────────────────────────────────────
bool writeManifest(const std::string& cache_dir, const Manifest& m) {
    std::error_code ec;
    fs::create_directories(cache_dir, ec);

    nlohmann::json j;
    j["version"]     = m.version;
    j["start_frame"] = m.start_frame;
    j["end_frame"]   = m.end_frame;
    j["fps"]         = m.fps;
    j["type"]        = "render_only";
    j["systems"]     = nlohmann::json::array();
    for (const auto& s : m.systems) {
        nlohmann::json sj;
        sj["id"]           = s.id;
        // Hash stored as a hex string so the full 64 bits survive JSON's double.
        char hex[32];
        std::snprintf(hex, sizeof(hex), "0x%016llx", static_cast<unsigned long long>(s.config_hash));
        sj["config_hash"]  = hex;
        sj["domain_count"] = s.domain_count;
        j["systems"].push_back(sj);
    }

    std::ofstream os(manifestPath(cache_dir), std::ios::trunc);
    if (!os) return false;
    os << j.dump(2);
    return static_cast<bool>(os);
}

bool readManifest(const std::string& cache_dir, Manifest& out) {
    std::ifstream is(manifestPath(cache_dir));
    if (!is) return false;

    nlohmann::json j;
    try {
        is >> j;
    } catch (...) {
        return false;
    }

    out = Manifest{};
    if (j.contains("version"))     out.version     = j["version"].get<uint32_t>();
    if (j.contains("start_frame")) out.start_frame = j["start_frame"].get<int>();
    if (j.contains("end_frame"))   out.end_frame   = j["end_frame"].get<int>();
    if (j.contains("fps"))         out.fps         = j["fps"].get<float>();
    if (j.contains("systems") && j["systems"].is_array()) {
        for (const auto& sj : j["systems"]) {
            SystemManifest s;
            if (sj.contains("id"))           s.id = sj["id"].get<uint32_t>();
            if (sj.contains("domain_count")) s.domain_count = sj["domain_count"].get<int>();
            if (sj.contains("config_hash")) {
                const std::string h = sj["config_hash"].get<std::string>();
                s.config_hash = std::strtoull(h.c_str(), nullptr, 0);
            }
            out.systems.push_back(s);
        }
    }
    return true;
}

bool clearCache(const std::string& cache_dir) {
    std::error_code ec;
    if (!fs::exists(cache_dir, ec)) return true;
    // Remove only our own artifacts so an accidental wrong dir doesn't nuke data.
    for (const auto& entry : fs::directory_iterator(cache_dir, ec)) {
        const std::string name = entry.path().filename().string();
        const std::string ext  = entry.path().extension().string();
        if (ext == ".rtfc" || name == "manifest.json") {
            fs::remove(entry.path(), ec);
        }
    }
    return true;
}

} // namespace SimCache
} // namespace RayTrophiSim
