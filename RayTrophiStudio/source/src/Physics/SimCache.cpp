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

#include "Fluid/SubstanceTag.h"
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

inline std::string softFramePath(const std::string& dir, int frame) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "soft_f%06d.rtfc", frame);
    return (fs::path(dir) / buf).string();
}

inline std::string rigidFramePath(const std::string& dir, int frame) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "rigid_f%06d.rtfc", frame);
    return (fs::path(dir) / buf).string();
}

// Row-major, component-wise for the same reason as writeVec3: never depend on
// the class's in-memory padding or on it staying a plain float[4][4].
inline void writeMatrix4x4(std::ostream& os, const Matrix4x4& m) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) writePod(os, m.m[r][c]);
}
inline bool readMatrix4x4(std::istream& is, Matrix4x4& m) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            if (!readPod(is, m.m[r][c])) return false;
    return true;
}

inline void writeString(std::ostream& os, const std::string& s) {
    const uint32_t n = static_cast<uint32_t>(s.size());
    writePod(os, n);
    if (n) os.write(s.data(), n);
}
inline bool readString(std::istream& is, std::string& s) {
    uint32_t n = 0;
    if (!readPod(is, n)) return false;
    s.resize(n);
    if (n) is.read(&s[0], n);
    return static_cast<bool>(is);
}

} // namespace

std::string frameFilePath(const std::string& cache_dir, uint32_t system_id, int frame) {
    return framePath(cache_dir, system_id, frame);
}

bool frameExists(const std::string& cache_dir, uint32_t system_id, int frame) {
    std::error_code ec;
    return fs::exists(framePath(cache_dir, system_id, frame), ec);
}

// ── Soft / cloth body frames ─────────────────────────────────────────────────
std::string softFrameFilePath(const std::string& cache_dir, int frame) {
    return softFramePath(cache_dir, frame);
}

bool softFrameExists(const std::string& cache_dir, int frame) {
    std::error_code ec;
    return fs::exists(softFramePath(cache_dir, frame), ec);
}

bool writeSoftFrame(const std::string& cache_dir, int frame,
                    const std::vector<SoftBodyFrame>& bodies) {
    std::error_code ec;
    fs::create_directories(cache_dir, ec);

    std::ofstream os(softFramePath(cache_dir, frame), std::ios::binary | std::ios::trunc);
    if (!os) return false;

    writePod(os, kMagic);
    writePod(os, kVersion);
    writePod(os, static_cast<uint32_t>(bodies.size()));
    for (const auto& b : bodies) {
        writeString(os, b.name);
        writePod(os, static_cast<uint64_t>(b.vertices.size()));
        for (const Vec3& v : b.vertices) writeVec3(os, v);
    }
    return static_cast<bool>(os);
}

bool readSoftFrame(const std::string& cache_dir, int frame,
                   std::vector<SoftBodyFrame>& out_bodies) {
    out_bodies.clear();
    std::ifstream is(softFramePath(cache_dir, frame), std::ios::binary);
    if (!is) return false;

    uint32_t magic = 0, version = 0, count = 0;
    if (!readPod(is, magic) || !readPod(is, version) || !readPod(is, count)) return false;
    if (magic != kMagic || version != kVersion) return false;

    out_bodies.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        SoftBodyFrame b;
        uint64_t n = 0;
        if (!readString(is, b.name) || !readPod(is, n)) return false;
        b.vertices.resize(static_cast<size_t>(n));
        for (uint64_t k = 0; k < n; ++k) {
            if (!readVec3(is, b.vertices[static_cast<size_t>(k)])) return false;
        }
        out_bodies.push_back(std::move(b));
    }
    return true;
}

// ── Dynamic rigid body frames ────────────────────────────────────────────────
// Unlike the grid/soft caches these carry VELOCITIES as well as poses, because
// a rigid body is resumed from here (advanceRigidTimelineToFrame continues from
// a restored frame), not merely displayed. Restoring pose alone would leave a
// falling body with zero velocity and it would drop from rest on the next step.
std::string rigidFrameFilePath(const std::string& cache_dir, int frame) {
    return rigidFramePath(cache_dir, frame);
}

bool rigidFrameExists(const std::string& cache_dir, int frame) {
    std::error_code ec;
    return fs::exists(rigidFramePath(cache_dir, frame), ec);
}

bool writeRigidFrame(const std::string& cache_dir, int frame,
                     const std::vector<RigidBodyFrameState>& bodies) {
    std::error_code ec;
    fs::create_directories(cache_dir, ec);

    std::ofstream os(rigidFramePath(cache_dir, frame), std::ios::binary | std::ios::trunc);
    if (!os) return false;

    writePod(os, kMagic);
    writePod(os, kVersion);
    writePod(os, static_cast<uint32_t>(bodies.size()));
    for (const auto& b : bodies) {
        writeString(os, b.source_name);
        writeMatrix4x4(os, b.pivot);
        writeMatrix4x4(os, b.body_xf);
        writeVec3(os, b.lin_vel);
        writeVec3(os, b.ang_vel);
        writePod(os, static_cast<uint8_t>(b.valid ? 1u : 0u));
    }
    return static_cast<bool>(os);
}

bool readRigidFrame(const std::string& cache_dir, int frame,
                    std::vector<RigidBodyFrameState>& out_bodies) {
    out_bodies.clear();
    std::ifstream is(rigidFramePath(cache_dir, frame), std::ios::binary);
    if (!is) return false;

    uint32_t magic = 0, version = 0, count = 0;
    if (!readPod(is, magic) || !readPod(is, version) || !readPod(is, count)) return false;
    if (magic != kMagic || version != kVersion) return false;

    out_bodies.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        RigidBodyFrameState b;
        uint8_t valid = 0;
        if (!readString(is, b.source_name) ||
            !readMatrix4x4(is, b.pivot) ||
            !readMatrix4x4(is, b.body_xf) ||
            !readVec3(is, b.lin_vel) ||
            !readVec3(is, b.ang_vel) ||
            !readPod(is, valid)) {
            return false;
        }
        b.valid = (valid != 0u);
        out_bodies.push_back(std::move(b));
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Write
// ─────────────────────────────────────────────────────────────────────────────
bool writeSystemFrame(const std::string& cache_dir, uint32_t system_id, int frame,
                      const std::vector<SimulationGridDomainState>& domains,
                      const std::vector<MaterialStateFieldSnapshot>& msf) {
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

        // Fluid particles — position + material coordinate. Both are render
        // source of truth: position places the surface, uvw anchors what is
        // drawn on it. Velocity/affine stay out; the renderer never reads them.
        const uint64_t pcount = d.particles.position.size();
        writePod(os, pcount);
        for (uint64_t i = 0; i < pcount; ++i) writeVec3(os, d.particles.position[i]);
        // Written unconditionally at pcount entries so the reader never has to
        // guess. A state whose uvw sidecar is short (only possible if a producer
        // skipped it) falls back to the position, which is exactly the identity
        // seed — a wrong-but-stable texture rather than a coordinate of zero,
        // which would collapse the whole surface onto one texel.
        for (uint64_t i = 0; i < pcount; ++i) {
            writeVec3(os, i < d.particles.uvw.size() ? d.particles.uvw[i]
                                                     : d.particles.position[i]);
        }
        // Generation B, plus the schedule counter that says where in its cycle
        // the pair is. ★ The counter matters as much as the coordinates: the
        // blend weights are derived from it, so a frame restored without it
        // would mix the two generations at the wrong ratio and the texture would
        // jump on the first replayed frame.
        for (uint64_t i = 0; i < pcount; ++i) {
            writeVec3(os, i < d.particles.uvw_b.size() ? d.particles.uvw_b[i]
                        : (i < d.particles.uvw.size() ? d.particles.uvw[i]
                                                      : d.particles.position[i]));
        }
        writePod(os, d.particles.uvw_step);
        writePod(os, static_cast<int32_t>(d.particles.uvw_refresh_period));

        // ── Substance identity ───────────────────────────────────────────────
        // ★★★ WHICH substance this parcel is, not just where it is. Render
        // source of truth exactly like position and uvw: the isosurface looks up
        // look (and later physics) from this, so a cache without it replays a
        // mixed pour as a single anonymous liquid.
        //
        // ★ The reader USED to assign 0 here unconditionally, which is why this
        // is being added rather than merely serialised: identity was surviving
        // the whole live pipeline and then being erased at the cache boundary.
        // A bug of that shape only shows on PLAYBACK, so it reads as "the bake
        // is wrong" rather than as a missing field.
        for (uint64_t i = 0; i < pcount; ++i) {
            writePod(os, i < d.particles.substance_tag.size()
                             ? d.particles.substance_tag[i]
                             : Fluid::kSubstanceUntagged);
        }

        // Foam — position + type + remaining lifetime.
        const uint64_t fcount = d.foam.position.size();
        writePod(os, fcount);
        for (uint64_t i = 0; i < fcount; ++i) writeVec3(os, d.foam.position[i]);
        if (fcount) {
            os.write(reinterpret_cast<const char*>(d.foam.type.data()), fcount * sizeof(uint8_t));
            os.write(reinterpret_cast<const char*>(d.foam.lifetime.data()), fcount * sizeof(float));
        }
    }

    // ── Material State Field (v2) ────────────────────────────────────────────
    // Named channels rather than the runtime's packed stride: the stride carries
    // per-step scratch and a reserved slot, and binding the file format to it
    // would invalidate every cache the day a scratch slot is added.
    writePod(os, static_cast<uint32_t>(msf.size()));
    for (const auto& f : msf) {
        writeString(os, f.object_key);
        writePod(os, static_cast<int32_t>(f.mask_resolution));
        writePod(os, f.element_count);
        writeFloatArray(os, f.temperature);
        writeFloatArray(os, f.fuel);
        writeFloatArray(os, f.charred);
        writeFloatArray(os, f.moisture);
        writeFloatArray(os, f.melt);
        writeFloatArray(os, f.mass_loss);
        writeFloatArray(os, f.transferred_mass);
    }

    return static_cast<bool>(os);
}

// ─────────────────────────────────────────────────────────────────────────────
// Read
// ─────────────────────────────────────────────────────────────────────────────
bool readSystemFrame(const std::string& cache_dir, uint32_t system_id, int frame,
                     std::vector<SimulationGridDomainState>& out_domains,
                     std::vector<MaterialStateFieldSnapshot>& out_msf) {
    out_msf.clear();
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
        d.particles.uvw.resize(static_cast<size_t>(pcount));
        for (uint64_t i = 0; i < pcount; ++i) {
            if (!readVec3(is, d.particles.uvw[i])) return false;
        }
        d.particles.uvw_b.resize(static_cast<size_t>(pcount));
        for (uint64_t i = 0; i < pcount; ++i) {
            if (!readVec3(is, d.particles.uvw_b[i])) return false;
        }
        int32_t period = 0;
        if (!readPod(is, d.particles.uvw_step)) return false;
        if (!readPod(is, period)) return false;
        d.particles.uvw_refresh_period = period > 1 ? period : 240;
        d.particles.velocity.assign(static_cast<size_t>(pcount), Vec3(0.0f));
        d.particles.affine.assign(static_cast<size_t>(pcount), Fluid::AffineC{});
        d.particles.flags.assign(static_cast<size_t>(pcount), 0u);
        d.particles.mass_fraction.assign(static_cast<size_t>(pcount), 1.0f);
        d.particles.temperature.assign(static_cast<size_t>(pcount), 0.0f);
        d.particles.combustible_fraction.assign(static_cast<size_t>(pcount), 0.0f);
        // Playback caches are render-oriented and do not preserve constitutive
        // history, but every particle sidecar must still remain cardinality-safe
        // if the frame is inspected or resumed through the live domain path.
        d.particles.ensureGranularStateSize();
        d.particles.substance_tag.resize(static_cast<size_t>(pcount));
        for (uint64_t i = 0; i < pcount; ++i) {
            if (!readPod(is, d.particles.substance_tag[i])) return false;
        }

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

    // ── Material State Field (v2) ────────────────────────────────────────────
    uint32_t msf_count = 0;
    if (!readPod(is, msf_count)) return false;
    out_msf.reserve(msf_count);
    for (uint32_t i = 0; i < msf_count; ++i) {
        MaterialStateFieldSnapshot f;
        int32_t res = 0;
        if (!readString(is, f.object_key)) return false;
        if (!readPod(is, res)) return false;
        if (!readPod(is, f.element_count)) return false;
        f.mask_resolution = static_cast<int>(res);
        if (!readFloatArray(is, f.temperature)) return false;
        if (!readFloatArray(is, f.fuel))        return false;
        if (!readFloatArray(is, f.charred))     return false;
        if (!readFloatArray(is, f.moisture))    return false;
        if (!readFloatArray(is, f.melt))        return false;
        if (!readFloatArray(is, f.mass_loss))   return false;
        if (!readFloatArray(is, f.transferred_mass)) return false;
        // A truncated/inconsistent entry is dropped rather than returning false:
        // losing one object's burn marks is recoverable, refusing the whole frame
        // would drop a perfectly good fluid/gas bake with it.
        if (f.valid()) out_msf.push_back(std::move(f));
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
