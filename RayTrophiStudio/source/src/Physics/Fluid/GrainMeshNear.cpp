#include "Fluid/GrainMeshNear.h"

#include "Hittable.h"
#include "GrainGenerator.h"
#include "MaterialManager.h"
#include "ParticleSimulation.h"
#include "Fluid/FluidRenderProxy.h"
#include "Fluid/MatterSubstanceState.h"
#include "Fluid/SubstanceTag.h"
#include "Triangle.h"
#include "TriangleMesh.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <unordered_map>

namespace RayTrophiSim::Fluid {
namespace {

std::mutex g_stats_mutex;
std::unordered_map<std::string, GrainMeshNearStats> g_stats;

uint64_t mix64(uint64_t x) {
    // splitmix64 finalizer: well distributed for consecutive particle ids.
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

float unit01(uint64_t bits) {
    return static_cast<float>(bits >> 40) * (1.0f / 16777216.0f);
}


} // namespace

bool gatherNormalizedSceneFaces(const std::vector<std::shared_ptr<Hittable>>& objects,
                                const std::string& node_name,
                                std::vector<NormalizedSceneFace>& out) {
    out.clear();
    if (node_name.empty()) {
        return false;
    }

    Vec3 mn(1e30f, 1e30f, 1e30f), mx(-1e30f, -1e30f, -1e30f);
    for (const auto& obj : objects) {
        auto mesh = std::dynamic_pointer_cast<TriangleMesh>(obj);
        if (!mesh || mesh->nodeName != node_name || !mesh->geometry) {
            continue;
        }
        const DNA::GeometryDetail& g = *mesh->geometry;
        const Vec3* P = g.get_attribute_data<Vec3>("P");
        const Vec3* P_orig = g.get_attribute_data<Vec3>("P_orig");
        const Vec3* N = g.get_attribute_data<Vec3>("N");
        const Vec3* N_orig = g.get_attribute_data<Vec3>("N_orig");
        const Vec2* uv = g.get_attribute_data<Vec2>("uv");
        const uint16_t* material = g.get_attribute_data<uint16_t>("materialID");
        if ((!P && !P_orig) || g.indices.size() < 3) {
            continue;
        }
        const Matrix4x4 xf = mesh->transform
            ? mesh->transform->getFinal() : Matrix4x4::identity();
        const Matrix4x4 nt = mesh->transform
            ? mesh->transform->getNormalTransform() : Matrix4x4::identity();
        const std::size_t vc = g.get_vertex_count();
        for (std::size_t f = 0; f + 2 < g.indices.size(); f += 3) {
            NormalizedSceneFace face;
            bool valid = true;
            for (int c = 0; c < 3; ++c) {
                const uint32_t vi = g.indices[f + static_cast<std::size_t>(c)];
                if (vi >= vc) {
                    valid = false;
                    break;
                }
                face.p[c] = P_orig ? xf.transform_point(P_orig[vi]) : P[vi];
                if (!std::isfinite(face.p[c].x) || !std::isfinite(face.p[c].y) ||
                    !std::isfinite(face.p[c].z)) {
                    valid = false;
                    break;
                }
                const Vec3 local_n = N_orig ? N_orig[vi] : (N ? N[vi] : Vec3(0.0f, 1.0f, 0.0f));
                face.n[c] = N_orig ? nt.transform_vector(local_n).normalize() : local_n.normalize();
                face.uv[c] = uv ? uv[vi] : Vec2(0.0f, 0.0f);
                mn = Vec3::min(mn, face.p[c]);
                mx = Vec3::max(mx, face.p[c]);
            }
            if (!valid) {
                out.clear();
                return false;
            }
            const Vec3 cross = Vec3::cross(face.p[1] - face.p[0], face.p[2] - face.p[0]);
            const float area = cross.length();
            if (!std::isfinite(area) || area <= 0.0f) {
                continue;
            }
            if (!N && !N_orig) {
                for (auto& normal : face.n) {
                    normal = cross * (1.0f / area);
                }
            }
            const uint16_t id = material ? material[g.indices[f]] : 0;
            face.material = id == MaterialManager::INVALID_MATERIAL_ID ? 0 : id;
            out.push_back(face);
        }
    }
    if (out.empty()) {
        return false;
    }

    const Vec3 center = (mn + mx) * 0.5f;
    const Vec3 ext = mx - mn;
    const float inv = 1.0f / std::max({ext.x, ext.y, ext.z, 1e-4f});
    for (auto& face : out) {
        for (auto& p : face.p) {
            p = (p - center) * inv;
        }
    }
    return true;
}

bool buildGrainExemplars(const std::vector<std::shared_ptr<Hittable>>& objects,
                         const std::vector<std::string>& nodes,
                         std::vector<GrainExemplar>& out, std::string& error,
                         const std::string& shape) {
    out.clear();
    error.clear();
    if (shape == "rock" || shape == "voronoi") {
        GrainGeneratorSettings settings;
        settings.type = shape;
        settings.count = 16;
        settings.roundness = shape == "rock" ? 1.0f : 0.0f;
        return generateGrainShapes(settings, out, error);
    }
    if (shape != "scene") {
        error = "unknown grain mesh shape: " + shape;
        return false;
    }
    if (nodes.empty()) {
        error = "no exemplar sources";
        return false;
    }
    for (const auto& node : nodes) {
        GrainExemplar exemplar;
        exemplar.node = node;
        if (!gatherNormalizedSceneFaces(objects, node, exemplar.faces)) {
            out.clear();
            error = "missing source: " + node;
            return false;
        }
        normalizeGrainShape(exemplar);
        out.push_back(std::move(exemplar));
    }
    return true;
}

uint64_t grainMeshSceneSignature(const std::vector<std::shared_ptr<Hittable>>& objects,
    const std::vector<std::string>& nodes) {
    uint64_t hash = 1469598103934665603ull;
    auto bytes = [&](const void* data, std::size_t size) {
        const auto* values = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            hash = (hash ^ values[i]) * 1099511628211ull;
        }
    };
    for (const auto& object : objects) {
        const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!mesh || !mesh->geometry ||
            std::find(nodes.begin(), nodes.end(), mesh->nodeName) == nodes.end()) {
            continue;
        }
        bytes(mesh->nodeName.data(), mesh->nodeName.size());
        const auto& geometry = *mesh->geometry;
        const std::size_t count = geometry.get_vertex_count();
        bytes(&count, sizeof(count));
        for (const char* key : {"P_orig", "P", "N_orig", "N"}) {
            if (const auto* values = geometry.get_attribute_data<Vec3>(key)) {
                for (std::size_t i = 0; i < count; ++i) {
                    // Hash components only: Vec3 may contain SIMD padding.
                    bytes(&values[i].x, sizeof(float));
                    bytes(&values[i].y, sizeof(float));
                    bytes(&values[i].z, sizeof(float));
                }
            }
        }
        if (const auto* values = geometry.get_attribute_data<Vec2>("uv")) {
            for (std::size_t i = 0; i < count; ++i) {
                bytes(&values[i].u, sizeof(float));
                bytes(&values[i].v, sizeof(float));
            }
        }
        if (const auto* values = geometry.get_attribute_data<uint16_t>("materialID")) {
            bytes(values, count * sizeof(uint16_t));
        }
        bytes(geometry.indices.data(), geometry.indices.size() * sizeof(uint32_t));
        const Matrix4x4 matrix = mesh->transform
            ? mesh->transform->getFinal() : Matrix4x4::identity();
        bytes(matrix.m, sizeof(matrix.m));
    }
    return hash;
}

bool isGrainMeshLibraryShape(const std::string& shape) {
    return shape == "scene" || shape == "rock" || shape == "voronoi";
}

uint32_t primarySplatVisualChildren(const SimulationGridDomainDesc& domain) {
    // Near meshes replace one whole primary carrier. Whitewater retains its
    // separate proxy refinement; toggling the mesh view never changes physics.
    if (domain.fluid_grain_mesh_near_enabled || domain.fluid_params.grain.enabled ||
        (domain.fluid_params.granular_enabled && domain.fluid_granular_physical_carriers)) {
        return 1u;
    }
    return static_cast<uint32_t>(std::clamp(domain.fluid_particle_visual_children,
        1, static_cast<int>(kFluidRenderProxyMaxChildren)));
}

float splatCarrierDiameter(const SimulationGridDomainDesc& domain,
    const FluidParticles& particles, std::size_t index, float parcel_diameter) {
    const auto model = index < particles.constitutive_model.size()
        ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[index])
        : MatterConstitutiveModel::Auto;
    const uint32_t tag = index < particles.substance_tag.size()
        ? particles.substance_tag[index] : kSubstanceUntagged;
    const bool dem = substanceTransportOwner(tag, model, domain.fluid_params) ==
        MatterTransportOwner::Grain;
    return dem ? 2.0f * matterGrainRadius(domain.fluid_params.grain, particles, index)
        : parcel_diameter;
}

uint32_t grainExemplarFor(uint64_t particle_id, uint32_t exemplar_count) {
    if (exemplar_count <= 1) {
        return 0;
    }
    return static_cast<uint32_t>(mix64(particle_id) % exemplar_count);
}

Vec3 grainEulerDegreesFor(uint64_t particle_id) {
    const uint64_t a = mix64(particle_id ^ 0x5851f42d4c957f2dull);
    const uint64_t b = mix64(a);
    const uint64_t c = mix64(b);
    // Euler angles are not uniform on SO(3); a grain pile does not show it,
    // and the measurement this serves does not depend on it.
    return Vec3(unit01(a) * 360.0f, unit01(b) * 360.0f, unit01(c) * 360.0f);
}

void buildGrainExemplarTriangles(const GrainExemplar& exemplar, int material,
                                 const std::string& node_name,
                                 std::vector<std::shared_ptr<Triangle>>& out) {
    out.clear();
    out.reserve(exemplar.faces.size());
    for (const auto& f : exemplar.faces) {
        const uint16_t face_material = material < 0
            ? f.material : static_cast<uint16_t>(material);
        auto tri = std::make_shared<Triangle>(f.p[0], f.p[1], f.p[2],
            f.n[0], f.n[1], f.n[2], f.uv[0], f.uv[1], f.uv[2], face_material);
        tri->setNodeName(node_name);
        out.push_back(std::move(tri));
    }
}

void publishGrainMeshNearStats(const std::string& domain, const GrainMeshNearStats& stats) {
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    g_stats[domain] = stats;
}

bool readGrainMeshNearStats(const std::string& domain, GrainMeshNearStats& out) {
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    const auto it = g_stats.find(domain);
    if (it == g_stats.end()) {
        return false;
    }
    out = it->second;
    return true;
}

void clearGrainMeshNearStats(const std::string& domain) {
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    g_stats.erase(domain);
}

} // namespace RayTrophiSim::Fluid
