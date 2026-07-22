/*
 * RayTrophi Studio - deterministic hair/groom scripting facade (Faz 5.4a)
 */

#include "RtApiInternal.h"
#include "Hair/HairSystem.h"
#include "Renderer.h"
#include "Triangle.h"
#include "TriangleMesh.h"
#include <algorithm>

namespace rtapi {
namespace {

HairSettings fromCore(const Hair::HairGenerationParams& p) {
    HairSettings s;
    s.guide_count = p.guideCount; s.children_per_guide = p.interpolatedPerGuide;
    s.points_per_strand = p.pointsPerStrand; s.length = p.length;
    s.length_variation = p.lengthVariation; s.root_radius = p.rootRadius;
    s.tip_radius = p.tipRadius; s.clumpiness = p.clumpiness;
    s.child_radius = p.childRadius; s.curl_frequency = p.curlFrequency;
    s.curl_radius = p.curlRadius; s.wave_frequency = p.waveFrequency;
    s.wave_amplitude = p.waveAmplitude; s.frizz = p.frizz;
    s.roughness = p.roughness; s.gravity = p.gravity;
    s.force_influence = p.forceInfluence; s.use_dynamics = p.useDynamics;
    s.physics_damping = p.physicsDamping; s.physics_stiffness = p.physicsStiffness;
    s.physics_mass = p.physicsMass; s.use_tangent_shading = p.useTangentShading;
    s.use_bspline = p.useBSpline; s.subdivisions = p.subdivisions;
    return s;
}

Hair::HairGenerationParams toCore(const HairSettings& s) {
    Hair::HairGenerationParams p;
    p.guideCount = s.guide_count; p.interpolatedPerGuide = s.children_per_guide;
    p.pointsPerStrand = s.points_per_strand; p.length = s.length;
    p.lengthVariation = s.length_variation; p.rootRadius = s.root_radius;
    p.tipRadius = s.tip_radius; p.clumpiness = s.clumpiness;
    p.childRadius = s.child_radius; p.curlFrequency = s.curl_frequency;
    p.curlRadius = s.curl_radius; p.waveFrequency = s.wave_frequency;
    p.waveAmplitude = s.wave_amplitude; p.frizz = s.frizz;
    p.roughness = s.roughness; p.gravity = s.gravity;
    p.forceInfluence = s.force_influence; p.useDynamics = s.use_dynamics;
    p.physicsDamping = s.physics_damping; p.physicsStiffness = s.physics_stiffness;
    p.physicsMass = s.physics_mass; p.useTangentShading = s.use_tangent_shading;
    p.useBSpline = s.use_bspline; p.subdivisions = s.subdivisions;
    return p;
}

Result validate(const HairSettings& s) {
    if (s.points_per_strand < 2 || s.points_per_strand > 64)
        return Result::fail("hair points_per_strand must be between 2 and 64");
    if (s.length <= 0.0f) return Result::fail("hair length must be greater than zero");
    if (s.root_radius < 0.0f || s.tip_radius < 0.0f)
        return Result::fail("hair radii cannot be negative");
    if (s.subdivisions > 8) return Result::fail("hair subdivisions must be between 0 and 8");
    return Result::success();
}

HairGroomInfo describe(const Hair::HairGroom& groom) {
    HairGroomInfo info;
    info.name = groom.name; info.bound_mesh = groom.boundMeshName;
    info.guide_count = groom.guides.size(); info.child_count = groom.interpolated.size();
    for (const auto& strand : groom.guides) info.point_count += strand.points.size();
    for (const auto& strand : groom.interpolated) info.point_count += strand.points.size();
    info.material = groom.materialName; info.visible = groom.isVisible;
    info.dirty = groom.isDirty; info.settings = fromCore(groom.params);
    return info;
}

std::vector<std::shared_ptr<Triangle>> meshTriangles(UIContext& ctx, const std::string& mesh_name) {
    std::vector<std::shared_ptr<Triangle>> triangles;
    for (const auto& object : ctx.scene.world.objects) {
        if (auto tri = std::dynamic_pointer_cast<Triangle>(object)) {
            if (tri->getNodeName() == mesh_name) triangles.push_back(tri);
        } else if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object)) {
            if (mesh->nodeName == mesh_name && mesh->geometry) {
                triangles.reserve(triangles.size() + mesh->num_triangles());
                for (size_t i = 0; i < mesh->num_triangles(); ++i)
                    triangles.push_back(std::make_shared<Triangle>(mesh, static_cast<uint32_t>(i)));
            }
        }
    }
    return triangles;
}

std::string uniqueName(Hair::HairSystem& system, const std::string& requested) {
    const std::string base = requested.empty() ? "HairGroom" : requested;
    std::string candidate = base;
    for (int suffix = 1; system.exists(candidate); ++suffix)
        candidate = base + "_" + std::to_string(suffix);
    return candidate;
}

void syncHairRenderer() {
    auto& renderer = g_ctx->renderer;
    renderer.getHairSystem().buildBVH(!renderer.hideInterpolatedHair);
    renderer.uploadHairToGPU();
    renderer.resetCPUAccumulation();
}

} // namespace

Result listHairGrooms(std::vector<HairGroomInfo>& out_grooms) {
    if (!g_ctx) return notBound();
    out_grooms.clear();
    auto& system = g_ctx->renderer.getHairSystem();
    for (const auto& name : system.getGroomNames()) {
        if (const auto* groom = system.getGroom(name)) out_grooms.push_back(describe(*groom));
    }
    std::sort(out_grooms.begin(), out_grooms.end(),
              [](const HairGroomInfo& a, const HairGroomInfo& b) { return a.name < b.name; });
    return Result::success();
}

Result getHairGroom(const std::string& groom_name, HairGroomInfo& out_info) {
    if (!g_ctx) return notBound();
    const auto* groom = g_ctx->renderer.getHairSystem().getGroom(groom_name);
    if (!groom) return Result::fail("hair groom not found: " + groom_name);
    out_info = describe(*groom);
    return Result::success();
}

Result createHairGroom(const std::string& mesh_name, const std::string& requested_name,
                       const HairSettings& settings, HairGroomInfo& out_info) {
    if (!g_ctx) return notBound();
    Result valid = validate(settings); if (!valid.ok) return valid;
    auto triangles = meshTriangles(*g_ctx, mesh_name);
    if (triangles.empty()) return Result::fail("hair target mesh not found or has no triangles: " + mesh_name);
    auto& system = g_ctx->renderer.getHairSystem();
    const std::string name = uniqueName(system, requested_name);
    system.generateOnMesh(triangles, toCore(settings), name);
    syncHairRenderer();
    return getHairGroom(name, out_info);
}

Result removeHairGroom(const std::string& groom_name) {
    if (!g_ctx) return notBound();
    auto& system = g_ctx->renderer.getHairSystem();
    if (!system.exists(groom_name)) return Result::fail("hair groom not found: " + groom_name);
    system.removeGroom(groom_name); syncHairRenderer();
    return Result::success();
}

Result renameHairGroom(const std::string& groom_name, const std::string& new_name,
                       HairGroomInfo& out_info) {
    if (!g_ctx) return notBound();
    if (new_name.empty()) return Result::fail("hair groom name cannot be empty");
    auto& system = g_ctx->renderer.getHairSystem();
    if (!system.exists(groom_name)) return Result::fail("hair groom not found: " + groom_name);
    if (system.exists(new_name)) return Result::fail("hair groom already exists: " + new_name);
    if (!system.renameGroom(groom_name, new_name)) return Result::fail("failed to rename hair groom");
    syncHairRenderer();
    return getHairGroom(new_name, out_info);
}

Result updateHairGroom(const std::string& groom_name, const HairSettings& settings,
                       const bool* visible) {
    if (!g_ctx) return notBound();
    Result valid = validate(settings); if (!valid.ok) return valid;
    auto& system = g_ctx->renderer.getHairSystem();
    auto* groom = system.getGroom(groom_name);
    if (!groom) return Result::fail("hair groom not found: " + groom_name);
    const auto triangles = groom->boundTriangles;
    if (triangles.empty()) return Result::fail("hair groom has no bound mesh triangles: " + groom_name);
    const auto material = groom->material;
    const std::string material_name = groom->materialName;
    const bool old_visible = groom->isVisible;
    system.generateOnMesh(triangles, toCore(settings), groom_name);
    groom = system.getGroom(groom_name);
    groom->material = material; groom->materialName = material_name;
    groom->isVisible = visible ? *visible : old_visible;
    syncHairRenderer();
    return Result::success();
}

Result restyleHairGroom(const std::string& groom_name) {
    if (!g_ctx) return notBound();
    auto& system = g_ctx->renderer.getHairSystem();
    if (!system.exists(groom_name)) return Result::fail("hair groom not found: " + groom_name);
    system.restyleGroom(groom_name); syncHairRenderer();
    return Result::success();
}

Result listHairPresets(std::vector<std::string>& out_presets) {
    out_presets = {"straight", "soft_clumps", "curly", "wavy", "wet", "short_fur", "wild_grass"};
    return Result::success();
}

Result applyHairPreset(const std::string& groom_name, const std::string& preset) {
    HairGroomInfo info;
    Result r = getHairGroom(groom_name, info); if (!r.ok) return r;
    HairSettings& s = info.settings;
    if (preset == "straight") {
        s.clumpiness = 0.0f; s.curl_frequency = 0.0f; s.wave_frequency = 0.0f;
        s.wave_amplitude = 0.0f; s.frizz = 0.0f; s.roughness = 0.0f;
    } else if (preset == "soft_clumps") {
        s.clumpiness = 0.65f; s.child_radius = 0.012f; s.frizz = 0.03f;
    } else if (preset == "curly") {
        s.curl_frequency = 6.0f; s.curl_radius = std::max(0.002f, s.length * 0.08f);
        s.clumpiness = 0.35f; s.frizz = 0.08f;
    } else if (preset == "wavy") {
        s.wave_frequency = 5.0f; s.wave_amplitude = std::max(0.002f, s.length * 0.06f);
        s.clumpiness = 0.25f;
    } else if (preset == "wet") {
        s.clumpiness = 0.95f; s.frizz = 0.0f; s.gravity = 1.5f;
        s.child_radius = 0.015f;
    } else if (preset == "short_fur") {
        s.length = 0.008f; s.length_variation = 0.12f; s.gravity = 0.05f;
        s.clumpiness = 0.15f; s.frizz = 0.08f;
    } else if (preset == "wild_grass") {
        s.length = 0.12f; s.length_variation = 0.6f; s.root_radius = 0.002f;
        s.tip_radius = 0.0004f; s.clumpiness = -0.4f; s.wave_frequency = 8.0f;
        s.wave_amplitude = 0.02f; s.frizz = 0.1f; s.gravity = 0.3f;
    } else {
        return Result::fail("unknown hair preset: " + preset);
    }
    return updateHairGroom(groom_name, s, nullptr);
}

Result trimHairGroom(const std::string& groom_name, float length_factor) {
    if (!g_ctx) return notBound();
    if (length_factor <= 0.0f || length_factor > 1.0f)
        return Result::fail("hair trim length_factor must be greater than 0 and at most 1");
    auto& system = g_ctx->renderer.getHairSystem();
    auto* groom = system.getGroom(groom_name);
    if (!groom) return Result::fail("hair groom not found: " + groom_name);
    for (auto& strand : groom->guides) {
        if (strand.groomedPositions.empty()) continue;
        strand.baseLength *= length_factor;
        const Vec3 root = strand.groomedPositions.front();
        for (size_t i = 1; i < strand.groomedPositions.size(); ++i)
            strand.groomedPositions[i] = root + (strand.groomedPositions[i] - root) * length_factor;
    }
    groom->params.length *= length_factor;
    groom->isDirty = true;
    system.bakeGroomToRest(groom_name);
    system.regenerateInterpolated(groom_name);
    syncHairRenderer();
    return Result::success();
}

Result growHairGroom(const std::string& groom_name, float length_factor) {
    if (!g_ctx) return notBound();
    if (length_factor <= 1.0f || length_factor > 100.0f)
        return Result::fail("hair grow length_factor must be greater than 1 and at most 100");
    auto& system = g_ctx->renderer.getHairSystem();
    auto* groom = system.getGroom(groom_name);
    if (!groom) return Result::fail("hair groom not found: " + groom_name);
    for (auto& strand : groom->guides) {
        if (strand.groomedPositions.empty()) continue;
        strand.baseLength *= length_factor;
        const Vec3 root = strand.groomedPositions.front();
        for (size_t i = 1; i < strand.groomedPositions.size(); ++i)
            strand.groomedPositions[i] = root + (strand.groomedPositions[i] - root) * length_factor;
    }
    groom->params.length *= length_factor;
    groom->isDirty = true;
    system.bakeGroomToRest(groom_name);
    system.regenerateInterpolated(groom_name);
    syncHairRenderer();
    return Result::success();
}

Result combHairGroom(const std::string& groom_name, Vec3 world_direction,
                     float strength, float root_stiffness) {
    if (!g_ctx) return notBound();
    if (strength < 0.0f || strength > 1.0f)
        return Result::fail("hair comb strength must be between 0 and 1");
    if (root_stiffness < 0.0f || root_stiffness > 1.0f)
        return Result::fail("hair comb root_stiffness must be between 0 and 1");
    if (world_direction.length() < 1e-6f)
        return Result::fail("hair comb direction cannot be zero");
    auto& system = g_ctx->renderer.getHairSystem();
    auto* groom = system.getGroom(groom_name);
    if (!groom) return Result::fail("hair groom not found: " + groom_name);
    Vec3 target = groom->transform.inverse().transform_vector(world_direction);
    const float target_length = target.length();
    if (target_length < 1e-6f) return Result::fail("hair comb direction became invalid in groom space");
    target = target / target_length;
    for (auto& strand : groom->guides) {
        auto& points = strand.groomedPositions;
        if (points.size() < 2) continue;
        const float segment_length = strand.baseLength / static_cast<float>(points.size() - 1);
        for (size_t i = 1; i < points.size(); ++i) {
            Vec3 current = points[i] - points[i - 1];
            const float current_length = current.length();
            current = current_length > 1e-6f ? current / current_length : strand.rootNormal;
            const float t = static_cast<float>(i) / static_cast<float>(points.size() - 1);
            const float influence = strength * ((1.0f - root_stiffness) + root_stiffness * t);
            Vec3 direction = Vec3::mix(current, target, influence);
            const float direction_length = direction.length();
            if (direction_length > 1e-6f) direction = direction / direction_length;
            points[i] = points[i - 1] + direction * segment_length;
        }
    }
    groom->isDirty = true;
    system.bakeGroomToRest(groom_name);
    system.regenerateInterpolated(groom_name);
    syncHairRenderer();
    return Result::success();
}

Result smoothHairGroom(const std::string& groom_name, float strength, int iterations) {
    if (!g_ctx) return notBound();
    if (strength < 0.0f || strength > 1.0f)
        return Result::fail("hair smooth strength must be between 0 and 1");
    if (iterations < 1 || iterations > 64)
        return Result::fail("hair smooth iterations must be between 1 and 64");
    auto& system = g_ctx->renderer.getHairSystem();
    auto* groom = system.getGroom(groom_name);
    if (!groom) return Result::fail("hair groom not found: " + groom_name);
    for (int pass = 0; pass < iterations; ++pass) {
        for (auto& strand : groom->guides) {
            auto& points = strand.groomedPositions;
            if (points.size() < 3) continue;
            std::vector<Vec3> next = points;
            for (size_t i = 1; i + 1 < points.size(); ++i)
                next[i] = Vec3::mix(points[i], (points[i - 1] + points[i + 1]) * 0.5f, strength);
            const float segment_length = strand.baseLength / static_cast<float>(points.size() - 1);
            for (size_t i = 1; i < next.size(); ++i) {
                Vec3 direction = next[i] - next[i - 1];
                const float length = direction.length();
                if (length > 1e-6f) next[i] = next[i - 1] + direction * (segment_length / length);
            }
            points = std::move(next);
        }
    }
    groom->isDirty = true;
    system.bakeGroomToRest(groom_name);
    system.regenerateInterpolated(groom_name);
    syncHairRenderer();
    return Result::success();
}

Result resetHairSimulation(const std::string& groom_name) {
    if (!g_ctx) return notBound();
    auto& system = g_ctx->renderer.getHairSystem();
    auto* groom = system.getGroom(groom_name);
    if (!groom) return Result::fail("hair groom not found: " + groom_name);
    auto reset = [](std::vector<Hair::HairStrand>& strands) {
        for (auto& strand : strands) strand.prevPositions = strand.groomedPositions;
    };
    reset(groom->guides); reset(groom->interpolated);
    groom->isDirty = true;
    syncHairRenderer();
    return Result::success();
}

Result bakeHairGroom(const std::string& groom_name) {
    if (!g_ctx) return notBound();
    auto& system = g_ctx->renderer.getHairSystem();
    if (!system.exists(groom_name)) return Result::fail("hair groom not found: " + groom_name);
    system.bakeGroomToRest(groom_name);
    system.regenerateInterpolated(groom_name);
    syncHairRenderer();
    return Result::success();
}

} // namespace rtapi
