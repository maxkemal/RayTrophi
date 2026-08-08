/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtApiForceField.cpp
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       MIT
* =========================================================================
*
* Force field facade (Faz 5.6a). One field feeds every simulation family at
* once through Physics::ForceFieldManager, which is why this is a small file
* with a large reach.
*
* Two things here are not obvious and must not be "simplified" away:
*
* 1. Every mutation ends in touchForceFields(), which is the force-field
*    panel's own post-edit step (invalidate the rigid-body simulation cache,
*    reset CPU accumulation, reset the backend's accumulation). The manager is
*    read live by the solvers, so the data lands without it — but the cached
*    rigid-body run and the already-converged image do not, and the edit looks
*    like a no-op until something unrelated dirties the scene.
*
* 2. createForceField() replays the per-type defaults from the panel's add
*    menu. A Vortex created without them is a bare rotation with no spiral or
*    lift, and a Turbulence field with use_noise=false emits nothing at all,
*    so a scripted field would silently differ from a UI one.
*/

#include "RtApiInternal.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "ForceField.h"

namespace rtapi {
namespace {

using Physics::FalloffType;
using Physics::ForceField;
using Physics::ForceFieldShape;
using Physics::ForceFieldType;

struct TypeName { ForceFieldType type; const char* name; };

const TypeName kTypes[] = {
    { ForceFieldType::Wind,             "wind" },
    { ForceFieldType::Gravity,          "gravity" },
    { ForceFieldType::Attractor,        "attractor" },
    { ForceFieldType::Repeller,         "repeller" },
    { ForceFieldType::Vortex,           "vortex" },
    { ForceFieldType::Turbulence,       "turbulence" },
    { ForceFieldType::CurlNoise,        "curlnoise" },
    { ForceFieldType::Drag,             "drag" },
    { ForceFieldType::Magnetic,         "magnetic" },
    { ForceFieldType::DirectionalNoise, "directionalnoise" },
    { ForceFieldType::Thermal,          "thermal" },
};

struct ShapeName { ForceFieldShape shape; const char* name; };

const ShapeName kShapes[] = {
    { ForceFieldShape::Infinite, "infinite" },
    { ForceFieldShape::Sphere,   "sphere" },
    { ForceFieldShape::Box,      "box" },
    { ForceFieldShape::Cylinder, "cylinder" },
    { ForceFieldShape::Cone,     "cone" },
};

struct FalloffName { FalloffType falloff; const char* name; };

const FalloffName kFalloffs[] = {
    { FalloffType::None,          "none" },
    { FalloffType::Linear,        "linear" },
    { FalloffType::Smooth,        "smooth" },
    { FalloffType::Sphere,        "sphere" },
    { FalloffType::InverseSquare, "inverse_square" },
    { FalloffType::Exponential,   "exponential" },
    { FalloffType::Custom,        "custom" },
};

std::string lowered(const std::string& text) {
    std::string out = text;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

// "curl_noise" / "Curl Noise" / "curlnoise" all reach the same enum: the panel
// spells these with spaces and the enum without, and a script should not have
// to know which. Separators are dropped on BOTH sides of a comparison, which
// is what lets a multi-word canonical name like "inverse_square" match the
// spelling a caller actually typed.
std::string canonical(const std::string& text) {
    std::string out;
    for (char c : lowered(text)) {
        if (c == ' ' || c == '-' || c == '_') continue;
        out.push_back(c);
    }
    return out;
}

const char* typeName(ForceFieldType type) {
    for (const TypeName& entry : kTypes)
        if (entry.type == type) return entry.name;
    return "wind";
}

const char* shapeName(ForceFieldShape shape) {
    for (const ShapeName& entry : kShapes)
        if (entry.shape == shape) return entry.name;
    return "sphere";
}

const char* falloffName(FalloffType falloff) {
    for (const FalloffName& entry : kFalloffs)
        if (entry.falloff == falloff) return entry.name;
    return "smooth";
}

std::string typeList() {
    std::string out;
    for (const TypeName& entry : kTypes) {
        if (!out.empty()) out += "|";
        out += entry.name;
    }
    return out;
}

bool parseType(const std::string& text, ForceFieldType& out) {
    const std::string key = canonical(text);
    for (const TypeName& entry : kTypes) {
        if (key == canonical(entry.name)) { out = entry.type; return true; }
    }
    return false;
}

bool parseShape(const std::string& text, ForceFieldShape& out) {
    const std::string key = canonical(text);
    for (const ShapeName& entry : kShapes) {
        if (key == canonical(entry.name)) { out = entry.shape; return true; }
    }
    return false;
}

bool parseFalloff(const std::string& text, FalloffType& out) {
    const std::string key = canonical(text);
    for (const FalloffName& entry : kFalloffs) {
        if (key == canonical(entry.name)) { out = entry.falloff; return true; }
    }
    return false;
}

Physics::ForceFieldManager& manager() { return g_ctx->scene.force_field_manager; }

// The panel's post-edit step. See the file header — this is the part that
// cannot be dropped.
void touchForceFields() {
    g_ctx->scene.invalidateRigidBodySimulationCache();
    g_ctx->renderer.resetCPUAccumulation();
    if (g_ctx->backend_ptr) g_ctx->backend_ptr->resetAccumulation();
}

// Fields are addressed by ID or name. An all-digit token is tried as an ID
// first, then as a name, so a field literally named "12" is still reachable.
std::shared_ptr<ForceField> resolve(const std::string& id_or_name) {
    Physics::ForceFieldManager& mgr = manager();
    const bool numeric = !id_or_name.empty() &&
        std::all_of(id_or_name.begin(), id_or_name.end(),
                    [](unsigned char c) { return std::isdigit(c) != 0; });
    if (numeric) {
        if (auto found = mgr.findById(std::atoi(id_or_name.c_str()))) return found;
    }
    return mgr.findByName(id_or_name);
}

std::string uniqueName(const std::string& requested, const std::string& fallback) {
    std::string base = requested.empty() ? fallback : requested;
    if (!manager().findByName(base)) return base;
    for (int suffix = 2; suffix < 10000; ++suffix) {
        const std::string candidate = base + " " + std::to_string(suffix);
        if (!manager().findByName(candidate)) return candidate;
    }
    return base;
}

ForceFieldInfo infoFromField(const ForceField& field) {
    ForceFieldInfo info;
    info.id = field.id;
    info.name = field.name;
    info.type = typeName(field.type);
    info.shape = shapeName(field.shape);
    info.falloff = falloffName(field.falloff_type);
    info.enabled = field.enabled;
    info.visible = field.visible;
    info.position = field.position;
    info.rotation = field.rotation;
    info.scale = field.scale;
    info.direction = field.direction;
    info.axis = field.axis;
    info.strength = field.strength;
    info.falloff_radius = field.falloff_radius;
    info.inner_radius = field.inner_radius;
    info.use_noise = field.use_noise;
    info.noise_octaves = field.noise.octaves;
    info.noise_seed = field.noise.seed;
    info.noise_frequency = field.noise.frequency;
    info.noise_lacunarity = field.noise.lacunarity;
    info.noise_persistence = field.noise.persistence;
    info.noise_amplitude = field.noise.amplitude;
    info.noise_speed = field.noise.speed;
    info.inward_force = field.inward_force;
    info.upward_force = field.upward_force;
    info.linear_drag = field.linear_drag;
    info.quadratic_drag = field.quadratic_drag;
    info.thermal_delta_kelvin = field.thermal_delta_kelvin;
    info.fluid_surface_drag = field.fluid_surface_drag;
    info.fluid_drag_coupling = field.fluid_drag_coupling;
    info.fluid_surface_depth = field.fluid_surface_depth;
    info.fluid_curl_detail = field.fluid_curl_detail;
    info.start_frame = field.start_frame;
    info.end_frame = field.end_frame;
    info.phase = field.phase;
    info.affects_gas = field.affects_gas;
    info.affects_particles = field.affects_particles;
    info.affects_cloth = field.affects_cloth;
    info.affects_rigidbody = field.affects_rigidbody;
    info.affects_fluid = field.affects_fluid;
    return info;
}

// Enum strings are validated BEFORE anything is written, so a rejected update
// leaves the field exactly as it was rather than half-applied.
Result applyInfoToField(const ForceFieldInfo& info, ForceField& field) {
    ForceFieldType type = field.type;
    if (!info.type.empty() && !parseType(info.type, type))
        return Result::fail("unknown force field type: " + info.type + " (" + typeList() + ")");
    ForceFieldShape shape = field.shape;
    if (!info.shape.empty() && !parseShape(info.shape, shape))
        return Result::fail("unknown force field shape: " + info.shape +
                            " (infinite|sphere|box|cylinder|cone)");
    FalloffType falloff = field.falloff_type;
    if (!info.falloff.empty() && !parseFalloff(info.falloff, falloff))
        return Result::fail("unknown falloff type: " + info.falloff +
                            " (none|linear|smooth|sphere|inverse_square|exponential|custom)");
    if (info.noise_octaves < 1 || info.noise_octaves > 8)
        return Result::fail("noise_octaves must be between 1 and 8");
    if (info.falloff_radius < 0.0f)
        return Result::fail("falloff_radius must not be negative");
    if (info.inner_radius > info.falloff_radius)
        return Result::fail("inner_radius must not exceed falloff_radius");

    field.type = type;
    field.shape = shape;
    field.falloff_type = falloff;
    if (!info.name.empty()) field.name = info.name;
    field.enabled = info.enabled;
    field.visible = info.visible;
    field.position = info.position;
    field.rotation = info.rotation;
    field.scale = info.scale;
    field.direction = info.direction;
    field.axis = info.axis;
    field.strength = info.strength;
    field.falloff_radius = info.falloff_radius;
    field.inner_radius = info.inner_radius;
    field.use_noise = info.use_noise;
    field.noise.octaves = info.noise_octaves;
    field.noise.seed = info.noise_seed;
    field.noise.frequency = info.noise_frequency;
    field.noise.lacunarity = info.noise_lacunarity;
    field.noise.persistence = info.noise_persistence;
    field.noise.amplitude = info.noise_amplitude;
    field.noise.speed = info.noise_speed;
    field.inward_force = info.inward_force;
    field.upward_force = info.upward_force;
    field.linear_drag = info.linear_drag;
    field.quadratic_drag = info.quadratic_drag;
    field.thermal_delta_kelvin = info.thermal_delta_kelvin;
    field.fluid_surface_drag = info.fluid_surface_drag;
    field.fluid_drag_coupling = info.fluid_drag_coupling;
    field.fluid_surface_depth = info.fluid_surface_depth;
    field.fluid_curl_detail = info.fluid_curl_detail;
    field.start_frame = info.start_frame;
    field.end_frame = info.end_frame;
    field.phase = info.phase;
    field.affects_gas = info.affects_gas;
    field.affects_particles = info.affects_particles;
    field.affects_cloth = info.affects_cloth;
    field.affects_rigidbody = info.affects_rigidbody;
    field.affects_fluid = info.affects_fluid;
    return Result::success();
}

} // namespace

std::vector<std::string> forceFieldTypes() {
    std::vector<std::string> out;
    out.reserve(sizeof(kTypes) / sizeof(kTypes[0]));
    for (const TypeName& entry : kTypes) out.emplace_back(entry.name);
    return out;
}

std::vector<ForceFieldInfo> listForceFields() {
    std::vector<ForceFieldInfo> out;
    if (!g_ctx) return out;
    const auto& fields = manager().getForceFields();
    out.reserve(fields.size());
    for (const auto& field : fields) {
        if (field) out.push_back(infoFromField(*field));
    }
    return out;
}

Result getForceField(const std::string& id_or_name, ForceFieldInfo& out) {
    if (!g_ctx) return notBound();
    auto field = resolve(id_or_name);
    if (!field) return Result::fail("force field not found: " + id_or_name);
    out = infoFromField(*field);
    return Result::success();
}

Result createForceField(const std::string& type, const std::string& requested_name,
                        ForceFieldInfo& out) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    ForceFieldType parsed = ForceFieldType::Wind;
    if (!parseType(type, parsed))
        return Result::fail("unknown force field type: " + type + " (" + typeList() + ")");

    auto field = std::make_shared<ForceField>();
    field->type = parsed;
    field->name = uniqueName(requested_name, std::string(typeName(parsed)) + " Field");

    // Mirrors the panel's add-field menu; see the file header for why.
    switch (parsed) {
        case ForceFieldType::Turbulence:
        case ForceFieldType::CurlNoise:
            field->use_noise = true;
            field->shape = ForceFieldShape::Sphere;
            break;
        case ForceFieldType::Vortex:
            field->shape = ForceFieldShape::Cylinder;
            field->inward_force = 0.5f;
            field->upward_force = 0.2f;
            break;
        case ForceFieldType::Drag:
            field->shape = ForceFieldShape::Sphere;
            field->linear_drag = 0.5f;
            break;
        case ForceFieldType::Thermal:
            field->shape = ForceFieldShape::Sphere;
            field->falloff_radius = 2.0f;
            field->thermal_delta_kelvin = 600.0f;
            // Thermal exerts no force; the snapshot zeroes its affect mask
            // anyway, but leaving these ticked reads as if it pushed gas around.
            field->affects_gas = false;
            field->affects_particles = false;
            field->affects_cloth = false;
            field->affects_rigidbody = false;
            field->affects_fluid = false;
            break;
        default:
            break;
    }

    manager().addForceField(field);
    touchForceFields();
    out = infoFromField(*field);
    return Result::success();
}

Result removeForceField(const std::string& id_or_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto field = resolve(id_or_name);
    if (!field) return Result::fail("force field not found: " + id_or_name);
    if (!manager().removeForceField(field))
        return Result::fail("could not remove force field: " + id_or_name);
    // The selection holds a shared_ptr, so a removed-but-selected field stays
    // alive rather than dangling — but the panel would then keep editing a
    // field the manager no longer owns, and those edits would go nowhere.
    if (g_ctx->selection.selected.type == SelectableType::ForceField &&
        g_ctx->selection.selected.force_field == field) {
        g_ctx->selection.clearSelection();
    }
    touchForceFields();
    return Result::success();
}

Result updateForceField(const std::string& id_or_name, const ForceFieldInfo& info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    auto field = resolve(id_or_name);
    if (!field) return Result::fail("force field not found: " + id_or_name);
    const Result applied = applyInfoToField(info, *field);
    if (!applied.ok) return applied;
    if (g_ctx->selection.selected.type == SelectableType::ForceField &&
        g_ctx->selection.selected.force_field == field) {
        g_ctx->selection.selected.name = field->name;
        g_ctx->selection.selected.position = field->position;
        g_ctx->selection.selected.rotation = field->rotation;
        g_ctx->selection.selected.scale = field->scale;
    }
    touchForceFields();
    return Result::success();
}

Result evaluateForceFields(Vec3 world_position, float time, Vec3 velocity, Vec3& out_force) {
    if (!g_ctx) return notBound();
    out_force = manager().evaluateAt(world_position, time, velocity);
    return Result::success();
}

} // namespace rtapi
