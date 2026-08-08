/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtIpc.cpp
 * Date:          July 2026
 * License:       MIT
 * =========================================================================
 *
 * IPC/JSON gate (Faz 4c). Windows Named Pipe server that exposes the rtapi
 * facade to any external process via a JSON-RPC-like protocol. The pipe name
 * is fixed: \\.\pipe\RayTrophiStudio. One client at a time.
 *
 * Architecture:
 *   Worker thread → ReadFile (message mode pipe) → parse JSON
 *     → rtapi::enqueue(lambda that sets a promise) → future.get()
 *     → WriteFile JSON response
 *
 * The worker NEVER touches scene data directly; all mutations run on the main
 * thread via rtapi::enqueue(), which is drained once per frame by Main.cpp.
 */
#include "Api/RtIpc.h"
#include "Api/RtApi.h"
#include "Api/RtPython.h"   // addons.* dispatch (rtpython::listAddons/enableAddon/...)
#include "RtIpcSecurity.h"
#include "RtIpcSession.h"
#include "RtIpcAudit.h"
#include "RtIpcTransport.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <future>
#include <string>
#include <vector>

#include "json.hpp"

using json = nlohmann::json;

namespace {

constexpr std::size_t kMaxMessageBytes = rtipc_transport::kMaxMessageBytes;
constexpr int kDispatchTimeoutSec = 30;

std::atomic<bool> g_running{false};
std::atomic<bool> g_stop_requested{false};
thread_local UIContext* g_inline_dispatch_context = nullptr;

enum class DispatchState : unsigned char { Pending, Executing, Completed, Cancelled };

struct DispatchTicket {
    std::atomic<DispatchState> state{DispatchState::Pending};
    std::promise<json> promise;
};

// ---------------------------------------------------------------------------
// Helpers: enqueue an rtapi call on the main thread and wait for the result.
// ---------------------------------------------------------------------------

// Wait for an enqueued call's result. Polls in short slices instead of one long
// blocking wait so that stop() (which sets g_stop_requested and stops draining the
// main-thread queue) makes this return promptly instead of hanging for the full
// dispatch timeout. The enqueued lambda holds its own shared_ptr to the promise,
// so bailing out early here is safe — it just sets a value nobody reads.
json waitForFuture(std::future<json>& future, const std::shared_ptr<DispatchTicket>& ticket) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(kDispatchTimeoutSec);
    for (;;) {
        if (future.wait_for(std::chrono::milliseconds(50)) == std::future_status::ready) {
            return future.get();
        }
        if (g_stop_requested.load(std::memory_order_acquire)) {
            DispatchState expected = DispatchState::Pending;
            ticket->state.compare_exchange_strong(expected, DispatchState::Cancelled,
                                                   std::memory_order_acq_rel);
            return json{{"__error", "server stopping"}};
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            DispatchState expected = DispatchState::Pending;
            if (ticket->state.compare_exchange_strong(expected, DispatchState::Cancelled,
                                                       std::memory_order_acq_rel)) {
                return json{{"__error", "main thread dispatch timeout; request cancelled before execution"}};
            }
            // Once execution has begun, returning a timeout would make mutation
            // outcome ambiguous and invite unsafe retries. Wait for the real result.
        }
    }
}

// Enqueue a void-returning rtapi mutation, returning Result as JSON.
json enqueueResult(std::function<rtapi::Result(UIContext&)> fn) {
    if (g_inline_dispatch_context) {
        try {
            rtapi::Result result = fn(*g_inline_dispatch_context);
            return result.ok ? json(true) : json{{"__error", result.error}};
        } catch (const std::exception& e) {
            return json{{"__error", e.what()}};
        }
    }
    auto ticket = std::make_shared<DispatchTicket>();
    auto future = ticket->promise.get_future();

    rtapi::enqueue([fn = std::move(fn), ticket](UIContext& ctx) {
        DispatchState expected = DispatchState::Pending;
        if (!ticket->state.compare_exchange_strong(expected, DispatchState::Executing,
                                                    std::memory_order_acq_rel)) {
            if (expected == DispatchState::Cancelled)
                ticket->promise.set_value(json{{"__error", "request cancelled before execution"}});
            return;
        }
        try {
            rtapi::Result r = fn(ctx);
            if (r.ok)
                ticket->promise.set_value(json(true));
            else
                ticket->promise.set_value(json{{"__error", r.error}});
        } catch (const std::exception& e) {
            ticket->promise.set_value(json{{"__error", e.what()}});
        }
        ticket->state.store(DispatchState::Completed, std::memory_order_release);
    });

    return waitForFuture(future, ticket);
}

// Enqueue a query that returns a json value directly.
json enqueueQuery(std::function<json(UIContext&)> fn) {
    if (g_inline_dispatch_context) {
        try { return fn(*g_inline_dispatch_context); }
        catch (const std::exception& e) { return json{{"__error", e.what()}}; }
    }
    auto ticket = std::make_shared<DispatchTicket>();
    auto future = ticket->promise.get_future();

    rtapi::enqueue([fn = std::move(fn), ticket](UIContext& ctx) {
        DispatchState expected = DispatchState::Pending;
        if (!ticket->state.compare_exchange_strong(expected, DispatchState::Executing,
                                                    std::memory_order_acq_rel)) {
            if (expected == DispatchState::Cancelled)
                ticket->promise.set_value(json{{"__error", "request cancelled before execution"}});
            return;
        }
        try {
            ticket->promise.set_value(fn(ctx));
        } catch (const std::exception& e) {
            ticket->promise.set_value(json{{"__error", e.what()}});
        }
        ticket->state.store(DispatchState::Completed, std::memory_order_release);
    });

    return waitForFuture(future, ticket);
}

// ---------------------------------------------------------------------------
// JSON param helpers.
// ---------------------------------------------------------------------------
std::string requireString(const json& params, const char* key) {
    if (!params.contains(key) || !params[key].is_string())
        throw std::runtime_error(std::string("missing or invalid string param: ") + key);
    return params[key].get<std::string>();
}

int requireInt(const json& params, const char* key) {
    if (!params.contains(key) || !params[key].is_number_integer())
        throw std::runtime_error(std::string("missing or invalid int param: ") + key);
    return params[key].get<int>();
}

float requireFloat(const json& params, const char* key) {
    if (!params.contains(key) || !params[key].is_number())
        throw std::runtime_error(std::string("missing or invalid number param: ") + key);
    return params[key].get<float>();
}

bool requireBool(const json& params, const char* key) {
    if (!params.contains(key) || !params[key].is_boolean())
        throw std::runtime_error(std::string("missing or invalid boolean param: ") + key);
    return params[key].get<bool>();
}

std::string optionalString(const json& params, const char* key, const std::string& default_val = "") {
    if (!params.contains(key)) return default_val;
    if (!params[key].is_string()) throw std::runtime_error(std::string("invalid string param: ") + key);
    return params[key].get<std::string>();
}

int optionalInt(const json& params, const char* key, int default_val = 0) {
    if (!params.contains(key)) return default_val;
    if (!params[key].is_number_integer()) throw std::runtime_error(std::string("invalid int param: ") + key);
    return params[key].get<int>();
}

float optionalFloat(const json& params, const char* key, float default_val = 0.0f) {
    if (!params.contains(key)) return default_val;
    if (!params[key].is_number()) throw std::runtime_error(std::string("invalid number param: ") + key);
    return params[key].get<float>();
}

bool optionalBool(const json& params, const char* key, bool default_val = false) {
    if (!params.contains(key)) return default_val;
    if (!params[key].is_boolean()) throw std::runtime_error(std::string("invalid boolean param: ") + key);
    return params[key].get<bool>();
}

Vec3 requireVec3(const json& params, const char* key) {
    if (!params.contains(key) || !params[key].is_array() || params[key].size() != 3)
        throw std::runtime_error(std::string("missing or invalid Vec3 param: ") + key);
    auto& a = params[key];
    return Vec3(a[0].get<float>(), a[1].get<float>(), a[2].get<float>());
}

Vec3 optionalVec3(const json& params, const char* key, const Vec3& default_val = Vec3(0.0f, 0.0f, 0.0f)) {
    if (!params.contains(key)) return default_val;
    if (!params[key].is_array() || params[key].size() != 3)
        throw std::runtime_error(std::string("invalid Vec3 param: ") + key);
    return Vec3(params[key][0].get<float>(), params[key][1].get<float>(), params[key][2].get<float>());
}

Matrix4x4 requireMatrix(const json& params, const char* key) {
    if (!params.contains(key) || !params[key].is_array() || params[key].size() != 4)
        throw std::runtime_error(std::string("missing or invalid 4x4 matrix param: ") + key);
    Matrix4x4 m = Matrix4x4::identity();
    for (int r = 0; r < 4; ++r) {
        auto& row = params[key][r];
        if (!row.is_array() || row.size() != 4)
            throw std::runtime_error("each matrix row must have 4 elements");
        for (int c = 0; c < 4; ++c)
            m.m[r][c] = row[c].get<float>();
    }
    return m;
}

json matrixToJson(const Matrix4x4& m) {
    json rows = json::array();
    for (int r = 0; r < 4; ++r) {
        json row = json::array();
        for (int c = 0; c < 4; ++c) row.push_back(m.m[r][c]);
        rows.push_back(std::move(row));
    }
    return rows;
}

json vec3ToJson(const Vec3& v) {
    return json::array({v.x, v.y, v.z});
}

json materialInfoToJson(const rtapi::MaterialInfo& info) {
    return json{{"id", info.id}, {"name", info.name}, {"type", info.type}};
}

// Force field (Faz 5.6a) <-> JSON. The key names match the Python dict exactly,
// so a script ported from rt.forcefield to remote IPC keeps working verbatim.
json forceFieldToJson(const rtapi::ForceFieldInfo& info) {
    return json{
        {"id", info.id}, {"name", info.name}, {"type", info.type},
        {"shape", info.shape}, {"falloff", info.falloff},
        {"enabled", info.enabled}, {"visible", info.visible},
        {"position", vec3ToJson(info.position)}, {"rotation", vec3ToJson(info.rotation)},
        {"scale", vec3ToJson(info.scale)}, {"direction", vec3ToJson(info.direction)},
        {"axis", vec3ToJson(info.axis)},
        {"strength", info.strength}, {"falloff_radius", info.falloff_radius},
        {"inner_radius", info.inner_radius},
        {"use_noise", info.use_noise}, {"noise_octaves", info.noise_octaves},
        {"noise_seed", info.noise_seed}, {"noise_frequency", info.noise_frequency},
        {"noise_lacunarity", info.noise_lacunarity},
        {"noise_persistence", info.noise_persistence},
        {"noise_amplitude", info.noise_amplitude}, {"noise_speed", info.noise_speed},
        {"inward_force", info.inward_force}, {"upward_force", info.upward_force},
        {"linear_drag", info.linear_drag}, {"quadratic_drag", info.quadratic_drag},
        {"thermal_delta_kelvin", info.thermal_delta_kelvin},
        {"fluid_surface_drag", info.fluid_surface_drag},
        {"fluid_drag_coupling", info.fluid_drag_coupling},
        {"fluid_surface_depth", info.fluid_surface_depth},
        {"fluid_curl_detail", info.fluid_curl_detail},
        {"start_frame", info.start_frame}, {"end_frame", info.end_frame},
        {"phase", info.phase},
        {"affects_gas", info.affects_gas}, {"affects_particles", info.affects_particles},
        {"affects_cloth", info.affects_cloth},
        {"affects_rigidbody", info.affects_rigidbody},
        {"affects_fluid", info.affects_fluid}};
}

// Skeletal animation (Faz 5.6c) <-> JSON.
json animCharacterToJson(const rtapi::AnimCharacterInfo& info) {
    return json{{"name", info.name}, {"has_animation", info.has_animation},
                {"clip_count", info.clip_count}, {"bone_count", info.bone_count},
                {"uses_graph", info.uses_graph},
                {"graph_asset_key", info.graph_asset_key},
                {"graph_follows_timeline", info.graph_follows_timeline},
                {"root_motion", info.root_motion},
                {"root_motion_bone", info.root_motion_bone},
                {"visible", info.visible}};
}

json animPlaybackToJson(const rtapi::AnimPlaybackInfo& info) {
    return json{{"clip", info.clip}, {"playing", info.playing},
                {"paused", info.paused}, {"blending", info.blending},
                {"time", info.time}, {"normalized_time", info.normalized_time},
                {"layer", info.layer}};
}

// Particle emitter / solver settings (Faz 5.6b) <-> JSON. Key names match the
// Python dicts so a script ports between rt.particle and remote IPC verbatim.
json particleEmitterToJson(const rtapi::ParticleEmitterInfo& info) {
    return json{
        {"index", info.index}, {"name", info.name},
        {"source_mode", info.source_mode}, {"spawn_mode", info.spawn_mode},
        {"source_name", info.source_name}, {"enabled", info.enabled},
        {"point", vec3ToJson(info.point)},
        {"local_offset", vec3ToJson(info.local_offset)},
        {"direction", vec3ToJson(info.direction)},
        {"surface_offset", info.surface_offset},
        {"rate_per_second", info.rate_per_second}, {"burst_count", info.burst_count},
        {"speed", info.speed}, {"spread", info.spread},
        {"lifetime_seconds", info.lifetime_seconds}, {"mass", info.mass},
        {"start_size", info.start_size}, {"end_size", info.end_size},
        {"size_jitter", info.size_jitter},
        {"start_opacity", info.start_opacity}, {"end_opacity", info.end_opacity},
        {"start_color", vec3ToJson(info.start_color)},
        {"end_color", vec3ToJson(info.end_color)},
        {"angular_velocity", info.angular_velocity},
        {"angular_jitter", info.angular_jitter}, {"seed", info.seed}};
}

void applyParticleEmitterPatch(const json& patch, rtapi::ParticleEmitterInfo& info) {
    auto str = [&](const char* key, std::string& target) {
        if (patch.contains(key)) target = patch[key].get<std::string>();
    };
    auto flt = [&](const char* key, float& target) {
        if (patch.contains(key)) target = patch[key].get<float>();
    };
    auto vector = [&](const char* key, Vec3& target) {
        if (patch.contains(key)) target = requireVec3(patch, key);
    };
    str("name", info.name); str("source_mode", info.source_mode);
    str("spawn_mode", info.spawn_mode); str("source_name", info.source_name);
    if (patch.contains("enabled")) info.enabled = patch["enabled"].get<bool>();
    vector("point", info.point); vector("local_offset", info.local_offset);
    vector("direction", info.direction);
    flt("surface_offset", info.surface_offset);
    flt("rate_per_second", info.rate_per_second);
    if (patch.contains("burst_count")) info.burst_count = patch["burst_count"].get<int>();
    flt("speed", info.speed); flt("spread", info.spread);
    flt("lifetime_seconds", info.lifetime_seconds); flt("mass", info.mass);
    flt("start_size", info.start_size); flt("end_size", info.end_size);
    flt("size_jitter", info.size_jitter);
    flt("start_opacity", info.start_opacity); flt("end_opacity", info.end_opacity);
    vector("start_color", info.start_color); vector("end_color", info.end_color);
    flt("angular_velocity", info.angular_velocity);
    flt("angular_jitter", info.angular_jitter);
    if (patch.contains("seed")) info.seed = patch["seed"].get<unsigned int>();
}

json particlePhysicsToJson(const rtapi::ParticlePhysicsInfo& info) {
    return json{
        {"mode", info.mode}, {"quality", info.quality},
        {"particle_radius", info.particle_radius},
        {"self_collision_enabled", info.self_collision_enabled},
        {"solver_iterations", info.solver_iterations},
        {"max_neighbors_per_particle", info.max_neighbors_per_particle},
        {"viscosity", info.viscosity}, {"cohesion", info.cohesion},
        {"pressure_stiffness", info.pressure_stiffness},
        {"rest_density", info.rest_density}, {"buoyancy", info.buoyancy},
        {"gravity_scale", info.gravity_scale}, {"vorticity", info.vorticity},
        {"grid_density_deposit", info.grid_density_deposit},
        {"grid_temperature_deposit", info.grid_temperature_deposit},
        {"grid_fuel_deposit", info.grid_fuel_deposit},
        {"grid_deposit_fade_with_age", info.grid_deposit_fade_with_age}};
}

void applyParticlePhysicsPatch(const json& patch, rtapi::ParticlePhysicsInfo& info) {
    auto str = [&](const char* key, std::string& target) {
        if (patch.contains(key)) target = patch[key].get<std::string>();
    };
    auto flt = [&](const char* key, float& target) {
        if (patch.contains(key)) target = patch[key].get<float>();
    };
    auto integer = [&](const char* key, int& target) {
        if (patch.contains(key)) target = patch[key].get<int>();
    };
    auto boolean = [&](const char* key, bool& target) {
        if (patch.contains(key)) target = patch[key].get<bool>();
    };
    str("mode", info.mode); str("quality", info.quality);
    flt("particle_radius", info.particle_radius);
    boolean("self_collision_enabled", info.self_collision_enabled);
    integer("solver_iterations", info.solver_iterations);
    integer("max_neighbors_per_particle", info.max_neighbors_per_particle);
    flt("viscosity", info.viscosity); flt("cohesion", info.cohesion);
    flt("pressure_stiffness", info.pressure_stiffness);
    flt("rest_density", info.rest_density); flt("buoyancy", info.buoyancy);
    flt("gravity_scale", info.gravity_scale); flt("vorticity", info.vorticity);
    flt("grid_density_deposit", info.grid_density_deposit);
    flt("grid_temperature_deposit", info.grid_temperature_deposit);
    flt("grid_fuel_deposit", info.grid_fuel_deposit);
    boolean("grid_deposit_fade_with_age", info.grid_deposit_fade_with_age);
}

void applyForceFieldPatch(const json& patch, rtapi::ForceFieldInfo& info) {
    auto str = [&](const char* key, std::string& target) {
        if (patch.contains(key)) target = patch[key].get<std::string>();
    };
    auto flt = [&](const char* key, float& target) {
        if (patch.contains(key)) target = patch[key].get<float>();
    };
    auto integer = [&](const char* key, int& target) {
        if (patch.contains(key)) target = patch[key].get<int>();
    };
    auto boolean = [&](const char* key, bool& target) {
        if (patch.contains(key)) target = patch[key].get<bool>();
    };
    auto vector = [&](const char* key, Vec3& target) {
        if (patch.contains(key)) target = requireVec3(patch, key);
    };
    str("name", info.name); str("type", info.type);
    str("shape", info.shape); str("falloff", info.falloff);
    boolean("enabled", info.enabled); boolean("visible", info.visible);
    vector("position", info.position); vector("rotation", info.rotation);
    vector("scale", info.scale); vector("direction", info.direction);
    vector("axis", info.axis);
    flt("strength", info.strength); flt("falloff_radius", info.falloff_radius);
    flt("inner_radius", info.inner_radius);
    boolean("use_noise", info.use_noise);
    integer("noise_octaves", info.noise_octaves); integer("noise_seed", info.noise_seed);
    flt("noise_frequency", info.noise_frequency);
    flt("noise_lacunarity", info.noise_lacunarity);
    flt("noise_persistence", info.noise_persistence);
    flt("noise_amplitude", info.noise_amplitude);
    flt("noise_speed", info.noise_speed);
    flt("inward_force", info.inward_force); flt("upward_force", info.upward_force);
    flt("linear_drag", info.linear_drag); flt("quadratic_drag", info.quadratic_drag);
    flt("thermal_delta_kelvin", info.thermal_delta_kelvin);
    boolean("fluid_surface_drag", info.fluid_surface_drag);
    flt("fluid_drag_coupling", info.fluid_drag_coupling);
    flt("fluid_surface_depth", info.fluid_surface_depth);
    flt("fluid_curl_detail", info.fluid_curl_detail);
    flt("start_frame", info.start_frame); flt("end_frame", info.end_frame);
    flt("phase", info.phase);
    boolean("affects_gas", info.affects_gas);
    boolean("affects_particles", info.affects_particles);
    boolean("affects_cloth", info.affects_cloth);
    boolean("affects_rigidbody", info.affects_rigidbody);
    boolean("affects_fluid", info.affects_fluid);
}

// Node parameter (Faz 5.1b) <-> JSON. Scalars map to JSON number/bool/string,
// vectors to a JSON array, an unset default to null.
json nodeParamToJson(const rtapi::NodeParamValue& v) {
    using K = rtapi::NodeParamValue::Kind;
    switch (v.kind) {
        case K::Float:   return json(v.floats[0]);
        case K::Int:     return json(v.int_value);
        case K::Bool:    return json(v.bool_value);
        case K::Vector2: return json::array({v.floats[0], v.floats[1]});
        case K::Vector3: return json::array({v.floats[0], v.floats[1], v.floats[2]});
        case K::Vector4: return json::array({v.floats[0], v.floats[1], v.floats[2], v.floats[3]});
        case K::String:  return json(v.string_value);
        case K::None:    default: return json(nullptr);
    }
}

rtapi::NodeParamValue nodeParamFromJson(const json& value) {
    using K = rtapi::NodeParamValue::Kind;
    rtapi::NodeParamValue out;
    if (value.is_boolean()) {
        out.kind = K::Bool; out.bool_value = value.get<bool>();
    } else if (value.is_number_integer() || value.is_number_unsigned()) {
        out.kind = K::Int; out.int_value = value.get<int>(); out.floats[0] = static_cast<float>(out.int_value);
    } else if (value.is_number_float()) {
        out.kind = K::Float; out.floats[0] = value.get<float>();
    } else if (value.is_string()) {
        out.kind = K::String; out.string_value = value.get<std::string>();
    } else if (value.is_array()) {
        if (value.size() < 2 || value.size() > 4)
            throw std::runtime_error("vector parameter must have 2, 3 or 4 components");
        out.kind = (value.size() == 2) ? K::Vector2 : (value.size() == 3) ? K::Vector3 : K::Vector4;
        for (size_t i = 0; i < value.size(); ++i) out.floats[i] = value[i].get<float>();
    } else {
        throw std::runtime_error("unsupported node parameter value (expected number/bool/string/array)");
    }
    return out;
}

// ---------------------------------------------------------------------------
// Method dispatch. Every rtapi function that is not mesh-binary gets a handler.
// ---------------------------------------------------------------------------
json dispatchMethod(const std::string& method, const json& params) {
    auto auditEventJson = [](const rtipc_audit::Event& event) {
        return json{{"sequence", event.sequence}, {"timestamp", event.timestamp},
                    {"connection_id", event.connection_id}, {"token_id", event.token_id},
                    {"peer_address", event.peer_address}, {"method", event.method},
                    {"allowed", event.allowed}, {"outcome", event.outcome},
                    {"duration_us", event.duration_us},
                    {"bytes_received", event.bytes_received}, {"bytes_sent", event.bytes_sent}};
    };
    if (method == "ipc.admin.audit.list") {
        json result = json::array();
        for (const auto& event : rtipc_audit::recent(
                 static_cast<size_t>((std::max)(1, (std::min)(params.value("maximum", 256), 2048)))))
            result.push_back(auditEventJson(event));
        return result;
    }
    if (method == "ipc.admin.audit.clear") {
        rtipc_audit::clear(); return json(true);
    }
    if (method == "ipc.admin.audit.export") {
        std::string error;
        return rtipc_audit::exportJsonl(requireString(params, "filepath"), error)
            ? json(true) : json{{"__error", error}};
    }
    auto sessionInfoJson = [](const rtipc_session::SessionInfo& session) {
        return json{{"connection_id", session.connection_id}, {"transport", session.transport},
                    {"peer_address", session.peer_address}, {"peer_port", session.peer_port},
                    {"tls_version", session.tls_version}, {"tls_cipher", session.tls_cipher},
                    {"token_id", session.token_id}, {"connected_at", session.connected_at},
                    {"last_activity_at", session.last_activity_at},
                    {"request_count", session.request_count}, {"error_count", session.error_count},
                    {"bytes_received", session.bytes_received}, {"bytes_sent", session.bytes_sent},
                    {"active", session.active},
                    {"disconnect_requested", session.disconnect_requested}};
    };
    if (method == "ipc.admin.sessions.list") {
        json result = json::array();
        for (const auto& session : rtipc_session::listSessions(params.value("include_closed", false)))
            result.push_back(sessionInfoJson(session));
        return result;
    }
    if (method == "ipc.admin.sessions.get") {
        rtipc_session::SessionInfo session;
        if (!rtipc_session::getSession(requireString(params, "connection_id"), session))
            return json{{"__error", "session not found"}};
        return sessionInfoJson(session);
    }
    if (method == "ipc.admin.sessions.disconnect")
        return rtipc::disconnectSession(requireString(params, "connection_id"))
            ? json(true) : json{{"__error", "active session not found"}};
    if (method == "ipc.admin.sessions.disconnect_all")
        return json(rtipc::disconnectAllSessions());
    auto tokenInfoJson = [](const rtipc::TokenInfo& token) {
        return json{{"id", token.id}, {"display_name", token.display_name},
                    {"capabilities", token.capabilities}, {"created_at", token.created_at},
                    {"expires_at", token.expires_at}, {"last_used_at", token.last_used_at},
                    {"revoked", token.revoked}, {"allowed_cidrs", token.allowed_cidrs}};
    };
    if (method == "ipc.admin.tokens.list") {
        return enqueueQuery([tokenInfoJson](UIContext&) {
            json result = json::array();
            for (const auto& token : rtipc::listTokens()) result.push_back(tokenInfoJson(token));
            return result;
        });
    }
    if (method == "ipc.admin.tokens.create") {
        const std::string name = requireString(params, "display_name");
        const int capabilities = requireInt(params, "capabilities");
        const int64_t expires = params.value("expires_at", int64_t{0});
        const auto allowed_cidrs = params.value("allowed_cidrs", std::vector<std::string>{});
        return enqueueQuery([name, capabilities, expires, allowed_cidrs, tokenInfoJson](UIContext&) {
            rtipc::TokenInfo token; std::string raw, error;
            if (!rtipc::createToken(name, static_cast<uint32_t>(capabilities), allowed_cidrs, expires,
                                    token, raw, error)) return json{{"__error", error}};
            return json{{"token", tokenInfoJson(token)}, {"raw_token", raw}};
        });
    }
    if (method == "ipc.admin.tokens.revoke") {
        const std::string id = requireString(params, "token_id");
        return enqueueQuery([id](UIContext&) {
            std::string error;
            return rtipc::revokeToken(id, error) ? json(true) : json{{"__error", error}};
        });
    }
    if (method == "ipc.admin.tokens.update") {
        const std::string id = requireString(params, "token_id");
        const int capabilities = requireInt(params, "capabilities");
        const int64_t expires = params.value("expires_at", int64_t{0});
        const auto cidrs = params.value("allowed_cidrs", std::vector<std::string>{});
        std::string error;
        return rtipc::updateToken(id, static_cast<uint32_t>(capabilities), cidrs, expires, error)
            ? json(true) : json{{"__error", error}};
    }
    if (method == "ipc.admin.tokens.rotate") {
        const std::string id = requireString(params, "token_id");
        return enqueueQuery([id, tokenInfoJson](UIContext&) {
            rtipc::TokenInfo token; std::string raw, error;
            if (!rtipc::rotateToken(id, token, raw, error)) return json{{"__error", error}};
            return json{{"token", tokenInfoJson(token)}, {"raw_token", raw}};
        });
    }
    if (method == "batch") {
        if (!params.contains("calls") || !params["calls"].is_array())
            throw std::runtime_error("batch requires a calls array");
        const json calls = params["calls"];
        if (calls.empty() || calls.size() > 64)
            throw std::runtime_error("batch must contain between 1 and 64 calls");
        return enqueueQuery([calls](UIContext& ctx) {
            struct InlineScope {
                explicit InlineScope(UIContext& value) { g_inline_dispatch_context = &value; }
                ~InlineScope() { g_inline_dispatch_context = nullptr; }
            } scope(ctx);
            json output = json::array();
            for (const auto& call : calls) {
                if (!call.is_object() || !call.contains("method") || !call["method"].is_string()) {
                    output.push_back(json{{"error", "invalid batch call"}}); continue;
                }
                const std::string child_method = call["method"].get<std::string>();
                if (child_method == "batch") {
                    output.push_back(json{{"error", "nested batch is not allowed"}}); continue;
                }
                json child_params = call.value("params", json::object());
                if (!child_params.is_object()) {
                    output.push_back(json{{"error", "batch call params must be an object"}}); continue;
                }
                try {
                    json result = dispatchMethod(child_method, child_params);
                    if (result.is_object() && result.contains("__error"))
                        output.push_back(json{{"error", result["__error"]}});
                    else output.push_back(json{{"result", result}});
                } catch (const std::exception& e) {
                    output.push_back(json{{"error", e.what()}});
                }
            }
            return output;
        });
    }
    // ── Version ─────────────────────────────────────────────────────────
    if (method == "version") {
        return enqueueQuery([](UIContext&) {
            rtapi::Version v = rtapi::version();
            return json(std::to_string(v.major) + "." +
                        std::to_string(v.minor) + "." +
                        std::to_string(v.patch));
        });
    }

    // ── Scene queries ───────────────────────────────────────────────────
    if (method == "scene.list_objects") {
        return enqueueQuery([](UIContext&) {
            return json(rtapi::listObjects());
        });
    }
    if (method == "scene.object_exists") {
        std::string name = requireString(params, "name");
        return enqueueQuery([name](UIContext&) {
            return json(rtapi::objectExists(name));
        });
    }
    if (method == "scene.object_info") {
        std::string name = requireString(params, "name");
        return enqueueQuery([name](UIContext&) {
            rtapi::ObjectInfo info;
            rtapi::Result r = rtapi::getObjectInfo(name, info);
            if (!r.ok) return json{{"__error", r.error}};
            return json{{"name", info.name},
                        {"triangles", info.triangle_count},
                        {"vertices", info.vertex_count}};
        });
    }

    // ── Transform ───────────────────────────────────────────────────────
    if (method == "scene.get_transform") {
        std::string name = requireString(params, "name");
        return enqueueQuery([name](UIContext&) {
            Matrix4x4 m;
            rtapi::Result r = rtapi::getObjectTransform(name, m);
            if (!r.ok) return json{{"__error", r.error}};
            return matrixToJson(m);
        });
    }
    if (method == "scene.set_transform") {
        std::string name = requireString(params, "name");
        Matrix4x4 matrix = requireMatrix(params, "matrix");
        return enqueueResult([name, matrix](UIContext&) {
            return rtapi::setObjectTransform(name, matrix);
        });
    }

    // ── Object lifecycle ────────────────────────────────────────────────
    if (method == "scene.delete") {
        std::string name = requireString(params, "name");
        return enqueueResult([name](UIContext&) {
            return rtapi::deleteObject(name);
        });
    }
    if (method == "scene.duplicate") {
        std::string name = requireString(params, "name");
        return enqueueQuery([name](UIContext&) {
            std::string newName;
            rtapi::Result r = rtapi::duplicateObject(name, newName);
            if (!r.ok) return json{{"__error", r.error}};
            return json(newName);
        });
    }
    if (method == "scene.import_model") {
        std::string path = requireString(params, "path");
        return enqueueResult([path](UIContext&) {
            return rtapi::importModel(path);
        });
    }
    if (method == "scene.add_primitive") {
        std::string type = requireString(params, "type");
        std::string name = params.value("name", "");
        float size = params.value("size", 1.0f);
        return enqueueQuery([type, name, size](UIContext&) {
            std::string newName;
            rtapi::Result r = rtapi::addPrimitive(type, name, size, newName);
            if (!r.ok) return json{{"__error", r.error}};
            return json(newName);
        });
    }

    // ── Material ────────────────────────────────────────────────────────
    if (method == "material.get") {
        std::string obj = requireString(params, "object_name");
        std::string param = requireString(params, "param");
        return enqueueQuery([obj, param](UIContext&) {
            rtapi::MaterialParamValue val;
            rtapi::Result r = rtapi::getMaterialParam(obj, param, val);
            if (!r.ok) return json{{"__error", r.error}};
            if (val.is_color) return vec3ToJson(val.color);
            return json(val.scalar);
        });
    }
    if (method == "material.set") {
        std::string obj = requireString(params, "object_name");
        std::string param = requireString(params, "param");
        // value can be scalar or [r,g,b]
        if (!params.contains("value"))
            throw std::runtime_error("missing param: value");
        if (params["value"].is_number()) {
            float value = params["value"].get<float>();
            return enqueueResult([obj, param, value](UIContext&) {
                return rtapi::setMaterialParam(obj, param, value);
            });
        } else if (params["value"].is_array() && params["value"].size() == 3) {
            Vec3 color(params["value"][0].get<float>(),
                       params["value"][1].get<float>(),
                       params["value"][2].get<float>());
            return enqueueResult([obj, param, color](UIContext&) {
                return rtapi::setMaterialParam(obj, param, color);
            });
        } else {
            throw std::runtime_error("value must be a number or [r,g,b] array");
        }
    }

    // ── Material assets (Faz 5.5a) ──────────────────────────────────────
    if (method == "material.list") {
        return enqueueQuery([](UIContext&) {
            json result = json::array();
            for (const rtapi::MaterialInfo& info : rtapi::listMaterials())
                result.push_back(materialInfoToJson(info));
            return result;
        });
    }
    if (method == "material.info") {
        std::string name = requireString(params, "name");
        return enqueueQuery([name](UIContext&) {
            rtapi::MaterialInfo info;
            rtapi::Result r = rtapi::getMaterial(name, info);
            if (!r.ok) return json{{"__error", r.error}};
            return materialInfoToJson(info);
        });
    }
    if (method == "material.create") {
        std::string type = requireString(params, "type");
        std::string name = params.value("name", "");
        return enqueueQuery([type, name](UIContext&) {
            std::string created;
            rtapi::Result r = rtapi::createMaterial(type, name, created);
            if (!r.ok) return json{{"__error", r.error}};
            return json(created);
        });
    }
    if (method == "material.of_object") {
        std::string obj = requireString(params, "object_name");
        return enqueueQuery([obj](UIContext&) {
            return json(rtapi::objectMaterials(obj));
        });
    }
    if (method == "material.assign") {
        std::string obj = requireString(params, "object_name");
        std::string mat = requireString(params, "material_name");
        return enqueueResult([obj, mat](UIContext&) {
            return rtapi::assignMaterial(obj, mat);
        });
    }
    if (method == "material.set_texture") {
        std::string mat = requireString(params, "material_name");
        std::string slot = requireString(params, "slot");
        std::string path = requireString(params, "path");
        return enqueueResult([mat, slot, path](UIContext&) {
            return rtapi::setMaterialTexture(mat, slot, path);
        });
    }
    if (method == "material.clear_texture") {
        std::string mat = requireString(params, "material_name");
        std::string slot = requireString(params, "slot");
        return enqueueResult([mat, slot](UIContext&) {
            return rtapi::clearMaterialTexture(mat, slot);
        });
    }
    if (method == "material.textures") {
        std::string mat = requireString(params, "material_name");
        return enqueueQuery([mat](UIContext&) {
            return json(rtapi::materialTextureSlots(mat));
        });
    }

    // ── Selection (Faz 5.5a) ────────────────────────────────────────────
    if (method == "select.list") {
        return enqueueQuery([](UIContext&) {
            json result = json::array();
            for (const rtapi::SelectionItem& item : rtapi::listSelection()) {
                result.push_back(json{{"type", item.type}, {"name", item.name},
                                      {"index", item.index}, {"primary", item.primary}});
            }
            return result;
        });
    }
    if (method == "select.object") {
        std::string name = requireString(params, "name");
        bool additive = params.value("additive", false);
        return enqueueResult([name, additive](UIContext&) {
            return rtapi::selectObject(name, additive);
        });
    }
    if (method == "select.deselect_object") {
        std::string name = requireString(params, "name");
        return enqueueResult([name](UIContext&) {
            return rtapi::deselectObject(name);
        });
    }
    if (method == "select.light") {
        int index = requireInt(params, "index");
        bool additive = params.value("additive", false);
        return enqueueResult([index, additive](UIContext&) {
            return rtapi::selectLight(index, additive);
        });
    }
    if (method == "select.all_objects") {
        return enqueueQuery([](UIContext&) {
            int count = 0;
            rtapi::Result r = rtapi::selectAllObjects(count);
            if (!r.ok) return json{{"__error", r.error}};
            return json(count);
        });
    }
    if (method == "select.clear") {
        return enqueueResult([](UIContext&) { return rtapi::clearSelection(); });
    }

    // ── Lights ──────────────────────────────────────────────────────────
    if (method == "lights.list") {
        return enqueueQuery([](UIContext&) {
            json result = json::array();
            for (const rtapi::LightInfo& info : rtapi::listLights()) {
                result.push_back(json{
                    {"index", info.index}, {"name", info.name},
                    {"type", info.type}, {"position", vec3ToJson(info.position)}
                });
            }
            return result;
        });
    }
    if (method == "lights.add") {
        std::string type = requireString(params, "type");
        Vec3 pos = requireVec3(params, "position");
        return enqueueQuery([type, pos](UIContext&) {
            std::string name;
            rtapi::Result r = rtapi::addLight(type, pos, name);
            if (!r.ok) return json{{"__error", r.error}};
            return json(name);
        });
    }
    if (method == "lights.delete") {
        int index = requireInt(params, "index");
        return enqueueResult([index](UIContext&) {
            return rtapi::deleteLight(index);
        });
    }
    if (method == "lights.set_position") {
        int index = requireInt(params, "index");
        Vec3 pos = requireVec3(params, "position");
        return enqueueResult([index, pos](UIContext&) {
            return rtapi::setLightPosition(index, pos);
        });
    }
    // Appearance / shape parameters (Faz 5.5a).
    if (method == "lights.get") {
        int index = requireInt(params, "index");
        return enqueueQuery([index](UIContext&) {
            rtapi::LightInfo info;
            rtapi::Result r = rtapi::getLight(index, info);
            if (!r.ok) return json{{"__error", r.error}};
            return json{{"index", info.index}, {"name", info.name}, {"type", info.type},
                        {"position", vec3ToJson(info.position)},
                        {"direction", vec3ToJson(info.direction)},
                        {"color", vec3ToJson(info.color)},
                        {"intensity", info.intensity}, {"radius", info.radius},
                        {"spot_angle", info.spot_angle}, {"spot_falloff", info.spot_falloff},
                        {"width", info.width}, {"height", info.height},
                        {"visible", info.visible}};
        });
    }
    if (method == "lights.set_direction") {
        int index = requireInt(params, "index");
        Vec3 dir = requireVec3(params, "direction");
        return enqueueResult([index, dir](UIContext&) {
            return rtapi::setLightDirection(index, dir);
        });
    }
    if (method == "lights.set_color") {
        int index = requireInt(params, "index");
        Vec3 color = requireVec3(params, "color");
        return enqueueResult([index, color](UIContext&) {
            return rtapi::setLightColor(index, color);
        });
    }
    if (method == "lights.set_intensity") {
        int index = requireInt(params, "index");
        float value = requireFloat(params, "intensity");
        return enqueueResult([index, value](UIContext&) {
            return rtapi::setLightIntensity(index, value);
        });
    }
    if (method == "lights.set_visible") {
        int index = requireInt(params, "index");
        bool visible = requireBool(params, "visible");
        return enqueueResult([index, visible](UIContext&) {
            return rtapi::setLightVisible(index, visible);
        });
    }
    if (method == "lights.set_param") {
        int index = requireInt(params, "index");
        std::string param = requireString(params, "param");
        float value = requireFloat(params, "value");
        return enqueueResult([index, param, value](UIContext&) {
            return rtapi::setLightParam(index, param, value);
        });
    }
    if (method == "lights.rename") {
        int index = requireInt(params, "index");
        std::string name = requireString(params, "name");
        return enqueueResult([index, name](UIContext&) {
            return rtapi::renameLight(index, name);
        });
    }

    // ── Camera (Faz 5.1a) ───────────────────────────────────────────────
    if (method == "camera.get") {
        return enqueueQuery([](UIContext&) {
            rtapi::CameraState s;
            rtapi::Result r = rtapi::getCamera(s);
            if (!r.ok) return json{{"__error", r.error}};
            return json{{"position", vec3ToJson(s.position)}, {"target", vec3ToJson(s.target)},
                        {"up", vec3ToJson(s.up)}, {"fov", s.fov},
                        {"focus_distance", s.focus_distance}, {"aperture", s.aperture}};
        });
    }
    if (method == "camera.set_position") {
        Vec3 p = requireVec3(params, "position");
        return enqueueResult([p](UIContext&) { return rtapi::setCameraPosition(p); });
    }
    if (method == "camera.set_target") {
        Vec3 t = requireVec3(params, "target");
        return enqueueResult([t](UIContext&) { return rtapi::setCameraTarget(t); });
    }
    if (method == "camera.set_fov") {
        float f = requireFloat(params, "fov");
        return enqueueResult([f](UIContext&) { return rtapi::setCameraFov(f); });
    }
    if (method == "camera.set_focus_distance") {
        float f = requireFloat(params, "focus_distance");
        return enqueueResult([f](UIContext&) { return rtapi::setCameraFocusDistance(f); });
    }
    if (method == "camera.set_aperture") {
        float f = requireFloat(params, "aperture");
        return enqueueResult([f](UIContext&) { return rtapi::setCameraAperture(f); });
    }

    // ── World / environment (Faz 5.1c) ──────────────────────────────────
    if (method == "world.get") {
        return enqueueQuery([](UIContext&) {
            rtapi::WorldState s;
            rtapi::Result r = rtapi::getWorld(s);
            if (!r.ok) return json{{"__error", r.error}};
            return json{{"mode", s.mode},
                        {"background_color", vec3ToJson(s.background_color)},
                        {"sun_elevation", s.sun_elevation}, {"sun_azimuth", s.sun_azimuth},
                        {"sun_intensity", s.sun_intensity},
                        {"atmosphere_intensity", s.atmosphere_intensity},
                        {"sun_size", s.sun_size}};
        });
    }
    if (method == "world.set_mode") {
        std::string m = requireString(params, "mode");
        return enqueueResult([m](UIContext&) { return rtapi::setWorldMode(m); });
    }
    if (method == "world.set_background_color") {
        Vec3 c = requireVec3(params, "background_color");
        return enqueueResult([c](UIContext&) { return rtapi::setWorldBackgroundColor(c); });
    }
    if (method == "world.set_sun_elevation") {
        float d = requireFloat(params, "sun_elevation");
        return enqueueResult([d](UIContext&) { return rtapi::setWorldSunElevation(d); });
    }
    if (method == "world.set_sun_azimuth") {
        float d = requireFloat(params, "sun_azimuth");
        return enqueueResult([d](UIContext&) { return rtapi::setWorldSunAzimuth(d); });
    }
    if (method == "world.set_sun_intensity") {
        float v = requireFloat(params, "sun_intensity");
        return enqueueResult([v](UIContext&) { return rtapi::setWorldSunIntensity(v); });
    }
    if (method == "world.set_atmosphere_intensity") {
        float v = requireFloat(params, "atmosphere_intensity");
        return enqueueResult([v](UIContext&) { return rtapi::setWorldAtmosphereIntensity(v); });
    }
    if (method == "world.set_sun_size") {
        float d = requireFloat(params, "sun_size");
        return enqueueResult([d](UIContext&) { return rtapi::setWorldSunSize(d); });
    }

    // ── Post-processing (Faz 5.1d) ──────────────────────────────────────
    if (method == "post.get") {
        return enqueueQuery([](UIContext&) {
            rtapi::PostState s;
            rtapi::Result r = rtapi::getPost(s);
            if (!r.ok) return json{{"__error", r.error}};
            return json{{"exposure", s.exposure}, {"gamma", s.gamma},
                        {"saturation", s.saturation}, {"color_temperature", s.color_temperature},
                        {"tone_mapping", s.tone_mapping}, {"vignette_enabled", s.vignette_enabled},
                        {"vignette_strength", s.vignette_strength}, {"stylize_enabled", s.stylize_enabled},
                        {"stylize_strength", s.stylize_strength}};
        });
    }
    if (method == "post.set_exposure") {
        float v = requireFloat(params, "exposure");
        return enqueueResult([v](UIContext&) { return rtapi::setPostExposure(v); });
    }
    if (method == "post.set_gamma") {
        float v = requireFloat(params, "gamma");
        return enqueueResult([v](UIContext&) { return rtapi::setPostGamma(v); });
    }
    if (method == "post.set_saturation") {
        float v = requireFloat(params, "saturation");
        return enqueueResult([v](UIContext&) { return rtapi::setPostSaturation(v); });
    }
    if (method == "post.set_color_temperature") {
        float v = requireFloat(params, "color_temperature");
        return enqueueResult([v](UIContext&) { return rtapi::setPostColorTemperature(v); });
    }
    if (method == "post.set_tone_mapping") {
        std::string s = requireString(params, "tone_mapping");
        return enqueueResult([s](UIContext&) { return rtapi::setPostToneMapping(s); });
    }
    if (method == "post.set_vignette_enabled") {
        bool v = requireBool(params, "vignette_enabled");
        return enqueueResult([v](UIContext&) { return rtapi::setPostVignetteEnabled(v); });
    }
    if (method == "post.set_vignette_strength") {
        float v = requireFloat(params, "vignette_strength");
        return enqueueResult([v](UIContext&) { return rtapi::setPostVignetteStrength(v); });
    }
    if (method == "post.set_stylize_enabled") {
        bool v = requireBool(params, "stylize_enabled");
        return enqueueResult([v](UIContext&) { return rtapi::setPostStylizeEnabled(v); });
    }
    if (method == "post.set_stylize_strength") {
        float v = requireFloat(params, "stylize_strength");
        return enqueueResult([v](UIContext&) { return rtapi::setPostStylizeStrength(v); });
    }

    // ── Mesh Modifiers (Faz 5.2b) ───────────────────────────────────────
    if (method == "modifiers.get_stack") {
        std::string name = requireString(params, "object");
        return enqueueQuery([name](UIContext&) {
            std::vector<rtapi::ModifierInfo> stack;
            rtapi::Result r = rtapi::getModifierStack(name, stack);
            if (!r.ok) return json{{"__error", r.error}};
            json arr = json::array();
            for (const auto& mod : stack) {
                arr.push_back({{"index", mod.index}, {"name", mod.name}, {"type", mod.type},
                               {"enabled", mod.enabled}, {"levels", mod.levels},
                               {"render_levels", mod.render_levels}, {"smooth_angle", mod.smooth_angle}});
            }
            return arr;
        });
    }
    if (method == "modifiers.add") {
        std::string obj = requireString(params, "object");
        std::string type = optionalString(params, "type", "catmull_clark");
        std::string name = optionalString(params, "name", "");
        int levels = optionalInt(params, "levels", 1);
        int rlevels = optionalInt(params, "render_levels", 2);
        return enqueueQuery([obj, type, name, levels, rlevels](UIContext&) {
            rtapi::ModifierInfo mod;
            rtapi::Result r = rtapi::addModifier(obj, type, name, levels, rlevels, mod);
            if (!r.ok) return json{{"__error", r.error}};
            return json{{"index", mod.index}, {"name", mod.name}, {"type", mod.type},
                        {"enabled", mod.enabled}, {"levels", mod.levels},
                        {"render_levels", mod.render_levels}, {"smooth_angle", mod.smooth_angle}};
        });
    }
    if (method == "modifiers.remove") {
        std::string obj = requireString(params, "object");
        int index = optionalInt(params, "index", 0);
        return enqueueResult([obj, index](UIContext&) { return rtapi::removeModifier(obj, index); });
    }
    if (method == "modifiers.set_param") {
        std::string obj = requireString(params, "object");
        int index = optionalInt(params, "index", 0);

        std::string name_str = optionalString(params, "name", "");
        const std::string* p_name = params.contains("name") ? &name_str : nullptr;

        bool enabled_val = false;
        const bool* p_enabled = nullptr;
        if (params.contains("enabled")) { enabled_val = params["enabled"].get<bool>(); p_enabled = &enabled_val; }

        int levels_val = 0;
        const int* p_levels = nullptr;
        if (params.contains("levels")) { levels_val = params["levels"].get<int>(); p_levels = &levels_val; }

        int rlevels_val = 0;
        const int* p_rlevels = nullptr;
        if (params.contains("render_levels")) { rlevels_val = params["render_levels"].get<int>(); p_rlevels = &rlevels_val; }

        float smooth_val = 0.0f;
        const float* p_smooth = nullptr;
        if (params.contains("smooth_angle")) { smooth_val = params["smooth_angle"].get<float>(); p_smooth = &smooth_val; }

        return enqueueResult([obj, index, name_str, p_name, enabled_val, p_enabled, levels_val, p_levels, rlevels_val, p_rlevels, smooth_val, p_smooth](UIContext&) {
            return rtapi::updateModifier(obj, index,
                                          p_name ? &name_str : nullptr,
                                          p_enabled ? &enabled_val : nullptr,
                                          p_levels ? &levels_val : nullptr,
                                          p_rlevels ? &rlevels_val : nullptr,
                                          p_smooth ? &smooth_val : nullptr);
        });
    }
    if (method == "modifiers.apply") {
        std::string obj = requireString(params, "object");
        int index = optionalInt(params, "index", 0);
        return enqueueResult([obj, index](UIContext&) { return rtapi::applyModifier(obj, index); });
    }

    // ── Scatter & Foliage System (Faz 5.2c) ─────────────────────────────
    if (method == "scatter.list_groups") {
        return enqueueQuery([](UIContext&) {
            std::vector<rtapi::ScatterGroupInfo> groups;
            rtapi::Result r = rtapi::listScatterGroups(groups);
            if (!r.ok) return json{{"__error", r.error}};
            json arr = json::array();
            for (const auto& g : groups) {
                json sources = json::array();
                for (const auto& s : g.sources) {
                    sources.push_back({{"name", s.name}, {"weight", s.weight},
                                       {"scale_min", s.scale_min}, {"scale_max", s.scale_max},
                                       {"rotation_y", s.rotation_random_y}, {"align_to_normal", s.align_to_normal}});
                }
                arr.push_back({{"id", g.id}, {"name", g.name}, {"target_type", g.target_type},
                               {"target_node_name", g.target_node_name}, {"instance_count", g.instance_count},
                               {"triangle_count", g.triangle_count}, {"sources", sources}});
            }
            return arr;
        });
    }
    if (method == "scatter.create_group") {
        std::string name = requireString(params, "name");
        std::string target_node = optionalString(params, "target_node", "");
        std::string target_type = optionalString(params, "target_type", "mesh");
        return enqueueQuery([name, target_node, target_type](UIContext&) {
            rtapi::ScatterGroupInfo info;
            rtapi::Result r = rtapi::createScatterGroup(name, target_node, target_type, info);
            if (!r.ok) return json{{"__error", r.error}};
            return json{{"id", info.id}, {"name", info.name}, {"target_type", info.target_type},
                        {"target_node_name", info.target_node_name}, {"instance_count", info.instance_count},
                        {"triangle_count", info.triangle_count}};
        });
    }
    if (method == "scatter.delete_group") {
        std::string group = requireString(params, "group");
        return enqueueResult([group](UIContext&) { return rtapi::deleteScatterGroup(group); });
    }
    if (method == "scatter.clear") {
        std::string group = requireString(params, "group");
        return enqueueResult([group](UIContext&) { return rtapi::clearScatterGroup(group); });
    }
    if (method == "scatter.add_source") {
        std::string group = requireString(params, "group");
        std::string mesh_name = requireString(params, "mesh");
        float weight = optionalFloat(params, "weight", 1.0f);
        float scale_min = optionalFloat(params, "scale_min", 0.8f);
        float scale_max = optionalFloat(params, "scale_max", 1.2f);
        float rot_y = optionalFloat(params, "rotation_y", 360.0f);
        bool align = optionalBool(params, "align_to_normal", true);
        return enqueueResult([group, mesh_name, weight, scale_min, scale_max, rot_y, align](UIContext&) {
            return rtapi::addScatterSource(group, mesh_name, weight, scale_min, scale_max, rot_y, align);
        });
    }
    if (method == "scatter.fill") {
        std::string group = requireString(params, "group");
        return enqueueQuery([group](UIContext&) {
            int spawned = 0;
            rtapi::Result r = rtapi::fillScatterGroup(group, spawned);
            if (!r.ok) return json{{"__error", r.error}};
            return json{{"spawned", spawned}};
        });
    }

    // ── Physics Engine (Faz 5.3a) ────────────────────────────────────────
    if (method == "physics.get_body") {
        std::string obj = requireString(params, "object");
        return enqueueQuery([obj](UIContext&) {
            rtapi::PhysicsBodyInfo info;
            rtapi::Result r = rtapi::getPhysicsBody(obj, info);
            if (!r.ok) return json{{"__error", r.error}};
            return json{{"object_name", info.object_name}, {"kind", info.kind},
                        {"motion_type", info.motion_type}, {"shape", info.shape},
                        {"enabled", info.enabled}, {"mass", info.mass},
                        {"friction", info.friction}, {"restitution", info.restitution},
                        {"linear_damping", info.linear_damping}, {"angular_damping", info.angular_damping},
                        {"gravity_scale", info.gravity_scale}};
        });
    }
    if (method == "physics.add_body") {
        std::string obj = requireString(params, "object");
        std::string kind = optionalString(params, "kind", "rigid");
        std::string motion_type = optionalString(params, "motion_type", "dynamic");
        std::string shape = optionalString(params, "shape", "box");
        float mass = optionalFloat(params, "mass", 1.0f);
        return enqueueQuery([obj, kind, motion_type, shape, mass](UIContext&) {
            rtapi::PhysicsBodyInfo info;
            rtapi::Result r = rtapi::addPhysicsBody(obj, kind, motion_type, shape, mass, info);
            if (!r.ok) return json{{"__error", r.error}};
            return json{{"object_name", info.object_name}, {"kind", info.kind},
                        {"motion_type", info.motion_type}, {"shape", info.shape},
                        {"enabled", info.enabled}, {"mass", info.mass}};
        });
    }
    if (method == "physics.remove_body") {
        std::string obj = requireString(params, "object");
        return enqueueResult([obj](UIContext&) { return rtapi::removePhysicsBody(obj); });
    }
    if (method == "physics.reset") {
        return enqueueResult([](UIContext&) { return rtapi::resetPhysicsSimulation(); });
    }
    if (method == "physics.step") {
        float dt = optionalFloat(params, "dt", 0.0166667f);
        return enqueueResult([dt](UIContext&) { return rtapi::stepPhysicsSimulation(dt); });
    }
    if (method == "physics.set_gravity") {
        Vec3 g = requireVec3(params, "gravity");
        return enqueueResult([g](UIContext&) { return rtapi::setPhysicsGravity(g); });
    }

    // ── Fluid Simulation Engine (Faz 5.3b) ──────────────────────────────
    if (method == "fluid.create_domain" || method == "gas.create_domain") {
        std::string default_type = (method == "gas.create_domain") ? "gas" : "fluid";
        std::string name = optionalString(params, "name", default_type == "gas" ? "GasDomain" : "Fluid");
        std::string type = optionalString(params, "type", default_type);
        Vec3 dmin = optionalVec3(params, "domain_min", Vec3(-1.0f, 0.0f, -1.0f));
        Vec3 dmax = optionalVec3(params, "domain_max", Vec3(1.0f, 2.0f, 1.0f));
        float vs = optionalFloat(params, "voxel_size", 0.05f);
        return enqueueQuery([name, dmin, dmax, vs, type](UIContext&) {
            rtapi::FluidDomainInfo info;
            rtapi::Result r = rtapi::createFluidDomain(name, dmin, dmax, vs, type, info);
            if (!r.ok) return json{{"__error", r.error}};
            return json{{"id", info.id}, {"name", info.name}, {"type", info.type}, {"voxel_size", info.voxel_size},
                        {"particle_count", info.particle_count}, {"render_mode", info.render_mode}};
        });
    }
    if (method == "fluid.get" || method == "gas.get") {
        std::string domain = requireString(params, "domain");
        return enqueueQuery([domain](UIContext&) {
            rtapi::FluidDomainInfo info;
            rtapi::Result r = rtapi::getFluidDomain(domain, info);
            if (!r.ok) return json{{"__error", r.error}};
            return json{{"id", info.id}, {"name", info.name}, {"type", info.type},
                        {"domain_min", json::array({info.domain_min.x, info.domain_min.y, info.domain_min.z})},
                        {"domain_max", json::array({info.domain_max.x, info.domain_max.y, info.domain_max.z})},
                        {"voxel_size", info.voxel_size}, {"particle_count", info.particle_count},
                        {"render_mode", info.render_mode}, {"backend", info.backend},
                        {"boundary", info.boundary}, {"preset", info.preset},
                        {"viscosity", info.viscosity}, {"enabled", info.enabled},
                        {"visible", info.visible}};
        });
    }
    // ".list" in the name is what classifies this as Read in RtIpcSecurity —
    // keep the suffix if this method is ever renamed.
    if (method == "fluid.list_domains" || method == "gas.list_domains") {
        return enqueueQuery([](UIContext&) {
            std::vector<rtapi::FluidDomainInfo> domains;
            rtapi::Result r = rtapi::listFluidDomains(domains);
            if (!r.ok) return json{{"__error", r.error}};
            json arr = json::array();
            for (const auto& info : domains) {
                arr.push_back(json{
                    {"id", info.id}, {"name", info.name}, {"type", info.type},
                    {"domain_min", json::array({info.domain_min.x, info.domain_min.y, info.domain_min.z})},
                    {"domain_max", json::array({info.domain_max.x, info.domain_max.y, info.domain_max.z})},
                    {"voxel_size", info.voxel_size}, {"particle_count", info.particle_count},
                    {"render_mode", info.render_mode}, {"backend", info.backend},
                    {"boundary", info.boundary}, {"preset", info.preset},
                    {"viscosity", info.viscosity}, {"enabled", info.enabled},
                    {"visible", info.visible}});
            }
            return json{{"domains", arr}};
        });
    }
    if (method == "fluid.seed") {
        std::string domain = requireString(params, "domain");
        Vec3 smin = optionalVec3(params, "seed_min", Vec3(-0.5f, 1.0f, -0.5f));
        Vec3 smax = optionalVec3(params, "seed_max", Vec3(0.5f, 1.5f, 0.5f));
        int ppc = optionalInt(params, "particles_per_cell", 4);
        bool replace = optionalBool(params, "replace", true);
        return enqueueResult([domain, smin, smax, ppc, replace](UIContext&) {
            return rtapi::seedFluidParticles(domain, smin, smax, ppc, replace);
        });
    }
    if (method == "fluid.clear" || method == "gas.clear") {
        std::string domain = requireString(params, "domain");
        return enqueueResult([domain](UIContext&) { return rtapi::clearFluidParticles(domain); });
    }
    if (method == "fluid.remove_domain" || method == "gas.remove_domain") {
        std::string domain = requireString(params, "domain");
        return enqueueResult([domain](UIContext&) { return rtapi::removeFluidDomain(domain); });
    }
    if (method == "fluid.set_param" || method == "gas.set_param") {
        std::string domain = requireString(params, "domain");
        return enqueueResult([domain, params](UIContext&) {
            Vec3 dmin, dmax;
            float voxel_size = 0.0f, viscosity = 0.0f;
            std::string render_mode, backend, boundary, preset;
            bool enabled = false, visible = false;
            const Vec3* p_dmin = nullptr; const Vec3* p_dmax = nullptr;
            const float* p_voxel = nullptr; const float* p_viscosity = nullptr;
            const std::string* p_render = nullptr; const std::string* p_backend = nullptr;
            const std::string* p_boundary = nullptr; const std::string* p_preset = nullptr;
            const bool* p_enabled = nullptr; const bool* p_visible = nullptr;
            if (params.contains("domain_min")) { dmin = requireVec3(params, "domain_min"); p_dmin = &dmin; }
            if (params.contains("domain_max")) { dmax = requireVec3(params, "domain_max"); p_dmax = &dmax; }
            if (params.contains("voxel_size")) { voxel_size = params.at("voxel_size").get<float>(); p_voxel = &voxel_size; }
            if (params.contains("render_mode")) { render_mode = params.at("render_mode").get<std::string>(); p_render = &render_mode; }
            if (params.contains("backend")) { backend = params.at("backend").get<std::string>(); p_backend = &backend; }
            else if (params.contains("device")) { backend = params.at("device").get<std::string>(); p_backend = &backend; }
            if (params.contains("boundary")) { boundary = params.at("boundary").get<std::string>(); p_boundary = &boundary; }
            if (params.contains("preset")) { preset = params.at("preset").get<std::string>(); p_preset = &preset; }
            if (params.contains("viscosity")) { viscosity = params.at("viscosity").get<float>(); p_viscosity = &viscosity; }
            if (params.contains("enabled")) { enabled = params.at("enabled").get<bool>(); p_enabled = &enabled; }
            if (params.contains("visible")) { visible = params.at("visible").get<bool>(); p_visible = &visible; }
            return rtapi::updateFluidDomain(domain, p_dmin, p_dmax, p_voxel, p_render,
                                            p_backend, p_boundary, p_preset, p_viscosity,
                                            p_enabled, p_visible);
        });
    }
    if (method == "fluid.reset" || method == "gas.reset") {
        return enqueueResult([](UIContext&) { return rtapi::resetFluidSimulation(); });
    }
    if (method == "fluid.step" || method == "gas.step") {
        float dt = optionalFloat(params, "dt", 0.0166667f);
        return enqueueResult([dt](UIContext&) { return rtapi::stepFluidSimulation(dt); });
    }
    if (method == "gas.get_settings") {
        std::string domain = requireString(params, "domain");
        return enqueueQuery([domain](UIContext&) {
            rtapi::GasDomainSettings s;
            auto r = rtapi::getGasDomainSettings(domain, s);
            if (!r.ok) return json{{"__error", r.error}};
            return json{{"quality_profile", s.quality_profile},
                        {"resource_budget_mb", s.resource_budget_mb},
                        {"enforce_resource_budget", s.enforce_resource_budget},
                        {"use_sparse_tiles", s.use_sparse_tiles},
                        {"render_to_nanovdb", s.render_to_nanovdb},
                        {"fire_enabled", s.fire_enabled},
                        {"ignition_temperature", s.ignition_temperature},
                        {"burn_rate", s.burn_rate}, {"heat_release", s.heat_release},
                        {"smoke_generation", s.smoke_generation},
                        {"flame_dissipation", s.flame_dissipation},
                        {"fire_max_temperature", s.fire_max_temperature},
                        {"buoyancy_heat", s.buoyancy_heat},
                        {"buoyancy_density", s.buoyancy_density},
                        {"vorticity", s.vorticity}, {"fire_expansion", s.fire_expansion},
                        {"turbulence_strength", s.turbulence_strength},
                        {"turbulence_scale", s.turbulence_scale},
                        {"turbulence_octaves", s.turbulence_octaves},
                        {"turbulence_lacunarity", s.turbulence_lacunarity},
                        {"turbulence_persistence", s.turbulence_persistence},
                        {"turbulence_speed", s.turbulence_speed}};
        });
    }
    if (method == "gas.set_settings") {
        std::string domain = requireString(params, "domain");
        return enqueueResult([domain, params](UIContext&) {
            rtapi::GasDomainSettings s;
            auto r = rtapi::getGasDomainSettings(domain, s);
            if (!r.ok) return r;
#define RT_GAS_JSON(name, type) if (params.contains(#name)) s.name = params.at(#name).get<type>()
            RT_GAS_JSON(quality_profile, std::string);
            RT_GAS_JSON(resource_budget_mb, uint32_t);
            RT_GAS_JSON(enforce_resource_budget, bool);
            RT_GAS_JSON(use_sparse_tiles, bool);
            RT_GAS_JSON(render_to_nanovdb, bool);
            RT_GAS_JSON(fire_enabled, bool);
            RT_GAS_JSON(ignition_temperature, float);
            RT_GAS_JSON(burn_rate, float);
            RT_GAS_JSON(heat_release, float);
            RT_GAS_JSON(smoke_generation, float);
            RT_GAS_JSON(flame_dissipation, float);
            RT_GAS_JSON(fire_max_temperature, float);
            RT_GAS_JSON(buoyancy_heat, float);
            RT_GAS_JSON(buoyancy_density, float);
            RT_GAS_JSON(vorticity, float);
            RT_GAS_JSON(fire_expansion, float);
            RT_GAS_JSON(turbulence_strength, float);
            RT_GAS_JSON(turbulence_scale, float);
            RT_GAS_JSON(turbulence_octaves, int);
            RT_GAS_JSON(turbulence_lacunarity, float);
            RT_GAS_JSON(turbulence_persistence, float);
            RT_GAS_JSON(turbulence_speed, float);
#undef RT_GAS_JSON
            return rtapi::updateGasDomainSettings(domain, s);
        });
    }
    if (method == "gas.get_shader") {
        std::string domain = requireString(params, "domain");
        return enqueueQuery([domain](UIContext&) {
            rtapi::GasShaderSettings s;
            auto r = rtapi::getGasShaderSettings(domain, s);
            if (!r.ok) return nlohmann::json{{"ok", false}, {"error", r.error}};
            return nlohmann::json{
                {"ok", true},
                {"preset", s.preset},
                {"density_multiplier", s.density_multiplier},
                {"density_cutoff", s.density_cutoff},
                {"blackbody_intensity", s.blackbody_intensity},
                {"temperature_min", s.temperature_min},
                {"temperature_max", s.temperature_max},
                {"scattering_coefficient", s.scattering_coefficient},
                {"absorption_coefficient", s.absorption_coefficient}};
        });
    }
    if (method == "gas.set_shader") {
        std::string domain = requireString(params, "domain");
        return enqueueResult([domain, params](UIContext&) {
            rtapi::GasShaderSettings s;
            auto r = rtapi::getGasShaderSettings(domain, s);
            if (!r.ok) return r;
            if (!params.contains("preset")) s.preset.clear();
#define RT_GASSHADER_JSON(name, type) if (params.contains(#name)) s.name = params.at(#name).get<type>()
            RT_GASSHADER_JSON(preset, std::string);
            RT_GASSHADER_JSON(density_multiplier, float);
            RT_GASSHADER_JSON(density_cutoff, float);
            RT_GASSHADER_JSON(blackbody_intensity, float);
            RT_GASSHADER_JSON(temperature_min, float);
            RT_GASSHADER_JSON(temperature_max, float);
            RT_GASSHADER_JSON(scattering_coefficient, float);
            RT_GASSHADER_JSON(absorption_coefficient, float);
#undef RT_GASSHADER_JSON
            return rtapi::updateGasShaderSettings(domain, s);
        });
    }
    if (method == "msf.substances") {
        return enqueueQuery([](UIContext&) {
            std::vector<std::string> names;
            auto r = rtapi::listMaterialSubstances(names);
            if (!r.ok) return nlohmann::json{{"ok", false}, {"error", r.error}};
            return nlohmann::json{{"ok", true}, {"substances", names}};
        });
    }
    if (method == "fluid.get_combustion") {
        std::string domain=requireString(params,"domain");
        return enqueueQuery([domain](UIContext&) {
            rtapi::CombustibleFluidSettings s;
            auto r=rtapi::getCombustibleFluidSettings(domain,s);
            if(!r.ok) return json{{"__error",r.error}};
            return json{{"chemistry_preset",s.chemistry_preset},
                        {"enabled",s.enabled},
                        {"auto_ignite",s.auto_ignite},
                        {"ignition_temperature",s.ignition_temperature},
                        {"evaporation_rate",s.evaporation_rate},
                        {"surface_fuel_capacity",s.surface_fuel_capacity},
                        {"heat_release",s.heat_release},
                        {"smoke_yield",s.smoke_yield},
                        {"surface_cooling",s.surface_cooling}};
        });
    }
    if (method == "fluid.set_combustion") {
        std::string domain=requireString(params,"domain");
        return enqueueResult([domain,params](UIContext&) {
            rtapi::CombustibleFluidSettings s;
            auto r=rtapi::getCombustibleFluidSettings(domain,s);
            if(!r.ok) return r;
#define RT_FLUID_FIRE_JSON(name,type) \
            if(params.contains(#name)) s.name=params.at(#name).get<type>()
            RT_FLUID_FIRE_JSON(enabled,bool);
            if(params.contains("chemistry_preset")) s.chemistry_preset=params.at("chemistry_preset").get<std::string>();
            RT_FLUID_FIRE_JSON(auto_ignite,bool);
            RT_FLUID_FIRE_JSON(ignition_temperature,float);
            RT_FLUID_FIRE_JSON(evaporation_rate,float);
            RT_FLUID_FIRE_JSON(surface_fuel_capacity,float);
            RT_FLUID_FIRE_JSON(heat_release,float);
            RT_FLUID_FIRE_JSON(smoke_yield,float);
            RT_FLUID_FIRE_JSON(surface_cooling,float);
#undef RT_FLUID_FIRE_JSON
            return rtapi::updateCombustibleFluidSettings(domain,s);
        });
    }

    auto terrainInfoJson = [](const rtapi::TerrainInfo& info) {
        return json{{"id", info.id}, {"name", info.name},
                    {"resolution", json::array({info.width, info.height})},
                    {"size", info.size}, {"height_scale", info.height_scale},
                    {"has_node_graph", info.has_node_graph}, {"dirty", info.dirty}};
    };
    if (method == "terrain.list") {
        return enqueueQuery([terrainInfoJson](UIContext&) {
            std::vector<rtapi::TerrainInfo> terrains;
            rtapi::Result r = rtapi::listTerrains(terrains);
            if (!r.ok) return json{{"__error", r.error}};
            json out = json::array();
            for (const auto& info : terrains) out.push_back(terrainInfoJson(info));
            return out;
        });
    }
    if (method == "terrain.get") {
        std::string name = requireString(params, "name");
        return enqueueQuery([name, terrainInfoJson](UIContext&) {
            rtapi::TerrainInfo info;
            rtapi::Result r = rtapi::getTerrain(name, info);
            if (!r.ok) return json{{"__error", r.error}};
            return terrainInfoJson(info);
        });
    }
    if (method == "terrain.create") {
        std::string name = optionalString(params, "name", "Terrain");
        int resolution = optionalInt(params, "resolution", 1024);
        float size = optionalFloat(params, "size", 1000.0f);
        float height_scale = optionalFloat(params, "height_scale", 100.0f);
        return enqueueQuery([name, resolution, size, height_scale, terrainInfoJson](UIContext&) {
            rtapi::TerrainInfo info;
            rtapi::Result r = rtapi::createTerrain(name, resolution, size, height_scale, info);
            if (!r.ok) return json{{"__error", r.error}};
            return terrainInfoJson(info);
        });
    }
    if (method == "terrain.import_heightmap") {
        std::string filepath = requireString(params, "filepath");
        std::string name = optionalString(params, "name", "TerrainImported");
        float size = optionalFloat(params, "size", 1000.0f);
        float height_scale = optionalFloat(params, "height_scale", 100.0f);
        int max_resolution = optionalInt(params, "max_resolution", 2048);
        return enqueueQuery([filepath, name, size, height_scale, max_resolution, terrainInfoJson](UIContext&) {
            rtapi::TerrainInfo info;
            rtapi::Result r = rtapi::importTerrainHeightmap(filepath, name, size, height_scale,
                                                            max_resolution, info);
            if (!r.ok) return json{{"__error", r.error}};
            return terrainInfoJson(info);
        });
    }
    if (method == "terrain.remove") {
        std::string name = requireString(params, "name");
        return enqueueResult([name](UIContext&) { return rtapi::removeTerrain(name); });
    }
    if (method == "terrain.export_heightmap") {
        std::string name = requireString(params, "name");
        std::string filepath = requireString(params, "filepath");
        return enqueueResult([name, filepath](UIContext&) {
            return rtapi::exportTerrainHeightmap(name, filepath);
        });
    }
    auto terrainEvaluationJson = [](const rtapi::TerrainEvaluationInfo& info) {
        return json{{"terrain", info.terrain_name}, {"state", info.state},
                    {"progress", info.progress}, {"current_node_id", info.current_node_id},
                    {"error", info.error}};
    };
    if (method == "terrain.evaluate") {
        std::string name = requireString(params, "name");
        return enqueueQuery([name, terrainEvaluationJson](UIContext&) {
            rtapi::TerrainEvaluationInfo info;
            rtapi::Result r = rtapi::evaluateTerrain(name, info);
            if (!r.ok) return json{{"__error", r.error}};
            return terrainEvaluationJson(info);
        });
    }
    if (method == "terrain.evaluation_status") {
        std::string name = requireString(params, "name");
        return enqueueQuery([name, terrainEvaluationJson](UIContext&) {
            rtapi::TerrainEvaluationInfo info;
            rtapi::Result r = rtapi::getTerrainEvaluationStatus(name, info);
            if (!r.ok) return json{{"__error", r.error}};
            return terrainEvaluationJson(info);
        });
    }
    if (method == "terrain.cancel_evaluation") {
        std::string name = requireString(params, "name");
        return enqueueResult([name](UIContext&) { return rtapi::cancelTerrainEvaluation(name); });
    }
    if (method == "terrain.erode") {
        std::string name = requireString(params, "name");
        rtapi::TerrainErosionSettings settings;
        settings.type = optionalString(params, "type", "hydraulic");
        settings.backend = optionalString(params, "backend", "auto");
        settings.iterations = optionalInt(params, "iterations", 0);
        settings.seed = static_cast<unsigned int>(optionalInt(params, "seed", 1337));
        settings.strength = optionalFloat(params, "strength", 0.2f);
        settings.direction = optionalFloat(params, "direction", 45.0f);
        settings.talus_angle = optionalFloat(params, "talus_angle", 0.5f);
        settings.amount = optionalFloat(params, "amount", 0.3f);
        settings.undo = optionalBool(params, "undo", true);
        return enqueueResult([name, settings](UIContext&) { return rtapi::erodeTerrain(name, settings); });
    }
    if (method == "terrain.apply_preset") {
        std::string name = requireString(params, "name");
        std::string preset = requireString(params, "preset");
        bool replace_graph = optionalBool(params, "replace_graph", false);
        return enqueueResult([name, preset, replace_graph](UIContext&) {
            return rtapi::applyTerrainPreset(name, preset, replace_graph);
        });
    }
    if (method == "terrain.calculate_flow") {
        std::string name = requireString(params, "name");
        return enqueueResult([name](UIContext&) { return rtapi::calculateTerrainFlow(name); });
    }
    if (method == "terrain.sample_height") {
        std::string name = requireString(params, "name");
        float world_x = requireFloat(params, "world_x");
        float world_z = requireFloat(params, "world_z");
        return enqueueQuery([name, world_x, world_z](UIContext&) {
            float height = 0.0f;
            rtapi::Result r = rtapi::sampleTerrainHeight(name, world_x, world_z, height);
            if (!r.ok) return json{{"__error", r.error}};
            return json(height);
        });
    }
    if (method == "terrain.carve_river") {
        std::string name = requireString(params, "name");
        std::string river = requireString(params, "river");
        rtapi::TerrainRiverCarveSettings settings;
        settings.mode = optionalString(params, "mode", "natural");
        settings.depth_multiplier = optionalFloat(params, "depth_multiplier", 1.0f);
        settings.smoothness = optionalFloat(params, "smoothness", 0.5f);
        settings.post_erosion = optionalBool(params, "post_erosion", false);
        settings.post_erosion_iterations = optionalInt(params, "post_erosion_iterations", 12);
        settings.noise_strength = optionalFloat(params, "noise_strength", 0.3f);
        settings.deep_pools = optionalBool(params, "deep_pools", true);
        settings.riffles = optionalBool(params, "riffles", true);
        settings.asymmetric_banks = optionalBool(params, "asymmetric_banks", true);
        settings.point_bars = optionalBool(params, "point_bars", true);
        settings.undo = optionalBool(params, "undo", true);
        return enqueueResult([name, river, settings](UIContext&) {
            return rtapi::carveTerrainRiver(name, river, settings);
        });
    }
    if (method == "terrain.list_rivers") {
        return enqueueQuery([](UIContext&) {
            std::vector<rtapi::TerrainRiverInfo> rivers;
            rtapi::Result r = rtapi::listTerrainRivers(rivers);
            if (!r.ok) return json{{"__error", r.error}};
            json out = json::array();
            for (const auto& river : rivers) {
                out.push_back(json{{"id", river.id}, {"name", river.name},
                                   {"control_points", river.control_point_count},
                                   {"follow_terrain", river.follow_terrain}});
            }
            return out;
        });
    }

    // ── Undo / Redo ─────────────────────────────────────────────────────
    auto hairSettingsJson = [](const rtapi::HairSettings& s) {
        return json{{"guide_count",s.guide_count},{"children_per_guide",s.children_per_guide},{"points_per_strand",s.points_per_strand},{"length",s.length},{"length_variation",s.length_variation},{"root_radius",s.root_radius},{"tip_radius",s.tip_radius},{"clumpiness",s.clumpiness},{"child_radius",s.child_radius},{"curl_frequency",s.curl_frequency},{"curl_radius",s.curl_radius},{"wave_frequency",s.wave_frequency},{"wave_amplitude",s.wave_amplitude},{"frizz",s.frizz},{"roughness",s.roughness},{"gravity",s.gravity},{"force_influence",s.force_influence},{"use_dynamics",s.use_dynamics},{"physics_damping",s.physics_damping},{"physics_stiffness",s.physics_stiffness},{"physics_mass",s.physics_mass},{"use_tangent_shading",s.use_tangent_shading},{"use_bspline",s.use_bspline},{"subdivisions",s.subdivisions}};
    };
    auto hairInfoJson = [hairSettingsJson](const rtapi::HairGroomInfo& i) {
        return json{{"name",i.name},{"bound_mesh",i.bound_mesh},{"guide_count",i.guide_count},{"child_count",i.child_count},{"point_count",i.point_count},{"material",i.material},{"visible",i.visible},{"dirty",i.dirty},{"settings",hairSettingsJson(i.settings)}};
    };
    auto readHairSettings = [](const json& p, rtapi::HairSettings s) {
#define RT_HAIR_JSON(n,t) if (p.contains(#n)) s.n=p.at(#n).get<t>()
        RT_HAIR_JSON(guide_count,uint32_t); RT_HAIR_JSON(children_per_guide,uint32_t); RT_HAIR_JSON(points_per_strand,uint32_t); RT_HAIR_JSON(length,float); RT_HAIR_JSON(length_variation,float); RT_HAIR_JSON(root_radius,float); RT_HAIR_JSON(tip_radius,float); RT_HAIR_JSON(clumpiness,float); RT_HAIR_JSON(child_radius,float); RT_HAIR_JSON(curl_frequency,float); RT_HAIR_JSON(curl_radius,float); RT_HAIR_JSON(wave_frequency,float); RT_HAIR_JSON(wave_amplitude,float); RT_HAIR_JSON(frizz,float); RT_HAIR_JSON(roughness,float); RT_HAIR_JSON(gravity,float); RT_HAIR_JSON(force_influence,float); RT_HAIR_JSON(use_dynamics,bool); RT_HAIR_JSON(physics_damping,float); RT_HAIR_JSON(physics_stiffness,float); RT_HAIR_JSON(physics_mass,float); RT_HAIR_JSON(use_tangent_shading,bool); RT_HAIR_JSON(use_bspline,bool); RT_HAIR_JSON(subdivisions,uint32_t);
#undef RT_HAIR_JSON
        return s;
    };
    if (method == "hair.list") return enqueueQuery([hairInfoJson](UIContext&) { std::vector<rtapi::HairGroomInfo> v; auto r=rtapi::listHairGrooms(v); if(!r.ok)return json{{"__error",r.error}}; json out=json::array(); for(const auto& i:v)out.push_back(hairInfoJson(i)); return out; });
    if (method == "hair.get") { auto name=requireString(params,"name"); return enqueueQuery([name,hairInfoJson](UIContext&){rtapi::HairGroomInfo i; auto r=rtapi::getHairGroom(name,i); return r.ok?hairInfoJson(i):json{{"__error",r.error}};}); }
    if (method == "hair.create") { auto mesh=requireString(params,"mesh"); auto name=optionalString(params,"name","HairGroom"); auto s=readHairSettings(params,{}); return enqueueQuery([mesh,name,s,hairInfoJson](UIContext&){rtapi::HairGroomInfo i; auto r=rtapi::createHairGroom(mesh,name,s,i); return r.ok?hairInfoJson(i):json{{"__error",r.error}};}); }
    if (method == "hair.update") { auto name=requireString(params,"name"); return enqueueResult([name,params,readHairSettings](UIContext&){rtapi::HairGroomInfo i; auto r=rtapi::getHairGroom(name,i); if(!r.ok)return r; i.settings=readHairSettings(params,i.settings); bool visible=params.value("visible",i.visible); return rtapi::updateHairGroom(name,i.settings,params.contains("visible")?&visible:nullptr);}); }
    if (method == "hair.rename") { auto name=requireString(params,"name"); auto next=requireString(params,"new_name"); return enqueueQuery([name,next,hairInfoJson](UIContext&){rtapi::HairGroomInfo i; auto r=rtapi::renameHairGroom(name,next,i); return r.ok?hairInfoJson(i):json{{"__error",r.error}};}); }
    if (method == "hair.remove") { auto name=requireString(params,"name"); return enqueueResult([name](UIContext&){return rtapi::removeHairGroom(name);}); }
    if (method == "hair.restyle") { auto name=requireString(params,"name"); return enqueueResult([name](UIContext&){return rtapi::restyleHairGroom(name);}); }
    if (method == "hair.list_presets") return enqueueQuery([](UIContext&){std::vector<std::string> presets; auto r=rtapi::listHairPresets(presets); return r.ok?json(presets):json{{"__error",r.error}};});
    if (method == "hair.apply_preset") { auto name=requireString(params,"name"); auto preset=requireString(params,"preset"); return enqueueResult([name,preset](UIContext&){return rtapi::applyHairPreset(name,preset);}); }
    if (method == "hair.trim") { auto name=requireString(params,"name"); float factor=requireFloat(params,"length_factor"); return enqueueResult([name,factor](UIContext&){return rtapi::trimHairGroom(name,factor);}); }
    if (method == "hair.grow") { auto name=requireString(params,"name"); float factor=requireFloat(params,"length_factor"); return enqueueResult([name,factor](UIContext&){return rtapi::growHairGroom(name,factor);}); }
    if (method == "hair.comb") { auto name=requireString(params,"name"); Vec3 direction=requireVec3(params,"direction"); float strength=optionalFloat(params,"strength",0.5f); float stiffness=optionalFloat(params,"root_stiffness",0.75f); return enqueueResult([name,direction,strength,stiffness](UIContext&){return rtapi::combHairGroom(name,direction,strength,stiffness);}); }
    if (method == "hair.smooth") { auto name=requireString(params,"name"); float strength=optionalFloat(params,"strength",0.5f); int iterations=optionalInt(params,"iterations",2); return enqueueResult([name,strength,iterations](UIContext&){return rtapi::smoothHairGroom(name,strength,iterations);}); }
    if (method == "hair.reset_simulation") { auto name=requireString(params,"name"); return enqueueResult([name](UIContext&){return rtapi::resetHairSimulation(name);}); }
    if (method == "hair.bake") { auto name=requireString(params,"name"); return enqueueResult([name](UIContext&){return rtapi::bakeHairGroom(name);}); }

    auto paintLayerJson = [](const rtapi::PaintLayerInfo& l) { return json{{"index",l.index},{"id",l.id},{"name",l.name},{"visible",l.visible},{"locked",l.locked},{"opacity",l.opacity},{"blend_mode",l.blend_mode},{"channels",l.channels}}; };
    auto paintTargetJson = [paintLayerJson](const rtapi::PaintTargetInfo& i) { json layers=json::array(); for(const auto& l:i.layers)layers.push_back(paintLayerJson(l)); return json{{"object",i.object_name},{"material_id",i.material_id},{"resolution",i.resolution},{"channels",i.channels},{"layers",layers}}; };
    if (method == "paint.get") { auto object=requireString(params,"object"); int material=optionalInt(params,"material_id",-1); return enqueueQuery([object,material,paintTargetJson](UIContext&){rtapi::PaintTargetInfo i; auto r=rtapi::getPaintTarget(object,material,i); return r.ok?paintTargetJson(i):json{{"__error",r.error}};}); }
    if (method == "paint.ensure") { auto object=requireString(params,"object"); int material=optionalInt(params,"material_id",-1); int resolution=optionalInt(params,"resolution",1024); return enqueueQuery([object,material,resolution,paintTargetJson](UIContext&){rtapi::PaintTargetInfo i; auto r=rtapi::ensurePaintTarget(object,material,resolution,i); return r.ok?paintTargetJson(i):json{{"__error",r.error}};}); }
    if (method == "paint.add_layer") { auto object=requireString(params,"object"); auto name=optionalString(params,"name","Paint Layer"); int material=optionalInt(params,"material_id",-1); int insert=optionalInt(params,"insert_at",-1); return enqueueQuery([object,name,material,insert,paintLayerJson](UIContext&){rtapi::PaintLayerInfo i; auto r=rtapi::addPaintLayer(object,material,name,insert,i); return r.ok?paintLayerJson(i):json{{"__error",r.error}};}); }
    if (method == "paint.remove_layer") { auto object=requireString(params,"object"); int layer=requireInt(params,"layer_index"); int material=optionalInt(params,"material_id",-1); return enqueueResult([object,layer,material](UIContext&){return rtapi::removePaintLayer(object,material,layer);}); }
    if (method == "paint.update_layer") { auto object=requireString(params,"object"); int layer=requireInt(params,"layer_index"); int material=optionalInt(params,"material_id",-1); return enqueueResult([object,layer,material,params](UIContext&){std::string name,blend; bool visible=true,locked=false; float opacity=1.0f; const std::string* pn=nullptr; const std::string* pb=nullptr; const bool* pv=nullptr; const bool* pl=nullptr; const float* po=nullptr; if(params.contains("name")){name=params.at("name").get<std::string>();pn=&name;} if(params.contains("visible")){visible=params.at("visible").get<bool>();pv=&visible;} if(params.contains("locked")){locked=params.at("locked").get<bool>();pl=&locked;} if(params.contains("opacity")){opacity=params.at("opacity").get<float>();po=&opacity;} if(params.contains("blend_mode")){blend=params.at("blend_mode").get<std::string>();pb=&blend;} return rtapi::updatePaintLayer(object,material,layer,pn,pv,pl,po,pb);}); }
    if (method == "paint.fill") { auto object=requireString(params,"object"); int layer=requireInt(params,"layer_index"); int material=optionalInt(params,"material_id",-1); auto channel=requireString(params,"channel"); Vec3 color=requireVec3(params,"color"); return enqueueResult([object,layer,material,channel,color](UIContext&){return rtapi::fillPaintLayer(object,material,layer,channel,color);}); }
    if (method == "paint.clear_channel") { auto object=requireString(params,"object"); int layer=requireInt(params,"layer_index"); int material=optionalInt(params,"material_id",-1); auto channel=requireString(params,"channel"); return enqueueResult([object,layer,material,channel](UIContext&){return rtapi::clearPaintLayerChannel(object,material,layer,channel);}); }
    if (method == "paint.duplicate_layer") { auto object=requireString(params,"object"); int layer=requireInt(params,"layer_index"); int material=optionalInt(params,"material_id",-1); return enqueueQuery([object,layer,material,paintLayerJson](UIContext&){rtapi::PaintLayerInfo i; auto r=rtapi::duplicatePaintLayer(object,material,layer,i); return r.ok?paintLayerJson(i):json{{"__error",r.error}};}); }
    if (method == "paint.move_layer") { auto object=requireString(params,"object"); int from=requireInt(params,"from_index"); int to=requireInt(params,"to_index"); int material=optionalInt(params,"material_id",-1); return enqueueResult([object,from,to,material](UIContext&){return rtapi::movePaintLayer(object,material,from,to);}); }
    if (method == "paint.merge_down") { auto object=requireString(params,"object"); int layer=requireInt(params,"layer_index"); int material=optionalInt(params,"material_id",-1); return enqueueResult([object,layer,material](UIContext&){return rtapi::mergePaintLayerDown(object,material,layer);}); }
    if (method == "paint.flatten") { auto object=requireString(params,"object"); int material=optionalInt(params,"material_id",-1); return enqueueResult([object,material](UIContext&){return rtapi::flattenPaintLayers(object,material);}); }
    if (method == "paint.bake_height_to_normal") { auto object=requireString(params,"object"); int material=optionalInt(params,"material_id",-1); float strength=optionalFloat(params,"strength",4.0f); bool clear=optionalBool(params,"clear_height",false); return enqueueResult([object,material,strength,clear](UIContext&){return rtapi::bakePaintHeightToNormal(object,material,strength,clear);}); }
    if (method == "paint.import_channel") { auto object=requireString(params,"object"); int layer=requireInt(params,"layer_index"); int material=optionalInt(params,"material_id",-1); auto channel=requireString(params,"channel"); auto path=requireString(params,"filepath"); return enqueueResult([object,layer,material,channel,path](UIContext&){return rtapi::importPaintChannel(object,material,layer,channel,path);}); }
    if (method == "paint.export_channel") { auto object=requireString(params,"object"); int layer=optionalInt(params,"layer_index",-1); int material=optionalInt(params,"material_id",-1); auto channel=requireString(params,"channel"); auto path=requireString(params,"filepath"); return enqueueResult([object,layer,material,channel,path](UIContext&){return rtapi::exportPaintChannel(object,material,layer,channel,path);}); }
    if (method == "paint.list_mask_presets") return enqueueQuery([](UIContext&){std::vector<std::string> presets; auto r=rtapi::listPaintMaskPresets(presets); return r.ok?json(presets):json{{"__error",r.error}};});
    if (method == "paint.apply_mask") { auto object=requireString(params,"object"); int layer=requireInt(params,"layer_index"); int material=optionalInt(params,"material_id",-1); auto preset=requireString(params,"preset"); float strength=optionalFloat(params,"strength",1.0f); unsigned int seed=static_cast<unsigned int>(optionalInt(params,"seed",1337)); return enqueueResult([object,layer,material,preset,strength,seed](UIContext&){return rtapi::applyPaintMaskPreset(object,material,layer,preset,strength,seed);}); }

    auto readSculptPoints = [](const json& p) {
        if (!p.contains("points") || !p.at("points").is_array())
            throw std::runtime_error("missing/invalid parameter: points");
        std::vector<Vec3> points;
        for (const auto& value : p.at("points")) {
            if (!value.is_array() || value.size() != 3)
                throw std::runtime_error("each sculpt point must be [x, y, z]");
            points.emplace_back(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
        }
        return points;
    };
    if (method == "sculpt.get") {
        auto object=requireString(params,"object");
        return enqueueQuery([object](UIContext&){rtapi::SculptInfo i; auto r=rtapi::getSculptInfo(object,i); return r.ok?json{{"object",i.object_name},{"vertex_count",i.vertex_count},{"has_mask",i.has_mask},{"mask_min",i.mask_min},{"mask_max",i.mask_max}}:json{{"__error",r.error}};});
    }
    if (method == "sculpt.stroke") {
        auto object=requireString(params,"object"); auto tool=requireString(params,"tool"); auto points=readSculptPoints(params);
        rtapi::SculptStrokeSettings s; s.tool=tool; s.points=std::move(points);
        s.radius=optionalFloat(params,"radius",0.25f); s.strength=optionalFloat(params,"strength",0.05f);
        s.falloff=optionalFloat(params,"falloff",0.75f); s.seed=static_cast<unsigned int>(optionalInt(params,"seed",1337));
        s.use_mask=optionalBool(params,"use_mask",true); s.undo=optionalBool(params,"undo",true);
        if (params.contains("direction")) s.direction=requireVec3(params,"direction");
        return enqueueResult([object,s](UIContext&){return rtapi::applySculptStroke(object,s);});
    }
    if (method == "sculpt.paint_mask") {
        auto object=requireString(params,"object"); auto points=readSculptPoints(params);
        float radius=requireFloat(params,"radius"); float value=requireFloat(params,"value");
        float strength=optionalFloat(params,"strength",1.0f); bool undo=optionalBool(params,"undo",true);
        return enqueueResult([object,points=std::move(points),radius,value,strength,undo](UIContext&){return rtapi::paintSculptMask(object,points,radius,value,strength,undo);});
    }
    if (method == "sculpt.mask_operation") {
        auto object=requireString(params,"object"); auto operation=requireString(params,"operation");
        unsigned int seed=static_cast<unsigned int>(optionalInt(params,"seed",1337)); bool undo=optionalBool(params,"undo",true);
        return enqueueResult([object,operation,seed,undo](UIContext&){return rtapi::applySculptMaskOperation(object,operation,seed,undo);});
    }

    if (method == "undo") {
        return enqueueResult([](UIContext&) { return rtapi::undo(); });
    }
    if (method == "redo") {
        return enqueueResult([](UIContext&) { return rtapi::redo(); });
    }
    if (method == "undo_description") {
        return enqueueQuery([](UIContext&) { return json(rtapi::undoDescription()); });
    }
    if (method == "redo_description") {
        return enqueueQuery([](UIContext&) { return json(rtapi::redoDescription()); });
    }

    // ── Render ──────────────────────────────────────────────────────────
    if (method == "request_render") {
        return enqueueResult([](UIContext&) { return rtapi::requestRender(); });
    }
    if (method == "reset_accumulation") {
        return enqueueResult([](UIContext&) { return rtapi::resetAccumulation(); });
    }
    if (method == "render.start") {
        std::string path = requireString(params, "output_path");
        int spp = requireInt(params, "spp");
        return enqueueResult([path, spp](UIContext&) {
            return rtapi::renderFrame(path, spp);
        });
    }
    if (method == "render.status") {
        return enqueueQuery([](UIContext&) {
            rtapi::RenderJobInfo info = rtapi::renderStatus();
            const char* state = "idle";
            switch (info.state) {
                case rtapi::RenderJobState::Rendering: state = "rendering"; break;
                case rtapi::RenderJobState::Completed: state = "completed"; break;
                case rtapi::RenderJobState::Failed:    state = "failed";    break;
                case rtapi::RenderJobState::Cancelled: state = "cancelled"; break;
                default: break;
            }
            return json{{"state", state}, {"output_path", info.output_path},
                        {"error", info.error}, {"current_samples", info.current_samples},
                        {"target_samples", info.target_samples}, {"progress", info.progress}};
        });
    }
    if (method == "render.cancel") {
        return enqueueResult([](UIContext&) { return rtapi::cancelRender(); });
    }
    if (method == "render.start_sequence") {
        std::string dir = requireString(params, "output_dir");
        int spp = requireInt(params, "spp");
        int sf = requireInt(params, "start_frame");
        int ef = requireInt(params, "end_frame");
        return enqueueResult([dir, spp, sf, ef](UIContext&) {
            return rtapi::renderSequence(dir, spp, sf, ef);
        });
    }
    if (method == "render.sequence_status") {
        return enqueueQuery([](UIContext&) {
            rtapi::SequenceJobInfo info = rtapi::sequenceStatus();
            return json{{"active", info.active}, {"current_frame", info.current_frame},
                        {"start_frame", info.start_frame}, {"end_frame", info.end_frame},
                        {"frame_progress", info.frame_progress},
                        {"total_progress", info.total_progress},
                        {"output_dir", info.output_dir}, {"error", info.error}};
        });
    }
    if (method == "render.cancel_sequence") {
        return enqueueResult([](UIContext&) { return rtapi::cancelSequence(); });
    }

    // ── Project ─────────────────────────────────────────────────────────
    if (method == "project.path") {
        return enqueueQuery([](UIContext&) {
            return json(rtapi::currentProjectPath());
        });
    }
    if (method == "project.save") {
        std::string path;
        if (params.contains("path") && params["path"].is_string())
            path = params["path"].get<std::string>();
        return enqueueResult([path](UIContext&) {
            return rtapi::saveProject(path);
        });
    }
    if (method == "project.open") {
        std::string path = requireString(params, "path");
        return enqueueResult([path](UIContext&) {
            return rtapi::openProject(path);
        });
    }

    // ── Timeline ────────────────────────────────────────────────────────
    if (method == "timeline.get_frame") {
        return enqueueQuery([](UIContext&) {
            return json(rtapi::currentFrame());
        });
    }
    if (method == "timeline.set_frame") {
        int frame = requireInt(params, "frame");
        return enqueueResult([frame](UIContext&) {
            return rtapi::setFrame(frame);
        });
    }

    // ── Keyframes ───────────────────────────────────────────────────────
    if (method == "anim.insert_key") {
        std::string obj = requireString(params, "object_name");
        std::string ch = requireString(params, "channel");
        int frame = requireInt(params, "frame");
        Vec3 val = requireVec3(params, "value");
        return enqueueResult([obj, ch, frame, val](UIContext&) {
            return rtapi::insertKeyframe(obj, ch, frame, val);
        });
    }
    if (method == "anim.remove_key") {
        std::string obj = requireString(params, "object_name");
        int frame = requireInt(params, "frame");
        return enqueueResult([obj, frame](UIContext&) {
            return rtapi::removeKeyframe(obj, frame);
        });
    }
    if (method == "anim.list_keys") {
        std::string obj = requireString(params, "object_name");
        return enqueueQuery([obj](UIContext&) {
            return json(rtapi::listKeyframes(obj));
        });
    }

    // ── Skeletal playback (Faz 5.6c) ────────────────────────────────────
    if (method == "anim.characters") {
        return enqueueQuery([](UIContext&) {
            json result = json::array();
            for (const rtapi::AnimCharacterInfo& info : rtapi::listAnimCharacters())
                result.push_back(animCharacterToJson(info));
            return result;
        });
    }
    if (method == "anim.character") {
        std::string character = requireString(params, "character");
        return enqueueQuery([character](UIContext&) {
            rtapi::AnimCharacterInfo info;
            rtapi::Result r = rtapi::getAnimCharacter(character, info);
            if (!r.ok) return json{{"__error", r.error}};
            return animCharacterToJson(info);
        });
    }
    if (method == "anim.clips") {
        std::string character = requireString(params, "character");
        return enqueueQuery([character](UIContext&) {
            std::vector<rtapi::AnimClipInfo> clips;
            rtapi::Result r = rtapi::listAnimClips(character, clips);
            if (!r.ok) return json{{"__error", r.error}};
            json result = json::array();
            for (const rtapi::AnimClipInfo& clip : clips) {
                result.push_back(json{{"name", clip.name},
                                      {"duration_seconds", clip.duration_seconds},
                                      {"ticks_per_second", clip.ticks_per_second},
                                      {"loop", clip.loop},
                                      {"start_frame", clip.start_frame},
                                      {"end_frame", clip.end_frame}});
            }
            return result;
        });
    }
    if (method == "anim.play") {
        std::string character = requireString(params, "character");
        std::string clip = requireString(params, "clip");
        float blend = params.value("blend", 0.3f);
        int layer = params.value("layer", 0);
        return enqueueResult([character, clip, blend, layer](UIContext&) {
            return rtapi::playAnimClip(character, clip, blend, layer);
        });
    }
    if (method == "anim.stop") {
        std::string character = requireString(params, "character");
        float blend_out = params.value("blend_out", 0.3f);
        int layer = params.value("layer", 0);
        return enqueueResult([character, blend_out, layer](UIContext&) {
            return rtapi::stopAnimation(character, blend_out, layer);
        });
    }
    if (method == "anim.set_paused") {
        std::string character = requireString(params, "character");
        bool paused = params.value("paused", true);
        return enqueueResult([character, paused](UIContext&) {
            return rtapi::setAnimPaused(character, paused);
        });
    }
    if (method == "anim.set_time") {
        std::string character = requireString(params, "character");
        float seconds = requireFloat(params, "seconds");
        int layer = params.value("layer", 0);
        return enqueueResult([character, seconds, layer](UIContext&) {
            return rtapi::setAnimTime(character, seconds, layer);
        });
    }
    if (method == "anim.set_speed") {
        std::string character = requireString(params, "character");
        float speed = requireFloat(params, "speed");
        int layer = params.value("layer", 0);
        return enqueueResult([character, speed, layer](UIContext&) {
            return rtapi::setAnimSpeed(character, speed, layer);
        });
    }
    if (method == "anim.set_loop") {
        std::string character = requireString(params, "character");
        bool loop = requireBool(params, "loop");
        int layer = params.value("layer", 0);
        return enqueueResult([character, loop, layer](UIContext&) {
            return rtapi::setAnimLoop(character, loop, layer);
        });
    }
    if (method == "anim.status") {
        std::string character = requireString(params, "character");
        int layer = params.value("layer", 0);
        return enqueueQuery([character, layer](UIContext&) {
            rtapi::AnimPlaybackInfo info;
            rtapi::Result r = rtapi::getAnimPlayback(character, layer, info);
            if (!r.ok) return json{{"__error", r.error}};
            return animPlaybackToJson(info);
        });
    }
    if (method == "anim.set_graph_param") {
        std::string character = requireString(params, "character");
        std::string name = requireString(params, "name");
        if (!params.contains("value")) throw std::runtime_error("missing param: value");
        if (params["value"].is_boolean()) {
            bool value = params["value"].get<bool>();
            return enqueueResult([character, name, value](UIContext&) {
                return rtapi::setAnimGraphBool(character, name, value);
            });
        }
        float value = params["value"].get<float>();
        return enqueueResult([character, name, value](UIContext&) {
            return rtapi::setAnimGraphFloat(character, name, value);
        });
    }
    if (method == "anim.trigger_graph_param") {
        std::string character = requireString(params, "character");
        std::string name = requireString(params, "name");
        return enqueueResult([character, name](UIContext&) {
            return rtapi::triggerAnimGraphParam(character, name);
        });
    }
    if (method == "anim.graph_status") {
        std::string character = requireString(params, "character");
        return enqueueQuery([character](UIContext&) {
            rtapi::AnimPlaybackInfo info;
            rtapi::Result r = rtapi::getAnimGraphPlayback(character, info);
            if (!r.ok) return json{{"__error", r.error}};
            return animPlaybackToJson(info);
        });
    }

    // ── Node graphs ─────────────────────────────────────────────────────
    if (method == "nodes.types") {
        return enqueueQuery([](UIContext&) {
            json result = json::array();
            for (const rtapi::NodeTypeDesc& t : rtapi::listNodeTypes()) {
                result.push_back(json{
                    {"type_id", t.type_id}, {"category", t.category},
                    {"display_name", t.display_name}, {"description", t.description}
                });
            }
            return result;
        });
    }
    // Graph lifecycle + apply (Faz 5.5b / 5.5c).
    if (method == "nodes.graphs") {
        std::string gt = requireString(params, "graph_type");
        return enqueueQuery([gt](UIContext&) {
            return json(rtapi::listNodeGraphs(gt));
        });
    }
    if (method == "nodes.create_graph") {
        std::string gt = requireString(params, "graph_type");
        std::string gn = requireString(params, "graph_name");
        return enqueueResult([gt, gn](UIContext&) {
            return rtapi::createNodeGraph(gt, gn);
        });
    }
    if (method == "nodes.remove_graph") {
        std::string gt = requireString(params, "graph_type");
        std::string gn = requireString(params, "graph_name");
        return enqueueResult([gt, gn](UIContext&) {
            return rtapi::removeNodeGraph(gt, gn);
        });
    }
    if (method == "nodes.apply") {
        std::string gt = requireString(params, "graph_type");
        std::string gn = requireString(params, "graph_name");
        return enqueueQuery([gt, gn](UIContext&) {
            rtapi::NodeGraphApplyInfo info;
            rtapi::Result r = rtapi::applyNodeGraph(gt, gn, info);
            if (!r.ok) return json{{"__error", r.error}};
            // warnings mirror the editor's diagnostics panel; a successful apply
            // can still report per-pixel slots or an unbound Attribute name.
            return json{{"ok", info.ok}, {"warnings", info.warnings},
                        {"errors", info.errors}};
        });
    }
    if (method == "nodes.add") {
        std::string gt = requireString(params, "graph_type");
        std::string gn = requireString(params, "graph_name");
        std::string tid = requireString(params, "type_id");
        return enqueueQuery([gt, gn, tid](UIContext&) {
            unsigned int id = 0;
            rtapi::Result r = rtapi::addNode(gt, gn, tid, id);
            if (!r.ok) return json{{"__error", r.error}};
            return json(id);
        });
    }
    if (method == "nodes.remove") {
        std::string gt = requireString(params, "graph_type");
        std::string gn = requireString(params, "graph_name");
        unsigned int nid = params.value("node_id", 0u);
        return enqueueResult([gt, gn, nid](UIContext&) {
            return rtapi::removeNode(gt, gn, nid);
        });
    }
    if (method == "nodes.link") {
        std::string gt = requireString(params, "graph_type");
        std::string gn = requireString(params, "graph_name");
        unsigned int fn = params.value("from_node", 0u);
        int fo = requireInt(params, "from_output");
        unsigned int tn = params.value("to_node", 0u);
        int ti = requireInt(params, "to_input");
        return enqueueQuery([gt, gn, fn, fo, tn, ti](UIContext&) {
            unsigned int lid = 0;
            rtapi::Result r = rtapi::linkNodes(gt, gn, fn, fo, tn, ti, lid);
            if (!r.ok) return json{{"__error", r.error}};
            return json(lid);
        });
    }
    if (method == "nodes.list") {
        std::string gt = requireString(params, "graph_type");
        std::string gn = requireString(params, "graph_name");
        return enqueueQuery([gt, gn](UIContext&) {
            std::vector<rtapi::NodeDesc> descs;
            rtapi::Result r = rtapi::listNodes(gt, gn, descs);
            if (!r.ok) return json{{"__error", r.error}};
            json result = json::array();
            for (const rtapi::NodeDesc& d : descs) {
                result.push_back(json{
                    {"id", d.id}, {"type_id", d.type_id},
                    {"display_name", d.display_name},
                    {"inputs", d.input_count}, {"outputs", d.output_count}
                });
            }
            return result;
        });
    }

    if (method == "nodes.list_params") {
        std::string gt = requireString(params, "graph_type");
        std::string gn = requireString(params, "graph_name");
        unsigned int nid = params.value("node_id", 0u);
        return enqueueQuery([gt, gn, nid](UIContext&) {
            std::vector<rtapi::NodeParamInfo> ps;
            rtapi::Result r = rtapi::listNodeParams(gt, gn, nid, ps);
            if (!r.ok) return json{{"__error", r.error}};
            json result = json::array();
            for (const rtapi::NodeParamInfo& p : ps) {
                result.push_back(json{
                    {"index", p.index}, {"name", p.name}, {"type", p.data_type},
                    {"connected", p.connected}, {"value", nodeParamToJson(p.value)}
                });
            }
            return result;
        });
    }
    if (method == "nodes.get_param") {
        std::string gt = requireString(params, "graph_type");
        std::string gn = requireString(params, "graph_name");
        unsigned int nid = params.value("node_id", 0u);
        int pin = requireInt(params, "pin_index");
        return enqueueQuery([gt, gn, nid, pin](UIContext&) {
            rtapi::NodeParamValue v;
            rtapi::Result r = rtapi::getNodeParam(gt, gn, nid, pin, v);
            if (!r.ok) return json{{"__error", r.error}};
            return nodeParamToJson(v);
        });
    }
    if (method == "nodes.set_param") {
        std::string gt = requireString(params, "graph_type");
        std::string gn = requireString(params, "graph_name");
        unsigned int nid = params.value("node_id", 0u);
        int pin = requireInt(params, "pin_index");
        if (!params.contains("value"))
            throw std::runtime_error("missing param: value");
        rtapi::NodeParamValue v = nodeParamFromJson(params["value"]);
        return enqueueResult([gt, gn, nid, pin, v](UIContext&) {
            return rtapi::setNodeParam(gt, gn, nid, pin, v);
        });
    }
    if (method == "nodes.list_properties") {
        std::string gt = requireString(params, "graph_type");
        std::string gn = requireString(params, "graph_name");
        unsigned int nid = params.value("node_id", 0u);
        return enqueueQuery([gt, gn, nid](UIContext&) {
            std::vector<rtapi::NodePropertyInfo> properties;
            rtapi::Result r = rtapi::listNodeProperties(gt, gn, nid, properties);
            if (!r.ok) return json{{"__error", r.error}};
            json result = json::array();
            for (const auto& p : properties) {
                result.push_back(json{{"name", p.name}, {"type", p.data_type},
                                      {"value", nodeParamToJson(p.value)}});
            }
            return result;
        });
    }
    if (method == "nodes.get_property") {
        std::string gt = requireString(params, "graph_type");
        std::string gn = requireString(params, "graph_name");
        unsigned int nid = params.value("node_id", 0u);
        std::string property = requireString(params, "property");
        return enqueueQuery([gt, gn, nid, property](UIContext&) {
            rtapi::NodeParamValue value;
            rtapi::Result r = rtapi::getNodeProperty(gt, gn, nid, property, value);
            if (!r.ok) return json{{"__error", r.error}};
            return nodeParamToJson(value);
        });
    }
    if (method == "nodes.set_property") {
        std::string gt = requireString(params, "graph_type");
        std::string gn = requireString(params, "graph_name");
        unsigned int nid = params.value("node_id", 0u);
        std::string property = requireString(params, "property");
        if (!params.contains("value")) throw std::runtime_error("missing param: value");
        rtapi::NodeParamValue value = nodeParamFromJson(params["value"]);
        return enqueueResult([gt, gn, nid, property, value](UIContext&) {
            return rtapi::setNodeProperty(gt, gn, nid, property, value);
        });
    }

    // ── Force fields (Faz 5.6a) ─────────────────────────────────────────
    // Same shape as the Python binding: the facade takes a whole struct, and
    // forcefield.set_param turns a partial params object into a read-patch-write.
    if (method == "forcefield.types") {
        return enqueueQuery([](UIContext&) { return json(rtapi::forceFieldTypes()); });
    }
    if (method == "forcefield.list") {
        return enqueueQuery([](UIContext&) {
            json result = json::array();
            for (const rtapi::ForceFieldInfo& info : rtapi::listForceFields())
                result.push_back(forceFieldToJson(info));
            return result;
        });
    }
    if (method == "forcefield.get") {
        std::string field = requireString(params, "field");
        return enqueueQuery([field](UIContext&) {
            rtapi::ForceFieldInfo info;
            rtapi::Result r = rtapi::getForceField(field, info);
            if (!r.ok) return json{{"__error", r.error}};
            return forceFieldToJson(info);
        });
    }
    if (method == "forcefield.create") {
        std::string type = requireString(params, "type");
        std::string name = params.value("name", "");
        return enqueueQuery([type, name](UIContext&) {
            rtapi::ForceFieldInfo info;
            rtapi::Result r = rtapi::createForceField(type, name, info);
            if (!r.ok) return json{{"__error", r.error}};
            return forceFieldToJson(info);
        });
    }
    if (method == "forcefield.remove") {
        std::string field = requireString(params, "field");
        return enqueueResult([field](UIContext&) { return rtapi::removeForceField(field); });
    }
    if (method == "forcefield.set_param") {
        std::string field = requireString(params, "field");
        json patch = params;
        patch.erase("field");
        return enqueueResult([field, patch](UIContext&) {
            rtapi::ForceFieldInfo info;
            rtapi::Result read = rtapi::getForceField(field, info);
            if (!read.ok) return read;
            applyForceFieldPatch(patch, info);
            return rtapi::updateForceField(field, info);
        });
    }
    if (method == "forcefield.evaluate") {
        Vec3 position = requireVec3(params, "position");
        float time = params.value("time", 0.0f);
        Vec3 velocity(0.0f, 0.0f, 0.0f);
        if (params.contains("velocity")) velocity = requireVec3(params, "velocity");
        return enqueueQuery([position, time, velocity](UIContext&) {
            Vec3 force;
            rtapi::Result r = rtapi::evaluateForceFields(position, time, velocity, force);
            if (!r.ok) return json{{"__error", r.error}};
            return vec3ToJson(force);
        });
    }

    // ── Particle systems (Faz 5.6b) ─────────────────────────────────────
    if (method == "particle.emitters") {
        return enqueueQuery([](UIContext&) {
            json result = json::array();
            for (const rtapi::ParticleEmitterInfo& info : rtapi::listParticleEmitters())
                result.push_back(particleEmitterToJson(info));
            return result;
        });
    }
    if (method == "particle.get_emitter") {
        std::string emitter = requireString(params, "emitter");
        return enqueueQuery([emitter](UIContext&) {
            rtapi::ParticleEmitterInfo info;
            rtapi::Result r = rtapi::getParticleEmitter(emitter, info);
            if (!r.ok) return json{{"__error", r.error}};
            return particleEmitterToJson(info);
        });
    }
    if (method == "particle.add_emitter") {
        json patch = params;
        return enqueueQuery([patch](UIContext&) {
            rtapi::ParticleEmitterInfo info;   // facade defaults
            applyParticleEmitterPatch(patch, info);
            rtapi::ParticleEmitterInfo created;
            rtapi::Result r = rtapi::addParticleEmitter(info, created);
            if (!r.ok) return json{{"__error", r.error}};
            return particleEmitterToJson(created);
        });
    }
    if (method == "particle.set_emitter") {
        std::string emitter = requireString(params, "emitter");
        json patch = params;
        patch.erase("emitter");
        return enqueueResult([emitter, patch](UIContext&) {
            rtapi::ParticleEmitterInfo info;
            rtapi::Result read = rtapi::getParticleEmitter(emitter, info);
            if (!read.ok) return read;
            applyParticleEmitterPatch(patch, info);
            return rtapi::updateParticleEmitter(emitter, info);
        });
    }
    if (method == "particle.remove_emitter") {
        std::string emitter = requireString(params, "emitter");
        return enqueueResult([emitter](UIContext&) {
            return rtapi::removeParticleEmitter(emitter);
        });
    }
    if (method == "particle.clear_emitters") {
        return enqueueResult([](UIContext&) { return rtapi::clearParticleEmitters(); });
    }
    // ".list" in the name is what classifies this as Read in RtIpcSecurity.
    if (method == "particle.list_systems") {
        return enqueueQuery([](UIContext&) {
            std::vector<rtapi::ParticleSystemInfo> systems;
            rtapi::Result r = rtapi::listParticleSystems(systems);
            if (!r.ok) return json{{"__error", r.error}};
            json arr = json::array();
            for (const auto& s : systems) {
                arr.push_back(json{
                    {"index", s.index}, {"id", s.id}, {"name", s.name},
                    {"active", s.active}, {"domain_count", s.domain_count},
                    {"flow_source_count", s.flow_source_count},
                    {"emitter_count", s.emitter_count},
                    {"collider_count", s.collider_count}});
            }
            return json{{"systems", arr}};
        });
    }
    if (method == "particle.clear_systems") {
        return enqueueResult([](UIContext&) { return rtapi::clearParticleSystems(); });
    }
    if (method == "particle.get_physics") {
        return enqueueQuery([](UIContext&) {
            rtapi::ParticlePhysicsInfo info;
            rtapi::Result r = rtapi::getParticlePhysics(info);
            if (!r.ok) return json{{"__error", r.error}};
            return particlePhysicsToJson(info);
        });
    }
    if (method == "particle.set_physics") {
        json patch = params;
        return enqueueResult([patch](UIContext&) {
            rtapi::ParticlePhysicsInfo info;
            rtapi::Result read = rtapi::getParticlePhysics(info);
            if (!read.ok) return read;
            applyParticlePhysicsPatch(patch, info);
            return rtapi::updateParticlePhysics(info);
        });
    }
    if (method == "particle.stats") {
        return enqueueQuery([](UIContext&) {
            rtapi::ParticleStatsInfo info;
            rtapi::Result r = rtapi::getParticleStats(info);
            if (!r.ok) return json{{"__error", r.error}};
            return json{{"alive_count", info.alive_count}, {"capacity", info.capacity},
                        {"emitter_count", info.emitter_count},
                        {"collider_count", info.collider_count},
                        {"domain_count", info.domain_count},
                        {"total_ms", info.total_ms}, {"emit_ms", info.emit_ms},
                        {"integrate_ms", info.integrate_ms},
                        {"self_collision_ms", info.self_collision_ms},
                        {"grid_domain_ms", info.grid_domain_ms}};
        });
    }
    if (method == "particle.spawn") {
        Vec3 position = requireVec3(params, "position");
        Vec3 velocity(0.0f, 0.0f, 0.0f);
        if (params.contains("velocity")) velocity = requireVec3(params, "velocity");
        float lifetime = params.value("lifetime_seconds", 5.0f);
        float mass = params.value("mass", 1.0f);
        float size = params.value("size", 0.05f);
        return enqueueQuery([position, velocity, lifetime, mass, size](UIContext&) {
            int index = -1;
            rtapi::Result r = rtapi::spawnParticle(position, velocity, lifetime, mass, size, index);
            if (!r.ok) return json{{"__error", r.error}};
            return json(index);
        });
    }
    if (method == "particle.clear") {
        return enqueueResult([](UIContext&) { return rtapi::clearParticles(); });
    }
    if (method == "particle.step") {
        float dt = params.value("dt", 0.0166667f);
        return enqueueResult([dt](UIContext&) { return rtapi::stepParticleSimulation(dt); });
    }

    // ── Script ──────────────────────────────────────────────────────────
    if (method == "script.run_file") {
        std::string path = requireString(params, "path");
        return enqueueResult([path](UIContext&) {
            return rtapi::runScriptFile(path);
        });
    }

    // ── Addons ──────────────────────────────────────────────────────────
    // rtpython::* run on the main thread here (via enqueue), where the embedded
    // interpreter holds the GIL — same context as the UI's addon checkbox path.
    if (method == "addons.list") {
        return enqueueQuery([](UIContext&) {
            json arr = json::array();
            for (const rtpython::AddonInfo& a : rtpython::listAddons()) {
                arr.push_back(json{
                    {"module_name", a.module_name}, {"display_name", a.display_name},
                    {"description", a.description}, {"version", a.version},
                    {"enabled", a.enabled}, {"loaded", a.loaded}});
            }
            return arr;
        });
    }
    if (method == "addons.enable" || method == "addons.disable" || method == "addons.reload") {
        std::string name = requireString(params, "module_name");
        const std::string op = method.substr(std::string("addons.").size());
        return enqueueQuery([name, op](UIContext&) {
            std::string err;
            bool ok = op == "enable"  ? rtpython::enableAddon(name, err)
                    : op == "disable" ? rtpython::disableAddon(name, err)
                                      : rtpython::reloadAddon(name, err);
            if (!ok) return json{{"__error", err}};
            return json(true);
        });
    }

    // ── Events (subscribe not exposed via IPC — stateless protocol) ────

    // ── Unknown method ──────────────────────────────────────────────────
    return json{{"__error", "unknown method: " + method}};
}

// ---------------------------------------------------------------------------
// Parse, authenticate and dispatch a single transport-independent request.
// ---------------------------------------------------------------------------
bool authorizeRemoteCall(uint32_t capabilities, const std::string& method,
                         const json& params, std::string& error) {
    if (!rtipc_security::authorize(capabilities, method, error)) return false;
    const char* key = nullptr;
    if (method == "scene.import_model" || method == "project.open" ||
        method == "project.save" || method == "script.run_file") key = "path";
    else if (method == "terrain.import_heightmap" || method == "terrain.export_heightmap" ||
             method == "paint.import_channel" || method == "paint.export_channel" ||
             method == "ipc.admin.audit.export") key = "filepath";
    else if (method == "render.start") key = "output_path";
    else if (method == "render.start_sequence") key = "output_dir";
    if (!key) return true;
    if (!params.contains(key) || !params[key].is_string() ||
        params[key].get_ref<const std::string&>().empty()) {
        error = "remote file operation requires an explicit path"; return false;
    }
    return rtipc_security::authorizePath(method, params[key].get<std::string>(), error);
}

bool jsonDepthAllowed(const std::string& message, int maximum = 64) {
    int depth = 0; bool quoted = false, escaped = false;
    for (const unsigned char value : message) {
        if (quoted) {
            if (escaped) escaped = false;
            else if (value == '\\') escaped = true;
            else if (value == '"') quoted = false;
            continue;
        }
        if (value == '"') quoted = true;
        else if (value == '{' || value == '[') { if (++depth > maximum) return false; }
        else if (value == '}' || value == ']') { if (--depth < 0) return false; }
    }
    return depth == 0 && !quoted;
}

std::string processJsonMessage(const std::string& message,
                               const rtipc_transport::RequestContext& context = {}) {
    const auto started = std::chrono::steady_clock::now();
    const bool remote_request = context.remote;
    json response;
    int request_id = 0;
    std::string audit_method;
    std::string audit_token;
    auto finish = [&](json payload, bool allowed, const std::string& outcome) {
        std::string encoded = payload.dump();
        if (encoded.size() > kMaxMessageBytes)
            encoded = json{{"id", request_id}, {"error", "response exceeds size limit"}}.dump();
        rtipc_audit::Event event;
        event.connection_id = context.connection_id; event.token_id = audit_token;
        event.peer_address = context.peer_address; event.method = audit_method;
        event.allowed = allowed; event.outcome = outcome;
        event.duration_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count());
        event.bytes_received = message.size(); event.bytes_sent = encoded.size();
        rtipc_audit::record(std::move(event));
        return encoded;
    };
    try {
        if (!jsonDepthAllowed(message)) throw std::runtime_error("JSON nesting exceeds limit");
        json request = json::parse(message);
        if (!request.is_object()) throw std::runtime_error("request must be a JSON object");
        if (request.contains("id")) {
            if (!request["id"].is_number_integer()) throw std::runtime_error("id must be an integer");
            request_id = request["id"].get<int>();
        }
        rtipc_security::Authentication authentication;
        if (remote_request) {
            std::string defense_error;
            if (!rtipc_audit::allowAuthenticationAttempt(context.peer_address, defense_error))
                return finish(json{{"id", request_id}, {"error", "authentication failed"}},
                              false, "auth_throttled");
            if (!request.contains("auth") || !request["auth"].is_string()) {
                rtipc_audit::recordAuthentication(context.peer_address, false);
                return finish(json{{"id", request_id}, {"error", "authentication failed"}},
                              false, "auth_missing");
            }
            authentication = rtipc_security::authenticate(
                request["auth"].get<std::string>(), context.peer_address);
            rtipc_audit::recordAuthentication(context.peer_address, authentication.ok);
            if (!authentication.ok)
                return finish(json{{"id", request_id}, {"error", "authentication failed"}},
                              false, "auth_failed");
            audit_token = authentication.token_id;
            rtipc_session::bindToken(context.connection_id, authentication.token_id);
        }
        if (!request.contains("method") || !request["method"].is_string())
            throw std::runtime_error("missing or invalid 'method' field");
        const std::string method = request["method"].get<std::string>();
        audit_method = method;
        json params = request.value("params", json::object());
        if (!params.is_object()) throw std::runtime_error("params must be a JSON object");
        if (remote_request) {
            std::string defense_error;
            if (!rtipc_audit::allowRequest(authentication.token_id, method, defense_error))
                return finish(json{{"id", request_id}, {"error", defense_error}},
                              false, "rate_limited");
        }
        if (remote_request && method != "batch") {
            std::string policy_error;
            if (!authorizeRemoteCall(authentication.capabilities, method, params, policy_error))
                return finish(json{{"id", request_id}, {"error", policy_error}},
                              false, "policy_denied");
        }
        if (remote_request && method == "batch" && params.contains("calls") && params["calls"].is_array()) {
            for (const auto& call : params["calls"]) {
                if (!call.is_object() || !call.contains("method") || !call["method"].is_string()) continue;
                const std::string child = call["method"].get<std::string>();
                const json child_params = call.value("params", json::object());
                std::string policy_error;
                if (!child_params.is_object() ||
                    !authorizeRemoteCall(authentication.capabilities, child, child_params, policy_error))
                    return finish(json{{"id", request_id}, {"error", policy_error}},
                                  false, "batch_policy_denied");
            }
        }
        json result = dispatchMethod(method, params);
        response = result.is_object() && result.contains("__error")
            ? json{{"id", request_id}, {"error", result["__error"].get<std::string>()}}
            : json{{"id", request_id}, {"result", result}};
    } catch (const json::exception& e) {
        response = json{{"id", request_id}, {"error", std::string("JSON error: ") + e.what()}};
    } catch (const std::exception& e) {
        response = json{{"id", request_id}, {"error", std::string("dispatch error: ") + e.what()}};
    }
    const bool failed = response.is_object() && response.contains("error");
    return finish(response, !failed, failed ? "dispatch_error" : "success");
}


} // namespace

namespace rtipc {

bool start(std::string& error) {
#ifndef _WIN32
    error = "IPC server is only supported on Windows";
    return false;
#else
    if (g_running.load()) {
        return true;  // already running
    }
    if (!rtapi::isBound()) {
        error = "rtapi must be bound before starting IPC server";
        return false;
    }

    g_stop_requested.store(false, std::memory_order_release);

    const char* remote_enabled = std::getenv("RAYTROPHI_REMOTE_IPC");
    const bool remote = remote_enabled && std::string(remote_enabled) == "1";
    const char* bootstrap_token = std::getenv("RAYTROPHI_REMOTE_IPC_TOKEN");
    const char* token_store = std::getenv("RAYTROPHI_REMOTE_IPC_TOKEN_STORE");
    const char* allow_files = std::getenv("RAYTROPHI_REMOTE_IPC_ALLOW_FILES");
    const char* allow_scripts = std::getenv("RAYTROPHI_REMOTE_IPC_ALLOW_SCRIPTS");
    const char* allow_cidrs = std::getenv("RAYTROPHI_REMOTE_IPC_ALLOW_CIDRS");
    const char* audit_path = std::getenv("RAYTROPHI_REMOTE_IPC_AUDIT_JSONL");
    if (remote && (!bootstrap_token || std::strlen(bootstrap_token) < 32)) {
        error = "remote IPC requires a TOKEN of at least 32 characters";
        return false;
    }
    uint32_t capabilities = rtipc_security::Read | rtipc_security::SceneWrite |
                            rtipc_security::Render;
    if (allow_files && std::string(allow_files) == "1")
        capabilities |= rtipc_security::FilesRead | rtipc_security::FilesWrite;
    if (allow_scripts && std::string(allow_scripts) == "1")
        capabilities |= rtipc_security::Scripts | rtipc_security::Addons;
    std::vector<std::string> bootstrap_cidrs;
    if (allow_cidrs && *allow_cidrs) {
        std::string rules = allow_cidrs;
        size_t begin = 0;
        while (begin <= rules.size()) {
            const size_t comma = rules.find(',', begin);
            std::string rule = rules.substr(begin, comma - begin);
            rule.erase(0, rule.find_first_not_of(" \t"));
            const size_t last = rule.find_last_not_of(" \t");
            if (last != std::string::npos) rule.erase(last + 1);
            if (!rule.empty()) bootstrap_cidrs.push_back(std::move(rule));
            if (comma == std::string::npos) break;
            begin = comma + 1;
        }
    }
    if (!rtipc_security::initialize(remote ? bootstrap_token : "", capabilities,
                                    bootstrap_cidrs,
                                    token_store ? token_store : "", error))
        return false;
    if (!rtipc_audit::initialize(audit_path ? audit_path : "", error)) {
        rtipc_security::shutdown(); return false;
    }

    const rtipc_transport::MessageHandler handler =
        [](const std::string& message, const rtipc_transport::RequestContext& context) {
            return processJsonMessage(message, context);
        };
    if (!rtipc_transport::startLocal(g_stop_requested, handler, error)) {
        rtipc_audit::shutdown();
        rtipc_security::shutdown();
        return false;
    }
    if (!rtipc_transport::startTls(g_stop_requested, handler, error)) {
        g_stop_requested.store(true, std::memory_order_release);
        rtipc_transport::stopLocal();
        rtipc_audit::shutdown();
        rtipc_security::shutdown();
        return false;
    }
    g_running.store(true, std::memory_order_release);
    return true;
#endif
}

void stop() noexcept {
    if (!g_running.load(std::memory_order_acquire)) return;

    g_stop_requested.store(true, std::memory_order_release);
    rtipc_transport::stopTls();
    rtipc_transport::stopLocal();
    rtipc_security::shutdown();
    rtipc_session::shutdown();
    rtipc_audit::shutdown();
    g_running.store(false, std::memory_order_release);
}

bool isRunning() {
    return g_running.load(std::memory_order_acquire);
}

bool isRemoteRunning() {
    return rtipc_transport::isTlsRunning();
}

TokenInfo publicTokenInfo(const rtipc_security::TokenInfo& source) {
    TokenInfo result;
    result.id = source.id; result.display_name = source.display_name;
    result.capabilities = source.capabilities; result.created_at = source.created_at;
    result.expires_at = source.expires_at; result.last_used_at = source.last_used_at;
    result.revoked = source.revoked;
    result.allowed_cidrs = source.allowed_cidrs;
    return result;
}

bool createToken(const std::string& display_name, uint32_t capabilities,
                 const std::vector<std::string>& allowed_cidrs,
                 int64_t expires_at, TokenInfo& out_info,
                 std::string& out_raw_token, std::string& error) {
    rtipc_security::TokenInfo internal;
    if (!rtipc_security::createToken(display_name, capabilities, allowed_cidrs, expires_at,
                                     internal, out_raw_token, error)) return false;
    out_info = publicTokenInfo(internal); return true;
}

bool revokeToken(const std::string& token_id, std::string& error) {
    if (!rtipc_security::revokeToken(token_id, error)) return false;
    rtipc_session::disconnectToken(token_id);
    rtipc_transport::interruptLocalDisconnectedSession();
    rtipc_transport::interruptDisconnectedSessions();
    return true;
}

bool updateToken(const std::string& token_id, uint32_t capabilities,
                 const std::vector<std::string>& allowed_cidrs,
                 int64_t expires_at, std::string& error) {
    return rtipc_security::updateToken(token_id, capabilities, allowed_cidrs,
                                      expires_at, error);
}

bool rotateToken(const std::string& token_id, TokenInfo& out_info,
                 std::string& out_raw_token, std::string& error) {
    rtipc_security::TokenInfo internal;
    if (!rtipc_security::rotateToken(token_id, internal, out_raw_token, error)) return false;
    out_info = publicTokenInfo(internal); return true;
}

std::vector<TokenInfo> listTokens() {
    std::vector<TokenInfo> result;
    for (const auto& token : rtipc_security::listTokens()) result.push_back(publicTokenInfo(token));
    return result;
}

SessionInfo publicSessionInfo(const rtipc_session::SessionInfo& source) {
    SessionInfo result;
    result.connection_id = source.connection_id; result.transport = source.transport;
    result.peer_address = source.peer_address; result.peer_port = source.peer_port;
    result.tls_version = source.tls_version; result.tls_cipher = source.tls_cipher;
    result.token_id = source.token_id; result.connected_at = source.connected_at;
    result.last_activity_at = source.last_activity_at; result.request_count = source.request_count;
    result.error_count = source.error_count; result.bytes_received = source.bytes_received;
    result.bytes_sent = source.bytes_sent; result.active = source.active;
    result.disconnect_requested = source.disconnect_requested;
    return result;
}

std::vector<SessionInfo> listSessions(bool include_closed) {
    std::vector<SessionInfo> result;
    for (const auto& session : rtipc_session::listSessions(include_closed))
        result.push_back(publicSessionInfo(session));
    return result;
}

bool disconnectSession(const std::string& connection_id) {
    const bool disconnected = rtipc_session::disconnect(connection_id);
    rtipc_transport::interruptLocalDisconnectedSession();
    rtipc_transport::interruptDisconnectedSessions();
    return disconnected;
}

size_t disconnectAllSessions() {
    const size_t count = rtipc_session::disconnectAll();
    rtipc_transport::interruptLocalDisconnectedSession();
    rtipc_transport::interruptDisconnectedSessions();
    return count;
}

std::vector<AuditEvent> recentAuditEvents(size_t maximum) {
    std::vector<AuditEvent> result;
    for (const auto& source : rtipc_audit::recent(maximum)) {
        AuditEvent event;
        event.sequence = source.sequence; event.timestamp = source.timestamp;
        event.connection_id = source.connection_id; event.token_id = source.token_id;
        event.peer_address = source.peer_address; event.method = source.method;
        event.outcome = source.outcome; event.duration_us = source.duration_us;
        event.bytes_received = source.bytes_received; event.bytes_sent = source.bytes_sent;
        event.allowed = source.allowed; result.push_back(std::move(event));
    }
    return result;
}

void clearAuditEvents() { rtipc_audit::clear(); }

void disableRemoteAccess() noexcept {
    rtipc_session::disconnectAll();
    rtipc_transport::stopTls();
}

RemoteStatus remoteStatus() {
    const auto source = rtipc_transport::tlsStatus();
    RemoteStatus result;
    result.running = source.running; result.bind_address = source.bind_address;
    result.port = source.port; result.certificate_path = source.certificate_path;
    result.certificate_sha256 = source.certificate_sha256;
    result.certificate_not_after = source.certificate_not_after;
    result.subject_alt_names = source.subject_alt_names;
    return result;
}

} // namespace rtipc
