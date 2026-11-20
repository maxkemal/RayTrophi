#include "Paint/BalloonPaintDemo.h"
#include "Paint/BalloonBirthAnimation.h"
#include "Paint/BalloonInteraction.h"
#include "Paint/BalloonBodyInteraction.h"
#include "Paint/HiddenGameObjects.h"
#include "Paint/HiddenPaintSafety.h"

#include "Api/RtApi.h"
#include "Backend/IBackend.h"
#include "Backend/IViewportBackend.h"
#include "GeometryNodesV2.h"
#include "Audio/AudioService.h"
#include "Camera.h"
#include "Material.h"
#include "MaterialManager.h"
#include "Ray.h"
#include "Paint/BalloonEffects.h"
#include "Paint/WetPaintGpu.h"
#include "ParticleAppearanceProfile.h"
#include "ParticleSimulation.h"
#include "SimulationWorld.h"
#include "SceneCommand.h"
#include "TriangleMesh.h"
#include "Texture.h"
#include "UIContext.h"
#include "globals.h"
#include "scene_data.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>

extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
extern std::unique_ptr<Backend::IBackend> g_backend;

namespace Paint {
namespace {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using RayTrophiSim::ParticleAppearanceProfile;
using RayTrophiSim::ParticleSpawnDesc;

constexpr const char* kVisualSystem = "BalloonDemoVisuals";
// Click target radius in metres. Balloons keep 0.8 m between centres, so two
// targets never overlap.
constexpr float kPickRadius = 0.4f;
constexpr std::array<const char*, 4> kSystems = {
    "BalloonPaintRed", "BalloonPaintBlue", "BalloonPaintYellow", "BalloonPaintGreen"
};
// Surprise balloon: gold metal, rainbow paint, confetti and firework sparkles.
// Effects (confetti, sparkles, trails, glitter, contact sparks) live in
// Paint/BalloonEffects; this file decides when they fire.
const Vec3 kGold(1.0f, 0.62f, 0.08f);
constexpr int kConfettiPerSurprise = 220;

// Default palette (config "colors" overrides it, 2..8 entries). LINEAR colour,
// as material base_color and particle profiles read it. Secondary channels are
// kept near zero on purpose: linear 0.08 is 31 % on screen, which is what made
// the first palette look pastel. Wet paint blends where splats overlap, so
// every pair should still mix into something bright (no near-complements).
const std::array<Vec3, 8> kDefaultColors = {
    Vec3(0.90f, 0.010f, 0.020f),   // red
    Vec3(1.00f, 0.180f, 0.000f),   // orange
    Vec3(1.00f, 0.700f, 0.000f),   // yellow
    Vec3(0.180f, 0.800f, 0.005f),  // lime green
    Vec3(0.000f, 0.550f, 0.800f),  // turquoise
    Vec3(0.010f, 0.080f, 0.950f),  // blue
    Vec3(0.350f, 0.010f, 0.900f),  // purple
    Vec3(1.00f, 0.030f, 0.300f)    // hot pink
};

struct Balloon {
    Vec3 position;
    float size = 1.0f;   // scale of the authored balloon (config size_min..size_max)
    bool surprise = false;   // gold balloon: rainbow paint + confetti when popped
    Vec3 anchor;             // drifting centre; position = anchor + bob
    Vec3 drift;              // current drift velocity, m/s
    float phase = 0.0f;      // bob phase
    int color = 0;
    size_t visual_slot = 0;
    bool inflated = false;
    double respawn_at = 0.0;
    double birth_started_at = 0.0;
    bool birth_settled = true;
};

struct State {
    SceneData* scene = nullptr;
    bool prepared = false;
    bool running = false;
    json config;
    std::mt19937 random;
    std::vector<Balloon> balloons;
    std::vector<Vec3> palette;               // config "colors" or kDefaultColors
    std::vector<uint32_t> visual_profiles;   // per palette colour, in the visual system
    std::vector<uint32_t> drop_profiles;     // per palette colour, in kSystems[c % 4]
    uint32_t gold_visual_profile = 0;        // particle-mode surprise balloon
    std::vector<float> visual_transmission; // ordinary balloon's authored value
    uint64_t surprise_pops = 0;
    // Shared room goal: the children fill the room together. room_progress is
    // painted coverage / room_goal, milestones 1..4 = 25/50/75/100 %.
    double coverage_clock = 0.0;
    double room_progress = 0.0;
    int milestones = 0;
    uint64_t shows = 0;
    std::vector<size_t> cascade;     // finale: balloons still to pop, one by one
    double next_cascade = 0.0;
    double finale_reset_at = -1.0;
    // Where balloons appear and drift: config spawn_min/max, else the paint
    // surfaces' bounds minus a margin (a raised ceiling raises the balloons).
    Vec3 spawn_min = Vec3(-3.0f, 0.8f, -3.0f);
    Vec3 spawn_max = Vec3(3.0f, 2.8f, 2.8f);
    bool spawn_auto = true;
    double motion_clock = 0.0;     // mesh transforms are pushed at 30 Hz
    double motion_ms_max = 0.0;    // cost of one push of every moving balloon
    uint64_t motion_pushes = 0;
    double seconds = 0.0;
    double next_event = 0.0;
    double prepare_ms = 0.0;
    double atlas_ms = 0.0;
    uint64_t event_serial = 0;
    uint64_t observed_frames = 0;
    double last_tick_ms = 0.0;     // tickBalloonPaintDemo, effects included
    double max_simulation_ms = 0.0;
    double max_frame_interval_ms = 0.0;
    json events = json::array();
    std::string error;
    // Room reset: idle_reset_seconds after the last player pop, or reset().
    double last_activity = 0.0;
    bool dirty = false;              // something popped since the last reset
    double clean_again_at = -1.0;    // drops in flight at reset land within their lifetime
    uint64_t resets = 0;
    std::vector<Matrix4x4> visual_base;   // authored transform of each balloon mesh
    uint64_t pop_sounds_played = 0;
    uint64_t pop_sounds_dropped = 0;
    std::string pop_sound_error;   // last failure; a pop never fails on its sound
};

State state;

// The surfaces the droplets paint. A config that lists "surfaces" is the
// self-contained sidecar form (the demo builds those surfaces itself). Without
// it the demo paints the scene's existing wet paint surfaces - the form the
// project's game design and the Studio panel use.
std::vector<std::string> surfaceNames(const json& config) {
    std::vector<std::string> names;
    if (config.contains("surfaces")) {
        for (const auto& target : config["surfaces"]) {
            names.push_back(target.get<std::string>());
        }
    } else {
        for (const auto& surface : wetPaintGpu().surfaces()) {
            if (std::find(names.begin(), names.end(), surface.object) == names.end()) {
                names.push_back(surface.object);
            }
        }
    }
    return names;
}

void validateMeshVisuals(SceneData& scene, const json& config) {
    if (!config.contains("visual_nodes")) {
        return;
    }
    const auto& nodes = config["visual_nodes"];
    if (!nodes.is_array() || nodes.size() != config["balloons"].get<size_t>()) {
        throw std::runtime_error("visual_nodes must name one flat mesh per balloon");
    }
    auto* context = rtapi::boundContext();
    if (!context || &context->scene != &scene) {
        throw std::runtime_error("mesh balloon visuals require the bound scene context");
    }
    std::vector<std::string> names;
    std::vector<uint16_t> materials;
    for (const auto& value : nodes) {
        if (!value.is_string() || value.get<std::string>().empty()) {
            throw std::runtime_error("visual_nodes entries must be non-empty strings");
        }
        const auto name = value.get<std::string>();
        if (std::find(names.begin(), names.end(), name) != names.end()) {
            throw std::runtime_error("duplicate visual node: " + name);
        }
        for (const auto& target : surfaceNames(config)) {
            if (target == name) {
                throw std::runtime_error("balloon visual cannot also be a paint surface: " + name);
            }
        }
        auto* mesh = scene.getFlatNodeMesh(name);
        if (!mesh || !mesh->geometry || !mesh->transform || !mesh->num_triangles()) {
            throw std::runtime_error("flat balloon mesh not found: " + name);
        }
        for (const auto& weights : mesh->geometry->skin_weights) {
            if (!weights.empty()) {
                throw std::runtime_error("balloon visual must be a rigid flat mesh: " + name);
            }
        }
        const auto* ids = mesh->geometry->get_material_ids();
        const size_t material_count =
            mesh->geometry->get_core_attribute_count(DNA::Attr::MaterialID);
        if (!ids || material_count != mesh->geometry->get_vertex_count() || material_count == 0) {
            throw std::runtime_error("balloon mesh needs a material: " + name);
        }
        const uint16_t material = ids[0];
        for (size_t i = 0; i < material_count; ++i) {
            if (ids[i] != material) {
                throw std::runtime_error("balloon mesh must use one material: " + name);
            }
        }
        if (std::find(materials.begin(), materials.end(), material) != materials.end()) {
            throw std::runtime_error("balloon meshes must have distinct materials");
        }
        size_t node_entries = 0;
        for (const auto& object : scene.world.objects) {
            auto other = std::dynamic_pointer_cast<TriangleMesh>(object);
            if (!other || !other->geometry) {
                continue;
            }
            if (other->nodeName == name) {
                ++node_entries;
                continue;
            }
            const auto* other_ids = other->geometry->get_material_ids();
            const size_t other_count =
                other->geometry->get_core_attribute_count(DNA::Attr::MaterialID);
            for (size_t i = 0; other_ids && i < other_count; ++i) {
                if (other_ids[i] == material) {
                    throw std::runtime_error(
                        "balloon material is shared with another object: " + name);
                }
            }
        }
        if (node_entries != 1) {
            throw std::runtime_error("balloon visual must resolve to exactly one flat mesh: " + name);
        }
        rtapi::MaterialParamValue color;
        const auto result = rtapi::getMaterialParam(name, "base_color", color);
        if (!result.ok || !color.is_color) {
            throw std::runtime_error("balloon mesh needs an editable color material: " + name);
        }
        // A base colour texture overrides the colour the demo sets: material.get
        // reports the balloon red while it renders the texture (a white paint
        // texture left by making the balloon a paint surface once). Refuse it.
        if (const Material* bound = MaterialManager::getInstance().getMaterial(material);
            bound && bound->albedoProperty.texture) {
            throw std::runtime_error(
                "balloon material has a base colour texture (" +
                bound->albedoProperty.texture->name +
                "), so the balloon colour cannot show; clear it (material.clear_texture "
                "slot base_color): " + name);
        }
        names.push_back(name);
        materials.push_back(material);
    }
}

// Instance transform to the viewport and render backends, as a gizmo drag does
// (scene_ui_gizmos.cpp updateGizmoObjectTransformOnActiveBackends). Silent.
void pushBalloonTransform(UIContext& context, const std::string& name, const Matrix4x4& matrix) {
    Backend::IBackend* viewport = g_viewport_backend
        ? static_cast<Backend::IBackend*>(g_viewport_backend.get()) : context.backend_ptr;
    if (viewport) {
        viewport->updateObjectTransform(name, matrix);
        viewport->resetAccumulation();
    }
    Backend::IBackend* render = g_backend ? g_backend.get() : nullptr;
    if (render && render != viewport && render->isUsingTLAS()) {
        render->updateObjectTransform(name, matrix);
        render->resetAccumulation();
    }
}

void updateMeshVisual(SceneData& scene, size_t index, const Balloon& balloon, bool visible) {
    if (!state.config.contains("visual_nodes")) {
        return;
    }
    const auto name = state.config["visual_nodes"][index].get<std::string>();
    auto* mesh = scene.getFlatNodeMesh(name);
    auto* context = rtapi::boundContext();
    if (!mesh || !mesh->transform || !context || &context->scene != &scene) {
        throw std::runtime_error("balloon visual node/context was removed: " + name);
    }
    if (visible) {
        // Authored rotation/scale times the balloon's size; never compounded
        // from the previous respawn's scale.
        Matrix4x4 matrix = index < state.visual_base.size() ? state.visual_base[index]
                                                             : mesh->transform->base;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                matrix.m[r][c] *= balloon.size;
            }
        }
        matrix.m[0][3] = balloon.position.x;
        matrix.m[1][3] = balloon.position.y;
        matrix.m[2][3] = balloon.position.z;
        // The flat part of TransformCommand::applyState (transform + CPU vertex
        // rebake, which also catches up a drifted balloon) and the gizmo's
        // backend push - no undo record, no log line per respawn.
        mesh->transform->setBase(matrix);
        GeometryNodesV2::rebakeFromOrig(*mesh);
        pushBalloonTransform(*context, name, matrix);
        const auto color = rtapi::setMaterialParam(
            name, "base_color", balloon.surprise ? kGold : state.palette[balloon.color]);
        if (!color.ok) {
            throw std::runtime_error(color.error);
        }
        // Gold reads as gold only as a metal; every respawn sets it back.
        const auto metal = rtapi::setMaterialParam(name, "metallic", balloon.surprise ? 1.0f : 0.0f);
        if (!metal.ok) {
            throw std::runtime_error(metal.error);
        }
        {
            const float authored = state.visual_transmission.at(index);
            const float wanted = balloon.surprise ? 0.0f :
                state.config.value("transmission", authored);
            rtapi::MaterialParamValue current;
            // Written only when it differs: every write is an undo record.
            if (!rtapi::getMaterialParam(name, "transmission", current).ok ||
                std::abs(current.scalar - wanted) > 1e-4f) {
                const auto clear = rtapi::setMaterialParam(name, "transmission", wanted);
                if (!clear.ok) {
                    throw std::runtime_error(clear.error);
                }
            }
        }
        if (!mesh->visible) {
            // Only the first show after prepare changes the instance set.
            const auto result = rtapi::setObjectVisible(name, true);
            if (!result.ok) {
                throw std::runtime_error(result.error);
            }
        }
    } else {
        // A popped balloon is SHRUNK in place, never hidden. Hiding drops its
        // instance; a changed instance count makes the RT TLAS take its
        // destroy/recreate branch, which does not wait for in-flight traces
        // (RT shadow / RayFusion) - the Player lost the Vulkan device after a
        // run of pops (SceneLog: createTLAS/topology_change, then device lost
        // at vkQueueSubmit). A scaled instance keeps the count, so the TLAS is
        // always rebuilt in place.
        Matrix4x4 matrix = index < state.visual_base.size() ? state.visual_base[index]
                                                             : mesh->transform->base;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                matrix.m[r][c] *= 1.0e-4f;
            }
        }
        matrix.m[0][3] = balloon.position.x;
        matrix.m[1][3] = balloon.position.y;
        matrix.m[2][3] = balloon.position.z;
        mesh->transform->setBase(matrix);
        GeometryNodesV2::rebakeFromOrig(*mesh);
        pushBalloonTransform(*context, name, matrix);
    }
}

void hideMeshVisuals() {
    auto* context = rtapi::boundContext();
    if (!context || &context->scene != state.scene || !state.config.contains("visual_nodes")) {
        return;
    }
    for (const auto& node : state.config["visual_nodes"]) {
        rtapi::setObjectVisible(node.get<std::string>(), false);
    }
}

double elapsedMs(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

float randomRange(float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(state.random);
}

RayTrophiSim::ParticleSimulationSystem* system(SceneData& scene, const std::string& name) {
    for (auto& entry : scene.particle_systems) {
        if (entry.name == name) {
            return entry.runtime.get();
        }
    }
    return nullptr;
}

// ---- Droplet colliders ------------------------------------------------------
// The droplets collide through the existing simulation collider assignment
// (ParticleColliderDesc, saved with each particle system). The game design
// only writes the SAME collider into all four droplet systems, named
// "<object> PaintCollider", bound to the object (source_name) so it follows it.

const char* colliderModeName(RayTrophiSim::ParticleColliderSourceMode mode) {
    using M = RayTrophiSim::ParticleColliderSourceMode;
    switch (mode) {
        case M::PlaneY: return "plane";
        case M::ObjectAABB: return "aabb";
        case M::ObjectOBB: return "obb";
        case M::Sphere: return "sphere";
        case M::Capsule: return "capsule";
        case M::ObjectMeshSDF: return "mesh_sdf";
        case M::ObjectConvexDecomp: return "convex";
        case M::ObjectMeshBVH: return "mesh_bvh";
    }
    return "unknown";
}

std::string managedColliderName(const std::string& object) {
    return object + " PaintCollider";
}

bool colliderCooked(const RayTrophiSim::ParticleColliderDesc& c) {
    if (c.source_mode != RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF) return true;
    return c.sdf_grid_data && !c.sdf_grid_data->empty() && c.sdf_nx > 0 && c.sdf_ny > 0 &&
           c.sdf_nz > 0;
}

std::shared_ptr<RayTrophiSim::ParticleSimulationSystem> systemShared(SceneData& scene,
                                                                    const std::string& name) {
    for (auto& entry : scene.particle_systems) {
        if (entry.name == name) return entry.runtime;
    }
    return nullptr;
}

json validatedConfig(const json& input, SceneData& scene) {
    if (!input.is_object()) {
        throw std::runtime_error("config must be an object");
    }
    const std::array allowed = {
        "surfaces", "resolution", "seed", "balloons", "particles_per_balloon",
        "interval_min", "interval_max", "autostart", "respawn_seconds", "birth_seconds",
        "visual_nodes", "pop_sound", "interaction_mode", "body_controls",
        "size_min", "size_max", "idle_reset_seconds", "colors",
        "surprise_chance", "surprise_sound", "room_goal", "celebrate_sound",
        "finale_reset_seconds", "spawn_min", "spawn_max", "float_amplitude", "drift_speed",
        "transmission", "sparkles", "sparkle_trails", "drop_sparks", "drop_trails",
        "burst_clumpiness", "burst_jets_max"
    };
    for (auto it = input.begin(); it != input.end(); ++it) {
        if (std::find(allowed.begin(), allowed.end(), it.key()) == allowed.end()) {
            throw std::runtime_error("unknown config field: " + it.key());
        }
    }
    json c = {{"resolution", 1024}, {"seed", 417}, {"balloons", 12},
              {"particles_per_balloon", 1800}, {"interval_min", 2.5},
              {"interval_max", 5.0}, {"autostart", true}, {"respawn_seconds", 3.5},
              {"birth_seconds", 0.75}, {"body_controls", false},
              {"interaction_mode", input.contains("visual_nodes") ? "carry" : "pop"},
              {"size_min", 1.0}, {"size_max", 1.0}, {"idle_reset_seconds", 0.0},
              {"surprise_chance", 0.1}, {"room_goal", 0.5}, {"finale_reset_seconds", 10.0},
              {"float_amplitude", 0.08}, {"drift_speed", 0.12},
              {"sparkles", 90}, {"sparkle_trails", true}, {"drop_sparks", 0.25}, {"drop_trails", true},
              {"burst_clumpiness", 0.7}, {"burst_jets_max", 10}};
    c.update(input);
    if (!c["interaction_mode"].is_string() ||
        (c["interaction_mode"] != "pop" && c["interaction_mode"] != "carry")) {
        throw std::runtime_error("interaction_mode must be pop|carry");
    }
    if (c["interaction_mode"] == "carry" && !c.contains("visual_nodes")) {
        throw std::runtime_error("carry requires mesh balloon visuals");
    }
    if (!c["body_controls"].is_boolean() ||
        (c["body_controls"] == true && !c.contains("visual_nodes"))) {
        throw std::runtime_error("body_controls requires bool and mesh balloon visuals");
    }
    // Pop shape (Paint::balloonBurstPattern): 0 = even fine spray, 1 = ligaments,
    // clumps and blobs only; ligaments per pop are drawn from 3..burst_jets_max.
    if (!c["burst_clumpiness"].is_number() || !(c["burst_clumpiness"].get<double>() >= 0.0) ||
        c["burst_clumpiness"].get<double>() > 1.0) {
        throw std::runtime_error("burst_clumpiness must be a number in 0..1 (0 = even spray)");
    }
    if (!c["burst_jets_max"].is_number_integer() || c["burst_jets_max"].get<int64_t>() < 3 ||
        c["burst_jets_max"].get<int64_t>() > 24) {
        throw std::runtime_error("burst_jets_max must be an integer 3..24");
    }
    for (const char* key : {"resolution", "seed", "balloons", "particles_per_balloon"}) {
        if (!c[key].is_number_integer()) {
            throw std::runtime_error(std::string(key) + " must be an integer");
        }
    }
    for (const char* key : {"size_min", "size_max", "idle_reset_seconds", "surprise_chance",
                            "room_goal", "finale_reset_seconds", "float_amplitude", "drift_speed"}) {
        if (!c[key].is_number()) {
            throw std::runtime_error(std::string(key) + " must be a number");
        }
    }
    if (c.contains("colors")) {
        // Linear RGB 0..1, as material base_color. 2..8 colours.
        const auto& colors = c["colors"];
        if (!colors.is_array() || colors.size() < 2 || colors.size() > 8) {
            throw std::runtime_error("colors must list 2..8 [r, g, b] colours");
        }
        for (const auto& color : colors) {
            if (!color.is_array() || color.size() != 3) {
                throw std::runtime_error("each colour must be [r, g, b]");
            }
            for (const auto& channel : color) {
                if (!channel.is_number() || !(channel.get<double>() >= 0.0) ||
                    channel.get<double>() > 1.0) {
                    throw std::runtime_error("colour channels must be numbers in 0..1 (linear)");
                }
            }
        }
    }
    const double size_min = c["size_min"].get<double>();
    const double size_max = c["size_max"].get<double>();
    const double idle_reset = c["idle_reset_seconds"].get<double>();
    if (!std::isfinite(size_min) || !std::isfinite(size_max) || size_min < 0.3 ||
        size_max > 2.0 || size_min > size_max) {
        throw std::runtime_error("balloon size: 0.3 <= size_min <= size_max <= 2.0");
    }
    if (!std::isfinite(idle_reset) || (idle_reset != 0.0 && (idle_reset < 10.0 || idle_reset > 3600.0))) {
        throw std::runtime_error("idle_reset_seconds: 0 (off) or 10..3600");
    }
    // One translucency for every balloon. Each balloon owns its material (the
    // colours differ), so editing twelve materials by hand drifted: popped
    // balloons are shrunk to nothing and cannot be selected, and measured
    // 2026-11-17 the twelve held 0..1 (one opaque, one fully clear). Absent =
    // the materials are left as authored.
    if (c.contains("transmission") &&
        (!c["transmission"].is_number() || !(c["transmission"].get<double>() >= 0.0) ||
         c["transmission"].get<double>() > 1.0)) {
        throw std::runtime_error("transmission must be a number in 0..1");
    }
    const double surprise = c["surprise_chance"].get<double>();
    if (!(surprise >= 0.0 && surprise <= 0.5)) {
        throw std::runtime_error("surprise_chance must be 0..0.5");
    }
    if (!c["sparkles"].is_number_integer() || c["sparkles"].get<int64_t>() < 0 ||
        c["sparkles"].get<int64_t>() > 400) {
        throw std::runtime_error("sparkles must be an integer 0..400 (0 = off)");
    }
    if (!c["sparkle_trails"].is_boolean() || !c["drop_trails"].is_boolean()) {
        throw std::runtime_error("sparkle_trails and drop_trails must be booleans");
    }
    if (!c["drop_sparks"].is_number() || !(c["drop_sparks"].get<double>() >= 0.0) ||
        c["drop_sparks"].get<double>() > 1.0) {
        throw std::runtime_error("drop_sparks must be a number in 0..1 (0 = off)");
    }
    const double goal = c["room_goal"].get<double>();
    if (!(goal == 0.0 || (goal >= 0.05 && goal <= 1.0))) {
        throw std::runtime_error("room_goal: 0 (off) or 0.05..1 of the paintable texels");
    }
    if (!(c["float_amplitude"].get<double>() >= 0.0 && c["float_amplitude"].get<double>() <= 0.5) ||
        !(c["drift_speed"].get<double>() >= 0.0 && c["drift_speed"].get<double>() <= 1.0)) {
        throw std::runtime_error("float_amplitude 0..0.5 m, drift_speed 0..1 m/s");
    }
    if (c.contains("spawn_min") != c.contains("spawn_max")) {
        throw std::runtime_error("spawn_min and spawn_max go together (or neither: automatic)");
    }
    if (c.contains("spawn_min")) {
        for (const char* key : {"spawn_min", "spawn_max"}) {
            const auto& v = c[key];
            if (!v.is_array() || v.size() != 3 || !v[0].is_number() || !v[1].is_number() ||
                !v[2].is_number()) {
                throw std::runtime_error(std::string(key) + " must be [x, y, z]");
            }
        }
        for (int k = 0; k < 3; ++k) {
            if (!(c["spawn_max"][k].get<double>() - c["spawn_min"][k].get<double>() >= 0.2)) {
                throw std::runtime_error("spawn_max must exceed spawn_min by 0.2 m on every axis");
            }
        }
    }
    const double finale_reset = c["finale_reset_seconds"].get<double>();
    if (!(finale_reset == 0.0 || (finale_reset >= 3.0 && finale_reset <= 120.0))) {
        throw std::runtime_error("finale_reset_seconds: 0 (no auto reset) or 3..120");
    }
    const int64_t resolution = c["resolution"].get<int64_t>();
    const int64_t balloons = c["balloons"].get<int64_t>();
    const int64_t particles = c["particles_per_balloon"].get<int64_t>();
    if (resolution < 32 || resolution > 2048 || balloons < 3 || balloons > 24 ||
        particles < 32 || particles > 8192 ||
        static_cast<double>(balloons) * particles * size_max * size_max > 131072.0 ||
        c["seed"].get<int64_t>() < 0 || c["seed"].get<int64_t>() > 2147483647) {
        throw std::runtime_error("resolution 32..2048, balloons 3..24, particles 32..8192; "
                                 "balloons * particles * size_max^2 <= 131072, seed 0..2147483647 required");
    }
    const double lo = c["interval_min"].get<double>();
    const double hi = c["interval_max"].get<double>();
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo < 2.0 || hi < lo || hi > 30.0) {
        throw std::runtime_error("intervals must be finite seconds: 2 <= min <= max <= 30");
    }
    if (!c["birth_seconds"].is_number() || !std::isfinite(c["birth_seconds"].get<double>()) ||
        c["birth_seconds"].get<double>() < 0.0 || c["birth_seconds"].get<double>() > 3.0) {
        throw std::runtime_error("birth_seconds must be finite seconds in 0..3 (0 = instant)");
    }
    if (!c["respawn_seconds"].is_number() ||
        !std::isfinite(c["respawn_seconds"].get<double>()) ||
        c["respawn_seconds"].get<double>() < 0.25 || c["respawn_seconds"].get<double>() > 60.0) {
        throw std::runtime_error("respawn_seconds must be finite seconds in 0.25..60");
    }
    if (!c["autostart"].is_boolean()) {
        throw std::runtime_error("autostart must be boolean");
    }
    for (const char* key : {"pop_sound", "surprise_sound", "celebrate_sound"}) {
        // An audio source id; its asset plays once at every popped (surprise) balloon.
        if (!c.contains(key)) continue;
        if (!c[key].is_string()) {
            throw std::runtime_error(std::string(key) + " must be an audio source id (string)");
        }
        if (c[key].get<std::string>().empty()) {
            c.erase(key);   // empty = silent / falls back to pop_sound
        } else if (scene.audio) {
            const json found = scene.audio->invoke("get", {{"id", c[key]}});
            if (found.contains("error") || found.contains("__error")) {
                throw std::runtime_error(std::string(key) + ": no audio source " +
                                         c[key].get<std::string>());
            }
        }
    }
    if (c.contains("surfaces")) {
        if (!c["surfaces"].is_array() || c["surfaces"].empty()) {
            throw std::runtime_error("surfaces must name at least one flat mesh");
        }
        for (const auto& target : c["surfaces"]) {
            if (!target.is_string()) {
                throw std::runtime_error("surfaces entries must be object names");
            }
        }
    } else if (wetPaintGpu().empty()) {
        throw std::runtime_error("the scene has no wet paint surface: add one (Game Design "
                                 "panel or paint.wet_add_surface) or list surfaces in the config");
    }
    const bool builds_surfaces = c.contains("surfaces");
    std::vector<std::string> targets;
    for (const std::string& name : surfaceNames(c)) {
        if (std::find(targets.begin(), targets.end(), name) != targets.end()) {
            throw std::runtime_error("duplicate surface: " + name);
        }
        bool found = false;
        for (const auto& object : scene.world.objects) {
            auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
            if (mesh && mesh->nodeName == name && mesh->geometry && mesh->num_triangles()) {
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("flat paint surface not found: " + name);
        }
        targets.push_back(name);
        for (const auto& entry : scene.mesh_paint_texture_sets) {
            // Only the sidecar form rebuilds the surface at config resolution.
            if (!builds_surfaces || entry.second.target_node_name != name) {
                continue;
            }
            const auto texture = entry.second.getTexture(PaintChannel::BaseColor);
            if (texture && (texture->width != resolution || texture->height != resolution)) {
                throw std::runtime_error("existing paint texture resolution differs: reload "
                                         "the clean project before configuring " + name);
            }
        }
    }
    for (const char* name : kSystems) {
        auto* runtime = system(scene, name);
        if (!runtime || runtime->colliders().empty() || !runtime->gridDomains().empty()) {
            throw std::runtime_error(std::string("the demo needs a particle system named ") + name +
                                     " with a collider on every paint surface and no domain");
        }
    }
    validateMeshVisuals(scene, c);
    std::string cooking;
    if (!balloonPaintCollidersReady(scene, cooking)) {
        // The pools are GPU-required and an uncooked SDF keeps a system on the
        // CPU, so warmup would fail with a less useful message.
        throw std::runtime_error("collider still cooking its SDF: " + cooking +
                                 " (configure again when ready)");
    }
    return c;
}

std::string colorKey(const Vec3& c) {
    char text[16];
    std::snprintf(text, sizeof(text), "#%02X%02X%02X",
                  static_cast<unsigned>(std::clamp(c.x, 0.0f, 1.0f) * 255.0f + 0.5f),
                  static_cast<unsigned>(std::clamp(c.y, 0.0f, 1.0f) * 255.0f + 0.5f),
                  static_cast<unsigned>(std::clamp(c.z, 0.0f, 1.0f) * 255.0f + 0.5f));
    return text;
}

ParticleAppearanceProfile profile(const std::string& name, const Vec3& color, float size) {
    ParticleAppearanceProfile p;
    p.name = name;
    p.blend = RayTrophiSim::ParticleAppearanceBlend::Alpha;
    p.color_ramp = {{0.0f, color}, {1.0f, color}};
    p.opacity_curve = {{0.0f, 1.0f}, {1.0f, 1.0f}};
    p.size_curve = {{0.0f, size}, {1.0f, size}};
    p.emission_curve = {{0.0f, 1.0f}, {1.0f, 1.0f}};
    return p;
}

// The room the balloons fill: config spawn_min/max, else the union of the wet
// paint surfaces' bounds, 0.9 m in from the walls, 0.8 m above the floor and
// 1.0 m under the ceiling. Falls back to the old fixed box.
void resolveSpawnVolume(SceneData& scene) {
    if (state.config.contains("spawn_min")) {
        const auto& a = state.config["spawn_min"];
        const auto& b = state.config["spawn_max"];
        state.spawn_min = Vec3(a[0].get<float>(), a[1].get<float>(), a[2].get<float>());
        state.spawn_max = Vec3(b[0].get<float>(), b[1].get<float>(), b[2].get<float>());
        state.spawn_auto = false;
        return;
    }
    state.spawn_auto = true;
    // The INSIDE of the room, not the outer bounds: a ceiling slab larger than
    // the room (z +-6.6 over walls at +-4) put balloons outside the side walls.
    // A plane-like surface (thinner than 0.3 m on one axis, over 1 m on the
    // other two) is a wall on that axis; the nearest wall on each side of the
    // centre bounds the room. An open side (no wall, e.g. the camera side) is
    // bounded by where the walls along it end.
    std::vector<AABB> boxes;
    Vec3 umn(std::numeric_limits<float>::max()), umx(-std::numeric_limits<float>::max());
    for (const auto& surface : wetPaintGpu().surfaces()) {
        auto* mesh = scene.getFlatNodeMesh(surface.object);
        AABB box;
        if (mesh && mesh->bounding_box(0.0f, 0.0f, box)) {
            boxes.push_back(box);
            umn = Vec3::min(umn, box.min);
            umx = Vec3::max(umx, box.max);
        }
    }
    const bool any = !boxes.empty();
    Vec3 mn = umn, mx = umx;
    if (any) {
        const auto axis = [](const Vec3& v, int k) { return k == 0 ? v.x : (k == 1 ? v.y : v.z); };
        const auto set = [](Vec3& v, int k, float value) {
            (k == 0 ? v.x : (k == 1 ? v.y : v.z)) = value;
        };
        const auto planeAlong = [&](const AABB& b, int k) {
            const Vec3 size = b.max - b.min;
            return axis(size, k) < 0.3f && axis(size, (k + 1) % 3) > 1.0f &&
                   axis(size, (k + 2) % 3) > 1.0f;
        };
        const Vec3 centre = (umn + umx) * 0.5f;
        for (int k = 0; k < 3; ++k) {
            bool low_wall = false, high_wall = false;
            float lo = axis(umn, k), hi = axis(umx, k);
            float side_lo = std::numeric_limits<float>::max();
            float side_hi = -std::numeric_limits<float>::max();
            for (const auto& b : boxes) {
                if (planeAlong(b, k)) {
                    const float mid = 0.5f * (axis(b.min, k) + axis(b.max, k));
                    if (mid < axis(centre, k)) {
                        lo = low_wall ? std::max(lo, axis(b.max, k)) : axis(b.max, k);
                        low_wall = true;
                    } else {
                        hi = high_wall ? std::min(hi, axis(b.min, k)) : axis(b.min, k);
                        high_wall = true;
                    }
                } else if (k == 1 ? (planeAlong(b, 0) || planeAlong(b, 2))
                                  : planeAlong(b, 2 - k)) {
                    // A VERTICAL wall perpendicular to the other horizontal axis
                    // (for y: any vertical wall): its extent along k. Floor and
                    // ceiling slabs never bound a horizontal open side.
                    side_lo = std::min(side_lo, axis(b.min, k));
                    side_hi = std::max(side_hi, axis(b.max, k));
                }
            }
            if (!low_wall && side_lo < side_hi) lo = std::max(lo, side_lo);
            if (!high_wall && side_lo < side_hi) hi = std::min(hi, side_hi);
            set(mn, k, lo);
            set(mx, k, hi);
        }
    }
    const Vec3 lo(mn.x + 0.9f, mn.y + 0.8f, mn.z + 0.9f);
    const Vec3 hi(mx.x - 0.9f, mx.y - 1.0f, mx.z - 0.9f);
    if (!any || hi.x - lo.x < 0.5f || hi.y - lo.y < 0.5f || hi.z - lo.z < 0.5f) {
        state.spawn_min = Vec3(-3.0f, 0.8f, -3.0f);
        state.spawn_max = Vec3(3.0f, 2.8f, 2.8f);
        return;
    }
    state.spawn_min = lo;
    state.spawn_max = hi;
}

// Move only, the gizmo-drag path (scene_ui_gizmos.cpp
// updateGizmoObjectTransformOnActiveBackends): the object's transform plus an
// instance-transform push to the viewport and render backends. No per-vertex
// CPU rebake and no command/log per call - this runs 30 times a second per
// balloon. The CPU vertices catch up at the next respawn (updateMeshVisual rebakes);
// nothing CPU-side reads a drifting balloon's vertices (pick uses state).
void moveMeshVisual(SceneData& scene, size_t index, const Balloon& balloon) {
    if (!state.config.contains("visual_nodes") || index >= state.visual_base.size()) {
        return;
    }
    const auto name = state.config["visual_nodes"][index].get<std::string>();
    auto* mesh = scene.getFlatNodeMesh(name);
    auto* context = rtapi::boundContext();
    if (!mesh || !mesh->transform || !context || &context->scene != &scene) {
        return;
    }
    Matrix4x4 matrix = state.visual_base[index];
    const auto birth = Paint::balloonBirthScale(state.seconds - balloon.birth_started_at,
                                               state.config.value("birth_seconds", 0.75));
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            matrix.m[r][c] *= balloon.size * (c == 1 ? birth.vertical : birth.radial);
        }
    }
    matrix.m[0][3] = balloon.position.x;
    matrix.m[1][3] = balloon.position.y;
    matrix.m[2][3] = balloon.position.z;
    mesh->transform->setBase(matrix);
    pushBalloonTransform(*context, name, matrix);
}

// Floating: each balloon bobs (float_amplitude, 3..5 s) and wanders slowly
// (drift_speed, scaled down for big balloons), steering away from the spawn
// volume's walls and from other balloons. Mesh balloons only (a particle
// balloon's visual is a fixed CPU particle).
void tickMotion(SceneData& scene, float dt) {
    const float amplitude = state.config["float_amplitude"].get<float>();
    const float speed = state.config["drift_speed"].get<float>();
    const bool birthing = std::any_of(state.balloons.begin(), state.balloons.end(),
        [](const Balloon& balloon) { return balloon.inflated && !balloon.birth_settled; });
    if (!state.config.contains("visual_nodes") ||
        (amplitude <= 0.0f && speed <= 0.0f && !birthing)) {
        return;
    }
    const float step = std::min(dt, 0.1f);
    for (size_t i = 0; i < state.balloons.size(); ++i) {
        Balloon& b = state.balloons[i];
        if (!b.inflated || balloonIsHeld(static_cast<int>(i))) continue;
        if (speed > 0.0f) {
            const float max_speed = speed / std::sqrt(std::max(0.3f, b.size));
            // Wander: a small random push, then steer.
            Vec3 accel(randomRange(-1.0f, 1.0f), randomRange(-0.5f, 0.5f), randomRange(-1.0f, 1.0f));
            accel = accel * (0.6f * max_speed);
            const float margin = 0.4f * b.size;
            const auto away = [&](float v, float lo, float hi) {
                if (v < lo + margin) return (lo + margin - v) * 2.0f;
                if (v > hi - margin) return (hi - margin - v) * 2.0f;
                return 0.0f;
            };
            accel = accel + Vec3(away(b.anchor.x, state.spawn_min.x, state.spawn_max.x),
                                 away(b.anchor.y, state.spawn_min.y, state.spawn_max.y),
                                 away(b.anchor.z, state.spawn_min.z, state.spawn_max.z));
            for (size_t j = 0; j < state.balloons.size(); ++j) {
                if (j == i || !state.balloons[j].inflated) continue;
                const Vec3 d = b.anchor - state.balloons[j].anchor;
                const float reach = 0.4f * (b.size + state.balloons[j].size) + 0.25f;
                const float len = d.length();
                if (len > 1e-4f && len < reach) {
                    accel = accel + d * ((reach - len) / len * 3.0f);
                }
            }
            b.drift = b.drift + accel * step;
            const float v = b.drift.length();
            if (v > max_speed) b.drift = b.drift * (max_speed / v);
            b.anchor = b.anchor + b.drift * step;
        }
        b.phase += step * (6.28318530718f / (3.0f + b.size * 1.5f));
        b.position = b.anchor + Vec3(0.0f, amplitude * std::sin(b.phase), 0.0f);
    }
    state.motion_clock += step;
    if (state.motion_clock < 1.0 / 30.0) {
        return;
    }
    state.motion_clock = 0.0;
    const auto start = Clock::now();
    for (size_t i = 0; i < state.balloons.size(); ++i) {
        auto& balloon = state.balloons[i];
        if (balloon.inflated) {
            moveMeshVisual(scene, i, balloon);
            if (state.seconds - balloon.birth_started_at >=
                state.config.value("birth_seconds", 0.75)) {
                balloon.birth_settled = true;
            }
        }
    }
    state.motion_ms_max = std::max(state.motion_ms_max, elapsedMs(start));
    ++state.motion_pushes;
}

void inflate(SceneData& scene, Balloon& balloon) {
    balloon.size = randomRange(state.config["size_min"].get<float>(),
                               state.config["size_max"].get<float>());
    // Inside the room's spawn volume, away from one another.
    for (int attempt = 0; attempt < 100; ++attempt) {
        balloon.position = Vec3(randomRange(state.spawn_min.x, state.spawn_max.x),
                                randomRange(state.spawn_min.y, state.spawn_max.y),
                                randomRange(state.spawn_min.z, state.spawn_max.z));
        bool overlaps = false;
        for (const auto& other : state.balloons) {
            if (&other != &balloon && other.inflated &&
                (other.position - balloon.position).length() <
                    0.4f * (other.size + balloon.size) + 0.15f) {
                overlaps = true;
            }
        }
        if (!overlaps) {
            break;
        }
    }
    balloon.anchor = balloon.position;
    balloon.drift = Vec3(0.0f);
    balloon.phase = randomRange(0.0f, 6.28318530718f);
    balloon.color = std::uniform_int_distribution<int>(
        0, static_cast<int>(state.palette.size()) - 1)(state.random);
    balloon.surprise = randomRange(0.0f, 1.0f) < state.config["surprise_chance"].get<float>();
    ParticleSpawnDesc visual;
    visual.position = balloon.position;
    visual.velocity = Vec3(0.0f);
    visual.lifetime_seconds = 3600.0f;
    visual.appearance_profile_id =
        balloon.surprise ? state.gold_visual_profile : state.visual_profiles[balloon.color];
    visual.size_scale = balloon.size;
    balloon.visual_slot = system(scene, kVisualSystem)->spawn(visual);
    balloon.inflated = true;
    balloon.birth_started_at = state.seconds;
    balloon.birth_settled = !state.config.contains("visual_nodes") ||
        state.config.value("birth_seconds", 0.75) <= 0.0;
    const auto index = static_cast<size_t>(&balloon - state.balloons.data());
    updateMeshVisual(scene, index, balloon, true);
    // Apply the tiny first pose before a frame can show the full-sized respawn.
    moveMeshVisual(scene, index, balloon);
}

// Pops exactly these inflated balloons: one event, one droplet burst each.
// Paint scales with the balloon's surface: drops = particles_per_balloon * size^2.
int dropsFor(float size) {
    const int base = state.config["particles_per_balloon"].get<int>();
    return std::max(32, static_cast<int>(std::lround(base * size * size)));
}

// One independent positional one-shot per popped balloon (audio play_at), so
// several pops in one frame each sound whole at their own balloon.
void playPopSound(SceneData& scene, const Vec3& position, float size, bool surprise) {
    // A surprise has its own sound when one is set; otherwise the pop, brighter.
    const bool own = surprise && state.config.contains("surprise_sound");
    const char* key = own ? "surprise_sound" : "pop_sound";
    if (!state.config.contains(key) || !scene.audio) {
        return;
    }
    const float brighter = surprise && !own ? 1.25f : 1.0f;
    try {
        const json result = scene.audio->invoke("play_at", {
            {"id", state.config[key]},
            {"position", {position.x, position.y, position.z}},
            // Slight per-pop pitch spread: identical pops sound like one machine.
            // Bigger balloon, lower pop (pitch ~ size^-0.7), plus a slight
            // per-pop spread: identical pops sound like one machine.
            {"pitch", std::clamp(std::pow(size, -0.7f) * brighter * randomRange(0.92f, 1.08f),
                                 0.25f, 4.0f)}});
        if (result.value("played", false)) {
            ++state.pop_sounds_played;
        } else {
            ++state.pop_sounds_dropped;
            state.pop_sound_error = result.value("reason", std::string("not played"));
        }
    } catch (const std::exception& e) {
        ++state.pop_sounds_dropped;
        state.pop_sound_error = e.what();
    }
}

void playCelebrateSound(SceneData& scene, int level) {
    if (!state.config.contains("celebrate_sound") || !scene.audio) {
        return;
    }
    try {
        // Room centre; each milestone a little higher, the finale fullest.
        scene.audio->invoke("play_at", {{"id", state.config["celebrate_sound"]},
                                        {"position", {0.0f, 1.6f, 0.0f}},
                                        {"pitch", 0.9f + 0.1f * static_cast<float>(level)},
                                        {"gain", level >= 4 ? 1.0f : 0.8f}});
    } catch (const std::exception& e) {
        state.pop_sound_error = e.what();
    }
}

// Milestone show: level 1..3 = that many confetti bursts in the air; level 4
// (goal reached) = confetti rain, four bursts and every balloon popping in a
// cascade, then (finale_reset_seconds) a clean room for the next children.
// No flashes: the show is paper, paint, sound and slowly fading sparkles (their
// emission only decays - nothing strobes, nothing lights the whole screen).
void celebrate(SceneData& scene, int level) {
    ++state.shows;
    const int bursts = std::min(level, 4);
    for (int i = 0; i < bursts; ++i) {
        const Vec3 centre(randomRange(-2.5f, 2.5f), randomRange(1.6f, 2.6f),
                          randomRange(-2.5f, 2.0f));
        balloonConfettiBurst(scene, centre, 0.2f, kConfettiPerSurprise, 2.0f, 4.5f);
        balloonSparkles(scene, centre, 0.15f);
    }
    playCelebrateSound(scene, level);
    state.events.push_back({{"id", ++state.event_serial}, {"time_s", state.seconds},
                            {"type", "milestone"}, {"level", level},
                            {"room_progress", state.room_progress}, {"frames", 0},
                            {"simulation_ms_max", 0.0}, {"paint_ms_max", 0.0},
                            {"frame_interval_ms_max", 0.0},
                            {"sources_dropped_start", wetPaintGpu().stats().sources_dropped},
                            {"sources_dropped", 0}});
    if (level >= 4) {
        balloonConfettiRain(scene, state.spawn_min, state.spawn_max, kConfettiPerSurprise * 3);
        state.cascade.clear();
        for (size_t i = 0; i < state.balloons.size(); ++i) {
            if (state.balloons[i].inflated) state.cascade.push_back(i);
        }
        std::shuffle(state.cascade.begin(), state.cascade.end(), state.random);
        state.next_cascade = state.seconds + 0.4;
        const double reset_after = state.config["finale_reset_seconds"].get<double>();
        state.finale_reset_at = reset_after > 0.0 ? state.seconds + reset_after : -1.0;
    }
}

void popBalloons(SceneData& scene, const std::vector<size_t>& available) {
    validateMeshVisuals(scene, state.config);
    const int count = static_cast<int>(available.size());
    if (count == 0) {
        return;
    }
    const auto start = Clock::now();
    json popped = json::array();
    int spawned = 0;
    for (int b = 0; b < count; ++b) {
        Balloon& balloon = state.balloons[available[b]];
        if (balloonIsHeld(static_cast<int>(available[b]))) {
            cancelBalloonInteraction();
        }
        // Colours share the four droplet systems; a system carries several profiles.
        auto* runtime = system(scene, kSystems[balloon.color % kSystems.size()]);
        if (balloon.surprise) {
            ++state.surprise_pops;
            balloonConfettiBurst(scene, balloon.position, 0.3f * balloon.size, kConfettiPerSurprise,
                                 1.5f, 3.5f);
            balloonSparkles(scene, balloon.position, 0.25f * balloon.size);
        }
        updateMeshVisual(scene, available[b], balloon, false);
        system(scene, kVisualSystem)->kill(balloon.visual_slot);
        balloon.inflated = false;
        balloon.respawn_at = state.seconds + state.config.value("respawn_seconds", 3.5);
        // A surprise throws half as much paint again, in every palette colour.
        const int particles = dropsFor(balloon.size) * (balloon.surprise ? 3 : 2) / 2;
        int emitted = 0;
        BalloonBurstShape shape;
        shape.clumpiness = state.config["burst_clumpiness"].get<float>();
        shape.jets_max = state.config["burst_jets_max"].get<int>();
        std::vector<BalloonBurstDrop> pattern;
        const BalloonBurstSummary burst =
            balloonBurstPattern(state.random, particles, balloon.size, shape, pattern);
        // A rainbow surprise colours a whole clump alike: it lands as one splat.
        const HiddenPaintSpawnGuard spawn_guard(scene);
        const bool origin_allowed = spawn_guard.allows(balloon.position);
        int color = balloon.color;
        for (size_t i = 0; i < pattern.size(); ++i) {
            const BalloonBurstDrop& p = pattern[i];
            ParticleSpawnDesc drop;
            drop.position = balloon.position + p.offset;
            if (!origin_allowed || !spawn_guard.allows(drop.position)) {
                continue;
            }
            drop.velocity = p.velocity;
            drop.lifetime_seconds = 2.8f;
            drop.mass = 0.001f;
            if (balloon.surprise && (i == 0 || p.clump != pattern[i - 1].clump)) {
                color = std::uniform_int_distribution<int>(
                    0, static_cast<int>(state.palette.size()) - 1)(state.random);
            }
            drop.appearance_profile_id = state.drop_profiles[color];
            drop.size_scale = p.size_scale;
            // A profile id belongs to one system: a rainbow drop goes to the
            // system that holds its colour's profile.
            auto* target = color == balloon.color ? runtime
                                                  : system(scene, kSystems[color % kSystems.size()]);
            target->spawn(drop);
            ++emitted;
            ++spawned;
            balloonDropLaunched(scene, drop.position, drop.velocity, color, drop.size_scale,
                                drop.lifetime_seconds);
        }
        playPopSound(scene, balloon.position, balloon.size, balloon.surprise);
        popped.push_back({{"balloon", available[b]}, {"color", balloon.color},
                          {"size", balloon.size}, {"drops", emitted}, {"surprise", balloon.surprise},
                          {"position", {balloon.position.x, balloon.position.y, balloon.position.z}},
                          {"burst", {{"jets", burst.jets}, {"blobs", burst.blobs},
                                     {"clumps", burst.clumps}, {"jet_drops", burst.jet_drops},
                                     {"blob_drops", burst.blob_drops},
                                     {"spray_drops", burst.spray_drops},
                                     {"heaviest_jet_share", burst.heaviest_jet_share},
                                     {"tear", {burst.tear.x, burst.tear.y, burst.tear.z}}}}});
    }
    state.events.push_back({{"id", ++state.event_serial}, {"time_s", state.seconds},
                            {"simultaneous", count}, {"spawned", spawned},
                            {"spawn_ms", elapsedMs(start)}, {"frames", 0},
                            {"simulation_ms_max", 0.0}, {"paint_ms_max", 0.0},
                            {"frame_interval_ms_max", 0.0},
                            {"sources_dropped_start", wetPaintGpu().stats().sources_dropped},
                            {"sources_dropped", 0}, {"balloons", popped}});
    if (state.events.size() > 128) {
        state.events.erase(state.events.begin());
    }
}

// Clean room: wet paint cleared, every popped balloon back. Pools are kept
// (no re-warm); drops still in flight land within their 2.8 s lifetime, so the
// paint is cleared once more 3 s later.
void resetRoom(SceneData& scene) {
    std::string placement_error;
    clearBalloonPaintInFlight(scene);
    resetHiddenGameObjects(scene, placement_error);
    cancelBalloonInteraction();
    resetBalloonBodyInteraction();
    auto& paint = wetPaintGpu();
    if (!paint.empty()) {
        paint.resetPaint();
    }
    for (auto& balloon : state.balloons) {
        if (!balloon.inflated) {
            inflate(scene, balloon);
        }
    }
    state.dirty = false;
    state.last_activity = state.seconds;
    state.clean_again_at = state.seconds + 3.0;
    ++state.resets;
    // A new round: the room goal starts again.
    state.room_progress = 0.0;
    state.milestones = 0;
    state.cascade.clear();
    state.finale_reset_at = -1.0;
}

void burst(SceneData& scene, int requested) {
    std::vector<size_t> available;
    for (size_t i = 0; i < state.balloons.size(); ++i) {
        if (state.balloons[i].inflated && !balloonIsHeld(static_cast<int>(i))) {
            available.push_back(i);
        }
    }
    std::shuffle(available.begin(), available.end(), state.random);
    available.resize(std::min<size_t>(static_cast<size_t>(requested), available.size()));
    popBalloons(scene, available);
}

// Nearest inflated balloon along the camera ray through a window point
// (x, y in 0..1 from the top-left corner), or -1. The same pick serves the
// Player's mouse click and paint.wet_demo pop over IPC.
std::vector<BalloonInteractionTarget> interactionTargets() {
    std::vector<BalloonInteractionTarget> targets;
    for (size_t i = 0; i < state.balloons.size(); ++i) {
        const auto& balloon = state.balloons[i];
        if (!balloon.inflated) {
            continue;
        }
        if (state.config.contains("visual_nodes")) {
            const auto name = state.config["visual_nodes"][i].get<std::string>();
            const auto* mesh = state.scene ? state.scene->getFlatNodeMesh(name) : nullptr;
            if (!mesh || !mesh->visible) {
                continue;
            }
        }
        const auto birth = state.config.contains("visual_nodes") ?
            balloonBirthScale(state.seconds - balloon.birth_started_at,
                               state.config.value("birth_seconds", 0.75)) : BalloonBirthScale{};
        targets.push_back({static_cast<int>(i), balloon.position,
            kPickRadius * balloon.size * std::max(birth.radial, birth.vertical),
            kPickRadius * balloon.size * 1.15f});
    }
    return targets;
}

int pickBalloon(SceneData& scene, float x, float y) {
    return pickBalloonTarget(scene, x, y, interactionTargets());
}

json status() {
    json balloons = json::array();
    for (size_t i = 0; i < state.balloons.size(); ++i) {
        const auto& b = state.balloons[i];
        json entry = {{"id", i}, {"inflated", b.inflated}, {"color", b.color}, {"size", b.size},
                      {"surprise", b.surprise},
                      {"position", {b.position.x, b.position.y, b.position.z}}};
        const auto birth = state.config.contains("visual_nodes") && b.inflated ?
            Paint::balloonBirthScale(state.seconds - b.birth_started_at,
                                     state.config.value("birth_seconds", 0.75)) :
            Paint::BalloonBirthScale{};
        entry["held"] = balloonIsHeld(static_cast<int>(i));
        entry["birth_active"] = b.inflated && !b.birth_settled;
        entry["visual_scale"] = {birth.radial, birth.vertical, birth.radial};
        if (state.config.contains("visual_nodes")) {
            const auto name = state.config["visual_nodes"][i].get<std::string>();
            auto* mesh = state.scene ? state.scene->getFlatNodeMesh(name) : nullptr;
            entry["visual_node"] = name;
            entry["visual_resolved"] = mesh != nullptr;
            // Shown = an inflated balloon on a visible mesh; a popped one is
            // shrunk to 1e-4 in place (see updateMeshVisual), not hidden.
            entry["visual_visible"] = mesh && mesh->visible && b.inflated;
        }
        balloons.push_back(std::move(entry));
    }
    json pools = json::array();
    if (state.prepared && state.scene) {
        for (const char* name : kSystems) {
            auto* runtime = system(*state.scene, name);
            if (runtime) {
                pools.push_back({{"system", name}, {"capacity", runtime->capacity()},
                                 {"resident_capacity", runtime->residentCapacity()},
                                 {"alive", runtime->aliveCount()},
                                 {"gpu_status", runtime->stats().gpu_status}});
            }
        }
    }
    return {{"prepared", state.prepared}, {"running", state.running},
            {"error", state.error}, {"time_s", state.seconds},
            {"prepare_ms", state.prepare_ms}, {"atlas_ms", state.atlas_ms},
            {"events_total", state.event_serial}, {"observed_frames", state.observed_frames},
            {"simulation_ms_max", state.max_simulation_ms},
            {"frame_interval_ms_max", state.max_frame_interval_ms},
            {"visual_mode", state.config.contains("visual_nodes") ? "flat_mesh" : "particle"},
            {"resets", state.resets}, {"dirty", state.dirty}, {"surprise_pops", state.surprise_pops},
            {"room_progress", state.room_progress}, {"milestones", state.milestones},
            {"shows", state.shows}, {"finale_cascade_left", state.cascade.size()},
            {"spawn_min", {state.spawn_min.x, state.spawn_min.y, state.spawn_min.z}},
            {"spawn_max", {state.spawn_max.x, state.spawn_max.y, state.spawn_max.z}},
            {"spawn_auto", state.spawn_auto}, {"motion_ms_max", state.motion_ms_max},
            {"motion_pushes", state.motion_pushes},
            {"palette", [] {
                json colors = json::array();
                for (const auto& c : state.palette) colors.push_back({c.x, c.y, c.z});
                return colors;
            }()},
            {"idle_seconds", state.seconds - state.last_activity},
            {"pop_sounds_played", state.pop_sounds_played},
            {"pop_sounds_dropped", state.pop_sounds_dropped},
            {"pop_sound_error", state.pop_sound_error},
            {"effects", balloonEffectsStatus(state.prepared ? state.scene : nullptr)},
            {"interaction", balloonInteractionStatus()},
            {"body_interaction", balloonBodyInteractionStatus(
                state.config.is_object() && state.config.value("body_controls", false))},
            {"config", state.config}, {"balloons", balloons}, {"pools", pools},
            {"events", state.events}};
}

void prepare(SceneData& scene, const json& config) {
    const auto start = Clock::now();
    std::map<std::string, float> previous_gold_transmission;
    if (state.scene == &scene && state.config.contains("visual_nodes")) {
        const auto& nodes = state.config["visual_nodes"];
        for (size_t i = 0; i < state.balloons.size() &&
             i < state.visual_transmission.size() && i < nodes.size(); ++i) {
            if (state.balloons[i].surprise) {
                previous_gold_transmission[nodes[i].get<std::string>()] =
                    state.visual_transmission[i];
            }
        }
    }
    hideMeshVisuals();
    cancelBalloonInteraction();
    resetBalloonBodyInteraction();
    state = State{};
    state.scene = &scene;
    state.config = config;
    state.random.seed(config["seed"].get<uint32_t>());
    // Authored droplet systems hold analytic colliders. No domain/SDF is created.
    for (size_t i = 0; i < kSystems.size(); ++i) {
        auto* runtime = system(scene, kSystems[i]);
        runtime->clear();
        runtime->clearEmitters();
        RayTrophiSim::ParticleEmitterDesc keep_context;
        keep_context.name = "PreallocatedBurstPool";
        keep_context.rate_per_second = 0.0f;
        keep_context.burst_count = 0;
        runtime->addEmitter(keep_context);
        runtime->reserve(config["balloons"].get<size_t>() *
                         static_cast<size_t>(dropsFor(config["size_max"].get<float>())) + 1);
        runtime->setGravity(Vec3(0.0f, -9.81f, 0.0f));
        runtime->setLinearDrag(0.15f);
        auto& physics = runtime->physicsSettings();
        physics.execution_policy = RayTrophiSim::ParticleExecutionPolicy::GPURequired;
        physics.record_collision_events = true;
        physics.collision_response = RayTrophiSim::ParticleCollisionResponse::Die;
    }
    state.palette.clear();
    if (config.contains("colors")) {
        for (const auto& c : config["colors"]) {
            state.palette.emplace_back(c[0].get<float>(), c[1].get<float>(), c[2].get<float>());
        }
    } else {
        state.palette.assign(kDefaultColors.begin(), kDefaultColors.end());
    }
    state.drop_profiles.clear();
    for (size_t c = 0; c < state.palette.size(); ++c) {
        auto* runtime = system(scene, kSystems[c % kSystems.size()]);
        // Named by the colour itself: profiles are found by name, so a changed
        // palette must never pick up an old profile of the same index.
        state.drop_profiles.push_back(runtime->findOrAddAppearanceProfile(
            profile("Paint Drops " + colorKey(state.palette[c]), state.palette[c], 0.065f)));
    }
    if (!system(scene, kVisualSystem)) {
        scene.addParticleSystemObject(kVisualSystem);
    }
    auto* visuals = system(scene, kVisualSystem);
    visuals->clear();
    visuals->clearEmitters();
    visuals->setGravity(Vec3(0.0f));
    visuals->physicsSettings().execution_policy = RayTrophiSim::ParticleExecutionPolicy::CPU;
    visuals->physicsSettings().record_collision_events = false;
    visuals->reserve(config["balloons"].get<size_t>());
    for (auto& entry : scene.particle_systems) {
        if (entry.name == kVisualSystem ||
            std::find(kSystems.begin(), kSystems.end(), entry.name) != kSystems.end()) {
            entry.enabled = true;
            entry.visible = true;
            entry.render.emitter_only = false;
            if (entry.name == kVisualSystem && config.contains("visual_nodes")) {
                // Retain the canonical 12 anchors/pop slots; render only the
                // authored flat meshes, never a second billboard on top.
                entry.render.emitter_only = true;
            }
            entry.render.shape = SceneData::ParticleRenderShape::Sphere;
            entry.render.emissive = false;
            // Non-emissive spheres are drawn as LIT billboards (roughness =
            // the GGX highlight): paint drops wet and glossy like the paint on
            // the walls, particle balloons satin.
            entry.render.roughness = entry.name == kVisualSystem ? 0.3f : 0.12f;
            SceneData::applyParticleSystemEnabledState(entry);
        }
    }
    state.visual_profiles.clear();
    for (size_t c = 0; c < state.palette.size(); ++c) {
        state.visual_profiles.push_back(visuals->findOrAddAppearanceProfile(
            profile("Balloon " + colorKey(state.palette[c]), state.palette[c], 0.65f)));
    }
    state.gold_visual_profile = visuals->findOrAddAppearanceProfile(
        profile("Balloon Surprise Gold", kGold, 0.65f));

    {
        BalloonEffectsSetup fx;
        fx.palette = state.palette;
        fx.gold = kGold;
        fx.droplet_systems.assign(kSystems.begin(), kSystems.end());
        fx.seed = config["seed"].get<uint32_t>();
        fx.sparkles = config["sparkles"].get<int>();
        fx.sparkle_trails = config["sparkle_trails"].get<bool>();
        fx.drop_sparks = config["drop_sparks"].get<float>();
        fx.drop_trails = config["drop_trails"].get<bool>();
        prepareBalloonEffects(scene, fx);
    }

    // Build all four CPU atlases in this one main-thread operation. No frame can
    // rebuild the partial 1-, 2- or 3-surface layout between additions.
    const auto atlas_start = Clock::now();
    auto& paint = wetPaintGpu();
    // Without "surfaces" the scene's game design owns the surfaces, their look
    // and the splat settings: paint them as they are.
    const bool builds_surfaces = config.contains("surfaces");
    if (builds_surfaces) {
        paint.clear();
        std::string error;
        for (const auto& object : config["surfaces"]) {
            if (!paint.addSurface(scene, object.get<std::string>(), -1,
                                  config["resolution"].get<int>(), error)) {
                throw std::runtime_error(error);
            }
        }
        // The room's look: paint that stays open and runs down the walls, glossy
        // and slightly raised while it does. The generic surface defaults dry a
        // splat in a few seconds (absorption 0.18 removes wetness on top of the
        // lifetime), before any runoff shows. Set before the first coverage()
        // so the channel textures are bound here, at load, not during play.
        WetPaintSurfaceSettings look;
        look.wetness = 0.95f;
        look.lifetime_seconds = 12.0f;
        look.diffusion = 0.1f;
        look.runoff = 0.7f;
        look.absorption = 0.06f;
        look.drip_head = 0.8f;
        look.paint_load = 0.95f;
        look.deposit_rate = 0.9f;
        look.pickup_rate = 0.2f;
        look.brush_radius = 6.0f;
        look.roughness_channel = true;
        look.height_normal_channel = true;
        for (const auto& object : config["surfaces"]) {
            paint.setSurfaceSettings(object.get<std::string>(), -1, look);
        }
        WetPaintSplatSettings splat;
        splat.radius = 0.09f;
        splat.amount = 0.9f;
        splat.max_events_per_tick = 65536;
        paint.setSplatSettings(splat);
    }
    WetPaintCoverage coverage;
    if (!paint.coverage(scene, coverage)) {
        throw std::runtime_error("could not prepare wet paint GPU atlas");
    }
    state.atlas_ms = elapsedMs(atlas_start);
    g_materials_dirty = true;

    // Warm the resident collision/spawn/ballistic buffers at full pool capacity.
    // Warm the largest same-colour triple burst, including slot-record capacity.
    // Kill sacrificial drops before gameplay; keep allocation and pipelines.
    RayTrophiSim::SimulationContext context;
    context.dt = 1.0f / 60.0f;
    context.compute = &scene.getSimulationWorld().compute();
    for (const char* name : kSystems) {
        auto* runtime = system(scene, name);
        std::vector<size_t> slots;
        const int warm_count = 3 * dropsFor(config["size_max"].get<float>());
        slots.reserve(warm_count + 1);
        for (int i = 0; i <= warm_count; ++i) {
            ParticleSpawnDesc warm;
            warm.position = Vec3(0.0f, 2.0f, 0.0f);
            // The last warm slot is killed after the final step, so its device
            // copy can outlive it until the first real spawn uploads slots
            // again; at default size that was a white sphere mid-room.
            warm.size_scale = 1.0e-3f;
            slots.push_back(runtime->spawn(warm));
        }
        runtime->step(context);
        if (!runtime->stats().device_resident) {
            throw std::runtime_error(std::string("GPU burst pool warmup failed: ") + name +
                                     " (" + runtime->stats().gpu_status + ")");
        }
        for (int i = 0; i < warm_count; ++i) {
            runtime->kill(slots[i]);
        }
        runtime->step(context);
        runtime->kill(slots.back());
    }
    context.compute->synchronize();
    resolveSpawnVolume(scene);
    state.balloons.resize(config["balloons"].get<size_t>());
    state.visual_base.clear();
    if (config.contains("visual_nodes")) {
        for (const auto& node : config["visual_nodes"]) {
            const std::string name = node.get<std::string>();
            auto* mesh = scene.getFlatNodeMesh(name);
            Matrix4x4 base = mesh && mesh->transform ? mesh->transform->base : Matrix4x4();
            // Keep the rotation, drop the scale: a balloon's size is its mesh
            // geometry times `size`. A project saved after play holds the last
            // respawn's scale in this transform; reading it back as the authored
            // scale would compound sizes on every save/load.
            for (int c = 0; c < 3; ++c) {
                const float length = std::sqrt(base.m[0][c] * base.m[0][c] +
                                               base.m[1][c] * base.m[1][c] +
                                               base.m[2][c] * base.m[2][c]);
                if (length > 1e-6f) {
                    for (int r = 0; r < 3; ++r) base.m[r][c] /= length;
                }
            }
            state.visual_base.push_back(base);
            rtapi::MaterialParamValue transmission;
            const auto read = rtapi::getMaterialParam(name, "transmission", transmission);
            if (!read.ok) {
                throw std::runtime_error(read.error);
            }
            const auto previous = previous_gold_transmission.find(name);
            state.visual_transmission.push_back(previous != previous_gold_transmission.end()
                ? previous->second : transmission.scalar);
        }
    }
    for (auto& balloon : state.balloons) {
        inflate(scene, balloon);
    }
    state.next_event = randomRange(config["interval_min"].get<float>(),
                                   config["interval_max"].get<float>());
    state.prepared = true;
    state.running = config["autostart"].get<bool>();
    // Adopt the authored pools/profiles before the first live frame. Otherwise
    // the generic setup-signature invalidation can rewind away our balloons.
    scene.preserveScriptSimulationPreview();
    state.prepare_ms = elapsedMs(start);
}

} // namespace

bool balloonPaintCommand(SceneData& scene, const std::string& operation,
                         const json& input, json& output, std::string& error) {
    try {
        if (operation == "hidden") {
            return hiddenGameObjectsCommand(scene, input, output, error);
        }
        int popped_balloon = -1;
        if (operation == "body") {
            if ((!state.prepared || state.scene != &scene) && !input.empty()) {
                error = "configure the demo before enabling body controls";
                return false;
            }
            return balloonBodyControlCommand(state.config, input, output, error);
        }
        if (operation == "pointer") {
            const bool cancel = input.is_object() && input.value("action", json()) == "cancel";
            if ((!state.prepared || state.scene != &scene) && !cancel) {
                throw std::runtime_error("configure the demo before pointer input");
            }
            if (!cancel) {
                for (const char* name : kSystems) {
                    if (!system(scene, name)) {
                        throw std::runtime_error("demo droplet system was removed");
                    }
                }
                if (!system(scene, kVisualSystem)) {
                    throw std::runtime_error("demo visual system was removed");
                }
            }
            const auto config = state.config.is_object() ? state.config : json::object();
            const auto action = balloonPointerCommand(scene, input,
                config.value("interaction_mode", std::string("pop")), interactionTargets(),
                config.value("visual_nodes", std::vector<std::string>{}));
            if (action.move >= 0) {
                auto& balloon = state.balloons[static_cast<size_t>(action.move)];
                balloon.position = balloon.anchor = action.position;
                balloon.drift = Vec3(0.0f);
                balloon.phase = 0.0f;
                moveMeshVisual(scene, static_cast<size_t>(action.move), balloon);
            }
            if (action.pop >= 0) {
                popBalloons(scene, {static_cast<size_t>(action.pop)});
                state.dirty = true;
            }
            if (action.move >= 0 || action.pop >= 0 || heldBalloon() >= 0) {
                state.last_activity = state.seconds;
            }
            output = balloonInteractionStatus();
            output["moved"] = action.move;
            output["popped"] = action.pop;
            return true;
        }
        if (operation == "interaction") {
            if (!state.prepared || state.scene != &scene || !input.is_object() ||
                input.size() != 1 || !input.contains("mode") || !input["mode"].is_string() ||
                (input["mode"] != "pop" && input["mode"] != "carry")) {
                throw std::runtime_error("prepared demo and interaction mode pop|carry required");
            }
            if (input["mode"] == "carry" && !state.config.contains("visual_nodes")) {
                throw std::runtime_error("carry requires mesh balloon visuals");
            }
            cancelBalloonInteraction();
            state.config["interaction_mode"] = input["mode"];
            output = status();
            return true;
        }
        if (!input.is_object()) {
            throw std::runtime_error("demo input must be an object");
        }
        if (operation != "configure" && operation != "burst" && operation != "pop" &&
            operation != "set_collider" && !input.empty()) {
            throw std::runtime_error("this demo operation takes an empty config");
        }
        if (operation == "configure") {
            const json config = validatedConfig(input, scene);
            prepare(scene, config);
        } else if (operation == "set_collider") {
            // Works before configure: colliders are part of the scene setup.
            if (!input.contains("object") || !input["object"].is_string() ||
                !input.contains("mode") || !input["mode"].is_string() ||
                (input.contains("sdf_resolution_mode") &&
                 !input["sdf_resolution_mode"].is_number_integer())) {
                throw std::runtime_error("set_collider takes object, mode "
                                         "(none|aabb|obb|mesh_sdf) and optional integer "
                                         "sdf_resolution_mode 0..2");
            }
            std::string collider_error;
            if (!setBalloonPaintCollider(scene, input["object"].get<std::string>(),
                                         input["mode"].get<std::string>(),
                                         static_cast<int>(input.value("sdf_resolution_mode", int64_t{1})),
                                         collider_error)) {
                throw std::runtime_error(collider_error);
            }
        } else if (operation == "status") {
            // Read-only, including before a demo is configured.
        } else if (operation == "reset") {
            if (!state.prepared || state.scene != &scene) {
                // No demo: still a clean room for whatever paint exists.
                clearBalloonPaintInFlight(scene);
                resetHiddenGameObjects(scene, error);
                if (!wetPaintGpu().empty()) wetPaintGpu().resetPaint();
            } else {
                resetRoom(scene);
            }
        } else if (operation == "reset_balloons") {
            if (!state.prepared || state.scene != &scene) {
                throw std::runtime_error("configure the demo before renewing balloons");
            }
            state.cascade.clear();
            state.finale_reset_at = -1.0;
            for (auto& balloon : state.balloons) {
                if (!balloon.inflated) {
                    inflate(scene, balloon);
                }
            }
        } else if (operation == "stop") {
            state.running = false;
        } else if (operation == "start") {
            if (!state.prepared || state.scene != &scene) {
                throw std::runtime_error("configure the demo before starting");
            }
            state.running = true;
            state.next_event = state.seconds + randomRange(
                state.config["interval_min"].get<float>(),
                state.config["interval_max"].get<float>());
        } else if (operation == "burst") {
            if (input.size() > 1 || (input.size() == 1 && !input.contains("count")) ||
                (input.contains("count") && !input["count"].is_number_integer())) {
                throw std::runtime_error("burst config only accepts an integer count");
            }
            const int64_t count = input.value("count", int64_t{1});
            if (!state.prepared || state.scene != &scene || count < 1 || count > 3) {
                throw std::runtime_error("prepared demo and burst count in 1..3 required");
            }
            for (const char* name : kSystems) {
                if (!system(scene, name)) {
                    throw std::runtime_error("demo droplet system was removed");
                }
            }
            if (!system(scene, kVisualSystem)) {
                throw std::runtime_error("demo visual system was removed");
            }
            burst(scene, static_cast<int>(count));
            state.last_activity = state.seconds;
            state.dirty = true;
        } else if (operation == "pop") {
            // One chosen balloon: by id, or the one under a window point.
            if (!state.prepared || state.scene != &scene) {
                throw std::runtime_error("configure the demo before popping");
            }
            for (const char* name : kSystems) {
                if (!system(scene, name)) {
                    throw std::runtime_error("demo droplet system was removed");
                }
            }
            if (!system(scene, kVisualSystem)) {
                throw std::runtime_error("demo visual system was removed");
            }
            const bool by_id = input.contains("balloon");
            const bool by_screen = input.contains("screen");
            if (input.size() != 1 || by_id == by_screen) {
                throw std::runtime_error(
                    "pop takes exactly one of balloon (id) or screen ([x, y] in 0..1)");
            }
            if (by_id) {
                if (!input["balloon"].is_number_integer() || input["balloon"].get<int64_t>() < 0 ||
                    input["balloon"].get<int64_t>() >= static_cast<int64_t>(state.balloons.size())) {
                    throw std::runtime_error("balloon id out of range");
                }
                popped_balloon = static_cast<int>(input["balloon"].get<int64_t>());
                if (!state.balloons[static_cast<size_t>(popped_balloon)].inflated) {
                    popped_balloon = -1;   // already popped, waiting to come back
                }
            } else {
                const auto& point = input["screen"];
                if (!point.is_array() || point.size() != 2 || !point[0].is_number() ||
                    !point[1].is_number()) {
                    throw std::runtime_error("screen must be [x, y]");
                }
                const float x = point[0].get<float>();
                const float y = point[1].get<float>();
                if (!(x >= 0.0f && x <= 1.0f && y >= 0.0f && y <= 1.0f)) {
                    throw std::runtime_error("screen x and y must be in 0..1");
                }
                popped_balloon = pickBalloon(scene, x, y);
            }
            if (popped_balloon >= 0) {
                popBalloons(scene, {static_cast<size_t>(popped_balloon)});
                state.last_activity = state.seconds;
                state.dirty = true;
            }
        } else {
            throw std::runtime_error(
                "unknown demo operation; use configure/status/start/stop/burst/pop/reset/set_collider");
        }
        output = status();
        output["colliders"] = balloonPaintColliders(scene);
        if (operation == "pop") {
            // -1: nothing under the point, or that balloon is not inflated.
            output["popped"] = popped_balloon;
        }
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        // Validation failures leave a running demo intact. Preparation failures
        // stop a partially constructed demo instead of continuing with stale handles.
        if (operation == "configure" && !state.prepared) {
            state.running = false;
            state.error = error;
        }
        return false;
    }
}

json balloonPaintColliders(SceneData& scene) {
    // One row per collider name of the first droplet system, with cook state
    // across all four (a collider missing from one system is reported).
    json rows = json::array();
    auto* first = system(scene, kSystems[0]);
    if (!first) return rows;
    for (const auto& c : first->colliders()) {
        bool in_all = true;
        bool cooked = true;
        for (const char* name : kSystems) {
            auto* runtime = system(scene, name);
            const RayTrophiSim::ParticleColliderDesc* match = nullptr;
            if (runtime) {
                for (const auto& other : runtime->colliders()) {
                    if (other.name == c.name) { match = &other; break; }
                }
            }
            in_all = in_all && match;
            cooked = cooked && match && colliderCooked(*match);
        }
        rows.push_back({{"name", c.name}, {"object", c.source_name},
                        {"mode", colliderModeName(c.source_mode)}, {"enabled", c.enabled},
                        {"sdf_resolution_mode", c.sdf_resolution_mode},
                        {"ready", cooked}, {"in_all_systems", in_all},
                        {"managed", c.name == managedColliderName(c.source_name)}});
    }
    return rows;
}

bool balloonPaintCollidersReady(SceneData& scene, std::string& waiting) {
    for (const char* name : kSystems) {
        auto* runtime = system(scene, name);
        if (!runtime) continue;   // configure reports a missing system
        for (const auto& c : runtime->colliders()) {
            if (c.enabled && !colliderCooked(c)) {
                waiting = c.name;
                return false;
            }
        }
    }
    return true;
}

bool setBalloonPaintCollider(SceneData& scene, const std::string& object, const std::string& mode,
                             int sdf_resolution_mode, std::string& error) {
    using M = RayTrophiSim::ParticleColliderSourceMode;
    M source_mode = M::ObjectAABB;
    if (mode == "aabb") source_mode = M::ObjectAABB;
    else if (mode == "obb") source_mode = M::ObjectOBB;
    else if (mode == "mesh_sdf") source_mode = M::ObjectMeshSDF;
    else if (mode != "none") {
        error = "collider mode must be none|aabb|obb|mesh_sdf";
        return false;
    }
    if (mode != "none" && !scene.getFlatNodeMesh(object)) {
        error = "collider object is not a flat mesh in the scene: " + object;
        return false;
    }
    if (sdf_resolution_mode < 0 || sdf_resolution_mode > 2) {
        error = "sdf_resolution_mode must be 0 (32), 1 (64) or 2 (128)";
        return false;
    }
    for (const char* name : kSystems) {
        if (!system(scene, name)) {
            error = std::string("droplet system missing: ") + name;
            return false;
        }
    }
    const std::string collider_name = managedColliderName(object);
    for (const char* name : kSystems) {
        auto runtime = systemShared(scene, name);
        auto& colliders = runtime->colliders();
        for (size_t i = 0; i < colliders.size(); ++i) {
            if (colliders[i].name == collider_name) {
                runtime->removeCollider(i);
                break;
            }
        }
        if (mode == "none") continue;
        RayTrophiSim::ParticleColliderDesc desc;
        desc.name = collider_name;
        desc.source_mode = source_mode;
        desc.source_name = object;
        desc.enabled = true;
        desc.sdf_resolution_mode = sdf_resolution_mode;
        auto& created = runtime->addCollider(desc);
        // Same asynchronous cook the collider panel and sim.collider use; each
        // droplet system holds its own cooked grid.
        scene.rebuildSDFColliderAsync(created, runtime);
    }
    mirrorBalloonEffectColliders(scene);   // no-op until configure prepared the effects
    scene.clearSimFrameCache();
    scene.preserveScriptSimulationPreview();
    return true;
}

json balloonPaintDesign() {
    if (!state.prepared) {
        return nullptr;
    }
    // The saved form never lists surfaces: they are saved with their own
    // settings in the game design's wet_paint section and exist again before
    // the demo is configured on load.
    json design = state.config;
    design.erase("surfaces");
    design.erase("resolution");
    return design;
}

bool loadBalloonPaintProject(SceneData& scene, const std::string& project_path,
                             std::string& error) {
    resetBalloonPaintDemo();
    const std::string sidecar = project_path + ".balloons.json";
    const auto path = std::filesystem::path(std::u8string(sidecar.begin(), sidecar.end()));
    if (!std::filesystem::exists(path)) {
        return true;
    }
    try {
        std::ifstream stream(path);
        const json config = json::parse(stream);
        json output;
        return balloonPaintCommand(scene, "configure", config, output, error);
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

void resetBalloonPaintDemo() {
    // Called after project replacement too: never look old names up in a new scene.
    cancelBalloonInteraction();
    resetBalloonBodyInteraction();
    state = State{};
    resetBalloonEffects();
    wetPaintGpu().clear();
}

void tickBalloonPaintDemo(SceneData& scene, float dt) {
    if (!state.prepared || state.scene != &scene) {
        cancelBalloonInteraction();
        resetBalloonBodyInteraction();
        return;
    }
    if (!std::isfinite(dt) || dt <= 0.0f) {
        return;
    }
    // This tick's own cost, for the per-event breakdown (observeBalloonPaintFrame).
    struct TickTimer {
        Clock::time_point start = Clock::now();
        ~TickTimer() { state.last_tick_ms = elapsedMs(start); }
    } tick_timer;
    for (const char* name : kSystems) {
        if (!system(scene, name)) {
            state.running = false;
            state.error = "demo particle system was removed";
            cancelBalloonInteraction();
            resetBalloonBodyInteraction();
            return;
        }
    }
    if (!system(scene, kVisualSystem)) {
        state.running = false;
        state.error = "demo visual system was removed";
        cancelBalloonInteraction();
        resetBalloonBodyInteraction();
        return;
    }
    try {
        state.seconds += std::min(dt, 0.25f);
        if (state.config.value("body_controls", false)) {
            tickBalloonBodyInteraction(scene, true, interactionTargets());
        }
        const int held = heldBalloon();
        if (held >= 0) {
            state.last_activity = state.seconds;
        }
        for (auto& balloon : state.balloons) {
            if (!balloon.inflated && state.seconds >= balloon.respawn_at) {
                inflate(scene, balloon);
            }
        }
        tickMotion(scene, std::min(dt, 0.25f));
        tickBalloonEffects(scene, std::min(dt, 0.25f));
        if (state.clean_again_at >= 0.0 && state.seconds >= state.clean_again_at) {
            state.clean_again_at = -1.0;
            if (!wetPaintGpu().empty()) wetPaintGpu().resetPaint();
        }
        // Shared goal: coverage twice a second (a GPU reduction + small download,
        // the same measurement the game session takes).
        const double goal = state.config["room_goal"].get<double>();
        if (goal > 0.0 && !wetPaintGpu().empty()) {
            state.coverage_clock += std::min(dt, 0.25f);
            if (state.coverage_clock >= 0.5) {
                state.coverage_clock = 0.0;
                WetPaintCoverage coverage;
                if (wetPaintGpu().coverage(scene, coverage, /*allow_stale=*/true)) {
                    state.room_progress = std::clamp(coverage.ratio / goal, 0.0, 1.0);
                    while (state.milestones < 4 &&
                           state.room_progress >= 0.25 * (state.milestones + 1) - 1e-9) {
                        ++state.milestones;
                        celebrate(scene, state.milestones);
                    }
                }
            }
        }
        if (!state.cascade.empty() && state.seconds >= state.next_cascade) {
            const size_t next = state.cascade.back();
            state.cascade.pop_back();
            if (state.balloons[next].inflated && !balloonIsHeld(static_cast<int>(next))) {
                popBalloons(scene, {next});
            }
            state.next_cascade = state.seconds + 0.18;
        }
        if (held < 0 && state.finale_reset_at >= 0.0 && state.seconds >= state.finale_reset_at) {
            resetRoom(scene);
        }
        const double idle_reset = state.config["idle_reset_seconds"].get<double>();
        if (idle_reset > 0.0 && state.dirty && state.seconds - state.last_activity >= idle_reset) {
            // Nobody has popped a balloon for a while: clean room for the next child.
            resetRoom(scene);
        }
        if (state.running && state.seconds >= state.next_event) {
            const int roll = std::uniform_int_distribution<int>(0, 9)(state.random);
            state.dirty = true;   // attract-mode pops paint too, but are not player activity
            burst(scene, roll < 6 ? 1 : (roll < 9 ? 2 : 3));
            state.next_event = state.seconds + randomRange(
                state.config["interval_min"].get<float>(), state.config["interval_max"].get<float>());
        }
    } catch (const std::exception& error) {
        cancelBalloonInteraction();
        resetBalloonBodyInteraction();
        state.running = false;
        state.prepared = false;
        state.error = error.what();
    }
}

void observeBalloonPaintFrame(double simulation_ms, float frame_interval_seconds) {
    if (!state.prepared) {
        return;
    }
    ++state.observed_frames;
    state.max_simulation_ms = std::max(state.max_simulation_ms, simulation_ms);
    const double interval = static_cast<double>(frame_interval_seconds) * 1000.0;
    state.max_frame_interval_ms = std::max(state.max_frame_interval_ms, interval);
    BalloonEffectsTimings fx_t;
    double droplet_step_ms = 0.0;
    if (state.scene) {
        fx_t = balloonEffectsTimings(*state.scene);
        for (const char* name : kSystems) {
            if (auto* runtime = system(*state.scene, name)) droplet_step_ms += runtime->stats().total_ms;
        }
    }
    for (auto it = state.events.rbegin(); it != state.events.rend(); ++it) {
        auto& event = *it;
        // Windows may overlap: every event records its following three seconds.
        if (state.seconds - event["time_s"].get<double>() > 3.0) {
            break;
        }
        event["frames"] = event["frames"].get<uint64_t>() + 1;
        event["simulation_ms_max"] = std::max(event["simulation_ms_max"].get<double>(),
                                               simulation_ms);
        event["paint_ms_max"] = std::max(event["paint_ms_max"].get<double>(),
                                          wetPaintGpu().stats().last_tick_ms);
        event["frame_interval_ms_max"] = std::max(
            event["frame_interval_ms_max"].get<double>(), interval);
        // Where a slow frame went (2026-11-18: a one-off ~30 ms spike after
        // configure, source unknown). demo_tick = this file's tick incl. the
        // effects (contacts / drop_trails / sparkles); droplet_step / effects_step =
        // the particle systems' own step (stats total_ms); paint = wet paint.
        const auto bump = [&event](const char* key, double value) {
            event[key] = std::max(event.value(key, 0.0), value);
        };
        bump("demo_tick_ms_max", state.last_tick_ms);
        bump("fx_contacts_ms_max", fx_t.contacts_ms);
        bump("fx_drop_trails_ms_max", fx_t.drop_trails_ms);
        bump("fx_sparkles_ms_max", fx_t.sparkles_ms);
        bump("effects_step_ms_max", fx_t.step_ms);
        bump("droplet_step_ms_max", droplet_step_ms);
        const uint64_t dropped = wetPaintGpu().stats().sources_dropped;
        const uint64_t initial = event["sources_dropped_start"].get<uint64_t>();
        event["sources_dropped"] = dropped >= initial ? dropped - initial : 0;
    }
}

} // namespace Paint
