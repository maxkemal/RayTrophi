#include "SceneSerializer.h"
#include "globals.h"
#include "Renderer.h"
#include "OptixWrapper.h"
#include "Triangle.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"
#include "AreaLight.h"
#include "TerrainManager.h"
#include "MaterialManager.h"
#include "InstanceManager.h"
#include "WaterSystem.h"
#include "Light.h"
#include "Transform.h"
#include "Material.h"
#include "MeshModifiers.h"
#include "json.hpp"
#include "simdjson.h"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cctype>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif
#include "Backend/IBackend.h"
#include "Backend/OptixBackend.h"
using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {
static fs::path pathFromUtf8(const std::string& utf8_path) {
#ifdef _WIN32
    auto toWide = [](const std::string& src, UINT codepage, DWORD flags) -> std::wstring {
        const int size = MultiByteToWideChar(codepage, flags, src.c_str(), -1, nullptr, 0);
        if (size <= 0) return {};
        std::wstring out(static_cast<size_t>(size - 1), L'\0');
        if (MultiByteToWideChar(codepage, flags, src.c_str(), -1, out.data(), size) <= 0) return {};
        return out;
    };

    std::wstring wide = toWide(utf8_path, CP_UTF8, MB_ERR_INVALID_CHARS);
    if (wide.empty()) {
        wide = toWide(utf8_path, CP_ACP, 0);
    }
    if (!wide.empty()) {
        return fs::path(wide);
    }
#endif
    return fs::path(utf8_path);
}

static simdjson::error_code loadJsonRootFromFile(simdjson::dom::parser& parser,
                                                 const std::string& filepath,
                                                 simdjson::dom::element& root,
                                                 std::string* read_error = nullptr) {
    std::ifstream in_json(pathFromUtf8(filepath), std::ios::binary);
    if (!in_json.is_open()) {
        if (read_error) {
            *read_error = "Failed to open file";
        }
        return simdjson::IO_ERROR;
    }

    std::string json_text((std::istreambuf_iterator<char>(in_json)), std::istreambuf_iterator<char>());
    if (!in_json.good() && !in_json.eof()) {
        if (read_error) {
            *read_error = "Failed while reading file stream";
        }
        return simdjson::IO_ERROR;
    }

    simdjson::padded_string padded_json(json_text);
    return parser.parse(padded_json).get(root);
}
}

// Helper to convert Vec3 to JSON array
json vec3ToJson(const Vec3& v) {
    return { v.x, v.y, v.z };
}

// Helper to convert JSON array to Vec3
Vec3 jsonToVec3(const json& j) {
    if (j.is_array() && j.size() >= 3)
        return Vec3(j[0], j[1], j[2]);
    return Vec3(0, 0, 0);
}

// simdjson helpers
static json sjsonToNlohmann(simdjson::dom::element el) {
    return json::parse(std::string(simdjson::minify(el)));
}
static json sjsonToNlohmann(simdjson::simdjson_result<simdjson::dom::element> res) {
    simdjson::dom::element el;
    if (res.get(el)) return json();
    return sjsonToNlohmann(el);
}

static Vec3 sjsonToVec3(simdjson::dom::element el) {
    simdjson::dom::array arr;
    if (el.get_array().get(arr)) return Vec3(0, 0, 0);
    float x = 0, y = 0, z = 0;
    size_t i = 0;
    for (simdjson::dom::element val : arr) {
        double d = 0;
        val.get(d);
        if (i == 0) x = (float)d;
        else if (i == 1) y = (float)d;
        else if (i == 2) z = (float)d;
        i++;
    }
    return Vec3(x, y, z);
}
static Vec3 sjsonToVec3(simdjson::simdjson_result<simdjson::dom::element> res) {
    simdjson::dom::element el;
    if (res.get(el)) return Vec3(0, 0, 0);
    return sjsonToVec3(el);
}

static Matrix4x4 sjsonToMat4(simdjson::dom::element el) {
    Matrix4x4 m;
    simdjson::dom::array arr;
    if (el.get_array().get(arr) || arr.size() != 16) return m;
    int idx = 0;
    for (simdjson::dom::element val : arr) {
        int r = idx / 4;
        int c = idx % 4;
        double d = 0;
        val.get(d);
        m.m[r][c] = (float)d;
        idx++;
    }
    return m;
}
static Matrix4x4 sjsonToMat4(simdjson::simdjson_result<simdjson::dom::element> res) {
    simdjson::dom::element el;
    if (res.get(el)) return Matrix4x4();
    return sjsonToMat4(el);
}

// Helper for Matrix4x4 serialization
json mat4ToJson(const Matrix4x4& m) {
    json j = json::array();
    for(int i=0; i<4; ++i)
        for(int k=0; k<4; ++k)
            j.push_back(m.m[i][k]);
    return j;
}

Matrix4x4 jsonToMat4(const json& j) {
    Matrix4x4 m;
    if (j.is_array() && j.size() == 16) {
        int idx = 0;
        for(int i=0; i<4; ++i)
            for(int k=0; k<4; ++k)
                m.m[i][k] = j[idx++];
    }
    return m;
}

// Helpers for Particle Emitter serialization
json emitterToJson(const RayTrophiSim::ParticleEmitterDesc& e) {
    json j;
    j["name"] = e.name;
    j["source_mode"] = (int)e.source_mode;
    j["spawn_mode"] = (int)e.spawn_mode;
    j["source_name"] = e.source_name;
    j["point"] = vec3ToJson(e.point);
    j["local_offset"] = vec3ToJson(e.local_offset);
    j["direction"] = vec3ToJson(e.direction);
    j["surface_offset"] = e.surface_offset;
    j["rate_per_second"] = e.rate_per_second;
    j["burst_count"] = e.burst_count;
    j["speed"] = e.speed;
    j["spread"] = e.spread;
    j["lifetime_seconds"] = e.lifetime_seconds;
    j["mass"] = e.mass;
    j["start_size"] = e.start_size;
    j["end_size"] = e.end_size;
    j["size_jitter"] = e.size_jitter;
    j["start_opacity"] = e.start_opacity;
    j["end_opacity"] = e.end_opacity;
    j["start_color"] = vec3ToJson(e.start_color);
    j["end_color"] = vec3ToJson(e.end_color);
    j["angular_velocity"] = e.angular_velocity;
    j["angular_jitter"] = e.angular_jitter;
    j["enabled"] = e.enabled;
    j["seed"] = e.seed;
    return j;
}

RayTrophiSim::ParticleEmitterDesc jsonToEmitter(const json& j) {
    RayTrophiSim::ParticleEmitterDesc e;
    if (j.contains("name")) e.name = j["name"];
    if (j.contains("source_mode")) e.source_mode = (RayTrophiSim::ParticleEmitterSourceMode)j["source_mode"];
    if (j.contains("spawn_mode")) e.spawn_mode = (RayTrophiSim::ParticleEmitterSpawnMode)j["spawn_mode"];
    if (j.contains("source_name")) e.source_name = j["source_name"];
    if (j.contains("point")) e.point = jsonToVec3(j["point"]);
    if (j.contains("local_offset")) e.local_offset = jsonToVec3(j["local_offset"]);
    if (j.contains("direction")) e.direction = jsonToVec3(j["direction"]);
    if (j.contains("surface_offset")) e.surface_offset = j["surface_offset"];
    if (j.contains("rate_per_second")) e.rate_per_second = j["rate_per_second"];
    if (j.contains("burst_count")) e.burst_count = j["burst_count"];
    if (j.contains("speed")) e.speed = j["speed"];
    if (j.contains("spread")) e.spread = j["spread"];
    if (j.contains("lifetime_seconds")) e.lifetime_seconds = j["lifetime_seconds"];
    if (j.contains("mass")) e.mass = j["mass"];
    if (j.contains("start_size")) e.start_size = j["start_size"];
    if (j.contains("end_size")) e.end_size = j["end_size"];
    if (j.contains("size_jitter")) e.size_jitter = j["size_jitter"];
    if (j.contains("start_opacity")) e.start_opacity = j["start_opacity"];
    if (j.contains("end_opacity")) e.end_opacity = j["end_opacity"];
    if (j.contains("start_color")) e.start_color = jsonToVec3(j["start_color"]);
    if (j.contains("end_color")) e.end_color = jsonToVec3(j["end_color"]);
    if (j.contains("angular_velocity")) e.angular_velocity = j["angular_velocity"];
    if (j.contains("angular_jitter")) e.angular_jitter = j["angular_jitter"];
    if (j.contains("enabled")) e.enabled = j["enabled"];
    if (j.contains("seed")) e.seed = j["seed"];
    return e;
}

// Helpers for Particle Collider serialization
json colliderToJson(const RayTrophiSim::ParticleColliderDesc& c) {
    json j;
    j["name"] = c.name;
    j["source_mode"] = (int)c.source_mode;
    j["source_name"] = c.source_name;
    j["enabled"] = c.enabled;
    j["plane_y"] = c.plane_y;
    j["sphere_center"] = vec3ToJson(c.sphere_center);
    j["sphere_radius"] = c.sphere_radius;
    j["capsule_start"] = vec3ToJson(c.capsule_start);
    j["capsule_end"] = vec3ToJson(c.capsule_end);
    j["capsule_radius"] = c.capsule_radius;
    j["bounds_min"] = vec3ToJson(c.bounds_min);
    j["bounds_max"] = vec3ToJson(c.bounds_max);
    j["restitution"] = c.restitution;
    j["friction"] = c.friction;
    j["thickness"] = c.thickness;
    j["sdf_resolution_mode"] = c.sdf_resolution_mode;
    j["decimation_ratio"] = c.decimation_ratio;
    j["draw_wireframe"] = c.draw_wireframe;
    j["draw_slice_preview"] = c.draw_slice_preview;
    j["slice_plane_distance"] = c.slice_plane_distance;
    j["slice_axis"] = c.slice_axis;
    return j;
}

RayTrophiSim::ParticleColliderDesc jsonToCollider(const json& j) {
    RayTrophiSim::ParticleColliderDesc c;
    if (j.contains("name")) c.name = j["name"];
    if (j.contains("source_mode")) c.source_mode = (RayTrophiSim::ParticleColliderSourceMode)j["source_mode"];
    if (j.contains("source_name")) c.source_name = j["source_name"];
    if (j.contains("enabled")) c.enabled = j["enabled"];
    if (j.contains("plane_y")) c.plane_y = j["plane_y"];
    if (j.contains("sphere_center")) c.sphere_center = jsonToVec3(j["sphere_center"]);
    if (j.contains("sphere_radius")) c.sphere_radius = j["sphere_radius"];
    if (j.contains("capsule_start")) c.capsule_start = jsonToVec3(j["capsule_start"]);
    if (j.contains("capsule_end")) c.capsule_end = jsonToVec3(j["capsule_end"]);
    if (j.contains("capsule_radius")) c.capsule_radius = j["capsule_radius"];
    if (j.contains("bounds_min")) c.bounds_min = jsonToVec3(j["bounds_min"]);
    if (j.contains("bounds_max")) c.bounds_max = jsonToVec3(j["bounds_max"]);
    if (j.contains("restitution")) c.restitution = j["restitution"];
    if (j.contains("friction")) c.friction = j["friction"];
    if (j.contains("thickness")) c.thickness = j["thickness"];
    if (j.contains("sdf_resolution_mode")) c.sdf_resolution_mode = j["sdf_resolution_mode"];
    if (j.contains("decimation_ratio")) c.decimation_ratio = j["decimation_ratio"];
    if (j.contains("draw_wireframe")) c.draw_wireframe = j["draw_wireframe"];
    if (j.contains("draw_slice_preview")) c.draw_slice_preview = j["draw_slice_preview"];
    if (j.contains("slice_plane_distance")) c.slice_plane_distance = j["slice_plane_distance"];
    if (j.contains("slice_axis")) c.slice_axis = j["slice_axis"];
    return c;
}

// Helpers for Grid Domain serialization
json domainToJson(const RayTrophiSim::SimulationGridDomainDesc& d) {
    json j;
    j["name"] = d.name;
    j["type"] = (int)d.type;
    j["backend"] = (int)d.backend;
    j["source_mode"] = (int)d.source_mode;
    j["boundary_mode"] = (int)d.boundary_mode;
    j["source_name"] = d.source_name;
    j["enabled"] = d.enabled;
    j["preserve_voxel_size_on_resize"] = d.preserve_voxel_size_on_resize;
    j["use_sparse_tiles"] = d.use_sparse_tiles;
    j["render_to_nanovdb"] = d.render_to_nanovdb;
    j["bounds_min"] = vec3ToJson(d.bounds_min);
    j["bounds_max"] = vec3ToJson(d.bounds_max);
    j["resolution_x"] = d.resolution_x;
    j["resolution_y"] = d.resolution_y;
    j["resolution_z"] = d.resolution_z;
    j["max_auto_resolution"] = d.max_auto_resolution;
    j["voxel_size"] = d.voxel_size;
    j["padding"] = d.padding;
    j["adaptive_lock_floor"] = d.adaptive_lock_floor;
    j["adaptive_floor_y"] = d.adaptive_floor_y;
    j["channels"] = d.channels;

    // APICSolverParams
    j["fluid_params"]["gravity"] = vec3ToJson(d.fluid_params.gravity);
    j["fluid_params"]["viscosity"] = d.fluid_params.viscosity;
    j["fluid_params"]["particles_per_cell"] = d.fluid_params.particles_per_cell;
    j["fluid_params"]["cfl"] = d.fluid_params.cfl;
    j["fluid_params"]["max_substeps"] = d.fluid_params.max_substeps;
    j["fluid_params"]["pressure_iterations"] = d.fluid_params.pressure_iterations;
    j["fluid_params"]["pressure_relative_residual"] = d.fluid_params.pressure_relative_residual;
    j["fluid_params"]["pressure_multigrid_preconditioner"] = d.fluid_params.pressure_multigrid_preconditioner;
    j["fluid_params"]["apic_blend"] = d.fluid_params.apic_blend;
    j["fluid_params"]["flip_blend"] = d.fluid_params.flip_blend;
    j["fluid_params"]["internal_friction"] = d.fluid_params.internal_friction;
    j["fluid_params"]["air_drag"] = d.fluid_params.air_drag;
    j["fluid_params"]["reseed_enabled"] = d.fluid_params.reseed_enabled;
    // Remaining preset-driven rheology so a material preset round-trips fully
    // (these were previously dropped, resetting Honey/Lava/Sand on reload).
    j["fluid_params"]["velocity_damping"] = d.fluid_params.velocity_damping;
    j["fluid_params"]["wall_damping"] = d.fluid_params.wall_damping;
    j["fluid_params"]["density_correction"] = d.fluid_params.density_correction;
    j["fluid_params"]["affine_damping"] = d.fluid_params.affine_damping;
    j["fluid_params"]["max_velocity"] = d.fluid_params.max_velocity;
    j["fluid_params"]["viscosity_iterations"] = d.fluid_params.viscosity_iterations;
    j["fluid_params"]["current_preset"] = static_cast<int>(d.fluid_params.current_preset);
    j["fluid_params"]["free_surface"] = d.fluid_params.free_surface;
    j["fluid_params"]["ghost_fluid_surface"] = d.fluid_params.ghost_fluid_surface;
    j["fluid_params"]["variational_solids"] = d.fluid_params.variational_solids;
    j["fluid_params"]["domain_motion_coupling"] = d.fluid_params.domain_motion_coupling;
    j["fluid_params"]["reseed_target_per_cell"] = d.fluid_params.reseed_target_per_cell;
    j["fluid_params"]["reseed_min_per_cell"] = d.fluid_params.reseed_min_per_cell;
    j["fluid_params"]["reseed_max_per_cell"] = d.fluid_params.reseed_max_per_cell;
    j["fluid_params"]["sor_omega"] = d.fluid_params.sor_omega;
    j["fluid_params"]["max_affine"] = d.fluid_params.max_affine;
    j["fluid_params"]["cpu_threads"] = d.fluid_params.cpu_threads;
    j["fluid_params"]["parallel_particle_threshold"] = d.fluid_params.parallel_particle_threshold;

    j["fluid_seed_min"] = vec3ToJson(d.fluid_seed_min);
    j["fluid_seed_max"] = vec3ToJson(d.fluid_seed_max);
    j["fluid_seed_particles_per_cell"] = d.fluid_seed_particles_per_cell;
    j["fluid_max_particles"] = (int)d.fluid_max_particles;
    j["fluid_replace_on_seed"] = d.fluid_replace_on_seed;
    j["fluid_pending_seed"] = d.fluid_pending_seed;
    j["fluid_seed_mode"] = (int)d.fluid_seed_mode;
    j["fluid_fill_level"] = d.fluid_fill_level;
    j["fluid_fill_wall_margin"] = d.fluid_fill_wall_margin;
    j["fluid_render_mode"] = (int)d.fluid_render_mode;
    j["fluid_particle_color"] = vec3ToJson(d.fluid_particle_color);
    j["fluid_particle_radius_factor"] = d.fluid_particle_radius_factor;
    j["fluid_particle_size_multiplier"] = d.fluid_particle_size_multiplier;
    j["fluid_particle_subdivisions"] = d.fluid_particle_subdivisions;
    j["fluid_particle_emissive"] = d.fluid_particle_emissive;
    j["fluid_particle_emission"] = d.fluid_particle_emission;
    j["fluid_particle_material_id"] = d.fluid_particle_material_id;

    // LevelSetParams
    j["fluid_level_set_params"]["narrow_band_voxels"] = d.fluid_level_set_params.narrow_band_voxels;
    j["fluid_level_set_params"]["kernel_radius_voxels"] = d.fluid_level_set_params.kernel_radius_voxels;
    j["fluid_level_set_params"]["particle_radius_voxels"] = d.fluid_level_set_params.particle_radius_voxels;
    j["fluid_level_set_params"]["smoothing_iterations"] = d.fluid_level_set_params.smoothing_iterations;
    j["fluid_level_set_params"]["surface_resolution_multiplier"] = d.fluid_level_set_params.surface_resolution_multiplier;
    j["fluid_level_set_params"]["anisotropy_enabled"] = d.fluid_level_set_params.anisotropy_enabled;
    j["fluid_level_set_params"]["anisotropy_radius_voxels"] = d.fluid_level_set_params.anisotropy_radius_voxels;
    j["fluid_level_set_params"]["anisotropy_max_stretch"] = d.fluid_level_set_params.anisotropy_max_stretch;
    j["fluid_level_set_params"]["anisotropy_neighbor_min"] = d.fluid_level_set_params.anisotropy_neighbor_min;
    j["fluid_level_set_params"]["position_smoothing"] = d.fluid_level_set_params.position_smoothing;

    j["fluid_surface_band_voxels"] = d.fluid_surface_band_voxels;
    j["fluid_surface_ior"] = d.fluid_surface_ior;
    j["fluid_surface_roughness"] = d.fluid_surface_roughness;
    j["fluid_surface_foam"] = d.fluid_surface_foam;
    j["fluid_debug_overlay"] = d.fluid_debug_overlay;

    // Whitewater (foam/spray/bubbles) — Ihmsen 2012.
    {
        const auto& fo = d.fluid_foam_params;
        auto& fj = j["fluid_foam_params"];
        fj["enabled"] = fo.enabled;
        fj["trapped_air_rate"] = fo.trapped_air_rate;
        fj["wave_crest_rate"] = fo.wave_crest_rate;
        fj["ta_min"] = fo.ta_min; fj["ta_max"] = fo.ta_max;
        fj["wc_min"] = fo.wc_min; fj["wc_max"] = fo.wc_max;
        fj["ke_min"] = fo.ke_min; fj["ke_max"] = fo.ke_max;
        fj["crest_cos"] = fo.crest_cos;
        fj["neighbor_radius_voxels"] = fo.neighbor_radius_voxels;
        fj["spray_max_neighbors"] = fo.spray_max_neighbors;
        fj["bubble_min_neighbors"] = fo.bubble_min_neighbors;
        fj["lifetime"] = fo.lifetime;
        fj["buoyancy"] = fo.buoyancy;
        fj["fluid_drag"] = fo.fluid_drag;
        fj["spray_drag"] = fo.spray_drag;
        fj["spawn_jitter_voxels"] = fo.spawn_jitter_voxels;
        fj["max_foam"] = static_cast<uint64_t>(fo.max_foam);
        fj["render_radius_voxels"] = fo.render_radius_voxels;
        fj["foam_sphere_subdivisions"] = fo.foam_sphere_subdivisions;
        fj["render_mode"] = static_cast<int>(fo.render_mode);
        fj["volume_density"] = fo.volume_density;
        fj["volume_color"] = { fo.volume_color.x, fo.volume_color.y, fo.volume_color.z };
        fj["volume_opacity"] = fo.volume_opacity;
        fj["volume_bubble_strength"] = fo.volume_bubble_strength;
        fj["volume_spray_strength"]  = fo.volume_spray_strength;
        fj["foam_material_id"] = fo.foam_material_id;
        fj["spray_material_id"] = fo.spray_material_id;
        fj["bubble_material_id"] = fo.bubble_material_id;
        fj["surface_kernel_radius_voxels"]   = fo.surface_kernel_radius_voxels;
        fj["surface_particle_radius_voxels"] = fo.surface_particle_radius_voxels;
        fj["surface_band_voxels"]            = fo.surface_band_voxels;
        fj["surface_smoothing_iterations"]   = fo.surface_smoothing_iterations;
        fj["surface_resolution_multiplier"]  = fo.surface_resolution_multiplier;
    }

    // Fire settings
    j["fire_enabled"] = d.fire_enabled;
    j["ignition_temperature"] = d.ignition_temperature;
    j["burn_rate"] = d.burn_rate;
    j["heat_release"] = d.heat_release;
    j["smoke_generation"] = d.smoke_generation;
    j["flame_dissipation"] = d.flame_dissipation;
    j["fire_max_temperature"] = d.fire_max_temperature;
    j["fire_expansion"] = d.fire_expansion;
    j["turbulence_strength"] = d.turbulence_strength;
    j["turbulence_scale"] = d.turbulence_scale;
    j["turbulence_octaves"] = d.turbulence_octaves;
    j["turbulence_lacunarity"] = d.turbulence_lacunarity;
    j["turbulence_persistence"] = d.turbulence_persistence;
    j["turbulence_speed"] = d.turbulence_speed;
    return j;
}

RayTrophiSim::SimulationGridDomainDesc jsonToDomain(const json& j) {
    RayTrophiSim::SimulationGridDomainDesc d;
    if (j.contains("name")) d.name = j["name"];
    if (j.contains("type")) d.type = (RayTrophiSim::SimulationDomainType)j["type"];
    if (j.contains("backend")) d.backend = (RayTrophiSim::SimulationDomainBackend)j["backend"];
    if (j.contains("source_mode")) d.source_mode = (RayTrophiSim::SimulationGridDomainSourceMode)j["source_mode"];
    if (j.contains("boundary_mode")) d.boundary_mode = (RayTrophiSim::SimulationGridDomainBoundaryMode)j["boundary_mode"];
    if (j.contains("source_name")) d.source_name = j["source_name"];
    if (j.contains("enabled")) d.enabled = j["enabled"];
    if (j.contains("preserve_voxel_size_on_resize")) d.preserve_voxel_size_on_resize = j["preserve_voxel_size_on_resize"];
    if (j.contains("use_sparse_tiles")) d.use_sparse_tiles = j["use_sparse_tiles"];
    if (j.contains("render_to_nanovdb")) d.render_to_nanovdb = j["render_to_nanovdb"];
    if (j.contains("bounds_min")) d.bounds_min = jsonToVec3(j["bounds_min"]);
    if (j.contains("bounds_max")) d.bounds_max = jsonToVec3(j["bounds_max"]);
    if (j.contains("resolution_x")) d.resolution_x = j["resolution_x"];
    if (j.contains("resolution_y")) d.resolution_y = j["resolution_y"];
    if (j.contains("resolution_z")) d.resolution_z = j["resolution_z"];
    if (j.contains("max_auto_resolution")) d.max_auto_resolution = j["max_auto_resolution"];
    if (j.contains("voxel_size")) d.voxel_size = j["voxel_size"];
    if (j.contains("padding")) d.padding = j["padding"];
    if (j.contains("adaptive_lock_floor")) d.adaptive_lock_floor = j["adaptive_lock_floor"];
    if (j.contains("adaptive_floor_y")) d.adaptive_floor_y = j["adaptive_floor_y"];
    if (j.contains("channels")) d.channels = j["channels"];

    // APICSolverParams
    if (j.contains("fluid_params")) {
        const auto& fp = j["fluid_params"];
        if (fp.contains("gravity")) d.fluid_params.gravity = jsonToVec3(fp["gravity"]);
        if (fp.contains("viscosity")) d.fluid_params.viscosity = fp["viscosity"];
        if (fp.contains("particles_per_cell")) d.fluid_params.particles_per_cell = fp["particles_per_cell"];
        if (fp.contains("cfl")) d.fluid_params.cfl = fp["cfl"];
        if (fp.contains("max_substeps")) d.fluid_params.max_substeps = fp["max_substeps"];
        if (fp.contains("pressure_iterations")) d.fluid_params.pressure_iterations = fp["pressure_iterations"];
        if (fp.contains("pressure_relative_residual")) d.fluid_params.pressure_relative_residual = fp["pressure_relative_residual"];
        if (fp.contains("pressure_multigrid_preconditioner")) d.fluid_params.pressure_multigrid_preconditioner = fp["pressure_multigrid_preconditioner"];
        if (fp.contains("apic_blend")) d.fluid_params.apic_blend = fp["apic_blend"];
        if (fp.contains("flip_blend")) d.fluid_params.flip_blend = fp["flip_blend"];
        if (fp.contains("internal_friction")) d.fluid_params.internal_friction = fp["internal_friction"];
        if (fp.contains("air_drag")) d.fluid_params.air_drag = fp["air_drag"];
        if (fp.contains("reseed_enabled")) d.fluid_params.reseed_enabled = fp["reseed_enabled"];
        if (fp.contains("velocity_damping")) d.fluid_params.velocity_damping = fp["velocity_damping"];
        if (fp.contains("wall_damping")) d.fluid_params.wall_damping = fp["wall_damping"];
        if (fp.contains("density_correction")) d.fluid_params.density_correction = fp["density_correction"];
        if (fp.contains("affine_damping")) d.fluid_params.affine_damping = fp["affine_damping"];
        if (fp.contains("max_velocity")) d.fluid_params.max_velocity = fp["max_velocity"];
        if (fp.contains("viscosity_iterations")) d.fluid_params.viscosity_iterations = fp["viscosity_iterations"];
        if (fp.contains("current_preset"))
            d.fluid_params.current_preset =
                static_cast<RayTrophiSim::Fluid::APICSolverParams::FluidPreset>(fp["current_preset"].get<int>());
        if (fp.contains("free_surface")) d.fluid_params.free_surface = fp["free_surface"];
        if (fp.contains("ghost_fluid_surface")) d.fluid_params.ghost_fluid_surface = fp["ghost_fluid_surface"];
        if (fp.contains("variational_solids")) d.fluid_params.variational_solids = fp["variational_solids"];
        if (fp.contains("domain_motion_coupling")) d.fluid_params.domain_motion_coupling = fp["domain_motion_coupling"];
        if (fp.contains("reseed_target_per_cell")) d.fluid_params.reseed_target_per_cell = fp["reseed_target_per_cell"];
        if (fp.contains("reseed_min_per_cell")) d.fluid_params.reseed_min_per_cell = fp["reseed_min_per_cell"];
        if (fp.contains("reseed_max_per_cell")) d.fluid_params.reseed_max_per_cell = fp["reseed_max_per_cell"];
        if (fp.contains("sor_omega")) d.fluid_params.sor_omega = fp["sor_omega"];
        if (fp.contains("max_affine")) d.fluid_params.max_affine = fp["max_affine"];
        if (fp.contains("cpu_threads")) d.fluid_params.cpu_threads = fp["cpu_threads"];
        if (fp.contains("parallel_particle_threshold")) d.fluid_params.parallel_particle_threshold = fp["parallel_particle_threshold"];
    }

    if (j.contains("fluid_seed_min")) d.fluid_seed_min = jsonToVec3(j["fluid_seed_min"]);
    if (j.contains("fluid_seed_max")) d.fluid_seed_max = jsonToVec3(j["fluid_seed_max"]);
    if (j.contains("fluid_seed_particles_per_cell")) d.fluid_seed_particles_per_cell = j["fluid_seed_particles_per_cell"];
    if (j.contains("fluid_max_particles")) d.fluid_max_particles = j["fluid_max_particles"];
    if (j.contains("fluid_replace_on_seed")) d.fluid_replace_on_seed = j["fluid_replace_on_seed"];
    if (j.contains("fluid_pending_seed")) d.fluid_pending_seed = j["fluid_pending_seed"];
    if (j.contains("fluid_seed_mode")) d.fluid_seed_mode = (RayTrophiSim::FluidSeedMode)(int)j["fluid_seed_mode"];
    if (j.contains("fluid_fill_level")) d.fluid_fill_level = j["fluid_fill_level"];
    if (j.contains("fluid_fill_wall_margin")) d.fluid_fill_wall_margin = j["fluid_fill_wall_margin"];
    if (j.contains("fluid_render_mode")) d.fluid_render_mode = (RayTrophiSim::Fluid::FluidRenderMode)j["fluid_render_mode"];
    if (j.contains("fluid_particle_color")) d.fluid_particle_color = jsonToVec3(j["fluid_particle_color"]);
    if (j.contains("fluid_particle_radius_factor")) d.fluid_particle_radius_factor = j["fluid_particle_radius_factor"];
    if (j.contains("fluid_particle_size_multiplier")) d.fluid_particle_size_multiplier = j["fluid_particle_size_multiplier"];
    if (j.contains("fluid_particle_subdivisions")) d.fluid_particle_subdivisions = j["fluid_particle_subdivisions"];
    if (j.contains("fluid_particle_emissive")) d.fluid_particle_emissive = j["fluid_particle_emissive"];
    if (j.contains("fluid_particle_emission")) d.fluid_particle_emission = j["fluid_particle_emission"];
    if (j.contains("fluid_particle_material_id")) d.fluid_particle_material_id = j["fluid_particle_material_id"];

    // LevelSetParams
    if (j.contains("fluid_level_set_params")) {
        const auto& lsp = j["fluid_level_set_params"];
        if (lsp.contains("narrow_band_voxels")) d.fluid_level_set_params.narrow_band_voxels = lsp["narrow_band_voxels"];
        if (lsp.contains("kernel_radius_voxels")) d.fluid_level_set_params.kernel_radius_voxels = lsp["kernel_radius_voxels"];
        if (lsp.contains("particle_radius_voxels")) d.fluid_level_set_params.particle_radius_voxels = lsp["particle_radius_voxels"];
        if (lsp.contains("smoothing_iterations")) d.fluid_level_set_params.smoothing_iterations = lsp["smoothing_iterations"];
        if (lsp.contains("surface_resolution_multiplier")) d.fluid_level_set_params.surface_resolution_multiplier = lsp["surface_resolution_multiplier"];
        if (lsp.contains("anisotropy_enabled")) d.fluid_level_set_params.anisotropy_enabled = lsp["anisotropy_enabled"];
        if (lsp.contains("anisotropy_radius_voxels")) d.fluid_level_set_params.anisotropy_radius_voxels = lsp["anisotropy_radius_voxels"];
        if (lsp.contains("anisotropy_max_stretch")) d.fluid_level_set_params.anisotropy_max_stretch = lsp["anisotropy_max_stretch"];
        if (lsp.contains("anisotropy_neighbor_min")) d.fluid_level_set_params.anisotropy_neighbor_min = lsp["anisotropy_neighbor_min"];
        if (lsp.contains("position_smoothing")) d.fluid_level_set_params.position_smoothing = lsp["position_smoothing"];
    }

    if (j.contains("fluid_surface_band_voxels")) d.fluid_surface_band_voxels = j["fluid_surface_band_voxels"];
    if (j.contains("fluid_surface_ior")) d.fluid_surface_ior = j["fluid_surface_ior"];
    if (j.contains("fluid_surface_roughness")) d.fluid_surface_roughness = j["fluid_surface_roughness"];
    if (j.contains("fluid_surface_foam")) d.fluid_surface_foam = j["fluid_surface_foam"];
    if (j.contains("fluid_debug_overlay")) d.fluid_debug_overlay = j["fluid_debug_overlay"];

    if (j.contains("fluid_foam_params")) {
        const auto& fj = j["fluid_foam_params"];
        auto& fo = d.fluid_foam_params;
        if (fj.contains("enabled")) fo.enabled = fj["enabled"];
        if (fj.contains("trapped_air_rate")) fo.trapped_air_rate = fj["trapped_air_rate"];
        if (fj.contains("wave_crest_rate")) fo.wave_crest_rate = fj["wave_crest_rate"];
        if (fj.contains("ta_min")) fo.ta_min = fj["ta_min"];
        if (fj.contains("ta_max")) fo.ta_max = fj["ta_max"];
        if (fj.contains("wc_min")) fo.wc_min = fj["wc_min"];
        if (fj.contains("wc_max")) fo.wc_max = fj["wc_max"];
        if (fj.contains("ke_min")) fo.ke_min = fj["ke_min"];
        if (fj.contains("ke_max")) fo.ke_max = fj["ke_max"];
        if (fj.contains("crest_cos")) fo.crest_cos = fj["crest_cos"];
        if (fj.contains("neighbor_radius_voxels")) fo.neighbor_radius_voxels = fj["neighbor_radius_voxels"];
        if (fj.contains("spray_max_neighbors")) fo.spray_max_neighbors = fj["spray_max_neighbors"];
        if (fj.contains("bubble_min_neighbors")) fo.bubble_min_neighbors = fj["bubble_min_neighbors"];
        if (fj.contains("lifetime")) fo.lifetime = fj["lifetime"];
        if (fj.contains("buoyancy")) fo.buoyancy = fj["buoyancy"];
        if (fj.contains("fluid_drag")) fo.fluid_drag = fj["fluid_drag"];
        if (fj.contains("spray_drag")) fo.spray_drag = fj["spray_drag"];
        if (fj.contains("spawn_jitter_voxels")) fo.spawn_jitter_voxels = fj["spawn_jitter_voxels"];
        if (fj.contains("max_foam")) fo.max_foam = static_cast<std::size_t>(fj["max_foam"].get<uint64_t>());
        if (fj.contains("render_radius_voxels")) fo.render_radius_voxels = fj["render_radius_voxels"];
        if (fj.contains("foam_sphere_subdivisions")) fo.foam_sphere_subdivisions = fj["foam_sphere_subdivisions"].get<int>();
        if (fj.contains("render_mode")) {
            // FoamRenderMode { Surface=0 (legacy, treated as Spheres), Spheres=1,
            // Volume=2 (foam rides the surface volume's temperature channel) }.
            int rm = fj["render_mode"].get<int>();
            if (rm < 0 || rm > 2) rm = 0;
            fo.render_mode = static_cast<RayTrophiSim::Fluid::FoamRenderMode>(rm);
        }
        if (fj.contains("volume_density")) fo.volume_density = fj["volume_density"];
        if (fj.contains("volume_color") && fj["volume_color"].is_array() && fj["volume_color"].size() == 3) {
            fo.volume_color = Vec3(fj["volume_color"][0].get<float>(),
                                   fj["volume_color"][1].get<float>(),
                                   fj["volume_color"][2].get<float>());
        }
        if (fj.contains("volume_opacity")) fo.volume_opacity = fj["volume_opacity"].get<float>();
        if (fj.contains("volume_bubble_strength")) fo.volume_bubble_strength = fj["volume_bubble_strength"].get<float>();
        if (fj.contains("volume_spray_strength"))  fo.volume_spray_strength  = fj["volume_spray_strength"].get<float>();
        if (fj.contains("foam_material_id")) fo.foam_material_id = fj["foam_material_id"].get<int>();
        if (fj.contains("spray_material_id")) fo.spray_material_id = fj["spray_material_id"].get<int>();
        if (fj.contains("bubble_material_id")) fo.bubble_material_id = fj["bubble_material_id"].get<int>();
        if (fj.contains("surface_kernel_radius_voxels")) fo.surface_kernel_radius_voxels = fj["surface_kernel_radius_voxels"];
        if (fj.contains("surface_particle_radius_voxels")) fo.surface_particle_radius_voxels = fj["surface_particle_radius_voxels"];
        if (fj.contains("surface_band_voxels")) fo.surface_band_voxels = fj["surface_band_voxels"];
        if (fj.contains("surface_smoothing_iterations")) fo.surface_smoothing_iterations = fj["surface_smoothing_iterations"];
        if (fj.contains("surface_resolution_multiplier")) fo.surface_resolution_multiplier = fj["surface_resolution_multiplier"];
    }

    if (j.contains("fire_enabled")) d.fire_enabled = j["fire_enabled"];
    if (j.contains("ignition_temperature")) d.ignition_temperature = j["ignition_temperature"];
    if (j.contains("burn_rate")) d.burn_rate = j["burn_rate"];
    if (j.contains("heat_release")) d.heat_release = j["heat_release"];
    if (j.contains("smoke_generation")) d.smoke_generation = j["smoke_generation"];
    if (j.contains("flame_dissipation")) d.flame_dissipation = j["flame_dissipation"];
    if (j.contains("fire_max_temperature")) d.fire_max_temperature = j["fire_max_temperature"];
    if (j.contains("fire_expansion")) d.fire_expansion = j["fire_expansion"];
    if (j.contains("turbulence_strength")) d.turbulence_strength = j["turbulence_strength"];
    if (j.contains("turbulence_scale")) d.turbulence_scale = j["turbulence_scale"];
    if (j.contains("turbulence_octaves")) d.turbulence_octaves = j["turbulence_octaves"];
    if (j.contains("turbulence_lacunarity")) d.turbulence_lacunarity = j["turbulence_lacunarity"];
    if (j.contains("turbulence_persistence")) d.turbulence_persistence = j["turbulence_persistence"];
    if (j.contains("turbulence_speed")) d.turbulence_speed = j["turbulence_speed"];
    return d;
}

// Helpers for Flow Source serialization
json flowSourceToJson(const RayTrophiSim::SimulationFlowSourceDesc& fs) {
    json j;
    j["name"] = fs.name;
    j["source_mode"] = (int)fs.source_mode;
    j["source_name"] = fs.source_name;
    j["domain_index"] = fs.domain_index;
    j["enabled"] = fs.enabled;
    j["position"] = vec3ToJson(fs.position);
    j["velocity"] = vec3ToJson(fs.velocity);
    j["radius"] = fs.radius;
    j["density"] = fs.density;
    j["temperature"] = fs.temperature;
    j["fuel"] = fs.fuel;
    j["falloff"] = fs.falloff;
    j["fluid_particles_per_second"] = fs.fluid_particles_per_second;
    j["fluid_velocity_spread"] = fs.fluid_velocity_spread;
    j["fluid_emit_along_normal"] = fs.fluid_emit_along_normal;
    j["use_time_limit"] = fs.use_time_limit;
    j["start_time"] = fs.start_time;
    j["end_time"] = fs.end_time;
    j["use_particle_limit"] = fs.use_particle_limit;
    j["max_emitted_particles"] = fs.max_emitted_particles;
    return j;
}

RayTrophiSim::SimulationFlowSourceDesc jsonToFlowSource(const json& j) {
    RayTrophiSim::SimulationFlowSourceDesc fs;
    if (j.contains("name")) fs.name = j["name"];
    if (j.contains("source_mode")) fs.source_mode = (RayTrophiSim::SimulationFlowSourceMode)j["source_mode"];
    if (j.contains("source_name")) fs.source_name = j["source_name"];
    if (j.contains("domain_index")) fs.domain_index = j["domain_index"];
    if (j.contains("enabled")) fs.enabled = j["enabled"];
    if (j.contains("position")) fs.position = jsonToVec3(j["position"]);
    if (j.contains("velocity")) fs.velocity = jsonToVec3(j["velocity"]);
    if (j.contains("radius")) fs.radius = j["radius"];
    if (j.contains("density")) fs.density = j["density"];
    if (j.contains("temperature")) fs.temperature = j["temperature"];
    if (j.contains("fuel")) fs.fuel = j["fuel"];
    if (j.contains("falloff")) fs.falloff = j["falloff"];
    if (j.contains("fluid_particles_per_second")) fs.fluid_particles_per_second = j["fluid_particles_per_second"];
    if (j.contains("fluid_velocity_spread")) fs.fluid_velocity_spread = j["fluid_velocity_spread"];
    if (j.contains("fluid_emit_along_normal")) fs.fluid_emit_along_normal = j["fluid_emit_along_normal"];
    if (j.contains("use_time_limit")) fs.use_time_limit = j["use_time_limit"];
    if (j.contains("start_time")) fs.start_time = j["start_time"];
    if (j.contains("end_time")) fs.end_time = j["end_time"];
    if (j.contains("use_particle_limit")) fs.use_particle_limit = j["use_particle_limit"];
    if (j.contains("max_emitted_particles")) fs.max_emitted_particles = j["max_emitted_particles"];
    return fs;
}

// NOTE: SceneSerializer::Serialize is currently NOT used. Saving is handled by ProjectManager.
void SceneSerializer::Serialize(const SceneData& scene, const RenderSettings& settings, const std::string& filepath) {
    json root;

    // 1. Metadata
    extern std::string active_model_path;
    root["model_path"] = active_model_path;

    // 2. Camera
    if (scene.camera) {
        root["camera"]["lookfrom"] = vec3ToJson(scene.camera->lookfrom);
        root["camera"]["lookat"] = vec3ToJson(scene.camera->lookat);
        root["camera"]["vup"] = vec3ToJson(scene.camera->vup);
        root["camera"]["vfov"] = scene.camera->vfov;
        root["camera"]["aperture"] = scene.camera->aperture;
        root["camera"]["focus_dist"] = scene.camera->focus_dist;
    }

    // 3. Lights
    root["lights"] = json::array();
    for (const auto& l : scene.lights) {
        json lj;
        lj["type"] = (int)l->type();
        lj["position"] = vec3ToJson(l->position);
        lj["color"] = vec3ToJson(l->color);
        lj["intensity"] = 1.0f; // Simplified for now as intensity is often baked in color
        lj["name"] = l->nodeName;

        if (l->type() == LightType::Point) {
            lj["radius"] = l->getRadius();
        } else if (l->type() == LightType::Directional) {
            lj["direction"] = vec3ToJson(std::static_pointer_cast<DirectionalLight>(l)->direction);
            lj["radius"] = l->getRadius();
        } else if (l->type() == LightType::Spot) {
            auto sl = std::static_pointer_cast<SpotLight>(l);
            lj["direction"] = vec3ToJson(sl->direction);
            lj["radius"] = sl->getRadius();
            lj["angle"] = sl->getAngleDegrees();
            lj["falloff"] = sl->getFalloff();
        } else if (l->type() == LightType::Area) {
            auto al = std::static_pointer_cast<AreaLight>(l);
            lj["u"] = vec3ToJson(al->getU());
            lj["v"] = vec3ToJson(al->getV());
            lj["width"] = al->getWidth();
            lj["height"] = al->getHeight();
        }
        root["lights"].push_back(lj);
    }

    // 4. Objects (Transforms only for scene serializer)
    root["objects"] = json::array();
    for (const auto& obj : scene.world.objects) {
        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
        if (tri) {
            json oj;
            oj["name"] = tri->getNodeName();
            oj["material_id"] = tri->getMaterialID();
            Transform* th = tri->getTransformPtr();
            if (th) {
                oj["transform"] = mat4ToJson(th->getPivotMatrix());
                oj["pivot_offset"] = vec3ToJson(th->pivot_offset);
            }
            root["objects"].push_back(oj);
        }
    }

    // 5. Settings
    root["settings"]["quality_preset"] = (int)settings.quality_preset;
    root["settings"]["samples_per_pixel"] = settings.samples_per_pixel;
    root["settings"]["max_bounces"] = settings.max_bounces;
    root["settings"]["diffuse_bounces"] = settings.diffuse_bounces;
    root["settings"]["transmission_bounces"] = settings.transmission_bounces;
    root["settings"]["use_adaptive"] = settings.use_adaptive_sampling;
    root["settings"]["use_denoiser"] = settings.use_denoiser;
    root["settings"]["denoiser_mode"] = static_cast<int64_t>(settings.denoiser_mode);
    root["settings"]["denoiser_quality"] = static_cast<int64_t>(settings.denoiser_quality);
    root["settings"]["use_optix"] = settings.use_optix;
    root["settings"]["use_vulkan"] = settings.use_vulkan;
    root["settings"]["backend"] = settings.use_vulkan ? "vulkan" : (settings.use_optix ? "optix" : "cpu");
    root["settings"]["persistent_tonemap"] = settings.persistent_tonemap;
    
    // Animation Sequencer Settings
    root["settings"]["animation_start_frame"] = settings.animation_start_frame;
    root["settings"]["animation_end_frame"] = settings.animation_end_frame;
    root["settings"]["animation_fps"] = settings.animation_fps;
    root["settings"]["animation_output_folder"] = settings.animation_output_folder;

    // 6. PostFX
    const auto& pp = scene.color_processor.params;
    root["postfx"]["exposure"] = pp.global_exposure;
    root["postfx"]["gamma"] = pp.global_gamma;
    root["postfx"]["saturation"] = pp.saturation;
    root["postfx"]["color_temperature"] = pp.color_temperature;
    root["postfx"]["tone_mapping"] = (int)pp.tone_mapping_type;
    root["postfx"]["vignette_enabled"] = pp.enable_vignette;
    root["postfx"]["vignette_strength"] = pp.vignette_strength;

    // 7. World
    root["world"]["background_color"] = vec3ToJson(scene.background_color);

    // 8. Timeline
    json timelineJson;
    scene.timeline.serialize(timelineJson);
    root["timeline"] = timelineJson;

    // 8.5 Modifiers
    root["mesh_modifiers"] = json::object();
    for (const auto& [nodeName, stack] : scene.mesh_modifiers) {
        if (!stack.modifiers.empty()) {
            json stackJson;
            stack.serialize(stackJson);
            root["mesh_modifiers"][nodeName] = stackJson;
        }
    }

    // 8.6 Geometry Node Graphs (Faz 8a Geo-DAG) — structure only. This serializer has
    // no binary sidecar, so the originalBaseMesh snapshot is skipped (bin = nullptr);
    // the full project save path (ProjectManager) persists it.
    root["geometry_node_graphs"] = json::object();
    for (const auto& [nodeName, graphPtr] : scene.geometry_node_graphs) {
        if (!graphPtr) continue;
        bool hasOperator = false;
        for (const auto& n : graphPtr->nodes) {
            const std::string t = n->getTypeId();
            if (t != "GeoV2.BaseMesh" && t != "GeoV2.Output") { hasOperator = true; break; }
        }
        if (!hasOperator) continue;
        json jg;
        GeometryNodesV2::serializeGeometryGraph(*graphPtr, jg, nullptr);
        root["geometry_node_graphs"][nodeName] = jg;
    }

    // 8.7 Material Node Graphs (MaterialNodesV2 Faz 1) — structure only; textures
    // are referenced by name. Graphs with just the default Output node are skipped
    // (equivalent to the material itself).
    root["material_node_graphs"] = json::object();
    for (const auto& [matName, graphPtr] : scene.material_node_graphs) {
        if (!graphPtr || graphPtr->nodes.size() <= 1) continue;
        json jg;
        MaterialNodesV2::serializeMaterialGraph(*graphPtr, jg);
        root["material_node_graphs"][matName] = jg;
    }

    // 9.6 Particle Systems
    root["active_particle_system_index"] = scene.active_particle_system_index;
    root["particle_systems"] = json::array();
    for (const auto& system : scene.particle_systems) {
        json sj;
        sj["id"] = system.id;
        sj["name"] = system.name;
        sj["visible"] = system.visible;
        sj["enabled"] = system.enabled;
        sj["blend_mode"] = (int)system.blend_mode;

        // Render Settings
        sj["render"]["render_in_raytrace"] = system.render.render_in_raytrace;
        sj["render"]["shape"] = (int)system.render.shape;
        sj["render"]["size_multiplier"] = system.render.size_multiplier;
        sj["render"]["sphere_subdivisions"] = system.render.sphere_subdivisions;
        sj["render"]["emissive"] = system.render.emissive;
        sj["render"]["inherit_color_from_emitter"] = system.render.inherit_color_from_emitter;
        sj["render"]["base_color"] = vec3ToJson(system.render.base_color);
        sj["render"]["color_end"] = vec3ToJson(system.render.color_end);
        sj["render"]["color_buckets"] = system.render.color_buckets;
        sj["render"]["over_life_color"] = system.render.over_life_color;
        sj["render"]["emission_strength"] = system.render.emission_strength;
        sj["render"]["roughness"] = system.render.roughness;

        sj["render"]["mesh_sources"] = json::array();
        for (const auto& ms : system.render.mesh_sources) {
            json msj;
            msj["node_name"] = ms.node_name;
            msj["weight"] = ms.weight;
            sj["render"]["mesh_sources"].push_back(msj);
        }

        // Solver Runtime Settings
        if (system.runtime) {
            sj["runtime"]["gravity"] = vec3ToJson(system.runtime->gravity());
            sj["runtime"]["linear_drag"] = system.runtime->linearDrag();
            sj["runtime"]["collision_plane_y"] = system.runtime->collisionPlaneY();
            sj["runtime"]["collision_plane_enabled"] = system.runtime->collisionPlaneEnabled();
            sj["runtime"]["collision_restitution"] = system.runtime->collisionRestitution();

            // Physics Settings
            const auto& ps = system.runtime->physicsSettings();
            sj["runtime"]["physics_settings"]["mode"] = (int)ps.mode;
            sj["runtime"]["physics_settings"]["quality"] = (int)ps.quality;
            sj["runtime"]["physics_settings"]["particle_radius"] = ps.particle_radius;
            sj["runtime"]["physics_settings"]["self_collision_enabled"] = ps.self_collision_enabled;
            sj["runtime"]["physics_settings"]["solver_iterations"] = ps.solver_iterations;
            sj["runtime"]["physics_settings"]["max_neighbors_per_particle"] = ps.max_neighbors_per_particle;
            sj["runtime"]["physics_settings"]["viscosity"] = ps.viscosity;
            sj["runtime"]["physics_settings"]["cohesion"] = ps.cohesion;
            sj["runtime"]["physics_settings"]["pressure_stiffness"] = ps.pressure_stiffness;
            sj["runtime"]["physics_settings"]["rest_density"] = ps.rest_density;
            sj["runtime"]["physics_settings"]["buoyancy"] = ps.buoyancy;
            sj["runtime"]["physics_settings"]["gravity_scale"] = ps.gravity_scale;
            sj["runtime"]["physics_settings"]["vorticity"] = ps.vorticity;

            // Emitters
            sj["runtime"]["emitters"] = json::array();
            for (const auto& em : system.runtime->emitters()) {
                sj["runtime"]["emitters"].push_back(emitterToJson(em));
            }

            // Colliders
            sj["runtime"]["colliders"] = json::array();
            for (const auto& col : system.runtime->colliders()) {
                sj["runtime"]["colliders"].push_back(colliderToJson(col));
            }

            // Grid Domains
            sj["runtime"]["grid_domains"] = json::array();
            for (const auto& dom : system.runtime->gridDomains()) {
                sj["runtime"]["grid_domains"].push_back(domainToJson(dom));
            }

            // Flow Sources
            sj["runtime"]["flow_sources"] = json::array();
            for (const auto& fs : system.runtime->flowSources()) {
                sj["runtime"]["flow_sources"].push_back(flowSourceToJson(fs));
            }
        }
        root["particle_systems"].push_back(sj);
    }

    // 9. Terrains
    auto abs_path = std::filesystem::absolute(filepath);
    std::string terrainDir = abs_path.parent_path().string();
    root["terrain_system"] = TerrainManager::getInstance().serialize(terrainDir);

    // Write to file with optimized buffer
    std::ofstream out(filepath);
    if (out.is_open()) {
        // Optimization: Use dump(0) for fastest serialization and smallest file size
        // Indentation is only for readability, which isn't needed for auto-saves or performance
        out << root.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
        out.close();
    }
}

// NOTE: SceneSerializer::Deserialize is only used as a fallback legacy loader (e.g. for non-.rtp files) in scene_ui.cpp.
bool SceneSerializer::Deserialize(SceneData& scene, RenderSettings& settings, Renderer& renderer, Backend::IBackend* backend, const std::string& filepath) {
    simdjson::dom::parser parser;
    simdjson::dom::element root;

    std::string read_error;
    auto error = loadJsonRootFromFile(parser, filepath, root, &read_error);
    if (error) {
        SCENE_LOG_ERROR("Failed to parse JSON scene file with simdjson: " + std::string(simdjson::error_message(error)) +
                        " | path=" + filepath +
                        (read_error.empty() ? "" : " | read_error=" + read_error));
        return false;
    }

    // 1. Load Base Scene (Model or Default)
    extern std::string active_model_path;
    std::string model_path = "Untitled";
    {
        std::string_view path_sv;
        if (!root["model_path"].get(path_sv)) model_path = std::string(path_sv);
    }
    active_model_path = model_path;

    // Clear current scene
    scene.clear();
    renderer.resetCPUAccumulation();
    if (backend) backend->resetAccumulation();

    // 2. Apply Camera
    simdjson::dom::element cam;
    if (!root["camera"].get(cam)) {
        if (!scene.camera) {
             scene.camera = std::make_shared<Camera>(Vec3(0,0,5), Vec3(0,0,0), Vec3(0,1,0), 60.0, 1.0, 0.0, 1.0, 5);
        }
        scene.camera->lookfrom = sjsonToVec3(cam["lookfrom"]);
        scene.camera->lookat = sjsonToVec3(cam["lookat"]);
        scene.camera->vup = sjsonToVec3(cam["vup"]);
        
        double vfov = 60.0, aperture = 0.0, focus_dist = 10.0;
        cam["vfov"].get(vfov);
        cam["aperture"].get(aperture);
        cam["focus_dist"].get(focus_dist);
        
        scene.camera->vfov = vfov;
        scene.camera->aperture = (float)aperture;
        scene.camera->focus_dist = (float)focus_dist;
        scene.camera->update_camera_vectors();
    }

    // 3. Apply Lights
    simdjson::dom::array lights;
    if (!root["lights"].get(lights)) {
        scene.lights.clear();
        for (auto l : lights) {
            std::shared_ptr<Light> lightPtr = nullptr;
            int64_t type = 0;
            l["type"].get(type);
            
            Vec3 pos = sjsonToVec3(l["position"]);
            Vec3 col = sjsonToVec3(l["color"]);
            double intensity = 1.0;
            l["intensity"].get(intensity);
            
            std::string_view name_sv = "Light";
            l["name"].get(name_sv);
            std::string name(name_sv);
            
            if (type == (int)LightType::Point) {
                double r = 0.1;
                l["radius"].get(r);
                lightPtr = std::make_shared<PointLight>(pos, col * (float)intensity, (float)r);
            } else if (type == (int)LightType::Directional) {
                Vec3 dir = sjsonToVec3(l["direction"]);
                double r = 0.0;
                l["radius"].get(r);
                auto dl = std::make_shared<DirectionalLight>(dir, col * (float)intensity, (float)r);
                dl->position = pos; 
                lightPtr = dl;
            } else if (type == (int)LightType::Spot) {
                Vec3 dir = sjsonToVec3(l["direction"]);
                double range = 20.0, angle = 45.0, falloff = 0.5;
                l["radius"].get(range);
                l["angle"].get(angle);
                l["falloff"].get(falloff);
                auto sl = std::make_shared<SpotLight>(pos, dir, col * (float)intensity, (float)angle, (float)range);
                sl->setFalloff((float)falloff);
                lightPtr = sl;
            } else if (type == (int)LightType::Area) {
                Vec3 u = sjsonToVec3(l["u"]);
                Vec3 v = sjsonToVec3(l["v"]);
                double w = 1.0, h = 1.0;
                l["width"].get(w);
                l["height"].get(h);
                lightPtr = std::make_shared<AreaLight>(pos, u, v, (float)w, (float)h, col * (float)intensity);
            }
            
            if (lightPtr) {
                lightPtr->nodeName = name;
                scene.lights.push_back(lightPtr);
            }
        }
    }

    // 4. Transform Object Updates
    simdjson::dom::array objs;
    if (!root["objects"].get(objs)) {
        size_t limit = std::min(scene.world.objects.size(), objs.size());
        size_t idx = 0;
        for (auto o : objs) {
            if (idx >= scene.world.objects.size()) break;
            auto tri = std::dynamic_pointer_cast<Triangle>(scene.world.objects[idx]);
            if (tri) {
                  Transform* th = tri->getTransformPtr(); 
                  if (!th) {
                      auto th_shared = std::make_shared<Transform>();
                      tri->setTransformHandle(th_shared);
                      th = th_shared.get();
                  }

                simdjson::dom::element pivot_offset_el;
                if (!o["pivot_offset"].get(pivot_offset_el)) {
                    th->setPivotOffset(sjsonToVec3(pivot_offset_el), false);
                } else {
                    th->setPivotOffset(Vec3(0, 0, 0), false);
                }
                
                simdjson::dom::element trans;
                if (!o["transform"].get(trans)) {
                    th->setPivotMatrix(sjsonToMat4(trans));
                }
                
                std::string_view obj_name;
                if (!o["name"].get(obj_name)) tri->setNodeName(std::string(obj_name));
                
                int64_t mat_id = 0;
                if (!o["material_id"].get(mat_id)) tri->setMaterialID((int)mat_id);

                tri->updateTransformedVertices();
            }
            idx++;
        }
    }

    // 5. Render Settings
    simdjson::dom::element s;
    if (!root["settings"].get(s)) {
        int64_t q = 0, spp = 1, bounces = 10, diffuse_bounces = 4, transmission_bounces = 8, denoiser_mode = static_cast<int64_t>(DenoiserMode::Quality);
        int64_t denoiser_quality = static_cast<int64_t>(DenoiserQuality::Fast);
        bool adaptive = true, denoiser = false, optix = true, vulkan = false, tonemap = false;
        std::string backend_name;
        std::string_view backend_name_sv;
        
        s["quality_preset"].get(q);
        s["samples_per_pixel"].get(spp);
        s["max_bounces"].get(bounces);
        s["diffuse_bounces"].get(diffuse_bounces);
        s["transmission_bounces"].get(transmission_bounces);
        s["use_adaptive"].get(adaptive);
        s["use_denoiser"].get(denoiser);
        s["denoiser_mode"].get(denoiser_mode);
        s["denoiser_quality"].get(denoiser_quality);
        s["use_optix"].get(optix);
        s["use_vulkan"].get(vulkan);
        if (!s["backend"].get(backend_name_sv)) {
            backend_name = std::string(backend_name_sv);
        }
        s["persistent_tonemap"].get(tonemap);

        if (!backend_name.empty()) {
            std::transform(backend_name.begin(), backend_name.end(), backend_name.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (backend_name == "vulkan") {
                vulkan = true;
                optix = false;
            } else if (backend_name == "optix") {
                optix = true;
                vulkan = false;
            } else if (backend_name == "cpu") {
                optix = false;
                vulkan = false;
            }
        }

        if (vulkan) optix = false;

        settings.quality_preset = (QualityPreset)q;
        settings.samples_per_pixel = (int)spp;
        settings.max_bounces = std::max(1, (int)bounces);
        settings.diffuse_bounces = std::clamp((int)diffuse_bounces, 1, settings.max_bounces);
        settings.transmission_bounces = std::clamp((int)transmission_bounces, 1, settings.max_bounces);
        settings.use_adaptive_sampling = adaptive;
        settings.use_denoiser = denoiser;
        settings.denoiser_mode = static_cast<DenoiserMode>(denoiser_mode);
        settings.denoiser_quality = static_cast<DenoiserQuality>(denoiser_quality);
        settings.use_optix = optix;
        settings.use_vulkan = vulkan;
        settings.persistent_tonemap = tonemap;
        
        // Animation Sequencer Settings
        int64_t anim_start = 0, anim_end = 100, anim_fps = 24;
        std::string_view anim_out_sv;
        if (!s["animation_start_frame"].get(anim_start)) settings.animation_start_frame = (int)anim_start;
        if (!s["animation_end_frame"].get(anim_end)) settings.animation_end_frame = (int)anim_end;
        if (!s["animation_fps"].get(anim_fps)) settings.animation_fps = (int)anim_fps;
        if (!s["animation_output_folder"].get(anim_out_sv)) settings.animation_output_folder = std::string(anim_out_sv);
    }

    // 6. PostFX
    simdjson::dom::element pf;
    if (!root["postfx"].get(pf)) {
        auto& pp = scene.color_processor.params;
        double exp = 1.0, gam = 2.2, sat = 1.0, temp = 6500.0, vig_s = 0.0;
        int64_t tone = 4;
        bool vig_e = false;

        pf["exposure"].get(exp);
        pf["gamma"].get(gam);
        pf["saturation"].get(sat);
        pf["color_temperature"].get(temp);
        pf["tone_mapping"].get(tone);
        pf["vignette_enabled"].get(vig_e);
        pf["vignette_strength"].get(vig_s);

        pp.global_exposure = (float)exp;
        pp.global_gamma = (float)gam;
        pp.saturation = (float)sat;
        pp.color_temperature = (float)temp;
        pp.tone_mapping_type = (ToneMappingType)tone;
        pp.enable_vignette = vig_e;
        pp.vignette_strength = (float)vig_s;
    }

    // 7. World
    simdjson::dom::element w;
    if (!root["world"].get(w)) {
        scene.background_color = sjsonToVec3(w["background_color"]);
    }

    // 8. Timeline (Requires nlohmann::json for now as it's complex)
    simdjson::dom::element t;
    if (!root["timeline"].get(t)) {
        scene.timeline.deserialize(sjsonToNlohmann(t));
    }
    
    // 9. Terrains
    simdjson::dom::element ts;
    if (!root["terrain_system"].get(ts)) {
        // Purge zombie chunks
        auto& objs_list = scene.world.objects;
        size_t purged_count = 0;
        for (auto it = objs_list.begin(); it != objs_list.end(); ) {
            auto tri = std::dynamic_pointer_cast<Triangle>(*it);
            if (tri && tri->getNodeName().find("Terrain_") == 0 && tri->getNodeName().find("_Chunk") != std::string::npos) {
                it = objs_list.erase(it);
                purged_count++;
            } else {
                ++it;
            }
        }
        if (purged_count > 0) SCENE_LOG_INFO("[SceneSerializer] Purged " + std::to_string(purged_count) + " zombie terrain chunks.");

        TerrainManager::getInstance().removeAllTerrains(scene);

        auto abs_path = std::filesystem::absolute(filepath);
        std::string terrainDir = abs_path.parent_path().string();
        
        TerrainManager::getInstance().deserialize(sjsonToNlohmann(ts), terrainDir, scene);
    }
    
    // 9.45 Geometry Node Graphs (Faz 8a Geo-DAG) — structure only (no binary sidecar
    // in this path, so no originalBaseMesh snapshot; see ProjectManager for the full
    // persistence). Not auto-evaluated: the scene's meshes are whatever was saved.
    scene.geometry_node_graphs.clear();
    {
        simdjson::dom::element geoGraphsRoot;
        if (!root["geometry_node_graphs"].get(geoGraphsRoot)) {
            nlohmann::json jGraphs = sjsonToNlohmann(geoGraphsRoot);
            for (auto it = jGraphs.begin(); it != jGraphs.end(); ++it) {
                auto graph = GeometryNodesV2::deserializeGeometryGraph(it.value(), nullptr);
                if (graph) scene.geometry_node_graphs[it.key()] = graph;
            }
        }
    }

    // 9.46 Material Node Graphs (MaterialNodesV2 Faz 1) — re-attached, not auto-applied
    // (the saved material values are the last applied result).
    scene.material_node_graphs.clear();
    {
        simdjson::dom::element matGraphsRoot;
        if (!root["material_node_graphs"].get(matGraphsRoot)) {
            nlohmann::json jGraphs = sjsonToNlohmann(matGraphsRoot);
            for (auto it = jGraphs.begin(); it != jGraphs.end(); ++it) {
                auto graph = MaterialNodesV2::deserializeMaterialGraph(it.value());
                if (graph) scene.material_node_graphs[it.key()] = graph;
            }
            // The FOLD is restored with the material, but the compiled per-pixel program is a
            // RAM-only artifact — rebuild it here, or the graph's per-pixel chains do nothing
            // until the node editor is opened on that material.
            MaterialNodesV2::compileGraphProgramsForScene(scene.material_node_graphs);
        }
    }

    // 9.5 Modifiers
    scene.mesh_modifiers.clear();
    scene.base_mesh_cache.clear();
    simdjson::dom::element modRoot;
    if (!root["mesh_modifiers"].get(modRoot)) {
        // Convert to nlohmann::json for easy parsing using our existing function
        nlohmann::json nlohmannMods = sjsonToNlohmann(modRoot);
        for (auto it = nlohmannMods.begin(); it != nlohmannMods.end(); ++it) {
            std::string nodeName = it.key();
            MeshModifiers::ModifierStack stack;
            stack.deserialize(it.value());
            scene.mesh_modifiers[nodeName] = stack;
        }

        // We must lazily build base_mesh_cache for these modifiers, evaluate them and set new objects
        for(const auto& [nodeName, stack] : scene.mesh_modifiers) {
            std::vector<std::shared_ptr<Triangle>> baseTriangles;
            std::vector<std::shared_ptr<Hittable>> remainingObjects;

            for (const auto& obj : scene.world.objects) {
                if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                    if (tri->getNodeName() == nodeName) {
                        baseTriangles.push_back(tri);
                    } else {
                        remainingObjects.push_back(obj);
                    }
                } else {
                    remainingObjects.push_back(obj);
                }
            }

            if (!baseTriangles.empty() && !stack.modifiers.empty()) {
                scene.base_mesh_cache[nodeName] = baseTriangles;
                const bool hasWater = std::any_of(
                    stack.modifiers.begin(), stack.modifiers.end(),
                    [](const MeshModifiers::ModifierData& modifier) {
                        return modifier.enabled &&
                               modifier.type == MeshModifiers::ModifierType::WaterSurface;
                    });
                const bool hasSimpleSubdivision = std::any_of(
                    stack.modifiers.begin(), stack.modifiers.end(),
                    [](const MeshModifiers::ModifierData& modifier) {
                        return modifier.enabled &&
                               modifier.type == MeshModifiers::ModifierType::FlatSubdivision;
                    });
                std::shared_ptr<TriangleMesh> evaluatedMesh;
                auto newMesh = (hasWater || hasSimpleSubdivision)
                    ? stack.evaluate(baseTriangles, false, &evaluatedMesh)
                    : stack.evaluate(baseTriangles, false);
                if (evaluatedMesh) {
                    evaluatedMesh->nodeName = nodeName;
                    remainingObjects.push_back(evaluatedMesh);
                    if (hasWater) {
                        WaterManager::getInstance().bindExistingWaterMesh(
                            evaluatedMesh, WaterSurface::Type::Plane);
                    }
                } else {
                    for (const auto& tri : newMesh) {
                        remainingObjects.push_back(tri);
                    }
                }
                scene.world.objects = remainingObjects;
            }
        }
    }

    // 9.6 Particle Systems
    simdjson::dom::element psRoot;
    if (!root["particle_systems"].get(psRoot)) {
        nlohmann::json particleSystemsJson = sjsonToNlohmann(psRoot);
        for (const auto& sj : particleSystemsJson) {
            SceneData::ParticleSystemObject system;
            if (sj.contains("id")) system.id = sj["id"];
            if (sj.contains("name")) system.name = sj["name"];
            if (sj.contains("visible")) system.visible = sj["visible"];
            if (sj.contains("enabled")) system.enabled = sj["enabled"];
            if (sj.contains("blend_mode")) system.blend_mode = (SceneData::ParticleBlendMode)sj["blend_mode"];

            // Deserialise Render Settings
            if (sj.contains("render")) {
                const auto& rj = sj["render"];
                if (rj.contains("render_in_raytrace")) system.render.render_in_raytrace = rj["render_in_raytrace"];
                if (rj.contains("shape")) system.render.shape = (SceneData::ParticleRenderShape)rj["shape"];
                if (rj.contains("size_multiplier")) system.render.size_multiplier = rj["size_multiplier"];
                if (rj.contains("sphere_subdivisions")) system.render.sphere_subdivisions = rj["sphere_subdivisions"];
                if (rj.contains("emissive")) system.render.emissive = rj["emissive"];
                if (rj.contains("inherit_color_from_emitter")) system.render.inherit_color_from_emitter = rj["inherit_color_from_emitter"];
                if (rj.contains("base_color")) system.render.base_color = jsonToVec3(rj["base_color"]);
                if (rj.contains("color_end")) system.render.color_end = jsonToVec3(rj["color_end"]);
                if (rj.contains("color_buckets")) system.render.color_buckets = rj["color_buckets"];
                if (rj.contains("over_life_color")) system.render.over_life_color = rj["over_life_color"];
                if (rj.contains("emission_strength")) system.render.emission_strength = rj["emission_strength"];
                if (rj.contains("roughness")) system.render.roughness = rj["roughness"];

                if (rj.contains("mesh_sources")) {
                    system.render.mesh_sources.clear();
                    for (const auto& msj : rj["mesh_sources"]) {
                        SceneData::ParticleRenderMeshSource ms;
                        if (msj.contains("node_name")) ms.node_name = msj["node_name"];
                        if (msj.contains("weight")) ms.weight = msj["weight"];
                        system.render.mesh_sources.push_back(ms);
                    }
                }
            }

            // Deserialise Runtime Settings
            if (sj.contains("runtime")) {
                const auto& rtj = sj["runtime"];
                // Create runtime
                system.runtime = scene.createParticleRuntime();

                // Set runtime parameters
                if (rtj.contains("gravity")) system.runtime->setGravity(jsonToVec3(rtj["gravity"]));
                if (rtj.contains("linear_drag")) system.runtime->setLinearDrag(rtj["linear_drag"]);
                
                float plane_y = 0.0f;
                bool plane_enabled = false;
                float plane_restitution = 0.35f;
                if (rtj.contains("collision_plane_y")) plane_y = rtj["collision_plane_y"];
                if (rtj.contains("collision_plane_enabled")) plane_enabled = rtj["collision_plane_enabled"];
                if (rtj.contains("collision_restitution")) plane_restitution = rtj["collision_restitution"];
                system.runtime->setCollisionPlane(plane_y, plane_enabled, plane_restitution);

                // Physics Settings
                if (rtj.contains("physics_settings")) {
                    const auto& psj = rtj["physics_settings"];
                    auto& ps = system.runtime->physicsSettings();
                    if (psj.contains("mode")) ps.mode = (RayTrophiSim::ParticlePhysicsMode)psj["mode"];
                    if (psj.contains("quality")) ps.quality = (RayTrophiSim::ParticleQualityMode)psj["quality"];
                    if (psj.contains("particle_radius")) ps.particle_radius = psj["particle_radius"];
                    if (psj.contains("self_collision_enabled")) ps.self_collision_enabled = psj["self_collision_enabled"];
                    if (psj.contains("solver_iterations")) ps.solver_iterations = psj["solver_iterations"];
                    if (psj.contains("max_neighbors_per_particle")) ps.max_neighbors_per_particle = psj["max_neighbors_per_particle"];
                    if (psj.contains("viscosity")) ps.viscosity = psj["viscosity"];
                    if (psj.contains("cohesion")) ps.cohesion = psj["cohesion"];
                    if (psj.contains("pressure_stiffness")) ps.pressure_stiffness = psj["pressure_stiffness"];
                    if (psj.contains("rest_density")) ps.rest_density = psj["rest_density"];
                    if (psj.contains("buoyancy")) ps.buoyancy = psj["buoyancy"];
                    if (psj.contains("gravity_scale")) ps.gravity_scale = psj["gravity_scale"];
                    if (psj.contains("vorticity")) ps.vorticity = psj["vorticity"];
                }

                // Emitters
                if (rtj.contains("emitters")) {
                    system.runtime->clearEmitters();
                    for (const auto& ej : rtj["emitters"]) {
                        system.runtime->addEmitter(jsonToEmitter(ej));
                    }
                }

                // Colliders
                if (rtj.contains("colliders")) {
                    system.runtime->clearColliders();
                    for (const auto& cj : rtj["colliders"]) {
                        auto col_desc = jsonToCollider(cj);
                        auto& added_col = system.runtime->addCollider(col_desc);

                        // ObjectConvexDecomp / ObjectMeshBVH are deprecated — the SDF
                        // collider supersedes both. Migrate legacy projects to SDF so
                        // they keep colliding (the enum values are still parsed).
                        if (added_col.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectConvexDecomp ||
                            added_col.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectMeshBVH) {
                            added_col.source_mode = RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF;
                        }

                        // If it's ObjectMeshSDF and has a valid source_name, trigger rebuild.
                        // Pass THIS system's runtime explicitly: during load it is not the
                        // active system yet (active index is set after all systems are read),
                        // so the default active-system lookup would attach the voxel SDF to
                        // the wrong system and the collider would not block fluid on reload.
                        if (added_col.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF && !added_col.source_name.empty()) {
                            scene.rebuildSDFColliderAsync(added_col, system.runtime);
                        }
                    }
                }

                // Grid Domains
                if (rtj.contains("grid_domains")) {
                    system.runtime->clearGridDomains();
                    for (const auto& dj : rtj["grid_domains"]) {
                        system.runtime->addGridDomain(jsonToDomain(dj));
                    }
                    system.runtime->synchronizeGridDomainsNow(); // Force initial state synchronization
                }

                // Flow Sources
                if (rtj.contains("flow_sources")) {
                    system.runtime->clearFlowSources();
                    for (const auto& fsj : rtj["flow_sources"]) {
                        system.runtime->addFlowSource(jsonToFlowSource(fsj));
                    }
                }
            }

            // Sync other run-time domain containers parallel lists to match grid domains count
            if (system.runtime) {
                size_t num_domains = system.runtime->gridDomains().size();
                system.domain_vdb_ids.assign(num_domains, -1);
                system.domain_volumes.assign(num_domains, nullptr);
                system.domain_particle_render_group_ids.assign(num_domains, -1);
                system.domain_particle_pool_capacities.assign(num_domains, 0);
                system.domain_sdf_buffers.resize(num_domains);
                system.domain_sdf_stats.resize(num_domains);
                system.domain_last_fluid_render_mode.assign(num_domains, -1);
            }

            scene.particle_systems.push_back(system);
        }

        // Restore next_particle_system_id so newly added systems have safe IDs
        uint32_t max_id = 0;
        for (const auto& sys : scene.particle_systems) {
            if (sys.id > max_id) max_id = sys.id;
        }
        scene.next_particle_system_id = max_id + 1;
    }

    // Restore active_particle_system_index if present
    int64_t active_sys_idx = -1;
    if (!root["active_particle_system_index"].get(active_sys_idx)) {
        scene.active_particle_system_index = (int)active_sys_idx;
        if (scene.active_particle_system_index >= (int)scene.particle_systems.size()) {
            scene.active_particle_system_index = (int)scene.particle_systems.size() - 1;
        }
    } else {
        scene.active_particle_system_index = scene.particle_systems.empty() ? -1 : 0;
    }

    // 9.7 Simulation bake cache (render-only on-disk point cache)
    // If a "<project>.simcache" folder sits next to the project and its manifest
    // matches the just-loaded systems' config, bind it so the timeline scrubs the
    // baked sim straight from disk (no re-simulation). A config mismatch leaves it
    // unbound — the sim falls back to live resimulation (Phase 2 UI flags it stale).
    {
        const std::string cache_dir = SceneData::simCacheDirForProject(filepath);
        if (!cache_dir.empty() && scene.setSimDiskCache(cache_dir)) {
            SCENE_LOG_INFO("Simulation bake cache bound from: " + cache_dir);
        }
    }

    // 10. Rebuild All
    renderer.rebuildBVH(scene, settings.UI_use_embree);
    
    if (backend) {
        renderer.rebuildBackendGeometry(scene);
        backend->setLights(scene.lights);
        if (scene.camera) {
            renderer.syncCameraToBackend(*scene.camera);
        }
    }

    SCENE_LOG_INFO("Scene loaded with turbo parser from: " + filepath);
    return true;
}
