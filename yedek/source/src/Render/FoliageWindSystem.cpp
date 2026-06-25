#include "FoliageWindSystem.h"

#include "Backend/IBackend.h"
#include "HittableInstance.h"
#include "InstanceManager.h"
#include "globals.h"
#include "scene_data.h"
#include <cmath>
#include <limits>

namespace {

struct WindSignature {
    Vec3 direction = Vec3(1.0f, 0.0f, 0.0f);
    float speed = 1.0f;
    float strength = 0.0f;
    float turbulence = 1.5f;
    float wave_size = 50.0f;
    bool use_source_profiles = false;
};

static bool nearlyEqual(float a, float b, float epsilon = 1e-3f) {
    return std::fabs(a - b) <= epsilon;
}

static bool nearlyEqualVec3(const Vec3& a, const Vec3& b, float epsilon = 1e-3f) {
    return nearlyEqual(a.x, b.x, epsilon) &&
           nearlyEqual(a.y, b.y, epsilon) &&
           nearlyEqual(a.z, b.z, epsilon);
}

static WindSignature makeSignature(const InstanceGroup::WindSettings& settings) {
    WindSignature signature;
    signature.direction = settings.direction.normalize();
    signature.speed = settings.speed;
    signature.strength = settings.strength;
    signature.turbulence = settings.turbulence;
    signature.wave_size = settings.wave_size > 0.1f ? settings.wave_size : 50.0f;
    signature.use_source_profiles = settings.use_source_profiles;
    return signature;
}

static bool sameSignature(const WindSignature& a, const WindSignature& b) {
    return nearlyEqualVec3(a.direction, b.direction) &&
           nearlyEqual(a.speed, b.speed) &&
           nearlyEqual(a.strength, b.strength) &&
           nearlyEqual(a.turbulence, b.turbulence) &&
           nearlyEqual(a.wave_size, b.wave_size) &&
           a.use_source_profiles == b.use_source_profiles;
}

static bool sourceHasCustomWindProfile(const ScatterSource::SourceSettings& settings) {
    return !nearlyEqual(settings.wind_strength_scale, 1.0f) ||
           !nearlyEqual(settings.wind_speed_scale, 1.0f) ||
           !nearlyEqual(settings.wind_turbulence_scale, 1.0f) ||
           !nearlyEqual(settings.wind_bend_limit_scale, 1.0f) ||
           !nearlyEqual(settings.wind_phase_offset, 0.0f);
}

static Vec3 estimateGroupCenter(const InstanceGroup& group) {
    const auto& source_instances = !group.initial_instances.empty() ? group.initial_instances : group.instances;
    if (source_instances.empty()) {
        return Vec3(0.0f);
    }

    Vec3 sum(0.0f);
    for (const auto& inst : source_instances) {
        sum += inst.position;
    }
    return sum / static_cast<float>(source_instances.size());
}

static void matrixToBackendTransform(const Matrix4x4& m, float out[12]) {
    out[0] = m.m[0][0];
    out[1] = m.m[0][1];
    out[2] = m.m[0][2];
    out[3] = m.m[0][3];
    out[4] = m.m[1][0];
    out[5] = m.m[1][1];
    out[6] = m.m[1][2];
    out[7] = m.m[1][3];
    out[8] = m.m[2][0];
    out[9] = m.m[2][1];
    out[10] = m.m[2][2];
    out[11] = m.m[2][3];
}

static bool backendSupportsCudaFoliage(Backend::IBackend* backend) {
    if (!backend) return false;
    return backend->getInfo().type == Backend::BackendType::OPTIX;
}

static bool applyCpuWindToGroup(
    InstanceGroup& group,
    float time,
    Backend::IBackend* backend)
{
    if (!group.wind_settings.enabled || group.instances.empty()) {
        return false;
    }

    if (group.initial_instances.empty() && !group.instances.empty()) {
        group.initial_instances = group.instances;
    }
    if (group.initial_instances.size() != group.instances.size()) {
        group.initial_instances = group.instances;
    }

    const float speed = group.wind_settings.speed;
    const float strength = group.wind_settings.strength;
    const float turbulence = group.wind_settings.turbulence;
    const float wave = group.wind_settings.wave_size > 0.1f ? group.wind_settings.wave_size : 50.0f;
    const Vec3 dir = group.wind_settings.direction.normalize();

    if (dir.length() < 0.001f) {
        return false;
    }

    const float lean_amount = strength * 0.6f;
    const float sway_amount = strength * 0.4f;
    const float max_bend_angle = 25.0f;
    const bool has_active_links = (group.active_hittables.size() == group.instances.size());
    bool group_changed = false;

    for (size_t i = 0; i < group.instances.size(); ++i) {
        const auto& init = group.initial_instances[i];
        auto& curr = group.instances[i];
        const ScatterSource::SourceSettings* source_settings = nullptr;
        if (group.wind_settings.use_source_profiles &&
            init.source_index >= 0 &&
            init.source_index < static_cast<int>(group.sources.size())) {
            source_settings = &group.sources[static_cast<size_t>(init.source_index)].settings;
        }

        const float source_speed = source_settings ? speed * source_settings->wind_speed_scale : speed;
        const float source_strength = source_settings ? strength * source_settings->wind_strength_scale : strength;
        const float source_turbulence = source_settings ? turbulence * source_settings->wind_turbulence_scale : turbulence;
        const float source_bend_limit = source_settings ? max_bend_angle * source_settings->wind_bend_limit_scale : max_bend_angle;
        const float source_phase_offset = source_settings ? source_settings->wind_phase_offset : 0.0f;

        const float pos_phase = (init.position.x * dir.x + init.position.z * dir.z) / wave + source_phase_offset;
        const float t_phase = time * source_speed;

        const float wave_primary = std::sin(pos_phase + t_phase);
        const float wave_secondary = std::sin(pos_phase * 2.3f + t_phase * 1.7f) * 0.35f;
        const float wave_tertiary = std::sin(pos_phase * 4.1f + t_phase * 2.9f * source_turbulence) * 0.15f;
        const float oscillation = (wave_primary + wave_secondary + wave_tertiary) / 1.5f;

        const float source_lean_amount = source_strength * 0.6f;
        const float source_sway_amount = source_strength * 0.4f;
        float final_rot_x = init.rotation.x + dir.z * (source_lean_amount + oscillation * source_sway_amount);
        float final_rot_z = init.rotation.z - dir.x * (source_lean_amount + oscillation * source_sway_amount);

        const float bend_x = final_rot_x - init.rotation.x;
        const float bend_z = final_rot_z - init.rotation.z;
        const float total_bend = std::sqrt(bend_x * bend_x + bend_z * bend_z);
        if (total_bend > source_bend_limit) {
            const float scale = source_bend_limit / total_bend;
            final_rot_x = init.rotation.x + bend_x * scale;
            final_rot_z = init.rotation.z + bend_z * scale;
        }

        curr.rotation.x = final_rot_x;
        curr.rotation.z = final_rot_z;

        const Matrix4x4 new_mat = curr.toMatrix();
        if (has_active_links) {
            if (auto hittable = group.active_hittables[i].lock()) {
                if (auto instance = std::dynamic_pointer_cast<HittableInstance>(hittable)) {
                    instance->setTransform(new_mat);

                    if (backend && !instance->optix_instance_ids.empty()) {
                        float packed[12];
                        matrixToBackendTransform(new_mat, packed);
                        for (int instance_id : instance->optix_instance_ids) {
                            backend->updateInstanceTransform(instance_id, packed);
                        }
                    }
                }
            }
        }

        group_changed = true;
    }

    if (group_changed) {
        group.gpu_dirty = true;
    }
    return group_changed;
}

} // namespace

FoliageWindUpdateStats FoliageWindSystem::update(SceneData& scene, float time, Backend::IBackend* backend) {
    (void)scene;

    FoliageWindUpdateStats stats;
    auto& groups = InstanceManager::getInstance().getGroups();

    bool can_use_cuda_deform = backendSupportsCudaFoliage(backend);
    bool mixed_profiles = false;
    bool has_disabled_populated_group = false;
    bool has_custom_source_profiles = false;
    bool exceeds_gpu_budget = false;
    WindSignature shared_signature;
    bool shared_signature_initialized = false;
    const std::shared_ptr<Camera> active_camera = scene.getActiveCamera();
    const Vec3 camera_position = active_camera ? active_camera->lookfrom : Vec3(0.0f);
    bool has_camera = (active_camera != nullptr);
    int total_gpu_candidate_instances = 0;
    int shared_gpu_instance_budget = std::numeric_limits<int>::max();
    float shared_gpu_distance_budget = std::numeric_limits<float>::max();

    for (const auto& group : groups) {
        if (!group.wind_settings.enabled) {
            if (!group.instances.empty()) {
                has_disabled_populated_group = true;
                can_use_cuda_deform = false;
            }
            continue;
        }

        ++stats.enabled_group_count;
        const WindSignature signature = makeSignature(group.wind_settings);
        if (!shared_signature_initialized) {
            shared_signature = signature;
            shared_signature_initialized = true;
        } else if (!sameSignature(shared_signature, signature)) {
            mixed_profiles = true;
            can_use_cuda_deform = false;
        }

        if (!group.wind_settings.allow_gpu_deform) {
            can_use_cuda_deform = false;
        }

        shared_gpu_instance_budget = (std::min)(shared_gpu_instance_budget,
            (std::max)(group.wind_settings.gpu_deform_max_instances, 1));
        shared_gpu_distance_budget = (std::min)(shared_gpu_distance_budget,
            (std::max)(group.wind_settings.gpu_deform_max_distance, 0.0f));

        total_gpu_candidate_instances += static_cast<int>(group.instances.size());

        if (!has_camera) {
            exceeds_gpu_budget = true;
            can_use_cuda_deform = false;
        } else {
            const Vec3 group_center = estimateGroupCenter(group);
            const float distance_to_camera = (group_center - camera_position).length();
            if (distance_to_camera > shared_gpu_distance_budget) {
                exceeds_gpu_budget = true;
                can_use_cuda_deform = false;
            }
        }

        if (group.wind_settings.use_source_profiles) {
            for (const auto& source : group.sources) {
                if (sourceHasCustomWindProfile(source.settings)) {
                    has_custom_source_profiles = true;
                    can_use_cuda_deform = false;
                    break;
                }
            }
        }
    }

    if (total_gpu_candidate_instances > shared_gpu_instance_budget) {
        exceeds_gpu_budget = true;
        can_use_cuda_deform = false;
    }

    for (auto& group : groups) {
        if (applyCpuWindToGroup(group, time, backend)) {
            stats.any_cpu_update = true;
        }
    }

    stats.used_cpu_fallback = stats.any_cpu_update;

    if (can_use_cuda_deform && shared_signature_initialized && stats.enabled_group_count > 0) {
        backend->setWindParams(
            shared_signature.direction,
            shared_signature.strength / 25.0f,
            shared_signature.speed,
            time);
        stats.gpu_deform_applied = true;
    } else if (backendSupportsCudaFoliage(backend) && (mixed_profiles || has_disabled_populated_group || has_custom_source_profiles || exceeds_gpu_budget)) {
        static bool logged_mixed_profile_warning = false;
        if (!logged_mixed_profile_warning) {
            SCENE_LOG_WARN("[FoliageWind] CUDA deform disabled because foliage groups exceed safe hero-foliage limits or do not share one safe global wind profile. CPU fallback remains active.");
            logged_mixed_profile_warning = true;
        }
    }

    return stats;
}
