// ParticleSimulationSystem: appearance profile ownership (particle roadmap
// Phase 1.5 Batch A). Kept out of ParticleSimulation.cpp, which is far past
// the 2000-line limit.

#include "ParticleSimulation.h"
#include "globals.h"

#include <algorithm>

namespace RayTrophiSim {

const std::vector<ParticleAppearanceProfile>& ParticleSimulationSystem::appearanceProfiles() const {
    return appearance_profiles_;
}

const ParticleAppearanceProfile* ParticleSimulationSystem::findAppearanceProfile(
    uint32_t id) const {
    if (id == 0) {
        return nullptr;
    }
    for (const auto& profile : appearance_profiles_) {
        if (profile.id == id) {
            return &profile;
        }
    }
    return nullptr;
}

const float* ParticleSimulationSystem::appearanceLutRow(uint32_t id) const {
    if (id != 0) {
        for (std::size_t i = 0; i < appearance_profiles_.size(); ++i) {
            if (appearance_profiles_[i].id == id && i < appearance_luts_.size()) {
                return appearance_luts_[i].data();
            }
        }
    }
    static const std::vector<float> fallback_lut = [] {
        std::vector<float> lut;
        bakeParticleAppearanceLut(fallbackParticleAppearance(), lut);
        return lut;
    }();
    return fallback_lut.data();
}

uint32_t ParticleSimulationSystem::addAppearanceProfile(
    const ParticleAppearanceProfile& profile, std::string* error) {
    ParticleAppearanceProfile copy = profile;
    const std::string problem = normalizeParticleAppearanceProfile(copy);
    if (!problem.empty()) {
        if (error) *error = problem;
        return 0;
    }
    if (copy.id == 0 || findAppearanceProfile(copy.id) != nullptr) {
        copy.id = next_appearance_profile_id_;
    }
    next_appearance_profile_id_ = std::max(next_appearance_profile_id_, copy.id + 1u);

    std::vector<float> lut;
    bakeParticleAppearanceLut(copy, lut);
    appearance_profiles_.push_back(std::move(copy));
    appearance_luts_.push_back(std::move(lut));
    return appearance_profiles_.back().id;
}

bool ParticleSimulationSystem::updateAppearanceProfile(
    const ParticleAppearanceProfile& profile, std::string* error) {
    for (std::size_t i = 0; i < appearance_profiles_.size(); ++i) {
        if (appearance_profiles_[i].id != profile.id) {
            continue;
        }
        ParticleAppearanceProfile copy = profile;
        const std::string problem = normalizeParticleAppearanceProfile(copy);
        if (!problem.empty()) {
            if (error) *error = problem;
            return false;
        }
        bakeParticleAppearanceLut(copy, appearance_luts_[i]);
        appearance_profiles_[i] = std::move(copy);
        return true;
    }
    if (error) *error = "no appearance profile with id " + std::to_string(profile.id);
    return false;
}

bool ParticleSimulationSystem::removeAppearanceProfile(uint32_t id, std::string* error) {
    for (const auto& emitter : emitters_) {
        if (emitter.appearance_profile_id == id) {
            if (error) {
                *error = "appearance profile " + std::to_string(id) +
                         " is still used by emitter '" + emitter.name + "'";
            }
            return false;
        }
    }
    for (std::size_t i = 0; i < appearance_profiles_.size(); ++i) {
        if (appearance_profiles_[i].id == id) {
            appearance_profiles_.erase(appearance_profiles_.begin() +
                                       static_cast<std::ptrdiff_t>(i));
            if (i < appearance_luts_.size()) {
                appearance_luts_.erase(appearance_luts_.begin() +
                                       static_cast<std::ptrdiff_t>(i));
            }
            return true;
        }
    }
    if (error) *error = "no appearance profile with id " + std::to_string(id);
    return false;
}

uint32_t ParticleSimulationSystem::findOrAddAppearanceProfile(
    const ParticleAppearanceProfile& profile) {
    for (const auto& existing : appearance_profiles_) {
        if (existing.name == profile.name) {
            return existing.id;
        }
    }
    ParticleAppearanceProfile copy = profile;
    copy.id = 0;
    return addAppearanceProfile(copy);
}

ParticleAppearanceSample ParticleSimulationSystem::sampleAppearance(std::size_t index) const {
    if (index >= buffers_.alive.size()) {
        return sampleParticleAppearanceLut(appearanceLutRow(0), 0.0f);
    }
    const float lifetime = buffers_.lifetime_seconds[index];
    const float t = lifetime > 1e-6f ? buffers_.age_seconds[index] / lifetime : 0.0f;
    const uint32_t id = index < buffers_.appearance_profile.size()
        ? buffers_.appearance_profile[index] : 0u;
    ParticleAppearanceSample s = sampleParticleAppearanceLut(appearanceLutRow(id), t);
    if (index < buffers_.size_scale.size()) {
        s.size *= buffers_.size_scale[index];
    }
    return s;
}

void ParticleSimulationSystem::setNextAppearanceProfileId(uint32_t next) {
    uint32_t floor = 1;
    for (const auto& profile : appearance_profiles_) {
        floor = std::max(floor, profile.id + 1u);
    }
    next_appearance_profile_id_ = std::max(next, floor);
}

// ── Serialization (shared by ProjectManager and SceneSerializer) ─────────────

void serializeParticleAppearanceProfiles(const ParticleSimulationSystem& system,
                                         nlohmann::json& system_json) {
    nlohmann::json profiles = nlohmann::json::array();
    for (const auto& profile : system.appearanceProfiles()) {
        profiles.push_back(serializeParticleAppearanceProfile(profile));
    }
    system_json["appearance_profiles"] = std::move(profiles);
    system_json["next_appearance_profile_id"] = system.nextAppearanceProfileId();
}

void deserializeParticleAppearanceProfiles(const nlohmann::json& system_json,
                                           ParticleSimulationSystem& system) {
    if (system_json.contains("appearance_profiles") &&
        system_json["appearance_profiles"].is_array()) {
        for (const auto& item : system_json["appearance_profiles"]) {
            ParticleAppearanceProfile profile;
            if (!deserializeParticleAppearanceProfile(item, profile)) {
                continue;
            }
            std::string error;
            const uint32_t id = system.addAppearanceProfile(profile, &error);
            if (id != profile.id) {
                // Emitters reference the saved id; a renumbered or rejected
                // profile would silently re-point them.
                SCENE_LOG_WARN("[Particles] Appearance profile '" + profile.name + "' (id " +
                               std::to_string(profile.id) + ") could not be restored" +
                               (error.empty() ? std::string(" with its id")
                                              : std::string(": ") + error));
            }
        }
    }
    system.setNextAppearanceProfileId(
        system_json.value("next_appearance_profile_id", system.nextAppearanceProfileId()));
}

void serializeEmitterAppearance(const ParticleEmitterDesc& emitter, nlohmann::json& emitter_json) {
    emitter_json["appearance_profile_id"] = emitter.appearance_profile_id;
}

void readEmitterAppearance(const nlohmann::json& emitter_json, int legacy_blend,
                           ParticleSimulationSystem& system, ParticleEmitterDesc& emitter) {
    if (emitter_json.contains("appearance_profile_id")) {
        emitter.appearance_profile_id = emitter_json.value("appearance_profile_id", 0u);
        return;
    }
    emitter.appearance_profile_id = system.addAppearanceProfile(
        migrateLegacyEmitterAppearance(emitter_json, legacy_blend, emitter.name));
}

void ParticleSimulationSystem::ensureEmitterAppearanceProfile(ParticleEmitterDesc& emitter) {
    if (emitter.appearance_profile_id != 0 &&
        findAppearanceProfile(emitter.appearance_profile_id) != nullptr) {
        return;
    }
    if (emitter.appearance_profile_id != 0) {
        SCENE_LOG_WARN("[Particles] Emitter '" + emitter.name +
                       "' referenced missing appearance profile " +
                       std::to_string(emitter.appearance_profile_id) +
                       "; it gets a default profile of its own.");
    }
    // The pre-profile emitter defaults, so a new emitter looks as it always did.
    const ParticleAppearanceProfile profile = makeTwoKeyParticleAppearance(
        (emitter.name.empty() ? std::string("Emitter") : emitter.name) + " Appearance",
        ParticleAppearanceBlend::Additive,
        0.06f, 0.02f, 1.0f, 0.0f,
        Vec3(1.0f, 0.85f, 0.5f), Vec3(1.0f, 0.25f, 0.08f));
    emitter.appearance_profile_id = addAppearanceProfile(profile);
}

} // namespace RayTrophiSim
