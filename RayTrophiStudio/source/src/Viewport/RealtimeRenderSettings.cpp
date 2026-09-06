#include "Viewport/RealtimeRenderSettings.h"

namespace ViewportRealtime {
namespace {

bool isPowerOfTwo(std::uint32_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

bool settingsEqual(const RenderSettings& a, const RenderSettings& b) {
    return a.profile == b.profile &&
           a.shadowAtlasSize == b.shadowAtlasSize &&
           a.directionalCascades == b.directionalCascades &&
           a.maxVisibleLights == b.maxVisibleLights &&
           a.maxShadowedLights == b.maxShadowedLights &&
           a.clusterTileSize == b.clusterTileSize &&
           a.clusterDepthSlices == b.clusterDepthSlices &&
           a.indirectLighting == b.indirectLighting &&
           a.transmission == b.transmission &&
           a.volumes == b.volumes &&
           a.instanceFidelity == b.instanceFidelity &&
           a.shadows == b.shadows &&
           a.ambientOcclusion == b.ambientOcclusion &&
           a.reflections == b.reflections &&
           a.temporalAA == b.temporalAA &&
           a.halfResolutionScreenSpace == b.halfResolutionScreenSpace &&
           a.halfResolutionVolumes == b.halfResolutionVolumes;
}

} // namespace

RenderSettings settingsForProfile(QualityProfile profile) {
    RenderSettings settings;
    settings.profile = profile;

    switch (profile) {
        case QualityProfile::Performance:
            settings.shadowAtlasSize = 2048;
            settings.directionalCascades = 2;
            settings.maxVisibleLights = 128;
            settings.maxShadowedLights = 4;
            settings.clusterTileSize = 16;
            settings.clusterDepthSlices = 16;
            settings.indirectLighting = IndirectLightingMode::ScreenSpace;
            settings.transmission = TransmissionMode::ScreenSpace;
            settings.volumes = VolumeQuality::Performance;
            settings.halfResolutionScreenSpace = true;
            settings.halfResolutionVolumes = true;
            settings.instanceFidelity = InstanceFidelity::AdaptiveLod;
            break;
        case QualityProfile::Quality:
            settings.shadowAtlasSize = 4096;
            settings.directionalCascades = 4;
            settings.maxVisibleLights = 512;
            settings.maxShadowedLights = 16;
            settings.clusterTileSize = 16;
            settings.clusterDepthSlices = 32;
            settings.indirectLighting = IndirectLightingMode::ProbeGrid;
            settings.transmission = TransmissionMode::ScreenSpace;
            settings.volumes = VolumeQuality::Quality;
            settings.halfResolutionScreenSpace = false;
            settings.halfResolutionVolumes = true;
            settings.instanceFidelity = InstanceFidelity::Full;
            break;
        case QualityProfile::Cinematic:
            settings.shadowAtlasSize = 8192;
            settings.directionalCascades = 4;
            settings.maxVisibleLights = 1024;
            settings.maxShadowedLights = 32;
            settings.clusterTileSize = 8;
            settings.clusterDepthSlices = 32;
            settings.indirectLighting = IndirectLightingMode::HybridRayQuery;
            settings.transmission = TransmissionMode::HybridRayQuery;
            settings.volumes = VolumeQuality::Quality;
            settings.halfResolutionScreenSpace = false;
            settings.halfResolutionVolumes = false;
            settings.instanceFidelity = InstanceFidelity::Full;
            break;
        case QualityProfile::Auto:
        case QualityProfile::Balanced:
        case QualityProfile::Custom:
        default:
            // Auto starts at the bounded Balanced baseline. Runtime telemetry may
            // later select another complete validated profile.
            settings.shadowAtlasSize = 4096;
            settings.directionalCascades = 4;
            settings.maxVisibleLights = 256;
            settings.maxShadowedLights = 8;
            settings.clusterTileSize = 16;
            settings.clusterDepthSlices = 24;
            settings.indirectLighting = IndirectLightingMode::ScreenSpace;
            settings.transmission = TransmissionMode::ScreenSpace;
            settings.volumes = VolumeQuality::Balanced;
            settings.halfResolutionScreenSpace = true;
            settings.halfResolutionVolumes = true;
            settings.instanceFidelity = InstanceFidelity::AdaptiveLod;
            break;
    }
    return settings;
}

ValidationResult validateSettings(const RenderSettings& settings) {
    ValidationResult result;

    if (!isPowerOfTwo(settings.shadowAtlasSize) ||
        settings.shadowAtlasSize < 1024 || settings.shadowAtlasSize > 8192) {
        result.error = "shadowAtlasSize must be a power of two in [1024, 8192]";
        return result;
    }
    if (settings.directionalCascades < 1 || settings.directionalCascades > 4) {
        result.error = "directionalCascades must be in [1, 4]";
        return result;
    }
    if (settings.maxVisibleLights < 1 || settings.maxVisibleLights > 4096) {
        result.error = "maxVisibleLights must be in [1, 4096]";
        return result;
    }
    if (settings.maxShadowedLights > 64) {
        result.error = "maxShadowedLights must be in [0, 64]";
        return result;
    }
    if (settings.maxShadowedLights > settings.maxVisibleLights) {
        result.error = "maxShadowedLights cannot exceed maxVisibleLights";
        return result;
    }
    if (settings.clusterTileSize != 8 && settings.clusterTileSize != 16 &&
        settings.clusterTileSize != 32) {
        result.error = "clusterTileSize must be 8, 16 or 32";
        return result;
    }
    if (settings.clusterDepthSlices < 8 || settings.clusterDepthSlices > 64) {
        result.error = "clusterDepthSlices must be in [8, 64]";
        return result;
    }
    if (!settings.shadows && settings.maxShadowedLights != 0) {
        result.warnings.emplace_back(
            "shadows are disabled; maxShadowedLights is retained but inactive");
    }
    if (!settings.temporalAA &&
        (settings.indirectLighting != IndirectLightingMode::Off ||
         settings.volumes != VolumeQuality::Off)) {
        result.warnings.emplace_back(
            "temporalAA is disabled; temporal GI and volume stability will be limited");
    }

    result.ok = true;
    return result;
}

const char* qualityProfileName(QualityProfile profile) {
    switch (profile) {
        case QualityProfile::Auto: return "auto";
        case QualityProfile::Performance: return "performance";
        case QualityProfile::Balanced: return "balanced";
        case QualityProfile::Quality: return "quality";
        case QualityProfile::Cinematic: return "cinematic";
        case QualityProfile::Custom: return "custom";
    }
    return "unknown";
}

const char* indirectLightingModeName(IndirectLightingMode mode) {
    switch (mode) {
        case IndirectLightingMode::Off: return "off";
        case IndirectLightingMode::ScreenSpace: return "screen_space";
        case IndirectLightingMode::ProbeGrid: return "probe_grid";
        case IndirectLightingMode::HybridRayQuery: return "hybrid_ray_query";
    }
    return "unknown";
}

const char* transmissionModeName(TransmissionMode mode) {
    switch (mode) {
        case TransmissionMode::Off: return "off";
        case TransmissionMode::ScreenSpace: return "screen_space";
        case TransmissionMode::HybridRayQuery: return "hybrid_ray_query";
    }
    return "unknown";
}

const char* volumeQualityName(VolumeQuality quality) {
    switch (quality) {
        case VolumeQuality::Off: return "off";
        case VolumeQuality::Performance: return "performance";
        case VolumeQuality::Balanced: return "balanced";
        case VolumeQuality::Quality: return "quality";
    }
    return "unknown";
}

const char* instanceFidelityName(InstanceFidelity fidelity) {
    switch (fidelity) {
        case InstanceFidelity::AdaptiveLod: return "adaptive_lod";
        case InstanceFidelity::Full: return "full";
    }
    return "unknown";
}

SettingsService::SettingsService()
    : settings_(settingsForProfile(QualityProfile::Balanced)) {}

SettingsSnapshot SettingsService::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return SettingsSnapshot{settings_, revision_};
}

ValidationResult SettingsService::apply(const RenderSettings& settings,
                                        std::uint64_t* resultingRevision) {
    ValidationResult result = validateSettings(settings);
    if (!result.ok) return result;

    std::lock_guard<std::mutex> lock(mutex_);
    if (!settingsEqual(settings_, settings)) {
        settings_ = settings;
        ++revision_;
    }
    if (resultingRevision) *resultingRevision = revision_;
    return result;
}

ValidationResult SettingsService::applyProfile(QualityProfile profile,
                                               std::uint64_t* resultingRevision) {
    return apply(settingsForProfile(profile), resultingRevision);
}

SettingsService& settingsService() {
    static SettingsService service;
    return service;
}

} // namespace ViewportRealtime
