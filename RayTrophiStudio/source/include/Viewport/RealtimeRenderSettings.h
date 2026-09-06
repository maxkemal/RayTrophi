#pragma once

// ============================================================================
// DURUM: PARK EDILDI - HICBIR SEY BUNU CAGIRMIYOR (2026-08-31)
//
// Ayri "Realtime" viewport modu 2026-08-31'de SOKULDU. Gerekce ve olcumler:
//   docs/dev/REALTIME_RENDERER_ROADMAP.md
//
// Bu dosya bilerek birakildi: sokulmus olan RENDER YOLUYDU, bu ise yalnizca
// bir DEGER/DOGRULAMA sozlesmesi -- kalite profilleri ve sinirlari. Ikinci bir
// canli kod yolu olusturmaz cunku hicbir cagirani yoktur; derlenmeye devam
// etmesi bit-rot'u onlemek icindir.
//
// ★ Buraya bakan bir sonraki kisi icin: bu tipler CANLI DEGIL. Bir alanini
// okuyup "ayar boyle" diye rapor eden bir sey yazma -- varsayilanlari
// dondurur, ve varsayilan bir olcum degildir. Kullanilacagi yer, gercek sahne
// isik/golge secenegi material preview yoluna eklendiginde orasidir.
// ============================================================================

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace ViewportRealtime {

enum class QualityProfile : std::uint8_t {
    Auto = 0,
    Performance,
    Balanced,
    Quality,
    Cinematic,
    Custom
};

enum class IndirectLightingMode : std::uint8_t {
    Off = 0,
    ScreenSpace,
    ProbeGrid,
    HybridRayQuery
};

enum class TransmissionMode : std::uint8_t {
    Off = 0,
    ScreenSpace,
    HybridRayQuery
};

enum class VolumeQuality : std::uint8_t {
    Off = 0,
    Performance,
    Balanced,
    Quality
};

enum class InstanceFidelity : std::uint8_t {
    AdaptiveLod = 0,
    Full
};

// Renderer-facing settings only. UI, scripting and IPC must translate names at
// their boundary and call the same SettingsService::apply operation.
struct RenderSettings {
    QualityProfile profile = QualityProfile::Balanced;

    std::uint32_t shadowAtlasSize = 4096;
    std::uint32_t directionalCascades = 4;
    std::uint32_t maxVisibleLights = 256;
    std::uint32_t maxShadowedLights = 8;
    std::uint32_t clusterTileSize = 16;
    std::uint32_t clusterDepthSlices = 24;

    IndirectLightingMode indirectLighting = IndirectLightingMode::ScreenSpace;
    TransmissionMode transmission = TransmissionMode::ScreenSpace;
    VolumeQuality volumes = VolumeQuality::Balanced;
    InstanceFidelity instanceFidelity = InstanceFidelity::AdaptiveLod;

    bool shadows = true;
    bool ambientOcclusion = true;
    bool reflections = true;
    bool temporalAA = true;
    bool halfResolutionScreenSpace = true;
    bool halfResolutionVolumes = true;
};

struct ValidationResult {
    bool ok = false;
    std::string error;
    std::vector<std::string> warnings;
};

struct SettingsSnapshot {
    RenderSettings settings;
    std::uint64_t revision = 0;
};

RenderSettings settingsForProfile(QualityProfile profile);
ValidationResult validateSettings(const RenderSettings& settings);
const char* qualityProfileName(QualityProfile profile);
const char* indirectLightingModeName(IndirectLightingMode mode);
const char* transmissionModeName(TransmissionMode mode);
const char* volumeQualityName(VolumeQuality quality);
const char* instanceFidelityName(InstanceFidelity fidelity);

class SettingsService final {
public:
    SettingsService();

    SettingsSnapshot snapshot() const;

    // Validation is strict: invalid budgets are rejected rather than silently
    // clamped. The returned revision changes only when the effective settings do.
    ValidationResult apply(const RenderSettings& settings,
                           std::uint64_t* resultingRevision = nullptr);

    ValidationResult applyProfile(QualityProfile profile,
                                  std::uint64_t* resultingRevision = nullptr);

private:
    mutable std::mutex mutex_;
    RenderSettings settings_;
    std::uint64_t revision_ = 1;
};

SettingsService& settingsService();

} // namespace ViewportRealtime
