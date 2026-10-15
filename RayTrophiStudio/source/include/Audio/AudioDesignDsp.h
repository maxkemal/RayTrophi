#pragma once

#include "Audio/AudioPcm.h"
#include "Audio/AudioDesignJobs.h"

namespace audio {
nlohmann::json designDspCatalogue();
void validateDesignDsp(const nlohmann::json& input, nlohmann::json& recipe);
AudioPcm renderDesignLayers(const nlohmann::json& recipe, const nlohmann::json& assets,
                            std::vector<float>& envelope, const AudioCancelCheck& check = {});
void applyDesignEffects(AudioPcm& pcm, const nlohmann::json& effects,
                        const AudioCancelCheck& check = {});
void applyDesignSpectralRegion(AudioPcm& pcm, const nlohmann::json& effect,
                              const AudioCancelCheck& check = {});
nlohmann::json designEqResponse(const nlohmann::json& effects);
nlohmann::json designEffectVisual(const nlohmann::json& effect);
void validateDesignAnalysis(const AudioPcm& pcm, const nlohmann::json& params);
nlohmann::json analyzeDesignPcm(const AudioPcm& pcm, const nlohmann::json& params =
                               nlohmann::json::object(),
                               const AudioCancelCheck& check = {});
nlohmann::json designPcmWaveform(const AudioPcm& pcm, std::uint64_t first,
                                std::uint64_t last, unsigned points,
                                const AudioCancelCheck& check = {});
} // namespace audio
