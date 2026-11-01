#pragma once

#include "Animation/RigEnvelopeWeights.h"
#include "Animation/RigGeodesicWeights.h"
#include "json.hpp"
#include <cstdint>
#include <string>

namespace rtapi {
struct Result;
Result previewRigEnvelopeWeights(const std::string &character,
                                 const RigAuthoring::EnvelopeWeightSettings &settings,
                                 nlohmann::json &output);
Result applyRigEnvelopeWeights(const std::string &character,
                               const RigAuthoring::EnvelopeWeightSettings &settings,
                               uint64_t revision);
Result previewRigGeodesicWeights(const std::string &character,
                                 const RigAuthoring::GeodesicWeightSettings &settings,
                                 nlohmann::json &output);
Result applyRigGeodesicWeights(const std::string &character,
                               const RigAuthoring::GeodesicWeightSettings &settings,
                               uint64_t revision);
Result getRigEnvelopeOverlay(nlohmann::json &output);
Result setRigEnvelopeOverlay(const std::string &character,
                             const RigAuthoring::EnvelopeWeightSettings &settings, bool visible);
Result getRigBoneEnvelope(const std::string &character, const std::string &bone,
                          nlohmann::json &output);
Result applyRigBoneEnvelope(const std::string &character,
                            const RigAuthoring::EnvelopeBoneProfile &profile,
                            uint64_t revision);
} // namespace rtapi
