#pragma once

#include "Audio/AudioReaction.h"
#include "json.hpp"

namespace audio {
// One resolved world pose/control state for playback, acoustic queries and diagnostics.
// The reaction position is already world-space; never parent-transform it again.
nlohmann::json resolveAudioSourceRuntime(const nlohmann::json& emitter,
    const nlohmann::json* resolvedPose, const ReactionOutput* control);
}
