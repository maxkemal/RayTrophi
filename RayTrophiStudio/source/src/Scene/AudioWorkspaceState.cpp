#include "Audio/AudioWorkspaceState.h"
#include <stdexcept>

namespace audio {
using json = nlohmann::json;
json defaultAudioWorkspace() {
    return {{"selected_clip", ""}, {"snap", true}, {"waveforms", true},
            {"inspector_open", false}, {"designer_open", false}};
}

json updateAudioWorkspace(const json& current, const json& patch, const json& clips) {
    auto candidate = current;
    for (auto entry = patch.begin(); entry != patch.end(); ++entry) {
        if (entry.key() == "selected_clip") {
            if (!entry.value().is_string()) {
                throw std::invalid_argument("selected_clip must be a string");
            }
            const auto id = entry.value().get<std::string>();
            bool found = id.empty();
            for (const auto& clip : clips) {
                found = found || clip.at("id") == id;
            }
            if (!found) {
                throw std::invalid_argument("Selected audio clip does not exist");
            }
        } else if (entry.key() == "snap" || entry.key() == "waveforms" ||
                   entry.key() == "inspector_open" || entry.key() == "designer_open") {
            if (!entry.value().is_boolean()) {
                throw std::invalid_argument("Audio workspace toggles must be boolean");
            }
        } else {
            throw std::invalid_argument("Unknown audio workspace parameter");
        }
        candidate[entry.key()] = entry.value();
    }
    return candidate;
}
}
