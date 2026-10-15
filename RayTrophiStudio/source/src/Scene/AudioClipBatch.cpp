#include "Audio/AudioClipBatch.h"
#include "Audio/AudioClips.h"
#include "Audio/AudioTracks.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace audio {
namespace {
using json = nlohmann::json;
void require(bool valid, const char* message) {
    if (!valid) {
        throw std::invalid_argument(message);
    }
}
double number(const json& input, const char* field, double fallback) {
    const auto value = input.value(field, json(fallback));
    require(value.is_number(), "Batch time values must be numeric");
    const double result = value.get<double>();
    require(std::isfinite(result) && std::abs(result) <= 86400,
            "Batch time value outside finite one-day bounds");
    return result;
}
} // namespace

json AudioClipBatch::mutate(json& scene, const json& params) {
    const auto& input = params.at("ids");
    require(input.is_array() && !input.empty() && input.size() <= 1024,
            "Batch requires 1..1024 unique clip IDs");
    std::set<std::string> ids;
    for (const auto& id : input) {
        require(id.is_string() && ids.insert(id.get<std::string>()).second,
                "Batch IDs must be unique strings");
    }
    const auto action = params.at("action").get<std::string>();
    require(action == "move" || action == "duplicate" || action == "remove" ||
            action == "align_start" || action == "align_end" || action == "update",
            "Unknown clip batch action");
    require(!params.contains("delta_seconds") || action == "move" || action == "duplicate",
            "delta_seconds applies only to move/duplicate");
    require(!params.contains("time_seconds") || action == "align_start" || action == "align_end",
            "time_seconds applies only to alignment");
    require(!params.contains("patch") || action == "update", "patch applies only to update");
    json candidate = scene;
    json selected = json::array();
    double first = 86400, last = 0;
    for (const auto& clip : candidate.at("clips")) {
        if (ids.contains(clip.at("id").get<std::string>())) {
            selected.push_back(clip);
            first = std::min(first, clip.at("start_seconds").get<double>());
            last = std::max(last, clip.at("start_seconds").get<double>() +
                                  clip.at("duration_seconds").get<double>());
        }
    }
    require(selected.size() == ids.size(), "Batch references an unknown clip");
    const double delta = number(params, "delta_seconds", action == "duplicate" ? last - first : 0);
    const double time = number(params, "time_seconds", 0);
    json patch = params.value("patch", json::object());
    if (action == "update") {
        require(patch.is_object(), "Batch patch must be an object");
        for (auto field = patch.begin(); field != patch.end(); ++field) {
            require(field.key() == "gain" || field.key() == "pan" || field.key() == "muted" ||
                    field.key() == "bus" || field.key() == "track_id",
                    "Unsupported shared batch property");
        }
    }
    json resultIds = json::array();
    if (action == "remove") {
        auto& clips = candidate["clips"];
        clips.erase(std::remove_if(clips.begin(), clips.end(), [&](const json& clip) {
            return ids.contains(clip.at("id").get<std::string>());
        }), clips.end());
    } else {
        for (auto clip : selected) {
            if (action == "move" || action == "duplicate") {
                clip["start_seconds"] = clip.at("start_seconds").get<double>() + delta;
            } else if (action == "align_start") {
                clip["start_seconds"] = time;
            } else if (action == "align_end") {
                clip["start_seconds"] = time - clip.at("duration_seconds").get<double>();
            } else {
                clip.update(patch);
            }
            if (action == "duplicate") {
                const auto next = candidate.at("next_id").get<std::uint64_t>();
                require(next > 0 && next < 9007199254740991ULL, "Audio ID counter exhausted");
                candidate["next_id"] = next + 1;
                clip["id"] = "audio_clip_" + std::to_string(next);
                candidate["clips"].push_back(clip);
            } else {
                for (auto& entry : candidate["clips"]) {
                    if (entry.at("id") == clip.at("id")) {
                        entry = clip;
                        break;
                    }
                }
            }
            resultIds.push_back(clip.at("id"));
        }
    }
    AudioTracks::bindClips(candidate);
    candidate["clips"] = AudioClips::validateCollection(candidate.at("clips"), candidate.at("assets"));
    scene = std::move(candidate);
    return {{"ids", resultIds}, {"count", selected.size()}, {"action", action}};
}
} // namespace audio
