#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "Audio/AudioTracks.h"
#include "Audio/AudioClips.h"
#include "Audio/AudioMixer.h"
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
json& find(json& tracks, const std::string& id) {
    for (auto& track : tracks) {
        if (track.at("id").get<std::string>() == id) {
            return track;
        }
    }
    throw std::invalid_argument("Unknown audio track id");
}
} // namespace

std::vector<std::string> AudioTracks::parameters(const std::string& operation) {
    if (operation == "track_list") {
        return {};
    }
    if (operation == "track_reorder") {
        return {"ids"};
    }
    if (operation == "track_remove") {
        return {"id", "clip_policy"};
    }
    if (operation == "track_create") {
        return {"name", "bus", "gain", "pan", "muted", "solo"};
    }
    return {"id", "name", "bus", "gain", "pan", "muted", "solo"};
}

json AudioTracks::validate(const json& tracks) {
    require(tracks.is_array() && tracks.size() <= 128, "Audio supports at most 128 tracks");
    json normalized = json::array();
    std::set<std::string> ids;
    const auto fields = parameters("track_update");
    for (const auto& input : tracks) {
        require(input.is_object(), "Track must be an object");
        for (auto field = input.begin(); field != input.end(); ++field) {
            require(std::find(fields.begin(), fields.end(), field.key()) != fields.end(),
                    "Unknown audio track field");
        }
        const auto id = input.at("id").get<std::string>();
        require(!id.empty() && ids.insert(id).second, "Track IDs must be nonempty and unique");
        const auto name = input.value("name", std::string("Audio Track"));
        require(!name.empty() && name.size() <= 256, "Track name must contain 1..256 bytes");
        const auto bus = input.value("bus", std::string("Effects"));
        const auto& buses = AudioMixer::names();
        require(std::find(buses.begin(), buses.end(), bus) != buses.end(), "Unknown track bus");
        json track = {{"id", id}, {"name", name}, {"bus", bus}};
        for (const char* field : {"gain", "pan"}) {
            const auto value = input.value(field, json(std::string(field) == "gain" ? 1 : 0));
            require(value.is_number(), "Track gain/pan must be numeric");
            const double number = value.get<double>();
            const bool gain = std::string(field) == "gain";
            require(std::isfinite(number) && number >= (gain ? 0 : -1) &&
                    number <= (gain ? 16 : 1), "Track gain/pan outside finite bounds");
            track[field] = number;
        }
        for (const char* field : {"muted", "solo"}) {
            const auto value = input.value(field, json(false));
            require(value.is_boolean(), "Track mute/solo must be boolean");
            track[field] = value;
        }
        normalized.push_back(track);
    }
    return normalized;
}

void AudioTracks::bindClips(json& scene) {
    if (!scene.contains("tracks")) {
        scene["tracks"] = json::array();
    }
    auto& tracks = scene["tracks"];
    for (auto& clip : scene["clips"]) {
        const auto id = clip.value("track_id", std::string());
        clip["track_id"] = id;
        if (!id.empty()) {
            clip["bus"] = find(tracks, id).at("bus");
        }
    }
}

json AudioTracks::mutate(json& scene, const std::string& operation, const json& params) {
    json candidate = scene;
    auto& tracks = candidate["tracks"];
    json result;
    if (operation == "track_create") {
        require(!params.contains("id"), "Track ID is allocated by the service");
        const auto next = candidate.at("next_id").get<std::uint64_t>();
        require(next > 0 && next < 9007199254740991ULL, "Audio id counter exhausted");
        auto track = params;
        track["id"] = "audio_track_" + std::to_string(next);
        candidate["next_id"] = next + 1;
        tracks.push_back(track);
        tracks = validate(tracks);
        result = tracks.back();
    } else if (operation == "track_update") {
        auto& track = find(tracks, params.at("id").get<std::string>());
        for (auto field = params.begin(); field != params.end(); ++field) {
            track[field.key()] = field.value();
        }
        const auto id = track.at("id").get<std::string>();
        tracks = validate(tracks);
        result = find(tracks, id);
    } else if (operation == "track_reorder") {
        const auto& order = params.at("ids");
        require(order.is_array() && order.size() == tracks.size(), "Track order must cover all IDs");
        json reordered = json::array();
        std::set<std::string> seen;
        for (const auto& id : order) {
            require(id.is_string() && seen.insert(id.get<std::string>()).second,
                    "Track order IDs must be unique strings");
            reordered.push_back(find(tracks, id.get<std::string>()));
        }
        tracks = std::move(reordered);
        result = tracks;
    } else {
        require(operation == "track_remove", "Unknown track operation");
        const auto id = params.at("id").get<std::string>();
        find(tracks, id);
        const auto policy = params.value("clip_policy", std::string("reject"));
        require(policy == "reject" || policy == "detach" || policy == "delete",
                "clip_policy must be reject, detach or delete");
        auto& clips = candidate["clips"];
        for (auto& clip : clips) {
            if (clip.value("track_id", std::string()) == id) {
                require(policy != "reject", "Track contains clips; choose detach or delete explicitly");
                if (policy == "detach") {
                    clip["track_id"] = "";
                }
            }
        }
        if (policy == "delete") {
            clips.erase(std::remove_if(clips.begin(), clips.end(), [&](const json& clip) {
                return clip.value("track_id", std::string()) == id;
            }), clips.end());
        }
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [&](const json& track) {
            return track.at("id").get<std::string>() == id;
        }), tracks.end());
        result = {{"ok", true}};
    }
    bindClips(candidate);
    scene = std::move(candidate);
    return result;
}

AudioTrackMix AudioTracks::mix(const json& clip, const json& tracks) {
    AudioTrackMix result;
    const bool solo = std::any_of(tracks.begin(), tracks.end(), [](const json& track) {
        return track.at("solo").get<bool>();
    });
    const auto id = clip.value("track_id", std::string());
    result.muted = solo;
    for (const auto& track : tracks) {
        if (track.at("id").get<std::string>() == id) {
            result.gain = track.at("gain").get<float>();
            result.pan = track.at("pan").get<float>();
            result.muted = track.at("muted").get<bool>() || (solo && !track.at("solo").get<bool>());
            return result;
        }
    }
    return result;
}

void AudioTracks::apply(AudioClipSample& sample, const AudioTrackMix& mix) {
    sample.active = sample.active && !mix.muted;
    sample.gain *= mix.gain;
    sample.pan = std::clamp(sample.pan + mix.pan, -1.0f, 1.0f);
}
} // namespace audio
