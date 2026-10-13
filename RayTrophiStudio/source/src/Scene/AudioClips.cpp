#include "Audio/AudioClips.h"
#include "Audio/AudioMixer.h"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>
#include <stdexcept>

namespace audio {
namespace {
using json = nlohmann::json;
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

double number(const json& value, const char* key, double fallback, double low, double high) {
    const double result = value.value(key, fallback);
    require(std::isfinite(result) && result >= low && result <= high,
            "Clip number must be finite and in range");
    return result;
}

json& find(json& clips, const std::string& id) {
    for (auto& clip : clips) {
        if (clip.at("id") == id) {
            return clip;
        }
    }
    throw std::invalid_argument("Unknown audio clip id");
}

std::string allocate(json& scene) {
    const auto next = scene.at("next_id").get<std::uint64_t>();
    require(next > 0 && next < 9007199254740991ULL, "Audio id counter exhausted");
    scene["next_id"] = next + 1;
    return "audio_clip_" + std::to_string(next);
}

double curveGain(const json& curve, double time, double fallback = 1) {
    if (curve.empty()) {
        return fallback;
    }
    if (time <= curve[0][0].get<double>()) {
        return curve[0][1];
    }
    const auto upper = std::upper_bound(curve.begin(), curve.end(), time,
        [](double value, const json& knot) { return value < knot[0].get<double>(); });
    if (upper != curve.end()) {
        const auto lower = std::prev(upper);
        const double blend = (time - (*lower)[0].get<double>()) /
                             ((*upper)[0].get<double>() - (*lower)[0].get<double>());
        return (*lower)[1].get<double>() * (1 - blend) + (*upper)[1].get<double>() * blend;
    }
    return curve.back()[1];
}

double curveIntegral(const json& curve, const json& prefix, double time) {
    if (curve.empty()) {
        return time;
    }
    if (time <= curve[0][0].get<double>()) {
        return time * curve[0][1].get<double>();
    }
    const auto upper = std::upper_bound(curve.begin(), curve.end(), time,
        [](double value, const json& knot) { return value < knot[0].get<double>(); });
    const std::size_t index = static_cast<std::size_t>(std::distance(curve.begin(), upper)) - 1;
    const double elapsed = time - curve[index][0].get<double>();
    const double first = curve[index][1];
    double slope = 0;
    if (index + 1 < curve.size()) {
        slope = (curve[index + 1][1].get<double>() - first) /
                (curve[index + 1][0].get<double>() - curve[index][0].get<double>());
    }
    return prefix[index].get<double>() + first * elapsed + 0.5 * slope * elapsed * elapsed;
}

json validateCurve(const json& value, double horizon, double minimum, double maximum) {
    require(value.is_array() && value.size() <= 4096, "Automation supports at most 4096 knots");
    double previous = -1;
    for (const auto& knot : value) {
        require(knot.is_array() && knot.size() == 2 && knot[0].is_number() && knot[1].is_number(),
                "Automation knot must be [time,value]");
        const double time = knot[0], magnitude = knot[1];
        require(std::isfinite(time) && time >= 0 && time > previous && time <= horizon &&
                std::isfinite(magnitude) && magnitude >= minimum && magnitude <= maximum,
                "Invalid automation knot range/order");
        previous = time;
    }
    return value;
}

double travel(const json& clip, double local) {
    const double origin = clip.at("envelope_offset_seconds");
    return clip.at("pitch").get<double>() *
        (curveIntegral(clip.at("pitch_curve"), clip.at("pitch_integral"), origin + local) -
         curveIntegral(clip.at("pitch_curve"), clip.at("pitch_integral"), origin));
}
}

std::vector<std::string> AudioClips::parameters(const std::string& operation) {
    if (operation == "clip_list") {
        return {};
    }
    if (operation == "clip_split") {
        return {"id", "time_seconds"};
    }
    if (operation == "clip_trim") {
        return {"id", "start_seconds", "end_seconds"};
    }
    if (operation == "clip_remove" || operation == "clip_duplicate") {
        return {"id"};
    }
    return {"id", "name", "asset_id", "start_seconds", "source_in_seconds", "source_out_seconds",
        "duration_seconds", "phase_seconds", "loop", "gain", "pitch", "pan", "bus", "muted",
        "fade_in_seconds", "fade_out_seconds", "gain_curve", "pitch_curve", "pan_curve",
        "envelope_offset_seconds",
        "envelope_duration_seconds"};
}

json AudioClips::validate(const json& input, const json& assets) {
    require(input.is_object(), "Clip must be an object");
    auto allowed = parameters("clip_create");
    allowed.push_back("pitch_integral"); // Derived on validation, never trusted from files.
    for (auto it = input.begin(); it != input.end(); ++it) {
        require(std::find(allowed.begin(), allowed.end(), it.key()) != allowed.end(),
                "Unknown clip property");
    }
    json clip = input;
    const std::string assetId = input.value("asset_id", std::string());
    const json* asset = nullptr;
    for (const auto& entry : assets) {
        if (entry.at("id") == assetId) {
            asset = &entry;
        }
    }
    require(asset != nullptr, "Clip asset_id must name an imported audio asset");
    const double assetDuration = asset->at("duration");
    clip["asset_id"] = assetId;
    clip["name"] = input.value("name", std::string("Audio Clip"));
    require(clip["name"].get<std::string>().size() <= 256, "Clip name is too long");
    clip["start_seconds"] = number(input, "start_seconds", 0, 0, 86400);
    clip["source_in_seconds"] = number(input, "source_in_seconds", 0, 0, assetDuration);
    clip["source_out_seconds"] = number(input, "source_out_seconds", assetDuration, 0, assetDuration);
    for (const char* field : {"source_in_seconds", "source_out_seconds"}) {
        clip[field] = std::floor(clip[field].get<double>() * 48000) / 48000;
    }
    const double span = clip["source_out_seconds"].get<double>() -
                        clip["source_in_seconds"].get<double>();
    require(span + 1e-12 >= 1.0 / 48000, "Clip source range must contain at least one sample");
    clip["pitch"] = number(input, "pitch", 1, 0.125, 8);
    clip["phase_seconds"] = number(input, "phase_seconds", 0, 0, span);
    clip["loop"] = input.value("loop", json(false));
    clip["muted"] = input.value("muted", json(false));
    require(clip["loop"].is_boolean() && clip["muted"].is_boolean(),
            "Clip loop/muted must be boolean");
    const double phase = clip["phase_seconds"];
    const double pitch = clip["pitch"];
    const double defaultDuration = (span - phase) / pitch;
    clip["duration_seconds"] = number(input, "duration_seconds", defaultDuration,
                                       1.0 / 48000, 86400);
    const double duration = clip["duration_seconds"];
    require(clip["start_seconds"].get<double>() + duration <= 86400,
            "Clip exceeds the one-day timeline range");
    clip["gain"] = number(input, "gain", 1, 0, 16);
    clip["pan"] = number(input, "pan", 0, -1, 1);
    clip["bus"] = input.value("bus", std::string("Effects"));
    const auto& buses = AudioMixer::names();
    require(std::find(buses.begin(), buses.end(), clip["bus"].get<std::string>()) != buses.end(),
            "Unknown clip bus");
    clip["envelope_offset_seconds"] = number(input, "envelope_offset_seconds", 0, 0, 86400);
    clip["envelope_duration_seconds"] = number(input, "envelope_duration_seconds", duration,
                                                duration, 86400);
    const double horizon = clip["envelope_duration_seconds"];
    require(clip["envelope_offset_seconds"].get<double>() + duration <= horizon + 1e-7,
            "Clip lies outside its envelope range");
    clip["fade_in_seconds"] = number(input, "fade_in_seconds", 0, 0, horizon);
    clip["fade_out_seconds"] = number(input, "fade_out_seconds", 0, 0, horizon);
    clip["gain_curve"] = validateCurve(input.value("gain_curve", json::array()), horizon, 0, 16);
    clip["pitch_curve"] = validateCurve(input.value("pitch_curve", json::array()),
                                       horizon, 0.0625 / pitch, 16 / pitch);
    clip["pan_curve"] = validateCurve(input.value("pan_curve", json::array()), horizon, -1, 1);
    clip["pitch_integral"] = json::array();
    double integral = 0, previousTime = 0;
    double previousPitch = clip["pitch_curve"].empty() ? 1 : clip["pitch_curve"][0][1].get<double>();
    for (const auto& knot : clip["pitch_curve"]) {
        const double time = knot[0], magnitude = knot[1];
        integral += 0.5 * (magnitude + previousPitch) * (time - previousTime);
        clip["pitch_integral"].push_back(integral);
        previousTime = time;
        previousPitch = magnitude;
    }
    require(clip["loop"].get<bool>() || phase + travel(clip, duration) <= span + 1e-7,
            "Nonlooping clip exceeds its source range");
    return clip;
}

json AudioClips::validateCollection(const json& clips, const json& assets) {
    require(clips.is_array() && clips.size() <= 1024, "Audio supports at most 1024 clips");
    json result = json::array();
    std::set<std::string> ids;
    for (const auto& value : clips) {
        const auto clip = validate(value, assets);
        require(clip.contains("id") && ids.insert(clip.at("id").get<std::string>()).second,
                "Invalid/duplicate clip id");
        result.push_back(clip);
    }
    return result;
}

AudioClipSample AudioClips::evaluate(const json& clip, double seconds) {
    AudioClipSample sample;
    const double local = seconds - clip.at("start_seconds").get<double>();
    const double duration = clip.at("duration_seconds");
    if (!std::isfinite(seconds) || local < 0 || local >= duration || clip.at("muted").get<bool>()) {
        return sample;
    }
    sample.active = true;
    const double envelopeTime = local + clip.at("envelope_offset_seconds").get<double>();
    sample.pitch = static_cast<float>(clip.at("pitch").get<double>() *
        curveGain(clip.at("pitch_curve"), envelopeTime));
    sample.pan = static_cast<float>(std::clamp(clip.at("pan").get<double>() +
        curveGain(clip.at("pan_curve"), envelopeTime, 0), -1.0, 1.0));
    const double span = clip.at("source_out_seconds").get<double>() -
                        clip.at("source_in_seconds").get<double>();
    double source = travel(clip, local) + clip.at("phase_seconds").get<double>();
    if (clip.at("loop").get<bool>()) {
        source = std::fmod(source, span);
    }
    sample.source_seconds = clip.at("source_in_seconds").get<double>() + source;
    const double fadeIn = clip.at("fade_in_seconds"), fadeOut = clip.at("fade_out_seconds");
    const double horizon = clip.at("envelope_duration_seconds");
    double envelope = fadeIn > 0 ? std::min(1.0, envelopeTime / fadeIn) : 1;
    envelope *= fadeOut > 0 ? std::clamp((horizon - envelopeTime) / fadeOut, 0.0, 1.0) : 1;
    sample.gain = static_cast<float>(clip.at("gain").get<double>() * envelope *
                                    curveGain(clip.at("gain_curve"), envelopeTime));
    return sample;
}

double AudioClips::sourceTravel(const json& validatedClip, double localSeconds) {
    return travel(validatedClip, localSeconds);
}

json AudioClips::mutate(json& scene, const std::string& operation, const json& params) {
    json candidate = scene;
    auto& clips = candidate["clips"];
    if (clips.is_null()) {
        clips = json::array();
    }
    json result;
    if (operation == "clip_create") {
        json clip = params;
        require(!clip.contains("id"), "clip_create allocates its own id");
        clip["id"] = allocate(candidate);
        result = validate(clip, candidate.at("assets"));
        clips.push_back(result);
    } else {
        const std::string id = params.at("id");
        json clip = find(clips, id);
        if (operation == "clip_remove") {
            clips.erase(std::remove_if(clips.begin(), clips.end(), [&](const json& entry) {
                return entry.at("id") == id;
            }), clips.end());
            result = {{"removed", id}};
        } else if (operation == "clip_duplicate") {
            clip["id"] = allocate(candidate);
            result = validate(clip, candidate.at("assets"));
            clips.push_back(result);
        } else if (operation == "clip_update") {
            clip.update(params);
            if (params.contains("duration_seconds") && !params.contains("envelope_duration_seconds")) {
                clip["envelope_duration_seconds"] = params.at("duration_seconds");
                clip["envelope_offset_seconds"] = 0;
            }
            result = validate(clip, candidate.at("assets"));
            find(clips, id) = result;
        } else if (operation == "clip_trim" || operation == "clip_split") {
            const double start = clip.at("start_seconds"), duration = clip.at("duration_seconds");
            const double end = start + duration;
            const double left = operation == "clip_trim" ? params.at("start_seconds").get<double>() :
                                params.at("time_seconds").get<double>();
            const double right = operation == "clip_trim" ? params.at("end_seconds").get<double>() : end;
            require(std::isfinite(left) && std::isfinite(right) && left >= start && right <= end &&
                    right > left, "Trim/split range lies outside the clip");
            json tail = clip;
            tail["start_seconds"] = left;
            tail["duration_seconds"] = right - left;
            tail["phase_seconds"] = clip.at("phase_seconds").get<double>() + travel(clip, left - start);
            if (clip.at("loop").get<bool>()) {
                tail["phase_seconds"] = std::fmod(tail["phase_seconds"].get<double>(),
                    clip.at("source_out_seconds").get<double>() - clip.at("source_in_seconds").get<double>());
            }
            tail["envelope_offset_seconds"] = clip.at("envelope_offset_seconds").get<double>() + left - start;
            if (operation == "clip_trim") {
                result = validate(tail, candidate.at("assets"));
                find(clips, id) = result;
            } else {
                require(left > start, "Split must be strictly inside the clip");
                clip["duration_seconds"] = left - start;
                tail["id"] = allocate(candidate);
                clip = validate(clip, candidate.at("assets"));
                tail = validate(tail, candidate.at("assets"));
                find(clips, id) = clip;
                clips.push_back(tail);
                result = {{"left", clip}, {"right", tail}};
            }
        } else {
            throw std::invalid_argument("Unknown clip operation");
        }
    }
    candidate["clips"] = validateCollection(clips, candidate.at("assets"));
    scene = std::move(candidate);
    return result;
}
} // namespace audio
