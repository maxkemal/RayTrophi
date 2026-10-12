#include "Audio/AudioService.h"
#include "Audio/AudioReaction.h"
#include "Audio/AudioEnvironment.h"
#include "Audio/AudioThunder.h"
#include "Audio/AudioTransport.h"

#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MINIAUDIO_IMPLEMENTATION
#include "../../../external/miniaudio/miniaudio.h"
#undef MINIAUDIO_IMPLEMENTATION
#include "Audio/AudioMixerRuntime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <map>
#include <set>
#include <stdexcept>

namespace audio {
std::string pathUtf8(const std::filesystem::path& path) {
    const auto utf8 = path.generic_u8string();
    return std::string(reinterpret_cast<const char*>(utf8.data()), utf8.size());
}

namespace {
using json = nlohmann::json;

std::filesystem::path pathFromUtf8(const std::string& value) {
    const std::u8string utf8(value.begin(), value.end());
    return std::filesystem::path(utf8);
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

float number(const json& p, const char* key, float fallback, float low, float high) {
    const float value = p.value(key, fallback);
    require(std::isfinite(value) && value >= low && value <= high,
            std::string(key) + " must be finite and in range");
    return value;
}

json vector(const json& p, const char* key, const json& fallback) {
    const json value = p.value(key, fallback);
    require(value.is_array() && value.size() == 3, std::string(key) + " needs three numbers");
    for (const auto& component : value) {
        require(component.is_number() && std::isfinite(component.get<float>()),
                std::string(key) + " must be finite");
    }
    return value;
}

json emitter(const json& p) {
    static const std::vector<std::string> keys = {
        "id", "name", "asset_id", "position", "direction", "gain", "pitch", "loop",
        "spatial", "min_distance", "max_distance", "rolloff", "cone_inner", "cone_outer",
        "cone_gain", "bus", "parent", "reaction", "environment_role"};
    for (auto it = p.begin(); it != p.end(); ++it) {
        require(std::find(keys.begin(), keys.end(), it.key()) != keys.end(),
                "Unknown emitter property: " + it.key());
    }
    json e = p;
    e["name"] = p.value("name", std::string("Audio Source"));
    e["asset_id"] = p.value("asset_id", std::string());
    e["parent"] = p.value("parent", std::string());
    e["reaction"] = AudioReaction::validate(p.value("reaction", json::object()));
    e["environment_role"] = p.value("environment_role", std::string());
    require(e["environment_role"] == "" || e["environment_role"] == "wind" ||
            e["environment_role"] == "rain", "Unknown environment role");
    e["position"] = vector(p, "position", json::array({0, 0, 0}));
    e["direction"] = vector(p, "direction", json::array({0, 0, -1}));
    float length = 0;
    for (const auto& component : e["direction"]) {
        length += component.get<float>() * component.get<float>();
    }
    require(length > 1e-12f && std::isfinite(length), "direction must be nonzero");
    e["gain"] = number(p, "gain", 1, 0, 16);
    e["pitch"] = number(p, "pitch", 1, 0.125f, 8);
    e["min_distance"] = number(p, "min_distance", 1, 0.001f, 1000000);
    e["max_distance"] = number(p, "max_distance", 100, 0.001f, 1000000);
    require(e["max_distance"].get<float>() >= e["min_distance"].get<float>(),
            "max_distance must be >= min_distance");
    e["rolloff"] = number(p, "rolloff", 1, 0, 10);
    e["cone_inner"] = number(p, "cone_inner", 360, 0, 360);
    e["cone_outer"] = number(p, "cone_outer", 360, 0, 360);
    require(e["cone_outer"].get<float>() >= e["cone_inner"].get<float>(),
            "cone_outer must be >= cone_inner");
    e["cone_gain"] = number(p, "cone_gain", 1, 0, 1);
    e["loop"] = p.value("loop", false);
    require(!e.at("reaction").at("automatic").get<bool>() || e.at("loop").get<bool>(),
            "Automatic continuous audio requires loop=true");
    e["spatial"] = p.value("spatial", true);
    e["bus"] = p.value("bus", std::string("Effects"));
    const auto& buses = AudioMixer::names();
    require(std::find(buses.begin(), buses.end(), e["bus"].get<std::string>()) != buses.end(),
            "Unknown audio bus");
    return e;
}
} // namespace

struct AudioService::Impl {
    json data = {{"version", 1}, {"next_id", 1}, {"assets", json::array()},
                 {"emitters", json::array()}, {"thunder", AudioThunder::validate(json::object())},
                 {"playback_mode", "timeline"}, {"playback_mode_explicit", false},
                 {"mixer", AudioMixer::validate(json::object())}};
    ma_engine engine{};
    AudioMixerRuntime mixer;
    std::map<std::string, std::string> routedBuses;
    bool initialized = false;
    bool muted = false;
    std::string status = "not_initialized";
    std::map<std::string, std::unique_ptr<ma_sound>> sounds;
    std::unique_ptr<ma_sound> preview;
    std::map<std::string, json> poses;
    AudioReaction reactions;
    std::map<std::string, ReactionOutput> controls;
    std::uint64_t generation = 1;
    std::set<std::string> suspended;
    std::set<std::string> managed;
    std::map<std::string, std::chrono::steady_clock::time_point> failedAt;
    std::map<std::string, std::string> autoErrors;
    AudioThunder thunder;
    AudioTransport transport;
    std::set<std::string> transportHeld;
    json listenerPosition = json::array({0, 0, 0});
    std::array<float, 9> listenerPose = {0, 0, 0, 0, 0, -1, 0, 1, 0};
    struct ThunderVoice {
        std::unique_ptr<ma_sound> sound;
        json event;
        double bytes = 0;
        bool held = false;
    };
    std::vector<ThunderVoice> thunderVoices;
    std::string thunderError;

    void holdTransport() {
        for (auto& [id, sound] : sounds) {
            if (transport.paused() && ma_sound_is_playing(sound.get())) {
                ma_sound_stop(sound.get());
                transportHeld.insert(id);
            } else if (!transport.paused() && transportHeld.erase(id) != 0 &&
                       !suspended.contains(id)) {
                ma_sound_start(sound.get());
            }
        }
        for (auto& voice : thunderVoices) {
            if (transport.paused() && ma_sound_is_playing(voice.sound.get())) {
                ma_sound_stop(voice.sound.get());
                voice.held = true;
            } else if (!transport.paused() && voice.held) {
                ma_sound_start(voice.sound.get());
                voice.held = false;
            }
        }
    }

    void stopThunder() {
        for (auto& voice : thunderVoices) {
            ma_sound_uninit(voice.sound.get());
        }
        thunderVoices.clear();
    }

    void validateThunderAssets(const json& config) {
        for (const char* range : {"near_assets", "far_assets"}) {
            for (const auto& id : config.at(range)) {
                require(find("assets", id).at("channels") == 1,
                        "Positional thunder requires mono assets; convert stereo before binding");
            }
        }
    }

    ~Impl() {
        stopAll();
        mixer.shutdown();
        if (initialized) {
            ma_engine_uninit(&engine);
        }
    }

    void stopAll() {
        routedBuses.clear();
        transportHeld.clear();
        stopThunder();
        thunder.reset(true);
        for (auto& item : sounds) {
            ma_sound_uninit(item.second.get());
        }
        sounds.clear();
        managed.clear();
        if (preview) {
            ma_sound_uninit(preview.get());
            preview.reset();
        }
    }

    void ensureDevice() {
        if (initialized) {
            return;
        }
        ma_engine_config config = ma_engine_config_init();
        config.sampleRate = 48000;
        config.channels = 2;
        const ma_result result = ma_engine_init(&config, &engine);
        if (result != MA_SUCCESS) {
            status = "device_unavailable";
            throw std::runtime_error("device_unavailable: " + std::to_string(result));
        }
        try {
            mixer.initialize(engine);
            mixer.apply(data.at("mixer"));
        } catch (...) {
            ma_engine_uninit(&engine);
            status = "mixer_unavailable";
            throw;
        }
        initialized = true;
        status = "ready";
        ma_engine_set_volume(&engine, muted ? 0.0f : 1.0f);
        ma_engine_listener_set_position(&engine, 0, listenerPose[0], listenerPose[1], listenerPose[2]);
        ma_engine_listener_set_direction(&engine, 0, listenerPose[3], listenerPose[4], listenerPose[5]);
        ma_engine_listener_set_world_up(&engine, 0, listenerPose[6], listenerPose[7], listenerPose[8]);
    }

    json& find(const char* collection, const std::string& id) {
        for (auto& item : data[collection]) {
            if (item.at("id") == id) {
                return item;
            }
        }
        throw std::invalid_argument(std::string(collection) + " id not found: " + id);
    }

    std::string id(const char* prefix) {
        const auto next = data.at("next_id").get<unsigned long long>();
        require(next < 9007199254740991ULL, "audio id space exhausted");
        data["next_id"] = next + 1;
        return std::string(prefix) + std::to_string(next);
    }

    void validateAsset(const json& e) {
        if (e.at("asset_id").get<std::string>().empty()) {
            require(!e.at("reaction").at("automatic").get<bool>(),
                    "Automatic audio needs a bound asset");
            return;
        }
        auto& asset = find("assets", e.at("asset_id"));
        require(!e.at("spatial").get<bool>() || asset.at("channels") == 1,
                "Positional sources require mono assets; stereo uses spatial=false");
    }

    void apply(ma_sound& sound, const json& e) {
        const std::string id = e.at("id");
        const std::string bus = e.at("bus");
        if (!routedBuses.contains(id) || routedBuses.at(id) != bus) {
            mixer.route(sound, bus);
            routedBuses[id] = bus;
        }
        const auto control = controls.find(e.at("id").get<std::string>());
        const auto modulation = control == controls.end() ? ReactionOutput{} : control->second;
        const bool reactive = e.at("reaction").at("driver") != "manual";
        const float gain = reactive && control == controls.end() ? 0 : modulation.gain;
        const float direct = bus == "Master" ? AudioMixer::directMasterGain(data.at("mixer")) : 1;
        ma_sound_set_volume(&sound, e.at("gain").get<float>() * gain * direct);
        ma_sound_set_pitch(&sound, std::clamp(e.at("pitch").get<float>() * modulation.pitch,
                                            0.125f, 8.0f));
        ma_sound_set_looping(&sound, e.at("loop").get<bool>());
        ma_sound_set_spatialization_enabled(&sound, e.at("spatial").get<bool>());
        const auto pose = poses.find(e.at("id").get<std::string>());
        const json& resolved = pose == poses.end() ? e : pose->second;
        const auto& p = modulation.position.is_null() ? resolved.at("position") : modulation.position;
        const auto& d = resolved.at("direction");
        ma_sound_set_position(&sound, p[0], p[1], p[2]);
        ma_sound_set_direction(&sound, d[0], d[1], d[2]);
        ma_sound_set_min_distance(&sound, e.at("min_distance"));
        ma_sound_set_max_distance(&sound, e.at("max_distance"));
        ma_sound_set_rolloff(&sound, e.at("rolloff"));
        constexpr float radians = 0.017453292519943295f;
        ma_sound_set_cone(&sound, e.at("cone_inner").get<float>() * radians,
                          e.at("cone_outer").get<float>() * radians, e.at("cone_gain"));
    }

    std::unique_ptr<ma_sound> load(const json& asset, const std::string& bus = "") {
        AudioService probe;
        const auto actual = probe.invoke("asset_import", {{"path", asset.at("path")}});
        require(actual.at("channels") == asset.at("channels"),
                "Audio file channels changed; relink the asset before playback");
        require(actual.at("frames") == asset.at("frames") &&
                actual.at("sample_rate") == asset.at("sample_rate"),
                "Audio file timing changed; relink the asset before playback");
        double residentBytes = actual.at("duration").get<double>() * 48000 *
                               actual.at("channels").get<unsigned>() * 4;
        for (const auto& voice : thunderVoices) {
            residentBytes += voice.bytes;
        }
        for (const auto& voice : sounds) {
            const auto& e = find("emitters", voice.first);
            const auto& a = find("assets", e.at("asset_id"));
            residentBytes += a.at("duration").get<double>() * 48000 * a.at("channels").get<unsigned>() * 4;
        }
        require(residentBytes <= 512 * 1024 * 1024, "Stage 1 decoded audio budget reached (512 MiB)");
        ensureDevice();
        auto sound = std::make_unique<ma_sound>();
        const auto path = pathFromUtf8(asset.at("path").get<std::string>());
        // Fully decode before playback: no file IO or scene work in our callback.
        #ifdef _WIN32
        const ma_result result = ma_sound_init_from_file_w(&engine, path.c_str(),
        #else
        const ma_result result = ma_sound_init_from_file(&engine, path.c_str(),
        #endif
            MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_NO_SPATIALIZATION,
            bus.empty() ? nullptr : mixer.group(bus), nullptr, sound.get());
        if (result != MA_SUCCESS) {
            throw std::runtime_error("audio_file_unavailable: " + std::to_string(result));
        }
        return sound;
    }
};

AudioService::AudioService() : impl_(std::make_unique<Impl>()) {
    impl_->thunder.clock(true, 0);
    impl_->reactions.clock(true, 0);
}
AudioService::~AudioService() = default;

const std::vector<std::string>& AudioService::operations() {
    static const std::vector<std::string> names = {
        "list", "get", "select", "status", "asset_import", "asset_relink", "asset_remove", "create", "update",
        "duplicate", "remove", "play", "pause", "stop", "stop_all", "preview", "preview_stop",
        "preview_mute", "preview_file", "append", "signal_set", "signals_clear",
        "environment_setup", "resume_automatic", "domains", "thunder_setup", "thunder_update",
        "thunder_event", "playback_update", "mixer_get", "mixer_update", "mixer_reset_peaks"};
    return names;
}

bool AudioService::isAuthoring(const std::string& op) {
    return op == "mixer_update" || op == "playback_update" ||
           op == "thunder_setup" || op == "thunder_update" ||
           op == "environment_setup" || op == "append" || op == "asset_import" ||
           op == "asset_relink" || op == "asset_remove" ||
           op == "create" || op == "update" || op == "duplicate" || op == "remove";
}

std::vector<std::string> AudioService::parameters(const std::string& op) {
    if (op == "mixer_update") {
        return {"bus", "gain", "muted", "solo"};
    }
    if (op == "playback_update") {
        return {"mode"};
    }
    if (op == "create" || op == "update") {
        std::vector<std::string> keys = {
            "name", "asset_id", "parent", "position", "direction", "gain", "pitch", "loop",
            "spatial", "min_distance", "max_distance", "rolloff", "cone_inner", "cone_outer",
            "cone_gain", "bus", "reaction", "environment_role"};
        if (op == "update") {
            keys.push_back("id");
        }
        return keys;
    }
    if (op == "append" || op == "asset_import" || op == "asset_relink") {
        std::vector<std::string> keys = {"path", "license", "source", "author", "tags"};
        if (op == "append") {
            keys.push_back("name");
            keys.push_back("position");
        } else if (op == "asset_relink") {
            keys.push_back("id");
        }
        return keys;
    }
    if (op == "preview_file") {
        return {"path"};
    }
    if (op == "preview_mute") {
        return {"muted"};
    }
    if (op == "signal_set") {
        return {"key", "value", "flow_kg_s", "speed_mps", "position", "source_material",
                "target_material", "ttl_seconds", "power_w"};
    }
    if (op == "environment_setup" || op == "thunder_setup") {
        return {"assets_root"};
    }
    if (op == "thunder_update") {
        return {"config"};
    }
    if (op == "thunder_event") {
        return {"key", "position", "intensity", "age_seconds"};
    }
    if (op == "list" || op == "status" || op == "stop_all" || op == "preview_stop" ||
        op == "signals_clear" || op == "domains" || op == "resume_automatic" ||
        op == "mixer_get" || op == "mixer_reset_peaks") {
        return {};
    }
    return {"id"};
}

json AudioService::invoke(const std::string& op, const json& p) {
    auto& s = *impl_;
    require(p.is_object(), "audio params must be an object");
    require(std::find(operations().begin(), operations().end(), op) != operations().end(),
            "Unknown audio operation: " + op);
    const auto allowed = parameters(op);
    for (auto it = p.begin(); it != p.end(); ++it) {
        require(std::find(allowed.begin(), allowed.end(), it.key()) != allowed.end(),
                "Unknown audio parameter: " + it.key());
    }
    if (op == "mixer_get") {
        json buses = json::array();
        const auto meters = s.mixer.meters();
        for (const auto& name : AudioMixer::names()) {
            auto settings = s.data.at("mixer").at(name);
            settings["name"] = name;
            settings["effective_gain"] = AudioMixer::effectiveGain(s.data.at("mixer"), name);
            settings["output_gain"] = settings.at("effective_gain").get<float>() *
                (name == "Master" ? 1 : AudioMixer::effectiveGain(s.data.at("mixer"), "Master"));
            settings["meter"] = meters.at(name);
            buses.push_back(settings);
        }
        return {{"buses", buses}, {"output_muted", s.muted},
                {"direct_master_gain", AudioMixer::directMasterGain(s.data.at("mixer"))},
                {"meter_stage", "post_bus_fader_before_preview_mute"},
                {"audition_bypasses_mixer", true}};
    }
    if (op == "mixer_reset_peaks") {
        s.mixer.resetPeaks();
        return {{"ok", true}};
    }
    if (op == "mixer_update") {
        const auto settings = AudioMixer::update(s.data.at("mixer"), p);
        s.data["mixer"] = settings;
        s.mixer.apply(settings);
        for (auto& [id, sound] : s.sounds) {
            s.apply(*sound, s.find("emitters", id));
        }
        return settings.at(p.at("bus").get<std::string>());
    }
    if (op == "playback_update") {
        const auto mode = AudioTransport::validateMode(p.at("mode"));
        s.data["playback_mode_explicit"] = true;
        if (s.data.at("playback_mode") != mode) {
            const auto previous = s.transport.status();
            s.stopAll();
            s.data["playback_mode"] = mode;
            s.transport.configure(mode);
            s.transport.advance(previous.at("time_seconds"), previous.at("playing"));
            s.thunder.clock(s.transport.timeline(), s.transport.time());
            s.reactions.clear();
            s.controls.clear();
            s.reactions.clock(s.transport.timeline(), s.transport.time());
            s.thunder.resume();
            s.suspended.clear();
        }
        return {{"mode", mode}};
    }
    if (op == "resume_automatic") {
        if (s.preview) {
            ma_sound_uninit(s.preview.get());
            s.preview.reset();
        }
        s.suspended.clear();
        s.failedAt.clear();
        s.autoErrors.clear();
        s.thunderError.clear();
        s.thunder.resume();
        if (!s.transport.paused()) {
            for (auto& [id, sound] : s.sounds) {
                if (s.find("emitters", id).at("reaction").at("automatic").get<bool>() &&
                    !ma_sound_is_playing(sound.get())) {
                    const auto result = ma_sound_start(sound.get());
                    if (result != MA_SUCCESS) {
                        s.autoErrors[id] = "Could not resume automatic audio";
                    }
                }
            }
        }
        return {{"ok", true}};
    }
    if (op == "thunder_setup") {
        AudioService candidate;
        candidate.restore(s.data);
        const auto result = installThunder(candidate,
            pathFromUtf8(p.at("assets_root").get<std::string>()));
        s.stopThunder();
        s.data = candidate.snapshot();
        s.thunder.configure(s.data.at("thunder"));
        s.thunderError.clear();
        return result;
    }
    if (op == "thunder_update") {
        json combined = s.data.value("thunder", AudioThunder::validate(json::object()));
        require(p.at("config").is_object(), "Thunder config must be an object");
        combined.update(p.at("config"));
        const auto config = AudioThunder::validate(combined);
        s.validateThunderAssets(config);
        s.stopThunder();
        s.thunder.configure(config);
        s.data["thunder"] = config;
        s.thunderError.clear();
        return config;
    }
    if (op == "thunder_event") {
        if (s.preview) {
            return {{"queued", false}, {"reason", "asset_preview_active"}};
        }
        return s.thunder.publish(p, s.listenerPosition);
    }
    if (op == "domains") {
        throw std::invalid_argument("audio.domains requires a bound scene context");
    }
    if (op == "signal_set") {
        return s.reactions.publish(p);
    }
    if (op == "environment_setup") {
        AudioService candidate;
        candidate.restore(s.data);
        const auto result = installEnvironment(candidate,
            pathFromUtf8(p.at("assets_root").get<std::string>()));
        s.data = candidate.snapshot();
        for (const char* role : {"wind_source_id", "rain_source_id"}) {
            const std::string id = result.at(role);
            s.suspended.erase(id);
            s.failedAt.erase(id);
            s.autoErrors.erase(id);
        }
        return result;
    }
    if (op == "signals_clear") {
        s.reactions.clear();
        s.controls.clear();
        return {{"ok", true}};
    }
    if (op == "append") {
        AudioService candidate;
        candidate.restore(s.data);
        json importParams = {{"path", p.at("path")}};
        for (const char* key : {"license", "source", "author", "tags"}) {
            if (p.contains(key)) {
                importParams[key] = p.at(key);
            }
        }
        const auto asset = candidate.invoke("asset_import", importParams);
        const auto source = candidate.invoke("create", {
            {"asset_id", asset.at("id")}, {"name", p.value("name", asset.at("name").get<std::string>())},
            {"spatial", asset.at("channels") == 1},
            {"position", vector(p, "position", json::array({0, 0, 0}))}});
        s.data = candidate.snapshot();
        return source;
    }
    if (op == "list") {
        return s.data;
    }
    if (op == "get" || op == "select") {
        return s.find("emitters", p.at("id"));
    }
    if (op == "status") {
        json voices = json::array();
        for (const auto& item : s.sounds) {
            ma_uint64 cursor = 0;
            ma_sound_get_cursor_in_pcm_frames(item.second.get(), &cursor);
            voices.push_back({{"id", item.first},
                {"playing", bool(ma_sound_is_playing(item.second.get()))},
                {"timeline_held", s.transportHeld.contains(item.first)}, {"cursor_frames", cursor}});
        }
        json thunderVoices = json::array();
        for (const auto& voice : s.thunderVoices) {
            auto event = voice.event;
            event["playing"] = bool(ma_sound_is_playing(voice.sound.get()));
            event["timeline_held"] = voice.held;
            ma_uint64 cursor = 0;
            ma_sound_get_cursor_in_pcm_frames(voice.sound.get(), &cursor);
            event["cursor_frames"] = cursor;
            thunderVoices.push_back(event);
        }
        return {{"device", s.status}, {"sample_rate", 48000}, {"preview_muted", s.muted},
                {"voices", voices}, {"voice_limit", 32}, {"resolved_poses", s.poses},
                {"reaction_runtime", s.reactions.status()}, {"automatic_errors", s.autoErrors},
                {"automatic_suspended", s.suspended}, {"thunder_runtime", s.thunder.status()},
                {"thunder_voices", thunderVoices}, {"thunder_error", s.thunderError},
                {"playback_runtime", s.transport.status()}, {"transport_held_sources", s.transportHeld}};
    }
    if (op == "asset_import" || op == "asset_relink") {
        const auto path = std::filesystem::absolute(pathFromUtf8(p.at("path").get<std::string>()));
        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return char(std::tolower(c));
        });
        require(ext == ".wav" || ext == ".flac" || ext == ".mp3", "Supported audio: WAV, FLAC, MP3");
        ma_decoder decoder{};
        auto config = ma_decoder_config_init(ma_format_f32, 0, 0);
        #ifdef _WIN32
        const auto result = ma_decoder_init_file_w(path.c_str(), &config, &decoder);
        #else
        const auto result = ma_decoder_init_file(path.c_str(), &config, &decoder);
        #endif
        require(result == MA_SUCCESS, "Cannot decode audio file");
        ma_uint64 frames = 0;
        const auto lengthResult = ma_decoder_get_length_in_pcm_frames(&decoder, &frames);
        const auto channels = decoder.outputChannels;
        const auto rate = decoder.outputSampleRate;
        ma_decoder_uninit(&decoder);
        require(lengthResult == MA_SUCCESS && rate > 0 && frames > 0 && channels > 0 && channels <= 2,
                "Audio must have known duration and one or two channels");
        const double duration = double(frames) / rate;
        require(duration <= 600 && duration * 48000 * channels * 4 <= 128 * 1024 * 1024,
                "Stage 1 decode limit: 600 seconds / 128 MiB per asset; streaming comes later");
        json asset = {{"path", pathUtf8(path)}, {"duration", duration},
                      {"channels", channels}, {"sample_rate", rate}, {"frames", frames},
                      {"license", p.value("license", std::string())},
                      {"author", p.value("author", std::string())},
                      {"source", p.value("source", std::string())},
                      {"tags", p.value("tags", std::vector<std::string>())},
                      {"name", pathUtf8(path.stem())}};
        if (op == "asset_import") {
            asset["id"] = s.id("audio_asset_");
            s.data["assets"].push_back(asset);
        } else {
            auto& old = s.find("assets", p.at("id"));
            for (const auto& e : s.data["emitters"]) {
                require(e.at("asset_id") != old.at("id") || !e.at("spatial").get<bool>() || channels == 1,
                        "Relink would give a positional source a stereo asset");
            }
            const auto config = s.data.value("thunder", AudioThunder::validate(json::object()));
            for (const char* range : {"near_assets", "far_assets"}) {
                const auto& ids = config.at(range);
                require(channels == 1 || std::find(ids.begin(), ids.end(), old.at("id")) == ids.end(),
                        "Relink would give positional thunder a stereo asset");
            }
            asset["id"] = old.at("id");
            for (const char* key : {"license", "author", "source"}) {
                if (!p.contains(key)) {
                    asset[key] = old.value(key, std::string());
                }
            }
            if (!p.contains("tags")) {
                asset["tags"] = old.value("tags", json::array());
            }
            s.stopAll();
            old = asset;
            for (const auto& e : s.data.at("emitters")) {
                if (e.at("asset_id") == asset.at("id")) {
                    const std::string id = e.at("id");
                    s.failedAt.erase(id);
                    s.autoErrors.erase(id);
                }
            }
        }
        return asset;
    }
    if (op == "asset_remove") {
        const std::string id = p.at("id");
        const auto config = s.data.value("thunder", AudioThunder::validate(json::object()));
        for (const char* range : {"near_assets", "far_assets"}) {
            const auto& ids = config.at(range);
            require(std::find(ids.begin(), ids.end(), id) == ids.end(),
                    "Asset is referenced by thunder; remove its binding first");
        }
        s.find("assets", id);
        for (const auto& e : s.data["emitters"]) {
            require(e.at("asset_id") != id, "Asset is referenced by an emitter");
        }
        s.stopAll();
        auto& assets = s.data["assets"];
        assets.erase(std::remove_if(assets.begin(), assets.end(), [&](const json& a) {
            return a.at("id") == id;
        }), assets.end());
        return {{"ok", true}};
    }
    if (op == "create" || op == "duplicate") {
        json e;
        if (op == "duplicate") {
            e = s.find("emitters", p.at("id"));
            e["environment_role"] = "";
            e["name"] = e.at("name").get<std::string>() + " Copy";
        } else {
            require(!p.contains("id"), "Emitter ids are allocated by the service");
            json settings = p;
            const auto assetId = p.value("asset_id", std::string());
            if (!p.contains("spatial") && !assetId.empty()) {
                settings["spatial"] = s.find("assets", assetId).at("channels") == 1;
            }
            e = emitter(settings);
        }
        s.validateAsset(e);
        e["id"] = s.id("audio_source_");
        s.data["emitters"].push_back(e);
        return e;
    }
    if (op == "update") {
        auto& old = s.find("emitters", p.at("id"));
        json combined = old;
        combined.update(p);
        if (p.contains("asset_id") && !p.contains("spatial")) {
            const auto assetId = p.at("asset_id").get<std::string>();
            if (!assetId.empty() && p.at("asset_id") != old.at("asset_id")) {
                combined["spatial"] = s.find("assets", assetId).at("channels") == 1;
            }
        }
        json e = emitter(combined);
        s.validateAsset(e);
        if (e.at("reaction") != old.at("reaction") || e.at("asset_id") != old.at("asset_id")) {
            s.reactions.forget(e.at("id"));
            s.controls.erase(e.at("id").get<std::string>());
            s.suspended.erase(e.at("id").get<std::string>());
            s.failedAt.erase(e.at("id").get<std::string>());
            s.autoErrors.erase(e.at("id").get<std::string>());
        }
        if (e.at("asset_id") != old.at("asset_id") ||
            (s.managed.contains(e.at("id").get<std::string>()) &&
             !e.at("reaction").at("automatic").get<bool>())) {
            const auto voice = s.sounds.find(p.at("id").get<std::string>());
            if (voice != s.sounds.end()) {
                ma_sound_uninit(voice->second.get());
                s.sounds.erase(voice);
                s.managed.erase(e.at("id").get<std::string>());
            }
        }
        old = e;
        const auto voice = s.sounds.find(e.at("id").get<std::string>());
        if (voice != s.sounds.end()) {
            s.apply(*voice->second, e);
        }
        return e;
    }
    if (op == "remove") {
        const std::string id = p.at("id");
        s.routedBuses.erase(id);
        s.transportHeld.erase(id);
        s.find("emitters", id);
        s.poses.erase(id);
        s.reactions.forget(id);
        s.controls.erase(id);
        s.suspended.erase(id);
        s.managed.erase(id);
        s.failedAt.erase(id);
        s.autoErrors.erase(id);
        const auto voice = s.sounds.find(id);
        if (voice != s.sounds.end()) {
            ma_sound_uninit(voice->second.get());
            s.sounds.erase(voice);
        }
        auto& emitters = s.data["emitters"];
        emitters.erase(std::remove_if(emitters.begin(), emitters.end(), [&](const json& e) {
            return e.at("id") == id;
        }), emitters.end());
        return {{"ok", true}};
    }
    if (op == "stop_all") {
        for (const auto& e : s.data.at("emitters")) {
            if (e.at("reaction").at("automatic").get<bool>()) {
                s.suspended.insert(e.at("id").get<std::string>());
            }
        }
        s.stopAll();
    } else if (op == "preview_mute") {
        s.muted = p.at("muted").get<bool>();
        if (s.initialized) {
            ma_engine_set_volume(&s.engine, s.muted ? 0.0f : 1.0f);
        }
    } else if (op == "preview_stop") {
        if (s.preview) {
            ma_sound_uninit(s.preview.get());
            s.preview.reset();
        }
    } else if (op == "preview" || op == "preview_file") {
        AudioService probe;
        const auto asset = op == "preview" ? s.find("assets", p.at("id")) :
            probe.invoke("asset_import", {{"path", p.at("path")}});
        // Asset audition owns the engine until explicitly stopped or a source is played.
        s.stopAll();
        s.preview = s.load(asset);
        require(ma_sound_start(s.preview.get()) == MA_SUCCESS, "Could not start asset preview");
    } else {
        const std::string id = p.at("id");
        const auto e = s.find("emitters", id);
        if (op == "play") {
            require(!e.at("asset_id").get<std::string>().empty(),
                    "No audio asset is bound. Import a file, then choose it in the Asset field.");
            const auto pose = s.poses.find(id);
            require(pose == s.poses.end() || pose->second.at("valid").get<bool>(),
                    "Audio parent is unavailable");
        }
        auto voice = s.sounds.find(id);
        if (op == "play") {
            if (s.preview) {
                ma_sound_uninit(s.preview.get());
                s.preview.reset();
            }
            if (voice == s.sounds.end()) {
                // Reclaim completed one-shot sounds before enforcing the resident voice budget.
                for (auto it = s.sounds.begin(); it != s.sounds.end();) {
                    if (ma_sound_at_end(it->second.get())) {
                        ma_sound_uninit(it->second.get());
                        it = s.sounds.erase(it);
                    } else {
                        ++it;
                    }
                }
                require(s.sounds.size() + s.thunderVoices.size() < 32,
                        "Stage 1 resident voice limit reached (32)");
                auto sound = s.load(s.find("assets", e.at("asset_id")), e.at("bus"));
                s.apply(*sound, e);
                voice = s.sounds.emplace(id, std::move(sound)).first;
            }
            if (ma_sound_at_end(voice->second.get())) {
                ma_sound_seek_to_pcm_frame(voice->second.get(), 0);
            }
            if (s.transport.paused()) {
                s.transportHeld.insert(id);
            } else {
                require(ma_sound_start(voice->second.get()) == MA_SUCCESS,
                        "Could not start audio source");
            }
            s.suspended.erase(id);
            s.failedAt.erase(id);
            s.autoErrors.erase(id);
            if (e.at("reaction").at("automatic").get<bool>()) {
                s.managed.insert(id);
            }
        } else if (voice != s.sounds.end()) {
            s.transportHeld.erase(id);
            if (e.at("reaction").at("automatic").get<bool>()) {
                s.suspended.insert(id);
            }
            ma_sound_stop(voice->second.get());
            if (op == "stop") {
                ma_sound_uninit(voice->second.get());
                s.sounds.erase(voice);
                s.managed.erase(id);
            }
        } else if (op == "pause" || op == "stop") {
            if (e.at("reaction").at("automatic").get<bool>()) {
                s.suspended.insert(id);
            }
        }
    }
    s.holdTransport();
    return {{"ok", true}};
}

json AudioService::snapshot() const {
    return impl_->data;
}

void AudioService::clear() {
    impl_->stopAll();
    impl_->poses.clear();
    impl_->reactions.clear();
    impl_->controls.clear();
    impl_->suspended.clear();
    impl_->failedAt.clear();
    impl_->autoErrors.clear();
    ++impl_->generation;
    impl_->data = {{"version", 1}, {"next_id", 1}, {"assets", json::array()},
                   {"emitters", json::array()},
                   {"thunder", AudioThunder::validate(json::object())},
                   {"playback_mode", "timeline"}, {"playback_mode_explicit", false},
                   {"mixer", AudioMixer::validate(json::object())}};
    impl_->mixer.apply(impl_->data.at("mixer"));
    impl_->mixer.resetPeaks();
    impl_->transport.configure("timeline");
    impl_->thunder.clock(true, 0);
    impl_->reactions.clock(true, 0);
    impl_->thunder.configure(impl_->data.at("thunder"));
    impl_->thunderError.clear();
}

void AudioService::restore(const json& input, const std::filesystem::path& base) {
    if (input.is_null()) {
        clear();
        return;
    }
    require(input.at("version") == 1, "Unsupported audio schema version");
    Impl candidate;
    candidate.data = input;
    require(input.at("assets").is_array() && input.at("emitters").is_array(), "Invalid audio collections");
    const auto next = input.at("next_id").get<unsigned long long>();
    require(next > 0 && next <= 9007199254740991ULL, "Invalid audio next_id");
    std::vector<std::string> ids;
    auto checkId = [&](const json& item, const std::string& prefix) {
        const std::string id = item.at("id");
        require(id.rfind(prefix, 0) == 0, "Invalid audio id prefix");
        const auto suffix = id.substr(prefix.size());
        require(!suffix.empty() && suffix.find_first_not_of("0123456789") == std::string::npos,
                "Invalid audio id");
        require(std::stoull(suffix) > 0 && std::stoull(suffix) < next, "Invalid audio id counter");
        require(std::find(ids.begin(), ids.end(), id) == ids.end(), "Duplicate audio id");
        ids.push_back(id);
    };
    for (auto& asset : candidate.data["assets"]) {
        checkId(asset, "audio_asset_");
        require(asset.at("channels") == 1 || asset.at("channels") == 2, "Invalid channel count");
        const double duration = asset.at("duration").get<double>();
        require(std::isfinite(duration) && duration > 0 && duration <= 600, "Invalid duration");
        const auto rate = asset.at("sample_rate").get<unsigned>();
        const auto frames = asset.at("frames").get<ma_uint64>();
        const auto channels = asset.at("channels").get<unsigned>();
        require(rate > 0 && rate <= 768000 && frames > 0, "Invalid audio sample metadata");
        require(std::abs(double(frames) / rate - duration) < 0.000001,
                "Audio duration does not match sample metadata");
        require(duration * 48000 * channels * 4 <= 128 * 1024 * 1024,
                "Asset exceeds decoded sample budget");
        require(asset.at("name").is_string(), "Invalid audio asset name");
        for (const char* key : {"license", "author", "source"}) {
            require(asset.value(key, json("")).is_string(), "Invalid asset provenance");
        }
        const auto tags = asset.value("tags", json::array());
        require(tags.is_array(), "Audio tags must be an array");
        for (const auto& tag : tags) {
            require(tag.is_string(), "Audio tags must be strings");
        }
        auto path = pathFromUtf8(asset.at("path").get<std::string>());
        if (path.is_relative() && !base.empty()) {
            path = base / path;
        }
        asset["path"] = pathUtf8(path);
    }
    for (auto& e : candidate.data["emitters"]) {
        checkId(e, "audio_source_");
        e = emitter(e);
        candidate.validateAsset(e);
    }
    candidate.data["thunder"] = AudioThunder::validate(input.value("thunder", json::object()));
    candidate.data["mixer"] = AudioMixer::validate(input.value("mixer", json::object()));
    const auto mode = AudioTransport::validateMode(input.value("playback_mode", json("timeline")));
    const bool explicitMode = input.value("playback_mode_explicit", false);
    candidate.data["playback_mode"] = explicitMode ? mode : "timeline";
    candidate.data["playback_mode_explicit"] = explicitMode;
    candidate.validateThunderAssets(candidate.data.at("thunder"));
    // Validate the complete candidate before replacing live data. Missing files are retained.
    impl_->stopAll();
    impl_->poses.clear();
    impl_->reactions.clear();
    impl_->controls.clear();
    impl_->suspended.clear();
    impl_->failedAt.clear();
    impl_->autoErrors.clear();
    ++impl_->generation;
    impl_->data = std::move(candidate.data);
    impl_->mixer.apply(impl_->data.at("mixer"));
    impl_->mixer.resetPeaks();
    impl_->transport.configure(impl_->data.at("playback_mode"));
    impl_->thunder.clock(impl_->transport.timeline(), 0);
    impl_->reactions.clock(impl_->transport.timeline(), 0);
    impl_->thunder.configure(impl_->data.at("thunder"));
    impl_->thunderError.clear();
}

json AudioService::save(const std::filesystem::path& project) const {
    json data = snapshot();
    auto folderName = project.stem();
    folderName += "_audio";
    const auto folder = project.parent_path() / folderName;
    for (auto& asset : data["assets"]) {
        const auto source = pathFromUtf8(asset.at("path").get<std::string>());
        std::error_code error;
        if (std::filesystem::is_regular_file(source, error)) {
            std::filesystem::create_directories(folder);
            const auto target = folder / (asset.at("id").get<std::string>() + source.extension().string());
            if (!std::filesystem::exists(target) || !std::filesystem::equivalent(source, target)) {
                std::filesystem::copy_file(source, target, std::filesystem::copy_options::overwrite_existing);
            }
            asset["path"] = pathUtf8(std::filesystem::relative(target, project.parent_path()));
            asset["missing"] = false;
        } else {
            asset["missing"] = true;
        }
    }
    return data;
}

void AudioService::listener(float x, float y, float z, float dx, float dy, float dz,
                            float ux, float uy, float uz) {
    for (float value : {x, y, z, dx, dy, dz, ux, uy, uz}) {
        if (!std::isfinite(value)) {
            return;
        }
    }
    impl_->listenerPosition = json::array({x, y, z});
    impl_->listenerPose = {x, y, z, dx, dy, dz, ux, uy, uz};
    if (!impl_->initialized) {
        return;
    }
    ma_engine_listener_set_position(&impl_->engine, 0, x, y, z);
    ma_engine_listener_set_direction(&impl_->engine, 0, dx, dy, dz);
    ma_engine_listener_set_world_up(&impl_->engine, 0, ux, uy, uz);
}

void AudioService::resolvedPose(const std::string& id, const json& position,
                               const json& direction, bool valid) {
    impl_->poses[id] = {{"position", position}, {"direction", direction}, {"valid", valid}};
    const auto voice = impl_->sounds.find(id);
    if (voice != impl_->sounds.end()) {
        if (valid) {
            impl_->apply(*voice->second, impl_->find("emitters", id));
        } else {
            ma_sound_stop(voice->second.get());
        }
    }
}

void AudioService::tickReactions(float windMps, float rain) {
    if (impl_->transport.paused()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const float dt = impl_->transport.dt();
    std::vector<std::string> starts;
    for (const auto& e : impl_->data.at("emitters")) {
        const std::string id = e.at("id");
        impl_->controls[id] = impl_->reactions.evaluate(id, e.at("reaction"), windMps, rain, dt);
        const auto voice = impl_->sounds.find(id);
        const auto& config = e.at("reaction");
        const bool automatic = config.at("automatic");
        const auto pose = impl_->poses.find(id);
        const bool validPose = pose == impl_->poses.end() || pose->second.at("valid").get<bool>();
        const float activity = impl_->controls.at(id).activity;
        if (voice != impl_->sounds.end() && impl_->managed.contains(id) &&
            ((!automatic || !validPose) || (!impl_->suspended.contains(id) &&
             activity <= config.at("stop_threshold").get<float>()))) {
            ma_sound_uninit(voice->second.get());
            impl_->sounds.erase(voice);
            impl_->managed.erase(id);
            continue;
        }
        if (voice != impl_->sounds.end()) {
            impl_->apply(*voice->second, e);
        }
        if (automatic && validPose && !impl_->preview && !impl_->suspended.contains(id) &&
            voice == impl_->sounds.end() && activity >= config.at("start_threshold").get<float>()) {
            const auto failed = impl_->failedAt.find(id);
            if (failed == impl_->failedAt.end() || now - failed->second >= std::chrono::seconds(5)) {
                if (starts.size() < 2) {
                    starts.push_back(id);
                }
            }
        }
    }
    for (const auto& id : starts) {
        try {
            invoke("play", {{"id", id}});
        } catch (const std::exception& error) {
            impl_->failedAt[id] = now;
            impl_->autoErrors[id] = error.what();
        }
    }
}

void AudioService::invalidateSignal(const std::string& key) {
    impl_->reactions.invalidateSignal(key);
}

void AudioService::resetThunder() {
    impl_->stopThunder();
    // Timeline discontinuity clears events without overriding an explicit Stop All suspension.
    const bool suspended = impl_->thunder.status().at("suspended");
    impl_->thunder.reset(suspended);
}

void AudioService::tickThunder() {
    auto& s = *impl_;
    if (s.transport.paused()) {
        return;
    }
    for (auto it = s.thunderVoices.begin(); it != s.thunderVoices.end();) {
        if (ma_sound_at_end(it->sound.get())) {
            ma_sound_uninit(it->sound.get());
            it = s.thunderVoices.erase(it);
        } else {
            ++it;
        }
    }
    for (const auto& event : s.thunder.due()) {
        try {
            require(!s.preview, "Thunder skipped during asset audition");
            require(s.thunderVoices.size() < 8 && s.sounds.size() + s.thunderVoices.size() < 32,
                    "Thunder voice budget reached (8 thunder / 32 total)");
            const auto asset = s.find("assets", event.at("asset_id"));
            auto sound = s.load(asset, "Weather");
            ma_sound_set_looping(sound.get(), false);
            ma_sound_set_volume(sound.get(), event.at("gain"));
            ma_sound_set_pitch(sound.get(), event.at("pitch"));
            ma_sound_set_spatialization_enabled(sound.get(), true);
            const auto& position = event.at("position");
            ma_sound_set_position(sound.get(), position[0], position[1], position[2]);
            ma_sound_set_min_distance(sound.get(), event.at("min_distance"));
            ma_sound_set_max_distance(sound.get(), event.at("max_distance"));
            ma_sound_set_rolloff(sound.get(), 1);
            const auto result = ma_sound_start(sound.get());
            if (result != MA_SUCCESS) {
                ma_sound_uninit(sound.get());
                throw std::runtime_error("Could not start thunder sound");
            }
            const double bytes = asset.at("duration").get<double>() * 48000 * 4;
            s.thunderVoices.push_back({std::move(sound), event, bytes});
        } catch (const std::exception& error) {
            s.thunderError = error.what();
        }
    }
}
std::uint64_t AudioService::generation() const {
    return impl_->generation;
}

bool AudioService::transportPaused() const {
    return impl_->transport.paused();
}

void AudioService::syncTransport(double seconds, bool playing) {
    auto& s = *impl_;
    const auto step = s.transport.advance(seconds, playing);
    if (s.transport.timeline() && step.seek) {
        s.stopAll();
        s.reactions.clear();
        s.controls.clear();
        s.suspended.clear();
        s.thunder.resume();
        ++s.generation;
    }
    s.thunder.clock(s.transport.timeline(), s.transport.time());
    s.reactions.clock(s.transport.timeline(), s.transport.time());
    if (step.resumed) {
        invoke("resume_automatic", json::object());
    }
    s.holdTransport();
}
} // namespace audio
