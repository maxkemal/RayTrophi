#include "Audio/AudioRequestValidation.h"
#include "Audio/AudioReaction.h"
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string_view>

namespace audio {
namespace {
using json = nlohmann::json;
bool listed(std::string_view value, std::initializer_list<std::string_view> names) {
    for (const auto name : names) {
        if (value == name) {
            return true;
        }
    }
    return false;
}
void require(bool valid, const std::string& message) {
    if (!valid) {
        throw std::invalid_argument(message);
    }
}
void required(const json& params, const char* key) {
    require(params.contains(key), std::string("Missing audio parameter: ") + key);
}
}

void validateAudioRequest(const std::string& operation, const json& params) {
    require(params.is_object(), "audio params must be an object");
    for (auto field = params.begin(); field != params.end(); ++field) {
        const auto& key = field.key();
        const auto& value = field.value();
        if (listed(key, {"id", "path", "name", "asset_id", "parent", "bus", "scope",
            "action", "mode", "key", "source", "author", "license", "folder",
            "environment_role", "source_material", "target_material", "assets_root", "track_id", "mixer_bus",
            "from_substance", "to_substance", "preset", "kind", "channel", "clip_policy",
            "analysis_mode", "analysis_view", "history_scope"})) {
            require(value.is_string(), "Audio parameter must be a string: " + key);
        } else if (listed(key, {"loop", "spatial", "muted", "solo", "enabled", "store",
            "background", "original", "continuous", "details", "overwrite", "auto_loop",
            "spectral", "reflections", "diffuse", "clear_history"})) {
            require(value.is_boolean(), "Audio parameter must be boolean: " + key);
        } else if (listed(key, {"ids", "tags", "controls", "events", "gain_curve",
            "pitch_curve", "pan_curve"})) {
            require(value.is_array(), "Audio parameter must be an array: " + key);
            if (key == "ids" || key == "tags") {
                for (const auto& entry : value) {
                    require(entry.is_string(), "Audio array entries must be strings: " + key);
                }
            }
        } else if (listed(key, {"generator", "generator_history"})) {
            require(value.is_null() || value.is_object(),
                "Audio parameter must be an object or null: " + key);
        } else if (listed(key, {"reaction", "acoustics", "config", "patch", "recipe", "options", "effect"})) {
            require(value.is_object(), "Audio parameter must be an object: " + key);
            if (key == "reaction") {
                AudioReaction::validate(value);
            }
        } else if (listed(key, {"position", "direction", "source_position", "listener_position"})) {
            require(value.is_array() && value.size() == 3,
                "Audio vector requires three finite numbers: " + key);
            for (const auto& component : value) {
                require(component.is_number() && std::isfinite(component.get<double>()),
                    "Audio vector requires three finite numbers: " + key);
            }
        } else if (listed(key, {"sequence", "start_frame", "end_frame_exclusive", "seed",
            "points", "fft_size", "time_bins", "fps_num", "fps_den"})) {
            require(value.is_number_integer(), "Audio parameter must be an integer: " + key);
        } else if (listed(key, {"gain", "pitch", "pan", "min_distance", "max_distance",
            "rolloff", "cone_inner", "cone_outer", "cone_gain", "duration_seconds",
            "energy_j", "start_seconds", "end_seconds", "time_seconds", "delta_seconds",
            "source_in_seconds", "source_out_seconds", "phase_seconds", "fade_in_seconds",
            "fade_out_seconds", "max_seconds", "value", "activity_value", "power_w",
            "flow_kg_s", "speed_mps", "ttl_seconds", "intensity", "age_seconds",
            "frequency_hz", "decay_seconds", "brightness"})) {
            require(value.is_number() && std::isfinite(value.get<double>()),
                "Audio parameter must be a finite number: " + key);
        }
        if (key == "id") {
            require(!value.get<std::string>().empty(), "Audio id must be nonempty");
        }
    }
    if (listed(operation, {"physical_configure", "get", "source_runtime", "select", "update", "duplicate", "remove",
        "play", "pause", "stop", "preview", "asset_remove", "asset_relink", "waveform",
        "acoustics_preview", "physical_impact", "physical_history_start",
        "physical_replay_start", "design_reuse", "track_update", "track_remove", "clip_update",
        "clip_remove", "clip_duplicate", "clip_trim", "clip_split", "clip_resize"})) {
        required(params, "id");
    }
    if (listed(operation, {"asset_import", "asset_relink", "append", "preview_file",
        "synth_generate", "physical_bake_start", "export_start"})) {
        required(params, "path");
    }
    if (operation == "physical_impact") {
        required(params, "sequence");
        required(params, "energy_j");
    } else if (operation == "physical_bake_start") {
        require(params.contains("id") || params.contains("scope") || params.contains("ids"),
            "Physical bake requires id or scoped source IDs");
    } else if (operation == "transport_batch" || operation == "acoustics_batch" ||
        operation == "clip_batch") {
        required(params, "action");
    } else if (operation == "playback_update") {
        required(params, "mode");
    } else if (operation == "signal_set" || operation == "thunder_event") {
        required(params, "key");
    } else if (operation == "mixer_update") {
        required(params, "bus");
    } else if (operation == "preview_mute") {
        required(params, "muted");
    } else if (operation == "thunder_update") {
        required(params, "config");
    } else if (operation == "acoustics_boundary") {
        required(params, "from_substance");
        required(params, "to_substance");
    } else if (operation == "track_reorder") {
        required(params, "ids");
    } else if (operation == "clip_create") {
        required(params, "asset_id");
    } else if (operation == "design_render") {
        required(params, "recipe");
    } else if (operation == "design_job_start") {
        required(params, "kind");
    }
    if (operation == "clip_batch") {
        required(params, "ids");
    } else if (operation == "clip_split") {
        required(params, "time_seconds");
    } else if (operation == "clip_trim" || operation == "clip_resize") {
        required(params, "end_seconds");
    } else if (operation == "environment_setup" || operation == "thunder_setup") {
        required(params, "assets_root");
    } else if (operation == "thunder_event") {
        required(params, "position");
    }
}
}
