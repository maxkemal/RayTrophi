#include "RtAudioBindings.h"
#include "RtApiInternal.h"
#include "Audio/AudioService.h"
#include "Audio/AudioAuthoring.h"
#include "RtIpcMethodRegistry.h"
#include <pybind11/pybind11.h>
#include <iterator>

namespace {
struct RegisterAudioMethods {
    std::vector<std::string> names;
    std::vector<std::vector<MethodParam>> schemas;
    RegisterAudioMethods() {
        names.reserve(audio::AudioService::operations().size());
        schemas.reserve(audio::AudioService::operations().size());
        for (const auto& op : audio::AudioService::operations()) {
            names.push_back("audio." + op);
        }
        static const MethodParam params[] = {
            {"recipe", "object", true, "Validated sound design v2 recipe; see AUDIO_SOUND_DESIGN.md",
                nullptr, nullptr},
            {"original", "bool", false, "Audition unprocessed input", "false", nullptr},
            {"folder", "string", false, "Output folder; defaults to current project/assets/audio/generated",
                nullptr, nullptr},
            {"designer_open", "bool", false, "Show sound designer in Audio bottom editor", "false", nullptr},
            {"auto_loop", "bool", false,
                "Resize beyond source range by looping; defaults false", "false", nullptr},
            {"enabled", "bool", false, "Manually collect domain physics audio telemetry; default off",
                "false", nullptr},
            {"preset", "string", false, "Offline procedural preset", "metal", "tone|wind|metal|wood"},
            {"seed", "int", false, "Deterministic uint32 synthesis seed", "1", nullptr},
            {"frequency_hz", "float", false, "Synthesis frequency 20..4000 Hz; preset default", nullptr, nullptr},
            {"decay_seconds", "float", false, "Impact decay 0.01..10 seconds; preset default", nullptr, nullptr},
            {"brightness", "float", false, "Noise cutoff and modal damping 0..1", "0.5", nullptr},
            {"selected_clip", "string", false, "Existing clip id, or empty to clear selection", "", nullptr},
            {"snap", "bool", false, "Snap audio timeline edits to scene frames", "true", nullptr},
            {"waveforms", "bool", false, "Display waveform envelopes on timeline clips", "true", nullptr},
            {"inspector_open", "bool", false, "Open or close the optional audio inspector", "false", nullptr},
            {"max_seconds", "float", false, "Capture limit in seconds 1..600", "120", nullptr},
            {"pitch_curve", "array", false, "Ordered [time,pitch multiplier] knots", "[]", nullptr},
            {"pan_curve", "array", false, "Ordered [time,stereo balance] knots", "[]", nullptr},
            {"points", "int", false, "Waveform envelope bins 16..4096", "512", nullptr},
            {"start_seconds", "float", false, "Clip start / trim start in timeline seconds", "0", nullptr},
            {"end_seconds", "float", true, "Trim end in timeline seconds", nullptr, nullptr},
            {"time_seconds", "float", true, "Split position strictly inside the clip", nullptr, nullptr},
            {"source_in_seconds", "float", false, "Asset source-in offset in seconds", "0", nullptr},
            {"source_out_seconds", "float", false, "Source-out; defaults to asset duration", nullptr, nullptr},
            {"duration_seconds", "float", false, "Clip duration; defaults to nonloop source range", nullptr, nullptr},
            {"phase_seconds", "float", false, "Initial source phase in the selected range", "0", nullptr},
            {"pan", "float", false, "Stereo balance -1..1", "0", nullptr},
            {"fade_in_seconds", "float", false, "Clip fade-in duration", "0", nullptr},
            {"fade_out_seconds", "float", false, "Clip fade-out duration", "0", nullptr},
            {"gain_curve", "array", false, "Up to 4096 ordered [time,gain] envelope knots", "[]", nullptr},
            {"envelope_offset_seconds", "float", false, "Preserved envelope offset after trim/split", "0", nullptr},
            {"envelope_duration_seconds", "float", false, "Envelope horizon; defaults to clip duration", nullptr, nullptr},
            {"start_frame", "int", false, "Offline range start frame, inclusive", "0", nullptr},
            {"end_frame_exclusive", "int", true, "Offline range end frame, exclusive", nullptr, nullptr},
            {"fps_num", "int", false, "Rational frames-per-second numerator", "24", nullptr},
            {"fps_den", "int", false, "Rational frames-per-second denominator", "1", nullptr},
            {"overwrite", "bool", false, "Atomically replace an existing completed WAV", "false", nullptr},
            {"solo", "bool", false, "Solo a child mixer bus; multiple solos combine",
                nullptr, nullptr},
            {"mode", "string", true, "Scene audio transport policy", nullptr, "continuous|timeline"},
            {"config", "object", true, "Thunder settings patch including near/far mono asset ids",
                nullptr, nullptr},
            {"intensity", "float", false, "Thunder event strength 0..100000; 30 is reference",
                "30", nullptr},
            {"age_seconds", "float", false, "Elapsed visual time since event onset, 0..2",
                "0", nullptr},
            {"id", "string", false, "Persistent asset or source id (operation dependent)", nullptr, nullptr},
            {"path", "string", false, "WAV/FLAC/MP3 file for import/relink", nullptr, nullptr},
            {"asset_id", "string", false, "Asset to bind to a source", nullptr, nullptr},
            {"position", "vec3", false, "Source world position in scene units", "[0,0,0]", nullptr},
            {"spatial", "bool", false, "Mono positional or stereo ambience; create auto-selects",
                nullptr, nullptr},
            {"gain", "float", false, "Linear amplitude 0..16", "1", nullptr},
            {"pitch", "float", false, "Playback pitch ratio 0.125..8", "1", nullptr},
            {"loop", "bool", false, "Repeat source", "false", nullptr},
            {"muted", "bool", false, "Preview output mute only", "false", nullptr},
            {"name", "string", false, "Source display name", "Audio Source", nullptr},
            {"parent", "string", false, "Flat scene node name; empty means world space", "", nullptr},
            {"direction", "vec3", false, "Nonzero source direction", "[0,0,-1]", nullptr},
            {"min_distance", "float", false, "Minimum attenuation distance > 0", "1", nullptr},
            {"max_distance", "float", false, "Maximum distance >= min_distance", "100", nullptr},
            {"rolloff", "float", false, "Distance rolloff 0..10", "1", nullptr},
            {"cone_inner", "float", false, "Inner angle in degrees 0..360", "360", nullptr},
            {"cone_outer", "float", false, "Outer angle >= inner, <=360", "360", nullptr},
            {"cone_gain", "float", false, "Outer cone amplitude 0..1", "1", nullptr},
            {"bus", "string", false, "Scene mixer routing / mixer bus identity", "Effects",
                "Master|Weather|Effects|Ambience|Music"},
            {"license", "string", false, "Asset license metadata", "", nullptr},
            {"author", "string", false, "Asset author metadata", "", nullptr},
            {"source", "string", false, "Asset source metadata", "", nullptr},
            {"tags", "array", false, "Asset category tags (strings)", "[]", nullptr},
            {"reaction", "object", false,
                "Driver mapping; domain_channel: mixed_contact|dem_contact|dem_friction|dem_rolling|"
                "fluid_flow|gas_flow|mpm_deformation|mpm_fracture; input_metric: energy_j|power_w|value",
                "{}", nullptr},
            {"assets_root", "string", false, "Assets library root; defaults to resolved runtime root",
                nullptr, nullptr},
            {"environment_role", "string", false, "Reserved starter environment identity", "",
                "wind|rain"},
            {"key", "string", true, "Runtime signal identity, including domain/contact region",
                nullptr, nullptr},
            {"value", "float", false, "Nonnegative signal magnitude; alternative to flow/speed",
                nullptr, nullptr},
            {"power_w", "float", false, "Optional measured power metadata 0..1e9 W",
                nullptr, nullptr},
            {"activity_value", "float", false, "Optional nonnegative physical activity metadata", nullptr, nullptr},
            {"flow_kg_s", "float", false, "Mass flow; requires speed_mps, excludes value",
                nullptr, nullptr},
            {"speed_mps", "float", false, "Contact speed; requires flow_kg_s, excludes value",
                nullptr, nullptr},
            {"source_material", "string", false, "Producer material label", "", nullptr},
            {"target_material", "string", false, "Receiver material label", "", nullptr},
            {"ttl_seconds", "float", false, "Signal expires after 0.01..10 seconds", "0.5", nullptr}
        };
        for (std::size_t i = 0; i < names.size(); ++i) {
            const auto& op = audio::AudioService::operations()[i];
            const bool read = op == "list" || op == "get" || op == "status" || op == "domains" ||
                              op == "mixer_get" || op == "clip_list" || op == "export_status" ||
                              op == "waveform" || op == "capture_status" || op == "workspace_get" ||
                              op == "synth_presets" || op == "telemetry_get" ||
                              op == "design_status" || op == "design_destination";
            MethodDescriptor descriptor{};
            descriptor.name = names[i].c_str();
            descriptor.domain = "audio";
            descriptor.summary = "Scene audio service operation";
            descriptor.notes = op == "telemetry_get" || op == "telemetry_update"
                ? "See docs/dev/SCENE_AUDIO_TELEMETRY_POLICY.md. Off by default; never auto-enabled."
                : (op.rfind("design_", 0) == 0 ? "See docs/dev/AUDIO_SOUND_DESIGN.md. Transient preview; explicit asset save."
                : "See docs/dev/AUDIO_PRE_UI_DELIVERY.md and SCENE_AUDIO_STAGE1_API.md.");
            descriptor.access = read ? "read" : "write";
            const bool readsFile = op == "append" || op == "asset_import" || op == "asset_relink" ||
                                   op == "play" || op == "preview" || op == "preview_file" ||
                                   op == "environment_setup" || op == "create" || op == "update" ||
                                   op == "duplicate" || op == "thunder_setup" ||
                                   op == "thunder_update" || op == "thunder_event" ||
                                   op == "resume_automatic" || op == "playback_update" ||
                                   op == "clip_create" || op == "clip_update" || op == "clip_duplicate" ||
                                   op == "clip_trim" || op == "clip_split" || op == "clip_resize" ||
                                   op == "capture_start" || op == "capture_finish";
            descriptor.capability = (op == "synth_generate" || op == "design_save") ? "SceneWrite|FilesRead|FilesWrite" :
                op == "design_render" ? "SceneWrite|FilesRead" :
                op == "design_destination" ? "Read|FilesRead" :
                op == "export_cancel" ? "Read|FilesWrite" :
                (op == "export_start" ? "Read|FilesRead|FilesWrite" :
                (op == "waveform" ? "Read|FilesRead" :
                 (read ? "Read" : (readsFile ? "SceneWrite|FilesRead" : "SceneWrite"))));
            descriptor.undoable = audio::AudioService::isAuthoring(op);
            descriptor.return_type = "object";
            descriptor.tags = "audio|scene|assets";
            descriptor.verify_with = op.rfind("design_", 0) == 0
                ? "audio.design_status|audio.list" : "audio.list|audio.status";
            schemas.emplace_back();
            for (const auto& key : audio::AudioService::parameters(op)) {
                for (const auto& definition : params) {
                    if (key == definition.name) {
                        auto param = definition;
                        param.required = key == "id" || key == "path" ||
                            (op == "preview_mute" && key == "muted") ||
                            key == "key" || key == "config" || key == "mode" ||
                            key == "recipe" || key == "end_frame_exclusive" || key == "end_seconds" || key == "time_seconds" ||
                            (op == "clip_create" && key == "asset_id") ||
                            (op == "thunder_event" && key == "position");
                        if (op == "design_save" && key == "name") {
                            param.default_value = "Sound";
                            param.description = "Safe sound filename; collisions receive numeric suffixes";
                        }
                        if (op.rfind("design_", 0) == 0 && key == "path") {
                            param.required = false;
                            param.description = "Optional explicit new WAV path; default uses folder/name";
                        }
                        if (op == "synth_generate") {
                            if (key == "duration_seconds") {
                                param.description = "Synthesis duration 0.05..10 seconds";
                                param.default_value = "2";
                            } else if (key == "gain") {
                                param.description = "Synthesis target peak 0..1";
                                param.default_value = "0.5";
                            } else if (key == "path") {
                                param.description = "New absolute WAV path; existing files are rejected";
                            }
                        }
                        if (op == "mixer_update") {
                            param.required = key == "bus";
                            param.default_value = nullptr;
                            if (key == "gain") {
                                param.description = "Linear mixer gain 0..4; omitted fields stay unchanged";
                            }
                        }
                        if (op == "clip_trim" && key == "start_seconds") {
                            param.required = true;
                        }
                        if (param.required) {
                            param.default_value = nullptr;
                        }
                        schemas.back().push_back(param);
                    }
                }
            }
            descriptor.params = schemas.back().data();
            descriptor.param_count = int(schemas.back().size());
            descriptor.documented = true;
            MethodRegistry::instance().registerMethod(descriptor);
        }
    }
} registration;
}

bool dispatchAudioIpc(const std::string& method, const nlohmann::json& params,
                      const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result) {
    if (method.rfind("audio.", 0) != 0) {
        return false;
    }
    result = enqueue([operation = method.substr(6), params](UIContext& ctx) {
        try {
            return audio::author(ctx, rtapi::g_history, operation, params);
        } catch (const std::exception& error) {
            return nlohmann::json{{"__error", error.what()}, {"code", "audio_error"}};
        }
    });
    return true;
}

namespace rtpy {
void registerAudioBindings(pybind11::module_& module) {
    namespace py = pybind11;
    auto audioModule = module.def_submodule("audio", "Scene audio authoring and preview");
    for (const auto& operation : audio::AudioService::operations()) {
        audioModule.def(operation.c_str(), [operation](py::kwargs kwargs) {
            if (!rtapi::g_ctx) {
                throw py::value_error("rtapi is not bound to a scene");
            }
            const auto jsonModule = py::module_::import("json");
            try {
                const auto params = nlohmann::json::parse(
                    jsonModule.attr("dumps")(kwargs).cast<std::string>());
                const auto result = audio::author(*rtapi::g_ctx, rtapi::g_history, operation, params);
                return jsonModule.attr("loads")(result.dump());
            } catch (const std::exception& error) {
                throw py::value_error(error.what());
            }
        }, "Keyword parameters match the corresponding audio.* IPC operation.");
    }
}
}
