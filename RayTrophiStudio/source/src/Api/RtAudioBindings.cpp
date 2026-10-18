#include "RtAudioBindings.h"
#include "RtApiInternal.h"
#include "Audio/AudioService.h"
#include "Audio/AudioAuthoring.h"
#include "RtIpcMethodRegistry.h"
#include <pybind11/pybind11.h>
#include <iterator>
#include <deque>

namespace {
struct RegisterAudioMethods {
    std::vector<std::string> names;
    std::vector<std::vector<MethodParam>> schemas;
    // The registry retains descriptor addresses for the lifetime of the process.
    std::deque<MethodDescriptor> descriptors;
    RegisterAudioMethods() {
        names.reserve(audio::AudioService::operations().size());
        schemas.reserve(audio::AudioService::operations().size());
        for (const auto& op : audio::AudioService::operations()) {
            names.push_back("audio." + op);
        }
        static const MethodParam params[] = {
            {"generator_history", "object", false,
                "Nullable versioned duration_seconds/controls/events history stored with a procedural source",
                "null", nullptr},
            {"store", "bool", false, "Persist a finished physical take on its source", "false", nullptr},
            {"start_frame", "int", false, "Physical replay seek/preroll start, within the sample range",
                "0", nullptr},
            {"controls", "array", false, "Physical bake sample-frame/activity/gain control history",
                "[]", nullptr},
            {"events", "array", false, "Physical bake sample-frame/sequence/energy_j events",
                "[]", nullptr},
            {"sequence", "int", true, "Monotonic per-resident-source event identity, 0..2^53-1",
                nullptr, nullptr},
            {"energy_j", "float", true, "Positive excitation energy in joules, at most 1e9",
                nullptr, nullptr},
            {"generator", "object", false,
                "Nullable procedural recipe: flow/friction/rolling, contact_activity, deformation, fracture_activity or impact; see PHYSICAL_AUDIO_GENERATION.md",
                "null", nullptr},
            {"scope", "string", false, "Source scope; selected/group require explicit ids",
                "selected", "selected|group|scene"},
            {"acoustics", "object", false,
                "Persistent source acoustic policy: mode scene/custom/off and config", nullptr, nullptr},
            {"mixer_bus", "string", false, "Mixer channel to highlight", nullptr,
                "Master|Weather|Effects|Ambience|Music"},
            {"spectral", "bool", false, "Frequency dependent live transmission", "true", nullptr},
            {"reflections", "bool", false, "Audition up to six verified specular paths", "false", nullptr},
            {"diffuse", "bool", false, "Single-bounce Lambert transport; omnidirectional sources only", "false", nullptr},
            {"source_position", "array", false, "Explicit source world position; mutually exclusive with id", nullptr, nullptr},
            {"listener_position", "array", false, "Listener world position; defaults to the active camera", nullptr, nullptr},
            {"meters_per_unit", "float", false, "Scene unit scale, finite 1e-6..1e6", "1", nullptr},
            {"sound_speed_mps", "float", false, "Authored sound speed, 100..2000 m/s", "343", nullptr},
            {"max_triangles", "int", false, "Acoustic snapshot triangle budget, 1..500000", "100000", nullptr},
            {"max_hits", "int", false, "Unique acoustic crossing limit, 1..256", "128", nullptr},
            {"include_hidden", "bool", false, "Include render-hidden flat acoustic surfaces", "true", nullptr},
            {"boundary_mode", "string", false, "Diagnostic or explicit independent thin-boundary model", "diagnostic", "diagnostic|thin_surface|object_barrier"},
            {"kind", "string", false, "Background task kind", nullptr, "render|analysis"},
            {"options", "object", false, "Background analysis options", nullptr, nullptr},
            {"audition_start_seconds", "float", false, "Arrangement audition range start", "0", nullptr},
            {"audition_end_seconds", "float", false, "Arrangement audition range end; span 0.05..30", "5", nullptr},
            {"selected_clips", "array", false, "Unique existing clip IDs; primary belongs to selection", "[]", nullptr},
            {"designer_envelope", "string", false, "Selected clip envelope view", "none", "none|gain|pan"},
            {"action", "string", true, "Atomic selected-clip operation", nullptr, "move|duplicate|remove|align_start|align_end|update"},
            {"delta_seconds", "float", false, "Signed batch movement; duplicate defaults to group width", nullptr, nullptr},
            {"patch", "object", false, "Shared gain/pan/muted/bus/track_id patch for update", nullptr, nullptr},
            {"selected_track", "string", false, "Existing track selection; empty clears", "", nullptr},
            {"track_id", "string", false, "Clip track; empty keeps legacy bus routing", "", nullptr},
            {"ids", "array", true, "Complete ordered track ID permutation", nullptr, nullptr},
            {"clip_policy", "string", false, "Nonempty track removal policy", "reject", "reject|detach|delete"},
            {"designer_edit_view", "string", false, "Source editor or existing clip arrangement",
                "source", "source|arrange"},
            {"from_substance", "string", true, "Defined incident acoustic medium", nullptr, nullptr},
            {"to_substance", "string", true, "Defined transmitted acoustic medium", nullptr, nullptr},
            {"incidence_degrees", "float", false, "Angle from boundary normal, 0..89.999", "0", nullptr},
            {"transmitted_distance_meters", "float", false, "Path length in transmitted medium, 0..1e6", "0", nullptr},
            {"acoustic_all_sources", "bool", false, "Sum up to four spatial manual sources", "false", nullptr},
            {"acoustic_band", "int", false, "-1 broadband or band index 0..5 (125..4000 Hz)", "-1", nullptr},
            {"acoustic_reflection_samples", "int", false, "First diffuse bounce samples 0..2", "1", nullptr},
            {"designer_canvas_zoom", "float", false, "Shared canvas zoom 1..8", "1", nullptr},
            {"designer_canvas_pan_x", "float", false, "Shared horizontal canvas pan 0..1", "0", nullptr},
            {"designer_canvas_pan_y", "float", false, "Shared vertical canvas pan 0..1", "0", nullptr},
            {"designer_layer", "int", false, "Selected production layer slot 0..7", "0", nullptr},
            {"designer_tools_open", "bool", false, "Expand bottom studio tools panel", "true", nullptr},
            {"designer_properties_open", "bool", false, "Expand bottom studio properties panel", "true", nullptr},
            {"continuous", "bool", false, "Full STFT time coverage; false samples time_bins windows", "true", nullptr},
            {"effect", "object", true, "Validated single effect for its visual control curve", nullptr, nullptr},
            {"details", "bool", false, "Include rendered graphs; false returns transport/revision only", "true", nullptr},
            {"fft_size", "int", false, "Power-of-two FFT size 512..8192", "2048", nullptr},
            {"time_bins", "int", false, "Aggregated STFT time columns 4..128", "64", nullptr},
            {"channel", "string", false, "Spectral channel; mix averages powers", "mix", "mix|left|right"},
            {"designer_page", "string", false, "Sound designer workflow", "generate", "generate|edit|effects|analyze"},
            {"designer_zoom", "float", false, "Source waveform zoom 1..2048", "1", nullptr},
            {"designer_pan", "float", false, "Source waveform window placement 0..1", "0", nullptr},
            {"designer_effect", "int", false, "Selected effect slot 0..7", "0", nullptr},
            {"designer_eq_band", "int", false, "Shared selected EQ band 0..7", "0", nullptr},
            {"designer_eq_range_db", "float", false, "EQ vertical half-range, 6..288 dB", "24", nullptr},
            {"designer_eq_auto_range", "bool", false, "Fit EQ response vertically", "true", nullptr},
            {"designer_eq_height", "float", false, "EQ frame height 180..600", "280", nullptr},
            {"designer_panel_zoom", "float", false, "Designer panel scale 0.5..1.5", "1", nullptr},
            {"designer_panel_pan_x", "float", false, "Whole-panel horizontal scroll 0..1", "0", nullptr},
            {"designer_panel_pan_y", "float", false, "Whole-panel vertical scroll 0..1", "0", nullptr},
            {"recipe", "object", true, "Validated sound design v3 recipe; see AUDIO_SOUND_DESIGN.md",
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
            {"background", "bool", false, "Render Arrange preview on worker", "false", nullptr},
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
            const bool read = op == "acoustics_get" || op == "source_runtime" || op == "acoustics_boundary" ||
                              op == "physical_bake_status" || op == "physical_history_status" ||
                              op == "physical_replay_status" ||
                              op == "acoustics_debug" || op == "acoustics_preview_status" ||
                              op == "acoustics_trace" || op == "list" || op == "get" ||
                              op == "status" || op == "domains" ||
                              op == "mixer_get" || op == "arrange_status" || op == "track_list" || op == "clip_list" || op == "export_status" ||
                              op == "waveform" || op == "capture_status" || op == "workspace_get" ||
                              op == "synth_presets" || op == "telemetry_get" ||
                              op == "design_job_status" || op == "design_status" || op == "design_destination" || op == "design_catalogue" ||
                              op == "design_analyze" || op == "design_waveform" || op == "design_effect_visual";
            MethodDescriptor descriptor{};
            descriptor.name = names[i].c_str();
            descriptor.domain = "audio";
            descriptor.summary = "Scene audio service operation";
            descriptor.notes = op == "acoustics_get" || op == "acoustics_update" ||
                op == "acoustics_reset_sources" || op == "acoustics_preview" || op == "acoustics_batch"
                ? "Persistent scene defaults and source scene/custom/off policies. "
                  "Dry/processed bypass is session-only. See docs/dev/AUDIO_ACOUSTIC_SETTINGS.md."
                : op == "source_runtime"
                ? "Resolved world pose and pre-mixer reaction controls. "
                  "See SCENE_AUDIO_PHYSICS_SOURCES.md."
                : op == "physical_impact"
                ? "Explicit energy excitation for an impact generator. Monotonic resident-source "
                  "sequence deduplication; not an automatic solver-contact event. "
                  "See docs/dev/PHYSICAL_AUDIO_GENERATION.md."
                : op.rfind("physical_replay_", 0) == 0
                ? "Worker-prepared, sample-scheduled native generator replay. Stop the source first; "
                  "explicit apply begins playback. See docs/dev/PHYSICAL_AUDIO_REPLAY.md."
                : op.rfind("physical_bake_", 0) == 0 || op.rfind("physical_history_", 0) == 0
                ? "Bounded sampled control history and worker dry-mono bake using the shared "
                  "physical DSP. Explicit apply imports an asset; no acoustic export parity. "
                  "See docs/dev/PHYSICAL_AUDIO_BAKE.md."
                : op.rfind("acoustics_preview", 0) == 0
                ? "See docs/dev/SCENE_ACOUSTICS_PREVIEW.md. Session spectral transmission and sparse reflections."
                : op == "acoustics_boundary"
                ? "Planar equivalent-fluid interface probe; no GPU paths or scene mutation. See SCENE_ACOUSTICS_DEBUG.md."
                : op == "acoustics_debug"
                ? "See docs/dev/SCENE_ACOUSTICS_DEBUG.md. Read existing results; no GPU dispatch."
                : op == "acoustics_trace"
                ? "See docs/dev/SCENE_ACOUSTICS_TRACE.md. Read-only snapshot; no audible propagation."
                : op == "telemetry_get" || op == "telemetry_update"
                ? "See docs/dev/SCENE_AUDIO_TELEMETRY_POLICY.md. Off by default; never auto-enabled."
                : (op.rfind("design_", 0) == 0 ? "See docs/dev/AUDIO_SOUND_DESIGN.md. Transient preview; explicit asset save."
                : "See docs/dev/AUDIO_PRE_UI_DELIVERY.md and SCENE_AUDIO_STAGE1_API.md.");
            descriptor.access = read ? "read" : "write";
            const bool readsFile = op == "arrange_preview" || (op.rfind("track_", 0) == 0 && op != "track_list") || op == "append" || op == "asset_import" || op == "asset_relink" ||
                                   op == "play" || op == "play_all" || op == "transport_batch" || op == "preview" || op == "preview_file" ||
                                   op == "environment_setup" || op == "create" || op == "update" ||
                                   op == "duplicate" || op == "thunder_setup" ||
                                   op == "thunder_update" || op == "thunder_event" ||
                                   op == "resume_automatic" || op == "playback_update" ||
                                   op == "clip_batch" || op == "clip_create" || op == "clip_update" || op == "clip_duplicate" ||
                                   op == "clip_trim" || op == "clip_split" || op == "clip_resize" ||
                                   op == "capture_start" || op == "capture_finish";
            descriptor.capability = (op == "synth_generate" || op == "design_save" ||
                op == "physical_bake_start" || op == "physical_bake_apply" ||
                op == "physical_bake_cancel" || op == "physical_history_finish") ?
                "SceneWrite|FilesRead|FilesWrite" :
                (op == "design_job_start" || op == "design_render" || op == "design_undo" || op == "design_redo" ||
                 op == "design_reuse") ? "SceneWrite|FilesRead" :
                (op == "design_destination" || op == "design_waveform") ? "Read|FilesRead" :
                op == "export_cancel" ? "Read|FilesWrite" :
                (op == "export_start" ? "Read|FilesRead|FilesWrite" :
                (op == "waveform" ? "Read|FilesRead" :
                 (read ? "Read" : (readsFile ? "SceneWrite|FilesRead" : "SceneWrite"))));
            descriptor.undoable = audio::AudioService::isAuthoring(op);
            descriptor.return_type = op == "acoustics_preview_status" || op == "track_list" || op == "track_reorder" ? "array" : "object";
            descriptor.tags = "audio|scene|assets";
            descriptor.verify_with = op.rfind("design_", 0) == 0
                ? "audio.design_status|audio.list" : "audio.list|audio.status";
            schemas.emplace_back();
            for (const auto& key : audio::AudioService::parameters(op)) {
                for (const auto& definition : params) {
                    if (key == definition.name) {
                        auto param = definition;
                        param.required = key == "id" || key == "path" ||
                            (op == "acoustics_boundary" &&
                                (key == "from_substance" || key == "to_substance")) ||
                            (op == "preview_mute" && key == "muted") ||
                            key == "key" || key == "config" || key == "mode" ||
                            (key == "kind" && op == "design_job_start") || key == "action" || key == "ids" || key == "effect" || (key == "recipe" && op == "design_render") || key == "end_frame_exclusive" || key == "end_seconds" || key == "time_seconds" ||
                            (op == "clip_create" && key == "asset_id") ||
                            (op == "thunder_event" && key == "position");
                        if (op == "acoustics_preview" && key == "enabled") {
                            param.required = true;
                            param.description = "Enable session spectral/echo audition for a spatial source";
                        }
                        if (op == "physical_impact") {
                            param.required = true;
                        }
                        if (op == "physical_bake_start" &&
                            (key == "id" || key == "ids" || key == "end_frame_exclusive")) {
                            param.required = false;
                            param.default_value = nullptr;
                        }
                        if (op == "physical_bake_start") {
                            if (key == "start_frame" || key == "end_frame_exclusive") {
                                param.description = "48 kHz history sample index; output range is half-open";
                            } else if (key == "ids") {
                                param.description = "Unique procedural source IDs; group accepts 1..32";
                            } else if (key == "scope") {
                                param.default_value = nullptr;
                                param.description = "Alternative to id; every scoped source must be procedural";
                            }
                        }
                        if ((op == "physical_bake_start" || op == "physical_replay_start") &&
                            key == "duration_seconds") {
                            param.description = "Physical history output duration, 0.05..60 seconds";
                            param.default_value = "2";
                        }
                        if (op == "physical_history_start" && key == "duration_seconds") {
                            param.description = "Maximum sampled take duration, 0.05..60 seconds";
                            param.default_value = "60";
                        }
                        if (op == "physical_history_finish" && key == "path") {
                            param.required = false;
                            param.default_value = nullptr;
                            param.description = "Optional new WAV path; starts a bake job from the finished take";
                        }
                        if (op == "acoustics_reset_sources") {
                            param.required = false;
                        }
                        if (op == "acoustics_update" && key == "config") {
                            param.description = "Patch of saved scene acoustic defaults";
                        }
                        if (op == "transport_batch" || op == "acoustics_batch") {
                            param.required = key == "action";
                            if (key == "ids") {
                                param.description = "Unique source IDs; selected=1, group=1..32, scene omits ids";
                                param.default_value = "[]";
                            } else if (key == "action") {
                                param.description = "Scoped runtime operation";
                                param.enum_values = op == "transport_batch" ? "play|pause|stop" :
                                    "configure|dry|processed";
                            } else if (key == "config") {
                                param.description = "Configure only: enabled plus acoustic preview controls";
                                param.default_value = nullptr;
                            }
                        }
                        if (op == "acoustics_preview" && key == "boundary_mode") {
                            param.default_value = "object_barrier";
                            param.enum_values = "object_barrier|thin_surface";
                        }
                        if (op == "acoustics_trace" && key == "id") {
                            param.required = false;
                            param.description = "Authored source id; mutually exclusive with source_position";
                        }
                        if (op == "arrange_preview" && (key == "ids" || key == "end_seconds")) {
                            param.required = false;
                            param.default_value = key == "ids" ? "[]" : "5";
                        }
                        if (op == "clip_batch" && key == "time_seconds") {
                            param.required = false;
                        }
                        if (op == "design_waveform" && (key == "start_seconds" || key == "end_seconds")) {
                            param.required = false;
                            param.default_value = nullptr;
                        }
                        if (op == "design_analyze" && key == "points") {
                            param.default_value = "256";
                            param.description = "Log-frequency columns 16..1024";
                        }
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
            descriptors.push_back(descriptor);
            MethodRegistry::instance().registerMethod(descriptors.back());
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
