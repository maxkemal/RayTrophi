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
            {"bus", "string", false, "Routing label; mixer arrives in stage 2", "Effects",
                "Master|Weather|Effects|Ambience|Music"},
            {"license", "string", false, "Asset license metadata", "", nullptr},
            {"author", "string", false, "Asset author metadata", "", nullptr},
            {"source", "string", false, "Asset source metadata", "", nullptr},
            {"tags", "array", false, "Asset category tags (strings)", "[]", nullptr},
            {"reaction", "object", false, "Continuous driver mapping; see audio reaction contract",
                "{}", nullptr},
            {"assets_root", "string", false, "Assets library root; defaults to resolved runtime root",
                nullptr, nullptr},
            {"environment_role", "string", false, "Reserved starter environment identity", "",
                "wind|rain"},
            {"key", "string", true, "Runtime signal identity, including domain/contact region",
                nullptr, nullptr},
            {"value", "float", false, "Nonnegative signal magnitude; alternative to flow/speed",
                nullptr, nullptr},
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
            const bool read = op == "list" || op == "get" || op == "status" || op == "domains";
            MethodDescriptor descriptor{};
            descriptor.name = names[i].c_str();
            descriptor.domain = "audio";
            descriptor.summary = "Scene audio service operation";
            descriptor.notes = "See docs/dev/SCENE_AUDIO_STAGE1_API.md for operation parameters and limits.";
            descriptor.access = read ? "read" : "write";
            const bool readsFile = op == "append" || op == "asset_import" || op == "asset_relink" ||
                                   op == "play" || op == "preview" || op == "preview_file" ||
                                   op == "environment_setup" || op == "create" || op == "update" ||
                                   op == "duplicate";
            descriptor.capability = read ? "Read" : (readsFile ? "SceneWrite|FilesRead" : "SceneWrite");
            descriptor.undoable = audio::AudioService::isAuthoring(op);
            descriptor.return_type = "object";
            descriptor.tags = "audio|scene|assets";
            descriptor.verify_with = "audio.list|audio.status";
            schemas.emplace_back();
            for (const auto& key : audio::AudioService::parameters(op)) {
                for (const auto& definition : params) {
                    if (key == definition.name) {
                        auto param = definition;
                        param.required = key == "id" || key == "path" || key == "muted" || key == "key";
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
