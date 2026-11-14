// Studio-only Python transport. Shared services and IPC live in the original module.
#include "RtAudioBindings.h"
#include "RtApiInternal.h"
#include "Audio/AudioService.h"
#include "Audio/AudioAuthoring.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

using nlohmann::json;
namespace py = pybind11;

namespace rtpy {
void registerAudioBindings(pybind11::module_& module) {
    namespace py = pybind11;
    auto audioModule = module.def_submodule("audio", "Scene audio authoring and preview");
    for (const auto& operation : audio::AudioService::operations()) {
        audioModule.def(
            operation.c_str(),
            [operation](py::kwargs kwargs) {
                if (!rtapi::g_ctx) {
                    throw py::value_error("rtapi is not bound to a scene");
                }
                const auto jsonModule = py::module_::import("json");
                try {
                    const auto params =
                        nlohmann::json::parse(jsonModule.attr("dumps")(kwargs).cast<std::string>());
                    const auto result =
                        audio::author(*rtapi::g_ctx, rtapi::g_history, operation, params);
                    return jsonModule.attr("loads")(result.dump());
                } catch (const std::exception& error) {
                    throw py::value_error(error.what());
                }
            },
            "Keyword parameters match the corresponding audio.* IPC operation.");
    }
}
} // namespace rtpy
