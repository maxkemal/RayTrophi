// SPDX-License-Identifier: MIT
#include "RtUvBindings.h"

#include "RtIpcMethodRegistry.h"
#include "UI/UvEditorUI.h"
#include "UV/UvService.h"

#include <pybind11/pybind11.h>
#include <array>
#include <stdexcept>

using nlohmann::json;

json invokeUv(const std::string& operation, const json& params) {
    if (operation == "open") {
        if (!params.is_object()) {
            return json{{"__error", "uv_invalid_params"}};
        }
        try {
            UvEditorUI::open(params.value("object", std::string()));
            return json{{"open", true}};
        } catch (const std::exception& error) {
            return json{{"__error", error.what()}};
        }
    }
    return UV::request(operation, params);
}

namespace {

const char* const operations[] = {
    "capabilities", "targets", "info", "layout", "transform", "normalize", "pack", "unwrap", "open",
    "flip_u", "flip_v", "swap_uv", "project_view", "set_corners"
};

// Kept with the adapter, not in the generated aggregate descriptor file.
const MethodParam read_params[] = {
    {"object", "string", true, "Direct flat mesh object name", nullptr, nullptr},
    {"mesh", "int", false, "Sibling mesh ordinal from uv.targets", "0", nullptr}
};
const MethodParam layout_params[] = {
    {"object", "string", true, "Direct flat mesh object name", nullptr, nullptr},
    {"mesh", "int", false, "Sibling mesh ordinal", "0", nullptr},
    {"first_face", "int", false, "First triangle", "0", nullptr},
    {"limit", "int", false, "Page size 1..10000", "1000", nullptr}
};
const MethodParam edit_params[] = {
    {"object", "string", true, "Direct flat mesh object name", nullptr, nullptr},
    {"mesh", "int", false, "Sibling mesh ordinal", "0", nullptr},
    {"revision", "string", true, "Exact revision returned by uv.info/layout", nullptr, nullptr},
    {"allow_texture_change", "bool", true, "Acknowledge that texture rebake is unavailable", nullptr, nullptr},
    {"islands", "array", false, "Transform/repair island IDs; empty means all; projection targets whole mesh", "[]", nullptr},
    {"vertices", "array", false, "Optional nonempty UV vertex IDs instead of islands; edges use endpoints", nullptr, nullptr},
    {"offset", "array", false, "Transform UV offset", "[0,0]", nullptr},
    {"scale", "array", false, "Transform UV scale", "[1,1]", nullptr},
    {"pivot", "array", false, "Transform UV pivot", "[0.5,0.5]", nullptr},
    {"degrees", "float", false, "Transform rotation in degrees", "0", nullptr},
    {"resolution", "int", false, "Atlas resolution 64..8192", "2048", nullptr},
    {"padding", "int", false, "Atlas padding in pixels", "8", nullptr}
};
const MethodParam corner_params[] = {
    {"object", "string", true, "Direct flat mesh object name", nullptr, nullptr},
    {"mesh", "int", false, "Sibling mesh ordinal", "0", nullptr},
    {"revision", "string", true, "Exact revision from uv.info/layout", nullptr, nullptr},
    {"allow_texture_change", "bool", true, "Allow changed texture mapping", nullptr, nullptr},
    {"corners", "array", true, "One numeric [u,v] for every triangle corner, max 60000", nullptr, nullptr}
};
const MethodParam open_params[] = {
    {"object", "string", false, "Initial UV Editor target", "", nullptr}
};

struct Descriptors {
    std::array<MethodDescriptor, 14> entries{};
    Descriptors() {
        const char* names[] = {"uv.capabilities", "uv.targets", "uv.info", "uv.layout",
            "uv.transform", "uv.normalize", "uv.pack", "uv.unwrap", "uv.open",
            "uv.flip_u", "uv.flip_v", "uv.swap_uv", "uv.project_view", "uv.set_corners"};
        const char* summaries[] = {"Inspect UV authoring capabilities", "List direct flat UV targets",
            "Inspect UV layout revision and counts", "Read a bounded UV triangle page",
            "Transform UV islands or explicit vertices", "Uniformly normalize UV selection",
            "Pack existing UVs using xatlas", "Auto unwrap using xatlas", "Open the UV Editor",
            "Flip selected islands across U=0.5", "Flip selected islands across V=0.5",
            "Swap selected UV coordinate axes", "Project the whole mesh from the active camera",
            "Set per-corner UVs with attribute-preserving vertex splits"};
        for (int i = 0; i < 14; ++i) {
            const bool edit = i >= 4 && i != 8;
            const MethodParam* params = nullptr;
            int count = 0;
            if (i == 13) {
                params = corner_params;
                count = 5;
            } else if (edit) {
                params = edit_params;
                count = 12;
            } else if (i == 3) {
                params = layout_params;
                count = 4;
            } else if (i == 2) {
                params = read_params;
                count = 2;
            } else if (i == 8) {
                params = open_params;
                count = 1;
            }
            entries[i] = MethodDescriptor{names[i], "uv", summaries[i],
                "Initial slice: set 0, indexed islands, no seam/pin or texture rebake. "
                "Atlas is synchronous up to 20000 triangles. Corner remaps preserve all vertex "
                "attributes; face order is fixed. Optional vertices selects UV vertex IDs "
                "instead of islands for transform/normalize/flip/swap; edges use endpoint IDs.",
                edit || i == 8 ? "write" : "read", edit || i == 8 ? "SceneWrite" : "Read",
                edit, "object", "mesh|uv|authoring", "uv.info|uv.layout",
                edit ? "uv.info" : nullptr, nullptr, "uv.info",
                edit ? "uv_preview|render_geometry" : nullptr, params, count, true};
            MethodRegistry::instance().registerMethod(entries[i]);
        }
    }
};

const Descriptors descriptors;

} // namespace

void registerUvPython(pybind11::module_& module) {
    namespace py = pybind11;
    auto uv = module.def_submodule("uv", "Flat mesh UV authoring (main thread)");
    for (const char* operation : operations) {
        uv.def(operation, [operation = std::string(operation)](py::kwargs kwargs) {
            const auto codec = py::module_::import("json");
            const std::string encoded = codec.attr("dumps")(kwargs,
                py::arg("allow_nan") = false).cast<std::string>();
            const json result = invokeUv(operation, json::parse(encoded));
            if (result.is_object() && result.contains("__error")) {
                throw std::runtime_error(result.at("__error").get<std::string>());
            }
            return codec.attr("loads")(result.dump());
        }, "Keyword arguments match the uv.* IPC method schema.");
    }
}
