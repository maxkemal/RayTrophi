#include "RtPhaseGrid.h"

#include <exception>

bool dispatchPhaseGridIpc(const std::string& method, const nlohmann::json& params,
                          const RtIpcTemplateEnqueue& enqueue, nlohmann::json& out) {
    if (method == "fluid.get_phase_grids") {
        const std::string domain = params.at("domain").get<std::string>();
        out = enqueue([domain](UIContext&) {
            try {
                return rtapi::getPhaseGrids(domain);
            } catch (const std::exception& error) {
                return nlohmann::json{{"__error", error.what()}};
            }
        });
        return true;
    }
    if (method == "fluid.set_phase_grid") {
        const std::string domain = params.at("domain").get<std::string>();
        const std::string phase = params.at("phase").get<std::string>();
        const bool inherit = params.value("inherit", false);
        const auto bounds_min = params.value("bounds_min", nlohmann::json{});
        const auto bounds_max = params.value("bounds_max", nlohmann::json{});
        const float voxel = params.value("voxel", 0.1f);
        out = enqueue([domain, phase, inherit, bounds_min, bounds_max, voxel](UIContext&) {
            try {
                return rtapi::setPhaseGrid(domain, phase, inherit, bounds_min, bounds_max, voxel);
            } catch (const std::exception& error) {
                return nlohmann::json{{"__error", error.what()}};
            }
        });
        return true;
    }
    return false;
}
