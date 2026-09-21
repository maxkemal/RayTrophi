#include "RtIpcScreenGi.h"
#include "RtScreenGiBindings.h"
#include <stdexcept>
#include <limits>

namespace {
uint32_t integer(const nlohmann::json& p,const char* key,uint32_t fallback) {
    if (!p.contains(key)) return fallback;
    const auto& v=p[key];
    if (!v.is_number_integer()) throw std::invalid_argument(std::string(key)+" must be an integer");
    const double n=v.get<double>();
    if (n<0 || n>(std::numeric_limits<uint32_t>::max)()) throw std::invalid_argument(std::string(key)+" out of range");
    return v.get<uint32_t>();
}
}
bool dispatchScreenGiIpc(const std::string& method,const nlohmann::json& params,
    const RtIpcTemplateEnqueue& enqueue,nlohmann::json& result) {
    using json=nlohmann::json;
    if (method=="rayfusion.screen_gi") {
        if (!params.empty()) throw std::invalid_argument("screen_gi takes no parameters");
        result=enqueue([](UIContext&) {return screenGiDictionary<json>();});
        return true;
    }
    if (method=="rayfusion.set_screen_gi") {
    if (!params.is_object() || !params.contains("enabled") || !params["enabled"].is_boolean())
        throw std::invalid_argument("required boolean parameter: enabled");
    for (auto i=params.begin();i!=params.end();++i)
        if (i.key()!="enabled" && i.key()!="samples" && i.key()!="filter_radius" && i.key()!="max_distance")
            throw std::invalid_argument("unknown screen GI parameter: "+i.key());
    RayFusion::ScreenGiSettings s;
    s.enabled=params["enabled"].get<bool>();
    if (params.contains("samples")) s.samples=integer(params,"samples",1);
    if (params.contains("filter_radius")) s.filterRadius=integer(params,"filter_radius",2);
    if (params.contains("max_distance")) {
        if (!params["max_distance"].is_number()) throw std::invalid_argument("max_distance must be numeric");
        s.maxDistance=params["max_distance"].get<float>();
    }
    std::string error;
    if (!RayFusion::validateScreenGi(s,error)) throw std::invalid_argument(error);
    result=enqueue([s](UIContext&) {
        std::string error;
        if (!rtapi::setScreenGi(s,error)) throw std::runtime_error(error);
        return screenGiDictionary<json>();
    });
    return true;
    }
    return false;
}
