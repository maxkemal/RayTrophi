#include "RtIpcReflection.h"
#include "RtReflectionBindings.h"
#include <stdexcept>
#include <limits>

namespace {
uint32_t integer(const nlohmann::json& p, const char* key, uint32_t fallback) {
    if (!p.contains(key)) return fallback;
    const auto& v = p[key];
    if (!v.is_number_integer()) throw std::invalid_argument(std::string(key) + " must be an integer");
    const double n = v.get<double>();
    if (n < 0 || n > (std::numeric_limits<uint32_t>::max)())
        throw std::invalid_argument(std::string(key) + " out of range");
    return v.get<uint32_t>();
}
float number(const nlohmann::json& p, const char* key, float fallback) {
    if (!p.contains(key)) return fallback;
    if (!p[key].is_number()) throw std::invalid_argument(std::string(key) + " must be numeric");
    return p[key].get<float>();
}
}

bool dispatchReflectionIpc(const std::string& method, const nlohmann::json& params,
    const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result) {
    using json = nlohmann::json;
    if (method == "rayfusion.reflections") {
        if (!params.empty()) throw std::invalid_argument("reflections takes no parameters");
        result = enqueue([](UIContext&) { return reflectionDictionary<json>(); });
        return true;
    }
    if (method == "rayfusion.set_reflections") {
        if (!params.is_object() || !params.contains("enabled") || !params["enabled"].is_boolean())
            throw std::invalid_argument("required boolean parameter: enabled");
        for (auto i = params.begin(); i != params.end(); ++i)
            if (i.key() != "enabled" && i.key() != "samples" && i.key() != "roughness_gate" &&
                i.key() != "weight_gate" && i.key() != "max_distance" &&
                i.key() != "follow_quality_preset")
                throw std::invalid_argument("unknown reflection parameter: " + i.key());
        RayFusion::ReflectionSettings s;
        s.enabled = params["enabled"].get<bool>();
        // ★★★★★ SAYISAL BIR KONTROLU ADLANDIRMAK MANUEL KONTROL DEMEKTIR.
        //   Varsayilan olarak butce kalite preset'inden turer. Bir script
        //   `samples` yazip bayragi unutursa degerinin SESSIZCE yok sayilmasi,
        //   "samples=4 yazdim ama degismedi" seklinde geri gelirdi. Bu yuzden
        //   bir sayi adlandirilinca takip kapanir -- ve `follow_quality_preset`
        //   acikca verilirse o kazanir, yani geri donmek de mumkun.
        const bool namedNumeric = params.contains("samples") ||
                                  params.contains("roughness_gate") ||
                                  params.contains("weight_gate");
        s.followQualityPreset = params.contains("follow_quality_preset")
            ? params["follow_quality_preset"].get<bool>()
            : !namedNumeric;
        if (params.contains("follow_quality_preset") &&
            !params["follow_quality_preset"].is_boolean())
            throw std::invalid_argument("follow_quality_preset must be a boolean");
        s.samples = integer(params, "samples", s.samples);
        s.roughnessGate = number(params, "roughness_gate", s.roughnessGate);
        s.weightGate = number(params, "weight_gate", s.weightGate);
        s.maxDistance = number(params, "max_distance", s.maxDistance);
        std::string error;
        if (!RayFusion::validateReflection(s, error)) throw std::invalid_argument(error);
        result = enqueue([s](UIContext&) {
            std::string error;
            if (!rtapi::setReflection(s, error)) throw std::runtime_error(error);
            return reflectionDictionary<json>();
        });
        return true;
    }
    return false;
}
