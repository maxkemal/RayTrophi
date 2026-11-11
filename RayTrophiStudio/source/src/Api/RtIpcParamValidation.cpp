/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtIpcParamValidation.cpp
 * Date:          November 2026
 * License:       MIT
 * =========================================================================
 */

#include "RtIpcParamValidation.h"

#include "RtIpcMethodRegistry.h"

#include <cstring>

namespace rtipc_validation {
namespace {

template <typename Predicate>
bool allElements(const nlohmann::json& value, Predicate is) {
    if (!value.is_array()) return false;
    for (const auto& element : value)
        if (!is(element)) return false;
    return true;
}

bool isNumber(const nlohmann::json& value) { return value.is_number(); }
bool isString(const nlohmann::json& value) { return value.is_string(); }

// Only the shapes whose conversion can abort are tested here. A kind this
// function does not know is let through: the table is generated, and an
// unknown spelling must never turn into a rejected valid call.
bool matches(const char* kind, const nlohmann::json& value) {
    if (std::strcmp(kind, "string") == 0) return value.is_string();
    if (std::strcmp(kind, "number") == 0) return value.is_number();
    if (std::strcmp(kind, "bool") == 0) return value.is_boolean();
    if (std::strcmp(kind, "vec3") == 0)
        return value.is_array() && value.size() == 3 &&
               allElements(value, isNumber);
    if (std::strcmp(kind, "string[]") == 0)
        return allElements(value, isString);
    if (std::strcmp(kind, "number[]") == 0)
        return allElements(value, isNumber);
    return true;
}

} // namespace

bool checkParams(const std::string& method, const nlohmann::json& params,
                 std::string& error) {
    if (!params.is_object()) return true;  // the transport already rejects this
    const MethodDescriptor* descriptor = MethodRegistry::instance().find(method);
    if (!descriptor || !descriptor->params) return true;
    for (int i = 0; i < descriptor->param_count; ++i) {
        const MethodParam& param = descriptor->params[i];
        if (!param.check && !param.check_required) continue;
        const auto found = params.find(param.name);
        const bool missing = found == params.end();
        // ★ null is a wrong type, not "absent": value("k", default) and
        // get<T>() both convert it and abort. Keys that really accept null
        // probe it with is_null(), and the generator leaves those unchecked.
        const bool invalid = !missing && param.check && !matches(param.check, *found);
        if ((missing && param.check_required) || invalid) {
            error = std::string("missing or invalid ") +
                    (param.check ? param.check : "any") + " param: " + param.name;
            return false;
        }
    }
    return true;
}

} // namespace rtipc_validation
