/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtIpcParamValidation.h
 * Date:          November 2026
 * License:       MIT
 * =========================================================================
 * Pre-dispatch parameter validation for every IPC method.
 *
 * This build compiles nlohmann::json with JSON_NOEXCEPTION in most
 * translation units, so `params.at("k").get<T>()` on a missing key, a null or
 * a wrong-typed value calls std::abort() and takes the whole application down
 * (2026-11-09: `fluid.get_seed_sources` with `domain: null`). Instead of
 * guarding the hundreds of read sites one by one, dispatchMethod() checks the
 * request against the generated descriptor table before any handler runs.
 * =========================================================================
 */

#pragma once

#include <string>

#include "json.hpp"

namespace rtipc_validation {

// Checks `params` against the MethodParam rows registered for `method`.
// Returns true when the call may be dispatched. On false, `error` holds
// "missing or invalid <kind> param: <key>". A method without a descriptor,
// or a key without a `check` kind, is let through unchanged.
bool checkParams(const std::string& method, const nlohmann::json& params,
                 std::string& error);

} // namespace rtipc_validation
