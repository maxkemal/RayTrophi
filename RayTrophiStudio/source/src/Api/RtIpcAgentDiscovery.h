/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtIpcAgentDiscovery.h
 * Date:          August 2026
 * License:       MIT
 * =========================================================================
 */

#pragma once

#include "json.hpp"
#include <string>

struct UIContext;
#include "Api/RtIpcTemplates.h" // For RtIpcTemplateEnqueue

// Dispatches an agent.* method. Returns true if the method was handled
// (even if it resulted in an error), false if the method is not an agent method.
bool dispatchAgentMethod(const std::string& method,
                         const nlohmann::json& params,
                         const RtIpcTemplateEnqueue& enqueue,
                         nlohmann::json& out_result);
