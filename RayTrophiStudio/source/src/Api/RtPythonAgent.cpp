/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtPythonAgent.cpp
 * Date:          August 2026
 * License:       MIT
 * =========================================================================
 * rt.agent — the discovery and measurement surface for in-process scripts.
 *
 * ★ These bindings call the SAME dispatchAgentMethod() the IPC layer calls,
 * with the handler run inline on the calling (main) thread. Reimplementing the
 * handlers here would create two answers to "what can this build do", and the
 * one nobody is looking at would be the one that goes stale. A script test can
 * therefore assert on exactly what an external agent sees.
 * =========================================================================
 */

#include "RtPythonAgent.h"

#include "RtIpcAgentDiscovery.h"
#include "Api/RtApi.h"

#include <pybind11/stl.h>

#include <stdexcept>

namespace py = pybind11;
using json = nlohmann::json;

namespace {

py::object toPython(const json& value) {
    switch (value.type()) {
        case json::value_t::null:            return py::none();
        case json::value_t::boolean:         return py::bool_(value.get<bool>());
        case json::value_t::number_integer:
        case json::value_t::number_unsigned: return py::int_(value.get<long long>());
        case json::value_t::number_float:    return py::float_(value.get<double>());
        case json::value_t::string:          return py::str(value.get<std::string>());
        case json::value_t::array: {
            py::list out;
            for (const auto& item : value) out.append(toPython(item));
            return out;
        }
        case json::value_t::object: {
            py::dict out;
            for (auto it = value.begin(); it != value.end(); ++it)
                out[py::str(it.key())] = toPython(it.value());
            return out;
        }
        default:                             return py::none();
    }
}

// Runs an agent.* handler inline. Scripts execute on the main thread, which is
// the thread the IPC path marshals its own work onto, so there is nothing to
// queue.
py::object callAgent(const std::string& method, const json& params) {
    UIContext* ctx = rtapi::boundContext();
    if (!ctx) throw std::runtime_error("rtapi is not bound yet");

    json result;
    const bool handled = dispatchAgentMethod(
        method, params,
        [ctx](RtIpcTemplateQuery query) { return query(*ctx); },
        result);
    if (!handled) throw std::runtime_error("not an agent method: " + method);
    if (result.is_object() && result.contains("__error"))
        throw std::runtime_error(result["__error"].get<std::string>());
    return toPython(result);
}

}  // namespace

namespace rtpy {

void registerAgentBindings(py::module_& root) {
    py::module_ agent = root.def_submodule(
        "agent",
        "Self-description and measurement: what this build can do, and what "
        "state it is in. Same answers an external agent gets over IPC.");

    agent.def("discover", [] { return callAgent("agent.discover", json::object()); },
              "Identity, capability domains, method counts and documented coverage.");

    agent.def("list_methods", [](const std::string& domain) {
        json params = json::object();
        if (!domain.empty()) params["domain"] = domain;
        return callAgent("agent.list_methods", params);
    }, py::arg("domain") = std::string(),
       "Every registered method, or one domain's worth.");

    agent.def("describe", [](const std::string& method) {
        return callAgent("agent.describe", json{{"method", method}});
    }, py::arg("method"),
       "Exact parameter schema, capability and related methods. The parameters "
       "are extracted from the dispatch code, so they cannot drift; `documented` "
       "reports whether prose was written for it.");

    agent.def("search", [](const std::string& query, int limit) {
        return callAgent("agent.search_capabilities",
                         json{{"query", query}, {"limit", limit}});
    }, py::arg("query"), py::arg("limit") = 10,
       "Workflow recipes and methods for a goal in plain words.");

    agent.def("examples", [](const std::string& method, const std::string& workflow) {
        json params = json::object();
        if (!method.empty()) params["method"] = method;
        if (!workflow.empty()) params["workflow"] = workflow;
        return callAgent("agent.get_examples", params);
    }, py::arg("method") = std::string(), py::arg("workflow") = std::string(),
       "Runnable call sequences for a method or a workflow recipe.");

    agent.def("state_summary", [](bool include_probe) {
        return callAgent("agent.get_state_summary",
                         json{{"include_probe", include_probe}});
    }, py::arg("include_probe") = false,
       "Compact snapshot of scene, lights, camera, domains, timeline, render "
       "and viewport. With include_probe the last viewport frame is measured; "
       "when no frame was captured the probe says so instead of reporting zeros.");

    agent.def("roles", [] { return callAgent("agent.roles", json::object()); },
              "The manager/controller/worker role descriptions. Informational.");

    agent.def("chat_send", [](const std::string& content, const std::string& sender,
                              const std::string& type) {
        return callAgent("agent.chat_send",
                         json{{"sender", sender}, {"content", content}, {"type", type}});
    }, py::arg("content"), py::arg("sender") = std::string("Script"),
       py::arg("type") = std::string("reply"),
       "Post into the Agent Chat panel: reply|activity|thought|error.");

    agent.def("send_prompt", [](const std::string& target, const std::string& content,
                                const std::string& sender) {
        return callAgent("agent.send_prompt",
                         json{{"target", target}, {"content", content}, {"sender", sender}});
    }, py::arg("target"), py::arg("content"), py::arg("sender") = std::string("Script"),
       "Queue a task for another agent ('all' broadcasts). Returns queued:true, "
       "not delivered - the target has to poll before anything happens.");

    agent.def("chat_poll", [] { return callAgent("agent.chat_poll", json::object()); },
              "Take the prompts queued in the Agent Chat panel. Also counts as a "
              "heartbeat, so the panel will report an agent as connected.");
}

}  // namespace rtpy
