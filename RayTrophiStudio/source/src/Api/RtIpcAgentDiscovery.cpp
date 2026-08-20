/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtIpcAgentDiscovery.cpp
 * Date:          August 2026
 * License:       MIT
 * =========================================================================
 * The agent.* endpoints: how an automated caller learns what this build can
 * do, and how it measures what it just did.
 *
 * ★ Everything here reports MEASURED numbers or says it could not measure.
 * The first cut hard-coded a method total and divided by it, so `coverage`
 * came out at 1.01 while 299 of 300 descriptors were empty. A discovery layer
 * that overstates itself is worse than none: an agent trusts it.
 * =========================================================================
 */

#include "RtIpcAgentDiscovery.h"
#include "RtIpcMethodRegistry.h"
#include "RtIpcWorkflowRecipes.h"
#include "api/RtApi.h"
#include "scene_ui.h"

#include <algorithm>
#include <stdexcept>

using json = nlohmann::json;

namespace {

std::string requireString(const json& params, const char* key) {
    if (!params.contains(key)) throw std::runtime_error("missing param: " + std::string(key));
    if (!params[key].is_string()) throw std::runtime_error(std::string(key) + " must be a string");
    return params[key].get<std::string>();
}

// Hand-written one-liners for the domains an agent meets first. Anything not
// listed falls back to a neutral phrase rather than an invented one.
const char* domainSummary(const std::string& domain) {
    if (domain == "scene")       return "Objects in the scene: create, delete, transform, inspect";
    if (domain == "select")      return "Selection state";
    if (domain == "camera")      return "Camera placement and lens";
    if (domain == "lights")      return "Scene lights";
    if (domain == "world")       return "Sky, sun and background environment";
    if (domain == "material")    return "Material creation, assignment and parameters";
    if (domain == "nodes")       return "Node graphs for materials, geometry and terrain";
    if (domain == "modifiers")   return "Object modifier stacks";
    if (domain == "timeline")    return "Playhead; stepping it advances the solvers";
    if (domain == "render")      return "Final renders, sequences and frame measurement";
    if (domain == "viewport")    return "Viewport capture, convergence and status";
    if (domain == "post")        return "Post-processing: exposure, tone mapping, grade";
    if (domain == "fluid")       return "APIC liquid domains, seeding and substances";
    if (domain == "gas")         return "Gas domains: smoke, fire, buoyancy, blast";
    if (domain == "flow_source") return "Emitters that feed fluid and gas domains";
    if (domain == "msf")         return "Material substance library and live material state fields";
    if (domain == "physics")     return "Rigid bodies, fracture and destruction";
    if (domain == "particle")    return "Particle systems, emitters and solver settings";
    if (domain == "forcefield")  return "Spatial force fields: wind, vortex, drag, noise";
    if (domain == "terrain")     return "Terrain heightfields, erosion and rivers";
    if (domain == "scatter")     return "Instancing objects over surfaces";
    if (domain == "hair")        return "Hair and fur grooms";
    if (domain == "paint")       return "Texture painting layers and channels";
    if (domain == "sculpt")      return "Mesh sculpting and masks";
    if (domain == "anim")        return "Character animation, clips and keyframes";
    if (domain == "sim_graph")   return "Simulation node graphs, scoped to an owner";
    if (domain == "sim_cache")   return "Simulation bake cache";
    if (domain == "debris")      return "Ash and debris particles";
    if (domain == "templates")   return "Scene templates and the Template Hub";
    if (domain == "project")     return "Project open, save and current path";
    if (domain == "editor")      return "Which editors are open and what they are showing";
    if (domain == "addons")      return "Addon modules";
    if (domain == "script")      return "Running Python scripts in-process";
    if (domain == "agent")       return "Discovery, measurement and chat for automated callers";
    if (domain == "ipc")         return "IPC tokens, sessions and audit log (admin)";
    return "";
}

json serializeParams(const MethodDescriptor* desc) {
    json params_obj = json::object();
    for (int i = 0; i < desc->param_count && desc->params; ++i) {
        const auto& p = desc->params[i];
        json entry = {
            {"type", p.type ? p.type : "any"},
            {"required", p.required}
        };
        if (p.description && p.description[0] != '\0') entry["description"] = p.description;
        if (p.default_value) entry["default"] = p.default_value;
        if (p.enum_values) entry["enum"] = p.enum_values;
        params_obj[p.name] = entry;
    }
    return params_obj;
}

json splitPipes(const char* text) {
    json out = json::array();
    if (!text) return out;
    std::string current;
    for (const char* c = text; *c; ++c) {
        if (*c == '|') {
            if (!current.empty()) out.push_back(current);
            current.clear();
        } else {
            current += *c;
        }
    }
    if (!current.empty()) out.push_back(current);
    return out;
}

json serializeMethodDescriptor(const MethodDescriptor* desc) {
    json j = {
        {"method", desc->name ? desc->name : ""},
        {"domain", desc->domain ? desc->domain : ""},
        {"summary", desc->summary ? desc->summary : ""},
        {"access", desc->access ? desc->access : ""},
        {"capability", desc->capability ? desc->capability : ""},
        {"undoable", desc->undoable},
        {"returns", desc->return_type ? desc->return_type : "any"},
        {"params", serializeParams(desc)},
        {"documented", desc->documented}
    };
    if (desc->notes && desc->notes[0] != '\0') j["notes"] = desc->notes;
    if (desc->related && desc->related[0] != '\0') j["related"] = splitPipes(desc->related);
    if (desc->tags && desc->tags[0] != '\0') j["tags"] = splitPipes(desc->tags);
    // ★★ Sequencing. Emitted only when written, so a caller can tell "nothing
    // has to happen first" (field absent, nobody recorded an order) from an
    // empty list, which would read as a measured "no prerequisites".
    if (desc->prerequisites && desc->prerequisites[0] != '\0')
        j["requires"] = splitPipes(desc->prerequisites);
    if (desc->next_steps && desc->next_steps[0] != '\0')
        j["next"] = splitPipes(desc->next_steps);
    if (desc->verify_with && desc->verify_with[0] != '\0')
        j["verify_with"] = splitPipes(desc->verify_with);
    if (desc->invalidates && desc->invalidates[0] != '\0')
        j["invalidates"] = splitPipes(desc->invalidates);
    if (!desc->documented) {
        j["documentation_note"] =
            "Parameters are extracted from the dispatch code and are exact; "
            "no summary has been written for this method yet.";
    }
    return j;
}

json serializeSteps(const std::vector<WorkflowStep>& steps) {
    json arr = json::array();
    for (const auto& s : steps) {
        arr.push_back({
            {"action", s.action ? s.action : ""},
            {"purpose", s.purpose ? s.purpose : ""},
            {"requires", s.requires_state ? s.requires_state : ""},
            {"verify", s.verify ? s.verify : ""},
            {"on_failure", s.on_failure ? s.on_failure : ""}
        });
    }
    return arr;
}

json serializeRecipe(const WorkflowRecipe* recipe) {
    return json{
        {"workflow", recipe->id},
        {"title", recipe->title},
        {"description", recipe->description},
        {"steps", serializeSteps(recipe->steps)},
        {"key_methods", recipe->key_methods}
    };
}

// One example per recipe, built from the recipe's own key methods and the
// registry's parameter schema, so an example cannot name a method or a
// parameter that does not exist.
json exampleCallsForRecipe(const WorkflowRecipe* recipe) {
    const auto& registry = MethodRegistry::instance();
    json calls = json::array();
    for (const std::string& method : recipe->key_methods) {
        const MethodDescriptor* desc = registry.find(method);
        if (!desc) continue;
        json params = json::object();
        for (int i = 0; i < desc->param_count && desc->params; ++i) {
            const auto& p = desc->params[i];
            if (!p.required) continue;
            if (p.default_value) params[p.name] = p.default_value;
            else params[p.name] = std::string("<") + (p.type ? p.type : "value") + ">";
        }
        calls.push_back(json{{"method", method}, {"params", params}});
    }
    return calls;
}

} // namespace

bool dispatchAgentMethod(const std::string& method,
                         const nlohmann::json& params,
                         const RtIpcTemplateEnqueue& enqueue,
                         nlohmann::json& out_result) {

    if (method.rfind("agent.", 0) != 0) {
        return false;
    }

    try {
        if (method == "agent.discover") {
            out_result = enqueue([](auto&) {
                const auto& registry = MethodRegistry::instance();

                json domains = json::array();
                for (const auto& d : registry.domains()) {
                    json entry = {
                        {"name", d.name},
                        {"method_count", d.method_count}
                    };
                    const char* summary = domainSummary(d.name);
                    if (summary[0] != '\0') entry["summary"] = summary;
                    domains.push_back(entry);
                }

                const int registered = registry.registeredCount();
                const int documented = registry.documentedCount();
                const float documented_coverage =
                    registered > 0 ? static_cast<float>(documented) / registered : 0.0f;

                return json{
                    {"app", "RayTrophi Studio"},
                    {"version", std::to_string(rtapi::version().major) + "." +
                                std::to_string(rtapi::version().minor) + "." +
                                std::to_string(rtapi::version().patch)},
                    {"discovery_version", 2},
                    {"protocol", "json-rpc-like (request: method/params/id, response: result or error)"},
                    {"description",
                     "Path tracing renderer with APIC fluid and gas simulation, rigid-body "
                     "destruction, material substance thermochemistry, terrain, hair, "
                     "particles and node graphs."},
                    {"domains", domains},
                    {"agent_methods", {
                        "agent.discover", "agent.describe", "agent.search_capabilities",
                        "agent.get_examples", "agent.get_state_summary",
                        "agent.list_methods", "agent.roles",
                        "agent.chat_send", "agent.chat_poll"
                    }},
                    {"roles", {"manager", "controller", "worker"}},
                    {"registered_methods", registered},
                    {"documented_methods", documented},
                    {"documented_coverage", documented_coverage},
                    // ★ No dispatch total is quoted here. The descriptor table is
                    // GENERATED from the dispatch sources and the capability audit
                    // fails when they differ, so registered_methods IS the dispatch
                    // count. A separate hand-kept total could only ever be a lie.
                    {"registry_source",
                     "generated from dispatch by scripts/gen_ipc_descriptors.py; "
                     "scripts/audit_ipc_capabilities.py fails the build if they drift"},
                    {"getting_started", {
                        "agent.search_capabilities with the goal in plain words",
                        "agent.describe on each method the recipe names",
                        "call the methods",
                        "agent.get_state_summary or render.probe to verify"
                    }}
                };
            });
            return true;
        }

        if (method == "agent.roles") {
            out_result = enqueue([](auto&) {
                return json{
                    {"roles", {
                        {
                            {"id", "manager"},
                            {"name", "Scene Manager"},
                            {"description", "Decomposes a goal into workflow steps and verifies the "
                                            "result by reading state summaries and viewport probes."},
                            {"suggested_methods", {"agent.discover", "agent.search_capabilities",
                                                   "agent.get_state_summary", "render.probe"}},
                            {"delegates_to", {"controller"}}
                        },
                        {
                            {"id", "controller"},
                            {"name", "Workflow Controller"},
                            {"description", "Plans and sequences a multi-step workflow, computes "
                                            "parameters and recovers from errors."},
                            {"suggested_methods", {"agent.describe", "agent.list_methods", "batch"}},
                            {"delegates_to", {"worker"}},
                            {"receives_from", {"manager"}}
                        },
                        {
                            {"id", "worker"},
                            {"name", "API Worker"},
                            {"description", "Executes individual calls and reports raw results."},
                            {"suggested_methods", {"scene.*", "material.*", "fluid.*", "physics.*"}},
                            {"receives_from", {"controller"}}
                        }
                    }},
                    {"delegation_chain", "manager -> controller -> worker"},
                    {"enforced", false},
                    {"notes", "Roles are informational. What a connection may actually do is decided "
                              "by the capability mask on its IPC token, not by the role it claims. "
                              "Local pipe connections are trusted by the pipe ACL."}
                };
            });
            return true;
        }

        if (method == "agent.list_methods") {
            const std::string domain = params.value("domain", "");
            out_result = enqueue([domain](auto&) {
                const auto& registry = MethodRegistry::instance();
                const std::vector<const MethodDescriptor*> list =
                    domain.empty() ? registry.all() : registry.byDomain(domain);

                json arr = json::array();
                for (const auto* desc : list) {
                    arr.push_back({
                        {"method", desc->name ? desc->name : ""},
                        {"summary", desc->summary ? desc->summary : ""},
                        {"access", desc->access ? desc->access : ""},
                        {"param_count", desc->param_count},
                        {"documented", desc->documented}
                    });
                }
                if (!domain.empty() && arr.empty())
                    return json{{"__error", "unknown domain: " + domain +
                                            " (call agent.discover for the domain list)"}};
                return json{
                    {"domain", domain.empty() ? "all" : domain},
                    {"count", arr.size()},
                    {"methods", arr}
                };
            });
            return true;
        }

        if (method == "agent.describe") {
            const std::string target = requireString(params, "method");
            out_result = enqueue([target](auto&) {
                const auto& registry = MethodRegistry::instance();
                const MethodDescriptor* desc = registry.find(target);
                if (desc) return serializeMethodDescriptor(desc);

                // A near miss is the common case: the caller guessed a name.
                // Saying which real names are close is more useful than "no".
                json suggestions = json::array();
                for (const auto& hit : registry.search(target)) {
                    if (suggestions.size() >= 5) break;
                    suggestions.push_back(hit.desc->name);
                }
                json error = json{{"__error", "no method named '" + target + "'"}};
                if (!suggestions.empty()) error["did_you_mean"] = suggestions;
                return error;
            });
            return true;
        }

        if (method == "agent.search_capabilities") {
            const std::string query = requireString(params, "query");
            const int limit = std::clamp(params.value("limit", 10), 1, 50);
            out_result = enqueue([query, limit](auto&) {
                json workflows = json::array();
                for (const auto* recipe : WorkflowRecipeRegistry::instance().search(query))
                    workflows.push_back(serializeRecipe(recipe));

                json methods = json::array();
                for (const auto& hit : MethodRegistry::instance().search(query)) {
                    if (static_cast<int>(methods.size()) >= limit) break;
                    methods.push_back({
                        {"method", hit.desc->name},
                        {"summary", hit.desc->summary ? hit.desc->summary : ""},
                        {"access", hit.desc->access ? hit.desc->access : ""},
                        {"score", hit.score}
                    });
                }

                json result = {
                    {"query", query},
                    {"relevant_workflows", workflows},
                    {"relevant_methods", methods}
                };
                if (workflows.empty() && methods.empty())
                    result["hint"] = "Nothing matched. Try a single concrete noun or verb "
                                     "(\"fire\", \"pour\", \"shatter\", \"terrain\"), or call "
                                     "agent.list_methods for a domain.";
                else if (!workflows.empty())
                    result["hint"] = "Follow a workflow's steps in order; each key_method has a "
                                     "full schema via agent.describe.";
                return result;
            });
            return true;
        }

        if (method == "agent.get_examples") {
            const std::string target = params.value("method", "");
            const std::string workflow = params.value("workflow", "");
            if (target.empty() && workflow.empty())
                throw std::runtime_error("pass either 'method' or 'workflow'");
            out_result = enqueue([target, workflow](auto&) {
                auto& recipes = WorkflowRecipeRegistry::instance();
                if (!workflow.empty()) {
                    const WorkflowRecipe* recipe = recipes.find(workflow);
                    if (!recipe)
                        return json{{"__error", "no workflow named '" + workflow +
                                                "' (use agent.search_capabilities)"}};
                    json example = serializeRecipe(recipe);
                    example["calls"] = exampleCallsForRecipe(recipe);
                    example["note"] = "Required parameters are shown with their type in angle "
                                      "brackets; call agent.describe for the optional ones.";
                    return example;
                }

                const MethodDescriptor* desc = MethodRegistry::instance().find(target);
                if (!desc)
                    return json{{"__error", "no method named '" + target + "'"}};

                json call = {{"method", target}, {"params", json::object()}};
                for (int i = 0; i < desc->param_count && desc->params; ++i) {
                    const auto& p = desc->params[i];
                    if (!p.required && !p.default_value) continue;
                    if (p.default_value) call["params"][p.name] = p.default_value;
                    else call["params"][p.name] = std::string("<") +
                                                  (p.type ? p.type : "value") + ">";
                }

                // Every recipe that uses this method is a working context for it.
                json contexts = json::array();
                for (const auto& recipe : recipes.all()) {
                    bool uses = false;
                    for (const char* key : recipe.key_methods)
                        if (target == key) { uses = true; break; }
                    if (!uses) continue;
                    contexts.push_back(json{{"workflow", recipe.id},
                                            {"title", recipe.title},
                                            {"steps", serializeSteps(recipe.steps)}});
                }
                json out = {{"method", target}, {"example_call", call}};
                if (!contexts.empty()) out["used_by_workflows"] = contexts;
                return out;
            });
            return true;
        }

        if (method == "agent.get_state_summary") {
            const bool include_probe = params.value("include_probe", true);
            out_result = enqueue([include_probe](auto&) {
                json scene = json::object();
                const std::vector<std::string> objects = rtapi::listObjects();
                scene["object_count"] = objects.size();
                json names = json::array();
                for (size_t i = 0; i < objects.size() && i < 40; ++i) names.push_back(objects[i]);
                scene["objects"] = names;
                if (objects.size() > 40) scene["objects_truncated"] = objects.size() - 40;

                json lights = json::array();
                for (const auto& light : rtapi::listLights())
                    lights.push_back({{"index", light.index}, {"name", light.name},
                                      {"type", light.type}, {"intensity", light.intensity}});

                json camera = json::object();
                rtapi::CameraState cam;
                if (rtapi::getCamera(cam).ok) {
                    camera = {{"position", {cam.position.x, cam.position.y, cam.position.z}},
                              {"target", {cam.target.x, cam.target.y, cam.target.z}},
                              {"fov", cam.fov}};
                }

                json domains = json::array();
                std::vector<rtapi::FluidDomainInfo> fluid_domains;
                if (rtapi::listFluidDomains(fluid_domains).ok) {
                    for (const auto& d : fluid_domains) {
                        json entry = {{"name", d.name}, {"type", d.type},
                                      {"voxel_size", d.voxel_size},
                                      {"render_mode", d.render_mode},
                                      {"backend", d.backend}};
                        // ★ particle_count is a measurement only when live_state
                        // is set; otherwise the number is a fallback mirror that
                        // is always empty. Reporting it unconditionally is how a
                        // burn script concluded there was nothing left to burn.
                        if (d.live_state) entry["particle_count"] = d.particle_count;
                        else entry["particle_count"] = "not measured (no live solver state)";
                        domains.push_back(entry);
                    }
                }

                const rtapi::RenderJobInfo render = rtapi::renderStatus();
                json render_json = {{"progress", render.progress},
                                    {"current_samples", render.current_samples},
                                    {"target_samples", render.target_samples}};
                if (!render.error.empty()) render_json["error"] = render.error;

                const rtapi::ViewportStatusInfo vs = rtapi::viewportStatus();
                json viewport = {{"available", vs.available},
                                 {"backend", vs.backend},
                                 {"samples", vs.samples},
                                 {"rendering_active", vs.rendering_active},
                                 {"capture_enabled", vs.capture_enabled},
                                 {"frame_available", vs.frame_available}};

                if (include_probe) {
                    const rtapi::ViewportProbeRegion region = {0, 0, 0, 0};
                    const rtapi::ViewportProbeInfo probe =
                        rtapi::probeViewportFrame(region, 0.001f);
                    if (probe.available) {
                        viewport["probe"] = {{"mean_luminance", probe.mean_luminance},
                                             {"max_luminance", probe.max_luminance},
                                             {"black_fraction", probe.black_fraction},
                                             {"nan_fraction", probe.nan_fraction},
                                             {"pixels", probe.pixels}};
                    } else {
                        // ★ NOT zeros. A zero mean luminance reads as "the scene
                        // is dark"; the truth is "nothing was measured". Say so.
                        viewport["probe"] = "unavailable - enable frame capture "
                                            "with viewport.capture {enabled:true} and render "
                                            "a frame first";
                    }
                }

                return json{
                    {"scene", scene},
                    {"lights", lights},
                    {"camera", camera},
                    {"simulation_domains", domains},
                    {"timeline", {{"frame", rtapi::currentFrame()}}},
                    {"render", render_json},
                    {"viewport", viewport},
                    {"note", "Compact snapshot. Use the domain .list/.get methods for detail."}
                };
            });
            return true;
        }

        if (method == "agent.chat_send") {
            const std::string sender = requireString(params, "sender");
            std::string content = requireString(params, "content");
            const std::string kind = params.value("type", std::string("reply"));
            if (params.contains("payload"))
                content += "\n[payload] " + params["payload"].dump();
            if (params.contains("image_base64"))
                content += "\n[image attached]";

            rtui::AgentMessageType message_type = rtui::AgentMessageType::AgentReply;
            if (kind == "activity") message_type = rtui::AgentMessageType::AgentActivity;
            else if (kind == "thought") message_type = rtui::AgentMessageType::AgentThought;
            else if (kind == "error") message_type = rtui::AgentMessageType::SystemEvent;
            else if (kind != "reply")
                throw std::runtime_error("type must be reply|activity|thought|error");

            out_result = enqueue([sender, content, message_type](UIContext& ctx) {
                if (!ctx.scene_ui_ptr) return json{{"__error", "no UI bound"}};
                ctx.scene_ui_ptr->agentChatUI.pushMessage(message_type, sender, "User", content);
                return json{{"delivered", true}};
            });
            return true;
        }

        if (method == "agent.send_prompt") {
            const std::string target = requireString(params, "target");
            const std::string content = requireString(params, "content");
            const std::string sender = params.value("sender", std::string("Agent"));
            out_result = enqueue([sender, target, content](UIContext& ctx) {
                if (!ctx.scene_ui_ptr) return json{{"__error", "no UI bound"}};
                ctx.scene_ui_ptr->agentChatUI.queuePrompt(sender, target, content);
                // ★ "queued", not "delivered". Nothing has run yet - the target
                // agent has to poll. Reporting delivery here would let a
                // delegating agent claim work was done that nobody has started.
                return json{{"queued", true}, {"target", target}};
            });
            return true;
        }

        if (method == "agent.chat_poll") {
            std::string agent_id = params.value("agent_id", "");
            out_result = enqueue([agent_id](UIContext& ctx) {
                json prompts = json::array();
                if (ctx.scene_ui_ptr) {
                    ctx.scene_ui_ptr->agentChatUI.markPoll();
                    rtui::AgentChatPanel::QueuedPrompt p;
                    while (ctx.scene_ui_ptr->agentChatUI.popUserPrompt(agent_id, p))
                        prompts.push_back({{"target", p.target}, {"content", p.content}});
                }
                return json{{"prompts", prompts}};
            });
            return true;
        }

        out_result = json{{"__error", "unknown agent method: " + method}};
        return true;

    } catch (const std::exception& e) {
        out_result = json{{"__error", std::string(e.what())}};
        return true;
    }
}
