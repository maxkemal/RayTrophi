/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtIpcMethodDescriptors.cpp
 * Date:          August 2026
 * License:       MIT
 * =========================================================================
 * GENERATED FILE - do not edit by hand.
 *
 *   python scripts/gen_ipc_descriptors.py
 *
 * Parameters, types, requiredness, defaults and the security capability are
 * read out of the dispatch sources, so that half cannot drift from the code.
 * Summaries, notes, units, tags and related-method links come from
 * scripts/ipc_descriptor_overlay.json - edit THAT file, then regenerate.
 *
 * A method with no overlay entry is emitted with documented = false. That is
 * deliberate: agent.discover reports documented_coverage from this flag, so an
 * undocumented method shows up as a measured gap instead of as an empty schema
 * that looks complete.
 * =========================================================================
 */

#include "RtIpcMethodRegistry.h"

namespace {

static const MethodParam params_addons_disable[] = {
    {"module_name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_addons_disable = {
    "addons.disable", "addons",
    "Disable an addon module",
    nullptr,
    "write", "Addons", false, "any",
    "addons|disable|plugin|extension",
    nullptr,
    params_addons_disable, 1,
    true
};
static const MethodRegistration reg_addons_disable(desc_addons_disable);

static const MethodParam params_addons_enable[] = {
    {"module_name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_addons_enable = {
    "addons.enable", "addons",
    "Enable an addon module",
    nullptr,
    "write", "Addons", false, "any",
    "addons|enable|plugin|extension",
    nullptr,
    params_addons_enable, 1,
    true
};
static const MethodRegistration reg_addons_enable(desc_addons_enable);

static const MethodDescriptor desc_addons_list = {
    "addons.list", "addons",
    "List the installed addons with their load state",
    nullptr,
    "read", "Read", false, "any",
    "addons|list|plugin|extension",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_addons_list(desc_addons_list);

static const MethodParam params_addons_reload[] = {
    {"module_name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_addons_reload = {
    "addons.reload", "addons",
    "Reload an addon module from disk",
    nullptr,
    "write", "Addons", false, "any",
    "addons|reload|plugin|extension|develop",
    nullptr,
    params_addons_reload, 1,
    true
};
static const MethodRegistration reg_addons_reload(desc_addons_reload);

static const MethodDescriptor desc_agent_chat_poll = {
    "agent.chat_poll", "agent",
    "Take the user prompts queued in the Agent Chat panel and mark the agent as alive",
    "Polling is also the panel's heartbeat: it shows the agent as online for a few seconds after each call. An agent that blocks on a long model turn should keep polling from a second thread, or the panel reports it offline.",
    "read", "Read", false, "QueuedPrompt[]",
    "agent|chat|poll|prompt|heartbeat",
    "agent.chat_send",
    nullptr, 0,
    true
};
static const MethodRegistration reg_agent_chat_poll(desc_agent_chat_poll);

static const MethodParam params_agent_chat_send[] = {
    {"sender", "string", true, "Display name shown in the panel, e.g. 'RayTrophi Agent'", nullptr, nullptr},
    {"content", "string", true, "Message text", nullptr, nullptr},
    {"type", "string", false, "Message kind", "reply", "reply|activity|thought|error"},
    {"payload", "any", false, "Optional structured data appended to the message", nullptr, nullptr},
    {"image_base64", "any", false, "Optional base64 image; the panel notes the attachment", nullptr, nullptr},
};
static const MethodDescriptor desc_agent_chat_send = {
    "agent.chat_send", "agent",
    "Post a message into the Studio Agent Chat panel",
    "This WRITES to the user interface, so it needs the AgentChat capability - it is the one agent.* method that is not read-only. Use type='activity' for per-call progress, 'thought' for reasoning, 'error' for failures; the panel colours and filters them separately.",
    "write", "AgentChat", false, "bool",
    "agent|chat|send|message|ui|report|progress",
    "agent.chat_poll",
    params_agent_chat_send, 5,
    true
};
static const MethodRegistration reg_agent_chat_send(desc_agent_chat_send);

static const MethodParam params_agent_describe[] = {
    {"method", "string", true, "Exact method name, e.g. 'fluid.create_domain'", nullptr, nullptr},
};
static const MethodDescriptor desc_agent_describe = {
    "agent.describe", "agent",
    "Return the full parameter schema, capability and related methods for one IPC method",
    "Parameters come from the dispatch code itself, so they are exact. `documented: false` means the schema is real but nobody has written the prose yet - trust the parameters, be careful with intent.",
    "read", "Read", false, "MethodDescriptor",
    "agent|describe|schema|introspection|parameters",
    "agent.list_methods|agent.get_examples",
    params_agent_describe, 1,
    true
};
static const MethodRegistration reg_agent_describe(desc_agent_describe);

static const MethodDescriptor desc_agent_discover = {
    "agent.discover", "agent",
    "Identify the application and list every capability domain, agent role and coverage metric",
    "First call for a new agent session. registered_coverage is registered/dispatched methods; documented_coverage is the share that carries a hand-written summary. A documented_coverage below 1.0 means agent.describe will answer some methods with parameters but no explanation.",
    "read", "Read", false, "DiscoveryInfo",
    "agent|discover|bootstrap|handshake|capabilities|introspection",
    "agent.list_methods|agent.describe|agent.search_capabilities",
    nullptr, 0,
    true
};
static const MethodRegistration reg_agent_discover(desc_agent_discover);

static const MethodParam params_agent_get_examples[] = {
    {"method", "string", false, "Method to show examples for, e.g. 'fluid.create_domain'", "", nullptr},
    {"workflow", "string", false, "Recipe id from agent.search_capabilities, e.g. 'combustion_setup'", "", nullptr},
};
static const MethodDescriptor desc_agent_get_examples = {
    "agent.get_examples", "agent",
    "Return runnable call sequences for a method or a named workflow recipe",
    nullptr,
    "read", "Read", false, "ExampleSet",
    "agent|get|examples|example|sample|recipe",
    "agent.search_capabilities|agent.describe",
    params_agent_get_examples, 2,
    true
};
static const MethodRegistration reg_agent_get_examples(desc_agent_get_examples);

static const MethodParam params_agent_get_state_summary[] = {
    {"include_probe", "bool", false, "Sample the last viewport frame for luminance and black/NaN fractions. Costs a full-frame scan.", "true", nullptr},
};
static const MethodDescriptor desc_agent_get_state_summary = {
    "agent.get_state_summary", "agent",
    "Compact snapshot of scene, lights, camera, timeline and viewport measurement state",
    "The viewport block is a MEASUREMENT and is only present when frame capture is on and a frame exists; when it is absent that means 'not measured', never 'measured zero'. Turn capture on with viewport.capture before relying on it.",
    "read", "Read", false, "StateSummary",
    "agent|get|state|summary|context|snapshot|scene|verify",
    "viewport.status|render.probe|scene.list_objects",
    params_agent_get_state_summary, 1,
    true
};
static const MethodRegistration reg_agent_get_state_summary(desc_agent_get_state_summary);

static const MethodParam params_agent_list_methods[] = {
    {"domain", "string", false, "Domain to restrict the listing to, e.g. 'fluid'. Omit for every method.", "", nullptr},
};
static const MethodDescriptor desc_agent_list_methods = {
    "agent.list_methods", "agent",
    "List every registered IPC method, optionally filtered to one domain",
    "The full list is around 30 KB. Cache it once per session instead of calling it per step.",
    "read", "Read", false, "MethodSummary[]",
    "agent|list|methods|catalogue|introspection",
    "agent.describe|agent.search_capabilities",
    params_agent_list_methods, 1,
    true
};
static const MethodRegistration reg_agent_list_methods(desc_agent_list_methods);

static const MethodDescriptor desc_agent_roles = {
    "agent.roles", "agent",
    "Describe the manager/controller/worker agent roles and their delegation chain",
    "Informational only. Access is decided by the security capability on the connection token, not by the role an agent claims.",
    "read", "Read", false, "RoleSet",
    "agent|roles|role|hierarchy|delegation|multi-agent",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_agent_roles(desc_agent_roles);

static const MethodParam params_agent_search_capabilities[] = {
    {"query", "string", true, "Plain language goal, e.g. 'make a wooden object burn'", nullptr, nullptr},
    {"limit", "int", false, "", "10", nullptr},
};
static const MethodDescriptor desc_agent_search_capabilities = {
    "agent.search_capabilities", "agent",
    "Search workflow recipes and method metadata for a plain-language goal",
    "Recipes are the valuable half: each one is an ordered list of calls that is known to work end to end. Prefer following a recipe over assembling calls from the method hits.",
    "read", "Read", false, "SearchResult",
    "agent|search|capabilities|howto|recipe|workflow|goal",
    "agent.get_examples|agent.describe",
    params_agent_search_capabilities, 2,
    true
};
static const MethodRegistration reg_agent_search_capabilities(desc_agent_search_capabilities);

static const MethodParam params_anim_character[] = {
    {"character", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_anim_character = {
    "anim.character", "anim",
    "Return one character's animation state",
    nullptr,
    "read", "Read", false, "any",
    "anim|character|animation",
    nullptr,
    params_anim_character, 1,
    true
};
static const MethodRegistration reg_anim_character(desc_anim_character);

static const MethodDescriptor desc_anim_characters = {
    "anim.characters", "anim",
    "List the animated characters",
    nullptr,
    "read", "Read", false, "any",
    "anim|characters|animation|inventory",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_anim_characters(desc_anim_characters);

static const MethodParam params_anim_clips[] = {
    {"character", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_anim_clips = {
    "anim.clips", "anim",
    "List a character's animation clips with frame ranges and loop flags",
    nullptr,
    "read", "Read", false, "any",
    "anim|clips|animation|clip",
    nullptr,
    params_anim_clips, 1,
    true
};
static const MethodRegistration reg_anim_clips(desc_anim_clips);

static const MethodParam params_anim_graph_status[] = {
    {"character", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_anim_graph_status = {
    "anim.graph_status", "anim",
    "Report a character's animation graph state",
    nullptr,
    "read", "Read", false, "any",
    "anim|graph|status|animation|statemachine",
    nullptr,
    params_anim_graph_status, 1,
    true
};
static const MethodRegistration reg_anim_graph_status(desc_anim_graph_status);

static const MethodParam params_anim_insert_key[] = {
    {"channel", "string", true, "Animated channel", nullptr, "location|rotation|scale"},
    {"frame", "int", true, "", nullptr, nullptr},
    {"object_name", "string", true, "", nullptr, nullptr},
    {"value", "vec3", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_anim_insert_key = {
    "anim.insert_key", "anim",
    "Insert a keyframe on an object channel at a frame",
    nullptr,
    "write", "SceneWrite", false, "any",
    "anim|insert|key|animation|keyframe|channel",
    nullptr,
    params_anim_insert_key, 4,
    true
};
static const MethodRegistration reg_anim_insert_key(desc_anim_insert_key);

static const MethodParam params_anim_list_keys[] = {
    {"object_name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_anim_list_keys = {
    "anim.list_keys", "anim",
    "List an object's keyframes",
    nullptr,
    "read", "Read", false, "any",
    "anim|list|keys|animation|keyframe",
    nullptr,
    params_anim_list_keys, 1,
    true
};
static const MethodRegistration reg_anim_list_keys(desc_anim_list_keys);

static const MethodParam params_anim_play[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"clip", "string", true, "", nullptr, nullptr},
    {"blend", "float", false, "", "0.3", nullptr},
    {"layer", "int", false, "", "0", nullptr},
};
static const MethodDescriptor desc_anim_play = {
    "anim.play", "anim",
    "Play an animation clip on a character layer with a blend time",
    nullptr,
    "write", "SceneWrite", false, "any",
    "anim|play|animation|clip",
    nullptr,
    params_anim_play, 4,
    true
};
static const MethodRegistration reg_anim_play(desc_anim_play);

static const MethodParam params_anim_remove_key[] = {
    {"frame", "int", true, "", nullptr, nullptr},
    {"object_name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_anim_remove_key = {
    "anim.remove_key", "anim",
    "Remove an object's keyframe at a frame",
    nullptr,
    "write", "SceneWrite", false, "any",
    "anim|remove|key|animation|keyframe",
    nullptr,
    params_anim_remove_key, 2,
    true
};
static const MethodRegistration reg_anim_remove_key(desc_anim_remove_key);

static const MethodParam params_anim_set_graph_param[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"name", "string", true, "", nullptr, nullptr},
    {"value", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_anim_set_graph_param = {
    "anim.set_graph_param", "anim",
    "Set a float or bool parameter on a character's animation graph",
    nullptr,
    "write", "SceneWrite", false, "any",
    "anim|set|graph|param|animation|parameter",
    nullptr,
    params_anim_set_graph_param, 3,
    true
};
static const MethodRegistration reg_anim_set_graph_param(desc_anim_set_graph_param);

static const MethodParam params_anim_set_loop[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"loop", "bool", true, "", nullptr, nullptr},
    {"layer", "int", false, "", "0", nullptr},
};
static const MethodDescriptor desc_anim_set_loop = {
    "anim.set_loop", "anim",
    "Set whether a character layer loops",
    nullptr,
    "write", "SceneWrite", false, "any",
    "anim|set|loop|animation",
    nullptr,
    params_anim_set_loop, 3,
    true
};
static const MethodRegistration reg_anim_set_loop(desc_anim_set_loop);

static const MethodParam params_anim_set_paused[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"paused", "bool", false, "", "true", nullptr},
};
static const MethodDescriptor desc_anim_set_paused = {
    "anim.set_paused", "anim",
    "Pause or resume a character's animation",
    nullptr,
    "write", "SceneWrite", false, "any",
    "anim|set|paused|animation|pause",
    nullptr,
    params_anim_set_paused, 2,
    true
};
static const MethodRegistration reg_anim_set_paused(desc_anim_set_paused);

static const MethodParam params_anim_set_speed[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"speed", "float", true, "", nullptr, nullptr},
    {"layer", "int", false, "", "0", nullptr},
};
static const MethodDescriptor desc_anim_set_speed = {
    "anim.set_speed", "anim",
    "Set playback speed on a character layer",
    nullptr,
    "write", "SceneWrite", false, "any",
    "anim|set|speed|animation",
    nullptr,
    params_anim_set_speed, 3,
    true
};
static const MethodRegistration reg_anim_set_speed(desc_anim_set_speed);

static const MethodParam params_anim_set_time[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"seconds", "float", true, "", nullptr, nullptr},
    {"layer", "int", false, "", "0", nullptr},
};
static const MethodDescriptor desc_anim_set_time = {
    "anim.set_time", "anim",
    "Set playback time in seconds on a character layer",
    nullptr,
    "write", "SceneWrite", false, "any",
    "anim|set|time|animation|scrub",
    nullptr,
    params_anim_set_time, 3,
    true
};
static const MethodRegistration reg_anim_set_time(desc_anim_set_time);

static const MethodParam params_anim_status[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"layer", "int", false, "", "0", nullptr},
};
static const MethodDescriptor desc_anim_status = {
    "anim.status", "anim",
    "Report a character layer's playback state",
    nullptr,
    "read", "Read", false, "any",
    "anim|status|animation|progress",
    nullptr,
    params_anim_status, 2,
    true
};
static const MethodRegistration reg_anim_status(desc_anim_status);

static const MethodParam params_anim_stop[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"blend_out", "float", false, "", "0.3", nullptr},
    {"layer", "int", false, "", "0", nullptr},
};
static const MethodDescriptor desc_anim_stop = {
    "anim.stop", "anim",
    "Stop playback on a character layer with a blend-out time",
    nullptr,
    "write", "SceneWrite", false, "any",
    "anim|stop|animation",
    nullptr,
    params_anim_stop, 3,
    true
};
static const MethodRegistration reg_anim_stop(desc_anim_stop);

static const MethodParam params_anim_trigger_graph_param[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_anim_trigger_graph_param = {
    "anim.trigger_graph_param", "anim",
    "Fire a trigger parameter on a character's animation graph",
    nullptr,
    "write", "SceneWrite", false, "any",
    "anim|trigger|graph|param|animation",
    nullptr,
    params_anim_trigger_graph_param, 2,
    true
};
static const MethodRegistration reg_anim_trigger_graph_param(desc_anim_trigger_graph_param);

static const MethodParam params_batch[] = {
    {"calls", "array", true, "Array of {method, params} objects", nullptr, nullptr},
};
static const MethodDescriptor desc_batch = {
    "batch", "batch",
    "Execute several IPC calls in order and return one result array",
    "Each child call is authorized on its own; nesting a batch inside a batch is refused. A failing child does not stop the rest.",
    "read", "Read", false, "any",
    "batch|sequence|bulk|transaction",
    nullptr,
    params_batch, 1,
    true
};
static const MethodRegistration reg_batch(desc_batch);

static const MethodDescriptor desc_camera_get = {
    "camera.get", "camera",
    "Return camera position, target, up vector, field of view, focus distance and aperture",
    nullptr,
    "read", "Read", false, "CameraState",
    "camera|get|view",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_camera_get(desc_camera_get);

static const MethodParam params_camera_set_aperture[] = {
    {"aperture", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_camera_set_aperture = {
    "camera.set_aperture", "camera",
    "Set the camera aperture; larger values give shallower depth of field",
    nullptr,
    "write", "SceneWrite", false, "any",
    "camera|set|aperture|view|lens|dof|blur|bokeh",
    nullptr,
    params_camera_set_aperture, 1,
    true
};
static const MethodRegistration reg_camera_set_aperture(desc_camera_set_aperture);

static const MethodParam params_camera_set_focus_distance[] = {
    {"focus_distance", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_camera_set_focus_distance = {
    "camera.set_focus_distance", "camera",
    "Set the depth-of-field focus distance in metres",
    nullptr,
    "write", "SceneWrite", false, "any",
    "camera|set|focus|distance|view|lens|dof|blur",
    nullptr,
    params_camera_set_focus_distance, 1,
    true
};
static const MethodRegistration reg_camera_set_focus_distance(desc_camera_set_focus_distance);

static const MethodParam params_camera_set_fov[] = {
    {"fov", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_camera_set_fov = {
    "camera.set_fov", "camera",
    "Set the camera field of view in degrees",
    nullptr,
    "write", "SceneWrite", false, "any",
    "camera|set|fov|view|lens|zoom",
    nullptr,
    params_camera_set_fov, 1,
    true
};
static const MethodRegistration reg_camera_set_fov(desc_camera_set_fov);

static const MethodParam params_camera_set_position[] = {
    {"position", "vec3", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_camera_set_position = {
    "camera.set_position", "camera",
    "Move the camera to a world position",
    nullptr,
    "write", "SceneWrite", false, "any",
    "camera|set|position|view|move",
    nullptr,
    params_camera_set_position, 1,
    true
};
static const MethodRegistration reg_camera_set_position(desc_camera_set_position);

static const MethodParam params_camera_set_target[] = {
    {"target", "vec3", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_camera_set_target = {
    "camera.set_target", "camera",
    "Aim the camera at a world position",
    nullptr,
    "write", "SceneWrite", false, "any",
    "camera|set|target|view|look|aim",
    nullptr,
    params_camera_set_target, 1,
    true
};
static const MethodRegistration reg_camera_set_target(desc_camera_set_target);

static const MethodParam params_debris_configure[] = {
    {"enabled", "bool", false, "", "true", nullptr},
};
static const MethodDescriptor desc_debris_configure = {
    "debris.configure", "debris",
    "Enable and configure the ash/debris particle system",
    nullptr,
    "write", "SceneWrite", false, "any",
    "debris|configure|fire|ash|particles",
    nullptr,
    params_debris_configure, 1,
    true
};
static const MethodRegistration reg_debris_configure(desc_debris_configure);

static const MethodParam params_debris_emit_ash[] = {
    {"center", "any", false, "", nullptr, nullptr},
    {"mass_kg", "float", false, "", "0.0", nullptr},
    {"velocity", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_debris_emit_ash = {
    "debris.emit_ash", "debris",
    "Emit ash debris of a given mass at a point",
    nullptr,
    "write", "SceneWrite", false, "any",
    "debris|emit|ash|fire",
    nullptr,
    params_debris_emit_ash, 3,
    true
};
static const MethodRegistration reg_debris_emit_ash(desc_debris_emit_ash);

static const MethodDescriptor desc_debris_stats = {
    "debris.stats", "debris",
    "Report ash debris counts, accepted mass and budget rejections",
    nullptr,
    "write", "SceneWrite", false, "any",
    "debris|stats|fire|ash|measure",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_debris_stats(desc_debris_stats);

static const MethodDescriptor desc_editor_get_state = {
    "editor.get_state", "editor",
    "Report which editors are open, the node editor's domain and the simulation graph scope being shown",
    "Editor state is a VALUE, so it travels over IPC. Panel drawing does not - that is the rt.ui exemption.",
    "read", "Read", false, "any",
    "editor|get|state|ui|context",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_editor_get_state(desc_editor_get_state);

static const MethodParam params_editor_set_bottom_editor[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_editor_set_bottom_editor = {
    "editor.set_bottom_editor", "editor",
    "Choose which editor fills the bottom dock",
    nullptr,
    "write", "SceneWrite", false, "any",
    "editor|set|bottom|ui",
    nullptr,
    params_editor_set_bottom_editor, 1,
    true
};
static const MethodRegistration reg_editor_set_bottom_editor(desc_editor_set_bottom_editor);

static const MethodParam params_editor_set_node_domain[] = {
    {"name", "string", true, "Node editor domain", nullptr, "material|geometry|terrain"},
};
static const MethodDescriptor desc_editor_set_node_domain = {
    "editor.set_node_domain", "editor",
    "Choose which domain's graph the node editor shows",
    nullptr,
    "write", "SceneWrite", false, "any",
    "editor|set|node|domain|ui|nodes",
    nullptr,
    params_editor_set_node_domain, 1,
    true
};
static const MethodRegistration reg_editor_set_node_domain(desc_editor_set_node_domain);

static const MethodParam params_editor_set_sim_graph_scope[] = {
    {"scope", "string", true, "", nullptr, nullptr},
    {"owner", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_editor_set_sim_graph_scope = {
    "editor.set_sim_graph_scope", "editor",
    "Choose which scope and owner the simulation node editor shows",
    nullptr,
    "write", "SceneWrite", false, "any",
    "editor|set|sim|graph|scope|ui|simulation",
    nullptr,
    params_editor_set_sim_graph_scope, 2,
    true
};
static const MethodRegistration reg_editor_set_sim_graph_scope(desc_editor_set_sim_graph_scope);

static const MethodParam params_flow_source_create[] = {
    {"name", "string", false, "Emitter name", nullptr, nullptr},
    {"domain", "string", false, "Target fluid or gas domain name", nullptr, nullptr},
    {"source_mode", "string", false, "Emission shape/source kind", nullptr, nullptr},
    {"source_object", "string", false, "Object whose surface or volume emits", nullptr, nullptr},
    {"parent_object", "string", false, "Object the emitter follows", nullptr, nullptr},
    {"position", "vec3", false, "Emitter position; parent-local when parented", nullptr, nullptr},
    {"velocity", "vec3", false, "Emitted velocity in metres per second", nullptr, nullptr},
    {"velocity_space", "string", false, "Frame the velocity is expressed in", nullptr, nullptr},
    {"radius", "float", false, "Emission radius in metres", nullptr, nullptr},
    {"temperature", "float", false, "Injected temperature in Kelvin", nullptr, nullptr},
    {"fuel", "float", false, "Injected fuel density", nullptr, nullptr},
    {"density", "float", false, "Injected smoke or liquid density", nullptr, nullptr},
    {"fluid_substance", "string", false, "Substance id emitted into a liquid domain", nullptr, nullptr},
    {"fluid_particles_per_second", "float", false, "Liquid particle emission rate", nullptr, nullptr},
    {"enabled", "bool", false, "Emitter active", nullptr, nullptr},
    {"end_time", "any", false, "", nullptr, nullptr},
    {"falloff", "any", false, "", nullptr, nullptr},
    {"fluid_emit_along_normal", "any", false, "", nullptr, nullptr},
    {"fluid_velocity_spread", "any", false, "", nullptr, nullptr},
    {"inherit_velocity", "any", false, "", nullptr, nullptr},
    {"max_emitted_particles", "any", false, "", nullptr, nullptr},
    {"start_time", "any", false, "", nullptr, nullptr},
    {"use_particle_limit", "any", false, "", nullptr, nullptr},
    {"use_time_limit", "any", false, "", nullptr, nullptr},
    {"velocity_coupling", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_flow_source_create = {
    "flow_source.create", "flow_source",
    "Create an emitter that feeds a fluid or gas domain with mass, heat, fuel or particles",
    "An emitter belongs to a domain by name. With a parent_object it follows that object, and velocity_space decides whether its velocity is read in world or parent-local space.",
    "write", "SceneWrite", false, "FlowSourceInfo",
    "flow_source|flow|source|create|simulation|emitter|inject|pour|jet|flame|inflow",
    "flow_source.update|flow_source.list|fluid.create_domain|gas.set_settings",
    params_flow_source_create, 25,
    true
};
static const MethodRegistration reg_flow_source_create(desc_flow_source_create);

static const MethodParam params_flow_source_get[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_flow_source_get = {
    "flow_source.get", "flow_source",
    "Return one emitter's full settings",
    nullptr,
    "read", "Read", false, "FlowSourceInfo",
    "flow_source|flow|source|get|simulation|emitter",
    nullptr,
    params_flow_source_get, 1,
    true
};
static const MethodRegistration reg_flow_source_get(desc_flow_source_get);

static const MethodDescriptor desc_flow_source_list = {
    "flow_source.list", "flow_source",
    "List every emitter with its settings",
    nullptr,
    "read", "Read", false, "FlowSourceInfo[]",
    "flow_source|flow|source|list|simulation|emitter|inventory",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_flow_source_list(desc_flow_source_list);

static const MethodParam params_flow_source_remove[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_flow_source_remove = {
    "flow_source.remove", "flow_source",
    "Delete an emitter",
    nullptr,
    "write", "SceneWrite", false, "any",
    "flow_source|flow|source|remove|simulation|emitter",
    nullptr,
    params_flow_source_remove, 1,
    true
};
static const MethodRegistration reg_flow_source_remove(desc_flow_source_remove);

static const MethodParam params_flow_source_update[] = {
    {"name", "string", true, "", nullptr, nullptr},
    {"density", "any", false, "", nullptr, nullptr},
    {"domain", "any", false, "", nullptr, nullptr},
    {"enabled", "any", false, "", nullptr, nullptr},
    {"end_time", "any", false, "", nullptr, nullptr},
    {"falloff", "any", false, "", nullptr, nullptr},
    {"fluid_emit_along_normal", "any", false, "", nullptr, nullptr},
    {"fluid_particles_per_second", "any", false, "", nullptr, nullptr},
    {"fluid_substance", "any", false, "", nullptr, nullptr},
    {"fluid_velocity_spread", "any", false, "", nullptr, nullptr},
    {"fuel", "any", false, "", nullptr, nullptr},
    {"inherit_velocity", "any", false, "", nullptr, nullptr},
    {"max_emitted_particles", "any", false, "", nullptr, nullptr},
    {"parent_object", "any", false, "", nullptr, nullptr},
    {"position", "any", false, "", nullptr, nullptr},
    {"radius", "any", false, "", nullptr, nullptr},
    {"source_mode", "any", false, "", nullptr, nullptr},
    {"source_object", "any", false, "", nullptr, nullptr},
    {"start_time", "any", false, "", nullptr, nullptr},
    {"temperature", "any", false, "", nullptr, nullptr},
    {"use_particle_limit", "any", false, "", nullptr, nullptr},
    {"use_time_limit", "any", false, "", nullptr, nullptr},
    {"velocity", "any", false, "", nullptr, nullptr},
    {"velocity_coupling", "any", false, "", nullptr, nullptr},
    {"velocity_space", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_flow_source_update = {
    "flow_source.update", "flow_source",
    "Update fields of an existing emitter, keeping everything you do not send",
    "Read-modify-write on purpose: sending only one field must not reset the rest.",
    "write", "SceneWrite", false, "any",
    "flow_source|flow|source|update|simulation|emitter|configure",
    "flow_source.create|flow_source.get",
    params_flow_source_update, 25,
    true
};
static const MethodRegistration reg_flow_source_update(desc_flow_source_update);

static const MethodParam params_fluid_clear[] = {
    {"domain", "string", true, "", nullptr, nullptr},
    {"clear_seed", "bool", false, "", "false", nullptr},
};
static const MethodDescriptor desc_fluid_clear = {
    "fluid.clear", "fluid",
    "Remove the particles from a fluid domain",
    nullptr,
    "write", "SceneWrite", false, "any",
    "fluid|clear|simulation|reset",
    nullptr,
    params_fluid_clear, 2,
    true
};
static const MethodRegistration reg_fluid_clear(desc_fluid_clear);

static const MethodParam params_fluid_create_domain[] = {
    {"name", "string", false, "Domain name; also the object name in the scene", nullptr, nullptr},
    {"type", "string", false, "Solver family", nullptr, "fluid|gas"},
    {"domain_min", "vec3", false, "World-space AABB minimum in metres", nullptr, nullptr},
    {"domain_max", "vec3", false, "World-space AABB maximum in metres", nullptr, nullptr},
    {"voxel_size", "float", false, "Grid cell size in metres. Drives resolution, cost and the smallest feature the solver can resolve.", "0.05", nullptr},
};
static const MethodDescriptor desc_fluid_create_domain = {
    "fluid.create_domain", "fluid",
    "Create an APIC liquid or gas grid domain over a world-space box",
    "voxel_size sets both resolution and cost: halving it multiplies cell count by eight. A liquid domain does nothing until fluid.seed or a flow_source fills it. Changing voxel_size later invalidates the bake.",
    "write", "SceneWrite", false, "FluidDomainInfo",
    "fluid|create|domain|simulation|liquid|water|smoke|fire|grid",
    "fluid.seed|fluid.set_param|flow_source.create|fluid.list_domains",
    params_fluid_create_domain, 5,
    true
};
static const MethodRegistration reg_fluid_create_domain(desc_fluid_create_domain);

static const MethodParam params_fluid_get[] = {
    {"domain", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_fluid_get = {
    "fluid.get", "fluid",
    "Return one fluid or gas domain's complete settings and live counters",
    nullptr,
    "read", "Read", false, "FluidDomainInfo",
    "fluid|get|simulation",
    nullptr,
    params_fluid_get, 1,
    true
};
static const MethodRegistration reg_fluid_get(desc_fluid_get);

static const MethodParam params_fluid_get_combustion[] = {
    {"domain", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_fluid_get_combustion = {
    "fluid.get_combustion", "fluid",
    "Read a liquid domain's combustion settings",
    nullptr,
    "read", "Read", false, "any",
    "fluid|get|combustion|simulation|fire|burn",
    "fluid.set_combustion",
    params_fluid_get_combustion, 1,
    true
};
static const MethodRegistration reg_fluid_get_combustion(desc_fluid_get_combustion);

static const MethodDescriptor desc_fluid_list_domains = {
    "fluid.list_domains", "fluid",
    "List every fluid and gas domain with its full settings",
    nullptr,
    "read", "Read", false, "FluidDomainInfo[]",
    "fluid|list|domains|simulation|inventory",
    "fluid.get",
    nullptr, 0,
    true
};
static const MethodRegistration reg_fluid_list_domains(desc_fluid_list_domains);

static const MethodParam params_fluid_remove_domain[] = {
    {"domain", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_fluid_remove_domain = {
    "fluid.remove_domain", "fluid",
    "Delete a fluid or gas domain",
    nullptr,
    "write", "SceneWrite", false, "any",
    "fluid|remove|domain|simulation",
    nullptr,
    params_fluid_remove_domain, 1,
    true
};
static const MethodRegistration reg_fluid_remove_domain(desc_fluid_remove_domain);

static const MethodDescriptor desc_fluid_reset = {
    "fluid.reset", "fluid",
    "Reset every fluid and gas simulation to frame zero",
    nullptr,
    "write", "SceneWrite", false, "any",
    "fluid|reset|simulation|rewind",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_fluid_reset(desc_fluid_reset);

static const MethodParam params_fluid_seed[] = {
    {"domain", "string", true, "", nullptr, nullptr},
    {"seed_min", "vec3", false, "World-space box minimum in metres", nullptr, nullptr},
    {"seed_max", "vec3", false, "World-space box maximum in metres", nullptr, nullptr},
    {"particles_per_cell", "int", false, "Particle density per grid cell; 4 is standard", "4", nullptr},
    {"replace", "bool", false, "Clear existing particles first", "true", nullptr},
    {"persistent", "bool", false, "Re-apply this seed after every reset", "false", nullptr},
};
static const MethodDescriptor desc_fluid_seed = {
    "fluid.seed", "fluid",
    "Fill a box inside a liquid domain with particles",
    "particles_per_cell of 4 is the standard APIC density; a thin jet seeded below that has no interior cells and therefore no pressure field, which reads as 'water that will not splash'. persistent=true re-seeds the box every reset.",
    "write", "SceneWrite", false, "any",
    "fluid|seed|simulation|liquid|water|fill|particles|initial",
    "fluid.create_domain|fluid.clear|flow_source.create",
    params_fluid_seed, 6,
    true
};
static const MethodRegistration reg_fluid_seed(desc_fluid_seed);

static const MethodParam params_fluid_set_combustion[] = {
    {"domain", "string", true, "", nullptr, nullptr},
    {"auto_ignite", "bool", false, "", nullptr, nullptr},
    {"chemistry_preset", "any", false, "", nullptr, nullptr},
    {"enabled", "bool", false, "", nullptr, nullptr},
    {"evaporation_rate", "float", false, "", nullptr, nullptr},
    {"heat_release", "float", false, "", nullptr, nullptr},
    {"ignition_temperature", "float", false, "", nullptr, nullptr},
    {"smoke_yield", "float", false, "", nullptr, nullptr},
    {"surface_cooling", "float", false, "", nullptr, nullptr},
    {"surface_fuel_capacity", "float", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_fluid_set_combustion = {
    "fluid.set_combustion", "fluid",
    "Configure a liquid domain as a combustible fuel: ignition temperature, heat release, smoke yield and evaporation",
    nullptr,
    "write", "SceneWrite", false, "any",
    "fluid|set|combustion|simulation|fire|burn|fuel|ignite",
    "gas.set_settings|fluid.set_substance_material",
    params_fluid_set_combustion, 10,
    true
};
static const MethodRegistration reg_fluid_set_combustion(desc_fluid_set_combustion);

static const MethodParam params_fluid_set_param[] = {
    {"domain", "string", true, "", nullptr, nullptr},
    {"backend", "any", false, "", nullptr, nullptr},
    {"boundary", "any", false, "", nullptr, nullptr},
    {"coord_space", "any", false, "", nullptr, nullptr},
    {"device", "any", false, "", nullptr, nullptr},
    {"domain_max", "vec3", false, "", nullptr, nullptr},
    {"domain_min", "vec3", false, "", nullptr, nullptr},
    {"enabled", "any", false, "", nullptr, nullptr},
    {"granular_cohesion", "any", false, "", nullptr, nullptr},
    {"granular_damage_rate", "any", false, "", nullptr, nullptr},
    {"granular_dilatancy", "any", false, "", nullptr, nullptr},
    {"granular_enabled", "any", false, "", nullptr, nullptr},
    {"granular_fracture_strain", "any", false, "", nullptr, nullptr},
    {"granular_friction_angle", "any", false, "", nullptr, nullptr},
    {"granular_hardening", "any", false, "", nullptr, nullptr},
    {"granular_healing_rate", "any", false, "", nullptr, nullptr},
    {"granular_max_solver_substeps", "any", false, "", nullptr, nullptr},
    {"granular_poisson_ratio", "any", false, "", nullptr, nullptr},
    {"granular_rebonding", "any", false, "", nullptr, nullptr},
    {"granular_residual_strength", "any", false, "", nullptr, nullptr},
    {"granular_softening_range", "any", false, "", nullptr, nullptr},
    {"granular_softening_temperature", "any", false, "", nullptr, nullptr},
    {"granular_tack_peak", "any", false, "", nullptr, nullptr},
    {"granular_tensile_cutoff", "any", false, "", nullptr, nullptr},
    {"granular_thermal_conductivity", "any", false, "", nullptr, nullptr},
    {"granular_young_modulus", "any", false, "", nullptr, nullptr},
    {"kinematic_viscosity", "any", false, "", nullptr, nullptr},
    {"pore_amount", "any", false, "", nullptr, nullptr},
    {"pore_detail", "any", false, "", nullptr, nullptr},
    {"pore_scale", "any", false, "", nullptr, nullptr},
    {"preset", "any", false, "", nullptr, nullptr},
    {"render_mode", "any", false, "", nullptr, nullptr},
    {"solid_phase", "any", false, "", nullptr, nullptr},
    {"solid_phase_fill", "any", false, "", nullptr, nullptr},
    {"surface_material", "any", false, "", nullptr, nullptr},
    {"surface_offset_voxels", "any", false, "", nullptr, nullptr},
    {"uvw_refresh_period", "any", false, "", nullptr, nullptr},
    {"viscosity", "any", false, "", nullptr, nullptr},
    {"viscosity_sweeps", "any", false, "", nullptr, nullptr},
    {"viscosity_wall_slip", "any", false, "", nullptr, nullptr},
    {"visible", "any", false, "", nullptr, nullptr},
    {"voxel_size", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_fluid_set_param = {
    "fluid.set_param", "fluid",
    "Update any field of a fluid or gas domain: bounds, voxel size, solver backend, viscosity, granular constitutive settings, render mode and surface material",
    "Overlay semantics - fields you do not send keep their value. Changing voxel_size or the bounds invalidates the bake for that domain.",
    "write", "SceneWrite", false, "any",
    "fluid|set|param|simulation|configure|viscosity|granular|render-mode|backend",
    "fluid.get|fluid.set_substance_material",
    params_fluid_set_param, 42,
    true
};
static const MethodRegistration reg_fluid_set_param(desc_fluid_set_param);

static const MethodParam params_fluid_set_splat_material[] = {
    {"domain", "string", true, "", nullptr, nullptr},
    {"material", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_fluid_set_splat_material = {
    "fluid.set_splat_material", "fluid",
    "Set the material used for particle splat rendering of a domain",
    nullptr,
    "write", "SceneWrite", false, "any",
    "fluid|set|splat|material|simulation|render|appearance",
    nullptr,
    params_fluid_set_splat_material, 2,
    true
};
static const MethodRegistration reg_fluid_set_splat_material(desc_fluid_set_splat_material);

static const MethodParam params_fluid_set_substance_material[] = {
    {"domain", "string", true, "Fluid domain name", nullptr, nullptr},
    {"substance", "string", true, "Substance id from msf.substances", nullptr, nullptr},
    {"phase", "string", false, "Physical phase", nullptr, "liquid|solid|gas"},
    {"representation", "string", false, "How the substance is solved and rendered", nullptr, nullptr},
    {"kinematic_viscosity", "any", false, "", nullptr, nullptr},
    {"material", "string", false, "", "", nullptr},
    {"miscibility", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_fluid_set_substance_material = {
    "fluid.set_substance_material", "fluid",
    "Bind a substance (its physical identity, phase and representation) to a fluid domain",
    "Phase is what the matter IS; representation is how it is solved and drawn. They are separate axes - setting one does not imply the other.",
    "write", "SceneWrite", false, "any",
    "fluid|set|substance|material|simulation|phase|msf",
    "msf.substances|fluid.set_splat_material|fluid.set_param",
    params_fluid_set_substance_material, 7,
    true
};
static const MethodRegistration reg_fluid_set_substance_material(desc_fluid_set_substance_material);

static const MethodParam params_fluid_step[] = {
    {"dt", "float", false, "", "0.0166667", nullptr},
};
static const MethodDescriptor desc_fluid_step = {
    "fluid.step", "fluid",
    "Advance the fluid solver by one timestep",
    "For scripted work prefer timeline.set_frame, which advances every solver consistently.",
    "write", "SceneWrite", false, "any",
    "fluid|step|simulation|advance",
    nullptr,
    params_fluid_step, 1,
    true
};
static const MethodRegistration reg_fluid_step(desc_fluid_step);

static const MethodParam params_forcefield_create[] = {
    {"type", "string", true, "", nullptr, nullptr},
    {"name", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_forcefield_create = {
    "forcefield.create", "forcefield",
    "Create a force field of the given type and return its name",
    nullptr,
    "write", "SceneWrite", false, "any",
    "forcefield|create|simulation|wind|vortex|turbulence|gravity",
    nullptr,
    params_forcefield_create, 2,
    true
};
static const MethodRegistration reg_forcefield_create(desc_forcefield_create);

static const MethodParam params_forcefield_evaluate[] = {
    {"position", "vec3", true, "", nullptr, nullptr},
    {"time", "float", false, "", "0.0", nullptr},
    {"velocity", "vec3", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_forcefield_evaluate = {
    "forcefield.evaluate", "forcefield",
    "Evaluate the combined force fields at a world position and return the resulting acceleration",
    "The way to check a field really reaches a place, rather than inferring it from how a simulation looks.",
    "read", "Read", false, "any",
    "forcefield|evaluate|measure|verify|probe|force",
    "forcefield.set_param",
    params_forcefield_evaluate, 3,
    true
};
static const MethodRegistration reg_forcefield_evaluate(desc_forcefield_evaluate);

static const MethodParam params_forcefield_get[] = {
    {"field", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_forcefield_get = {
    "forcefield.get", "forcefield",
    "Return one force field's full settings",
    nullptr,
    "read", "Read", false, "any",
    "forcefield|get|simulation",
    nullptr,
    params_forcefield_get, 1,
    true
};
static const MethodRegistration reg_forcefield_get(desc_forcefield_get);

static const MethodDescriptor desc_forcefield_list = {
    "forcefield.list", "forcefield",
    "List the force fields in the scene",
    nullptr,
    "read", "Read", false, "any",
    "forcefield|list|simulation|inventory",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_forcefield_list(desc_forcefield_list);

static const MethodParam params_forcefield_remove[] = {
    {"field", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_forcefield_remove = {
    "forcefield.remove", "forcefield",
    "Delete a force field",
    nullptr,
    "write", "SceneWrite", false, "any",
    "forcefield|remove|simulation",
    nullptr,
    params_forcefield_remove, 1,
    true
};
static const MethodRegistration reg_forcefield_remove(desc_forcefield_remove);

static const MethodParam params_forcefield_set_param[] = {
    {"field", "string", true, "", nullptr, nullptr},
    {"affects_cloth", "bool", false, "", nullptr, nullptr},
    {"affects_fluid", "bool", false, "", nullptr, nullptr},
    {"affects_gas", "bool", false, "", nullptr, nullptr},
    {"affects_particles", "bool", false, "", nullptr, nullptr},
    {"affects_rigidbody", "bool", false, "", nullptr, nullptr},
    {"axis", "vec3", false, "", nullptr, nullptr},
    {"direction", "vec3", false, "", nullptr, nullptr},
    {"enabled", "bool", false, "", nullptr, nullptr},
    {"end_frame", "float", false, "", nullptr, nullptr},
    {"falloff", "string", false, "", nullptr, nullptr},
    {"falloff_radius", "float", false, "", nullptr, nullptr},
    {"fluid_curl_detail", "float", false, "", nullptr, nullptr},
    {"fluid_drag_coupling", "float", false, "", nullptr, nullptr},
    {"fluid_surface_depth", "float", false, "", nullptr, nullptr},
    {"fluid_surface_drag", "bool", false, "", nullptr, nullptr},
    {"inner_radius", "float", false, "", nullptr, nullptr},
    {"inward_force", "float", false, "", nullptr, nullptr},
    {"linear_drag", "float", false, "", nullptr, nullptr},
    {"name", "string", false, "", nullptr, nullptr},
    {"noise_amplitude", "float", false, "", nullptr, nullptr},
    {"noise_frequency", "float", false, "", nullptr, nullptr},
    {"noise_lacunarity", "float", false, "", nullptr, nullptr},
    {"noise_octaves", "int", false, "", nullptr, nullptr},
    {"noise_persistence", "float", false, "", nullptr, nullptr},
    {"noise_seed", "int", false, "", nullptr, nullptr},
    {"noise_speed", "float", false, "", nullptr, nullptr},
    {"phase", "float", false, "", nullptr, nullptr},
    {"position", "vec3", false, "", nullptr, nullptr},
    {"quadratic_drag", "float", false, "", nullptr, nullptr},
    {"rotation", "vec3", false, "", nullptr, nullptr},
    {"scale", "vec3", false, "", nullptr, nullptr},
    {"shape", "string", false, "", nullptr, nullptr},
    {"start_frame", "float", false, "", nullptr, nullptr},
    {"strength", "float", false, "", nullptr, nullptr},
    {"thermal_delta_kelvin", "float", false, "", nullptr, nullptr},
    {"type", "string", false, "", nullptr, nullptr},
    {"upward_force", "float", false, "", nullptr, nullptr},
    {"use_noise", "bool", false, "", nullptr, nullptr},
    {"visible", "bool", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_forcefield_set_param = {
    "forcefield.set_param", "forcefield",
    "Update a force field: shape, strength, falloff, noise, drag and which solvers it affects",
    "A force field is spatial - it is NOT attached to a domain. Which solvers feel it is decided by the affects_* flags.",
    "write", "SceneWrite", false, "any",
    "forcefield|set|param|simulation|wind|turbulence|drag|configure",
    "forcefield.evaluate",
    params_forcefield_set_param, 40,
    true
};
static const MethodRegistration reg_forcefield_set_param(desc_forcefield_set_param);

static const MethodDescriptor desc_forcefield_types = {
    "forcefield.types", "forcefield",
    "List the available force-field types",
    nullptr,
    "read", "Read", false, "any",
    "forcefield|types|simulation|wind|vortex|drag",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_forcefield_types(desc_forcefield_types);

static const MethodParam params_gas_clear[] = {
    {"domain", "string", true, "", nullptr, nullptr},
    {"clear_seed", "bool", false, "", "false", nullptr},
};
static const MethodDescriptor desc_gas_clear = {
    "gas.clear", "gas",
    "Clear a gas domain's fields",
    nullptr,
    "write", "SceneWrite", false, "any",
    "gas|clear|simulation|reset",
    nullptr,
    params_gas_clear, 2,
    true
};
static const MethodRegistration reg_gas_clear(desc_gas_clear);

static const MethodParam params_gas_create_domain[] = {
    {"domain_max", "vec3", false, "", nullptr, nullptr},
    {"domain_min", "vec3", false, "", nullptr, nullptr},
    {"name", "string", false, "", nullptr, nullptr},
    {"type", "string", false, "", nullptr, nullptr},
    {"voxel_size", "float", false, "", "0.05", nullptr},
};
static const MethodDescriptor desc_gas_create_domain = {
    "gas.create_domain", "gas",
    "Create a gas grid domain over a world-space box",
    "Same handler as fluid.create_domain with type defaulting to gas.",
    "write", "SceneWrite", false, "FluidDomainInfo",
    "gas|create|domain|simulation|smoke|fire",
    "gas.set_settings|flow_source.create",
    params_gas_create_domain, 5,
    true
};
static const MethodRegistration reg_gas_create_domain(desc_gas_create_domain);

static const MethodParam params_gas_get[] = {
    {"domain", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_gas_get = {
    "gas.get", "gas",
    "Return one gas domain's complete settings and live counters",
    nullptr,
    "read", "Read", false, "any",
    "gas|get|simulation",
    nullptr,
    params_gas_get, 1,
    true
};
static const MethodRegistration reg_gas_get(desc_gas_get);

static const MethodParam params_gas_get_settings[] = {
    {"domain", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_gas_get_settings = {
    "gas.get_settings", "gas",
    "Read a gas domain's solver settings: fire, buoyancy, turbulence, quality and resource budget",
    nullptr,
    "read", "Read", false, "any",
    "gas|get|settings|simulation|fire|smoke",
    "gas.set_settings",
    params_gas_get_settings, 1,
    true
};
static const MethodRegistration reg_gas_get_settings(desc_gas_get_settings);

static const MethodParam params_gas_get_shader[] = {
    {"domain", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_gas_get_shader = {
    "gas.get_shader", "gas",
    "Read a gas domain's volume shading settings",
    nullptr,
    "read", "Read", false, "any",
    "gas|get|shader|render|appearance|volume",
    "gas.set_shader",
    params_gas_get_shader, 1,
    true
};
static const MethodRegistration reg_gas_get_shader(desc_gas_get_shader);

static const MethodDescriptor desc_gas_list_domains = {
    "gas.list_domains", "gas",
    "List every gas and fluid domain with its settings",
    nullptr,
    "read", "Read", false, "any",
    "gas|list|domains|simulation|inventory",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_gas_list_domains(desc_gas_list_domains);

static const MethodParam params_gas_pressure_pulse[] = {
    {"domain", "any", true, "", nullptr, nullptr},
    {"center", "any", false, "", nullptr, nullptr},
    {"coupling", "float", false, "", "1.0", nullptr},
    {"duration_seconds", "float", false, "", "0.02", nullptr},
    {"peak_pressure_kpa", "float", false, "", "0.0", nullptr},
    {"radius", "float", false, "", "1.0", nullptr},
};
static const MethodDescriptor desc_gas_pressure_pulse = {
    "gas.pressure_pulse", "gas",
    "Inject a pressure pulse into a gas domain, as an explosion or blast wave",
    nullptr,
    "write", "SceneWrite", false, "any",
    "gas|pressure|pulse|simulation|explosion|blast|shockwave",
    nullptr,
    params_gas_pressure_pulse, 6,
    true
};
static const MethodRegistration reg_gas_pressure_pulse(desc_gas_pressure_pulse);

static const MethodParam params_gas_remove_domain[] = {
    {"domain", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_gas_remove_domain = {
    "gas.remove_domain", "gas",
    "Delete a gas domain",
    nullptr,
    "write", "SceneWrite", false, "any",
    "gas|remove|domain|simulation",
    nullptr,
    params_gas_remove_domain, 1,
    true
};
static const MethodRegistration reg_gas_remove_domain(desc_gas_remove_domain);

static const MethodDescriptor desc_gas_reset = {
    "gas.reset", "gas",
    "Reset every gas and fluid simulation to frame zero",
    nullptr,
    "write", "SceneWrite", false, "any",
    "gas|reset|simulation",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_gas_reset(desc_gas_reset);

static const MethodParam params_gas_set_param[] = {
    {"domain", "string", true, "", nullptr, nullptr},
    {"backend", "any", false, "", nullptr, nullptr},
    {"boundary", "any", false, "", nullptr, nullptr},
    {"coord_space", "any", false, "", nullptr, nullptr},
    {"device", "any", false, "", nullptr, nullptr},
    {"domain_max", "vec3", false, "", nullptr, nullptr},
    {"domain_min", "vec3", false, "", nullptr, nullptr},
    {"enabled", "any", false, "", nullptr, nullptr},
    {"granular_cohesion", "any", false, "", nullptr, nullptr},
    {"granular_damage_rate", "any", false, "", nullptr, nullptr},
    {"granular_dilatancy", "any", false, "", nullptr, nullptr},
    {"granular_enabled", "any", false, "", nullptr, nullptr},
    {"granular_fracture_strain", "any", false, "", nullptr, nullptr},
    {"granular_friction_angle", "any", false, "", nullptr, nullptr},
    {"granular_hardening", "any", false, "", nullptr, nullptr},
    {"granular_healing_rate", "any", false, "", nullptr, nullptr},
    {"granular_max_solver_substeps", "any", false, "", nullptr, nullptr},
    {"granular_poisson_ratio", "any", false, "", nullptr, nullptr},
    {"granular_rebonding", "any", false, "", nullptr, nullptr},
    {"granular_residual_strength", "any", false, "", nullptr, nullptr},
    {"granular_softening_range", "any", false, "", nullptr, nullptr},
    {"granular_softening_temperature", "any", false, "", nullptr, nullptr},
    {"granular_tack_peak", "any", false, "", nullptr, nullptr},
    {"granular_tensile_cutoff", "any", false, "", nullptr, nullptr},
    {"granular_thermal_conductivity", "any", false, "", nullptr, nullptr},
    {"granular_young_modulus", "any", false, "", nullptr, nullptr},
    {"kinematic_viscosity", "any", false, "", nullptr, nullptr},
    {"pore_amount", "any", false, "", nullptr, nullptr},
    {"pore_detail", "any", false, "", nullptr, nullptr},
    {"pore_scale", "any", false, "", nullptr, nullptr},
    {"preset", "any", false, "", nullptr, nullptr},
    {"render_mode", "any", false, "", nullptr, nullptr},
    {"solid_phase", "any", false, "", nullptr, nullptr},
    {"solid_phase_fill", "any", false, "", nullptr, nullptr},
    {"surface_material", "any", false, "", nullptr, nullptr},
    {"surface_offset_voxels", "any", false, "", nullptr, nullptr},
    {"uvw_refresh_period", "any", false, "", nullptr, nullptr},
    {"viscosity", "any", false, "", nullptr, nullptr},
    {"viscosity_sweeps", "any", false, "", nullptr, nullptr},
    {"viscosity_wall_slip", "any", false, "", nullptr, nullptr},
    {"visible", "any", false, "", nullptr, nullptr},
    {"voxel_size", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_gas_set_param = {
    "gas.set_param", "gas",
    "Update any field of a gas domain (same overlay setter as fluid.set_param)",
    nullptr,
    "write", "SceneWrite", false, "any",
    "gas|set|param|simulation|configure",
    nullptr,
    params_gas_set_param, 42,
    true
};
static const MethodRegistration reg_gas_set_param(desc_gas_set_param);

static const MethodParam params_gas_set_settings[] = {
    {"domain", "string", true, "", nullptr, nullptr},
    {"fire_enabled", "bool", false, "Run combustion in this domain", nullptr, nullptr},
    {"ignition_temperature", "float", false, "Temperature at which fuel ignites, in Kelvin", nullptr, nullptr},
    {"burn_rate", "float", false, "How fast fuel is consumed once lit", nullptr, nullptr},
    {"heat_release", "float", false, "Temperature added per unit of fuel burned", nullptr, nullptr},
    {"smoke_generation", "float", false, "Smoke produced per unit of fuel burned", nullptr, nullptr},
    {"buoyancy_density", "float", false, "", nullptr, nullptr},
    {"buoyancy_heat", "float", false, "", nullptr, nullptr},
    {"enforce_resource_budget", "bool", false, "", nullptr, nullptr},
    {"fire_expansion", "float", false, "", nullptr, nullptr},
    {"fire_max_temperature", "float", false, "", nullptr, nullptr},
    {"flame_dissipation", "float", false, "", nullptr, nullptr},
    {"quality_profile", "string", false, "", nullptr, nullptr},
    {"render_to_nanovdb", "bool", false, "", nullptr, nullptr},
    {"resource_budget_mb", "int", false, "", nullptr, nullptr},
    {"structural_coupling_enabled", "bool", false, "", nullptr, nullptr},
    {"structural_event_interval", "float", false, "", nullptr, nullptr},
    {"structural_min_intensity", "float", false, "", nullptr, nullptr},
    {"structural_pressure_scale", "float", false, "", nullptr, nullptr},
    {"turbulence_lacunarity", "float", false, "", nullptr, nullptr},
    {"turbulence_octaves", "int", false, "", nullptr, nullptr},
    {"turbulence_persistence", "float", false, "", nullptr, nullptr},
    {"turbulence_scale", "float", false, "", nullptr, nullptr},
    {"turbulence_speed", "float", false, "", nullptr, nullptr},
    {"turbulence_strength", "float", false, "", nullptr, nullptr},
    {"use_sparse_tiles", "bool", false, "", nullptr, nullptr},
    {"vorticity", "float", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_gas_set_settings = {
    "gas.set_settings", "gas",
    "Configure a gas domain's solver: fire and ignition, burn rate, heat release, buoyancy, turbulence, vorticity and resource budget",
    "This is the SOLVER. Appearance - colour, emission, density scale - lives in gas.set_shader, and the two are read back separately; a fire that simulates but looks wrong is usually a shader setting, not a solver setting.",
    "write", "SceneWrite", false, "any",
    "gas|set|settings|simulation|fire|smoke|ignite|burn|buoyancy|turbulence",
    "gas.get_settings|gas.set_shader|flow_source.create",
    params_gas_set_settings, 27,
    true
};
static const MethodRegistration reg_gas_set_settings(desc_gas_set_settings);

static const MethodParam params_gas_set_shader[] = {
    {"domain", "string", true, "", nullptr, nullptr},
    {"absorption_coefficient", "float", false, "", nullptr, nullptr},
    {"blackbody_intensity", "float", false, "", nullptr, nullptr},
    {"density_cutoff", "float", false, "", nullptr, nullptr},
    {"density_multiplier", "float", false, "", nullptr, nullptr},
    {"preset", "string", false, "", nullptr, nullptr},
    {"scattering_coefficient", "float", false, "", nullptr, nullptr},
    {"temperature_max", "float", false, "", nullptr, nullptr},
    {"temperature_min", "float", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_gas_set_shader = {
    "gas.set_shader", "gas",
    "Set a gas domain's volume appearance: preset, density and absorption/scattering, blackbody emission and the temperature range it maps",
    "temperature_min/max define the window mapped to emission colour; a flame outside that window renders black however hot it is.",
    "write", "SceneWrite", false, "any",
    "gas|set|shader|render|appearance|volume|fire|colour|emission|blackbody",
    "gas.get_shader|gas.set_settings",
    params_gas_set_shader, 9,
    true
};
static const MethodRegistration reg_gas_set_shader(desc_gas_set_shader);

static const MethodParam params_gas_step[] = {
    {"dt", "float", false, "", "0.0166667", nullptr},
};
static const MethodDescriptor desc_gas_step = {
    "gas.step", "gas",
    "Advance the gas solver by one timestep",
    nullptr,
    "write", "SceneWrite", false, "any",
    "gas|step|simulation|advance",
    nullptr,
    params_gas_step, 1,
    true
};
static const MethodRegistration reg_gas_step(desc_gas_step);

static const MethodDescriptor desc_gas_structural_impulse_stats = {
    "gas.structural_impulse_stats", "gas",
    "Report the blast impulses the gas solver handed to the structural/fracture side",
    nullptr,
    "write", "SceneWrite", false, "any",
    "gas|structural|impulse|stats|diagnostics|blast|fracture|coupling",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_gas_structural_impulse_stats(desc_gas_structural_impulse_stats);

static const MethodParam params_hair_apply_preset[] = {
    {"name", "string", true, "", nullptr, nullptr},
    {"preset", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_hair_apply_preset = {
    "hair.apply_preset", "hair",
    "Apply a built-in hair preset to a groom",
    nullptr,
    "write", "SceneWrite", false, "any",
    "hair|apply|preset|style",
    nullptr,
    params_hair_apply_preset, 2,
    true
};
static const MethodRegistration reg_hair_apply_preset(desc_hair_apply_preset);

static const MethodParam params_hair_bake[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_hair_bake = {
    "hair.bake", "hair",
    "Bake a hair groom to its cached geometry",
    nullptr,
    "write", "SceneWrite", false, "any",
    "hair|bake",
    nullptr,
    params_hair_bake, 1,
    true
};
static const MethodRegistration reg_hair_bake(desc_hair_bake);

static const MethodParam params_hair_comb[] = {
    {"direction", "vec3", true, "", nullptr, nullptr},
    {"name", "string", true, "", nullptr, nullptr},
    {"root_stiffness", "float", false, "", "0.75", nullptr},
    {"strength", "float", false, "", "0.5", nullptr},
};
static const MethodDescriptor desc_hair_comb = {
    "hair.comb", "hair",
    "Comb hair strands towards a direction with a root stiffness",
    nullptr,
    "write", "SceneWrite", false, "any",
    "hair|comb|style|direction",
    nullptr,
    params_hair_comb, 4,
    true
};
static const MethodRegistration reg_hair_comb(desc_hair_comb);

static const MethodParam params_hair_create[] = {
    {"mesh", "string", true, "", nullptr, nullptr},
    {"name", "string", false, "", "HairGroom", nullptr},
};
static const MethodDescriptor desc_hair_create = {
    "hair.create", "hair",
    "Create a hair groom bound to a mesh",
    nullptr,
    "write", "SceneWrite", false, "any",
    "hair|create|fur|grass",
    nullptr,
    params_hair_create, 2,
    true
};
static const MethodRegistration reg_hair_create(desc_hair_create);

static const MethodParam params_hair_get[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_hair_get = {
    "hair.get", "hair",
    "Return one hair groom's full settings",
    nullptr,
    "read", "Read", false, "any",
    "hair|get",
    nullptr,
    params_hair_get, 1,
    true
};
static const MethodRegistration reg_hair_get(desc_hair_get);

static const MethodParam params_hair_grow[] = {
    {"length_factor", "float", true, "", nullptr, nullptr},
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_hair_grow = {
    "hair.grow", "hair",
    "Scale hair length by a factor",
    nullptr,
    "write", "SceneWrite", false, "any",
    "hair|grow|length|style",
    nullptr,
    params_hair_grow, 2,
    true
};
static const MethodRegistration reg_hair_grow(desc_hair_grow);

static const MethodDescriptor desc_hair_list = {
    "hair.list", "hair",
    "List the hair grooms",
    nullptr,
    "read", "Read", false, "any",
    "hair|list|fur|grass|inventory",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_hair_list(desc_hair_list);

static const MethodDescriptor desc_hair_list_presets = {
    "hair.list_presets", "hair",
    "List the available hair presets",
    nullptr,
    "read", "Read", false, "any",
    "hair|list|presets|preset",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_hair_list_presets(desc_hair_list_presets);

static const MethodParam params_hair_remove[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_hair_remove = {
    "hair.remove", "hair",
    "Delete a hair groom",
    nullptr,
    "write", "SceneWrite", false, "any",
    "hair|remove",
    nullptr,
    params_hair_remove, 1,
    true
};
static const MethodRegistration reg_hair_remove(desc_hair_remove);

static const MethodParam params_hair_rename[] = {
    {"name", "string", true, "", nullptr, nullptr},
    {"new_name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_hair_rename = {
    "hair.rename", "hair",
    "Rename a hair groom",
    nullptr,
    "write", "SceneWrite", false, "any",
    "hair|rename",
    nullptr,
    params_hair_rename, 2,
    true
};
static const MethodRegistration reg_hair_rename(desc_hair_rename);

static const MethodParam params_hair_reset_simulation[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_hair_reset_simulation = {
    "hair.reset_simulation", "hair",
    "Reset hair dynamics to the rest pose",
    nullptr,
    "write", "SceneWrite", false, "any",
    "hair|reset|simulation",
    nullptr,
    params_hair_reset_simulation, 1,
    true
};
static const MethodRegistration reg_hair_reset_simulation(desc_hair_reset_simulation);

static const MethodParam params_hair_restyle[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_hair_restyle = {
    "hair.restyle", "hair",
    "Restyle a hair groom",
    nullptr,
    "write", "SceneWrite", false, "any",
    "hair|restyle|style",
    nullptr,
    params_hair_restyle, 1,
    true
};
static const MethodRegistration reg_hair_restyle(desc_hair_restyle);

static const MethodParam params_hair_smooth[] = {
    {"name", "string", true, "", nullptr, nullptr},
    {"iterations", "int", false, "", "2", nullptr},
    {"strength", "float", false, "", "0.5", nullptr},
};
static const MethodDescriptor desc_hair_smooth = {
    "hair.smooth", "hair",
    "Smooth hair strand shapes",
    nullptr,
    "write", "SceneWrite", false, "any",
    "hair|smooth|style",
    nullptr,
    params_hair_smooth, 3,
    true
};
static const MethodRegistration reg_hair_smooth(desc_hair_smooth);

static const MethodParam params_hair_trim[] = {
    {"length_factor", "float", true, "", nullptr, nullptr},
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_hair_trim = {
    "hair.trim", "hair",
    "Trim hair strands",
    nullptr,
    "write", "SceneWrite", false, "any",
    "hair|trim|length|style",
    nullptr,
    params_hair_trim, 2,
    true
};
static const MethodRegistration reg_hair_trim(desc_hair_trim);

static const MethodParam params_hair_update[] = {
    {"name", "string", true, "", nullptr, nullptr},
    {"visible", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_hair_update = {
    "hair.update", "hair",
    "Update hair groom settings, keeping what you do not send",
    nullptr,
    "write", "SceneWrite", false, "any",
    "hair|update|configure",
    nullptr,
    params_hair_update, 2,
    true
};
static const MethodRegistration reg_hair_update(desc_hair_update);

static const MethodDescriptor desc_ipc_admin_audit_clear = {
    "ipc.admin.audit.clear", "ipc",
    "Clear the IPC audit log",
    nullptr,
    "admin", "Admin", false, "any",
    "ipc|admin|audit|clear|security",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_ipc_admin_audit_clear(desc_ipc_admin_audit_clear);

static const MethodParam params_ipc_admin_audit_export[] = {
    {"filepath", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_ipc_admin_audit_export = {
    "ipc.admin.audit.export", "ipc",
    "Export the IPC audit log to a JSONL file",
    nullptr,
    "admin", "Admin|FilesWrite", false, "any",
    "ipc|admin|audit|export|security",
    nullptr,
    params_ipc_admin_audit_export, 1,
    true
};
static const MethodRegistration reg_ipc_admin_audit_export(desc_ipc_admin_audit_export);

static const MethodParam params_ipc_admin_audit_list[] = {
    {"maximum", "int", false, "", "256", nullptr},
};
static const MethodDescriptor desc_ipc_admin_audit_list = {
    "ipc.admin.audit.list", "ipc",
    "Return recent IPC audit events: method, caller, outcome and duration",
    nullptr,
    "admin", "Admin", false, "any",
    "ipc|admin|audit|list|security|log",
    nullptr,
    params_ipc_admin_audit_list, 1,
    true
};
static const MethodRegistration reg_ipc_admin_audit_list(desc_ipc_admin_audit_list);

static const MethodDescriptor desc_ipc_admin_sessions_disconnect = {
    "ipc.admin.sessions.disconnect", "ipc",
    "Disconnect one IPC session",
    nullptr,
    "admin", "Admin", false, "any",
    "ipc|admin|sessions|disconnect|security|session",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_ipc_admin_sessions_disconnect(desc_ipc_admin_sessions_disconnect);

static const MethodDescriptor desc_ipc_admin_sessions_disconnect_all = {
    "ipc.admin.sessions.disconnect_all", "ipc",
    "Disconnect every IPC session except the caller",
    nullptr,
    "admin", "Admin", false, "any",
    "ipc|admin|sessions|disconnect|all|security|session",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_ipc_admin_sessions_disconnect_all(desc_ipc_admin_sessions_disconnect_all);

static const MethodParam params_ipc_admin_sessions_get[] = {
    {"connection_id", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_ipc_admin_sessions_get = {
    "ipc.admin.sessions.get", "ipc",
    "Return one IPC session's details",
    nullptr,
    "admin", "Admin", false, "any",
    "ipc|admin|sessions|get|security|session",
    nullptr,
    params_ipc_admin_sessions_get, 1,
    true
};
static const MethodRegistration reg_ipc_admin_sessions_get(desc_ipc_admin_sessions_get);

static const MethodParam params_ipc_admin_sessions_list[] = {
    {"include_closed", "bool", false, "", "false", nullptr},
};
static const MethodDescriptor desc_ipc_admin_sessions_list = {
    "ipc.admin.sessions.list", "ipc",
    "List the connected IPC sessions with traffic counters",
    nullptr,
    "admin", "Admin", false, "any",
    "ipc|admin|sessions|list|security|session",
    nullptr,
    params_ipc_admin_sessions_list, 1,
    true
};
static const MethodRegistration reg_ipc_admin_sessions_list(desc_ipc_admin_sessions_list);

static const MethodParam params_ipc_admin_tokens_create[] = {
    {"capabilities", "int", true, "", nullptr, nullptr},
    {"display_name", "string", true, "", nullptr, nullptr},
    {"allowed_cidrs", "any", false, "", nullptr, nullptr},
    {"expires_at", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_ipc_admin_tokens_create = {
    "ipc.admin.tokens.create", "ipc",
    "Create a remote IPC token with a capability mask and return the raw secret once",
    nullptr,
    "admin", "Admin", false, "any",
    "ipc|admin|tokens|create|security|token",
    nullptr,
    params_ipc_admin_tokens_create, 4,
    true
};
static const MethodRegistration reg_ipc_admin_tokens_create(desc_ipc_admin_tokens_create);

static const MethodDescriptor desc_ipc_admin_tokens_list = {
    "ipc.admin.tokens.list", "ipc",
    "List the remote IPC access tokens and their capabilities",
    nullptr,
    "admin", "Admin", false, "any",
    "ipc|admin|tokens|list|security|token",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_ipc_admin_tokens_list(desc_ipc_admin_tokens_list);

static const MethodParam params_ipc_admin_tokens_revoke[] = {
    {"token_id", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_ipc_admin_tokens_revoke = {
    "ipc.admin.tokens.revoke", "ipc",
    "Revoke a remote IPC token",
    nullptr,
    "admin", "Admin", false, "any",
    "ipc|admin|tokens|revoke|security|token",
    nullptr,
    params_ipc_admin_tokens_revoke, 1,
    true
};
static const MethodRegistration reg_ipc_admin_tokens_revoke(desc_ipc_admin_tokens_revoke);

static const MethodParam params_ipc_admin_tokens_rotate[] = {
    {"token_id", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_ipc_admin_tokens_rotate = {
    "ipc.admin.tokens.rotate", "ipc",
    "Rotate a token's secret and return the new one once",
    nullptr,
    "admin", "Admin", false, "any",
    "ipc|admin|tokens|rotate|security|token",
    nullptr,
    params_ipc_admin_tokens_rotate, 1,
    true
};
static const MethodRegistration reg_ipc_admin_tokens_rotate(desc_ipc_admin_tokens_rotate);

static const MethodParam params_ipc_admin_tokens_update[] = {
    {"capabilities", "int", true, "", nullptr, nullptr},
    {"token_id", "string", true, "", nullptr, nullptr},
    {"allowed_cidrs", "any", false, "", nullptr, nullptr},
    {"expires_at", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_ipc_admin_tokens_update = {
    "ipc.admin.tokens.update", "ipc",
    "Update a token's name, capabilities, expiry or address allowlist",
    nullptr,
    "admin", "Admin", false, "any",
    "ipc|admin|tokens|update|security|token",
    nullptr,
    params_ipc_admin_tokens_update, 4,
    true
};
static const MethodRegistration reg_ipc_admin_tokens_update(desc_ipc_admin_tokens_update);

static const MethodParam params_lights_add[] = {
    {"type", "string", true, "Light type", nullptr, "point|directional|spot|area"},
    {"position", "vec3", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_lights_add = {
    "lights.add", "lights",
    "Add a light of the given type at a world position and return its name",
    nullptr,
    "write", "SceneWrite", true, "any",
    "lights|add|lighting|create|illuminate",
    "lights.set_intensity|lights.set_color|world.set_mode",
    params_lights_add, 2,
    true
};
static const MethodRegistration reg_lights_add(desc_lights_add);

static const MethodParam params_lights_delete[] = {
    {"index", "int", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_lights_delete = {
    "lights.delete", "lights",
    "Delete a light by index",
    nullptr,
    "write", "SceneWrite", true, "any",
    "lights|delete|lighting|remove",
    nullptr,
    params_lights_delete, 1,
    true
};
static const MethodRegistration reg_lights_delete(desc_lights_delete);

static const MethodParam params_lights_get[] = {
    {"index", "int", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_lights_get = {
    "lights.get", "lights",
    "Return every property of one light",
    nullptr,
    "read", "Read", false, "LightInfo",
    "lights|get|lighting",
    nullptr,
    params_lights_get, 1,
    true
};
static const MethodRegistration reg_lights_get(desc_lights_get);

static const MethodDescriptor desc_lights_list = {
    "lights.list", "lights",
    "List the scene lights with index, name, type and position",
    nullptr,
    "read", "Read", false, "LightSummary[]",
    "lights|list|lighting",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_lights_list(desc_lights_list);

static const MethodParam params_lights_rename[] = {
    {"index", "int", true, "", nullptr, nullptr},
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_lights_rename = {
    "lights.rename", "lights",
    "Rename a light",
    nullptr,
    "write", "SceneWrite", false, "any",
    "lights|rename|lighting",
    nullptr,
    params_lights_rename, 2,
    true
};
static const MethodRegistration reg_lights_rename(desc_lights_rename);

static const MethodParam params_lights_set_color[] = {
    {"color", "vec3", true, "", nullptr, nullptr},
    {"index", "int", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_lights_set_color = {
    "lights.set_color", "lights",
    "Set a light's RGB colour",
    nullptr,
    "write", "SceneWrite", false, "any",
    "lights|set|color|lighting|colour",
    nullptr,
    params_lights_set_color, 2,
    true
};
static const MethodRegistration reg_lights_set_color(desc_lights_set_color);

static const MethodParam params_lights_set_direction[] = {
    {"direction", "vec3", true, "", nullptr, nullptr},
    {"index", "int", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_lights_set_direction = {
    "lights.set_direction", "lights",
    "Set the direction a directional or spot light points in",
    nullptr,
    "write", "SceneWrite", false, "any",
    "lights|set|direction|lighting|aim",
    nullptr,
    params_lights_set_direction, 2,
    true
};
static const MethodRegistration reg_lights_set_direction(desc_lights_set_direction);

static const MethodParam params_lights_set_intensity[] = {
    {"index", "int", true, "", nullptr, nullptr},
    {"intensity", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_lights_set_intensity = {
    "lights.set_intensity", "lights",
    "Set a light's intensity",
    nullptr,
    "write", "SceneWrite", false, "any",
    "lights|set|intensity|lighting|brightness|exposure",
    nullptr,
    params_lights_set_intensity, 2,
    true
};
static const MethodRegistration reg_lights_set_intensity(desc_lights_set_intensity);

static const MethodParam params_lights_set_param[] = {
    {"index", "int", true, "", nullptr, nullptr},
    {"param", "string", true, "", nullptr, nullptr},
    {"value", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_lights_set_param = {
    "lights.set_param", "lights",
    "Set one numeric light parameter by name (radius, spot_angle, spot_falloff, width, height)",
    "spot_angle and spot_falloff apply to spot lights only; width and height to area lights only. The call fails rather than silently ignoring a mismatch.",
    "write", "SceneWrite", false, "any",
    "lights|set|param|lighting|softness|cone",
    nullptr,
    params_lights_set_param, 3,
    true
};
static const MethodRegistration reg_lights_set_param(desc_lights_set_param);

static const MethodParam params_lights_set_position[] = {
    {"index", "int", true, "", nullptr, nullptr},
    {"position", "vec3", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_lights_set_position = {
    "lights.set_position", "lights",
    "Move a light to a world position",
    nullptr,
    "write", "SceneWrite", false, "any",
    "lights|set|position|lighting|move",
    nullptr,
    params_lights_set_position, 2,
    true
};
static const MethodRegistration reg_lights_set_position(desc_lights_set_position);

static const MethodParam params_lights_set_visible[] = {
    {"index", "int", true, "", nullptr, nullptr},
    {"visible", "bool", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_lights_set_visible = {
    "lights.set_visible", "lights",
    "Show or hide a light without deleting it",
    nullptr,
    "write", "SceneWrite", false, "any",
    "lights|set|visible|lighting|visibility",
    nullptr,
    params_lights_set_visible, 2,
    true
};
static const MethodRegistration reg_lights_set_visible(desc_lights_set_visible);

static const MethodParam params_material_assign[] = {
    {"material_name", "string", true, "", nullptr, nullptr},
    {"object_name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_material_assign = {
    "material.assign", "material",
    "Assign an existing material to an object",
    nullptr,
    "write", "SceneWrite", false, "any",
    "material|assign|shading|apply",
    nullptr,
    params_material_assign, 2,
    true
};
static const MethodRegistration reg_material_assign(desc_material_assign);

static const MethodParam params_material_clear_texture[] = {
    {"material_name", "string", true, "", nullptr, nullptr},
    {"slot", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_material_clear_texture = {
    "material.clear_texture", "material",
    "Clear one material texture slot",
    nullptr,
    "write", "SceneWrite", false, "any",
    "material|clear|texture|shading",
    nullptr,
    params_material_clear_texture, 2,
    true
};
static const MethodRegistration reg_material_clear_texture(desc_material_clear_texture);

static const MethodParam params_material_create[] = {
    {"type", "string", true, "Material type, e.g. 'principled'", nullptr, nullptr},
    {"name", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_material_create = {
    "material.create", "material",
    "Create a material of the given type and return its name",
    nullptr,
    "write", "SceneWrite", false, "any",
    "material|create|shading",
    nullptr,
    params_material_create, 2,
    true
};
static const MethodRegistration reg_material_create(desc_material_create);

static const MethodParam params_material_get[] = {
    {"object_name", "string", true, "", nullptr, nullptr},
    {"param", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_material_get = {
    "material.get", "material",
    "Read one material parameter of an object",
    nullptr,
    "read", "Read", false, "any",
    "material|get|shading",
    nullptr,
    params_material_get, 2,
    true
};
static const MethodRegistration reg_material_get(desc_material_get);

static const MethodParam params_material_info[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_material_info = {
    "material.info", "material",
    "Return a material's full parameter set",
    nullptr,
    "read", "Read", false, "MaterialInfo",
    "material|info|shading",
    nullptr,
    params_material_info, 1,
    true
};
static const MethodRegistration reg_material_info(desc_material_info);

static const MethodDescriptor desc_material_list = {
    "material.list", "material",
    "List every material in the scene",
    nullptr,
    "read", "Read", false, "string[]",
    "material|list|shading",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_material_list(desc_material_list);

static const MethodParam params_material_of_object[] = {
    {"object_name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_material_of_object = {
    "material.of_object", "material",
    "List the materials used by one object",
    nullptr,
    "read", "Read", false, "string[]",
    "material|of|object|shading",
    nullptr,
    params_material_of_object, 1,
    true
};
static const MethodRegistration reg_material_of_object(desc_material_of_object);

static const MethodParam params_material_set[] = {
    {"object_name", "string", true, "", nullptr, nullptr},
    {"param", "string", true, "", nullptr, nullptr},
    {"value", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_material_set = {
    "material.set", "material",
    "Set one material parameter on an object (base_color, roughness, metallic, emission, ior, transmission, ...)",
    "Colour parameters take a 3-element array, scalars take a number. An unknown parameter name is an error, not a silent no-op.",
    "write", "SceneWrite", false, "any",
    "material|set|shading|colour|roughness|metallic|emission",
    "material.get|material.info",
    params_material_set, 3,
    true
};
static const MethodRegistration reg_material_set(desc_material_set);

static const MethodParam params_material_set_texture[] = {
    {"material_name", "string", true, "", nullptr, nullptr},
    {"path", "string", true, "", nullptr, nullptr},
    {"slot", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_material_set_texture = {
    "material.set_texture", "material",
    "Bind an image file to a material texture slot",
    nullptr,
    "write", "SceneWrite", false, "any",
    "material|set|texture|shading|image",
    nullptr,
    params_material_set_texture, 3,
    true
};
static const MethodRegistration reg_material_set_texture(desc_material_set_texture);

static const MethodParam params_material_textures[] = {
    {"material_name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_material_textures = {
    "material.textures", "material",
    "List a material's texture slots and their bound files",
    nullptr,
    "read", "Read", false, "any",
    "material|textures|shading|texture",
    nullptr,
    params_material_textures, 1,
    true
};
static const MethodRegistration reg_material_textures(desc_material_textures);

static const MethodParam params_modifiers_add[] = {
    {"object", "string", true, "", nullptr, nullptr},
    {"levels", "int", false, "", "1", nullptr},
    {"name", "string", false, "", "", nullptr},
    {"render_levels", "int", false, "", "2", nullptr},
    {"type", "string", false, "", "catmull_clark", nullptr},
};
static const MethodDescriptor desc_modifiers_add = {
    "modifiers.add", "modifiers",
    "Add a modifier to an object and return the new stack entry",
    nullptr,
    "write", "SceneWrite", false, "any",
    "modifiers|add|mesh|modifier|subdivide|create",
    nullptr,
    params_modifiers_add, 5,
    true
};
static const MethodRegistration reg_modifiers_add(desc_modifiers_add);

static const MethodParam params_modifiers_apply[] = {
    {"object", "string", true, "", nullptr, nullptr},
    {"index", "int", false, "", "0", nullptr},
};
static const MethodDescriptor desc_modifiers_apply = {
    "modifiers.apply", "modifiers",
    "Apply a modifier destructively into the mesh",
    nullptr,
    "write", "SceneWrite", false, "any",
    "modifiers|apply|mesh|modifier|bake",
    nullptr,
    params_modifiers_apply, 2,
    true
};
static const MethodRegistration reg_modifiers_apply(desc_modifiers_apply);

static const MethodParam params_modifiers_get_stack[] = {
    {"object", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_modifiers_get_stack = {
    "modifiers.get_stack", "modifiers",
    "List an object's modifier stack",
    nullptr,
    "read", "Read", false, "any",
    "modifiers|get|stack|mesh|modifier",
    nullptr,
    params_modifiers_get_stack, 1,
    true
};
static const MethodRegistration reg_modifiers_get_stack(desc_modifiers_get_stack);

static const MethodParam params_modifiers_remove[] = {
    {"object", "string", true, "", nullptr, nullptr},
    {"index", "int", false, "", "0", nullptr},
};
static const MethodDescriptor desc_modifiers_remove = {
    "modifiers.remove", "modifiers",
    "Remove a modifier from an object",
    nullptr,
    "write", "SceneWrite", false, "any",
    "modifiers|remove|mesh|modifier",
    nullptr,
    params_modifiers_remove, 2,
    true
};
static const MethodRegistration reg_modifiers_remove(desc_modifiers_remove);

static const MethodParam params_modifiers_set_param[] = {
    {"object", "string", true, "", nullptr, nullptr},
    {"enabled", "any", false, "", nullptr, nullptr},
    {"index", "int", false, "", "0", nullptr},
    {"levels", "any", false, "", nullptr, nullptr},
    {"name", "string", false, "", "", nullptr},
    {"render_levels", "any", false, "", nullptr, nullptr},
    {"smooth_angle", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_modifiers_set_param = {
    "modifiers.set_param", "modifiers",
    "Set a modifier parameter: enabled, levels, render_levels or smooth_angle",
    nullptr,
    "write", "SceneWrite", false, "any",
    "modifiers|set|param|mesh|modifier|configure",
    nullptr,
    params_modifiers_set_param, 7,
    true
};
static const MethodRegistration reg_modifiers_set_param(desc_modifiers_set_param);

static const MethodDescriptor desc_msf_fields = {
    "msf.fields", "msf",
    "Report the live material state fields (temperature, moisture, char, mass) per object",
    "This is the measurement side of burning and melting: mass loss, integrity and mass-conservation error are reported here.",
    "read", "Read", false, "any",
    "msf|fields|substance|thermal|burn|melt|measure|verify|temperature|moisture",
    "msf.substances",
    nullptr, 0,
    true
};
static const MethodRegistration reg_msf_fields(desc_msf_fields);

static const MethodDescriptor desc_msf_substances = {
    "msf.substances", "msf",
    "List the material substance library: every substance an object or fluid can be made of",
    "Substance ids from here are what fluid.set_substance_material and the combustion path expect.",
    "read", "Read", false, "any",
    "msf|substances|substance|material|library|wood|water|metal|thermochemistry",
    "fluid.set_substance_material|msf.fields",
    nullptr, 0,
    true
};
static const MethodRegistration reg_msf_substances(desc_msf_substances);

static const MethodParam params_nodes_add[] = {
    {"graph_name", "string", true, "", nullptr, nullptr},
    {"graph_type", "string", true, "", nullptr, nullptr},
    {"type_id", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_nodes_add = {
    "nodes.add", "nodes",
    "Add a node of the given type to a graph",
    nullptr,
    "write", "SceneWrite", false, "any",
    "nodes|add|graph|create",
    nullptr,
    params_nodes_add, 3,
    true
};
static const MethodRegistration reg_nodes_add(desc_nodes_add);

static const MethodParam params_nodes_apply[] = {
    {"graph_name", "string", true, "", nullptr, nullptr},
    {"graph_type", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_nodes_apply = {
    "nodes.apply", "nodes",
    "Evaluate a node graph and apply its result, reporting errors and warnings",
    nullptr,
    "write", "SceneWrite", false, "any",
    "nodes|apply|graph|evaluate|bake",
    nullptr,
    params_nodes_apply, 2,
    true
};
static const MethodRegistration reg_nodes_apply(desc_nodes_apply);

static const MethodParam params_nodes_create_graph[] = {
    {"graph_name", "string", true, "", nullptr, nullptr},
    {"graph_type", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_nodes_create_graph = {
    "nodes.create_graph", "nodes",
    "Create a node graph in a domain",
    nullptr,
    "write", "SceneWrite", false, "any",
    "nodes|create|graph",
    nullptr,
    params_nodes_create_graph, 2,
    true
};
static const MethodRegistration reg_nodes_create_graph(desc_nodes_create_graph);

static const MethodParam params_nodes_get_param[] = {
    {"graph_name", "string", true, "", nullptr, nullptr},
    {"graph_type", "string", true, "", nullptr, nullptr},
    {"pin_index", "int", true, "", nullptr, nullptr},
    {"node_id", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_nodes_get_param = {
    "nodes.get_param", "nodes",
    "Read one node parameter",
    nullptr,
    "read", "Read", false, "any",
    "nodes|get|param|graph",
    nullptr,
    params_nodes_get_param, 4,
    true
};
static const MethodRegistration reg_nodes_get_param(desc_nodes_get_param);

static const MethodParam params_nodes_get_property[] = {
    {"graph_name", "string", true, "", nullptr, nullptr},
    {"graph_type", "string", true, "", nullptr, nullptr},
    {"property", "string", true, "", nullptr, nullptr},
    {"node_id", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_nodes_get_property = {
    "nodes.get_property", "nodes",
    "Read one node property",
    nullptr,
    "read", "Read", false, "any",
    "nodes|get|property|graph",
    nullptr,
    params_nodes_get_property, 4,
    true
};
static const MethodRegistration reg_nodes_get_property(desc_nodes_get_property);

static const MethodParam params_nodes_graphs[] = {
    {"graph_type", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_nodes_graphs = {
    "nodes.graphs", "nodes",
    "List the node graphs that exist",
    nullptr,
    "read", "Read", false, "any",
    "nodes|graphs|graph|inventory",
    nullptr,
    params_nodes_graphs, 1,
    true
};
static const MethodRegistration reg_nodes_graphs(desc_nodes_graphs);

static const MethodParam params_nodes_link[] = {
    {"from_output", "int", true, "", nullptr, nullptr},
    {"graph_name", "string", true, "", nullptr, nullptr},
    {"graph_type", "string", true, "", nullptr, nullptr},
    {"to_input", "int", true, "", nullptr, nullptr},
    {"from_node", "any", false, "", nullptr, nullptr},
    {"to_node", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_nodes_link = {
    "nodes.link", "nodes",
    "Connect one node's output to another node's input",
    nullptr,
    "write", "SceneWrite", false, "any",
    "nodes|link|graph|connect|wire",
    nullptr,
    params_nodes_link, 6,
    true
};
static const MethodRegistration reg_nodes_link(desc_nodes_link);

static const MethodParam params_nodes_list[] = {
    {"graph_name", "string", true, "", nullptr, nullptr},
    {"graph_type", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_nodes_list = {
    "nodes.list", "nodes",
    "List the nodes of a graph with their inputs and outputs",
    nullptr,
    "read", "Read", false, "any",
    "nodes|list|graph",
    nullptr,
    params_nodes_list, 2,
    true
};
static const MethodRegistration reg_nodes_list(desc_nodes_list);

static const MethodParam params_nodes_list_params[] = {
    {"graph_name", "string", true, "", nullptr, nullptr},
    {"graph_type", "string", true, "", nullptr, nullptr},
    {"node_id", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_nodes_list_params = {
    "nodes.list_params", "nodes",
    "List a node's parameters, their types, values and whether they are driven by a link",
    nullptr,
    "read", "Read", false, "any",
    "nodes|list|params|graph",
    nullptr,
    params_nodes_list_params, 3,
    true
};
static const MethodRegistration reg_nodes_list_params(desc_nodes_list_params);

static const MethodParam params_nodes_list_properties[] = {
    {"graph_name", "string", true, "", nullptr, nullptr},
    {"graph_type", "string", true, "", nullptr, nullptr},
    {"node_id", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_nodes_list_properties = {
    "nodes.list_properties", "nodes",
    "List a node's non-parameter properties",
    nullptr,
    "read", "Read", false, "any",
    "nodes|list|properties|graph",
    nullptr,
    params_nodes_list_properties, 3,
    true
};
static const MethodRegistration reg_nodes_list_properties(desc_nodes_list_properties);

static const MethodParam params_nodes_remove[] = {
    {"graph_name", "string", true, "", nullptr, nullptr},
    {"graph_type", "string", true, "", nullptr, nullptr},
    {"node_id", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_nodes_remove = {
    "nodes.remove", "nodes",
    "Remove a node from a graph",
    nullptr,
    "write", "SceneWrite", false, "any",
    "nodes|remove|graph",
    nullptr,
    params_nodes_remove, 3,
    true
};
static const MethodRegistration reg_nodes_remove(desc_nodes_remove);

static const MethodParam params_nodes_remove_graph[] = {
    {"graph_name", "string", true, "", nullptr, nullptr},
    {"graph_type", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_nodes_remove_graph = {
    "nodes.remove_graph", "nodes",
    "Delete a node graph",
    nullptr,
    "write", "SceneWrite", false, "any",
    "nodes|remove|graph",
    nullptr,
    params_nodes_remove_graph, 2,
    true
};
static const MethodRegistration reg_nodes_remove_graph(desc_nodes_remove_graph);

static const MethodParam params_nodes_set_param[] = {
    {"graph_name", "string", true, "", nullptr, nullptr},
    {"graph_type", "string", true, "", nullptr, nullptr},
    {"pin_index", "int", true, "", nullptr, nullptr},
    {"node_id", "any", false, "", nullptr, nullptr},
    {"value", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_nodes_set_param = {
    "nodes.set_param", "nodes",
    "Set one node parameter",
    nullptr,
    "write", "SceneWrite", false, "any",
    "nodes|set|param|graph|configure",
    nullptr,
    params_nodes_set_param, 5,
    true
};
static const MethodRegistration reg_nodes_set_param(desc_nodes_set_param);

static const MethodParam params_nodes_set_property[] = {
    {"graph_name", "string", true, "", nullptr, nullptr},
    {"graph_type", "string", true, "", nullptr, nullptr},
    {"property", "string", true, "", nullptr, nullptr},
    {"node_id", "any", false, "", nullptr, nullptr},
    {"value", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_nodes_set_property = {
    "nodes.set_property", "nodes",
    "Set one node property",
    nullptr,
    "write", "SceneWrite", false, "any",
    "nodes|set|property|graph|configure",
    nullptr,
    params_nodes_set_property, 5,
    true
};
static const MethodRegistration reg_nodes_set_property(desc_nodes_set_property);

static const MethodDescriptor desc_nodes_types = {
    "nodes.types", "nodes",
    "List every node type with its category and description",
    nullptr,
    "read", "Read", false, "any",
    "nodes|types|catalogue",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_nodes_types(desc_nodes_types);

static const MethodParam params_paint_add_layer[] = {
    {"object", "string", true, "", nullptr, nullptr},
    {"insert_at", "int", false, "", "-1", nullptr},
    {"material_id", "int", false, "", "-1", nullptr},
    {"name", "string", false, "", "Paint Layer", nullptr},
};
static const MethodDescriptor desc_paint_add_layer = {
    "paint.add_layer", "paint",
    "Add a paint layer to an object's texture set",
    nullptr,
    "write", "SceneWrite", false, "any",
    "paint|add|layer|texture",
    nullptr,
    params_paint_add_layer, 4,
    true
};
static const MethodRegistration reg_paint_add_layer(desc_paint_add_layer);

static const MethodParam params_paint_apply_mask[] = {
    {"layer_index", "int", true, "", nullptr, nullptr},
    {"object", "string", true, "", nullptr, nullptr},
    {"preset", "string", true, "", nullptr, nullptr},
    {"material_id", "int", false, "", "-1", nullptr},
    {"seed", "int", false, "", "1337", nullptr},
    {"strength", "float", false, "", "1.0", nullptr},
};
static const MethodDescriptor desc_paint_apply_mask = {
    "paint.apply_mask", "paint",
    "Apply a mask preset to a paint layer",
    nullptr,
    "write", "SceneWrite", false, "any",
    "paint|apply|mask|texture|wear|dirt",
    nullptr,
    params_paint_apply_mask, 6,
    true
};
static const MethodRegistration reg_paint_apply_mask(desc_paint_apply_mask);

static const MethodParam params_paint_bake_height_to_normal[] = {
    {"object", "string", true, "", nullptr, nullptr},
    {"clear_height", "bool", false, "", "false", nullptr},
    {"material_id", "int", false, "", "-1", nullptr},
    {"strength", "float", false, "", "4.0", nullptr},
};
static const MethodDescriptor desc_paint_bake_height_to_normal = {
    "paint.bake_height_to_normal", "paint",
    "Bake a paint height channel into the normal channel",
    nullptr,
    "write", "SceneWrite", false, "any",
    "paint|bake|height|to|normal|texture",
    nullptr,
    params_paint_bake_height_to_normal, 4,
    true
};
static const MethodRegistration reg_paint_bake_height_to_normal(desc_paint_bake_height_to_normal);

static const MethodParam params_paint_clear_channel[] = {
    {"channel", "string", true, "", nullptr, nullptr},
    {"layer_index", "int", true, "", nullptr, nullptr},
    {"object", "string", true, "", nullptr, nullptr},
    {"material_id", "int", false, "", "-1", nullptr},
};
static const MethodDescriptor desc_paint_clear_channel = {
    "paint.clear_channel", "paint",
    "Clear one channel of a paint layer",
    nullptr,
    "write", "SceneWrite", false, "any",
    "paint|clear|channel|texture",
    nullptr,
    params_paint_clear_channel, 4,
    true
};
static const MethodRegistration reg_paint_clear_channel(desc_paint_clear_channel);

static const MethodParam params_paint_duplicate_layer[] = {
    {"layer_index", "int", true, "", nullptr, nullptr},
    {"object", "string", true, "", nullptr, nullptr},
    {"material_id", "int", false, "", "-1", nullptr},
};
static const MethodDescriptor desc_paint_duplicate_layer = {
    "paint.duplicate_layer", "paint",
    "Duplicate a paint layer",
    nullptr,
    "write", "SceneWrite", false, "any",
    "paint|duplicate|layer|texture",
    nullptr,
    params_paint_duplicate_layer, 3,
    true
};
static const MethodRegistration reg_paint_duplicate_layer(desc_paint_duplicate_layer);

static const MethodParam params_paint_ensure[] = {
    {"object", "string", true, "", nullptr, nullptr},
    {"material_id", "int", false, "", "-1", nullptr},
    {"resolution", "int", false, "", "1024", nullptr},
};
static const MethodDescriptor desc_paint_ensure = {
    "paint.ensure", "paint",
    "Create or return the paint target (texture set) for an object",
    nullptr,
    "write", "SceneWrite", false, "any",
    "paint|ensure|texture|setup",
    nullptr,
    params_paint_ensure, 3,
    true
};
static const MethodRegistration reg_paint_ensure(desc_paint_ensure);

static const MethodParam params_paint_export_channel[] = {
    {"channel", "string", true, "", nullptr, nullptr},
    {"filepath", "string", true, "", nullptr, nullptr},
    {"object", "string", true, "", nullptr, nullptr},
    {"layer_index", "int", false, "", "-1", nullptr},
    {"material_id", "int", false, "", "-1", nullptr},
};
static const MethodDescriptor desc_paint_export_channel = {
    "paint.export_channel", "paint",
    "Export a paint layer channel to an image file",
    nullptr,
    "write", "FilesWrite", false, "any",
    "paint|export|channel|texture",
    nullptr,
    params_paint_export_channel, 5,
    true
};
static const MethodRegistration reg_paint_export_channel(desc_paint_export_channel);

static const MethodParam params_paint_fill[] = {
    {"channel", "string", true, "", nullptr, nullptr},
    {"color", "vec3", true, "", nullptr, nullptr},
    {"layer_index", "int", true, "", nullptr, nullptr},
    {"object", "string", true, "", nullptr, nullptr},
    {"material_id", "int", false, "", "-1", nullptr},
};
static const MethodDescriptor desc_paint_fill = {
    "paint.fill", "paint",
    "Fill a paint layer channel with a value or colour",
    nullptr,
    "write", "SceneWrite", false, "any",
    "paint|fill|texture",
    nullptr,
    params_paint_fill, 5,
    true
};
static const MethodRegistration reg_paint_fill(desc_paint_fill);

static const MethodParam params_paint_flatten[] = {
    {"object", "string", true, "", nullptr, nullptr},
    {"material_id", "int", false, "", "-1", nullptr},
};
static const MethodDescriptor desc_paint_flatten = {
    "paint.flatten", "paint",
    "Flatten every paint layer into one",
    nullptr,
    "write", "SceneWrite", false, "any",
    "paint|flatten|texture|layer",
    nullptr,
    params_paint_flatten, 2,
    true
};
static const MethodRegistration reg_paint_flatten(desc_paint_flatten);

static const MethodParam params_paint_get[] = {
    {"object", "string", true, "", nullptr, nullptr},
    {"material_id", "int", false, "", "-1", nullptr},
};
static const MethodDescriptor desc_paint_get = {
    "paint.get", "paint",
    "Return an object's paint target: layers, channels and resolution",
    nullptr,
    "read", "Read", false, "any",
    "paint|get|texture",
    nullptr,
    params_paint_get, 2,
    true
};
static const MethodRegistration reg_paint_get(desc_paint_get);

static const MethodParam params_paint_import_channel[] = {
    {"channel", "string", true, "", nullptr, nullptr},
    {"filepath", "string", true, "", nullptr, nullptr},
    {"layer_index", "int", true, "", nullptr, nullptr},
    {"object", "string", true, "", nullptr, nullptr},
    {"material_id", "int", false, "", "-1", nullptr},
};
static const MethodDescriptor desc_paint_import_channel = {
    "paint.import_channel", "paint",
    "Import an image into a paint layer channel",
    nullptr,
    "write", "FilesRead|SceneWrite", false, "any",
    "paint|import|channel|texture",
    nullptr,
    params_paint_import_channel, 5,
    true
};
static const MethodRegistration reg_paint_import_channel(desc_paint_import_channel);

static const MethodDescriptor desc_paint_list_mask_presets = {
    "paint.list_mask_presets", "paint",
    "List the available paint mask presets",
    nullptr,
    "read", "Read", false, "any",
    "paint|list|mask|presets|texture",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_paint_list_mask_presets(desc_paint_list_mask_presets);

static const MethodParam params_paint_merge_down[] = {
    {"layer_index", "int", true, "", nullptr, nullptr},
    {"object", "string", true, "", nullptr, nullptr},
    {"material_id", "int", false, "", "-1", nullptr},
};
static const MethodDescriptor desc_paint_merge_down = {
    "paint.merge_down", "paint",
    "Merge a paint layer into the one below it",
    nullptr,
    "write", "SceneWrite", false, "any",
    "paint|merge|down|texture|layer",
    nullptr,
    params_paint_merge_down, 3,
    true
};
static const MethodRegistration reg_paint_merge_down(desc_paint_merge_down);

static const MethodParam params_paint_move_layer[] = {
    {"from_index", "int", true, "", nullptr, nullptr},
    {"object", "string", true, "", nullptr, nullptr},
    {"to_index", "int", true, "", nullptr, nullptr},
    {"material_id", "int", false, "", "-1", nullptr},
};
static const MethodDescriptor desc_paint_move_layer = {
    "paint.move_layer", "paint",
    "Reorder a paint layer in the stack",
    nullptr,
    "write", "SceneWrite", false, "any",
    "paint|move|layer|texture",
    nullptr,
    params_paint_move_layer, 4,
    true
};
static const MethodRegistration reg_paint_move_layer(desc_paint_move_layer);

static const MethodParam params_paint_remove_layer[] = {
    {"layer_index", "int", true, "", nullptr, nullptr},
    {"object", "string", true, "", nullptr, nullptr},
    {"material_id", "int", false, "", "-1", nullptr},
};
static const MethodDescriptor desc_paint_remove_layer = {
    "paint.remove_layer", "paint",
    "Remove a paint layer",
    nullptr,
    "write", "SceneWrite", false, "any",
    "paint|remove|layer|texture",
    nullptr,
    params_paint_remove_layer, 3,
    true
};
static const MethodRegistration reg_paint_remove_layer(desc_paint_remove_layer);

static const MethodParam params_paint_update_layer[] = {
    {"layer_index", "int", true, "", nullptr, nullptr},
    {"object", "string", true, "", nullptr, nullptr},
    {"blend_mode", "any", false, "", nullptr, nullptr},
    {"locked", "any", false, "", nullptr, nullptr},
    {"material_id", "int", false, "", "-1", nullptr},
    {"name", "any", false, "", nullptr, nullptr},
    {"opacity", "any", false, "", nullptr, nullptr},
    {"visible", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_paint_update_layer = {
    "paint.update_layer", "paint",
    "Set a paint layer's name, opacity, blend mode, visibility or lock",
    nullptr,
    "write", "SceneWrite", false, "any",
    "paint|update|layer|texture",
    nullptr,
    params_paint_update_layer, 8,
    true
};
static const MethodRegistration reg_paint_update_layer(desc_paint_update_layer);

static const MethodParam params_particle_add_emitter[] = {
    {"angular_jitter", "float", false, "", nullptr, nullptr},
    {"angular_velocity", "float", false, "", nullptr, nullptr},
    {"burst_count", "any", false, "", nullptr, nullptr},
    {"direction", "vec3", false, "", nullptr, nullptr},
    {"enabled", "any", false, "", nullptr, nullptr},
    {"end_color", "vec3", false, "", nullptr, nullptr},
    {"end_opacity", "float", false, "", nullptr, nullptr},
    {"end_size", "float", false, "", nullptr, nullptr},
    {"lifetime_seconds", "float", false, "", nullptr, nullptr},
    {"local_offset", "vec3", false, "", nullptr, nullptr},
    {"mass", "float", false, "", nullptr, nullptr},
    {"name", "string", false, "", nullptr, nullptr},
    {"point", "vec3", false, "", nullptr, nullptr},
    {"rate_per_second", "float", false, "", nullptr, nullptr},
    {"seed", "any", false, "", nullptr, nullptr},
    {"size_jitter", "float", false, "", nullptr, nullptr},
    {"source_mode", "string", false, "", nullptr, nullptr},
    {"source_name", "string", false, "", nullptr, nullptr},
    {"spawn_mode", "string", false, "", nullptr, nullptr},
    {"speed", "float", false, "", nullptr, nullptr},
    {"spread", "float", false, "", nullptr, nullptr},
    {"start_color", "vec3", false, "", nullptr, nullptr},
    {"start_opacity", "float", false, "", nullptr, nullptr},
    {"start_size", "float", false, "", nullptr, nullptr},
    {"surface_offset", "float", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_particle_add_emitter = {
    "particle.add_emitter", "particle",
    "Add a particle emitter and return it",
    nullptr,
    "write", "SceneWrite", false, "any",
    "particle|add|emitter|particles|create|spawn|emit",
    nullptr,
    params_particle_add_emitter, 25,
    true
};
static const MethodRegistration reg_particle_add_emitter(desc_particle_add_emitter);

static const MethodDescriptor desc_particle_clear = {
    "particle.clear", "particle",
    "Delete the live particles without touching the emitters",
    nullptr,
    "write", "SceneWrite", false, "any",
    "particle|clear|particles|reset",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_particle_clear(desc_particle_clear);

static const MethodDescriptor desc_particle_clear_emitters = {
    "particle.clear_emitters", "particle",
    "Remove every particle emitter",
    nullptr,
    "write", "SceneWrite", false, "any",
    "particle|clear|emitters|particles|remove",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_particle_clear_emitters(desc_particle_clear_emitters);

static const MethodDescriptor desc_particle_clear_systems = {
    "particle.clear_systems", "particle",
    "Remove every particle system",
    nullptr,
    "write", "SceneWrite", false, "any",
    "particle|clear|systems|particles|remove",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_particle_clear_systems(desc_particle_clear_systems);

static const MethodDescriptor desc_particle_emitters = {
    "particle.emitters", "particle",
    "List the particle emitters",
    nullptr,
    "read", "Read", false, "any",
    "particle|emitters|particles|inventory",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_particle_emitters(desc_particle_emitters);

static const MethodParam params_particle_get_emitter[] = {
    {"emitter", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_particle_get_emitter = {
    "particle.get_emitter", "particle",
    "Return one particle emitter's full settings",
    nullptr,
    "read", "Read", false, "any",
    "particle|get|emitter|particles",
    nullptr,
    params_particle_get_emitter, 1,
    true
};
static const MethodRegistration reg_particle_get_emitter(desc_particle_get_emitter);

static const MethodDescriptor desc_particle_get_physics = {
    "particle.get_physics", "particle",
    "Read the particle solver settings",
    nullptr,
    "read", "Read", false, "any",
    "particle|get|physics|particles|solver",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_particle_get_physics(desc_particle_get_physics);

static const MethodDescriptor desc_particle_list_systems = {
    "particle.list_systems", "particle",
    "List the particle systems with their emitter, domain and collider counts",
    nullptr,
    "read", "Read", false, "any",
    "particle|list|systems|particles|inventory",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_particle_list_systems(desc_particle_list_systems);

static const MethodParam params_particle_remove_emitter[] = {
    {"emitter", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_particle_remove_emitter = {
    "particle.remove_emitter", "particle",
    "Remove a particle emitter",
    nullptr,
    "write", "SceneWrite", false, "any",
    "particle|remove|emitter|particles",
    nullptr,
    params_particle_remove_emitter, 1,
    true
};
static const MethodRegistration reg_particle_remove_emitter(desc_particle_remove_emitter);

static const MethodParam params_particle_set_emitter[] = {
    {"emitter", "string", true, "", nullptr, nullptr},
    {"angular_jitter", "float", false, "", nullptr, nullptr},
    {"angular_velocity", "float", false, "", nullptr, nullptr},
    {"burst_count", "any", false, "", nullptr, nullptr},
    {"direction", "vec3", false, "", nullptr, nullptr},
    {"enabled", "any", false, "", nullptr, nullptr},
    {"end_color", "vec3", false, "", nullptr, nullptr},
    {"end_opacity", "float", false, "", nullptr, nullptr},
    {"end_size", "float", false, "", nullptr, nullptr},
    {"lifetime_seconds", "float", false, "", nullptr, nullptr},
    {"local_offset", "vec3", false, "", nullptr, nullptr},
    {"mass", "float", false, "", nullptr, nullptr},
    {"name", "string", false, "", nullptr, nullptr},
    {"point", "vec3", false, "", nullptr, nullptr},
    {"rate_per_second", "float", false, "", nullptr, nullptr},
    {"seed", "any", false, "", nullptr, nullptr},
    {"size_jitter", "float", false, "", nullptr, nullptr},
    {"source_mode", "string", false, "", nullptr, nullptr},
    {"source_name", "string", false, "", nullptr, nullptr},
    {"spawn_mode", "string", false, "", nullptr, nullptr},
    {"speed", "float", false, "", nullptr, nullptr},
    {"spread", "float", false, "", nullptr, nullptr},
    {"start_color", "vec3", false, "", nullptr, nullptr},
    {"start_opacity", "float", false, "", nullptr, nullptr},
    {"start_size", "float", false, "", nullptr, nullptr},
    {"surface_offset", "float", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_particle_set_emitter = {
    "particle.set_emitter", "particle",
    "Update fields of a particle emitter, keeping what you do not send",
    nullptr,
    "write", "SceneWrite", false, "any",
    "particle|set|emitter|particles|configure",
    nullptr,
    params_particle_set_emitter, 26,
    true
};
static const MethodRegistration reg_particle_set_emitter(desc_particle_set_emitter);

static const MethodParam params_particle_set_physics[] = {
    {"buoyancy", "float", false, "", nullptr, nullptr},
    {"cohesion", "float", false, "", nullptr, nullptr},
    {"gravity_scale", "float", false, "", nullptr, nullptr},
    {"grid_density_deposit", "float", false, "", nullptr, nullptr},
    {"grid_deposit_fade_with_age", "bool", false, "", nullptr, nullptr},
    {"grid_fuel_deposit", "float", false, "", nullptr, nullptr},
    {"grid_temperature_deposit", "float", false, "", nullptr, nullptr},
    {"max_neighbors_per_particle", "int", false, "", nullptr, nullptr},
    {"mode", "string", false, "", nullptr, nullptr},
    {"particle_radius", "float", false, "", nullptr, nullptr},
    {"pressure_stiffness", "float", false, "", nullptr, nullptr},
    {"quality", "string", false, "", nullptr, nullptr},
    {"rest_density", "float", false, "", nullptr, nullptr},
    {"self_collision_enabled", "bool", false, "", nullptr, nullptr},
    {"solver_iterations", "int", false, "", nullptr, nullptr},
    {"viscosity", "float", false, "", nullptr, nullptr},
    {"vorticity", "float", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_particle_set_physics = {
    "particle.set_physics", "particle",
    "Set the particle solver: mode, quality, rest density, viscosity, cohesion, self-collision and grid deposit rates",
    nullptr,
    "write", "SceneWrite", false, "any",
    "particle|set|physics|particles|solver|configure|sph",
    nullptr,
    params_particle_set_physics, 17,
    true
};
static const MethodRegistration reg_particle_set_physics(desc_particle_set_physics);

static const MethodParam params_particle_spawn[] = {
    {"position", "vec3", true, "", nullptr, nullptr},
    {"lifetime_seconds", "float", false, "", "5.0", nullptr},
    {"mass", "float", false, "", "1.0", nullptr},
    {"size", "float", false, "", "0.05", nullptr},
    {"velocity", "vec3", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_particle_spawn = {
    "particle.spawn", "particle",
    "Spawn one particle with an explicit position, velocity, size, mass and lifetime",
    nullptr,
    "write", "SceneWrite", false, "any",
    "particle|spawn|particles|manual|inject",
    nullptr,
    params_particle_spawn, 5,
    true
};
static const MethodRegistration reg_particle_spawn(desc_particle_spawn);

static const MethodDescriptor desc_particle_stats = {
    "particle.stats", "particle",
    "Report live particle counts and per-stage solver timings",
    nullptr,
    "read", "Read", false, "any",
    "particle|stats|particles|measure|performance|verify",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_particle_stats(desc_particle_stats);

static const MethodParam params_particle_step[] = {
    {"dt", "float", false, "", "0.0166667", nullptr},
};
static const MethodDescriptor desc_particle_step = {
    "particle.step", "particle",
    "Advance the particle solver by one timestep",
    nullptr,
    "write", "SceneWrite", false, "any",
    "particle|step|particles|advance",
    nullptr,
    params_particle_step, 1,
    true
};
static const MethodRegistration reg_particle_step(desc_particle_step);

static const MethodParam params_physics_add_body[] = {
    {"object", "string", true, "Object name", nullptr, nullptr},
    {"kind", "string", false, "Body family, e.g. rigid", "rigid", nullptr},
    {"motion_type", "string", false, "dynamic, kinematic or static", "dynamic", nullptr},
    {"mass", "float", false, "Mass in kilograms", "1.0", nullptr},
    {"shape", "string", false, "Collision shape, e.g. box, sphere, mesh", "box", nullptr},
};
static const MethodDescriptor desc_physics_add_body = {
    "physics.add_body", "physics",
    "Give an object a rigid-body with a mass, motion type and collision shape",
    nullptr,
    "write", "SceneWrite", false, "any",
    "physics|add|body|simulation|rigid|collision|gravity|mass",
    "physics.set_gravity|physics.get_body|physics.step",
    params_physics_add_body, 5,
    true
};
static const MethodRegistration reg_physics_add_body(desc_physics_add_body);

static const MethodParam params_physics_apply_fracture_impulse[] = {
    {"group", "any", true, "", nullptr, nullptr},
    {"direction", "any", false, "", nullptr, nullptr},
    {"impulse", "float", false, "", "1.0", nullptr},
    {"point", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_physics_apply_fracture_impulse = {
    "physics.apply_fracture_impulse", "physics",
    "Apply an impulse at a point on a fracture group and report whether it broke",
    nullptr,
    "write", "SceneWrite", false, "any",
    "physics|apply|fracture|impulse|destruction|impact|blast",
    nullptr,
    params_physics_apply_fracture_impulse, 4,
    true
};
static const MethodRegistration reg_physics_apply_fracture_impulse(desc_physics_apply_fracture_impulse);

static const MethodParam params_physics_break_fracture_group[] = {
    {"group", "any", true, "", nullptr, nullptr},
    {"strength", "float", false, "", "6.0", nullptr},
};
static const MethodDescriptor desc_physics_break_fracture_group = {
    "physics.break_fracture_group", "physics",
    "Break a fracture group apart immediately with a given strength",
    nullptr,
    "write", "SceneWrite", false, "any",
    "physics|break|fracture|group|destruction|collapse",
    nullptr,
    params_physics_break_fracture_group, 2,
    true
};
static const MethodRegistration reg_physics_break_fracture_group(desc_physics_break_fracture_group);

static const MethodParam params_physics_fracture_cluster_groups[] = {
    {"object", "any", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_physics_fracture_cluster_groups = {
    "physics.fracture_cluster_groups", "physics",
    "List or build the cluster groups of a fractured object",
    nullptr,
    "write", "SceneWrite", false, "any",
    "physics|fracture|cluster|groups|destruction|group",
    nullptr,
    params_physics_fracture_cluster_groups, 1,
    true
};
static const MethodRegistration reg_physics_fracture_cluster_groups(desc_physics_fracture_cluster_groups);

static const MethodParam params_physics_fracture_object[] = {
    {"object", "string", true, "Object to shatter", nullptr, nullptr},
    {"site_count", "int", false, "Number of Voronoi sites, i.e. shards", "15", nullptr},
    {"pattern", "int", false, "Fracture site distribution pattern", "0", nullptr},
    {"cluster_count", "int", false, "Number of shard clusters to form", "4", nullptr},
    {"exact_surface", "bool", false, "Clip shards against the exact surface rather than an approximation", "true", nullptr},
    {"preview_gap", "float", false, "Visual gap between shards in metres, for preview only", "0.02", nullptr},
    {"seed", "any", false, "Random seed for reproducible shard layouts", nullptr, nullptr},
};
static const MethodDescriptor desc_physics_fracture_object = {
    "physics.fracture_object", "physics",
    "Shatter an object into Voronoi shards and return the shard objects",
    "site_count sets how many shards; exact_surface clips shards against the real surface instead of approximating it. The shards are ordinary objects afterwards - group them with physics.make_fracture_group so they hold together until something breaks them.",
    "write", "SceneWrite", false, "any",
    "physics|fracture|object|destruction|shatter|break|voronoi|shards|debris",
    "physics.make_fracture_group|physics.fracture_cluster_groups|physics.unfracture_object",
    params_physics_fracture_object, 7,
    true
};
static const MethodRegistration reg_physics_fracture_object(desc_physics_fracture_object);

static const MethodParam params_physics_get_body[] = {
    {"object", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_physics_get_body = {
    "physics.get_body", "physics",
    "Return a rigid body's mass, damping, friction, restitution and motion type",
    nullptr,
    "read", "Read", false, "any",
    "physics|get|body|simulation|rigid",
    nullptr,
    params_physics_get_body, 1,
    true
};
static const MethodRegistration reg_physics_get_body(desc_physics_get_body);

static const MethodParam params_physics_get_fracture_group[] = {
    {"group", "any", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_physics_get_fracture_group = {
    "physics.get_fracture_group", "physics",
    "Report a fracture group's mass, integrity and break thresholds",
    nullptr,
    "read", "Read", false, "any",
    "physics|get|fracture|group|destruction|measure|verify",
    nullptr,
    params_physics_get_fracture_group, 1,
    true
};
static const MethodRegistration reg_physics_get_fracture_group(desc_physics_get_fracture_group);

static const MethodParam params_physics_make_fracture_group[] = {
    {"group", "any", true, "", nullptr, nullptr},
    {"break_velocity", "float", false, "", "5.0", nullptr},
    {"integrity_exponent", "float", false, "", "1.5", nullptr},
    {"integrity_weakening", "bool", false, "", "true", nullptr},
    {"minimum_threshold_scale", "float", false, "", "0.15", nullptr},
    {"shard_objects", "any", false, "", nullptr, nullptr},
    {"source_object", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_physics_make_fracture_group = {
    "physics.make_fracture_group", "physics",
    "Bond shards into a fracture group that holds together until the break threshold is exceeded",
    "break_velocity is a velocity threshold, not an impulse - the group's mass is taken into account when the threshold is evaluated.",
    "write", "SceneWrite", false, "any",
    "physics|make|fracture|group|destruction|bond|threshold|collapse",
    "physics.fracture_object|physics.break_fracture_group|physics.apply_fracture_impulse",
    params_physics_make_fracture_group, 7,
    true
};
static const MethodRegistration reg_physics_make_fracture_group(desc_physics_make_fracture_group);

static const MethodParam params_physics_remove_body[] = {
    {"object", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_physics_remove_body = {
    "physics.remove_body", "physics",
    "Remove an object's rigid body",
    nullptr,
    "write", "SceneWrite", false, "any",
    "physics|remove|body|simulation|rigid",
    nullptr,
    params_physics_remove_body, 1,
    true
};
static const MethodRegistration reg_physics_remove_body(desc_physics_remove_body);

static const MethodDescriptor desc_physics_reset = {
    "physics.reset", "physics",
    "Reset the rigid-body simulation to its initial state",
    nullptr,
    "write", "SceneWrite", false, "any",
    "physics|reset|simulation",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_physics_reset(desc_physics_reset);

static const MethodParam params_physics_set_gravity[] = {
    {"gravity", "vec3", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_physics_set_gravity = {
    "physics.set_gravity", "physics",
    "Set the world gravity vector in metres per second squared",
    nullptr,
    "write", "SceneWrite", false, "any",
    "physics|set|gravity|simulation",
    nullptr,
    params_physics_set_gravity, 1,
    true
};
static const MethodRegistration reg_physics_set_gravity(desc_physics_set_gravity);

static const MethodParam params_physics_step[] = {
    {"dt", "float", false, "", "0.0166667", nullptr},
};
static const MethodDescriptor desc_physics_step = {
    "physics.step", "physics",
    "Advance the rigid-body solver by one timestep",
    nullptr,
    "write", "SceneWrite", false, "any",
    "physics|step|simulation|advance",
    nullptr,
    params_physics_step, 1,
    true
};
static const MethodRegistration reg_physics_step(desc_physics_step);

static const MethodParam params_physics_unfracture_object[] = {
    {"object", "any", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_physics_unfracture_object = {
    "physics.unfracture_object", "physics",
    "Restore a fractured object back to its unbroken form",
    nullptr,
    "write", "SceneWrite", false, "any",
    "physics|unfracture|object|destruction|undo|restore",
    nullptr,
    params_physics_unfracture_object, 1,
    true
};
static const MethodRegistration reg_physics_unfracture_object(desc_physics_unfracture_object);

static const MethodDescriptor desc_post_get = {
    "post.get", "post",
    "Read the post-processing settings: exposure, gamma, tone mapping, saturation, vignette and stylize",
    nullptr,
    "read", "Read", false, "any",
    "post|get|grade|look",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_post_get(desc_post_get);

static const MethodParam params_post_set_color_temperature[] = {
    {"color_temperature", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_post_set_color_temperature = {
    "post.set_color_temperature", "post",
    "Set post-process colour temperature",
    nullptr,
    "write", "SceneWrite", false, "any",
    "post|set|color|temperature|grade|white-balance",
    nullptr,
    params_post_set_color_temperature, 1,
    true
};
static const MethodRegistration reg_post_set_color_temperature(desc_post_set_color_temperature);

static const MethodParam params_post_set_exposure[] = {
    {"exposure", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_post_set_exposure = {
    "post.set_exposure", "post",
    "Set post-process exposure",
    "Post changes must not reset sample accumulation.",
    "write", "SceneWrite", false, "any",
    "post|set|exposure|brightness|grade",
    nullptr,
    params_post_set_exposure, 1,
    true
};
static const MethodRegistration reg_post_set_exposure(desc_post_set_exposure);

static const MethodParam params_post_set_gamma[] = {
    {"gamma", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_post_set_gamma = {
    "post.set_gamma", "post",
    "Set post-process gamma",
    nullptr,
    "write", "SceneWrite", false, "any",
    "post|set|gamma|grade",
    nullptr,
    params_post_set_gamma, 1,
    true
};
static const MethodRegistration reg_post_set_gamma(desc_post_set_gamma);

static const MethodParam params_post_set_saturation[] = {
    {"saturation", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_post_set_saturation = {
    "post.set_saturation", "post",
    "Set post-process saturation",
    nullptr,
    "write", "SceneWrite", false, "any",
    "post|set|saturation|grade|colour",
    nullptr,
    params_post_set_saturation, 1,
    true
};
static const MethodRegistration reg_post_set_saturation(desc_post_set_saturation);

static const MethodParam params_post_set_stylize_enabled[] = {
    {"stylize_enabled", "bool", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_post_set_stylize_enabled = {
    "post.set_stylize_enabled", "post",
    "Enable or disable the stylize pass",
    nullptr,
    "write", "SceneWrite", false, "any",
    "post|set|stylize|enabled|look|npr",
    nullptr,
    params_post_set_stylize_enabled, 1,
    true
};
static const MethodRegistration reg_post_set_stylize_enabled(desc_post_set_stylize_enabled);

static const MethodParam params_post_set_stylize_strength[] = {
    {"stylize_strength", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_post_set_stylize_strength = {
    "post.set_stylize_strength", "post",
    "Set stylize pass strength",
    nullptr,
    "write", "SceneWrite", false, "any",
    "post|set|stylize|strength|look|npr",
    nullptr,
    params_post_set_stylize_strength, 1,
    true
};
static const MethodRegistration reg_post_set_stylize_strength(desc_post_set_stylize_strength);

static const MethodParam params_post_set_tone_mapping[] = {
    {"tone_mapping", "string", true, "Operator", nullptr, "agx|aces|uncharted|filmic|none"},
};
static const MethodDescriptor desc_post_set_tone_mapping = {
    "post.set_tone_mapping", "post",
    "Set the tone mapping operator",
    nullptr,
    "write", "SceneWrite", false, "any",
    "post|set|tone|mapping|grade|filmic",
    nullptr,
    params_post_set_tone_mapping, 1,
    true
};
static const MethodRegistration reg_post_set_tone_mapping(desc_post_set_tone_mapping);

static const MethodParam params_post_set_vignette_enabled[] = {
    {"vignette_enabled", "bool", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_post_set_vignette_enabled = {
    "post.set_vignette_enabled", "post",
    "Enable or disable the vignette",
    nullptr,
    "write", "SceneWrite", false, "any",
    "post|set|vignette|enabled|look",
    nullptr,
    params_post_set_vignette_enabled, 1,
    true
};
static const MethodRegistration reg_post_set_vignette_enabled(desc_post_set_vignette_enabled);

static const MethodParam params_post_set_vignette_strength[] = {
    {"vignette_strength", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_post_set_vignette_strength = {
    "post.set_vignette_strength", "post",
    "Set vignette strength",
    nullptr,
    "write", "SceneWrite", false, "any",
    "post|set|vignette|strength|look",
    nullptr,
    params_post_set_vignette_strength, 1,
    true
};
static const MethodRegistration reg_post_set_vignette_strength(desc_post_set_vignette_strength);

static const MethodParam params_project_open[] = {
    {"path", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_project_open = {
    "project.open", "project",
    "Open a project file, replacing the current scene",
    nullptr,
    "write", "FilesRead|SceneWrite", false, "any",
    "project|open|file|load",
    nullptr,
    params_project_open, 1,
    true
};
static const MethodRegistration reg_project_open(desc_project_open);

static const MethodDescriptor desc_project_path = {
    "project.path", "project",
    "Return the current project file path",
    nullptr,
    "read", "Read", false, "any",
    "project|path|file",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_project_path(desc_project_path);

static const MethodParam params_project_save[] = {
    {"path", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_project_save = {
    "project.save", "project",
    "Save the project, optionally to a new path",
    nullptr,
    "write", "FilesWrite", false, "any",
    "project|save|file",
    nullptr,
    params_project_save, 1,
    true
};
static const MethodRegistration reg_project_save(desc_project_save);

static const MethodDescriptor desc_redo = {
    "redo", "redo",
    "Redo the last undone scene command",
    nullptr,
    "write", "SceneWrite", false, "any",
    "redo|history",
    "undo|redo_description",
    nullptr, 0,
    true
};
static const MethodRegistration reg_redo(desc_redo);

static const MethodDescriptor desc_redo_description = {
    "redo_description", "redo_description",
    "Name of the command that redo would reapply",
    nullptr,
    "read", "Read", false, "any",
    "redo_description|redo|description|history",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_redo_description(desc_redo_description);

static const MethodDescriptor desc_render_cancel = {
    "render.cancel", "render",
    "Cancel the running single-frame render",
    nullptr,
    "render", "Render", false, "any",
    "render|cancel|abort",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_render_cancel(desc_render_cancel);

static const MethodDescriptor desc_render_cancel_sequence = {
    "render.cancel_sequence", "render",
    "Cancel the running sequence render",
    nullptr,
    "render", "Render", false, "any",
    "render|cancel|sequence|abort|animation",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_render_cancel_sequence(desc_render_cancel_sequence);

static const MethodParam params_render_probe[] = {
    {"x", "int", false, "Region origin in pixels", "0", nullptr},
    {"y", "int", false, "Region origin in pixels", "0", nullptr},
    {"width", "int", false, "Region width in pixels; 0 means to the right edge", "0", nullptr},
    {"height", "int", false, "Region height in pixels; 0 means to the bottom edge", "0", nullptr},
    {"threshold", "float", false, "Luminance at or below which a pixel counts as black", "0.001", nullptr},
};
static const MethodDescriptor desc_render_probe = {
    "render.probe", "render",
    "Measure a region of the last viewport frame: mean/min/max luminance, black fraction, NaN fraction and a histogram",
    "This is the measurement an agent verifies its own work with. `available: false` means no frame was captured - it is not a dark scene. A non-zero nan_fraction means a shader produced invalid pixels.",
    "render", "Render", false, "ProbeInfo",
    "render|probe|measure|verify|luminance|black|nan|check",
    "viewport.capture|viewport.render_frames|viewport.status",
    params_render_probe, 5,
    true
};
static const MethodRegistration reg_render_probe(desc_render_probe);

static const MethodDescriptor desc_render_sequence_status = {
    "render.sequence_status", "render",
    "Report sequence render progress and current frame",
    nullptr,
    "render", "Render", false, "SequenceInfo",
    "render|sequence|status|progress|animation",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_render_sequence_status(desc_render_sequence_status);

static const MethodParam params_render_start[] = {
    {"output_path", "string", true, "Absolute path of the image file to write", nullptr, nullptr},
    {"spp", "int", true, "Samples per pixel", nullptr, nullptr},
};
static const MethodDescriptor desc_render_start = {
    "render.start", "render",
    "Render one frame at the given sample count and write it to an image file",
    "Blocking work runs on the render job; poll render.status for progress. The written image can be read back, so visual checks can be automated.",
    "render", "Render", false, "any",
    "render|start|output|image|final|save",
    "render.status|render.cancel|render.start_sequence",
    params_render_start, 2,
    true
};
static const MethodRegistration reg_render_start(desc_render_start);

static const MethodParam params_render_start_sequence[] = {
    {"end_frame", "int", true, "", nullptr, nullptr},
    {"output_dir", "string", true, "", nullptr, nullptr},
    {"spp", "int", true, "", nullptr, nullptr},
    {"start_frame", "int", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_render_start_sequence = {
    "render.start_sequence", "render",
    "Render a frame range to an output directory",
    nullptr,
    "render", "Render", false, "any",
    "render|start|sequence|animation|output|batch",
    "render.sequence_status|render.cancel_sequence",
    params_render_start_sequence, 4,
    true
};
static const MethodRegistration reg_render_start_sequence(desc_render_start_sequence);

static const MethodDescriptor desc_render_status = {
    "render.status", "render",
    "Report render job state, progress and current sample count",
    nullptr,
    "render", "Render", false, "RenderJobInfo",
    "render|status|progress",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_render_status(desc_render_status);

static const MethodParam params_render_volume_counters[] = {
    {"enabled", "bool", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_render_volume_counters = {
    "render.volume_counters", "render",
    "Enable or disable volume instrumentation counters",
    nullptr,
    "render", "Render", false, "any",
    "render|volume|counters|diagnostics|performance",
    nullptr,
    params_render_volume_counters, 1,
    true
};
static const MethodRegistration reg_render_volume_counters(desc_render_volume_counters);

static const MethodDescriptor desc_render_volume_stats = {
    "render.volume_stats", "render",
    "Return volume traversal counters for the last frame",
    nullptr,
    "render", "Render", false, "any",
    "render|volume|stats|diagnostics|performance",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_render_volume_stats(desc_render_volume_stats);

static const MethodDescriptor desc_request_render = {
    "request_render", "request_render",
    "Ask the viewport to render another frame",
    nullptr,
    "render", "Render", false, "any",
    "request_render|request|render|viewport|refresh",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_request_render(desc_request_render);

static const MethodDescriptor desc_reset_accumulation = {
    "reset_accumulation", "reset_accumulation",
    "Restart progressive sample accumulation in the viewport",
    "Post-process changes must NOT reset accumulation - only changes to scene, camera or lighting should.",
    "render", "Render", false, "any",
    "reset_accumulation|reset|accumulation|viewport|samples|refresh",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_reset_accumulation(desc_reset_accumulation);

static const MethodParam params_scatter_add_source[] = {
    {"group", "string", true, "", nullptr, nullptr},
    {"mesh", "string", true, "", nullptr, nullptr},
    {"align_to_normal", "bool", false, "", "true", nullptr},
    {"rotation_y", "float", false, "", "360.0", nullptr},
    {"scale_max", "float", false, "", "1.2", nullptr},
    {"scale_min", "float", false, "", "0.8", nullptr},
    {"weight", "float", false, "", "1.0", nullptr},
};
static const MethodDescriptor desc_scatter_add_source = {
    "scatter.add_source", "scatter",
    "Add a source object that a scatter group may instance",
    nullptr,
    "write", "SceneWrite", false, "any",
    "scatter|add|source|instancing|vegetation",
    nullptr,
    params_scatter_add_source, 7,
    true
};
static const MethodRegistration reg_scatter_add_source(desc_scatter_add_source);

static const MethodParam params_scatter_clear[] = {
    {"group", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_scatter_clear = {
    "scatter.clear", "scatter",
    "Remove the instances of a scatter group, keeping the group",
    nullptr,
    "write", "SceneWrite", false, "any",
    "scatter|clear|instancing|reset",
    nullptr,
    params_scatter_clear, 1,
    true
};
static const MethodRegistration reg_scatter_clear(desc_scatter_clear);

static const MethodParam params_scatter_create_group[] = {
    {"name", "string", true, "", nullptr, nullptr},
    {"target_node", "string", false, "", "", nullptr},
    {"target_type", "string", false, "", "mesh", nullptr},
};
static const MethodDescriptor desc_scatter_create_group = {
    "scatter.create_group", "scatter",
    "Create a scatter group that instances objects over a target surface",
    nullptr,
    "write", "SceneWrite", false, "any",
    "scatter|create|group|instancing|vegetation|forest|rocks",
    nullptr,
    params_scatter_create_group, 3,
    true
};
static const MethodRegistration reg_scatter_create_group(desc_scatter_create_group);

static const MethodParam params_scatter_delete_group[] = {
    {"group", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_scatter_delete_group = {
    "scatter.delete_group", "scatter",
    "Delete a scatter group",
    nullptr,
    "write", "SceneWrite", false, "any",
    "scatter|delete|group|instancing|remove",
    nullptr,
    params_scatter_delete_group, 1,
    true
};
static const MethodRegistration reg_scatter_delete_group(desc_scatter_delete_group);

static const MethodParam params_scatter_fill[] = {
    {"group", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_scatter_fill = {
    "scatter.fill", "scatter",
    "Populate a scatter group and report how many instances were spawned",
    nullptr,
    "write", "SceneWrite", false, "any",
    "scatter|fill|instancing|vegetation|populate|spawn",
    nullptr,
    params_scatter_fill, 1,
    true
};
static const MethodRegistration reg_scatter_fill(desc_scatter_fill);

static const MethodDescriptor desc_scatter_list_groups = {
    "scatter.list_groups", "scatter",
    "List the scatter groups with their sources, counts and placement settings",
    nullptr,
    "read", "Read", false, "any",
    "scatter|list|groups|instancing|vegetation|inventory",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_scatter_list_groups(desc_scatter_list_groups);

static const MethodParam params_scene_add_primitive[] = {
    {"type", "string", true, "Primitive shape", nullptr, "cube|sphere|plane|cylinder|torus"},
    {"name", "string", false, "Requested name; a numeric suffix is added if it is taken", "", nullptr},
    {"size", "float", false, "Edge length or radius in metres; must be positive", "1.0", nullptr},
};
static const MethodDescriptor desc_scene_add_primitive = {
    "scene.add_primitive", "scene",
    "Create a primitive mesh object and return its final name",
    "The name is made unique by suffix, so the returned name may differ from the requested one - always use the returned name afterwards.",
    "write", "SceneWrite", true, "string",
    "scene|add|primitive|create|mesh|cube|sphere|plane|cylinder|torus",
    "scene.delete|scene.set_transform|material.assign",
    params_scene_add_primitive, 3,
    true
};
static const MethodRegistration reg_scene_add_primitive(desc_scene_add_primitive);

static const MethodParam params_scene_delete[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_scene_delete = {
    "scene.delete", "scene",
    "Delete an object from the scene",
    nullptr,
    "write", "SceneWrite", true, "any",
    "scene|delete|remove",
    nullptr,
    params_scene_delete, 1,
    true
};
static const MethodRegistration reg_scene_delete(desc_scene_delete);

static const MethodParam params_scene_duplicate[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_scene_duplicate = {
    "scene.duplicate", "scene",
    "Duplicate an object and return the new object's name",
    nullptr,
    "write", "SceneWrite", true, "string",
    "scene|duplicate|copy|clone",
    nullptr,
    params_scene_duplicate, 1,
    true
};
static const MethodRegistration reg_scene_duplicate(desc_scene_duplicate);

static const MethodParam params_scene_get_transform[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_scene_get_transform = {
    "scene.get_transform", "scene",
    "Return an object's translation, rotation, scale and full matrix",
    nullptr,
    "read", "Read", false, "TransformInfo",
    "scene|get|transform",
    "scene.set_transform",
    params_scene_get_transform, 1,
    true
};
static const MethodRegistration reg_scene_get_transform(desc_scene_get_transform);

static const MethodParam params_scene_import_model[] = {
    {"path", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_scene_import_model = {
    "scene.import_model", "scene",
    "Import a model file (glTF/FBX/OBJ) into the scene",
    nullptr,
    "write", "FilesRead|SceneWrite", false, "any",
    "scene|import|model|load|asset|mesh",
    nullptr,
    params_scene_import_model, 1,
    true
};
static const MethodRegistration reg_scene_import_model(desc_scene_import_model);

static const MethodDescriptor desc_scene_list_objects = {
    "scene.list_objects", "scene",
    "List every object in the scene by name",
    "Covers both flat SoA meshes and legacy facade objects, so it is the reliable inventory call.",
    "read", "Read", false, "string[]",
    "scene|list|objects|inventory|browse",
    "scene.object_info|scene.object_exists",
    nullptr, 0,
    true
};
static const MethodRegistration reg_scene_list_objects(desc_scene_list_objects);

static const MethodParam params_scene_object_exists[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_scene_object_exists = {
    "scene.object_exists", "scene",
    "Report whether an object with this name exists",
    nullptr,
    "read", "Read", false, "bool",
    "scene|object|exists",
    "scene.list_objects",
    params_scene_object_exists, 1,
    true
};
static const MethodRegistration reg_scene_object_exists(desc_scene_object_exists);

static const MethodParam params_scene_object_info[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_scene_object_info = {
    "scene.object_info", "scene",
    "Return vertex and triangle counts for one object",
    nullptr,
    "write", "SceneWrite", false, "ObjectInfo",
    "scene|object|info",
    "scene.list_objects",
    params_scene_object_info, 1,
    true
};
static const MethodRegistration reg_scene_object_info(desc_scene_object_info);

static const MethodParam params_scene_set_transform[] = {
    {"name", "string", true, "Object name", nullptr, nullptr},
    {"matrix", "matrix", false, "Row-major 4x4 matrix; wins over the component form when both are sent", nullptr, nullptr},
    {"translation", "vec3", false, "World position in metres", nullptr, nullptr},
    {"rotation", "vec3", false, "Euler angles in degrees", nullptr, nullptr},
    {"scale", "vec3", false, "Per-axis scale factors", nullptr, nullptr},
};
static const MethodDescriptor desc_scene_set_transform = {
    "scene.set_transform", "scene",
    "Set an object's transform, either as a matrix or as translation/rotation/scale components",
    "Send `matrix`, or any combination of translation/rotation/scale - components you omit keep their current value. Rotation is in degrees.",
    "write", "SceneWrite", true, "any",
    "scene|set|transform|move|rotate|scale|position|placement",
    "scene.get_transform",
    params_scene_set_transform, 5,
    true
};
static const MethodRegistration reg_scene_set_transform(desc_scene_set_transform);

static const MethodParam params_script_run_file[] = {
    {"path", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_script_run_file = {
    "script.run_file", "script",
    "Run a Python script file inside the application",
    nullptr,
    "write", "Scripts|FilesRead", false, "any",
    "script|run|file|python|automation",
    nullptr,
    params_script_run_file, 1,
    true
};
static const MethodRegistration reg_script_run_file(desc_script_run_file);

static const MethodParam params_sculpt_get[] = {
    {"object", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_sculpt_get = {
    "sculpt.get", "sculpt",
    "Report an object's sculpt state: vertex count and mask range",
    nullptr,
    "read", "Read", false, "any",
    "sculpt|get|mesh",
    nullptr,
    params_sculpt_get, 1,
    true
};
static const MethodRegistration reg_sculpt_get(desc_sculpt_get);

static const MethodParam params_sculpt_mask_operation[] = {
    {"object", "string", true, "", nullptr, nullptr},
    {"operation", "string", true, "", nullptr, nullptr},
    {"seed", "int", false, "", "1337", nullptr},
    {"undo", "bool", false, "", "true", nullptr},
};
static const MethodDescriptor desc_sculpt_mask_operation = {
    "sculpt.mask_operation", "sculpt",
    "Run an operation over the sculpt mask (invert, clear, grow, ...)",
    nullptr,
    "write", "SceneWrite", false, "any",
    "sculpt|mask|operation",
    nullptr,
    params_sculpt_mask_operation, 4,
    true
};
static const MethodRegistration reg_sculpt_mask_operation(desc_sculpt_mask_operation);

static const MethodParam params_sculpt_paint_mask[] = {
    {"object", "string", true, "", nullptr, nullptr},
    {"radius", "float", true, "", nullptr, nullptr},
    {"value", "float", true, "", nullptr, nullptr},
    {"strength", "float", false, "", "1.0", nullptr},
    {"undo", "bool", false, "", "true", nullptr},
};
static const MethodDescriptor desc_sculpt_paint_mask = {
    "sculpt.paint_mask", "sculpt",
    "Paint the sculpt mask on a mesh",
    nullptr,
    "write", "SceneWrite", false, "any",
    "sculpt|paint|mask",
    nullptr,
    params_sculpt_paint_mask, 5,
    true
};
static const MethodRegistration reg_sculpt_paint_mask(desc_sculpt_paint_mask);

static const MethodParam params_sculpt_stroke[] = {
    {"object", "string", true, "", nullptr, nullptr},
    {"tool", "string", true, "", nullptr, nullptr},
    {"direction", "vec3", false, "", nullptr, nullptr},
    {"falloff", "float", false, "", "0.75", nullptr},
    {"radius", "float", false, "", "0.25", nullptr},
    {"seed", "int", false, "", "1337", nullptr},
    {"strength", "float", false, "", "0.05", nullptr},
    {"undo", "bool", false, "", "true", nullptr},
    {"use_mask", "bool", false, "", "true", nullptr},
};
static const MethodDescriptor desc_sculpt_stroke = {
    "sculpt.stroke", "sculpt",
    "Apply a sculpt brush stroke to a mesh",
    nullptr,
    "write", "SceneWrite", false, "any",
    "sculpt|stroke|mesh|deform|brush",
    nullptr,
    params_sculpt_stroke, 9,
    true
};
static const MethodRegistration reg_sculpt_stroke(desc_sculpt_stroke);

static const MethodDescriptor desc_select_all_objects = {
    "select.all_objects", "select",
    "Select every object in the scene",
    nullptr,
    "write", "SceneWrite", false, "any",
    "select|all|objects|selection",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_select_all_objects(desc_select_all_objects);

static const MethodDescriptor desc_select_clear = {
    "select.clear", "select",
    "Clear the selection",
    nullptr,
    "write", "SceneWrite", false, "any",
    "select|clear|selection",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_select_clear(desc_select_clear);

static const MethodParam params_select_deselect_object[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_select_deselect_object = {
    "select.deselect_object", "select",
    "Remove one object from the selection",
    nullptr,
    "write", "SceneWrite", false, "any",
    "select|deselect|object|selection",
    nullptr,
    params_select_deselect_object, 1,
    true
};
static const MethodRegistration reg_select_deselect_object(desc_select_deselect_object);

static const MethodParam params_select_light[] = {
    {"index", "int", true, "", nullptr, nullptr},
    {"additive", "bool", false, "", "false", nullptr},
};
static const MethodDescriptor desc_select_light = {
    "select.light", "select",
    "Select a light by index, optionally adding to the current selection",
    nullptr,
    "write", "SceneWrite", false, "any",
    "select|light|selection",
    nullptr,
    params_select_light, 2,
    true
};
static const MethodRegistration reg_select_light(desc_select_light);

static const MethodDescriptor desc_select_list = {
    "select.list", "select",
    "List the currently selected objects and lights",
    nullptr,
    "read", "Read", false, "SelectionEntry[]",
    "select|list|selection",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_select_list(desc_select_list);

static const MethodParam params_select_object[] = {
    {"name", "string", true, "", nullptr, nullptr},
    {"additive", "bool", false, "", "false", nullptr},
};
static const MethodDescriptor desc_select_object = {
    "select.object", "select",
    "Select an object by name, optionally adding to the current selection",
    nullptr,
    "write", "SceneWrite", false, "any",
    "select|object|selection",
    nullptr,
    params_select_object, 2,
    true
};
static const MethodRegistration reg_select_object(desc_select_object);

static const MethodParam params_sim_cache_bake[] = {
    {"cache_dir", "string", true, "", nullptr, nullptr},
    {"end_frame", "int", true, "", nullptr, nullptr},
    {"fps", "float", false, "", "24.0", nullptr},
    {"start_frame", "int", false, "", "0", nullptr},
};
static const MethodDescriptor desc_sim_cache_bake = {
    "sim_cache.bake", "sim_cache",
    "Bake a frame range of the simulation to the cache",
    nullptr,
    "write", "SceneWrite", false, "any",
    "sim_cache|sim|cache|bake|simulation",
    nullptr,
    params_sim_cache_bake, 4,
    true
};
static const MethodRegistration reg_sim_cache_bake(desc_sim_cache_bake);

static const MethodDescriptor desc_sim_cache_clear = {
    "sim_cache.clear", "sim_cache",
    "Clear the simulation cache",
    nullptr,
    "write", "SceneWrite", false, "any",
    "sim_cache|sim|cache|clear|simulation|reset",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_sim_cache_clear(desc_sim_cache_clear);

static const MethodDescriptor desc_sim_cache_status = {
    "sim_cache.status", "sim_cache",
    "Report the simulation cache: valid range, frames in RAM, cache directory and the config signature",
    "The config signature is what decides whether a bake is still valid; when it changes, cached frames belong to a different setup.",
    "read", "Read", false, "any",
    "sim_cache|sim|cache|status|simulation|bake|verify",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_sim_cache_status(desc_sim_cache_status);

static const MethodParam params_sim_graph_add_node[] = {
    {"scope", "string", true, "", nullptr, nullptr},
    {"type", "string", true, "", nullptr, nullptr},
    {"owner", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_sim_graph_add_node = {
    "sim_graph.add_node", "sim_graph",
    "Add a node of the given type to a simulation graph",
    nullptr,
    "write", "SceneWrite", false, "any",
    "sim_graph|sim|graph|add|node|simulation|nodes|create",
    nullptr,
    params_sim_graph_add_node, 3,
    true
};
static const MethodRegistration reg_sim_graph_add_node(desc_sim_graph_add_node);

static const MethodParam params_sim_graph_apply[] = {
    {"scope", "string", true, "", nullptr, nullptr},
    {"allow_restart", "bool", false, "", "false", nullptr},
    {"owner", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_sim_graph_apply = {
    "sim_graph.apply", "sim_graph",
    "Apply an evaluated simulation graph, reporting what was applied, refused or held",
    nullptr,
    "write", "SceneWrite", false, "any",
    "sim_graph|sim|graph|apply|simulation|nodes|commit",
    "sim_graph.evaluate",
    params_sim_graph_apply, 3,
    true
};
static const MethodRegistration reg_sim_graph_apply(desc_sim_graph_apply);

static const MethodParam params_sim_graph_attributes[] = {
    {"domain", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_sim_graph_attributes = {
    "sim_graph.attributes", "sim_graph",
    "List the attributes a simulation domain exposes",
    nullptr,
    "read", "Read", false, "any",
    "sim_graph|sim|graph|attributes|simulation|nodes",
    nullptr,
    params_sim_graph_attributes, 1,
    true
};
static const MethodRegistration reg_sim_graph_attributes(desc_sim_graph_attributes);

static const MethodParam params_sim_graph_clear[] = {
    {"scope", "string", true, "", nullptr, nullptr},
    {"owner", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_sim_graph_clear = {
    "sim_graph.clear", "sim_graph",
    "Remove every node from a simulation graph",
    nullptr,
    "write", "SceneWrite", false, "any",
    "sim_graph|sim|graph|clear|simulation|nodes|reset",
    nullptr,
    params_sim_graph_clear, 2,
    true
};
static const MethodRegistration reg_sim_graph_clear(desc_sim_graph_clear);

static const MethodDescriptor desc_sim_graph_clear_overrides = {
    "sim_graph.clear_overrides", "sim_graph",
    "Drop the overrides a simulation graph is holding",
    nullptr,
    "write", "SceneWrite", false, "any",
    "sim_graph|sim|graph|clear|overrides|simulation|nodes|reset",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_sim_graph_clear_overrides(desc_sim_graph_clear_overrides);

static const MethodParam params_sim_graph_connect[] = {
    {"from_node", "int", true, "", nullptr, nullptr},
    {"scope", "string", true, "", nullptr, nullptr},
    {"to_node", "int", true, "", nullptr, nullptr},
    {"from_pin", "int", false, "", "0", nullptr},
    {"owner", "string", false, "", "", nullptr},
    {"to_pin", "int", false, "", "0", nullptr},
};
static const MethodDescriptor desc_sim_graph_connect = {
    "sim_graph.connect", "sim_graph",
    "Connect two nodes in a simulation graph",
    nullptr,
    "write", "SceneWrite", false, "any",
    "sim_graph|sim|graph|connect|simulation|nodes|wire",
    nullptr,
    params_sim_graph_connect, 6,
    true
};
static const MethodRegistration reg_sim_graph_connect(desc_sim_graph_connect);

static const MethodDescriptor desc_sim_graph_couplings = {
    "sim_graph.couplings", "sim_graph",
    "Report declared versus actually running couplings between simulation domains",
    "declared_not_running and running_not_declared are the two ways a graph and the live solvers can disagree.",
    "write", "SceneWrite", false, "any",
    "sim_graph|sim|graph|couplings|simulation|nodes|coupling|verify|diagnostics",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_sim_graph_couplings(desc_sim_graph_couplings);

static const MethodParam params_sim_graph_create[] = {
    {"scope", "string", true, "", nullptr, nullptr},
    {"owner", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_sim_graph_create = {
    "sim_graph.create", "sim_graph",
    "Create a simulation node graph for a scope and owner",
    "Scope and owner are mandatory and identify what the graph drives; a graph is not global.",
    "write", "SceneWrite", false, "any",
    "sim_graph|sim|graph|create|simulation|nodes|scope",
    nullptr,
    params_sim_graph_create, 2,
    true
};
static const MethodRegistration reg_sim_graph_create(desc_sim_graph_create);

static const MethodParam params_sim_graph_delete[] = {
    {"scope", "string", true, "", nullptr, nullptr},
    {"owner", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_sim_graph_delete = {
    "sim_graph.delete", "sim_graph",
    "Delete a simulation node graph",
    nullptr,
    "write", "SceneWrite", false, "any",
    "sim_graph|sim|graph|delete|simulation|nodes|remove",
    nullptr,
    params_sim_graph_delete, 2,
    true
};
static const MethodRegistration reg_sim_graph_delete(desc_sim_graph_delete);

static const MethodParam params_sim_graph_evaluate[] = {
    {"scope", "string", true, "", nullptr, nullptr},
    {"owner", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_sim_graph_evaluate = {
    "sim_graph.evaluate", "sim_graph",
    "Evaluate a simulation graph and report the commands it would issue, without applying them",
    "The graph DECLARES; the solver REPORTS. Evaluate first, then apply - that split is what makes a graph inspectable.",
    "read", "Read", false, "any",
    "sim_graph|sim|graph|evaluate|simulation|nodes|dry-run|inspect",
    "sim_graph.apply|sim_graph.couplings",
    params_sim_graph_evaluate, 2,
    true
};
static const MethodRegistration reg_sim_graph_evaluate(desc_sim_graph_evaluate);

static const MethodDescriptor desc_sim_graph_list = {
    "sim_graph.list", "sim_graph",
    "List the simulation node graphs with their scope, owner and whether the owner still exists",
    "owner_missing means the graph outlived the entity it was written for - it will not run.",
    "write", "SceneWrite", false, "any",
    "sim_graph|sim|graph|list|simulation|nodes|inventory|scope",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_sim_graph_list(desc_sim_graph_list);

static const MethodParam params_sim_graph_nodes[] = {
    {"scope", "string", true, "", nullptr, nullptr},
    {"owner", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_sim_graph_nodes = {
    "sim_graph.nodes", "sim_graph",
    "List a simulation graph's nodes, their channels, sources and restart requirements",
    nullptr,
    "read", "Read", false, "any",
    "sim_graph|sim|graph|nodes|simulation",
    nullptr,
    params_sim_graph_nodes, 2,
    true
};
static const MethodRegistration reg_sim_graph_nodes(desc_sim_graph_nodes);

static const MethodParam params_sim_graph_set_node[] = {
    {"key", "string", true, "", nullptr, nullptr},
    {"node", "int", true, "", nullptr, nullptr},
    {"scope", "string", true, "", nullptr, nullptr},
    {"value", "string", true, "", nullptr, nullptr},
    {"owner", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_sim_graph_set_node = {
    "sim_graph.set_node", "sim_graph",
    "Set a text-valued key on a simulation node",
    nullptr,
    "write", "SceneWrite", false, "any",
    "sim_graph|sim|graph|set|node|simulation|nodes|configure",
    nullptr,
    params_sim_graph_set_node, 5,
    true
};
static const MethodRegistration reg_sim_graph_set_node(desc_sim_graph_set_node);

static const MethodParam params_sim_graph_set_node_value[] = {
    {"key", "string", true, "", nullptr, nullptr},
    {"node", "int", true, "", nullptr, nullptr},
    {"scope", "string", true, "", nullptr, nullptr},
    {"value", "float", true, "", nullptr, nullptr},
    {"owner", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_sim_graph_set_node_value = {
    "sim_graph.set_node_value", "sim_graph",
    "Set a numeric key on a simulation node",
    nullptr,
    "write", "SceneWrite", false, "any",
    "sim_graph|sim|graph|set|node|value|simulation|nodes|configure",
    nullptr,
    params_sim_graph_set_node_value, 5,
    true
};
static const MethodRegistration reg_sim_graph_set_node_value(desc_sim_graph_set_node_value);

static const MethodParam params_sim_graph_surface_attributes[] = {
    {"object", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_sim_graph_surface_attributes = {
    "sim_graph.surface_attributes", "sim_graph",
    "List the surface attributes an object exposes to simulation nodes",
    nullptr,
    "write", "SceneWrite", false, "any",
    "sim_graph|sim|graph|surface|attributes|simulation|nodes",
    nullptr,
    params_sim_graph_surface_attributes, 1,
    true
};
static const MethodRegistration reg_sim_graph_surface_attributes(desc_sim_graph_surface_attributes);

static const MethodParam params_templates_delete_user[] = {
    {"id", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_templates_delete_user = {
    "templates.delete_user", "templates",
    "Delete a user template",
    nullptr,
    "write", "SceneWrite", false, "any",
    "templates|delete|user|template|remove",
    nullptr,
    params_templates_delete_user, 1,
    true
};
static const MethodRegistration reg_templates_delete_user(desc_templates_delete_user);

static const MethodParam params_templates_get[] = {
    {"id", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_templates_get = {
    "templates.get", "templates",
    "Return one template's metadata",
    nullptr,
    "read", "Read", false, "any",
    "templates|get|template",
    nullptr,
    params_templates_get, 1,
    true
};
static const MethodRegistration reg_templates_get(desc_templates_get);

static const MethodDescriptor desc_templates_hide_hub = {
    "templates.hide_hub", "templates",
    "Hide the Template Hub window",
    nullptr,
    "write", "SceneWrite", false, "any",
    "templates|hide|hub|template|ui",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_templates_hide_hub(desc_templates_hide_hub);

static const MethodDescriptor desc_templates_is_hub_visible = {
    "templates.is_hub_visible", "templates",
    "Report whether the Template Hub is visible",
    nullptr,
    "write", "SceneWrite", false, "any",
    "templates|is|hub|visible|template|ui",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_templates_is_hub_visible(desc_templates_is_hub_visible);

static const MethodParam params_templates_list[] = {
    {"include_invalid", "bool", false, "", "false", nullptr},
};
static const MethodDescriptor desc_templates_list = {
    "templates.list", "templates",
    "List the available scene templates",
    nullptr,
    "read", "Read", false, "any",
    "templates|list|template|starter|scene",
    nullptr,
    params_templates_list, 1,
    true
};
static const MethodRegistration reg_templates_list(desc_templates_list);

static const MethodParam params_templates_open[] = {
    {"conflict_policy", "string", false, "", "reject", nullptr},
    {"id", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_templates_open = {
    "templates.open", "templates",
    "Open a template as the current scene",
    nullptr,
    "write", "SceneWrite", false, "any",
    "templates|open|template|load|start",
    nullptr,
    params_templates_open, 2,
    true
};
static const MethodRegistration reg_templates_open(desc_templates_open);

static const MethodParam params_templates_prepare[] = {
    {"conflict_policy", "string", false, "", "reject", nullptr},
    {"id", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_templates_prepare = {
    "templates.prepare", "templates",
    "Run a template's preflight without opening it",
    nullptr,
    "read", "Read", false, "any",
    "templates|prepare|template|preflight|validate",
    nullptr,
    params_templates_prepare, 2,
    true
};
static const MethodRegistration reg_templates_prepare(desc_templates_prepare);

static const MethodDescriptor desc_templates_refresh = {
    "templates.refresh", "templates",
    "Rescan the template directories",
    nullptr,
    "read", "Read", false, "any",
    "templates|refresh|template|reload",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_templates_refresh(desc_templates_refresh);

static const MethodParam params_templates_save_user[] = {
    {"category", "string", false, "", "user", nullptr},
    {"description", "string", false, "", "", nullptr},
    {"display_name", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_templates_save_user = {
    "templates.save_user", "templates",
    "Save the current scene as a user template",
    nullptr,
    "write", "SceneWrite", false, "any",
    "templates|save|user|template",
    nullptr,
    params_templates_save_user, 3,
    true
};
static const MethodRegistration reg_templates_save_user(desc_templates_save_user);

static const MethodDescriptor desc_templates_show_hub = {
    "templates.show_hub", "templates",
    "Show the Template Hub window",
    nullptr,
    "write", "SceneWrite", false, "any",
    "templates|show|hub|template|ui",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_templates_show_hub(desc_templates_show_hub);

static const MethodParam params_templates_validate[] = {
    {"id", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_templates_validate = {
    "templates.validate", "templates",
    "Validate a template and report the errors that would block it",
    nullptr,
    "read", "Read", false, "any",
    "templates|validate|template",
    nullptr,
    params_templates_validate, 1,
    true
};
static const MethodRegistration reg_templates_validate(desc_templates_validate);

static const MethodParam params_terrain_apply_preset[] = {
    {"preset", "string", true, "Preset name", nullptr, "default|snow_layer|snowy_mountain_valley|river_network"},
    {"name", "string", true, "", nullptr, nullptr},
    {"replace_graph", "bool", false, "", "false", nullptr},
};
static const MethodDescriptor desc_terrain_apply_preset = {
    "terrain.apply_preset", "terrain",
    "Apply a built-in terrain node preset",
    nullptr,
    "write", "SceneWrite", false, "any",
    "terrain|apply|preset|landscape|mountain|snow|river",
    nullptr,
    params_terrain_apply_preset, 3,
    true
};
static const MethodRegistration reg_terrain_apply_preset(desc_terrain_apply_preset);

static const MethodParam params_terrain_calculate_flow[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_calculate_flow = {
    "terrain.calculate_flow", "terrain",
    "Compute the water flow map over a terrain",
    nullptr,
    "write", "SceneWrite", false, "any",
    "terrain|calculate|flow|landscape|water|hydrology",
    nullptr,
    params_terrain_calculate_flow, 1,
    true
};
static const MethodRegistration reg_terrain_calculate_flow(desc_terrain_calculate_flow);

static const MethodParam params_terrain_cancel_evaluation[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_cancel_evaluation = {
    "terrain.cancel_evaluation", "terrain",
    "Cancel the running terrain evaluation",
    nullptr,
    "write", "SceneWrite", false, "any",
    "terrain|cancel|evaluation|landscape|abort",
    nullptr,
    params_terrain_cancel_evaluation, 1,
    true
};
static const MethodRegistration reg_terrain_cancel_evaluation(desc_terrain_cancel_evaluation);

static const MethodParam params_terrain_carve_river[] = {
    {"name", "string", true, "", nullptr, nullptr},
    {"river", "string", true, "", nullptr, nullptr},
    {"asymmetric_banks", "bool", false, "", "true", nullptr},
    {"deep_pools", "bool", false, "", "true", nullptr},
    {"depth_multiplier", "float", false, "", "1.0", nullptr},
    {"mode", "string", false, "", "natural", nullptr},
    {"noise_strength", "float", false, "", "0.3", nullptr},
    {"point_bars", "bool", false, "", "true", nullptr},
    {"post_erosion", "bool", false, "", "false", nullptr},
    {"post_erosion_iterations", "int", false, "", "12", nullptr},
    {"riffles", "bool", false, "", "true", nullptr},
    {"smoothness", "float", false, "", "0.5", nullptr},
    {"undo", "bool", false, "", "true", nullptr},
};
static const MethodDescriptor desc_terrain_carve_river = {
    "terrain.carve_river", "terrain",
    "Carve a river channel along a path into the terrain",
    nullptr,
    "write", "SceneWrite", false, "any",
    "terrain|carve|river|landscape|water|erosion",
    nullptr,
    params_terrain_carve_river, 13,
    true
};
static const MethodRegistration reg_terrain_carve_river(desc_terrain_carve_river);

static const MethodParam params_terrain_create[] = {
    {"size", "float", false, "World-space extent in metres", "1000.0", nullptr},
    {"resolution", "int", false, "Heightmap resolution in samples per side", "1024", nullptr},
    {"height_scale", "float", false, "Vertical scale in metres", "100.0", nullptr},
    {"name", "string", false, "", "Terrain", nullptr},
};
static const MethodDescriptor desc_terrain_create = {
    "terrain.create", "terrain",
    "Create a terrain heightfield of a given world size and resolution",
    nullptr,
    "write", "SceneWrite", false, "any",
    "terrain|create|landscape|mountain|ground|heightmap",
    "terrain.apply_preset|terrain.erode|terrain.evaluate",
    params_terrain_create, 4,
    true
};
static const MethodRegistration reg_terrain_create(desc_terrain_create);

static const MethodParam params_terrain_erode[] = {
    {"name", "string", true, "", nullptr, nullptr},
    {"type", "string", false, "Erosion model", "hydraulic", "hydraulic|thermal|fluvial|wind"},
    {"iterations", "int", false, "Solver iterations; 0 uses the model's own default", "0", nullptr},
    {"strength", "float", false, "Erosion strength", "0.2", nullptr},
    {"talus_angle", "float", false, "Repose angle for thermal erosion", "0.5", nullptr},
    {"backend", "string", false, "Compute backend; auto picks GPU when available", "auto", nullptr},
    {"amount", "float", false, "", "0.3", nullptr},
    {"direction", "float", false, "", "45.0", nullptr},
    {"seed", "int", false, "", "1337", nullptr},
    {"undo", "bool", false, "", "true", nullptr},
};
static const MethodDescriptor desc_terrain_erode = {
    "terrain.erode", "terrain",
    "Run an erosion pass over a terrain",
    nullptr,
    "write", "SceneWrite", false, "any",
    "terrain|erode|landscape|erosion|weathering|realism",
    nullptr,
    params_terrain_erode, 10,
    true
};
static const MethodRegistration reg_terrain_erode(desc_terrain_erode);

static const MethodParam params_terrain_evaluate[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_evaluate = {
    "terrain.evaluate", "terrain",
    "Evaluate the terrain node graph and bake the result into the heightfield",
    nullptr,
    "write", "SceneWrite", false, "any",
    "terrain|evaluate|landscape|bake|apply",
    "terrain.evaluation_status|terrain.cancel_evaluation",
    params_terrain_evaluate, 1,
    true
};
static const MethodRegistration reg_terrain_evaluate(desc_terrain_evaluate);

static const MethodParam params_terrain_evaluation_status[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_evaluation_status = {
    "terrain.evaluation_status", "terrain",
    "Report terrain evaluation progress and the node being processed",
    nullptr,
    "write", "SceneWrite", false, "any",
    "terrain|evaluation|status|landscape|progress",
    nullptr,
    params_terrain_evaluation_status, 1,
    true
};
static const MethodRegistration reg_terrain_evaluation_status(desc_terrain_evaluation_status);

static const MethodParam params_terrain_export_heightmap[] = {
    {"filepath", "string", true, "", nullptr, nullptr},
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_export_heightmap = {
    "terrain.export_heightmap", "terrain",
    "Export a terrain heightfield to an image file",
    nullptr,
    "write", "FilesWrite", false, "any",
    "terrain|export|heightmap|landscape",
    nullptr,
    params_terrain_export_heightmap, 2,
    true
};
static const MethodRegistration reg_terrain_export_heightmap(desc_terrain_export_heightmap);

static const MethodParam params_terrain_get[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_get = {
    "terrain.get", "terrain",
    "Return one terrain's size, resolution, height scale and node-graph state",
    nullptr,
    "read", "Read", false, "any",
    "terrain|get|landscape",
    nullptr,
    params_terrain_get, 1,
    true
};
static const MethodRegistration reg_terrain_get(desc_terrain_get);

static const MethodParam params_terrain_import_heightmap[] = {
    {"filepath", "string", true, "", nullptr, nullptr},
    {"height_scale", "float", false, "", "100.0", nullptr},
    {"max_resolution", "int", false, "", "2048", nullptr},
    {"name", "string", false, "", "TerrainImported", nullptr},
    {"size", "float", false, "", "1000.0", nullptr},
};
static const MethodDescriptor desc_terrain_import_heightmap = {
    "terrain.import_heightmap", "terrain",
    "Import a heightmap image into a terrain",
    nullptr,
    "write", "FilesRead|SceneWrite", false, "any",
    "terrain|import|heightmap|landscape",
    nullptr,
    params_terrain_import_heightmap, 5,
    true
};
static const MethodRegistration reg_terrain_import_heightmap(desc_terrain_import_heightmap);

static const MethodDescriptor desc_terrain_list = {
    "terrain.list", "terrain",
    "List the terrain objects",
    nullptr,
    "read", "Read", false, "any",
    "terrain|list|landscape|inventory",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_terrain_list(desc_terrain_list);

static const MethodDescriptor desc_terrain_list_rivers = {
    "terrain.list_rivers", "terrain",
    "List the rivers carved into a terrain",
    nullptr,
    "read", "Read", false, "any",
    "terrain|list|rivers|landscape|river|inventory",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_terrain_list_rivers(desc_terrain_list_rivers);

static const MethodParam params_terrain_remove[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_remove = {
    "terrain.remove", "terrain",
    "Delete a terrain",
    nullptr,
    "write", "SceneWrite", false, "any",
    "terrain|remove|landscape",
    nullptr,
    params_terrain_remove, 1,
    true
};
static const MethodRegistration reg_terrain_remove(desc_terrain_remove);

static const MethodParam params_terrain_sample_height[] = {
    {"name", "string", true, "", nullptr, nullptr},
    {"world_x", "float", true, "", nullptr, nullptr},
    {"world_z", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_sample_height = {
    "terrain.sample_height", "terrain",
    "Sample terrain height at a world position",
    nullptr,
    "read", "Read", false, "any",
    "terrain|sample|height|landscape|measure|probe|placement",
    nullptr,
    params_terrain_sample_height, 3,
    true
};
static const MethodRegistration reg_terrain_sample_height(desc_terrain_sample_height);

static const MethodDescriptor desc_timeline_get_frame = {
    "timeline.get_frame", "timeline",
    "Return the current timeline frame",
    nullptr,
    "read", "Read", false, "any",
    "timeline|get|frame|time|playhead",
    "timeline.set_frame",
    nullptr, 0,
    true
};
static const MethodRegistration reg_timeline_get_frame(desc_timeline_get_frame);

static const MethodParam params_timeline_set_frame[] = {
    {"frame", "int", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_timeline_set_frame = {
    "timeline.set_frame", "timeline",
    "Move the timeline playhead to a frame",
    "This is how simulations are advanced from a script: stepping the frame runs the solvers that are live for that frame.",
    "write", "SceneWrite", false, "any",
    "timeline|set|frame|time|playhead|simulate|advance",
    "timeline.get_frame|sim_cache.bake",
    params_timeline_set_frame, 1,
    true
};
static const MethodRegistration reg_timeline_set_frame(desc_timeline_set_frame);

static const MethodDescriptor desc_undo = {
    "undo", "undo",
    "Undo the last recorded scene command",
    nullptr,
    "write", "SceneWrite", false, "any",
    "undo|history",
    "redo|undo_description",
    nullptr, 0,
    true
};
static const MethodRegistration reg_undo(desc_undo);

static const MethodDescriptor desc_undo_description = {
    "undo_description", "undo_description",
    "Name of the command that undo would revert",
    nullptr,
    "read", "Read", false, "any",
    "undo_description|undo|description|history",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_undo_description(desc_undo_description);

static const MethodDescriptor desc_version = {
    "version", "version",
    "Return the RayTrophi Studio version string",
    nullptr,
    "read", "Read", false, "any",
    "version",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_version(desc_version);

static const MethodParam params_viewport_capture[] = {
    {"enabled", "bool", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_viewport_capture = {
    "viewport.capture", "viewport",
    "Turn viewport frame capture on or off",
    "Capture must be on before render.probe or the state summary can measure anything.",
    "render", "Render", false, "any",
    "viewport|capture|measure|verify|enable",
    "render.probe|viewport.render_frames",
    params_viewport_capture, 1,
    true
};
static const MethodRegistration reg_viewport_capture(desc_viewport_capture);

static const MethodParam params_viewport_render_frames[] = {
    {"count", "int", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_viewport_render_frames = {
    "viewport.render_frames", "viewport",
    "Render a fixed number of viewport frames and report timing and convergence",
    "Use this to converge the viewport deliberately before probing it, instead of guessing how long to wait.",
    "render", "Render", false, "any",
    "viewport|render|frames|measure|converge|samples|wait",
    "render.probe|viewport.capture",
    params_viewport_render_frames, 1,
    true
};
static const MethodRegistration reg_viewport_render_frames(desc_viewport_render_frames);

static const MethodParam params_viewport_set_shading[] = {
    {"mode", "string", true, "", nullptr, nullptr},
    {"matcap_preset", "int", false, "", "-1", nullptr},
};
static const MethodDescriptor desc_viewport_set_shading = {
    "viewport.set_shading", "viewport",
    "Switch the viewport shading mode (solid, material, rendered, matcap) and optionally the matcap preset",
    "Resets accumulation like the panel buttons do, so probe AFTER switching - otherwise you measure the frame from the mode you left. Fails loudly when a mode is unavailable instead of silently falling back to rendered.",
    "render", "Render", false, "any",
    "viewport|set|shading|display|solid|rendered|matcap|preview",
    "viewport.shading|viewport.render_frames|render.probe",
    params_viewport_set_shading, 2,
    true
};
static const MethodRegistration reg_viewport_set_shading(desc_viewport_set_shading);

static const MethodDescriptor desc_viewport_shading = {
    "viewport.shading", "viewport",
    "Report which viewport shading mode is on screen and whether the interactive raster viewport exists",
    "interactive_available false means this build has no raster viewport (no Vulkan), so 'rendered' is the only selectable mode - a rejected set_shading there is the machine, not a bad request.",
    "render", "Render", false, "any",
    "viewport|shading|display|measure",
    "viewport.set_shading|viewport.status",
    nullptr, 0,
    true
};
static const MethodRegistration reg_viewport_shading(desc_viewport_shading);

static const MethodDescriptor desc_viewport_status = {
    "viewport.status", "viewport",
    "Report viewport backend, shading mode, resolution, sample count, capture state and whether a frame is available",
    nullptr,
    "render", "Render", false, "ViewportStatusInfo",
    "viewport|status|measure|verify|backend|samples",
    "viewport.capture|render.probe",
    nullptr, 0,
    true
};
static const MethodRegistration reg_viewport_status(desc_viewport_status);

static const MethodDescriptor desc_world_get = {
    "world.get", "world",
    "Return world mode, background colour, sun angles and atmosphere settings",
    nullptr,
    "read", "Read", false, "WorldState",
    "world|get|environment|sky",
    nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_world_get(desc_world_get);

static const MethodParam params_world_set_atmosphere_intensity[] = {
    {"atmosphere_intensity", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_world_set_atmosphere_intensity = {
    "world.set_atmosphere_intensity", "world",
    "Set the strength of atmospheric scattering in the physical sky",
    nullptr,
    "write", "SceneWrite", false, "any",
    "world|set|atmosphere|intensity|environment|sky|haze|fog",
    nullptr,
    params_world_set_atmosphere_intensity, 1,
    true
};
static const MethodRegistration reg_world_set_atmosphere_intensity(desc_world_set_atmosphere_intensity);

static const MethodParam params_world_set_background_color[] = {
    {"background_color", "vec3", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_world_set_background_color = {
    "world.set_background_color", "world",
    "Set the flat background colour used in solid world mode",
    nullptr,
    "write", "SceneWrite", false, "any",
    "world|set|background|color|environment|sky|colour",
    nullptr,
    params_world_set_background_color, 1,
    true
};
static const MethodRegistration reg_world_set_background_color(desc_world_set_background_color);

static const MethodParam params_world_set_mode[] = {
    {"mode", "string", true, "Background model", nullptr, "solid|hdri|nishita"},
};
static const MethodDescriptor desc_world_set_mode = {
    "world.set_mode", "world",
    "Switch the world background between flat colour, HDRI and physical sky",
    nullptr,
    "write", "SceneWrite", false, "any",
    "world|set|mode|environment|sky|background|hdri",
    nullptr,
    params_world_set_mode, 1,
    true
};
static const MethodRegistration reg_world_set_mode(desc_world_set_mode);

static const MethodParam params_world_set_sun_azimuth[] = {
    {"sun_azimuth", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_world_set_sun_azimuth = {
    "world.set_sun_azimuth", "world",
    "Set the sun compass angle of the physical sky",
    nullptr,
    "write", "SceneWrite", false, "any",
    "world|set|sun|azimuth|environment|sky|direction",
    nullptr,
    params_world_set_sun_azimuth, 1,
    true
};
static const MethodRegistration reg_world_set_sun_azimuth(desc_world_set_sun_azimuth);

static const MethodParam params_world_set_sun_elevation[] = {
    {"sun_elevation", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_world_set_sun_elevation = {
    "world.set_sun_elevation", "world",
    "Set the sun elevation angle of the physical sky",
    nullptr,
    "write", "SceneWrite", false, "any",
    "world|set|sun|elevation|environment|sky|time-of-day",
    nullptr,
    params_world_set_sun_elevation, 1,
    true
};
static const MethodRegistration reg_world_set_sun_elevation(desc_world_set_sun_elevation);

static const MethodParam params_world_set_sun_intensity[] = {
    {"sun_intensity", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_world_set_sun_intensity = {
    "world.set_sun_intensity", "world",
    "Set the sun intensity of the physical sky",
    nullptr,
    "write", "SceneWrite", false, "any",
    "world|set|sun|intensity|environment|sky|brightness",
    nullptr,
    params_world_set_sun_intensity, 1,
    true
};
static const MethodRegistration reg_world_set_sun_intensity(desc_world_set_sun_intensity);

static const MethodParam params_world_set_sun_size[] = {
    {"sun_size", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_world_set_sun_size = {
    "world.set_sun_size", "world",
    "Set the sun's angular size; larger values give softer shadows",
    nullptr,
    "write", "SceneWrite", false, "any",
    "world|set|sun|size|environment|sky|shadow|softness",
    nullptr,
    params_world_set_sun_size, 1,
    true
};
static const MethodRegistration reg_world_set_sun_size(desc_world_set_sun_size);

} // namespace
