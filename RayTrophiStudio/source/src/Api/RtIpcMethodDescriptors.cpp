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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_addons_reload, 1,
    true
};
static const MethodRegistration reg_addons_reload(desc_addons_reload);

static const MethodParam params_agent_chat_poll[] = {
    {"agent_id", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_agent_chat_poll = {
    "agent.chat_poll", "agent",
    "Take the user prompts queued in the Agent Chat panel and mark the agent as alive",
    "Polling is also the panel's heartbeat: it shows the agent as online for a few seconds after each call. An agent that blocks on a long model turn should keep polling from a second thread, or the panel reports it offline.",
    "read", "Read", false, "QueuedPrompt[]",
    "agent|chat|poll|prompt|heartbeat",
    "agent.chat_send",
    nullptr, nullptr, nullptr, nullptr,
    params_agent_chat_poll, 1,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_agent_describe, 1,
    true
};
static const MethodRegistration reg_agent_describe(desc_agent_describe);

static const MethodDescriptor desc_agent_discover = {
    "agent.discover", "agent",
    "Identify the application and list every capability domain, agent role and coverage metric",
    "First call for a new agent session. registered_methods is how many methods are dispatched; documented_coverage is the share that carries a hand-written summary. A documented_coverage below 1.0 means agent.describe will answer some methods with parameters but no explanation.",
    "read", "Read", false, "DiscoveryInfo",
    "agent|discover|bootstrap|handshake|capabilities|introspection",
    "agent.list_methods|agent.describe|agent.search_capabilities",
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_agent_search_capabilities, 2,
    true
};
static const MethodRegistration reg_agent_search_capabilities(desc_agent_search_capabilities);

static const MethodParam params_agent_send_prompt[] = {
    {"content", "string", true, "", nullptr, nullptr},
    {"target", "string", true, "", nullptr, nullptr},
    {"sender", "string", false, "", "Agent", nullptr},
};
static const MethodDescriptor desc_agent_send_prompt = {
    "agent.send_prompt", "agent",
    "Queue a task for another agent to pick up on its next poll, addressed by agent id or 'all'",
    "Returns queued:true, NOT delivered - nothing has run yet and the target agent may never poll. Do not report the work as done on the strength of this call; ask for the result, or check the scene yourself. The target must match the id the other agent polls with (agent.chat_poll agent_id), and 'all' broadcasts.",
    "write", "AgentChat", false, "any",
    "agent|send|prompt|delegate|multi-agent|task|handoff",
    "agent.chat_poll|agent.chat_send|agent.roles",
    nullptr, nullptr, "agent.get_state_summary", nullptr,
    params_agent_send_prompt, 3,
    true
};
static const MethodRegistration reg_agent_send_prompt(desc_agent_send_prompt);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_anim_set_time, 3,
    true
};
static const MethodRegistration reg_anim_set_time(desc_anim_set_time);

static const MethodParam params_anim_source_channels[] = {
    {"clip", "string", false, "Clip name; empty or omitted uses the first imported clip", "", nullptr},
};
static const MethodDescriptor desc_anim_source_channels = {
    "anim.source_channels", "anim",
    "Per-node key counts for one raw imported clip",
    "The breakdown behind anim.source_clips, sorted by node name so two runs diff without post-processing. Kept separate because a rigged character has hundreds of nodes and the clip totals answer most questions; reach for this when the totals match but the pose does not, which means the keys landed on different NODES.",
    "read", "Read", false, "Array of objects with node_name, position_keys, rotation_keys, scaling_keys",
    "anim|source|channels|animation|clip|bones|import|acceptance",
    "anim.source_clips",
    nullptr, nullptr, nullptr, nullptr,
    params_anim_source_channels, 1,
    true
};
static const MethodRegistration reg_anim_source_channels(desc_anim_source_channels);

static const MethodDescriptor desc_anim_source_clips = {
    "anim.source_clips", "anim",
    "List the RAW imported animation clips with channel and key counts",
    "SceneData::animationDataList as the LOADER produced it - not a character's AnimationController state (that is anim.clips). This is the acceptance instrument for replacing the Assimp importer: the same file must yield identical channel and key counts on the old and the new path, and 'it still animates' is not a measurement. A channel is one animated node name; position_keys/rotation_keys/scaling_keys are summed over all channels. Times are in TICKS (duration_ticks, first_key_time, last_key_time); divide by ticks_per_second for seconds. first_key_time > last_key_time would mean a reader emitted unsorted keys - the sampling code assumes time-sorted keys.",
    "read", "Read", false, "Array of objects with name, model_name, duration_ticks, ticks_per_second, duration_seconds, start_frame, end_frame, position_channels, rotation_channels, scaling_channels, position_keys, rotation_keys, scaling_keys, first_key_time, last_key_time",
    "anim|source|clips|animation|clip|import|acceptance|assimp|gltf|channels|keys",
    "anim.source_channels|anim.clips|scene.import_model",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_anim_source_clips(desc_anim_source_clips);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_anim_trigger_graph_param, 2,
    true
};
static const MethodRegistration reg_anim_trigger_graph_param(desc_anim_trigger_graph_param);

static const MethodParam params_attr_list[] = {
    {"scope", "string", true, "", nullptr, nullptr},
    {"id", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_attr_list = {
    "attr.list", "attr",
    "List the attribute names that exist for an object/domain/world scope",
    "Unifies the old sim_graph.attributes (domain) and sim_graph.surface_attributes (object) into one surface that also covers world.",
    "read", "Read", false, "any",
    "attr|list|simulation|attributes|discover",
    "attr.stats",
    nullptr, nullptr, nullptr, nullptr,
    params_attr_list, 2,
    true
};
static const MethodRegistration reg_attr_list(desc_attr_list);

static const MethodParam params_attr_stats[] = {
    {"name", "string", true, "", nullptr, nullptr},
    {"scope", "string", true, "", nullptr, nullptr},
    {"id", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_attr_stats = {
    "attr.stats", "attr",
    "Measure one named attribute directly: count, min, max, mean",
    "available=false means the attribute could NOT be measured (no live particles, no MSF field, unknown name) - not the same as a value that measured zero. A world field is a single scalar: min/max/mean all read the same number.",
    "read", "Read", false, "any",
    "attr|stats|simulation|attributes|measure",
    "attr.list",
    nullptr, nullptr, nullptr, nullptr,
    params_attr_stats, 3,
    true
};
static const MethodRegistration reg_attr_stats(desc_attr_stats);

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
    nullptr, nullptr, nullptr, nullptr,
    params_batch, 1,
    true
};
static const MethodRegistration reg_batch(desc_batch);

static const MethodDescriptor desc_camera_get = {
    "camera.get", "camera",
    "Return camera position, target, up vector, field of view, focus distance, aperture and the physical exposure state",
    "EXPOSURE IS A RELATIVE MODEL, NOT AN ABSOLUTE PHOTOMETRIC ONE. The textbook formula 1/(1.2*2^EV100) assumes scene radiance in cd/m2; this engine's light intensities are arbitrary, so the multiplier is a ratio against a calibrated baseline (0.00003125) chosen to avoid a black viewport. Do not 'fix' it into the absolute formula - every scene would go black and the symptom would read as 'too dark', not 'wrong formula'. 'iso_value', 'shutter_seconds', 'f_number' and 'exposure_factor' are DERIVED read-only outputs, not settings: a preset INDEX is not a measurement, and exposure_factor is the number actually handed to the shaders. 'aperture' is the depth-of-field dial and does NOT affect exposure; 'fstop_preset_index' is the exposure/lens dial and does NOT affect depth of field. PRIORITY ORDER, and it is the most common reason these dials look dead: if auto_exposure is ON only ev_compensation is applied and ISO/shutter/f-stop are NOT read; if auto_exposure is OFF but use_physical_exposure is OFF too, again only ev_compensation applies. The call still SUCCEEDS because it really did write the setting - the image not moving is the model's precedence, not a fault. Read camera.get exposure_factor to see the multiplier actually applied.",
    "read", "Read", false, "any",
    "camera|get",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_camera_set_aperture, 1,
    true
};
static const MethodRegistration reg_camera_set_aperture(desc_camera_set_aperture);

static const MethodParam params_camera_set_auto_exposure[] = {
    {"enabled", "bool", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_camera_set_auto_exposure = {
    "camera.set_auto_exposure", "camera",
    "Turn auto exposure on; while on, only EV compensation is applied",
    "PRIORITY ORDER, and it is the most common reason these dials look dead: if auto_exposure is ON only ev_compensation is applied and ISO/shutter/f-stop are NOT read; if auto_exposure is OFF but use_physical_exposure is OFF too, again only ev_compensation applies. The call still SUCCEEDS because it really did write the setting - the image not moving is the model's precedence, not a fault. Read camera.get exposure_factor to see the multiplier actually applied.",
    "write", "SceneWrite", false, "any",
    "camera|set|auto|exposure",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_camera_set_auto_exposure, 1,
    true
};
static const MethodRegistration reg_camera_set_auto_exposure(desc_camera_set_auto_exposure);

static const MethodParam params_camera_set_ev_compensation[] = {
    {"ev", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_camera_set_ev_compensation = {
    "camera.set_ev_compensation", "camera",
    "Set exposure compensation in stops; this one applies in EVERY mode",
    "Unlike the preset dials this is always read, so it is the reliable way to move exposure without first arranging auto_exposure/use_physical_exposure. Rejected outside [-10, +10].",
    "write", "SceneWrite", false, "any",
    "camera|set|ev|compensation",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_camera_set_ev_compensation, 1,
    true
};
static const MethodRegistration reg_camera_set_ev_compensation(desc_camera_set_ev_compensation);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_camera_set_fov, 1,
    true
};
static const MethodRegistration reg_camera_set_fov(desc_camera_set_fov);

static const MethodParam params_camera_set_fstop_preset[] = {
    {"index", "int", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_camera_set_fstop_preset = {
    "camera.set_fstop_preset", "camera",
    "Select the f-stop preset by index; drives exposure and Cinema lens imperfections, NOT depth of field",
    "Depth of field still comes from the separate 'aperture' dial. On a real camera one f-number would do both; they are separate here because unifying them would silently change the blur of every existing scene - a recorded debt. PRIORITY ORDER, and it is the most common reason these dials look dead: if auto_exposure is ON only ev_compensation is applied and ISO/shutter/f-stop are NOT read; if auto_exposure is OFF but use_physical_exposure is OFF too, again only ev_compensation applies. The call still SUCCEEDS because it really did write the setting - the image not moving is the model's precedence, not a fault. Read camera.get exposure_factor to see the multiplier actually applied.",
    "write", "SceneWrite", false, "any",
    "camera|set|fstop|preset",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_camera_set_fstop_preset, 1,
    true
};
static const MethodRegistration reg_camera_set_fstop_preset(desc_camera_set_fstop_preset);

static const MethodParam params_camera_set_iso_preset[] = {
    {"index", "int", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_camera_set_iso_preset = {
    "camera.set_iso_preset", "camera",
    "Select the ISO preset by index; camera.get returns the resolved iso_value",
    "The index is not a measurement - read 'iso_value' from camera.get for the actual ISO. PRIORITY ORDER, and it is the most common reason these dials look dead: if auto_exposure is ON only ev_compensation is applied and ISO/shutter/f-stop are NOT read; if auto_exposure is OFF but use_physical_exposure is OFF too, again only ev_compensation applies. The call still SUCCEEDS because it really did write the setting - the image not moving is the model's precedence, not a fault. Read camera.get exposure_factor to see the multiplier actually applied.",
    "write", "SceneWrite", false, "any",
    "camera|set|iso|preset",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_camera_set_iso_preset, 1,
    true
};
static const MethodRegistration reg_camera_set_iso_preset(desc_camera_set_iso_preset);

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
    nullptr, nullptr, nullptr, nullptr,
    params_camera_set_position, 1,
    true
};
static const MethodRegistration reg_camera_set_position(desc_camera_set_position);

static const MethodParam params_camera_set_shutter_preset[] = {
    {"index", "int", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_camera_set_shutter_preset = {
    "camera.set_shutter_preset", "camera",
    "Select the shutter-speed preset by index; camera.get returns shutter_seconds",
    "Index, not seconds. Read 'shutter_seconds' from camera.get for the resolved exposure time. This dial does NOT drive motion blur. PRIORITY ORDER, and it is the most common reason these dials look dead: if auto_exposure is ON only ev_compensation is applied and ISO/shutter/f-stop are NOT read; if auto_exposure is OFF but use_physical_exposure is OFF too, again only ev_compensation applies. The call still SUCCEEDS because it really did write the setting - the image not moving is the model's precedence, not a fault. Read camera.get exposure_factor to see the multiplier actually applied.",
    "write", "SceneWrite", false, "any",
    "camera|set|shutter|preset",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_camera_set_shutter_preset, 1,
    true
};
static const MethodRegistration reg_camera_set_shutter_preset(desc_camera_set_shutter_preset);

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
    nullptr, nullptr, nullptr, nullptr,
    params_camera_set_target, 1,
    true
};
static const MethodRegistration reg_camera_set_target(desc_camera_set_target);

static const MethodParam params_camera_set_use_physical_exposure[] = {
    {"enabled", "bool", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_camera_set_use_physical_exposure = {
    "camera.set_use_physical_exposure", "camera",
    "Enable the ISO/shutter/f-stop preset chain (needs auto_exposure off)",
    "PRIORITY ORDER, and it is the most common reason these dials look dead: if auto_exposure is ON only ev_compensation is applied and ISO/shutter/f-stop are NOT read; if auto_exposure is OFF but use_physical_exposure is OFF too, again only ev_compensation applies. The call still SUCCEEDS because it really did write the setting - the image not moving is the model's precedence, not a fault. Read camera.get exposure_factor to see the multiplier actually applied.",
    "write", "SceneWrite", false, "any",
    "camera|set|use|physical|exposure",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_camera_set_use_physical_exposure, 1,
    true
};
static const MethodRegistration reg_camera_set_use_physical_exposure(desc_camera_set_use_physical_exposure);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_fluid_reset(desc_fluid_reset);

static const MethodParam params_fluid_seed[] = {
    {"domain", "string", true, "", nullptr, nullptr},
    {"seed_min", "vec3", false, "World-space box minimum in metres; omit to derive from the domain", nullptr, nullptr},
    {"seed_max", "vec3", false, "World-space box maximum in metres; omit to derive from the domain", nullptr, nullptr},
    {"particles_per_cell", "int", false, "Particle density per grid cell; 4 is standard", "4", nullptr},
    {"replace", "bool", false, "Clear existing particles first", "true", nullptr},
    {"persistent", "bool", false, "Re-apply this seed after every reset", "false", nullptr},
};
static const MethodDescriptor desc_fluid_seed = {
    "fluid.seed", "fluid",
    "Fill a liquid domain with particles - a given box, or the domain's lower half",
    "Omit seed_min and seed_max to fill the bottom half of the domain; there is no fixed default box any more, because the old one was derived from nothing and seeded zero particles in any domain that did not contain it. A region that does not overlap the domain is now refused with both boxes in the error, instead of succeeding having created nothing. particles_per_cell of 4 is the standard APIC density; a thin jet seeded below that has no interior cells and therefore no pressure field, which reads as 'water that will not splash'. persistent=true re-seeds the box every reset.",
    "write", "SceneWrite", false, "any",
    "fluid|seed|simulation|liquid|water|fill|particles|initial",
    "fluid.create_domain|fluid.clear|flow_source.create",
    nullptr, nullptr, "fluid.get", nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    "fluid.create_domain", "timeline.set_frame", "fluid.get|render.probe", "simulation_cache",
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    "gas.create_domain", "gas.set_shader|flow_source.create|timeline.set_frame", "gas.get_settings|render.probe", "simulation_cache",
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_gas_structural_impulse_stats(desc_gas_structural_impulse_stats);

static const MethodParam params_geometry_cache_bake[] = {
    {"object_name", "string", true, "Flat mesh / Geometry Graph host object", nullptr, nullptr},
    {"start_frame", "int", true, "First sampled timeline frame", nullptr, nullptr},
    {"end_frame", "int", true, "Last sampled timeline frame", nullptr, nullptr},
    {"frame_step", "int", false, "Distance between stored samples", "1", nullptr},
};
static const MethodDescriptor desc_geometry_cache_bake = {
    "geometry_cache.bake", "geometry_cache",
    "Bake a fixed-topology mesh deformation clip from the canonical evaluated geometry",
    "Stores only local vertex positions. Bake is rejected if vertex count or index connectivity changes.",
    "write", "SceneWrite", false, "any",
    "geometry_cache|geometry|cache|bake|animation|deformation|vertex",
    "geometry_cache.status|geometry_cache.set_enabled|geometry_cache.clear|nodes.apply",
    nullptr, nullptr, nullptr, nullptr,
    params_geometry_cache_bake, 4,
    true
};
static const MethodRegistration reg_geometry_cache_bake(desc_geometry_cache_bake);

static const MethodParam params_geometry_cache_clear[] = {
    {"object_name", "string", true, "Cached flat mesh object", nullptr, nullptr},
};
static const MethodDescriptor desc_geometry_cache_clear = {
    "geometry_cache.clear", "geometry_cache",
    "Remove a baked deformation clip and restore live Geometry Graph evaluation",
    nullptr,
    "write", "SceneWrite", false, "any",
    "geometry_cache|geometry|cache|clear|animation",
    "geometry_cache.bake|geometry_cache.status",
    nullptr, nullptr, nullptr, nullptr,
    params_geometry_cache_clear, 1,
    true
};
static const MethodRegistration reg_geometry_cache_clear(desc_geometry_cache_clear);

static const MethodDescriptor desc_geometry_cache_self_test = {
    "geometry_cache.self_test", "geometry_cache",
    "Run fixed-topology position capture, interpolation and memory-accounting tests",
    nullptr,
    "read", "Read", false, "any",
    "geometry_cache|geometry|cache|self|test|animation|selftest|validation",
    "geometry_cache.bake|geometry_cache.status",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_geometry_cache_self_test(desc_geometry_cache_self_test);

static const MethodParam params_geometry_cache_set_enabled[] = {
    {"object_name", "string", true, "Cached flat mesh object", nullptr, nullptr},
    {"enabled", "bool", true, "Use cached positions when true", nullptr, nullptr},
};
static const MethodDescriptor desc_geometry_cache_set_enabled = {
    "geometry_cache.set_enabled", "geometry_cache",
    "Switch an existing geometry cache between cached and live procedural playback",
    nullptr,
    "write", "SceneWrite", false, "any",
    "geometry_cache|geometry|cache|set|enabled|animation|playback|toggle",
    "geometry_cache.status|geometry_cache.bake",
    nullptr, nullptr, nullptr, nullptr,
    params_geometry_cache_set_enabled, 2,
    true
};
static const MethodRegistration reg_geometry_cache_set_enabled(desc_geometry_cache_set_enabled);

static const MethodParam params_geometry_cache_status[] = {
    {"object_name", "string", true, "Cached flat mesh object", nullptr, nullptr},
};
static const MethodDescriptor desc_geometry_cache_status = {
    "geometry_cache.status", "geometry_cache",
    "Inspect deformation-cache range, memory, topology validity and source staleness",
    nullptr,
    "read", "Read", false, "any",
    "geometry_cache|geometry|cache|status|animation|memory",
    "geometry_cache.bake|geometry_cache.set_enabled|geometry_cache.clear",
    nullptr, nullptr, nullptr, nullptr,
    params_geometry_cache_status, 1,
    true
};
static const MethodRegistration reg_geometry_cache_status(desc_geometry_cache_status);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_hair_comb, 4,
    true
};
static const MethodRegistration reg_hair_comb(desc_hair_comb);

static const MethodParam params_hair_create[] = {
    {"mesh", "string", true, "", nullptr, nullptr},
    {"child_radius", "float", false, "", nullptr, nullptr},
    {"children_per_guide", "int", false, "", nullptr, nullptr},
    {"clumpiness", "float", false, "", nullptr, nullptr},
    {"curl_frequency", "float", false, "", nullptr, nullptr},
    {"curl_radius", "float", false, "", nullptr, nullptr},
    {"force_influence", "float", false, "", nullptr, nullptr},
    {"frizz", "float", false, "", nullptr, nullptr},
    {"gravity", "float", false, "", nullptr, nullptr},
    {"guide_count", "int", false, "", nullptr, nullptr},
    {"length", "float", false, "", nullptr, nullptr},
    {"length_variation", "float", false, "", nullptr, nullptr},
    {"name", "string", false, "", "HairGroom", nullptr},
    {"physics_damping", "float", false, "", nullptr, nullptr},
    {"physics_mass", "float", false, "", nullptr, nullptr},
    {"physics_stiffness", "float", false, "", nullptr, nullptr},
    {"points_per_strand", "int", false, "", nullptr, nullptr},
    {"root_radius", "float", false, "", nullptr, nullptr},
    {"roughness", "float", false, "", nullptr, nullptr},
    {"subdivisions", "int", false, "", nullptr, nullptr},
    {"tip_radius", "float", false, "", nullptr, nullptr},
    {"use_bspline", "bool", false, "", nullptr, nullptr},
    {"use_dynamics", "bool", false, "", nullptr, nullptr},
    {"use_tangent_shading", "bool", false, "", nullptr, nullptr},
    {"wave_amplitude", "float", false, "", nullptr, nullptr},
    {"wave_frequency", "float", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_hair_create = {
    "hair.create", "hair",
    "Create a hair groom bound to a mesh",
    nullptr,
    "write", "SceneWrite", false, "any",
    "hair|create|fur|grass",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_hair_create, 26,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_hair_trim, 2,
    true
};
static const MethodRegistration reg_hair_trim(desc_hair_trim);

static const MethodParam params_hair_update[] = {
    {"name", "string", true, "", nullptr, nullptr},
    {"child_radius", "float", false, "", nullptr, nullptr},
    {"children_per_guide", "int", false, "", nullptr, nullptr},
    {"clumpiness", "float", false, "", nullptr, nullptr},
    {"curl_frequency", "float", false, "", nullptr, nullptr},
    {"curl_radius", "float", false, "", nullptr, nullptr},
    {"force_influence", "float", false, "", nullptr, nullptr},
    {"frizz", "float", false, "", nullptr, nullptr},
    {"gravity", "float", false, "", nullptr, nullptr},
    {"guide_count", "int", false, "", nullptr, nullptr},
    {"length", "float", false, "", nullptr, nullptr},
    {"length_variation", "float", false, "", nullptr, nullptr},
    {"physics_damping", "float", false, "", nullptr, nullptr},
    {"physics_mass", "float", false, "", nullptr, nullptr},
    {"physics_stiffness", "float", false, "", nullptr, nullptr},
    {"points_per_strand", "int", false, "", nullptr, nullptr},
    {"root_radius", "float", false, "", nullptr, nullptr},
    {"roughness", "float", false, "", nullptr, nullptr},
    {"subdivisions", "int", false, "", nullptr, nullptr},
    {"tip_radius", "float", false, "", nullptr, nullptr},
    {"use_bspline", "bool", false, "", nullptr, nullptr},
    {"use_dynamics", "bool", false, "", nullptr, nullptr},
    {"use_tangent_shading", "bool", false, "", nullptr, nullptr},
    {"visible", "any", false, "", nullptr, nullptr},
    {"wave_amplitude", "float", false, "", nullptr, nullptr},
    {"wave_frequency", "float", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_hair_update = {
    "hair.update", "hair",
    "Update hair groom settings, keeping what you do not send",
    nullptr,
    "write", "SceneWrite", false, "any",
    "hair|update|configure",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_hair_update, 26,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_lights_set_intensity, 2,
    true
};
static const MethodRegistration reg_lights_set_intensity(desc_lights_set_intensity);

static const MethodParam params_lights_set_param[] = {
    {"param", "string", true, "Which light property to set. Not every key applies to every light type - a key the light does not have is refused, not ignored.", nullptr, "intensity|radius|width|height|spot_angle|spot_falloff"},
    {"index", "int", true, "", nullptr, nullptr},
    {"value", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_lights_set_param = {
    "lights.set_param", "lights",
    "Set one numeric light parameter by name (radius, spot_angle, spot_falloff, width, height)",
    "spot_angle and spot_falloff apply to spot lights only; width and height to area lights only. The call fails rather than silently ignoring a mismatch.",
    "write", "SceneWrite", false, "any",
    "lights|set|param|lighting|softness|cone",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_material_get, 2,
    true
};
static const MethodRegistration reg_material_get(desc_material_get);

static const MethodParam params_material_get_param[] = {
    {"material_name", "string", true, "", nullptr, nullptr},
    {"param", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_material_get_param = {
    "material.get_param", "material",
    "Read one material parameter directly by material name (not through an object)",
    "material.get reads through an OBJECT and averages nothing but still names an object; this names the material ASSET, which is what you want when several materials share one flat mesh (e.g. a carafe's glass body + brass cap + rubber foot) and you need the one that actually has that name.",
    "read", "Read", false, "any",
    "material|get|param|shading",
    "material.set_param|material.get|material.info",
    nullptr, nullptr, nullptr, nullptr,
    params_material_get_param, 2,
    true
};
static const MethodRegistration reg_material_get_param(desc_material_get_param);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_material_of_object, 1,
    true
};
static const MethodRegistration reg_material_of_object(desc_material_of_object);

static const MethodParam params_material_set[] = {
    {"param", "string", true, "Which Principled BSDF property to set. Colour keys (base_color, emission, resin_color, dust_color_a/b, resin_dirt_color) need an RGB value; the rest need a scalar. roughness, metallic, specular, transmission and opacity are clamped to [0,1]; ior to [1,10].", nullptr, "base_color|bubble_film|bubble_ior|dust_color_a|dust_color_b|dust_style|emission|emission_strength|ior|is_bubble|metallic|opacity|resin_color|resin_density|resin_dirt|resin_dirt_color|resin_inclusion|resin_inclusion_scale|resin_object_space|resin_roughness|resin_shard|resin_shard_hue|roughness|shard_shape|specular|transmission|uv_offset_x|uv_offset_y|uv_scale_x|uv_scale_y"},
    {"object_name", "string", true, "", nullptr, nullptr},
    {"value", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_material_set = {
    "material.set", "material",
    "Set one material parameter on an object (base_color, roughness, metallic, emission, ior, transmission, ...)",
    "Colour parameters take a 3-element array, scalars take a number. An unknown parameter name is an error, not a silent no-op.",
    "write", "SceneWrite", false, "any",
    "material|set|shading|colour|roughness|metallic|emission",
    "material.get|material.info",
    nullptr, nullptr, nullptr, nullptr,
    params_material_set, 3,
    true
};
static const MethodRegistration reg_material_set(desc_material_set);

static const MethodParam params_material_set_param[] = {
    {"param", "string", true, "Which Principled BSDF property to set. Colour keys (base_color, emission, resin_color, dust_color_a/b, resin_dirt_color) need an RGB value; the rest need a scalar. roughness, metallic, specular, transmission and opacity are clamped to [0,1]; ior to [1,10].", nullptr, "base_color|bubble_film|bubble_ior|dust_color_a|dust_color_b|dust_style|emission|emission_strength|ior|is_bubble|metallic|opacity|resin_color|resin_density|resin_dirt|resin_dirt_color|resin_inclusion|resin_inclusion_scale|resin_object_space|resin_roughness|resin_shard|resin_shard_hue|roughness|shard_shape|specular|transmission|uv_offset_x|uv_offset_y|uv_scale_x|uv_scale_y"},
    {"material_name", "string", true, "", nullptr, nullptr},
    {"value", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_material_set_param = {
    "material.set_param", "material",
    "Set one material parameter directly by material name (base_color, roughness, metallic, emission, ior, transmission, ...)",
    "material.set edits through an OBJECT: every Principled BSDF that object's flat meshes reference gets the same value, which is wrong the instant one object carries several DIFFERENT materials — a carafe's glass body + brass cap + rubber foot is one flat mesh, three materials, and object.set would smear transmission=1 onto the cap and foot too. This edits exactly the named material asset and nothing else that happens to share its object. Colour parameters take a 3-element array, scalars take a number; an unknown parameter name is an error, not a silent no-op. Not undoable, unlike material.set.",
    "write", "SceneWrite", false, "any",
    "material|set|param|shading|colour|roughness|metallic|emission",
    "material.get_param|material.set|material.set_texture",
    nullptr, nullptr, nullptr, nullptr,
    params_material_set_param, 3,
    true
};
static const MethodRegistration reg_material_set_param(desc_material_set_param);

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
    nullptr, nullptr, nullptr, nullptr,
    params_material_set_texture, 3,
    true
};
static const MethodRegistration reg_material_set_texture(desc_material_set_texture);

static const MethodParam params_material_textures[] = {
    {"material_name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_material_textures = {
    "material.textures", "material",
    "List a material's populated texture slots with the identity of each bound texture",
    "Returns {slot, texture} pairs. 'texture' is the texture's identity - a file path, or an 'embedded_<import>_<slot>_<type>' cache key for a texture that lived inside the model file. Two materials reporting the SAME texture share one decoded image: intended when a map is reused, a defect when the two materials came from different source files.",
    "read", "Read", false, "any",
    "material|textures|shading|texture",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_material_textures, 1,
    true
};
static const MethodRegistration reg_material_textures(desc_material_textures);

static const MethodParam params_mesh_asset_validate[] = {
    {"object", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_mesh_asset_validate = {
    "mesh.asset.validate", "mesh",
    nullptr,
    nullptr,
    "read", "Read", false, "any",
    "mesh|asset|validate",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_mesh_asset_validate, 1,
    false
};
static const MethodRegistration reg_mesh_asset_validate(desc_mesh_asset_validate);

static const MethodParam params_mesh_operation_commit_positions[] = {
    {"object", "string", false, "", "", nullptr},
    {"positions", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_mesh_operation_commit_positions = {
    "mesh.operation.commit_positions", "mesh",
    nullptr,
    nullptr,
    "write", "SceneWrite", false, "any",
    "mesh|operation|commit|positions",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_mesh_operation_commit_positions, 2,
    false
};
static const MethodRegistration reg_mesh_operation_commit_positions(desc_mesh_operation_commit_positions);

static const MethodParam params_mesh_operation_plan[] = {
    {"backend", "string", false, "", "auto", nullptr},
    {"object", "string", false, "", "", nullptr},
    {"tool", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_mesh_operation_plan = {
    "mesh.operation.plan", "mesh",
    nullptr,
    nullptr,
    "read", "Read", false, "any",
    "mesh|operation|plan",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_mesh_operation_plan, 3,
    false
};
static const MethodRegistration reg_mesh_operation_plan(desc_mesh_operation_plan);

static const MethodDescriptor desc_mesh_operation_self_test = {
    "mesh.operation.self_test", "mesh",
    nullptr,
    nullptr,
    "read", "Read", false, "any",
    "mesh|operation|self|test",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    false
};
static const MethodRegistration reg_mesh_operation_self_test(desc_mesh_operation_self_test);

static const MethodParam params_mesh_profile_loft_commit[] = {
    {"object", "string", true, "Destination mesh object name", "", nullptr},
};
static const MethodDescriptor desc_mesh_profile_loft_commit = {
    "mesh.profile.loft.commit", "mesh",
    "Commit an undoable flat DNA SoA mesh generated from multiple spline sections",
    nullptr,
    "write", "SceneWrite", true, "any",
    "mesh|profile|loft|commit|spline",
    "mesh.profile.loft.preview|mesh.profile.loft.self_test",
    nullptr, nullptr, nullptr, nullptr,
    params_mesh_profile_loft_commit, 1,
    true
};
static const MethodRegistration reg_mesh_profile_loft_commit(desc_mesh_profile_loft_commit);

static const MethodDescriptor desc_mesh_profile_loft_preview = {
    "mesh.profile.loft.preview", "mesh",
    "Preview a loft between multiple closed spline sections",
    nullptr,
    "read", "Read", false, "any",
    "mesh|profile|loft|preview|spline",
    "mesh.profile.loft.commit|mesh.profile.loft.self_test",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_mesh_profile_loft_preview(desc_mesh_profile_loft_preview);

static const MethodDescriptor desc_mesh_profile_loft_self_test = {
    "mesh.profile.loft.self_test", "mesh",
    "Run the deterministic profile loft core self-test",
    nullptr,
    "read", "Read", false, "any",
    "mesh|profile|loft|self|test",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_mesh_profile_loft_self_test(desc_mesh_profile_loft_self_test);

static const MethodParam params_mesh_profile_revolve_commit[] = {
    {"object", "string", false, "Destination mesh object name", "", nullptr},
};
static const MethodDescriptor desc_mesh_profile_revolve_commit = {
    "mesh.profile.revolve.commit", "mesh",
    "Commit an undoable partial or full revolve mesh",
    "Uses the same open/closed profile, axis and angle contract as preview.",
    "write", "SceneWrite", true, "any",
    "mesh|profile|revolve|commit|screw|axis",
    "mesh.profile.revolve.preview|mesh.profile.revolve.self_test",
    nullptr, nullptr, nullptr, nullptr,
    params_mesh_profile_revolve_commit, 1,
    true
};
static const MethodRegistration reg_mesh_profile_revolve_commit(desc_mesh_profile_revolve_commit);

static const MethodDescriptor desc_mesh_profile_revolve_preview = {
    "mesh.profile.revolve.preview", "mesh",
    "Preview a partial or full revolve around a selected axis",
    "The radial side profile may be open or closed. start_angle/end_angle are radians; partial ranges keep their angular seams open.",
    "read", "Read", false, "any",
    "mesh|profile|revolve|preview|screw|axis",
    "mesh.profile.revolve.commit|mesh.profile.revolve.self_test",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_mesh_profile_revolve_preview(desc_mesh_profile_revolve_preview);

static const MethodDescriptor desc_mesh_profile_revolve_self_test = {
    "mesh.profile.revolve.self_test", "mesh",
    "Run full closed and partial open-profile revolve core tests",
    nullptr,
    "read", "Read", false, "any",
    "mesh|profile|revolve|self|test|screw",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_mesh_profile_revolve_self_test(desc_mesh_profile_revolve_self_test);

static const MethodParam params_mesh_profile_sweep_commit[] = {
    {"object", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_mesh_profile_sweep_commit = {
    "mesh.profile.sweep.commit", "mesh",
    nullptr,
    nullptr,
    "write", "SceneWrite", false, "any",
    "mesh|profile|sweep|commit",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_mesh_profile_sweep_commit, 1,
    false
};
static const MethodRegistration reg_mesh_profile_sweep_commit(desc_mesh_profile_sweep_commit);

static const MethodDescriptor desc_mesh_profile_sweep_preview = {
    "mesh.profile.sweep.preview", "mesh",
    nullptr,
    nullptr,
    "read", "Read", false, "any",
    "mesh|profile|sweep|preview",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    false
};
static const MethodRegistration reg_mesh_profile_sweep_preview(desc_mesh_profile_sweep_preview);

static const MethodDescriptor desc_mesh_profile_sweep_self_test = {
    "mesh.profile.sweep.self_test", "mesh",
    nullptr,
    nullptr,
    "read", "Read", false, "any",
    "mesh|profile|sweep|self|test",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    false
};
static const MethodRegistration reg_mesh_profile_sweep_self_test(desc_mesh_profile_sweep_self_test);

static const MethodParam params_mesh_tools_describe[] = {
    {"tool", "string", false, "", "", nullptr},
};
static const MethodDescriptor desc_mesh_tools_describe = {
    "mesh.tools.describe", "mesh",
    nullptr,
    nullptr,
    "read", "Read", false, "any",
    "mesh|tools|describe",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_mesh_tools_describe, 1,
    false
};
static const MethodRegistration reg_mesh_tools_describe(desc_mesh_tools_describe);

static const MethodParam params_mesh_tools_list[] = {
    {"include_unavailable", "bool", false, "", "false", nullptr},
    {"workspace", "string", false, "", "edit", nullptr},
};
static const MethodDescriptor desc_mesh_tools_list = {
    "mesh.tools.list", "mesh",
    nullptr,
    nullptr,
    "read", "Read", false, "any",
    "mesh|tools|list",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_mesh_tools_list, 2,
    false
};
static const MethodRegistration reg_mesh_tools_list(desc_mesh_tools_list);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_nodes_link, 6,
    true
};
static const MethodRegistration reg_nodes_link(desc_nodes_link);

static const MethodParam params_nodes_link_by_key[] = {
    {"from_output", "string", true, "", nullptr, nullptr},
    {"graph_name", "string", true, "", nullptr, nullptr},
    {"graph_type", "string", true, "", nullptr, nullptr},
    {"to_input", "string", true, "", nullptr, nullptr},
    {"from_node", "any", false, "", nullptr, nullptr},
    {"to_node", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_nodes_link_by_key = {
    "nodes.link_by_key", "nodes",
    "Connect nodes using stable input and output port keys",
    nullptr,
    "write", "SceneWrite", false, "any",
    "nodes|link|by|key|graph|connect|wire|port",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_nodes_link_by_key, 6,
    true
};
static const MethodRegistration reg_nodes_link_by_key(desc_nodes_link_by_key);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_nodes_list_params, 3,
    true
};
static const MethodRegistration reg_nodes_list_params(desc_nodes_list_params);

static const MethodParam params_nodes_list_ports[] = {
    {"graph_name", "string", true, "", nullptr, nullptr},
    {"graph_type", "string", true, "", nullptr, nullptr},
    {"node_id", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_nodes_list_ports = {
    "nodes.list_ports", "nodes",
    "List stable node ports, exposure tiers, visibility and connection state",
    nullptr,
    "read", "Read", false, "any",
    "nodes|list|ports|graph",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_nodes_list_ports, 3,
    true
};
static const MethodRegistration reg_nodes_list_ports(desc_nodes_list_ports);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_nodes_set_param, 5,
    true
};
static const MethodRegistration reg_nodes_set_param(desc_nodes_set_param);

static const MethodParam params_nodes_set_port_visible[] = {
    {"direction", "string", true, "", nullptr, nullptr},
    {"graph_name", "string", true, "", nullptr, nullptr},
    {"graph_type", "string", true, "", nullptr, nullptr},
    {"port_key", "string", true, "", nullptr, nullptr},
    {"node_id", "any", false, "", nullptr, nullptr},
    {"visible", "bool", false, "", "true", nullptr},
};
static const MethodDescriptor desc_nodes_set_port_visible = {
    "nodes.set_port_visible", "nodes",
    "Show or hide an optional node port without changing evaluation",
    nullptr,
    "write", "SceneWrite", false, "any",
    "nodes|set|port|visible|graph|ports|configure",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_nodes_set_port_visible, 6,
    true
};
static const MethodRegistration reg_nodes_set_port_visible(desc_nodes_set_port_visible);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_particle_step, 1,
    true
};
static const MethodRegistration reg_particle_step(desc_particle_step);

static const MethodParam params_perf_get[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_perf_get = {
    "perf.get", "perf",
    "Read one timing section by name",
    "Returns found:false when the section has never been recorded. It never answers with zeros: a zeroed timing reads as 'measured, and it was free', which turns a missing measurement into a false one. 'rss_measured' false means the working-set fields are absence rather than zero - per-frame sections (loop.*) skip the working-set query so the instrument does not distort the section it measures. See perf.list for the loop.* section map and for why loop.throttle_sleep must be read before any frame-rate conclusion.",
    "read", "Read", false, "PerfSection",
    "perf|get|performance|timing|profile|measure",
    "perf.list",
    nullptr, nullptr, nullptr, nullptr,
    params_perf_get, 1,
    true
};
static const MethodRegistration reg_perf_get(desc_perf_get);

static const MethodDescriptor desc_perf_list = {
    "perf.list", "perf",
    "List every recorded build/render timing section, newest write first",
    "Sections are named by the code that performs the work. BUILD sections: terrain.graph.evaluate/.height/.aux_outputs/.finalize_mesh, terrain.mesh_fill/.create/.update/.publish_fields, terrain.splat_resize, Renderer::rebuildBVH(...), Renderer::rebuildBackendGeometry(GPU), accel.vulkan_solid.raster_geometry. PER-FRAME sections (added 2026-09-03), all prefixed loop.: loop.frame is the whole main-loop iteration and the others are its parts - loop.events (SDL event pump), loop.ui_draw (building every ImGui panel), loop.viewport_render (the backend's renderProgressive; loop.viewport_render_cpu for the legacy CPU path), loop.display_post (original_surface to display surface), loop.imgui_render (building draw lists), loop.present (SDL_RenderClear/Copy/RenderPresent - a vsync wait, if any, lands HERE, so a large value must be read as 'work OR waiting'), loop.throttle_sleep and loop.frame_tail (deferred geometry/BLAS/TLAS rebuilds and the rest of the iteration). WHY THIS EXISTS: before it, only the backend frame and two display copies were timed. Measured on a 695k-triangle interior at 1680x945, the backend frame was 1.2 ms while the loop period was 17.8-26 ms - about 95% of the frame was in code no instrument looked at, so a 142 ms frame could not be decomposed at all. READ loop.throttle_sleep BEFORE CONCLUDING ANYTHING: the main loop deliberately sleeps 16 ms (tier1) or 48 ms (tier2) when the app is not in tier0, and tier0 requires camera_moved, which is set ONLY from mouse/keyboard. An IPC-driven session therefore never reaches tier0, so every frame rate a script measures includes that sleep and is NOT the engine's throughput. Use max_ms to catch the rare slow frame - the mean dissolves it - and call perf.reset first to open a clean window. Each section carries last/total/max ms and a call count. 'rss_measured' says whether the working-set fields mean anything: the loop.* sections set it false because a per-frame scope must not pay for two GetProcessMemoryInfo calls, and their last_rss_delta_mb/rss_after_mb are ABSENCE, not a measurement of zero.",
    "read", "Read", false, "PerfSection[]",
    "perf|list|performance|timing|profile|measure|slow|memory",
    "perf.get|perf.reset|perf.set_logging",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_perf_list(desc_perf_list);

static const MethodDescriptor desc_perf_reset = {
    "perf.reset", "perf",
    "Clear the timing counters before a measured run",
    "Write ordering (seq) stays monotonic across a reset, so a section recorded after it cannot be confused with a stale one. Changes no scene, render or file state - which is why it carries the read capability. Call it before a measured run: max_ms never decays, so one scene-load frame keeps dominating the table until it is cleared.",
    "read", "Read", false, "any",
    "perf|reset|performance|timing|profile",
    "perf.list|perf.get",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_perf_reset(desc_perf_reset);

static const MethodParam params_perf_set_logging[] = {
    {"enabled", "bool", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_perf_set_logging = {
    "perf.set_logging", "perf",
    "Also mirror completed timing sections into the Scene Log",
    "Off by default. The registry is the readable surface - script and IPC callers cannot read the Scene Log, which is exactly why the previous profiler produced no usable numbers.",
    "read", "Read", false, "any",
    "perf|set|logging|performance|timing|profile|log",
    "perf.list",
    nullptr, nullptr, nullptr, nullptr,
    params_perf_set_logging, 1,
    true
};
static const MethodRegistration reg_perf_set_logging(desc_perf_set_logging);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_physics_set_gravity, 1,
    true
};
static const MethodRegistration reg_physics_set_gravity(desc_physics_set_gravity);

static const MethodParam params_physics_step[] = {
    {"dt", "float", false, "", "0.0166667", nullptr},
};
static const MethodDescriptor desc_physics_step = {
    "physics.step", "physics",
    "Advance the physics solver by dt and move the playhead with it",
    "Advances the solver AND the timeline playhead, and claims the timeline until the user scrubs, plays or stops. It used to advance only the solver: the playhead stayed put, the frame loop found a rigid state that disagreed with the displayed frame, and it reset the runtime to the rest pose - erasing every scripted step while this call returned success. Check sim.control_state around a measurement to learn whether the claim was taken back.",
    "write", "SceneWrite", false, "any",
    "physics|step|simulation|advance",
    nullptr,
    nullptr, "scene.get_world_transform", "sim.control_state", nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_physics_unfracture_object, 1,
    true
};
static const MethodRegistration reg_physics_unfracture_object(desc_physics_unfracture_object);

static const MethodParam params_post_configure_exposure[] = {
    {"settings", "object", true, "Exposure patch: mode, ev, min_ev, max_ev, low_percent, high_percent, key, speed_up, speed_down, center_weight, locked, locked_ev", nullptr, nullptr},
};
static const MethodDescriptor desc_post_configure_exposure = {
    "post.configure_exposure", "post",
    "Atomically configure manual, physical-camera or GPU-histogram exposure",
    "UI and Python call the same service. Mode is manual|physical|auto_histogram. EV in [-24,24]; min_ev <= max_ev. 0 <= low_percent < high_percent <= 100. key in [0.001,1], speed_up/down in (0,20] per second, center_weight in [0,1]. Non-finite values and unknown fields fail without mutation. locked=true captures current applied EV unless locked_ev is supplied. Final render freezes adaptation and rejects edits. No accumulation reset.",
    "write", "SceneWrite", false, "object",
    "post|configure|exposure|histogram",
    nullptr,
    nullptr, nullptr, "post.get_exposure", "display",
    params_post_configure_exposure, 1,
    true
};
static const MethodRegistration reg_post_configure_exposure(desc_post_configure_exposure);

static const MethodDescriptor desc_post_get = {
    "post.get", "post",
    "Read the post-processing settings: exposure, gamma, tone mapping, saturation, vignette and stylize",
    nullptr,
    "read", "Read", false, "any",
    "post|get|grade|look",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_post_get(desc_post_get);

static const MethodDescriptor desc_post_get_exposure = {
    "post.get_exposure", "post",
    "Read exposure settings, applied EV and HDR histogram",
    "Relative scene-linear luminance, not photometric EV100. GPU meters use a deterministic 128x128 sample grid and 256 weighted log2 bins. Raster preview shares the last resolved EV; it does not provide an HDR meter. meter_valid and meter_source report available data.",
    "read", "Read", false, "object",
    "post|get|exposure|histogram",
    "post.configure_exposure|post.reset_exposure",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_post_get_exposure(desc_post_get_exposure);

static const MethodDescriptor desc_post_reset_exposure = {
    "post.reset_exposure", "post",
    "Reset exposure histogram and adaptation history",
    "Invalidates in-flight measurements by generation. Does not change settings or the explicit locked EV. Rejected while a final render is active.",
    "write", "SceneWrite", false, "object",
    "post|reset|exposure",
    nullptr,
    nullptr, nullptr, "post.get_exposure", "display",
    nullptr, 0,
    true
};
static const MethodRegistration reg_post_reset_exposure(desc_post_reset_exposure);

static const MethodParam params_post_set_color_temperature[] = {
    {"color_temperature", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_post_set_color_temperature = {
    "post.set_color_temperature", "post",
    "Set scene-linear Bradford white balance",
    "Finite [4000,25000] Kelvin, daylight-locus source white adapted to D65. 6500 K is exactly neutral. Applied before the view transform. Higher Kelvin warms.",
    "write", "SceneWrite", false, "any",
    "post|set|color|temperature|whitebalance",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
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
    "Linear gain in [0,65504]. Multiplies the EV/physical/auto exposure selected by post.configure_exposure. Does not reset sample accumulation.",
    "write", "SceneWrite", false, "any",
    "post|set|exposure|brightness|grade",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_post_set_stylize_strength, 1,
    true
};
static const MethodRegistration reg_post_set_stylize_strength(desc_post_set_stylize_strength);

static const MethodParam params_post_set_tone_mapping[] = {
    {"tone_mapping", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_post_set_tone_mapping = {
    "post.set_tone_mapping", "post",
    "Select AgX, ACES Fitted, Uncharted, Hejl Filmic, Linear or Reinhard",
    "Shared CPU/CUDA/GLSL scene-linear Rec.709 implementation. ACES is a fitted approximation, not a full ACES/OCIO output transform. None/linear clips without tone mapping. Old AGX/ACES looks are intentionally replaced.",
    "write", "SceneWrite", false, "any",
    "post|set|tone|mapping|tonemap|color",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    "write", "Render|FilesWrite", false, "any",
    "render|start|output|image|final|save",
    "render.status|render.cancel|render.start_sequence",
    nullptr, nullptr, nullptr, nullptr,
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
    "write", "Render|FilesWrite", false, "any",
    "render|start|sequence|animation|output|batch",
    "render.sequence_status|render.cancel_sequence",
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_render_volume_stats(desc_render_volume_stats);

static const MethodDescriptor desc_render_volume_tables = {
    "render.volume_tables", "render",
    "Report the published volume SSBO state per backend (render / raster viewport), including whether the simulation compute device is that backend's own device",
    "The raster viewport can be a SECOND VulkanBackendAdapter with its own VkDevice and its own volume table. A 'viewport' row with instance_count 0 while 'render' is non-zero means the realtime viewport was never handed the volumes and will draw none — render.volume_stats cannot see this, because a pass that never ran counts zero exactly like a scene with no volumes. sim_device_is_this_backends false means live dense gas addresses are suppressed for that backend and it falls back to the per-adapter NanoVDB upload (a frozen grid, not a fault).",
    "render", "Render", false, "VolumeTablesInfo",
    "render|volume|tables|diagnostics|backend|verify",
    "render.volume_stats|viewport.status",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_render_volume_tables(desc_render_volume_tables);

static const MethodDescriptor desc_request_render = {
    "request_render", "request_render",
    "Ask the viewport to render another frame",
    nullptr,
    "render", "Render", false, "any",
    "request_render|request|render|viewport|refresh",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_reset_accumulation(desc_reset_accumulation);

static const MethodParam params_scatter_add_library_source[] = {
    {"group", "string", true, "", nullptr, nullptr},
    {"relative_path", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_scatter_add_library_source = {
    "scatter.add_library_source", "scatter",
    nullptr,
    nullptr,
    "write", "SceneWrite", false, "any",
    "scatter|add|library|source",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_scatter_add_library_source, 2,
    false
};
static const MethodRegistration reg_scatter_add_library_source(desc_scatter_add_library_source);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_scatter_fill, 1,
    true
};
static const MethodRegistration reg_scatter_fill(desc_scatter_fill);

static const MethodDescriptor desc_scatter_list_assets = {
    "scatter.list_assets", "scatter",
    nullptr,
    nullptr,
    "read", "Read", false, "any",
    "scatter|list|assets",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    false
};
static const MethodRegistration reg_scatter_list_assets(desc_scatter_list_assets);

static const MethodDescriptor desc_scatter_list_groups = {
    "scatter.list_groups", "scatter",
    "List the scatter groups with their sources, counts and placement settings",
    nullptr,
    "read", "Read", false, "any",
    "scatter|list|groups|instancing|vegetation|inventory",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_scatter_list_groups(desc_scatter_list_groups);

static const MethodParam params_scatter_set_settings[] = {
    {"group", "string", true, "Scatter group id or name", nullptr, nullptr},
    {"target_count", "int", false, "How many instances to aim for on the next fill", nullptr, nullptr},
    {"seed", "int", false, "Placement seed; the same seed reproduces the same layout", nullptr, nullptr},
    {"min_distance", "float", false, "Minimum spacing between instances, in metres", nullptr, nullptr},
    {"slope_max", "float", false, "Reject ground steeper than this, in degrees", nullptr, nullptr},
    {"height_min", "float", false, "Lowest world height that accepts an instance, in metres", nullptr, nullptr},
    {"height_max", "float", false, "Highest world height that accepts an instance, in metres", nullptr, nullptr},
    {"density_mask", "string", false, "INCLUDE field name; its value is a placement PROBABILITY, so 0.3 thins rather than cuts. Empty string turns it off", nullptr, nullptr},
    {"exclusion_mask", "string", false, "EXCLUDE field name; placement is forbidden where the field is >= exclusion_threshold. Empty string turns it off", nullptr, nullptr},
    {"exclusion_threshold", "float", false, "Reject at or above this value; drives BOTH exclusion_mask and splat_exclude_channel", nullptr, nullptr},
    {"scale_mask", "string", false, "Field name that scales instances. Empty string turns it off", nullptr, nullptr},
    {"scale_mask_influence", "float", false, "0 ignores scale_mask, 1 lets it drive scale fully", nullptr, nullptr},
    {"splat_include_channel", "int", false, "Splat map channel used as an include weight: -1 off, 0..3 = RGBA", nullptr, nullptr},
    {"splat_exclude_channel", "int", false, "Splat map channel used as an exclusion: -1 off, 0..3 = RGBA", nullptr, nullptr},
};
static const MethodDescriptor desc_scatter_set_settings = {
    "scatter.set_settings", "scatter",
    "Patch a scatter group's placement rules and its density / exclusion / scale masks",
    "Only the keys you pass are written; the rest are left alone. The two mask roles are NOT symmetric: density_mask INCLUDES probabilistically (its value is a placement probability, so 0.3 thins a stand), while exclusion_mask FORBIDS placement wherever the field is >= exclusion_threshold. exclusion_threshold is shared by exclusion_mask and splat_exclude_channel. Mask names are terrain analysis fields - get the ones a terrain actually has from terrain.list_fields, and build a combined mask upstream with a Publish Field node rather than expecting more mask slots here. splat channels are -1 (off) or 0..3 (RGBA); an out-of-range channel is refused, not clamped. Read the result back with scatter.list_groups.",
    "write", "SceneWrite", false, "any",
    "scatter|set|settings|foliage|mask|density|exclusion|placement",
    "scatter.list_groups|terrain.list_fields|scatter.fill",
    nullptr, nullptr, nullptr, nullptr,
    params_scatter_set_settings, 14,
    true
};
static const MethodRegistration reg_scatter_set_settings(desc_scatter_set_settings);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_scene_duplicate, 1,
    true
};
static const MethodRegistration reg_scene_duplicate(desc_scene_duplicate);

static const MethodParam params_scene_export_estimate[] = {
    {"geometry", "bool", false, "Count meshes", "true", nullptr},
    {"materials", "bool", false, "Count materials and textures", "true", nullptr},
    {"cameras", "bool", false, "Count the active camera as a glTF camera node", "false", nullptr},
    {"lights", "bool", false, "Count scene lights", "false", nullptr},
    {"animations", "bool", false, "Count animation clips", "true", nullptr},
    {"skinning", "bool", false, "Count skinning data", "true", nullptr},
    {"selected_only", "bool", false, "Estimate the current multi-selection only", "false", nullptr},
    {"bake_terrain_materials", "bool", false, "Assume terrain layers get baked", "true", nullptr},
    {"terrain_bake_resolution", "int", false, "Terrain bake resolution assumed by the estimate", "1024", nullptr},
    {"gpu_instancing", "bool", false, "Assume EXT_mesh_gpu_instancing collapses scatter nodes", "true", nullptr},
};
static const MethodDescriptor desc_scene_export_estimate = {
    "scene.export_estimate", "scene",
    "Report what an export WOULD cost, without writing anything",
    "The same pre-export estimate the Export Settings panel shows, computed by the same function - so a script can catch the panel drifting away from the writer. It already did: the writer was fixed to read scatter from InstanceManager (on Vulkan scatter is NEVER expanded into world.objects), the panel's estimate was not, and it reported instances=0 for a scene the exporter then wrote a thousand instances of. Compare `instances` here against scene.export_gltf's measured `instances`; a gap means they have drifted again. Note `triangles` counts non-instanced geometry only - add `instance_triangles` for the total. `estimated_peak_mb` covers only geometry the writer must MATERIALISE (legacy Triangle facades); flat SoA meshes stream from where they live and cost no extra heap.",
    "read", "Read", false, "Object with objects, triangles, legacy_triangles, instances, unique_instance_sources, instance_triangles, materialised_instance_triangles, estimated_peak_mb",
    "scene|export|estimate|gltf|cost|preflight|scatter|instances",
    "scene.export_gltf|scatter.list_groups",
    nullptr, nullptr, nullptr, nullptr,
    params_scene_export_estimate, 10,
    true
};
static const MethodRegistration reg_scene_export_estimate(desc_scene_export_estimate);

static const MethodParam params_scene_export_gltf[] = {
    {"path", "string", true, "Output file path; extension must be .glb or .gltf", nullptr, nullptr},
    {"geometry", "bool", false, "Write meshes", "true", nullptr},
    {"materials", "bool", false, "Write materials and textures", "true", nullptr},
    {"cameras", "bool", false, "Write the active camera as a glTF camera node", "false", nullptr},
    {"lights", "bool", false, "Write scene lights via KHR_lights_punctual", "false", nullptr},
    {"animations", "bool", false, "Write animation clips as glTF samplers/channels", "true", nullptr},
    {"skinning", "bool", false, "Write the skeleton, JOINTS_0/WEIGHTS_0 and inverse bind matrices", "true", nullptr},
    {"selected_only", "bool", false, "Export only the current multi-selection; fails if nothing is selected", "false", nullptr},
    {"bake_terrain_materials", "bool", false, "Flatten splat-blended terrain layers into export textures", "true", nullptr},
    {"terrain_bake_resolution", "int", false, "Square resolution of each baked terrain texture", "1024", nullptr},
    {"gpu_instancing", "bool", false, "Collapse scattered objects sharing one source mesh into EXT_mesh_gpu_instancing nodes", "true", nullptr},
};
static const MethodDescriptor desc_scene_export_gltf = {
    "scene.export_gltf", "scene",
    "Export the scene to glTF 2.0 (.glb or .gltf + sidecar .bin) and report measured cost",
    "Written directly from the in-memory flat SoA geometry - no Assimp round trip, no per-triangle facade. Blocks the main thread until the file is closed. The reply carries what was actually written plus a per-phase time breakdown and the writer's own peak heap (peak_writer_mb, NOT process RSS), so export cost is regressable from a script. .glb is capped at 4 GB by its 32-bit chunk length; use .gltf for anything larger.",
    "write", "FilesWrite", false, "Object with path, meshes, primitives, triangles, vertices, nodes, instances, instanced_groups, materials, images, file_bytes, peak_writer_mb and seconds_total/collect/materials/plan/write",
    "scene|export|gltf|glb|save|interchange|asset",
    "scene.import_model|project.save",
    nullptr, nullptr, nullptr, nullptr,
    params_scene_export_gltf, 11,
    true
};
static const MethodRegistration reg_scene_export_gltf(desc_scene_export_gltf);

static const MethodParam params_scene_get_transform[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_scene_get_transform = {
    "scene.get_transform", "scene",
    "Return an object's translation, rotation, scale and full matrix",
    "This is the AUTHORED transform and a solver never writes it: a rigid body can fall for a thousand steps while this keeps returning the spawn pose. To measure physics use scene.get_world_transform.",
    "read", "Read", false, "TransformInfo",
    "scene|get|transform",
    "scene.set_transform|scene.get_world_transform",
    nullptr, nullptr, nullptr, nullptr,
    params_scene_get_transform, 1,
    true
};
static const MethodRegistration reg_scene_get_transform(desc_scene_get_transform);

static const MethodParam params_scene_get_world_transform[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_scene_get_world_transform = {
    "scene.get_world_transform", "scene",
    "Return an object's SIMULATED world pose: the solver's motion composed onto the authored transform",
    "The rigid solver bakes its motion into the mesh vertices and never writes a transform, so scene.get_transform reports the spawn pose for a body that has been falling for hundreds of steps. Every physics measurement - free fall, buoyancy draft, restitution, settling - has to read this one. The result carries simulated: false when no solver has contributed, which is not the same as the body having stayed still.",
    "read", "Read", false, "any",
    "scene|get|world|transform|physics|simulated|pose|measure|verify",
    "scene.get_transform|physics.step|physics.add_body",
    nullptr, nullptr, "scene.get_transform", nullptr,
    params_scene_get_world_transform, 1,
    true
};
static const MethodRegistration reg_scene_get_world_transform(desc_scene_get_world_transform);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_scene_object_info, 1,
    true
};
static const MethodRegistration reg_scene_object_info(desc_scene_object_info);

static const MethodParam params_scene_raycast[] = {
    {"origin", "vec3", false, "Ray origin in world space", nullptr, nullptr},
    {"direction", "vec3", false, "Ray direction; normalized internally, must be non-zero", nullptr, nullptr},
    {"filter", "string", false, "mesh_and_terrain (default) | terrain_only | ground_plane. An unknown value is REFUSED, not defaulted: silently widening a terrain_only query would report a mesh hit that looks correct", "mesh_and_terrain", nullptr},
};
static const MethodDescriptor desc_scene_raycast = {
    "scene.raycast", "scene",
    "Reports what a ray hits: scene geometry, terrain, or the Y=0 ground plane. This is the VALUE behind the viewport's draw-on-surface tools - without it that workflow is reachable only by clicking, and therefore not testable.",
    "Returns hit, kind (mesh|terrain|ground_plane|none), object (hit mesh name; empty for terrain and the ground plane), position, normal and distance. The mesh and terrain hits are compared by distance, so a mesh behind the terrain does not win.",
    "read", "Read", false, "any",
    "scene|raycast",
    "spline.append_point|terrain.sample_height|scene.list_objects",
    nullptr, nullptr, nullptr, nullptr,
    params_scene_raycast, 3,
    true
};
static const MethodRegistration reg_scene_raycast(desc_scene_raycast);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_sculpt_mask_operation, 4,
    true
};
static const MethodRegistration reg_sculpt_mask_operation(desc_sculpt_mask_operation);

static const MethodParam params_sculpt_paint_mask[] = {
    {"object", "string", true, "", nullptr, nullptr},
    {"radius", "float", true, "", nullptr, nullptr},
    {"value", "float", true, "", nullptr, nullptr},
    {"points", "any", false, "", nullptr, nullptr},
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
    nullptr, nullptr, nullptr, nullptr,
    params_sculpt_paint_mask, 6,
    true
};
static const MethodRegistration reg_sculpt_paint_mask(desc_sculpt_paint_mask);

static const MethodParam params_sculpt_stroke[] = {
    {"object", "string", true, "", nullptr, nullptr},
    {"tool", "string", true, "", nullptr, nullptr},
    {"points", "array", true, "Stroke path: list of [x, y, z] world points. Required - a stroke with no points does nothing.", nullptr, nullptr},
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
    nullptr, nullptr, nullptr, nullptr,
    params_sculpt_stroke, 10,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_select_object, 2,
    true
};
static const MethodRegistration reg_select_object(desc_select_object);

static const MethodDescriptor desc_sim_control_state = {
    "sim.control_state", "sim",
    "Who is currently driving the solvers, and an epoch that changes whenever anything re-poses them",
    "Read epoch before and after a measurement. If it changed, the solvers were re-posed under you - the user scrubbed, playback ran, or a reset fired - and the numbers you just read are not a physics result. Without this a reverted pose and a body that never moved read exactly the same. driver says which of those it was; script_driving is true while physics.step holds the timeline, which it keeps only until the user scrubs, plays or stops. dropped_seeds names any fluid domain whose one-shot fluid.seed() (no persistent=true) the most recent auto-reset wiped and did NOT refill -- that auto-reset fires from a config-signature change ANYWHERE in the scene (e.g. an unrelated fluid.create_domain), so a domain's particles can go to zero with no error at all; check this field when that happens instead of assuming a reader fault.",
    "read", "Read", false, "any",
    "sim|control|state|simulation|epoch|driver|timeline|measure|verify|fluid|seed",
    "physics.step|timeline.set_frame|timeline.get_frame|scene.get_world_transform|fluid.seed|fluid.create_domain",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_sim_control_state(desc_sim_control_state);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_sim_graph_apply, 3,
    true
};
static const MethodRegistration reg_sim_graph_apply(desc_sim_graph_apply);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_sim_graph_connect, 6,
    true
};
static const MethodRegistration reg_sim_graph_connect(desc_sim_graph_connect);

static const MethodDescriptor desc_sim_graph_couplings = {
    "sim_graph.couplings", "sim_graph",
    "Report declared versus actually running couplings between simulation domains",
    "declared_not_running and running_not_declared are the two ways a graph and the live solvers can disagree.",
    "read", "Read", false, "any",
    "sim_graph|sim|graph|couplings|simulation|nodes|coupling|verify|diagnostics",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_sim_graph_delete, 2,
    true
};
static const MethodRegistration reg_sim_graph_delete(desc_sim_graph_delete);

static const MethodParam params_sim_graph_domain_intersections[] = {
    {"domain", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_sim_graph_domain_intersections = {
    "sim_graph.domain_intersections", "sim_graph",
    "Report which force fields and colliders geometrically overlap a domain's box",
    "A MEASUREMENT, not a declared link - neither force fields nor colliders carry a domain field, because force is spatial. Bounding-volume overlap, not exact clipping: an obb collider's box ignores rotation (over-reports only, never hides a real intersection); mesh_sdf/convex/mesh_bvh colliders report measurable=false rather than a guessed non-intersection.",
    "read", "Read", false, "any",
    "sim_graph|sim|graph|domain|intersections|simulation|nodes|forcefield|collider|measure|diagnostics",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_sim_graph_domain_intersections, 1,
    true
};
static const MethodRegistration reg_sim_graph_domain_intersections(desc_sim_graph_domain_intersections);

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
    nullptr, nullptr, nullptr, nullptr,
    params_sim_graph_evaluate, 2,
    true
};
static const MethodRegistration reg_sim_graph_evaluate(desc_sim_graph_evaluate);

static const MethodDescriptor desc_sim_graph_list = {
    "sim_graph.list", "sim_graph",
    "List the simulation node graphs with their scope, owner and whether the owner still exists",
    "owner_missing means the graph outlived the entity it was written for - it will not run.",
    "read", "Read", false, "any",
    "sim_graph|sim|graph|list|simulation|nodes|inventory|scope",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_sim_graph_set_node_value, 5,
    true
};
static const MethodRegistration reg_sim_graph_set_node_value(desc_sim_graph_set_node_value);

static const MethodDescriptor desc_spline_animation_self_test = {
    "spline.animation.self_test", "spline",
    "Run the deterministic spline transform, control-point and radius interpolation self-test",
    nullptr,
    "read", "Read", false, "{ok:bool,details:string}",
    "spline|animation|self|test|selftest|validation",
    "spline.keyframe.insert|spline.keyframe.list",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_spline_animation_self_test(desc_spline_animation_self_test);

static const MethodParam params_spline_append_point[] = {
    {"name", "string", true, "Spline object name, as reported by spline.list", nullptr, nullptr},
    {"position", "vec3", false, "Point position in the spline's LOCAL space", nullptr, nullptr},
};
static const MethodDescriptor desc_spline_append_point = {
    "spline.append_point", "spline",
    "Appends a control point at the tail of an open spline, valid from ZERO points. spline.extrude needs two existing points to derive a tangent, so it can extend a curve but never build one.",
    "Returns the inserted index. Pair with scene.raycast to reproduce the viewport draw-on-surface flow from script. Keyframe topology propagation runs only once the curve has more than two points, because the first two create the curve rather than change its shape.",
    "write", "SceneWrite", false, "any",
    "spline|append|point",
    "spline.create|spline.extrude|scene.raycast|spline.get",
    nullptr, nullptr, nullptr, nullptr,
    params_spline_append_point, 2,
    true
};
static const MethodRegistration reg_spline_append_point(desc_spline_append_point);

static const MethodParam params_spline_create[] = {
    {"primitive", "string", false, "Initial spline shape", "open_line", "circle|rectangle|open_line|open_arc"},
    {"name", "string", false, "Requested object name; made unique when necessary", "Spline", nullptr},
    {"plane", "string", false, "Authoring plane", "xy", "xy|xz|yz"},
};
static const MethodDescriptor desc_spline_create = {
    "spline.create", "spline",
    "Create an editable spline source through the canonical scene-object service",
    "Returns the unique scene name. The operation is undoable and produces the same SplineObject used by the viewport UI, Python and Geometry Nodes. primitive also accepts 'empty' (no points) and plane also accepts 'free' - the pair that starts a drawn route, since a profile is planar but a path is not.",
    "write", "SceneWrite", true, "string",
    "spline|create|curve|authoring|undo",
    "spline.list|spline.get|scene.delete|nodes.create_graph",
    nullptr, nullptr, nullptr, nullptr,
    params_spline_create, 3,
    true
};
static const MethodRegistration reg_spline_create(desc_spline_create);

static const MethodParam params_spline_extrude[] = {
    {"name", "string", true, "Spline object name", nullptr, nullptr},
    {"endpoint", "int", true, "0 for first endpoint or last point index", nullptr, nullptr},
    {"position", "float[3]", true, "New control point position", nullptr, nullptr},
};
static const MethodDescriptor desc_spline_extrude = {
    "spline.extrude", "spline",
    "Extrude an open spline endpoint to a new control point",
    "Only endpoint 0 or the final point is accepted, and closed splines are rejected.",
    "write", "SceneWrite", false, "{index:int}",
    "spline|extrude|endpoint|profile",
    "spline.get|spline.insert_point",
    nullptr, nullptr, nullptr, nullptr,
    params_spline_extrude, 3,
    true
};
static const MethodRegistration reg_spline_extrude(desc_spline_extrude);

static const MethodParam params_spline_get[] = {
    {"name", "string", true, "Spline object name", nullptr, nullptr},
};
static const MethodDescriptor desc_spline_get = {
    "spline.get", "spline",
    "Read the versioned JSON authoring payload of a spline",
    "The payload includes curve type, plane, closed state, render transform, pivot_offset, optional cubic B-Spline knots and all point/handle data.",
    "read", "Read", false, "SplinePayload",
    "spline|get|serialize|profile|curve",
    "spline.set|spline.list",
    nullptr, nullptr, nullptr, nullptr,
    params_spline_get, 1,
    true
};
static const MethodRegistration reg_spline_get(desc_spline_get);

static const MethodParam params_spline_insert_point[] = {
    {"name", "string", true, "Spline object name", nullptr, nullptr},
    {"segment", "int", true, "Zero-based segment start index", nullptr, nullptr},
    {"t", "float", true, "Segment parameter in [0,1]", "0.5", nullptr},
};
static const MethodDescriptor desc_spline_insert_point = {
    "spline.insert_point", "spline",
    "Insert a control point on a spline segment at normalized parameter t",
    "Bezier insertion preserves the curve with De Casteljau; Linear insertion splits the segment; cubic B-Spline insertion uses shape-preserving Boehm knot insertion.",
    "write", "SceneWrite", false, "{index:int}",
    "spline|insert|point|subdivide|bezier|linear",
    "spline.subdivide|spline.get",
    nullptr, nullptr, nullptr, nullptr,
    params_spline_insert_point, 3,
    true
};
static const MethodRegistration reg_spline_insert_point(desc_spline_insert_point);

static const MethodParam params_spline_keyframe_insert[] = {
    {"name", "string", true, "Spline object name", nullptr, nullptr},
    {"frame", "int", true, "Non-negative timeline frame", nullptr, nullptr},
    {"object_transform", "bool", false, "Capture the spline object's location, rotation and scale", "True", nullptr},
    {"points", "bool", false, "Capture all control positions, handles, radius/user data and colors", "True", nullptr},
};
static const MethodDescriptor desc_spline_keyframe_insert = {
    "spline.keyframe.insert", "spline",
    "Capture spline object transform and control-point deformation at a timeline frame",
    "Point keys include Linear, Bezier and B-Spline controls, Bezier handles, radius/user data and color. Point count, curve type, open/closed state and knot topology must remain stable between keys.",
    "write", "SceneWrite", false, "bool",
    "spline|keyframe|insert|animation|deform|hose|cable|radius",
    "spline.keyframe.list|spline.keyframe.remove|timeline.set_frame|spline.get",
    nullptr, nullptr, nullptr, nullptr,
    params_spline_keyframe_insert, 4,
    true
};
static const MethodRegistration reg_spline_keyframe_insert(desc_spline_keyframe_insert);

static const MethodParam params_spline_keyframe_list[] = {
    {"name", "string", true, "Spline object name", nullptr, nullptr},
};
static const MethodDescriptor desc_spline_keyframe_list = {
    "spline.keyframe.list", "spline",
    "List spline animation keys and the channels captured at each frame",
    nullptr,
    "read", "Read", false, "SplineKeyInfo[]",
    "spline|keyframe|list|animation",
    "spline.keyframe.insert|spline.keyframe.remove|timeline.set_frame",
    nullptr, nullptr, nullptr, nullptr,
    params_spline_keyframe_list, 1,
    true
};
static const MethodRegistration reg_spline_keyframe_list(desc_spline_keyframe_list);

static const MethodParam params_spline_keyframe_remove[] = {
    {"name", "string", true, "Spline object name", nullptr, nullptr},
    {"frame", "int", true, "Timeline frame containing the key", nullptr, nullptr},
    {"object_transform", "bool", false, "Remove the object transform portion", "True", nullptr},
    {"points", "bool", false, "Remove the control-point deformation portion", "True", nullptr},
};
static const MethodDescriptor desc_spline_keyframe_remove = {
    "spline.keyframe.remove", "spline",
    "Remove spline transform and/or point channels from one timeline key",
    "Other keyframe domains stored at the same frame are preserved.",
    "write", "SceneWrite", false, "bool",
    "spline|keyframe|remove|animation",
    "spline.keyframe.insert|spline.keyframe.list",
    nullptr, nullptr, nullptr, nullptr,
    params_spline_keyframe_remove, 4,
    true
};
static const MethodRegistration reg_spline_keyframe_remove(desc_spline_keyframe_remove);

static const MethodDescriptor desc_spline_list = {
    "spline.list", "spline",
    "List editable 2D spline sources and their curve types",
    "Returns names, plane, closed state and control-point counts. This is a read-only scene query.",
    "read", "Read", false, "SplineInfo[]",
    "spline|list|curve|profile|control-point|authoring",
    "spline.get|spline.set|spline.insert_point|spline.subdivide|spline.extrude",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_spline_list(desc_spline_list);

static const MethodParam params_spline_set[] = {
    {"name", "string", true, "Spline object name", nullptr, nullptr},
    {"spline", "object", true, "Versioned spline payload object", nullptr, nullptr},
};
static const MethodDescriptor desc_spline_set = {
    "spline.set", "spline",
    "Replace a spline authoring payload with validated JSON data",
    "Rejects unknown curve types, malformed points, invalid transforms, pivot_offset or knot vectors, and B-Splines with fewer than four controls.",
    "write", "SceneWrite", false, "bool",
    "spline|set|serialize|edit|profile",
    "spline.get|spline.insert_point|spline.subdivide",
    nullptr, nullptr, nullptr, nullptr,
    params_spline_set, 2,
    true
};
static const MethodRegistration reg_spline_set(desc_spline_set);

static const MethodParam params_spline_skin_clear[] = {
    {"spline", "string", true, "Spline source owning the live skin display", nullptr, nullptr},
};
static const MethodDescriptor desc_spline_skin_clear = {
    "spline.skin.clear", "spline",
    "Remove a spline's non-destructive skin display",
    "Deletes only the linked preview host and graph; the editable spline source remains unchanged.",
    "write", "SceneWrite", false, "any",
    "spline|skin|clear|preview|remove",
    "spline.skin.create|spline.skin.finalize",
    nullptr, nullptr, nullptr, nullptr,
    params_spline_skin_clear, 1,
    true
};
static const MethodRegistration reg_spline_skin_clear(desc_spline_skin_clear);

static const MethodParam params_spline_skin_create[] = {
    {"spline", "string", true, "Open spline path object", nullptr, nullptr},
    {"output", "string", false, "Requested output mesh name; blank derives one from the spline", "", nullptr},
    {"radius", "float", false, "Global bevel radius", "0.1", nullptr},
    {"path_samples", "int", false, "Samples along the spline", "48", nullptr},
    {"radial_segments", "int", false, "Circular cross-section segments", "12", nullptr},
    {"cap_start", "bool", false, "Cap the first endpoint", "True", nullptr},
    {"cap_end", "bool", false, "Cap the final endpoint", "True", nullptr},
    {"use_point_radius", "bool", false, "Multiply bevel radius by interpolated per-point Curve Radius", "True", nullptr},
    {"custom_profile", "string", false, "Optional closed spline used as the swept cross-section", "", nullptr},
    {"taper_start", "float", false, "Radius scale at the first control", "1.0", nullptr},
    {"taper_end", "float", false, "Radius scale at the final control", "1.0", nullptr},
    {"taper_falloff", "float", false, "Taper interpolation exponent", "1.0", nullptr},
    {"twist_start_degrees", "float", false, "Profile rotation at the first control", "0.0", nullptr},
    {"twist_end_degrees", "float", false, "Profile rotation at the final control", "0.0", nullptr},
    {"wave_amplitude", "float", false, "Wave displacement amplitude", "0.0", nullptr},
    {"wave_cycles", "float", false, "Wave cycles along the controls", "1.0", nullptr},
    {"wave_phase_degrees", "float", false, "Wave phase in degrees", "0.0", nullptr},
    {"wave_noise", "float", false, "Deterministic noise amplitude", "0.0", nullptr},
    {"wave_seed", "int", false, "Deterministic noise seed", "0", nullptr},
    {"wave_axis", "int", false, "Local offset axis: 0=X, 1=Y, 2=Z", "1", nullptr},
};
static const MethodDescriptor desc_spline_skin_create = {
    "spline.skin.create", "spline",
    "Create or update a non-destructive skin display on an open spline",
    "Reuses one linked preview host while parameters change. Builds Spline Object to Taper to Twist to Wave/Noise to Curve to Mesh to Output; an optional closed custom_profile drives the cross-section.",
    "write", "SceneWrite", false, "any",
    "spline|skin|create|bevel|cable|tube|curve-to-mesh|geometry-nodes",
    "spline.get|spline.keyframe.insert|geometry_cache.bake|nodes.set_property",
    nullptr, nullptr, nullptr, nullptr,
    params_spline_skin_create, 20,
    true
};
static const MethodRegistration reg_spline_skin_create(desc_spline_skin_create);

static const MethodParam params_spline_skin_finalize[] = {
    {"spline", "string", true, "Open spline source owning the live skin display", nullptr, nullptr},
};
static const MethodDescriptor desc_spline_skin_finalize = {
    "spline.skin.finalize", "spline",
    "Convert a spline's live skin display into an ordinary mesh",
    "Evaluates the current frame, removes the procedural Geometry Graph link and keeps the existing flat preview host as the final mesh.",
    "write", "SceneWrite", false, "any",
    "spline|skin|finalize|preview|convert|apply|mesh",
    "spline.skin.create|spline.skin.clear|geometry_cache.bake",
    nullptr, nullptr, nullptr, nullptr,
    params_spline_skin_finalize, 1,
    true
};
static const MethodRegistration reg_spline_skin_finalize(desc_spline_skin_finalize);

static const MethodParam params_spline_subdivide[] = {
    {"name", "string", true, "Spline object name", nullptr, nullptr},
    {"segments", "int[]", true, "Segment start indices", nullptr, nullptr},
    {"cuts", "int", true, "Cuts per segment, from 1 to 128", "1", nullptr},
};
static const MethodDescriptor desc_spline_subdivide = {
    "spline.subdivide", "spline",
    "Subdivide one or more spline segments with a fixed cut count",
    "Segments are processed in descending index order so a batch request remains stable. The operation is validated as one mutation.",
    "write", "SceneWrite", false, "{last_index:int}",
    "spline|subdivide|batch|curve",
    "spline.insert_point|spline.get",
    nullptr, nullptr, nullptr, nullptr,
    params_spline_subdivide, 3,
    true
};
static const MethodRegistration reg_spline_subdivide(desc_spline_subdivide);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_templates_validate, 1,
    true
};
static const MethodRegistration reg_templates_validate(desc_templates_validate);

static const MethodParam params_terrain_apply_preset[] = {
    {"preset", "string", true, "Preset name", nullptr, "default|snow_layer|snowy_mountain_valley|river_network|biome_temperate|biome_lush|biome_alpine|biome_arid|biome_boreal|biome_foliage|geology_foundation"},
    {"name", "string", true, "", nullptr, nullptr},
    {"add_satmap", "bool", false, "Also add and wire a matching SatMap colorizer", "False", nullptr},
    {"replace_graph", "bool", false, "", "false", nullptr},
};
static const MethodDescriptor desc_terrain_apply_preset = {
    "terrain.apply_preset", "terrain",
    "Apply a built-in terrain node preset and report links it could not wire",
    "Returns wiring_faults: links the setup asked for and did not get. A refused link leaves the graph looking complete while the consumer falls back to a synthesized value, so an empty wiring_faults list is the only evidence the setup actually connected everything. wiring_fault_count > 0 means the resulting splat/biome masks are partly synthesized, not driven by the graph.",
    "write", "SceneWrite", false, "any",
    "terrain|apply|preset|landscape|mountain|snow|river",
    nullptr,
    nullptr, nullptr, "nodes.list", nullptr,
    params_terrain_apply_preset, 4,
    true
};
static const MethodRegistration reg_terrain_apply_preset(desc_terrain_apply_preset);

static const MethodParam params_terrain_apply_satmap_preset[] = {
    {"name", "string", true, "", nullptr, nullptr},
    {"preset", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_apply_satmap_preset = {
    "terrain.apply_satmap_preset", "terrain",
    "Build a SatMap node recipe from flow, slope, curvature and other terrain fields",
    nullptr,
    "write", "SceneWrite", false, "any",
    "terrain|apply|satmap|preset|landscape|color|flow|slope|curvature",
    "terrain.list_satmap_presets|terrain.evaluate",
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_apply_satmap_preset, 2,
    true
};
static const MethodRegistration reg_terrain_apply_satmap_preset(desc_terrain_apply_satmap_preset);

static const MethodParam params_terrain_calculate_flow[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_calculate_flow = {
    "terrain.calculate_flow", "terrain",
    "Reconstruct final-height flow diagnostics and report likely inland stalls",
    "Reconstructs a diagnostic accumulation field from the final heightmap, then returns channel cells with no greater-accumulation neighbour as inland_terminations. This does not inspect Watershed Analysis's authoritative direction/receiver raster (Hydraulic Erosion stopped publishing one when the compact port contract moved flow direction to its authoritative owner), so it detects wholesale depression-routing failures but is not an exact river-topology measurement. Use terrain.flow_authority to identify the graph's real source.",
    "write", "SceneWrite", false, "any",
    "terrain|calculate|flow|landscape|water|hydrology|measurement",
    "terrain.erosion_stats|terrain.erode",
    nullptr, nullptr, "terrain.erosion_stats", nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_carve_river, 13,
    true
};
static const MethodRegistration reg_terrain_carve_river(desc_terrain_carve_river);

static const MethodParam params_terrain_create[] = {
    {"size", "float", false, "World-space extent in metres", "1000.0", nullptr},
    {"resolution", "int", false, "Heightmap resolution in samples per side", "1024", nullptr},
    {"height_scale", "float", false, "Vertical scale in metres", "100.0", nullptr},
    {"mesh_resolution", "int", false, "", "0", nullptr},
    {"name", "string", false, "", "Terrain", nullptr},
};
static const MethodDescriptor desc_terrain_create = {
    "terrain.create", "terrain",
    "Create a terrain heightfield of a given world size and resolution",
    "resolution is the FIELD grid: heights and every analysis product (slope, flow, erosion, analysisFields) live at that resolution. mesh_resolution is the separate VERTEX grid; 0 follows the field. Set it at creation rather than decimating afterwards - creating at 4096 and decimating still pays one full-resolution acceleration-structure build (measured 5.6 s Vulkan solid raster + 2.8 s Embree). With a 1024 mesh over a 4096 field the same build is 0.42 s + 0.15 s.",
    "write", "SceneWrite", false, "any",
    "terrain|create|landscape|mountain|ground|heightmap",
    "terrain.set_mesh_resolution|terrain.get|perf.list",
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_create, 5,
    true
};
static const MethodRegistration reg_terrain_create(desc_terrain_create);

static const MethodParam params_terrain_erode[] = {
    {"name", "string", true, "", nullptr, nullptr},
    {"type", "string", false, "Erosion model", "hydraulic", "hydraulic|thermal|fluvial|wind"},
    {"iterations", "int", false, "Droplet count for the hydraulic model; 0 uses the model's own default", "0", nullptr},
    {"strength", "float", false, "Erosion strength", "0.2", nullptr},
    {"talus_angle", "float", false, "Repose angle for thermal erosion", "0.5", nullptr},
    {"fluvial_cycle", "int", false, "Enable the landscape-evolution cycle (1/0); omit to keep the solver setting", "-1", nullptr},
    {"fluvial_quality", "string", false, "Convergence budget preset. Sets iteration count, relaxation passes, transport steps and pyramid depth ONLY - never the shape parameters - so raising it refines the same landscape instead of producing a different one. Applied before any explicit budget parameter in the same call, so you can pick a preset and override one dial of it.", "", "draft|balanced|high"},
    {"fluvial_iterations", "int", false, "Cycle iterations. Refines the solve; does NOT change how much material moves", "-1", nullptr},
    {"fluvial_time_step", "float", false, "Whole-cycle time multiplier: this is the dial for how much material moves", "-1.0", nullptr},
    {"rain_rate", "float", false, "Runoff depth per unit time; only its ratio to settling_velocity matters", "-1.0", nullptr},
    {"orographic_rain", "float", false, "Windward/lee rain split, 0..1. The second symmetry breaker after drainage area", "-1.0", nullptr},
    {"rain_wind_degrees", "float", false, "Prevailing wind bearing in degrees for orographic rain", "-1000.0", nullptr},
    {"incision_k", "float", false, "Stream-power coefficient, in metres of incision at 1 km2 catchment and slope 1", "-1.0", nullptr},
    {"stream_power_m", "float", false, "Drainage-area exponent, near 0.5. This is what makes rivers cut faster than rills", "-1.0", nullptr},
    {"stream_power_n", "float", false, "Slope exponent, near 1.0", "-1.0", nullptr},
    {"transport_k", "float", false, "Sediment transport capacity coefficient, same units as incision_k", "-1.0", nullptr},
    {"sediment_cover", "float", false, "Cover effect 0..1: a bed already carrying its capacity is armoured", "-1.0", nullptr},
    {"settling_velocity", "float", false, "How readily suspended load drops. Larger settles sooner and shortens deltas", "-1.0", nullptr},
    {"sediment_route_steps", "int", false, "Cells of downstream sediment travel per iteration", "-1", nullptr},
    {"drainage_refresh_interval", "int", false, "Iterations between depression-fill and drainage-area re-solves", "-1", nullptr},
    {"drainage_fill_passes", "int", false, "GPU depression-fill budget per pyramid level. Too low leaves spurious lakes", "-1", nullptr},
    {"drainage_accumulate_passes", "int", false, "GPU area-accumulation budget. Too low under-counts long trunk rivers", "-1", nullptr},
    {"drainage_coarsest_size", "int", false, "Coarsest pyramid grid for the GPU drainage solve", "-1", nullptr},
    {"flat_gradient", "float", false, "Slope in m/m assumed for a depression-filled FLAT. A flat has no gradient of its own, so this is what water steers by there; 0 disables it and flat channels go back to straight, grid-locked and unable to incise", "-1.0", nullptr},
    {"flat_resolve_passes", "int", false, "GPU flat-resolution budget, one cell per pass: the widest flat in cells that resolves. Check unresolved_flat_cells in the stats, not the render", "-1", nullptr},
    {"mass_wasting", "int", false, "Enable in-loop landslides (1/0). Off means valley walls never collapse", "-1", nullptr},
    {"repose_angle_degrees", "float", false, "Angle of repose for mass wasting", "-1.0", nullptr},
    {"mass_wasting_rate", "float", false, "Fraction of the excess above repose shed per pass, 0..1", "-1.0", nullptr},
    {"mass_wasting_steps", "int", false, "Talus relaxation passes per iteration", "-1", nullptr},
    {"hillslope_diffusion", "float", false, "Creep coefficient in m2 over the whole cycle; rounds ridges, keeps channels", "-1.0", nullptr},
    {"incision_safety", "float", false, "Anti-pit limit: max fraction of the drop to the receiver cut in one step", "-1.0", nullptr},
    {"deposition_safety", "float", false, "Anti-spike limit: max fraction of the rise to the donor built in one step", "-1.0", nullptr},
    {"max_step_meters", "float", false, "Absolute per-step height change limit; 0 selects half a cell", "-1.0", nullptr},
    {"lake_epsilon_meters", "float", false, "Fill residue below this is not treated as standing water", "-1.0", nullptr},
    {"headwater_area_km2", "float", false, "Catchment area at which a cell counts as a channel", "-1.0", nullptr},
    {"backend", "string", false, "Compute backend; auto picks GPU when available", "auto", nullptr},
    {"alluvium_consolidation", "float", false, "", "-1.0", nullptr},
    {"alluvium_rate", "float", false, "", "-1.0", nullptr},
    {"alluvium_slope_degrees", "float", false, "", "-1.0", nullptr},
    {"alluvium_steps", "int", false, "", "-1", nullptr},
    {"amount", "float", false, "", "0.3", nullptr},
    {"avulsion_interval", "int", false, "", "-1", nullptr},
    {"direction", "float", false, "", "45.0", nullptr},
    {"seed", "int", false, "", "1337", nullptr},
    {"undo", "bool", false, "", "true", nullptr},
};
static const MethodDescriptor desc_terrain_erode = {
    "terrain.erode", "terrain",
    "Run an erosion pass over a terrain; the hydraulic model brackets its droplet stages with the fluvial cycle - drainage-area feedback, lake spill and outlet incision, downstream sediment transport",
    "Every fluvial_* parameter defaults to leaving the solver setting alone, so tuning one dial does not reset the other twenty. The cycle is what produces a river hierarchy and deltas; with fluvial_cycle=0 erosion falls back to the local-slope droplet walk, which is statistically isotropic and leaves closed basins undrained. Check the result with terrain.erosion_stats rather than by eye.",
    "write", "SceneWrite", false, "any",
    "terrain|erode|landscape|erosion|weathering|realism|river|delta|lake|sediment",
    "terrain.erosion_stats|terrain.calculate_flow|terrain.evaluate",
    nullptr, nullptr, "terrain.erosion_stats", nullptr,
    params_terrain_erode, 45,
    true
};
static const MethodRegistration reg_terrain_erode(desc_terrain_erode);

static const MethodDescriptor desc_terrain_erosion_stats = {
    "terrain.erosion_stats", "terrain",
    "Read the sediment mass ledger and drainage diagnostics of the last erosion run",
    "This is how an erosion result is checked without a screenshot, and the field to read FIRST is max_drainage_area_fraction: the largest catchment as a share of the map. Single digits mean the drainage graph is in fragments and no trunk river exists - a state that still renders as a convincing river network, which is why looking at the picture has repeatedly sent debugging the wrong way. max_drainage_area_km2 alone is NOT interpretable: 0.03 km2 is a shredded network on a 1 km terrain and a healthy trunk on a 100 m one. lake_cells and lake_area_fraction count ANY standing water including the depression-fill ladder's few-ulp lift, so they can report a quarter of the map and mean nothing; act on deep_lake_cells / deep_lake_area_fraction (10 cm threshold) and deepest_lake_meters instead. A high deep_lake_area_fraction on sloping terrain means the surface fed to the solver is genuinely full of pits - suspect the droplet stage, not the fill. mass_error_fraction away from zero means sediment transport is leaking. cycle_iterations == 0 is not a measurement, it is a cleared default.",
    "read", "Read", false, "any",
    "terrain|erosion|stats|landscape|hydrology|measurement|validation|mass-balance|drainage|lakes",
    "terrain.erode|terrain.calculate_flow|terrain.flow_authority|terrain.evaluate",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_terrain_erosion_stats(desc_terrain_erosion_stats);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_export_heightmap, 2,
    true
};
static const MethodRegistration reg_terrain_export_heightmap(desc_terrain_export_heightmap);

static const MethodParam params_terrain_field_stats[] = {
    {"terrain", "string", true, "Terrain name, as reported by terrain.list", nullptr, nullptr},
    {"field", "string", true, "Live field name, as reported by terrain.list_fields", nullptr, nullptr},
    {"histogram_bins", "int", false, "0 omits the histogram; otherwise 2..256 bins over the finite min/max range", "0", nullptr},
    {"samples", "array", false, "Optional list of integer [x, y] field-grid coordinates", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_field_stats = {
    "terrain.field_stats", "terrain",
    "Measure the values and coverage of one live terrain analysis field",
    "Reads the published field grid itself, not node settings or mesh attributes. Use this before judging a mask visually: min/max/mean and nonzero_fraction expose empty, saturated and constant fields; non_finite_count exposes poisoned data. histogram_bins is 0 to omit or 2..256. samples contains integer [x, y] field-grid coordinates and out-of-range coordinates are refused.",
    "read", "Read", false, "any",
    "terrain|field|stats|mask|measurement|histogram|sample|diagnostic",
    "terrain.list_fields|terrain.get|scatter.set_settings",
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_field_stats, 4,
    true
};
static const MethodRegistration reg_terrain_field_stats(desc_terrain_field_stats);

static const MethodParam params_terrain_flow_authority[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_flow_authority = {
    "terrain.flow_authority", "terrain",
    "Report which discharge field the graph's Flow node classified",
    "Two quantities were both called flow: DISCHARGE, the physical field the erosion sim measures against the terrain it is carving, and CHANNEL, a 0-1 selection of which cells read as a watercourse. Flow is now the single authority - measured discharge goes into it and every consumer reads out of it. source == 'derived_erosion_unwired' is the state nothing else reports: the graph HAS an erosion sim whose discharge is not wired into Flow, so channels come from bare geometry (no lakes, no infiltration, no erosion history) while the render shows the eroded surface, and the result still looks like a river network.",
    "write", "SceneWrite", false, "any",
    "terrain|flow|authority|landscape|hydrology|measurement",
    "terrain.calculate_flow|terrain.erosion_stats|terrain.apply_preset",
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_flow_authority, 1,
    true
};
static const MethodRegistration reg_terrain_flow_authority(desc_terrain_flow_authority);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_import_heightmap, 5,
    true
};
static const MethodRegistration reg_terrain_import_heightmap(desc_terrain_import_heightmap);

static const MethodParam params_terrain_landform_stats[] = {
    {"name", "string", true, "Terrain name", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_landform_stats = {
    "terrain.landform_stats", "terrain",
    "Measure the realised shape of a terrain: slope distribution, landform width, local-relief spread and hypsometry",
    "Measured on the BAKED heightfield, never read back from a generator's own settings, so it can see the gap between what a node was asked for and what the terrain became. relief_window_meters / relief_window_relief are the relief-vs-window ladder itself; broad_growth is the relief added by its last doubling and is the sharpest read on whether the tile has landforms at its own scale - near 1.0 the field is already uncorrelated at a quarter of the tile and the map is one texture repeated (measured 1.05 with Feature Size 600 m on a 4096 m terrain, against 1.37 at 2048 m). landform_scale_meters is the ladder's growth knee, octave-quantised and NOT expected to equal the authored Feature Size. cliff_fraction is the area over 40 degrees - ground too steep to hold soil, so the share that must read as bare rock - and roughness_slope_ratio is cell-scale roughness on the steepest fifth over the gentlest fifth. Read those two TOGETHER: a terrain with no steep ground still scores a high ratio because its steepest fifth is only its least flat fifth. local_relief_ratio near 1 means every part of the map is equally rugged - no massifs, no basins. lowland_fraction below midland_fraction means the hypsometry is gaussian rather than depositional. spectrum_kink is macro-micro coherence in one number: the largest departure of any octave from the fitted power law. A landscape whose layers agree has ONE law from the grid to the landform, so a detail pass that does not match what it sits on, or a band guard quietly dropping octaves, shows up here and ONLY here - slope, relief and hypsometry all stay healthy while it is there. Under about 0.05 the scales are one landscape; spectrum_kink_meters names the window and spectrum_kink_signed says whether that scale is starved of detail (negative) or carrying too much (positive). realised_hurst is fitted only below landform_scale_meters; include the saturated windows and any exponent is dragged toward 0.5 regardless of the field.",
    "read", "Read", false, "any",
    "terrain|landform|stats|landscape|measure|diagnostic|shape|slope|relief",
    "terrain.slope_area_fit|terrain.flow_authority|terrain.get",
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_landform_stats, 1,
    true
};
static const MethodRegistration reg_terrain_landform_stats(desc_terrain_landform_stats);

static const MethodDescriptor desc_terrain_list = {
    "terrain.list", "terrain",
    "List the terrain objects",
    nullptr,
    "read", "Read", false, "any",
    "terrain|list|landscape|inventory",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_terrain_list(desc_terrain_list);

static const MethodParam params_terrain_list_fields[] = {
    {"terrain", "string", true, "Terrain name, as reported by terrain.list", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_list_fields = {
    "terrain.list_fields", "terrain",
    "List the named analysis fields a terrain is currently publishing",
    "Measures the LIVE terrain: a name appears only because an output node (Terrain Fields Output, or a Publish Field node) actually wrote it during this evaluation. It is not a catalogue of what the node library could produce, so an empty result means the graph has not published anything yet - not that the feature is missing. These are the valid mask names for scatter.set_settings.",
    "read", "Read", false, "any",
    "terrain|list|fields|field|mask|introspection",
    "scatter.set_settings|terrain.list_layers",
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_list_fields, 1,
    true
};
static const MethodRegistration reg_terrain_list_fields(desc_terrain_list_fields);

static const MethodParam params_terrain_list_layers[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_list_layers = {
    "terrain.list_layers", "terrain",
    "List the eight terrain layer slots and what drives each one",
    "Slots 0-3 are weighted by the splat map and normalized against each other - they partition the surface. Slots 4-7 are weighted by the semantic map (Flow, Wetness, Ice, Hardness) and composited OVER that blend by their own unnormalized weight. bound=false on a semantic slot means the built-in shading still applies to that channel. TWO dead cases to check on a bound overlay: channel_coverage 0 means the graph never fills that channel (Auto Splat's semantic output writes Flow and hard-zeroes the other three, while Surface Composer writes all four); channel_constant true means min==max, a flat fill that reports full coverage yet selects nothing and washes the material evenly over the terrain - this is how an unwired Surface Composer Hardness input presents, falling back to a constant 0.45. Hardness is produced by Lithology or Strata.",
    "read", "Read", false, "any",
    "terrain|list|layers|landscape|material|layer|splat|semantic",
    "terrain.set_layer|material.list",
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_list_layers, 1,
    true
};
static const MethodRegistration reg_terrain_list_layers(desc_terrain_list_layers);

static const MethodDescriptor desc_terrain_list_rivers = {
    "terrain.list_rivers", "terrain",
    "List the rivers carved into a terrain",
    nullptr,
    "read", "Read", false, "any",
    "terrain|list|rivers|landscape|river|inventory",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_terrain_list_rivers(desc_terrain_list_rivers);

static const MethodDescriptor desc_terrain_list_satmap_presets = {
    "terrain.list_satmap_presets", "terrain",
    "List data-driven multi-field SatMap preset recipes",
    nullptr,
    "read", "Read", false, "any",
    "terrain|list|satmap|presets|landscape|color|preset|flow|curvature",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_terrain_list_satmap_presets(desc_terrain_list_satmap_presets);

static const MethodParam params_terrain_paint_splat[] = {
    {"name", "string", true, "", nullptr, nullptr},
    {"dabs", "array", true, "Stroke path: list of [world_x, world_z] pairs, all of which must land on the tile.", nullptr, nullptr},
    {"channel", "int", false, "Splat channel / layer index 0..3", "0", nullptr},
    {"dt", "float", false, "Seconds one dab represents; the panel feeds one frame", "0.0166667", nullptr},
    {"radius", "float", false, "", "5.0", nullptr},
    {"strength", "float", false, "", "1.0", nullptr},
    {"undo", "bool", false, "", "true", nullptr},
};
static const MethodDescriptor desc_terrain_paint_splat = {
    "terrain.paint_splat", "terrain",
    "Paint one terrain splat channel (layer) along a list of world-space dabs",
    "Fails rather than initialising when the terrain has no splat map or layers: an auto-created layer stack would hide the missing setup step. coverage_before/after are the mean weight of the painted channel over the whole splat map, which is what separates 'the stroke ran' from 'the stroke painted'; a stroke fully outside the tile is refused, not silently ignored.",
    "write", "SceneWrite", false, "any",
    "terrain|paint|splat|landscape|brush",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_paint_splat, 7,
    true
};
static const MethodRegistration reg_terrain_paint_splat(desc_terrain_paint_splat);

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
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_remove, 1,
    true
};
static const MethodRegistration reg_terrain_remove(desc_terrain_remove);

static const MethodParam params_terrain_road_assign_profile[] = {
    {"spline", "string", true, "SplineObject name, as reported by spline.list", nullptr, nullptr},
    {"profile", "string", true, "Profile id: footpath | dirt_road | main_road. An unknown id is REFUSED, never substituted - carving a main road where a footpath was asked for looks entirely plausible", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_road_assign_profile = {
    "terrain.road.assign_profile", "terrain",
    "Marks an existing SplineObject as a road under the named profile. Stores NO curve geometry - the curve stays owned by the spline.* surface.",
    "A spline that does not exist is refused: an assignment pointing at a missing curve would look identical to a working one in every listing. This is also what removes the graph explosion - Road Network reads the registry, so N roads are one node.",
    "write", "SceneWrite", false, "any",
    "terrain|road|assign|profile",
    "terrain.road.clear_profile|terrain.road.get_assignment|spline.list",
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_road_assign_profile, 2,
    true
};
static const MethodRegistration reg_terrain_road_assign_profile(desc_terrain_road_assign_profile);

static const MethodParam params_terrain_road_build_mesh[] = {
    {"spline", "string", true, "SplineObject name of an assigned road", nullptr, nullptr},
    {"object", "string", false, "Scene object to write. Empty reuses the object this assignment already owns, or creates <spline>_RoadSurface", "", nullptr},
    {"include_shoulder", "bool", false, "Include the shoulder in the ribbon. The ditch is never meshed - a ditch is terrain", "true", nullptr},
    {"surface_offset", "float", false, "Metres above the graded surface. At zero the continuous ribbon and the discretised heightfield interpenetrate and the road reads as torn", nullptr, nullptr},
    {"uv_meters_per_tile", "float", false, "Metres of road per V tile", nullptr, nullptr},
    {"skip_tunnels", "bool", false, "Split the ribbon at tunnel spans instead of drawing a deck through the mountain", "true", nullptr},
};
static const MethodDescriptor desc_terrain_road_build_mesh = {
    "terrain.road.build_mesh", "terrain",
    "Generates the optional road surface mesh from the solved route - the very samples that carved the terrain, never a second sampling of the curve.",
    "A rebuild REPLACES the geometry of the object the assignment owns, so repeated generation cannot leave a stack of stale roads behind. Publishes flat TriangleMesh / DNA SoA geometry. The terrain-only road stays valid without it: the mesh is a view of the route, never the authority for it.",
    "write", "SceneWrite", false, "object, vertex_count, triangle_count, span_count, length_meters, replaced_existing",
    "terrain|road|build|mesh|geometry",
    "terrain.road.clear_mesh|terrain.road.get_route",
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_road_build_mesh, 6,
    true
};
static const MethodRegistration reg_terrain_road_build_mesh(desc_terrain_road_build_mesh);

static const MethodParam params_terrain_road_clear_carve_override[] = {
    {"spline", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_road_clear_carve_override = {
    "terrain.road.clear_carve_override", "terrain",
    nullptr,
    nullptr,
    "write", "SceneWrite", false, "any",
    "terrain|road|clear|carve|override",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_road_clear_carve_override, 1,
    false
};
static const MethodRegistration reg_terrain_road_clear_carve_override(desc_terrain_road_clear_carve_override);

static const MethodParam params_terrain_road_clear_mesh[] = {
    {"spline", "string", true, "SplineObject name of an assigned road", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_road_clear_mesh = {
    "terrain.road.clear_mesh", "terrain",
    "Deletes the generated road surface and forgets the ownership record.",
    "The ownership record is cleared even when the object is already gone, so the next build is not blocked by a pointer to nothing. The terrain-only road is unaffected.",
    "write", "SceneWrite", false, "any",
    "terrain|road|clear|mesh",
    "terrain.road.build_mesh",
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_road_clear_mesh, 1,
    true
};
static const MethodRegistration reg_terrain_road_clear_mesh(desc_terrain_road_clear_mesh);

static const MethodParam params_terrain_road_clear_profile[] = {
    {"spline", "string", true, "SplineObject name", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_road_clear_profile = {
    "terrain.road.clear_profile", "terrain",
    "Removes a road assignment. The SplineObject itself is untouched.",
    nullptr,
    "write", "SceneWrite", false, "any",
    "terrain|road|clear|profile",
    "terrain.road.assign_profile|terrain.road.list_assignments",
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_road_clear_profile, 1,
    true
};
static const MethodRegistration reg_terrain_road_clear_profile(desc_terrain_road_clear_profile);

static const MethodParam params_terrain_road_get_assignment[] = {
    {"spline", "string", true, "SplineObject name", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_road_get_assignment = {
    "terrain.road.get_assignment", "terrain",
    "Reads one road assignment back, including whether its curve still exists.",
    "Returns spline_object, profile_id, crossing_mode, enabled, has_override and curve_exists. A setter with no readback cannot be tested.",
    "read", "Read", false, "any",
    "terrain|road|get|assignment",
    "terrain.road.assign_profile|terrain.road.get_diagnostics",
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_road_get_assignment, 1,
    true
};
static const MethodRegistration reg_terrain_road_get_assignment(desc_terrain_road_get_assignment);

static const MethodDescriptor desc_terrain_road_get_diagnostics = {
    "terrain.road.get_diagnostics", "terrain",
    "Reports assignment health: counts, assignments whose SplineObject is gone, and assignments naming a profile this build does not have.",
    "Returns assignment_count, enabled_count, dangling and unknown_profiles. A road that silently stops grading because its curve was deleted is the failure nobody files, so it is surfaced here and on the Road Network node.",
    "read", "Read", false, "any",
    "terrain|road|get|diagnostics",
    "terrain.road.list_assignments|terrain.field_stats",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_terrain_road_get_diagnostics(desc_terrain_road_get_diagnostics);

static const MethodParam params_terrain_road_get_route[] = {
    {"spline", "string", true, "SplineObject name of an assigned road", nullptr, nullptr},
    {"max_samples", "int", false, "0 (default) returns only the counters. A positive value returns that many samples, decimated with both endpoints kept", "0", nullptr},
};
static const MethodDescriptor desc_terrain_road_get_route = {
    "terrain.road.get_route", "terrain",
    "Reads the SOLVED route of one assigned road out of the terrain graph: sample positions, road and ground height, and the crossing each sample resolved to.",
    "This is the measurement surface for crossings: bridge_samples, ford_samples and tunnel_samples are how a script proves a declared crossing actually resolved, and crossing_diagnostic names one that could not. It fails explicitly when no Road Network node has carved this assignment yet - an empty route and an unevaluated graph must not read the same.",
    "read", "Read", false, "spline_object, terrain, sample_count, bridge_samples, ford_samples, tunnel_samples, length_meters, peak_cut_meters, peak_fill_meters, revision, crossing_diagnostic, samples[]",
    "terrain|road|get|route|measurement",
    "terrain.road.build_mesh|terrain.road.set_crossing_mode|terrain.field_stats",
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_road_get_route, 2,
    true
};
static const MethodRegistration reg_terrain_road_get_route(desc_terrain_road_get_route);

static const MethodDescriptor desc_terrain_road_list_assignments = {
    "terrain.road.list_assignments", "terrain",
    "Lists every road assignment in the scene.",
    "Same fields as terrain.road.get_assignment, one row per assignment.",
    "read", "Read", false, "any",
    "terrain|road|list|assignments",
    "terrain.road.get_assignment|terrain.road.get_diagnostics",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_terrain_road_list_assignments(desc_terrain_road_list_assignments);

static const MethodDescriptor desc_terrain_road_list_profiles = {
    "terrain.road.list_profiles", "terrain",
    "Lists the built-in road profiles. 'Path' and 'road' are the SAME measurement under a different profile, not two solvers - a footpath is narrow, steep-capable and barely moves earth; a main road buys a shallow grade with earthworks.",
    "Returns id, display_name and the carve parameters each profile sets: road_width, shoulder_width, grading_falloff, foliage_margin, max_grade_percent, max_cut_meters, max_fill_meters. Ids: footpath, dirt_road, main_road.",
    "read", "Read", false, "any",
    "terrain|road|list|profiles",
    "terrain.road.assign_profile|terrain.road.list_assignments",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_terrain_road_list_profiles(desc_terrain_road_list_profiles);

static const MethodParam params_terrain_road_set_carve_override[] = {
    {"spline", "string", true, "", nullptr, nullptr},
    {"crown_meters", "float", false, "", nullptr, nullptr},
    {"ditch_depth", "float", false, "", nullptr, nullptr},
    {"ditch_width", "float", false, "", nullptr, nullptr},
    {"elevation_offset", "float", false, "", nullptr, nullptr},
    {"foliage_margin", "float", false, "", nullptr, nullptr},
    {"grading_falloff", "float", false, "", nullptr, nullptr},
    {"max_cut_meters", "float", false, "", nullptr, nullptr},
    {"max_fill_meters", "float", false, "", nullptr, nullptr},
    {"max_grade_percent", "float", false, "", nullptr, nullptr},
    {"road_width", "float", false, "", nullptr, nullptr},
    {"shoulder_width", "float", false, "", nullptr, nullptr},
    {"use_point_width", "bool", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_road_set_carve_override = {
    "terrain.road.set_carve_override", "terrain",
    nullptr,
    nullptr,
    "write", "SceneWrite", false, "any",
    "terrain|road|set|carve|override",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_road_set_carve_override, 13,
    false
};
static const MethodRegistration reg_terrain_road_set_carve_override(desc_terrain_road_set_carve_override);

static const MethodParam params_terrain_road_set_crossing_mode[] = {
    {"spline", "string", true, "SplineObject name", nullptr, nullptr},
    {"mode", "string", true, "auto | terrain | bridge | ford | tunnel. An unknown mode is refused. auto bridges over water when the Road Network node's Water pin is connected and otherwise behaves exactly as terrain; bridge also spans wherever the required fill exceeds max_fill_meters; tunnel bores wherever the required cut exceeds max_cut_meters; ford follows the bed and needs a Water field, reporting crossing_diagnostic when it has none", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_road_set_crossing_mode = {
    "terrain.road.set_crossing_mode", "terrain",
    "Sets how a road resolves where the ground cannot carry it. The solver applies the mode PER ROUTE SAMPLE, so only the span that needs a crossing gets one.",
    "Bridge and tunnel spans leave the terrain untouched and are straightened between their abutments; a tunnel also publishes no surface masks at all, so a forest still grows over it. Measure the result with terrain.road.get_route.",
    "write", "SceneWrite", false, "any",
    "terrain|road|set|crossing|mode",
    "terrain.road.get_route|terrain.road.get_assignment",
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_road_set_crossing_mode, 2,
    true
};
static const MethodRegistration reg_terrain_road_set_crossing_mode(desc_terrain_road_set_crossing_mode);

static const MethodParam params_terrain_road_set_enabled[] = {
    {"spline", "string", true, "SplineObject name", nullptr, nullptr},
    {"enabled", "bool", false, "false leaves the assignment in place but stops it carving", "true", nullptr},
};
static const MethodDescriptor desc_terrain_road_set_enabled = {
    "terrain.road.set_enabled", "terrain",
    "Enables or disables one road without losing its profile assignment.",
    nullptr,
    "write", "SceneWrite", false, "any",
    "terrain|road|set|enabled",
    "terrain.road.get_assignment|terrain.road.get_diagnostics",
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_road_set_enabled, 2,
    true
};
static const MethodRegistration reg_terrain_road_set_enabled(desc_terrain_road_set_enabled);

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
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_sample_height, 3,
    true
};
static const MethodRegistration reg_terrain_sample_height(desc_terrain_sample_height);

static const MethodParam params_terrain_sculpt[] = {
    {"name", "string", true, "", nullptr, nullptr},
    {"dabs", "array", true, "Stroke path: list of [world_x, world_z] pairs. One dab is a click, several are a drag; every dab must land on the tile or the call is refused.", nullptr, nullptr},
    {"mode", "string", false, "", "raise", "raise|lower|flatten|smooth|stamp"},
    {"dt", "float", false, "Seconds one dab represents; the panel feeds one frame", "0.0166667", nullptr},
    {"strength", "float", false, "Metres of height change per second at full falloff", "0.5", nullptr},
    {"radius", "float", false, "Brush radius in metres", "5.0", nullptr},
    {"curve", "float", false, "Falloff exponent, 0.25..4", "2.0", nullptr},
    {"flatten_target", "float", false, "", "0.0", nullptr},
    {"stamp_rotation", "float", false, "", "0.0", nullptr},
    {"stamp_texture", "string", false, "", "", nullptr},
    {"undo", "bool", false, "", "true", nullptr},
    {"use_fixed_height", "bool", false, "", "false", nullptr},
};
static const MethodDescriptor desc_terrain_sculpt = {
    "terrain.sculpt", "terrain",
    "Run a terrain sculpt stroke (raise|lower|flatten|smooth|stamp) over a list of world-space dabs",
    "The panel brush and this method share TerrainManager::sculpt, so this is the only way to regression-test terrain sculpting. Read height_delta, NOT ok: the height field is metres/scale_y and a graph-authored landform routinely exceeds 1.0, so a bug that clamps the field reports success while flattening the ground under the brush. dt is the seconds one dab represents (the panel feeds 1/60); strength is metres per second at full falloff, so a visible bump needs either a large strength or many dabs. flatten samples its target from the surface under the first dab unless use_fixed_height is set.",
    "write", "SceneWrite", false, "any",
    "terrain|sculpt|landscape|brush|authoring",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_sculpt, 12,
    true
};
static const MethodRegistration reg_terrain_sculpt(desc_terrain_sculpt);

static const MethodParam params_terrain_set_layer[] = {
    {"slot", "int", true, "0-3 splat-weighted, 4-7 semantic overlay (4=Flow 5=Wetness 6=Ice 7=Hardness)", nullptr, nullptr},
    {"name", "string", true, "", nullptr, nullptr},
    {"material", "any", false, "Material name to bind; \"\" clears the slot; omit to leave unchanged", nullptr, nullptr},
    {"uv_scale", "any", false, "Tiling scale; omit to leave unchanged", nullptr, nullptr},
    {"overlay_strength", "any", false, "Overlay dial [0,1] for slots 4-7; omit to leave unchanged", nullptr, nullptr},
    {"overlay_ignore_cover", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_set_layer = {
    "terrain.set_layer", "terrain",
    "Bind or edit one terrain layer slot",
    "Omit a field to leave it unchanged. material=\"\" CLEARS the slot; clearing a semantic overlay restores the built-in shading for that channel. overlay_strength applies to slots 4-7 only and is refused on 0-3, which are normalized against each other and have no independent strength.",
    "write", "SceneWrite", false, "any",
    "terrain|set|layer|landscape|material|splat|semantic|river",
    "terrain.list_layers",
    nullptr, nullptr, "terrain.list_layers", nullptr,
    params_terrain_set_layer, 6,
    true
};
static const MethodRegistration reg_terrain_set_layer(desc_terrain_set_layer);

static const MethodParam params_terrain_set_mesh_resolution[] = {
    {"name", "string", true, "Terrain name", nullptr, nullptr},
    {"mesh_resolution", "int", true, "Vertices per side for the mesh grid, or 0 to follow the field resolution", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_set_mesh_resolution = {
    "terrain.set_mesh_resolution", "terrain",
    "Set the terrain vertex-grid resolution independently of the field (analysis) resolution",
    "0 means the mesh follows the field, which is the historical behaviour. The field grid (heights, slope, flow, erosion, every analysisFields entry) is untouched; only the triangle count changes, and analysis fields are resampled onto the coarser vertex grid rather than dropped. Measured at 4096^2: mesh fill is 380 ms but the acceleration structures built from its 33.5 M triangles cost 6.4 s (Vulkan solid raster) + 2.6 s (Embree), and that cost is linear in triangle count - a 1024 mesh over a 4096 field is roughly 18x cheaper end to end. Normals are still sampled from the field, so shading detail largely survives; the silhouette does not. A value above the field resolution is REFUSED rather than clamped.",
    "write", "SceneWrite", false, "TerrainInfo",
    "terrain|set|mesh|resolution|lod|performance|triangles|decimate",
    "terrain.get|terrain.create|perf.list",
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_set_mesh_resolution, 2,
    true
};
static const MethodRegistration reg_terrain_set_mesh_resolution(desc_terrain_set_mesh_resolution);

static const MethodParam params_terrain_set_paint_resolution[] = {
    {"name", "string", true, "", nullptr, nullptr},
    {"paint_resolution", "int", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_set_paint_resolution = {
    "terrain.set_paint_resolution", "terrain",
    nullptr,
    nullptr,
    "write", "SceneWrite", false, "any",
    "terrain|set|paint|resolution",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_set_paint_resolution, 2,
    false
};
static const MethodRegistration reg_terrain_set_paint_resolution(desc_terrain_set_paint_resolution);

static const MethodParam params_terrain_slope_area_fit[] = {
    {"name", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_terrain_slope_area_fit = {
    "terrain.slope_area_fit", "terrain",
    "Fit the slope-area relation over the channel network to test for real fluvial form",
    "Stream-power erosion gives S = k * A^-theta, so on log-log axes channel cells fall on a straight line. Noise, diffusion and sculpted relief scatter instead. This is the only measurement that separates 'an erosion pass ran' from 'the erosion pass produced fluvial form' - both render as plausible terrain. Read r_squared BEFORE concavity_index: a confident theta fitted to scatter is exactly the number that ends a debugging session early. Real landscapes sit near theta 0.4-0.6. MEASURED: the flow field is filled by terrain.calculate_flow ONLY. Evaluating the node graph does not fill it - the buffer ends up allocated, correctly sized and entirely zero, which used to sail past the size guard and answer too_few_channels, an empty input dressed as a statement about the landscape. Call terrain.calculate_flow first; status flow_field_empty now names that case and flow_peak reports what the fit actually read. status no_flow_field means the buffer is missing or the wrong size.",
    "write", "SceneWrite", false, "any",
    "terrain|slope|area|fit|landscape|hydrology|erosion|measurement|validation",
    "terrain.calculate_flow|terrain.erosion_stats|terrain.flow_authority",
    nullptr, nullptr, nullptr, nullptr,
    params_terrain_slope_area_fit, 1,
    true
};
static const MethodRegistration reg_terrain_slope_area_fit(desc_terrain_slope_area_fit);

static const MethodDescriptor desc_timeline_get_frame = {
    "timeline.get_frame", "timeline",
    "Return the current timeline frame",
    nullptr,
    "read", "Read", false, "any",
    "timeline|get|frame|time|playhead",
    "timeline.set_frame",
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_viewport_capture, 1,
    true
};
static const MethodRegistration reg_viewport_capture(desc_viewport_capture);

static const MethodDescriptor desc_viewport_frame_telemetry = {
    "viewport.frame_telemetry", "viewport",
    "Report raster/Realtime viewport frame presentation telemetry: per-stage host timings, async frame-slot counters and stale-present count",
    "'available' false means NO raster frame has ever been presented (Rendered mode, no Vulkan viewport backend, or nothing drawn yet) - the missing fields are ABSENCE, not zeros. GEOMETRY SUBMISSION: 'gpu_culling' false while 'global_instance_buffer' is true means the scene is drawn with NO frustum culling and NO scatter proxy - every instance of every mesh, wherever the camera looks. That combination is a FAULT that looks CORRECT on screen and only shows up as cost, so check it before trusting any raster timing. 'visible_triangles'/'full_triangles'/'proxy_triangles'/'full_instances'/'proxy_instances' are read back from the GPU when gpu_culling is on and are therefore ONE FRAME BEHIND; identical on a still camera, one frame late while it moves. 'scatter_triangle_target' is a TARGET, not a hard cap: GPU culling converges a distance threshold toward it across frames, so a fast camera move can exceed it for a frame or two. 'async_present' false means the driver refused persistent frame slots and the old synchronous readback path is live; 'image_readback_ms' is non-zero ONLY there. COUNTERS COUNT BLOCKS, NOT CALLS (corrected 2026-09-02; earlier numbers are not comparable). 'stale_presents' means NO new frame could be consumed during a whole render pass - the viewer really is seeing pixels from before this pass. It used to be incremented by the opportunistic second consume that runs right after submit, which can essentially never succeed, so it tracked frame count in healthy async operation and read as 'the ring is permanently behind'. A one-frame delivery delay is what the async ring BUYS, not a fault. 'slot_waits' likewise counts only waits that actually blocked on an unsignalled fence; read it together with 'slot_wait_ms', which is the honest number and stays 0.0 when nothing blocked. viewport.capture(enabled=true) FORCES synchronous presentation so a probe reads the frame just recorded, so timings taken with capture on are not the interactive timings. 'resource_drains' counts host blocks caused by RESOURCE MUTATION (a buffer/descriptor edit that had to wait for submitted frames), and as of 2026-09-02 it increments ONLY when the wait actually blocked on an unsignalled fence - it used to count every call, including waits on already-signalled fences, so it reported a CALL COUNT and not a cost. Read it per submitted frame: above about 1 per frame means some edit path mutates GPU-read state every frame and the two-slot ring is producing no overlap at all, which shows up as 'stale_presents' tracking 'frames_submitted'. Numbers from before that fix are not comparable. 'host_read_ms' was ALSO looking at the wrong place until 2026-09-02: it timed only the opportunistic post-submit consume (which essentially never succeeds), so it read 0.00 and made the GPU-to-CPU readback look free. It now covers the top-of-pass consume that performs the real invalidate + full-frame memcpy. Any architecture decision about removing the readback needs THIS number, measured at the target resolution - the copy scales with pixel count, so a figure taken at 1680x945 does not describe 4K. DISPLAY PATH (added 2026-09-02): every timing above ends when the backend's renderProgressive returns, and the frame is NOT finished there - the main loop still rebuilds the display surface and uploads it to the SDL texture. Those two passes are 'display_post_ms' and 'display_texture_upload_ms', with 'display_loop_period_ms' as the real main-loop period; they are published ONLY on frames that were actually presented (skipped idle/background frames would otherwise average in as free). They are reported even when 'available' is false, because Rendered mode goes down the SAME display path - check 'display_available'. 'display_post_was_noop_copy' true means color processing was a no-op and the pass degenerated to a plain full-frame copy: still a full-frame cost, but REMOVABLE work rather than needed work. MEASURED 2026-09-02 at 3840x2160, material shading, Vulkan viewport: host_read_ms 3.06 + present_ms 2.79 = 5.84 ms of a 10.57 ms frame (55%), i.e. two 31.6 MB CPU copies per frame; the copy scales with pixel count, so numbers taken at 1080p are ~1/4 of these and must not be compared across resolutions.",
    "render", "Render", false, "any",
    "viewport|frame|telemetry|realtime|performance|measure|latency|present",
    "viewport.status|perf.list|render.probe",
    nullptr, nullptr, "viewport.status", nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_viewport_frame_telemetry(desc_viewport_frame_telemetry);

static const MethodDescriptor desc_viewport_get_screenshot = {
    "viewport.get_screenshot", "viewport",
    "Return the captured viewport frame as a base64 JPEG, for a vision-capable model to look at",
    "Requires viewport.capture(enabled=true) and at least one rendered frame; an empty result means NOT CAPTURED, never 'the screen is empty'. Looking is not measuring - pair it with render.probe when the question is numeric (is there a black band, did the frame change), because a model reading a JPEG cannot tell 0.001 luminance from 0.",
    "render", "Render", false, "any",
    "viewport|get|screenshot|image|vision|look|see",
    "viewport.capture|render.probe|viewport.render_frames",
    "viewport.capture|viewport.render_frames", nullptr, "render.probe", nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_viewport_get_screenshot(desc_viewport_get_screenshot);

static const MethodDescriptor desc_viewport_preview_lighting = {
    "viewport.preview_lighting", "viewport",
    "Report realtime/material raster lighting mode, shadow allocation, world background and Nishita direct-sun status",
    "Canonical modes are 'scene' and 'three_point'. The default 'scene' reads the renderer's own light buffer; 'three_point' is the fixed material-inspection rig. Scene draws canonical world color/HDRI/Nishita behind geometry and uses it for ambient/specular. HDRI uses generated diffuse irradiance, GGX roughness prefilter mips and a split-sum BRDF LUT when world_ibl_ready is true; world_ibl_fallback explicitly reports the bounded raw-environment path. Nishita contributes a direct sun with its own directional atlas tile. 'shadowed_light_count' counts scene lights only; 'world_sun_shadow' reports the reserved sun tile. A gap from 'scene_light_count' means later lights still illuminate without an atlas tile. A gap between 'scene_light_count' and 'scene_light_total' means the preview light loop itself is clamped. 'material_preview_active' false means the mode is stored but the material raster viewport is not on screen. DISPLAY TRANSFORM: Rendered and raster preview use PostProcess/ColorMath.h. display_* reports the GPU transport; post.get reports user gain, while post.get_exposure reports the resolved adaptive gain. None/linear means clipping; Reinhard is explicit. Raster preview uses the resolved EV from the Rendered HDR meter and does not meter its own display-encoded image.",
    "render", "Render", false, "any",
    "viewport|preview|lighting|scene|lights|measure",
    "viewport.set_preview_lighting|viewport.set_shading|lights.list",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_viewport_preview_lighting(desc_viewport_preview_lighting);

static const MethodDescriptor desc_viewport_quality = {
    "viewport.quality", "viewport",
    "Report the raster viewport quality preset and, as a value, whether scatter proxy substitution is active",
    "Read values rather than matching on the preset name. 'scatter_lod_split' changes geometry submission. Scene shadows keep one fixed 4096 atlas so preset changes never destroy an image referenced by an in-flight frame; shadow_tile_resolution, shadow_tile_capacity, shadow_light_budget and shadow_pcf_samples report the live bounded contract. directional_shadow_cascades is 2 for Performance and 3 otherwise; directional lights and the Physical Sky sun consume that many atlas tiles and select the smallest camera-centred projection containing the receiver. A scene light beyond the budget still illuminates but has no atlas shadow. 'scene_pbr_shader' reports the comparison shader (GGX for Scene). The material parity strings are an explicit capability contract: opaque_core_parity is RT-aligned; material_graph_surface is bounded because pointiness/named attributes/AO/time inputs are neutral in raster; clearcoat is an iridescent lobe using the RT thin-film model; subsurface is a radius/scale profile approximation; translucency is a bounded thin-surface approximation; surface_anisotropy is unsupported because the current ABI fields are also legacy water wave controls; transparency is unsorted_alpha; transmission is screen_space_thickness with frozen opaque color/depth, bounded screen-space reflection, closed-mesh front/back thickness, RT texture/graph/IOR/Fresnel/Beer/roughness/dispersion semantics and environment fallback; overlapping or off-screen continuation/reflection remains bounded; resin_interior is a procedural bounded approximation; sdf_surface=shared_nanovdb_depth_pbr means realtime reads Vulkan RT's same field/transform/iso threshold and writes real raster depth, while lighting and environment continuation remain bounded rather than recursive. 'raster_viewport_available' false means the preset is stored but nothing on this machine reads it.",
    "render", "Render", false, "any",
    "viewport|quality|lod|proxy|scatter|measure",
    "viewport.set_quality|viewport.frame_telemetry",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_viewport_quality(desc_viewport_quality);

static const MethodParam params_viewport_render_frames[] = {
    {"count", "int", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_viewport_render_frames = {
    "viewport.render_frames", "viewport",
    "Render a fixed number of viewport frames and report timing and convergence",
    "Use this to converge the viewport deliberately before probing it, instead of guessing how long to wait. Over IPC the captured frame is refreshed between calls by the display loop, so viewport.capture -> viewport.render_frames -> render.probe works. It does NOT work inside a single script.run_file: the script holds the main thread, the display loop never runs, and viewport.status frame_available stays false however many frames were rendered - measured 2026-08-19.",
    "render", "Render", false, "any",
    "viewport|render|frames|measure|converge|samples|wait",
    "render.probe|viewport.capture|viewport.status",
    nullptr, nullptr, nullptr, nullptr,
    params_viewport_render_frames, 1,
    true
};
static const MethodRegistration reg_viewport_render_frames(desc_viewport_render_frames);

static const MethodParam params_viewport_set_preview_lighting[] = {
    {"preset", "string", true, "Canonical mode name. Realtime/real_lights aliases map to scene; legacy classic/studio/outdoor aliases map to three_point.", nullptr, "three_point|scene"},
};
static const MethodDescriptor desc_viewport_set_preview_lighting = {
    "viewport.set_preview_lighting", "viewport",
    "Set realtime/material raster lighting to the default scene path or the three-point inspection rig",
    "'scene' uses scene lights plus the canonical world background/ambient; 'three_point' isolates material inspection from scene lighting. Directional, point, spot and area lights share a bounded shadow atlas; Nishita adds a direct world sun with one reserved directional tile. Physical Sky uses the Vulkan atmosphere LUT when available and an analytic fallback otherwise. HDRI uses generated irradiance, GGX prefilter mips and a BRDF LUT when ready, with an explicitly reported raw-environment fallback. Resets accumulation and only takes visible effect in 'material' shading mode.",
    "render", "Render", false, "any",
    "viewport|set|preview|lighting|scene|lights|shading",
    "viewport.preview_lighting|viewport.set_shading|viewport.get_screenshot",
    nullptr, nullptr, "viewport.preview_lighting", nullptr,
    params_viewport_set_preview_lighting, 1,
    true
};
static const MethodRegistration reg_viewport_set_preview_lighting(desc_viewport_set_preview_lighting);

static const MethodParam params_viewport_set_quality[] = {
    {"preset", "string", true, "Canonical preset name. 'no_proxy' and 'full (no proxy)' are accepted aliases for full because that is what the panel combo reads.", nullptr, "auto|performance|balanced|quality|full"},
};
static const MethodDescriptor desc_viewport_set_quality = {
    "viewport.set_quality", "viewport",
    "Set the raster viewport quality preset; 'full' disables scatter proxy substitution entirely",
    "The preset controls both geometry and bounded Scene quality. Performance/Balanced/Quality select 256/512/1024 shadow tiles, 4/8/16 shadowed scene lights and 9/9/25 receiver PCF samples; Auto currently maps to Balanced. The atlas remains 4096 and is not reallocated. 'full' additionally turns scatter proxy substitution OFF so every visible instance draws its own mesh; frustum culling stays on. Rebuilds the raster scene and resets accumulation, so render frames again before probing.",
    "render", "Render", false, "any",
    "viewport|set|quality|lod|proxy|scatter|full|performance",
    "viewport.quality|viewport.frame_telemetry|viewport.render_frames",
    nullptr, nullptr, "viewport.quality|viewport.frame_telemetry", nullptr,
    params_viewport_set_quality, 1,
    true
};
static const MethodRegistration reg_viewport_set_quality(desc_viewport_set_quality);

static const MethodParam params_viewport_set_shading[] = {
    {"mode", "string", true, "Canonical mode name. 'preview' is accepted as an alias for material because the panel button reads Preview.", nullptr, "solid|material|rendered|matcap"},
    {"matcap_preset", "int", false, "0..9, meaningful in matcap mode; -1 leaves it unchanged.", "-1", nullptr},
};
static const MethodDescriptor desc_viewport_set_shading = {
    "viewport.set_shading", "viewport",
    "Switch the viewport shading mode (solid, material, rendered, matcap) and optionally the matcap preset",
    "Resets accumulation like the panel buttons do, so probe AFTER switching - otherwise you measure the frame from the mode you left. Fails loudly when a mode is unavailable instead of silently falling back to rendered.",
    "render", "Render", false, "any",
    "viewport|set|shading|display|solid|rendered|matcap|preview",
    "viewport.shading|viewport.render_frames|render.probe",
    nullptr, nullptr, nullptr, nullptr,
    params_viewport_set_shading, 2,
    true
};
static const MethodRegistration reg_viewport_set_shading(desc_viewport_set_shading);

static const MethodDescriptor desc_viewport_shading = {
    "viewport.shading", "viewport",
    "Report which viewport shading mode is on screen and whether the interactive raster viewport exists",
    "interactive_available false means this build has no raster viewport (no Vulkan), so 'rendered' is the only selectable mode - a rejected viewport.set_shading there is the machine, not a bad request.",
    "render", "Render", false, "any",
    "viewport|shading|display|measure",
    "viewport.set_shading|viewport.status",
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_world_get(desc_world_get);

static const MethodDescriptor desc_world_get_atmosphere = {
    "world.get_atmosphere", "world",
    "Return the physical atmosphere the sky LUT is baked from",
    "Separate from world.get on purpose: world.get is the SUN plus the two intensities, this is the MEDIUM. Every field here feeds makeAtmosphereLUTParamsGPU, so every one of them dirties the transmittance/SkyView LUT when it changes.",
    "read", "Read", false, "WorldAtmosphereInfo",
    "world|get|atmosphere|environment|sky|nishita",
    "world.set_atmosphere|world.get",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_world_get_atmosphere(desc_world_get_atmosphere);

static const MethodDescriptor desc_world_get_thermal = {
    "world.get_thermal", "world",
    "Return the ambient thermal condition every uncoupled substance relaxes toward",
    "Distinct from world.get: that is the render sky, this is WorldThermalState -- the room temperature (ambient_kelvin), the Kelvin-per-normalized-unit calibration a substance's MSF temperature is read against (kelvin_per_unit), the passive-cooling multiplier (convection_coefficient) and the pyrolysis rate scale (oxygen_availability). Formerly no scripting surface existed for any of this -- see docs/dev/SIMULATION_NODE_OBJECT_MODEL.md section 7 item 1.",
    "read", "Read", false, "WorldThermalInfo",
    "world|get|thermal|simulation|ambient|msf",
    "world.set_thermal|sim_graph.list",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_world_get_thermal(desc_world_get_thermal);

static const MethodParam params_world_set_atmosphere[] = {
    {"air_density", "any", false, "Rayleigh scattering multiplier (Blender 'Air'). 1 = Earth. Must be >= 0.", nullptr, nullptr},
    {"dust_density", "any", false, "Mie/aerosol scattering multiplier (Blender 'Dust'). Must be >= 0.", nullptr, nullptr},
    {"ozone_density", "any", false, "Ozone column multiplier; drives blue-hour saturation. Must be >= 0.", nullptr, nullptr},
    {"ozone_absorption_scale", "any", false, "Scales the ozone absorption coefficients on top of ozone_density. Must be >= 0.", nullptr, nullptr},
    {"humidity", "any", false, "0..1, dry to hazy. Rejected outside that range.", nullptr, nullptr},
    {"temperature", "any", false, "Air temperature in CELSIUS. Scales BOTH Rayleigh and Mie scale heights by (T+273.15)/288.15, so it moves the whole atmosphere profile, not just a tint.", nullptr, nullptr},
    {"altitude", "any", false, "Viewer height above sea level in METRES; the SkyView LUT is baked from this camera altitude.", nullptr, nullptr},
    {"mie_anisotropy", "any", false, "Henyey-Greenstein g for the Mie phase function; forward scattering as it approaches 1. Must be within (-1, 1).", nullptr, nullptr},
    {"planet_radius", "any", false, "Ground sphere radius in METRES (Earth = 6360000). Must be >= 1000.", nullptr, nullptr},
    {"atmosphere_height", "any", false, "Atmosphere shell thickness above the ground in METRES (Earth = 60000). Must be >= 1000.", nullptr, nullptr},
    {"rayleigh_scattering", "vec3", false, "Per-channel Rayleigh scattering coefficients as [r, g, b], in 1/metre.", nullptr, nullptr},
    {"mie_scattering", "vec3", false, "Per-channel Mie scattering coefficients as [r, g, b], in 1/metre.", nullptr, nullptr},
    {"rayleigh_density", "any", false, "Rayleigh SCALE HEIGHT in metres (Earth = 8000). The name says density, the unit is a height. Must be >= 1.", nullptr, nullptr},
    {"mie_density", "any", false, "Mie SCALE HEIGHT in metres (Earth = 1200). Same naming trap as rayleigh_density. Must be >= 1.", nullptr, nullptr},
};
static const MethodDescriptor desc_world_set_atmosphere = {
    "world.set_atmosphere", "world",
    "Set the physical atmosphere parameters of the Nishita sky",
    "Every field optional: omit a key to leave it unchanged. Out-of-range values are REJECTED, not clamped, because a silently clamped planet radius or scale height reads as 'the sky looks odd' rather than as an error. Any changed field marks the atmosphere LUT dirty; the runtime rebuilds it on the GPU compute path (throttled to about 15 Hz while a value is being dragged) and falls back to a CPU bake only when that pipeline is missing. UNITS ARE ABSOLUTE: planet_radius/atmosphere_height/altitude are METRES, rayleigh_density and mie_density are SCALE HEIGHTS in metres (NOT densities, despite the field names), temperature is CELSIUS and scales both scale heights.",
    "write", "SceneWrite", false, "any",
    "world|set|atmosphere|environment|sky|nishita|scattering",
    "world.get_atmosphere|world.set_atmosphere_intensity|world.set_sun_elevation",
    nullptr, nullptr, nullptr, nullptr,
    params_world_set_atmosphere, 14,
    true
};
static const MethodRegistration reg_world_set_atmosphere(desc_world_set_atmosphere);

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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
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
    nullptr, nullptr, nullptr, nullptr,
    params_world_set_sun_size, 1,
    true
};
static const MethodRegistration reg_world_set_sun_size(desc_world_set_sun_size);

static const MethodParam params_world_set_thermal[] = {
    {"ambient_kelvin", "any", false, "Room temperature everything relaxes toward, in Kelvin. Must be positive.", nullptr, nullptr},
    {"kelvin_per_unit", "any", false, "Kelvin per normalized solver unit -- the calibration MSF temperature reads through. Must be positive.", nullptr, nullptr},
    {"convection_coefficient", "any", false, "Scales every substance's passive cooling toward ambient: 1 = authored, higher = draughty, 0 = a perfect thermos.", nullptr, nullptr},
    {"oxygen_availability", "any", false, "0..1, scales pyrolysis burn rate; 0 smothers combustion entirely. Clamped, not rejected, if out of range.", nullptr, nullptr},
};
static const MethodDescriptor desc_world_set_thermal = {
    "world.set_thermal", "world",
    "Set the ambient thermal condition every uncoupled substance relaxes toward",
    "Every field optional: omit a key to leave it unchanged. oxygen_availability is clamped to 0..1 rather than rejected. A domain's own thermal_override_enabled (SimulationGridDomainDesc) still wins over this where it is set -- this call changes only the default a domain has not overridden.",
    "write", "SceneWrite", false, "any",
    "world|set|thermal|simulation|ambient|msf",
    "world.get_thermal",
    nullptr, nullptr, nullptr, nullptr,
    params_world_set_thermal, 4,
    true
};
static const MethodRegistration reg_world_set_thermal(desc_world_set_thermal);

} // namespace
