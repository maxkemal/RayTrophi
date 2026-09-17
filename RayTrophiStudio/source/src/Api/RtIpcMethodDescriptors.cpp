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

static const MethodParam params_anim_bind_clip[] = {
    {"source_character", "any", true, "Imported source rig name", nullptr, nullptr},
    {"source_clip", "any", true, "Canonical clip name owned by source_character", nullptr, nullptr},
    {"target_character", "any", true, "Imported destination rig name", nullptr, nullptr},
    {"output_name", "string", false, "Optional new unique canonical clip name", "", nullptr},
    {"node_map", "any", false, "Optional source unique node name -> target unique node name overrides. Full parent/helper chains must map consistently; unknown nodes and duplicate targets are rejected.", nullptr, nullptr},
    {"mode", "string", false, "same_rig copies absolute local TRS. rest_basis transfers rest-frame motion deltas for matching parent chains; positive uniform rest/animated scale required.", "same_rig", "[\"['same_rig', 'rest_basis']\"]"},
    {"translation_scale", "float", false, "Positive world-space translation delta multiplier, at most 10000, only in rest_basis mode. Includes source/target parent rest unit conversion. Does not estimate limb proportions or solve contact.", "1.", nullptr},
};
static const MethodDescriptor desc_anim_bind_clip = {
    "anim.bind_clip", "anim",
    "Bake an imported-rig clip onto a target with optional rest-basis retarget conversion and undo",
    "Default same_rig behavior is preserved. Preview runs the same clone/conversion used by bind; ready means mapping and numerical validation passed, not a live visual preview. rest_basis supports matching parent chains, rest-oriented rotation deltas, target rest translation plus scaled motion and relative uniform scale. No IK, contacts, role inference, reparenting or persisted mapping presets.",
    "write", "SceneWrite", true, "ClipBindingReport",
    "anim|bind|clip",
    "anim.preview_clip_binding|anim.clips|anim.source_channels",
    nullptr, nullptr, "anim.clips|anim.source_channels", nullptr,
    params_anim_bind_clip, 7,
    true
};
static const MethodRegistration reg_anim_bind_clip(desc_anim_bind_clip);

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

static const MethodParam params_anim_force_state[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"state", "string", true, "", nullptr, nullptr},
    {"node_id", "int", false, "", "0", nullptr},
};
static const MethodDescriptor desc_anim_force_state = {
    "anim.force_state", "anim",
    "Jump a character's state machine straight to a named state",
    "Cancels any running transition. node_id 0 targets the first state machine. An unknown state name is reported, never silently ignored.",
    "write", "SceneWrite", false, "any",
    "anim|force|state|animation|statemachine",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_anim_force_state, 3,
    true
};
static const MethodRegistration reg_anim_force_state(desc_anim_force_state);

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

static const MethodParam params_anim_preview_clip_binding[] = {
    {"source_character", "any", true, "Imported source rig name", nullptr, nullptr},
    {"source_clip", "any", true, "Canonical clip name owned by source_character", nullptr, nullptr},
    {"target_character", "any", true, "Imported destination rig name", nullptr, nullptr},
    {"node_map", "any", false, "Optional source unique node name -> target unique node name overrides. Full parent/helper chains must map consistently; unknown nodes and duplicate targets are rejected.", nullptr, nullptr},
    {"mode", "string", false, "same_rig copies absolute local TRS. rest_basis transfers rest-frame motion deltas for matching parent chains; positive uniform rest/animated scale required.", "same_rig", "[\"['same_rig', 'rest_basis']\"]"},
    {"translation_scale", "float", false, "Positive world-space translation delta multiplier, at most 10000, only in rest_basis mode. Includes source/target parent rest unit conversion. Does not estimate limb proportions or solve contact.", "1.", nullptr},
};
static const MethodDescriptor desc_anim_preview_clip_binding = {
    "anim.preview_clip_binding", "anim",
    "Validate imported-rig clip mapping and optional rest-basis retarget conversion without scene mutation",
    "Default same_rig behavior is preserved. Preview runs the same clone/conversion used by bind; ready means mapping and numerical validation passed, not a live visual preview. rest_basis supports matching parent chains, rest-oriented rotation deltas, target rest translation plus scaled motion and relative uniform scale. No IK, contacts, role inference, reparenting or persisted mapping presets.",
    "read", "Read", false, "ClipBindingReport",
    "anim|preview|clip|binding",
    "anim.bind_clip|rig.list_characters|anim.source_clips|anim.source_channels",
    nullptr, nullptr, nullptr, nullptr,
    params_anim_preview_clip_binding, 6,
    true
};
static const MethodRegistration reg_anim_preview_clip_binding(desc_anim_preview_clip_binding);

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

static const MethodParam params_anim_sample_clip_binding[] = {
    {"source_character", "any", true, "Imported source rig name", nullptr, nullptr},
    {"source_clip", "any", true, "Canonical clip name owned by source_character", nullptr, nullptr},
    {"target_character", "any", true, "Imported destination rig name", nullptr, nullptr},
    {"node_map", "any", false, "Optional source unique node name -> target unique node name overrides. Full parent/helper chains must map consistently; unknown nodes and duplicate targets are rejected.", nullptr, nullptr},
    {"mode", "string", false, "same_rig copies absolute local TRS. rest_basis transfers rest-frame motion deltas for matching parent chains; positive uniform rest/animated scale required.", "same_rig", "[\"['same_rig', 'rest_basis']\"]"},
    {"translation_scale", "float", false, "Positive world-space translation delta multiplier, at most 10000, only in rest_basis mode. Includes source/target parent rest unit conversion. Does not estimate limb proportions or solve contact.", "1.", nullptr},
    {"time_seconds", "float", false, "Finite nonnegative time in seconds, looped by clip duration. Both skeletons sample the same time.", "0.", nullptr},
    {"source_pose_view", "string", false, "", "animated", "['rest', 'animated']"},
    {"target_pose_view", "string", false, "", "animated", "['rest', 'animated']"},
};
static const MethodDescriptor desc_anim_sample_clip_binding = {
    "anim.sample_clip_binding", "anim",
    "Sample synchronized source and target model-space skeleton poses without changing scene playback",
    "Uses the same imported-rig mapping/retarget clone as bind. Returns all hierarchy nodes with unique name, parent name and row-major 16-float model-space world_transform. Seconds loop at clip duration using the existing sampler. Incompatible mapping preflight returns binding.ready=false with animated source and target bind fallback; invalid parameters/conversion return named errors. No controller, selection, history or scene mutation. No scene placement or contact IK. Optional source_pose_view and target_pose_view (animated/rest, default animated) independently choose canonical rest or sampled clip for each preview side. Does not change scene rig pose view, binding report or saved clips. Returns source_pose_source and target_pose_source. Invalid selectors return invalid_rig_pose_view.",
    "Read", "Read", false, "ClipPosePreview {binding,time_seconds,duration_seconds,target_pose_source,source[],target[]}",
    "anim|sample|clip|binding",
    "anim.preview_clip_binding|anim.bind_clip|rig.select_bone",
    nullptr, nullptr, nullptr, nullptr,
    params_anim_sample_clip_binding, 9,
    true
};
static const MethodRegistration reg_anim_sample_clip_binding(desc_anim_sample_clip_binding);

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

static const MethodParam params_anim_state_machines[] = {
    {"character", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_anim_state_machines = {
    "anim.state_machines", "anim",
    "Report the live state machines of a character's runtime animation graph",
    "Reads the RUNTIME graph clone the renderer evaluates, not the editor asset. Each state carries pose_connected: false means that state resolves to an EMPTY pose, which freezes the character silently instead of erroring.",
    "read", "Read", false, "any",
    "anim|state|machines|animation|statemachine|inventory",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_anim_state_machines, 1,
    true
};
static const MethodRegistration reg_anim_state_machines(desc_anim_state_machines);

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
    "EXPOSURE IS A RELATIVE MODEL, NOT AN ABSOLUTE PHOTOMETRIC ONE. The textbook formula 1/(1.2*2^EV100) assumes scene radiance in cd/m2; this engine's light intensities are arbitrary, so the multiplier is a ratio against a calibrated baseline (0.00003125) chosen to avoid a black viewport. Do not 'fix' it into the absolute formula - every scene would go black and the symptom would read as 'too dark', not 'wrong formula'. 'iso_value', 'shutter_seconds', 'f_number' and 'exposure_factor' are DERIVED read-only outputs, not settings: a preset INDEX is not a measurement, and exposure_factor is the number actually handed to the shaders. 'aperture' is the depth-of-field dial and does NOT affect exposure; 'fstop_preset_index' is the exposure/lens dial and does NOT affect depth of field. PRIORITY ORDER, and it is the most common reason these dials look dead: the exposure MODE now lives in the post chain, not on the camera. Read post.get_exposure 'mode' - only 'physical' reads ISO/shutter/f-stop; under 'manual' and 'auto_histogram' they are NOT read and the camera term is 1.0. Set it first with post.configure_exposure {\"settings\":{\"mode\":\"physical\"}}. The camera's own auto_exposure / use_physical_exposure flags are LEGACY: syncDisplay forces both in physical mode, so toggling them does not change the Vulkan or Rendered image. These setters still SUCCEED because they really do write the setting - the image not moving is the model's precedence, not a fault. Read camera.get exposure_factor for the multiplier ACTUALLY applied (it is g_display_post.camera_exposure, so it reads 1.0 whenever the mode is not physical).",
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
    "The AMOUNT, not the switch: writing an aperture does not turn depth of field on. Call camera.set_depth_of_field first (or read camera.get 'depth_of_field'), then read 'effective_lens_radius' - it is 0 whenever the lens is off, and that, not 'aperture', is what every renderer samples. The aperture is the twin of the f-number: camera.get 'f_number' is derived from it through the focal length, and the panel f-stop dial writes both.",
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
    "LEGACY flag - the exposure mode lives in post.configure_exposure; this does not change the image",
    "PRIORITY ORDER, and it is the most common reason these dials look dead: the exposure MODE now lives in the post chain, not on the camera. Read post.get_exposure 'mode' - only 'physical' reads ISO/shutter/f-stop; under 'manual' and 'auto_histogram' they are NOT read and the camera term is 1.0. Set it first with post.configure_exposure {\"settings\":{\"mode\":\"physical\"}}. The camera's own auto_exposure / use_physical_exposure flags are LEGACY: syncDisplay forces both in physical mode, so toggling them does not change the Vulkan or Rendered image. These setters still SUCCEED because they really do write the setting - the image not moving is the model's precedence, not a fault. Read camera.get exposure_factor for the multiplier ACTUALLY applied (it is g_display_post.camera_exposure, so it reads 1.0 whenever the mode is not physical).",
    "write", "SceneWrite", false, "any",
    "camera|set|auto|exposure",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_camera_set_auto_exposure, 1,
    true
};
static const MethodRegistration reg_camera_set_auto_exposure(desc_camera_set_auto_exposure);

static const MethodParam params_camera_set_depth_of_field[] = {
    {"enabled", "bool", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_camera_set_depth_of_field = {
    "camera.set_depth_of_field", "camera",
    "Turn the camera lens on or off; off means a pinhole and nothing blurs",
    "THIS IS THE SWITCH, 'aperture' IS THE AMOUNT. Until 2026-09-06 the only off switch was aperture == 0, which is a sentinel rather than a value: no f-number produces a zero opening, so once the f-stop dial started writing the aperture there was no way back and depth of field could never be turned off again. Turning this off does NOT clear the aperture, so turning it back on restores exactly the previous blur. Enabling it on a camera whose aperture was never written seeds the aperture from the current f-number, otherwise the call would succeed and change nothing. Everything downstream (CPU renderer, OptiX, Vulkan RT, the realtime raster DoF pass, the viewport focus ring and the AF tolerance) reads camera.get 'effective_lens_radius', which is 0 while this is off.",
    "write", "SceneWrite", false, "any",
    "camera|set|depth|of|field|lens|dof|blur|bokeh|pinhole|switch",
    "camera.set_aperture|camera.set_focus_distance|camera.get|viewport.get_depth_of_field",
    nullptr, nullptr, nullptr, nullptr,
    params_camera_set_depth_of_field, 1,
    true
};
static const MethodRegistration reg_camera_set_depth_of_field(desc_camera_set_depth_of_field);

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
    "Select the f-stop preset by index; drives exposure, the physical aperture and Cinema lens imperfections",
    "The recorded debt is PAID (2026-09-06): this preset now writes the physical 'aperture' as well, so the panel f-stop dial, the viewport HUD triangle and this method all produce the same f-number - previously turning one left the others showing a different value. It still does NOT force blur on: the lens is sampled only while camera.set_depth_of_field is on, and camera.get 'effective_lens_radius' is the value every renderer actually uses. Index 0 is 'Custom' and leaves the aperture untouched. PRIORITY ORDER, and it is the most common reason these dials look dead: the exposure MODE now lives in the post chain, not on the camera. Read post.get_exposure 'mode' - only 'physical' reads ISO/shutter/f-stop; under 'manual' and 'auto_histogram' they are NOT read and the camera term is 1.0. Set it first with post.configure_exposure {\"settings\":{\"mode\":\"physical\"}}. The camera's own auto_exposure / use_physical_exposure flags are LEGACY: syncDisplay forces both in physical mode, so toggling them does not change the Vulkan or Rendered image. These setters still SUCCEED because they really do write the setting - the image not moving is the model's precedence, not a fault. Read camera.get exposure_factor for the multiplier ACTUALLY applied (it is g_display_post.camera_exposure, so it reads 1.0 whenever the mode is not physical).",
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
    "The index is not a measurement - read 'iso_value' from camera.get for the actual ISO. PRIORITY ORDER, and it is the most common reason these dials look dead: the exposure MODE now lives in the post chain, not on the camera. Read post.get_exposure 'mode' - only 'physical' reads ISO/shutter/f-stop; under 'manual' and 'auto_histogram' they are NOT read and the camera term is 1.0. Set it first with post.configure_exposure {\"settings\":{\"mode\":\"physical\"}}. The camera's own auto_exposure / use_physical_exposure flags are LEGACY: syncDisplay forces both in physical mode, so toggling them does not change the Vulkan or Rendered image. These setters still SUCCEED because they really do write the setting - the image not moving is the model's precedence, not a fault. Read camera.get exposure_factor for the multiplier ACTUALLY applied (it is g_display_post.camera_exposure, so it reads 1.0 whenever the mode is not physical).",
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
    "Index, not seconds. Read 'shutter_seconds' from camera.get for the resolved exposure time. This dial does NOT drive motion blur. PRIORITY ORDER, and it is the most common reason these dials look dead: the exposure MODE now lives in the post chain, not on the camera. Read post.get_exposure 'mode' - only 'physical' reads ISO/shutter/f-stop; under 'manual' and 'auto_histogram' they are NOT read and the camera term is 1.0. Set it first with post.configure_exposure {\"settings\":{\"mode\":\"physical\"}}. The camera's own auto_exposure / use_physical_exposure flags are LEGACY: syncDisplay forces both in physical mode, so toggling them does not change the Vulkan or Rendered image. These setters still SUCCEED because they really do write the setting - the image not moving is the model's precedence, not a fault. Read camera.get exposure_factor for the multiplier ACTUALLY applied (it is g_display_post.camera_exposure, so it reads 1.0 whenever the mode is not physical).",
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
    "LEGACY flag - use post.configure_exposure mode='physical' to actually read the ISO/shutter/f-stop chain",
    "PRIORITY ORDER, and it is the most common reason these dials look dead: the exposure MODE now lives in the post chain, not on the camera. Read post.get_exposure 'mode' - only 'physical' reads ISO/shutter/f-stop; under 'manual' and 'auto_histogram' they are NOT read and the camera term is 1.0. Set it first with post.configure_exposure {\"settings\":{\"mode\":\"physical\"}}. The camera's own auto_exposure / use_physical_exposure flags are LEGACY: syncDisplay forces both in physical mode, so toggling them does not change the Vulkan or Rendered image. These setters still SUCCEED because they really do write the setting - the image not moving is the model's precedence, not a fault. Read camera.get exposure_factor for the multiplier ACTUALLY applied (it is g_display_post.camera_exposure, so it reads 1.0 whenever the mode is not physical).",
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

static const MethodDescriptor desc_perf_get_gpu_memory = {
    "perf.get_gpu_memory", "perf",
    "VRAM by device (render / dedicated viewport) and purpose, BLAS compaction savings, and the driver's process-wide usage.",
    "'vram_usage_bytes' is the driver's VK_EXT_memory_budget number for the WHOLE PROCESS; with a dedicated viewport VkDevice and a render VkDevice on one card it cannot say which is full - that is what 'devices' is for. Each device lists its own allocations by category (geometry, accel_struct, scratch, texture, render_target, other) in device-local and host-visible bytes. 'untracked_bytes' = usage minus the sum of tracked device-local bytes: CUDA/OptiX, simulation compute, the exposure meter and driver overhead live there, and a large value means the bytes are NOT in a VulkanDevice allocation. 'vram_measured' false means the zeros are not a reading. 'tracked' false on a device means it is not Vulkan (OptiX/CPU) and has no categories. 'compaction.bytes_before/bytes_after' cover only the BLAS that were compacted; 'skipped_skinned' counts skinned BLAS, which are rebuilt in place and therefore left at build size (refit-only ALLOW_UPDATE BLAS ARE compacted; their MODE_UPDATE reuses the recorded build flags), and 'failures' counts BLAS kept at build size because the copy could not be made. Read it in Solid, then in Rendered after a frame: the viewport device should lose its accel_struct bytes while Rendered is shown (see rayfusion.scene_as 'yielded').",
    "read", "Read", false, "any",
    "perf|get|gpu|memory|vulkan",
    "rayfusion.scene_as|perf.set_blas_compaction",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_perf_get_gpu_memory(desc_perf_get_gpu_memory);

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

static const MethodParam params_perf_set_blas_compaction[] = {
    {"enabled", "bool", true, "true compacts static BLAS built from now on; false keeps build-size allocations.", nullptr, nullptr},
};
static const MethodDescriptor desc_perf_set_blas_compaction = {
    "perf.set_blas_compaction", "perf",
    "Process-wide BLAS compaction switch; applies to acceleration structures built afterwards.",
    "Compaction copies each static BLAS into an allocation sized to what the build used; PREFER_FAST_TRACE builds are allocated for the worst case. Traversal and the image are unchanged, the cost is one extra copy submit per few BLAS at build time. The switch survives a backend recreate. It does NOT touch existing structures: switch, then force a rebuild (re-enter Rendered after a geometry change, or reopen the project) and compare perf.get_gpu_memory accel_struct bytes. Kept as a switch so a driver problem in the copy path can be ruled out without a rebuild of the app.",
    "render", "Render", false, "any",
    "perf|set|blas|compaction|memory|vulkan",
    "perf.get_gpu_memory",
    nullptr, nullptr, nullptr, nullptr,
    params_perf_set_blas_compaction, 1,
    true
};
static const MethodRegistration reg_perf_set_blas_compaction(desc_perf_set_blas_compaction);

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

static const MethodParam params_project_autosave_now[] = {
    {"reason", "any", false, "Tag recorded in last_reason so a later status read says what triggered the write. Defaults to 'manual'.", nullptr, nullptr},
};
static const MethodDescriptor desc_project_autosave_now = {
    "project.autosave_now", "project",
    "Force-writes the session autosave now, ignoring the interval and the modified flag.",
    "Writes the session autosave immediately, ignoring both the interval and the modified flag. Reports the FULL autosave status back, not just success, because the useful question after a failure is 'why did it not write' and the answer is in those fields. Autosave was a lie until 2026-09-16: the preference (auto_save_enabled, default true), the restore path (StartupMode::RestoreAutosave) and the Hub's 'Recover the last autosaved scene session' button all existed and NOTHING EVER WROTE THE FILE, so the recovery offered was of a file that could not exist. The writer also has to protect project identity: ProjectManager::saveProject sets current_file_path, renames an Untitled project after the file, and adds the path to the recent list -- correct for a user save, catastrophic for an autosave, because the user's next Ctrl+S would go to autosave.rtp instead of their project. The autosave unit restores path, name and the is_modified flag around the write; is_modified in particular, so an autosave never makes unsaved work look saved and silently suppress the quit prompt. The same writer is called on VK_ERROR_DEVICE_LOST before the viewport backend is torn down.",
    "write", "FilesWrite", false, "any",
    "project|autosave|now",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_project_autosave_now, 1,
    true
};
static const MethodRegistration reg_project_autosave_now(desc_project_autosave_now);

static const MethodDescriptor desc_project_autosave_status = {
    "project.autosave_status", "project",
    "Reads the autosave instrument without writing.",
    "Reads the autosave instrument without writing. Read write_count FIRST: if it is 0 while the app has been open longer than interval_sec, autosave is not running at all. Then read skipped_unmodified -- if THAT is growing while write_count stays 0, the broken thing is not autosave but the is_modified flag, which is why the two counters are separate rather than one 'healthy' boolean. seconds_until_next goes negative only between the due moment and the next tick. file_exists plus file_bytes describe the file on disk, so a status that claims success while file_bytes stays 0 means the write path is reporting an outcome it did not achieve. last_reason distinguishes an interval write from the forced one taken on VK_ERROR_DEVICE_LOST.",
    "write", "SceneWrite", false, "any",
    "project|autosave|status",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_project_autosave_status(desc_project_autosave_status);

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

static const MethodDescriptor desc_rayfusion_core_status = {
    "rayfusion.core_status", "rayfusion",
    "Report RayFusion probe control-plane availability and planned quality budgets",
    "Phase 0.1 only: renderer_available and gi_active are false until GPU tracing and raster composition are connected. planned_* values are budgets, not dispatched rays or measured performance. Reads the existing viewport quality preset; changes no settings.",
    "read", "Read", false, "RayFusionCoreInfo",
    "rayfusion|core|status|gi|probe|development",
    "rayfusion.validate_core|viewport.quality",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_rayfusion_core_status(desc_rayfusion_core_status);

static const MethodDescriptor desc_rayfusion_probe_field = {
    "rayfusion.probe_field", "rayfusion",
    "RayFusion probe field as it actually reaches the GPU: grid, fill progress and which producer filled it.",
    "Step 1a. 'producer' is 'sky_bake' while the field is filled from the baked sky irradiance, which means the rendered image is meant to look IDENTICAL to the direct sky read - the counters, not the picture, are what show the pipe is live. 'uploaded' false with 'configured' true means the CPU field exists but no slot has reached the GPU, so shading still falls back to the global sky texture. 'valid' climbs toward 'total' over several frames because scheduling is budget-limited by 'budget_preset'. 'rejected' counts stale or duplicate tickets and should stay flat once the field is full; a climbing 'rejected' means results are arriving against a revision that has already been invalidated. Real indirect light needs probe rays, which need an acceleration structure in the raster viewport - until then this reports a working pipe, not GI. STEP 1b: 'producer' reports what actually ran - 'traced' (rays), 'sky_bake' (the step-1a bake), or 'none' - never the request; 'producer_traced_requested' is the request and 'producer_reason' explains any refusal. 'hit_fraction' is the share of probe rays that hit anything: if it stays 0.0 while geometry is clearly within range, the rays are not seeing the scene at all, and the result then looks EXACTLY like the sky-only producer - a correct-looking image that measures nothing. 'rejected_inside' counts probes born inside geometry; those publish with alpha 0 so the slot stops being re-traced while the shader falls back to the global read instead of going black. 'trace_ms' is the GPU dispatch plus readback for the last batch. Switch producers with rayfusion.set_probe_producer. STEP 1b-beta bounce keys: 'bounce_supported_materials' is the number of materials actually inside the single-diffuse-bounce subset, and it is the one to read first - 'bounce_unsupported_materials' alone cannot be checked against the scene, because a fully textured scene and a gate that rejects everything print the same number. When supported is 0 the bounce cannot change a single pixel and the image is identical to 1b-alpha, which is why 'bounce_reason' says so in words. 'bounce_rejected_textured', 'bounce_rejected_transparent', 'bounce_rejected_layered' and 'bounce_rejected_flagged' name the clause that rejected each material and overlap (a material can trip several), so they do not sum to the unsupported count. 'bounce_rejected_flag_bits' is the OR of the material flags that tripped the flag clause: its low bits are packed texture-channel selectors rather than material features, so a material that looks plain in the material panel can still be rejected here, and this field is what names the bit. STEP 1b-beta TEXTURE slice: 'bounce_shaded_hits' is the acceptance number and the only one measured on the GPU rather than counted off the CPU table -- how many probe rays actually landed on a material this slice could shade, during the last dispatch. 'bounce_hits' greater than zero with 'bounce_shaded_hits' at zero means every ray met an occluder and the published field is byte-identical to the bounce being off; the material counts cannot report that, because they describe what the slice COULD shade, not what it DID. 'bounce_alpha_tested' and 'bounce_alpha_occluded' measure the alpha cutout: shadow and primary rays no longer carry the opaque flag, so a masked leaf or curtain lets light through instead of shadowing like a solid sheet, and the candidate loop that costs is reported rather than assumed. Albedo, emission, opacity, metallic and specular textures are read at the bounce hit through the SAME bindless array and the same packed-channel policy the RT closest-hit uses, with the same UV transform; roughness, normal and height maps are deliberately not read (a diffuse lobe never reads roughness, and a probe texel is a hemisphere average, below the frequency of a normal map). transmission_tex, scalar opacity below 1 and transmission above 0 still fall outside the slice. INDIRECT SUN: 'bounce_sun_included' says whether the Physical Sky sun is carried as a directional bounce light. Until 2026-09-15 it was NOT, and the bounce could therefore only carry SKY light: an interior lit through a window went blue instead of taking the warm bounce off the sunlit floor. Measured against the RT reference on the same scene and settings, the ceiling ratio was 3.2/7.0/12.0 in R/G/B -- a blue-weighted excess, not a scalar gain, which is the signature of missing sun transport rather than a double-applied intensity. 'bounce_sun_tint_from_lut' says whether the sun tint came from the real transmittance LUT or from the shader's constant fallback; the fallback is visibly less warm at low sun elevations, so it is reported rather than applied silently. PER-FRAME CPU COST: 'bounce_prepare_ms' is how long the hit/material/light tables took to rebuild on the CPU. The emissive triangle scan behind it is now gated on an INPUT signature (geometry + instances + materials): measured 2026-09-15 it was 8.89 ms of a 9.21 ms total, every frame, in a scene with zero emissive triangles. 'bounce_emissive_cached' distinguishes 'the cache hit' from 'the scan never ran', which a bare 0.00 ms cannot. Read it beside rayfusion.scene_as 'signature_ms' (the AS change gate, also per frame): together they are what RayFusion spends before a single ray is cast. Measured 2026-09-08 in a foliage scene, the table build re-hashed each mesh's whole material-ID stream once per PLACEMENT, so the cost was multiplied by the instance count and one core sat at 100% while the GPU idled -- on a 16-thread machine that reads as '6% CPU', which is why the number has to be here rather than inferred from task-manager totals. WHY A HIT WAS NOT SHADED: 'bounce_shaded_hits' at zero is true but useless on its own, because five different exits produce the same zero. The named counters split it, and bounce_skipped_disabled + bounce_rejected_unresolved + bounce_rejected_unsupported_hit + bounce_rejected_degenerate + bounce_shaded_hits must equal bounce_hits -- if they do not, there is an unnamed exit left. 'bounce_backface_shaded' is NOT part of that sum: it is a SUBSET of the shaded hits, not an exit. RENAMED 2026-09-15 from 'bounce_skipped_backface' because the behaviour changed, not just the label: the front-face gate was REMOVED. Measured in a closed interior, 681 of 806 hits (84.5%) were back faces and every one of them returned black, so enabling the bounce cost CPU and changed almost nothing. The RT reference never rejected them (closesthit.rchit flips the normal on a back face) and rfBounceRadiance already flipped the normal toward the incoming ray, so the gate was discarding hits the shading function could handle correctly. A high value here now just means the scene is an interior, not that anything is wrong. 'bounce_rejected_unresolved' means the hit instance/index/material address could not be resolved at all -- a table problem, not a material one, and the CPU material counts cannot see it. 'bounce_rejected_unsupported_hit' is the per-RAY twin of the CPU-side material counts: it says the material the ray actually landed on is outside the slice, which is a different question from how many materials in the scene are.",
    "read", "Read", false, "any",
    "rayfusion|probe|field|gi|grid|status|development",
    "rayfusion.core_status|rayfusion.validate_core|viewport.preview_lighting",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_rayfusion_probe_field(desc_rayfusion_probe_field);

static const MethodDescriptor desc_rayfusion_reflections = {
    "rayfusion.reflections", "rayfusion",
    "Read per-pixel specular reflection settings and the last measured dispatch",
    "This pass REPLACES the environment radiance lookup rather than adding a metal-only term: the fragment shader writes the split-sum weight (F0*brdf.x + brdf.y) into a G-buffer and the compute pass adds weight * (traced - env(R)). Two consequences that decide how to read these numbers. A ray that MISSES changes nothing at all -- the subtracted term is the same lookup, so the difference is exactly zero and there is no seam between traced and untraced regions by construction. And if the pass never runs, the image is identical to reflections being off, so 'ready' false is never a dark image. Because the weight is a Fresnel term, this is NOT metal-only: varnished wood, painted floor, ceramic and plastic inherit it through the same multiplier, and the gate is on the LOBE (roughness_gate plus weight_gate) -- a metallic>x gate would have excluded exactly those surfaces. 'shaded_hits' is the acceptance number and the only one measured on the GPU: 'rays' above zero with 'shaded_hits' at zero means every ray was rejected and the image is byte-identical to the pass being off, which 'gated_pixels' can never report because it only counts how many pixels WANTED a reflection. 'sky_misses' equal to 'rays' also means the image did not change, but for the opposite reason: there is nothing in the scene to reflect. Counters lag ONE FRAME -- they are zeroed in the frame command buffer, filled by the GPU and read by the CPU next frame, so a test that writes a setting and measures immediately sees the PREVIOUS batch. One bounce only: no mirror inside a mirror, and the hit surface contributes diffuse, emission and one direct light, not its own specular. Clearcoat is a second lobe and stays on the environment lookup.",
    "read", "Read", false, "object",
    "rayfusion|reflections|raster|lighting|specular|reflection|same-frame",
    "rayfusion.set_reflections|rayfusion.screen_gi|viewport.frame_timings",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_rayfusion_reflections(desc_rayfusion_reflections);

static const MethodDescriptor desc_rayfusion_scene_as = {
    "rayfusion.scene_as", "rayfusion",
    "Acceleration structure residency in the raster viewport: BLAS/instance counts, bytes and build cost.",
    "Step 2. 'hardware_rt' false means this GPU or driver declined ray tracing, so every ray-based RayFusion step is blocked here and the UI disables them with this reason - 'cannot' and 'has not built yet' are different states and must not be shown the same way. 'as_bytes' is the memory the viewport keeps resident whether or not a ray is traced this frame; that residency, not the tracing, is the cost that has to be budgeted. REBUILD GATE: 'geometry_signature' (the mesh set) and 'instance_signature' (placements) are what trigger work, NOT 'built_geometry_generation' - that global counter is reported for reference only, because scene.delete removes an object from the drawn image without bumping it, and a generation-gated structure kept a DELETED object with no error. A change in the geometry signature rebuilds every BLAS; a change in only the instance signature refreshes the TLAS alone and increments 'tlas_only_refreshes'. 'signature_ms' is what the gate itself costs each frame. 'builds' rising while only the camera moves is a bug. 'instances_skipped' above zero means the instance cap was hit, so part of the raster image is NOT in the traced scene - probe rays read those directions as empty. 'meshes_skipped' counts meshes with no device address or too few vertices. Scatter impostor proxies are excluded on purpose: tracing a camera-facing billboard would put it into the world the rays see. A raster instance whose mesh is gone is skipped rather than traced against a stale BLAS. DELETE IS A HIDE, NOT AN ERASE: scene.delete sets the raster instance mask to 0 and leaves both the mesh and the instance resident so undo is instant, which is why the global geometry generation does not move. Those instances are counted in 'instances_hidden' and are excluded from the TLAS, so 'instance_count' FALLS on delete. 'blas_count' does NOT fall: BLAS residency mirrors raster residency (the hidden mesh still holds its vertex buffer), so it is a memory number, not a scene-content number. VRAM HANDOFF: while the viewport shows Rendered the dedicated viewport device releases this AS so the render backend (same GPU) gets the memory; 'yielded' is true then, 'yields' counts releases, and 'ready' false with that reason is the CORRECT state, not a failure. Returning to a raster mode lifts the gate and the AS rebuilds lazily. 'vram_usage_bytes'/'vram_budget_bytes' are the driver's device-local numbers for the WHOLE PROCESS (every VkDevice); 'vram_measured' false means VK_EXT_memory_budget is missing and the zeros are not a reading. Measure the handoff by reading this in Solid, switching to Rendered, waiting a frame, and reading again: usage must fall by about 'as_bytes'.",
    "read", "Read", false, "any",
    "rayfusion|scene|as|gi|acceleration|blas|tlas|memory|status",
    "rayfusion.probe_field|rayfusion.core_status",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_rayfusion_scene_as(desc_rayfusion_scene_as);

static const MethodDescriptor desc_rayfusion_screen_gi = {
    "rayfusion.screen_gi", "rayfusion",
    "Read same-frame diffuse RTGI settings and last recorded dispatch",
    "Reports MEASUREMENT as well as settings: mean_sky_visibility, mean_gi_luminance and full_confidence_fraction, sampled from 1/64 of pixels and ONE FRAME BEHIND (read without stalling the GPU); measured=false means no measurement yet, never 'zero'. full_confidence_fraction is the discriminator: when it is high the consumer skips the ambient fallback entirely, so the sky-visibility term does NOT apply to those pixels and excess brightness lies in the GI radiance instead. GI owns no history buffer, but sample directions rotate with the TAA accumulation index (temporal_seed_rotation), so noise resolves while TAA converges and is frozen when TAA is off. ready means recorded, not GPU-completed. primary_ray_budget excludes secondary visibility rays. Experimental depth-derived geometric normals; requires RT shadows.",
    "read", "Read", false, "object",
    "rayfusion|screen|gi|raster|lighting|diffuse|same-frame",
    "rayfusion.set_screen_gi|viewport.frame_timings",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_rayfusion_screen_gi(desc_rayfusion_screen_gi);

static const MethodParam params_rayfusion_set_probe_bounce[] = {
    {"enabled", "bool", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rayfusion_set_probe_bounce = {
    "rayfusion.set_probe_bounce", "rayfusion",
    "Enable the bounded single diffuse bounce producer for RayFusion probes.",
    "Requires traced producer and ready hit/material tables. enabled must be a boolean; unknown keys are rejected. applied acknowledges the request, active reports the last published field. Read rayfusion.probe_field bounce_requested/ready/active/reason and unsupported material/light counts. Supports untextured opaque diffuse/emission, environment, point/directional lights only; unsupported hits remain opaque occluders. No volume, alpha, glass, area/spot or analytic sky-sun bounce. At most 192 rays per updated probe including secondary and shadow rays, within the existing quality ray ceiling. Developer switch is session-local. Disable for the 1b-alpha visibility comparison.",
    "render", "Render", false, "any",
    "rayfusion|set|probe|bounce",
    "rayfusion.probe_field|rayfusion.set_probe_producer",
    nullptr, nullptr, nullptr, nullptr,
    params_rayfusion_set_probe_bounce, 1,
    true
};
static const MethodRegistration reg_rayfusion_set_probe_bounce(desc_rayfusion_set_probe_bounce);

static const MethodParam params_rayfusion_set_probe_follow_camera[] = {
    {"enabled", "boolean", true, "Follow camera position with a cell dead band", nullptr, nullptr},
};
static const MethodDescriptor desc_rayfusion_set_probe_follow_camera = {
    "rayfusion.set_probe_follow_camera", "rayfusion",
    "Scroll the fixed-budget probe window with camera translation",
    "Strict boolean enabled; session-local, default false. Rotation does not move the window. Existing world-cell values survive scrolling; only new cells are scheduled. Turning off freezes the current window. Does not perform scene fitting, density changes or relocation.",
    "render", "Render", false, "any",
    "rayfusion|set|probe|follow|camera",
    "rayfusion.probe_field",
    nullptr, nullptr, nullptr, nullptr,
    params_rayfusion_set_probe_follow_camera, 1,
    true
};
static const MethodRegistration reg_rayfusion_set_probe_follow_camera(desc_rayfusion_set_probe_follow_camera);

static const MethodParam params_rayfusion_set_probe_grid[] = {
    {"counts", "array", false, "Cells per axis [x,y,z], each 1..128; the product may not exceed max_slots", nullptr, nullptr},
    {"spacing", "float", false, "World units per cell; probes sit at (cell + 0.5) * spacing", nullptr, nullptr},
    {"minimum", "array", false, "Lowest cell index [x,y,z] in cells (integers); mutually exclusive with center", nullptr, nullptr},
    {"center", "array", false, "World point [x,y,z] to place at the middle of the window; mutually exclusive with minimum", nullptr, nullptr},
    {"auto_fit", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rayfusion_set_probe_grid = {
    "rayfusion.set_probe_grid", "rayfusion",
    "Move or reshape the RayFusion probe window, or hand placement back to auto-fit",
    "'auto_fit' (2026-09-15) is the DEFAULT and owns both shape and placement: the grid is sized to the traced scene bounds and re-fitted when they change. Send auto_fit=false, or any explicit 'counts'/'spacing'/'minimum'/'center', to take placement over -- an explicit grid alongside auto-fit would be accepted, live one frame and be silently rewritten, which reads as 'the request was forgotten' with no error anywhere. Read 'auto_fit_mode' back from rayfusion.probe_field: 'scene' means the whole traced scene fit the slot budget at usable density and follow-camera was turned off; 'camera_local' means it did not (measured 2026-09-15: hit_fraction 3.2%, mean_hit_distance 194 m -- the scene was far too large) so the entire budget is spent on a small volume around the camera instead. Partial edit: every key is optional and what you do not send keeps the value the field already has. The window was a BUILD CONSTANT until now (4x2x4 cells, spacing 3, nailed to the world origin), which is why every RayFusion image question so far has been confounded by coverage: in the measured room scene 6 of the 16 useful probes stood outdoors and hit_fraction read 0.105. A grid that needs a rebuild to change cannot be A/B'd against the thing it is blamed for. 'counts' is cells per axis (1..128 each) and their product may not exceed 'max_slots' from rayfusion.probe_field -- the texel buffer is allocated once for that ceiling so a grid change never reallocates a buffer an in-flight frame is reading. 'spacing' is world units per cell. Placement takes EITHER 'minimum' (the lowest cell index, integers, the same numbers probe_field reports) OR 'center' (world units, placed in the middle of the window and resolved against the FINAL spacing and counts) -- sending both is an error rather than a precedence rule. An explicit placement turns follow_camera OFF, because otherwise following would overwrite it on the next frame and the accepted request would appear to do nothing. Changing counts or spacing rebuilds the slot array and drops every measurement: the cells are different volumes of world, and carrying old packets across would keep a value measured for a different place. Moving only the placement scrolls, so world cells that stay inside the window keep their values and only newly exposed cells are rescheduled. Per-frame RAY cost does not scale with the grid -- it is bounded by the quality budget's probes-per-update -- but a larger window costs convergence time (more frames before valid reaches total) and a larger buffer copy per publish. The reply and rayfusion.probe_field report the APPLIED window, which is still the OLD one until the viewport produces a frame: toggle, move the camera, wait for traced_publishes to climb, THEN read. Scene fitting, density selection and relocating probes out of walls are not done here; this method is the lever those would eventually drive.",
    "render", "Render", false, "any",
    "rayfusion|set|probe|grid",
    "rayfusion.probe_field|rayfusion.set_probe_follow_camera|rayfusion.set_probe_producer",
    nullptr, nullptr, "rayfusion.probe_field", nullptr,
    params_rayfusion_set_probe_grid, 5,
    true
};
static const MethodRegistration reg_rayfusion_set_probe_grid(desc_rayfusion_set_probe_grid);

static const MethodParam params_rayfusion_set_probe_overlay[] = {
    {"enabled", "boolean", true, "Enable depth-tested raster probe markers", nullptr, nullptr},
};
static const MethodDescriptor desc_rayfusion_set_probe_overlay = {
    "rayfusion.set_probe_overlay", "rayfusion",
    "Show depth-tested probe markers inside the raster viewport",
    "Strict boolean enabled; session-local, default false. UI/Python/IPC use one service. Applied means accepted, not GPU success. probe_field reports overlay_requested, overlay_ready, overlay_markers and overlay_reason. Markers use scene depth and stay below UI panels; unavailable in Rendered mode. No per-frame marker work while disabled. Missing shader reports reason.",
    "render", "Render", false, "any",
    "rayfusion|set|probe|overlay",
    "rayfusion.probe_field",
    nullptr, nullptr, nullptr, nullptr,
    params_rayfusion_set_probe_overlay, 1,
    true
};
static const MethodRegistration reg_rayfusion_set_probe_overlay(desc_rayfusion_set_probe_overlay);

static const MethodParam params_rayfusion_set_probe_producer[] = {
    {"traced", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rayfusion_set_probe_producer = {
    "rayfusion.set_probe_producer", "rayfusion",
    "Choose which producer fills the RayFusion probe field: traced rays (step 1b) or the sky bake (step 1a).",
    "This is an A/B MEASUREMENT lever on the live image, which is why it is Render and not Read. Step 1b swapped the probe PRODUCER while leaving the consumer (the ambient lookup in the fragment shader) untouched, so any visible difference is attributable to the producer alone - but that argument only holds if the old producer can still be selected on the same scene. Pass traced=false to get the step-1a sky bake back, traced=true for ray-traced visibility. The reply reports 'producer', which is what the field ACTUALLY runs afterwards, and it can differ from the request: with no hardware ray tracing, no built scene AS, or a missing rayfusion_probe_trace.spv, the request is refused and 'producer_reason' says why. Switching invalidates every probe on purpose - a half-swapped field would blend two producers and the comparison would measure the mixture. Read the result back with rayfusion.probe_field.",
    "render", "Render", false, "any",
    "rayfusion|set|probe|producer",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_rayfusion_set_probe_producer, 1,
    true
};
static const MethodRegistration reg_rayfusion_set_probe_producer(desc_rayfusion_set_probe_producer);

static const MethodParam params_rayfusion_set_reflections[] = {
    {"enabled", "any", false, "", nullptr, nullptr},
    {"follow_quality_preset", "any", false, "", nullptr, nullptr},
    {"roughness_gate", "any", false, "", nullptr, nullptr},
    {"samples", "any", false, "", nullptr, nullptr},
    {"weight_gate", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rayfusion_set_reflections = {
    "rayfusion.set_reflections", "rayfusion",
    "Replace per-pixel specular reflection settings",
    "Requires material viewport, scene lighting and a prefiltered environment -- the same gate the fragment shader uses to fill the weight, because the subtracted term is only valid on that path. Independent of RT shadows: the G-buffer comes from the main pass. Invalid settings reject atomically; unknown keys reject. samples is per PIXEL (1/2/4) and cost is linear in it. roughness_gate drops glossier pixels; weight_gate drops pixels whose split-sum weight is too small to see. Produce a viewport frame before reading rayfusion.reflections -- the counters describe the last dispatch, not this request.",
    "render", "Render", false, "object",
    "rayfusion|set|reflections|raster|lighting|specular|reflection",
    "rayfusion.reflections|viewport.frame_timings",
    nullptr, nullptr, "rayfusion.reflections", nullptr,
    params_rayfusion_set_reflections, 5,
    true
};
static const MethodRegistration reg_rayfusion_set_reflections(desc_rayfusion_set_reflections);

static const MethodParam params_rayfusion_set_screen_gi[] = {
    {"enabled", "any", false, "Enable same-frame diffuse RTGI", nullptr, nullptr},
    {"samples", "any", false, "", nullptr, "['1', '2', '4']"},
    {"filter_radius", "any", false, "", nullptr, "['0', '1', '2']"},
    {"max_distance", "any", false, "Finite ray range in world units [0.1,10000]", nullptr, nullptr},
};
static const MethodDescriptor desc_rayfusion_set_screen_gi = {
    "rayfusion.set_screen_gi", "rayfusion",
    "Replace same-frame diffuse RTGI settings",
    "Runtime experiment. Requires material viewport, scene lighting and RT shadows. No GI-owned history buffer, but sample seeds rotate with the TAA accumulation index. Invalid settings reject atomically; unknown keys reject. Unsupported hit BSDFs use the established ambient fallback.",
    "render", "Render", false, "object",
    "rayfusion|set|screen|gi",
    "rayfusion.screen_gi|viewport.frame_timings",
    nullptr, nullptr, "rayfusion.screen_gi", "raster_frame",
    params_rayfusion_set_screen_gi, 4,
    true
};
static const MethodRegistration reg_rayfusion_set_screen_gi(desc_rayfusion_set_screen_gi);

static const MethodDescriptor desc_rayfusion_validate_core = {
    "rayfusion.validate_core", "rayfusion",
    "Run isolated native checks of the RayFusion probe cache and update scheduler",
    "Uses the real C++ core with local fixtures; checks world-cell reuse, ray budgets, stale scene/device/light/geometry completions, cross-field isolation, invalid packets and distance-moment visibility. Does not touch the scene or GPU. passed=true is not a rendered GI validation; gpu_tested remains false. Takes no parameters.",
    "read", "Read", false, "RayFusionCoreValidation",
    "rayfusion|validate|core|gi|probe|validation|qa",
    "rayfusion.core_status",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_rayfusion_validate_core(desc_rayfusion_validate_core);

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

static const MethodDescriptor desc_render_optix_accum_status = {
    "render.optix_accum_status", "render",
    "Counts how often the OptiX accumulation buffer was WIPED, independently of the sample counter.",
    "Answers 'the sample counter climbs but the image never accumulates'. The two numbers are deliberately independent because they can disagree, and that disagreement is the finding. OptixWrapper reallocates and zeroes the float4 accumulation buffer whenever accumulation_valid is false; the resolution-change branch sets exactly that, and it does NOT reset accumulated_samples. So if the render resolution alternates between two values across frames, the buffer is zeroed every frame while the counter keeps counting: progress is reported that the image does not have, and each pass is displayed alone because the kernel takes its prev_samples==0 branch. Read wipe_count against accumulated_samples: a healthy progressive render wipes ONCE per camera move, so wipe_count should stay near 1 while samples climb. If wipe_count tracks the frame count, the buffer is being destroyed every pass. There are THREE wipe routes and they are counted separately because they are different mechanisms and only one of them is visible in wipe_count: the resolution branch and the lazy realloc both free and re-zero the buffer (wipe_count, wipe_resolution), while the camera-change branch uses cudaMemsetAsync instead (wipe_camera). Counting only the realloc would have shown wipe_count=0 while the camera path re-zeroed every single frame -- an instrument going quiet read as proof of absence. Note also that the camera branch DOES reset accumulated_samples while the resolution branch does not, so a climbing sample counter alongside a non-accumulating image points away from camera_changed and toward the resolution branch. Then read wipe_resolution and the last_wipe_from / last_wipe_to pairs: if those two sizes keep swapping, the caller is handing OptiX a different resolution on alternating frames and that is the root, not the accumulation code. available=false simply means the OptiX backend is not active and is not an error.",
    "render", "Render", false, "any",
    "render|optix|accum|status",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_render_optix_accum_status(desc_render_optix_accum_status);

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

static const MethodParam params_rig_add_bone[] = {
    {"character", "any", false, "Owned rig identity.", nullptr, nullptr},
    {"name", "any", false, "New authored joint name; full key becomes character_name.", nullptr, nullptr},
    {"parent", "any", false, "Existing full unique parent joint key.", nullptr, nullptr},
    {"rest_transform", "any", false, "Row-major 16-number local rest transform. Translation and proper rotation only; scale/shear/reflection rejected.", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_add_bone = {
    "rig.add_bone", "rig",
    "Append a child joint to an owned unskinned rig without bound clips",
    "Parent is an existing full unique joint key. Preserves existing global bone indices; refreshes all derived rig representations, selects the new joint and increments persisted rig revision. Imported rigs, mesh-bound/weighted rigs and rigs with clips are rejected in this first pass. Service failures return named codes (Python ValueError / IPC code): rig_name_conflict, invalid_rig_name/invalid_bone_name, unknown_rig_template, unknown_character, unknown_parent_bone/unknown_bone, bone_name_conflict, rig_not_owned, rig_edit_requires_unskinned, rig_edit_requires_no_clips, invalid_rest_transform, rig_rest_requires_rigid_transform, scene_locked or history_not_bound. Shape/type errors use Python TypeError/ValueError and IPC invalid_parameter. Validation/staging happens before scene changes. Empty name requests automatic Joint1/Joint2 naming inside the staged core operation. Success selects the new child as the next parent; each addition is one undo command. Explicit duplicate names still return bone_name_conflict. Unskinned staging also scans actual canonical flat scene weight buffers for references to this rig hierarchy indices; references block with rig_edit_requires_unskinned even if weighted metadata is missing, including zero/invalid entries and extra rows. No weighted delete/remap/transfer is implemented.",
    "SceneWrite", "SceneWrite", true, "{ok:true}",
    "rig|add|bone",
    "rig.create|rig.set_rest_transform|rig.list_bones",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_add_bone, 4,
    true
};
static const MethodRegistration reg_rig_add_bone(desc_rig_add_bone);

static const MethodParam params_rig_apply_bone_envelope[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"bone", "string", true, "", nullptr, nullptr},
    {"rig_revision", "integer", true, "", nullptr, nullptr},
    {"start_radius", "float", true, "", nullptr, nullptr},
    {"end_radius", "float", true, "", nullptr, nullptr},
    {"start_extension", "float", true, "", nullptr, nullptr},
    {"end_extension", "float", true, "", nullptr, nullptr},
    {"falloff", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_apply_bone_envelope = {
    "rig.apply_bone_envelope", "rig",
    "Store one tapered bone envelope and rebuild all authored skin weights",
    "Atomically stores the per-bone start/end radius, axial extensions and falloff, then recomputes every registered flat bind part against all competing envelopes. Strongest-four normalization, backend invalidation, native persistence and one heavy undo step match apply_envelope_weights. Radii .005..0.5 character height; extensions 0..0.75 bone length; falloff .5..8.",
    "write", "SceneWrite", true, "ok",
    "rig|apply|bone|envelope|weights|profile",
    "rig.get_bone_envelope|rig.preview_envelope_weights|rig.get_weight_map",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_apply_bone_envelope, 8,
    true
};
static const MethodRegistration reg_rig_apply_bone_envelope(desc_rig_apply_bone_envelope);

static const MethodParam params_rig_apply_envelope_weights[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"rig_revision", "integer", true, "", nullptr, nullptr},
    {"torso_radius", "float", false, "", "0.16", nullptr},
    {"limb_radius", "float", false, "", "0.065", nullptr},
    {"extremity_radius", "float", false, "", "0.05", nullptr},
    {"falloff", "float", false, "", "2.0", nullptr},
};
static const MethodDescriptor desc_rig_apply_envelope_weights = {
    "rig.apply_envelope_weights", "rig",
    "Replace authored bound-mesh weights with bounded anatomical capsule weights",
    "Recomputes from canonical flat P_orig using the same deterministic core as preview, then swaps all authored bound-part skin weights atomically in one heavy undo step. CPU, raster, Vulkan and OptiX animation geometry are invalidated; native project geometry persists the resulting weights. Requires the current nonnegative rig_revision from preview/get_binding. Imported skins and missing authored parts reject. Radius/falloff ranges and fallback semantics match preview.",
    "write", "SceneWrite", true, "ok",
    "rig|apply|envelope|weights|capsule|skin",
    "rig.preview_envelope_weights|rig.get_weight_map|rig.weight_stats|rig.get_binding",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_apply_envelope_weights, 6,
    true
};
static const MethodRegistration reg_rig_apply_envelope_weights(desc_rig_apply_envelope_weights);

static const MethodParam params_rig_apply_pose_preview[] = {
    {"character", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_apply_pose_preview = {
    "rig.apply_pose_preview", "rig",
    "Commit a pose preview and Auto Key IK controls or undriven FK bones",
    "7A/8A: requires an owned rig with a bound flat mesh; imported rigs remain read-only for pose authoring. Mutations require active pose mode and no render job. Rest transforms and weights are never modified. Frame time uses the scene FPS; create_pose_clip.fps controls clip ticks per second. Matrices are row-major 16 numbers, rigid translation/rotation only. Preview replaces the previous preview relative to the committed pose; frame changes and mode exit discard unkeyed overrides. Keys persist in native projects; clip selection and Auto Key are transient. Unweighted vertices retain their bind position.",
    "write", "SceneWrite", true, "ok",
    "rig|apply|pose|preview|FK|bone keys|authoring",
    "rig.set_mode|rig.get_pose_state|rig.preview_pose_locals|rig.apply_pose_preview",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_apply_pose_preview, 1,
    true
};
static const MethodRegistration reg_rig_apply_pose_preview(desc_rig_apply_pose_preview);

static const MethodParam params_rig_bake_ik_channels[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"rig_revision", "integer", true, "", nullptr, nullptr},
    {"start_frame", "integer", true, "", nullptr, nullptr},
    {"end_frame", "integer", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_bake_ik_channels = {
    "rig.bake_ik_channels", "rig",
    "Bake saved IK channels into a new bone clip over an inclusive frame range",
    "Same mutation guards as insert_ik_key. Samples original bone clip -> saved IK -> joint limits -> all bone keys at every frame, using current fixed actor placement. Excludes transient live/FK edits. Creates/selects a new named clip with no IK channels; original unchanged. Existing bone channels outside range are copied. At most 1001 frames and 100000 bone samples; frames nonnegative <=1000000, ticks <=1000000. One undo step/native persistence. Errors: rig_ik_bake_limit, rig_ik_channels_empty, rig_pose_clip_name_conflict, invalid_clip_name.",
    "write", "SceneWrite", false, "ok",
    "rig|bake|ik|channels|pose|IK|timeline|contact",
    "rig.get_ik_channels|rig.insert_ik_key|rig.set_ik_contact_interval|rig.bake_ik_channels",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_bake_ik_channels, 4,
    true
};
static const MethodRegistration reg_rig_bake_ik_channels(desc_rig_bake_ik_channels);

static const MethodParam params_rig_bind_mesh[] = {
    {"preview", "any", false, "Unmodified preview payload; generated weights are recomputed", nullptr, nullptr},
    {"character", "any", false, "", nullptr, nullptr},
    {"mesh", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_bind_mesh = {
    "rig.bind_mesh", "rig",
    "Bind actual flat mesh parts and publish initial weights as one undoable transaction",
    "Initial bind only: authoring-owned meshless, clip-free rig and static unskinned actual flat mesh parts. Exact mesh name takes precedence over model:<importName> group. Preserves part names/topology/materials/UV; converts flat local P_orig/N_orig (fallback P/N) through inverse(actor placement)*source base into common rig bind space. All parts share actor Transform; inverse joint-global rest offsets and model inverse identity. Root anatomy role is nondeforming; segment weights belong to parent joint, nearest outgoing segment per joint, inverse squared distance regularized by extent*1e-4 (minimum 1e-9), strongest four merged/normalized via common contract, negligible entries pruned. No visibility/interior or alignment certification. Rejects animation, skin, modifiers/geometry graphs/deltas, incomplete or invalid normals/geometry, reflected/singular transforms. Limits: 256 parts, 2M vertices, 100M vertex-segment evaluations, 4096 joints. Existing canEditRig guards apply. Requires preview object from rig.preview_bind. Rebuilds weights and validates rig_revision/rig_token/mesh_token; never trusts submitted weights/can_bind. Stale rig/mesh tokens fail rig_bind_stale_preview before canonical/history mutation. Commit swaps cloned DNA geometry, transforms, owned context, global bone snapshot and affected source memberships. Refreshes CPU BVH and OptiX/Vulkan/raster influence geometry, invalidates animation groups; undo/redo restore state and schedule the same refresh. Explicit part registry persists to native project; missing/deleted part identity retained, not manufactured. No weighted rebind, second bind append, weight painting, weighted rest/topology/clip migration or Pose/IK authoring. Errors additionally rig_bind_invalid_preview, rig_already_bound, api_not_bound, history_not_bound, scene_locked, rig_bind_failed. Python rt.rig.bind_mesh(character,mesh,preview).",
    "SceneWrite", "SceneWrite", true, "{ok: bool}",
    "rig|bind|mesh",
    "rig.preview_bind|rig.get_binding|rig.weight_stats",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_bind_mesh, 3,
    true
};
static const MethodRegistration reg_rig_bind_mesh(desc_rig_bind_mesh);

static const MethodParam params_rig_cancel_pose_preview[] = {
    {"character", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_cancel_pose_preview = {
    "rig.cancel_pose_preview", "rig",
    "Discard a pose preview and restore the committed pose",
    "7A/8A: requires an owned rig with a bound flat mesh; imported rigs remain read-only for pose authoring. Mutations require active pose mode and no render job. Rest transforms and weights are never modified. Frame time uses the scene FPS; create_pose_clip.fps controls clip ticks per second. Matrices are row-major 16 numbers, rigid translation/rotation only. Preview replaces the previous preview relative to the committed pose; frame changes and mode exit discard unkeyed overrides. Keys persist in native projects; clip selection and Auto Key are transient. Unweighted vertices retain their bind position.",
    "write", "SceneWrite", false, "ok",
    "rig|cancel|pose|preview|FK|bone keys|authoring",
    "rig.set_mode|rig.get_pose_state|rig.preview_pose_locals|rig.apply_pose_preview",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_cancel_pose_preview, 1,
    true
};
static const MethodRegistration reg_rig_cancel_pose_preview(desc_rig_cancel_pose_preview);

static const MethodParam params_rig_clear_ik_channels[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"rig_revision", "integer", true, "", nullptr, nullptr},
    {"control", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_clear_ik_channels = {
    "rig.clear_ik_channels", "rig",
    "Remove all saved IK keys and contact intervals for one limb",
    "Same mutation guards as insert_ik_key. Clears only selected clip/control and its live override, preserving bone channels and clip duration. One undo step. Missing channel fails rig_edit_no_change.",
    "write", "SceneWrite", false, "ok",
    "rig|clear|ik|channels|pose|IK|timeline|contact|bake",
    "rig.get_ik_channels|rig.insert_ik_key|rig.set_ik_contact_interval|rig.bake_ik_channels",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_clear_ik_channels, 3,
    true
};
static const MethodRegistration reg_rig_clear_ik_channels(desc_rig_clear_ik_channels);

static const MethodDescriptor desc_rig_clear_selection = {
    "rig.clear_selection", "rig",
    "Clear the shared skeleton selection",
    "Works with skeleton-only and meshless animated imports. Selection/overlay are transient view state, not authoring edits; no undo. Names are import-prefixed. Joint world transforms are row-major 16 floats from evaluated hierarchy globals, never skin-offset matrices. pose_source is bind, graph, controller or ozz. Bind fallback is explicit per node. Independent skeleton root-motion placement and bind/pose editing are subsequent roadmap work. Clears complete multi-selection, active and anchor; pivot preference is retained. Successful selection releases the selected IK handle and discards its uncommitted preview; committed IK controls, timed channels and contacts remain active. Direct FK edits of IK-driven bones still require returning their control to FK.",
    "write", "SceneWrite", false, "ok",
    "rig|clear|selection|skeleton|joint|meshless|overlay",
    "rig.list_characters|rig.list_bones|rig.select_bone|rig.get_selected_bone",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_rig_clear_selection(desc_rig_clear_selection);

static const MethodParam params_rig_commit_fit[] = {
    {"character", "any", false, "Owned unskinned clip-free rig name", nullptr, nullptr},
    {"mesh", "any", false, "Exact flat mesh node name or model:<importName> group target", nullptr, nullptr},
    {"preview", "any", false, "Preview payload from rig.preview_fit; landmarks are revalidated", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_commit_fit = {
    "rig.commit_fit", "rig",
    "Apply reviewed manual rest landmarks as one undoable rig edit",
    "Requires same character/mesh, unchanged rig revision and exact current flat mesh content token. Recomputes validation; client can_commit cannot bypass it. Rejects outside-bounds/zero-length landmarks. Preserves model-space global joint rotations, converts world positions through inverse actor placement and derives canonical local rest transforms; rebuilds BoneData/skeleton/runtime through existing common stage/undo service. Native snapshot persists resulting hierarchy and anatomy, not transient drafts. No mesh mutation, binding, weights, automatic fit or verified surface interior. Errors: rig_fit_invalid_preview, rig_fit_stale_preview, rig_fit_landmarks_outside_bounds and preview/shared staged rig errors. Imported weighted rigs and clip-bound rigs remain gated. Targets also accept model:<importName> multipart groups from rig.list_fit_targets. Exact existing mesh names take precedence. Group processing preserves independent scene geometry/transforms and hashes combined viewport world positions/indices for stale preview detection. Group errors: unknown_mesh_group, ambiguous_mesh_group, mesh_group_has_no_flat_parts, mesh_group_invalid_geometry, mesh_group_too_large.",
    "SceneWrite", "SceneWrite", true, "{ok:true}",
    "rig|commit|fit",
    "rig.preview_fit",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_commit_fit, 3,
    true
};
static const MethodRegistration reg_rig_commit_fit(desc_rig_commit_fit);

static const MethodParam params_rig_copy_from[] = {
    {"source_character", "any", false, "Existing source import/rig name; meshless and unskinned, may contain animation clips", nullptr, nullptr},
    {"character", "any", false, "New distinct owned rig name, ASCII letters/digits/underscore/hyphen, 1-128 characters", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_copy_from = {
    "rig.copy_from", "rig",
    "Create a separate editable rest rig from a meshless unskinned source skeleton",
    "Source canonical hierarchy is authoritative. Copies skeleton joints and required ancestors in parent-first order; source clips, runtime and geometry are untouched. New rig has no clips, mesh, weights or loader. Positive uniform scales are baked into joint positions and proper world rotations are preserved. Nonuniform scale, shear and reflection return rig_copy_unsupported_rest. Invalid/duplicate authored names are sanitized deterministically and reported in bone_map; forests get a synthetic SceneRoot. One undo command; existing native hierarchy serialization persists the owned copy. Exit Rig Edit before copying. Errors: rig_edit_active, invalid_rig_name, rig_name_conflict, unknown_source_character, rig_copy_requires_meshless_unskinned, rig_copy_requires_skeleton_hierarchy, rig_copy_incomplete_hierarchy, invalid_preview_hierarchy, invalid_preview_pose, rig_copy_unsupported_rest, scene_locked. Anatomy references are preserved/remapped. Reparent may return rig_anatomy_chain_disconnected; delete may return rig_bone_anatomy_referenced until metadata references are removed.",
    "SceneWrite", "SceneWrite", true, "{character: string, bone_map: [{source_bone: string, target_bone: string}]}",
    "rig|copy|from",
    "rig.list_characters|rig.list_bones|rig.set_mode|rig.rename_bone|rig.reparent_bone|rig.delete_bone|anim.bind_clip",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_copy_from, 2,
    true
};
static const MethodRegistration reg_rig_copy_from(desc_rig_copy_from);

static const MethodParam params_rig_create[] = {
    {"character", "any", false, "New character identity; must be unique and not overlap another character prefix.", nullptr, nullptr},
    {"template_id", "string", false, "", "root", "['root', 'chain3', 'humanoid', 'quadruped', 'insect6', 'avian']"},
    {"height", "float", false, "Positive finite rest layout height <=10000; default 1.8. Catalogue gives family-specific UI suggestions.", "1.8", nullptr},
};
static const MethodDescriptor desc_rig_create = {
    "rig.create", "rig",
    "Create an owned meshless root or three-joint-chain rig with undo and project persistence",
    "Creates NodeHierarchy canonical rest state, BoneData, skeletonNodes and ozz bridge together. Stable joint keys use character_name. Character/name use 1-128 ASCII letters, digits, underscore or hyphen. Rejects existing/overlapping character prefixes and flat-mesh node prefixes. No skin weights or mesh binding. Service failures return named codes (Python ValueError / IPC code): rig_name_conflict, invalid_rig_name/invalid_bone_name, unknown_rig_template, unknown_character, unknown_parent_bone/unknown_bone, bone_name_conflict, rig_not_owned, rig_edit_requires_unskinned, rig_edit_requires_no_clips, invalid_rest_transform, rig_rest_requires_rigid_transform, scene_locked or history_not_bound. Shape/type errors use Python TypeError/ValueError and IPC invalid_parameter. Validation/staging happens before scene changes. Template IDs now include humanoid (27 joints), quadruped (26), insect6 (36), avian (32), alongside root and chain3. The shared builder creates canonical rest hierarchy plus anatomy roles, symmetry pairs and contiguous limb chains before existing four-representation finish/undo/persistence. No mesh fit, weights, IK or detailed fingers/feathers. Avian includes paired five-joint wing chains, two leg chains, tail and beak; it is a bird fitting seed, not a bat/insect flight system. Height is overall layout scale in scene units, insect including antennae. Root/chain3 layouts and empty anatomy remain compatible.",
    "SceneWrite", "SceneWrite", true, "{ok:true}",
    "rig|create",
    "rig.add_bone|rig.set_rest_transform|rig.list_bones|project.save|project.open",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_create, 3,
    true
};
static const MethodRegistration reg_rig_create(desc_rig_create);

static const MethodParam params_rig_create_aim_control[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"rig_revision", "integer", true, "", nullptr, nullptr},
    {"role", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_create_aim_control = {
    "rig.create_aim_control", "rig",
    "Add a one-bone aim control from a canonical anatomy role",
    "Owned bound rig, matching revision, no render job/preview. role must name an existing canonical anatomy role. The service appends role.aim and derives its local aim axis from the first nonzero direct child, falling back to local +Y; a perpendicular roll-up axis is stored explicitly. The world target controls direction and pole_world controls roll. Local translation is preserved, quaternion blend occurs once, then shared joint limits may leave angular residual. Aim controls support IK/FK blend, target keys and bake; position contacts, spline shape and tip-orientation channels are rejected. Persistent anatomy version 5 stores solver, aim_axis and up_axis. Duplicate/overlapping driven bones fail. Errors include rig_ik_unknown_role, rig_ik_invalid_aim_axes, rig_ik_aim_target_at_origin, rig_ik_overlapping_controls and rig_edit_stale_revision.",
    "write", "SceneWrite", false, "ok",
    "rig|create|aim|control|pose|IK|look|head|wing|target|roll",
    "rig.get_anatomy|rig.create_controls|rig.get_controls|rig.set_ik_target|rig.insert_ik_key|rig.bake_ik_channels",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_create_aim_control, 3,
    true
};
static const MethodRegistration reg_rig_create_aim_control(desc_rig_create_aim_control);

static const MethodParam params_rig_create_chain_control[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"rig_revision", "integer", true, "", nullptr, nullptr},
    {"chain", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_create_chain_control = {
    "rig.create_chain_control", "rig",
    "Add a multi-joint FABRIK control from a canonical anatomy chain",
    "Owned bound rig, matching revision, no render job/preview. chain names an existing anatomy chain of 4..64 distinct consecutive rigid bones with nonzero segments. Appends a control with that name, preserving limb controls. Duplicate names/overlapping driven bones fail. Shared create_controls service publishes one undo step and persistent anatomy version 4; re-query revision. Target solving fixes the root, preserves local translations/lengths, blends once, then shared joint limits may leave residual. Pole seeds bending when solving a changed endpoint; this is not a spline curve editor. Same target/orientation/FK/key/contact/bake services as limb IK. Errors: rig_ik_unknown_chain, rig_ik_invalid_chain, rig_ik_overlapping_controls, rig_ik_duplicate_control, rig_ik_zero_length, rig_edit_stale_revision.",
    "write", "SceneWrite", false, "ok",
    "rig|create|chain|control|pose|IK|spine|neck|tail|fabrik",
    "rig.get_anatomy|rig.create_controls|rig.get_controls|rig.set_ik_target|rig.bake_ik_channels",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_create_chain_control, 3,
    true
};
static const MethodRegistration reg_rig_create_chain_control(desc_rig_create_chain_control);

static const MethodParam params_rig_create_controls[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"rig_revision", "integer", true, "", nullptr, nullptr},
    {"controls", "array|null", false, "null/omitted = derive anatomy limbs; array = replace; [] = clear.", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_create_controls = {
    "rig.create_controls", "rig",
    "Create or replace canonical limb, chain and aim definitions",
    "Owned bound rig, no render job or pending preview. Null/omitted controls derive anatomy _arm/_leg chains; humanoid arms skip clavicle, legs start at index zero. Supply [{name,root,mid,tip}] for explicit custom limbs; add chain:[root,...,tip] for FABRIK (4..64 bones, mid = second bone). Aim rows use solver:aim, the same bone in root/mid/tip, and finite orthonormal aim_axis/up_axis vectors. [] clears definitions unless saved channels reference them. At most 256 controls/4096 joints, rigid bones, no duplicate names or shared driven joints. Errors include rig_ik_invalid_controls, rig_ik_invalid_solver, rig_ik_invalid_aim_axes, rig_ik_invalid_chain, rig_ik_zero_length, rig_ik_overlapping_controls, rig_ik_duplicate_control, rig_ik_invalid_name, unknown_bone, rig_ik_no_limb_chains, rig_edit_stale_revision, rig_revision_overflow and rig_edit_no_change. One undo step persists definitions/revision and resets handles; undo restores prior runtime state. Rest/bind/weights remain unchanged. Re-query revision after success.",
    "write", "SceneWrite", false, "ok",
    "rig|create|controls|pose|IK|setup",
    "rig.get_controls|rig.create_chain_control|rig.create_aim_control|rig.set_mode|rig.set_ik_target|rig.set_ik_fk",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_create_controls, 3,
    true
};
static const MethodRegistration reg_rig_create_controls(desc_rig_create_controls);

static const MethodParam params_rig_create_human_walk_clip[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"rig_revision", "integer", true, "", nullptr, nullptr},
    {"name", "string", true, "", nullptr, nullptr},
    {"fps", "float", true, "", nullptr, nullptr},
    {"cadence", "float", true, "", nullptr, nullptr},
    {"cycles", "integer", true, "", nullptr, nullptr},
    {"stride", "float", true, "", nullptr, nullptr},
    {"step_height", "float", true, "", nullptr, nullptr},
    {"body_bounce", "float", true, "", nullptr, nullptr},
    {"arm_swing", "float", true, "", nullptr, nullptr},
    {"body_motion", "float", false, "", "0.75", nullptr},
};
static const MethodDescriptor desc_rig_create_human_walk_clip = {
    "rig.create_human_walk_clip", "rig",
    "Generate a loopable editable humanoid walk clip",
    "Owned bound humanoid in Pose mode, matching revision, no render job or pending preview. Generates every frame through canonical anatomy-derived two-bone limb IK and shared joint limits, then stores ordinary rigid local bone keys. Feet retain planted orientation with swing toe pitch; the pelvis transfers weight laterally, tilts and yaws while mapped spine bones counter-rotate. Optional mapped head and clavicle roles stabilize gaze and involve the shoulder girdle. body_motion 0..1 controls pelvis/torso naturalization and defaults to .75 when omitted. Reusing a name atomically replaces only an editable rig-authored clip for the same character in one undo step; imported clips still return rig_pose_clip_name_conflict. The clip is in-place and actor-placement independent; no persistent IK targets or controls are added. Height fractions: stride .05..0.80, step_height 0..0.25, body_bounce 0..0.15; arm_swing/body_motion 0..1, cadence 20..300 steps/min, cycles 1..8, fps 1..120. Maximum 1000 frames and 100000 frame-joint samples. Native project persistence stores it. Errors include rig_walk_requires_humanoid, rig_walk_requires_humanoid_roles, rig_walk_requires_humanoid_limbs, rig_walk_invalid_recipe, rig_walk_sample_limit, rig_walk_invalid_height, rig_pose_clip_name_conflict, rig_pose_preview_active and rig_edit_stale_revision.",
    "write", "SceneWrite", false, "ok; selects the new authored bone clip",
    "rig|create|human|walk|clip|animation|gait|recipe|generate|humanoid",
    "rig.preview_human_walk|rig.get_pose_state|rig.insert_pose_keys|rig.create_controls",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_create_human_walk_clip, 11,
    true
};
static const MethodRegistration reg_rig_create_human_walk_clip(desc_rig_create_human_walk_clip);

static const MethodParam params_rig_create_mirrored_bone[] = {
    {"character", "any", false, "Owned meshless, unweighted, clip-free rig name", nullptr, nullptr},
    {"rig_revision", "any", false, "Expected revision from rig.list_bones; stale operations reject before publication", nullptr, nullptr},
    {"axis", "string", false, "Mirror plane normal in canonical rig space: x, y or z", "x", nullptr},
    {"offset", "float", false, "Finite plane coordinate in rig units; actor placement does not change this coordinate", "0.", nullptr},
    {"bone", "any", false, "One existing unpaired non-root source key", nullptr, nullptr},
    {"name", "any", false, "New unprefixed valid bone name; canonical key is character_name", nullptr, nullptr},
    {"source_side", "any", false, "left or right; defines pair orientation explicitly", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_create_mirrored_bone = {
    "rig.create_mirrored_bone", "rig",
    "Create one opposite non-root bone and its explicit symmetry pair",
    "One bone, not a subtree clone. Reflect source global rest and preserve handedness. Parent is its paired opposite parent when available, otherwise the same shared parent. Add symmetry metadata; anatomy roles/chains remain explicit, not guessed or copied. Existing indices preserved, one new slot, four representations/runtime/revision rebuilt and current selection becomes new key; one production undo/redo and native hierarchy/anatomy persistence. Shared owned/unweighted/meshless/clip-free gates apply. Errors include rig_mirror_invalid_side, rig_mirror_already_paired, rig_root_edit_blocked, invalid_bone_name, bone_name_conflict, rig_mirror_invalid_plane, rig_mirror_limit, rig_mirror_invalid_transform, unknown_bone, rig_edit_stale_revision and common edit/anatomy/scene lock errors.",
    "SceneWrite", "SceneWrite", false, "{ok:true}",
    "rig|create|mirrored|bone",
    "rig.mirror_rest|rig.get_anatomy|rig.get_next_bone_name",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_create_mirrored_bone, 7,
    true
};
static const MethodRegistration reg_rig_create_mirrored_bone(desc_rig_create_mirrored_bone);

static const MethodParam params_rig_create_pose_clip[] = {
    {"fps", "float", false, "Clip ticks per second, 1..240; does not change the scene FPS.", "24.", nullptr},
    {"character", "any", false, "", nullptr, nullptr},
    {"name", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_create_pose_clip = {
    "rig.create_pose_clip", "rig",
    "Create and select an editable native bone-channel clip seeded with rest keys",
    "7A/8A: requires an owned rig with a bound flat mesh; imported rigs remain read-only for pose authoring. Mutations require active pose mode and no render job. Rest transforms and weights are never modified. Frame time uses the scene FPS; create_pose_clip.fps controls clip ticks per second. Matrices are row-major 16 numbers, rigid translation/rotation only. Preview replaces the previous preview relative to the committed pose; frame changes and mode exit discard unkeyed overrides. Keys persist in native projects; clip selection and Auto Key are transient. Unweighted vertices retain their bind position.",
    "write", "SceneWrite", true, "ok",
    "rig|create|pose|clip|FK|bone keys|authoring",
    "rig.set_mode|rig.get_pose_state|rig.preview_pose_locals|rig.apply_pose_preview",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_create_pose_clip, 3,
    true
};
static const MethodRegistration reg_rig_create_pose_clip(desc_rig_create_pose_clip);

static const MethodParam params_rig_delete_bone[] = {
    {"character", "any", false, "Owned rig name", nullptr, nullptr},
    {"bone", "any", false, "Existing prefixed bone key", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_delete_bone = {
    "rig.delete_bone", "rig",
    "Delete one unweighted leaf bone and select its parent",
    "Owned, unskinned, clip-free rigs only. Uses one shared staged core and undo command; other rig selection/edit is blocked in scoped Rig Edit. Shared errors: unknown_character, unknown_bone, rig_not_owned, rig_edit_requires_unskinned, rig_edit_requires_no_clips, rig_edit_character_locked, rig_revision_overflow, scene_locked. Cross-rig references return rig_bone_external_reference. Root or non-leaf deletion is rejected (rig_root_edit_blocked, rig_delete_requires_leaf). Erases canonical node and BoneData entries; surviving scene bone indices stay stable. Native serialization and undo/redo preserve topology. No recursive deletion or weight transfer. Anatomy references are preserved/remapped. Reparent may return rig_anatomy_chain_disconnected; delete may return rig_bone_anatomy_referenced until metadata references are removed. Unskinned staging also scans actual canonical flat scene weight buffers for references to this rig hierarchy indices; references block with rig_edit_requires_unskinned even if weighted metadata is missing, including zero/invalid entries and extra rows. No weighted delete/remap/transfer is implemented.",
    "SceneWrite", "SceneWrite", true, "{ok: true}",
    "rig|delete|bone",
    "rig.list_bones|rig.set_mode|rig.rename_bone|rig.reparent_bone|rig.delete_bone",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_delete_bone, 2,
    true
};
static const MethodRegistration reg_rig_delete_bone(desc_rig_delete_bone);

static const MethodParam params_rig_get_anatomy[] = {
    {"character", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_get_anatomy = {
    "rig.get_anatomy", "rig",
    "Read canonical rig anatomy roles, symmetry pairs and limb chains",
    "Returns version 1 anatomy schema. Old projects default to custom family and empty arrays. Read works for any known model; setting requires an owned unskinned clip-free rig. Family is a semantic tag, not auto-rig generation.",
    "Read", "Read", false, "{version: 1, family: custom|humanoid|quadruped|insect|avian, roles: [{role,bone}], symmetry: [{left,right}], chains: [{name,bones}]}",
    "rig|get|anatomy",
    "rig.set_anatomy|rig.list_bones",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_get_anatomy, 1,
    true
};
static const MethodRegistration reg_rig_get_anatomy(desc_rig_get_anatomy);

static const MethodParam params_rig_get_binding[] = {
    {"character", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_get_binding = {
    "rig.get_binding", "rig",
    "Report canonical bound part identities and current presence for a rig",
    "Read-only authoring binding registry, not imported skin inventory. Returns character, bound, algorithm (nearest_segment, anatomical_capsule_v1 or null), parts[{mesh,present}], rig_revision. anatomical_capsule_v1 also returns persisted envelope_settings. Uses exact persisted names; ownership takes precedence over original source hierarchy. Deleted/missing parts remain recorded with present=false. Errors unknown_character, api_not_bound, scene_locked, rig_bind_failed. Python rt.rig.get_binding(character).",
    "Read", "Read", false, "RigMeshBindingInfo",
    "rig|get|binding",
    "rig.preview_bind|rig.bind_mesh|rig.weight_stats",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_get_binding, 1,
    true
};
static const MethodRegistration reg_rig_get_binding(desc_rig_get_binding);

static const MethodParam params_rig_get_bone_envelope[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"bone", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_get_bone_envelope = {
    "rig.get_bone_envelope", "rig",
    "Read the resolved tapered envelope profile for one bone segment",
    "Returns a stored per-bone override or category-derived defaults. Radii are fractions of canonical character height; start/end extensions are fractions of bone length. The bone must own an outgoing deforming segment.",
    "read", "Read", false, "Bone profile, override flag and rig revision",
    "rig|get|bone|envelope|weights|profile|read",
    "rig.apply_bone_envelope|rig.set_envelope_overlay|rig.get_weight_map",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_get_bone_envelope, 2,
    true
};
static const MethodRegistration reg_rig_get_bone_envelope(desc_rig_get_bone_envelope);

static const MethodParam params_rig_get_controls[] = {
    {"character", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_get_controls = {
    "rig.get_controls", "rig",
    "Inspect owned bound-rig IK, chain and aim controls",
    "Owned bound rig, no render job. Rows report solver, bones, root/mid/tip keys, enabled, blend, contact, target_world, pole_world, tip_world, target_error_world, aim_error_degrees, aim_axis/up_axis, orientation state, spline state/guide and length_actor. Residual is measured after IK blend and joint rules. Limb definitions persist in anatomy v3, full chains in v4 and aim controls in v5; runtime handle selection is transient. Position contacts apply to limb/chain controls. Timed target keys and bake support all solver types.",
    "read", "Read", false, "character, rig_revision, controls, selected_control, handle, coordinate_space, contact_kind, target_keys",
    "rig|get|controls|pose|IK|FK|contact",
    "rig.create_controls|rig.create_chain_control|rig.create_aim_control|rig.select_control|rig.set_ik_target|rig.set_ik_fk|rig.set_ik_contact|rig.get_pose_state",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_get_controls, 1,
    true
};
static const MethodRegistration reg_rig_get_controls(desc_rig_get_controls);

static const MethodDescriptor desc_rig_get_envelope_overlay = {
    "rig.get_envelope_overlay", "rig",
    "Read the transient viewport envelope-overlay state",
    "Read-only transient view state. The overlay is not saved in native projects and does not change skin weights. Capsule geometry is evaluated from the current displayed skeleton pose while radii remain fractions of canonical rest height.",
    "read", "Read", false, "Visibility, character and current torso/limb/extremity radius and falloff values",
    "rig|get|envelope|overlay|weights|viewport|read",
    "rig.set_envelope_overlay|rig.preview_envelope_weights|rig.get_binding",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_rig_get_envelope_overlay(desc_rig_get_envelope_overlay);

static const MethodParam params_rig_get_fit_setup[] = {
    {"character", "any", false, "Owned unskinned clip-free rig name", nullptr, nullptr},
    {"mesh", "any", false, "Exact flat mesh node name or model:<importName> group target", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_get_fit_setup = {
    "rig.get_fit_setup", "rig",
    "Prepare read-only coarse alignment and 2D landmark editing data",
    "Uses canonical flat mesh preflight and actual rest hierarchy plus actor placement. Mesh samples and bounds match viewport world coordinates using local P_orig (fallback P) and the final object transform once. Produces height/center-based coarse world-space positions for every node and at most 8192 projected mesh sample points. Does not infer anatomy or change the scene. Requires editable owned rig and finite volumetric unskinned mesh. +Y up / +Z forward and rest pose require user confirmation. Errors include rig_fit_requires_height, rig_fit_mesh_not_ready, rig_fit_input_missing, rig_fit_limit and shared ownership/preflight errors. Draft landmarks are request data, not persisted authoring state. Targets also accept model:<importName> multipart groups from rig.list_fit_targets. Exact existing mesh names take precedence. Group processing preserves independent scene geometry/transforms and hashes combined viewport world positions/indices for stale preview detection. Group errors: unknown_mesh_group, ambiguous_mesh_group, mesh_group_has_no_flat_parts, mesh_group_invalid_geometry, mesh_group_too_large.",
    "Read", "Read", false, "RigFitSetup",
    "rig|get|fit|setup",
    "rig.preflight|rig.preview_fit",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_get_fit_setup, 2,
    true
};
static const MethodRegistration reg_rig_get_fit_setup(desc_rig_get_fit_setup);

static const MethodParam params_rig_get_ik_channels[] = {
    {"character", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_get_ik_channels = {
    "rig.get_ik_channels", "rig",
    "Inspect persistent IK keys and contact intervals in the selected clip",
    "Active owned bound Pose, no render job, selected editable clip required. Version 1/2 schema (v2 adds spline_enabled and two spline_world points), seconds, world-space target/pole/unit quaternion [w,x,y,z], orientation_enabled, enabled and blend. Contacts [start,end) override keys; live controls override saved channels temporarily. Saved channels evaluate in authored playback as well as Pose.",
    "read", "Read", false, "ok",
    "rig|get|ik|channels|pose|IK|timeline|contact|bake",
    "rig.get_ik_channels|rig.insert_ik_key|rig.set_ik_contact_interval|rig.bake_ik_channels",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_get_ik_channels, 1,
    true
};
static const MethodRegistration reg_rig_get_ik_channels(desc_rig_get_ik_channels);

static const MethodDescriptor desc_rig_get_joint_limit_overlay = {
    "rig.get_joint_limit_overlay", "rig",
    "Inspect transient active-joint axes and limits display/edit settings",
    "Read-only transient session state, not project metadata. Detailed axes/limits display is scoped to the active joint. It does not enable or disable joint enforcement. No envelope display is included.",
    "read", "Read", false, "visible, edit, scope=active_joint",
    "rig|get|joint|limit|overlay|limits|viewport",
    "rig.set_joint_limit_overlay|rig.get_joint_limit_view",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_rig_get_joint_limit_overlay(desc_rig_get_joint_limit_overlay);

static const MethodParam params_rig_get_joint_limit_view[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"bone", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_get_joint_limit_view = {
    "rig.get_joint_limit_view", "rig",
    "Inspect the selected joint's rest-relative rotation frame, limits and IK/FK state",
    "Read-only, no render job. Full bone keys required. neutral_world is the current parent joint global composed with canonical local rest rotation, anchored at the current joint origin; joint_world is the actual evaluated joint transform. Both are row-major 16-float world matrices, independent of inverse skin offsets. Angles in degrees use the same rest-relative swing/twist convention as joint rules; IK-driven state follows active Pose controls. Imported rigs can be inspected but mutations require ownership. Missing hierarchy/parent or unstable transform fails explicitly. No envelope or skin weight editing is implied.",
    "read", "Read", false, "character, bone, rig_revision, owned, has_rule, rule, neutral_world, joint_world, twist_degrees, swing_degrees, outside_limits, ik_driven, pose_source, limit_space",
    "rig|get|joint|limit|view|limits|overlay|hinge|ball|IK|FK",
    "rig.get_joint_profile|rig.set_joint_limits|rig.set_joint_limit_overlay",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_get_joint_limit_view, 2,
    true
};
static const MethodRegistration reg_rig_get_joint_limit_view(desc_rig_get_joint_limit_view);

static const MethodParam params_rig_get_joint_profile[] = {
    {"character", "any", false, "Exact character name from rig.list_characters.", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_get_joint_profile = {
    "rig.get_joint_profile", "rig",
    "Read persistent rest-relative joint rules and rig revision",
    "Profile schema version 1: joints array of bone/type/enabled/lock_translation/axis/minimum/maximum/swing. Defaults: free, disabled, lock_translation true, unit axis [1,0,0], twist range [-180,180], swing 180 degrees. Types free/hinge/ball/fixed. Axis is in the joint rest-relative local frame. Ranges must contain neutral zero; no stretch or anatomical certainty is implied. Enabled rules project previews and authored playback; Auto Key records the constrained pose. A fully blocked changed request applies as a successful no-op with no keys/history. Read get_pose_state.limit_hits for projected bones. Queries are read-only; suggestions remain disabled. Set requires owned rig and current nonnegative rig_revision, rejects active Pose preview/render job, preserves rest/weights/clip keys, increments rig revision and is undoable. Native anatomy uses version 2 when rules exist, version 1 otherwise. IK integration, soft limits, elliptical cones and muscle deformation remain later work.",
    "read", "Read", false, "JointProfileState",
    "rig|get|joint|profile|anatomy|limits|hinge|swing twist|constraints",
    "rig.get_joint_profile|rig.suggest_joint_profile|rig.set_joint_profile|rig.get_pose_state|rig.preview_pose_locals",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_get_joint_profile, 1,
    true
};
static const MethodRegistration reg_rig_get_joint_profile(desc_rig_get_joint_profile);

static const MethodParam params_rig_get_mirrored_landmarks[] = {
    {"character", "any", false, "Owned meshless, unweighted, clip-free rig name", nullptr, nullptr},
    {"rig_revision", "any", false, "Expected revision from rig.list_bones; stale operations reject before publication", nullptr, nullptr},
    {"axis", "string", false, "Mirror plane normal in canonical rig space: x, y or z", "x", nullptr},
    {"offset", "float", false, "Finite plane coordinate in rig units; actor placement does not change this coordinate", "0.", nullptr},
    {"landmarks", "any", false, "Complete finite world XYZ map for every canonical hierarchy node", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_get_mirrored_landmarks = {
    "rig.get_mirrored_landmarks", "rig",
    "Return a mirrored copy of a world-space fitting landmark draft",
    "Read-only; returns a new complete map, leaving scene/history/selection and input unchanged. Uses the same anatomy pairing and plane service as rest editing; world reflection is actor * rig reflection * inverse(actor), including translated/rotated/scaled actors. Reflects request source positions, not rest defaults. Bounds containment and zero-length validation remain in preview_fit/commit_fit, so drafts can be manually corrected before Apply. Alignment live mirror and directional buttons call this API; UI starts X offset at mesh bounds center converted to rig space. Named errors include rig_mirror_invalid_plane, rig_mirror_invalid_direction, rig_mirror_limit, rig_mirror_unpaired_bone, rig_mirror_ambiguous_pair, rig_mirror_no_pairs, rig_selection_empty, rig_selection_duplicate_bone, unknown_bone, rig_edit_stale_revision, rig_mirror_invalid_transform and shared anatomy/edit/scene lock errors. Also rig_fit_incomplete_landmarks, rig_fit_invalid_landmark and rig_mirror_failed.",
    "Read", "Read", false, "map<bone,array<float,3>>",
    "rig|get|mirrored|landmarks",
    "rig.get_fit_setup|rig.preview_fit|rig.commit_fit|rig.mirror_rest",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_get_mirrored_landmarks, 5,
    true
};
static const MethodRegistration reg_rig_get_mirrored_landmarks(desc_rig_get_mirrored_landmarks);

static const MethodDescriptor desc_rig_get_mode = {
    "rig.get_mode", "rig",
    "Read transient viewport rig interaction mode and locked character",
    " Pose mode is supported for owned bound rigs. Scene exits Pose and clears transient overrides; Edit requires exiting Pose first.",
    "Read", "Read", false, "{mode: scene|edit, character: string}",
    "rig|get|mode",
    "rig.set_mode",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_rig_get_mode(desc_rig_get_mode);

static const MethodParam params_rig_get_next_bone_name[] = {
    {"seed", "string", false, "ASCII authored name seed; defaults to Joint1", "Joint1", nullptr},
    {"character", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_get_next_bone_name = {
    "rig.get_next_bone_name", "rig",
    "Suggest a free authored bone name for an editable owned rig",
    "Returns seed unchanged when free; otherwise increments its trailing integer (or appends 1). Checks canonical hierarchy and global BoneData keys. Read-only suggestion does not reserve the name. Errors include invalid_bone_name, bone_name_exhausted and shared rig eligibility errors.",
    "Read", "Read", false, "string",
    "rig|get|next|bone|name",
    "rig.add_bone",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_get_next_bone_name, 2,
    true
};
static const MethodRegistration reg_rig_get_next_bone_name(desc_rig_get_next_bone_name);

static const MethodParam params_rig_get_next_name[] = {
    {"seed", "string", false, "", "Rig", nullptr},
};
static const MethodDescriptor desc_rig_get_next_name = {
    "rig.get_next_name", "rig",
    "Suggest a collision-free scene rig name from an ASCII seed",
    "Uses the same scene/prefix/global bone/flat mesh name collision checks as rig.create. Returns seed if free, otherwise seed1, seed2, etc. Does not reserve names. Creation of multiple rigs is supported; creation/naming require Scene mode. Errors: invalid_rig_name, rig_name_exhausted, rig_edit_active, scene_locked.",
    "Read", "Read", false, "string",
    "rig|get|next|name",
    "rig.create",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_get_next_name, 1,
    true
};
static const MethodRegistration reg_rig_get_next_name(desc_rig_get_next_name);

static const MethodDescriptor desc_rig_get_overlay_visible = {
    "rig.get_overlay_visible", "rig",
    "Read whether the skeleton viewport overlay is enabled",
    "Works with skeleton-only and meshless animated imports. Selection/overlay are transient view state, not authoring edits; no undo. Names are import-prefixed. Joint world transforms are row-major 16 floats from evaluated hierarchy globals, never skin-offset matrices. pose_source is bind, graph, controller or ozz. Bind fallback is explicit per node. Independent skeleton root-motion placement and bind/pose editing are subsequent roadmap work.",
    "read", "Read", false, "bool",
    "rig|get|overlay|visible|skeleton|joint|meshless",
    "rig.list_characters|rig.list_bones|rig.select_bone|rig.get_selected_bone",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_rig_get_overlay_visible(desc_rig_get_overlay_visible);

static const MethodParam params_rig_get_pose_coverage[] = {
    {"character", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_get_pose_coverage = {
    "rig.get_pose_coverage", "rig",
    "Read flat bound-mesh weight coverage without changing weights",
    "7A/8A: requires an owned rig with a bound flat mesh; imported rigs remain read-only for pose authoring. Mutations require active pose mode and no render job. Rest transforms and weights are never modified. Frame time uses the scene FPS; create_pose_clip.fps controls clip ticks per second. Matrices are row-major 16 numbers, rigid translation/rotation only. Preview replaces the previous preview relative to the committed pose; frame changes and mode exit discard unkeyed overrides. Keys persist in native projects; clip selection and Auto Key are transient. Unweighted vertices retain their bind position.",
    "read", "Read", false, "PoseCoverage",
    "rig|get|pose|coverage|FK|bone keys|authoring",
    "rig.set_mode|rig.get_pose_state|rig.preview_pose_locals|rig.apply_pose_preview",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_get_pose_coverage, 1,
    true
};
static const MethodRegistration reg_rig_get_pose_coverage(desc_rig_get_pose_coverage);

static const MethodParam params_rig_get_pose_state[] = {
    {"character", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_get_pose_state = {
    "rig.get_pose_state", "rig",
    "Read the evaluated local pose, editable clips and rig revision",
    "7A/8A: requires an owned rig with a bound flat mesh; imported rigs remain read-only for pose authoring. Mutations require active pose mode and no render job. Rest transforms and weights are never modified. Frame time uses the scene FPS; create_pose_clip.fps controls clip ticks per second. Matrices are row-major 16 numbers, rigid translation/rotation only. Preview replaces the previous preview relative to the committed pose; frame changes and mode exit discard unkeyed overrides. Keys persist in native projects; clip selection and Auto Key are transient. Unweighted vertices retain their bind position. limit_hits lists bones projected by enabled joint rules; during preview it reports the original request, otherwise sampled clip/pose constraints.",
    "read", "Read", false, "PoseState",
    "rig|get|pose|state|FK|bone keys|authoring",
    "rig.set_mode|rig.get_pose_state|rig.preview_pose_locals|rig.apply_pose_preview",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_get_pose_state, 1,
    true
};
static const MethodRegistration reg_rig_get_pose_state(desc_rig_get_pose_state);

static const MethodParam params_rig_get_pose_view[] = {
    {"character", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_get_pose_view = {
    "rig.get_pose_view", "rig",
    "Read requested and effective evaluation view for one rig",
    "effective_mode is rest while scoped Rig Edit owns this character; requested mode is preserved for leaving Edit. Transient default is animated. On import/native reopen, valid skinned characters default to Rest; meshless clips retain prior animated defaults. Explicit per-character views are preserved during append/reinitialization. Rest graphs are excluded from autonomous viewport wake and file-animation reset scheduling. Select animated explicitly to resume existing animation.",
    "Read", "Read", false, "{mode: rest|animated, effective_mode: rest|animated}",
    "rig|get|pose|view",
    "rig.set_pose_view",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_get_pose_view, 1,
    true
};
static const MethodRegistration reg_rig_get_pose_view(desc_rig_get_pose_view);

static const MethodParam params_rig_get_scene_transform[] = {
    {"character", "any", false, "Owned meshless rig name", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_get_scene_transform = {
    "rig.get_scene_transform", "rig",
    "Read the independent placement matrix of an owned meshless rig",
    "Returns actor placement separately from canonical local rest, sampled joint globals and clips. Available in Scene and Rig Edit. Imported/weighted rigs are not supported by this owned placement service. Errors: unknown_character, rig_placement_requires_meshless_unskinned, scene_locked. Actor placement may include positive uniform scale.",
    "Read", "Read", false, "number[16] (row-major)",
    "rig|get|scene|transform",
    "rig.set_scene_transform|rig.list_bones",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_get_scene_transform, 1,
    true
};
static const MethodRegistration reg_rig_get_scene_transform(desc_rig_get_scene_transform);

static const MethodDescriptor desc_rig_get_selected_bone = {
    "rig.get_selected_bone", "rig",
    "Read the currently selected skeleton node or null when selection is absent or stale",
    "Works with skeleton-only and meshless animated imports. Selection/overlay are transient view state, not authoring edits; no undo. Names are import-prefixed. Joint world transforms are row-major 16 floats from evaluated hierarchy globals, never skin-offset matrices. pose_source is bind, graph, controller or ozz. Bind fallback is explicit per node. Independent skeleton root-motion placement and bind/pose editing are subsequent roadmap work. Returns active joint of multi-selection; rig.get_selection returns full selection set.",
    "read", "Read", false, "BoneView|null",
    "rig|get|selected|bone|skeleton|joint|meshless|overlay",
    "rig.list_characters|rig.list_bones|rig.select_bone|rig.get_selected_bone",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_rig_get_selected_bone(desc_rig_get_selected_bone);

static const MethodDescriptor desc_rig_get_selection = {
    "rig.get_selection", "rig",
    "Read the canonical selection set, active joint and pivot preference",
    "Read-only shared scene selection. bones are unique keys in canonical skeleton order. Empty/absent/stale active yields empty character/active/bones. Missing members are excluded; legacy single active is fallback when older authoring operations changed identity. pivot active or center; choice is transient and remains when selection clears. Active joint drives inspector/gizmo orientation and selected-bone weight map; other selected joints retain highlight. Native save does not persist selection, reopening clears it via existing clearSelection. Errors api_not_bound, scene_locked, rig_selection_failed. Python rt.rig.get_selection(). anchor is the normalized valid range anchor or active fallback; restore via select_bones anchor parameter.",
    "Read", "Read", false, "{character,bones,active,anchor,pivot}",
    "rig|get|selection",
    "rig.select_bones|rig.get_selected_bone|rig.set_selection_pivot",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_rig_get_selection(desc_rig_get_selection);

static const MethodParam params_rig_get_template[] = {
    {"template_id", "any", false, "", nullptr, "['root', 'chain3', 'humanoid', 'quadruped', 'insect6', 'avian']"},
    {"height", "float", false, "Positive finite layout height, <=10000; default 1.8", "1.8", nullptr},
};
static const MethodDescriptor desc_rig_get_template = {
    "rig.get_template", "rig",
    "Read a scaled template hierarchy and anatomy without creating scene state",
    "Calls the same template builder as rig.create; returned keys use Template_ prefix for inspection. Hierarchy/anatomy schema v1 contains canonical hierarchy serialization and anatomy; actual creation uses requested character prefix. All joints have identity rest rotation, unit scale and template positions. Height is ground-origin to highest joint endpoint, in scene units; insect height includes antennae. No scene/history/selection/runtime mutations. Errors: unknown_rig_template, invalid_rig_height, invalid_rig_template_definition and anatomy validation errors. template_version is recipe provenance: humanoid is v3, quadruped/insect6 are v2; avian/root/chain3 are v1. Read and create use identical recipes.",
    "Read", "Read", false, "{template_id,template_version,height,nodeHierarchy,anatomy}",
    "rig|get|template",
    "rig.list_templates|rig.create|rig.get_anatomy",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_get_template, 2,
    true
};
static const MethodRegistration reg_rig_get_template(desc_rig_get_template);

static const MethodParam params_rig_get_weight_map[] = {
    {"bone", "any", false, "", nullptr, nullptr},
    {"character", "any", false, "", nullptr, nullptr},
    {"mesh", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_get_weight_map = {
    "rig.get_weight_map", "rig",
    "Read a selected bone scalar weight field in canonical flat vertex order",
    "Exact individual mesh, character and unique bone key required. Shared RigWeights::boneWeightField reads actual DNA skin_weights; verifies unique canonical ownership (authored binding registry takes precedence), hierarchy membership and unambiguous forward bone index. Missing/empty rows contribute zero; matching duplicate entries sum in double; nonfinite/negative matching values are excluded and counted in invalid_entries. Display values clamp to [0,1], no stored weight normalization/repair or sculpt-mask mutation. Reports mesh, character, bone, bone_index, vertex_count, values, invalid_entries, display_clamped=true. Max 2M vertices. Unknown bone/owner/index/foreign errors: unknown_bone, rig_weight_map_unverified_owner, rig_weight_map_foreign_character, rig_weight_map_invalid_index; limit rig_weight_map_limit; exact mesh errors propagate; api_not_bound, scene_locked, rig_weight_map_failed. Numeric/index diagnostics remain rig.weight_stats, this display map is not a contract/deformation certificate. Python rt.rig.get_weight_map(mesh,character,bone).",
    "Read", "Read", false, "BoneWeightMap",
    "rig|get|weight|map",
    "rig.get_weights|rig.weight_stats|rig.set_weight_map_visible",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_get_weight_map, 3,
    true
};
static const MethodRegistration reg_rig_get_weight_map(desc_rig_get_weight_map);

static const MethodDescriptor desc_rig_get_weight_map_visible = {
    "rig.get_weight_map_visible", "rig",
    "Read the transient selected-bone weight tint toggle",
    "Same inspector/script/IPC ViewState flag. Defaults false, remains independent of bone overlay visibility and sculpt mask visibility. No mutation/history. Error api_not_bound. Python rt.rig.get_weight_map_visible().",
    "Read", "Read", false, "bool",
    "rig|get|weight|map|visible",
    "rig.set_weight_map_visible",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_rig_get_weight_map_visible(desc_rig_get_weight_map_visible);

static const MethodParam params_rig_get_weights[] = {
    {"object", "any", false, "Exact individual flat mesh nodeName", nullptr, nullptr},
    {"vertex", "any", false, "Zero-based canonical flat vertex index; nonnegative", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_get_weights = {
    "rig.get_weights", "rig",
    "Read one canonical flat vertex influence list with bone identities",
    "Read-only stored TriangleMesh/DNA skin_weights row, exact individual mesh nodeName only. object parameter names the mesh, vertex is a zero-based nonnegative integer in canonical flat vertex order (not a face/corner selection). Preserves influence order/values, never normalizes. influences records bone_index, unique bone name (null if unknown/ambiguous), weight (null for nonfinite stored values), value_valid, index_known, index_ambiguous, belongs_to_character (null when ownership or index cannot be established). weight_sum includes finite positive values with nonnegative IDs, matching weight_stats finite-sum semantics; values_valid does not certify normalization, duplicate/limit rules or index ownership. Empty/missing row yields empty influences and sum 0; weight_row_present distinguishes them. Reports resolved/unresolved/ambiguous ownership and character, using same core as weight_stats. Nonfinite values are null for safe JSON parity; no scene/rest/pose/selection/history mutation. IPC requires object string and nonnegative integer vertex; bool/float/negative shape errors invalid_parameter. Python uses typed object string/uint64 vertex and may raise binding conversion errors for invalid types or negative/outside uint64 inputs. Shared service errors: invalid_mesh_name, unknown_mesh, ambiguous_mesh_name, mesh_has_no_geometry, rig_vertex_out_of_range, rig_get_weights_failed, api_not_bound, scene_locked. Python rt.rig.get_weights(object, vertex). Initial authored binding registry takes precedence over original source hierarchy/membership; preserves correct owner after multipart binding and native reopen.",
    "Read", "Read", false, "VertexSkinWeights",
    "rig|get|weights",
    "rig.weight_stats",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_get_weights, 2,
    true
};
static const MethodRegistration reg_rig_get_weights(desc_rig_get_weights);

static const MethodParam params_rig_insert_ik_key[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"rig_revision", "integer", true, "", nullptr, nullptr},
    {"control", "string", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_insert_ik_key = {
    "rig.insert_ik_key", "rig",
    "Key the effective limb IK control at the current frame",
    "Requires active owned bound Pose, no preview/render job, selected editable clip and matching rig_revision. Captures effective handles/blend/orientation/enabled, replacing an existing key at the same time. Before first key no control; after last hold; positions/pole/blend linear, orientation slerp, spline points linear when both keys carry them, orientation/spline activation stepped; IK ramps between enabled endpoints with positive interpolated blend. Time is frame/Pose fps; clip duration extends. Live override for this limb clears. One undo step, native persistence. Max 256 channels/10000 keys+contacts. Errors: rig_pose_clip_required, rig_ik_unknown_control, rig_edit_stale_revision, rig_pose_time_limit.",
    "write", "SceneWrite", false, "ok",
    "rig|insert|ik|key|pose|IK|timeline|contact|bake",
    "rig.get_ik_channels|rig.insert_ik_key|rig.set_ik_contact_interval|rig.bake_ik_channels",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_insert_ik_key, 3,
    true
};
static const MethodRegistration reg_rig_insert_ik_key(desc_rig_insert_ik_key);

static const MethodParam params_rig_insert_pose_keys[] = {
    {"bones", "any", false, "", nullptr, nullptr},
    {"character", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_insert_pose_keys = {
    "rig.insert_pose_keys", "rig",
    "Insert or replace position and quaternion keys for explicit bones at the current frame",
    "7A/8A: requires an owned rig with a bound flat mesh; imported rigs remain read-only for pose authoring. Mutations require active pose mode and no render job. Rest transforms and weights are never modified. Frame time uses the scene FPS; create_pose_clip.fps controls clip ticks per second. Matrices are row-major 16 numbers, rigid translation/rotation only. Preview replaces the previous preview relative to the committed pose; frame changes and mode exit discard unkeyed overrides. Keys persist in native projects; clip selection and Auto Key are transient. Unweighted vertices retain their bind position. When the selected clip has timed IK, driven bone insertion fails rig_ik_use_control_keys_or_bake; use insert_ik_key or bake_ik_channels.",
    "write", "SceneWrite", true, "ok",
    "rig|insert|pose|keys|FK|bone keys|authoring",
    "rig.set_mode|rig.get_pose_state|rig.preview_pose_locals|rig.apply_pose_preview",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_insert_pose_keys, 2,
    true
};
static const MethodRegistration reg_rig_insert_pose_keys(desc_rig_insert_pose_keys);

static const MethodParam params_rig_list_bones[] = {
    {"character", "any", false, "Import name from rig.list_characters", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_list_bones = {
    "rig.list_bones", "rig",
    "Inspect skeleton joints and helper nodes with their current world transforms",
    "Works with skeleton-only and meshless animated imports. Selection/overlay are transient view state, not authoring edits; no undo. Names are import-prefixed. Joint world transforms are row-major 16 floats from evaluated hierarchy globals, never skin-offset matrices. pose_source is bind, graph, controller or ozz. Bind fallback is explicit per node. Independent skeleton root-motion placement and bind/pose editing are subsequent roadmap work. Rows now include local_rest_transform, authoring_owned, rig_revision, template_id and separate in_bonedata/in_skeleton_nodes/in_node_hierarchy/in_ozz_skeleton flags. scene_transform reports actor placement; owned meshless world_transform includes it. Scene selection of any joint can place its entire owned rig. template_version is saved recipe provenance; legacy missing versions are 1.",
    "read", "Read", false, "BoneView[]",
    "rig|list|bones|skeleton|joint|meshless|overlay",
    "rig.list_characters|rig.list_bones|rig.select_bone|rig.get_selected_bone",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_list_bones, 1,
    true
};
static const MethodRegistration reg_rig_list_bones(desc_rig_list_bones);

static const MethodDescriptor desc_rig_list_characters = {
    "rig.list_characters", "rig",
    "List imports containing a skeleton, including rigs without mesh or clips",
    "Works with skeleton-only and meshless animated imports. Selection/overlay are transient view state, not authoring edits; no undo. Names are import-prefixed. Joint world transforms are row-major 16 floats from evaluated hierarchy globals, never skin-offset matrices. pose_source is bind, graph, controller or ozz. Bind fallback is explicit per node. Independent skeleton root-motion placement and bind/pose editing are subsequent roadmap work.",
    "read", "Read", false, "string[]",
    "rig|list|characters|skeleton|joint|meshless|overlay",
    "rig.list_characters|rig.list_bones|rig.select_bone|rig.get_selected_bone",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_rig_list_characters(desc_rig_list_characters);

static const MethodDescriptor desc_rig_list_fit_targets = {
    "rig.list_fit_targets", "rig",
    "List individual flat mesh and multipart imported-character targets for manual fitting",
    "Individual targets are exact flat node names. Multipart groups use model:<importName> and include canonical flat scene meshes associated via model flat members or exact NodeHierarchy keys. Groups preserve separate scene parts; all are sampled into a temporary world-space diagnostic flat snapshot, never inserted into scene or bound. No subset filtering yet; choose an individual body mesh when accessories affect bounds. Existing skin on any group part blocks owned unskinned fitting.",
    "Read", "Read", false, "[{target,label,kind,part_count}]",
    "rig|list|fit|targets",
    "rig.preflight|rig.get_fit_setup",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_rig_list_fit_targets(desc_rig_list_fit_targets);

static const MethodDescriptor desc_rig_list_templates = {
    "rig.list_templates", "rig",
    "List shared rig template catalogue with anatomy family, version and joint counts",
    "Read-only, scene-independent catalogue: root, chain3, humanoid, quadruped, insect6, avian. Layout is +Y up, +Z forward, +X character-left. default_height is a UI suggestion; rig.create/get_template retain height=1.8 historical default unless supplied. Templates are rest skeleton layouts with neutral rotations; no fitting, weights or IK. Humanoid basic v3 restores sagittal depth and mild forward knee/elbow hints; quadruped and insect6 basic v2 use revised proportions; avian, root and chain3 are v1. Avian default_height is 0.6 scene units. Existing saved rigs retain their concrete hierarchy.",
    "Read", "Read", false, "[{template_id,template_version,label,family,joint_count,default_height,up_axis,forward_axis,left_axis}]",
    "rig|list|templates",
    "rig.get_template|rig.create",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_rig_list_templates(desc_rig_list_templates);

static const MethodParam params_rig_mirror_pose[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"bones", "array", true, "", nullptr, nullptr},
    {"rig_revision", "integer", true, "", nullptr, nullptr},
    {"direction", "string", false, "", "selected", nullptr},
    {"axis", "string", false, "", "x", nullptr},
};
static const MethodDescriptor desc_rig_mirror_pose = {
    "rig.mirror_pose", "rig",
    "Preview paired rest-relative FK motion on the opposite side",
    "Active owned bound Pose, no render job or pending preview; switch IK to FK first. Copies local rest-relative motion through joint rest orientation bases across actor-local x/y/z, preserving target rest origins/lengths. Uses anatomy pairs, no name guessing. selected requires one side per pair; both sides fail rig_mirror_ambiguous_pair. Directional copy with empty bones selects all pairs. Apply commits one history step, Auto Key writes changed channels; Cancel discards preview. Matching nonnegative revision required.",
    "write", "SceneWrite", false, "ok; transient preview",
    "rig|mirror|pose|FK",
    "rig.get_anatomy|rig.apply_pose_preview|rig.cancel_pose_preview",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_mirror_pose, 5,
    true
};
static const MethodRegistration reg_rig_mirror_pose(desc_rig_mirror_pose);

static const MethodParam params_rig_mirror_rest[] = {
    {"character", "any", false, "Owned meshless, unweighted, clip-free rig name", nullptr, nullptr},
    {"rig_revision", "any", false, "Expected revision from rig.list_bones; stale operations reject before publication", nullptr, nullptr},
    {"axis", "string", false, "Mirror plane normal in canonical rig space: x, y or z", "x", nullptr},
    {"offset", "float", false, "Finite plane coordinate in rig units; actor placement does not change this coordinate", "0.", nullptr},
};
static const MethodDescriptor desc_rig_mirror_rest = {
    "rig.mirror_rest", "rig",
    "Copy mirrored paired rest transforms in one canonical rig edit",
    "Uses anatomy symmetry metadata, never bone-name guessing. Axis/offset are rig-space coordinates. Reflect immutable source joint globals and reflect a basis axis to preserve proper handedness, then solve target locals against updated parents; selected parent/child targets receive reflection once. Untargeted descendants inherit their parent changes. Targets become current selection. Stable existing indices, all derived representations/revision/runtime refresh and one production undo/redo; no new geometry/weights. Rejects weighted/bound/imported/non-owned rigs, clips, stale revision and no-op. Selected both sides of a pair is ambiguous; directional mode filters requested side, [] copies all pairs. Named errors include rig_mirror_invalid_plane, rig_mirror_invalid_direction, rig_mirror_limit, rig_mirror_unpaired_bone, rig_mirror_ambiguous_pair, rig_mirror_no_pairs, rig_selection_empty, rig_selection_duplicate_bone, unknown_bone, rig_edit_stale_revision, rig_mirror_invalid_transform and shared anatomy/edit/scene lock errors.",
    "SceneWrite", "SceneWrite", false, "{ok:true}",
    "rig|mirror|rest",
    "rig.get_anatomy|rig.get_mirrored_landmarks|rig.create_mirrored_bone",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_mirror_rest, 4,
    true
};
static const MethodRegistration reg_rig_mirror_rest(desc_rig_mirror_rest);

static const MethodParam params_rig_preflight[] = {
    {"mesh", "any", false, "Exact flat mesh node name or model:<importName> group target", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_preflight = {
    "rig.preflight", "rig",
    "Inspect canonical flat mesh geometry before anatomical landmark setup",
    "Read-only snapshot over TriangleMesh/DNA viewport positions with object final transform: unskinned local P_orig (fallback P), skinned current P. Static P can be a baked cache and is not transformed again when P_orig exists. No Triangle facades. Requires unique exact mesh nodeName. Reports counts, transformed bounds, nonfinite vertices, invalid/out-of-range/numerically invalid triangles, exact zero-area faces, trailing indices and existing skin. can_start_landmarks only checks finite volumetric unskinned geometry; fit_ready is always false. Axes, rest pose, symmetry, interior and watertightness are not evaluated. No fit, bind, weights, scene/history/selection mutation. Re-run after mesh changes. Errors: invalid_mesh_name, unknown_mesh, ambiguous_mesh_name, mesh_has_no_geometry, mesh_has_no_positions, mesh_position_buffer_incomplete, invalid_mesh_transform, rig_preflight_failed, scene_locked. Targets also accept model:<importName> multipart groups from rig.list_fit_targets. Exact existing mesh names take precedence. Group processing preserves independent scene geometry/transforms and hashes combined viewport world positions/indices for stale preview detection. Group errors: unknown_mesh_group, ambiguous_mesh_group, mesh_group_has_no_flat_parts, mesh_group_invalid_geometry, mesh_group_too_large. Reports blockers and group part_count/parts. Isolated zero-area faces are warnings for manual setup when valid faces remain; nonfinite/out-of-range/trailing indices and no valid faces still block. This does not relax future bind/weight validation.",
    "Read", "Read", false, "MeshPreflightReport",
    "rig|preflight",
    "rig.create|rig.get_template",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_preflight, 1,
    true
};
static const MethodRegistration reg_rig_preflight(desc_rig_preflight);

static const MethodParam params_rig_preview_bind[] = {
    {"character", "any", false, "Exact owned rig character", nullptr, nullptr},
    {"mesh", "any", false, "Exact flat part or model: group target", nullptr, nullptr},
    {"axes_confirmed", "bool", false, "Caller has inspected axes and rest alignment", "false", nullptr},
};
static const MethodDescriptor desc_rig_preview_bind = {
    "rig.preview_bind", "rig",
    "Preview initial nearest-segment weights for an aligned owned rig and actual flat mesh parts",
    "Initial bind only: authoring-owned meshless, clip-free rig and static unskinned actual flat mesh parts. Exact mesh name takes precedence over model:<importName> group. Preserves part names/topology/materials/UV; converts flat local P_orig/N_orig (fallback P/N) through inverse(actor placement)*source base into common rig bind space. All parts share actor Transform; inverse joint-global rest offsets and model inverse identity. Root anatomy role is nondeforming; segment weights belong to parent joint, nearest outgoing segment per joint, inverse squared distance regularized by extent*1e-4 (minimum 1e-9), strongest four merged/normalized via common contract, negligible entries pruned. No visibility/interior or alignment certification. Rejects animation, skin, modifiers/geometry graphs/deltas, incomplete or invalid normals/geometry, reflected/singular transforms. Limits: 256 parts, 2M vertices, 100M vertex-segment evaluations, 4096 joints. Existing canEditRig guards apply. Read-only; no scene/history/backend mutation. axes_confirmed defaults false and must be true. Aggregate counts/sums, outside bounds warning and up to 16 sampled rows per part; can_bind true is not a quality certificate. Tokens cover canonical rig/rest/placement/indices/root roles and selected mesh local P_orig/N_orig (fallback P/N)/indices/base. Python rt.rig.preview_bind(character,mesh,axes_confirmed=False). Errors include rig_bind_axes_unconfirmed, rig_bind_limit, rig_bind_requires_segments, rig_bind_requires_unskinned_mesh, rig_bind_requires_static_mesh, rig_bind_source_has_animation, rig_bind_requires_normals, rig_bind_invalid_normals, rig_bind_invalid_geometry, rig_bind_invalid_transform, rig_bind_invalid_rest, rig_bind_invalid_indices, rig_bind_zero_length_segment, mesh_already_bound, rig_bind_mesh_pending_delete, rig_bind_invalid_mesh_name, rig_bind_empty_weights, rig_bind_failed; shared name/rig/preflight errors also propagate.",
    "Read", "Read", false, "RigBindPreview",
    "rig|preview|bind",
    "rig.bind_mesh|rig.get_binding|rig.preview_fit",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_preview_bind, 3,
    true
};
static const MethodRegistration reg_rig_preview_bind(desc_rig_preview_bind);

static const MethodParam params_rig_preview_envelope_weights[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"torso_radius", "float", false, "", "0.16", nullptr},
    {"limb_radius", "float", false, "", "0.065", nullptr},
    {"extremity_radius", "float", false, "", "0.05", nullptr},
    {"falloff", "float", false, "", "2.0", nullptr},
};
static const MethodDescriptor desc_rig_preview_envelope_weights = {
    "rig.preview_envelope_weights", "rig",
    "Preview bounded anatomical capsule weights on an authored bound rig",
    "Read-only recomputation over canonical flat P_orig positions of every part in the authored binding registry. Each parent-owned bone segment receives a height-relative capsule radius selected from anatomy roles: torso, limb or hand/foot extremity. Influence is zero outside the capsule, falls smoothly inside, keeps the strongest four and normalizes through the shared weight contract. Vertices outside every capsule use the nearest segment and are counted as fallback_vertices, so preview never invents unweighted rows. Edit Bone UI enables the separate transient viewport overlay after a successful preview; the IPC operation itself leaves view state unchanged. Does not support arbitrary imported skins or parts missing canonical bind positions. Limits: 256 parts, 2M vertices and 4096 joints. Settings: torso_radius .02..0.50, limb_radius .01..0.30, extremity_radius .005..0.20 of character height; falloff .5..8.",
    "read", "Read", false, "Envelope report with counts, radii and fallback vertices; scene unchanged",
    "rig|preview|envelope|weights|capsule|skin",
    "rig.apply_envelope_weights|rig.get_weight_map|rig.weight_stats|rig.get_binding",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_preview_envelope_weights, 5,
    true
};
static const MethodRegistration reg_rig_preview_envelope_weights(desc_rig_preview_envelope_weights);

static const MethodParam params_rig_preview_fit[] = {
    {"character", "any", false, "Owned unskinned clip-free rig name", nullptr, nullptr},
    {"mesh", "any", false, "Exact flat mesh node name or model:<importName> group target", nullptr, nullptr},
    {"landmarks", "any", false, "Every canonical node key maps to three finite world-space coordinates", nullptr, nullptr},
    {"axes_confirmed", "bool", false, "Caller confirms +Y up, +Z forward and rest pose", "false", nullptr},
};
static const MethodDescriptor desc_rig_preview_fit = {
    "rig.preview_fit", "rig",
    "Validate complete world-space manual joint landmarks and return a commit preview",
    "Read-only manual mode. Requires complete finite landmarks and nonzero parent segments. Reports segment lengths and bounds containment, not surface interior or automatic fit residual. can_commit is false for any outside-bounds point; interior_verified remains false. Returns rig_revision and a content token over flat viewport source positions, indices and object final transform. No scene/history changes. Errors: rig_fit_axes_unconfirmed, rig_fit_incomplete_landmarks, rig_fit_invalid_landmark, rig_fit_zero_length_bone, rig_fit_mesh_not_ready, rig_fit_failed plus shared rig/preflight errors. Targets also accept model:<importName> multipart groups from rig.list_fit_targets. Exact existing mesh names take precedence. Group processing preserves independent scene geometry/transforms and hashes combined viewport world positions/indices for stale preview detection. Group errors: unknown_mesh_group, ambiguous_mesh_group, mesh_group_has_no_flat_parts, mesh_group_invalid_geometry, mesh_group_too_large. Bounds checks include a reported scale/float-precision tolerance to avoid rejecting boundary landmarks solely for rounding.",
    "Read", "Read", false, "RigFitPreview",
    "rig|preview|fit",
    "rig.get_fit_setup|rig.commit_fit",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_preview_fit, 4,
    true
};
static const MethodRegistration reg_rig_preview_fit(desc_rig_preview_fit);

static const MethodParam params_rig_preview_human_walk[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"rig_revision", "any", true, "", nullptr, nullptr},
    {"fps", "float", true, "", nullptr, nullptr},
    {"cadence", "float", true, "", nullptr, nullptr},
    {"cycles", "integer", true, "", nullptr, nullptr},
    {"stride", "float", true, "", nullptr, nullptr},
    {"step_height", "float", true, "", nullptr, nullptr},
    {"body_bounce", "float", true, "", nullptr, nullptr},
    {"arm_swing", "float", true, "", nullptr, nullptr},
    {"body_motion", "float", false, "", "0.75", nullptr},
    {"name", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_preview_human_walk = {
    "rig.preview_human_walk", "rig",
    "Validate and describe a humanoid in-place walk recipe",
    "Read-only validation for the first quick-motion recipe. Requires active Pose on an owned bound humanoid with canonical pelvis, left/right ankle roles and left/right arm/leg chains. Recipe values are height fractions, except cadence in steps/minute, fps and cycles. body_motion 0..1 controls pelvis/torso naturalization and defaults to .75 when omitted. Reports planted foot orientation, pelvis sway and availability of upper-spine counter rotation, head stabilization and shoulder-girdle motion from optional mapped roles. It does not mutate pose, clips, controls or history. The v1 recipe has no root motion, path, footprints, terrain query or contact channel.",
    "write", "SceneWrite", false, "Recipe plan with frames, duration, world-scaled motion values and requirements",
    "rig|preview|human|walk|animation|gait|recipe|humanoid",
    "rig.create_human_walk_clip|rig.get_anatomy|rig.get_pose_state|rig.get_controls",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_preview_human_walk, 11,
    true
};
static const MethodRegistration reg_rig_preview_human_walk(desc_rig_preview_human_walk);

static const MethodParam params_rig_preview_pose_locals[] = {
    {"local_transforms", "any", false, "Object mapping bone names to absolute local row-major rigid matrices.", nullptr, nullptr},
    {"rig_revision", "any", false, "Nonnegative revision from get_pose_state; stale revisions reject without mutation.", nullptr, nullptr},
    {"character", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_preview_pose_locals = {
    "rig.preview_pose_locals", "rig",
    "Preview explicit absolute local rigid matrices with revision validation",
    "7A/8A: requires an owned rig with a bound flat mesh; imported rigs remain read-only for pose authoring. Mutations require active pose mode and no render job. Rest transforms and weights are never modified. Frame time uses the scene FPS; create_pose_clip.fps controls clip ticks per second. Matrices are row-major 16 numbers, rigid translation/rotation only. Preview replaces the previous preview relative to the committed pose; frame changes and mode exit discard unkeyed overrides. Keys persist in native projects; clip selection and Auto Key are transient. Unweighted vertices retain their bind position.",
    "write", "SceneWrite", false, "ok",
    "rig|preview|pose|locals|FK|bone keys|authoring",
    "rig.set_mode|rig.get_pose_state|rig.preview_pose_locals|rig.apply_pose_preview",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_preview_pose_locals, 3,
    true
};
static const MethodRegistration reg_rig_preview_pose_locals(desc_rig_preview_pose_locals);

static const MethodParam params_rig_preview_pose_transform[] = {
    {"world_delta", "any", false, "Row-major rigid world delta; selected parent/child transforms apply once.", nullptr, nullptr},
    {"rig_revision", "any", false, "Nonnegative revision from get_pose_state; stale revisions reject without mutation.", nullptr, nullptr},
    {"bones", "any", false, "", nullptr, nullptr},
    {"character", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_preview_pose_transform = {
    "rig.preview_pose_transform", "rig",
    "Preview a rigid world delta on explicit bones with revision validation",
    "7A/8A: requires an owned rig with a bound flat mesh; imported rigs remain read-only for pose authoring. Mutations require active pose mode and no render job. Rest transforms and weights are never modified. Frame time uses the scene FPS; create_pose_clip.fps controls clip ticks per second. Matrices are row-major 16 numbers, rigid translation/rotation only. Preview replaces the previous preview relative to the committed pose; frame changes and mode exit discard unkeyed overrides. Keys persist in native projects; clip selection and Auto Key are transient. Unweighted vertices retain their bind position.",
    "write", "SceneWrite", false, "ok",
    "rig|preview|pose|transform|FK|bone keys|authoring",
    "rig.set_mode|rig.get_pose_state|rig.preview_pose_locals|rig.apply_pose_preview",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_preview_pose_transform, 4,
    true
};
static const MethodRegistration reg_rig_preview_pose_transform(desc_rig_preview_pose_transform);

static const MethodParam params_rig_rename_bone[] = {
    {"character", "any", false, "Owned rig name", nullptr, nullptr},
    {"bone", "any", false, "Existing prefixed bone key", nullptr, nullptr},
    {"name", "any", false, "New authored name, 1-128 ASCII letters/digits/underscore/hyphen", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_rename_bone = {
    "rig.rename_bone", "rig",
    "Rename an owned rig bone while preserving its scene index",
    "Owned, unskinned, clip-free rigs only. Uses one shared staged core and undo command; other rig selection/edit is blocked in scoped Rig Edit. Shared errors: unknown_character, unknown_bone, rig_not_owned, rig_edit_requires_unskinned, rig_edit_requires_no_clips, rig_edit_character_locked, rig_revision_overflow, scene_locked. Cross-rig references return rig_bone_external_reference. Authored ASCII name, not the prefixed key. Updates canonical hierarchy and BoneData keys/parent references, selects renamed bone and rebuilds skeleton/ozz. Errors: invalid_bone_name, bone_name_conflict, rig_edit_no_change. Anatomy references are preserved/remapped. Reparent may return rig_anatomy_chain_disconnected; delete may return rig_bone_anatomy_referenced until metadata references are removed. Unskinned staging also scans actual canonical flat scene weight buffers for references to this rig hierarchy indices; references block with rig_edit_requires_unskinned even if weighted metadata is missing, including zero/invalid entries and extra rows. No weighted delete/remap/transfer is implemented.",
    "SceneWrite", "SceneWrite", true, "{ok: true}",
    "rig|rename|bone",
    "rig.list_bones|rig.set_mode|rig.rename_bone|rig.reparent_bone|rig.delete_bone",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_rename_bone, 3,
    true
};
static const MethodRegistration reg_rig_rename_bone(desc_rig_rename_bone);

static const MethodParam params_rig_reparent_bone[] = {
    {"character", "any", false, "Owned rig name", nullptr, nullptr},
    {"bone", "any", false, "Existing prefixed bone key", nullptr, nullptr},
    {"parent", "any", false, "Existing prefixed parent bone key in the same rig", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_reparent_bone = {
    "rig.reparent_bone", "rig",
    "Change an owned bone parent while preserving world rest pose",
    "Owned, unskinned, clip-free rigs only. Uses one shared staged core and undo command; other rig selection/edit is blocked in scoped Rig Edit. Shared errors: unknown_character, unknown_bone, rig_not_owned, rig_edit_requires_unskinned, rig_edit_requires_no_clips, rig_edit_character_locked, rig_revision_overflow, scene_locked. Cross-rig references return rig_bone_external_reference. Parent must belong to the same rig. Selected bone and descendants retain world rest pose; local rest is recomputed. Root moves, self-parenting and descendant cycles are rejected. Errors: unknown_parent_bone, rig_root_edit_blocked, rig_parent_cycle, rig_edit_no_change. Anatomy references are preserved/remapped. Reparent may return rig_anatomy_chain_disconnected; delete may return rig_bone_anatomy_referenced until metadata references are removed. Unskinned staging also scans actual canonical flat scene weight buffers for references to this rig hierarchy indices; references block with rig_edit_requires_unskinned even if weighted metadata is missing, including zero/invalid entries and extra rows. No weighted delete/remap/transfer is implemented.",
    "SceneWrite", "SceneWrite", true, "{ok: true}",
    "rig|reparent|bone",
    "rig.list_bones|rig.set_mode|rig.rename_bone|rig.reparent_bone|rig.delete_bone",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_reparent_bone, 3,
    true
};
static const MethodRegistration reg_rig_reparent_bone(desc_rig_reparent_bone);

static const MethodParam params_rig_select_bone[] = {
    {"character", "any", false, "Import name from rig.list_characters", nullptr, nullptr},
    {"bone", "any", false, "Node name from rig.list_bones; helper nodes are also selectable", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_select_bone = {
    "rig.select_bone", "rig",
    "Select one named skeleton node in the shared hierarchy and viewport state",
    "Works with skeleton-only and meshless animated imports. Selection/overlay are transient view state, not authoring edits; no undo. Names are import-prefixed. Joint world transforms are row-major 16 floats from evaluated hierarchy globals, never skin-offset matrices. pose_source is bind, graph, controller or ozz. Bind fallback is explicit per node. Independent skeleton root-motion placement and bind/pose editing are subsequent roadmap work. Replaces multi-selection with one active bone through shared RigSelection; no additive behavior. Successful selection releases the selected IK handle and discards its uncommitted preview; committed IK controls, timed channels and contacts remain active. Direct FK edits of IK-driven bones still require returning their control to FK.",
    "write", "SceneWrite", false, "ok",
    "rig|select|bone|skeleton|joint|meshless|overlay",
    "rig.list_characters|rig.list_bones|rig.select_bone|rig.get_selected_bone",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_select_bone, 2,
    true
};
static const MethodRegistration reg_rig_select_bone(desc_rig_select_bone);

static const MethodParam params_rig_select_bones[] = {
    {"bones", "any", false, "Array of unique bone name strings", nullptr, nullptr},
    {"active", "string", false, "Active member or empty for automatic choice", "", nullptr},
    {"mode", "any", false, "", nullptr, "['replace', 'add', 'toggle', 'range']"},
    {"anchor", "any", false, "Optional anchor member override for restoring a selection snapshot; empty uses click/range behavior", nullptr, nullptr},
    {"character", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_select_bones = {
    "rig.select_bones", "rig",
    "Select multiple unique joints of one rig with an active joint",
    "Shared RigSelection service and transient scene ViewState, no geometry edit/history/project dirty mark. Unique bone names from rig.list_bones; max 4096. Modes replace/add/toggle/range; add/toggle on another character start a new single-rig set (scoped Edit rejects other character). Duplicate/unknown bone requests fail atomically. Optional active must belong to resulting set; otherwise default last requested or existing active/last canonical selected when toggled off. Range requires exactly one target; inclusive full depth-first hierarchy order including collapsed branches, from stable selection anchor. Output selection enumeration follows canonical skeleton order. Empty replace clears selection; old rig.select_bone replaces with one active bone. Clear selection clears the complete set. UI Ctrl toggle, Shift range, Rig Edit Shift-blank drag box replaces; Ctrl+Shift box adds. Active orange, other selected green, outgoing segment identity remains its starting joint. Errors rig_selection_invalid_mode, rig_selection_duplicate_bone, rig_selection_active_not_selected, rig_selection_invalid_range, rig_selection_invalid_hierarchy, rig_selection_limit, unknown_bone, unknown_character, rig_edit_character_locked, api_not_bound, scene_locked, rig_selection_failed. Python rt.rig.select_bones(character,bones,active=\"\",mode=\"replace\"). Optional anchor override must be a resulting member (rig_selection_anchor_not_selected); read rig.get_selection.anchor for exact selection restoration. Successful selection releases the selected IK handle and discards its uncommitted preview; committed IK controls, timed channels and contacts remain active. Direct FK edits of IK-driven bones still require returning their control to FK.",
    "SceneWrite", "SceneWrite", false, "{ok: bool}",
    "rig|select|bones",
    "rig.get_selection|rig.select_bone|rig.set_selection_pivot|rig.transform_rest",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_select_bones, 5,
    true
};
static const MethodRegistration reg_rig_select_bones(desc_rig_select_bones);

static const MethodParam params_rig_select_control[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"control", "string", true, "", nullptr, nullptr},
    {"handle", "string", false, "", "target", nullptr},
};
static const MethodDescriptor desc_rig_select_control = {
    "rig.select_control", "rig",
    "Select an IK target, pole, aim roll, orientation or spline handle",
    "Active Pose, no render job or pending preview. Transient selection, no undo/project modification. Empty control returns to bone FK gizmo. Handle is target or pole (Translate), orientation (Rotate), or spline_0/spline_1 (Translate, chain controls only). For aim controls target sets direction and pole is the roll-up handle; independent orientation is unsupported. Errors: rig_pose_mode_required, rig_pose_preview_active, rig_ik_unknown_control, rig_ik_invalid_handle. Successful bone selection returns to the FK bone gizmo, discards an uncommitted IK preview and preserves committed IK/contacts. Re-select a control to resume its target gizmo.",
    "write", "SceneWrite", false, "ok",
    "rig|select|control|pose|IK|selection|viewport",
    "rig.get_controls|rig.set_ik_target|rig.select_bone",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_select_control, 3,
    true
};
static const MethodRegistration reg_rig_select_control(desc_rig_select_control);

static const MethodParam params_rig_select_pose_clip[] = {
    {"character", "any", false, "", nullptr, nullptr},
    {"clip", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_select_pose_clip = {
    "rig.select_pose_clip", "rig",
    "Select an editable native clip and clear transient pose overrides",
    "7A/8A: requires an owned rig with a bound flat mesh; imported rigs remain read-only for pose authoring. Mutations require active pose mode and no render job. Rest transforms and weights are never modified. Frame time uses the scene FPS; create_pose_clip.fps controls clip ticks per second. Matrices are row-major 16 numbers, rigid translation/rotation only. Preview replaces the previous preview relative to the committed pose; frame changes and mode exit discard unkeyed overrides. Keys persist in native projects; clip selection and Auto Key are transient. Unweighted vertices retain their bind position.",
    "write", "SceneWrite", false, "ok",
    "rig|select|pose|clip|FK|bone keys|authoring",
    "rig.set_mode|rig.get_pose_state|rig.preview_pose_locals|rig.apply_pose_preview",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_select_pose_clip, 2,
    true
};
static const MethodRegistration reg_rig_select_pose_clip(desc_rig_select_pose_clip);

static const MethodParam params_rig_set_anatomy[] = {
    {"anatomy", "any", false, "Full version 1 canonical anatomy object; use rig.get_anatomy to read/edit existing values", nullptr, nullptr},
    {"character", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_set_anatomy = {
    "rig.set_anatomy", "rig",
    "Replace a rig anatomy definition through validated shared authoring core",
    "One undo command; native project persistence stores anatomy beside canonical hierarchy. Full version 1 schema required, unknown fields rejected. Unique role identifiers, existing bone keys, disjoint distinct symmetry pairs and unique contiguous base-to-tip parent chains (2..4096 joints) required. Max 4096 roles/pairs and 1024 chains. IDs are 1..128 ASCII letters/digits/underscore/hyphen/dot. Family is custom/humanoid/quadruped/insect/avian. Owned unskinned clip-free eligibility and scoped edit lock apply. Rename rekeys anatomy; reparent that disconnects a chain is rejected; delete of an anatomy-referenced bone requires removing those references first. Copy remaps anatomy to new keys. No IK, fitting or limits; basic templates can generate this metadata. Errors: rig_anatomy_invalid_schema, rig_anatomy_invalid_family, rig_anatomy_invalid_role, rig_anatomy_duplicate_role, rig_anatomy_unknown_bone, rig_anatomy_invalid_symmetry, rig_anatomy_invalid_chain_name, rig_anatomy_invalid_chain_length, rig_anatomy_duplicate_chain_bone, rig_anatomy_chain_disconnected, rig_anatomy_limit, rig_edit_no_change and shared eligibility/scene errors.",
    "SceneWrite", "SceneWrite", true, "{ok: true}",
    "rig|set|anatomy",
    "rig.get_anatomy|rig.rename_bone|rig.reparent_bone|rig.delete_bone|rig.copy_from",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_anatomy, 2,
    true
};
static const MethodRegistration reg_rig_set_anatomy(desc_rig_set_anatomy);

static const MethodParam params_rig_set_envelope_overlay[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"visible", "boolean", true, "", nullptr, nullptr},
    {"torso_radius", "float", false, "", "0.16", nullptr},
    {"limb_radius", "float", false, "", "0.065", nullptr},
    {"extremity_radius", "float", false, "", "0.05", nullptr},
    {"falloff", "float", false, "", "2.0", nullptr},
};
static const MethodDescriptor desc_rig_set_envelope_overlay = {
    "rig.set_envelope_overlay", "rig",
    "Show or hide anatomical influence capsules in the viewport",
    "Transient display control shared with Edit Bone UI. When visible=true, validates an authored bound rig and draws every parent-owned capsule: cyan for ordinary segments and amber for the active bone. It follows the current displayed pose, performs no weight mutation and creates no undo entry. Settings use the same ranges as preview_envelope_weights.",
    "write", "SceneWrite", false, "ok",
    "rig|set|envelope|overlay|weights|viewport|display",
    "rig.get_envelope_overlay|rig.preview_envelope_weights|rig.get_binding",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_envelope_overlay, 6,
    true
};
static const MethodRegistration reg_rig_set_envelope_overlay(desc_rig_set_envelope_overlay);

static const MethodParam params_rig_set_ik_contact[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"rig_revision", "integer", true, "", nullptr, nullptr},
    {"control", "string", true, "", nullptr, nullptr},
    {"enabled", "bool", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_set_ik_contact = {
    "rig.set_ik_contact", "rig",
    "Preview a world-position contact captured from the achieved limb pose",
    "Active owned bound Pose, no render job, matching revision. true captures achieved tip position/current pole, enables full IK and pins position across frame scrub/root FK edits. false releases the scrub pin, retaining current-frame IK; blend=0 returns to FK with pose preserved. Apply creates one undo step; Auto Key writes control channels for IK edits and undriven FK bone channels; use bake for solved IK bones. Optional tip orientation is controlled by set_ik_orientation and recaptured with the contact. No inferred ground; timed contacts use set_ik_contact_interval. Contacts clear on mode exit/clip selection; key controls or bake solved bones for native persistence. Inspect residual and limit_hits.",
    "write", "SceneWrite", false, "ok; transient pose preview",
    "rig|set|ik|contact|pose|IK|plant",
    "rig.get_controls|rig.set_ik_target|rig.apply_pose_preview|rig.cancel_pose_preview|rig.insert_pose_keys",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_ik_contact, 4,
    true
};
static const MethodRegistration reg_rig_set_ik_contact(desc_rig_set_ik_contact);

static const MethodParam params_rig_set_ik_contact_interval[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"rig_revision", "integer", true, "", nullptr, nullptr},
    {"start_frame", "integer", true, "", nullptr, nullptr},
    {"end_frame", "integer", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_set_ik_contact_interval = {
    "rig.set_ik_contact_interval", "rig",
    "Capture the current achieved limb pose for a timed contact",
    "Same mutation guards as insert_ik_key. Frames become seconds using Pose fps. Captures CURRENT achieved world tip and pole, optional achieved orientation; full IK, half-open [start_frame,end_frame). Exact same interval replaces capture; overlapping intervals fail rig_ik_contact_overlap; adjacent intervals valid. End must exceed start. Outside interval keyed controls resume, otherwise FK. One undo step and native persistence; clears limb live pin. Errors include rig_ik_invalid_contact_interval.",
    "write", "SceneWrite", false, "ok",
    "rig|set|ik|contact|interval|pose|IK|timeline|bake",
    "rig.get_ik_channels|rig.insert_ik_key|rig.set_ik_contact_interval|rig.bake_ik_channels",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_ik_contact_interval, 4,
    true
};
static const MethodRegistration reg_rig_set_ik_contact_interval(desc_rig_set_ik_contact_interval);

static const MethodParam params_rig_set_ik_fk[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"rig_revision", "integer", true, "", nullptr, nullptr},
    {"control", "string", true, "", nullptr, nullptr},
    {"blend", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_set_ik_fk = {
    "rig.set_ik_fk", "rig",
    "Preview IK/FK blend with matched activation and pose-preserving FK return",
    "Active owned bound Pose, no render job, matching revision. Finite blend 0..1. Activation matches current target/pole, avoiding positional snap. 0 disables IK/contact and copies achieved limb rotations into FK input. Local quaternions blend once. Apply/cancel/Auto Key use common pose transaction. Direct FK edits of driven joints fail rig_ik_bone_driven; un-driven pelvis/root remain editable for world contacts. Pose mirror requires FK (rig_ik_switch_to_fk). Errors include rig_ik_invalid_blend, rig_ik_unknown_control and rig_edit_stale_revision.",
    "write", "SceneWrite", false, "ok; transient pose preview",
    "rig|set|ik|fk|pose|IK|FK|blend",
    "rig.get_controls|rig.set_ik_target|rig.apply_pose_preview|rig.cancel_pose_preview",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_ik_fk, 4,
    true
};
static const MethodRegistration reg_rig_set_ik_fk(desc_rig_set_ik_fk);

static const MethodParam params_rig_set_ik_orientation[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"rig_revision", "integer", true, "", nullptr, nullptr},
    {"control", "string", true, "", nullptr, nullptr},
    {"orientation_world", "array", true, "", nullptr, nullptr},
    {"enabled", "bool", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_set_ik_orientation = {
    "rig.set_ik_orientation", "rig",
    "Preview a world-space hand/foot tip orientation",
    "Active owned bound Pose, matching revision, no render job. orientation_world is a finite unit quaternion [w,x,y,z]; enabled=true activates IK. Rotation blends once with IK/FK blend after positional solving and before joint limits. Apply creates one undo step; Auto Key writes IK control keys; bake writes solved tip bone keys. Live orientation is transient; position contacts retain it while scrubbing. No timed control channels yet. Errors include rig_ik_invalid_orientation, rig_edit_stale_revision, rig_ik_unknown_control.",
    "write", "SceneWrite", false, "ok; transient pose preview",
    "rig|set|ik|orientation|pose|IK",
    "rig.get_controls|rig.select_control|rig.apply_pose_preview|rig.cancel_pose_preview",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_ik_orientation, 5,
    true
};
static const MethodRegistration reg_rig_set_ik_orientation(desc_rig_set_ik_orientation);

static const MethodParam params_rig_set_ik_spline[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"rig_revision", "integer", true, "", nullptr, nullptr},
    {"control", "string", true, "", nullptr, nullptr},
    {"points_world", "array", true, "Two [x,y,z] interior world points; empty only to disable/clear shape.", nullptr, nullptr},
    {"enabled", "bool", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_set_ik_spline = {
    "rig.set_ik_spline", "rig",
    "Preview a cubic spline guide for a multi-joint IK control",
    "Active owned bound Pose, no render job, matching rig_revision. Requires a multi-joint chain control. points_world contains exactly two finite world-space cubic Bezier interior points; [] allowed only when enabled=false. Root is the live chain anchor, end is target_world. Enabling activates IK and uses full blend if it was zero. A 257-sample curve seeds arc-length chain stations, then bounded FABRIK preserves bone lengths and anchor and blends once before joint limits. Shape is a guide, not exact curve adherence. Unreachable endpoints straighten to reach; collapsed curves fail rig_ik_degenerate_spline when solving. Default handles approximate current end tangents, so review activation preview. Apply creates one undo step; Auto Key and insert_ik_key store spline state/points in version 2 IK channels. Existing version 1 channels load unchanged. Contacts retain curve points; bake writes solved bones. Errors: rig_ik_invalid_spline, rig_ik_spline_requires_chain, rig_ik_degenerate_spline, rig_edit_stale_revision.",
    "write", "SceneWrite", false, "ok; transient pose preview",
    "rig|set|ik|spline|pose|IK|shape|spine|neck|tail|bezier",
    "rig.get_controls|rig.select_control|rig.insert_ik_key|rig.apply_pose_preview|rig.cancel_pose_preview|rig.bake_ik_channels",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_ik_spline, 5,
    true
};
static const MethodRegistration reg_rig_set_ik_spline(desc_rig_set_ik_spline);

static const MethodParam params_rig_set_ik_target[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"rig_revision", "integer", true, "", nullptr, nullptr},
    {"control", "string", true, "", nullptr, nullptr},
    {"target_world", "vec3", true, "", nullptr, nullptr},
    {"pole_world", "vec3", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_set_ik_target = {
    "rig.set_ik_target", "rig",
    "Preview a limb target and bend pole in world space",
    "Active owned bound Pose, no render job, matching revision. target_world/pole_world are finite 3-number world points. First activation matches the current pose, enables IK. Replaceable preview: FK input -> parent-first limb solve -> joint rules -> skin. Blends once, preserves segment lengths/translations. Unreachable targets clamp to reach; collinear pole falls back to current bend then a deterministic perpendicular. Inspect get_controls.target_error_world and get_pose_state.limit_hits. Apply creates one undo step; Auto Key writes IK control channels and undriven FK bones; Cancel discards preview. Noncontact handles reset on frame change; contacts survive scrub during Pose. Native files store definitions, bone keys and explicit IK channels; unkeyed live targets remain transient. Errors include rig_ik_invalid_target, rig_ik_unknown_control, rig_ik_invalid_placement and rig_edit_stale_revision.",
    "write", "SceneWrite", false, "ok; transient pose preview",
    "rig|set|ik|target|pose|IK|pole",
    "rig.get_controls|rig.apply_pose_preview|rig.cancel_pose_preview|rig.set_pose_auto_key",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_ik_target, 5,
    true
};
static const MethodRegistration reg_rig_set_ik_target(desc_rig_set_ik_target);

static const MethodParam params_rig_set_joint_limit_overlay[] = {
    {"visible", "bool", true, "", nullptr, nullptr},
    {"edit", "bool", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_set_joint_limit_overlay = {
    "rig.set_joint_limit_overlay", "rig",
    "Show selected joint axes/limits and enable viewport limit handles",
    "Transient flags; no project modification or history entry. edit=true requires visible=true, no render job and no pose preview; pauses playback. Edit mode gives angular boundary handles priority over FK/IK/actor gizmos. Viewport handles preview bounds visually; release calls set_joint_limits once, Escape discards. Unknown/unsupported rule or imported rig remains display-only. Rule activation/axis changes use existing joint-profile service. Errors: rig_joint_overlay_hidden, rig_pose_preview_active, scene_locked. No separate UI business logic or persistent preview state.",
    "write", "SceneWrite", false, "ok",
    "rig|set|joint|limit|overlay|limits|viewport|authoring",
    "rig.get_joint_limit_overlay|rig.get_joint_limit_view|rig.set_joint_limits",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_joint_limit_overlay, 2,
    true
};
static const MethodRegistration reg_rig_set_joint_limit_overlay(desc_rig_set_joint_limit_overlay);

static const MethodParam params_rig_set_joint_limits[] = {
    {"character", "string", true, "", nullptr, nullptr},
    {"bone", "string", true, "", nullptr, nullptr},
    {"minimum", "float", true, "", nullptr, nullptr},
    {"maximum", "float", true, "", nullptr, nullptr},
    {"swing", "float", true, "", nullptr, nullptr},
    {"rig_revision", "integer", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_set_joint_limits = {
    "rig.set_joint_limits", "rig",
    "Atomically update an existing hinge/ball joint's angular limits",
    "Owned rig, no render job or outstanding pose preview, matching nonnegative rig_revision. Existing hinge/ball rule required. Finite degrees: minimum -180..0, maximum 0..180, swing 0..180. Hinge does not use swing for evaluation but stores the validated field. Axis, enabled and lock_translation remain unchanged. Same core/profile history transaction as inspector settings, one undo step; limits persist in native anatomy, evaluator invalidates once. Re-query revision after success. Errors include rig_not_owned, rig_edit_stale_revision, rig_joint_rule_required, rig_joint_limits_not_applicable, rig_joint_invalid_range, rig_joint_no_change, rig_pose_preview_active and scene_locked. Rest/bind/weights are not modified.",
    "write", "SceneWrite", false, "ok",
    "rig|set|joint|limits|hinge|ball|authoring|undo",
    "rig.get_joint_limit_view|rig.get_joint_profile|rig.set_joint_profile",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_joint_limits, 6,
    true
};
static const MethodRegistration reg_rig_set_joint_limits(desc_rig_set_joint_limits);

static const MethodParam params_rig_set_joint_profile[] = {
    {"character", "any", false, "Exact character name from rig.list_characters.", nullptr, nullptr},
    {"profile", "any", false, "Complete version 1 profile; omitted existing rows are removed. Unknown fields/bones, duplicate rows, nonfinite numbers, nonunit axes and invalid ranges reject atomically.", nullptr, nullptr},
    {"rig_revision", "any", false, "Current nonnegative revision from get_joint_profile; stale writes reject.", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_set_joint_profile = {
    "rig.set_joint_profile", "rig",
    "Replace persistent joint rules atomically with validation and undo",
    "Profile schema version 1: joints array of bone/type/enabled/lock_translation/axis/minimum/maximum/swing. Defaults: free, disabled, lock_translation true, unit axis [1,0,0], twist range [-180,180], swing 180 degrees. Types free/hinge/ball/fixed. Axis is in the joint rest-relative local frame. Ranges must contain neutral zero; no stretch or anatomical certainty is implied. Enabled rules project previews and authored playback; Auto Key records the constrained pose. A fully blocked changed request applies as a successful no-op with no keys/history. Read get_pose_state.limit_hits for projected bones. Queries are read-only; suggestions remain disabled. Set requires owned rig and current nonnegative rig_revision, rejects active Pose preview/render job, preserves rest/weights/clip keys, increments rig revision and is undoable. Native anatomy uses version 2 when rules exist, version 1 otherwise. IK integration, soft limits, elliptical cones and muscle deformation remain later work.",
    "write", "SceneWrite", true, "ok",
    "rig|set|joint|profile|anatomy|limits|hinge|swing twist|constraints",
    "rig.get_joint_profile|rig.suggest_joint_profile|rig.set_joint_profile|rig.get_pose_state|rig.preview_pose_locals",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_joint_profile, 3,
    true
};
static const MethodRegistration reg_rig_set_joint_profile(desc_rig_set_joint_profile);

static const MethodParam params_rig_set_mode[] = {
    {"mode", "any", false, "", nullptr, "['scene', 'edit', 'pose']"},
    {"character", "string", false, "Owned rig required when mode is edit", "", nullptr},
};
static const MethodDescriptor desc_rig_set_mode = {
    "rig.set_mode", "rig",
    "Set Scene or scoped Rig Edit viewport interaction mode",
    "Transient, not serialized or undoable. Edit requires an owned unskinned clip-free rig and no active mesh/sculpt/paint edit session. Exit to scene before switching rigs. Errors: unsupported_rig_mode, viewport_edit_mode_conflict, rig_edit_character_locked, rig_not_owned, rig_edit_requires_unskinned, rig_edit_requires_no_clips, unknown_character, scene_locked. Entering Edit also rejects actual flat skin references to owned rig indices even when weighted metadata is missing; rig_edit_requires_unskinned. Pose mode is supported for owned bound rigs. Scene exits Pose and clears transient overrides; Edit requires exiting Pose first.",
    "SceneWrite", "SceneWrite", false, "any",
    "rig|set|mode",
    "rig.get_mode|rig.select_bone|rig.set_rest_transform",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_mode, 2,
    true
};
static const MethodRegistration reg_rig_set_mode(desc_rig_set_mode);

static const MethodParam params_rig_set_overlay_visible[] = {
    {"visible", "any", false, "Overlay visibility for all visible skeletons", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_set_overlay_visible = {
    "rig.set_overlay_visible", "rig",
    "Show or hide the skeleton viewport overlay",
    "Works with skeleton-only and meshless animated imports. Selection/overlay are transient view state, not authoring edits; no undo. Names are import-prefixed. Joint world transforms are row-major 16 floats from evaluated hierarchy globals, never skin-offset matrices. pose_source is bind, graph, controller or ozz. Bind fallback is explicit per node. Independent skeleton root-motion placement and bind/pose editing are subsequent roadmap work.",
    "write", "SceneWrite", false, "ok",
    "rig|set|overlay|visible|skeleton|joint|meshless",
    "rig.list_characters|rig.list_bones|rig.select_bone|rig.get_selected_bone",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_overlay_visible, 1,
    true
};
static const MethodRegistration reg_rig_set_overlay_visible(desc_rig_set_overlay_visible);

static const MethodParam params_rig_set_pose_auto_key[] = {
    {"enabled", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_set_pose_auto_key = {
    "rig.set_pose_auto_key", "rig",
    "Enable or disable key insertion when a pose preview is applied",
    "7A/8A: requires an owned rig with a bound flat mesh; imported rigs remain read-only for pose authoring. Mutations require active pose mode and no render job. Rest transforms and weights are never modified. Frame time uses the scene FPS; create_pose_clip.fps controls clip ticks per second. Matrices are row-major 16 numbers, rigid translation/rotation only. Preview replaces the previous preview relative to the committed pose; frame changes and mode exit discard unkeyed overrides. Keys persist in native projects; clip selection and Auto Key are transient. Unweighted vertices retain their bind position.",
    "write", "SceneWrite", false, "ok",
    "rig|set|pose|auto|key|FK|bone keys|authoring",
    "rig.set_mode|rig.get_pose_state|rig.preview_pose_locals|rig.apply_pose_preview",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_pose_auto_key, 1,
    true
};
static const MethodRegistration reg_rig_set_pose_auto_key(desc_rig_set_pose_auto_key);

static const MethodParam params_rig_set_pose_frame[] = {
    {"frame", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_set_pose_frame = {
    "rig.set_pose_frame", "rig",
    "Pause playback and move to an integer authoring frame",
    "7A/8A: requires an owned rig with a bound flat mesh; imported rigs remain read-only for pose authoring. Mutations require active pose mode and no render job. Rest transforms and weights are never modified. Frame time uses the scene FPS; create_pose_clip.fps controls clip ticks per second. Matrices are row-major 16 numbers, rigid translation/rotation only. Preview replaces the previous preview relative to the committed pose; frame changes and mode exit discard unkeyed overrides. Keys persist in native projects; clip selection and Auto Key are transient. Unweighted vertices retain their bind position.",
    "write", "SceneWrite", false, "ok",
    "rig|set|pose|frame|FK|bone keys|authoring",
    "rig.set_mode|rig.get_pose_state|rig.preview_pose_locals|rig.apply_pose_preview",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_pose_frame, 1,
    true
};
static const MethodRegistration reg_rig_set_pose_frame(desc_rig_set_pose_frame);

static const MethodParam params_rig_set_pose_view[] = {
    {"mode", "any", false, "", nullptr, "['rest', 'animated']"},
    {"character", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_set_pose_view = {
    "rig.set_pose_view", "rig",
    "Switch one rig between stored Rest and Animated evaluation views",
    "Transient, not serialized or undoable; does not alter clips or bind data. Rest skips graph/controller/ozz playback and root-motion evaluation for the chosen model. Computes skin matrices from canonical hierarchy rest globals and bind offsets; CPU skinning uses flat TriangleMesh only. Other models continue evaluating. Rig Edit overrides effective view to Rest. Animated resumes assigned clips and refreshes CPU/GPU display even when paused. Source must have canonical skeleton hierarchy for Rest. Errors: invalid_rig_pose_view, unknown_character, rig_pose_view_requires_hierarchy, rig_pose_view_incomplete_bind, rig_pose_view_invalid_bind, invalid_preview_hierarchy, invalid_preview_pose, scene_locked. On import/native reopen, valid skinned characters default to Rest; meshless clips retain prior animated defaults. Explicit per-character views are preserved during append/reinitialization. Rest graphs are excluded from autonomous viewport wake and file-animation reset scheduling. Select animated explicitly to resume existing animation.",
    "SceneWrite", "SceneWrite", false, "any",
    "rig|set|pose|view",
    "rig.get_pose_view|rig.set_mode|anim.sample_clip_binding",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_pose_view, 2,
    true
};
static const MethodRegistration reg_rig_set_pose_view(desc_rig_set_pose_view);

static const MethodParam params_rig_set_rest_transform[] = {
    {"character", "any", false, "Owned rig identity.", nullptr, nullptr},
    {"bone", "any", false, "Existing full unique joint key.", nullptr, nullptr},
    {"rest_transform", "any", false, "Row-major 16-number local rest transform. Translation and proper rotation only; scale/shear/reflection rejected.", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_set_rest_transform = {
    "rig.set_rest_transform", "rig",
    "Change an owned unskinned clip-free joint local rest position and rotation with undo",
    "Recomputes descendant rest globals/offsets and BoneData/hierarchy/skeletonNodes/ozz caches from canonical NodeHierarchy. Increments persisted revision. Rest transform must be finite and rigid; existing clips must be absent. Service failures return named codes (Python ValueError / IPC code): rig_name_conflict, invalid_rig_name/invalid_bone_name, unknown_rig_template, unknown_character, unknown_parent_bone/unknown_bone, bone_name_conflict, rig_not_owned, rig_edit_requires_unskinned, rig_edit_requires_no_clips, invalid_rest_transform, rig_rest_requires_rigid_transform, scene_locked or history_not_bound. Shape/type errors use Python TypeError/ValueError and IPC invalid_parameter. Validation/staging happens before scene changes. Unskinned staging also scans actual canonical flat scene weight buffers for references to this rig hierarchy indices; references block with rig_edit_requires_unskinned even if weighted metadata is missing, including zero/invalid entries and extra rows. No weighted delete/remap/transfer is implemented.",
    "SceneWrite", "SceneWrite", true, "{ok:true}",
    "rig|set|rest|transform",
    "rig.add_bone|rig.list_bones|project.save|project.open",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_rest_transform, 3,
    true
};
static const MethodRegistration reg_rig_set_rest_transform(desc_rig_set_rest_transform);

static const MethodParam params_rig_set_scene_transform[] = {
    {"scene_transform", "any", false, "16 finite row-major numbers describing translation, proper rotation and positive uniform scale (0.0001 to 10000)", nullptr, nullptr},
    {"character", "any", false, "Owned meshless rig name", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_set_scene_transform = {
    "rig.set_scene_transform", "rig",
    "Move, rotate or uniformly scale an entire owned meshless rig without editing its rest hierarchy",
    "Scene mode only. Shared service and undo command used by viewport, inspector, Python and IPC. Clips on meshless owned rigs are allowed; local rest, anatomy, bind offsets and clip curves remain unchanged. Native snapshots persist translation/rotation/uniform scale; legacy snapshots default to identity. Positive uniform actor scale in [0.0001, 10000] is supported. Nonuniform scale/shear/reflection are rejected; individual joint rest transforms remain rigid. Errors: unknown_character, rig_not_owned, rig_edit_active, rig_placement_requires_meshless_unskinned, invalid_rig_scene_transform, rig_scene_requires_uniform_scale, rig_revision_overflow, rig_edit_no_change, scene_locked. Placement staging rejects actual flat weight references to owned rig hierarchy indices even if metadata is missing; rig_placement_requires_meshless_unskinned. Per-frame capability queries do not scan all skin buffers.",
    "SceneWrite", "SceneWrite", true, "{ok: true}",
    "rig|set|scene|transform",
    "rig.get_scene_transform|rig.set_rest_transform",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_scene_transform, 2,
    true
};
static const MethodRegistration reg_rig_set_scene_transform(desc_rig_set_scene_transform);

static const MethodParam params_rig_set_selection_pivot[] = {
    {"mode", "any", false, "", nullptr, "['active', 'center']"},
};
static const MethodDescriptor desc_rig_set_selection_pivot = {
    "rig.set_selection_pivot", "rig",
    "Choose active-joint or selection-center pivot for batch rest gizmos",
    "Transient shared view preference, defaults active. center is arithmetic mean of selected world joint origins; orientation remains active joint basis, local/world axes remain existing viewport choice. Used for rest gizmo in scoped Rig Edit; Scene G/R/S still transforms the whole owned meshless actor. Pivot changes during drag cancel pending rest preview. No geometry/history/project dirty mark, not serialized. Errors rig_selection_invalid_pivot, api_not_bound, scene_locked. Python rt.rig.set_selection_pivot(mode).",
    "SceneWrite", "SceneWrite", false, "{ok: bool}",
    "rig|set|selection|pivot",
    "rig.get_selection|rig.transform_rest",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_selection_pivot, 1,
    true
};
static const MethodRegistration reg_rig_set_selection_pivot(desc_rig_set_selection_pivot);

static const MethodParam params_rig_set_weight_map_visible[] = {
    {"visible", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_set_weight_map_visible = {
    "rig.set_weight_map_visible", "rig",
    "Toggle transient viewport tint for the currently selected bone weights",
    "Same shared API as inspector checkbox. Transient ViewState flag, defaults off, no scene geometry/skin/sculpt protection mask change, no history or project dirty mark; not persisted. Selection changes automatically change displayed bone across owned/imported weighted flat parts. Missing/ambiguous ownership is not guessed. Reuses sculpt scalar triangle tint primitive, reads current deformed flat P/N and object transform. ImGui surface overlay works across viewport modes but has no depth buffer (backface culling only); overlapping surfaces can show through. Bounded display (2M vertices and 120k examined faces) may omit dense parts. No weight painting delivered. Errors api_not_bound, scene_locked; IPC expects boolean visible, invalid_parameter on wrong shape. Python rt.rig.set_weight_map_visible(visible).",
    "SceneWrite", "SceneWrite", false, "{ok: bool}",
    "rig|set|weight|map|visible",
    "rig.get_weight_map_visible|rig.get_weight_map|rig.select_bone",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_set_weight_map_visible, 1,
    true
};
static const MethodRegistration reg_rig_set_weight_map_visible(desc_rig_set_weight_map_visible);

static const MethodParam params_rig_suggest_joint_profile[] = {
    {"character", "any", false, "Exact character name from rig.list_characters.", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_suggest_joint_profile = {
    "rig.suggest_joint_profile", "rig",
    "Propose disabled limb joint rules from explicit anatomy roles and rest geometry",
    "Profile schema version 1: joints array of bone/type/enabled/lock_translation/axis/minimum/maximum/swing. Defaults: free, disabled, lock_translation true, unit axis [1,0,0], twist range [-180,180], swing 180 degrees. Types free/hinge/ball/fixed. Axis is in the joint rest-relative local frame. Ranges must contain neutral zero; no stretch or anatomical certainty is implied. Enabled rules project previews and authored playback; Auto Key records the constrained pose. A fully blocked changed request applies as a successful no-op with no keys/history. Read get_pose_state.limit_hits for projected bones. Queries are read-only; suggestions remain disabled. Set requires owned rig and current nonnegative rig_revision, rejects active Pose preview/render job, preserves rest/weights/clip keys, increments rig revision and is undoable. Native anatomy uses version 2 when rules exist, version 1 otherwise. IK integration, soft limits, elliptical cones and muscle deformation remain later work.",
    "read", "Read", false, "JointProfileProposal",
    "rig|suggest|joint|profile|anatomy|limits|hinge|swing twist|constraints",
    "rig.get_joint_profile|rig.suggest_joint_profile|rig.set_joint_profile|rig.get_pose_state|rig.preview_pose_locals",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_suggest_joint_profile, 1,
    true
};
static const MethodRegistration reg_rig_suggest_joint_profile(desc_rig_suggest_joint_profile);

static const MethodParam params_rig_transform_rest[] = {
    {"bones", "any", false, "Unique selected joint keys, >=1 and <=4096", nullptr, nullptr},
    {"world_delta", "any", false, "16 row-major finite numbers; rigid world transform about caller-chosen pivot", nullptr, nullptr},
    {"rig_revision", "any", false, "Expected nonnegative uint64 rig revision from rig.list_bones", nullptr, nullptr},
    {"character", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_transform_rest = {
    "rig.transform_rest", "rig",
    "Apply one rigid world delta to multiple owned unskinned joints as one undo step",
    "Explicit bone names and expected rig_revision from list_bones required. Owned meshless/unskinned/clip-free rest edit gate, including scan of actual flat skin references. Does not require UI Edit but respects another scoped rig edit lock. World_delta is row-major 16 numeric floats and must be finite rigid affine (no scale/reflection). Pure RigBatchRest helper conjugates delta through actor placement, computes all selected original global targets, then derives locals relative to updated parent globals; selected parent+child do not move twice, unselected descendants inherit hierarchy movement. Uses canonical owned NodeHierarchy; common RigEditing::finish rebuilds BoneData/default/parent/index/inverse rest offset, skeletonNodes, ozz/runtime and revision. Stable scene bone IDs; no flat geometry/skin change allowed. Existing RigEditCommand snapshots bone/model plus complete selection for one-step undo/redo; explicit target list becomes selection, existing selected target active retained else last input. UI uses same helper for immutable preview and same API on release; Escape/camera absence/scene load/revision/selection/pivot/operation changes cancel. No-op rejects rig_edit_no_change, stale rig revision rig_edit_stale_revision; empty/duplicate/unknown list rig_selection_empty/rig_selection_duplicate_bone/unknown_bone; limits4096, invalid delta invalid_rest_transform or rig_rest_requires_rigid_transform, invalid parent rig_batch_rest_invalid_parent; existing rig/ownership/skin/clip/overflow gates propagate. api_not_bound/history_not_bound/scene_locked/rig_edit_failed. IPC bool/float/negative revision or invalid matrix/list shape invalid_parameter; Python typed conversion errors may precede service validation. Rest edits persist via native hierarchy/bone snapshot; transient selection/pivot do not. Mirror/batch topology/weighted rebind/pose authoring are not delivered. Python rt.rig.transform_rest(character,bones,world_delta,rig_revision). Existing valid range anchor is retained if it belongs to the target set.",
    "SceneWrite", "SceneWrite", true, "{ok: bool}",
    "rig|transform|rest",
    "rig.select_bones|rig.get_selection|rig.set_rest_transform|rig.list_bones",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_transform_rest, 4,
    true
};
static const MethodRegistration reg_rig_transform_rest(desc_rig_transform_rest);

static const MethodParam params_rig_weight_stats[] = {
    {"mesh", "any", false, "Exact individual flat mesh nodeName; groups not accepted", nullptr, nullptr},
};
static const MethodDescriptor desc_rig_weight_stats = {
    "rig.weight_stats", "rig",
    "Inspect stored flat mesh skin influence contract",
    "Read-only exact individual TriangleMesh nodeName; no model group or Triangle facade geometry. Existing contract fields retained: max_influences, min/max sums over nonempty rows (null when none), unweighted_vertices, invalid/nonpositive entries, duplicate IDs, descending-order violations, over-four rows, normalization deviations (1e-5), extra rows. contract_valid permits empty rows and checks numeric/top-four format only. fully_weighted is separate. Resolves mesh character through exact canonical nodeHierarchy unique keys or direct flat mesh membership, never authored-name/prefix guessing. ownership=resolved/unresolved/ambiguous; ownership_verified requires a unique owner with hierarchy keys. Checks scene BoneData forward index map, including missing/gap indices and duplicate index assignments; stale reverse caches are not consulted. unknown_bone_entries, ambiguous_bone_entries, foreign_bone_entries are independent; foreign counts require resolved hierarchy ownership. index_range_valid checks known unambiguous indices; bone_indices_verified additionally requires resolved ownership and no foreign entries. weights_valid combines contract_valid, fully_weighted, bone_indices_verified; it does not validate bind matrices or deformation quality. Extra rows fail contract_valid and are not sampled beyond actual DNA vertex count. No mutation/backend invalidation/history. Re-run after edits. Errors: invalid_mesh_name, unknown_mesh, ambiguous_mesh_name, mesh_has_no_geometry, rig_weight_stats_failed, api_not_bound, scene_locked. Python rt.rig.weight_stats(mesh). Initial authored binding registry takes precedence over original source hierarchy/membership; preserves correct owner after multipart binding and native reopen.",
    "Read", "Read", false, "SkinWeightStats",
    "rig|weight|stats",
    "rig.preflight|rig.get_weights",
    nullptr, nullptr, nullptr, nullptr,
    params_rig_weight_stats, 1,
    true
};
static const MethodRegistration reg_rig_weight_stats(desc_rig_weight_stats);

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

static const MethodDescriptor desc_scene_get_fbx_reader = {
    "scene.get_fbx_reader", "scene",
    nullptr,
    nullptr,
    "read", "Read", false, "any",
    "scene|get|fbx|reader",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    false
};
static const MethodRegistration reg_scene_get_fbx_reader(desc_scene_get_fbx_reader);

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

static const MethodParam params_scene_pick_gpu[] = {
    {"u", "number", true, "normalized X in [0,1]", nullptr, nullptr},
    {"v", "number", true, "normalized Y in [0,1], measured UP from the bottom", nullptr, nullptr},
};
static const MethodDescriptor desc_scene_pick_gpu = {
    "scene.pick_gpu", "scene",
    "Pick the object under a screen position using the GPU, reading what the raster viewport actually drew",
    "Immune by construction to the CPU-geometry failure class that scene.pick_ray measures: it draws an object-ID target with the SAME vertex and instance buffers, the SAME draw loop and the SAME (unjittered) viewProj the raster frame used, then reads back one pixel. u,v are normalized screen coordinates in [0,1] with v UP -- never pixels, so the caller never has to know the viewport resolution. The pass runs only on this call and touches nothing the normal frame uses; cost is one frame of vertex work with the raster clipped to a single pixel. Read the three result fields separately and do NOT collapse them: 'ok' says the call ran, 'hit' says there was geometry at that pixel (false is a legitimate answer, not a failure), and 'reason' explains a non-run. The result is resolved by IDENTITY, never by name: the shader writes (mesh slot, instance slot) and the backend converts that to an index into the raster instance list, reported as 'instance_index'. This matters because the OptiX GPU pick was switched off precisely for resolving id -> name -> selection cache, and that cache was stale. GPU culling is deliberately bypassed, since the compacted instance buffer makes gl_InstanceIndex unmappable back to a scene instance.",
    "read", "Read", false, "any",
    "scene|pick|gpu",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_scene_pick_gpu, 2,
    true
};
static const MethodRegistration reg_scene_pick_gpu(desc_scene_pick_gpu);

static const MethodParam params_scene_pick_ray[] = {
    {"u", "float", true, "", nullptr, nullptr},
    {"v", "float", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_scene_pick_ray = {
    "scene.pick_ray", "scene",
    "Diagnose viewport object picking at a screen position without changing the selection",
    "Answers 'why did the click select the wrong object'. Selection is now GPU-first: the click path calls scene.pick_gpu's backend (an ID render pass on the raster viewport device) for IDENTITY, then hits ONLY the named object on the CPU for the surface record and the distance. This method is the CPU half's instrument and stays meaningful: compare its linear_object against scene.pick_gpu at the same u,v, because a disagreement between them is now a disagreement between what is DRAWN and what would be SELECTED. The legacy OptiX pick buffer (getPickedObjectId, ensurePickBuffers, params.pick_buffer and its kernel writes) has been removed outright, so there is exactly one GPU pick path and it is the raster one this method is compared against. This method builds the camera ray from normalized screen coordinates (u,v in [0,1], v up -- NOT pixels, so the caller never has to know the current viewport resolution) and then asks the two picking paths SEPARATELY: the CPU BVH, and a linear scan over scene.world.objects that ignores the BVH entirely. Read 'paths_agree' first. FALSE means the BVH disagrees with brute force, which is what a stale BVH looks like: its AABB pruning drops a small object and returns a larger neighbour, and because the interactive path queries the BVH FIRST and only falls back to the scan when the BVH misses, the wrong answer wins and the scan that would have corrected it never runs. Both hits missing is NOT agreement and is reported as paths_agree false. 'world_objects' catches the other root: if it is small or zero the objects never reached world.objects at all (flat SoA seen only through the facade, or scatter not expanded on Vulkan) and no picking path can find them. 'ray_divergence_deg' separates a third cause -- the render ray carries lens distortion and a random aperture sample while the raster viewport draws a deterministic pinhole, so with depth_of_field on the two differ per click and small objects are missed at random; 'interactive_fallback_ray' says which one selection would actually have used. Then read 'linear_handles_agree'. The interactive selection tries rec.triangle (facade) FIRST and rec.tri_mesh (flat SoA) second, and resolves the facade branch through a tri_to_index lookup; in this repo geometry is always flat SoA and the facade is the legacy path, so when both handles are populated and they name different objects the hit is correct while the SELECTION is wrong -- the exact shape of 'a small object is picked but the wrong one gets selected'. Two empty handles are not agreement and are reported as false. Read 'gate_blocks_selection' BEFORE any of the above: the ray can be perfectly healthy while selection is dead, because the click never reaches the pick code. Measured 2026-09-16 -- selection stopped working entirely in a session while this method reported paths_agree true, divergence 0 and a correct hit; the user had to restart the app because the gate could not be asked. The gate fields are the STICKY ones, the ones that stay true and are only cleared by a scene or project reload, which is what 'reopening fixed it' actually means: rigView.edit_mode (rig edit owns the mouse and handleMouseSelection returns immediately), hud_captured_mouse (an overlay swallowed the click and the flag is only reset on the next click that is also swallowed) and is_dragging. Per-frame values like WantCaptureMouse and ImGuizmo::IsOver are deliberately NOT published: sampled outside an actual click they are meaningless, and reporting them as the gate would be an instrument describing something it did not measure. Unlike scene.raycast, which takes an origin and direction you already know, this one constructs the ray the viewport would and reports the disagreement.",
    "read", "Read", false, "any",
    "scene|pick|ray",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_scene_pick_ray, 2,
    true
};
static const MethodRegistration reg_scene_pick_ray(desc_scene_pick_ray);

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

static const MethodParam params_scene_set_fbx_reader[] = {
    {"reader", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_scene_set_fbx_reader = {
    "scene.set_fbx_reader", "scene",
    nullptr,
    nullptr,
    "write", "SceneWrite", false, "any",
    "scene|set|fbx|reader",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_scene_set_fbx_reader, 1,
    false
};
static const MethodRegistration reg_scene_set_fbx_reader(desc_scene_set_fbx_reader);

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

static const MethodDescriptor desc_viewport_automatic_cutout = {
    "viewport.automatic_cutout", "viewport",
    nullptr,
    nullptr,
    "render", "Render", false, "any",
    "viewport|automatic|cutout",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    false
};
static const MethodRegistration reg_viewport_automatic_cutout(desc_viewport_automatic_cutout);

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

static const MethodDescriptor desc_viewport_device_recovery_status = {
    "viewport.device_recovery_status", "viewport",
    "Whether the Vulkan viewport came back after a TDR, and why it has not yet.",
    "Answers 'after a TDR the viewport never comes back, not even if I switch modes by hand'. Two separate defects produced that, and the fields separate them. First: the device-lost handler tore the viewport backend down and then fell through, in the SAME loop iteration with no continue or return between them, to an unconditional initializeViewportBackendIfAvailable() -- opening a new VkDevice while the driver was still resetting. Second: that function begins with `if (g_viewport_backend) return true`, so a non-null pointer is taken as proof of a working device; once a poisoned backend was stored, every later attempt short-circuited and no manual mode switch could ever rebuild it. Recovery is now behind a time gate and the main loop retries on its own rather than waiting for the user to change backends. Read consecutive_losses against max_streak FIRST: automatic recovery is budgeted precisely because the loop it enables can be self-feeding -- if the raster work is what trips the TDR, rebuilding the viewport reproduces the TDR, and a time gate only slows that down rather than stopping it. When consecutive_losses reaches max_streak, given_up goes true and nothing is retried until a manual viewport mode change or viewport.retry_device_recovery re-arms it; given_up=true with viewport_alive=false is therefore a DELIBERATE stop, not a hang. The streak counts losses that follow a successful rebuild, not failed rebuild attempts, because those are different faults: a rebuild that fails means the device will not come back, while a rebuild that succeeds and is then lost again means the viewport work itself is killing it. The streak clears only after a rebuilt viewport has survived 30 seconds, so an unrelated TDR hours later is not charged to an old streak. Read viewport_alive with rebuild_pending, never alone: alive=true plus pending=true means a backend object exists but recovery has not been confirmed. gate_remaining_ms > 0 means we are deliberately waiting for the driver to reset and the absence of a rebuild is correct, not a hang. rebuild_attempts climbing while viewport_alive stays false means the device is genuinely refusing to come back, which is a different problem from never having tried.",
    "render", "Render", false, "any",
    "viewport|device|recovery|status",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_viewport_device_recovery_status(desc_viewport_device_recovery_status);

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

static const MethodDescriptor desc_viewport_frame_timings = {
    "viewport.frame_timings", "viewport",
    "Per-pass CPU and GPU cost of the raster viewport frame over the window since the last reset",
    "HOW TO USE IT: viewport.reset_frame_timings, then DRIVE THE CAMERA (camera.set_position/set_target) for as many frames as you want to average, then read this. The raster viewport renders ONLY when it is marked dirty, so a still camera produces an empty window - measured 2026-09-09, 24 s of a still camera gave 0 raster frames while the display loop kept re-presenting one old image, and a wall-clock FPS A/B over it correctly reported 'noise'. The app also draws NOTHING while its window is not focused. Neither is a bug; both make a measurement silently empty, and 'warnings' names them. WHAT THE STAGES MEAN: shadow_atlas, sky, depth_prepass, rt_shadow, main_pass, volume_sdf, transmission, post, overlay - in frame order. GPU marks are written at BOTTOM_OF_PIPE, so a stage is 'time until this boundary retired', NOT 'what this pass would cost alone'. They sum to the frame exactly; they do not attribute an isolated cost, and deleting a 10 ms stage does not promise a 10 ms frame. ZERO IS TWO DIFFERENT ANSWERS: 'frames_ran' 0 with 0 ms means the stage was SKIPPED every frame - which is a RESULT, not missing data. That is exactly what shadow_atlas reports once the RT shadow handoff takes the light (cross-check applied.rt_cascades_replaced). A stage that ran and cost nothing has frames_ran > 0. GPU NUMBERS LAG: marks are read without waiting, so the newest retired slot is one or two frames back and 'frames_with_gpu' trails 'frames'. Over a window that is irrelevant; on the first few frames after a reset it is why gpu_* can be absent. 'gpu_supported' false means this device or queue cannot timestamp at all - every gpu_* is ABSENT, not zero, and the cpu_* half is still real. 'applied' IS PART OF THE MEASUREMENT: it is what the measured frames ACTUALLY did, sampled from those same frames - never the request side. depth_prepass and gpu_culling in particular have both shipped here as requested-but-not-applied. If any of it changed DURING the window, 'warnings' says so and the means average two different machines: reset and measure again rather than comparing them. 'window_wall_ms' far exceeding frames x frame_cpu_mean_ms means these were individually driven frames, not a sustained frame rate - quote them as ms/frame, never as FPS.",
    "render", "Render", false, "any",
    "viewport|frame|timings|timing|profile|gpu|cpu|cost|measure|benchmark|stage|pass",
    "viewport.reset_frame_timings|viewport.frame_telemetry|viewport.rt_shadow",
    "viewport.reset_frame_timings", nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_viewport_frame_timings(desc_viewport_frame_timings);

static const MethodDescriptor desc_viewport_get_af = {
    "viewport.get_af", "viewport",
    "Report the AF grid state, its point count, and why it is inactive if it is",
    "'active' is not the setting: the overlay also needs the Camera HUD on, a camera, a scene BVH (AF measures distance by casting rays), and no sequence render in progress - each closed gate is named in 'inactive_reason'. 'point_count' is derived from area_mode (25 for Zone21, otherwise 9) so selected_point can be validated before writing it.",
    "render", "Render", false, "any",
    "viewport|get|af|camera|focus|autofocus",
    "viewport.set_af|camera.get",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_viewport_get_af(desc_viewport_get_af);

static const MethodDescriptor desc_viewport_get_depth_of_field = {
    "viewport.get_depth_of_field", "viewport",
    "Report realtime depth-of-field settings plus whether it is ACTUALLY blurring",
    "'enabled' is the setting; 'active' is the measurement, and they are not the same question. 'active' is false whenever any gate is closed - the setting is off, the shading mode is not Material, the camera is orthographic, or the lens is shut (aperture 0) - and 'inactive_reason' names which one. A bare false would let a caller record 'measured no blur' when the truth is 'could not blur'.",
    "render", "Render", false, "any",
    "viewport|get|depth|of|field|camera|post",
    "viewport.set_depth_of_field|camera.get",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_viewport_get_depth_of_field(desc_viewport_get_depth_of_field);

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
    "Canonical modes are 'scene' and 'three_point'. The default 'scene' reads the renderer's own light buffer; 'three_point' is the fixed material-inspection rig. Scene draws canonical world color/HDRI/Nishita behind geometry and uses it for ambient/specular. Both HDRI and Physical Sky use generated diffuse irradiance, GGX roughness prefilter mips and a split-sum BRDF LUT when world_ibl_ready is true; world_ibl_source says which producer filled them ('hdri' = uploaded environment texture, 'sky' = Physical Sky baked to an equirect, 'none' = neither), and world_ibl_fallback reports the bounded per-fragment cone path. world_sky_capture_supported false means the bake shader is missing, so Physical Sky cannot leave that fallback. The sun disc is excluded from the bake because the sun is delivered as an analytic directional light; near-mirror surfaces still sample it directly. Nishita contributes a direct sun with its own directional atlas tile. 'shadowed_light_count' counts scene lights only; 'world_sun_shadow' reports the reserved sun tile. A gap from 'scene_light_count' means later lights still illuminate without an atlas tile. A gap between 'scene_light_count' and 'scene_light_total' means the preview light loop itself is clamped. 'material_preview_active' false means the mode is stored but the material raster viewport is not on screen. DISPLAY TRANSFORM: Rendered and raster preview use PostProcess/ColorMath.h. display_* reports the GPU transport; post.get reports user gain, while post.get_exposure reports the resolved adaptive gain. None/linear means clipping; Reinhard is explicit. Raster preview uses the resolved EV from the Rendered HDR meter and does not meter its own display-encoded image. ★ 'world_background_source' is what the sky pass ACTUALLY draws, read from the adapter that drives the raster viewport and NOT OR-ed across backends: 'color' (world is a flat colour), 'hdri', 'sky', 'solid_fallback' (HDRI mode but no environment texture reached this adapter -- the background degrades to the flat world colour) or 'sky_analytic_fallback' (Physical Sky mode but no atmosphere LUT on this device -- a crude gradient). The two *_fallback values are the failure mode where Vulkan RT renders every world mode correctly while the realtime viewport sits on a solid background; world_ibl_ready cannot tell them apart because it is OR-ed over both adapters. It is 'not_drawn' whenever the preset is not 'scene': three_point skips the sky pass entirely, so there is no background to measure and reporting 'color' there would turn 'could not measure' into 'measured a flat colour'.",
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
    "Read values rather than matching on the preset name. 'scatter_lod_split' changes geometry submission. Scene shadows keep one fixed 4096 atlas so preset changes never destroy an image referenced by an in-flight frame; shadow_tile_resolution, shadow_tile_capacity, shadow_light_budget and shadow_pcf_samples report the live bounded contract. directional_shadow_cascades is 2 for Performance and 3 otherwise; directional lights and the Physical Sky sun consume that many atlas tiles and select the smallest camera-centred projection containing the receiver. A scene light beyond the budget still illuminates but has no atlas shadow. 'scene_pbr_shader' reports the comparison shader (GGX for Scene). The material parity strings are an explicit capability contract: opaque_core_parity is RT-aligned; material_graph_surface is bounded because pointiness/named attributes/AO/time inputs are neutral in raster; clearcoat is an iridescent lobe using the RT thin-film model; subsurface is a radius/scale profile approximation; translucency is a bounded thin-surface approximation; surface_anisotropy is unsupported because the current ABI fields are also legacy water wave controls; transparency is unsorted_alpha; transmission is screen_space_thickness with frozen opaque color/depth, bounded screen-space reflection, closed-mesh front/back thickness, RT texture/graph/IOR/Fresnel/Beer/roughness/dispersion semantics and environment fallback; overlapping or off-screen continuation/reflection remains bounded; resin_interior is a procedural bounded approximation; sdf_surface=shared_nanovdb_depth_pbr means realtime reads Vulkan RT's same field/transform/iso threshold and writes real raster depth, while lighting and environment continuation remain bounded rather than recursive. 'raster_viewport_available' false means the preset is stored but nothing on this machine reads it. Volume shadow budgets are volume_shadow_tile_resolution, volume_shadow_depth_layers and volume_shadow_steps (32/8/16 Performance, 128/12/48 Balanced or Auto, 256/16/64 Quality or Full). They describe configured budgets, not shader availability or measured GPU cost. Scene lighting shares a layered optical-depth shadow atlas between raster surfaces, SDF receivers and gas/VDB; lights outside the existing shadow budget remain unshadowed on surfaces. Up to 16 intersecting media are integrated per shadow ray; scalar transmittance and fixed depth layers are bounded approximations.",
    "render", "Render", false, "any",
    "viewport|quality|lod|proxy|scatter|measure",
    "viewport.set_quality|viewport.frame_telemetry",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_viewport_quality(desc_viewport_quality);

static const MethodDescriptor desc_viewport_raster_depth_prepass = {
    "viewport.raster_depth_prepass", "viewport",
    "Is the raster depth prepass requested?",
    "This is the REQUEST. viewport.frame_telemetry depth_prepass reports whether the pass actually ran; the two differ when the prepass pipeline failed to build, in which case the viewport silently falls back to the old single-pass path and a warning is logged.",
    "render", "Render", false, "any",
    "viewport|raster|depth|prepass",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_viewport_raster_depth_prepass(desc_viewport_raster_depth_prepass);

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

static const MethodDescriptor desc_viewport_reset_frame_timings = {
    "viewport.reset_frame_timings", "viewport",
    "Clear the per-pass raster timing window before a measured run",
    "Always call this first: the window is a MEASUREMENT of one configuration, and without a reset it still holds frames from whatever the viewport was doing before - a lifetime average across settings changes is the one thing a per-pass timing must never be. It also marks the viewport dirty so the first frame after it is real rather than an empty window. Then drive the camera and read viewport.frame_timings.",
    "render", "Render", false, "any",
    "viewport|reset|frame|timings|timing|profile|measure|benchmark",
    "viewport.frame_timings",
    nullptr, "viewport.frame_timings", nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_viewport_reset_frame_timings(desc_viewport_reset_frame_timings);

static const MethodDescriptor desc_viewport_retry_device_recovery = {
    "viewport.retry_device_recovery", "viewport",
    "Re-arms automatic viewport recovery after it gave up on repeated device losses.",
    "Clears the given_up state and schedules one more rebuild attempt. Deliberately manual: automatic recovery stops after max_streak consecutive losses because the loop can feed itself -- if the raster viewport is what trips the TDR, rebuilding it trips the TDR again, and each round resets the display driver. So calling this while the scene that caused the loss is still loaded most likely means another driver reset; change what the viewport is drawing first, or accept that cost knowingly. A manual viewport mode change in the UI re-arms recovery the same way, so a user at the panel is never stuck without IPC.",
    "render", "Render", false, "any",
    "viewport|retry|device|recovery",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_viewport_retry_device_recovery(desc_viewport_retry_device_recovery);

static const MethodDescriptor desc_viewport_rt_shadow = {
    "viewport.rt_shadow", "viewport",
    "RT shadow mask state: supported, requested, built, ray count and reason",
    "'enabled' is the REQUEST; 'ready' is what actually built. They differ whenever ray query is unsupported on the device, rayfusion_rt_shadow.spv is missing, the scene acceleration structure has not been built yet, or a material graph is bound - 'reason' says which. 'rays' is one ray per mask pixel, i.e. the per-frame ray count. 'cascades_replaced' is the number of cascade shadow VIEWS the ray pass took over this frame, and it is the only number that shows the cost actually MOVED rather than being added: rays rising while this stays 0 means both paths ran. Zero with ready=true is usually volumes in the scene - the SDF surface shader, the volume shader and the deep transmittance atlas never read the screen mask, so the cascades stay up for them and nothing is saved. IPC WRITE-THEN-MEASURE: toggling and reading in the same batch reports the PREVIOUS frame; put a frame between the write and the read.",
    "render", "Render", false, "any",
    "viewport|rt|shadow",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_viewport_rt_shadow(desc_viewport_rt_shadow);

static const MethodDescriptor desc_viewport_scene_load_guard = {
    "viewport.scene_load_guard", "viewport",
    "Report whether the scene-load Solid guard is active",
    "ON (the default) means opening a project or template drops Material/Rendered to Solid before touching the scene. Tearing a loaded scene down while a heavy viewport mode is bound loses the Vulkan device. OFF means openings run in whatever mode is bound - reproduce-the-fault mode, and every load logs a warning.",
    "render", "Render", false, "any",
    "viewport|scene|load|guard|shading|project|open|diagnostic",
    "viewport.set_scene_load_guard|viewport.frame_telemetry|project.open",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_viewport_scene_load_guard(desc_viewport_scene_load_guard);

static const MethodParam params_viewport_set_af[] = {
    {"area_mode", "int", true, "", nullptr, nullptr},
    {"enabled", "bool", true, "", nullptr, nullptr},
    {"focus_mode", "int", true, "", nullptr, nullptr},
    {"selected_point", "int", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_viewport_set_af = {
    "viewport.set_af", "viewport",
    "Configure the viewfinder AF point grid, its area mode and the focus mode",
    "NOT a monitoring overlay: AF WRITES the camera. In AF-C (focus_mode 2) the selected point re-measures the scene every frame and overwrites camera.focus_distance, so a focus distance you set over IPC will be reverted while it is on - set focus_mode 0 (MF) first if you want to own the value. area_mode 0=Single 1=Zone9 2=Zone21 3=Wide 4=CenterWeighted; the grid is 3x3 for every mode except Zone21, which is 5x5, so selected_point is in [0,8] or [0,24] and an out-of-range index is REJECTED, not clamped. Nothing is drawn unless the Camera HUD is on; read viewport.get_af 'active' and 'inactive_reason' rather than inferring from an unchanged image.",
    "render", "Render", false, "any",
    "viewport|set|af|camera|focus|autofocus|viewfinder",
    "viewport.get_af|camera.set_focus_distance|camera.set_depth_of_field|viewport.get_depth_of_field",
    nullptr, nullptr, nullptr, nullptr,
    params_viewport_set_af, 4,
    true
};
static const MethodRegistration reg_viewport_set_af(desc_viewport_set_af);

static const MethodParam params_viewport_set_automatic_cutout[] = {
    {"enabled", "any", false, "", nullptr, nullptr},
};
static const MethodDescriptor desc_viewport_set_automatic_cutout = {
    "viewport.set_automatic_cutout", "viewport",
    nullptr,
    nullptr,
    "render", "Render", false, "any",
    "viewport|set|automatic|cutout",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_viewport_set_automatic_cutout, 1,
    false
};
static const MethodRegistration reg_viewport_set_automatic_cutout(desc_viewport_set_automatic_cutout);

static const MethodParam params_viewport_set_depth_of_field[] = {
    {"enabled", "bool", false, "", "true", nullptr},
    {"max_coc_pixels", "float", false, "", "24.0", nullptr},
    {"max_taps", "int", false, "", "32", nullptr},
};
static const MethodDescriptor desc_viewport_set_depth_of_field = {
    "viewport.set_depth_of_field", "viewport",
    "Enable realtime raster depth of field and set its cost ceilings",
    "THE BLUR STRENGTH IS NOT HERE. The circle of confusion comes from the camera's own 'aperture' and 'focus_distance' with the same thin-lens formula the path tracer uses, so Rendered and Realtime cannot drift apart; 'max_coc_pixels' and 'max_taps' are COST ceilings. The default camera has aperture 0, so enabling this alone changes nothing - open a lens with camera.set_aperture first and set camera.set_focus_distance. Runs ONLY in Material shading mode (viewport.set_shading mode='material') and only on a perspective camera; Solid/Matcap content is display-referred and never enters the HDR pass. Values are REJECTED, not clamped. Read viewport.get_depth_of_field 'active' and 'inactive_reason' to see which of those gates is closed instead of guessing from an unchanged image.",
    "render", "Render", false, "any",
    "viewport|set|depth|of|field|camera|post",
    "viewport.get_depth_of_field|camera.set_aperture|camera.set_focus_distance|viewport.set_shading",
    nullptr, nullptr, nullptr, nullptr,
    params_viewport_set_depth_of_field, 3,
    true
};
static const MethodRegistration reg_viewport_set_depth_of_field(desc_viewport_set_depth_of_field);

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

static const MethodParam params_viewport_set_raster_depth_prepass[] = {
    {"enabled", "bool", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_viewport_set_raster_depth_prepass = {
    "viewport.set_raster_depth_prepass", "viewport",
    "Turn the alpha-tested depth prepass in the raster viewport on or off",
    "ON by default. The prepass does two jobs: early-Z rejects hidden fragments before the 1562-line scene shader runs, and it writes the per-pixel depth a ray-traced shadow pass needs as its ray origin. Measured motivation on a forest scene: fusion 427.6 ms/frame vs solid 17.5 ms at 2.6x MORE triangles, i.e. ~98% of the frame was fragment shading times 18.4 triangles per pixel. Switchable on purpose - a fix that cannot be turned off kills the measurement that would judge it. Read viewport.frame_telemetry depth_prepass for what ACTUALLY ran; the lever is only the request.",
    "render", "Render", false, "any",
    "viewport|set|raster|depth|prepass",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_viewport_set_raster_depth_prepass, 1,
    true
};
static const MethodRegistration reg_viewport_set_raster_depth_prepass(desc_viewport_set_raster_depth_prepass);

static const MethodParam params_viewport_set_raster_gpu_instancing[] = {
    {"enabled", "boolean", true, "True (default) allows the global instance buffer, GPU culling and LOD proxies; False forces the per-mesh fallback", nullptr, nullptr},
};
static const MethodDescriptor desc_viewport_set_raster_gpu_instancing = {
    "viewport.set_raster_gpu_instancing", "viewport",
    "Turn the raster viewport's GPU instancing, culling and LOD proxies on or off",
    "A/B lever, ON by default. OFF forces the per-mesh instance-buffer fallback -- which is what the REALTIME viewport ran unconditionally until 2026-09-08, because its buildRasterGeometry override carried a copy of the base class's tail and silently dropped the rebuildRasterInstanceLayout() call. One missing call cost two things at once, both measured on a 1000-instance foliage scene: (1) GPU culling and the scatter LOD proxies never engaged, so 45.9M triangles were submitted every frame while the proxies carried 30k of them against a 29.6M triangle target; (2) setRasterVisibleInstances fell to the per-mesh upload whose first statement is a full frame-ring drain, measured at 0.98 drains PER FRAME, so there was no CPU/GPU overlap at all. THE SCENE LOOKS CORRECT EITHER WAY -- this is a speed difference with no visual tell, which is exactly why it survived and why the switch has to exist: a fix that cannot be turned off destroys the measurement that would judge it. Read the outcome from viewport.frame_telemetry rather than from the image: 'gpu_culling' and 'global_instance_buffer' say whether it engaged, 'cull_mesh_count' is 0 whenever it did not, 'visible_triangles'/'proxy_triangles' say what was actually submitted, and 'resource_drains' delta over N frames says whether the pipeline is still draining every frame. This method reports the REQUEST; the telemetry reports what engaged, and they differ whenever the cull resources failed to build. Turning it off logs a warning, because a slow-but-correct viewport months later has no other trace.",
    "render", "Render", false, "any",
    "viewport|set|raster|gpu|instancing",
    "viewport.frame_telemetry|viewport.set_quality|viewport.set_shading",
    nullptr, nullptr, "viewport.frame_telemetry", nullptr,
    params_viewport_set_raster_gpu_instancing, 1,
    true
};
static const MethodRegistration reg_viewport_set_raster_gpu_instancing(desc_viewport_set_raster_gpu_instancing);

static const MethodParam params_viewport_set_rt_shadow[] = {
    {"enabled", "bool", true, "", nullptr, nullptr},
};
static const MethodDescriptor desc_viewport_set_rt_shadow = {
    "viewport.set_rt_shadow", "viewport",
    "Turn the screen-space RT shadow mask pass on or off",
    "OFF by default; turn it on wherever ray query is supported. The mask is consumed by the raster MESH shader (rtPreviewShadow returns the screen visibility for the covered light), and the cascade views for that light are then NOT drawn at all - read 'cascades_replaced' from viewport.rt_shadow to confirm the handoff happened. Exactly ONE light is covered: the first visible directional light, else the Nishita world sun. Point, spot and area lights keep their cascades, which is why they look identical with this on and off. WHAT IS GIVEN UP: fragments the screen mask declines - impostors, transparent replay, and any fragment whose depth does not match the prepass - lose that light's shadow instead of falling back to the atlas. WHEN NOTHING IS SAVED: with volumes in the scene the cascades stay up, because the SDF surface shader, the volume shader and the deep transmittance atlas read the atlas and have no screen mask. Motivation: the cascade shadow term measured 113 ms/frame, while a ray budget on this scene's own TLAS gave >=334 Mrays/s - about 5 ms for one ray per pixel at 1680x945. In a foliage scene the cascade cost is dominated by the alpha pipeline: one opacity texture fetch and discard per leaf fragment, per cascade. Requires a depth prepass and turns one on.",
    "render", "Render", false, "any",
    "viewport|set|rt|shadow",
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    params_viewport_set_rt_shadow, 1,
    true
};
static const MethodRegistration reg_viewport_set_rt_shadow(desc_viewport_set_rt_shadow);

static const MethodParam params_viewport_set_scene_load_guard[] = {
    {"enabled", "bool", true, "true restores the guard (openings drop to Solid). false reproduces the fault; leave it false only for the duration of a measurement.", nullptr, nullptr},
};
static const MethodDescriptor desc_viewport_set_scene_load_guard = {
    "viewport.set_scene_load_guard", "viewport",
    "Turn the scene-load Solid guard on or off (off = deliberately reproduce the device-lost fault)",
    "Turn it OFF only to reproduce the project-open device-lost on purpose. With the guard ON the faulty path is never taken, so the stale-descriptor tripwire can never fire and its silence proves NOTHING - the switch and the thing being measured would otherwise be the same switch. Read the outcome as values from viewport.frame_telemetry: stale_descset_rebuilds > 0 names the root cause class, device_lost true means the driver was lost anyway. Deliberately not exposed in any panel: leaving the faulty path selectable from a menu would turn the rule into a preference.",
    "render", "Render", false, "any",
    "viewport|set|scene|load|guard|shading|project|open|diagnostic|device_lost",
    "viewport.scene_load_guard|viewport.frame_telemetry|project.open",
    nullptr, nullptr, nullptr, nullptr,
    params_viewport_set_scene_load_guard, 1,
    true
};
static const MethodRegistration reg_viewport_set_scene_load_guard(desc_viewport_set_scene_load_guard);

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

static const MethodParam params_viewport_set_taa[] = {
    {"enabled", "bool", false, "Master switch. Off restores the un-jittered, history-free image.", "true", nullptr},
    {"samples", "int", false, "Stopping condition: how many jittered samples to accumulate before the viewport goes idle. 1-256.", "16", nullptr},
};
static const MethodDescriptor desc_viewport_set_taa = {
    "viewport.set_taa", "viewport",
    "Replace realtime raster temporal anti-aliasing settings",
    "Changing either field RESETS the accumulated history, so the next read of viewport.taa reports accumulated_samples near zero -- that is the reset, not a failure. 'samples' is rejected outside [1, 256] rather than clamped, because a silently clamped dial means the caller records a number it never actually measured. Turning TAA off does not just remove the blending: it also removes the sub-pixel jitter, so edges go back to hard 1:1 aliasing AND the screen-space GI/reflection noise stops being averaged over frames. Those two symptoms have one cause and one switch.",
    "render", "Render", false, "ok",
    "viewport|set|taa|antialiasing|aa|jitter|temporal|noise|denoise|realtime|raster",
    "viewport.taa|viewport.set_quality|viewport.set_depth_of_field",
    nullptr, nullptr, "viewport.taa", nullptr,
    params_viewport_set_taa, 2,
    true
};
static const MethodRegistration reg_viewport_set_taa(desc_viewport_set_taa);

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

static const MethodDescriptor desc_viewport_taa = {
    "viewport.taa", "viewport",
    "Read realtime raster temporal anti-aliasing state plus how many samples have ACTUALLY accumulated",
    "'enabled' is the REQUEST and 'accumulated_samples' is the MEASUREMENT; they answer different questions and the pipe can be off in four places that all coexist with enabled=true (raster_taa.spv missing, the compute pipeline failed to build, the history images could not be allocated, or the raster viewport is not the mode on screen). 'inactive_reason' names which one. Reading this right after a camera move gives 0 or 1: that is correct, not a failure -- accumulation restarts whenever the image changes, because a history that no longer matches the image is a ghost, not an average. 'converged' true means the viewport has stopped asking for extra frames, which is also where its GPU cost drops back to idle: TAA is not free while it converges and this field is the honest place to see that. 'target_samples' is a STOPPING CONDITION rather than a quality dial -- raising it buys a cleaner still image and pays for it in how long the GPU keeps working after the camera stops. 'last_ms' is the TAA dispatch plus the full-resolution copy back into the HDR target that the post pass reads.",
    "render", "Render", false, "enabled, target_samples, accumulated_samples, converged, supported, last_ms, inactive_reason",
    "viewport|taa|antialiasing|aa|jitter|temporal|noise|denoise|realtime|raster",
    "viewport.set_taa|viewport.frame_timings|rayfusion.screen_gi|rayfusion.reflections",
    nullptr, nullptr, nullptr, nullptr,
    nullptr, 0,
    true
};
static const MethodRegistration reg_viewport_taa(desc_viewport_taa);

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
