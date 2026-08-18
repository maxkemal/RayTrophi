/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          UI/scene_ui_simnodes.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 * Contents of the "Nodes" editor for the SIMULATION domain (plan D.4).
 *
 * ★★★ Three rules hold this panel together:
 *
 *   1. The panel owns NO graph. It draws rtapi::simulationGraph() — the same
 *      object `rt.sim_graph.*` reads and writes. A human can wire a graph and a
 *      script can then inspect it, and vice versa. A panel that kept its own
 *      copy is how the fracture UI cache outlived a scene change.
 *
 *   2. Every edit here calls the SAME rtapi:: entry point a script calls. No
 *      action exists in this file that has no script equivalent, so the panel
 *      cannot become the untestable manual step in the middle of an otherwise
 *      automated chain — which is the whole point of rule 1 in CLAUDE.md.
 *
 *   3. ★★★ EVERY EDITABLE FIELD MUST BE REACHABLE HERE. The first cut of this
 *      panel had a canvas and no inspector, so `sim.domain_ref` could only be
 *      pointed at a domain from a script: adding the node from the UI produced
 *      one that referred to nothing, forever, with no error and no hint. That is
 *      rule 1 inverted — a script-only capability the panel cannot reach — and
 *      it is just as untestable in the other direction.
 *
 * ★★ The sidebar therefore answers the three questions a node cannot answer by
 * existing: what does this node DO, what belongs on each pin, and what real
 * scene names can it point at. The last one is why the pickers list live
 * domains/objects/materials instead of taking free text: a name that does not
 * resolve fails at apply time, far away from where it was typed.
 */

#include "scene_ui.h"
#include "Api/RtApi.h"
#include "NodeSystem/SimulationNodes.h"
#include "NodeSystem/NodeRegistry.h"

#include "imgui.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// Presentation-only memory of the last action's report. This is a VIEW of
// something that already happened, not state anyone else needs: the authority
// for overrides is rtapi::simGraphOverrideCount(), read live below.
struct LastReport {
    bool        has = false;
    bool        ok = false;
    std::string text;
};
LastReport g_last_report;

// Fixed for now, and named as a constant rather than `properties_width` so the
// name does not promise a drag handle that is not there.
const float kPropertiesWidth = 300.0f;

void setReport(bool ok, std::string text) {
    g_last_report.has = true;
    g_last_report.ok = ok;
    g_last_report.text = std::move(text);
}

std::string joinFirst(const std::vector<std::string>& items, size_t max_items) {
    std::string out;
    for (size_t i = 0; i < items.size() && i < max_items; ++i) {
        if (!out.empty()) out += "; ";
        out += items[i];
    }
    if (items.size() > max_items) out += " (+" + std::to_string(items.size() - max_items) + " more)";
    return out;
}

const ImVec4 kWarn(1.00f, 0.78f, 0.35f, 1.0f);
const ImVec4 kBad (1.00f, 0.62f, 0.55f, 1.0f);
const ImVec4 kGood(0.55f, 0.88f, 0.62f, 1.0f);

void helpText(const char* text) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("%s", text);
    ImGui::PopStyleColor();
}

// A name picker over real scene names.
//
// ★★ Deliberately not a free-text box. These fields are IDENTITIES — a domain
// name, an object name, a material name — and an identity that does not resolve
// produces no error until apply time, in a different panel, with a message that
// names the node rather than the typo. Offering only names that exist moves that
// failure to the moment of choosing. `(none)` stays available because empty is a
// legitimate value for the optional ones.
//
// When the current value is NOT in the list it is shown anyway, marked, because
// hiding it would silently rewrite what the graph says. A script may have set a
// name for a domain that has not been created yet, and that is the graph's
// statement to make, not the panel's to erase.
bool namePicker(const char* label, const std::string& current,
                const std::vector<std::string>& options, bool allow_empty,
                std::string& out_choice)
{
    bool changed = false;
    const bool resolves = current.empty() || std::find(options.begin(), options.end(), current) != options.end();
    std::string preview = current.empty() ? "(none)" : current;
    if (!resolves) preview += "   [not in scene]";

    if (!resolves) ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    const bool open = ImGui::BeginCombo(label, preview.c_str());
    if (!resolves) ImGui::PopStyleColor();

    if (open) {
        if (allow_empty && ImGui::Selectable("(none)", current.empty())) {
            out_choice.clear();
            changed = true;
        }
        for (const auto& name : options) {
            if (ImGui::Selectable(name.c_str(), name == current)) {
                out_choice = name;
                changed = true;
            }
        }
        if (options.empty()) ImGui::TextDisabled("nothing in the scene to pick");
        ImGui::EndCombo();
    }
    if (!resolves && ImGui::IsItemHovered()) {
        ImGui::SetTooltip("'%s' does not exist in this scene.\n"
                          "The graph still says it; nothing was rewritten.\n"
                          "It will be reported as a failure when applied.",
                          current.c_str());
    }
    return changed;
}

} // namespace

void SceneUI::drawSimulationNodePanel(UIContext& ctx)
{
    (void)ctx;   // the simulation graphs are reached through rtapi, not the scene

    // -- Scope bar: whose graph is this? -------------------------------------
    //
    // *** The canvas shows ONE scoped graph and always says which. The scope
    // and owner are held as editor view state and published through
    // rt.editor.get_state, so a script can read what the user is looking at --
    // the same reasoning that put bottom_editor there. The panel keeps no copy.
    const std::string scope = sim_graph_scope;
    const std::string owner = sim_graph_owner;

    {
        static const char* kScopes[] = { "object", "domain", "world" };
        int scope_index = 1;
        for (int i = 0; i < 3; ++i) if (scope == kScopes[i]) scope_index = i;
        ImGui::TextDisabled("Scope");
        ImGui::SameLine();
        ImGui::PushItemWidth(110.0f);
        if (ImGui::BeginCombo("##sim_scope", kScopes[scope_index])) {
            for (int i = 0; i < 3; ++i) {
                if (ImGui::Selectable(kScopes[i], i == scope_index)) {
                    // Through the API, exactly as a script would: the selection
                    // is a value the core owns, not a panel-local variable.
                    const rtapi::Result r = rtapi::setSimGraphScope(kScopes[i], "");
                    if (!r.ok) setReport(false, r.error);
                }
            }
            ImGui::EndCombo();
        }
        ImGui::PopItemWidth();

        // The owner picker lists REAL scene entities. A free-text field here
        // would let the user name something that does not exist and get a graph
        // that drives nothing -- the failure simGraphCreate refuses.
        if (scope != "world") {
            std::vector<std::string> owners;
            if (scope == "domain") {
                std::vector<rtapi::FluidDomainInfo> domains;
                if (rtapi::listFluidDomains(domains).ok)
                    for (const auto& d : domains) owners.push_back(d.name);
            } else {
                owners = rtapi::listObjects();
            }
            ImGui::SameLine();
            ImGui::PushItemWidth(200.0f);
            const char* preview = owner.empty() ? "(pick one)" : owner.c_str();
            if (ImGui::BeginCombo("##sim_owner", preview)) {
                for (const auto& name : owners) {
                    if (ImGui::Selectable(name.c_str(), name == owner)) {
                        const rtapi::Result r = rtapi::setSimGraphScope(scope, name);
                        if (!r.ok) setReport(false, r.error);
                    }
                }
                if (owners.empty())
                    ImGui::TextDisabled("nothing of this kind in the scene");
                ImGui::EndCombo();
            }
            ImGui::PopItemWidth();
        }
    }

    // ** Null means NO GRAPH for this scope, and the panel says exactly that
    // instead of falling back to another owner's canvas. Drawing someone else's
    // nodes under this owner's name is the panel-lies failure class this whole
    // layer exists to end.
    NodeSystem::Sim::SimulationNodeGraph* graph_ptr =
        rtapi::simulationGraph(scope, owner);
    if (!graph_ptr) {
        ImGui::Separator();
        if (scope != "world" && owner.empty()) {
            ImGui::TextColored(kWarn, "No %s selected.", scope.c_str());
            helpText("Pick one above. A simulation graph belongs to a scene "
                     "entity -- there is no global simulation graph, so there is "
                     "nothing to show until you say whose graph you mean.");
        } else {
            ImGui::TextColored(kWarn, "No graph for this %s yet.", scope.c_str());
            helpText("The graph reflects an entity that already exists; creating "
                     "it here adds the canvas, never the entity.");
            if (ImGui::Button("Create graph")) {
                const rtapi::Result r = rtapi::simGraphCreate(scope, owner);
                setReport(r.ok, r.ok ? "graph created" : r.error);
            }
        }
        if (g_last_report.has)
            ImGui::TextColored(g_last_report.ok ? kGood : kBad, "%s",
                               g_last_report.text.c_str());
        return;
    }
    NodeSystem::Sim::SimulationNodeGraph& graph = *graph_ptr;

    // ── Toolbar ─────────────────────────────────────────────────────────────
    // `allow_restart` is a deliberate opt-in and stays OFF between sessions.
    // Some parameters (voxel_size and friends) cannot change without throwing
    // away a running simulation, and a graph edit must never do that silently —
    // so the default is refuse-and-report, and the user has to ask.
    static bool allow_restart = false;

    if (ImGui::Button("Evaluate")) {
        const rtapi::SimGraphEvaluation ev = rtapi::simGraphEvaluate(scope, owner);
        // ★ Evaluation produces INTENT. It applies nothing, and it must not
        // disturb the solver — no cache clear, no dirty sweep. The wording here
        // says so, because a button called "Evaluate" next to a running sim
        // otherwise reads like "run the simulation".
        std::string text = std::to_string(ev.commands.size()) + " command(s)";
        if (!ev.restart_requests.empty())
            text += ", " + std::to_string(ev.restart_requests.size()) + " need a restart";
        // * On failure report WHY. "graph did not evaluate" alone cannot be
        // told apart from a graph that evaluated and declared nothing.
        setReport(ev.evaluated, ev.evaluated ? text
                  : (ev.error.empty() ? "graph did not evaluate" : ev.error));
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Runs the graph and reports what it WOULD do.\nApplies nothing; the solver is untouched.");

    ImGui::SameLine();
    if (ImGui::Button("Apply")) {
        const rtapi::SimApplyResult r = rtapi::simGraphApply(scope, owner, allow_restart);
        std::string text = std::to_string(r.applied) + " applied";
        if (!r.refused.empty()) text += ", refused: " + joinFirst(r.refused, 2);
        if (!r.failed.empty())  text += ", failed: "  + joinFirst(r.failed, 2);
        // ★★ `refused` and `failed` are NOT the same thing and are never merged.
        // Refused = the graph asked for something that would restart the sim and
        // was not allowed. Failed = the application layer has no way to honour
        // the command at all (a missing script surface, say). Collapsing them
        // would turn a known gap into a permission problem.
        setReport(r.ok && r.failed.empty(), text);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Applies the graph as a REVERSIBLE override layer.\nAuthored values are captured before the first write.");

    ImGui::SameLine();
    ImGui::Checkbox("allow restart", &allow_restart);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Off: parameters that would restart the simulation are refused and reported.\nOn: they are applied and the running simulation is discarded.");

    ImGui::SameLine();
    const uint32_t held = rtapi::simGraphOverrideCount();
    ImGui::BeginDisabled(held == 0);
    if (ImGui::Button("Clear Overrides")) {
        const rtapi::Result r = rtapi::simGraphClearOverrides();
        setReport(r.ok, r.ok ? "authored values restored" : r.error);
    }
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Restores every captured authored value.\nAn override is reversible by construction; this is the undo.");

    ImGui::SameLine();
    // Read live, every frame. Mirroring this into a member would create a
    // second copy of a fact the API already owns — the failure this whole
    // layer exists to avoid.
    if (held > 0) {
        ImGui::TextColored(kWarn, "%u override%s held", held, held == 1 ? "" : "s");
    } else {
        ImGui::TextDisabled("no overrides");
    }

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    ImGui::TextDisabled("%zu node%s", graph.nodes.size(), graph.nodes.size() == 1 ? "" : "s");
    ImGui::SameLine();
    if (ImGui::SmallButton("Clear Graph")) {
        const rtapi::Result r = rtapi::simGraphClear(scope, owner);
        setReport(r.ok, r.ok ? "graph cleared (owner node re-seeded)" : r.error);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Removes every node.\nDoes NOT clear overrides — use Clear Overrides for that,\nor the authored values stay overwritten with no graph left to explain why.");

    if (g_last_report.has) {
        ImGui::TextColored(g_last_report.ok ? kGood : kBad, "%s", g_last_report.text.c_str());
    } else {
        ImGui::TextDisabled("Right-click the canvas to add nodes. Select one to edit it on the right.");
    }

    ImGui::Separator();

    // ── Live scene names, fetched once per frame ────────────────────────────
    // Read fresh every frame rather than cached on selection: a domain created
    // while this panel is open must appear without the user having to know that
    // reselecting refreshes a list.
    std::vector<std::string> domain_names;
    {
        std::vector<rtapi::FluidDomainInfo> domains;
        if (rtapi::listFluidDomains(domains).ok) {
            domain_names.reserve(domains.size());
            for (const auto& d : domains) domain_names.push_back(d.name);
        }
    }
    const std::vector<std::string> object_names = rtapi::listObjects();
    std::vector<std::string> material_names;
    for (const auto& m : rtapi::listMaterials()) material_names.push_back(m.name);

    // ── Add-node menu, sourced from the registry ────────────────────────────
    //
    // ★ Built from NodeRegistry rather than a hand-written list: a node type
    // that exists for scripts but is missing from this menu would be a
    // capability the panel silently does not have, and nobody would notice
    // until they went looking for it.
    simulationNodeEditorUI.onDrawBackgroundMenu = [this, &graph, scope, owner]() {
        const ImVec2 spawn = simulationNodeEditorUI.mousePosOnRightClick;
        auto types = NodeSystem::NodeRegistry::instance().getAllTypes();
        types.erase(std::remove_if(types.begin(), types.end(),
                        [](const NodeSystem::NodeTypeInfo& t) {
                            return t.typeId.rfind("sim.", 0) != 0;
                        }),
                    types.end());
        std::sort(types.begin(), types.end(),
                  [](const NodeSystem::NodeTypeInfo& a, const NodeSystem::NodeTypeInfo& b) {
                      return a.displayName < b.displayName;
                  });
        for (const auto& type : types) {
            if (ImGui::MenuItem(type.displayName.c_str())) {
                // Through the API, exactly as a script would.
                uint32_t new_id = 0;
                const rtapi::Result r =
                    rtapi::simGraphAddNode(scope, owner, type.typeId, new_id);
                if (r.ok) {
                    if (NodeSystem::NodeBase* node = graph.getNode(new_id)) {
                        node->x = spawn.x;
                        node->y = spawn.y;
                        simulationNodeEditorUI.onNodeAdded(graph, node);
                    }
                } else {
                    setReport(false, r.error);
                }
            }
            if (!type.description.empty() && ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::PushTextWrapPos(420.0f);
                ImGui::TextUnformatted(type.description.c_str());
                ImGui::PopTextWrapPos();
                ImGui::EndTooltip();
            }
        }
    };

    // ── Canvas + properties split ───────────────────────────────────────────
    const float avail_w = ImGui::GetContentRegionAvail().x;
    const float side_w = std::clamp(kPropertiesWidth, 220.0f, std::max(240.0f, avail_w - 260.0f));
    const float canvas_w = std::max(200.0f, avail_w - side_w - 8.0f);

    ImGui::BeginChild("##sim_canvas", ImVec2(canvas_w, 0), false);
    simulationNodeEditorUI.config.showMinimap = true;
    simulationNodeEditorUI.draw(graph);
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("##sim_properties", ImVec2(0, 0), true);
    drawSimulationNodeProperties(graph, domain_names, object_names, material_names);
    ImGui::EndChild();
}

void SceneUI::drawSimulationNodeProperties(NodeSystem::Sim::SimulationNodeGraph& graph,
                                           const std::vector<std::string>& domain_names,
                                           const std::vector<std::string>& object_names,
                                           const std::vector<std::string>& material_names)
{
    // Both, and the outer one is not optional: the simulation nodes derive from
    // the same NodeSystem V2 core the geometry graph uses (SimNodeBase ->
    // NodeSystem::NodeBase, SimulationNodeGraph -> NodeSystem::GraphBase), so
    // NodeBase/DataType/Link live one namespace ABOVE Sim. `using namespace
    // NodeSystem::Sim` alone does not reach them.
    using namespace NodeSystem;
    using namespace NodeSystem::Sim;

    // ★★ Taken from the GRAPH, not passed in. The graph carries the scope it
    // belongs to, so there is exactly one answer to "whose graph is this?" —
    // threading a second copy through the signature would create a place for
    // the two to disagree, which is the defect this layer exists to end.
    const std::string scope = scopeName(graph.scope);
    const std::string owner = graph.owner;

    // ── Scene context: what this graph can point AT ─────────────────────────
    //
    // ★★★ This block exists because of a real confusion report: adding a Domain
    // node and expecting a domain to appear in the viewport. It never will —
    // N0's first rule is that a node owns no state, so Domain NAMES a domain
    // that already exists. Saying that only in a comment left the user to
    // discover it by the absence of anything happening, which is the worst
    // possible teacher.
    if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (domain_names.empty()) {
            ImGui::TextColored(kWarn, "No simulation domains in this scene.");
            helpText("A Domain node NAMES a domain; it does not create one, and "
                     "it will never make one appear in the viewport. Create a "
                     "fluid or gas domain first (Properties > Simulation, or "
                     "rt.fluid.create_domain from a script), then pick it here.");
        } else {
            ImGui::Text("%zu domain%s:", domain_names.size(), domain_names.size() == 1 ? "" : "s");
            for (const auto& name : domain_names) ImGui::BulletText("%s", name.c_str());
            helpText("A Domain node points at one of these by name. Names survive "
                     "rebuilds and mean the same thing over IPC, which is why no "
                     "pin ever carries a handle.");
        }
    }

    ImGui::Separator();

    // Selection comes from the canvas: NodeBase::selected is what the editor
    // sets, so the sidebar follows the click instead of keeping its own idea of
    // what is selected.
    NodeBase* selected = nullptr;
    int selected_count = 0;
    for (auto& node : graph.nodes) {
        if (node->selected) { ++selected_count; if (!selected) selected = node.get(); }
    }

    if (!selected) {
        if (graph.nodes.empty()) {
            ImGui::TextColored(kWarn, "The graph is empty.");
            helpText("Right-click the canvas to add nodes. A minimal useful graph "
                     "is Domain -> Set Parameter: the Domain names what to drive, "
                     "and Set Parameter overrides one of its solver settings.");
            if (graph.ownerNodeId != 0 && ImGui::Button("Add starter graph")) {
                // Built through the API, exactly like a script would build it —
                // so the button cannot do anything a script could not.
                // * The owner node is already on the canvas, so the starter
                // graph wires the SETTER to it rather than adding a second
                // Domain node that would name the same entity twice.
                uint32_t setter = 0;
                rtapi::Result r =
                    rtapi::simGraphAddNode(scope, owner, "sim.set_parameter", setter);
                if (r.ok && graph.ownerNodeId != 0)
                    r = rtapi::simGraphConnect(scope, owner, graph.ownerNodeId, 0,
                                               setter, 0);
                if (r.ok) {
                    if (NodeBase* n = graph.getNode(graph.ownerNodeId))
                        { n->x =  60.0f; n->y = 120.0f; }
                    if (NodeBase* n = graph.getNode(setter))
                        { n->x = 340.0f; n->y = 120.0f; }
                }
                setReport(r.ok, r.ok ? "starter graph added" : r.error);
            }
        } else {
            ImGui::TextDisabled("Select a node on the canvas to edit it.");
        }
        return;
    }
    if (selected_count > 1) {
        ImGui::TextColored(kWarn, "%d nodes selected; editing the first.", selected_count);
    }

    // ── Identity and purpose ────────────────────────────────────────────────
    ImGui::TextUnformatted(selected->metadata.displayName.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("#%u  %s", selected->id, selected->metadata.typeId.c_str());
    if (!selected->metadata.description.empty()) {
        ImGui::Spacing();
        helpText(selected->metadata.description.c_str());
    }

    // ── Wiring: what belongs on each pin, and whether it is there ───────────
    //
    // ★★ "What connects to what and why" was the other half of the confusion
    // report. A pin's NAME and TYPE are on the canvas; what is missing is
    // whether it is satisfied. An unconnected required input is not an error the
    // graph reports — the node simply produces nothing — so it has to be shown.
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Wiring", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto pinConnected = [&graph](uint32_t pin_id) {
            for (const auto& link : graph.links)
                if (link.startPinId == pin_id || link.endPinId == pin_id) return true;
            return false;
        };
        if (selected->inputs.empty()) {
            helpText("No inputs — this node is a source. It names something and "
                     "hands that identity downstream.");
        }
        for (const auto& pin : selected->inputs) {
            const bool connected = pinConnected(pin.id);
            ImGui::TextColored(connected ? kGood : kWarn, connected ? "in  %s" : "in  %s  (empty)",
                               pin.name.c_str());
            if (!connected) {
                // Say what would satisfy it, in scene terms rather than type terms.
                const char* want =
                    pin.dataType == NodeSystem::DataType::DomainRef  ? "Connect a Domain node (or the pass-through output of another domain-scoped node)." :
                    pin.dataType == NodeSystem::DataType::SurfaceField ? "Connect an Object node (or the pass-through output of another surface node)." :
                    pin.dataType == NodeSystem::DataType::Field      ? "Connect a Field node." :
                                                           "Connect a matching output.";
                ImGui::Indent();
                helpText(want);
                ImGui::Unindent();
            }
        }
        for (const auto& pin : selected->outputs) {
            const bool connected = pinConnected(pin.id);
            ImGui::TextDisabled("out %s%s", pin.name.c_str(), connected ? "" : "  (unused)");
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // ── Editable fields ─────────────────────────────────────────────────────
    //
    // ★★★ Every write goes through rtapi::simGraphSetNode*, never through the
    // node pointer directly. That is not ceremony: those functions set `dirty`,
    // and a field written behind their back would be a change the graph does not
    // know it has. It also guarantees the panel cannot edit anything a script
    // cannot, which is what keeps this surface testable.
    const uint32_t id = selected->id;
    auto setText = [&](const char* key, const std::string& value) {
        const rtapi::Result r = rtapi::simGraphSetNodeText(scope, owner, id, key, value);
        if (!r.ok) setReport(false, r.error);
    };
    auto setValue = [&](const char* key, float value) {
        const rtapi::Result r = rtapi::simGraphSetNodeValue(scope, owner, id, key, value);
        if (!r.ok) setReport(false, r.error);
    };

    ImGui::PushItemWidth(-1.0f);

    // ★★★ Opt-in parameter grid for Solver / Domain Settings / Emitter.
    //
    // The tick box is not decoration: an unticked field is one this graph has NO
    // OPINION about and will not write. Drawing only the value would make an
    // untouched dial look like an authored zero and quietly flatten the user's
    // own setting on the next Apply.
    auto drawOptInFields = [&](auto& owner_node) {
        for (auto& f : owner_node.fields) {
            ImGui::PushID(f.key);
            bool use = f.use;
            if (ImGui::Checkbox("##use", &use))
                setValue((std::string(f.key) + ".use").c_str(), use ? 1.0f : 0.0f);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Off = this graph does not write %s.\n"
                                  "That is NOT the same as writing zero.", f.key);
            ImGui::SameLine();
            ImGui::TextUnformatted(f.key);
            float value = f.value;
            ImGui::PushItemWidth(-1.0f);
            // Editing a value turns the field on, matching the API: a number
            // typed into an inert field would be a silent no-op.
            if (ImGui::DragFloat("##v", &value, 0.01f)) setValue(f.key, value);
            ImGui::PopItemWidth();
            if (use && NodeSystem::Sim::SetParameterNode::keyRequiresRestart(f.key))
                ImGui::TextColored(kWarn, "requires a restart");
            ImGui::PopID();
        }
    };

    // ** The OWNER node is drawn pinned, not as a picker. Its target IS the
    // graph's scope, so offering a name field would present a choice the API
    // refuses -- an editable control that cannot take an edit is worse than no
    // control, because the user learns it is broken only after trying.
    if (graph.isOwnerNode(selected->id)) {
        ImGui::TextUnformatted(scope == "object" ? "Object" : "Domain");
        ImGui::TextDisabled("%s", owner.c_str());
        helpText("This graph's owner. It cannot be retargeted here: a graph "
                 "filed under one entity that drove another would make every "
                 "later reading agree with the wrong one. To work on a "
                 "different one, switch it in the Scope bar above.");
    }
    else if (auto* aspect = dynamic_cast<DomainParamNodeBase*>(selected)) {
        drawOptInFields(*aspect);
        helpText("Only ticked fields are written. Everything else keeps the "
                 "value authored on the domain.");
    }
    else if (auto* emitter = dynamic_cast<EmitterNode*>(selected)) {
        ImGui::TextUnformatted("Flow source");
        std::vector<std::string> emitter_names;
        {
            std::vector<rtapi::SimulationFlowSourceInfo> sources;
            if (rtapi::listSimulationFlowSources(sources).ok)
                for (const auto& src : sources) emitter_names.push_back(src.name);
        }
        std::string choice;
        if (namePicker("##emitter", emitter->emitterName, emitter_names, true, choice))
            setText("emitter", choice);
        helpText("The flow source this node overrides. It creates none, and it "
                 "does not change which domain the source feeds -- that binding "
                 "resolves an ambiguity and is authored on the source itself.");
        ImGui::Spacing();
        drawOptInFields(*emitter);

        ImGui::PushID("substance");
        bool use_sub = emitter->useSubstance;
        if (ImGui::Checkbox("##use", &use_sub))
            setValue("fluid_substance.use", use_sub ? 1.0f : 0.0f);
        ImGui::SameLine();
        ImGui::TextUnformatted("fluid_substance");
        char sub_buf[128];
        std::snprintf(sub_buf, sizeof(sub_buf), "%s", emitter->substance.c_str());
        ImGui::PushItemWidth(-1.0f);
        if (ImGui::InputText("##sub", sub_buf, sizeof(sub_buf),
                             ImGuiInputTextFlags_EnterReturnsTrue))
            setText("fluid_substance", sub_buf);
        ImGui::PopItemWidth();
        // ★ An empty substance is a REAL value, so emptiness cannot encode
        // "unset" -- the tick box is the only thing that says whether this
        // graph has an opinion about it.
        helpText("Empty is a legitimate substance value, so the tick box -- not "
                 "emptiness -- decides whether it is written. Enter to commit.");
        ImGui::PopID();
    }
    else if (auto* dom = dynamic_cast<DomainRefNode*>(selected)) {
        ImGui::TextUnformatted("Domain");
        std::string choice;
        if (namePicker("##domain", dom->domainName, domain_names, true, choice))
            setText("domain", choice);
    }
    else if (auto* obj = dynamic_cast<ObjectRefNode*>(selected)) {
        ImGui::TextUnformatted("Object");
        std::string choice;
        if (namePicker("##object", obj->objectName, object_names, true, choice))
            setText("object", choice);
    }
    else if (auto* setter = dynamic_cast<SetParameterNode*>(selected)) {
        ImGui::TextUnformatted("Parameter key");
        char key_buf[128];
        std::snprintf(key_buf, sizeof(key_buf), "%s", setter->key.c_str());
        if (ImGui::InputText("##key", key_buf, sizeof(key_buf),
                             ImGuiInputTextFlags_EnterReturnsTrue)) {
            setText("key", key_buf);
        }
        helpText("Solver parameter name, e.g. kinematic_viscosity, surface_tension, "
                 "pore_amount. Press Enter to commit.");
        float value = setter->value;
        ImGui::TextUnformatted("Value");
        if (ImGui::DragFloat("##value", &value, 0.01f)) setValue("value", value);
        if (SetParameterNode::keyRequiresRestart(setter->key)) {
            ImGui::TextColored(kWarn, "Requires a restart");
            helpText(setter->restartReason().c_str());
            helpText("Apply refuses this unless 'allow restart' is ticked. The node "
                     "says so; it never decides.");
        }
    }
    else if (auto* coupling = dynamic_cast<CouplingNodeBase*>(selected)) {
        bool active = coupling->active;
        if (ImGui::Checkbox("Active", &active)) setValue("active", active ? 1.0f : 0.0f);
        helpText("A coupling is on or off — there is no strength dial, because "
                 "there is no single authored gain behind it and a dial that "
                 "scales nothing real would be a lie. Set coupling parameters by "
                 "chaining Set Parameter nodes after this one.");
        ImGui::Spacing();
        ImGui::TextDisabled("coupling id: %s", coupling->couplingId());
        helpText("The graph DECLARES this coupling; the solver decides the order "
                 "it actually runs in and reports it back. Compare the two with "
                 "rt.sim_graph.couplings().");
    }
    else if (auto* inspect = dynamic_cast<FieldInspectNode*>(selected)) {
        ImGui::TextUnformatted("Attribute");
        // Attribute names come from the naming layer, for the domain this node
        // is actually bound to — not a hardcoded list that can drift from it.
        std::vector<std::string> attrs;
        for (const auto& d : domain_names) {
            for (const auto& a : rtapi::simListAttributes(d))
                if (std::find(attrs.begin(), attrs.end(), a) == attrs.end()) attrs.push_back(a);
        }
        std::string choice;
        if (namePicker("##channel", inspect->channel, attrs, false, choice))
            setText("channel", choice);
        ImGui::Spacing();
        // ★ "not measured" and "measured zero" are different answers and are
        // never allowed to look the same.
        if (!inspect->stats.available) {
            ImGui::TextColored(kWarn, "no reading");
            helpText("Not the same as zero: the value could not be measured. Bind "
                     "a domain that has live particles, then Evaluate.");
        } else {
            ImGui::Text("n = %u", inspect->stats.particle_count);
            ImGui::Text("min %.4f  max %.4f", inspect->stats.min_value, inspect->stats.max_value);
            ImGui::Text("mean %.4f", inspect->stats.mean_value);
            if (!inspect->stats.in_sync) {
                ImGui::TextColored(kBad, "array %u != %u particles",
                                   inspect->stats.array_size, inspect->stats.particle_count);
                helpText("The solver's arrays disagree with each other, so entry i "
                         "no longer describes particle i. This is a solver bug, not "
                         "a display problem.");
            }
        }
    }
    else if (auto* surf = dynamic_cast<SurfaceInspectNode*>(selected)) {
        ImGui::TextUnformatted("Channel");
        static const char* kSurfaceChannels[] = {
            "temperature", "char", "melt", "moisture", "fuel_remaining", "mass_loss"
        };
        std::vector<std::string> channels(std::begin(kSurfaceChannels), std::end(kSurfaceChannels));
        std::string choice;
        if (namePicker("##surface_channel", surf->channel, channels, false, choice))
            setText("channel", choice);
        ImGui::Spacing();
        if (!surf->stats.available) {
            ImGui::TextColored(kWarn, "no reading");
            helpText("Connect an Object node whose object has a material state "
                     "field, then Evaluate.");
        } else {
            ImGui::Text("n = %u", surf->stats.particle_count);
            ImGui::Text("min %.4f  max %.4f", surf->stats.min_value, surf->stats.max_value);
            ImGui::Text("mean %.4f", surf->stats.mean_value);
            if (!surf->stats.host_fresh) {
                ImGui::TextColored(kWarn, "stale host mirror");
                helpText("Read from the host copy as of the last readback — before "
                         "any readback these are INITIALISATION values, not "
                         "measurements. -1 means 'not seeded yet', not a level.");
            }
        }
    }
    else if (auto* field = dynamic_cast<FieldReadNode*>(selected)) {
        int source = field->source == FieldReadNode::Source::GridChannel ? 0 : 1;
        ImGui::TextUnformatted("Source");
        if (ImGui::Combo("##source", &source, "grid channel\0element attribute\0"))
            setText("source", source == 0 ? "grid" : "attribute");
        helpText("A grid channel lives on the solver's voxel grid. An element "
                 "attribute is per-particle data that already exists as parallel "
                 "arrays. They are different representations, not a performance "
                 "choice.");
        ImGui::TextUnformatted("Channel");
        std::vector<std::string> options;
        if (source == 0) {
            options = { "density", "temperature", "velocity", "fuel", "flame", "sdf" };
        } else {
            for (const auto& d : domain_names)
                for (const auto& a : rtapi::simListAttributes(d))
                    if (std::find(options.begin(), options.end(), a) == options.end())
                        options.push_back(a);
        }
        std::string choice;
        if (namePicker("##field_channel", field->channel, options, false, choice))
            setText("channel", choice);
    }
    else if (auto* sub = dynamic_cast<SubstanceNode*>(selected)) {
        ImGui::TextUnformatted("Substance");
        std::vector<std::string> substances;
        rtapi::listMaterialSubstances(substances);
        std::string choice;
        if (namePicker("##substance", sub->substanceName, substances, false, choice))
            setText("substance", choice);
        ImGui::Spacing();
        bool override_ignition = sub->overrideIgnition;
        if (ImGui::Checkbox("Override ignition point", &override_ignition))
            setValue("override_ignition", override_ignition ? 1.0f : 0.0f);
        ImGui::BeginDisabled(!override_ignition);
        float kelvin = sub->ignitionKelvin;
        ImGui::TextUnformatted("Ignition (K)");
        if (ImGui::DragFloat("##ignition", &kelvin, 1.0f, 0.0f, 4000.0f))
            setValue("ignition_kelvin", kelvin);
        ImGui::EndDisabled();
        helpText("Authored in Kelvin and applied before the normalized "
                 "conversion, so the system keeps exactly one Kelvin point.");
        float burn = sub->burnRateScale;
        ImGui::TextUnformatted("Burn rate scale");
        if (ImGui::DragFloat("##burn", &burn, 0.01f, 0.0f, 10.0f))
            setValue("burn_rate_scale", burn);
        float fuel = sub->fuelCapacityScale;
        ImGui::TextUnformatted("Fuel capacity scale");
        if (ImGui::DragFloat("##fuel", &fuel, 0.01f, 0.0f, 10.0f))
            setValue("fuel_capacity_scale", fuel);
        helpText("The scales are this object's deviation from the substance, not "
                 "a replacement for it.");
    }
    else if (auto* pyro = dynamic_cast<PyrolysisNode*>(selected)) {
        bool active = pyro->active;
        if (ImGui::Checkbox("Pyrolyses on flame contact", &active))
            setValue("active", active ? 1.0f : 0.0f);
    }
    else if (auto* phase = dynamic_cast<PhaseChangeNode*>(selected)) {
        bool flow = phase->meltFlow;
        if (ImGui::Checkbox("Molten material flows", &flow))
            setValue("melt_flow", flow ? 1.0f : 0.0f);
        float loss = phase->heightLoss;
        ImGui::TextUnformatted("Height loss");
        if (ImGui::DragFloat("##height_loss", &loss, 0.01f, 0.0f, 1.0f))
            setValue("melt_height_loss", loss);
        float spread = phase->spread;
        ImGui::TextUnformatted("Spread");
        if (ImGui::DragFloat("##spread", &spread, 0.01f, 0.0f, 8.0f))
            setValue("melt_spread", spread);
        helpText("No melting point here on purpose: it comes from the substance, "
                 "so a graph cannot disagree with what the object is made of.");
    }
    else if (auto* liquid = dynamic_cast<LiquidMaterialNode*>(selected)) {
        ImGui::TextUnformatted("Surface material");
        std::string choice;
        if (namePicker("##surface_material", liquid->surfaceMaterial, material_names, true, choice))
            setText("surface_material", choice);
        helpText("Shades the SDF isosurface. Empty = the built-in dielectric.");
        ImGui::TextUnformatted("Splat material");
        if (namePicker("##splat_material", liquid->splatMaterial, material_names, true, choice))
            setText("splat_material", choice);
        helpText("Shades splat geometry. Empty = the scene default. "
                 "★ This one is currently reported as failed on Apply: "
                 "updateFluidDomain has no slot for it yet.");
    }
    else if (auto* vol = dynamic_cast<VolumeMaterialNode*>(selected)) {
        int preset = vol->preset == "fire" ? 0 : 1;
        ImGui::TextUnformatted("Preset");
        if (ImGui::Combo("##preset", &preset, "fire\0smoke\0"))
            setText("preset", preset == 0 ? "fire" : "smoke");
        ImGui::Spacing();
        bool override_values = vol->overrideValues;
        if (ImGui::Checkbox("Override values", &override_values))
            setValue("override_values", override_values ? 1.0f : 0.0f);
        // ★★★ The reason this switch exists, said where it is switched. The
        // numeric fields carry struct defaults, not the preset's values; sending
        // them unconditionally would install the fire recipe and then overwrite
        // every number that makes it fire.
        helpText("Off: the preset alone is sent, and its own values survive. On: "
                 "the numbers below replace them — including the ones that make "
                 "the preset look like itself.");
        ImGui::BeginDisabled(!override_values);
        float v = vol->densityMultiplier;
        ImGui::TextUnformatted("Density multiplier");
        if (ImGui::DragFloat("##density_mul", &v, 0.01f, 0.0f, 50.0f)) setValue("density_multiplier", v);
        v = vol->densityCutoff;
        ImGui::TextUnformatted("Density cutoff");
        if (ImGui::DragFloat("##density_cut", &v, 0.001f, 0.0f, 1.0f)) setValue("density_cutoff", v);
        v = vol->temperatureMin;
        ImGui::TextUnformatted("Temperature min");
        if (ImGui::DragFloat("##temp_min", &v, 1.0f)) setValue("temperature_min", v);
        v = vol->temperatureMax;
        ImGui::TextUnformatted("Temperature max");
        if (ImGui::DragFloat("##temp_max", &v, 1.0f)) setValue("temperature_max", v);
        v = vol->scattering;
        ImGui::TextUnformatted("Scattering");
        if (ImGui::DragFloat("##scattering", &v, 0.01f, 0.0f, 10.0f)) setValue("scattering", v);
        v = vol->absorption;
        ImGui::TextUnformatted("Absorption");
        if (ImGui::DragFloat("##absorption", &v, 0.01f, 0.0f, 10.0f)) setValue("absorption", v);
        ImGui::EndDisabled();
    }
    else if (auto* cache = dynamic_cast<CacheNode*>(selected)) {
        char dir_buf[512];
        std::snprintf(dir_buf, sizeof(dir_buf), "%s", cache->cacheDir.c_str());
        ImGui::TextUnformatted("Cache directory");
        if (ImGui::InputText("##cache_dir", dir_buf, sizeof(dir_buf),
                             ImGuiInputTextFlags_EnterReturnsTrue)) {
            setText("cache_dir", dir_buf);
        }
        helpText("Empty = RAM timeline cache only. Press Enter to commit.");
        int frame = cache->startFrame;
        ImGui::TextUnformatted("Start frame");
        if (ImGui::DragInt("##start_frame", &frame, 1.0f, 0, 100000))
            setValue("start_frame", static_cast<float>(frame));
        frame = cache->endFrame;
        ImGui::TextUnformatted("End frame");
        if (ImGui::DragInt("##end_frame", &frame, 1.0f, 0, 100000))
            setValue("end_frame", static_cast<float>(frame));
        ImGui::Spacing();
        if (!cache->status.available) {
            ImGui::TextDisabled("cache state unread — Evaluate to refresh");
        } else if (cache->status.stale) {
            // ★★★ The sneakiest of the three states: it still serves frames, and
            // they describe a scene that no longer exists. Nothing else tells it
            // apart from a healthy cache.
            ImGui::TextColored(kBad, "STALE");
            helpText("A cache exists but was built from a different authored "
                     "config. It still serves frames — of a scene that is gone. "
                     "Re-bake, or clear it.");
        } else if (cache->status.baking) {
            ImGui::TextColored(kWarn, "baking...");
        } else if (cache->status.valid) {
            ImGui::TextColored(kGood, "valid (%u frames in RAM)", cache->status.ram_frames);
        } else {
            ImGui::TextDisabled("no cache");
        }
        helpText("This node does not bake. Baking walks the whole simulation, and "
                 "a graph evaluation has to stay instant — otherwise inspecting a "
                 "graph would run the sim. Bake with rt.sim_cache.bake().");
    }
    else {
        ImGui::TextDisabled("This node has no editable fields.");
    }

    ImGui::PopItemWidth();
}
