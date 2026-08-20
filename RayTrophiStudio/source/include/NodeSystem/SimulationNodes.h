/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          NodeSystem/SimulationNodes.h
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 * Simulation node layer — Faz N0 (contract) and N2 (read-only nodes).
 * Decision record: docs/dev/NODE_SIMULATION_ARCHITECTURE_PLAN.md, BOLUM D.
 *
 * ★★★ THIS IS NOT A SIMULATION CORE. It is a thin driver over the solvers that
 * already exist (APIC fluid, gas/combustion, MSF, foam, rigid). The layer is
 * DISCONTINUOUS: it owns no run loop, produces no frames, and holds no state.
 * If a node is inventing physics, the physics is in the wrong place.
 */

#pragma once

#include "NodeSystem/Graph.h"
#include "NodeSystem/Node.h"
#include "NodeSystem/EvaluationContext.h"

#include <memory>
#include <string>
#include <vector>

namespace NodeSystem {
namespace Sim {

// ============================================================================
// N0 — THE EXECUTION CONTRACT
// ============================================================================
//
// ★★★ GraphBase::evaluate() is PURE: it calls clearCache() and markAllDirty()
// so every run recomputes from inputs. That is right for geometry and material
// graphs, where the output IS a function of the inputs.
//
// It would be fatal here. In a solver the state IS the accumulated history, so
// "recompute from inputs" means "reset the simulation" — and the symptom would
// not be an error, it would be "the sim never advances", which is the hardest
// kind of bug this repository keeps paying for.
//
// The fix is NOT to give nodes their own state. Three rules instead:
//
//   1. A node OWNS NO STATE. State stays where it already lives
//      (ParticleSimulationSystem, grid domain states, MSF). A node refers to it
//      by IDENTITY — a name — which is also why no pin carries a handle.
//   2. Evaluating a node EMITS A COMMAND; it does not recompute. This mirrors
//      the rule the UI already follows: it RECORDS a SceneCommand rather than
//      executing one.
//   3. `dirty` means "re-apply configuration", NEVER "reset". A reset is an
//      explicit, visible action a user asks for.
//
// ★★ If a parameter genuinely requires a restart, the node must SAY SO and wait
// for confirmation (see SimNodeBase::requiresRestart). Resetting silently is the
// exact failure shape this codebase keeps rediscovering.

// One command the graph wants applied to the existing solvers. Values only:
// a name plus parameters, never a pointer into the scene.
struct SimCommand {
    enum class Kind : uint8_t {
        None = 0,
        BindDomain,      // "this graph drives the domain called <target>"
        SetParameter,    // override a solver parameter (N3)
        Couple,          // declare an ordered coupling (N4)
        BindObject,      // "this graph drives the object called <target>" (N5)
        SetSurface,      // override a per-object material/thermal setting (N5)
        SetRender,       // override a domain's LOOK (N7)
        SetEmitter,      // override a flow source's emission properties
    };
    Kind        kind = Kind::None;
    std::string target;      // domain / object name — the identity, not a handle
    // ★ Objects and domains are both named, and a name alone cannot say which
    // one it is. Without this the apply layer would look "Crate" up among the
    // domains, fail, and report "unknown domain" for an object that exists.
    enum class Scope : uint8_t { Domain = 0, Object, World };
    Scope       scope = Scope::Domain;
    std::string key;         // parameter or coupling name
    float       value = 0.0f;
    std::string text;        // string-valued parameter (substance name, mode)
    // Producer node, so a rejected command can be reported on the right node
    // instead of as a nameless graph error.
    uint32_t    sourceNode = 0;
};

// Base for every simulation node.
class SimNodeBase : public NodeBase {
public:
    // Commands produced by the last evaluation. The graph collects these; the
    // node never applies them itself. Keeping emission and application apart is
    // what lets an agent inspect a graph's intent WITHOUT running it.
    std::vector<SimCommand> commands;

    // ★ Does changing this node's configuration invalidate accumulated state?
    // Default false: most nodes re-apply cleanly mid-run. A node that answers
    // true is telling the UI to ask the user before discarding a simulation —
    // it must never act on that answer itself.
    virtual bool requiresRestart() const { return false; }

    // Human-readable reason shown next to the restart prompt. Empty when
    // requiresRestart() is false.
    virtual std::string restartReason() const { return {}; }

protected:
    void emit(SimCommand cmd) {
        cmd.sourceNode = id;
        commands.push_back(std::move(cmd));
    }
    void clearCommands() { commands.clear(); }
};

// ============================================================================
// SCOPES — which scene entity a graph belongs to
// ============================================================================
// Decision record: docs/dev/SIMULATION_NODE_OBJECT_MODEL.md, section 8 step 1.
//
// ★★★ There is no "the" simulation graph. A graph belongs to a NAMED scene
// entity, exactly the way material_node_graphs is keyed by material name, and
// the entity is the owner — the graph only reflects it. One global graph made
// every question ambiguous the moment a scene held two domains: "set the voxel
// size" had no answer to "of what".
//
// ★★ World carries no owner name: there is one world. Its map key is the empty
// string so the three storages can share one lookup shape.
enum class GraphScope : uint8_t { Object = 0, Domain, World };

inline const char* scopeName(GraphScope s) {
    switch (s) {
        case GraphScope::Object: return "object";
        case GraphScope::Domain: return "domain";
        case GraphScope::World:  return "world";
    }
    return "domain";
}

// ★ Name at the boundary. This is the ONLY place a scope string becomes an
// enum; nothing downstream compares scope strings.
inline bool parseScope(const std::string& text, GraphScope& out) {
    if (text == "object") { out = GraphScope::Object; return true; }
    if (text == "domain") { out = GraphScope::Domain; return true; }
    if (text == "world")  { out = GraphScope::World;  return true; }
    return false;
}

// The simulation graph. Differs from GraphBase in exactly one respect, and that
// difference is the whole of N0.
class SimulationNodeGraph : public GraphBase {
public:
    // Which scene entity this graph belongs to. Set once when the graph is
    // created and never edited: retargeting a graph is deleting one and making
    // another, because every node inside it names the old owner.
    GraphScope  scope = GraphScope::Domain;
    std::string owner;            // domain / object name; empty for World

    // ★★ The implicit OWNER NODE. Every scoped graph opens with one node naming
    // its owner, so the canvas answers "whose graph is this?" without a panel
    // caption, and so surface/parameter nodes have something to connect to on
    // an empty graph. It is created with the graph and re-created by clear();
    // 0 means the scope has no owner node (World — see section 7 item 1: the
    // world has no scripting surface yet, so its graph is honestly empty).
    uint32_t ownerNodeId = 0;

    bool isOwnerNode(uint32_t node_id) const {
        return ownerNodeId != 0 && node_id == ownerNodeId;
    }

    // Commands gathered from every node in the last evaluation, in topological
    // order. ORDER IS MEANINGFUL — for coupling nodes it IS the physics.
    std::vector<SimCommand> collected;

    // ★★★ Deliberately NOT calling clearCache() or markAllDirty(). See the
    // contract note above: those two lines are what make GraphBase pure, and
    // purity here would reset the solvers every evaluation.
    //
    // Nodes are still evaluated in dependency order, and `dirty` still
    // propagates downstream through markDirtyDownstream() — but it means
    // "re-apply your configuration", so a clean node may legitimately re-emit
    // its command without recomputing anything.
    void evaluateSimulation(EvaluationContext& ctx) {
        ctx.startTiming();
        ctx.clearErrors();
        collected.clear();

        for (auto& node : nodes) {
            if (ctx.isCancelled()) break;
            auto* sim = dynamic_cast<SimNodeBase*>(node.get());
            if (!sim || !sim->enabled) continue;
            sim->commands.clear();
        }

        // Terminal-first pull, same traversal GraphBase uses, so upstream nodes
        // are evaluated before their consumers and the command order matches the
        // dependency order rather than node creation order.
        for (auto& node : nodes) {
            if (ctx.isCancelled()) break;
            if (!isTerminalNode(node.get())) continue;
            if (!node->enabled) continue;
            if (node->outputs.empty()) {
                node->compute(0, ctx);
            } else {
                for (size_t i = 0; i < node->outputs.size(); ++i)
                    node->requestOutput(static_cast<int>(i), ctx);
            }
        }

        for (auto& node : nodes) {
            auto* sim = dynamic_cast<SimNodeBase*>(node.get());
            if (!sim || !sim->enabled) continue;
            collected.insert(collected.end(),
                             sim->commands.begin(), sim->commands.end());
        }
        ctx.setProgress(1.0f);
    }

    // Any node currently asking for a restart, for the UI prompt. Returning the
    // nodes rather than a bool keeps the prompt specific ("Substance on node 4
    // changes the material and restarts the sim") instead of generic.
    std::vector<const SimNodeBase*> nodesRequiringRestart() const {
        std::vector<const SimNodeBase*> out;
        for (const auto& node : nodes) {
            const auto* sim = dynamic_cast<const SimNodeBase*>(node.get());
            if (sim && sim->enabled && sim->requiresRestart()) out.push_back(sim);
        }
        return out;
    }
};

// ============================================================================
// N2 — READ-ONLY NODES
// ============================================================================
// These touch no solver. They exist so the node model can be seen and corrected
// on real data before anything is allowed to write.

// Names a fluid or gas domain. The output is the domain NAME: identity, not a
// handle, so it stays valid across rebuilds and means the same thing over IPC.
class DomainRefNode : public SimNodeBase {
public:
    std::string domainName;

    DomainRefNode() {
        metadata.typeId = "sim.domain_ref";
        metadata.displayName = "Domain";
        metadata.description =
            "NAMES a fluid or gas domain that ALREADY EXISTS in the scene. It "
            "does not create one -- a node owns no state, it refers by "
            "identity. Pick the name in the properties panel; if the list is "
            "empty, create the domain first. ";
        metadata.category = "Simulation";
        addOutput("Domain", DataType::DomainRef);
    }
    PinValue compute(int, EvaluationContext&) override {
        SimCommand cmd;
        cmd.kind = SimCommand::Kind::BindDomain;
        cmd.target = domainName;
        emit(std::move(cmd));
        return PinValue{ domainName };
    }

    // ★★★ The node's whole job is to NAME a domain, so the name has to be on
    // the canvas. Without it every Domain node in every graph looks identical
    // and the canvas cannot answer "which domain is this graph about?" —
    // reported by the user with two domains open and no way to tell them apart.
    bool wantsInlineContent() const override { return true; }
    void drawContent() override {
        if (domainName.empty()) {
            // ★ Not a styling choice: an unnamed Domain node resolves to
            // nothing and every node downstream of it silently does nothing.
            ImGui::TextDisabled("no domain picked");
        } else {
            ImGui::TextUnformatted(domainName.c_str());
        }
    }
};

// ★★ A Field is one of TWO things, and the distinction is representation, not
// performance (plan D.3):
//
//   Grid channel      — "density", "temperature", "velocity", "fuel", "flame",
//                       "sdf". Lives on the solver's voxel grid, GPU-resident,
//                       with NanoVDB topology and majorants. NOT an attribute:
//                       copying it into one would create a second representation
//                       of the same data, and parallel representations are a
//                       recurring source of silent failure here.
//
//   Element attribute — per-particle / per-texel / per-vertex data that ALREADY
//                       exists as parallel arrays (temperature, mass_fraction,
//                       substance_tag, granular_softening, granular_bond_scale,
//                       MSF melt/moisture/char). These are an attribute system
//                       already; what was missing is only the NAMING layer.
//
// ★★★ Name at the boundary, index in the loop. The name is resolved ONCE when
// the graph is compiled and becomes a stable index; a solver inner loop must
// never look up a string. Breaking that rule is how an attribute model dies.
class FieldReadNode : public SimNodeBase {
public:
    enum class Source : uint8_t { GridChannel = 0, ElementAttribute };
    Source source = Source::GridChannel;
    // Grid channel name, or the attribute name when source is ElementAttribute.
    std::string channel = "density";

    FieldReadNode() {
        metadata.typeId = "sim.field_read";
        metadata.displayName = "Field";
        metadata.description =
            "Names one channel of a domain: a grid channel (density, "
            "temperature, fuel, flame, sdf) or a per-element attribute. "
            "Produces an identity, reads nothing. ";
        metadata.category = "Simulation";
        addInput("Domain", DataType::DomainRef);
        addOutput("Field", DataType::Field);
    }
    PinValue compute(int, EvaluationContext& ctx) override {
        std::string domain = inputString(0, ctx);
        if (domain.empty()) return PinValue{};
        // Field identity is "<domain>:<channel>" — still just a name.
        return PinValue{ domain + ":" + channel };
    }

protected:
    // Small helper shared by the read-only nodes: pull an upstream string.
    std::string inputString(int index, EvaluationContext& ctx);
};

// Reports statistics about a field. This is the node that makes the graph
// TESTABLE from a script: it turns "looks about right" into a number.
class FieldInspectNode : public FieldReadNode {
public:
    // Filled in during evaluation; read back over IPC.
    //
    // ★ Doubles, not floats, and that is not pedantry: `substance_tag` is a
    // uint32 hash, and a float cannot hold one past 2^24. The first probe read
    // tag 1951804163 back as 1951804160 — an IDENTITY silently rounded into a
    // different identity. A double represents every uint32 exactly.
    struct Stats {
        bool  available = false;
        uint32_t particle_count = 0;   // elements actually measured
        uint32_t array_size = 0;       // raw array length; see `in_sync`
        bool  in_sync = true;          // array_size == the domain's particle count
        double min_value = 0.0;
        double max_value = 0.0;
        double mean_value = 0.0;
    } stats;

    FieldInspectNode() {
        metadata.typeId = "sim.field_inspect";
        metadata.displayName = "Field Inspect";
        metadata.description =
            "Measures a per-element attribute over the live particles and "
            "reports min, max and mean. This is what turns \"looks right\" into a "
            "number. ";
        metadata.category = "Simulation";
        source = Source::ElementAttribute;
        channel = "temperature";
        addOutput("Mean", DataType::Float);
    }
    PinValue compute(int outputIndex, EvaluationContext& ctx) override;
};

// ============================================================================
// N3 — PARAMETER BINDING
// ============================================================================

// Writes one solver parameter as an OVERRIDE. The authored value is never
// mutated: the apply layer captures it before the first write and restores it
// when overrides are cleared. That is plan B.5's decision — MSF and solver
// configuration are runtime state and must stay resettable to frame 0; writing
// them into the authored data makes the change irreversible.
class SetParameterNode : public SimNodeBase {
public:
    std::string key;              // "kinematic_viscosity", "voxel_size", ...
    float       value = 0.0f;

    SetParameterNode() {
        metadata.typeId = "sim.set_parameter";
        metadata.displayName = "Set Parameter";
        metadata.description =
            "Overrides one solver parameter on the incoming domain. Reversible: "
            "the authored value is captured before the first write and restored "
            "by Clear Overrides. Chain several to configure a domain. ";
        metadata.category = "Simulation";
        addInput("Domain", DataType::DomainRef);
        addOutput("Domain", DataType::DomainRef);   // pass-through, so nodes chain
    }

    // ★★ Some parameters cannot be changed without discarding accumulated state
    // — a grid resolution change reallocates the field the simulation lives in.
    // The node SAYS SO; it never decides. simGraphApply() refuses these unless
    // the caller explicitly allows a restart, so a user's running simulation is
    // never thrown away by a graph edit alone.
    static bool keyRequiresRestart(const std::string& k) {
        return k == "voxel_size";
    }
    bool requiresRestart() const override { return keyRequiresRestart(key); }
    std::string restartReason() const override {
        if (!requiresRestart()) return {};
        return "'" + key + "' changes the grid discretisation; applying it "
               "discards the accumulated simulation state";
    }

    PinValue compute(int, EvaluationContext& ctx) override;
};

// ============================================================================
// DOMAIN ASPECT NODES — one entity, several aspects
// ============================================================================
// Decision record: SIMULATION_NODE_OBJECT_MODEL.md section 5 and section 8 step 3.
//
// ★★★ Splitting a domain across several nodes is legitimate BECAUSE a node is a
// reflection, not an authority: identity/bounds, discretisation/solver, cache
// and look are different aspects of one scene entity, and the material graph
// already splits a material the same way. Deleting one of these nodes removes a
// view, never the domain.
//
// ★★★ EVERY FIELD IS OPT-IN. A node that wrote all of its fields on every apply
// would flatten authored values the user never touched, and the override layer
// would faithfully "restore" numbers this graph had invented. Only fields whose
// `use` flag is set emit anything.
class DomainParamNodeBase : public SimNodeBase {
public:
    // ★ `key` is the SAME string readParameter/writeParameter use. One
    // vocabulary end to end: a second spelling here would work in the panel and
    // fail over IPC, which this repository has already paid for once.
    struct Field {
        const char* key;
        bool        use = false;
        float       value = 0.0f;
    };
    std::vector<Field> fields;

    DomainParamNodeBase() {
        metadata.category = "Simulation";
        addInput("Domain", DataType::DomainRef);
        addOutput("Domain", DataType::DomainRef);
    }

    Field* find(const std::string& key) {
        for (auto& f : fields)
            if (key == f.key) return &f;
        return nullptr;
    }

    // ★★★ The canvas must say what this node DOES, not just what type it is.
    // A row of identical boxes labelled "Solver" forces a click to learn whether
    // any of them overrides anything — which makes the graph decorative and
    // pushes the real state into a side panel. The title carries the summary
    // because the node editor draws no body content.
    //
    // ★★ "(nothing)" is the important case: with every field opt-in, a node that
    // overrides nothing looks exactly like one that overrides everything. That
    // is the single most confusing state this model can produce, so it is the
    // one the title states outright.
    // Which word the title starts with. Virtual so one call site can refresh
    // any aspect node without knowing which kind it holds.
    virtual const char* titleWord() const { return "Parameters"; }
    void refreshTitle() { refreshTitle(titleWord()); }
    void refreshTitle(const char* base) {
        size_t count = 0;
        for (const auto& f : fields) if (f.use) ++count;
        // ★ Short: the field NAMES and values are drawn in the body below. A
        // title long enough to list them is truncated by the node width, which
        // is how "Solver: kinematic_viscosity, vi..." ends up telling nobody
        // anything.
        metadata.displayName = std::string(base) +
            (count == 0 ? ": nothing"
                        : ": " + std::to_string(count) + " override" +
                          (count == 1 ? "" : "s"));
    }

    // ★★★ THE CANVAS MUST SHOW WHAT THE NODE DOES.
    //
    // Reported by the user against the first cut: a row of boxes reading only
    // "Solver", with every value hidden behind a click into a side panel. That
    // makes the graph decorative — if the parameters only ever appear in a
    // panel, the node is a label and the panel is the tool, and the question
    // "why a node graph at all?" answers itself.
    //
    // NodeBase already supports this (wantsInlineContent/drawContent); the
    // default is off, so no other graph family changes shape.
    bool wantsInlineContent() const override { return true; }
    void drawContent() override {
        bool any = false;
        for (const auto& f : fields) {
            if (!f.use) continue;
            any = true;
            ImGui::TextDisabled("%s", f.key);
            ImGui::SameLine();
            // %g: a cell size of 0.05 and a sweep count of 12 both have to read
            // correctly, and a fixed precision makes one of them a lie.
            ImGui::Text("%g", static_cast<double>(f.value));
        }
        // ★★ The empty case is the one worth drawing. With every field opt-in,
        // a node that overrides nothing is invisible otherwise — it looks
        // exactly like one that overrides everything.
        if (!any) ImGui::TextDisabled("no overrides");
    }

    // ★★ Only a field that is actually IN USE can demand a restart. Asking the
    // user to discard a simulation for a voxel_size dial they never enabled is
    // how a restart prompt becomes noise that gets clicked through.
    bool requiresRestart() const override {
        for (const auto& f : fields)
            if (f.use && SetParameterNode::keyRequiresRestart(f.key)) return true;
        return false;
    }
    std::string restartReason() const override {
        for (const auto& f : fields) {
            if (f.use && SetParameterNode::keyRequiresRestart(f.key))
                return std::string("'") + f.key + "' changes the grid "
                       "discretisation; applying it discards the accumulated "
                       "simulation state";
        }
        return {};
    }

    PinValue compute(int, EvaluationContext& ctx) override;
};

// Discretisation and solver settings. Separate from the domain's identity
// because they answer a different question: not "which region" but "how it is
// solved". That separation was invisible before, and the invisibility cost a
// bake failure — nothing showed which grid a domain was actually running.
class SolverNode : public DomainParamNodeBase {
public:
    SolverNode() {
        metadata.typeId = "sim.solver";
        metadata.displayName = "Solver";
        metadata.description =
            "How the domain is DISCRETISED and solved: cell size, viscosity "
            "sweeps, granular strength. It creates no domain and chooses no "
            "solver -- it overrides settings on the domain wired into it. "
            "Every field is opt-in: untouched fields keep their authored value. ";
        fields = {
            {"voxel_size"},
            {"kinematic_viscosity"},
            {"viscosity_sweeps"},
            {"viscosity_wall_slip"},
            {"granular_friction_angle_degrees"},
            {"granular_cohesion"},
        };
        refreshTitle("Solver");
    }
    const char* titleWord() const override { return "Solver"; }
};

// The domain's own switches: is it running, is it drawn, how its surface and
// porosity are resolved.
//
// ★ granular_enabled is deliberately absent — see writeParameter. Whether
// toggling it invalidates accumulated state has not been measured, and a dial
// whose restart semantics are guessed is worse than a missing one.
class DomainSettingsNode : public DomainParamNodeBase {
public:
    DomainSettingsNode() {
        metadata.typeId = "sim.domain_settings";
        metadata.displayName = "Domain Settings";
        metadata.description =
            "The domain's own switches and surface settings: enabled, visible, "
            "solid phase, porosity, surface offset. Like every node here it "
            "REFERS -- it cannot create or delete a domain. Every field is "
            "opt-in: untouched fields keep their authored value. ";
        fields = {
            {"enabled"},
            {"visible"},
            {"solid_phase"},
            {"solid_phase_fill"},
            {"surface_offset_voxels"},
            {"pore_amount"},
            {"pore_scale"},
            {"pore_detail"},
            {"uvw_refresh_period"},
        };
        refreshTitle("Domain Settings");
    }
    const char* titleWord() const override { return "Domain Settings"; }
};

// ── WORLD THERMAL ────────────────────────────────────────────────────────────
// Decision record: docs/dev/SIMULATION_NODE_OBJECT_MODEL.md section 7 item 1
// (closed) and section 8 step 4.
//
// ★★★ NOT a DomainParamNodeBase: that base wires a Domain pin in and out so
// aspect nodes can chain onto the domain they configure. There is exactly one
// world, so there is nothing to name and nothing to chain — the node's
// presence in the World graph IS the reference, the same way DomainRefNode's
// presence names a domain. No pins at all.
//
// Same opt-in contract as every other aspect node: an unticked field is a
// parameter this graph has no opinion about, not a zero. And the same
// asymmetry as everywhere else this ambient value appears — a domain's own
// `thermal_override_enabled` (SimulationGridDomainDesc) still wins where a
// domain has set it; this node changes only the default a domain has NOT
// overridden.
class WorldThermalNode : public SimNodeBase {
public:
    struct Field {
        const char* key;
        bool        use = false;
        float       value = 0.0f;
    };
    std::vector<Field> fields;

    WorldThermalNode() {
        metadata.typeId = "sim.world_thermal";
        metadata.displayName = "World Thermal";
        metadata.description =
            "The ambient thermal condition every uncoupled substance relaxes "
            "toward: room temperature, the Kelvin-per-unit calibration, "
            "passive cooling and the pyrolysis oxygen scale. A domain's own "
            "thermal override still wins where it is set -- this changes only "
            "the default a domain has not overridden. Every field is opt-in. ";
        metadata.category = "Simulation";
        fields = {
            {"ambient_kelvin"},
            {"kelvin_per_unit"},
            {"convection_coefficient"},
            {"oxygen_availability"},
        };
        refreshTitle();
    }

    Field* find(const std::string& key) {
        for (auto& f : fields)
            if (key == f.key) return &f;
        return nullptr;
    }

    void refreshTitle() {
        size_t count = 0;
        for (const auto& f : fields) if (f.use) ++count;
        metadata.displayName = std::string("World Thermal") +
            (count == 0 ? ": nothing"
                        : ": " + std::to_string(count) + " override" +
                          (count == 1 ? "" : "s"));
    }

    bool wantsInlineContent() const override { return true; }
    void drawContent() override {
        bool any = false;
        for (const auto& f : fields) {
            if (!f.use) continue;
            any = true;
            ImGui::TextDisabled("%s", f.key);
            ImGui::SameLine();
            ImGui::Text("%g", static_cast<double>(f.value));
        }
        if (!any) ImGui::TextDisabled("no overrides");
    }

    // World keys never require a restart today -- see
    // SetParameterNode::keyRequiresRestart, which this shares.
    bool requiresRestart() const override {
        for (const auto& f : fields)
            if (f.use && SetParameterNode::keyRequiresRestart(f.key)) return true;
        return false;
    }

    PinValue compute(int, EvaluationContext&) override {
        for (const auto& f : fields) {
            if (!f.use) continue;
            SimCommand cmd;
            cmd.kind = SimCommand::Kind::SetParameter;
            cmd.scope = SimCommand::Scope::World;
            // No target: there is exactly one world, so there is no name to
            // carry. The apply layer keys its authored-value capture on scope
            // alone for World commands.
            cmd.key = f.key;
            cmd.value = f.value;
            emit(std::move(cmd));
        }
        return PinValue{};
    }
};

// ── EMITTER ─────────────────────────────────────────────────────────────────
//
// ★★★ Emission is a property of MATTER (section 9.1): "this object emits
// substance S at rate R". The flow source's explicit `domain` field is not a
// property but a RESOLUTION — an object sitting inside two overlapping domains
// has no unambiguous geometric answer, and this repository has already paid for
// overlapping volume boxes. So this node edits the emission properties and
// deliberately does NOT touch the binding: retargeting which domain a source
// feeds is a different decision, made where the source is authored.
//
// ★★★ IT TAKES A DOMAIN INPUT, and emits NOTHING when that pin is empty.
// The first cut had no pins at all, on the reasoning that a flow source already
// knows which domain it feeds. The result was visible on the canvas: an Emitter
// sitting unconnected, wired to nothing, still emitting commands. A node whose
// EFFECT does not follow its WIRING makes the graph decorative — the reader
// sees no cause and the solver sees one — and "the panel disagrees with the
// core" is the most expensive defect class in this repository. Connection is
// causation here, so an unwired Emitter must do nothing.
class EmitterNode : public SimNodeBase {
public:
    std::string emitterName;

    // Opt-in, same contract as the domain aspect nodes: a field this graph has
    // no opinion about must not overwrite an authored value with a default.
    struct Field {
        const char* key;
        bool        use = false;
        float       value = 0.0f;
    };
    std::vector<Field> fields;

    // Text-valued, and separate because an empty string is a LEGITIMATE value
    // (no substance override) — so "unset" cannot be encoded as emptiness.
    bool        useSubstance = false;
    std::string substance;

    EmitterNode() {
        metadata.typeId = "sim.emitter";
        metadata.displayName = "Emitter";
        metadata.description =
            "NAMES an existing flow source and overrides what it emits: rate, "
            "radius, temperature, fuel, substance. It does not create a source "
            "and does not change which domain the source feeds -- that binding "
            "resolves an ambiguity and is authored on the source itself. Every "
            "field is opt-in. Wire the Domain pin: an unconnected Emitter emits "
            "nothing, because a node that acted without being wired would make "
            "the graph unreadable. ";
        metadata.category = "Simulation";
        addInput("Domain", DataType::DomainRef);
        addOutput("Domain", DataType::DomainRef);
        fields = {
            {"enabled"},
            {"radius"},
            {"density"},
            {"temperature"},
            {"fuel"},
            {"falloff"},
            {"velocity_coupling"},
            {"inherit_velocity"},
            {"fluid_particles_per_second"},
            {"fluid_velocity_spread"},
        };
    }

    Field* find(const std::string& key) {
        for (auto& f : fields)
            if (key == f.key) return &f;
        return nullptr;
    }

    // Same reasoning as the aspect nodes: the canvas names the flow source and
    // says whether anything is actually overridden.
    void refreshTitle() {
        metadata.displayName =
            emitterName.empty() ? "Emitter" : ("Emitter [" + emitterName + "]");
    }

    bool wantsInlineContent() const override { return true; }
    void drawContent() override {
        if (emitterName.empty()) {
            // ★ Not "no overrides" — an unnamed emitter drives nothing at all,
            // and saying the milder thing would hide the real problem.
            ImGui::TextDisabled("no flow source picked");
            return;
        }
        bool any = false;
        for (const auto& f : fields) {
            if (!f.use) continue;
            any = true;
            ImGui::TextDisabled("%s", f.key);
            ImGui::SameLine();
            ImGui::Text("%g", static_cast<double>(f.value));
        }
        if (useSubstance) {
            any = true;
            ImGui::TextDisabled("substance");
            ImGui::SameLine();
            // Empty is a legitimate substance value, so it is shown as such
            // rather than as an absence.
            ImGui::Text("%s", substance.empty() ? "(none)" : substance.c_str());
        }
        if (!any) ImGui::TextDisabled("no overrides");
    }

    PinValue compute(int, EvaluationContext&) override;
};

// ============================================================================
// N4 — COUPLING NODES
// ============================================================================
//
// ★★★ THE POINT OF THIS PHASE IS ORDER, and order is the one thing the graph
// must not lie about. Today the coupling order is implicit inside
// stepGridDomains: fluid combustion runs after the gas snapshot upload, foam
// after the density splat, wetting after both. "Producer ≠ consumer" is a
// recurring failure class here precisely because that order is invisible.
//
// So a coupling node DECLARES a coupling. It does not schedule one. The solver
// reports which couplings actually ran and in what order (see
// ParticleSimulationSystem::couplingTrace), and the API compares declaration
// against reality. A graph that showed a user-chosen order while the solver ran
// a different one would be worse than no graph at all: it would look like
// control.
//
// ★★ Coupling knobs go through the SAME reversible override layer as N3. A
// coupling is configuration, and configuration must stay resettable to frame 0.
class CouplingNodeBase : public SimNodeBase {
public:
    // Stable coupling id. Matches the name the solver records in its trace, and
    // that match is checked — a node whose id no consumer knows would silently
    // declare nothing.
    virtual const char* couplingId() const = 0;

    // ★★ A coupling is ON or OFF here, and nothing else. It deliberately carries
    // no strength dial: there is no single authored gain behind these couplings,
    // so a 0..1 "strength" would have had to invent one — and a dial that scales
    // nothing real is the exact thing this project tests for elsewhere ("the
    // dead dials stay dead"). Coupling PARAMETERS are set by chaining N3 Set
    // Parameter nodes after this one, through the reversible override layer.
    //
    // `active = false` is not the same as deleting the node: it DECLARES that
    // this coupling should be off, which is a statement the apply layer acts on.
    // A disabled node (NodeBase::enabled) is not evaluated at all and declares
    // nothing.
    bool active = true;

    CouplingNodeBase() {
        metadata.category = "Simulation";
        addInput("Source", DataType::DomainRef);
        addInput("Target", DataType::DomainRef);
        addOutput("Source", DataType::DomainRef);   // pass-through, so a chain
                                                    // of couplings has an order
    }

    PinValue compute(int, EvaluationContext& ctx) override;
};

// A burning liquid surface releasing heat, smoke and fuel into a gas domain.
// Maps onto the combustible-fluid settings that already exist; invents nothing.
class FluidToGasCouplingNode : public CouplingNodeBase {
public:
    FluidToGasCouplingNode() {
        metadata.typeId = "sim.couple_fluid_to_gas";
        metadata.displayName = "Fluid to Gas";
        metadata.description =
            "Declares that a burning liquid surface feeds heat, smoke and fuel "
            "into a gas domain. Source = the fluid domain, Target = the gas "
            "domain. ";
    }
    const char* couplingId() const override { return "fluid_to_gas"; }
};

// Gas temperature igniting the liquid surface. Separate node from the one
// above, and deliberately so: they are opposite directions, they run at
// different points in the step, and merging them would hide exactly the
// ordering this phase exists to expose.
class GasToFluidIgnitionNode : public CouplingNodeBase {
public:
    GasToFluidIgnitionNode() {
        metadata.typeId = "sim.couple_gas_ignition";
        metadata.displayName = "Gas Ignites Fluid";
        metadata.description =
            "Declares that gas temperature ignites the liquid surface -- the "
            "opposite direction to Fluid to Gas, and a separate node because it "
            "runs at a different point in the step. ";
    }
    const char* couplingId() const override { return "gas_to_fluid_ignition"; }
};

// Whitewater generated from the fluid it belongs to.
class FoamFromFluidNode : public CouplingNodeBase {
public:
    FoamFromFluidNode() {
        metadata.typeId = "sim.couple_foam";
        metadata.displayName = "Foam From Fluid";
        metadata.description =
            "Declares whitewater generation from the fluid it belongs to. "
            "Source and Target are the same fluid domain. ";
    }
    const char* couplingId() const override { return "foam_from_fluid"; }
};

// ============================================================================
// N5 — SUBSTANCE AND CHEMISTRY
// ============================================================================
//
// These nodes drive the Material State Field: what an object is MADE OF, and
// what it does when it gets hot. Every knob below already exists on
// ParticleColliderDesc and is derived from real physical constants in Kelvin —
// no node here invents a number.
//
// ★★ TWO PLANNED NODES ARE DELIBERATELY ABSENT, and their absence is the
// honest answer rather than an omission:
//
//   Moisture — has NO authored knob. Moisture is WRITTEN by fluid contact
//     (Phase 5) and evaporates against the ambient. A "Moisture node" would
//     have to invent an authoring surface the solver does not have, which is
//     precisely the failure D.0 forbids: a node inventing physics. It is
//     exposed as a READING instead, through Surface Inspect.
//
//   Thermal Transfer — WorldThermalState (ambient Kelvin, oxygen, convection)
//     now HAS a scripting surface (rt.world.get_thermal/set_thermal) and a
//     node (WorldThermalNode, in the WORLD ASPECT NODE section below, next to
//     the other domain/world aspect nodes rather than here among the N5
//     object-scoped ones — it configures no object).
//
// Char / Ash stays out too: plan Faz 7 has not landed.

// Names a simulation OBJECT. Output is the object name — identity, not a
// handle, exactly like DomainRefNode. Typed as SurfaceField because that is
// what the object contributes to a graph: its per-surface material state.
class ObjectRefNode : public SimNodeBase {
public:
    std::string objectName;

    ObjectRefNode() {
        metadata.typeId = "sim.object_ref";
        metadata.displayName = "Object";
        metadata.description =
            "NAMES a scene object that already exists, and contributes its "
            "per-surface material state (MSF) to the graph. Like Domain, it "
            "refers -- it creates nothing. ";
        metadata.category = "Simulation";
        addOutput("Surface", DataType::SurfaceField);
    }
    PinValue compute(int, EvaluationContext&) override;

    // Same reason as DomainRefNode: identity belongs on the canvas.
    bool wantsInlineContent() const override { return true; }
    void drawContent() override {
        if (objectName.empty()) ImGui::TextDisabled("no object picked");
        else ImGui::TextUnformatted(objectName.c_str());
    }
};

// Base for the object-scoped setters. They all take a surface in, pass it out,
// and write through the same reversible override layer as N3 — a substance
// change is configuration, and configuration stays resettable to frame 0.
class SurfaceNodeBase : public SimNodeBase {
public:
    SurfaceNodeBase() {
        metadata.category = "Simulation";
        addInput("Surface", DataType::SurfaceField);
        addOutput("Surface", DataType::SurfaceField);
    }

protected:
    // Returns the bound object name, or empty when the pin is unconnected.
    std::string boundObject(EvaluationContext& ctx);
    void emitSurface(const std::string& object, const char* key, float value);
    void emitSurfaceText(const std::string& object, const char* key,
                         const std::string& text);
};

// What the object is made of. The name resolves through the built-in substance
// library; the three scales are the per-object deviation from it.
//
// ★ The override is authored in KELVIN and applied BEFORE the normalized
// conversion, so the system still has exactly one Kelvin->normalized point.
class SubstanceNode : public SurfaceNodeBase {
public:
    std::string substanceName = "Wood (Oak)";
    bool  overrideIgnition = false;
    float ignitionKelvin = 573.0f;
    float burnRateScale = 1.0f;
    float fuelCapacityScale = 1.0f;

    SubstanceNode() {
        metadata.typeId = "sim.substance";
        metadata.displayName = "Substance";
        metadata.description =
            "What the incoming object is MADE OF. The name resolves through the "
            "built-in substance library; the scales are this object's deviation "
            "from it. ";
    }
    PinValue compute(int, EvaluationContext& ctx) override;
};

// Whether this object pyrolyses on contact with flame at all. One switch,
// because that is what the solver actually has (`gas_ignite_on_contact`).
class PyrolysisNode : public SurfaceNodeBase {
public:
    bool active = true;

    PyrolysisNode() {
        metadata.typeId = "sim.pyrolysis";
        metadata.displayName = "Pyrolysis";
        metadata.description =
            "Whether the incoming object pyrolyses on contact with flame at "
            "all. One switch, because that is what the solver actually has. ";
    }
    PinValue compute(int, EvaluationContext& ctx) override;
};

// Melting: whether molten material flows, and how far it slumps. These are the
// authored knobs behind Faz 6c, not new physics.
//
// ★ It does NOT expose a melting point. That comes from the substance (iron
// melts at 1811 K because iron does), and putting a second melting point on a
// node would let a graph disagree with the material it is made of.
class PhaseChangeNode : public SurfaceNodeBase {
public:
    bool  meltFlow = true;
    float heightLoss = 0.85f;
    float spread = 1.50f;

    PhaseChangeNode() {
        metadata.typeId = "sim.phase_change";
        metadata.displayName = "Phase Change";
        metadata.description =
            "Melting behaviour for the incoming object: whether molten material "
            "flows and how far it slumps. The melting POINT comes from the "
            "substance, not here. ";
    }
    PinValue compute(int, EvaluationContext& ctx) override;
};

// Reads per-element MSF state for one object: temperature, char, melt,
// moisture, fuel.
//
// ★★ This is the naming layer from D.3 reaching per-TEXEL data, and it needed
// no new storage either: MSF already keeps parallel arrays. Same FieldStats,
// same "false means unmeasured, not zero" rule as Field Inspect — the only
// difference is that the identity is an object instead of a domain.
class SurfaceInspectNode : public SimNodeBase {
public:
    std::string channel = "temperature";
    struct Stats {
        bool  available = false;
        uint32_t particle_count = 0;   // elements measured
        uint32_t array_size = 0;
        bool  in_sync = true;
        bool  host_fresh = true;       // see FieldStats::host_fresh
        double min_value = 0.0;
        double max_value = 0.0;
        double mean_value = 0.0;
    } stats;

    SurfaceInspectNode() {
        metadata.typeId = "sim.surface_inspect";
        metadata.displayName = "Surface Inspect";
        metadata.description =
            "Measures per-texel MSF state on the incoming object -- "
            "temperature, char, melt, moisture, fuel. Reports whether the "
            "reading is fresh. ";
        metadata.category = "Simulation";
        addInput("Surface", DataType::SurfaceField);
        addOutput("Mean", DataType::Float);
    }
    PinValue compute(int outputIndex, EvaluationContext& ctx) override;
};

// ============================================================================
// N7 — RENDER BINDING
// ============================================================================
//
// These bind EXISTING shader parameters. No new shader, no new look — plan B.5:
// "shader graph and GPU fusion are not in the first release; existing shader
// parameters get bound to Material nodes first."
//
// ★★ Two planned nodes are absent, and for the same reason as in N5 — the thing
// they would author does not exist to be authored:
//
//   Foam Material — `FoamParams::foam_material_id` has no scripting surface,
//     exactly like the foam coupling switch in N4. Same missing slice, and it
//     should be filled once rather than faked twice.
//
//   Char / Molten Surface — has no knob of its own: char colour and molten
//     emission are DERIVED from the substance (that is why iron glows like iron).
//     A node with its own char colour would let a graph disagree with the
//     material the object is made of, which is the same mistake as putting a
//     melting point on Phase Change.

class RenderNodeBase : public SimNodeBase {
public:
    RenderNodeBase() {
        metadata.category = "Simulation";
        addInput("Domain", DataType::DomainRef);
        addOutput("Domain", DataType::DomainRef);
    }

protected:
    std::string boundDomain(EvaluationContext& ctx);
    void emitRender(const std::string& domain, const char* key, float value);
    void emitRenderText(const std::string& domain, const char* key,
                        const std::string& text);
};

// The liquid's look: which material shades the SDF isosurface, and which shades
// splat geometry. Both are NAMES — material ids shift as a scene is edited, and
// an id in a graph is a number nobody can check.
class LiquidMaterialNode : public RenderNodeBase {
public:
    std::string surfaceMaterial;   // empty = built-in dielectric
    std::string splatMaterial;     // empty = scene default

    LiquidMaterialNode() {
        metadata.typeId = "sim.material_liquid";
        metadata.displayName = "Liquid Material";
        metadata.description =
            "Which materials shade the incoming liquid domain: one for the SDF "
            "isosurface, one for splat geometry. Both by NAME. ";
    }
    PinValue compute(int, EvaluationContext& ctx) override;
};

// Smoke AND fire, in one node — deliberately.
//
// ★★ The plan listed "Smoke Volume" and "Fire / Blackbody" as separate nodes,
// but they are one authored struct (GasShaderSettings) with a preset that
// selects between them. Two nodes writing the same struct would let a graph hold
// both, and the second would silently overwrite the first with no error and no
// visible cause. One node, one preset field.
class VolumeMaterialNode : public RenderNodeBase {
public:
    std::string preset = "smoke";      // "fire" | "smoke"
    // ★★★ Off by default, and that is the whole reason this flag exists. The
    // numeric fields below carry arbitrary struct defaults, not the preset's
    // values — emitting them unconditionally would install the fire recipe and
    // then overwrite every characteristic number it has (blackbody intensity,
    // the 800..1900 K emission range) with these placeholders. The preset would
    // become a label that changes nothing visible.
    //
    // So: preset alone by default; the sliders speak only when asked to.
    bool overrideValues = false;
    float densityMultiplier = 1.0f;
    float densityCutoff = 0.01f;
    float temperatureMin = 0.0f;
    float temperatureMax = 1.0f;
    float scattering = 0.5f;
    float absorption = 0.5f;

    VolumeMaterialNode() {
        metadata.typeId = "sim.material_volume";
        metadata.displayName = "Volume Material";
        metadata.description =
            "Smoke and fire look for the incoming gas domain. The preset "
            "selects the recipe; the numeric fields stay silent unless Override "
            "values is on. ";
    }
    PinValue compute(int, EvaluationContext& ctx) override;
};

// ============================================================================
// N6 — CACHE / BAKE
// ============================================================================
//
// ★★★ THIS NODE DOES NOT BAKE. Baking walks the whole simulation and takes as
// long as it takes; a graph EVALUATION must stay instant and side-effect free,
// or every inspection of the graph would run the sim. So the node declares the
// intent and reports the state, and the bake itself is an explicit action
// (`sim_cache.bake`) the user or a script invokes — which is also plan B.5's
// answer: "user-controlled bake plus automatic signature validation".
//
// ★★ What it adds over reading `sim_cache.status` directly is the SIGNATURE
// comparison in context: a cache that still exists but was built from a
// different authored config is the failure that silently renders stale physics,
// and it looks exactly like a healthy cache from the outside.
class CacheNode : public SimNodeBase {
public:
    std::string cacheDir;          // empty = RAM timeline cache only
    int startFrame = 0;
    int endFrame = 100;

    struct Status {
        bool     available = false;   // false = could not read, NOT "no cache"
        bool     valid = false;
        bool     baking = false;
        uint32_t ram_frames = 0;
        bool     has_range = false;
        int      first_frame = 0;
        int      last_frame = 0;
        uint64_t config_signature = 0;
        // ★ True when a cache exists but the authored config has moved on since
        // it was built. NOT the same as "no cache": this one still serves
        // frames, and they describe a scene that no longer exists.
        bool     stale = false;
    } status;

    CacheNode() {
        metadata.typeId = "sim.cache";
        metadata.displayName = "Cache";
        metadata.description =
            "Reports the bake state of the incoming domain and compares the "
            "cache signature against the authored config. It does NOT bake -- "
            "baking is an explicit action, so that inspecting a graph never "
            "runs the simulation. ";
        metadata.category = "Simulation";
        addInput("Domain", DataType::DomainRef);
        addOutput("Domain", DataType::DomainRef);
    }
    PinValue compute(int, EvaluationContext& ctx) override;
};

// Seam for the cache reading, installed by the application like the others.
struct CacheStatusReading {
    bool     valid = false;
    bool     baking = false;
    uint32_t ram_frames = 0;
    bool     has_range = false;
    int      first_frame = 0;
    int      last_frame = 0;
    uint64_t config_signature = 0;
    uint64_t live_signature = 0;   // what the scene hashes to RIGHT NOW
};
using CacheStatusFn = bool (*)(CacheStatusReading& out);
void setCacheStatusResolver(CacheStatusFn fn);

// ============================================================================
// DATA ACCESS SEAM
// ============================================================================
//
// ★ The node layer must not reach into the engine, and the engine must not
// depend on the node layer. So reading is done through a resolver the
// application installs at startup (RtApiSimNodes.cpp). Nodes stay drivers; this
// header stays free of solver includes.
//
// ★★ The resolver returns FALSE when it cannot measure, and the caller must
// keep `available` false rather than reporting zeros. A zero that means "I could
// not read this" is indistinguishable from a real zero, and that confusion has
// cost this project entire debugging rounds.
// ★★ `count` is what was MEASURED and it is the domain's particle count, never
// the raw array length. Those two disagreeing is a real condition in this engine
// (a granular array outlived six removed particles on 2026-08-17), and averaging
// over the dead tail produced a number that looked perfectly reasonable. The
// mismatch is reported, not smoothed away.
struct FieldStats {
    uint32_t count = 0;         // elements measured (= the domain's particle count)
    uint32_t array_size = 0;    // raw length of the backing array
    bool     in_sync = true;    // array_size == count
    // ★★★ Surface (MSF) readings only: the numbers come from the HOST MIRROR,
    // and during a step the DEVICE copy is authoritative. False means this
    // mirror has not been refreshed since the last dispatch, so the values are
    // the previous readback — or, before any readback at all, the INITIALISATION
    // values. Measured 2026-08-17: a fresh crate reported fuel_remaining = -1
    // across 99072 elements, which is the sentinel for "not seeded yet" and not
    // a fuel level. Reporting that as a measurement is the exact shape of "a
    // default is not a measurement".
    bool     host_fresh = true;
    double min_value = 0.0;
    double max_value = 0.0;
    double mean_value = 0.0;
};

// domain name + attribute name -> stats. Attribute names are the per-particle
// arrays that already exist ("temperature", "mass_fraction", "substance_tag",
// "granular_softening", "granular_bond_scale", "granular_damage", ...).
using AttributeStatsFn =
    bool (*)(const std::string& domain, const std::string& attribute,
             FieldStats& out);
void setAttributeStatsResolver(AttributeStatsFn fn);

// Names the resolver can answer for, so a script can DISCOVER what exists
// instead of reading the source. This is the whole point of the naming layer.
using AttributeListFn = std::vector<std::string> (*)(const std::string& domain);
void setAttributeListResolver(AttributeListFn fn);
std::vector<std::string> listAttributes(const std::string& domain);

// Same seam again, for per-object surface (MSF) attributes. Separate resolver
// rather than a scope flag on the one above: the two read completely different
// systems, and a single function that branched on a string would be one place
// to accidentally answer a domain question with object data.
using SurfaceStatsFn =
    bool (*)(const std::string& object, const std::string& attribute,
             FieldStats& out);
void setSurfaceStatsResolver(SurfaceStatsFn fn);
using SurfaceListFn = std::vector<std::string> (*)(const std::string& object);
void setSurfaceListResolver(SurfaceListFn fn);
std::vector<std::string> listSurfaceAttributes(const std::string& object);

// Registers every simulation node type with NodeRegistry. Called once at
// startup, alongside the geometry/material/terrain registrations.
void registerSimulationNodeTypes();

// ── The implicit owner node ─────────────────────────────────────────────────
//
// Seeds `graph` with the one node that names its owner and records its id.
// Idempotent: an owner node that is already present is left alone, so this can
// be called after deserialization without duplicating it.
//
// ★ World seeds nothing on purpose. WorldThermalState has no scripting surface
// (section 7 item 1), so a World owner node would be a node that names a thing
// no reader can query — a graph pretending to more reach than the core has.
inline void seedOwnerNode(SimulationNodeGraph& graph) {
    if (graph.ownerNodeId != 0 && graph.getNode(graph.ownerNodeId)) return;
    graph.ownerNodeId = 0;
    if (graph.scope == GraphScope::Domain) {
        auto node = std::make_unique<DomainRefNode>();
        node->domainName = graph.owner;
        NodeBase* added = graph.registerNode(std::move(node));
        if (added) graph.ownerNodeId = added->id;
    } else if (graph.scope == GraphScope::Object) {
        auto node = std::make_unique<ObjectRefNode>();
        node->objectName = graph.owner;
        NodeBase* added = graph.registerNode(std::move(node));
        if (added) graph.ownerNodeId = added->id;
    }
}

// Creates a graph already bound to its owner. The ONLY way a scoped graph
// should come into existence, so no code path can produce one with an empty
// scope or a missing owner node.
inline std::shared_ptr<SimulationNodeGraph> makeScopedGraph(GraphScope scope,
                                                            const std::string& owner) {
    auto graph = std::make_shared<SimulationNodeGraph>();
    graph->scope = scope;
    graph->owner = (scope == GraphScope::World) ? std::string() : owner;
    seedOwnerNode(*graph);
    return graph;
}

} // namespace Sim
} // namespace NodeSystem
