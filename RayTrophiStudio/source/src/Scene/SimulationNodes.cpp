/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Scene/SimulationNodes.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 * Simulation node layer implementation — Faz N0/N1/N2.
 * Decision record: docs/dev/NODE_SIMULATION_ARCHITECTURE_PLAN.md, BOLUM D.
 */

#include "NodeSystem/SimulationNodes.h"
#include "NodeSystem/NodeRegistry.h"

namespace NodeSystem {
namespace Sim {

namespace {
AttributeStatsFn g_stats_resolver = nullptr;
AttributeListFn  g_list_resolver = nullptr;
SurfaceStatsFn   g_surface_stats_resolver = nullptr;
SurfaceListFn    g_surface_list_resolver = nullptr;
CacheStatusFn    g_cache_status_resolver = nullptr;
} // namespace

void setCacheStatusResolver(CacheStatusFn fn) { g_cache_status_resolver = fn; }

void setAttributeStatsResolver(AttributeStatsFn fn) { g_stats_resolver = fn; }
void setAttributeListResolver(AttributeListFn fn)   { g_list_resolver = fn; }
void setSurfaceStatsResolver(SurfaceStatsFn fn)     { g_surface_stats_resolver = fn; }
void setSurfaceListResolver(SurfaceListFn fn)       { g_surface_list_resolver = fn; }

std::vector<std::string> listAttributes(const std::string& domain) {
    if (!g_list_resolver) return {};
    return g_list_resolver(domain);
}

std::vector<std::string> listSurfaceAttributes(const std::string& object) {
    if (!g_surface_list_resolver) return {};
    return g_surface_list_resolver(object);
}

// Pull a string-valued input (a DomainRef or Field identity). Returns empty when
// the pin is unconnected — which callers must treat as "no domain", never as a
// domain named "".
std::string FieldReadNode::inputString(int index, EvaluationContext& ctx) {
    if (index < 0 || index >= static_cast<int>(inputs.size())) return {};
    // getInputValue already resolves the upstream link, falls back to the pin
    // default, and yields monostate when neither exists — so an unconnected pin
    // arrives here as "not a string" rather than as an empty domain name.
    PinValue v = getInputValue(index, ctx);
    if (const auto* s = std::get_if<std::string>(&v)) return *s;
    return {};
}

PinValue FieldInspectNode::compute(int outputIndex, EvaluationContext& ctx) {
    stats = Stats{};   // ★ Reset FIRST. A stale reading surviving a failed
                       // resolve is exactly the "the panel shows a plausible
                       // number that is not a measurement" trap.
    const std::string domain = inputString(0, ctx);
    if (domain.empty() || !g_stats_resolver) {
        // available stays false: no domain bound, or the app never installed a
        // resolver. Reporting zeros here would be indistinguishable from a real
        // all-zero field.
        return PinValue{};
    }

    FieldStats fs;
    if (!g_stats_resolver(domain, channel, fs)) return PinValue{};

    stats.available = true;
    stats.particle_count = fs.count;
    stats.array_size = fs.array_size;
    stats.in_sync = fs.in_sync;
    stats.min_value = fs.min_value;
    stats.max_value = fs.max_value;
    stats.mean_value = fs.mean_value;

    // Field identity still flows downstream so this node can sit inline.
    // The Mean pin narrows to float because a pin value is a float — the exact
    // reading stays in `stats`, which is what a test reads.
    if (outputIndex == 1) return PinValue{ static_cast<float>(stats.mean_value) };
    return PinValue{ domain + ":" + channel };
}

PinValue DomainParamNodeBase::compute(int, EvaluationContext& ctx) {
    PinValue upstream = getInputValue(0, ctx);
    const auto* domain = std::get_if<std::string>(&upstream);
    if (!domain || domain->empty()) return PinValue{};

    // ★★★ Only fields in USE emit. An unused field is not "zero" — it is a
    // parameter this graph has no opinion about, and writing a zero there would
    // silently overwrite an authored value with a number nobody chose.
    for (const auto& f : fields) {
        if (!f.use) continue;
        SimCommand cmd;
        cmd.kind = SimCommand::Kind::SetParameter;
        cmd.target = *domain;
        cmd.key = f.key;
        cmd.value = f.value;
        emit(std::move(cmd));
    }
    // Pass the domain through so aspect nodes chain onto one another and the
    // command order follows the chain.
    return PinValue{ *domain };
}

PinValue EmitterNode::compute(int, EvaluationContext& ctx) {
    // ★★★ No wire, no effect. The Domain pin is what puts this node IN the
    // chain; without it the canvas would show an isolated box that still
    // reconfigures a solver, and nobody reading the graph could tell.
    PinValue upstream = getInputValue(0, ctx);
    const auto* domain = std::get_if<std::string>(&upstream);
    if (!domain || domain->empty()) return PinValue{};
    // ★ The TARGET is still the emitter's own name, not the incoming domain: a
    // flow source already knows which region it feeds, and overwriting that
    // from the wire would silently move emission somewhere else.
    if (emitterName.empty()) return PinValue{};

    for (const auto& f : fields) {
        if (!f.use) continue;
        SimCommand cmd;
        cmd.kind = SimCommand::Kind::SetEmitter;
        cmd.target = emitterName;
        cmd.key = f.key;
        cmd.value = f.value;
        emit(std::move(cmd));
    }
    if (useSubstance) {
        SimCommand cmd;
        cmd.kind = SimCommand::Kind::SetEmitter;
        cmd.target = emitterName;
        cmd.key = "fluid_substance";
        cmd.text = substance;
        emit(std::move(cmd));
    }
    // Pass the domain through so emitters chain like every other aspect node.
    return PinValue{ *domain };
}

PinValue SetParameterNode::compute(int, EvaluationContext& ctx) {
    // Reuse FieldReadNode's resolver shape without inheriting from it: this node
    // is not a reader, and making it one would blur "acts" and "observes".
    PinValue upstream = getInputValue(0, ctx);
    const auto* domain = std::get_if<std::string>(&upstream);
    if (!domain || domain->empty() || key.empty()) return PinValue{};

    SimCommand cmd;
    cmd.kind = SimCommand::Kind::SetParameter;
    cmd.target = *domain;
    cmd.key = key;
    cmd.value = value;
    emit(std::move(cmd));

    // Pass the domain through so several parameter nodes can be chained and the
    // command ORDER follows the chain rather than node creation order.
    return PinValue{ *domain };
}

PinValue CouplingNodeBase::compute(int, EvaluationContext& ctx) {
    PinValue source_in = getInputValue(0, ctx);
    PinValue target_in = getInputValue(1, ctx);
    const auto* source = std::get_if<std::string>(&source_in);
    const auto* target = std::get_if<std::string>(&target_in);
    // ★ A coupling with only one end is not a weak coupling, it is an
    // unfinished node. Emitting it with an empty target would declare a
    // coupling to a domain named "" and the apply layer would then report a
    // failure about a domain nobody named.
    if (!source || source->empty()) return PinValue{};

    SimCommand cmd;
    cmd.kind = SimCommand::Kind::Couple;
    cmd.target = *source;
    cmd.key = couplingId();
    cmd.value = active ? 1.0f : 0.0f;
    cmd.text = (target && !target->empty()) ? *target : std::string();
    emit(std::move(cmd));

    // Pass the SOURCE through, so chaining coupling nodes produces commands in
    // chain order. That order is what gets compared against the solver's trace.
    return PinValue{ *source };
}

// ── N5: substance and chemistry ─────────────────────────────────────────────

PinValue ObjectRefNode::compute(int, EvaluationContext&) {
    SimCommand cmd;
    cmd.kind = SimCommand::Kind::BindObject;
    cmd.scope = SimCommand::Scope::Object;
    cmd.target = objectName;
    emit(std::move(cmd));
    return PinValue{ objectName };
}

std::string SurfaceNodeBase::boundObject(EvaluationContext& ctx) {
    PinValue upstream = getInputValue(0, ctx);
    if (const auto* s = std::get_if<std::string>(&upstream)) return *s;
    return {};
}

void SurfaceNodeBase::emitSurface(const std::string& object, const char* key,
                                  float value) {
    SimCommand cmd;
    cmd.kind = SimCommand::Kind::SetSurface;
    cmd.scope = SimCommand::Scope::Object;
    cmd.target = object;
    cmd.key = key;
    cmd.value = value;
    emit(std::move(cmd));
}

void SurfaceNodeBase::emitSurfaceText(const std::string& object, const char* key,
                                      const std::string& text) {
    SimCommand cmd;
    cmd.kind = SimCommand::Kind::SetSurface;
    cmd.scope = SimCommand::Scope::Object;
    cmd.target = object;
    cmd.key = key;
    cmd.text = text;
    emit(std::move(cmd));
}

PinValue SubstanceNode::compute(int, EvaluationContext& ctx) {
    const std::string object = boundObject(ctx);
    if (object.empty()) return PinValue{};

    emitSurfaceText(object, "substance", substanceName);
    // ★ The three scales are emitted ALWAYS, including at their no-op defaults.
    // Emitting only non-defaults would mean a graph could never put a value
    // back: clearing an override in the UI would leave the last written scale
    // in place, and the node would silently stop describing the object.
    emitSurface(object, "override_ignition", overrideIgnition ? 1.0f : 0.0f);
    emitSurface(object, "ignition_kelvin", ignitionKelvin);
    emitSurface(object, "burn_rate_scale", burnRateScale);
    emitSurface(object, "fuel_capacity_scale", fuelCapacityScale);
    return PinValue{ object };
}

PinValue PyrolysisNode::compute(int, EvaluationContext& ctx) {
    const std::string object = boundObject(ctx);
    if (object.empty()) return PinValue{};
    emitSurface(object, "ignite_on_contact", active ? 1.0f : 0.0f);
    return PinValue{ object };
}

PinValue PhaseChangeNode::compute(int, EvaluationContext& ctx) {
    const std::string object = boundObject(ctx);
    if (object.empty()) return PinValue{};
    emitSurface(object, "melt_flow", meltFlow ? 1.0f : 0.0f);
    emitSurface(object, "melt_height_loss", heightLoss);
    emitSurface(object, "melt_spread", spread);
    return PinValue{ object };
}

PinValue SurfaceInspectNode::compute(int outputIndex, EvaluationContext& ctx) {
    stats = Stats{};   // reset FIRST, same reason as FieldInspectNode
    PinValue upstream = getInputValue(0, ctx);
    const auto* object = std::get_if<std::string>(&upstream);
    if (!object || object->empty() || !g_surface_stats_resolver) return PinValue{};

    FieldStats fs;
    if (!g_surface_stats_resolver(*object, channel, fs)) return PinValue{};

    stats.available = true;
    stats.particle_count = fs.count;
    stats.array_size = fs.array_size;
    stats.in_sync = fs.in_sync;
    stats.host_fresh = fs.host_fresh;
    stats.min_value = fs.min_value;
    stats.max_value = fs.max_value;
    stats.mean_value = fs.mean_value;

    if (outputIndex == 0) return PinValue{ static_cast<float>(stats.mean_value) };
    return PinValue{ *object };
}

// ── N7: render binding ──────────────────────────────────────────────────────

std::string RenderNodeBase::boundDomain(EvaluationContext& ctx) {
    PinValue upstream = getInputValue(0, ctx);
    if (const auto* s = std::get_if<std::string>(&upstream)) return *s;
    return {};
}

void RenderNodeBase::emitRender(const std::string& domain, const char* key,
                                float value) {
    SimCommand cmd;
    cmd.kind = SimCommand::Kind::SetRender;
    cmd.target = domain;
    cmd.key = key;
    cmd.value = value;
    emit(std::move(cmd));
}

void RenderNodeBase::emitRenderText(const std::string& domain, const char* key,
                                    const std::string& text) {
    SimCommand cmd;
    cmd.kind = SimCommand::Kind::SetRender;
    cmd.target = domain;
    cmd.key = key;
    cmd.text = text;
    emit(std::move(cmd));
}

PinValue LiquidMaterialNode::compute(int, EvaluationContext& ctx) {
    const std::string domain = boundDomain(ctx);
    if (domain.empty()) return PinValue{};
    // ★ Emitted even when empty: empty is a MEANINGFUL value here ("built-in
    // dielectric" / "scene default"), so skipping it would make clearing a
    // material in the graph impossible — the last non-empty name would stick.
    emitRenderText(domain, "surface_material", surfaceMaterial);
    emitRenderText(domain, "splat_material", splatMaterial);
    return PinValue{ domain };
}

PinValue VolumeMaterialNode::compute(int, EvaluationContext& ctx) {
    const std::string domain = boundDomain(ctx);
    if (domain.empty()) return PinValue{};
    emitRenderText(domain, "volume_preset", preset);
    // See `overrideValues`: silent by default so the preset's own numbers stand.
    if (overrideValues) {
        emitRender(domain, "density_multiplier", densityMultiplier);
        emitRender(domain, "density_cutoff", densityCutoff);
        emitRender(domain, "temperature_min", temperatureMin);
        emitRender(domain, "temperature_max", temperatureMax);
        emitRender(domain, "scattering", scattering);
        emitRender(domain, "absorption", absorption);
    }
    return PinValue{ domain };
}

// ── N6: cache / bake ────────────────────────────────────────────────────────

PinValue CacheNode::compute(int, EvaluationContext& ctx) {
    status = Status{};   // reset FIRST; a stale reading is worse than none
    PinValue upstream = getInputValue(0, ctx);
    const auto* domain = std::get_if<std::string>(&upstream);

    if (g_cache_status_resolver) {
        CacheStatusReading r;
        if (g_cache_status_resolver(r)) {
            status.available = true;
            status.valid = r.valid;
            status.baking = r.baking;
            status.ram_frames = r.ram_frames;
            status.has_range = r.has_range;
            status.first_frame = r.first_frame;
            status.last_frame = r.last_frame;
            status.config_signature = r.config_signature;
            // ★★★ The whole point. A cache that EXISTS but was built from a
            // different authored config still serves frames — and they describe
            // a scene that no longer exists. Nothing else distinguishes it from
            // a healthy cache, which is how stale physics reaches a render.
            const bool something_cached = r.valid || r.ram_frames > 0;
            status.stale = something_cached && r.config_signature != 0 &&
                           r.config_signature != r.live_signature;
        }
    }

    // ★ Evaluation NEVER bakes. Baking walks the whole simulation, and a graph
    // evaluation has to stay instant and side-effect free or merely inspecting
    // the graph would run the sim. The bake is an explicit action.
    if (domain && !domain->empty()) return PinValue{ *domain };
    return PinValue{};
}

void registerSimulationNodeTypes() {
    auto& reg = NodeRegistry::instance();
    reg.registerType("sim.domain_ref",
                     [] { return std::static_pointer_cast<NodeBase>(
                              std::make_shared<DomainRefNode>()); });
    reg.registerType("sim.field_read",
                     [] { return std::static_pointer_cast<NodeBase>(
                              std::make_shared<FieldReadNode>()); });
    reg.registerType("sim.set_parameter",
                     [] { return std::static_pointer_cast<NodeBase>(
                              std::make_shared<SetParameterNode>()); });
    reg.registerType("sim.solver",
                     [] { return std::static_pointer_cast<NodeBase>(
                              std::make_shared<SolverNode>()); });
    reg.registerType("sim.domain_settings",
                     [] { return std::static_pointer_cast<NodeBase>(
                              std::make_shared<DomainSettingsNode>()); });
    reg.registerType("sim.emitter",
                     [] { return std::static_pointer_cast<NodeBase>(
                              std::make_shared<EmitterNode>()); });
    reg.registerType("sim.field_inspect",
                     [] { return std::static_pointer_cast<NodeBase>(
                              std::make_shared<FieldInspectNode>()); });
    reg.registerType("sim.couple_fluid_to_gas",
                     [] { return std::static_pointer_cast<NodeBase>(
                              std::make_shared<FluidToGasCouplingNode>()); });
    reg.registerType("sim.couple_gas_ignition",
                     [] { return std::static_pointer_cast<NodeBase>(
                              std::make_shared<GasToFluidIgnitionNode>()); });
    reg.registerType("sim.couple_foam",
                     [] { return std::static_pointer_cast<NodeBase>(
                              std::make_shared<FoamFromFluidNode>()); });
    reg.registerType("sim.object_ref",
                     [] { return std::static_pointer_cast<NodeBase>(
                              std::make_shared<ObjectRefNode>()); });
    reg.registerType("sim.substance",
                     [] { return std::static_pointer_cast<NodeBase>(
                              std::make_shared<SubstanceNode>()); });
    reg.registerType("sim.pyrolysis",
                     [] { return std::static_pointer_cast<NodeBase>(
                              std::make_shared<PyrolysisNode>()); });
    reg.registerType("sim.phase_change",
                     [] { return std::static_pointer_cast<NodeBase>(
                              std::make_shared<PhaseChangeNode>()); });
    reg.registerType("sim.surface_inspect",
                     [] { return std::static_pointer_cast<NodeBase>(
                              std::make_shared<SurfaceInspectNode>()); });
    reg.registerType("sim.cache",
                     [] { return std::static_pointer_cast<NodeBase>(
                              std::make_shared<CacheNode>()); });
    reg.registerType("sim.material_liquid",
                     [] { return std::static_pointer_cast<NodeBase>(
                              std::make_shared<LiquidMaterialNode>()); });
    reg.registerType("sim.material_volume",
                     [] { return std::static_pointer_cast<NodeBase>(
                              std::make_shared<VolumeMaterialNode>()); });
}

} // namespace Sim
} // namespace NodeSystem
