#include "Fluid/FluidViewResolver.h"

#include "Fluid/FluidParticles.h"
#include "Fluid/FluidRenderMode.h"
#include "Fluid/SubstanceTag.h"
#include "ParticleSimulation.h"

#include <algorithm>

namespace RayTrophiSim {
namespace Fluid {

const char* fluidViewName(FluidView view) {
    switch (view) {
        case FluidView::Splat: return "splat";
        case FluidView::Fog:   return "fog";
        case FluidView::Hidden: return "hidden";
        case FluidView::Surface:
        default:               return "sdf";
    }
}

const char* labelRouteName(LabelRoute route) {
    switch (route) {
        case LabelRoute::Surface: return "sdf";
        case LabelRoute::Splat:   return "splat";
        case LabelRoute::Fog:     return "fog";
        case LabelRoute::Hidden:  return "hidden";
        case LabelRoute::Follow:
        default:                  return "follow";
    }
}

bool parseLabelRoute(const std::string& name, LabelRoute& out) {
    for (uint8_t r = 0; r < static_cast<uint8_t>(LabelRoute::Count); ++r) {
        if (name == labelRouteName(static_cast<LabelRoute>(r))) {
            out = static_cast<LabelRoute>(r);
            return true;
        }
    }
    return false;
}

LabelRoutes defaultLabelRoutes() {
    LabelRoutes r;
    r.fill(LabelRoute::Follow);
    r[static_cast<std::size_t>(ParticleLabel::Spray)]  = LabelRoute::Splat;
    r[static_cast<std::size_t>(ParticleLabel::Foam)]   = LabelRoute::Splat;
    r[static_cast<std::size_t>(ParticleLabel::Bubble)] = LabelRoute::Splat;
    r[static_cast<std::size_t>(ParticleLabel::Mist)]   = LabelRoute::Fog;
    return r;
}

bool parseParticleLabel(const std::string& name, ParticleLabel& out) {
    for (std::size_t li = 0; li < kParticleLabelCount; ++li) {
        const ParticleLabel l = static_cast<ParticleLabel>(li);
        if (name == particleLabelName(l)) { out = l; return true; }
    }
    return false;
}

std::vector<std::pair<std::string, std::string>> labelRoutesToNames(const LabelRoutes& routes) {
    std::vector<std::pair<std::string, std::string>> out;
    for (std::size_t li = 0; li < routes.size(); ++li)
        out.emplace_back(particleLabelName(static_cast<ParticleLabel>(li)),
                         labelRouteName(routes[li]));
    return out;
}

bool setLabelRouteByName(LabelRoutes& routes, const std::string& label, const std::string& route) {
    ParticleLabel l;
    LabelRoute r;
    if (!parseParticleLabel(label, l) || !parseLabelRoute(route, r)) return false;
    if (l == ParticleLabel::Unknown && r != LabelRoute::Follow) return false;
    routes[static_cast<std::size_t>(l)] = r;
    return true;
}

FluidView defaultViewForMode(FluidRenderMode mode) {
    switch (mode) {
        case FluidRenderMode::Particles: return FluidView::Splat;
        case FluidRenderMode::VolumeFog: return FluidView::Fog;
        // Volume is the invalid liquid value and is normalised to the
        // isosurface where the mode is consumed; resolve it the same way so
        // no consumer invents a third answer.
        case FluidRenderMode::SurfaceSDF:
        case FluidRenderMode::Volume:
        default:                         return FluidView::Surface;
    }
}

FluidView FluidViewPlan::viewForTag(uint32_t tag) const {
    for (const auto& e : entries) {
        if (e.tag == tag) return e.view;
    }
    return default_view;
}

FluidView FluidViewPlan::viewFor(uint32_t tag, ParticleLabel label) const {
    // Unclassified parcels follow their substance whatever the table says; see
    // defaultLabelRoutes. Enforced here, not only in the setters.
    if (label == ParticleLabel::Unknown) return viewForTag(tag);
    const std::size_t li = static_cast<std::size_t>(label);
    const LabelRoute route = li < label_routes.size() ? label_routes[li] : LabelRoute::Follow;
    switch (route) {
        case LabelRoute::Surface: return FluidView::Surface;
        case LabelRoute::Splat:   return FluidView::Splat;
        case LabelRoute::Fog:     return FluidView::Fog;
        case LabelRoute::Hidden:  return FluidView::Hidden;
        case LabelRoute::Follow:
        default:                  return viewForTag(tag);
    }
}

FluidView FluidViewPlan::viewForParticle(const FluidParticles& particles,
                                         std::size_t i) const {
    // Past the end of a sidecar = untagged / unclassified, exactly as the
    // gathers have always treated a short tag array.
    const uint32_t tag = i < particles.substance_tag.size()
        ? particles.substance_tag[i] : kSubstanceUntagged;
    const ParticleLabel label = i < particles.flags.size()
        ? particleLabel(particles.flags[i]) : ParticleLabel::Unknown;
    return viewFor(tag, label);
}

bool FluidViewPlan::anyLiveIn(FluidView view) const {
    for (const auto& k : live_keys)
        if (viewFor(k.tag, k.label) == view) return true;
    return false;
}

bool FluidViewPlan::allLiveIn(FluidView view) const {
    if (live_keys.empty()) return false;
    for (const auto& k : live_keys)
        if (viewFor(k.tag, k.label) != view) return false;
    return true;
}

std::vector<FluidViewKey> FluidViewPlan::liveKeysIn(FluidView view) const {
    std::vector<FluidViewKey> out;
    for (const auto& k : live_keys)
        if (viewFor(k.tag, k.label) == view) out.push_back(k);
    return out;
}

FluidView FluidViewPlan::viewForWhitewater(uint8_t foam_type) const {
    const ParticleLabel label = secondaryParticleLabel(foam_type);
    if (label == ParticleLabel::Unknown) return FluidView::Hidden;
    return viewFor(kSubstanceUntagged, label);
}

std::array<bool, kWhitewaterTypeCount> FluidViewPlan::whitewaterTypesIn(FluidView view) const {
    std::array<bool, kWhitewaterTypeCount> in{};
    for (std::size_t t = 0; t < kWhitewaterTypeCount; ++t)
        in[t] = viewForWhitewater(static_cast<uint8_t>(t)) == view;
    return in;
}

FluidViewCounts countWhitewaterPerView(const FluidViewPlan& plan,
                                       const std::vector<uint8_t>& types) {
    FluidView view_of[kWhitewaterTypeCount];
    for (std::size_t t = 0; t < kWhitewaterTypeCount; ++t)
        view_of[t] = plan.viewForWhitewater(static_cast<uint8_t>(t));
    FluidViewCounts c;
    for (const uint8_t t : types) {
        const FluidView v = t < kWhitewaterTypeCount ? view_of[t] : FluidView::Hidden;
        switch (v) {
            case FluidView::Surface: ++c.surface; break;
            case FluidView::Splat:   ++c.splat; break;
            case FluidView::Fog:     ++c.fog; break;
            case FluidView::Hidden:  ++c.hidden; break;
        }
    }
    return c;
}

FluidViewCounts countParticlesPerView(const FluidViewPlan& plan,
                                      const FluidParticles& particles) {
    FluidViewCounts c;
    const std::size_t n = particles.position.size();
    for (std::size_t i = 0; i < n; ++i) {
        switch (plan.viewForParticle(particles, i)) {
            case FluidView::Surface: ++c.surface; break;
            case FluidView::Splat:   ++c.splat; break;
            case FluidView::Fog:     ++c.fog; break;
            case FluidView::Hidden:  ++c.hidden; break;
        }
    }
    return c;
}

std::vector<FluidViewKey> distinctViewKeys(const FluidParticles& particles) {
    // A (tag, label) set is tiny (a few substances x 7 labels), so a linear
    // probe beats hashing and keeps first-seen order deterministic.
    std::vector<FluidViewKey> keys;
    const std::size_t n = particles.position.size();
    for (std::size_t i = 0; i < n; ++i) {
        FluidViewKey k;
        k.tag = i < particles.substance_tag.size()
            ? particles.substance_tag[i] : kSubstanceUntagged;
        k.label = i < particles.flags.size()
            ? particleLabel(particles.flags[i]) : ParticleLabel::Unknown;
        if (std::find(keys.begin(), keys.end(), k) == keys.end()) keys.push_back(k);
    }
    return keys;
}

FluidViewPlan resolveFluidViews(const SimulationGridDomainDesc& desc,
                                const std::vector<FluidViewKey>& live_keys) {
    FluidViewPlan plan;
    plan.default_view = defaultViewForMode(desc.fluid_render_mode);
    plan.label_routes = desc.fluid_label_routes;
    plan.live_keys = live_keys;

    FluidViewEntry untagged;
    untagged.tag = kSubstanceUntagged;
    untagged.view = plan.default_view;
    plan.entries.push_back(untagged);

    for (const auto& b : desc.fluid_substance_materials) {
        if (b.substance.empty()) continue;
        FluidViewEntry e;
        e.substance = b.substance;
        e.tag = substanceTag(b.substance);
        switch (b.representation) {
            case SubstanceRepresentation::Splat:
                e.view = FluidView::Splat; e.is_override = true; break;
            case SubstanceRepresentation::SurfaceSDF:
                e.view = FluidView::Surface; e.is_override = true; break;
            case SubstanceRepresentation::Fog:
                e.view = FluidView::Fog; e.is_override = true; break;
            case SubstanceRepresentation::Inherit:
            default:
                e.view = plan.default_view; break;
        }
        // First binding for a tag wins, matching every consumer before this.
        const bool seen = std::any_of(plan.entries.begin(), plan.entries.end(),
            [&e](const FluidViewEntry& x) { return x.tag == e.tag; });
        if (!seen) plan.entries.push_back(e);
    }

    auto mark = [&plan](FluidView v) {
        switch (v) {
            case FluidView::Surface: plan.surface = true; break;
            case FluidView::Splat:   plan.splat = true; break;
            case FluidView::Fog:     plan.fog = true; break;
            case FluidView::Hidden:  break;
        }
    };
    // ★ The flags decide which RESOURCES exist, so they are the union of what
    // is live and what the domain is authored to draw — never live content
    // alone. An empty frame (emitter not started, rewound to frame 0) or a
    // substance that comes and goes would otherwise retire a volume and
    // re-register it a frame later under a new id: the slot-identity churn
    // behind the domain-edge black band (VOLUME_BOX_REENTRY_POSTMORTEM.md).
    // A slot with nothing in it stays registered with inactive content, which
    // costs nothing. What is actually drawn is anyLiveIn(view).
    //
    // ★★ Label routes ARE authored resources. The first version marked them
    // only while a key was live, and that is the churn this comment warns
    // about: MEASURED 2026-09-28 on a dam break in Vulkan RT, a full
    // accel.vulkan_rt.rebuild fired on exactly the frames where spray went
    // 0 <-> >0 (23, 24, 26, 53, 54, 58) and on none of the 26 frames where it
    // ranged 2..71 - the splat pool was deleted and recreated every time the
    // last droplet rejoined. A fog route would do worse: syncDomainFogVolume
    // REMOVES the slot when plan.fog is false. So a route keeps its resource,
    // and consumers skip the work while anyLiveIn(view) is false.
    for (const auto& e : plan.entries) mark(e.view);
    for (std::size_t li = 0; li < plan.label_routes.size(); ++li) {
        if (static_cast<ParticleLabel>(li) == ParticleLabel::Unknown) continue;  // always follows
        switch (plan.label_routes[li]) {
            case LabelRoute::Surface: mark(FluidView::Surface); break;
            case LabelRoute::Splat:   mark(FluidView::Splat); break;
            case LabelRoute::Fog:     mark(FluidView::Fog); break;
            default: break;
        }
    }
    for (const auto& k : live_keys) mark(plan.viewFor(k.tag, k.label));
    return plan;
}

} // namespace Fluid
} // namespace RayTrophiSim
