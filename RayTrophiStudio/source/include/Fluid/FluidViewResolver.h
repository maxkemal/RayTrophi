#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// View resolver — the ONE place that decides how each substance of a liquid
// domain is drawn.
//
// ★★★ Before this existed the same question was answered four times: the
// volume route in scene_data.h, the splat bridge, fluid.get's
// effective_representation and the panel's "Now drawing" line. They agreed by
// copying each other, and the copies were how the panel came to show a fog
// that was not drawn. Every consumer calls resolveFluidViews and reads the
// answer; none may re-derive a view from fluid_render_mode or a binding.
//
// The key is (substance tag, state label). The substance decides a view
// (binding representation, else the domain default); the parcel's state label
// written by the simulation (body / spray / mist ..., see
// docs/dev/BIRLESIK_MADDE_DOMAIN_TASARIMI.md §4.2) may then route it
// elsewhere through the domain's label routes. Untagged parcels follow the
// domain default. The choice is per LABEL, never per domain.
// ═══════════════════════════════════════════════════════════════════════════

#include "Fluid/FluidParticleLabels.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace RayTrophiSim {
struct SimulationGridDomainDesc;
namespace Fluid {

class FluidParticles;
enum class FluidRenderMode : int;

enum class FluidView : uint8_t {
    Surface = 0,  // narrow-band level set, drawn as an isosurface volume
    Splat   = 1,  // one instance per particle
    Fog     = 2,  // splatted density, drawn as a participating medium
    Hidden  = 3   // routed nowhere (a label route); no resource, not drawn
};

// "sdf" | "splat" | "fog" | "hidden" — the names fluid.get and the panel use.
const char* fluidViewName(FluidView view);

// Where parcels carrying one state label go. Follow = whatever their
// substance resolves to; the others override it.
enum class LabelRoute : uint8_t {
    Follow  = 0,
    Surface = 1,
    Splat   = 2,
    Fog     = 3,
    Hidden  = 4,
    Count
};
using LabelRoutes = std::array<LabelRoute, kParticleLabelCount>;

// "follow" | "sdf" | "splat" | "fog" | "hidden".
const char* labelRouteName(LabelRoute route);
// false when `name` is not one of the names above.
bool parseLabelRoute(const std::string& name, LabelRoute& out);
// The design table (§4.2): spray / foam / bubble -> splat, mist -> fog;
// body, frozen and unknown follow their substance. Unknown MUST follow: an
// unclassified parcel may not be presented as anything it was not measured to be.
LabelRoutes defaultLabelRoutes();
// Parse a particleLabelName() back ("body", "spray", ...).
bool parseParticleLabel(const std::string& name, ParticleLabel& out);
// (label name, route name) for every label, in label order: the stored form
// (project files, fluid.get_label_views). A reader applies only the pairs it
// recognises, so a missing or unknown entry keeps the default.
std::vector<std::pair<std::string, std::string>> labelRoutesToNames(const LabelRoutes& routes);
// Returns false and changes nothing when either name is not recognised, or
// when `unknown` is routed anywhere but follow.
bool setLabelRouteByName(LabelRoutes& routes, const std::string& label, const std::string& route);

// One distinct (substance, label) pair present in the particles.
struct FluidViewKey {
    uint32_t      tag = 0;
    ParticleLabel label = ParticleLabel::Unknown;
    bool operator==(const FluidViewKey& o) const { return tag == o.tag && label == o.label; }
};

struct FluidViewEntry {
    std::string substance;   // empty = untagged parcels
    uint32_t    tag = 0;
    FluidView   view = FluidView::Surface;
    bool        is_override = false;  // binding chose it, not the domain default
};

struct FluidViewPlan {
    FluidView default_view = FluidView::Surface;
    // One entry for untagged parcels, then one per named binding.
    std::vector<FluidViewEntry> entries;
    LabelRoutes label_routes = defaultLabelRoutes();
    // (tag, label) pairs actually present in the particles.
    std::vector<FluidViewKey> live_keys;
    // Which view RESOURCES exist: an authored substance entry, a label route or
    // a live key resolves there. Stable across empty frames on purpose (a
    // resource that comes and goes rebuilds the RT scene / churns a volume
    // slot); what is drawn right now is anyLiveIn(view), and consumers skip
    // their gather while it is false.
    bool surface = false;
    bool splat   = false;
    bool fog     = false;

    // The substance's view, before any label route.
    FluidView viewForTag(uint32_t tag) const;
    FluidView viewFor(uint32_t tag, ParticleLabel label) const;
    FluidView viewForParticle(const FluidParticles& particles, std::size_t i) const;
    bool anyLiveIn(FluidView view) const;
    // Every live key resolves to `view` (so a whole-domain field can stand in).
    bool allLiveIn(FluidView view) const;
    std::vector<FluidViewKey> liveKeysIn(FluidView view) const;

    // ★ Whitewater (FoamParticles) is massless secondary state: the subgrid
    // stand-in for spray, foam and bubbles the solver cannot resolve. It is NOT
    // parcels and never joins the solve, but it is drawn by the SAME label
    // routes, so one table decides where "spray" goes whether the solver
    // resolved it or the whitewater model stands in for it. It has no
    // substance: Follow resolves through the untagged entry (the domain view).
    // Surface = the surface volume's whitewater channel (a white medium riding
    // the liquid's isosurface volume), not the level set.
    FluidView viewForWhitewater(uint8_t foam_type) const;
    // Which whitewater types (indexed by FoamType) resolve to `view`.
    std::array<bool, kWhitewaterTypeCount> whitewaterTypesIn(FluidView view) const;
};

// The parcels a gather keeps: exactly those whose resolved view is `view`.
// The level set, material coordinates, composition and fog density take this
// so all of them agree, parcel by parcel, on what one surface/fog is made of.
struct FluidViewSelection {
    const FluidViewPlan* plan = nullptr;
    FluidView view = FluidView::Surface;
    bool keeps(const FluidParticles& particles, std::size_t i) const {
        return !plan || plan->viewForParticle(particles, i) == view;
    }
};

// Parcels per resolved view, counted with viewForParticle: what is drawn.
struct FluidViewCounts {
    uint64_t surface = 0, splat = 0, fog = 0, hidden = 0;
};
FluidViewCounts countParticlesPerView(const FluidViewPlan& plan,
                                      const FluidParticles& particles);
// Whitewater particles per resolved view (`types` = FoamParticles::type).
FluidViewCounts countWhitewaterPerView(const FluidViewPlan& plan,
                                       const std::vector<uint8_t>& types);

FluidView defaultViewForMode(FluidRenderMode mode);

// Distinct (tag, label) pairs present in `particles`. Parcels past the end of
// the tag array count as untagged, exactly as the gathers treat them.
std::vector<FluidViewKey> distinctViewKeys(const FluidParticles& particles);

FluidViewPlan resolveFluidViews(const SimulationGridDomainDesc& desc,
                                const std::vector<FluidViewKey>& live_keys);

} // namespace Fluid
} // namespace RayTrophiSim
