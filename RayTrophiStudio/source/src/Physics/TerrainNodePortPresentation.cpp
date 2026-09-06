#include "TerrainNodePortPresentation.h"

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <iterator>
#include <string>
#include <unordered_set>

namespace TerrainNodesV2 {
namespace {

std::string makeStableKey(const std::string& name) {
    std::string key;
    key.reserve(name.size());
    bool pendingSeparator = false;
    for (unsigned char c : name) {
        if (std::isalnum(c)) {
            if (pendingSeparator && !key.empty()) key.push_back('_');
            key.push_back(static_cast<char>(std::tolower(c)));
            pendingSeparator = false;
        } else {
            pendingSeparator = !key.empty();
        }
    }
    return key.empty() ? "port" : key;
}

bool contains(std::initializer_list<const char*> values, const std::string& name) {
    return std::any_of(values.begin(), values.end(),
        [&name](const char* value) { return name == value; });
}

void assignKeys(std::vector<NodeSystem::Pin>& pins) {
    std::unordered_set<std::string> used;
    for (size_t i = 0; i < pins.size(); ++i) {
        auto& pin = pins[i];
        std::string base = makeStableKey(pin.name);
        std::string key = base;
        for (int suffix = 2; !used.insert(key).second; ++suffix)
            key = base + "_" + std::to_string(suffix);
        pin.stableKey = std::move(key);
    }
}

void compact(std::vector<NodeSystem::Pin>& pins,
             std::initializer_list<const char*> primary,
             const char* optionalSection,
             std::initializer_list<const char*> diagnostic = {}) {
    for (auto& pin : pins) {
        if (contains(primary, pin.name)) {
            pin.exposure = NodeSystem::PinExposure::Primary;
            pin.section.clear();
            pin.hidden = false;
        } else {
            pin.exposure = contains(diagnostic, pin.name)
                ? NodeSystem::PinExposure::Diagnostic
                : NodeSystem::PinExposure::Optional;
            pin.section = pin.exposure == NodeSystem::PinExposure::Diagnostic
                ? "Diagnostics" : optionalSection;
            pin.hidden = true;
        }
    }
}

void setSection(std::vector<NodeSystem::Pin>& pins,
                std::initializer_list<const char*> names,
                const char* section) {
    for (auto& pin : pins)
        if (contains(names, pin.name)) pin.section = section;
}

NodeSystem::Pin* named(std::vector<NodeSystem::Pin>& pins, const char* name) {
    const auto it = std::find_if(pins.begin(), pins.end(),
        [name](const NodeSystem::Pin& pin) { return pin.name == name; });
    return it == pins.end() ? nullptr : &*it;
}

// Every terrain node gets a profile, not only the ones with a hand-written
// branch below. Without this, 62 of 79 node types published every socket as an
// equally important authoring choice -- measured on a real SatMap graph, the
// five Color Ramp nodes alone drew 45 input pins to carry 12 connections, and
// none of their pins could be tiered away because the whole `Terrain.*` family
// sits outside the branch list.
//
// The rule is deliberately NOT an opinion about each node. It reads what the
// node's author already declared:
//
//   * input 0 is the node's subject and stays Primary. Measured across all 65
//     terrain node classes, the first input is always the thing the node acts
//     on -- Height, Base Height, Bed Height, Mask, In, A, Source, Splat,
//     Accumulation. There is no counter-example.
//   * every other input marked `optional` at construction becomes Optional:
//     hidden until connected, always listed in Properties.
//   * a required input stays Primary. Hiding one would hide an error.
//
// Outputs are left alone. An unread output is a question about consumers, and
// hiding it would answer that question by making it invisible.
//
// The failure modes are mild in both directions, which is why a mechanical
// rule is safe here: a pin wrongly tiered Optional still appears the moment it
// is connected and is always reachable from Properties, and a pin wrongly left
// Primary just keeps today's behaviour.
void applyDefaultProfile(NodeSystem::NodeBase& node) {
    for (size_t i = 0; i < node.inputs.size(); ++i) {
        auto& pin = node.inputs[i];
        const bool subject = (i == 0);
        if (subject || !pin.optional) {
            pin.exposure = NodeSystem::PinExposure::Primary;
            pin.section.clear();
            pin.hidden = false;
        } else {
            pin.exposure = NodeSystem::PinExposure::Optional;
            pin.section = "Optional Inputs";
            pin.hidden = true;
        }
    }
}

void configureHydraulic(NodeSystem::NodeBase& node) {
    compact(node.inputs, {"Height", "Area"}, "Material Control");
    // "Flow" was renamed to "Sediment" and a real "Discharge" added. These
    // lists match by pin NAME, so leaving "Flow" here would have silently
    // demoted both pins out of Products into the collapsed optional section -
    // the fix would have shipped invisible.
    compact(node.outputs, {"Height", "Wear", "Deposits", "Sediment", "Discharge"}, "Products");

    if (auto* p = named(node.outputs, "Sediment")) {
        p->tooltip = "Suspended sediment transport, normalised 0-1. This pin was "
                     "called Flow and that name cost a wiring bug: it is NOT the "
                     "water, it is what the water is carrying.";
    }
    if (auto* p = named(node.outputs, "Discharge")) {
        p->tooltip = "Measured water discharge in m3/s (catchment area x runoff). "
                     "Wire this into Flow.Discharge - otherwise the Flow mask "
                     "re-derives drainage from the height and flats come out "
                     "straight and angular.";
    }
}

} // namespace

const char* pinExposureName(NodeSystem::PinExposure exposure) {
    switch (exposure) {
        case NodeSystem::PinExposure::Primary: return "Primary";
        case NodeSystem::PinExposure::Optional: return "Optional";
        case NodeSystem::PinExposure::Diagnostic: return "Diagnostic";
    }
    return "Primary";
}

int legacyTerrainPortSlot(const std::string& typeId,
                          NodeSystem::PinKind kind,
                          size_t savedSlot) {
    if (typeId == "TerrainV2.HydraulicErosion" && kind == NodeSystem::PinKind::Output) {
        // Legacy order: Height Out, Erosion, Deposition, Discharge,
        // Sediment Flux, Flow Direction, Channel Width, Water Depth,
        // Water Level. Sediment Flux is the field the pin formerly called Flow
        // actually carries, so it maps to slot 3 (now named "Sediment").
        //
        // ★★★ Legacy Discharge is NO LONGER "Removed". It used to map to
        // nothing on the stated grounds that discharge "has no equivalent here"
        // and belonged to River Hydraulics. That was true of the pin list and
        // false of the solver: the LEM has computed a real m3/s discharge all
        // along and simply had nowhere to publish it, which is why the Flow
        // mask fell back to re-deriving drainage from geometry in every graph
        // without a river network. Slot 4 is that pin, so a project saved
        // before the pruning gets its Discharge link back instead of having it
        // counted as dropped.
        static const int kOutputSlots[] = {0, 1, 2,
                                           4, 3,
                                           kTerrainPortSlotRemoved,
                                           kTerrainPortSlotRemoved,
                                           kTerrainPortSlotRemoved,
                                           kTerrainPortSlotRemoved};
        if (savedSlot < std::size(kOutputSlots)) return kOutputSlots[savedSlot];
        return kTerrainPortSlotRemoved;
    }
    return kTerrainPortSlotUnknown;
}

void configureTerrainNodePorts(NodeSystem::NodeBase& node) {
    assignKeys(node.inputs);
    assignKeys(node.outputs);
    applyDefaultProfile(node);

    // A hand-written branch below fully overrides the default for the pins it
    // names, so the order here matters: default first, opinion second.
    const std::string type = node.getTypeId();
    if (type == "TerrainV2.HydraulicErosion") {
        configureHydraulic(node);
    } else if (type == "TerrainV2.ThermalErosion") {
        compact(node.inputs, {"Height In", "Mask"}, "Material Control");
        compact(node.outputs, {"Height Out", "Erosion", "Deposition"}, "Surface Products");
    } else if (type == "TerrainV2.NoiseGenerator") {
        compact(node.outputs, {"Height"}, "Derived Masks");
    } else if (type == "TerrainV2.CurveToMask") {
        compact(node.inputs, {"Curve"}, "Curve Inputs");
        compact(node.outputs, {"Mask"}, "Mask Products");
    } else if (type == "TerrainV2.RoadCarve") {
        compact(node.inputs, {"Height", "Curve"}, "Road Inputs");
        // Every one of these outputs is REQUIRED by Road Fields Output, which
        // publishes the bundle atomically or not at all. Tiering them away as
        // diagnostics hid pins the author has to connect for the node to do
        // anything - a pin that must be wired and cannot be seen is the worst
        // of the two failure directions this file weighs.
        compact(node.outputs, {"Height", "Road Core", "Shoulder", "Cut", "Fill",
                               "Foliage Exclusion", "Ditch", "Snapshot Revision"},
                "Infrastructure Fields");
    } else if (type == "TerrainV2.RoadNetwork") {
        // Water is Primary despite being optional: it is the only pin that tells
        // the solver where a crossing IS, and a Ford whose Water pin is empty
        // reports that it could not be resolved. A hidden pin would make that
        // message read as a bug in the crossing rather than a missing wire.
        compact(node.inputs, {"Height", "Water"}, "Road Inputs");
        compact(node.outputs, {"Height", "Road Core", "Shoulder", "Cut", "Fill",
                               "Foliage Exclusion", "Ditch", "Snapshot Revision"},
                "Infrastructure Fields");
    } else if (type == "TerrainV2.RoadFieldsOutput") {
        compact(node.inputs, {"Road Core", "Shoulder", "Cut", "Fill",
                              "Foliage Exclusion", "Snapshot Revision", "Ditch"},
                "Physical Diagnostics");
    } else if (type == "TerrainV2.MountainRange") {
        compact(node.inputs, {"Base Height", "Mask"}, "Optional Inputs");
        compact(node.outputs, {"Height", "Ridge", "Valley Seed"}, "Geology Products");
    } else if (type == "TerrainV2.PlateTectonics") {
        compact(node.outputs, {"Height", "Uplift", "Plate Boundary"}, "Geology Products", {"Crust ID"});
    } else if (type == "TerrainV2.WatershedAnalysis") {
        compact(node.inputs, {"Height", "Precipitation"}, "Physical Overrides");
        compact(node.outputs, {"Conditioned Height", "Accumulation", "Drainage Basins"}, "Hydrology Products");
    } else if (type == "TerrainV2.LakeBasin") {
        compact(node.inputs, {"Original Height", "Conditioned Height"}, "Hydrology Inputs");
        compact(node.outputs, {"Lake Mask", "Lake Depth", "Water Level"}, "Lake Diagnostics", {"Lake IDs"});
    } else if (type == "TerrainV2.RiverNetwork") {
        compact(node.inputs, {"Accumulation", "Flow Direction", "Lake Mask"}, "Network Inputs",
                {"Catchment Area", "Lake Spill Points", "Channel Exclusion"});
        compact(node.outputs, {"Channels", "Stream Order", "Sources"}, "Network Products");
    } else if (type == "TerrainV2.RiverHydraulics") {
        compact(node.inputs, {"Bed Height", "Catchment Area", "Channels"}, "Hydrology Inputs");
        compact(node.outputs, {"Discharge", "Flow Speed", "Water Level"}, "Hydraulic Diagnostics", {"Froude", "Foam Potential"});
    } else if (type == "TerrainV2.RiverBedCarve") {
        compact(node.inputs, {"Height", "Channels", "River Width", "Water Depth"}, "Carve Constraints");
    } else if (type == "TerrainV2.RiverSplineOutput") {
        compact(node.inputs, {"Height", "Accumulation", "Flow Direction", "Channels"}, "Water Rendering");
    } else if (type == "TerrainV2.TerrainFieldsOutput") {
        compact(node.inputs, {}, "Published Fields");
    } else if (type == "TerrainV2.SurfaceComposer") {
        compact(node.inputs, {"Height", "Soil", "Flow"}, "Surface Influences");
        compact(node.outputs, {"Splat"}, "Derived Surface Data");
    } else if (type == "TerrainV2.AutoSplat") {
        compact(node.inputs, {"Height", "Flow", "Slope"}, "Surface Influences");
        compact(node.outputs, {"Splat"}, "Derived Surface Data");
    } else if (type == "TerrainV2.BiomeComposer") {
        compact(node.inputs, {"Height"}, "Biome Influences");
        compact(node.outputs, {"Biome Splat"}, "Biome Masks");
    } else if (type == "TerrainV2.SnowClimate") {
        compact(node.inputs, {"Base Height", "Mask"}, "Climate Inputs");
        compact(node.outputs, {"Surface Height", "Snow", "Ice"}, "Climate Products");
    } else if (type == "TerrainV2.FoliageSet") {
        compact(node.inputs, {"Layer 1"}, "Additional Layers");
    }
}

NodeSystem::Pin* findTerrainPort(NodeSystem::NodeBase& node,
                                 NodeSystem::PinKind kind,
                                 const char* stableKey) {
    auto& pins = kind == NodeSystem::PinKind::Input ? node.inputs : node.outputs;
    const auto it = std::find_if(pins.begin(), pins.end(),
        [stableKey](const NodeSystem::Pin& pin) { return pin.stableKey == stableKey; });
    return it == pins.end() ? nullptr : &*it;
}

const NodeSystem::Pin* findTerrainPort(const NodeSystem::NodeBase& node,
                                       NodeSystem::PinKind kind,
                                       const char* stableKey) {
    const auto& pins = kind == NodeSystem::PinKind::Input ? node.inputs : node.outputs;
    const auto it = std::find_if(pins.begin(), pins.end(),
        [stableKey](const NodeSystem::Pin& pin) { return pin.stableKey == stableKey; });
    return it == pins.end() ? nullptr : &*it;
}

uint32_t terrainPortId(const NodeSystem::NodeBase* node,
                       NodeSystem::PinKind kind,
                       const char* stableKey) {
    if (!node) return 0;
    const auto* pin = findTerrainPort(*node, kind, stableKey);
    return pin ? pin->id : 0;
}

} // namespace TerrainNodesV2
