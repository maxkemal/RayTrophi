#pragma once

#include <cstdint>
#include <cctype>
#include <string>

namespace RayTrophiSim::Fluid {

// How a parcel converts deformation/velocity into stress. This is deliberately
// separate from thermodynamic phase: ice and sand are both solid, but an
// elastic solid and a frictional granular skeleton require different solvers.
enum class MatterConstitutiveModel : uint8_t {
    Auto = 0,
    Fluid = 1,
    Granular = 2,
    Elastic = 3
};

enum class MatterGranularTransport : uint8_t {
    Mpm = 0,
    Dem = 1
};

inline const char* matterGranularTransportName(MatterGranularTransport transport) {
    return transport == MatterGranularTransport::Dem ? "dem" : "mpm";
}

inline bool parseMatterGranularTransport(const std::string& text,
                                        MatterGranularTransport& out) {
    if (text == "dem") {
        out = MatterGranularTransport::Dem;
        return true;
    }
    if (text == "mpm") {
        out = MatterGranularTransport::Mpm;
        return true;
    }
    return false;
}

inline const char* matterConstitutiveModelName(MatterConstitutiveModel model) {
    switch (model) {
        case MatterConstitutiveModel::Fluid: return "fluid";
        case MatterConstitutiveModel::Granular: return "granular";
        case MatterConstitutiveModel::Elastic: return "elastic";
        default: return "auto";
    }
}

inline bool parseMatterConstitutiveModel(
    const std::string& text,
    MatterConstitutiveModel& out) {
    std::string value;
    value.reserve(text.size());
    for (unsigned char ch : text) {
        if (ch == '_' || ch == '-' || std::isspace(ch)) continue;
        value.push_back(static_cast<char>(std::tolower(ch)));
    }
    if (value.empty() || value == "auto" || value == "fromsubstance") {
        out = MatterConstitutiveModel::Auto;
        return true;
    }
    if (value == "fluid" || value == "liquid") {
        out = MatterConstitutiveModel::Fluid;
        return true;
    }
    if (value == "granular" || value == "granule" || value == "sand") {
        out = MatterConstitutiveModel::Granular;
        return true;
    }
    if (value == "elastic") {
        out = MatterConstitutiveModel::Elastic;
        return true;
    }
    return false;
}

} // namespace RayTrophiSim::Fluid
