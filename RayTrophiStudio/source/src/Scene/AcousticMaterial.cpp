#include "AcousticMaterial.h"
#include "Audio/AcousticBoundary.h"
#include "SubstanceLibrary.h"
#include <algorithm>

namespace audio {
nlohmann::json acousticMaterialResponse(const RayTrophiSim::SubstanceProfile& profile) {
    using json = nlohmann::json;
    json result = {{"medium", acousticMediumState(profile)},
        {"defined", profile.acoustic_defined},
        {"frequencies_hz", {125, 250, 500, 1000, 2000, 4000}},
        {"quantity", "energy_fraction"}};
    if (!profile.acoustic_defined) {
        return result;
    }
    const float absorbed[] = {profile.acoustic_absorption_125, profile.acoustic_absorption_250,
        profile.acoustic_absorption_500, profile.acoustic_absorption_1000,
        profile.acoustic_absorption_2000, profile.acoustic_absorption_4000};
    const float transmitted[] = {profile.acoustic_transmission_125,
        profile.acoustic_transmission_250, profile.acoustic_transmission_500,
        profile.acoustic_transmission_1000, profile.acoustic_transmission_2000,
        profile.acoustic_transmission_4000};
    result["bands"] = json::array();
    for (unsigned band = 0; band < 6; ++band) {
        const float reflected = std::max(0.0f, 1 - absorbed[band] - transmitted[band]);
        const float diffuse = reflected * profile.acoustic_scattering;
        result["bands"].push_back({{"absorption", absorbed[band]},
            {"transmission", transmitted[band]}, {"reflection", reflected},
            {"specular_reflection", reflected - diffuse}, {"diffuse_reflection", diffuse}});
    }
    return result;
}
} // namespace audio
