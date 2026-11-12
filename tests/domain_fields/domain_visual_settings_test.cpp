#include "Fluid/DomainVisualSettings.h"

#include <cstdlib>
#include <iostream>

using namespace RayTrophiSim::Fluid;

int main() {
    DomainVisualSettings settings;
    const nlohmann::json original = settings;
    for (const auto& patch : std::vector<nlohmann::json>{
             {{"enabled", 1}}, {{"target_volume", -1}}, {{"max_samples", true}},
             {{"max_samples", -1}}, {{"max_samples", 4000001}}, {{"seed", 1.5}},
             {{"max_substeps", 65537}}, {{"working_budget_mb", 1}},
             {{"schema", 2}}, {{"typo", 1}}, {{"target_volume", nullptr}},
             {{"cfl", 0}}, {{"enabled", true}, {"max_samples", 4000000}}}) {
        bool rejected = false;
        try {
            from_json(patch, settings);
        } catch (const std::exception&) {
            rejected = true;
        }
        if (!rejected || nlohmann::json(settings) != original) {
            std::cerr << "invalid settings patch was not rejected atomically: " << patch << '\n';
            return EXIT_FAILURE;
        }
    }
    from_json({{"enabled", true}, {"target_volume", 0.00001}, {"max_samples", 20000}}, settings);
    auto roundtrip = nlohmann::json(settings).get<DomainVisualSettings>();
    if (nlohmann::json(roundtrip) != nlohmann::json(settings) ||
        domainVisualSettingsKey(roundtrip) != domainVisualSettingsKey(settings)) {
        return EXIT_FAILURE;
    }
    std::cout << "domain visual settings PASS\n";
    return EXIT_SUCCESS;
}
