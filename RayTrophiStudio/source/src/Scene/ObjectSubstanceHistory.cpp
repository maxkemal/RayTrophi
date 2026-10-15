#include "ObjectSubstance.h"
#include <map>
#include <mutex>
#include <set>

namespace matter {
namespace {
struct References {
    std::mutex mutex;
    std::map<std::string, std::size_t> counts;
};

std::shared_ptr<References> references() {
    static auto value = std::make_shared<References>();
    return value;
}

void collect(const nlohmann::json& value, std::set<std::string>& names) {
    if (value.is_object() && value.contains("substance") && value.at("substance").is_string()) {
        names.insert(value.at("substance").get<std::string>());
    }
    if (value.is_object() || value.is_array()) {
        for (const auto& child : value) {
            collect(child, names);
        }
    }
}

struct Lease {
    std::shared_ptr<References> owner = references();
    std::set<std::string> names;

    explicit Lease(const nlohmann::json& state) {
        collect(state, names);
        std::lock_guard lock(owner->mutex);
        for (const auto& name : names) {
            ++owner->counts[name];
        }
    }

    ~Lease() {
        std::lock_guard lock(owner->mutex);
        for (const auto& name : names) {
            const auto found = owner->counts.find(name);
            if (found != owner->counts.end() && --found->second == 0) {
                owner->counts.erase(found);
            }
        }
    }
};
}

std::shared_ptr<void> retainObjectSubstanceHistory(const nlohmann::json& state) {
    return std::make_shared<Lease>(state);
}

bool objectSubstanceHistoryUses(const std::string& substance) {
    const auto owner = references();
    std::lock_guard lock(owner->mutex);
    return owner->counts.contains(substance);
}
} // namespace matter
