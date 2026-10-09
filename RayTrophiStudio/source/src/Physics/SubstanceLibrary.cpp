// Substance library: built-ins + project substances derived from them.
// See SubstanceLibrary.h and docs/dev/MADDE_TIPLERI_TASARIMI.md.
#include "SubstanceLibrary.h"
#include "Fluid/SubstanceTag.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace RayTrophiSim {

namespace {

using nlohmann::json;
using Kind = SubstanceFieldKind;

struct FieldAccess {
    SubstanceFieldSpec spec;
    float SubstanceProfile::*number = nullptr;
    bool SubstanceProfile::*flag = nullptr;
};

#define RT_FLOAT(key, lo, hi, unit, group) \
    FieldAccess{{#key, Kind::Float, lo, hi, unit, group}, &SubstanceProfile::key, nullptr}
#define RT_BOOL(key, group) \
    FieldAccess{{#key, Kind::Bool, 0.0f, 1.0f, "", group}, nullptr, &SubstanceProfile::key}

const std::vector<FieldAccess>& fieldTable() {
    static const std::vector<FieldAccess> table = {
        FieldAccess{{"category", Kind::Category, 0.0f, 0.0f, "", "Identity"}},
        FieldAccess{{"default_constitutive_model", Kind::Model, 0.0f, 0.0f, "", "Identity"}},
        RT_FLOAT(density, 1.0f, 30000.0f, "kg/m^3", "Thermal"),
        RT_FLOAT(specific_heat, 1.0f, 1.0e5f, "J/(kg K)", "Thermal"),
        RT_FLOAT(conductivity, 0.0f, 1000.0f, "W/(m K)", "Thermal"),
        RT_FLOAT(emissivity, 0.0f, 1.0f, "", "Thermal"),
        RT_FLOAT(thermal_response, 0.0f, 100.0f, "1/s", "Thermal"),
        RT_FLOAT(cooling_rate, 0.0f, 10.0f, "1/s", "Thermal"),
        RT_FLOAT(liquid_density, 1.0f, 30000.0f, "kg/m^3", "Liquid"),
        RT_FLOAT(liquid_kinematic_viscosity, 0.0f, 1000.0f, "m^2/s", "Liquid"),
        RT_FLOAT(absorbency, 0.0f, 1.0f, "", "Moisture"),
        RT_FLOAT(dry_rate, 0.0f, 10.0f, "1/s", "Moisture"),
        RT_BOOL(combustible, "Combustion"),
        RT_FLOAT(ignition_kelvin, 0.0f, 10000.0f, "K", "Combustion"),
        RT_FLOAT(fuel_capacity, 0.0f, 1000.0f, "", "Combustion"),
        RT_FLOAT(burn_rate, 0.0f, 100.0f, "1/s", "Combustion"),
        RT_FLOAT(char_rate, 0.0f, 100.0f, "", "Combustion"),
        RT_FLOAT(ash_yield, 0.0f, 1.0f, "", "Combustion"),
        RT_FLOAT(smoke_yield, 0.0f, 100.0f, "", "Combustion"),
        RT_FLOAT(heat_release, 0.0f, 100.0f, "", "Combustion"),
        RT_FLOAT(flame_level, 0.0f, 1.0f, "", "Combustion"),
        RT_BOOL(fluid_flammable, "Liquid fuel"),
        RT_BOOL(fluid_extinguishing, "Liquid fuel"),
        RT_FLOAT(flash_kelvin, 0.0f, 10000.0f, "K", "Liquid fuel"),
        RT_FLOAT(autoignition_kelvin, 0.0f, 10000.0f, "K", "Liquid fuel"),
        RT_FLOAT(latent_heat_vaporization, 0.0f, 1.0e8f, "J/kg", "Liquid fuel"),
        RT_FLOAT(vaporization_rate, 0.0f, 100.0f, "1/s", "Liquid fuel"),
        RT_FLOAT(cooling_power, 0.0f, 100.0f, "", "Liquid fuel"),
        RT_FLOAT(oxygen_dilution, 0.0f, 1.0f, "", "Liquid fuel"),
        RT_FLOAT(flame_persistence, 0.0f, 100.0f, "", "Liquid fuel"),
        RT_BOOL(meltable, "Phase change"),
        RT_FLOAT(melt_kelvin, 0.0f, 10000.0f, "K", "Phase change"),
        RT_FLOAT(boiling_kelvin, 0.0f, 10000.0f, "K", "Phase change"),
        RT_FLOAT(latent_heat_fusion, 0.0f, 1.0e8f, "J/kg", "Phase change"),
        RT_FLOAT(melt_viscosity, 0.0f, 1.0f, "", "Phase change"),
        // Granular ranges match the solver's own clamps (sanitizeGranularMaterial)
        // so a value the library accepts is a value the solver runs unchanged.
        FieldAccess{{"granular_transport", Kind::Transport, 0.0f, 0.0f, "", "Granular"}},
        RT_FLOAT(granular_friction_degrees, 0.0f, 55.0f, "deg", "Granular"),
        RT_FLOAT(granular_cohesion, 0.0f, 1.0e7f, "Pa", "Granular"),
        RT_FLOAT(granular_dilatancy_degrees, 0.0f, 30.0f, "deg", "Granular"),
        RT_FLOAT(granular_young_modulus, 10.0f, 1.0e7f, "Pa", "Granular"),
        RT_FLOAT(granular_poisson_ratio, 0.0f, 0.49f, "", "Granular"),
        RT_FLOAT(granular_tensile_cutoff, 0.0f, 1.0e7f, "Pa", "Granular"),
        RT_FLOAT(granular_hardening, 0.0f, 100.0f, "", "Granular"),
        RT_FLOAT(granular_compaction_hardening, 0.0f, 30.0f, "", "Granular"),
        RT_FLOAT(granular_compaction_limit, 0.0f, 0.5f, "", "Granular"),
        RT_FLOAT(granular_fracture_strain, 1.0e-4f, 1.0f, "", "Granular"),
        RT_FLOAT(granular_damage_rate, 0.0f, 100.0f, "1/s", "Granular"),
        RT_FLOAT(granular_healing_rate, 0.0f, 20.0f, "1/s", "Granular"),
        RT_BOOL(granular_rebonding, "Granular"),
        RT_FLOAT(granular_softening_kelvin, 0.0f, 10000.0f, "K", "Granular"),
        RT_FLOAT(granular_softening_range, 1.0f, 2000.0f, "K", "Granular"),
        RT_FLOAT(granular_residual_strength, 0.0f, 1.0f, "", "Granular"),
        RT_FLOAT(granular_tack_peak, 0.0f, 20.0f, "", "Granular"),
        RT_FLOAT(parcel_conduction, 0.0f, 200.0f, "1/s", "Granular"),
        // Grains (DEM): ranges are the grain solver's own limits.
        RT_FLOAT(grain_friction, 0.0f, 2.0f, "", "Grains (DEM)"),
        RT_FLOAT(grain_rolling_friction, 0.0f, 1.0f, "", "Grains (DEM)"),
        RT_FLOAT(grain_twisting_friction, 0.0f, 1.0f, "", "Grains (DEM)"),
        RT_FLOAT(grain_restitution, 0.01f, 1.0f, "", "Grains (DEM)"),
        RT_FLOAT(grain_tangential_stiffness_ratio, 0.0f, 1.0f, "", "Grains (DEM)"),
        RT_FLOAT(grain_packing_fraction, 0.3f, 0.74f, "", "Grains (DEM)"),
        RT_FLOAT(grain_real_radius_m, 0.0f, 1.0f, "m", "Grains (DEM)"),
        RT_FLOAT(grain_water_capacity_fraction, 0.0f, 0.5f, "", "Grains (DEM)"),
        RT_FLOAT(grain_absorption_rate_per_s, 0.0f, 1000.0f, "1/s", "Grains (DEM)"),
        RT_FLOAT(grain_drying_rate_per_s, 0.0f, 100.0f, "1/s", "Grains (DEM)"),
        RT_FLOAT(grain_contact_angle_deg, 0.0f, 89.0f, "deg", "Grains (DEM)"),
        RT_FLOAT(liquid_surface_tension_n_m, 0.0f, 1.0f, "N/m", "Liquid"),
        RT_FLOAT(liquid_freeze_viscosity_range, 1.0f, 2000.0f, "K", "Liquid"),
        RT_FLOAT(liquid_cold_viscosity, 0.0f, 1000.0f, "m^2/s", "Liquid"),
        RT_FLOAT(solver_flip_blend, 0.0f, 1.0f, "", "Solver hints"),
        RT_FLOAT(solver_apic_blend, 0.0f, 1.0f, "", "Solver hints"),
        RT_FLOAT(solver_velocity_damping, 0.0f, 1.0f, "", "Solver hints"),
        RT_FLOAT(solver_density_correction, 0.0f, 10.0f, "", "Solver hints"),
        RT_FLOAT(solver_air_drag, 0.0f, 100.0f, "1/m", "Solver hints"),
        RT_FLOAT(solver_wall_damping, 0.0f, 1.0f, "", "Solver hints"),
        RT_FLOAT(solver_affine_damping, 0.0f, 1.0f, "", "Solver hints"),
        RT_FLOAT(solver_max_velocity, 0.1f, 1000.0f, "m/s", "Solver hints"),
        RT_FLOAT(solver_viscosity_sweeps, 1.0f, 256.0f, "", "Solver hints"),
        RT_FLOAT(solver_viscosity_wall_slip, 0.0f, 1.0f, "", "Solver hints"),
        RT_FLOAT(solver_internal_friction, 0.0f, 100.0f, "1/s", "Solver hints"),
        RT_FLOAT(solver_granular_max_substeps, 1.0f, 64.0f, "", "Solver hints"),
        RT_BOOL(solver_thermal_chain, "Solver hints"),
        RT_FLOAT(solver_air_cooling_rate, 0.0f, 100.0f, "1/s", "Solver hints"),
        RT_FLOAT(solver_contact_cooling_rate, 0.0f, 100.0f, "1/s", "Solver hints"),
        FieldAccess{{"char_color", Kind::Color, 0.0f, 1.0f, "", "Optical"}},
        RT_FLOAT(molten_emission, 0.0f, 100.0f, "", "Optical"),
    };
    return table;
}

#undef RT_FLOAT
#undef RT_BOOL

const FieldAccess* findField(const std::string& key) {
    for (const auto& field : fieldTable()) {
        if (key == field.spec.key) return &field;
    }
    return nullptr;
}

json fieldValue(const SubstanceProfile& p, const FieldAccess& field) {
    switch (field.spec.kind) {
        case Kind::Float: return p.*field.number;
        case Kind::Bool: return p.*field.flag;
        case Kind::Model: return Fluid::matterConstitutiveModelName(p.default_constitutive_model);
        case Kind::Transport: return Fluid::matterGranularTransportName(p.granular_transport);
        case Kind::Category: return substanceCategoryName(p.category);
        case Kind::Color: return json::array({p.char_color[0], p.char_color[1], p.char_color[2]});
    }
    return nullptr;
}

bool setField(SubstanceProfile& p, const FieldAccess& field, const json& value,
              std::string& error) {
    const std::string key = field.spec.key;
    const auto in_range = [&](double v) {
        if (!std::isfinite(v) || v < field.spec.min || v > field.spec.max) {
            error = key + " = " + std::to_string(v) + " is outside [" +
                std::to_string(field.spec.min) + ", " + std::to_string(field.spec.max) + "]";
            return false;
        }
        return true;
    };
    switch (field.spec.kind) {
        case Kind::Float:
            if (!value.is_number()) { error = key + " expects a number"; return false; }
            if (!in_range(value.get<double>())) return false;
            p.*field.number = value.get<float>();
            return true;
        case Kind::Bool:
            if (!value.is_boolean()) { error = key + " expects true/false"; return false; }
            p.*field.flag = value.get<bool>();
            return true;
        case Kind::Model: {
            Fluid::MatterConstitutiveModel model;
            if (!value.is_string() ||
                !Fluid::parseMatterConstitutiveModel(value.get<std::string>(), model) ||
                model == Fluid::MatterConstitutiveModel::Auto) {
                error = key + " expects fluid|granular|elastic";
                return false;
            }
            p.default_constitutive_model = model;
            return true;
        }
        case Kind::Transport: {
            Fluid::MatterGranularTransport transport;
            if (!value.is_string() ||
                !Fluid::parseMatterGranularTransport(value.get<std::string>(), transport)) {
                error = key + " expects dem|mpm";
                return false;
            }
            p.granular_transport = transport;
            return true;
        }
        case Kind::Category: {
            SubstanceCategory category;
            if (!value.is_string() ||
                !parseSubstanceCategory(value.get<std::string>(), category)) {
                error = key + " expects liquid|granular|solid|fuel";
                return false;
            }
            p.category = category;
            return true;
        }
        case Kind::Color:
            if (!value.is_array() || value.size() != 3) {
                error = key + " expects [r, g, b]";
                return false;
            }
            for (int c = 0; c < 3; ++c) {
                if (!value[c].is_number() || !in_range(value[c].get<double>())) {
                    if (error.empty()) error = key + " expects numbers";
                    return false;
                }
            }
            for (int c = 0; c < 3; ++c) p.char_color[c] = value[c].get<float>();
            return true;
    }
    error = key + ": unknown field kind";
    return false;
}

// Rules spanning fields. Checked on every resolved profile, built-ins included,
// so a derived substance cannot reach a state no built-in could.
bool validateProfile(const SubstanceProfile& p, std::string& error) {
    if (p.meltable && !(p.melt_kelvin < p.boiling_kelvin)) {
        error = p.name + ": melt_kelvin must be below boiling_kelvin";
        return false;
    }
    return true;
}

bool validName(const std::string& name, std::string& error) {
    if (name.empty() || name.size() > 63) {
        error = "substance name must be 1..63 characters";
        return false;
    }
    if (std::isspace(static_cast<unsigned char>(name.front())) ||
        std::isspace(static_cast<unsigned char>(name.back()))) {
        error = "substance name has leading/trailing whitespace: '" + name + "'";
        return false;
    }
    return true;
}

struct Snapshot {
    std::vector<const SubstanceProfile*> list;
    std::unordered_map<std::string, const SubstanceProfile*> by_name;
    std::unordered_map<uint32_t, const SubstanceProfile*> by_tag;
};

struct Library {
    std::mutex write;                        // editors only; readers never lock
    std::vector<ProjectSubstance> project;   // creation order
    std::deque<SubstanceProfile> storage;    // never erased: published pointers stay valid
    std::vector<std::unique_ptr<Snapshot>> snapshots;  // never freed, same reason
    std::atomic<const Snapshot*> current{nullptr};
    std::atomic<uint64_t> revision{0};
};

Library& library() {
    static Library instance;
    return instance;
}

// Resolves `project` against the built-ins into a candidate snapshot without
// publishing it. Profiles land in `staged`; the caller moves them to storage.
bool resolve(const std::vector<ProjectSubstance>& project,
             std::deque<SubstanceProfile>& staged, std::unique_ptr<Snapshot>& out,
             std::string& error) {
    auto snapshot = std::make_unique<Snapshot>();
    const auto add = [&](const SubstanceProfile* p) {
        const uint32_t tag = Fluid::substanceTag(p->name);
        if (snapshot->by_name.count(p->name)) {
            error = "substance name already exists: " + p->name;
            return false;
        }
        if (snapshot->by_tag.count(tag)) {
            error = "substance '" + p->name + "' hashes to the same tag as '" +
                snapshot->by_tag[tag]->name + "'; pick another name";
            return false;
        }
        snapshot->list.push_back(p);
        snapshot->by_name.emplace(p->name, p);
        snapshot->by_tag.emplace(tag, p);
        return true;
    };
    for (const auto& builtin : builtinSubstanceProfiles()) {
        if (!add(&builtin)) return false;
    }
    for (const auto& item : project) {
        if (!validName(item.name, error)) return false;
        const auto base = snapshot->by_name.find(item.based_on);
        if (base == snapshot->by_name.end()) {
            // Also catches a cycle: a base must precede what derives from it.
            error = "substance '" + item.name + "' is based on '" + item.based_on +
                "', which does not exist (or is defined after it)";
            return false;
        }
        SubstanceProfile resolved = *base->second;
        resolved.name = item.name;
        resolved.based_on = item.based_on;
        if (!item.overrides.is_object()) {
            error = "substance '" + item.name + "': overrides must be an object";
            return false;
        }
        for (const auto& [key, value] : item.overrides.items()) {
            const FieldAccess* field = findField(key);
            if (!field) {
                error = "substance '" + item.name + "': unknown field '" + key + "'";
                return false;
            }
            std::string field_error;
            if (!setField(resolved, *field, value, field_error)) {
                error = "substance '" + item.name + "': " + field_error;
                return false;
            }
        }
        if (!validateProfile(resolved, error)) return false;
        staged.push_back(std::move(resolved));
        if (!add(&staged.back())) return false;
    }
    out = std::move(snapshot);
    return true;
}

// Caller holds lib.write.
bool publish(Library& lib, std::vector<ProjectSubstance> project, std::string& error) {
    std::deque<SubstanceProfile> staged;
    std::unique_ptr<Snapshot> snapshot;
    if (!resolve(project, staged, snapshot, error)) return false;
    // Re-point the snapshot at permanent storage before anyone can read it.
    std::unordered_map<const SubstanceProfile*, const SubstanceProfile*> moved;
    for (auto& profile : staged) {
        lib.storage.push_back(std::move(profile));
        moved.emplace(&profile, &lib.storage.back());
    }
    const auto fix = [&](const SubstanceProfile*& p) {
        const auto it = moved.find(p);
        if (it != moved.end()) p = it->second;
    };
    for (auto& p : snapshot->list) fix(p);
    for (auto& [name, p] : snapshot->by_name) fix(p);
    for (auto& [tag, p] : snapshot->by_tag) fix(p);
    lib.project = std::move(project);
    lib.snapshots.push_back(std::move(snapshot));
    lib.current.store(lib.snapshots.back().get(), std::memory_order_release);
    lib.revision.fetch_add(1, std::memory_order_acq_rel);
    return true;
}

const Snapshot& snapshot() {
    Library& lib = library();
    if (const Snapshot* s = lib.current.load(std::memory_order_acquire)) return *s;
    std::lock_guard<std::mutex> lock(lib.write);
    if (const Snapshot* s = lib.current.load(std::memory_order_acquire)) return *s;
    std::string error;
    publish(lib, {}, error);  // built-ins only; they are validated by the contract check
    return *lib.current.load(std::memory_order_acquire);
}

} // namespace

const char* substanceCategoryName(SubstanceCategory category) {
    switch (category) {
        case SubstanceCategory::Liquid: return "liquid";
        case SubstanceCategory::Granular: return "granular";
        case SubstanceCategory::Fuel: return "fuel";
        default: return "solid";
    }
}

bool parseSubstanceCategory(const std::string& text, SubstanceCategory& out) {
    if (text == "liquid") out = SubstanceCategory::Liquid;
    else if (text == "granular") out = SubstanceCategory::Granular;
    else if (text == "solid") out = SubstanceCategory::Solid;
    else if (text == "fuel") out = SubstanceCategory::Fuel;
    else return false;
    return true;
}

std::vector<const SubstanceProfile*> substanceProfiles() {
    return snapshot().list;
}

const SubstanceProfile* tryFindSubstance(const std::string& name) {
    const auto& by_name = snapshot().by_name;
    const auto it = by_name.find(name);
    return it != by_name.end() ? it->second : nullptr;
}

const SubstanceProfile* tryFindSubstanceByTag(uint32_t tag) {
    if (tag == Fluid::kSubstanceUntagged) return nullptr;
    const auto& by_tag = snapshot().by_tag;
    const auto it = by_tag.find(tag);
    return it != by_tag.end() ? it->second : nullptr;
}

const SubstanceProfile& findSubstance(const std::string& name) {
    if (const SubstanceProfile* profile = tryFindSubstance(name)) return *profile;
    // Unknown name (project authored against a newer build, or an empty string
    // from a collider that predates substances): fall back to the default rather
    // than refusing to load.
    return builtinSubstanceProfiles().front();
}

uint64_t substanceLibraryRevision() {
    snapshot();
    return library().revision.load(std::memory_order_acquire);
}

const std::vector<SubstanceFieldSpec>& substanceFieldSpecs() {
    static const std::vector<SubstanceFieldSpec> specs = [] {
        std::vector<SubstanceFieldSpec> out;
        for (const auto& field : fieldTable()) out.push_back(field.spec);
        return out;
    }();
    return specs;
}

json substanceFieldsToJson(const SubstanceProfile& profile) {
    json out = json::object();
    for (const auto& field : fieldTable()) out[field.spec.key] = fieldValue(profile, field);
    return out;
}

bool isBuiltinSubstance(const std::string& name) {
    for (const auto& builtin : builtinSubstanceProfiles()) {
        if (builtin.name == name) return true;
    }
    return false;
}

std::vector<std::string> substanceOverriddenFields(const std::string& name) {
    Library& lib = library();
    std::lock_guard<std::mutex> lock(lib.write);
    for (const auto& item : lib.project) {
        if (item.name != name) continue;
        std::vector<std::string> keys;
        for (const auto& [key, value] : item.overrides.items()) keys.push_back(key);
        return keys;
    }
    return {};
}

bool deriveSubstance(const std::string& name, const std::string& based_on,
                     std::string& error) {
    snapshot();
    Library& lib = library();
    std::lock_guard<std::mutex> lock(lib.write);
    auto project = lib.project;
    project.push_back({name, based_on, json::object()});
    return publish(lib, std::move(project), error);
}

bool patchSubstance(const std::string& name, const json& fields, std::string& error) {
    if (!fields.is_object() || fields.empty()) {
        error = "substance patch expects a non-empty object of fields";
        return false;
    }
    if (isBuiltinSubstance(name)) {
        error = "'" + name + "' is a built-in substance and read-only; derive from it";
        return false;
    }
    snapshot();
    Library& lib = library();
    std::lock_guard<std::mutex> lock(lib.write);
    auto project = lib.project;
    auto it = std::find_if(project.begin(), project.end(),
        [&](const ProjectSubstance& item) { return item.name == name; });
    if (it == project.end()) {
        error = "unknown substance: " + name;
        return false;
    }
    for (const auto& [key, value] : fields.items()) {
        if (!findField(key)) {
            error = "unknown substance field '" + key + "'";
            return false;
        }
        if (value.is_null()) it->overrides.erase(key);
        else it->overrides[key] = value;
    }
    return publish(lib, std::move(project), error);
}

bool removeSubstance(const std::string& name, std::string& error) {
    if (isBuiltinSubstance(name)) {
        error = "'" + name + "' is a built-in substance and cannot be removed";
        return false;
    }
    snapshot();
    Library& lib = library();
    std::lock_guard<std::mutex> lock(lib.write);
    auto project = lib.project;
    auto it = std::find_if(project.begin(), project.end(),
        [&](const ProjectSubstance& item) { return item.name == name; });
    if (it == project.end()) {
        error = "unknown substance: " + name;
        return false;
    }
    for (const auto& item : project) {
        if (item.based_on == name) {
            error = "'" + item.name + "' derives from '" + name + "'; remove it first";
            return false;
        }
    }
    project.erase(it);
    return publish(lib, std::move(project), error);
}

std::vector<ProjectSubstance> projectSubstances() {
    Library& lib = library();
    std::lock_guard<std::mutex> lock(lib.write);
    return lib.project;
}

bool replaceProjectSubstances(const std::vector<ProjectSubstance>& substances,
                              std::string& error) {
    snapshot();
    Library& lib = library();
    std::lock_guard<std::mutex> lock(lib.write);
    return publish(lib, substances, error);
}

} // namespace RayTrophiSim
