#include "Animation/RigAnatomy.h"
#include "Animation/RigDrivenControls.h"
#include "Animation/NodeHierarchy.h"
#include "json.hpp"
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <cmath>

namespace RigAuthoring {
namespace {
bool identifier(const std::string& id) {
    if (id.empty() || id.size() > 128)
        return false;
    for (unsigned char c : id)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
              c == '_' || c == '-' || c == '.'))
            return false;
    return true;
}
bool fields(const nlohmann::json& value, std::initializer_list<const char*> allowed) {
    if (!value.is_object())
        return false;
    for (auto it = value.begin(); it != value.end(); ++it) {
        bool found = false;
        for (const char* key : allowed)
            if (it.key() == key) {
                found = true;
                break;
            }
        if (!found)
            return false;
    }
    return true;
}
}
bool validateRigAnatomy(const RigAnatomy& a, const RayTrophi::NodeHierarchy& h,
                        std::string& error) {
    error.clear();
    if (a.family != "custom" && a.family != "humanoid" && a.family != "quadruped" &&
        a.family != "insect" && a.family != "avian") {
        error = "rig_anatomy_invalid_family";
        return false;
    }
    if (a.roles.size() > 4096 || a.symmetry.size() > 4096 || a.chains.size() > 1024) {
        error = "rig_anatomy_limit";
        return false;
    }
    std::unordered_map<std::string, int> nodes;
    for (size_t i = 0; i < h.size(); ++i) {
        if (!nodes.emplace(h.nodes[i].uniqueName, static_cast<int>(i)).second) {
            error = "rig_invalid_hierarchy";
            return false;
        }
    }
    auto known = [&](const std::string& key) { return nodes.count(key) != 0; };
    std::unordered_set<std::string> roles, pairs, chains;
    for (const auto& role : a.roles) {
        if (!identifier(role.role)) {
            error = "rig_anatomy_invalid_role";
            return false;
        }
        if (!roles.insert(role.role).second) {
            error = "rig_anatomy_duplicate_role";
            return false;
        }
        if (!known(role.bone)) {
            error = "rig_anatomy_unknown_bone";
            return false;
        }
    }
    for (const auto& pair : a.symmetry) {
        if (!known(pair.left) || !known(pair.right)) {
            error = "rig_anatomy_unknown_bone";
            return false;
        }
        if (pair.left == pair.right || !pairs.insert(pair.left).second ||
            !pairs.insert(pair.right).second) {
            error = "rig_anatomy_invalid_symmetry";
            return false;
        }
    }
    for (const auto& chain : a.chains) {
        if (!identifier(chain.name) || !chains.insert(chain.name).second) {
            error = "rig_anatomy_invalid_chain_name";
            return false;
        }
        if (chain.bones.size() < 2 || chain.bones.size() > 4096) {
            error = "rig_anatomy_invalid_chain_length";
            return false;
        }
        std::unordered_set<std::string> visited;
        int previous = -1;
        for (const auto& bone : chain.bones) {
            const auto node = nodes.find(bone);
            if (node == nodes.end()) {
                error = "rig_anatomy_unknown_bone";
                return false;
            }
            if (!visited.insert(bone).second) {
                error = "rig_anatomy_duplicate_chain_bone";
                return false;
            }
            if (previous >= 0 && h.nodes[node->second].parent != previous) {
                error = "rig_anatomy_chain_disconnected";
                return false;
            }
            previous = node->second;
        }
    }
    std::unordered_set<std::string> derived;
    for (const auto& rule : a.fitRules) {
        if (!known(rule.bone) || !known(rule.start) || !known(rule.end)) {
            error = "rig_anatomy_unknown_bone";
            return false;
        }
        if (rule.bone == rule.start || rule.bone == rule.end || rule.start == rule.end ||
            !std::isfinite(rule.position) || rule.position <= 0.f || rule.position >= 1.f ||
            !derived.insert(rule.bone).second) {
            error = "rig_anatomy_invalid_fit_rule";
            return false;
        }
    }
    for (const auto& rule : a.fitRules) {
        if (derived.count(rule.start) || derived.count(rule.end)) {
            error = "rig_anatomy_fit_rule_dependency";
            return false;
        }
    }
    if (a.drivenControls.size() > 4096) {
        error = "rig_anatomy_limit";
        return false;
    }
    std::unordered_set<std::string> drivenIds;
    size_t driverCount = 0;
    for (const auto& control : a.drivenControls) {
        if (!identifier(control.id) || !identifier(control.group) || control.label.empty() ||
            control.label.size() > 128 || !known(control.anchor) ||
            (control.side != "left" && control.side != "right" &&
             control.side != "center") ||
            (control.shape != "ring" && control.shape != "hand" &&
             control.shape != "foot" && control.shape != "arc" &&
             control.shape != "root" && control.shape != "aim" &&
             control.shape != "diamond") ||
            !std::isfinite(control.minimum) || !std::isfinite(control.maximum) ||
            !std::isfinite(control.defaultValue) || control.minimum >= control.maximum ||
            control.defaultValue < control.minimum || control.defaultValue > control.maximum ||
            control.drivers.empty() || control.drivers.size() > 4096 ||
            !drivenIds.insert(control.id).second) {
            error = "rig_anatomy_invalid_driven_control";
            return false;
        }
        driverCount += control.drivers.size();
        if (driverCount > 16384) {
            error = "rig_anatomy_limit";
            return false;
        }
        for (const auto& driver : control.drivers) {
            if (!known(driver.bone) || !std::isfinite(driver.axis.x) ||
                !std::isfinite(driver.axis.y) || !std::isfinite(driver.axis.z) ||
                driver.axis.length_squared() < 1e-8f || !std::isfinite(driver.degrees) ||
                std::fabs(driver.degrees) < 1e-6f || std::fabs(driver.degrees) > 360.f) {
                error = "rig_anatomy_invalid_control_driver";
                return false;
            }
        }
    }
    return validateJointRules(a.joints, h, error) && validateIKControls(a.controls, h, error);
}
nlohmann::json serializeRigAnatomy(const RigAnatomy& a) {
    auto roles = nlohmann::json::array(), symmetry = nlohmann::json::array(),
         chains = nlohmann::json::array();
    for (const auto& r : a.roles)
        roles.push_back({{"role", r.role}, {"bone", r.bone}});
    for (const auto& p : a.symmetry)
        symmetry.push_back({{"left", p.left}, {"right", p.right}});
    for (const auto& c : a.chains)
        chains.push_back({{"name", c.name}, {"bones", c.bones}});
    nlohmann::json result = {{"version", 1},
                             {"family", a.family},
                             {"roles", roles},
                             {"symmetry", symmetry},
                             {"chains", chains}};
    if (!a.joints.empty()) {
        result["version"] = 2;
        result["joint_profile"] = serializeJointRules(a.joints);
    }
    if (!a.controls.empty()) {
        result["version"] = 3;
        result["controls"] = serializeIKControls(a.controls);
    }
    for (const auto& control : a.controls)
        if (!control.chain.empty())
            result["version"] = 4;
    for (const auto& control : a.controls)
        if (control.solver == "aim")
            result["version"] = 5;
    if (!a.fitRules.empty()) {
        auto rules = nlohmann::json::array();
        for (const auto& rule : a.fitRules) {
            rules.push_back({{"bone", rule.bone},
                             {"start", rule.start},
                             {"end", rule.end},
                             {"position", rule.position}});
        }
        result["version"] = 6;
        result["fit_rules"] = std::move(rules);
    }
    if (!a.drivenControls.empty()) {
        result["version"] = 7;
        result["control_rig"] = serializeRigDrivenControls(a.drivenControls);
    }
    return result;
}
bool deserializeRigAnatomy(const nlohmann::json& value, const RayTrophi::NodeHierarchy& h,
                           RigAnatomy& output, std::string& error) {
    error.clear();
    RigAnatomy a;
    try {
        if (!fields(value, {"version", "family", "roles", "symmetry", "chains", "joint_profile",
                            "controls", "fit_rules", "control_rig"}) ||
            !value.contains("version") || !value.at("version").is_number_integer() ||
            (value.at("version") != 1 && value.at("version") != 2 && value.at("version") != 3 &&
             value.at("version") != 4 && value.at("version") != 5 &&
             value.at("version") != 6 && value.at("version") != 7) ||
            !value.contains("family") || !value.at("family").is_string()) {
            error = "rig_anatomy_invalid_schema";
            return false;
        }
        for (const char* key : {"roles", "symmetry", "chains"})
            if (!value.contains(key) || !value.at(key).is_array()) {
                error = "rig_anatomy_invalid_schema";
                return false;
            }
        if (value.at("roles").size() > 4096 || value.at("symmetry").size() > 4096 ||
            value.at("chains").size() > 1024) {
            error = "rig_anatomy_limit";
            return false;
        }
        a.family = value.at("family").get<std::string>();
        for (const auto& r : value.at("roles")) {
            if (!fields(r, {"role", "bone"}) || !r.contains("role") || !r.at("role").is_string() ||
                !r.contains("bone") || !r.at("bone").is_string()) {
                error = "rig_anatomy_invalid_schema";
                return false;
            }
            a.roles.push_back({r.at("role").get<std::string>(), r.at("bone").get<std::string>()});
        }
        for (const auto& p : value.at("symmetry")) {
            if (!fields(p, {"left", "right"}) || !p.contains("left") || !p.at("left").is_string() ||
                !p.contains("right") || !p.at("right").is_string()) {
                error = "rig_anatomy_invalid_schema";
                return false;
            }
            a.symmetry.push_back(
                {p.at("left").get<std::string>(), p.at("right").get<std::string>()});
        }
        for (const auto& c : value.at("chains")) {
            if (!fields(c, {"name", "bones"}) || !c.contains("name") || !c.at("name").is_string() ||
                !c.contains("bones") || !c.at("bones").is_array()) {
                error = "rig_anatomy_invalid_schema";
                return false;
            }
            if (c.at("bones").size() > 4096) {
                error = "rig_anatomy_limit";
                return false;
            }
            RigChain chain;
            chain.name = c.at("name").get<std::string>();
            for (const auto& bone : c.at("bones")) {
                if (!bone.is_string()) {
                    error = "rig_anatomy_invalid_schema";
                    return false;
                }
                chain.bones.push_back(bone.get<std::string>());
            }
            a.chains.push_back(std::move(chain));
        }
        if (value.at("version") == 2 && !value.contains("joint_profile")) {
            error = "rig_anatomy_invalid_schema";
            return false;
        }
        if (value.contains("joint_profile")) {
            if (value.at("version") == 1) {
                error = "rig_anatomy_invalid_schema";
                return false;
            }
            if (!deserializeJointRules(value["joint_profile"], h, a.joints, error))
                return false;
        }
        if ((value.at("version") >= 3 && value.at("version") <= 5) ||
            value.contains("controls")) {
            if (!value.contains("controls") ||
                !deserializeIKControls(value["controls"], h, a.controls, error)) {
                if (error.empty())
                    error = "rig_anatomy_invalid_schema";
                return false;
            }
        } else if (value.contains("controls")) {
            error = "rig_anatomy_invalid_schema";
            return false;
        }
        if (value.at("version") < 4)
            for (const auto& control : a.controls)
                if (!control.chain.empty()) {
                    error = "rig_anatomy_invalid_schema";
                    return false;
                }
        if (value.at("version") < 5)
            for (const auto& control : a.controls)
                if (control.solver == "aim") {
                    error = "rig_anatomy_invalid_schema";
                    return false;
                }
        if (value.at("version") >= 6) {
            if ((value.at("version") == 6 && !value.contains("fit_rules")) ||
                (value.contains("fit_rules") && !value.at("fit_rules").is_array()) ||
                (value.contains("fit_rules") &&
                 value.at("fit_rules").size() > 4096)) {
                error = "rig_anatomy_invalid_schema";
                return false;
            }
            for (const auto& item : value.value("fit_rules", nlohmann::json::array())) {
                if (!fields(item, {"bone", "start", "end", "position"}) ||
                    !item.contains("bone") || !item.at("bone").is_string() ||
                    !item.contains("start") || !item.at("start").is_string() ||
                    !item.contains("end") || !item.at("end").is_string() ||
                    !item.contains("position") || !item.at("position").is_number()) {
                    error = "rig_anatomy_invalid_schema";
                    return false;
                }
                a.fitRules.push_back({item.at("bone").get<std::string>(),
                                      item.at("start").get<std::string>(),
                                      item.at("end").get<std::string>(),
                                      item.at("position").get<float>()});
            }
        } else if (value.contains("fit_rules")) {
            error = "rig_anatomy_invalid_schema";
            return false;
        }
        if (value.at("version") == 7) {
            if (!value.contains("control_rig") || !value.at("control_rig").is_array() ||
                value.at("control_rig").empty() || value.at("control_rig").size() > 4096) {
                error = "rig_anatomy_invalid_schema";
                return false;
            }
            size_t driverCount = 0;
            for (const auto& item : value.at("control_rig")) {
                if (!fields(item, {"id", "label", "group", "anchor", "side", "shape",
                                   "minimum", "maximum", "default", "drivers"}) ||
                    !item.contains("id") || !item.at("id").is_string() ||
                    !item.contains("label") || !item.at("label").is_string() ||
                    !item.contains("group") || !item.at("group").is_string() ||
                    !item.contains("anchor") || !item.at("anchor").is_string() ||
                    !item.contains("side") || !item.at("side").is_string() ||
                    !item.contains("shape") || !item.at("shape").is_string() ||
                    !item.contains("minimum") || !item.at("minimum").is_number() ||
                    !item.contains("maximum") || !item.at("maximum").is_number() ||
                    !item.contains("default") || !item.at("default").is_number() ||
                    !item.contains("drivers") || !item.at("drivers").is_array()) {
                    error = "rig_anatomy_invalid_schema";
                    return false;
                }
                driverCount += item.at("drivers").size();
                if (item.at("drivers").size() > 4096 || driverCount > 16384) {
                    error = "rig_anatomy_limit";
                    return false;
                }
                RigDrivenControl control;
                control.id = item.at("id").get<std::string>();
                control.label = item.at("label").get<std::string>();
                control.group = item.at("group").get<std::string>();
                control.anchor = item.at("anchor").get<std::string>();
                control.side = item.at("side").get<std::string>();
                control.shape = item.at("shape").get<std::string>();
                control.minimum = item.at("minimum").get<float>();
                control.maximum = item.at("maximum").get<float>();
                control.defaultValue = item.at("default").get<float>();
                for (const auto& driver : item.at("drivers")) {
                    if (!fields(driver, {"bone", "axis", "degrees"}) ||
                        !driver.contains("bone") || !driver.at("bone").is_string() ||
                        !driver.contains("axis") || !driver.at("axis").is_array() ||
                        driver.at("axis").size() != 3 ||
                        !driver.contains("degrees") || !driver.at("degrees").is_number()) {
                        error = "rig_anatomy_invalid_schema";
                        return false;
                    }
                    for (const auto& component : driver.at("axis")) {
                        if (!component.is_number()) {
                            error = "rig_anatomy_invalid_schema";
                            return false;
                        }
                    }
                    control.drivers.push_back(
                        {driver.at("bone").get<std::string>(),
                         Vec3(driver.at("axis")[0].get<float>(),
                              driver.at("axis")[1].get<float>(),
                              driver.at("axis")[2].get<float>()),
                         driver.at("degrees").get<float>()});
                }
                a.drivenControls.push_back(std::move(control));
            }
        } else if (value.contains("control_rig")) {
            error = "rig_anatomy_invalid_schema";
            return false;
        }
        if (!validateRigAnatomy(a, h, error))
            return false;
        output = std::move(a);
        return true;
    } catch (const nlohmann::json::exception&) {
        error = "rig_anatomy_invalid_schema";
        return false;
    }
}
void renameAnatomyBone(RigAnatomy& a, const std::string& oldKey, const std::string& newKey) {
    for (auto& c : a.controls) {
        if (c.root == oldKey)
            c.root = newKey;
        if (c.mid == oldKey)
            c.mid = newKey;
        if (c.tip == oldKey)
            c.tip = newKey;
        for (auto& bone : c.chain)
            if (bone == oldKey)
                bone = newKey;
    }
    for (auto& j : a.joints)
        if (j.bone == oldKey)
            j.bone = newKey;
    for (auto& r : a.roles)
        if (r.bone == oldKey)
            r.bone = newKey;
    for (auto& p : a.symmetry) {
        if (p.left == oldKey)
            p.left = newKey;
        if (p.right == oldKey)
            p.right = newKey;
    }
    for (auto& c : a.chains)
        for (auto& b : c.bones)
            if (b == oldKey)
                b = newKey;
    for (auto& rule : a.fitRules) {
        if (rule.bone == oldKey)
            rule.bone = newKey;
        if (rule.start == oldKey)
            rule.start = newKey;
        if (rule.end == oldKey)
            rule.end = newKey;
    }
    for (auto& control : a.drivenControls) {
        if (control.anchor == oldKey)
            control.anchor = newKey;
        for (auto& driver : control.drivers)
            if (driver.bone == oldKey)
                driver.bone = newKey;
    }
}
bool anatomyReferencesBone(const RigAnatomy& a, const std::string& bone) {
    for (const auto& c : a.controls)
        for (const auto& key : ikControlBones(c))
            if (key == bone)
                return true;
    for (const auto& j : a.joints)
        if (j.bone == bone)
            return true;
    for (const auto& r : a.roles)
        if (r.bone == bone)
            return true;
    for (const auto& p : a.symmetry)
        if (p.left == bone || p.right == bone)
            return true;
    for (const auto& c : a.chains)
        for (const auto& b : c.bones)
            if (b == bone)
                return true;
    for (const auto& rule : a.fitRules)
        if (rule.bone == bone || rule.start == bone || rule.end == bone)
            return true;
    for (const auto& control : a.drivenControls) {
        if (control.anchor == bone)
            return true;
        for (const auto& driver : control.drivers)
            if (driver.bone == bone)
                return true;
    }
    return false;
}
bool buildLimbIKControls(const RigAnatomy& anatomy, const RayTrophi::NodeHierarchy& h,
                         std::vector<IKControl>& output, std::string& error) {
    if (!validateRigAnatomy(anatomy, h, error))
        return false;
    std::vector<IKControl> controls;
    for (const auto& chain : anatomy.chains) {
        const bool arm =
            chain.name.size() >= 4 && chain.name.compare(chain.name.size() - 4, 4, "_arm") == 0;
        const bool leg =
            chain.name.size() >= 4 && chain.name.compare(chain.name.size() - 4, 4, "_leg") == 0;
        if (!arm && !leg)
            continue;
        const size_t first = arm && anatomy.family == "humanoid" ? 1 : 0;
        if (chain.bones.size() < first + 3) {
            error = "rig_ik_invalid_chain";
            return false;
        }
        controls.push_back(
            {chain.name, chain.bones[first], chain.bones[first + 1], chain.bones[first + 2]});
    }
    if (controls.empty()) {
        error = "rig_ik_no_limb_chains";
        return false;
    }
    if (!validateIKControls(controls, h, error))
        return false;
    output = std::move(controls);
    return true;
}
}
