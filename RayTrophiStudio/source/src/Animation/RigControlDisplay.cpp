#include "Animation/RigControlDisplay.h"
#include "Animation/RigAnatomy.h"
#include "Animation/RigIK.h"
#include <algorithm>

namespace RigAuthoring {
namespace {

bool contains(const std::string& value, const char* token) {
    return value.find(token) != std::string::npos;
}

std::string roleForControl(const RigAnatomy& anatomy, const IKControl& control) {
    const auto bones = ikControlBones(control);
    for (auto bone = bones.rbegin(); bone != bones.rend(); ++bone) {
        const auto role = std::find_if(anatomy.roles.begin(), anatomy.roles.end(),
                                       [&](const RigRole& item) { return item.bone == *bone; });
        if (role != anatomy.roles.end()) {
            return role->role;
        }
    }
    return {};
}

std::string sideFromRole(const std::string& role, const std::string& name) {
    if (role.rfind("left_", 0) == 0 || name.rfind("left_", 0) == 0) {
        return "left";
    }
    if (role.rfind("right_", 0) == 0 || name.rfind("right_", 0) == 0) {
        return "right";
    }
    return "center";
}

nlohmann::json handleJson(const RigControlHandleDisplay& handle) {
    return {{"shape", handle.shape},
            {"level", handle.level},
            {"scale", handle.scale},
            {"channels", handle.channels}};
}

} // namespace

RigControlDisplay deriveRigControlDisplay(const RigAnatomy& anatomy,
                                          const IKControl& control) {
    RigControlDisplay display;
    const std::string role = roleForControl(anatomy, control);
    display.side = sideFromRole(role, control.name);
    display.colorRole = display.side;

    if (control.solver == "aim") {
        display.semantic = contains(role, "head") ? "head_aim" : "aim";
        display.target.shape = "aim";
        display.target.level = contains(role, "head") ? "primary" : "secondary";
        display.target.channels = {"translate"};
        return display;
    }
    if (contains(role, "_arm.hand")) {
        display.semantic = "hand";
        display.target.shape = "hand";
    } else if (contains(role, "_leg.ankle") || contains(role, "_leg.toe")) {
        display.semantic = "foot";
        display.target.shape = "foot";
    } else if (contains(role, "_hand.")) {
        display.semantic = "finger";
        display.target.shape = "arc";
        display.target.level = "secondary";
        display.target.scale = .8f;
    } else if (contains(role, "spine") || contains(role, "pelvis") ||
               contains(role, "chest") || contains(role, "head") || role == "root") {
        display.semantic = "body";
        display.target.shape = role == "root" ? "root" : "ring";
    } else {
        display.semantic = control.chain.empty() ? "limb" : "chain";
        display.target.level = "secondary";
    }
    display.target.channels = {"translate", "rotate", "ik_fk", "contact"};
    return display;
}

nlohmann::json serializeRigControlDisplay(const RigControlDisplay& display) {
    return {{"semantic", display.semantic},
            {"side", display.side},
            {"color_role", display.colorRole},
            {"target", handleJson(display.target)},
            {"pole", handleJson(display.pole)},
            {"fk", handleJson(display.fk)}};
}

nlohmann::json rigControlDisplayContract() {
    return {{"version", 2},
            {"space", "hybrid_anatomical_clamped"},
            {"scale_source", "length_world"},
            {"hit_space", "screen_constant_minimum"},
            {"levels", {"primary", "secondary", "deform"}},
            {"shapes", {"ring", "root", "hand", "foot", "aim", "diamond", "arc"}},
            {"color_roles", {"left", "right", "center", "selected", "warning"}},
            {"contact_marker", "inner_dot"}};
}

} // namespace RigAuthoring
