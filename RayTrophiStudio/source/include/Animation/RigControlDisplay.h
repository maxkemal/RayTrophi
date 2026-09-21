#pragma once

#include "json.hpp"
#include <string>
#include <vector>

namespace RigAuthoring {

struct IKControl;
struct RigAnatomy;

struct RigControlHandleDisplay {
    std::string shape = "ring";
    std::string level = "primary";
    float scale = 1.f;
    std::vector<std::string> channels;
};

struct RigControlDisplay {
    std::string semantic = "generic";
    std::string side = "center";
    std::string colorRole = "center";
    RigControlHandleDisplay target;
    RigControlHandleDisplay pole = {"diamond", "secondary", .78f, {"translate"}};
    RigControlHandleDisplay fk = {"ring", "secondary", .72f, {"rotate"}};
};

RigControlDisplay deriveRigControlDisplay(const RigAnatomy& anatomy, const IKControl& control);
nlohmann::json serializeRigControlDisplay(const RigControlDisplay& display);
nlohmann::json rigControlDisplayContract();

} // namespace RigAuthoring
