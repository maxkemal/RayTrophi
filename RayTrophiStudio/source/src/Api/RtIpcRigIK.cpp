#include "Api/RtApi.h"
#include "RtRigIKBindings.h"
using json = nlohmann::json;
namespace {
bool character(const json &p) {
    return p.contains("character") && p["character"].is_string();
}
bool revision(const json &p) {
    return p.contains("rig_revision") && p["rig_revision"].is_number_integer() &&
           (p["rig_revision"].is_number_unsigned() || p["rig_revision"].get<int64_t>() >= 0);
}
bool point(const json &value) {
    if (!value.is_array() || value.size() != 3)
        return false;
    for (const auto &v : value)
        if (!v.is_number())
            return false;
    return true;
}
Vec3 vector(const json &v) {
    return Vec3(v[0].get<float>(), v[1].get<float>(), v[2].get<float>());
}
bool walkRecipe(const json &params, RigAuthoring::HumanWalkRecipe &output) {
    for (const auto *key :
         {"fps", "cadence", "stride", "step_height", "body_bounce", "arm_swing"}) {
        if (!params.contains(key) || !params[key].is_number()) {
            return false;
        }
    }
    if (!params.contains("cycles") || !params["cycles"].is_number_integer()) {
        return false;
    }
    if (params.contains("body_motion") && !params["body_motion"].is_number()) {
        return false;
    }
    output.fps = params["fps"].get<float>();
    output.cadence = params["cadence"].get<float>();
    output.cycles = params["cycles"].get<int>();
    output.stride = params["stride"].get<float>();
    output.stepHeight = params["step_height"].get<float>();
    output.bodyBounce = params["body_bounce"].get<float>();
    output.armSwing = params["arm_swing"].get<float>();
    output.bodyMotion = params.value("body_motion", .75f);
    return true;
}
json response(const rtapi::Result &r) {
    return r.ok ? json{{"ok", true}} : json{{"__error", r.error}, {"code", r.error}};
}
} // namespace
bool dispatchRigIKIpc(const std::string &method, const json &params,
                      const RtIpcTemplateEnqueue &enqueue, json &out) {
    auto invalid = [&]() {
        out = {{"__error", "Invalid IK parameter type or missing required field"},
               {"code", "invalid_parameter"}};
        return true;
    };
    if (method == "rig.preview_human_walk" || method == "rig.create_human_walk_clip") {
        RigAuthoring::HumanWalkRecipe recipe;
        const bool create = method == "rig.create_human_walk_clip";
        if (!character(params) || !walkRecipe(params, recipe) ||
            (create &&
             (!revision(params) || !params.contains("name") || !params["name"].is_string()))) {
            return invalid();
        }
        const auto c = params.at("character").get<std::string>();
        if (create) {
            const auto name = params.at("name").get<std::string>();
            const auto r = params.at("rig_revision").get<uint64_t>();
            out = enqueue([c, name, recipe, r](UIContext &) {
                return response(rtapi::createRigHumanWalkClip(c, name, recipe, r));
            });
        } else {
            out = enqueue([c, recipe](UIContext &) {
                json value;
                const auto result = rtapi::previewRigHumanWalk(c, recipe, value);
                return result.ok ? value : response(result);
            });
        }
        return true;
    }
    if (method == "rig.get_ik_channels") {
        if (!character(params))
            return invalid();
        const auto c = params.at("character").get<std::string>();
        out = enqueue([c](UIContext &) {
            json value;
            auto r = rtapi::getRigIKChannels(c, value);
            return r.ok ? value : response(r);
        });
        return true;
    }
    if (method == "rig.insert_ik_key" || method == "rig.remove_ik_key" ||
        method == "rig.clear_ik_channels") {
        if (!character(params) || !revision(params) || !params.contains("control") ||
            !params["control"].is_string())
            return invalid();
        const auto c = params.at("character").get<std::string>();
        const auto control = params.at("control").get<std::string>();
        const auto r = params.at("rig_revision").get<uint64_t>();
        const bool clear = method == "rig.clear_ik_channels";
        const bool remove = method == "rig.remove_ik_key";
        out = enqueue([c, control, r, clear, remove](UIContext &) {
            return response(clear ? rtapi::clearRigIKChannels(c, control, r)
                            : remove ? rtapi::removeRigIKKey(c, control, r)
                                     : rtapi::insertRigIKKey(c, control, r));
        });
        return true;
    }
    if (method == "rig.set_ik_contact_interval" || method == "rig.bake_ik_channels") {
        auto frame = [&](const char *key) {
            if (!params.contains(key) || !params[key].is_number_integer())
                return false;
            const auto &v = params[key];
            return v.is_number_unsigned() ? v.get<uint64_t>() <= 1000000
                                          : (v.get<int64_t>() >= 0 && v.get<int64_t>() <= 1000000);
        };
        const bool bake = method == "rig.bake_ik_channels";
        const char *key = bake ? "name" : "control";
        if (!character(params) || !revision(params) || !params.contains(key) ||
            !params[key].is_string() || !frame("start_frame") || !frame("end_frame"))
            return invalid();
        const auto c = params.at("character").get<std::string>();
        const auto name = params.at(key).get<std::string>();
        const auto r = params.at("rig_revision").get<uint64_t>();
        const int start = params.at("start_frame").get<int>();
        const int end = params.at("end_frame").get<int>();
        out = enqueue([c, name, r, start, end, bake](UIContext &) {
            return response(bake ? rtapi::bakeRigIKChannels(c, name, start, end, r)
                                 : rtapi::setRigIKContactInterval(c, name, start, end, r));
        });
        return true;
    }
    if (method == "rig.create_chain_control") {
        if (!character(params) || !revision(params) || !params.contains("chain") ||
            !params["chain"].is_string())
            return invalid();
        const auto c = params.at("character").get<std::string>();
        const auto chain = params.at("chain").get<std::string>();
        const auto r = params.at("rig_revision").get<uint64_t>();
        out = enqueue([c, chain, r](UIContext &) {
            return response(rtapi::createRigChainControl(c, chain, r));
        });
        return true;
    }
    if (method == "rig.create_aim_control") {
        if (!character(params) || !revision(params) || !params.contains("role") ||
            !params["role"].is_string()) {
            return invalid();
        }
        const auto c = params.at("character").get<std::string>();
        const auto role = params.at("role").get<std::string>();
        const auto r = params.at("rig_revision").get<uint64_t>();
        out = enqueue(
            [c, role, r](UIContext &) { return response(rtapi::createRigAimControl(c, role, r)); });
        return true;
    }
    if (method == "rig.set_ik_spline") {
        if (!character(params) || !revision(params) || !params.contains("control") ||
            !params["control"].is_string() || !params.contains("enabled") ||
            !params["enabled"].is_boolean() || !params.contains("points_world") ||
            !params["points_world"].is_array() || params["points_world"].size() > 2) {
            return invalid();
        }
        std::vector<Vec3> points;
        for (const auto &value : params["points_world"]) {
            if (!point(value)) {
                return invalid();
            }
            points.push_back(vector(value));
        }
        const auto c = params.at("character").get<std::string>();
        const auto control = params.at("control").get<std::string>();
        const auto enabled = params.at("enabled").get<bool>();
        const auto r = params.at("rig_revision").get<uint64_t>();
        out = enqueue([c, control, enabled, r, points](UIContext &) {
            return response(rtapi::setRigIKSpline(c, control, points, enabled, r));
        });
        return true;
    }
    if (method == "rig.get_controls") {
        if (!character(params))
            return invalid();
        const auto c = params.at("character").get<std::string>();
        out = enqueue([c](UIContext &) {
            json value;
            auto r = rtapi::getRigControls(c, value);
            return r.ok ? value : response(r);
        });
        return true;
    }
    if (method == "rig.create_controls") {
        if (!character(params) || !revision(params))
            return invalid();
        if (params.contains("controls") && !params["controls"].is_null() &&
            !params["controls"].is_array())
            return invalid();
        const auto c = params.at("character").get<std::string>();
        const auto r = params.at("rig_revision").get<uint64_t>();
        const auto controls = params.value("controls", json());
        out = enqueue([c, r, controls](UIContext &) {
            return response(rtapi::createRigControls(c, controls, r));
        });
        return true;
    }
    if (method == "rig.select_control") {
        if (!character(params) || !params.contains("control") || !params["control"].is_string() ||
            (params.contains("handle") && !params["handle"].is_string()))
            return invalid();
        const auto c = params.at("character").get<std::string>(),
                   control = params.at("control").get<std::string>();
        const auto handle = params.value("handle", std::string("target"));
        out = enqueue([c, control, handle](UIContext &) {
            return response(rtapi::selectRigControl(c, control, handle));
        });
        return true;
    }
    if (method == "rig.set_ik_target") {
        if (!character(params) || !revision(params) || !params.contains("control") ||
            !params["control"].is_string() || !params.contains("target_world") ||
            !point(params["target_world"]) || !params.contains("pole_world") ||
            !point(params["pole_world"]))
            return invalid();
        const auto c = params.at("character").get<std::string>(),
                   control = params.at("control").get<std::string>();
        const auto r = params.at("rig_revision").get<uint64_t>();
        const auto target = vector(params.at("target_world")),
                   pole = vector(params.at("pole_world"));
        out = enqueue([c, control, r, target, pole](UIContext &) {
            return response(rtapi::setRigIKTarget(c, control, target, pole, r));
        });
        return true;
    }
    if (method == "rig.set_ik_orientation") {
        if (!character(params) || !revision(params) || !params.contains("control") ||
            !params["control"].is_string() || !params.contains("enabled") ||
            !params["enabled"].is_boolean() || !params.contains("orientation_world"))
            return invalid();
        const auto &q = params.at("orientation_world");
        if (!q.is_array() || q.size() != 4)
            return invalid();
        for (const auto &v : q)
            if (!v.is_number())
                return invalid();
        const auto c = params.at("character").get<std::string>();
        const auto control = params.at("control").get<std::string>();
        const auto r = params.at("rig_revision").get<uint64_t>();
        const bool enabled = params.at("enabled").get<bool>();
        const Quaternion orientation(q[0].get<float>(), q[1].get<float>(), q[2].get<float>(),
                                     q[3].get<float>());
        out = enqueue([c, control, r, enabled, orientation](UIContext &) {
            return response(rtapi::setRigIKOrientation(c, control, orientation, enabled, r));
        });
        return true;
    }
    if (method == "rig.set_ik_fk") {
        if (!character(params) || !revision(params) || !params.contains("control") ||
            !params["control"].is_string() || !params.contains("blend") ||
            !params["blend"].is_number())
            return invalid();
        const auto c = params.at("character").get<std::string>(),
                   control = params.at("control").get<std::string>();
        const auto r = params.at("rig_revision").get<uint64_t>();
        const float blend = params.at("blend").get<float>();
        out = enqueue([c, control, r, blend](UIContext &) {
            return response(rtapi::setRigIKFK(c, control, blend, r));
        });
        return true;
    }
    if (method == "rig.match_ik_to_fk" || method == "rig.match_fk_to_ik") {
        if (!character(params) || !revision(params) || !params.contains("control") ||
            !params["control"].is_string()) {
            return invalid();
        }
        const auto c = params.at("character").get<std::string>();
        const auto control = params.at("control").get<std::string>();
        const auto r = params.at("rig_revision").get<uint64_t>();
        const bool toIK = method == "rig.match_ik_to_fk";
        out = enqueue([c, control, r, toIK](UIContext&) {
            return response(toIK ? rtapi::matchRigIKToFK(c, control, r)
                                 : rtapi::matchRigFKToIK(c, control, r));
        });
        return true;
    }
    if (method == "rig.set_ik_contact") {
        if (!character(params) || !revision(params) || !params.contains("control") ||
            !params["control"].is_string() || !params.contains("enabled") ||
            !params["enabled"].is_boolean())
            return invalid();
        const auto c = params.at("character").get<std::string>(),
                   control = params.at("control").get<std::string>();
        const auto r = params.at("rig_revision").get<uint64_t>();
        const bool enabled = params.at("enabled").get<bool>();
        out = enqueue([c, control, r, enabled](UIContext &) {
            return response(rtapi::setRigIKContact(c, control, enabled, r));
        });
        return true;
    }
    return false;
}
