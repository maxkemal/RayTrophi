#include "RtRigBindings.h"
#include "RtRigIKBindings.h"
#include "Api/RtApi.h"
#include "Api/RtApiRig.h"
using json = nlohmann::json;
namespace {
json boneJson(const RigAuthoring::BoneView& b) {
    json matrix = json::array();
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) matrix.push_back(b.world.m[r][c]);
    json placement=json::array();for(int r=0;r<4;++r)for(int c=0;c<4;++c)placement.push_back(b.scene_transform.m[r][c]);
    json rest=json::array();for(int r=0;r<4;++r)for(int c=0;c<4;++c)rest.push_back(b.local_rest.m[r][c]);
    return {{"scene_transform",placement},{"authoring_owned",b.authoring_owned},{"rig_revision",b.rig_revision},{"template_id",b.template_id},{"template_version",b.template_version},
            {"in_bonedata",b.in_bonedata},{"in_skeleton_nodes",b.in_skeleton_nodes},
            {"in_node_hierarchy",b.in_node_hierarchy},{"in_ozz_skeleton",b.in_ozz_skeleton},{"local_rest_transform",rest},{"character", b.character}, {"name", b.name}, {"parent", b.parent},
            {"bone_index", b.bone_index}, {"weighted", b.weighted}, {"pose_source", b.pose_source},
            {"world_transform", matrix}, {"world_position", {b.world.m[0][3], b.world.m[1][3], b.world.m[2][3]}}};
}
bool rigRestParam(const json& params,Matrix4x4& m,bool required) {
    m=Matrix4x4::identity();if(!params.contains("rest_transform"))return !required;
    const auto &value = params["rest_transform"];
    if (!value.is_array() || value.size() != 16)
        return false;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            if (!value[r * 4 + c].is_number())
                return false;
            m.m[r][c] = value[r * 4 + c].get<float>();
        }
    return true;
}
bool envelopeSettings(const json &params, RigAuthoring::EnvelopeWeightSettings &settings) {
    for (const auto *key : {"torso_radius", "limb_radius", "extremity_radius", "falloff"}) {
        if (params.contains(key) && !params[key].is_number())
            return false;
    }
    settings.torsoRadius = params.value("torso_radius", .16f);
    settings.limbRadius = params.value("limb_radius", .065f);
    settings.extremityRadius = params.value("extremity_radius", .05f);
    settings.falloff = params.value("falloff", 2.f);
    return true;
}
bool envelopeProfile(const json &params, RigAuthoring::EnvelopeBoneProfile &profile) {
    for (const auto *key : {"start_radius", "end_radius", "start_extension", "end_extension",
                            "falloff"}) {
        if (!params.contains(key) || !params[key].is_number())
            return false;
    }
    if (!params.contains("bone") || !params["bone"].is_string())
        return false;
    profile.bone = params["bone"].get<std::string>();
    profile.startRadius = params["start_radius"].get<float>();
    profile.endRadius = params["end_radius"].get<float>();
    profile.startExtension = params["start_extension"].get<float>();
    profile.endExtension = params["end_extension"].get<float>();
    profile.falloff = params["falloff"].get<float>();
    return true;
}
json failure(const rtapi::Result &r) {
    return {{"__error", r.error}, {"code", r.error}};
}
} // namespace
bool dispatchRigIpc(const std::string &method, const json &params,
                    const RtIpcTemplateEnqueue &enqueue, json &out) {
    if (dispatchRigIKIpc(method, params, enqueue, out))
        return true;
    if (method == "rig.preview_envelope_weights") {
        RigAuthoring::EnvelopeWeightSettings settings;
        if (!params.contains("character") || !params["character"].is_string() ||
            !envelopeSettings(params, settings)) {
            out = {{"__error", "Expected character and numeric envelope settings"},
                   {"code", "invalid_parameter"}};
            return true;
        }
        const auto character = params["character"].get<std::string>();
        out = enqueue([character, settings](UIContext &) {
            json value;
            auto r = rtapi::previewRigEnvelopeWeights(character, settings, value);
            return r.ok ? value : failure(r);
        });
        return true;
    }
    if (method == "rig.apply_envelope_weights") {
        RigAuthoring::EnvelopeWeightSettings settings;
        if (!params.contains("character") || !params["character"].is_string() ||
            !envelopeSettings(params, settings) || !params.contains("rig_revision") ||
            !params["rig_revision"].is_number_integer() ||
            (!params["rig_revision"].is_number_unsigned() &&
             params["rig_revision"].get<int64_t>() < 0)) {
            out = {{"__error", "Expected character, numeric envelope settings and nonnegative "
                               "rig_revision"},
                   {"code", "invalid_parameter"}};
            return true;
        }
        const auto character = params["character"].get<std::string>();
        const auto revision = params["rig_revision"].get<uint64_t>();
        out = enqueue([character, settings, revision](UIContext &) {
            auto r = rtapi::applyRigEnvelopeWeights(character, settings, revision);
            return r.ok ? json{{"ok", true}} : failure(r);
        });
        return true;
    }
    if (method == "rig.get_envelope_overlay") {
        out = enqueue([](UIContext &) {
            json value;
            auto r = rtapi::getRigEnvelopeOverlay(value);
            return r.ok ? value : failure(r);
        });
        return true;
    }
    if (method == "rig.set_envelope_overlay") {
        RigAuthoring::EnvelopeWeightSettings settings;
        if (!params.contains("character") || !params["character"].is_string() ||
            !params.contains("visible") || !params["visible"].is_boolean() ||
            !envelopeSettings(params, settings)) {
            out = {{"__error", "Expected character, visible and numeric envelope settings"},
                   {"code", "invalid_parameter"}};
            return true;
        }
        const auto character = params["character"].get<std::string>();
        const bool visible = params["visible"].get<bool>();
        out = enqueue([character, settings, visible](UIContext &) {
            auto r = rtapi::setRigEnvelopeOverlay(character, settings, visible);
            return r.ok ? json{{"ok", true}} : failure(r);
        });
        return true;
    }
    if (method == "rig.get_bone_envelope") {
        if (!params.contains("character") || !params["character"].is_string() ||
            !params.contains("bone") || !params["bone"].is_string()) {
            out = {{"__error", "Expected character and bone strings"},
                   {"code", "invalid_parameter"}};
            return true;
        }
        const auto character = params["character"].get<std::string>();
        const auto bone = params["bone"].get<std::string>();
        out = enqueue([character, bone](UIContext &) {
            json value;
            auto r = rtapi::getRigBoneEnvelope(character, bone, value);
            return r.ok ? value : failure(r);
        });
        return true;
    }
    if (method == "rig.apply_bone_envelope") {
        RigAuthoring::EnvelopeBoneProfile profile;
        if (!params.contains("character") || !params["character"].is_string() ||
            !envelopeProfile(params, profile) || !params.contains("rig_revision") ||
            !params["rig_revision"].is_number_integer() ||
            (!params["rig_revision"].is_number_unsigned() &&
             params["rig_revision"].get<int64_t>() < 0)) {
            out = {{"__error", "Expected character, bone, profile values and rig_revision"},
                   {"code", "invalid_parameter"}};
            return true;
        }
        const auto character = params["character"].get<std::string>();
        const auto revision = params["rig_revision"].get<uint64_t>();
        out = enqueue([character, profile, revision](UIContext &) {
            auto r = rtapi::applyRigBoneEnvelope(character, profile, revision);
            return r.ok ? json{{"ok", true}} : failure(r);
        });
        return true;
    }
    if (method == "rig.get_joint_limit_view") {
        if (!params.contains("character") || !params["character"].is_string() ||
            !params.contains("bone") || !params["bone"].is_string()) {
            out = {{"__error", "Expected character and bone strings"},
                   {"code", "invalid_parameter"}};
            return true;
        }
        const auto character = params.at("character").get<std::string>(),
                   bone = params.at("bone").get<std::string>();
        out = enqueue([character, bone](UIContext &) {
            json value;
            auto r = rtapi::getRigJointLimitView(character, bone, value);
            return r.ok ? value : failure(r);
        });
        return true;
    }
    if (method == "rig.set_joint_limits") {
        auto invalid = [&]() {
            out = {{"__error", "Expected character, bone, numeric minimum/maximum/swing and "
                               "nonnegative rig_revision"},
                   {"code", "invalid_parameter"}};
            return true;
        };
        if (!params.contains("character") || !params["character"].is_string() ||
            !params.contains("bone") || !params["bone"].is_string() ||
            !params.contains("rig_revision") || !params["rig_revision"].is_number_integer() ||
            (!params["rig_revision"].is_number_unsigned() &&
             params["rig_revision"].get<int64_t>() < 0))
            return invalid();
        if (!params.contains("minimum") || !params["minimum"].is_number() ||
            !params.contains("maximum") || !params["maximum"].is_number() ||
            !params.contains("swing") || !params["swing"].is_number())
            return invalid();
        const auto character = params.at("character").get<std::string>(),
                   bone = params.at("bone").get<std::string>();
        const auto revision = params.at("rig_revision").get<uint64_t>();
        const float minimum = params.at("minimum").get<float>(),
                    maximum = params.at("maximum").get<float>(),
                    swing = params.at("swing").get<float>();
        out = enqueue([character, bone, revision, minimum, maximum, swing](UIContext &) {
            auto r = rtapi::setRigJointLimits(character, bone, minimum, maximum, swing, revision);
            return r.ok ? json{{"ok", true}} : failure(r);
        });
        return true;
    }
    if (method == "rig.get_joint_limit_overlay") {
        out = enqueue([](UIContext &) {
            json value;
            auto r = rtapi::getRigJointLimitOverlay(value);
            return r.ok ? value : failure(r);
        });
        return true;
    }
    if (method == "rig.set_joint_limit_overlay") {
        if(!params.contains("visible")||!params["visible"].is_boolean()||!params.contains("edit")||!params["edit"].is_boolean()){out={{"__error","Expected visible and edit booleans"},{"code","invalid_parameter"}};return true;}
        const bool visible=params.at("visible").get<bool>(),edit=params.at("edit").get<bool>();out=enqueue([visible,edit](UIContext&){auto r=rtapi::setRigJointLimitOverlay(visible,edit);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.get_joint_profile") {
        if(!params.contains("character")||!params["character"].is_string()){out={{"__error","Expected character string"},{"code","invalid_parameter"}};return true;}
        const auto character=params.at("character").get<std::string>();out=enqueue([character](UIContext&){json value;auto r=rtapi::getRigJointProfile(character,value);return r.ok?value:failure(r);});return true;
    }
    if(method=="rig.suggest_joint_profile") {
        if(!params.contains("character")||!params["character"].is_string()){out={{"__error","Expected character string"},{"code","invalid_parameter"}};return true;}
        const auto character=params.at("character").get<std::string>();out=enqueue([character](UIContext&){json value;auto r=rtapi::suggestRigJointProfile(character,value);return r.ok?value:failure(r);});return true;
    }
    if(method=="rig.set_joint_profile") {
        if(!params.contains("character")||!params["character"].is_string()||!params.contains("profile")||!params["profile"].is_object()||!params.contains("rig_revision")||!params["rig_revision"].is_number_integer()||(!params["rig_revision"].is_number_unsigned()&&params["rig_revision"].get<int64_t>()<0)){out={{"__error","Expected character string, profile object and nonnegative rig_revision integer"},{"code","invalid_parameter"}};return true;}
        const auto character=params.at("character").get<std::string>();const auto profile=params.at("profile");const uint64_t revision=params.at("rig_revision").get<uint64_t>();out=enqueue([character,profile,revision](UIContext&){auto r=rtapi::setRigJointProfile(character,profile,revision);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.get_pose_state") {
        if(!params.contains("character")||!params["character"].is_string()){out={{"__error","Expected character string"},{"code","invalid_parameter"}};return true;}
        const auto character=params.at("character").get<std::string>();
        out=enqueue([character](UIContext&){json value;auto r=rtapi::getRigPoseState(character,value);return r.ok?value:failure(r);});return true;
    }
    if(method=="rig.get_pose_coverage") {
        if(!params.contains("character")||!params["character"].is_string()){out={{"__error","Expected character string"},{"code","invalid_parameter"}};return true;}
        const auto character=params.at("character").get<std::string>();
        out=enqueue([character](UIContext&){json value;auto r=rtapi::getRigPoseCoverage(character,value);return r.ok?value:failure(r);});return true;
    }
    if(method=="rig.apply_pose_preview") {
        if(!params.contains("character")||!params["character"].is_string()){out={{"__error","Expected character string"},{"code","invalid_parameter"}};return true;}
        const auto character=params.at("character").get<std::string>();
        out=enqueue([character](UIContext&){auto r=rtapi::applyRigPosePreview(character);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.cancel_pose_preview") {
        if(!params.contains("character")||!params["character"].is_string()){out={{"__error","Expected character string"},{"code","invalid_parameter"}};return true;}
        const auto character=params.at("character").get<std::string>();
        out=enqueue([character](UIContext&){auto r=rtapi::cancelRigPosePreview(character);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.create_pose_clip") {
        if(!params.contains("character")||!params["character"].is_string()||!params.contains("name")||!params["name"].is_string() || (params.contains("fps")&&!params["fps"].is_number())){out={{"__error","Invalid pose clip parameter types"},{"code","invalid_parameter"}};return true;}
        const auto character=params.at("character").get<std::string>(),value=params.at("name").get<std::string>();const float fps=params.value("fps",24.f);
        out=enqueue([character,value,fps](UIContext&){auto r=rtapi::createRigPoseClip(character,value,fps);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.select_pose_clip") {
        if(!params.contains("character")||!params["character"].is_string()||!params.contains("clip")||!params["clip"].is_string()){out={{"__error","Invalid pose clip parameter types"},{"code","invalid_parameter"}};return true;}
        const auto character=params.at("character").get<std::string>(),value=params.at("clip").get<std::string>();
        out=enqueue([character,value](UIContext&){auto r=rtapi::selectRigPoseClip(character,value);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.set_pose_auto_key") {
        if(!params.contains("enabled")||!params["enabled"].is_boolean()){out={{"__error","Expected enabled boolean"},{"code","invalid_parameter"}};return true;}
        const bool enabled=params.at("enabled").get<bool>();out=enqueue([enabled](UIContext&){auto r=rtapi::setRigPoseAutoKey(enabled);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.set_pose_frame") {
        if(!params.contains("frame")||!params["frame"].is_number_integer()||params["frame"]<0||params["frame"]>1000000){out={{"__error","Expected integer frame in 0..1000000"},{"code","invalid_parameter"}};return true;}
        const int frame=params.at("frame").get<int>();out=enqueue([frame](UIContext&){auto r=rtapi::setRigPoseFrame(frame);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.mirror_pose") {
        auto invalid=[&](){out={{"__error","Expected character, bones, nonnegative rig_revision and optional direction/axis strings"},{"code","invalid_parameter"}};return true;};
        if(!params.contains("character")||!params["character"].is_string()||!params.contains("bones")||!params["bones"].is_array()||!params.contains("rig_revision")||!params["rig_revision"].is_number_integer()||(!params["rig_revision"].is_number_unsigned()&&params["rig_revision"].get<int64_t>()<0))return invalid();
        if(params["bones"].size()>4096)return invalid();for(const auto& b:params["bones"])if(!b.is_string())return invalid();
        for(const auto* key:{"direction","axis"})if(params.contains(key)&&!params[key].is_string())return invalid();
        const auto character=params["character"].get<std::string>();
        const auto direction=params.value("direction",std::string("selected"));
        const auto axis=params.value("axis",std::string("x"));
        const auto bones=params["bones"].get<std::vector<std::string>>();const auto revision=params["rig_revision"].get<uint64_t>();
        out=enqueue([character,bones,revision,direction,axis](UIContext&){auto r=rtapi::mirrorRigPose(character,bones,revision,direction,axis);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.insert_pose_keys"||method=="rig.remove_pose_keys") {
        auto invalid=[&](){out={{"__error","Invalid pose parameter types"},{"code","invalid_parameter"}};return true;};
        if(!params.contains("character")||!params["character"].is_string())return invalid();
        const auto character=params.at("character").get<std::string>();
        if(!params.contains("bones")||!params["bones"].is_array())return invalid();for(const auto& bone:params["bones"])if(!bone.is_string())return invalid();
        const auto bones=params.at("bones").get<std::vector<std::string>>();
        const bool remove=method=="rig.remove_pose_keys";
        out=enqueue([character,bones,remove](UIContext&){auto r=remove?rtapi::removeRigPoseKeys(character,bones):rtapi::insertRigPoseKeys(character,bones);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if (method == "rig.edit_pose_key") {
        auto invalid = [&]() {
            out = {{"__error", "Invalid rig curve parameter types"},
                   {"code", "invalid_parameter"}};
            return true;
        };
        for (const auto* key : {"character", "bone", "channel"}) {
            if (!params.contains(key) || !params[key].is_string())
                return invalid();
        }
        for (const auto* key : {"source_frame", "target_frame"}) {
            if (!params.contains(key) || !params[key].is_number_integer() ||
                params[key].get<int64_t>() < 0 ||
                params[key].get<int64_t>() > 1000000)
                return invalid();
        }
        if (params.contains("value") && !params["value"].is_null() &&
            !params["value"].is_array())
            return invalid();
        const auto character = params["character"].get<std::string>();
        const auto bone = params["bone"].get<std::string>();
        const auto channel = params["channel"].get<std::string>();
        const int source = params["source_frame"].get<int>();
        const int target = params["target_frame"].get<int>();
        const auto value = params.contains("value") ? params["value"] : json(nullptr);
        out = enqueue([character, bone, channel, source, target, value](UIContext&) {
            auto result = rtapi::editRigPoseKey(
                character, bone, channel, source, target, value);
            return result.ok ? json{{"ok", true}} : failure(result);
        });
        return true;
    }
    if (method == "rig.get_driven_controls") {
        if (!params.contains("character") || !params["character"].is_string()) {
            out = {{"__error", "Expected character string"},
                   {"code", "invalid_parameter"}};
            return true;
        }
        const auto character = params["character"].get<std::string>();
        out = enqueue([character](UIContext&) {
            json value;
            auto result = rtapi::getRigDrivenControls(character, value);
            return result.ok ? value : failure(result);
        });
        return true;
    }
    if(method=="rig.preview_pose_transform") {
        auto invalid=[&](){out={{"__error","Invalid pose parameter types"},{"code","invalid_parameter"}};return true;};
        if(!params.contains("character")||!params["character"].is_string())return invalid();
        const auto character=params.at("character").get<std::string>();
        if(!params.contains("rig_revision")||!params["rig_revision"].is_number_integer()||(!params["rig_revision"].is_number_unsigned()&&params["rig_revision"].get<int64_t>()<0))return invalid();
        const uint64_t revision=params.at("rig_revision").get<uint64_t>();
        if(!params.contains("bones")||!params["bones"].is_array())return invalid();for(const auto& bone:params["bones"])if(!bone.is_string())return invalid();
        const auto bones=params.at("bones").get<std::vector<std::string>>();
        if(!params.contains("world_delta")||!params["world_delta"].is_array()||params["world_delta"].size()!=16)return invalid();Matrix4x4 delta;
        for(int r=0;r<4;++r)for(int c=0;c<4;++c){if(!params["world_delta"][r*4+c].is_number())return invalid();delta.m[r][c]=params["world_delta"][r*4+c].get<float>();}
        out=enqueue([character,bones,delta,revision](UIContext&){auto r=rtapi::previewRigPoseTransform(character,bones,delta,revision);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.preview_pose_locals") {
        auto invalid=[&](){out={{"__error","Invalid pose parameter types"},{"code","invalid_parameter"}};return true;};
        if(!params.contains("character")||!params["character"].is_string())return invalid();
        const auto character=params.at("character").get<std::string>();
        if(!params.contains("rig_revision")||!params["rig_revision"].is_number_integer()||(!params["rig_revision"].is_number_unsigned()&&params["rig_revision"].get<int64_t>()<0))return invalid();
        const uint64_t revision=params.at("rig_revision").get<uint64_t>();
        if(!params.contains("local_transforms")||!params["local_transforms"].is_object())return invalid();const auto values=params.at("local_transforms");
        out=enqueue([character,values,revision](UIContext&){auto r=rtapi::previewRigPoseLocals(character,values,revision);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if (method == "rig.preview_control_values") {
        auto invalid = [&]() {
            out = {{"__error", "Expected character, numeric values object and rig_revision"},
                   {"code", "invalid_parameter"}};
            return true;
        };
        if (!params.contains("character") || !params["character"].is_string() ||
            !params.contains("values") || !params["values"].is_object() ||
            params["values"].empty() || params["values"].size() > 4096 ||
            !params.contains("rig_revision") ||
            !params["rig_revision"].is_number_integer() ||
            (!params["rig_revision"].is_number_unsigned() &&
             params["rig_revision"].get<int64_t>() < 0)) {
            return invalid();
        }
        for (const auto& value : params["values"].items())
            if (!value.value().is_number())
                return invalid();
        const auto character = params["character"].get<std::string>();
        const auto values = params["values"];
        const uint64_t revision = params["rig_revision"].get<uint64_t>();
        out = enqueue([character, values, revision](UIContext&) {
            auto result = rtapi::previewRigControlValues(character, values, revision);
            return result.ok ? json{{"ok", true}} : failure(result);
        });
        return true;
    }
    if(method=="rig.mirror_rest" || method=="rig.create_mirrored_bone" || method=="rig.get_mirrored_landmarks") {
        auto invalid=[&](){out={{"__error","Expected mirror character, nonnegative rig_revision and correctly typed mirror parameters"},{"code","invalid_parameter"}};return true;};
        if(!params.contains("character") || !params["character"].is_string() || !params.contains("rig_revision") || !params["rig_revision"].is_number_integer() ||
           (!params["rig_revision"].is_number_unsigned() && params["rig_revision"].get<int64_t>()<0))return invalid();
        for(const auto* key:{"axis","direction"})if(params.contains(key) && !params[key].is_string())return invalid();
        if(params.contains("offset") && !params["offset"].is_number())return invalid();
        const std::string character=params["character"].get<std::string>(),axis=params.value("axis",std::string("x")),direction=params.value("direction",std::string("selected"));
        const float offset=params.value("offset",0.f);const uint64_t revision=params["rig_revision"].get<uint64_t>();
        if(method=="rig.create_mirrored_bone") {
            for(const auto* key:{"bone","name","source_side"})if(!params.contains(key) || !params[key].is_string())return invalid();
            const std::string bone=params["bone"].get<std::string>(),name=params["name"].get<std::string>(),side=params["source_side"].get<std::string>();
            out=enqueue([character,bone,name,side,axis,offset,revision](UIContext&){auto r=rtapi::createMirroredRigBone(character,bone,name,side,axis,offset,revision);return r.ok?json{{"ok",true}}:failure(r);});return true;
        }
        if(!params.contains("bones") || !params["bones"].is_array())return invalid();
        for(const auto& bone:params["bones"])if(!bone.is_string())return invalid();
        const auto bones=params["bones"].get<std::vector<std::string>>();
        if(method=="rig.get_mirrored_landmarks") {
            if(!params.contains("landmarks") || !params["landmarks"].is_object())return invalid();const auto marks=params["landmarks"];
            out=enqueue([character,bones,marks,direction,axis,offset,revision](UIContext&){json output;auto r=rtapi::mirrorRigLandmarks(character,marks,bones,direction,axis,offset,revision,output);return r.ok?output:failure(r);});return true;
        }
        out=enqueue([character,bones,direction,axis,offset,revision](UIContext&){auto r=rtapi::mirrorRigRest(character,bones,direction,axis,offset,revision);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.get_selection") {out=enqueue([](UIContext&){json value;auto r=rtapi::getRigSelection(value);return r.ok?value:failure(r);});return true;}
    if(method=="rig.set_selection_pivot") {
        if(!params.contains("mode") || !params["mode"].is_string()){out={{"__error","Expected pivot mode string"},{"code","invalid_parameter"}};return true;}
        const std::string mode=params.at("mode").get<std::string>();out=enqueue([mode](UIContext&){auto r=rtapi::setRigSelectionPivot(mode);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.select_bones") {
        if(!params.contains("character") || !params["character"].is_string() || !params.contains("bones") || !params["bones"].is_array() ||
           (params.contains("active") && !params["active"].is_string()) || (params.contains("anchor") && !params["anchor"].is_string()) || (params.contains("mode") && !params["mode"].is_string())){out={{"__error","Expected character, bone array and optional active/mode/anchor strings"},{"code","invalid_parameter"}};return true;}
        for(const auto& b:params["bones"])if(!b.is_string()){out={{"__error","Expected bone strings"},{"code","invalid_parameter"}};return true;}
        const std::string character=params.at("character").get<std::string>();const auto bones=params.at("bones").get<std::vector<std::string>>();
        const std::string active=params.value("active",std::string("")),mode=params.value("mode",std::string("replace")),anchor=params.value("anchor",std::string(""));
        out=enqueue([character,bones,active,mode,anchor](UIContext&){auto r=rtapi::selectRigBones(character,bones,active,mode,anchor);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.transform_rest") {
        if(!params.contains("character") || !params["character"].is_string() || !params.contains("bones") || !params["bones"].is_array() ||
           !params.contains("world_delta") || !params["world_delta"].is_array() || params["world_delta"].size()!=16 ||
           !params.contains("rig_revision") || !params["rig_revision"].is_number_integer() ||
           (!params["rig_revision"].is_number_unsigned() && params["rig_revision"].get<int64_t>()<0)){out={{"__error","Expected character, bone strings, row-major world_delta (16 numbers) and nonnegative rig_revision"},{"code","invalid_parameter"}};return true;}
        for(const auto& b:params["bones"])if(!b.is_string()){out={{"__error","Expected bone strings"},{"code","invalid_parameter"}};return true;}
        Matrix4x4 delta;for(int r=0;r<4;++r)for(int c=0;c<4;++c){if(!params["world_delta"][r*4+c].is_number()){out={{"__error","Expected numeric world_delta"},{"code","invalid_parameter"}};return true;}delta.m[r][c]=params["world_delta"][r*4+c].get<float>();}
        const std::string character=params.at("character").get<std::string>();const auto bones=params.at("bones").get<std::vector<std::string>>();const uint64_t revision=params.at("rig_revision").get<uint64_t>();
        out=enqueue([character,bones,delta,revision](UIContext&){auto r=rtapi::transformRigRest(character,bones,delta,revision);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.get_binding") {
        if(!params.contains("character") || !params["character"].is_string()){out={{"__error","Expected character string"},{"code","invalid_parameter"}};return true;}
        const std::string character=params.at("character").get<std::string>();
        out=enqueue([character](UIContext&){json value;auto r=rtapi::getRigMeshBinding(character,value);return r.ok?value:failure(r);});return true;
    }
    if(method=="rig.preview_bind") {
        if(!params.contains("character") || !params["character"].is_string() || !params.contains("mesh") || !params["mesh"].is_string() || (params.contains("axes_confirmed") && !params["axes_confirmed"].is_boolean())){out={{"__error","Expected character/mesh strings and axes_confirmed boolean"},{"code","invalid_parameter"}};return true;}
        const std::string character=params.at("character").get<std::string>();const std::string mesh=params.at("mesh").get<std::string>();const bool confirmed=params.value("axes_confirmed",false);
        out=enqueue([character,mesh,confirmed](UIContext&){json value;auto r=rtapi::previewRigMeshBinding(character,mesh,confirmed,value);return r.ok?value:failure(r);});return true;
    }
    if(method=="rig.bind_mesh") {
        if(!params.contains("character") || !params["character"].is_string() || !params.contains("mesh") || !params["mesh"].is_string() || !params.contains("preview") || !params["preview"].is_object()){out={{"__error","Expected character/mesh strings and preview object"},{"code","invalid_parameter"}};return true;}
        const std::string character=params.at("character").get<std::string>();const std::string mesh=params.at("mesh").get<std::string>();const json preview=params.at("preview");
        out=enqueue([character,mesh,preview](UIContext&){auto r=rtapi::bindRigMesh(character,mesh,preview);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.unbind_mesh") {
        if(!params.contains("character") || !params["character"].is_string()){
            out={{"__error","Expected character string"},{"code","invalid_parameter"}};
            return true;
        }
        const std::string character=params.at("character").get<std::string>();
        out=enqueue([character](UIContext&){
            auto r=rtapi::unbindRigMesh(character);
            return r.ok?json{{"ok",true}}:failure(r);
        });
        return true;
    }
    if(method=="rig.set_weight_map_visible") {
        if(!params.contains("visible") || !params["visible"].is_boolean()){out={{"__error","Expected visible boolean"},{"code","invalid_parameter"}};return true;}
        const bool visible=params.at("visible").get<bool>();out=enqueue([visible](UIContext&){auto r=rtapi::setRigWeightMapVisible(visible);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.get_weight_map_visible") {out=enqueue([](UIContext&){bool visible;auto r=rtapi::getRigWeightMapVisible(visible);return r.ok?json(visible):failure(r);});return true;}
    if(method=="rig.get_weight_map") {
        if(!params.contains("mesh") || !params["mesh"].is_string() || !params.contains("character") || !params["character"].is_string() || !params.contains("bone") || !params["bone"].is_string()){out={{"__error","Expected exact mesh, character and bone strings"},{"code","invalid_parameter"}};return true;}
        const std::string mesh=params.at("mesh").get<std::string>();const std::string character=params.at("character").get<std::string>();const std::string bone=params.at("bone").get<std::string>();
        out=enqueue([mesh,character,bone](UIContext&){json report;auto r=rtapi::getRigWeightMap(mesh,character,bone,report);return r.ok?report:failure(r);});return true;
    }
    if(method=="rig.get_weights") {
        if(!params.contains("object") || !params["object"].is_string() || !params.contains("vertex") ||
           !params["vertex"].is_number_integer() ||
           (!params["vertex"].is_number_unsigned() && params["vertex"].get<int64_t>()<0)) {
            out={{"__error","Expected exact object string and nonnegative integer vertex"},{"code","invalid_parameter"}};return true;
        }
        const std::string object=params.at("object").get<std::string>();
        const uint64_t vertex=params.at("vertex").get<uint64_t>();
        out=enqueue([object,vertex](UIContext&){json report;auto r=rtapi::getRigVertexWeights(object,vertex,report);return r.ok?report:failure(r);});return true;
    }
    if(method=="rig.weight_stats") {
        if(!params.contains("mesh") || !params["mesh"].is_string()){out={{"__error","Expected exact mesh string"},{"code","invalid_parameter"}};return true;}
        const std::string mesh=params.at("mesh").get<std::string>();
        out=enqueue([mesh](UIContext&){json report;auto r=rtapi::getRigWeightStats(mesh,report);return r.ok?report:failure(r);});return true;
    }
    if(method=="rig.set_mode") {
        if(!params.contains("mode") || !params["mode"].is_string() ||
           (params.contains("character") && !params["character"].is_string())) {
            out={{"__error","mode and character must be strings"},{"code","invalid_parameter"}};return true;
        }
        const std::string mode=params.at("mode").get<std::string>();
        const std::string character=params.value("character",std::string(""));
        out=enqueue([mode,character](UIContext&){auto r=rtapi::setRigMode(mode,character);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.get_mode") {
        out=enqueue([](UIContext&){std::string mode,character;auto r=rtapi::getRigMode(mode,character);return r.ok?json{{"mode",mode},{"character",character}}:failure(r);});return true;
    }
    if(method=="rig.get_next_name") {
        if(params.contains("seed") && !params["seed"].is_string()){out={{"__error","Expected seed string"},{"code","invalid_parameter"}};return true;}
        const std::string seed=params.value("seed",std::string("Rig"));
        out=enqueue([seed](UIContext&){std::string name;auto r=rtapi::getNextRigName(seed,name);return r.ok?json(name):failure(r);});return true;
    }
    if(method=="rig.get_scene_transform") {
        if(!params.contains("character") || !params["character"].is_string()){out={{"__error","Expected character string"},{"code","invalid_parameter"}};return true;}
        const std::string character=params.at("character").get<std::string>();
        out=enqueue([character](UIContext&){Matrix4x4 matrix;auto r=rtapi::getRigSceneTransform(character,matrix);if(!r.ok)return failure(r);json value=json::array();for(int row=0;row<4;++row)for(int col=0;col<4;++col)value.push_back(matrix.m[row][col]);return value;});return true;
    }
    if(method=="rig.set_scene_transform") {
        if(!params.contains("character") || !params["character"].is_string() || !params.contains("scene_transform") ||
           !params["scene_transform"].is_array() || params["scene_transform"].size()!=16) {
            out={{"__error","Expected character and row-major scene_transform of 16 numbers"},{"code","invalid_parameter"}};return true;
        }
        Matrix4x4 matrix;const auto& values=params["scene_transform"];
        for(int r=0;r<4;++r)for(int c=0;c<4;++c){if(!values[r*4+c].is_number()){out={{"__error","Expected numeric scene_transform"},{"code","invalid_parameter"}};return true;}matrix.m[r][c]=values[r*4+c].get<float>();}
        const std::string character=params.at("character").get<std::string>();
        out=enqueue([character,matrix](UIContext&){auto r=rtapi::setRigSceneTransform(character,matrix);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.get_next_bone_name") {
        if(!params.contains("character") || !params["character"].is_string() ||
           (params.contains("seed") && !params["seed"].is_string())) {
            out={{"__error","character and seed must be strings"},{"code","invalid_parameter"}};return true;
        }
        const std::string character=params.at("character").get<std::string>();
        const std::string seed=params.value("seed",std::string("Joint1"));
        out=enqueue([character,seed](UIContext&){std::string name;auto r=rtapi::getNextRigBoneName(character,seed,name);return r.ok?json(name):failure(r);});return true;
    }
    if(method=="rig.rename_bone") {
        if(!params.contains("character") || !params["character"].is_string() ||
           !params.contains("bone") || !params["bone"].is_string() || !params.contains("name") || !params["name"].is_string()) {
            out={{"__error","Expected string rig topology parameters"},{"code","invalid_parameter"}};return true;
        }
        const std::string character=params.at("character").get<std::string>();
        const std::string bone=params.at("bone").get<std::string>();
        const std::string name=params.at("name").get<std::string>();
        out=enqueue([character,bone,name](UIContext&){auto r=rtapi::renameRigBone(character,bone,name);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.reparent_bone") {
        if(!params.contains("character") || !params["character"].is_string() ||
           !params.contains("bone") || !params["bone"].is_string() || !params.contains("parent") || !params["parent"].is_string()) {
            out={{"__error","Expected string rig topology parameters"},{"code","invalid_parameter"}};return true;
        }
        const std::string character=params.at("character").get<std::string>();
        const std::string bone=params.at("bone").get<std::string>();
        const std::string parent=params.at("parent").get<std::string>();
        out=enqueue([character,bone,parent](UIContext&){auto r=rtapi::reparentRigBone(character,bone,parent);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.delete_bone") {
        if(!params.contains("character") || !params["character"].is_string() ||
           !params.contains("bone") || !params["bone"].is_string()) {
            out={{"__error","Expected string rig topology parameters"},{"code","invalid_parameter"}};return true;
        }
        const std::string character=params.at("character").get<std::string>();
        const std::string bone=params.at("bone").get<std::string>();
        out=enqueue([character,bone](UIContext&){auto r=rtapi::deleteRigBone(character,bone);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.copy_from") {
        if(!params.contains("source_character") || !params["source_character"].is_string() ||
           !params.contains("character") || !params["character"].is_string()) {
            out={{"__error","source_character and character must be strings"},{"code","invalid_parameter"}};return true;
        }
        const std::string source=params.at("source_character").get<std::string>();
        const std::string character=params.at("character").get<std::string>();
        out=enqueue([source,character](UIContext&){
            std::vector<RigAuthoring::RigCopyBone> mapping;auto r=rtapi::copyRigFrom(source,character,mapping);
            if(!r.ok)return failure(r);
            json bones=json::array();for(const auto& bone:mapping)bones.push_back({{"source_bone",bone.source_bone},{"target_bone",bone.target_bone}});
            return json{{"character",character},{"bone_map",bones}};
        });return true;
    }
    if(method=="rig.get_anatomy") {
        if(!params.contains("character") || !params["character"].is_string()) {
            out={{"__error","character must be a string"},{"code","invalid_parameter"}};return true;
        }
        const std::string character=params.at("character").get<std::string>();
        out=enqueue([character](UIContext&){json value;auto r=rtapi::getRigAnatomy(character,value);return r.ok?value:failure(r);});return true;
    }
    if(method=="rig.set_anatomy") {
        if(!params.contains("character") || !params["character"].is_string() || !params.contains("anatomy") || !params["anatomy"].is_object()) {
            out={{"__error","Expected character string and anatomy object"},{"code","invalid_parameter"}};return true;
        }
        const std::string character=params.at("character").get<std::string>();const json anatomy=params.at("anatomy");
        out=enqueue([character,anatomy](UIContext&){auto r=rtapi::setRigAnatomy(character,anatomy);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.set_pose_view") {
        if(!params.contains("character") || !params["character"].is_string() || !params.contains("mode") || !params["mode"].is_string()) {
            out={{"__error","Expected character and mode strings"},{"code","invalid_parameter"}};return true;
        }
        const std::string character=params.at("character").get<std::string>();const std::string mode=params.at("mode").get<std::string>();
        out=enqueue([character,mode](UIContext&){auto r=rtapi::setRigPoseView(character,mode);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.get_pose_view") {
        if(!params.contains("character") || !params["character"].is_string()) {
            out={{"__error","character must be a string"},{"code","invalid_parameter"}};return true;
        }
        const std::string character=params.at("character").get<std::string>();
        out=enqueue([character](UIContext&){std::string mode,effective;auto r=rtapi::getRigPoseView(character,mode,effective);return r.ok?json{{"mode",mode},{"effective_mode",effective}}:failure(r);});return true;
    }
    if(method=="rig.list_fit_targets"){out=enqueue([](UIContext&){json value;auto r=rtapi::listRigFitTargets(value);return r.ok?value:failure(r);});return true;}
    if(method=="rig.get_fit_setup") {
        if(!params.contains("character")||!params["character"].is_string()||!params.contains("mesh")||!params["mesh"].is_string()) {out={{"__error","Expected character and mesh strings"},{"code","invalid_parameter"}};return true;}
        const std::string character=params.at("character").get<std::string>();const std::string mesh=params.at("mesh").get<std::string>();
        out=enqueue([character,mesh](UIContext&){json value;auto r=rtapi::getRigFitSetup(character,mesh,value);return r.ok?value:failure(r);});return true;
    }
    if(method=="rig.preview_fit") {
        if(!params.contains("character")||!params["character"].is_string()||!params.contains("mesh")||!params["mesh"].is_string()) {out={{"__error","Expected character and mesh strings"},{"code","invalid_parameter"}};return true;}
        const std::string character=params.at("character").get<std::string>();const std::string mesh=params.at("mesh").get<std::string>();
        if(!params.contains("landmarks")||!params["landmarks"].is_object()||(params.contains("axes_confirmed")&&!params["axes_confirmed"].is_boolean())){out={{"__error","Expected landmarks object and axes_confirmed boolean"},{"code","invalid_parameter"}};return true;}
        const json landmarks=params["landmarks"];const bool confirmed=params.value("axes_confirmed",false);
        out=enqueue([character,mesh,landmarks,confirmed](UIContext&){json value;auto r=rtapi::previewRigFit(character,mesh,landmarks,confirmed,value);return r.ok?value:failure(r);});return true;
    }
    if(method=="rig.commit_fit") {
        if(!params.contains("character")||!params["character"].is_string()||!params.contains("mesh")||!params["mesh"].is_string()) {out={{"__error","Expected character and mesh strings"},{"code","invalid_parameter"}};return true;}
        const std::string character=params.at("character").get<std::string>();const std::string mesh=params.at("mesh").get<std::string>();
        if(!params.contains("preview")||!params["preview"].is_object()){out={{"__error","Expected preview object"},{"code","invalid_parameter"}};return true;}
        const json preview=params["preview"];out=enqueue([character,mesh,preview](UIContext&){auto r=rtapi::commitRigFit(character,mesh,preview);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.preflight") {
        if(!params.contains("mesh") || !params["mesh"].is_string()){out={{"__error","Expected mesh name string"},{"code","invalid_parameter"}};return true;}
        const std::string mesh=params.at("mesh").get<std::string>();
        out=enqueue([mesh](UIContext&){json report;auto r=rtapi::preflightRigMesh(mesh,report);return r.ok?report:failure(r);});return true;
    }
    if(method=="rig.list_templates") {
        out=enqueue([](UIContext&){json value;auto r=rtapi::listRigTemplates(value);return r.ok?value:failure(r);});return true;
    }
    if(method=="rig.get_template") {
        if(!params.contains("template_id") || !params["template_id"].is_string() || (params.contains("height") && !params["height"].is_number())) {
            out={{"__error","Expected template_id string and numeric height"},{"code","invalid_parameter"}};return true;
        }
        const std::string id=params.at("template_id").get<std::string>();const float height=params.value("height",1.8f);
        out=enqueue([id,height](UIContext&){json value;auto r=rtapi::getRigTemplate(id,height,value);return r.ok?value:failure(r);});return true;
    }
    if(method=="rig.create") {
        if(!params.contains("character")||!params["character"].is_string() ||
           (params.contains("template_id")&&!params["template_id"].is_string()) || (params.contains("height")&&!params["height"].is_number())) {
            out={{"__error","Invalid rig creation parameter types"},{"code","invalid_parameter"}};return true;
        }
        const std::string character=params.at("character").get<std::string>();
        const std::string template_id=params.value("template_id",std::string("root"));
        const float height=params.value("height",1.8f);
        out=enqueue([character,template_id,height](UIContext&){auto r=rtapi::createRig(character,template_id,height);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.add_bone") {
        Matrix4x4 rest;
        if(!params.contains("character")||!params["character"].is_string()||!params.contains("name")||!params["name"].is_string()||
           !params.contains("parent")||!params["parent"].is_string()||!rigRestParam(params,rest,false)) {
            out={{"__error","Expected character/name/parent strings and a row-major rest_transform array of 16 numbers"},{"code","invalid_parameter"}};return true;
        }
        const std::string character=params.at("character").get<std::string>();const std::string name=params.at("name").get<std::string>();
        const std::string parent=params.at("parent").get<std::string>();
        out=enqueue([character,name,parent,rest](UIContext&){auto r=rtapi::addRigBone(character,name,parent,rest);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if(method=="rig.set_rest_transform") {
        Matrix4x4 rest;
        if(!params.contains("character")||!params["character"].is_string()||!params.contains("bone")||!params["bone"].is_string()||!rigRestParam(params,rest,true)) {
            out={{"__error","Expected character/bone strings and a row-major rest_transform array of 16 numbers"},{"code","invalid_parameter"}};return true;
        }
        const std::string character=params.at("character").get<std::string>();const std::string bone=params.at("bone").get<std::string>();
        out=enqueue([character,bone,rest](UIContext&){auto r=rtapi::setRigRestTransform(character,bone,rest);return r.ok?json{{"ok",true}}:failure(r);});return true;
    }
    if (method == "rig.list_characters") {
        out = enqueue([](UIContext&) { std::vector<std::string> names; auto r = rtapi::listRigCharacters(names); return r.ok ? json(names) : failure(r); }); return true;
    }
    if (method == "rig.list_bones") {
        if (!params.contains("character") || !params["character"].is_string()) {
            out = {{"__error", "character must be a string"}, {"code", "invalid_parameter"}}; return true;
        }
        const std::string character = params.at("character").get<std::string>();
        out = enqueue([character](UIContext&) {
            std::vector<RigAuthoring::BoneView> bones; auto r = rtapi::listRigBones(character, bones);
            if (!r.ok) return failure(r);
            json result = json::array(); for (const auto& b : bones) result.push_back(boneJson(b)); return result;
        }); return true;
    }
    if (method == "rig.select_bone") {
        if (!params.contains("character") || !params["character"].is_string() ||
            !params.contains("bone") || !params["bone"].is_string()) {
            out = {{"__error", "character and bone must be strings"}, {"code", "invalid_parameter"}}; return true;
        }
        const std::string character = params.at("character").get<std::string>();
        const std::string bone = params.at("bone").get<std::string>();
        out = enqueue([character, bone](UIContext&) { auto r = rtapi::selectRigBone(character, bone); return r.ok ? json{{"ok", true}} : failure(r); }); return true;
    }
    if (method == "rig.get_selected_bone") {
        out = enqueue([](UIContext&) { RigAuthoring::BoneView b; bool selected; auto r = rtapi::getSelectedRigBone(b, selected); return r.ok ? (selected ? boneJson(b) : json(nullptr)) : failure(r); }); return true;
    }
    if (method == "rig.clear_selection") {
        out = enqueue([](UIContext&) { auto r = rtapi::clearRigSelection(); return r.ok ? json{{"ok", true}} : failure(r); }); return true;
    }
    if (method == "rig.get_overlay_visible") {
        out = enqueue([](UIContext&) { bool visible; auto r = rtapi::getRigOverlayVisible(visible); return r.ok ? json(visible) : failure(r); }); return true;
    }
    if (method == "rig.set_overlay_visible") {
        if (!params.contains("visible") || !params["visible"].is_boolean()) { out = {{"__error", "visible must be boolean"}, {"code", "invalid_parameter"}}; return true; }
        const bool visible = params.at("visible").get<bool>();
        out = enqueue([visible](UIContext&) { auto r = rtapi::setRigOverlayVisible(visible); return r.ok ? json{{"ok", true}} : failure(r); }); return true;
    }
    return false;
}
