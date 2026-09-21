#include "RtRigBindings.h"
#include "RtRigIKBindings.h"
#include "Api/RtApi.h"
#include "Api/RtApiRig.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include "json.hpp"
namespace py = pybind11;
namespace {
void require(const rtapi::Result& r) { if (!r.ok) throw std::invalid_argument(r.error); }
py::dict boneDict(const RigAuthoring::BoneView& b) {
    py::dict d; d["character"] = b.character; d["name"] = b.name; d["parent"] = b.parent;
    d["authoring_owned"]=b.authoring_owned;d["rig_revision"]=b.rig_revision;d["template_id"]=b.template_id;d["template_version"]=b.template_version;
    d["in_bonedata"]=b.in_bonedata;d["in_skeleton_nodes"]=b.in_skeleton_nodes;
    d["in_node_hierarchy"]=b.in_node_hierarchy;d["in_ozz_skeleton"]=b.in_ozz_skeleton;
    py::list placement;for(int r=0;r<4;++r)for(int c=0;c<4;++c)placement.append(b.scene_transform.m[r][c]);d["scene_transform"]=placement;
    py::list rest;for(int r=0;r<4;++r)for(int c=0;c<4;++c)rest.append(b.local_rest.m[r][c]);d["local_rest_transform"]=rest;
    d["bone_index"] = b.bone_index; d["weighted"] = b.weighted; d["pose_source"] = b.pose_source;
    py::list matrix, position;
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) matrix.append(b.world.m[r][c]);
    for (int r = 0; r < 3; ++r) position.append(b.world.m[r][3]);
    d["world_transform"] = matrix; d["world_position"] = position; return d;
}
}
Matrix4x4 rigRestMatrix(const std::vector<float>& values) {
    if(values.size()!=16)throw std::invalid_argument("invalid_rest_transform");
    Matrix4x4 m;for(int r=0;r<4;++r)for(int c=0;c<4;++c)m.m[r][c]=values[r*4+c];return m;
}
void registerRigPython(py::module_& module) {
    auto rig = module.def_submodule("rig", "Shared skeleton view and bone selection; works without mesh/skin.");
    registerRigIKPython(rig);
    rig.def("get_joint_limit_view",[](const std::string& character,const std::string& bone){nlohmann::json out;require(rtapi::getRigJointLimitView(character,bone,out));return py::module_::import("json").attr("loads")(out.dump());},py::arg("character"),py::arg("bone"));
    rig.def("set_joint_limits",[](const std::string& character,const std::string& bone,float minimum,float maximum,float swing,uint64_t revision){require(rtapi::setRigJointLimits(character,bone,minimum,maximum,swing,revision));},py::arg("character"),py::arg("bone"),py::arg("minimum"),py::arg("maximum"),py::arg("swing"),py::arg("rig_revision").noconvert());
    rig.def("get_joint_limit_overlay",[](){nlohmann::json out;require(rtapi::getRigJointLimitOverlay(out));return py::module_::import("json").attr("loads")(out.dump());});
    rig.def("set_joint_limit_overlay",[](bool visible,bool edit){require(rtapi::setRigJointLimitOverlay(visible,edit));},py::arg("visible").noconvert(),py::arg("edit").noconvert());
    rig.def("get_joint_profile",[](const std::string& character){nlohmann::json out;require(rtapi::getRigJointProfile(character,out));return py::module_::import("json").attr("loads")(out.dump());},py::arg("character"));
    rig.def("suggest_joint_profile",[](const std::string& character){nlohmann::json out;require(rtapi::suggestRigJointProfile(character,out));return py::module_::import("json").attr("loads")(out.dump());},py::arg("character"));
    rig.def("set_joint_profile",[](const std::string& character,const py::dict& profile,uint64_t revision){auto value=nlohmann::json::parse(py::module_::import("json").attr("dumps")(profile).cast<std::string>(),nullptr,false);if(value.is_discarded())throw std::invalid_argument("rig_joint_invalid_schema");require(rtapi::setRigJointProfile(character,value,revision));},py::arg("character"),py::arg("profile"),py::arg("rig_revision").noconvert());
    rig.def("get_pose_state",[](const std::string& character){nlohmann::json out;require(rtapi::getRigPoseState(character,out));return py::module_::import("json").attr("loads")(out.dump());},py::arg("character"));
    rig.def("get_pose_coverage",[](const std::string& character){nlohmann::json out;require(rtapi::getRigPoseCoverage(character,out));return py::module_::import("json").attr("loads")(out.dump());},py::arg("character"));
    rig.def("create_pose_clip",[](const std::string& character,const std::string& name,float fps){require(rtapi::createRigPoseClip(character,name,fps));},py::arg("character"),py::arg("name"),py::arg("fps")=24.f);
    rig.def("select_pose_clip",[](const std::string& character,const std::string& clip){require(rtapi::selectRigPoseClip(character,clip));},py::arg("character"),py::arg("clip"));
    rig.def("set_pose_auto_key",[](bool enabled){require(rtapi::setRigPoseAutoKey(enabled));},py::arg("enabled").noconvert());
    rig.def("set_pose_frame",[](int frame){require(rtapi::setRigPoseFrame(frame));},py::arg("frame").noconvert());
    rig.def("preview_pose_transform",[](const std::string& character,const std::vector<std::string>& bones,const std::vector<float>& delta,uint64_t revision){require(rtapi::previewRigPoseTransform(character,bones,rigRestMatrix(delta),revision));},py::arg("character"),py::arg("bones"),py::arg("world_delta"),py::arg("rig_revision").noconvert());
    rig.def("preview_pose_locals",[](const std::string& character,const py::dict& transforms,uint64_t revision){auto value=nlohmann::json::parse(py::module_::import("json").attr("dumps")(transforms).cast<std::string>(),nullptr,false);if(value.is_discarded())throw std::invalid_argument("invalid_pose_transforms");require(rtapi::previewRigPoseLocals(character,value,revision));},py::arg("character"),py::arg("local_transforms"),py::arg("rig_revision").noconvert());
    rig.def("get_driven_controls", [](const std::string& character) {
        nlohmann::json output;
        require(rtapi::getRigDrivenControls(character, output));
        return py::module_::import("json").attr("loads")(output.dump());
    }, py::arg("character"));
    rig.def("preview_control_values",
            [](const std::string& character, const py::dict& values, uint64_t revision) {
                const auto encoded = py::module_::import("json")
                                         .attr("dumps")(values)
                                         .cast<std::string>();
                const auto value = nlohmann::json::parse(encoded, nullptr, false);
                if (value.is_discarded())
                    throw std::invalid_argument("rig_control_values_invalid");
                require(rtapi::previewRigControlValues(character, value, revision));
            },
            py::arg("character"), py::arg("values"),
            py::arg("rig_revision").noconvert());
    rig.def("insert_pose_keys",[](const std::string& character,const std::vector<std::string>& bones){require(rtapi::insertRigPoseKeys(character,bones));},py::arg("character"),py::arg("bones"));
    rig.def("remove_pose_keys",[](const std::string& character,const std::vector<std::string>& bones){require(rtapi::removeRigPoseKeys(character,bones));},py::arg("character"),py::arg("bones"));
    rig.def(
        "edit_pose_key",
        [](const std::string& character, const std::string& bone,
           const std::string& channel, int sourceFrame, int targetFrame,
           const py::object& value) {
            nlohmann::json encoded = nullptr;
            if (!value.is_none()) {
                encoded = nlohmann::json::parse(
                    py::module_::import("json")
                        .attr("dumps")(value)
                        .cast<std::string>(),
                    nullptr, false);
                if (encoded.is_discarded())
                    throw std::invalid_argument("rig_curve_invalid_value");
            }
            require(rtapi::editRigPoseKey(
                character, bone, channel, sourceFrame, targetFrame, encoded));
        },
        py::arg("character"), py::arg("bone"), py::arg("channel"),
        py::arg("source_frame").noconvert(),
        py::arg("target_frame").noconvert(), py::arg("value") = py::none());
    rig.def("mirror_pose",[](const std::string& character,const std::vector<std::string>& bones,uint64_t revision,const std::string& direction,const std::string& axis){require(rtapi::mirrorRigPose(character,bones,revision,direction,axis));},py::arg("character"),py::arg("bones"),py::arg("rig_revision").noconvert(),py::arg("direction")="selected",py::arg("axis")="x");
    rig.def("apply_pose_preview",[](const std::string& character){require(rtapi::applyRigPosePreview(character));},py::arg("character"));
    rig.def("cancel_pose_preview",[](const std::string& character){require(rtapi::cancelRigPosePreview(character));},py::arg("character"));
    rig.def("select_bones",[](const std::string& character,const std::vector<std::string>& bones,const std::string& active,const std::string& mode,const std::string& anchor){require(rtapi::selectRigBones(character,bones,active,mode,anchor));},py::arg("character"),py::arg("bones"),py::arg("active")="",py::arg("mode")="replace",py::arg("anchor")="");
    rig.def("get_selection",[](){nlohmann::json out;require(rtapi::getRigSelection(out));return py::module_::import("json").attr("loads")(out.dump());});
    rig.def("set_selection_pivot",[](const std::string& mode){require(rtapi::setRigSelectionPivot(mode));},py::arg("mode"));
    rig.def("mirror_rest",[](const std::string& character,const std::vector<std::string>& bones,uint64_t revision,const std::string& direction,const std::string& axis,float offset){require(rtapi::mirrorRigRest(character,bones,direction,axis,offset,revision));},py::arg("character"),py::arg("bones"),py::arg("rig_revision"),py::arg("direction")="selected",py::arg("axis")="x",py::arg("offset")=0.f);
    rig.def("create_mirrored_bone",[](const std::string& character,const std::string& bone,const std::string& name,const std::string& side,uint64_t revision,const std::string& axis,float offset){require(rtapi::createMirroredRigBone(character,bone,name,side,axis,offset,revision));},py::arg("character"),py::arg("bone"),py::arg("name"),py::arg("source_side"),py::arg("rig_revision"),py::arg("axis")="x",py::arg("offset")=0.f);
    rig.def("get_mirrored_landmarks",[](const std::string& character,const py::dict& landmarks,const std::vector<std::string>& bones,uint64_t revision,const std::string& direction,const std::string& axis,float offset){auto marks=nlohmann::json::parse(py::module_::import("json").attr("dumps")(landmarks).cast<std::string>(),nullptr,false);if(marks.is_discarded())throw std::invalid_argument("rig_fit_invalid_landmark");nlohmann::json output;require(rtapi::mirrorRigLandmarks(character,marks,bones,direction,axis,offset,revision,output));return py::module_::import("json").attr("loads")(output.dump());},py::arg("character"),py::arg("landmarks"),py::arg("bones"),py::arg("rig_revision"),py::arg("direction")="selected",py::arg("axis")="x",py::arg("offset")=0.f);
    rig.def("transform_rest",[](const std::string& character,const std::vector<std::string>& bones,const std::vector<float>& delta,uint64_t revision){require(rtapi::transformRigRest(character,bones,rigRestMatrix(delta),revision));},py::arg("character"),py::arg("bones"),py::arg("world_delta"),py::arg("rig_revision"));
    rig.def("set_mode",[](const std::string& mode,const std::string& character){require(rtapi::setRigMode(mode,character));},py::arg("mode"),py::arg("character")="");
    rig.def("get_mode",[](){std::string mode,character;require(rtapi::getRigMode(mode,character));py::dict d;d["mode"]=mode;d["character"]=character;return d;});
    rig.def("get_next_name",[](const std::string& seed){std::string name;require(rtapi::getNextRigName(seed,name));return name;},py::arg("seed")="Rig");
    rig.def("get_scene_transform",[](const std::string& character){Matrix4x4 matrix;require(rtapi::getRigSceneTransform(character,matrix));py::list values;for(int r=0;r<4;++r)for(int c=0;c<4;++c)values.append(matrix.m[r][c]);return values;},py::arg("character"));
    rig.def("set_scene_transform",[](const std::string& character,const std::vector<float>& values){
        if(values.size()!=16)throw std::invalid_argument("invalid_rig_scene_transform");
        require(rtapi::setRigSceneTransform(character,rigRestMatrix(values)));
    },py::arg("character"),py::arg("scene_transform"));
    rig.def("get_next_bone_name",[](const std::string& character,const std::string& seed){std::string name;require(rtapi::getNextRigBoneName(character,seed,name));return name;},py::arg("character"),py::arg("seed")="Joint1");
    rig.def("rename_bone",[](const std::string& character,const std::string& bone,const std::string& name){require(rtapi::renameRigBone(character,bone,name));},py::arg("character"),py::arg("bone"),py::arg("name"));
    rig.def("reparent_bone",[](const std::string& character,const std::string& bone,const std::string& parent){require(rtapi::reparentRigBone(character,bone,parent));},py::arg("character"),py::arg("bone"),py::arg("parent"));
    rig.def("delete_bone",[](const std::string& character,const std::string& bone){require(rtapi::deleteRigBone(character,bone));},py::arg("character"),py::arg("bone"));
    rig.def("copy_from",[](const std::string& source,const std::string& character){
        std::vector<RigAuthoring::RigCopyBone> mapping;require(rtapi::copyRigFrom(source,character,mapping));
        py::dict result;result["character"]=character;py::list bones;
        for(const auto& bone:mapping){py::dict entry;entry["source_bone"]=bone.source_bone;entry["target_bone"]=bone.target_bone;bones.append(entry);}
        result["bone_map"]=bones;return result;
    },py::arg("source_character"),py::arg("character"));
    rig.def("get_anatomy",[](const std::string& character){nlohmann::json value;require(rtapi::getRigAnatomy(character,value));return py::module_::import("json").attr("loads")(value.dump());},py::arg("character"));
    rig.def("set_anatomy",[](const std::string& character,const py::dict& anatomy){
        const auto encoded=py::module_::import("json").attr("dumps")(anatomy).cast<std::string>();
        const auto value=nlohmann::json::parse(encoded,nullptr,false);
        if(value.is_discarded())throw std::invalid_argument("rig_anatomy_invalid_schema");
        require(rtapi::setRigAnatomy(character,value));
    },py::arg("character"),py::arg("anatomy"));
    rig.def("set_pose_view",[](const std::string& character,const std::string& mode){require(rtapi::setRigPoseView(character,mode));},py::arg("character"),py::arg("mode"));
    rig.def("get_pose_view",[](const std::string& character){std::string mode,effective;require(rtapi::getRigPoseView(character,mode,effective));py::dict d;d["mode"]=mode;d["effective_mode"]=effective;return d;},py::arg("character"));
    rig.def("list_fit_targets",[](){nlohmann::json out;require(rtapi::listRigFitTargets(out));return py::module_::import("json").attr("loads")(out.dump());});
    rig.def("get_fit_setup",[](const std::string& character,const std::string& mesh){nlohmann::json out;require(rtapi::getRigFitSetup(character,mesh,out));return py::module_::import("json").attr("loads")(out.dump());},py::arg("character"),py::arg("mesh"));
    rig.def("preview_fit",[](const std::string& character,const std::string& mesh,const py::dict& landmarks,bool confirmed){auto value=nlohmann::json::parse(py::module_::import("json").attr("dumps")(landmarks).cast<std::string>(),nullptr,false);if(value.is_discarded())throw std::invalid_argument("rig_fit_invalid_landmark");nlohmann::json out;require(rtapi::previewRigFit(character,mesh,value,confirmed,out));return py::module_::import("json").attr("loads")(out.dump());},py::arg("character"),py::arg("mesh"),py::arg("landmarks"),py::arg("axes_confirmed")=false);
    rig.def("commit_fit",[](const std::string& character,const std::string& mesh,const py::dict& preview){auto value=nlohmann::json::parse(py::module_::import("json").attr("dumps")(preview).cast<std::string>(),nullptr,false);if(value.is_discarded())throw std::invalid_argument("rig_fit_invalid_preview");require(rtapi::commitRigFit(character,mesh,value));},py::arg("character"),py::arg("mesh"),py::arg("preview"));
    rig.def(
        "set_weight_map_visible",
        [](bool visible) { require(rtapi::setRigWeightMapVisible(visible)); }, py::arg("visible"));
    rig.def("get_weight_map_visible", []() {
        bool visible;
        require(rtapi::getRigWeightMapVisible(visible));
        return visible;
    });
    rig.def(
        "get_weight_map",
        [](const std::string &mesh, const std::string &character, const std::string &bone) {
            nlohmann::json out;
            require(rtapi::getRigWeightMap(mesh, character, bone, out));
            return py::module_::import("json").attr("loads")(out.dump());
        },
        py::arg("mesh"), py::arg("character"), py::arg("bone"));
    rig.def(
        "get_weights",
        [](const std::string &object, uint64_t vertex) {
            nlohmann::json report;
            require(rtapi::getRigVertexWeights(object, vertex, report));
            return py::module_::import("json").attr("loads")(report.dump());
        },
        py::arg("object"), py::arg("vertex"));
    rig.def(
        "weight_stats",
        [](const std::string &mesh) {
            nlohmann::json report;
            require(rtapi::getRigWeightStats(mesh, report));
            return py::module_::import("json").attr("loads")(report.dump());
        },
        py::arg("mesh"));
    rig.def(
        "preview_envelope_weights",
        [](const std::string &character, float torso, float limb, float extremity, float falloff) {
            RigAuthoring::EnvelopeWeightSettings settings{torso, limb, extremity, falloff};
            nlohmann::json out;
            require(rtapi::previewRigEnvelopeWeights(character, settings, out));
            return py::module_::import("json").attr("loads")(out.dump());
        },
        py::arg("character"), py::arg("torso_radius") = .16f, py::arg("limb_radius") = .065f,
        py::arg("extremity_radius") = .05f, py::arg("falloff") = 2.f);
    rig.def(
        "apply_envelope_weights",
        [](const std::string &character, uint64_t revision, float torso, float limb,
           float extremity, float falloff) {
            RigAuthoring::EnvelopeWeightSettings settings{torso, limb, extremity, falloff};
            require(rtapi::applyRigEnvelopeWeights(character, settings, revision));
        },
        py::arg("character"), py::arg("rig_revision").noconvert(), py::arg("torso_radius") = .16f,
        py::arg("limb_radius") = .065f, py::arg("extremity_radius") = .05f,
        py::arg("falloff") = 2.f);
    rig.def("get_envelope_overlay", []() {
        nlohmann::json out;
        require(rtapi::getRigEnvelopeOverlay(out));
        return py::module_::import("json").attr("loads")(out.dump());
    });
    rig.def(
        "set_envelope_overlay",
        [](const std::string &character, bool visible, float torso, float limb, float extremity,
           float falloff) {
            RigAuthoring::EnvelopeWeightSettings settings{torso, limb, extremity, falloff};
            require(rtapi::setRigEnvelopeOverlay(character, settings, visible));
        },
        py::arg("character"), py::arg("visible").noconvert(), py::arg("torso_radius") = .16f,
        py::arg("limb_radius") = .065f, py::arg("extremity_radius") = .05f,
        py::arg("falloff") = 2.f);
    rig.def(
        "get_bone_envelope",
        [](const std::string &character, const std::string &bone) {
            nlohmann::json out;
            require(rtapi::getRigBoneEnvelope(character, bone, out));
            return py::module_::import("json").attr("loads")(out.dump());
        },
        py::arg("character"), py::arg("bone"));
    rig.def(
        "apply_bone_envelope",
        [](const std::string &character, const std::string &bone, uint64_t revision,
           float startRadius, float endRadius, float startExtension, float endExtension,
           float falloff) {
            RigAuthoring::EnvelopeBoneProfile profile{bone, startRadius, endRadius,
                                                       startExtension, endExtension, falloff};
            require(rtapi::applyRigBoneEnvelope(character, profile, revision));
        },
        py::arg("character"), py::arg("bone"), py::arg("rig_revision").noconvert(),
        py::arg("start_radius"), py::arg("end_radius"), py::arg("start_extension") = 0.f,
        py::arg("end_extension") = 0.f, py::arg("falloff") = 2.f);
    rig.def(
        "preview_bind",
        [](const std::string &character, const std::string &mesh, bool confirmed) {
            nlohmann::json out;
            require(rtapi::previewRigMeshBinding(character, mesh, confirmed, out));
            return py::module_::import("json").attr("loads")(out.dump());
        },
        py::arg("character"), py::arg("mesh"), py::arg("axes_confirmed") = false);
    rig.def(
        "bind_mesh",
        [](const std::string &character, const std::string &mesh, const py::dict &preview) {
            auto value = nlohmann::json::parse(
                py::module_::import("json").attr("dumps")(preview).cast<std::string>(), nullptr,
                false);
            if (value.is_discarded())
                throw std::invalid_argument("rig_bind_invalid_preview");
            require(rtapi::bindRigMesh(character, mesh, value));
        },
        py::arg("character"), py::arg("mesh"), py::arg("preview"));
    rig.def(
        "unbind_mesh",
        [](const std::string &character) {
            require(rtapi::unbindRigMesh(character));
        },
        py::arg("character"));
    rig.def(
        "get_binding",
        [](const std::string &character) {
            nlohmann::json out;
            require(rtapi::getRigMeshBinding(character, out));
            return py::module_::import("json").attr("loads")(out.dump());
        },
        py::arg("character"));
    rig.def(
        "preflight",
        [](const std::string &mesh) {
            nlohmann::json report;
            require(rtapi::preflightRigMesh(mesh, report));
            return py::module_::import("json").attr("loads")(report.dump());
        },
        py::arg("mesh"));
    rig.def("list_templates", []() {
        nlohmann::json value;
        require(rtapi::listRigTemplates(value));
        return py::module_::import("json").attr("loads")(value.dump());
    });
    rig.def(
        "get_template",
        [](const std::string &id, float height) {
            nlohmann::json value;
            require(rtapi::getRigTemplate(id, height, value));
            return py::module_::import("json").attr("loads")(value.dump());
        },
        py::arg("template_id"), py::arg("height") = 1.8f);
    rig.def(
        "create",
        [](const std::string &character, const std::string &id, float height) {
            require(rtapi::createRig(character, id, height));
        },
        py::arg("character"), py::arg("template_id") = "root", py::arg("height") = 1.8f);
    rig.def(
        "add_bone",
        [](const std::string &character, const std::string &name, const std::string &parent,
           const std::vector<float> &rest) {
            require(rtapi::addRigBone(character, name, parent, rigRestMatrix(rest)));
        },
        py::arg("character"), py::arg("name"), py::arg("parent"),
        py::arg("rest_transform") =
            std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
    rig.def("set_rest_transform",[](const std::string& character,const std::string& bone,const std::vector<float>& rest){require(rtapi::setRigRestTransform(character,bone,rigRestMatrix(rest)));},
            py::arg("character"),py::arg("bone"),py::arg("rest_transform"));
    rig.def("list_characters", []() { std::vector<std::string> names; require(rtapi::listRigCharacters(names)); py::list out; for (const auto& name : names) out.append(name); return out; });
    rig.def("list_bones", [](const std::string& character) { std::vector<RigAuthoring::BoneView> bones; require(rtapi::listRigBones(character, bones)); py::list out; for (const auto& b : bones) out.append(boneDict(b)); return out; }, py::arg("character"));
    rig.def("select_bone", [](const std::string& character, const std::string& bone) { require(rtapi::selectRigBone(character, bone)); }, py::arg("character"), py::arg("bone"));
    rig.def("get_selected_bone", []() -> py::object { RigAuthoring::BoneView b; bool selected; require(rtapi::getSelectedRigBone(b, selected)); if (selected) return boneDict(b); return py::none(); });
    rig.def("clear_selection", []() { require(rtapi::clearRigSelection()); });
    rig.def("get_overlay_visible", []() { bool visible; require(rtapi::getRigOverlayVisible(visible)); return visible; });
    rig.def("set_overlay_visible", [](bool visible) { require(rtapi::setRigOverlayVisible(visible)); }, py::arg("visible").noconvert());
}
