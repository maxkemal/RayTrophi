#include "RtClipBindingBindings.h"
#include "Api/RtApi.h"
#include "Api/RtApiClipBinding.h"
#include "Animation/ClipBinding.h"
#include "Animation/RigPosePreview.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
namespace py = pybind11;
namespace {
py::dict reportDict(const RigAuthoring::ClipBindingReport& r) {
    py::dict d; d["ready"] = r.ready; d["mode"] = r.mode; d["translation_scale"] = r.translation_scale;
    d["source_character"] = r.source_character; d["source_clip"] = r.source_clip;
    d["target_character"] = r.target_character; d["output_clip"] = r.output_clip;
    d["unmapped"] = r.unmapped; d["ambiguous"] = r.ambiguous; d["hierarchy_mismatches"] = r.hierarchy_mismatches;
    d["rest_difference_count"] = r.rest_difference_count; py::list matches;
    for (const auto& m : r.matches) { py::dict item; item["source"] = m.source; item["target"] = m.target; item["authored_name"] = m.authored_name; matches.append(item); }
    d["matches"] = matches; return d;
}
}
void registerClipBindingPython(py::module_& anim) {
    anim.def("sample_clip_binding", [](const std::string& source, const std::string& clip, const std::string& target,
             double time, const std::map<std::string, std::string>& map, const std::string& mode, float scale, const std::string& sourceView, const std::string& targetView) {
        RigAuthoring::ClipPosePreview p; auto r=rtapi::sampleClipBinding(source,clip,target,time,p,map,mode,scale,sourceView,targetView);
        if (!r.ok) throw std::invalid_argument(r.error);
        auto joints=[](const std::vector<RigAuthoring::PreviewJoint>& values) {
            py::list list;
            for (const auto& joint : values) {
                py::dict item; item["name"]=joint.name; item["parent"]=joint.parent;
                std::vector<float> matrix; for(int a=0;a<4;++a) for(int b=0;b<4;++b) matrix.push_back(joint.world.m[a][b]);
                item["world_transform"]=matrix; list.append(item);
            }
            return list;
        };
        py::dict d; d["binding"]=reportDict(p.binding); d["time_seconds"]=p.time_seconds;
        d["duration_seconds"]=p.duration_seconds;d["source_pose_source"]=p.source_pose_source; d["target_pose_source"]=p.target_pose_source;
        d["source"]=joints(p.source); d["target"]=joints(p.target); return d;
    }, py::arg("source_character"), py::arg("source_clip"), py::arg("target_character"),
       py::arg("time_seconds")=0., py::arg("node_map")=std::map<std::string,std::string>{},
       py::arg("mode")="same_rig", py::arg("translation_scale")=1.f, py::arg("source_pose_view")="animated",py::arg("target_pose_view")="animated");
    anim.def("preview_clip_binding", [](const std::string& source, const std::string& clip, const std::string& target, const std::map<std::string, std::string>& nodeMap, const std::string& mode, float translationScale) {
        RigAuthoring::ClipBindingReport report; auto r = rtapi::previewClipBinding(source, clip, target, report, nodeMap, mode, translationScale);
        if (!r.ok) throw std::invalid_argument(r.error); return reportDict(report);
    }, py::arg("source_character"), py::arg("source_clip"), py::arg("target_character"), py::arg("node_map") = std::map<std::string, std::string>{}, py::arg("mode") = "same_rig", py::arg("translation_scale") = 1.f);
    anim.def("bind_clip", [](const std::string& source, const std::string& clip, const std::string& target, const std::string& name, const std::map<std::string, std::string>& nodeMap, const std::string& mode, float translationScale) {
        RigAuthoring::ClipBindingReport report; auto r = rtapi::bindAnimationClip(source, clip, target, name, report, nodeMap, mode, translationScale);
        if (!r.ok) throw std::invalid_argument(r.error); return reportDict(report);
    }, py::arg("source_character"), py::arg("source_clip"), py::arg("target_character"), py::arg("output_name") = "", py::arg("node_map") = std::map<std::string, std::string>{}, py::arg("mode") = "same_rig", py::arg("translation_scale") = 1.f);
}
