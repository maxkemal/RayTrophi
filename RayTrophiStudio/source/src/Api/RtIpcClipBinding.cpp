#include "RtClipBindingBindings.h"
#include "Api/RtApi.h"
#include "Api/RtApiClipBinding.h"
#include "Animation/ClipBinding.h"
#include "Animation/RigPosePreview.h"
using json = nlohmann::json;
namespace {
json reportJson(const RigAuthoring::ClipBindingReport& r) {
    json matches = json::array();
    for (const auto& m : r.matches) matches.push_back({{"source", m.source}, {"target", m.target}, {"authored_name", m.authored_name}});
    return {{"ready", r.ready}, {"mode", r.mode}, {"translation_scale", r.translation_scale}, {"source_character", r.source_character}, {"source_clip", r.source_clip},
            {"target_character", r.target_character}, {"output_clip", r.output_clip}, {"matches", matches},
            {"unmapped", r.unmapped}, {"ambiguous", r.ambiguous}, {"hierarchy_mismatches", r.hierarchy_mismatches},
            {"rest_difference_count", r.rest_difference_count}};
}
bool valid(const json& params) {
    for (const auto* key : {"source_character", "source_clip", "target_character"})
        if (!params.contains(key) || !params[key].is_string()) return false;
    if (params.contains("node_map")) {
        if (!params["node_map"].is_object()) return false;
        for (const auto& entry : params["node_map"].items()) if (!entry.value().is_string()) return false;
    }
    if (params.contains("mode") && !params["mode"].is_string()) return false;
    if (params.contains("translation_scale") && !params["translation_scale"].is_number()) return false;
    return true;
}
}
bool dispatchClipBindingIpc(const std::string& method, const json& params, const RtIpcTemplateEnqueue& enqueue, json& out) {
    if (method == "anim.sample_clip_binding") {
        if ((params.contains("source_pose_view") && !params["source_pose_view"].is_string()) ||
            (params.contains("target_pose_view") && !params["target_pose_view"].is_string()) || !valid(params) || (params.contains("time_seconds") && !params["time_seconds"].is_number())) {
            out={{"__error","Invalid clip preview parameter types"},{"code","invalid_parameter"}}; return true;
        }
        const std::string source_character=params.at("source_character").get<std::string>();
        const std::string source_clip=params.at("source_clip").get<std::string>();
        const std::string target_character=params.at("target_character").get<std::string>();
        const double time_seconds=params.value("time_seconds",0.);
        const auto node_map=params.value("node_map",std::map<std::string,std::string>{});
        const std::string mode=params.value("mode",std::string("same_rig"));
        const float translation_scale=params.value("translation_scale",1.f);
        const std::string sourceView=params.value("source_pose_view",std::string("animated"));
        const std::string targetView=params.value("target_pose_view",std::string("animated"));
        out=enqueue([source_character,source_clip,target_character,time_seconds,node_map,mode,translation_scale,sourceView,targetView](UIContext&) {
            RigAuthoring::ClipPosePreview p; auto r=rtapi::sampleClipBinding(source_character,source_clip,target_character,time_seconds,p,node_map,mode,translation_scale,sourceView,targetView);
            if (!r.ok) return json{{"__error",r.error},{"code",r.error}};
            auto joints=[](const std::vector<RigAuthoring::PreviewJoint>& values) {
                json list=json::array();
                for(const auto& joint:values) {
                    std::vector<float> matrix; for(int a=0;a<4;++a) for(int b=0;b<4;++b) matrix.push_back(joint.world.m[a][b]);
                    list.push_back({{"name",joint.name},{"parent",joint.parent},{"world_transform",matrix}});
                }
                return list;
            };
            return json{{"binding",reportJson(p.binding)},{"time_seconds",p.time_seconds},{"duration_seconds",p.duration_seconds},
                        {"source_pose_source",p.source_pose_source},{"target_pose_source",p.target_pose_source},{"source",joints(p.source)},{"target",joints(p.target)}};
        }); return true;
    }
    if (method == "anim.preview_clip_binding") {
        if (!valid(params)) { out = {{"__error", "Character/clip parameters must be strings; node_map must map strings to strings; mode must be string and translation_scale numeric"}, {"code", "invalid_parameter"}}; return true; }
        const std::string source_character = params.at("source_character").get<std::string>();
        const std::string source_clip = params.at("source_clip").get<std::string>();
        const std::string target_character = params.at("target_character").get<std::string>();
        const auto node_map = params.value("node_map", std::map<std::string, std::string>{});
        const std::string mode = params.value("mode", std::string("same_rig"));
        const float translation_scale = params.value("translation_scale", 1.f);
        out = enqueue([source_character, source_clip, target_character, node_map, mode, translation_scale](UIContext&) {
            RigAuthoring::ClipBindingReport report; auto r = rtapi::previewClipBinding(source_character, source_clip, target_character, report, node_map, mode, translation_scale);
            return r.ok ? reportJson(report) : json{{"__error", r.error}, {"code", r.error}};
        }); return true;
    }
    if (method == "anim.bind_clip") {
        if (!valid(params) || (params.contains("output_name") && !params["output_name"].is_string())) {
            out = {{"__error", "Character/clip parameters must be strings; node_map must map strings to strings; mode must be string and translation_scale numeric"}, {"code", "invalid_parameter"}}; return true;
        }
        const std::string source_character = params.at("source_character").get<std::string>();
        const std::string source_clip = params.at("source_clip").get<std::string>();
        const std::string target_character = params.at("target_character").get<std::string>();
        const std::string output_name = params.value("output_name", std::string());
        const auto node_map = params.value("node_map", std::map<std::string, std::string>{});
        const std::string mode = params.value("mode", std::string("same_rig"));
        const float translation_scale = params.value("translation_scale", 1.f);
        out = enqueue([source_character, source_clip, target_character, output_name, node_map, mode, translation_scale](UIContext&) {
            RigAuthoring::ClipBindingReport report; auto r = rtapi::bindAnimationClip(source_character, source_clip, target_character, output_name, report, node_map, mode, translation_scale);
            return r.ok ? reportJson(report) : json{{"__error", r.error}, {"code", r.error}, {"report", reportJson(report)}};
        }); return true;
    }
    return false;
}
