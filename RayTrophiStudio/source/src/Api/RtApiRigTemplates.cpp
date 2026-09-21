#include "RtApiInternal.h"
#include "Api/RtApiRigTemplates.h"
#include "Animation/RigTemplates.h"
#include "Animation/RigPreflight.h"
#include "Animation/RigWeights.h"
#include "Animation/RigFitting.h"
#include "Animation/RigSerialization.h"
#include <exception>
namespace rtapi {
Result listRigFitTargets(nlohmann::json& output) {
    output=nullptr;if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");
    try{output=RigAuthoring::fitTargets(g_ctx->scene);return Result::success();}catch(const std::exception&){return Result::fail("rig_preflight_failed");}
}
Result getRigFitSetup(const std::string& character,const std::string& mesh,nlohmann::json& output) {
    output=nullptr;if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");
    try{std::string error;return RigAuthoring::fitSetup(g_ctx->scene,character,mesh,output,error)?Result::success():Result::fail(error);}catch(const std::exception&){return Result::fail("rig_fit_failed");}
}
Result previewRigFit(const std::string& character,const std::string& mesh,const nlohmann::json& landmarks,bool confirmed,nlohmann::json& output) {
    output=nullptr;if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");
    try{std::string error;return RigAuthoring::previewFit(g_ctx->scene,character,mesh,landmarks,confirmed,output,error)?Result::success():Result::fail(error);}catch(const std::exception&){return Result::fail("rig_fit_failed");}
}
Result getRigVertexWeights(const std::string& object,uint64_t vertex,nlohmann::json& report) {
    report=nullptr;if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");
    try{std::string error;return RigAuthoring::vertexWeights(g_ctx->scene,object,vertex,report,error)?Result::success():Result::fail(error);}
    catch(const std::exception&){return Result::fail("rig_get_weights_failed");}
}
Result getRigWeightStats(const std::string& mesh,nlohmann::json& report) {
    report=nullptr;if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");
    try{std::string error;return RigAuthoring::weightStats(g_ctx->scene,mesh,report,error)?Result::success():Result::fail(error);}
    catch(const std::exception&){return Result::fail("rig_weight_stats_failed");}
}
Result preflightRigMesh(const std::string& mesh,nlohmann::json& report) {
    report=nullptr;if(!g_ctx)return Result::fail("api_not_bound");
    if(renderJobActive())return Result::fail("scene_locked");
    try{std::string error;return RigAuthoring::preflightMesh(g_ctx->scene,mesh,report,error)?Result::success():Result::fail(error);}
    catch(const std::exception&){return Result::fail("rig_preflight_failed");}
}
Result listRigTemplates(nlohmann::json& output) {
    output=nlohmann::json::array();
    for(const auto& info:RigAuthoring::rigTemplateCatalogue())output.push_back({
        {"template_id",info.id},{"template_version",info.version},{"label",info.label},{"family",info.family},{"joint_count",info.joint_count},
        {"default_height",info.default_height},{"up_axis","+Y"},{"forward_axis","+Z"},{"left_axis","+X"}});
    return Result::success();
}
Result getRigTemplate(const std::string& id,float height,nlohmann::json& output) {
    output=nullptr;RayTrophi::NodeHierarchy h;RigAuthoring::RigAnatomy a;std::string error;
    if(!RigAuthoring::buildRigTemplate(id,"Template",height,h,a,error))return Result::fail(error);
    int version=1;for(const auto& info:RigAuthoring::rigTemplateCatalogue())if(info.id==id){version=info.version;break;}
    output={{"template_id",id},{"template_version",version},{"height",height},{"nodeHierarchy",RigAuthoring::serializeRigHierarchy(h)},
        {"anatomy",RigAuthoring::serializeRigAnatomy(a)}};return Result::success();
}
}
