#include "RtApiInternal.h"
#include "Api/RtApiRigWeightMap.h"
#include "Animation/RigWeights.h"
#include <exception>
namespace rtapi {
Result setRigWeightMapVisible(bool visible) {
    if(!g_ctx)return Result::fail("api_not_bound");
    if(renderJobActive())return Result::fail("scene_locked");
    g_ctx->scene.rigView.weight_map_visible=visible;return Result::success();
}
Result getRigWeightMapVisible(bool& visible) {
    visible=false;if(!g_ctx)return Result::fail("api_not_bound");
    visible=g_ctx->scene.rigView.weight_map_visible;return Result::success();
}
Result getRigWeightMap(const std::string& mesh,const std::string& character,const std::string& bone,nlohmann::json& output) {
    output=nullptr;if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");
    try {
        RigAuthoring::BoneWeightField field;std::string error;
        if(!RigAuthoring::boneWeightField(g_ctx->scene,mesh,character,bone,field,error))return Result::fail(error);
        output={{"mesh",mesh},{"character",character},{"bone",bone},{"bone_index",field.bone_index},{"vertex_count",field.values.size()},{"values",field.values},{"invalid_entries",field.invalid_entries},{"display_clamped",true}};
        return Result::success();
    }catch(const std::exception&){return Result::fail("rig_weight_map_failed");}
}
}
