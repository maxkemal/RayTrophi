#include "RtApiInternal.h"
#include "Api/RtApiRigMirror.h"
#include "Animation/RigMirror.h"
#include "Animation/RigEditing.h"
#include <exception>
namespace rtapi {
Result mirrorRigLandmarks(const std::string& character,const nlohmann::json& landmarks,const std::vector<std::string>& bones,const std::string& direction,const std::string& axis,float offset,uint64_t revision,nlohmann::json& output) {
    output=nullptr;if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");
    try {
        std::string error;if(!RigAuthoring::canEditRig(g_ctx->scene,character,error))return Result::fail(error);
        for(const auto& model:g_ctx->scene.importedModelContexts)if(model.importName==character) {
            if(model.rigRevision!=revision)return Result::fail("rig_edit_stale_revision");
            if(!RigAuthoring::mirrorLandmarks(model.nodeHierarchy,model.rigAnatomy,model.rigSceneTransform,landmarks,bones,direction,{axis,offset},output,error))return Result::fail(error);
            return Result::success();
        }
        return Result::fail("unknown_character");
    }catch(const std::exception&){output=nullptr;return Result::fail("rig_mirror_failed");}
}
}
