#include "RtApiInternal.h"
#include "Api/RtApiRigSelection.h"
#include "Animation/RigSelection.h"
#include <exception>
#include <algorithm>
namespace rtapi {
Result selectRigBones(const std::string& character,const std::vector<std::string>& bones,const std::string& active,const std::string& mode,const std::string& anchor) {
    if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");
    try{std::string error;if(!RigAuthoring::selectBones(g_ctx->scene,character,bones,active,mode,error,anchor))return Result::fail(error);
        g_ctx->selection.clearSelection();return Result::success();}catch(const std::exception&){return Result::fail("rig_selection_failed");}
}
Result getRigSelection(nlohmann::json& output) {
    output=nullptr;if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");
    try {const auto names=RigAuthoring::selectedBones(g_ctx->scene);const auto& view=g_ctx->scene.rigView;
        const auto anchor=view.selection_character==view.character && std::find(names.begin(),names.end(),view.selection_anchor)!=names.end()?view.selection_anchor:(names.empty()?std::string():view.bone);
        output={{"anchor",anchor},{"character",names.empty()?"":view.character},{"bones",names},{"active",names.empty()?"":view.bone},{"pivot",view.selection_pivot}};return Result::success();
    }catch(const std::exception&){return Result::fail("rig_selection_failed");}
}
Result setRigSelectionPivot(const std::string& mode) {
    if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");
    std::string error;return RigAuthoring::setSelectionPivot(g_ctx->scene,mode,error)?Result::success():Result::fail(error);
}
}
