#include "RtApiInternal.h"
#include "Api/RtApiRigAnatomy.h"
#include "Animation/RigAnatomy.h"
#include "json.hpp"
namespace rtapi {
Result getRigAnatomy(const std::string& character,nlohmann::json& output) {
    output=nullptr;
    if(!g_ctx)return Result::fail("api_not_bound");
    if(renderJobActive())return Result::fail("scene_locked");
    for(const auto& model:g_ctx->scene.importedModelContexts)if(model.importName==character) {
        output=RigAuthoring::serializeRigAnatomy(model.rigAnatomy);return Result::success();
    }
    return Result::fail("unknown_character");
}
}
