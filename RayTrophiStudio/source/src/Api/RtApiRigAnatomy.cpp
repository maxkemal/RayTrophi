#include "RtApiInternal.h"
#include "Api/RtApiRigAnatomy.h"
#include "Animation/RigAnatomy.h"
#include "Animation/RigPosePreview.h"
#include "json.hpp"
#include <algorithm>
namespace rtapi {
namespace {
bool endsWith(const std::string& value,const char* suffix) {
    const std::string tail(suffix);
    return value.size()>=tail.size()&&value.compare(value.size()-tail.size(),tail.size(),tail)==0;
}
Vec3 jointPosition(const RayTrophi::NodeHierarchy& h,const std::vector<RigAuthoring::PreviewJoint>& joints,const std::string& bone) {
    const auto& m=joints[static_cast<size_t>(h.find(bone)-h.nodes.data())].world;
    return Vec3(m.m[0][3],m.m[1][3],m.m[2][3]);
}
}
Result getRigAnatomy(const std::string& character,nlohmann::json& output) {
    output=nullptr;
    if(!g_ctx)return Result::fail("api_not_bound");
    if(renderJobActive())return Result::fail("scene_locked");
    for(const auto& model:g_ctx->scene.importedModelContexts)if(model.importName==character) {
        output=RigAuthoring::serializeRigAnatomy(model.rigAnatomy);return Result::success();
    }
    return Result::fail("unknown_character");
}
// Fills bend hints and foot pivots for rigs created before anatomy v8 or after a refit. The
// character forward comes from the feet (ankle to toe), never from a fixed axis, so imported
// rigs facing -Z or +X get the same knees-forward / elbows-back result as the templates.
Result deriveRigLimbHints(const std::string& character,bool overwrite,nlohmann::json& report) {
    report=nullptr;
    if(!g_ctx)return Result::fail("api_not_bound");
    if(renderJobActive())return Result::fail("scene_locked");
    for(const auto& model:g_ctx->scene.importedModelContexts) {
        if(model.importName!=character)continue;
        auto anatomy=model.rigAnatomy;
        const auto& h=model.nodeHierarchy;
        std::vector<RigAuthoring::PreviewJoint> joints;std::string error;
        if(!RigAuthoring::sampleRigPose(h,nullptr,0,joints,error))return Result::fail(error);
        const Vec3 up(0,1,0);
        Vec3 forward(0,0,0);
        for(const auto& chain:anatomy.chains) {
            if(!endsWith(chain.name,"_leg")||chain.bones.size()<4)continue;
            auto flat=jointPosition(h,joints,chain.bones[3])-jointPosition(h,joints,chain.bones[2]);
            flat-=up*Vec3::dot(flat,up);
            if(flat.length_squared()>1e-12f)forward+=flat.normalize();
        }
        const bool forwardAssumed=forward.length_squared()<1e-8f;
        forward=forwardAssumed?Vec3(0,0,1):forward.normalize();
        auto pivots=nlohmann::json::array(),hints=nlohmann::json::array(),estimated=nlohmann::json::array();
        for(const auto& chain:anatomy.chains) {
            if(endsWith(chain.name,"_leg")&&chain.bones.size()>=3&&(overwrite||!RigAuthoring::findFootPivot(anatomy,chain.name))) {
                RigAuthoring::RigFootPivot pivot;bool guessed=false;
                if(!RigAuthoring::deriveFootPivots(anatomy,h,chain.name,pivot,guessed,error))return Result::fail(error);
                auto& rows=anatomy.footPivots;
                rows.erase(std::remove_if(rows.begin(),rows.end(),[&](const auto& row){return row.chain==chain.name;}),rows.end());
                rows.push_back(pivot);pivots.push_back(chain.name);
                if(guessed)estimated.push_back(chain.name);
            }
            if(!overwrite&&RigAuthoring::findBendHint(anatomy,chain.name))continue;
            Vec3 direction(0,0,0);
            const bool leg=endsWith(chain.name,"_leg"),arm=endsWith(chain.name,"_arm");
            if(anatomy.family=="humanoid"&&(leg||arm)) {
                direction=leg?forward:forward*-1.f;
            } else if(chain.bones.size()>=3) {
                // Non-humanoid limbs keep the rest bend, but only where it is clearly readable.
                const auto a=jointPosition(h,joints,chain.bones[0]),b=jointPosition(h,joints,chain.bones[1]),t=jointPosition(h,joints,chain.bones[2]);
                const auto axis=(t-a).length_squared()>1e-12f?(t-a).normalize():Vec3(0,1,0);
                const auto bend=(b-a)-axis*Vec3::dot(b-a,axis);
                if(bend.length()>.02f*((b-a).length()+(t-b).length()))direction=bend.normalize();
            }
            if(direction.length_squared()<1e-8f)continue;
            auto& rows=anatomy.bendHints;
            rows.erase(std::remove_if(rows.begin(),rows.end(),[&](const auto& row){return row.chain==chain.name;}),rows.end());
            rows.push_back({chain.name,direction});hints.push_back(chain.name);
        }
        const auto saved=setRigAnatomy(character,RigAuthoring::serializeRigAnatomy(anatomy));
        if(!saved.ok)return saved;
        report={{"character",character},{"foot_pivots_set",pivots},{"bend_hints_set",hints},
                {"pivots_estimated",estimated},{"forward",{forward.x,forward.y,forward.z}},
                {"forward_assumed",forwardAssumed}};
        return Result::success();
    }
    return Result::fail("unknown_character");
}
}
