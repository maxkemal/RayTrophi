#include "RtApiInternal.h"
#include "Api/RtApiRigEditing.h"
#include "Animation/RigEditing.h"
#include "Animation/RigSelection.h"
#include <algorithm>
#include "ProjectManager.h"
#include <functional>
#include "json.hpp"
#include <utility>

namespace {
class RigEditCommand final:public SceneCommand {
    BoneData pendingBones;
    SceneData::ImportedModelContext pendingModel;
    RigAuthoring::ViewState pendingSelection;
    std::string character;
    bool created;
public:
    RigEditCommand(RigAuthoring::RigEditState state,const RigAuthoring::ViewState& view,bool create):
        pendingBones(std::move(state.bones)),pendingModel(std::move(state.model)),pendingSelection(view),
        character(pendingModel.importName),created(create) {
        pendingSelection.character=character;pendingSelection.bone=state.selectedBone;
        if(!state.selectedBones.empty()) {
            const bool previousOwner=pendingSelection.selection_character==character;
            pendingSelection.selected_bones=std::move(state.selectedBones);pendingSelection.selection_character=character;
            if(!previousOwner || std::find(pendingSelection.selected_bones.begin(),pendingSelection.selected_bones.end(),pendingSelection.selection_anchor)==pendingSelection.selected_bones.end())pendingSelection.selection_anchor=pendingSelection.bone;
        } else if(pendingSelection.selection_character!=character || std::find(pendingSelection.selected_bones.begin(),pendingSelection.selected_bones.end(),pendingSelection.bone)==pendingSelection.selected_bones.end()) {
            pendingSelection.selected_bones.clear();if(!pendingSelection.bone.empty())pendingSelection.selected_bones.push_back(pendingSelection.bone);
            pendingSelection.selection_character=character;pendingSelection.selection_anchor=pendingSelection.bone;
        }
    }
    void prepare(UIContext& ctx) {if(created)ctx.scene.importedModelContexts.reserve(ctx.scene.importedModelContexts.size()+1);}
    void execute(UIContext& ctx) override {
        auto& models=ctx.scene.importedModelContexts;
        if(created) {
            models.reserve(models.size()+1); // Fallible allocation before scene mutation.
            models.push_back(std::move(pendingModel));
        } else {
            for(auto& model:models)if(model.importName==character){std::swap(model,pendingModel);break;}
        }
        std::swap(ctx.scene.boneData,pendingBones);RigAuthoring::exchangeSelection(ctx.scene.rigView,pendingSelection);
        ProjectManager::getInstance().markModified();
    }
    void undo(UIContext& ctx) override {
        auto& models=ctx.scene.importedModelContexts;
        for(auto i=models.begin();i!=models.end();++i)if(i->importName==character) {
            if(created){pendingModel=std::move(*i);models.erase(i);}else std::swap(*i,pendingModel);
            break;
        }
        std::swap(ctx.scene.boneData,pendingBones);RigAuthoring::exchangeSelection(ctx.scene.rigView,pendingSelection);
        ProjectManager::getInstance().markModified();
    }
    Type getType()const override{return Type::Generic;}
    std::string getDescription()const override{return "Edit rig: "+character;}
};
rtapi::Result apply(const std::function<bool(RigAuthoring::RigEditState&,std::string&)>& stage,bool created) {
    if(!rtapi::g_ctx)return rtapi::Result::fail("api_not_bound");
    if(rtapi::renderJobActive())return rtapi::Result::fail("scene_locked");
    if(!rtapi::g_history)return rtapi::Result::fail("history_not_bound");
    try {
        RigAuthoring::RigEditState state;std::string error;
        if(!stage(state,error))return rtapi::Result::fail(error);
        auto command=std::make_unique<RigEditCommand>(std::move(state),rtapi::g_ctx->scene.rigView,created);
        command->prepare(*rtapi::g_ctx);auto* recorded=command.get();
        rtapi::g_history->record(std::move(command));recorded->execute(*rtapi::g_ctx);
        return rtapi::Result::success();
    }catch(const std::exception&){return rtapi::Result::fail("rig_edit_failed");}
}
}
namespace rtapi {
Result getNextRigName(const std::string& seed,std::string& name) {
    name.clear();if(!g_ctx)return Result::fail("api_not_bound");
    if(renderJobActive())return Result::fail("scene_locked");
    std::string error;return RigAuthoring::nextRigName(g_ctx->scene,seed,name,error)?Result::success():Result::fail(error);
}
Result getRigSceneTransform(const std::string& character,Matrix4x4& matrix) {
    matrix=Matrix4x4::identity();if(!g_ctx)return Result::fail("api_not_bound");
    if(renderJobActive())return Result::fail("scene_locked");
    for(const auto& model:g_ctx->scene.importedModelContexts)if(model.importName==character) {
        if(!model.authoringOwned || !model.members.empty() || model.weightedBoneCount)return Result::fail("rig_placement_requires_meshless_unskinned");
        matrix=model.rigSceneTransform;return Result::success();
    }
    return Result::fail("unknown_character");
}
Result setRigSceneTransform(const std::string& character,const Matrix4x4& matrix) {
    return apply([&](RigAuthoring::RigEditState& out,std::string& error){return RigAuthoring::stageRigPlacement(g_ctx->scene,character,matrix,out,error);},false);
}
Result getNextRigBoneName(const std::string& character,const std::string& seed,std::string& name) {
    name.clear();if(!g_ctx)return Result::fail("api_not_bound");
    if(renderJobActive())return Result::fail("scene_locked");
    std::string error;
    return RigAuthoring::nextRigBoneName(g_ctx->scene,character,seed,name,error)?Result::success():Result::fail(error);
}
Result renameRigBone(const std::string& character,const std::string& bone,const std::string& name) {
    return apply([&](RigAuthoring::RigEditState& out,std::string& error){return RigAuthoring::stageRenameRigBone(g_ctx->scene,character,bone,name,out,error);},false);
}
Result reparentRigBone(const std::string& character,const std::string& bone,const std::string& parent) {
    return apply([&](RigAuthoring::RigEditState& out,std::string& error){return RigAuthoring::stageReparentRigBone(g_ctx->scene,character,bone,parent,out,error);},false);
}
Result deleteRigBone(const std::string& character,const std::string& bone) {
    return apply([&](RigAuthoring::RigEditState& out,std::string& error){return RigAuthoring::stageDeleteRigBone(g_ctx->scene,character,bone,out,error);},false);
}
Result copyRigFrom(const std::string& sourceCharacter,const std::string& character,std::vector<RigAuthoring::RigCopyBone>& mapping) {
    mapping.clear();std::vector<RigAuthoring::RigCopyBone> staged;
    const auto result=apply([&](RigAuthoring::RigEditState& out,std::string& error){return RigAuthoring::stageCopyRig(g_ctx->scene,sourceCharacter,character,out,staged,error);},true);
    if(result.ok)mapping=std::move(staged);
    return result;
}
Result setRigAnatomy(const std::string& character,const nlohmann::json& anatomy) {
    return apply([&](RigAuthoring::RigEditState& out,std::string& error){return RigAuthoring::stageSetRigAnatomy(g_ctx->scene,character,anatomy,out,error);},false);
}
Result commitRigFit(const std::string& character,const std::string& mesh,const nlohmann::json& preview) {
    return apply([&](RigAuthoring::RigEditState& out,std::string& error){return RigAuthoring::stageCommitRigFit(g_ctx->scene,character,mesh,preview,out,error);},false);
}
Result createRig(const std::string& character,const std::string& id,float height) {
    return apply([&](RigAuthoring::RigEditState& out,std::string& error){return RigAuthoring::stageCreateRig(g_ctx->scene,character,id,height,out,error);},true);
}
Result addRigBone(const std::string& character,const std::string& name,const std::string& parent,const Matrix4x4& rest) {
    return apply([&](RigAuthoring::RigEditState& out,std::string& error){return RigAuthoring::stageAddRigBone(g_ctx->scene,character,name,parent,rest,out,error);},false);
}
Result mirrorRigRest(const std::string& character,const std::vector<std::string>& bones,const std::string& direction,const std::string& axis,float offset,uint64_t revision) {
    return apply([&](RigAuthoring::RigEditState& out,std::string& error){return RigAuthoring::stageMirrorRigRest(g_ctx->scene,character,bones,direction,{axis,offset},revision,out,error);},false);
}
Result createMirroredRigBone(const std::string& character,const std::string& bone,const std::string& name,const std::string& sourceSide,const std::string& axis,float offset,uint64_t revision) {
    return apply([&](RigAuthoring::RigEditState& out,std::string& error){return RigAuthoring::stageCreateMirrorBone(g_ctx->scene,character,bone,name,sourceSide,{axis,offset},revision,out,error);},false);
}
Result transformRigRest(const std::string& character,const std::vector<std::string>& bones,const Matrix4x4& delta,uint64_t revision) {
    return apply([&](RigAuthoring::RigEditState& out,std::string& error){return RigAuthoring::stageBatchRigRest(g_ctx->scene,character,bones,delta,revision,out,error);},false);
}
Result setRigRestTransform(const std::string& character,const std::string& bone,const Matrix4x4& rest) {
    return apply([&](RigAuthoring::RigEditState& out,std::string& error){return RigAuthoring::stageRigRestEdit(g_ctx->scene,character,bone,rest,out,error);},false);
}
}
