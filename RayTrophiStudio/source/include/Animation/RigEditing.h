#pragma once
#include "Animation/RigView.h"
#include "Animation/RigMirror.h"
#include "scene_data.h"
namespace RigAuthoring {
struct RigEditState {
    BoneData bones;
    SceneData::ImportedModelContext model;
    std::string selectedBone;
    std::vector<std::string> selectedBones;
};
bool stageMirrorRigRest(const SceneData&,const std::string& character,const std::vector<std::string>& sources,const std::string& direction,const MirrorPlane&,uint64_t revision,RigEditState&,std::string& error);
bool stageCreateMirrorBone(const SceneData&,const std::string& character,const std::string& source,const std::string& name,const std::string& sourceSide,const MirrorPlane&,uint64_t revision,RigEditState&,std::string& error);
bool stageBatchRigRest(const SceneData&,const std::string& character,const std::vector<std::string>& bones,const Matrix4x4& worldDelta,uint64_t expectedRevision,RigEditState&,std::string& error);
bool nextRigName(const SceneData&, const std::string& seed, std::string& name, std::string& error);
bool canPlaceRig(const SceneData&, const std::string& character, std::string& error);
bool stageRigPlacement(const SceneData&, const std::string& character, const Matrix4x4&, RigEditState&, std::string& error);
nlohmann::json serializeRigPlacement(const Matrix4x4&);
bool deserializeRigPlacement(const nlohmann::json&, Matrix4x4&, std::string& error);
bool nextRigBoneName(const SceneData&, const std::string& character, const std::string& seed, std::string& name, std::string& error);
bool canEditRig(const SceneData&, const std::string& character, std::string& error);
bool setInteractionMode(SceneData&, const std::string& mode, const std::string& character, std::string& error);
// NodeHierarchy is canonical rest/topology state for owned, unweighted rigs.
bool stageCopyRig(const SceneData&, const std::string& sourceCharacter, const std::string& character,
                  RigEditState&, std::vector<RigCopyBone>& mapping, std::string& error);
bool stageSetRigAnatomy(const SceneData&, const std::string& character, const nlohmann::json& anatomy,
                        RigEditState&, std::string& error);
bool stageCommitRigFit(const SceneData&,const std::string& character,const std::string& mesh,const nlohmann::json& preview,RigEditState&,std::string& error);
bool stageCreateRig(const SceneData&, const std::string& character, const std::string& templateId,
                    float height, RigEditState&, std::string& error);
bool stageAddRigBone(const SceneData&, const std::string& character, const std::string& name,
                     const std::string& parent, const Matrix4x4& localRest, RigEditState&, std::string& error);
bool stageRigRestEdit(const SceneData&, const std::string& character, const std::string& bone,
                      const Matrix4x4& localRest, RigEditState&, std::string& error);
bool stageRenameRigBone(const SceneData&, const std::string& character, const std::string& bone,
                        const std::string& name, RigEditState&, std::string& error);
bool stageReparentRigBone(const SceneData&, const std::string& character, const std::string& bone,
                          const std::string& parent, RigEditState&, std::string& error);
bool stageDeleteRigBone(const SceneData&, const std::string& character, const std::string& bone,
                        RigEditState&, std::string& error);
bool restoreOwnedRigRuntime(SceneData&, std::string& error);
}
