#pragma once
#include <string>
#include "Matrix4x4.h"
#include "Animation/RigView.h"
#include "Api/RtApiRigAnatomy.h"
#include "Api/RtApiRigTemplates.h"
#include "Api/RtApiRigBinding.h"
#include "Api/RtApiRigWeightMap.h"
#include "Api/RtApiRigSelection.h"
#include "Api/RtApiRigMirror.h"
#include "Api/RtApiRigPoseAuthoring.h"
#include "Api/RtApiRigIK.h"
#include "Api/RtApiRigJointProfile.h"
namespace rtapi {
struct Result;
Result getNextRigName(const std::string& seed, std::string& name);
Result getRigSceneTransform(const std::string& character, Matrix4x4& matrix);
Result setRigSceneTransform(const std::string& character, const Matrix4x4& matrix);
Result getNextRigBoneName(const std::string& character, const std::string& seed, std::string& name);
Result renameRigBone(const std::string& character,const std::string& bone,const std::string& name);
Result reparentRigBone(const std::string& character,const std::string& bone,const std::string& parent);
Result deleteRigBone(const std::string& character,const std::string& bone);
Result copyRigFrom(const std::string& sourceCharacter,const std::string& character,std::vector<RigAuthoring::RigCopyBone>& mapping);
Result createRig(const std::string& character,const std::string& templateId="root",float height=1.8f);
Result addRigBone(const std::string& character,const std::string& name,const std::string& parent,const Matrix4x4& localRest);
Result transformRigRest(const std::string& character,const std::vector<std::string>& bones,const Matrix4x4& worldDelta,uint64_t rigRevision);
Result setRigRestTransform(const std::string& character,const std::string& bone,const Matrix4x4& localRest);
}
