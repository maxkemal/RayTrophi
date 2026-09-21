#pragma once
#include "Matrix4x4.h"
#include "Animation/RigIK.h"
#include <unordered_map>
#include <string>
#include <vector>
#include <cstdint>
namespace RigAuthoring {
struct PoseAuthoringState {
 bool active=false,autoKey=false,hasPreview=false;
 std::string character;
 std::vector<std::string> limitHits;
 std::unordered_map<std::string,std::string> clips;
 std::unordered_map<std::string,Matrix4x4> locals,preview;
 IKPoses ik,previewIK;
 std::string control,controlHandle="target";
 Matrix4x4 evaluatedPlacement=Matrix4x4::identity();bool placementAcknowledged=false;
 int frame=0;float fps=24;uint64_t revision=0,serial=0;
 // Interaction serial cancels stale gestures; evaluation dirtiness must not.
 bool evaluationDirty=true;int evaluatedFrame=-1;
 bool needsEvaluation(int currentFrame)const{return active&&(evaluationDirty||evaluatedFrame!=currentFrame);}
 void invalidateEvaluation(){evaluationDirty=true;}
 void acknowledgeEvaluation(int currentFrame){evaluatedFrame=currentFrame;evaluationDirty=false;}
};
}
