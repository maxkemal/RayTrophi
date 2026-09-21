#include "Animation/RigIKTimeline.h"
#include "Animation/RigPoseAuthoringMath.h"
#include <cmath>
#include <cstdint>
namespace RigAuthoring {
const AnimationData* authoredIKClip(const SceneData& scene,const std::string& character){const auto key=scene.rigView.pose.clips.find(character);if(key==scene.rigView.pose.clips.end())return nullptr;for(const auto& c:scene.animationDataList)if(c&&c->rigAuthoring&&c->modelName==character&&c->name==key->second)return c.get();return nullptr;}
bool validateIKClipChannels(const SceneData& scene,const AnimationData& clip,std::string& error){
 if(clip.ikChannels.empty()){error.clear();return true;}if(!clip.rigAuthoring||!std::isfinite(clip.duration)||clip.duration<=0||!std::isfinite(clip.ticksPerSecond)||clip.ticksPerSecond<=0){error="invalid_clip_timing";return false;}
 for(const auto& m:scene.importedModelContexts)if(m.importName==clip.modelName){if(!validateIKChannelControls(clip.ikChannels,m.rigAnatomy.controls,error))return false;const double duration=clip.duration/clip.ticksPerSecond;for(const auto& entry:clip.ikChannels){for(const auto& k:entry.second.keys)if(k.seconds>=duration){error="rig_ik_key_outside_clip";return false;}for(const auto& k:entry.second.contacts)if(k.end>duration){error="rig_ik_contact_outside_clip";return false;}}return true;}error="unknown_character";return false;
}
bool effectiveIKControls(const SceneData& scene,const std::string& character,bool preview,IKPoses& output,std::string& error){
 output.clear();const auto& s=scene.rigView.pose;const auto* clip=authoredIKClip(scene,character);
 if(clip&&!clip->ikChannels.empty()){if(!std::isfinite(clip->duration)||clip->duration<=0||!std::isfinite(clip->ticksPerSecond)||clip->ticksPerSecond<=0||!std::isfinite(s.fps)||s.fps<=0){error="invalid_clip_timing";return false;}if(!validateIKClipChannels(scene,*clip,error))return false;const double seconds=double(scene.timeline.current_frame)/s.fps;output=sampleIKChannels(clip->ikChannels,std::fmod(seconds,clip->duration/clip->ticksPerSecond));}
 if(s.active&&s.character==character){const bool editing=s.frame==scene.timeline.current_frame;const auto& live=preview&&editing&&s.hasPreview?s.previewIK:s.ik;for(const auto& p:live)if(editing||p.second.contact)output[p.first]=p.second;}
 return true;
}
bool bakeIKChannels(const SceneData& scene,const SceneData::ImportedModelContext& model,const AnimationData& source,int start,int end,AnimationData& output,std::string& error){
 const double fps=scene.rigView.pose.fps;const auto samples=static_cast<int64_t>(end)-start+1;if(start<0||end<start||end>1000000||samples>1001||samples*static_cast<int64_t>(model.nodeHierarchy.size())>100000){error="rig_ik_bake_limit";return false;}if(!std::isfinite(fps)||fps<=0||!std::isfinite(source.duration)||source.duration<=0||!std::isfinite(source.ticksPerSecond)||source.ticksPerSecond<=0){error="invalid_clip_timing";return false;}if(!validateIKChannelControls(source.ikChannels,model.rigAnatomy.controls,error))return false;
 Matrix4x4 placement;if(!rigScenePlacement(scene,model.importName,placement,error))return false;auto staged=source;staged.ikChannels.clear();std::vector<std::string> bones;for(const auto& n:model.nodeHierarchy.nodes)bones.push_back(n.uniqueName);
 for(int frame=start;frame<=end;++frame){const double seconds=double(frame)/fps;RayTrophi::NodeHierarchy base,solved,limited;std::vector<std::string> hits;if(!poseHierarchy(model.nodeHierarchy,&source,seconds,{},base,error))return false;const auto controls=sampleIKChannels(source.ikChannels,std::fmod(seconds,source.duration/source.ticksPerSecond));if(!solveIKPose(base,model.rigAnatomy.controls,controls,placement,solved,error)||!constrainJointPose(model.nodeHierarchy,model.rigAnatomy.joints,solved,limited,hits,error)||!insertPoseKeys(staged,limited,bones,seconds,error))return false;}
 output=std::move(staged);return true;
}
}
