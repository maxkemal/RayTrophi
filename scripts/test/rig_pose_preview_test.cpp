#include "Animation/RigAnatomy.h"
#include "Animation/RigControlDisplay.h"
#include "Animation/RigJointRules.h"
#include "Animation/RigIKChannels.h"
#include "Animation/RigPoseAuthoringMath.h"
#include "Animation/RigBoneCurves.h"
#include "Animation/RigPosePreview.h"
#include "Animation/RigTimelineProjection.h"
#include "Animation/RigDrivenControls.h"
#include "Animation/RigBindMath.h"
#include "Animation/RigBatchRest.h"
#include "Animation/RigMirror.h"
#include "Animation/RigMeshSpace.h"
#include "KeyframeSystem.h"
#include <cassert>
#include <cmath>
#include <limits>
#include <utility>
int main() {
    {
        RayTrophi::NodeHierarchy fingers;
        fingers.addNode("Hand", "Rig_LeftHand", Matrix4x4::identity(), -1);
        fingers.addNode("Finger1", "Rig_Finger1",
                        Matrix4x4::translation(Vec3(.1f, 0, 0)), 0);
        fingers.addNode("Finger2", "Rig_Finger2",
                        Matrix4x4::translation(Vec3(.1f, 0, 0)), 1);
        RigAuthoring::RigDrivenControl curl;
        curl.id = "gripper.curl";
        curl.label = "Curl";
        curl.group = "gripper";
        curl.anchor = "Rig_LeftHand";
        curl.side = "center";
        curl.shape = "hand";
        curl.drivers = {{"Rig_Finger1", Vec3(0, 0, 1), -60.f},
                        {"Rig_Finger2", Vec3(0, 0, 1), -80.f}};
        RayTrophi::NodeHierarchy posed;
        std::vector<std::string> affected;
        std::string fingerError;
        assert(RigAuthoring::evaluateRigDrivenControls(
            fingers, {curl}, {{"gripper.curl", 1.f}}, posed, affected, fingerError));
        assert(affected.size() == 2 && posed.size() == fingers.size());
        assert(std::fabs(posed.nodes[1].localBind.m[0][0] -
                         fingers.nodes[1].localBind.m[0][0]) > 1e-4f);
        assert(fingers.nodes[2].localBind.m[0][0] == 1.f);
        const auto unchanged = posed;
        assert(!RigAuthoring::evaluateRigDrivenControls(
            fingers, {curl}, {{"missing", 1.f}}, posed, affected, fingerError));
        assert(fingerError == "rig_control_unknown" && posed.size() == unchanged.size());
        assert(!RigAuthoring::evaluateRigDrivenControls(
            fingers, {curl}, {{"gripper.curl", 0.f}}, posed, affected, fingerError));
        assert(fingerError == "rig_edit_no_change");
        assert(!RigAuthoring::evaluateRigDrivenControls(
            fingers, {curl}, {{"gripper.curl", 2.f}}, posed, affected, fingerError));
        assert(fingerError == "rig_control_value_out_of_range" && affected.empty());
        RigAuthoring::RigAnatomy controlAnatomy;
        controlAnatomy.drivenControls = {curl};
        assert(RigAuthoring::validateRigAnatomy(controlAnatomy, fingers, fingerError));
        const auto encodedControls = RigAuthoring::serializeRigAnatomy(controlAnatomy);
        assert(encodedControls["version"] == 7 && encodedControls["control_rig"].size() == 1);
        RigAuthoring::RigAnatomy decodedControls;
        assert(RigAuthoring::deserializeRigAnatomy(
            encodedControls, fingers, decodedControls, fingerError));
        assert(decodedControls.drivenControls[0].drivers.size() == 2);
        RigAuthoring::renameAnatomyBone(decodedControls, "Rig_Finger1", "RenamedFinger");
        assert(decodedControls.drivenControls[0].drivers[0].bone == "RenamedFinger");
        assert(RigAuthoring::anatomyReferencesBone(decodedControls, "RenamedFinger"));
        curl.drivers[0].bone = "Missing";
        assert(!RigAuthoring::evaluateRigDrivenControls(
            fingers, {curl}, {{"gripper.curl", 1.f}}, posed, affected, fingerError));
        assert(fingerError == "rig_control_unknown_bone" && affected.empty());
    }
    {
        RigAuthoring::RigAnatomy anatomy;
        anatomy.roles = {{"left_arm.hand", "Hand"}, {"head", "Head"}};
        RigAuthoring::IKControl hand{"left_arm", "UpperArm", "Forearm", "Hand"};
        const auto handDisplay = RigAuthoring::deriveRigControlDisplay(anatomy, hand);
        assert(handDisplay.semantic == "hand" && handDisplay.side == "left");
        assert(handDisplay.target.shape == "hand" && handDisplay.target.level == "primary");
        assert(handDisplay.pole.level == "secondary" && handDisplay.fk.channels[0] == "rotate");

        RigAuthoring::IKControl look{"head.aim", "Head", "Head", "Head"};
        look.solver = "aim";
        const auto lookDisplay = RigAuthoring::deriveRigControlDisplay(anatomy, look);
        assert(lookDisplay.semantic == "head_aim" && lookDisplay.target.shape == "aim");
        assert(lookDisplay.target.channels == std::vector<std::string>{"translate"});

        const auto contract = RigAuthoring::rigControlDisplayContract();
        assert(contract["version"] == 2 &&
               contract["space"] == "hybrid_anatomical_clamped" &&
               contract["scale_source"] == "length_world" &&
               contract["hit_space"] == "screen_constant_minimum");
        assert(contract["levels"] == nlohmann::json({"primary", "secondary", "deform"}));

        RayTrophi::NodeHierarchy limb;
        limb.addNode("Upper", "Upper", Matrix4x4::identity(), -1);
        limb.addNode("Mid", "Mid", Matrix4x4::translation(Vec3(1, 0, 0)), 0);
        limb.addNode("Tip", "Tip", Matrix4x4::translation(Vec3(1, 0, 0)), 1);
        RigAuthoring::IKControl limbControl{"limb", "Upper", "Mid", "Tip"};
        const auto inspected = RigAuthoring::inspectIKPose(
            limb, {limbControl}, {}, Matrix4x4::scaling(Vec3(2.f)));
        assert(inspected.size() == 1);
        assert(std::fabs(inspected[0]["length_actor"].get<float>() - 2.f) < 1e-5f);
        assert(std::fabs(inspected[0]["length_world"].get<float>() - 4.f) < 1e-5f);
    }
    {
        // Pose mirror preserves asymmetric rest and copies parent/child motion once.
        RayTrophi::NodeHierarchy rest;
        rest.addNode("Root","Root",Matrix4x4::rotationY(.3f),-1);
        rest.addNode("L","L",Matrix4x4::translation(Vec3(1,2,0))*Matrix4x4::rotationZ(.2f),0);
        rest.addNode("R","R",Matrix4x4::translation(Vec3(-2,3,0))*Matrix4x4::rotationZ(-.4f),0);
        rest.addNode("LC","LC",Matrix4x4::translation(Vec3(0,1,0)),1);
        rest.addNode("RC","RC",Matrix4x4::translation(Vec3(0,2,0)),2);
        RigAuthoring::RigAnatomy anatomy;anatomy.symmetry={{"L","R"},{"LC","RC"}};
        auto pose=rest;RayTrophi::NodeHierarchy mirrored,roundtrip;std::string mirrorError;
        assert(RigAuthoring::mirrorPose(rest,pose,anatomy,{"L","LC"},"selected","x",mirrored,mirrorError));
        for(size_t i=0;i<rest.size();++i)for(int r=0;r<4;++r)for(int c=0;c<4;++c)assert(std::fabs(rest.nodes[i].localBind.m[r][c]-mirrored.nodes[i].localBind.m[r][c])<1e-5f);
        pose.nodes[1].localBind=rest.nodes[1].localBind*Matrix4x4::rotationZ(.5f);
        pose.nodes[3].localBind=rest.nodes[3].localBind*Matrix4x4::translation(Vec3(.1f,.2f,.3f))*Matrix4x4::rotationX(.4f);
        assert(RigAuthoring::mirrorPose(rest,pose,anatomy,{},"left_to_right","x",mirrored,mirrorError));
        assert(RigAuthoring::mirrorPose(rest,mirrored,anatomy,{},"right_to_left","x",roundtrip,mirrorError));
        for(size_t i:{size_t(0),size_t(1),size_t(3)})for(int r=0;r<4;++r)for(int c=0;c<4;++c) {
            assert(std::fabs(pose.nodes[i].localBind.m[r][c]-mirrored.nodes[i].localBind.m[r][c])<1e-5f);
            assert(std::fabs(pose.nodes[i].localBind.m[r][c]-roundtrip.nodes[i].localBind.m[r][c])<1e-5f);
        }
        assert(!RigAuthoring::mirrorPose(rest,pose,anatomy,{"L","R"},"selected","x",mirrored,mirrorError)&&mirrorError=="rig_mirror_ambiguous_pair");
        assert(!RigAuthoring::mirrorPose(rest,pose,anatomy,{"Root"},"selected","x",mirrored,mirrorError)&&mirrorError=="rig_mirror_unpaired_bone");
        assert(!RigAuthoring::mirrorPose(rest,pose,anatomy,{"L"},"selected","bad",mirrored,mirrorError)&&mirrorError=="rig_mirror_invalid_plane");
        rest.nodes[0].localBind=Matrix4x4::identity();
        rest.nodes[1].localBind=Matrix4x4::translation(Vec3(1,2,0));
        rest.nodes[2].localBind=Matrix4x4::translation(Vec3(-2,3,0));
        pose=rest;pose.nodes[1].localBind=rest.nodes[1].localBind*Matrix4x4::translation(Vec3(.1f,.2f,.3f))*Matrix4x4::rotationZ(.5f);
        assert(RigAuthoring::mirrorPose(rest,pose,anatomy,{"L"},"selected","x",mirrored,mirrorError));
        const auto expected=rest.nodes[2].localBind*Matrix4x4::translation(Vec3(-.1f,.2f,.3f))*Matrix4x4::rotationZ(-.5f);
        for(int r=0;r<4;++r)for(int c=0;c<4;++c)assert(std::fabs(expected.m[r][c]-mirrored.nodes[2].localBind.m[r][c])<1e-5f);
    }
    {
        // Rules use the bone rest frame, not world Euler axes, and never mutate inputs.
        RayTrophi::NodeHierarchy rest;
        rest.addNode("RuleRoot","RuleRoot",Matrix4x4::identity(),-1);
        rest.addNode("RuleJoint","RuleJoint",Matrix4x4::translation(Vec3(0,2,0))*Matrix4x4::rotationY(.4f),0);
        RigAuthoring::JointRule hinge;hinge.bone="RuleJoint";hinge.type="hinge";hinge.enabled=true;hinge.axis=Vec3(0,0,1);hinge.minimum=-30;hinge.maximum=30;
        auto input=rest;input.nodes[1].localBind=Matrix4x4::translation(Vec3(3,2,0))*Matrix4x4::rotationY(.4f)*Matrix4x4::rotationZ(1.f);
        RayTrophi::NodeHierarchy constrained;std::vector<std::string> hits;std::string ruleError;
        assert(RigAuthoring::constrainJointPose(rest,{hinge},input,constrained,hits,ruleError));
        assert(hits==std::vector<std::string>{"RuleJoint"});
        const auto expected=rest.nodes[1].localBind*Matrix4x4::rotationZ(3.14159265359f/6);
        for(int r=0;r<4;++r)for(int c=0;c<4;++c)assert(std::fabs(constrained.nodes[1].localBind.m[r][c]-expected.m[r][c])<1e-5f);
        assert(input.nodes[1].localBind.m[0][3]==3 && rest.nodes[1].localBind.m[0][3]==0);
        assert(RigAuthoring::constrainJointPose(rest,{hinge},constrained,constrained,hits,ruleError)&&hits.empty());
        hinge.enabled=false;
        assert(RigAuthoring::constrainJointPose(rest,{hinge},input,constrained,hits,ruleError)&&hits.empty()&&constrained.nodes[1].localBind.m[0][3]==3);
        hinge.enabled=true;hinge.type="ball";hinge.swing=20;hinge.axis=Vec3(0,0,1);
        input=rest;input.nodes[1].localBind=rest.nodes[1].localBind*Matrix4x4::rotationX(1.f);
        assert(RigAuthoring::constrainJointPose(rest,{hinge},input,constrained,hits,ruleError)&&hits.size()==1);
        const auto ballExpected=rest.nodes[1].localBind*Matrix4x4::rotationX(20*3.14159265359f/180);
        for(int r=0;r<4;++r)for(int c=0;c<4;++c)assert(std::fabs(constrained.nodes[1].localBind.m[r][c]-ballExpected.m[r][c])<1e-5f);
        hinge.type="fixed";
        assert(RigAuthoring::constrainJointPose(rest,{hinge},input,constrained,hits,ruleError)&&hits.size()==1);
        for(int r=0;r<4;++r)for(int c=0;c<4;++c)assert(std::fabs(constrained.nodes[1].localBind.m[r][c]-rest.nodes[1].localBind.m[r][c])<1e-5f);
        std::vector<RigAuthoring::JointRule> restored;
        assert(RigAuthoring::deserializeJointRules(RigAuthoring::serializeJointRules({hinge}),rest,restored,ruleError)&&restored.size()==1);
        auto invalid=RigAuthoring::serializeJointRules({hinge});invalid["joints"][0]["axis"]={0,0,0};
        assert(!RigAuthoring::deserializeJointRules(invalid,rest,restored,ruleError)&&ruleError=="rig_joint_invalid_axis"&&restored.size()==1);
        assert(!RigAuthoring::validateJointRules({hinge,hinge},rest,ruleError)&&ruleError=="rig_joint_duplicate_bone");
        RigAuthoring::RigAnatomy anatomy;anatomy.joints={hinge};
        const auto saved=RigAuthoring::serializeRigAnatomy(anatomy);assert(saved["version"]==2);
        RigAuthoring::RigAnatomy loaded;assert(RigAuthoring::deserializeRigAnatomy(saved,rest,loaded,ruleError)&&loaded.joints.size()==1);
        assert(RigAuthoring::serializeRigAnatomy(RigAuthoring::RigAnatomy())["version"]==1);
        RigAuthoring::renameAnatomyBone(loaded,"RuleJoint","Renamed");assert(loaded.joints[0].bone=="Renamed"&&RigAuthoring::anatomyReferencesBone(loaded,"Renamed"));
    }
    {
        RigAuthoring::PoseAuthoringState state;
        assert(!state.needsEvaluation(0));
        state.active=true;assert(state.needsEvaluation(0));
        const auto gestureSerial=state.serial;
        state.acknowledgeEvaluation(0);assert(!state.needsEvaluation(0));
        assert(state.needsEvaluation(1));
        state.invalidateEvaluation();assert(state.needsEvaluation(0));
        assert(state.serial==gestureSerial); // Preview redraw cannot cancel a drag.
        state.acknowledgeEvaluation(0);assert(!state.needsEvaluation(0));
        state.active=false;assert(!state.needsEvaluation(1));
    }
    {
        // Authored channels are atomic, rigid and sampled in seconds through clip ticks.
        std::string poseError;
        RayTrophi::NodeHierarchy authoredRest;
        authoredRest.addNode("PoseRoot","PoseRoot",Matrix4x4::identity(),-1);
        authoredRest.addNode("PoseChild","PoseChild",Matrix4x4::translation(Vec3(0,2,0)),0);
        AnimationData authored;authored.rigAuthoring=true;authored.duration=1;authored.ticksPerSecond=24;
        assert(RigAuthoring::insertPoseKeys(authored,authoredRest,{"PoseRoot","PoseChild"},0,poseError));
        auto keyed=authoredRest;keyed.nodes[0].localBind=Matrix4x4::rotationZ(1.f);
        assert(RigAuthoring::insertPoseKeys(authored,keyed,{"PoseRoot"},1,poseError));
        assert(authored.rotationKeys["PoseRoot"].size()==2 && authored.duration==25);
        RayTrophi::NodeHierarchy evaluated;
        assert(RigAuthoring::poseHierarchy(authoredRest,&authored,.5,{},evaluated,poseError));
        std::vector<RigAuthoring::PreviewJoint> poseJoints;
        assert(RigAuthoring::sampleRigPose(evaluated,nullptr,0,poseJoints,poseError));
        const auto expectedChild=Matrix4x4::rotationZ(.5f).transform_point(Vec3(0,2,0));
        assert((Vec3(poseJoints[1].world.m[0][3],poseJoints[1].world.m[1][3],0)-expectedChild).length_squared()<1e-6f);
        assert(authoredRest.nodes[0].localBind.m[0][0]==1);
        keyed.nodes[0].localBind=Matrix4x4::rotationZ(.8f);
        assert(RigAuthoring::insertPoseKeys(authored,keyed,{"PoseRoot"},1,poseError));
        assert(authored.rotationKeys["PoseRoot"].size()==2);
        assert(RigAuthoring::poseHierarchy(authoredRest,&authored,2.0,{},evaluated,poseError));
        const auto held=Matrix4x4::rotationZ(.8f);
        for(int r=0;r<4;++r)for(int c=0;c<4;++c)
            assert(std::fabs(evaluated.nodes[0].localBind.m[r][c]-held.m[r][c])<1e-5f);
        const auto unchanged=authored.rotationKeys["PoseRoot"][1].value;
        keyed.nodes[0].localBind=Matrix4x4::scaling(Vec3(2,1,1));
        assert(!RigAuthoring::insertPoseKeys(authored,keyed,{"PoseChild","PoseRoot"},2,poseError));
        assert(poseError=="rig_pose_requires_rigid_transform" && authored.rotationKeys["PoseChild"].size()==1);
        assert(authored.rotationKeys["PoseRoot"][1].value.w==unchanged.w && authored.duration==25);
        assert(!RigAuthoring::insertPoseKeys(authored,authoredRest,{"PoseRoot","PoseRoot"},2,poseError));
        assert(poseError=="rig_selection_duplicate_bone");
        assert(RigAuthoring::removePoseKeys(authored,authoredRest,{"PoseRoot"},1,poseError));
        assert(authored.rotationKeys["PoseRoot"].size()==1);
        const auto remainingRotation=authored.rotationKeys["PoseRoot"][0].value;
        assert(!RigAuthoring::removePoseKeys(authored,authoredRest,{"PoseRoot"},1,poseError));
        assert(poseError=="rig_edit_no_change" && authored.rotationKeys["PoseRoot"].size()==1);
        assert(authored.rotationKeys["PoseRoot"][0].value.w==remainingRotation.w);
        const Vec3 movedPosition(3,4,5);
        assert(RigAuthoring::editBoneCurveKey(
            authored,"PoseChild",RigAuthoring::BoneCurveChannel::Position,
            0,0.5,&movedPosition,nullptr,poseError));
        assert(authored.positionKeys["PoseChild"].size()==1);
        assert(authored.positionKeys["PoseChild"][0].time==12);
        assert(authored.positionKeys["PoseChild"][0].value.y==4);
        assert(RigAuthoring::editBoneCurveKey(
            authored,"PoseChild",RigAuthoring::BoneCurveChannel::Rotation,
            0,0.5,nullptr,nullptr,poseError));
        assert(authored.rotationKeys["PoseChild"][0].time==12);
        const auto curveSnapshot=authored;
        assert(!RigAuthoring::editBoneCurveKey(
            authored,"PoseChild",RigAuthoring::BoneCurveChannel::Position,
            0,1,nullptr,nullptr,poseError));
        assert(poseError=="rig_curve_key_not_found");
        assert(authored.positionKeys["PoseChild"][0].time==
               curveSnapshot.positionKeys.at("PoseChild")[0].time);
        auto conflictKey=authored.positionKeys["PoseChild"][0];
        conflictKey.time=24;
        authored.positionKeys["PoseChild"].push_back(conflictKey);
        const auto conflictSnapshot=authored;
        assert(!RigAuthoring::editBoneCurveKey(
            authored,"PoseChild",RigAuthoring::BoneCurveChannel::Position,
            .5,1,nullptr,nullptr,poseError));
        assert(poseError=="rig_curve_key_conflict");
        assert(authored.positionKeys["PoseChild"][0].time==
               conflictSnapshot.positionKeys.at("PoseChild")[0].time);
        assert(authored.positionKeys["PoseChild"][1].time==
               conflictSnapshot.positionKeys.at("PoseChild")[1].time);

        ObjectAnimationTrack projected;
        Keyframe sharedKey(0);
        sharedKey.has_transform=true;
        sharedKey.transform.has_position=true;
        sharedKey.transform.has_rotation=true;
        sharedKey.transform.has_scale=false;
        sharedKey.transform.position=Vec3(1,2,3);
        Keyframe occupiedPosition(10);
        occupiedPosition.has_transform=true;
        occupiedPosition.transform.has_position=true;
        occupiedPosition.transform.has_rotation=false;
        occupiedPosition.transform.has_scale=false;
        projected.addKeyframe(sharedKey);
        projected.addKeyframe(occupiedPosition);
        assert(RigAuthoring::previewRigTimelineCurveDrag(projected,3,0,10,0));
        Vec3 previewPosition;
        assert(RigAuthoring::readRigTimelineCurvePreview(
            projected,3,10,previewPosition));
        assert(RigAuthoring::readRigTimelineCurvePreview(
            projected,0,0,previewPosition));
        assert(previewPosition.x==1);
        assert(!RigAuthoring::previewRigTimelineCurveDrag(projected,0,0,10,9));
        assert(RigAuthoring::previewRigTimelineCurveDrag(projected,0,0,0,9));
        assert(RigAuthoring::readRigTimelineCurvePreview(
            projected,0,0,previewPosition));
        assert(previewPosition.x==9);
        assert(!RigAuthoring::poseHierarchy(authoredRest,nullptr,0,{{"Missing",Matrix4x4::identity()}},evaluated,poseError));
        assert(poseError=="unknown_bone");
        authored.rigAuthoring=false;
        assert(!RigAuthoring::insertPoseKeys(authored,authoredRest,{"PoseRoot"},0,poseError));
        assert(poseError=="rig_pose_clip_not_editable");

        RigAuthoring::IKChannels ikChannels;
        RigAuthoring::IKPose ikPose;
        assert(RigAuthoring::insertIKChannelKey(ikChannels,"hand",.5,ikPose,poseError));
        assert(RigAuthoring::removeIKChannelKey(ikChannels,"hand",.5,poseError));
        assert(ikChannels.empty());
        assert(!RigAuthoring::removeIKChannelKey(ikChannels,"hand",.5,poseError));
        assert(poseError=="rig_edit_no_change");

    RayTrophi::NodeHierarchy h;
        h.addNode("Root","Root",Matrix4x4::translation(Vec3(3,2,0)),-1);
        h.addNode("L","L",Matrix4x4::translation(Vec3(2,1,0))*Matrix4x4::rotationZ(.3f),0);
        h.addNode("R","R",Matrix4x4::translation(Vec3(-2,1,0)),0);
        h.addNode("LC","LC",Matrix4x4::translation(Vec3(1,0,0)),1);
        h.addNode("RC","RC",Matrix4x4::translation(Vec3(-.5f,0,0)),2);
        RigAuthoring::RigAnatomy a;a.symmetry={{"L","R"},{"LC","RC"}};
        RayTrophi::NodeHierarchy reflected;std::vector<std::string> targets;std::string error;
        assert(RigAuthoring::mirrorRest(h,a,{"L","LC"},"selected",{"x",3},reflected,targets,error));
        std::vector<RigAuthoring::PreviewJoint> oldPose,newPose;
        assert(RigAuthoring::sampleRigPose(h,nullptr,0,oldPose,error));
        assert(RigAuthoring::sampleRigPose(reflected,nullptr,0,newPose,error));
        for(auto pair:{std::pair<int,int>{1,2},std::pair<int,int>{3,4}}) {
            assert(std::fabs(newPose[pair.second].world.m[0][3]-(6-oldPose[pair.first].world.m[0][3]))<1e-5f);
            assert(std::fabs(newPose[pair.second].world.m[1][3]-oldPose[pair.first].world.m[1][3])<1e-5f);
            const auto& m=newPose[pair.second].world;
            assert(m.m[0][0]*m.m[1][1]-m.m[0][1]*m.m[1][0]>.999f);
        }
        assert(h.nodes[2].localBind.m[0][3]==-2 && targets.size()==2);
        assert(!RigAuthoring::mirrorRest(h,a,{"L","R"},"selected",{"x",3},reflected,targets,error) && error=="rig_mirror_ambiguous_pair");
        assert(!RigAuthoring::mirrorRest(h,a,{"Root"},"selected",{"x",3},reflected,targets,error) && error=="rig_mirror_unpaired_bone");
        assert(!RigAuthoring::mirrorRest(h,a,{"L","L"},"selected",{"x",3},reflected,targets,error) && error=="rig_selection_duplicate_bone");
        assert(!RigAuthoring::mirrorRest(h,a,{"L"},"selected",{"w",3},reflected,targets,error) && error=="rig_mirror_invalid_plane");
        nlohmann::json marks=nlohmann::json::object();
        const auto actor=Matrix4x4::translation(Vec3(10,20,0))*Matrix4x4::rotationZ(.5f)*Matrix4x4::scaling(Vec3(.0001f));
        for(const auto& j:oldPose){const auto p=actor.transform_point(Vec3(j.world.m[0][3],j.world.m[1][3],j.world.m[2][3]));marks[j.name]={p.x,p.y,p.z};}
        const auto snapshot=marks; nlohmann::json mirrored;
        assert(RigAuthoring::mirrorLandmarks(h,a,actor,marks,{"L"},"selected",{"x",3},mirrored,error));
        const auto desired=actor.transform_point(Vec3(1,3,0));
        assert(std::fabs(mirrored["R"][0].get<float>()-desired.x)<1e-4f);
        assert(std::fabs(mirrored["R"][1].get<float>()-desired.y)<1e-4f && marks==snapshot);
        marks["L"][0]=std::numeric_limits<float>::infinity();
        assert(!RigAuthoring::mirrorLandmarks(h,a,actor,marks,{"L"},"selected",{"x",3},mirrored,error) && error=="rig_fit_invalid_landmark");
        RigAuthoring::RigAnatomy create=a;create.symmetry.pop_back();
        assert(RigAuthoring::createMirrorBone(h,create,"LC","NewRC","NewRC","left",{"x",3},reflected,a,error));
        assert(reflected.nodes.back().parent==2 && a.symmetry.back().right=="NewRC");
        assert(RigAuthoring::sampleRigPose(reflected,nullptr,0,newPose,error));
        assert(std::fabs(newPose.back().world.m[0][3]-(6-oldPose[3].world.m[0][3]))<1e-5f);
        assert(!RigAuthoring::createMirrorBone(h,create,"L","DuplicatePair","DuplicatePair","left",{"x",3},reflected,a,error) && error=="rig_mirror_already_paired");
    }

    // Imported static P may already be world-baked. Match viewport local P_orig
    // plus placement once, even if P is stale after a subsequent object move.
    DNA::GeometryDetail imported;
    imported.resize_vertices(2);imported.add_attribute<Vec3>("P");imported.add_attribute<Vec3>("P_orig");
    auto* local=imported.get_attribute_data_mut<Vec3>("P_orig");
    local[0]=Vec3(0,0,0);local[1]=Vec3(0,0,2);
    const auto placement=Matrix4x4::translation(Vec3(3,4,5))*Matrix4x4::rotationX(-3.14159265f*.5f);
    auto* baked=imported.get_positions_mut();
    baked[0]=placement.transform_point(local[0]);baked[1]=placement.transform_point(local[1]);
    assert(RigAuthoring::meshSpacePositionCount(imported)==2);
    const auto fitted=placement.transform_point(RigAuthoring::meshSpacePositions(imported)[1]);
    assert(std::fabs(fitted.x-3)<1e-5f && std::fabs(fitted.y-6)<1e-5f && std::fabs(fitted.z-5)<1e-5f);
    const auto twice=placement.transform_point(baked[1]);
    assert(std::fabs(twice.y-fitted.y)>1.f);
    baked[1]=Vec3(900,800,700);
    assert(std::fabs(placement.transform_point(RigAuthoring::meshSpacePositions(imported)[1]).y-6)<1e-5f);
    assert(RigAuthoring::meshSpacePositions(imported,true)[1].x==900);
    DNA::GeometryDetail primitive;primitive.resize_vertices(1);primitive.add_attribute<Vec3>("P");primitive.get_positions_mut()[0]=Vec3(1,2,3);
    assert(RigAuthoring::meshSpacePositions(primitive)[0].y==2 && RigAuthoring::meshSpacePositionCount(primitive)==1);
    // Selected ancestor+descendant get exactly one rigid world delta.
    RayTrophi::NodeHierarchy chain;
    chain.addNode("Root","Root",Matrix4x4::translation(Vec3(10,0,0)),-1);
    chain.addNode("Child","Child",Matrix4x4::translation(Vec3(0,2,0)),0);
    chain.addNode("Tip","Tip",Matrix4x4::translation(Vec3(0,3,0)),1);
    RayTrophi::NodeHierarchy moved;std::string batchError;std::vector<RigAuthoring::PreviewJoint> movedPose;
    assert(RigAuthoring::transformRestHierarchy(chain,Matrix4x4::identity(),{"Root","Child"},Matrix4x4::translation(Vec3(1,0,0)),moved,batchError));
    assert(RigAuthoring::sampleRigPose(moved,nullptr,0,movedPose,batchError));
    for(const auto& joint:movedPose)assert(std::fabs(joint.world.m[0][3]-11)<1e-5f);
    assert(std::fabs(moved.nodes[1].localBind.m[0][3])<1e-5f && chain.nodes[0].localBind.m[0][3]==10);
    assert(RigAuthoring::transformRestHierarchy(chain,Matrix4x4::identity(),{"Child"},Matrix4x4::translation(Vec3(1,0,0)),moved,batchError));
    assert(RigAuthoring::sampleRigPose(moved,nullptr,0,movedPose,batchError));
    assert(movedPose[0].world.m[0][3]==10 && movedPose[1].world.m[0][3]==11 && movedPose[2].world.m[0][3]==11);
    const auto aroundRoot=Matrix4x4::translation(Vec3(10,0,0))*Matrix4x4::rotationZ(3.14159265f*.5f)*Matrix4x4::translation(Vec3(-10,0,0));
    assert(RigAuthoring::transformRestHierarchy(chain,Matrix4x4::identity(),{"Root","Child"},aroundRoot,moved,batchError));
    assert(RigAuthoring::sampleRigPose(moved,nullptr,0,movedPose,batchError));
    assert(std::fabs(movedPose[1].world.m[0][3]-8)<1e-4f && std::fabs(movedPose[1].world.m[1][3])<1e-4f);
    assert(!RigAuthoring::transformRestHierarchy(chain,Matrix4x4::identity(),{"Child","Child"},aroundRoot,moved,batchError) && batchError=="rig_selection_duplicate_bone");
    assert(!RigAuthoring::transformRestHierarchy(chain,Matrix4x4::identity(),{},aroundRoot,moved,batchError) && batchError=="rig_selection_empty");
    assert(!RigAuthoring::transformRestHierarchy(chain,Matrix4x4::identity(),{"missing"},aroundRoot,moved,batchError) && batchError=="unknown_bone");
    assert(!RigAuthoring::transformRestHierarchy(chain,Matrix4x4::identity(),{"Child"},Matrix4x4::scaling(Vec3(-1,1,1)),moved,batchError));
    assert(!RigAuthoring::transformRestHierarchy(chain,Matrix4x4::identity(),{"Child"},Matrix4x4::scaling(Vec3(2,1,1)),moved,batchError));
    // Binding math: actor scale and source affine transform preserve world Rest.
    const auto actor=Matrix4x4::translation(Vec3(4,-2,3))*Matrix4x4::rotationZ(.4f)*Matrix4x4::scaling(Vec3(.0001f,.0001f,.0001f));
    assert(RigAuthoring::transformRestHierarchy(chain,actor,{"Root","Child"},Matrix4x4::translation(Vec3(.1f,.05f,0)),moved,batchError));
    assert(RigAuthoring::sampleRigPose(moved,nullptr,0,movedPose,batchError));
    std::vector<RigAuthoring::PreviewJoint> originalPose;assert(RigAuthoring::sampleRigPose(chain,nullptr,0,originalPose,batchError));
    for(size_t i=0;i<movedPose.size();++i) {
        const auto oldWorld=actor.transform_point(Vec3(originalPose[i].world.m[0][3],originalPose[i].world.m[1][3],originalPose[i].world.m[2][3]));
        const auto newWorld=actor.transform_point(Vec3(movedPose[i].world.m[0][3],movedPose[i].world.m[1][3],movedPose[i].world.m[2][3]));
        assert((newWorld-oldWorld-Vec3(.1f,.05f,0)).length_squared()<1e-8f);
    }
    const auto source=Matrix4x4::translation(Vec3(2,3,-1))*Matrix4x4::rotationX(.3f)*Matrix4x4::scaling(Vec3(2,3,4));
    Matrix4x4 inverseActor,inverseSource;
    assert(RigAuthoring::bindAffineInverse(actor,inverseActor));
    assert(RigAuthoring::bindAffineInverse(source,inverseSource));
    const Vec3 original(.2f,.3f,-.4f);
    const auto expected=source.transform_point(original);
    const auto actual=actor.transform_point((inverseActor*source).transform_point(original));
    assert((actual-expected).length_squared()<1e-6f);
    assert(!RigAuthoring::bindAffineInverse(Matrix4x4::scaling(Vec3(-1,1,1)),inverseSource));
    assert(!RigAuthoring::bindAffineInverse(Matrix4x4::scaling(Vec3(0,1,1)),inverseSource));
    // Coincident outgoing segments merge by bone; ties deterministically use IDs.
    std::vector<RigAuthoring::SkinSegment> segments;
    for(int id=6;id>=0;--id)segments.push_back({id,Vec3(0,0,0),Vec3(0,1,0)});
    segments.push_back(segments.back());double distance;
    auto weights=RigAuthoring::distanceSkinWeights(Vec3(0,.5f,0),segments,1e-4,distance);
    assert(distance==0 && weights.size()==4);
    for(size_t i=0;i<4;++i)assert(weights[i].first==static_cast<int>(i) && std::fabs(weights[i].second-.25f)<1e-6f);
    assert(RigAuthoring::segmentDistanceSquared(Vec3(0,2,0),segments[0])==1);
    std::reverse(segments.begin(),segments.end());
    assert(RigAuthoring::distanceSkinWeights(Vec3(0,.5f,0),segments,1e-4,distance)==weights);
    segments={{7,Vec3(0,0,0),Vec3(0,1,0)},{9,Vec3(10000,0,0),Vec3(10000,1,0)}};
    weights=RigAuthoring::distanceSkinWeights(Vec3(0,.5f,0),segments,1e-9,distance);
    assert(weights.size()==1 && weights[0].first==7 && weights[0].second==1);

        // Paired parent+child mirror uses immutable source globals once.
    RayTrophi::NodeHierarchy h;
    h.addNode("Root","Root",Matrix4x4::translation(Vec3(10,0,0)),-1);
    h.addNode("Child","Child",Matrix4x4::translation(Vec3(0,2,0)),0);
    AnimationData clip;clip.name="Move";clip.modelName="Model";clip.duration=2;clip.ticksPerSecond=1;
    clip.positionKeys["Root"]={{0,Vec3(10,0,0)},{1,Vec3(12,0,0)}};
    std::vector<RigAuthoring::PreviewJoint> out;std::string error;
    assert(RigAuthoring::sampleRigPose(h,&clip,.5,out,error));
    assert(out.size()==2 && out[1].parent=="Root");
    assert(std::fabs(out[0].world.m[0][3]-11)<1e-4f);
    assert(std::fabs(out[1].world.m[0][3]-11)<1e-4f && out[1].world.m[1][3]==2);
    // Pure bind sampling and loop semantics; original clip/hierarchy are unchanged.
    assert(RigAuthoring::sampleRigPose(h,nullptr,.5,out,error) && out[0].world.m[0][3]==10);
    assert(RigAuthoring::sampleRigPose(h,&clip,2.5,out,error) && std::fabs(out[0].world.m[0][3]-11)<1e-4f);
    assert(clip.positionKeys["Root"][1].value.x==12 && h.nodes[0].localBind.m[0][3]==10);
    std::swap(h.nodes[0],h.nodes[1]);h.nodes[0].parent=1;h.nodes[1].parent=-1;
    assert(RigAuthoring::sampleRigPose(h,&clip,.5,out,error) && out[0].parent=="Root" && std::fabs(out[0].world.m[0][3]-11)<1e-4f);
    h.nodes[1].parent=0;
    assert(!RigAuthoring::sampleRigPose(h,&clip,0,out,error) && out.empty() && error=="invalid_preview_hierarchy");
    h.nodes[1].parent=99;
    assert(!RigAuthoring::sampleRigPose(h,&clip,0,out,error) && out.empty());
    assert(!RigAuthoring::sampleRigPose(h,&clip,-1,out,error) && error=="invalid_preview_time");
    assert(!RigAuthoring::sampleRigPose(h,&clip,std::numeric_limits<double>::quiet_NaN(),out,error));
    h.nodes[1].parent=-1;h.nodes[1].localBind.m[0][0]=std::numeric_limits<float>::infinity();
    assert(!RigAuthoring::sampleRigPose(h,nullptr,0,out,error) && out.empty() && error=="invalid_preview_pose");
}
