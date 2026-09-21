#include "Animation/RigSerialization.h"
#include <cassert>
#include <limits>
int main() {
    RayTrophi::NodeHierarchy original;
    original.addNode("Export Root","7_Root",Matrix4x4::identity(),-1);
    original.addNode("Arm:Left","7_Left",Matrix4x4::translation(Vec3(1,2,3)),0);
    original.addNode("Arm:Left","7_OtherLeft",Matrix4x4::scaling(Vec3(1,2,1)),0);
    const auto saved=RigAuthoring::serializeRigHierarchy(original);
    RayTrophi::NodeHierarchy restored;std::string error;
    assert(RigAuthoring::deserializeRigHierarchy(saved,restored,error));
    assert(RigAuthoring::serializeRigHierarchy(restored)==saved);
    assert(restored.nodes[0].children.size()==2 && restored.nodes[1].name=="Arm:Left");
    const auto unchanged=RigAuthoring::serializeRigHierarchy(restored);
    auto bad=saved;bad["nodes"][0]["parent"]=1;
    assert(!RigAuthoring::deserializeRigHierarchy(bad,restored,error) && error=="cyclic_rig_hierarchy");
    assert(RigAuthoring::serializeRigHierarchy(restored)==unchanged);
    bad=saved;bad["nodes"][1]["parent"]=99;
    assert(!RigAuthoring::deserializeRigHierarchy(bad,restored,error) && error=="invalid_rig_parent");
    bad=saved;bad["nodes"][1]["uniqueName"]="7_Root";
    assert(!RigAuthoring::deserializeRigHierarchy(bad,restored,error) && error=="invalid_rig_node_key");
    bad=saved;bad["nodes"][1]["localBind"][0]="not a number";
    assert(!RigAuthoring::deserializeRigHierarchy(bad,restored,error) && error=="invalid_rig_transform");
    bad=saved;bad["nodes"][1]["localBind"]=nlohmann::json::array({1,2});
    assert(!RigAuthoring::deserializeRigHierarchy(bad,restored,error));
    bad=saved;bad["version"]=2;
    assert(!RigAuthoring::deserializeRigHierarchy(bad,restored,error) && error=="unsupported_rig_hierarchy_version");
    original.nodes[0].localBind.m[0][0]=std::numeric_limits<float>::infinity();
    bool rejected=false;try{RigAuthoring::serializeRigHierarchy(original);}catch(const std::exception&){rejected=true;}
    assert(rejected);
    BoneData bones;
    bones.boneDefaultTransforms["7_ZRoot"]=Matrix4x4::identity();
    bones.boneDefaultTransforms["7_AChild"]=Matrix4x4::translation(Vec3(0,1,0));
    bones.boneParents["7_AChild"]="7_ZRoot";
    assert(RigAuthoring::rebuildLegacyRigHierarchy(bones,"7",restored,error));
    assert(restored.nodes[0].uniqueName=="7_ZRoot" && restored.nodes[1].parent==0);
    assert(restored.nodes[0].children.size()==1);
    bones.boneDefaultTransforms["7_SeparateRoot"]=Matrix4x4::identity();
    assert(RigAuthoring::rebuildLegacyRigHierarchy(bones,"7",restored,error) && restored.size()==4);
    assert(restored.nodes[0].name=="__LegacyRoot" && restored.nodes[0].children.size()==2);
    bones.boneParents["7_ZRoot"]="7_AChild";
    assert(!RigAuthoring::rebuildLegacyRigHierarchy(bones,"7",restored,error) && error=="cyclic_rig_hierarchy");
    bones.boneParents["7_ZRoot"]="7_Missing";
    assert(!RigAuthoring::rebuildLegacyRigHierarchy(bones,"7",restored,error) && error=="missing_legacy_rig_parent");
}
