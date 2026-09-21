// CPU regression: build with Animation/ClipBinding.cpp, assertions enabled.
#include "../../RayTrophiStudio/source/include/Animation/ClipBinding.h"
#include <cassert>

int main() {
    RayTrophi::NodeHierarchy source, target;
    source.addNode("Armature", "2_Armature", Matrix4x4::identity(), -1);
    source.addNode("Hips", "2_Hips", Matrix4x4::identity(), 0);
    source.addNode("Spine", "2_Spine", Matrix4x4::identity(), 1);
    target.addNode("Armature", "1_Armature", Matrix4x4::identity(), -1);
    target.addNode("Hips", "1_Hips", Matrix4x4::identity(), 0);
    target.addNode("Spine", "1_Spine", Matrix4x4::identity(), 1);
    AnimationData clip;
    clip.name = "2_Talking"; clip.modelName = "2";
    clip.duration = 1; clip.ticksPerSecond = 24;
    clip.positionKeys["2_Hips"] = {{0, Vec3(0, 1, 0)}, {1, Vec3(0, 2, 0)}};
    clip.rotationKeys["2_Spine"] = {{0, Quaternion(1, 0, 0, 0)}};
    RigAuthoring::ClipBindingReport report;
    std::shared_ptr<AnimationData> output; std::string error;
    auto build = [&] { return RigAuthoring::buildSameRigClip(clip, source, target, "1", "1_Talking", report, output, error); };
    assert(build() && report.ready && output);
    assert(output->modelName == "1" && output->name == "1_Talking");
    assert(output->positionKeys.count("1_Hips") && !output->positionKeys.count("2_Hips"));
    assert(output->rotationKeys.count("1_Spine"));
    assert(output->positionKeys.at("1_Hips")[1].time == 1);
    assert(output->positionKeys.at("1_Hips")[1].value.y == 2);
    assert(clip.positionKeys.count("2_Hips") && clip.modelName == "2");

    target.nodes[2].parent = 0;
    assert(build() && !report.ready && !output && !report.hierarchy_mismatches.empty());
    target.nodes[2].parent = 1;
    target.nodes[2].name = "OtherSpine";
    assert(build() && !report.ready && !output && !report.unmapped.empty());
    const auto manual = [&](const std::map<std::string, std::string>& map) {
        return RigAuthoring::buildSameRigClip(clip, source, target, "1", "1_Talking", report, output, error, map);
    };
    assert(manual({{"2_Spine", "1_Spine"}}) && report.ready && output);
    assert(output->rotationKeys.count("1_Spine"));
    assert(!manual({{"missing", "1_Spine"}}) && !output && error == "unknown_source_node");
    assert(!manual({{"2_Spine", "missing"}}) && !output && error == "unknown_target_node");
    assert(!manual({{"2_Spine", "1_Hips"}}) && !output && error == "duplicate_target_mapping");
    target.nodes[2].parent = 0;
    assert(manual({{"2_Spine", "1_Spine"}}) && !report.ready && !output);
    target.nodes[2].parent = 1;
    target.nodes[0].name = "OtherRoot";
    assert(manual({{"2_Spine", "1_Spine"}}) && !report.ready && !output);
    assert(manual({{"2_Spine", "1_Spine"}, {"2_Armature", "1_Armature"}}) && report.ready && output);
    target.nodes[0].name = "Armature";
    target.nodes[2].name = "Spine";
    target.addNode("Spine", "1_SecondSpine", Matrix4x4::identity(), 1);
    assert(build() && !report.ready && !output && !report.ambiguous.empty());
    assert(manual({{"2_Spine", "1_Spine"}}) && report.ready && output);
    target.nodes.pop_back();
    target.nodes[1].localBind.m[0][3] = 2;
    assert(build() && report.ready && report.rest_difference_count == 1);
    // Direct transfer does not silently apply a rest-pose correction.
    assert(output->positionKeys.at("1_Hips")[1].value.y == 2);
    clip.ticksPerSecond = 0;
    assert(!build() && !output && error == "invalid_clip_timing");
}
