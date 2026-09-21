// CPU regression. User compiles/runs; keep assertions enabled.
#include "Animation/ClipBinding.h"
#include <cassert>
#include <utility>
#include <cmath>
#include <limits>

static bool near(const Matrix4x4& a, const Matrix4x4& b) {
    for (int r=0; r<4; ++r) for (int c=0; c<4; ++c)
        if (std::fabs(a.m[r][c]-b.m[r][c]) > 2e-4f) return false;
    return true;
}
int main() {
    const float h = std::sqrt(0.5f);
    const Quaternion z90(h, 0, 0, h), x90(h, h, 0, 0);
    RayTrophi::NodeHierarchy s, t;
    s.addNode("Root", "s_Root", Matrix4x4::identity(), -1);
    s.addNode("Arm", "s_Arm", Matrix4x4::translation(Vec3(0,1,0)), 0);
    t.addNode("Root", "t_Root", Matrix4x4::identity(), -1);
    t.addNode("Arm", "t_Arm", Matrix4x4::translation(Vec3(0,2,0)), 0);
    AnimationData clip; clip.name="s_Move"; clip.modelName="s"; clip.duration=2; clip.ticksPerSecond=1;
    clip.rotationKeys["s_Arm"]={{0, Quaternion()}, {1, x90}};
    clip.positionKeys["s_Arm"]={{0, Vec3(0,1,0)}, {1, Vec3(1,1,0)}};
    RigAuthoring::ClipBindingReport report; std::shared_ptr<AnimationData> out; std::string error;
    auto build = [&](float scale=1.f, const std::string& mode="rest_basis") {
        return RigAuthoring::buildSameRigClip(clip,s,t,"t","t_Move",report,out,error,{},mode,scale);
    };
    assert(build() && out && report.ready && report.mode=="rest_basis");
    assert(near(out->rotationKeys.at("t_Arm")[1].value.toMatrix(),x90.toMatrix()));
    assert(out->positionKeys.at("t_Arm")[0].value.y==2);
    assert(build(2) && std::fabs(out->positionKeys.at("t_Arm")[1].value.x-2)<1e-4f);
    // Target array ordering differs from parent order; matching uses unique keys.
    std::swap(t.nodes[0],t.nodes[1]); t.nodes[0].parent=1; t.nodes[1].parent=-1;
    assert(build() && std::fabs(out->positionKeys.at("t_Arm")[0].value.y-2)<1e-4f);
    std::swap(t.nodes[0],t.nodes[1]); t.nodes[0].parent=-1; t.nodes[1].parent=0;
    // Different A/T local rest orientation: source rest keys yield target rest.
    s.nodes[1].localBind=Matrix4x4::translation(Vec3(0,1,0))*z90.toMatrix();
    clip.rotationKeys["s_Arm"]={{0,z90},{1,z90*x90}};
    assert(build() && near(out->rotationKeys.at("t_Arm")[0].value.toMatrix(),Matrix4x4::identity()));
    const auto expected=z90*x90*z90.conjugate();
    assert(near(out->rotationKeys.at("t_Arm")[1].value.toMatrix(),expected.toMatrix()));
    // Root basis rotates parent-space translation; parent uniform units cancel.
    s.nodes[0].localBind=z90.toMatrix()*Matrix4x4::scaling(Vec3(2,2,2));
    t.nodes[0].localBind=Matrix4x4::scaling(Vec3(4,4,4));
    assert(build());
    const auto p=out->positionKeys.at("t_Arm")[1].value;
    assert(std::fabs(p.x)<1e-4f && std::fabs(p.y-2.5f)<1e-4f);
    // Missing channels remain missing: runtime must use target bind fallback.
    assert(!out->rotationKeys.count("t_Root") && out->scalingKeys.empty());
    assert(clip.positionKeys.at("s_Arm")[1].value.x==1 && clip.modelName=="s");
    // Rest scale ratios, and unsupported transforms/keys, fail without output.
    clip.scalingKeys["s_Arm"]={{0,Vec3(1,1,1)},{1,Vec3(2,2,2)}};
    assert(build() && out->scalingKeys.at("t_Arm")[1].value.x==2);
    clip.scalingKeys["s_Arm"][1].value=Vec3(1,2,1);
    assert(!build() && !out && !report.ready && error=="unsupported_retarget_scale_keys");
    clip.scalingKeys.clear();
    t.nodes[1].localBind=Matrix4x4::scaling(Vec3(1,2,1));
    assert(!build() && !out && error=="unsupported_retarget_rest");
    t.nodes[1].localBind=Matrix4x4::identity(); t.nodes[1].localBind.m[0][1]=0.5f;
    assert(!build() && !out && error=="unsupported_retarget_rest");
    t.nodes[1].localBind=Matrix4x4::scaling(Vec3(-1,1,1));
    assert(!build() && !out && error=="unsupported_retarget_rest");
    t.nodes[1].localBind=Matrix4x4::identity();
    clip.rotationKeys["s_Arm"][1].value=Quaternion(0,0,0,0);
    assert(!build() && !out && error=="invalid_retarget_keys");
    clip.rotationKeys["s_Arm"][1].value=x90;
    clip.positionKeys["s_Arm"][1].time=0;
    assert(!build() && !out && error=="invalid_retarget_keys");
    clip.positionKeys["s_Arm"][1].time=1;
    assert(!build(0) && !out && error=="invalid_translation_scale");
    assert(!build(std::numeric_limits<float>::quiet_NaN()) && !out);
    assert(!build(1,"bad") && !out && error=="invalid_retarget_mode");
    assert(!build(2,"same_rig") && !out && error=="translation_scale_requires_retarget");
    // Existing direct mode retains absolute TRS and does not impose rest restrictions.
    t.nodes[1].localBind=Matrix4x4::scaling(Vec3(1,2,1));
    assert(build(1,"same_rig") && out && out->positionKeys.at("t_Arm")[0].value.y==1);
}
