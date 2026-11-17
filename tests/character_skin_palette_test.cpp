// Native regression fixture; user builds/runs it. No application is launched.
// Link with source/src/Math/Matrix4x4.cpp and the project's normal math dependencies.
#include "Animation/CharacterSkinPalette.h"

#include <cassert>
#include <cmath>

namespace {
void near(const Vec3& actual, const Vec3& expected) {
    assert(std::abs(actual.x - expected.x) < 1e-5f);
    assert(std::abs(actual.y - expected.y) < 1e-5f);
    assert(std::abs(actual.z - expected.z) < 1e-5f);
}
}

int main() {
    BoneData bones;
    // Sparse global slots and a second character must survive publication.
    bones.boneNameToIndex = {{"A_hand", 7}, {"B_hand", 2}};
    bones.weightedBoneNames = {"A_hand", "B_hand"};
    bones.boneOffsetMatrices["A_hand"] = Matrix4x4::translation(Vec3(-10, 0, 0));
    bones.boneOffsetMatrices["B_hand"] = Matrix4x4::identity();
    bones.globalInverseTransform = Matrix4x4::translation(Vec3(1000, 0, 0));
    bones.perModelInverses["A"] = Matrix4x4::scaling(Vec3(0.01f, 0.01f, 0.01f));
    bones.rebuildReverseLookup();
    std::vector<Matrix4x4> palette(8, Matrix4x4::identity());
    palette[2] = Matrix4x4::translation(Vec3(0, 12, 0));
    std::unordered_map<std::string, Matrix4x4> globals = {
        {"A_hand", Matrix4x4::translation(Vec3(30, 0, 0))}};

    assert(Animation::publishCharacterSkinPalette(bones, "A", globals, palette));
    // A bind vertex at x=10 follows the animated joint at x=30 centimetres.
    near(palette[7].transform_point(Vec3(10, 0, 0)), Vec3(0.3f, 0, 0));
    near(palette[2].transform_point(Vec3(0, 0, 0)), Vec3(0, 12, 0));
    near(palette[0].transform_point(Vec3(1, 2, 3)), Vec3(1, 2, 3));

    // A missing weighted global must not publish a partial new pose.
    bones.boneNameToIndex["A_elbow"] = 4;
    bones.weightedBoneNames.insert("A_elbow");
    bones.boneOffsetMatrices["A_elbow"] = Matrix4x4::identity();
    globals["A_hand"] = Matrix4x4::translation(Vec3(90, 0, 0));
    assert(!Animation::publishCharacterSkinPalette(bones, "A", globals, palette));
    near(palette[7].transform_point(Vec3(10, 0, 0)), Vec3(0.3f, 0, 0));
    return 0;
}
