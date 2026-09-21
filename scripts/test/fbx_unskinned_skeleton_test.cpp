// CPU-only regression test. No application, mesh, skin or animation required.
#include "../../RayTrophiStudio/source/include/Import/UfbxSkeletonSeeds.h"
#include <cassert>

int main() {
    ufbx_bone bone{};
    ufbx_node root{}, armature{}, hip{}, knee{}, ordinary{};
    armature.parent = &root;
    hip.parent = &armature;
    hip.bone = &bone;
    knee.parent = &hip;
    knee.bone = &bone;
    ordinary.parent = &root;

    // Reverse order exercises shared ancestor closure and joint indexing.
    ufbx_node* nodes[] = {&knee, &ordinary, &hip, &armature, &root};
    ufbx_scene scene{};
    scene.nodes.data = nodes;
    scene.nodes.count = 5;
    std::unordered_set<const ufbx_node*> technical, indexed;
    rtimport::collectExplicitFbxJoints(scene, technical, indexed);
    assert(indexed.size() == 2 && indexed.count(&hip) && indexed.count(&knee));
    assert(technical.size() == 4 && technical.count(&root) && technical.count(&armature));
    assert(!technical.count(&ordinary) && !indexed.count(&armature));

    rtimport::collectExplicitFbxJoints(scene, technical, indexed);
    assert(technical.size() == 4 && indexed.size() == 2);

    hip.bone = nullptr;
    knee.bone = nullptr;
    technical.clear();
    indexed.clear();
    rtimport::collectExplicitFbxJoints(scene, technical, indexed);
    assert(technical.empty() && indexed.empty());
}
