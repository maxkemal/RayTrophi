#include "../RayTrophiStudio/source/include/Fluid/ColliderMeshClosure.h"

#include <cassert>
#include <limits>
#include <vector>

struct TestPoint {
    double x;
    double y;
    double z;
};

struct TestTriangle {
    TestPoint p0;
    TestPoint p1;
    TestPoint p2;
};

int main() {
    using RayTrophiSim::Fluid::colliderMeshHasClosedVolume;
    const TestPoint a{0, 0, 0};
    const TestPoint b{1, 0, 0};
    const TestPoint c{0, 1, 0};
    const TestPoint d{0, 0, 1};
    std::vector<TestTriangle> tetra{{a, c, b}, {a, b, d}, {a, d, c}, {b, c, d}};
    assert(colliderMeshHasClosedVolume(tetra));

    // Losing one face destroys the meaning of an inside/outside volume.
    auto open = tetra;
    open.pop_back();
    assert(!colliderMeshHasClosedVolume(open));

    auto nonmanifold = tetra;
    nonmanifold.push_back(tetra.front());
    assert(!colliderMeshHasClosedVolume(nonmanifold));

    // Two opposite copies of a wall are still a zero-volume sheet.
    std::vector<TestTriangle> sheet{{a, b, c}, {a, c, b}};
    assert(!colliderMeshHasClosedVolume(sheet));
    assert(!colliderMeshHasClosedVolume(std::vector<TestTriangle>{}));

    // Position seams below the weld tolerance must not turn a solid into a shell.
    auto seam = tetra;
    seam.front().p0.x += 1e-8;
    assert(colliderMeshHasClosedVolume(seam));

    tetra.front().p0.x = std::numeric_limits<double>::quiet_NaN();
    assert(!colliderMeshHasClosedVolume(tetra));
}
