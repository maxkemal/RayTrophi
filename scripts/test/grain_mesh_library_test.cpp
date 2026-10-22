#include "Fluid/GrainMeshNear.h"
#include "ParticleSimulation.h"
#include "Triangle.h"
#include "TriangleMesh.h"

#include <cassert>
#include <cmath>
#include <limits>

using namespace RayTrophiSim;
using namespace RayTrophiSim::Fluid;

namespace {

void checkProceduralLibrary(const std::string& shape) {
    std::vector<GrainExemplar> library;
    std::vector<GrainExemplar> replay;
    std::string error;
    assert(buildGrainExemplars({}, {}, library, error, shape));
    assert(!library.empty());
    assert(buildGrainExemplars({}, {}, replay, error, shape));
    assert(library.size() == replay.size());
    for (std::size_t index = 0; index < library.size(); ++index) {
        const auto& exemplar = library[index];
        assert(exemplar.faces.size() == replay[index].faces.size());
        Vec3 area_sum(0.0f);
        double volume6 = 0.0;
        for (std::size_t face_index = 0; face_index < exemplar.faces.size(); ++face_index) {
            const auto& face = exemplar.faces[face_index];
            const Vec3 cross = Vec3::cross(face.p[1] - face.p[0], face.p[2] - face.p[0]);
            assert(cross.length() > 0.0f);
            area_sum = area_sum + cross;
            volume6 += Vec3::dot(face.p[0], Vec3::cross(face.p[1], face.p[2]));
            for (int corner = 0; corner < 3; ++corner) {
                const auto& point = face.p[corner];
                assert(std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z));
                assert((point - replay[index].faces[face_index].p[corner]).length() < 1e-7f);
                assert(face.n[corner].length() > 0.9f);
            }
        }
        // A closed library must have zero net surface-area vector and positive
        // outward volume. Check the sphere-equivalent volume independently.
        assert(area_sum.length() < 1e-4f);
        assert(volume6 > 0.0);
        const double scale = exemplar.volume_scale;
        if (exemplar.volume_measured && scale > kGrainMeshMinVolumeScale &&
            scale < kGrainMeshMaxVolumeScale) {
            assert(std::abs(volume6 / 6.0 * scale * scale * scale - 0.52359878) < 1e-4);
        }
    }
}

} // namespace

int main() {
    checkProceduralLibrary("rock");
    checkProceduralLibrary("voronoi");

    // One source node with two disjoint, differently materialed flat triangles.
    auto mesh = std::make_shared<TriangleMesh>();
    mesh->nodeName = "TwoMaterials";
    mesh->geometry = std::make_shared<DNA::GeometryDetail>();
    auto& geometry = *mesh->geometry;
    geometry.resize_vertices(6);
    geometry.add_attribute<Vec3>("P");
    geometry.add_attribute<uint16_t>("materialID");
    Vec3* positions = geometry.get_attribute_data_mut<Vec3>("P");
    uint16_t* materials = geometry.get_attribute_data_mut<uint16_t>("materialID");
    const Vec3 points[] = {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0),
        Vec3(0, 0, 1), Vec3(1, 0, 1), Vec3(0, 1, 1)};
    for (uint32_t index = 0; index < 6; ++index) {
        positions[index] = points[index];
        materials[index] = index < 3 ? 5u : 9u;
    }
    geometry.indices = {3, 4, 5, 0, 1, 2};
    std::vector<GrainExemplar> library;
    std::string error;
    assert(buildGrainExemplars({mesh}, {mesh->nodeName}, library, error));
    assert(library[0].faces[0].material == 9u);
    assert(library[0].faces[1].material == 5u);
    std::vector<std::shared_ptr<Triangle>> render_triangles;
    buildGrainExemplarTriangles(library[0], -1, "Original", render_triangles);
    assert(render_triangles[0]->getMaterialID() == 9u);
    assert(render_triangles[1]->getMaterialID() == 5u);
    buildGrainExemplarTriangles(library[0], 12, "Override", render_triangles);
    assert(render_triangles[0]->getMaterialID() == 12u);
    assert(render_triangles[1]->getMaterialID() == 12u);

    const uint64_t signature = grainMeshSceneSignature({mesh}, {mesh->nodeName});
    materials[0] = 7u;
    assert(grainMeshSceneSignature({mesh}, {mesh->nodeName}) != signature);
    materials[0] = 5u;
    assert(grainMeshSceneSignature({mesh}, {mesh->nodeName}) == signature);

    positions[0].x = std::numeric_limits<float>::quiet_NaN();
    assert(!buildGrainExemplars({mesh}, {mesh->nodeName}, library, error));
    assert(library.empty());
    assert(!buildGrainExemplars({}, {}, library, error, "invalid"));

    SimulationGridDomainDesc domain;
    domain.fluid_particle_visual_children = 8;
    domain.fluid_params.grain.enabled = false;
    domain.fluid_params.granular_enabled = false;
    assert(primarySplatVisualChildren(domain) == 8u);
    domain.fluid_grain_mesh_near_enabled = true;
    assert(primarySplatVisualChildren(domain) == 1u);
    // MPM selection also replaces whole carriers, irrespective of proxy settings.
    domain.fluid_params.granular_enabled = true;
    domain.fluid_granular_physical_carriers = false;
    assert(primarySplatVisualChildren(domain) == 1u);
    FluidParticles particles;
    assert(splatCarrierDiameter(domain, particles, 0, 0.12f) == 0.12f);
    domain.fluid_grain_mesh_near_enabled = false;
    assert(primarySplatVisualChildren(domain) == 8u);
}
