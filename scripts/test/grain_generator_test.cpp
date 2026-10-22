#include "GrainGenerator.h"

#include <algorithm>
#include <cassert>
#include <cmath>

using namespace RayTrophiSim;

void checkShape(const Fluid::GrainExemplar& shape) {
    assert(!shape.faces.empty());
    Vec3 flux(0.0f);
    double volume6 = 0.0;
    Vec3 low(1e30f), high(-1e30f);
    for (const auto& triangle : shape.faces) {
        const Vec3 cross = Vec3::cross(triangle.p[1] - triangle.p[0],
            triangle.p[2] - triangle.p[0]);
        assert(cross.length() > 1e-12f);
        flux = flux + cross;
        volume6 += Vec3::dot(triangle.p[0], Vec3::cross(triangle.p[1], triangle.p[2]));
        for (int corner = 0; corner < 3; ++corner) {
            const auto& point = triangle.p[corner];
            assert(std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z));
            assert(triangle.n[corner].length() > 0.9f);
            low = Vec3::min(low, point);
            high = Vec3::max(high, point);
        }
    }
    assert(flux.length() < 1e-4f);
    assert(volume6 > 0.0);
    const Vec3 extent = high - low;
    assert(std::abs(std::max({extent.x, extent.y, extent.z}) - 1.0f) < 1e-5f);
}

int main() {
    for (const std::string type : {"rock", "voronoi", "snow", "crystal"}) {
        GrainGeneratorSettings settings;
        std::string error;
        assert(parseGrainGeneratorSettings({{"type", type}, {"count", 3},
            {"seed", 8127}, {"aspect", {0.7f, 1.6f, 1.0f}}}, settings, error));
        std::vector<Fluid::GrainExemplar> first, replay;
        assert(generateGrainShapes(settings, first, error));
        assert(generateGrainShapes(settings, replay, error));
        assert(first.size() == replay.size());
        assert(first.size() <= 3);
        for (std::size_t index = 0; index < first.size(); ++index) {
            checkShape(first[index]);
            assert(first[index].faces.size() == replay[index].faces.size());
            for (std::size_t face = 0; face < first[index].faces.size(); ++face) {
                assert((first[index].faces[face].p[0] - replay[index].faces[face].p[0]).length()
                    < 1e-7f);
            }
        }
    }
    // One requested Voronoi example must still be a seeded cut cell, not the cube.
    GrainGeneratorSettings cell;
    cell.count = 1;
    std::string cell_error;
    std::vector<Fluid::GrainExemplar> sharp, rounded, detailed, other_seed;
    assert(generateGrainShapes(cell, sharp, cell_error));
    assert(sharp.size() == 1);
    bool oblique_face = false;
    for (const auto& triangle : sharp.front().faces) {
        const Vec3 n = triangle.n[0];
        if (std::max({std::abs(n.x), std::abs(n.y), std::abs(n.z)}) < 0.99f) {
            oblique_face = true;
        }
    }
    assert(oblique_face);
    cell.seed += 1;
    assert(generateGrainShapes(cell, other_seed, cell_error));
    assert((sharp.front().faces.front().p[0] -
        other_seed.front().faces.front().p[0]).length() > 1e-5f);
    cell.seed -= 1;
    cell.roundness = 0.7f;
    cell.aspect = Vec3(0.7f, 1.6f, 1.0f);
    assert(generateGrainShapes(cell, rounded, cell_error));
    checkShape(rounded.front());
    assert(rounded.front().faces.size() > sharp.front().faces.size());
    bool curved = false;
    for (const auto& triangle : rounded.front().faces) {
        curved = curved || (triangle.n[0] - triangle.n[1]).length() > 0.01f;
    }
    assert(curved);
    cell.resolution = 8;
    assert(generateGrainShapes(cell, detailed, cell_error));
    checkShape(detailed.front());
    assert(detailed.front().faces.size() > rounded.front().faces.size());
    // Dendrite topology boundaries: no twigs, maximum twig count, thin plates,
    // many arms, and asymmetric seeded growth must remain closed solids.
    for (const int arms : {3, 6, 12}) {
        for (const int branches : {0, 5}) {
            GrainGeneratorSettings settings;
            std::string error;
            assert(parseGrainGeneratorSettings({{"type", "snow"}, {"count", 1},
                {"arms", arms}, {"branches", branches}, {"thickness", 0.005f},
                {"branch_length", 0.4f}, {"roughness", 0.5f}}, settings, error));
            std::vector<Fluid::GrainExemplar> shapes;
            assert(generateGrainShapes(settings, shapes, error));
            checkShape(shapes.front());
        }
    }
    for (const auto& invalid : {
        nlohmann::json{{"type", "unsupported"}}, nlohmann::json{{"count", 65}},
        nlohmann::json{{"count", true}}, nlohmann::json{{"resolution", 4.5}},
        nlohmann::json{{"aspect", {1, 0, 1}}}, nlohmann::json{{"thickness", 0}},
        nlohmann::json{{"unexpected_key", 1}}, nlohmann::json{{"seed", -1}}}) {
        GrainGeneratorSettings settings;
        settings.count = 7;
        std::string error;
        assert(!parseGrainGeneratorSettings(invalid, settings, error));
        assert(!error.empty());
        assert(settings.count == 7);
    }
}
