#include "GrainGenerator.h"

#include "FractureGenerator.h"
#include "Perlin.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace RayTrophiSim {
using Fluid::GrainExemplar;
using Fluid::NormalizedSceneFace;
using Fluid::kGrainMeshMinVolumeScale;
using Fluid::kGrainMeshMaxVolumeScale;

namespace {
constexpr float pi = 3.14159265358979323846f;
uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
float unit01(uint64_t bits) {
    return static_cast<float>(bits >> 40) / 16777216.0f;
}
} // namespace

// Divergence-theorem volume. Meaningful only for a closed, consistently
// wound surface; the caller checks the sign and size.
float signedVolume(const std::vector<NormalizedSceneFace>& faces) {
    double v = 0.0;
    for (const auto& f : faces) {
        const Vec3 c = Vec3::cross(f.p[1], f.p[2]);
        v += static_cast<double>(Vec3::dot(f.p[0], c));
    }
    return static_cast<float>(v / 6.0);
}

void normalizeGrainShape(GrainExemplar& exemplar) {
    exemplar.volume_scale = 1.0f;
    exemplar.volume_measured = false;
    Vec3 low(1e30f), high(-1e30f);
    for (const auto& face : exemplar.faces) {
        for (const auto& point : face.p) {
            low = Vec3::min(low, point);
            high = Vec3::max(high, point);
        }
    }
    const Vec3 center = (low + high) * 0.5f;
    const Vec3 extent = high - low;
    const float inverse = 1.0f / std::max({extent.x, extent.y, extent.z, 1e-4f});
    for (auto& face : exemplar.faces) {
        for (auto& point : face.p) {
            point = (point - center) * inverse;
        }
    }
    const float volume = std::abs(signedVolume(exemplar.faces));
    if (std::isfinite(volume) && volume > 0.01f && volume <= 1.0001f) {
        exemplar.volume_measured = true;
        exemplar.volume_scale = std::clamp(std::cbrt(0.52359878f / volume),
            kGrainMeshMinVolumeScale, kGrainMeshMaxVolumeScale);
    }
}

// Same seeded Perlin radial displacement as the procedural rock workflow,
// at grain-library resolution. Shared edge coordinates sample the same noise,
// so the six cube-sphere faces stay closed without any scene Triangle facades.
void buildRockLibrary(const GrainGeneratorSettings& settings,
    std::vector<GrainExemplar>& out) {
    struct Frame {
        Vec3 normal;
        Vec3 tangent;
        Vec3 bitangent;
    };
    const Frame frames[] = {
        {Vec3(1, 0, 0), Vec3(0, 0, -1), Vec3(0, 1, 0)},
        {Vec3(-1, 0, 0), Vec3(0, 0, 1), Vec3(0, 1, 0)},
        {Vec3(0, 1, 0), Vec3(1, 0, 0), Vec3(0, 0, -1)},
        {Vec3(0, -1, 0), Vec3(1, 0, 0), Vec3(0, 0, 1)},
        {Vec3(0, 0, 1), Vec3(1, 0, 0), Vec3(0, 1, 0)},
        {Vec3(0, 0, -1), Vec3(-1, 0, 0), Vec3(0, 1, 0)}
    };
    const int resolution = settings.resolution;
    for (uint32_t variant = 0; variant < static_cast<uint32_t>(settings.count); ++variant) {
        Perlin noise(settings.seed + variant * 7919u);
        GrainExemplar exemplar;
        exemplar.node = "Rock " + std::to_string(variant);
        const Vec3 aspect(0.8f + 0.4f * unit01(mix64(variant)),
            0.7f + 0.5f * unit01(mix64(variant + 101u)), 1.0f);
        auto position = [&](const Frame& frame, int x, int y) {
            const float u = -1.0f + 2.0f * static_cast<float>(x) / resolution;
            const float v = -1.0f + 2.0f * static_cast<float>(y) / resolution;
            const Vec3 direction = (frame.normal + frame.tangent * u +
                frame.bitangent * v).normalize();
            const float radius = 1.0f + settings.roughness * noise.noise(direction * 1.7f) +
                0.4f * settings.roughness * noise.noise(direction * 3.4f);
            const Vec3 cube = frame.normal + frame.tangent * u + frame.bitangent * v;
            const Vec3 form = direction * settings.roundness +
                cube * ((1.0f - settings.roundness) / 1.732050808f);
            return Vec3(form.x * radius * aspect.x,
                form.y * radius * aspect.y, form.z * radius * aspect.z);
        };
        for (const auto& frame : frames) {
            for (int y = 0; y < resolution; ++y) {
                for (int x = 0; x < resolution; ++x) {
                    const Vec3 quad[] = {position(frame, x, y), position(frame, x + 1, y),
                        position(frame, x + 1, y + 1), position(frame, x, y + 1)};
                    const int corners[][3] = {{0, 1, 2}, {0, 2, 3}};
                    for (const auto& indices : corners) {
                        NormalizedSceneFace face;
                        for (int corner = 0; corner < 3; ++corner) {
                            face.p[corner] = quad[indices[corner]];
                            face.n[corner] = face.p[corner].normalize();
                            face.uv[corner] = Vec2(0.0f, 0.0f);
                        }
                        exemplar.faces.push_back(face);
                    }
                }
            }
        }
        normalizeGrainShape(exemplar);
        out.push_back(std::move(exemplar));
    }
}

// Smooth the intersection of the cell's supporting half-spaces. Unlike moving
// the original corners toward a sphere, this adds surface samples and rounds
// edges while retaining broad planar facets. Analytic gradients give continuous
// normals across triangulation seams.
void roundVoronoiCell(GrainExemplar& shape, const GrainGeneratorSettings& settings) {
    if (settings.roundness <= 0.0f) {
        return;
    }
    struct Plane {
        Vec3 normal;
        float distance;
    };
    Vec3 center(0.0f);
    for (const auto& triangle : shape.faces) {
        center = center + triangle.p[0] + triangle.p[1] + triangle.p[2];
    }
    center = center * (1.0f / static_cast<float>(shape.faces.size() * 3));
    std::vector<Plane> planes;
    float clearance = 1e30f;
    for (const auto& triangle : shape.faces) {
        const Vec3 normal = Vec3::cross(triangle.p[1] - triangle.p[0],
            triangle.p[2] - triangle.p[0]).normalize();
        const float distance = Vec3::dot(normal, triangle.p[0] - center);
        if (distance <= 1e-6f) {
            continue;
        }
        bool duplicate = false;
        for (const auto& plane : planes) {
            if (Vec3::dot(normal, plane.normal) > 0.99999f &&
                std::abs(distance - plane.distance) < 1e-5f) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            planes.push_back({normal, distance});
            clearance = std::min(clearance, distance);
        }
    }
    if (planes.size() < 4) {
        return;
    }
    const float width = std::max(1e-7f, settings.roundness * clearance * 0.8f /
        (1.0f + std::log(static_cast<float>(planes.size()))));
    auto field = [&](const Vec3& point, Vec3* gradient) {
        float maximum = -1e30f;
        for (const auto& plane : planes) {
            maximum = std::max(maximum, Vec3::dot(plane.normal, point) - plane.distance);
        }
        float sum = 0.0f;
        Vec3 normal(0.0f);
        for (const auto& plane : planes) {
            const float weight = std::exp((Vec3::dot(plane.normal, point) -
                plane.distance - maximum) / width);
            sum += weight;
            normal = normal + plane.normal * weight;
        }
        if (gradient) {
            *gradient = normal.normalize();
        }
        return maximum + width * std::log(sum);
    };
    std::vector<NormalizedSceneFace> rounded;
    const int detail = settings.resolution;
    for (const auto& triangle : shape.faces) {
        auto sample = [&](int x, int y, Vec3& normal) {
            const Vec3 target = triangle.p[0] +
                (triangle.p[1] - triangle.p[0]) * (static_cast<float>(x) / detail) +
                (triangle.p[2] - triangle.p[0]) * (static_cast<float>(y) / detail) - center;
            float low = 0.0f;
            float high = 1.0f;
            for (int step = 0; step < 24; ++step) {
                const float middle = (low + high) * 0.5f;
                if (field(target * middle, nullptr) > 0.0f) {
                    high = middle;
                } else {
                    low = middle;
                }
            }
            const Vec3 point = target * ((low + high) * 0.5f);
            field(point, &normal);
            return center + point;
        };
        const int stride = detail + 1;
        std::vector<Vec3> positions(stride * stride), normals(stride * stride);
        for (int y = 0; y <= detail; ++y) {
            for (int x = 0; x <= detail - y; ++x) {
                positions[y * stride + x] = sample(x, y, normals[y * stride + x]);
            }
        }
        auto append = [&](int ax, int ay, int bx, int by, int cx, int cy) {
            NormalizedSceneFace value;
            const int indices[] = {ay * stride + ax, by * stride + bx, cy * stride + cx};
            for (int corner = 0; corner < 3; ++corner) {
                value.p[corner] = positions[indices[corner]];
                value.n[corner] = normals[indices[corner]];
            }
            rounded.push_back(value);
        };
        for (int y = 0; y < detail; ++y) {
            for (int x = 0; x < detail - y; ++x) {
                append(x, y, x + 1, y, x, y + 1);
                if (x + y + 1 < detail) {
                    append(x + 1, y, x + 1, y + 1, x, y + 1);
                }
            }
        }
    }
    shape.faces = std::move(rounded);
}

// Build a small deterministic library through the existing convex Voronoi core.
// This is transient render geometry: it creates no scene objects or rigid bodies.
bool buildVoronoiLibrary(const GrainGeneratorSettings& settings,
    std::vector<GrainExemplar>& out, std::string& error) {
    const Vec3 points[] = {
        Vec3(-1, -1, -1), Vec3(1, -1, -1), Vec3(1, 1, -1), Vec3(-1, 1, -1),
        Vec3(-1, -1, 1), Vec3(1, -1, 1), Vec3(1, 1, 1), Vec3(-1, 1, 1)
    };
    const int faces[][3] = {
        {0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7},
        {0, 1, 5}, {0, 5, 4}, {3, 7, 6}, {3, 6, 2},
        {0, 4, 7}, {0, 7, 3}, {1, 2, 6}, {1, 6, 5}
    };
    std::vector<FractureInputTri> source;
    for (const auto& face : faces) {
        FractureInputTri triangle;
        triangle.a = points[face[0]];
        triangle.b = points[face[1]];
        triangle.c = points[face[2]];
        source.push_back(triangle);
    }
    FractureParams params;
    // Always cut a populated tessellation, even when requesting one example.
    params.site_count = std::max(32, settings.count * 2);
    params.seed = settings.seed;
    std::vector<FractureShard> shards;
    if (!generateFracture(source, params, shards)) {
        error = "Voronoi grain generation failed";
        return false;
    }
    for (const auto& shard : shards) {
        if (shard.tris.empty()) {
            continue;
        }
        GrainExemplar exemplar;
        exemplar.node = "Voronoi " + std::to_string(out.size());
        for (const auto& triangle : shard.tris) {
            NormalizedSceneFace face;
            face.p[0] = triangle.a;
            face.p[1] = triangle.b;
            face.p[2] = triangle.c;
            for (int corner = 0; corner < 3; ++corner) {
                face.n[corner] = triangle.n;
                face.uv[corner] = Vec2(0.0f, 0.0f);
            }
            exemplar.faces.push_back(face);
        }
        normalizeGrainShape(exemplar);
        roundVoronoiCell(exemplar, settings);
        out.push_back(std::move(exemplar));
        if (out.size() >= static_cast<std::size_t>(settings.count)) {
            break;
        }
    }
    if (out.empty()) {
        error = "Voronoi grain library is empty";
        return false;
    }
    return true;
}


namespace {

void face(GrainExemplar& shape, const Vec3& a, const Vec3& b, const Vec3& c) {
    NormalizedSceneFace value;
    value.p[0] = a;
    value.p[1] = b;
    value.p[2] = c;
    const Vec3 cross = Vec3::cross(b - a, c - a);
    const float length = cross.length();
    if (!(length > 1e-12f)) {
        return;
    }
    for (int corner = 0; corner < 3; ++corner) {
        value.n[corner] = cross * (1.0f / length);
        value.uv[corner] = Vec2(value.p[corner].x + 0.5f, value.p[corner].z + 0.5f);
    }
    shape.faces.push_back(value);
}

GrainExemplar crystal(const GrainGeneratorSettings& settings, uint32_t variant) {
    GrainExemplar shape;
    const int sides = settings.crystal_sides;
    const float phase = unit01(mix64(settings.seed + variant)) * 2.0f * pi;
    const float shoulder = 0.5f * (1.0f - settings.tip_height);
    auto ring = [&](int index, float y) {
        const float angle = phase + 2.0f * pi * static_cast<float>(index) / sides;
        return Vec3(0.5f * std::cos(angle), y, 0.5f * std::sin(angle));
    };
    for (int side = 0; side < sides; ++side) {
        const Vec3 a = ring(side, -shoulder);
        const Vec3 b = ring(side + 1, -shoulder);
        const Vec3 c = ring(side + 1, shoulder);
        const Vec3 d = ring(side, shoulder);
        face(shape, a, d, c);
        face(shape, a, c, b);
        face(shape, d, Vec3(0, 0.5f, 0), c);
        face(shape, a, b, Vec3(0, -0.5f, 0));
    }
    return shape;
}

float cross2(const Vec2& a, const Vec2& b, const Vec2& c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

// Ear clipping keeps the dendrite a single closed solid, without overlapping
// translucent branch bars or internal faces.
bool extrudeOutline(const std::vector<Vec2>& outline, float thickness, GrainExemplar& shape) {
    std::vector<int> remaining(outline.size());
    std::iota(remaining.begin(), remaining.end(), 0);
    auto vertex = [&](int index, float y) {
        return Vec3(outline[index].x, y, outline[index].y);
    };
    const float top = thickness * 0.5f;
    while (remaining.size() > 2) {
        bool clipped = false;
        for (std::size_t index = 0; index < remaining.size(); ++index) {
            const int a = remaining[(index + remaining.size() - 1) % remaining.size()];
            const int b = remaining[index];
            const int c = remaining[(index + 1) % remaining.size()];
            const float turn = cross2(outline[a], outline[b], outline[c]);
            if (std::abs(turn) <= 1e-10f) {
                remaining.erase(remaining.begin() + index);
                clipped = true;
                break;
            }
            if (turn < 0.0f) {
                continue;
            }
            bool occupied = false;
            for (const int other : remaining) {
                if (other == a || other == b || other == c) {
                    continue;
                }
                if (cross2(outline[a], outline[b], outline[other]) >= -1e-10f &&
                    cross2(outline[b], outline[c], outline[other]) >= -1e-10f &&
                    cross2(outline[c], outline[a], outline[other]) >= -1e-10f) {
                    occupied = true;
                    break;
                }
            }
            if (!occupied) {
                face(shape, vertex(a, top), vertex(c, top), vertex(b, top));
                face(shape, vertex(a, -top), vertex(b, -top), vertex(c, -top));
                remaining.erase(remaining.begin() + index);
                clipped = true;
                break;
            }
        }
        if (!clipped) {
            return false;
        }
    }
    for (std::size_t index = 0; index < outline.size(); ++index) {
        const int a = static_cast<int>(index);
        const int b = static_cast<int>((index + 1) % outline.size());
        face(shape, vertex(a, -top), vertex(a, top), vertex(b, top));
        face(shape, vertex(a, -top), vertex(b, top), vertex(b, -top));
    }
    return true;
}

bool snow(const GrainGeneratorSettings& settings, uint32_t variant, GrainExemplar& shape) {
    std::vector<Vec2> outline;
    constexpr float root = 0.16f;
    constexpr float width = 0.018f;
    const float spacing = 0.70f / static_cast<float>(settings.branches + 1);
    const float branch = std::min(settings.branch_length, spacing * 0.6f);
    const float phase = unit01(mix64(settings.seed + variant)) * 2.0f * pi;
    for (int arm = 0; arm < settings.arms; ++arm) {
        const float angle = phase + 2.0f * pi * static_cast<float>(arm) / settings.arms;
        const float irregularity = 1.0f + settings.roughness * 0.2f *
            (2.0f * unit01(mix64(settings.seed + variant * 37u + arm * 101u)) - 1.0f);
        auto point = [&](float x, float y) {
            outline.emplace_back((x * std::cos(angle) - y * std::sin(angle)) * irregularity,
                (x * std::sin(angle) + y * std::cos(angle)) * irregularity);
        };
        point(root, -width);
        for (int twig = 0; twig < settings.branches; ++twig) {
            const float x = root + spacing * (twig + 1);
            point(x - branch * 0.25f, -width);
            point(x + branch * 0.5f, -width - branch * 0.866025404f);
            point(x + branch * 0.25f, -width);
        }
        point(1.0f, 0.0f);
        for (int twig = settings.branches - 1; twig >= 0; --twig) {
            const float x = root + spacing * (twig + 1);
            point(x + branch * 0.25f, width);
            point(x + branch * 0.5f, width + branch * 0.866025404f);
            point(x - branch * 0.25f, width);
        }
        point(root, width);
    }
    return extrudeOutline(outline, settings.thickness, shape);
}

} // namespace

bool parseGrainGeneratorSettings(const nlohmann::json& input,
    GrainGeneratorSettings& settings, std::string& error) {
    error.clear();
    if (!input.is_object()) {
        error = "grain generator settings must be an object";
        return false;
    }
    try {
        const std::vector<std::string> keys = {"type", "count", "seed", "resolution",
            "roundness", "roughness", "aspect", "arms", "branches", "branch_length",
            "thickness", "crystal_sides", "tip_height", "name", "material",
            "material_weights", "origin", "diameter_min", "diameter_max",
            "size_distribution"};
        for (auto it = input.begin(); it != input.end(); ++it) {
            if (std::find(keys.begin(), keys.end(), it.key()) == keys.end()) {
                throw std::runtime_error("unknown grain generator setting: " + it.key());
            }
        }
        GrainGeneratorSettings candidate;
        candidate.type = input.value("type", candidate.type);
        if (candidate.type != "rock" && candidate.type != "voronoi" &&
            candidate.type != "snow" && candidate.type != "crystal") {
            throw std::runtime_error("type must be rock, voronoi, snow or crystal");
        }
        auto integer = [&](const char* key, int fallback, int low, int high) {
            if (!input.contains(key)) {
                return fallback;
            }
            if (!input.at(key).is_number_integer()) {
                throw std::runtime_error(std::string(key) + " must be an integer");
            }
            const auto value = input.at(key).get<int64_t>();
            if (value < low || value > high) {
                throw std::runtime_error(std::string(key) + " outside allowed range");
            }
            return static_cast<int>(value);
        };
        auto scalar = [&](const char* key, float fallback, float low, float high) {
            if (!input.contains(key)) {
                return fallback;
            }
            if (!input.at(key).is_number()) {
                throw std::runtime_error(std::string(key) + " must be numeric");
            }
            const float value = input.at(key).get<float>();
            if (!std::isfinite(value) || value < low || value > high) {
                throw std::runtime_error(std::string(key) + " outside allowed range");
            }
            return value;
        };
        candidate.count = integer("count", 12, 1, 64);
        candidate.seed = static_cast<uint32_t>(integer("seed", 1337, 0, 2147483647));
        candidate.resolution = integer("resolution", 4, 2, 16);
        candidate.roundness = scalar("roundness", candidate.type == "rock" ? 1.0f : 0.0f, 0, 1);
        candidate.roughness = scalar("roughness", 0.25f, 0, 0.5f);
        candidate.arms = integer("arms", 6, 3, 12);
        candidate.branches = integer("branches", 3, 0, 5);
        candidate.branch_length = scalar("branch_length", 0.15f, 0.001f, 0.4f);
        candidate.thickness = scalar("thickness", 0.04f, 0.005f, 0.3f);
        candidate.crystal_sides = integer("crystal_sides", 6, 3, 12);
        candidate.tip_height = scalar("tip_height", 0.3f, 0, 0.9f);
        if (input.contains("aspect")) {
            const auto& aspect = input.at("aspect");
            if (!aspect.is_array() || aspect.size() != 3) {
                throw std::runtime_error("aspect must be three positive numbers");
            }
            float* components[] = {&candidate.aspect.x, &candidate.aspect.y, &candidate.aspect.z};
            for (int axis = 0; axis < 3; ++axis) {
                if (!aspect[axis].is_number()) {
                    throw std::runtime_error("aspect must be numeric");
                }
                *components[axis] = aspect[axis].get<float>();
                if (!std::isfinite(*components[axis]) || *components[axis] < 0.05f ||
                    *components[axis] > 5.0f) {
                    throw std::runtime_error("aspect components must be in [0.05, 5]");
                }
            }
        }
        settings = candidate;
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool generateGrainShapes(const GrainGeneratorSettings& settings,
    std::vector<GrainExemplar>& shapes, std::string& error) {
    shapes.clear();
    error.clear();
    // Typed callers receive exactly the same validation as IPC/Python callers.
    GrainGeneratorSettings checked;
    if (!parseGrainGeneratorSettings({{"type", settings.type}, {"count", settings.count},
            {"seed", settings.seed}, {"resolution", settings.resolution},
            {"roundness", settings.roundness}, {"roughness", settings.roughness},
            {"aspect", {settings.aspect.x, settings.aspect.y, settings.aspect.z}},
            {"arms", settings.arms}, {"branches", settings.branches},
            {"branch_length", settings.branch_length}, {"thickness", settings.thickness},
            {"crystal_sides", settings.crystal_sides}, {"tip_height", settings.tip_height}},
            checked, error)) {
        return false;
    }
    if (settings.type == "rock") {
        buildRockLibrary(settings, shapes);
    } else if (settings.type == "voronoi") {
        if (!buildVoronoiLibrary(settings, shapes, error)) {
            return false;
        }
    } else {
        for (int index = 0; index < settings.count; ++index) {
            GrainExemplar shape;
            if (settings.type == "crystal") {
                shape = crystal(settings, index);
            } else if (!snow(settings, index, shape)) {
                shapes.clear();
                error = "snow outline could not be triangulated; no objects were created";
                return false;
            }
            shapes.push_back(std::move(shape));
        }
    }
    for (std::size_t index = 0; index < shapes.size(); ++index) {
        auto& shape = shapes[index];
        shape.node = settings.type + " " + std::to_string(index);
        for (auto& triangle : shape.faces) {
            for (auto& point : triangle.p) {
                point = Vec3(point.x * settings.aspect.x, point.y * settings.aspect.y,
                    point.z * settings.aspect.z);
            }
            const Vec3 cross = Vec3::cross(triangle.p[1] - triangle.p[0],
                triangle.p[2] - triangle.p[0]);
            const Vec3 normal = cross * (1.0f / std::max(cross.length(), 1e-20f));
            for (auto& value : triangle.n) {
                if (settings.type == "voronoi" && settings.roundness > 0.0f) {
                    value = Vec3(value.x / settings.aspect.x, value.y / settings.aspect.y,
                        value.z / settings.aspect.z).normalize();
                } else {
                    value = normal;
                }
            }
        }
        normalizeGrainShape(shape);
        for (auto& triangle : shape.faces) {
            float minimum_u = 1.0f;
            float maximum_u = 0.0f;
            for (int corner = 0; corner < 3; ++corner) {
                const Vec3 point = triangle.p[corner];
                if (settings.type == "snow") {
                    triangle.uv[corner] = Vec2(point.x + 0.5f, point.z + 0.5f);
                } else {
                    const float radius = std::max(point.length(), 1e-12f);
                    triangle.uv[corner] = Vec2(
                        0.5f + std::atan2(point.z, point.x) / (2.0f * pi),
                        0.5f + std::asin(std::clamp(point.y / radius, -1.0f, 1.0f)) / pi);
                    minimum_u = std::min(minimum_u, triangle.uv[corner].u);
                    maximum_u = std::max(maximum_u, triangle.uv[corner].u);
                }
            }
            if (settings.type != "snow" && maximum_u - minimum_u > 0.5f) {
                for (auto& uv : triangle.uv) {
                    if (uv.u < 0.5f) {
                        uv.u += 1.0f;
                    }
                }
            }
        }
    }
    return !shapes.empty();
}

} // namespace RayTrophiSim
