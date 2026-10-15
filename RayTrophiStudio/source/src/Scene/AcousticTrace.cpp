#include "Audio/AcousticScene.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace audio {
namespace {
using Point = AcousticPoint;
using json = nlohmann::json;
Point subtract(Point a, Point b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Point add(Point a, Point b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
Point scale(Point p, double value) {
    return {p.x * value, p.y * value, p.z * value};
}
double dot(Point a, Point b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
Point cross(Point a, Point b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double axis(Point p, unsigned index) {
    return index == 0 ? p.x : index == 1 ? p.y : p.z;
}
json vector(Point p) {
    return {p.x, p.y, p.z};
}
struct Bounds {
    Point low{1e300, 1e300, 1e300};
    Point high{-1e300, -1e300, -1e300};
    void include(Point p) {
        low = {std::min(low.x, p.x), std::min(low.y, p.y), std::min(low.z, p.z)};
        high = {std::max(high.x, p.x), std::max(high.y, p.y), std::max(high.z, p.z)};
    }
    bool intersects(Point origin, Point direction) const {
        double near = 0, far = 1;
        for (unsigned index = 0; index < 3; ++index) {
            const double d = axis(direction, index), o = axis(origin, index);
            if (std::abs(d) < 1e-20) {
                if (o < axis(low, index) || o > axis(high, index)) {
                    return false;
                }
            } else {
                const double a = (axis(low, index) - o) / d;
                const double b = (axis(high, index) - o) / d;
                near = std::max(near, std::min(a, b));
                far = std::min(far, std::max(a, b));
                if (near > far) {
                    return false;
                }
            }
        }
        return true;
    }
};
struct Hit {
    double fraction;
    Point normal;
    std::size_t object;
};
struct Node {
    Bounds bounds;
    std::size_t begin = 0, end = 0, left = 0, right = 0;
};

class SegmentBvh {
public:
    explicit SegmentBvh(const AcousticSceneSnapshot& snapshot) : scene(snapshot) {
        order.resize(scene.faces.size());
        std::iota(order.begin(), order.end(), 0);
        nodes.reserve(order.size() * 2);
        if (!order.empty()) {
            build(0, order.size());
        }
    }

    std::vector<Hit> hits(Point source, Point listener, double epsilon, unsigned maximum) const {
        std::vector<Hit> result;
        if (nodes.empty()) {
            return result;
        }
        const auto direction = subtract(listener, source);
        std::vector<std::size_t> stack{0};
        while (!stack.empty()) {
            const auto index = stack.back();
            stack.pop_back();
            const auto& node = nodes[index];
            if (!node.bounds.intersects(source, direction)) {
                continue;
            }
            if (node.left != node.right) {
                stack.push_back(node.left);
                stack.push_back(node.right);
                continue;
            }
            for (auto item = node.begin; item < node.end; ++item) {
                const auto& face = scene.faces[order[item]];
                const auto e1 = subtract(face.b, face.a), e2 = subtract(face.c, face.a);
                const auto normal = cross(e1, e2);
                const double area = std::sqrt(dot(normal, normal));
                if (area == 0) {
                    continue;
                }
                const auto h = cross(direction, e2);
                const double determinant = dot(e1, h);
                if (std::abs(determinant) <= 1e-12 * area *
                    std::sqrt(dot(direction, direction))) {
                    continue;
                }
                const auto offset = subtract(source, face.a);
                const double u = dot(offset, h) / determinant;
                const auto q = cross(offset, e1);
                const double v = dot(direction, q) / determinant;
                const double t = dot(e2, q) / determinant;
                if (u < -1e-10 || v < -1e-10 || u + v > 1 + 1e-10 ||
                    t <= epsilon || t >= 1 - epsilon) {
                    continue;
                }
                if (result.size() >= maximum * 8) {
                    throw std::invalid_argument("Acoustic raw-intersection budget exceeded");
                }
                result.push_back({t, scale(normal, 1 / area), face.object});
            }
        }
        std::sort(result.begin(), result.end(), [](const Hit& a, const Hit& b) {
            if (a.fraction != b.fraction) {
                return a.fraction < b.fraction;
            }
            return a.object < b.object;
        });
        std::vector<Hit> unique;
        for (const auto& hit : result) {
            const bool duplicate = std::any_of(unique.rbegin(), unique.rend(), [&](const Hit& old) {
                return old.object == hit.object && std::abs(old.fraction - hit.fraction) <= epsilon &&
                    std::abs(dot(old.normal, hit.normal)) >= 1 - 1e-6;
            });
            if (!duplicate) {
                unique.push_back(hit);
            }
        }
        if (unique.size() > maximum) {
            throw std::invalid_argument("Acoustic intersection budget exceeded");
        }
        return unique;
    }
private:
    const AcousticSceneSnapshot& scene;
    std::vector<std::size_t> order;
    std::vector<Node> nodes;

    std::size_t build(std::size_t begin, std::size_t end) {
        const auto index = nodes.size();
        nodes.push_back({});
        Bounds bounds;
        for (auto item = begin; item < end; ++item) {
            const auto& face = scene.faces[order[item]];
            bounds.include(face.a);
            bounds.include(face.b);
            bounds.include(face.c);
        }
        nodes[index].bounds = bounds;
        nodes[index].begin = begin;
        nodes[index].end = end;
        if (end - begin <= 8) {
            return index;
        }
        const auto extent = subtract(bounds.high, bounds.low);
        const unsigned split = extent.x >= extent.y && extent.x >= extent.z ? 0 :
            extent.y >= extent.z ? 1 : 2;
        const auto middle = begin + (end - begin) / 2;
        std::nth_element(order.begin() + begin, order.begin() + middle, order.begin() + end,
            [&](std::size_t a, std::size_t b) {
                const auto& first = scene.faces[a];
                const auto& second = scene.faces[b];
                return axis(add(add(first.a, first.b), first.c), split) <
                    axis(add(add(second.a, second.b), second.c), split);
            });
        const auto left = build(begin, middle), right = build(middle, end);
        nodes[index].left = left;
        nodes[index].right = right;
        return index;
    }
};
}

nlohmann::json traceAcousticSegment(const AcousticSceneSnapshot& scene, Point source,
    Point listener, double metersPerUnit, double soundSpeed, unsigned maxHits, bool thinSurfaces, bool objectBarriers) {
    if (!std::isfinite(metersPerUnit) || metersPerUnit < 1e-6 || metersPerUnit > 1e6 ||
        !std::isfinite(soundSpeed) || soundSpeed < 100 || soundSpeed > 2000 ||
        maxHits < 1 || maxHits > 256) {
        throw std::invalid_argument("Invalid acoustic trace units, sound speed or hit budget");
    }
    const auto direction = subtract(listener, source);
    const double length = std::sqrt(dot(direction, direction));
    const double distance = length * metersPerUnit;
    if (!std::isfinite(distance) || distance < 1e-4) {
        throw std::invalid_argument("Acoustic segment must be at least 0.1 mm long");
    }
    const SegmentBvh bvh(scene);
    const double epsilon = std::max(64 * std::numeric_limits<double>::epsilon(), 1e-5 / distance);
    const auto hits = bvh.hits(source, listener, epsilon, maxHits);
    const bool covered = scene.unsupported == 0 && scene.transient == 0 && scene.hidden == 0;
    json result = {{"source_position", vector(source)}, {"listener_position", vector(listener)},
        {"distance_meters", distance}, {"meters_per_unit", metersPerUnit},
        {"sound_speed_mps", soundSpeed}, {"delay_seconds", distance / soundSpeed},
        {"geometrically_obstructed", !hits.empty()}, {"triangles", scene.faces.size()},
        {"coverage", "authored_flat_mesh_only"}, {"snapshot_complete", covered},
        {"unsupported_objects", scene.unsupported}, {"transient_objects", scene.transient},
        {"hidden_objects_skipped", scene.hidden},
        {"hits", json::array()}, {"transmission_energy", nullptr},
        {"transmission_status", "requires_boundary_model"}};
    std::array<double, 6> transmission{1, 1, 1, 1, 1, 1};
    bool known = covered;
    bool ambiguous = false;
    for (std::size_t index = 1; index < hits.size(); ++index) {
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (hits[index].object == hits[previous].object &&
                std::abs(hits[index].fraction - hits[previous].fraction) <= epsilon) {
                ambiguous = true;
            }
        }
    }
    known = known && !ambiguous;
    std::vector<std::size_t> barriers;
    for (const auto& hit : hits) {
        const auto& object = scene.objects.at(hit.object);
        const auto response = object.value("acoustics", json::object());
        const bool defined = response.value("defined", false);
        known = known && defined;
        result["hits"].push_back({{"distance_meters", hit.fraction * distance},
            {"position", vector(add(source, scale(direction, hit.fraction)))},
            {"normal", vector(hit.normal)}, {"material", object}});
        const bool firstBarrier = std::find(barriers.begin(), barriers.end(), hit.object) ==
            barriers.end();
        barriers.push_back(hit.object);
        if (defined && (!objectBarriers || firstBarrier)) {
            for (unsigned band = 0; band < 6; ++band) {
                transmission[band] *= response.at("bands").at(band).at("transmission").get<double>();
            }
        }
    }
    if (!covered) {
        result["transmission_status"] = "unsupported_geometry";
    } else if (ambiguous) {
        result["transmission_status"] = "ambiguous_boundary";
    } else if (!known) {
        result["transmission_status"] = "unknown_material";
    } else if (hits.empty() || thinSurfaces || objectBarriers) {
        result["transmission_energy"] = transmission;
        result["transmission_status"] = hits.empty() ? "clear" :
            objectBarriers ? "evaluated_object_barrier" : "evaluated_thin_surface";
    }
    result["frequencies_hz"] = {125, 250, 500, 1000, 2000, 4000};
    result["object_barrier_assumption"] = objectBarriers;
    result["thin_surface_assumption"] = thinSurfaces;
    result["ambiguous_boundary"] = ambiguous;
    return result;
}
} // namespace audio
