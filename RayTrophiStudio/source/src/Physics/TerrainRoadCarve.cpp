#include "TerrainRoadCarve.h"
#include "TerrainRoadProfile.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace TerrainNodesV2 {
namespace {

float saturate(float value) {
    return (std::max)(0.0f, (std::min)(1.0f, value));
}

float smoothCoverage(float distance, float radius, float falloff) {
    if (distance <= radius) return 1.0f;
    if (falloff <= 1.0e-6f || distance >= radius + falloff) return 0.0f;
    const float t = saturate((distance - radius) / falloff);
    return 1.0f - t * t * (3.0f - 2.0f * t);
}

float sampleField(const NodeSystem::Image2DData& image, float x, float z,
                  float terrainScaleXZ) {
    const float gx = saturate(x / terrainScaleXZ) * static_cast<float>(image.width - 1);
    const float gy = saturate(z / terrainScaleXZ) * static_cast<float>(image.height - 1);
    const int x0 = static_cast<int>(std::floor(gx));
    const int y0 = static_cast<int>(std::floor(gy));
    const int x1 = (std::min)(x0 + 1, image.width - 1);
    const int y1 = (std::min)(y0 + 1, image.height - 1);
    const float tx = gx - static_cast<float>(x0);
    const float ty = gy - static_cast<float>(y0);
    const auto at = [&](int px, int py) {
        return (*image.data)[(static_cast<size_t>(py) * image.width + px) * image.channels];
    };
    const float a = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
    const float b = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
    return a + (b - a) * ty;
}

NodeSystem::Image2DData makeImage(int width, int height,
                                  NodeSystem::ImageSemantic semantic,
                                  NodeSystem::ImageUnit unit) {
    NodeSystem::Image2DData image;
    image.width = width;
    image.height = height;
    image.channels = 1;
    image.semantic = semantic;
    image.unit = unit;
    image.data = std::make_shared<std::vector<float>>(
        static_cast<size_t>(width) * height, 0.0f);
    return image;
}

bool ditchEnabled(const RoadCarveSettings& settings) {
    return settings.ditchWidthMeters > 1.0e-3f && settings.ditchDepthMeters > 1.0e-3f;
}

// The road's shape across its own width, in metres relative to the route height.
//
// A road without this is a flat linear TRENCH, and a flat linear trench is a
// perfect channel to any flow solver that reads the carved height afterwards -
// which is exactly why carved roads were being classified as river beds. A
// crowned surface sheds water sideways and the ditch carries it downhill: the
// road pushes water, the ditch moves it. That is the authority declaration,
// made as GEOMETRY rather than as a flag, and it is what real road engineering
// does for the same reason.
float crossSectionMeters(float distance, float coreRadius, float shoulderWidth,
                         float crown, float ditchWidth, float ditchDepth) {
    if (coreRadius > 1.0e-4f && distance <= coreRadius) {
        const float t = distance / coreRadius;
        return crown * (1.0f - t * t);
    }
    const float shoulderEdge = coreRadius + shoulderWidth;
    if (distance <= shoulderEdge) return 0.0f;
    if (ditchWidth <= 1.0e-3f || ditchDepth <= 1.0e-3f) return 0.0f;
    const float u = (distance - shoulderEdge) / ditchWidth;
    if (u >= 1.0f) return 0.0f;
    return -ditchDepth * 4.0f * u * (1.0f - u);
}

float ditchCoverage(float distance, float coreRadius, float shoulderWidth,
                    float ditchWidth, float ditchDepth) {
    if (ditchWidth <= 1.0e-3f || ditchDepth <= 1.0e-3f) return 0.0f;
    const float shoulderEdge = coreRadius + shoulderWidth;
    if (distance <= shoulderEdge) return 0.0f;
    const float u = (distance - shoulderEdge) / ditchWidth;
    if (u >= 1.0f) return 0.0f;
    return saturate(4.0f * u * (1.0f - u));
}

// Grade limiter only - no envelope. This is the alignment the road WANTS: how
// close to the ground a vehicle-legal profile can stay. The difference between
// it and the ground is what tells the solver where the ground cannot carry the
// road at all, which is the only honest place to put a bridge or a tunnel.
void applyGradeLimit(const std::vector<RoadRouteSample>& route,
                     float gradeRatio, float terrainScaleY, int passes,
                     std::vector<float>& profile) {
    for (int pass = 0; pass < passes; ++pass) {
        for (size_t i = 1; i < route.size(); ++i) {
            const float limit = gradeRatio *
                (route[i].distanceMeters - route[i - 1].distanceMeters) / terrainScaleY;
            profile[i] = (std::max)(profile[i - 1] - limit,
                (std::min)(profile[i - 1] + limit, profile[i]));
        }
        for (size_t i = route.size() - 1; i > 0; --i) {
            const float limit = gradeRatio *
                (route[i].distanceMeters - route[i - 1].distanceMeters) / terrainScaleY;
            profile[i - 1] = (std::max)(profile[i] - limit,
                (std::min)(profile[i] + limit, profile[i - 1]));
        }
    }
}

// Grow every declared span by one road width and merge spans that end up closer
// than that. A bridge that starts exactly where the fill limit binds has no
// abutment, and two spans separated by three samples of embankment is not two
// bridges - it is one with a pothole in the middle.
void dilateSpans(const std::vector<RoadRouteSample>& route,
                 float marginMeters,
                 std::vector<uint8_t>& flags) {
    if (marginMeters <= 0.0f) return;
    const size_t count = flags.size();
    std::vector<uint8_t> grown(count, 0);
    for (size_t i = 0; i < count; ++i) {
        if (!flags[i]) continue;
        for (size_t j = i; j < count; ++j) {
            if (route[j].distanceMeters - route[i].distanceMeters > marginMeters) break;
            grown[j] = 1;
        }
        for (size_t j = i + 1; j-- > 0;) {
            if (route[i].distanceMeters - route[j].distanceMeters > marginMeters) break;
            grown[j] = 1;
        }
    }
    flags.swap(grown);
}

struct RouteSolveOutput {
    std::vector<float> height;
    std::vector<RoadCrossingMode> crossing;
    int envelopeClampedSamples = 0;
    int gradeExceededSamples = 0;
    int bridgeSamples = 0;
    int tunnelSamples = 0;
    int fordSamples = 0;
    std::string diagnostic;
};

// One road's longitudinal profile: ground-following, grade limited, bounded by
// the cut/fill envelope, and interrupted where the author declared a crossing.
//
// The envelope is not a refinement - it is what makes the solve a ROAD. A grade
// limiter alone is unbounded: a 12% road crossing a mountain cannot climb fast
// enough, so the profile stays near its entry height and the rasterizer builds
// exactly what that implies, a canyon through the peak and a mountain-high
// embankment across the valley.
void solveRouteProfile(const NodeSystem::Image2DData& sourceHeight,
                       const NodeSystem::Image2DData* waterMask,
                       float terrainScaleXZ,
                       float terrainScaleY,
                       std::vector<RoadRouteSample>& route,
                       const RoadCarveSettings& settings,
                       RoadCrossingMode declaredCrossing,
                       const std::string& label,
                       RouteSolveOutput& out) {
    const size_t count = route.size();
    std::vector<float> groundHeight(count, 0.0f);
    for (size_t i = 0; i < count; ++i) {
        groundHeight[i] = sampleField(sourceHeight, route[i].x, route[i].z, terrainScaleXZ);
        route[i].groundHeight = groundHeight[i];
    }

    const float gradeRatio = settings.maxGradePercent * 0.01f;
    const float cutLimit = settings.maxCutMeters / terrainScaleY;
    const float fillLimit = settings.maxFillMeters / terrainScaleY;
    const float offset = settings.elevationOffsetMeters / terrainScaleY;

    // The alignment the road wants, before anything pins it to the ground.
    std::vector<float> ideal(count, 0.0f);
    for (size_t i = 0; i < count; ++i) ideal[i] = groundHeight[i] + offset;
    applyGradeLimit(route, gradeRatio, terrainScaleY, 4, ideal);

    const bool haveWater = waterMask && waterMask->isValid() && waterMask->data &&
                           waterMask->width > 1 && waterMask->height > 1;
    std::vector<uint8_t> overWater(count, 0);
    if (haveWater) {
        for (size_t i = 0; i < count; ++i) {
            overWater[i] = sampleField(*waterMask, route[i].x, route[i].z,
                                       terrainScaleXZ) > 0.5f ? 1 : 0;
        }
    }

    std::vector<uint8_t> bridge(count, 0), tunnel(count, 0), ford(count, 0);
    switch (declaredCrossing) {
        case RoadCrossingMode::Terrain:
            break;
        case RoadCrossingMode::Auto:
            // Auto acts only on what it can actually observe. Promoting a
            // deep-fill sample to a bridge here would replace the cut/fill
            // envelope on every route in the scene by default, and terrain that
            // nobody asked to bridge would silently stop being graded.
            if (haveWater) {
                for (size_t i = 0; i < count; ++i) bridge[i] = overWater[i];
            }
            break;
        case RoadCrossingMode::Bridge:
            for (size_t i = 0; i < count; ++i) {
                bridge[i] = (overWater[i] || (ideal[i] - groundHeight[i]) > fillLimit) ? 1 : 0;
            }
            break;
        case RoadCrossingMode::Tunnel:
            for (size_t i = 0; i < count; ++i) {
                tunnel[i] = ((groundHeight[i] - ideal[i]) > cutLimit) ? 1 : 0;
            }
            break;
        case RoadCrossingMode::Ford:
            if (!haveWater) {
                out.diagnostic = (label.empty() ? std::string() : "'" + label + "': ") +
                    "Ford needs a Water field; no crossing was resolved";
            } else {
                for (size_t i = 0; i < count; ++i) ford[i] = overWater[i];
            }
            break;
    }

    const float abutment = (std::max)(settings.roadWidthMeters, 1.0f);
    dilateSpans(route, abutment, bridge);
    dilateSpans(route, abutment, tunnel);
    // A ford is deliberately NOT dilated: its whole point is that the road only
    // touches the water where it must. Widening it would grade the banks.

    out.crossing.assign(count, RoadCrossingMode::Terrain);
    for (size_t i = 0; i < count; ++i) {
        if (tunnel[i])      { out.crossing[i] = RoadCrossingMode::Tunnel; ++out.tunnelSamples; }
        else if (bridge[i]) { out.crossing[i] = RoadCrossingMode::Bridge; ++out.bridgeSamples; }
        else if (ford[i])   { out.crossing[i] = RoadCrossingMode::Ford;   ++out.fordSamples; }
    }

    std::vector<float>& routeHeight = out.height;
    routeHeight.assign(count, 0.0f);
    for (size_t i = 0; i < count; ++i) routeHeight[i] = groundHeight[i] + offset;

    // Two constraints that can disagree: the grade limiter smooths the profile
    // along the route, the envelope pins it to the ground. Applying each once
    // lets the second undo the first. Both pull toward the ground, so alternating
    // them converges; the pass count is fixed so a pathological route cannot spin.
    //
    // Where they genuinely conflict - a cliff no 12% road can climb inside its cut
    // budget - the ENVELOPE wins. A slightly too-steep road is a road; a road
    // hanging in the air is the bug this exists to prevent. That case is counted
    // and reported rather than silently accepted.
    constexpr int kProfilePasses = 8;
    for (int pass = 0; pass < kProfilePasses; ++pass) {
        applyGradeLimit(route, gradeRatio, terrainScaleY, 1, routeHeight);
        for (size_t i = 0; i < count; ++i) {
            switch (out.crossing[i]) {
                case RoadCrossingMode::Ford:
                    // A ford does not float over the water: it follows the bed.
                    routeHeight[i] = groundHeight[i];
                    break;
                case RoadCrossingMode::Bridge:
                case RoadCrossingMode::Tunnel:
                    // Free of the envelope - that IS the crossing.
                    break;
                default:
                    routeHeight[i] = (std::max)(groundHeight[i] - cutLimit,
                        (std::min)(groundHeight[i] + fillLimit, routeHeight[i]));
                    break;
            }
        }
    }

    // A deck and a bore run straight through. Interpolating between the samples
    // on either side of the span is what makes a bridge a bridge instead of a
    // grade-limited curve that merely escaped the envelope.
    for (size_t i = 0; i < count;) {
        const bool spanned = out.crossing[i] == RoadCrossingMode::Bridge ||
                             out.crossing[i] == RoadCrossingMode::Tunnel;
        if (!spanned) { ++i; continue; }
        size_t end = i;
        while (end + 1 < count && (out.crossing[end + 1] == RoadCrossingMode::Bridge ||
                                   out.crossing[end + 1] == RoadCrossingMode::Tunnel)) {
            ++end;
        }
        const size_t before = i > 0 ? i - 1 : i;
        const size_t after = end + 1 < count ? end + 1 : end;
        const float startHeight = routeHeight[before];
        const float endHeight = routeHeight[after];
        const float startDistance = route[before].distanceMeters;
        const float span = route[after].distanceMeters - startDistance;
        for (size_t j = i; j <= end; ++j) {
            const float t = span > 1.0e-6f
                ? saturate((route[j].distanceMeters - startDistance) / span) : 0.0f;
            routeHeight[j] = startHeight + (endHeight - startHeight) * t;
        }
        i = end + 1;
    }

    const float envelopeEpsilon = 1.0e-4f;
    for (size_t i = 0; i < count; ++i) {
        route[i].height = routeHeight[i];
        route[i].crossing = out.crossing[i];
        if (out.crossing[i] == RoadCrossingMode::Bridge ||
            out.crossing[i] == RoadCrossingMode::Tunnel) continue;
        const float deviation = routeHeight[i] - groundHeight[i];
        if (deviation <= -cutLimit + envelopeEpsilon ||
            deviation >= fillLimit - envelopeEpsilon) {
            ++out.envelopeClampedSamples;
        }
        if (i == 0) continue;
        const float distanceSpan = route[i].distanceMeters - route[i - 1].distanceMeters;
        if (distanceSpan <= 1.0e-6f) continue;
        const float actualGrade =
            std::fabs(routeHeight[i] - routeHeight[i - 1]) * terrainScaleY / distanceSpan;
        if (actualGrade > gradeRatio + 1.0e-4f) ++out.gradeExceededSamples;
    }
}

} // namespace

bool solveRoadNetworkCarve(const NodeSystem::Image2DData& sourceHeight,
                           float terrainScaleXZ,
                           float terrainScaleY,
                           const std::vector<RoadCarveInput>& roads,
                           const Matrix4x4& terrainWorldToLocal,
                           const NodeSystem::Image2DData* waterMask,
                           uint64_t revision,
                           RoadCarveResult& result,
                           std::string* error) {
    if (!sourceHeight.isValid() || sourceHeight.channels < 1 ||
        sourceHeight.semantic != NodeSystem::ImageSemantic::Height) {
        if (error) *error = "Height input must be a valid height field";
        return false;
    }
    if (sourceHeight.width < 2 || sourceHeight.height < 2 ||
        !std::isfinite(terrainScaleXZ) || terrainScaleXZ <= 0.0f ||
        !std::isfinite(terrainScaleY) || terrainScaleY <= 0.0f) {
        if (error) *error = "invalid terrain dimensions or scale";
        return false;
    }
    if (roads.empty()) {
        if (error) *error = "no roads to carve";
        return false;
    }

    const int width = sourceHeight.width;
    const int height = sourceHeight.height;
    const float pixelX = terrainScaleXZ / static_cast<float>(width - 1);
    const float pixelZ = terrainScaleXZ / static_cast<float>(height - 1);
    const size_t pixelCount = static_cast<size_t>(width) * height;

    // Every road writes into ONE nearest-distance field. Overlaps therefore
    // resolve by proximity - the closest road owns the pixel - so a junction is
    // carved once instead of once per branch, and a narrow path crossing a wide
    // road does not punch a groove through it.
    std::vector<float> nearest(pixelCount, (std::numeric_limits<float>::max)());
    std::vector<float> target(pixelCount, 0.0f);
    std::vector<float> coreRadius(pixelCount, 0.0f);
    std::vector<float> shoulderWidth(pixelCount, 0.0f);
    std::vector<float> gradingFalloff(pixelCount, 0.0f);
    std::vector<float> exclusionMargin(pixelCount, 0.0f);
    std::vector<float> crown(pixelCount, 0.0f);
    std::vector<float> ditchWidth(pixelCount, 0.0f);
    std::vector<float> ditchDepth(pixelCount, 0.0f);
    std::vector<uint8_t> crossing(pixelCount,
                                  static_cast<uint8_t>(RoadCrossingMode::Terrain));

    result = {};
    result.revision = revision;

    for (const RoadCarveInput& road : roads) {
        if (!road.curve) {
            if (error) *error = road.label.empty()
                ? std::string("road has no curve")
                : ("'" + road.label + "': road has no curve");
            return false;
        }
        if (!validateRoadCarveSettings(road.settings, road.label, error)) return false;

        std::vector<TerrainCurveSample> sampled;
        std::string sampleError;
        if (!sampleTerrainCurveXZ(*road.curve, terrainWorldToLocal,
                                  (std::min)(pixelX, pixelZ), road.settings.usePointWidth,
                                  sampled, &sampleError) || sampled.size() < 2) {
            if (error) {
                const std::string detail = sampleError.empty()
                    ? std::string("curve produced fewer than two route samples") : sampleError;
                *error = road.label.empty() ? detail : ("'" + road.label + "': " + detail);
            }
            return false;
        }

        RoadSolvedRoute solved;
        solved.label = road.label;
        solved.settings = road.settings;
        solved.declaredCrossing = road.crossing;
        solved.samples.resize(sampled.size());
        for (size_t i = 0; i < sampled.size(); ++i) {
            solved.samples[i].x = sampled[i].x;
            solved.samples[i].z = sampled[i].z;
            solved.samples[i].widthMultiplier = sampled[i].widthMultiplier;
            solved.samples[i].distanceMeters = sampled[i].distanceMeters;
        }

        RouteSolveOutput routeOut;
        solveRouteProfile(sourceHeight, waterMask, terrainScaleXZ, terrainScaleY,
                          solved.samples, road.settings, road.crossing, road.label,
                          routeOut);
        result.envelopeClampedSamples += routeOut.envelopeClampedSamples;
        result.gradeExceededSamples += routeOut.gradeExceededSamples;
        result.bridgeSamples += routeOut.bridgeSamples;
        result.tunnelSamples += routeOut.tunnelSamples;
        result.fordSamples += routeOut.fordSamples;
        result.routeSampleCount += static_cast<int>(solved.samples.size());
        if (!routeOut.diagnostic.empty()) {
            if (!result.crossingDiagnostic.empty()) result.crossingDiagnostic += "; ";
            result.crossingDiagnostic += routeOut.diagnostic;
        }
        ++result.roadCount;

        const bool hasDitch = ditchEnabled(road.settings);
        const auto& route = solved.samples;
        for (size_t segment = 0; segment + 1 < route.size(); ++segment) {
            const auto& a = route[segment];
            const auto& b = route[segment + 1];
            const float reach = road.settings.roadWidthMeters * 0.5f *
                (std::max)(a.widthMultiplier, b.widthMultiplier) +
                road.settings.shoulderWidthMeters +
                (hasDitch ? road.settings.ditchWidthMeters : 0.0f) +
                road.settings.foliageExclusionMarginMeters +
                road.settings.gradingFalloffMeters;
            const int minX = (std::max)(0, static_cast<int>(std::floor(((std::min)(a.x, b.x) - reach) / pixelX)));
            const int maxX = (std::min)(width - 1, static_cast<int>(std::ceil(((std::max)(a.x, b.x) + reach) / pixelX)));
            const int minY = (std::max)(0, static_cast<int>(std::floor(((std::min)(a.z, b.z) - reach) / pixelZ)));
            const int maxY = (std::min)(height - 1, static_cast<int>(std::ceil(((std::max)(a.z, b.z) + reach) / pixelZ)));
            const float dx = b.x - a.x;
            const float dz = b.z - a.z;
            const float lengthSquared = dx * dx + dz * dz;
            if (lengthSquared <= 1.0e-12f) continue;
            for (int y = minY; y <= maxY; ++y) {
                const float pz = static_cast<float>(y) * pixelZ;
                for (int x = minX; x <= maxX; ++x) {
                    const float px = static_cast<float>(x) * pixelX;
                    const float projection = saturate(((px - a.x) * dx + (pz - a.z) * dz) / lengthSquared);
                    const float qx = a.x + dx * projection;
                    const float qz = a.z + dz * projection;
                    const float ddx = px - qx;
                    const float ddz = pz - qz;
                    const float distance = std::sqrt(ddx * ddx + ddz * ddz);
                    const size_t index = static_cast<size_t>(y) * width + x;
                    if (distance >= nearest[index]) continue;
                    nearest[index] = distance;
                    target[index] = a.height + (b.height - a.height) * projection;
                    const float multiplier = a.widthMultiplier +
                        (b.widthMultiplier - a.widthMultiplier) * projection;
                    coreRadius[index] = road.settings.roadWidthMeters * 0.5f * multiplier;
                    // The winning road's profile has to travel with the pixel.
                    // Reading these from a single shared settings block is what
                    // would make a footpath inherit a main road's shoulder.
                    shoulderWidth[index] = road.settings.shoulderWidthMeters;
                    gradingFalloff[index] = road.settings.gradingFalloffMeters;
                    exclusionMargin[index] = road.settings.foliageExclusionMarginMeters;
                    crown[index] = road.settings.crownMeters;
                    ditchWidth[index] = hasDitch ? road.settings.ditchWidthMeters : 0.0f;
                    ditchDepth[index] = hasDitch ? road.settings.ditchDepthMeters : 0.0f;
                    // The crossing travels with the pixel for the same reason the
                    // profile does: the nearest sample is the one that owns it.
                    crossing[index] = static_cast<uint8_t>(
                        projection < 0.5f ? a.crossing : b.crossing);
                }
            }
        }
        result.routes.push_back(std::move(solved));
    }

    result.height = makeImage(width, height, NodeSystem::ImageSemantic::Height,
                              sourceHeight.unit);
    result.roadCore = makeImage(width, height, NodeSystem::ImageSemantic::Mask,
                                NodeSystem::ImageUnit::Unitless);
    result.shoulder = makeImage(width, height, NodeSystem::ImageSemantic::Mask,
                                NodeSystem::ImageUnit::Unitless);
    result.ditch = makeImage(width, height, NodeSystem::ImageSemantic::Mask,
                             NodeSystem::ImageUnit::Unitless);
    result.cut = makeImage(width, height, NodeSystem::ImageSemantic::PhysicalScalar,
                           NodeSystem::ImageUnit::Meters);
    result.fill = makeImage(width, height, NodeSystem::ImageSemantic::PhysicalScalar,
                            NodeSystem::ImageUnit::Meters);
    result.foliageExclusion = makeImage(width, height, NodeSystem::ImageSemantic::Mask,
                                        NodeSystem::ImageUnit::Unitless);

    for (size_t i = 0; i < pixelCount; ++i) {
        const float original = (*sourceHeight.data)[i * sourceHeight.channels];
        if (nearest[i] == (std::numeric_limits<float>::max)()) {
            (*result.height.data)[i] = original;
            continue;
        }
        const auto mode = static_cast<RoadCrossingMode>(crossing[i]);
        // A tunnel is the one crossing that leaves NOTHING on the surface: no
        // grading, no road, and no foliage exclusion either. A forest grows over
        // a tunnel, and pretending otherwise would strip a hillside of trees for
        // a road that is not there.
        if (mode == RoadCrossingMode::Tunnel) {
            (*result.height.data)[i] = original;
            continue;
        }

        const bool ford = mode == RoadCrossingMode::Ford;
        const bool bridge = mode == RoadCrossingMode::Bridge;
        const float sectionCrown = (ford || bridge) ? 0.0f : crown[i];
        const float sectionDitchWidth = (ford || bridge) ? 0.0f : ditchWidth[i];
        const float sectionDitchDepth = (ford || bridge) ? 0.0f : ditchDepth[i];

        const float edgeFalloff = (std::min)(0.5f, gradingFalloff[i]);
        const float core = smoothCoverage(nearest[i], coreRadius[i], edgeFalloff);
        const float shoulderEdge = smoothCoverage(nearest[i],
            coreRadius[i] + shoulderWidth[i], edgeFalloff);
        const float support = coreRadius[i] + shoulderWidth[i] + sectionDitchWidth;
        const float grading = smoothCoverage(nearest[i], support, gradingFalloff[i]);
        const float exclusion = smoothCoverage(nearest[i],
            support + exclusionMargin[i], gradingFalloff[i]);

        (*result.roadCore.data)[i] = core;
        (*result.shoulder.data)[i] = (std::max)(0.0f, shoulderEdge - core);
        (*result.ditch.data)[i] = ditchCoverage(nearest[i], coreRadius[i],
            shoulderWidth[i], sectionDitchWidth, sectionDitchDepth);
        (*result.foliageExclusion.data)[i] = exclusion;

        if (bridge) {
            // The deck spans; the ground under it is not touched. That is the
            // whole contract - a bridge that grades the river bed is a culvert
            // with extra steps.
            (*result.height.data)[i] = original;
            continue;
        }

        const float sectionMeters = crossSectionMeters(nearest[i], coreRadius[i],
            shoulderWidth[i], sectionCrown, sectionDitchWidth, sectionDitchDepth);
        const float pixelTarget = target[i] + sectionMeters / terrainScaleY;
        float carved = original + (pixelTarget - original) * grading;
        // A ford wades. It may cut down to the bed it crosses, but it may never
        // fill the channel - filling it is how a crossing quietly becomes a dam.
        if (ford) carved = (std::min)(carved, original);
        (*result.height.data)[i] = carved;
        const float cutMeters = (std::max)(0.0f, original - carved) * terrainScaleY;
        const float fillMeters = (std::max)(0.0f, carved - original) * terrainScaleY;
        (*result.cut.data)[i] = cutMeters;
        (*result.fill.data)[i] = fillMeters;
        result.peakCutMeters = (std::max)(result.peakCutMeters, cutMeters);
        result.peakFillMeters = (std::max)(result.peakFillMeters, fillMeters);
    }
    return true;
}

bool solveRoadCarve(const NodeSystem::Image2DData& sourceHeight,
                    float terrainScaleXZ,
                    float terrainScaleY,
                    const MeshEdit::CurveNodeData& curve,
                    const Matrix4x4& terrainWorldToLocal,
                    const RoadCarveSettings& settings,
                    uint64_t revision,
                    RoadCarveResult& result,
                    std::string* error) {
    // The single-curve entry point is the multi-road solve with one road. Two
    // implementations of the same carve would drift, and the drift would show up
    // as "the network node and the single node disagree" long after the change
    // that caused it.
    RoadCarveInput road;
    road.curve = &curve;
    road.settings = settings;
    road.crossing = RoadCrossingMode::Terrain;
    std::vector<RoadCarveInput> roads{road};
    return solveRoadNetworkCarve(sourceHeight, terrainScaleXZ, terrainScaleY, roads,
                                 terrainWorldToLocal, nullptr, revision, result, error);
}

} // namespace TerrainNodesV2
