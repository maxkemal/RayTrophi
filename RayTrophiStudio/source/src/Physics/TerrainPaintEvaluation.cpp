#include "TerrainPaintEvaluation.h"
#include "TerrainFieldMath.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace TerrainPaintEvaluation {
namespace {

    float clamp01(float value) {
        return (std::max)(0.0f, (std::min)(1.0f, value));
    }

    float smoothstep(float edge0, float edge1, float value) {
        const float denominator = edge1 - edge0;
        const float t = clamp01((value - edge0) /
            (std::abs(denominator) > 1e-6f ? denominator : 1e-6f));
        return t * t * (3.0f - 2.0f * t);
    }

    float hashNoise(int x, int y, int seed) {
        int n = x + y * 57 + seed * 131;
        n = (n << 13) ^ n;
        return (1.0f - ((n * (n * n * 15731 + 789221) + 1376312589) &
            0x7fffffff) / 1073741824.0f) * 0.5f + 0.5f;
    }

    float patchNoise(float u, float v, float scale, int seed) {
        const float px = u * scale;
        const float py = v * scale;
        const int x0 = static_cast<int>(std::floor(px));
        const int y0 = static_cast<int>(std::floor(py));
        const float tx0 = px - static_cast<float>(x0);
        const float ty0 = py - static_cast<float>(y0);
        const float tx = tx0 * tx0 * (3.0f - 2.0f * tx0);
        const float ty = ty0 * ty0 * (3.0f - 2.0f * ty0);
        const float a = hashNoise(x0, y0, seed);
        const float b = hashNoise(x0 + 1, y0, seed);
        const float c = hashNoise(x0, y0 + 1, seed);
        const float d = hashNoise(x0 + 1, y0 + 1, seed);
        return (a + (b - a) * tx) +
            ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty;
    }

    float fractalPatchNoise(float u, float v, float scale, int seed) {
        const float baseScale = (std::max)(scale, 1.0f);
        return patchNoise(u, v, baseScale, seed) * 0.55f +
               patchNoise(u, v, baseScale * 4.0f, seed + 101) * 0.24f +
               patchNoise(u, v, baseScale * 16.0f, seed + 307) * 0.14f +
               patchNoise(u, v, baseScale * 64.0f, seed + 911) * 0.07f;
    }

    float sampleScalar(const NodeSystem::Image2DData& image, float u, float v,
                       float fallback = 0.0f) {
        if (!image.isValid() || image.width <= 0 || image.height <= 0 ||
            image.channels <= 0 || !image.data || image.data->empty()) {
            return fallback;
        }
        const float px = clamp01(u) * static_cast<float>((std::max)(image.width - 1, 0));
        const float py = clamp01(v) * static_cast<float>((std::max)(image.height - 1, 0));
        const int x0 = static_cast<int>(std::floor(px));
        const int y0 = static_cast<int>(std::floor(py));
        const int x1 = (std::min)(x0 + 1, image.width - 1);
        const int y1 = (std::min)(y0 + 1, image.height - 1);
        const float tx = px - static_cast<float>(x0);
        const float ty = py - static_cast<float>(y0);
        const auto at = [&](int x, int y) {
            const size_t index = (static_cast<size_t>(y) * image.width + x) * image.channels;
            return index < image.data->size() ? (*image.data)[index] : fallback;
        };
        const float top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
        const float bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
        return top + (bottom - top) * ty;
    }

    /**
     * @brief Gradient (rise/run, meters) of the height field under the domain.
     *
     * Every angle in this file derives from this one number through
     * TerrainFieldMath, so degrees and 0-1 mask space can never drift apart
     * the way they did when each consumer carried its own conversion.
     */
    float heightGradient(const NodeSystem::Image2DData& height, int x, int y,
                         const PaintDomain& domain) {
        const int w = (std::max)(domain.width, 2);
        const int h = (std::max)(domain.height, 2);
        const auto metric = TerrainFieldMath::makeFieldMetric(
            domain.worldScale, domain.heightScale, w);
        const float u = static_cast<float>(x) / (w - 1);
        const float v = static_cast<float>(y) / (h - 1);
        return TerrainFieldMath::gradientAtUV(
            [&](float su, float sv) { return sampleScalar(height, su, sv); },
            u, v, w, h, metric);
    }

    /**
     * @brief Slope in 0-1, preferring the shared field over a local derivation.
     *
     * A wired Slope pin and an unwired one must classify the same pixel the
     * same way, otherwise connecting Terrain Analysis silently restyles the
     * terrain. Both paths land on TerrainFieldMath::slope01.
     */
    float resolveSlope01(const NodeSystem::Image2DData& slopeField,
                         const NodeSystem::Image2DData& height,
                         int x, int y, float u, float v,
                         const PaintDomain& domain) {
        if (slopeField.isValid()) return clamp01(sampleScalar(slopeField, u, v));
        return TerrainFieldMath::slope01(heightGradient(height, x, y, domain));
    }

    NodeSystem::Image2DData createOutput(int width, int height, int channels) {
        NodeSystem::Image2DData result;
        result.width = width;
        result.height = height;
        result.channels = channels;
        result.semantic = channels == 4
            ? NodeSystem::ImageSemantic::PackedData
            : NodeSystem::ImageSemantic::Mask;
        result.data = std::make_shared<std::vector<float>>(
            static_cast<size_t>(width) * height * channels, 0.0f);
        return result;
    }

} // namespace

NodeSystem::Image2DData evaluateAutoSplat(
    const NodeSystem::Image2DData& height,
    const NodeSystem::Image2DData& slopeField,
    const PaintDomain& requestedDomain,
    const AutoSplatSettings& settings) {
    PaintDomain domain = requestedDomain;
    domain.width = (std::max)(domain.width, 2);
    domain.height = (std::max)(domain.height, 2);
    domain.worldScale = (std::max)(domain.worldScale, 1e-3f);
    domain.heightScale = (std::max)(std::abs(domain.heightScale), 1e-3f);
    auto result = createOutput(domain.width, domain.height, 4);

#pragma omp parallel for
    for (int y = 0; y < domain.height; ++y) {
        const float v = static_cast<float>(y) / (domain.height - 1);
        for (int x = 0; x < domain.width; ++x) {
            const float u = static_cast<float>(x) / (domain.width - 1);
            const size_t index = static_cast<size_t>(y) * domain.width + x;
            const float worldHeight = sampleScalar(height, u, v) * domain.heightScale;
            // Rules are authored in degrees, so the shared 0-1 slope is
            // converted back through the canonical curve rather than being
            // compared against a second, privately-scaled angle.
            const float slope = resolveSlope01(slopeField, height, x, y, u, v, domain) * 90.0f;
            float weights[4] = {};

            for (int layer = 0; layer < 4; ++layer) {
                const AutoSplatRule& rule = settings.rules[static_cast<size_t>(layer)];
                if (!rule.enabled) continue;
                float heightWeight = 0.0f;
                if (worldHeight >= rule.heightMin && worldHeight <= rule.heightMax) heightWeight = 1.0f;
                else if (worldHeight < rule.heightMin)
                    heightWeight = smoothstep(rule.heightMin - rule.falloff, rule.heightMin, worldHeight);
                else
                    heightWeight = 1.0f - smoothstep(rule.heightMax, rule.heightMax + rule.falloff, worldHeight);

                float slopeWeight = 0.0f;
                if (slope >= rule.slopeMin && slope <= rule.slopeMax) slopeWeight = 1.0f;
                else if (slope < rule.slopeMin)
                    slopeWeight = smoothstep(rule.slopeMin - rule.falloff, rule.slopeMin, slope);
                else
                    slopeWeight = 1.0f - smoothstep(rule.slopeMax, rule.slopeMax + rule.falloff, slope);

                float weight = heightWeight * rule.heightWeight + slopeWeight * rule.slopeWeight;
                if (rule.noiseAmount > 0.0f) {
                    const float detail = fractalPatchNoise(
                        u, v, 16.0f, settings.noiseSeed + layer * 977);
                    weight += (detail * 2.0f - 1.0f) * rule.noiseAmount;
                }
                weights[layer] = clamp01(weight);
            }

            if (settings.normalizeOutput) {
                float sum = weights[0] + weights[1] + weights[2] + weights[3];
                if (sum > 0.001f) {
                    for (float& weight : weights) weight /= sum;
                } else weights[0] = 1.0f;
            }
            for (int channel = 0; channel < 4; ++channel)
                (*result.data)[index * 4 + channel] = weights[channel];
        }
    }
    return result;
}

NodeSystem::Image2DData evaluateSemanticControls(
    const SemanticInputs& inputs,
    const PaintDomain& requestedDomain) {
    PaintDomain domain = requestedDomain;
    domain.width = (std::max)(domain.width, 2);
    domain.height = (std::max)(domain.height, 2);
    auto result = createOutput(domain.width, domain.height, 4);

    // Only R used to be written; G/B/A were hard-zeroed, so a material bound
    // to Wetness, Ice or Hardness behind this node could never appear. All
    // four are packed now, and an unconnected one stays 0 rather than being
    // filled with a guess: these channels reach erosion resistance and soil
    // capacity, and an invented value there is a guess steering a solver.
    // terrain.list_layers reports the empty case as channel_coverage 0.
#pragma omp parallel for
    for (int y = 0; y < domain.height; ++y) {
        const float v = static_cast<float>(y) / (domain.height - 1);
        for (int x = 0; x < domain.width; ++x) {
            const float u = static_cast<float>(x) / (domain.width - 1);
            const size_t index = static_cast<size_t>(y) * domain.width + x;
            (*result.data)[index * 4 + 0] = clamp01(sampleScalar(inputs.flow, u, v));
            (*result.data)[index * 4 + 1] = clamp01(sampleScalar(inputs.wetness, u, v));
            (*result.data)[index * 4 + 2] = clamp01(sampleScalar(inputs.ice, u, v));
            (*result.data)[index * 4 + 3] = clamp01(sampleScalar(inputs.hardness, u, v));
        }
    }
    return result;
}

NodeSystem::Image2DData evaluateSurfaceComposer(
    const SurfaceComposerInputs& inputs,
    const PaintDomain& requestedDomain,
    const SurfaceComposerSettings& settings,
    int outputKind) {
    PaintDomain domain = requestedDomain;
    domain.width = (std::max)(domain.width, 2);
    domain.height = (std::max)(domain.height, 2);
    domain.worldScale = (std::max)(domain.worldScale, 1e-3f);
    domain.heightScale = (std::max)(std::abs(domain.heightScale), 1e-3f);
    const bool packedOutput = outputKind != 0;
    const bool semanticOutput = outputKind == 2;
    auto result = createOutput(domain.width, domain.height, packedOutput ? 4 : 1);
    const float influenceTotal = (std::max)(settings.patchiness + settings.slopeInfluence +
        settings.soilInfluence + settings.flowInfluence + settings.wetnessInfluence +
        settings.hardnessInfluence + (inputs.grass.isValid() ? settings.grassInfluence : 0.0f) +
        (inputs.rock.isValid() ? settings.rockInfluence : 0.0f) +
        (inputs.snow.isValid() ? settings.snowInfluence : 0.0f) +
        (inputs.ice.isValid() ? settings.iceInfluence : 0.0f), 1e-6f);

#pragma omp parallel for
    for (int y = 0; y < domain.height; ++y) {
        const float v = static_cast<float>(y) / (domain.height - 1);
        for (int x = 0; x < domain.width; ++x) {
            const float u = static_cast<float>(x) / (domain.width - 1);
            const size_t index = static_cast<size_t>(y) * domain.width + x;
            const float patch = fractalPatchNoise(
                u, v, settings.textureScale, settings.seed);
            const float slope = resolveSlope01(inputs.slope, inputs.height, x, y, u, v, domain);
            const float hard = inputs.hardness.isValid()
                ? clamp01(sampleScalar(inputs.hardness, u, v)) : 0.45f;
            const float soil = inputs.soil.isValid()
                ? clamp01(sampleScalar(inputs.soil, u, v))
                : clamp01((1.0f - slope) * (1.0f - hard));
            const bool authoredGrass = inputs.grass.isValid();
            const float authoredGrassValue = authoredGrass
                ? clamp01(sampleScalar(inputs.grass, u, v)) : 0.0f;
            // Loose debris at a cliff base is rock too, so it raises the rock
            // claim wherever it lands - including on ground flat enough that
            // slope alone would never have called it rocky, which is exactly
            // where a talus cone sits.
            const float talus = inputs.talus.isValid()
                ? clamp01(sampleScalar(inputs.talus, u, v)) : 0.0f;
            const float rockBase = inputs.rock.isValid()
                ? clamp01(sampleScalar(inputs.rock, u, v))
                : clamp01(slope * 0.75f + hard * settings.hardnessInfluence * 0.55f);
            const float rock = (std::max)(rockBase, talus);
            const float erosionFlow = inputs.flow.isValid()
                ? clamp01(sampleScalar(inputs.flow, u, v)) : 0.0f;
            const float climateFlow = inputs.meltwater.isValid()
                ? clamp01(sampleScalar(inputs.meltwater, u, v)) : 0.0f;
            const float flow = 1.0f - (1.0f - erosionFlow) * (1.0f - climateFlow);
            const float wet = inputs.wetness.isValid()
                ? clamp01(sampleScalar(inputs.wetness, u, v))
                : flow * (1.0f - slope * 0.6f);
            const float snow = inputs.snow.isValid()
                ? clamp01(sampleScalar(inputs.snow, u, v)) : 0.0f;
            const float ice = inputs.ice.isValid()
                ? clamp01(sampleScalar(inputs.ice, u, v)) : 0.0f;
            const float weightedSnow = clamp01(snow * settings.snowInfluence);
            const float weightedIce = clamp01(ice * settings.iceInfluence);
            const float frozenCover = 1.0f - (1.0f - weightedSnow) * (1.0f - weightedIce);
            const float flatness = 1.0f - slope;
            // Grass is a biome-suitability decision, not a copy of Soil Depth.
            // Soil contributes capacity, while slope, water channels, rock,
            // moisture and coherent patchiness independently gate coverage.
            const float generatedGrass = clamp01(
                (0.22f + soil * 0.78f) * std::pow(flatness, 1.55f) *
                (0.48f + wet * 0.52f) * (1.0f - flow * 0.88f) *
                (1.0f - hard * 0.62f) * (0.68f + patch * 0.32f) *
                // Active scree is bare by definition: the cone keeps shedding,
                // so nothing gets long enough to root.
                (1.0f - talus * 0.92f));
            const float grass = authoredGrass ? authoredGrassValue : generatedGrass;

            float surface = (patch * settings.patchiness + slope * settings.slopeInfluence +
                soil * settings.soilInfluence + flow * settings.flowInfluence +
                wet * settings.wetnessInfluence + hard * settings.hardnessInfluence +
                grass * settings.grassInfluence + rock * settings.rockInfluence +
                snow * settings.snowInfluence + ice * settings.iceInfluence) / influenceTotal;
            surface = clamp01((surface - 0.5f) * settings.contrast + 0.5f);
            if (!packedOutput) {
                (*result.data)[index] = surface;
                continue;
            }
            if (semanticOutput) {
                // Non-normalized control texture:
                // R=Flow, G=Wetness, B=Ice, A=Hardness.
                (*result.data)[index * 4 + 0] = flow;
                (*result.data)[index * 4 + 1] = wet;
                (*result.data)[index * 4 + 2] = ice;
                (*result.data)[index * 4 + 3] = hard;
                continue;
            }

            // The influence dials used to shape only the single-channel
            // surface mask; in Splat mode the weighted sum above was computed
            // and then discarded, so five sliders moved nothing. They drive
            // the channel they name here too.
            const float patchBreak = clamp01(
                1.0f + (patch * 2.0f - 1.0f) * settings.patchiness);
            // Deep, wet soil buries rock rather than competing with it.
            const float soilCapacity = clamp01(soil * settings.soilInfluence +
                wet * settings.wetnessInfluence * 0.35f);

            const bool authoredRock = inputs.rock.isValid();
            const float requestedRock = authoredRock
                ? rock
                : clamp01((slope * settings.slopeInfluence +
                           hard * settings.hardnessInfluence +
                           flow * settings.flowInfluence * 0.5f) *
                          settings.rockInfluence * patchBreak) *
                  (1.0f - soilCapacity * 0.65f);
            const float requestedGrass = authoredGrass
                ? grass
                : clamp01(grass * settings.grassInfluence * patchBreak);

            // Snow and ice are cover laid over the substrate, so they claim
            // budget first regardless of what lies beneath.
            float wSnow = clamp01(frozenCover);
            float remaining = 1.0f - wSnow;
            float wGrass = 0.0f;
            float wRock = 0.0f;

            // An authored mask is a statement of intent; a synthesized one is
            // a guess. The old fixed snow->rock->grass order let generated
            // rock consume the budget an explicitly wired grass mask needed,
            // which is why authored masks looked overwritten.
            const auto grantClaim = [&remaining](float request, float& target) {
                target = (std::min)(clamp01(request), (std::max)(remaining, 0.0f));
                remaining -= target;
            };
            if (authoredRock && !authoredGrass) {
                grantClaim(requestedRock, wRock);
                grantClaim(requestedGrass, wGrass);
            } else if (authoredGrass && !authoredRock) {
                grantClaim(requestedGrass, wGrass);
                grantClaim(requestedRock, wRock);
            } else {
                // Equal standing: scale both against the contested budget
                // instead of letting a fixed order starve the second layer.
                const float total = requestedRock + requestedGrass;
                if (total > 1e-6f) {
                    const float scale = (std::min)(1.0f, (std::max)(remaining, 0.0f) / total);
                    wRock = requestedRock * scale;
                    wGrass = requestedGrass * scale;
                    remaining -= wRock + wGrass;
                }
            }

            // Soil is the substrate: whatever no other layer claimed is soil.
            // Turning this off publishes the raw claims instead, which Splat
            // Output now preserves rather than silently renormalizing.
            float wSoil = settings.soilFillsRemainder ? (std::max)(remaining, 0.0f) : 0.0f;

            (*result.data)[index * 4 + 0] = wGrass;
            (*result.data)[index * 4 + 1] = wRock;
            (*result.data)[index * 4 + 2] = wSnow;
            (*result.data)[index * 4 + 3] = wSoil;
        }
    }
    return result;
}

} // namespace TerrainPaintEvaluation
