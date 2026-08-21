#include "TerrainSatMapNodes.h"
#include <imgui.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <NodeSystem/NodeRegistry.h>

namespace TerrainNodesV2 {

    namespace {
        using SatStop = TerrainSatMapColorRampNode::Stop;

        SatStop sampleSatRamp(const std::vector<SatStop>& ramp, float value) {
            value = std::clamp(value, 0.0f, 1.0f);
            if (ramp.empty()) return {value, value, value, value, 1.0f};
            if (value <= ramp.front().pos) return ramp.front();
            if (value >= ramp.back().pos) return ramp.back();
            for (size_t i = 0; i + 1 < ramp.size(); ++i) {
                if (value < ramp[i].pos || value > ramp[i + 1].pos) continue;
                const float span = ramp[i + 1].pos - ramp[i].pos;
                const float t = span > 1e-6f ? (value - ramp[i].pos) / span : 0.0f;
                return {value,
                    ramp[i].r + (ramp[i + 1].r - ramp[i].r) * t,
                    ramp[i].g + (ramp[i + 1].g - ramp[i].g) * t,
                    ramp[i].b + (ramp[i + 1].b - ramp[i].b) * t,
                    ramp[i].a + (ramp[i + 1].a - ramp[i].a) * t};
            }
            return ramp.back();
        }

        std::vector<SatStop> makeMaskRamp(float r, float g, float b, bool coolHigh) {
            const auto c = [](float v) { return std::clamp(v, 0.0f, 1.0f); };
            return {
                {0.0f, c(r * 0.55f + 0.08f), c(g * 0.55f + 0.06f), c(b * 0.55f + 0.04f), 1.0f},
                {0.38f, c(r * 0.80f + 0.05f), c(g * 0.82f + 0.04f), c(b * 0.78f + 0.03f), 1.0f},
                {0.72f, c(r), c(g), c(b), 1.0f},
                {1.0f, c(r * (coolHigh ? 0.78f : 1.22f) + (coolHigh ? 0.02f : 0.06f)),
                       c(g * (coolHigh ? 0.92f : 1.18f) + (coolHigh ? 0.08f : 0.05f)),
                       c(b * (coolHigh ? 1.18f : 1.12f) + (coolHigh ? 0.10f : 0.04f)), 1.0f}
            };
        }

        void enrichPresetRamp(std::vector<SatStop>& ramp, float strength, int phase) {
            if (ramp.size() < 2 || ramp.size() >= 9) return;
            std::vector<SatStop> enriched;
            enriched.reserve(ramp.size() * 2 - 1);
            const auto clampColor = [](float value) { return std::clamp(value, 0.0f, 1.0f); };
            for (size_t i = 0; i + 1 < ramp.size(); ++i) {
                enriched.push_back(ramp[i]);
                SatStop middle;
                middle.pos = (ramp[i].pos + ramp[i + 1].pos) * 0.5f;
                middle.r = (ramp[i].r + ramp[i + 1].r) * 0.5f;
                middle.g = (ramp[i].g + ramp[i + 1].g) * 0.5f;
                middle.b = (ramp[i].b + ramp[i + 1].b) * 0.5f;
                middle.a = (ramp[i].a + ramp[i + 1].a) * 0.5f;
                const float sign = ((static_cast<int>(i) + phase) & 1) ? -1.0f : 1.0f;
                // Subtle alternating warm/cool and luminance movement prevents
                // broad terrain intervals from collapsing into one flat color.
                middle.r = clampColor(middle.r + sign * strength);
                middle.g = clampColor(middle.g + sign * strength * 0.22f);
                middle.b = clampColor(middle.b - sign * strength * 0.55f);
                enriched.push_back(middle);
            }
            enriched.push_back(ramp.back());
            ramp = std::move(enriched);
        }

        void reserveSnowWhite(float& r, float& g, float& b) {
            const float maximum = (std::max)(r, (std::max)(g, b));
            const float minimum = (std::min)(r, (std::min)(g, b));
            const float luminance = r * 0.2126f + g * 0.7152f + b * 0.0722f;
            // A bright low-chroma result reads as snow regardless of which
            // height/slope/soil ramps produced it. Named SatMap presets reserve
            // that visual range for the protected Snow/Ice composition.
            constexpr float substrateLuminanceCeiling = 0.52f;
            if (luminance > substrateLuminanceCeiling && maximum - minimum < 0.20f) {
                const float scale = substrateLuminanceCeiling /
                    (std::max)(luminance, 1e-6f);
                r *= scale;
                g *= scale;
                b *= scale;
            }
        }

        void suppressSnowLikeRockColors(std::vector<SatStop>& ramp) {
            for (SatStop& stop : ramp) reserveSnowWhite(stop.r, stop.g, stop.b);
        }

        void configurePresetMaskRamps(const std::string& name,
                                      std::vector<SatStop>& slope,
                                      std::vector<SatStop>& flow,
                                      std::vector<SatStop>& soil,
                                      std::vector<SatStop>& grass) {
            if (name == "Alpine") {
                slope = {{0.00f,.18f,.27f,.13f,1}, {.16f,.25f,.29f,.17f,1},
                         {.31f,.36f,.29f,.20f,1}, {.46f,.46f,.36f,.27f,1},
                         {.60f,.29f,.30f,.31f,1}, {.73f,.47f,.48f,.48f,1},
                         {.87f,.34f,.37f,.41f,1}, {1.00f,.22f,.25f,.29f,1}};
                flow  = {{0.0f, .30f,.25f,.14f,1}, {.32f,.20f,.31f,.17f,1}, {.62f,.10f,.27f,.25f,1}, {.82f,.08f,.22f,.34f,1}, {1.0f,.12f,.36f,.48f,1}};
                soil  = {{0.0f, .18f,.13f,.08f,1}, {.35f,.30f,.20f,.11f,1}, {.65f,.42f,.29f,.17f,1}, {1.0f,.54f,.45f,.31f,1}};
            } else if (name == "Desert") {
                slope = {{0.0f, .62f,.43f,.20f,1}, {.32f,.70f,.48f,.22f,1}, {.58f,.52f,.31f,.16f,1}, {.80f,.34f,.23f,.19f,1}, {1.0f,.22f,.19f,.18f,1}};
                flow  = {{0.0f, .70f,.48f,.23f,1}, {.30f,.48f,.30f,.15f,1}, {.58f,.30f,.27f,.15f,1}, {.78f,.13f,.34f,.21f,1}, {1.0f,.08f,.31f,.34f,1}};
                soil  = {{0.0f, .37f,.20f,.08f,1}, {.30f,.55f,.31f,.11f,1}, {.62f,.72f,.47f,.19f,1}, {.84f,.82f,.61f,.32f,1}, {1.0f,.91f,.76f,.48f,1}};
            } else if (name == "Tropical") {
                slope = {{0.0f, .06f,.28f,.10f,1}, {.30f,.13f,.36f,.13f,1}, {.55f,.25f,.29f,.17f,1}, {.78f,.25f,.27f,.23f,1}, {1.0f,.38f,.40f,.34f,1}};
                flow  = {{0.0f, .25f,.20f,.09f,1}, {.30f,.10f,.33f,.14f,1}, {.58f,.04f,.31f,.20f,1}, {.80f,.03f,.27f,.28f,1}, {1.0f,.04f,.20f,.36f,1}};
                soil  = {{0.0f, .18f,.07f,.03f,1}, {.30f,.32f,.12f,.04f,1}, {.58f,.45f,.19f,.06f,1}, {.80f,.36f,.28f,.10f,1}, {1.0f,.25f,.38f,.13f,1}};
            } else if (name == "Boreal") {
                slope = {{0.00f,.08f,.23f,.14f,1}, {.18f,.13f,.28f,.17f,1},
                         {.36f,.24f,.31f,.22f,1}, {.54f,.35f,.34f,.29f,1},
                         {.70f,.28f,.31f,.33f,1}, {.86f,.42f,.45f,.47f,1},
                         {1.00f,.25f,.29f,.33f,1}};
                flow  = {{0.0f, .22f,.18f,.10f,1}, {.30f,.12f,.25f,.15f,1}, {.58f,.06f,.23f,.21f,1}, {.80f,.05f,.20f,.30f,1}, {1.0f,.08f,.28f,.42f,1}};
                soil  = {{0.0f, .12f,.08f,.05f,1}, {.32f,.22f,.13f,.07f,1}, {.62f,.31f,.21f,.12f,1}, {.82f,.25f,.31f,.18f,1}, {1.0f,.34f,.40f,.26f,1}};
            } else if (name == "Volcanic") {
                slope = {{0.0f, .11f,.10f,.09f,1}, {.28f,.18f,.15f,.13f,1}, {.55f,.27f,.20f,.16f,1}, {.78f,.18f,.18f,.18f,1}, {1.0f,.42f,.39f,.35f,1}};
                flow  = {{0.0f, .20f,.12f,.07f,1}, {.30f,.17f,.10f,.08f,1}, {.58f,.10f,.15f,.14f,1}, {.80f,.06f,.19f,.22f,1}, {1.0f,.06f,.28f,.34f,1}};
                soil  = {{0.0f, .09f,.07f,.06f,1}, {.30f,.18f,.09f,.05f,1}, {.58f,.34f,.14f,.05f,1}, {.80f,.49f,.23f,.07f,1}, {1.0f,.63f,.37f,.12f,1}};
            } else if (name == "Mediterranean") {
                slope = {{0.0f, .29f,.39f,.16f,1}, {.30f,.43f,.45f,.24f,1}, {.58f,.49f,.43f,.31f,1}, {.80f,.61f,.56f,.46f,1}, {1.0f,.78f,.72f,.59f,1}};
                flow  = {{0.0f, .47f,.31f,.13f,1}, {.30f,.29f,.35f,.15f,1}, {.58f,.12f,.31f,.18f,1}, {.80f,.08f,.25f,.25f,1}, {1.0f,.08f,.30f,.39f,1}};
                soil  = {{0.0f, .32f,.13f,.05f,1}, {.30f,.49f,.20f,.06f,1}, {.58f,.62f,.31f,.10f,1}, {.80f,.71f,.48f,.20f,1}, {1.0f,.82f,.65f,.37f,1}};
            } else if (name == "Autumn") {
                slope = {{0.0f, .29f,.19f,.08f,1}, {.28f,.48f,.25f,.06f,1}, {.55f,.59f,.31f,.08f,1}, {.78f,.39f,.28f,.20f,1}, {1.0f,.57f,.51f,.43f,1}};
                flow  = {{0.0f, .45f,.22f,.05f,1}, {.30f,.34f,.29f,.07f,1}, {.58f,.18f,.31f,.10f,1}, {.80f,.09f,.26f,.18f,1}, {1.0f,.08f,.24f,.31f,1}};
                soil  = {{0.0f, .25f,.09f,.03f,1}, {.28f,.43f,.13f,.03f,1}, {.55f,.60f,.22f,.04f,1}, {.78f,.70f,.39f,.08f,1}, {1.0f,.76f,.57f,.23f,1}};
            } else { // Temperate
                slope = {{0.00f,.14f,.29f,.11f,1}, {.16f,.20f,.33f,.13f,1},
                         {.31f,.30f,.35f,.18f,1}, {.46f,.39f,.31f,.22f,1},
                         {.60f,.29f,.28f,.25f,1}, {.73f,.45f,.42f,.36f,1},
                         {.87f,.34f,.36f,.38f,1}, {1.00f,.23f,.26f,.29f,1}};
                flow  = {{0.0f, .35f,.25f,.12f,1}, {.30f,.22f,.32f,.13f,1}, {.58f,.10f,.29f,.15f,1}, {.80f,.07f,.24f,.24f,1}, {1.0f,.08f,.25f,.38f,1}};
                soil  = {{0.0f, .18f,.10f,.05f,1}, {.30f,.31f,.17f,.07f,1}, {.58f,.44f,.26f,.11f,1}, {.80f,.52f,.36f,.18f,1}, {1.0f,.60f,.49f,.31f,1}};
            }
            if (name == "Alpine") {
                grass = {{0.0f,.10f,.18f,.07f,1}, {.30f,.14f,.28f,.09f,1},
                         {.58f,.24f,.36f,.13f,1}, {.80f,.34f,.39f,.20f,1}, {1.0f,.43f,.45f,.29f,1}};
            } else if (name == "Desert") {
                grass = {{0.0f,.22f,.22f,.08f,1}, {.34f,.31f,.34f,.10f,1},
                         {.65f,.42f,.43f,.14f,1}, {1.0f,.55f,.50f,.22f,1}};
            } else if (name == "Tropical") {
                grass = {{0.0f,.02f,.18f,.06f,1}, {.28f,.03f,.30f,.08f,1},
                         {.55f,.06f,.42f,.10f,1}, {.78f,.12f,.52f,.15f,1}, {1.0f,.24f,.60f,.23f,1}};
            } else if (name == "Boreal") {
                grass = {{0.0f,.04f,.14f,.08f,1}, {.32f,.07f,.23f,.12f,1},
                         {.62f,.13f,.31f,.17f,1}, {.82f,.22f,.36f,.23f,1}, {1.0f,.34f,.42f,.31f,1}};
            } else if (name == "Volcanic") {
                grass = {{0.0f,.06f,.08f,.04f,1}, {.35f,.10f,.14f,.05f,1},
                         {.65f,.18f,.22f,.07f,1}, {1.0f,.30f,.31f,.12f,1}};
            } else if (name == "Mediterranean") {
                grass = {{0.0f,.10f,.21f,.05f,1}, {.30f,.20f,.34f,.07f,1},
                         {.62f,.36f,.44f,.12f,1}, {.82f,.49f,.49f,.18f,1}, {1.0f,.61f,.56f,.28f,1}};
            } else if (name == "Autumn") {
                grass = {{0.0f,.16f,.18f,.04f,1}, {.28f,.35f,.27f,.04f,1},
                         {.55f,.56f,.34f,.05f,1}, {.80f,.69f,.43f,.08f,1}, {1.0f,.77f,.57f,.18f,1}};
            } else {
                grass = {{0.0f,.04f,.18f,.06f,1}, {.28f,.08f,.29f,.08f,1},
                         {.55f,.14f,.39f,.10f,1}, {.80f,.25f,.47f,.16f,1}, {1.0f,.39f,.53f,.25f,1}};
            }
        }

        float presetHeightHighPercentile(const std::string& name) {
            if (name == "Alpine") return 90.0f;
            if (name == "Desert") return 92.0f;
            if (name == "Volcanic") return 92.0f;
            if (name == "Boreal") return 93.0f;
            if (name == "Temperate") return 94.0f;
            if (name == "Mediterranean") return 95.0f;
            if (name == "Autumn") return 95.0f;
            if (name == "Tropical") return 96.0f;
            return 94.0f;
        }

        struct NormalizedScalarField {
            std::vector<float> values;
            int width = 0;
            int height = 0;
            bool valid = false;
        };

        NormalizedScalarField normalizeScalarField(const NodeSystem::Image2DData& image,
                                                   bool autoNormalize,
                                                   float lowPercentile,
                                                   float highPercentile) {
            NormalizedScalarField result;
            if (!image.isValid() || image.channels != 1) return result;
            result.width = image.width;
            result.height = image.height;
            result.values.resize(static_cast<size_t>(result.width) * result.height, 0.0f);
            const bool physical = image.semantic == NodeSystem::ImageSemantic::Height ||
                                  image.semantic == NodeSystem::ImageSemantic::PhysicalScalar;
            float lo = 0.0f, hi = 1.0f;
            float minimum = 0.0f, maximum = 1.0f;
            if (autoNormalize && physical) {
                std::vector<float> finite;
                finite.reserve(image.data->size());
                for (float value : *image.data) if (std::isfinite(value)) finite.push_back(value);
                if (!finite.empty()) {
                    const auto limits = std::minmax_element(finite.begin(), finite.end());
                    minimum = *limits.first;
                    maximum = *limits.second;
                    const auto percentile = [&finite](float pct) {
                        const size_t index = static_cast<size_t>(
                            std::clamp(pct, 0.0f, 100.0f) * 0.01f * static_cast<float>(finite.size() - 1));
                        std::nth_element(finite.begin(), finite.begin() + index, finite.end());
                        return finite[index];
                    };
                    lo = percentile(lowPercentile);
                    hi = percentile(std::max(lowPercentile, highPercentile));
                    if (!(hi > lo + 1e-6f)) {
                        lo = minimum;
                        hi = maximum;
                    }
                }
            }
            const float span = hi - lo;
            const bool collapsed = physical && !(span > 1e-6f);
            for (size_t i = 0; i < result.values.size(); ++i) {
                float value = (*image.data)[i];
                if (!std::isfinite(value)) value = 0.0f;
                else if (autoNormalize && physical && span > 1e-6f) {
                    // Percentiles define soft distribution shoulders, not hard
                    // clipping planes. The old mapping sent every sample above
                    // `hi` to 1.0, turning the top 5-10% of a mountain into one
                    // broad final-stop color. Reserve compact tails so extrema
                    // remain distinct while the main terrain uses most of the ramp.
                    constexpr float lowShoulder = 0.06f;
                    constexpr float highShoulder = 0.92f;
                    if (value < lo && lo > minimum + 1e-6f) {
                        value = lowShoulder * (value - minimum) / (lo - minimum);
                    } else if (value > hi && maximum > hi + 1e-6f) {
                        value = highShoulder + (1.0f - highShoulder) *
                            (value - hi) / (maximum - hi);
                    } else {
                        value = lowShoulder + (highShoulder - lowShoulder) *
                            (value - lo) / span;
                    }
                }
                else if (collapsed) value = 0.5f;
                result.values[i] = std::clamp(value, 0.0f, 1.0f);
            }
            result.valid = true;
            return result;
        }

        float sampleScalarField(const NormalizedScalarField& field,
                                int x, int y, int targetWidth, int targetHeight) {
            if (!field.valid || field.width <= 0 || field.height <= 0) return 0.0f;
            const float fx = std::clamp(
                ((static_cast<float>(x) + 0.5f) * field.width / targetWidth) - 0.5f,
                0.0f, static_cast<float>(field.width - 1));
            const float fy = std::clamp(
                ((static_cast<float>(y) + 0.5f) * field.height / targetHeight) - 0.5f,
                0.0f, static_cast<float>(field.height - 1));
            const int x0 = static_cast<int>(std::floor(fx));
            const int y0 = static_cast<int>(std::floor(fy));
            const int x1 = std::min(x0 + 1, field.width - 1);
            const int y1 = std::min(y0 + 1, field.height - 1);
            const float tx = fx - x0;
            const float ty = fy - y0;
            const float v00 = field.values[static_cast<size_t>(y0) * field.width + x0];
            const float v10 = field.values[static_cast<size_t>(y0) * field.width + x1];
            const float v01 = field.values[static_cast<size_t>(y1) * field.width + x0];
            const float v11 = field.values[static_cast<size_t>(y1) * field.width + x1];
            return (v00 + (v10 - v00) * tx) * (1.0f - ty) +
                   (v01 + (v11 - v01) * tx) * ty;
        }

        float sampleImageChannel(const NodeSystem::Image2DData& image,
                                 float u, float v, int channel = 0) {
            if (!image.isValid() || image.width < 1 || image.height < 1 ||
                channel < 0 || channel >= image.channels) return 0.0f;
            const float px = std::clamp(u, 0.0f, 1.0f) * (image.width - 1);
            const float py = std::clamp(v, 0.0f, 1.0f) * (image.height - 1);
            const int x0 = static_cast<int>(std::floor(px));
            const int y0 = static_cast<int>(std::floor(py));
            const int x1 = std::min(x0 + 1, image.width - 1);
            const int y1 = std::min(y0 + 1, image.height - 1);
            const float tx = px - x0, ty = py - y0;
            const auto at = [&](int sx, int sy) {
                return (*image.data)[(static_cast<size_t>(sy) * image.width + sx) *
                    image.channels + channel];
            };
            const float top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
            const float bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
            return top + (bottom - top) * ty;
        }

        float satHash(int x, int y) {
            uint32_t value = static_cast<uint32_t>(x) * 0x8da6b343u ^
                             static_cast<uint32_t>(y) * 0xd8163841u;
            value ^= value >> 13; value *= 0x85ebca6bu; value ^= value >> 16;
            return static_cast<float>(value & 0x00ffffffu) / 16777215.0f;
        }

        float satValueNoise(float x, float y) {
            const int ix = static_cast<int>(std::floor(x));
            const int iy = static_cast<int>(std::floor(y));
            float fx = x - ix, fy = y - iy;
            fx = fx * fx * (3.0f - 2.0f * fx);
            fy = fy * fy * (3.0f - 2.0f * fy);
            const float a = satHash(ix, iy), b = satHash(ix + 1, iy);
            const float c = satHash(ix, iy + 1), d = satHash(ix + 1, iy + 1);
            return (a + (b - a) * fx) * (1.0f - fy) + (c + (d - c) * fx) * fy;
        }

        float satFbm(float x, float y) {
            float value = 0.0f, amplitude = 0.5714286f;
            for (int octave = 0; octave < 3; ++octave) {
                value += satValueNoise(x, y) * amplitude;
                x = x * 2.07f + 19.1f; y = y * 2.03f - 7.7f;
                amplitude *= 0.5f;
            }
            return value;
        }
    }

    // --- TerrainSatMapColorRampNode ---

    void TerrainSatMapColorRampNode::sortStops() {
        std::sort(stops.begin(), stops.end(), [](const Stop& a, const Stop& b) {
            return a.pos < b.pos;
        });
    }

    void TerrainSatMapColorRampNode::applyPreset(const std::string& name) {
        preset = name;
        // Presets own their distribution window. 75% collapses too much of a
        // mountain into the final color, while a universal 98% pushes broad
        // slopes too far back toward the low/green part of the ramp. Mountain
        // biomes therefore close their upper range earlier than lush biomes.
        autoNormalize = true;
        normalizeLowPercentile = 2.0f;
        normalizeHighPercentile = presetHeightHighPercentile(name);
        slopeBlend = 0.72f;
        flowBlend = 0.55f;
        soilBlend = 0.45f;
        grassBlend = name == "Tropical" ? 0.82f :
                     name == "Desert" ? 0.38f :
                     name == "Volcanic" ? 0.28f :
                     name == "Alpine" ? 0.56f :
                     name == "Boreal" ? 0.72f :
                     name == "Mediterranean" ? 0.58f :
                     name == "Autumn" ? 0.74f : 0.68f;
        if (name == "Alpine") {
            stops = {{0.00f, 0.12f, 0.11f, 0.07f, 1.0f}, {0.18f, 0.18f, 0.16f, 0.10f, 1.0f},
                     {0.36f, 0.27f, 0.23f, 0.16f, 1.0f}, {0.52f, 0.36f, 0.30f, 0.22f, 1.0f},
                     {0.66f, 0.41f, 0.37f, 0.31f, 1.0f}, {0.78f, 0.46f, 0.45f, 0.43f, 1.0f},
                     {0.90f, 0.51f, 0.52f, 0.52f, 1.0f}, {1.00f, 0.59f, 0.61f, 0.61f, 1.0f}};
            slopeR = 0.48f; slopeG = 0.48f; slopeB = 0.46f;
            flowR = 0.08f; flowG = 0.24f; flowB = 0.28f;
            soilR = 0.34f; soilG = 0.22f; soilB = 0.12f;
        } else if (name == "Desert") {
            stops = {{0.0f, 0.28f, 0.18f, 0.08f, 1.0f}, {0.35f, 0.58f, 0.38f, 0.16f, 1.0f},
                     {0.72f, 0.78f, 0.54f, 0.28f, 1.0f}, {1.0f, 0.90f, 0.78f, 0.48f, 1.0f}};
            slopeR = 0.30f; slopeG = 0.25f; slopeB = 0.20f;
            flowR = 0.18f; flowG = 0.30f; flowB = 0.24f;
            soilR = 0.62f; soilG = 0.38f; soilB = 0.16f;
        } else if (name == "Tropical") {
            stops = {{0.0f, 0.10f, 0.07f, 0.035f, 1.0f}, {0.26f, 0.19f, 0.11f, 0.05f, 1.0f},
                     {0.50f, 0.28f, 0.18f, 0.08f, 1.0f}, {0.72f, 0.34f, 0.27f, 0.15f, 1.0f},
                     {0.88f, 0.39f, 0.36f, 0.27f, 1.0f}, {1.0f, 0.47f, 0.46f, 0.39f, 1.0f}};
            slopeR = 0.22f; slopeG = 0.25f; slopeB = 0.19f;
            flowR = 0.03f; flowG = 0.28f; flowB = 0.20f;
            soilR = 0.38f; soilG = 0.16f; soilB = 0.07f;
        } else if (name == "Boreal") {
            stops = {{0.0f, 0.08f, 0.07f, 0.05f, 1.0f}, {0.25f, 0.15f, 0.12f, 0.08f, 1.0f},
                     {0.48f, 0.24f, 0.20f, 0.15f, 1.0f}, {0.68f, 0.32f, 0.29f, 0.24f, 1.0f},
                     {0.84f, 0.39f, 0.40f, 0.38f, 1.0f}, {1.0f, 0.48f, 0.52f, 0.51f, 1.0f}};
            slopeR = 0.28f; slopeG = 0.30f; slopeB = 0.32f;
            flowR = 0.05f; flowG = 0.18f; flowB = 0.26f;
            soilR = 0.28f; soilG = 0.16f; soilB = 0.09f;
        } else if (name == "Volcanic") {
            stops = {{0.0f, 0.06f, 0.05f, 0.04f, 1.0f}, {0.38f, 0.16f, 0.12f, 0.10f, 1.0f},
                     {0.72f, 0.35f, 0.22f, 0.12f, 1.0f}, {1.0f, 0.42f, 0.38f, 0.34f, 1.0f}};
            slopeR = 0.16f; slopeG = 0.15f; slopeB = 0.14f;
            flowR = 0.08f; flowG = 0.16f; flowB = 0.18f;
            soilR = 0.30f; soilG = 0.12f; soilB = 0.06f;
        } else if (name == "Mediterranean") {
            stops = {{0.0f, 0.19f, 0.13f, 0.055f, 1.0f}, {0.25f, 0.31f, 0.22f, 0.09f, 1.0f},
                     {0.48f, 0.45f, 0.33f, 0.15f, 1.0f}, {0.68f, 0.57f, 0.45f, 0.26f, 1.0f},
                     {0.84f, 0.68f, 0.59f, 0.43f, 1.0f}, {1.0f, 0.76f, 0.70f, 0.58f, 1.0f}};
            slopeR = 0.42f; slopeG = 0.38f; slopeB = 0.30f;
            flowR = 0.08f; flowG = 0.22f; flowB = 0.16f;
            soilR = 0.52f; soilG = 0.22f; soilB = 0.08f;
        } else if (name == "Autumn") {
            stops = {{0.0f, 0.14f, 0.075f, 0.03f, 1.0f}, {0.24f, 0.25f, 0.13f, 0.045f, 1.0f},
                     {0.47f, 0.38f, 0.21f, 0.08f, 1.0f}, {0.67f, 0.48f, 0.31f, 0.15f, 1.0f},
                     {0.84f, 0.55f, 0.43f, 0.29f, 1.0f}, {1.0f, 0.63f, 0.56f, 0.46f, 1.0f}};
            slopeR = 0.35f; slopeG = 0.20f; slopeB = 0.12f;
            flowR = 0.14f; flowG = 0.25f; flowB = 0.10f;
            soilR = 0.50f; soilG = 0.18f; soilB = 0.05f;
        } else if (name == "Layer: Soil") {
            stops = {{0.00f, 0.075f, 0.045f, 0.025f, 1.0f}, {0.20f, 0.14f, 0.075f, 0.035f, 1.0f},
                     {0.40f, 0.24f, 0.13f, 0.055f, 1.0f}, {0.60f, 0.34f, 0.20f, 0.085f, 1.0f},
                     {0.80f, 0.43f, 0.29f, 0.14f, 1.0f}, {1.00f, 0.52f, 0.39f, 0.23f, 1.0f}};
            slopeR = soilR = 0.34f; slopeG = soilG = 0.20f; slopeB = soilB = 0.085f;
            flowR = 0.16f; flowG = 0.10f; flowB = 0.05f;
        } else if (name == "Layer: Flow") {
            stops = {{0.00f, 0.055f, 0.065f, 0.050f, 1.0f}, {0.18f, 0.075f, 0.11f, 0.075f, 1.0f},
                     {0.38f, 0.075f, 0.17f, 0.13f, 1.0f}, {0.58f, 0.065f, 0.23f, 0.20f, 1.0f},
                     {0.78f, 0.075f, 0.29f, 0.31f, 1.0f}, {1.00f, 0.12f, 0.34f, 0.40f, 1.0f}};
            slopeR = 0.10f; slopeG = 0.13f; slopeB = 0.12f;
            flowR = 0.07f; flowG = 0.24f; flowB = 0.24f;
            soilR = 0.13f; soilG = 0.12f; soilB = 0.075f;
        } else if (name == "Layer: Grass") {
            stops = {{0.00f, 0.11f, 0.10f, 0.035f, 1.0f}, {0.20f, 0.16f, 0.18f, 0.045f, 1.0f},
                     {0.40f, 0.15f, 0.25f, 0.055f, 1.0f}, {0.60f, 0.12f, 0.32f, 0.075f, 1.0f},
                     {0.80f, 0.10f, 0.39f, 0.10f, 1.0f}, {1.00f, 0.18f, 0.46f, 0.16f, 1.0f}};
            slopeR = 0.13f; slopeG = 0.18f; slopeB = 0.06f;
            flowR = 0.08f; flowG = 0.26f; flowB = 0.12f;
            soilR = 0.20f; soilG = 0.18f; soilB = 0.07f;
        } else if (name == "Layer: Rock") {
            stops = {{0.00f, 0.10f, 0.085f, 0.070f, 1.0f}, {0.18f, 0.17f, 0.15f, 0.13f, 1.0f},
                     {0.38f, 0.25f, 0.23f, 0.20f, 1.0f}, {0.58f, 0.32f, 0.31f, 0.29f, 1.0f},
                     {0.78f, 0.39f, 0.40f, 0.39f, 1.0f}, {1.00f, 0.47f, 0.49f, 0.48f, 1.0f}};
            slopeR = 0.33f; slopeG = 0.34f; slopeB = 0.33f;
            flowR = 0.17f; flowG = 0.20f; flowB = 0.20f;
            soilR = 0.27f; soilG = 0.19f; soilB = 0.12f;
        } else if (name == "Layer: Mud") {
            stops = {{0.00f, .055f, .040f, .025f, 1}, {.22f, .085f, .060f, .035f, 1},
                     {.44f, .12f, .085f, .050f, 1}, {.66f, .15f, .115f, .075f, 1},
                     {.84f, .13f, .125f, .095f, 1}, {1.00f, .10f, .13f, .12f, 1}};
            slopeR = .12f; slopeG = .10f; slopeB = .08f;
            flowR = .08f; flowG = .13f; flowB = .12f;
            soilR = .16f; soilG = .10f; soilB = .055f;
        } else if (name == "Layer: Moss") {
            stops = {{0.00f, .045f, .075f, .025f, 1}, {.20f, .065f, .12f, .035f, 1},
                     {.40f, .085f, .18f, .045f, 1}, {.60f, .11f, .24f, .065f, 1},
                     {.80f, .16f, .29f, .09f, 1}, {1.00f, .22f, .34f, .13f, 1}};
            slopeR = .10f; slopeG = .17f; slopeB = .06f;
            flowR = .07f; flowG = .20f; flowB = .11f;
            soilR = .15f; soilG = .16f; soilB = .06f;
        } else if (name == "Layer: Cavity") {
            stops = {{0.00f, .15f, .14f, .125f, 1}, {.20f, .13f, .125f, .115f, 1},
                     {.40f, .105f, .105f, .10f, 1}, {.60f, .085f, .09f, .09f, 1},
                     {.80f, .065f, .075f, .08f, 1}, {1.00f, .045f, .060f, .07f, 1}};
            slopeR = .10f; slopeG = .10f; slopeB = .10f;
            flowR = .06f; flowG = .08f; flowB = .09f;
            soilR = .12f; soilG = .09f; soilB = .06f;
        } else { // Temperate
            preset = "Temperate";
            stops = {{0.00f, 0.10f, 0.075f, 0.04f, 1.0f}, {0.18f, 0.17f, 0.12f, 0.06f, 1.0f},
                     {0.36f, 0.25f, 0.18f, 0.09f, 1.0f}, {0.54f, 0.33f, 0.25f, 0.15f, 1.0f},
                     {0.70f, 0.40f, 0.34f, 0.26f, 1.0f}, {0.84f, 0.46f, 0.43f, 0.38f, 1.0f},
                     {1.00f, 0.56f, 0.57f, 0.54f, 1.0f}};
            slopeR = 0.34f; slopeG = 0.31f; slopeB = 0.27f;
            flowR = 0.10f; flowG = 0.24f; flowB = 0.12f;
            soilR = 0.42f; soilG = 0.25f; soilB = 0.12f;
        }
        const bool layerPreset = name.rfind("Layer: ", 0) == 0;
        if (layerPreset) {
            // The primary scalar drives the layer ramp; the external SatMap
            // Blend mask owns coverage, so implicit terrain overlays must not
            // color the layer a second time.
            autoNormalize = false;
            autoDeriveMasks = false;
            slopeBlend = flowBlend = soilBlend = grassBlend = 0.0f;
            detailStrength = 0.075f;
        }
        configurePresetMaskRamps(preset, slopeStops, flowStops, soilStops, grassStops);
        const SatStop representativeGrass = sampleSatRamp(grassStops, 0.72f);
        grassR = representativeGrass.r;
        grassG = representativeGrass.g;
        grassB = representativeGrass.b;
        enrichPresetRamp(stops, 0.026f, 0);
        enrichPresetRamp(slopeStops, 0.020f, 1);
        enrichPresetRamp(flowStops, 0.016f, 0);
        enrichPresetRamp(soilStops, 0.020f, 1);
        enrichPresetRamp(grassStops, 0.018f, 0);
        suppressSnowLikeRockColors(stops);
        suppressSnowLikeRockColors(slopeStops);
        sortStops();
    }

    NodeSystem::PinValue TerrainSatMapColorRampNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto heightImage = getImageInput(0, ctx);
        if (!heightImage.isValid() || heightImage.channels != 1) return NodeSystem::PinValue{};
        TerrainContext* terrainContext = getTerrainContext(ctx);
        TerrainObject* terrain = terrainContext ? terrainContext->terrain : nullptr;
        const int w = terrain ? terrain->paintGridWidth() : heightImage.width;
        const int h = terrain ? terrain->paintGridHeight() : heightImage.height;
        if (w < 2 || h < 2) return NodeSystem::PinValue{};

        sortStops();
        auto slopeImage = getImageInput(1, ctx);
        auto flowImage = getImageInput(2, ctx);
        auto soilImage = getImageInput(3, ctx);
        const auto snowImage = getImageInput(4, ctx);
        const auto iceImage = getImageInput(5, ctx);
        const auto meltImage = getImageInput(6, ctx);
        const auto avalancheImage = getImageInput(7, ctx);
        const auto grassImage = getImageInput(8, ctx);

        // Semantic fallback: use the already-published erosion fields when a
        // pin is unconnected, and derive slope directly from Height. Explicit
        // graph links always win.
        NodeSystem::Image2DData derivedSlope;
        NodeSystem::Image2DData terrainFlow;
        NodeSystem::Image2DData terrainSoil;
        if (autoDeriveMasks && (!slopeImage.isValid() || slopeImage.channels != 1)) {
            derivedSlope.width = heightImage.width;
            derivedSlope.height = heightImage.height;
            derivedSlope.channels = 1;
            derivedSlope.semantic = NodeSystem::ImageSemantic::Mask;
            derivedSlope.data = std::make_shared<std::vector<float>>(
                static_cast<size_t>(derivedSlope.width) * derivedSlope.height, 0.0f);
            const float cellSize = terrainContext
                ? terrainContext->scale_xz / static_cast<float>(std::max(derivedSlope.width, derivedSlope.height))
                : 1.0f;
            const float heightScale = terrainContext ? terrainContext->scale_y : 1.0f;
            for (int y = 0; y < derivedSlope.height; ++y) {
                const int y0 = std::max(0, y - 1), y1 = std::min(derivedSlope.height - 1, y + 1);
                for (int x = 0; x < derivedSlope.width; ++x) {
                    const int x0 = std::max(0, x - 1), x1 = std::min(derivedSlope.width - 1, x + 1);
                    const float dx = ((*heightImage.data)[static_cast<size_t>(y) * derivedSlope.width + x1] -
                                      (*heightImage.data)[static_cast<size_t>(y) * derivedSlope.width + x0]) *
                                     heightScale / std::max(2.0f * cellSize, 1e-6f);
                    const float dz = ((*heightImage.data)[static_cast<size_t>(y1) * derivedSlope.width + x] -
                                      (*heightImage.data)[static_cast<size_t>(y0) * derivedSlope.width + x]) *
                                     heightScale / std::max(2.0f * cellSize, 1e-6f);
                    (*derivedSlope.data)[static_cast<size_t>(y) * derivedSlope.width + x] =
                        std::atan(std::sqrt(dx * dx + dz * dz)) / 1.57079632679f;
                }
            }
            slopeImage = derivedSlope;
        }
        if (autoDeriveMasks && terrain && (!flowImage.isValid() || flowImage.channels != 1) &&
            terrain->flowMap.size() == static_cast<size_t>(terrain->heightmap.width) * terrain->heightmap.height) {
            terrainFlow.width = terrain->heightmap.width; terrainFlow.height = terrain->heightmap.height;
            terrainFlow.channels = 1; terrainFlow.semantic = NodeSystem::ImageSemantic::PhysicalScalar;
            terrainFlow.data = std::make_shared<std::vector<float>>(terrain->flowMap);
            flowImage = terrainFlow;
        }
        if (autoDeriveMasks && terrain && (!soilImage.isValid() || soilImage.channels != 1) &&
            terrain->hardnessMap.size() == static_cast<size_t>(terrain->heightmap.width) * terrain->heightmap.height) {
            terrainSoil.width = terrain->heightmap.width; terrainSoil.height = terrain->heightmap.height;
            terrainSoil.channels = 1; terrainSoil.semantic = NodeSystem::ImageSemantic::Mask;
            terrainSoil.data = std::make_shared<std::vector<float>>(terrain->hardnessMap.size());
            for (size_t i = 0; i < terrain->hardnessMap.size(); ++i)
                (*terrainSoil.data)[i] = 1.0f - std::clamp(terrain->hardnessMap[i], 0.0f, 1.0f);
            soilImage = terrainSoil;
        }

        const NormalizedScalarField heightField = normalizeScalarField(
            heightImage, autoNormalize, normalizeLowPercentile, normalizeHighPercentile);
        const NormalizedScalarField slopeField = normalizeScalarField(
            slopeImage, autoNormalize, normalizeLowPercentile, normalizeHighPercentile);
        const NormalizedScalarField flowField = normalizeScalarField(
            flowImage, autoNormalize, normalizeLowPercentile, normalizeHighPercentile);
        const NormalizedScalarField soilField = normalizeScalarField(
            soilImage, autoNormalize, normalizeLowPercentile, normalizeHighPercentile);
        const NormalizedScalarField snowField = normalizeScalarField(snowImage, false, 0.0f, 100.0f);
        const NormalizedScalarField iceField = normalizeScalarField(iceImage, false, 0.0f, 100.0f);
        const NormalizedScalarField meltField = normalizeScalarField(meltImage, false, 0.0f, 100.0f);
        const NormalizedScalarField avalancheField = normalizeScalarField(avalancheImage, false, 0.0f, 100.0f);
        const NormalizedScalarField grassField = normalizeScalarField(grassImage, false, 0.0f, 100.0f);

        auto outputData = std::make_shared<std::vector<float>>(static_cast<size_t>(w) * h * 4, 0.0f);
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t pixel = static_cast<size_t>(y) * w + x;
                const float heightValue = sampleScalarField(heightField, x, y, w, h);
                const float slope = sampleScalarField(slopeField, x, y, w, h);
                const float flow = sampleScalarField(flowField, x, y, w, h);
                const float soil = sampleScalarField(soilField, x, y, w, h);
                const float snow = sampleScalarField(snowField, x, y, w, h);
                const float ice = sampleScalarField(iceField, x, y, w, h);
                const float melt = sampleScalarField(meltField, x, y, w, h);
                const float avalanche = sampleScalarField(avalancheField, x, y, w, h);
                const float grass = grassField.valid
                    ? sampleScalarField(grassField, x, y, w, h)
                    : (autoDeriveMasks
                        ? std::clamp((0.25f + soil * 0.75f) *
                            std::pow(1.0f - slope, 1.55f) * (1.0f - flow * 0.88f), 0.0f, 1.0f)
                        : 0.0f);

                float debugValue = heightValue;
                if (debugView == 2) debugValue = slope;
                else if (debugView == 3) debugValue = flow;
                else if (debugView == 4) debugValue = soil;
                else if (debugView == 5) debugValue = snow;
                else if (debugView == 6) debugValue = ice;
                else if (debugView == 7) debugValue = melt;
                else if (debugView == 8) debugValue = avalanche;
                else if (debugView == 9) debugValue = grass;
                if (debugView != 0) {
                    (*outputData)[pixel * 4 + 0] = debugValue;
                    (*outputData)[pixel * 4 + 1] = debugValue;
                    (*outputData)[pixel * 4 + 2] = debugValue;
                    (*outputData)[pixel * 4 + 3] = 1.0f;
                    continue;
                }

                const float nx = (static_cast<float>(x) + 0.5f) / w * detailScale;
                const float ny = (static_cast<float>(y) + 0.5f) / h * detailScale;
                const float detailSignal = (satFbm(nx, ny) - 0.5f) * 2.0f;
                const float detail = detailSignal * detailStrength;
                const SatStop base = sampleSatRamp(
                    stops, std::clamp(heightValue + detail * 0.80f, 0.0f, 1.0f));
                float r = base.r, g = base.g, b = base.b;
                const auto mixChannel = [](float a, float overlay, float amount) {
                    return a + (overlay - a) * std::clamp(amount, 0.0f, 1.0f);
                };
                if (slopeField.valid) {
                    const float slopeGate = std::clamp((slope - 0.20f) / 0.68f, 0.0f, 1.0f);
                    const float smoothGate = slopeGate * slopeGate * (3.0f - 2.0f * slopeGate);
                    // Keep at least 22% of the height/detail color even on a
                    // vertical face. Full replacement was the main source of
                    // broad monochrome slope patches.
                    const float amount = smoothGate * slopeBlend * 0.78f;
                    const float slopeCoordinate = std::clamp(
                        slope + detail * 0.45f + (heightValue - 0.5f) * 0.08f, 0.0f, 1.0f);
                    const SatStop color = sampleSatRamp(slopeStops, slopeCoordinate);
                    r = mixChannel(r, color.r, amount); g = mixChannel(g, color.g, amount); b = mixChannel(b, color.b, amount);
                }
                if (flowField.valid) {
                    const float flowCoordinate = std::clamp(
                        flow + detail * 0.24f + (heightValue - 0.5f) * 0.035f, 0.0f, 1.0f);
                    const SatStop color = sampleSatRamp(flowStops, flowCoordinate);
                    r = mixChannel(r, color.r, flow * flowBlend);
                    g = mixChannel(g, color.g, flow * flowBlend);
                    b = mixChannel(b, color.b, flow * flowBlend);
                }
                if (soilField.valid) {
                    const float soilCoordinate = std::clamp(
                        soil + detail * 0.36f - slope * 0.045f, 0.0f, 1.0f);
                    const SatStop color = sampleSatRamp(soilStops, soilCoordinate);
                    // Soil is substrate capacity, not a full steep-face coat.
                    // Attenuating it with slope prevents Soil + Rock from
                    // converging into one dominant high-altitude color.
                    const float steepness = std::clamp((slope - 0.20f) / 0.68f, 0.0f, 1.0f);
                    const float soilAmount = soil * soilBlend * (1.0f - steepness * 0.48f);
                    r = mixChannel(r, color.r, soilAmount);
                    g = mixChannel(g, color.g, soilAmount);
                    b = mixChannel(b, color.b, soilAmount);
                }
                if (grassField.valid || autoDeriveMasks) {
                    const float grassCoordinate = std::clamp(
                        grass + detail * 0.30f + (0.5f - heightValue) * 0.035f, 0.0f, 1.0f);
                    const SatStop color = sampleSatRamp(grassStops, grassCoordinate);
                    // Vegetation coats suitable substrate but retreats from
                    // exposed cliffs and active drainage. Explicit Biome Grass
                    // remains authoritative; these gates only soften boundaries.
                    const float grassAmount = grass * grassBlend *
                        (1.0f - slope * 0.62f) * (1.0f - flow * 0.58f);
                    r = mixChannel(r, color.r, grassAmount);
                    g = mixChannel(g, color.g, grassAmount);
                    b = mixChannel(b, color.b, grassAmount);
                }
                // A second decorrelated field breaks up broad exposed rock
                // even when no Snow overlay is connected. It is applied to
                // the base material before protected snow/ice composition.
                const float breakupSignal = (satFbm(nx * 0.43f + 31.7f, ny * 0.47f - 18.2f) - 0.5f) * 2.0f;
                const float rockPresence = std::clamp((slope - 0.18f) / 0.72f, 0.0f, 1.0f);
                const float breakup = breakupSignal * detailStrength * (0.20f + rockPresence * 0.42f);
                r = std::clamp(r + breakup * 0.20f, 0.0f, 1.0f);
                g = std::clamp(g + breakup * 0.055f, 0.0f, 1.0f);
                b = std::clamp(b - breakup * 0.14f, 0.0f, 1.0f);
                // Apply paint-grid contrast to the substrate first, then enforce
                // the named-preset contract. Snow is composed afterwards and is
                // therefore the only path allowed into the near-white range.
                const float microContrast = 1.0f + detail * 0.24f;
                r = std::clamp(r * microContrast, 0.0f, 1.0f);
                g = std::clamp(g * microContrast, 0.0f, 1.0f);
                b = std::clamp(b * microContrast, 0.0f, 1.0f);
                if (preset != "Custom") reserveSnowWhite(r, g, b);
                if (snowField.valid || iceField.valid) {
                    const float sideWeathering = slopeField.valid ? slope * 0.22f : 0.0f;
                    const float dirty = std::clamp(melt * meltBlend + avalanche * avalancheBlend + sideWeathering, 0.0f, 1.0f);
                    const float wetPhase = std::clamp(melt * 1.25f, 0.0f, 1.0f);
                    float sr = mixChannel(freshSnowR, wetSnowR, wetPhase);
                    float sg = mixChannel(freshSnowG, wetSnowG, wetPhase);
                    float sb = mixChannel(freshSnowB, wetSnowB, wetPhase);
                    sr = mixChannel(sr, dirtySnowR, dirty); sg = mixChannel(sg, dirtySnowG, dirty); sb = mixChannel(sb, dirtySnowB, dirty);
                    sr = mixChannel(sr, iceR, ice); sg = mixChannel(sg, iceG, ice); sb = mixChannel(sb, iceB, ice);
                    const float coverage = std::clamp(
                        (snow + ice * (1.0f - snow)) * snowBlend * (1.0f - melt * 0.45f), 0.0f, 1.0f);
                    r = mixChannel(r, sr, coverage); g = mixChannel(g, sg, coverage); b = mixChannel(b, sb, coverage);
                }
                (*outputData)[pixel * 4 + 0] = std::clamp(r, 0.0f, 1.0f);
                (*outputData)[pixel * 4 + 1] = std::clamp(g, 0.0f, 1.0f);
                (*outputData)[pixel * 4 + 2] = std::clamp(b, 0.0f, 1.0f);
                (*outputData)[pixel * 4 + 3] = 1.0f;
            }
        }

        NodeSystem::Image2DData outImg;
        outImg.data = outputData;
        outImg.width = w;
        outImg.height = h;
        outImg.channels = 4;
        outImg.semantic = NodeSystem::ImageSemantic::Albedo;
        return NodeSystem::PinValue{outImg};
    }

    void TerrainSatMapColorRampNode::serializeToJson(nlohmann::json& j) const {
        TerrainNodeBase::serializeToJson(j);
        j["satMapSchemaVersion"] = 10;
        j["autoNormalize"] = autoNormalize;
        j["autoDeriveMasks"] = autoDeriveMasks;
        j["normalizeLowPercentile"] = normalizeLowPercentile;
        j["normalizeHighPercentile"] = normalizeHighPercentile;
        j["detailStrength"] = detailStrength;
        j["detailScale"] = detailScale;
        j["debugView"] = debugView;
        j["preset"] = preset;
        j["slopeBlend"] = slopeBlend;
        j["flowBlend"] = flowBlend;
        j["soilBlend"] = soilBlend;
        j["grassBlend"] = grassBlend;
        j["slopeColor"] = {slopeR, slopeG, slopeB};
        j["flowColor"] = {flowR, flowG, flowB};
        j["soilColor"] = {soilR, soilG, soilB};
        j["grassColor"] = {grassR, grassG, grassB};
        j["snowBlend"] = snowBlend;
        j["meltBlend"] = meltBlend;
        j["avalancheBlend"] = avalancheBlend;
        j["freshSnowColor"] = {freshSnowR, freshSnowG, freshSnowB};
        j["wetSnowColor"] = {wetSnowR, wetSnowG, wetSnowB};
        j["dirtySnowColor"] = {dirtySnowR, dirtySnowG, dirtySnowB};
        j["iceColor"] = {iceR, iceG, iceB};
        const auto writeRamp = [](const std::vector<Stop>& ramp) {
            nlohmann::json result = nlohmann::json::array();
            for (const auto& s : ramp) result.push_back({
                {"pos", s.pos}, {"r", s.r}, {"g", s.g}, {"b", s.b}, {"a", s.a}});
            return result;
        };
        j["slopeStops"] = writeRamp(slopeStops);
        j["flowStops"] = writeRamp(flowStops);
        j["soilStops"] = writeRamp(soilStops);
        j["grassStops"] = writeRamp(grassStops);
        nlohmann::json jStops = nlohmann::json::array();
        for (const auto& s : stops) {
            jStops.push_back({
                {"pos", s.pos},
                {"r", s.r}, {"g", s.g}, {"b", s.b}, {"a", s.a}
            });
        }
        j["stops"] = jStops;
    }

    void TerrainSatMapColorRampNode::deserializeFromJson(const nlohmann::json& j) {
        TerrainNodeBase::deserializeFromJson(j);
        const std::string requestedPreset = j.value("preset", std::string("Temperate"));
        // Generic scripting/IPC property edits round-trip through this JSON
        // contract. When only `preset` changed, the serialized ramps still
        // describe the node's current preset; detect that case so the same
        // applyPreset path used by the UI runs here as well. A project load
        // with authored/custom ramp values does not match the constructor's
        // current ramps and therefore remains lossless.
        const auto rampMatchesJson = [&j](const char* key, const std::vector<Stop>& ramp) {
            if (!j.contains(key) || !j[key].is_array() || j[key].size() != ramp.size()) return false;
            constexpr float epsilon = 1e-6f;
            for (size_t i = 0; i < ramp.size(); ++i) {
                const auto& item = j[key][i];
                const Stop& stop = ramp[i];
                if (std::abs(item.value("pos", 0.0f) - stop.pos) > epsilon ||
                    std::abs(item.value("r", 1.0f) - stop.r) > epsilon ||
                    std::abs(item.value("g", 1.0f) - stop.g) > epsilon ||
                    std::abs(item.value("b", 1.0f) - stop.b) > epsilon ||
                    std::abs(item.value("a", 1.0f) - stop.a) > epsilon) return false;
            }
            return true;
        };
        const bool presetPropertyEdit = requestedPreset != preset &&
            rampMatchesJson("stops", stops) &&
            rampMatchesJson("slopeStops", slopeStops) &&
            rampMatchesJson("flowStops", flowStops) &&
            rampMatchesJson("soilStops", soilStops) &&
            rampMatchesJson("grassStops", grassStops);
        autoNormalize = j.value("autoNormalize", true);
        autoDeriveMasks = j.value("autoDeriveMasks", true);
        normalizeLowPercentile = std::clamp(j.value("normalizeLowPercentile", 2.0f), 0.0f, 25.0f);
        normalizeHighPercentile = std::clamp(j.value("normalizeHighPercentile", 94.0f), 75.0f, 100.0f);
        if (normalizeHighPercentile < normalizeLowPercentile) normalizeHighPercentile = normalizeLowPercentile;
        detailStrength = std::clamp(j.value("detailStrength", 0.11f), 0.0f, 0.35f);
        detailScale = std::clamp(j.value("detailScale", 180.0f), 4.0f, 2048.0f);
        debugView = std::clamp(j.value("debugView", 0), 0, 9);
        preset = requestedPreset;
        slopeBlend = std::clamp(j.value("slopeBlend", 0.72f), 0.0f, 1.0f);
        flowBlend = std::clamp(j.value("flowBlend", 0.55f), 0.0f, 1.0f);
        soilBlend = std::clamp(j.value("soilBlend", 0.45f), 0.0f, 1.0f);
        grassBlend = std::clamp(j.value("grassBlend", 0.68f), 0.0f, 1.0f);
        snowBlend = std::clamp(j.value("snowBlend", 1.0f), 0.0f, 1.0f);
        meltBlend = std::clamp(j.value("meltBlend", 0.75f), 0.0f, 1.0f);
        avalancheBlend = std::clamp(j.value("avalancheBlend", 0.35f), 0.0f, 1.0f);
        const int schemaVersion = j.value("satMapSchemaVersion", 1);
        if (schemaVersion < 4 && preset != "Custom") {
            // Named presets own their distribution. Migrate the early SatMap
            // builds that either left High Percentile at 75% or forced every
            // biome to 98%. Both create broad false-color summit patches.
            normalizeLowPercentile = 2.0f;
            normalizeHighPercentile = presetHeightHighPercentile(preset);
            slopeBlend = std::max(slopeBlend, 0.95f);
        }
        const auto readColor = [&j](const char* key, float& r, float& g, float& b) {
            if (!j.contains(key) || !j[key].is_array() || j[key].size() < 3) return;
            r = j[key][0].get<float>(); g = j[key][1].get<float>(); b = j[key][2].get<float>();
        };
        readColor("freshSnowColor", freshSnowR, freshSnowG, freshSnowB);
        readColor("wetSnowColor", wetSnowR, wetSnowG, wetSnowB);
        readColor("dirtySnowColor", dirtySnowR, dirtySnowG, dirtySnowB);
        readColor("iceColor", iceR, iceG, iceB);
        if (j.contains("slopeColor") && j["slopeColor"].is_array() && j["slopeColor"].size() >= 3) {
            slopeR = j["slopeColor"][0].get<float>(); slopeG = j["slopeColor"][1].get<float>(); slopeB = j["slopeColor"][2].get<float>();
        }
        if (j.contains("flowColor") && j["flowColor"].is_array() && j["flowColor"].size() >= 3) {
            flowR = j["flowColor"][0].get<float>(); flowG = j["flowColor"][1].get<float>(); flowB = j["flowColor"][2].get<float>();
        }
        if (j.contains("soilColor") && j["soilColor"].is_array() && j["soilColor"].size() >= 3) {
            soilR = j["soilColor"][0].get<float>(); soilG = j["soilColor"][1].get<float>(); soilB = j["soilColor"][2].get<float>();
        }
        if (j.contains("grassColor") && j["grassColor"].is_array() && j["grassColor"].size() >= 3) {
            grassR = j["grassColor"][0].get<float>(); grassG = j["grassColor"][1].get<float>(); grassB = j["grassColor"][2].get<float>();
        }
        const auto readRamp = [&j](const char* key, std::vector<Stop>& ramp) {
            if (!j.contains(key) || !j[key].is_array()) return false;
            std::vector<Stop> loaded;
            for (const auto& item : j[key]) loaded.push_back({
                item.value("pos", 0.0f), item.value("r", 1.0f), item.value("g", 1.0f),
                item.value("b", 1.0f), item.value("a", 1.0f)});
            if (loaded.empty()) return false;
            std::sort(loaded.begin(), loaded.end(), [](const Stop& a, const Stop& b) { return a.pos < b.pos; });
            ramp = std::move(loaded);
            return true;
        };
        if (!readRamp("slopeStops", slopeStops)) slopeStops = makeMaskRamp(slopeR, slopeG, slopeB, false);
        if (!readRamp("flowStops", flowStops)) flowStops = makeMaskRamp(flowR, flowG, flowB, true);
        if (!readRamp("soilStops", soilStops)) soilStops = makeMaskRamp(soilR, soilG, soilB, false);
        if (!readRamp("grassStops", grassStops)) grassStops = makeMaskRamp(grassR, grassG, grassB, false);
        if (j.contains("stops") && j["stops"].is_array()) {
            stops.clear();
            for (const auto& sj : j["stops"]) {
                Stop s;
                s.pos = sj.value("pos", 0.0f);
                s.r = sj.value("r", 1.0f);
                s.g = sj.value("g", 1.0f);
                s.b = sj.value("b", 1.0f);
                s.a = sj.value("a", 1.0f);
                stops.push_back(s);
            }
            sortStops();
        }
        if (schemaVersion < 7 && preset != "Custom") {
            // v5-v7 expand sparse preset ramps, stop steep masks from
            // replacing the complete base color. Reload named presets so old
            // pale summit/slope endpoints do not survive migration. Custom
            // ramps remain byte-for-byte authored data.
            applyPreset(preset);
        }
        if (schemaVersion < 7 && std::abs(detailStrength - 0.08f) < 1e-5f) {
            detailStrength = 0.11f;
        }
        if (schemaVersion < 10 && preset != "Custom") applyPreset(preset);
        if (presetPropertyEdit && requestedPreset != "Custom") applyPreset(requestedPreset);
    }

    void TerrainSatMapColorRampNode::drawContent() {
        bool edited = false;
        const char* presets[] = {"Temperate", "Alpine", "Desert", "Tropical", "Boreal", "Volcanic", "Mediterranean", "Autumn",
                                 "Layer: Soil", "Layer: Flow", "Layer: Grass", "Layer: Rock",
                                 "Layer: Mud", "Layer: Moss", "Layer: Cavity", "Custom"};
        int presetIndex = 0;
        for (int i = 0; i < 16; ++i) if (preset == presets[i]) presetIndex = i;
        ImGui::Text("Terrain Color Preset");
        if (ImGui::Combo("##terrain_preset", &presetIndex, presets, 16)) {
            if (presetIndex < 15) applyPreset(presets[presetIndex]);
            else preset = "Custom";
            edited = true;
        }
        ImGui::TextDisabled("Height + optional Slope / Flow / Soil / Grass; Snow inputs are protected overlays.");
        if (ImGui::Checkbox("Auto Derive Missing Masks", &autoDeriveMasks)) edited = true;
        if (ImGui::SliderFloat("Slope Rock", &slopeBlend, 0.0f, 1.0f, "%.2f")) edited = true;
        if (ImGui::SliderFloat("Flow Wetness", &flowBlend, 0.0f, 1.0f, "%.2f")) edited = true;
        if (ImGui::SliderFloat("Soil Overlay", &soilBlend, 0.0f, 1.0f, "%.2f")) edited = true;
        if (ImGui::SliderFloat("Grass Overlay", &grassBlend, 0.0f, 1.0f, "%.2f")) edited = true;
        if (ImGui::CollapsingHeader("Paint Resolution Detail")) {
            ImGui::TextDisabled("Generated directly on the paint grid; not an upscale.");
            if (ImGui::SliderFloat("Detail Strength", &detailStrength, 0.0f, 0.35f, "%.3f")) edited = true;
            if (ImGui::DragFloat("Detail Scale", &detailScale, 1.0f, 4.0f, 2048.0f, "%.0f")) edited = true;
        }
        const char* debugViews[] = {"Final Color", "Height", "Slope", "Flow", "Soil", "Snow", "Ice", "Meltwater", "Avalanche", "Grass"};
        if (ImGui::Combo("Mask Debug View", &debugView, debugViews, 10)) edited = true;
        if (ImGui::CollapsingHeader("Protected Snow Overlay")) {
            if (ImGui::SliderFloat("Snow Coverage", &snowBlend, 0.0f, 1.0f, "%.2f")) edited = true;
            if (ImGui::SliderFloat("Melt Weathering", &meltBlend, 0.0f, 1.0f, "%.2f")) edited = true;
            if (ImGui::SliderFloat("Avalanche Dirt", &avalancheBlend, 0.0f, 1.0f, "%.2f")) edited = true;
            float fresh[3] = {freshSnowR, freshSnowG, freshSnowB};
            float wet[3] = {wetSnowR, wetSnowG, wetSnowB};
            float dirtySnow[3] = {dirtySnowR, dirtySnowG, dirtySnowB};
            float ice[3] = {iceR, iceG, iceB};
            if (ImGui::ColorEdit3("Fresh Snow", fresh)) {
                freshSnowR = fresh[0]; freshSnowG = fresh[1]; freshSnowB = fresh[2]; edited = true;
            }
            if (ImGui::ColorEdit3("Wet Snow", wet)) {
                wetSnowR = wet[0]; wetSnowG = wet[1]; wetSnowB = wet[2]; edited = true;
            }
            if (ImGui::ColorEdit3("Dirty Snow", dirtySnow)) {
                dirtySnowR = dirtySnow[0]; dirtySnowG = dirtySnow[1]; dirtySnowB = dirtySnow[2]; edited = true;
            }
            if (ImGui::ColorEdit3("Ice", ice)) {
                iceR = ice[0]; iceG = ice[1]; iceB = ice[2]; edited = true;
            }
        }
        if (ImGui::CollapsingHeader("Advanced Mask Ramps")) {
            const auto drawMaskRamp = [](const char* label, std::vector<Stop>& ramp) {
                bool changed = false;
                ImGui::PushID(label);
                if (ImGui::TreeNode(label)) {
                    int deleteIndex = -1;
                    for (size_t i = 0; i < ramp.size(); ++i) {
                        ImGui::PushID(static_cast<int>(i));
                        ImGui::SetNextItemWidth(105.0f);
                        changed |= ImGui::SliderFloat("##position", &ramp[i].pos, 0.0f, 1.0f, "%.2f");
                        ImGui::SameLine();
                        float color[3] = {ramp[i].r, ramp[i].g, ramp[i].b};
                        if (ImGui::ColorEdit3("##color", color, ImGuiColorEditFlags_NoInputs)) {
                            ramp[i].r = color[0]; ramp[i].g = color[1]; ramp[i].b = color[2];
                            changed = true;
                        }
                        ImGui::SameLine();
                        if (ImGui::SmallButton("X") && ramp.size() > 2) deleteIndex = static_cast<int>(i);
                        ImGui::PopID();
                    }
                    if (deleteIndex >= 0) {
                        ramp.erase(ramp.begin() + deleteIndex);
                        changed = true;
                    }
                    if (ramp.size() < 16 && ImGui::SmallButton("+ Add Color Stop")) {
                        float position = 0.5f;
                        if (ramp.size() >= 2) {
                            size_t gapIndex = 0;
                            float largestGap = -1.0f;
                            for (size_t i = 0; i + 1 < ramp.size(); ++i) {
                                const float gap = ramp[i + 1].pos - ramp[i].pos;
                                if (gap > largestGap) { largestGap = gap; gapIndex = i; }
                            }
                            position = (ramp[gapIndex].pos + ramp[gapIndex + 1].pos) * 0.5f;
                        }
                        Stop color = sampleSatRamp(ramp, position);
                        color.pos = position;
                        ramp.push_back(color);
                        changed = true;
                    }
                    ImGui::TreePop();
                }
                if (changed) {
                    std::sort(ramp.begin(), ramp.end(), [](const Stop& a, const Stop& b) { return a.pos < b.pos; });
                }
                ImGui::PopID();
                return changed;
            };
            const bool rampEdited = drawMaskRamp("Slope / Rock Colors", slopeStops) |
                                    drawMaskRamp("Flow / Wetness Colors", flowStops) |
                                    drawMaskRamp("Soil / Hardness Colors", soilStops) |
                                    drawMaskRamp("Grass / Vegetation Colors", grassStops);
            if (rampEdited) {
                preset = "Custom";
                edited = true;
            }
        }
        ImGui::Text("Gradient Stops:");
        if (ImGui::Checkbox("Auto Normalize Scalar", &autoNormalize)) edited = true;
        if (autoNormalize) {
            if (ImGui::SliderFloat("Low Percentile", &normalizeLowPercentile, 0.0f, 25.0f, "%.1f%%")) edited = true;
            if (ImGui::SliderFloat("High Percentile", &normalizeHighPercentile, 75.0f, 100.0f, "%.1f%%")) edited = true;
            if (normalizeHighPercentile < normalizeLowPercentile) {
                normalizeHighPercentile = normalizeLowPercentile;
                edited = true;
            }
            if (normalizeHighPercentile < 85.0f) {
                ImGui::TextColored(ImVec4(0.95f, 0.65f, 0.20f, 1.0f),
                    "Low high-percentile can flatten summit colors.");
                if (ImGui::SmallButton("Reset To Preset Distribution")) {
                    normalizeLowPercentile = 2.0f;
                    normalizeHighPercentile = presetHeightHighPercentile(preset);
                    edited = true;
                }
            }
        }

        bool heightRampEdited = false;
        int deleteIdx = -1;
        for (size_t i = 0; i < stops.size(); ++i) {
            ImGui::PushID((int)i);

            float pos = stops[i].pos;
            ImGui::SetNextItemWidth(100.0f);
            if (ImGui::SliderFloat("##pos", &pos, 0.0f, 1.0f, "Pos: %.2f")) {
                stops[i].pos = pos;
                edited = true;
                heightRampEdited = true;
            }

            ImGui::SameLine();
            float col[4] = { stops[i].r, stops[i].g, stops[i].b, stops[i].a };
            if (ImGui::ColorEdit4("##col", col, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoAlpha)) {
                stops[i].r = col[0];
                stops[i].g = col[1];
                stops[i].b = col[2];
                stops[i].a = col[3];
                edited = true;
                heightRampEdited = true;
            }

            ImGui::SameLine();
            if (ImGui::Button("X")) {
                deleteIdx = (int)i;
            }
            ImGui::PopID();
        }

        if (deleteIdx >= 0 && stops.size() > 2) {
            stops.erase(stops.begin() + deleteIdx);
            edited = true;
            heightRampEdited = true;
        }

        if (ImGui::Button("+ Add Stop")) {
            // Copy last stop, or create a neutral stop if the serialized
            // gradient was empty.
            Stop newStop = stops.empty()
                ? Stop{0.0f, 1.0f, 1.0f, 1.0f, 1.0f} : stops.back();
            newStop.pos = (std::min)(1.0f, newStop.pos + 0.1f);
            stops.push_back(newStop);
            edited = true;
            heightRampEdited = true;
        }

        if (heightRampEdited) preset = "Custom";

        if (edited) {
            sortStops();
            dirty = true;
        }
    }

    // --- TerrainGrassMaskNode ---

    TerrainGrassMaskNode::TerrainGrassMaskNode() {
        name = "Grass Mask";
        terrainNodeType = NodeType::GrassMask;
        inputs.push_back(NodeSystem::Pin::createInput(
            "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
        for (const char* label : {"Soil (optional)", "Flow (optional)", "Slope (optional)",
                                  "Wetness (optional)", "Hardness (optional)"}) {
            inputs.push_back(NodeSystem::Pin::createInput(
                label, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.back().acceptImageSemantic(NodeSystem::ImageSemantic::PhysicalScalar);
        }
        outputs.push_back(NodeSystem::Pin::createOutput(
            "Grass", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
        metadata.displayName = "Grass Mask";
        metadata.category = "Mask";
        metadata.description = "Reusable vegetation suitability from terrain fields";
        metadata.headerColor = IM_COL32(72, 154, 78, 255);
        headerColor = ImVec4(0.28f, 0.60f, 0.31f, 1.0f);
        applyPreset(GrassMaskPreset::Temperate);
    }

    const char* TerrainGrassMaskNode::presetName(GrassMaskPreset value) {
        switch (value) {
            case GrassMaskPreset::Lush: return "Lush";
            case GrassMaskPreset::Alpine: return "Alpine";
            case GrassMaskPreset::Arid: return "Arid";
            case GrassMaskPreset::Boreal: return "Boreal";
            case GrassMaskPreset::Custom: return "Custom";
            default: return "Temperate";
        }
    }

    void TerrainGrassMaskNode::applyPreset(GrassMaskPreset value) {
        preset = value;
        if (value == GrassMaskPreset::Lush) {
            density = .96f; maxSlope = .58f; soilInfluence = .86f; flowAvoidance = .68f;
            wetnessPreference = .70f; wetnessRange = .62f; hardnessAvoidance = .48f; patchiness = .18f;
        } else if (value == GrassMaskPreset::Alpine) {
            density = .62f; maxSlope = .42f; soilInfluence = .72f; flowAvoidance = .88f;
            wetnessPreference = .48f; wetnessRange = .42f; hardnessAvoidance = .72f; patchiness = .38f;
        } else if (value == GrassMaskPreset::Arid) {
            density = .34f; maxSlope = .48f; soilInfluence = .58f; flowAvoidance = .42f;
            wetnessPreference = .42f; wetnessRange = .28f; hardnessAvoidance = .54f; patchiness = .58f;
        } else if (value == GrassMaskPreset::Boreal) {
            density = .78f; maxSlope = .50f; soilInfluence = .82f; flowAvoidance = .76f;
            wetnessPreference = .62f; wetnessRange = .55f; hardnessAvoidance = .66f; patchiness = .30f;
        } else if (value == GrassMaskPreset::Temperate) {
            density = .82f; maxSlope = .52f; soilInfluence = .78f; flowAvoidance = .86f;
            wetnessPreference = .58f; wetnessRange = .52f; hardnessAvoidance = .62f; patchiness = .28f;
        }
    }

    NodeSystem::PinValue TerrainGrassMaskNode::compute(int, NodeSystem::EvaluationContext& ctx) {
        const auto height = getImageInput(0, ctx);
        if (!height.isValid() || height.channels != 1) {
            ctx.addError(id, "Grass Mask requires a scalar Height input");
            return NodeSystem::PinValue{};
        }
        const auto soilImage = getImageInput(1, ctx);
        const auto flowImage = getImageInput(2, ctx);
        const auto slopeImage = getImageInput(3, ctx);
        const auto wetImage = getImageInput(4, ctx);
        const auto hardImage = getImageInput(5, ctx);
        TerrainContext* tctx = getTerrainContext(ctx);
        TerrainObject* terrain = tctx ? tctx->terrain : nullptr;
        const int w = terrain ? terrain->paintGridWidth() : height.width;
        const int h = terrain ? terrain->paintGridHeight() : height.height;
        if (w < 2 || h < 2) return NodeSystem::PinValue{};

        NodeSystem::Image2DData result;
        result.width = w; result.height = h; result.channels = 1;
        result.semantic = NodeSystem::ImageSemantic::Mask;
        result.data = std::make_shared<std::vector<float>>(static_cast<size_t>(w) * h, 0.0f);
        const float worldScale = tctx ? std::max(tctx->scale_xz, 1e-3f) : static_cast<float>(w - 1);
        const float heightScale = tctx ? std::max(std::abs(tctx->scale_y), 1e-3f) : 1.0f;
        const float cell = worldScale / std::max(w - 1, 1);

        #pragma omp parallel for schedule(static)
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const float u = static_cast<float>(x) / (w - 1);
                const float v = static_cast<float>(y) / (h - 1);
                const float hard = hardImage.isValid()
                    ? std::clamp(sampleImageChannel(hardImage, u, v), 0.0f, 1.0f) : 0.35f;
                const float soil = soilImage.isValid()
                    ? std::clamp(sampleImageChannel(soilImage, u, v), 0.0f, 1.0f) : 1.0f - hard;
                const float flow = flowImage.isValid()
                    ? std::clamp(sampleImageChannel(flowImage, u, v), 0.0f, 1.0f) : 0.0f;
                const float wet = wetImage.isValid()
                    ? std::clamp(sampleImageChannel(wetImage, u, v), 0.0f, 1.0f)
                    : std::clamp(soil * .62f + flow * .28f, 0.0f, 1.0f);
                float slope = 0.0f;
                if (slopeImage.isValid()) slope = std::clamp(sampleImageChannel(slopeImage, u, v), 0.0f, 1.0f);
                else {
                    const float du = 1.0f / (w - 1), dv = 1.0f / (h - 1);
                    const float dx = (sampleImageChannel(height, u + du, v) -
                                      sampleImageChannel(height, u - du, v)) * heightScale /
                                     std::max(2.0f * cell, 1e-6f);
                    const float dz = (sampleImageChannel(height, u, v + dv) -
                                      sampleImageChannel(height, u, v - dv)) * heightScale /
                                     std::max(2.0f * cell, 1e-6f);
                    slope = std::atan(std::sqrt(dx * dx + dz * dz)) / 1.57079632679f;
                }
                const float slopeLo = std::max(0.0f, maxSlope - slopeSoftness);
                const float slopeHi = std::min(1.0f, maxSlope + slopeSoftness);
                float slopeT = std::clamp((slope - slopeLo) /
                    std::max(slopeHi - slopeLo, 1e-5f), 0.0f, 1.0f);
                slopeT = slopeT * slopeT * (3.0f - 2.0f * slopeT);
                const float moisture = 1.0f - std::clamp(
                    std::abs(wet - wetnessPreference) / std::max(wetnessRange, .02f), 0.0f, 1.0f);
                const float patch = 1.0f - patchiness + patchiness * satFbm(
                    u * detailScale + seed * .017f, v * detailScale - seed * .011f);
                const float value = density * (.22f + soil * soilInfluence) * (1.0f - slopeT) *
                    (1.0f - flow * flowAvoidance) * (1.0f - hard * hardnessAvoidance) *
                    (.45f + moisture * .55f) * patch;
                (*result.data)[static_cast<size_t>(y) * w + x] = std::clamp(value, 0.0f, 1.0f);
            }
        }
        return NodeSystem::PinValue{result};
    }

    void TerrainGrassMaskNode::drawContent() {
        const char* names[] = {"Temperate", "Lush", "Alpine", "Arid", "Boreal", "Custom"};
        int selected = static_cast<int>(preset);
        if (ImGui::Combo("Preset", &selected, names, 6)) {
            applyPreset(static_cast<GrassMaskPreset>(selected)); dirty = true;
        }
        bool edited = false;
        edited |= ImGui::SliderFloat("Density", &density, 0.0f, 1.0f);
        edited |= ImGui::SliderFloat("Max Slope", &maxSlope, 0.0f, 1.0f);
        edited |= ImGui::SliderFloat("Slope Softness", &slopeSoftness, 0.01f, 0.5f);
        edited |= ImGui::SliderFloat("Soil Need", &soilInfluence, 0.0f, 1.0f);
        edited |= ImGui::SliderFloat("Flow Avoidance", &flowAvoidance, 0.0f, 1.0f);
        edited |= ImGui::SliderFloat("Wetness Target", &wetnessPreference, 0.0f, 1.0f);
        edited |= ImGui::SliderFloat("Wetness Range", &wetnessRange, 0.02f, 1.0f);
        edited |= ImGui::SliderFloat("Hardness Avoid", &hardnessAvoidance, 0.0f, 1.0f);
        edited |= ImGui::SliderFloat("Patchiness", &patchiness, 0.0f, 1.0f);
        edited |= ImGui::DragFloat("Detail Scale", &detailScale, 1.0f, 4.0f, 2048.0f, "%.0f");
        edited |= ImGui::DragInt("Seed", &seed, 1.0f);
        if (edited) { preset = GrassMaskPreset::Custom; dirty = true; }
    }

    void TerrainGrassMaskNode::serializeToJson(nlohmann::json& j) const {
        TerrainNodeBase::serializeToJson(j);
        j["preset"] = static_cast<int>(preset); j["density"] = density;
        j["maxSlope"] = maxSlope; j["slopeSoftness"] = slopeSoftness;
        j["soilInfluence"] = soilInfluence; j["flowAvoidance"] = flowAvoidance;
        j["wetnessPreference"] = wetnessPreference; j["wetnessRange"] = wetnessRange;
        j["hardnessAvoidance"] = hardnessAvoidance; j["patchiness"] = patchiness;
        j["detailScale"] = detailScale; j["seed"] = seed;
    }

    void TerrainGrassMaskNode::deserializeFromJson(const nlohmann::json& j) {
        TerrainNodeBase::deserializeFromJson(j);
        const GrassMaskPreset previous = preset;
        const auto requested = static_cast<GrassMaskPreset>(std::clamp(j.value("preset", 0), 0, 5));
        density = std::clamp(j.value("density", density), 0.0f, 1.0f);
        maxSlope = std::clamp(j.value("maxSlope", maxSlope), 0.0f, 1.0f);
        slopeSoftness = std::clamp(j.value("slopeSoftness", slopeSoftness), .01f, .5f);
        soilInfluence = std::clamp(j.value("soilInfluence", soilInfluence), 0.0f, 1.0f);
        flowAvoidance = std::clamp(j.value("flowAvoidance", flowAvoidance), 0.0f, 1.0f);
        wetnessPreference = std::clamp(j.value("wetnessPreference", wetnessPreference), 0.0f, 1.0f);
        wetnessRange = std::clamp(j.value("wetnessRange", wetnessRange), .02f, 1.0f);
        hardnessAvoidance = std::clamp(j.value("hardnessAvoidance", hardnessAvoidance), 0.0f, 1.0f);
        patchiness = std::clamp(j.value("patchiness", patchiness), 0.0f, 1.0f);
        detailScale = std::clamp(j.value("detailScale", detailScale), 4.0f, 2048.0f);
        seed = j.value("seed", seed);
        preset = requested;
        // A changed preset is an operation and should load its complete recipe.
        // Otherwise preserve individual scalar edits made through JSON-backed
        // scripting/IPC reflection, even while a named preset label is present.
        if (requested != previous && requested != GrassMaskPreset::Custom) applyPreset(requested);
    }

    // --- TerrainSatMapBlendNode ---

    TerrainSatMapBlendNode::TerrainSatMapBlendNode() {
        name = "SatMap Blend";
        terrainNodeType = NodeType::SatMapBlend;
        inputs.push_back(NodeSystem::Pin::createInput(
            "Base Color", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Albedo));
        inputs.push_back(NodeSystem::Pin::createInput(
            "Layer Color", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Albedo));
        inputs.push_back(NodeSystem::Pin::createInput(
            "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
        outputs.push_back(NodeSystem::Pin::createOutput(
            "Color", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Albedo));
        metadata.displayName = "SatMap Blend";
        metadata.category = "Texture";
        metadata.description = "Blend two SatMap RGBA colors through a scalar mask";
        metadata.headerColor = IM_COL32(188, 112, 62, 255);
        headerColor = ImVec4(.74f, .44f, .24f, 1.0f);
    }

    NodeSystem::PinValue TerrainSatMapBlendNode::compute(int, NodeSystem::EvaluationContext& ctx) {
        const auto base = getImageInput(0, ctx);
        const auto layer = getImageInput(1, ctx);
        const auto mask = getImageInput(2, ctx);
        if (!base.isValid() || base.channels != 4 || !layer.isValid() || layer.channels != 4 ||
            !mask.isValid() || mask.channels != 1) {
            ctx.addError(id, "SatMap Blend requires Base RGBA, Layer RGBA and scalar Mask");
            return NodeSystem::PinValue{};
        }
        TerrainContext* tctx = getTerrainContext(ctx);
        TerrainObject* terrain = tctx ? tctx->terrain : nullptr;
        const int w = terrain ? terrain->paintGridWidth() : base.width;
        const int h = terrain ? terrain->paintGridHeight() : base.height;
        NodeSystem::Image2DData result;
        result.width = w; result.height = h; result.channels = 4;
        result.semantic = NodeSystem::ImageSemantic::Albedo;
        result.data = std::make_shared<std::vector<float>>(static_cast<size_t>(w) * h * 4, 0.0f);
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            const float u = static_cast<float>(x) / std::max(w - 1, 1);
            const float v = static_cast<float>(y) / std::max(h - 1, 1);
            float amount = std::clamp(sampleImageChannel(mask, u, v), 0.0f, 1.0f);
            if (invertMask) amount = 1.0f - amount;
            amount = std::pow(amount, std::max(maskPower, .01f)) * opacity;
            amount = std::clamp(amount, 0.0f, 1.0f);
            const size_t pixel = (static_cast<size_t>(y) * w + x) * 4;
            for (int channel = 0; channel < 4; ++channel) {
                const float a = sampleImageChannel(base, u, v, channel);
                const float b = sampleImageChannel(layer, u, v, channel);
                (*result.data)[pixel + channel] = a + (b - a) * amount;
            }
        }
        return NodeSystem::PinValue{result};
    }

    void TerrainSatMapBlendNode::drawContent() {
        bool edited = false;
        edited |= ImGui::SliderFloat("Opacity", &opacity, 0.0f, 1.0f);
        edited |= ImGui::DragFloat("Mask Power", &maskPower, .02f, .05f, 8.0f);
        edited |= ImGui::Checkbox("Invert Mask", &invertMask);
        if (edited) dirty = true;
    }

    void TerrainSatMapBlendNode::serializeToJson(nlohmann::json& j) const {
        TerrainNodeBase::serializeToJson(j);
        j["opacity"] = opacity; j["maskPower"] = maskPower; j["invertMask"] = invertMask;
    }

    void TerrainSatMapBlendNode::deserializeFromJson(const nlohmann::json& j) {
        TerrainNodeBase::deserializeFromJson(j);
        opacity = std::clamp(j.value("opacity", opacity), 0.0f, 1.0f);
        maskPower = std::clamp(j.value("maskPower", maskPower), .05f, 8.0f);
        invertMask = j.value("invertMask", invertMask);
    }

    NodeSystem::PinValue computeAdaptiveCurvatureMask(
        CurvatureMaskNode& node, NodeSystem::EvaluationContext& ctx) {
        const auto input = node.getHeightInput(0, ctx);
        if (!input.isValid() || input.channels != 1 || input.width < 2 || input.height < 2) {
            ctx.addError(node.id, "Curvature Mask requires at least a 2x2 Height input");
            return NodeSystem::PinValue{};
        }
        NodeSystem::Image2DData raw;
        raw.width = input.width; raw.height = input.height; raw.channels = 1;
        raw.semantic = NodeSystem::ImageSemantic::PhysicalScalar;
        raw.data = std::make_shared<std::vector<float>>(
            static_cast<size_t>(raw.width) * raw.height, 0.0f);
        float peakCurve = 0.0f;
        for (int y = 0; y < raw.height; ++y) {
            const int y0 = std::max(0, y - 1), y1 = std::min(raw.height - 1, y + 1);
            for (int x = 0; x < raw.width; ++x) {
                const int x0 = std::max(0, x - 1), x1 = std::min(raw.width - 1, x + 1);
                const size_t index = static_cast<size_t>(y) * raw.width + x;
                const float center = (*input.data)[index];
                const float localAverage = ((*input.data)[static_cast<size_t>(y) * raw.width + x0] +
                    (*input.data)[static_cast<size_t>(y) * raw.width + x1] +
                    (*input.data)[static_cast<size_t>(y0) * raw.width + x] +
                    (*input.data)[static_cast<size_t>(y1) * raw.width + x]) * 0.25f;
                const float signedCurve = localAverage - center;
                (*raw.data)[index] = std::max(node.selectConvex ? -signedCurve : signedCurve, 0.0f);
                peakCurve = std::max(peakCurve, (*raw.data)[index]);
            }
        }
        NormalizedScalarField normalized = normalizeScalarField(raw, true, 0.0f, 98.0f);
        if (peakCurve <= 1e-8f) std::fill(normalized.values.begin(), normalized.values.end(), 0.0f);
        else for (float& value : normalized.values)
            value = std::clamp((value - .06f) / .86f, 0.0f, 1.0f);
        auto result = node.createMaskOutput(raw.width, raw.height);
        const float low = std::min(node.minCurve, node.maxCurve);
        const float high = std::max(node.minCurve, node.maxCurve);
        const float span = std::max(high - low, 1e-5f);
        for (size_t i = 0; i < normalized.values.size(); ++i)
            (*result.data)[i] = std::clamp((normalized.values[i] - low) / span, 0.0f, 1.0f);
        return NodeSystem::PinValue{result};
    }

    // --- TerrainSurfaceMasksNode ---

    TerrainSurfaceMasksNode::TerrainSurfaceMasksNode() {
        name = "Surface Detail Masks";
        terrainNodeType = NodeType::SurfaceMasks;
        inputs.push_back(NodeSystem::Pin::createInput(
            "Height", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Height));
        for (const char* label : {"Wetness (optional)", "Flow (optional)",
                                  "Soil (optional)", "Exposure (optional)"}) {
            inputs.push_back(NodeSystem::Pin::createInput(
                label, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask, true));
            inputs.back().acceptImageSemantic(NodeSystem::ImageSemantic::PhysicalScalar);
        }
        for (const char* label : {"Cavity", "Mud", "Moss"})
            outputs.push_back(NodeSystem::Pin::createOutput(
                label, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
        metadata.displayName = "Surface Detail Masks";
        metadata.category = "Mask";
        metadata.description = "Paint-resolution cavity, mud and moss material masks";
        metadata.headerColor = IM_COL32(84, 132, 92, 255);
        headerColor = ImVec4(.33f, .52f, .36f, 1.0f);
        applyPreset("Temperate");
    }

    void TerrainSurfaceMasksNode::applyPreset(const std::string& value) {
        preset = value;
        if (value == "Humid") {
            cavityPower = .68f; mudStrength = .94f; mossStrength = .96f; slopeSuppression = .62f;
        } else if (value == "Arid") {
            cavityPower = .92f; mudStrength = .38f; mossStrength = .16f; slopeSuppression = .80f;
        } else if (value == "Alpine") {
            cavityPower = .76f; mudStrength = .52f; mossStrength = .42f; slopeSuppression = .58f;
        } else {
            preset = "Temperate";
            cavityPower = .80f; mudStrength = .82f; mossStrength = .72f; slopeSuppression = .72f;
        }
    }

    NodeSystem::PinValue TerrainSurfaceMasksNode::compute(
        int outputIndex, NodeSystem::EvaluationContext& ctx) {
        const auto height = getImageInput(0, ctx);
        if (!height.isValid() || height.channels != 1 || height.width < 2 || height.height < 2) {
            ctx.addError(id, "Surface Detail Masks requires at least a 2x2 Height input");
            return NodeSystem::PinValue{};
        }
        const auto wetImage = getImageInput(1, ctx);
        const auto flowImage = getImageInput(2, ctx);
        const auto soilImage = getImageInput(3, ctx);
        const auto exposureImage = getImageInput(4, ctx);
        TerrainContext* tctx = getTerrainContext(ctx);
        TerrainObject* terrain = tctx ? tctx->terrain : nullptr;
        const int w = terrain ? terrain->paintGridWidth() : height.width;
        const int h = terrain ? terrain->paintGridHeight() : height.height;

        NodeSystem::Image2DData rawCavity;
        rawCavity.width = height.width; rawCavity.height = height.height; rawCavity.channels = 1;
        rawCavity.semantic = NodeSystem::ImageSemantic::PhysicalScalar;
        rawCavity.data = std::make_shared<std::vector<float>>(
            static_cast<size_t>(height.width) * height.height, 0.0f);
        float peakCavity = 0.0f;
        for (int y = 0; y < height.height; ++y) {
            const int y0 = std::max(0, y - 1), y1 = std::min(height.height - 1, y + 1);
            for (int x = 0; x < height.width; ++x) {
                const int x0 = std::max(0, x - 1), x1 = std::min(height.width - 1, x + 1);
                const size_t index = static_cast<size_t>(y) * height.width + x;
                const float center = (*height.data)[index];
                const float average = ((*height.data)[static_cast<size_t>(y) * height.width + x0] +
                    (*height.data)[static_cast<size_t>(y) * height.width + x1] +
                    (*height.data)[static_cast<size_t>(y0) * height.width + x] +
                    (*height.data)[static_cast<size_t>(y1) * height.width + x]) * .25f;
                (*rawCavity.data)[index] = std::max(average - center, 0.0f);
                peakCavity = std::max(peakCavity, (*rawCavity.data)[index]);
            }
        }
        NormalizedScalarField cavityField = normalizeScalarField(rawCavity, true, 0.0f, 98.0f);
        if (peakCavity <= 1e-8f) std::fill(cavityField.values.begin(), cavityField.values.end(), 0.0f);
        else for (float& value : cavityField.values)
            value = std::clamp((value - .06f) / .86f, 0.0f, 1.0f);
        std::array<NodeSystem::Image2DData, 3> result;
        for (auto& image : result) {
            image.width = w; image.height = h; image.channels = 1;
            image.semantic = NodeSystem::ImageSemantic::Mask;
            image.data = std::make_shared<std::vector<float>>(static_cast<size_t>(w) * h, 0.0f);
        }
        const float worldScale = tctx ? std::max(tctx->scale_xz, 1e-3f) : static_cast<float>(w - 1);
        const float heightScale = tctx ? std::max(std::abs(tctx->scale_y), 1e-3f) : 1.0f;
        const float cell = worldScale / std::max(w - 1, 1);
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            const float u = static_cast<float>(x) / std::max(w - 1, 1);
            const float v = static_cast<float>(y) / std::max(h - 1, 1);
            const float du = 1.0f / std::max(w - 1, 1), dv = 1.0f / std::max(h - 1, 1);
            const float dx = (sampleImageChannel(height, u + du, v) - sampleImageChannel(height, u - du, v)) *
                heightScale / std::max(2.0f * cell, 1e-6f);
            const float dz = (sampleImageChannel(height, u, v + dv) - sampleImageChannel(height, u, v - dv)) *
                heightScale / std::max(2.0f * cell, 1e-6f);
            const float slope = std::atan(std::sqrt(dx * dx + dz * dz)) / 1.57079632679f;
            const float cavity = std::pow(std::clamp(
                sampleScalarField(cavityField, x, y, w, h), 0.0f, 1.0f), cavityPower);
            const float flow = flowImage.isValid() ? std::clamp(sampleImageChannel(flowImage, u, v), 0.0f, 1.0f) : 0.0f;
            const float soil = soilImage.isValid() ? std::clamp(sampleImageChannel(soilImage, u, v), 0.0f, 1.0f) : .65f;
            const float wet = wetImage.isValid() ? std::clamp(sampleImageChannel(wetImage, u, v), 0.0f, 1.0f) :
                std::clamp(soil * .55f + flow * .45f, 0.0f, 1.0f);
            const float exposure = exposureImage.isValid() ?
                std::clamp(sampleImageChannel(exposureImage, u, v), 0.0f, 1.0f) : .5f;
            const float detail = .72f + .28f * satFbm(
                u * detailScale + seed * .019f, v * detailScale - seed * .013f);
            const float flatness = std::clamp(1.0f - slope * slopeSuppression, 0.0f, 1.0f);
            const float mud = std::clamp(mudStrength * soil * (.30f + wet * .70f) *
                (.58f + flow * .42f) * flatness * detail, 0.0f, 1.0f);
            const float moss = std::clamp(mossStrength * wet * (.30f + cavity * .70f) *
                (1.0f - exposure * .58f) * flatness * detail, 0.0f, 1.0f);
            const size_t index = static_cast<size_t>(y) * w + x;
            (*result[0].data)[index] = cavity;
            (*result[1].data)[index] = mud;
            (*result[2].data)[index] = moss;
        }
        return outputIndex >= 0 && outputIndex < 3
            ? NodeSystem::PinValue{result[outputIndex]} : NodeSystem::PinValue{};
    }

    void TerrainSurfaceMasksNode::drawContent() {
        const char* presets[] = {"Temperate", "Humid", "Arid", "Alpine", "Custom"};
        int selected = 0;
        for (int i = 0; i < 5; ++i) if (preset == presets[i]) selected = i;
        if (ImGui::Combo("Preset", &selected, presets, 5)) {
            if (selected < 4) applyPreset(presets[selected]); else preset = "Custom";
            dirty = true;
        }
        bool edited = false;
        edited |= ImGui::SliderFloat("Cavity Contrast", &cavityPower, .25f, 3.0f);
        edited |= ImGui::SliderFloat("Mud Strength", &mudStrength, 0.0f, 1.5f);
        edited |= ImGui::SliderFloat("Moss Strength", &mossStrength, 0.0f, 1.5f);
        edited |= ImGui::SliderFloat("Slope Suppression", &slopeSuppression, 0.0f, 1.5f);
        edited |= ImGui::DragFloat("Detail Scale", &detailScale, 1.0f, 4.0f, 2048.0f, "%.0f");
        edited |= ImGui::DragInt("Seed", &seed, 1.0f);
        if (edited) { preset = "Custom"; dirty = true; }
    }

    void TerrainSurfaceMasksNode::serializeToJson(nlohmann::json& j) const {
        TerrainNodeBase::serializeToJson(j);
        j["preset"] = preset; j["cavityPower"] = cavityPower;
        j["mudStrength"] = mudStrength; j["mossStrength"] = mossStrength;
        j["slopeSuppression"] = slopeSuppression; j["detailScale"] = detailScale; j["seed"] = seed;
    }

    void TerrainSurfaceMasksNode::deserializeFromJson(const nlohmann::json& j) {
        TerrainNodeBase::deserializeFromJson(j);
        const std::string previous = preset;
        const std::string requested = j.value("preset", preset);
        cavityPower = std::clamp(j.value("cavityPower", cavityPower), .25f, 3.0f);
        mudStrength = std::clamp(j.value("mudStrength", mudStrength), 0.0f, 1.5f);
        mossStrength = std::clamp(j.value("mossStrength", mossStrength), 0.0f, 1.5f);
        slopeSuppression = std::clamp(j.value("slopeSuppression", slopeSuppression), 0.0f, 1.5f);
        detailScale = std::clamp(j.value("detailScale", detailScale), 4.0f, 2048.0f);
        seed = j.value("seed", seed); preset = requested;
        if (requested != previous && requested != "Custom") applyPreset(requested);
    }

    TerrainPaintMaskCombineNode::TerrainPaintMaskCombineNode() {
        name = "Paint Mask Combine";
        terrainNodeType = NodeType::PaintMaskCombine;
        for (const char* label : {"Mask A", "Mask B"}) {
            inputs.push_back(NodeSystem::Pin::createInput(
                label, NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
            inputs.back().acceptImageSemantic(NodeSystem::ImageSemantic::Height);
            inputs.back().acceptImageSemantic(NodeSystem::ImageSemantic::PhysicalScalar);
        }
        outputs.push_back(NodeSystem::Pin::createOutput(
            "Mask", NodeSystem::DataType::Image2D, NodeSystem::ImageSemantic::Mask));
        metadata.displayName = "Paint Mask Combine";
        metadata.category = "Mask";
        metadata.description = "Multiply masks at terrain paint resolution, resampling inputs as needed";
        metadata.headerColor = IM_COL32(171, 120, 194, 255);
        headerColor = ImVec4(.67f, .47f, .76f, 1.0f);
    }

    NodeSystem::PinValue TerrainPaintMaskCombineNode::compute(
        int, NodeSystem::EvaluationContext& ctx) {
        const auto first = getImageInput(0, ctx);
        const auto second = getImageInput(1, ctx);
        if (!first.isValid() || first.channels != 1 || !second.isValid() || second.channels != 1) {
            ctx.addError(id, "Paint Mask Combine requires two scalar masks");
            return NodeSystem::PinValue{};
        }
        TerrainContext* tctx = getTerrainContext(ctx);
        TerrainObject* terrain = tctx ? tctx->terrain : nullptr;
        const int w = terrain ? terrain->paintGridWidth() : std::max(first.width, second.width);
        const int h = terrain ? terrain->paintGridHeight() : std::max(first.height, second.height);
        auto result = createMaskOutput(w, h);
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            const float u = static_cast<float>(x) / std::max(w - 1, 1);
            const float v = static_cast<float>(y) / std::max(h - 1, 1);
            (*result.data)[static_cast<size_t>(y) * w + x] = std::clamp(
                sampleImageChannel(first, u, v) * sampleImageChannel(second, u, v), 0.0f, 1.0f);
        }
        return NodeSystem::PinValue{result};
    }

    void TerrainPaintMaskCombineNode::drawContent() {
        ImGui::TextDisabled("A x B at Paint Resolution");
    }

    // --- TerrainSatMapOutputNode ---

    NodeSystem::PinValue TerrainSatMapOutputNode::compute(int outputIndex, NodeSystem::EvaluationContext& ctx) {
        if (!publicationEnabled) return NodeSystem::PinValue{}; // Output is skipped

        auto colorImg = getImageInput(0, ctx);
        if (!colorImg.isValid() || colorImg.channels != 4) {
            ctx.addError(id, "SatMap input must be a valid RGBA image");
            return NodeSystem::PinValue{};
        }

        NodeSystem::PinValue strVal = getInputValue(1, ctx);
        if (auto* pf = std::get_if<float>(&strVal)) {
            strength = *pf;
        }

        TerrainContext* tCtx = getTerrainContext(ctx);
        if (!tCtx || !tCtx->terrain) return NodeSystem::PinValue{};

        TerrainObject* terrain = tCtx->terrain;

        // SatMap and splat textures share the same terrain UV contract. Graph
        // images use terrain row order (row 0 = local Z / UV 0), while
        // Texture::pixels uses image-storage order (row 0 is sampled at UV 1).
        // Resample to the paint grid and perform the single required Y flip at
        // this output boundary, exactly like SplatOutputNode.
        const int sourceWidth = colorImg.width;
        const int sourceHeight = colorImg.height;
        const int w = terrain->paintGridWidth();
        const int h = terrain->paintGridHeight();
        if (w <= 0 || h <= 0) {
            ctx.addError(id, "Terrain paint resolution is invalid");
            return NodeSystem::PinValue{};
        }

        if (!terrain->macroColorMap) {
            terrain->macroColorMap = std::make_shared<Texture>(nullptr, TextureType::Albedo, "MacroColorMap");
        }
        terrain->macroColorMap->width = w;
        terrain->macroColorMap->height = h;

        // Populate CompactVec4 pixels for GPU upload compatibility
        auto& pixels = terrain->macroColorMap->pixels;
        pixels.resize(w * h);

        const auto& source = *colorImg.data;
        for (int y = 0; y < h; ++y) {
            const int terrainRow = (h - 1) - y;
            const float sourceY = std::clamp(
                ((static_cast<float>(terrainRow) + 0.5f) * sourceHeight / h) - 0.5f,
                0.0f, static_cast<float>(sourceHeight - 1));
            const int y0 = static_cast<int>(std::floor(sourceY));
            const int y1 = std::min(y0 + 1, sourceHeight - 1);
            const float ty = sourceY - static_cast<float>(y0);
            for (int x = 0; x < w; ++x) {
                const float sourceX = std::clamp(
                    ((static_cast<float>(x) + 0.5f) * sourceWidth / w) - 0.5f,
                    0.0f, static_cast<float>(sourceWidth - 1));
                const int x0 = static_cast<int>(std::floor(sourceX));
                const int x1 = std::min(x0 + 1, sourceWidth - 1);
                const float tx = sourceX - static_cast<float>(x0);
                auto& pixel = pixels[static_cast<size_t>(y) * w + x];
                uint8_t* channels[4] = {&pixel.r, &pixel.g, &pixel.b, &pixel.a};
                for (int channel = 0; channel < 4; ++channel) {
                    const float v00 = source[(static_cast<size_t>(y0) * sourceWidth + x0) * 4 + channel];
                    const float v10 = source[(static_cast<size_t>(y0) * sourceWidth + x1) * 4 + channel];
                    const float v01 = source[(static_cast<size_t>(y1) * sourceWidth + x0) * 4 + channel];
                    const float v11 = source[(static_cast<size_t>(y1) * sourceWidth + x1) * 4 + channel];
                    const float top = v00 + (v10 - v00) * tx;
                    const float bottom = v01 + (v11 - v01) * tx;
                    const float value = std::clamp(top + (bottom - top) * ty, 0.0f, 1.0f);
                    *channels[channel] = static_cast<uint8_t>(value * 255.0f + 0.5f);
                }
            }
        }

        terrain->macroColorMap->m_is_loaded = true;
        terrain->macroColorMap->m_uid = Texture::nextUid(); // Force GPU texture reload
        terrain->macro_color_strength = strength;

        return NodeSystem::PinValue{};
    }

    void TerrainSatMapOutputNode::serializeToJson(nlohmann::json& j) const {
        TerrainNodeBase::serializeToJson(j);
        j["strength"] = strength;
    }

    void TerrainSatMapOutputNode::deserializeFromJson(const nlohmann::json& j) {
        TerrainNodeBase::deserializeFromJson(j);
        strength = std::clamp(j.value("strength", 1.0f), 0.0f, 1.0f);
    }

    void TerrainSatMapOutputNode::drawContent() {
        if (ImGui::SliderFloat("Strength", &strength, 0.0f, 1.0f)) {
            dirty = true;
        }
    }

} // namespace TerrainNodesV2

namespace {
    NodeSystem::AutoRegisterNode<TerrainNodesV2::TerrainSatMapColorRampNode> reg_SatMapColorRamp("Terrain.SatMapColorRamp");
    NodeSystem::AutoRegisterNode<TerrainNodesV2::TerrainSatMapBlendNode>      reg_SatMapBlend("Terrain.SatMapBlend");
    NodeSystem::AutoRegisterNode<TerrainNodesV2::TerrainGrassMaskNode>       reg_GrassMask("Terrain.GrassMask");
    NodeSystem::AutoRegisterNode<TerrainNodesV2::TerrainSurfaceMasksNode>    reg_SurfaceMasks("Terrain.SurfaceMasks");
    NodeSystem::AutoRegisterNode<TerrainNodesV2::TerrainPaintMaskCombineNode> reg_PaintMaskCombine("Terrain.PaintMaskCombine");
    NodeSystem::AutoRegisterNode<TerrainNodesV2::TerrainSatMapOutputNode>    reg_SatMapOutput("Terrain.SatMapOutput");
}
