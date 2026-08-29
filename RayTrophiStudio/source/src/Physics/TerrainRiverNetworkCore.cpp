#include "TerrainRiverNetworkCore.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

namespace TerrainHydrology {
    namespace {
        constexpr int kDx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
        constexpr int kDy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
        constexpr float kInvSqrtTwo = 0.7071067811865475f;
        constexpr float kUnitDx[8] = {
            -kInvSqrtTwo, 0.0f, kInvSqrtTwo, -1.0f,
             1.0f, -kInvSqrtTwo, 0.0f, kInvSqrtTwo
        };
        constexpr float kUnitDy[8] = {
            -kInvSqrtTwo, -1.0f, -kInvSqrtTwo, 0.0f,
             0.0f, kInvSqrtTwo, 1.0f, kInvSqrtTwo
        };

        bool fail(std::string* error, const char* message) {
            if (error) *error = message;
            return false;
        }

        bool hasFieldSize(const std::vector<float>& values,
                          std::size_t pixelCount,
                          int channels) {
            return channels > 0 && values.size() == pixelCount * static_cast<std::size_t>(channels);
        }

        float firstChannel(const std::vector<float>& values, int channels, std::size_t pixelIndex) {
            return values[pixelIndex * static_cast<std::size_t>(channels)];
        }
    }

    int decodeFlowDirection(const std::vector<float>& values,
                            int channels,
                            std::size_t pixelIndex) {
        if (channels <= 0) return -1;
        const std::size_t base = pixelIndex * static_cast<std::size_t>(channels);
        if (base >= values.size()) return -1;

        if (channels == 1) {
            const float encoded = values[base];
            if (!std::isfinite(encoded)) return -1;
            const int direction = static_cast<int>(std::lround(encoded * 9.0f)) - 1;
            return direction >= 0 && direction < 8 ? direction : -1;
        }

        if (base + 1 >= values.size()) return -1;
        const float vx = values[base];
        const float vy = values[base + 1];
        const float lengthSquared = vx * vx + vy * vy;
        if (!std::isfinite(lengthSquared) || lengthSquared <= 1.0e-20f) return -1;

        int bestDirection = -1;
        float bestDot = -(std::numeric_limits<float>::max)();
        for (int direction = 0; direction < 8; ++direction) {
            const float dot = vx * kUnitDx[direction] + vy * kUnitDy[direction];
            if (dot > bestDot) {
                bestDot = dot;
                bestDirection = direction;
            }
        }
        return bestDirection;
    }

    bool buildRiverNetwork(int width,
                           int height,
                           const std::vector<float>& accumulation,
                           int accumulationChannels,
                           const std::vector<float>& direction,
                           int directionChannels,
                           const std::vector<float>* catchmentArea,
                           const std::vector<float>* lakeMask,
                           const std::vector<float>* lakeSpillPoints,
                           const RiverNetworkParams& params,
                           RiverNetworkResult& result,
                           std::string* error) {
        if (width <= 0 || height <= 0) return fail(error, "invalid river-network dimensions");
        const std::size_t pixelCount = static_cast<std::size_t>(width) * height;
        if (!hasFieldSize(accumulation, pixelCount, accumulationChannels))
            return fail(error, "invalid accumulation field layout");
        if (!hasFieldSize(direction, pixelCount, directionChannels))
            return fail(error, "invalid flow-direction field layout");
        if (catchmentArea && catchmentArea->size() != pixelCount)
            return fail(error, "invalid catchment-area field layout");
        if (lakeMask && lakeMask->size() != pixelCount)
            return fail(error, "invalid lake-mask field layout");
        if (lakeSpillPoints && lakeSpillPoints->size() != pixelCount)
            return fail(error, "invalid lake-spill field layout");

        result = {};
        std::vector<int> directParent(pixelCount, -1);
        std::vector<uint8_t> lakeBlocked(pixelCount, 0u);
        result.active.assign(pixelCount, 0u);
        const float minimumArea = (std::max)(params.minimumCatchmentAreaSquareMeters, 1.0e-6f);

        for (std::size_t index = 0; index < pixelCount; ++index) {
            const int flowDirection = decodeFlowDirection(direction, directionChannels, index);
            if (flowDirection >= 0) {
                const int x = static_cast<int>(index % static_cast<std::size_t>(width));
                const int y = static_cast<int>(index / static_cast<std::size_t>(width));
                const int nx = x + kDx[flowDirection];
                const int ny = y + kDy[flowDirection];
                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                    directParent[index] = ny * width + nx;
            }

            const float accumulated = firstChannel(accumulation, accumulationChannels, index);
            const bool channelActive = catchmentArea
                ? (*catchmentArea)[index] >= minimumArea
                : params.accumulationIsPhysicalArea
                    ? accumulated >= minimumArea
                    : accumulated >= params.legacyAccumulationThreshold;
            const bool insideLake = lakeMask && (*lakeMask)[index] >= 0.5f;
            lakeBlocked[index] = insideLake ? 1u : 0u;
            result.active[index] = channelActive && !insideLake ? 1u : 0u;
        }

        // A lake is a hydrological super-node, not a hole in the drainage
        // graph. Visible channels still remain outside its footprint, but an
        // inlet must transfer its hierarchy to the dry outlet instead of
        // making that outlet look like a new headwater. First try the paired
        // direction field; conditioned D8 fields normally cross the lake. If
        // a vector field is flat inside standing water, Lake Basin's explicit
        // spill point supplies the same transfer without inventing a river
        // ribbon across the lake surface.
        std::vector<int> lakeOutlet(pixelCount, -1);
        if (lakeMask && lakeSpillPoints) {
            std::vector<uint8_t> visited(pixelCount, 0u);
            std::queue<int> frontier;
            for (std::size_t seed = 0; seed < pixelCount; ++seed) {
                if (!lakeBlocked[seed] || visited[seed]) continue;
                std::vector<int> component;
                std::vector<int> spills;
                visited[seed] = 1u;
                frontier.push(static_cast<int>(seed));
                while (!frontier.empty()) {
                    const int index = frontier.front();
                    frontier.pop();
                    component.push_back(index);
                    if ((*lakeSpillPoints)[static_cast<std::size_t>(index)] >= 0.5f)
                        spills.push_back(index);
                    const int x = index % width;
                    const int y = index / width;
                    for (int directionIndex = 0; directionIndex < 8; ++directionIndex) {
                        const int nx = x + kDx[directionIndex];
                        const int ny = y + kDy[directionIndex];
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                        const int neighbor = ny * width + nx;
                        if (!lakeBlocked[static_cast<std::size_t>(neighbor)]) {
                            // The spill is normally the dry sill cell just
                            // outside the wet footprint, not necessarily a
                            // member of the component itself.
                            if ((*lakeSpillPoints)[static_cast<std::size_t>(neighbor)] >= 0.5f &&
                                std::find(spills.begin(), spills.end(), neighbor) == spills.end())
                                spills.push_back(neighbor);
                            continue;
                        }
                        if (visited[static_cast<std::size_t>(neighbor)]) continue;
                        visited[static_cast<std::size_t>(neighbor)] = 1u;
                        frontier.push(neighbor);
                    }
                }

                int outlet = -1;
                float outletArea = -1.0f;
                for (int spill : spills) {
                    const int direct = directParent[static_cast<std::size_t>(spill)];
                    if (direct >= 0 && !lakeBlocked[static_cast<std::size_t>(direct)] &&
                        result.active[static_cast<std::size_t>(direct)]) {
                        const float area = firstChannel(accumulation, accumulationChannels,
                                                        static_cast<std::size_t>(direct));
                        if (area > outletArea) { outlet = direct; outletArea = area; }
                    }
                    const int x = spill % width;
                    const int y = spill / width;
                    for (int directionIndex = 0; directionIndex < 8; ++directionIndex) {
                        const int nx = x + kDx[directionIndex];
                        const int ny = y + kDy[directionIndex];
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                        const int candidate = ny * width + nx;
                        if (lakeBlocked[static_cast<std::size_t>(candidate)] ||
                            !result.active[static_cast<std::size_t>(candidate)]) continue;
                        // An inlet points back into this component. Prefer the
                        // bank cell whose own receiver continues away from it.
                        const int candidateReceiver = directParent[static_cast<std::size_t>(candidate)];
                        if (candidateReceiver >= 0 &&
                            lakeBlocked[static_cast<std::size_t>(candidateReceiver)]) continue;
                        const float area = firstChannel(accumulation, accumulationChannels,
                                                        static_cast<std::size_t>(candidate));
                        if (area > outletArea) { outlet = candidate; outletArea = area; }
                    }
                }
                if (outlet >= 0) {
                    for (int index : component)
                        lakeOutlet[static_cast<std::size_t>(index)] = outlet;
                }
            }
        }

        result.parent = directParent;
        for (std::size_t index = 0; index < pixelCount; ++index) {
            if (!result.active[index]) continue;
            int downstream = directParent[index];
            if (downstream < 0 || !lakeBlocked[static_cast<std::size_t>(downstream)]) continue;
            const int firstLakeCell = downstream;
            int guard = 0;
            while (downstream >= 0 && lakeBlocked[static_cast<std::size_t>(downstream)] &&
                   guard++ < static_cast<int>(pixelCount)) {
                downstream = directParent[static_cast<std::size_t>(downstream)];
            }
            if (downstream < 0 || lakeBlocked[static_cast<std::size_t>(downstream)])
                downstream = lakeOutlet[static_cast<std::size_t>(firstLakeCell)];
            result.parent[index] = downstream >= 0 && downstream != static_cast<int>(index) &&
                result.active[static_cast<std::size_t>(downstream)] ? downstream : -1;
        }

        // Remove short source twigs without deleting their downstream junction.
        const int minimumBranchLength = (std::max)(params.minimumBranchLength, 2);
        for (int pruningPass = 0; pruningPass < 4; ++pruningPass) {
            std::vector<int> incoming(pixelCount, 0);
            for (std::size_t index = 0; index < pixelCount; ++index) {
                const int downstream = result.parent[index];
                if (result.active[index] && downstream >= 0 && result.active[static_cast<std::size_t>(downstream)])
                    ++incoming[static_cast<std::size_t>(downstream)];
            }

            bool removedAny = false;
            for (std::size_t source = 0; source < pixelCount; ++source) {
                if (!result.active[source] || incoming[source] != 0) continue;
                std::vector<int> branch;
                int cursor = static_cast<int>(source);
                while (cursor >= 0 && result.active[static_cast<std::size_t>(cursor)] &&
                       static_cast<int>(branch.size()) < minimumBranchLength) {
                    branch.push_back(cursor);
                    const int downstream = result.parent[static_cast<std::size_t>(cursor)];
                    if (downstream < 0 || !result.active[static_cast<std::size_t>(downstream)] ||
                        incoming[static_cast<std::size_t>(downstream)] > 1) break;
                    cursor = downstream;
                }
                const int endpoint = branch.empty() ? -1 : branch.back();
                const bool reachedJunction = endpoint >= 0 &&
                    result.parent[static_cast<std::size_t>(endpoint)] >= 0 &&
                    incoming[static_cast<std::size_t>(
                        result.parent[static_cast<std::size_t>(endpoint)])] > 1;
                if (static_cast<int>(branch.size()) < minimumBranchLength && reachedJunction) {
                    for (int index : branch) result.active[static_cast<std::size_t>(index)] = 0u;
                    removedAny = true;
                }
            }
            if (!removedAny) break;
        }

        std::vector<int> incoming(pixelCount, 0);
        for (std::size_t index = 0; index < pixelCount; ++index) {
            const int downstream = result.parent[index];
            if (result.active[index] && downstream >= 0 && result.active[static_cast<std::size_t>(downstream)])
                ++incoming[static_cast<std::size_t>(downstream)];
        }

        std::queue<int> ready;
        std::vector<int> remainingIncoming = incoming;
        result.streamOrder.assign(pixelCount, 0);
        std::vector<int> highestUpstreamOrder(pixelCount, 0);
        std::vector<int> highestOrderCount(pixelCount, 0);
        for (std::size_t index = 0; index < pixelCount; ++index) {
            if (result.active[index] && incoming[index] == 0) {
                result.streamOrder[index] = 1;
                ready.push(static_cast<int>(index));
            }
        }

        result.maximumStreamOrder = 1;
        while (!ready.empty()) {
            const int index = ready.front();
            ready.pop();
            result.maximumStreamOrder = (std::max)(
                result.maximumStreamOrder, result.streamOrder[static_cast<std::size_t>(index)]);
            const int downstream = result.parent[static_cast<std::size_t>(index)];
            if (downstream < 0 || !result.active[static_cast<std::size_t>(downstream)]) continue;
            const int currentOrder = result.streamOrder[static_cast<std::size_t>(index)];
            int& best = highestUpstreamOrder[static_cast<std::size_t>(downstream)];
            int& bestCount = highestOrderCount[static_cast<std::size_t>(downstream)];
            if (currentOrder > best) {
                best = currentOrder;
                bestCount = 1;
            } else if (currentOrder == best) {
                ++bestCount;
            }
            if (--remainingIncoming[static_cast<std::size_t>(downstream)] == 0) {
                result.streamOrder[static_cast<std::size_t>(downstream)] =
                    best + (bestCount >= 2 ? 1 : 0);
                ready.push(downstream);
            }
        }

        float maximumPhysicalArea = minimumArea;
        if (params.accumulationIsPhysicalArea) {
            for (std::size_t index = 0; index < pixelCount; ++index) {
                if (!result.active[index]) continue;
                maximumPhysicalArea = (std::max)(
                    maximumPhysicalArea, firstChannel(accumulation, accumulationChannels, index));
            }
        }
        const float physicalLogSpan = std::log((std::max)(maximumPhysicalArea / minimumArea, 1.0f));

        result.channelStrength.assign(pixelCount, 0.0f);
        result.normalizedStreamOrder.assign(pixelCount, 0.0f);
        result.sources.assign(pixelCount, 0.0f);
        for (std::size_t index = 0; index < pixelCount; ++index) {
            if (!result.active[index]) continue;
            const float accumulated = firstChannel(accumulation, accumulationChannels, index);
            if (params.accumulationIsPhysicalArea) {
                const float relativeArea = (std::max)(accumulated / minimumArea, 1.0f);
                const float hierarchy = physicalLogSpan > 1.0e-6f
                    ? std::clamp(std::log(relativeArea) / physicalLogSpan, 0.0f, 1.0f)
                    : 0.0f;
                // Threshold-sized tributaries stay visible while large trunks
                // retain enough dynamic range for width/depth authoring.
                result.channelStrength[index] = 0.05f + hierarchy * 0.95f;
            } else {
                result.channelStrength[index] = std::clamp(accumulated, 0.0f, 1.0f);
            }
            result.normalizedStreamOrder[index] =
                static_cast<float>(result.streamOrder[index]) /
                static_cast<float>((std::max)(result.maximumStreamOrder, 1));
            result.sources[index] = incoming[index] == 0 ? 1.0f : 0.0f;
        }

        // The raster mask is also a preview/debug surface. Give every inlet a
        // one-cell shoreline overlap so it visibly reaches the lake boundary;
        // downstream river/carve nodes still receive Lake Mask and explicitly
        // exclude this sample from bed cutting and water-ribbon generation.
        for (std::size_t index = 0; index < pixelCount; ++index) {
            if (!result.active[index]) continue;
            const int downstream = directParent[index];
            if (downstream < 0 || !lakeBlocked[static_cast<std::size_t>(downstream)]) continue;
            result.channelStrength[static_cast<std::size_t>(downstream)] = (std::max)(
                result.channelStrength[static_cast<std::size_t>(downstream)],
                result.channelStrength[index]);
            result.normalizedStreamOrder[static_cast<std::size_t>(downstream)] = (std::max)(
                result.normalizedStreamOrder[static_cast<std::size_t>(downstream)],
                result.normalizedStreamOrder[index]);
        }
        return true;
    }

} // namespace TerrainHydrology
