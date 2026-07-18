#pragma once

#include "AssetRegistry.h"
#include "InstanceGroup.h"

#include <filesystem>
#include <string>
#include <vector>

namespace FoliageAssets {

// Shared, metadata-only catalogue used by foliage nodes. Geometry is loaded
// lazily by loadScatterSource(), never appended to the scene hierarchy.
const AssetRegistry& catalog(bool forceRefresh = false);
const std::filesystem::path& libraryRoot();

std::vector<const AssetRecord*> recommendedAssets(
    const std::string& layerType,
    const std::string& biome,
    const std::string& searchText);

bool loadScatterSource(const std::string& relativeEntryPath,
                       const std::string& displayName,
                       float weight,
                       ScatterSource& outSource,
                       std::string* errorMessage = nullptr);

float defaultTargetHeight(const std::string& layerType);
void configurePlacement(ScatterSource& source,
                        float targetHeight,
                        float heightVariation,
                        bool alignToNormal,
                        float normalInfluence);

} // namespace FoliageAssets
