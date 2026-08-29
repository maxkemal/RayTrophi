/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          scene_ui_node_palette.hpp
* Author:        Kemal Demirtaş
* Date:          2026
* License:       Proprietary / RayTrophi Studio
* =========================================================================
*/
#pragma once

#include "imgui.h"
#include "TerrainNodesV2.h"
#include "ui_modern.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <functional>

namespace TerrainNodesV2 {

struct ModernNodeDescriptor {
    NodeType type;
    const char* label;
    const char* description;
    UIWidgets::IconType iconType;
};

struct ModernNodeCategory {
    std::string name;
    ImVec4 defaultColor;
    std::vector<ModernNodeDescriptor> nodes;
    bool expanded = true;
};

class ModernTerrainNodePalette {
public:
    static ModernTerrainNodePalette& instance() {
        static ModernTerrainNodePalette inst;
        return inst;
    }

    ModernTerrainNodePalette();

    /**
     * @brief Render the modern node library palette.
     */
    void render(TerrainNodeGraphV2& graph, TerrainObject* terrain, char* searchBuffer, size_t searchBufferSize);

    // Callback fired when a node is clicked in the palette
    std::function<void(NodeType)> onNodeClicked;

    /**
     * @brief Get list of registered category names.
     */
    std::vector<std::string> getCategoryNames() const;

    /**
     * @brief Get node type names within a category.
     */
    std::vector<std::string> getNodeNamesInCategory(const std::string& categoryName) const;

    /**
     * @brief Access category list for context menus or custom views.
     */
    const std::vector<ModernNodeCategory>& getCategories() const { return categories_; }

private:
    void initializeCategories();
    void renderNodeCard(const ModernNodeCategory& category, const ModernNodeDescriptor& nodeDesc, float itemWidth);

    std::vector<ModernNodeCategory> categories_;
    int selectedCategoryIndex_ = 0; // State for dual-pane sidebar layout
};

} // namespace TerrainNodesV2
