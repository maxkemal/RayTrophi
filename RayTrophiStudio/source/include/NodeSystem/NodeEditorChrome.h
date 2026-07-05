#pragma once

#include "Node.h"
#include "imgui.h"
#include <algorithm>

namespace NodeSystem {

    inline float scaleNodeChromeMetric(float zoom, float base, float minValue, float maxValue) {
        return std::clamp(base * zoom, minValue, maxValue);
    }

    struct NodeChromeLayout {
        float headerHeight = 0.0f;
        float pinSpacing = 0.0f;
        float pinRadius = 0.0f;
        float cornerRadius = 0.0f;
        float bodyPadding = 0.0f;
        float shadowOffset = 0.0f;
        float collapsedPinSpacing = 0.0f;
        float toggleSize = 0.0f;
        float resizeHandleWidth = 0.0f;
        float width = 0.0f;
        float height = 0.0f;
        float labelWidth = 0.0f;
        bool showTitle = false;
        bool showPinLabels = false;
        bool collapsed = false;
    };

    inline NodeChromeLayout buildNodeChromeLayout(const NodeBase& node, float zoom,
        float defaultWidthScreen, size_t inputCount, size_t outputCount, float titleWidthScreen) {
        NodeChromeLayout layout;
        layout.headerHeight = scaleNodeChromeMetric(zoom, 20.0f, 16.0f, 26.0f);
        layout.pinSpacing = scaleNodeChromeMetric(zoom, 18.0f, 14.0f, 24.0f);
        layout.pinRadius = scaleNodeChromeMetric(zoom, 5.0f, 4.0f, 7.5f);
        layout.cornerRadius = scaleNodeChromeMetric(zoom, 4.0f, 3.0f, 8.0f);
        layout.bodyPadding = scaleNodeChromeMetric(zoom, 8.0f, 6.0f, 11.0f);
        layout.shadowOffset = scaleNodeChromeMetric(zoom, 4.0f, 2.5f, 6.5f);
        layout.collapsedPinSpacing = scaleNodeChromeMetric(zoom, 6.0f, 4.0f, 8.0f);
        layout.toggleSize = scaleNodeChromeMetric(zoom, 15.0f, 13.0f, 18.0f);
        layout.resizeHandleWidth = scaleNodeChromeMetric(zoom, 6.0f, 4.5f, 9.5f);
        layout.showTitle = zoom >= 0.32f;
        layout.showPinLabels = zoom >= 0.72f;
        layout.collapsed = node.collapsed;

        const float minWidthScreen = scaleNodeChromeMetric(zoom, 140.0f, 96.0f, 200.0f);
        const float maxWidthScreen = scaleNodeChromeMetric(zoom, 220.0f, 150.0f, 280.0f);
        const float titleWidthClamped = std::min(titleWidthScreen, scaleNodeChromeMetric(zoom, 180.0f, 120.0f, 260.0f));

        const float customWidth = node.getCustomWidth();
        const float baseWidth = node.uiWidth > 0.0f
            ? node.uiWidth
            : ((customWidth > 0.0f) ? customWidth : std::max(minWidthScreen / zoom, titleWidthClamped / zoom));

        layout.width = std::clamp(baseWidth * zoom, minWidthScreen, maxWidthScreen);

        const int maxPins = static_cast<int>(std::max(inputCount, outputCount));
        const float pinsHeight = maxPins > 0 ? (maxPins * layout.pinSpacing + layout.bodyPadding) : 0.0f;
        layout.height = layout.collapsed
            ? layout.headerHeight
            : std::min(layout.headerHeight + pinsHeight + layout.bodyPadding,
                scaleNodeChromeMetric(zoom, 220.0f, 90.0f, 280.0f));

        layout.labelWidth = std::max(0.0f, layout.width - layout.bodyPadding * 2.0f - layout.pinRadius * 2.0f - 8.0f);
        return layout;
    }

    inline float getNodePinStartY(const NodeChromeLayout& layout, float topY, size_t pinCount) {
        if (layout.collapsed) {
            return topY + layout.headerHeight * 0.5f -
                std::max(0, static_cast<int>(pinCount) - 1) * layout.collapsedPinSpacing * 0.5f;
        }
        return topY + layout.headerHeight + layout.bodyPadding + layout.pinSpacing * 0.5f;
    }
}
