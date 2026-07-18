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
        float defaultWidthScreen, size_t inputCount, size_t outputCount, float titleWidthScreen,
        float contentHeightScreen = 0.0f) {
        NodeChromeLayout layout;
        layout.headerHeight = scaleNodeChromeMetric(zoom, 20.0f, 16.0f, 26.0f);
        // A node that draws real ImGui widgets on its pin rows (NodeBase::pinRowHeight)
        // needs a row taller than a bare label's 18px, or the frames overlap their neighbours.
        const float rowBase = node.pinRowHeight();
        layout.pinSpacing = (rowBase > 0.0f)
            ? scaleNodeChromeMetric(zoom, rowBase, rowBase * 0.7f, rowBase * 1.4f)
            : scaleNodeChromeMetric(zoom, 18.0f, 14.0f, 24.0f);
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

        // The max clamp exists to stop AUTO-sized nodes running away with a long title. An
        // explicit getCustomWidth() or a user resize is a stated intent, not a suggestion —
        // clamping those to 220px silently capped every wide node (and made the resize handle
        // look broken past that point).
        const float explicitWidth = std::max(node.uiWidth, customWidth);
        const float maxAllowed = (explicitWidth > 0.0f)
            ? std::max(maxWidthScreen, explicitWidth * zoom)
            : maxWidthScreen;
        layout.width = std::clamp(baseWidth * zoom, minWidthScreen, maxAllowed);

        const int maxPins = static_cast<int>(std::max(inputCount, outputCount));
        const float pinsHeight = maxPins > 0 ? (maxPins * layout.pinSpacing + layout.bodyPadding) : 0.0f;
        // Height must grow with the pin count. The old hard cap
        // (min(..., ~220px)) clipped any node with ~11+ pins — e.g. the Material
        // Output node — drawing its lower sockets past the body border. The cap
        // only ever mattered for pin-heavy nodes (nothing else contributes to
        // height here), so it protected nothing worth keeping.
        // contentHeightScreen: inline node-body widgets (see NodeBase::
        // wantsInlineContent) — already in screen pixels, NOT zoom-scaled,
        // because ImGui widgets don't scale with canvas zoom.
        layout.height = layout.collapsed
            ? layout.headerHeight
            : layout.headerHeight + pinsHeight + layout.bodyPadding + contentHeightScreen;

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
