#pragma once

#include "Backend/VulkanVolumeInstrumentation.h"

struct UIContext;

void DrawVolumePerformancePanel(UIContext& ctx);
bool GetCachedVolumePerformanceStats(VulkanRT::VolumePerformanceStats& stats);
