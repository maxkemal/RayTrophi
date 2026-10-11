#pragma once

class TimelineWidget;
struct SceneData;
struct RenderSettings;

// Loader threads publish a debt; only the main/UI thread mutates the widget.
void requestProjectTimelineReset();
bool applyPendingProjectTimelineReset(TimelineWidget& widget, SceneData& scene,
                                     RenderSettings& settings);
