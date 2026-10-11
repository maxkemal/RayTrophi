#pragma once

namespace Backend {
class IBackend;
void syncWeatherClock(IBackend* render, IBackend* viewport, float timelineTime,
                      float previewTime, bool realtimePreview);
}
