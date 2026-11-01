#pragma once

struct UIContext;

namespace SimulationTransport {
void drain(UIContext& context);
void setTimelineMode(UIContext& context, bool timeline_mode);
}
