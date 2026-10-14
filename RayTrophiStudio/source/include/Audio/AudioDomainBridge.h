#pragma once
#include "json.hpp"
struct UIContext;
namespace audio {
nlohmann::json domainAudioInputs(UIContext& ctx);
void syncDomainAudioTelemetry(UIContext& ctx);
void tickDomainAudioInputs(UIContext& ctx);
}
