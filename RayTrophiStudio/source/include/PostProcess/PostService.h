#pragma once
#include "json.hpp"
#include "ColorProcessingParams.h"
class Camera;
struct UIContext;
namespace rtpost {
nlohmann::json inspect(UIContext&);
nlohmann::json configure(UIContext&, const nlohmann::json&);
nlohmann::json reset(UIContext&);
nlohmann::json saveExposure(const ExposureSettings&);
void loadExposure(const nlohmann::json&, ColorProcessor&);
void syncDisplay(ColorProcessor&, const Camera*, bool freeze);
void tick(UIContext&);
void drawPanel(UIContext&);
bool parseModernTone(const std::string&, ToneMappingType&);
const char* modernToneName(ToneMappingType);
}
