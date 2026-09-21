#pragma once
#include "RayFusion/ScreenGi.h"
namespace rtapi {
RayFusion::ScreenGiStatus screenGiStatus();
bool setScreenGi(const RayFusion::ScreenGiSettings& settings, std::string& error);
}
