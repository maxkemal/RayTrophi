#pragma once

struct UIContext;
class VDBVolume;
namespace atmosphere { struct CloudState; struct CloudLayer; }

namespace cloud_ui {
bool isWorldCloud(const VDBVolume* volume);
bool drawMaterialControls(atmosphere::CloudState& clouds);
bool drawWeatherLightningControls(UIContext& ctx);
void beginLegacyShaftControls(const UIContext& ctx, const atmosphere::CloudState& clouds);
bool drawLayerControls(atmosphere::CloudLayer& layer);
void drawWorldCloudProperties(UIContext& ctx);
}
