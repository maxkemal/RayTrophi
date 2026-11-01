#pragma once

#include <memory>

class VolumeShader;
bool drawVolumeAdaptiveStepping(VolumeShader& shader);

class VolumeShader;
class VDBVolume;
struct UIContext;
class SceneUI;
namespace RayTrophiSim { struct SimulationGridDomainDesc; }

// One function per panel section: the sections own the layout, these own the
// widgets, and every edit goes through volume_material::configure.
bool drawVolumePresetsUI(VolumeShader& shader);
bool drawSmokeAppearanceUI(VolumeShader& shader);
bool drawVolumeDensityRampUI(VolumeShader& shader);
bool drawFireLightingUI(VolumeShader& shader);
bool drawFireResponseUI(VolumeShader& shader);
bool volumeShaderHasGasDomain(UIContext& context, const std::shared_ptr<VolumeShader>& shader);
// Measure and fit buttons, shown only for a material owned by a gas domain.
bool drawGasTemperatureFitUI(UIContext& context, const std::shared_ptr<VolumeShader>& shader);
bool drawMatterFogVisibility(UIContext& context, VDBVolume* volume);
void drawMatterFogLinks(UIContext& context, SceneUI& ui,
    const RayTrophiSim::SimulationGridDomainDesc& domain);
