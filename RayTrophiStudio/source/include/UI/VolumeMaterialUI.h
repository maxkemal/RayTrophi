#pragma once

class VolumeShader;
class VDBVolume;
struct UIContext;
class SceneUI;
namespace RayTrophiSim { struct SimulationGridDomainDesc; }

bool drawPhysicalVolumeMaterialUI(VolumeShader& shader);
bool drawMatterFogVisibility(UIContext& context, VDBVolume* volume);
void drawMatterFogLinks(UIContext& context, SceneUI& ui,
    const RayTrophiSim::SimulationGridDomainDesc& domain);
