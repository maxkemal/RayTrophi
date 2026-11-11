#pragma once

namespace RayTrophiSim {
struct SubstanceProfile;
namespace Fluid {
// Dense mud uses existing Hencky elastoplastic MPM, viscous stress and tearing.
// Slurry remains a viscous liquid. Values are starting presets, not measured fits.
SubstanceProfile makeDenseMud();
SubstanceProfile makeMudSlurry();
}
}
