/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          FluidRenderMode.h
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 *
 * Shared enum so both the legacy FluidObject and the active
 * SimulationGridDomain (type == Fluid) reference the same render-mode set
 * without a circular include. See ParticleRenderBridge / scene_data render
 * bridge for how each value is consumed.
 */

#pragma once

namespace RayTrophiSim {
namespace Fluid {

enum class FluidRenderMode : int {
    Volume     = 0,  // APIC density splatted to NanoVDB (fog look — default).
    Particles  = 1,  // Each particle mirrored as an instanced sphere (debug).
    SurfaceSDF = 2,  // Narrow-band level set + density-proxy band as a surface.
};

} // namespace Fluid
} // namespace RayTrophiSim
