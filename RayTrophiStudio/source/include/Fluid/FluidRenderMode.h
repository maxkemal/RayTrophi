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
    // Gas domains, and the legacy/invalid liquid value: a liquid holding it is
    // normalised to SurfaceSDF where the mode is consumed. Old .rtp files
    // saved it for liquids, so it can never mean "fog" -- see VolumeFog.
    Volume     = 0,
    Particles  = 1,  // Each particle mirrored as an instanced sphere (debug).
    SurfaceSDF = 2,  // Narrow-band level set + density-proxy band as a surface.
    // Value 3 is retired: it was "VirtualParticles", which drew liquids
    // exactly like Particles and only reclassified granular domains. Stored
    // scenes still carry it -- read them through fluidRenderModeFromStored.
    // Liquid drawn as a participating medium: the density the solver splats
    // every step, raymarched with the domain's VolumeShader (fog/gas look).
    // A new value on purpose: 0 already means "render as surface" in saved
    // liquid scenes, and reusing it would turn those into fog on load.
    VolumeFog  = 4,
};

// Single decode point for a persisted render mode. The retired value 3 had
// particle semantics, so it maps to Particles; anything unknown falls back to
// SurfaceSDF, the same answer the consumer gives the invalid liquid 'Volume'.
inline FluidRenderMode fluidRenderModeFromStored(int stored) {
    switch (stored) {
        case 0: return FluidRenderMode::Volume;
        case 1: return FluidRenderMode::Particles;
        case 2: return FluidRenderMode::SurfaceSDF;
        case 3: return FluidRenderMode::Particles;
        case 4: return FluidRenderMode::VolumeFog;
        default: return FluidRenderMode::SurfaceSDF;
    }
}

} // namespace Fluid
} // namespace RayTrophiSim
