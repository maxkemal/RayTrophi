#pragma once

#include <cstddef>

namespace FluidSim { class FluidGrid; }

namespace RayTrophiSim::Fluid {
class FluidParticles;

// Relocates particles that are already inside a collider solid cell to the
// nearest non-solid cell. Mass/chemistry sidecars remain untouched.
//
// keep_mpm_on_phi: MPM parcels (Granular, Elastic, and Auto when auto_is_mpm)
// in a cell whose collider has a distance (grid.solid_phi) are NOT moved. The
// mask marks a cell solid by its centre, so a curved surface leaves parcels
// that rest on it inside a "solid" cell; teleporting them to the next cell
// centre and cutting their speed to a quarter every frame crushed a jelly on
// a dome to a tenth of its volume (2026-11-03 live) and is the snow-sticks-to-
// colliders report. Those parcels take applyMpmParticleContact instead.
std::size_t recoverParticlesFromSolidCells(FluidParticles& particles,
                                           const FluidSim::FluidGrid& grid,
                                           int maximum_search_radius = 6,
                                           bool keep_mpm_on_phi = false,
                                           bool auto_is_mpm = false);
}
