#pragma once
// Parameter block of shaders/atmosphere_lut.comp (binding 3, std430).
//
// ★★ ONE definition. This used to be copy-pasted into VulkanBackend.cpp and
//   VulkanDevicePipelines.cpp; both packed the dead `humidity` field into
//   weather.x, which the shader never read. A second copy is how one side
//   silently keeps an old meaning -- include this header instead.
//
// Layout must match the shader's AtmosphereParams block:
//   weather.x = hygroscopic Mie extinction scale (derived from climate RH)
//   weather.y = surface temperature, Celsius (scales both scale heights)
//   weather.z = ozone absorption scale

#include "World.h"
#include "Atmosphere/AtmosphereClimate.h"

struct AtmosphereLUTParamsGPU {
    float sunDir_intensity[4];
    float density_intensity[4];
    float physical[4];
    float weather[4];
    float rayleigh[4];
    float mie[4];
};

inline AtmosphereLUTParamsGPU makeAtmosphereLUTParamsGPU(const WorldData& world) {
    const NishitaSkyParams& n = world.nishita;
    AtmosphereLUTParamsGPU p{};
    p.sunDir_intensity[0] = n.sun_direction.x;
    p.sunDir_intensity[1] = n.sun_direction.y;
    p.sunDir_intensity[2] = n.sun_direction.z;
    p.sunDir_intensity[3] = n.sun_intensity;
    p.density_intensity[0] = n.air_density;
    p.density_intensity[1] = n.dust_density;
    p.density_intensity[2] = n.ozone_density;
    p.density_intensity[3] = n.atmosphere_intensity;
    p.physical[0] = n.planet_radius;
    p.physical[1] = n.atmosphere_height;
    p.physical[2] = n.altitude;
    p.physical[3] = n.mie_anisotropy;
    p.weather[0] = atmosphere::hygroscopicMieScale(n.derived_relative_humidity);
    p.weather[1] = n.derived_temperature_c;
    p.weather[2] = n.ozone_absorption_scale;
    p.weather[3] = 0.0f;
    p.rayleigh[0] = n.rayleigh_scattering.x;
    p.rayleigh[1] = n.rayleigh_scattering.y;
    p.rayleigh[2] = n.rayleigh_scattering.z;
    p.rayleigh[3] = n.rayleigh_density;
    p.mie[0] = n.mie_scattering.x;
    p.mie[1] = n.mie_scattering.y;
    p.mie[2] = n.mie_scattering.z;
    p.mie[3] = n.mie_density;
    return p;
}
