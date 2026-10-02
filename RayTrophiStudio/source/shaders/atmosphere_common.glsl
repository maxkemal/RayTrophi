// Atmosphere medium model shared by atmosphere_lut.comp and
// atmosphere_aerial_froxel.comp.
//
// ★★ ONE definition of "how dense is the air at altitude h and how does it
//   scatter". The sky-view LUT and the aerial froxel must agree on it, or the
//   horizon shows a seam where the sky meets distant terrain -- and neither
//   backend reports anything. The CPU side has the same rule for the parameter
//   block (Backend/AtmosphereLutParams.h).
//
// The including shader defines ATMOSPHERE_PARAMS_BINDING before including
// (defaults to 3, the LUT shader's binding).
//
// Parameter block layout (see AtmosphereLutParams.h):
//   sunDir_intensity  = sun direction xyz, sun intensity
//   density_intensity = air density, dust density, ozone density, atmosphere intensity
//   physical          = planet radius, atmosphere height, altitude offset, Mie g
//   weather           = hygroscopic Mie scale, surface temperature (C), ozone scale, -
//   rayleigh          = Rayleigh scattering rgb, Rayleigh scale height
//   mie               = Mie scattering rgb, Mie scale height

#ifndef ATMOSPHERE_COMMON_GLSL
#define ATMOSPHERE_COMMON_GLSL

#ifndef ATMOSPHERE_PARAMS_BINDING
#define ATMOSPHERE_PARAMS_BINDING 3
#endif

layout(std430, binding = ATMOSPHERE_PARAMS_BINDING) readonly buffer AtmosphereParams {
    vec4 sunDir_intensity;
    vec4 density_intensity;
    vec4 physical;
    vec4 weather;
    vec4 rayleigh;
    vec4 mie;
} params;

const float PI = 3.14159265359;
const float INV_PI = 0.31830988618;

float saturate(float x) { return clamp(x, 0.0, 1.0); }
vec3 saturate(vec3 x) { return clamp(x, vec3(0.0), vec3(1.0)); }

float phaseRayleigh(float cosTheta) {
    return 3.0 / (16.0 * PI) * (1.0 + cosTheta * cosTheta);
}

float phaseHenyeyGreenstein(float g, float cosTheta) {
    g = clamp(g, -0.95, 0.95);
    float g2 = g * g;
    float d = max(1.0 + g2 - 2.0 * g * cosTheta, 0.001);
    return (1.0 - g2) / (4.0 * PI * pow(d, 1.5));
}

float phaseMie(float cosTheta) {
    return phaseHenyeyGreenstein(params.physical.w, cosTheta);
}

float raySphereExitDistance(vec3 pos, vec3 dir, float radius) {
    float b = 2.0 * dot(pos, dir);
    float c = dot(pos, pos) - radius * radius;
    float delta = b * b - 4.0 * c;
    if (delta < 0.0) return -1.0;
    return (-b + sqrt(delta)) * 0.5;
}

float atmoPlanetRadius()     { return max(params.physical.x, 1000.0); }
float atmoHeight()           { return max(params.physical.y, 1000.0); }
float atmoTopRadius()        { return atmoPlanetRadius() + atmoHeight(); }
float atmoAltitudeOffset()   { return max(params.physical.z, 0.0); }
float atmoIntensity()        { return max(params.density_intensity.w, 0.0); }

vec3 atmoSunDirection() {
    vec3 sunDir = params.sunDir_intensity.xyz;
    float sunLen = length(sunDir);
    return (sunLen > 0.001) ? sunDir / sunLen : vec3(0.0, 1.0, 0.0);
}

float temperatureScale() {
    return max((params.weather.y + 273.15) / 288.15, 0.1);
}

// Hygroscopic aerosol growth, computed on the CPU from the climate's relative
// humidity (atmosphere::hygroscopicMieScale, 1.0 = dry aerosol). weather.x used
// to carry the raw humidity and was never read -- the slider was dead.
float mieHumidityScale() {
    return max(params.weather.x, 0.0);
}

float rayleighScaleHeight() { return max(params.rayleigh.w * temperatureScale(), 1.0); }
float mieScaleHeight()      { return max(params.mie.w * temperatureScale(), 1.0); }

vec3 ozoneExtinction() {
    return vec3(0.000000650, 0.000001881, 0.000000085)
         * max(params.density_intensity.z, 0.0)
         * max(params.weather.z, 0.0);
}

// Scattering coefficients (1/m) at altitude h above the planet surface.
vec3 atmoRayleighScattering(float h) {
    return max(params.rayleigh.xyz, vec3(0.0)) * max(params.density_intensity.x, 0.0)
         * exp(-h / rayleighScaleHeight());
}

vec3 atmoMieScattering(float h) {
    return max(params.mie.xyz, vec3(0.0)) * max(params.density_intensity.y, 0.0)
         * mieHumidityScale() * exp(-h / mieScaleHeight());
}

// Transmittance LUT parameterisation, shared by every reader:
//   u = (cos(zenith) + 0.2) / 1.2,  v = altitude / atmosphere height.
vec2 transmittanceLutUV(float cosZenith, float altitude) {
    return vec2((clamp(cosZenith, -0.2, 1.0) + 0.2) / 1.2,
                clamp(altitude / atmoHeight(), 0.0, 1.0));
}

// Sky-view LUT parameterisation (miss.rmiss / material_preview_sky.frag):
//   u = azimuth / 2pi (atan2(z, x), wrapped),  v = (1 - dir.y) * 0.5.
vec2 skyViewLutUV(vec3 dir) {
    float u = atan(dir.z, dir.x) / (2.0 * PI);
    if (u < 0.0) u += 1.0;
    return vec2(u, (1.0 - clamp(dir.y, -1.0, 1.0)) * 0.5);
}

#endif // ATMOSPHERE_COMMON_GLSL
