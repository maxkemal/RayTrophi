// Lightning event + flash-light ABI and the light it throws OUT of the cloud.
// ONE definition for every consumer: the cloud field (cloud_common.glsl), the
// RT surface NEE (closesthit.rchit), the precipitation curtain (cloud_rt.glsl)
// and the RayFusion surface (material_preview_frag.frag), which has no cloud
// field bound and therefore reads only the CPU-built flash lights below.
// CPU mirror: Backend/CloudLightningParams.h (CloudLightningGPU,
// CloudFlashLightGPU, cloudFlashProfile).
#ifndef CLOUD_LIGHTNING_LIGHT_GLSL
#define CLOUD_LIGHTNING_LIGHT_GLSL

struct CloudLightningG {
    vec4 start;   // xyz, pulse radiance
    vec4 end;     // xyz, bolt radius m
    vec4 color;   // rgb, channel segment count
    vec4 stage;   // growth, brightness, kind (0 ground / 1 cloud / 2 upward / 3 vertical), glow
};

// One straight piece of an event's channel network (CPU: buildCloudBolt).
struct CloudBoltSegG {
    vec4 a;   // xyz, weight (1 = main channel, < 1 branch / twig)
    vec4 b;   // xyz, leader progress at which it starts to grow
    vec4 growth; // completion, parent index (-1 origin), depth, tip weight
};
const int CLOUD_BOLT_SEGMENTS = 80;   // per event (kCloudBoltSegments)

// The cloud base lit from inside by one event: a downward-facing disc.
struct CloudFlashLightG {
    vec4 position;   // xyz centre on the cloud base, w = disc radius m (0 = off)
    vec4 radiance;   // rgb radiance leaving the base, w = event index
};

// Radial falloff of the in-cloud flash around the channel. Diffusion-like
// (r^2 / (r^2 + d^2)) instead of a Gaussian: a Gaussian has a visible rim,
// and that rim on the cloud base read as a hovering disc. Windowed at 4 r so
// the cost stays local. The CPU applies the same formula for the base glow.
float cloudFlashProfile(float d, float r) {
    float r2 = r * r;
    return r2 / (r2 + d * d) * (1.0 - smoothstep(2.5 * r, 4.0 * r, d));
}

// Irradiance at P (normal N) from the lit base disc, unshadowed. Disc form
// factor with the receiver on the axis, cosines evaluated toward the disc
// centre: exact on the axis, smooth off it. `wi` = direction to the centre.
// N = vec3(0) gives the fluence-like value a scattering medium sees.
vec3 cloudFlashIrradiance(CloudFlashLightG f, vec3 P, vec3 N, out vec3 wi) {
    wi = vec3(0.0, 1.0, 0.0);
    float r = f.position.w;
    if (r <= 0.0) return vec3(0.0);
    vec3 toC = f.position.xyz - P;
    float d2 = dot(toC, toC);
    wi = toC * inversesqrt(max(d2, 1e-6));
    // The disc faces down: a receiver above the base sees its back.
    float cosL = wi.y;
    if (cosL <= 0.0) return vec3(0.0);
    float cosR = dot(N, N) > 0.0 ? max(dot(N, wi), 0.0) : 1.0;
    if (cosR <= 0.0) return vec3(0.0);
    float r2 = r * r;
    return f.radiance.rgb * (3.14159265358979 * r2 / (r2 + d2)) * cosL * cosR;
}

#endif
