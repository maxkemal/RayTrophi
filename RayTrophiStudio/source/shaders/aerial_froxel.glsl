// Aerial perspective froxel -- the READ side, shared by raygen.rgen (Vulkan RT)
// and raster_post.comp (RayFusion). atmosphere_aerial_froxel.comp is the one
// writer; both devices run that same shader on the same parameter block, so
// "RT and RayFusion agree" is a property of this file, not of two formulas
// that happen to look alike.
//
// Atlas: 1024 x 128, RGBA16F.
//   x = slice * 32 + cell.x     (32 view-depth slices of 32x32 cells)
//   y = block * 32 + cell.y     (cell.y = 0 is the BOTTOM of the image)
//   block 0: in-scatter, air + height fog      (radiance, sky LUT units)
//   block 1: transmittance, air + height fog   (rgb)
//   block 2: in-scatter, height fog only
//   block 3: transmittance, height fog only
// The fog-only blocks exist for sky pixels: the sky-view LUT already holds the
// air to infinity, so adding the air again there double-hazes the sky; the
// fog layer is NOT in the sky LUT and must still cover the horizon.
//
// Slice k holds the integral from the camera to VIEW-SPACE depth
// afSliceEndDepth(k) (depth along the camera forward axis, not ray length --
// the raster depth buffer can only give view depth). Slices are quadratic so
// the first one ends at 62.5 m and the last at 64 km. Beyond 64 km the last
// slice is used (distant terrain is under-hazed there, not over-hazed).

#ifndef AERIAL_FROXEL_GLSL
#define AERIAL_FROXEL_GLSL

const int   AF_CELLS     = 32;
const int   AF_SLICES    = 32;
const float AF_MAX_DEPTH = 64000.0;
const vec2  AF_ATLAS_SIZE = vec2(1024.0, 128.0);

const int AF_BLOCK_INSCATTER     = 0;
const int AF_BLOCK_TRANSMITTANCE = 1;
const int AF_BLOCK_FOG_INSCATTER = 2;
const int AF_BLOCK_FOG_TRANS     = 3;

float afSliceEndDepth(int slice) {
    float s = float(slice + 1) / float(AF_SLICES);
    return AF_MAX_DEPTH * s * s;
}

// screenUV: u from the image's left edge, v from its BOTTOM edge, both 0..1.
vec2 afAtlasUV(vec2 screenUV, int slice, int block) {
    // Clamp inside the tile so bilinear filtering never bleeds into the
    // neighbouring slice or block.
    vec2 c = clamp(screenUV * float(AF_CELLS), vec2(0.5), vec2(float(AF_CELLS) - 0.5));
    return (vec2(float(slice * AF_CELLS), float(block * AF_CELLS)) + c) / AF_ATLAS_SIZE;
}

struct AerialSample {
    vec3 inscatter;
    vec3 transmittance;
};

AerialSample afSample(sampler2D atlas, vec2 screenUV, float viewDepth, bool fogOnly) {
    int blockS = fogOnly ? AF_BLOCK_FOG_INSCATTER : AF_BLOCK_INSCATTER;
    int blockT = fogOnly ? AF_BLOCK_FOG_TRANS : AF_BLOCK_TRANSMITTANCE;

    // Continuous slice-END coordinate: f == k means "exactly at the end of
    // slice k"; f in [-1, 0) lerps from the camera (nothing) to slice 0.
    float f = sqrt(clamp(viewDepth / AF_MAX_DEPTH, 0.0, 1.0)) * float(AF_SLICES) - 1.0;
    int   k0 = int(floor(f));
    float w  = f - float(k0);
    int   k1 = min(k0 + 1, AF_SLICES - 1);

    vec3 s0 = vec3(0.0), t0 = vec3(1.0);
    if (k0 >= 0) {
        s0 = textureLod(atlas, afAtlasUV(screenUV, k0, blockS), 0.0).rgb;
        t0 = textureLod(atlas, afAtlasUV(screenUV, k0, blockT), 0.0).rgb;
    }
    vec3 s1 = textureLod(atlas, afAtlasUV(screenUV, k1, blockS), 0.0).rgb;
    vec3 t1 = textureLod(atlas, afAtlasUV(screenUV, k1, blockT), 0.0).rgb;

    AerialSample r;
    r.inscatter     = max(mix(s0, s1, w), vec3(0.0));
    r.transmittance = clamp(mix(t0, t1, w), vec3(0.0), vec3(1.0));
    return r;
}

vec3 afApply(vec3 radiance, AerialSample s) {
    return radiance * s.transmittance + s.inscatter;
}

// Project a world-space direction onto the image plane the froxel was built
// from (origin, lowerLeft, horizontal, vertical -- the RT camera push
// constants; the plane's distance does not matter, u/v are ratios).
// Returns false for directions at or behind the plane's horizon.
bool afProjectDirection(vec3 origin, vec3 lowerLeft, vec3 horizontal, vec3 vertical,
                        vec3 dir, out vec2 screenUV, out float cosToForward) {
    vec3 toLL = lowerLeft - origin;
    vec3 fwd  = toLL + 0.5 * horizontal + 0.5 * vertical;
    vec3 n    = normalize(cross(horizontal, vertical));
    if (dot(n, fwd) < 0.0) n = -n;
    cosToForward = dot(dir, n);
    if (cosToForward <= 1e-4) { screenUV = vec2(0.5); return false; }
    vec3 q = dir * (dot(fwd, n) / cosToForward) - toLL;
    screenUV = vec2(dot(q, horizontal) / max(dot(horizontal, horizontal), 1e-12),
                    dot(q, vertical)   / max(dot(vertical, vertical), 1e-12));
    return true;
}

#endif // AERIAL_FROXEL_GLSL
