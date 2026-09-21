#ifndef SURFACE_COVERAGE_H
#define SURFACE_COVERAGE_H

// Bit is shared by Vulkan/OptiX material flags. No material-buffer ABI change.
#define MATERIAL_FLAG_ALPHA_CUTOUT (1u << 25u)
#define MATERIAL_FLAG_VIEWPORT_ALPHA_CUTOUT (1u << 26u)
#define MATERIAL_FLAGS_PREVIEW_CUTOUT (MATERIAL_FLAG_ALPHA_CUTOUT | MATERIAL_FLAG_VIEWPORT_ALPHA_CUTOUT)
#ifdef __cplusplus
namespace SurfaceCoverage {
#if defined(__CUDACC__)
__host__ __device__
#endif
inline
#endif
float materialCoverageOpacity(float opacity, bool cutout) {
    // Explicit binary coverage. Legacy fractional opacity remains unchanged.
    return cutout ? (opacity >= float(0.5) ? float(1.0) : float(0.0)) : opacity;
}
#ifdef __cplusplus
} // namespace SurfaceCoverage
#endif
#endif
