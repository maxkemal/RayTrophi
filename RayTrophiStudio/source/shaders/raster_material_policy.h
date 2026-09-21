#ifndef RASTER_MATERIAL_POLICY_H
#define RASTER_MATERIAL_POLICY_H

// Shared by C++ draw classification and GLSL replay rejection. This is a
// conservative possibility test, not a replacement for per-pixel evaluation.
#ifdef __cplusplus
namespace RasterMaterialPolicy {
inline
#endif
bool rasterMaterialMayTransmit(float transmission, float opacity,
                              bool transmissionMapped, bool opacityMapped,
                              bool hasProgram, bool bubble, bool cutout) {
    // Graphs can replace opacity/transmission; maps are authoritative. Partial
    // opacity can become legacy glass after metallic evaluation. Negated
    // comparisons deliberately retain unknown/NaN scalar values.
    return hasProgram || transmissionMapped || bubble ||
           !(transmission <= float(0.001)) ||
           (!cutout && (opacityMapped || !(opacity >= float(0.99))));
}
#ifdef __cplusplus
} // namespace RasterMaterialPolicy
#endif
#endif
