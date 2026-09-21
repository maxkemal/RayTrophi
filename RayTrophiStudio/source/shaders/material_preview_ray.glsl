#ifndef RT_MATERIAL_PREVIEW_RAY_GLSL
#define RT_MATERIAL_PREVIEW_RAY_GLSL

// Unproject two finite Vulkan depth points. The raster projection uses
// near=0.01, far=1e6: in float32 its far plane can have homogeneous w=0.
// Dividing the z=1 point by w produces an invalid ray (or reverses it when
// inversion roundoff makes w slightly negative). Do not clamp that w.
// Starting on the near plane also handles orthographic views and cameras
// inside a medium, without assuming that every ray starts at cameraPos.
void rtPreviewWorldRay(mat4 invViewProj, vec2 ndc, out vec3 origin,
                       out vec3 direction) {
    vec4 nearH = invViewProj * vec4(ndc, 0.0, 1.0);
    vec4 interiorH = invViewProj * vec4(ndc, 0.5, 1.0);
    origin = nearH.xyz / nearH.w;
    vec3 interior = interiorH.xyz / interiorH.w;
    direction = normalize(interior - origin);
}

#endif
