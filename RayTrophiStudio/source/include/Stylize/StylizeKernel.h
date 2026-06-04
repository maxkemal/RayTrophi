#pragma once
//
// StylizeKernel.h — host-callable launcher for the GPU stylize pass.
//
// Runs the SAME StylizeCore math the CPU path uses (StylizeCore.h), directly on
// the OptiX backend's device AOV buffers. The display color comes from the
// already-graded SDL surface (post tonemap + color grading), uploaded to a
// device buffer, so the GPU result is bire-bir with the CPU stylize that runs
// at the same pipeline position. AOVs stay resident on the device (no readback)
// and only a small 8-bit color buffer round-trips — the costly per-pixel CPU
// compute (sky lobes, fbm, wet-oil, outline) moves to the GPU.
//
// The kernel works in SURFACE space, exactly mirroring
// applyStylizeToSurfaceWithCamera (Main.cpp):
//   * surface pixel (x, y): color = d_color[y*width + x], decoded via the
//     surface format masks/shifts (passed in KernelParams).
//   * AOV is sampled at buffer coords (x, height-1-y); edge neighbours at
//     (x+1, height-1-y) and (x, height-1-y+1).
//   * noise coords passed to applyPostProcess are the surface (x, y).
//

#include "Stylize/StylizeCore.h"

#include <vector_types.h>   // float3, float4
#include <cstdint>

namespace StylizeGPU {

// Per-frame scalars the CPU path pulls from Camera + WorldData when building the
// stylize AOV (see makeStylizeAOV in Main.cpp), plus the SDL surface pixel
// format so the kernel can decode/encode in place.
struct KernelParams {
    int width = 0;
    int height = 0;
    int frame_index = 0;

    // Camera basis for view_dir (mirrors Camera::lower_left_corner + u*horizontal
    // + v*vertical - origin). cam_origin == Camera::origin.
    float3 cam_lower_left{};
    float3 cam_horizontal{};
    float3 cam_vertical{};
    float3 cam_origin{};

    // Ray origin used to reconstruct linear depth from world position. This is
    // Camera::lookfrom (the GPU ray origin per syncCameraToBackend), which the
    // CPU AOV path uses as cameraOrigin in fillStylizeAOVFromBackend.
    float3 ray_origin{};

    // Sky / sun world data (WorldData::nishita) for the stylize sky layer.
    float3 sun_direction{};
    float  sun_size = 0.545f;
    float  sun_elevation = 35.0f;
    int    clouds_enabled = 0;
    float  cloud_coverage = 0.45f;
    float  cloud_density = 0.65f;
    float  cloud_scale = 1.0f;
    float  cloud_offset_x = 0.0f;
    float  cloud_offset_z = 0.0f;
    int    cloud_seed = 0;

    // SDL surface pixel format (32-bit packed). The kernel decodes color from
    // each uint32 with the same masks/shifts the CPU path uses and re-encodes,
    // preserving the alpha bits.
    uint32_t rMask = 0, gMask = 0, bMask = 0, aMask = 0;
    int rShift = 0, gShift = 0, bShift = 0;
};

// Launch the in-place stylize pass on d_color (uint32 packed, width*height,
// SURFACE space). d_position is required (stylize AOV); d_albedo/d_normal are
// optional but the surface-locked stylize needs them for parity with the CPU
// path. `stream` is a cudaStream_t (nullable). Returns false on a null
// color/position buffer or a launch error.
bool launchStylize(uint32_t* d_color,
                   const float4* d_position,
                   const float4* d_albedo,
                   const float4* d_normal,
                   const KernelParams& params,
                   const StylizeCore::StyleProfileCore& profile,
                   void* stream);

} // namespace StylizeGPU
