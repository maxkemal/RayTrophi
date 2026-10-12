// RayFusion cloud interleave: which pixel of each N x N cell is marched this
// frame. Shared by cloud_raster.comp (march) and cloud_resolve.comp (resolve);
// CPU side: VulkanDevice::recordCloudRaster (push ABI CloudInterleavePush).
#ifndef CLOUD_INTERLEAVE_GLSL
#define CLOUD_INTERLEAVE_GLSL

layout(push_constant) uniform CloudInterleavePush {
    uint n;         // cell size N (1..4); 1 = every pixel every frame
    uint phaseX;    // marched pixel inside the cell this frame
    uint phaseY;
    uint freshW;    // fresh grid = ceil(target / N)
    uint freshH;
} cloudInterleave;

ivec2 cloudInterleavePixel(ivec2 cell) {
    return cell * int(cloudInterleave.n) +
           ivec2(int(cloudInterleave.phaseX), int(cloudInterleave.phaseY));
}

#endif
