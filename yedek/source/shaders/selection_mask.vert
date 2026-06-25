#version 450

// Selection-outline mask pass: re-draws ONLY the selected instances into a
// small R8G8 target. Two pipelines share this shader; they differ in depth
// test + color write mask (G = full silhouette, R = depth-tested visible).
// Vertex stream 0 reuses the raster mesh position buffer verbatim; stream 1
// is the tiny per-selected-node instance matrix buffer.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inModelCol0;
layout(location = 2) in vec4 inModelCol1;
layout(location = 3) in vec4 inModelCol2;
layout(location = 4) in vec4 inModelCol3;

layout(location = 0) out float vMask;

layout(push_constant) uniform SelectionMaskPushConstants {
    mat4 viewProj;
    vec4 maskValue; // x: 1.0 = primary selection, 0.5 = secondary (rest unused)
} pc;

void main() {
    mat4 model = mat4(inModelCol0, inModelCol1, inModelCol2, inModelCol3);
    gl_Position = pc.viewProj * (model * vec4(inPosition, 1.0));
    vMask = pc.maskValue.x;
}
