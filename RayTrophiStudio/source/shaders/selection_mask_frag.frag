#version 450

// Writes the selection weight into both channels; the pipeline's color write
// mask picks which channel this draw actually lands in (R or G).

layout(location = 0) in float vMask;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(vMask, vMask, 0.0, 0.0);
}
