#version 450

// Edit-mesh overlay: flat color passthrough for edge / face-fill pipelines.

layout(location = 0) in vec4 vColor;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vColor;
}
