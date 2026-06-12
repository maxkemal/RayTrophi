#version 450

// Fullscreen triangle for the selection-outline composite pass (no vertex
// buffer; covers the viewport with 3 vertices from gl_VertexIndex).

layout(location = 0) out vec2 vUV;

void main() {
    vUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(vUV * 2.0 - 1.0, 0.0, 1.0);
}
