#version 450

layout(location = 0) out vec2 vNdc;

void main() {
    vec2 p = gl_VertexIndex == 0 ? vec2(-1.0, -1.0) :
             gl_VertexIndex == 1 ? vec2( 3.0, -1.0) : vec2(-1.0, 3.0);
    vNdc = p;
    gl_Position = vec4(p, 0.999999, 1.0);
}
