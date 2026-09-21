#version 450
layout(push_constant) uniform Push {
    mat4 viewProj;
    vec4 centerRadius;
    vec4 color;
} pc;
layout(location = 0) out vec4 markerColor;
// Eight triangles form a small octahedron at the REAL producer centre.
const vec3 corners[6] = vec3[6](vec3(1,0,0),vec3(-1,0,0),vec3(0,1,0),
                                vec3(0,-1,0),vec3(0,0,1),vec3(0,0,-1));
const int indices[24] = int[24](2,0,4, 2,4,1, 2,1,5, 2,5,0,
                               3,4,0, 3,1,4, 3,5,1, 3,0,5);
void main() {
    vec3 p = pc.centerRadius.xyz + corners[indices[gl_VertexIndex]] * pc.centerRadius.w;
    gl_Position = pc.viewProj * vec4(p, 1);
    markerColor = pc.color;
}
