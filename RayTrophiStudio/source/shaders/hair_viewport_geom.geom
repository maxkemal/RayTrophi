#version 450

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

layout(location = 0) in float vVCoord[];
layout(location = 1) in float vThickness[];
layout(location = 2) in vec3 vColor[];

layout(location = 0) out float fVCoord;
layout(location = 1) out vec3 fColor;

layout(push_constant) uniform PC {
    mat4 viewProj;
    mat4 view;
    int useMatcap;
    float overrideR, overrideG, overrideB;
} pc;

void main() {
    vec3 p0 = gl_in[0].gl_Position.xyz;
    vec3 p1 = gl_in[1].gl_Position.xyz;
    
    vec3 dir = p1 - p0;
    if (length(dir) < 1e-6) {
        return; // Degenerate line
    }
    dir = normalize(dir);
    
    // Extract forward vector from view matrix (third row)
    vec3 wForward = vec3(pc.view[0][2], pc.view[1][2], pc.view[2][2]); 
    // Right vector for the billboard
    vec3 rightDir = normalize(cross(dir, wForward));
    
    // Use the actual hair thickness, scaled by 0.7 to match RT cylinder appearance
    float w0 = max(vThickness[0] * 0.5 * 0.7, 1e-4);
    float w1 = max(vThickness[1] * 0.5 * 0.7, 1e-4);
    
    vec4 p00 = pc.viewProj * vec4(p0 - rightDir * w0, 1.0);
    vec4 p01 = pc.viewProj * vec4(p0 + rightDir * w0, 1.0);
    vec4 p10 = pc.viewProj * vec4(p1 - rightDir * w1, 1.0);
    vec4 p11 = pc.viewProj * vec4(p1 + rightDir * w1, 1.0);
    
    fVCoord = vVCoord[0];
    fColor = vColor[0];
    gl_Position = p00; EmitVertex();
    
    fVCoord = vVCoord[0];
    fColor = vColor[0];
    gl_Position = p01; EmitVertex();
    
    fVCoord = vVCoord[1];
    fColor = vColor[1];
    gl_Position = p10; EmitVertex();
    
    fVCoord = vVCoord[1];
    fColor = vColor[1];
    gl_Position = p11; EmitVertex();
    
    EndPrimitive();
}
