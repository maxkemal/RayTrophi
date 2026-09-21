#version 450

layout(location = 0) in float vVCoord;
layout(location = 1) in vec3 fColor;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    mat4 viewProj;
    mat4 view;
    int useMatcap;
    float overrideR, overrideG, overrideB;
} pc;

void main() {
    // Grid/wireframe override path (useMatcap == -1)
    if (pc.useMatcap == -1) {
        outColor = vec4(pc.overrideR, pc.overrideG, pc.overrideB, 1.0);
        return;
    }
    
    vec3 baseColor = fColor;
    
    // Simple directional + ambient lighting
    vec3 fakeLightDir = normalize(vec3(0.5, 0.8, 0.3));
    vec3 viewNormal = vec3(0.0, 0.0, 1.0); // Billboard normal facing camera
    float ndotl = max(dot(viewNormal, fakeLightDir), 0.0);
    
    vec3 ambient = vec3(0.15, 0.15, 0.18); // Neutral ambient
    vec3 diffuse = baseColor * (ambient + ndotl * 0.7);
    
    // Add a subtle rim/specular highlight for depth
    float rim = pow(1.0 - abs(dot(viewNormal, fakeLightDir)), 3.0) * 0.1;
    
    outColor = vec4(diffuse + vec3(rim), 1.0);
}
