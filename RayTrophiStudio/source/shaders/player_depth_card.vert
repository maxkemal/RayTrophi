#version 450
layout(location = 0) in uint rgba;
layout(location = 1) in float depthOffset;
layout(location = 0) flat out vec4 color;
layout(push_constant) uniform Card {
    mat4 viewProj;
    vec4 right;
    vec4 up;
    vec4 anchor; // xyz floor anchor, w physical height
    vec4 image;  // width, height, reserved
} pc;

void main() {
    const vec2 corners[6] = vec2[6](vec2(0,0),vec2(1,0),vec2(1,1),
                                  vec2(0,0),vec2(1,1),vec2(0,1));
    uint width = uint(pc.image.x);
    vec2 pixel = vec2(gl_InstanceIndex % width, gl_InstanceIndex / width);
    vec2 uv = (pixel + corners[gl_VertexIndex]) / pc.image.xy;
    float physicalWidth = pc.anchor.w * pc.image.x / pc.image.y;
    vec3 forward = normalize(cross(pc.up.xyz, pc.right.xyz));
    vec3 world = pc.anchor.xyz + pc.right.xyz * ((uv.x - 0.5) * physicalWidth)
               + pc.up.xyz * ((1.0 - uv.y) * pc.anchor.w) + forward * depthOffset;
    color = unpackUnorm4x8(rgba);
    gl_Position = pc.viewProj * vec4(world, 1);
    if (isnan(depthOffset) || isinf(depthOffset) || abs(depthOffset) > 20.0) {
        gl_Position = vec4(2, 2, 2, 1);
    }
}
