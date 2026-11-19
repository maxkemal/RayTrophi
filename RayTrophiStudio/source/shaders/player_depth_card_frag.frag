#version 450
layout(location = 0) flat in vec4 color;
layout(location = 0) out vec4 outColor;

void main() {
    // Spatial coverage preserves soft segmentation edges while writing real depth.
    // Empty samples write neither colour nor depth. No blended fragment writes
    // a full opaque depth edge over particles behind the card.
    const int bayer[16] = int[16](0,8,2,10,12,4,14,6,3,11,1,9,15,7,13,5);
    ivec2 p = ivec2(gl_FragCoord.xy) & 3;
    float threshold = (float(bayer[p.y * 4 + p.x]) + 0.5) / 16.0;
    if (color.a <= threshold) discard;
    outColor = vec4(color.rgb, 1);
}
