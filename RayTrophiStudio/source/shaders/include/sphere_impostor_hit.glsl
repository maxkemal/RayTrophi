// Rays start at the near plane: works for perspective and orthographic views.
void sphereHit() {
    vec3 direction = normalize(sphereFar - sphereNear);
    vec3 offset = sphereNear - sphereCenterRadius.xyz;
    float b = dot(offset, direction);
    float c = dot(offset, offset) - sphereCenterRadius.w * sphereCenterRadius.w;
    float discriminant = b * b - c;
    if (discriminant < 0.0) discard;
    float root = sqrt(discriminant);
    float distanceAlongRay = -b - root;
    if (distanceAlongRay < 0.0) distanceAlongRay = -b + root;
    if (distanceAlongRay < 0.0) discard;
    vWorldPos = sphereNear + direction * distanceAlongRay;
    vec4 clip = solidParams.viewProj * vec4(vWorldPos, 1.0);
    float depth = clip.z / clip.w;
    if (clip.w <= 0.0 || depth < 0.0 || depth > 1.0) discard;
    gl_FragDepth = depth;
    vNormal = normalize(vWorldPos - sphereCenterRadius.xyz);
    if (solidParams.useMatcap != 0) {
        vNormal = mat3(solidParams.view) * vNormal;
    }
}
