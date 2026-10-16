// First-order image-source path on a sampled finite triangle. Candidate discovery
// is bounded; a missed plane is not evidence that the scene has no reflected path.
bool specularPath(Source source, vec3 receiver, vec3 receiverNormal,
    vec3 a, vec3 b, vec3 c, vec3 normal, Instance wall,
    out float energy, out bool found) {
    energy = 0.0;
    found = false;
    if (source.positionGain.w <= 0.0 ||
        response(wall.specular0, wall.specular1) <= 0.0) return true;
    float sourceSide = dot(source.positionGain.xyz - a, normal);
    float receiverSide = dot(receiver - a, normal);
    if (sourceSide <= 0.0001 || receiverSide <= 0.0001) return true;
    vec3 imageSource = source.positionGain.xyz - 2.0 * sourceSide * normal;
    vec3 delta = imageSource - receiver;
    float denominator = dot(delta, normal);
    if (abs(denominator) < 1e-8) return true;
    float t = -receiverSide / denominator;
    if (t <= 0.0 || t >= 1.0) return true;
    vec3 point = receiver + t * delta;
    vec3 e0 = b - a, e1 = c - a, offset = point - a;
    float d00 = dot(e0, e0), d01 = dot(e0, e1), d11 = dot(e1, e1);
    float determinant = d00 * d11 - d01 * d01;
    if (determinant <= 1e-16) return true;
    float u = (d11 * dot(offset, e0) - d01 * dot(offset, e1)) / determinant;
    float v = (d00 * dot(offset, e1) - d01 * dot(offset, e0)) / determinant;
    if (u < -1e-5 || v < -1e-5 || u + v > 1.00001) return true;
    float firstLength = length(point - receiver);
    float secondLength = length(source.positionGain.xyz - point);
    if (firstLength < 0.0004 || secondLength < 0.0004) return true;
    float cosine = max(0.0, dot(receiverNormal, (point - receiver) / firstLength));
    float incoming = sourceEnergyPath(source, point, firstLength + secondLength) * cosine;
    if (incoming <= 0.0) return true;
    vec4 first0, first1, second0, second1;
    vec3 surface = point + normal * 0.0002;
    if (!transmission(receiver, surface, first0, first1) ||
        !transmission(surface, source.positionGain.xyz, second0, second1)) return false;
    energy = incoming * response(first0 * second0 * wall.specular0,
        first1 * second1 * wall.specular1);
    found = true;
    return true;
}
