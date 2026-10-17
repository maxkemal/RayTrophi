// Single-bounce energetic transport. Uniform-sphere source quadrature followed
// by next-event estimation to the listener; no source ray is counted twice.
//
// Lambert BRDF = Rd/pi. For p(direction)=1/(4*pi), cos_i*dA/r_source^2
// cancels the source sampling measure. Relative to free-field direct energy:
//   contribution = 4 * Rd * cos_o * r_direct^2 / (N * r_listener^2).
// This is a receiver response estimate, NOT a material probability coefficient.
void diffusePaths(Ray source, inout Summary result) {
    if (source.options.z < 0.5) return;
    const uint samples = 64u;
    const float goldenAngle = 2.399963229728653;
    float units = source.source.w;
    float epsilon = max(0.0001 / units, result.path.x / units * 1e-6);
    float horizon = (result.path.x + 343.0 * source.options.y) / units;
    result.diffuseState.x = samples;
    for (uint sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
        float z = 1.0 - 2.0 * (float(sampleIndex) + 0.5) / float(samples);
        float radius = sqrt(max(0.0, 1.0 - z * z));
        float phi = goldenAngle * float(sampleIndex);
        vec3 direction = vec3(radius * cos(phi), radius * sin(phi), z);
        rayQueryEXT query;
        rayQueryInitializeEXT(query, sceneAS, gl_RayFlagsNoOpaqueEXT, 0xFFu,
            source.source.xyz, epsilon, direction, horizon);
        bool valid = true;
        uint candidates = 0u;
        while (rayQueryProceedEXT(query)) {
            if (++candidates > 1024u) {
                valid = false;
                break;
            }
            uint index = rayQueryGetIntersectionInstanceIdEXT(query, false);
            if (index >= pc.instanceCount) {
                valid = false;
                break;
            }
            if (instances[index].meta.w == 2u) continue;
            if (rayQueryGetIntersectionTypeEXT(query, false) !=
                gl_RayQueryCandidateIntersectionTriangleEXT) {
                valid = false;
                break;
            }
            rayQueryConfirmIntersectionEXT(query);
        }
        if (!valid) {
            ++result.diffuseState.z;
            continue;
        }
        if (rayQueryGetIntersectionTypeEXT(query, true) ==
            gl_RayQueryCommittedIntersectionNoneEXT) continue;
        uint index = rayQueryGetIntersectionInstanceIdEXT(query, true);
        uint primitive = rayQueryGetIntersectionPrimitiveIndexEXT(query, true);
        if (index >= pc.instanceCount) {
            ++result.diffuseState.z;
            continue;
        }
        Instance wall = instances[index];
        if (wall.meta.w != 1u || wall.positions == uint64_t(0) || primitive >= wall.meta.y) {
            ++result.diffuseState.z;
            continue;
        }
        if (dot(wall.diffuse0, vec4(1)) + dot(wall.diffuse1.xy, vec2(1)) <= 0.0) continue;
        uvec3 corners = uvec3(primitive * 3u) + uvec3(0, 1, 2);
        if (wall.indices != uint64_t(0)) {
            Indices indices = Indices(wall.indices);
            corners = uvec3(indices.v[corners.x], indices.v[corners.y], indices.v[corners.z]);
        }
        if (any(greaterThanEqual(corners, uvec3(wall.meta.x)))) {
            ++result.diffuseState.z;
            continue;
        }
        mat4x3 transform = rayQueryGetIntersectionObjectToWorldEXT(query, true);
        vec3 a = transform * vec4(position(wall, corners.x), 1);
        vec3 b = transform * vec4(position(wall, corners.y), 1);
        vec3 c = transform * vec4(position(wall, corners.z), 1);
        vec3 normal = cross(b - a, c - a);
        float magnitude = length(normal);
        if (isnan(magnitude) || isinf(magnitude) || magnitude < 1e-12) {
            ++result.diffuseState.z;
            continue;
        }
        normal /= magnitude;
        if (dot(normal, -direction) < 0.0) normal = -normal;
        float firstLength = rayQueryGetIntersectionTEXT(query, true);
        vec3 point = source.source.xyz + direction * firstLength;
        vec3 outgoing = source.listener.xyz - point;
        float secondLength = length(outgoing);
        // Explicit finite receiver exclusion: point-receiver quadrature is
        // singular close to a sample. Report the excluded region, never clamp
        // its response to an invented material coefficient.
        if (secondLength * units < 0.1) {
            ++result.diffuseState.w;
            continue;
        }
        float cosine = dot(normal, outgoing / secondLength);
        if (cosine <= 0.0) continue;
        float metres = (firstLength + secondLength) * units;
        float delay = max(0.0, (metres - result.path.x) / 343.0);
        if (delay > source.options.y) continue;
        bool visibilityComplete;
        if (!visible(point, source.listener.xyz, epsilon, visibilityComplete)) {
            if (!visibilityComplete) ++result.diffuseState.z;
            continue;
        }
        float listenerMetres = secondLength * units;
        float weight = 4.0 * cosine * result.path.x * result.path.x /
            (float(samples) * listenerMetres * listenerMetres);
        vec4 energy0 = wall.diffuse0 * weight;
        vec4 energy1 = wall.diffuse1 * weight;
        float mass = dot(energy0, vec4(1)) + dot(energy1.xy, vec2(1));
        if (isnan(mass) || isinf(mass) || mass <= 0.0) {
            ++result.diffuseState.z;
            continue;
        }
        uint bin = min(7u, uint(floor(delay / source.options.y * 8.0)));
        uint slot = 6u + bin;
        result.reflections[slot].energy0 += energy0;
        result.reflections[slot].energy1 += energy1;
        result.reflections[slot].path += vec4(metres * mass, delay * mass, mass, 0);
        ++result.diffuseState.y;
    }
    for (uint bin = 0u; bin < 8u; ++bin) {
        uint slot = 6u + bin;
        float mass = result.reflections[slot].path.z;
        if (mass > 0.0) {
            result.reflections[slot].path.xy /= mass;
            result.reflections[slot].path.zw = vec2(1, 1);
        }
    }
}
