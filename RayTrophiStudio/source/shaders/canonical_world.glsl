// Canonical world radiance — the ONE producer of sky/HDRI radiance for the
// raster path. The fragment shader draws it per pixel; the sky capture pass
// bakes it into an equirect so the IBL builder can prefilter it. A second copy
// of this function would drift silently: the background and the ambient term
// would disagree and nothing would report it.
//
// Requires, from the including stage: worldMode, sceneFlags, worldParams,
// worldColor, worldSun, atmosphereA, atmosphereB, pc.cameraPos, and the
// samplers worldEnvironment, atmosphereSkyView, atmosphereTransmittance.
//
// includeSunDisc=false drops ONLY the sun disc. The sun is delivered to the
// raster path as an analytic directional light with its own shadow atlas tile;
// baking the disc into the diffuse irradiance would add the sun twice.

vec2 worldDirToUV(vec3 d) {
    d = normalize(d);
    float phi = atan(d.z, d.x) - worldParams.x;
    return vec2(fract(phi / (2.0 * 3.14159265359) + 0.5),
                acos(clamp(d.y, -1.0, 1.0)) / 3.14159265359);
}

vec3 sampleCanonicalWorldEx(vec3 d, bool includeSunDisc) {
    d = normalize(d);
    if (worldMode == 1u) {
        if ((sceneFlags & 2u) != 0u)
            return texture(worldEnvironment, worldDirToUV(d)).rgb * max(worldParams.y, 0.0);
        return vec3(0.0);
    }

    if (worldMode == 2u) {
        vec3 result;
        if ((sceneFlags & 8u) != 0u) {
            float azimuth = atan(d.z, d.x) / (2.0 * 3.14159265359);
            if (azimuth < 0.0) azimuth += 1.0;
            result = texture(atmosphereSkyView,
                             vec2(azimuth, (1.0 - clamp(d.y, -1.0, 1.0)) * 0.5)).rgb;
            // Match miss.rmiss: the LUT is single scatter; RT adds bounded
            // second/third order scatter before exposing it as sky radiance.
            if (atmosphereA.x > 0.5) {
                vec3 scatteringAlbedo = vec3(0.8, 0.85, 0.9);
                vec3 secondOrder = result * scatteringAlbedo * 0.5 * exp(-0.5 * 0.3);
                vec3 thirdOrder = secondOrder * scatteringAlbedo * 0.25 * exp(-0.5 * 0.1);
                result += secondOrder * atmosphereA.y +
                          thirdOrder * (atmosphereA.y * 0.5);
            }
        } else {
            float up = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
            vec3 horizon = vec3(0.42, 0.53, 0.68);
            vec3 zenith = vec3(0.09, 0.24, 0.52);
            vec3 ground = max(worldColor.rgb, vec3(0.025));
            vec3 sky = mix(horizon, zenith, pow(up, 0.65));
            result = mix(ground, sky, smoothstep(0.0, 0.12, d.y)) *
                     max(worldParams.z / 10.0, 0.0);
        }
        vec3 sunDir = dot(worldSun.xyz, worldSun.xyz) > 1e-8
            ? normalize(worldSun.xyz) : vec3(0.0, 1.0, 0.0);
        float sunSize = max(worldColor.w, 0.05);
        float elevation = degrees(asin(clamp(sunDir.y, -1.0, 1.0)));
        if (elevation < 15.0)
            sunSize *= 1.0 + (15.0 - max(elevation, -10.0)) * 0.04;
        float sunRadius = radians(sunSize * 0.5);
        float mu = dot(d, sunDir);
        if (includeSunDisc && mu > cos(sunRadius) && worldSun.w > 0.0) {
            float radial = acos(clamp(mu, -1.0, 1.0)) / max(sunRadius, 1e-6);
            float limb = 1.0 - 0.6 * (1.0 - sqrt(max(0.0, 1.0 - radial * radial)));
            float edge = 1.0 - smoothstep(0.85, 1.0, radial);
            vec3 transSun = vec3(1.0);
            if ((sceneFlags & 8u) != 0u) {
                float u = clamp((max(0.01, sunDir.y) + 0.2) / 1.2, 0.0, 1.0);
                float radius = max(atmosphereB.x, 1.0);
                float altitude = max(0.0, length(pc.cameraPos.xyz + vec3(0.0, radius, 0.0)) - radius);
                float v = clamp(altitude / max(atmosphereB.y, 1.0), 0.0, 1.0);
                transSun = texture(atmosphereTransmittance, vec2(u, v)).rgb;
            }
            result += transSun * worldSun.w * 80000.0 * limb * edge;
        }
        if ((sceneFlags & 4u) != 0u) {
            vec3 sampled = texture(worldEnvironment, worldDirToUV(d)).rgb;
            float strength = max(worldParams.y, 0.0);
            float amount = min(strength, 1.0);
            vec3 overlay = sampled * strength;
            int blendMode = int(worldParams.w + 0.5);
            if (blendMode == 1) result *= mix(vec3(1.0), sampled, amount);
            else if (blendMode == 2) result += overlay;
            else if (blendMode == 3) result = overlay;
            else result = mix(result, overlay, amount);
        }
        return result;
    }
    return max(worldColor.rgb * worldColor.w, vec3(0.0));
}

vec3 sampleCanonicalWorld(vec3 d) { return sampleCanonicalWorldEx(d, true); }
