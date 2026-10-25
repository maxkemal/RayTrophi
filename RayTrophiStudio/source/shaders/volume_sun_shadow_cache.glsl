// A scene directional and the physical sky may request the same shadow chord
// at the same camera sample. Reuse only that exact query, never a nearby point
// or approximately aligned light. Both lighting contributions remain intact.
bool volumeSunQueryValid = false;
vec3 volumeSunQueryPosition;
vec3 volumeSunQueryDirection;
float volumeSunQueryDistance;
float volumeSunQueryVisibility;
int volumeSunQueryStep;

void volumeRememberSunShadow(int lightType, VkVolumeInstance primary,
    bool hasCompanion, VkVolumeInstance companion, vec3 position, vec3 direction,
    float lightLimit, float primaryDistance, float minStep, int step, float visibility) {
    if (lightType != 1 || worldData.w.mode != 2 ||
        !all(equal(direction, normalize(worldData.w.sunDir)))) return;
    float fullPrimary = max(minStep, volumeExitDistance(primary, position, direction));
    float fullCompanion = hasCompanion
        ? max(minStep, volumeExitDistance(companion, position, direction)) : 0.0;
    // Scene directional lights have a finite limit; the sky has none. They
    // represent the same query only when neither medium's chord was clipped.
    if (primaryDistance != fullPrimary || lightLimit < fullCompanion) return;
    volumeSunQueryPosition = position;
    volumeSunQueryDirection = direction;
    volumeSunQueryDistance = primaryDistance;
    volumeSunQueryVisibility = visibility;
    volumeSunQueryStep = step;
    volumeSunQueryValid = true;
}

bool volumeReuseSunShadow(vec3 position, vec3 direction, float distance,
    int step, out float visibility) {
    visibility = 1.0;
    if (!volumeSunQueryValid || volumeSunQueryStep != step ||
        volumeSunQueryDistance != distance ||
        !all(equal(volumeSunQueryPosition, position)) ||
        !all(equal(volumeSunQueryDirection, direction))) return false;
    visibility = volumeSunQueryVisibility;
    return true;
}
