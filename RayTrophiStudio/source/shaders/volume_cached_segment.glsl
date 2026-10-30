// Shared entry for ordinary and general-mixture bounce segments. Keep the
// expensive vertex fill reachable through exactly one march/fetch call site.
bool volumeCachedSegment(uint selected, int companion, uint count, bool mixture,
    vec3 origin, vec3 direction, float nearT, float farT,
    pnanovdb_buf_t buf, pnanovdb_map_handle_t mapH,
    inout pnanovdb_readaccessor_t acc,
    pnanovdb_buf_t compBuf, pnanovdb_map_handle_t compMapH,
    inout pnanovdb_readaccessor_t compAcc, out vec3 radiance, out vec3 transmission) {
    radiance = vec3(0.0);
    transmission = vec3(1.0);
    uint marchLod = worldData.w.volumeMarchLod;
    if (volumeCameraSegment || ((marchLod >> 21) & 1u) == 0u) return false;
    uint slots = companion >= 0 ? 2u : 1u;
    uint members = mixture ? min(count, 16u) : slots;
    for (uint m = 0u; m < members; ++m) {
        uint index = mixture ? m : (m == 0u ? selected : uint(companion));
        VkVolumeInstance v = volumes.v[index];
        if (mixture && !gasOverlapCandidate(v)) continue;
        uint program = v._reserved[1] > 0.5
            ? matProgramOffset(uint(v._reserved[1] - 1.0)) : MATPROG_NONE;
        if (program != MATPROG_NONE || v.random_walk_enabled > 0.5 ||
            (worldData.w.mode == 2 && (v.volume_type == 3 || v.source_type == 3))) {
            return false;
        }
    }
    float jitter = fract(sin(dot(origin + direction * 17.0,
        vec3(12.9898, 78.233, 37.719)) + float(payload.seed) * 0.000173) * 43758.5453);
    bool cached = volumeRadianceCacheMarch(selected,
        companion >= 0 ? uint(companion) : selected, slots, mixture ? count : 0u,
        origin, direction, nearT, farT, jitter, buf, mapH, acc,
        compBuf, compMapH, compAcc, radiance, transmission);
    volumeRecordRadianceCache(cached);
    return cached;
}
