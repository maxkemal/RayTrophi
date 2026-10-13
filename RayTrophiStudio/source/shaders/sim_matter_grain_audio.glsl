// Per-grain frame totals in the diagnostics tail, owned by one invocation.
// No float atomics, fixed-point quantization, or additional GPU submission.
const uint AUDIO_HEADER_WORDS = 17u;
const uint AUDIO_WORDS_PER_GRAIN = 4u;

void recordContactDamping(uint i, vec3 point, float normal_speed,
                         float normal_force, float elastic_force,
                         float duration, bool grain_pair) {
    // Actual dashpot contribution after the unilateral normal-force clamp.
    // Elastic spring work, tangential/rolling plasticity, drag and MPM are excluded.
    float power = max(0.0, (elastic_force - normal_force) * normal_speed);
    float energy = power * duration * (grain_pair ? 0.5 : 1.0);
    uint base = AUDIO_HEADER_WORDS + AUDIO_WORDS_PER_GRAIN * i;
    diagnostics[base] = floatBitsToUint(uintBitsToFloat(diagnostics[base]) + energy);
    for (uint axis = 0u; axis < 3u; ++axis) {
        diagnostics[base + 1u + axis] = floatBitsToUint(
            uintBitsToFloat(diagnostics[base + 1u + axis]) + energy * point[axis]);
    }
}
