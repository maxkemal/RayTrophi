// Per-grain frame totals in the diagnostics tail, owned by one invocation.
// No float atomics, fixed-point quantization, or additional GPU submission.
const uint AUDIO_HEADER_WORDS = 17u;
const uint AUDIO_WORDS_PER_GRAIN = 12u;
vec4 contactAudio[3] = vec4[3](vec4(0), vec4(0), vec4(0));

bool audioTelemetryEnabled() {
    // The opt-out host binds only the small solver diagnostics header.
    return uint(diagnostics.length()) > AUDIO_HEADER_WORDS;
}

void recordContactWork(uint channel, vec3 point, float energy, bool grain_pair) {
    if (!audioTelemetryEnabled()) {
        return;
    }
    float share = max(0.0, energy) * (grain_pair ? 0.5 : 1.0);
    contactAudio[channel] += vec4(share, share * point);
}

void recordContactDamping(uint i, vec3 point, float normal_speed,
                         float normal_force, float elastic_force,
                         float duration, bool grain_pair) {
    if (!audioTelemetryEnabled()) {
        return;
    }
    // Actual dashpot contribution after the unilateral normal-force clamp.
    // Elastic spring work, tangential/rolling plasticity, drag and MPM are excluded.
    float power = max(0.0, (elastic_force - normal_force) * normal_speed);
    recordContactWork(0u, point, power * duration, grain_pair);
}

void commitContactAudio(uint i) {
    if (!audioTelemetryEnabled()) {
        return;
    }
    uint base = AUDIO_HEADER_WORDS + AUDIO_WORDS_PER_GRAIN * i;
    for (uint channel = 0u; channel < 3u; ++channel) {
        for (uint component = 0u; component < 4u; ++component) {
            uint word = base + channel * 4u + component;
            diagnostics[word] = floatBitsToUint(uintBitsToFloat(diagnostics[word]) +
                                               contactAudio[channel][component]);
        }
    }
}
