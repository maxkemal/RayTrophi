// One light invocation per grain. The heavy solver receives only the packed
// active indices; sleeping grains carry bank state and elapsed spring time here.
// Source positions/velocities remain in readBank for every pair this substep.
void classifyGrainWork(uint i) {
    uint bank = readBank();
    vec3 p = position(i, bank);
    vec3 v = velocity(i, bank);
    vec3 w = omega(i, bank);
    if (invalidGrainAttempt()) {
        store(i, writeBank(), p, v, w);
        return;
    }
    uint entry = history_blocks[i];
    g_block = entry & ~FRESH;
    bool valid_history = (entry & FRESH) == 0u && history_blocks[ownerWord()] == ids[i];
    vec4 drag = coupling[3u * i];
    vec4 lift = coupling[3u * i + 1u];
    bool can_sleep = drag.w == 0.0 && lift == vec4(0.0);
    bool slow = dot(v, v) < pc.sleep.y * pc.sleep.y && length(w) * radius(i) < pc.sleep.y;
    bool skip = false;
    uint elapsed_word = ownerWord() + 6u;
    if (sleepOn() && valid_history && can_sleep && slow &&
        floatBitsToUint(pc.sleep.z) == 0u && !grainSleepAudit()) {
        uint rest = history_blocks[ownerWord() + 5u];
        skip = (rest & AUDIT) == 0u && (rest & ~AUDIT) >= sleepRestLimit();
    }
    if (skip) {
        // No pending audit is consumed here. Requests raised by the active
        // solver remain pending and are observed by the next classifier.
        float elapsed = uintBitsToFloat(history_blocks[elapsed_word]);
        history_blocks[elapsed_word] = floatBitsToUint(elapsed + pc.step_contact.x);
        store(i, writeBank(), p, vec3(0.0), vec3(0.0));
        grainSleepProbeCount(0u);
        return;
    }
    uint slot = atomicAdd(diagnostics[46], 1u);
    scratch[9u * pc.meta.x + slot] = uintBitsToFloat(i);
}
