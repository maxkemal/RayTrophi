// Opt-in counters only. No history, rest clock or force-law mutations.
// 28..32: frame invocation/events; 33..43: final-substep grain snapshot.
void grainSleepProbeCount(uint counter) {
    if (!contactProbeEnabled()) return;
    uint previous = atomicAdd(diagnostics[28u + counter], 1u);
    if (previous == 0xffffffffu) atomicOr(diagnostics[44], 1u);
}

void grainSleepProbeFinal(bool valid_history, bool can_sleep, bool context_changed,
                         vec3 v, vec3 w, float r, uint next_rest) {
    if (!contactProbeEnabled() || pc.substep.x != pc.substep.w) return;
    grainSleepProbeCount(14u); // sampled grains
    if (!sleepOn()) {
        grainSleepProbeCount(15u);
        return;
    }
    bool linear_slow = dot(v, v) < pc.sleep.y * pc.sleep.y;
    bool angular_slow = length(w) * r < pc.sleep.y;
    if (!linear_slow) grainSleepProbeCount(5u);
    if (!angular_slow) grainSleepProbeCount(6u);
    if (!grainSleepBalanceParts.x) grainSleepProbeCount(7u);
    if (!grainSleepBalanceParts.y) grainSleepProbeCount(8u);
    if (!can_sleep) grainSleepProbeCount(9u);
    if (!valid_history) grainSleepProbeCount(10u);
    if (context_changed) grainSleepProbeCount(11u);
    bool eligible = can_sleep && linear_slow && angular_slow && all(grainSleepBalanceParts);
    if (eligible) {
        grainSleepProbeCount(13u);
        if (next_rest < sleepRestLimit()) grainSleepProbeCount(12u);
    }
}
