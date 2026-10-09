#ifndef RT_MATTER_GRAIN_SLEEP
#define RT_MATTER_GRAIN_SLEEP

// Included after the grain push constants, history storage and AUDIT constant.
// No extra per-grain storage or dispatch. Most sleeping substeps only copy state;
// an audit evaluates real contacts at least once per frame, and every <=20 ms
// (or every substep when the integration step itself is larger).
bool grainSleepAudit() {
    float interval = min(0.02, pc.sleep.w);
    uint period = max(1u, uint(floor(interval / pc.step_contact.x)));
    return pc.substep.x == pc.substep.w || (pc.substep.x + 1u) % period == 0u;
}

bool grainSleepBalanced(vec3 acceleration, vec3 angular_acceleration, float radius,
                       vec3 position, float inverse_mass, uint contacts) {
    // A low instantaneous velocity can be a turning point. Require the residual
    // force and torque to change surface speed by less than the sleep threshold
    // over the requested sleep duration, not just one tiny integration step.
    float horizon = max(pc.sleep.w, pc.step_contact.x);
    float tolerance = pc.sleep.y / horizon;
    // Millimetre grains in a metre-scale pile cannot resolve an arbitrarily
    // small force residual with float positions. Bound spring-force uncertainty
    // by two position ULPs per contact, rather than keeping such grains awake
    // forever. Sphere inertia turns that into 2.5x surface angular acceleration.
    float coordinate = max(radius, max(abs(position.x),
        max(abs(position.y), abs(position.z))));
    uint exponent = (floatBitsToUint(coordinate) >> 23u) & 255u;
    float ulp = exponent > 23u ? uintBitsToFloat((exponent - 23u) << 23u)
                              : uintBitsToFloat(1u);
    float uncertainty = 2.0 * ulp * pc.high_stiffness.w * inverse_mass * float(contacts);
    float linear_tolerance = max(tolerance, uncertainty);
    float angular_tolerance = max(tolerance, 2.5 * uncertainty);
    // Neither loose user settings nor an imprecise contact may hide the full
    // gravity acceleration of a removed support / unsupported free fall.
    float gravity = length(pc.rolling.yzw);
    if (gravity > 0.0) {
        linear_tolerance = min(linear_tolerance, 0.25 * gravity);
        angular_tolerance = min(angular_tolerance, 0.625 * gravity);
    }
    return dot(acceleration, acceleration) < linear_tolerance * linear_tolerance &&
        dot(angular_acceleration, angular_acceleration) * radius * radius <
            angular_tolerance * angular_tolerance;
}

uint grainSleepTakeAudit(uint word) {
    uint previous = history_blocks[word];
    if ((previous & AUDIT) != 0u) {
        return atomicAnd(history_blocks[word], ~AUDIT);
    }
    // An audit arriving after this read survives commit for the next substep.
    return previous;
}

void grainSleepCommit(uint word, uint rest) {
    if (pc.sleep.y <= 0.0) {
        // The host's context change wakes all grains when sleep is re-enabled.
        return;
    }
    if (rest == 0u) {
        // This owner is already physically awake. A concurrent audit request need not
        // survive a zero commit; it cannot cause this owner to skip contacts.
        if (history_blocks[word] != 0u) {
            atomicExchange(history_blocks[word], 0u);
        }
        return;
    }
    // One atomic replacement preserves a concurrent audit request. The former
    // atomicAnd(AUDIT), atomicOr(rest) pair briefly exposed rest=0 to neighbours.
    uint previous = history_blocks[word];
    for (;;) {
        uint desired = (previous & AUDIT) | rest;
        uint observed = atomicCompSwap(history_blocks[word], previous, desired);
        if (observed == previous) {
            return;
        }
        previous = observed;
    }
}

#endif
