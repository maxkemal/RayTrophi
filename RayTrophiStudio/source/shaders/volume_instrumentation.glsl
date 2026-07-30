#ifndef RAYTROPHI_VOLUME_INSTRUMENTATION_GLSL
#define RAYTROPHI_VOLUME_INSTRUMENTATION_GLSL

layout(set = 0, binding = 29, scalar) buffer VolumeInstrumentationBuffer {
    uint volumeRays;
    uint densitySamples;
    uint emptySegmentsSkipped;
    uint topologySegmentsSkipped;
    uint majorantSegmentsSkipped;
    uint shadowDensitySamples;
    uint extinctionTerminations;
    uint stepBudgetExhausted;
    uint completedIntervals;
    uint temporalAccepted;
    uint temporalRejected;
    uint majorantQueries;
    uint majorantAvailableQueries;
    uint reserved2;
    uint reserved3;
    uint enabled;
} volumeInstrumentation;

const uint VOLUME_MARCH_COMPLETED = 0u;
const uint VOLUME_MARCH_EXTINCTION = 1u;
const uint VOLUME_MARCH_STEP_BUDGET = 2u;

bool volumeInstrumentationEnabled() {
    return volumeInstrumentation.enabled != 0u;
}

void volumeRecordRay(
    uint densityCount,
    uint emptyCount,
    uint topologyEmptyCount,
    uint densityLeafEmptyCount,
    uint outcome)
{
    if (!volumeInstrumentationEnabled()) return;
    atomicAdd(volumeInstrumentation.volumeRays, 1u);
    atomicAdd(volumeInstrumentation.densitySamples, densityCount);
    atomicAdd(volumeInstrumentation.emptySegmentsSkipped, emptyCount);
    atomicAdd(volumeInstrumentation.topologySegmentsSkipped, topologyEmptyCount);
    atomicAdd(volumeInstrumentation.majorantSegmentsSkipped, densityLeafEmptyCount);
    if (outcome == VOLUME_MARCH_EXTINCTION)
        atomicAdd(volumeInstrumentation.extinctionTerminations, 1u);
    else if (outcome == VOLUME_MARCH_STEP_BUDGET)
        atomicAdd(volumeInstrumentation.stepBudgetExhausted, 1u);
    else
        atomicAdd(volumeInstrumentation.completedIntervals, 1u);
}

void volumeRecordShadowSamples(uint count) {
    if (volumeInstrumentationEnabled() && count != 0u) {
        atomicAdd(volumeInstrumentation.shadowDensitySamples, count);
    }
}

void volumeRecordMajorantQuery(bool available) {
    if (!volumeInstrumentationEnabled()) return;
    atomicAdd(volumeInstrumentation.majorantQueries, 1u);
    if (available)
        atomicAdd(volumeInstrumentation.majorantAvailableQueries, 1u);
}

void volumeRecordTemporal(bool accepted) {
    if (!volumeInstrumentationEnabled()) return;
    if (accepted) atomicAdd(volumeInstrumentation.temporalAccepted, 1u);
    else atomicAdd(volumeInstrumentation.temporalRejected, 1u);
}

#endif
