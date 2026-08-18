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
    uint solidProbeRuns;
    uint solidProbeHits;
    uint enabled;
    // ── Handoff accounting (black-band / cost tripwire) ──────────────────────
    // A gas segment can leave its box FOUR ways, and on screen three of them can
    // look identical. These separate them:
    //   gasHandoff    - found a solid, handed the ray over (progress)
    //   layeredHandoff- the arbiter found a coincident liquid surface (progress)
    //   arbiterReject - a liquid box overlapped but could not be sampled here
    //   teleport      - nothing found, ray jumped to tFar PAST everything inside
    // A black band with teleport >> 0 means the liquid is being jumped over. The
    // same band with gasHandoff/layeredHandoff >> volumeRays means the ray is
    // ping-ponging through raygen's free-pass budget, which is a COST bug that
    // also reads as black once the budget runs out before any light is reached.
    uint gasHandoffs;
    uint layeredHandoffs;
    uint arbiterRejects;
    uint teleports;
    // ★★★ How many SurfaceSDF candidates the arbiter actually SAW (passed
    // is_active && source_type == 4). This is the one number that separates the
    // two remaining explanations for the black band:
    //   > 0  the liquid IS visible to the arbiter and the fault is downstream
    //   = 0  the liquid is NOT in the arbiter's reach at all -- its slot sits
    //        beyond volCount (cam.pad0), so `candidateIndex < count` never gets
    //        to it. That is the customIndex/count contract, not the mask logic,
    //        and the VolumeSSBO/VolumeGuard tripwires already watch for it.
    uint arbiterCandidates;
    // Times the layered-surface GATE opened at all, i.e. this volume
    // decided it is NOT itself a liquid/cloud and went looking for one.
    //
    // Pairs with arbiterCandidates to separate three outcomes that all
    // look like the same black band on screen:
    //   gate == 0                  the GAS volume believes it IS a liquid
    //                              (source_type read as 4) -- the shader is
    //                              reading another volume's record, i.e. a
    //                              TLAS customIndex / SSBO order mismatch
    //   gate > 0, candidates == 0  gate opened but no liquid was reachable
    //                              within volCount slots
    //   candidates > 0             the arbiter saw it; the fault is later
    uint arbiterGateOpen;
    // ★★★ The three SILENT exits of nearestSurfaceSDFCrossing.
    //
    // arbiterCandidates is incremented BEFORE all three tests, so it reads 100%
    // no matter which one fails — it cannot diagnose anything on its own. These
    // partition the failure exactly once per candidate:
    //   noBox       volumeRayInterval said the ray misses the liquid's AABB.
    //               The liquid's box transform / aabb_min-max is wrong, or it is
    //               genuinely elsewhere.
    //   emptyRange  the box was hit but the usable span collapsed after clipping
    //               to the GAS interval (endT <= beginT). The two boxes do not
    //               overlap along this ray even though both contain it.
    //   noCrossing  the field WAS marched and never crossed ISO=0.5. Sampling or
    //               threshold semantics, not geometry.
    //
    // found = arbiterGateOpen - noBox - emptyRange - noCrossing.
    uint arbiterNoBox;
    uint arbiterEmptyRange;
    uint arbiterNoCrossing;
} volumeInstrumentation;

const uint VOLUME_MARCH_COMPLETED = 0u;
const uint VOLUME_MARCH_EXTINCTION = 1u;
const uint VOLUME_MARCH_STEP_BUDGET = 2u;

bool volumeInstrumentationEnabled() {
    return volumeInstrumentation.enabled != 0u;
}

// Embedded-solid probe accounting. A surface standing INSIDE an active volume
// box is only shaded because this probe finds it: on a hit the march is clamped
// to the surface and the ray is handed to the triangle closesthit, and on a miss
// the ray is advanced to the box exit instead — straight past that surface, into
// whatever lies behind it. The two failures look identical on screen, so they
// need separate counters: runs==0 means the gate suppressed the probe, runs>0
// with hits==0 means the probe ran and did not see the geometry.
// Occupies reserved2/reserved3, so the buffer layout is unchanged.
void volumeRecordSolidProbe(bool foundSolid) {
    if (!volumeInstrumentationEnabled()) return;
    atomicAdd(volumeInstrumentation.solidProbeRuns, 1u);
    if (foundSolid) atomicAdd(volumeInstrumentation.solidProbeHits, 1u);
}

void volumeRecordGasHandoff()     { if (volumeInstrumentationEnabled()) atomicAdd(volumeInstrumentation.gasHandoffs, 1u); }
void volumeRecordLayeredHandoff() { if (volumeInstrumentationEnabled()) atomicAdd(volumeInstrumentation.layeredHandoffs, 1u); }
void volumeRecordArbiterReject()  { if (volumeInstrumentationEnabled()) atomicAdd(volumeInstrumentation.arbiterRejects, 1u); }
void volumeRecordTeleport()       { if (volumeInstrumentationEnabled()) atomicAdd(volumeInstrumentation.teleports, 1u); }
void volumeRecordArbiterCandidate(){ if (volumeInstrumentationEnabled()) atomicAdd(volumeInstrumentation.arbiterCandidates, 1u); }
void volumeRecordArbiterGateOpen(){ if (volumeInstrumentationEnabled()) atomicAdd(volumeInstrumentation.arbiterGateOpen, 1u); }
void volumeRecordArbiterNoBox()      { if (volumeInstrumentationEnabled()) atomicAdd(volumeInstrumentation.arbiterNoBox, 1u); }
void volumeRecordArbiterEmptyRange() { if (volumeInstrumentationEnabled()) atomicAdd(volumeInstrumentation.arbiterEmptyRange, 1u); }
void volumeRecordArbiterNoCrossing() { if (volumeInstrumentationEnabled()) atomicAdd(volumeInstrumentation.arbiterNoCrossing, 1u); }

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
