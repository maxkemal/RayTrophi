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
    // ── Pixel region (normalized launch coordinates, [min, max)) ─────────────
    // Every counter below AND above only counts launches inside this rectangle.
    // Whole-image totals drown a 2% dark patch in background rays; a region
    // turns "what happens on THAT surface" into a measurement. Host writes
    // 0,0,1,1 when no region was asked for.
    float regionMinX;
    float regionMinY;
    float regionMaxX;
    float regionMaxY;
    // ── Path budget accounting (raygen) ──────────────────────────────────────
    // Why a path ENDED, and which bounce kind SPENT the budget. A surface that
    // renders black because its paths die on the bounce cap looks exactly like
    // one that is genuinely unlit; these separate the two.
    //   pathsBounceCapped - loop left with the path still scattering because
    //                       `bounce` reached maxBounces
    //   pathsPassCapped   - loop left with the path still scattering because
    //                       totalPasses reached maxBounces + 32 (free passes
    //                       ran out before the bounce budget did)
    // charged* partition every `bounce++` by the payload's bounceType;
    // freePasses counts traces that were NOT charged (transparent passes and
    // volume handoffs). Specular includes the gas march continuation, whose
    // bounceType is left at its default.
    uint pathsTraced;
    uint pathsBounceCapped;
    uint pathsPassCapped;
    uint chargedSpecular;
    uint chargedDiffuse;
    uint chargedTransmission;
    uint chargedOther;
    uint freePasses;
    // ── Volume-side attribution (volume closest-hit) ─────────────────────────
    //   mediumPasses         - straight gas/fog continuations (BOUNCE_MEDIUM_PASS).
    //                          FREE in the bounce budget, counted in freePasses.
    //                          ~1 per box crossed is healthy; many per path is
    //                          short-hop re-entry. (Was gasSegmentsCharged when
    //                          these still cost a bounce; renamed, not reused.)
    //   arbiterStartedInside - nearestSurfaceSDFCrossing began its walk already
    //                          inside the liquid (d0 > ISO): the ray came from
    //                          a refraction and is looking for the EXIT.
    //   arbiterInsideFound   - ...and found that exit.
    uint mediumPasses;
    uint arbiterStartedInside;
    uint arbiterInsideFound;
    // Random walk (volume_closesthit volumeRandomWalk). Separates the walk's
    // per-event cost: samples (densitySamples / shadowDensitySamples) versus
    // the two traceRayEXT calls an event can make.
    //   walkPaths          - walks that scattered at least once
    //   walkEvents         - scattering events
    //   walkProbeTraces    - solid probes after a flight (one per event)
    //   walkShadowTraces   - geometry shadow rays toward the light (NEE)
    //   walkShadowSkipped  - NEE whose in-volume transmittance was ~0: no trace
    //   walkEventCapped    - walks cut by random_walk_max_events
    uint walkPaths;
    uint walkEvents;
    uint walkProbeTraces;
    uint walkShadowTraces;
    uint walkShadowSkipped;
    uint walkEventCapped;
} volumeInstrumentation;

const uint VOLUME_MARCH_COMPLETED = 0u;
const uint VOLUME_MARCH_EXTINCTION = 1u;
const uint VOLUME_MARCH_STEP_BUDGET = 2u;

bool volumeInstrumentationEnabled() {
    if (volumeInstrumentation.enabled == 0u) return false;
    vec2 p = (vec2(gl_LaunchIDEXT.xy) + 0.5) / vec2(gl_LaunchSizeEXT.xy);
    return p.x >= volumeInstrumentation.regionMinX &&
           p.y >= volumeInstrumentation.regionMinY &&
           p.x <  volumeInstrumentation.regionMaxX &&
           p.y <  volumeInstrumentation.regionMaxY;
}

// Path budget accounting, called from raygen only.
void volumeRecordPathEnd(bool bounceCapped, bool passCapped) {
    if (!volumeInstrumentationEnabled()) return;
    atomicAdd(volumeInstrumentation.pathsTraced, 1u);
    if (bounceCapped) atomicAdd(volumeInstrumentation.pathsBounceCapped, 1u);
    if (passCapped) atomicAdd(volumeInstrumentation.pathsPassCapped, 1u);
}
// kind: 0 specular, 1 diffuse, 2 transmission (incl. glass reflect), 3 other,
// 4 free (not charged).
void volumeRecordPass(uint kind) {
    if (!volumeInstrumentationEnabled()) return;
    if (kind == 0u)      atomicAdd(volumeInstrumentation.chargedSpecular, 1u);
    else if (kind == 1u) atomicAdd(volumeInstrumentation.chargedDiffuse, 1u);
    else if (kind == 2u) atomicAdd(volumeInstrumentation.chargedTransmission, 1u);
    else if (kind == 3u) atomicAdd(volumeInstrumentation.chargedOther, 1u);
    else                 atomicAdd(volumeInstrumentation.freePasses, 1u);
}
void volumeRecordWalk(uint events, uint probeTraces, uint shadowTraces, uint shadowSkipped,
                      bool capped) {
    if (!volumeInstrumentationEnabled()) return;
    atomicAdd(volumeInstrumentation.walkPaths, 1u);
    atomicAdd(volumeInstrumentation.walkEvents, events);
    atomicAdd(volumeInstrumentation.walkProbeTraces, probeTraces);
    atomicAdd(volumeInstrumentation.walkShadowTraces, shadowTraces);
    atomicAdd(volumeInstrumentation.walkShadowSkipped, shadowSkipped);
    if (capped) atomicAdd(volumeInstrumentation.walkEventCapped, 1u);
}
void volumeRecordMediumPass() { if (volumeInstrumentationEnabled()) atomicAdd(volumeInstrumentation.mediumPasses, 1u); }
void volumeRecordArbiterStartedInside(bool found) {
    if (!volumeInstrumentationEnabled()) return;
    atomicAdd(volumeInstrumentation.arbiterStartedInside, 1u);
    if (found) atomicAdd(volumeInstrumentation.arbiterInsideFound, 1u);
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
