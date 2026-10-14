// Updraft turrets of a storm cell (docs/dev/ATMOSPHERE_STORMS.md §3.1,
// ATMOSPHERE_CLOUD_VOLUMES.md §3.3). ONE definition shared by the cell map
// (cloud_cells.comp: dome tops, updraft weight) and the cell volume generator
// (cloud_volume_gen.comp: thermal bubble chains), so both see the same towers.
//
// A mature cell is not one uniform column but a few updrafts, each born,
// rising to its own height and collapsing on its own cycle (~15 min), new ones
// replacing them on the inflow side. Pure function of (id, turret, age): the
// same frame twice gives the same turrets.

#ifndef CLOUD_TURRETS_GLSL
#define CLOUD_TURRETS_GLSL

const int kTurrets = 5;

uint cellHash(uint v) {
    v ^= v >> 16u;
    v *= 0x7feb352du;
    v ^= v >> 15u;
    v *= 0x846ca68bu;
    return v ^ (v >> 16u);
}

float cellRandom(inout uint s) {
    s = cellHash(s + 0x9e3779b9u);
    return float(s & 0x00ffffffu) / 16777216.0;
}

struct CloudTurret {
    vec2 centre;      // world (or box-local) x, z
    float radius;     // m
    float height;     // 0.7..1: share of the cell's top this turret reaches
    float rise;       // growth of this generation 0..1 (collapses late)
    float strength;   // updraft strength 0..1
    float u;          // phase within the generation 0..1
    uint seed;        // stream left after the shape draws (per generation)
};

// `cellXZ` cell centre, `windDir` unit forward-flank direction, `spinAngle`
// the ring's turn (rotation x age).
CloudTurret cloudTurret(uint id, int t, float age, float r, vec2 cellXZ, vec2 windDir,
                        float spinAngle) {
    CloudTurret o;
    uint s0 = cellHash(id * 7919u + uint(t) * 104729u + 17u);
    float period = 900.0 * (0.75 + 0.5 * cellRandom(s0));
    float cycle = age / period + float(t) / float(kTurrets) + cellRandom(s0);
    float generation = floor(cycle);
    o.u = cycle - generation;
    uint s = cellHash(s0 ^ (uint(generation) * 2654435761u));
    float angle = cellRandom(s) * 6.28318530718 + spinAngle;
    float ring = r * (0.06 + 0.42 * sqrt(cellRandom(s)));
    o.centre = cellXZ + vec2(cos(angle), sin(angle)) * ring - windDir * 0.12 * r;
    // A turret is a DOME (sqrt profile), kilometres wide: a linear fall-off
    // across a narrow turret drew needles and cones.
    o.radius = r * (0.3 + 0.18 * cellRandom(s));
    o.height = mix(0.7, 1.0, cellRandom(s));
    o.rise = smoothstep(0.0, 0.35, o.u) * (1.0 - 0.45 * smoothstep(0.6, 1.0, o.u));
    o.strength = smoothstep(0.0, 0.12, o.u) * (1.0 - smoothstep(0.65, 1.0, o.u));
    o.seed = s;
    return o;
}

#endif
