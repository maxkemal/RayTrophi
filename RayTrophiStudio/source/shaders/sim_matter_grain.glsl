// Shared dry DEM stages. Packed xyz and 3-column affine match canonical SoA.
// Bank 0 is the grain runtime's own device copy of the grain-owned carriers
// (identity order); liquid parcels of the same domain never enter it.
//
// One step dispatch per contact substep. State ping-pongs between bank 0 (the
// canonical position/velocity/affine buffers) and bank 1 (grain scratch), so a
// substep reads only the previous substep's complete state and writes only its
// own grain.
// Neighbours: a per-grain Verlet list (docs/dev/DEM_VERLET_LISTESI.md) of the
// grains within the hash cell size (2r + bridge rupture cap + skin). It is
// rebuilt from one fixed-capacity bucket table only when a grain has moved
// skin/2 since the last build: substep k runs list_clear, hash, list_build
// (each a no-op unless flag k is set) and then the step, which raises flag
// k+1. Flags rotate over three diagnostics words: list_clear of substep k
// zeroes word k+2 (last read in k-1, next written in k+1).
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Positions { float positions[]; };
layout(std430, binding = 1) buffer Velocities { float velocities[]; };
layout(std430, binding = 2) buffer Affines { float affines[]; };
layout(std430, binding = 3) readonly buffer Mass { float masses[]; };
layout(std430, binding = 4) readonly buffer Ids { uint ids[]; };
layout(std430, binding = 5) buffer BucketCounts { uint bucket_counts[]; };
layout(std430, binding = 6) buffer BucketSlots { uint bucket_slots[]; };
layout(std430, binding = 7) buffer Scratch { float scratch[]; };
// Contact history, one block of SLOTS records per grain (7 words each:
// key, tangential spring xyz, rolling spring xyz), updated in place.
layout(std430, binding = 8) buffer History { uint history[]; };
// [0, capacity): grain -> history block (FRESH bit: ignore the block's old
// records this frame). [capacity + 4b]: owner id of block b, [+1]: mask of
// its occupied slots, [+2]: rest seconds (float bits, AUDIT bit set by a
// neighbour audit request), [+3]: skipped solve time while asleep, or rest-clock
// rounding compensation while awake (float bits; only the owner reads/writes it).
// The host keeps each grain on its block across frames.
layout(std430, binding = 9) coherent buffer HistoryBlocks { uint history_blocks[]; };
layout(std430, binding = 10) buffer Diagnostics { uint diagnostics[]; };
layout(std430, binding = 11) readonly buffer Triangles { float triangles[]; };
struct ColliderNode { vec3 low; uint first; vec3 high; uint second; };
layout(std430, binding = 12) readonly buffer ColliderNodes { ColliderNode nodes[]; };
layout(std430, binding = 13) readonly buffer SurfacePatches { uint patches[]; };
// Liquid coupling, three vec4 per grain, touched only by the grain's own
// invocation: {lump velocity, drag coefficient beta}, {buoyancy acceleration,
// lump mass}, {accumulated drag impulse, bridge water volume m^3}.
// beta == 0: uncoupled; water 0: dry (no liquid bridge).
layout(std430, binding = 14) buffer Coupling { vec4 coupling[]; };
// Per grain {count, LIST neighbour indices}, and the positions it was built from.
layout(std430, binding = 15) buffer NeighbourList { uint neighbour_list[]; };
layout(std430, binding = 16) buffer BuildPositions { float build_positions[]; };
layout(push_constant) uniform Constants {
    uvec4 meta; // count, buckets per table, twisting-friction float bits, BVH nodes
    vec4 low_radius;
    vec4 high_stiffness;
    vec4 step_contact; // dt, normal damping, sliding damping, friction
    vec4 rolling; // rolling coefficient, gravity xyz
    uvec4 substep; // index, history blocks (capacity), tangential stiffness bits, last index
    // Hash cell size = list cutoff (2r + bridge rupture cap + skin), capillary prefactor
    // 2 pi gamma cos(theta) x cohesion scale, rupture cap (m), and the float
    // offset of the collider vertex velocities in `triangles` (uint bits;
    // 0 = static colliders).
    vec4 wet;
    // Sleeping grains: still substeps before sleep (uint bits), sleep speed
    // m/s (0 = off for this frame), wake all this frame (uint bits), sleep time s.
    vec4 sleep;
} pc;

// Contact budget shared with the host CFL (kMatterGrainContactBudget). Every
// grain, domain-wall and mesh-patch contact of one grain needs a history slot.
const uint SLOTS = 24u;
const float BUDGET = 24.0;
// Shared with the host (kMatterGrainBucketCapacity).
const uint BUCKET = 16u;
const uint EMPTY = 0xffffffffu;
const uint WALL_KEY = 0x80000000u;
const uint PATCH_KEY = 0xc0000000u;
const uint REVISION = 24u;
const uint AUDIT = 0x80000000u;
const uint SLOT_WORDS = 7u;
const uint FRESH = 0x80000000u;
#include "sim_matter_grain_sleep.glsl"
#include "sim_matter_grain_sleep_transfer.glsl"
// Shared with the host (kMatterGrainListCapacity).
const uint LIST = 32u;
// diagnostics words: 12..14 rebuild flags (k%3), 15 builds this frame,
// 16 sleeping grains on the last substep.
uint flagWord(uint k) { return 12u + k % 3u; }
bool rebuildNow() { return diagnostics[flagWord(pc.substep.x)] != 0u; }

uint readBank() { return pc.substep.x & 1u; }
uint writeBank() { return readBank() ^ 1u; }

vec3 position(uint i, uint bank) {
    if (bank == 0u) return vec3(positions[3*i], positions[3*i+1], positions[3*i+2]);
    return vec3(scratch[3*i], scratch[3*i+1], scratch[3*i+2]);
}
vec3 velocity(uint i, uint bank) {
    if (bank == 0u) return vec3(velocities[3*i], velocities[3*i+1], velocities[3*i+2]);
    uint b = 3u*pc.meta.x+3u*i;
    return vec3(scratch[b], scratch[b+1], scratch[b+2]);
}
vec3 omega(uint i, uint bank) {
    if (bank == 0u) {
        uint b = 9*i;
        return .5 * vec3(affines[b+5]-affines[b+7],
            affines[b+6]-affines[b+2], affines[b+1]-affines[b+3]);
    }
    uint b = 6u*pc.meta.x+3u*i;
    return vec3(scratch[b], scratch[b+1], scratch[b+2]);
}
void store(uint i, uint bank, vec3 x, vec3 v, vec3 w) {
    if (bank == 0u) {
        for (uint k=0u; k<3u; ++k) {
            positions[3u*i+k] = x[k];
            velocities[3u*i+k] = v[k];
        }
        uint b = 9u*i;
        affines[b]=0.0; affines[b+1]=w.z; affines[b+2]=-w.y;
        affines[b+3]=-w.z; affines[b+4]=0.0; affines[b+5]=w.x;
        affines[b+6]=w.y; affines[b+7]=-w.x; affines[b+8]=0.0;
        return;
    }
    for (uint k=0u; k<3u; ++k) {
        scratch[3u*i+k] = x[k];
        scratch[3u*pc.meta.x+3u*i+k] = v[k];
        scratch[6u*pc.meta.x+3u*i+k] = w[k];
    }
}

ivec3 cell(vec3 p) {
    return ivec3(floor((p-pc.low_radius.xyz)/pc.wet.x));
}
uint bucket(ivec3 c) {
    uvec3 u = uvec3(c);
    return ((u.x*73856093u) ^ (u.y*19349663u) ^ (u.z*83492791u)) & (pc.meta.y-1u);
}
// A full bucket refuses publication: dropping a neighbour would silently
// delete a contact.
void insert(uint i, vec3 p, uint table) {
    uint b = table*pc.meta.y+bucket(cell(p));
    uint slot = atomicAdd(bucket_counts[b],1u);
    if (slot < BUCKET) bucket_slots[b*BUCKET+slot] = i;
    else atomicOr(diagnostics[0],4u);
}

vec3 closestTriangle(vec3 p, vec3 a, vec3 b, vec3 c) {
    vec3 ab = b-a, ac = c-a, ap = p-a;
    float d1 = dot(ab,ap), d2 = dot(ac,ap);
    if (d1 <= 0.0 && d2 <= 0.0) return a;
    vec3 bp = p-b;
    float d3 = dot(ab,bp), d4 = dot(ac,bp);
    if (d3 >= 0.0 && d4 <= d3) return b;
    float vc = d1*d4-d3*d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return a+ab*(d1/(d1-d3));
    vec3 cp = p-c;
    float d5 = dot(ab,cp), d6 = dot(ac,cp);
    if (d6 >= 0.0 && d5 <= d6) return c;
    float vb = d5*d2-d1*d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return a+ac*(d2/(d2-d6));
    float va = d3*d6-d5*d4;
    if (va <= 0.0 && d4-d3 >= 0.0 && d5-d6 >= 0.0)
        return b+(c-b)*((d4-d3)/((d4-d3)+(d5-d6)));
    return a+(ab*vb+ac*vc)/(va+vb+vc);
}

// Per-grain contact history: the grain's own block only, so no other
// invocation touches it. A key absent from the block starts fresh springs; a
// contact not seen this substep drops out of the occupied mask. In place: a
// found record is rewritten in its own slot after it was read, and a new
// contact takes a slot that was free at the start of the substep, so no
// record is overwritten before every lookup of it is done.
uint g_contacts = 0u;
// Cost counters, summed on the last substep only (diagnostics[7..11]):
// list entries read, grain-grain contacts, history slots read, BVH nodes.
uint g_scanned = 0u;
uint g_pairs = 0u;
uint g_history_probes = 0u;
uint g_nodes = 0u;
uint g_sticking = 0u;
uint g_block = 0u;
uint g_old_mask = 0u;
uint g_touched = 0u;
uint g_allocated = 0u;

uint slotWord(uint s) { return (g_block*SLOTS+s)*SLOT_WORDS; }
uint ownerWord() { return pc.substep.y+4u*g_block; }
// Rest counter word of grain j's block (j's map entry may still carry FRESH).
uint restWordOf(uint j) { return pc.substep.y+4u*(history_blocks[j] & ~FRESH)+2u; }
uint sleepRestLimit() { return floatBitsToUint(max(pc.sleep.w,pc.step_contact.x)); }
bool sleepOn() { return pc.sleep.y > 0.0; }
// This grain is moving; only a meaningful transferred kick requests an audit.
bool g_fast = false;
bool g_probe_sleep_transfer = false;
float g_history_dt = 0.0;
vec3 g_pair_dynamic_impulse = vec3(0.0);
vec3 g_pair_dynamic_angular_impulse = vec3(0.0);
vec3 g_pair_forecast_impulse = vec3(0.0);
vec3 g_pair_forecast_angular_impulse = vec3(0.0);
vec3 g_bridge_forecast_impulse = vec3(0.0);

void requestNeighbourAudit(uint j) {
    // Awake grains evaluate contact motion and force balance themselves. Do not
    // repeatedly erase their rest progress, or write every active contact pair.
    // This flag requests a full solve; it does not declare the target unbalanced.
    // Commit never exposes a temporary zero for an already sleeping owner.
    uint word = restWordOf(j);
    if ((history_blocks[word] & ~AUDIT) >= sleepRestLimit()) {
        atomicOr(history_blocks[word], AUDIT);
    }
}

uint previousSprings(uint key, out vec3 tangential, out vec3 rolling) {
    tangential = vec3(0.0);
    rolling = vec3(0.0);
    for (uint m=g_old_mask; m != 0u; m &= m-1u) {
        uint s = uint(findLSB(m));
        uint b = slotWord(s);
        ++g_history_probes;
        if (history[b] != key) continue;
        tangential = uintBitsToFloat(uvec3(history[b+1u],history[b+2u],history[b+3u]));
        rolling = uintBitsToFloat(uvec3(history[b+4u],history[b+5u],history[b+6u]));
        return s;
    }
    return EMPTY;
}
// More records in flight than SLOTS (old ones still held this substep plus
// new ones) refuses publication like the contact budget itself.
void keepSprings(uint slot, uint key, vec3 tangential, vec3 rolling) {
    if (slot == EMPTY) {
        uint open_slots = ~(g_old_mask|g_allocated) & ((1u<<SLOTS)-1u);
        if (open_slots == 0u) { atomicOr(diagnostics[0],1u); return; }
        slot = uint(findLSB(open_slots));
        g_allocated |= 1u<<slot;
    } else {
        g_touched |= 1u<<slot;
    }
    uint b = slotWord(slot);
    uvec3 t = floatBitsToUint(tangential), r = floatBitsToUint(rolling);
    history[b] = key;
    history[b+1u] = t.x; history[b+2u] = t.y; history[b+3u] = t.z;
    history[b+4u] = r.x; history[b+5u] = r.y; history[b+6u] = r.z;
}
// A stored displacement is carried onto the current tangent plane with its
// magnitude kept (the contact frame rotates between substeps).
vec3 carry(vec3 previous, vec3 n) {
    vec3 projected = previous-dot(previous,n)*n;
    float after = length(projected);
    return after > 1e-12 ? projected*(length(previous)/after) : vec3(0.0);
}

// Pendular liquid bridge between two wet grains, Willett et al. (2000) for
// equal spheres: F = 2 pi R gamma cos(theta) / (1 + 2.1 s + 10 s^2),
// s = S sqrt(R / V), rupture at S = V^(1/3) (capped by the hash skin).
// Each grain shares its water among ~6 bridges (random packing coordination).
// Central force only, equal and opposite by symmetry of the pair.
uint g_bridges = 0u;
void bridge(uint i, uint j, float d, vec3 n, float r, inout vec3 force) {
    float volume = (coupling[3u*i+2u].w+coupling[3u*j+2u].w)/12.0;
    if (volume <= 0.0 || pc.wet.y <= 0.0) return;
    float gap = max(d-2.0*r,0.0);
    if (gap >= min(pow(volume,1.0/3.0),pc.wet.z)) return;
    float s = gap*sqrt(r/volume);
    force -= n*(pc.wet.y*r/(1.0+2.1*s+10.0*s*s));
    ++g_bridges;
    if (g_probe_sleep_transfer) {
        vec3 relative = velocity(i, readBank()) - velocity(j, readBank());
        float inverse_pair_mass = 1.0 / masses[i] + 1.0 / masses[j];
        float ceiling = min(pc.wet.y * r * grainSleepAuditHorizon(),
                            2.0 * length(relative) / inverse_pair_mass);
        g_bridge_forecast_impulse = n * ceiling;
    }
}

void contact(uint i, uint key, vec3 n, float overlap, vec3 arm, vec3 relative,
             vec3 spin, float inv_inertia, float other_inv_inertia,
             float effective_radius, float inverse_tangent_mass,
             float inverse_normal_mass, inout vec3 force, inout vec3 torque) {
    if (overlap <= 0.0) return;
    ++g_contacts;
    float dt = pc.step_contact.x;
    float k = pc.high_stiffness.w;
    float normal_speed = dot(relative,n);
    // step_contact.y is the damping ratio of the material's restitution;
    // c = 2 zeta sqrt(k m_eff) gives every contact the same rebound.
    float cn = 2.0*pc.step_contact.y*sqrt(k/inverse_normal_mass);
    float fn = max(0.0, k*overlap-cn*normal_speed);
    vec3 tangential, rolling;
    uint slot = previousSprings(key, tangential, rolling);
    float history_dt = slot == EMPTY ? dt : g_history_dt;
    vec3 previous_tangent = carry(tangential,n);
    vec3 previous_roll = carry(rolling,n);

    // Sliding: Cundall-Strack spring + viscous part on the Coulomb cone. The
    // aggregate cap keeps a full contact budget from reversing the slip
    // within one substep.
    vec3 slip = relative-normal_speed*n;
    float speed = length(slip);
    vec3 damping = vec3(0.0);
    if (speed > 1e-9) {
        damping = -slip/speed * min(pc.step_contact.z*speed,
            speed/(BUDGET*dt*inverse_tangent_mass));
    }
    float kt = uintBitsToFloat(pc.substep.z);
    tangential = kt > 0.0 ? previous_tangent+slip*history_dt : vec3(0.0);
    vec3 ft = damping-kt*tangential;
    float coulomb = pc.step_contact.w*fn;
    float magnitude = length(ft);
    if (magnitude > coulomb) {
        ft = magnitude > 1e-12 ? ft*(coulomb/magnitude) : vec3(0.0);
        tangential = kt > 0.0 && coulomb > 0.0 ? -(ft-damping)/kt : vec3(0.0);
    } else if (kt > 0.0) {
        ++g_sticking;
    }

    // Rolling: elastic-plastic spring-dashpot (EPSD2, Ai et al. 2011).
    // k_r = 2.25 mu_r^2 k R^2, damping 0.3 of critical, torque capped at
    // mu_r Fn R. A kinetic-only torque vanishes at zero spin, so a grain on a
    // slope crept by rolling; the spring holds it below tan(theta) = mu_r.
    vec3 roll = spin-dot(spin,n)*n;
    float mu_r = pc.rolling.x;
    float inverse_pair_inertia = inv_inertia+other_inv_inertia;
    vec3 rt = vec3(0.0);
    vec3 previous_rolling_torque = vec3(0.0);
    if (mu_r > 0.0) {
        float kr = 2.25*mu_r*mu_r*k*effective_radius*effective_radius;
        float cr = .6*sqrt(kr/inverse_pair_inertia);
        previous_rolling_torque = -kr*previous_roll;
        rolling = previous_roll+roll*history_dt;
        vec3 roll_damping = -cr*roll;
        rt = roll_damping-kr*rolling;
        float limit = mu_r*fn*effective_radius;
        float roll_magnitude = length(rt);
        if (roll_magnitude > limit) {
            rt = roll_magnitude > 1e-12 ? rt*(limit/roll_magnitude) : vec3(0.0);
            rolling = limit > 0.0 ? -(rt-roll_damping)/kr : vec3(0.0);
        }
    } else {
        rolling = vec3(0.0);
    }
    keepSprings(slot, key, tangential, rolling);
    force += n*fn+ft;
    // Finite contact-patch twist resistance. Equal/opposite pair torque;
    // no uniform angular damping, and aggregate impulses cannot reverse spin.
    float twist_speed = dot(spin,n);
    float patch_radius = min(effective_radius,sqrt(max(0.0,effective_radius*overlap)));
    float twist_limit = abs(twist_speed)/(BUDGET*dt*inverse_pair_inertia);
    float twist_torque = min(uintBitsToFloat(pc.meta.z)*fn*patch_radius,twist_limit);
    torque += cross(arm,ft)+rt-sign(twist_speed)*twist_torque*n;
    if (g_probe_sleep_transfer) {
        // Remove the retained elastic preload. Only the changed, post-damping
        // and post-Coulomb reaction is an instantaneous disturbance.
        float previous_normal = slot == EMPTY ? 0.0 :
            max(0.0, k*(overlap+normal_speed*history_dt));
        vec3 changed_tangent = ft+kt*previous_tangent;
        g_pair_dynamic_impulse = -(n*(fn-previous_normal)+changed_tangent)*dt;
        g_pair_dynamic_angular_impulse =
            (cross(arm,changed_tangent)-(rt-previous_rolling_torque)+
             sign(twist_speed)*twist_torque*n)*dt;
        grainSleepContactForecast(n,arm,relative,spin,fn,inverse_normal_mass,
            inverse_tangent_mass,inverse_pair_inertia,effective_radius,patch_radius,
            g_pair_forecast_impulse,g_pair_forecast_angular_impulse);
    }
}

void main() {
    // 2-D dispatch folded to one index: Vulkan guarantees only 65535 groups
    // per axis, the host spills the rest into y (setLinearGroups).
    uint i = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * 256u;
#if defined(GRAIN_LIST_CLEAR)
    // Word k+2 was last read in substep k-1 and is next written in k+1.
    if (i == 0u) diagnostics[flagWord(pc.substep.x+2u)] = 0u;
    if (!rebuildNow()) return;
    if (i < pc.meta.y) bucket_counts[i] = 0u;
#elif defined(GRAIN_HASH)
    if (i < pc.meta.x && rebuildNow()) insert(i, position(i,readBank()), 0u);
#elif defined(GRAIN_LIST_BUILD)
    // Every grain within the cutoff, from its own cell's bucket only (two
    // cells sharing a bucket would otherwise list a neighbour twice).
    if (!rebuildNow()) return;
    if (i == 0u) atomicAdd(diagnostics[15],1u);
    if (i >= pc.meta.x) return;
    uint bank = readBank();
    vec3 p = position(i,bank);
    float cutoff2 = pc.wet.x*pc.wet.x;
    ivec3 own = cell(p);
    uint base = i*(LIST+1u);
    uint listed = 0u;
    for (int z=-1; z<=1; ++z) for (int y=-1; y<=1; ++y) for (int x=-1; x<=1; ++x) {
        ivec3 wanted = own+ivec3(x,y,z);
        uint b = bucket(wanted);
        uint stored = min(bucket_counts[b],BUCKET);
        for (uint s=0u; s<stored; ++s) {
            uint j = bucket_slots[b*BUCKET+s];
            if (j == i) continue;
            vec3 pj = position(j,bank);
            if (!all(equal(cell(pj),wanted))) continue;
            vec3 separation = p-pj;
            if (dot(separation,separation) >= cutoff2) continue;
            if (listed == LIST) { atomicOr(diagnostics[0],8u); continue; }
            neighbour_list[base+1u+listed++] = j;
        }
    }
    neighbour_list[base] = listed;
    for (uint k=0u; k<3u; ++k) build_positions[3u*i+k] = p[k];
#elif defined(GRAIN_STEP)
    if (i >= pc.meta.x) return;
    // Proves the fused-step SPIR-V ran; the host refuses publication otherwise.
    if (i == 0u) diagnostics[1] = REVISION;
    // Reject an old host's three-word metadata before indexing the new layout.
    // The descriptor/push ABI is unchanged, so reflection alone cannot catch it.
    if (uint(history_blocks.length()) < 5u*pc.substep.y) {
        if (i == 0u) atomicOr(diagnostics[0],16u);
        return;
    }
    uint bank = readBank();
    bool valid_block = false;
    // The FRESH bit holds for substep 0 of the frame the host assigned it.
    uint entry = history_blocks[i];
    g_block = entry & ~FRESH;
    if ((entry & FRESH) != 0u) {
        if (pc.substep.x == 0u) history_blocks[i] = g_block;
    } else if (history_blocks[ownerWord()] == ids[i]) {
        g_old_mask = history_blocks[ownerWord()+1u];
        valid_block = true;
    }
    vec3 p = position(i,bank), v = velocity(i,bank), w = omega(i,bank);
    float r = pc.low_radius.w, im = 1.0/masses[i], ii = 2.5*im/(r*r);
    uint dst = writeBank();
    vec4 drag = coupling[3u*i];
    vec4 lift = coupling[3u*i+1u];
    bool can_sleep = drag.w == 0.0 && lift == vec4(0.0);
    bool input_slow = dot(v,v) < pc.sleep.y*pc.sleep.y && length(w)*r < pc.sleep.y;
    // Sleeping (docs/dev/DEM_UYUYAN_TANELER.md): skip the expensive contact
    // solve between audits. The last substep always evaluates actual contacts
    // and bridges, so diagnostics and the next frame's CFL include sleepers.
    uint rest = 0u;
    uint rest_word = ownerWord()+2u;
    uint elapsed_word = ownerWord()+3u;
    bool context_changed = floatBitsToUint(pc.sleep.z) != 0u;
    uint stored_rest = valid_block ? history_blocks[rest_word] & ~AUDIT : 0u;
    bool was_sleeping = valid_block && stored_rest >= sleepRestLimit();
    float clock_auxiliary = valid_block && !context_changed
        ? uintBitsToFloat(history_blocks[elapsed_word]) : 0.0;
    float elapsed = was_sleeping ? clock_auxiliary : 0.0;
    float rest_compensation = !was_sleeping && can_sleep && input_slow
        ? clock_auxiliary : 0.0;
    g_history_dt = elapsed+pc.step_contact.x;
    bool sleeping = false;
    if (sleepOn() && valid_block && can_sleep && input_slow) {
        uint raw = grainSleepTakeAudit(rest_word);
        bool audit_requested = (raw & AUDIT) != 0u;
        rest = context_changed ? 0u : raw & ~AUDIT;
        sleeping = rest >= sleepRestLimit();
        if (sleeping && !audit_requested && !grainSleepAudit()) {
            history_blocks[elapsed_word] = floatBitsToUint(g_history_dt);
            store(i,dst,p,vec3(0.0),vec3(0.0));
            return;
        }
    }
    g_fast = sleepOn() && (dot(v,v) >= pc.sleep.y*pc.sleep.y || length(w)*r >= pc.sleep.y);
    vec3 f = vec3(0.0), t = vec3(0.0);
    uint list_base = i*(LIST+1u);
    uint listed = neighbour_list[list_base];
    g_scanned += listed;
    for (uint s=0u; s<listed; ++s) {
        g_pair_dynamic_impulse = vec3(0.0);
        g_pair_dynamic_angular_impulse = vec3(0.0);
        g_pair_forecast_impulse = vec3(0.0);
        g_pair_forecast_angular_impulse = vec3(0.0);
        g_bridge_forecast_impulse = vec3(0.0);
        uint j = neighbour_list[list_base+1u+s];
        // A slow heavy grain can still move a light neighbour. Do not gate
        // transfer by the source's sleep speed; gate by the recipient's response.
        g_probe_sleep_transfer = sleepOn() &&
            (dot(v,v) > 0.0 || dot(w,w) > 0.0 || !can_sleep) &&
            (history_blocks[restWordOf(j)] & ~AUDIT) >= sleepRestLimit();
        vec3 separation = p-position(j,bank);
        float d = length(separation);
        uint bridges_before = g_bridges;
        if (d < 2.0*r+pc.wet.z && d > 1e-9) bridge(i,j,d,separation/d,r,f);
        bool linked = g_bridges != bridges_before;
        if (d < 2.0*r) {
            linked = true;
            ++g_pairs;
            vec3 n = d > 1e-9 ? separation/d : vec3(i<j ? -1.0 : 1.0,0,0);
            vec3 arm = -n*(.5*d);
            vec3 wj = omega(j,bank);
            float jm = 1.0/masses[j], ji = 2.5*jm/(r*r);
            vec3 relative = v+cross(w,arm)-velocity(j,bank)-cross(wj,-arm);
            contact(i,ids[j],n,2.0*r-d,arm,relative,w-wj,ii,ji,.5*r,
                im+jm+dot(arm,arm)*(ii+ji),im+jm,f,t);
        }
        if (g_probe_sleep_transfer && linked) {
            float jm = 1.0/masses[j], ji = 2.5*jm/(r*r);
            bool significant = grainSleepKickSignificant(g_pair_dynamic_impulse,
                g_pair_dynamic_angular_impulse,jm,ji,r) ||
                grainSleepKickSignificant(g_pair_forecast_impulse,
                    g_pair_forecast_angular_impulse,jm,ji,r) ||
                grainSleepKickSignificant(g_bridge_forecast_impulse,vec3(0.0),jm,ji,r);
            if (significant || !can_sleep) {
                requestNeighbourAudit(j);
            }
        }
    }
    g_probe_sleep_transfer = false;
    // Closed domain walls. Static collider reaction belongs to the support.
    for (int axis=0; axis<3; ++axis) for (int side=0; side<2; ++side) {
        vec3 n = vec3(0.0); n[axis] = side == 0 ? 1.0 : -1.0;
        float distance = side == 0 ? p[axis]-pc.low_radius[axis] : pc.high_stiffness[axis]-p[axis];
        vec3 arm = -n*r;
        contact(i,WALL_KEY|uint(2*axis+side),n,r-distance,arm,v+cross(w,arm),w,ii,0.0,r,
            im+r*r*ii,im,f,t);
    }
    // Balanced BVH, up to four independent support features. Connected
    // coplanar triangles share one patch; a common edge/vertex is not doubled.
    vec3 manifold_n[4];
    float manifold_d[4];
    uint manifold_key[4];
    vec3 manifold_v[4];
    vec3 patch_n[8];
    float patch_d[8];
    uint patch_id[8];
    vec3 patch_v[8];
    // Moving colliders: a face sits at end - v * (frame time left), so it
    // sweeps through the substeps; the wall velocity at the contact point
    // enters the relative velocity (friction drags grains along, a foot
    // pushes them away). Static scenes read no velocity.
    uint velocity_offset = floatBitsToUint(pc.wet.w);
    float time_left = float(pc.substep.w+1u-pc.substep.x)*pc.step_contact.x;
    uint patch_count = 0u;
    uint manifold_count = 0u;
    uint stack[64];
    uint size = 0u;
    if (pc.meta.w > 0u) stack[size++] = 0u;
    while (size > 0u) {
        ColliderNode node = nodes[stack[--size]];
        ++g_nodes;
        vec3 gap = max(max(node.low-p,p-node.high),vec3(0.0));
        if (dot(gap,gap) >= r*r) continue;
        if ((node.first & 0x80000000u) == 0u) {
            if (size+2u > 64u) {
                atomicOr(diagnostics[0],2u);
                break;
            }
            stack[size++] = node.second;
            stack[size++] = node.first;
            continue;
        }
        uint start = node.first & 0x7fffffffu;
        for (uint k=start; k<start+node.second; ++k) {
            uint b = 9u*k;
            vec3 a = vec3(triangles[b],triangles[b+1],triangles[b+2]);
            vec3 bp = vec3(triangles[b+3],triangles[b+4],triangles[b+5]);
            vec3 c = vec3(triangles[b+6],triangles[b+7],triangles[b+8]);
            vec3 va = vec3(0.0), vb = vec3(0.0), vc = vec3(0.0);
            if (velocity_offset != 0u) {
                uint e = velocity_offset+b;
                va = vec3(triangles[e],triangles[e+1],triangles[e+2]);
                vb = vec3(triangles[e+3],triangles[e+4],triangles[e+5]);
                vc = vec3(triangles[e+6],triangles[e+7],triangles[e+8]);
                a -= va*time_left; bp -= vb*time_left; c -= vc*time_left;
            }
            vec3 q = closestTriangle(p,a,bp,c);
            float d = length(p-q);
            if (d >= r) continue;
            vec3 n = d > 1e-9 ? (p-q)/d : normalize(cross(bp-a,c-a));
            vec3 wall = vec3(0.0);
            if (velocity_offset != 0u) {
                // Barycentric weights of the contact point.
                vec3 e0 = bp-a, e1 = c-a, e2 = q-a;
                float d00 = dot(e0,e0), d01 = dot(e0,e1), d11 = dot(e1,e1);
                float d20 = dot(e2,e0), d21 = dot(e2,e1);
                float den = d00*d11-d01*d01;
                float wb = den > 1e-20 ? (d11*d20-d01*d21)/den : 0.0;
                float wc = den > 1e-20 ? (d00*d21-d01*d20)/den : 0.0;
                wall = (1.0-wb-wc)*va+wb*vb+wc*vc;
            }
            // Adjacent faces of a tessellated curve (each its own patch) reach
            // the grain through the same shared edge/vertex: one feature, one
            // slot. A sphere pole fans 24 faces into one point.
            bool same_feature = false;
            for (uint m=0u; m<patch_count; ++m) {
                if (patch_id[m] != patches[k] && dot(patch_n[m],n) > .99999 &&
                    abs(patch_d[m]-d) < r*1e-4) { same_feature = true; break; }
            }
            if (same_feature) continue;
            uint slot = patch_count;
            for (uint m=0u; m<patch_count; ++m) {
                if (patch_id[m] == patches[k]) { slot = m; break; }
            }
            if (slot == patch_count && patch_count == 8u) {
                // Full: keep the eight deepest features (counted, not fatal).
                uint far = 0u;
                for (uint m=1u; m<8u; ++m) if (patch_d[m] > patch_d[far]) far = m;
                atomicAdd(diagnostics[6],1u);
                if (d >= patch_d[far]) continue;
                slot = far;
                patch_n[slot] = n;
                patch_d[slot] = d;
                patch_id[slot] = patches[k];
                patch_v[slot] = wall;
            } else if (slot == patch_count) {
                ++patch_count;
                patch_n[slot] = n;
                patch_d[slot] = d;
                patch_id[slot] = patches[k];
                patch_v[slot] = wall;
            } else if (d < patch_d[slot]) {
                patch_n[slot] = n;
                patch_d[slot] = d;
                patch_v[slot] = wall;
            }
        }
    }
    // Deduplicate common edges/vertices after each patch's closest point
    // is known; replacing a candidate cannot lose a patch's membership.
    for (uint k=0u; k<patch_count; ++k) {
        bool duplicate = false;
        for (uint m=0u; m<manifold_count; ++m) {
            if (dot(manifold_n[m],patch_n[k]) > .99999 &&
                abs(manifold_d[m]-patch_d[k]) < r*1e-4) { duplicate = true; break; }
        }
        if (duplicate) continue;
        if (manifold_count == 4u) {
            // Keep the four deepest supports (counted, not fatal).
            uint far = 0u;
            for (uint m=1u; m<4u; ++m) if (manifold_d[m] > manifold_d[far]) far = m;
            atomicAdd(diagnostics[6],1u);
            if (patch_d[k] >= manifold_d[far]) continue;
            manifold_n[far] = patch_n[k];
            manifold_d[far] = patch_d[k];
            manifold_v[far] = patch_v[k];
            manifold_key[far] = PATCH_KEY|(patch_id[k] & 0x3fffffffu);
            continue;
        }
        manifold_n[manifold_count] = patch_n[k];
        manifold_d[manifold_count] = patch_d[k];
        manifold_v[manifold_count] = patch_v[k];
        manifold_key[manifold_count++] = PATCH_KEY|(patch_id[k] & 0x3fffffffu);
    }
    for (uint m=0u; m<manifold_count; ++m) {
        vec3 n = manifold_n[m], arm = -n*r;
        contact(i,manifold_key[m],n,r-manifold_d[m],arm,v+cross(w,arm)-manifold_v[m],w,ii,0.0,r,
            im+r*r*ii,im,f,t);
    }
    if (g_contacts > SLOTS) atomicOr(diagnostics[0],1u);
    history_blocks[ownerWord()] = ids[i];
    history_blocks[ownerWord()+1u] = g_touched|g_allocated;
    atomicMax(diagnostics[2],g_contacts);
    if (pc.substep.x == pc.substep.w) {
        atomicAdd(diagnostics[3],g_sticking);
        atomicAdd(diagnostics[4],g_contacts);
        atomicAdd(diagnostics[5],g_bridges);
        atomicAdd(diagnostics[7],g_scanned);
        atomicAdd(diagnostics[8],g_pairs);
        atomicAdd(diagnostics[9],g_history_probes);
        atomicAdd(diagnostics[10],g_contacts == 0u ? 1u : 0u);
        atomicAdd(diagnostics[11],g_nodes);
    }
    // Symplectic Euler from the complete previous-substep state.
    float dt = pc.step_contact.x;
    vec3 acceleration = f*im+pc.rolling.yzw+lift.xyz;
    bool force_balanced = sleepOn() &&
        grainSleepBalanced(acceleration,t*ii,r,p,im,g_contacts);
    bool balanced = can_sleep && force_balanced;
    if (sleeping && balanced) {
        history_blocks[elapsed_word] = 0u;
        store(i,dst,p,vec3(0.0),vec3(0.0));
        grainSleepCommit(rest_word, sleepRestLimit());
        if (pc.substep.x == pc.substep.w) {
            atomicAdd(diagnostics[16],1u);
        }
        return;
    }
    if (sleeping) {
        rest = 0u;
    }
    if (sleepOn() && !force_balanced && !g_fast) {
        // Newly unbalanced supports propagate wake even before they acquire
        // speed. Include bridge reach, since cohesion may link separated grains.
        float reach = 2.0*r+pc.wet.z;
        for (uint s=0u; s<listed; ++s) {
            uint j = neighbour_list[list_base+1u+s];
            vec3 separation = p-position(j,bank);
            if (dot(separation,separation) < reach*reach) {
                requestNeighbourAudit(j);
            }
        }
    }
    v += dt*acceleration;
    if (drag.w > 0.0 && lift.w > 0.0) {
        // Implicit drag pair: grain (m) and its private liquid lump (M).
        // The relative velocity decays by 1/(1 + dt beta (1/m + 1/M)) and
        // m v + M u is unchanged, so no stiffness bound and no overshoot.
        float m = 1.0/im, M = lift.w;
        vec3 relative = (drag.xyz-v)/(1.0+dt*drag.w*(im+1.0/M));
        vec3 settled = (m*v+M*(drag.xyz-relative))/(m+M);
        coupling[3u*i] = vec4(settled+relative, drag.w);
        coupling[3u*i+2u].xyz += m*(settled-v);
        v = settled;
    }
    p += dt*v;
    w += dt*t*ii;
    store(i,dst,p,v,w);
    // Still this substep: accumulate physical rest time. Liquid drag and force fields
    // (lift row) keep a grain awake: they change between frames. A AUDIT set by
    // a neighbour since the read above survives the rewrite.
    bool still = sleepOn() && balanced &&
        dot(v,v) < pc.sleep.y*pc.sleep.y && length(w)*r < pc.sleep.y;
    uint next_rest = 0u;
    float next_compensation = 0.0;
    if (still) {
        // Compensated addition keeps a long rest time progressing even when
        // an adaptive dt is smaller than the float ULP of the accumulated age.
        float previous_time = uintBitsToFloat(rest);
        float increment = dt-rest_compensation;
        float advanced_time = previous_time+increment;
        float target_time = max(pc.sleep.w,dt);
        next_rest = floatBitsToUint(min(advanced_time,target_time));
        if (advanced_time < target_time) {
            next_compensation = (advanced_time-previous_time)-increment;
        }
    }
    history_blocks[elapsed_word] = floatBitsToUint(next_compensation);
    grainSleepCommit(rest_word, next_rest);
    // Rebuild the lists before substep k+1 once this grain has moved half the
    // skin since the build: two grains then closed at most the whole skin, so
    // every pair inside 2r + rupture cap was inside the cutoff at the build.
    float half_skin = .5*(pc.wet.x-2.0*r-pc.wet.z);
    vec3 moved = p-vec3(build_positions[3u*i],build_positions[3u*i+1u],build_positions[3u*i+2u]);
    uint next_flag = flagWord(pc.substep.x+1u);
    if (dot(moved,moved) > half_skin*half_skin && diagnostics[next_flag] == 0u)
        atomicOr(diagnostics[next_flag],1u);
#endif
}
