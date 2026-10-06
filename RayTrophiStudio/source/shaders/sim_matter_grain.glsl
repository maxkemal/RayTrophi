// Shared dry DEM stages. Packed xyz and 3-column affine match canonical SoA.
// Bank 0 is the grain runtime's own device copy of the grain-owned carriers
// (identity order); liquid parcels of the same domain never enter it.
//
// One dispatch per contact substep. State ping-pongs between bank 0 (the
// canonical position/velocity/affine buffers) and bank 1 (grain scratch), so a
// substep reads only the previous substep's complete state and writes only its
// own grain: contact, integration and the next hash insert fuse safely.
// Neighbours live in fixed-capacity hash buckets (count + BUCKET slots), not
// linked lists: a lookup is a few independent loads instead of a chain of
// dependent ones. Three bucket tables rotate per substep k: read k%3, insert
// (k+1)%3, clear (k+2)%3 -- the cleared table was last read in substep k-1 and
// is next written in k+1, so the backend barrier orders both.
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Positions { float positions[]; };
layout(std430, binding = 1) buffer Velocities { float velocities[]; };
layout(std430, binding = 2) buffer Affines { float affines[]; };
layout(std430, binding = 3) readonly buffer Mass { float masses[]; };
layout(std430, binding = 4) readonly buffer Ids { uint ids[]; };
layout(std430, binding = 5) buffer BucketCounts { uint bucket_counts[]; };
layout(std430, binding = 6) buffer BucketSlots { uint bucket_slots[]; };
layout(std430, binding = 7) buffer Scratch { float scratch[]; };
layout(std430, binding = 8) buffer History { uvec4 history[]; };
layout(std430, binding = 9) buffer HistoryOwner { uint history_owner[]; };
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
layout(push_constant) uniform Constants {
    uvec4 meta; // count, buckets per table, twisting-friction float bits, BVH nodes
    vec4 low_radius;
    vec4 high_stiffness;
    vec4 step_contact; // dt, normal damping, sliding damping, friction
    vec4 rolling; // rolling coefficient, gravity xyz
    uvec4 substep; // index, reset history, tangential stiffness bits, last index
    // Hash cell size (2r + skin: bridge rupture cap, XPBD prediction margin),
    // capillary prefactor 2 pi gamma cos(theta) x cohesion scale, rupture cap
    // (m), solver kind (0 DEM, 1 XPBD).
    vec4 wet;
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
const uint REVISION = 12u;

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

// Per-grain contact history: own slots only, read from one bank and written to
// the other, so no other invocation touches them. A key absent from the
// previous substep starts fresh springs; a vanished contact is dropped.
// Slot = two uvec4: {key, tangential spring xyz}, {rolling spring xyz, -}.
uint g_written = 0u;
uint g_contacts = 0u;
uint g_sticking = 0u;
bool g_history_valid = false;

uint slotBase(uint bank, uint i) { return (bank*pc.meta.x+i)*SLOTS*2u; }

void previousSprings(uint i, uint key, out vec3 tangential, out vec3 rolling) {
    tangential = vec3(0.0);
    rolling = vec3(0.0);
    if (!g_history_valid) return;
    uint base = slotBase(readBank(), i);
    for (uint s=0u; s<SLOTS; ++s) {
        uvec4 head = history[base+2u*s];
        if (head.x == EMPTY) return;
        if (head.x == key) {
            tangential = uintBitsToFloat(head.yzw);
            rolling = uintBitsToFloat(history[base+2u*s+1u].xyz);
            return;
        }
    }
}
void keepSprings(uint i, uint key, vec3 tangential, vec3 rolling) {
    if (g_written == SLOTS) return;
    uint slot = slotBase(writeBank(), i)+2u*g_written++;
    history[slot] = uvec4(key, floatBitsToUint(tangential));
    history[slot+1u] = uvec4(floatBitsToUint(rolling), 0u);
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
}

// ---- XPBD candidate (grain_solver_kind = xpbd, H1-G0 comparison) --------
// Small-steps XPBD (Macklin et al. 2019): one Jacobi projection per substep
// on predicted positions, compliance alpha = 1/k (same material stiffness as
// the DEM), positional Coulomb friction (static while the tangential
// correction fits inside mu * normal correction) acting through the contact
// arm, so friction spins grains as in the DEM. Normal contact is fully
// inelastic; rolling resistance caps the spin change by mu_r * lambda * R.
// Neighbour predictions use gravity + buoyancy only (their drag lump is
// being written concurrently).
vec3 g_dx = vec3(0.0), g_dtheta = vec3(0.0);
float g_normal_sum = 0.0, g_alpha = 0.0;
uint g_constraints = 0u;

vec3 predicted(uint j, uint bank) {
    float dt = pc.step_contact.x;
    return position(j,bank)+dt*(velocity(j,bank)+dt*(pc.rolling.yzw+coupling[3u*j+1u].xyz));
}

void xpbdConstraint(vec3 n, float penetration, vec3 motion, float wi, float wj,
                    vec3 arm, float ii) {
    if (penetration <= 0.0) return;
    ++g_contacts;
    ++g_constraints;
    float lambda = penetration/(wi+wj+g_alpha);
    g_dx += wi*lambda*n;
    g_normal_sum += lambda;
    vec3 slip = motion-dot(motion,n)*n;
    float s = length(slip);
    if (s <= 1e-12) return;
    // Friction as translation while the rolling resistance can carry its
    // torque: the full-stop translational correction, capped by the sliding
    // cone, is F; if F <= mu_r * lambda the contact neither rolls (torque
    // F r fits inside mu_r lambda r) nor needs rotation, so the grain sticks
    // (F below the cone) or slides without spinning. Routing it through the
    // 3.5/m rolling inverse mass instead stopped only the contact point; the
    // spin cap then removed the spin but not the centre's motion, and a grain
    // held on a slope crept at 2.5 h g sin(theta) forever (live: 3.3 mm/s).
    float translate = wi+wj;
    float cone = pc.step_contact.w*lambda;
    float force = min(s/translate, cone);
    if (force <= pc.rolling.x*lambda) {
        if (s/translate <= cone) ++g_sticking;
        g_dx -= wi*slip*(force/s);
        return;
    }
    // Rolls: tangential generalized inverse mass of a sphere 1/m + r^2/I.
    float wt = 3.5*(wi+wj);
    float limit = cone;
    vec3 correction = s/wt <= limit ? slip/wt : slip*(limit/s);
    if (s/wt <= limit) ++g_sticking;
    g_dx -= wi*correction;
    g_dtheta -= ii*cross(arm,correction);
}

void contact(uint i, uint key, vec3 n, float overlap, vec3 arm, vec3 relative,
             vec3 spin, float inv_inertia, float other_inv_inertia,
             float effective_radius, float inverse_tangent_mass,
             inout vec3 force, inout vec3 torque) {
    if (overlap <= 0.0) return;
    ++g_contacts;
    float dt = pc.step_contact.x;
    float k = pc.high_stiffness.w;
    float normal_speed = dot(relative,n);
    float fn = max(0.0, k*overlap-pc.step_contact.y*normal_speed);
    vec3 tangential, rolling;
    previousSprings(i, key, tangential, rolling);

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
    tangential = kt > 0.0 ? carry(tangential,n)+slip*dt : vec3(0.0);
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
    if (mu_r > 0.0) {
        float kr = 2.25*mu_r*mu_r*k*effective_radius*effective_radius;
        float cr = .6*sqrt(kr/inverse_pair_inertia);
        rolling = carry(rolling,n)+roll*dt;
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
    keepSprings(i, key, tangential, rolling);
    force += n*fn+ft;
    // Finite contact-patch twist resistance. Equal/opposite pair torque;
    // no uniform angular damping, and aggregate impulses cannot reverse spin.
    float twist_speed = dot(spin,n);
    float patch_radius = min(effective_radius,sqrt(max(0.0,effective_radius*overlap)));
    float twist_limit = abs(twist_speed)/(BUDGET*dt*inverse_pair_inertia);
    float twist_torque = min(uintBitsToFloat(pc.meta.z)*fn*patch_radius,twist_limit);
    torque += cross(arm,ft)+rt-sign(twist_speed)*twist_torque*n;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
#if defined(GRAIN_CLEAR)
    // Frame start: all three bucket tables empty; history owners invalidated
    // when the history buffers were (re)allocated.
    if (i < 3u*pc.meta.y) bucket_counts[i] = 0u;
    if (pc.substep.y != 0u && i < 2u*pc.meta.x) history_owner[i] = EMPTY;
#elif defined(GRAIN_PERMUTE)
    // Frame start, before the clear, only when the grain order changed
    // (cell sort, births, removals elsewhere in the array). bucket_slots holds
    // new index -> previous index (EMPTY = new grain). Bank 0 holds the last
    // frame's final history (its bank stride is irrelevant: offset 0).
    if (i >= pc.meta.x) return;
    uint previous = bucket_slots[i];
    uint dst = slotBase(1u,i);
    if (previous == EMPTY) {
        history_owner[pc.meta.x+i] = EMPTY;
        return;
    }
    uint src = slotBase(0u,previous);
    for (uint s=0u; s<2u*SLOTS; ++s) history[dst+s] = history[src+s];
    history_owner[pc.meta.x+i] = history_owner[previous];
#elif defined(GRAIN_PERMUTE_COPY)
    // Second half: the gathered bank 1 becomes bank 0 at the new indices.
    if (i >= pc.meta.x) return;
    uint dst = slotBase(0u,i), src = slotBase(1u,i);
    for (uint s=0u; s<2u*SLOTS; ++s) history[dst+s] = history[src+s];
    history_owner[i] = history_owner[pc.meta.x+i];
#elif defined(GRAIN_HASH)
    if (i < pc.meta.x) insert(i, position(i,0u), 0u);
#elif defined(GRAIN_STEP)
    // The dispatch covers max(count, buckets): every invocation clears one
    // bucket of the table the NEXT substep writes, grains do the rest.
    uint table = pc.substep.x % 3u;
    if (i < pc.meta.y) bucket_counts[((pc.substep.x+2u)%3u)*pc.meta.y+i] = 0u;
    if (i >= pc.meta.x) return;
    // Proves the fused-step SPIR-V ran; the host refuses publication otherwise.
    if (i == 0u) diagnostics[1] = REVISION;
    uint bank = readBank();
    g_history_valid = history_owner[bank*pc.meta.x+i] == ids[i];
    vec3 p = position(i,bank), v = velocity(i,bank), w = omega(i,bank);
    float r = pc.low_radius.w, im = 1.0/masses[i], ii = 2.5*im/(r*r);
    vec3 f = vec3(0.0), t = vec3(0.0);
    bool xpbd = pc.wet.w > .5;
    float h = pc.step_contact.x;
    vec3 own_pred = p+h*(v+h*(pc.rolling.yzw+coupling[3u*i+1u].xyz));
    g_alpha = 1.0/(pc.high_stiffness.w*h*h);
    ivec3 own = cell(p);
    for (int z=-1; z<=1; ++z) for (int y=-1; y<=1; ++y) for (int x=-1; x<=1; ++x) {
        ivec3 wanted = own+ivec3(x,y,z);
        uint b = table*pc.meta.y+bucket(wanted);
        uint stored = min(bucket_counts[b],BUCKET);
        for (uint s=0u; s<stored; ++s) {
            uint j = bucket_slots[b*BUCKET+s];
            vec3 pj = position(j,bank);
            if (j != i && all(equal(cell(pj),wanted))) {
                vec3 separation = p-pj;
                float d = length(separation);
                if (d < 2.0*r+pc.wet.z && d > 1e-9) bridge(i,j,d,separation/d,r,f);
                if (xpbd) {
                    vec3 pred_j = predicted(j,bank);
                    vec3 gap = own_pred-pred_j;
                    float dp = length(gap);
                    if (dp < 2.0*r) {
                        vec3 n = dp > 1e-9 ? gap/dp : vec3(i<j ? -1.0 : 1.0,0,0);
                        vec3 motion = (own_pred-p)-(pred_j-pj)+
                            h*(cross(w,-n*r)-cross(omega(j,bank),n*r));
                        xpbdConstraint(n,2.0*r-dp,motion,im,1.0/masses[j],-n*r,ii);
                    }
                } else if (d < 2.0*r) {
                    vec3 n = d > 1e-9 ? separation/d : vec3(i<j ? -1.0 : 1.0,0,0);
                    vec3 arm = -n*(.5*d);
                    vec3 wj = omega(j,bank);
                    float jm = 1.0/masses[j], ji = 2.5*jm/(r*r);
                    vec3 relative = v+cross(w,arm)-velocity(j,bank)-cross(wj,-arm);
                    contact(i,ids[j],n,2.0*r-d,arm,relative,w-wj,ii,ji,.5*r,
                        im+jm+dot(arm,arm)*(ii+ji),f,t);
                }
            }
        }
    }
    // Closed domain walls. Static collider reaction belongs to the support.
    for (int axis=0; axis<3; ++axis) for (int side=0; side<2; ++side) {
        vec3 n = vec3(0.0); n[axis] = side == 0 ? 1.0 : -1.0;
        float distance = side == 0 ? p[axis]-pc.low_radius[axis] : pc.high_stiffness[axis]-p[axis];
        vec3 arm = -n*r;
        if (xpbd) {
            float predicted_distance = distance+dot(own_pred-p,n);
            xpbdConstraint(n,r-predicted_distance,(own_pred-p)+h*cross(w,arm),im,0.0,arm,ii);
        } else {
            contact(i,WALL_KEY|uint(2*axis+side),n,r-distance,arm,v+cross(w,arm),w,ii,0.0,r,
                im+r*r*ii,f,t);
        }
    }
    // Balanced BVH, up to four independent support features. Connected
    // coplanar triangles share one patch; a common edge/vertex is not doubled.
    vec3 manifold_n[4];
    float manifold_d[4];
    uint manifold_key[4];
    vec3 patch_n[8];
    float patch_d[8];
    uint patch_id[8];
    uint patch_count = 0u;
    uint manifold_count = 0u;
    uint stack[64];
    uint size = 0u;
    if (pc.meta.w > 0u) stack[size++] = 0u;
    while (size > 0u) {
        ColliderNode node = nodes[stack[--size]];
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
            vec3 q = closestTriangle(p,a,bp,c);
            float d = length(p-q);
            if (d >= r) continue;
            vec3 n = d > 1e-9 ? (p-q)/d : normalize(cross(bp-a,c-a));
            uint slot = patch_count;
            for (uint m=0u; m<patch_count; ++m) {
                if (patch_id[m] == patches[k]) { slot = m; break; }
            }
            if (slot == patch_count) {
                if (patch_count == 8u) {
                    atomicOr(diagnostics[0],2u);
                    continue;
                }
                ++patch_count;
                patch_n[slot] = n;
                patch_d[slot] = d;
                patch_id[slot] = patches[k];
            } else if (d < patch_d[slot]) {
                patch_n[slot] = n;
                patch_d[slot] = d;
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
            atomicOr(diagnostics[0],2u);
            continue;
        }
        manifold_n[manifold_count] = patch_n[k];
        manifold_d[manifold_count] = patch_d[k];
        manifold_key[manifold_count++] = PATCH_KEY|(patch_id[k] & 0x3fffffffu);
    }
    for (uint m=0u; m<manifold_count; ++m) {
        vec3 n = manifold_n[m], arm = -n*r;
        if (xpbd) {
            float predicted_distance = manifold_d[m]+dot(own_pred-p,n);
            xpbdConstraint(n,r-predicted_distance,(own_pred-p)+h*cross(w,arm),im,0.0,arm,ii);
        } else {
            contact(i,manifold_key[m],n,r-manifold_d[m],arm,v+cross(w,arm),w,ii,0.0,r,
                im+r*r*ii,f,t);
        }
    }
    // The contact budget sizes DEM spring history; XPBD keeps none.
    if (!xpbd && g_contacts > SLOTS) atomicOr(diagnostics[0],1u);
    uint dst = writeBank();
    if (g_written < SLOTS) history[slotBase(dst,i)+2u*g_written] = uvec4(EMPTY);
    history_owner[dst*pc.meta.x+i] = ids[i];
    atomicMax(diagnostics[2],g_contacts);
    if (pc.substep.x == pc.substep.w) {
        atomicAdd(diagnostics[3],g_sticking);
        atomicAdd(diagnostics[4],g_contacts);
        atomicAdd(diagnostics[5],g_bridges);
    }
    // Symplectic Euler from the complete previous-substep state.
    float dt = pc.step_contact.x;
    vec4 drag = coupling[3u*i];
    vec4 lift = coupling[3u*i+1u];
    if (xpbd) {
        // Jacobi averaging with over-relaxation 1.5 (Macklin 2014).
        float relax = g_constraints > 1u ? min(1.0,1.5/float(g_constraints)) : 1.0;
        // Bridges act as external forces on the prediction.
        vec3 x_new = own_pred+dt*dt*f*im+relax*g_dx;
        v = (x_new-p)/dt;
        w += relax*g_dtheta/dt;
        float spin = length(w);
        float cap = pc.rolling.x*g_normal_sum*r*ii/dt;
        if (spin > 0.0 && cap > 0.0) w *= max(0.0,1.0-cap/spin);
        f = vec3(0.0);
        t = vec3(0.0);
        p = x_new-dt*v;  // the shared update below adds dt*v back
    } else {
        v += dt*(f*im+pc.rolling.yzw+lift.xyz);
    }
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
    // Next substep's table; other invocations still read `table`.
    insert(i,p,(pc.substep.x+1u)%3u);
#endif
}
