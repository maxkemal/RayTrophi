#include "sim_domain_wall_contact.glsl"

float wallSnapshot(int component, ivec3 f) {
    ivec3 last = ivec3(pc.nx - 1, pc.ny - 1, pc.nz - 1) + unitAxis(component);
    return snapFace(component, clamp(f, ivec3(0), last));
}

// CPU twin: resolveDomainWallFace. Snapshot-only gathers prevent races.
float domainWallFace(float value, int component, ivec3 f) {
    if ((pc.flags & 3u) != 3u) {
        return value;
    }
    ivec3 size = ivec3(pc.nx, pc.ny, pc.nz);
    for (int axis = 0; axis < 3; ++axis) {
        int last = size[axis] - int(axis != component);
        if (f[axis] != 0 && f[axis] != last) {
            continue;
        }
        float sign = f[axis] == 0 ? 1.0 : -1.0;
        vec3 velocity;
        for (int a = 0; a < 3; ++a) {
            float speed = value;
            if (a != component) {
                ivec3 q = f - unitAxis(component);
                speed = wallSnapshot(a, q);
                q[a] += 1;
                speed += wallSnapshot(a, q);
                q[component] += 1;
                speed += wallSnapshot(a, q);
                q[a] -= 1;
                speed = (speed + wallSnapshot(a, q)) * 0.25;
            }
            velocity[a] = speed;
        }
        if (axis != component) {
            ivec3 q = f;
            q[axis] = sign > 0.0 ? 0 : size[axis];
            float normal = wallSnapshot(axis, q);
            q[component] -= 1;
            velocity[axis] = 0.5 * (normal + wallSnapshot(axis, q));
        }
        // MPM stress already supplies the material's elastic response.
        // Do not rebound its normal support impulses or restore the normal
        // velocity that plainContact/phiContact has just constrained.
        value = domainWallVelocity(velocity, axis, sign, pc.friction,
            0.0, vec3(pc.gravity_x, pc.gravity_y, pc.gravity_z), pc.dt,
            max(pc.adhesion_pa, 0.0) * pc.dt / max(pc.density * pc.h, 1e-6))[component];
    }
    return value;
}
