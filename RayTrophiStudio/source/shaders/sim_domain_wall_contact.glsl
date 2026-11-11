// CPU twin: Fluid/DomainWallContact.h. Stationary, axis-aligned domain planes.
vec3 domainWallVelocity(vec3 velocity, int axis, float sign, float friction,
    float restitution, vec3 acceleration, float dt, float glue) {
    float vn = sign * velocity[axis];
    if (vn > 1e-6 && glue <= 0.0) {
        return velocity;
    }
    float incoming = max(-vn, 0.0);
    float support = max(-sign * acceleration[axis], 0.0) * dt;
    float rebound = restitution * max(incoming - support, 0.0);
    float normal_change = incoming + rebound;
    float cohesion = max(glue - max(vn, 0.0), 0.0);
    float load = vn > 1e-6 ? 0.0 : max(normal_change, support);
    vec3 tangent = velocity;
    tangent[axis] = 0.0;
    float speed = length(tangent);
    if (speed > 1e-9) {
        tangent *= max(speed - friction * load - cohesion, 0.0) / speed;
    }
    float separating = vn > 0.0 ? vn : rebound;
    tangent[axis] = sign * max(separating - glue, 0.0);
    return tangent;
}

void domainWallParticle(inout vec3 position, inout vec3 velocity, vec3 low,
    vec3 high, float friction, float restitution, vec3 acceleration, float dt,
    float glue, float h) {
    float band = glue > 0.0 ? h : 1e-6;
    vec3 before = velocity;
    for (int axis = 0; axis < 3; ++axis) {
        if (position[axis] <= low[axis] + band) {
            velocity = domainWallVelocity(velocity, axis, 1.0, friction,
                restitution, acceleration, dt, glue);
        } else if (position[axis] >= high[axis] - band) {
            velocity = domainWallVelocity(velocity, axis, -1.0, friction,
                restitution, acceleration, dt, glue);
        }
        position[axis] = clamp(position[axis], low[axis], high[axis]);
    }
    if (glue > 0.0) {
        position = clamp(position + (velocity - before) * dt, low, high);
    }
}
