
void InstanceGroup::updateWind(float time) {
    if (!wind_settings.enabled) return;

    // Wind parameters
    float t = time * wind_settings.speed;
    // Normalize direction
    Vec3 dir = wind_settings.direction;
    float len = dir.length();
    if (len > 0.001f) dir = dir / len;
    else return;

    float strength = wind_settings.strength;
    float turbulence = wind_settings.turbulence;
    float wave_size = wind_settings.wave_size;
    if (wave_size < 0.1f) wave_size = 0.1f;

    // Rotate around axis perpendicular to wind and UP
    Vec3 up(0, 1, 0);
    Vec3 axis = dir.cross(up);
    float axis_len = axis.length();
    if (axis_len < 0.001f) {
        // Wind is vertical? Use X axis fallback
        axis = Vec3(1, 0, 0);
    } else {
        axis = axis / axis_len;
    }

    // Precompute axis-angle rotation math constants that don't change per instance
    // Actually angle changes per instance based on noise.
    float ax = axis.x, ay = axis.y, az = axis.z;

    // Iterate
    size_t count = std::min(instances.size(), active_hittables.size());
    if (initial_instances.size() < count) count = initial_instances.size(); // Safety

    for (size_t i = 0; i < count; ++i) {
        // Check if weak pointer is valid
        if (active_hittables[i].expired()) continue;
        
        auto hittable_ptr = active_hittables[i].lock();
        if (!hittable_ptr) continue;

        auto inst = std::dynamic_pointer_cast<HittableInstance>(hittable_ptr);
        if (!inst) continue;

        const auto& initial = initial_instances[i];
        
        // Compute Sway Phase based on position
        float phase = (initial.position.x * dir.x + initial.position.z * dir.z) * (1.0f / wave_size);
        
        // Multi-frequency noise for natural movement
        float noise = sinf(t + phase) + sinf(t * turbulence + phase * 2.5f) * 0.5f;
        
        // Calculate deflection angle (degrees)
        float angle = noise * strength;

        // --- Build Rotation Matrix (Axis-Angle) ---
        float rad = angle * 3.14159f / 180.0f;
        float c = cosf(rad);
        float s = sinf(rad);
        float t_val = 1.0f - c;

        Matrix4x4 swayMat;
        swayMat.m[0][0] = t_val*ax*ax + c;    swayMat.m[0][1] = t_val*ax*ay - az*s; swayMat.m[0][2] = t_val*ax*az + ay*s; swayMat.m[0][3] = 0;
        swayMat.m[1][0] = t_val*ax*ay + az*s; swayMat.m[1][1] = t_val*ay*ay + c;    swayMat.m[1][2] = t_val*ay*az - ax*s; swayMat.m[1][3] = 0;
        swayMat.m[2][0] = t_val*ax*az - ay*s; swayMat.m[2][1] = t_val*ay*az + ax*s; swayMat.m[2][2] = t_val*az*az + c;    swayMat.m[2][3] = 0;
        swayMat.m[3][0] = 0;                  swayMat.m[3][1] = 0;                  swayMat.m[3][2] = 0;                  swayMat.m[3][3] = 1;

        // --- Apply Transform ---
        // Final Matrix = Translate(Pos) * Sway * Rotate(InitialRot) * Scale(InitialScale)
        // OR: Final = Translate(Pos) * Rotate(InitialRot) * Sway * Scale(InitialScale) ?
        // Foliage "bend" is usually applied on the object's local frame before its own rotation if we want it to bend relative to wind world space?
        // Actually, "Wind Bending" is usually a shear in World Space relative to the object pivot.
        
        // Approach A: World Space Sway around Pivot
        // 1. Get Initial Local Matrix (Scaling * Rotation)
        // 2. Apply Sway (Rotation around Pivot)
        // 3. Translate to Position
        
        // Reconstruct base matrix components
        Matrix4x4 baseMat = initial.toMatrix();
        // Remove translation to get Orientation * Scale
        Matrix4x4 localMat = baseMat;
        localMat.m[0][3] = 0; localMat.m[1][3] = 0; localMat.m[2][3] = 0;

        // Final = Translate * Sway * Local
        Matrix4x4 T; 
        T.m[0][3] = initial.position.x; 
        T.m[1][3] = initial.position.y; 
        T.m[2][3] = initial.position.z;

        // Multiply: T * Sway * Local
        // T * Sway
        Matrix4x4 TSway = T * swayMat; 
        // Result * Local
        Matrix4x4 finalMat = TSway * localMat;

        // Update Instance
        inst->setTransform(finalMat);
    }
}
