// ═══════════════════════════════════════════════════════════════════════════════
// EMITTER SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

nlohmann::json Emitter::toJson() const {
    nlohmann::json j;
    j["shape"] = static_cast<int>(shape);
    j["position"] = {position.x, position.y, position.z};
    j["size"] = {size.x, size.y, size.z};
    j["radius"] = radius;
    j["height"] = height;
    j["inner_radius"] = inner_radius;
    j["cone_angle"] = cone_angle;
    j["density_rate"] = density_rate;
    j["fuel_rate"] = fuel_rate;
    j["temperature"] = temperature;
    j["velocity"] = {velocity.x, velocity.y, velocity.z};
    j["fuel_phase"] = static_cast<int>(fuel_phase);
    j["phase_change_temperature"] = phase_change_temperature;
    j["fuel_release_rate"] = fuel_release_rate;
    j["flame_contact_sensitivity"] = flame_contact_sensitivity;
    j["falloff_type"] = static_cast<int>(falloff_type);
    j["falloff_start"] = falloff_start;
    j["falloff_end"] = falloff_end;
    j["noise_enabled"] = noise_enabled;
    j["noise_frequency"] = noise_frequency;
    j["noise_amplitude"] = noise_amplitude;
    j["noise_speed"] = noise_speed;
    j["noise_seed"] = noise_seed;
    j["noise_modulate_density"] = noise_modulate_density;
    j["noise_modulate_temperature"] = noise_modulate_temperature;
    j["noise_modulate_velocity"] = noise_modulate_velocity;
    j["spray_cone_angle"] = spray_cone_angle;
    j["speed_min"] = speed_min;
    j["speed_max"] = speed_max;
    j["emission_mode"] = static_cast<int>(emission_mode);
    j["start_frame"] = start_frame;
    j["end_frame"] = end_frame;
    j["pulse_interval"] = pulse_interval;
    j["pulse_duration"] = pulse_duration;
    j["name"] = name;
    j["uid"] = uid;
    
    nlohmann::json kfs_j = nlohmann::json::object();
    for (auto const& [frame, kf] : keyframes) {
        kfs_j[std::to_string(frame)] = kf;
    }
    j["keyframes"] = kfs_j;
    
    return j;
}

void Emitter::fromJson(const nlohmann::json& j) {
    if (j.contains("shape")) shape = static_cast<EmitterShape>(j["shape"].get<int>());
    if (j.contains("position")) {
        auto p = j["position"];
        position = Vec3(p[0], p[1], p[2]);
    }
    if (j.contains("size")) {
        auto s = j["size"];
        size = Vec3(s[0], s[1], s[2]);
    }
    if (j.contains("radius")) radius = j["radius"];
    if (j.contains("height")) height = j["height"];
    if (j.contains("inner_radius")) inner_radius = j["inner_radius"];
    if (j.contains("cone_angle")) cone_angle = j["cone_angle"];
    if (j.contains("density_rate")) density_rate = j["density_rate"];
    if (j.contains("fuel_rate")) fuel_rate = j["fuel_rate"];
    if (j.contains("temperature")) temperature = j["temperature"];
    if (j.contains("fuel_phase")) fuel_phase = static_cast<FuelPhase>(j["fuel_phase"].get<int>());
    if (j.contains("phase_change_temperature")) phase_change_temperature = j["phase_change_temperature"];
    if (j.contains("fuel_release_rate")) fuel_release_rate = j["fuel_release_rate"];
    if (j.contains("flame_contact_sensitivity")) flame_contact_sensitivity = j["flame_contact_sensitivity"];
    if (j.contains("velocity")) {
        auto v = j["velocity"];
        velocity = Vec3(v[0], v[1], v[2]);
    }
    if (j.contains("falloff_type")) falloff_type = static_cast<EmitterFalloffType>(j["falloff_type"].get<int>());
    if (j.contains("falloff_start")) falloff_start = j["falloff_start"];
    if (j.contains("falloff_end")) falloff_end = j["falloff_end"];
    if (j.contains("noise_enabled")) noise_enabled = j["noise_enabled"];
    if (j.contains("noise_frequency")) noise_frequency = j["noise_frequency"];
    if (j.contains("noise_amplitude")) noise_amplitude = j["noise_amplitude"];
    if (j.contains("noise_speed")) noise_speed = j["noise_speed"];
    if (j.contains("noise_seed")) noise_seed = j["noise_seed"];
    if (j.contains("noise_modulate_density")) noise_modulate_density = j["noise_modulate_density"];
    if (j.contains("noise_modulate_temperature")) noise_modulate_temperature = j["noise_modulate_temperature"];
    if (j.contains("noise_modulate_velocity")) noise_modulate_velocity = j["noise_modulate_velocity"];
    if (j.contains("spray_cone_angle")) spray_cone_angle = j["spray_cone_angle"];
    if (j.contains("speed_min")) speed_min = j["speed_min"];
    if (j.contains("speed_max")) speed_max = j["speed_max"];
    if (j.contains("emission_mode")) emission_mode = static_cast<EmitterEmissionMode>(j["emission_mode"].get<int>());
    if (j.contains("start_frame")) start_frame = j["start_frame"];
    if (j.contains("end_frame")) end_frame = j["end_frame"];
    if (j.contains("pulse_interval")) pulse_interval = j["pulse_interval"];
    if (j.contains("pulse_duration")) pulse_duration = j["pulse_duration"];
    if (j.contains("name")) name = j["name"];
    if (j.contains("uid")) uid = j["uid"];
    
    if (j.contains("keyframes")) {
        keyframes.clear();
        for (auto it = j["keyframes"].begin(); it != j["keyframes"].end(); ++it) {
            int frame = std::stoi(it.key());
            keyframes[frame] = it.value().get<::EmitterKeyframe>();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EMITTER KEYFRAME METHODS
// ═══════════════════════════════════════════════════════════════════════════════

::EmitterKeyframe Emitter::getInterpolatedKeyframe(float current_frame) const {
    if (keyframes.empty()) {
        ::EmitterKeyframe kf;
        kf.fuel_rate = fuel_rate; kf.has_fuel_rate = true;
        kf.density_rate = density_rate; kf.has_density_rate = true;
        kf.temperature = temperature; kf.has_temperature = true;
        kf.velocity = velocity; kf.has_velocity = true;
        kf.position = position; kf.has_position = true;
        kf.size = size; kf.has_size = true;
        kf.radius = radius; kf.has_radius = true;
        kf.enabled = enabled; kf.has_enabled = true;
        return kf;
    }
    
    auto it_after = keyframes.lower_bound((int)std::ceil(current_frame));
    
    if (it_after == keyframes.begin()) {
        return it_after->second;
    }
    
    if (it_after == keyframes.end()) {
        return keyframes.rbegin()->second;
    }
    
    auto it_before = std::prev(it_after);
    
    int frame_before = it_before->first;
    int frame_after = it_after->first;
    
    float t = (current_frame - frame_before) / (float)(frame_after - frame_before);
    t = std::max(0.0f, std::min(1.0f, t));
    
    return ::EmitterKeyframe::lerp(it_before->second, it_after->second, t);
}

void Emitter::applyKeyframe(const ::EmitterKeyframe& kf) {
    if (kf.has_fuel_rate) fuel_rate = kf.fuel_rate;
    if (kf.has_density_rate) density_rate = kf.density_rate;
    if (kf.has_temperature) temperature = kf.temperature;
    if (kf.has_velocity) velocity = kf.velocity;
    if (kf.has_position) position = kf.position;
    if (kf.has_size) size = kf.size;
    if (kf.has_radius) radius = kf.radius;
    if (kf.has_enabled) enabled = kf.enabled;
}


// ═══════════════════════════════════════════════════════════════════════════════
// SETTINGS SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

nlohmann::json GasSimulationSettings::toJson() const {
    nlohmann::json j;
    j["resolution"] = {resolution_x, resolution_y, resolution_z};
    j["grid_size"] = {grid_size.x, grid_size.y, grid_size.z};
    j["voxel_size"] = voxel_size;
    j["preserve_voxel_size_on_resize"] = preserve_voxel_size_on_resize;
    j["max_auto_resolution"] = max_auto_resolution;
    j["grid_offset"] = {grid_offset.x, grid_offset.y, grid_offset.z};
    j["timestep"] = timestep;
    j["substeps"] = substeps;
    j["time_scale"] = time_scale;
    j["pressure_iterations"] = pressure_iterations;
    
    // CFL Adaptive Timestep
    j["adaptive_timestep"] = adaptive_timestep;
    j["cfl_number"] = cfl_number;
    j["min_timestep"] = min_timestep;
    j["max_timestep"] = max_timestep;
    
    j["density_dissipation"] = density_dissipation;
    j["velocity_dissipation"] = velocity_dissipation;
    j["temperature_dissipation"] = temperature_dissipation;
    j["fuel_dissipation"] = fuel_dissipation;
    
    j["ignition_temperature"] = ignition_temperature;
    j["burn_rate"] = burn_rate;
    j["heat_release"] = heat_release;
    j["expansion_strength"] = expansion_strength;
    j["smoke_generation"] = smoke_generation;
    j["soot_generation"] = soot_generation;
    j["buoyancy_density"] = buoyancy_density;
    j["buoyancy_temperature"] = buoyancy_temperature;
    j["ambient_temperature"] = ambient_temperature;
    j["vorticity_strength"] = vorticity_strength;
    j["turbulence_strength"] = turbulence_strength;
    j["turbulence_scale"] = turbulence_scale;
    j["turbulence_octaves"] = turbulence_octaves;
    j["turbulence_lacunarity"] = turbulence_lacunarity;
    j["turbulence_persistence"] = turbulence_persistence;
    j["advection_mode"] = static_cast<int>(advection_mode);
    j["pressure_solver"] = static_cast<int>(pressure_solver);
    j["boundary_mode"] = static_cast<int>(boundary_mode);
    j["sor_omega"] = sor_omega;
    j["gravity"] = {gravity.x, gravity.y, gravity.z};
    j["wind"] = {wind.x, wind.y, wind.z};
    j["sparse_mode"] = sparse_mode;
    j["sparse_threshold"] = sparse_threshold;
    j["max_velocity"] = max_velocity;
    j["max_temperature"] = max_temperature;
    j["max_density"] = max_density;
    j["mode"] = static_cast<int>(mode);
    j["backend"] = static_cast<int>(backend);
    j["cache_directory"] = cache_directory;
    j["bake_start_frame"] = bake_start_frame;
    j["bake_end_frame"] = bake_end_frame;
    j["bake_fps"] = bake_fps;
    return j;
}

void GasSimulationSettings::fromJson(const nlohmann::json& j) {
    if (j.contains("resolution")) {
        auto r = j["resolution"];
        resolution_x = r[0]; resolution_y = r[1]; resolution_z = r[2];
    }
    if (j.contains("grid_size")) {
        auto s = j["grid_size"];
        grid_size = Vec3(s[0], s[1], s[2]);
    }
    if (j.contains("voxel_size")) voxel_size = j["voxel_size"];
    if (j.contains("preserve_voxel_size_on_resize")) preserve_voxel_size_on_resize = j["preserve_voxel_size_on_resize"];
    if (j.contains("max_auto_resolution")) max_auto_resolution = j["max_auto_resolution"];
    if (j.contains("grid_offset")) {
        auto o = j["grid_offset"];
        grid_offset = Vec3(o[0], o[1], o[2]);
    }
    if (j.contains("timestep")) timestep = j["timestep"];
    if (j.contains("substeps")) substeps = j["substeps"];
    if (j.contains("time_scale")) time_scale = j["time_scale"];
    if (j.contains("pressure_iterations")) pressure_iterations = j["pressure_iterations"];
    
    // CFL Adaptive Timestep
    if (j.contains("adaptive_timestep")) adaptive_timestep = j["adaptive_timestep"];
    if (j.contains("cfl_number")) cfl_number = j["cfl_number"];
    if (j.contains("min_timestep")) min_timestep = j["min_timestep"];
    if (j.contains("max_timestep")) max_timestep = j["max_timestep"];
    
    if (j.contains("density_dissipation")) density_dissipation = j["density_dissipation"];
    if (j.contains("velocity_dissipation")) velocity_dissipation = j["velocity_dissipation"];
    if (j.contains("temperature_dissipation")) temperature_dissipation = j["temperature_dissipation"];
    if (j.contains("fuel_dissipation")) fuel_dissipation = j["fuel_dissipation"];

    if (j.contains("ignition_temperature")) ignition_temperature = j["ignition_temperature"];
    if (j.contains("burn_rate")) burn_rate = j["burn_rate"];
    if (j.contains("heat_release")) heat_release = j["heat_release"];
    if (j.contains("expansion_strength")) expansion_strength = j["expansion_strength"];
    if (j.contains("smoke_generation")) smoke_generation = j["smoke_generation"];
    if (j.contains("soot_generation")) soot_generation = j["soot_generation"];
    if (j.contains("buoyancy_density")) buoyancy_density = j["buoyancy_density"];
    if (j.contains("buoyancy_temperature")) buoyancy_temperature = j["buoyancy_temperature"];
    if (j.contains("ambient_temperature")) ambient_temperature = j["ambient_temperature"];
    if (j.contains("vorticity_strength")) vorticity_strength = j["vorticity_strength"];
    if (j.contains("turbulence_strength")) turbulence_strength = j["turbulence_strength"];
    if (j.contains("turbulence_scale")) turbulence_scale = j["turbulence_scale"];
    if (j.contains("turbulence_octaves")) turbulence_octaves = j["turbulence_octaves"];
    if (j.contains("turbulence_lacunarity")) turbulence_lacunarity = j["turbulence_lacunarity"];
    if (j.contains("turbulence_persistence")) turbulence_persistence = j["turbulence_persistence"];
    if (j.contains("advection_mode")) advection_mode = static_cast<AdvectionMode>(j["advection_mode"].get<int>());
    if (j.contains("pressure_solver")) pressure_solver = static_cast<PressureSolverMode>(j["pressure_solver"].get<int>());
    if (j.contains("boundary_mode")) boundary_mode = static_cast<BoundaryMode>(j["boundary_mode"].get<int>());
    if (j.contains("sor_omega")) sor_omega = j["sor_omega"];
    if (j.contains("gravity")) {
        auto g = j["gravity"];
        gravity = Vec3(g[0], g[1], g[2]);
    }
    if (j.contains("wind")) {
        auto w = j["wind"];
        wind = Vec3(w[0], w[1], w[2]);
    }
    if (j.contains("sparse_mode")) sparse_mode = j["sparse_mode"];
    if (j.contains("sparse_threshold")) sparse_threshold = j["sparse_threshold"];
    if (j.contains("max_velocity")) max_velocity = j["max_velocity"];
    if (j.contains("max_temperature")) max_temperature = j["max_temperature"];
    if (j.contains("max_density")) max_density = j["max_density"];
    if (j.contains("mode")) mode = static_cast<SimulationMode>(j["mode"].get<int>());
    if (j.contains("backend")) backend = static_cast<SolverBackend>(j["backend"].get<int>());
    if (j.contains("cache_directory")) cache_directory = j["cache_directory"];
    if (j.contains("bake_start_frame")) bake_start_frame = j["bake_start_frame"];
    if (j.contains("bake_end_frame")) bake_end_frame = j["bake_end_frame"];
    if (j.contains("bake_fps")) bake_fps = j["bake_fps"];
}
